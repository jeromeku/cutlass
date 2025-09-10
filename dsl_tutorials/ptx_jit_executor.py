import os
import io
import ctypes
import inspect
from cutlass._mlir import ir

# Local modules imports
from cutlass.base_dsl import typing as t
from cutlass.base_dsl.common import DSLRuntimeError
from cutlass.base_dsl.runtime import cuda as cuda_helpers
from cutlass.base_dsl.runtime.jit_arg_adapters import JitArgAdapterRegistry, is_arg_spec_constexpr
from cutlass.base_dsl.typing import get_c_pointers
from cutlass.base_dsl.utils.logger import log
from cutlass.base_dsl.utils.timer import timer
from cutlass.base_dsl.jit_executor import CudaModules, CudaSingleModule
import io, os, re, ctypes
from typing import List, Dict, Optional

# ---- robust unescaper for MLIR-printed strings (handles \xx hex and \\) ----
def _unescape_mlir_quoted_bytes(s: str) -> bytes:
    """
    Convert MLIR-escaped string payload into raw bytes.
    Handles \\xx hex escapes and escaped backslashes/quotes commonly used by MLIR.
    """
    b = s.encode("utf-8", "replace")
    out = bytearray()
    i = 0
    def ishex(c: int) -> bool:
        return 0x30 <= c <= 0x39 or 0x41 <= c <= 0x46 or 0x61 <= c <= 0x66  # 0-9 A-F a-f
    while i < len(b):
        if b[i] == 0x5C:  # backslash
            # \xx (hex)
            if i + 2 < len(b) and ishex(b[i+1]) and ishex(b[i+2]):
                out += bytearray.fromhex(bytes(b[i+1:i+3]).decode("ascii"))
                i += 3
                continue
            # \\ or \" -> keep the next char literally (common in MLIR prints)
            if i + 1 < len(b):
                out.append(b[i+1])
                i += 2
                continue
            i += 1
        else:
            out.append(b[i]); i += 1
    return bytes(out)

_PTXX = re.compile(rb'(?m)^\s*(?:\.visible\s+)?\.entry\s+([A-Za-z0-9_.$@-]+)')
def _looks_like_ptx(buf: bytes) -> bool:
    head = buf[:128].lstrip()
    return (head.startswith(b".version") or head.startswith(b".target")) and b".entry" in buf

def _guess_entries(ptx: bytes) -> List[str]:
    return [m.group(1).decode("utf-8", "replace") for m in _PTXX.finditer(ptx)]

_ASM_RE = re.compile(r'assembly\s*=\s*"((?:\\.|[^"])*)"', re.S)

# def module_to_text(ir_module) -> str:
#     # Get textual assembly for the whole module. Use keyword args to avoid the binding trap.
#     s = io.StringIO()
#     try:
#         ir_module.operation.print(file=s, binary=False, print_generic_op_form=False)
#         return s.getvalue()
#     except Exception:
#         # Fallback – stringify; good enough for post-processing
#         return str(ir_module.operation)

# --- 1) stringify the module once ---
def module_to_text(ir_module) -> str:
    s = io.StringIO()
    try:
        # IMPORTANT: keyword args so the binding treats 's' as file, not AsmState
        ir_module.operation.print(file=s, binary=False, print_generic_op_form=False)
        return s.getvalue()
    except Exception:
        return str(ir_module.operation)

# --- 2) robust unescaper for MLIR-quoted strings ---
def unescape_mlir_quoted_bytes(escaped: str) -> bytes:
    b = escaped.encode("utf-8", "replace")
    out = bytearray()
    i, n = 0, len(b)

    def ishex(x: int) -> bool:
        return (48 <= x <= 57) or (65 <= x <= 70) or (97 <= x <= 102)  # 0-9 A-F a-f

    while i < n:
        c = b[i]
        if c == 0x5C:  # backslash
            # \xx hex (exactly two)
            if i + 2 < n and ishex(b[i+1]) and ishex(b[i+2]):
                out += bytearray.fromhex(bytes(b[i+1:i+3]).decode("ascii"))
                i += 3
                continue
            # common escapes: \\ \" -> take next char literally
            if i + 1 < n:
                out.append(b[i+1])
                i += 2
                continue
            i += 1
        else:
            out.append(c); i += 1
    return bytes(out)

# --- 3) tiny state machine to collect all assembly="...": returns list[str] of escaped payloads ---
def iter_assembly_strings(mod_text: str):
    key = 'assembly'
    i, n = 0, len(mod_text)
    while i < n:
        j = mod_text.find(key, i)
        if j < 0: break
        k = j + len(key)
        # skip spaces and "= "
        while k < n and mod_text[k].isspace(): k += 1
        if k >= n or mod_text[k] != '=':
            i = k; continue
        k += 1
        while k < n and mod_text[k].isspace(): k += 1
        if k >= n or mod_text[k] != '"':
            i = k; continue
        # we are at opening quote
        k += 1
        start = k
        esc = False
        while k < n:
            ch = mod_text[k]
            if esc:
                esc = False
                k += 1
                continue
            if ch == '\\':
                esc = True
                k += 1
                continue
            if ch == '"':
                # closing quote
                yield mod_text[start:k]
                k += 1
                break
            k += 1
        i = k

# --- 4) helpers for PTX sanity ---
_PTXX = re.compile(rb'(?m)^\s*(?:\.visible\s+)?\.entry\s+([A-Za-z0-9_.$@-]+)')
def looks_like_ptx(buf: bytes) -> bool:
    head = buf[:128].lstrip()
    return (head.startswith(b".version") or head.startswith(b"//") or head.startswith(b".target"))

def guess_entries(ptx: bytes):
    return [m.group(1).decode("utf-8", "replace") for m in _PTXX.finditer(ptx)]

# --- 5) one-shot API: get PTX blobs from module text ---
def extract_all_ptx_from_text(mod_text: str):
    blobs = []
    for esc in iter_assembly_strings(mod_text):
        raw = unescape_mlir_quoted_bytes(esc)
        if looks_like_ptx(raw):
            blobs.append({"ptx": raw, "entries": guess_entries(raw)})
    return blobs

class JitExecutor:
    def __init__(
        self,
        dsl,
        engine,
        capi_func,
        ir_module,
        args_spec,
        function_name,
        cuda_modules: CudaModules = None,
        jit_time_profiling=False,
    ):
        self.dsl = dsl
        self.engine = engine
        self.capi_func = capi_func
        self.ir_module = ir_module
        self.args_spec = args_spec
        self.function_name = function_name
        if args_spec is not None:
            self.original_args_spec = args_spec
            self.args_spec = self.filter_runtime_arg_spec(args_spec)
        # cuda kernels
        self.cuda_modules = cuda_modules
        self.jit_time_profiling = jit_time_profiling

    def filter_runtime_arg_spec(self, arg_spec: inspect.FullArgSpec):
        runtime_args = []
        runtime_annotations = {}
        runtime_defaults = []

        if arg_spec.defaults:
            defaults_start_idx = len(arg_spec.args) - len(arg_spec.defaults)
        else:
            defaults_start_idx = len(arg_spec.args)

        for i, arg_name in enumerate(arg_spec.args):
            arg_type = arg_spec.annotations.get(arg_name, None)
            if is_arg_spec_constexpr(arg_type, arg_name, i, self.function_name):
                continue
            runtime_args.append(arg_name)
            if arg_name in arg_spec.annotations:
                runtime_annotations[arg_name] = arg_type
            if i >= defaults_start_idx:
                default_idx = i - defaults_start_idx
                runtime_defaults.append(arg_spec.defaults[default_idx])

        runtime_kwonlyargs = []
        runtime_kwonlydefaults = {}

        if arg_spec.kwonlyargs:
            for kwarg in arg_spec.kwonlyargs:
                arg_type = arg_spec.annotations.get(kwarg, None)
                if is_arg_spec_constexpr(arg_type, kwarg, i, self.function_name):
                    continue
                runtime_kwonlyargs.append(kwarg)
                if kwarg in arg_spec.annotations:
                    runtime_annotations[kwarg] = arg_type
                if arg_spec.kwonlydefaults and kwarg in arg_spec.kwonlydefaults:
                    runtime_kwonlydefaults[kwarg] = arg_spec.kwonlydefaults[kwarg]

        runtime_defaults = tuple(runtime_defaults) if runtime_defaults else None

        return inspect.FullArgSpec(
            args=runtime_args,
            varargs=arg_spec.varargs,
            varkw=arg_spec.varkw,
            defaults=runtime_defaults,
            kwonlyargs=runtime_kwonlyargs,
            kwonlydefaults=runtime_kwonlydefaults if runtime_kwonlydefaults else None,
            annotations=runtime_annotations,
        )

    def __del__(self):
        if self.cuda_modules:
            cuda_modules = [module.cuda_module for module in self.cuda_modules.modules]
            for module in set(cuda_modules):
                cuda_helpers.unload_cubin_module(module)

    def get_constexpr_args(self) -> list[dict[str, int | str]]:
        if self.original_args_spec is None:
            return list()
        constexpr_args = list()
        for i, arg_name in enumerate(self.original_args_spec.args):
            if arg_name not in self.args_spec.args:
                constexpr_args.append({"argument_index": i, "argument_name": arg_name})

        if self.original_args_spec.kwonlyargs:
            for kwarg in self.original_args_spec.kwonlyargs:
                if kwarg not in self.args_spec.kwonlyargs:
                    constexpr_args.append(
                        {"argument_index": None, "argument_name": kwarg}
                    )
        return constexpr_args

    def generate_execution_args(self, args, kwargs, args_spec: inspect.FullArgSpec):
        rectified_args = list(args)
        if args_spec.defaults and len(args) < len(args_spec.args):
            rectified_args.extend(args_spec.defaults[len(args) - len(args_spec.args) :])
        for k, v in kwargs.items():
            if k in args_spec.args:
                idx = args_spec.args.index(k)
                if idx < len(rectified_args):
                    rectified_args[idx] = v
                else:
                    rectified_args.append(v)

        rectified_kwargs = {k: v for k, v in kwargs.items() if k not in args_spec.args}
        if args_spec.kwonlydefaults and len(rectified_kwargs) < len(
            args_spec.kwonlyargs
        ):
            rectified_kwargs.update(args_spec.kwonlydefaults)

        if len(rectified_args) != len(args_spec.args) or len(rectified_kwargs) != len(
            args_spec.kwonlyargs
        ):
            raise DSLRuntimeError(
                "input args/kwargs length does not match runtime function signature!",
                context={
                    "input args length": len(rectified_args),
                    "input kwargs length": len(rectified_kwargs),
                    "function signature args length": len(args_spec.args),
                    "function signature kwonlyargs length": len(args_spec.kwonlyargs),
                },
            )

        exe_args = []
        adapted_args = []
        input_args = rectified_args + list(rectified_kwargs.values())
        input_arg_names = args_spec.args + args_spec.kwonlyargs
        for arg, arg_name in zip(input_args, input_arg_names):
            if hasattr(arg, "__c_pointers__"):
                exe_args.extend(arg.__c_pointers__())
                continue

            arg_type = args_spec.annotations.get(arg_name, None)

            if isinstance(arg_type, t.NumericMeta):
                arg = t.cast(arg, arg_type)
            else:
                adapter = JitArgAdapterRegistry.get_registered_adapter(type(arg))
                if adapter:
                    arg = adapter(arg)
                    adapted_args.append(arg)

            exe_args.extend(get_c_pointers(arg))

        return exe_args, adapted_args

    def __call__(self, *args, **kwargs):
        exe_args, adapted_args = self.generate_execution_args(
            args, kwargs, self.args_spec
        )
        self.run_compiled_program(exe_args)

    def get_invoke_packed_args(self, exe_args):
        if self.cuda_modules:
            exe_args += self.cuda_modules.args
        packed_args = (ctypes.c_void_p * len(exe_args))()
        for argNum in range(len(exe_args)):
            packed_args[argNum] = exe_args[argNum]
        return packed_args

    def run_compiled_program(self, exe_args):
        if self.jit_time_profiling:
            profiler = timer(enable=True)
            try:
                packed_args = profiler(self.get_invoke_packed_args)(exe_args)
                profiler(self.capi_func)(packed_args)
            except Exception as e:
                raise DSLRuntimeError(f"💥💥💥 Runtime Crash 💥💥💥", cause=e)
        else:
            try:
                packed_args = self.get_invoke_packed_args(exe_args)
                self.capi_func(packed_args)
            except Exception as e:
                raise DSLRuntimeError(f"💥💥💥 Runtime Crash 💥💥💥", cause=e)
    def _dump_ptx(self, sym_name: str, func_sym: str, ptx_bytes: bytes) -> str | None:
        """
        If CUTE_DSL_DUMP_PTX is truthy, write PTX to disk and return the path.
        Controls:
        - CUTE_DSL_DUMP_PTX: 1/true/yes to enable
        - CUTE_DSL_PTX_DIR: output directory (default: ".ptx")
        - CUTE_DSL_PTX_GZ: 1/true/yes to gzip the file (suffix .ptx.gz)
        """
        import re
        import hashlib

        breakpoint()
        dump = str(os.getenv("CUTE_DSL_DUMP_PTX", "")).strip().lower() in ("1", "true", "yes", "y", "on")
        if not dump:
            return None

        outdir = os.getenv("CUTE_DSL_PTX_DIR", ".ptx")
        os.makedirs(outdir, exist_ok=True)

        # Derive nice filename bits from the PTX header
        tgt = "target"
        ver = "ptx"
        try:
            text = ptx_bytes.decode("utf-8", errors="replace")
            m_ver = re.search(r"^\s*\.version\s+(\d+)\.(\d+)", text, re.M)
            m_tgt = re.search(r"^\s*\.target\s+([^\s,]+)", text, re.M)
            if m_ver:
                ver = f"ptx{m_ver.group(1)}{m_ver.group(2)}"  # e.g., ptx80
            if m_tgt:
                tgt = m_tgt.group(1).replace(",", "_")
        except Exception:
            pass

        h = hashlib.sha1(ptx_bytes).hexdigest()[:12]
        base = f"{sym_name}__{func_sym}__{tgt}__{ver}__{h}.ptx"

        outpath = os.path.join(outdir, base)
        breakpoint()
        # Atomic-ish write
        tmp = outpath + ".tmp"
        ptx_bytes = ptx_bytes.rstrip(b"\x00")

        try:
            with open(tmp, "wb") as f:
                f.write(ptx_bytes.rstrip())
            os.replace(tmp, outpath)

            try:
                log().info(f"Dumped PTX to {outpath}")
            except Exception:
                pass

            return outpath

        finally:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass

    def update_jit_cuda_modules(self, kernel_symbols):
        """
        Preload CUDA module(s) from the MLIR module. By default, we extract embedded cubin(s).
        If CUTE_DSL_PTX is truthy, we instead extract PTX text and JIT it via the CUDA driver.
        """
        if len(kernel_symbols) > 0:
            prefer_ptx = str(os.getenv("CUTE_DSL_PTX", "")).strip().lower() in (
                "1",
                "true",
                "yes",
                "y",
                "on",
            )
            extra_args = []
            module = self.ir_module
            cuda_kernel_cache = dict()
            cuda_driver_version = cuda_helpers.get_driver_version()

            for sym in kernel_symbols:
                if sym in cuda_kernel_cache:
                    log().debug(f"Symbol {sym} already in cache")
                    continue

                log().debug(
                    f"Loading CUDA module for symbol: {sym} (prefer_ptx={prefer_ptx})"
                )
                breakpoint()
                loaded = False

                # --- PTX path (optional) ---
                if prefer_ptx:
                  # --- Pure-text path: dump whole module once, then regex out PTX blobs ---
                    text = module_to_text(self.ir_module)   # full module text
                    breakpoint()
                    blobs = extract_all_ptx_from_text(text) # list of PTX blobs
                    # Try to find a blob whose entries include our symbol; else take the first blob.
                    chosen = None
                    for item in blobs:
                        if any(sym == e or sym.endswith("_kernel") and sym[:-7] == e for e in item["entries"]):
                            chosen = item; break
                    if chosen is None and blobs:
                        chosen = blobs[0]

                    if chosen is not None:
                        ptx_bytes = chosen["ptx"]
                        # Optional: dump to disk if requested
                        try:
                            self._dump_ptx(sym, sym, ptx_bytes)  # no-op unless you added the dumper earlier
                        except Exception:
                            pass
                        try:
                            cubin_module = cuda_helpers.load_cubin_module_data(ptx_bytes)  # cuModuleLoadDataEx accepts PTX
                            func_sym = sym
                            if func_sym.encode() not in ptx_bytes and chosen["entries"]:
                                func_sym = chosen["entries"][0]  # pick the first .entry if the sym isn’t present
                            kernel_ptr = cuda_helpers.get_kernel_function(cubin_module, func_sym)
                            if cuda_driver_version >= 11080:
                                cuda_helpers.set_kernel_attribute(
                                    kernel_ptr,
                                    cuda_helpers.cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
                                    1,
                                )
                            cuda_kernel_cache[sym] = CudaSingleModule(cubin_module, kernel_ptr)
                            loaded = True
                        except Exception as e:
                            log().warning(f"PTX JIT load failed for {sym}: {e}")

                # --- Default cubin path (fallback or primary) ---
                if not loaded:
                    def walk_callback_cubin(sym_name, func_sym, cubin_data):
                        nonlocal loaded
                        cubin_module = cuda_helpers.load_cubin_module_data(cubin_data)
                        kernel_ptr = cuda_helpers.get_kernel_function(
                            cubin_module, func_sym
                        )
                        if cuda_driver_version >= 11080:
                            cuda_helpers.set_kernel_attribute(
                                kernel_ptr,
                                cuda_helpers.cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
                                1,
                            )
                        cuda_kernel_cache[sym_name] = CudaSingleModule(
                            cubin_module, kernel_ptr
                        )
                        loaded = True
                        log().debug(f"Loaded CUBIN for {sym_name}/{func_sym}")

                    self.walk_module_and_get_cubin_data(module, sym, walk_callback_cubin)

                # Append kernel ptr if we actually loaded something.
                if sym in cuda_kernel_cache:
                    extra_args.append(
                        ctypes.c_void_p(cuda_kernel_cache[sym].kernel_ptr.getPtr())
                    )

            self.cuda_modules = CudaModules(cuda_kernel_cache.values(), extra_args)

        return self

    def _get_escaped_cubin_bytes(self, cubin_data):
        """Unescape MLIR-quoted byte payload (works for both CUBIN and PTX text)."""

        def ishex(inp):
            return (
                inp in range(0x30, 0x3A)
                or inp in range(0x61, 0x67)
                or inp in range(0x41, 0x47)
            )

        converted = bytearray()
        idx = 0
        while idx < len(cubin_data):
            if cubin_data[idx] == 0x5C:  # backslash
                if idx + 2 < len(cubin_data) and ishex(cubin_data[idx + 1]) and ishex(
                    cubin_data[idx + 2]
                ):
                    converted += bytearray.fromhex(
                        cubin_data[idx + 1 : idx + 3].decode()
                    )
                    idx += 3
                elif idx + 1 < len(cubin_data) and cubin_data[idx + 1] == 0x5C:
                    converted.append(cubin_data[idx])
                    idx += 2
                else:
                    # Other escapes like \n, \t etc. — just drop the backslash.
                    idx += 1
            else:
                converted.append(cubin_data[idx])
                idx += 1
        return bytes(converted)

    def walk_module_and_get_cubin_data(self, module, sym, callback):
        """Walk gpu.binary ops, extract CUBIN payload, invoke callback(sym, func_sym, cubin_bytes)."""

        def walk_gpu_binary_op(op):
            if op.name != "gpu.binary":
                return ir.WalkResult.ADVANCE
            s = io.BytesIO()
            op.write_bytecode(s)
            blob = s.getvalue()
            if sym.encode() not in blob:
                return ir.WalkResult.ADVANCE

            if ("kernels" != op.opview.sym_name.value and sym != op.opview.sym_name.value):
                return ir.WalkResult.ADVANCE

            func_sym = sym
            if sym == op.opview.sym_name.value and not sym.endswith("_kernel"):
                func_sym = sym.rsplit("_", 1)[0]

            try:
                payload = blob.split(b'bin = "')[1].split(b'">')[0]
            except Exception:
                return ir.WalkResult.ADVANCE

            cubin_data = self._get_escaped_cubin_bytes(payload)

            # Quick guard: if the payload *looks* like PTX (starts with '.version' etc.), skip here.
            head = cubin_data[:16].lstrip()
            if head.startswith(b".version") or head.startswith(b"// .globl"):
                return ir.WalkResult.ADVANCE

            callback(sym, func_sym, cubin_data)
            return ir.WalkResult.ADVANCE

        module.operation.walk(walk_gpu_binary_op)

    # def walk_module_and_get_ptx_data(self, module, sym, callback):
    #     """Walk gpu.binary ops, extract PTX text payload, invoke callback(sym, func_sym, ptx_bytes)."""

    #     def looks_like_ptx(buf: bytes) -> bool:
    #         head = buf[:64].lstrip()
    #         return (
    #             head.startswith(b".version")
    #             or head.startswith(b".target")
    #             or b".entry" in head
    #         )

    #     def walk_gpu_binary_op(op):
    #         if op.name != "gpu.binary":
    #             return ir.WalkResult.ADVANCE
    #         s = io.BytesIO()
    #         op.write_bytecode(s)
    #         blob = s.getvalue()
    #         if sym.encode() not in blob:
    #             return ir.WalkResult.ADVANCE

    #         if ("kernels" != op.opview.sym_name.value and sym != op.opview.sym_name.value):
    #             return ir.WalkResult.ADVANCE

    #         func_sym = sym
    #         if sym == op.opview.sym_name.value and not sym.endswith("_kernel"):
    #             func_sym = sym.rsplit("_", 1)[0]

    #         try:
    #             payload = blob.split(b'bin = "')[1].split(b'">')[0]
    #         except Exception:
    #             return ir.WalkResult.ADVANCE

    #         ptx_bytes = self._get_escaped_cubin_bytes(payload)
    #         breakpoint()
    #         # Accept only if it really looks like PTX; otherwise bail.
    #         if not looks_like_ptx(ptx_bytes):
    #             return ir.WalkResult.ADVANCE

    #         callback(sym, func_sym, ptx_bytes)
    #         return ir.WalkResult.ADVANCE

    #     module.operation.walk(walk_gpu_binary_op)


    def walk_module_and_get_ptx_data(self, module, sym, callback):
        """
        Extract PTX from gpu.binary's 'objects' ArrayAttr.
        Fires callback(sym, func_sym, ptx_bytes) once per PTX object found.
        """
        import re
        TRACE = True #bool(int(os.getenv("CUTE_DSL_TRACE_WALK", "0")))

        def looks_like_ptx(buf: bytes) -> bool:
            h = buf[:128].lstrip()
            return (h.startswith(b".version") or h.startswith(b".target")) and b".entry" in buf

        def first_entry_name(ptx: bytes) -> str | None:
            m = re.search(rb'(?m)^\s*(?:\.visible\s+)?\.entry\s+([A-Za-z0-9_.$@-]+)', ptx)
            return m.group(1).decode() if m else None

        def unescape_quoted_bytes(s: str) -> bytes:
            # MLIR prints C-escaped strings (\" \\ \xx). Reuse your cubin unescaper,
            # but it expects bytes with the escapes literally present.
            return self._get_escaped_cubin_bytes(s.encode("utf-8"))

        def walk(op):
            # IMPORTANT: reliable way to get the op name in this binding:
            name = op.operation.name
            if TRACE:
                try:
                    print(f"[walk] saw: {name}")
                except Exception:
                    pass

            if name != "gpu.binary":
                return ir.WalkResult.ADVANCE

            attrs = op.operation.attributes
            objs = attrs.get("objects", None)
            if objs is None:
                # Some builds use 'object' singular; handle both just in case.
                objs = attrs.get("object", None)
            if objs is None:
                if TRACE:
                    print("[walk] gpu.binary has no 'objects' attr")
                return ir.WalkResult.ADVANCE

            # objs is an ArrayAttr; iterate its elements.
            found_any = False
            for idx, obj in enumerate(objs):
                s = str(obj)  # e.g., '#gpu.object<#nvvm.target<...>, assembly = "..." >'
                # Prefer PTX assembly
                m = re.search(r'assembly\s*=\s*"((?:\\.|[^"])*)"', s, re.S)
                payload_escaped = None
                kind = None
                if m:
                    payload_escaped = m.group(1)
                    kind = "assembly"
                else:
                    # fallback: binary cubin
                    m2 = re.search(r'\bbin\s*=\s*"((?:\\.|[^"])*)"', s, re.S)
                    if m2:
                        payload_escaped = m2.group(1)
                        kind = "bin"

                if payload_escaped is None:
                    if TRACE:
                        print(f"[walk] gpu.object[{idx}]: neither assembly nor bin found")
                    continue

                blob = unescape_quoted_bytes(payload_escaped)

                # Only accept it as PTX for this walker.
                if not looks_like_ptx(blob):
                    if TRACE:
                        print(f"[walk] gpu.object[{idx}]: {kind} not PTX; skipping")
                    continue

                func_sym = sym
                if not func_sym or func_sym.encode() not in blob:
                    guessed = first_entry_name(blob)
                    if guessed:
                        func_sym = guessed

                if TRACE:
                    head = blob[:80].decode("utf-8", "replace").replace("\n", "\\n")
                    print(f"[walk] PTX[{idx}] .entry={func_sym!r} head={head}")

                callback(sym, func_sym, blob)
                found_any = True

            if TRACE and not found_any:
                print("[walk] gpu.binary had objects, but none looked like PTX")

            return ir.WalkResult.ADVANCE

        # Walk the whole module
        module.operation.walk(walk)
