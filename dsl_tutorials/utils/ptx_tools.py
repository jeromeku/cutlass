"""
Utilities for extracting PTX from CuTe DSL JIT-compiled kernels.

This module offers `extract_ptx_from_cute`, which accepts either:
- a function decorated with `@cute.jit` (plus its example compile-time args), or
- an already-compiled JIT executor object (from `cute.compile`).

It will:
1) Ensure the kernel is compiled (without launching),
2) Walk the MLIR module to locate `gpu.binary` blobs, and
3) Return a mapping: {symbol_name: PTX_string} when possible.

If the binary is ELF/CUBIN-only, it optionally shells out to `cuobjdump --dump-ptx`
to recover embedded PTX if present. If PTX is not embedded, the function will
return an empty dict for that symbol.

Usage example (from a CuTe tutorial):

    import cutlass.cute as cute
    from dsl_tutorials.utils.ptx_tools import extract_ptx_from_cute

    @cute.jit
    def run(a, b, c):
        kernel(a, b, c).launch(grid=(1,1,1), block=(128,1,1))

    # Compile only; no launch. Provide compile-time arguments if needed.
    ptx_map = extract_ptx_from_cute(run)
    for sym, ptx in ptx_map.items():
        print(f"===== {sym} =====\n{ptx[:500]}\n...")

Notes:
- This reads the IR module attached to the JIT executor and looks for `gpu.binary`.
- If the pipeline was configured to emit PTX in the binary, it will decode UTF-8 text.
- If the pipeline emits CUBIN-only (the default in the Cutlass CuTe pipeline), we try
  `cuobjdump --dump-ptx` as a best-effort fallback.
"""

from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Dict, Optional, Tuple, Union
from contextlib import contextmanager


def _is_ptx_text(blob: bytes) -> bool:
    """Heuristically detect PTX text inside a blob."""
    # PTX usually starts with ".version X.Y" and contains many ASCII directives
    head = blob[:256].lower()
    if b".version" in head or b".target" in head or b".address_size" in head:
        try:
            head.decode("utf-8")
            return True
        except UnicodeDecodeError:
            return False
    return False


def _is_elf(blob: bytes) -> bool:
    """Detect ELF header in blob (CUBIN is an ELF variant)."""
    return len(blob) >= 4 and blob.startswith(b"\x7fELF")


def _walk_gpu_binaries(ir_module) -> Dict[str, bytes]:
    """Walk gpu.binary ops in an MLIR module and collect their payloads.

    Returns dict mapping {symbol_name: raw_bytes}.
    """
    from cutlass._mlir import ir  # type: ignore

    results: Dict[str, bytes] = {}

    def walker(op):
        if op.name != "gpu.binary":
            return ir.WalkResult.ADVANCE
        buf = io.BytesIO()
        op.write_bytecode(buf)
        data = buf.getvalue()

        # Extract symbol and the `bin = "..."` payload
        # Expected bytecode pattern contains:  sym_name = "<sym>"  ...  bin = "<escaped-bytes>"
        try:
            # Symbol name
            m_sym = re.search(rb"sym_name\s*=\s*\"([^\"]+)\"", data)
            sym_name = m_sym.group(1).decode("utf-8") if m_sym else "kernels"

            # Binary payload (escaped)
            payload = data.split(b'bin = "', 1)[1].split(b'">', 1)[0]

            # Unescape MLIR bytecode string (handles "\\AA" hex and "\\\\")
            out = bytearray()
            i = 0
            def _ishex(b):
                return 0x30 <= b <= 0x39 or 0x41 <= b <= 0x46 or 0x61 <= b <= 0x66
            while i < len(payload):
                b0 = payload[i]
                if b0 == 0x5C:  # '\\'
                    if i + 2 < len(payload) and _ishex(payload[i+1]) and _ishex(payload[i+2]):
                        out += bytearray.fromhex(payload[i+1:i+3].decode())
                        i += 3
                        continue
                    if i + 1 < len(payload) and payload[i+1] == 0x5C:
                        out.append(0x5C)
                        i += 2
                        continue
                out.append(b0)
                i += 1

            results[sym_name] = bytes(out)
        except Exception:
            # Ignore malformed entries
            return ir.WalkResult.ADVANCE

        return ir.WalkResult.ADVANCE

    ir_module.operation.walk(walker)
    return results


def _cuobjdump_dump_ptx(cubin_bytes: bytes, cuobjdump_path: Optional[str]) -> Optional[str]:
    """Attempt to extract PTX from a CUBIN via cuobjdump.

    Returns PTX text on success, else None.
    """
    path = cuobjdump_path or shutil.which("cuobjdump")
    if not path:
        return None

    with tempfile.TemporaryDirectory() as td:
        cubin_file = os.path.join(td, "kernel.cubin")
        with open(cubin_file, "wb") as f:
            f.write(cubin_bytes)
        try:
            # --dump-ptx prints any embedded PTX (if present)
            out = subprocess.check_output([path, "--dump-ptx", cubin_file], stderr=subprocess.STDOUT)
        except Exception:
            return None
    try:
        text = out.decode("utf-8", errors="ignore")
    except Exception:
        return None
    # Heuristic: ensure PTX markers present
    return text if ".version" in text and ".entry" in text else None


def _ensure_executor(obj: Union[Callable[..., Any], Any], *compile_args, **compile_kwargs):
    """Given a @cute.jit wrapper or a JIT executor, return a JIT executor.

    This calls `cute.compile` if needed with `compile_only=True`.
    """
    # Already looks like a JitExecutor (duck-typing via attributes)
    if hasattr(obj, "ir_module") and hasattr(obj, "capi_func"):
        return obj

    try:
        import cutlass.cute as cute  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "cutlass.cute is not importable. Ensure the CuTe DSL Python package is installed and on PYTHONPATH."
        ) from e

    # Ensure compile without caching the launch-time state
    compile_kwargs = dict(compile_kwargs) if compile_kwargs else {}
    compile_kwargs.setdefault("options", "")
    return cute.compile(obj, *compile_args, **compile_kwargs)


@contextmanager
def override_cute_arch(
    arch: Optional[str] = None,
    *,
    toolkit_path: Optional[str] = None,
    dsl_prefix: str = "CUTE_DSL",
):
    """Temporarily override CuTe DSL target arch/toolkit via env vars.

    - Sets `{dsl_prefix}_ARCH` (e.g., `CUTE_DSL_ARCH`) if `arch` is provided.
    - Sets `CUDA_TOOLKIT_PATH` if `toolkit_path` is provided.
    - Clears CuTe env helper caches so the override is observed on next DSL init.

    Example:
        with override_cute_arch("sm_90a", toolkit_path="/usr/local/cuda"):
            exec = cute.compile(func, *args)
    """
    var_arch = f"{dsl_prefix}_ARCH"
    old_arch = os.environ.get(var_arch)
    old_toolkit = os.environ.get("CUDA_TOOLKIT_PATH")

    try:
        if arch is not None:
            os.environ[var_arch] = arch
        if toolkit_path is not None:
            os.environ["CUDA_TOOLKIT_PATH"] = toolkit_path

        # Clear CuTe env caches so new DSL instances re-read env
        try:
            from cutlass.CuTeDSL.base_dsl import env_manager as _env  # type: ignore
        except Exception:
            try:
                # Alternate package layout
                from CuTeDSL.base_dsl import env_manager as _env  # type: ignore
            except Exception:
                _env = None
        if _env is not None:
            for fn_name in ("get_str_env_var", "get_bool_env_var", "get_int_env_var"):
                fn = getattr(_env, fn_name, None)
                if fn and hasattr(fn, "cache_clear"):
                    fn.cache_clear()  # type: ignore[attr-defined]
        yield
    finally:
        # restore
        if arch is not None:
            if old_arch is None:
                os.environ.pop(var_arch, None)
            else:
                os.environ[var_arch] = old_arch
        if toolkit_path is not None:
            if old_toolkit is None:
                os.environ.pop("CUDA_TOOLKIT_PATH", None)
            else:
                os.environ["CUDA_TOOLKIT_PATH"] = old_toolkit
        # clear again to pick up restored values
        try:
            from cutlass.CuTeDSL.base_dsl import env_manager as _env  # type: ignore
        except Exception:
            try:
                from CuTeDSL.base_dsl import env_manager as _env  # type: ignore
            except Exception:
                _env = None
        if _env is not None:
            for fn_name in ("get_str_env_var", "get_bool_env_var", "get_int_env_var"):
                fn = getattr(_env, fn_name, None)
                if fn and hasattr(fn, "cache_clear"):
                    fn.cache_clear()  # type: ignore[attr-defined]


def _effective_arch_report(executor, dsl_prefix: str) -> str:
    env_arch = os.environ.get(f"{dsl_prefix}_ARCH")
    env_toolkit = os.environ.get("CUDA_TOOLKIT_PATH")
    dsl_arch = getattr(getattr(executor, "dsl", object()), "envar", object())
    dsl_arch = getattr(dsl_arch, "arch", None)
    pass_key = getattr(getattr(executor, "dsl", object()), "pass_sm_arch_name", None)
    # Best-effort device capability
    dev_cc = None
    try:
        from cutlass.CuTeDSL.base_dsl.runtime.cuda import get_compute_capability_major_minor  # type: ignore
        dev_cc = get_compute_capability_major_minor()
    except Exception:
        try:
            from CuTeDSL.base_dsl.runtime.cuda import get_compute_capability_major_minor  # type: ignore
            dev_cc = get_compute_capability_major_minor()
        except Exception:
            dev_cc = None
    dev_cc_s = f"{dev_cc[0]}.{dev_cc[1]}" if isinstance(dev_cc, tuple) and dev_cc[0] is not None else "unknown"
    return (
        "[CuTe PTX Tools] Architecture Registration\n"
        f"  - Env {dsl_prefix}_ARCH: {env_arch or 'not set'}\n"
        f"  - Env CUDA_TOOLKIT_PATH: {env_toolkit or 'not set'}\n"
        f"  - DSL pass key: {pass_key or 'unknown'}\n"
        f"  - DSL envar.arch: {dsl_arch or 'unknown'}\n"
        f"  - Detected device CC: {dev_cc_s}\n"
        f"  - Effective pipeline option: {pass_key or 'cubin-chip'}={dsl_arch or env_arch or 'unknown'}\n"
    )


def compile_with_arch(
    obj: Union[Callable[..., Any], Any],
    *compile_args,
    arch: Optional[str] = None,
    toolkit_path: Optional[str] = None,
    dsl_prefix: str = "CUTE_DSL",
    verbose: bool = False,
    **compile_kwargs,
):
    """Compile a @cute.jit function or return executor with an arch override.

    This is a thin wrapper over `cute.compile` that temporarily sets
    `{dsl_prefix}_ARCH` and `CUDA_TOOLKIT_PATH` during compilation.
    """
    with override_cute_arch(arch, toolkit_path=toolkit_path, dsl_prefix=dsl_prefix):
        executor = _ensure_executor(obj, *compile_args, **compile_kwargs)
        if verbose:
            print(_effective_arch_report(executor, dsl_prefix))
        return executor


def extract_ptx_from_cute(
    obj: Union[Callable[..., Any], Any],
    *compile_args,
    arch: Optional[str] = None,
    toolkit_path: Optional[str] = None,
    dsl_prefix: str = "CUTE_DSL",
    cuobjdump: Optional[str] = None,
    verbose: bool = False,
    **compile_kwargs,
) -> Dict[str, str]:
    """Extract PTX text for each kernel in a CuTe JIT-compiled function.

    Parameters:
    - obj: Either a `@cute.jit`-decorated callable or an already-compiled executor.
    - *compile_args/**compile_kwargs: Arguments to compile the function (no launch).
      These are only used if `obj` is a callable. They should include any values
      necessary for specializing Constexpr parameters, if present.
    - cuobjdump: Optional explicit path to `cuobjdump`. When provided (or discoverable
      on PATH), used as a fallback to derive PTX from CUBINs that still embed PTX.

    Returns: Dict mapping `{symbol_name: ptx_text}`. Entries may be omitted when PTX
    is not available (e.g., CUBIN without embedded PTX).
    """
    # Ensure JIT with optional arch/toolkit override
    if arch is not None or toolkit_path is not None:
        executor = compile_with_arch(
            obj,
            *compile_args,
            arch=arch,
            toolkit_path=toolkit_path,
            dsl_prefix=dsl_prefix,
            verbose=verbose,
            **compile_kwargs,
        )
    else:
        executor = _ensure_executor(obj, *compile_args, **compile_kwargs)
        if verbose:
            print(_effective_arch_report(executor, dsl_prefix))

    # Collect gpu.binary payloads from the IR
    blobs = _walk_gpu_binaries(executor.ir_module)
    out: Dict[str, str] = {}

    for sym, blob in blobs.items():
        if _is_ptx_text(blob):
            try:
                out[sym] = blob.decode("utf-8")
                continue
            except UnicodeDecodeError:
                pass
        if _is_elf(blob):
            ptx = _cuobjdump_dump_ptx(blob, cuobjdump)
            if ptx:
                # Optionally split by symbols; keep full text to preserve directives
                out[sym] = ptx
                if verbose:
                    print(f"[CuTe PTX Tools] PTX recovered via cuobjdump for symbol '{sym}' (ELF/CUBIN).")
            else:
                # No PTX embedded or cuobjdump unavailable
                if verbose:
                    print(f"[CuTe PTX Tools] No embedded PTX for symbol '{sym}' or cuobjdump unavailable.")
                continue
        else:
            # Unknown format; skip
            if verbose:
                print(f"[CuTe PTX Tools] Unrecognized binary payload for symbol '{sym}', skipping.")
            continue

    return out


__all__ = ["extract_ptx_from_cute"]
