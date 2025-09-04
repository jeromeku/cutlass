#!/usr/bin/env python3
import argparse
import importlib
import importlib.util
import inspect
import io
import json
import os
import sys
import time
from typing import Any, Tuple


def _import_from_path(path: str):
    """Import a module given a filesystem path to a .py file."""
    spec = importlib.util.spec_from_file_location("_user_mod_", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import from path: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def _stringify_ir(op, enable_debug_info=True) -> str:
    buf = io.StringIO()
    try:
        op.print(file=buf, enable_debug_info=enable_debug_info)
    except TypeError:
        # Older MLIR Python bindings use different kw names
        op.print(file=buf)
    return buf.getvalue()


def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def _walk_gpu_binaries_and_extract(module, outdir: str) -> Tuple[list, list]:
    """Walks gpu.binary ops, writes .cubin files, returns (symbols, filepaths)."""
    from cutlass_package._mlir import ir

    def ishex(b: int) -> bool:
        return (0x30 <= b <= 0x39) or (0x61 <= b <= 0x66) or (0x41 <= b <= 0x46)

    def unescape_cubin(data: bytes) -> bytes:
        out = bytearray()
        i = 0
        n = len(data)
        while i < n:
            if data[i] == 0x5C:  # '\\'
                if i + 2 < n and ishex(data[i + 1]) and ishex(data[i + 2]):
                    out += bytearray.fromhex(data[i + 1 : i + 3].decode())
                    i += 3
                    continue
                if i + 1 < n and data[i + 1] == 0x5C:
                    out.append(data[i])
                    i += 2
                    continue
            out.append(data[i])
            i += 1
        return bytes(out)

    symbols = []
    paths = []

    def visit(op):
        if op.name == "gpu.binary":
            s = io.BytesIO()
            op.write_bytecode(s)
            b = s.getvalue()
            # Extract symbol name and raw bin payload
            # bytecode fragment looks like: bin = "<escaped bytes>", sym_name = "..."
            try:
                raw = b.split(b'bin = "')[1].split(b'">')[0]
            except Exception:
                raw = b""
            try:
                sym = op.opview.sym_name.value  # type: ignore[attr-defined]
            except Exception:
                sym = "kernels"
            cubin = unescape_cubin(raw)
            if cubin:
                fname = f"{sym}.cubin"
                fpath = os.path.join(outdir, fname)
                with open(fpath, "wb") as f:
                    f.write(cubin)
                symbols.append(sym)
                paths.append(fpath)
        return ir.WalkResult.ADVANCE

    module.operation.walk(visit)
    return symbols, paths


def main():
    ap = argparse.ArgumentParser(description="Dump MLIR IR and extracted cubins for a cute.jit program")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--module", help="Python module path to import (e.g., pkg.mod)")
    g.add_argument("--file", help="Filesystem path to a .py file to import")
    ap.add_argument("--func", required=True, help="Function name to run (decorated with @cute.jit)")
    ap.add_argument("--args", default="[]", help="JSON list of positional args to pass (default: [])")
    ap.add_argument("--kwargs", default="{}", help="JSON dict of keyword args to pass (default: {})")
    ap.add_argument("--out", default=None, help="Output directory (default: cuteDSL_exploration/dumps/<ts>-<func>)")
    ap.add_argument("--pipeline", default=None, help="Override MLIR pipeline string (optional)")
    ap.add_argument("--print-each-pass", action="store_true", help="Attempt to dump IR after each pass (best-effort)")
    ap.add_argument("--arch", default=None, help="Override target arch for pipeline (e.g., sm_90a)")
    args = ap.parse_args()

    # Import runtime pieces we’ll need
    from cutlass_package._mlir import ir, passmanager
    from cutlass_package._mlir.dialects import func as func_d
    from cutlass_package.base_dsl.dsl import BaseDSL, DSLCallable
    from cutlass_package.cutlass_dsl import cutlass as cutlass_mod
    from cutlass_package.cutlass_dsl.cutlass import CutlassBaseDSL
    from cutlass_package.base_dsl import compiler as base_compiler
    from cutlass_package._mlir import execution_engine as ee, passmanager as pm_mod

    # Build a safe CuTeDSL replacement that avoids CuTeDSL.__init__ breakpoint
    class SafeCuTeDSL(CutlassBaseDSL):
        def __init__(self):
            provider = base_compiler.Compiler(pm_mod, ee)
            super().__init__(
                name="CUTE_DSL",
                compiler_provider=provider,
                pass_sm_arch_name="cubin-chip",
                device_compilation_only=False,
                preprocess=True,
            )

    safe_dsl = SafeCuTeDSL()
    # Monkey-patch CuTeDSL._get_dsl to return our safe instance before importing the user module
    cutlass_mod.CuTeDSL._get_dsl = classmethod(lambda cls: safe_dsl)  # type: ignore

    if args.module:
        mod = importlib.import_module(args.module)
    else:
        mod = _import_from_path(args.file)

    if not hasattr(mod, args.func):
        raise AttributeError(f"Function '{args.func}' not found in module {mod}")
    decorated = getattr(mod, args.func)

    # Materialize Python function (handle AST preprocessor) without executing
    func_ptr = BaseDSL._preprocess_and_execute(decorated)
    if isinstance(func_ptr, DSLCallable):
        f_body = func_ptr
        arg_spec = func_ptr.get_arg_spec()
    else:
        f_body = func_ptr
        arg_spec = inspect.getfullargspec(f_body)

    # Parse args/kwargs
    pos_args = json.loads(args.args)
    kw_args = json.loads(args.kwargs)

    # Create DSL instance and compute canonical call signature
    dsl = safe_dsl
    dsl.funcBody = f_body
    sig = dsl._check_arg_count(*pos_args, **kw_args)
    can_args, can_kwargs = dsl._canonicalize_args(sig, *pos_args, **kw_args)

    # Mangle symbol name similarly to normal JIT path
    mangled_name = dsl.mangle_name(args.func, can_args, arg_spec)

    # Build an original MLIR module (pre-pass)
    with ir.Context(), ir.Location.unknown():
        # Types of function parameters
        exe_args, func_types, _ = dsl.generate_mlir_function_types(
            f_body, mangled_name, can_args, can_kwargs, arg_spec
        )
        module, result = (None, None)

        def build_ir_module():
            nonlocal module, result
            module = ir.Module.create()
            unit_attr = ir.UnitAttr.get()
            module.operation.attributes["gpu.container_module"] = unit_attr
            with ir.InsertionPoint(module.body):
                # Build gpu.module @kernels with attrs set by CuTe DSL
                dsl._build_gpu_module({})
                fop = func_d.FuncOp(mangled_name, (func_types, []))
                fop.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
                with ir.InsertionPoint(fop.add_entry_block()):
                    ir_args, ir_kwargs = dsl.generate_execution_arguments(
                        can_args, can_kwargs, fop, arg_spec
                    )
                    result = f_body(*ir_args, **ir_kwargs)
                    func_d.ReturnOp([])

        build_ir_module()

        # Prepare output directory
        ts = _timestamp()
        default_out = os.path.join(
            os.path.dirname(__file__), "dumps", f"{ts}-{args.func}"
        )
        outdir = _ensure_dir(args.out or default_out)

        # Dump pre-pass IR
        with open(os.path.join(outdir, "00-before.mlir"), "w") as f:
            f.write(_stringify_ir(module.operation, enable_debug_info=True))

        # Build the pipeline string
        if args.pipeline is not None:
            pipeline = args.pipeline
        else:
            arch = args.arch or dsl.envar.arch or ""
            pipeline = dsl.preprocess_pipeline(dsl._get_pipeline(None), arch)

        # Run pass manager with optional IR printing per pass
        pm = passmanager.PassManager.parse(pipeline)
        # Attempt to enable per-pass IR printing (best-effort; may not exist in all wheels)
        if args.print_each_pass and hasattr(pm, "enable_ir_printing"):
            def _printer(stage, *, ir_module):
                # stage: 'before-pass' or 'after-pass'; we do after-pass snapshots
                if stage != "after-pass":
                    return
                # Derive a unique name using length of directory listing
                idx = len([n for n in os.listdir(outdir) if n.endswith('.mlir') and n.startswith('pass-')])
                fname = os.path.join(outdir, f"pass-{idx:03d}.mlir")
                try:
                    with open(fname, "w") as fh:
                        fh.write(_stringify_ir(ir_module.operation, enable_debug_info=False))
                except Exception:
                    pass

            try:
                pm.enable_ir_printing(print_module=True, print_after_only=False, print_before_only=False, print_callback=_printer)  # type: ignore
            except Exception:
                # Fallback silently
                pass

        # Actually run the passes
        pm.run(module.operation)

        # Dump post-pass IR
        with open(os.path.join(outdir, "zz-after.mlir"), "w") as f:
            f.write(_stringify_ir(module.operation, enable_debug_info=False))

        # Extract and dump cubin blobs (if any)
        cubin_dir = _ensure_dir(os.path.join(outdir, "cubin"))
        syms, files = _walk_gpu_binaries_and_extract(module, cubin_dir)
        with open(os.path.join(outdir, "cubin_index.json"), "w") as f:
            json.dump({"symbols": syms, "files": files}, f, indent=2)

        # Optional: report detected gpu.launch_func sites
        launches = []
        def _collect_launches(op):
            if op.name == "gpu.launch_func":
                # Best-effort textual capture
                launches.append(str(op))
            return ir.WalkResult.ADVANCE
        module.operation.walk(_collect_launches)
        with open(os.path.join(outdir, "launch_sites.txt"), "w") as f:
            f.write("\n\n".join(launches))

        # Summary
        report = {
            "function": args.func,
            "mangled": mangled_name,
            "pipeline": pipeline,
            "outdir": outdir,
            "cubin_symbols": syms,
            "cubin_files": files,
        }
        with open(os.path.join(outdir, "REPORT.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    sys.exit(main())
