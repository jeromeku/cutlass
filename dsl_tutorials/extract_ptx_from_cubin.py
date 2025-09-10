# ruff: noqa E402
from __future__ import annotations

import argparse
import importlib
import io
import os
import re
import subprocess
import sys
import tempfile
from typing import Dict, Tuple
from cutlass_logging import set_logging, set_ir_dump
set_logging(log_to_console=True)
set_ir_dump(dump_ptx=True)

import torch
from ptx_jit_executor import JitExecutor as PTXJitExecutor

from ptx_jit_executor import JitExecutor as PTXJitExecutor
import cutlass.base_dsl.jit_executor as jit_executor_mod  # import the *submodule*
import cutlass.base_dsl.dsl as dsl_mod

# Keep the old one so you can restore if needed
_ORIGINAL_JITEXECUTOR = jit_executor_mod.JitExecutor
jit_executor_mod.JitExecutor = PTXJitExecutor
dsl_mod.JitExecutor = PTXJitExecutor

import cutlass.base_dsl.jit_executor as jit_executor_mod
assert jit_executor_mod.JitExecutor is PTXJitExecutor
assert dsl_mod.JitExecutor is PTXJitExecutor

from cutlass.base_dsl import BaseDSL as base_dsl_mod
original_preprocess = base_dsl_mod.preprocess_pipeline

def _patched_preprocess(self, pipeline: str, arch: str) -> str:  # type: ignore[override]
    p = original_preprocess(self, pipeline, arch)
    # Replace the binary output selection to PTX text
    breakpoint()
    # cubin-features=\"+ptx87\"
    return p.replace("cubin-format=bin", "cubin-format=assembly")

# Apply patches
base_dsl_mod.preprocess_pipeline = _patched_preprocess  # type: ignore[assignment]

def _is_elf(blob: bytes) -> bool:
    return len(blob) >= 4 and blob.startswith(b"\x7fELF")


def _extract_cubins_from_ir(ir_module) -> Dict[str, bytes]:
    """Walk gpu.binary ops and return {symbol: cubin_bytes}."""
    from cutlass._mlir import ir  # type: ignore

    results: Dict[str, bytes] = {}

    def walker(op):
        if op.name != "gpu.binary":
            return ir.WalkResult.ADVANCE
        buf = io.BytesIO()
        op.write_bytecode(buf)
        data = buf.getvalue()

        # symbol (if present)
        m_sym = re.search(rb"sym_name\s*=\s*\"([^\"]+)\"", data)
        sym = m_sym.group(1).decode("utf-8") if m_sym else "kernels"

        try:
            payload = data.split(b'bin = "', 1)[1].split(b'">', 1)[0]
        except Exception:
            return ir.WalkResult.ADVANCE

        # unescape MLIR bytecode string
        out = bytearray()
        i = 0
        def _ishex(b: int) -> bool:
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

        blob = bytes(out)
        if _is_elf(blob):
            results[sym] = blob
        return ir.WalkResult.ADVANCE

    ir_module.operation.walk(walker)
    return results


def _cuobjdump_dump_ptx(cubin_bytes: bytes, cuobjdump_path: str) -> Tuple[bool, str]:
    with tempfile.TemporaryDirectory() as td:
        cubin_file = os.path.join(td, "kernel.cubin")
        with open(cubin_file, "wb") as f:
            f.write(cubin_bytes)
        try:
            out = subprocess.check_output(
                [cuobjdump_path, "--dump-ptx", cubin_file], stderr=subprocess.STDOUT
            ).decode("utf-8", errors="ignore")
        except subprocess.CalledProcessError as e:
            return False, e.output.decode("utf-8", errors="ignore")
        except FileNotFoundError:
            return False, f"cuobjdump not found at: {cuobjdump_path}"

    has_ptx = (".version" in out) and (".entry" in out)
    return has_ptx, out


def _default_smem_allocation_fixture():
    # Minimal inputs that allow compile-only specialization of run_allocation_kernel
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from ampere.smem_allocator import run_allocation_kernel

    cutlass.cuda.initialize_cuda_context()
    import torch
    dst_a = torch.zeros((8, 4), dtype=torch.float32, device="cuda")
    dst_b = torch.zeros((8, 2), dtype=torch.float32, device="cuda")
    dst_c = torch.zeros((16, 2), dtype=torch.float32, device="cuda")
    args = (
        0.5,
        from_dlpack(dst_a),
        1.0,
        from_dlpack(dst_b),
        2.0,
        from_dlpack(dst_c),
    )
    return run_allocation_kernel, args


def _compile_only(func, *args, arch: str | None, toolkit: str | None):
    import cutlass.cute as cute
    # from dsl_tutorials.utils.ptx_tools import override_cute_arch

    # with override_cute_arch(arch, toolkit_path=toolkit):
        # cute.compile returns the JitExecutor when compile_only=True (internal default of compile)
    executor = cute.compile(func, *args)
    executor.ir_module.dump()
    return executor


def main():
    p = argparse.ArgumentParser(description="Extract PTX from CuTe CUBIN via cuobjdump")
    p.add_argument("--function", type=str, default=None, help="module:function path. If omitted, uses smem allocator example.")
    p.add_argument("--arch", type=str, default=None, help="Target arch like sm_90a")
    p.add_argument("--toolkit", type=str, default=None, help="CUDA toolkit path")
    p.add_argument("--cuobjdump", type=str, default=os.environ.get("CUOBJDUMP", "cuobjdump"), help="Path to cuobjdump")
    p.add_argument("--max-chars", type=int, default=800, help="Chars of PTX preview to print per symbol")
    args = p.parse_args()

    # if args.function:
    #     if ":" not in args.function:
    #         print("--function must be module:function", file=sys.stderr)
    #         return 2
    #     mod_name, fn_name = args.function.split(":", 1)
    #     mod = importlib.import_module(mod_name)
    #     func = getattr(mod, fn_name)
    #     # No generic arg synthesis here; user-provided function should not require runtime-only values.
    #     exe = _compile_only(func, arch=args.arch, toolkit=args.toolkit)
    # else:
    func, fargs = _default_smem_allocation_fixture()
    exe = _compile_only(func, *fargs, arch=args.arch, toolkit=args.toolkit)

    # cubins = _extract_cubins_from_ir(exe.ir_module)
    # if not cubins:
    #     print("No gpu.binary CUBIN payloads found in IR.")
    #     return 1

    # any_ptx = False
    # for sym, cub in cubins.items():
    #     ok, txt = _cuobjdump_dump_ptx(cub, args.cuobjdump)
    #     print(f"=== {sym} ===")
    #     if not ok:
    #         print("[cuobjdump] PTX not found or error. Output:")
    #         print(txt)
    #         continue
    #     any_ptx = True
    #     preview = txt[: args.max_chars]
    #     print(preview)
    #     if len(txt) > len(preview):
    #         print("... [truncated]")

    # return 0 if any_ptx else 3


if __name__ == "__main__":
    sys.exit(main())

