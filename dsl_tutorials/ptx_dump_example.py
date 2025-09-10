#!/usr/bin/env python3
"""
Example: generate PTX for a CuTe JIT function by temporarily forcing
the pipeline to emit PTX and using ptx_tools to extract it.

This example compiles the smem allocation tutorial's @cute.jit wrapper
(`run_allocation_kernel`) under a PTX-emitting pipeline (no CUDA driver
preload), then dumps the embedded PTX from the MLIR gpu.binary payload.

Usage:
  python dsl_tutorials/ampere/ptx_dump_example.py \
    --arch sm_90a \
    --toolkit /usr/local/cuda \
    --max-chars 800

Notes:
- Do not attempt to execute kernels while the PTX pipeline patch is active.
- This is for inspection/debugging only.
"""
# ruff: noqa E402

from __future__ import annotations
from pathlib import Path
import sys

sys.path.insert(0, "home/jeromeku/cutlass")
print(sys.path)

from utils.ptx_tools import (
    extract_ptx_from_cute,
    override_cute_arch,
)
from utils.patch_ptx_pipeline import patch_pipeline_to_ptx

import argparse

import torch



def main():

    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from ampere.smem_allocator import run_allocation_kernel

    parser = argparse.ArgumentParser(description="Dump PTX from a CuTe JIT function")
    parser.add_argument("--arch", type=str, default=None, help="Target arch, e.g. sm_90a")
    parser.add_argument("--toolkit", type=str, default=None, help="CUDA toolkit path")
    parser.add_argument(
        "--max-chars", type=int, default=600, help="Preview chars to print per symbol"
    )
    args = parser.parse_args()

    # Prepare dummy tensors and constants matching run_allocation_kernel signature
    cutlass.cuda.initialize_cuda_context()
    dst_a = torch.zeros((8, 4), dtype=torch.float32, device="cuda")
    dst_b = torch.zeros((8, 2), dtype=torch.float32, device="cuda")
    dst_c = torch.zeros((16, 2), dtype=torch.float32, device="cuda")
    const_a, const_b, const_c = 0.5, 1.0, 2.0

    compile_args = (
        const_a,
        from_dlpack(dst_a),
        const_b,
        from_dlpack(dst_b),
        const_c,
        from_dlpack(dst_c),
    )

    # Force PTX emission and disable driver CUBIN preload during compile_only
    with override_cute_arch(args.arch, toolkit_path=args.toolkit):
        with patch_pipeline_to_ptx(skip_driver_load=True):
            ptx_map = extract_ptx_from_cute(
                run_allocation_kernel,
                *compile_args,
                verbose=True,
            )

    if not ptx_map:
        print("No PTX found. Ensure the patch is active and function compiled.")
        return 1

    for sym, ptx in ptx_map.items():
        print(f"===== PTX for symbol: {sym} =====")
        preview = ptx[: args.max_chars]
        print(preview)
        if len(ptx) > len(preview):
            print("... [truncated]")

    return 0


if __name__ == "__main__":
    sys.exit(main())

