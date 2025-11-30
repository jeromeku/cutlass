
# ruff: noqa
import os
import sys
# Make the parent experiments directory importable when running this file directly.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ["MLIR_CUDA_DEBUG"] = "1"
os.environ["MLIR_ENABLE_DUMP"] = "1"
os.environ["LLVM_ENABLE_DUMP"] = "1"
os.environ["LLVM_DEBUG_ONLY"] = "orc"

from utils.mlir_pipeline import dump_mlir_pipeline
from utils.logging import patch_cutlass_env

patch_cutlass_env(log_to_console=False, logdir="tvm_logs", disable_cache=True)

import cutlass.cute as cute
import torch
from contextlib import nullcontext
from functools import partial

from cutlass.base_dsl.compiler import Compiler
from cutlass.base_dsl.jit_executor import JitCompiledFunction, JitExecutor
from cutlass.cutlass_dsl.cuda_jit_executor import CudaDialectJitCompiledFunction
from cutlass.cutlass_dsl.tvm_ffi_provider import TVMFFIJitCompiledFunction

CompiledTypes = JitCompiledFunction | JitExecutor

DUMP_PIPELINE = False

if DUMP_PIPELINE:
    pipeline_context = partial(dump_mlir_pipeline, dump_dir="pipeline")
    pipeline_context_tvm = partial(dump_mlir_pipeline, dump_dir="pipeline_tvm")
else:
    pipeline_context = nullcontext()
    pipeline_context_tvm = nullcontext()

@cute.kernel
def device_add_one(a: cute.Tensor, b: cute.Tensor):
    threads_per_block = 128
    cta_x_, _, _ = cute.arch.block_idx()
    tid_x, _, _ = cute.arch.thread_idx()
    tid = cta_x_ * threads_per_block + tid_x
    if tid < a.shape[0]:
        b[tid] = a[tid] + 1.0

@cute.jit
def add_one(a: cute.Tensor, b: cute.Tensor):
    n = a.shape[0]
    threads_per_block = 128
    blocks = (n + threads_per_block - 1) // threads_per_block
    device_add_one(a, b).launch(
        grid=(blocks, 1, 1),
        block=(threads_per_block, 1, 1),
    )

def example_add_one():
    n = cute.sym_int()
    a_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
    b_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
    # compile the kernel with "--enable-tvm-ffi" option and example input tensors
    enabled_tvm = "--enable-tvm-ffi"
    from cutlass.base_dsl.compiler import CompileCallable
    compiler: CompileCallable = cute.compile

    compiled_add_one: CudaDialectJitCompiledFunction = compiler(add_one, a_cute, b_cute)
    print(type(compiled_add_one))
    
    with pipeline_context_tvm:    
        compiled_add_one_tvm: TVMFFIJitCompiledFunction = compiler(add_one, a_cute, b_cute, options=enabled_tvm)
        print(type(compiled_add_one_tvm))

    breakpoint()
    a_torch = torch.arange(10, dtype=torch.float32, device="cuda")
    b_torch = torch.empty(10, dtype=torch.float32, device="cuda")
    compiled_add_one.__call__(a_torch, b_torch)
    print("result of b_torch after compiled_add_one(a_torch, b_torch)")
    print(b_torch)

if __name__ == "__main__":
    example_add_one()
