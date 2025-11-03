# ruff: noqa E402
from cutlass_logging import patch_cutlass_loggers
patch_cutlass_loggers()
import shutil
import torch
from functools import partial
import os
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from contextlib import redirect_stdout
from cutlass._mlir import ir

def override_compiler(dump_ptx: bool = True, dump_dir="cute_mlir_dump", save_pipeline: bool = True):
    from cutlass.base_dsl import compiler as _cute_compiler    
    CompilationError = _cute_compiler.CompilationError
    
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir, exist_ok=True)

    def _compile(
        self: _cute_compiler.Compiler,
        module: ir.Module,
        pipeline: str,
        cuda_toolkit: str = "",
        arch: str = "",
        enable_verifier=False,
    ):
        """Compiles the module by invoking the pipeline."""
        print("Overriding compiler!!")
        try:
            ctx = module.context
            if dump_ptx:
                pipeline = pipeline.replace("cubin-format=bin", "cubin-format=isa")
            
            print(f"OVERRIDE_PIPELINE: {pipeline}")
            
            pm = self.passmanager.PassManager.parse(pipeline)
            parsed_pipeline = str(pm)

            if save_pipeline:
                print(parsed_pipeline, file=open(os.path.join(dump_dir, "pass_pipeline.txt"),'w'))
            print(f"Parsed pipeline: {parsed_pipeline}")

            ctx.enable_multithreading(False)
            ctx.emit_error_diagnostics = True
            pm.enable_verifier(enable_verifier)
            with redirect_stdout(open(os.path.join(dump_dir, 'module_before.mlir'), 'w')):
                module.operation.print()

            # Print before/after every pass, include locations, and dump each pass’s IR to a directory.
            pm.enable_ir_printing(
                print_before_all=False,
                print_after_all=True,
                print_module_scope=True,
                print_after_change=False,
                print_after_failure=False,
                enable_debug_info=True,
                tree_printing_dir_path=os.path.join(dump_dir, "MLIR_DUMP")
            )
            pm.run(module.operation)

        except Exception as e:
            shutil.rmtree(dump_dir)

            error_msg = str(e)
            nvvm_error, ir_msg = self._process_error(error_msg)

            if nvvm_error:
                raise CompilationError(
                    error_msg,
                    nvvm_error=nvvm_error,
                    ir_context=ir_msg,
                    cuda_toolkit=cuda_toolkit,
                    arch=arch,
                ) from e
            raise e
        else:
            with redirect_stdout(open(os.path.join(dump_dir, 'module_after.mlir'), 'w')):
                module.operation.print()
            print(f"Saved mlir outputs to {dump_dir}")

    _original_compile = _cute_compiler.Compiler.compile
    _cute_compiler.Compiler.compile = _compile

override_compiler()    
 
@cute.kernel
def naive_elementwise_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx

    # Map thread index to logical index of input tensor
    m, n = gA.shape
    ni = thread_idx % n
    mi = thread_idx // n

    # Map logical index to physical address via tensor layout
    a_val = gA[mi, ni]
    b_val = gB[mi, ni]

    # Perform element-wise addition
    gC[mi, ni] = a_val + b_val

@cute.jit
def naive_elementwise_add(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor
):
    num_threads_per_block = 256

    m, n = mA.shape
    kernel = naive_elementwise_add_kernel(mA, mB, mC)
    kernel.launch(grid=((m * n) // num_threads_per_block, 1, 1),
                  block=(num_threads_per_block, 1, 1))

M, N = 2048, 2048

a = torch.randn(M, N, device="cuda", dtype=torch.float16)
b = torch.randn(M, N, device="cuda", dtype=torch.float16)
c = torch.zeros(M, N, device="cuda", dtype=torch.float16)

a_ = from_dlpack(a, assumed_align=16)
b_ = from_dlpack(b, assumed_align=16)
c_ = from_dlpack(c, assumed_align=16)

# Compile kernel
try:
    naive_elementwise_add_ = cute.compile(naive_elementwise_add, a_, b_, c_)
except Exception as e:
    raise e