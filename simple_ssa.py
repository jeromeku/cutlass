import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import numpy as np

def override_compiler(dump_ptx: bool = False, print_before: bool = False, print_after: bool = True, dump_dir="cute_pipeline"):
    from cutlass.base_dsl import compiler as _cute_compiler    
    CompilationError = _cute_compiler.CompilationError
    
    assert print_before ^ print_after, "Only one of print_before or print_after can be True"
    def _compile(
        self: _cute_compiler.Compiler,
        module,
        pipeline: str,
        cuda_toolkit: str = "",
        arch: str = "",
        enable_verifier=False,
    ):
        """Compiles the module by invoking the pipeline."""
        try:
            ctx = module.context
            if dump_ptx:
                pipeline = pipeline.replace("cubin-format=bin", "cubin-format=isa")
            print(f"OVERRIDE_PIPELINE: {pipeline}")
            
            pm = self.passmanager.PassManager.parse(pipeline)
            ctx.enable_multithreading(False)
            ctx.emit_error_diagnostics = True
            pm.enable_verifier(enable_verifier)

            # Print before/after every pass, include locations, and dump each pass’s IR to a directory.
            pm.enable_ir_printing(
                print_before_all=print_before,
                print_after_all=print_after,
                print_module_scope=False,
                print_after_change=False,
                print_after_failure=True,
                enable_debug_info=True,
                tree_printing_dir_path=dump_dir
            )
            pm.run(module.operation)

        except Exception as e:
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
    
    _original_compile = _cute_compiler.Compiler.compile
    _cute_compiler.Compiler.compile = _compile


@cute.jit
def load_and_store(res: cute.Tensor, a: cute.Tensor, b: cute.Tensor):
    """
    Load data from memory and store the result to memory.

    :param res: The destination tensor to store the result.
    :param a: The source tensor to be loaded.
    :param b: The source tensor to be loaded.
    """
    a_vec = a.load()
    print(f"a_vec: {a_vec}")      # prints `a_vec: vector<12xf32> o (3, 4)`
    b_vec = b.load()
    print(f"b_vec: {b_vec}")      # prints `b_vec: vector<12xf32> o (3, 4)`
    res.store(a_vec + b_vec)
    cute.print_tensor(res)

a = np.ones(12).reshape((3, 4)).astype(np.float32)
b = np.ones(12).reshape((3, 4)).astype(np.float32)
c = np.zeros(12).reshape((3, 4)).astype(np.float32)
override_compiler(dump_ptx=False, print_before=True, print_after=False, dump_dir="jit_dump_after")
load_and_store(from_dlpack(c), from_dlpack(a), from_dlpack(b))