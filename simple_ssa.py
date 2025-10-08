import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import numpy as np

def override_pass_pipeline():
    from cutlass.base_dsl import compiler as _compiler
    _orig_compile = _compiler.Compiler.compile
    print("OVERRIDING_COMPILER")
    def _traced_compile(self, module, pipeline, cuda_toolkit="", arch="", enable_verifier=False):
        pm = self.passmanager.PassManager.parse(pipeline)
        pm.enable_ir_printing(
            print_before_all=False,
            print_after_all=True,
            print_module_scope=False,
            print_after_change=True,
            print_after_failure=True,
            tree_printing_dir_path="mlir_pass_dumps"  # optional: write pass tree + IR to files
        )
        pm.enable_verifier(enable_verifier)
        pm.run(module.operation)

    _compiler.Compiler.compile = _traced_compile


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
load_and_store(from_dlpack(c), from_dlpack(a), from_dlpack(b))