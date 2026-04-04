from utils.logging import patch_cutlass_env
import logging
patch_cutlass_env(log_to_console=True, log_level=logging.WARNING, print_after_preprocessor=False)

import cutlass.cute as cute
import cutlass
from cutlass.cutlass_dsl.cuda_jit_executor import CudaDialectJitCompiledFunction
from cutlass.base_dsl.compiler import CompileCallable, KeepPTX, DumpDir, GenerateLineInfo

@cute.jit
def main(const_var: cutlass.Constexpr, dynamic_var: cutlass.Int32):
    # ✅ This branch is Python branch, evaluated at compile time.
    if cutlass.const_expr(const_var):
        print("Const branch python")
        cute.printf("Const branch\\n")
    else:
        cute.printf("Const else\\n")

    # ✅ This branch is dynamic branch, emitted IR branch.
    if dynamic_var == 10:
        print("Dynamic True python")
        cute.printf("Dynamic True\\n")
    else:
        print("Dynamic False python")
        cute.printf("Dynamic False\\n")

const_var = True
dynamic_var = 11
compiler: CompileCallable = cute.compile

kernel: CudaDialectJitCompiledFunction = cute.compile(main, const_var, dynamic_var)
kernel(10)
kernel(11)
