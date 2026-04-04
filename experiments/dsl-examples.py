import torch
# from utils.logging import patch_cutlass_env
# from datetime import datetime

# ts = datetime.now().strftime("%Y%m%d_%H%M%S")
# patch_cutlass_env(
#     log_to_console=True,
#     keep_ir=True,
#     keep_ptx=True,
#     dumpdir="dsl_dump",
#     print_after_preprocessor=True,
#     preprocessed_ast_path=f"dsl_ast_${ts}"    
# )

# import cutlass
import cutlass.cute as cute
# from cutlass import Int32, Float32, dsl_user_op
from cutlass.cute.runtime import from_dlpack
# from cutlass._mlir.dialects import arith as arith_dialect
# from cutlass._mlir.dialects import llvm

# ---------------------------------------------------------------------------
# A) Pure DSL — write Python, CuTe traces it into MLIR automatically.
#
# Every Python operation inside @cute.kernel is *traced*, not executed.
# `a[i]` doesn't load a float — it emits an MLIR "load" op.
# `+ 1`  doesn't add numbers  — it emits an `arith.addf` op.
# The result is an MLIR module that the compiler lowers to PTX.
# ---------------------------------------------------------------------------


@cute.kernel
def vector_add_kernel(a: cute.Tensor, b: cute.Tensor):
    """b[i] = a[i] + 1  — one thread does all work (toy example)."""
    for i in range(a.shape[0]):
        b[i] = a[i] + 1


@cute.jit
def vector_add(a: cute.Tensor, b: cute.Tensor):
    vector_add_kernel(a, b).launch(grid=(1, 1, 1), block=(1, 1, 1))


# # ---------------------------------------------------------------------------
# # B) @dsl_user_op — manually cross the DSL/MLIR boundary.
# #
# # Inside a @dsl_user_op you work with DSL types (Int32, Float32) whose
# # arithmetic is still traced.  But you can call .ir_value() to get the raw
# # MLIR ir.Value, then call MLIR dialect functions directly.
# #
# # Think of it as:
# #   DSL layer:  Int32(x) + Int32(y)    →  emits arith.addi under the hood
# #   IR  layer:  arith_dialect.addi(...)  →  you emit arith.addi yourself
# # ---------------------------------------------------------------------------


# @dsl_user_op
# def manual_add_i32(x: Int32, y: Int32, *, loc=None, ip=None) -> Int32:
#     """
#     Same as `x + y` but we do it by hand at the MLIR level.

#     Steps:
#       1. x.ir_value()  — unwrap DSL Int32 → raw MLIR ir.Value (type i32)
#       2. y.ir_value()  — same
#       3. arith_dialect.addi(a, b) — emit an `arith.addi` MLIR operation
#       4. Int32(result) — wrap the raw ir.Value back into a DSL Int32
#     """
#     # --- cross the boundary: DSL → IR ---
#     x_ir = x.ir_value(loc=loc, ip=ip)  # ir.Value of type i32
#     y_ir = y.ir_value(loc=loc, ip=ip)  # ir.Value of type i32

#     # --- pure MLIR: emit an arith.addi operation ---
#     sum_ir = arith_dialect.addi(x_ir, y_ir, loc=loc, ip=ip)  # ir.Value

#     # --- cross back: IR → DSL ---
#     return Int32(sum_ir)


# @cute.kernel
# def manual_add_kernel(a: cute.Tensor, b: cute.Tensor):
#     """b[i] = a[i] + 1, but the +1 goes through our manual_add_i32."""
#     for i in range(a.shape[0]):
#         # a[i] returns a DSL Float32 (traced, not a real float).
#         # We cast to Int32, use our manual adder, cast back.
#         val_f = a[i]
#         val_i = Int32(val_f)
#         one = Int32(1)
#         result_i = manual_add_i32(val_i, one)
#         b[i] = Float32(result_i)


# @cute.jit
# def manual_add_host(a: cute.Tensor, b: cute.Tensor):
#     manual_add_kernel(a, b).launch(grid=(1, 1, 1), block=(1, 1, 1))


# # ---------------------------------------------------------------------------
# # C) llvm.inline_asm — drop to raw PTX.
# #
# # When even the MLIR arith dialect isn't enough (e.g. you need a PTX
# # instruction with no MLIR counterpart), you use llvm.inline_asm.
# #
# # This requires:
# #   - Every operand as a raw ir.Value  (no DSL wrappers)
# #   - A PTX asm string with $0, $1, ... operand placeholders
# #   - A constraint string telling LLVM which register class per operand
# #   - AsmDialect.AD_ATT  (PTX uses AT&T-style $N substitution)
# # ---------------------------------------------------------------------------


# @dsl_user_op
# def ptx_add_i32(x: Int32, y: Int32, *, loc=None, ip=None) -> Int32:
#     """
#     x + y via raw PTX:  add.s32 %result, %x, %y
#     """
#     x_ir = x.ir_value(loc=loc, ip=ip)
#     y_ir = y.ir_value(loc=loc, ip=ip)

#     # llvm.inline_asm signature:
#     #   res       — MLIR type of the result (i32), or None for void
#     #   operands  — list of raw ir.Value
#     #   asm_str   — the PTX instruction with $0..$N placeholders
#     #               $0 = first *output*, $1 = first input, $2 = second input
#     #   constraints — "=r,r,r" means:
#     #                 =r : output in a 32-bit register
#     #                  r : input  in a 32-bit register  (x)
#     #                  r : input  in a 32-bit register  (y)
#     result_ir = llvm.inline_asm(
#         Int32.mlir_type,  # result type: i32
#         [x_ir, y_ir],  # operands (raw ir.Values)
#         "add.s32 $0, $1, $2;",  # PTX assembly
#         "=r,r,r",  # constraints
#         has_side_effects=False,
#         is_align_stack=False,
#         asm_dialect=llvm.AsmDialect.AD_ATT,  # PTX uses AT&T-style $N refs
#         loc=loc,
#         ip=ip,
#     )
#     return Int32(result_ir)


# @cute.kernel
# def ptx_add_kernel(a: cute.Tensor, b: cute.Tensor):
#     """b[i] = a[i] + 1  via raw PTX add.s32."""
#     for i in range(a.shape[0]):
#         val_i = Int32(a[i])
#         result_i = ptx_add_i32(val_i, Int32(1))
#         b[i] = Float32(result_i)


# @cute.jit
# def ptx_add_host(a: cute.Tensor, b: cute.Tensor):
#     ptx_add_kernel(a, b).launch(grid=(1, 1, 1), block=(1, 1, 1))



def run_example(name, host_fn, compiled_fn):
    a = torch.arange(8, dtype=torch.float32, device="cuda")
    b = torch.zeros(8, dtype=torch.float32, device="cuda")
    compiled_fn(a, b)
    torch.cuda.synchronize()
    print(f"  a = {a.tolist()}")
    print(f"  b = {b.tolist()}")
    expected = (a + 1).tolist()
    assert b.tolist() == expected, f"MISMATCH: {b.tolist()} != {expected}"
    print(f"  PASS\n")


def main():
    a_torch = torch.arange(8, dtype=torch.float32, device="cuda")
    b_torch = torch.zeros(8, dtype=torch.float32, device="cuda")
    a_cute = from_dlpack(a_torch, enable_tvm_ffi=True).mark_layout_dynamic()
    b_cute = from_dlpack(b_torch, enable_tvm_ffi=True).mark_layout_dynamic()

    # print("=" * 70)
    # print("A) Pure DSL — Python traced into MLIR automatically")
    # print("=" * 70)
    from cutlass.cutlass_dsl import CudaDialectJitCompiledFunction

    compiled_a: CudaDialectJitCompiledFunction = cute.compile(
        vector_add, a_cute, b_cute, options="--keep-ptx --enable-tvm-ffi"
    )
    print(compiled_a.__ptx__)    
if __name__ == "__main__":
    main()
