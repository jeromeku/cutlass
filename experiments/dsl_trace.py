import torch
from utils.logging import patch_cutlass_env
patch_cutlass_env(log_to_console=True, keep_ir=True, dumpdir="dsl-trace-dump", print_after_preprocessor=True, preprocessed_ast_path="dsl-trace-ast.py")

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32, dsl_user_op
from cutlass.cute.runtime import from_dlpack
from cutlass._mlir.dialects import arith as arith_dialect
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import CudaDialectJitCompiledFunction


# ---------------------------------------------------------------------------
# D) Dynamic tracing — instrument the DSL dispatch chain.
#
# When Python executes `a[i] + 1` inside a @cute.kernel during tracing,
# the runtime call chain is:
#
#   1. cute.memref.load   →  a[i] returns an ir.Value (f32), auto-wrapped
#                             as ArithValue via @register_value_caster
#   2. Python `+` operator → ArithValue.__add__(self, 1)
#      2a. @_dispatch_to_rhs_r_op  — checks if rhs wants to handle it
#      2b. @_binary_op decorator   — promotes `1` (int) → ArithValue(f32),
#                                    type-promotes both operands
#      2c. ArithValue.__add__ body — checks is_float → calls arith.addf()
#   3. arith.addf()        → AddFOp.__init__(lhs, rhs)
#   4. AddFOp.__init__     → self.build_generic(operands=[lhs,rhs], ...)
#   5. OpView.build_generic  ← THIS IS THE C++ BOUNDARY (pybind11)
#                              calls mlirOperationCreate() in MLIR-C
#   6. Returns ir.Value representing the new SSA result
#
# Steps 1-4 are pure Python.  Step 5 is C++.  We instrument steps 2c and 3
# by monkey-patching ArithValue.__add__ and AddFOp.__init__ to print what
# they're doing.  This lets you watch the dispatch live.
#
# NOTE: ArithValue is a subclass of ir.Value registered via
# @ir.register_value_caster.  When any MLIR op returns an f32 or i32
# result, the pybind11 bindings automatically wrap it as ArithValue.
# This is WHY `a[i] + 1` triggers ArithValue.__add__ — the load result
# is already an ArithValue, so Python dispatches `+` to it.
# ---------------------------------------------------------------------------

def instrument_dispatch():
    """Monkey-patch DSL dispatch functions to trace the `+` chain."""
    from cutlass.base_dsl._mlir_helpers import arith as _arith_helper
    from cutlass._mlir.dialects import _arith_ops_gen

    originals = {}
    indent = [0]

    def _log(msg):
        print(f"  {'  ' * indent[0]}{msg}")

    # --- ArithValue.__add__ (the operator overload) ---
    # This is decorated with @dsl_user_op, @_dispatch_to_rhs_r_op, @_binary_op.
    # The outermost callable is the @dsl_user_op wrapper.  We wrap THAT.
    originals["av_add"] = _arith_helper.ArithValue.__add__

    def traced_av_add(self, other, **kwargs):
        breakpoint()
        is_float = getattr(self, "is_float", "?")
        signed = getattr(self, "signed", "?")
        _log(f"→ ArithValue.__add__(self, other)")
        _log(f"    self  : type={self.type}, is_float={is_float}, signed={signed}")
        try:
            other_type = other.type
        except AttributeError:
            other_type = type(other).__name__
        _log(f"    other : type={other_type}, value={other}")
        _log(f"    decorators will: promote types, check float vs int, emit op")
        indent[0] += 1
        result = originals["av_add"](self, other, **kwargs)
        indent[0] -= 1
        _log(f"  ← result: type={result.type}")
        return result
    _arith_helper.ArithValue.__add__ = traced_av_add

    # --- AddFOp.__init__ (arith.addf constructor) ---
    originals["addf_init"] = _arith_ops_gen.AddFOp.__init__

    def traced_addf_init(self, lhs, rhs, **kwargs):
        breakpoint()
        _log(f"→ AddFOp.__init__(lhs={lhs.type}, rhs={rhs.type})")
        _log(f"    will call: self.build_generic(operands=[lhs,rhs], ...)")
        _log(f"    build_generic is C++ (pybind11) → mlirOperationCreate()")
        originals["addf_init"](self, lhs, rhs, **kwargs)
        _log(f"  ← created MLIR op 'arith.addf', result={self.result}")
    _arith_ops_gen.AddFOp.__init__ = traced_addf_init

    # --- AddIOp.__init__ (arith.addi constructor) ---
    originals["addi_init"] = _arith_ops_gen.AddIOp.__init__

    def traced_addi_init(self, lhs, rhs, **kwargs):
        breakpoint()
        _log(f"→ AddIOp.__init__(lhs={lhs.type}, rhs={rhs.type})")
        _log(f"    will call: self.build_generic(operands=[lhs,rhs], ...)")
        originals["addi_init"](self, lhs, rhs, **kwargs)
        _log(f"  ← created MLIR op 'arith.addi', result={self.result}")
    _arith_ops_gen.AddIOp.__init__ = traced_addi_init

    # --- ConstantOp.__init__ (arith.constant — materializes literals) ---
    originals["const_init"] = _arith_ops_gen.ConstantOp.__init__

    def traced_const_init(self, value, *, loc=None, ip=None):
        breakpoint()
        _log(f"→ ConstantOp.__init__(value={value})")
        originals["const_init"](self, value, loc=loc, ip=ip)
        _log(f"  ← created 'arith.constant' = {self.result} (type={self.result.type})")
    _arith_ops_gen.ConstantOp.__init__ = traced_const_init

    def restore():
        _arith_helper.ArithValue.__add__ = originals["av_add"]
        _arith_ops_gen.AddFOp.__init__ = originals["addf_init"]
        _arith_ops_gen.AddIOp.__init__ = originals["addi_init"]
        _arith_ops_gen.ConstantOp.__init__ = originals["const_init"]

    return restore


@cute.kernel
def traced_kernel(a: cute.Tensor, b: cute.Tensor):
    """Same b[i] = a[i] + 1, but we'll trace every MLIR op it emits."""
    for i in range(a.shape[0]):
        breakpoint()
        a_val = a[i]
        res = a_val + 1
        b[i] = res

@cute.jit
def traced_host(a: cute.Tensor, b: cute.Tensor):
    traced_kernel(a, b).launch(grid=(1, 1, 1), block=(1, 1, 1))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_example(name, compiled_fn):
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
    # print("C) llvm.inline_asm — raw PTX add.s32 instruction")
    # print("=" * 70)
    # compiled_c = cute.compile(ptx_add_host, a_cute, b_cute, options="--enable-tvm-ffi")
    # run_example("ptx_add", compiled_c)

    # -------------------------------------------------------------------
    # D) Instrumented dispatch chain.
    #
    # We monkey-patch the 4 key functions in the DSL→MLIR dispatch chain
    # so you can watch exactly what happens when the tracer evaluates
    # `a[i] + 1`.  The output shows:
    #
    #   → ArithValue.__add__(self, other)    ← Python `+` dispatches here
    #       self: f32 ir.Value from cute.memref.load
    #       other: Python int 1
    #     → ConstantOp.__init__(1.0)         ← `1` promoted to f32 constant
    #     → AddFOp.__init__(lhs, rhs)        ← emit arith.addf
    #       → build_generic → C++            ← MLIR-C boundary
    #     ← result: new f32 ir.Value
    # -------------------------------------------------------------------
    # print("=" * 70)
    # print("D) Instrumented DSL dispatch chain for `a[i] + 1`")
    # print("=" * 70)
    # print()
    # print("  Patching: ArithValue.__add__, AddFOp, AddIOp, ConstantOp")
    # print("  These are the Python functions that sit between your code")
    # print("  and the MLIR-C library.  Watch the indented trace below.")
    # print()

    restore = instrument_dispatch()
    compiled_d: CudaDialectJitCompiledFunction = cute.compile(traced_host, a_cute, b_cute, options="--keep-ptx")
    print(compiled_d.__mlir__)    
    # restore()

    # print()
    # run_example("traced_dispatch", compiled_d)

    # Print the summary.
#     print("=" * 70)
#     print("Summary: the full dispatch chain for `a[i] + 1`")
#     print("=" * 70)
#     print("""
#   Your code              b[i] = a[i] + 1

#   cute.memref.load       a[i] → ir.Value (f32)
#                          The result is auto-wrapped as ArithValue because
#                          @register_value_caster maps f32 → ArithValue.

#   Python + operator      ArithValue.__add__(self=<f32 val>, other=1)
#                          Decorated with:
#                            @dsl_user_op     — injects MLIR loc/ip
#                            @_binary_op      — promotes int 1 → ArithValue(f32)
#                                               via arith.constant(1.0 : f32)

#   ArithValue.__add__     Checks self.is_float → calls arith.addf(lhs, rhs)
#     body

#   arith.addf()           → AddFOp(lhs, rhs)
#   AddFOp.__init__        → self.build_generic(operands=[lhs,rhs], results=[f32])
#   OpView.build_generic   → C++ (pybind11) → mlirOperationCreate()
#                          Creates the 'arith.addf' operation in the MLIR module.
#                          Returns ir.Value = the SSA result (%N : f32).

#   cute.memref.store      b[i] = %N  → stores the result.

#   Key insight: nothing executes on the GPU during this process.  Every
#   Python operation *builds an MLIR IR graph*.  The GPU only runs later,
#   after the pass pipeline lowers MLIR → PTX → CUBIN.
# """)


if __name__ == "__main__":
    main()
