"""
Minimal example: tracing the DSL ↔ MLIR boundary in CuTe DSL.

Four progressively deeper examples:
  A) Normal DSL code — just write Python, see the MLIR it generates.
  B) @dsl_user_op — manually emit MLIR arith ops from DSL types.
  C) llvm.inline_asm — drop all the way to raw PTX from Python.
  D) Dynamic tracing — instrument the DSL dispatch chain to watch every
     step of what happens when Python executes `a[i] + 1` during tracing.

Run with:
    CUTE_DSL_NO_CACHE=1 python experiments/dsl_mlir_boundary.py

    # To also see generated MLIR and PTX:
    CUTE_DSL_PRINT_IR=1 CUTE_DSL_KEEP_PTX=1 \
    CUTE_DSL_DUMP_DIR=/tmp/cute_debug CUTE_DSL_NO_CACHE=1 \
    python experiments/dsl_mlir_boundary.py
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32, dsl_user_op
from cutlass.cute.runtime import from_dlpack
from cutlass._mlir.dialects import arith as arith_dialect
from cutlass._mlir.dialects import llvm

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


# ---------------------------------------------------------------------------
# B) @dsl_user_op — manually cross the DSL/MLIR boundary.
#
# Inside a @dsl_user_op you work with DSL types (Int32, Float32) whose
# arithmetic is still traced.  But you can call .ir_value() to get the raw
# MLIR ir.Value, then call MLIR dialect functions directly.
#
# Think of it as:
#   DSL layer:  Int32(x) + Int32(y)    →  emits arith.addi under the hood
#   IR  layer:  arith_dialect.addi(...)  →  you emit arith.addi yourself
# ---------------------------------------------------------------------------

@dsl_user_op
def manual_add_i32(x: Int32, y: Int32, *, loc=None, ip=None) -> Int32:
    """
    Same as `x + y` but we do it by hand at the MLIR level.

    Steps:
      1. x.ir_value()  — unwrap DSL Int32 → raw MLIR ir.Value (type i32)
      2. y.ir_value()  — same
      3. arith_dialect.addi(a, b) — emit an `arith.addi` MLIR operation
      4. Int32(result) — wrap the raw ir.Value back into a DSL Int32
    """
    # --- cross the boundary: DSL → IR ---
    x_ir = x.ir_value(loc=loc, ip=ip)      # ir.Value of type i32
    y_ir = y.ir_value(loc=loc, ip=ip)      # ir.Value of type i32

    # --- pure MLIR: emit an arith.addi operation ---
    sum_ir = arith_dialect.addi(x_ir, y_ir, loc=loc, ip=ip)  # ir.Value

    # --- cross back: IR → DSL ---
    return Int32(sum_ir)


@cute.kernel
def manual_add_kernel(a: cute.Tensor, b: cute.Tensor):
    """b[i] = a[i] + 1, but the +1 goes through our manual_add_i32."""
    for i in range(a.shape[0]):
        # a[i] returns a DSL Float32 (traced, not a real float).
        # We cast to Int32, use our manual adder, cast back.
        val_f = a[i]
        val_i = Int32(val_f)
        one = Int32(1)
        result_i = manual_add_i32(val_i, one)
        b[i] = Float32(result_i)


@cute.jit
def manual_add_host(a: cute.Tensor, b: cute.Tensor):
    manual_add_kernel(a, b).launch(grid=(1, 1, 1), block=(1, 1, 1))


# ---------------------------------------------------------------------------
# C) llvm.inline_asm — drop to raw PTX.
#
# When even the MLIR arith dialect isn't enough (e.g. you need a PTX
# instruction with no MLIR counterpart), you use llvm.inline_asm.
#
# This requires:
#   - Every operand as a raw ir.Value  (no DSL wrappers)
#   - A PTX asm string with $0, $1, ... operand placeholders
#   - A constraint string telling LLVM which register class per operand
#   - AsmDialect.AD_ATT  (PTX uses AT&T-style $N substitution)
# ---------------------------------------------------------------------------

@dsl_user_op
def ptx_add_i32(x: Int32, y: Int32, *, loc=None, ip=None) -> Int32:
    """
    x + y via raw PTX:  add.s32 %result, %x, %y
    """
    x_ir = x.ir_value(loc=loc, ip=ip)
    y_ir = y.ir_value(loc=loc, ip=ip)

    # llvm.inline_asm signature:
    #   res       — MLIR type of the result (i32), or None for void
    #   operands  — list of raw ir.Value
    #   asm_str   — the PTX instruction with $0..$N placeholders
    #               $0 = first *output*, $1 = first input, $2 = second input
    #   constraints — "=r,r,r" means:
    #                 =r : output in a 32-bit register
    #                  r : input  in a 32-bit register  (x)
    #                  r : input  in a 32-bit register  (y)
    result_ir = llvm.inline_asm(
        Int32.mlir_type,                    # result type: i32
        [x_ir, y_ir],                       # operands (raw ir.Values)
        "add.s32 $0, $1, $2;",             # PTX assembly
        "=r,r,r",                           # constraints
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, # PTX uses AT&T-style $N refs
        loc=loc,
        ip=ip,
    )
    return Int32(result_ir)


@cute.kernel
def ptx_add_kernel(a: cute.Tensor, b: cute.Tensor):
    """b[i] = a[i] + 1  via raw PTX add.s32."""
    for i in range(a.shape[0]):
        val_i = Int32(a[i])
        result_i = ptx_add_i32(val_i, Int32(1))
        b[i] = Float32(result_i)


@cute.jit
def ptx_add_host(a: cute.Tensor, b: cute.Tensor):
    ptx_add_kernel(a, b).launch(grid=(1, 1, 1), block=(1, 1, 1))


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
        _log(f"→ AddFOp.__init__(lhs={lhs.type}, rhs={rhs.type})")
        _log(f"    will call: self.build_generic(operands=[lhs,rhs], ...)")
        _log(f"    build_generic is C++ (pybind11) → mlirOperationCreate()")
        originals["addf_init"](self, lhs, rhs, **kwargs)
        _log(f"  ← created MLIR op 'arith.addf', result={self.result}")
    _arith_ops_gen.AddFOp.__init__ = traced_addf_init

    # --- AddIOp.__init__ (arith.addi constructor) ---
    originals["addi_init"] = _arith_ops_gen.AddIOp.__init__

    def traced_addi_init(self, lhs, rhs, **kwargs):
        _log(f"→ AddIOp.__init__(lhs={lhs.type}, rhs={rhs.type})")
        _log(f"    will call: self.build_generic(operands=[lhs,rhs], ...)")
        originals["addi_init"](self, lhs, rhs, **kwargs)
        _log(f"  ← created MLIR op 'arith.addi', result={self.result}")
    _arith_ops_gen.AddIOp.__init__ = traced_addi_init

    # --- ConstantOp.__init__ (arith.constant — materializes literals) ---
    originals["const_init"] = _arith_ops_gen.ConstantOp.__init__

    def traced_const_init(self, value, *, loc=None, ip=None):
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
        b[i] = a[i] + 1


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

    print("=" * 70)
    print("A) Pure DSL — Python traced into MLIR automatically")
    print("=" * 70)
    compiled_a = cute.compile(vector_add, a_cute, b_cute, options="--enable-tvm-ffi")
    run_example("vector_add", compiled_a)

    print("=" * 70)
    print("B) @dsl_user_op — manual arith.addi at the MLIR level")
    print("=" * 70)
    compiled_b = cute.compile(manual_add_host, a_cute, b_cute, options="--enable-tvm-ffi")
    run_example("manual_add", compiled_b)

    print("=" * 70)
    print("C) llvm.inline_asm — raw PTX add.s32 instruction")
    print("=" * 70)
    compiled_c = cute.compile(ptx_add_host, a_cute, b_cute, options="--enable-tvm-ffi")
    run_example("ptx_add", compiled_c)

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
    print("=" * 70)
    print("D) Instrumented DSL dispatch chain for `a[i] + 1`")
    print("=" * 70)
    print()
    print("  Patching: ArithValue.__add__, AddFOp, AddIOp, ConstantOp")
    print("  These are the Python functions that sit between your code")
    print("  and the MLIR-C library.  Watch the indented trace below.")
    print()

    restore = instrument_dispatch()
    compiled_d = cute.compile(traced_host, a_cute, b_cute, options="--enable-tvm-ffi")
    restore()

    print()
    run_example("traced_dispatch", compiled_d)

    # Print the summary.
    print("=" * 70)
    print("Summary: the full dispatch chain for `a[i] + 1`")
    print("=" * 70)
    print("""
  Your code              b[i] = a[i] + 1

  cute.memref.load       a[i] → ir.Value (f32)
                         The result is auto-wrapped as ArithValue because
                         @register_value_caster maps f32 → ArithValue.

  Python + operator      ArithValue.__add__(self=<f32 val>, other=1)
                         Decorated with:
                           @dsl_user_op     — injects MLIR loc/ip
                           @_binary_op      — promotes int 1 → ArithValue(f32)
                                              via arith.constant(1.0 : f32)

  ArithValue.__add__     Checks self.is_float → calls arith.addf(lhs, rhs)
    body

  arith.addf()           → AddFOp(lhs, rhs)
  AddFOp.__init__        → self.build_generic(operands=[lhs,rhs], results=[f32])
  OpView.build_generic   → C++ (pybind11) → mlirOperationCreate()
                         Creates the 'arith.addf' operation in the MLIR module.
                         Returns ir.Value = the SSA result (%N : f32).

  cute.memref.store      b[i] = %N  → stores the result.

  Key insight: nothing executes on the GPU during this process.  Every
  Python operation *builds an MLIR IR graph*.  The GPU only runs later,
  after the pass pipeline lowers MLIR → PTX → CUBIN.
""")


if __name__ == "__main__":
    main()
