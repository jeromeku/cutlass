# CuteDSL Compilation Process

This document provides a detailed trace of the CuteDSL compilation process through three stages: **Pre-Staging (AST Rewrite)**, **Meta-Staging (Python Interpreter + Tracing)**, and **Object-Staging (MLIR Compilation)**. We'll trace the `elementwise_apply.py` example to make this concrete.

## Overview

CuteDSL uses a **hybrid approach** combining AST-based transformation with tracing-based execution:

```
Input Python Code
       ↓
[Stage 1] Pre-Staging - Python AST Rewrite
       ↓
Intermediate Python Code (with callbacks)
       ↓
[Stage 2] Meta-Staging - Python Interpreter Execution + Tracing
       ↓
MLIR IR
       ↓
[Stage 3] Object-Staging - MLIR Compilation
       ↓
GPU Binary
```

---

## Example Code: elementwise_apply.py

We'll trace this simplified kernel from [elementwise_apply.py:79-154](examples/python/CuTeDSL/ampere/elementwise_apply.py#L79-L154):

```python
@cute.kernel
def elementwise_apply_kernel(
    op: cutlass.Constexpr,
    mInputs: List[cute.Tensor],
    mC: cute.Tensor,
    cC: cute.Tensor,
    shape: cute.Shape,
    tv_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    # Slice to local tile of thread block
    blk_crd = ((None, None), (bidx, bidy))
    gInputs = [t[blk_crd] for t in mInputs]
    gC = mC[blk_crd]
    gCrd = cC[blk_crd]

    # Compose with thread block TV layout
    tidfrgInputs = [cute.composition(t, tv_layout) for t in gInputs]
    tidfrgC = cute.composition(gC, tv_layout)
    tidfrgCrd = cute.composition(gCrd, tv_layout)

    thr_crd = (tidx, cute.repeat_like(None, tidfrgInputs[0][1]))

    # Slice to local tile of thread
    thrInputs = [t[thr_crd] for t in tidfrgInputs]
    thrC = tidfrgC[thr_crd]
    thrCrd = tidfrgCrd[thr_crd]

    # Compute predicate for boundary checks
    frgPred = cute.make_rmem_tensor(thrCrd.shape, cutlass.Boolean)
    for i in cutlass.range_constexpr(cute.size(frgPred)):
        frgPred[i] = cute.elem_less(thrCrd[i], shape)

    # Load data and compute result
    result = op(*[thrInput.load() for thrInput in thrInputs])
    thrC.store(result)
```

The host function that calls this kernel:

```python
@cute.jit
def elementwise_apply(
    op: cutlass.Constexpr,
    inputs,
    result: cute.Tensor,
    stream: cuda.CUstream
):
    # ... tiling setup code ...

    elementwise_apply_kernel(op, mInputs, mC, cC, result.shape, tv_layout).launch(
        grid=cute.product_each(mC.shape[1]),
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
        stream=stream,
    )
```

---

## Stage 1: Pre-Staging (Python AST Rewrite)

**Location**: [cutlass/base_dsl/ast_preprocessor.py](../python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py)

### Purpose
Transform Python AST to insert callbacks that will capture control flow during execution.

### Key Transformations

#### 1.1 Loop Transformation

**Input AST** (from line 141):
```python
for i in cutlass.range_constexpr(cute.size(frgPred)):
    frgPred[i] = cute.elem_less(thrCrd[i], shape)
```

**Detected by** [ast_preprocessor.py:981-992](../python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py#L981-L992):
```python
def visit_For(self, node):
    range_kind, is_builtin_range, has_keyword = self._get_range_kind(node.iter)
    if range_kind == "range_constexpr" or range_kind == None:
        self.generic_visit(node)
        if range_kind == "range_constexpr":
            check_call = self._insert_cf_symbol_check(node.iter.func)
            # Rewrite range_constexpr to range
            node.iter.func = ast.Name(id="range", ctx=ast.Load())
            self._insert_range_value_check(node)
            return [check_call, node]
        return node
```

**Output AST** (simplified):
```python
# Check that range_constexpr is from the DSL module
_dsl_.ast_helpers.cf_symbol_check(cutlass.range_constexpr)

# Convert to normal Python range - will be evaluated by interpreter
for i in range(_dsl_.ast_helpers.range_value_check(cute.size(frgPred))):
    frgPred[i] = cute.elem_less(thrCrd[i], shape)
```

**Why?** `range_constexpr` tells the DSL: "evaluate this loop at compile time using the Python interpreter" (meta-stage), not "emit MLIR loop ops" (object-stage).

#### 1.2 Decorator Removal

**Input**:
```python
@cute.kernel
def elementwise_apply_kernel(...):
    ...
```

**Transformation** [ast_preprocessor.py:1464-1507](../python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py#L1464-L1507):
```python
def remove_dsl_decorator(self, decorator_list):
    """Remove .jit and .kernel decorators"""
    new_decorator_list = []
    decorator_names = ["jit", "kernel"]
    for d in decorator_list:
        is_jit_or_kernel = False
        if isinstance(d, ast.Call):
            if isinstance(d.func, ast.Attribute):
                if d.func.attr in decorator_names:
                    is_jit_or_kernel = True
        # ... skip if it's a DSL decorator
        if not is_jit_or_kernel:
            new_decorator_list.append(d)
    return new_decorator_list
```

**Output**:
```python
def elementwise_apply_kernel(...):  # Decorator removed
    ...
```

**Why?** The decorator has served its purpose (marking the function for DSL processing). Now it's a regular Python function that will be traced.

#### 1.3 Boolean Operator Rewriting

**Input**:
```python
if condition1 and condition2:
    ...
```

**Transformation** [ast_preprocessor.py:837-919](../python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py#L837-L919):
```python
def visit_BoolOp(self, node):
    # Transform "and" to explicit short-circuit evaluation
    if isinstance(node.op, ast.And):
        # if type(lhs) == bool and lhs == False:
        #     return lhs
        # else:
        #     return and_(lhs, rhs)
```

**Output**:
```python
# Short-circuit if Python bool, otherwise call DSL and_ operation
(lhs if (type(lhs) == bool and lhs == False) else cutlass.and_(lhs, condition2))
```

**Why?** Python's native `and`/`or` can't be overloaded. This transformation allows the DSL to handle both Python bools (evaluated immediately) and DSL types (traced).

#### 1.4 Import Injection

**Added by** [ast_preprocessor.py:466-473](../python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py#L466-L473):
```python
import cutlass.base_dsl as _dsl_
```

This provides access to helper functions like `cf_symbol_check`, `range_value_check`, etc.

### Pre-Staging Output

The transformed Python code looks like this (conceptually):

```python
import cutlass.base_dsl as _dsl_

def elementwise_apply_kernel(
    op,
    mInputs,
    mC,
    cC,
    shape,
    tv_layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    blk_crd = ((None, None), (bidx, bidy))
    gInputs = [t[blk_crd] for t in mInputs]
    gC = mC[blk_crd]
    gCrd = cC[blk_crd]

    tidfrgInputs = [cute.composition(t, tv_layout) for t in gInputs]
    tidfrgC = cute.composition(gC, tv_layout)
    tidfrgCrd = cute.composition(gCrd, tv_layout)

    thr_crd = (tidx, cute.repeat_like(None, tidfrgInputs[0][1]))

    thrInputs = [t[thr_crd] for t in tidfrgInputs]
    thrC = tidfrgC[thr_crd]
    thrCrd = tidfrgCrd[thr_crd]

    frgPred = cute.make_rmem_tensor(thrCrd.shape, cutlass.Boolean)

    # Transformed loop:
    _dsl_.ast_helpers.cf_symbol_check(cutlass.range_constexpr)
    for i in range(_dsl_.ast_helpers.range_value_check(cute.size(frgPred))):
        frgPred[i] = cute.elem_less(thrCrd[i], shape)

    result = op(*[thrInput.load() for thrInput in thrInputs])
    thrC.store(result)
```

---

## Stage 2: Meta-Staging (Python Interpreter + Tracing)

**Location**: Various files in [cutlass/cute/](../python/CuTeDSL/cutlass/cute/) and [cutlass/base_dsl/](../python/CuTeDSL/cutlass/base_dsl/)

### Purpose
Execute the transformed Python code with special objects that trace operations and emit MLIR IR.

### Key Concepts

#### 2.1 Constexpr vs Dynamic Values

The DSL distinguishes between:
- **Constexpr**: Values known at compile time, evaluated by Python interpreter
- **Dynamic**: Values that become MLIR IR operations

From the presentation slide 36-39:

```python
@cute.jit
def add_dynamicexpr(b: cutlass.Float32):  # b is dynamic
    a = cutlass.Float32(2.0)               # a is dynamic
    result = a + b                         # Emits MLIR add operation
    print(result)                          # Prints "?" (not yet computed)

@cute.jit
def add_constexpr(b: cutlass.Constexpr):   # b is constexpr
    a = 2.0                                 # a is Python float
    result = a + b                          # Python addition
    print(result)                           # Prints "7.0" (computed now!)

@cute.jit
def add_hybrid(b: cutlass.Constexpr):      # b is constexpr
    a = cutlass.Float32(2.0)                # a is dynamic
    result = a + b                          # Constexpr b is converted to dynamic
    print(result)                           # Prints "?" at meta-stage
    cute.printf(result)                     # Prints "7.0" at object-stage
```

#### 2.2 Tracing Mechanism

When Python executes the transformed code, operations on DSL types emit MLIR:

**Example**: `tidx, _, _ = cute.arch.thread_idx()`

This calls a function that:
1. Emits MLIR IR: `%tidx = gpu.thread_id x`
2. Returns a wrapper object representing this MLIR value
3. Further operations on `tidx` emit more MLIR operations

**Conceptual Python-to-MLIR mapping**:

```python
# Python (meta-stage)                    # MLIR IR (emitted)
tidx, _, _ = cute.arch.thread_idx()   →  %0 = gpu.thread_id x
bidx, bidy, _ = cute.arch.block_idx() →  %1 = gpu.block_id x
                                         %2 = gpu.block_id y
```

#### 2.3 Constexpr Loop Unrolling

The `range_constexpr` loop from Stage 1 is now executed by Python:

**Python execution** (meta-stage):
```python
frgPred = cute.make_rmem_tensor(thrCrd.shape, cutlass.Boolean)  # Creates MLIR tensor type

# Assuming cute.size(frgPred) returns 4:
for i in range(4):  # Python evaluates this
    frgPred[i] = cute.elem_less(thrCrd[i], shape)  # Each iteration emits MLIR
```

**MLIR emitted** (conceptual):
```mlir
%frgPred = cute.make_rmem_tensor ...

// Iteration 0
%cmp0 = cute.elem_less %thrCrd[0], %shape
cute.store %frgPred[0], %cmp0

// Iteration 1
%cmp1 = cute.elem_less %thrCrd[1], %shape
cute.store %frgPred[1], %cmp1

// Iteration 2
%cmp2 = cute.elem_less %thrCrd[2], %shape
cute.store %frgPred[2], %cmp2

// Iteration 3
%cmp3 = cute.elem_less %thrCrd[3], %shape
cute.store %frgPred[3], %cmp3
```

**Key insight**: The loop structure disappears! It's fully unrolled because Python executed it at compile time.

#### 2.4 Polymorphism via Constexpr

From the presentation slides 14-31, the `op: cutlass.Constexpr` parameter enables compile-time polymorphism:

**First call**: `kernel(Epilogue, A, B, C)`
```python
# Python evaluates:
Epilogue().run(matrix)
    ↓
print("identity epilogue")  # Executes at meta-stage
return matrix               # No MLIR emitted (identity)
```

**MLIR emitted**:
```mlir
func.func @cutlass_kernel_Epilogue(...) {
  // ... gemm operations ...
  cute.print("identity epilogue\0A")
  cute.memref.store_vec %result, ...  // Direct store
}
```

**Second call**: `kernel(ReLU, A, B, C)`
```python
# Python evaluates:
ReLU().run(matrix)
    ↓
print("ReLU epilogue")      # Executes at meta-stage
cute.where(matrix > 0.0, matrix, cute.full_like(matrix, 0.0))  # Emits MLIR
```

**MLIR emitted**:
```mlir
func.func @cutlass_kernel_ReLU(...) {
  // ... gemm operations ...
  cute.print("ReLU epilogue\0A")
  %zero = arith.constant 0.0
  %mask = arith.cmpf "ogt", %result, %zero
  %relu = arith.select %mask, %result, %zero
  cute.memref.store_vec %relu, ...
}
```

**Result**: Two different MLIR functions from the same Python source, specialized at compile time!

### Meta-Staging Output

After meta-staging, we have MLIR IR. For the `elementwise_apply_kernel` with ReLU:

```mlir
!memref_gmem_f16 = !cute.memref<f16, gmem, "(128,256):(256,1)">

func.func @cutlass_kernel_ReLU(
    %A: !memref_gmem_f16,
    %B: !memref_gmem_f16,
    %C: !memref_gmem_f16
) {
  // Thread/block indices
  %tidx = gpu.thread_id x
  %bidx = gpu.block_id x
  %bidy = gpu.block_id y

  // Slice to thread block tile
  %gInputs = cute.slice %mInputs[%bidx, %bidy]
  %gC = cute.slice %mC[%bidx, %bidy]
  %gCrd = cute.slice %cC[%bidx, %bidy]

  // Compose with layout
  %tidfrgInputs = cute.composition %gInputs, %tv_layout
  %tidfrgC = cute.composition %gC, %tv_layout
  %tidfrgCrd = cute.composition %gCrd, %tv_layout

  // Slice to thread tile
  %thrInputs = cute.slice %tidfrgInputs[%tidx]
  %thrC = cute.slice %tidfrgC[%tidx]
  %thrCrd = cute.slice %tidfrgCrd[%tidx]

  // Predicate computation (unrolled loop)
  %frgPred = cute.make_rmem_tensor ...
  %cmp0 = cute.elem_less %thrCrd[0], %shape
  cute.store %frgPred[0], %cmp0
  %cmp1 = cute.elem_less %thrCrd[1], %shape
  cute.store %frgPred[1], %cmp1
  // ... (more unrolled iterations)

  // Load and compute
  %input0 = cute.load %thrInputs[0]
  %input1 = cute.load %thrInputs[1]
  %result = arith.addf %input0, %input1  // or whatever op was

  // ReLU epilogue (emitted because op=ReLU)
  cute.print("ReLU epilogue\0A")
  %zero = arith.constant 0.0 : f16
  %mask = arith.cmpf "ogt", %result, %zero
  %relu_result = arith.select %mask, %result, %zero

  // Store
  cute.store %thrC, %relu_result

  return
}
```

---

## Stage 3: Object-Staging (MLIR Compilation)

**Location**: MLIR compiler infrastructure + CUTLASS-specific dialects

### Purpose
Lower high-level MLIR dialects (CuTe dialect, SCF, Arith) to LLVM IR and then to GPU binary (PTX/SASS).

### Key Transformations

#### 3.1 Dialect Lowering Pipeline

The MLIR goes through several lowering passes:

**High-level CuTe Dialect** → **Mid-level Dialects** → **Low-level LLVM Dialect** → **PTX/SASS**

```
CuTe Dialect                      GPU Dialect                  LLVM Dialect
-----------                       -----------                  ------------
cute.make_rmem_tensor        →    gpu.alloc                →  llvm.alloca
cute.load                    →    memref.load              →  llvm.load
cute.store                   →    memref.store             →  llvm.store
cute.elem_less               →    arith.cmpi "slt"         →  llvm.icmp slt
arith.select                 →    (stays)                  →  llvm.select
gpu.thread_id                →    nvvm.read.ptx.sreg.tid.x →  mov.u32 %tid, %tid.x
```

#### 3.2 Example Lowering: Predicate Check

**High-level MLIR** (from meta-stage):
```mlir
%frgPred = cute.make_rmem_tensor <4xi1>
%cmp0 = cute.elem_less %thrCrd[0], %shape
cute.store %frgPred[0], %cmp0
```

**After CuTe lowering**:
```mlir
%frgPred = memref.alloca() : memref<4xi1>
%coord0 = memref.load %thrCrd[0]
%shape_val = memref.load %shape
%cmp0 = arith.cmpi "slt", %coord0, %shape_val : i32
memref.store %cmp0, %frgPred[0]
```

**After LLVM lowering**:
```llvm
%frgPred = alloca [4 x i1]
%coord0 = load i32, ptr %thrCrd[0]
%shape_val = load i32, ptr %shape
%cmp0 = icmp slt i32 %coord0, %shape_val
store i1 %cmp0, ptr %frgPred[0]
```

**Final PTX**:
```ptx
.reg .pred %p<4>;
ld.param.u32 %r0, [thrCrd];
ld.param.u32 %r1, [shape];
setp.lt.u32 %p0, %r0, %r1;
```

#### 3.3 Optimization Passes

MLIR optimization passes run between lowerings:

1. **Inlining**: Small functions are inlined
2. **Constant folding**: Compile-time constants are evaluated
3. **Dead code elimination**: Unused operations removed
4. **Loop optimizations**: (if any loops remain after unrolling)
5. **Memory coalescing**: Optimize memory access patterns
6. **Register allocation**: Assign MLIR SSA values to GPU registers

#### 3.4 Thread Block Launch Configuration

The `.launch()` call from the host code:

```python
elementwise_apply_kernel(...).launch(
    grid=cute.product_each(mC.shape[1]),  # e.g., (32, 16)
    block=[cute.size(tv_layout, mode=[0]), 1, 1],  # e.g., (256, 1, 1)
    stream=stream,
)
```

Becomes CUDA launch parameters:
```cpp
elementwise_apply_kernel<<<dim3(32, 16, 1), dim3(256, 1, 1), 0, stream>>>(...);
```

### Object-Staging Output

**Final GPU Binary**: PTX (human-readable) or SASS (machine code):

```ptx
.version 8.0
.target sm_80
.address_size 64

.visible .entry cutlass_kernel_ReLU(
    .param .u64 A,
    .param .u64 B,
    .param .u64 C
) {
    .reg .pred %p<8>;
    .reg .f16 %f<32>;
    .reg .u32 %r<32>;
    .reg .u64 %rd<16>;

    // Thread indices
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ctaid.y;

    // Address calculations
    mad.lo.u64 %rd0, %r1, 256, %r0;  // Compute offset
    ld.param.u64 %rd1, [A];
    add.u64 %rd2, %rd1, %rd0;

    // Load input
    ld.global.f16 %f0, [%rd2];
    ld.global.f16 %f1, [%rd2+2];

    // Compute
    add.f16 %f2, %f0, %f1;

    // ReLU
    setp.gt.f16 %p0, %f2, 0.0;
    selp.f16 %f3, %f2, 0.0, %p0;

    // Store result
    ld.param.u64 %rd3, [C];
    add.u64 %rd4, %rd3, %rd0;
    st.global.f16 [%rd4], %f3;

    ret;
}
```

**Execution**: This binary is loaded by the CUDA driver and executed on the GPU.

---

## Summary: Data Flow Through All Stages

Let's trace a single operation through all three stages:

### Example: `cute.arch.thread_idx()`

**Stage 1 - Pre-Staging**:
- Input: `tidx, _, _ = cute.arch.thread_idx()`
- Transformation: None (no special AST rewriting needed for function calls)
- Output: `tidx, _, _ = cute.arch.thread_idx()`

**Stage 2 - Meta-Staging**:
- Python interpreter executes: `cute.arch.thread_idx()`
- This function (in CuTe Python library):
  1. Creates MLIR context
  2. Emits: `%0 = gpu.thread_id x`, `%1 = gpu.thread_id y`, `%2 = gpu.thread_id z`
  3. Returns Python wrappers around these MLIR values: `(MLIRValue(%0), MLIRValue(%1), MLIRValue(%2))`
- Python assigns: `tidx = MLIRValue(%0)`, `_ = MLIRValue(%1)`, `_ = MLIRValue(%2)`

**Stage 3 - Object-Staging**:
- MLIR: `%0 = gpu.thread_id x`
- Lowered to: `%0 = nvvm.read.ptx.sreg.tid.x`
- PTX: `mov.u32 %r0, %tid.x`
- Execution: GPU reads hardware register containing thread ID

### Example: Constexpr loop unrolling

**Stage 1 - Pre-Staging**:
```python
# Input
for i in cutlass.range_constexpr(cute.size(frgPred)):
    frgPred[i] = cute.elem_less(thrCrd[i], shape)

# Output
_dsl_.ast_helpers.cf_symbol_check(cutlass.range_constexpr)
for i in range(cute.size(frgPred)):  # range_constexpr → range
    frgPred[i] = cute.elem_less(thrCrd[i], shape)
```

**Stage 2 - Meta-Staging**:
```python
# Python executes:
size = cute.size(frgPred)  # Returns 4 (a Python int)
for i in range(4):  # Python loop, executed 4 times at compile time
    # Each iteration emits MLIR:
    frgPred[i] = cute.elem_less(thrCrd[i], shape)
    # → %cmp_i = arith.cmpi "slt", %thrCrd[i], %shape
    # → memref.store %cmp_i, %frgPred[i]
```

**Stage 3 - Object-Staging**:
```mlir
// No loop in MLIR! Loop was unrolled during meta-stage:
%cmp0 = arith.cmpi "slt", %thrCrd[0], %shape
memref.store %cmp0, %frgPred[0]
%cmp1 = arith.cmpi "slt", %thrCrd[1], %shape
memref.store %cmp1, %frgPred[1]
%cmp2 = arith.cmpi "slt", %thrCrd[2], %shape
memref.store %cmp2, %frgPred[2]
%cmp3 = arith.cmpi "slt", %thrCrd[3], %shape
memref.store %cmp3, %frgPred[3]
```

---

## Key Takeaways

### 1. Hybrid Staging Model

CuteDSL's hybrid approach gives you control over **when** things happen:

- **Stage 1 (Pre)**: Structural transformations (AST surgery)
- **Stage 2 (Meta)**: Metaprogramming via Python interpreter
- **Stage 3 (Object)**: GPU code generation via MLIR

### 2. Constexpr = Meta-Stage

Marking something as `Constexpr` means:
- "Evaluate this using the Python interpreter during compilation"
- Enables: loop unrolling, constant folding, polymorphism
- Result: Specialized code with zero runtime overhead

### 3. Dynamic = Object-Stage

Regular DSL types (like `cutlass.Float32`) mean:
- "Trace this operation and emit MLIR IR"
- Operations become GPU instructions
- Result: Actual computation happens on the GPU

### 4. Best of Both Worlds

The hybrid model combines:
- **AST-based DSL**: Clear program structure, good error messages, explicit control flow
- **Tracing-based DSL**: Natural Python syntax, powerful metaprogramming, automatic specialization

### 5. Performance Without Sacrifice

The three-stage compilation ensures:
- No Python interpreter overhead at runtime (meta-stage executes once)
- Aggressive optimizations (loop unrolling, constant folding, inlining)
- Direct GPU code generation (no intermediate C++ or CUDA code)
- Specialized kernels for each use case (via constexpr polymorphism)

---

## Further Reading

- **AST Preprocessor**: [cutlass/base_dsl/ast_preprocessor.py](../python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py)
- **CuTe Operations**: [cutlass/cute/](../python/CuTeDSL/cutlass/cute/)
- **MLIR Dialects**: [cutlass/_mlir/](../python/CuTeDSL/cutlass/_mlir/)
- **Example Kernels**: [examples/python/CuTeDSL/](examples/python/CuTeDSL/)
- **Presentation**: [ozen.pdf](ozen.pdf) - Slides from LLVM Developers Conference

