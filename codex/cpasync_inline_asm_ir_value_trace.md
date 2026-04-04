# Why `ir_value()` is needed at the `llvm.inline_asm` site

Primary source sites:

- [copy_utils.py](../thirdparty/quack/quack/copy_utils.py#L563)
- [typing.py](../python/CuTeDSL/cutlass/base_dsl/typing.py#L977)
- [core.py](../python/CuTeDSL/cutlass/cute/core.py#L1358)
- [_llvm_ops_gen.py](../python/CuTeDSL/cutlass/_mlir/dialects/_llvm_ops_gen.py#L4084)
- [_ods_common.py](../python/CuTeDSL/cutlass/_mlir/dialects/_ods_common.py#L86)
- [ast_preprocessor.py](../python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py#L1638)
- [typing.py](../python/CuTeDSL/cutlass/base_dsl/typing.py#L1947)

Useful comparison sites:

- [cpasync/helpers.py](../python/CuTeDSL/cutlass/cute/nvgpu/cpasync/helpers.py#L323)
- [utils.py](../thirdparty/quack/quack/utils.py#L64)
- [tensormap_manager.py](../thirdparty/quack/quack/tensormap_manager.py#L104)

External references:

- [MLIR LLVM dialect: `llvm.inline_asm`](https://mlir.llvm.org/docs/Dialects/LLVM/#llvminline_asm-llvminlineasmop)
- [LLVM LangRef: inline asm and constraint strings](https://llvm.org/docs/LangRef.html#inline-asm-constraint-string)

## Big picture

There are three layers involved here:

1. Python/CuteDSL wrapper objects such as `Int32` and `cute.Pointer`
2. MLIR SSA values, represented as `ir.Value`
3. The LLVM dialect op `llvm.inline_asm`, which is a low-level MLIR op

The important boundary is:

- CuteDSL wrappers are convenient Python objects
- `llvm.inline_asm(...)` wants raw MLIR/LLVM operands, not the wrappers

That is why the code explicitly unwraps operands before building the inline asm op.

## The exact site

At [copy_utils.py](../thirdparty/quack/quack/copy_utils.py#L572), the code is:

```python
smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
llvm.inline_asm(
    None,
    [gmem_ptr.llvm_ptr, smem_ptr_i32, Int32(store_bytes).ir_value()],
    "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [$0], [$1], $2;",
    "l,r,r",
    has_side_effects=True,
    is_align_stack=False,
    asm_dialect=llvm.AsmDialect.AD_ATT,
)
```

Read this as:

- no return value
- three operands
- PTX template string uses `$0`, `$1`, `$2`
- constraint string says what operand class each `$i` must use

## Why `ir_value()` is needed

`Int32`, `Float32`, `Uint32`, etc. are not themselves raw MLIR values. They are DSL wrapper objects.

`Numeric.ir_value()` in [typing.py](../python/CuTeDSL/cutlass/base_dsl/typing.py#L977) converts the wrapper to the underlying `ir.Value`. If the object is still a Python constant, it materializes an MLIR constant first via `to(ir.Value)`.

That matters because the LLVM dialect builder eventually does:

```python
operands.extend(_get_op_results_or_values(operands_))
```

in [_llvm_ops_gen.py](../python/CuTeDSL/cutlass/_mlir/dialects/_llvm_ops_gen.py#L4094), and `_get_op_results_or_values` in [_ods_common.py](../python/CuTeDSL/cutlass/_mlir/dialects/_ods_common.py#L107) only accepts:

- MLIR ops / op views
- MLIR `ir.Value`s
- sequences of those

So by the time `llvm.inline_asm` is called, each element inside the operand list must already be an `ir.Value` or an op result.

## The subtle part: why the automatic CuteDSL downcast is not enough

CuteDSL does have an automatic rewrite for calls into `_mlir.dialects.*` modules. In [ast_preprocessor.py](../python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py#L1657), it wraps each top-level argument with `implicitDowncastNumericType(...)`.

That helper is defined in [typing.py](../python/CuTeDSL/cutlass/base_dsl/typing.py#L1947):

```python
def implicitDowncastNumericType(value):
    if isinstance(value, Numeric):
        return value.ir_value()
    return value
```

But this only applies to the top-level call arguments.

For `llvm.inline_asm`, the operands are passed as a Python list:

```python
[gmem_ptr.llvm_ptr, smem_ptr_i32, Int32(store_bytes).ir_value()]
```

The list itself is the top-level argument, not its elements. So CuteDSL will not recursively walk the list and downcast each member. That is the main reason the code is explicit here.

## Why each operand is written the way it is

### 1. `gmem_ptr.llvm_ptr`

This is the pointer case.

`gmem_ptr` is a CuteDSL pointer wrapper, not a raw LLVM pointer value. The property [core.py](../python/CuTeDSL/cutlass/cute/core.py#L1358) returns the LLVM pointer `ir.Value` by emitting an `unrealized_conversion_cast` in [core.py](../python/CuTeDSL/cutlass/cute/core.py#L1373).

So:

- `gmem_ptr` = DSL pointer object
- `gmem_ptr.llvm_ptr` = raw LLVM pointer SSA value

This is the pointer equivalent of calling `.ir_value()` on a numeric wrapper.

### 2. `smem_ptr.toint(...).ir_value()`

This is the shared-memory pointer case.

[core.py](../python/CuTeDSL/cutlass/cute/core.py#L1412) shows that `Pointer.toint()` returns:

- `Int64` for global/generic pointers
- `Int32` for other spaces such as shared memory

So `smem_ptr.toint()` produces a DSL `Int32` wrapper around the shared-memory address, and `.ir_value()` unwraps that to the raw MLIR integer operand.

This is consistent with nearby PTX helpers such as [cpasync/helpers.py](../python/CuTeDSL/cutlass/cute/nvgpu/cpasync/helpers.py#L348), which also convert global pointers to 64-bit integer operands and shared pointers to 32-bit integer operands before `llvm.inline_asm`.

### 3. `Int32(store_bytes).ir_value()`

`store_bytes` is a Python-side value or DSL integer, but the inline asm op still needs an MLIR operand.

Wrapping with `Int32(...)` ensures the operand has the intended width, and `.ir_value()` turns it into the actual SSA integer passed as `$2`.

If `store_bytes` is static, this emits an MLIR constant; if it is already dynamic, it reuses the existing value.

## What the other `llvm.inline_asm` arguments mean

### `None`

The inline asm returns no value. This PTX instruction is used only for side effects.

### The asm string

```text
cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [$0], [$1], $2;
```

This is the literal PTX template. `$0`, `$1`, `$2` refer to the operands chosen from the list according to the constraint string.

### `"l,r,r"`

This is the inline-asm constraint string.

Per LLVM LangRef, it is a comma-separated list matching the asm operand positions.

In this repo, the convention is consistently:

- `l` for 64-bit integer-like operands
- `r` for 32-bit integer-like operands
- `f` for `f32`

You can see that mapping explicitly in [utils.py](../thirdparty/quack/quack/utils.py#L66).

For this site, that means:

- `$0`: `l` -> the global-memory destination operand
- `$1`: `r` -> the shared-memory address operand
- `$2`: `r` -> the byte count operand

The first operand is passed as `gmem_ptr.llvm_ptr`, but it still matches the low-level operand slot selected by LLVM for this inline-asm operand. The important point here is that by op-construction time it is already a raw LLVM-compatible value, not a CuteDSL wrapper.

### `has_side_effects=True`

This tells LLVM/MLIR that the asm touches memory or otherwise does observable work, so it must not be treated as a pure expression and optimized away.

That is necessary here because the instruction performs a memory operation.

### `is_align_stack=False`

This is the LLVM inline-asm flag for whether the compiler should assume the asm requires extra stack alignment. That concern matters mostly for CPU targets; it is not needed here.

### `asm_dialect=llvm.AsmDialect.AD_ATT`

This is the LLVM dialect enum value for the default “ATT” dialect. The generated enum is in [_llvm_enum_gen.py](../python/CuTeDSL/cutlass/_mlir/dialects/_llvm_enum_gen.py#L9).

Important nuance:

- this name comes from LLVM’s generic inline-asm API
- it does **not** mean PTX itself is x86 AT&T assembly

MLIR’s LLVM dialect documents `asm_dialect` as `ATT (0) or Intel (1)`, and LLVM LangRef says the default inline-asm dialect is ATT, with Intel as the alternative. In this CUDA/PTX codebase, every `llvm.inline_asm` site uses `AD_ATT`; there are no `AD_Intel` sites.

So here `AD_ATT` is best understood as:

- “use LLVM’s default inline-asm dialect setting”
- not “PTX has switched to AT&T syntax”

## Short answer

- `ir_value()` is needed because `llvm.inline_asm` is a low-level LLVM-dialect op builder that wants raw `ir.Value` operands.
- CuteDSL’s automatic numeric downcast only applies to top-level call arguments, not elements inside the operand list passed to `llvm.inline_asm`.
- `gmem_ptr.llvm_ptr` is the pointer version of that unwrapping step.
- `smem_ptr.toint().ir_value()` is needed because this PTX operand is passed as a shared-memory integer address, not as a CuteDSL pointer object.
- `AsmDialect.AD_ATT` is LLVM’s enum for the default non-Intel inline-asm dialect; in this PTX context it is just the standard setting used by the LLVM API, not a statement that PTX is x86 AT&T assembly.
