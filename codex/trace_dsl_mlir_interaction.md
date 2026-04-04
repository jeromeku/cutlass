# Trace: dynamic CuTeDSL -> MLIR interaction

This example is for understanding what happens during tracing, not for inspecting the final MLIR text.

Files:

- [trace_dsl_mlir_interaction.py](trace_dsl_mlir_interaction.py)
- [dsl.py](../python/CuTeDSL/cutlass/base_dsl/dsl.py#L578)
- [ast_preprocessor.py](../python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py#L1548)
- [typing.py](../python/CuTeDSL/cutlass/base_dsl/typing.py#L977)
- [typing.py](../python/CuTeDSL/cutlass/base_dsl/typing.py#L1947)
- [cutlass_ast_decorators.py](../python/CuTeDSL/cutlass/cutlass_dsl/cutlass_ast_decorators.py#L436)
- [cutlass.py](../python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py#L2205)
- [arith.py](../python/CuTeDSL/cutlass/base_dsl/_mlir_helpers/arith.py#L292)

## What the script shows

The traced function mixes three styles on purpose:

1. A `cutlass.const_expr(...)` branch.
2. Ordinary DSL arithmetic like `y = x + bias`.
3. One direct MLIR dialect call, `arith.addi(y, cutlass.Int32(5))`.

That combination exposes the main boundaries:

- `BaseDSL.run_preprocessor(...)` shows when the source function is AST-rewritten.
- `BaseDSL.generate_execution_arguments(...)` plus `new_from_mlir_values(...)` show when the original Python arguments become DSL wrapper objects around MLIR block arguments.
- `arith_helper.const(...)` shows when DSL numerics materialize IR constants.
- `implicitDowncastNumericType(...)` shows the special case for direct `_mlir.dialects.*` calls.
- `_if_execute_dynamic(...)` shows when a Python `if` on a dynamic value becomes an `scf.IfOp`.

## Why the direct `arith.addi(...)` call matters

Most user-facing DSL arithmetic does not call the raw MLIR dialect API directly. It goes through DSL wrappers like `cutlass.Int32`, `Numeric.ir_value()`, and helper code in [arith.py](../python/CuTeDSL/cutlass/base_dsl/_mlir_helpers/arith.py#L292).

The direct `arith.addi(y, cutlass.Int32(5))` call is different:

- the AST preprocessor sees that `arith` comes from `cutlass._mlir.dialects`
- it rewrites each top-level argument as `implicitDowncastNumericType(...)`
- `y` is a DSL numeric wrapper, so it is unwrapped to its `ir.Value`
- `cutlass.Int32(5)` is also unwrapped, which forces constant materialization first
- the final call into the MLIR binding receives raw MLIR operands, not DSL wrappers

That is the cleanest place to watch the DSL-to-MLIR handoff happen live.

## How to run it

From the repo root:

```bash
.venv/bin/python codex/trace_dsl_mlir_interaction.py
```

By default the script sets `CUTE_DSL_DRYRUN=1`, so it stops after IR generation. If you want the same trace but without forcing dry-run from inside the script, edit the env defaults at the top or override them before launch.

Useful optional env vars:

- `CUTE_DSL_PRINT_AFTER_PREPROCESSOR=1`
- `CUTE_DSL_LOG_TO_CONSOLE=1`

The script also prints the actual `cutlass` import path at startup. In this environment that path may point at the package installed in `.venv`, even though the source links in this note point at the repo checkout.

## Expected reading order

Read the output in this order:

1. `run_preprocessor`: Python source is transformed.
2. `generate_execution_arguments` and `new_from_mlir_values`: block args become `cutlass.Int32(...)` wrappers.
3. `[dsl] ...` lines: your Python function body is executing during tracing.
4. `arith_helper.const` and `mlir_arith.*`: DSL values are being turned into MLIR ops.
5. `compare_executor` and `_if_execute_dynamic`: a dynamic Python `if` is lowered into MLIR control flow.

One surprising but important detail:

- both the `then` and `else` branch bodies run during tracing
- that is expected because MLIR needs both regions populated while building `scf.if`
- this is different from normal Python execution, where only one branch body runs

## Key functions index

| Function | File | Purpose |
|----------|------|---------|
| `BaseDSL.run_preprocessor` | [dsl.py](../python/CuTeDSL/cutlass/base_dsl/dsl.py#L1419) | AST rewrite entry point before tracing |
| `BaseDSL.generate_execution_arguments` | [dsl.py](../python/CuTeDSL/cutlass/base_dsl/dsl.py#L578) | Rebuilds DSL wrapper objects from MLIR block arguments |
| `new_from_mlir_values` | [dsl.py](../python/CuTeDSL/cutlass/base_dsl/dsl.py#L154) | Populates wrapper objects with fresh MLIR values |
| `implicitDowncastNumericType` | [typing.py](../python/CuTeDSL/cutlass/base_dsl/typing.py#L1947) | Unwraps DSL numerics to raw `ir.Value` for direct MLIR dialect calls |
| `_if_execute_dynamic` | [cutlass_ast_decorators.py](../python/CuTeDSL/cutlass/cutlass_dsl/cutlass_ast_decorators.py#L436) | Builds `scf.if` for dynamic Python control flow |
| `_compare_executor` | [cutlass.py](../python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py#L2205) | Dispatches traced comparison operators |
| `const` | [arith.py](../python/CuTeDSL/cutlass/base_dsl/_mlir_helpers/arith.py#L292) | DSL helper that materializes MLIR constants |
