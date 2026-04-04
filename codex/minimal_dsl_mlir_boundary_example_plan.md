# Plan: minimal dynamic CuTeDSL -> MLIR tracing example

Goal: create the smallest runnable example that shows the live interaction between Python/CuTeDSL and the MLIR Python bindings during tracing, without relying on kernel execution.

## Chunk 1: choose the minimal boundary

Use a scalar-only `@cute.jit` function plus lightweight tracing instrumentation rather than a tensor or `@cute.kernel` launch example.

Why:

- it still exercises the important dynamic boundary:
  - Python source
  - AST preprocessing
  - traced execution with IR proxy values
  - DSL helper calls
  - MLIR builder calls
- it avoids unrelated complexity from tensors, launch config, and runtime execution
- it can demonstrate the key distinction between:
  - compile-time Python branches using `cutlass.const_expr(...)`
  - dynamic branches that trigger MLIR control-flow construction

## Chunk 2: create the runnable example

Add a small script under `codex/` that:

1. defines a tiny `@cute.jit` function with:
   - one `Constexpr` argument
   - one dynamic `Int32` argument
   - a compile-time Python `if`
   - a dynamic `if` that lowers to IR
   - one direct call into `cutlass._mlir.dialects.arith` so the auto-downcast boundary is visible
2. installs a focused trace/probe layer around a few dynamic boundary points such as:
   - `BaseDSL.run_preprocessor`
   - `BaseDSL.generate_mlir_function_types`
   - `BaseDSL.generate_execution_arguments`
   - `new_from_mlir_values`
   - `implicitDowncastNumericType`
   - `_if_execute_dynamic`
   - selected MLIR dialect builder wrappers like `arith.constant` / arithmetic ops
3. calls `cute.compile(...)` in dry-run mode by default
4. prints a compact event stream showing:
   - when Python code is still running normally
   - when arguments become IR proxy objects
   - when DSL helpers invoke MLIR builder functions
   - when a direct MLIR dialect call receives auto-downcasted `ir.Value` operands
5. does not execute the compiled function or any kernel

## Chunk 3: create the companion walkthrough

Add a short markdown note under `codex/` that explains:

- what runs in plain Python
- what runs during trace-time with IR proxy objects
- which probes correspond to AST rewrite, IR proxy reconstruction, and MLIR builder calls
- how to run the example with useful env vars such as:
  - `CUTE_DSL_DRYRUN=1` if we want to stop after IR generation
  - `CUTE_DSL_PRINT_AFTER_PREPROCESSOR=1` when useful
  - `CUTE_DSL_LOG_TO_CONSOLE=1`

## Proposed files

- `codex/trace_dsl_mlir_interaction.py`
- `codex/trace_dsl_mlir_interaction.md`

## Expected outcome

After running the example, the user should be able to see:

- a compile-time-only Python branch execute as ordinary Python during tracing
- a dynamic branch force creation of IR-level control flow
- the exact handoff from CuTeDSL wrapper code to MLIR-facing Python builder calls
- the special case where calling `cutlass._mlir.dialects.arith.*` directly causes the AST preprocessor to inject `implicitDowncastNumericType(...)` around top-level arguments
