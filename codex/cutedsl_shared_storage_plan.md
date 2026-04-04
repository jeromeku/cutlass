# Plan: CuTe DSL Shared Storage Lowering

## Goal

Explain how code like:

```python
@cute.struct
class SharedStorage:
    staging_buffer: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 1], 1024
    ]
```

is compiled by cuteDSL, with emphasis on:

- what `@cute.struct` means
- how `MemRange` and `Align` are represented
- how static shared memory is expressed in MLIR / LLVM / NVVM
- why this does not require explicit CUDA driver allocation calls in user code

## Chunks

1. Read the surface example in `experiments/async_pipeline.py`.
2. Trace the Python DSL implementation of `@cute.struct`, `MemRange`, and `Align`.
3. Trace how the resulting object is lowered into IR and how shared memory pointers are materialized.
4. Explain the CUDA model distinction:
   - static shared memory as kernel metadata / address-space allocation
   - dynamic runtime launch config vs user-visible driver calls
5. Write the trace note in `codex/cutedsl_shared_storage.md`.
