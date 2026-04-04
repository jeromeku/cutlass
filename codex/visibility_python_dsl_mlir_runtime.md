# Visibility across Python / DSL, MLIR, and kernel execution

Question source:

- [VISIBILITY.md](../VISIBILITY.md)

Key code/documentation sites:

- [python/CuTeDSL/cutlass/base_dsl/dsl.py#L1102](../python/CuTeDSL/cutlass/base_dsl/dsl.py#L1102)
- [python/CuTeDSL/cutlass/base_dsl/compiler.py#L136](../python/CuTeDSL/cutlass/base_dsl/compiler.py#L136)
- [python/CuTeDSL/cutlass/base_dsl/env_manager.py#L312](../python/CuTeDSL/cutlass/base_dsl/env_manager.py#L312)
- [python/CuTeDSL/cutlass/base_dsl/compiler.py#L367](../python/CuTeDSL/cutlass/base_dsl/compiler.py#L367)
- [python/CuTeDSL/cutlass/base_dsl/cache_helpers.py#L179](../python/CuTeDSL/cutlass/base_dsl/cache_helpers.py#L179)
- [python/CuTeDSL/cutlass/base_dsl/jit_executor.py#L910](../python/CuTeDSL/cutlass/base_dsl/jit_executor.py#L910)
- [media/docs/pythonDSL/cute_dsl_general/debugging.rst#L18](../media/docs/pythonDSL/cute_dsl_general/debugging.rst#L18)
- [codex/cuTeDSL-pipeline-trace.md](./cuTeDSL-pipeline-trace.md)

External references:

- [py-spy README](https://github.com/benfred/py-spy)
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/2022.2/UserGuide/index.html)
- [Nsight Compute Documentation](https://archive.docs.nvidia.com/nsight-compute/2019.5.1/NsightCompute/index.html)

## Short answer

Yes, `py-spy` helps, but only for part of the problem.

- `py-spy` is useful for **compile-time Python/DSL behavior**
- it is **not** the right tool for understanding GPU kernel execution
- it is only partially useful for MLIR/MLIR-C internals unless you use native stack sampling and have symbols

To really see “under the hood”, you want three layers of visibility:

1. Python / DSL trace-time behavior
2. MLIR generation + pass pipeline artifacts
3. Runtime host launch + GPU kernel execution

## The actual phase split in this repo

The repo’s compile path is:

```text
Python source
  -> AST preprocessing
  -> traced execution with IR proxy args
  -> MLIR module construction
  -> PassManager.parse(...).run(...)
  -> PTX / CUBIN artifacts
  -> ExecutionEngine / launch path
```

The two most important frames are:

- [BaseDSL.generate_original_ir](../python/CuTeDSL/cutlass/base_dsl/dsl.py#L1102): executes the transformed Python function and emits MLIR
- [Compiler.compile](../python/CuTeDSL/cutlass/base_dsl/compiler.py#L136): runs the MLIR pass pipeline

So:

- “What is Python doing?” mostly means `generate_original_ir` and the AST preprocessor path
- “What is MLIR doing?” mostly means the module built by tracing plus the pass manager pipeline
- “What is the GPU doing?” starts after compilation, during launch and kernel execution

## Where `py-spy` helps

`py-spy` is strongest when the question is:

- which Python functions dominate compile time?
- is AST preprocessing expensive?
- is IR generation spending time in Python loops or helper code?
- which host-side code path is active before launch?

This fits CuTeDSL well for the trace-time layer because `generate_original_ir` literally executes Python to build MLIR.

Good targets:

- [python/CuTeDSL/cutlass/base_dsl/dsl.py#L1102](../python/CuTeDSL/cutlass/base_dsl/dsl.py#L1102)
- [python/CuTeDSL/cutlass/base_dsl/dsl.py#L1165](../python/CuTeDSL/cutlass/base_dsl/dsl.py#L1165)
- [python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py](../python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py)

Limits:

- plain `py-spy` samples Python stacks, so it does **not** explain the semantics of the MLIR being built
- it also does **not** show what the GPU kernel is doing after launch
- if time is spent inside native extensions, you need `py-spy --native`, and even then you are sampling native stacks, not getting IR dumps or GPU timelines

That means `py-spy` is useful for answering:

- “why is compile time slow?”

but not sufficient for:

- “what MLIR was generated?”
- “which MLIR pass changed the IR?”
- “what did the kernel do on the GPU?”

## What to use for compile-time visibility instead of only `py-spy`

### 1. Built-in CuTeDSL logging and timing

The repo already exposes:

- `CUTE_DSL_JIT_TIME_PROFILING`
- `CUTE_DSL_LOG_TO_CONSOLE`
- `CUTE_DSL_LOG_LEVEL`

via [env_manager.py](../python/CuTeDSL/cutlass/base_dsl/env_manager.py#L291).

These tell you how long IR generation / compilation / execution take at a coarse level, and they follow the actual CuTeDSL compile path instead of only sampling Python.

### 2. Print or keep the generated MLIR

Built-in knobs:

- `CUTE_DSL_PRINT_IR`
- `CUTE_DSL_KEEP_IR`
- `CUTE_DSL_DUMP_DIR`

documented in [debugging.rst](../media/docs/pythonDSL/cute_dsl_general/debugging.rst#L71) and wired in [dsl.py](../python/CuTeDSL/cutlass/base_dsl/dsl.py#L1067).

This is the best first step if your question is:

- “what did tracing emit?”

because it gives you the actual MLIR module, not a sampled stack.

### 3. Keep PTX and CUBIN too

Built-in knobs:

- `CUTE_DSL_KEEP_PTX`
- `CUTE_DSL_KEEP_CUBIN`
- `CUTE_DSL_LINEINFO`

documented in [debugging.rst](../media/docs/pythonDSL/cute_dsl_general/debugging.rst#L86) and implemented through compile options in [compiler.py](../python/CuTeDSL/cutlass/base_dsl/compiler.py#L367).

This is the best next step if your question is:

- “did lowering produce the PTX I expected?”
- “can I correlate source/PTX/SASS in Nsight?”

### 4. Use the programmatic artifact handles

Compiled functions expose:

- `__mlir__`
- `__ptx__`
- `__cubin__`

at [jit_executor.py](../python/CuTeDSL/cutlass/base_dsl/jit_executor.py#L910).

This is often the cleanest way to inspect artifacts from a notebook or test without chasing temp files.

## What to use for MLIR / MLIR-C visibility

If the question is “what MLIR exists before and after compilation?”, the dump knobs above are enough.

If the question is “what did each pass do?”, the binding in this repo already exposes per-pass IR printing on the pass manager:

- [passmanager.pyi#L17](../python/CuTeDSL/cutlass/_mlir/_mlir_libs/_cutlass_ir/_mlir/passmanager.pyi#L17)

```python
pm.enable_ir_printing(
    print_before_all=True,
    print_after_all=True,
    enable_debug_info=True,
)
```

But the current `Compiler.compile` path in [compiler.py](../python/CuTeDSL/cutlass/base_dsl/compiler.py#L136) does not enable this today; it only does:

```python
pm = self.passmanager.PassManager.parse(pipeline)
pm.enable_verifier(enable_verifier)
pm.run(module.operation)
```

So if you want true per-pass MLIR visibility, the capability exists in the bindings, but you would need a small opt-in hook in the compiler path.

## What to use for runtime / kernel execution visibility

This is where `py-spy` stops helping.

### 1. Nsight Systems

Use this when the question is:

- when does compile happen vs launch happen?
- how much time is spent in Python, CUDA runtime, synchronization, and kernel launches?
- how do host phases line up on the timeline?

`CUTE_DSL_LINEINFO=1` is useful here because the repo docs explicitly say line info enables profiler/debugger support at [debugging.rst](../media/docs/pythonDSL/cute_dsl_general/debugging.rst#L21).

### 2. Nsight Compute

Use this when the question is:

- what is the kernel doing on the GPU?
- what are occupancy, memory throughput, stalls, tensor core usage, source/PTX/SASS correlation?

This is the right tool for kernel behavior, not `py-spy`.

### 3. Compute Sanitizer

Use this when the question is:

- is the kernel wrong because of OOB, race, init, or sync bugs?

The repo docs already recommend it at [debugging.rst](../media/docs/pythonDSL/cute_dsl_general/debugging.rst#L162).

## Practical workflow

If you want full visibility with minimal thrash:

1. Start with artifact dumps:
   - `CUTE_DSL_KEEP_IR=1`
   - `CUTE_DSL_KEEP_PTX=1`
   - `CUTE_DSL_KEEP_CUBIN=1`
   - `CUTE_DSL_DUMP_DIR=/tmp/cutedsl_debug`
2. Add source correlation:
   - `CUTE_DSL_LINEINFO=1`
3. If compile time is suspicious:
   - use `CUTE_DSL_JIT_TIME_PROFILING=1`
   - then use `py-spy` on the compile-driving Python process
4. If runtime is suspicious:
   - use Nsight Systems for the timeline
   - use Nsight Compute for a specific kernel
5. If correctness is suspicious:
   - use Compute Sanitizer

## Bottom line

- `py-spy` is a good tool for **Python-side compile-time visibility**
- it is not the primary tool for **MLIR visibility**
- it is not the primary tool for **kernel execution visibility**

For this repo, the highest-value combination is:

- CuTeDSL dump/log knobs for MLIR/PTX/CUBIN
- `py-spy` for Python trace-time hotspots
- Nsight Systems for host/runtime timeline
- Nsight Compute for kernel behavior
- Compute Sanitizer for runtime correctness issues
