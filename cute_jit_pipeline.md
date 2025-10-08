# CuTeDSL `@cute.jit` MLIR Pass Pipeline

This note documents exactly where the MLIR pass pipeline is built and executed for host‑side `@cute.jit` functions, how the default pipeline string is assembled, and how to print IR after each pass for fine‑grained inspection.

## TL;DR

- Pipeline executes inside MLIR’s `PassManager.run` invoked by CuTeDSL’s `Compiler.compile`:
  - `python/CuTeDSL/cutlass/base_dsl/compiler.py:145` parses the pipeline string, and `python/CuTeDSL/cutlass/base_dsl/compiler.py:147` runs it.
- The default pipeline string for CuTeDSL host code is assembled here:
  - `python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py:215` `_get_pipeline(...)` → returns `builtin.module(cute-to-nvvm{cubin-format=bin ...})`.
  - `python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py:227` `preprocess_pipeline(...)` → appends `,external-kernel-for-gpu-launch)`.
  - `python/CuTeDSL/cutlass/base_dsl/dsl.py:974` base `preprocess_pipeline(...)` injects options like `toolkitPath` and `cubin-chip=<arch>`.
- To see pass‑by‑pass IR: enable `PassManager.enable_ir_printing(...)` (bindings at `python/CuTeDSL/cutlass/_mlir/_mlir_libs/_mlir/passmanager.pyi:12`). A minimal monkey‑patch is provided below.

---

## End‑to‑end Call Flow for a Host‑Side `@cute.jit`

1) Decorator entrypoint
- `python/CuTeDSL/cutlass/base_dsl/dsl.py:494` `BaseDSL.jit(...)` sets up the host JIT wrapper.

2) Wrapper → executor
- `python/CuTeDSL/cutlass/base_dsl/dsl.py:466` `BaseDSL.jit_runner(...)` wires the call through the internal executor.

3) Host executor builds the MLIR module
- `python/CuTeDSL/cutlass/base_dsl/dsl.py:1328` `BaseDSL._func(...)` generates the IR module for the Python function body, handling argument canonicalization and typing. It accepts an optional `pipeline=` kwarg for overrides (`python/CuTeDSL/cutlass/base_dsl/dsl.py:1346`).

4) Compile and cache
- `python/CuTeDSL/cutlass/base_dsl/dsl.py:1125` `compile_and_cache(...)` constructs the final pipeline string via base `preprocess_pipeline(...)` and calls `compile_and_jit(...)`.

5) Pipeline + JIT
- `python/CuTeDSL/cutlass/base_dsl/dsl.py:936` `compile_and_jit(...)` delegates to the `Compiler` provider:
  - `python/CuTeDSL/cutlass/base_dsl/compiler.py:135` `Compiler.compile(...)` →
  - `python/CuTeDSL/cutlass/base_dsl/compiler.py:145` `pm = self.passmanager.PassManager.parse(pipeline)` →
  - `python/CuTeDSL/cutlass/base_dsl/compiler.py:147` `pm.run(module.operation)` executes the pass pipeline →
  - `python/CuTeDSL/cutlass/base_dsl/compiler.py:160` `Compiler.jit(...)` wraps with `ExecutionEngine`.

Notes
- The CuTeDSL subclass sets `pass_sm_arch_name = "cubin-chip"` at `python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py:554`. Base preprocessing (`python/CuTeDSL/cutlass/base_dsl/dsl.py:974`) uses this key to inject the selected arch (e.g., `sm_90`).
- The base JIT path temporarily captures `stdout/stderr` (`python/CuTeDSL/cutlass/base_dsl/dsl.py:944–961`) and prints them after compilation; pass printing will therefore appear after the compile returns.

---

## Where the Pipeline String Comes From

CuTeDSL default (host side):
- `python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py:215` → `_get_pipeline(...)` returns
  - ``builtin.module(cute-to-nvvm{cubin-format=bin <opts>})``
- `python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py:227` → `preprocess_pipeline(...)` appends
  - `,external-kernel-for-gpu-launch)`
- `python/CuTeDSL/cutlass/base_dsl/dsl.py:974` → base `preprocess_pipeline(...)` adds
  - `toolkitPath=<CUDA_TOOLKIT_PATH>` (if set)
  - `<pass_sm_arch_name>=<arch>` (for CuTe, `cubin-chip=<arch>`)
  - It logs the final string: “Using pipeline = …” (enable console logging to see it; see below).

You can override the pipeline per‑function:
- Pass `pipeline="..."` to `@cute.jit` (consumed at `python/CuTeDSL/cutlass/base_dsl/dsl.py:1346`).

---

## Printing IR After Every Pass (Fine‑Grained Lowering)

MLIR bindings expose `PassManager.enable_ir_printing(...)` at `python/CuTeDSL/cutlass/_mlir/_mlir_libs/_mlir/passmanager.pyi:12`. Without modifying CuTeDSL sources, you can monkey‑patch the compiler in a small harness:

```python
from cutlass.base_dsl import compiler as _compiler

_orig_compile = _compiler.Compiler.compile

def _traced_compile(self, module, pipeline, cuda_toolkit="", arch="", enable_verifier=False):
    pm = self.passmanager.PassManager.parse(pipeline)
    pm.enable_ir_printing(
        print_before_all=False,
        print_after_all=True,
        print_module_scope=False,
        print_after_change=True,
        print_after_failure=True,
        tree_printing_dir_path="mlir_pass_dumps"  # optional: write files per pass
    )
    pm.enable_verifier(enable_verifier)
    pm.run(module.operation)

_compiler.Compiler.compile = _traced_compile
```

Then invoke your `@cute.jit` function. You’ll get per‑pass IR on stderr (emitted after compilation returns), and—if `tree_printing_dir_path` is set—files under `mlir_pass_dumps/`.

Tip: to also see the final pipeline string and other debug logs, enable CuTeDSL logging to console:

```bash
export CUTE_DSL_LOG_TO_CONSOLE=1
export CUTE_DSL_LOG_LEVEL=10   # DEBUG per python logging
# Optional runtime context
export CUDA_TOOLKIT_PATH=/path/to/cuda
export CUTE_DSL_ARCH=sm_90
```

---

## Applying to the Example

For the example:

```python
@cute.jit
def load_and_store(res: cute.Tensor, a: cute.Tensor, b: cute.Tensor):
    a_vec = a.load()
    print(f"a_vec: {a_vec}")
    b_vec = b.load()
    print(f"b_vec: {b_vec}")
    res.store(a_vec + b_vec)
    cute.print_tensor(res)
```

Run with the monkey‑patch active. You should see, in order:
- The assembled pipeline string (if logging enabled).
- A sequence of pass names/IR dumps from `PassManager.enable_ir_printing(...)`.
- Final JIT and execution output.

---

## Key Source Pointers (clickable in IDE)

- Execution of pipeline
  - `python/CuTeDSL/cutlass/base_dsl/compiler.py:145` (parse), `python/CuTeDSL/cutlass/base_dsl/compiler.py:147` (run)
- Host JIT flow
  - `python/CuTeDSL/cutlass/base_dsl/dsl.py:494` `BaseDSL.jit`
  - `python/CuTeDSL/cutlass/base_dsl/dsl.py:466` `BaseDSL.jit_runner`
  - `python/CuTeDSL/cutlass/base_dsl/dsl.py:1328` `BaseDSL._func`
  - `python/CuTeDSL/cutlass/base_dsl/dsl.py:1125` `compile_and_cache`
  - `python/CuTeDSL/cutlass/base_dsl/dsl.py:936` `compile_and_jit`
- Pipeline assembly
  - `python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py:215` `_get_pipeline`
  - `python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py:227` `preprocess_pipeline`
  - `python/CuTeDSL/cutlass/base_dsl/dsl.py:974` base `preprocess_pipeline`
- MLIR pass printing API
  - `python/CuTeDSL/cutlass/_mlir/_mlir_libs/_mlir/passmanager.pyi:12` `enable_ir_printing`

---

## Optional Future Improvement

If desired, one could add an environment‑guarded switch (e.g., `CUTE_DSL_PASS_TRACE=1`) inside `Compiler.compile(...)` to call `enable_ir_printing(...)` without monkey‑patching. That change would be limited to `python/CuTeDSL/cutlass/base_dsl/compiler.py` and can respect a dump directory env var.

