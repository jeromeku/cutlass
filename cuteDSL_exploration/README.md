CuTeDSL JIT/Kernel Pipeline — Cutlass Python DSL (local package: `cutlass_package`)

This document maps what happens when you write a CuTeDSL program in Python and decorate it with `@cute.jit` and/or `@cute.kernel`, using the local copy of the NVIDIA Cutlass DSL you placed under `cutlass_package/`.

Contents

- What lives where in `cutlass_package/`
- End‑to‑end trace for `@cute.jit`
- End‑to‑end trace for `@cute.kernel`
- MLIR bindings, pass manager, execution engine
- CUDA module binding and kernel launch path

All paths below are relative to the repo root. Line numbers are taken from your local files at the time of this analysis; open the referenced file and jump to the line range to inspect the code.

What lives where

- `cutlass_package/_mlir/`: Thin Python wrappers around MLIR’s Python C-API.
  - `ir.py`, `execution_engine.py`, `passmanager.py` — wrappers re‑exporting MLIR IR, ExecutionEngine, and PassManager from `_mlir_libs` native modules.
  - `dialects/` — auto‑generated Python bindings for MLIR dialects used by the DSL (e.g., `gpu`, `nvvm`, `nvgpu`, `cute`).
  - `runtime/np_to_memref.py` — helpers for memref argument ABI.
- `cutlass_package/base_dsl/`: Dialect‑agnostic DSL framework.
  - `dsl.py` — core decorator plumbing, AST preprocessing hook, IR building, module caching, compile+JIT orchestration, kernel scaffolding.
  - `compiler.py` — MLIR PassManager + ExecutionEngine integration.
  - `jit_executor.py` — converts Python args to C ABI, calls JITed functions, preloads CUDA kernels from MLIR `gpu.binary` and launches them.
  - `runtime/cuda.py` — CUDA Driver API bindings used for kernel module loading and launching.
  - `ast_preprocessor.py` — optional AST transform for control‑flow constructs.
- `cutlass_package/cutlass_dsl/`: CuTe/CUTLASS‑specific extensions atop `base_dsl`.
  - `cutlass.py` — defines `CutlassBaseDSL` and the concrete `CuTeDSL`; provides `KernelLauncher` and the `gpu.launch_func` generation.
- `cutlass_package/cute/`: The public “cute” surface that re‑exports operations and exposes `cute.jit` and `cute.kernel`.

Minimal example

```python
from cutlass_package import cute

@cute.kernel
def _empty():
    # No body — ok for a minimal launch
    return

@cute.jit
def run():
    # Build MLIR, compile to cubin, load kernel, launch
    _empty().launch(grid=[1,1,1], block=[1,1,1])

run()  # Triggers the full pipeline
```

See `TRACE_end_to_end.md` for a step‑by‑step walk through what gets called for each decorator.

IR/cubin dump tool

- Script: `cuteDSL_exploration/dump_ir_and_cubin.py`
  - Dumps MLIR before and after the pass pipeline.
  - Best‑effort per‑pass snapshots (if your MLIR Python wheel exposes `PassManager.enable_ir_printing`).
  - Extracts embedded cubins from `gpu.binary` and writes them to `cubin/*.cubin` with an index.

Usage examples

- For the included minimal demo:

```bash
python3 cuteDSL_exploration/dump_ir_and_cubin.py \
  --file cuteDSL_exploration/example_minimal.py \
  --func run \
  --print-each-pass
```

- For a function in an importable module (adjust module and function names):

```bash
python3 cuteDSL_exploration/dump_ir_and_cubin.py \
  --module mypkg.mymod \
  --func my_host_entry \
  --args "[]" --kwargs "{}" \
  --arch sm_90a
```

Outputs

- `00-before.mlir`: original IR module before the pass pipeline.
- `pass-XXX.mlir`: optional after‑each‑pass snapshots (best‑effort).
- `zz-after.mlir`: final IR after the pipeline (contains `gpu.binary` with cubin if codegen ran).
- `cubin/*.cubin`: extracted cubin blobs keyed by their `gpu.binary` symbol.
- `cubin_index.json`: list of symbols and filepaths.
- `launch_sites.txt`: textual capture of `gpu.launch_func` call sites.
- `REPORT.json`: summary (function name, mangled name, pipeline, outputs).
