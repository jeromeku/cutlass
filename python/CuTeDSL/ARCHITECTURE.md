CuTeDSL Architecture

This document explains how CuTeDSL compiles and runs your Python code, how to inspect each stage of the pipeline, and how to tweak the MLIR pipeline (e.g., to cross‑compile for a different GPU architecture). It follows the philosophy from matklad’s ARCHITECTURE.md post: what exists, how pieces fit together, and how to extend/debug safely.

High‑Level Flow

- Frontend (Python)
  - You write host code with `@cute.jit` and device code with `@cute.kernel` (cutlass.cute aliases onto cutlass.cutlass_dsl.CuTeDSL).
  - Optional AST preprocessing rewrites Python control flow (for/if/while) into a structured form suitable for IR generation.
- IR Builder (MLIR)
  - Builds an MLIR `module` with a `gpu.module` named `kernels` for device functions and a host `func.func` that launches the GPU kernel.
- Pass Pipeline (MLIR → NVVM/CUBIN)
  - Default pipeline: `builtin.module(cute-to-nvvm{cubin-format=bin ...})` plus `external-kernel-for-gpu-launch`.
  - Lowers Cute/Cute-NVGPU/NVGPU/GPU/SCF/Arith/Func to NVVM and embeds CUBIN in `gpu.binary` ops.
- JIT + Runtime
  - MLIR ExecutionEngine JITs the host function (`llvm.emit_c_interface`).
  - Python extracts embedded CUBIN from `gpu.binary`, loads it via CUDA Driver API (`cuModuleLoadData`), resolves kernel function pointer, and passes it to the JITed host function which performs `gpu.launch_func`.

Key Modules And Responsibilities

- Frontend and orchestration
  - `cutlass/cutlass_dsl/cutlass.py:546` class `CuTeDSL` (subclasses `BaseDSL`).
  - `cutlass/base_dsl/dsl.py` core machinery: decorators, AST preprocessor hookup, IR building, compilation, caching, and execution.
  - `cutlass/base_dsl/ast_preprocessor.py` rewrites Python AST for structured control flow.
  - `cutlass/base_dsl/compiler.py` wraps MLIR PassManager + ExecutionEngine.
  - `cutlass/base_dsl/jit_executor.py` loads CUBINs, binds kernels (cuModule*), and invokes the JITed host function.
- MLIR bindings and dialects
  - `cutlass/_mlir/ir.py`, `passmanager.py`, `execution_engine.py`: thin wrappers around MLIR’s C Python bindings.
  - `cutlass/_mlir/dialects/*`: autogen’d ops/enums; includes upstream dialects (arith, func, scf, gpu, nvgpu, nvvm, llvm, vector) and Cutlass/Cute dialects.

End‑to‑End Pipeline (Detailed)

1) Decorators and AST Preprocessing

- `@cute.jit` on a host function/method wraps the call through `BaseDSL.jit_runner` which optionally enables AST transformation (default on):
  - `dsl.py:468` `jit_runner` sets up preprocessor state.
  - `dsl.py:556` `BaseDSL._preprocess_and_execute` runs `DSLPreprocessor.transform` (`ast_preprocessor.py`) and materializes a new Python function (via `exec`).
  - The preprocessor rewrites supported control flow:
    - For loops: `range`, `range_constexpr`, `range_dynamic` → `loop_selector` machinery.
    - If/elif/else, while, assert, any/all, and compare ops (
      see constants in `ast_preprocessor.py:87–114`).

2) IR Generation (Host func + GPU module + kernel func)

- MLIR context and module creation
  - `dsl.py:1162` `generate_mlir` enters `with ir.Context(), ir.Location.unknown()`.
  - `dsl.py:1026` `generate_original_ir` creates `ir.Module` and marks it with `gpu.container_module`.
  - `dsl.py:1089` calls `_build_gpu_module(...)` to create a `gpu.module @kernels` region.
- Host function (the `@jit` body)
  - `dsl.py:1094` `func.FuncOp(function_name, (func_types, []))` builds a host `func.func` with `llvm.emit_c_interface`.
  - DSL converts Python arguments to MLIR types/values (`dsl.py:850` `generate_mlir_function_types`, plus adapters in `runtime/jit_arg_adapters.py`).
- Kernel function (the `@kernel` body)
  - The `@cute.kernel` wrapper (`dsl.py:1569` `kernel_launcher`) computes a unique name and produces:
    - `func @kernel_*` inside `gpu.module @kernels` (via helper in `cutlass_dsl/cutlass.py:235`), annotated with `gpu.kernel`, `cute.kernel`, and NVVM attributes like `nvvm.reqntid` (threads per block).
    - A host‐side `gpu.launch_func` call that references `@kernels::@kernel_*` with launch dims, cluster dims, dynamic shared memory, and operands (built in `cutlass_dsl/cutlass.py:377–420`).
  - The body of `@kernel` uses Cute/Cute-NVGPU ops (e.g., TMA, WGMMA) which are MLIR ops generated via Python APIs in `cutlass/_mlir/dialects` and high‑level helpers in `cutlass/cute/*`.

3) Pass Pipeline and CUBIN Emission

- Default pipeline (`cutlass_dsl/cutlass.py:214`):
  - `builtin.module(cute-to-nvvm{cubin-format=bin ...})`
  - `dsl.py:930` `preprocess_pipeline` injects options: CUDA toolkit path and target SM arch.
  - `cutlass_dsl/cutlass.py:222` appends `external-kernel-for-gpu-launch` to make `gpu.launch_func` consume an external kernel pointer argument (provided from Python via CUDA Driver API).
- What the pipeline does
  - Lowers custom Cute/Cute-NVGPU ops to NVGPU/NVVM, converts GPU ops to NVVM, lowers control flow, emits `gpu.binary` with CUBIN bytes for target arch.
  - Verification (`dsl.py:1066`) ensures module well‑formedness before running passes.

4) JIT and Execution

- Compilation and JIT
  - `dsl.py:1121` composes the pipeline string; `compiler.Compiler.compile_and_jit` (`compiler.py:199`) runs PassManager and then wraps module in an MLIR `ExecutionEngine`.
  - Shared runtime libraries are discovered via `EnvironmentVarManager` and passed to the engine (CUDA runtime shims: `mlir_cuda_runtime`, etc.).
- CUBIN extraction and cuModule binding
  - `jit_executor.py:318` `walk_module_and_get_cubin_data` visits `gpu.binary`, decodes CUBIN bytes, and calls:
    - `cuModuleLoadData` (via `runtime/cuda.py`) to create a CUDA module.
    - `cuModuleGetFunction` to obtain a function pointer for `@kernel_*`.
    - Optionally sets attributes, e.g., non‑portable cluster size for modern drivers.
  - Kernel pointers are appended to the packed arg list for the host function (`jit_executor.py:210`).
- Host function invocation
  - `execution_engine.ExecutionEngine.lookup` exposes a `CFUNCTYPE(void(void**))` entrypoint for `llvm.emit_c_interface` functions (`_mlir/execution_engine.py`).
  - `JitExecutor.run_compiled_program` packs arguments (memrefs, scalars, stream, kernel pointer) and invokes the JITed host function. That function executes `gpu.launch_func` against the provided kernel pointer.

Inspecting Each Stage

- Environment variables (prefix `CUTE_DSL_`) are read by `EnvironmentVarManager` (`base_dsl/env_manager.py`). Useful ones:
  - `CUTE_DSL_PRINT_AFTER_PREPROCESSOR=1` print transformed Python after AST rewriting.
  - `CUTE_DSL_PRINT_IR=1` print MLIR with debug info.
  - `CUTE_DSL_KEEP_IR=1` save MLIR bytecode to `cutlass_<func>.mlir` (see `base_dsl/cache_helpers.py`).
  - `CUTE_DSL_DRYRUN=1` build IR only, skip compile/JIT.
  - `CUTE_DSL_JIT_TIME_PROFILING=1` print timing for IR gen/compile/launch.
  - `CUTE_DSL_NO_SOURCE_LOCATION=1` disable file/line locations in IR.
  - `CUTE_DSL_ARCH=sm_90|sm_100|...` override target arch.
  - `CUTE_DSL_LIBS=/path/to/libmlir_cuda_runtime.so:...` override runtime libs.
- Programmatic peeking (documented/public)
  - Compile‑only to get a handle: `compiled = cute.compile(fn, *args, options="--opt-level 3", pipeline=...)` returns a `JitExecutor`.
    - `compiled.ir_module` gives the MLIR module (print or `write_bytecode`).
    - `compiled.engine` is the `ExecutionEngine`.
    - `compiled.function_name` is the host entrypoint symbol.
    - `compiled.cuda_modules` contains loaded CUDA modules and kernel pointers (after first update/launch).
  - To get CUBIN bytes: call `compiled.update_jit_cuda_modules(kernel_symbols)` or directly reuse `walk_module_and_get_cubin_data`.
- Programmatic peeking (internal but stable enough for debugging)
  - `BaseDSL.generate_original_ir(...)` builds the unlowered module; it’s not returned publicly, but you can approximate by setting `CUTE_DSL_DRYRUN=1` and/or `CUTE_DSL_KEEP_IR=1`.
  - `BaseDSL.preprocess_pipeline(...)` shows how CUDA toolkit and arch options are stitched into the pipeline string for inspection.
- MLIR diagnostics
  - Run Python with `-diagnostic` arg (handled in `dsl.py:1225`) to enable MLIR diagnostic handlers and filter types (fail/success/info/suggestion) via `mlir.ir._GlobalDebug`.

Example Walkthrough: dense_gemm.py (Hopper)

File: `examples/python/CuTeDSL/hopper/dense_gemm.py`

- Kernel class: `HopperWgmmaGemmKernel` at `examples/python/CuTeDSL/hopper/dense_gemm.py:203`.
- Host entrypoint: `__call__` decorated with `@cute.jit` at `examples/python/CuTeDSL/hopper/dense_gemm.py:372`.
  - Sets up data types, layouts, attributes, shared storage type, grid/block/cluster dims.
  - Calls the device kernel via `self.kernel(...).launch(grid=..., block=..., cluster=..., stream=...)` at the end of `__call__`.
- Device kernel: `kernel` decorated with `@cute.kernel` at `examples/python/CuTeDSL/hopper/dense_gemm.py:482`.
  - Emits Cute/Cute‑NVGPU ops for TMA copies, WGMMA (mainloop + epilogue), pipelines, mbarriers, etc.
- Compilation site: `compiled_gemm = cute.compile(gemm, mA, mB, mC, stream)` at `examples/python/CuTeDSL/hopper/dense_gemm.py:1508`.
  - `cute.compile` calls the underlying `@cute.jit` pipeline with `compile_only=True` and `no_cache=True`, returning a `JitExecutor` you can introspect.

Lowering Trace (conceptual mapping to IR)

1) `@cute.jit` function `__call__`
  - Host `func @cutlass___call__...` is created in the top‑level module and marked `llvm.emit_c_interface` (see `dsl.py:1094`).
  - It constructs operands (memrefs/pointers/scalars) from Python types via `generate_mlir_function_types` and adapters.
  - Emits a `gpu.launch_func` whose `kernel` attribute is a `SymbolRefAttr` to `@kernels::@kernel_<mangled>_0`.
2) `@cute.kernel` function `kernel`
  - Inside `gpu.module @kernels` a `func @kernel_<mangled>_0` is created with attributes `gpu.kernel`, `cute.kernel`, `nvvm.reqntid=[block.x, block.y, block.z]` (see `cutlass_dsl/cutlass.py:206–253` and `:316–420`).
  - The kernel body is the Python body lowered to Cute/Cute‑NVGPU ops via the high‑level Python wrappers in `cutlass/cute/*` and dialects in `cutlass/_mlir/dialects/*`.
3) Passes
  - `cute-to-nvvm` lowers everything and emits `gpu.binary` with CUBIN for the symbol `@kernels` and each `@kernel_*`.
  - `external-kernel-for-gpu-launch` adjusts the host for external kernel pointers.
4) Execution
  - Python loads the CUBIN (`jit_executor.py`), resolves `@kernel_*` symbols, and invokes the JITed host function which calls `gpu.launch_func` using those pointers.

Python → MLIR Type/Value Translation

- Numeric types: `cutlass/base_dsl/typing.py` defines `Integer`, `Float`, `Boolean`, etc., with `.mlir_type` and value wrappers.
- Tensor/Layout types: `cutlass/cute/typing.py` define `Tensor`, `Layout`, `ComposedLayout`, etc., each implementing:
  - `__get_mlir_types__()` → list of MLIR types for block arguments.
  - `__extract_mlir_values__()` → convert a Python wrapper into MLIR SSA values.
  - `__new_from_mlir_values__(values)` → rebuild the Python object from SSA values for call‑site plumbing.
- JIT ArgAdapters (`runtime/jit_arg_adapters.py`) convert common Python types to DSL types:
  - Scalars (int/float/bool) → `Int32`/`Float32`/`Boolean`.
  - Sequences (list/tuple) → elementwise conversion.
  - Reserved args `self`/`cls`, constexprs and type arguments are compile‑time only and don’t become MLIR block arguments.

Exposed MLIR APIs and Mapping to Upstream

- Import path: `import cutlass._mlir as mlir` (submodules: `ir`, `execution_engine`, `passmanager`, `dialects`, `extras.types as T`).
- The classes/functions in `cutlass._mlir.ir`, `execution_engine`, `passmanager` are thin wrappers around upstream MLIR C Python bindings:
  - `cutlass._mlir.ir` re‑exports upstream types (`Module`, `Context`, `Location`, `Attribute`, `Type`, `Operation`, …), plus adds a few attribute builders/helpers.
  - `cutlass._mlir.execution_engine.ExecutionEngine` extends lookup/invoke with `_mlir_ciface_` naming.
  - `cutlass._mlir.dialects.*` are generated from TableGen, mirroring upstream dialect APIs for `arith`, `func`, `scf`, `gpu`, `nvgpu`, `nvvm`, `llvm`, `vector`, plus Cutlass/Cute dialects.
- Practical usage mirrors upstream MLIR examples (create `Context`, `Module`, `InsertionPoint`, then emit ops through dialect modules).

Changing The Target Architecture (Cross‑Compilation)

- Recommended: set `CUTE_DSL_ARCH` environment variable prior to compilation:
  - Example: `CUTE_DSL_ARCH=sm_100 python examples/python/CuTeDSL/hopper/dense_gemm.py ...`
  - `dsl.py:930` injects this into the pass pipeline as `{ cubin-chip=sm_100 toolkitPath=... }`.
- Programmatic pipeline override:
  - Any `@cute.jit` call accepts `pipeline="..."` and `options="..."` via `cute.compile`/decorator plumbing. The default pipeline is:
    - `builtin.module(cute-to-nvvm{cubin-format=bin <compile-options>})`
  - You can pass a complete pipeline string; `BaseDSL.preprocess_pipeline` merges your options with env‑derived `toolkitPath` and `cubin-chip`.
- Notes on execution vs. compilation
  - You can cross‑compile on a machine without the target GPU. Loading/executing the CUBIN on incompatible hardware will fail with `CUDA_ERROR_NO_BINARY_FOR_GPU`. For “compile‑only”, use `cute.compile(..., options=..., pipeline=...)` and avoid executing.

Operational Tips

- Caching: generated MLIR bytecode can be saved/loaded (`base_dsl/cache_helpers.py`). In‑memory JIT cache is keyed by a hash of IR + env + compile options (`dsl.py:979`).
- Source locations: enable by default; disable with `CUTE_DSL_NO_SOURCE_LOCATION=1`. Host/kernel MLIR ops include `file`/`line` when `PRINT_IR=1` to help map errors to `dense_gemm.py` lines.
- Debugging NVVM errors: `Compiler.CompilationError` (`compiler.py`) extracts NVVM logs and prints an actionable checklist with toolkit/arch context.

Appendix: Where To Start Reading Code

- Decorators/runtime: `cutlass/cutlass_dsl/cutlass.py` and `cutlass/base_dsl/dsl.py`.
- AST preprocessor: `cutlass/base_dsl/ast_preprocessor.py`.
- MLIR bindings/dialects: `cutlass/_mlir/*` and `cutlass/_mlir/dialects/*`.
- CUDA driver bindings and kernel launch: `cutlass/base_dsl/runtime/cuda.py`, `jit_executor.py`.
- Hopper GEMM example: `examples/python/CuTeDSL/hopper/dense_gemm.py`.

