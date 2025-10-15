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

Quick Map: Stages → Entry Points (clickable)

| Stage | Entry point | File |
|---|---|---|
| Decorators | `CuTeDSL.jit` | [python/CuTeDSL/cutlass/base_dsl/dsl.py:494](../python/CuTeDSL/cutlass/base_dsl/dsl.py#L494) |
| Decorators | `CuTeDSL.kernel` | [python/CuTeDSL/cutlass/base_dsl/dsl.py:504](../python/CuTeDSL/cutlass/base_dsl/dsl.py#L504) |
| AST | `DSLPreprocessor` | [python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py:131](../python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py#L131) |
| IR build (host) | `_func` | [python/CuTeDSL/cutlass/base_dsl/dsl.py:1327](../python/CuTeDSL/cutlass/base_dsl/dsl.py#L1327) |
| IR build (kernel) | `_kernel_helper` | [python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py:318](../python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py#L318) |
| GPU module | `_build_gpu_module` | [python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py:206](../python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py#L206) |
| Pipeline str | `_get_pipeline` | [python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py:214](../python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py#L214) |
| Pipeline inject | `preprocess_pipeline` | [python/CuTeDSL/cutlass/base_dsl/dsl.py:973](../python/CuTeDSL/cutlass/base_dsl/dsl.py#L973) |
| Compile + JIT | `Compiler.compile_and_jit` | [python/CuTeDSL/cutlass/base_dsl/compiler.py:168](../python/CuTeDSL/cutlass/base_dsl/compiler.py#L168) |
| CUBIN load | `JitExecutor.update_jit_cuda_modules` | [python/CuTeDSL/cutlass/base_dsl/jit_executor.py:259](../python/CuTeDSL/cutlass/base_dsl/jit_executor.py#L259) |
| CUBIN walk | `walk_module_and_get_cubin_data` | [python/CuTeDSL/cutlass/base_dsl/jit_executor.py:330](../python/CuTeDSL/cutlass/base_dsl/jit_executor.py#L330) |
| Host call | `ExecutionEngine.lookup` | [python/CuTeDSL/cutlass/_mlir/execution_engine.py:13](../python/CuTeDSL/cutlass/_mlir/execution_engine.py#L13) |

Key Modules And Responsibilities

- Frontend and orchestration
  - `python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py:546` class `CuTeDSL` (subclasses `BaseDSL`).
  - `python/CuTeDSL/cutlass/base_dsl/dsl.py` core machinery: decorators, AST preprocessor hookup, IR building, compilation, caching, and execution.
  - `python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py` rewrites Python AST for structured control flow.
  - `python/CuTeDSL/cutlass/base_dsl/compiler.py` wraps MLIR PassManager + ExecutionEngine.
  - `python/CuTeDSL/cutlass/base_dsl/jit_executor.py` loads CUBINs, binds kernels (cuModule*), and invokes the JITed host function.
- MLIR bindings and dialects
  - `python/CuTeDSL/cutlass/_mlir/ir.py`, `passmanager.py`, `execution_engine.py`: thin wrappers around MLIR’s C Python bindings.
  - `python/CuTeDSL/cutlass/_mlir/dialects/*`: autogen’d ops/enums; includes upstream dialects (arith, func, scf, gpu, nvgpu, nvvm, llvm, vector) and Cutlass/Cute dialects.

End‑to‑End Pipeline (Detailed)

1) Decorators and AST Preprocessing

- `@cute.jit` on a host function/method wraps the call through `BaseDSL.jit_runner` which optionally enables AST transformation (default on):
  - `python/CuTeDSL/cutlass/base_dsl/dsl.py:468` `jit_runner` sets up preprocessor state.
  - `python/CuTeDSL/cutlass/base_dsl/dsl.py:556` `BaseDSL._preprocess_and_execute` runs `DSLPreprocessor.transform` (`python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py:445`) and materializes a new Python function (via `exec`, `python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py:238`).
  - The preprocessor rewrites supported control flow:
    - For loops: `range`, `range_constexpr`, `range_dynamic` → `loop_selector` machinery.
    - If/elif/else, while, assert, any/all, and compare ops (constants at `python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py:96`).

Example: decorator entry points

```python
# python/CuTeDSL/cutlass/base_dsl/dsl.py:494-512
@classmethod
def jit(cls, *dargs, **dkwargs):
    frame = inspect.currentframe().f_back
    main_dsl = cls._get_dsl()
    return main_dsl.jit_runner(main_dsl._func, frame, *dargs, **dkwargs)

@classmethod
def kernel(cls, *dargs, **dkwargs):
    frame = inspect.currentframe().f_back
    main_dsl = cls._get_dsl()
    return main_dsl.jit_runner(main_dsl._kernel_helper, frame, *dargs, **dkwargs)
```

2) IR Generation (Host func + GPU module + kernel func)

- MLIR context and module creation
  - `python/CuTeDSL/cutlass/base_dsl/dsl.py:1181` `generate_mlir` enters `with ir.Context(), ir.Location.unknown()`.
  - `python/CuTeDSL/cutlass/base_dsl/dsl.py:1067` `generate_original_ir` creates `ir.Module` and marks it with `gpu.container_module`.
  - `python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py:206` `_build_gpu_module(...)` creates a `gpu.module @kernels` region.

- Host function (the `@jit` body)
  - `python/CuTeDSL/cutlass/base_dsl/dsl.py:1094` creates a host `func.func` with `llvm.emit_c_interface`.
  - DSL converts Python arguments to MLIR types/values (`python/CuTeDSL/cutlass/base_dsl/dsl.py:842` `generate_mlir_function_types`, plus adapters in `python/CuTeDSL/cutlass/base_dsl/runtime/jit_arg_adapters.py`).

Example: building the host function op ([source](../python/CuTeDSL/cutlass/base_dsl/dsl.py#L1090))

```python
# python/CuTeDSL/cutlass/base_dsl/dsl.py:1090-1100
fop = func.FuncOp(function_name, (func_types, []), loc=loc)
fop.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
with ir.InsertionPoint(fop.add_entry_block()):
    ir_args, ir_kwargs = self.generate_execution_arguments(args, kwargs, fop, args_spec)
    result = funcBody(*ir_args, **ir_kwargs)
    func.ReturnOp([])
```

- Kernel function (the `@kernel` body)
  - The `@cute.kernel` wrapper (`python/CuTeDSL/cutlass/base_dsl/dsl.py:1536` `kernel_launcher`) computes a unique name and produces:
    - `func @kernel_*` inside `gpu.module @kernels` (via helper in `python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py:235`), annotated with `gpu.kernel`, `cute.kernel`, and NVVM attributes like `nvvm.reqntid`.
    - A host‐side `gpu.launch_func` referencing `@kernels::@kernel_*` with launch/cluster dims and dynamic shared memory (built in `python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py:377`–`python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py:420`).
  - Device body uses Cute/Cute‑NVGPU ops emitted by Python wrappers (`python/CuTeDSL/cutlass/_mlir/dialects/*`, `python/CuTeDSL/cutlass/cute/*`).

Example: generating launch_func ([source](../python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py#L377))

```python
# python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py:377-410
kernel_sym = ir.SymbolRefAttr.get(["kernels", kernel_name])
token = gpu.launch_func(
    gpu.AsyncTokenType.get() if is_async else None,
    cfg.async_deps,
    kernelSym,
    *cfg.grid,
    *cfg.block,
    kernelOperands,
    **dict(zip(("cluster_size_x","cluster_size_y","cluster_size_z"), tuple(cfg.cluster))),
    dynamic_shared_memory_size=cfg.smem,
)
```

3) Pass Pipeline and CUBIN Emission

- Default pipeline (`python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py:214`):
  - `builtin.module(cute-to-nvvm{cubin-format=bin ...})`
  - `python/CuTeDSL/cutlass/base_dsl/dsl.py:973` `preprocess_pipeline` injects `toolkitPath` and target arch (e.g., `cubin-chip=sm_90a`).
  - `python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py:222` appends `external-kernel-for-gpu-launch`.

Example: pipeline injection ([source](../python/CuTeDSL/cutlass/base_dsl/dsl.py#L973))

```python
# python/CuTeDSL/cutlass/base_dsl/dsl.py:973-1001
options = {"toolkitPath": self.envar.cuda_toolkit if self.envar.cuda_toolkit else None, self.pass_sm_arch_name: arch}
opt_str = " ".join(f"{k}={v}" for k,v in options.items() if v)
if opt_str:
    pattern = re.compile(r"{(.+)}"); match = pattern.search(pipeline)
    if match:
        opt_str = f"{{{match[1]} {opt_str}}}"; pipeline = re.sub(r"{.+}", opt_str, pipeline)
    else:
        pipeline = pipeline.rstrip(")") + f"{{{opt_str}}})"
```

4) JIT and Execution

- Compilation and JIT
  - `python/CuTeDSL/cutlass/base_dsl/dsl.py:1181` composes the pipeline string; `compiler.Compiler.compile_and_jit` (`python/CuTeDSL/cutlass/base_dsl/compiler.py:168`) runs PassManager and then wraps module in an MLIR `ExecutionEngine`.
  - Shared runtime libraries discovered via `EnvironmentVarManager` (see `python/CuTeDSL/cutlass/base_dsl/env_manager.py:286`).

- CUBIN extraction and cuModule binding
  - `python/CuTeDSL/cutlass/base_dsl/jit_executor.py:330` `walk_module_and_get_cubin_data` visits `gpu.binary`, decodes CUBIN bytes, and calls:
    - `cuModuleLoadData` (via `runtime/cuda.py`) to create a CUDA module.
    - `cuModuleGetFunction` to obtain a function pointer for `@kernel_*`.
    - Optionally sets attributes (non‑portable cluster size where supported).
  - Kernel pointers are appended to the packed arg list for the host function (`python/CuTeDSL/cutlass/base_dsl/jit_executor.py:210`).

- Host function invocation
  - `execution_engine.ExecutionEngine.lookup` returns a `CFUNCTYPE(void(void**))` entrypoint for functions with `llvm.emit_c_interface` (`python/CuTeDSL/cutlass/_mlir/execution_engine.py:13`).

Code excerpt (CUBIN walk and extraction) ([source](../python/CuTeDSL/cutlass/base_dsl/jit_executor.py#L330))

```python
# python/CuTeDSL/cutlass/base_dsl/jit_executor.py:330-354
def walk_module_and_get_cubin_data(self, module, sym, callback):
    def walk_gpu_binary_op(op):
        if op.name != "gpu.binary":
            return ir.WalkResult.ADVANCE
        s = io.BytesIO(); op.write_bytecode(s)
        cubin_data = s.getvalue()
        if sym.encode() not in cubin_data:
            return ir.WalkResult.ADVANCE
        func_sym = sym
        if sym == op.opview.sym_name.value and not sym.endswith("_kernel"):
            func_sym = sym.rsplit("_", 1)[0]
        cubin_data = cubin_data.split(b'bin = "')[1].split(b'">')[0]
        cubin_data = self._get_escaped_cubin_bytes(cubin_data)
        callback(sym, func_sym, cubin_data)
        return ir.WalkResult.ADVANCE
    module.operation.walk(walk_gpu_binary_op)
```

Inspecting Each Stage

- Environment variables (prefix `CUTE_DSL_`) are read by `EnvironmentVarManager` (`python/CuTeDSL/cutlass/base_dsl/env_manager.py`). Useful ones:
  - `CUTE_DSL_PRINT_AFTER_PREPROCESSOR=1` print transformed Python after AST rewriting.
  - `CUTE_DSL_PRINT_IR=1` print MLIR with debug info.
  - `CUTE_DSL_KEEP_IR=1` save MLIR bytecode to `cutlass_<func>.mlir` (see `python/CuTeDSL/cutlass/base_dsl/cache_helpers.py`).
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
  - To get CUBIN bytes: call `compiled.update_jit_cuda_modules(kernel_symbols)` or reuse `walk_module_and_get_cubin_data`.
- Programmatic peeking (internal but useful)
  - `BaseDSL.generate_original_ir(...)` builds the unlowered module; approximate by setting `CUTE_DSL_DRYRUN=1` and/or `CUTE_DSL_KEEP_IR=1`.
  - `BaseDSL.preprocess_pipeline(...)` shows how CUDA toolkit and arch options are stitched into the pipeline.

Example Walkthrough: dense_gemm.py (Hopper)

File: `examples/python/CuTeDSL/hopper/dense_gemm.py`

- Kernel class: `HopperWgmmaGemmKernel` at `examples/python/CuTeDSL/hopper/dense_gemm.py:203`.
- Host entrypoint: `__call__` decorated with `@cute.jit` at `examples/python/CuTeDSL/hopper/dense_gemm.py:372`.
  - Sets up data types, layouts, attributes, shared storage type, grid/block/cluster dims.
  - Calls the device kernel via `self.kernel(...).launch(grid=..., block=..., cluster=..., stream=...)` at the end of `__call__`.
- Device kernel: `kernel` decorated with `@cute.kernel` at `examples/python/CuTeDSL/hopper/dense_gemm.py:482`.
  - Emits Cute/Cute‑NVGPU ops for TMA copies, WGMMA (mainloop + epilogue), pipelines, mbarriers, etc.
- Compilation site: `compiled_gemm = cute.compile(gemm, mA, mB, mC, stream)` at `examples/python/CuTeDSL/hopper/dense_gemm.py:1508`.

Lowering Trace (conceptual mapping to IR)

1) `@cute.jit` function `__call__`
  - Host `func @cutlass___call__...` is created in the top‑level module and marked `llvm.emit_c_interface` (see `python/CuTeDSL/cutlass/base_dsl/dsl.py:1094`).
  - It constructs operands (memrefs/pointers/scalars) from Python types via `generate_mlir_function_types` and adapters.
  - Emits a `gpu.launch_func` whose `kernel` attribute is a `SymbolRefAttr` to `@kernels::@kernel_<mangled>_0`.
2) `@cute.kernel` function `kernel`
  - Inside `gpu.module @kernels` a `func @kernel_<mangled>_0` is created with attributes `gpu.kernel`, `cute.kernel`, `nvvm.reqntid=[block.x, block.y, block.z]` (see `python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py:206`–`python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py:253` and `:316`–`:420`).
3) Passes
  - `cute-to-nvvm` lowers everything and emits `gpu.binary` with CUBIN for `@kernels` and each `@kernel_*`.
  - `external-kernel-for-gpu-launch` adjusts the host for external kernel pointers.
4) Execution
  - Python loads the CUBIN, resolves `@kernel_*`, and invokes the JITed host function which calls `gpu.launch_func` using those pointers.

Python → MLIR Type/Value Translation

- Numeric types: `python/CuTeDSL/cutlass/base_dsl/typing.py` defines `Integer`, `Float`, `Boolean`, etc., with `.mlir_type` and value wrappers.
- Tensor/Layout types: `python/CuTeDSL/cutlass/cute/typing.py` define `Tensor`, `Layout`, `ComposedLayout`, etc., each implementing:
  - `__get_mlir_types__()` → list of MLIR types for block arguments.
  - `__extract_mlir_values__()` → convert a Python wrapper into MLIR SSA values.
  - `__new_from_mlir_values__(values)` → rebuild the Python object from SSA values for call‑site plumbing.
- JIT ArgAdapters (`python/CuTeDSL/cutlass/base_dsl/runtime/jit_arg_adapters.py`) convert common Python types to DSL types.

Exposed MLIR APIs and Mapping to Upstream

- Import path: `from cutlass._mlir import ir, passmanager, execution_engine; from cutlass._mlir.dialects import arith, func, scf, gpu, nvgpu, nvvm, llvm`.
- Classes/functions in `python/CuTeDSL/cutlass/_mlir/*` are thin wrappers around upstream MLIR C Python bindings; dialect files are auto‑generated from TableGen.

Changing The Target Architecture (Cross‑Compilation)

- Recommended: set `CUTE_DSL_ARCH` environment variable prior to compilation:
  - Example: `CUTE_DSL_ARCH=sm_100 python examples/python/CuTeDSL/hopper/dense_gemm.py ...`
  - `python/CuTeDSL/cutlass/base_dsl/dsl.py:973` injects this into the pass pipeline as `{ cubin-chip=sm_100 toolkitPath=... }`.
- Programmatic pipeline override:
  - Any `@cute.jit` call accepts `pipeline="..."` and `options="..."` via `cute.compile`/decorator plumbing. The default pipeline is:
    - `builtin.module(cute-to-nvvm{cubin-format=bin <compile-options>})`
  - You can pass a complete pipeline string; `BaseDSL.preprocess_pipeline` merges your options with env‑derived `toolkitPath` and `cubin-chip`.
- Notes on execution vs. compilation
  - Cross‑compile on a machine without the target GPU. Loading/executing the CUBIN on incompatible hardware will fail with `CUDA_ERROR_NO_BINARY_FOR_GPU`. For compile‑only, use `cute.compile(..., ...)` and don’t run.

Operational Tips

- Caching: generated MLIR bytecode can be saved/loaded (`python/CuTeDSL/cutlass/base_dsl/cache_helpers.py`). In‑memory JIT cache is keyed by a hash of IR + env + compile options (`python/CuTeDSL/cutlass/base_dsl/dsl.py:1012`).
- Source locations: enable by default; disable with `CUTE_DSL_NO_SOURCE_LOCATION=1`. Host/kernel MLIR ops include `file`/`line` when `PRINT_IR=1` to help map errors to `dense_gemm.py` lines.
- Debugging NVVM errors: `Compiler.CompilationError` (`python/CuTeDSL/cutlass/base_dsl/compiler.py`) extracts NVVM logs and prints an actionable checklist with toolkit/arch context.

Appendix: Where To Start Reading Code

- Decorators/runtime: `python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py:546`, `python/CuTeDSL/cutlass/base_dsl/dsl.py:494`, `python/CuTeDSL/cutlass/base_dsl/dsl.py:504`.
- AST preprocessor: `python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py:131`, `python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py:445`.
- MLIR bindings/dialects: `python/CuTeDSL/cutlass/_mlir/*`, `python/CuTeDSL/cutlass/_mlir/dialects/*`.
- CUDA driver bindings and kernel launch: `python/CuTeDSL/cutlass/base_dsl/runtime/cuda.py`, `python/CuTeDSL/cutlass/base_dsl/jit_executor.py:259`.
- Hopper GEMM example: `examples/python/CuTeDSL/hopper/dense_gemm.py:203`, `examples/python/CuTeDSL/hopper/dense_gemm.py:372`, `examples/python/CuTeDSL/hopper/dense_gemm.py:482`, `examples/python/CuTeDSL/hopper/dense_gemm.py:1508`.

Additional Diagram: Data Flow from Host to Device

```
Host func.func           Passes            GPU module          CUDA Driver
   |  build args            |                  |                    |
   v                        v                  v                    v
 generate_mlir() --> cute-to-nvvm --> gpu.binary(CUBIN) --> cuModuleLoadData --> kernel ptr
   |                                                                  |
   +-- ExecutionEngine.lookup/_mlir_ciface_* -------------------------+
```
