CuTeDSL End-to-End Traces

This file traces, with code references, the full call-paths when running a program that uses CuTeDSL via `@cute.jit` and `@cute.kernel`.

Legend for references: `path:LINE_START–LINE_END`.

1) `@cute.jit` (host entrypoint) flow

- Public surface resolves `cute.jit` to `CuTeDSL.jit`:
  - `cutlass_package/cute/__init__.py:...` — `jit = _dsl.CuTeDSL.jit` (export alias from CuTe DSL).

- Concrete DSL instantiation (`CuTeDSL`) wires MLIR pass manager and execution engine:
  - `cutlass_package/cutlass_dsl/cutlass.py:408–422` — `class CuTeDSL(CutlassBaseDSL)`, `compiler_provider = compiler.Compiler(passmanager, execution_engine)`.
  Excerpt:
  ```python
  # cutlass_package/cutlass_dsl/cutlass.py
  class CuTeDSL(CutlassBaseDSL):
      def __init__(self):
          name = "CUTE_DSL"
          compiler_provider = compiler.Compiler(passmanager, execution_engine)
          pass_sm_arch_name = "cubin-chip"
          super().__init__(name, compiler_provider, pass_sm_arch_name, preprocess=True)
  ```

- Decorator entry point in the base DSL:
  - `cutlass_package/base_dsl/dsl.py:485–493` — `@classmethod def jit(...)` captures the caller frame, creates a singleton DSL instance, and returns `main_dsl.jit_runner(main_dsl._func, frame, ...)`.
  - `cutlass_package/base_dsl/dsl.py:457–483` — `jit_runner(...)` builds `jit_runner_decorator` that:
    - attaches DSL object to the Python function and records the decorator frame;
    - lazily applies AST preprocessing (if enabled);
    - returns `jit_wrapper`.
  - `cutlass_package/base_dsl/dsl.py:473–477` — `jit_wrapper` calls `BaseDSL._preprocess_and_execute(func)` to materialize a callable, then invokes the executor (`_func`).
  Excerpts:
  ```python
  # cutlass_package/base_dsl/dsl.py
  @classmethod
  def jit(cls, *dargs, **dkwargs):
      frame = inspect.currentframe().f_back
      main_dsl = cls._get_dsl()
      return main_dsl.jit_runner(main_dsl._func, frame, *dargs, **dkwargs)

  def jit_runner(self, executor, frame, *dargs, **dkwargs):
      def jit_runner_decorator(func):
          func._dsl_object = self
          if self.enable_preprocessor and BaseDSL._can_preprocess(**dkwargs):
              func._decorator_frame = frame
              func._transformed_ast = None
          @wraps(func)
          def jit_wrapper(*args, **kwargs):
              func_ptr = BaseDSL._preprocess_and_execute(func)
              return executor(func_ptr, *args, **kwargs)
          return jit_wrapper
      ...
  ```
  - `cutlass_package/base_dsl/dsl.py:436–460` — `_preprocess_and_execute`: runs `DSLPreprocessor.transform(...)` on first call, compiles the transformed AST to a Python function, unwraps to the original function if needed, and returns a `DSLCallable` wrapper.
  - AST preprocessor details:
    - `cutlass_package/base_dsl/ast_preprocessor.py:120–141` — `class DSLPreprocessor(ast.NodeTransformer)` features for `for/if/while` lowering.
    - `cutlass_package/base_dsl/ast_preprocessor.py:374–387` — `.transform(...)` builds an AST module for execution.

- Host executor `_func` builds MLIR, compiles, JITs, and runs:
  - `cutlass_package/base_dsl/dsl.py:1311–1360` — `_func(...)` canonicalizes args, mangles the symbol, and calls `generate_mlir(...)`.
  - `cutlass_package/base_dsl/dsl.py:1217–1266` — `generate_mlir(...)`:
    - converts Python args to MLIR types/values;
    - calls `generate_original_ir(...)` to create the IR module with a `func.func @<mangled>` and a `gpu.module @kernels` (always present);
    - computes a module hash (including environment and compile options) and caches or compiles as needed via `compile_and_cache(...)`;
    - if not `compile_only`, invokes the compiled program with `JitExecutor`.
  - `cutlass_package/base_dsl/dsl.py:1083–1146` — `generate_original_ir(...)` builds:
    - a `builtin.module` tagged as GPU container;
    - a `gpu.module @kernels` via `_build_gpu_module(...)` (see CuTeDSL implementation below);
    - a public `func.func @<mangled>` with `llvm.emit_c_interface` and IR for the Python body;
    - returns both the IR module and the Python‑level result (if any).
  - `cutlass_package/cutlass_dsl/cutlass.py:150–167` — `CutlassBaseDSL._get_pipeline` chooses default pipeline: `builtin.module(cute-to-nvvm{cubin-format=bin ...})`.
  - `cutlass_package/cutlass_dsl/cutlass.py:170–173` — `preprocess_pipeline` appends `,external-kernel-for-gpu-launch)`.
  - `cutlass_package/base_dsl/dsl.py:951–982` — `compile_and_jit(...)` delegates to the `Compiler` provider.
  - `cutlass_package/base_dsl/compiler.py:135–166` — `Compiler.compile(...)` parses the pass pipeline with `PassManager.parse`, runs it, and on success `jit(...)` wraps the module in an `ExecutionEngine`.
- `cutlass_package/_mlir/execution_engine.py:1–38` — `class ExecutionEngine` extends MLIR’s engine; `.lookup(name)` returns a `ctypes` callable for `_mlir_ciface_<name>`, `.invoke(name, ...)` packs `void*` args.
  Code (excerpt):
  ```python
  # cutlass_package/_mlir/execution_engine.py
  class ExecutionEngine(_execution_engine.ExecutionEngine):
      def lookup(self, name):
          func = self.raw_lookup("_mlir_ciface_" + name)
          if not func:
              raise RuntimeError("Unknown function " + name)
          prototype = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
          return prototype(func)
  ```
  - `cutlass_package/base_dsl/dsl.py:1147–1192` — `compile_and_cache(...)` constructs `JitExecutor(engine, capi_func, ...)` and preloads CUDA kernels from IR (details below).
  - `cutlass_package/base_dsl/jit_executor.py:228–257` — `JitExecutor.__call__/run_compiled_program` converts Python args to C pointers, appends CUDA kernel pointers if present, and calls the function pointer returned by the `ExecutionEngine`.

2) `@cute.kernel` (device kernel) flow

- Public surface resolves `cute.kernel` to `CuTeDSL.kernel`:
  - `cutlass_package/cute/__init__.py:...` — `kernel = _dsl.CuTeDSL.kernel`.

- Base decorator entry matches `@cute.jit` but calls the kernel executor:
  - `cutlass_package/base_dsl/dsl.py:495–504` — `@classmethod def kernel(...)` returns `main_dsl.jit_runner(main_dsl._kernel_helper, frame, ...)`.
  - `cutlass_package/base_dsl/dsl.py:457–483` — same `jit_runner` flow; ultimately calls `CuTeDSL._kernel_helper` with a possibly preprocessed body.

- CuTe‑specific kernel scaffolding — two pieces:
  - `cutlass_package/cutlass_dsl/cutlass.py:232–277` — `CuTeDSL._kernel_helper` returns a `KernelLauncher` that:
    - validates and captures the user kernel body and arguments;
    - on `.launch(...)`, delegates to `BaseDSL.kernel_launcher(...)` to emit IR for the GPU kernel and the `gpu.launch_func` at the call site.
- `cutlass_package/base_dsl/dsl.py:1559–1712` — `BaseDSL.kernel_launcher(...)`:
    - mangles a unique kernel symbol (e.g., `kernel_<host_mangled>_<ordinal>`);
    - builds kernel operands/types (`generate_kernel_operands_and_types(...)`)
      and a kernel function op via the provided helper;
    - encloses kernel bodies in the `gpu.module @kernels` via `_enter_gpu_module()`;
    - emits a `gpu.launch_func` call‑site referencing `@kernels::<kernel_sym>` with grid/block/cluster/smem/args;
    - returns both the kernel’s function body result and the launch op.
  - The helper for CuTe constructs the function and launch op using MLIR dialects:
    - `cutlass_package/cutlass_dsl/cutlass.py:232–277` — `_CutlassIrKernelGenHelper.generate_func_op/ret_op/get_func_body_start/generate_launch_op`:
      - wraps the user body in `func.func @<kernel_sym>` without returns;
      - emits `gpu.launch_func` with optional `gpu.AsyncTokenType` when async;
      - converts launch sizes to `index` via `to_index()` and sets dynamic smem.
    Excerpt:
    ```python
    # cutlass_package/cutlass_dsl/cutlass.py
    token = gpu.launch_func(
        gpu.AsyncTokenType.get() if is_async else None,
        cfg.async_deps,
        kernelSym,
        *cfg.grid, *cfg.block,
        kernelOperands,
        **dict(zip(("cluster_size_x","cluster_size_y","cluster_size_z"), tuple(cfg.cluster))),
        dynamic_shared_memory_size=cfg.smem,
    )
    ```

- GPU module and attributes
  - `cutlass_package/cutlass_dsl/cutlass.py:150–157` — `_build_gpu_module` constructs `gpu.module @kernels` and applies attributes supplied by `_generate_kernel_attrs`.
  - `cutlass_package/cutlass_dsl/cutlass.py:178–191` — `_generate_kernel_attrs` sets `nvvm.reqntid` (required block dims) and optionally `nvvm.minctasm`.

- Pipeline and codegen
  - Default pipeline becomes e.g.: `builtin.module(cute-to-nvvm{cubin-format=bin opt-level=3 ...},external-kernel-for-gpu-launch)`; see `cutlass_package/cutlass_dsl/cutlass.py:150–173`.
  - `cutlass_package/base_dsl/compiler.py:135–166` — `PassManager.parse().run()` lowers from CuTe/gpu to NVVM and emits `gpu.binary` with an embedded cubin blob.

- CUDA module binding and function pointer extraction
  - `cutlass_package/base_dsl/jit_executor.py:330–357` — `walk_module_and_get_cubin_data(...)` walks IR to find `gpu.binary`, extracts the `bin = "..."` payload, unescapes bytes, and calls the callback.
- `cutlass_package/base_dsl/jit_executor.py:260–297` — `update_jit_cuda_modules(...)` loads the cubin via CUDA Driver, looks up the kernel function symbol, and sets function attributes (e.g., `CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED` on >= 11.8), then remembers `kernel_ptr` as an extra argument to the JIT entrypoint.
  Excerpt:
  ```python
  # cutlass_package/base_dsl/jit_executor.py
  cubin_module = cuda_helpers.load_cubin_module_data(cubin_data)
  kernel_ptr = cuda_helpers.get_kernel_function(cubin_module, func_sym)
  if cuda_driver_version >= 11080:
      cuda_helpers.set_kernel_attribute(
          kernel_ptr,
          cuda_helpers.cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
          1,
      )
  ```
- `cutlass_package/base_dsl/runtime/cuda.py:288–309` — `load_cubin_module_data(...)` and `get_kernel_function(...)` wrappers over `cuModuleLoadData` / `cuModuleGetFunction`.
- `cutlass_package/base_dsl/runtime/cuda.py:445–451` — `set_kernel_attribute(...)` wrapper for `cuFuncSetAttribute`.
  Excerpt:
  ```python
  # cutlass_package/base_dsl/runtime/cuda.py
  def launch_kernel(kernel, grid_dims, block_dims, stream, smem_size=0, kernel_args=None):
      checkCudaErrors(cuda.cuLaunchKernel(
          kernel,
          grid_dims[0], grid_dims[1], grid_dims[2],
          block_dims[0], block_dims[1], block_dims[2],
          smem_size,
          stream,
          kernel_args,
          0,
      ))
  ```

- Execution of the JIT entrypoint
  - Host `func.func` takes all marshalled parameters plus a trailing `kernel_ptr` for each referenced kernel; `JitExecutor.get_invoke_packed_args` appends them.
  - `cutlass_package/_mlir/execution_engine.py:12–24` — `ExecutionEngine.lookup` returns a `ctypes` function pointer taking `void**`; `JitExecutor.run_compiled_program` packs and calls it.
  - Inside the generated call‑site, MLIR’s lowered code launches the kernel via the provided `kernel_ptr` (enabled by the `external-kernel-for-gpu-launch` option in the pipeline).

Auxiliary: types, ABI, and helpers

- `cutlass_package/base_dsl/typing.py` — maps DSL types to MLIR types; `get_c_pointers` and adapters produce C ABI arguments.
- `cutlass_package/_mlir/runtime/np_to_memref.py` — C‑ABI memref structs for arrays.
- `cutlass_package/base_dsl/compiler.py:187–233` — `CompileOptions` influence pipeline options (e.g., `opt-level`, device assertions) that are appended to the pass pipeline string.

Where the user edits plug in

- User code: define kernels with `@cute.kernel` and host entrypoints with `@cute.jit`.
- Launch: call `kernel(...).launch(grid=[...], block=[...], smem=..., stream=...)` or `kernel(...)(grid=..., block=...)`.
- Optional: AST preprocessor can be disabled with `@CuTeDSL.jit(preprocess=False)` (example at `cutlass_package/cute/testing.py:151`).
