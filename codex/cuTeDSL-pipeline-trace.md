# CuTeDSL Pipeline Trace: `experiments/rmsnorm.py:325`

## Scope

This trace follows the exact compile entrypoint:

- compile call: [experiments/rmsnorm.py#L325](../experiments/rmsnorm.py#L325)
- traced callable: `CtaNorm.__call__` at [experiments/rmsnorm.py#L84](../experiments/rmsnorm.py#L84)
- launched kernel: `CtaNorm.kernel` at [experiments/rmsnorm.py#L115](../experiments/rmsnorm.py#L115)

Questions answered in this document:

1. How user Python is parsed
2. How parsed code becomes MLIR
3. How AST preprocessing and tracing are combined
4. `@cute.jit` vs `@cute.kernel` compile-path differences
5. MLIR pipeline stages
6. PTX/CUBIN generation details
7. `enable_tvm_ffi=True/False` differences
8. Whether `nvcc`/`ptxas` are used

## Code Map

- Entry script compile site: [experiments/rmsnorm.py#L306](../experiments/rmsnorm.py#L306), [experiments/rmsnorm.py#L325](../experiments/rmsnorm.py#L325)
- `cute.compile` alias: [python/CuTeDSL/cutlass/cute/__init__.py#L207](../python/CuTeDSL/cutlass/cute/__init__.py#L207)
- `CompileCallable._compile`: [python/CuTeDSL/cutlass/base_dsl/compiler.py#L579](../python/CuTeDSL/cutlass/base_dsl/compiler.py#L579)
- Decorator wiring (`jit`/`kernel`): [python/CuTeDSL/cutlass/base_dsl/dsl.py#L470](../python/CuTeDSL/cutlass/base_dsl/dsl.py#L470), [python/CuTeDSL/cutlass/base_dsl/dsl.py#L478](../python/CuTeDSL/cutlass/base_dsl/dsl.py#L478)
- Lazy preprocess + wrapper dispatch: [python/CuTeDSL/cutlass/base_dsl/dsl.py#L401](../python/CuTeDSL/cutlass/base_dsl/dsl.py#L401), [python/CuTeDSL/cutlass/base_dsl/dsl.py#L428](../python/CuTeDSL/cutlass/base_dsl/dsl.py#L428)
- AST preprocessor entry: [python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py#L761](../python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py#L761)
- Host compile entry (`_func`): [python/CuTeDSL/cutlass/base_dsl/dsl.py#L1505](../python/CuTeDSL/cutlass/base_dsl/dsl.py#L1505)
- IR creation by tracing function execution: [python/CuTeDSL/cutlass/base_dsl/dsl.py#L1102](../python/CuTeDSL/cutlass/base_dsl/dsl.py#L1102)
- Kernel lowering helper: [python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py#L766](../python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py#L766)
- Pipeline string construction: [python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py#L254](../python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py#L254)
- Compile+JIT: [python/CuTeDSL/cutlass/base_dsl/dsl.py#L1165](../python/CuTeDSL/cutlass/base_dsl/dsl.py#L1165)
- Compiler passmanager invocation: [python/CuTeDSL/cutlass/base_dsl/compiler.py#L136](../python/CuTeDSL/cutlass/base_dsl/compiler.py#L136)
- TVM FFI hook path: [python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py#L462](../python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py#L462)

## Sequence Diagram

```text
User Script (rmsnorm.py)
  |
  | cute.compile(layernorm, _y, _x, _weight, _bias)
  v
CompileCallable._compile
  |-- normalize callable (class instance -> __call__.__func__)
  |-- set compile_only=True, no_cache=True
  |-- BaseDSL._lazy_initialize_dsl
  |-- BaseDSL._preprocess_and_replace_code
  |      \-> DSLPreprocessor.transform + exec (AST rewrite)
  v
BaseDSL._func (host JIT path)
  |-- canonicalize args + infer MLIR arg types
  |-- generate_original_ir
  |      \-> execute traced host function body with IR proxy args
  |            \-> @cute.kernel call -> _kernel_helper -> KernelLauncher
  |                   \-> create cuda.kernel op + cuda.launch_ex callsite op
  v
IR Module (host func + gpu module + kernel func)
  |
  | compile_and_cache
  |-- resolve pipeline: builtin.module(cute-to-nvvm{...})
  |-- PassManager.run(module.operation)
  |      \-> MLIR lowering + cuda-to-binary
  v
ExecutionEngine + CudaDialectJitCompiledFunction
  |
  | (compile_only=True in cute.compile path)
  v
Return compiled function handle (+ artifacts if enabled)
```

## Module/Class Map

```text
cutlass.cute (public API)
  |
  +-- compile -> CompileCallable
  +-- jit     -> CuTeDSL.jit  (BaseDSL.jit wrapper)
  +-- kernel  -> CuTeDSL.kernel (BaseDSL.kernel wrapper)

CompileCallable
  |
  +-- calls BaseDSL._func on decorated function
        |
        +-- BaseDSL (generic JIT framework)
        |     +-- DSLPreprocessor (AST rewrite)
        |     +-- generate_original_ir (trace execution into MLIR)
        |     +-- compile_and_cache
        |
        +-- CuTeDSL (CutlassBaseDSL specialization)
              +-- _kernel_helper / KernelLauncher (cuda kernel lowering)
              +-- _get_pipeline -> cute-to-nvvm pass pipeline
              +-- compile_and_cache override (optional TVM FFI hook)

Backend runtime objects
  |
  +-- Compiler (PassManager + ExecutionEngine)
  +-- CudaDialectJitCompiledFunction (callable compiled function)
  +-- JitExecutor (invocation marshalling + runtime launch)
```

## Flowchart

```text
[Python function + decorators]
      |
      v
[AST preprocess?]
  yes -> rewrite loops/ifs/calls into helper-based form
  no  -> keep original function source
      |
      v
[Trace execution with IR proxy args]
      |
      +--> arithmetic ops emit dialect ops
      +--> @cute.kernel emits kernel op + launch op
      +--> helper-decorated control flow emits scf.* ops
      |
      v
[Build MLIR module]
      |
      v
[Run cute-to-nvvm pipeline]
      |
      v
[cuda-to-binary pass]
      |
      +--> PTX text
      +--> cubin/fatbin binary
      |
      v
[ExecutionEngine + callable handle]
```

## Frame-by-Frame Trace

## Frame 0: Runtime entry in `rmsnorm.py`

Source: [experiments/rmsnorm.py#L306](../experiments/rmsnorm.py#L306)

```python
_x = from_dlpack(x, assumed_align=16, enable_tvm_ffi=True)
...
layernorm = CtaNorm(N, norm_type, threads_per_cta)
compiler: CompileCallable = cute.compile
compiled_func = compiler(layernorm, _y, _x, _weight, _bias, options=options)
```

State before:

- `layernorm` is a class instance whose `__call__` is decorated with `@cute.jit`
- `CtaNorm.__call__` launches a `@cute.kernel` function (`self.kernel(...).launch(...)`)
- compile options argument is `None` at callsite here

State after:

- `cute.compile` returns a compiled callable object (`CudaDialectJitCompiledFunction`)
- script exits early at [experiments/rmsnorm.py#L341](../experiments/rmsnorm.py#L341), so this path is compile-only inspection

## Frame 1: `cute.compile` -> `CompileCallable._compile`

Sources:

- alias export: [python/CuTeDSL/cutlass/cute/__init__.py#L207](../python/CuTeDSL/cutlass/cute/__init__.py#L207)
- implementation: [python/CuTeDSL/cutlass/base_dsl/compiler.py#L579](../python/CuTeDSL/cutlass/base_dsl/compiler.py#L579)

```python
kwargs["compile_only"] = True
kwargs["no_cache"] = True
...
elif inspect.isclass(type(func)) and hasattr(func, "__call__"):
    args = [func] + list(args)
    func = func.__call__.__func__
...
BaseDSL._lazy_initialize_dsl(func)
...
func._dsl_object._preprocess_and_replace_code(func)
return func._dsl_object._func(func, *args, **kwargs)
```

Important behavior:

- `cute.compile` forces `compile_only=True` and `no_cache=True` (explicit compilation path)
- class instance is normalized to underlying function object (`__call__.__func__`) and instance is inserted as first arg
- compile options are installed on DSL singleton (`func._dsl_object.compile_options = ...`)

## Frame 2: `@cute.jit` wrapper dispatch

Source: [python/CuTeDSL/cutlass/base_dsl/dsl.py#L428](../python/CuTeDSL/cutlass/base_dsl/dsl.py#L428)

```python
def jit_runner(cls, executor_name, frame, *dargs, **dkwargs):
    ...
    def jit_wrapper(*args, **kwargs):
        BaseDSL._preprocess_and_replace_code(func)
        return getattr(func._dsl_object, executor_name)(func, *args, **kwargs)
```

`@cute.jit` and `@cute.kernel` are both the same wrapper mechanism, with different `executor_name`:

- `@cute.jit` -> `_func`
- `@cute.kernel` -> `_kernel_helper`

## Frame 3: AST preprocessing (parse -> transform -> re-exec)

Sources:

- preprocessor run: [python/CuTeDSL/cutlass/base_dsl/dsl.py#L1419](../python/CuTeDSL/cutlass/base_dsl/dsl.py#L1419)
- parser entry: [python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py#L541](../python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py#L541)
- transform entry: [python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py#L761](../python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py#L761)

```python
lines, start_line = inspect.getsourcelines(function_pointer)
tree = ast.parse(dedented_source, filename=file_name)
...
transformed_tree = self.visit(tree)
...
code_object = compile(transformed_ast, filename=file_name, mode="exec")
exec(code_object, exec_globals)
```

What is parsed:

- Python source of the decorated function (not bytecode)
- decorators, loops, if/while, builtin calls, typed args

What is rewritten:

- dynamic `for/if/while` into helper-decorated region functions
- control-flow symbol checks (`cf_symbol_check`)
- bool/comparison/assert handling into helper functions
- removes top-level DSL decorators from transformed function

Example rewrite mechanics:

Source: [python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py#L982](../python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py#L982)

```python
decorator = ast.Call(
    func=_create_module_attribute(self.DECORATOR_FOR_STATEMENT),
    args=[start, stop, step],
    keywords=[... write_args ...],
)
```

Then runtime helper dispatch is provided by:

- generic helper APIs: [python/CuTeDSL/cutlass/base_dsl/ast_helpers.py#L184](../python/CuTeDSL/cutlass/base_dsl/ast_helpers.py#L184)
- CuTe-specific dynamic builders wired in: [python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py#L2239](../python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py#L2239)

## Frame 4: Host compile path (`BaseDSL._func`)

Source: [python/CuTeDSL/cutlass/base_dsl/dsl.py#L1505](../python/CuTeDSL/cutlass/base_dsl/dsl.py#L1505)

```python
function_name = self.mangle_name(...)
self.compile_options.apply_envar_settings(self.envar, function_name)
result = self.generate_mlir(
    funcBody, canonicalized_kwargs, function_name, ...,
    compile_only=compile_only,
)
```

Pre-IR steps:

- argument binding and canonicalization
- name mangling with compile-time/static args
- env + compile-option merge (`CUTE_DSL_*` + explicit options)

## Frame 5: Argument tracing to MLIR function signature

Source: [python/CuTeDSL/cutlass/base_dsl/dsl.py#L700](../python/CuTeDSL/cutlass/base_dsl/dsl.py#L700)

```python
jit_exec_arg, jit_arg_type, jit_arg_attr = self._generate_jit_func_args_for_known_types(...)
...
jit_exec_arg.extend(get_c_pointers(arg))
jit_arg_type.extend(get_mlir_types(arg))
```

For `cute.Tensor`/runtime tensor wrappers:

- host ABI pointers come from `__c_pointers__` on runtime tensor objects
- dynamic expression extraction for device-side objects uses `__extract_mlir_values__` path

Relevant runtime tensor implementation:

- [python/CuTeDSL/cutlass/cute/runtime.py#L382](../python/CuTeDSL/cutlass/cute/runtime.py#L382)
- [python/CuTeDSL/cutlass/cute/runtime.py#L746](../python/CuTeDSL/cutlass/cute/runtime.py#L746)

## Frame 6: Trace execution to build original IR

Source: [python/CuTeDSL/cutlass/base_dsl/dsl.py#L1102](../python/CuTeDSL/cutlass/base_dsl/dsl.py#L1102)

```python
module = ir.Module.create(...)
self._build_gpu_module(...)
fop = func.FuncOp(function_name, (func_types, ret_types), loc=loc)
...
ir_args, ir_kwargs = self.generate_execution_arguments(args, kwargs, fop, args_spec)
result = funcBody(*ir_args, **ir_kwargs)
func.ReturnOp(default_ret_values, loc=loc)
```

This is where AST + tracing combine:

- AST preprocessing already rewrote control-flow structure into helper calls/functions
- executing transformed `funcBody` with IR proxy args records emitted operations
- helper-decorated control-flow regions create structured `scf.*` ops at trace-time
- arithmetic/tensor ops emit dialect ops through overloaded DSL functions

## Frame 7: `@cute.kernel` lowering path inside traced host function

Host function (`CtaNorm.__call__`) calls:

```python
self.kernel(...).launch(grid=[M, 1, 1], block=[...])
```

`@cute.kernel` resolves to `_kernel_helper` path:

- decorator endpoint: [python/CuTeDSL/cutlass/base_dsl/dsl.py#L478](../python/CuTeDSL/cutlass/base_dsl/dsl.py#L478)
- CuTe specialization: [python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py#L766](../python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py#L766)

Lowering actions:

1. Create `cuda.kernel` function op for device body  
   Source: [python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py#L1070](../python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py#L1070)
2. Trace kernel body into that op's entry block
3. Emit host call-site launch op (`cuda.launch_ex`) in host function  
   Source: [python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py#L701](../python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py#L701)

Net result: one IR module contains both host launch function and outlined GPU kernel.

## Frame 8: Dynamic control-flow lowering after preprocessing

Key wiring:

- helper API layer: [python/CuTeDSL/cutlass/base_dsl/ast_helpers.py#L184](../python/CuTeDSL/cutlass/base_dsl/ast_helpers.py#L184)
- CuTe SCF builders: [python/CuTeDSL/cutlass/cutlass_dsl/cutlass_ast_decorators.py#L281](../python/CuTeDSL/cutlass/cutlass_dsl/cutlass_ast_decorators.py#L281)

Example: dynamic loop builder creates `scf.ForOp`:

```python
for_op = scf.ForOp(start_, stop_, step_, list(dyn_yield_ops))
```

Example: dynamic branch builder creates `scf.IfOp`:

```python
if_op = scf.IfOp(pred_.ir_value(), hasElse=(else_block is not None), results_=result_types)
```

This is the concrete mechanism for "AST + tracing in conjunction":

- AST pass creates call structure for region functions and carried variables
- tracing execution of these helper calls emits `scf` IR objects and region bodies

## Frame 9: MLIR pipeline construction and stages

Pipeline string source:

- default pipeline builder: [python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py#L254](../python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py#L254)

```python
return "builtin.module(cute-to-nvvm{cubin-format=bin " + self.compile_options.to_str() + "})"
```

Then options are injected:

- architecture option + dialect flags: [python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py#L266](../python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py#L266)
- compile execution: [python/CuTeDSL/cutlass/base_dsl/compiler.py#L136](../python/CuTeDSL/cutlass/base_dsl/compiler.py#L136)

Observed pass stages (from dumped pipeline artifacts under `experiments/*/builtin_module_no-symbol-name`):

- `0_cute-remove-static-args.mlir`
- `1_materialize-tma-multicast.mlir`
- `2_software-pipeline.mlir`
- `3_cute-desugar.mlir`
- `4_cute-expand-ops.mlir`
- `5_cute-fold-static.mlir`
- `6_insert-range-for-threadids.mlir`
- `7_canonicalize.mlir`
- `8_cse.mlir`
- `9_loop-invariant-code-motion.mlir`
- `10_loop-strength-reduction.mlir`
- `11_convert-vector-to-scf.mlir`
- `12_convert-cute-types-in-scf-ops.mlir`
- `13_convert-scf-to-cf.mlir`
- `14_arith-expand.mlir`
- `15_convert-math-to-funcs.mlir`
- `16_gpu-launch-sink-index-computations.mlir`
- `17_convert-to-llvm.mlir`
- `18_reconcile-unrealized-casts.mlir`
- `19_gpu-kernel-outlining.mlir`
- `20_convert-scf-to-cf.mlir`
- `21_reconcile-unrealized-casts.mlir`
- `22_convert-to-llvm.mlir`
- `23_canonicalize.mlir`
- `24_cse.mlir`
- `25_convert-vector-to-llvm.mlir`
- `26_convert-nvvm-to-llvm.mlir`
- `27_finalize-memref-to-llvm.mlir`
- `28_expand-strided-metadata.mlir`
- `29_convert-arith-to-llvm.mlir`
- `30_store-gpu-launch-attribute.mlir`
- `31_gpu-to-llvm.mlir`
- `32_store-gpu-launch-attribute.mlir`
- `33_convert-math-to-llvm.mlir`
- `34_canonicalize.mlir`
- `35_cse.mlir`
- `36_reconcile-unrealized-casts.mlir`
- `38_cuda-to-binary.mlir` (or nearby index depending on options)
- `39_convert-to-llvm.mlir`
- `40_reconcile-unrealized-casts.mlir`
- `41_cuda-error-handling.mlir`

Kernel-submodule pipeline (`gpu_module_kernels/...`) includes:

- `cuda-compute-target`
- `convert-fastmath-ops`
- `convert-gpu-to-nvvm`

Reference directories used:

- [experiments/sol_kernel/builtin_module_no-symbol-name](../experiments/sol_kernel/builtin_module_no-symbol-name)
- [experiments/ssa_dump/builtin_module_no-symbol-name](../experiments/ssa_dump/builtin_module_no-symbol-name)
- [experiments/tvm-ffi/tvm_example/builtin_module_no-symbol-name](../experiments/tvm-ffi/tvm_example/builtin_module_no-symbol-name)

## Frame 10: PTX and CUBIN generation

Configuration knobs:

- compile options include `ptx-options`, `dump-ptx-path`, `dump-cubin-path`, `cubin-chip`  
  Source: [python/CuTeDSL/cutlass/base_dsl/compiler.py#L315](../python/CuTeDSL/cutlass/base_dsl/compiler.py#L315), [python/CuTeDSL/cutlass/base_dsl/compiler.py#L327](../python/CuTeDSL/cutlass/base_dsl/compiler.py#L327), [python/CuTeDSL/cutlass/base_dsl/compiler.py#L340](../python/CuTeDSL/cutlass/base_dsl/compiler.py#L340)

Artifacts are attached to compiled function:

- [python/CuTeDSL/cutlass/base_dsl/dsl.py#L1272](../python/CuTeDSL/cutlass/base_dsl/dsl.py#L1272)
- [python/CuTeDSL/cutlass/base_dsl/jit_executor.py#L910](../python/CuTeDSL/cutlass/base_dsl/jit_executor.py#L910)

Toolchain evidence:

1. Official docs state CuTe DSL compiles to PTX then uses CUDA PTX compiler toolchain for SASS:
   - [media/docs/pythonDSL/overview.rst#L35](../media/docs/pythonDSL/overview.rst#L35)
   - [media/docs/pythonDSL/faqs.rst#L87](../media/docs/pythonDSL/faqs.rst#L87)
2. Binary symbols in `_cutlass_ir.so` include `cuda-to-binary`, `libNVVM`, `nvPTXCompilerCreate`, `nvFatbin*`, `ptx-options`, `cubin-format`.

Inference from (1)+(2):

- pipeline lowering is MLIR-native until binary emission
- final PTX->cubin compilation is done through embedded CUDA compiler libraries (`libNVVM` + `libnvptxcompiler`/fatbin APIs), not by shelling out to `nvcc`

## Frame 11: JIT object and invocation model

Compile+JIT path:

- [python/CuTeDSL/cutlass/base_dsl/dsl.py#L1223](../python/CuTeDSL/cutlass/base_dsl/dsl.py#L1223)
- [python/CuTeDSL/cutlass/base_dsl/compiler.py#L166](../python/CuTeDSL/cutlass/base_dsl/compiler.py#L166)

For CuTe DSL compiled functions, concrete runtime object type is:

- `CudaDialectJitCompiledFunction` at [python/CuTeDSL/cutlass/cutlass_dsl/cuda_jit_executor.py#L70](../python/CuTeDSL/cutlass/cutlass_dsl/cuda_jit_executor.py#L70)

In `cute.compile(...)` path here (`compile_only=True`):

- returns compiled function handle immediately
- no runtime kernel execution is performed by `cute.compile`

## `@cute.jit` vs `@cute.kernel` path differences

Source references:

- decorators: [python/CuTeDSL/cutlass/base_dsl/dsl.py#L470](../python/CuTeDSL/cutlass/base_dsl/dsl.py#L470), [python/CuTeDSL/cutlass/base_dsl/dsl.py#L478](../python/CuTeDSL/cutlass/base_dsl/dsl.py#L478)
- call conventions doc: [media/docs/pythonDSL/cute_dsl_general/dsl_introduction.rst#L75](../media/docs/pythonDSL/cute_dsl_general/dsl_introduction.rst#L75)

| Aspect | `@cute.jit` | `@cute.kernel` |
|---|---|---|
| Executor target | `_func` (host compile path) | `_kernel_helper` |
| Call from Python | Yes | No (must be launched from `@jit`/DSL context) |
| IR emitted | host function (`func.func`) | device kernel op (`cuda.kernel`) |
| Runtime role | creates/owns host entry callable | defines GPU kernel body + launch-site operand schema |
| Launch config | N/A directly | required via `.launch(grid=..., block=..., ...)` |
| In RMSNorm path | `CtaNorm.__call__` | `CtaNorm.kernel` |

## `enable_tvm_ffi=True/False` differences

Important distinction:

- `cute.runtime.from_dlpack(..., enable_tvm_ffi=True)` affects tensor wrapper interop behavior
- compile mode switch comes from compile option/environment (`--enable-tvm-ffi` or `CUTE_DSL_ENABLE_TVM_FFI=1`)

Compile-mode control sources:

- option env merge: [python/CuTeDSL/cutlass/base_dsl/compiler.py#L406](../python/CuTeDSL/cutlass/base_dsl/compiler.py#L406)
- TVM FFI branch: [python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py#L496](../python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py#L496)
- TVM FFI docs: [media/docs/pythonDSL/cute_dsl_general/compile_with_tvm_ffi.rst#L28](../media/docs/pythonDSL/cute_dsl_general/compile_with_tvm_ffi.rst#L28)

Behavioral differences:

1. Core kernel lowering pipeline: effectively same `cute-to-nvvm` path
2. TVM mode adds a post-compile hook that appends a TVM ABI entry function (`attach_ffi_func`) into module
3. Returned callable type differs:
   - non-TVM: `CudaDialectJitCompiledFunction`
   - TVM: `TVMFFIJitCompiledFunction` / kwargs wrapper variant
4. Call overhead and argument marshaling differ (TVM FFI path optimized for direct framework tensors)
5. `EnableTVMFFI` is not serialized into pass options string directly (`EmptyCompileOption`), so the switch is handled in Python compile orchestration

## Direct Answers

1. How user-level Python is parsed  
   - `inspect.getsourcelines` + `ast.parse` in `DSLPreprocessor.transform_function`, then transformed AST is `compile(..., mode="exec")` + `exec`.

2. How parsed code is converted to MLIR  
   - transformed function executes with IR proxy arguments inside `generate_original_ir`; overloaded DSL ops and helper builders emit MLIR ops directly.

3. How AST + tracing work together  
   - AST rewrite captures control-flow structure (region functions, carried values). Tracing executes rewritten function once and emits concrete MLIR ops for arithmetic plus structured SCF/GPU ops.

4. `@cute.jit` vs `@cute.kernel` compilation path  
   - `@cute.jit` enters host `_func` compilation; `@cute.kernel` enters `_kernel_helper`, generating kernel op and launch op semantics used from host tracing.

5. MLIR pipeline stages  
   - `cute-to-nvvm` pipeline with staged passes from CuTe dialect lowering through LLVM/NVVM conversion and `cuda-to-binary`/`cuda-error-handling` (pass list shown above from dumps).

6. How MLIR becomes PTX/CUBIN  
   - MLIR pass pipeline lowers to NVVM/LLVM IR, then `cuda-to-binary` emits PTX and compiles to cubin/fatbin via CUDA compiler libraries.

7. Difference when `enable-tvm` true/false  
   - core kernel lowering same; TVM mode adds ABI wrapper function and returns TVM FFI callable type with different invocation path.

8. Is `nvcc` or `ptxas` used?  
   - not `nvcc` as an external compiler process in this path; PTX->SASS compilation is done through CUDA PTX compiler toolchain libraries (`ptxas`-equivalent backend via `libnvptxcompiler`/libNVVM APIs) invoked by MLIR backend machinery.

## Key Functions Index

| Function | File | Purpose |
|----------|------|---------|
| `CompileCallable._compile` | `python/CuTeDSL/cutlass/base_dsl/compiler.py` | Entry for `cute.compile`; normalizes callable and forces compile-only path |
| `BaseDSL.jit_runner` | `python/CuTeDSL/cutlass/base_dsl/dsl.py` | Shared decorator wrapper for `@jit`/`@kernel` |
| `BaseDSL._func` | `python/CuTeDSL/cutlass/base_dsl/dsl.py` | Host compile path: args -> IR -> compile -> callable |
| `DSLPreprocessor.transform_function` | `python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py` | Parses and rewrites Python AST for control flow |
| `BaseDSL.generate_original_ir` | `python/CuTeDSL/cutlass/base_dsl/dsl.py` | Executes traced function with IR args to build module |
| `CutlassBaseDSL._kernel_helper` | `python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py` | Lowers `@cute.kernel` into kernel op + launch op |
| `CutlassBaseDSL._get_pipeline` | `python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py` | Builds `cute-to-nvvm` pass pipeline string |
| `Compiler.compile` | `python/CuTeDSL/cutlass/base_dsl/compiler.py` | Runs MLIR pass manager |
| `CutlassBaseDSL.compile_and_cache` | `python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py` | CuTe compile orchestration + TVM FFI hook |
| `CudaDialectJitCompiledFunction.to` | `python/CuTeDSL/cutlass/cutlass_dsl/cuda_jit_executor.py` | Loads CUDA library symbols for runtime invocation |

## Process Log

- Approach:
  - traced from concrete callsite first (`rmsnorm.py:325`)
  - followed compile dispatch (`CompileCallable`) into DSL core (`BaseDSL`)
  - expanded preprocessing internals (`DSLPreprocessor`) to explain AST rewrite mechanics
  - traced kernel lowering path (`_kernel_helper`, `KernelLauncher`, `cuda.launch_ex`)
  - corroborated MLIR pass stages from on-disk pass dumps under `experiments/*/builtin_module_no-symbol-name`
  - corroborated PTX/CUBIN toolchain behavior from docs and `_cutlass_ir.so` exported strings
- Delegation:
  - no delegated sub-agents were used
- Tools used:
  - `rg`, `nl`, `cat`, `find`, `strings`

