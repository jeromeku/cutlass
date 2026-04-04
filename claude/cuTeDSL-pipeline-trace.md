# CuTeDSL Compilation Pipeline Trace

## Tracing Process

**Entry point:** `experiments/rmsnorm.py` line 325 — `cute.compile(layernorm, _y, _x, _weight, _bias)`

**Methodology:** Dispatched 4 parallel exploration agents to trace:
1. `cute.compile` entry point and orchestration
2. AST parsing and tracing mechanism
3. MLIR dialects, pass pipeline, and module structure
4. PTX/CUBIN generation and TVM FFI path

Then manually read key functions (`generate_original_ir`, `_kernel_helper`, `_preprocess_and_replace_code`, `run_preprocessor`, `transform`) to connect the dots.

---

## 1. High-Level Pipeline Overview

```
Python Source (@cute.jit / @cute.kernel decorated)
        │
        ▼
   ┌─────────────────────────────────┐
   │  AST Preprocessing              │  ast_preprocessor.py
   │  (rewrite for/if/while → DSL    │
   │   generator-based control flow)  │
   └──────────────┬──────────────────┘
                  │
                  ▼
   ┌─────────────────────────────────┐
   │  Tracing / IR Generation        │  dsl.py:generate_original_ir()
   │  (execute preprocessed Python   │
   │   with MLIR builder context →   │
   │   emits MLIR ops inline)        │
   └──────────────┬──────────────────┘
                  │
                  ▼
   ┌─────────────────────────────────┐
   │  MLIR Pass Pipeline             │  compiler.py:compile()
   │  cute-to-nvvm{cubin-format=bin} │  → PassManager.parse() + run()
   └──────────────┬──────────────────┘
                  │
                  ▼
   ┌─────────────────────────────────┐
   │  PTX / CUBIN (embedded in       │  gpu.binary op in MLIR module
   │  gpu.binary MLIR operations)    │
   └──────────────┬──────────────────┘
                  │
                  ▼
   ┌─────────────────────────────────┐
   │  Runtime Loading & Execution    │  jit_executor.py / cuda_jit_executor.py
   │  (extract CUBIN → CUDA driver   │
   │   load → kernel launch)         │
   └─────────────────────────────────┘
```

---

## 2. Entry Point: `cute.compile()`

### 2a. `compile` is a `CompileCallable` instance

**File:** `python/CuTeDSL/cutlass/cute/__init__.py:207`
```python
compile = _dsl.CompileCallable()
```

### 2b. `CompileCallable.__call__()` → `_compile()`

**File:** `python/CuTeDSL/cutlass/base_dsl/compiler.py:555-656`

When `cute.compile(layernorm, _y, _x, _weight, _bias)` is called:

1. Detects `layernorm` is a class instance with `__call__` decorated by `@cute.jit` (line 609-617)
2. Sets `compile_only=True`, `no_cache=True` (lines 599-600)
3. Unwraps the decorator to get the original function
4. Delegates to: `func._dsl_object._func(func, *args, compile_only=True, no_cache=True)`

---

## 3. Decorators: `@cute.jit` vs `@cute.kernel`

**File:** `python/CuTeDSL/cutlass/base_dsl/dsl.py:469-483`

```python
@classmethod
def jit(cls, *dargs, **dkwargs):
    frame = inspect.currentframe().f_back
    return BaseDSL.jit_runner(cls, "_func", frame, *dargs, **dkwargs)   # ← host executor

@classmethod
def kernel(cls, *dargs, **dkwargs):
    frame = inspect.currentframe().f_back
    return BaseDSL.jit_runner(cls, "_kernel_helper", frame, *dargs, **dkwargs)  # ← GPU executor
```

Both go through `jit_runner()` (line 428), which wraps the function in a `jit_wrapper` closure. When called, `jit_wrapper`:

1. Calls `_preprocess_and_replace_code(func)` — AST preprocessing (once, then cached)
2. Dispatches to the named executor on the DSL object

### Key Difference

| Aspect | `@cute.jit` → `_func()` | `@cute.kernel` → `_kernel_helper()` |
|--------|--------------------------|--------------------------------------|
| **Scope** | Host (CPU) code, top-level entry | GPU device kernel code |
| **MLIR output** | `func.func` in module body | `gpu.func` inside `gpu.module("kernels")` |
| **Context** | Creates the MLIR module, IR context, compilation pipeline | Operates inside an already-open MLIR module; inserts into GPU module |
| **Launch** | N/A — orchestrates compilation | Generates `gpu.launch_func` call site on host side |
| **When called** | At top level (e.g., `layernorm(...)`) | When a `@cute.jit` host function calls `self.kernel(...)` |

### Typical call chain (rmsnorm example)

```
cute.compile(layernorm, ...)
  → CtaNorm.__call__  (@cute.jit → _func)    # HOST: creates module, builds host func
    → self.kernel(...)  (@cute.kernel → _kernel_helper)  # DEVICE: inserts GPU kernel into module
      → .launch(grid=..., block=...)  # generates gpu.launch_func in host func
```

---

## 4. AST Preprocessing (Phase 1)

**File:** `python/CuTeDSL/cutlass/base_dsl/ast_preprocessor.py`
**Trigger:** `_preprocess_and_replace_code()` at `dsl.py:401-425`

### What it does

The `DSLPreprocessor` (extends `ast.NodeTransformer`, line 276) rewrites the Python AST of decorated functions **before** they are traced. This happens **once** per function and replaces `func.__code__`.

### Key transformations

1. **`for` loops** (`visit_For`, line 1175): Dynamic `range()` loops are rewritten into generator-based `yield` patterns via `loop_selector` / `loop_executor`. Static loops (`range_constexpr`) are kept as-is.

2. **`if/elif/else`** (`visit_If`, line 2011): Rewritten using `if_selector` / `if_executor` to support SSA-style variable flow through branches (variables modified in branches are captured via `yield`).

3. **`while` loops** (`visit_While`, line 1843): Similarly rewritten via `while_selector` / `while_executor`.

4. **Variable analysis** (`analyze_region_variables`, line 777): Determines which variables are read/written in control flow regions. Written variables become `yield` values to maintain SSA form.

### Preprocessing flow

```
_preprocess_and_replace_code(func)                  # dsl.py:401
  → _lazy_initialize_dsl(func)                      # materializes CuTeDSL singleton
  → run_preprocessor(original_function)              # dsl.py:1419
    → preprocessor_session.transform(func, globals)  # ast_preprocessor.py:761
      → transform_function(name, func_pointer)       # get source → parse AST → visit/transform
        → visit_For / visit_If / visit_While         # rewrite control flow
    → compile(transformed_ast, ...)                  # Python compile to code object
    → exec(code_object, globals)                     # execute to get new function pointer
  → func.__code__ = fcn_ptr.__code__                 # replace original function's bytecode
```

### Example: `for` loop transformation

```python
# Before preprocessing:
for i in range(cute.size(x)):
    val += x[i].to(cutlass.Float32)

# After preprocessing (conceptual):
def _loop_body(i, val):
    val += x[i].to(cutlass.Float32)
    yield val  # SSA: return modified variables
val = __base_dsl__.loop_selector(range(cute.size(x)), _loop_body, val)
```

This allows the DSL to intercept `for` at trace time and emit `scf.for` MLIR ops.

### `const_expr` handling

**File:** `python/CuTeDSL/cutlass/base_dsl/ast_helpers.py`

`cutlass.const_expr(self.norm_type == "layer")` evaluates **at Python level before tracing**. The AST preprocessor does **not** transform `if` blocks guarded by `const_expr` — they are evaluated as regular Python conditionals during tracing, effectively pruning dead branches from the generated MLIR.

```python
if cutlass.const_expr(self.norm_type == "rms"):
    # This branch is included/excluded at trace time based on Python evaluation
    tYrY = self.apply_rmsnorm(...)
```

---

## 5. Tracing / IR Generation (Phase 2)

**File:** `python/CuTeDSL/cutlass/base_dsl/dsl.py:1102-1163`

### The core mechanism: `generate_original_ir()`

CuTeDSL uses **tracing-based IR generation**. After AST preprocessing, the (now-modified) Python function is **executed** with MLIR builder objects as arguments. Each DSL operation (e.g., `cute.make_copy_atom(...)`) emits MLIR operations into the current `InsertionPoint`.

```python
def generate_original_ir(self, ir, func, funcBody, kwargs, function_name, func_types, ...):
    module = ir.Module.create(loc=loc)
    module.operation.attributes["gpu.container_module"] = ir.UnitAttr.get()

    with ir.InsertionPoint(module.body):
        self._build_gpu_module(gpu_module_attrs, loc=loc)     # empty gpu.module("kernels")

        fop = func.FuncOp(function_name, (func_types, ret_types))  # host func.func
        fop.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()

        entry_block = fop.add_entry_block()
        with ir.InsertionPoint(entry_block):
            ir_args, ir_kwargs = self.generate_execution_arguments(...)
            result = funcBody(*ir_args, **ir_kwargs)   # ← TRACE: execute user code
            func.ReturnOp(default_ret_values)

    return module, result
```

### How AST + Tracing work together

| Phase | Mechanism | What it handles |
|-------|-----------|-----------------|
| **AST Preprocessing** | `ast.NodeTransformer` | Rewrites `for/if/while` into generator-based patterns the DSL can intercept |
| **Tracing** | Execute preprocessed Python with MLIR builders | All other operations: arithmetic, memory access, function calls, `const_expr` branching |

The AST rewriting is necessary because Python's native `for` and `if` are opaque to tracing — the tracer only sees the final values, not the control flow structure. By rewriting them into DSL-interceptable patterns, the tracer can emit proper `scf.for`, `scf.if`, and `scf.while` MLIR operations.

---

## 6. `_func()` — The Host Compilation Driver

**File:** `python/CuTeDSL/cutlass/base_dsl/dsl.py:1505-1581`

```
_func(funcBody, *args, **kwargs)
  1. sig = _check_arg_count(...)                    # validate arguments
  2. canonicalized_args, kwargs = _canonicalize_args(sig, ...)  # bind defaults
  3. function_name = mangle_name(...)               # unique name for caching
  4. compile_options.apply_envar_settings(...)       # apply env var overrides
  5. result = generate_mlir(funcBody, ...)           # main compilation pipeline
```

### `generate_mlir()` orchestration

**File:** `python/CuTeDSL/cutlass/base_dsl/dsl.py:1330-1417`

```
generate_mlir(funcBody, kwargs, function_name, ...)
  with ir.Context():
    1. generate_mlir_function_types()   # Convert Python types → MLIR types (memref, i32, f16, ...)
    2. extract_dynamic_args()           # Separate compile-time constants from runtime args
    3. generate_original_ir()           # Build MLIR module by tracing (see Phase 2)
    4. if cache miss:
         compile_and_cache()            # Run MLIR passes + JIT engine
       else:
         reuse cached JitCompiledFunction
    5. if compile_only: return jit_function
       else: jit_function.run_compiled_program(exe_args)
```

---

## 7. `_kernel_helper()` — The GPU Kernel Path

**File:** `python/CuTeDSL/cutlass/base_dsl/dsl.py:1738-1870`

When `@cute.kernel` code is called from within a `@cute.jit` host function, `_kernel_helper` runs **inside the already-open MLIR module**:

```
_kernel_helper(funcBody, *args, **kwargs)
  1. kernel_name = f"kernel_{mangled_name}_{counter}"
  2. Generate kernel operands and types
  3. with self._enter_gpu_module():                     # Insert into gpu.module("kernels")
       fop = helper.generate_func_op(...)               # Create GPU function op (cuda.func)
       fop.attributes["nvvm.reqntid"] = ...             # Set thread count attrs
       with InsertionPoint(func_body):
           ir_args = generate_execution_arguments(...)
           kernel_ret = funcBody(*ir_args, **ir_kwargs)  # ← TRACE kernel body
           helper.generate_func_ret_op()
  4. Generate gpu.launch_func at host call site          # Links host → kernel
  5. Return KernelReturns(kernel_func_ret, launch_op_ret)
```

The result is a **dual structure** in the MLIR module:
- **Inside `gpu.module("kernels")`**: The actual GPU kernel function
- **In the host `func.func`**: A `gpu.launch_func` op that references the kernel

---

## 8. MLIR Module Structure

After IR generation, the MLIR module looks like:

```mlir
builtin.module attributes {gpu.container_module} {
  gpu.module @kernels {
    // GPU kernel (generated by @cute.kernel)
    gpu.func @kernel_kernel_... (%arg0: memref<...>, ...)
        attributes {nvvm.reqntid = array<i32: 256, 1, 1>} {
      // ... kernel body ops (cute.*, arith.*, scf.*, etc.)
      gpu.return
    }
  }

  // Host wrapper (generated by @cute.jit)
  func.func @__call___... (%arg0: memref<...>, ...)
      attributes {llvm.emit_c_interface} {
    // ... host setup code
    gpu.launch_func @kernels::@kernel_kernel_...
        blocks in (%grid_x, %c1, %c1)
        threads in (%block_x, %c1, %c1)
        args(%arg0 : memref<...>, ...)
    func.return
  }
}
```

### MLIR Dialects Used

**File:** `python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py:56-64`

| Dialect | Purpose |
|---------|---------|
| `func` | Host function definitions |
| `gpu` | GPU module, kernel launch |
| `arith` | Arithmetic operations (add, mul, cmp, ...) |
| `scf` | Structured control flow (for, if, while) |
| `cf` | Unstructured control flow |
| `cute` | CuTe algebra ops (layout, tiling, copy atoms) — **custom NVIDIA dialect** |
| `cute_nvgpu` | GPU-specific CuTe ops (MMA, TMA, barriers) — **custom NVIDIA dialect** |
| `cuda` | CUDA-specific ops (kernel launch, stream management) |
| `nvvm` | NVVM intrinsics (thread_idx, sync, shuffle) |
| `llvm` | LLVM-level ops (loop annotations, global variables) |

---

## 9. MLIR Pass Pipeline (Phase 3)

### Pipeline construction

**File:** `python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py:254-272`

```python
def _get_pipeline(self, pipeline):
    if pipeline is None:
        return (
            "builtin.module(cute-to-nvvm{cubin-format=bin "
            + self.compile_options.to_str()   # opt-level, gpu-arch, lineinfo, etc.
            + "})"
        )

def preprocess_pipeline(self, pipeline, arch):
    pipeline = super().preprocess_pipeline(pipeline, arch)  # appends cubin-chip=sm_XXX
    pipeline = (
        pipeline.rstrip("})")
        + " enable-cuda-dialect=true cuda-dialect-external-module=true})"
    )
    return pipeline
```

Final pipeline string example:
```
builtin.module(cute-to-nvvm{cubin-format=bin opt-level=3 cubin-chip=sm_100a
    enable-cuda-dialect=true cuda-dialect-external-module=true})
```

### Pipeline execution

**File:** `python/CuTeDSL/cutlass/base_dsl/compiler.py:136-164`

```python
def compile(self, module, pipeline, cuda_toolkit="", arch="", enable_verifier=False):
    pm = self.passmanager.PassManager.parse(pipeline)
    pm.enable_verifier(enable_verifier)
    pm.run(module.operation)    # ← runs all passes
```

### What `cute-to-nvvm` does (monolithic C++ pass)

The `cute-to-nvvm` pass is a **single monolithic MLIR pass** implemented in the C++ shared library (`_cutlass_ir.cpython*.so`). It internally handles the full lowering chain:

```
cute dialect ops  →  nvvm / gpu ops  →  LLVM dialect  →  LLVM IR  →  PTX  →  CUBIN
```

The `cubin-format=bin` option tells it to embed the compiled CUBIN as binary data inside `gpu.binary` operations in the output MLIR module. Neither `nvcc` nor `ptxas` are invoked externally — the entire pipeline runs inside MLIR's infrastructure, using LLVM's PTX backend for code generation and NVIDIA's `libNVVM` (or equivalent) for PTX→CUBIN assembly.

---

## 10. PTX → CUBIN Generation

### Where CUBIN lives after compilation

After `pm.run(module.operation)`, the MLIR module contains `gpu.binary` operations with CUBIN data embedded as escaped binary strings:

```mlir
gpu.binary @kernels <#gpu.object<bin = "\7FELF...">
```

### CUBIN extraction

**File:** `python/CuTeDSL/cutlass/base_dsl/jit_executor.py:77-101`

```python
def walk_module_and_get_cubin_data(module, sym, callback):
    def walk_gpu_binary_op(op):
        if op.name != "gpu.binary":
            return ir.WalkResult.ADVANCE
        s = io.BytesIO()
        op.write_bytecode(s)
        cubin_data = s.getvalue()
        cubin_data = cubin_data.split(b'bin = "')[1].split(b'">')[0]
        cubin_data = get_escaped_cubin_bytes(cubin_data)
        callback(sym, func_sym, cubin_data)
    module.operation.walk(walk_gpu_binary_op)
```

### Runtime loading

**File:** `python/CuTeDSL/cutlass/base_dsl/jit_executor.py:104-136`

```python
def load_kernels_from_ir_module(module, kernel_info):
    for sym in kernel_symbols:
        def walk_callback(sym, func_sym, cubin_data):
            cubin_module = cuda_helpers.load_library_data(cubin_data)  # cuLibraryLoadData
            kernel = cuda_helpers.get_library_kernel(cubin_module, func_sym)
            kernel_modules[sym] = CudaModuleAndKernel(sym, cubin_module, kernel, attrs)
        walk_module_and_get_cubin_data(module, sym, walk_callback)
    return list(kernel_modules.values())
```

---

## 11. TVM FFI Path vs Standard Path

### Standard path (`enable_tvm_ffi=False`)

```
MLIR module → PassManager (cute-to-nvvm) → CUBIN in gpu.binary
  → ExecutionEngine (MLIR JIT for host wrapper)
  → Extract CUBIN → CUDA driver load → kernel pointer
  → Host function calls kernel via CUDA driver API
```

The `ExecutionEngine` JIT-compiles the **host wrapper** (the `func.func` with `llvm.emit_c_interface`). GPU kernels are loaded separately from the embedded CUBIN.

### TVM FFI path (`enable_tvm_ffi=True`)

**File:** `python/CuTeDSL/cutlass/cutlass_dsl/tvm_ffi_provider.py`
**File:** `python/CuTeDSL/cutlass/base_dsl/tvm_ffi_builder/tvm_ffi_builder.py`

```
MLIR module → PassManager (cute-to-nvvm) → CUBIN in gpu.binary
  → TVMFFIFunctionBuilder wraps host function in TVM calling convention
  → ExecutionEngine JIT
  → tvm_ffi.Function.__from_mlir_packed_safe_call__(engine.raw_lookup("__tvm_ffi_..."))
```

The TVM FFI path wraps the host function in TVM's `PackedFunc` interface:
```python
def attach_ffi_func(self, symbol_name, params, call_provider):
    # Creates: func @__tvm_ffi_<name>(ptr, ptr, i32, ptr) -> i32
    # TVM calling convention: (args_ptr, type_codes, num_args, ret_val_ptr) -> error_code
```

**Differences:**
| Aspect | Standard | TVM FFI |
|--------|----------|---------|
| Host function interface | C calling convention (`llvm.emit_c_interface`) | TVM PackedFunc convention |
| Argument passing | ctypes packed array | TVM `DLTensor` / `NDArray` |
| Kwargs/defaults | Manual handling | TVM wrapper handles |
| Integration | CUDA driver API direct | TVM runtime system |
| Use case | Standalone CUTLASS DSL | Integration with TVM ecosystem |

---

## 12. CompileOptions

**File:** `python/CuTeDSL/cutlass/base_dsl/compiler.py:240-496`

| Option | Pipeline key | Effect |
|--------|-------------|--------|
| `OptLevel(3)` | `opt-level` | LLVM optimization level (0-3) |
| `GPUArch("sm_100a")` | `cubin-chip` | Target GPU architecture |
| `KeepPTX(True)` | `dump-ptx-path` | Saves intermediate PTX to file |
| `KeepCUBIN(True)` | `dump-cubin-path` | Saves CUBIN to file |
| `GenerateLineInfo(True)` | `preserve-line-info` | Debug line numbers in PTX/CUBIN |
| `PtxasOptions("")` | `ptx-options` | Extra flags for PTX assembler |
| `EnableTVMFFI(False)` | N/A | Switch to TVM FFI compilation path |
| `DumpDir("./dir")` | Affects file paths | Directory for dumped artifacts |

---

## 13. Runtime Execution

### `JitCompiledFunction`

**File:** `python/CuTeDSL/cutlass/base_dsl/jit_executor.py:862-1018`

```
JitCompiledFunction.__call__(*args)
  → args_spec.generate_execution_args(args)  # convert Python args → ctypes
  → run_compiled_program(exe_args)
    → _default_executor.to(device)           # lazy: load CUBIN, create CUDA context
    → JitExecutor.run_compiled_program()
      → packed_args = pack_ctypes(exe_args)
      → capi_func(packed_args)               # call JIT'd host wrapper
        → internally calls cuLaunchKernel    # kernel launch happens inside host wrapper
```

### `CudaDialectJitCompiledFunction`

**File:** `python/CuTeDSL/cutlass/cutlass_dsl/cuda_jit_executor.py:70-267`

Extends `JitCompiledFunction` with:
- CUDA library lifecycle management (`cudaLibraryLoad` / `cudaLibraryUnload`)
- Kernel attribute setup (block size, shared memory, min CTA)
- Device-specific module loading via CUDA runtime bindings

---

## 14. Complete Call Trace for `experiments/rmsnorm.py`

```
cute.compile(layernorm, _y, _x, _weight, _bias)
│
├─ CompileCallable._compile()                     # compiler.py:579
│   sets compile_only=True, no_cache=True
│   detects layernorm.__call__ is @cute.jit
│
├─ _preprocess_and_replace_code(CtaNorm.__call__)  # dsl.py:401
│   └─ run_preprocessor()                          # dsl.py:1419
│       └─ DSLPreprocessor.transform()             # ast_preprocessor.py:761
│           rewrites for/if in __call__ body
│       └─ compile(transformed_ast) → new __code__
│
├─ CuTeDSL._func(CtaNorm.__call__, _y, _x, ...)   # dsl.py:1505
│   └─ generate_mlir()                              # dsl.py:1330
│       │
│       ├─ generate_mlir_function_types()           # Python → MLIR types
│       │   _y (CuTe Tensor) → memref<4096x4096xf16>
│       │   eps (Float32)    → f32
│       │
│       ├─ generate_original_ir()                   # dsl.py:1102
│       │   Creates: builtin.module { gpu.module @kernels {} ; func.func @__call__... }
│       │   Traces CtaNorm.__call__:
│       │     ├─ cute.make_copy_atom(...)           → cute dialect ops
│       │     ├─ cute.make_tiled_copy_tv(...)       → cute dialect ops
│       │     ├─ self.kernel(...).launch(...)        → triggers _kernel_helper
│       │     │   │
│       │     │   ├─ _kernel_helper()               # dsl.py:1738
│       │     │   │   Inserts into gpu.module:
│       │     │   │   └─ gpu.func @kernel_kernel_... { ... }
│       │     │   │       Traces kernel body:
│       │     │   │         cute.arch.thread_idx()   → nvvm.read.ptx.sreg.tid.x
│       │     │   │         cute.local_tile(...)     → cute.* ops
│       │     │   │         for loop                 → scf.for (via preprocessed AST)
│       │     │   │         cute.autovec_copy(...)   → memory load/store ops
│       │     │   │         self.apply_rmsnorm(...)  → inlined @cute.jit call
│       │     │   │           ├─ SmemAllocator       → shared memory alloc
│       │     │   │           ├─ warp_reduce         → shuffle_sync_bfly → nvvm.shfl.sync
│       │     │   │           ├─ cta_reduce          → sync_threads + smem read/write
│       │     │   │           └─ rsqrt               → math.rsqrt
│       │     │   │
│       │     │   └─ Generates gpu.launch_func in host func body
│       │
│       ├─ compile_and_cache()                      # dsl.py:1165
│       │   ├─ preprocess_pipeline()                # Adds cubin-chip=sm_100a, enable-cuda-dialect
│       │   ├─ compiler_provider.compile(module, pipeline)  # compiler.py:136
│       │   │   └─ PassManager.parse("builtin.module(cute-to-nvvm{...})")
│       │   │   └─ pm.run(module.operation)         # C++ pass: cute→nvvm→llvm→ptx→cubin
│       │   │       Output: module with gpu.binary containing CUBIN bytes
│       │   │
│       │   ├─ compiler_provider.jit(module)        # compiler.py:166
│       │   │   └─ ExecutionEngine(module, ...)     # JIT compile host wrapper
│       │   │
│       │   └─ Creates CudaDialectJitCompiledFunction
│       │       ├─ load_kernels_from_ir_module()    # Extract CUBIN from gpu.binary
│       │       │   └─ cuda_helpers.load_library_data(cubin_bytes)  # cuLibraryLoadData
│       │       └─ Wraps: engine + capi_func + CUDA library + args_spec
│       │
│       └─ return jit_function  (compile_only=True, so skip execution)
│
└─ Returns CudaDialectJitCompiledFunction
   (can be called later with compiled_func(y, x, weight, bias, eps))
```

---

## 15. Key Files Reference

| File | Lines | Purpose |
|------|-------|---------|
| `base_dsl/dsl.py` | 401-425 | `_preprocess_and_replace_code` — trigger AST rewrite |
| `base_dsl/dsl.py` | 427-483 | `jit_runner`, `jit`, `kernel` decorators |
| `base_dsl/dsl.py` | 1102-1163 | `generate_original_ir` — tracing/IR generation |
| `base_dsl/dsl.py` | 1165-1296 | `compile_and_cache` — compilation orchestration |
| `base_dsl/dsl.py` | 1330-1417 | `generate_mlir` — full pipeline driver |
| `base_dsl/dsl.py` | 1505-1581 | `_func` — host entry point |
| `base_dsl/dsl.py` | 1738-1870 | `kernel_launcher` — GPU kernel generation |
| `base_dsl/ast_preprocessor.py` | 276-299 | `DSLPreprocessor` — AST NodeTransformer |
| `base_dsl/ast_preprocessor.py` | 761-775 | `transform` — AST transformation entry |
| `base_dsl/ast_preprocessor.py` | 1175-1217 | `visit_For` — for loop rewriting |
| `base_dsl/ast_preprocessor.py` | 2011+ | `visit_If` — if statement rewriting |
| `base_dsl/compiler.py` | 136-164 | `Compiler.compile` — PassManager execution |
| `base_dsl/compiler.py` | 166-194 | `Compiler.jit` — ExecutionEngine creation |
| `base_dsl/compiler.py` | 240-496 | `CompileOptions` — pipeline configuration |
| `base_dsl/compiler.py` | 555-656 | `CompileCallable` — `cute.compile` impl |
| `base_dsl/jit_executor.py` | 77-101 | `walk_module_and_get_cubin_data` — CUBIN extraction |
| `base_dsl/jit_executor.py` | 104-136 | `load_kernels_from_ir_module` — CUDA module loading |
| `base_dsl/jit_executor.py` | 862-1018 | `JitCompiledFunction` — compiled kernel wrapper |
| `cutlass_dsl/cutlass.py` | 254-272 | `_get_pipeline` — cute-to-nvvm pipeline string |
| `cutlass_dsl/cutlass.py` | 56-64 | MLIR dialect imports |
| `cutlass_dsl/cuda_jit_executor.py` | 70-267 | `CudaDialectJitCompiledFunction` |
| `cutlass_dsl/tvm_ffi_provider.py` | — | TVM FFI path implementation |

All paths relative to `python/CuTeDSL/cutlass/`.
