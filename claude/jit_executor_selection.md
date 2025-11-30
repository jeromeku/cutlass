# JIT Executor Selection in CuTeDSL

## Quick Answer

The choice between `JitCompiledFunction` (legacy) and `CudaDialectJitCompiledFunction` (modern) is determined by the **DSL class hierarchy**, specifically in `CutlassBaseDSL.compile_and_cache()`.

**TL;DR:**
- **CuTeDSL** (used by `@cute.jit`/`@cute.kernel`) → **Always uses `CudaDialectJitCompiledFunction`** (modern, CUDA dialect)
- **BaseDSL** (base class) → Uses `JitCompiledFunction` (legacy, direct CUDA driver)
- **With TVM FFI enabled** → Uses `TVMFFIJitCompiledFunction` (TVM interop)

---

## The Decision Point

### Location: [cutlass/cutlass_dsl/cutlass.py:420-487](../python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py#L420-L487)

The decision happens in `CutlassBaseDSL.compile_and_cache()`, which **overrides** the base class implementation:

```python
def compile_and_cache(
    self,
    module,
    module_hash,
    function_name,
    pipeline,
    args_spec,
    no_cache,
    *,
    full_args=None,
    full_kwargs=None,
    dynamic_args=None,
    dynamic_kwargs=None,
    original_function_name=None,
):
    # Check if TVM FFI is enabled
    if self.compile_options.enable_tvm_ffi:
        # Path 1: TVM FFI Mode
        from .tvm_ffi_provider import TVMFFIJitCompiledFunction

        # ... attach TVM FFI wrapper ...

        return super().compile_and_cache(
            module, module_hash, function_name, pipeline, args_spec, no_cache,
            TVMFFIJitCompiledFunction,  # ← TVM FFI executor
            full_args=full_args, full_kwargs=full_kwargs,
            dynamic_args=dynamic_args, dynamic_kwargs=dynamic_kwargs,
        )

    # Path 2: Normal Mode (CUDA Dialect)
    return super().compile_and_cache(
        module, module_hash, function_name, pipeline, args_spec, no_cache,
        CudaDialectJitCompiledFunction,  # ← Modern CUDA dialect executor
        full_args=full_args, full_kwargs=full_kwargs,
        dynamic_args=dynamic_args, dynamic_kwargs=dynamic_kwargs,
        original_function_name=original_function_name,
    )
```

### Base Class Default: [cutlass/base_dsl/dsl.py:1207-1292](../python/CuTeDSL/cutlass/base_dsl/dsl.py#L1207-L1292)

```python
def compile_and_cache(
    self,
    module,
    module_hash,
    function_name,
    pipeline,
    args_spec,
    no_cache,
    func_type=JitCompiledFunction,  # ← Default is legacy executor
    *,
    full_args=None,
    full_kwargs=None,
    dynamic_args=None,
    dynamic_kwargs=None,
    original_function_name=None,
):
    # ... compilation logic ...

    # Create JIT function with specified type
    fn = func_type(  # ← Instantiate the executor class
        module,
        engine,
        capi_func,
        args_spec,
        function_name,
        self.kernel_info,
        jit_time_profiling=self.envar.jit_time_profiling,
        jit_function_artifacts=JitFunctionArtifacts(...),
    )

    return fn
```

---

## Class Hierarchy

```
BaseDSL
    │
    ├─ compile_and_cache()
    │   └─ Default: func_type=JitCompiledFunction (legacy)
    │
    └─ CutlassBaseDSL (cutlass/cutlass_dsl/cutlass.py)
        │
        ├─ compile_and_cache() [OVERRIDDEN]
        │   ├─ If TVM FFI: func_type=TVMFFIJitCompiledFunction
        │   └─ Else:        func_type=CudaDialectJitCompiledFunction
        │
        └─ CuTeDSL (cutlass/cutlass_dsl/cutlass.py:920)
            └─ Inherits CutlassBaseDSL behavior
```

---

## Decision Flow Diagram

```
User calls @cute.jit or @cute.kernel decorated function
    ↓
BaseDSL._func() or _kernel_helper()
    ↓
BaseDSL.generate_mlir()
    ↓
BaseDSL.compile_and_cache()
    ↓
[Virtual function dispatch based on DSL class]
    ↓
CutlassBaseDSL.compile_and_cache() [OVERRIDE]
    ↓
Check: self.compile_options.enable_tvm_ffi
    ├─ YES → super().compile_and_cache(func_type=TVMFFIJitCompiledFunction)
    │         ↓
    │     TVMFFIJitCompiledFunction(...)
    │         │
    │         └─ Inherits from CudaDialectJitCompiledFunction
    │            └─ Uses CUDA dialect runtime + TVM FFI wrapper
    │
    └─ NO → super().compile_and_cache(func_type=CudaDialectJitCompiledFunction)
              ↓
          CudaDialectJitCompiledFunction(...)
              └─ Uses CUDA dialect runtime (modern, context-free)
```

---

## The Three Executor Types

### 1. JitCompiledFunction (Legacy)

**Location:** [cutlass/base_dsl/jit_executor.py:553+](../python/CuTeDSL/cutlass/base_dsl/jit_executor.py#L553)

**Characteristics:**
- Direct CUDA driver API calls (`cuModuleLoadData`, `cuModuleGetFunction`)
- Device-specific contexts (`DevicePrimaryContext`)
- Explicit CUBIN extraction from MLIR module
- Manual kernel attribute management

**When Used:**
- Only if using `BaseDSL` directly (not common)
- Not used by CuTeDSL/CutlassBaseDSL

**Execution Path:**
```python
JitCompiledFunction
    ↓
load_kernels_from_ir_module()
    ├─ Walk MLIR module for gpu.binary ops
    ├─ Extract CUBIN bytes
    ├─ cuda_helpers.load_library_data(cubin)  # cuModuleLoadData
    └─ cuda_helpers.get_library_kernel(module, name)  # cuModuleGetFunction
        ↓
get_device_execute_context(device_id)
    ├─ Create DevicePrimaryContext(device_id)
    └─ Set kernel attributes
        ↓
run_compiled_program(exe_args)
    ├─ Pack arguments as C array
    └─ capi_func(packed_args)  # ExecutionEngine call
```

---

### 2. CudaDialectJitCompiledFunction (Modern)

**Location:** [cutlass/cutlass_dsl/cuda_jit_executor.py:64+](../python/CuTeDSL/cutlass/cutlass_dsl/cuda_jit_executor.py#L64)

**Characteristics:**
- CUDA dialect runtime (`cudaLibrary_t` API)
- Context-free execution (no explicit device binding)
- Multi-device support out of the box
- Cleaner abstraction

**When Used:**
- **Default for all `@cute.jit` and `@cute.kernel` functions**
- Unless TVM FFI is explicitly enabled

**Execution Path:**
```python
CudaDialectJitCompiledFunction
    ↓
_load_cuda_library()
    ├─ Get cuda_init and cuda_load_to_device from ExecutionEngine
    ├─ cuda_init(&library, &err)
    └─ For each device:
        └─ cuda_load_to_device(library, device_id, &err)
            ↓
to(device=None)  # Optional device binding
    └─ Returns JitExecutor (context-free)
        ↓
run_compiled_program(exe_args)
    ├─ Pack arguments
    └─ capi_func(packed_args)  # Uses CUDA dialect runtime
```

**Key Advantage:**
```python
# No need for explicit device management
jit_fn = cute.compile(my_kernel, args)

# Works on any device without rebinding
with cuda.device(0):
    jit_fn(args)

with cuda.device(1):
    jit_fn(args)  # No recompilation or rebinding needed!
```

---

### 3. TVMFFIJitCompiledFunction (TVM Interop)

**Location:** [cutlass/cutlass_dsl/tvm_ffi_provider.py](../python/CuTeDSL/cutlass/cutlass_dsl/tvm_ffi_provider.py)

**Characteristics:**
- Inherits from `CudaDialectJitCompiledFunction`
- Adds TVM FFI ABI wrapper
- Enables calling from TVM/Apache ecosystem
- Standard calling convention

**When Used:**
- Only when `EnableTVMFFI` compile option is set
- For integration with TVM/Apache projects

**Usage:**
```python
# Enable TVM FFI
@cute.compile[cute.EnableTVMFFI]
def my_kernel(a: Tensor, b: Tensor):
    return a + b

jit_fn = cute.compile[cute.EnableTVMFFI](my_kernel, a, b)

# Can now be called from TVM via FFI
```

**What It Adds:**
- Converts CuTe arguments to TVM FFI spec
- Generates ABI-compliant wrapper function
- Attaches wrapper to MLIR module
- Standard error handling via TVM FFI

---

## How to Control Executor Selection

### Option 1: Use CuTeDSL (Default)

```python
import cutlass.cute as cute

@cute.jit
def my_func(a, b):
    return a + b

# Uses: CudaDialectJitCompiledFunction (modern)
```

### Option 2: Enable TVM FFI

```python
import cutlass.cute as cute

@cute.compile[cute.EnableTVMFFI]
def my_func(a, b):
    return a + b

# Uses: TVMFFIJitCompiledFunction (TVM interop)
```

### Option 3: Use BaseDSL Directly (Not Recommended)

```python
from cutlass.base_dsl import BaseDSL

class MyDSL(BaseDSL):
    # Don't override compile_and_cache
    pass

# Uses: JitCompiledFunction (legacy)
```

---

## Comparison Table

| Feature | JitCompiledFunction (Legacy) | CudaDialectJitCompiledFunction (Modern) | TVMFFIJitCompiledFunction (TVM) |
|---------|----------------------------|----------------------------------------|--------------------------------|
| **Used By** | BaseDSL (base class) | CuTeDSL (default) | CuTeDSL with EnableTVMFFI |
| **CUDA API** | Driver API (cu*) | CUDA dialect runtime | CUDA dialect runtime + TVM FFI |
| **Context Management** | Device-specific contexts | Context-free | Context-free |
| **Module Loading** | `cuModuleLoadData` | `cudaLibraryLoad` | `cudaLibraryLoad` |
| **Multi-Device** | Requires per-device contexts | Automatic | Automatic |
| **Performance** | Slightly lower overhead | Slightly higher overhead | Slightly higher overhead |
| **Interoperability** | CUDA only | CUDA only | TVM/Apache ecosystem |
| **Abstraction Level** | Low (direct driver calls) | Medium (dialect runtime) | High (FFI + dialect) |
| **When to Use** | Custom DSL without CUDA dialect | General CuTeDSL usage (default) | TVM integration |

---

## Key Insights

### 1. **CuTeDSL Always Uses CUDA Dialect**

Every `@cute.jit` and `@cute.kernel` function uses `CudaDialectJitCompiledFunction` by default (unless TVM FFI is enabled). This is hardcoded in `CutlassBaseDSL.compile_and_cache()`.

### 2. **Override Pattern**

The executor type is passed as the `func_type` parameter to `BaseDSL.compile_and_cache()`. Subclasses can override this to use different executor types.

### 3. **TVM FFI is Opt-In**

TVM FFI support is only enabled when explicitly requested via `cute.EnableTVMFFI` compile option. This adds overhead, so it's only used when needed.

### 4. **Virtual Dispatch**

Python's virtual function dispatch ensures that `CutlassBaseDSL.compile_and_cache()` is called instead of `BaseDSL.compile_and_cache()`, allowing executor customization.

### 5. **No Runtime Detection**

The choice is **compile-time**, not runtime. It's determined by:
- DSL class type (CuTeDSL vs BaseDSL)
- Compile options (EnableTVMFFI)

---

## Debugging: How to Check Which Executor is Used

### Method 1: Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

@cute.jit
def my_func(a, b):
    return a + b

my_func(1, 2)
# Logs will show: "Creating CudaDialectJitCompiledFunction for function my_func_1_2"
```

### Method 2: Inspect Compiled Function

```python
jit_fn = cute.compile(my_func, 1, 2)

print(type(jit_fn))
# <class 'cutlass.cutlass_dsl.cuda_jit_executor.CudaDialectJitCompiledFunction'>
```

### Method 3: Check DSL Class

```python
@cute.jit
def my_func(a, b):
    return a + b

print(type(my_func._dsl_object))
# <class 'cutlass.cutlass_dsl.cutlass.CuTeDSL'>

# CuTeDSL → inherits CutlassBaseDSL → uses CudaDialectJitCompiledFunction
```

---

## Summary

**The executor type is determined by:**

1. **DSL Class Hierarchy**
   - `CuTeDSL` (via `CutlassBaseDSL`) → `CudaDialectJitCompiledFunction`
   - `BaseDSL` → `JitCompiledFunction`

2. **Compile Options**
   - `EnableTVMFFI` → `TVMFFIJitCompiledFunction`
   - Default → `CudaDialectJitCompiledFunction`

3. **Override in `compile_and_cache()`**
   - `CutlassBaseDSL` overrides and explicitly passes `CudaDialectJitCompiledFunction`
   - This is hardcoded, not configurable at runtime

**For 99% of users:** You're using `CudaDialectJitCompiledFunction` (modern, context-free, CUDA dialect runtime).

**You only use the legacy `JitCompiledFunction` if:** You're implementing a custom DSL that directly inherits `BaseDSL` without overriding `compile_and_cache()`.
