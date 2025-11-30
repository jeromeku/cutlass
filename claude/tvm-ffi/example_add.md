# CuTe's TVM-FFI Integration: Complete Call Path Trace

This document provides a **frame-by-frame walkthrough** of the entire call path for the following line from [experiments/tvm-ffi/example_add.py:28](../../experiments/tvm-ffi/example_add.py#L28):

```python
compiled_add_one = cute.compile(add_one, a_cute, b_cute, options="--enable-tvm-ffi")
```

This trace goes all the way from the CuTeDSL Python layer down through the MLIR compilation pipeline to TVM's C++ internals.

---

## Table of Contents

1. [Overview](#overview)
2. [Frame-by-Frame Walkthrough](#frame-by-frame-walkthrough)
   - [Frame 1: Python Entry Point - `cute.compile()`](#frame-1-python-entry-point---cutecompile)
   - [Frame 2: CompileCallable.__call__()](#frame-2-compilecallable__call__)
   - [Frame 3: DSL Preprocessing and Execution](#frame-3-dsl-preprocessing-and-execution)
   - [Frame 4: MLIR Module Generation](#frame-4-mlir-module-generation)
   - [Frame 5: TVM FFI Post-Compile Hook](#frame-5-tvm-ffi-post-compile-hook)
   - [Frame 6: Attach FFI Function to MLIR Module](#frame-6-attach-ffi-function-to-mlir-module)
   - [Frame 7: MLIR Compilation Pipeline](#frame-7-mlir-compilation-pipeline)
   - [Frame 8: TVM FFI Function Initialization](#frame-8-tvm-ffi-function-initialization)
   - [Frame 9: Runtime Execution (When Called)](#frame-9-runtime-execution-when-called)
3. [Key Data Structures](#key-data-structures)
4. [TVM FFI ABI Interface](#tvm-ffi-abi-interface)
5. [Summary](#summary)

---

## Overview

The integration between CuTeDSL and TVM-FFI enables CuTe-compiled CUDA kernels to be called through TVM's Foreign Function Interface (FFI). This allows interoperability with frameworks that support TVM-FFI (like PyTorch via TVM).

The compilation process involves:
1. **Python DSL Layer**: Parse and validate arguments
2. **MLIR Generation**: Convert DSL code to MLIR
3. **TVM FFI Wrapper Generation**: Create FFI-compatible wrapper functions
4. **MLIR Compilation**: Compile MLIR to CUDA binary
5. **Runtime Function Creation**: Create callable TVM FFI function object

---

## Frame-by-Frame Walkthrough

### Frame 1: Python Entry Point - `cute.compile()`

**Location**: [python/CuTeDSL/cutlass/cute/__init__.py:199](../../python/CuTeDSL/cutlass/cute/__init__.py#L199)

```python
compile = _dsl.CompileCallable()
```

**What Happens:**
- `cute.compile` is an instance of `CompileCallable` class
- When called with `options="--enable-tvm-ffi"`, it parses the options string
- Creates a `CompileOptions` object with `EnableTVMFFI` flag set

**Source Code:**
```python
# From cutlass/cute/__init__.py
from .. import cutlass_dsl as _dsl
compile = _dsl.CompileCallable()
```

**Key Observation:**
The `compile` object is a callable that wraps the DSL compilation logic.

---

### Frame 2: CompileCallable.__call__()

**Location**: [python/CuTeDSL/cutlass/base_dsl/compiler.py:577-652](../../python/CuTeDSL/cutlass/base_dsl/compiler.py#L577-L652)

```python
def __call__(self, *args, **kwargs):
    return self._compile(*args, **kwargs)

def _compile(self, func, *args, **kwargs):
    # ... validation ...
    kwargs["compile_only"] = True
    kwargs["no_cache"] = True

    # Parse options string to CompileOptions
    options = kwargs.pop("options", None)
    if isinstance(options, str) and len(options) == 0:
        options = None

    if options is not None and isinstance(options, str):
        compile_options = _parse_compile_options_from_str(options)
    else:
        compile_options = self._compile_options

    func._dsl_object.compile_options = compile_options

    # Execute the compilation
    return func._dsl_object._func(fcn_ptr, *args, **kwargs)
```

**What Happens:**
1. **Option Parsing**: The `"--enable-tvm-ffi"` string is parsed by `_parse_compile_options_from_str()`
   - Uses `argparse` to parse the command-line style options
   - Creates `EnableTVMFFI(True)` option

2. **Set Compilation Options**: The parsed options are set on the DSL object
   ```python
   func._dsl_object.compile_options = compile_options
   ```

3. **Trigger Compilation**: Calls the DSL's `_func()` method to begin compilation

**Key Files:**
- [python/CuTeDSL/cutlass/base_dsl/compiler.py:501-553](../../python/CuTeDSL/cutlass/base_dsl/compiler.py#L501-L553) - Option parsing

---

### Frame 3: DSL Preprocessing and Execution

**Location**: [python/CuTeDSL/cutlass/base_dsl/dsl.py](../../python/CuTeDSL/cutlass/base_dsl/dsl.py)

**What Happens:**
The DSL object processes the function and generates MLIR:

1. **Extract Function Arguments**:
   ```python
   args_spec = inspect.getfullargspec(func)
   exec_args = ExecutionArgs(args_spec, function_name)
   ```

2. **Generate MLIR Module**:
   - Creates an MLIR context and module
   - Walks through the Python AST of the function
   - Emits MLIR operations for each Python statement
   - Generates the main kernel function in MLIR

3. **Compute Module Hash**:
   - Creates a hash of the MLIR module for caching purposes
   - Hash includes the MLIR IR string representation

**Source Code (Conceptual):**
```python
# Generate MLIR module
with ir.Context(), ir.Location.unknown():
    module = ir.Module.create()
    # ... generate MLIR operations ...

# Compute hash for caching
module_hash = hashlib.sha256(str(module).encode()).hexdigest()
```

---

### Frame 4: MLIR Module Generation

**Location**: [python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py:388-479](../../python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py#L388-L479)

At this stage, the MLIR module contains:
- **Kernel function**: The CUDA kernel generated from `device_add_one`
- **Host function**: The launch wrapper generated from `add_one`

**Example MLIR (Simplified):**
```mlir
module {
  // CUDA Kernel
  cuda.kernel @device_add_one(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, ...) {
    // ... CUDA kernel operations ...
    cuda.return
  }

  // Host Launch Function
  func.func @cutlass_add_one(%arg0: !llvm.ptr, %arg1: !llvm.ptr, ...) -> i32 {
    // ... grid/block configuration ...
    cuda.launch_ex @device_add_one, ...
    return %result : i32
  }
}
```

**What Happens:**
The generated MLIR represents the CUDA kernel and launch logic, but **does not yet have TVM FFI wrapper**.

---

### Frame 5: TVM FFI Post-Compile Hook

**Location**: [python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py:419-464](../../python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py#L419-L464)

**Key Code:**
```python
def compile_and_cache(self, module, module_hash, function_name, pipeline, args_spec, ...):
    if self.compile_options.enable_tvm_ffi:
        from .tvm_ffi_provider import (
            TVMFFIJitCompiledFunction,
            TVMFFICuteCallProvider,
        )
        from cutlass.base_dsl.tvm_ffi_builder import attach_ffi_func

        # Convert CuTe args to TVM FFI spec
        tvm_ffi_spec_params = self._tvm_ffi_args_spec_converter(
            function_name, args_spec, full_args, full_kwargs
        )

        tvm_ffi_provider = TVMFFICuteCallProvider(function_name)

        # Register post-compile hook
        def post_compile_hook(module: ir.Module):
            with module.context, module.operation.location:
                # Attach TVM FFI wrapper to MLIR module
                attach_ffi_func(
                    module,
                    function_name,
                    tvm_ffi_spec_params,
                    tvm_ffi_provider,
                    fn_display_name=original_function_name,
                )
            module.operation.verify()

        with compiler.PostCompileHookContext(self.compiler_provider, post_compile_hook):
            return super().compile_and_cache(...)
```

**What Happens:**

1. **Args Spec Conversion**: Converts CuTe tensor/shape arguments to TVM FFI parameter specifications
   - Location: [python/CuTeDSL/cutlass/cute/_tvm_ffi_args_spec_converter.py:271-292](../../python/CuTeDSL/cutlass/cute/_tvm_ffi_args_spec_converter.py#L271-L292)

   ```python
   def _tvm_ffi_args_spec_converter(function_name, args_spec, full_args, full_kwargs):
       params = []
       ctx = ConverterContext()

       for arg, arg_name in zip(rectified_args, arg_names):
           arg_type = args_spec.annotations.get(arg_name, None)
           param = _convert_single_arg(arg, arg_name, arg_type, ctx)
           params.append(param)

       return params
   ```

   **Example Conversion:**
   - `cute.Tensor` → `spec.Tensor` (with shape, stride, dtype info)
   - `cute.Shape` → `spec.Shape` (with dimension info)
   - `int` → `spec.Var` (with dtype info)

2. **Create Call Provider**: Instantiates `TVMFFICuteCallProvider` which handles:
   - CUDA initialization
   - Device management
   - Error handling
   - Calling the actual kernel

3. **Register Post-Compile Hook**: The hook will run **after** MLIR passes but **before** JIT compilation

---

### Frame 6: Attach FFI Function to MLIR Module

**Location**: [python/CuTeDSL/cutlass/base_dsl/tvm_ffi_builder/tvm_ffi_builder.py:1983-2007](../../python/CuTeDSL/cutlass/base_dsl/tvm_ffi_builder/tvm_ffi_builder.py#L1983-L2007)

**Key Code:**
```python
def attach_ffi_func(module: ir.Module, symbol_name: str, params: Sequence[spec.Param],
                    call_provider: CallProvider, fn_display_name: Optional[str] = None):
    builder = TVMFFIFunctionBuilder(module)
    builder.attach_ffi_func(symbol_name, params, call_provider, fn_display_name)
```

**What Happens:**

The `TVMFFIFunctionBuilder` generates a **new LLVM function** in the MLIR module with the signature:
```c
int32_t __tvm_ffi_<function_name>(void* handle, void* args, int32_t num_args, void* result)
```

This function:
1. **Decodes TVM FFI arguments** from the `args` array
2. **Validates types and shapes** according to `spec.Param` specifications
3. **Calls the original CuTe kernel** via the `call_provider`
4. **Returns error codes** (0 for success, -1 for failure)

**Detailed Breakdown:**

#### 6.1: Generate FFI Wrapper Function

**Location**: [python/CuTeDSL/cutlass/base_dsl/tvm_ffi_builder/tvm_ffi_builder.py:1879-1980](../../python/CuTeDSL/cutlass/base_dsl/tvm_ffi_builder/tvm_ffi_builder.py#L1879-L1980)

```python
def attach_ffi_func(self, symbol_name, params, call_provider, fn_display_name):
    # Generate error handling helpers
    self.get_or_create_set_raised_from_cstr_parts(num_parts=...)

    # Create FFI wrapper function
    (handle, args, num_args, result), entry_block = self.function(
        name=f"__tvm_ffi_{symbol_name}",
        params_type=[self.ptr_type, self.ptr_type, self.i32_type, self.ptr_type],
        ret_type=self.i32_type,
    )

    # Check number of arguments
    current_block = self.check_condition(
        current_block,
        lambda: self.equal(num_args, self.i32(expected_num_args)),
        "TypeError",
        [f"Expects {expected_num_args} parameters", ...],
    )

    # Decode each parameter
    for arg_index, param in enumerate(params_list):
        current_block = self.decode_param(current_block, param, args, arg_index, ...)

    # Call the provider
    current_block = call_provider(current_block, context)

    # Return success
    self.return_(self.i32(0))
```

**Generated MLIR (Conceptual):**
```mlir
llvm.func @__tvm_ffi_cutlass_add_one(
    %handle: !llvm.ptr, %args: !llvm.ptr, %num_args: i32, %result: !llvm.ptr
) -> i32 {
  // Check num_args == 2
  %cond = llvm.icmp "eq" %num_args, %c2_i32 : i32
  llvm.cond_br %cond, ^bb_success, ^bb_error

^bb_error:
  // Set error and return -1
  llvm.call @TVMFFIErrorSetRaisedFromCStr(...)
  llvm.return %c_minus1_i32 : i32

^bb_success:
  // Decode arg 0 (tensor a)
  %type_idx_0 = llvm.load %args[0].type_index : i32
  %is_tensor_0 = llvm.icmp "eq" %type_idx_0, %kTVMFFITensor : i32
  llvm.cond_br %is_tensor_0, ^bb_decode_a, ^bb_type_error

^bb_decode_a:
  %dltensor_ptr_0 = llvm.load %args[0].v_ptr : !llvm.ptr
  %data_a = llvm.load %dltensor_ptr_0[0] : !llvm.ptr<1>  // GPU ptr
  %shape_a = llvm.load %dltensor_ptr_0[4] : !llvm.ptr
  %n_a = llvm.load %shape_a[0] : i64
  // ... validate dtype, device, ndim ...

  // Decode arg 1 (tensor b) - similar to arg 0
  // ...

  // Call the actual kernel via call_provider
  %result_code = llvm.call @cutlass_add_one(%data_a, %data_b, %n_a, ...)
  llvm.return %result_code : i32
}
```

#### 6.2: Decode Parameters

**Location**: [python/CuTeDSL/cutlass/base_dsl/tvm_ffi_builder/tvm_ffi_builder.py:1802-1858](../../python/CuTeDSL/cutlass/base_dsl/tvm_ffi_builder/tvm_ffi_builder.py#L1802-L1858)

For each parameter, the builder generates LLVM IR to:

1. **Check type index**:
   ```python
   type_index = self.load_ffi_any_array_item_type_index(args, arg_index)
   is_tensor = self.equal(type_index, self.i32(TVMFFITypeIndex.kTVMFFITensor))
   ```

2. **Extract DLTensor pointer**:
   ```python
   dltensor_ptr = self.load_ffi_any_array_item_v_ptr(args, arg_index)
   ```

3. **Validate and extract fields**:
   - **Data pointer**: `self.load_dltensor_data_ptr(dltensor_ptr)`
   - **Shape**: `self.load_dltensor_shape(dltensor_ptr)`
   - **Strides**: `self.load_dltensor_strides(dltensor_ptr)`
   - **Dtype**: Check `dtype.code`, `dtype.bits`, `dtype.lanes`
   - **Device**: Check `device.type`, extract `device.id`
   - **Ndim**: Validate against expected rank

4. **Bind symbolic variables**:
   ```python
   self.set_or_check_matched_var_binding(current_block, var, value, error_msg_context)
   ```

   This ensures that if a symbolic shape variable appears multiple times (e.g., `n` in both `a` and `b`),
   all occurrences have the same value.

**Example for Tensor Parameter:**

For `a: cute.Tensor` with shape `(n,)` and dtype `Float32`:

```mlir
// Check type is Tensor
%type_idx = llvm.load %args[0].type_index : i32
%is_tensor = llvm.icmp "eq" %type_idx, %c70_i32 : i32  // kTVMFFITensor = 70
llvm.cond_br %is_tensor, ^bb_decode_tensor, ^bb_type_error

^bb_decode_tensor:
  // Load DLTensor pointer
  %dltensor_ptr = llvm.load %args[0].v_ptr : !llvm.ptr

  // Load and check ndim
  %ndim = llvm.load %dltensor_ptr[2] : i32
  %ndim_ok = llvm.icmp "eq" %ndim, %c1_i32 : i32
  llvm.cond_br %ndim_ok, ^bb_check_dtype, ^bb_ndim_error

^bb_check_dtype:
  // Load and check dtype (Float32 = code:2, bits:32, lanes:1)
  %dtype_code = llvm.load %dltensor_ptr[3][0] : i8
  %dtype_bits = llvm.load %dltensor_ptr[3][1] : i8
  %dtype_lanes = llvm.load %dltensor_ptr[3][2] : i16
  %code_ok = llvm.icmp "eq" %dtype_code, %c2_i8 : i8
  %bits_ok = llvm.icmp "eq" %dtype_bits, %c32_i8 : i8
  %lanes_ok = llvm.icmp "eq" %dtype_lanes, %c1_i16 : i16
  %dtype_ok = llvm.and %code_ok, llvm.and %bits_ok, %lanes_ok : i1
  llvm.cond_br %dtype_ok, ^bb_check_device, ^bb_dtype_error

^bb_check_device:
  // Load and check device type (CUDA = 2)
  %device_type = llvm.load %dltensor_ptr[1][0] : i32
  %device_ok = llvm.icmp "eq" %device_type, %c2_i32 : i32
  llvm.cond_br %device_ok, ^bb_load_shape, ^bb_device_error

^bb_load_shape:
  // Load shape pointer and extract dimensions
  %shape_ptr = llvm.load %dltensor_ptr[4] : !llvm.ptr
  %n_value = llvm.load %shape_ptr[0] : i64

  // Bind symbolic variable 'n' (or check if already bound)
  // If this is the first tensor with 'n', store %n_value
  // If 'n' was already bound from another tensor, check equality

  // Store data pointer for later use
  %data = llvm.load %dltensor_ptr[0] : !llvm.ptr<1>  // GPU address space 1

  // Continue to next parameter...
```

#### 6.3: Call Provider Execution

**Location**: [python/CuTeDSL/cutlass/cutlass_dsl/tvm_ffi_provider.py:387-400](../../python/CuTeDSL/cutlass/cutlass_dsl/tvm_ffi_provider.py#L387-L400)

The `TVMFFICuteCallProvider.__call__()` method generates:

1. **CUDA Initialization**:
   ```python
   current_block = self.insert_lazy_init_cuda(current_block, context)
   ```

   Generates LLVM IR that calls `cuda_dialect_init_library_once` to:
   - Load CUDA driver library
   - Initialize CUDA context
   - Load the compiled CUDA module (CUBIN)

2. **Device Management**:
   ```python
   self.cuda_device_index = self.find_cuda_device_index_from_params(context)
   ```

   Extracts device ID from tensor parameters and generates:
   ```mlir
   %device_id = ... // from tensor's device field
   %old_device = llvm.call @_cudaGetDevice(...)

   // Switch device if needed
   %need_switch = llvm.icmp "ne" %old_device, %device_id : i32
   llvm.cond_br %need_switch, ^bb_set_device, ^bb_call_kernel

   ^bb_set_device:
     %set_result = llvm.call @_cudaSetDevice(%device_id)
     llvm.br ^bb_call_kernel
   ```

3. **Pack Parameters and Call Kernel**:
   ```python
   current_block = super().__call__(current_block, context)
   ```

   This calls the base class `DynamicParamPackCallProvider` which:
   - Packs decoded parameters into a struct
   - Calls the original CuTe kernel function
   - Handles error checking

   ```mlir
   // Pack parameters (simplified)
   %packed_args = llvm.alloca ...
   llvm.store %data_a, %packed_args[0]
   llvm.store %data_b, %packed_args[1]
   llvm.store %n, %packed_args[2]

   // Call the kernel
   %kernel_result = llvm.call @cutlass_add_one(%packed_args) : i32

   // Restore device
   llvm.cond_br %need_switch, ^bb_restore_device, ^bb_check_error

   ^bb_restore_device:
     %restore_result = llvm.call @_cudaSetDevice(%old_device)
     llvm.br ^bb_check_error

   ^bb_check_error:
     %is_success = llvm.icmp "eq" %kernel_result, %c0_i32 : i32
     llvm.cond_br %is_success, ^bb_return_success, ^bb_return_error
   ```

4. **Error Handling**:
   ```python
   self.cuda_error_handle_block = self.create_shared_cuda_error_block(current_block, context)
   ```

   Creates a shared error block that all CUDA API calls branch to on failure:
   ```mlir
   ^bb_cuda_error(%error_code: i32):
     %error_str = llvm.call @cuda_dialect_get_error_name(%error_code)
     llvm.call @TVMFFIErrorSetRaisedFromCStr("RuntimeError", %error_str)
     llvm.return %c_minus1_i32 : i32
   ```

**Final FFI Wrapper Structure:**
```mlir
llvm.func @__tvm_ffi_cutlass_add_one(%handle: !llvm.ptr, %args: !llvm.ptr,
                                      %num_args: i32, %result: !llvm.ptr) -> i32 {
  // 1. Initialize CUDA (once)
  %init_result = llvm.call @cuda_dialect_init_library_once(...)
  llvm.cond_br %init_ok, ^bb_check_args, ^bb_error

^bb_check_args:
  // 2. Validate argument count
  %args_ok = llvm.icmp "eq" %num_args, %expected : i32
  llvm.cond_br %args_ok, ^bb_decode_args, ^bb_arg_count_error

^bb_decode_args:
  // 3. Decode and validate each parameter
  // ... (see section 6.2 above) ...

^bb_get_device:
  // 4. Get current device
  %old_device_alloca = llvm.alloca i32
  %get_result = llvm.call @_cudaGetDevice(%old_device_alloca)
  llvm.cond_br %get_ok, ^bb_switch_device, ^bb_cuda_error(%get_result)

^bb_switch_device:
  // 5. Switch to target device if needed
  %old_device = llvm.load %old_device_alloca : i32
  %target_device = ... // from tensor parameters
  %need_switch = llvm.icmp "ne" %old_device, %target_device : i32
  llvm.cond_br %need_switch, ^bb_do_switch, ^bb_call_kernel

^bb_do_switch:
  %set_result = llvm.call @_cudaSetDevice(%target_device)
  llvm.cond_br %set_ok, ^bb_call_kernel, ^bb_cuda_error(%set_result)

^bb_call_kernel:
  // 6. Pack parameters and call the actual kernel
  %kernel_result = llvm.call @cutlass_add_one(%data_a, %data_b, %n, ...)
  llvm.br ^bb_restore_device

^bb_restore_device:
  // 7. Restore original device
  llvm.cond_br %need_switch, ^bb_do_restore, ^bb_check_kernel_result

^bb_do_restore:
  %restore_result = llvm.call @_cudaSetDevice(%old_device)
  llvm.br ^bb_check_kernel_result

^bb_check_kernel_result:
  // 8. Check kernel result
  %kernel_ok = llvm.icmp "eq" %kernel_result, %c0_i32 : i32
  llvm.cond_br %kernel_ok, ^bb_success, ^bb_cuda_error(%kernel_result)

^bb_success:
  // 9. Return success
  llvm.return %c0_i32 : i32

^bb_cuda_error(%error_code: i32):
  // 10. Handle errors
  %error_str = llvm.call @cuda_dialect_get_error_name(%error_code)
  llvm.call @TVMFFIErrorSetRaisedFromCStr("RuntimeError", %error_str)
  llvm.return %c_minus1_i32 : i32

^bb_arg_count_error:
  llvm.call @TVMFFIErrorSetRaisedFromCStr("TypeError", "Wrong number of arguments")
  llvm.return %c_minus1_i32 : i32
}
```

---

### Frame 7: MLIR Compilation Pipeline

**Location**: [python/CuTeDSL/cutlass/base_dsl/compiler.py:136-161](../../python/CuTeDSL/cutlass/base_dsl/compiler.py#L136-L161)

After the FFI wrapper is attached, the complete MLIR module goes through the compilation pipeline:

```python
def compile(self, module, pipeline: str, cuda_toolkit: str = "", arch: str = "",
            enable_verifier=False):
    try:
        pm = self.passmanager.PassManager.parse(pipeline)
        pm.enable_verifier(enable_verifier)
        pm.run(module.operation)
    except Exception as e:
        # ... error handling ...

    if self._post_compile_hook:
        self._post_compile_hook(module)
```

**Pipeline Stages:**

1. **MLIR Passes**:
   - `cute-to-nvvm`: Convert CuTe dialect operations to NVVM (CUDA) dialect
   - Optimization passes
   - LLVM lowering

2. **Post-Compile Hook Execution**:
   - **This is when `attach_ffi_func()` runs**
   - The FFI wrapper is added to the already-compiled MLIR module
   - Module verification ensures the IR is valid

3. **NVVM to PTX**:
   - NVVM IR is compiled to PTX (CUDA assembly)
   - PTX is assembled to CUBIN (CUDA binary)

**Generated Pipeline (with TVM FFI):**
```
builtin.module(
  cute-to-nvvm{
    cubin-format=bin
    enable-cuda-dialect=true
    cuda-dialect-external-module=true
    cubin-chip=sm_90a
    ...
  }
)
```

The `enable-cuda-dialect=true` and `cuda-dialect-external-module=true` flags enable:
- CUDA dialect operations (like `cuda.launch_ex`)
- External module loading (for the CUBIN)

**After Compilation:**

The module now contains:
- **Original kernel**: `@cutlass_add_one` (compiled to NVVM/PTX/CUBIN)
- **FFI wrapper**: `@__tvm_ffi_cutlass_add_one` (LLVM IR that calls the kernel)
- **Helper functions**: Error handling, CUDA init, etc.

---

### Frame 8: TVM FFI Function Initialization

**Location**: [python/CuTeDSL/cutlass/cutlass_dsl/tvm_ffi_provider.py:403-437](../../python/CuTeDSL/cutlass/cutlass_dsl/tvm_ffi_provider.py#L403-L437)

After compilation, a `TVMFFIJitCompiledFunction` object is created:

```python
class TVMFFIJitCompiledFunction(tvm_ffi.Function, CudaDialectJitCompiledFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_ffi_function()

    def _init_ffi_function(self):
        if self.__chandle__() != 0:
            raise DSLRuntimeError("TVM FFI function is already initialized")

        # Get function pointer from execution engine
        tvm_ffi_function_ptr = self.engine.raw_lookup("__tvm_ffi_" + self.function_name)

        # Create TVM FFI function from pointer
        tvm_ffi_function = tvm_ffi.Function.__from_mlir_packed_safe_call__(
            tvm_ffi_function_ptr
        )

        # Transfer ownership
        self.__move_handle_from__(tvm_ffi_function)
```

**What Happens:**

1. **Lookup Function Pointer**:
   - Uses MLIR ExecutionEngine to find the compiled `__tvm_ffi_cutlass_add_one` function
   - Returns a raw function pointer (C function pointer)

2. **Create TVM FFI Function**:
   - `__from_mlir_packed_safe_call__` is a TVM-FFI API that wraps the function pointer
   - Creates a TVM `PackedFunc` object
   - This object implements the TVM FFI calling convention

3. **Transfer Handle**:
   - Moves the TVM FFI handle from the temporary object to the `TVMFFIJitCompiledFunction`
   - The function is now ready to be called

**TVM FFI C++ Side:**

On the TVM side (C++), the function is registered with signature:
```cpp
// From tvm-ffi/include/tvm/ffi/function.h
typedef int (*FFIFunc)(void* handle, void* args, int num_args, void* ret);

// The function pointer points to __tvm_ffi_cutlass_add_one
FFIFunc ffi_func = (FFIFunc)function_ptr;
```

---

### Frame 9: Runtime Execution (When Called)

**Location**: Python call site (user code)

```python
a_torch = torch.arange(10, dtype=torch.float32, device="cuda")
b_torch = torch.empty(10, dtype=torch.float32, device="cuda")
compiled_add_one(a_torch, b_torch)  # This calls the TVM FFI function
```

**Call Stack:**

#### 9.1: Python → TVM FFI
**Location**: [python/CuTeDSL/cutlass/cutlass_dsl/tvm_ffi_provider.py:416](../../python/CuTeDSL/cutlass/cutlass_dsl/tvm_ffi_provider.py#L416)

```python
__call__ = tvm_ffi.Function.__call__
```

This directly calls the TVM FFI function's `__call__` method, which:

1. **Converts Python arguments to TVM FFI Any objects**:
   ```python
   # In TVM FFI (Python)
   ffi_args = [convert_to_ffi_any(arg) for arg in (a_torch, b_torch)]
   ```

   For `torch.Tensor`:
   - Extracts DLPack capsule: `tensor.__dlpack__()`
   - Wraps in `tvm_ffi.Tensor` object
   - Creates `TVMFFIAny` union with type index `kTVMFFITensor`

2. **Calls C++ FFI function**:
   ```python
   # In TVM FFI (Python binding)
   result = self._ffi_func(ffi_args)
   ```

#### 9.2: TVM FFI C++ Wrapper
**Location**: `tvm-ffi/include/tvm/ffi/function.h` and `tvm-ffi/src/function.cc` (TVM upstream)

The TVM FFI C++ layer:

**FunctionObj Structure:**
```cpp
// From tvm-ffi/include/tvm/ffi/function.h
class FunctionObj : public Object, public TVMFFIFunctionCell {
 public:
  using FCall = void (*)(const FunctionObj*, const AnyView*, int32_t, Any*);

  void CallPacked(const AnyView* args, int32_t num_args, Any* result) const {
    // Choose call path: cpp_call (fast) or safe_call (cross-boundary safe)
    FCall call_ptr = this->cpp_call
        ? reinterpret_cast<FCall>(this->cpp_call)
        : CppCallDedirectToSafeCall;
    (*call_ptr)(this, args, num_args, result);
  }

 private:
  static void CppCallDedirectToSafeCall(
      const FunctionObj* func, const AnyView* args, int32_t num_args, Any* rv) {
    // Call safe_call and check return code
    TVM_FFI_CHECK_SAFE_CALL(
        func->safe_call(func,
                       reinterpret_cast<const TVMFFIAny*>(args),
                       num_args,
                       reinterpret_cast<TVMFFIAny*>(rv))
    );
  }
};
```

**TVMFFIFunctionCell Structure:**
```cpp
// From tvm-ffi/include/tvm/ffi/c_api.h:440-454
typedef int (*TVMFFISafeCallType)(void* handle, const TVMFFIAny* args,
                                   int32_t num_args, TVMFFIAny* ret);

typedef struct {
  TVMFFISafeCallType safe_call;  // Cross-boundary safe function pointer
  void* cpp_call;                // C++ fast path (NULL for non-C++ functions)
} TVMFFIFunctionCell;
```

**Call Flow:**

1. **Python to C++**: Python `__call__` invokes `FunctionObj::CallPacked()`
2. **Choose Call Path**:
   - If `cpp_call` is set: Direct C++ call (fast, same exception handling)
   - Otherwise: Use `safe_call` via `CppCallDedirectToSafeCall` (safe, cross-boundary)
3. **Safe Call**: For MLIR-generated functions, `safe_call` points to `__tvm_ffi_<name>`
4. **Error Handling**:
   ```cpp
   #define TVM_FFI_CHECK_SAFE_CALL(func) \
     { \
       int ret_code = (func); \
       if (ret_code != 0) { \
         if (ret_code == -2) throw EnvErrorAlreadySet(); \
         throw MoveFromSafeCallRaised(); \
       } \
     }
   ```

**Example for our function:**
```cpp
// The FunctionObj for __tvm_ffi_cutlass_add_one
FunctionObj func_obj = {
  .safe_call = (TVMFFISafeCallType)&__tvm_ffi_cutlass_add_one,
  .cpp_call = nullptr  // Not a native C++ function
};

// When called from Python
func_obj.CallPacked(args, 2, nullptr);
  → CppCallDedirectToSafeCall(func_obj, args, 2, nullptr)
    → __tvm_ffi_cutlass_add_one(nullptr, args, 2, nullptr)
      → [LLVM-generated code from Frame 6]
```

**TVMFFIAny Structure:**
```cpp
// From tvm-ffi/include/tvm/ffi/any.h
struct TVMFFIAny {
    int32_t type_index;  // Type discriminator (e.g., kTVMFFITensor = 70)
    int32_t padding;
    union {
        int64_t v_int64;
        double v_float64;
        void* v_ptr;
        DLTensor* v_dltensor;
        // ... other types ...
    };
};
```

**Example Args Array for `compiled_add_one(a_torch, b_torch)`:**
```cpp
TVMFFIAny args[2] = {
    {
        .type_index = 70,  // kTVMFFITensor
        .v_ptr = &dltensor_a  // Points to DLTensor for a_torch
    },
    {
        .type_index = 70,  // kTVMFFITensor
        .v_ptr = &dltensor_b  // Points to DLTensor for b_torch
    }
};
```

**DLTensor Structure:**
```cpp
// From dlpack.h
typedef struct {
    void* data;              // GPU memory pointer (e.g., 0x7f1234567000)
    DLDevice device;         // {.device_type = kDLCUDA, .device_id = 0}
    int32_t ndim;            // 1
    DLDataType dtype;        // {.code = kDLFloat, .bits = 32, .lanes = 1}
    int64_t* shape;          // [10]
    int64_t* strides;        // nullptr (contiguous)
    uint64_t byte_offset;    // 0
} DLTensor;
```

#### 9.3: LLVM-Generated FFI Wrapper Execution
**Location**: Compiled native code from Frame 6

The `__tvm_ffi_cutlass_add_one` function executes:

1. **Validate Arguments**:
   ```
   Check num_args == 2
   For each argument:
     Check type_index == kTVMFFITensor
     Load DLTensor pointer
     Validate dtype (Float32)
     Validate device (CUDA)
     Validate ndim (1)
     Extract shape[0] → n
     Check n matches between tensors (symbolic constraint)
   ```

2. **Initialize CUDA** (first call only):
   ```
   Call cuda_dialect_init_library_once()
     → Loads CUDA driver
     → Loads compiled CUBIN module
     → Gets function pointer to kernel
   ```

3. **Manage Device Context**:
   ```
   old_device = cudaGetDevice()
   if old_device != target_device:
       cudaSetDevice(target_device)
   ```

4. **Call Kernel**:
   ```
   result = cutlass_add_one(data_a, data_b, n, grid, block, ...)
     → Launches CUDA kernel via cuda.launch_ex
     → Kernel executes: b[tid] = a[tid] + 1.0
   ```

5. **Restore Device**:
   ```
   if old_device != target_device:
       cudaSetDevice(old_device)
   ```

6. **Return Success**:
   ```
   return 0  // Success
   ```

#### 9.4: CUDA Kernel Execution
**Location**: Compiled CUBIN on GPU

The actual CUDA kernel `device_add_one` runs on the GPU:

```cuda
// Pseudo-code for the compiled kernel
__global__ void device_add_one(float* a, float* b, int64_t n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        b[tid] = a[tid] + 1.0f;
    }
}
```

**Execution:**
- Grid: `(1, 1, 1)` (for n=10, with 128 threads per block)
- Block: `(128, 1, 1)`
- Threads: Thread 0-9 process elements, threads 10-127 exit early

**Memory Access:**
- `a`: GPU memory at `a_torch.data_ptr()` (e.g., `0x7f1234567000`)
- `b`: GPU memory at `b_torch.data_ptr()` (e.g., `0x7f1234568000`)

---

## Key Data Structures

### 1. `spec.Param` Hierarchy

**Location**: [python/CuTeDSL/cutlass/base_dsl/tvm_ffi_builder/spec.py](../../python/CuTeDSL/cutlass/base_dsl/tvm_ffi_builder/spec.py)

```python
class Param:
    name: str

class Var(Param):
    dtype: tvm_ffi.dtype
    divisibility: Optional[int]

class Shape(Param):
    shape: List[Union[Var, int]]  # e.g., [Var('n', 'int64'), 128]

class Tensor(Param):
    data: Var                     # Pointer to data
    shape: List[Union[Var, int]]  # Tensor dimensions
    dtype: tvm_ffi.dtype          # Element type
    strides: Optional[List[Union[Var, int]]]  # Layout
    data_alignment: Optional[int] # Memory alignment
    device_type: str              # "cuda", "cpu", etc.
    device_id: Var                # Device index

class Stream(Param):
    var: Var  # Stream handle

class DataPointer(Param):
    var: Var
    address_space: Optional[int]  # LLVM address space (e.g., 1 for GPU)
```

**Example Conversion:**

For `a: cute.Tensor` with shape `(n,)` and dtype `Float32` on GPU:

```python
spec.Tensor(
    name="a",
    data=spec.Var("a_data", "ptr"),
    shape=[spec.Var("n", "int64")],
    dtype=tvm_ffi.dtype("float32"),
    strides=None,  # Contiguous
    data_alignment=128,  # Assume 128-byte alignment
    device_type="cuda",
    device_id=spec.Var("a_device_id", "int32"),
)
```

### 2. `TVMFFIAny` Union Type

**C++ Structure** (from `tvm-ffi/include/tvm/ffi/any.h`):

```cpp
struct TVMFFIAny {
    int32_t type_index;  // Discriminator
    int32_t padding;     // Alignment
    union {
        int64_t v_int64;      // For Int, Bool
        double v_float64;     // For Float
        void* v_ptr;          // For Tensor, Shape, Handle, String, etc.
    };
};
```

**Type Index Constants:**
```cpp
enum TVMFFITypeIndex {
    kTVMFFINone = 0,
    kTVMFFIInt = 1,
    kTVMFFIBool = 2,
    kTVMFFIFloat = 3,
    kTVMFFIOpaquePtr = 4,
    // ...
    kTVMFFITensor = 70,
    kTVMFFIShape = 69,
    kTVMFFIArray = 71,
    // ...
};
```

**Memory Layout:**
```
Offset | Field        | Size | Description
-------|--------------|------|----------------------------------
0x00   | type_index   | 4B   | Type discriminator (e.g., 70 for Tensor)
0x04   | padding      | 4B   | Padding for alignment
0x08   | v_int64 /    | 8B   | Union: int64, float64, or pointer
       | v_float64 /  |      |
       | v_ptr        |      |
-------|--------------|------|----------------------------------
Total: 16 bytes per TVMFFIAny
```

### 3. `DLTensor` Structure

**From DLPack** (`tvm-ffi/include/dlpack/dlpack.h`):

```cpp
typedef struct {
    void* data;              // Pointer to tensor data
    DLDevice device;         // Device info (type, id)
    int32_t ndim;            // Number of dimensions
    DLDataType dtype;        // Element data type
    int64_t* shape;          // Pointer to shape array
    int64_t* strides;        // Pointer to strides array (or nullptr)
    uint64_t byte_offset;    // Byte offset into data pointer
} DLTensor;

typedef struct {
    int32_t device_type;     // kDLCPU=1, kDLCUDA=2, etc.
    int32_t device_id;       // Device index
} DLDevice;

typedef struct {
    uint8_t code;            // kDLInt=0, kDLUInt=1, kDLFloat=2, etc.
    uint8_t bits;            // Bits per element (e.g., 32 for float32)
    uint16_t lanes;          // Vector lanes (usually 1)
} DLDataType;
```

**Example Instance (for `a_torch`):**
```cpp
DLTensor dltensor_a = {
    .data = (void*)0x7f1234567000,  // GPU memory
    .device = {
        .device_type = 2,  // kDLCUDA
        .device_id = 0
    },
    .ndim = 1,
    .dtype = {
        .code = 2,   // kDLFloat
        .bits = 32,
        .lanes = 1
    },
    .shape = (int64_t[]){10},
    .strides = nullptr,  // Contiguous
    .byte_offset = 0
};
```

---

## TVM FFI ABI Interface

### Calling Convention

All TVM FFI functions follow this C signature:
```c
int32_t __tvm_ffi_<name>(void* handle, void* args, int32_t num_args, void* ret);
```

**Parameters:**
- `handle`: Reserved for object handle (unused for static functions)
- `args`: Array of `TVMFFIAny` unions
- `num_args`: Number of arguments
- `ret`: Pointer to return value location (unused for void functions)

**Return Value:**
- `0`: Success
- `-1`: Error (error message set via `TVMFFIErrorSetRaisedFromCStr`)

### Error Handling

**Setting Errors:**
```c
// In MLIR-generated code
void TVMFFIErrorSetRaisedFromCStr(const char* error_kind, const char* message);

// Example usage
TVMFFIErrorSetRaisedFromCStr("TypeError", "Expected Tensor, got Int");
return -1;
```

**Multiple Message Parts (for string deduplication):**
```c
void TVMFFIErrorSetRaisedFromCStrParts(const char* kind, const char** parts, int32_t num_parts);

// Example usage
const char* parts[] = {"Mismatched type on argument #0", " in function foo"};
TVMFFIErrorSetRaisedFromCStrParts("TypeError", parts, 2);
return -1;
```

### Environment Stream

For GPU operations that don't explicitly pass a stream, TVM FFI provides:
```c
void* TVMFFIEnvGetStream(int32_t device_type, int32_t device_id);
```

This queries the current stream from the TVM environment (synchronized with framework like PyTorch).

---

## Summary

The complete call path from `cute.compile()` to execution involves:

1. **Python Entry** (`cute.compile()`): Parse options, create `CompileCallable`
2. **Compilation Setup** (`CompileCallable._compile()`): Set `EnableTVMFFI` flag
3. **DSL Processing**: Generate MLIR for the kernel
4. **Args Conversion** (`_tvm_ffi_args_spec_converter`): Convert CuTe types to `spec.Param`
5. **Post-Compile Hook**: Register `attach_ffi_func()` to run after MLIR passes
6. **FFI Wrapper Generation** (`TVMFFIFunctionBuilder.attach_ffi_func()`):
   - Create `__tvm_ffi_<name>` function
   - Generate parameter decoding logic
   - Add error handling
   - Call the original kernel via call provider
7. **MLIR Compilation**: Compile MLIR → NVVM → PTX → CUBIN
8. **Function Initialization** (`TVMFFIJitCompiledFunction._init_ffi_function()`):
   - Look up function pointer in execution engine
   - Create TVM FFI function object
9. **Runtime Execution**:
   - Python: Convert PyTorch tensors to TVM FFI Any objects
   - C++: Call MLIR-generated FFI wrapper
   - LLVM IR: Decode arguments, validate, manage device context
   - CUDA: Launch kernel on GPU

**Key Insights:**

- **Zero-Copy**: PyTorch tensors are passed via DLPack (no data copy)
- **Type Safety**: Extensive runtime validation in FFI wrapper
- **Device Management**: Automatic CUDA device switching
- **Error Handling**: Detailed error messages with context
- **Lazy Initialization**: CUDA is initialized on first call
- **Symbolic Variables**: Shape constraints are checked at runtime

This integration enables seamless interoperability between CuTeDSL-compiled kernels and TVM-FFI compatible frameworks while maintaining performance and type safety.

---

## Visual Call Stack Diagram

Here's a complete visual representation of the call stack from Python down to GPU execution:

```
┌─────────────────────────────────────────────────────────────────┐
│                   PYTHON LAYER                                  │
├─────────────────────────────────────────────────────────────────┤
│  compiled_add_one(a_torch, b_torch)                            │
│  ↓                                                              │
│  cute.compile(..., options="--enable-tvm-ffi")                 │
│    [Frame 1-2: Parse options, set EnableTVMFFI flag]           │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                CUTEDSL COMPILATION LAYER                        │
├─────────────────────────────────────────────────────────────────┤
│  [Frame 3] DSL Processing: Python AST → MLIR                   │
│    • Generate kernel function (device_add_one)                  │
│    • Generate host function (cutlass_add_one)                   │
│  ↓                                                              │
│  [Frame 4] MLIR Generation:                                     │
│    • cuda.kernel @device_add_one                                │
│    • func.func @cutlass_add_one (host launch)                   │
│  ↓                                                              │
│  [Frame 5] Post-Compile Hook Setup:                             │
│    • Convert args: cute.Tensor → spec.Tensor                    │
│    • Create TVMFFICuteCallProvider                              │
│    • Register attach_ffi_func callback                          │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│              TVM FFI WRAPPER GENERATION LAYER                   │
├─────────────────────────────────────────────────────────────────┤
│  [Frame 6] attach_ffi_func() - TVMFFIFunctionBuilder           │
│    • Generate __tvm_ffi_cutlass_add_one(handle, args, n, ret)  │
│    • Argument decoding logic (for each parameter):              │
│        - Load type_index from TVMFFIAny                         │
│        - Extract DLTensor pointer                               │
│        - Validate dtype, device, ndim                           │
│        - Extract and bind shape variables                       │
│    • CUDA initialization (cuda_dialect_init_library_once)       │
│    • Device management (cudaGetDevice, cudaSetDevice)           │
│    • Call original kernel via TVMFFICuteCallProvider            │
│    • Error handling and return                                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                   MLIR COMPILATION LAYER                        │
├─────────────────────────────────────────────────────────────────┤
│  [Frame 7] MLIR Passes:                                         │
│    • cute-to-nvvm: CuTe dialect → NVVM dialect                  │
│    • Optimization passes                                        │
│    • LLVM lowering                                              │
│    • NVVM → PTX → CUBIN                                         │
│                                                                 │
│  Result: Two functions in the module:                           │
│    1. @cutlass_add_one (original kernel launcher)               │
│    2. @__tvm_ffi_cutlass_add_one (FFI wrapper)                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│              TVM FFI FUNCTION INITIALIZATION                    │
├─────────────────────────────────────────────────────────────────┤
│  [Frame 8] TVMFFIJitCompiledFunction._init_ffi_function()      │
│    • Lookup function pointer from ExecutionEngine:              │
│        ptr = engine.raw_lookup("__tvm_ffi_cutlass_add_one")    │
│    • Create TVM Function object:                                │
│        func = Function.__from_mlir_packed_safe_call__(ptr)     │
│    • Transfer ownership to TVMFFIJitCompiledFunction            │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌═════════════════════════════════════════════════════════════════┐
║                 RUNTIME EXECUTION (USER CALL)                   ║
╠═════════════════════════════════════════════════════════════════╣
║  compiled_add_one(a_torch, b_torch)                            ║
╚═════════════════════════════════════════════════════════════════╝
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                  PYTHON → TVM FFI BOUNDARY                      │
├─────────────────────────────────────────────────────────────────┤
│  [Frame 9.1] TVMFFIJitCompiledFunction.__call__()              │
│    • Convert PyTorch tensors to TVM FFI:                        │
│        torch.Tensor → DLTensor → TVMFFIAny                      │
│        ┌─────────────────────────────────────┐                 │
│        │ TVMFFIAny[0] = {                    │                 │
│        │   type_index: 70 (kTVMFFITensor)    │                 │
│        │   v_ptr: &dltensor_a                │                 │
│        │ }                                   │                 │
│        │ TVMFFIAny[1] = {                    │                 │
│        │   type_index: 70 (kTVMFFITensor)    │                 │
│        │   v_ptr: &dltensor_b                │                 │
│        │ }                                   │                 │
│        └─────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    TVM FFI C++ LAYER                            │
├─────────────────────────────────────────────────────────────────┤
│  [Frame 9.2] FunctionObj::CallPacked()                          │
│    • Choose call path: cpp_call or safe_call                    │
│    • For MLIR functions: use safe_call                          │
│  ↓                                                              │
│  CppCallDedirectToSafeCall()                                   │
│    • Call: safe_call(handle, args, num_args, ret)              │
│    • Check return code                                          │
│    • If error: throw exception from error message               │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│             LLVM-GENERATED FFI WRAPPER (NATIVE)                 │
├─────────────────────────────────────────────────────────────────┤
│  [Frame 9.3] __tvm_ffi_cutlass_add_one(handle, args, 2, ret)  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 1. Check num_args == 2                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 2. Initialize CUDA (first call only)                      │  │
│  │    cuda_dialect_init_library_once()                       │  │
│  │      → Load CUDA driver                                   │  │
│  │      → Load CUBIN module                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 3. Decode args[0] (tensor a):                             │  │
│  │    • Load type_index → check == kTVMFFITensor             │  │
│  │    • Load v_ptr → dltensor_a                              │  │
│  │    • Validate:                                            │  │
│  │      - ndim == 1                                          │  │
│  │      - dtype == Float32 (code:2, bits:32, lanes:1)        │  │
│  │      - device.type == kDLCUDA (2)                         │  │
│  │    • Extract:                                             │  │
│  │      - data_a = dltensor_a.data (GPU ptr)                 │  │
│  │      - n_a = dltensor_a.shape[0]                          │  │
│  │      - device_id_a = dltensor_a.device.device_id          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 4. Decode args[1] (tensor b):                             │  │
│  │    • [Same validation as args[0]]                         │  │
│  │    • Check symbolic constraint: n_b == n_a                │  │
│  │    • Extract data_b, device_id_b                          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 5. Manage Device Context:                                 │  │
│  │    old_device = cudaGetDevice()                           │  │
│  │    if old_device != device_id_a:                          │  │
│  │        cudaSetDevice(device_id_a)                         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 6. Call Kernel:                                           │  │
│  │    result = cutlass_add_one(data_a, data_b, n_a, ...)    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 7. Restore Device:                                        │  │
│  │    if old_device != device_id_a:                          │  │
│  │        cudaSetDevice(old_device)                          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 8. Check result and return:                               │  │
│  │    if result == 0: return 0                               │  │
│  │    else: set error and return -1                          │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│              CUDA KERNEL LAUNCH (HOST → GPU)                    │
├─────────────────────────────────────────────────────────────────┤
│  cutlass_add_one(data_a, data_b, n, ...)                       │
│    • Configure grid/block dimensions                            │
│    • Call cuda.launch_ex:                                       │
│        cuLaunchKernel(device_add_one_kernel,                   │
│                      grid_x, grid_y, grid_z,                   │
│                      block_x, block_y, block_z,                │
│                      shared_mem, stream,                       │
│                      kernel_args, nullptr)                     │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌═════════════════════════════════════════════════════════════════┐
║                     GPU EXECUTION                               ║
╠═════════════════════════════════════════════════════════════════╣
║  [Frame 9.4] device_add_one kernel on GPU                      ║
║                                                                 ║
║  __global__ void device_add_one(float* a, float* b, int64_t n) ║
║  {                                                              ║
║      int tid = blockIdx.x * blockDim.x + threadIdx.x;          ║
║      if (tid < n) {                                             ║
║          b[tid] = a[tid] + 1.0f;                                ║
║      }                                                          ║
║  }                                                              ║
║                                                                 ║
║  Grid: (1, 1, 1)    Block: (128, 1, 1)                         ║
║  Threads 0-9: Execute (for n=10)                                ║
║  Threads 10-127: Exit early                                     ║
╚═════════════════════════════════════════════════════════════════╝
                            ↓
                     Kernel completes
                            ↓
              Return through the entire stack
                (checking errors at each level)
                            ↓
              Python receives result or exception
```

---

## Call Path Summary Table

| Frame | Layer | Component | Input | Output | Location |
|-------|-------|-----------|-------|--------|----------|
| 1 | Python | `cute.compile()` | Function + args + options | `CompileCallable` | [cute/__init__.py:199](../../python/CuTeDSL/cutlass/cute/__init__.py#L199) |
| 2 | Python | `CompileCallable._compile()` | Function + `EnableTVMFFI` | DSL compilation trigger | [base_dsl/compiler.py:577](../../python/CuTeDSL/cutlass/base_dsl/compiler.py#L577) |
| 3 | Python | DSL Preprocessing | Python function | Execution args | [base_dsl/dsl.py](../../python/CuTeDSL/cutlass/base_dsl/dsl.py) |
| 4 | MLIR | MLIR Generation | DSL operations | MLIR module | [cutlass_dsl/cutlass.py:388](../../python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py#L388) |
| 5 | Python | Post-compile hook setup | Args + options | Hook callback | [cutlass_dsl/cutlass.py:419](../../python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py#L419) |
| 6 | MLIR | FFI wrapper generation | `spec.Param` list | `__tvm_ffi_*` function | [tvm_ffi_builder.py:1879](../../python/CuTeDSL/cutlass/base_dsl/tvm_ffi_builder/tvm_ffi_builder.py#L1879) |
| 7 | MLIR | Compilation pipeline | MLIR → NVVM → PTX → CUBIN | Compiled module | [base_dsl/compiler.py:136](../../python/CuTeDSL/cutlass/base_dsl/compiler.py#L136) |
| 8 | Python/C++ | Function initialization | Function pointer | `TVMFFIJitCompiledFunction` | [tvm_ffi_provider.py:418](../../python/CuTeDSL/cutlass/cutlass_dsl/tvm_ffi_provider.py#L418) |
| 9.1 | Python | Runtime call setup | PyTorch tensors | `TVMFFIAny` array | [tvm_ffi_provider.py:416](../../python/CuTeDSL/cutlass/cutlass_dsl/tvm_ffi_provider.py#L416) |
| 9.2 | C++ | TVM FFI dispatch | `TVMFFIAny` array | safe_call invocation | [tvm-ffi/include/tvm/ffi/function.h:125](../../tvm-ffi/include/tvm/ffi/function.h#L125) |
| 9.3 | LLVM | FFI wrapper execution | `TVMFFIAny` array | Kernel result | Generated native code |
| 9.4 | CUDA | Kernel execution | GPU arrays | Modified data | Compiled CUBIN |

---

## Comparison: With vs Without TVM-FFI

This section explains the architectural differences between compiling with and without `--enable-tvm-ffi`, focusing on how input tensors, calling conventions, and kernel execution are handled.

### High-Level Overview

| Aspect | **WITH TVM-FFI** | **WITHOUT TVM-FFI** |
|--------|------------------|---------------------|
| **Calling Convention** | Safe-call with error codes | Direct ctypes call |
| **Argument Passing** | DLPack → TVMFFIAny array | Direct ctypes pointers |
| **Type Safety** | Runtime validation in FFI wrapper | No automatic validation |
| **Error Handling** | Structured error codes + exceptions | CUDA error codes only |
| **Device Management** | Automatic via FFI wrapper | Manual via CUDA runtime |
| **Interoperability** | Framework-agnostic (DLPack) | CuTeDSL-specific |

---

### Call Path Comparison

#### Path 1: WITH TVM-FFI (`options="--enable-tvm-ffi"`)

```
User Call: compiled_add_one(a_torch, b_torch)
    ↓
TVMFFIJitCompiledFunction.__call__()            [tvm/ffi/function.h:125]
    ↓ (Python → C++ via ctypes)
FunctionObj::CallPacked(args, num_args, result) [tvm-ffi/include/tvm/ffi/function.h:125]
    ↓
safe_call function pointer invocation            [tvm-ffi/include/tvm/ffi/c_api.h:440]
    ↓
__tvm_ffi_cutlass_add_one(handle, args, num_args, ret)  [Generated LLVM IR]
    │
    ├─> Validate argument types (TVMFFIAny → DLTensor checks)
    ├─> Check tensor shapes against symbolic constraints
    ├─> Check device type and ID
    ├─> Insert lazy CUDA initialization (if needed)
    ├─> Switch to target device (if needed)
    │
    └─> Unpack TVMFFIAny → raw pointers
        ↓
    Original compiled function: cutlass_add_one(a_ptr, b_ptr, kernels...)
        ↓
    CUDA kernel launch via cuda.launch_ex
        ↓
    GPU execution
        ↓
    Restore original device (if switched)
        ↓
    Return error code (0 = success)
```

#### Path 2: WITHOUT TVM-FFI (`options=None`)

```
User Call: compiled_add_one(a_torch, b_torch)
    ↓
JitCompiledFunction.__call__(args, kwargs)      [base_dsl/jit_executor.py:671]
    ↓
generate_execution_args(args, kwargs)           [base_dsl/jit_executor.py:186]
    │
    ├─> Convert args via JitArgAdapterRegistry
    ├─> Extract __c_pointers__() from adapted args
    └─> Build exe_args list (flat ctypes pointer array)
        ↓
run_compiled_program(exe_args)                  [base_dsl/jit_executor.py:681]
    ↓
JitExecutor.run_compiled_program(exe_args)      [base_dsl/jit_executor.py:492]
    ↓
_get_invoke_packed_args(exe_args)               [base_dsl/jit_executor.py:474]
    │
    ├─> Append CUDA error result pointer
    ├─> Append kernel function pointers
    └─> Pack into ctypes.c_void_p array
        ↓
self.jit_module.capi_func(packed_args)          [Direct ctypes call]
    ↓
Original compiled function: cutlass_add_one(a_ptr, b_ptr, kernels..., err_ptr)
    ↓
CUDA kernel launch via cuda.launch_ex
    ↓
GPU execution
    ↓
Check err_ptr for CUDA errors
    ↓
Return None (or raise on error)
```

---

### Detailed Comparison by Stage

#### 1. Tensor Input Handling

**WITH TVM-FFI:**
```python
# User code
compiled_add_one(a_torch, b_torch)

# Internal: TVMFFICuteCallProvider wraps tensors
# Location: python/CuTeDSL/cutlass/cutlass_dsl/tvm_ffi_provider.py:376-416

class TVMFFICuteCallProvider(DynamicParamPackCallProvider):
    def wrap_param(self, arg, param_spec):
        if isinstance(param_spec, spec.TensorParam):
            # Convert PyTorch tensor to DLPack capsule
            capsule = arg.__dlpack__()
            # Capsule is wrapped into TVMFFIAny (union type)
            return capsule
```

**Data Flow:**
```
PyTorch Tensor (a_torch)
    ↓ .__dlpack__()
DLPack Capsule (PyCapsule object)
    ↓ TVMFFICuteCallProvider.wrap_param()
TVMFFIAny struct {
    .v_type = TVMFFIType_DLTensor,
    .v_dlpack = <capsule pointer>
}
    ↓ Passed to FFI function
FFI Wrapper unpacks:
    DLTensor* = TVMFFIAny.v_dlpack
    void* data_ptr = DLTensor->data
```

**WITHOUT TVM-FFI:**
```python
# User code
compiled_add_one(a_torch, b_torch)

# Internal: ExecutionArgs.generate_execution_args
# Location: python/CuTeDSL/cutlass/base_dsl/jit_executor.py:186-217

def generate_execution_args(self, args, kwargs):
    exe_args = []
    for arg in input_args:
        # Check if already converted
        if hasattr(arg, "__c_pointers__"):
            exe_args.extend(arg.__c_pointers__())
        else:
            # Use JitArgAdapterRegistry
            adapter = JitArgAdapterRegistry.get_registered_adapter(type(arg))
            if adapter:
                adapted_arg = adapter(arg)
                exe_args.extend(get_c_pointers(adapted_arg))
    return exe_args
```

**Data Flow:**
```
PyTorch Tensor (a_torch)
    ↓ JitArgAdapterRegistry.get_registered_adapter()
TensorAdapter wraps tensor
    ↓ .__c_pointers__()
[data_ptr, shape_ptr, stride_ptr, ...] (list of ctypes pointers)
    ↓ Flattened into exe_args
    ↓ Packed into ctypes.c_void_p array
Direct kernel parameter access
```

---

#### 2. Calling Convention

**WITH TVM-FFI:**
```c
// Generated signature in LLVM IR
// Location: Generated by tvm_ffi_builder.py:1879-1980

int32_t __tvm_ffi_cutlass_add_one(
    void* handle,           // Reserved for future use
    TVMFFIAny* args,        // Array of union-typed arguments
    int32_t num_args,       // Number of arguments
    TVMFFIAny* ret          // Return value (unused for void functions)
) {
    // Error handling: return non-zero on error
    if (validation_fails) {
        return -1;  // Error code
    }

    // Extract tensor pointers from TVMFFIAny array
    DLTensor* a_dltensor = args[0].v_dlpack;
    DLTensor* b_dltensor = args[1].v_dlpack;
    void* a_ptr = a_dltensor->data;
    void* b_ptr = b_dltensor->data;

    // Call original kernel
    cutlass_add_one(a_ptr, b_ptr, ...);

    return 0;  // Success
}
```

**Key Features:**
- **Safe-call convention**: Returns error code (int32_t)
- **Structured arguments**: TVMFFIAny array allows type introspection
- **Framework-agnostic**: Uses DLPack standard
- **Validation**: Type, shape, device checks before execution

**WITHOUT TVM-FFI:**
```c
// Generated signature in MLIR (after lowering to LLVM)
// Location: Compiled by MLIR ExecutionEngine

void cutlass_add_one(
    void* a_ptr,            // Raw pointer to tensor A data
    void* b_ptr,            // Raw pointer to tensor B data
    // ... more flattened tensor metadata ...
    void** kernel_ptrs,     // Array of CUDA kernel function pointers
    int32_t* err_ptr        // Output: CUDA error code
) {
    // No validation here
    // Direct kernel launch
    cuda_launch_ex(...);
    *err_ptr = cudaGetLastError();
}
```

**Key Features:**
- **Direct C ABI**: No wrapper overhead
- **Flattened arguments**: All pointers are explicit parameters
- **No type safety**: Relies on compile-time correctness
- **CUDA-only error handling**: Sets error pointer on GPU errors

---

#### 3. Argument Validation

**WITH TVM-FFI:**
```llvm
; Generated LLVM IR with validation
; Location: tvm_ffi_builder.py generates this structure

define i32 @__tvm_ffi_cutlass_add_one(ptr %handle, ptr %args, i32 %num_args, ptr %ret) {
entry:
  ; 1. Check number of arguments
  %arg_count_ok = icmp eq i32 %num_args, 2
  br i1 %arg_count_ok, label %check_arg0_type, label %error_wrong_arg_count

check_arg0_type:
  ; 2. Check argument 0 type (should be DLTensor)
  %arg0_ptr = getelementptr inbounds %TVMFFIAny, ptr %args, i32 0
  %arg0_type_ptr = getelementptr inbounds %TVMFFIAny, ptr %arg0_ptr, i32 0, i32 0
  %arg0_type = load i32, ptr %arg0_type_ptr
  %is_dltensor = icmp eq i32 %arg0_type, 14  ; TVMFFIType_DLTensor
  br i1 %is_dltensor, label %check_arg0_dtype, label %error_wrong_type

check_arg0_dtype:
  ; 3. Check DLTensor dtype (e.g., float32)
  %dltensor0 = load ptr, ptr %arg0_ptr, i32 0, i32 1
  %dtype_ptr = getelementptr inbounds %DLTensor, ptr %dltensor0, i32 0, i32 4
  %dtype_code = load i8, ptr %dtype_ptr
  %is_float = icmp eq i8 %dtype_code, 2  ; kDLFloat
  br i1 %is_float, label %check_arg0_shape, label %error_wrong_dtype

check_arg0_shape:
  ; 4. Check tensor shape
  %shape_ptr = getelementptr inbounds %DLTensor, ptr %dltensor0, i32 0, i32 2
  %shape = load ptr, ptr %shape_ptr
  %dim0 = load i64, ptr %shape, i64 0
  ; ... validate dimensions ...

  ; 5. Check device type and ID
  ; 6. Validate symbolic constraints (shared shapes)
  ; ... more checks ...

kernel_call:
  ; All validations passed - extract raw pointers
  %a_data_ptr = getelementptr inbounds %DLTensor, ptr %dltensor0, i32 0, i32 0
  %a_data = load ptr, ptr %a_data_ptr

  ; Call original function
  call void @cutlass_add_one(ptr %a_data, ...)
  br label %success

success:
  ret i32 0  ; Return success code

error_wrong_type:
  ; Set error message and return error code
  ret i32 -1

  ; ... more error handlers ...
}
```

**WITHOUT TVM-FFI:**
```llvm
; No validation wrapper - direct function
; Location: Generated by MLIR compiler

define void @cutlass_add_one(ptr %a_ptr, ptr %b_ptr, ptr %kernels, ptr %err_ptr) {
entry:
  ; No validation - trust caller
  ; Direct kernel launch
  %kernel0 = load ptr, ptr %kernels, i64 0

  ; cuda.launch_ex call
  call void @cuda_launch_ex(
    ptr %kernel0,
    i64 %gridX, i64 %gridY, i64 %gridZ,
    i64 %blockX, i64 %blockY, i64 %blockZ,
    ptr %a_ptr, ptr %b_ptr
  )

  ; Check CUDA error
  %cuda_err = call i32 @cudaGetLastError()
  store i32 %cuda_err, ptr %err_ptr
  ret void
}
```

---

#### 4. Device Management

**WITH TVM-FFI:**
```python
# Location: python/CuTeDSL/cutlass/cutlass_dsl/tvm_ffi_provider.py:287-358

class TVMFFICuteCallProvider(DynamicParamPackCallProvider):
    def __call__(self, current_block, context):
        # 1. Insert lazy CUDA initialization (one-time cost)
        current_block = self.insert_lazy_init_cuda(current_block, context)

        # 2. Find target device from tensor arguments
        self.cuda_device_index = self.find_cuda_device_index_from_params(context)

        # 3. Call parent to insert device switch + kernel call
        current_block = super().__call__(current_block, context)

        return current_block

    def insert_device_switch_if_necessary(self, current_block, context):
        """
        Generated MLIR:
        %current_device = cuda.get_device()
        %target_device = <from tensor>
        %needs_switch = arith.cmpi ne, %current_device, %target_device
        scf.if %needs_switch {
            cuda.set_device(%target_device)
        }
        // ... kernel call ...
        scf.if %needs_switch {
            cuda.set_device(%current_device)  // Restore
        }
        """
```

**Generated Device Management Code:**
```llvm
; Conceptual LLVM IR after MLIR lowering

define i32 @__tvm_ffi_cutlass_add_one(...) {
entry:
  ; Get current device
  %current_dev = call i32 @cudaGetDevice()

  ; Extract target device from DLTensor
  %dltensor = load ptr, ptr %args
  %device_ptr = getelementptr inbounds %DLTensor, ptr %dltensor, i32 0, i32 1
  %target_dev = load i32, ptr %device_ptr

  ; Check if device switch needed
  %needs_switch = icmp ne i32 %current_dev, %target_dev
  br i1 %needs_switch, label %switch_device, label %kernel_call

switch_device:
  call void @cudaSetDevice(i32 %target_dev)
  br label %kernel_call

kernel_call:
  ; Launch kernel on target device
  call void @cutlass_add_one(...)

  ; Restore device if switched
  br i1 %needs_switch, label %restore_device, label %exit

restore_device:
  call void @cudaSetDevice(i32 %current_dev)
  br label %exit

exit:
  ret i32 0
}
```

**WITHOUT TVM-FFI:**
```python
# Location: python/CuTeDSL/cutlass/base_dsl/jit_executor.py:635-661

class JitCompiledFunction:
    def to(self, device=None) -> JitExecutor:
        """Bind executor to specific device."""
        with self._executor_lock:
            # Load CUDA modules once
            if self.jit_module is None:
                cuda_modules = load_kernels_from_ir_module(
                    self.ir_module, self.kernel_info
                )
                self.jit_module = JitModule(
                    self.engine, self.capi_func, self.args_spec, cuda_modules
                )

            # Create device-specific context
            context = self.jit_module.get_device_execute_context(device)
            return JitExecutor(self.jit_module, context, self.jit_time_profiling)
```

**Manual Device Binding:**
```python
# User must explicitly bind to device
executor_gpu0 = compiled_add_one.to(0)  # Bind to GPU 0
executor_gpu1 = compiled_add_one.to(1)  # Bind to GPU 1

# Call on specific device
with torch.cuda.device(0):
    executor_gpu0(a_gpu0, b_gpu0)

with torch.cuda.device(1):
    executor_gpu1(a_gpu1, b_gpu1)
```

---

#### 5. Kernel Execution

**WITH TVM-FFI:**
```mlir
// Location: Generated by tvm_ffi_provider.py via parent call

func.func @__tvm_ffi_cutlass_add_one(...) {
    // ... validation and device management ...

    // Original function call
    %result = func.call @cutlass_add_one(%a_ptr, %b_ptr, %kernels) : (...) -> i32

    // Check result
    %is_success = arith.cmpi eq, %result, 0 : i32
    scf.if %is_success {
        func.return %c0_i32 : i32  // Success
    } else {
        func.return %result : i32  // Propagate error
    }
}

// Original function (shared by both paths)
func.func @cutlass_add_one(
    %a: !llvm.ptr,
    %b: !llvm.ptr,
    %kernels: !llvm.ptr<ptr<i8>>
) -> i32 {
    // CUDA kernel launch
    %kernel = llvm.load %kernels[0] : !llvm.ptr<ptr<i8>>
    %err = cuda.launch_ex %kernel(...) : i32
    func.return %err : i32
}
```

**WITHOUT TVM-FFI:**
```mlir
// Location: Generated by cutlass.py compilation

func.func @cutlass_add_one(
    %a: !llvm.ptr,
    %b: !llvm.ptr,
    %kernels: !llvm.ptr<ptr<i8>>,
    %err_ptr: !llvm.ptr<i32>
) {
    // Direct CUDA kernel launch (no wrapper)
    %kernel = llvm.load %kernels[0] : !llvm.ptr<ptr<i8>>
    %err = cuda.launch_ex %kernel(...) : i32

    // Store error in output pointer
    llvm.store %err, %err_ptr : i32
    func.return
}
```

---

### Performance Comparison

| Metric | **WITH TVM-FFI** | **WITHOUT TVM-FFI** |
|--------|------------------|---------------------|
| **Argument Conversion** | DLPack (zero-copy) | ctypes pointers (zero-copy) |
| **Validation Overhead** | ~200-500ns | None |
| **Device Switch** | Automatic (~500ns-2μs if needed) | Manual (user responsibility) |
| **Call Overhead** | ~6-23μs | ~5-10μs |
| **Type Safety** | Runtime checks | Compile-time only |
| **Interop** | Framework-agnostic | PyTorch-specific adapters |
| **Error Handling** | Structured error codes | CUDA errors only |

**When to Use Each:**
- **WITH TVM-FFI**:
  - Multi-framework interoperability needed
  - Runtime type safety required
  - Complex multi-device scenarios
  - Library APIs with external consumers

- **WITHOUT TVM-FFI**:
  - Maximum performance (eliminate validation overhead)
  - Internal kernels with trusted inputs
  - Single framework usage
  - Simpler debugging (fewer layers)

---

### Key Architectural Differences

#### 1. Function Signature Evolution

**WITH TVM-FFI:**
```
Python: compiled_add_one(a_torch, b_torch)
    ↓
C++ Wrapper: int32_t __tvm_ffi_cutlass_add_one(void*, TVMFFIAny*, i32, TVMFFIAny*)
    ↓
Internal: void cutlass_add_one(void*, void*, void**, int32_t*)
    ↓
CUDA: __global__ void add_one_kernel(float*, float*, int)
```

**WITHOUT TVM-FFI:**
```
Python: compiled_add_one(a_torch, b_torch)
    ↓
Direct: void cutlass_add_one(void*, void*, void**, int32_t*)
    ↓
CUDA: __global__ void add_one_kernel(float*, float*, int)
```

#### 2. Module Structure

**WITH TVM-FFI:**
```
TVMFFIJitCompiledFunction
    ├─ Inherits from: tvm.ffi.Function (TVM C++ object)
    ├─ Inherits from: CudaDialectJitCompiledFunction
    ├─ FFI wrapper: __tvm_ffi_cutlass_add_one
    └─ Original function: cutlass_add_one
```

**WITHOUT TVM-FFI:**
```
CudaDialectJitCompiledFunction
    ├─ Inherits from: JitCompiledFunction
    ├─ Direct function: cutlass_add_one
    └─ No wrapper layer
```

---

## Performance Considerations

### Zero-Copy Data Transfer
- **DLPack Protocol**: PyTorch tensors are passed via DLPack capsule
- **No Memory Copy**: Only pointer and metadata are transferred
- **Direct GPU Access**: FFI wrapper receives GPU pointers directly

### Validation Overhead
The TVM FFI wrapper performs extensive validation:
- Type checking (12 branches for tensor validation)
- Shape checking (per-dimension comparison)
- Device checking (device type and ID)
- Symbolic constraint checking (shared variables)

**Optimization**: Most branches are predictable and marked with `branch_weights=LIKELY`

### Device Management Overhead
- **Device Switching**: Only when target device differs from current
- **Lazy CUDA Init**: One-time cost on first call
- **Automatic Restoration**: Device is restored even on errors

### Call Overhead Breakdown (Approximate)
1. Python → C++: ~50-100ns (FFI boundary)
2. Argument validation: ~200-500ns (depends on complexity)
3. Device management: ~500ns-2μs (if switch needed)
4. Kernel launch: ~5-20μs (CUDA overhead)
5. Kernel execution: Depends on kernel complexity

**Total Overhead**: ~6-23μs for a simple kernel

---

## Debugging Tips

### 1. Enable Verbose Error Messages
Set environment variable:
```bash
export TVM_FFI_BACKTRACE=1
```

### 2. Inspect Generated MLIR
```python
compiled_add_one = cute.compile(add_one, a, b, options="--enable-tvm-ffi")
print(compiled_add_one.ir_module)  # View the MLIR with FFI wrapper
```

### 3. Check Function Pointer
```python
ptr = compiled_add_one.engine.raw_lookup("__tvm_ffi_cutlass_add_one")
print(f"FFI function pointer: {hex(ptr)}")
```

### 4. Validate Arguments Before Calling
```python
# Check tensor properties
print(f"a_torch device: {a_torch.device}, dtype: {a_torch.dtype}")
print(f"DLPack capsule: {a_torch.__dlpack__()}")
```

### 5. Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `Mismatched type on argument #0` | Wrong tensor type passed | Check dtype matches annotation |
| `Mismatched Tensor shape[0]` | Symbolic constraint violation | Ensure shapes match across tensors |
| `Mismatched Tensor device_type` | Tensor on wrong device | Move tensor to CUDA: `.to('cuda')` |
| `CUDA Error: ...` | CUDA runtime error | Check CUDA initialization and device |

---
