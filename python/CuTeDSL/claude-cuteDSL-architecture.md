# CuTeDSL Architecture

This document describes the internal architecture of CuTeDSL (CUTLASS Tensor DSL), following the template outlined in [matklad's ARCHITECTURE.md guide](https://matklad.github.io/2021/02/06/ARCHITECTURE.md.html).

## Table of Contents

1. [Overview](#overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Compilation Pipeline](#compilation-pipeline)
4. [Source Code Organization](#source-code-organization)
5. [Detailed Compilation Stages](#detailed-compilation-stages)
6. [MLIR Integration](#mlir-integration)
7. [Debugging and Introspection](#debugging-and-introspection)
8. [Example Trace: dense_gemm.py](#example-trace-dense_gemmpy)
9. [Cross-Compilation and Architecture Override](#cross-compilation-and-architecture-override)

## Overview

CuTeDSL is a Python-embedded DSL for writing high-performance CUDA kernels using the CuTe (CUTLASS Template Extensions) programming model. It compiles Python functions decorated with `@cute.jit` and `@cute.kernel` into optimized CUDA kernels via MLIR.

**Key Design Principles:**
- Python functions are transformed via AST preprocessing
- MLIR is used as the intermediate representation
- NVVM dialect is the final lowering target before PTX/CUBIN generation
- The DSL supports both host-side JIT functions and device-side kernel functions
- Compilation results are cached for performance

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Python User Code                              │
│  @cute.jit / @cute.kernel decorated functions                   │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              AST Preprocessing (Optional)                        │
│  - Transform for/if/while statements                            │
│  - Inject DSL helper functions                                  │
│  - Scope analysis for yield generation                          │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              Python Function Execution                           │
│  - Extract type information from runtime arguments              │
│  - Build MLIR IR using Python execution trace                   │
│  - Generate func.func ops for host/device                       │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                   MLIR Module                                    │
│  Dialects: func, arith, scf, cute, gpu, nvgpu, nvvm            │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              MLIR Pass Pipeline                                  │
│  cute-to-nvvm{cubin-format=bin arch=sm_90a ...}                │
│  - Dialect lowering passes                                      │
│  - GPU kernel outlining                                         │
│  - TMA lowering                                                 │
│  - NVVM translation                                             │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              CUBIN Generation                                    │
│  gpu.binary op contains embedded CUBIN                          │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│         Runtime: JitExecutor + CUDA Driver                      │
│  - Extract CUBIN from gpu.binary                                │
│  - Load module via CUDA Driver API                              │
│  - Launch kernels with configured parameters                    │
└─────────────────────────────────────────────────────────────────┘
```

## Compilation Pipeline

The complete compilation pipeline from Python AST to executable CUDA kernel:

### Stage 1: Decoration and Registration
**Location:** [cutlass/cute/__init__.py:175-176](cutlass/cute/__init__.py#L175-L176)
```python
jit = _dsl.CuTeDSL.jit
kernel = _dsl.CuTeDSL.kernel
```

When you decorate a function with `@cute.jit` or `@cute.kernel`, it:
1. Registers the function with the DSL singleton
2. Wraps it with preprocessing hooks
3. Returns a wrapper that triggers compilation on first call

**Key Source Files:**
- [cutlass/base_dsl/dsl.py:493-520](cutlass/base_dsl/dsl.py#L493-L520) - `BaseDSL.jit()` classmethod
- [cutlass/base_dsl/dsl.py:522-570](cutlass/base_dsl/dsl.py#L522-L570) - `BaseDSL.kernel()` classmethod

### Stage 2: AST Preprocessing (Optional)
**Location:** [cutlass/base_dsl/ast_preprocessor.py](cutlass/base_dsl/ast_preprocessor.py)

If `preprocess=True` (default), the DSL transforms Python AST before execution:

- **For loops** → `@loop_selector` decorated functions
- **If statements** → `@if_selector` decorated functions
- **While loops** → `@while_selector` decorated functions
- Automatic `yield` generation for read-write variables

**Key Methods:**
- `DSLPreprocessor.visit_For()` - line 430+
- `DSLPreprocessor.visit_If()` - line 530+
- `DSLPreprocessor.visit_While()` - line 600+

### Stage 3: Function Execution and IR Building
**Location:** [cutlass/base_dsl/dsl.py:600-850](cutlass/base_dsl/dsl.py#L600-L850)

When the wrapped function is called:

1. **Type Inference** (line 650-700): Extract MLIR types from Python runtime arguments
2. **IR Context Creation** (line 710-750): Set up MLIR context and module
3. **Function Execution** (line 760-850): Execute Python function body, intercepting DSL operations

**For `@cute.jit` functions:**
- Creates `func.func` with signature matching Python arguments
- Returns `JitExecutor` for host-side execution

**For `@cute.kernel` functions:**
- Creates GPU kernel wrapper with `gpu.launch_func`
- Returns `KernelLauncher` that can be invoked with `.launch()`

**Key Source Lines:**
- [cutlass/base_dsl/dsl.py:600-650](cutlass/base_dsl/dsl.py#L600-L650) - `BaseDSL._func()`
- [cutlass/cutlass_dsl/cutlass.py:318-400](cutlass/cutlass_dsl/cutlass.py#L318-L400) - `_kernel_helper()`

### Stage 4: MLIR Dialect Operations
**Location:** [cutlass/_mlir/dialects/](cutlass/_mlir/dialects/)

During Python function execution, DSL operations generate MLIR ops:

| Python DSL | MLIR Dialect | Operation |
|------------|--------------|-----------|
| `cute.copy()` | cute/nvgpu | `cute.copy`, `nvgpu.tma.load` |
| `cute.gemm()` | nvgpu | `nvgpu.wgmma` |
| `cutlass.range()` | scf | `scf.for` |
| `if cutlass.const_expr()` | (compile-time) | Constant folding |
| `if x > y` (dynamic) | scf | `scf.if` |
| `a + b` | arith | `arith.addi`, `arith.addf` |

**Generated Dialect Files:**
- `_cute_ops_gen.py` - CuTe operations (121K LOC)
- `_cute_nvgpu_ops_gen.py` - CuTe NVGPU operations (128K LOC)
- `_gpu_ops_gen.py` - GPU dialect (148K LOC)
- `_nvvm_ops_gen.py` - NVVM intrinsics (348K LOC)

### Stage 5: MLIR Pass Pipeline Execution
**Location:** [cutlass/base_dsl/compiler.py:135-161](cutlass/base_dsl/compiler.py#L135-L161)

The pipeline string (default: `"builtin.module(cute-to-nvvm{cubin-format=bin})"`) is executed:

```python
pm = self.passmanager.PassManager.parse(pipeline)
pm.enable_verifier(enable_verifier)
pm.run(module.operation)
```

**Pipeline Processing:**
1. **Architecture injection** - [cutlass/base_dsl/dsl.py:973-1000](cutlass/base_dsl/dsl.py#L973-L1000)
   - Injects `toolkitPath`, `arch` into pass options
   - Example: `cute-to-nvvm{arch=sm_90a toolkitPath=/usr/local/cuda-12.3}`

2. **cute-to-nvvm pass** - C++ implementation (not in Python)
   - Lowers CuTe dialect operations
   - Converts TMA operations to NVVM intrinsics
   - Generates NVVM dialect code

3. **NVVM to CUBIN** - NVVM compiler backend
   - Translates to PTX
   - Assembles to CUBIN binary
   - Embeds in `gpu.binary` operation

**Key Source Lines:**
- [cutlass/base_dsl/compiler.py:93-166](cutlass/base_dsl/compiler.py#L93-L166) - `Compiler` class
- [cutlass/cutlass_dsl/cutlass.py:214-224](cutlass/cutlass_dsl/cutlass.py#L214-L224) - Pipeline string construction

### Stage 6: CUBIN Extraction and Module Loading
**Location:** [cutlass/base_dsl/jit_executor.py:259-297](cutlass/base_dsl/jit_executor.py#L259-L297)

After compilation, the runtime extracts CUBIN from MLIR:

```python
def walk_module_and_get_cubin_data(self, module, sym, callback):
    # Walk gpu.binary ops and extract embedded CUBIN
    cubin_data = cubin_data.split(b'bin = "')[1].split(b'">')[0]
    cubin_data = self._get_escaped_cubin_bytes(cubin_data)
    callback(sym, func_sym, cubin_data)
```

**Loading Process:**
1. Parse MLIR to find `gpu.binary` operations
2. Extract CUBIN byte string (escaped format)
3. Load CUBIN via CUDA Driver API: `cuda_helpers.load_cubin_module_data()`
4. Get kernel function pointer: `cuda_helpers.get_kernel_function()`
5. Cache in `JitExecutor.cuda_modules`

**Key Source Lines:**
- [cutlass/base_dsl/jit_executor.py:330-358](cutlass/base_dsl/jit_executor.py#L330-L358) - `walk_module_and_get_cubin_data()`
- [cutlass/base_dsl/jit_executor.py:259-297](cutlass/base_dsl/jit_executor.py#L259-L297) - `update_jit_cuda_modules()`

### Stage 7: Kernel Launch
**Location:** [cutlass/base_dsl/jit_executor.py:228-258](cutlass/base_dsl/jit_executor.py#L228-L258)

When `JitExecutor.__call__()` is invoked:

1. **Argument Preparation** (line 228-231):
   - Convert Python arguments to C pointers
   - Append kernel function pointer to arguments

2. **Packed Arguments** (line 236-242):
   - Create ctypes array of void pointers

3. **Execution** (line 244-257):
   - Call MLIR ExecutionEngine with packed arguments
   - ExecutionEngine dispatches to `gpu.launch_func`
   - CUDA Driver launches kernel

**For device kernels**, the launch configuration is set via `KernelLauncher`:

```python
self.kernel(...).launch(
    grid=(nx, ny, nz),
    block=(tx, ty, tz),
    cluster=(cx, cy, cz),
    stream=cuda_stream
)
```

## Source Code Organization

```
cutlass/python/CuTeDSL/
├── cutlass/
│   ├── cute/                      # User-facing CuTe DSL
│   │   ├── __init__.py            # Main exports (jit, kernel, etc.)
│   │   ├── core.py                # Core CuTe operations
│   │   ├── typing.py              # Type system (Tensor, Layout, etc.)
│   │   ├── arch/                  # Architecture-specific ops
│   │   │   └── nvvm_wrappers.py   # NVVM intrinsic wrappers
│   │   ├── nvgpu/                 # NVGPU operations
│   │   │   ├── cpasync/           # TMA/CP.ASYNC operations
│   │   │   ├── warp/              # Warp-level operations
│   │   │   └── warpgroup/         # Warpgroup operations (WGMMA)
│   │   └── runtime.py             # Runtime utilities
│   │
│   ├── cutlass_dsl/               # DSL implementation
│   │   ├── __init__.py            # Re-exports from base_dsl
│   │   ├── cutlass.py             # CutlassBaseDSL class
│   │   └── cutlass_ast_decorators.py  # AST transformation decorators
│   │
│   ├── base_dsl/                  # Generic DSL framework
│   │   ├── dsl.py                 # BaseDSL class (core logic)
│   │   ├── compiler.py            # MLIR PassManager wrapper
│   │   ├── jit_executor.py        # Runtime executor
│   │   ├── ast_preprocessor.py    # Python AST transformation
│   │   ├── env_manager.py         # Environment variable handling
│   │   ├── typing.py              # DSL type system
│   │   ├── cache_helpers.py       # Compilation cache
│   │   └── runtime/
│   │       ├── cuda.py            # CUDA Driver API wrappers
│   │       └── jit_arg_adapters.py  # Argument type adapters
│   │
│   ├── _mlir/                     # MLIR Python bindings
│   │   ├── ir.py                  # Core IR classes
│   │   ├── passmanager.py         # PassManager wrapper
│   │   ├── execution_engine.py    # ExecutionEngine wrapper
│   │   ├── dialects/              # MLIR dialect bindings
│   │   │   ├── cute.py            # CuTe dialect
│   │   │   ├── nvgpu.py           # NVGPU dialect
│   │   │   ├── gpu.py             # GPU dialect
│   │   │   ├── nvvm.py            # NVVM dialect
│   │   │   ├── arith.py           # Arithmetic ops
│   │   │   ├── scf.py             # Structured control flow
│   │   │   ├── func.py            # Function ops
│   │   │   └── _*_ops_gen.py      # Auto-generated op definitions
│   │   └── _mlir_libs/            # C++ shared library
│   │       └── libCutlassIRPythonCAPI.so  # MLIR C API bindings
│   │
│   ├── pipeline/                  # Pipeline abstractions
│   │   ├── sm90.py                # Hopper pipeline
│   │   └── sm100.py               # Blackwell pipeline
│   │
│   └── utils/                     # Utility functions
│       ├── hopper_helpers.py      # SM90-specific helpers
│       └── blackwell_helpers.py   # SM100-specific helpers
│
└── examples/python/CuTeDSL/
    └── hopper/
        └── dense_gemm.py          # Example: Hopper GEMM kernel
```

## Detailed Compilation Stages

### Python to MLIR Type Mapping

**Location:** [cutlass/base_dsl/typing.py:200-350](cutlass/base_dsl/typing.py#L200-L350)

| Python Type | CuTe Type | MLIR Type |
|-------------|-----------|-----------|
| `int` (static) | `cutlass.Int64` | `i64` |
| `int` (dynamic) | IR value | `index` or `i32` |
| `cute.Tensor` | `cute.Tensor` | `!cute.tensor<...>` |
| `cute.Layout` | `cute.Layout` | `!cute.layout<...>` |
| `cuda.CUstream` | (opaque) | `!llvm.ptr` |
| `cutlass.Float16` | `cutlass.Float16` | `f16` |

**Type Inference Flow:**

1. `get_mlir_types(obj)` - Extract MLIR types from Python objects
2. `get_c_pointers(obj)` - Extract C pointer representations
3. `is_dynamic_expression(obj)` - Check if contains runtime values

**Key Functions:**
- [cutlass/base_dsl/typing.py:264-310](cutlass/base_dsl/typing.py#L264-L310) - `get_mlir_types()`
- [cutlass/base_dsl/typing.py:312-360](cutlass/base_dsl/typing.py#L312-L360) - `get_c_pointers()`
- [cutlass/base_dsl/dsl.py:161-173](cutlass/base_dsl/dsl.py#L161-L173) - `is_dynamic_expression()`

### MLIR Dialect Hierarchy

The CuTe DSL uses multiple MLIR dialects in a layered approach:

```
┌─────────────────────────────────────┐
│     High-Level: cute, nvgpu         │  User-facing abstractions
├─────────────────────────────────────┤
│     Mid-Level: gpu, scf, func       │  Control flow, GPU abstractions
├─────────────────────────────────────┤
│     Low-Level: nvvm, llvm           │  NVIDIA intrinsics, LLVM IR
├─────────────────────────────────────┤
│     Target: PTX/CUBIN               │  Machine code
└─────────────────────────────────────┘
```

**Dialect Responsibilities:**

- **cute** (`_cute_ops_gen.py`): CuTe algebra (Layout, Tensor, Copy, GEMM)
- **nvgpu** (`_cute_nvgpu_ops_gen.py`): GPU-specific CuTe ops (TMA, WGMMA)
- **gpu** (`_gpu_ops_gen.py`): GPU abstractions (launch, module, barriers)
- **nvvm** (`_nvvm_ops_gen.py`): NVIDIA NVVM intrinsics (PTX-level)
- **scf** (`_scf_ops_gen.py`): Structured control flow (for, if, while)
- **func** (`_func_ops_gen.py`): Function definitions and calls
- **arith** (`_arith_ops_gen.py`): Arithmetic operations

### Const vs Dynamic Expressions

CuTeDSL distinguishes between compile-time and runtime expressions:

**Compile-Time (Static):**
```python
tile_m = 128  # Python constant
for i in cutlass.range_constexpr(4):  # Unrolled at compile time
    if cutlass.const_expr(tile_m == 128):  # Compile-time branch
        ...
```

**Runtime (Dynamic):**
```python
for k in cutlass.range(k_tiles):  # scf.for in MLIR
    if x > threshold:  # scf.if in MLIR
        ...
```

**Implementation:**
- [cutlass/base_dsl/ast_helpers.py:350-400](cutlass/base_dsl/ast_helpers.py#L350-L400) - `const_expr()`, `dynamic_expr()`
- Compile-time expressions are evaluated during Python execution
- Runtime expressions generate MLIR SSA values

## MLIR Integration

### MLIR Python Bindings

CuTeDSL includes custom MLIR Python bindings in `cutlass/_mlir/`:

**Core Classes:**
- `ir.Context` - MLIR context (thread-local)
- `ir.Module` - Top-level MLIR module
- `ir.Operation` - Generic MLIR operation
- `ir.Value` - SSA value
- `ir.Type` - MLIR type
- `ir.Attribute` - MLIR attribute

**Relationship to Upstream MLIR:**

CuTeDSL's MLIR bindings are based on LLVM's official Python bindings but extended with custom dialects:

```python
# Standard MLIR (upstream):
from mlir.ir import Context, Module
from mlir.dialects import arith, func, scf

# CuTe DSL (custom):
from cutlass._mlir import ir
from cutlass._mlir.dialects import cute, nvgpu, gpu
```

**Key Differences:**
1. Custom dialects: `cute`, `nvgpu` (not in upstream MLIR)
2. Tight integration with CuTe C++ library
3. Extended with CUTLASS-specific operations

**Location of Bindings:**
- Python wrappers: `cutlass/_mlir/dialects/*.py`
- C++ implementation: `libCutlassIRPythonCAPI.so`

### MLIR Module Structure

A typical compiled CuTeDSL program generates this MLIR structure:

```mlir
module {
  // GPU module container
  gpu.module @kernels {
    // Device kernel
    func.func @kernel_name(...) attributes {gpu.kernel, nvvm.reqntid = ...} {
      // Kernel body with cute/nvgpu ops
      ...
      func.return
    }
  }

  // Host function
  func.func @host_function(...) -> ... {
    // Host logic
    ...
    // Launch kernel
    gpu.launch_func @kernels::@kernel_name
      blocks in (%bx, %by, %bz)
      threads in (%tx, %ty, %tz)
      args(...)
    ...
    func.return
  }

  // After compilation: embedded CUBIN
  gpu.binary @kernels [#gpu.object<...>] {
    // Binary PTX/CUBIN data
  }
}
```

### Pass Pipeline Details

The default pipeline `"builtin.module(cute-to-nvvm{cubin-format=bin})"` expands to multiple sub-passes:

**Pass Sequence (internal to cute-to-nvvm):**

1. **CuTe Lowering Passes:**
   - `cute-lower-copy` - Lower copy operations to memory operations
   - `cute-lower-gemm` - Lower GEMM to WGMMA instructions
   - `cute-tensor-to-memref` - Convert CuTe tensors to MLIR memref

2. **GPU Transformation Passes:**
   - `gpu-kernel-outlining` - Outline kernel regions into gpu.func
   - `convert-gpu-to-nvvm` - Lower GPU dialect to NVVM

3. **NVVM Compilation:**
   - `gpu-to-cubin` - Invoke NVVM compiler to generate CUBIN
   - Embeds result in `gpu.binary` operation

**Pass Options:**
- `arch=sm_90a` - Target architecture (SM 9.0, Hopper)
- `cubin-format=bin` - Output binary format (vs PTX)
- `toolkitPath=/path/to/cuda` - CUDA toolkit location
- `opt-level=3` - LLVM optimization level

**Source Location:**
- Pipeline string construction: [cutlass/cutlass_dsl/cutlass.py:214-224](cutlass/cutlass_dsl/cutlass.py#L214-L224)
- Pipeline preprocessing: [cutlass/base_dsl/dsl.py:973-1000](cutlass/base_dsl/dsl.py#L973-L1000)
- Pipeline execution: [cutlass/base_dsl/compiler.py:135-161](cutlass/base_dsl/compiler.py#L135-L161)

## Debugging and Introspection

### Environment Variables

CuTeDSL provides extensive environment variables for debugging (documented in [cutlass/base_dsl/env_manager.py:247-270](cutlass/base_dsl/env_manager.py#L247-L270)):

**Printing/Debugging:**
```bash
export CUTE_DSL_PRINT_IR=1              # Print generated MLIR
export CUTE_DSL_KEEP_IR=1               # Save MLIR to file
export CUTE_DSL_LOG_TO_CONSOLE=1        # Enable logging
export CUTE_DSL_LOG_LEVEL=10            # DEBUG level logging
export CUTE_DSL_PRINT_AFTER_PREPROCESSOR=1  # Print transformed AST
```

**Compilation Control:**
```bash
export CUTE_DSL_ARCH=sm_90a             # Target architecture
export CUTE_DSL_DRYRUN=1                # Generate IR only, don't compile
export CUTE_DSL_DISABLE_FILE_CACHING=1  # Disable cache
```

**Error Handling:**
```bash
export CUTE_DSL_FILTER_STACKTRACE=1     # Filter internal frames (default)
export CUTE_DSL_WARNINGS_AS_ERRORS=1    # Treat warnings as errors
```

**CUDA Configuration:**
```bash
export CUDA_TOOLKIT_PATH=/usr/local/cuda-12.3  # CUDA toolkit location
export CUTE_DSL_LIBS=/path/to/libs      # MLIR runtime libraries
```

### Introspection APIs

**1. Examining MLIR IR:**

```python
import cutlass.cute as cute
from cutlass._mlir import ir

@cute.jit
def my_function(x: cute.Tensor):
    ...

# Compile and get executor
executor = cute.compile(my_function, tensor_arg)

# Access MLIR module
mlir_module = executor.ir_module
print(mlir_module)

# Walk operations
def walk_callback(op):
    print(f"Op: {op.name}, Location: {op.location}")
    return ir.WalkResult.ADVANCE

mlir_module.operation.walk(walk_callback)
```

**2. Cache Inspection:**

```python
from cutlass.base_dsl.cache_helpers import load_cache_from_path

# Load cache
cache = load_cache_from_path("CuTeDSL", capacity=1000)

# Inspect cached compilations
for key, (ir_module, executor) in cache.items():
    print(f"Cache key: {key}")
    print(f"Module: {ir_module}")
```

**3. Type Information:**

```python
from cutlass.base_dsl.typing import get_mlir_types

# Get MLIR types from Python objects
types = get_mlir_types(my_tensor)
print(f"MLIR types: {types}")
```

**4. Constexpr Arguments:**

```python
# After compilation
executor = cute.compile(my_function, ...)

# Get constexpr arguments (pruned from signature)
constexpr_args = executor.get_constexpr_args()
# Returns: [{"argument_index": 0, "argument_name": "tile_size"}, ...]
```

### Logging Infrastructure

**Logger Setup:** [cutlass/base_dsl/utils/logger.py](cutlass/base_dsl/utils/logger.py)

```python
from cutlass.base_dsl.utils.logger import log

# Log levels:
log().debug("Detailed debug info")
log().info("General information")
log().warning("Warning message")
log().error("Error occurred")
```

**Profiling:**

```bash
export CUTE_DSL_JIT_TIME_PROFILING=1
```

This enables timing for:
- AST preprocessing time
- MLIR IR generation time
- Compilation time
- Kernel execution time

**Source:** [cutlass/base_dsl/utils/timer.py](cutlass/base_dsl/utils/timer.py)

## Example Trace: dense_gemm.py

Let's trace the compilation of [examples/python/CuTeDSL/hopper/dense_gemm.py](../../examples/python/CuTeDSL/hopper/dense_gemm.py) step by step:

### Step 1: Class Instantiation (Host Side)

**File:** [dense_gemm.py:1503](../../examples/python/CuTeDSL/hopper/dense_gemm.py#L1503)
```python
gemm = HopperWgmmaGemmKernel(acc_dtype, tile_shape_mn, cluster_shape_mn)
```

- Creates kernel configuration object
- Stores tile shape, cluster shape, accumulator dtype
- No compilation happens yet

### Step 2: JIT Decoration

**File:** [dense_gemm.py:372-379](../../examples/python/CuTeDSL/hopper/dense_gemm.py#L372-L379)
```python
@cute.jit
def __call__(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, stream: cuda.CUstream):
    ...
```

**What happens:**
1. `cute.jit` → `cutlass.cutlass_dsl.CuTeDSL.jit()` (line 175 in [cute/__init__.py](cutlass/cute/__init__.py#L175))
2. Calls `BaseDSL.jit()` → [cutlass/base_dsl/dsl.py:493-520](cutlass/base_dsl/dsl.py#L493-L520)
3. Returns `jit_runner_decorator` that wraps `__call__`
4. Wrapped function stores reference to DSL singleton

**Key transformations:**
- Function gets `._dsl_object` attribute pointing to `CuTeDSL` singleton
- If `preprocess=True`, marks for AST transformation with `._transformed_ast = None`
- Wrapped in `jit_wrapper` that triggers compilation on first call

### Step 3: Kernel Decoration

**File:** [dense_gemm.py:482-496](../../examples/python/CuTeDSL/hopper/dense_gemm.py#L482-L496)
```python
@cute.kernel
def kernel(self, tma_atom_a, mA_mkl, tma_atom_b, mB_nkl, ...):
    ...
```

**What happens:**
1. `cute.kernel` → `cutlass.cutlass_dsl.CuTeDSL.kernel()` (line 176 in [cute/__init__.py](cutlass/cute/__init__.py#L176))
2. Calls `BaseDSL.kernel()` → [cutlass/base_dsl/dsl.py:522-570](cutlass/base_dsl/dsl.py#L522-L570)
3. Returns `device_jit_decorator` that wraps `kernel` method
4. Wrapped function creates `KernelLauncher` object

**Key transformations:**
- Device kernel marked with special attributes
- Will be outlined to GPU module during compilation
- Returns launcher that can be invoked with `.launch(grid=..., block=..., cluster=...)`

### Step 4-13: Detailed Compilation Steps

(Due to length constraints, the full trace with 13 steps spanning AST preprocessing, IR generation, MLIR pass execution, CUBIN extraction, and kernel launch is omitted here. The key insight is that each Python operation in the kernel body generates corresponding MLIR operations, which are then lowered through multiple dialects until reaching CUBIN.)

**Key Operations:**
- [dense_gemm.py:752-760](../../examples/python/CuTeDSL/hopper/dense_gemm.py#L752-L760) - TMA load generates `nvgpu.tma.async.load`
- [dense_gemm.py:811-817](../../examples/python/CuTeDSL/hopper/dense_gemm.py#L811-L817) - GEMM generates `nvgpu.wgmma.mma_async`
- [dense_gemm.py:931](../../examples/python/CuTeDSL/hopper/dense_gemm.py#L931) - Barriers generate `nvvm.barrier0`

## Cross-Compilation and Architecture Override

### Current Architecture Detection

**Location:** [cutlass/base_dsl/env_manager.py:66-89](cutlass/base_dsl/env_manager.py#L66-L89)

```python
def detect_gpu_arch(prefix):
    arch = (None, None)
    try:
        arch = get_compute_capability_major_minor()  # Query GPU via CUDA
    except Exception as e:
        log().info(f"Failed to get CUDA compute capability: {e}")

    if arch == (None, None):
        arch = (10, 0)  # Default to sm_100 (Blackwell)

    major, minor = arch
    suffix = ""
    if major >= 9:
        suffix = "a"  # Hopper/Blackwell use 'a' suffix

    return f"sm_{major}{minor}{suffix}"
```

**Environment variable override:**
```bash
export CUTE_DSL_ARCH=sm_90a  # Override detected architecture
```

### Cross-Compiling for Different Architecture

**Problem:** You want to compile for Blackwell (sm_100) on a Hopper (sm_90a) machine.

**Solution:** Set architecture via environment variable before importing CuTeDSL:

```python
import os
os.environ["CUTE_DSL_ARCH"] = "sm_100"  # Must be before import!

import cutlass.cute as cute

@cute.kernel
def my_blackwell_kernel(...):
    # Use Blackwell-specific features
    ...
```

**Important:** Set the environment variable **before** the first import of `cutlass.cute`, as the DSL singleton is created at import time and caches the architecture.

### Modifying Architecture at Runtime

**Not recommended, but possible:**

```python
import cutlass.cute as cute

# Access DSL singleton
dsl = cute._dsl.CuTeDSL._get_dsl()

# Override architecture
dsl.envar.arch = "sm_100"

# Now subsequent compilations will target sm_100
@cute.jit
def my_func(...):
    ...
```

**Caveat:** This does NOT clear the compilation cache, so previously compiled functions will still use the old architecture.

### Cross-Compilation Example

**Scenario:** Compile Blackwell WGMMA kernel on Hopper machine

```bash
#!/bin/bash

# Cross-compile for Blackwell
export CUTE_DSL_ARCH=sm_100
export CUDA_TOOLKIT_PATH=/usr/local/cuda-12.5  # Blackwell requires CUDA 12.5+

# Generate CUBIN only (don't try to run)
export CUTE_DSL_KEEP_IR=1  # Save MLIR
export CUTE_DSL_DRYRUN=1   # Don't execute

python my_blackwell_kernel.py
```

**Result:**
- MLIR saved with `arch=sm_100` in pass options
- CUBIN generated for sm_100 architecture
- Can be deployed to Blackwell GPU later

### Architecture-Specific Code Paths

CuTeDSL provides architecture-specific helpers:

**Hopper (sm_90a):**
- [cutlass/utils/hopper_helpers.py](cutlass/utils/hopper_helpers.py)
- [cutlass/pipeline/sm90.py](cutlass/pipeline/sm90.py)
- WGMMA instructions, TMA multicast, cluster-level operations

**Blackwell (sm_100):**
- [cutlass/utils/blackwell_helpers.py](cutlass/utils/blackwell_helpers.py)
- [cutlass/pipeline/sm100.py](cutlass/pipeline/sm100.py)
- Enhanced WGMMA, expanded TMA, new atomic operations

**Usage:**
```python
from cutlass.utils import hopper_helpers as sm90
from cutlass.utils import blackwell_helpers as sm100

# Conditionally select helpers based on target arch
if target_arch == "sm_90a":
    helpers = sm90
else:
    helpers = sm100

tiled_mma = helpers.make_trivial_tiled_mma(...)
```

### Verifying Target Architecture

**Inspect compiled MLIR:**

```bash
export CUTE_DSL_PRINT_IR=1
export CUTE_DSL_KEEP_IR=1

python my_kernel.py
```

Look for architecture in pass options:
```mlir
// In saved .mlir file:
module attributes {
  gpu.container_module,
  nvvm.target = #nvvm.target<chip = "sm_100", ...>
}
```

**Inspect CUBIN metadata:**

```bash
cuobjdump --dump-elf my_kernel.cubin
```

Look for:
```
.nv.info.sm_100
```

### Known Issues and Limitations

1. **CUDA Toolkit Version:** Newer architectures require newer CUDA toolkits:
   - sm_90a (Hopper): CUDA 12.0+
   - sm_100 (Blackwell): CUDA 12.5+

2. **Instruction Availability:** Cross-compiling for newer architecture may succeed, but actual instructions must be supported:
   ```python
   # This will COMPILE on Hopper, but may fail at runtime on sm_90a:
   cute.nvgpu.blackwell_specific_operation(...)
   ```

3. **Cache Invalidation:** Changing architecture requires clearing cache:
   ```bash
   rm -rf ~/.cache/CuTeDSL/*
   ```

4. **Pipeline Compatibility:** The `cute-to-nvvm` pass must support the target architecture. Check MLIR/NVVM support.

---

## Summary of Key Source Files

**Compilation Pipeline:**
- [cutlass/base_dsl/dsl.py](cutlass/base_dsl/dsl.py) - Core DSL logic, function tracing
- [cutlass/base_dsl/compiler.py](cutlass/base_dsl/compiler.py) - MLIR pass manager wrapper
- [cutlass/base_dsl/jit_executor.py](cutlass/base_dsl/jit_executor.py) - Runtime execution, CUBIN loading
- [cutlass/cutlass_dsl/cutlass.py](cutlass/cutlass_dsl/cutlass.py) - CuTe-specific DSL implementation

**AST Transformation:**
- [cutlass/base_dsl/ast_preprocessor.py](cutlass/base_dsl/ast_preprocessor.py) - Python AST rewriting

**Type System:**
- [cutlass/base_dsl/typing.py](cutlass/base_dsl/typing.py) - DSL type system
- [cutlass/cute/typing.py](cutlass/cute/typing.py) - CuTe-specific types

**MLIR Integration:**
- [cutlass/_mlir/ir.py](cutlass/_mlir/ir.py) - MLIR IR Python bindings
- [cutlass/_mlir/dialects/cute.py](cutlass/_mlir/dialects/cute.py) - CuTe dialect
- [cutlass/_mlir/dialects/nvgpu.py](cutlass/_mlir/dialects/nvgpu.py) - NVGPU dialect

**User-Facing API:**
- [cutlass/cute/__init__.py](cutlass/cute/__init__.py) - Main exports
- [cutlass/cute/core.py](cutlass/cute/core.py) - Core CuTe operations

**Environment and Caching:**
- [cutlass/base_dsl/env_manager.py](cutlass/base_dsl/env_manager.py) - Environment variables
- [cutlass/base_dsl/cache_helpers.py](cutlass/base_dsl/cache_helpers.py) - Compilation cache

**Runtime:**
- [cutlass/base_dsl/runtime/cuda.py](cutlass/base_dsl/runtime/cuda.py) - CUDA Driver API wrappers

This architecture document provides a comprehensive overview of CuTeDSL's internal structure and compilation pipeline. For specific implementation details, refer to the source files mentioned throughout this document.
