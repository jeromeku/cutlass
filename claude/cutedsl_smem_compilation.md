# How CuTe DSL Compiles Shared Memory Declarations

## The Question

How can this Python code produce a GPU kernel with shared memory, without explicit CUDA driver calls at compile time?

```python
@cute.struct
class SharedStorage:
    staging_buffer: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 1], 1024
    ]

smem = cutlass.utils.SmemAllocator()
storage = smem.allocate(SharedStorage, 64)
kernel.launch(grid=(1,1,1), block=(64,1,1), smem=SharedStorage.size_in_bytes())
```

**Short answer:** CuTe DSL never "allocates" shared memory. It computes a byte size at Python trace time, emits MLIR IR with an address-space-3 global symbol, and lets the standard MLIR→NVVM→PTX pipeline turn that into a `.shared` declaration. The actual hardware allocation happens at kernel launch via `cuLaunchKernel(..., smem_size, ...)`.

---

## The Full Pipeline, Stage by Stage

### Stage 0: Python — struct layout and size computation

**No code generation happens here.** `@cute.struct` is pure Python arithmetic.

**File:** `cute/core.py:4107-4475`

```python
@cute.struct
class SharedStorage:
    staging_buffer: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 1], 1024
    ]
```

The decorator walks `__annotations__` and computes a C-like struct layout:

1. **`MemRange[Float32, 1]`** → 1 element × 4 bytes = **4 bytes** (`core.py:4203`)
2. **`Align[..., 1024]`** → round the field offset up to a 1024-byte boundary (`core.py:4467-4474`):
   ```python
   def align_offset(offset, align):
       return (offset + (align - 1)) & ~(align - 1)
   ```
3. **`SharedStorage.size_in_bytes()`** returns the final padded size (e.g., 1024 bytes).

At this point we have a plain Python integer. No MLIR, no LLVM, no PTX.

---

### Stage 1: Python trace → MLIR IR emission

When the `@cute.kernel` function body executes under the DSL tracer, Python statements become MLIR operations:

#### 1a. `SmemAllocator()` — get the shared memory base pointer

**File:** `utils/smem_allocator.py:95-109`

```python
def __init__(self):
    self._base = get_dyn_smem(Int8, alignment=1024)   # ← emits MLIR op
    self._allocated_bytes = 0
```

**File:** `cute/arch/smem.py:64-96`

```python
ptr_ty = PtrType.get(
    element_type.mlir_type,
    AddressSpace.smem,      # ← address space 3 (shared memory)
    alignment,
)
return cute_nvgpu_ir.arch_get_dyn_smem(ptr=ptr_ty)
```

This emits a single MLIR operation:
```mlir
%base = cute_nvgpu.arch.get_dyn_smem : !cute.ptr<i8, smem, align=1024>
```

This op says: "give me a pointer to the start of dynamic shared memory." It does **not** allocate anything — it just names a symbol that the backend will resolve.

#### 1b. `smem.allocate(SharedStorage, 64)` — bump-pointer offset arithmetic

**File:** `smem_allocator.py:126-194`

The allocator does pointer arithmetic in Python (tracked by `_allocated_bytes`) and emits MLIR pointer-offset ops for each field. For our struct:

```
base + 0  →  staging_buffer (aligned to 1024)
_allocated_bytes += 1024
```

The struct constructor `SharedStorage(base_ptr)` calls into each field, generating typed pointer casts from the base. This is pure address arithmetic — no memory allocation op.

#### 1c. `kernel.launch(..., smem=SharedStorage.size_in_bytes())` — embed size as a constant

**File:** `cutlass_dsl/cutlass.py:604-720`

The smem size (a Python int) becomes an MLIR constant operand on the launch op:

```mlir
%smem_size = llvm.mlir.constant(1024 : i64)
cuda.launch_ex %kernel, %cfg  // cfg includes dynamicSharedMemorySize = %smem_size
```

This is how the host tells the GPU "reserve N bytes of shared memory for this kernel launch."

---

### Stage 2: MLIR Lowering — CuTe ops → LLVM dialect

The `cute-to-nvvm` pass pipeline runs ~40 transformation passes. The critical ones for shared memory:

#### Pass: `gpu-kernel-outlining`

The kernel body is extracted into a `gpu.module`. The `get_dyn_smem` op becomes an LLVM global:

**From actual IR dump** (`21_gpu-kernel-outlining.mlir:11`):
```mlir
gpu.module @kernels {
    llvm.mlir.global external @__dynamic_shmem__0()
        {addr_space = 3 : i32, alignment = 1024 : i64, dso_local}
        : !llvm.array<0 x i8>
    // ^^^^^^^^^^^^^^^^^^^^
    //  - addr_space = 3    → CUDA shared memory
    //  - array<0 x i8>     → zero-sized (dynamic; real size comes at launch)
    //  - alignment = 1024  → from SmemAllocator
    //  - external           → no initializer (runtime-provided)

    llvm.func @kernel_...(...)
        attributes {gpu.kernel, nvvm.kernel, nvvm.reqntid = array<i32: 64, 1, 1>} {
        ...
    }
}
```

**This is the key insight.** The "shared memory declaration" is just an LLVM global variable in **address space 3** with **zero static size**. There is no allocation call. The NVPTX backend knows that address space 3 means `.shared` memory.

#### Pass: `convert-gpu-to-nvvm`

References to the global become `addressof` operations:

**From actual IR dump** (`31_2_convert-gpu-to-nvvm.mlir:14`):
```mlir
%1 = llvm.mlir.addressof @__dynamic_shmem__0 : !llvm.ptr<3>
//                                                ^^^^^^^^
//                                         pointer to addr space 3
```

All subsequent loads/stores through `%1` (with offset arithmetic) naturally target shared memory.

---

### Stage 3: NVVM IR → PTX

The MLIR LLVM dialect is essentially LLVM IR. The NVPTX backend (built into `_cutlass_ir.so`) translates it to PTX. The lowering rules are:

| LLVM IR (MLIR) | PTX |
|----------------|-----|
| `@__dynamic_shmem__0` with `addr_space = 3` | `.extern .shared .align 1024 .b8 __dynamic_shmem__0[]` |
| `llvm.load` from `!llvm.ptr<3>` | `ld.shared.f32 %f1, [__dynamic_shmem__0 + offset]` |
| `llvm.store` to `!llvm.ptr<3>` | `st.shared.u32 [__dynamic_shmem__0 + offset], %r10` |

**From the actual generated PTX** (`async_pp_dump/...sm_90a.ptx`):

```ptx
.extern .shared .align 1024 .b8 __dynamic_shmem__0[];   // declaration

.visible .entry kernel_cutlass_synced_producer_consumer_...(
    .param .align 8 .b8 ..._param_0[8]
)
.reqntid 64, 1, 1
{
    ...
    st.shared.u32 [__dynamic_shmem__0], %r10;     // write to smem
    bar.sync 0;                                     // __syncthreads()
    ld.shared.f32 %f1, [__dynamic_shmem__0];      // read from smem
    ...
}
```

Note: `.extern .shared ... []` with empty brackets means "dynamic shared memory — size determined at launch time."

---

### Stage 4: Runtime — Two-layer execution via MLIR ExecutionEngine + `cudaLaunchKernelEx`

The kernel is **not** launched by a direct Python `cuLaunchKernel` call. Instead, two layers cooperate:

#### Layer 1: MLIR ExecutionEngine (LLVM JIT)

The `ExecutionEngine` JIT-compiles the **host-side** LLVM IR into native x86 code. When Python calls `self.capi_func(packed_args)` (`jit_executor.py:787`), it enters a JITted native function — not an interpreter.

#### Layer 2: CUDA Runtime API calls embedded in the JITted code

The `cuda-to-binary` MLIR pass (pass 40 in the pipeline) does two things:
1. Compiles `gpu.module @kernels` into a CUBIN binary blob and embeds it as a global constant `@kernels_binary`
2. Generates host-side functions that call the CUDA **runtime** API (not the driver API)

**From the final IR** (`42_reconcile-unrealized-casts.mlir`):

**Initialization — `cuda_init` loads the embedded CUBIN:**
```mlir
llvm.func @cuda_init(%arg0: !llvm.ptr) -> i32 {
    %1 = llvm.mlir.addressof @kernels_binary : !llvm.ptr       // embedded CUBIN blob
    %4 = llvm.call @_cudaLibraryLoadData(%3, %1, ...) -> i32   // CUDA runtime: load library
}
```

**Resolution — `cuda_load` resolves the kernel and sets attributes:**
```mlir
llvm.func @cuda_load(%arg0: !llvm.ptr) -> i32 {
    llvm.call @_cudaLibraryGetKernel(...)       // look up kernel by name from library
    llvm.call @_cudaFuncSetAttribute(...)       // set max dynamic smem size, etc.
}
```

**Launch — the host function calls `cudaLaunchKernelEx`:**
```mlir
llvm.func @cutlass___call_...(args...) -> i32 {
    // Build launch config struct:
    //   gridDim = (4096, 1, 1), blockDim = (128, 1, 1)
    //   dynamicSmemBytes = <smem_size>, stream = nullptr

    // Pack kernel arguments into void** array

    // Load the kernel handle and launch:
    %85 = llvm.mlir.addressof @kernels_kernel_... : !llvm.ptr
    %86 = llvm.load %85 : !llvm.ptr -> !llvm.ptr
    %87 = llvm.call @_cudaLaunchKernelEx(%66, %86, %69) -> i32   // ← THE ACTUAL LAUNCH
}
```

`cudaLaunchKernelEx` is a CUDA 12.x runtime API that takes a config struct (grid, block, smem size, stream, launch attributes) instead of positional arguments.

**Python-side initialization** (`cutlass_dsl/cuda_jit_executor.py:199-237`) calls the JITted `cuda_init` and `cuda_load_to_device` before the first launch:

```python
# _load_cuda_library():
cuda_init, cuda_load_to_device = self._get_cuda_init_and_load()  # look up from JIT engine
cuda_init(packed_args)           # → JITted code calls _cudaLibraryLoadData(@kernels_binary)
cuda_load_to_device(packed_args) # → JITted code calls _cudaLibraryGetKernel + _cudaFuncSetAttribute
```

**The full call chain:**
```
Python                          JIT'd Native Code                    CUDA Runtime
──────                          ─────────────────                    ────────────

capi_func(packed_args)  ──→  @_mlir_ciface_cutlass___call_...
                              │
                              └─→ @cutlass___call_...
                                   │
                                   ├─ build launch config struct
                                   ├─ set launch attributes
                                   ├─ pack kernel args into void**
                                   └─ _cudaLaunchKernelEx(           ──→  CUDA runtime
                                        config,                           provisions smem,
                                        kernel_ptr,                       dispatches to
                                        args)                             GPU hardware
```

The CUDA runtime then:
1. Reads the `.extern .shared` declaration from the CUBIN (already loaded via `cudaLibraryLoadData`)
2. Allocates `smem_size` bytes of shared memory on the SM
3. Sets the `__dynamic_shmem__0` symbol to point to the start of that allocation
4. Launches the kernel

---

## Summary Diagram

```
Python                          MLIR                              PTX / Hardware
──────                          ────                              ──────────────

@cute.struct                   (nothing)                          (nothing)
  └─ computes size=1024
     alignment=1024

SmemAllocator()                cute_nvgpu.arch.get_dyn_smem      (nothing yet)
  └─ "give me smem base"       └─ ptr<i8, smem, align=1024>

smem.allocate(SharedStorage)   ptr arithmetic ops                 (nothing yet)
  └─ bump _allocated_bytes      └─ base + offset → field ptrs

staging_smem.fill(0)           llvm.store %0, %ptr<3>            st.shared.u32 [__dyn..], 0

                               ──── gpu-kernel-outlining ────
                               llvm.mlir.global external
                                 @__dynamic_shmem__0
                                 {addr_space=3, align=1024}      .extern .shared .align 1024
                                 : !llvm.array<0 x i8>             .b8 __dynamic_shmem__0[]

kernel.launch(smem=1024)       cuda.launch_ex with                cudaLaunchKernelEx(
  └─ Python int → MLIR const     dynamicSharedMemorySize=1024       config, kernel_ptr, args)
```

## Why No Driver Calls at Compile Time?

The compilation pipeline never needs the CUDA driver because:

1. **Shared memory is not "allocated" in the traditional sense.** It's declared as a zero-sized external symbol in a specific address space. The NVPTX backend knows that address space 3 = `.shared` memory, just as address space 1 = `.global` memory. This is a convention baked into the LLVM NVPTX target, not a runtime operation.

2. **The size is a launch parameter, not a compile-time allocation.** The `.extern .shared .b8 name[]` syntax in PTX means "this kernel uses dynamic shared memory whose size will be provided at launch." The compiler doesn't need to know the size — it just emits address arithmetic relative to the symbol.

3. **MLIR→PTX is purely a compiler pipeline.** The chain is:
   ```
   MLIR (CuTe dialect) → MLIR (LLVM dialect) → LLVM IR → PTX (text) → CUBIN (via ptxas)
   ```
   Every step is a syntactic transformation — rewriting IR from one form to another. No GPU hardware is touched until `cudaLaunchKernelEx` at runtime.

4. **The CUDA runtime's role is deferred to runtime.** The `cuda-to-binary` pass embeds the compiled CUBIN as a constant blob in the host LLVM IR and generates functions that call `cudaLibraryLoadData` (to load the CUBIN) and `cudaLaunchKernelEx` (to dispatch the kernel). These calls are JIT-compiled by the MLIR `ExecutionEngine` into native code, then invoked from Python via ctypes. The compile-time pipeline only needs to produce a valid CUBIN binary — the CUDA runtime handles hardware provisioning (shared memory, registers, warps) at launch time.

## Key Files

| Component | File | Lines |
|-----------|------|-------|
| `@cute.struct` layout computation | `cute/core.py` | 4107-4475 |
| `SmemAllocator` + bump allocator | `utils/smem_allocator.py` | 31-301 |
| `get_dyn_smem` MLIR op emission | `cute/arch/smem.py` | 64-96 |
| Kernel launch (smem size operand) | `cutlass_dsl/cutlass.py` | 604-720 |
| `cuda_init` / `cuda_load` JIT lookup | `cutlass_dsl/cuda_jit_executor.py` | 157-237 |
| JIT execution (`capi_func` call) | `base_dsl/jit_executor.py` | 787-803 |
| `cute-to-nvvm` pipeline definition | `cutlass_dsl/cutlass.py` | 254-261 |
| Final IR with `cudaLaunchKernelEx` | `42_reconcile-unrealized-casts.mlir` | 7-296 |

## Process

**Phase 1 — Initial trace (compile-time pipeline):**
Dispatched two parallel research agents:
1. **Python agent** — traced `@cute.struct`, `SmemAllocator`, `get_dyn_smem`, kernel launch ops across the Python DSL codebase
2. **MLIR/C++ agent** — found the MLIR pass pipeline stages, located actual IR dumps from prior compilation runs, traced `gpu-kernel-outlining` and `convert-gpu-to-nvvm` passes

Then read the actual IR dump files (`21_gpu-kernel-outlining.mlir`, `31_2_convert-gpu-to-nvvm.mlir`) and generated PTX (`async_pp_dump/*.ptx`) to show concrete examples at each stage.

**Phase 2 — Runtime execution correction:**
The original document incorrectly claimed `cuLaunchKernel` (CUDA driver API) was called directly from Python. Investigation of the runtime execution path revealed:
1. Read `cuda_jit_executor.py` — found that `cuda_init` and `cuda_load_to_device` symbols are looked up from the JIT engine, not called via Python ctypes to the driver API
2. Grepped the MLIR IR dumps for `cudaLaunchKernel` and found the actual call in `42_reconcile-unrealized-casts.mlir` — the final lowered IR shows `llvm.call @_cudaLaunchKernelEx(...)` embedded in the JIT-compiled host function
3. Read the full host launch function (lines 150-296) to confirm the two-layer model: MLIR ExecutionEngine JITs native x86 host code, and that host code calls `cudaLaunchKernelEx` (CUDA runtime API, not driver API)
4. Traced the initialization path: `cuda_init` → `_cudaLibraryLoadData(@kernels_binary)` loads the CUBIN blob that the `cuda-to-binary` pass embedded as a global constant
