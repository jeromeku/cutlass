# CuTeDSL Shared Storage Trace

## Question

How can CuTeDSL compile code like this:

```python
@cute.struct
class SharedStorage:
    staging_buffer: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 1], 1024
    ]
```

without making an explicit CUDA-driver allocation call inside the kernel?

## Short Answer

In this example, CuTeDSL is **not** compiling a C++-style fixed `__shared__` variable declaration. It is compiling:

1. a **type/layout description** (`@cute.struct`),
2. a **typed view** over a byte pointer in shared memory,
3. plus a **kernel launch** that tells CUDA how many bytes of **dynamic shared memory** to reserve.

So the allocation does not happen as a driver API call from inside the kernel. The host launch sets the dynamic shared-memory size, and the kernel code receives/accesses that region through shared-memory address-space operations.

## Code Map

- [async_pipeline.py](../experiments/async_pipeline.py#L72)
- [async_pipeline.py](../experiments/async_pipeline.py#L95)
- [async_pipeline.py](../experiments/async_pipeline.py#L232)
- [core.py](../python/CuTeDSL/cutlass/cute/core.py#L4213)
- [core.py](../python/CuTeDSL/cutlass/cute/core.py#L4348)
- [core.py](../python/CuTeDSL/cutlass/cute/core.py#L4413)
- [smem_allocator.py](../python/CuTeDSL/cutlass/utils/smem_allocator.py#L95)
- [smem_allocator.py](../python/CuTeDSL/cutlass/utils/smem_allocator.py#L126)
- [smem.py](../python/CuTeDSL/cutlass/cute/arch/smem.py#L23)
- [smem.py](../python/CuTeDSL/cutlass/cute/arch/smem.py#L64)
- [cutlass.py](../python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py#L858)
- [cuda.py](../python/CuTeDSL/cutlass/base_dsl/runtime/cuda.py#L565)
- [00_input.mlir](../experiments/rms_norm_mlir_dump/00_input.mlir#L300)
- [cutlass_run_synced_producer_consumer_Tensorgmemo81.sm_90a.ptx](../experiments/async_pp_dump/cutlass_run_synced_producer_consumer_Tensorgmemo81.sm_90a.ptx#L14)

## Key Functions Index

| Function | File | Purpose |
|----------|------|---------|
| `struct.__init__` | [core.py](../python/CuTeDSL/cutlass/cute/core.py#L4348) | Compute field offsets, struct alignment, and total byte size |
| `struct.__call__` | [core.py](../python/CuTeDSL/cutlass/cute/core.py#L4413) | Reinterpret a base byte pointer as a typed struct view |
| `_MemRangeData.get_tensor` | [core.py](../python/CuTeDSL/cutlass/cute/core.py#L4247) | Turn a memory-range field into a tensor/view |
| `SmemAllocator.__init__` | [smem_allocator.py](../python/CuTeDSL/cutlass/utils/smem_allocator.py#L95) | Grab the dynamic shared-memory base pointer |
| `SmemAllocator.allocate` | [smem_allocator.py](../python/CuTeDSL/cutlass/utils/smem_allocator.py#L126) | Bump-allocate inside dynamic shared memory and build a typed view |
| `get_dyn_smem` | [smem.py](../python/CuTeDSL/cutlass/cute/arch/smem.py#L64) | Emit the DSL op that retrieves the dynamic shared-memory pointer |
| `alloc_smem` | [smem.py](../python/CuTeDSL/cutlass/cute/arch/smem.py#L23) | Emit the DSL op for static shared-memory allocation |
| `launch_kernel` | [cuda.py](../python/CuTeDSL/cutlass/base_dsl/runtime/cuda.py#L565) | Call `cuLaunchKernel(..., smem_size, ...)` |

## Frame-By-Frame Trace

### Frame 1: the example defines a shared-memory layout

In the simple example, the kernel does this:

```python
@cute.kernel
def synced_producer_consumer(SharedStorage: cutlass.Constexpr, res: cute.Tensor):
    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage, 64)
    staging_smem = storage.staging_buffer.get_tensor(cute.make_layout(1))
```

and the launcher does this:

```python
@cute.jit
def run_synced_producer_consumer(res: cute.Tensor):
    @cute.struct
    class SharedStorage:
        staging_buffer: cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, 1], 1024
        ]

    synced_producer_consumer(SharedStorage, res).launch(
        grid=(1, 1, 1), block=(64, 1, 1), smem=SharedStorage.size_in_bytes()
    )
```

Source: [async_pipeline.py](../experiments/async_pipeline.py#L72), [async_pipeline.py](../experiments/async_pipeline.py#L95)

State before this frame:
- You have only a Python class annotation.

State after this frame:
- CuTeDSL knows the struct's byte layout and asks the launch to reserve that many bytes of dynamic shared memory.

The most important line is:

```python
smem=SharedStorage.size_in_bytes()
```

That is the tell that this path uses **dynamic shared memory**.

### Frame 2: `@cute.struct` computes a byte layout, not an allocation

The decorator implementation walks the annotations, computes offsets, tracks the strictest alignment, and rounds the final size up to that alignment:

```python
def __init__(self, cls):
    self._annotations = getattr(cls, "__annotations__", {})
    self._offsets = {}
    offset = 0
    alignment = 1
    for name, object in self._annotations.items():
        ...
        elif isinstance(object, struct._MemRangeMeta):
            sub_align = max(object.elem_width // 8, sub_align)
            offset = self.align_offset(offset, sub_align)
            self._offsets[name] = offset
            offset = add_offset(object.size_in_bytes)
        ...
        alignment = max(alignment, sub_align)
    self._align_of = alignment
    self._size_of = self.align_offset(offset, alignment)
```

Source: [core.py](../python/CuTeDSL/cutlass/cute/core.py#L4348)

For your field:

```python
staging_buffer: Align[MemRange[Float32, 1], 1024]
```

that means:
- payload size = `1 * sizeof(float32) = 4` bytes
- required field alignment = `max(4, 1024) = 1024`
- field offset = round current offset up to 1024
- total struct alignment = at least 1024
- total struct size = round total bytes up to a multiple of 1024

So `SharedStorage.size_in_bytes()` becomes 1024, not 4.

This is just layout computation. No memory has been reserved yet.

### Frame 3: `struct.__call__` turns a base pointer into a typed overlay

Once some other part of the system provides a byte pointer, `struct.__call__` builds the Python object by reinterpreting subranges of that pointer:

```python
def __call__(self, base):
    cls = self._cls()
    setattr(cls, "_base", base)
    for name, off in self._offsets.items():
        obj = self._annotations[name]
        if isinstance(obj, struct._AlignMeta):
            obj = obj.dtype
        if isinstance(obj, struct._MemRangeMeta):
            new_obj = struct._MemRangeData(obj._dtype, obj._size, base + off)
            setattr(cls, name, new_obj)
```

Source: [core.py](../python/CuTeDSL/cutlass/cute/core.py#L4413)

State before this frame:
- A raw shared-memory byte pointer exists.

State after this frame:
- `storage.staging_buffer` becomes a typed range rooted at `base + offset`.

Again, this is pointer reinterpretation, not allocation.

### Frame 4: `SmemAllocator` gets the dynamic shared-memory base pointer

The allocator constructor does this:

```python
def __init__(self, *, loc=None, ip=None):
    self._base = get_dyn_smem(Int8, alignment=1024, loc=loc, ip=ip)
    self._allocated_bytes = 0
```

Source: [smem_allocator.py](../python/CuTeDSL/cutlass/utils/smem_allocator.py#L95)

So the allocator is not asking CUDA to allocate memory. It is asking the DSL/IR to give it the **base pointer of the dynamic shared-memory region already associated with this kernel launch**.

Then `allocate()` does simple bump-pointer logic:

```python
elif isinstance(size_or_type, cute.struct):
    size_in_bytes = size_or_type.__sizeof__()
    alignment = max(byte_alignment, size_or_type.__alignof__())
    base_ptr = self.allocate(size_in_bytes, alignment, loc=loc, ip=ip)
    return size_or_type(base_ptr)
...
self._base = self._base.align(byte_alignment)
ptr = self._base
self._base += size_in_bytes
...
assert self._allocated_bytes <= get_dyn_smem_size(...)
return ptr
```

Source: [smem_allocator.py](../python/CuTeDSL/cutlass/utils/smem_allocator.py#L126)

State before this frame:
- The kernel has access to a dynamic-shared-memory base pointer.

State after this frame:
- A byte subrange of that region is assigned to `SharedStorage`, and a typed struct view is returned.

### Frame 5: the DSL has explicit ops for static and dynamic shared memory

The shared-memory helpers make the distinction explicit:

```python
def alloc_smem(element_type, size_in_elems, alignment=None, ...):
    """Statically allocates SMEM."""
    return _cute_nvgpu_ir.arch_alloc_smem(...)

def get_dyn_smem(element_type, alignment=None, ...):
    """Retrieves a pointer to a dynamic SMEM allocation."""
    return _cute_nvgpu_ir.arch_get_dyn_smem(...)

def get_dyn_smem_size(...):
    return _cute_nvgpu_ir.arch_get_dyn_smem_size(...)
```

Source: [smem.py](../python/CuTeDSL/cutlass/cute/arch/smem.py#L23), [smem.py](../python/CuTeDSL/cutlass/cute/arch/smem.py#L64)

This matters because it answers the conceptual question:

- **Static shared memory** is represented in the IR as a fixed shared-memory object/allocation.
- **Dynamic shared memory** is represented in the IR as a special per-kernel shared-memory base pointer plus a launch-time size.

Neither requires a driver call from inside the kernel body.

### Frame 6: the host launch passes the dynamic shared-memory byte count

The CuTeDSL launch path forwards the `smem=` value as `dynamic_shared_memory_size`:

```python
CutlassBaseDSL.cuda_launch_func(
    ...,
    dynamic_shared_memory_size=cfg.smem,
    ...
)
```

Source: [cutlass.py](../python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py#L858)

The runtime ultimately launches with:

```python
cuda.cuLaunchKernel(
    kernel,
    ...,
    smem_size,
    stream,
    kernel_args,
    0,
)
```

Source: [cuda.py](../python/CuTeDSL/cutlass/base_dsl/runtime/cuda.py#L565)

That is where CUDA is told how many bytes of dynamic shared memory to reserve for each block.

State before this frame:
- The compiled kernel expects a dynamic-shared-memory region.

State after this frame:
- CUDA launches the kernel with a per-block shared-memory allocation of the requested size.

## What the MLIR says

A representative MLIR dump shows exactly this model:

```mlir
%smem_ptr = cute_nvgpu.arch.get_dyn_smem() : !cute.ptr<i8, smem, align<1024>>
...
%ptr = cute.add_offset(%smem_ptr, %int_tuple_84) : (!cute.ptr<i8, smem, align<1024>>, !cute.int_tuple<"20">) -> !cute.ptr<i8, smem, align<4>>
%smem_size = cute_nvgpu.arch.get_dyn_smem_size() : i32
...
cf.assert %19, "Allocation failed: shared memory allocation exceeds available memory set in kernel launch..."
```

Source: [00_input.mlir](../experiments/rms_norm_mlir_dump/00_input.mlir#L300)

This is the important compiler-level idea:

- MLIR is modeling shared memory as a **GPU address space / kernel resource**.
- The kernel IR uses ops like “get the dynamic shared-memory base pointer”.
- The kernel then does pointer arithmetic in shared memory.

There is no need for the kernel to emit a runtime “allocate shared memory now” call.

## What the PTX says

The corresponding PTX for the simple producer/consumer example contains:

```ptx
.extern .shared .align 1024 .b8 __dynamic_shmem__0[];
```

and then direct shared-memory accesses:

```ptx
st.shared.u32 [__dynamic_shmem__0], %r10;
ld.shared.f32 %f1, [__dynamic_shmem__0];
```

Source: [cutlass_run_synced_producer_consumer_Tensorgmemo81.sm_90a.ptx](../experiments/async_pp_dump/cutlass_run_synced_producer_consumer_Tensorgmemo81.sm_90a.ptx#L14)

That is the low-level evidence that this is just ordinary CUDA dynamic shared memory:

- `.extern .shared` means “this kernel has dynamic shared memory”
- the launch decides how large that region is
- the kernel code uses the symbol/pointer directly

## Conceptual Model for MLIR / LLVM Beginners

If you are coming from CUDA C++, it is easy to imagine that “allocating shared memory” must mean a runtime API call. For GPU compilers, that is usually the wrong mental model.

A better model is:

1. shared memory is a **special address space** owned by a thread block
2. the compiler emits IR/PTX that refers to that address space
3. for **static** shared memory, the compiler encodes a fixed-size object in the kernel
4. for **dynamic** shared memory, the host launch specifies the byte count
5. inside the kernel, code only computes addresses within that region

So LLVM/MLIR is not “replacing a CUDA driver allocation API”. It is compiling to GPU code that already understands the notion of shared-memory address spaces and kernel resources.

## Bottom Line

For this exact CuTeDSL example:

- `@cute.struct` describes a layout
- `Align[..., 1024]` forces alignment and rounds the struct size up
- `SmemAllocator` obtains the **dynamic shared-memory base pointer**
- `allocate(SharedStorage, 64)` carves out a byte range and returns a typed overlay
- `.launch(..., smem=SharedStorage.size_in_bytes())` tells CUDA how much shared memory to reserve
- MLIR lowers this to shared-memory address-space ops
- PTX lowers this to `.extern .shared` plus `ld.shared` / `st.shared`

So the answer is: CuTeDSL is not doing a driver allocation from inside LLVM-generated kernel code. It is lowering your Python abstraction into the normal GPU shared-memory model.

## Launch-Path Clarification

There are two related but distinct launch paths in this repo:

1. A direct CUDA-driver helper path, where Python loads a cubin and calls `cuLaunchKernel` explicitly:
   - [dsl.py](../python/CuTeDSL/cutlass/base_dsl/dsl.py#L1618)
   - [cuda.py](../python/CuTeDSL/cutlass/base_dsl/runtime/cuda.py#L565)

2. The `@cute.jit` host-JIT path, where MLIR's `ExecutionEngine` executes a generated host stub:
   - [dsl.py](../python/CuTeDSL/cutlass/base_dsl/dsl.py#L1255)
   - [jit_executor.py](../python/CuTeDSL/cutlass/base_dsl/jit_executor.py#L787)

For the `@cute.jit` path, the host stub is still not an opaque "internal CUDA runner". The lowered LLVM MLIR shows external CUDA launch symbols:

```mlir
llvm.func @_cudaLaunchKernelEx(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
llvm.func @_cudaLaunchKernel(!llvm.ptr, i32, i32, i32, i32, i32, i32, !llvm.ptr, i64, !llvm.ptr) -> i32
```

Source: [99_output.mlir](../experiments/rms_norm_mlir_dump/99_output.mlir#L1)

and the generated host function calls:

```mlir
%87 = llvm.call @_cudaLaunchKernelEx(%66, %86, %69) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
```

Source: [99_output.mlir](../experiments/rms_norm_mlir_dump/99_output.mlir#L286)

So the precise statement is:

- the repo does contain an explicit Python helper path that calls `cuLaunchKernel`
- the `@cute.jit` path goes through MLIR `ExecutionEngine`
- but the lowered host stub in that path still calls CUDA launch entry points (`_cudaLaunchKernelEx` / `_cudaLaunchKernel`), not a standalone MLIR-only kernel runner
