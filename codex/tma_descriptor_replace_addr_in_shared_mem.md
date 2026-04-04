# `tma_descriptor_replace_addr_in_shared_mem` trace

## Key Functions Index

| Function | File | Purpose |
|----------|------|---------|
| `tma_descriptor_replace_addr_in_shared_mem` | [../include/cute/arch/copy_sm90_desc.hpp](../include/cute/arch/copy_sm90_desc.hpp#L343) | Rewrites the tensor base address field in a shared-memory copy of a TMA descriptor |
| `tma_descriptor_cp_fence_release` | [../include/cute/arch/copy_sm90_desc.hpp](../include/cute/arch/copy_sm90_desc.hpp#L425) | Publishes the modified shared-memory descriptor back to a global-memory descriptor with the required fence-proxy semantics |
| `tma_descriptor_fence_acquire` | [../include/cute/arch/copy_sm90_desc.hpp](../include/cute/arch/copy_sm90_desc.hpp#L459) | Makes later TMA operations observe the updated global descriptor |
| `cast_smem_ptr_to_uint` | [../include/cute/arch/util.hpp](../include/cute/arch/util.hpp#L93) | Converts a generic pointer to a shared-memory state-space address usable by inline PTX |

## Code Under Discussion

```c++
CUTE_HOST_DEVICE
void
tma_descriptor_replace_addr_in_shared_mem(TmaDescriptor& smem_desc,
                                          void const* const new_tensor_ptr)
{
#if defined(CUTE_ARCH_DEVICE_MODIFIABLE_TMA_SM90_ENABLED)
  uint32_t smem_int_desc = cast_smem_ptr_to_uint(&smem_desc);
  uint64_t const new_desc_addr = reinterpret_cast<uint64_t>(new_tensor_ptr);
  asm volatile (
    "tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%0], %1;"
    :: "r"(smem_int_desc), "l"(new_desc_addr));
#else
  CUTE_INVALID_CONTROL_PATH("Using TMA Descriptor modification without CUTE_ARCH_DEVICE_MODIFIABLE_TMA_SM90_ENABLED and CUDA 12.3");
#endif
}
```

Source: [../include/cute/arch/copy_sm90_desc.hpp](../include/cute/arch/copy_sm90_desc.hpp#L343)

## Big Picture

A `TmaDescriptor` is just the in-memory tensor-map blob that Hopper/Blackwell TMA hardware consumes. In host builds, CUTLASS aliases it to `CUtensorMap`; in fallback/device-only builds it aliases it to a 128-byte, 64-byte-aligned POD blob:

```c++
#if (__CUDACC_VER_MAJOR__ >= 12) && !defined(__CUDACC_RTC__)
  using TmaDescriptor = CUtensorMap;
#else
  using TmaDescriptor = struct alignas(64) { char bytes[128]; };
#endif
```

Source: [../include/cute/arch/copy_sm90_desc.hpp](../include/cute/arch/copy_sm90_desc.hpp#L291)

That means it can physically live anywhere ordinary bytes can live:

- global memory
- parameter/constant storage
- CTA shared memory

The important distinction is not "can bytes live there?" but "which PTX instructions know how to interpret or mutate those bytes in that state space?"

## Why A `TmaDescriptor` Can Be In Shared Memory

The shared-memory form is a staging copy used when a CTA wants to patch a descriptor at runtime without rebuilding it on the host.

CUTLASS does exactly that in grouped/array kernels:

```c++
Tensor pA_tensormap = make_tensor(mainloop_params.tma_load_a.get_tma_descriptor(), Int<1>{}, Int<1>{});
Tensor sA_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_A), Int<1>{}, Int<1>{});
copy(recast<uint128_t>(pA_tensormap), recast<uint128_t>(sA_tensormap));
```

Source: [../include/cutlass/gemm/collective/sm90_mma_array_tma_gmma_ss_warpspecialized.hpp](../include/cutlass/gemm/collective/sm90_mma_array_tma_gmma_ss_warpspecialized.hpp#L642)

So the lifecycle is:

1. Host code encodes a canonical `CUtensorMap` with `cuTensorMapEncodeTiled(...)`.
2. The kernel receives a pointer to that 128-byte descriptor in global/param memory.
3. One thread/warp copies those 128 bytes into shared memory.
4. `tensormap.replace.*.shared::cta` mutates fields in the shared copy.
5. `tensormap.cp_fenceproxy.global.shared::cta...` publishes the modified 128-byte shared copy back into a global descriptor.
6. `fence.proxy.tensormap::generic.acquire.gpu` makes subsequent TMA users observe the updated descriptor.

The PTX ISA supports this flow explicitly. The `tensormap.replace` family has both `.global` and `.shared::cta` variants for a 1024-bit tensor-map object, and the CUTLASS helpers pair the shared-memory replace with the required fence-proxy publish step.

## Frame-By-Frame Trace Of This Function

### Frame 0: Entry state

Inputs:

- `smem_desc`: a `TmaDescriptor` object already resident in CTA shared memory
- `new_tensor_ptr`: a new global-memory base address for the tensor payload

Assumption:

- the descriptor's shape/stride/swizzle metadata is still valid, and only the backing tensor address is changing

### Frame 1: Convert the descriptor pointer into a shared-memory address

```c++
uint32_t smem_int_desc = cast_smem_ptr_to_uint(&smem_desc);
```

`cast_smem_ptr_to_uint` uses `__cvta_generic_to_shared` or equivalent PTX to turn a generic C++ pointer into the shared-memory state-space address expected by inline PTX shared instructions.

Source: [../include/cute/arch/util.hpp](../include/cute/arch/util.hpp#L93)

Before:

- `&smem_desc` is a generic device pointer value at the C++ level

After:

- `smem_int_desc` is the 32-bit shared-memory address of the first byte of the 128-byte tensor-map blob

### Frame 2: Materialize the replacement global address

```c++
uint64_t const new_desc_addr = reinterpret_cast<uint64_t>(new_tensor_ptr);
```

This is the new tensor base pointer encoded as a raw 64-bit global address. The PTX form uses `.b64`, so the replacement value must be a 64-bit operand.

### Frame 3: Issue the PTX tensor-map field replacement

```c++
asm volatile (
  "tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%0], %1;"
  :: "r"(smem_int_desc), "l"(new_desc_addr));
```

This is the real operation. Parsed left to right:

- `tensormap.replace`
  Replaces one field inside an opaque tensor-map object
- `.tile`
  The tensor-map mode is tiled
- `.global_address`
  The field being replaced is the tensor's global base address
- `.shared::cta`
  The tensor-map object being modified is in CTA shared memory
- `.b1024`
  The instruction treats the tensor-map as a 1024-bit object, i.e. 128 bytes
- `.b64`
  The replacement field value is 64 bits wide

Semantically, PTX reads the tensor-map blob starting at `smem_int_desc`, overwrites only its `global_address` field with `new_desc_addr`, and writes the updated blob back to that same shared-memory location.

What it does **not** do:

- it does not move tensor payload data
- it does not re-encode the entire descriptor
- it does not by itself publish the update to the global descriptor later used by TMA consumers

## Why The Shared-Memory Update Is Not The End Of The Story

After mutating the shared copy, CUTLASS publishes it back:

```c++
asm volatile (
  "tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [%0], [%1], 128;"
  :: "l"(gmem_int_desc), "r"(smem_int_desc));
```

Source: [../include/cute/arch/copy_sm90_desc.hpp](../include/cute/arch/copy_sm90_desc.hpp#L425)

And later acquires it:

```c++
asm volatile (
  "fence.proxy.tensormap::generic.acquire.gpu [%0], 128;"
  :
  : "l"(gmem_int_desc)
  : "memory");
```

Source: [../include/cute/arch/copy_sm90_desc.hpp](../include/cute/arch/copy_sm90_desc.hpp#L459)

So the full intent is:

- edit cheaply in shared memory
- publish the edited 128-byte descriptor back to global memory with tensor-map proxy ordering
- acquire before using the global descriptor in later TMA operations

## How This Relates To The CUDA Driver API

`CUtensorMap` is the driver API object encoded by routines such as `cuTensorMapEncodeTiled(...)` and updated on the host by routines such as `cuTensorMapReplaceAddress(...)`.

CUTLASS is doing the device-side analog:

- host side: build or patch the `CUtensorMap` through driver APIs
- device side: patch fields in the same 128-byte object through `tensormap.replace.*`

The object layout is opaque at the API level, but both the PTX ISA and the CUDA driver API agree on the same underlying tensor-map object format.

## Practical Interpretation

So the answer to your two questions is:

1. A `TmaDescriptor` can be in shared memory because it is just a 128-byte opaque descriptor blob, and Hopper PTX defines shared-memory tensor-map replacement instructions that operate on that blob in `shared::cta`.
2. `tensormap.replace.tile.global_address.shared::cta.b1024.b64` rewrites only the `global_address` field of that 128-byte shared-memory tensor-map object with a new 64-bit global pointer, leaving the rest of the descriptor unchanged.

## References

- PTX ISA `tensormap.replace`: https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-tensormap-replace
- CUDA Driver API tensor memory management: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html

## Process Note

- I inspected the local CUTLASS/CUTE definitions for `TmaDescriptor`, `cast_smem_ptr_to_uint`, and the related fence helpers.
- I checked representative CUTLASS call sites that copy a descriptor into shared memory, mutate it, and publish it back.
- I cross-checked the PTX ISA and CUDA Driver API docs to line up the `.shared::cta`, `.b1024`, and `.b64` semantics with the CUTLASS wrapper.
- No sub-agents were used.
