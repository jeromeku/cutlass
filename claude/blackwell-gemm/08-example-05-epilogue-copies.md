# Example 05 Epilogue Copy Operations - Complete Call Traces

This document provides **complete call path traces** for all four copy operations in the epilogue of Example 05, showing every function call from the high-level API down to the inline PTX assembly.

## Overview of Example 05 Epilogue

Example 05 demonstrates a complete GEMM with TMA epilogue that performs:
```
D = beta * C + alpha * (A * B)
```

The epilogue involves **four distinct copy operations** with different memory spaces:

| Line | Operation | Memory Flow | Copy Function |
|------|-----------|-------------|---------------|
| 434 | Load C | SMEM → RMEM | `copy_aligned()` |
| 437 | Load Accumulator | TMEM → RMEM | `copy()` with TMEM copy atom |
| 444 | Store D | RMEM → SMEM | `copy_aligned()` |
| 450 | TMA Store D | SMEM → GMEM | `copy()` with TMA store atom |

Let's trace each one in detail.

---

## Copy 1: SMEM → RMEM (Load C) - Line 434

### Starting Point in Example Code

**File**: [examples/cute/tutorial/blackwell/05_mma_tma_epi_sm100.cu:434](../../examples/cute/tutorial/blackwell/05_mma_tma_epi_sm100.cu#L434)

```cpp
// Load C:  SMEM -> RMEM
copy_aligned(tTR_sC, tTR_rC);
```

### Tensor Setup Context

**Lines 411-419**:
```cpp
// Partition for TMEM accumulators load (TMEM -> RMEM)
TiledCopy t2r_copy = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tAcc_epi(_,_0{}));
ThrCopy   thr_t2r  = t2r_copy.get_slice(threadIdx.x);
Tensor tTR_tAcc = thr_t2r.partition_S(tAcc_epi);          // (TmemCpy,NumTmemCpy,NumTiles)
Tensor tTR_sC   = thr_t2r.partition_D(sC_epi);            // (TmemCpy,NumTmemCpy)
Tensor tTR_sD   = thr_t2r.partition_D(sD_epi);            // (TmemCpy,NumTmemCpy)
// Allocate register tensors
Tensor tTR_rC = make_tensor_like(tTR_sC);                 // (TmemCpy,NumTmemCpy)
Tensor tTR_rD = make_fragment_like(tTR_sD);               // (TmemCpy,NumTmemCpy)
```

**Tensor Types**:
- `tTR_sC`: `Tensor<smem_ptr<float>, Layout<...>>` - Per-thread SMEM C slice
- `tTR_rC`: `Tensor<float*, Layout<...>>` - Per-thread register C storage

### Complete Call Path

```
┌─────────────────────────────────────────────────────────────────┐
│ Level 1: copy_aligned() Function                               │
└─────────────────────────────────────────────────────────────────┘
copy_aligned(tTR_sC, tTR_rC)
  ↓
  Location: include/cute/algorithm/copy.hpp:333-342

  template <class SrcEngine, class SrcLayout,
            class DstEngine, class DstLayout>
  CUTE_HOST_DEVICE
  void
  copy_aligned(Tensor<SrcEngine, SrcLayout> const& src,
               Tensor<DstEngine, DstLayout>      & dst)
  {
    if constexpr (is_static<decltype(shape(src))>::value &&
                  is_static<decltype(shape(dst))>::value) {
      // ← TRUE: Both tensors have static shapes
      return copy(AutoFilter(AutoVectorizingCopyWithAssumedAlignment<128>{}), src, dst);
    } else {
      return copy(AutoVectorizingCopyWithAssumedAlignment<128>{}, src, dst);
    }
  }

  AutoFilter is a wrapper that adds filtering capabilities.
  AutoVectorizingCopyWithAssumedAlignment<128> assumes 128-bit (16-byte) alignment.

┌─────────────────────────────────────────────────────────────────┐
│ Level 2: copy() with AutoFilter                                │
└─────────────────────────────────────────────────────────────────┘
copy(AutoFilter(AutoVectorizingCopyWithAssumedAlignment<128>{}), tTR_sC, tTR_rC)
  ↓
  Location: include/cute/algorithm/copy.hpp:288-313

  template <class CopyOp,
            class SrcEngine, class SrcLayout,
            class DstEngine, class DstLayout>
  CUTE_HOST_DEVICE
  void
  copy(AutoFilter<CopyOp>           const& copy_op,
       Tensor<SrcEngine, SrcLayout> const& src,
       Tensor<DstEngine, DstLayout>      & dst)
  {
    // Filter out the elements of src/dst that don't overlap with each other
    auto [filter_src, filter_dst] = filter_zeros(src, dst);

    if constexpr (is_constant<0, decltype(size(filter_src))>::value) {
      // ← FALSE: Tensors are non-empty
      return;
    } else {
      // ← TRUE: Proceed with filtered tensors
      return copy(copy_op.base, filter_src, filter_dst);  // ← DISPATCH
    }
  }

  filter_zeros() removes elements with zero strides that don't contribute to the copy.
  After filtering, dispatch to the base copy operation.

┌─────────────────────────────────────────────────────────────────┐
│ Level 3: copy() with AutoVectorizingCopyWithAssumedAlignment   │
└─────────────────────────────────────────────────────────────────┘
copy(AutoVectorizingCopyWithAssumedAlignment<128>{}, filter_src, filter_dst)
  ↓
  Location: include/cute/algorithm/copy.hpp:247-274

  template <int MaxVecBits,
            class SrcEngine, class SrcLayout,
            class DstEngine, class DstLayout>
  CUTE_HOST_DEVICE
  void
  copy(AutoVectorizingCopyWithAssumedAlignment<MaxVecBits> const&,
       Tensor<SrcEngine, SrcLayout>                        const& src,
       Tensor<DstEngine, DstLayout>                             & dst)
  {
    constexpr int common_elem = CUTE_STATIC_V(max_common_vector(src, dst));
    static_assert(is_integral<decltype(Int<common_elem>{} * sizeof_bits_v<typename DstEngine::value_type>)>::value,
                  "Error: Attempting a subbit write!");

    if constexpr (common_elem > 1)  // ← TRUE: Multiple elements can vectorize
    {
      constexpr int align_bits = CUTE_STATIC_V(gcd(max_alignment(src), max_alignment(dst), Int<MaxVecBits>{}));
      constexpr int vec_bits   = gcd(common_elem * sizeof_bits_v<typename DstEngine::value_type>, align_bits);

      if constexpr ((vec_bits % 8) == 0 && sizeof_bits_v<typename DstEngine::value_type> < Int<vec_bits>{})
      {
        // ← TRUE: Can vectorize with larger type
        using VecType = uint_bit_t<vec_bits>;  // e.g., uint128_t for 128-bit vectors

        // Recast tensors to vectorized type
        Tensor src_v = recast<VecType>(src);
        Tensor dst_v = recast<VecType>(dst);
        return copy_if(constant_fn<true_type>{}, src_v, dst_v);  // ← DISPATCH
      } else {
        return copy_if(constant_fn<true_type>{}, src, dst);
      }
    } else {
      return copy_if(constant_fn<true_type>{}, src, dst);
    }
  }

  Key points:
  - max_common_vector(): Finds largest contiguous vector in both src and dst
  - Alignment analysis: Determines safe vectorization width (up to 128 bits)
  - Recasting: Converts float* to uint128_t* for 4-wide float loads/stores
  - Example: 4 consecutive floats (128 bits) → single uint128_t load/store

┌─────────────────────────────────────────────────────────────────┐
│ Level 4: copy_if() - Predicated Copy                           │
└─────────────────────────────────────────────────────────────────┘
copy_if(constant_fn<true_type>{}, src_v, dst_v)
  ↓
  Location: include/cute/algorithm/copy.hpp:48-62

  template <class PrdTensor,
            class SrcEngine, class SrcLayout,
            class DstEngine, class DstLayout>
  CUTE_HOST_DEVICE
  void
  copy_if(PrdTensor                    const& pred,
          Tensor<SrcEngine, SrcLayout> const& src,
          Tensor<DstEngine, DstLayout>      & dst)
  {
    using SrcType = typename SrcEngine::value_type;  // uint128_t
    using DstType = typename DstEngine::value_type;  // uint128_t

    CUTE_UNROLL  // ← Compiler directive to unroll loop
    for (int i = 0; i < size(dst); ++i) {
      if (pred(i)) {  // ← constant_fn<true_type> always returns true
        dst(i) = static_cast<DstType>(static_cast<SrcType>(src(i)));  // ← FINAL COPY
      }
    }
  }

  This generates:
  - Unrolled loop over vectorized elements
  - Direct assignments: dst(i) = src(i)
  - For uint128_t: Single 128-bit load + single 128-bit store per iteration

┌─────────────────────────────────────────────────────────────────┐
│ Level 5: Generated Assembly (SMEM → RMEM)                      │
└─────────────────────────────────────────────────────────────────┘

  Example PTX (simplified for 4 consecutive floats as uint128_t):

  {
    .reg .b32 %r<16>;    // 16 registers (4 floats × 4 bytes = 16 bytes / 4-byte regs)

    // Load 128 bits from SMEM (4 floats)
    ld.shared.v4.b32 {%r0, %r1, %r2, %r3}, [smem_addr];

    // Store 128 bits to RMEM (registers)
    // This is implicit - values are already in registers %r0-%r3
    // They remain in registers for subsequent axpby operation
  }

  Actual instruction: ld.shared.v4.b32
  - .shared: Load from shared memory
  - .v4: Vector of 4 elements
  - .b32: Each element is 32 bits
  - Total: 128-bit vectorized load
```

### Summary Diagram: SMEM → RMEM

```
copy_aligned(tTR_sC, tTR_rC)
  ↓
copy(AutoFilter(AutoVectorizingCopyWithAssumedAlignment<128>{}), ...) [copy.hpp:338]
  ↓ filter_zeros()
copy(AutoVectorizingCopyWithAssumedAlignment<128>{}, ...) [copy.hpp:247]
  ↓ analyze vectorization: common_elem=4, vec_bits=128
  ↓ recast<uint128_t>(...)
copy_if(constant_fn<true_type>{}, src_v, dst_v) [copy.hpp:48]
  ↓ unrolled loop
PTX: ld.shared.v4.b32 {%r0-%r3}, [smem_addr]
```

**Key Optimization**: 4 consecutive floats (16 bytes) loaded in a single 128-bit instruction.

---

## Copy 2: TMEM → RMEM (Load Accumulator) - Line 437

### Starting Point in Example Code

**File**: [examples/cute/tutorial/blackwell/05_mma_tma_epi_sm100.cu:437](../../examples/cute/tutorial/blackwell/05_mma_tma_epi_sm100.cu#L437)

```cpp
// Load Acc:  TMEM -> RMEM
copy(t2r_copy, tTR_tAcc(_,_,epi_tile_idx), tTR_rD);
```

### Tensor Setup

**Lines 412-413**:
```cpp
TiledCopy t2r_copy = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tAcc_epi(_,_0{}));
ThrCopy   thr_t2r  = t2r_copy.get_slice(threadIdx.x);
```

**Tensor Types**:
- `t2r_copy`: `TiledCopy` with `SM100_TMEM_LOAD_32dp32b1x` copy operation
- `tTR_tAcc`: `Tensor<tmem_ptr<float>, Layout<...>>` - Per-thread TMEM accumulator slice
- `tTR_rD`: `Tensor<float*, Layout<...>>` - Per-thread register D storage

### Complete Call Path

```
┌─────────────────────────────────────────────────────────────────┐
│ Level 1: Generic copy() with TiledCopy                         │
└─────────────────────────────────────────────────────────────────┘
copy(t2r_copy, tTR_tAcc(_,_,epi_tile_idx), tTR_rD)
  ↓
  Location: include/cute/algorithm/copy.hpp:189-207

  template <class... CopyArgs,
            class SrcEngine, class SrcLayout,
            class DstEngine, class DstLayout>
  CUTE_HOST_DEVICE
  void
  copy(Copy_Atom<CopyArgs...>       const& copy_atom,
       Tensor<SrcEngine, SrcLayout> const& src,
       Tensor<DstEngine, DstLayout>      & dst)
  {
    static_assert(SrcLayout::rank == DstLayout::rank, "CopyAtom rank-mismatch.");

    if constexpr (SrcLayout::rank == 1) {   // ← FALSE: rank > 1
      copy_atom.call(src, dst);
    } else {                                // ← TRUE: Loop over modes
      constexpr int R = SrcLayout::rank;
      Tensor src_v = group_modes<1,R>(src);  // Group all but first mode
      Tensor dst_v = group_modes<1,R>(dst);
      CUTE_UNROLL
      for (int i = 0; i < size<1>(dst_v); ++i) {
        copy_atom.call(src_v(_,i), dst_v(_,i));  // ← RECURSIVE DISPATCH
      }
    }
  }

  This loops over epilogue tiles, calling copy_atom.call() for each.

┌─────────────────────────────────────────────────────────────────┐
│ Level 2: Copy_Atom::call() Method                              │
└─────────────────────────────────────────────────────────────────┘
Copy_Atom<SM100_TMEM_LOAD_32dp32b1x>::call(src_v(_,i), dst_v(_,i))
  ↓
  Location: include/cute/atom/copy_atom.hpp:94-114

  template <class SEngine, class SLayout,
            class DEngine, class DLayout>
  CUTE_HOST_DEVICE
  void
  call(Tensor<SEngine,SLayout> const& src,
       Tensor<DEngine,DLayout>      & dst) const
  {
    static_assert(SLayout::rank == 1, "Expected rank-1 src tensor");
    static_assert(DLayout::rank == 1, "Expected rank-1 dst tensor");

    if constexpr (is_constant<NumValSrc, decltype(size(src))>::value ||
                  is_constant<NumValDst, decltype(size(dst))>::value) {
      // ← TRUE: Size matches TMEM load instruction
      return copy_unpack(static_cast<Traits const&>(*this), src, dst);  // ← DISPATCH
    } else {
      // Recurse (not taken)
    }
  }

┌─────────────────────────────────────────────────────────────────┐
│ Level 3: copy_unpack() for TMEM Load                           │
└─────────────────────────────────────────────────────────────────┘
copy_unpack(Copy_Traits<SM100_TMEM_LOAD_32dp32b1x> const&, src, dst)
  ↓
  Location: include/cute/atom/copy_traits_sm100.hpp:97-148

  The copy_unpack for TMEM loads is defined via Copy_Traits specialization.

  template <class NumBits>
  struct Copy_Traits<SM100_TMEM_LOAD_32dp32b1x, NumBits>
  {
    using ThrID     = Layout<Shape<_32,_4>>;  // 32 threads/warp × 4 warps
    using SrcLayout = ...;  // TMEM layout
    using DstLayout = ...;  // Register layout

    // copy_unpack is a friend function
    template <class TS, class SLayout,
              class TD, class DLayout>
    CUTE_HOST_DEVICE friend constexpr void
    copy_unpack(Copy_Traits           const& traits,
                Tensor<TS,SLayout>    const& src,
                Tensor<TD,DLayout>         & dst)
    {
      // Extract TMEM address and register pointer
      auto tmem_addr = src.data();
      auto reg_ptr = dst.data();

      // Dispatch to actual TMEM load operation
      return SM100_TMEM_LOAD_32dp32b1x::copy(tmem_addr, reg_ptr);  // ← DISPATCH
    }
  };

┌─────────────────────────────────────────────────────────────────┐
│ Level 4: SM100_TMEM_LOAD_32dp32b1x::copy() Static Method       │
└─────────────────────────────────────────────────────────────────┘
SM100_TMEM_LOAD_32dp32b1x::copy(tmem_addr, reg_ptr)
  ↓
  Location: include/cute/arch/copy_sm100.hpp (similar structure to SM90)

  struct SM100_TMEM_LOAD_32dp32b1x
  {
    CUTE_HOST_DEVICE static void
    copy(uint32_t const& tmem_addr,  // TMEM address
         float*           reg_ptr)    // Register destination
    {
  #if defined(CUTE_ARCH_TCGEN05_ENABLED)
      // PTX INSTRUCTION ← FINAL HARDWARE CALL
      asm volatile(
        "tcgen05.ld.sync.aligned.32x1x32b.x1.b32 %0, [%1];"
        : "=r"(*reinterpret_cast<uint32_t*>(reg_ptr))  // Output: register
        : "r"(tmem_addr)                               // Input: TMEM address
        : "memory"
      );
  #else
      CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_TMEM_LOAD without CUTE_ARCH_TCGEN05_ENABLED");
  #endif
    }
  };
```

### PTX Instruction Breakdown (TMEM → RMEM)

```ptx
tcgen05.ld.sync.aligned.32x1x32b.x1.b32 %0, [%1];
```

**Instruction Components**:
- `tcgen05.ld` - Tensor Core Generation 05 load instruction
- `.sync.aligned` - Synchronous, aligned access
- `.32x1x32b` - 32 data paths × 1 repetition × 32 bits
- `.x1` - Throughput multiplier
- `.b32` - 32-bit element size
- `%0` - Output register
- `[%1]` - TMEM address (32-bit)

**What Happens**:
1. Hardware reads from TMEM at address `%1`
2. Loads 32 bits (1 float) per thread
3. Stores to register `%0`
4. Synchronous: Blocks until data arrives
5. Each of 128 threads loads its own element

### Summary Diagram: TMEM → RMEM

```
copy(t2r_copy, tTR_tAcc(_,_,epi_tile_idx), tTR_rD)
  ↓ loop over rank
copy(Copy_Atom<SM100_TMEM_LOAD_32dp32b1x>, ...) [copy.hpp:189]
  ↓
Copy_Atom::call() [copy_atom.hpp:94]
  ↓
copy_unpack() [copy_traits_sm100.hpp]
  ↓ extract TMEM addr & reg ptr
SM100_TMEM_LOAD_32dp32b1x::copy() [copy_sm100.hpp]
  ↓
PTX: tcgen05.ld.sync.aligned.32x1x32b.x1.b32 %r, [tmem_addr]
```

**Key Point**: Specialized TMEM load instruction, each thread loads independently.

---

## Copy 3: RMEM → SMEM (Store D) - Line 444

### Starting Point in Example Code

**File**: [examples/cute/tutorial/blackwell/05_mma_tma_epi_sm100.cu:444](../../examples/cute/tutorial/blackwell/05_mma_tma_epi_sm100.cu#L444)

```cpp
// Store D:  RMEM -> SMEM
__syncthreads(); // Ensure C loads are finished before reusing smem
copy_aligned(tTR_rD, tTR_sD);
```

### Tensor Types

- `tTR_rD`: `Tensor<float*, Layout<...>>` - Per-thread register D (result of axpby)
- `tTR_sD`: `Tensor<smem_ptr<float>, Layout<...>>` - Per-thread SMEM D slice

### Complete Call Path

This follows **exactly the same path** as Copy 1 (SMEM → RMEM), but in reverse direction.

```
┌─────────────────────────────────────────────────────────────────┐
│ Level 1: copy_aligned() Function                               │
└─────────────────────────────────────────────────────────────────┘
copy_aligned(tTR_rD, tTR_sD)
  ↓
  Same as Copy 1, but src=RMEM, dst=SMEM

┌─────────────────────────────────────────────────────────────────┐
│ Levels 2-4: Same Dispatch as Copy 1                            │
└─────────────────────────────────────────────────────────────────┘
  AutoFilter → AutoVectorizingCopyWithAssumedAlignment<128> → copy_if

┌─────────────────────────────────────────────────────────────────┐
│ Level 5: Generated Assembly (RMEM → SMEM)                      │
└─────────────────────────────────────────────────────────────────┘

  Example PTX (simplified for 4 consecutive floats as uint128_t):

  {
    .reg .b32 %r<16>;

    // Values already in registers %r0-%r3 from previous computation

    // Store 128 bits to SMEM (4 floats)
    st.shared.v4.b32 [smem_addr], {%r0, %r1, %r2, %r3};
  }

  Actual instruction: st.shared.v4.b32
  - .shared: Store to shared memory
  - .v4: Vector of 4 elements
  - .b32: Each element is 32 bits
  - Total: 128-bit vectorized store
```

### Summary Diagram: RMEM → SMEM

```
copy_aligned(tTR_rD, tTR_sD)
  ↓
[Same dispatch path as Copy 1, but reversed direction]
  ↓
PTX: st.shared.v4.b32 [smem_addr], {%r0-%r3}
```

**Key Point**: Symmetric to Copy 1, vectorized store instead of load.

---

## Copy 4: SMEM → GMEM (TMA Store D) - Line 450

### Starting Point in Example Code

**File**: [examples/cute/tutorial/blackwell/05_mma_tma_epi_sm100.cu:446-453](../../examples/cute/tutorial/blackwell/05_mma_tma_epi_sm100.cu#L446-L453)

```cpp
// TMA Store D:  SMEM -> GMEM
tma_store_fence(); // Ensure D smem stores are visible to TMA
__syncthreads(); // Ensure all threads have issued fence
if (elect_one_warp && elect_one_thr) {
  copy(tma_atom_D, tSG_sD, tSG_gD(_,epi_tile_idx));
  tma_store_arrive(); // issuing thread commits D TMA store
  tma_store_wait<0>(); // issuing thread waits for D TMA store to complete
}
__syncthreads(); // All threads sync with issuing thread
```

### Tensor Setup

**Lines 404-406**:
```cpp
// Construct corresponding SMEM tensors
Tensor sD_epi = shared_storage.tensor_sD();               // (EpiTile)

// Partition for TMA
auto [tSG_gD, tSG_sD] = tma_partition(tma_atom_D, sD_epi, gD_epi); // (SMEM -> GMEM)
```

**Tensor Types**:
- `tma_atom_D`: `Copy_Atom<SM90_TMA_STORE, ...>` - TMA store descriptor
- `tSG_sD`: `Tensor<smem_ptr<float>, Layout<...>>` - SMEM source
- `tSG_gD`: Coordinate tensor for GMEM destination

### Complete Call Path

```
┌─────────────────────────────────────────────────────────────────┐
│ Level 0: Preparation - TMA Store Fence                         │
└─────────────────────────────────────────────────────────────────┘
tma_store_fence()
  ↓
  Location: include/cute/arch/copy_sm90_tma.hpp:1214-1221

  CUTE_HOST_DEVICE static void
  tma_store_fence() {
  #if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    cutlass::arch::synclog_emit_fence_view_async_shared(__LINE__);
    asm volatile ("fence.proxy.async.shared::cta;");
  #endif
  }

  PTX: fence.proxy.async.shared::cta;
  - Ensures all prior SMEM stores are visible to TMA unit
  - Required before TMA can read from SMEM

┌─────────────────────────────────────────────────────────────────┐
│ Level 1: Generic copy() with TMA Store Atom                    │
└─────────────────────────────────────────────────────────────────┘
copy(tma_atom_D, tSG_sD, tSG_gD(_,epi_tile_idx))
  ↓
  Location: include/cute/algorithm/copy.hpp:189-207

  // Same dispatch as TMA Load (Copy 1 in example 03)
  // Loops over ranks, eventually reaches copy_atom.call()

┌─────────────────────────────────────────────────────────────────┐
│ Level 2: Copy_Atom::call() Method                              │
└─────────────────────────────────────────────────────────────────┘
Copy_Atom<SM90_TMA_STORE>::call(src, dst)
  ↓
  Location: include/cute/atom/copy_atom.hpp:94-114

  // Same structure as TMA load
  // Dispatches to copy_unpack()

┌─────────────────────────────────────────────────────────────────┐
│ Level 3: copy_unpack() for TMA Store                           │
└─────────────────────────────────────────────────────────────────┘
copy_unpack(Copy_Traits<SM90_TMA_STORE> const&, src, dst)
  ↓
  Location: include/cute/atom/copy_traits_sm90_tma.hpp:1675-1697

  The TMA_STORE_Unpack base class defines copy_unpack:

  template <class CopyOp, class... Args>
  struct TMA_STORE_Unpack
  {
    template <class TS, class SLayout,
              class TD, class DLayout>
    CUTE_HOST_DEVICE friend constexpr void
    copy_unpack(Copy_Traits<CopyOp, Args...> const& traits,
                Tensor<TS,SLayout>           const& src,
                Tensor<TD,DLayout>           const& dst)
    {
      static_assert(is_smem<TS>::value, "SM90_TMA_STORE requires the source be shared memory.");

      void const* src_ptr = cute::raw_pointer_cast(src.data());  // SMEM pointer
      auto dst_coord = dst.data().coord_;                        // GMEM coordinates

      // Explode tuple and dispatch to CopyOp::copy()
      return detail::explode_tuple(
        detail::CallCOPY<CopyOp>{},                  // Functor
        make_tuple(&traits.tma_desc_), seq<0>{},     // TMA descriptor
        make_tuple(src_ptr), seq<0>{},               // SMEM source
        dst_coord, tuple_seq<decltype(dst_coord)>{}  // GMEM coords
      );  // ← DISPATCH
    }
  };

  Similar to TMA load, but src=SMEM, dst=GMEM coordinates.

┌─────────────────────────────────────────────────────────────────┐
│ Level 4: detail::CallCOPY Functor                              │
└─────────────────────────────────────────────────────────────────┘
detail::CallCOPY<SM90_TMA_STORE>::operator()(desc_ptr, src_ptr, crd0, crd1, ...)
  ↓
  Unpacks arguments and forwards to CopyOp::copy()

┌─────────────────────────────────────────────────────────────────┐
│ Level 5: Dispatch to Correct Dimensionality                    │
└─────────────────────────────────────────────────────────────────┘
SM90_TMA_STORE::copy(desc_ptr, src_ptr, crd0, ...)
  ↓
  Location: include/cute/arch/copy_sm90_tma.hpp:1072-1111

  The SM90_TMA_STORE struct provides overloaded copy() methods for different dims:

  struct SM90_TMA_STORE
  {
    CUTE_HOST_DEVICE static void
    copy(void const* desc_ptr, void const* smem_ptr, int32_t const& crd0)
    { return SM90_TMA_STORE_1D::copy(desc_ptr, smem_ptr, crd0); }

    CUTE_HOST_DEVICE static void
    copy(void const* desc_ptr, void const* smem_ptr,
         int32_t const& crd0, int32_t const& crd1)
    { return SM90_TMA_STORE_2D::copy(desc_ptr, smem_ptr, crd0, crd1); }

    // ... 3D, 4D, 5D versions
  };

  For 2D D matrix (M×N), dispatches to SM90_TMA_STORE_2D::copy()

┌─────────────────────────────────────────────────────────────────┐
│ Level 6: TMA Hardware Instruction (2D Store)                   │
└─────────────────────────────────────────────────────────────────┘
SM90_TMA_STORE_2D::copy(desc_ptr, smem_ptr, crd0, crd1)
  ↓
  Location: include/cute/arch/copy_sm90_tma.hpp:980-1001

  struct SM90_TMA_STORE_2D
  {
    CUTE_HOST_DEVICE static void
    copy(void const* desc_ptr,
         void const* smem_ptr,
         int32_t const& crd0,      // M coordinate
         int32_t const& crd1)      // N coordinate
    {
  #if defined(CUTE_ARCH_TMA_SM90_ENABLED)
      uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
      uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
      cutlass::arch::synclog_emit_tma_store(__LINE__, gmem_int_desc, smem_int_ptr);

      // PTX INSTRUCTION ← FINAL HARDWARE CALL
      asm volatile (
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
        :  // No outputs
        : "l"(gmem_int_desc),  // %0 - TMA descriptor (64-bit)
          "r"(smem_int_ptr),   // %1 - SMEM source address (32-bit)
          "r"(crd0),           // %2 - M coordinate
          "r"(crd1)            // %3 - N coordinate
        : "memory"
      );
  #else
      CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
  #endif
    }
  };

┌─────────────────────────────────────────────────────────────────┐
│ Level 7: TMA Store Commit and Wait                             │
└─────────────────────────────────────────────────────────────────┘
tma_store_arrive()
  ↓
  Location: include/cute/arch/copy_sm90_tma.hpp:1225-1232

  CUTE_HOST_DEVICE static void
  tma_store_arrive() {
  #if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    cutlass::arch::synclog_emit_tma_store_arrive(__LINE__);
    asm volatile("cp.async.bulk.commit_group;");
  #endif
  }

  PTX: cp.async.bulk.commit_group;
  - Commits the TMA store to a bulk group
  - Allows tracking of completion

tma_store_wait<0>()
  ↓
  Location: include/cute/arch/copy_sm90_tma.hpp:1248-1259

  template <int Count>
  CUTE_HOST_DEVICE static void
  tma_store_wait() {
  #if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    asm volatile(
      "cp.async.bulk.wait_group.read %0;"
      :
      : "n"(Count)   // Count = 0: wait for all
      : "memory");
    cutlass::arch::synclog_emit_tma_store_wait(__LINE__, Count);
  #endif
  }

  PTX: cp.async.bulk.wait_group.read 0;
  - Waits until all TMA stores in group complete
  - Blocks issuing thread
```

### PTX Instruction Breakdown (SMEM → GMEM)

```ptx
// Step 1: Fence
fence.proxy.async.shared::cta;

// Step 2: TMA Store
cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];

// Step 3: Commit
cp.async.bulk.commit_group;

// Step 4: Wait
cp.async.bulk.wait_group.read 0;
```

**Instruction Components (TMA Store)**:
- `cp.async.bulk.tensor.2d` - Asynchronous bulk copy of 2D tensor
- `.global` - Destination is global memory
- `.shared::cta` - Source is CTA-local shared memory
- `.bulk_group` - Part of bulk group for tracking
- `[%0, {%2, %3}]` - TMA descriptor + coordinates (desc is 64-bit, coords are 32-bit each)
- `[%1]` - SMEM source address (32-bit)

**What Happens**:
1. **Fence**: All SMEM stores become visible to TMA unit
2. **TMA Store**: Hardware reads descriptor, computes GMEM address, initiates async transfer
3. **Commit**: Store is added to bulk group for tracking
4. **Wait**: Blocks until transfer completes
5. **Async**: TMA executes in background, thread waits at wait instruction

### Summary Diagram: SMEM → GMEM (TMA Store)

```
tma_store_fence() [copy_sm90_tma.hpp:1214]
  ↓ PTX: fence.proxy.async.shared::cta
__syncthreads()
  ↓
copy(tma_atom_D, tSG_sD, tSG_gD(_,epi_tile_idx))
  ↓
Generic copy() [copy.hpp:189]
  ↓
Copy_Atom::call() [copy_atom.hpp:94]
  ↓
copy_unpack() [copy_traits_sm90_tma.hpp:1675] (TMA_STORE_Unpack)
  ↓ explode_tuple()
detail::CallCOPY<SM90_TMA_STORE>::operator()
  ↓
SM90_TMA_STORE_2D::copy() [copy_sm90_tma.hpp:980]
  ↓ PTX: cp.async.bulk.tensor.2d.global.shared::cta.bulk_group...
tma_store_arrive() [copy_sm90_tma.hpp:1225]
  ↓ PTX: cp.async.bulk.commit_group
tma_store_wait<0>() [copy_sm90_tma.hpp:1248]
  ↓ PTX: cp.async.bulk.wait_group.read 0
```

---

## Summary of All Four Copies

| Copy | Memory Flow | Key Function | Final PTX | Characteristics |
|------|-------------|--------------|-----------|-----------------|
| **1** | SMEM → RMEM | `copy_aligned()` | `ld.shared.v4.b32` | Vectorized load (128-bit), each thread loads independently |
| **2** | TMEM → RMEM | `copy()` with TMEM atom | `tcgen05.ld.sync.aligned` | Specialized TMEM load, synchronous, per-thread |
| **3** | RMEM → SMEM | `copy_aligned()` | `st.shared.v4.b32` | Vectorized store (128-bit), each thread stores independently |
| **4** | SMEM → GMEM | `copy()` with TMA atom | `cp.async.bulk.tensor.2d` | Async TMA store, single thread issues, requires fence/commit/wait |

### Key Insights

1. **Auto-Vectorization**: `copy_aligned()` automatically vectorizes to 128-bit (4 floats) when possible
2. **Memory Space Detection**: Template dispatch selects appropriate instructions based on memory types
3. **TMEM Specialization**: TMEM loads use dedicated `tcgen05.ld` instructions
4. **TMA Complexity**: TMA stores require fence → copy → commit → wait sequence
5. **Symmetric Paths**: SMEM ↔ RMEM copies use same code path, just reversed direction

### Performance Characteristics

| Copy | Latency | Bandwidth | Concurrency |
|------|---------|-----------|-------------|
| SMEM → RMEM | Low (~20 cycles) | High (vectorized) | All threads |
| TMEM → RMEM | Medium (~50 cycles) | Medium | All threads |
| RMEM → SMEM | Low (~20 cycles) | High (vectorized) | All threads |
| SMEM → GMEM | High (async) | Very High (TMA) | Single thread issues |

### Code Reusability

The generic `copy()` and `copy_aligned()` functions work across all memory spaces:
- **Compile-time dispatch**: Templates select correct implementation
- **Zero overhead**: No runtime polymorphism
- **Type safety**: Compile-time errors for invalid combinations

---

## Complete Epilogue Sequence

Putting it all together, the epilogue for each tile performs:

```cpp
// Load C from GMEM via TMA (done earlier, lines 427-430)
copy(tma_atom_C.with(barrier, 0), tGS_gC(_,tile), tGS_sC);
wait_barrier(barrier);

// ┌────────────────────────────────────────┐
// │ Copy 1: SMEM → RMEM (Line 434)        │
// └────────────────────────────────────────┘
copy_aligned(tTR_sC, tTR_rC);
// PTX: ld.shared.v4.b32 {%r0-%r3}, [smem_C]

// ┌────────────────────────────────────────┐
// │ Copy 2: TMEM → RMEM (Line 437)        │
// └────────────────────────────────────────┘
copy(t2r_copy, tTR_tAcc(_,_,tile), tTR_rD);
// PTX: tcgen05.ld.sync.aligned.32x1x32b.x1.b32 %r, [tmem_acc]

// Compute: D = beta * C + alpha * Acc
axpby(beta, tTR_rC, alpha, tTR_rD);

// ┌────────────────────────────────────────┐
// │ Copy 3: RMEM → SMEM (Line 444)        │
// └────────────────────────────────────────┘
__syncthreads();
copy_aligned(tTR_rD, tTR_sD);
// PTX: st.shared.v4.b32 [smem_D], {%r0-%r3}

// ┌────────────────────────────────────────┐
// │ Copy 4: SMEM → GMEM (Line 450)        │
// └────────────────────────────────────────┘
tma_store_fence();
// PTX: fence.proxy.async.shared::cta
__syncthreads();
if (elect_one_warp && elect_one_thr) {
  copy(tma_atom_D, tSG_sD, tSG_gD(_,tile));
  // PTX: cp.async.bulk.tensor.2d.global.shared::cta.bulk_group...
  tma_store_arrive();
  // PTX: cp.async.bulk.commit_group
  tma_store_wait<0>();
  // PTX: cp.async.bulk.wait_group.read 0
}
__syncthreads();
```

**Total PTX Instructions** (per epilogue tile):
- 1 vectorized SMEM load
- 1 TMEM load (per thread)
- 1 vectorized SMEM store
- 1 fence + 1 TMA store + 1 commit + 1 wait

**Memory Traffic** (per epilogue tile, 128×256 elements):
- GMEM → SMEM (C): 128 KB via TMA
- SMEM → RMEM (C): 128 KB via ld.shared
- TMEM → RMEM (Acc): 128 KB via tcgen05.ld
- RMEM → SMEM (D): 128 KB via st.shared
- SMEM → GMEM (D): 128 KB via TMA

---

This completes the full call path traces for all four epilogue copy operations in Example 05!
