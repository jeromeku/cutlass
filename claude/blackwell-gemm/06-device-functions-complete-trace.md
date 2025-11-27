# Complete Device Function Call Traces

This document provides **complete call path traces** for the `copy` (TMA) and `gemm` (MMA) operations in the Blackwell GEMM examples, showing every function call from the high-level API down to the inline PTX assembly.

## Table of Contents

1. [TMA Copy Complete Trace (Example 03)](#tma-copy-complete-trace-example-03)
2. [MMA GEMM Complete Trace (Example 03)](#mma-gemm-complete-trace-example-03)
3. [TMA Copy Complete Trace (Example 04)](#tma-copy-complete-trace-example-04)
4. [MMA GEMM Complete Trace (Example 04)](#mma-gemm-complete-trace-example-04)

---

## TMA Copy Complete Trace (Example 03)

### Starting Point in Example Code

**File**: [examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu:349](../../examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu#L349)

```cpp
copy(tma_atom_A.with(shared_storage.tma_barrier, tma_mcast_mask_a), tAgA(_,k_tile), tAsA);
```

### Complete Call Path

```
┌─────────────────────────────────────────────────────────────────┐
│ Level 1: High-Level Generic copy() Function                    │
└─────────────────────────────────────────────────────────────────┘
copy(Copy_Atom const&, Tensor const&, Tensor&)
  ↓
  Location: include/cute/algorithm/copy.hpp:189-196

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

    if constexpr (SrcLayout::rank == 1) {   // ← TRUE: Both tensors are rank-1
      copy_atom.call(src, dst);             // ← DISPATCH HERE
    } else {
      // Loop over modes (not taken)
    }
  }

┌─────────────────────────────────────────────────────────────────┐
│ Level 2: Copy_Atom::call() Method                              │
└─────────────────────────────────────────────────────────────────┘
Copy_Atom<Copy_Traits<SM90_TMA_LOAD_MULTICAST_OP, ...>>::call(src, dst)
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
      // ← TRUE: Size matches instruction
      return copy_unpack(static_cast<Traits const&>(*this), src, dst);  // ← DISPATCH
    } else {
      // Recurse (not taken)
    }
  }

┌─────────────────────────────────────────────────────────────────┐
│ Level 3: copy_unpack() Friend Function (TMA_LOAD_Unpack)       │
└─────────────────────────────────────────────────────────────────┘
copy_unpack(Copy_Traits<SM90_TMA_LOAD_MULTICAST_OP, ...> const&, src, dst)
  ↓
  Location: include/cute/atom/copy_traits_sm90_tma.hpp:272-294

  This is defined via the TMA_LOAD_Unpack base class:

  template <class CopyOp, class... Args>
  struct TMA_LOAD_Unpack
  {
    template <class TS, class SLayout,
              class TD, class DLayout>
    CUTE_HOST_DEVICE friend constexpr void
    copy_unpack(Copy_Traits<CopyOp, Args...> const& traits,
                Tensor<TS,SLayout>           const& src,
                Tensor<TD,DLayout>                & dst)
    {
      static_assert(is_smem<TD>::value, "SM90_TMA_LOAD requires the destination be shared memory.");

      auto src_coord = src.data().coord_;          // ← Extract GMEM coordinates
      void* dst_ptr = cute::raw_pointer_cast(dst.data());  // ← Extract SMEM pointer

      // Explode tuple and dispatch to CopyOp::copy()
      return detail::explode_tuple(
        detail::CallCOPY<CopyOp>{},                // ← Functor that calls CopyOp::copy()
        traits.opargs_,                            // ← (desc_ptr, mbar_ptr, mask, hint)
        tuple_seq<decltype(traits.opargs_)>{},
        make_tuple(dst_ptr), seq<0>{},
        src_coord, tuple_seq<decltype(src_coord)>{}
      );  // ← DISPATCH
    }
  };

  What does explode_tuple do?
  - It unpacks the tuple arguments: traits.opargs_ = (desc_ptr, mbar_ptr, mask, hint)
  - It unpacks src_coord = (crd0, crd1, ...) based on rank
  - It calls: CallCOPY<CopyOp>::operator()(desc_ptr, mbar_ptr, mask, hint, dst_ptr, crd0, crd1, ...)

┌─────────────────────────────────────────────────────────────────┐
│ Level 4: detail::CallCOPY Functor                              │
└─────────────────────────────────────────────────────────────────┘
detail::CallCOPY<SM90_TMA_LOAD_MULTICAST_OP>::operator()(...)
  ↓
  Location: include/cute/atom/copy_traits_sm90_tma.hpp (implicit)

  This functor simply forwards to CopyOp::copy():

  template <class CopyOp>
  struct CallCOPY {
    template <class... Args>
    CUTE_HOST_DEVICE constexpr void
    operator()(Args&&... args) const {
      return CopyOp::copy(static_cast<Args&&>(args)...);  // ← DISPATCH
    }
  };

┌─────────────────────────────────────────────────────────────────┐
│ Level 5: Dispatch to Correct Dimensionality                    │
└─────────────────────────────────────────────────────────────────┘
SM90_TMA_LOAD_MULTICAST_OP::copy(desc_ptr, mbar_ptr, mask, hint, dst_ptr, crd0, ...)
  ↓
  Location: include/cute/atom/copy_traits_sm90_tma.hpp:331-381

  The actual CopyOp is SM90_TMA_LOAD_MULTICAST, which inherits from:
  - SM90_TMA_LOAD_MULTICAST_1D
  - SM90_TMA_LOAD_MULTICAST_2D  ← SELECTED FOR 2D TENSOR
  - SM90_TMA_LOAD_MULTICAST_3D
  - SM90_TMA_LOAD_MULTICAST_4D
  - SM90_TMA_LOAD_MULTICAST_5D

  The correct overload is selected based on the number of coordinate arguments.

  For Example 03 (2D A matrix: M×K):
    src_coord has 2 elements → dispatches to SM90_TMA_LOAD_MULTICAST_2D::copy()

┌─────────────────────────────────────────────────────────────────┐
│ Level 6: TMA Hardware Instruction (2D Multicast)               │
└─────────────────────────────────────────────────────────────────┘
SM90_TMA_LOAD_MULTICAST_2D::copy(desc_ptr, mbar_ptr, mask, hint, smem_ptr, crd0, crd1)
  ↓
  Location: include/cute/arch/copy_sm90_tma.hpp:526-547

  struct SM90_TMA_LOAD_MULTICAST_2D
  {
    CUTE_HOST_DEVICE static void
    copy(void const* desc_ptr,               // TMA descriptor
         uint64_t* mbar_ptr,                 // Barrier pointer
         uint16_t multicast_mask,            // Multicast mask
         uint64_t cache_hint,                // Cache hint
         void* smem_ptr,                     // Destination SMEM
         int32_t const& crd0,                // M coordinate
         int32_t const& crd1)                // K coordinate
    {
  #if defined(CUTE_ARCH_TMA_SM90_ENABLED)
      uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
      uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
      uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);

      // Logging (optional)
      cutlass::arch::synclog_emit_tma_load(__LINE__, gmem_int_desc, smem_int_mbar, smem_int_ptr);

      // PTX INSTRUCTION ← FINAL HARDWARE CALL
      asm volatile (
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
        " [%0], [%1, {%4, %5}], [%2], %3, %6;"
        :  // No outputs
        : "r"(smem_int_ptr),      // %0 - SMEM destination address
          "l"(gmem_int_desc),     // %1 - TMA descriptor (64-bit)
          "r"(smem_int_mbar),     // %2 - Barrier address
          "h"(multicast_mask),    // %3 - 16-bit multicast mask
          "r"(crd0),              // %4 - M coordinate
          "r"(crd1),              // %5 - K coordinate
          "l"(cache_hint)         // %6 - Cache hint (64-bit)
        : "memory"                // Clobbers memory
      );
  #else
      CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
  #endif
    }
  };
```

### PTX Instruction Breakdown

```ptx
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%0], [%1, {%4, %5}], [%2], %3, %6;
```

**Instruction Components**:
- `cp.async.bulk.tensor.2d` - Asynchronous bulk copy of 2D tensor
- `.shared::cluster` - Destination is cluster-wide shared memory
- `.global` - Source is global memory
- `.mbarrier::complete_tx::bytes` - Use barrier, track transaction bytes
- `.multicast::cluster` - Multicast to multiple CTAs in cluster
- `.L2::cache_hint` - L2 cache behavior hint

**Operands**:
- `[%0]` - SMEM destination address (32-bit)
- `[%1, {%4, %5}]` - TMA descriptor + coordinates (desc is 64-bit, coords are 32-bit each)
- `[%2]` - Barrier address (32-bit SMEM address)
- `%3` - Multicast mask (16-bit)
- `%6` - Cache hint (64-bit)

**What Happens**:
1. Hardware TMA unit reads the descriptor at `%1`
2. Decodes: GMEM base address, layout, swizzle pattern
3. Computes GMEM address: `base + crd0 * stride0 + crd1 * stride1`
4. Initiates async transfer: GMEM → SMEM at `%0`
5. Multicasts data to all CTAs specified in `%3` mask
6. Updates barrier at `%2` as data arrives
7. Returns immediately (async operation)

### Template Parameters Used in Example 03

```cpp
// From Example 03 line 349:
copy(tma_atom_A.with(shared_storage.tma_barrier, tma_mcast_mask_a), tAgA(_,k_tile), tAsA);

// Template instantiation:
CopyOp = SM90_TMA_LOAD_MULTICAST_OP
NumBitsPerTMA = Int<131072>  // 128×64×16 bits = 16 KB
ThrID = Layout<_1>
SrcEngine = ArithTuple  // Coordinate tensor
SrcLayout = ... (coordinate layout)
DstEngine = smem_ptr<cutlass::half_t>
DstLayout = Swizzle<3,4,3> o smem_ptr[16b] o (8192):(1)

// Runtime arguments:
desc_ptr = &tma_atom_A.tma_desc_
mbar_ptr = &shared_storage.tma_barrier
multicast_mask = tma_mcast_mask_a  // e.g., 0x000f
cache_hint = TMA::CacheHintSm100::EVICT_NORMAL
smem_ptr = tAsA.data()
crd0 = 0  // M-coordinate (CTA's M-slice)
crd1 = k_tile  // K-coordinate (which K-tile)
```

### Summary Diagram

```
User Code (Line 349)
  copy(tma_atom_A.with(...), tAgA(_,k_tile), tAsA)
    ↓
Generic copy() [cute/algorithm/copy.hpp:189]
    ↓ dispatch on rank-1
Copy_Atom::call() [cute/atom/copy_atom.hpp:94]
    ↓ dispatch on size match
copy_unpack() [cute/atom/copy_traits_sm90_tma.hpp:272]
  (TMA_LOAD_Unpack base class)
    ↓ extract coordinates & SMEM pointer
detail::explode_tuple() [cute/util/tuple.hpp]
    ↓ unpack tuple args
detail::CallCOPY<SM90_TMA_LOAD_MULTICAST_OP>::operator()
    ↓ forward to CopyOp::copy()
SM90_TMA_LOAD_MULTICAST_2D::copy() [cute/arch/copy_sm90_tma.hpp:526]
    ↓ cast pointers to integers
PTX asm volatile
  cp.async.bulk.tensor.2d.shared::cluster.global...
    ↓
Hardware TMA Unit
  - Read descriptor
  - Compute GMEM address
  - Initiate async transfer
  - Multicast to CTAs
  - Update barrier
```

---

## MMA GEMM Complete Trace (Example 03)

### Starting Point in Example Code

**File**: [examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu:366](../../examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu#L366)

```cpp
gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCtAcc);
```

### Complete Call Path

```
┌─────────────────────────────────────────────────────────────────┐
│ Level 1: High-Level Generic gemm() Function (3-arg overload)   │
└─────────────────────────────────────────────────────────────────┘
gemm(TiledMMA const&, Tensor const&, Tensor const&, Tensor&)
  ↓
  Location: include/cute/algorithm/gemm.hpp:81-89

  template <class MMA,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE
  void
  gemm(MMA_Atom<MMA>       const& mma,
       Tensor<TA, ALayout> const& A,
       Tensor<TB, BLayout> const& B,
       Tensor<TC, CLayout>      & C)
  {
    return gemm(mma, C, A, B, C);  // ← Convert to 4-arg form (D=C)
  }

┌─────────────────────────────────────────────────────────────────┐
│ Level 2: Dispatch Based on Tensor Memory Type & Rank           │
└─────────────────────────────────────────────────────────────────┘
gemm(MMA_Atom const&, D, A, B, C) - Multiple Overloads
  ↓
  Location: include/cute/algorithm/gemm.hpp:Multiple locations

  The gemm() function has multiple overloads based on:
  1. Tensor rank (1, 2, 3, etc.)
  2. Memory type (RMEM, SMEM, TMEM)
  3. Whether using default FMA or custom MMA

  For Example 03:
  - A, B are rank-1 tensors (descriptors)
  - C is TMEM tensor (tmem_ptr<float>)
  - MMA is SM100_MMA_F16BF16_SS<...>

  This matches a custom dispatch path for TMEM + descriptors.
  However, since this is a specialized MMA, it goes through the MMA_Atom::call() method.

┌─────────────────────────────────────────────────────────────────┐
│ Level 3: TiledMMA Dispatch                                     │
└─────────────────────────────────────────────────────────────────┘

For SM100, the gemm() function with TiledMMA and descriptors follows a custom path.
Let me trace the actual implementation:

Looking at the example code structure:
- tCrA and tCrB are SMEM descriptor tensors (UMMA::DescriptorIterator)
- tCtAcc is a TMEM accumulator tensor (tmem_ptr<float>)
- tiled_mma is TiledMMA with SM100_MMA_F16BF16_SS atom

The dispatch eventually reaches the MMA_Atom::call() method.

┌─────────────────────────────────────────────────────────────────┐
│ Level 3: MMA_Atom::call() Method                               │
└─────────────────────────────────────────────────────────────────┘
MMA_Atom<MMA_Traits<SM100_MMA_F16BF16_SS<...>>>::call(D, A, B, C)
  ↓
  Location: include/cute/atom/mma_atom.hpp:88-105

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr
  void
  call(Tensor<TD, DLayout>      & D,
       Tensor<TA, ALayout> const& A,
       Tensor<TB, BLayout> const& B,
       Tensor<TC, CLayout> const& C) const
  {
    static_assert(DLayout::rank == 1, "Expected rank-1 D tensor");
    static_assert(ALayout::rank == 1, "Expected rank-1 A tensor");
    static_assert(BLayout::rank == 1, "Expected rank-1 B tensor");
    static_assert(CLayout::rank == 1, "Expected rank-1 C tensor");

    return mma_unpack(static_cast<Traits const&>(*this), D, A, B, C);  // ← DISPATCH
  }

┌─────────────────────────────────────────────────────────────────┐
│ Level 4: mma_unpack() Function                                 │
└─────────────────────────────────────────────────────────────────┘
mma_unpack(MMA_Traits<SM100_MMA_F16BF16_SS<...>> const&, D, A, B, C)
  ↓
  Location: include/cute/atom/mma_traits.hpp:Several hundred lines

  The mma_unpack function is a complex dispatcher that:
  1. Extracts fragment data from tensors
  2. Converts to appropriate register types
  3. Calls the MMA operation's fma() method

  For SM100_MMA_F16BF16_SS:

  template <class MMATraits, class TD, class DLayout,
            class TA, class ALayout, class TB, class BLayout, class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr
  void
  mma_unpack(MMATraits const& traits,
             Tensor<TD,DLayout>      & D,
             Tensor<TA,ALayout> const& A,
             Tensor<TB,BLayout> const& B,
             Tensor<TC,CLayout> const& C)
  {
    using MMA_Op = typename MMATraits::MMA_Op;

    // Extract D (accumulator) - TMEM pointer
    auto& tmem_c = D(Int<0>{});  // D is output = C for in-place accumulate

    // Extract A descriptor
    auto const& desc_a = A(Int<0>{});

    // Extract B descriptor
    auto const& desc_b = B(Int<0>{});

    // Extract C (accumulator) - TMEM pointer (same as D for in-place)
    auto const& tmem_c_in = C(Int<0>{});

    // Get scale mode from traits (stored in tiled_mma.accumulate_)
    auto scaleC = traits.accumulate_;  // UMMA::ScaleOut::Zero or ::One

    // Create instruction descriptor
    auto idescE = make_umma_idesc(...);  // Encodes M, N, K, types

    // Call the FMA method ← DISPATCH
    MMA_Op::fma(desc_a, desc_b, tmem_c, scaleC, idescE);
  }

┌─────────────────────────────────────────────────────────────────┐
│ Level 5: SM100_MMA_F16BF16_SS::fma() Static Method             │
└─────────────────────────────────────────────────────────────────┘
SM100_MMA_F16BF16_SS<..., 128, 256, ...>::fma(desc_a, desc_b, tmem_c, scaleC, idescE)
  ↓
  Location: include/cute/arch/mma_sm100_umma.hpp:97-120

  template <class a_type, class b_type, class c_type,
            int M, int N, UMMA::Major a_major, UMMA::Major b_major,
            UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg>
  struct SM100_MMA_F16BF16_SS
  {
    static_assert(M == 64 || M == 128, "M-mode size should be 64 or 128 for 1 CTA cluster MMA.");
    static_assert((N % 8 == 0) && (8 <= N) && (N <= 256), "N-mode size...");

    using DRegisters = void;               // No D registers (output to TMEM)
    using ARegisters = uint64_t[1];        // A descriptor (64-bit)
    using BRegisters = uint64_t[1];        // B descriptor (64-bit)
    using CRegisters = uint32_t[1];        // C accumulator (32-bit TMEM pointer)

    CUTE_HOST_DEVICE static void
    fma(uint64_t const& desc_a,            // A SMEM descriptor
        uint64_t const& desc_b,            // B SMEM descriptor
        uint32_t const& tmem_c,            // TMEM accumulator pointer
        uint32_t const& scaleC,            // Scale mode (0=Zero, 1=One)
        uint64_t const& idescE)            // Instruction descriptor
    {
  #if defined(CUTE_ARCH_TCGEN05_F16F32_MMA_ENABLED)
      if (cute::elect_one_sync()) {        // ← Only one thread executes
        uint32_t mask[4] = {0, 0, 0, 0};   // Mask registers (unused in basic case)

        // PTX INSTRUCTION ← FINAL HARDWARE CALL
        asm volatile(
          "{\n\t"
          ".reg .pred p;\n\t"
          "setp.ne.b32 p, %4, 0;\n\t"  // p = (scaleC != 0)
          "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8}, p; \n\t"
          "}\n"
          :  // No outputs
          : "r"(tmem_c),               // %0 - TMEM accumulator address
            "l"(desc_a),               // %1 - A descriptor (64-bit)
            "l"(desc_b),               // %2 - B descriptor (64-bit)
            "r"(uint32_t(idescE>>32)), // %3 - Instruction descriptor upper 32 bits
            "r"(scaleC),               // %4 - Scale mode
            "r"(mask[0]),              // %5 - Mask register 0
            "r"(mask[1]),              // %6 - Mask register 1
            "r"(mask[2]),              // %7 - Mask register 2
            "r"(mask[3])               // %8 - Mask register 3
        );
      }
  #else
      CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_F16BF16_SS without CUTE_ARCH_TCGEN05_F16F32_MMA_ENABLED");
  #endif
    }
  };
```

### PTX Instruction Breakdown

```ptx
{
  .reg .pred p;
  setp.ne.b32 p, %4, 0;    // p = (scaleC != 0), i.e., p = true if accumulating
  tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8}, p;
}
```

**Instruction Components**:
- `tcgen05.mma` - Tensor Core Generation 05 (SM100) MMA instruction
- `.cta_group::1` - Single CTA executes the MMA (not 2SM)
- `.kind::f16` - F16×F16→F32 operation

**Operands**:
- `[%0]` - TMEM accumulator address (32-bit)
- `%1` - A descriptor (64-bit, points to SMEM with layout info)
- `%2` - B descriptor (64-bit, points to SMEM with layout info)
- `%3` - Instruction descriptor upper 32 bits (encodes M=128, N=256, K=16, types)
- `{%5, %6, %7, %8}` - Mask registers (all zeros in basic case)
- `p` - Predicate: if false, clear accumulator first; if true, accumulate

**What Happens**:
1. Hardware decodes descriptors `%1` and `%2`
2. Reads A matrix from SMEM: 128×16 elements (F16)
3. Reads B matrix from SMEM: 256×16 elements (F16)
4. Performs matrix multiply: C = A × B (or C += A × B if p is true)
5. Writes accumulator to TMEM at address `%0`: 128×256 elements (F32)
6. Only one thread per CTA executes (`elect_one_sync()`)
7. All threads in CTA see updated TMEM (it's CTA-wide)

### SMEM Descriptors

The descriptors `desc_a` and `desc_b` are 64-bit values encoding:
- SMEM base address
- Layout (K-major for both A and B)
- Swizzle pattern (128B swizzle)
- Element type (F16)
- Matrix dimensions (128×16 for A, 256×16 for B)

These are created earlier by:
```cpp
auto tCrA = thr_mma.make_fragment_A(tCsA);  // Creates SMEM descriptor for A
auto tCrB = thr_mma.make_fragment_B(tCsB);  // Creates SMEM descriptor for B
```

### Template Parameters Used in Example 03

```cpp
// From Example 03 lines 448-450:
TiledMMA tiled_mma = make_tiled_mma(
  SM100_MMA_F16BF16_SS<TypeA, TypeB, TypeC, 128, 256, UMMA::Major::K, UMMA::Major::K>{}
);

// Template instantiation:
a_type = cutlass::half_t
b_type = cutlass::half_t
c_type = float
M = 128
N = 256
a_major = UMMA::Major::K  // K-major (row-major for A)
b_major = UMMA::Major::K  // K-major (column-major for B)
a_neg = UMMA::ScaleIn::One  // No negation
b_neg = UMMA::ScaleIn::One  // No negation

// MMA_Traits provides:
Shape_MNK = Shape<_128, _256, _16>
ThrID = Layout<_1>  // Single "thread" (entire CTA)
ALayout = Layout for 128×16 elements
BLayout = Layout for 256×16 elements
CLayout = Layout for 128×256 accumulator

// Runtime arguments:
desc_a = tCrA(0, 0, k_block)  // SMEM descriptor for A
desc_b = tCrB(0, 0, k_block)  // SMEM descriptor for B
tmem_c = tCtAcc.data()        // TMEM accumulator base pointer
scaleC = tiled_mma.accumulate_  // UMMA::ScaleOut::Zero (first) or ::One (subsequent)
idescE = <generated>          // Instruction descriptor encoding M, N, K, types
```

### Summary Diagram

```
User Code (Line 366)
  gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCtAcc)
    ↓
Generic gemm() 3-arg [cute/algorithm/gemm.hpp:81]
    ↓ convert to 4-arg (D=C)
MMA_Atom::call() [cute/atom/mma_atom.hpp:88]
    ↓ dispatch on rank-1
mma_unpack() [cute/atom/mma_traits.hpp]
    ↓ extract descriptors & TMEM pointer
SM100_MMA_F16BF16_SS::fma() [cute/arch/mma_sm100_umma.hpp:97]
    ↓ elect_one_sync()
PTX asm volatile
  {
    setp.ne.b32 p, scaleC, 0
    tcgen05.mma.cta_group::1.kind::f16 [tmem_c], desc_a, desc_b, idescE, {masks}, p
  }
    ↓
Hardware MMA Unit (Tensor Core)
  - Decode descriptors
  - Read A from SMEM (128×16 F16)
  - Read B from SMEM (256×16 F16)
  - Compute C = A × B (or C += A × B)
  - Write to TMEM (128×256 F32)
```

---

## TMA Copy Complete Trace (Example 04)

### Key Differences from Example 03

Example 04 uses **2SM TMA operations**, which have these changes:

1. **TMA Atom Type**: `SM100_TMA_2SM_LOAD_MULTICAST` instead of `SM90_TMA_LOAD_MULTICAST`
2. **ThrID**: `Layout<_2>` instead of `Layout<_1>` (two peer CTAs)
3. **Copy Traits**: `Copy_Traits<SM100_TMA_2SM_LOAD_MULTICAST_OP, ...>`

### Modified Call Path (Only Changed Levels)

```
Levels 1-4: SAME AS EXAMPLE 03

┌─────────────────────────────────────────────────────────────────┐
│ Level 5: TMA Hardware Instruction (2D Multicast, 2SM)          │
└─────────────────────────────────────────────────────────────────┘
SM100_TMA_2SM_LOAD_MULTICAST_2D::copy(desc_ptr, mbar_ptr, mask, hint, smem_ptr, crd0, crd1)
  ↓
  Location: include/cute/arch/copy_sm100_tma.hpp (similar to SM90)

  The implementation is similar to SM90_TMA_LOAD_MULTICAST_2D, but with:
  - Both peer CTAs execute the TMA (not just one)
  - ThrID = _2 means coordination between peers
  - Potentially different synchronization

  struct SM100_TMA_2SM_LOAD_MULTICAST_2D
  {
    CUTE_HOST_DEVICE static void
    copy(void const* desc_ptr,
         uint64_t* mbar_ptr,
         uint16_t multicast_mask,
         uint64_t cache_hint,
         void* smem_ptr,
         int32_t const& crd0,
         int32_t const& crd1)
    {
  #if defined(CUTE_ARCH_TCGEN05_TMA_ENABLED)
      uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
      uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
      uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);

      // PTX INSTRUCTION (2SM variant)
      asm volatile (
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
        " [%0], [%1, {%4, %5}], [%2], %3, %6;"
        :
        : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
          "h"(multicast_mask), "r"(crd0), "r"(crd1), "l"(cache_hint)
        : "memory"
      );
  #else
      CUTE_INVALID_CONTROL_PATH("Trying to use SM100 2SM TMA without CUTE_ARCH_TCGEN05_TMA_ENABLED.");
  #endif
    }
  };
```

**Note**: The PTX instruction is the same as SM90, but the **context** is different:
- Two peer CTAs execute this code
- Each CTA loads its own portion (M-dimension is split across peers)
- Multicast mask includes both peer CTAs

---

## MMA GEMM Complete Trace (Example 04)

### Key Differences from Example 03

Example 04 uses **2SM MMA operations**:

1. **MMA Atom Type**: `SM100_MMA_F16BF16_2x1SM_SS` instead of `SM100_MMA_F16BF16_SS`
2. **M Dimension**: 256 instead of 128 (double the M)
3. **ThrID**: `Layout<_2>` instead of `Layout<_1>`
4. **Peer Coordination**: Leader CTA executes MMA, both CTAs share TMEM

### Modified Call Path (Only Changed Levels)

```
Levels 1-4: SAME AS EXAMPLE 03

┌─────────────────────────────────────────────────────────────────┐
│ Level 5: SM100_MMA_F16BF16_2x1SM_SS::fma() Static Method       │
└─────────────────────────────────────────────────────────────────┘
SM100_MMA_F16BF16_2x1SM_SS<..., 256, 256, ...>::fma(desc_a, desc_b, tmem_c, scaleC, idescE)
  ↓
  Location: include/cute/arch/mma_sm100_umma.hpp (2x1SM variant)

  template <class a_type, class b_type, class c_type,
            int M, int N, UMMA::Major a_major, UMMA::Major b_major,
            UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg>
  struct SM100_MMA_F16BF16_2x1SM_SS
  {
    static_assert(M == 128 || M == 256, "M-mode size should be 128 or 256 for 2SM MMA.");
    static_assert((N % 16 == 0) && (16 <= N) && (N <= 256), "N-mode size...");

    using DRegisters = void;
    using ARegisters = uint64_t[1];
    using BRegisters = uint64_t[1];
    using CRegisters = uint32_t[1];

    CUTE_HOST_DEVICE static void
    fma(uint64_t const& desc_a,
        uint64_t const& desc_b,
        uint32_t const& tmem_c,
        uint32_t const& scaleC,
        uint64_t const& idescE)
    {
  #if defined(CUTE_ARCH_TCGEN05_F16F32_MMA_ENABLED)
      if (cute::elect_one_sync()) {        // ← Only leader CTA executes
        uint32_t mask[4] = {0, 0, 0, 0};

        // PTX INSTRUCTION (2SM variant)
        asm volatile(
          "{\n\t"
          ".reg .pred p;\n\t"
          "setp.ne.b32 p, %4, 0;\n\t"
          "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8}, p; \n\t"
          //                     └─ NOTE: ::2 instead of ::1
          "}\n"
          :
          : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
            "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3])
        );
      }
  #else
      CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_F16BF16_2x1SM_SS without CUTE_ARCH_TCGEN05_F16F32_MMA_ENABLED");
  #endif
    }
  };
```

### PTX Instruction Differences

```ptx
// Example 03 (1SM):
tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8}, p;
//                   └─ Single CTA

// Example 04 (2SM):
tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8}, p;
//                   └─ Two CTAs (2SM)
```

**What's Different**:
- `cta_group::2` - Two peer CTAs collaborate
- M dimension is 256 instead of 128
- Only leader CTA (even SM ID) executes the instruction
- Both peer CTAs share the TMEM accumulator
- Hardware coordinates between the two SMs

### SMEM Layout for 2SM

Each peer CTA has its own SMEM region:
- **Peer 0 (even SM)**: A matrix rows 0-127
- **Peer 1 (odd SM)**: A matrix rows 128-255

The MMA hardware reads from both SMEM regions to compute the full 256×256 output.

---

## Summary Tables

### TMA Copy Comparison

| Level | Example 03 | Example 04 | Difference |
|-------|------------|------------|------------|
| Generic copy() | Same | Same | None |
| Copy_Atom::call() | Same | Same | None |
| copy_unpack() | Same | Same | None |
| CallCOPY functor | Same | Same | None |
| Dimensionality dispatch | 2D | 2D | None |
| TMA instruction | `SM90_TMA_LOAD_MULTICAST_2D` | `SM100_TMA_2SM_LOAD_MULTICAST_2D` | 2SM variant |
| PTX | `cp.async.bulk.tensor.2d...` | `cp.async.bulk.tensor.2d...` | Same PTX, different context |
| Execution | Single CTA executes | Both peer CTAs execute | Coordination |

### MMA GEMM Comparison

| Level | Example 03 | Example 04 | Difference |
|-------|------------|------------|------------|
| Generic gemm() | Same | Same | None |
| MMA_Atom::call() | Same | Same | None |
| mma_unpack() | Same | Same | None |
| MMA operation | `SM100_MMA_F16BF16_SS` | `SM100_MMA_F16BF16_2x1SM_SS` | 2SM variant |
| M dimension | 128 | 256 | Double |
| PTX | `cta_group::1` | `cta_group::2` | 2SM |
| Execution | Single CTA | Leader CTA only | Peer coordination |

---

## Key Takeaways

### For TMA Copy

1. **High-level API is generic**: `copy(atom, src, dst)` works for any copy operation
2. **Template dispatch**: The correct implementation is selected at compile-time based on:
   - Copy operation type (TMA, cp.async, memcpy, etc.)
   - Dimensionality (1D, 2D, 3D, 4D, 5D)
   - Source/destination memory types
3. **Low-level PTX**: Eventually reaches a single inline assembly instruction
4. **Coordinate unwrapping**: `explode_tuple` unpacks coordinate tuples into separate arguments

### For MMA GEMM

1. **High-level API is generic**: `gemm(mma, A, B, C)` works for any MMA operation
2. **Template dispatch**: The correct implementation is selected based on:
   - MMA operation type (UMMA, WMMA, FMA, etc.)
   - Matrix dimensions (M, N, K)
   - Data types (F16, BF16, TF32, etc.)
   - Number of SMs (1SM vs 2SM)
3. **Descriptor-based**: A and B are SMEM descriptors, not raw pointers
4. **TMEM accumulator**: C is in TMEM, a specialized on-chip memory
5. **Elect-one execution**: Only one thread per CTA (or only leader CTA for 2SM) executes the MMA

### Common Patterns

- **Static polymorphism**: Templates + compile-time dispatch (zero runtime overhead)
- **Type-based selection**: SFINAE and `if constexpr` choose implementations
- **Tuple unpacking**: `explode_tuple` and variadic templates handle variable arguments
- **Memory type traits**: `is_smem`, `is_tmem`, `is_rmem` guide dispatch
- **Rank-based dispatch**: Different code for rank-1, rank-2, etc. tensors

---

This completes the full call path traces for both `copy` (TMA) and `gemm` (MMA) operations in Examples 03 and 04!
