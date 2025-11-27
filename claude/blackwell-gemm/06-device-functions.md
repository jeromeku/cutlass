# Device Function Deep Dive

This document provides detailed analysis of key device-side functions used in the Blackwell GEMM examples, tracing through template specializations and showing the actual PTX instructions generated.

## Table of Contents

1. [tma_partition](#tma_partition)
2. [create_tma_multicast_mask](#create_tma_multicast_mask)
3. [copy (TMA)](#copy-tma)
4. [gemm (MMA)](#gemm-mma)
5. [make_tmem_copy](#make_tmem_copy)
6. [partition_S / partition_D](#partition_s--partition_d)
7. [get_slice](#get_slice)
8. [tmem_allocator.allocate](#tmem_allocatorallocate)
9. [umma_arrive_multicast_2x1SM](#umma_arrive_multicast_2x1sm)

---

## tma_partition

### Purpose

Partitions TMA operations across CTAs in a cluster, accounting for multicast patterns. It computes which portion of GMEM and SMEM each CTA should access.

### Function Signature

```cpp
template <class... Args,
          class CtaCoord, class TShape, class TStride,
          class SEngine, class SLayout,
          class GEngine, class GLayout>
CUTE_DEVICE
auto
tma_partition(Copy_Atom<Args...>      const& copy_atom,
              CtaCoord                const& cta_coord,     // CTA coordinate in cluster
              Layout<TShape,TStride>  const& cta_layout,    // CTA layout for multicast
              Tensor<SEngine,SLayout> const& stensor,       // SMEM tensor
              Tensor<GEngine,GLayout> const& gtensor)       // GMEM tensor (coordinate)
```

**Location**: [include/cute/atom/copy_traits_sm90_tma.hpp:1387-1420](../../include/cute/atom/copy_traits_sm90_tma.hpp#L1387-L1420)

### Usage in Example 03

**Line**: [examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu:288-291](../../examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu#L288-L291)

```cpp
auto [tAgA, tAsA] = tma_partition(tma_atom_A,
                                  get<2>(cta_in_cluster_coord_vmnk),          // CTA N-coord
                                  make_layout(size<2>(cluster_layout_vmnk)),  // CTA N-layout
                                  group_modes<0,3>(tCsA),
                                  group_modes<0,3>(tCgA));
```

### Step-by-Step Trace for Example 03

#### Input Parameters

```cpp
copy_atom = tma_atom_A  // Copy_Atom<SM90_TMA_LOAD_MULTICAST, ...>

cta_coord = get<2>(cta_in_cluster_coord_vmnk)  // CTA's N-coordinate in cluster (0, 1, 2, or 3)

cta_layout = make_layout(size<2>(cluster_layout_vmnk))  // Layout<_4, _1> (4 CTAs in N-mode)

stensor = group_modes<0,3>(tCsA)
  // Original tCsA: ((_128,_16), _1, _4):((_64,_1), _0, _16)
  // After group_modes<0,3>: ((_128,_16,_1,_4)):((_64,_1,_0,_16))
  // Shape: 8192 elements (128×16×1×4)

gtensor = group_modes<0,3>(tCgA)
  // Original tCgA: ((_128,_16), _1, _4, 4):((_1@1,_1@0), _0, _16@0, _64@0)
  // After group_modes<0,3>: ((8192), 4)
  // Coordinate tensor with K-tiles
```

#### Algorithm Walkthrough

**Step 1: Invert SMEM Layout to Find Contiguous Vector**

```cpp
// Line 1396
Layout inv_smem_layout = right_inverse(get_nonswizzle_portion(layout<0>(stensor)));
```

This computes the "natural" access pattern - the ordering that creates the most contiguous access.

```cpp
// layout<0>(stensor) = (8192):(1)  (after flattening)
// get_nonswizzle_portion removes Swizzle<3,4,3>
// right_inverse finds the inverse mapping
// Result: Layout that maps value index -> memory coordinate
```

**Step 2: Scale to Cover All SMEM**

```cpp
// Line 1398
Layout layout_v = tile_to_shape(make_layout(inv_smem_layout), size<0>(stensor));
```

Tiles the inverted layout to cover the entire SMEM tensor.

```cpp
// size<0>(stensor) = 8192
// layout_v describes how to access all 8192 elements efficiently
```

**Step 3: Factor Out Single TMA Instruction**

```cpp
// Line 1401-1402
Layout tma_layout_v = make_layout(Int<Copy_Atom<Args...>::NumValSrc>{});
auto layout_V = make_tile(logical_divide(layout_v, tma_layout_v));
```

Divides the layout into chunks that fit in a single TMA instruction.

```cpp
// NumValSrc = 8192 (the TMA box size: 128×64 elements)
// layout_V groups elements into TMA-sized chunks
```

**Step 4: Compute Multicast Offset**

```cpp
// Line 1411
auto multicast_offset = cta_layout(cta_coord) * (size(tma_layout_v) / cosize(cta_layout));
```

**Example for CTA at N-coord = 2**:
```cpp
cta_layout(2) = 2         // 3rd CTA in layout
size(tma_layout_v) = 8192
cosize(cta_layout) = 4    // 4 CTAs total
multicast_offset = 2 * (8192 / 4) = 2 * 2048 = 4096
```

**Interpretation**: This CTA starts at offset 4096 within the TMA-tiled space.

**Step 5: Apply Offset**

```cpp
// Lines 1412-1417
auto multicast_coord = make_coord(make_coord(multicast_offset, Int<0>{}));
auto gcoord = append<GLayout::rank>(multicast_coord, Int<0>{});
auto scoord = append<SLayout::rank>(multicast_coord, Int<0>{});

Tensor gresult = domain_offset(gcoord, gtensor_v);
Tensor sresult = domain_offset(scoord, stensor_v);
```

Creates coordinate offsets and applies them to both GMEM and SMEM tensors.

#### Return Value

```cpp
cute::tuple<
  Tensor<...>,  // tAgA - GMEM portion for this CTA
  Tensor<...>   // tAsA - SMEM portion for this CTA
>
```

**For CTA at (V=0, M=0, N=2)**:
- `tAgA`: Points to GMEM coordinates for this CTA's slice
- `tAsA`: Points to SMEM starting at offset computed for N=2

### Visual Representation

```
Cluster (4 CTAs in N-dimension):

GMEM (512×256):                    SMEM (per CTA, 128×64):
┌────────────────────────────┐     ┌─────────┐
│ CTA N=0                    │────>│ SMEM 0  │
├────────────────────────────┤     ├─────────┤
│ CTA N=1                    │────>│ SMEM 1  │
├────────────────────────────┤     ├─────────┤
│ CTA N=2   ← tma_partition  │────>│ SMEM 2  │ ← Result for N=2
├────────────────────────────┤     ├─────────┤
│ CTA N=3                    │────>│ SMEM 3  │
└────────────────────────────┘     └─────────┘

Each CTA:
- Gets its own slice of GMEM coordinates (tAgA)
- Gets its own SMEM region (tAsA)
- Multicast: All 4 CTAs broadcast to each other
```

### Example 04 Differences

In Example 04, `tma_partition` is called identically, but the internal behavior accounts for 2SM:

```cpp
// Example 04: Line 283-286
auto [tAgA, tAsA] = tma_partition(tma_atom_A,
                                  get<2>(cta_in_cluster_coord_vmnk),
                                  make_layout(size<2>(cluster_layout_vmnk)),
                                  group_modes<0,3>(tCsA),
                                  group_modes<0,3>(tCgA));
```

**Key difference**: `tma_atom_A` has `ThrID = Layout<_2>`, so the function knows to account for peer CTAs.

---

## create_tma_multicast_mask

### Purpose

Computes a 16-bit bitmask that specifies which CTAs in a cluster should receive a multicasted TMA load.

### Function Signature

```cpp
template <int... Modes, class CtaLayout, class CtaCoord>
CUTE_HOST_DEVICE constexpr
uint16_t
create_tma_multicast_mask(CtaLayout const& cta_layout_vmnk,
                          CtaCoord  const& cta_coord_vmnk)
```

**Location**: [include/cute/atom/copy_traits_sm90_tma.hpp:1475-1479](../../include/cute/atom/copy_traits_sm90_tma.hpp#L1475-L1479)

### Usage in Example 03

**Line**: [examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu:300-305](../../examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu#L300-L305)

```cpp
// Project the cluster_layout and cta_coord along the N-mode to determine the multicast mask for A
uint16_t tma_mcast_mask_a = create_tma_multicast_mask<2>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk);

// Project the cluster_layout and cta_coord along the M-mode to determine the multicast mask for B
uint16_t tma_mcast_mask_b = create_tma_multicast_mask<1>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk);

// Project the cluster_layout and cta_coord along the VM + VN-modes to determine the multicast mask for C
uint16_t mma_mcast_mask_c = create_tma_multicast_mask<0,1>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk) |
                            create_tma_multicast_mask<0,2>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk);
```

### Step-by-Step Trace for Example 03

#### Input for tma_mcast_mask_a

```cpp
Modes = <2>  // Project along N-mode (keep all CTAs with same N-coordinate)

cluster_layout_vmnk = Layout<(2, 2, 4, 1), (8, 4, 1, 0)>
  // V=2 (peer CTAs), M=2, N=4, K=1
  // Strides: V-stride=8, M-stride=4, N-stride=1, K-stride=0

cta_in_cluster_coord_vmnk = (0, 1, 2, 0)  // Example: V=0, M=1, N=2, K=0
```

#### Implementation Detail

The template parameters `<2>` mean "replace mode 2 (N) with wildcard":

```cpp
// Line 1478
return create_tma_multicast_mask<Modes...>(cta_layout_vmnk, replace<2>(cta_coord_vmnk, _));
```

This becomes:
```cpp
create_tma_multicast_mask(cta_layout_vmnk, (0, 1, _, 0))
//                                              └─ Wildcard means "all values"
```

#### Algorithm Walkthrough

**Step 1: Slice and Offset** (Line 1442)

```cpp
auto [cta_layout, elected_cta] = slice_and_offset((0, 1, _, 0), cluster_layout_vmnk);
```

**Slicing** extracts the sub-layout for mode 2 (N-mode):
```cpp
// Keep only N-mode, setting V=0, M=1, K=0
cta_layout = Layout<_4, _1>  // 4 CTAs in N-mode, stride-1
elected_cta = 10              // Flat index for V=0, M=1, N=0, K=0 = 0*8 + 1*4 + 0*1 + 0*0 = 4
                             // Wait, let me recalculate: V=0, M=1: 0*8 + 1*4 = 4
                             // This is the base CTA index for this (V,M) pair
```

**Step 2: Build Mask** (Lines 1445-1465)

For the rank-1 optimized path (Line 1449):
```cpp
// shape<0>(cta_layout) = 4
// stride<0>(cta_layout) = 1

mcast_mask = uint16_t(1);          // 0b0000000000000001

// Smear by stride (stride=1):
mcast_mask |= mcast_mask << 1;     // 0b0000000000000011
mcast_mask |= mcast_mask << 2;     // 0b0000000000001111
mcast_mask |= mcast_mask << 4;     // 0b0000000011111111
mcast_mask |= mcast_mask << 8;     // 0b1111111111111111

// Select shape (4 CTAs, stride 1):
mcast_mask &= (uint16_t(-1) >> (16 - 4 * 1));  // Keep lower 4 bits
// Result: 0b0000000000001111
```

**Step 3: Shift by Elected CTA** (Line 1467)

```cpp
mcast_mask <<= elected_cta;  // elected_cta = 4 (from V=0, M=1)
// mcast_mask = 0b0000000011110000
```

#### Final Masks for Example CTA at (V=0, M=1, N=2, K=0)

```cpp
tma_mcast_mask_a = 0b0000000011110000  // Bits 4,5,6,7 (CTAs at M=1, all N)
tma_mcast_mask_b = 0b0000000000110011  // Bits 0,1,4,5 (CTAs at N=2, all M, both V)
```

**Interpretation**:
- **A matrix multicast**: All CTAs with same M-coordinate (M=1) receive the data
- **B matrix multicast**: All CTAs with same N-coordinate (N=2) receive the data

### Visual Representation

```
Cluster Layout (V=0 slice, 4×4 grid of CTAs):
    N=0  N=1  N=2  N=3
M=0  0    1    2    3
M=1  4    5    6    7   ← tma_mcast_mask_a targets these CTAs
         ↑
         └─ CTA (V=0, M=1, N=2) is here

A multicast pattern (for CTA at M=1):
    N=0  N=1  N=2  N=3
M=0  .    .    .    .
M=1  ✓    ✓    ✓    ✓   ← All N-coords, same M
         └─ This CTA

B multicast pattern (for CTA at N=2):
    N=0  N=1  N=2  N=3
M=0  .    .    ✓    .
M=1  .    .    ✓    .   ← All M-coords, same N
              └─ This CTA

Bitmask encoding (16 bits for 16 CTAs max):
Bit position = CTA flat index
Bit value = 1 if CTA receives multicast, 0 otherwise
```

### Example 04 Differences

Example 04 uses identical logic, but with 2SM considerations:

```cpp
// Example 04: Lines 295-300
uint16_t tma_mcast_mask_a = create_tma_multicast_mask<2>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk);
uint16_t tma_mcast_mask_b = create_tma_multicast_mask<1>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk);
uint16_t mma_mcast_mask_c = create_tma_multicast_mask<0,1>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk) |
                            create_tma_multicast_mask<0,2>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk);
```

The `cluster_layout_vmnk` includes V=2 for peer CTAs, so masks account for both peers.

---

## copy (TMA)

### Purpose

Executes an asynchronous TMA copy operation, transferring data from GMEM to SMEM with hardware acceleration.

### Function Signature

```cpp
template <class CopyOp, class... Args,
          class TS, class SLayout,
          class TD, class DLayout>
CUTE_HOST_DEVICE friend constexpr void
copy_unpack(Copy_Traits<CopyOp, Args...> const& traits,
            Tensor<TS,SLayout>           const& src,
            Tensor<TD,DLayout>                & dst)
```

**Location**: Various specializations in copy traits files

### Usage in Example 03

**Line**: [examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu:349-350](../../examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu#L349-L350)

```cpp
copy(tma_atom_A.with(shared_storage.tma_barrier, tma_mcast_mask_a), tAgA(_,k_tile), tAsA);
copy(tma_atom_B.with(shared_storage.tma_barrier, tma_mcast_mask_b), tBgB(_,k_tile), tBsB);
```

### Step-by-Step Trace for Example 03

#### Input Parameters

```cpp
// For A matrix copy:
copy_atom = tma_atom_A.with(shared_storage.tma_barrier, tma_mcast_mask_a)
  // Returns Copy_Atom<SM90_TMA_LOAD_MULTICAST_OP, ...> with:
  // - TMA descriptor
  // - Barrier pointer
  // - Multicast mask

src = tAgA(_,k_tile)
  // Coordinate tensor slice for k_tile
  // Shape: (((_64,_128),_1), 1)  (single K-tile worth of coordinates)

dst = tAsA
  // SMEM tensor
  // Shape: ((_8192,_1)):((_1,_0))
```

#### Generated PTX (Simplified)

The `copy_unpack` function ultimately generates PTX like:

```ptx
// From SM90_TMA_LOAD_MULTICAST implementation
{
  .reg .b64 desc;
  .reg .b32 smem_addr;
  .reg .b64 mbar;
  .reg .b16 mask;

  // Load TMA descriptor address
  ld.param.u64 desc, [tma_desc_ptr];

  // Compute SMEM address
  cvta.to.shared.u32 smem_addr, dst_ptr;

  // Load barrier and mask
  ld.param.u64 mbar, [barrier_ptr];
  mov.b16 mask, multicast_mask;

  // Execute TMA load with multicast
  cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster
    [smem_addr], [desc, {coord_m, coord_k}], [mbar], mask;
}
```

**Key Points**:
- `.shared::cluster` - SMEM is cluster-wide (visible to all CTAs)
- `.mbarrier::complete_tx::bytes` - Barrier tracks transaction bytes
- `.multicast::cluster` - Broadcasts to multiple CTAs
- `{coord_m, coord_k}` - GMEM coordinates from coordinate tensor

#### Execution Flow

**Step 1: Elect Single Thread**

Only one thread per CTA issues the TMA:
```cpp
if (elect_one_warp && elect_one_thr) {
  // TMA execution
}
```

**Step 2: Set Barrier Transaction Bytes**

```cpp
cute::set_barrier_transaction_bytes(shared_storage.tma_barrier, tma_transaction_bytes);
```

Tells the barrier how many bytes to expect. Barrier auto-decrements as data arrives.

**Step 3: Issue TMA**

```cpp
copy(...);  // Executes PTX instruction above
```

**Async**: Returns immediately, data arrives later.

**Step 4: Wait for Completion**

```cpp
cute::wait_barrier(shared_storage.tma_barrier, tma_barrier_phase_bit);
```

Blocks until all TMA bytes have arrived.

### Visual Representation

```
TMA Copy Flow:

Thread 0 in CTA:                         Hardware TMA Unit:
┌──────────────┐                         ┌─────────────────┐
│ elect_one    │                         │                 │
│ set_barrier  │────────────────────────>│ Set expect bytes│
│ copy(...)    │────TMA descriptor──────>│ Decode layout   │
│              │    + coordinates         │ Compute address │
│              │    + SMEM ptr            │ Initiate load   │
│              │                          │ Multicast to    │
│              │                          │ CTAs in mask    │
└──────────────┘                         └─────────────────┘
                                                  │
                                                  v
All Threads in CTA:                        SMEM (cluster):
┌──────────────┐                         ┌─────────────────┐
│ wait_barrier │<───────────────────────│ Data arrives    │
│ ...          │       Data ready        │ Barrier updates │
│ use data     │                         │ Decrement count │
└──────────────┘                         └─────────────────┘
```

### Example 04 Differences

**Line**: [examples/cute/tutorial/blackwell/04_mma_tma_2sm_sm100.cu:349-350](../../examples/cute/tutorial/blackwell/04_mma_tma_2sm_sm100.cu#L349-L350)

```cpp
copy(tma_atom_A.with(shared_storage.tma_barrier, tma_mcast_mask_a), tAgA(_,k_tile), tAsA);
copy(tma_atom_B.with(shared_storage.tma_barrier, tma_mcast_mask_b), tBgB(_,k_tile), tBsB);
```

**Key differences**:
- Uses `SM100_TMA_2SM_LOAD_MULTICAST` operation
- Both peer CTAs issue the TMA (not just one)
- Transaction bytes account for both CTAs: `size<0>(cluster_layout_vmnk) * sizeof(...)`
- Only leader CTA waits on barrier

---

## gemm (MMA)

### Purpose

Executes the matrix multiply-accumulate operation using hardware MMA instructions (tcgen05.mma on SM100).

### Function Signature

```cpp
template <class TiledMMA,
          class EngineA, class LayoutA,
          class EngineB, class LayoutB,
          class EngineC, class LayoutC>
CUTE_HOST_DEVICE
void
gemm(TiledMMA          const& mma,
     Tensor<EngineA, LayoutA> const& A,
     Tensor<EngineB, LayoutB> const& B,
     Tensor<EngineC, LayoutC>      & C)
```

**Location**: [include/cute/algorithm/gemm.hpp](../../include/cute/algorithm/gemm.hpp)

### Usage in Example 03

**Line**: [examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu:366](../../examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu#L366)

```cpp
gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCtAcc);
```

### Step-by-Step Trace for Example 03

#### Input Parameters

```cpp
mma = tiled_mma
  // TiledMMA<..., MMA_Atom<SM100_MMA_F16BF16_SS<..., 128, 256, ...>>>
  // accumulate_ = UMMA::ScaleOut::Zero (first iteration) or ::One (subsequent)

A = tCrA(_,_,k_block)
  // SMEM descriptor for A matrix
  // Type: Tensor<UMMA::DescriptorIterator, ...>
  // Shape: (_1, _1)  (single MMA worth)

B = tCrB(_,_,k_block)
  // SMEM descriptor for B matrix
  // Type: Tensor<UMMA::DescriptorIterator, ...>
  // Shape: (_1, _1)

C = tCtAcc
  // TMEM accumulator
  // Type: Tensor<tmem_ptr<float>, ((_128, _256), _1, _1)>
  // Shape: 128×256 accumulator
```

#### Implementation Path

The `gemm` function calls `mma.call()` which dispatches to:

**Location**: [include/cute/atom/mma_atom.hpp:94-105](../../include/cute/atom/mma_atom.hpp#L94-L105)

```cpp
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
  return mma_unpack(static_cast<Traits const&>(*this), D, A, B, C);
}
```

Which calls `mma_unpack`, which extracts the descriptors and calls:

**Location**: [include/cute/arch/mma_sm100_umma.hpp:97-120](../../include/cute/arch/mma_sm100_umma.hpp#L97-L120)

```cpp
CUTE_HOST_DEVICE static void
fma(uint64_t const& desc_a,       // A SMEM descriptor
    uint64_t const& desc_b,       // B SMEM descriptor
    uint32_t const& tmem_c,       // TMEM accumulator pointer
    uint32_t const& scaleC,       // Scale mode (Zero or One)
    uint64_t const& idescE)       // Instruction descriptor
{
#if defined(CUTE_ARCH_TCGEN05_F16F32_MMA_ENABLED)
  if (cute::elect_one_sync()) {
    uint32_t mask[4] = {0, 0, 0, 0};
    asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8}, p; \n\t"
      "}\n"
      :
      : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
        "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
  }
#endif
}
```

#### Generated PTX

```ptx
{
  .reg .pred p;
  .reg .b64 desc_a, desc_b;
  .reg .b32 tmem_c, scale_c;
  .reg .b32 idesc_e;
  .reg .b32 mask0, mask1, mask2, mask3;

  // Load descriptors
  ld.param.u64 desc_a, [A_descriptor];
  ld.param.u64 desc_b, [B_descriptor];
  ld.param.u32 tmem_c, [C_tmem_ptr];
  ld.param.u32 scale_c, [accumulate_mode];  // 0=Zero, 1=One
  ld.param.u32 idesc_e, [instruction_descriptor + 4];

  // Masks (all zeros for basic case)
  mov.b32 mask0, 0;
  mov.b32 mask1, 0;
  mov.b32 mask2, 0;
  mov.b32 mask3, 0;

  // Set predicate based on scale mode
  setp.ne.b32 p, scale_c, 0;  // p = (scale_c != 0)

  // Execute MMA
  tcgen05.mma.cta_group::1.kind::f16 [tmem_c], desc_a, desc_b, idesc_e, {mask0, mask1, mask2, mask3}, p;
  // Reads A from SMEM via desc_a
  // Reads B from SMEM via desc_b
  // Writes accumulator to TMEM at tmem_c
  // If p is false (scale_c==0): C = A×B (clear accumulator first)
  // If p is true (scale_c==1):  C += A×B (accumulate)
}
```

**Instruction breakdown**:
- `cta_group::1` - Single CTA executes MMA
- `kind::f16` - F16×F16→F32 operation
- `[tmem_c]` - Write accumulator to TMEM
- `desc_a`, `desc_b` - SMEM descriptors encode layout/address
- `idesc_e` - Instruction descriptor (encodes M, N, K, types)
- `{mask0...mask3}` - Mask registers (unused in basic case)
- `p` - Predicate controls accumulation mode

### Visual Representation

```
MMA Execution Flow:

SMEM:                           Hardware MMA Unit:              TMEM:
┌──────────────┐               ┌────────────────┐              ┌──────────────┐
│ A (128×16)   │─desc_a───────>│                │              │              │
│ [Swizzled]   │               │ Read A via desc│              │ Accumulator  │
└──────────────┘               │ Read B via desc│              │ (128×256)    │
                               │ Compute:       │              │              │
┌──────────────┐               │ C = A × B      │              │              │
│ B (256×16)   │─desc_b───────>│ (or C += A×B)  │──result─────>│ C += result  │
│ [Swizzled]   │               │                │              │              │
└──────────────┘               └────────────────┘              └──────────────┘
                                      ↑
                                  idesc_e
                              (M=128, N=256, K=16,
                               A=f16, B=f16, C=f32)

Threads:
- Only one thread executes the MMA (elect_one_sync)
- All threads see updated TMEM (it's CTA-wide)
```

### Example 04 Differences

**Line**: [examples/cute/tutorial/blackwell/04_mma_tma_2sm_sm100.cu:367](../../examples/cute/tutorial/blackwell/04_mma_tma_2sm_sm100.cu#L367)

```cpp
gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCtAcc);
```

**Key differences**:
- Uses `SM100_MMA_F16BF16_2x1SM_SS` (M=256 instead of 128)
- PTX instruction: `tcgen05.mma.cta_group::2.kind::f16` (note `::2` instead of `::1`)
- Only leader CTA executes the MMA
- Both peer CTAs share the TMEM accumulator

---

## make_tmem_copy

### Purpose

Creates a tiled copy operation for transferring data from TMEM (tensor memory) to registers.

### Function Signature

```cpp
template <class CopyOp,
          class TEngine, class TLayout>
CUTE_HOST_DEVICE constexpr
auto
make_tmem_copy(CopyOp const&,
               Tensor<TEngine,TLayout> const& tmem)
```

**Location**: [include/cute/atom/copy_traits_sm100.hpp:309-313](../../include/cute/atom/copy_traits_sm100.hpp#L309-L313)

### Usage in Example 03

**Line**: [examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu:380](../../examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu#L380)

```cpp
TiledCopy tiled_t2r_copy = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
```

### Step-by-Step Trace for Example 03

#### Input Parameters

```cpp
CopyOp = SM100_TMEM_LOAD_32dp32b1x{}
  // TMEM load operation: 32 data path, 32-bit, 1× throughput

tmem = tCtAcc
  // TMEM accumulator tensor
  // Type: Tensor<tmem_ptr<float>, ((_128, _256), _1, _1)>
  // Shape: 128×256 float accumulator
```

#### Implementation

```cpp
// Line 312
return make_tmem_copy(Copy_Atom<CopyOp, typename TEngine::value_type>{}, tmem);
```

Calls the full implementation:

```cpp
// Lines 288-303
template <class CopyOp, class CopyT,
          class TEngine, class TLayout>
CUTE_HOST_DEVICE constexpr
auto
make_tmem_copy(Copy_Atom<CopyOp,CopyT> const& atom,
               Tensor<TEngine,TLayout> const& tmem)
{
  static_assert(is_tmem<TEngine>::value, "Expected TMEM tensor.");
  using T = typename TEngine::value_type;

  // atom thr idx -> tmem addr    4 warps where each warp points to the same position
  auto atom_t_layout = Layout<Shape<_32,_4>, Stride<_0, decltype(Int<32>{} * TMEM::DP<T>{})>>{};

  // atom val idx -> tmem addr    Cast the CopyOp's value ids to the proper data width
  auto atom_v_layout = coalesce(upcast<sizeof_bits<T>::value>(typename Traits::ValID{}));

  return make_cotiled_copy(atom, make_layout(atom_t_layout, atom_v_layout), tmem.layout());
}
```

**Key points**:

**atom_t_layout**: Maps thread index to TMEM address
```cpp
Layout<Shape<_32,_4>, Stride<_0, decltype(Int<32>{} * TMEM::DP<T>{})>>
  // 32 threads per warp, 4 warps
  // Stride 0 in warp: All threads in a warp access same base address
  // Stride 32*DP between warps: DP = data path width
```

**atom_v_layout**: Maps value index to TMEM offset

**Return type**: `TiledCopy` that partitions TMEM → RMEM transfers across threads

#### Usage Pattern

```cpp
// Line 381
ThrCopy thr_t2r_copy = tiled_t2r_copy.get_slice(threadIdx.x);

// Line 388-391
Tensor tDtAcc = thr_t2r_copy.partition_S(tCtAcc);  // TMEM source
Tensor tDrAcc = make_tensor<AccType>(shape(tDgD)); // RMEM destination
copy(tiled_t2r_copy, tDtAcc, tDrAcc);              // Execute TMEM→RMEM
```

### Visual Representation

```
TMEM Layout (128×256):

Warp partitioning (4 warps × 32 threads):
         0─────────────────────────31  ← Warp 0 (threads 0-31)
TMEM  ┌──────────────────────────────┐
128×  │ ████████████████████████████ │
256   │ ████████████████████████████ │ Each warp loads a slice
      │ ████████████████████████████ │
      ├──────────────────────────────┤
      │ ████████████████████████████ │ ← Warp 1 (threads 32-63)
      ├──────────────────────────────┤
      │ ████████████████████████████ │ ← Warp 2 (threads 64-95)
      ├──────────────────────────────┤
      │ ████████████████████████████ │ ← Warp 3 (threads 96-127)
      └──────────────────────────────┘

Each thread in warp:
- Loads multiple values from TMEM
- Stores to its own registers
- Threads collaborate to cover entire accumulator
```

---

## partition_S / partition_D

### Purpose

Partitions a tensor into per-thread slices for the source (S) or destination (D) of a copy operation.

### Usage in Example 03

**Lines**: [examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu:383-391](../../examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu#L383-L391)

```cpp
Tensor tDgC   = thr_t2r_copy.partition_D(tCgC);    // Partition GMEM C for destination
Tensor tDrC   = make_fragment_like(tDgC);          // Allocate RMEM fragment
Tensor tDtAcc = thr_t2r_copy.partition_S(tCtAcc);  // Partition TMEM accumulator for source
Tensor tDgD   = thr_t2r_copy.partition_D(tCgD);    // Partition GMEM D for destination
Tensor tDrAcc = make_tensor<AccType>(shape(tDgD)); // Allocate RMEM for accumulator
```

### What partition_S and partition_D Do

These functions apply the thread layout from the `ThrCopy` to slice tensors:

```cpp
// Simplified implementation concept
template <class Tensor>
auto partition_S(Tensor const& src) {
  return src.compose(thr_layout_);  // Apply thread-to-coordinate mapping
}
```

**Result**: Each thread sees only its portion of the tensor.

### Example for Thread 42

```cpp
threadIdx.x = 42

thr_t2r_copy = tiled_t2r_copy.get_slice(42);
// Extracts thread 42's layout from the tiled copy

tDtAcc = thr_t2r_copy.partition_S(tCtAcc);
// Thread 42 sees: Tensor covering elements [start_42, end_42) of TMEM

tDrAcc = make_tensor<float>(shape(tDtAcc));
// Allocates registers to hold thread 42's portion

copy(tiled_t2r_copy, tDtAcc, tDrAcc);
// Copies thread 42's TMEM elements to its registers
```

---

## get_slice

### Purpose

Extracts a thread-specific "slice" from a tiled operation (MMA or Copy), applying the thread index to get per-thread layouts.

### Usage in Example 03

**Lines**:
- MMA: [examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu:210](../../examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu#L210)
- Copy: [examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu:381](../../examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu#L381)

```cpp
// MMA
ThrMMA cta_mma = tiled_mma.get_slice(mma_v);   // Use Peer CTA coordinate

// Copy
ThrCopy thr_t2r_copy = tiled_t2r_copy.get_slice(threadIdx.x);
```

### Implementation Concept

```cpp
template <class ThrIdx>
auto get_slice(ThrIdx const& thr_idx) const {
  return ThrMMA{
    thr_idx,
    thr_layout_(thr_idx),  // Extract this thread's layout
    // ...
  };
}
```

**For MMA**: `get_slice` typically takes a CTA coordinate (for 2SM)
**For Copy**: `get_slice` takes thread index within CTA

---

## tmem_allocator.allocate

### Purpose

Allocates a region of tensor memory (TMEM) for use by the CTA. TMEM is a limited on-chip resource that must be explicitly allocated and freed.

### Function Signature

```cpp
// From TMEM::Allocator1Sm or Allocator2Sm
void allocate(uint32_t num_columns, uint32_t* base_ptr);
```

**Location**: [include/cute/arch/tmem_allocator_sm100.hpp](../../include/cute/arch/tmem_allocator_sm100.hpp)

### Usage in Example 03

**Lines**: [examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu:243-247](../../examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu#L243-L247)

```cpp
using TmemAllocator = cute::TMEM::Allocator1Sm;
TmemAllocator tmem_allocator{};

if (elect_one_warp) {
  tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &shared_storage.tmem_base_ptr);
}
__syncthreads();
tCtAcc.data() = shared_storage.tmem_base_ptr;
```

### How It Works

**Step 1: Allocator Type**
```cpp
using TmemAllocator = cute::TMEM::Allocator1Sm;
```
Chooses 1SM allocator (Example 04 uses `Allocator2Sm`).

**Step 2: Allocate**
```cpp
tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &shared_storage.tmem_base_ptr);
```
- `Sm100TmemCapacityColumns` = Maximum TMEM columns available (hardware constant)
- Stores allocated base pointer in `shared_storage.tmem_base_ptr`

**Step 3: Synchronize**
```cpp
__syncthreads();
```
All threads wait for warp 0 to finish allocation.

**Step 4: Set Accumulator Pointer**
```cpp
tCtAcc.data() = shared_storage.tmem_base_ptr;
```
Updates the TMEM tensor to point at allocated memory.

### Visual Representation

```
TMEM (Tensor Memory):
┌────────────────────────────────┐
│ Pool (hardware-managed)        │
├────────────────────────────────┤
│ Allocated Region for CTA 0     │ ← base_ptr (CTA 0)
├────────────────────────────────┤
│ Allocated Region for CTA 1     │ ← base_ptr (CTA 1)
├────────────────────────────────┤
│ Free Space                     │
└────────────────────────────────┘

Allocation flow:
1. CTA requests allocation (elect_one_warp)
2. Hardware allocator finds free region
3. Returns base pointer
4. CTA uses region for accumulator
5. CTA frees region when done
```

### Example 04 Differences

**Line**: [examples/cute/tutorial/blackwell/04_mma_tma_2sm_sm100.cu:243](../../examples/cute/tutorial/blackwell/04_mma_tma_2sm_sm100.cu#L243)

```cpp
using TmemAllocator = cute::TMEM::Allocator2Sm;
```

Uses `Allocator2Sm` which allocates TMEM shared across two peer CTAs.

---

## umma_arrive_multicast_2x1SM

### Purpose

Signals a barrier arrival for a 2SM MMA operation, notifying all CTAs in the multicast mask that the MMA has completed.

### Function Signature

```cpp
CUTLASS_HOST_DEVICE
void umma_arrive_multicast_2x1SM(uint64_t const* smem_ptr, uint16_t cta_mask);
```

**Location**: [include/cutlass/arch/barrier.h:814-828](../../include/cutlass/arch/barrier.h#L814-L828)

### Implementation

```cpp
CUTLASS_HOST_DEVICE
void umma_arrive_multicast_2x1SM(uint64_t const* smem_ptr, uint16_t cta_mask) {
#if defined(CUTLASS_ARCH_TCGEN_ENABLED)
  uint32_t bar_intptr = cute::cast_smem_ptr_to_uint(smem_ptr);
  if (cute::elect_one_sync()) {
    asm volatile(
      "{\n\t"
      "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1; \n\t"
      "}"
      :
      :"r"(bar_intptr), "h"(cta_mask));
  }
#elif defined(__CUDA_ARCH__)
  asm volatile ("brkpt;\n" ::);
#endif
}
```

### Usage in Example 04

**Line**: [examples/cute/tutorial/blackwell/04_mma_tma_2sm_sm100.cu:371](../../examples/cute/tutorial/blackwell/04_mma_tma_2sm_sm100.cu#L371)

```cpp
cutlass::arch::umma_arrive_multicast_2x1SM(&shared_storage.mma_barrier, mma_mcast_mask_c);
```

### PTX Instruction Breakdown

```ptx
tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [barrier_addr], mask;
```

**Components**:
- `tcgen05.commit` - Commit operation for tcgen05 (SM100 tensor core generation 5)
- `cta_group::2` - 2 CTAs collaborate (2SM)
- `mbarrier::arrive::one` - Arrive at barrier with count of 1
- `shared::cluster` - Barrier is in cluster-wide shared memory
- `multicast::cluster` - Signal to all CTAs in mask
- `.b64` - 64-bit barrier
- `[barrier_addr]` - Barrier address in SMEM
- `mask` - 16-bit mask specifying which CTAs to signal

### What It Does

1. **Elect one thread** to execute the instruction
2. **Cast SMEM pointer** to integer for PTX
3. **Execute barrier arrive** with multicast to all CTAs in `cta_mask`
4. **All CTAs in mask** see their barrier counters update

### Visual Representation

```
Cluster (4×4 CTAs with 2SM pairs):

CTA (0,0) + CTA (0,1) ← Peer pair (2SM)
     │
     └─ umma_arrive_multicast_2x1SM(&barrier, mask)
        └─ Signals all CTAs in mask:

Multicast pattern (example mask):
    N=0  N=1  N=2  N=3
M=0  ✓    ✓    .    .    ← CTAs 0,1 (peer pair at M=0)
M=1  ✓    ✓    .    .    ← CTAs 4,5 (same N as 0,1)

Each ✓ CTA:
- Receives barrier signal
- Decrements its barrier counter
- Continues when count reaches zero
```

---

## Summary Table

| Function | Example 03 | Example 04 | Key Insight |
|----------|------------|------------|-------------|
| **tma_partition** | Partitions for 1SM TMA | Partitions for 2SM TMA | Accounts for peer CTAs internally |
| **create_tma_multicast_mask** | Creates 1SM multicast masks | Creates 2SM multicast masks | Projects cluster layout along modes |
| **copy (TMA)** | 1SM TMA load with multicast | 2SM TMA load with multicast | Hardware broadcasts to multiple CTAs |
| **gemm (MMA)** | 1SM MMA (128×256×16) | 2SM MMA (256×256×16) | PTX `cta_group::1` vs `::2` |
| **make_tmem_copy** | Creates TMEM→RMEM copy | Same | 4 warps partition accumulator |
| **partition_S/D** | Per-thread tensor slicing | Same | Applies thread layout to tensors |
| **get_slice** | Extract thread-specific layout | Same | Returns `ThrMMA` or `ThrCopy` |
| **tmem_allocator.allocate** | `Allocator1Sm` | `Allocator2Sm` | Allocates shared TMEM resource |
| **umma_arrive_multicast_2x1SM** | N/A (uses non-2SM variant) | Signals 2SM barrier | Multicasts to all CTAs in mask |

---

## Next Steps

- See [07-minimal-examples.md](07-minimal-examples.md) for standalone code to test these functions
- Refer back to [05-host-functions.md](05-host-functions.md) for host-side setup

