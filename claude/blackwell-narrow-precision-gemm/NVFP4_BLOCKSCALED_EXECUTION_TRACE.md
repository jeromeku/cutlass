# DEEP EXECUTION TRACE: NVFP4 BlockScaled GEMM Scale Factor Pipeline

**Complete Frame-by-Frame Analysis of Scale Factor Loading and MMA Integration**

This document provides comprehensive execution trace for scale factor (SFA/SFB) handling in the NVFP4 blockscaled mainloop (sm100_blockscaled_mma_warpspecialized.hpp), covering TMA descriptor setup, SMEM loading, TMEM copying, and MMA instruction execution.

## Table of Contents

1. [Data Flow Overview](#part-1-data-flow-overview)
2. [Scale Factor Layout and Organization](#part-2-scale-factor-layout-organization)
3. [TMA Descriptor Setup](#part-3-tma-descriptor-setup)
4. [Producer Warp: Scale Factor TMA Load](#part-4-producer-warp-scale-factor-tma-load)
5. [Consumer Warp: SMEM to TMEM Copy](#part-5-consumer-warp-smem-to-tmem-copy)
6. [MMA Instruction Execution with Scale Factors](#part-6-mma-instruction-execution-with-scale-factors)
7. [Detailed PTX Instructions](#part-7-detailed-ptx-instructions)

---

## PART 1: Data Flow Overview

### Frame 1.1: Complete Scale Factor Data Pipeline

**File**: `/include/cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp`

```
Host Memory (GMEM)
    |
    +----> SFA (Scale Factors for A matrix)
    |       Size: (M // 128) * (K // SFVecSize) elements
    |
    +----> SFB (Scale Factors for B matrix)
            Size: (N // 128) * (K // SFVecSize) elements
    |
    v
[TMA Descriptors: tma_load_sfa_, tma_load_sfb_]
    |
    v
SMEM Buffers (Double-buffered across Stages)
    |       smem_SFA: (MMA_TILE_M, MMA_TILE_K, Stages)
    |       smem_SFB: (MMA_TILE_N, MMA_TILE_K, Stages)
    |
    +----> UTCCP (SM100 SMEM-to-TMEM Copy)
    |
    v
TMEM Accumulators (Thread-local)
    |       tCtSFA: ((SFVecSize, MMA_NSF), stages)
    |       tCtSFB: ((SFVecSize, MMA_NSF), stages)
    |
    v
MMA Instruction (tcgen05.mma with SFA/SFB)
    |
    v
Output Accumulator (C matrix)
```

---

## PART 2: Scale Factor Layout and Organization

### Frame 2.1: Scale Factor Block Structure (Sm1xxBlockScaledBasicChunk)

**Location**: `/include/cutlass/detail/sm100_blockscaled_layout.hpp:48-59`

**Scale Factor Blocking Configuration**:

```cpp
template<int SFVecSize, UMMA::Major major = UMMA::Major::K>
struct Sm1xxBlockScaledBasicChunk {
  using Blk_MN    = _128;    // Block size in M or N dimension (128)
  using Blk_SF    = _4;      // Number of SFs per block (4)

  // K-major layout (for SFA and SFB)
  using SfKMajorAtom  = Layout< 
    Shape< Shape<_32,_4>, Shape<Int<SFVecSize>, _4>>, 
    Stride<Stride<_16,_4>, Stride<_0, _1>>
  >;
};
```

**What this means**:
- Each 128-element block in M or N has **4 scale factors**
- SFVecSize is typically 4 or 8 (compressed data width)
- Example for SFVecSize=4:
  - 128 rows of data need 128/4=32 scale factors
  - These 32 SFs are partitioned as (32/4) x 4 = 8x4 grid
  - 4 SFs per block = efficient use of 32-bit TMEM words

### Frame 2.2: SMEM Layout for Scale Factors

**Location**: `/include/cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp:229-236`

**SMEM Layout Construction**:

```cpp
// SmemLayoutSFA: (((MMA_TILE_M, MMA_TILE_K), stages))
using SmemLayoutSFA = decltype(make_layout(
  append(shape(SmemLayoutAtomSFA{}), Int<DispatchPolicy::Stages>{}),
  append(stride(SmemLayoutAtomSFA{}), 
         size(filter_zeros(SmemLayoutAtomSFA{})))
));

// For a typical case:
// SmemLayoutAtomSFA shape = ((32, 4), (4, MMA_NSF))
// After append stages:
// SmemLayoutSFA shape = ((32, 4), (4, MMA_NSF), Stages)
// Total SMEM per stage: 32*4*4*MMA_NSF elements
```

**Example for 128x256 tile with SFVecSize=4**:
```
K dimension partitions into: K / SFVecSize = 256 / 4 = 64 SFs
M dimension partitions into: M / 128 = 128 / 128 = 1 block
So 1 CTA has 64 scale factors for A

Layout in SMEM:
  Shape: ((32, 4), (4, 16), Stages)
         ((col_32, subblock_4), (sf_4, mma_nsf_16), num_stages)
  
  This represents:
  - 32 columns x 4 subblocks = 128 total (but organized)
  - 4 SFs x 16 MMA blocks = 64 total SFs
  - Stages = typically 2-4 buffers
```

### Frame 2.3: TMEM Layout for Scale Factors

**Location**: `/include/cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp:659-660`

**TMEM Fragment Creation**:

```cpp
Tensor tCtSFA = make_tensor<typename TiledMma::FrgTypeSFA>(
  shape(SmemLayoutAtomSFA{})
);

// tCtSFA has structure:
// Shape: (SFVecSize, MMA_NSF)  or  ((SFVecSize, MMA_NSF),)
// Type: SmemDescriptor (for UTCCP source) or raw registers (for destination)
```

**Example TMEM layout for 64x128 UMMA**:
```
tCtSFA fragment:
  Shape: (4, 16)           // SFVecSize=4, MMA_NSF=16
  Allocated in TMEM as:    // One logical register block
  Contains: 64 scale factors total
  
  For blockscaled MMA:
  - tCtSFA acts as multiplicative scale for A-operand data
  - tCtSFB acts as multiplicative scale for B-operand data
```

---

## PART 3: TMA Descriptor Setup

### Frame 3.1: TMA Descriptor Configuration for SFA

**Location**: `/include/cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp:421-428, 540-546`

**Type Definition**:

```cpp
using TMA_SFA = decltype(make_tma_atom_A_sm100<uint16_t>(
    GmemTiledCopySFA{},
    make_tensor(static_cast<ElementSF const*>(nullptr), LayoutSFA{}),
    SmemLayoutSFA{}(_,_,_,cute::Int<0>{}),  // Pick stage 0 layout template
    TileShape{},                             // Use full tile shape (M, N, K)
    TiledMma{},
    ClusterLayout_VMNK{}
));
```

**Key Differences from Data TMA (A/B)**:

| Aspect | Data TMA (A) | Scale Factor TMA (SFA) |
|--------|--------------|------------------------|
| Element Type | ElementA (F4/F6/F8) | uint16_t (2x uint8_t) |
| Input Shape | (M, K, L) | (M, K, L) |
| SMEM Shape | SmemLayoutA | SmemLayoutSFA |
| Data Volume | Large (128x256) | Tiny (32x4) |
| Transaction Bytes | ~16KB per stage | ~16 bytes per stage |
| Multicast Mask | mcast_mask_a | mcast_mask_sfa |

### Frame 3.2: SFA Descriptor Initialization at Kernel Launch

**Location**: `/include/cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp:540-546`

```cpp
typename Params::TMA_SFA tma_load_sfa = 
  make_tma_atom_A_sm100<uint16_t>(
    GmemTiledCopySFA{},
    tensor_sfa,           // Create tensor from ptr_SFA + layout_SFA
    SmemLayoutSFA{}(_,_,_,cute::Int<0>{}),
    TileShape{},
    TiledMma{},
    cluster_layout_vmnk
  );
```

**What make_tma_atom_A_sm100 does**:
1. Takes pointer, stride, GMEM tensor shape
2. Queries SMEM layout to determine box dimensions
3. Computes TMA transaction details (strides, leading offsets)
4. Encodes all parameters into TMA descriptor struct

**Returned TMA_SFA descriptor encodes**:
- Start address of SFA data in GMEM
- Strides (for multi-dimensional access)
- Box shape: How many SFs to load per TMA transaction
- Layout swizzle type (matching SMEM layout type)

### Frame 3.3: TMA Descriptor for SFB (Special Case)

**Location**: `/include/cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp:430-437, 548-554`

**Special handling for SFB with N=192 CTA**:

```cpp
using TMA_SFB = decltype(make_tma_atom_B_sm100<uint16_t>(
    GmemTiledCopySFB{},
    make_tensor(static_cast<ElementSF const*>(nullptr), LayoutSFB{}),
    SmemLayoutSFB{}(_,_,_,cute::Int<0>{}),
    TileShape_SF{},      // DIFFERENT from TileShape!
    TiledMMA_SF{},       // DIFFERENT TiledMma!
    ClusterLayoutSfb_VMNK{}
));
```

**Why TileShape_SF and TiledMMA_SF are different**:

```cpp
// Line 134-139 in sm100_blockscaled_mma_warpspecialized.hpp
static constexpr int CTA_N_SF = 
  cutlass::ceil_div(size<1>(CtaShape_MNK{}), Blk_MN{}) * Blk_MN{};

using TileShape_SF = decltype(make_shape(
  get<0>(CtaShape_MNK{}),          // M (unchanged)
  Int<CTA_N_SF>{} * ...,           // N (padded to 128-boundary)
  get<2>(TileShape{})              // K (unchanged)
));

using TiledMMA_SF = TiledMMA<MMA_Atom<typename TiledMma::MMA_ScaleFactor>,
                              Layout<Shape<_1,_1,_1>>,
                              Tile<Underscore,Underscore,Underscore>>;
```

**Example: When CTA N=192**:
```
Original TileShape N dimension: 192
CTA_N_SF = ceil_div(192, 128) * 128 = 256
TileShape_SF N dimension: 256 (padded!)

This padding ensures:
- UTCCP can work with 128-aligned blocks
- SFB covers N=64, N=128-192, N=192-256 partitions
- Load granularity matches UTCCP write granularity
```

---

## PART 4: Producer Warp: Scale Factor TMA Load

### Frame 4.1: TMA Load Initiation

**Location**: `/include/cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp:910-914`

**Context**: Producer warp (1 thread per CTA, elected via `elect_one_sync()`)

```cpp
if (cute::elect_one_sync()) {
  // Line 911-914: Issue all 4 TMA copies in sequence
  copy(observed_tma_load_a_->with(*tma_barrier, mcast_mask_a), 
       tAgA(_,*k_tile_iter), tAsA(_,write_stage));
  
  copy(observed_tma_load_b_->with(*tma_barrier, mcast_mask_b), 
       tBgB(_,*k_tile_iter), tBsB(_,write_stage));
  
  copy(observed_tma_load_sfa_->with(*tma_barrier, mcast_mask_sfa), 
       tAgSFA(_,*k_tile_iter), tAsSFA(_,write_stage));
  
  copy(observed_tma_load_sfb_->with(*tma_barrier, mcast_mask_sfb), 
       tBgSFB(_,*k_tile_iter), tBsSFB(_,write_stage));
}
```

**Execution Sequence**:

1. **Load A data** (128x256 FP4): ~16KB, starts TMA transaction
2. **Load B data** (128x256 FP4): ~16KB, starts TMA transaction  
3. **Load SFA data** (32x4 FP8): ~256 bytes, starts TMA transaction
4. **Load SFB data** (4x32 FP8): ~256 bytes, starts TMA transaction

**All 4 transactions use the same barrier** (`*tma_barrier`), so:
- Barrier counts arrivals from all 4 TMA units
- When all 4 arrive, barrier flips phase and consumer can proceed

### Frame 4.2: TMA Transaction Details for SFA

**What happens at hardware level**:

```
TMA Load Command for SFA:
  Source Address: &GMEM[SFA][k_tile * SFVecSize]
  Destination SMEM: &SMEM[SmemLayoutSFA](_,_,write_stage)
  
  Boxes to load:
    - Shape: (32, 4)  [from SmemLayoutAtomSFA]
    - Strides: Computed from LayoutSFA
    - Swizzle: Depends on SMEM layout type (B32/B64/B128)
  
  Barrier: Full_barrier[write_stage]
  Multicast: mcast_mask_sfa
```

**Bytes transferred per TMA transaction**:
```
SFA_bytes = 32 SFs * 4 groups * 1 byte/SF = 128 bytes (single stage)
or
SFA_bytes = (MMA_M / 128) * (K / SFVecSize) * 1 byte = varies
```

### Frame 4.3: Scale Factor Barrier Arrival

**File**: `include/cutlass/arch/barrier.h` (TMA hardware behavior)

```cpp
// When TMA transaction completes, hardware automatically does:
asm volatile(
  "mbarrier.arrive.umma.b64 _, [%0];"
  :: "l"(smem_barrier_ptr)
);

// This causes:
// 1. Barrier arrival count decrements by 1
// 2. If count reaches 0:
//    - Phase bit flips
//    - All waiting threads are woken up
// 3. Producer remains stalled on EMPTY barrier for next iteration
```

**For SFA/SFB specifically**:
- Both TMA_SFA and TMA_SFB signal to **same barrier** as A/B data
- So stage is only "full" when **all 4** data transfers complete
- No partial data states for scale factors

---

## PART 5: Consumer Warp: SMEM to TMEM Copy

### Frame 5.1: Scale Factor SMEM to TMEM Copy Setup

**Location**: `/include/cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp:832-850`

**Copy Atom Selection**:

```cpp
// Line 833-835: Choose UTCCP operation based on 1SM vs 2SM mode
using AtomThrID = typename TiledMma::AtomThrID;
using UtccpOp = cute::conditional_t<(size(AtomThrID{}) == Int<2>{}),
  SM100_UTCCP_4x32dp128bit_2cta,   // 2SM mode
  SM100_UTCCP_4x32dp128bit_1cta    // 1SM mode
>;

// Line 836: Create UTCCP copy operation
auto tiled_copy_s2t_SFA = make_utccp_copy(UtccpOp{}, tCtSFA_compact);
auto tiled_copy_s2t_SFB = make_utccp_copy(UtccpOp{}, tCtSFB_compact);
```

**UTCCP Overview** (Universal TMEM Copy Pipeline):
- Hardware-assisted copy from SMEM to TMEM
- Only 1 thread per UMMA can issue the copy (via elect_one_sync)
- Direct memory copy with minimal overhead
- Supports up to 128 bits per instruction

### Frame 5.2: SMEM Descriptor Creation for UTCCP

**Location**: `/include/cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp:839-849`

```cpp
// Line 839: Get the copy thread
auto thr_copy_s2t_SFA = tiled_copy_s2t_SFA.get_slice(0);

// Line 840: Partition SMEM for this thread (source)
auto thr_tCsSFA_compact_s2t_ = thr_copy_s2t_SFA.partition_S(tCsSFA_compact);

// Line 842: Convert SMEM partition to SMEM descriptor format
// This is CRUCIAL: UTCCP can only work with smem_desc tensors
auto thr_tCsSFA_compact_s2t = 
  get_utccp_smem_desc_tensor<UtccpOp>(thr_tCsSFA_compact_s2t_);

// Line 843: Partition TMEM for this thread (destination)
auto thr_tCtSFA_compact_s2t = thr_copy_s2t_SFA.partition_D(tCtSFA_compact);
```

**What get_utccp_smem_desc_tensor does** (line 400-419 in mma_traits_sm100.hpp):
1. Takes raw SMEM tensor partition
2. Computes logical bit indices for core matrix elements
3. Creates SmemDescriptor tensor covering same data
4. Returns descriptor that UTCCP copy will interpret

**Example descriptor structure**:
```
thr_tCsSFA_compact_s2t:
  Type: Tensor<DescriptorIterator, Layout<...>>
  Data: SmemDescriptor (4-element tuple: lo, hi, low, high)
  Points to SMEM layout of scale factors in specific stage
```

### Frame 5.3: Actual UTCCP Copy Execution

**Location**: `/include/cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp:1014-1015` (in mma function)

```cpp
if (cute::elect_one_sync()) {
  // During iteration for k_block in K dimension:
  copy(tiled_copy_s2t_SFA, 
       thr_tCsSFA_s2t(_,_,_,_,read_stage),   // Source: SMEM descriptor
       thr_tCtSFA_s2t);                      // Destination: TMEM

  copy(tiled_copy_s2t_SFB, 
       thr_tCsSFB_s2t(_,_,_,_,read_stage),   // Source: SMEM descriptor
       thr_tCtSFB_s2t);                      // Destination: TMEM
}
```

**Execution detail**:

```cpp
// Pseudo-code for what copy() does
for_each_fragment_element {
  uint32_t smem_addr = descriptor_iterator[element];
  uint32_t tmem_addr = tmem_output[element];
  
  asm volatile(
    "utccp.copy.async.shared::cta.global::cta.b128 "
    "[ %0 ], [ %1 ];"
    :: "r"(tmem_addr), "l"(smem_addr)
  );
}
```

**Timing relative to MMA**:
```
K-iteration N:
  |---> elect_one_sync()
  |       +---> copy SFA,SFB from SMEM to TMEM [async]
  |
  +---> all threads proceed to MMA
          (UTCCP still moving data asynchronously)
          
  MMA instruction waits if TMEM not ready
  (implicit hazard due to register dependency)
```

### Frame 5.4: UTCCP Operation Variants

**For 1SM MMA** (SM100_UTCCP_4x32dp128bit_1cta):
```
- Opcode uses CTA-scope synchronization
- Safe for single SM execution
- No cross-SM communication
```

**For 2SM MMA** (SM100_UTCCP_4x32dp128bit_2cta):
```
- Coordinates with peer SM in cluster
- Both SMs have equal TMEM addressing
- Scale factor data duplicated across SMs for efficiency
```

---

## PART 6: MMA Instruction Execution with Scale Factors

### Frame 6.1: MMA Initialization with Scale Factors

**Location**: `/include/cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp:993-1032`

```cpp
// Line 993: Set accumulation mode
tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

// Line 1025-1027: Prepare MMA with scale factors
cute::gemm(
  tiled_mma.with(
    tiled_mma.accumulate_,    // ScaleOut::Zero for first iter
    tCtSFA(_,_,k_block),       // Scale factor for A
    tCtSFB_mma(_,_,k_block)    // Scale factor for B
  ),
  tCrA(_,_,k_block,read_stage),    // A operand descriptors
  tCrB(_,_,k_block,read_stage),    // B operand descriptors
  accumulators                      // Output accumulator
);

// Line 1031: After first iteration, use ScaleOut::One
tiled_mma.accumulate_ = UMMA::ScaleOut::One;
```

**What `.with()` does**:
- Returns new MMA_Atom with modified parameters
- Encodes scale factors into instruction descriptor
- Sets accumulation mode (Zero = overwrite, One = accumulate)

**Scale factor tensor structure**:
```cpp
tCtSFA shape: (SFVecSize, MMA_NSF)       // e.g., (4, 16)
tCtSFB_mma shape: (SFVecSize, MMA_NSF)   // e.g., (4, 16)

These are TMEM registers containing:
  - tCtSFA[i,j]: Scale factor for A[i*32 : i*32+32, j*VecSize : j*VecSize+VecSize]
  - tCtSFB[i,j]: Scale factor for B[i*32 : i*32+32, j*VecSize : j*VecSize+VecSize]
```

### Frame 6.2: Scale Factor Application in MMA

**Location**: UMMA Hardware (PTX tcgen05.mma instruction)

**Data flow inside MMA**:

```
Input A matrix (K_TILE x MMA_M) of F4:
  Row i = [F4_0, F4_1, ... F4_K]
  
Apply SFA[i]:
  Row i (scaled) = [F4_0, F4_1, ... F4_K] * SFA[i]
  (Multiply all elements in row by same scale factor)
  
Input B matrix (K_TILE x MMA_N) of F4:
  Col j = [F4_0; F4_1; ... F4_K]
  
Apply SFB[j]:
  Col j (scaled) = [F4_0; F4_1; ... F4_K] * SFB[j]
  (Multiply all elements in col by same scale factor)
  
Actual matrix multiply:
  C += (A_scaled) @ (B_scaled)
    = (A * SFA) @ (B * SFB)
    = A @ B * SFA * SFB  (per element)
```

### Frame 6.3: SFB Special Handling for N=192 CTA

**Location**: `/include/cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp:967-985`

```cpp
auto tCtSFB_mma = [tCtSFB = tCtSFB, cta_tile_coord]() {
  if constexpr (IsCtaN192) {
    // For ODD tile in N=192 case, shift TMEM address
    auto tCtSFB_tmp = tCtSFB;
    if (size<1>(cta_tile_coord) % 2 == 1) {
      tCtSFB_tmp.data() = tCtSFB_tmp.data().get() + 2;  // Offset by 2 words
    }
    return tCtSFB_tmp;
  }
  else if constexpr (IsCtaN64) {
    // For N=64 case, shift in increments of 2
    auto tCtSFB_tmp = tCtSFB;
    tCtSFB_tmp.data() = tCtSFB_tmp.data().get() + (size<1>(cta_tile_coord) % 2) * 2;
    return tCtSFB_tmp;
  }
  else {
    return tCtSFB;
  }
}();
```

**Why this offset is needed**:

```
When CTA N = 192:
  TileShape_SF N dimension was padded to 256
  SMEM allocation covers 256 columns
  But this CTA only needs columns [0:192] or [128:320] (with overlap)
  
  Odd CTAs (N=192 case) need to skip first 64 columns
  Offset = 2 words (each word is 32 SFs, so 2*32=64 columns)
  
Similarly for N=64:
  Odd CTAs skip 64 columns of SFB
```

---

## PART 7: Detailed PTX Instructions

### Frame 7.1: TMA Load Instructions for Scale Factors

**Generated for SFA Load**:

```ptx
// Pseudo-PTX showing TMA load for SFA
// Actual PTX is complex, but conceptually:

tma.load.1d.shared::cta.global::cta.v1
  [SmemLayoutSFA::addr],          // destination SMEM address
  [sfa_descriptor],               // TMA descriptor (encodes all params)
  mcast_mask_sfa;                 // multicast mask (same as data)

// This expands to multiple instructions:
// 1. mov.b64    %rd0, descriptor_address
// 2. tma.load.sync [%r0], [%rd0], mask
// 3. (hardware handles multi-dim indexing based on descriptor)
```

**Execution characteristics**:
- **Asynchronous**: Completes in background
- **Multi-box**: Can load multiple non-contiguous boxes
- **Width**: Up to 128 bits per box (SFA is only 16-32 bits)
- **Arrival signal**: Automatically arrives at FULL barrier

### Frame 7.2: UTCCP Copy Instructions

**Generated for SFA SMEM->TMEM copy**:

```ptx
// UTCCP copy for SFA (4x32 core matrix)
// Executed by 1 thread per CTA

// Each fragment element becomes one copy instruction:
utccp.copy.async.shared::cta.global::cta.b128
  [%0],     // TMEM destination (registers encoded as tmem address)
  [%1];     // SMEM source (from descriptor iterator)

// For SFA_compact with shape (SFVecSize, MMA_NSF):
// Example: (4, 16) = 64 elements
// But grouped into 128-bit chunks:
//   - 8 instructions (64 elements / 8 per instruction)
//   - Each loads 8 uint16_t scale factors
```

**Detailed loop** (conceptual):

```cpp
// What the copy() primitive does internally:
for (int i = 0; i < 4; ++i) {        // SFVecSize dimension
  for (int j = 0; j < 16; ++j) {     // MMA_NSF dimension
    uint128_t smem_val = load_from_smem_descriptor(thr_tCsSFA_s2t(i,j));
    store_to_tmem(thr_tCtSFA_s2t(i,j), smem_val);
    // This generates UTCCP instructions
  }
}
```

### Frame 7.3: UMMA MMA Instruction with Scale Factors

**The main blockscaled MMA instruction**:

**File**: `/include/cute/arch/mma_sm100_umma.hpp` (shows structure, actual PTX is generated by compiler)

```ptx
// tcgen05.mma instruction with scale factors
// Called from: cute::gemm(tiled_mma.with(scale_out, tCtSFA, tCtSFB), ...)

tcgen05.mma.cta_group::1.kind::f32  [%tmem_c],
  %desc_a,                  // SMEM descriptor for A (DescriptorIterator)
  %desc_b,                  // SMEM descriptor for B (DescriptorIterator)
  %scale_out,               // ScaleOut::Zero or ::One
  %idesc_e,                 // Instruction descriptor (contains SF info)
  {%mask[0..3]},            // 4-element mask (all zeros)
  %pred;                    // Predicate (condition code)
```

**Parameter encoding**:

```cpp
// From cute::SM100_MMA_TF32_SS::fma (sm100_umma.hpp:57-80)
// Simplified pseudo-code:

asm volatile(
  "{\n\t"
  ".reg .pred p;\n\t"
  "setp.ne.b32 p, %4, 0;\n\t"           // Predicate = (scaleC != 0)
  "tcgen05.mma.cta_group::1.kind::f32 [%0], %1, %2, %3, "
                        // %0=tmem_c, %1=desc_a, %2=desc_b, %3=idescE>>32
  "{%5, %6, %7, %8}, p; \n\t"           // {masks}, predicate
  "}\n"
  :
  : "r"(tmem_c),                        // C accumulator address
    "l"(desc_a),                        // A descriptor (64-bit)
    "l"(desc_b),                        // B descriptor (64-bit)
    "r"(uint32_t(idescE>>32)),          // Instruction descriptor upper 32
    "r"(scaleC),                        // ScaleOut value
    "r"(mask[0]), "r"(mask[1]),         // Mask bits
    "r"(mask[2]), "r"(mask[3])
);
```

**What idescE contains** (Instruction Descriptor):

```
idescE encodes:
  - idesc.a_format_: 3 bits for A data type (F8/F6/F4 code)
  - idesc.b_format_: 3 bits for B data type
  - idesc.a_scale_: Scale factor control for A
  - idesc.b_scale_: Scale factor control for B
  - (other control bits)

For blockscaled:
  - a_scale_ = 1 (enabled, reads from tCtSFA)
  - b_scale_ = 1 (enabled, reads from tCtSFB)
  - Data format bits match input types
```

**Actual scale factor application** (hardware behavior):

```
Inside UMMA datapath:

1. Fetch element A[i,k] (compressed F4)
2. Fetch SFA[i] from TMEM
3. Decompress: A_decompressed = expand_f4(A[i,k])
4. Scale: A_scaled = A_decompressed * SFA[i]
5. Fetch element B[j,k]
6. Fetch SFB[j] from TMEM
7. Decompress: B_decompressed = expand_f4(B[j,k])
8. Scale: B_scaled = B_decompressed * SFB[j]
9. Multiply: A_scaled * B_scaled
10. Accumulate: C[i,j] += result (or = result if ScaleOut::Zero)
```

### Frame 7.4: Complete MMA Loop with Scale Factors

**Actual kernel code** (lines 1022-1032 in sm100_blockscaled_mma_warpspecialized.hpp):

```cpp
CUTLASS_PRAGMA_UNROLL
for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
  // Create MMA variant with scale factors for this K block
  cute::gemm(
    tiled_mma.with(
      tiled_mma.accumulate_,        // ScaleOut control
      tCtSFA(_,_,k_block),          // SFA for k_block
      tCtSFB_mma(_,_,k_block)       // SFB for k_block
    ),
    tCrA(_,_,k_block,read_stage),   // A descriptors for k_block
    tCrB(_,_,k_block,read_stage),   // B descriptors for k_block
    accumulators                      // TMEM output
  );

  // After first k_block, switch to accumulate mode
  tiled_mma.accumulate_ = UMMA::ScaleOut::One;
}

// Result: C += (A * SFA) @ (B * SFB) for all K blocks
```

**Instruction sequence**:
```
K_block=0, ScaleOut=Zero:
  tcgen05.mma [...tCtSFA[0], ...tCtSFB[0], ...] // Zero accumulator first
  
K_block=1, ScaleOut=One:
  tcgen05.mma [...tCtSFA[1], ...tCtSFB[1], ...] // Accumulate to existing
  
K_block=2, ScaleOut=One:
  tcgen05.mma [...tCtSFA[2], ...tCtSFB[2], ...] // Continue accumulating
  
... (repeats for each K block)
```

---

## Appendix A: Scale Factor Memory Layout Examples

### Example 1: 128x256 GEMM with SFVecSize=4

```
Input matrices:
  A: 128x256 (F4 compressed)
  B: 128x256 (F4 compressed)
  
Scale factors in GMEM:
  SFA: (M/128) x (K/4) = 1 x 64 = 64 elements
  SFB: (N/128) x (K/4) = 1 x 64 = 64 elements
  
SMEM buffers (per stage):
  smem_A: 128x256 F4 = 16 KB
  smem_B: 128x256 F4 = 16 KB
  smem_SFA: 32x4 F8 = 256 B  (logical shape reflects blocks)
  smem_SFB: 32x4 F8 = 256 B
  
TMEM per iteration:
  tCtSFA: 4x16 = 64 registers (SFVecSize=4, MMA_NSF=16)
  tCtSFB: 4x16 = 64 registers
```

### Example 2: 192x256 GEMM (Padded to 256 N)

```
Input matrices:
  A: 192x256 (F4)
  B: 192x256 (F4)
  
Tile dimensions in blocks:
  M blocks: 192/128 = 1.5 -> round up for SFA computation
  N blocks: 192/128 = 1.5 -> padded to 2 for SFB (256 total)
  
Scale factors:
  SFA: 32x4 (covers first 128 cols, then used for 64-192)
  SFB: 64x4 (covers 256 cols due to padding)
  
Memory overhead:
  SFA: 128 bytes  (unchanged)
  SFB: 256 bytes  (2x due to padding)
  
CTA handling:
  CTA[0]: Uses SFB[0:64] (cols 0-64)
  CTA[1]: Uses SFB[64:128] (cols 64-128)
  CTA[2]: Uses SFB+offset[0:64] (cols 128-192), offset=+2 words
  CTA[3]: Uses SFB+offset[64:128] (cols 192-256), offset=+2 words
```

---

## Summary: Complete Data Path

```
Kernel Launch:
  1. TMA descriptors created (tma_load_sfa_, tma_load_sfb_)
  2. SMEM layouts defined (SmemLayoutSFA, SmemLayoutSFB)
  3. UTCCP operations initialized (tiled_copy_s2t_SFA, tiled_copy_s2t_SFB)

Load Phase (Producer, 1 thread):
  1. TMA loads A data -> smem_A[stage]
  2. TMA loads B data -> smem_B[stage]
  3. TMA loads SFA data -> smem_SFA[stage]  [256 bytes via 1 transaction]
  4. TMA loads SFB data -> smem_SFB[stage]  [256 bytes via 1 transaction]
  5. All 4 arrive at FULL_barrier[stage]

Compute Phase (Consumer, 32 threads):
  1. Wait for FULL_barrier[stage]
  2. (1 thread) UTCCP copies SFA: smem_SFA[stage] -> tCtSFA
  3. (1 thread) UTCCP copies SFB: smem_SFB[stage] -> tCtSFB
  4. (all threads) For each K block:
       - MMA: C += (A @ B) with scale factors tCtSFA, tCtSFB
  5. Signal EMPTY_barrier[stage]

Output:
  Accumulator C contains: A_scaled @ B_scaled
  = A * SFA @ B * SFB (element-wise scaling applied)
```

