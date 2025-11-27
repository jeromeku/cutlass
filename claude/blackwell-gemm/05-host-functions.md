# Host Function Deep Dive

This document provides detailed, line-by-line analysis of the key host-side templated functions used in the Blackwell GEMM examples, with full template specialization traces.

## Table of Contents

1. [make_tiled_mma](#make_tiled_mma)
2. [UMMA::tile_to_mma_shape](#ummatile_to_mma_shape)
3. [make_tma_atom](#make_tma_atom)
4. [tma_atom.get_tma_tensor](#tma_atomget_tma_tensor)

---

## make_tiled_mma

### Purpose

Creates a `TiledMMA` object that represents a tiled matrix multiply-accumulate operation. This function wraps hardware MMA instructions and their thread/CTA-level partitioning logic.

### Function Signature

```cpp
template <class MMAOp, class... Args>
CUTE_HOST_DEVICE constexpr
auto make_tiled_mma(MMAOp const& mma_op, Args const&... args)
```

**Location**: [include/cute/atom/mma_atom.hpp](../../include/cute/atom/mma_atom.hpp)

### Usage in Example 03

**Line**: [examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu:448-450](../../examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu#L448-L450)

```cpp
TiledMMA tiled_mma = make_tiled_mma(
  SM100_MMA_F16BF16_SS<TypeA, TypeB, TypeC, 128, 256, UMMA::Major::K, UMMA::Major::K>{}
);
```

### Template Specialization Trace

#### Step 1: Template Parameter Expansion

```cpp
MMAOp = SM100_MMA_F16BF16_SS<cutlass::half_t,    // TypeA
                             cutlass::half_t,    // TypeB
                             float,              // TypeC
                             128,                // M dimension
                             256,                // N dimension
                             UMMA::Major::K,     // A layout (K-major)
                             UMMA::Major::K>     // B layout (K-major)
```

#### Step 2: SM100_MMA_F16BF16_SS Definition

**Location**: [include/cute/arch/mma_sm100_umma.hpp:86-121](../../include/cute/arch/mma_sm100_umma.hpp#L86-L121)

```cpp
template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One,
          UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_F16BF16_SS
{
  static_assert(M == 64 || M == 128,
    "SM100_MMA_F16BF16 M-mode size should be 64 or 128 for 1 CTA cluster MMA.");
  static_assert((N % 8 == 0) && (8 <= N) && (N <= 256),
    "SM100_MMA_F16BF16 N-mode size should be a multiple of 8 between 8 and 256.");

  using DRegisters = void;               // No D registers (output to TMEM)
  using ARegisters = uint64_t[1];        // A descriptor (64-bit)
  using BRegisters = uint64_t[1];        // B descriptor (64-bit)
  using CRegisters = uint32_t[1];        // C accumulator (32-bit TMEM pointer)

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,            // SMEM descriptor for A
      uint64_t const& desc_b,            // SMEM descriptor for B
      uint32_t const& tmem_c,            // TMEM pointer for accumulator
      uint32_t const& scaleC,            // Scale for accumulator
      uint64_t const& idescE)            // Instruction descriptor
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
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_F16BF16_SS without CUTE_ARCH_MMA_SM100A_ENABLED");
#endif
  }
};
```

**Key Points**:
- `SS` suffix means "SMEM → SMEM" (both A and B from shared memory)
- `cta_group::1` means 1 CTA (not 2SM)
- `kind::f16` specifies F16 × F16 → F32 operation
- `elect_one_sync()` ensures only one thread executes the instruction
- Inline assembly for the `tcgen05.mma` PTX instruction

#### Step 3: MMA_Traits Specialization

**Location**: [include/cute/atom/mma_traits_sm100.hpp](../../include/cute/atom/mma_traits_sm100.hpp)

The `make_tiled_mma` function creates `MMA_Atom<MMA_Traits<SM100_MMA_F16BF16_SS<...>>>`.

```cpp
// Simplified MMA_Traits for SM100_MMA_F16BF16_SS<..., 128, 256, ...>
template <>
struct MMA_Traits<SM100_MMA_F16BF16_SS<half_t, half_t, float, 128, 256, UMMA::Major::K, UMMA::Major::K>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_128, _256, _16>;  // MMA operates on 128×256×16

  using ThrID = Layout<_1>;                  // Single "thread" (entire CTA)

  // A layout: Maps (thread, value) to (M, K) coordinates
  using ALayout = Layout<Shape<_1, Shape<_128, _16>>,
                         Stride<_0, Stride<_1, _128>>>;
  // Interpretation: 128 M-elements × 16 K-elements, stored contiguously in M then K

  // B layout: Maps (thread, value) to (N, K) coordinates
  using BLayout = Layout<Shape<_1, Shape<_256, _16>>,
                         Stride<_0, Stride<_1, _256>>>;
  // Interpretation: 256 N-elements × 16 K-elements, stored contiguously in N then K

  // C layout: Maps (thread, value) to (M, N) coordinates
  using CLayout = Layout<Shape<_1, Shape<_128, _256>>,
                         Stride<_0, Stride<_1, _128>>>;
  // Interpretation: 128 M-elements × 256 N-elements, accumulator in TMEM

  using FrgTypeA = UMMA::DescriptorIterator;  // A fragment is a descriptor
  using FrgTypeB = UMMA::DescriptorIterator;  // B fragment is a descriptor
  using FrgTypeC = uint32_t;                  // C fragment is TMEM pointer
};
```

#### Step 4: Return Type

The `make_tiled_mma` returns:

```cpp
TiledMMA<
  ThrLayoutVMNK_Atom,               // Thread layout (_1, _1, _1, _1)
  MMA_Atom<MMA_Traits<SM100_MMA_F16BF16_SS<..., 128, 256, ...>>>
>
```

**Interpretation**:
- **Atom Shape**: 128×256×16 (M×N×K per MMA instruction)
- **Thread ID**: Layout with a single "thread" (_1)
- **CTA Participation**: 1 CTA (`cta_group::1`)

### Usage in Example 04

**Line**: [examples/cute/tutorial/blackwell/04_mma_tma_2sm_sm100.cu:450-452](../../examples/cute/tutorial/blackwell/04_mma_tma_2sm_sm100.cu#L450-L452)

```cpp
TiledMMA tiled_mma = make_tiled_mma(
  SM100_MMA_F16BF16_2x1SM_SS<TypeA, TypeB, TypeC, 256, 256, UMMA::Major::K, UMMA::Major::K>{}
);
```

#### Key Differences from Example 03

**MMA Operation**: `SM100_MMA_F16BF16_2x1SM_SS` (notice `2x1SM`)

**Location**: [include/cute/arch/mma_sm100_umma.hpp](../../include/cute/arch/mma_sm100_umma.hpp)

```cpp
template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One,
          UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_F16BF16_2x1SM_SS
{
  static_assert(M == 128 || M == 256,
    "SM100_MMA_F16BF16_2x1SM M-mode size should be 128 or 256 for 2SM MMA.");
  static_assert((N % 16 == 0) && (16 <= N) && (N <= 256),
    "SM100_MMA_F16BF16_2x1SM N-mode size should be a multiple of 16 between 16 and 256.");

  // ... similar to 1SM version but with:
  // - cta_group::2 in PTX instruction
  // - ThrID = Layout<_2> (two peer CTAs)
  // - Different M dimension (256 vs 128)
```

**MMA_Traits Specialization**:

```cpp
template <>
struct MMA_Traits<SM100_MMA_F16BF16_2x1SM_SS<..., 256, 256, ...>>
{
  using Shape_MNK = Shape<_256, _256, _16>;  // 256×256×16 (double the M)

  using ThrID = Layout<_2>;  // TWO "threads" (two peer CTAs)

  // Layouts account for 2SM participation
  using ALayout = Layout<Shape<_2, Shape<_128, _16>>,
                         Stride<_128, Stride<_1, _256>>>;
  // Each peer CTA handles 128 M-elements

  // ... similar for B and C
};
```

**Visual Comparison**:

| Feature | Example 03 (1SM) | Example 04 (2SM) |
|---------|------------------|------------------|
| MMA Instruction | `SM100_MMA_F16BF16_SS` | `SM100_MMA_F16BF16_2x1SM_SS` |
| M Dimension | 128 | 256 |
| N Dimension | 256 | 256 |
| K Dimension | 16 | 16 |
| CTA Group | `cta_group::1` | `cta_group::2` |
| ThrID Layout | `_1` | `_2` |
| Execution | Single CTA | Two peer CTAs |

---

## UMMA::tile_to_mma_shape

### Purpose

Transforms a layout "atom" (basic swizzled layout pattern) into a complete shared memory layout that matches the shape expected by MMA instructions.

### Function Signature

```cpp
template <class LayoutAtom, class MMATileShape, class ModeOrder = GenColMajor>
CUTE_HOST_DEVICE constexpr
auto
tile_to_mma_shape(LayoutAtom const& atom,
                  MMATileShape const& mma_tile_shape,
                  ModeOrder const& order = {})
```

**Location**: [include/cute/atom/mma_traits_sm100.hpp:113-122](../../include/cute/atom/mma_traits_sm100.hpp#L113-L122)

### Implementation

```cpp
CUTE_HOST_DEVICE constexpr
auto
tile_to_mma_shape(LayoutAtom const& atom, MMATileShape const& mma_tile_shape, ModeOrder const& order = {})
{
  constexpr int R = decltype(rank(mma_tile_shape))::value;

  // Step 1: Create interleaved MN shape
  auto mn_shape = cute::tuple_cat(
    zip(shape<0>(mma_tile_shape), take<1,3>(mma_tile_shape)),
    take<3,R>(mma_tile_shape)
  );

  // Step 2: Tile the atom to match mn_shape
  auto mn_tiled = tile_to_shape(atom, mn_shape, order);

  // Step 3: Divide by MMA atom size to get ((MMA_M, MMA_N), M_MMAs, N_MMAs, ...)
  return tiled_divide(mn_tiled, product_each(shape<0>(mma_tile_shape)));
}
```

### Usage in Example 03

**Line**: [examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu:509](../../examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu#L509)

```cpp
auto sA_layout = UMMA::tile_to_mma_shape(
  UMMA::Layout_K_SW128_Atom<TypeA>{},  // Swizzled layout atom
  mma_shape_A                           // ((_128, _16), _1, _4)
);
```

### Step-by-Step Trace for Example 03

#### Input Parameters

```cpp
LayoutAtom = UMMA::Layout_K_SW128_Atom<half_t>
  // Expands to: Swizzle<3,4,3> o smem_ptr[16b] o ((16b_elements, 8), (8, 2))
  // This is the basic 128B-swizzled pattern for K-major layout

MMATileShape = mma_shape_A = ((_128, _16), _1, _4)
  // Mode 0: (_128, _16) - MMA atom size (128 M × 16 K)
  // Mode 1: _1          - Number of MMAs in M dimension
  // Mode 2: _4          - Number of MMAs in K dimension
```

#### Step 1: Create MN Shape

```cpp
constexpr int R = 3;  // rank(mma_tile_shape) = 3

// zip(shape<0>(mma_tile_shape), take<1,3>(mma_tile_shape))
// = zip((_128, _16), (_1, _4))
// = ((_128, _1), (_16, _4))

// tuple_cat(((_128, _1), (_16, _4)), take<3,3>(mma_tile_shape))
// = tuple_cat(((_128, _1), (_16, _4)), ())
// = ((_128, _1), (_16, _4))

mn_shape = ((_128, _1), (_16, _4))
```

**Interpretation**: Interleave M and N modes (but N=1 here since this is A matrix with MK layout).

#### Step 2: Tile the Atom

```cpp
mn_tiled = tile_to_shape(
  UMMA::Layout_K_SW128_Atom<half_t>{},
  ((_128, _1), (_16, _4))
)
```

The `tile_to_shape` function replicates the atom pattern to cover the requested shape:

```cpp
// Atom covers: 128 elements × 16 elements (in the swizzled pattern)
// Target shape: (128 × 1) × (16 × 4) = 128 × 64

// Result: Swizzle<3,4,3> o smem_ptr[16b] o ((_128, _16), _1, _4):((_64, _1), _0, _16)
//         └───────┬──────┘                  └──────┬─────┘  └─┬┘ └─┬┘  └──┬─┘ └──┬┘
//            Swizzle pattern               MMA atom   M    K   Strides
```

#### Step 3: Divide by MMA Atom Size

```cpp
product_each(shape<0>(mma_tile_shape))
  = product_each((_128, _16))
  = (_128, _16)

tiled_divide(mn_tiled, (_128, _16))
  // Divides mode 0 by (_128, _16) to extract the MMA atom size
  // Result: Swizzle<3,4,3> o smem_ptr[16b] o ((_128, _16), _1, _4):((_64, _1), _0, _16)
```

#### Final Result

```cpp
sA_layout = Swizzle<3,4,3> o smem_ptr[16b] o ((_128, _16), _1, _4):((_64, _1), _0, _16)
```

**Interpretation**:
- **Shape**: `((_128, _16), _1, _4)`
  - Mode 0: `(_128, _16)` - Single MMA atom (128 M × 16 K)
  - Mode 1: `_1` - 1 repetition in M
  - Mode 2: `_4` - 4 repetitions in K
- **Stride**: `((_64, _1), _0, _16)`
  - Within mode 0: Stride of 64 in M-direction (due to swizzling), stride of 1 in K-direction
  - Mode 1: Stride 0 (only 1 repetition)
  - Mode 2: Stride 16 between K-blocks
- **Swizzle**: `Swizzle<3,4,3>` - 128B swizzle pattern (2³=8, 2⁴=16 base, 2³=8 shift)

### Visual Representation

```
SMEM Layout (K-major, 128×64 elements, half_t = 16 bits):

M →
  ┌─────────────────────────────────────┐
K │ ▓▓▓▓░░░░▓▓▓▓░░░░▓▓▓▓░░░░▓▓▓▓░░░░ │  ← K=0..15  (MMA K-block 0)
↓ │ ▓▓▓▓░░░░▓▓▓▓░░░░▓▓▓▓░░░░▓▓▓▓░░░░ │
  │ ... (128 M-elements) ...          │
  ├─────────────────────────────────────┤
  │ ▓▓▓▓░░░░▓▓▓▓░░░░▓▓▓▓░░░░▓▓▓▓░░░░ │  ← K=16..31 (MMA K-block 1)
  │ ... (128 M-elements) ...          │
  ├─────────────────────────────────────┤
  │ ... (K-blocks 2, 3) ...           │
  └─────────────────────────────────────┘

▓ = swizzled bank 0
░ = swizzled bank 1
(Simplified - actual swizzling is more complex)
```

### Example 04 Differences

In Example 04, the layout is similar but with 2SM considerations:

```cpp
// Example 04: 256×64 elements (double the M dimension)
sA_layout = Swizzle<3,4,3> o smem_ptr[16b] o ((_128, _16), _1, _4):((_64, _1), _0, _16)
```

The layout is **per-CTA**, so each peer CTA has its own 128×64 SMEM region with the same swizzled pattern.

---

## make_tma_atom

### Purpose

Creates a TMA (Tensor Memory Accelerator) descriptor on the host that will be used by the device to perform asynchronous memory transfers between global memory (GMEM) and shared memory (SMEM).

### Function Signatures

There are two main interfaces for SM100:

#### SM100 Interface (Example 04)

```cpp
template <class TmaInternalType = void,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class Cluster_Tiler,
          class... Args>
CUTE_HOST
auto
make_tma_copy_A_sm100(
  CopyOp                  const& copy_op,      // TMA operation type
  Tensor<GEngine,GLayout> const& gtensor,      // GMEM tensor
  SLayout                 const& slayout,      // SMEM layout
  Cluster_Tiler           const& cluster_tiler,// Cluster tiler shape
  TiledMMA<Args...>       const& mma)          // TiledMMA for partitioning
```

**Location**: [include/cute/atom/copy_traits_sm100_tma.hpp:277-282](../../include/cute/atom/copy_traits_sm100_tma.hpp#L277-L282)

#### SM90 Interface (Example 03)

```cpp
template <class... Args>
CUTE_HOST
auto
make_tma_atom(
  Copy_Atom<Args...> const& copy_atom,      // TMA operation type
  Tensor              const& gtensor,        // GMEM tensor
  Layout              const& slayout,        // SMEM layout
  Tile                const& cta_tile,       // Tile shape for CTA
  Int                 const& multicast_size) // Number of CTAs in multicast
```

**Location**: [include/cute/atom/copy_traits_sm90_tma.hpp](../../include/cute/atom/copy_traits_sm90_tma.hpp)

### Usage in Example 03

**Line**: [examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu:528-534](../../examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu#L528-L534)

```cpp
Copy_Atom tma_atom_A = make_tma_atom(
    SM90_TMA_LOAD_MULTICAST{},       // TMA load operation with multicast
    mA,                              // Source GMEM tensor
    sA_layout,                       // Destination SMEM layout
    select<0,2>(mma_tiler),          // MK Tiler for TMA operation
    size<2>(cluster_layout_vmnk)     // The number of CTAs in the N-mode for multicasting
  );
```

### Step-by-Step Trace for Example 03

#### Input Parameters

```cpp
CopyOp = SM90_TMA_LOAD_MULTICAST{}
  // Multicast TMA load (broadcasts to multiple CTAs)

mA = Tensor<gmem_ptr<half_t>, Layout<(512, 256), (256, _1)>>
  // GMEM tensor: 512 M × 256 K, K-major (row-major)

sA_layout = Swizzle<3,4,3> o smem_ptr[16b] o ((_128, _16), _1, _4):((_64, _1), _0, _16)
  // SMEM layout: swizzled 128×64

select<0,2>(mma_tiler) = select<0,2>((_128, _256, _64)) = (_128, _64)
  // Extract M and K dimensions from MMA tiler

size<2>(cluster_layout_vmnk) = size<2>((2, 2, 4, 1)) = _4
  // 4 CTAs in the N-mode for multicasting
```

#### What make_tma_atom Does

The function performs several steps:

**Step 1: Determine TMA Box Size**

The "TMA box" is the contiguous region of GMEM that will be transferred in a single TMA operation.

```cpp
// TMA constraints:
// - Box dimensions must align with hardware limits
// - Max 5D tensor
// - Each dimension ≤ 256 elements (for most types)
// - Total box size ≤ 256 bytes per dimension product

// For this example:
// - Box covers (_128, _64) = 128 M × 64 K elements
// - Element size: 16 bits (half_t)
// - Total size: 128 × 64 × 2 bytes = 16 KB
```

**Step 2: Create TMA Descriptor**

```cpp
// The TMA descriptor encodes:
// - GMEM base address
// - GMEM strides
// - SMEM swizzle pattern
// - Box dimensions
// - Element size

CUtensorMap tensor_map;
cuTensorMapEncodeTiled(
  &tensor_map,
  CU_TENSOR_MAP_DATA_TYPE_FLOAT16,  // half_t
  rank,                              // 2D (M, K)
  gmem_address,                      // mA.data()
  box_dims,                          // [128, 64]
  gmem_strides,                      // [256 * sizeof(half_t), sizeof(half_t)]
  smem_strides,                      // Computed from sA_layout swizzle
  CU_TENSOR_MAP_INTERLEAVE_NONE,
  CU_TENSOR_MAP_SWIZZLE_128B,       // 128B swizzle
  CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
);
```

**Step 3: Wrap in Copy_Atom**

```cpp
Copy_Atom<Copy_Traits<SM90_TMA_LOAD_MULTICAST, NumBitsPerTMA, AuxParams>> tma_atom_A;
tma_atom_A.tma_desc_ = tensor_map;
tma_atom_A.aux_params_ = { gmem_stride };
```

#### Return Type

```cpp
Copy_Atom<
  Copy_Traits<
    SM90_TMA_LOAD_MULTICAST,
    Int<128*64*16>,  // NumBitsPerTMA = 131072 bits = 16 KB
    AuxParams
  >
>
```

**Interpretation**:
- **TMA operation**: Multicast load
- **Transfer size**: 128×64 half_t elements = 16 KB
- **Multicast**: Broadcasts to 4 CTAs along N-dimension
- **Descriptor**: Embedded `CUtensorMap` with all GMEM↔SMEM mapping info

### Usage in Example 04

**Line**: [examples/cute/tutorial/blackwell/04_mma_tma_2sm_sm100.cu:531-537](../../examples/cute/tutorial/blackwell/04_mma_tma_2sm_sm100.cu#L531-L537)

```cpp
Copy_Atom tma_atom_A = make_tma_atom_A_sm100(
    SM100_TMA_2SM_LOAD_MULTICAST{}, // TMA load operation -- Multicasting 2SM instruction.
    mA,                             // Source GMEM tensor
    sA_layout,                      // Destination SMEM layout
    mma_tiler,                      // MmaTiler_MNK. Unlike Sm90 interface where the tiler only included M and K modes.
    tiled_mma,                      // Sm100 also requires the TiledMma to perform CTA-level partitioning.
    cluster_layout_vmnk);           // ClusterLayout_VMNK. Unlike Sm90 interface where only the multicasting mode is passed.
```

#### Key Differences from Example 03

1. **Interface**: `make_tma_atom_A_sm100` (SM100-specific)
2. **Operation**: `SM100_TMA_2SM_LOAD_MULTICAST` (2SM variant)
3. **Additional parameters**: Full `mma_tiler` and `tiled_mma` for CTA-level partitioning
4. **ThrID**: `Layout<_2>` (two peer CTAs participate)

**Return Type**:

```cpp
Copy_Atom<
  Copy_Traits<
    SM100_TMA_2SM_LOAD_MULTICAST,
    Int<2*128*64*16>,  // NumBitsPerTMA = 262144 bits = 32 KB (2× due to 2SM)
    AuxParams
  >
>
```

### TMA Descriptor Visualization

```
TMA Descriptor (embedded CUtensorMap):
┌────────────────────────────────────────┐
│ GMEM Base Address: 0x7f8a00000000     │  ← mA.data()
├────────────────────────────────────────┤
│ Dimensions: 2D (M=512, K=256)          │
├────────────────────────────────────────┤
│ Box Size: [128, 64]                    │  ← Transfer size per TMA call
├────────────────────────────────────────┤
│ GMEM Strides: [256*2B, 2B]             │  ← K-major (row-major)
├────────────────────────────────────────┤
│ SMEM Swizzle: 128B, Base 16B           │  ← From sA_layout
├────────────────────────────────────────┤
│ Element Type: FP16 (16-bit)            │
├────────────────────────────────────────┤
│ Multicast: 4 CTAs along N-dimension    │
└────────────────────────────────────────┘
```

---

## tma_atom.get_tma_tensor

### Purpose

Creates a "coordinate tensor" that represents the global memory tensor in a form that can be indexed with TMA coordinates.

### Function Signature

```cpp
template <class GShape>
CUTE_HOST_DEVICE constexpr
auto
get_tma_tensor(GShape const& g_shape) const
```

**Location**: Defined in `Copy_Traits` specializations
- SM90: [include/cute/atom/copy_traits_sm90_tma.hpp](../../include/cute/atom/copy_traits_sm90_tma.hpp)
- SM100: [include/cute/atom/copy_traits_sm100_tma.hpp:105-108](../../include/cute/atom/copy_traits_sm100_tma.hpp#L105-L108)

### Implementation

```cpp
template <class GShape>
CUTE_HOST_DEVICE constexpr
auto
get_tma_tensor(GShape const& g_shape) const {
  static_assert(is_congruent<decltype(g_shape), decltype(aux_params_.g_stride_)>::value);
  return make_coord_tensor(make_layout(g_shape, aux_params_.g_stride_));
}
```

### Usage in Example 03

**Line**: [examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu:535](../../examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu#L535)

```cpp
Tensor mA_tma = tma_atom_A.get_tma_tensor(shape(mA));   // (Gemm_M, Gemm_K)
```

### Step-by-Step Trace for Example 03

#### Input Parameters

```cpp
g_shape = shape(mA) = (_512, _256)
  // Global M × K dimensions

aux_params_.g_stride_ = stride(mA) = (_256, _1)
  // K-major strides (row-major)
```

#### Step 1: Create Layout

```cpp
auto layout = make_layout(
  (_512, _256),   // shape
  (_256, _1)      // stride
);
// Result: Layout<(_512, _256), (_256, _1)>
```

**Interpretation**: 512×256 tensor, K-major (stride-256 in M, stride-1 in K).

#### Step 2: Create Coordinate Tensor

```cpp
mA_tma = make_coord_tensor(layout);
```

A "coordinate tensor" is a special tensor where each element stores its own coordinate instead of actual data.

```cpp
// Conceptually:
mA_tma(i, j) = (i, j)  // Returns coordinate, not data!

// Actual implementation:
Tensor<ArithTuple, Layout<(_512, _256), (_256, _1)>> mA_tma;
```

#### Return Type

```cpp
Tensor<
  ArithTuple,  // Engine that stores coordinates
  Layout<Shape<_512, _256>, Stride<_256, _1>>
>
```

### Why Coordinate Tensors?

Coordinate tensors enable elegant TMA partitioning:

```cpp
// Instead of manually calculating offsets:
int tma_m_offset = blockIdx.x * 128;
int tma_k_offset = k_tile * 64;

// Use coordinate tensor:
auto tma_slice = local_tile(mA_tma, mma_tiler, mma_coord, Step<_1, X, _1>{});
// Automatically computes the correct (m, k) coordinates for this CTA
```

### Visual Representation

```
Regular Tensor vs Coordinate Tensor:

Regular GMEM Tensor mA:
  K →
M ┌─────────────────┐
↓ │ 1.2  0.3  -0.5 ... │  ← Stores actual float16 values
  │ 0.1  2.1   1.2 ... │
  │ ... (512×256) ...  │
  └─────────────────┘

Coordinate Tensor mA_tma:
  K →
M ┌─────────────────┐
↓ │ (0,0) (0,1) (0,2) ... │  ← Stores (m,k) coordinates
  │ (1,0) (1,1) (1,2) ... │
  │ ... (512×256) ...     │
  └─────────────────┘

When used with TMA:
  tma_slice = local_tile(mA_tma, (_128, _64), (0, k_tile))
  // Returns coordinates of the 128×64 region to load
  // TMA hardware uses these coordinates with the descriptor
```

### Example 04 Usage

**Line**: [examples/cute/tutorial/blackwell/04_mma_tma_2sm_sm100.cu:539](../../examples/cute/tutorial/blackwell/04_mma_tma_2sm_sm100.cu#L539)

```cpp
Tensor mA_tma = tma_atom_A.get_tma_tensor(shape(mA));   // (Gemm_M, Gemm_K)
```

The usage is identical to Example 03. The difference is internal: the TMA atom has `ThrID = Layout<_2>`, so it's aware of 2SM participation.

---

## Summary Table

| Function | Example 03 | Example 04 | Key Difference |
|----------|------------|------------|----------------|
| **make_tiled_mma** | `SM100_MMA_F16BF16_SS<..., 128, 256, ...>` | `SM100_MMA_F16BF16_2x1SM_SS<..., 256, 256, ...>` | 1SM vs 2SM, M dimension (128 vs 256) |
| **tile_to_mma_shape** | Creates swizzled SMEM layout for 128×64 | Creates swizzled SMEM layout for 256×64 | Larger M for 2SM |
| **make_tma_atom** | `SM90_TMA_LOAD_MULTICAST`, SM90 interface | `SM100_TMA_2SM_LOAD_MULTICAST`, SM100 interface | 1SM vs 2SM TMA, interface differences |
| **get_tma_tensor** | Creates coordinate tensor for 512×256 | Creates coordinate tensor for 512×256 | Identical usage, 2SM aware internally |

---

## Next Steps

- Continue to [06-device-functions.md](06-device-functions.md) for device-side function analysis
- See [07-minimal-examples.md](07-minimal-examples.md) for standalone code to test these functions

