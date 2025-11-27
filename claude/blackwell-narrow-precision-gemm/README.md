# Blackwell Narrow Precision GEMM: Complete Walkthrough

This document provides a comprehensive, line-by-line walkthrough of the Blackwell narrow precision GEMM example, tracing the execution flow from user call through the fully unrolled call stack.

## Table of Contents

1. [Overview](#overview)
2. [Type Definitions and Configuration](#type-definitions-and-configuration)
3. [Narrow Precision Types Deep Dive](#narrow-precision-types-deep-dive)
4. [Host-Side API Walkthrough](#host-side-api-walkthrough)
5. [CollectiveBuilder Template Instantiation](#collectivebuilder-template-instantiation)
6. [Block Scaling Configuration](#block-scaling-configuration)
7. [Device-Side Execution Flow](#device-side-execution-flow)
8. [Testing Components in Isolation](#testing-components-in-isolation)
9. [Related Unit Tests](#related-unit-tests)

---

## Overview

**Example File**: [examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu](../../examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu)

This example demonstrates NVFP4 (4-bit floating point) GEMM on Blackwell SM100 architecture with:
- **Input A**: `nv_float4_t<float_e2m1_t>` (4-bit data with FP8 scale factors)
- **Input B**: `nv_float4_t<float_e2m1_t>` (4-bit data with FP8 scale factors)
- **Output D**: `float_e2m1_t` (4-bit) with generated block scale factors
- **Accumulator**: `float` (32-bit for accuracy)

### Key Features

1. **Block-scaled tcgen05.mma instructions** - Hardware MMA operations with per-block scaling
2. **Tensor Memory (TMEM)** - Per-SM memory for intermediate storage
3. **Warp-specialized kernel design** - Separate warps for MMA and epilogue
4. **Dynamic cluster launch control** - Software-controlled scheduler

---

## Type Definitions and Configuration

### Matrix Element Types

**Lines 95-113** in [72b_blackwell_nvfp4_nvfp4_gemm.cu:95-113](../../examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu#L95-L113):

```cpp
// A matrix configuration
using ElementA    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;  // Line 95
using LayoutATag  = cutlass::layout::RowMajor;                    // Line 96
constexpr int AlignmentA  = 32;                                   // Line 97

// B matrix configuration
using ElementB    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;  // Line 100
using LayoutBTag  = cutlass::layout::ColumnMajor;                 // Line 101
constexpr int AlignmentB  = 32;                                   // Line 102

// C/D matrix configuration
using ElementD    = cutlass::float_e2m1_t;                        // Line 105
using ElementSFD  = cutlass::float_ue8m0_t;                       // Line 106
using ElementC    = float;                                        // Line 107
using LayoutCTag  = cutlass::layout::RowMajor;                    // Line 108
using LayoutDTag  = cutlass::layout::RowMajor;                    // Line 109
```

**Type Breakdown**:

| Type | Purpose | Definition | Bit Width |
|------|---------|------------|-----------|
| `nv_float4_t<float_e2m1_t>` | Input matrices A, B | Wrapper containing data type + scale factor type | 4-bit data + 8-bit SF |
| `float_e2m1_t` | 4-bit float data | 2 exponent bits, 1 mantissa bit | 4 bits |
| `float_ue8m0_t` | Scale factor (NV) | 8-bit unsigned exponent, 0 mantissa | 8 bits |
| `float` | Accumulator, Compute | Standard IEEE 754 | 32 bits |

### Kernel Configuration

**Lines 116-126** in [72b_blackwell_nvfp4_nvfp4_gemm.cu:116-126](../../examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu#L116-L126):

```cpp
using ElementAccumulator  = float;                                // Line 116
using ElementCompute      = float;                                // Line 117
using ArchTag             = cutlass::arch::Sm100;                 // Line 118
using OperatorClass       = cutlass::arch::OpClassBlockScaledTensorOp; // Line 119

// Kernel Perf config
using MmaTileShape        = Shape<_128,_128,_256>;                // Line 122
using ClusterShape        = Shape<_1,_1,_1>;                      // Line 123
```

**Key Parameters**:
- **ArchTag**: `Sm100` - Blackwell architecture
- **OperatorClass**: `OpClassBlockScaledTensorOp` - Signals block-scaled tensor operations
- **MmaTileShape**: `128×128×256` - MMA tile dimensions (M, N, K)
- **ClusterShape**: `1×1×1` - Single threadblock cluster (can be larger for multi-cast)

---

## Narrow Precision Types Deep Dive

### 1. float_e2m1_t - The Base 4-bit Type

**Defined in** [include/cutlass/float_subbyte.h:79-100](../../include/cutlass/float_subbyte.h#L79-L100):

```cpp
// E2M1: 2 Exponent bits with 1 Mantissa bit
// Range: +-[0,0.5,1,1.5,2,3,4,5,6]
// has_Inf: false, has_NaN: false, has_denorm: true
// Exponent bias (exp_bias): 1

struct float_e2m1_t : public float_exmy_base<cutlass::detail::FpEncoding::E2M1, float_e2m1_t> {
  using Base = float_exmy_base<cutlass::detail::FpEncoding::E2M1, float_e2m1_t>;

  float_e2m1_t() = default;

  CUTLASS_HOST_DEVICE
  explicit float_e2m1_t(float x) : Base(x) {}

  // ... additional constructors
};

// Size specialization
template <>
struct sizeof_bits<float_e2m1_t> {
  static constexpr int value = 4;  // Line 137
};
```

**Bit Layout** (4 bits total):
```
| Sign (1) | Exponent (2) | Mantissa (1) |
```

**Representable Values**:
- Exponent bias = 1
- Can represent: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0} and their negatives
- No infinity or NaN (finite range only)

### 2. nv_float4_t - Block-Scaled Wrapper

**Defined in** [include/cutlass/float_subbyte.h:506-513](../../include/cutlass/float_subbyte.h#L506-L513):

```cpp
template <class F4Type>
struct nv_float4_t
{
  static_assert(cute::is_same_v<F4Type,cutlass::float_e2m1_t> ||
                cute::is_same_v<F4Type,type_erased_dynamic_float4_t>,
                "Only float_e2m1_t type_erased_dynamic_float4_t can have scale factors for NVFP4");
  using ScaleFactorType = cutlass::float_ue4m3_t;  // 8-bit unsigned exp
  using DataType = F4Type;                         // 4-bit float_e2m1_t
};
```

**Purpose**:
- Groups data type with its associated scale factor type
- Used by builders to automatically deduce scale factor tensor layouts
- **ScaleFactorType**: `float_ue4m3_t` (4 exponent bits, 3 mantissa bits, unsigned)

### 3. Block Scaling Concept

Block-scaled arithmetic partitions matrices into blocks and applies per-block scale factors:

```
Actual_value = (Quantized_value) × (Block_Scale_Factor)
```

For a 128×256 block of matrix A:
- **Data tensor**: 128×256 elements of `float_e2m1_t` (4-bit each)
- **Scale factor tensor**: (128/128)×(256/16) = 1×16 scale factors
  - One scale factor per 128 rows × 16 columns
  - Scale factors stored as `float_ue4m3_t` (8-bit)

---

## Host-Side API Walkthrough

### Execution Flow in main()

**Entry Point**: [72b_blackwell_nvfp4_nvfp4_gemm.cu:553-595](../../examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu#L553-L595)

```cpp
int main(int argc, char const **args) {
  // Line 553: Entry point

  // Lines 556-561: CUDA version check
  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 8)) {
    std::cerr << "This example requires CUDA 12.8 or newer." << std::endl;
    return 0;
  }

  // Lines 563-572: Device capability check (SM100)
  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));

  if (props.major != 10 || (props.minor != 0 && props.minor != 1 && props.minor != 3)) {
    std::cerr << "This example requires a GPU with compute capability 100a|f, 101a|f, or 103a|f)." << std::endl;
    return 0;
  }

  // Lines 578-585: Parse command line options
  Options options;
  options.parse(argc, args);
  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  // Line 591: Run GEMM with instantiated types
  run<Gemm>(options);  // ← Main execution starts here

  return 0;
}
```

### run() Function - GEMM Execution

**Defined at**: [72b_blackwell_nvfp4_nvfp4_gemm.cu:487-547](../../examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu#L487-L547)

```cpp
template <typename Gemm>
int run(Options &options)
{
  // Line 489: Initialize host tensors and layouts
  initialize(options);  // ← Sets up A, B, C, SFA, SFB tensors

  // Line 492: Instantiate CUTLASS kernel
  Gemm gemm;  // Device adapter for GemmKernel

  // Line 495: Create arguments structure
  auto arguments = args_from_options(options);

  // Line 498: Query workspace size
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Line 501: Allocate workspace
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Line 504: Check problem size support
  CUTLASS_CHECK(gemm.can_implement(arguments));

  // Line 507: Initialize kernel with arguments
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

  // Line 510: Run kernel (warmup)
  CUTLASS_CHECK(gemm.run());

  cudaDeviceSynchronize();

  // Line 516: Verify correctness
  Result result;
  result.passed = verify(options);  // ← Reference comparison

  // Lines 524-543: Profiling loop
  if (options.iterations > 0) {
    GpuTimer timer;
    timer.start();
    for (int iter = 0; iter < options.iterations; ++iter) {
      CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
      CUTLASS_CHECK(gemm.run());
    }
    timer.stop();

    float elapsed_ms = timer.elapsed_millis();
    result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
    result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);

    std::cout << "  Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << std::endl;
    std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  GFLOPS: " << result.gflops << std::endl;
  }

  return 0;
}
```

### initialize() - Tensor Setup

**Defined at**: [72b_blackwell_nvfp4_nvfp4_gemm.cu:360-408](../../examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu#L360-L408)

```cpp
void initialize(const Options &options) {
  using namespace cute;

  // Lines 363-365: Get config types for layouts
  using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  // Lines 367-370: Create strides for A, B, C, D
  stride_A = cutlass::make_cute_packed_stride(StrideA{}, {options.m, options.k, 1});
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, {options.n, options.k, 1});
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, {options.m, options.n, 1});
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, {options.m, options.n, 1});

  // Lines 372-375: Create layouts for A, B, C, D
  layout_A = make_layout(make_shape(options.m, options.k, 1), stride_A);
  layout_B = make_layout(make_shape(options.n, options.k, 1), stride_B);
  layout_C = make_layout(make_shape(options.m, options.n, 1), stride_C);
  layout_D = make_layout(make_shape(options.m, options.n, 1), stride_D);

  // Lines 376-378: Create scale factor layouts
  layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
      cute::make_shape(options.m, options.n, options.k, 1));
  layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
      cute::make_shape(options.m, options.n, options.k, 1));
  layout_SFD = SfdOutputCfg::tile_atom_to_shape_SFD(
      cute::make_shape(options.m, options.n, options.k, 1));

  // Lines 380-390: Allocate host tensors
  block_A.reset(cutlass::make_Coord(size(layout_A)));
  block_B.reset(cutlass::make_Coord(size(layout_B)));
  block_C.reset(cutlass::make_Coord(size(layout_C)));
  block_D.reset(cutlass::make_Coord(size(layout_D)));
  block_SFA.reset(cutlass::make_Coord(size(filter_zeros(layout_SFA))));
  block_SFB.reset(cutlass::make_Coord(size(filter_zeros(layout_SFB))));
  block_SFD.reset(cutlass::make_Coord(size(filter_zeros(layout_SFD))));

  // Lines 392-397: Initialize with random data
  initialize_block(block_A.host_view(), seed + 2021);
  initialize_block(block_B.host_view(), seed + 2022);
  initialize_block(block_C.host_view(), seed + 2023);
  initialize_block(block_SFA.host_view(), seed + 2024);
  initialize_block(block_SFB.host_view(), seed + 2025);

  // Lines 399-406: Copy to device
  block_A.sync_device();
  block_B.sync_device();
  block_C.sync_device();
  block_D.sync_device();
  block_SFA.sync_device();
  block_SFB.sync_device();
  block_SFD.sync_device();
}
```

---

## CollectiveBuilder Template Instantiation

This section traces how the `CollectiveBuilder` dispatches to the appropriate specialized implementation.

### 1. Mainloop Collective Builder

**User-facing instantiation** at [72b_blackwell_nvfp4_nvfp4_gemm.cu:148-156](../../examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu#L148-L156):

```cpp
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,                  // = cutlass::arch::Sm100
    OperatorClass,            // = cutlass::arch::OpClassBlockScaledTensorOp
    ElementA,                 // = nv_float4_t<float_e2m1_t>
    LayoutATag,               // = cutlass::layout::RowMajor
    AlignmentA,               // = 32
    ElementB,                 // = nv_float4_t<float_e2m1_t>
    LayoutBTag,               // = cutlass::layout::ColumnMajor
    AlignmentB,               // = 32
    ElementAccumulator,       // = float
    MmaTileShape,             // = Shape<_128,_128,_256>
    ClusterShape,             // = Shape<_1,_1,_1>
    StageCountAutoCarveout<...>,  // Auto stage count
    KernelScheduleAuto        // = Auto schedule
  >::CollectiveOp;
```

### 2. Builder Dispatch Path

The `CollectiveBuilder` is defined in [include/cutlass/gemm/collective/collective_builder.hpp](../../include/cutlass/gemm/collective/collective_builder.hpp).

**Step 1**: Primary template ([collective_builder.hpp:38-63](../../include/cutlass/gemm/collective/collective_builder.hpp#L38-L63)):

```cpp
// Primary template - catches unspecialized cases
template <
  class ArchTag,
  class OpClass,
  class ElementA,
  class GmemLayoutATag,
  int AlignmentA,
  class ElementB,
  class GmemLayoutBTag,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class BuilderScheduleTag,
  class Enable = void
>
struct CollectiveBuilder;
```

**Step 2**: Includes specialized builders:
- Line 46: `#include "cutlass/gemm/collective/builders/sm100_blockscaled_umma_builder.inl"`

**Step 3**: Specialized template for block-scaled operations

Defined in [include/cutlass/gemm/collective/builders/sm100_blockscaled_umma_builder.inl:107-294](../../include/cutlass/gemm/collective/builders/sm100_blockscaled_umma_builder.inl#L107-L294):

```cpp
template <
  class ElementPairA,      // = nv_float4_t<float_e2m1_t>
  class GmemLayoutATag,    // = layout::RowMajor
  int AlignmentA,          // = 32
  class ElementPairB,      // = nv_float4_t<float_e2m1_t>
  class GmemLayoutBTag,    // = layout::ColumnMajor
  int AlignmentB,          // = 32
  class ElementAccumulator,// = float
  class TileShape_MNK,     // = Shape<_128,_128,_256>
  class ClusterShape_MNK,  // = Shape<_1,_1,_1>
  class StageCountType,    // = StageCountAutoCarveout<...>
  class BuilderScheduleTag // = KernelScheduleAuto
>
struct CollectiveBuilder<
    arch::Sm100,                      // ← Matches ArchTag
    arch::OpClassBlockScaledTensorOp, // ← Matches OperatorClass
    ElementPairA,
    GmemLayoutATag,
    AlignmentA,
    ElementPairB,
    GmemLayoutBTag,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    BuilderScheduleTag,
    cute::enable_if_t<
      // Enable conditions:
      (not cute::is_same_v<KernelMixedTmaCpAsyncWarpSpecialized1SmBlockScaledSm100, BuilderScheduleTag>) &&
      (cute::is_base_of_v<KernelScheduleBlockScaledGemmSm100, BuilderScheduleTag> ||
       cute::is_same_v<KernelScheduleAuto, BuilderScheduleTag>) &&
      // Alignment check
      detail::sm1xx_blockscaled_gemm_is_aligned<...>()>>
{
  // Builder body...
};
```

### 3. Builder Implementation Details

**Type extraction** ([sm100_blockscaled_umma_builder.inl:134-138](../../include/cutlass/gemm/collective/builders/sm100_blockscaled_umma_builder.inl#L134-L138)):

```cpp
// Extract data and scale factor types from ElementPairA/B
using ElementSFA = typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairA>::sf_type;
using ElementSFB = typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairB>::sf_type;
using ElementA = typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairA>::data_type;
using ElementB = typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairB>::data_type;
```

For `nv_float4_t<float_e2m1_t>`:
- `ElementA` / `ElementB` → `float_e2m1_t` (4-bit data)
- `ElementSFA` / `ElementSFB` → `float_ue4m3_t` (8-bit scale factors)

**MMA instruction selection** ([sm100_blockscaled_umma_builder.inl:140-147](../../include/cutlass/gemm/collective/builders/sm100_blockscaled_umma_builder.inl#L140-L147)):

```cpp
static constexpr cute::UMMA::Major UmmaMajorA = cutlass::gemm::collective::detail::tag_to_umma_major_A<GmemLayoutATag>();
static constexpr cute::UMMA::Major UmmaMajorB = cutlass::gemm::collective::detail::tag_to_umma_major_B<GmemLayoutBTag>();

static constexpr bool is_2sm = detail::blockscaled::is_2sm<TileShape_MNK, ClusterShape_MNK, BuilderScheduleTag>();
static constexpr auto Instr = detail::blockscaled::select_instr<ElementPairA, ElementPairB, ElementAccumulator, UmmaMajorA, UmmaMajorB, BuilderScheduleTag>();

using TiledMma = typename cutlass::gemm::collective::detail::TrivialBlockscaledMma<ElementPairA, ElementPairB, ElementAccumulator,
                                                                  TileShape_MNK, ClusterShape_MNK,
                                                                  UmmaMajorA, UmmaMajorB, Instr, BuilderScheduleTag, is_2sm>::type;
```

**Scale factor config** ([sm100_blockscaled_umma_builder.inl:166-170](../../include/cutlass/gemm/collective/builders/sm100_blockscaled_umma_builder.inl#L166-L170)):

```cpp
static constexpr uint32_t SFVectorSize = TiledMma::SFVecSize;  // = 16

using AtomThrID = typename TiledMma::AtomThrID;
using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<SFVectorSize>;
```

**Final Collective type** ([sm100_blockscaled_umma_builder.inl:277-293](../../include/cutlass/gemm/collective/builders/sm100_blockscaled_umma_builder.inl#L277-L293)):

```cpp
using CollectiveOp = cutlass::gemm::collective::CollectiveMma<
    DispatchPolicy,            // TMA + UMMA warp-specialized
    TileShape_MNK,             // Shape<_128,_128,_256>
    cute::tuple<ElementA, ElementSF>,  // A data + scale factor types
    StridePairA,               // Stride for A + Layout for SFA
    cute::tuple<ElementB, ElementSF>,  // B data + scale factor types
    StridePairB,               // Stride for B + Layout for SFB
    TiledMma,                  // MMA instruction configuration
    GmemTiledCopyPairA,        // TMA copy for A + SFA
    SmemLayoutAtomsA,          // SMEM layout for A + SFA
    void,
    cute::identity,
    GmemTiledCopyPairB,        // TMA copy for B + SFB
    SmemLayoutAtomsB,          // SMEM layout for B + SFB
    void,
    cute::identity
  >;
```

### 4. Epilogue Collective Builder

**User-facing instantiation** at [72b_blackwell_nvfp4_nvfp4_gemm.cu:137-146](../../examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu#L137-L146):

```cpp
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,                  // = cutlass::arch::Sm100
    OperatorClass,            // = cutlass::arch::OpClassBlockScaledTensorOp
    MmaTileShape,             // = Shape<_128,_128,_256>
    ClusterShape,             // = Shape<_1,_1,_1>
    EpilogueTileAuto,         // Auto-select epilogue tile
    ElementAccumulator,       // = float
    ElementAccumulator,       // = float
    ElementC,                 // = float
    LayoutCTag,               // = cutlass::layout::RowMajor
    AlignmentC,               // = 4
    ElementD,                 // = float_e2m1_t
    LayoutDTag,               // = cutlass::layout::RowMajor
    AlignmentD,               // = 32
    EpilogueScheduleAuto,     // Auto schedule
    FusionOperation           // LinCombBlockScaleFactor
  >::CollectiveOp;
```

**Fusion operation**: [72b_blackwell_nvfp4_nvfp4_gemm.cu:130-135](../../examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu#L130-L135):

```cpp
using FusionOperation = cutlass::epilogue::fusion::LinCombBlockScaleFactor<
    OutputSFVectorSize,  // = 16
    ElementD,            // = float_e2m1_t
    ElementCompute,      // = float
    ElementSFD,          // = float_ue8m0_t
    LayoutSFDTag,        // = RowMajor
    ElementC>;           // = float
```

This fusion operation:
- Computes: `D = alpha * acc + beta * C`
- Generates block scale factors for output D
- Stores quantized D values and their scale factors

---

## Block Scaling Configuration

### Overview

Block scaling partitions matrices into fixed-size blocks and assigns one scale factor per block.

**Configuration defined in** [include/cutlass/detail/sm100_blockscaled_layout.hpp:62-178](../../include/cutlass/detail/sm100_blockscaled_layout.hpp#L62-L178):

```cpp
template<int SFVecSize_>
struct Sm1xxBlockScaledConfig {
  static constexpr int SFVecSize = SFVecSize_;  // = 16

  using Sm1xxBlkScaledChunk = Sm1xxBlockScaledBasicChunk<SFVecSize>;
  using Blk_MN = typename Sm1xxBlkScaledChunk::Blk_MN;  // = _128
  using Blk_SF = typename Sm1xxBlkScaledChunk::Blk_SF;  // = _4
  using SfAtom = typename Sm1xxBlkScaledChunk::SfAtom;

  // ... layout deduction functions
};
```

### Block Dimensions

**Basic chunk** ([sm100_blockscaled_layout.hpp:49-59](../../include/cutlass/detail/sm100_blockscaled_layout.hpp#L49-L59)):

```cpp
template<int SFVecSize, UMMA::Major major = UMMA::Major::K>
struct Sm1xxBlockScaledBasicChunk {
  using Blk_MN    = _128;  // Block size in M/N dimension
  using Blk_SF    =   _4;  // Number of scale factors per block row/col

  // K-major scale factor atom layout
  using SfKMajorAtom  = Layout< Shape< Shape<_32,_4>, Shape<Int<SFVecSize>, _4>>,
                               Stride<Stride<_16,_4>, Stride<           _0, _1>>>;

  // MN-major scale factor atom layout
  using SfMNMajorAtom = Layout< Shape< Shape<Int<SFVecSize>, _4>,  Shape<_32,_4>>,
                               Stride<Stride<            _0, _1>, Stride<_16,_4>>>;

  using SfAtom = cute::conditional_t<major == UMMA::Major::K, SfKMajorAtom, SfMNMajorAtom>;
};
```

### Scale Factor Layout for Input A

**Deduction function** ([sm100_blockscaled_layout.hpp:87-94](../../include/cutlass/detail/sm100_blockscaled_layout.hpp#L87-L94)):

```cpp
template <class ProblemShape, class LayoutSFA = LayoutSF>
CUTE_HOST_DEVICE
static constexpr auto
tile_atom_to_shape_SFA(ProblemShape problem_shape, LayoutSFA layout_sfa = LayoutSFA{}) {
  auto problem_shape_MNKL = append<4>(problem_shape, 1);
  auto [M, N, K, L] = problem_shape_MNKL;
  return tile_to_shape(SfAtom{}, make_shape(M,K,L), Step<_2,_1,_3>{});
}
```

For problem size M=2048, K=2048:
- Data tensor A: 2048 × 2048 elements
- Scale factor tensor SFA: (2048/128) × (2048/16) = 16 × 128 scale factors

**Visualization**:

```
Matrix A (2048 × 2048):
┌─────────────────┬─────────────────┬─────────────────┬───
│ Block (0,0)     │ Block (0,1)     │ Block (0,2)     │ ...
│ 128×16 elements │ 128×16 elements │ 128×16 elements │
│ SF[0,0]         │ SF[0,1]         │ SF[0,2]         │
├─────────────────┼─────────────────┼─────────────────┼───
│ Block (1,0)     │ Block (1,1)     │ Block (1,2)     │ ...
│ 128×16 elements │ 128×16 elements │ 128×16 elements │
│ SF[1,0]         │ SF[1,1]         │ SF[1,2]         │
├─────────────────┼─────────────────┼─────────────────┼───
```

Each block has 128 rows × 16 columns = 2048 elements sharing one scale factor.

### SMEM Layout for Scale Factors

**SFA SMEM layout** ([sm100_blockscaled_layout.hpp:106-138](../../include/cutlass/detail/sm100_blockscaled_layout.hpp#L106-L138)):

```cpp
template<class TiledMma, class TileShape_MNK>
CUTE_HOST_DEVICE
static constexpr auto
deduce_smem_layoutSFA(TiledMma tiled_mma, TileShape_MNK tileshape_mnk) {

  constexpr int MMA_NSF = TiledMma::K / SFVecSize;  // Number of SF per K dimension

  // Basic storage block: 32 elements × 4 scale factors
  using mnBasicBlockShape  =  Shape<_32,_4>;
  using mnBasicBlockStride = Stride<_16,_4>;
  using kBasicBlockShape  = Shape<Int<SFVecSize>, Int<MMA_NSF>>;
  using kBasicBlockStride = Stride<_0, _1>;

  // ...compute SMEM layout based on MMA shape

  return SmemLayoutAtomSFA{};
}
```

**Purpose**: Organize scale factors in SMEM for efficient access by MMA instructions.

---

## Device-Side Execution Flow

### 1. GemmUniversal Kernel

**Kernel defined in** [include/cutlass/gemm/kernel/gemm_universal.hpp](../../include/cutlass/gemm/kernel/gemm_universal.hpp)

**Kernel instantiation** ([72b_blackwell_nvfp4_nvfp4_gemm.cu:158-164](../../examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu#L158-L164)):

```cpp
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int, int>,  // ProblemShape (M, N, K, batch)
    CollectiveMainloop,       // Mainloop collective from builder
    CollectiveEpilogue,       // Epilogue collective from builder
    void>;                    // TileScheduler (void = default)

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```

### 2. Kernel Launch

When `gemm.run()` is called ([72b_blackwell_nvfp4_nvfp4_gemm.cu:510](../../examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu#L510)):

1. **Grid configuration**: Number of CTAs determined by problem size and tile shape
2. **Block configuration**: Warps per CTA based on warp-specialized design
3. **Shared memory**: Allocated based on pipeline stages and carveout

### 3. Warp-Specialized Execution

The Blackwell kernel uses **warp specialization**:
- **Producer warps**: Load data from GMEM → TMEM/SMEM using TMA
- **Consumer warps**: Execute MMA instructions using TMEM data
- **Epilogue warps**: Perform epilogue fusion and write results

**Synchronization**: Uses barriers and pipelines for producer-consumer coordination.

### 4. MMA Instruction

For block-scaled FP4×FP4, the hardware MMA instruction is:

```
tcgen05.mma with block scaling
```

**Instruction characteristics**:
- Operand A: 4-bit data + 8-bit scale factors
- Operand B: 4-bit data + 8-bit scale factors
- Accumulator: 32-bit float
- Automatically applies: `acc += (A_data × A_sf) × (B_data × B_sf)`

### 5. Epilogue Fusion

**Epilogue operation**: `LinCombBlockScaleFactor`

**Steps**:
1. Load accumulator values (float)
2. Load C matrix (float)
3. Compute: `temp = alpha * acc + beta * C`
4. Quantize temp to 4-bit blocks
5. Generate scale factors per block
6. Store quantized D and scale factors SFD

**Block scale factor generation**:
- For each 128×16 block of D:
  - Compute max absolute value in block
  - Generate scale factor: `SF = max_abs / quantization_range`
  - Quantize elements: `D_quantized = D_float / SF`

---

## Testing Components in Isolation

This section provides minimal code snippets for testing individual components.

### 1. Testing Narrow Precision Types

**File**: `claude/blackwell-narrow-precision-gemm/test_types.cu`

```cpp
#include <iostream>
#include "cutlass/float_subbyte.h"
#include "cutlass/numeric_conversion.h"

int main() {
  // Test float_e2m1_t
  cutlass::float_e2m1_t a(2.0f);
  cutlass::float_e2m1_t b(1.5f);

  float a_f = float(a);
  float b_f = float(b);

  std::cout << "a = " << a_f << ", b = " << b_f << std::endl;
  std::cout << "a + b = " << (a_f + b_f) << std::endl;

  // Test nv_float4_t wrapper
  using NVF4 = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using DataType = typename NVF4::DataType;
  using ScaleFactorType = typename NVF4::ScaleFactorType;

  std::cout << "DataType size: " << cutlass::sizeof_bits<DataType>::value << " bits" << std::endl;
  std::cout << "ScaleFactorType size: " << cutlass::sizeof_bits<ScaleFactorType>::value << " bits" << std::endl;

  return 0;
}
```

**Compile**:
```bash
nvcc -std=c++17 -I/path/to/cutlass/include test_types.cu -o test_types
./test_types
```

### 2. Testing Block Scale Layout

**File**: `claude/blackwell-narrow-precision-gemm/test_layout.cu`

```cpp
#include <iostream>
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cute/tensor.hpp"

using namespace cute;

int main() {
  // Problem size
  int M = 2048, N = 2048, K = 2048;

  // Create block scaled config
  constexpr int SFVecSize = 16;
  using Config = cutlass::detail::Sm1xxBlockScaledConfig<SFVecSize>;

  // Get layout for SFA
  auto layout_SFA = Config::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));

  std::cout << "Matrix A shape: " << M << " × " << K << std::endl;
  std::cout << "SFA layout shape: " << shape(layout_SFA) << std::endl;
  std::cout << "SFA layout stride: " << stride(layout_SFA) << std::endl;
  std::cout << "SFA size (elements): " << size(filter_zeros(layout_SFA)) << std::endl;

  // Expected: (M/128) × (K/16) = 16 × 128 = 2048 scale factors

  return 0;
}
```

### 3. Testing TiledMma Configuration

**File**: `claude/blackwell-narrow-precision-gemm/test_tiledmma.cu`

```cpp
#include <iostream>
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_sm100.hpp"

using namespace cute;

int main() {
  // Define MMA shape
  using MmaShape = Shape<_128, _128, _256>;

  // Define element types
  using ElementA = cutlass::float_e2m1_t;
  using ElementB = cutlass::float_e2m1_t;
  using ElementC = float;

  // Create MMA atom (simplified - actual builder does more)
  // This is a simplified example; actual TiledMma construction
  // is done by the builder

  std::cout << "MMA Tile Shape: " << MmaShape{} << std::endl;
  std::cout << "Element A size: " << cutlass::sizeof_bits<ElementA>::value << " bits" << std::endl;
  std::cout << "Element B size: " << cutlass::sizeof_bits<ElementB>::value << " bits" << std::endl;

  return 0;
}
```

### 4. Testing Fusion Operation

**File**: `claude/blackwell-narrow-precision-gemm/test_fusion.cu`

```cpp
#include <iostream>
#include "cutlass/epilogue/fusion/operations.hpp"

int main() {
  // Define fusion operation
  using FusionOp = cutlass::epilogue::fusion::LinCombBlockScaleFactor<
      16,                          // SFVectorSize
      cutlass::float_e2m1_t,       // ElementD
      float,                       // ElementCompute
      cutlass::float_ue8m0_t,      // ElementSFD
      cutlass::layout::RowMajor,   // LayoutSFDTag
      float>;                      // ElementC

  std::cout << "IsBlockScaleSupported: " << FusionOp::IsBlockScaleSupported << std::endl;
  std::cout << "SFVecSize: " << FusionOp::SFVecSize << std::endl;

  // Check element types
  using ElementD = typename FusionOp::ElementOutput;
  using ElementSFD = typename FusionOp::ElementBlockScaleFactor;

  std::cout << "ElementD size: " << cutlass::sizeof_bits<ElementD>::value << " bits" << std::endl;
  std::cout << "ElementSFD size: " << cutlass::sizeof_bits<ElementSFD>::value << " bits" << std::endl;

  return 0;
}
```

---

## Related Unit Tests

### SM100 Block-Scaled Tests

Located in: `/home/jeromeku/cutlass/test/unit/gemm/device/sm100_blockscaled_tensorop_gemm/`

**Key test files**:

1. **nvf4_nvf4_bf16_bf16.cu** - Basic NVFP4×NVFP4 GEMM
   - [test/unit/gemm/device/sm100_blockscaled_tensorop_gemm/nvf4_nvf4_bf16_bf16.cu](../../test/unit/gemm/device/sm100_blockscaled_tensorop_gemm/nvf4_nvf4_bf16_bf16.cu)
   - Tests FP4 input with BF16 output
   - Various tile sizes: 128×128×256, 128×192×256, 128×256×256, 256×128×256, 256×192×256, 256×256×256

2. **nvf4_nvf4_f16_nvfp4_epilogue.cu** - NVFP4 output with epilogue fusion
   - [test/unit/gemm/device/sm100_blockscaled_tensorop_gemm/nvf4_nvf4_f16_nvfp4_epilogue.cu](../../test/unit/gemm/device/sm100_blockscaled_tensorop_gemm/nvf4_nvf4_f16_nvfp4_epilogue.cu)
   - Tests block scale factor generation in epilogue

3. **nvf4_nvf4_bf16_bf16_features.cu** - Feature tests
   - [test/unit/gemm/device/sm100_blockscaled_tensorop_gemm/nvf4_nvf4_bf16_bf16_features.cu](../../test/unit/gemm/device/sm100_blockscaled_tensorop_gemm/nvf4_nvf4_bf16_bf16_features.cu)
   - Tests various kernel features and edge cases

### SM120 Block-Scaled Tests

Located in: `/home/jeromeku/cutlass/test/unit/gemm/device/sm120_blockscaled_tensorop_gemm/`

**Key test files**:

1. **sm120_bs_gemm_nvf4_nvf4_f32_f32.cu** - SM120 basic test
   - [test/unit/gemm/device/sm120_blockscaled_tensorop_gemm/sm120_bs_gemm_nvf4_nvf4_f32_f32.cu](../../test/unit/gemm/device/sm120_blockscaled_tensorop_gemm/sm120_bs_gemm_nvf4_nvf4_f32_f32.cu)

2. **sm120_bs_gemm_nvf4_nvf4_f32_nvf4_epilogue_fusion.cu** - Epilogue fusion
   - [test/unit/gemm/device/sm120_blockscaled_tensorop_gemm/sm120_bs_gemm_nvf4_nvf4_f32_nvf4_epilogue_fusion.cu](../../test/unit/gemm/device/sm120_blockscaled_tensorop_gemm/sm120_bs_gemm_nvf4_nvf4_f32_nvf4_epilogue_fusion.cu)
   - Tests epilogue fusion with NVFP4 output

3. **sm120_bs_gemm_nvf4_nvf4_f32_f32_stream_k.cu** - Stream-K scheduling
   - [test/unit/gemm/device/sm120_blockscaled_tensorop_gemm/sm120_bs_gemm_nvf4_nvf4_f32_f32_stream_k.cu](../../test/unit/gemm/device/sm120_blockscaled_tensorop_gemm/sm120_bs_gemm_nvf4_nvf4_f32_f32_stream_k.cu)

### Running Unit Tests

```bash
# Build all tests
cd /path/to/cutlass/build
make test_unit_gemm_device_sm100_blockscaled_tensorop

# Run specific test
./test/unit/gemm/device/test_unit_gemm_device_sm100_blockscaled_tensorop_nvf4_nvf4_bf16_bf16
```

---

## Summary

This walkthrough has covered:

1. **Type System**: `float_e2m1_t` (4-bit), `nv_float4_t` wrapper, scale factor types
2. **Host API**: Initialization, tensor setup, kernel launch
3. **Builder Pattern**: Template dispatch from `CollectiveBuilder` to specialized implementations
4. **Block Scaling**: Layout configuration for scale factors, SMEM organization
5. **Device Execution**: Warp-specialized design, MMA instructions, epilogue fusion
6. **Testing**: Isolated component tests and comprehensive unit tests

For deeper exploration:
- See individual source files linked throughout this document
- Examine unit tests for specific features and edge cases
- Refer to CUTLASS documentation for architectural details

---

**Document Version**: 1.0
**Last Updated**: 2025-11-02
**CUTLASS Branch**: blackwell-examples
