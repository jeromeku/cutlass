# Quick Reference: Blackwell Narrow Precision GEMM

A quick lookup guide for Blackwell narrow precision GEMM components and APIs.

## Table of Contents

1. [Type Reference](#type-reference)
2. [Key Classes and Functions](#key-classes-and-functions)
3. [Configuration Parameters](#configuration-parameters)
4. [Code Snippets](#code-snippets)
5. [Common Patterns](#common-patterns)
6. [Troubleshooting](#troubleshooting)

---

## Type Reference

### Narrow Precision Data Types

| Type | Bits | Format | Range | Use Case | Header |
|------|------|--------|-------|----------|--------|
| `float_e2m1_t` | 4 | E2M1 (2 exp, 1 mant) | ±[0, 0.5, 1, 1.5, 2, 3, 4, 6] | Base 4-bit float | [float_subbyte.h:79](../../include/cutlass/float_subbyte.h#L79) |
| `float_e2m3_t` | 6 | E2M3 (2 exp, 3 mant) | ±[-7.5, +7.5] | 6-bit float | [float_subbyte.h:160](../../include/cutlass/float_subbyte.h#L160) |
| `float_e3m2_t` | 6 | E3M2 (3 exp, 2 mant) | ±[-28, +28] | 6-bit float | [float_subbyte.h:267](../../include/cutlass/float_subbyte.h#L267) |
| `float_ue4m3_t` | 8 | UE4M3 | Powers of 2 | NV scale factor | [float8.h](../../include/cutlass/float8.h) |
| `float_ue8m0_t` | 8 | UE8M0 | Powers of 2 | Scale factor (exponent only) | [numeric_types.h](../../include/cutlass/numeric_types.h) |

### Block-Scaled Wrapper Types

| Type | Description | Scale Factor Type | Usage |
|------|-------------|-------------------|-------|
| `nv_float4_t<float_e2m1_t>` | NVIDIA 4-bit block-scaled | `float_ue4m3_t` | Input A/B in FP4 GEMM |
| `mx_float4_t<float_e2m1_t>` | Microscaling 4-bit | `float_ue8m0_t` | Alternative scaling |
| `mx_float6_t<float_e2m3_t>` | Microscaling 6-bit E2M3 | `float_ue8m0_t` | FP6 GEMM |
| `mx_float6_t<float_e3m2_t>` | Microscaling 6-bit E3M2 | `float_ue8m0_t` | FP6 GEMM |

### Accessing Wrapper Components

```cpp
using NVF4 = cutlass::nv_float4_t<cutlass::float_e2m1_t>;

// Extract types
using DataType = typename NVF4::DataType;              // float_e2m1_t
using ScaleFactorType = typename NVF4::ScaleFactorType; // float_ue4m3_t
```

---

## Key Classes and Functions

### CollectiveBuilder (Mainloop)

**Location**: [include/cutlass/gemm/collective/collective_builder.hpp](../../include/cutlass/gemm/collective/collective_builder.hpp)

**Usage**:
```cpp
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,              // cutlass::arch::Sm100
    OperatorClass,        // cutlass::arch::OpClassBlockScaledTensorOp
    ElementA,             // nv_float4_t<float_e2m1_t>
    LayoutATag,           // cutlass::layout::RowMajor
    AlignmentA,           // 32
    ElementB,             // nv_float4_t<float_e2m1_t>
    LayoutBTag,           // cutlass::layout::ColumnMajor
    AlignmentB,           // 32
    ElementAccumulator,   // float
    MmaTileShape,         // Shape<_128,_128,_256>
    ClusterShape,         // Shape<_1,_1,_1>
    StageCountType,       // StageCountAutoCarveout<...>
    KernelScheduleTag     // KernelScheduleAuto
  >::CollectiveOp;
```

**Key Parameters**:
- `OperatorClass`: Must be `OpClassBlockScaledTensorOp` for block-scaled ops
- `ElementA/B`: Use wrapper types (`nv_float4_t`, `mx_float4_t`, etc.)
- `AlignmentA/B`: Typically 32 for FP4, 16 for FP6
- `MmaTileShape`: Tile size (M, N, K) - common: `128×128×256`, `256×128×256`

### CollectiveBuilder (Epilogue)

**Location**: [include/cutlass/epilogue/collective/collective_builder.hpp](../../include/cutlass/epilogue/collective/collective_builder.hpp)

**Usage**:
```cpp
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,                  // cutlass::arch::Sm100
    OperatorClass,            // cutlass::arch::OpClassBlockScaledTensorOp
    MmaTileShape,             // Shape<_128,_128,_256>
    ClusterShape,             // Shape<_1,_1,_1>
    EpilogueTileType,         // EpilogueTileAuto
    ElementAccumulator,       // float
    ElementCompute,           // float
    ElementC,                 // float or bfloat16_t
    LayoutCTag,               // cutlass::layout::RowMajor
    AlignmentC,               // 4 or 8
    ElementD,                 // float_e2m1_t (for FP4 output)
    LayoutDTag,               // cutlass::layout::RowMajor
    AlignmentD,               // 32
    EpilogueScheduleType,     // EpilogueScheduleAuto
    FusionOperation           // LinCombBlockScaleFactor
  >::CollectiveOp;
```

### Fusion Operations

**Location**: [include/cutlass/epilogue/fusion/operations.hpp](../../include/cutlass/epilogue/fusion/operations.hpp)

**Common Operations**:

```cpp
// Standard linear combination: D = alpha * acc + beta * C
using FusionOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementScalar>;

// With block scale factor generation
using FusionOp = cutlass::epilogue::fusion::LinCombBlockScaleFactor<
    SFVectorSize,          // 16
    ElementD,              // float_e2m1_t
    ElementCompute,        // float
    ElementSFD,            // float_ue8m0_t
    LayoutSFDTag,          // cutlass::layout::RowMajor
    ElementC>;             // float

// With activation
using FusionOp = cutlass::epilogue::fusion::LinCombEltAct<
    cutlass::epilogue::thread::ReLU,  // Activation function
    ElementD, ElementCompute, ElementC, ElementScalar>;
```

### Block Scale Configuration

**Location**: [include/cutlass/detail/sm100_blockscaled_layout.hpp](../../include/cutlass/detail/sm100_blockscaled_layout.hpp)

**Usage**:
```cpp
// Create configuration
constexpr int SFVectorSize = 16;
using Config = cutlass::detail::Sm1xxBlockScaledConfig<SFVectorSize>;

// Get scale factor layouts
auto layout_SFA = Config::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
auto layout_SFB = Config::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));

// Get SMEM layouts (within builder)
auto smem_layout_SFA = Config::deduce_smem_layoutSFA(tiled_mma, tile_shape_mnk);
auto smem_layout_SFB = Config::deduce_smem_layoutSFB(tiled_mma, tile_shape_mnk);
```

**Constants**:
- `Blk_MN`: 128 (block size in M/N dimension)
- `Blk_SF`: 4 (number of scale factors per block row/col)
- `SFVectorSize`: 16 (typical - elements per scale factor in K dimension)

### GemmUniversal Kernel

**Location**: [include/cutlass/gemm/kernel/gemm_universal.hpp](../../include/cutlass/gemm/kernel/gemm_universal.hpp)

**Usage**:
```cpp
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,  // ProblemShape (M, N, K, Batch)
    CollectiveMainloop,
    CollectiveEpilogue,
    void                     // TileScheduler (void = default)
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```

**Key Methods**:
```cpp
Gemm gemm;
size_t workspace_size = Gemm::get_workspace_size(arguments);
cutlass::Status status = gemm.can_implement(arguments);
status = gemm.initialize(arguments, workspace_ptr);
status = gemm.run();
```

---

## Configuration Parameters

### Supported Tile Shapes (SM100)

| 1SM/2SM | Tile Size (M×N×K) | Layouts | Notes |
|---------|-------------------|---------|-------|
| 1SM | 128×128×256 | TN, TT, NT, NN | Most common |
| 1SM | 128×192×256 | TN, TT, NT, NN | Rectangular |
| 1SM | 128×256×256 | TN, TT, NT, NN | Wide N |
| 2SM | 256×128×256 | TN, TT, NT, NN | Tall M |
| 2SM | 256×192×256 | TN, TT, NT, NN | 2SM rectangular |
| 2SM | 256×256×256 | TN, TT, NT, NN | Large tile |

**Legend**: T=Transposed (Column Major), N=Non-transposed (Row Major)

### Alignment Requirements

| Type | Typical Alignment | Min Alignment | Notes |
|------|-------------------|---------------|-------|
| `nv_float4_t<float_e2m1_t>` | 32 elements | 16 elements | FP4 input |
| `mx_float6_t<float_e2m3_t>` | 16 elements | 8 elements | FP6 input |
| `bfloat16_t` | 8 elements | 4 elements | BF16 output |
| `float` | 4 elements | 1 element | FP32 C/D |
| Scale factors | 1 element | 1 element | Auto-managed |

### Cluster Shapes

| Cluster Shape | CTAs | Use Case | Multicast |
|---------------|------|----------|-----------|
| `Shape<_1,_1,_1>` | 1 | Single CTA, simple | No |
| `Shape<_2,_2,_1>` | 4 | Small problems | Yes |
| `Shape<_4,_4,_1>` | 16 | Large problems | Yes |
| `Shape<_8,_8,_1>` | 64 | Very large | Yes |

---

## Code Snippets

### Complete GEMM Setup

```cpp
// 1. Define types
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementC = float;
using ElementD = cutlass::bfloat16_t;
using ElementAccumulator = float;

constexpr int M = 2048, N = 2048, K = 2048;

// 2. Build collectives
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,
    Shape<_128,_128,_256>, Shape<_1,_1,_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, float,
    ElementC, cutlass::layout::RowMajor, 4,
    ElementD, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,
    ElementA, cutlass::layout::RowMajor, 32,
    ElementB, cutlass::layout::ColumnMajor, 32,
    ElementAccumulator,
    Shape<_128,_128,_256>, Shape<_1,_1,_1>,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

// 3. Build kernel
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// 4. Create arguments
typename Gemm::Arguments arguments{
  cutlass::gemm::GemmUniversalMode::kGemm,
  {M, N, K, 1},
  {A_ptr, stride_A, B_ptr, stride_B, SFA_ptr, layout_SFA, SFB_ptr, layout_SFB},
  {{alpha, beta}, C_ptr, stride_C, D_ptr, stride_D}
};

// 5. Run
Gemm gemm;
auto status = gemm.can_implement(arguments);
status = gemm.initialize(arguments, workspace_ptr);
status = gemm.run();
```

### Getting Scale Factor Layouts

```cpp
// Get config from kernel
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

// Create layouts
auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
    cute::make_shape(M, N, K, 1));
auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
    cute::make_shape(M, N, K, 1));

// Allocate scale factor tensors
auto sfa_size = size(filter_zeros(layout_SFA));
auto sfb_size = size(filter_zeros(layout_SFB));

cutlass::HostTensor<typename ElementA::ScaleFactorType, ...> tensor_SFA;
tensor_SFA.reset(cutlass::make_Coord(sfa_size));
```

### Type Conversions

```cpp
// Float to FP4
float val = 2.5f;
cutlass::float_e2m1_t fp4_val(val);

// FP4 to float
float reconstructed = float(fp4_val);

// Block-scaled value
cutlass::float_e2m1_t quantized(2.0f);
cutlass::float_ue4m3_t scale_factor(8.0f);
float actual = float(quantized) * float(scale_factor);  // = 16.0
```

---

## Common Patterns

### Pattern 1: FP4 Input, BF16 Output (Most Common)

```cpp
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using ElementAccumulator = float;

using FusionOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, float, ElementC, float>;
```

### Pattern 2: FP4 Input, FP4 Output with Scale Factor Generation

```cpp
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementC = float;
using ElementD = cutlass::float_e2m1_t;  // FP4 output
using ElementSFD = cutlass::float_ue8m0_t;
using ElementAccumulator = float;

using FusionOp = cutlass::epilogue::fusion::LinCombBlockScaleFactor<
    16,  // SFVectorSize
    ElementD, float, ElementSFD, cutlass::layout::RowMajor, ElementC>;
```

### Pattern 3: Mixed Precision (FP6 × FP4 → FP8)

```cpp
using ElementA = cutlass::mx_float6_t<cutlass::float_e2m3_t>;  // FP6
using ElementB = cutlass::mx_float4_t<cutlass::float_e2m1_t>;  // FP4
using ElementC = cutlass::float_e4m3_t;  // FP8 output
using ElementD = cutlass::float_e4m3_t;
using ElementAccumulator = float;
```

---

## Troubleshooting

### Common Errors and Solutions

#### Error: "TileSize and MNK Major does not met with MMA Mix 8-bit TMA load requirement"

**Cause**: Tile size or layout combination not supported for mixed narrow precision.

**Solution**:
- Use supported tile sizes: 128×128×256, 128×192×256, 256×128×256, 256×256×256
- Ensure layouts are compatible (typically TN: A=RowMajor, B=ColumnMajor)

#### Error: "Smem usage is too high. Can't create any SMEM buffers"

**Cause**: Not enough shared memory for pipeline stages.

**Solution**:
```cpp
// Use explicit stage count instead of auto
using StageCountType = cutlass::gemm::collective::StageCount<2>;  // Reduce stages

// Or increase carveout for epilogue
using StageCountType = cutlass::gemm::collective::StageCountAutoCarveout<
    static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage)) + 4096>;
```

#### Error: "Could not build a collective epilogue for given parameters"

**Cause**: Type combination not supported or missing fusion operation.

**Solution**:
- Verify `OperatorClass` is `OpClassBlockScaledTensorOp` for both mainloop and epilogue
- Check element types are compatible (e.g., don't mix FP4 input with incompatible output)
- Ensure fusion operation is correctly specified

#### Error: Alignment issues

**Cause**: Insufficient alignment for data type.

**Solution**:
```cpp
// FP4 types require at least 16-element alignment, recommend 32
constexpr int AlignmentA = 32;
constexpr int AlignmentB = 32;

// BF16 output requires at least 4-element alignment, recommend 8
constexpr int AlignmentD = 8;
```

### Verification Tips

1. **Check device capability**:
   ```cpp
   cudaDeviceProp props;
   cudaGetDeviceProperties(&props, 0);
   if (props.major < 10) {
     std::cerr << "Requires SM100 or later" << std::endl;
   }
   ```

2. **Verify tensor sizes**:
   ```cpp
   auto sfa_expected = (M / 128) * (K / 16);
   auto sfa_actual = size(filter_zeros(layout_SFA));
   assert(sfa_expected == sfa_actual);
   ```

3. **Test with minimal size**:
   ```cpp
   // Start with smallest supported size
   constexpr int M = 128, N = 128, K = 256;
   ```

### Performance Optimization

1. **Choose appropriate tile size**:
   - 1SM (128×N×K): Better for small problems or when occupancy is important
   - 2SM (256×N×K): Better for large problems with good occupancy

2. **Tune cluster shape**:
   - Larger clusters improve TMA multicast efficiency
   - But require larger problem sizes to maintain occupancy
   - Start with `Shape<_1,_1,_1>`, increase for large GEMMs

3. **Balance stage count**:
   - More stages hide latency but use more SMEM
   - Use `StageCountAuto` to let CUTLASS decide
   - Manually tune with `StageCount<N>` if needed

---

## File Locations Cheat Sheet

| Component | File |
|-----------|------|
| Narrow precision types | [include/cutlass/float_subbyte.h](../../include/cutlass/float_subbyte.h) |
| Block scale config | [include/cutlass/detail/sm100_blockscaled_layout.hpp](../../include/cutlass/detail/sm100_blockscaled_layout.hpp) |
| Mainloop builder | [include/cutlass/gemm/collective/builders/sm100_blockscaled_umma_builder.inl](../../include/cutlass/gemm/collective/builders/sm100_blockscaled_umma_builder.inl) |
| Epilogue builder | [include/cutlass/epilogue/collective/builders/sm100_builder.inl](../../include/cutlass/epilogue/collective/builders/sm100_builder.inl) |
| Fusion operations | [include/cutlass/epilogue/fusion/operations.hpp](../../include/cutlass/epilogue/fusion/operations.hpp) |
| GemmUniversal kernel | [include/cutlass/gemm/kernel/gemm_universal.hpp](../../include/cutlass/gemm/kernel/gemm_universal.hpp) |
| Main example | [examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu](../../examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu) |

---

## Additional Resources

- **Full Walkthrough**: [README.md](./README.md)
- **Minimal Examples**: [minimal_examples.md](./minimal_examples.md)
- **CUTLASS Documentation**: https://github.com/NVIDIA/cutlass
- **Blackwell PTX Guide**: https://docs.nvidia.com/cuda/parallel-thread-execution/

---

**Document Version**: 1.0
**Last Updated**: 2025-11-02
