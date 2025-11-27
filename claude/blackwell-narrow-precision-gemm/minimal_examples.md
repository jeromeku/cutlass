# Minimal Runnable Examples

This document provides self-contained, minimal examples for testing CUTLASS narrow precision components in isolation.

## Table of Contents

1. [Setup](#setup)
2. [Example 1: Testing Narrow Precision Types](#example-1-testing-narrow-precision-types)
3. [Example 2: Block Scale Layout Computation](#example-2-block-scale-layout-computation)
4. [Example 3: Simple Block-Scaled GEMM](#example-3-simple-block-scaled-gemm)
5. [Example 4: Epilogue Fusion Configuration](#example-4-epilogue-fusion-configuration)

---

## Setup

### Environment Requirements

- CUDA Toolkit 12.8 or later
- Blackwell GPU (SM100, SM101, or SM103)
- CUTLASS installed/cloned

### Compilation Template

```bash
#!/bin/bash
CUTLASS_DIR="/path/to/cutlass"
nvcc -std=c++17 \
  -I${CUTLASS_DIR}/include \
  -I${CUTLASS_DIR}/tools/util/include \
  -arch=sm_100 \
  -O3 \
  your_file.cu -o your_program
```

---

## Example 1: Testing Narrow Precision Types

### test_fp4_types.cu

This example tests the basic FP4 type operations and conversions.

```cpp
/***************************************************************************************************
 * Minimal example: Testing float_e2m1_t (FP4) types
 **************************************************************************************************/

#include <iostream>
#include <iomanip>
#include <vector>
#include "cutlass/float_subbyte.h"
#include "cutlass/numeric_conversion.h"

void print_fp4_range() {
  std::cout << "\n=== FP4 (E2M1) Representable Values ===" << std::endl;
  std::cout << "Format: 1 sign bit, 2 exponent bits, 1 mantissa bit" << std::endl;
  std::cout << "Exponent bias: 1" << std::endl;
  std::cout << "\nPositive values:" << std::endl;

  // Test all 8 positive bit patterns (sign bit = 0)
  std::vector<uint8_t> bit_patterns = {
    0b0000, // 0.0
    0b0001, // 0.5
    0b0010, // 1.0
    0b0011, // 1.5
    0b0100, // 2.0
    0b0101, // 3.0
    0b0110, // 4.0
    0b0111  // 6.0
  };

  for (auto bits : bit_patterns) {
    auto fp4_val = cutlass::float_e2m1_t::bitcast(bits);
    float float_val = float(fp4_val);
    std::cout << "  Bits: 0x" << std::hex << static_cast<int>(bits) << std::dec
              << " → " << std::setw(6) << float_val << std::endl;
  }
}

void test_conversions() {
  std::cout << "\n=== Float → FP4 Conversions ===" << std::endl;

  std::vector<float> test_values = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

  for (float val : test_values) {
    cutlass::float_e2m1_t fp4(val);
    float reconstructed = float(fp4);
    std::cout << "  " << std::setw(6) << val << " → FP4 → " << std::setw(6) << reconstructed
              << " (error: " << std::abs(val - reconstructed) << ")" << std::endl;
  }
}

void test_nv_float4_wrapper() {
  std::cout << "\n=== NV Float4 Wrapper Test ===" << std::endl;

  using NVF4 = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using DataType = typename NVF4::DataType;
  using ScaleFactorType = typename NVF4::ScaleFactorType;

  std::cout << "DataType: " << cutlass::sizeof_bits<DataType>::value << " bits" << std::endl;
  std::cout << "ScaleFactorType: " << cutlass::sizeof_bits<ScaleFactorType>::value << " bits" << std::endl;

  // Simulate block-scaled representation
  std::cout << "\nSimulating block-scaled value:" << std::endl;
  DataType quantized_val(2.0f);        // Quantized data
  ScaleFactorType scale_factor(8.0f);  // Scale factor

  float actual_value = float(quantized_val) * float(scale_factor);
  std::cout << "  Quantized: " << float(quantized_val) << std::endl;
  std::cout << "  Scale Factor: " << float(scale_factor) << std::endl;
  std::cout << "  Actual Value: " << actual_value << std::endl;
}

int main() {
  std::cout << "╔═══════════════════════════════════════════════════════╗" << std::endl;
  std::cout << "║  CUTLASS FP4 (float_e2m1_t) Type Testing             ║" << std::endl;
  std::cout << "╚═══════════════════════════════════════════════════════╝" << std::endl;

  print_fp4_range();
  test_conversions();
  test_nv_float4_wrapper();

  return 0;
}
```

### Compile and Run

```bash
nvcc -std=c++17 -I/path/to/cutlass/include test_fp4_types.cu -o test_fp4_types
./test_fp4_types
```

### Expected Output

```
╔═══════════════════════════════════════════════════════╗
║  CUTLASS FP4 (float_e2m1_t) Type Testing             ║
╚═══════════════════════════════════════════════════════╝

=== FP4 (E2M1) Representable Values ===
Format: 1 sign bit, 2 exponent bits, 1 mantissa bit
Exponent bias: 1

Positive values:
  Bits: 0x0 →    0.0
  Bits: 0x1 →    0.5
  Bits: 0x2 →    1.0
  Bits: 0x3 →    1.5
  Bits: 0x4 →    2.0
  Bits: 0x5 →    3.0
  Bits: 0x6 →    4.0
  Bits: 0x7 →    6.0

=== Float → FP4 Conversions ===
     0.0 → FP4 →    0.0 (error: 0)
     0.5 → FP4 →    0.5 (error: 0)
     1.0 → FP4 →    1.0 (error: 0)
     1.5 → FP4 →    1.5 (error: 0)
     2.0 → FP4 →    2.0 (error: 0)
     2.5 → FP4 →    2.0 (error: 0.5)
     3.0 → FP4 →    3.0 (error: 0)
     4.0 → FP4 →    4.0 (error: 0)
     5.0 → FP4 →    4.0 (error: 1)
     6.0 → FP4 →    6.0 (error: 0)
     7.0 → FP4 →    6.0 (error: 1)

=== NV Float4 Wrapper Test ===
DataType: 4 bits
ScaleFactorType: 8 bits

Simulating block-scaled value:
  Quantized: 2
  Scale Factor: 8
  Actual Value: 16
```

---

## Example 2: Block Scale Layout Computation

### test_blockscale_layout.cu

This example computes scale factor tensor shapes for different problem sizes.

```cpp
/***************************************************************************************************
 * Minimal example: Block scale layout computation
 **************************************************************************************************/

#include <iostream>
#include <iomanip>
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cute/tensor.hpp"

using namespace cute;

void test_scale_factor_layout(int M, int N, int K) {
  std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
  std::cout << "Problem Size: M=" << M << ", N=" << N << ", K=" << K << std::endl;
  std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;

  // Configure block scaling
  constexpr int SFVecSize = 16;  // Scale factor vector size
  using Config = cutlass::detail::Sm1xxBlockScaledConfig<SFVecSize>;

  // Block dimensions
  constexpr int Blk_MN = 128;  // Block size in M/N dimension
  constexpr int Blk_K = 16;    // Block size in K dimension

  // Get layouts for scale factors
  auto layout_SFA = Config::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
  auto layout_SFB = Config::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));

  // Compute sizes
  auto shape_A = make_shape(M, K);
  auto shape_B = make_shape(N, K);
  auto sfa_shape = shape(layout_SFA);
  auto sfb_shape = shape(layout_SFB);

  // Expected sizes
  int expected_sfa_m = (M + Blk_MN - 1) / Blk_MN;
  int expected_sfa_k = (K + Blk_K - 1) / Blk_K;
  int expected_sfb_n = (N + Blk_MN - 1) / Blk_MN;
  int expected_sfb_k = (K + Blk_K - 1) / Blk_K;

  std::cout << "\n┌─ Matrix A ─────────────────────────────────────┐" << std::endl;
  std::cout << "│  Shape: " << M << " × " << K << std::endl;
  std::cout << "│  Elements: " << (M * K) << std::endl;
  std::cout << "│  Size: " << ((M * K * 4) / 8) << " bytes (4-bit)" << std::endl;
  std::cout << "└────────────────────────────────────────────────┘" << std::endl;

  std::cout << "\n┌─ Scale Factor A (SFA) ─────────────────────────┐" << std::endl;
  std::cout << "│  Layout shape: " << sfa_shape << std::endl;
  std::cout << "│  Layout stride: " << stride(layout_SFA) << std::endl;
  std::cout << "│  Number of scale factors: " << size(filter_zeros(layout_SFA)) << std::endl;
  std::cout << "│  Expected: " << expected_sfa_m << " × " << expected_sfa_k
            << " = " << (expected_sfa_m * expected_sfa_k) << std::endl;
  std::cout << "│  Block coverage: " << Blk_MN << " rows × " << Blk_K << " cols" << std::endl;
  std::cout << "│  Elements per SF: " << (Blk_MN * Blk_K) << std::endl;
  std::cout << "└────────────────────────────────────────────────┘" << std::endl;

  std::cout << "\n┌─ Matrix B ─────────────────────────────────────┐" << std::endl;
  std::cout << "│  Shape: " << N << " × " << K << std::endl;
  std::cout << "│  Elements: " << (N * K) << std::endl;
  std::cout << "│  Size: " << ((N * K * 4) / 8) << " bytes (4-bit)" << std::endl;
  std::cout << "└────────────────────────────────────────────────┘" << std::endl;

  std::cout << "\n┌─ Scale Factor B (SFB) ─────────────────────────┐" << std::endl;
  std::cout << "│  Layout shape: " << sfb_shape << std::endl;
  std::cout << "│  Layout stride: " << stride(layout_SFB) << std::endl;
  std::cout << "│  Number of scale factors: " << size(filter_zeros(layout_SFB)) << std::endl;
  std::cout << "│  Expected: " << expected_sfb_n << " × " << expected_sfb_k
            << " = " << (expected_sfb_n * expected_sfb_k) << std::endl;
  std::cout << "│  Block coverage: " << Blk_MN << " rows × " << Blk_K << " cols" << std::endl;
  std::cout << "│  Elements per SF: " << (Blk_MN * Blk_K) << std::endl;
  std::cout << "└────────────────────────────────────────────────┘" << std::endl;

  // Compression ratio
  float data_size_bytes = (M * K + N * K) * 4.0f / 8.0f;
  float sf_size_bytes = (size(filter_zeros(layout_SFA)) + size(filter_zeros(layout_SFB))) * 1.0f;
  float total_size = data_size_bytes + sf_size_bytes;
  float compression_ratio = (M * K + N * K) * 4.0f / total_size;  // vs FP32

  std::cout << "\n┌─ Memory Usage Summary ─────────────────────────┐" << std::endl;
  std::cout << "│  Data size: " << data_size_bytes << " bytes" << std::endl;
  std::cout << "│  Scale factor size: " << sf_size_bytes << " bytes" << std::endl;
  std::cout << "│  Total size: " << total_size << " bytes" << std::endl;
  std::cout << "│  Compression vs FP32: " << compression_ratio << "×" << std::endl;
  std::cout << "└────────────────────────────────────────────────┘" << std::endl;
}

int main() {
  std::cout << "╔═══════════════════════════════════════════════════════╗" << std::endl;
  std::cout << "║  Block Scale Factor Layout Computation                ║" << std::endl;
  std::cout << "╚═══════════════════════════════════════════════════════╝" << std::endl;

  // Test different problem sizes
  test_scale_factor_layout(2048, 2048, 2048);
  test_scale_factor_layout(1024, 1024, 1024);
  test_scale_factor_layout(4096, 4096, 4096);
  test_scale_factor_layout(128, 128, 256);

  return 0;
}
```

### Compile and Run

```bash
nvcc -std=c++17 -I/path/to/cutlass/include test_blockscale_layout.cu -o test_blockscale_layout
./test_blockscale_layout
```

---

## Example 3: Simple Block-Scaled GEMM

### simple_fp4_gemm.cu

Minimal GEMM example using block-scaled FP4 types.

```cpp
/***************************************************************************************************
 * Minimal example: Simple FP4 block-scaled GEMM
 * Computes: D = A × B where A, B are block-scaled FP4
 **************************************************************************************************/

#include <iostream>
#include <vector>
#include <random>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// Simple configuration
constexpr int M = 128;
constexpr int N = 128;
constexpr int K = 256;

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

constexpr int AlignmentA = 32;
constexpr int AlignmentB = 32;
constexpr int AlignmentC = 8;
constexpr int AlignmentD = 8;

using MmaTileShape = Shape<_128, _128, _256>;
using ClusterShape = Shape<_1, _1, _1>;

// Build epilogue
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100,
    cutlass::arch::OpClassBlockScaledTensorOp,
    MmaTileShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    float,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

// Build mainloop
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100,
    cutlass::arch::OpClassBlockScaledTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    MmaTileShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

// Build kernel
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

void print_info() {
  std::cout << "\n╔═══════════════════════════════════════════════════════╗" << std::endl;
  std::cout << "║  Simple FP4 Block-Scaled GEMM Example                 ║" << std::endl;
  std::cout << "╚═══════════════════════════════════════════════════════╝" << std::endl;

  std::cout << "\nProblem Configuration:" << std::endl;
  std::cout << "  Matrix sizes: M=" << M << ", N=" << N << ", K=" << K << std::endl;
  std::cout << "  Element A: nv_float4_t<float_e2m1_t> (4-bit + scale)" << std::endl;
  std::cout << "  Element B: nv_float4_t<float_e2m1_t> (4-bit + scale)" << std::endl;
  std::cout << "  Element C/D: bfloat16" << std::endl;
  std::cout << "  Accumulator: float" << std::endl;
  std::cout << "  MMA Tile: 128×128×256" << std::endl;
}

int main() {
  print_info();

  // Check device
  cudaDeviceProp props;
  int device_id;
  cudaGetDevice(&device_id);
  cudaGetDeviceProperties(&props, device_id);

  if (props.major < 10) {
    std::cerr << "\n❌ This example requires SM100 (Blackwell) or later." << std::endl;
    std::cerr << "   Current device: SM" << props.major << props.minor << std::endl;
    return 0;
  }

  std::cout << "\n✓ Running on: " << props.name << " (SM" << props.major << props.minor << ")" << std::endl;

  // Get types from kernel
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;

  // Create strides and layouts
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

  using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));

  // Allocate host tensors
  cutlass::HostTensor<typename ElementA::DataType, cutlass::layout::PackedVectorLayout> tensor_A;
  cutlass::HostTensor<typename ElementA::ScaleFactorType, cutlass::layout::PackedVectorLayout> tensor_SFA;
  cutlass::HostTensor<typename ElementB::DataType, cutlass::layout::PackedVectorLayout> tensor_B;
  cutlass::HostTensor<typename ElementB::ScaleFactorType, cutlass::layout::PackedVectorLayout> tensor_SFB;
  cutlass::HostTensor<ElementC, cutlass::layout::PackedVectorLayout> tensor_C;
  cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> tensor_D;

  tensor_A.reset(cutlass::make_Coord(M * K));
  tensor_SFA.reset(cutlass::make_Coord(size(filter_zeros(layout_SFA))));
  tensor_B.reset(cutlass::make_Coord(N * K));
  tensor_SFB.reset(cutlass::make_Coord(size(filter_zeros(layout_SFB))));
  tensor_C.reset(cutlass::make_Coord(M * N));
  tensor_D.reset(cutlass::make_Coord(M * N));

  std::cout << "\n✓ Allocated tensors" << std::endl;
  std::cout << "  A: " << (M * K) << " elements" << std::endl;
  std::cout << "  SFA: " << size(filter_zeros(layout_SFA)) << " scale factors" << std::endl;
  std::cout << "  B: " << (N * K) << " elements" << std::endl;
  std::cout << "  SFB: " << size(filter_zeros(layout_SFB)) << " scale factors" << std::endl;

  // Initialize with dummy data (in real code, use proper initialization)
  // ... (initialize tensors) ...

  // Sync to device
  tensor_A.sync_device();
  tensor_SFA.sync_device();
  tensor_B.sync_device();
  tensor_SFB.sync_device();
  tensor_C.sync_device();

  // Create arguments
  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, 1},
    {tensor_A.device_data(), stride_A,
     tensor_B.device_data(), stride_B,
     tensor_SFA.device_data(), layout_SFA,
     tensor_SFB.device_data(), layout_SFB},
    {{1.0f, 0.0f},
     tensor_C.device_data(), stride_C,
     tensor_D.device_data(), stride_D}
  };

  // Initialize and run
  Gemm gemm;
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  auto status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "\n❌ Gemm::can_implement() failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
    return -1;
  }

  status = gemm.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "\n❌ Gemm::initialize() failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
    return -1;
  }

  std::cout << "\n✓ Kernel initialized" << std::endl;
  std::cout << "  Workspace size: " << workspace_size << " bytes" << std::endl;

  status = gemm.run();
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "\n❌ Gemm::run() failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
    return -1;
  }

  cudaDeviceSynchronize();

  std::cout << "\n✓ GEMM completed successfully!" << std::endl;

  return 0;
}

#else

int main() {
  std::cerr << "This example requires SM100 support (compile with -DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1)" << std::endl;
  return 0;
}

#endif
```

### Compile and Run

```bash
nvcc -std=c++17 \
  -I/path/to/cutlass/include \
  -I/path/to/cutlass/tools/util/include \
  -DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1 \
  -arch=sm_100 \
  simple_fp4_gemm.cu -o simple_fp4_gemm

./simple_fp4_gemm
```

---

## Example 4: Epilogue Fusion Configuration

### test_epilogue_fusion.cu

Tests epilogue fusion operation configuration.

```cpp
/***************************************************************************************************
 * Minimal example: Epilogue fusion configuration testing
 **************************************************************************************************/

#include <iostream>
#include "cutlass/epilogue/fusion/operations.hpp"

template <typename FusionOp>
void print_fusion_config(const char* name) {
  std::cout << "\n┌─ " << name << " ─" << std::endl;
  std::cout << "│  IsSourceSupported: " << FusionOp::IsSourceSupported << std::endl;
  std::cout << "│  IsBlockScaleSupported: " << FusionOp::IsBlockScaleSupported << std::endl;

  if constexpr (FusionOp::IsBlockScaleSupported) {
    std::cout << "│  SFVecSize: " << FusionOp::SFVecSize << std::endl;
    std::cout << "│  ElementBlockScaleFactor size: "
              << cutlass::sizeof_bits<typename FusionOp::ElementBlockScaleFactor>::value << " bits" << std::endl;
  }

  std::cout << "│  ElementOutput size: "
            << cutlass::sizeof_bits<typename FusionOp::ElementOutput>::value << " bits" << std::endl;
  std::cout << "│  ElementCompute: float" << std::endl;
  std::cout << "└─" << std::endl;
}

int main() {
  std::cout << "╔═══════════════════════════════════════════════════════╗" << std::endl;
  std::cout << "║  Epilogue Fusion Configuration Testing               ║" << std::endl;
  std::cout << "╚═══════════════════════════════════════════════════════╝" << std::endl;

  // Test 1: Standard linear combination (no block scaling)
  using LinearComb = cutlass::epilogue::fusion::LinearCombination<
      cutlass::bfloat16_t,  // ElementOutput
      float,                // ElementCompute
      cutlass::bfloat16_t,  // ElementSource
      float>;               // ElementScalar

  print_fusion_config<LinearComb>("LinearCombination");

  // Test 2: Linear combination with block scale factor generation
  using LinCombBS = cutlass::epilogue::fusion::LinCombBlockScaleFactor<
      16,                         // SFVectorSize
      cutlass::float_e2m1_t,      // ElementOutput (FP4)
      float,                      // ElementCompute
      cutlass::float_ue8m0_t,     // ElementBlockScaleFactor
      cutlass::layout::RowMajor,  // GmemLayoutTagScalefactor
      float>;                     // ElementSource

  print_fusion_config<LinCombBS>("LinCombBlockScaleFactor");

  // Test 3: With activation function
  using LinCombEltAct = cutlass::epilogue::fusion::LinCombEltAct<
      cutlass::epilogue::thread::ReLU,  // Activation
      cutlass::bfloat16_t,               // ElementOutput
      float,                             // ElementCompute
      cutlass::bfloat16_t,               // ElementSource
      float>;                            // ElementScalar

  print_fusion_config<LinCombEltAct>("LinCombEltAct (ReLU)");

  return 0;
}
```

### Compile and Run

```bash
nvcc -std=c++17 -I/path/to/cutlass/include test_epilogue_fusion.cu -o test_epilogue_fusion
./test_epilogue_fusion
```

---

## Next Steps

After running these minimal examples:

1. **Modify and experiment**: Change problem sizes, tile shapes, data types
2. **Add verification**: Compare results against reference implementations
3. **Profile performance**: Use NSight Compute to analyze kernel performance
4. **Explore variants**: Try different layouts (TN, NT, TT), cluster shapes, scheduling policies

For complete working examples, see:
- [examples/72_blackwell_narrow_precision_gemm/](../../examples/72_blackwell_narrow_precision_gemm/)
- [test/unit/gemm/device/sm100_blockscaled_tensorop_gemm/](../../test/unit/gemm/device/sm100_blockscaled_tensorop_gemm/)

---

**Document Version**: 1.0
**Last Updated**: 2025-11-02
