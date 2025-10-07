
#include <iostream>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"

#include "cutlass/numeric_types.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"

// #include "cutlass_unit_test.h"

// #include "gemm_testbed_3x.hpp"

#define CUTLASS_ARCH_MMA_SM90_SUPPORTED

using namespace cute;

///////////////////////////////////////////////////////////////////////////////

int main() {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::bfloat16_t, LayoutA, 8,
      cutlass::bfloat16_t, LayoutB, 8,
      float,
      Shape<_64,_128,_64>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelCpAsyncWarpSpecializedPingpong
    >::CollectiveOp;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      Shape<_64,_128,_64>, Shape<_1,_1,_1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::bfloat16_t, LayoutC, 8,
      cutlass::bfloat16_t, LayoutC, 8,
      cutlass::epilogue::NoSmemWarpSpecialized
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
//   EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

// TEST(SM90_Device_Gemm_bf16t_bf16n_bf16n_align4_tensor_op_gmma_f32_warpspecialized_pingpong, 64x128x64) {
//   using LayoutA = cutlass::layout::RowMajor;
//   using LayoutB = cutlass::layout::ColumnMajor;
//   using LayoutC = cutlass::layout::ColumnMajor;

//   using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
//       cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
//       cutlass::bfloat16_t, LayoutA, 4,
//       cutlass::bfloat16_t, LayoutB, 4,
//       float,
//       Shape<_64,_128,_64>, Shape<_1,_1,_1>,
//       cutlass::gemm::collective::StageCountAuto,
//       cutlass::gemm::KernelCpAsyncWarpSpecializedPingpong
//     >::CollectiveOp;

//   using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
//       cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
//       Shape<_64,_128,_64>, Shape<_1,_1,_1>,
//       cutlass::epilogue::collective::EpilogueTileAuto,
//       float, float,
//       cutlass::bfloat16_t, LayoutC, 4,
//       cutlass::bfloat16_t, LayoutC, 4,
//       cutlass::epilogue::collective::EpilogueScheduleAuto
//     >::CollectiveOp;

//   using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
//       Shape<int,int,int,int>,
//       CollectiveOp,
//       CollectiveEpilogue
//   >;

//   using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
//   EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
// }

// ///////////////////////////////////////////////////////////////////////////////

// TEST(SM90_Device_Gemm_bf16t_bf16n_bf16n_align2_tensor_op_gmma_f32_warpspecialized_pingpong, 64x128x64) {
//   using LayoutA = cutlass::layout::RowMajor;
//   using LayoutB = cutlass::layout::ColumnMajor;
//   using LayoutC = cutlass::layout::ColumnMajor;

//   using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
//       cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
//       cutlass::bfloat16_t, LayoutA, 2,
//       cutlass::bfloat16_t, LayoutB, 2,
//       float,
//       Shape<_64,_128,_64>, Shape<_1,_1,_1>,
//       cutlass::gemm::collective::StageCountAuto,
//       cutlass::gemm::KernelCpAsyncWarpSpecializedPingpong
//     >::CollectiveOp;

//   using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
//       cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
//       Shape<_64,_128,_64>, Shape<_1,_1,_1>,
//       cutlass::epilogue::collective::EpilogueTileAuto,
//       float, float,
//       cutlass::bfloat16_t, LayoutC, 2,
//       cutlass::bfloat16_t, LayoutC, 2,
//       cutlass::epilogue::collective::EpilogueScheduleAuto
//     >::CollectiveOp;

//   using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
//       Shape<int,int,int,int>,
//       CollectiveOp,
//       CollectiveEpilogue
//   >;

//   using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
//   EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
// }

// ///////////////////////////////////////////////////////////////////////////////

// #endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)














