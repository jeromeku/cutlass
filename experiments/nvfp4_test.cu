
#include <iostream>

#include "cute/arch/copy_sm90_tma.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits.hpp"
#include "cute/util/type_traits.hpp"
#include "cute/numeric/numeric_types.hpp"
#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/tensor_ref.h"

// #include "cutlass/util/command_line.h"
// #include "cutlass/util/distribution.h"
// #include "cutlass/util/host_tensor.h"
// #include "cutlass/util/packed_stride.hpp"
// #include "cutlass/util/tensor_view_io.h"
// #include "cutlass/util/reference/device/gemm.h"
// #include "cutlass/util/reference/device/tensor_compare.h"
// #include "cutlass/util/reference/host/tensor_fill.h"
// #include "cutlass/util/reference/host/gett.hpp"
// #include "cutlass/util/reference/host/tensor_norm.h"
// #include "cutlass/util/reference/host/tensor_compare.h"

#include "cute/arch/mma_sm100_desc.hpp" // cute::UMMA::Major
#include "cute/arch/mma_sm100_umma.hpp" // SM100_*MMA_SS_*
#include "cute/arch/tmem_allocator_sm100.hpp"
#include "cute/atom/copy_traits_sm100_tma.hpp" // SM100_TMA_*SM_LOAD_*
#include "cute/atom/mma_traits_sm100.hpp"      // UMMA::Layout_MN_SW*
#include "cute/numeric/integral_constant.hpp" // is_static_v, cute::integral_constant
#include "cute/util/type_traits.hpp"          // cute::alignment_of_v

#include <iostream>

// #include "helper.h"

using namespace cute;

// A matrix configuration
using ElementA =
    cutlass::nv_float4_t<cutlass::float_e2m1_t>; // Element type for A matrix
                                                 // operand
using LayoutATag =
    cutlass::layout::RowMajor; // Layout type for A matrix operand
constexpr int AlignmentA = 32; // Memory access granularity/alignment of A
                               // matrix in units of elements (up to 16 bytes)

// B matrix configuration
using ElementB =
    cutlass::nv_float4_t<cutlass::float_e2m1_t>; // Element type for A matrix
                                                 // operand
using LayoutBTag =
    cutlass::layout::ColumnMajor; // Layout type for B matrix operand
constexpr int AlignmentB = 32;    // Memory access granularity/alignment of B
                               // matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
using ElementD = cutlass::float_e2m1_t; // Element type for D matrix operand
using ElementSFD =
    cutlass::float_ue8m0_t; // Element type for SFB matrix operand
using ElementC = float;     // Element type for C matrix operand
using LayoutCTag =
    cutlass::layout::RowMajor; // Layout type for C matrix operand
using LayoutDTag =
    cutlass::layout::RowMajor; // Layout type for D matrix operand
using LayoutSFDTag =
    LayoutDTag; // Layout type for SFD should be same as D matrix operand

constexpr int AlignmentD =
    128 / cutlass::sizeof_bits<
              ElementD>::value; // Memory access granularity/alignment of C
                                // matrix in units of elements (up to 16 bytes)
constexpr int AlignmentC =
    128 / cutlass::sizeof_bits<
              ElementC>::value; // Memory access granularity/alignment of C
                                // matrix in units of elements (up to 16 bytes)

// Kernel functional config
using ElementAccumulator = float;     // Element type for internal accumulation
using ElementCompute = float;         // Element type for internal accumulation
using ArchTag = cutlass::arch::Sm100; // Tag indicating the minimum SM that
                                      // supports the intended feature
using OperatorClass =
    cutlass::arch::OpClassBlockScaledTensorOp; // Operator class tag

// Kernel Perf config
using MmaTileShape = Shape<_128, _128, _256>; // MMA's tile size
using ClusterShape =
    Shape<_1, _1, _1>; // Shape of the threadblocks in a cluster

constexpr int InputSFVectorSize = 16;
constexpr int OutputSFVectorSize = InputSFVectorSize;

// D = alpha * acc + beta * C
//      With BlockScaleFactor generation.
using FusionOperation = cutlass::epilogue::fusion::LinCombBlockScaleFactor<
    OutputSFVectorSize, ElementD, ElementCompute, ElementSFD, LayoutSFDTag,
    ElementC>;

using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
        ElementAccumulator, ElementC, LayoutCTag, AlignmentC, ElementD,
        LayoutDTag, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto, // Epilogue
                                                             // schedule policy
        FusionOperation>::CollectiveOp;

using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB,
        LayoutBTag, AlignmentB, ElementAccumulator, MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto // Kernel schedule policy.
                                                      // Auto or using targeted
                                                      // scheduling policy
        >::CollectiveOp;

using Mainloop = cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB,
    LayoutBTag, AlignmentB, ElementAccumulator, MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto // Kernel schedule policy.
                                                  // Auto or using targeted
                                                  // scheduling policy
    >;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, // Indicates ProblemShape
    CollectiveMainloop, CollectiveEpilogue, void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

auto is_base =
    cute::is_base_of_v<cutlass::gemm::collective::KernelScheduleAuto,
                       cutlass::gemm::collective::KernelScheduleAuto>;

// Reference device GEMM implementation type
using StrideA = typename Gemm::GemmKernel::StrideA;
using LayoutA = decltype(cute::make_layout(make_shape(0, 0, 0), StrideA{}));
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::
    LayoutSFA; // Scale Factor tensors have an interleaved layout. Bring Layout
               // instead of stride.
using StrideB = typename Gemm::GemmKernel::StrideB;
using LayoutB = decltype(cute::make_layout(make_shape(0, 0, 0), StrideB{}));
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::
    LayoutSFB; // Scale Factor tensors have an interleaved layout. Bring Layout
               // instead of stride.
using StrideC = typename Gemm::GemmKernel::StrideC;
using LayoutC = decltype(cute::make_layout(make_shape(0, 0, 0), StrideC{}));
using StrideD = typename Gemm::GemmKernel::StrideD;
using LayoutD = decltype(cute::make_layout(make_shape(0, 0, 0), StrideD{}));

using FusionOp = typename Gemm::EpilogueOutputOp;
constexpr bool IsBlockScaleSupported = FusionOp::IsBlockScaleSupported;
using SfdOutputCfg =
    cutlass::detail::Sm1xxBlockScaledOutputConfig<OutputSFVectorSize>;
using LayoutSFD = typename SfdOutputCfg::LayoutSF;

//
// Data members
//

/// Initialization
StrideA stride_A;
LayoutA layout_A;
LayoutSFA layout_SFA;
StrideB stride_B;
LayoutB layout_B;
LayoutSFB layout_SFB;
StrideC stride_C;
LayoutC layout_C;
StrideD stride_D;
LayoutD layout_D;
LayoutSFD layout_SFD;

uint64_t seed;

template <typename T> void print_cute(const char *msg, const T& obj) {
  printf("%s:\n", msg);
  cute::print(obj);
  printf("\n");
}

// template <typename PermutationMNK, typename AtomShape_MNK, int I>
//   CUTE_HOST_DEVICE constexpr
//   auto
//   permutation_mnk() {
//     static_assert(0 <= I && I < 3);
//     auto perm = get<I>(PermutationMNK{});
//     return conditional_return(is_underscore<decltype(perm)>{}, size<I>(AtomShape_MNK{}) * size<I+1>(get_thr_layout_vmnk()), perm);
//   }

// template <class ATensor, class Permutation>
// CUTE_HOST_DEVICE constexpr auto thrfrg_A(ATensor &&atensor, Permutation permutation_mnk) {
//   // Reorder the tensor for the TiledAtom
//   auto t_tile = make_tile(permutation_mnk<0>(), permutation_mnk<2>());
//   auto t_tensor = logical_divide(atensor, t_tile); // (PermM,PermK)

//   // Tile the tensor for the Atom
//   auto a_tile = make_tile(make_layout(size<0>(AtomShape_MNK{})),
//                           make_layout(size<2>(AtomShape_MNK{})));
//   auto a_tensor =
//       zipped_divide(t_tensor, a_tile); // ((AtomM,AtomK),(RestM,RestK))

//   // Transform the Atom mode from (M,K) to (Thr,Val)
//   auto tv_tensor =
//       a_tensor.compose(AtomLayoutA_TV{}, _); // ((ThrV,FrgV),(RestM,RestK))

//   // Tile the tensor for the Thread
//   auto thr_tile =
//       make_tile(_, make_tile(make_layout(size<1>(thr_layout_vmnk_)),
//                              make_layout(size<3>(thr_layout_vmnk_))));
//   auto thr_tensor = zipped_divide(
//       tv_tensor, thr_tile); // ((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK)))

//   return thr_tensor;
// }

int main(int argc, char const **args) {
  using namespace cutlass::gemm::collective::detail;

  using SF_t = cutlass::gemm::collective::detail::blockscaled::blockscaled_type<
      cutlass::gemm::collective::KernelScheduleAuto, ElementA>;
  using ElementA_t = SF_t::data_type;
  using SFA = typename SF_t::sf_type;

  constexpr auto UmmaMajorA =
      cutlass::gemm::collective::detail::tag_to_umma_major_A<LayoutATag>();
  constexpr auto UmmaMajorB =
      cutlass::gemm::collective::detail::tag_to_umma_major_B<LayoutBTag>();
  using BuilderScheduleTag = cutlass::gemm::collective::KernelScheduleAuto;
  using ElementPairA = ElementA;
  using ElementPairB = ElementB;
  auto correct_input_types =
      cutlass::gemm::collective::detail::blockscaled::check_input_datatypes<
          BuilderScheduleTag, ElementPairA, ElementPairB, UmmaMajorA,
          UmmaMajorB>();

  constexpr auto SfVectorSize = SF_t::SfVectorSize;

  constexpr auto is_2sm = Mainloop::is_2sm;
  constexpr auto Instr =
      cutlass::gemm::collective::detail::blockscaled::select_instr<
          ElementPairA, ElementPairB, ElementAccumulator, UmmaMajorA,
          UmmaMajorB, BuilderScheduleTag>();
  constexpr auto UseMxf8f6f4 = Instr ==
                             cutlass::gemm::collective::detail::blockscaled::
                                 BlockScaledInstr::MXF4F6F8;
  auto is_type_erased =
      cute::is_same_v<ElementA, cutlass::type_erased_dynamic_float4_t>;

  using ElementAMma =
      decltype(cutlass::gemm::collective::detail::
                   sm1xx_kernel_input_element_to_mma_input_element<
                       ElementA_t,
                       Instr == cutlass::gemm::collective::detail::blockscaled::
                                    BlockScaledInstr::MXF4F6F8>());
  using ElementBMma = ElementAMma;

  using TileMma_t = cutlass::gemm::collective::detail::TrivialBlockscaledMma<
      ElementPairA, ElementPairB, ElementAccumulator, MmaTileShape,
      ClusterShape, UmmaMajorA, UmmaMajorB, Instr, BuilderScheduleTag, is_2sm>;
  constexpr int M = cute::size<0>(MmaTileShape{});
  constexpr int N = cute::size<1>(MmaTileShape{});

  using TiledMma = TileMma_t::type;
  using TileShape_MNK = MmaTileShape;

  using MmaOp =
      cute::SM100_MMA_MXF4_SS<ElementAMma, ElementBMma, ElementAccumulator, SFA,
                              M, N, SfVectorSize, UmmaMajorA, UmmaMajorB>;
  auto mma_atom = MMA_Atom<MmaOp>{};
  using MmaTraits = MMA_Traits<MmaOp>;
  using MmaScaleFactor = MmaTraits::MMA_ScaleFactor;
  constexpr auto mma_op = MmaOp{};
  auto traits = MmaTraits{};
  using IDescriptor = decltype(mma_atom.idesc_);
  auto bitsMmaA = cute::sizeof_bits_v<ElementAMma>;
  auto bitsA = cute::sizeof_bits_v<ElementA_t>;
  using PermutationMNK = Tile<Underscore, Underscore, Underscore>;
  auto permutation = PermutationMNK{};
  auto tiledmma = make_tiled_mma(mma_atom, {}, {});

  // auto tiledmma = make_tiled_mma(mma_op);
  // using TiledMma = decltype(tiledmma);
  constexpr auto SFVectorSize = TiledMma::SFVecSize;

  using AtomThrID = typename TiledMma::AtomThrID;
  using Sm1xxBlkScaledConfig =
      cutlass::detail::Sm1xxBlockScaledConfig<SFVectorSize>;
  using Sm1xxBlkScaledChunk =
      cutlass::detail::Sm1xxBlockScaledBasicChunk<SFVectorSize>;
  using Blk_MN = typename Sm1xxBlkScaledChunk::Blk_MN;
  using Blk_SF = typename Sm1xxBlkScaledChunk::Blk_SF;
  using SfAtom = typename Sm1xxBlkScaledChunk::SfAtom;

  using LayoutSF = decltype(blocked_product(
      SfAtom{}, make_layout(make_shape(int32_t(0), int32_t(0), int32_t(0)),
                            make_stride(int32_t(0), _1{}, int32_t(0)))));

  // ((MMA_TILE_M,MMA_TILE_K), MMA_M, MMA_K)
  using MmaShapeA_MK = decltype(partition_shape_A(
      TiledMma{}, make_shape(cute::size<0>(TileShape_MNK{}),
                             cute::size<2>(TileShape_MNK{}))));

  auto tileshape = TileShape_MNK{};
  auto shape_MK = make_shape(size<0>(tileshape), size<2>(tileshape));
  auto mma = TiledMma{};
  using AtomThrID = TiledMma::AtomThrID;
  // using AtomLayoutA_TV = TiledMma::AtomLayoutA_TV;
  // print_cute("AtomLayoutA_TV", AtomLayoutA_TV{});

  // Layout dummy = make_layout(shape(shape_MK));
  // // dummy.compose()
  // auto dummy_tv = mma.thrfrg_A(dummy);
  // print_cute("dummy_tv", dummy_tv);
  // // Slice+rearrange like partition_A
  // auto dummy_v = dummy_tv(Int<0>{}, make_coord(_, repeat<rank(dummy)>(_)));
  // print_cute("dummy_v", dummy_v);
  // auto mmashapeA = shape(dummy_v);
  using ClusterShape_MNK = ClusterShape;
  using GmemTiledCopyA = decltype(cutlass::gemm::collective::detail::sm100_cluster_shape_to_tma_atom_A(
      ClusterShape_MNK{}, AtomThrID{}));
  auto tiledcopyA = GmemTiledCopyA{};
  using ElementAMma_SmemAllocType = cute::conditional_t<UseMxf8f6f4, uint8_t, ElementAMma>;
  using ElementType = ElementAMma_SmemAllocType;
  print_cute("ElementAMMa Type", cute::sizeof_bits_v<ElementType>);
  using BlockTileA_M = decltype(cute::size<0,0>(MmaShapeA_MK{}) * cute::size<1>(MmaShapeA_MK{}));
  using BlockTileA_K = decltype(cute::size<0,1>(MmaShapeA_MK{}) * cute::size<2>(MmaShapeA_MK{}));
  using SmemLayoutAtomA = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
      UmmaMajorA, ElementAMma_SmemAllocType, BlockTileA_M, BlockTileA_K>());
  auto layoutK_SW128_atom = UMMA::Layout_K_SW128_Atom<ElementAMma_SmemAllocType>{};
  auto layoutK_SW128_atom_bf16 = UMMA::Layout_K_SW128_Atom<cute::bfloat16_t>{};

  using Blk_MN    = typename Sm1xxBlkScaledConfig::Blk_MN;
  using Blk_SF    = typename Sm1xxBlkScaledConfig::Blk_SF; 
  using Blk_Elems = decltype(Blk_MN{} * Blk_SF{});
  using SmemLayoutAtomSFA = decltype(Sm1xxBlkScaledConfig::deduce_smem_layoutSFA(TiledMma{}, TileShape_MNK{}));
  using SmemLayoutAtomsA = decltype(cute::make_tuple(SmemLayoutAtomA{}, SmemLayoutAtomSFA{}));
  auto tileMmaK = TiledMma::K;
  using Blk_MN    = typename Sm1xxBlkScaledChunk::Blk_MN;
  using Blk_SF    = typename Sm1xxBlkScaledChunk::Blk_SF; 
  using Blk_Elems = decltype(Blk_MN{} * Blk_SF{});
  

  using StageCountType = cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogue::SharedStorage))>;

  constexpr int SFVecSize = Sm1xxBlkScaledConfig::SFVecSize;
  constexpr int MMA_NSF = TiledMma::K / SFVecSize;
  
  using mnBasicBlockShape  =  Shape<_32,_4>;
  using mnBasicBlockStride = Stride<_16,_4>;
  using kBasicBlockShape  = Shape<Int<SFVecSize>, Int<MMA_NSF>>;
  using kBasicBlockStride = Stride<_0, _1>;

  using TL_VMNK = typename TiledMma::ThrLayoutVMNK;
  constexpr TL_VMNK tl_vmnk{};
  constexpr int MMA_M = cute::size<0>(TileShape_MNK{}) / cute::size<0>(tl_vmnk);
  using mma_SFA_shape  = decltype( make_shape( prepend(Int<MMA_M>{}/Blk_MN{},  mnBasicBlockShape{}),  kBasicBlockShape{}));
  using mma_SFA_stride = decltype(make_stride( prepend(          Blk_Elems{}, mnBasicBlockStride{}), kBasicBlockStride{}));
  using sSFA_shape     = decltype( make_shape( mma_SFA_shape{}, _1{},   make_shape( Blk_SF{}/Int<MMA_NSF>{}, Int<size<2>(TileShape_MNK{}) / SFVecSize / Blk_SF{}>{})));
  using sSFA_stride    = decltype(make_stride(mma_SFA_stride{}, _0{},  make_stride(          Int<MMA_NSF>{},                   Int<MMA_M /Blk_MN{} * Blk_Elems{}>{})));
  using SmemLayoutAtomSFA = decltype(make_layout(sSFA_shape{}, sSFA_stride{}));
  
  using StrideA = cutlass::gemm::TagToStrideA_t<LayoutATag>;
  using StrideB = cutlass::gemm::TagToStrideB_t<LayoutBTag>;
  using InternalStrideA  = cute::remove_pointer_t<StrideA>;
  using InternalStrideB  = cute::remove_pointer_t<StrideB>;
  using InternalLayoutSFA = decltype(Sm1xxBlkScaledConfig::deduce_layoutSFA());
  using InternalLayoutSFB = decltype(Sm1xxBlkScaledConfig::deduce_layoutSFB());
  using LayoutSFA = cute::conditional_t<cute::is_same_v<InternalStrideA, StrideA>, InternalLayoutSFA, InternalLayoutSFA *>;
  using LayoutSFB = cute::conditional_t<cute::is_same_v<InternalStrideB, StrideB>, InternalLayoutSFB, InternalLayoutSFB *>;
  constexpr int SchedulerPipelineStageCount = Mainloop::SchedulerPipelineStageCount;
  constexpr int AccumulatorPipelineStageCount = Mainloop::AccumulatorPipelineStageCount;

  constexpr bool IsArrayOfPointersGemm = cute::is_base_of_v<cutlass::gemm::KernelSchedulePtrArrayBlockScaledGemmSm100, BuilderScheduleTag>;
  
  using KernelSmemCarveout_t = cutlass::gemm::collective::detail::Sm100DenseGemmTmaUmmaCarveout<
      ClusterShape_MNK,
      AccumulatorPipelineStageCount,
      SchedulerPipelineStageCount,
      cutlass::gemm::collective::detail::CLCResponseSize,
      IsArrayOfPointersGemm,
      4 // 4 Tensor maps for A, SFA, B and SFB
    >;
  using PipelineStorage_t = typename cutlass::PipelineUmmaAsync<AccumulatorPipelineStageCount>::SharedStorage;
  constexpr auto AccumulatorStorage = sizeof(PipelineStorage_t);
  constexpr int NumCLCResponses = SchedulerPipelineStageCount;
  constexpr auto CLCPipelineStorage = sizeof(typename cutlass::PipelineCLCFetchAsync<SchedulerPipelineStageCount, ClusterShape_MNK>::SharedStorage) * NumCLCResponses;
  
  constexpr int KernelSmemCarveout = KernelSmemCarveout_t::KernelSmemCarveout;
  constexpr int ReducedSmemCapacityBytes = cutlass::gemm::collective::detail::sm100_smem_capacity_bytes - KernelSmemCarveout;
  
  using ElementBMma_SmemAllocType = Mainloop::ElementBMma_SmemAllocType;
  using SmemTileShape = Mainloop::SmemTileShape;
  using SmemLayoutAtomSFB = Mainloop::SmemLayoutAtomSFB;
  
  using TileShapeSFA = SmemLayoutAtomSFA;
  using TileShapeSFB = SmemLayoutAtomSFB;
  constexpr auto filteredSFA = filter_zeros(TileShapeSFA{});
  constexpr auto stage_sfa_bytes = size(filter_zeros(TileShapeSFA{}));
  constexpr auto stage_sfb_bytes = size(filter_zeros(TileShapeSFB{}));
  print_cute("TileShapeSFA", TileShapeSFA{});
  print_cute("FilteredSFA", filteredSFA);
  print_cute("stage_sfa_bytes", stage_sfa_bytes);
  // ReducedSmem = Total Smem - (Scheduler + Pipeline Smem)
  // NumStages = (ReducedSmem - EpilogueSmem) // (SmemA + SmemB + SmemSFA + SFBSmemSFB) 
  constexpr int PipelineStages = cutlass::gemm::collective::detail::sm100_compute_stage_count_or_override_blockscaled<
    ReducedSmemCapacityBytes, ElementAMma_SmemAllocType, ElementBMma_SmemAllocType, SmemTileShape, SmemLayoutAtomSFA, SmemLayoutAtomSFB>(StageCountType{});

  using DispatchPolicy = cutlass::gemm::MainloopSm100TmaUmmaWarpSpecializedBlockScaled<
          PipelineStages,
          SchedulerPipelineStageCount,
          AccumulatorPipelineStageCount,
          ClusterShape_MNK
      >;
  using Schedule = cutlass::gemm::KernelTmaWarpSpecializedBlockScaledSm100<SchedulerPipelineStageCount, AccumulatorPipelineStageCount>;

  auto dispatch = DispatchPolicy{};
  using ElementSF = Mainloop::ElementSF;
  using StridePairA = Mainloop::StridePairA;
  using StridePairB = Mainloop::StridePairB;
  using GmemTiledCopyPairA = Mainloop::GmemTiledCopyPairA;
  using GmemTiledCopyPairB = Mainloop::GmemTiledCopyPairB;
  using SmemLayoutAtomsB = Mainloop::SmemLayoutAtomsB;

  using CollectiveOp = cutlass::gemm::collective::CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      cute::tuple<ElementA, ElementSF>,
      StridePairA,
      cute::tuple<ElementB, ElementSF>,
      StridePairB,
      TiledMma,
      GmemTiledCopyPairA,
      SmemLayoutAtomsA,
      void,
      cute::identity,
      GmemTiledCopyPairB,
      SmemLayoutAtomsB,
      void,
      cute::identity
    >;
  auto mma = CollectiveOp::ElementA;
  print_cute("Blk_MN", Blk_MN{});
  print_cute("Blk_SF", Blk_SF{});
  print_cute("SFVectorSize", SFVecSize);
  print_cute("MMA_NSF", MMA_NSF);
  print_cute("tl_vmnk", tl_vmnk);
  print_cute("MMA_M", MMA_M);
  print_cute("mma_SFA_shape", mma_SFA_shape{});
  print_cute("mma_SFA_stride", mma_SFA_stride{});
  print_cute("sSFA_shape", sSFA_shape{});
  print_cute("sSFA_stride", sSFA_stride{});
  print_cute("SmemLayoutAtomSFA", SmemLayoutAtomSFA{});

  print_cute("LayoutSFA", LayoutSFA{});
  print_cute("LayoutSFB", LayoutSFB{});
  
  print_cute("Tile Shape", TileShape_MNK{});
  print_cute("TiledMma", TiledMma{});
  print_cute("MmaShapeA_MK", MmaShapeA_MK{});
  print_cute("BlockTileA_M", BlockTileA_M{});
  print_cute("BlockTileA_K", BlockTileA_K{});
  print_cute("ElementAMma_SmemAllocType", ElementAMma_SmemAllocType{});
  print_cute("SmemLayoutAtomA", SmemLayoutAtomA{});
  print_cute("Layout_K_SW128_Atom", layoutK_SW128_atom);
  print_cute("LayoutK bf16", layoutK_SW128_atom_bf16);

}

/////////////////////////////////////////////////////////////////////////////////////////////////
