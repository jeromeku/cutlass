// #include <__clang_cuda_builtin_vars.h>
#define KERNEL_DBG_TRACE false

// #include "../common/cutlass_unit_test.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/tensor.hpp>

#include <cutlass/cluster_launch.hpp>
#include <cutlass/util/reference/host/gemm.h>

#include "cutlass/core_io.h"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/print_error.hpp"

#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "testbed.h"

// #define CUTLASS_UNIT_TEST_PIPELINE true
using namespace cute;
using namespace cutlass;

//////////////////// KERNEL /////////////////////////

template <uint32_t Stages, typename PingPongBarrier> struct SharedStorage {
  typename cutlass::PipelineTmaAsync<Stages>::SharedStorage pipeline_storage;
  typename PingPongBarrier::SharedStorage pingpong_storage;
};

template <typename ClusterShape, uint32_t Stages> struct CollectiveSimulation {
  using MainloopPipeline = typename cutlass::PipelineTmaAsync<Stages>;
  using PipelineState = typename cutlass::PipelineState<Stages>;

  CUTLASS_DEVICE
  static void dma_wg_simulation(MainloopPipeline pipeline,
                                PipelineState tile_start_state_pipe,
                                uint32_t const num_iterations) {
    uint32_t const per_cta_bytes = sizeof(uint32_t);
    int warp_idx_in_warpgroup =
        __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
    int lane_predicate = cute::elect_one_sync();
    if (warp_idx_in_warpgroup == 0 && lane_predicate) {

      int tma_k_prologue = min(Stages, num_iterations);

      // Simulating Prologue TMA Loads
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tma_k_prologue; ++i) {
        printf("PRODUCER PROLOGUE ACQUIRING: %d of %d; %d %d\n", i,
               tma_k_prologue, tile_start_state_pipe.index(),
               tile_start_state_pipe.phase());
        pipeline.producer_acquire(tile_start_state_pipe);
        // Simulating cp.async.bulk.tensor behavior
        pipeline.producer_commit(tile_start_state_pipe, per_cta_bytes);
        ++tile_start_state_pipe;
      }
      int tma_k_iter = num_iterations - tma_k_prologue;
      printf("PRODUCER PROLOGUE Finished Acquiring: %d = %d - %d; %d %d\n",
             tma_k_iter, num_iterations, tma_k_prologue,
             tile_start_state_pipe.index(), tile_start_state_pipe.phase());

      PipelineState wr_pipe = tile_start_state_pipe;
      // Simulating Mainloop TMA Loads
      CUTE_NO_UNROLL
      for (; tma_k_iter > 0; --tma_k_iter) {
        printf("PRODUCER MAINLOOP ACQUIRING: %d %d %d\n", tma_k_iter,
               wr_pipe.index(), wr_pipe.phase());

        pipeline.producer_acquire(wr_pipe);

        // Simulating cp.async.bulk.tensor behavior
        pipeline.producer_commit(wr_pipe, per_cta_bytes);

        // Advance write stage
        ++wr_pipe;
      }
    }
  }

  CUTLASS_DEVICE
  static void math_wg_simulation(MainloopPipeline pipeline,
                                 PipelineState tile_start_state_pipe,
                                 uint32_t const num_iterations, int *data_ptr) {
    PipelineState rd_pipe = tile_start_state_pipe;
    PipelineState release_pipe = rd_pipe;

    // simulates accumulators + extra reg. pressure
    int arr[168];

    // Init Shared Memory read stages & PhaseBit
    static constexpr uint32_t K_PIPE_MMAS = 1;
    static_assert(K_PIPE_MMAS < Stages, "ERROR : Too many MMAs in flight");
    int warp_group_thread_idx = threadIdx.x % NumThreadsPerWarpGroup;
    // Total number of gemm iterations
    auto gemm_k_iterations = num_iterations;

    // Simulating Prologue MMAs
    int mma_k_prologue = min(K_PIPE_MMAS, gemm_k_iterations);

    CUTLASS_PRAGMA_UNROLL
    for (int iter = 0; iter < mma_k_prologue; ++iter) {
      if (warp_group_thread_idx == 0)
        printf("CONSUMER PROLOGUE WAITING: %d of %d, %d %d\n", iter,
               mma_k_prologue, rd_pipe.index(), rd_pipe.phase());

      pipeline.consumer_wait(rd_pipe);
      if (warp_group_thread_idx == 0)
        printf("CONSUMER PROLOGUE ARRIVING: %d of %d, %d %d\n", iter,
               mma_k_prologue, rd_pipe.index(), rd_pipe.phase());

      warpgroup_arrive();
      // GMMA would typically happen here

      ++rd_pipe;
    }
    gemm_k_iterations -= mma_k_prologue;

    // Simulating Mainloop MMAs
    CUTLASS_PRAGMA_NO_UNROLL
    for (; gemm_k_iterations > 0; --gemm_k_iterations) {
      if (warp_group_thread_idx == 0)
        printf(
            "CONSUMER MAINLOOP WAIT: %d, rd pipe: %d %d release pipe: %d %d\n",
            gemm_k_iterations, rd_pipe.index(), rd_pipe.phase(),
            release_pipe.index(), release_pipe.phase());

      /// Wait on the rd_pipe stage / phase
      pipeline.consumer_wait(rd_pipe);
      if (warp_group_thread_idx == 0)
        printf("CONSUMER MAINLOOP FINISHED WAITING: %d, rd pipe: %d %d release "
               "pipe: %d %d\n",
               gemm_k_iterations, rd_pipe.index(), rd_pipe.phase(),
               release_pipe.index(), release_pipe.phase());

      warpgroup_arrive();
      // GMMA would typically happen here

      // Dummy op - which will never happen
      // But simulates high register usage.
      CUTE_UNROLL
      for (int i = 0; i < 168; ++i) {
        if (threadIdx.x > 384) {
          arr[i] += data_ptr[i];
        }
      }

      if (warp_group_thread_idx == 0)
        printf(
            "CONSUMER RELEASING PIPE: %d, rd pipe: %d %d release pipe: %d %d\n",
            gemm_k_iterations, rd_pipe.index(), rd_pipe.phase(),
            release_pipe.index(), release_pipe.phase());

      pipeline.consumer_release(release_pipe);

      // Advance stages
      ++rd_pipe;
      ++release_pipe;
    }

    // Dummy op - which will never happen
    CUTE_UNROLL
    for (int i = 0; i < 168; ++i) {
      if (threadIdx.x > 384) {
        data_ptr[i] = arr[i];
      }
    }

    // Tail Loop
    for (int i = 0; i < K_PIPE_MMAS; ++i) {
      pipeline.consumer_release(release_pipe);
      ++release_pipe;
    }
  }
};

struct KernelParams {
  uint32_t num_iterations;
  int tiles_per_cluster;
  int *data_ptr;
};

// Goal of this kernel is to complete deadlock-free
template <typename ClusterShape, uint32_t Stages>
__launch_bounds__(384, 1) __global__
    static void pipeline_device(KernelParams params) {
  extern __shared__ char shared_memory[];
  using MainloopPipeline = typename cutlass::PipelineTmaAsync<Stages>;
  using PipelineState = typename cutlass::PipelineState<Stages>;

  /* One for Mainloop and one for Epilogue */
  constexpr int StagesPerMathWarpGroup = 2;
  constexpr int MathWarpGroupCountPersistent = 2;
  using PingPongBarrier =
      typename cutlass::OrderedSequenceBarrier<StagesPerMathWarpGroup,
                                               MathWarpGroupCountPersistent>;

  using SharedStorage = SharedStorage<Stages, PingPongBarrier>;
  SharedStorage &shared_storage =
      *reinterpret_cast<SharedStorage *>(shared_memory);

  [[maybe_unused]] auto cta_layout = Layout<ClusterShape>{}; // (m,n) -> cta_id
  int warp_group_idx =
      __shfl_sync(0xffffffff, threadIdx.x / NumThreadsPerWarpGroup, 0);
  int warp_group_thread_idx = threadIdx.x % NumThreadsPerWarpGroup;
  dim3 block_id_in_cluster = cute::block_id_in_cluster();

  auto cluster_shape = ClusterShape{};

  // #Producers = #RowsInCluster + #ColsInCluster - 1
  uint32_t const NumProducers =
      cute::size<0>(cluster_shape) + cute::size<1>(cluster_shape) - 1;
  uint32_t const TmaTransactionBytes =
      static_cast<uint32_t>(sizeof(uint32_t) * NumProducers);

  // mbarrier.init
  typename MainloopPipeline::Params pipeline_params;
  pipeline_params.transaction_bytes = TmaTransactionBytes;
  if (warp_group_idx == 0) {
    pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
  } else {
    pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
  }
  pipeline_params.is_leader = warp_group_thread_idx == 0;
  pipeline_params.num_consumers = NumThreadsPerWarpGroup;

  MainloopPipeline pipeline(shared_storage.pipeline_storage, pipeline_params,
                            cluster_shape);
  PipelineState tile_start_state_pipe;

  int tiles_per_cluster = params.tiles_per_cluster;

  /* Offset pipeline start state for Math WG 2 */
  if (warp_group_idx == 2) {
    // Update pipeline state for next persistent tile
    tile_start_state_pipe.advance(params.num_iterations);
    tiles_per_cluster--;
  }

  typename PingPongBarrier::Params pingpong_params;
  pingpong_params.group_id =
      warp_group_idx - 1; // Since DMA Warp Group Idx 0 will not participate
  pingpong_params.group_size =
      NumThreadsPerWarpGroup; // Number of threads / participants in a group
  PingPongBarrier math_wg_barrier(shared_storage.pingpong_storage,
                                  pingpong_params);

  __syncthreads();

  // Ensure All CTAs in Cluster have completed init before issuing commits
  cute::cluster_arrive_relaxed();
  cute::cluster_wait();

  // Producer/DMA WarpGroup
  if (warp_group_idx == 0) {
    cutlass::arch::warpgroup_reg_dealloc<40>();
    // For the DMA (prologue) - we start with an opposite phase - since we skip
    // all waits i.e., we know that the buffer is indeed empty
    PipelineState tile_prologue_state_pipe =
        make_producer_start_state<MainloopPipeline>();
    while (tiles_per_cluster > 0) {
      if (warp_group_thread_idx == 0) {
        printf("PRODUCER::%d %d %d: tile_per_cluster %d index %d phase "
               "%d...Starting TMA\n",
               blockIdx.x, blockIdx.y, threadIdx.x, tiles_per_cluster,
               tile_prologue_state_pipe.index(),
               tile_prologue_state_pipe.phase());
      }
      CollectiveSimulation<ClusterShape, Stages>::dma_wg_simulation(
          pipeline, tile_prologue_state_pipe, params.num_iterations);
      if (warp_group_thread_idx == 0) {
        printf("PRODUCER::%d %d %d: tile_per_cluster %d...Finished TMA\n",
               blockIdx.x, blockIdx.y, threadIdx.x, tiles_per_cluster);
      }

      // Update pipeline state for next persistent tile
      tile_prologue_state_pipe.advance(params.num_iterations);
      tiles_per_cluster--;
    }
  }
  // Math WarpGropups
  if (warp_group_idx == 1 || warp_group_idx == 2) {
    cutlass::arch::warpgroup_reg_alloc<232>();
    while (tiles_per_cluster > 0) {
      if (warp_group_idx == 1 && warp_group_thread_idx == 0)
        printf("CONSUMER::%d %d %d: tile_per_cluster %d index %d phase "
               "%d...Waiting for producer\n",
               blockIdx.x, blockIdx.y, threadIdx.x, tiles_per_cluster,
               tile_start_state_pipe.index(), tile_start_state_pipe.phase());
      // MMA
      math_wg_barrier.wait();
      if (warp_group_idx == 1 && warp_group_thread_idx == 0)
        printf(
            "CONSUMER::%d %d %d: tile_per_cluster %d...Starting simulation\n",
            blockIdx.x, blockIdx.y, threadIdx.x, tiles_per_cluster);

      CollectiveSimulation<ClusterShape, Stages>::math_wg_simulation(
          pipeline, tile_start_state_pipe, params.num_iterations,
          params.data_ptr);
      math_wg_barrier.arrive();
      if (warp_group_idx == 1 && warp_group_thread_idx == 0)
        printf(
            "CONSUMER::%d %d %d: tile_per_cluster %d...Finished simulation\n",
            blockIdx.x, blockIdx.y, threadIdx.x, tiles_per_cluster);

      // Epilogue
      math_wg_barrier.wait();
      // Simulates long running stage
      if (warp_group_idx == 1 && warp_group_thread_idx == 0)
        printf("CONSUMER::%d %d %d: tile_per_cluster %d...Starting epilogue\n",
               blockIdx.x, blockIdx.y, threadIdx.x, tiles_per_cluster);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
      __nanosleep(1000);
#endif

      math_wg_barrier.arrive();
      if (warp_group_idx == 1 && warp_group_thread_idx == 0)
        printf("CONSUMER::%d %d %d: tile_per_cluster %d...Finished `epilogue\n",
               blockIdx.x, blockIdx.y, threadIdx.x, tiles_per_cluster);

      // Update pipeline state for next persistent tile
      tile_start_state_pipe.advance(params.num_iterations * 2);
      tiles_per_cluster -= 2;
    }
  }

  // Makes sure remote SMEM doesn't get destroyed
  cute::cluster_arrive_relaxed();
  cute::cluster_wait();
}
/////////////////////////////////////////////////////

/// Device NT GMMA + TMA specialized
template <uint32_t Stages_, typename ClusterShape_> struct PipelineTest {

  //
  // Data members
  //
  static constexpr uint32_t Stages = Stages_;
  static constexpr uint32_t kBlockSize = 128 * 3;
  using ClusterShape = ClusterShape_;

  //
  // Methods
  //

  // Run CuTe GEMM kernel
  cudaError_t run(uint32_t const kNumIters, cudaStream_t stream = 0) {

    float elapsed_ms = 0.0f;
    // Pipeline (multistage pipeline)
    auto cluster_shape =
        Shape<Int<ClusterShape::kM>, Int<ClusterShape::kN>, _1>{};

    //
    // Configure and launch
    //
    int iterations = 1;
    cudaEvent_t events[2];
    cudaError_t result;

    for (cudaEvent_t &event : events) {
      result = cudaEventCreate(&event);
      if (result != cudaSuccess) {
        std::cerr << "Error: Failed to create event.";
        return result;
      }
    }

    result = cudaEventRecord(events[0]);

    if (result != cudaSuccess) {
      std::cerr << "Error: Failed to record start event.";
      return result;
    }

    for (int iter = 0; iter < iterations; ++iter) {
      constexpr int StagesPerMathWarpGroup = 2;
      constexpr int MathWarpGroupCountPersistent = 2;
      int smem_size = int(
          sizeof(SharedStorage<Stages, typename cutlass::OrderedSequenceBarrier<
                                           StagesPerMathWarpGroup,
                                           MathWarpGroupCountPersistent>>));

      result = cudaFuncSetAttribute(
          pipeline_device<decltype(cluster_shape), Stages>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

      // Launch a single Cluster, with kBlockSize threads per CTA
      dim3 dimCluster(size<0>(cluster_shape), size<1>(cluster_shape), 1);
      dim3 dimGrid(size<0>(cluster_shape), size<1>(cluster_shape), 1);
      dim3 dimBlock(kBlockSize, 1, 1);

      int tiles_per_cluster = (kNumIters % 10) + 1;
      printf("Persistent version: Tiles per Cluster = %d\n", tiles_per_cluster);

      const void *kernel =
          (const void *)pipeline_device<decltype(cluster_shape), Stages>;
      KernelParams params{kNumIters, tiles_per_cluster, nullptr};
      void *kernel_params[] = {&params};
      cutlass::ClusterLauncher::launch(dimGrid, dimCluster, dimBlock, smem_size,
                                       stream, kernel, kernel_params);
    }

    result = cudaEventRecord(events[1]);

    if (result != cudaSuccess) {
      std::cerr << "Error: Failed to record stop event.";
      return result;
    }

    result = cudaDeviceSynchronize();

    if (result != cudaSuccess) {
      std::cerr << "Error: cudaDeviceSynchronize() failed" << std::endl;
      return result;
    }

    result = cudaEventElapsedTime(&elapsed_ms, events[0], events[1]);

    if (result != cudaSuccess) {
      std::cerr << "Failed to create event.";
      return result;
    }

    for (cudaEvent_t &event : events) {
      (void)cudaEventDestroy(event);
    }

    return cudaSuccess;
  }
};

int main() {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static constexpr uint32_t Stages = 2;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  bool passed = testbed.verification();
  printf("Test: Stages %d, Cluster size: ", Stages);
  cute::print(Shape<_1, _1, _1>{});
  printf(": %s\n", passed ? "Passed" : "Failed");
}

#if 0
TEST(SM90_Verify_PipelineTmaAsync_WS_Persistent, Cluster1x1_Stage5) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static constexpr uint32_t Stages = 5;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineTmaAsync_WS_Persistent, Cluster1x1_Stage10) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static constexpr uint32_t Stages = 10;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineTmaAsync_WS_Persistent, Cluster2x2_Stage2) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<2, 2, 1>;
  static constexpr uint32_t Stages = 2;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineTmaAsync_WS_Persistent, Cluster2x2_Stage5) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<2, 2, 1>;
  static constexpr uint32_t Stages = 5;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineTmaAsync_WS_Persistent, Cluster2x2_Stage7) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<2, 2, 1>;
  static constexpr uint32_t Stages = 7;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineTmaAsync_WS_Persistent, Cluster4x4_Stage2) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<4, 4, 1>;
  static constexpr uint32_t Stages = 2;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineTmaAsync_WS_Persistent, Cluster4x4_Stage7) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<4, 4, 1>;
  static constexpr uint32_t Stages = 7;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineTmaAsync_WS_Persistent, Cluster2x1_Stage2) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<2, 1, 1>;
  static constexpr uint32_t Stages = 2;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineTmaAsync_WS_Persistent, Cluster2x1_Stage7) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<2, 1, 1>;
  static constexpr uint32_t Stages = 7;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineTmaAsync_WS_Persistent, Cluster1x2_Stage2) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<1, 2, 1>;
  static constexpr uint32_t Stages = 2;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineTmaAsync_WS_Persistent, Cluster1x2_Stage7) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<1, 2, 1>;
  static constexpr uint32_t Stages = 7;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineTmaAsync_WS_Persistent, Cluster4x1_Stage2) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<4, 1, 1>;
  static constexpr uint32_t Stages = 2;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineTmaAsync_WS_Persistent, Cluster4x1_Stage7) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<4, 1, 1>;
  static constexpr uint32_t Stages = 7;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineTmaAsync_WS_Persistent, Cluster1x4_Stage2) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<1, 4, 1>;
  static constexpr uint32_t Stages = 2;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineTmaAsync_WS_Persistent, Cluster1x4_Stage7) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<1, 4, 1>;
  static constexpr uint32_t Stages = 7;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineTmaAsync_WS_Persistent, Cluster2x4_Stage2) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<2, 4, 1>;
  static constexpr uint32_t Stages = 2;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineTmaAsync_WS_Persistent, Cluster2x4_Stage7) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<2, 4, 1>;
  static constexpr uint32_t Stages = 7;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineTmaAsync_WS_Persistent, Cluster4x2_Stage2) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<4, 2, 1>;
  static constexpr uint32_t Stages = 2;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_PipelineTmaAsync_WS_Persistent, Cluster4x2_Stage7) {
  Options options;
  using ClusterShape = cutlass::gemm::GemmShape<4, 2, 1>;
  static constexpr uint32_t Stages = 7;
  using Test = PipelineTest<Stages, ClusterShape>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}
#endif
