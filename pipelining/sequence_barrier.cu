
// #include "cutlass_unit_test.h"
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
#include "cutlass/pipeline/pipeline.hpp"
#include "testbed.h"

using namespace cute;
constexpr int SLEEP_DURATION = 10000;
//////////////////// KERNEL /////////////////////////
// Sequence barrier:
/*
Stages (Depth) => MMA / Epilogue
Length => Num MMA groups
OrderedSequenceBarrier<Stages, Length>
Each iteration:
MathWG0 waits on barriers 0 & 2 for MMA / Epilogue respectively 
MathWG1 waits on barriers 1 & 3 for MMA / Epi, respectively
MathWG0 arrives on barriers 1 & 3 to signal to WG1 that it is done with MMA / Epi, respectively; vice versa

Each WG holds a PipelineState that resets after 2 stages (MMA & Epi)
- Phase is flipped after NumStages arrivals so that the MMA and Epi barriers are kept in sync -- these two barriers share the same PipelineState whose Stage and index
determine when phase is flipped.
- E.g., on the first iteration, WG0 starts with phase = 1
- Since there are no pending barriers, waits on MMA (stage 1) and Epi (stage 2) should completely immediately
- If index resets after each arrive (each WG calls arrive after both MMA and Epi), WG0 would block on wait for epi barrier

For Producer <-> Consumer sync
producer waits on empty, expect_tx (arrive) on full
- stages = num pipeline stages
            Full                        Empty
stage  full index  full phase  empty index empty phase
0          0            1           0                  
1
2
...
0
OrderedSequenceBarrier(SharedStorage& storage, Params const& params) :
      params_(params),
      barrier_ptr_(&storage.barrier_[0][0]),
      // Group 0 - starts with an opposite phase
      stage_({0, params.group_id == 0, 0}) {
Params params_;
Barrier *barrier_ptr_;
PipelineState<SequenceDepth> stage_;


template<int SequenceDepth, int SequenceLength>
struct OrderedSequenceBarrierSharedStorage {
  using Barrier = cutlass::arch::ClusterBarrier;
  Barrier barrier_[SequenceDepth][SequenceLength];
};
*/

#define PRINT_WAIT(label, tid_in_group, group_idx, i, index, phase) \
    if (tid_in_group == 0) { \
      printf("%s::(%d,%d): iter = %d, index = %d, phase = %d, waited on barrier = %d\n", label, group_idx, threadIdx.x, i, index, phase, index * 2 + group_idx); \
    } \

#define PRINT_ARRIVAL(label, tid_in_group, group_idx, i, index, phase, arrival_idx) \
    if (tid_in_group == 0) { \
      printf("%s::(%d,%d): iter = %d, index = %d, phase = %d, arrived on barrier = %d\n", label, group_idx, threadIdx.x, i, index, phase, arrival_idx * 2 + (group_idx + 1) % 2); \
    } \

template <typename OrderedSequencer> struct SharedStorage {
  typename OrderedSequencer::SharedStorage storage;
};

// Goal of this kernel is to complete deadlock-free
template <int Stages, int GroupCount, int ThreadsPerGroup>
__global__ static void ordered_sequence_device(uint32_t const num_iterations) {

  extern __shared__ char shared_memory[];
  using SequenceBarrier =
      typename cutlass::OrderedSequenceBarrier<Stages, GroupCount>;
  using SmemStorage = SharedStorage<SequenceBarrier>;

  SmemStorage &shared_storage = *reinterpret_cast<SmemStorage *>(shared_memory);

  int group_idx = threadIdx.x / ThreadsPerGroup;

  typename SequenceBarrier::Params params;
  params.group_id = group_idx; // sequence ID
  params.group_size =
      ThreadsPerGroup; // Number of threads / participants in a group

  SequenceBarrier barrier(shared_storage.storage, params);
  int thread_idx_in_group = threadIdx.x % ThreadsPerGroup;

  // Ensure All CTAs in Cluster have completed init before issuing commits
  __syncthreads();
  cute::cluster_arrive_relaxed();
  cute::cluster_wait();

  CUTLASS_PRAGMA_NO_UNROLL
  for (int i = 0; i < num_iterations; ++i) {
    if (thread_idx_in_group == 0) printf("START_ITERATION i=%d tid=%d\n", i, threadIdx.x);

    barrier.wait(); // Mainloop barrier
    // STAGE 1 CODE...
    PRINT_WAIT("MAINLOOP::START", thread_idx_in_group, group_idx, i, barrier.stage_.index(), barrier.stage_.phase());

    // Simulates long running stage
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    __nanosleep(SLEEP_DURATION);
#endif
   int arrival_idx = barrier.stage_.index();
   barrier.arrive();
   PRINT_ARRIVAL("MAINLOOP::DONE", thread_idx_in_group, group_idx, i, barrier.stage_.index(), barrier.stage_.phase(), arrival_idx);

    barrier.wait(); // Epilogue Barrier
  // STAGE 2 CODE...

  PRINT_WAIT("EPILOGUE::START", thread_idx_in_group, group_idx, i, barrier.stage_.index(), barrier.stage_.phase());
  // Simulates long running stage
  #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
      __nanosleep(100000);
  #endif
    arrival_idx = barrier.stage_.index();
    barrier.arrive();
    PRINT_ARRIVAL("EPILOGUE::DONE", thread_idx_in_group, group_idx, i, barrier.stage_.index(), barrier.stage_.phase(), arrival_idx);

  }

  // To make sure remote SMEM doesn't get destroyed
  cute::cluster_arrive();
  cute::cluster_wait();
}
/////////////////////////////////////////////////////

template <uint32_t Stages_, uint32_t GroupCount_> struct PipelineTest {

  //
  // Data members
  //
  static constexpr uint32_t ThreadsPerGroup = 128;
  static constexpr uint32_t BlockSize = GroupCount_ * ThreadsPerGroup;
  static constexpr uint32_t Stages = Stages_;
  static constexpr uint32_t GroupCount = GroupCount_;
  using SequenceBarrier =
      typename cutlass::OrderedSequenceBarrier<Stages, GroupCount>;
  using SmemStorage = SharedStorage<SequenceBarrier>;

  //
  // Methods
  //

  // Run CuTe GEMM kernel
  cudaError_t run(uint32_t const kNumIters, cudaStream_t stream = nullptr) {

    // Pipeline (multistage pipeline)
    auto cluster_shape = Shape<_1, _1, _1>{};

    //
    // Configure and launch
    //
    int iterations = 1;
    cudaError_t result;

    for (int iter = 0; iter < iterations; ++iter) {

      int smem_size = int(sizeof(SmemStorage));

      result = cudaFuncSetAttribute(
          ordered_sequence_device<Stages, GroupCount, ThreadsPerGroup>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

      // Launch a single Cluster, with 128 thread per CTA
      dim3 dimCluster(size<0>(cluster_shape), size<1>(cluster_shape),
                      size<2>(cluster_shape));
      dim3 dimGrid(size<0>(cluster_shape), size<1>(cluster_shape), 1);
      dim3 dimBlock(BlockSize, 1, 1);

      const void *kernel = (const void *)
          ordered_sequence_device<Stages, GroupCount, ThreadsPerGroup>;
      int iters = kNumIters;
      void *kernel_params[] = {reinterpret_cast<void *>(&iters)};
      cutlass::ClusterLauncher::launch(dimGrid, dimCluster, dimBlock, smem_size,
                                       stream, kernel, kernel_params);

    } // profiling loop ends

    result = cudaDeviceSynchronize();

    if (result != cudaSuccess) {
      std::cerr << "Error: cudaDeviceSynchronize() failed" << std::endl;
      return result;
    }

    return cudaSuccess;
  }
};

int main() {
  Options options;
  static constexpr uint32_t GroupCount = 2;
  static constexpr uint32_t Stages = 2;
  using Test = PipelineTest<Stages, GroupCount>;
  Testbed<Test> testbed(options);
  bool passed = testbed.verification();
  printf("Passed? %s\n", passed ? "True" : "False");
}

#if 0
TEST(SM90_Verify_OrderedSequence, Depth_2_Length_3) {
  Options options;
  static constexpr uint32_t GroupCount = 3;
  static constexpr uint32_t Stages = 2;
  using Test = PipelineTest<Stages, GroupCount>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_OrderedSequence, Depth_2_Length_4) {
  Options options;
  static constexpr uint32_t GroupCount = 4;
  static constexpr uint32_t Stages = 2;
  using Test = PipelineTest<Stages, GroupCount>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}

TEST(SM90_Verify_OrderedSequence, Depth_2_Length_5) {
  Options options;
  static constexpr uint32_t GroupCount = 5;
  static constexpr uint32_t Stages = 2;
  using Test = PipelineTest<Stages, GroupCount>;
  Testbed<Test> testbed(options);
  EXPECT_TRUE(testbed.verification());
}
#endif
