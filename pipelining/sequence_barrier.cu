
// #include "cutlass_unit_test.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>
#include <cute/arch/cluster_sm90.hpp> 

#include <cutlass/util/reference/host/gemm.h>
#include <cutlass/cluster_launch.hpp>

#include "cutlass/core_io.h"

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"

#include "testbed.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/arch/barrier.h"
#include "cute/arch/cluster_sm90.hpp"

using namespace cute;
constexpr int SLEEP_DURATION = 10;
//////////////////// KERNEL /////////////////////////

template<typename OrderedSequencer>
struct SharedStorage
{
  typename OrderedSequencer::SharedStorage storage;
};

// Goal of this kernel is to complete deadlock-free
template<int Stages, int GroupCount, int ThreadsPerGroup>
__global__ static
void ordered_sequence_device(uint32_t const num_iterations)
{

  extern __shared__ char shared_memory[];
  using SequenceBarrier = typename cutlass::OrderedSequenceBarrier<Stages, GroupCount>;
  using SmemStorage = SharedStorage<SequenceBarrier>;

  SmemStorage& shared_storage = *reinterpret_cast<SmemStorage*>(shared_memory);

  int group_idx = threadIdx.x / ThreadsPerGroup;

  typename SequenceBarrier::Params params;
  params.group_id = group_idx;              // sequence ID
  params.group_size = ThreadsPerGroup;      // Number of threads / participants in a group

  SequenceBarrier barrier(shared_storage.storage, params);

  // Ensure All CTAs in Cluster have completed init before issuing commits
  __syncthreads();
  cute::cluster_arrive_relaxed();  
  cute::cluster_wait();

  CUTLASS_PRAGMA_NO_UNROLL
  for (int i = 0; i < num_iterations; ++i){

    barrier.wait();
    // STAGE 1 CODE...
    #ifndef NDEBUG
    int thread_idx_in_group = threadIdx.x % ThreadsPerGroup;
    if (thread_idx_in_group == 0) {
      printf("STAGE 0 : Group_IDX : %d, id = %d, iter = %d, tidx = %d\n", group_idx, params.group_id, i, threadIdx.x);
    }
    #endif
    // Simulates long running stage
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    __nanosleep(SLEEP_DURATION);
    #endif
    barrier.arrive();

    barrier.wait();
    // STAGE 2 CODE...
    #ifndef NDEBUG
    if (thread_idx_in_group == 0) {
      printf("STAGE 1 : Group_IDX : %d, id = %d, iter = %d, tidx = %d\n", group_idx, params.group_id, i, threadIdx.x);
    }
    #endif
    // Simulates long running stage
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    __nanosleep(100000);
    #endif
    barrier.arrive();
  }

  // To make sure remote SMEM doesn't get destroyed
  cute::cluster_arrive();  
  cute::cluster_wait();  
}
/////////////////////////////////////////////////////

template<uint32_t Stages_, uint32_t GroupCount_>
struct PipelineTest {

  //
  // Data members
  //
  static constexpr uint32_t ThreadsPerGroup = 128;
  static constexpr uint32_t BlockSize = GroupCount_ * ThreadsPerGroup;
  static constexpr uint32_t Stages = Stages_;
  static constexpr uint32_t GroupCount = GroupCount_;
  using SequenceBarrier = typename cutlass::OrderedSequenceBarrier<Stages, GroupCount>;
  using SmemStorage = SharedStorage<SequenceBarrier>;

  //
  // Methods
  //

  // Run CuTe GEMM kernel
  cudaError_t run(uint32_t const kNumIters,
                  cudaStream_t stream = nullptr) {

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
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);

      // Launch a single Cluster, with 128 thread per CTA
      dim3 dimCluster(size<0>(cluster_shape), size<1>(cluster_shape), size<2>(cluster_shape));    
      dim3 dimGrid(size<0>(cluster_shape), size<1>(cluster_shape), 1);    
      dim3 dimBlock(BlockSize,1,1);

      const void* kernel = (const void*)ordered_sequence_device<Stages, GroupCount, ThreadsPerGroup>;
      int iters = kNumIters;
      void* kernel_params[] = {reinterpret_cast<void*>(&iters)};
      cutlass::ClusterLauncher::launch(dimGrid, dimCluster, dimBlock, smem_size, stream, kernel, kernel_params);
  
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
  printf("Passed? %s\n", passed ? "True": "False");
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
