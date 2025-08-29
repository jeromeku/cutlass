#include <cstdio>
#include <cuda/ptx>
#include <cuda/barrier>
#include <cooperative_groups.h>
#include <iostream>
#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>

using namespace cute;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while(0)

// Function to check and print last CUDA error
void checkLastCudaError(const char* msg = "CUDA Error") {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(error) << std::endl;
    } else {
        std::cout << msg << ": No error detected" << std::endl;
    }
}

template <class ElementType, class SmemLayout>
struct SharedStorage
{
  cute::ArrayEngine<ElementType, cute::cosize_v<SmemLayout>> smem;
  alignas(16) cute::uint64_t tma_load_mbar[1];
};

template <typename T, typename VecLayout>
__global__ void kernel()
{
  using cuda::ptx::scope_cluster;
  using cuda::ptx::sem_acquire;
  using cuda::ptx::sem_release;
  using cuda::ptx::space_cluster;
  using cuda::ptx::space_shared;
  dim3 cluster_dims = cluster_grid_dims();
  if(thread0()){
    printf("Cluster dims (%d, %d)\n", cluster_dims.x, cluster_dims.y);
  }
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<T, VecLayout>;
  SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(shared_memory);

  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();

  using barrier_t = cuda::barrier<cuda::thread_scope_block>;

#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ int receive_buffer[4];
  __shared__ barrier_t bar;
  // only thread 0 arrives
  init(&bar, 1);

  // Sync cluster to ensure remote barrier is initialized.
  cluster.sync();

  // Get address of remote cluster barrier:
  unsigned int other_block_rank = cluster.block_rank() ^ 1;
  uint64_t *remote_bar = cluster.map_shared_rank(cuda::device::barrier_native_handle(bar), other_block_rank);
  int *remote_buffer = cluster.map_shared_rank(&receive_buffer[0], other_block_rank);

  // Arrive on local barrier:
  uint64_t arrival_token;
  if (threadIdx.x == 0)
  {
    // Thread 0 arrives and indicates it expects to receive a certain number of bytes as well
    arrival_token = cuda::ptx::mbarrier_arrive_expect_tx(sem_release, scope_cluster, space_shared, cuda::device::barrier_native_handle(bar), sizeof(receive_buffer));
  }
  // } else {
  //   arrival_token = cuda::ptx::mbarrier_arrive(sem_release, scope_cluster, space_shared, cuda::device::barrier_native_handle(bar));
  // }
#if defined(DEBUG)
  if (threadIdx.x == 0)
  {
    printf("[block %d] arrived with expected tx count = %llu, sending to block rank: %u\n", cluster.block_rank(), sizeof(receive_buffer), other_block_rank);
  }
#endif

  // Send bytes to remote buffer, arriving on remote barrier
  if (threadIdx.x == 0)
  {
    cuda::ptx::st_async(remote_buffer, {int(cluster.block_rank()), 2, 3, 4}, remote_bar);
  }

#if defined(DEBUG)
  if (threadIdx.x == 0)
  {
    printf("[block %d] -> %u: st_async to %p, %p\n",
           cluster.block_rank(),
           other_block_rank,
           remote_buffer,
           remote_bar);
  }
#endif

  // Wait on local barrier:
  while (!cuda::ptx::mbarrier_try_wait(sem_acquire, scope_cluster, cuda::device::barrier_native_handle(bar), arrival_token))
  {
  }

#if defined(DEBUG)
  // Print received values:
  if (threadIdx.x == 0)
  {
    printf(
        "[block %d] receive_buffer = { %d, %d, %d, %d }\n",
        cluster.block_rank(),
        receive_buffer[0], receive_buffer[1], receive_buffer[2], receive_buffer[3]);
  }
#endif

}

int main()
{
  constexpr dim3 cluster_dims = {4, 1, 1};
  constexpr int num_blocks = [&]()
  { 
    return cluster_dims.x * cluster_dims.y * cluster_dims.z;
  }();

  constexpr int num_threads = 128;
#if defined(DEBUG)
  printf("Store remote with %d blocks, %d threads\n", num_blocks, num_threads);
 #endif
  
  using T = cute::uint16_t;
  constexpr int NUM_ELEMS = 4;

  using VecLayout = Layout<Shape<Int<NUM_ELEMS>>>;
  dim3 dimBlock(128);
  constexpr int cluster_size = 4;
  dim3 dimCluster(cluster_size);
  dim3 dimGrid = dimCluster;
  int smem_size = sizeof(SharedStorage<T, VecLayout>);

  void * kernel_ptr = (void *)kernel<T, VecLayout>;
  cutlass::launch_kernel_on_cluster({dimGrid, dimBlock, dimCluster, smem_size},
                                    kernel_ptr);

  checkLastCudaError("After kernel launch");
    
  CUDA_CHECK(cudaDeviceSynchronize());
  checkLastCudaError("After device synchronization");
}