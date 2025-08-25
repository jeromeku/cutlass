#include <cstdio>
#include <cuda/ptx>
#include <cuda/barrier>
#include <cooperative_groups.h>
#include <iostream>

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

template <int CLUSTER_X, int CLUSTER_Y, int CLUSTER_Z>
__global__ void __cluster_dims__(CLUSTER_X, CLUSTER_Y, CLUSTER_Z) kernel()
{
  using cuda::ptx::scope_cluster;
  using cuda::ptx::sem_acquire;
  using cuda::ptx::sem_release;
  using cuda::ptx::space_cluster;
  using cuda::ptx::space_shared;

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

  if (threadIdx.x == 0)
  {
    printf("[block %d] arrived with expected tx count = %llu, sending to block rank: %u\n", cluster.block_rank(), sizeof(receive_buffer), other_block_rank);
  }

  // Send bytes to remote buffer, arriving on remote barrier
  if (threadIdx.x == 0)
  {
    cuda::ptx::st_async(remote_buffer, {int(cluster.block_rank()), 2, 3, 4}, remote_bar);
  }

  if (threadIdx.x == 0)
  {
    printf("[block %d] -> %u: st_async to %p, %p\n",
           cluster.block_rank(),
           other_block_rank,
           remote_buffer,
           remote_bar);
  }

  // Wait on local barrier:
  while (!cuda::ptx::mbarrier_try_wait(sem_acquire, scope_cluster, cuda::device::barrier_native_handle(bar), arrival_token))
  {
  }

  // Print received values:
  if (threadIdx.x == 0)
  {
    printf(
        "[block %d] receive_buffer = { %d, %d, %d, %d }\n",
        cluster.block_rank(),
        receive_buffer[0], receive_buffer[1], receive_buffer[2], receive_buffer[3]);
  }
}

int main()
{
  constexpr dim3 cluster_dims = {4, 1, 1};
  constexpr int num_blocks = [&]()
  { 
    return cluster_dims.x * cluster_dims.y * cluster_dims.z;
  }();

  constexpr int num_threads = 128;
  printf("Store remote with %d blocks, %d threads\n", num_blocks, num_threads);
  kernel<cluster_dims.x, cluster_dims.y, cluster_dims.z><<<num_blocks, num_threads>>>();
  checkLastCudaError("After kernel launch");
    
  CUDA_CHECK(cudaDeviceSynchronize());
  checkLastCudaError("After device synchronization");
}