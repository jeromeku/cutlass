#include <cstdio>
#include <cuda/ptx>
#include <cuda/barrier>
#include <cooperative_groups.h>
#include <iostream>
#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>

using namespace cute;

using T = cute::uint32_t;
constexpr int NUM_ELEMS = 2;

__device__ inline bool should_print()
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  if (tx == 0 && ty == 0)
  {
    return true;
  }
  else
  {
    return false;
  }
}
#ifndef PRINT_CLUSTER_DELAY_CYCLES
#define PRINT_CLUSTER_DELAY_CYCLES 10000000ULL
#endif

#define PRINT_CLUSTER(expr)                                                                              \
  do                                                                                                     \
  {                                                                                                      \
    if (threadIdx.x == 0 && threadIdx.y == 0)                                                            \
    {                                                                                                    \
      int _cluster_rank = cute::block_rank_in_cluster();                                                 \
      unsigned long long _delay_cycles = PRINT_CLUSTER_DELAY_CYCLES * (unsigned long long)_cluster_rank; \
      if (_delay_cycles)                                                                                 \
      {                                                                                                  \
        unsigned long long _start = clock64();                                                           \
        while (clock64() - _start < _delay_cycles)                                                       \
        { /* spin */                                                                                     \
        }                                                                                                \
      }                                                                                                  \
      printf("ClusterRank[%d]: ", _cluster_rank);                                                        \
      expr;                                                                                              \
      printf("\n");                                                                                      \
    }                                                                                                    \
  } while (0)

#define CUDA_CHECK(call)                                            \
  do                                                                \
  {                                                                 \
    cudaError_t error = call;                                       \
    if (error != cudaSuccess)                                       \
    {                                                               \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__  \
                << " - " << cudaGetErrorString(error) << std::endl; \
      std::exit(EXIT_FAILURE);                                      \
    }                                                               \
  } while (0)

// Function to check and print last CUDA error
void checkLastCudaError(const char *msg = "CUDA Error")
{
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess)
  {
    std::cerr << msg << ": " << cudaGetErrorString(error) << std::endl;
  }
  else
  {
    std::cout << msg << ": No error detected" << std::endl;
  }
}

template <class ElementType, class SmemLayout>
struct SharedStorage
{
  cute::ArrayEngine<ElementType, cute::cosize_v<SmemLayout>> smem;
  alignas(16) cute::uint64_t tma_load_mbar[1];
};

__device__ void print_block_id(const char *msg)
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int cta_rank_in_cluster = cute::block_rank_in_cluster();
  if (should_print())
  {
    printf("BlockId: (%d, %d), Cluster rank: %d :: %s\n", bx, by, cta_rank_in_cluster, msg);
  }
}

// Store value to remote shared memory in the cluster
CUTE_DEVICE
void
store_shared_remote(uint32_t value, uint32_t smem_addr, uint32_t mbarrier_addr, uint32_t dst_cta_rank)
{
  uint32_t dsmem_addr = set_block_rank(smem_addr, dst_cta_rank);
  uint32_t remote_barrier_addr = set_block_rank(mbarrier_addr, dst_cta_rank);
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.u32 [%0], %1, [%2];"
               : : "r"(dsmem_addr), "r"(value), "r"(remote_barrier_addr));
}

// Store value to remote shared memory in the cluster
CUTE_DEVICE
void
store_shared_remote_u64(uint64_t value, uint32_t smem_addr, uint32_t mbarrier_addr, uint32_t dst_cta_rank)
{
  uint32_t dsmem_addr = set_block_rank(smem_addr, dst_cta_rank);
  uint32_t remote_barrier_addr = set_block_rank(mbarrier_addr, dst_cta_rank);
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.u64 [%0], %1, [%2];"
               : : "r"(dsmem_addr), "l"(value), "r"(remote_barrier_addr));
}

template <typename T, typename VecLayout>
__global__ void kernel(int cluster_size)
{

  bool elect_one_thr = cute::elect_one_sync();
  int warp_idx = cutlass::canonical_warp_idx();

  // One thread per CTA sends (and arrives on mbarrier)
  bool elected = warp_idx == 0 && elect_one_thr;

  extern __shared__ __align__(16) unsigned char shared_bytes[];
  using Storage = SharedStorage<T, VecLayout>;
  Storage& shared_storage = *reinterpret_cast<Storage*>(shared_bytes);

  // Sanity check that mbarrier is properly aligned
  if (elected) {
    printf("mbar align = %llu\n",
          (unsigned long long)((uintptr_t)shared_storage.tma_load_mbar & 0xF));
  }

  cute::Tensor sA = make_tensor(make_smem_ptr(shared_storage.smem.begin()), VecLayout{});
  uint64_t *tma_load_mbar = shared_storage.tma_load_mbar;

  int cta_rank_in_cluster = cute::block_rank_in_cluster();

  // Send in ring
  unsigned int dst_rank = (cta_rank_in_cluster + 1) % cluster_size;

  if (elected)
  {
    // Initialize TMA barrier
    cute::initialize_barrier(tma_load_mbar[0], /* num_threads */ 1);
  }

  // Ensures all CTAs in the Cluster have initialized
  __syncthreads();
  cute::cluster_sync();

  // Send self cluster rank
  constexpr int num_vals = int(size(sA));
  T vals[num_vals];
  constexpr int kTmaTransactionBytes = sizeof(T) * num_vals;

  for(int i = 0; i < num_vals; i++){
    vals[i] = cta_rank_in_cluster;
  }

  // T val = cta_rank_in_cluster;
  // constexpr int kTmaTransactionBytes = sizeof(ArrayEngine<T, CUTE_STATIC_V(size(filter_zeros(sA)))>);
  PRINT_CLUSTER(printf("Sending to rank %d, payload size: %d", dst_rank, kTmaTransactionBytes));
  
  if (elected)
  {
    cute::set_barrier_transaction_bytes(tma_load_mbar[0], kTmaTransactionBytes);
  }
  __syncthreads();
  PRINT_CLUSTER(printf("Value before send %u\n", sA(0)));

  uint32_t barrier_address = cute::cast_smem_ptr_to_uint(tma_load_mbar);
  uint32_t remote_barrier_address = cute::set_block_rank(barrier_address, dst_rank);
  uint32_t smem_address = cute::cast_smem_ptr_to_uint(shared_storage.smem.begin());
  uint32_t remote_smem_address = cute::set_block_rank(smem_address, dst_rank);

  __syncthreads();
#if defined(DEBUG)

  PRINT_CLUSTER(printf("Sending %u to %d\n", vals[0], dst_rank));
#endif

  uint64_t *payload = reinterpret_cast<uint64_t *>(vals);
  if (elect_one_thr){
    store_shared_remote_u64(payload[0], remote_smem_address, remote_barrier_address, dst_rank);
  }

  __syncthreads();
  int tma_phase_bit = 0;
  cute::wait_barrier(tma_load_mbar[0], tma_phase_bit);
  for(int i = 0; i < int(size(sA)); i++){
    PRINT_CLUSTER(printf("Received sA(%d) %u\n", i, sA(i)));
  }

  __syncthreads();
  cute::cluster_sync();

}

int main()
{
  constexpr dim3 cluster_dims = {4, 1, 1};
  constexpr int num_blocks = [&]()
  {
    return cluster_dims.x * cluster_dims.y * cluster_dims.z;
  }();

  constexpr int num_threads = 32;
#if defined(DEBUG)
  printf("Store remote with %d blocks, %d threads\n", num_blocks, num_threads);
#endif

  using VecLayout = Layout<Shape<Int<NUM_ELEMS>>>;
  dim3 dimBlock(num_threads);
  constexpr int cluster_size = 4;
  dim3 dimCluster(cluster_size);
  dim3 dimGrid = dimCluster;
  int smem_size = sizeof(SharedStorage<T, VecLayout>);

  void *kernel_ptr = (void *)kernel<T, VecLayout>;
  cutlass::launch_kernel_on_cluster({dimGrid, dimBlock, dimCluster, smem_size},
                                    kernel_ptr,
                                    cluster_size);

  checkLastCudaError("After kernel launch");

  CUDA_CHECK(cudaDeviceSynchronize());
  checkLastCudaError("After device synchronization");
}
