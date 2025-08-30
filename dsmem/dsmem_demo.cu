#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>
#include "utils.hpp"

using namespace cute;

using T = cute::uint32_t;
constexpr int NUM_ELEMS = 2;
constexpr int NUM_THREADS = 32;
constexpr int CLUSTER_SIZE = 4;

// Store value to remote shared memory in the cluster
CUTE_DEVICE
void store_shared_remote_u64(uint64_t value, uint32_t smem_addr, uint32_t mbarrier_addr, uint32_t dst_cta_rank)
{
  uint32_t dsmem_addr = set_block_rank(smem_addr, dst_cta_rank);
  uint32_t remote_barrier_addr = set_block_rank(mbarrier_addr, dst_cta_rank);
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.u64 [%0], %1, [%2];"
               : : "r"(dsmem_addr), "l"(value), "r"(remote_barrier_addr));
}

template <class ElementType, class SmemLayout>
struct SharedStorage
{
  cute::ArrayEngine<ElementType, cute::cosize_v<SmemLayout>> smem;
  // mbarrier must be 16B aligned
  alignas(16) cute::uint64_t mbar[1];
};

template <typename T, typename VecLayout>
__global__ void dsmem_store_kernel(int cluster_size)
{
  // Only support sending 2 uint32_t for this demo
  CUTE_STATIC_ASSERT(cute::is_same_v<T, uint32_t>);
  CUTE_STATIC_ASSERT(cute::is_same_v<decltype(size(VecLayout{})), Int<2>>);

  extern __shared__ __align__(16) unsigned char shared_bytes[];
  using Storage = SharedStorage<T, VecLayout>;
  Storage &shared_storage = *reinterpret_cast<Storage *>(shared_bytes);

  // One thread per CTA sends (and arrives on mbarrier)
  bool elect_one_thr = cute::elect_one_sync();
  int warp_idx = cutlass::canonical_warp_idx();
  bool elected = warp_idx == 0 && elect_one_thr;

  // Rank in cluster
  int cta_rank_in_cluster = cute::block_rank_in_cluster();

  // Sanity check that mbarrier is properly aligned
  if (elected)
  {
    unsigned long long align = (unsigned long long)((uintptr_t)shared_storage.mbar & 0xF);
    assert(align == 0);
  }

  // Aliases for smem recv buffer and mbarrier
  cute::Tensor sA = make_tensor(make_smem_ptr(shared_storage.smem.begin()), VecLayout{});
  uint64_t *mbar = shared_storage.mbar;

  // Send in ring (rank 0 -> rank 1 -> rank2 -> rank3 -> rank0)
  unsigned int dst_rank = (cta_rank_in_cluster + 1) % cluster_size;

  if (elected)
  {
    // Initialize mbarrier, **1** thread arrives
    cute::initialize_barrier(mbar[0], 1);
  }

  // Ensures all CTAs in the Cluster have initialized
  __syncthreads();
  cute::cluster_sync();

  // Send self cluster rank
  constexpr int num_vals = int(size(sA));
  T vals[num_vals];
  constexpr int kTransactionBytes = sizeof(T) * num_vals;
  for (int i = 0; i < num_vals; i++)
  {
    vals[i] = cta_rank_in_cluster;
  }
  uint64_t *payload = reinterpret_cast<uint64_t *>(vals);

  // Map remote addresses for mbarrier and smem buffer
  uint32_t barrier_address = cute::cast_smem_ptr_to_uint(mbar);
  uint32_t remote_barrier_address = cute::set_block_rank(barrier_address, dst_rank);
  uint32_t smem_address = cute::cast_smem_ptr_to_uint(shared_storage.smem.begin());
  uint32_t remote_smem_address = cute::set_block_rank(smem_address, dst_rank);

  if (elected)
  {
    // Set transaction bytes to await
    cute::set_barrier_transaction_bytes(mbar[0], kTransactionBytes);
    // Send using st.async, no need for TMA given small payload
    store_shared_remote_u64(payload[0], remote_smem_address, remote_barrier_address, dst_rank);
  }
  __syncthreads();

  // Wait in loop, mbarrier initial state is 0;
  int phase = 0;
  cute::wait_barrier(mbar[0], phase);

  // Print results, cluster rank i should print the previous rank
  PRINT_CLUSTER(printf("Received sA: %u %u", sA(0), sA(1)));

  __syncthreads();
  cute::cluster_sync();
}

int main()
{
  using VecLayout = Layout<Shape<Int<NUM_ELEMS>>>;
  dim3 dimBlock(NUM_THREADS);
  dim3 dimCluster(CLUSTER_SIZE);
  dim3 dimGrid = dimCluster;
  int smem_size = sizeof(SharedStorage<T, VecLayout>);

  void *kernel_ptr = (void *)dsmem_store_kernel<T, VecLayout>;
  cutlass::launch_kernel_on_cluster({dimGrid, dimBlock, dimCluster, smem_size},
                                    kernel_ptr,
                                    CLUSTER_SIZE);

  checkLastCudaError("After kernel launch");

  CUDA_CHECK(cudaDeviceSynchronize());
  checkLastCudaError("After device synchronization");
}
