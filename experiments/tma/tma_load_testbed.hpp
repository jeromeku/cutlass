#include <iostream>
#include <cstdint>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

namespace cutlass::test {

template <class ElementType, class SmemLayout>
struct SharedStorage
{
  cute::ArrayEngine<ElementType, cute::cosize_v<SmemLayout>> smem;
  alignas(16) cute::uint64_t tma_load_mbar[1];
};

template <class T, class TiledCopy, class CTA_Tiler, class GmemLayout, class SmemLayout>
__global__ void
tma_test_device_cute(T const* g_in, T* g_out,
                     CUTE_GRID_CONSTANT TiledCopy const tma, CTA_Tiler cta_tiler,
                     GmemLayout gmem_layout, SmemLayout smem_layout)
{
  using namespace cute;
  CUTE_STATIC_ASSERT_V(product_each(shape(cta_tiler)) == product_each(shape(smem_layout)));

  // Use Shared Storage structure to allocate and distribute aligned SMEM addresses
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<T, SmemLayout>;
  SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(shared_memory);

  // Construct SMEM tensor
  Tensor sA = make_tensor(make_smem_ptr(shared_storage.smem.begin()), smem_layout);  // (CTA_TILE_M,CTA_TILE_N,...)
  // Shared memory barriers use 64bits in SMEM for synchronization
  uint64_t* tma_load_mbar = shared_storage.tma_load_mbar;

  // TMA requires special handling of strides to deal with coord codomain mapping
  // Represent the full tensors -- get these from TMA
  Tensor mA = tma.get_tma_tensor(shape(gmem_layout));
  Tensor mB = make_tensor(make_gmem_ptr<T>(g_out), gmem_layout);

  constexpr int R = rank_v<CTA_Tiler>;
  Tensor gA = flat_divide(mA, cta_tiler);               // (CTA_TILE_M,CTA_TILE_N,...REST_M,REST_N,...)
  Tensor gB = flat_divide(mB, cta_tiler);               // (CTA_TILE_M,CTA_TILE_N,...REST_M,REST_N,...)

  //
  // Prepare the TMA_LOAD
  //

  auto cta_tma = tma.get_slice(Int<0>{});                            // CTA slice
  Tensor tAgA_x = cta_tma.partition_S(gA);                           // (TMA,TMA_M,TMA_N,REST_M,REST_N)
  Tensor tAsA_x = cta_tma.partition_D(sA);                           // (TMA,TMA_M,TMA_N)

  // INPUT: Group the REST_X modes and the TMA_X modes to easily iterate through the tiles
  Tensor tAgA = group_modes<1,rank(tAgA_x)>(tAgA_x);                 // (TMA,REST)
  Tensor tAsA = group_modes<1,rank(tAsA_x)>(tAsA_x);                 // (TMA,REST)
  static_assert(size<1>(tAsA) == 1);

  // OUTPUT: Group the CTA_TILE_X modes and REST_X modes for output
  Tensor tBgB = group_modes<0,R>(group_modes<R,rank(gB)>(gB));       // (CTA_TILE, REST)

  // Test L2 prefetch
  if (threadIdx.x == 0) {
    prefetch(tma, tAgA);
  }

  // Loop over the TMA stages, using smem as our buffer
  for (int stage = 0; stage < size<1>(tAgA); ++stage)
  {
    // Set the bytes transferred in this TMA transaction (may involve multiple issues)
    constexpr int kTmaTransactionBytes = sizeof(make_tensor_like(tensor<0>(tAsA)));

    if (threadIdx.x == 0)
    {
      /// Initialize shared memory barrier
      tma_load_mbar[0] = 0;
      cute::initialize_barrier(tma_load_mbar[0], 1 /*numThreads*/);
      cute::set_barrier_transaction_bytes(tma_load_mbar[0], kTmaTransactionBytes);

      copy(tma.with(tma_load_mbar[0]), tAgA(_,stage), tAsA(_,0));
    }
    __syncthreads();

    /// Wait on the shared memory barrier until the phase bit flips from kPhaseBit value
    constexpr int kPhaseBit = 0;
    cute::wait_barrier(tma_load_mbar[0], kPhaseBit);
    if (thread0()){
      printf("SMem:\n");
      Tensor sA_swizzled = make_tensor(make_smem_ptr(shared_storage.smem.begin()), gmem_layout);  
      cute::print_tensor(sA_swizzled);
    }
    __syncthreads();
  }
}

template <class T = uint16_t, class TmaType = T, class CopyOp, class GMEM_Layout, class SMEM_Layout, class CTA_Tile>
auto
test_tma_load(CopyOp      const& copy_op,
              GMEM_Layout const& gmem_layout,
              SMEM_Layout const& smem_layout,
              CTA_Tile    const& cta_tile)
{
  using namespace cute;

  // Allocate and initialize host test data
  size_t NUMEL = cosize(gmem_layout);
  int M = size<0>(gmem_layout);
  int N = size<1>(gmem_layout);
  assert(NUMEL == M * N);
  thrust::host_vector<T> h_in(NUMEL);
  for (size_t i = 0; i < h_in.size(); ++i) {
    h_in[i] = i % N;
  }
  Tensor hA_in  = make_tensor(recast_ptr<T>(h_in.data()), gmem_layout);
  cute::print_tensor(hA_in);
  // Allocate and initialize device test data
  thrust::device_vector<T> d_in = h_in;
  thrust::device_vector<T> d_out(h_in.size()); // overflow uint

  // // Create TMA for this device Tensor
  Tensor gA = make_tensor(make_gmem_ptr<T>(raw_pointer_cast(d_in.data())), gmem_layout);
  auto tma = make_tma_copy<TmaType>(copy_op, gA, smem_layout, cta_tile, Int<1>{});
  print(tma);

  // // Launch
  int smem_size = int(sizeof(SharedStorage<T, decltype(smem_layout)>));
  tma_test_device_cute<<<1, 128, smem_size>>>(
    reinterpret_cast<T const*>(raw_pointer_cast(d_in.data())),
    reinterpret_cast<T*>      (raw_pointer_cast(d_out.data())),
    tma, cta_tile,
    gmem_layout,
    smem_layout);

  // Copy results back to host
  // thrust::host_vector<uint8_t> h_out = d_out;
  // Tensor hA_out = make_tensor(recast_ptr<T>(h_out.data()), gmem_layout);

  // Validate the results. Print only the first 3 errors.
  // int count = 3;
  // for (int i = 0; i < int(size(hA_out)) && count > 0; ++i) {
  //   EXPECT_EQ(hA_in(i), hA_out(i));
  //   if (hA_in(i) != hA_out(i)) {
  //     --count;
  //   }
  // }

  // return tma;
}


} // end namespace cutlass::test
