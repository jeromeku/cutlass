/**
 * nvcc -std=c++17 -arch=sm_90 -O0 tma_swizzle_64x64.cu -o tma_swizzle_64x64 \
 *      -lcuda
 *
 * Requires CUDA 12.2+ and an SM_90+ GPU (H100 or newer).
 */

#include <cuda.h>             // CUtensorMap, cuTensorMapEncodeTiled
#include <cuda_runtime.h>
#include <cuda/barrier>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

// ---- Device kernel: TMA load 64x64 into swizzled SMEM and print it ----

__global__ void print_swizzled_64x64(const __grid_constant__ CUtensorMap tensor_map) {
  // TMA requires shared memory dest to be 128B aligned for 128B swizzle.
  __shared__ alignas(128) uint16_t smem[64][64];

  // Block-scoped barrier used by the TMA wrapper.
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;

  // Initialize barrier in one thread, then make visible to async proxy (TMA engine).
  if (threadIdx.x == 0) {
    init(&bar, blockDim.x);              // all blockDim.x threads participate
    cde::fence_proxy_async_shared_cta(); // make barrier visible to TMA
  }
  __syncthreads();

  barrier::arrival_token token;

  if (threadIdx.x == 0) {
    // Launch TMA: copy a 64x64 tile starting at (0,0) from GMEM -> SMEM
    // Layout and swizzle are encoded in tensor_map, so this is all you pass.
    cde::cp_async_bulk_tensor_2d_global_to_shared(
        smem,              // destination shared memory tile
        &tensor_map,       // tensor map descriptor
        /*c0=*/0,          // Y coordinate (row offset) in tiles
        /*c1=*/0,          // X coordinate (col offset) in tiles
        bar);

    // Tell the barrier how many bytes to expect (for completion tracking).
    token = cuda::device::barrier_arrive_tx(
        bar, 1, sizeof(smem));
  } else {
    token = bar.arrive();
  }

  // Wait until TMA transfer has finished writing SMEM.
  bar.wait(std::move(token));
  __syncthreads();

  // At this point, smem[][] holds the **swizzled** layout.
  // Each thread prints one or more rows so we see the full 64x64 contents.
  for (int row = threadIdx.x; row < 64; row += blockDim.x) {
    //printf("row %02d:", row);
    for (int col = 0; col < 64; ++col) {
      printf(" %4u", smem[row][col]);
    }
    printf("\n");
  }
}

// ---- Host-side helpers ----

static inline void checkCuda(cudaError_t e, const char* where) {
  if (e != cudaSuccess) {
    fprintf(stderr, "CUDA runtime error at %s: %s\n", where, cudaGetErrorString(e));
    std::exit(1);
  }
}

static inline void checkDriver(CUresult r, const char* where) {
  if (r != CUDA_SUCCESS) {
    const char* msg = nullptr;
    cuGetErrorString(r, &msg);
    fprintf(stderr, "CUDA driver error at %s: %s\n", where, msg ? msg : "<unknown>");
    std::exit(1);
  }
}

int main() {
  checkCuda(cudaSetDevice(0), "cudaSetDevice");

  // 1. Create a 64x64 uint16 matrix in global memory with a recognizable pattern
  constexpr int H = 64;
  constexpr int W = 64;
  const size_t num_elems = H * W;
  const size_t bytes = num_elems * sizeof(uint16_t);

  uint16_t* h_src = (uint16_t*)std::malloc(bytes);
  if (!h_src) {
    fprintf(stderr, "host malloc failed\n");
    return 1;
  }

  // Fill: value = row * 100 + col, so you can see row/col in printout easily.
  for (int r = 0; r < H; ++r) {
    for (int c = 0; c < W; ++c) {
      h_src[r * W + c] = static_cast<uint16_t>(c);
    }
  }

  uint16_t* d_src = nullptr;
  checkCuda(cudaMalloc(&d_src, bytes), "cudaMalloc(d_src)");
  checkCuda(cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy H2D");

  // 2. Encode a CUtensorMap with 128B swizzle for a 64x64 uint16 matrix.
  //
  //    - rank = 2 (rows, cols)
  //    - globalDim = {rows, cols} in elements
  //    - globalStrides[0] = row stride in BYTES (must be multiple of 16)
  //    - boxDim = tile size loaded by a single TMA op (here 64x64)
  //    - elementStrides = {1, 1} (contiguous)
  //    - swizzle mode = CU_TENSOR_MAP_SWIZZLE_128B
  //
  CUtensorMap tensor_map{};
  constexpr uint32_t rank = 2;

  uint64_t global_dim[rank]    = {H, W};
  uint64_t global_strides[rank - 1] = {
      static_cast<uint64_t>(W * sizeof(uint16_t))  // bytes between rows
  };

  // We want the TMA tile == whole matrix for this demo
  uint32_t box_dim[rank]       = {H, W};

  uint32_t elem_strides[rank]  = {1, 1};

  // Make sure driver API is initialized
  checkDriver(cuInit(0), "cuInit");

  CUresult res = cuTensorMapEncodeTiled(
      &tensor_map,
      CU_TENSOR_MAP_DATA_TYPE_UINT16,           // data type
      rank,
      d_src,                                    // base global address
      global_dim,
      global_strides,
      box_dim,
      elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B,              // <--- 128B swizzle
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  checkDriver(res, "cuTensorMapEncodeTiled");

  // 3. Launch kernel that TMA-loads 64x64 into SMEM and prints *swizzled* layout
  dim3 grid(1, 1);
  dim3 block(1, 1);  // 64 threads; each prints a row or more

  print_swizzled_64x64<<<grid, block>>>(tensor_map);
  checkCuda(cudaGetLastError(), "kernel launch");
  checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  std::free(h_src);
  cudaFree(d_src);

  return 0;
}
