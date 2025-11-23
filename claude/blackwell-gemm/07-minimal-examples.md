# Minimal Test Examples

This document provides standalone, minimal code examples for testing individual APIs from the Blackwell GEMM tutorials. Each example is self-contained and can be compiled independently to understand and debug specific components.

## Table of Contents

1. [Testing make_tiled_mma](#testing-make_tiled_mma)
2. [Testing tile_to_mma_shape](#testing-tile_to_mma_shape)
3. [Testing make_tma_atom](#testing-make_tma_atom)
4. [Testing tma_partition](#testing-tma_partition)
5. [Testing create_tma_multicast_mask](#testing-create_tma_multicast_mask)
6. [Testing TMA Copy](#testing-tma-copy)
7. [Testing MMA (gemm)](#testing-mma-gemm)
8. [Testing make_tmem_copy](#testing-make_tmem_copy)
9. [Testing TMEM Allocator](#testing-tmem-allocator)
10. [Complete Minimal GEMM](#complete-minimal-gemm)

---

## Testing make_tiled_mma

### Purpose

Create and inspect a `TiledMMA` object to understand its structure and properties.

### Minimal Example

```cpp
#include <cute/tensor.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/atom/mma_atom.hpp>
#include <iostream>

using namespace cute;

int main() {
  // Create a 1SM MMA
  auto mma_1sm = make_tiled_mma(
    SM100_MMA_F16BF16_SS<cutlass::half_t, cutlass::half_t, float,
                         128, 256,
                         UMMA::Major::K, UMMA::Major::K>{}
  );

  // Print MMA properties
  std::cout << "=== 1SM MMA Properties ===" << std::endl;
  std::cout << "Shape MNK: ";
  print(typename decltype(mma_1sm)::TiledShape_MNK{});
  std::cout << std::endl;

  std::cout << "ThrID: ";
  print(typename decltype(mma_1sm)::ThrLayoutVMNK{});
  std::cout << std::endl;

  // Create a 2SM MMA
  auto mma_2sm = make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_SS<cutlass::half_t, cutlass::half_t, float,
                               256, 256,
                               UMMA::Major::K, UMMA::Major::K>{}
  );

  std::cout << "\n=== 2SM MMA Properties ===" << std::endl;
  std::cout << "Shape MNK: ";
  print(typename decltype(mma_2sm)::TiledShape_MNK{});
  std::cout << std::endl;

  std::cout << "ThrID: ";
  print(typename decltype(mma_2sm)::ThrLayoutVMNK{});
  std::cout << std::endl;

  return 0;
}
```

### Compile and Run

```bash
nvcc -std=c++17 -arch=sm_100a \
     -I/path/to/cutlass/include \
     test_make_tiled_mma.cu -o test_make_tiled_mma

./test_make_tiled_mma
```

### Expected Output

```
=== 1SM MMA Properties ===
Shape MNK: (_128,_256,_16)
ThrID: (_1,_1,_1,_1)

=== 2SM MMA Properties ===
Shape MNK: (_256,_256,_16)
ThrID: (_2,_1,_1,_1)
```

---

## Testing tile_to_mma_shape

### Purpose

Understand how `tile_to_mma_shape` transforms layout atoms into complete SMEM layouts.

### Minimal Example

```cpp
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <iostream>

using namespace cute;

int main() {
  using TypeA = cutlass::half_t;

  // Create layout atom (128B swizzled, K-major)
  auto atom = UMMA::Layout_K_SW128_Atom<TypeA>{};

  // Define MMA tile shape
  auto mma_shape = make_shape(
    make_shape(Int<128>{}, Int<16>{}),  // MMA atom (128 M × 16 K)
    Int<1>{},                            // 1 repetition in M
    Int<4>{}                             // 4 repetitions in K
  );

  // Apply tile_to_mma_shape
  auto smem_layout = UMMA::tile_to_mma_shape(atom, mma_shape);

  // Print results
  std::cout << "=== Input MMA Shape ===" << std::endl;
  std::cout << "mma_shape: ";
  print(mma_shape);
  std::cout << std::endl;

  std::cout << "\n=== Output SMEM Layout ===" << std::endl;
  std::cout << "smem_layout: ";
  print(smem_layout);
  std::cout << std::endl;

  std::cout << "\n=== Layout Properties ===" << std::endl;
  std::cout << "Total size: " << size(smem_layout) << " elements" << std::endl;
  std::cout << "Cosize: " << cosize(smem_layout) << " bytes" << std::endl;

  return 0;
}
```

### Compile and Run

```bash
nvcc -std=c++17 -arch=sm_100a \
     -I/path/to/cutlass/include \
     test_tile_to_mma_shape.cu -o test_tile_to_mma_shape

./test_tile_to_mma_shape
```

### Expected Output

```
=== Input MMA Shape ===
mma_shape: ((_128,_16),_1,_4)

=== Output SMEM Layout ===
smem_layout: Sw<3,4,3> o smem_ptr[16b](unset) o ((_128,_16),_1,_4):((_64,_1),_0,_16)

=== Layout Properties ===
Total size: 8192 elements
Cosize: 16384 bytes
```

---

## Testing make_tma_atom

### Purpose

Create TMA descriptors on the host and inspect their properties.

### Minimal Example (Host-Side)

```cpp
#include <cute/tensor.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>

using namespace cute;

int main() {
  using TypeA = cutlass::half_t;

  // Create host data
  int M = 512, K = 256;
  thrust::host_vector<TypeA> host_A(M * K);

  // Transfer to device
  thrust::device_vector<TypeA> device_A = host_A;

  // Create GMEM tensor
  auto layout_A = make_layout(
    make_shape(Int<512>{}, Int<256>{}),
    make_stride(Int<256>{}, Int<1>{})  // K-major
  );
  auto mA = make_tensor(make_gmem_ptr(device_A.data().get()), layout_A);

  // Create SMEM layout (swizzled)
  auto atom = UMMA::Layout_K_SW128_Atom<TypeA>{};
  auto mma_shape_A = make_shape(
    make_shape(Int<128>{}, Int<16>{}),
    Int<1>{},
    Int<4>{}
  );
  auto sA_layout = UMMA::tile_to_mma_shape(atom, mma_shape_A);

  // Create TMA atom
  auto tma_atom_A = make_tma_atom(
    SM90_TMA_LOAD_MULTICAST{},
    mA,
    sA_layout,
    make_shape(Int<128>{}, Int<64>{}),  // MK tile
    Int<4>{}                             // Multicast to 4 CTAs
  );

  // Inspect TMA descriptor
  std::cout << "=== TMA Atom Properties ===" << std::endl;
  std::cout << "TMA descriptor created successfully" << std::endl;
  std::cout << "Transfer size: " << sizeof(make_tensor_like(sA_layout)) << " bytes" << std::endl;

  // Get TMA tensor
  auto mA_tma = tma_atom_A.get_tma_tensor(shape(mA));
  std::cout << "\n=== TMA Tensor ===" << std::endl;
  std::cout << "mA_tma shape: ";
  print(shape(mA_tma));
  std::cout << std::endl;

  return 0;
}
```

### Compile and Run

```bash
nvcc -std=c++17 -arch=sm_100a \
     -I/path/to/cutlass/include \
     test_make_tma_atom.cu -o test_make_tma_atom

./test_make_tma_atom
```

---

## Testing tma_partition

### Purpose

Test TMA partitioning logic in a simplified kernel.

### Minimal Example (Device Kernel)

```cpp
#include <cute/tensor.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>

using namespace cute;

__global__ void test_tma_partition_kernel() {
  // Simulate cluster layout
  auto cluster_layout_vmnk = make_layout(
    make_shape(Int<2>{}, Int<2>{}, Int<4>{}, Int<1>{}),  // (V, M, N, K)
    make_stride(Int<8>{}, Int<4>{}, Int<1>{}, Int<0>{})
  );

  // Get CTA coordinate in cluster
  auto cta_rank = cute::block_rank_in_cluster();
  auto cta_coord_vmnk = cluster_layout_vmnk.get_flat_coord(int(cta_rank));

  if (thread0()) {
    printf("CTA rank: %d\n", int(cta_rank));
    printf("CTA coord (V,M,N,K): (%d,%d,%d,%d)\n",
           int(get<0>(cta_coord_vmnk)),
           int(get<1>(cta_coord_vmnk)),
           int(get<2>(cta_coord_vmnk)),
           int(get<3>(cta_coord_vmnk)));
  }
}

int main() {
  // Launch kernel in cluster mode
  dim3 dimBlock(128);
  dim3 dimGrid(2, 2);
  dim3 dimCluster(2, 2, 1);

  cudaLaunchKernelEx(
    dimGrid, dimBlock,
    test_tma_partition_kernel,
    dimCluster,
    0, nullptr
  );

  cudaDeviceSynchronize();
  return 0;
}
```

---

## Testing create_tma_multicast_mask

### Purpose

Test multicast mask generation for different cluster configurations.

### Minimal Example

```cpp
#include <cute/tensor.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <iostream>

using namespace cute;

int main() {
  // Define cluster layout (V=2, M=2, N=4, K=1)
  auto cluster_layout_vmnk = make_layout(
    make_shape(Int<2>{}, Int<2>{}, Int<4>{}, Int<1>{}),
    make_stride(Int<8>{}, Int<4>{}, Int<1>{}, Int<0>{})
  );

  // Test different CTA coordinates
  std::cout << "=== Multicast Masks ===" << std::endl;

  for (int v = 0; v < 2; ++v) {
    for (int m = 0; m < 2; ++m) {
      for (int n = 0; n < 4; ++n) {
        auto cta_coord = make_coord(v, m, n, 0);

        // Multicast along N (for A matrix)
        uint16_t mask_a = create_tma_multicast_mask<2>(cluster_layout_vmnk, cta_coord);

        // Multicast along M (for B matrix)
        uint16_t mask_b = create_tma_multicast_mask<1>(cluster_layout_vmnk, cta_coord);

        printf("CTA (%d,%d,%d): mask_A=0x%04x, mask_B=0x%04x\n",
               v, m, n, mask_a, mask_b);
      }
    }
  }

  return 0;
}
```

### Expected Output

```
=== Multicast Masks ===
CTA (0,0,0): mask_A=0x000f, mask_B=0x0055
CTA (0,0,1): mask_A=0x000f, mask_B=0x0055
CTA (0,0,2): mask_A=0x000f, mask_B=0x00aa
CTA (0,0,3): mask_A=0x000f, mask_B=0x00aa
CTA (0,1,0): mask_A=0x00f0, mask_B=0x0055
...
```

---

## Testing TMA Copy

### Purpose

Minimal kernel demonstrating TMA copy from GMEM to SMEM.

### Minimal Example

```cpp
#include <cute/tensor.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>

using namespace cute;

template <class TmaAtom>
__global__ void test_tma_copy_kernel(TmaAtom tma_atom) {
  extern __shared__ char smem[];

  // Allocate SMEM buffer
  using TypeA = cutlass::half_t;
  auto sA = make_tensor(make_smem_ptr((TypeA*)smem),
                       make_shape(Int<128>{}, Int<64>{}));

  // Barrier in SMEM
  __shared__ alignas(16) uint64_t barrier;

  // Initialize barrier
  if (threadIdx.x == 0) {
    cute::initialize_barrier(barrier, 1);
  }
  __syncthreads();

  // Execute TMA load
  if (threadIdx.x == 0) {
    cute::set_barrier_transaction_bytes(barrier, sizeof(sA));

    // Get TMA tensor (coordinate tensor)
    auto mA_tma = tma_atom.get_tma_tensor(make_shape(Int<512>{}, Int<256>{}));

    // Partition for this CTA
    auto [gA, sA_dst] = tma_partition(tma_atom, sA, mA_tma);

    // Issue TMA copy
    uint16_t mask = 1;  // Single CTA
    copy(tma_atom.with(barrier, mask), gA(_, 0), sA_dst);
  }

  // Wait for TMA completion
  cute::wait_barrier(barrier, 0);

  // Verify data loaded
  if (thread0()) {
    printf("TMA copy completed. First element: %f\n",
           float(sA(0, 0)));
  }
}
```

---

## Testing MMA (gemm)

### Purpose

Minimal kernel demonstrating MMA execution.

### Minimal Example

```cpp
#include <cute/tensor.hpp>
#include <cute/arch/mma_sm100_umma.hpp>

using namespace cute;

__global__ void test_mma_kernel() {
  extern __shared__ char smem[];

  using TypeA = cutlass::half_t;
  using TypeB = cutlass::half_t;
  using TypeC = float;

  // Allocate SMEM for A, B
  auto sA = make_tensor(make_smem_ptr((TypeA*)smem),
                       make_shape(Int<128>{}, Int<16>{}));
  auto sB = make_tensor(make_smem_ptr((TypeB*)(smem + 128*16*2)),
                       make_shape(Int<256>{}, Int<16>{}));

  // Initialize SMEM with test data
  for (int i = threadIdx.x; i < 128*16; i += blockDim.x) {
    sA.data()[i] = TypeA(1.0f);
  }
  for (int i = threadIdx.x; i < 256*16; i += blockDim.x) {
    sB.data()[i] = TypeB(1.0f);
  }
  __syncthreads();

  // Allocate TMEM (simplified)
  __shared__ uint32_t tmem_base;
  if (threadIdx.x == 0) {
    cute::TMEM::Allocator1Sm allocator;
    allocator.allocate(cute::TMEM::Allocator1Sm::Sm100TmemCapacityColumns, &tmem_base);
  }
  __syncthreads();

  // Create MMA
  auto mma = make_tiled_mma(
    SM100_MMA_F16BF16_SS<TypeA, TypeB, TypeC, 128, 256, UMMA::Major::K, UMMA::Major::K>{}
  );

  // Create accumulator tensor
  auto tCtAcc = make_tensor(make_tmem_ptr<TypeC>(tmem_base),
                           make_shape(Int<128>{}, Int<256>{}));

  // Get MMA slice
  auto cta_mma = mma.get_slice(0);

  // Create fragments (SMEM descriptors)
  auto tCrA = cta_mma.make_fragment_A(sA);
  auto tCrB = cta_mma.make_fragment_B(sB);

  // Execute MMA
  if (threadIdx.x / 32 == 0) {  // Warp 0
    mma.accumulate_ = UMMA::ScaleOut::Zero;
    gemm(mma, tCrA, tCrB, tCtAcc);
  }

  if (thread0()) {
    printf("MMA completed. Accumulator allocated at TMEM base: 0x%x\n", tmem_base);
  }
}
```

---

## Testing make_tmem_copy

### Purpose

Test TMEM to register copy.

### Minimal Example

```cpp
#include <cute/tensor.hpp>
#include <cute/atom/copy_traits_sm100.hpp>

using namespace cute;

__global__ void test_tmem_copy_kernel() {
  using TypeC = float;

  // Allocate TMEM
  __shared__ uint32_t tmem_base;
  if (threadIdx.x == 0) {
    cute::TMEM::Allocator1Sm allocator;
    allocator.allocate(cute::TMEM::Allocator1Sm::Sm100TmemCapacityColumns, &tmem_base);
  }
  __syncthreads();

  // Create TMEM tensor
  auto tAcc = make_tensor(make_tmem_ptr<TypeC>(tmem_base),
                         make_shape(Int<128>{}, Int<256>{}));

  // Create TMEM copy
  auto tiled_copy = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tAcc);
  auto thr_copy = tiled_copy.get_slice(threadIdx.x);

  // Partition TMEM for this thread
  auto tDtAcc = thr_copy.partition_S(tAcc);

  // Allocate registers
  auto tDrAcc = make_fragment_like(tDtAcc);

  // Copy TMEM -> RMEM
  copy(tiled_copy, tDtAcc, tDrAcc);

  if (thread0()) {
    printf("TMEM copy completed for %d threads\n", blockDim.x);
  }
}
```

---

## Testing TMEM Allocator

### Purpose

Test TMEM allocation and deallocation.

### Minimal Example

```cpp
#include <cute/tensor.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>

using namespace cute;

__global__ void test_tmem_allocator_kernel() {
  __shared__ uint32_t tmem_base;

  // Allocate TMEM
  if (threadIdx.x / 32 == 0 && threadIdx.x % 32 == 0) {  // Warp 0, thread 0
    cute::TMEM::Allocator1Sm allocator;

    printf("Thread %d: Allocating TMEM...\n", threadIdx.x);
    allocator.allocate(cute::TMEM::Allocator1Sm::Sm100TmemCapacityColumns, &tmem_base);
    printf("Thread %d: TMEM allocated at base: 0x%x\n", threadIdx.x, tmem_base);

    // Use TMEM...

    printf("Thread %d: Deallocating TMEM...\n", threadIdx.x);
    allocator.release_allocation_lock();
    allocator.free(tmem_base, cute::TMEM::Allocator1Sm::Sm100TmemCapacityColumns);
    printf("Thread %d: TMEM freed\n", threadIdx.x);
  }
  __syncthreads();
}

int main() {
  test_tmem_allocator_kernel<<<1, 128>>>();
  cudaDeviceSynchronize();
  return 0;
}
```

---

## Complete Minimal GEMM

### Purpose

A complete but simplified GEMM that demonstrates all components working together.

### Minimal Example

```cpp
#include <cute/tensor.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace cute;

// Simplified GEMM kernel (single K-tile, single CTA)
template <class TmaAtomA, class TmaAtomB>
__global__ void minimal_gemm_kernel(
  TmaAtomA tma_atom_A,
  TmaAtomB tma_atom_B,
  float* C_ptr,
  int M, int N, int K)
{
  extern __shared__ char smem[];

  using TypeA = cutlass::half_t;
  using TypeB = cutlass::half_t;
  using TypeC = float;

  // SMEM tensors
  auto sA = make_tensor(make_smem_ptr((TypeA*)smem),
                       make_shape(Int<128>{}, Int<64>{}));
  auto sB = make_tensor(make_smem_ptr((TypeB*)(smem + 128*64*2)),
                       make_shape(Int<256>{}, Int<64>{}));

  // Barriers
  __shared__ alignas(16) uint64_t tma_barrier;
  __shared__ alignas(16) uint64_t mma_barrier;
  __shared__ uint32_t tmem_base;

  // Initialize
  if (threadIdx.x == 0) {
    cute::initialize_barrier(tma_barrier, 1);
    cute::initialize_barrier(mma_barrier, 1);

    cute::TMEM::Allocator1Sm allocator;
    allocator.allocate(cute::TMEM::Allocator1Sm::Sm100TmemCapacityColumns, &tmem_base);
  }
  __syncthreads();

  // Create MMA
  auto mma = make_tiled_mma(
    SM100_MMA_F16BF16_SS<TypeA, TypeB, TypeC, 128, 256, UMMA::Major::K, UMMA::Major::K>{}
  );

  // TMA load
  if (threadIdx.x == 0) {
    auto mA_tma = tma_atom_A.get_tma_tensor(make_shape(M, K));
    auto mB_tma = tma_atom_B.get_tma_tensor(make_shape(N, K));

    auto [gA, sA_dst] = tma_partition(tma_atom_A, sA, mA_tma);
    auto [gB, sB_dst] = tma_partition(tma_atom_B, sB, mB_tma);

    cute::set_barrier_transaction_bytes(tma_barrier, sizeof(sA) + sizeof(sB));

    uint16_t mask = 1;
    copy(tma_atom_A.with(tma_barrier, mask), gA(_, 0), sA_dst);
    copy(tma_atom_B.with(tma_barrier, mask), gB(_, 0), sB_dst);
  }

  // Wait for TMA
  cute::wait_barrier(tma_barrier, 0);

  // MMA
  auto cta_mma = mma.get_slice(0);
  auto tCrA = cta_mma.make_fragment_A(sA);
  auto tCrB = cta_mma.make_fragment_B(sB);
  auto tCtAcc = make_tensor(make_tmem_ptr<TypeC>(tmem_base),
                           make_shape(Int<128>{}, Int<256>{}));

  if (threadIdx.x / 32 == 0) {
    mma.accumulate_ = UMMA::ScaleOut::Zero;
    gemm(mma, tCrA, tCrB, tCtAcc);
    cutlass::arch::umma_arrive_multicast(&mma_barrier, 1);
  }

  // Wait for MMA
  cute::wait_barrier(mma_barrier, 0);

  // TMEM -> RMEM -> GMEM
  auto tiled_copy = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
  auto thr_copy = tiled_copy.get_slice(threadIdx.x);

  auto tDtAcc = thr_copy.partition_S(tCtAcc);
  auto tDrAcc = make_fragment_like(tDtAcc);
  copy(tiled_copy, tDtAcc, tDrAcc);

  // Store to GMEM (simplified, no C matrix)
  auto gD = make_tensor(make_gmem_ptr(C_ptr),
                       make_shape(Int<128>{}, Int<256>{}));
  auto tDgD = thr_copy.partition_D(gD);
  copy(tDrAcc, tDgD);

  if (threadIdx.x / 32 == 0) {
    cute::TMEM::Allocator1Sm allocator;
    allocator.release_allocation_lock();
    allocator.free(tmem_base, cute::TMEM::Allocator1Sm::Sm100TmemCapacityColumns);
  }
}

int main() {
  int M = 128, N = 256, K = 64;

  // Allocate and initialize host data
  thrust::host_vector<cutlass::half_t> h_A(M * K, cutlass::half_t(1.0f));
  thrust::host_vector<cutlass::half_t> h_B(N * K, cutlass::half_t(1.0f));
  thrust::host_vector<float> h_C(M * N, 0.0f);

  // Transfer to device
  thrust::device_vector<cutlass::half_t> d_A = h_A;
  thrust::device_vector<cutlass::half_t> d_B = h_B;
  thrust::device_vector<float> d_C(M * N);

  // Create tensors
  auto layout_A = make_layout(make_shape(M, K), make_stride(K, Int<1>{}));
  auto layout_B = make_layout(make_shape(N, K), make_stride(K, Int<1>{}));
  auto mA = make_tensor(make_gmem_ptr(d_A.data().get()), layout_A);
  auto mB = make_tensor(make_gmem_ptr(d_B.data().get()), layout_B);

  // Create SMEM layouts
  auto sA_layout = UMMA::tile_to_mma_shape(
    UMMA::Layout_K_SW128_Atom<cutlass::half_t>{},
    make_shape(make_shape(Int<128>{}, Int<16>{}), Int<1>{}, Int<4>{})
  );
  auto sB_layout = UMMA::tile_to_mma_shape(
    UMMA::Layout_K_SW128_Atom<cutlass::half_t>{},
    make_shape(make_shape(Int<256>{}, Int<16>{}), Int<1>{}, Int<4>{})
  );

  // Create TMA atoms (host-side)
  auto tma_atom_A = make_tma_atom(
    SM90_TMA_LOAD_MULTICAST{}, mA, sA_layout,
    make_shape(Int<128>{}, Int<64>{}), Int<1>{}
  );
  auto tma_atom_B = make_tma_atom(
    SM90_TMA_LOAD_MULTICAST{}, mB, sB_layout,
    make_shape(Int<256>{}, Int<64>{}), Int<1>{}
  );

  // Launch kernel
  int smem_size = 128*64*2 + 256*64*2;  // A + B SMEM
  minimal_gemm_kernel<<<1, 128, smem_size>>>(
    tma_atom_A, tma_atom_B, d_C.data().get(), M, N, K
  );

  // Check results
  h_C = d_C;
  float expected = K;  // Since A and B are all 1.0
  bool success = (abs(h_C[0] - expected) < 0.01f);

  std::cout << "Result: " << h_C[0] << " (expected: " << expected << ")" << std::endl;
  std::cout << "Test " << (success ? "PASSED" : "FAILED") << std::endl;

  return 0;
}
```

### Compile and Run

```bash
nvcc -std=c++17 -arch=sm_100a \
     -I/path/to/cutlass/include \
     minimal_gemm.cu -o minimal_gemm

./minimal_gemm
```

---

## Tips for Using These Examples

### 1. Start Simple

Begin with host-side examples (`make_tiled_mma`, `tile_to_mma_shape`) before moving to device kernels.

### 2. Add Printing

Insert print statements to inspect intermediate values:

```cpp
if (thread0()) {
  print("tensor shape: "); print(shape(tensor)); print("\n");
}
```

### 3. Use CUDA-GDB

Debug device code:

```bash
cuda-gdb ./test_program
(gdb) break minimal_gemm_kernel
(gdb) run
(gdb) print tCtAcc
```

### 4. Check PTX

Inspect generated PTX:

```bash
nvcc --ptx -arch=sm_100a test.cu -o test.ptx
less test.ptx
```

Look for `tcgen05.mma` and `cp.async.bulk.tensor` instructions.

### 5. Verify SMEM Usage

Check SMEM requirements:

```bash
nvcc --ptxas-options=-v test.cu
```

Look for: `Used 16384 bytes smem`

---

## Common Issues and Solutions

### Issue: "No kernel image available"

**Solution**: Ensure `-arch=sm_100a` matches your GPU:
```bash
nvcc -arch=sm_100a ...  # For Blackwell (SM100)
```

### Issue: "Illegal memory access"

**Solution**: Check tensor bounds and layouts. Add bounds checking:
```cpp
assert(i < size(tensor));
```

### Issue: "TMA descriptor creation failed"

**Solution**: Verify SMEM layout is swizzled correctly:
```cpp
// Must use swizzled atoms
auto atom = UMMA::Layout_K_SW128_Atom<TypeA>{};  // Correct
// NOT: auto atom = Layout<Shape<_128, _16>>{};  // Wrong
```

### Issue: "TMEM allocation failed"

**Solution**: Only allocate TMEM in one thread:
```cpp
if (threadIdx.x / 32 == 0 && threadIdx.x % 32 == 0) {  // Warp 0, thread 0
  allocator.allocate(...);
}
```

---

## Next Steps

- Combine these examples with [05-host-functions.md](05-host-functions.md) and [06-device-functions.md](06-device-functions.md)
- Build incrementally: Start with single components, then combine
- Profile with Nsight Compute to understand performance

---

## Summary

These minimal examples provide:
- **Isolated testing** of individual APIs
- **Debugging starting points** for understanding behavior
- **Reference implementations** for building larger kernels

Each example can be modified and extended to explore different configurations, data types, and problem sizes.
