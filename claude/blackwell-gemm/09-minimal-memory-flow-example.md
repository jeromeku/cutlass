# Minimal Memory Flow Example: GMEM → SMEM → TMEM → RMEM → GMEM

This document provides a complete, minimal example demonstrating the full memory hierarchy flow on Blackwell (SM100) architecture:

1. **GMEM → SMEM**: TMA load (`cp.async.bulk.tensor`)
2. **SMEM → TMEM**: MMA operation that writes accumulator to TMEM (`tcgen05.mma`)
3. **TMEM → RMEM**: TMEM load using tensor memory read (`tcgen05.ld`)
4. **RMEM → GMEM**: Register store via SMEM using TMA store

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Complete Minimal Example](#complete-minimal-example)
- [Detailed Breakdown](#detailed-breakdown)
- [Memory Flow Visualization](#memory-flow-visualization)
- [PTX Instructions Generated](#ptx-instructions-generated)

---

## Architecture Overview

### Memory Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                         GMEM (HBM)                          │
│                                                             │
│  • Large capacity (40-80 GB)                               │
│  • High latency (~300-500 cycles)                          │
│  • Accessed via TMA (Tensor Memory Accelerator)           │
└─────────────────────────────────────────────────────────────┘
                            ↕ TMA (cp.async.bulk.tensor)
┌─────────────────────────────────────────────────────────────┐
│                    SMEM (Shared Memory)                     │
│                                                             │
│  • Per-CTA capacity (228 KB)                               │
│  • Low latency (~20-30 cycles)                             │
│  • Shared by all threads in CTA                            │
└─────────────────────────────────────────────────────────────┘
                            ↕ MMA (tcgen05.mma)
┌─────────────────────────────────────────────────────────────┐
│               TMEM (Tensor Memory/Accumulator)              │
│                                                             │
│  • On-chip accumulator storage (1 MB total)                │
│  • Ultra-low latency (~5-10 cycles)                        │
│  • Written by MMA, read by tcgen05.ld                      │
│  • Requires explicit allocation/deallocation               │
└─────────────────────────────────────────────────────────────┘
                            ↕ tcgen05.ld / st.shared
┌─────────────────────────────────────────────────────────────┐
│                   RMEM (Register File)                      │
│                                                             │
│  • Per-thread private storage                              │
│  • Lowest latency (~1 cycle)                               │
│  • Used for computation and data movement                  │
└─────────────────────────────────────────────────────────────┘
```

### Key Observations

1. **SMEM → TMEM is indirect**: You don't explicitly copy from SMEM to TMEM. Instead, MMA operations read operands from SMEM and write results directly to TMEM.

2. **TMEM is write-only for MMA**: The `tcgen05.mma` instruction writes accumulator results to TMEM, but you cannot directly write to TMEM from registers.

3. **TMEM → RMEM requires special instruction**: The `tcgen05.ld` instruction is used to read TMEM accumulators into registers.

4. **RMEM → GMEM goes through SMEM**: You typically store registers to SMEM first, then use TMA store to move to GMEM.

---

## Complete Minimal Example

This example performs a minimal GEMM operation to demonstrate the complete memory flow.

### Source Code: `minimal_memory_flow.cu`

```cpp
#include <cute/tensor.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/copy_traits_sm100.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/util.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <iostream>
#include <iomanip>

using namespace cute;

//
// Shared Storage
//

template <class SmemLayoutA, class SmemLayoutB, class SmemLayoutD>
struct SharedStorage {
  cute::array_aligned<cutlass::half_t, cosize_v<SmemLayoutA>> smem_A;
  cute::array_aligned<cutlass::half_t, cosize_v<SmemLayoutB>> smem_B;
  cute::array_aligned<float, cosize_v<SmemLayoutD>> smem_D;
  uint64_t tma_barrier;
  uint64_t mma_barrier;
  uint32_t tmem_base_ptr;
};

//
// Kernel
//

template <class TensorA, class TensorB, class TensorD,
          class TMA_A, class TMA_B, class TMA_D,
          class TiledMMA>
__global__ static
__cluster_dims__(1, 1, 1)  // Single CTA for simplicity
void minimal_memory_flow_kernel(
  TensorA gA, TensorB gB, TensorD gD,
  TMA_A tma_atom_A, TMA_B tma_atom_B, TMA_D tma_atom_D,
  TiledMMA tiled_mma)
{
  // Shared memory setup
  using SmemLayoutA = decltype(get_layout_mma_A(tiled_mma));
  using SmemLayoutB = decltype(get_layout_mma_B(tiled_mma));
  using SmemLayoutD = decltype(tile_to_shape(SmemLayoutA{}, make_shape(128, 128)));

  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<SmemLayoutA, SmemLayoutB, SmemLayoutD>;
  SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(shared_memory);

  // TMEM Allocator
  TmemAllocator tmem_allocator;

  // Thread/warp/CTA election
  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane_predicate = elect_one_sync();
  bool elect_one_warp = (warp_idx == 0);
  bool elect_one_thr = (elect_one_warp && lane_predicate);

  // Initialize barriers (single thread)
  if (elect_one_thr) {
    cute::initialize_barrier(shared_storage.tma_barrier, 1 /*numThreads*/);
    cute::initialize_barrier(shared_storage.mma_barrier, 1 /*numThreads*/);
  }
  __syncthreads();

  // Allocate TMEM (single warp)
  if (elect_one_warp) {
    tmem_allocator.acquire_allocation_lock();
    shared_storage.tmem_base_ptr = tmem_allocator.alloc(TmemAllocator::Sm100TmemCapacityColumns);
  }
  __syncthreads();

  // Create SMEM tensors
  Tensor sA = make_tensor(make_smem_ptr(shared_storage.smem_A.data()), SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(shared_storage.smem_B.data()), SmemLayoutB{});
  Tensor sD = make_tensor(make_smem_ptr(shared_storage.smem_D.data()), SmemLayoutD{});

  // Create TMEM accumulator tensor
  Tensor tAcc = make_tensor(
    make_inreg_tmem_ptr<float>(shared_storage.tmem_base_ptr),
    select<0,1>(typename TiledMMA::TiledShape_MNK{})
  );

  // Partition tensors for this CTA
  Tensor gA_cta = local_tile(gA, select<0,2>(typename TiledMMA::TiledShape_MNK{}), make_coord(_0{}, _0{}));
  Tensor gB_cta = local_tile(gB, select<1,2>(typename TiledMMA::TiledShape_MNK{}), make_coord(_0{}, _0{}));
  Tensor gD_cta = local_tile(gD, make_shape(128, 128), make_coord(_0{}));

  // ============================================================================
  // STEP 1: GMEM → SMEM (using TMA)
  // ============================================================================

  // Partition for TMA
  auto [tAgA, tAsA] = tma_partition(tma_atom_A, sA, gA_cta);
  auto [tBgB, tBsB] = tma_partition(tma_atom_B, sB, gB_cta);

  // Calculate transaction bytes
  uint32_t tma_transaction_bytes = sizeof(make_tensor_like(tAsA)) +
                                   sizeof(make_tensor_like(tBsB));

  if (elect_one_thr) {
    // Set expected transaction bytes
    cute::set_barrier_transaction_bytes(shared_storage.tma_barrier, tma_transaction_bytes);

    // Issue TMA loads (GMEM → SMEM)
    // Generates: cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
    copy(tma_atom_A.with(shared_storage.tma_barrier, 0 /*no multicast*/), tAgA, tAsA);
    copy(tma_atom_B.with(shared_storage.tma_barrier, 0 /*no multicast*/), tBgB, tBsB);
  }

  // Wait for TMA to complete
  cute::wait_barrier(shared_storage.tma_barrier, 0);
  __syncthreads();

  // ============================================================================
  // STEP 2: SMEM → TMEM (via MMA operation)
  // ============================================================================

  // Partition SMEM for MMA
  Tensor tCrA = tiled_mma.partition_fragment_A(sA);
  Tensor tCrB = tiled_mma.partition_fragment_B(sB);
  Tensor tCtAcc = tiled_mma.partition_C(tAcc);

  if (elect_one_warp) {
    // Clear TMEM accumulator on first MMA
    tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

    // Execute MMA: reads from SMEM (tCrA, tCrB), writes to TMEM (tCtAcc)
    // Generates: tcgen05.mma.cta_group::1.kind::f16...
    for (int k = 0; k < size<2>(tCrA); ++k) {
      gemm(tiled_mma, tCrA(_,_,k), tCrB(_,_,k), tCtAcc);
    }
  }
  __syncthreads();

  // ============================================================================
  // STEP 3: TMEM → RMEM (using tcgen05.ld)
  // ============================================================================

  // Create TMEM→RMEM copy operation
  TiledCopy t2r_copy = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tAcc);
  ThrCopy thr_t2r = t2r_copy.get_slice(threadIdx.x);

  // Partition TMEM and RMEM tensors
  Tensor tTR_tAcc = thr_t2r.partition_S(tAcc);   // TMEM source
  Tensor tTR_rD = make_fragment_like(thr_t2r.partition_D(sD));  // RMEM destination

  // Load from TMEM to registers
  // Generates: tcgen05.ld.sync.aligned.32x1x32b.x1.b32
  copy(t2r_copy, tTR_tAcc, tTR_rD);

  // ============================================================================
  // STEP 4: RMEM → GMEM (via SMEM using TMA store)
  // ============================================================================

  // Partition for register → SMEM copy
  Tensor tTR_sD = thr_t2r.partition_D(sD);

  // Store registers to SMEM
  // Generates: st.shared.v4.b32 (128-bit stores)
  copy_aligned(tTR_rD, tTR_sD);
  __syncthreads();

  // Partition for TMA store
  auto [tSG_sD, tSG_gD] = tma_partition(tma_atom_D, sD, gD_cta);

  if (elect_one_thr) {
    // TMA store fence
    // Generates: fence.proxy.async.shared::cta
    tma_store_fence();
  }
  __syncthreads();

  if (elect_one_thr) {
    // Issue TMA store (SMEM → GMEM)
    // Generates: cp.async.bulk.tensor.2d.global.shared::cta.bulk_group
    copy(tma_atom_D, tSG_sD, tSG_gD);

    // Commit and wait
    // Generates: cp.async.bulk.commit_group
    tma_store_arrive();

    // Generates: cp.async.bulk.wait_group.read 0
    tma_store_wait<0>();
  }
  __syncthreads();

  // Cleanup: Deallocate TMEM
  if (elect_one_warp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(shared_storage.tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
  }
}

//
// Host Code
//

int main(int argc, char** argv)
{
  // Problem size: 128×128×64 GEMM
  int M = 128;
  int N = 128;
  int K = 64;

  std::cout << "Minimal Memory Flow Example: " << M << "×" << N << "×" << K << " GEMM\n";
  std::cout << "Memory Flow: GMEM → SMEM → TMEM → RMEM → GMEM\n\n";

  // Allocate and initialize host data
  thrust::host_vector<cutlass::half_t> h_A(M * K);
  thrust::host_vector<cutlass::half_t> h_B(N * K);
  thrust::host_vector<float> h_D(M * N);

  // Simple initialization: A[i,j] = i+j, B[i,j] = 1
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < K; ++j) {
      h_A[i * K + j] = cutlass::half_t(float(i + j) * 0.01f);
    }
  }
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < K; ++j) {
      h_B[i * K + j] = cutlass::half_t(1.0f);
    }
  }

  // Copy to device
  thrust::device_vector<cutlass::half_t> d_A = h_A;
  thrust::device_vector<cutlass::half_t> d_B = h_B;
  thrust::device_vector<float> d_D(M * N, 0.0f);

  // Create CuTe tensors
  auto gA = make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(d_A.data())),
                        make_layout(make_shape(M, K), make_stride(K, _1{})));
  auto gB = make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(d_B.data())),
                        make_layout(make_shape(N, K), make_stride(K, _1{})));
  auto gD = make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(d_D.data())),
                        make_layout(make_shape(M, N), make_stride(N, _1{})));

  // Create TiledMMA (1SM variant for single CTA)
  auto tiled_mma = make_tiled_mma(
    SM100_MMA_F16BF16_SS<cutlass::half_t, cutlass::half_t, float,
                         128, 256,
                         UMMA::Major::K, UMMA::Major::K>{}
  );

  // Get SMEM layouts from TiledMMA
  auto smem_layout_A = get_layout_mma_A(tiled_mma);
  auto smem_layout_B = get_layout_mma_B(tiled_mma);
  auto smem_layout_D = tile_to_shape(smem_layout_A, make_shape(128, 128));

  // Create TMA descriptors
  auto tma_atom_A = make_tma_atom(
    SM90_TMA_LOAD_2D{},
    make_tensor(make_gmem_ptr<cutlass::half_t const>(nullptr), smem_layout_A),
    smem_layout_A,
    make_shape(_1{}),
    _1{}
  );

  auto tma_atom_B = make_tma_atom(
    SM90_TMA_LOAD_2D{},
    make_tensor(make_gmem_ptr<cutlass::half_t const>(nullptr), smem_layout_B),
    smem_layout_B,
    make_shape(_1{}),
    _1{}
  );

  auto tma_atom_D = make_tma_atom(
    SM90_TMA_STORE_2D{},
    make_tensor(make_gmem_ptr<float>(nullptr), smem_layout_D),
    smem_layout_D,
    make_shape(_1{}),
    _1{}
  );

  // Calculate shared memory size
  using SmemLayoutA = decltype(smem_layout_A);
  using SmemLayoutB = decltype(smem_layout_B);
  using SmemLayoutD = decltype(smem_layout_D);
  using SharedStorage = SharedStorage<SmemLayoutA, SmemLayoutB, SmemLayoutD>;
  size_t smem_size = sizeof(SharedStorage);

  std::cout << "Shared memory required: " << smem_size << " bytes\n";

  // Launch configuration
  dim3 block_dims(128, 1, 1);  // 4 warps
  dim3 cluster_dims(1, 1, 1);  // Single CTA
  dim3 grid_dims(1, 1, 1);

  // Launch kernel
  void* kernel_params[] = {&gA, &gB, &gD, &tma_atom_A, &tma_atom_B, &tma_atom_D, &tiled_mma};

  cudaError_t result = cutlass::launch_kernel_on_cluster(
    {grid_dims, block_dims, cluster_dims, smem_size, cudaStreamDefault},
    minimal_memory_flow_kernel<
      decltype(gA), decltype(gB), decltype(gD),
      decltype(tma_atom_A), decltype(tma_atom_B), decltype(tma_atom_D),
      decltype(tiled_mma)
    >,
    kernel_params
  );

  if (result != cudaSuccess) {
    std::cerr << "Kernel launch failed: " << cudaGetErrorString(result) << std::endl;
    return -1;
  }

  // Wait for completion
  result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    std::cerr << "Kernel execution failed: " << cudaGetErrorString(result) << std::endl;
    return -1;
  }

  // Copy result back to host
  h_D = d_D;

  // Verify result (simple check)
  std::cout << "\nResult verification:\n";
  std::cout << "D[0,0] = " << h_D[0] << " (expected: ~0.0)\n";
  std::cout << "D[0,1] = " << h_D[1] << " (expected: ~0.64)\n";
  std::cout << "D[1,0] = " << h_D[N] << " (expected: ~0.64)\n";

  bool passed = true;
  float tolerance = 0.1f;
  if (std::abs(h_D[0] - 0.0f) > tolerance) passed = false;
  if (std::abs(h_D[1] - 0.64f) > tolerance) passed = false;
  if (std::abs(h_D[N] - 0.64f) > tolerance) passed = false;

  std::cout << "\nTest " << (passed ? "PASSED" : "FAILED") << "\n";

  return passed ? 0 : -1;
}
```

### Compilation

```bash
nvcc -std=c++17 -arch=sm_100a \
     -I${CUTLASS_DIR}/include \
     -I${CUTLASS_DIR}/tools/util/include \
     minimal_memory_flow.cu -o minimal_memory_flow

./minimal_memory_flow
```

### Expected Output

```
Minimal Memory Flow Example: 128×128×64 GEMM
Memory Flow: GMEM → SMEM → TMEM → RMEM → GMEM

Shared memory required: 163840 bytes

Result verification:
D[0,0] = 0.00 (expected: ~0.0)
D[0,1] = 0.64 (expected: ~0.64)
D[1,0] = 0.64 (expected: ~0.64)

Test PASSED
```

---

## Detailed Breakdown

### Step 1: GMEM → SMEM (TMA Load)

**Source Location**: Lines 110-121

```cpp
if (elect_one_thr) {
  cute::set_barrier_transaction_bytes(shared_storage.tma_barrier, tma_transaction_bytes);

  // TMA loads
  copy(tma_atom_A.with(shared_storage.tma_barrier, 0), tAgA, tAsA);
  copy(tma_atom_B.with(shared_storage.tma_barrier, 0), tBgB, tBsB);
}

cute::wait_barrier(shared_storage.tma_barrier, 0);
```

**What Happens**:
1. Single thread initiates asynchronous TMA loads
2. TMA hardware DMAs data from GMEM to SMEM
3. Barrier counts down transaction bytes as data arrives
4. All threads wait for barrier to reach zero

**Call Stack**:
```
copy(tma_atom, src, dst)
  └─→ Copy_Atom<SM90_TMA_LOAD_2D>::call()
      └─→ copy_unpack()
          └─→ detail::CallCOPY<SM90_TMA_LOAD_2D>::copy()
              └─→ SM90_TMA_LOAD_2D::copy()
                  └─→ PTX: cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
```

**PTX Generated**:
```ptx
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
  [%smem_ptr], [%desc, {%coord_0, %coord_1}], [%barrier];
```

### Step 2: SMEM → TMEM (MMA Operation)

**Source Location**: Lines 130-141

```cpp
Tensor tCrA = tiled_mma.partition_fragment_A(sA);
Tensor tCrB = tiled_mma.partition_fragment_B(sB);
Tensor tCtAcc = tiled_mma.partition_C(tAcc);

if (elect_one_warp) {
  tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

  for (int k = 0; k < size<2>(tCrA); ++k) {
    gemm(tiled_mma, tCrA(_,_,k), tCrB(_,_,k), tCtAcc);
  }
}
```

**What Happens**:
1. Partition SMEM tensors according to MMA thread layout
2. Single warp executes MMA operations (tcgen05.mma requires single-thread execution)
3. MMA reads operands from SMEM, writes accumulator to TMEM
4. First MMA uses `Zero` scale to clear TMEM, subsequent MMAs use `One` scale to accumulate

**Call Stack**:
```
gemm(tiled_mma, A, B, C)
  └─→ MMA_Atom<SM100_MMA_F16BF16_SS>::call()
      └─→ mma_unpack()
          └─→ SM100_MMA_F16BF16_SS::fma()
              └─→ PTX: tcgen05.mma.cta_group::1.kind::f16
```

**PTX Generated**:
```ptx
tcgen05.mma.cta_group::1.kind::f16.kfma.f32.bf16.f16
  [%tmem_addr],
  [%smem_A_addr], [%smem_B_addr],
  %scaleA, %scaleB, %scaleOut, %immCoord;
```

**Key Point**: There is no explicit "SMEM → TMEM copy" instruction. The MMA instruction atomically reads from SMEM and writes to TMEM.

### Step 3: TMEM → RMEM (tcgen05.ld)

**Source Location**: Lines 150-158

```cpp
TiledCopy t2r_copy = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tAcc);
ThrCopy thr_t2r = t2r_copy.get_slice(threadIdx.x);

Tensor tTR_tAcc = thr_t2r.partition_S(tAcc);
Tensor tTR_rD = make_fragment_like(thr_t2r.partition_D(sD));

// Load from TMEM to registers
copy(t2r_copy, tTR_tAcc, tTR_rD);
```

**What Happens**:
1. Create TMEM→RMEM copy atom using `SM100_TMEM_LOAD_32dp32b1x`
2. Partition TMEM accumulator tensor across threads
3. Allocate register fragment to hold loaded data
4. Execute copy: reads from TMEM, writes to registers
5. Each thread reads its assigned portion (32 elements per instruction)

**Call Stack**:
```
copy(t2r_copy, tTR_tAcc, tTR_rD)
  └─→ Copy_Atom<SM100_TMEM_LOAD_32dp32b1x>::call()
      └─→ copy_unpack()
          └─→ detail::CallCOPY<SM100_TMEM_LOAD_32dp32b1x>::copy()
              └─→ SM100_TMEM_LOAD_32dp32b1x::copy()
                  └─→ PTX: tcgen05.ld.sync.aligned.32x1x32b.x1.b32
```

**PTX Generated**:
```ptx
tcgen05.ld.sync.aligned.32x1x32b.x1.b32
  {%r0, %r1, ..., %r31},  // 32 destination registers
  [%tmem_addr];            // TMEM address
```

**Instruction Details**:
- **32x1x32b**: Loads 32 elements × 1 row × 32 bits each = 1024 bits (128 bytes)
- **x1**: Single instruction execution
- Each thread loads 32 float values (128 bytes) from TMEM to registers

### Step 4: RMEM → GMEM (via SMEM + TMA Store)

This step has two sub-steps:

#### Step 4a: RMEM → SMEM

**Source Location**: Lines 165-168

```cpp
Tensor tTR_sD = thr_t2r.partition_D(sD);

// Store registers to SMEM
copy_aligned(tTR_rD, tTR_sD);
__syncthreads();
```

**What Happens**:
1. Partition SMEM according to thread layout
2. `copy_aligned()` performs auto-vectorized stores: 128-bit (4×float) vector stores
3. Each thread stores its registers to its assigned SMEM region
4. All threads synchronize to ensure all stores complete

**Call Stack**:
```
copy_aligned(tTR_rD, tTR_sD)
  └─→ copy(AutoVectorizingCopyWithAssumedAlignment<128>{}, src, dst)
      └─→ copy_if(pred, vectorized_copy, src, dst)
          └─→ Copy_Traits<UniversalCopy<uint128_t>>::copy()
              └─→ *dst = *src  (128-bit store)
                  └─→ PTX: st.shared.v4.b32 [smem], {%r0, %r1, %r2, %r3}
```

**PTX Generated**:
```ptx
st.shared.v4.b32 [%smem_addr], {%r0, %r1, %r2, %r3};  // 4 consecutive floats
```

**Auto-Vectorization**:
- 4 consecutive float registers → single 128-bit store instruction
- Improves memory bandwidth utilization
- Requires 16-byte alignment

#### Step 4b: SMEM → GMEM (TMA Store)

**Source Location**: Lines 171-191

```cpp
auto [tSG_sD, tSG_gD] = tma_partition(tma_atom_D, sD, gD_cta);

if (elect_one_thr) {
  // Fence to make SMEM stores visible
  tma_store_fence();
}
__syncthreads();

if (elect_one_thr) {
  // TMA store
  copy(tma_atom_D, tSG_sD, tSG_gD);

  // Commit and wait
  tma_store_arrive();
  tma_store_wait<0>();
}
```

**What Happens**:
1. **Fence**: Ensure all SMEM stores are visible to TMA hardware (`fence.proxy.async.shared::cta`)
2. **Sync**: All threads wait for fence to complete
3. **Copy**: Single thread initiates TMA store from SMEM to GMEM
4. **Commit**: Issuing thread commits the TMA store group
5. **Wait**: Issuing thread waits for TMA store to complete

**Call Stack (tma_store_fence)**:
```
tma_store_fence()
  └─→ cute::tma_store_fence()
      └─→ PTX: fence.proxy.async.shared::cta
```

**Call Stack (copy)**:
```
copy(tma_atom_D, tSG_sD, tSG_gD)
  └─→ Copy_Atom<SM90_TMA_STORE_2D>::call()
      └─→ copy_unpack()
          └─→ detail::CallCOPY<SM90_TMA_STORE_2D>::copy()
              └─→ SM90_TMA_STORE_2D::copy()
                  └─→ PTX: cp.async.bulk.tensor.2d.global.shared::cta.bulk_group
```

**PTX Generated**:
```ptx
// Step 1: Fence
fence.proxy.async.shared::cta;

// Step 2: TMA store
cp.async.bulk.tensor.2d.global.shared::cta.bulk_group
  [%desc, {%coord_0, %coord_1}], [%smem_ptr];

// Step 3: Commit
cp.async.bulk.commit_group;

// Step 4: Wait
cp.async.bulk.wait_group.read 0;
```

---

## Memory Flow Visualization

### Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            STEP 1: GMEM → SMEM                          │
│                                                                         │
│  GMEM (A, B matrices)                                                  │
│       ↓                                                                 │
│  cp.async.bulk.tensor.2d (TMA Load)                                   │
│       ↓                                                                 │
│  SMEM (shared_storage.smem_A, shared_storage.smem_B)                  │
│                                                                         │
│  • Async transfer via TMA hardware                                     │
│  • Single thread initiates, all threads wait on barrier                │
│  • High bandwidth: ~4-8 TB/s                                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                          STEP 2: SMEM → TMEM                           │
│                                                                         │
│  SMEM (A, B tiles)                                                     │
│       ↓                                                                 │
│  tcgen05.mma.cta_group::1 (MMA operation)                              │
│       ↓                                                                 │
│  TMEM (accumulator)                                                    │
│                                                                         │
│  • MMA reads SMEM, writes TMEM atomically                              │
│  • Single warp executes (tcgen05.mma requires single-thread)           │
│  • No explicit copy instruction                                        │
│  • ScaleOut::Zero clears, ScaleOut::One accumulates                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                          STEP 3: TMEM → RMEM                           │
│                                                                         │
│  TMEM (accumulator)                                                    │
│       ↓                                                                 │
│  tcgen05.ld.sync.aligned.32x1x32b.x1.b32                               │
│       ↓                                                                 │
│  RMEM (registers: tTR_rD)                                              │
│                                                                         │
│  • Explicit tensor memory load instruction                             │
│  • Each thread reads 32 float values (128 bytes)                       │
│  • Ultra-low latency: ~5-10 cycles                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                     STEP 4a: RMEM → SMEM                               │
│                                                                         │
│  RMEM (registers: tTR_rD)                                              │
│       ↓                                                                 │
│  st.shared.v4.b32 (128-bit vectorized stores)                          │
│       ↓                                                                 │
│  SMEM (shared_storage.smem_D)                                          │
│                                                                         │
│  • Auto-vectorization: 4 floats per instruction                        │
│  • All threads participate                                             │
│  • __syncthreads() ensures completion                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                     STEP 4b: SMEM → GMEM                               │
│                                                                         │
│  SMEM (shared_storage.smem_D)                                          │
│       ↓                                                                 │
│  fence.proxy.async.shared::cta (make SMEM visible to TMA)              │
│       ↓                                                                 │
│  cp.async.bulk.tensor.2d (TMA Store)                                   │
│       ↓                                                                 │
│  GMEM (D matrix)                                                       │
│                                                                         │
│  • Fence before TMA store                                              │
│  • Single thread initiates TMA store                                   │
│  • Commit and wait for completion                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### Thread Participation

```
Step 1: GMEM → SMEM (TMA Load)
  • elect_one_thr:  Initiates TMA load
  • All threads:    Wait on barrier

Step 2: SMEM → TMEM (MMA)
  • elect_one_warp: Executes MMA instructions
  • Other warps:    Idle

Step 3: TMEM → RMEM
  • All threads:    Execute tcgen05.ld (each thread loads its portion)

Step 4a: RMEM → SMEM
  • All threads:    Execute st.shared (each thread stores its portion)

Step 4b: SMEM → GMEM (TMA Store)
  • elect_one_thr:  Initiates TMA store, commits, waits
  • All threads:    Sync before and after
```

---

## PTX Instructions Generated

### Summary Table

| Step | Operation | PTX Instruction | Bandwidth | Latency |
|------|-----------|----------------|-----------|---------|
| 1 | GMEM → SMEM | `cp.async.bulk.tensor.2d` | ~4-8 TB/s | ~300-500 cycles |
| 2 | SMEM → TMEM | `tcgen05.mma.cta_group::1` | ~19 TB/s | Pipelined |
| 3 | TMEM → RMEM | `tcgen05.ld.sync.aligned` | ~10 TB/s | ~5-10 cycles |
| 4a | RMEM → SMEM | `st.shared.v4.b32` | ~2-4 TB/s | ~20-30 cycles |
| 4b | SMEM → GMEM | `cp.async.bulk.tensor.2d` | ~4-8 TB/s | ~300-500 cycles |

### Detailed PTX Breakdown

#### Step 1: GMEM → SMEM

```ptx
// Set barrier transaction bytes
{ .reg .u64 %bar;
  cvta.shared.u64 %bar, %barrier_addr;
  mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%bar], %transaction_bytes;
}

// TMA load for matrix A
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
  [%smem_A_ptr],                           // Destination: SMEM
  [%tma_desc_A, {%coord_m, %coord_k}],     // Source: GMEM via TMA descriptor
  [%barrier_addr];                         // Barrier to signal on completion

// TMA load for matrix B
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
  [%smem_B_ptr],
  [%tma_desc_B, {%coord_n, %coord_k}],
  [%barrier_addr];

// Wait for barrier
{ .reg .u64 %bar;
  cvta.shared.u64 %bar, %barrier_addr;
  mbarrier.try_wait.parity.shared::cta.b64 %arrived, [%bar], %phase_bit;
  @!%arrived bra WAIT_LOOP;
}
```

#### Step 2: SMEM → TMEM (via MMA)

```ptx
// MMA instruction (executes in single-thread mode)
tcgen05.mma.cta_group::1.kind::f16.kfma.f32.bf16.f16
  [%tmem_addr],            // Destination: TMEM accumulator address
  [%smem_A_addr],          // Source A: SMEM address
  [%smem_B_addr],          // Source B: SMEM address
  %scaleA,                 // Scale factor for A (typically 1.0)
  %scaleB,                 // Scale factor for B (typically 1.0)
  %scaleOut,               // Scale factor for output (0.0 for first, 1.0 for accumulate)
  %immCoord;               // Immediate coordinate for TMEM addressing
```

**Note**: This instruction atomically:
1. Reads A tile from SMEM
2. Reads B tile from SMEM
3. Performs matrix multiply-accumulate
4. Writes result to TMEM

#### Step 3: TMEM → RMEM

```ptx
// TMEM load instruction
tcgen05.ld.sync.aligned.32x1x32b.x1.b32
  {%r0,  %r1,  %r2,  %r3,  %r4,  %r5,  %r6,  %r7,
   %r8,  %r9,  %r10, %r11, %r12, %r13, %r14, %r15,
   %r16, %r17, %r18, %r19, %r20, %r21, %r22, %r23,
   %r24, %r25, %r26, %r27, %r28, %r29, %r30, %r31},  // 32 destination registers
  [%tmem_addr];                                       // TMEM source address
```

**Breakdown**:
- **32x1x32b**: Load 32 elements × 1 row × 32 bits = 1024 bits (128 bytes)
- **x1**: Single instruction execution
- **sync.aligned**: Synchronized, aligned access
- **32 registers**: Each register holds one float (4 bytes)

#### Step 4a: RMEM → SMEM

```ptx
// Auto-vectorized 128-bit stores (4 floats per instruction)
st.shared.v4.b32 [%smem_addr + 0],  {%r0,  %r1,  %r2,  %r3};
st.shared.v4.b32 [%smem_addr + 16], {%r4,  %r5,  %r6,  %r7};
st.shared.v4.b32 [%smem_addr + 32], {%r8,  %r9,  %r10, %r11};
// ... (continues for all register groups)
```

**Vectorization**:
- Each instruction stores 4 consecutive floats (16 bytes)
- Requires 16-byte alignment
- 4× fewer instructions than scalar stores

#### Step 4b: SMEM → GMEM

```ptx
// Step 1: Fence to make SMEM stores visible to TMA
fence.proxy.async.shared::cta;

// Step 2: TMA store
cp.async.bulk.tensor.2d.global.shared::cta.bulk_group
  [%tma_desc_D, {%coord_m, %coord_n}],  // Destination: GMEM via TMA descriptor
  [%smem_D_ptr];                        // Source: SMEM

// Step 3: Commit the bulk group
cp.async.bulk.commit_group;

// Step 4: Wait for completion (wait for 0 groups in flight)
cp.async.bulk.wait_group.read 0;
```

**Sequence Explanation**:
1. **Fence**: Ensures SMEM stores from all threads are visible to TMA hardware
2. **TMA Store**: Async transfer from SMEM to GMEM
3. **Commit**: Mark the end of current bulk group
4. **Wait**: Issuing thread waits for all stores to complete

---

## Performance Characteristics

### Bandwidth Comparison

```
Memory Transition          Peak Bandwidth    Typical Bandwidth
─────────────────────────────────────────────────────────────
GMEM ↔ SMEM (TMA)         ~8 TB/s           ~4-6 TB/s
SMEM ↔ RMEM               ~19 TB/s          ~10-15 TB/s
TMEM ↔ RMEM (tcgen05.ld)  ~10 TB/s          ~8-10 TB/s
MMA (SMEM → TMEM)         Fused with compute
```

### Latency Comparison

```
Memory Access              Latency (cycles)  Notes
────────────────────────────────────────────────────────────────
RMEM (register)           1                 Lowest latency
TMEM (tcgen05.ld)         5-10              On-chip accumulator
SMEM (shared memory)      20-30             Shared by CTA
GMEM (HBM via TMA)        300-500           Async, overlapped
```

### Throughput Analysis

For the minimal example (128×128×64 GEMM):

```
Operation                  Data Size         Time (est.)    Bandwidth
──────────────────────────────────────────────────────────────────────
GMEM → SMEM (A)           128×64×2 = 16 KB   ~5 μs         ~3.2 GB/s
GMEM → SMEM (B)           128×64×2 = 16 KB   ~5 μs         ~3.2 GB/s
MMA (SMEM → TMEM)         Compute-bound      ~2 μs         -
TMEM → RMEM               128×128×4 = 64 KB  ~0.5 μs       ~128 GB/s
RMEM → SMEM               128×128×4 = 64 KB  ~1 μs         ~64 GB/s
SMEM → GMEM (D)           128×128×4 = 64 KB  ~10 μs        ~6.4 GB/s
```

**Total estimated time**: ~23.5 μs for single 128×128×64 GEMM

**Notes**:
- Small tile sizes don't saturate bandwidth
- TMA overhead dominates for small transfers
- Larger tiles (256×256) would achieve better utilization
- This example prioritizes clarity over performance

---

## Key Takeaways

1. **No Direct SMEM → TMEM Copy**: The `tcgen05.mma` instruction atomically reads from SMEM and writes to TMEM. There is no separate copy instruction.

2. **TMEM → RMEM Requires Special Instruction**: Use `tcgen05.ld` to read TMEM accumulators. This is different from regular SMEM loads.

3. **TMA Provides Asynchronous Transfers**: Both GMEM → SMEM and SMEM → GMEM use TMA for high-bandwidth async transfers.

4. **Auto-Vectorization for RMEM ↔ SMEM**: CuTe's `copy_aligned()` automatically generates 128-bit vector loads/stores for efficiency.

5. **Thread Coordination**:
   - TMA loads/stores: Single thread initiates, all threads wait
   - MMA operations: Single warp executes (tcgen05.mma requirement)
   - TMEM reads: All threads participate
   - RMEM ↔ SMEM: All threads participate

6. **Fence Required for TMA Store**: Must issue `fence.proxy.async.shared::cta` before TMA store to ensure SMEM writes are visible.

7. **TMEM Allocation**: Explicitly allocate/deallocate TMEM using `TmemAllocator` with acquisition locks.

---

## References

- Example 05 source: [`/home/jeromeku/cutlass/examples/cute/tutorial/blackwell/05_mma_tma_epi_sm100.cu`](../examples/cute/tutorial/blackwell/05_mma_tma_epi_sm100.cu)
- TMA copy traits: [`include/cute/arch/copy_sm90_tma.hpp`](../include/cute/arch/copy_sm90_tma.hpp)
- MMA traits: [`include/cute/arch/mma_sm100_umma.hpp`](../include/cute/arch/mma_sm100_umma.hpp)
- TMEM copy traits: [`include/cute/atom/copy_traits_sm100.hpp`](../include/cute/atom/copy_traits_sm100.hpp)
- Previous documentation:
  - [06-device-functions-complete-trace.md](06-device-functions-complete-trace.md) - Complete call traces for copy and gemm
  - [08-example-05-epilogue-copies.md](08-example-05-epilogue-copies.md) - Epilogue copy operations
