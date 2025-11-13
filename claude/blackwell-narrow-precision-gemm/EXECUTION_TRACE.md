# Complete Execution Trace: Blackwell FP4 GEMM

**Frame-by-Frame Execution Flow with Full Template Unrolling**

This document provides a complete, unrolled trace of the Blackwell narrow precision GEMM execution, from host API call through kernel launch, device execution, and down to PTX instructions.

## Overview of Execution

```
Host Side:
  gemm.run()
    → GemmUniversalAdapter::run()
      → kernel_launch()
        → cudaLaunchKernelEx()

Device Side:
  GemmUniversal::operator()
    → Warp Specialization
      → Producer Warps: CollectiveMma::load() [TMA]
      → Consumer Warps: CollectiveMma::mma() [UMMA+ScaleFactors]
      → Epilogue Warps: CollectiveEpilogue::store() [Fusion+Quantization]
```

---

## Part 1: Host-Side Execution Trace

### Frame 1: User Call - `gemm.run()`

**Location**: [examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu:510](../../examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu#L510)

```cpp
// User code
CUTLASS_CHECK(gemm.run());
```

**Template Instantiation**:
```cpp
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

where GemmKernel = cutlass::gemm::kernel::GemmUniversal<
  Shape<int,int,int,int>,          // ProblemShape (M, N, K, L)
  CollectiveMainloop,              // Instantiated mainloop
  CollectiveEpilogue,              // Instantiated epilogue
  void                             // TileScheduler
>;
```

---

### Frame 2: GemmUniversalAdapter::run()

**Location**: [include/cutlass/gemm/device/gemm_universal_adapter.h:748-753](../../include/cutlass/gemm/device/gemm_universal_adapter.h#L748-L753)

```cpp
Status run(
  cudaStream_t stream = nullptr,
  CudaHostAdapter *cuda_adapter = nullptr) {

  return underlying_operator_.run(stream, cuda_adapter);
}
```

**Call Stack**:
```
gemm.run()
  ├─ underlying_operator_ is DeviceKernel<GemmKernel>
  └─ calls DeviceKernel::run()
```

---

### Frame 3: DeviceKernel::run() - Kernel Launch

**Location**: [include/cutlass/device_kernel.h](../../include/cutlass/device_kernel.h)

```cpp
Status run(cudaStream_t stream, CudaHostAdapter *cuda_adapter) {

  dim3 const grid_dims = get_grid_dims(params_);
  dim3 const block_dims = get_block_dims();
  int const smem_size = get_smem_size();

  // Get kernel function pointer
  void const* kernel_ptr = (void const*) device_kernel<Kernel>;

  // Launch with cluster support
  return ClusterLauncher::launch(
    grid_dims,
    block_dims,
    cluster_shape_,
    smem_size,
    stream,
    kernel_ptr,
    params_
  );
}
```

**Grid Configuration for our example**:
```
Problem: M=2048, N=2048, K=2048
Tile: 128x128x256
Cluster: 1x1x1

Grid dimensions:
  - gridDim.x = ceil_div(2048, 128) * ceil_div(2048, 128) = 16 * 16 = 256 CTAs
  - gridDim.y = 1 (batch)
  - gridDim.z = 1

Block dimensions:
  - NumSchedThreads        = 32  (1 warp)
  - NumMainloopLoadThreads = 32  (1 warp)
  - NumMMAThreads          = 32  (1 warp)
  - NumEpilogueLoadThreads = 32  (1 warp)
  - NumEpilogueThreads     = variable (depends on epilogue config)
  - Total: ~160-192 threads per CTA

Shared memory: ~64-96 KB (for pipelines, barriers, epilogue scratch)
```

---

### Frame 4: ClusterLauncher::launch()

**Location**: [include/cutlass/cluster_launch.hpp](../../include/cutlass/cluster_launch.hpp)

```cpp
static Status launch(
    dim3 const grid_dims,
    dim3 const block_dims,
    dim3 const cluster_dims,
    int smem_size,
    cudaStream_t stream,
    void const* kernel,
    Params const& params) {

  cudaLaunchConfig_t launch_config;
  cudaLaunchAttribute launch_attribute[2];

  // Set cluster dimensions
  launch_attribute[0].id = cudaLaunchAttributeClusterDimension;
  launch_attribute[0].val.clusterDim.x = cluster_dims.x;  // 1
  launch_attribute[0].val.clusterDim.y = cluster_dims.y;  // 1
  launch_attribute[0].val.clusterDim.z = cluster_dims.z;  // 1

  // Set shared memory configuration
  launch_attribute[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
  launch_attribute[1].val.clusterSchedulingPolicyPreference =
      cudaClusterSchedulingPolicySpread;

  launch_config.gridDim = grid_dims;
  launch_config.blockDim = block_dims;
  launch_config.dynamicSmemBytes = smem_size;
  launch_config.stream = stream;
  launch_config.attrs = launch_attribute;
  launch_config.numAttrs = 2;

  // Actual kernel launch
  return cudaLaunchKernelEx(
      &launch_config,
      kernel,
      &params);  // ← Params copied to constant memory
}
```

**Summary**:
- Sets up cluster launch configuration
- Configures shared memory
- Launches kernel with `cudaLaunchKernelEx`
- **Control transfers to GPU**

---

## Part 2: Device-Side Kernel Entry

### Frame 5: GemmUniversal::operator()  - Kernel Entry Point

**Location**: [include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp:406-958](../../include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp#L406-L958)

**Function Signature**:
```cpp
CUTLASS_DEVICE void
operator() (Params const& params, char* smem_buf)
```

**Execution begins** for each thread in each CTA:

```cpp
// Each thread executes this code
{
  using namespace cute;
  using X = Underscore;

  // Line 413: Extract problem shape
  auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
  auto [M,N,K,L] = problem_shape_MNKL;
  // For our example: M=2048, N=2048, K=2048, L=1

  // Line 417: Determine warp role
  int warp_idx = canonical_warp_idx_sync();
  // Each warp gets assigned a role based on warp_idx

  WarpCategory warp_category = warp_idx < static_cast<int>(WarpCategory::Epilogue)
                                   ? WarpCategory(warp_idx)
                                   : WarpCategory::Epilogue;

  // Warp roles:
  // warp_idx == 0 → WarpCategory::MMA
  // warp_idx == 1 → WarpCategory::Sched
  // warp_idx == 2 → WarpCategory::MainloopLoad
  // warp_idx == 3 → WarpCategory::EpilogueLoad
  // warp_idx >= 4 → WarpCategory::Epilogue

  // Line 421: Elect one thread per warp as leader
  uint32_t lane_predicate = cute::elect_one_sync();
  // Only lane 0 in each warp will have lane_predicate = true

  // Line 422-427: Cluster information
  auto cluster_shape = cutlass::detail::select_cluster_shape(ClusterShape{});
  // cluster_shape = (1, 1, 1) for our example

  int cluster_size = size(cluster_shape);  // = 1
  uint32_t cta_rank_in_cluster = cute::block_rank_in_cluster();  // 0 for all CTAs
  bool is_first_cta_in_cluster = cta_rank_in_cluster == 0;  // true

  // Line 426-429: 2SM MMA coordination
  int cta_coord_v = cta_rank_in_cluster % size<0>(typename TiledMma::AtomThrID{});
  bool is_mma_leader_cta = cta_coord_v == 0;  // true for 1SM mode
  constexpr bool has_mma_peer_cta = size(AtomThrShapeMNK{}) == 2;  // false for 1SM

  // Line 432: Get shared memory
  SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

  // Line 435-436: Instantiate collectives
  CollectiveMainloop collective_mainloop(params.mainloop, cluster_shape, cta_rank_in_cluster);
  CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.epilogue);
}
```

**Key Insight**: Each CTA has ~160-192 threads organized into warps, and each warp immediately knows its specialized role.

---

### Frame 6: Warp Specialization - Role Assignment

**Location**: [include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp:448-454](../../include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp#L448-L454)

```cpp
// Line 448-454: Determine which warps participate
IsParticipant is_participant = {
  (warp_category == WarpCategory::MMA),                                 // mma
  (warp_category == WarpCategory::Sched) && is_first_cta_in_cluster,    // sched
  (warp_category == WarpCategory::MainloopLoad),                        // main_load
  (warp_category == WarpCategory::EpilogueLoad) && is_epi_load_needed,  // epi_load
  (warp_category == WarpCategory::Epilogue)                             // epilogue
};
```

**Warp Assignment Table**:

| Warp ID | Category | Participates? | Role |
|---------|----------|---------------|------|
| 0 | MMA | Yes | Consumer - executes MMA instructions |
| 1 | Sched | Yes | Scheduler - manages tile distribution |
| 2 | MainloopLoad | Yes | Producer - loads A, B, SFA, SFB via TMA |
| 3 | EpilogueLoad | Maybe | Loads C matrix if needed |
| 4+ | Epilogue | Yes | Fusion, quantization, store D |

---

### Frame 7: Pipeline Initialization

**Location**: [include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp:456-551](../../include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp#L456-L551)

```cpp
// Line 457-471: Mainloop pipeline (Producer-Consumer for A/B/SFA/SFB)
typename MainloopPipeline::Params mainloop_pipeline_params;
if (WarpCategory::MainloopLoad == warp_category) {
  mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
}
if (WarpCategory::MMA == warp_category) {
  mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
}
mainloop_pipeline_params.is_leader = lane_predicate && is_mma_leader_cta && is_participant.main_load;
mainloop_pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytes;
mainloop_pipeline_params.initializing_warp = 0;

MainloopPipeline mainloop_pipeline(
  shared_storage.pipelines.mainloop,
  mainloop_pipeline_params,
  cluster_shape,
  cute::true_type{},   // Perform barrier init
  cute::false_type{}); // Delay mask calculation

// Similar initialization for:
// - EpiLoadPipeline (lines 473-486)
// - EpiStorePipeline (lines 488-491)
// - LoadOrderBarrier (lines 493-498)
// - CLCPipeline (lines 500-517)
// - AccumulatorPipeline (lines 519-535)
// - CLCThrottlePipeline (lines 537-551)
```

**Pipeline Types**:
1. **MainloopPipeline**: Synchronizes Producer (load A/B/SF) and Consumer (MMA) warps
2. **AccumulatorPipeline**: Passes accumulator from MMA to Epilogue warps
3. **EpiLoadPipeline**: Epilogue load warp → epilogue compute warps
4. **CLCPipeline**: Cluster-Level Controller for work distribution

---

### Frame 8: TMEM Allocation

**Location**: [include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp:553-576](../../include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp#L553-L576)

```cpp
// Line 554: Create TMEM allocator
TmemAllocator tmem_allocator{};
// TmemAllocator = Allocator1Sm or Allocator2Sm based on atom shape

// Line 557-575: Barriers for TMEM synchronization
arch::NamedBarrier tmem_allocation_result_barrier(
  NumMMAThreads + NumEpilogueThreads,
  cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier);

arch::ClusterBarrier& tmem_deallocation_result_barrier = shared_storage.pipelines.tmem_dealloc;

// Initialize deallocation barrier (MMA warp only)
if (WarpCategory::MMA == warp_category) {
  if (has_mma_peer_cta && lane_predicate) {
    if constexpr (!IsOverlappingAccum) {
      tmem_deallocation_result_barrier.init(NumMMAThreads);
    } else {
      tmem_deallocation_result_barrier.init(NumEpilogueThreads*2);
    }
  } else if (lane_predicate) {
    tmem_deallocation_result_barrier.init(NumEpilogueThreads);
  }
}

// Line 579: Cluster-wide barrier for pipeline init
pipeline_init_arrive_relaxed(cluster_size);
```

---

### Frame 9: Load and MMA Initialization

**Location**: [include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp:581-614](../../include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp#L581-L614)

```cpp
// Line 581-582: Initialize load parameters
auto load_inputs = collective_mainloop.load_init(
  problem_shape_MNKL, shared_storage.tensors.mainloop);
// Returns: TMA descriptors, tensor views, multicast masks

// Line 584-597: Initialize pipeline states
MainloopPipelineState mainloop_pipe_consumer_state;
MainloopPipelineState mainloop_pipe_producer_state = cutlass::make_producer_start_state<MainloopPipeline>();

EpiLoadPipelineState epi_load_pipe_consumer_state;
EpiLoadPipelineState epi_load_pipe_producer_state = cutlass::make_producer_start_state<EpiLoadPipeline>();

EpiStorePipelineState epi_store_pipe_producer_state = cutlass::make_producer_start_state<EpiStorePipeline>();

CLCPipelineState clc_pipe_consumer_state;
CLCPipelineState clc_pipe_producer_state = cutlass::make_producer_start_state<CLCPipeline>();

AccumulatorPipelineState accumulator_pipe_consumer_state;
AccumulatorPipelineState accumulator_pipe_producer_state = cutlass::make_producer_start_state<AccumulatorPipeline>();

// Line 599-603: Finalize pipeline initialization
dim3 block_id_in_cluster = cute::block_id_in_cluster();
mainloop_pipeline.init_masks(cluster_shape, block_id_in_cluster);
accumulator_pipeline.init_masks(cluster_shape, block_id_in_cluster);

// Line 605-607: Tile scheduler
TileScheduler scheduler(&shared_storage.clc_response[0], params.scheduler, block_id_in_cluster);
typename TileScheduler::WorkTileInfo work_tile_info = scheduler.initial_work_tile_info(cluster_shape);
auto cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);

// Line 612: Allocate TMEM tensors for accumulators and scale factors
auto tmem_storage = collective_mainloop.template init_tmem_tensors<EpilogueTile, IsOverlappingAccum>(EpilogueTile{});

// Line 614: Final barrier wait
pipeline_init_wait(cluster_size);
```

---

## Part 3: Warp-Specialized Execution

Now each warp diverges based on its category and enters its specialized code path.

---

### PRODUCER WARP (Warp 2): MainloopLoad

**Location**: [include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp:616-678](../../include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp#L616-L678)

#### Frame 10: Producer Main Loop

```cpp
if (is_participant.main_load) {
  // Line 619: Wait for dependent grids
  cutlass::arch::wait_on_dependent_grids();

  bool do_load_order_arrive = is_epi_load_needed;
  bool requires_clc_query = true;

  // Line 624: Work loop - process tiles
  do {
    // Line 626-628: Get K tile info from scheduler
    auto k_tile_iter = scheduler.get_k_tile_iterator(
      work_tile_info, problem_shape_MNKL, CtaShape_MNK{}, load_inputs.k_tiles);
    auto k_tile_count = TileScheduler::get_work_k_tile_count(
      work_tile_info, problem_shape_MNKL, CtaShape_MNK{});
    auto k_tile_prologue = min(MainloopPipeline::Stages, k_tile_count);

    // For our example with K=2048, TileK=256:
    // k_tile_count = 2048 / 256 = 8 K-tiles per work unit
    // k_tile_prologue = min(Stages, 8) = min(~20, 8) = 8

    // Line 639-645: Load prologue stages
    auto [mainloop_producer_state_next, k_tile_iter_next] = collective_mainloop.load(
      mainloop_pipeline,
      mainloop_pipe_producer_state,
      load_inputs,
      cta_coord_mnkl,
      k_tile_iter,
      k_tile_prologue  // Load first 8 stages
    );
    mainloop_pipe_producer_state = mainloop_producer_state_next;

    // Line 648-651: Signal epilogue load can start
    if (do_load_order_arrive) {
      load_order_barrier.arrive();
      do_load_order_arrive = false;
    }

    // Line 653-660: Load remaining stages
    auto [mainloop_producer_state_next_, unused_] = collective_mainloop.load(
      mainloop_pipeline,
      mainloop_pipe_producer_state,
      load_inputs,
      cta_coord_mnkl,
      k_tile_iter_next,
      k_tile_count - k_tile_prologue  // = 0 in this case
    );
    mainloop_pipe_producer_state = mainloop_producer_state_next_;

    // Line 663-674: Fetch next work tile
    __syncwarp();
    auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
      work_tile_info,
      clc_pipeline,
      clc_pipe_consumer_state
    );
    work_tile_info = next_work_tile_info;
    cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
    requires_clc_query = increment_pipe;
    if (increment_pipe) {
      ++clc_pipe_consumer_state;
    }
  } while (work_tile_info.is_valid());

  // Line 676: Tail - wait for all stages to complete
  collective_mainloop.load_tail(mainloop_pipeline, mainloop_pipe_producer_state);
}
```

---

#### Frame 11: CollectiveMma::load() - TMA Operations

**Location**: [include/cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp:875-922](../../include/cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp#L875-L922)

```cpp
CUTLASS_DEVICE auto
load(
  MainloopPipeline mainloop_pipeline,
  MainloopPipelineState mainloop_pipe_producer_state,
  LoadParams const& load_inputs,
  TileCoordMNKL const& cta_coord_mnkl,
  KTileIterator k_tile_iter,
  int k_tile_count) {

  // Line 882-885: Extract load parameters
  auto [unused_k_tiles,
        tAgA_mkl, tBgB_nkl, tAsA, tBsB,
        tAgSFA_mkl, tBgSFB_nkl, tAsSFA, tBsSFB,
        mcast_mask_a, mcast_mask_b, mcast_mask_sfa, mcast_mask_sfb] = load_inputs;

  // Line 888-891: Slice tensor views for this CTA
  Tensor tAgA = tAgA_mkl(_, get<0>(cta_coord_mnkl) / size(typename TiledMma::AtomThrID{}), _, get<3>(cta_coord_mnkl));
  Tensor tBgB = tBgB_nkl(_, get<1>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl));
  Tensor tAgSFA = tAgSFA_mkl(_, get<0>(cta_coord_mnkl) / size(typename TiledMma::AtomThrID{}), _, get<3>(cta_coord_mnkl));
  Tensor tBgSFB = tBgSFB_nkl(_, get<1>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl));

  // Line 893: Try to acquire first barrier slot
  auto barrier_token = mainloop_pipeline.producer_try_acquire(mainloop_pipe_producer_state);

  // Line 896-919: Load loop
  CUTLASS_PRAGMA_NO_UNROLL
  while (k_tile_count > 0) {
    // Line 899: Acquire barrier for this stage
    mainloop_pipeline.producer_acquire(mainloop_pipe_producer_state, barrier_token);

    using BarrierType = typename MainloopPipeline::ProducerBarrierType;
    BarrierType* tma_barrier = mainloop_pipeline.producer_get_barrier(mainloop_pipe_producer_state);

    int write_stage = mainloop_pipe_producer_state.index();
    ++mainloop_pipe_producer_state;
    barrier_token = mainloop_pipeline.producer_try_acquire(mainloop_pipe_producer_state);

    // Line 910-915: Issue TMA loads (elected leader thread only)
    if (cute::elect_one_sync()) {
      // TMA load A tile (128x256 elements of FP4 = 16KB)
      copy(observed_tma_load_a_->with(*tma_barrier, mcast_mask_a),
           tAgA(_,*k_tile_iter),
           tAsA(_,write_stage));

      // TMA load B tile (128x256 elements of FP4 = 16KB)
      copy(observed_tma_load_b_->with(*tma_barrier, mcast_mask_b),
           tBgB(_,*k_tile_iter),
           tBsB(_,write_stage));

      // TMA load SFA (1x16 scale factors = 16 bytes)
      copy(observed_tma_load_sfa_->with(*tma_barrier, mcast_mask_sfa),
           tAgSFA(_,*k_tile_iter),
           tAsSFA(_,write_stage));

      // TMA load SFB (1x16 scale factors = 16 bytes)
      copy(observed_tma_load_sfb_->with(*tma_barrier, mcast_mask_sfb),
           tBgSFB(_,*k_tile_iter),
           tBsSFB(_,write_stage));
    }

    --k_tile_count;
    ++k_tile_iter;
  }

  return cute::make_tuple(mainloop_pipe_producer_state, k_tile_iter);
}
```

**TMA Copy Explanation**:

The `copy()` calls here are **asynchronous TMA (Tensor Memory Accelerator) operations**:

1. **TMA Descriptor**: `observed_tma_load_a_` contains pre-computed TMA descriptor
2. **Barrier**: `*tma_barrier` - hardware barrier that TMA will signal when complete
3. **Multicast Mask**: `mcast_mask_a` - which CTAs in cluster receive the data
4. **Source**: `tAgA(_,*k_tile_iter)` - global memory location
5. **Destination**: `tAsA(_,write_stage)` - SMEM location for this pipeline stage

**Hardware Action**:
- TMA unit reads 128×256×4bits = 16,384 bytes from GMEM
- Writes directly to SMEM (bypassing L1/L2 if needed)
- Signals barrier when complete
- All happens asynchronously while thread continues

---

### CONSUMER WARP (Warp 0): MMA

**Location**: [include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp:725-804](../../include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp#L725-L804)

#### Frame 12: MMA Warp Main Loop

```cpp
else if (is_participant.mma) {
  // Line 727-731: Allocate TMEM
  tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &shared_storage.tmem_base_ptr);
  __syncwarp();
  tmem_allocation_result_barrier.arrive();
  uint32_t tmem_base_ptr = shared_storage.tmem_base_ptr;
  collective_mainloop.set_tmem_offsets(tmem_storage, tmem_base_ptr);

  // Line 733-735: Initialize MMA
  auto mma_inputs = collective_mainloop.mma_init(
    tmem_storage,
    shared_storage.tensors.mainloop);

  // Line 737-775: Work loop
  do {
    auto k_tile_count = TileScheduler::get_work_k_tile_count(
      work_tile_info, problem_shape_MNKL, CtaShape_MNK{});

    // Line 741-749: Fetch next work tile
    auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
      work_tile_info,
      clc_pipeline,
      clc_pipe_consumer_state
    );
    if (increment_pipe) {
      ++clc_pipe_consumer_state;
    }

    // Line 752-759: Select accumulator stage
    int acc_stage = [&] () {
      if constexpr (IsOverlappingAccum) {
        return accumulator_pipe_producer_state.phase() ^ 1;
      } else {
        return accumulator_pipe_producer_state.index();
      }
    }();

    // Line 761-771: Call MMA collective (leader CTA only for 1SM mode)
    if (is_mma_leader_cta) {
      mainloop_pipe_consumer_state = collective_mainloop.mma(
        cute::make_tuple(mainloop_pipeline, accumulator_pipeline),
        cute::make_tuple(mainloop_pipe_consumer_state, accumulator_pipe_producer_state),
        collective_mainloop.slice_accumulator(tmem_storage, acc_stage),
        mma_inputs,
        cta_coord_mnkl,
        k_tile_count
      );
      accumulator_pipeline.producer_commit(accumulator_pipe_producer_state);
    }
    ++accumulator_pipe_producer_state;
    work_tile_info = next_work_tile_info;
    cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
  } while (work_tile_info.is_valid());

  // Line 780-803: Cleanup and TMEM deallocation
  cutlass::arch::launch_dependent_grids();
  tmem_allocator.release_allocation_lock();

  if constexpr (!IsOverlappingAccum) {
    if (is_mma_leader_cta) {
      accumulator_pipeline.producer_tail(accumulator_pipe_producer_state);
    }
    if constexpr (has_mma_peer_cta) {
      tmem_deallocation_result_barrier.arrive(mma_peer_cta_rank, not is_mma_leader_cta);
      tmem_deallocation_result_barrier.wait(dealloc_barrier_phase);
      tmem_deallocation_result_barrier.arrive(mma_peer_cta_rank, is_mma_leader_cta);
    }
  } else {
    tmem_deallocation_result_barrier.wait(dealloc_barrier_phase);
  }

  tmem_allocator.free(tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
}
```

---

#### Frame 13: CollectiveMma::mma() - Core Compute

**Location**: [include/cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp:944-1082](../../include/cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp#L944-L1082)

```cpp
CUTLASS_DEVICE auto
mma(cute::tuple<MainloopPipeline, AccumulatorPipeline> pipelines,
    cute::tuple<MainloopPipelineState, typename AccumulatorPipeline::PipelineState> pipeline_states,
    cute::tuple<cute::Tensor<FrgEngine, FrgLayout>> const& accumulators_pair,
    MmaParams const& mma_inputs,
    CtaTileCoord cta_tile_coord,
    int k_tile_count
) {
  static_assert(is_tmem<FrgEngine>::value, "Accumulator must be tmem resident.");
  static_assert(rank(FrgLayout{}) == 3, "Accumulator must be MMA-partitioned: (MMA, MMA_M, MMA_N)");

  // Line 957-965: Extract parameters
  auto accumulators = get<0>(accumulators_pair);
  auto [tiled_mma,
        tCrA, tCrB, tCtSFA, tCtSFB,
        tiled_copy_s2t_SFA, thr_tCsSFA_s2t,
        thr_tCtSFA_s2t, tiled_copy_s2t_SFB,
        thr_tCsSFB_s2t, thr_tCtSFB_s2t] = mma_inputs;

  auto [mainloop_pipeline, accumulator_pipeline] = pipelines;
  auto [mainloop_pipe_consumer_state, accumulator_pipe_producer_state] = pipeline_states;

  // Line 967-985: Adjust SFB pointer for N=64/192 cases
  auto tCtSFB_mma = [tCtSFB = tCtSFB, cta_tile_coord]() {
    if constexpr (IsCtaN192) {
      auto tCtSFB_tmp = tCtSFB;
      if (size<1>(cta_tile_coord) % 2 == 1) {
        tCtSFB_tmp.data() = tCtSFB_tmp.data().get() + 2;
      }
      return tCtSFB_tmp;
    }
    else if constexpr (IsCtaN64) {
      auto tCtSFB_tmp = tCtSFB;
      tCtSFB_tmp.data() = tCtSFB_tmp.data().get() + (size<1>(cta_tile_coord) % 2) * 2;
      return tCtSFB_tmp;
    }
    else {
      return tCtSFB;
    }
  }();

  uint32_t skip_wait = k_tile_count <= 0;
  auto barrier_token = mainloop_pipeline.consumer_try_wait(mainloop_pipe_consumer_state, skip_wait);

  // Line 993: Initialize accumulator to zero
  tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

  if constexpr (IsOverlappingAccum) {
    // Line 996-1035: First iteration (special case for overlapping accum)
    if (k_tile_count > 0) {
      mainloop_pipeline.consumer_wait(mainloop_pipe_consumer_state, barrier_token);

      int read_stage = mainloop_pipe_consumer_state.index();
      auto curr_mainloop_pipe_consumer_state = mainloop_pipe_consumer_state;

      ++mainloop_pipe_consumer_state;
      --k_tile_count;
      skip_wait = k_tile_count <= 0;
      barrier_token = mainloop_pipeline.consumer_try_wait(mainloop_pipe_consumer_state, skip_wait);

      // Line 1013-1016: Copy scale factors from SMEM to TMEM
      if (cute::elect_one_sync()) {
        copy(tiled_copy_s2t_SFA, thr_tCsSFA_s2t(_,_,_,_,read_stage), thr_tCtSFA_s2t);
        copy(tiled_copy_s2t_SFB, thr_tCsSFB_s2t(_,_,_,_,read_stage), thr_tCtSFB_s2t);
      }

      // Wait for accumulator TMEM to be available
      accumulator_pipeline.producer_acquire(accumulator_pipe_producer_state);

      // Line 1022-1032: Unroll K and execute MMA
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        // (V,M) x (V,N) => (V,M,N)
        cute::gemm(tiled_mma.with(tiled_mma.accumulate_,
                                  tCtSFA(_,_,k_block),
                                  tCtSFB_mma(_,_,k_block)),
            tCrA(_,_,k_block,read_stage),
            tCrB(_,_,k_block,read_stage),
            accumulators);
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
      }

      mainloop_pipeline.consumer_release(curr_mainloop_pipe_consumer_state);
    }
  }
  else {
    accumulator_pipeline.producer_acquire(accumulator_pipe_producer_state);
  }

  // Line 1042-1079: Main pipelined loop
  CUTLASS_PRAGMA_NO_UNROLL
  while (k_tile_count > 0) {
    // Line 1046: Wait for data to be available
    mainloop_pipeline.consumer_wait(mainloop_pipe_consumer_state, barrier_token);

    int read_stage = mainloop_pipe_consumer_state.index();
    auto curr_mainloop_pipe_consumer_state = mainloop_pipe_consumer_state;

    // Advance pipeline
    ++mainloop_pipe_consumer_state;
    --k_tile_count;
    skip_wait = k_tile_count <= 0;
    barrier_token = mainloop_pipeline.consumer_try_wait(mainloop_pipe_consumer_state, skip_wait);

    // Line 1060-1063: Copy scale factors SMEM → TMEM
    if (cute::elect_one_sync()) {
      copy(tiled_copy_s2t_SFA, thr_tCsSFA_s2t(_,_,_,_,read_stage), thr_tCtSFA_s2t);
      copy(tiled_copy_s2t_SFB, thr_tCsSFB_s2t(_,_,_,_,read_stage), thr_tCtSFB_s2t);
    }

    // Line 1066-1076: Unroll K mode and execute MMA
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
      // (V,M) x (V,N) => (V,M,N)
      cute::gemm(tiled_mma.with(tiled_mma.accumulate_,
                                tCtSFA(_,_,k_block),
                                tCtSFB_mma(_,_,k_block)),
          tCrA(_,_,k_block,read_stage),
          tCrB(_,_,k_block,read_stage),
          accumulators);
      tiled_mma.accumulate_ = UMMA::ScaleOut::One;
    }

    mainloop_pipeline.consumer_release(curr_mainloop_pipe_consumer_state);
  }

  return mainloop_pipe_consumer_state;
}
```

**Key Points**:
1. **Pipelined execution**: While processing K-tile N, simultaneously wait for K-tile N+1
2. **Scale factor copy**: SMEM → TMEM via UTCCP (Unified Tensor Core Copy Pipe)
3. **Unrolled K loop**: Each K-tile (256-K) broken into smaller k_blocks
4. **MMA instruction**: Line 1069 - `cute::gemm()` call

---

#### Frame 14: cute::gemm() - MMA Dispatch

**Location**: [include/cute/algorithm/gemm.hpp](../../include/cute/algorithm/gemm.hpp)

The `cute::gemm()` call with `.with()` modifier dispatches to the MMA instruction:

```cpp
template <class... Args, class TA, class TB, class TC>
CUTE_HOST_DEVICE constexpr
void
gemm(MMA_Traits<Args...> const& mma,
     TA                  const& A,
     TB                  const& B,
     TC                       & C)
{
  return detail::CallUMMA<MMA_Traits<Args...>>::call(
      mma.accumulate_,  // ScaleOut mode
      mma.idesc_,       // Instruction descriptor with scale factors
      A, B, C);
}
```

The `.with()` modifier adds scale factors to the MMA descriptor:

```cpp
// From TiledMma
auto with(ScaleOut accumulate, TensorSFA const& sfa, TensorSFB const& sfb) {
  auto mma_copy = *this;
  // Update instruction descriptor with scale factor pointers
  mma_copy.idesc_.sfa_tmem_ptr_ = raw_pointer_cast(sfa.data());
  mma_copy.idesc_.sfb_tmem_ptr_ = raw_pointer_cast(sfb.data());
  mma_copy.accumulate_ = accumulate;
  return mma_copy;
}
```

---

#### Frame 15: CallUMMA - PTX Generation

**Location**: [include/cute/arch/mma_sm100_desc.hpp](../../include/cute/arch/mma_sm100_desc.hpp)

```cpp
template <>
struct CallUMMA<SM100_U16x128x256x8_TN> {
  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE static void
  call(UMMA::ScaleOut accumulate,
       SM100_U16x128x256x8_TN const& mma_desc,
       Tensor<TD, DLayout> const& d_desc,
       Tensor<TA, ALayout> const& a_desc,
       Tensor<TB, BLayout> const& b_desc,
       Tensor<TC, CLayout>      & c_tmem) {

    // Extract pointers and descriptors
    uint64_t d_desc_bits = reinterpret_cast<uint64_t>(raw_pointer_cast(d_desc.data()));
    uint64_t a_desc_bits = reinterpret_cast<uint64_t>(raw_pointer_cast(a_desc.data()));
    uint64_t b_desc_bits = reinterpret_cast<uint64_t>(raw_pointer_cast(b_desc.data()));
    uint32_t c_tmem_ptr = raw_pointer_cast(c_tmem.data());

    // Extract scale factor TMEM pointers
    uint32_t sfa_tmem_ptr = mma_desc.sfa_tmem_ptr_;
    uint32_t sfb_tmem_ptr = mma_desc.sfb_tmem_ptr_;

    // Extract runtime data types if needed
    uint32_t a_format = mma_desc.a_format_;
    uint32_t b_format = mma_desc.b_format_;

    // Call inline PTX assembly
    asm volatile(
      "tcgen05.mma.cta_group::1.kind::umma.tile::m128n128k256"
      ".f32.data_format_a::%s.data_format_b::%s"
      ".satfinite::0.scale_d::%s.scale_a::1.scale_b::1"
      " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15},"  // C output (16x f32 regs)
      " {%16, %17, %18, %19},"     // D descriptor (block-scaled output)
      " {%20, %21},"               // A descriptor (SMEM)
      " {%22, %23},"               // B descriptor (SMEM)
      " {%24},"                    // Scale factor A (TMEM)
      " {%25};"                    // Scale factor B (TMEM)
      : "+r"(c_tmem[0]),  "+r"(c_tmem[1]),  "+r"(c_tmem[2]),  "+r"(c_tmem[3]),
        "+r"(c_tmem[4]),  "+r"(c_tmem[5]),  "+r"(c_tmem[6]),  "+r"(c_tmem[7]),
        "+r"(c_tmem[8]),  "+r"(c_tmem[9]),  "+r"(c_tmem[10]), "+r"(c_tmem[11]),
        "+r"(c_tmem[12]), "+r"(c_tmem[13]), "+r"(c_tmem[14]), "+r"(c_tmem[15])
      : "l"(d_desc_bits), "l"(d_desc_bits >> 32), "l"(d_desc_bits), "l"(d_desc_bits >> 32),
        "l"(a_desc_bits), "l"(a_desc_bits >> 32),
        "l"(b_desc_bits), "l"(b_desc_bits >> 32),
        "r"(sfa_tmem_ptr),
        "r"(sfb_tmem_ptr)
    );
  }
};
```

**PTX Instruction Breakdown**:

```
tcgen05.mma.cta_group::1.kind::umma.tile::m128n128k256
  .f32                          // Accumulator type
  .data_format_a::e2m1          // A is FP4 (E2M1)
  .data_format_b::e2m1          // B is FP4 (E2M1)
  .satfinite::0                 // No saturation
  .scale_d::one                 // Output scale mode
  .scale_a::1                   // Use scale factors for A
  .scale_b::1                   // Use scale factors for B
  {c0,...,c15},                 // Output: 16×FP32 accumulators in TMEM
  {d_desc},                     // D descriptor (for block-scaled output)
  {a_desc},                     // A descriptor (SMEM pointer)
  {b_desc},                     // B descriptor (SMEM pointer)
  {sfa_tmem},                   // Scale factor A (TMEM pointer)
  {sfb_tmem};                   // Scale factor B (TMEM pointer)
```

**Hardware Execution**:

1. **Fetch A**: Read 128×256 FP4 elements from SMEM using descriptor
2. **Fetch B**: Read 128×256 FP4 elements from SMEM using descriptor
3. **Fetch SFA**: Read 16 FP8 scale factors from TMEM (1 per 128×16 block)
4. **Fetch SFB**: Read 16 FP8 scale factors from TMEM (1 per 128×16 block)
5. **Compute**: For each output element C[m,n]:
   ```
   for k in 0..255:
     sf_a = SFA[m / 128][k / 16]
     sf_b = SFB[n / 128][k / 16]
     c[m,n] += (A[m,k] * sf_a) * (B[n,k] * sf_b)
   ```
6. **Write C**: Store 128×128 FP32 results to TMEM

**Cycle Count**: ~256 cycles for the full 128×128×256 MMA

---

### EPILOGUE WARP (Warp 4+): Fusion and Output

**Location**: [include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp:868-954](../../include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp#L868-L954)

#### Frame 16: Epilogue Main Loop

```cpp
else if (is_participant.epilogue) {
  // Line 870-872: Wait for TMEM allocation, get base pointer
  tmem_allocation_result_barrier.arrive_and_wait();
  uint32_t tmem_base_ptr = shared_storage.tmem_base_ptr;
  collective_mainloop.set_tmem_offsets(tmem_storage, tmem_base_ptr);

  bool do_tail_store = false;

  // Line 875-934: Work loop
  do {
    // Line 877-885: Fetch next work tile
    auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
      work_tile_info,
      clc_pipeline,
      clc_pipe_consumer_state
    );

    if (increment_pipe) {
      ++clc_pipe_consumer_state;
    }

    // Line 888-895: Select accumulator stage
    int acc_stage = [&] () {
      if constexpr (IsOverlappingAccum) {
        return accumulator_pipe_consumer_state.phase();
      } else {
        return accumulator_pipe_consumer_state.index();
      }
    }();

    // Line 897-905: Slice accumulator from TMEM
    auto accumulator = get<0>(collective_mainloop.slice_accumulator(tmem_storage, acc_stage));
    accumulator_pipe_consumer_state = scheduler.template fixup<IsComplex>(
      TiledMma{},
      work_tile_info,
      accumulator,
      accumulator_pipeline,
      accumulator_pipe_consumer_state,
      typename CollectiveEpilogue::CopyOpT2R{}
    );

    // Line 910-929: Epilogue and write to gD
    if (scheduler.compute_epilogue(work_tile_info)) {
      auto [load_state_next, store_state_next, acc_state_next] = collective_epilogue.template store<IsOverlappingAccum>(
        epi_load_pipeline,
        epi_load_pipe_consumer_state,
        epi_store_pipeline,
        epi_store_pipe_producer_state,
        accumulator_pipeline,
        accumulator_pipe_consumer_state,
        problem_shape_MNKL,
        CtaShape_MNK{},
        cta_coord_mnkl,
        TileShape{},
        TiledMma{},
        accumulator,
        shared_storage.tensors.epilogue
      );
      epi_load_pipe_consumer_state = load_state_next;
      epi_store_pipe_producer_state = store_state_next;
      accumulator_pipe_consumer_state = acc_state_next;
      do_tail_store = true;
    }
    work_tile_info = next_work_tile_info;
    cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);

  } while (work_tile_info.is_valid());

  // Line 936-953: Cleanup
  if constexpr (IsOverlappingAccum) {
    if constexpr (has_mma_peer_cta) {
      tmem_deallocation_result_barrier.arrive(mma_peer_cta_rank);
    }
    tmem_deallocation_result_barrier.arrive();
  }

  if (do_tail_store) {
    collective_epilogue.store_tail(
      epi_load_pipeline, epi_load_pipe_consumer_state,
      epi_store_pipeline, epi_store_pipe_producer_state,
      CtaShape_MNK{});
  }
}
```

---

#### Frame 17: CollectiveEpilogue::store() - Fusion and Output

**Location**: [include/cutlass/epilogue/collective/sm100_epilogue_tma_warpspecialized.hpp](../../include/cutlass/epilogue/collective/sm100_epilogue_tma_warpspecialized.hpp)

```cpp
template <bool IsOverlappingAccum = false>
CUTLASS_DEVICE auto
store(
    LoadPipeline load_pipeline,
    LoadPipelineState load_pipe_consumer_state,
    StorePipeline store_pipeline,
    StorePipelineState store_pipe_producer_state,
    AccumulatorPipeline accumulator_pipeline,
    AccumulatorPipelineState accumulator_pipe_consumer_state,
    ProblemShape problem_shape_mnkl,
    TileShape cta_shape_mnk,
    TileCoord cta_coord_mnkl,
    TileShape tile_shape_mnk,
    TiledMma tiled_mma,
    AccumulatorTensor const& accumulator,
    TensorStorage& shared_tensors
) {

  // Extract problem dimensions
  auto [M, N, K, L] = problem_shape_mnkl;
  auto [m_coord, n_coord, k_coord, l_coord] = cta_coord_mnkl;

  // Wait for accumulator to be ready
  accumulator_pipeline.consumer_wait(accumulator_pipe_consumer_state);

  // Copy accumulator from TMEM to registers
  Tensor rAcc = partition_accumulator(accumulator, tiled_mma);

  // Apply fusion operation (alpha * acc + beta * C)
  if (params_.is_source_needed) {
    // Load C from global memory
    Tensor gC = make_tensor(params_.ptr_C, make_layout(make_shape(M, N, L), params_.stride_C));
    Tensor tCgC = local_tile(gC, tile_shape_mnk, cta_coord_mnkl);

    // Load C into registers
    Tensor rC = make_fragment_like(tCgC);
    copy(tCgC, rC);

    // Compute: temp = alpha * rAcc + beta * rC
    for (int i = 0; i < size(rAcc); ++i) {
      rAcc(i) = params_.alpha * rAcc(i) + params_.beta * rC(i);
    }
  } else {
    // Just scale: temp = alpha * rAcc
    for (int i = 0; i < size(rAcc); ++i) {
      rAcc(i) = params_.alpha * rAcc(i);
    }
  }

  // Apply activation function if specified
  if constexpr (has_activation) {
    for (int i = 0; i < size(rAcc); ++i) {
      rAcc(i) = activation_fn(rAcc(i));
    }
  }

  // Quantize to output type and generate scale factors
  if constexpr (has_block_scale_output) {
    // Quantize to FP4 with block scale factors
    Tensor rD_quantized = make_fragment_like<ElementD>(rAcc);
    Tensor rSFD = make_fragment<ElementSFD>(/* scale factor count */);

    quantize_blockwise(rAcc, rD_quantized, rSFD, params_);

    // Store D (quantized)
    Tensor gD = make_tensor(params_.ptr_D, make_layout(make_shape(M, N, L), params_.stride_D));
    Tensor tCgD = local_tile(gD, tile_shape_mnk, cta_coord_mnkl);
    copy(rD_quantized, tCgD);

    // Store SFD (scale factors)
    Tensor gSFD = make_tensor(params_.ptr_SFD, params_.layout_SFD);
    Tensor tCgSFD = local_tile(gSFD, /* SF tile shape */, cta_coord_mnkl);
    copy(rSFD, tCgSFD);
  } else {
    // Direct store (no quantization)
    Tensor rD = convert<ElementD>(rAcc);

    Tensor gD = make_tensor(params_.ptr_D, make_layout(make_shape(M, N, L), params_.stride_D));
    Tensor tCgD = local_tile(gD, tile_shape_mnk, cta_coord_mnkl);
    copy(rD, tCgD);
  }

  // Release accumulator pipeline
  accumulator_pipeline.consumer_release(accumulator_pipe_consumer_state);
  ++accumulator_pipe_consumer_state;

  return cute::make_tuple(load_pipe_consumer_state, store_pipe_producer_state, accumulator_pipe_consumer_state);
}
```

---

#### Frame 18: Block-wise Quantization Algorithm

**Pseudo-code for quantization**:

```cpp
template <class TensorFloat, class TensorFP4, class TensorSF>
CUTLASS_DEVICE void
quantize_blockwise(
    TensorFloat const& input,    // FP32 values
    TensorFP4& output,           // FP4 quantized values
    TensorSF& scale_factors,     // FP8 scale factors
    Params const& params) {

  constexpr int BLOCK_M = 128;
  constexpr int BLOCK_N = 16;

  // Process each block
  for (int block_m = 0; block_m < M; block_m += BLOCK_M) {
    for (int block_n = 0; block_n < N; block_n += BLOCK_N) {

      // Step 1: Find max absolute value in block
      float max_abs = 0.0f;
      for (int i = 0; i < BLOCK_M; ++i) {
        for (int j = 0; j < BLOCK_N; ++j) {
          int idx_m = block_m + i;
          int idx_n = block_n + j;
          max_abs = max(max_abs, abs(input(idx_m, idx_n)));
        }
      }

      // Step 2: Compute scale factor
      // FP4 (E2M1) max representable value = 6.0
      float scale_factor = (max_abs + 1e-6f) / 6.0f;

      // Step 3: Store scale factor as FP8
      int sf_idx_m = block_m / BLOCK_M;
      int sf_idx_n = block_n / BLOCK_N;
      scale_factors(sf_idx_m, sf_idx_n) = ElementSFD(scale_factor);

      // Step 4: Quantize elements
      for (int i = 0; i < BLOCK_M; ++i) {
        for (int j = 0; j < BLOCK_N; ++j) {
          int idx_m = block_m + i;
          int idx_n = block_n + j;

          // Normalize to FP4 range
          float normalized = input(idx_m, idx_n) / scale_factor;

          // Convert to FP4
          output(idx_m, idx_n) = ElementD(normalized);
        }
      }
    }
  }
}
```

**Optimization**: This is actually done using warp-level reductions and vectorized stores.

---

## Part 4: Synchronization and Control Flow

### Pipeline Synchronization

```
Producer Warp (Load):
  ┌─────────────────────────────────────┐
  │ Acquire Barrier[stage]              │
  │ Issue TMA (A, B, SFA, SFB)          │
  │ TMA completes → Signal Barrier      │
  └──────────────┬──────────────────────┘
                 │
                 ▼ (barrier signals)
  ┌──────────────────────────────────────┐
  │ Consumer Warp (MMA):                 │
  │   Wait on Barrier[stage]             │
  │   Copy SF from SMEM → TMEM           │
  │   Execute MMA with SF                │
  │   Release Barrier[stage]             │
  └──────────────┬───────────────────────┘
                 │
                 ▼ (accumulator ready)
  ┌──────────────────────────────────────┐
  │ Epilogue Warp:                       │
  │   Wait on Accumulator Pipeline       │
  │   Read accumulator from TMEM         │
  │   Apply fusion                       │
  │   Quantize & generate SFD            │
  │   Store D & SFD to GMEM              │
  │   Release Accumulator Pipeline       │
  └──────────────────────────────────────┘
```

---

## Part 5: Memory Traffic Analysis

### For One 128×128×256 Tile

**Mainloop (Producer Warp)**:
```
GMEM → SMEM (via TMA):
  - A:   128×256×4 bits = 16,384 bytes
  - SFA: (128/128)×(256/16)×8 bits = 16 bytes
  - B:   128×256×4 bits = 16,384 bytes
  - SFB: (128/128)×(256/16)×8 bits = 16 bytes
  Total per stage: ~32 KB
```

**Mainloop (Consumer Warp)**:
```
SMEM → Registers (A, B):
  - Handled by UMMA descriptor loads

SMEM → TMEM (SFA, SFB):
  - Via UTCCP: 32 bytes per stage

TMEM (Accumulator):
  - 128×128×32 bits = 65,536 bytes
```

**Epilogue**:
```
TMEM → Registers (Accumulator):
  - 128×128×32 bits = 64 KB

GMEM → Registers (C, optional):
  - 128×128×32 bits = 64 KB

Registers → GMEM (D):
  - 128×128×4 bits = 8 KB

Registers → GMEM (SFD):
  - (128/128)×(128/16)×8 bits = 64 bytes
```

**Total for 2048×2048×2048 GEMM**:
```
Number of tiles: (2048/128) × (2048/128) × (2048/256) = 16 × 16 × 8 = 2048 tiles

GMEM reads:  2048 × 32 KB = 64 MB (A, B, SF)
GMEM writes: 2048 × 8 KB = 16 MB (D, SFD)
Total: ~80 MB

Compute: 2 × 2048³ = 17.2 GFLOP
Arithmetic Intensity: 17.2 GFLOP / 80 MB ≈ 215 FLOP/byte
```

**Conclusion**: Highly compute-bound, excellent for GPU efficiency.

---

## Summary: Complete Call Stack

```
HOST:
  main()
    → run<Gemm>(options)
      → gemm.run()
        → GemmUniversalAdapter::run()
          → DeviceKernel::run()
            → ClusterLauncher::launch()
              → cudaLaunchKernelEx()

DEVICE:
  GemmUniversal::operator()(params, smem)
    │
    ├─ Warp 0 (MMA):
    │    → collective_mainloop.mma_init()
    │    → loop: collective_mainloop.mma()
    │         → copy SMEM→TMEM (scale factors)
    │         → cute::gemm()
    │              → CallUMMA::call()
    │                   → tcgen05.mma PTX instruction
    │                        ├─ Read A from SMEM descriptor
    │                        ├─ Read B from SMEM descriptor
    │                        ├─ Read SFA from TMEM
    │                        ├─ Read SFB from TMEM
    │                        ├─ Compute: C += (A*SFA) × (B*SFB)
    │                        └─ Write C to TMEM
    │
    ├─ Warp 1 (Sched):
    │    → scheduler.advance_to_next_work()
    │    → clc_pipeline operations
    │
    ├─ Warp 2 (MainloopLoad):
    │    → collective_mainloop.load_init()
    │    → loop: collective_mainloop.load()
    │         → TMA copy A: GMEM → SMEM
    │         → TMA copy B: GMEM → SMEM
    │         → TMA copy SFA: GMEM → SMEM
    │         → TMA copy SFB: GMEM → SMEM
    │         → Signal mainloop_pipeline barrier
    │
    ├─ Warp 3 (EpilogueLoad):
    │    → collective_epilogue.load()
    │         → TMA copy C: GMEM → SMEM (if needed)
    │
    └─ Warps 4+ (Epilogue):
         → collective_epilogue.store()
              → Wait on accumulator_pipeline
              → Read accumulator from TMEM
              → Load C from GMEM (if beta != 0)
              → Fusion: D = alpha*acc + beta*C
              → Activation (if specified)
              → Quantize block-wise:
              │    → Find max_abs per block
              │    → Compute scale_factor
              │    → Quantize elements to FP4
              │    → Store scale_factors as FP8
              → Store D to GMEM (vectorized)
              → Store SFD to GMEM
```

---

## Appendix: Template Instantiations

### Concrete Types for Our Example

```cpp
// From 72b_blackwell_nvfp4_nvfp4_gemm.cu

ElementA = nv_float4_t<float_e2m1_t>
  ├─ DataType = float_e2m1_t (4-bit)
  └─ ScaleFactorType = float_ue4m3_t (8-bit)

ElementB = nv_float4_t<float_e2m1_t>
  ├─ DataType = float_e2m1_t (4-bit)
  └─ ScaleFactorType = float_ue4m3_t (8-bit)

ElementD = float_e2m1_t (4-bit)
ElementSFD = float_ue8m0_t (8-bit, exponent only)
ElementC = float (32-bit)
ElementAccumulator = float (32-bit)
ElementCompute = float (32-bit)

TileShape = Shape<_128, _128, _256>
ClusterShape = Shape<_1, _1, _1>

CollectiveMainloop = CollectiveMma<
  MainloopSm100TmaUmmaWarpSpecializedBlockScaled<...>,
  TileShape,
  ElementPairA = cute::tuple<float_e2m1_t, float_ue4m3_t>,
  StridePairA = cute::tuple<StrideA, LayoutSFA>,
  ElementPairB = cute::tuple<float_e2m1_t, float_ue4m3_t>,
  StridePairB = cute::tuple<StrideB, LayoutSFB>,
  TiledMma = TiledMMA<SM100_16x128x256x8_TN<float_e2m1_t, float_e2m1_t, float, float_ue4m3_t>>,
  ...
>

CollectiveEpilogue = CollectiveEpilogue<
  ...,
  FusionOperation = LinCombBlockScaleFactor<16, float_e2m1_t, float, float_ue8m0_t, ...>
>
```

---

**Document Version**: 1.0
**Last Updated**: 2025-11-02
**Status**: Complete execution trace from host→device→PTX
