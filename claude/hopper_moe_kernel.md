# Deep Trace: `HopperWgmma_MoE_kernel` — SonicMoE Grouped GEMM on Hopper

## Process Notes

**Approach**: Bottom-up trace starting from the test entry point (`moe_minimal.py:91`), through the MoE module dispatch, routing metadata computation, and into the CUTLASS kernel body. I read every file in the call chain, focusing on the four requested optimization patterns.

**Agent delegation**: Used an Explore agent to map the full file tree under `thirdparty/sonic-moe/`, identifying all kernel-related files. A second Explore agent traced the MoE module's `forward()` dispatch and the Triton routing metadata kernels. All kernel code was read directly in the main context.

**Tools used**: Read (13 calls across grouped_gemm.py sections, forward.py, moe_config.py, tile_scheduler.py, moe_minimal.py), Explore agent (2 invocations), Grep/Glob for cross-referencing.

**Key source files** (all under `thirdparty/sonic-moe/sonicmoe/`):

| File | Role |
|------|------|
| `moe.py` | `MoE` module — top-level forward dispatch |
| `functional/__init__.py` | `moe_TC_softmax_topk_layer()` — routing + projection orchestration |
| `functional/triton_kernels/__init__.py` | Routing metadata (histogram, prefix-sum, reorder) |
| `functional/forward.py` | `_up_projection_forward` / `_down_projection_forward` custom ops |
| `functional/moe_config.py` | `HopperWgmma_MoE_Up_proj_Fwd` / `Down_proj_Fwd` config wrappers |
| `functional/grouped_gemm.py` | **`HopperWgmma_MoE_kernel`** — the kernel itself (3070 lines) |
| `functional/tile_scheduler.py` | `SonicMoETileScheduler` / `SonicMoEVarlenMTileScheduler` |

---

## Table of Contents

1. [Entry Point: Test → MoE Module](#1-entry-point)
2. [Routing & Token Metadata](#2-routing)
3. [Kernel Configuration & Tile Tuning Heuristics](#3-config)
4. [Kernel Launch: `__call__`](#4-launch)
5. [GPU Kernel Body Overview](#5-kernel-overview)
6. [Producer Phase: TMA + gatherA](#6-producer)
7. [Consumer Phase: WGMMA Mainloop](#7-consumer)
8. [Epilogue: Tiled Writeback + Activation Fusion](#8-epilogue)
9. [Ping-Pong Scheduling](#9-pingpong)
10. [Shared Memory Budget & Stage Computation](#10-smem)

---

<a id="1-entry-point"></a>
## 1. Entry Point: Test → MoE Module

### `moe_minimal.py:91`
```python
y_kernel = moe_kernel(x_kernel, kernel_backend_moe=kernel_backend_moe)[0]
#          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#          moe_kernel is an MoE(...) instance
#          kernel_backend_moe = KernelBackendMoE.sonicmoe
```

The `MoE` class (`moe.py:208-260`) dispatches based on `kernel_backend_moe`:

```python
# moe.py:forward()
if kernel_backend_moe == KernelBackendMoE.sonicmoe and self.num_experts <= 32768:
    # Fast path: CUTLASS grouped GEMM
    hidden_states, router_logits, expert_frequency = moe_TC_softmax_topk_layer(
        hidden_states,                            # (T, H) input tokens
        self.router.weight,                       # (E, H) router weights
        self.c_fc.weight.permute(1, 2, 0),        # (I, H, E) -> w1 for up-proj
        self.c_fc.bias,                           # optional bias
        self.c_proj.weight.permute(1, 2, 0),      # (H, I, E) -> w2 for down-proj
        self.c_proj.bias,
        self.top_k,                               # K = experts per token
        self.stream_id,
        self.activation_function,                  # e.g., SWIGLU
        is_inference_mode or not self.training,
    )
```

**Key dimensions** for the default test case `(T=8192, H=768, I=256, E=128, K=8)`:
- **T**: 8192 tokens
- **H**: 768 hidden size (A-matrix K dim for up-proj)
- **I**: 256 intermediate size (output N dim for up-proj, input K dim for down-proj)
- **E**: 128 experts
- **K**: 8 experts per token → `T*K = 65536` total expert assignments

---

<a id="2-routing"></a>
## 2. Routing & Token Metadata

`moe_TC_softmax_topk_layer()` (`functional/__init__.py:428-501`) runs a 3-stage pipeline before any GEMM:

### Stage 1: Router + TopK + Softmax
```python
router_logits = F.linear(x, router_w)                        # (T, E)
topk_scores, topk_indices = TC_Softmax_Topk_Router(logits)   # fused CUTLASS kernel
# topk_scores:  (T, K) float32 — softmax probabilities per selected expert
# topk_indices: (T, K) int32   — which expert each token selected
```

### Stage 2: Routing Metadata (Triton)
```python
TC_topk_router_metadata_triton(
    topk_indices, E,
    expert_frequency,          # (E,)   — count of tokens per expert
    expert_frequency_offset,   # (E+1,) — exclusive prefix sum: where each expert's block starts
    x_gather_idx,              # (T*K,) — token index for each slot in expert-sorted order
    s_scatter_idx,             # (T*K,) — inverse permutation
    s_reverse_scatter_idx,     # (T*K,) — forward permutation
)
```

This runs 3 Triton kernels:

| Kernel | Purpose |
|--------|---------|
| `_compute_col_partial_sum` | Per-tile histogram of expert assignments |
| `_bitmatrix_metadata_stage1` | Prefix sum across tiles; compute `expert_frequency_offset` |
| `_bitmatrix_metadata_stage2` | Reorder entries: populate `x_gather_idx`, `s_scatter_idx` |

**The critical output is `expert_frequency_offset`**: a (E+1,) array where `offset[e]` gives the starting row index in the grouped output for expert `e`. This lets the kernel treat the problem as a variable-length batched GEMM — each expert's "batch" has a different number of tokens `M_e = offset[e+1] - offset[e]`.

**`x_gather_idx`**: For each position in the expert-sorted output, tells which original token to load. This is the basis for the **gatherA** pattern.

### Stage 3: Up/Down Projections
```python
# Up projection: y = activation(x @ W1)
#   A = x (gathered by expert), B = W1[expert], D = z (pre-activation output)
y1, z = _UpProjection.apply(x, w1, ..., x_gather_idx, ...)

# Down projection: o = y1 @ W2
#   A = y1 (already in expert-sorted order), B = W2[expert], D = y2
o = _DownProjection.apply(y1, z, w2, ..., x_gather_idx, ...)

# Final: weighted sum back to token order
o = token_gather_and_sum_varlen_K(y2, topk_scores, ...)
```

---

<a id="3-config"></a>
## 3. Kernel Configuration & Tile Tuning Heuristics

`moe_config.py` chooses tile shapes based on the problem dimensions. Here are the key configs:

### Up Projection Forward (`HopperWgmma_MoE_Up_proj_Fwd`)

```python
# moe_config.py:39-120
# For I >= 128 (GLU) or I >= 256 (non-GLU):
up_config = HopperGEMMConfig(
    tile_shape_mnk=(128, 256, 64),    # CTA tile: 128 output rows × 256 cols × 64 K-reduction
    cluster_shape_mnk=(2, 1),          # 2-CTA cluster along M, 1 along N
    epi_tile_size=32,                  # epilogue tile N-dimension
    is_pingpong=False,                 # single warp group
    initial_d_epi_stage=2,             # epilogue pipeline depth
    raster_order=AlongM,               # tile raster order for L2 locality
)

# For smaller I (64 GLU or 128 non-GLU):
up_config = HopperGEMMConfig(
    tile_shape_mnk=(192, 128, 64),    # Taller tile for less N work
    cluster_shape_mnk=(1, 1),          # No clustering
    is_pingpong=True,                  # Two warp groups alternate MMA/epilogue
    initial_d_epi_stage=8,             # Deeper epilogue pipeline
)
```

### Down Projection Forward (`HopperWgmma_MoE_Down_proj_Fwd`)

```python
# moe_config.py:155-202
# For I >= 1024:
down_config = HopperGEMMConfig(
    tile_shape_mnk=(128, 256, 64),
    cluster_shape_mnk=(2, 1),
    raster_order=AlongN,    # ← note: AlongN for down-proj (different from up-proj)
)

# For 256 <= I < 1024:
down_config = HopperGEMMConfig(
    tile_shape_mnk=(128, 192, 64),
    cluster_shape_mnk=(2, 1),
    is_pingpong=True,
    epi_tile_size=(96 if H % 96 == 0 else 64),   # ← dynamic epi tile based on H alignment
)
```

### TensorCore Shape Tuning Heuristics

**`grouped_gemm.py:166-182` — Atom layout selection**:

The atom layout `(atom_M, atom_N, 1)` controls how warp groups partition the CTA tile:

```python
# grouped_gemm.py:168-182
if not self.pingpong:
    if tile_M == 320:
        # tile_M/64 is not even → must split along N instead
        atom_layout_m, atom_layout_n = 1, 2
    elif tile_M == 192:
        if tile_N <= 128:
            # Enough room for 3 WGs along M (3 × 64 = 192)
            atom_layout_m, atom_layout_n = 3, 1
        else:
            # Too much N work for 3 M-atoms; split N
            atom_layout_m, atom_layout_n = 1, 2
    else:
        # Standard: divide M by 64 up to 2 atoms
        atom_layout_m = tile_M // 64 if tile_M < 256 else 2
        atom_layout_n = 1
else:
    # Ping-pong: always (1,1,1) — each WG processes the full tile
    atom_layout_m, atom_layout_n = 1, 1
```

**Why this matters**: The atom layout determines how many WGMMA (warp group MMA) threads handle each sub-tile. With `atom_layout_m=2, atom_layout_n=1`, two warp groups each handle a 64-row × 256-col sub-tile, doubling throughput along M while keeping N contiguous for TMA stores.

**Register pressure management** (`grouped_gemm.py:222-235`):

```python
# Registers per thread = (tile_M × tile_N) / num_mma_threads
# For (128, 256, 64) with 1 WG: regs = 128*256/128 = 256 → heavy pressure
regs_per_thread = math.prod(self.tile_shape_mnk[:2]) // self.num_mma_threads
heavy_register_pressure = regs_per_thread >= 208

if not is_A_gather:
    if self.mma_warp_groups == 3:
        self.num_regs_load, self.num_regs_mma = 32, 160
    else:
        self.num_regs_load, self.num_regs_mma = (40, 232) if not heavy else (24, 240)
else:
    # gatherA needs more load registers for address computation
    self.num_regs_load, self.num_regs_mma = (56, 224)
```

The kernel uses `setmaxregister_decrease`/`increase` PTX instructions to dynamically partition the register file between producer (TMA/load) warps and consumer (MMA) warps. This is critical: MMA warps holding a 128×256 accumulator need ~240 registers, but the TMA warp doing simple loads only needs ~32.

---

<a id="4-launch"></a>
## 4. Kernel Launch: `__call__`

`grouped_gemm.py:751-1089` — the `__call__` method sets up everything before launching:

### 4.1 Type Setup and Attribute Initialization
```python
# grouped_gemm.py:776-829
self.a_dtype = mA.element_type       # BFloat16
self.b_dtype = mB.element_type       # BFloat16
self.d_dtype = mD.element_type       # BFloat16
self.a_layout = LayoutEnum.from_tensor(mA)   # ROW_MAJOR for tokens
self._setup_attributes()             # compute stages, smem layouts, epilogue tiles
```

### 4.2 Tiled MMA Construction
```python
# grouped_gemm.py:831-854
tiled_mma = sm90_utils.make_trivial_tiled_mma(
    self.a_dtype, self.b_dtype,
    self.a_layout.sm90_mma_major_mode(),    # K-major or M-major
    self.b_layout.sm90_mma_major_mode(),
    self.acc_dtype,                          # Float32
    self.atom_layout_mnk,                    # (2, 1, 1) for 128×256 tile
    tiler_mn=(64, self.tile_N // atom_layout_n),  # sub-tile per atom
)
```

This creates the WGMMA descriptor. On Hopper, WGMMA operates on 64×N sub-tiles using the Tensor Memory Accelerator, with the accumulator held in registers.

### 4.3 TMA vs. gatherA Copy Setup

```python
# grouped_gemm.py:856-874
if self.is_A_gather:
    # Up projection: A is scattered tokens — cannot use TMA
    A_tiled_copy = self._make_tiled_copy_2D(mA, ...)   # cp.async based
    tma_atom_a = tma_tensor_a = None
else:
    # Down projection: A is contiguous — use TMA
    tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(mA, ...)

# B (expert weights) always uses TMA — contiguous per expert
tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(mB, ...)
```

**Key insight**: The up-projection's A matrix (`x`) is **non-contiguous** — tokens are scattered according to `x_gather_idx`. TMA requires contiguous tiles, so the kernel falls back to `cp.async` with manual address computation for A. The B matrix (expert weights) is always contiguous within each expert slice and uses TMA with optional multicast.

### 4.4 Tile Scheduler Setup

```python
# grouped_gemm.py:949-968
# Forward pass: variable-length M per expert → VarlenMTileScheduler
problem_shape_ntile_mnl = (
    None,                                        # M tiles computed dynamically per expert
    ceil_div(mD.shape[1], tile_N),               # N tiles
    mMoffset.shape[0] - 1,                       # L = number of experts (batches)
)
TileScheduler = SonicMoEVarlenMTileScheduler
tile_sched_args = VarlenMTileSchedulerArguments(
    total_m=mD.shape[0],              # T*K total output rows
    cu_seqlens_m=mMoffset,            # expert_frequency_offset — defines M ranges
    ...
)
```

The `VarlenMTileScheduler` is essential for MoE: each expert has a different number of tokens. The scheduler uses `expert_frequency_offset` (passed as `cu_seqlens_m`) to compute per-expert M tile counts and dynamically assigns tiles to CTAs.

### 4.5 Shared Memory Layout

```python
# grouped_gemm.py:987-1034
class SharedStorage:
    mainloop_pipeline_array_ptr: ...   # Barrier storage for TMA/cp.async pipeline
    tensormap_buffer: ...              # TensorMap descriptors (128B aligned)
    sD: ...                            # Epilogue output buffer (d_epi_stage stages)
    sched_pipeline_array_ptr: ...      # Scheduler pipeline barriers
    tile_count: ...                    # Tile count semaphore
    sY: ...                            # Activation output buffer (if GLU/SiLU)
    sA: ...                            # A-matrix double/multi-buffer (ab_stage stages)
    sB: ...                            # B-matrix double/multi-buffer (ab_stage stages)
    sAIdx_prefetch: ...                # Gather index prefetch buffer (weight grad only)
```

### 4.6 Launch
```python
# grouped_gemm.py:1081-1088
self.kernel(...).launch(
    grid=grid,                           # Determined by tile scheduler
    block=[self.threads_per_cta, 1, 1],  # 3*128=384 threads (2 MMA WGs + 1 TMA WG)
    cluster=self.cluster_shape_mnk,      # (2, 1, 1) for 2-CTA clusters
    smem=allocated_smem_size,            # Typically ~200KB of SMEM
    stream=stream,
    min_blocks_per_mp=1,                 # Occupancy = 1 CTA per SM
)
```

---

<a id="5-kernel-overview"></a>
## 5. GPU Kernel Body Overview

`grouped_gemm.py:1360-2581` — The kernel splits threads into **producer** and **consumer** roles:

```
Thread Block Layout (threads_per_cta = 384 for 2 MMA WGs):
┌─────────────────────────────────────────────────────┐
│ Warp 0-3   (128 threads) │ MMA Warp Group 0         │
│ Warp 4-7   (128 threads) │ MMA Warp Group 1 (ping.) │
│ Warp 8+    (128 threads) │ TMA + gatherA Producer   │
└─────────────────────────────────────────────────────┘
```

The kernel uses **warp specialization** (`grouped_gemm.py:1680-1949` producer, `1949-2581` consumer):

```python
# grouped_gemm.py:1680-1681
if warp_idx >= self.tma_warp_id:          # Producer warps
    cute.arch.setmaxregister_decrease(self.num_regs_load)  # 56 regs for A-gather
    # ... producer loop (TMA + gatherA)

# grouped_gemm.py:1949-1950
if warp_idx < self.tma_warp_id:           # Consumer warps
    cute.arch.setmaxregister_increase(self.num_regs_mma)   # 224 regs for accumulators
    # ... consumer loop (WGMMA + epilogue)
```

---

<a id="6-producer"></a>
## 6. Producer Phase: TMA + gatherA

### 6.1 TMA Descriptor Prefetch

```python
# grouped_gemm.py:1409-1418
if warp_idx == self.tma_warp_id:
    if not self.is_A_gather:
        cpasync.prefetch_descriptor(tma_atom_a)    # Warm L2 for A TMA descriptor
    cpasync.prefetch_descriptor(tma_atom_b)        # Always prefetch B descriptor
    if not self.inference_mode:
        cpasync.prefetch_descriptor(tma_atom_d)    # Epilogue D store descriptor
    if self.need_adhoc_epilogue_store:
        cpasync.prefetch_descriptor(tma_atom_y)    # Activation output descriptor
```

**Memory bandwidth optimization**: TMA descriptor prefetch hides the latency of loading the 128-byte descriptor from GMEM. Without this, the first TMA copy would stall waiting for the descriptor.

### 6.2 Pipeline Setup

```python
# grouped_gemm.py:1435-1459
if self.is_A_gather:
    # Hybrid pipeline: TMA for B + cp.async for A
    mainloop_pipeline_producer_group = CooperativeGroup(Agent.Thread, 1 + num_load_A_threads)
    #                                                              ^   ^^^^^^^^^^^^^^^^^
    #                                                   1 TMA warp + A-gather warps
    pipeline_class = PipelineTmaCpAsync
    mcast_size = self.num_mcast_ctas_b    # B multicast only (A can't multicast)
else:
    # Pure TMA pipeline for both A and B
    pipeline_class = PipelineTmaAsync
    mcast_size = num_mcast_ctas_a + num_mcast_ctas_b - 1  # Joint multicast
```

**Why `PipelineTmaCpAsync`?** When `is_A_gather=True`, A-matrix loads use `cp.async` (software-managed async copies) because the addresses are non-contiguous. B-matrix loads still use TMA (hardware-managed). The hybrid pipeline coordinates both mechanisms through a shared barrier: the TMA warp arrives once (for B), and the A-gather warps collectively arrive once (for A). The consumer waits on the combined barrier.

### 6.3 Expert Boundary Handling

```python
# grouped_gemm.py:1737-1782  — Persistent scheduler outer loop
while work_tile.is_valid_tile:
    tile_coord_mnkl = work_tile.tile_idx     # (block_M, block_N, _, batch_idx)
    batch_idx = tile_coord_mnkl[3]           # batch_idx = expert index

    if batch_idx != last_batch_idx:
        # New expert: read its token range from expert_frequency_offset
        TIdx_cur_group  = mTokenoffset[batch_idx]      # Start token for this expert
        TIdx_next_group = mTokenoffset[batch_idx + 1]   # End token (exclusive)
        token_group_size = TIdx_next_group - TIdx_cur_group  # Number of tokens for this expert

        if self.is_A_gather:
            # Create identity tensor sized to this expert's token count
            mcA_mkl = cute.make_identity_tensor((token_group_size, mA.shape[1]))
            mAIdx_mk = cute.domain_offset((TIdx_cur_group,), mAIdx_mkl)
            # ^^^ offset the gather index pointer to this expert's section
```

**Memory bandwidth pattern**: Rather than copying tokens into a contiguous buffer per expert, the kernel reads them in-place from the original token tensor using gather indices. This avoids an O(T*K*H) memory copy before the GEMM.

### 6.4 gatherA: Token Gather Load

This is the core MoE-specific load pattern. It has two modes:

#### Mode 1: `prefetch_gather_idx_for_A_when_vary_M` (forward pass)

```python
# grouped_gemm.py:459-483
def prefetch_gather_idx_for_A_when_vary_M(self, mAIdx, M_offset, M_boundary, copy_elems):
    """Pre-load gather indices into registers for this tile's M range."""
    # Each thread loads indices for its assigned M-rows
    # stride_1_tile = K (contiguous dim), other_tile = M (gathered dim)
    M, K = self.tile_M, self.tile_K                    # e.g., 128, 64

    threads_per_stride_1_dim = K // copy_elems         # 64/8 = 8 threads per K-row
    num_other_dim_per_load = num_load_A_threads // 8   # threads available for M dimension

    tmAIdx = cute.make_rmem_tensor((num_other_dim_per_load,))  # Register buffer

    for i in range(num_other_dim_per_thread):
        other_dim_offset = i * num_other_dim_per_load + tidx // threads_per_stride_1_dim
        if other_dim_offset < M_boundary:
            M_i = M_offset + other_dim_offset
            tmAIdx[i] = mAIdx[M_i]       # Load gather index from GMEM → register
    return tmAIdx
```

**Why register-prefetch?** The gather indices are small (one int32 per M-row = 128 ints for a 128-row tile), but they're accessed repeatedly across K-tile iterations. Loading them once into registers amortizes the GMEM latency across all K-tiles.

#### `load_A_gather` (the actual scattered load)

```python
# grouped_gemm.py:551-605
def load_A_gather(self, mA, tmAIdx, sAIdx_prefetch, M_offset, tAsA, tApA,
                  A_g2s_thr_copy, K_offset, token_group_size, copy_elems):
    """Load A-matrix tile from scattered GMEM locations into SMEM."""

    # For forward pass (vary_M mode):
    for i in range(ceil_div(M, num_other_dim_per_load)):
        stride_1_dim_offset = (tidx % threads_per_K) * copy_elems   # K offset within row
        other_dim_offset = i * num_other_dim_per_load + tidx // threads_per_K  # M position

        MIdx = tmAIdx[i]          # ← Gather index from prefetched registers!
        KIdx = K_offset + stride_1_dim_offset

        # Compute element pointer: base + MIdx * stride_M + KIdx * stride_K
        tPrAptr = self.elem_pointer(mA, (MIdx, KIdx)).align(...)

        # Create 1D view at that pointer for vectorized copy (128 bits = 8 BF16 elements)
        mA_cur_copy = cute.make_tensor(tPrAptr, ((copy_elems, 1), 1))

        # cp.async from GMEM → SMEM with predication for boundary tiles
        cute.copy(A_g2s_thr_copy, mA_cur_copy, tAsA[..., stage_idx], pred=tApA[...])
```

**Diagram: gatherA data flow**
```
GMEM (original token tensor):                    SMEM (contiguous tile):
┌──────────────────────────┐                     ┌─────────────────┐
│ Token 0: [h0 h1 ... h767]│──x_gather_idx[0]──▶│ Row 0: [k0..k63]│
│ Token 1: [h0 h1 ... h767]│                     │ Row 1: [k0..k63]│
│ Token 2: [h0 h1 ... h767]│──x_gather_idx[1]──▶│ Row 2: [k0..k63]│
│ ...                      │                     │ ...              │
│ Token 8191               │──x_gather_idx[M-1]─▶│ Row M-1          │
└──────────────────────────┘                     └─────────────────┘
     scattered accesses                           contiguous for WGMMA
```

**Memory bandwidth impact**: Each A load does `M × copy_elems × sizeof(BF16)` bytes of potentially non-sequential GMEM reads. With 128-bit vectorized loads (`copy_elems=8`), each thread reads 16 bytes. For a 128×64 tile, that's `128 × 64 × 2 = 16KB` of scattered reads per K-tile. The cp.async instruction hides this latency by overlapping with computation from the previous tile.

#### Mode 2: `prefetch_gather_idx_for_A_when_vary_K` (weight gradient)

```python
# grouped_gemm.py:525-548
def prefetch_gather_idx_for_A_when_vary_K(self, mAIdx, sAIdx, token_group_size, K_offset):
    """For weight gradients: K dimension varies (K = num tokens), M = weight rows.
    Prefetch gather indices into SMEM instead of registers — larger buffer needed."""

    # Barrier before/after to ensure coherent SMEM writes
    cute.arch.barrier(barrier_id=NamedBarrierGemm.Prolog, number_of_threads=num_load_A_threads)

    for i in range(ceil_div(prefetch_token_idx_size, num_load_A_threads)):
        offset = i * num_load_A_threads + tidx
        kidx = K_offset + offset
        if kidx < token_group_size:
            sAIdx[offset] = mAIdx[kidx]   # GMEM → SMEM prefetch

    cute.arch.barrier(barrier_id=NamedBarrierGemm.Prolog, number_of_threads=num_load_A_threads)
```

For weight gradients, the transposed GEMM has K = number of tokens (potentially large), so indices are prefetched into a shared memory ring buffer of `prefetch_token_idx_size=2048` entries, refreshed every 2048 K-tiles.

### 6.5 B-matrix TMA Load

```python
# grouped_gemm.py:1886-1893
if is_tma_warp:
    cute.copy(
        tma_atom_b,
        tBgB_nkl[None, k_tile],                               # Source: GMEM B tile
        tBsB[None, mainloop_producer_state.index],             # Dest: SMEM B buffer
        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(...),# Pipeline barrier
        mcast_mask=b_mcast_mask,                               # Multicast to M-cluster CTAs
        tma_desc_ptr=b_tma_desc_ptr,                           # Dynamic TMA descriptor
    )
```

**Multicast optimization**: When `cluster_shape_mnk=(2,1)`, two CTAs share the same B tile. TMA multicast loads the tile once from GMEM and broadcasts it to both CTAs' SMEM, halving the GMEM bandwidth for B.

### 6.6 Producer Pipeline Commit

```python
# grouped_gemm.py:1926-1935
if not self.is_A_gather:
    # Pure TMA: commit is a NOP (TMA hardware manages barriers)
    mainloop_pipeline.producer_commit(mainloop_producer_state)
else:
    # Hybrid: cp.async for A needs explicit commit + fence
    mainloop_pipeline.producer_cpasync_commit(mainloop_producer_state)
mainloop_producer_state.advance()    # Move to next pipeline stage

# Pre-acquire next stage to hide acquire latency
peek_ab_empty_status = True
if k_tile + 1 < k_tile_cnt:
    peek_ab_empty_status = mainloop_pipeline.producer_try_acquire(mainloop_producer_state)
```

**Latency hiding**: `producer_try_acquire` is a non-blocking test of the next stage's availability. By checking early, the producer can overlap the acquire with the current copy, reducing pipeline stalls.

---

<a id="7-consumer"></a>
## 7. Consumer Phase: WGMMA Mainloop

### 7.1 Consumer Setup

```python
# grouped_gemm.py:1949-1990
if warp_idx < self.tma_warp_id:
    cute.arch.setmaxregister_increase(self.num_regs_mma)   # 224 registers for MMA
    cute.arch.setmaxregister_increase(self.num_regs_mma)   # Double-call ensures effect

    # Thread-to-warpgroup mapping
    warp_group_idx = tidx // 128                   # 0 or 1 (if ping-pong)
    thr_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx))

    # Partition SMEM for WGMMA operands
    tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sA))   # A register fragments
    tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sB))   # B register fragments

    # Allocate accumulator — held entirely in registers
    acc_shape = tiled_mma.partition_shape_C(tile_M × tile_N)
    acc = cute.make_rmem_tensor(acc_shape, Float32)  # e.g., 128×256/128 = 256 Float32 regs
```

### 7.2 Expert Boundary Handling (Consumer Side)

```python
# grouped_gemm.py:2032-2086
while work_tile.is_valid_tile:
    batch_idx = tile_coord_mnkl[3]
    is_group_changed = batch_idx != last_batch_idx

    if is_group_changed:
        # Read expert token range (same as producer)
        TIdx_cur_group  = mTokenoffset[batch_idx]
        TIdx_next_group = mTokenoffset[batch_idx + 1]
        token_group_size = TIdx_next_group - TIdx_cur_group

        # Update TMA descriptors for D/Y output tensors to point to this expert's range
        self.update_tma_desc_ptr(
            mD_mnl, tma_atom_d, tensormap_manager,
            d_tensormap_ptr,
            TIdx_cur_group, token_group_size,   # ← Dynamic TMA reshape
            is_tma_warp,
            tensormap_smem_ptr=d_tensormap_smem_ptr,
        )
```

**Dynamic TMA descriptor update**: Each expert has a different number of tokens. The kernel updates the TMA descriptor's shape field to `(token_group_size, N)` and its base pointer to `mD + TIdx_cur_group * stride_M`. This is done through `TensorMapManagerSm90` which writes the updated descriptor to SMEM and fences it before use.

### 7.3 WGMMA Compute Loop

This is the heart of the kernel — the K-reduction loop that calls WGMMA:

```python
# grouped_gemm.py:2088-2143

k_pipe_mmas = 1     # Number of WGMMA groups to overlap (pipelining depth within the MMA)

# ─── Prologue: Prime the pipeline ───
mainloop_consumer_release_state = mainloop_consumer_read_state.clone()
num_prologue_mma = min(k_pipe_mmas, k_tile_cnt)   # = 1

# Ping-pong: wait for our turn
if self.pingpong:
    self.pingpong_barrier_sync(warp_group_idx, stage="mma")  # ← BLOCK until partner WG signals

peek_ab_full_status = True
if k_tile_cnt > 0:
    peek_ab_full_status = mainloop_pipeline.consumer_try_wait(mainloop_consumer_read_state)

tiled_mma.set(warpgroup.Field.ACCUMULATE, False)    # First iteration: overwrite acc (not accumulate)
num_k_blocks = cute.size(tCrA, mode=[2])             # K-blocks within one SMEM stage

# ─── Prologue WGMMA (1 iteration) ───
for k_tile in range(num_prologue_mma):
    mainloop_pipeline.consumer_wait(consumer_read_state, peek_status)  # Wait for A/B data
    warpgroup.fence()                                    # Ensure previous WGMMA is visible

    for k_blk_idx in range(num_k_blocks):                # Iterate K sub-blocks (unrolled)
        k_blk_coord = (None, None, k_blk_idx, consumer_read_state.index)
        cute.gemm(tiled_mma, acc, tCrA[k_blk_coord], tCrB[k_blk_coord], acc)
        #         ^^^^^^^^^  ^^^  ^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^
        #         WGMMA op   C    A fragment (SMEM)    B fragment (SMEM)
        # First call: acc = A × B  (ACCUMULATE=False)
        # Subsequent: acc += A × B (ACCUMULATE=True)
        tiled_mma.set(warpgroup.Field.ACCUMULATE, True)

    warpgroup.commit_group()         # Submit all WGMMA in this group to the async engine
    consumer_read_state.advance()    # Move read pointer to next pipeline stage

# ─── Main WGMMA Loop ───
for k_tile in range(num_prologue_mma, k_tile_cnt):
    mainloop_pipeline.consumer_wait(consumer_read_state, peek_status)

    warpgroup.fence()
    for k_blk_idx in range(num_k_blocks):
        cute.gemm(tiled_mma, acc, tCrA[...], tCrB[...], acc)
    warpgroup.commit_group()

    # KEY: Wait for the WGMMA from k_pipe_mmas iterations ago
    warpgroup.wait_group(k_pipe_mmas)    # ← This is where compute/load overlap happens!

    # Release the SMEM stage we just consumed (producer can refill it)
    mainloop_pipeline.consumer_release(consumer_release_state)
    consumer_read_state.advance()
    consumer_release_state.advance()

    # Pre-check next stage availability
    peek_status = mainloop_pipeline.consumer_try_wait(consumer_read_state)

# ─── Drain: wait for final WGMMA ───
warpgroup.wait_group(0)    # Wait for ALL outstanding WGMMAs
# Release remaining pipeline stages
for k_tile in range(num_prologue_mma):
    mainloop_pipeline.consumer_release(consumer_release_state)
    consumer_release_state.advance()
```

**Instructions-per-cycle optimization**:
1. `consumer_try_wait` is non-blocking — it checks availability without stalling, allowing the WGMMA engine to continue
2. `warpgroup.wait_group(k_pipe_mmas)` only waits for the oldest WGMMA group. While waiting, newer WGMMAs from the current iteration are already executing on the Tensor Cores
3. The pipeline release after wait_group frees SMEM stages for the producer, which can start filling the next K-tile immediately

**Occupancy considerations**: With 1 CTA per SM and 384 threads, the SM has enough warps to hide instruction latency through warp scheduling. The WGMMA instruction itself has ~40 cycle latency but can issue every ~4 cycles, so `k_pipe_mmas=1` provides sufficient overlap.

### 7.4 Ping-Pong Signaling After MMA

```python
# grouped_gemm.py:2137-2143
if self.pingpong:
    # Signal the OTHER warp group that MMA is done — it can start its MMA
    self.pingpong_barrier_arrive(1 - warp_group_idx, stage="mma")

warpgroup.wait_group(0)     # Drain all WGMMAs

# Advance consumer pipeline for the partner WG's tile
if self.pingpong:
    mainloop_consumer_read_state.advance_iters(k_tile_cnt)
```

---

<a id="8-epilogue"></a>
## 8. Epilogue: Tiled Writeback + Activation Fusion

The epilogue writes the accumulator to GMEM, optionally fusing activation functions. It processes the tile in smaller "epi-tiles" to reduce SMEM pressure.

### 8.1 Epilogue Tile Partitioning

```python
# grouped_gemm.py:2307-2311
epi_tile_num = cute.size(tdgd_for_tma_partition, mode=[1])
# For tile (128, 256) with epi_tile (64, 32): epi_tile_num = (128/64) × (256/32) = 16
epi_tile_layout = cute.make_layout(epi_tile_shape, stride=(epi_tile_shape[1], 1))
```

**Why tile the epilogue?** The accumulator lives in registers (256 FP32 = 1024 bytes per thread × 128 threads = 128KB). Writing it all to SMEM at once would require the entire accumulator's worth of SMEM. Instead, the epilogue writes one epi-tile (64×32 = 2048 elements = 4KB BF16) at a time to a multi-buffered SMEM region, then TMA-stores it to GMEM.

### 8.2 Register → SMEM → GMEM Pipeline

```python
# grouped_gemm.py:2171-2184 — Setup StMatrix copy atoms
copy_atom_D_r2s = sm90_get_smem_store_op(d_layout, d_dtype, acc_dtype)
copy_atom_D = cute.make_copy_atom(
    warp.StMatrix8x8x16bOp(d_layout.is_m_major_c(), 4),   # StMatrix: 4 matrices at once
    self.d_dtype,
)
tiled_copy_D_r2s = cute.make_tiled_copy_S(copy_atom_D_r2s, tiled_copy_D_atom)
tRS_sD = tiled_copy_D_r2s.get_slice(tidx).partition_D(sD)
```

**StMatrix**: Hopper's `stmatrix` instruction stores an 8×8 matrix from registers to SMEM in a single instruction. By grouping 4 matrices (`num_matrices=4`), each `stmatrix` call writes 4×8×8×2 = 512 bytes. This is the most efficient register→SMEM path on Hopper.

### 8.3 Epilogue Main Loop

```python
# grouped_gemm.py:2407-2539
for epi_idx in range(epi_tile_num):    # Iterate over epi-tiles (e.g., 16 iterations)

    # ─── Step 1: Copy accumulator slice to registers ───
    for epi_v in range(cute.size(tRS_rD)):    # 16 values per thread per epi-tile
        tRS_rD[epi_v] = tRS_rAcc[epi_idx * 16 + epi_v]

    # ─── Step 2: Optional bias addition ───
    if self.use_bias:
        for epi_v in range(cute.size(tRS_rD)):
            tRS_rD[epi_v] += Float32(rBias_retiled_epi_r[epi_v])

    # ─── Step 3: Fused activation (e.g., SwiGLU for up-projection) ───
    if self.is_glu or self.is_normal_act:
        tRS_rY = cute.make_rmem_tensor_like(...)
        self.compute_activation(tRS_rD, tRS_rY)
        # For SwiGLU: y[i] = silu(z[2i]) * z[2i+1]
        # Then permute_gated_Cregs_b16 rearranges for correct SMEM store layout

    # ─── Step 4: Convert to output dtype ───
    tRS_rD_out = cute.make_rmem_tensor_like(tRS_rD, self.d_dtype)
    tRS_rD_out.store(tRS_rD.load().to(self.d_dtype))   # FP32 → BF16 conversion

    # ─── Step 5: StMatrix from registers → SMEM ───
    epi_buffer = (num_prev_subtiles + epi_idx) % d_epi_stage   # Ring buffer index
    cute.copy(tiled_copy_D_r2s, tRS_rD_out, tRS_sD[..., epi_buffer])

    if self.need_adhoc_epilogue_store:
        cute.copy(tiled_copy_Y_r2s, tRS_rY, tRS_sY[..., epi_buffer])

    # ─── Step 6: SMEM → GMEM (TMA store or scatter) ───
    cute.arch.fence_proxy(ProxyKind.async_shared, ...)   # Ensure SMEM writes visible
    epilogue_barrier.arrive_and_wait()                    # Sync all threads

    if mDIdx_mnl is not None:
        # Scatter store: D goes to non-contiguous positions
        tDsD = D_r2g_thr_copy.partition_S(sD[..., epi_buffer])
        tDrD = cute.make_rmem_tensor_like(tDsD)
        cute.autovec_copy(tDsD, tDrD)                    # SMEM → registers

        self.store_D_scatter(mD, mDIdx, tmDIdx, tDrD, ...)  # Scattered GMEM writes
        epilogue_barrier.arrive_and_wait()                   # Sync before next epi-tile
    else:
        # TMA store: contiguous output
        if is_tma_warp:
            cute.copy(tma_atom_d, bSG_sD[..., epi_buffer], bSG_gD[..., gmem_coord],
                      tma_desc_ptr=d_tma_desc_ptr)

            if self.need_adhoc_epilogue_store:
                cute.copy(tma_atom_y, bSG_sY[..., epi_buffer], bSG_gY[..., gmem_coord],
                          tma_desc_ptr=y_tma_desc_ptr)

            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(d_epi_stage - 1, read=True)
            # ^^^ Wait until at most (d_epi_stage-1) TMA stores outstanding
            # This keeps the pipeline full without overrunning SMEM buffers

        epilogue_barrier.arrive_and_wait()     # All threads sync before reusing SMEM
```

### 8.4 Activation Fusion: SwiGLU

```python
# grouped_gemm.py:1272-1289
def compute_activation(self, tRS_rD, tRS_rY):
    if self.is_glu:    # SwiGLU, ReGLU, GeGLU
        act_func = self.silu   # For SwiGLU

        # tRS_rD has interleaved gate/value pairs: [g0, v0, g1, v1, ...]
        for i in range(cute.size(tRS_rD) // 2):
            tRS_rY[i] = (act_func(tRS_rD[2*i]) * tRS_rD[2*i + 1]).to(y_dtype)
            # ^^^ y = silu(gate) × value — done entirely in registers

        self.permute_gated_Cregs_b16(tRS_rY)
        # ^^^ Rearrange interleaved halves for correct StMatrix layout
```

**`silu` implementation** (`grouped_gemm.py:412-420`):
```python
def silu(self, a):
    """silu(a) = a * sigmoid(a) — compiled to just 3 SASS instructions:
    1. FMUL   (0.5 * a)
    2. MUFU.TANH
    3. FFMA   (fma(half_a, tanh(half_a), half_a))
    """
    a_half = 0.5 * a
    return self.fma(a_half, self.tanh(a_half), a_half)
```

Using `tanh` approximation and FMA gives a 3-instruction SiLU, vs. the naive 5+ instructions with `exp` and division.

### 8.5 Scatter Store (store_D_scatter)

```python
# grouped_gemm.py:607-647
def store_D_scatter(self, mD, mDIdx, tmDIdx, tDrD, tDcD_slice, D_r2g_thr_copy,
                    epi_idx, copy_elems, tile_coord_mnkl, MIdx_cur, MIdx_next):
    """Write output tile to scattered GMEM positions (for down-projection scatter)."""

    M_offset = block_M * tile_M + MIdx_cur_group
    N_offset = block_N * tile_N

    for i in range(num_load_per_thread):
        MIdx_in_tile, NIdx_in_tile = tDcD_slice[0, i, 0]   # Coordinates within epi-tile
        MIdx = M_offset + MIdx_in_tile
        NIdx = N_offset + NIdx_in_tile

        if MIdx < MIdx_next_group and NIdx < mD.shape[1]:
            if self.is_scatter_idx_prefetched:
                SIdx = tmDIdx[i + epi_idx * num_load_per_thread]  # Pre-fetched scatter index
            else:
                SIdx = mDIdx[MIdx]     # On-demand scatter index load

            # Compute scattered output address
            tPDptr = self.elem_pointer(mD, (SIdx, NIdx)).align(...)
            mD_cur_copy = cute.make_tensor(tPDptr, ((copy_elems, 1), 1))
            cute.copy(D_r2g_thr_copy, tDrD[..., i, ...], mD_cur_copy)
```

This mirrors `load_A_gather` but in reverse: the output is scattered back to the original token positions.

---

<a id="9-pingpong"></a>
## 9. Ping-Pong Scheduling

Ping-pong mode uses **two warp groups** that alternate between MMA and epilogue phases, hiding epilogue latency behind the next tile's MMA:

```
Timeline:
  WG0: ──[MMA tile0]──[Epilogue tile0]──[MMA tile2]──[Epilogue tile2]──
  WG1: ────────────────[MMA tile1]──[Epilogue tile1]──[MMA tile3]──────
       ▲              ▲            ▲              ▲
       │              │            │              │
   WG0 signals   WG1 signals   WG0 signals   WG1 signals
   WG1 can MMA   WG0 can epi   WG1 can MMA   WG0 can epi
```

### Barrier Protocol

```python
# grouped_gemm.py:2606-2620
def pingpong_barrier_sync(self, warp_group_idx, stage):
    """Block until the OTHER warp group has arrived at this barrier."""
    barrier = NamedBarrierGemm.MmaWG0 if stage == "mma" else NamedBarrierGemm.EpiWG0
    cute.arch.barrier(
        barrier_id=int(barrier) + warp_group_idx,
        number_of_threads=2 * 128,    # Both WGs must participate (256 threads total)
    )

def pingpong_barrier_arrive(self, warp_group_idx, stage):
    """Signal that this warp group is done with its phase."""
    barrier = NamedBarrierGemm.MmaWG0 if stage == "mma" else NamedBarrierGemm.EpiWG0
    cute.arch.barrier_arrive(
        barrier_id=int(barrier) + warp_group_idx,
        number_of_threads=2 * 128,
    )
```

Named barriers used (from `NamedBarrierGemm` enum):
- `MmaWG0/MmaWG1`: Coordinate MMA start between WG0 and WG1
- `EpiWG0/EpiWG1`: Coordinate epilogue (SMEM reuse) between WG0 and WG1
- `Epilogue`: General epilogue sync (all epi threads)

### Ping-Pong Pipeline State Management

```python
# grouped_gemm.py:2006-2025  — WG1 advances past WG0's work
if const_expr(self.pingpong):
    if warp_idx >= 4:    # WG1
        # WG0 processes tile0 first. WG1 starts at tile1.
        # So WG1's mainloop_consumer_read_state must skip WG0's K-tiles.
        tile_scheduler.advance_to_next_work()
        mainloop_consumer_read_state.advance_iters(k_tile_cnt)
        # This ensures WG1's consumer pipeline reads from the right SMEM stage
```

### Why Ping-Pong Helps

For small tile configs (e.g., 192×128), the epilogue can be a significant fraction of the total time. Ping-pong overlaps WG0's epilogue with WG1's MMA:

- **Without ping-pong**: `MMA → Epilogue → MMA → Epilogue` (serial)
- **With ping-pong**: `MMA(WG0) → [MMA(WG1) | Epilogue(WG0)] → [MMA(WG0) | Epilogue(WG1)]`

The SMEM for D/Y output is **shared** between WG0 and WG1, which is why the `EpiWG` barrier ensures one WG's TMA store completes before the other overwrites SMEM:

```python
# grouped_gemm.py:2568-2577
if self.pingpong:
    # Wait for all TMA stores to complete before releasing SMEM
    if warp_idx == 0 or warp_idx == 4:
        cute.arch.cp_async_bulk_wait_group(0, read=True)
    # Signal the other WG that SMEM is free for its epilogue
    self.pingpong_barrier_arrive(1 - warp_group_idx, stage="epi")
```

---

<a id="10-smem"></a>
## 10. Shared Memory Budget & Stage Computation

### Stage Computation (`_compute_stages`, lines 2641-2720)

The kernel maximizes the number of pipeline stages (double/multi-buffering) within the 228KB SMEM budget:

```python
# grouped_gemm.py:2683-2696
# A tile: (tile_M, tile_K) = (128, 64) × 2B = 16KB per stage
# B tile: (tile_N, tile_K) = (256, 64) × 2B = 32KB per stage
# A+B: 48KB per stage
ab_bytes_per_stage = size(a_shape) * a_dtype.width/8 + size(b_shape) * b_dtype.width/8

# Reserved:
# - Barriers:           ~1KB
# - Tensormap:           ~1KB
# - Epilogue D buffer:   d_epi_stage × (epi_tile_M × epi_tile_N × sizeof(d_dtype))
# - Epilogue Y buffer:   y_epi_stage × (y_tile_M × y_tile_N × sizeof(y_dtype))
# - Bias buffer:         tile_N × sizeof(bias_dtype) if present
# - Prefetch indices:    prefetch_token_idx_size × 4B if weight gradient

remaining_bytes = (smem_capacity - 1KB) - mbar_bytes - epi_bytes - ...
ab_stage = remaining_bytes // ab_bytes_per_stage
# Result: typically ab_stage = 3-5 depending on epi config
```

For the default up-projection config `(128, 256, 64)` with BF16:
- A per stage: 128×64×2 = 16KB
- B per stage: 256×64×2 = 32KB
- A+B per stage: 48KB
- Epilogue D: 2 stages × 64×32×2 = 8KB
- Epilogue Y: 2 stages × 64×16×2 = 4KB (GLU halves the N dim)
- Available ≈ 228KB - 1KB - 8KB - 4KB ≈ 215KB
- `ab_stage = 215KB / 48KB ≈ 4`

With 4 A/B stages, the producer can fill 3 stages ahead while the consumer processes 1, providing substantial latency hiding for TMA and gatherA loads.

### Epilogue Stage Refinement

```python
# grouped_gemm.py:2698-2718
# After computing ab_stage, reclaim any leftover SMEM for more epilogue stages
if not overlap_sD_sA:
    epi_stage_delta = (remaining - ab_stage * ab_bytes) // (d_bytes + y_bytes + c_bytes)
    d_epi_stage += epi_stage_delta
    y_epi_stage += epi_stage_delta
```

More epilogue stages means the TMA store pipeline has more buffers, reducing stalls when GMEM write bandwidth is saturated.

### Inference Mode Optimization

```python
# grouped_gemm.py:2660-2661
if self.inference_mode and self.need_adhoc_epilogue_store:
    d_epi_stage = 0     # Don't buffer D at all — only Y (activated output) is needed
```

During inference, the pre-activation output `D` is never needed outside the kernel (no backward pass). The kernel sets `d_epi_stage=0` and aliases `sD` with `sY` in SMEM, reclaiming the D buffer space for more A/B stages.

---

## Summary: Key Optimization Patterns

| Pattern | Where | Impact |
|---------|-------|--------|
| **gatherA** | `load_A_gather` (line 551), `prefetch_gather_idx_for_A_when_vary_M` (line 459) | Avoids O(T*K*H) token copy; reads scattered tokens directly with cp.async |
| **Register prefetch of gather indices** | Lines 459-483, stored in `tmAIdx` | Amortizes index load latency across all K-tile iterations |
| **TMA multicast for B** | Line 1886-1893, `mcast_mask=b_mcast_mask` | Halves GMEM bandwidth for weights in 2-CTA clusters |
| **Dynamic TMA descriptors** | `update_tma_desc_ptr` (line 1091) | Per-expert output tensor reshaping without kernel relaunch |
| **warpgroup.wait_group(k_pipe_mmas)** | Line 2130 | Overlaps WGMMA execution with pipeline management |
| **producer_try_acquire / consumer_try_wait** | Lines 1847, 2101, 2135 | Non-blocking pipeline probes hide acquire latency |
| **setmaxregister** | Lines 1681, 1950 | Dynamic register partitioning: 56 regs for producers, 224 for MMA |
| **StMatrix 4-wide** | Line 2177 | 512-byte register→SMEM transfers per instruction |
| **Epilogue tiling** | Lines 2407-2539, `d_epi_tile=(64,32)` | Reduces SMEM pressure from 128KB (full tile) to 4KB per iteration |
| **SiLU = 3 SASS instructions** | `silu()` at line 412 | FMA+TANH.APPROX instead of exp/div chain |
| **Ping-pong** | Lines 2606-2620, barrier_arrive/sync | Overlaps WG0 epilogue with WG1 MMA |
| **Inference sD/sY aliasing** | Lines 1503-1507, `d_epi_stage=0` | Reclaims epilogue SMEM for more A/B pipeline stages |
| **SwiGLU GLU permutation** | `permute_gated_Cregs_b16` (line 718) | Rearranges interleaved gate/value pairs via warp shuffles for correct store layout |
