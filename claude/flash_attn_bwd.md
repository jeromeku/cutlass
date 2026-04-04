# Flash Attention SM90 Backward Pass — Deep Trace

**Date:** 2026-02-14
**Source branch:** HEAD
**Kernel files:** `thirdparty/flash-attention/flash_attn/cute/`

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Backward Algorithm — The Math](#2-backward-algorithm--the-math)
3. [Launcher: `_flash_attn_bwd`](#3-launcher-_flash_attn_bwd)
4. [Phase 1 — Preprocess Kernel](#4-phase-1--preprocess-kernel)
5. [Phase 2 — Main Backward Kernel](#5-phase-2--main-backward-kernel)
   - 5.1 [`FlashAttentionBackwardSm90.__init__`](#51-flashattentionbackwardsm90__init__)
   - 5.2 [`__call__` — Host Launcher](#52-__call__--host-launcher)
   - 5.3 [`kernel` — GPU Entry Point](#53-kernel--gpu-entry-point)
   - 5.4 [`load` — Producer Warp](#54-load--producer-warp)
   - 5.5 [`mma` — Consumer Warpgroups (Setup)](#55-mma--consumer-warpgroups-setup)
   - 5.6 [`mma_one_m_block` — Inner Loop Body](#56-mma_one_m_block--inner-loop-body)
   - 5.7 [`dQaccum_store` — dQ Atomic Writer Warp](#57-dqaccum_store--dq-atomic-writer-warp)
   - 5.8 [`epilogue_dKV` — dK/dV Write-Back](#58-epilogue_dkv--dkdv-write-back)
6. [score_mod and mask_mod — Forward vs Backward Differences](#6-score_mod-and-mask_mod--forward-vs-backward-differences)
   - 6.1 [Execution Context](#61-execution-context)
   - 6.2 [score_mod: Forward vs Backward](#62-score_mod-forward-vs-backward)
   - 6.3 [mask_mod: Forward vs Backward](#63-mask_mod-forward-vs-backward)
   - 6.4 [Gradient Flow Through dV, dQ, dK](#64-gradient-flow-through-dv-dq-dk)
   - 6.5 [SdP_swapAB and Transposed Index Coordinates](#65-sdp_swapab-and-transposed-index-coordinates)
   - 6.6 [Compile-time Embedding and Constraints](#66-compile-time-embedding-and-constraints)
7. [VJP Derivation — How `score_mod_bwd` is Created](#7-vjp-derivation--how-score_mod_bwd-is-created)
   - 7.1 [Stage 1 — Autograd dispatch: `create_fw_bw_graph`](#71-stage-1--autograd-dispatch-create_fw_bw_graph)
   - 7.2 [Stage 2 — `create_joint`: symbolic differentiation](#72-stage-2--create_joint-symbolic-differentiation)
   - 7.3 [Stage 3 — Inductor lowering: subgraph buffers](#73-stage-3--inductor-lowering-subgraph-buffers)
   - 7.4 [Stage 4 — Jinja template: rendering `score_mod_bwd`](#74-stage-4--jinja-template-rendering-score_mod_bwd)
   - 7.5 [End-to-end call chain](#75-end-to-end-call-chain)
8. [Phase 3 — Postprocess Kernels](#8-phase-3--postprocess-kernels)
9. [Thread Layout and Register Budget](#9-thread-layout-and-register-budget)
10. [Key Performance Optimizations](#10-key-performance-optimizations)
11. [Key Functions Index](#11-key-functions-index)
12. [Code Map](#12-code-map)

---

## 1. Architecture Overview

The CuTeDSL Flash Attention backward pass on SM90 is split into **three GPU kernel launches**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         _flash_attn_bwd  (Python)                           │
│  ┌──────────────────┐  ┌───────────────────────────────┐  ┌──────────────┐ │
│  │  PREPROCESS      │  │  MAIN BACKWARD                │  │ POSTPROCESS  │ │
│  │  (128 threads)   │─▶│  FlashAttentionBackwardSm90   │─▶│ (256 threads)│ │
│  │                  │  │  (384 threads/block)           │  │              │ │
│  │ • delta = ΣdO*O  │  │  • S = Q @ Kᵀ  (GEMM 1)      │  │ • fp32 dQ    │ │
│  │ • LSE → log2 dom │  │  • dP = dO @ Vᵀ (GEMM 2)     │  │   → fp16 dq  │ │
│  │ • zero dQaccum   │  │  • P = exp2(S*log2e - LSE)    │  │ • [GQA]      │ │
│  └──────────────────┘  │  • dS = P*(dP - dPsum)        │  │   fp32 dK,dV │ │
│                        │  • dV += Pᵀ @ dO  (GEMM 3)   │  │   → fp16     │ │
│                        │  • dQ += dS @ K   (GEMM 4)   │  └──────────────┘ │
│                        │  • dK += dSᵀ @ Q  (GEMM 5)  │                    │
│                        │  • dQ: reg→smem→TMA add      │                    │
│                        │  • dK, dV: STMATRIX + TMA    │                    │
│                        └───────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Grid dimension** (main backward): one thread block per K/V tile `(n_block, head, batch)`.
Each block iterates over all Q tiles that attend to its K/V tile (the "outer KV, inner Q" loop).
This is the **transpose** of the forward loop (which iterates over K/V tiles per Q tile).

**Thread-block layout** (384 threads = 3 warpgroups):

```
warps 0–3  (wg idx 0, 128 threads) → MMA consumer warpgroup 0
warps 4–7  (wg idx 1, 128 threads) → MMA consumer warpgroup 1 (if present)
warp  0    (wg 0, warp 0)          → load() TMA producer
warp  1    (wg 0, warp 1)          → dQaccum_store() TMA reduce writer
```

For SM90 with `num_threads=384`: `num_mma_warp_groups = (384 // 128) - 1 = 2`.

---

## 2. Backward Algorithm — The Math

Given: `Q, K, V` (fp16), `out` (fp16), `dout` (fp16), `lse` (fp32, from forward).

**Preprocess** (per query token `i`):
```
delta_i  = Σ_j  out_i[j] * dout_i[j]          # row-wise dot product
lse_log2 = lse * log₂e                         # convert nats→bits for exp2
dQaccum  = zeros(seqlen_q_rounded, head_dim, fp32)  # accumulator for atomic dQ
```

**Main backward** (per K/V tile, iterating over Q tiles):

For each (Q tile `i`, K/V tile `j`):

```
S_ij     = Q_i @ Kⱼᵀ                           # raw attention logits (GEMM 1)
dP_ij    = dO_i @ Vⱼᵀ                           # gradient w.r.t. P  (GEMM 2)
[score_mod(S_ij)]                               # optional score modification
[mask(S_ij)]                                    # optional masking → -inf
P_ij     = exp2(S_ij * softmax_scale_log2 - LSE_i)   # recomputed attention weights
dS_ij    = P_ij * (dP_ij - dPsum_i)            # backward through softmax
[score_mod_bwd(dS_ij, S_ij)]                   # optional VJP of score_mod
dV_j    += P_ijᵀ @ dO_i                        # (GEMM 3)
dQ_i    += dS_ij @ K_j                         # (GEMM 4, atomic)
dK_j    += dS_ijᵀ @ Q_i                        # (GEMM 5)
dK_j    *= softmax_scale                        # MHA: scale after all Q tiles
```

**Key identity:** `dPsum_i = delta_i = Σ_j P_ij * dP_ij` (precomputed).

---

## 3. Launcher: `_flash_attn_bwd`

**File:** [interface.py#L554](../thirdparty/flash-attention/flash_attn/cute/interface.py#L554)

### Frame 1 — Entry and SM90 Hyperparameter Selection

```python
def _flash_attn_bwd(q, k, v, out, dout, lse, softmax_scale, causal, ...):
    compute_capability = _get_device_capability()

    if compute_capability == 9:
        m_block_size = 80 if not causal else 64   # M-dim tile (Q rows per block)
        n_block_size = 128                          # N-dim tile (K/V rows per block)
        num_stages_Q = 2                            # double-buffered Q pipeline
        num_stages_dO = 2                           # double-buffered dO pipeline
        num_stages_PdS = 2                          # double-buffered P/dS pipeline
        SdP_swapAB = True                           # transpose S GEMM for efficiency
        dQ_swapAB = not causal                      # layout for dQ GEMM
```

`SdP_swapAB=True` swaps A/B operands in the S=QKᵀ and dP=dOVᵀ GEMMs.
When swapped, the effective GEMM becomes `Kᵀ @ Q` or `Vᵀ @ dO`, which
changes the data-layout assumptions and allows different smem layouts.

### Frame 2 — Allocate Intermediate Buffers

```python
# fp32 dQ accumulator — receives atomic-add contributions from multiple Q tiles
dq_accum = torch.empty(
    batch_size, num_head,
    seqlen_q_rounded * head_dim_rounded,   # flattened
    dtype=torch.float32, device=device
)
# dPsum: delta = Σ dO*O per token — precomputed in preprocess kernel
dpsum = torch.empty(batch_size, num_head, seqlen_q_rounded, dtype=torch.float32, ...)
# LSE converted to log2 domain for exp2 efficiency
lse_log2 = torch.empty(batch_size, num_head, seqlen_q_rounded, dtype=torch.float32, ...)

# GQA: dK/dV are first accumulated as fp32 (multi-head reduce), then postprocessed
if dKV_postprocess:   # qhead_per_kvhead > 1
    dk_accum = torch.zeros(batch_size, num_head_kv, seqlen_k_rounded * head_dim_rounded, ...)
    dv_accum = torch.zeros(...)
```

`dq_accum` is zeroed by the preprocess kernel (not `torch.zeros`), saving a memset.
`dk_accum / dv_accum` must be pre-zeroed (`torch.zeros`) because multiple thread blocks
can contribute to the same K/V tile via `cp.reduce.bulk.async.global.add.f32`.

### Frame 3 — Compile Cache Key

```python
compile_key = (
    compute_capability,  # 9
    dtype,               # cutlass.Float16 / BFloat16
    head_dim,
    head_dim_v,
    qhead_per_kvhead,
    causal,
    softcap != 0.0,
    m_block_size, n_block_size, num_threads,
    pack_gqa,
    num_stages_Q, num_stages_dO,
    SdP_swapAB, dKV_swapAB, dQ_swapAB,
    AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ,
    V_in_regs,
    cu_seqlens_q is None, cu_seqlens_k is None,
    seqused_q is None, seqused_k is None,
    score_mod_hash, score_mod_bwd_hash, mask_mod_hash,
    num_aux_tensors,
    use_block_sparsity, block_sparse_broadcast_pattern,
)
```

Each unique key corresponds to a distinct JIT-compiled CUBIN.
`score_mod_hash` and `score_mod_bwd_hash` are Python function id hashes — different
callables produce distinct kernels baked with the functions as `Constexpr`.

### Frame 4 — Three-Phase Dispatch

```python
# Phase 1: preprocess (num_threads=128)
_flash_attn_bwd.compile_cache_pre[compile_key_pre](out, dout, dpsum, lse, lse_log2, dq_accum, ...)

# Phase 2: main backward (num_threads hardcoded to 384 after pre)
num_threads = 384
_flash_attn_bwd.compile_cache[compile_key](q, k, v, dout, lse_log2, dpsum, dq_accum, dk, dv, ...)

# Phase 3: postprocess — fp32 dQaccum → fp16 dq  (num_threads=256)
_flash_attn_bwd.compile_cache_post[compile_key_post](dq_accum, dq, softmax_scale, ...)
# GQA path: also convert fp32 dk_accum / dv_accum → fp16 dk / dv
```

---

## 4. Phase 1 — Preprocess Kernel

**File:** [flash_bwd_preprocess.py](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_preprocess.py)
**Class:** `FlashAttentionBackwardPreprocess`
**Grid:** `(ceil(seqlen_q / m_block_size), num_head, batch_size)`
**Threads:** 128

Each thread block processes one `(m_block, head, batch)` tile. Three tasks:

### Task A — Compute `delta = dPsum` (line 301)

```python
# load O tile: (m_block_size, head_dim) → registers
# load dO tile: same shape
tOrO  = cute.make_fragment_like(tOgO)
tOrdO = cute.make_fragment_like(tOgdO)
cute.copy(gmem_thr_copy_O, tOgO, tOrO, pred=tOpO)    # global → regs
cute.copy(gmem_thr_copy_O, tOgdO, tOrdO, pred=tOpdO) # global → regs

# element-wise dot product: sum over head_dim
dpsum = (tOrO.load().to(Float32) * tOrdO.load().to(Float32)).reduce(
    cute.ReductionOp.ADD, init_val=0.0, reduction_profile=(0, None, 1)  # reduce over K dim
)
dpsum = utils.warp_reduce(dpsum, operator.add, width=threads_per_row)   # warp shuffle
# write per-row delta to gdPsum global tensor
```

**Why delta?** In the backward through softmax: `dS = P * (dP - dPsum)`.
The `dPsum` term is `Σ_j P_ij * dP_ij = Σ_j attn_weight_ij * (dO_i · V_j)`.
By definition this equals `Σ_k dO_i[k] * out_i[k] = delta_i` (the row-wise dot).

### Task B — Convert LSE to log₂ domain (line 354)

```python
LOG2_E = math.log2(math.e)   # ≈ 1.4426950408
if tidx < seqlen_q - m_block * self.m_block_size:
    lse = gLSE[tidx]          # read forward LSE (in nats)
# ...
gLSElog2[tidx] = lse * LOG2_E if lse != -Float32.inf else 0.0
```

The forward kernel stores LSE in nats; `exp(S - LSE) = exp2(S*log2e - LSE*log2e)`.
This pre-multiplication lets the main backward use the faster `exp2` instruction.

### Task C — Zero `dQaccum` (line 319)

```python
gdQaccum = cute.local_tile(mdQaccum_cur, (m_block_size * head_dim_padded,), (m_block,))
zero = cute.make_fragment_like(tdQgdQaccum)
zero.fill(0.0)
cute.copy(gmem_tiled_copy_dQaccum, zero, tdQgdQaccum)   # 128-bit stores
```

Zeroing is co-located with `delta` computation so both O/dO tensors are already hot
in caches. Block size = `m_block_size * head_dim_rounded` elements of fp32.

---

## 5. Phase 2 — Main Backward Kernel

### 5.1 `FlashAttentionBackwardSm90.__init__`

**File:** [flash_bwd_sm90.py#L38](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py#L38)

Key fields stored at construction time (all become `Constexpr` in JIT):

| Field | Default (SM90 causal) | Notes |
|---|---|---|
| `tile_m` | 64 | Q rows per tile |
| `tile_n` | 128 | K/V rows per tile |
| `Q_stage` | 2 | Double-buffered Q pipeline |
| `dO_stage` | 2 | Double-buffered dO pipeline |
| `PdS_stage` | 2 | Double-buffered P/dS smem |
| `SdP_swapAB` | True | Transpose S=QKᵀ GEMM |
| `dQ_swapAB` | False (causal) | Layout for dQ=dS@K GEMM |
| `AtomLayoutMSdP` | 1 | Warpgroup atoms along M for S/dP |
| `AtomLayoutNdKV` | 2 | Warpgroup atoms along N for dK/dV |
| `num_mma_warp_groups` | 2 | `(384 // 128) - 1` |
| `mma_dkv_is_rs` | True | P/dS stay in registers for dK/dV GEMMs |
| `vec_size` | 4 (no aux) / 1 (aux) | Elements per `score_mod` call |

`mma_dkv_is_rs=True` when `AtomLayoutMSdP==1 and AtomLayoutNdKV==num_wg and SdP_swapAB and not dKV_swapAB`.
When true, P (fp16) converted from `acc_S` stays in registers as `tCrA` for the `dV += P@dO` GEMM,
skipping a round-trip through smem. Similarly `dS` stays in regs for `dK += dS@Q`.

### 5.2 `__call__` — Host Launcher

**File:** [flash_bwd_sm90.py#L302](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py#L302)

#### Frame 1 — Layout Transpose

```python
layout_transpose = [1, 3, 2, 0]   # (b, s, n, h) → (s, h, n, b)
mQ, mK, mV, mdO = [layout_utils.select(t, layout_transpose) for t in (mQ, mK, mV, mdO)]
```

Reorders logical dimensions for CuTe `local_tile` access.
After transpose: `mQ[s, h, n, b]` → stride order puts the seqlen (`s`) and head_dim (`h`)
dimensions contiguously for 2D TMA copy atoms of shape `(tile_m, tile_hdim)`.

For GQA (`qhead_per_kvhead > 1`), `mdK/mdV` are fp32 accumulator tensors already
in `(b, n_kv, seqlen*hdim)` order, so they use `accum_transpose = [2, 1, 0]`.

#### Frame 2 — TMA Descriptors

```python
tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
    cpasync.CopyBulkTensorTileG2SOp(),   # g2s = global-to-shared
    mQ,
    cute.select(sQ_layout, mode=[0, 1]), # swizzled smem layout for this tile
    (tile_m, tile_hdim),                 # tile shape
)
# similarly: K (single-stage, not pipelined), V, dO
# and for MHA epilogue: tma_atom_dK, tma_atom_dV  (s2g = shared-to-global)
```

TMA descriptors are created once on host, passed as kernel arguments, and prefetched
to L2 on GPU. K and V are loaded **once per n_block** (not pipelined); Q and dO
are pipelined with double-buffering across m_blocks.

For the dK/dV TMA write back (`CopyBulkTensorTileS2GOp`), GQA path uses a different
codepath: `tma_atom_dK = None` and instead uses `cp.reduce.bulk.async.global.add.f32`.

#### Frame 3 — `softmax_scale_log2` Encoding

```python
LOG2_E = math.log2(math.e)
if const_expr(self.score_mod is None):
    softmax_scale_log2 = softmax_scale * LOG2_E  # bake scale in
else:
    softmax_scale_log2 = LOG2_E                  # scale applied inside score_mod
```

Matches forward kernel behavior: with `score_mod`, the user function is expected to
apply its own scale factor. Without `score_mod`, the kernel folds `scale * log2e`
into the exponent so `exp2(S * softmax_scale_log2 - lse_log2)` computes `softmax(S*scale)`.

#### Frame 4 — Grid and Launch

```python
TileScheduler = SingleTileScheduler
# Grid = (n_blocks, num_head, batch_size): one block per KV tile
grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

self.kernel(...).launch(
    grid=grid_dim,
    block=[384, 1, 1],
    smem=SharedStorage.size_in_bytes(),
    stream=stream,
    min_blocks_per_mp=1,
)
```

Unlike the forward which can use `LPT` or `Varlen` schedulers, the backward always
uses `SingleTileScheduler` (no persistent kernel). `min_blocks_per_mp=1` requests
at least one block per SM to improve occupancy.

### 5.3 `kernel` — GPU Entry Point

**File:** [flash_bwd_sm90.py#L517](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py#L517)

```python
@cute.kernel
def kernel(self, ...):
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    # All warps prefetch TMA descriptors to L2
    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_atom_Q)
        cpasync.prefetch_descriptor(tma_atom_K)
        cpasync.prefetch_descriptor(tma_atom_V)
        cpasync.prefetch_descriptor(tma_atom_dO)

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)   # SharedStorageQKV struct
```

**Smem layout** (`SharedStorageQKV`):

| Field | Size (tile_m=64, tile_n=128, hdim=128) | Notes |
|---|---|---|
| `mbar_ptr_Q` | 2×stages × 8B = 32B | Pipeline mbarriers for Q |
| `mbar_ptr_dO` | 2×stages × 8B | Pipeline mbarriers for dO |
| `sLSE` | 64 rows × 2 stages × 4B = 512B | LSE per-row values |
| `sdPsum` | 64 rows × 2 stages × 4B | delta values per row |
| `sQ` | 64×128 × 2 stages × 2B = 32KB | Double-buffered Q |
| `sV` | 128×128 × 1 stage × 2B = 32KB | Single-stage V |
| `sK` | 128×128 × 1 stage × 2B = 32KB | Single-stage K |
| `sdO` | 64×128 × 2 stages × 2B = 32KB | Double-buffered dO |
| `sP` | 64×128 × 2B (if !mma_dkv_is_rs) | Attention weights P |
| `sdS` | 64×128 × 2B | Gradient dS |
| `sdQaccum` | 64×128/2WG × fp32 | dQ register→smem staging |

```python
    # Create TMA pipelines
    pipeline_Q = pipeline.PipelineTmaAsync.create(
        barrier_storage=storage.mbar_ptr_Q.data_ptr(),
        num_stages=Q_stage,
        tx_count=tma_copy_bytes["Q"] + tma_copy_bytes["LSE"],  # combined transaction
        ...
    )
    pipeline_dO = pipeline.PipelineTmaAsync.create(
        ...
        tx_count=tma_copy_bytes["dO"] + tma_copy_bytes["dPsum"],  # combined
        defer_sync=False,
    )
```

Each pipeline covers two TMA transactions (Q+LSE, or dO+dPsum) in a single mbarrier
`tx_count`. The producer issues both TMA loads for the same smem stage before
calling `producer_get_barrier`, so both are counted together.

```python
    # Warp dispatch
    if warp_idx < 4:                                    # warps 0-3 = producer warpgroup
        cute.arch.setmaxregister_decrease(24)           # free regs to consumers
        if warp_idx == 0:
            self.load(...)                              # TMA producer
        if warp_idx == 1:
            self.dQaccum_store(...)                     # dQ TMA reduce writer
    else:                                               # warps 4-11 = 2 consumer WGs
        cute.arch.setmaxregister_increase(240)          # 240 regs for MMA
        tidx = cute.arch.thread_idx()[0] - 128          # re-base thread index
        self.mma(...)
```

Register split: producer warps sacrifice registers (`setmaxregister_decrease(24)`)
to give the MMA warps 240 registers each. This asymmetric split maximizes WGMMA
throughput by keeping accumulators register-resident.

### 5.4 `load` — Producer Warp

**File:** [flash_bwd_sm90.py#L720](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py#L720)

The single producer warp (warp 0) runs a tile-scheduler loop over K/V tiles,
and for each tile iterates over all Q tiles that interact with it:

```python
# For each (n_block, head, batch) tile in this block's scheduler:
while work_tile.is_valid_tile:
    n_block, head_idx, batch_idx, _ = work_tile.tile_idx
    m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
    # causal: m_block_max = ceil(n_block * tile_n / tile_m)
    # full: m_block_min=0, m_block_max=seqlen_q/tile_m

    # Load K and V once (they are fixed for this KV tile)
    # K is loaded in the first Q-stage slot; V in the first dO-stage slot
    pipeline_Q.producer_acquire(
        producer_state_Q, extra_tx_count=tma_copy_bytes["K"]  # K in same barrier
    )
    load_K(tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q))  # K → sK
    load_Q(first_m_block, producer_state=producer_state_Q)                 # Q → sQ
    load_LSE(first_m_block, producer_state=producer_state_Q)               # LSE → sLSE

    pipeline_dO.producer_acquire(
        producer_state_dO_cur, extra_tx_count=tma_copy_bytes["V"]  # V in same barrier
    )
    load_V(tma_bar_ptr=pipeline_dO.producer_get_barrier(...))       # V → sV
    load_dO(first_m_block, ...)                                     # dO → sdO
    load_dPsum(first_m_block, ...)                                  # dPsum → sdPsum

    # Subsequent Q tiles (K and V already in smem, not reloaded)
    for m_block in range(m_block_min + 1, m_block_max):
        pipeline_Q.producer_acquire(producer_state_Q)
        load_Q(m_block, ...)
        load_LSE(m_block, ...)
        load_dO(m_block, ...)
        load_dPsum(m_block, ...)
```

**Key asymmetry**: K and V are loaded **once** into single-stage smem buffers at the
start of each n_block. Q and dO are loaded every m_block into double-buffered smem.
This saves bandwidth since K/V are shared across all Q tiles for a given KV tile.

The `extra_tx_count` technique piggybacks K's transaction count onto pipeline_Q's
first mbarrier, so K and Q[0] share the same completion barrier.

### 5.5 `mma` — Consumer Warpgroups (Setup)

**File:** [flash_bwd_sm90.py#L966](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py#L966)

`mma()` is called by all MMA consumer threads (256 threads = 2 × 128-thread WGs).
It first partitions all GEMM operands and creates partial-application closures:

```python
# Each WG gets its own ThrMma slice
thr_mma_SdP = tiled_mma_SdP.get_slice(tidx)
wg_mma_SdP  = tiled_mma_SdP.get_slice(warp_group_thread_layout(warp_group_idx))

# GEMM 1: S = Q @ Kᵀ  (or Kᵀ @ Q when SdP_swapAB)
_, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(wg_mma_SdP, (tile_m, tile_n, hdim), sQ, sK,
                                                   swap_AB=SdP_swapAB)
mma_qk_fn = partial(gemm_zero_init, tiled_mma_SdP, (tile_m, tile_n), tSrQ, tSrK, swap_AB=SdP_swapAB)

# GEMM 2: dP = dO @ Vᵀ
_, tdPrdO, tdPrV = sm90_utils.partition_fragment_ABC(wg_mma_SdP, (tile_m, tile_n, hdimv), sdO, sV,
                                                      swap_AB=SdP_swapAB)
mma_dov_fn = partial(gemm_zero_init, tiled_mma_SdP, (tile_m, tile_n), tdPrdO, tdPrV, ...)

# GEMM 3: dV += Pᵀ @ dO  — acc_dV lives in WG registers across all m_blocks
acc_dV, tdVrPt, tdVrdOt = sm90_utils.partition_fragment_ABC(wg_mma_dV, (tile_n, hdimv, tile_m),
                                                              sPt, sdOt, swap_AB=dKV_swapAB)
# when mma_dkv_is_rs: P stays in registers, no smem write
mma_pdo_fn = partial(gemm_w_idx, tiled_mma_dV, acc_dV, tCrB=tdVrdOt)

# GEMM 5: dK += dSᵀ @ Q  — acc_dK lives in WG registers
acc_dK, tdKrdSt, tdKrQt = sm90_utils.partition_fragment_ABC(wg_mma_dK, (tile_n, hdim, tile_m), ...)
mma_dsq_fn = partial(gemm_w_idx, tiled_mma_dK, acc_dK, tCrB=tdKrQt)

# GEMM 4: dQ = dS @ K
_, tdQrdS, tdQrKt = sm90_utils.partition_fragment_ABC(wg_mma_dQ, (tile_m, hdim, tile_n), sdS, sKt, ...)
mma_dsk_fn = partial(gemm_zero_init, tiled_mma_dQ, (tile_m, hdim), tdQrdS, tdQrKt, ...)
```

`acc_dV` and `acc_dK` are **persistent** register tensors across the entire inner loop
over m_blocks. They accumulate contributions from each Q tile and are only written to
global memory in `epilogue_dKV` after all m_blocks are processed.

`acc_dQ`, by contrast, is **fresh** each m_block (computed by `gemm_zero_init`) and
immediately written to smem / reduced to global via `dQaccum_store`.

### 5.6 `mma_one_m_block` — Inner Loop Body

**File:** [flash_bwd_sm90.py#L1269](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py#L1269)

This is the heart of the backward pass — one iteration per Q tile `m_block`.
State entering: `acc_dV` and `acc_dK` hold accumulated dV/dK from previous m_blocks.

#### Step 1 — GEMM 1: S = Q @ Kᵀ  (line 1304)

```python
# Wait for Q (and K in first iteration) to arrive in smem
pipeline_Q.consumer_wait(consumer_state_Q, pipeline_Q.consumer_try_wait(consumer_state_Q))
acc_S = mma_qk_fn(A_idx=smem_idx_Q, wg_wait=-1)  # wg_wait=-1 = don't wait for completion yet
```

`wg_wait=-1`: issue WGMMA instructions but do not emit `wgmma.wait_group.sync.aligned`.
This allows GEMM 2 to be issued immediately after (intra-block overlap).

Simultaneously, load the LSE row-vector for this Q tile from smem:

```python
tLSErLSE = copy_utils.load_s2r(tLSEsLSE[None, smem_idx_Q])  # smem → regs (scalars per row)
```

#### Step 2 — GEMM 2: dP = dO @ Vᵀ  (line 1311)

```python
pipeline_dO.consumer_wait(consumer_state_dO_cur, ...)
acc_dP = mma_dov_fn(A_idx=smem_idx_Q, wg_wait=1)   # wg_wait=1 = wait for GEMM 2 complete
                                                     # (also implicitly waits for GEMM 1)
```

`wg_wait=1` emits `wgmma.wait_group 1`, which waits until at most 1 outstanding WGMMA
group remains — i.e., GEMM 1 (`acc_S`) has completed.

#### Pre-save S for `score_mod_bwd`

```python
if const_expr(self.score_mod_bwd is not None):
    acc_S_pre = cute.make_fragment_like(acc_S)
    cute.autovec_copy(acc_S, acc_S_pre)   # save raw QKᵀ logits before modification
```

`score_mod_bwd` needs the **unmodified** scores as input (see §6).

#### Step 3a — Apply `score_mod` to S  (line 1318)

```python
if const_expr(self.score_mod is not None):
    score_mod_fn(acc_S, m_block=m_block)   # in-place modification of acc_S
```

Applies the forward score modification (same function as forward). This recomputes the
modified scores that were seen during the forward pass.

#### Step 3b — Apply mask to S  (line 1322)

```python
if cutlass.const_expr(mask_fn is not None):
    mask_fn(acc_S, m_block=m_block)
```

Same three-path dispatch as forward: seqlen-only R2P, mask_mod per-element, or
causal/local with row-dependent column limits and R2P bit manipulation.

#### Step 4 — Recompute P = exp2(S·log2e − LSE)  (line 1323)

```python
acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S, transpose=self.SdP_swapAB)
for r in cutlass.range_constexpr(cute.size(acc_S_mn, mode=[0])):
    for c in cutlass.range(cute.size(acc_S_mn, mode=[1]), unroll_full=True):
        acc_S_mn[r, c] = cute.math.exp2(
            acc_S_mn[r, c] * softmax_scale_log2 - tLSErLSE[r], fastmath=True
        )
tLSErdPsum = copy_utils.load_s2r(tLSEsdPsum[None, smem_idx_dO])  # load delta values
```

`reshape_acc_to_mn` reshapes the 1-D register fragment to (rows, cols) indexed by
thread position. `tLSErLSE[r]` is the pre-loaded LSE scalar for row `r`.

The attention weights `P_ij = softmax(S_ij)` are recomputed from scratch — no need
to store them from the forward pass. This is the key memory-bandwidth reduction of Flash Attention.

#### Step 5 — Convert P: fp32 → fp16  (line 1332)

```python
tdVrP = utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_S), self.dtype)
```

Converts the register accumulator (fp32 P) to fp16/bf16 for GEMM 3 (dV += Pᵀ @ dO).
When `mma_dkv_is_rs=True`, this stays in registers as `tCrA`.

When `mma_dkv_is_rs=False`, P is written to `sP` smem:
```python
tPrP = smem_thr_copy_PdS.retile(tdVrP)
cute.copy(smem_thr_copy_PdS, tPrP, tPsP[None, None, None, smem_idx_PdS])
```

#### Step 6 — dS = P · (dP − dPsum)  (line 1343)

```python
warpgroup.wait_group(0)   # wait for GEMM 2 (acc_dP) to complete
acc_dP_mn = layout_utils.reshape_acc_to_mn(acc_dP, transpose=self.SdP_swapAB)
for r in cutlass.range_constexpr(cute.size(acc_dP_mn, mode=[0])):
    for c in cutlass.range(cute.size(acc_dP_mn, mode=[1]), unroll_full=True):
        acc_dP_mn[r, c] = acc_S_mn[r, c] * (acc_dP_mn[r, c] - tLSErdPsum[r])
```

This in-place operation overwrites `acc_dP` with `dS`. The formula is the backward
through the softmax normalization: `dS = P * (dP - delta)` where `delta = Σ P_ij dP_ij`.

#### Apply `score_mod_bwd` to dS  (line 1348)

```python
if const_expr(self.score_mod_bwd is not None):
    score_mod_bwd_fn(acc_dP, acc_S_pre, m_block=m_block)
```

See §6 for detailed explanation. `acc_dP` now holds `dS` (overwritten in-place);
`acc_S_pre` holds the original `S = Q @ Kᵀ` before any modification.

#### Step 7 — Convert dS: fp32 → fp16 + write to smem  (line 1352)

```python
tdKrdS = utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_dP), self.dtype)
# barrier to ensure previous P read is done (if mma_dkv_is_rs, guards P was consumed)
cute.arch.fence_view_async_shared()
PdS_barrier.arrive_and_wait()
# write dS to smem sdS
tdSrdS = smem_thr_copy_PdS.retile(tdKrdS)
cute.copy(smem_thr_copy_PdS, tdSrdS, tdSsdS[None, None, None, smem_idx_PdS])
```

`PdS_barrier` (a `NamedBarrier`) synchronizes all MMA threads before writing `dS`
to smem, since both WGs share `sdS`.

#### Step 8 — GEMM 3: dV += Pᵀ @ dO  (line 1368)

```python
if const_expr(not self.mma_dkv_is_rs):
    # P read from smem sP
    mma_pdo_fn(A_idx=smem_idx_PdS, B_idx=smem_idx_dO, zero_init=not dKV_accumulate, wg_wait=-1)
else:
    # P (as tdVrP) comes from registers
    mma_pdo_fn(tCrA=tdVrP, B_idx=smem_idx_dO, zero_init=not dKV_accumulate, wg_wait=-1)
```

`zero_init=not dKV_accumulate`: first m_block zeros `acc_dV`; subsequent blocks
accumulate into the existing register contents.

`wg_wait=-1`: don't wait for completion, allowing overlap with the following GEMM 4.

#### Step 9 — GEMM 4: dQ = dS @ K  (line 1380)

```python
cute.arch.fence_view_async_shared()   # ensure sdS is visible to WGMMA
PdS_barrier.arrive_and_wait()         # sync before reading sdS
acc_dQ = mma_dsk_fn(A_idx=smem_idx_PdS, wg_wait=1)  # waits for GEMM 3 to complete
pipeline_dO.consumer_release(consumer_state_dO_cur)   # release dO slot (dV done with it)
```

`gemm_zero_init` always starts `acc_dQ` from zero (each m_block produces a fresh dQ tile).
The `wg_wait=1` waits for GEMM 3 (`acc_dV`) to complete.

#### Step 10 — GEMM 5: dK += dSᵀ @ Q  (line 1385)

```python
if const_expr(not self.mma_dkv_is_rs):
    mma_dsq_fn(A_idx=smem_idx_PdS, B_idx=smem_idx_Q, zero_init=not dKV_accumulate, wg_wait=1)
else:
    mma_dsq_fn(tCrA=tdKrdS, B_idx=smem_idx_Q, zero_init=not dKV_accumulate, wg_wait=1)
```

When `mma_dkv_is_rs=True`, `tCrA=tdKrdS` (the fp16 dS in registers) is used directly
as the A operand — no smem read for A required.

#### Step 11 — Write dQ to smem + TMA reduce (line 1393)

```python
# Sync: ensure the dQaccum_store warp has finished the previous TMA reduce
cute.arch.barrier(
    barrier_id=int(NamedBarrierBwd.dQEmptyWG0) + warp_group_idx,
    number_of_threads=num_threads_per_warp_group + WARP_SIZE,  # MMA WG + dQaccum warp
)
# R2S: copy dQ register fragment to smem staging area sdQaccum
tdQrdQaccum_flat = cute.make_tensor(acc_dQ.iterator, cute.make_layout(tdQsdQaccum.shape))
cute.autovec_copy(tdQrdQaccum_flat, tdQsdQaccum)
# Signal: dQ is in smem, ready for TMA reduce
cute.arch.fence_view_async_shared()
cute.arch.barrier_arrive(
    barrier_id=int(NamedBarrierBwd.dQFullWG0) + warp_group_idx,
    number_of_threads=num_threads_per_warp_group + WARP_SIZE,
)
```

The MMA WG writes `acc_dQ` (registers → `sdQaccum` smem), then signals the
`dQaccum_store` warp (warp 1) to issue a `cp.reduce.async.bulk.global.add.f32`
TMA instruction. This atomically accumulates the fp32 tile into global `dQaccum`.

```python
warpgroup.wait_group(0)         # wait for GEMM 5 (acc_dK) to complete
pipeline_Q.consumer_release(consumer_state_Q)  # release Q slot
consumer_state_Q.advance()
consumer_state_dO.advance()
return consumer_state_Q, consumer_state_dO
```

### 5.7 `dQaccum_store` — dQ Atomic Writer Warp

**File:** [flash_bwd_sm90.py#L1561](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py#L1561)

Warp 1 (the second warp in the producer warpgroup) runs a matching tile-scheduler loop
and waits for the MMA warpgroups to fill `sdQaccum`, then issues TMA bulk-add:

```python
for warp_group_idx in range(num_mma_warp_groups):
    # Wait for previous TMA reduce to complete (frees smem for next write)
    cute.arch.cp_async_bulk_wait_group(num_mma_warp_groups - 1 - warp_group_idx, read=True)
    # Signal MMA WG: smem is empty, you can write next dQ
    cute.arch.barrier_arrive(NamedBarrierBwd.dQEmptyWG0 + warp_group_idx, ...)

for warp_group_idx in range(num_mma_warp_groups):
    # Wait for MMA WG to fill sdQaccum with fresh dQ
    cute.arch.barrier(NamedBarrierBwd.dQFullWG0 + warp_group_idx, ...)
    with cute.arch.elect_one():
        # TMA bulk-add: smem → global, accumulate into dq_accum
        copy_utils.cpasync_reduce_bulk_add_f32(
            sdQaccum[None, warp_group_idx].iterator,    # source: smem
            gdQaccum[None, warp_group_idx, m_block].iterator,  # dest: global dQaccum
            tma_copy_bytes["dQ"],                        # bytes to transfer
        )
    cute.arch.cp_async_bulk_commit_group()
```

This implements `dQaccum[m_block] += dQ_tile` atomically via PTX
`cp.reduce.async.bulk.global.add.f32`. Only one elected thread per warpgroup issues
the TMA; it is a bulk operation (covers the entire tile in one instruction).

The ping-pong between `dQEmpty` and `dQFull` named barriers serializes the
MMA→smem and smem→global steps without a global `__syncthreads`.

### 5.8 `epilogue_dKV` — dK/dV Write-Back

**File:** [flash_bwd_sm90.py#L1414](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py#L1414)

Called once per n_block after all m_blocks are processed.

#### MHA path (`qhead_per_kvhead == 1`)

```python
# Scale dK by softmax_scale (applied once after all accumulation)
acc_dK.store(acc_dK.load() * softmax_scale)   # line 1219, in mma()

# Convert fp32 → fp16
rdV = cute.make_fragment_like(acc_dV, dtype);  rdV.store(acc_dV.load().to(dtype))
rdK = utils.cvt_f16(acc_dK, dtype)

# STMATRIX: register → smem (for both warpgroups simultaneously)
smem_copy_atom_dKV = StMatrix8x8x16bOp(transpose=dKV_swapAB, num_matrices=4)
cute.copy(smem_copy_atom_dKV, taccdVrdV, taccdVsdV)   # rdV → sV (reuse sV buffer)
cute.arch.fence_view_async_shared()
# barrier: all WGs sync before TMA store
if warp_idx == 4:
    store_dV()    # TMA s2g: sV → global dV
    store_dK()    # TMA s2g: sK → global dK
    cute.arch.cp_async_bulk_commit_group()
    cute.arch.cp_async_bulk_wait_group(0, read=True)
```

The dK/dV buffers reuse the sK/sV smem (which is no longer needed after all m_blocks).
Only one elected warp (warp_idx==4, the first MMA warp) issues the TMA stores.

#### GQA path (`qhead_per_kvhead > 1`)

When there are multiple Q heads per KV head, multiple thread blocks (one per Q head)
all contribute gradients to the same K/V position. Accumulation is done in fp32:

```python
# Recast sV smem as fp32 staging area (sdKVaccum)
sdKVaccum = cute.make_tensor(cute.recast_ptr(sV.iterator, dtype=Float32), sdKVaccum_layout)

# R2S: acc_dK → sdKVaccum  (fp32 register → smem)
cute.autovec_copy(tdKrdKaccum_flat, tdKsdKVaccum)
cute.arch.fence_view_async_shared()

if warp_idx == 4:
    with cute.arch.elect_one():
        for wg_idx in range(num_mma_warp_groups):
            # TMA atomic add: smem → global dk_accum
            copy_utils.cpasync_reduce_bulk_add_f32(
                sdKVaccum[None, wg_idx].iterator,
                gdKaccum[None, wg_idx].iterator,
                tma_copy_bytes["dKacc"] // num_mma_warp_groups,
            )
# Similarly for acc_dV → dv_accum
```

The GQA postprocess kernel then converts the fp32 `dk_accum / dv_accum` to fp16.

---

## 6. score_mod and mask_mod — Forward vs Backward Differences

This section contrasts precisely how `score_mod`, `score_mod_bwd`, and `mask_mod` each behave
in the backward kernel versus the forward kernel, and traces their effect on each of the three
gradient tensors (dV, dQ, dK).

### 6.1 Execution Context

**Forward pass** (one block per Q tile, iterating over K/V tiles):
```
for each Q tile (m_block):
    for each KV tile (n_block):
        S = Q_m @ K_nᵀ
        score_mod(S)         ← modify logits before softmax
        mask(S)              ← zero-out forbidden positions
        P = softmax(S)
        O_m += P @ V_n       ← accumulate output
```

**Backward pass** (one block per K/V tile, iterating over Q tiles):
```
for each KV tile (n_block):
    K_n, V_n loaded once
    for each Q tile (m_block):
        S = Q_m @ K_nᵀ       ← recompute (no stored P)
        acc_S_pre = copy(S)  ← save raw logits (if score_mod_bwd)
        score_mod(S)         ← same forward modification (reapplied)
        mask(S)              ← same forward mask (reapplied)
        P = exp2(S - LSE)    ← recomputed attention weights
        dS = P * (dP - dPsum)
        score_mod_bwd(dS, acc_S_pre)  ← VJP of score_mod
        dV_n += Pᵀ @ dO_m
        dQ_m += dS @ K_n     ← atomic-add
        dK_n += dSᵀ @ Q_m
```

The key structural difference: the forward loops `Q→KV` while the backward loops `KV→Q`.
`score_mod` and `mask_mod` are re-applied to the **recomputed** S every iteration —
there is no stored `P` or `S` tensor from the forward pass.

---

### 6.2 score_mod: Forward vs Backward

#### Forward (`apply_score_mod`)

**File:** [flash_fwd.py — `apply_score_mod`](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py)
**Called:** once per (m_block, n_block) pair, on `acc_S` in-place
**Input:** raw `S = Q @ Kᵀ`  (fp32 register fragment)
**Output:** `score_mod(S * softmax_scale, b, h, q_idx, kv_idx)` written back to `acc_S`
**When:** immediately after GEMM 1, before masking, before softmax

The forward bakes `softmax_scale` into the score before calling the user function:
```python
score_vec[j] = score_tensor[i + j] * softmax_scale   # scale then pass to user
post_mod = score_mod(score_ssa, batch, head, q_idx, kv_idx, ...)
score_tensor[i + j] = post_mod                        # write back
```

This means the user function sees the fully-scaled logit: `s_scaled = S[i,j] / sqrt(d)`.

#### Backward (`apply_score_mod` + `apply_score_mod_bwd`)

**Forward re-application:** same `apply_score_mod` is called in the backward, for the same
reason: P must be recomputed from the *modified* scores exactly as they were in the forward.
Without this, `exp2(S_modified - LSE)` would produce incorrect P values.

**Backward VJP:** `apply_score_mod_bwd` is called *after* `dS = P*(dP-dPsum)` is computed.

**File:** [softmax.py#L472](../thirdparty/flash-attention/flash_attn/cute/softmax.py#L472)
**Called:** once per (m_block, n_block), on `acc_dP` (now containing dS) in-place
**Inputs:**
- `grad_tensor` = `acc_dP` = dS (after backward-softmax); overwritten in-place
- `score_tensor` = `acc_S_pre` = raw `S = Q @ Kᵀ` before any modification
**Output:** `score_mod_bwd(dS, S*scale, b, h, q_idx, kv_idx)` written back to `acc_dP`

```python
for i in range(0, n_vals, vec_size):
    for j in range(vec_size):
        grad_vec[j]  = grad_tensor[i+j]               # current dS (in-place input)
        score_vec[j] = score_tensor[i+j] * softmax_scale  # raw QK^T, scaled

    new_grad = score_mod_bwd(grad_ssa, score_ssa, batch, head, q_idx_ssa, kv_idx_ssa, ...)
    for j in range(vec_size):
        grad_tensor[i+j] = new_grad[j]                # overwrite dS with modified gradient
```

The `score_tensor` argument receives `acc_S_pre * softmax_scale` — i.e. the same value the
user function saw in the forward pass (`S_scaled = S / sqrt(d)`). This allows `score_mod_bwd`
to be the exact chain-rule VJP: `d(loss)/d(S) = d(loss)/d(f(S)) * f'(S)` where `f'(S)` can
depend on `S` itself (e.g. for tanh softcap: `f'(S) = 1 - tanh²(S/c)`).

**Why `acc_S_pre` and not just `acc_S`?**

After `score_mod(acc_S)` modifies S in-place and the mask writes `-inf` to forbidden positions,
the original raw logits are gone. `acc_S_pre` is a register copy made *before* those operations:

```python
# flash_bwd_sm90.py line 1313
if const_expr(self.score_mod_bwd is not None):
    acc_S_pre = cute.make_fragment_like(acc_S)
    cute.autovec_copy(acc_S, acc_S_pre)    # save NOW, before score_mod or mask mutate acc_S
```

`score_mod_bwd` receives the pre-modification logits, not the masked/modified ones.
If `score_mod_bwd` received modified logits, the VJP would be incorrect for any
`score_mod` whose derivative depends on the input (all nonlinear score_mods).

**Concrete examples of the difference:**

| score_mod | Forward effect | Backward VJP |
|---|---|---|
| `s * scale` (alibi, bias) | `S → S + bias` | `dS_final = dS` (bias has no derivative w.r.t. S; scale already applied) |
| `tanh(s / c) * c` (softcap) | `S → S_cap` | `dS_final = dS * (1 - tanh²(S_raw/c))` — needs `S_raw` |
| `s + relative_pos_bias(q,k)` | `S → S + bias` | `dS_final = dS` (additive bias, derivative = 1) |
| `s * learnable_temp` | `S → S * T` | `dS_final = dS * T`; also `dT = Σ S * dS` (aux tensor grad) |

For the additive-only case (alibi, relative position bias), `score_mod_bwd` is the identity on
`dS` — it just passes through. For softcap, the backward must know `S_raw` to compute `sech²`.

---

### 6.3 mask_mod: Forward vs Backward

#### Forward

**File:** [mask.py — `AttentionMask.apply_mask`](../thirdparty/flash-attention/flash_attn/cute/mask.py#L127)
**Effect:** writes `-inf` to forbidden positions in `acc_S`
**When:** after `score_mod`, before `softmax`

Three dispatch paths (same as forward — see §6 of `flash_attn_fwd.md`):
- Path A: seqlen-only → R2P bit manipulation (`mask_r2p`)
- Path B: `mask_mod` per-element → register loop calling user function
- Path C: causal/local → row-dependent `col_limit` then R2P

#### Backward

**Same three paths, same function, same call site:**

```python
# flash_bwd_sm90.py line 1171–1183
mask_fn = partial(
    mask.apply_mask,
    batch_idx=batch_idx, head_idx=head_idx, n_block=n_block,
    thr_mma=thr_mma_SdP,
    mask_seqlen=True,
    mask_causal=self.is_causal,
    mask_mod=self.mask_mod,    # same callable as forward
    ...
)
# called in mma_one_m_block line 1322
mask_fn(acc_S, m_block=m_block)
```

The mask is applied to the **recomputed** `acc_S` in the backward, producing `-inf`
at the same positions that were masked in the forward.

#### Why no separate `mask_mod_bwd` is needed

The mask is **not differentiable** — it is a hard zeroing of attention weights. Positions
that were masked in the forward have `P_ij = softmax(-inf) = 0`. When `P_ij = 0`:

```
dS_ij = P_ij * (dP_ij - dPsum_i)
       = 0   * (...)
       = 0
```

The backward-softmax formula naturally produces `dS = 0` for any masked position, without any
special handling. No `-inf` propagation, no branch, no `mask_mod_bwd` callable — the
mathematical property of softmax automatically kills the gradient at masked positions.

The only purpose of re-applying the mask in the backward is to ensure P is **recomputed correctly**:
the masked positions must receive `-inf` before `exp2` so that P = 0 there, which then drives
dS = 0 through the formula above. If the mask were omitted in the backward, `exp2(S - LSE)`
would produce a small-but-nonzero P at masked positions, yielding spurious gradients.

#### Effect on each gradient

| Gradient | Effect of mask |
|---|---|
| dV | `dV_n += Pᵀ @ dO_m` — masked rows of P are 0, so dV receives no contribution from masked Q positions |
| dQ | `dQ_m += dS @ K_n` — dS is 0 for masked positions, so those K rows contribute 0 to dQ |
| dK | `dK_n += dSᵀ @ Q_m` — same: dS = 0 at masked positions, no Q contribution to dK |

In all three cases the mask's effect propagates through `dS`. Crucially, **there is no
separate backward path for the mask** — it enters dV, dQ, and dK identically, via dS.

---

### 6.4 Gradient Flow Through dV, dQ, dK

The diagram below shows which operations each gradient depends on and where
`score_mod_bwd` / `mask_mod` enter the computation for each gradient:

```
  GEMM 1: S = Q @ Kᵀ
     │
     ├─ [save acc_S_pre]  ← needed by score_mod_bwd
     │
     ↓ score_mod(S)       ← same as forward, exact recomputation
     ↓ mask(S)            ← same as forward, ensures P = 0 at masked positions
     │
  P = exp2(S·log2e - LSE)
     │
     │   GEMM 2: dP = dO @ Vᵀ
     │        │
  dS = P * (dP - dPsum)
     │
     ↓ score_mod_bwd(dS, acc_S_pre)   ← apply VJP of score_mod to dS
     │
     dS_final  (fp16, written to sdS smem)
     │
     ├──────────────────────────────────────────────────────────────┐
     │                                                              │
  GEMM 3:                                                     GEMM 5:
  dV_n += Pᵀ @ dO_m                                          dK_n += dSᵀ @ Q_m
  (uses P before score_mod_bwd,                               (uses dS_final after
   score_mod_bwd does NOT affect dV)                           score_mod_bwd)
     │                                                              │
     └──────────────────────┬───────────────────────────────────────┘
                            │
                       GEMM 4:
                       dQ_m += dS_final @ K_n
                       (uses dS_final after score_mod_bwd)
```

**Critical asymmetry: dV does not see `score_mod_bwd`.**

`dV_n += Pᵀ @ dO_m` uses the recomputed attention weights `P`, **not** `dS_final`.
P is computed *before* `score_mod_bwd` is applied (P comes from `exp2(S_modified - LSE)`).
`score_mod_bwd` only modifies `acc_dP` (which holds dS), never `acc_S` (which holds P).

Therefore:
- **dV**: depends on `score_mod` (via recomputed P) and `mask_mod` (P=0 at masked positions), but **not** on `score_mod_bwd`
- **dQ** and **dK**: depend on `score_mod` (via P in dS), `score_mod_bwd` (VJP applied to dS), and `mask_mod` (dS=0 at masked positions)

This is mathematically correct: `dV = ∂L/∂V = Pᵀ @ dO` does not depend on `score_mod_bwd`
because V is downstream of softmax, not of the score modification. The score_mod_bwd
correction is needed only when differentiating with respect to the *pre-softmax scores*,
which enter dQ and dK (as `dS @ K` and `dSᵀ @ Q` respectively).

**Code confirmation** — ordering in `mma_one_m_block`:
```python
# Step 4: P = exp2(S_modified - LSE)    [acc_S_mn computed]
# Step 5: P → fp16 as tdVrP             [converted for GEMM 3]
# Step 6: dS = P * (dP - dPsum)         [acc_dP overwritten with dS]
# Step 7: score_mod_bwd(dS, acc_S_pre)  [acc_dP = dS_final]
# Step 8: GEMM 3 dV += Pᵀ @ dO         [uses tdVrP = P BEFORE score_mod_bwd]
# Step 9: GEMM 4 dQ += dS @ K          [reads sdS = dS_final]
# Step 10:GEMM 5 dK += dSᵀ @ Q         [reads sdS = dS_final]
```

P (as `tdVrP`) is converted to fp16 in Step 5, before `score_mod_bwd` runs in Step 7.
The GEMM 3 operand is therefore the pre-VJP P, which is correct.

---

### 6.5 SdP_swapAB and Transposed Index Coordinates

In the backward kernel, `SdP_swapAB=True` for SM90. This swaps the A and B operands
of the `S = Q @ Kᵀ` and `dP = dO @ Vᵀ` GEMMs, effectively computing `Kᵀ @ Q` and
`Vᵀ @ dO` instead (same mathematical result, different smem layout).

When swapped, the register fragment `acc_S` stores elements in **(kv, q) order**
instead of (q, kv) order. This affects index extraction in both `apply_score_mod`
and `apply_score_mod_bwd_inner`:

```python
# softmax.py line 382
if cutlass.const_expr(transpose_indices):    # transpose_indices = SdP_swapAB
    q_idx_pos  = cutlass.const_expr(1)       # q_idx is at position 1 in the pair
    kv_idx_pos = cutlass.const_expr(0)       # kv_idx is at position 0
else:
    q_idx_pos  = cutlass.const_expr(0)
    kv_idx_pos = cutlass.const_expr(1)
```

Both `apply_score_mod` (forward recomputation) and `apply_score_mod_bwd_inner` (VJP)
receive `transpose_indices=self.SdP_swapAB` so that `q_idx` and `kv_idx` are always
passed to the user callable in the correct logical order — independent of the physical
tile layout. The user's `score_mod(s, b, h, q_idx, kv_idx)` always sees `q_idx < seqlen_q`
and `kv_idx < seqlen_k`.

The `AttentionMask.apply_mask` is also passed `swap_AB=self.SdP_swapAB` at construction
time, so the causal/local column limit calculation accounts for the transposed tile layout.

---

### 6.6 Compile-time Embedding and Constraints

Both `score_mod`, `score_mod_bwd`, and `mask_mod` are stored as `cutlass.Constexpr`
fields on `FlashAttentionBackwardSm90` and baked into the JIT-compiled CUBIN:

```python
# flash_bwd_sm90.py line 61
score_mod:     cutlass.Constexpr | None = None
score_mod_bwd: cutlass.Constexpr | None = None
mask_mod:      cutlass.Constexpr | None = None
```

The compile cache key includes all three hashes (`score_mod_hash`, `score_mod_bwd_hash`,
`mask_mod_hash`). Different callables produce different kernels. This means:
- The `if score_mod is not None` branches are resolved at JIT time — no runtime branch
- `score_mod` and `score_mod_bwd` must be passed **as a pair** — the launcher asserts
  `score_mod_bwd is not None` whenever `score_mod is not None`
- There is no `mask_mod_bwd` argument (see §6.3 — mask is not differentiable)
- `softcap` and `score_mod` are **mutually exclusive** (both are forms of logit modification
  but use different `softmax_scale_log2` encodings)
- `score_mod` + varlen is not yet supported in backward (asserted in `_flash_attn_bwd`)

**`vec_size`** (elements processed per `score_mod` / `score_mod_bwd` call) is also
baked at compile time:
- `vec_size = 4` when there are no aux_tensors (default: SIMD-vectorized calls)
- `vec_size = 1` when aux_tensors are present (each element needs independent index wrapping)

Both the forward VJP (`apply_score_mod_inner`) and backward VJP (`apply_score_mod_bwd_inner`)
use the same `vec_size` decision.

---

## 7. VJP Derivation — How `score_mod_bwd` is Created

When the user writes:

```python
out = flex_attention(Q, K, V, score_mod=f)
out.backward(dout)
```

`score_mod_bwd` — the VJP of `f` — is **not hand-written**. It is automatically derived by the
PyTorch compiler stack in four stages:

```
flex_attention_autograd  →  create_fw_bw_graph / create_joint / make_fx
  → FlexAttentionAutogradOp.backward / flex_attention_backward HOP
  → Inductor lowering (build_subgraph_buffer)
  → Jinja template rendering (flash_attention_backward.py.jinja)
  → @cute.jit score_mod_bwd  →  _flash_attn_bwd
```

---

### 7.1 Stage 1 — Autograd dispatch: `create_fw_bw_graph`

**File:** [torch/_higher_order_ops/flex_attention.py#L891](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L891)

The first time `flex_attention` is called with `requires_grad=True` inputs, it hits the
`DispatchKey.Autograd` kernel `flex_attention_autograd`:

```python
# flex_attention.py:891
def flex_attention_autograd(..., score_mod, ...):
    if any(input_requires_grad):
        fw_graph, joint_graph = create_fw_bw_graph(
            score_mod, example_vals, other_buffers
        )
```

`create_fw_bw_graph` ([line 644](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L644))
wraps `score_mod` and **symbolically differentiates it**:

```python
# flex_attention.py:644
def create_fw_bw_graph(score_mod, index_values, other_buffers):
    # 1. Wrap: surface requires_grad flag as a second return value
    def fw_with_masks(*args):
        fw_out = score_mod(*args)
        return ((fw_out,), (fw_out.requires_grad,))

    # 2. Derive VJP using AOT Autograd's create_joint
    joint = create_joint(fw_with_masks, aot_config=dummy_aot_config)

    # 3. Trace joint into an FX GraphModule via make_fx (fake-tensor mode)
    joint_graph = make_fx(joint_f)(
        *unwrapped_score_mod_indexes, example_grad, *unwrapped_other_buffers
    )
    return score_mod, joint_graph
```

`joint_graph` is an `fx.GraphModule` whose nodes encode **both** the forward `score_mod` **and**
its gradient computation (the VJP).

`fw_graph` and `joint_graph` are stored on `ctx` in
[`FlexAttentionAutogradOp.forward`](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L750)
and retrieved during
[`FlexAttentionAutogradOp.backward`](../thirdparty/pytorch/torch/_higher_order_ops/flex_attention.py#L809).

---

### 7.2 Stage 2 — `create_joint`: symbolic differentiation

**File:** [torch/_functorch/_aot_autograd/graph_capture_wrappers.py#L292](../thirdparty/pytorch/torch/_functorch/_aot_autograd/graph_capture_wrappers.py#L292)

```python
# graph_capture_wrappers.py:292
def create_joint(fn, *, aot_config):
    def inner(primals, tangents):
        outs, _ = fn(*primals)          # forward pass (under fake-tensor tracing)
        grad_primals = [p for p in primals if p.requires_grad]
        # torch.autograd.grad captures gradient nodes into the FX graph
        grads = torch.autograd.grad(
            outs_to_grad, grad_primals, grad_outputs=tangents,
            allow_unused=True, create_graph=False,
        )
        return outs, grads
    return inner
```

When `make_fx(joint_f)(...)` runs this function in **fake-tensor tracing** mode,
`torch.autograd.grad()` is evaluated symbolically: each chain-rule step for the specific
`score_mod` expression becomes a node in the FX graph.

- Forward nodes get `node.meta["partitioner_tag"] = "is_forward"`.
- Backward nodes get `"is_backward"`.

**Concrete example** — if `score_mod = lambda s, b, h, q, k: s + bias[b, h, q, k]`:

| Graph node | Operation | Tag |
|---|---|---|
| `add(s, bias[b,h,q,k])` | forward | `is_forward` |
| `grad_output * 1` | ∂(add)/∂s = 1 | `is_backward` |

The VJP of an `add` is just `grad_output`; for `softcap` it would be
`grad_output * (1 - tanh²(s/cap)) / cap`; all of this is unrolled symbolically and captured as FX nodes.

---

### 7.3 Stage 3 — Inductor lowering: subgraph buffers

**File:** [torch/_inductor/kernel/flex/flex_attention.py#L616](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_attention.py#L616)

During backward, `flex_attention_backward` HOP lowering receives `fw_graph` and `joint_graph`:

```python
# flex_attention.py:616
def flex_attention_backward(..., fw_graph, joint_graph, ...):
    fw_subgraph_buffer     = build_subgraph_buffer(args, fw_graph)     # subgraph 0
    all_joint_outputs      = build_subgraph_buffer(args, joint_graph)  # subgraph 1
    joint_outputs = process_joint_outputs(all_joint_outputs)
    # joint_outputs.grad_input    = dS node (grad of score_mod output w.r.t. score input)
    # joint_outputs.captured_grads = grads w.r.t. aux tensors (bias, rope, etc.)

    create_flex_flash_attention_backward_kernel(
        ...,
        fw_subgraph_buffer,
        joint_outputs.grad_input,   # the VJP subgraph buffer
    )
```

**File:** [torch/_inductor/kernel/flex/flex_flash_attention.py#L477](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_flash_attention.py#L477)

```python
# flex_flash_attention.py:477
def create_flex_flash_attention_backward_kernel(
    ..., fw_subgraph_buffer, joint_subgraph_buffer, ...
):
    # subgraphs list: index 0 = forward, index 1 = VJP, index 2 = mask
    subgraphs = [fw_subgraph_buffer, joint_subgraph_buffer, mask_graph_buffer]
    flash_attention_backward_cutedsl_template.maybe_append_choice(
        ..., subgraphs=subgraphs, HAS_SCORE_MOD=True
    )
```

---

### 7.4 Stage 4 — Jinja template: rendering `score_mod_bwd`

**File:** [torch/_inductor/kernel/flex/templates/flash_attention_backward.py.jinja](../thirdparty/pytorch/torch/_inductor/kernel/flex/templates/flash_attention_backward.py.jinja)

The template renders two separate `@cute.jit` functions:

```jinja
{%- if HAS_SCORE_MOD %}
@cute.jit
def score_mod(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
    # Inlined from fw_subgraph_buffer (subgraph_number=0)
    {{ modification(subgraph_number=0, output_name="tSrS_ssa", ...) }}
    return tSrS_ssa

@cute.jit
def score_mod_bwd(grad_score_mod_ssa, tSrS_ssa, b_idx, h_idx, q_idx, kv_idx,
                  seqlen_info, aux_tensors):
    # Inlined from joint_subgraph_buffer (subgraph_number=1) — the VJP
    {{ modification(subgraph_number=1, output_name="grad_score_mod_ssa_out", ...) }}
    return grad_score_mod_ssa_out
{%- endif %}

_flash_attn_bwd(
    ...,
    score_mod=score_mod,
    score_mod_bwd=score_mod_bwd,
    mask_mod=mask_mod,
)
```

`modification(subgraph_number=1, ...)` emits the lowered IR of `joint_graph` as inline Python
statements inside the `@cute.jit` body. CuTeDSL then JIT-compiles these statements to CUDA PTX,
and the resulting code is inlined at step 7 of `mma_one_m_block` (the `apply_score_mod_bwd_inner`
call in `softmax.py`).

---

### 7.5 End-to-end call chain

```
User code
  flex_attention(Q, K, V, score_mod=f).backward(dout)
  │
  ├─ DispatchKey.Autograd  →  flex_attention_autograd  (flex_attention.py:891)
  │    └─ create_fw_bw_graph(f, ...)  (flex_attention.py:644)
  │         ├─ create_joint(fw_with_masks)          # AOT-autograd VJP wrapper
  │         │    └─ torch.autograd.grad(...)         # symbolic differentiation
  │         └─ make_fx(joint_f)(...)                 # → joint_graph FX GraphModule
  │
  ├─ FlexAttentionAutogradOp.forward  (flex_attention.py:750)
  │    └─ ctx.fw_graph = fw_graph,  ctx.joint_graph = joint_graph
  │
  ├─ FlexAttentionAutogradOp.backward  (flex_attention.py:809)
  │    └─ flex_attention_backward HOP(fw_graph, joint_graph, ...)
  │
  ├─ Inductor lowering  (flex_attention.py:616)
  │    ├─ build_subgraph_buffer(fw_graph)    → fw_subgraph_buffer    (subgraph 0)
  │    └─ build_subgraph_buffer(joint_graph) → joint_subgraph_buffer (subgraph 1)
  │
  ├─ create_flex_flash_attention_backward_kernel(...)  (flex_flash_attention.py:477)
  │    └─ subgraphs = [fw_subgraph_buffer, joint_subgraph_buffer, mask_graph_buffer]
  │
  ├─ flash_attention_backward.py.jinja  (template rendering)
  │    ├─ @cute.jit score_mod      ← modification(subgraph_number=0)
  │    └─ @cute.jit score_mod_bwd  ← modification(subgraph_number=1)  ← VJP
  │
  └─ _flash_attn_bwd(..., score_mod=score_mod, score_mod_bwd=score_mod_bwd)
       └─ mma_one_m_block  step 7:  apply_score_mod_bwd_inner(score_mod_bwd, ...)
            └─ score_mod_bwd(dS_ij, S_ij, b, h, q_idx, kv_idx)  [per register tile]
```

**Key invariants:**
- `score_mod_bwd` receives `(grad_output=dS_ij, score=acc_S_pre_ij, b, h, q_idx, kv_idx)`.
  `acc_S_pre` is the **raw** QKᵀ logit saved before `score_mod` mutated `acc_S` (step 3 of
  `mma_one_m_block`); it is the primal input needed by the VJP.
- The function returns the VJP-transformed `dS_ij` (i.e., `dS_ij * f'(S_ij)` for
  element-wise `score_mod`).
- `dV` is computed from `P` **before** this step (step 6), so dV does **not** see `score_mod_bwd`.
  dQ and dK (steps 8–9) use the VJP-transformed `dS`, so they do.

---

## 8. Phase 3 — Postprocess Kernels

**File:** [interface.py#L1124](../thirdparty/flash-attention/flash_attn/cute/interface.py#L1124)
**Class:** `FlashAttentionBackwardPostprocess`

### dQ Postprocess (256 threads)

```python
# Compile key includes: AtomLayoutMdQ, dQ_swapAB
fa_bwd_post = FlashAttentionBackwardPostprocess(
    dtype, head_dim, arch=90, m_block_size, num_threads=256, AtomLayoutMdQ, dQ_swapAB
)
fa_bwd_post(dq_accum, dq, softmax_scale, cu_seqlens_q, seqused_q, stream)
```

Reads fp32 `dq_accum` (atomically accumulated from all n_blocks), multiplies by
`softmax_scale`, and converts to fp16/bf16 `dq`. The scale is applied here because
the main backward kernel accumulates unscaled `dS @ K` — the scale was not applied
to `dK/dV` GEMMs but was baked into `softmax_scale_log2` for the exp2 computation.

Wait — for the MHA path, `dK` is scaled before epilogue: `acc_dK *= softmax_scale` (line 1219).
For `dQ`, the postprocess kernel applies the scale during the fp32→fp16 conversion.

### dK/dV Postprocess (GQA only)

```python
if dKV_postprocess:    # qhead_per_kvhead > 1
    fa_bwd_post = FlashAttentionBackwardPostprocess(
        dtype, head_dim, arch=90, n_block_size, num_threads=256, AtomLayoutNdKV, dKV_swapAB
    )
    fa_bwd_post(dk_accum, dk, softmax_scale, ...)
    # same for dv_accum → dv
```

Converts fp32 accumulated `dk_accum / dv_accum` to fp16/bf16. The fp32 intermediate
is necessary because multiple thread blocks add their contributions atomically.

---

## 9. Thread Layout and Register Budget

```
Thread Block: 384 threads = 3 warpgroups × 4 warps × 32 threads

Warp 0  (producer WG, warp 0):  load()           [setmaxregister 24]
Warp 1  (producer WG, warp 1):  dQaccum_store()  [setmaxregister 24]
Warps 2-3 (producer WG):        idle / barrier ops

Warps 4-7  (consumer WG 0): mma() warp_group_idx=0  [setmaxregister 240]
Warps 8-11 (consumer WG 1): mma() warp_group_idx=1  [setmaxregister 240]
```

The 240-register budget for consumer warpgroups accommodates:
- `acc_S` / `acc_dP` / `acc_dQ`: (tile_m × tile_n / WG / 32-threads) × 4B each
- `acc_dV` / `acc_dK`: (tile_n × hdim / WG / 32-threads) × 4B, persistent across loop
- `tLSErLSE`, `tLSErdPsum`: per-row scalars
- `tdVrP`, `tdKrdS`: fp16 registers for RS GEMMs

The `setmaxregister_decrease(24)` for producer warps forces the compiler to spill
producer-side state to L1 scratch, freeing register file capacity for consumers.

---

## 10. Key Performance Optimizations

| Optimization | Location | Effect |
|---|---|---|
| K/V loaded once per n_block | `load()` line 821 | Saves (m_block_max - 1) × K/V bandwidth |
| `mma_dkv_is_rs=True` | `__init__` line 95 | P and dS skip smem → register for dV/dK GEMMs |
| GEMM 1+2 overlap | `mma_one_m_block` line 1305–1311 | QKᵀ and dOVᵀ in flight simultaneously |
| GEMM 3+4 overlap | line 1374, 1380 | dV and dQ GEMMs pipelined |
| `acc_dV/acc_dK` persistent | `mma()` — not re-zeroed | Accumulate across m_blocks; write once |
| TMA bulk add for dQ | `dQaccum_store` line 1620 | Single instruction replaces atomic loop |
| `exp2` instead of `exp` | line 1326 | ~2× faster: `MUFU.EX2` PTX instruction |
| `shuffle_LSE` / `shuffle_dPsum` | `__init__` line 111 | Reduces register pressure when `SdP_swapAB` |
| Smem reuse for epilogue | `epilogue_dKV` line 1469 | sV / sK reused for dV / dK staging |
| `wg_wait=-1` for overlap | `mma_one_m_block` line 1305 | Launch GEMM 1 without waiting |
| `setmaxregister` asymmetry | `kernel` lines 643/680 | Consumer gets 240 regs, producer 24 |

---

## 11. Key Functions Index

| Function | File | Line | Purpose |
|----------|------|------|---------|
| `_flash_attn_bwd` | interface.py | 554 | Top-level Python launcher |
| `FlashAttentionBackwardPreprocess.__call__` | flash_bwd_preprocess.py | 110 | JIT preprocess dispatch |
| `FlashAttentionBackwardPreprocess.kernel` | flash_bwd_preprocess.py | 188 | GPU preprocess kernel |
| `FlashAttentionBackwardSm90.__init__` | flash_bwd_sm90.py | 41 | Hyperparams, mma_dkv_is_rs |
| `FlashAttentionBackwardSm90.__call__` | flash_bwd_sm90.py | 302 | Host launcher, TMA setup, grid |
| `FlashAttentionBackwardSm90.kernel` | flash_bwd_sm90.py | 517 | GPU entry point, warp dispatch |
| `FlashAttentionBackwardSm90.load` | flash_bwd_sm90.py | 720 | TMA producer warp |
| `FlashAttentionBackwardSm90.mma` | flash_bwd_sm90.py | 966 | Consumer WG setup, GEMM closure creation |
| `FlashAttentionBackwardSm90.mma_one_m_block` | flash_bwd_sm90.py | 1269 | 7-step backward body per Q tile |
| `FlashAttentionBackwardSm90.dQaccum_store` | flash_bwd_sm90.py | 1561 | TMA atomic-add dQ to global |
| `FlashAttentionBackwardSm90.epilogue_dKV` | flash_bwd_sm90.py | 1414 | STMATRIX + TMA store for dK, dV |
| `FlashAttentionBackwardSm90.apply_score_mod` | flash_bwd_sm90.py | 879 | Build coord tensor, call inner |
| `FlashAttentionBackwardSm90.apply_score_mod_bwd` | flash_bwd_sm90.py | 922 | Build coord tensor, call VJP inner |
| `apply_score_mod_bwd_inner` | softmax.py | 472 | Vectorized register-resident VJP loop |

---

## 12. Code Map

| File | Path |
|------|------|
| interface.py | [thirdparty/flash-attention/flash_attn/cute/interface.py](../thirdparty/flash-attention/flash_attn/cute/interface.py) |
| flash_bwd_preprocess.py | [thirdparty/flash-attention/flash_attn/cute/flash_bwd_preprocess.py](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_preprocess.py) |
| flash_bwd_sm90.py | [thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py) |
| softmax.py | [thirdparty/flash-attention/flash_attn/cute/softmax.py](../thirdparty/flash-attention/flash_attn/cute/softmax.py) |
| mask.py | [thirdparty/flash-attention/flash_attn/cute/mask.py](../thirdparty/flash-attention/flash_attn/cute/mask.py) |
| pipeline.py | [thirdparty/flash-attention/flash_attn/cute/pipeline.py](../thirdparty/flash-attention/flash_attn/cute/pipeline.py) |
| tile_scheduler.py | [thirdparty/flash-attention/flash_attn/cute/tile_scheduler.py](../thirdparty/flash-attention/flash_attn/cute/tile_scheduler.py) |

---

*Document produced by Claude (Sonnet 4.5), 2026-02-14. Based on direct source reading of the SM90 Flash Attention CuTeDSL implementation.*
