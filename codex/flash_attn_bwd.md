# FlashAttention CuTeDSL Backward Trace (SM90-focused)

## Scope
This document traces the CuTeDSL backward execution path rooted at:
- [interface.py:_flash_attn_bwd](../thirdparty/flash-attention/flash_attn/cute/interface.py#L554)

Deep trace target:
1. Backward launcher orchestration (`preprocess -> main -> postprocess`).
2. SM90 main kernel internals (`flash_bwd_sm90.py`) frame-by-frame.
3. Hook behavior (`score_mod`, `score_mod_bwd`, `mask_mod`) and how they alter execution.
4. Data layouts, thread-value mapping, synchronization, and performance strategy.

Notes:
- `interface.py` can dispatch to SM80/SM90/SM100. This trace is SM90-focused for kernel internals.
- Preprocess/postprocess kernels are included because they are required for end-to-end bwd flow.

## Code Map
| Area | Link |
|---|---|
| Backward launcher entry | [interface.py](../thirdparty/flash-attention/flash_attn/cute/interface.py#L554) |
| Backward preprocess object | [flash_bwd_preprocess.py](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_preprocess.py#L26) |
| Backward SM90 main object | [flash_bwd_sm90.py](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py#L38) |
| Backward postprocess object | [flash_bwd_postprocess.py](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_postprocess.py#L34) |
| SM90 kernel body | [flash_bwd_sm90.py](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py#L517) |
| SM90 producer path (`load`) | [flash_bwd_sm90.py](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py#L721) |
| SM90 consumer path (`mma`) | [flash_bwd_sm90.py](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py#L967) |
| Per-`m_block` step | [flash_bwd_sm90.py](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py#L1270) |
| dKV epilogue | [flash_bwd_sm90.py](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py#L1415) |
| dQaccum writer | [flash_bwd_sm90.py](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py#L1562) |
| Score-mod fwd hook inner | [softmax.py](../thirdparty/flash-attention/flash_attn/cute/softmax.py#L343) |
| Score-mod bwd hook inner | [softmax.py](../thirdparty/flash-attention/flash_attn/cute/softmax.py#L473) |
| Mask logic (incl. swap_AB) | [mask.py](../thirdparty/flash-attention/flash_attn/cute/mask.py#L128) |
| Sequence metadata | [seqlen_info.py](../thirdparty/flash-attention/flash_attn/cute/seqlen_info.py#L38) |
| Tile bounds | [block_info.py](../thirdparty/flash-attention/flash_attn/cute/block_info.py#L24) |
| Tile scheduler | [tile_scheduler.py](../thirdparty/flash-attention/flash_attn/cute/tile_scheduler.py#L78) |
| Pipeline wrappers | [pipeline.py](../thirdparty/flash-attention/flash_attn/cute/pipeline.py#L101) |
| Block sparse bwd helpers | [block_sparse_utils.py](../thirdparty/flash-attention/flash_attn/cute/block_sparse_utils.py#L1144) |
| Named barriers | [named_barrier.py](../thirdparty/flash-attention/flash_attn/cute/named_barrier.py#L15) |

## Key Functions Index
| Function | File | Purpose |
|---|---|---|
| `_flash_attn_bwd` | [interface.py](../thirdparty/flash-attention/flash_attn/cute/interface.py#L554) | Validates bwd inputs and orchestrates preprocess/main/postprocess kernels. |
| `FlashAttentionBackwardPreprocess.__call__` | [flash_bwd_preprocess.py](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_preprocess.py#L110) | Computes `dPsum`, `lse_log2`, and initializes `dQaccum`. |
| `FlashAttentionBackwardSm90.__call__` | [flash_bwd_sm90.py](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py#L303) | Launch-time setup for SM90 main bwd kernel. |
| `FlashAttentionBackwardSm90.kernel` | [flash_bwd_sm90.py](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py#L517) | CTA role split and producer/consumer orchestration. |
| `FlashAttentionBackwardSm90.load` | [flash_bwd_sm90.py](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py#L721) | Producer loads Q/dO (+stats) and K/V into staged shared memory. |
| `FlashAttentionBackwardSm90.mma` | [flash_bwd_sm90.py](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py#L967) | Consumer computes dV/dK/dQ through GEMM+pointwise sequence. |
| `FlashAttentionBackwardSm90.mma_one_m_block` | [flash_bwd_sm90.py](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py#L1270) | Core per-m-block backward math/update unit. |
| `FlashAttentionBackwardSm90.apply_score_mod_bwd` | [flash_bwd_sm90.py](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py#L923) | Applies user score-mod backward rule over tile. |
| `FlashAttentionBackwardPostprocess.__call__` | [flash_bwd_postprocess.py](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_postprocess.py#L201) | Converts fp32 accumulators to final gradient dtype/scaled form. |

## Big Picture Backward Pipeline
1. `preprocess`: compute `dPsum`, convert LSE to log2 domain, zero/prepare `dQaccum`.
2. `main kernel`: for each KV tile, sweep contributing Q tiles and compute dS/dV/dK/dQaccum.
3. `postprocess`: convert accumulator buffers (`dQaccum`, optionally `dKaccum/dVaccum`) into output dtype tensors.

## Frame-By-Frame Trace

### Frame A: Backward launcher entry and mode setup
Reference: [interface.py#L554-L833](../thirdparty/flash-attention/flash_attn/cute/interface.py#L554)

**Before**
- Inputs: `q,k,v,out,dout,lse` and optional varlen metadata/hooks.

**Action**
- Validates shapes, dtypes, devices, alignment.
- Chooses architecture defaults (`m_block_size`, `n_block_size`, stage counts).
- Disables unsupported combinations (e.g. varlen + score_mod bwd on current path).
- Allocates intermediary tensors:
  - `dQaccum` (fp32)
  - `dPsum` (fp32)
  - `lse_log2` (fp32)
  - optionally `dKaccum`, `dVaccum` for GQA postprocess path.

**After**
- Backward workspace is ready for preprocess/main/postprocess kernels.

Annotated snippet:

```python
# file: thirdparty/flash-attention/flash_attn/cute/interface.py
# link: ../thirdparty/flash-attention/flash_attn/cute/interface.py#L834
# Preprocess kernel prepares dPsum + lse_log2 + zeroes dQaccum
compile_key_pre = (...)
...
# Main kernel computes dK/dV/dQaccum
compile_key = (..., score_mod_hash, score_mod_bwd_hash, mask_mod_hash, ...)
...
# Postprocess converts fp32 accumulators to output gradients
compile_key_post = (...)
```

Performance relevance:
- Staging gradients through fp32 accumulators improves numerical stability and enables reduction-friendly writeback schemes.

---

### Frame B: Preprocess kernel (`FlashAttentionBackwardPreprocess`)
References:
- object: [flash_bwd_preprocess.py#L26](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_preprocess.py#L26)
- launch: [flash_bwd_preprocess.py#L110](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_preprocess.py#L110)
- kernel: [flash_bwd_preprocess.py#L188](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_preprocess.py#L188)

What it does:
1. Loads `O` and `dO` tiles.
2. Computes rowwise dot product `(O * dO).sum(axis=head_dim)` -> `dPsum`.
3. Converts `LSE` to log2-domain buffer `lse_log2`.
4. Initializes/zeros `dQaccum` tile region.

Why needed:
- Main backward inner loop uses `dPsum` and `lse_log2` directly for stable `dS` formation.

---

### Frame C: Main compile key and SM90 object creation
References:
- key/hash: [interface.py#L895-L980](../thirdparty/flash-attention/flash_attn/cute/interface.py#L895)
- object dispatch: [interface.py#L1021-L1045](../thirdparty/flash-attention/flash_attn/cute/interface.py#L1021)
- compile: [interface.py#L1071](../thirdparty/flash-attention/flash_attn/cute/interface.py#L1071)

`score_mod` / `score_mod_bwd` / `mask_mod` are all part of the compile key:

```python
# file: thirdparty/flash-attention/flash_attn/cute/interface.py
# link: ../thirdparty/flash-attention/flash_attn/cute/interface.py#L895
score_mod_hash = utils.hash_callable(score_mod) if score_mod else False
score_mod_bwd_hash = utils.hash_callable(score_mod_bwd) if score_mod_bwd else False
mask_mod_hash = utils.hash_callable(mask_mod) if mask_mod else False
```

Effect:
- Different hook bodies produce specialized binaries.
- Hook-disabled path folds away related callsites via `const_expr` guards.

---

### Frame D: SM90 launch setup (`FlashAttentionBackwardSm90.__call__`)
Reference: [flash_bwd_sm90.py#L303-L515](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py#L303)

**Before**
- CuTe tensors exist, but kernel launch plumbing (layouts, TMA descriptors, scheduler params) is not finalized.

**Action**
- Reorders tensor layouts to `(s, h, n, b)`-oriented views for tiled access.
- Builds tiled MMA operators for 5 compute stages:
  1. `S = Q @ K^T`
  2. `dP = dO @ V^T`
  3. `dV += P^T @ dO`
  4. `dK += dS^T @ Q`
  5. `dQ = dS @ K`
- Configures shared memory layouts and copy atoms.
- Creates TMA atoms for Q/K/V/dO loads and optional dK/dV stores.
- Builds scheduler args and launches kernel.

**After**
- One CTA per `(n_block, head, batch)` tile begins execution.

Annotated snippet:

```python
# file: thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py
# link: ../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py#L471
self.kernel(
    tma_tensor_Q, tma_tensor_K, tma_tensor_V, tma_tensor_dO,
    ...,
    tiled_mma_SdP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ,
    softmax_scale_log2, softmax_scale,
    tile_sched_params, TileScheduler, SharedStorage,
    aux_tensors, fastdiv_mods, blocksparse_tensors,
).launch(grid=grid_dim, block=[self.num_threads, 1, 1], smem=SharedStorage.size_in_bytes(), ...)
```

---

### Frame E: Kernel entry and role split
Reference: [flash_bwd_sm90.py#L517-L719](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py#L517)

Within each CTA:
- Creates two async pipelines:
  - `pipeline_Q` for Q + LSE side
  - `pipeline_dO` for dO + dPsum side
- Materializes shared buffers for Q/K/V/dO/P/dS plus stats and dQaccum region.
- Splits warp roles:
  - producer (warps `< 4`): loads/stages data + dQaccum writer helper
  - consumer (remaining warps): runs MMA and pointwise math

Annotated snippet:

```python
# file: thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py
# link: ../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py#L642
if warp_idx < 4:
    if warp_idx == 0:
        self.load(...)
    if warp_idx == 1:
        self.dQaccum_store(...)
else:
    tidx = cute.arch.thread_idx()[0] - 128
    self.mma(..., tidx, ...)
```

---

### Frame F: Producer loading path (`load`)
Reference: [flash_bwd_sm90.py#L721-L878](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py#L721)

Looping over scheduler work tiles `(n_block, head, batch)`:
1. Load K and V tile (once per n_block tile).
2. Stream Q/LSE tiles across all contributing `m_block`s.
3. Stream dO/dPsum tiles in lockstep with Q side.
4. Use pipeline acquire/commit and stage-index advance.

Key point:
- Backward iterates fixed KV tile and sweeps Q tiles (`m_block`) that interact with it.

---

### Frame G: Consumer setup (`mma`)
Reference: [flash_bwd_sm90.py#L967-L1268](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py#L967)

Setup phase:
- Partitions thread fragments for all GEMMs.
- Builds `score_mod_fn` and `score_mod_bwd_fn` closures.
- Creates mask closure from `AttentionMask` with `swap_AB` awareness.
- Iterates over `m_block` range per current `n_block`.

Important threading model:
- `warp_group_idx = tidx // 128` among consumer threads.
- Multiple consumer warpgroups cooperatively produce dK/dV and stage dQaccum reductions.

---

### Frame H: Core per-`m_block` math (`mma_one_m_block`)
Reference: [flash_bwd_sm90.py#L1270-L1413](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py#L1270)

Per-step sequence:
1. **GEMM1:** `S = Q @ K^T`.
2. **GEMM2:** `dP = dO @ V^T`.
3. If needed, save `S_pre` for `score_mod_bwd`.
4. Optional `score_mod(S)` (replays forward score transform before softmax path).
5. Optional mask on `S`.
6. Pointwise softmax reconstruction: `P = exp2(S * scale_log2 - LSE_log2)`.
7. Pointwise derivative core: `dS = P * (dP - dPsum)`.
8. Optional `score_mod_bwd(dS, S_pre)` to chain rule through custom score transform.
9. Convert/store `P` and `dS` to shared memory fragments.
10. **GEMM3:** `dV += P^T @ dO`.
11. **GEMM4:** `dQ = dS @ K` (written to `dQaccum` staging area).
12. **GEMM5:** `dK += dS^T @ Q`.

Annotated snippet:

```python
# file: thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py
# link: ../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py#L1320
# Pointwise softmax reconstruction
acc_S_mn[r, c] = exp2(acc_S_mn[r, c] * softmax_scale_log2 - tLSErLSE[r])

# Core Jacobian contraction
acc_dP_mn[r, c] = acc_S_mn[r, c] * (acc_dP_mn[r, c] - tLSErdPsum[r])

if self.score_mod_bwd is not None:
    score_mod_bwd_fn(acc_dP, acc_S_pre, m_block=m_block)
```

Performance relevance:
- Keeps high-volume matrix products in WGMMA paths.
- Restricts custom-hook overhead to pointwise stage around `S`/`dS`.

---

### Frame I: `score_mod`, `score_mod_bwd`, `mask_mod` effects in backward

#### I1. `score_mod` (forward replay in backward)
References:
- wrapper: [flash_bwd_sm90.py#L880](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py#L880)
- inner: [softmax.py#L343](../thirdparty/flash-attention/flash_attn/cute/softmax.py#L343)

Effect:
- Reapplies forward score transformation to `S` before softmax reconstruction.
- Uses tile coordinates + logical indices (`transpose_indices=self.SdP_swapAB`) for correct indexing.

#### I2. `score_mod_bwd` (custom gradient chain)
References:
- wrapper: [flash_bwd_sm90.py#L923](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py#L923)
- inner: [softmax.py#L473](../thirdparty/flash-attention/flash_attn/cute/softmax.py#L473)

Effect:
- Receives `(grad_tensor=dS, score_tensor=S_pre)` and writes transformed gradient in-place.
- This is the chain-rule insertion for user-defined score transforms.

#### I3. `mask_mod`
References:
- callsite: [flash_bwd_sm90.py#L1171-L1181](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py#L1171)
- mask implementation: [mask.py#L128](../thirdparty/flash-attention/flash_attn/cute/mask.py#L128)

Effect:
- Applied before softmax reconstruction; masked entries become `-inf` then produce zero probabilities/gradients.
- In block-sparse mode, partial and full block lists are separated so `mask_mod` is used only where needed.

Block-sparse helper references:
- producer: [block_sparse_utils.py#L1144](../thirdparty/flash-attention/flash_attn/cute/block_sparse_utils.py#L1144)
- consumer: [block_sparse_utils.py#L1240](../thirdparty/flash-attention/flash_attn/cute/block_sparse_utils.py#L1240)

---

## Deep Dive: How Backward `score_mod`/`mask_mod` differ from Forward (by gradient)

Forward only needs to produce `P` and `O`; backward must split influence across `dV`, `dK`, and `dQ`.

### 1. Forward vs backward hook roles

Forward:
1. `S = QK`
2. optional `score_mod(S)`
3. optional `mask_mod` / causal / seqlen masking
4. softmax -> `P`
5. `O = P @ V`

Backward (SM90 path):
1. recompute `S = QK`
2. optional `score_mod(S)` replay (same logical transform as forward)
3. optional `mask_mod` / causal / seqlen masking replay
4. reconstruct `P` via `exp2(S * scale_log2 - LSE)`
5. form `dP = dO @ V^T`
6. form base `dS = P * (dP - dPsum)`
7. optional `score_mod_bwd(dS, S_pre)` chain rule through custom score transform
8. matrix gradients:
   - `dV += P^T @ dO`
   - `dK += dS^T @ Q`
   - `dQ += dS @ K`

Key extra backward-only requirement:
- `score_mod_bwd` is needed to map gradient from post-modified scores back to pre-modified scores.

References:
- replay + dS path: [flash_bwd_sm90.py#L1317-L1350](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py#L1317)
- score-mod-bwd inner: [softmax.py#L473](../thirdparty/flash-attention/flash_attn/cute/softmax.py#L473)

### 2. Per-gradient effect of `score_mod` / `mask_mod`

#### `dV` path (`dV += P^T @ dO`)

`dV` depends on `P`, not directly on `dS`.

Implications:
- `score_mod` changes `P`, so it changes `dV` indirectly via attention probabilities.
- `score_mod_bwd` does **not** directly participate in `dV` update.
- `mask_mod` zeros masked `P` entries, so masked lanes contribute zero to `dV`.

Reference:
- dV GEMM: [flash_bwd_sm90.py#L1368-L1375](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py#L1368)

#### `dK` path (`dK += dS^T @ Q`)

`dK` depends on `dS`, so it is directly sensitive to backward hook logic.

Implications:
- `score_mod` replay affects the reconstructed `P` and therefore base `dS`.
- `score_mod_bwd` directly transforms `dS`; this directly changes `dK`.
- `mask_mod` zeros masked logits -> zero `P` and zero `dS` on those lanes -> no `dK` contribution.

Reference:
- dK GEMM: [flash_bwd_sm90.py#L1384-L1391](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py#L1384)

#### `dQ` path (`dQ += dS @ K`)

`dQ` also depends on `dS`, so it matches `dK` sensitivity pattern.

Implications:
- same `dS` transformations (`score_mod` replay + optional `score_mod_bwd`) flow into `dQ`.
- `mask_mod` prunes lanes before `dS` GEMMs, so masked lanes do not contribute to `dQ`.
- implementation detail: `dQ` is staged into `dQaccum` and reduced asynchronously; hook effects are already baked into staged `dS`.

References:
- dQ GEMM: [flash_bwd_sm90.py#L1379-L1381](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py#L1379)
- dQaccum store path: [flash_bwd_sm90.py#L1562](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py#L1562)

### 3. Why this differs from forward operationally

1. Forward has one hook application point (`score_mod`, `mask_mod`) before softmax.
2. Backward must replay that point to reconstruct `P`, then apply a second hook (`score_mod_bwd`) on `dS`.
3. Therefore backward has split hook influence:
   - probability path (`P`) influences `dV`
   - score-gradient path (`dS`) influences `dK` and `dQ`
4. `mask_mod` affects both paths by zeroing masked lanes early; those lanes are removed from both `P`-weighted and `dS`-weighted contractions.

### 4. Block-sparse nuance

In block-sparse backward, partial and full block lists are separated:
- partial blocks: `mask_mod` is evaluated
- full blocks: `mask_mod` is skipped

This keeps dense/full regions on a cheaper path while preserving exact masking semantics on sparse boundary blocks.

References:
- sparse consumer split: [block_sparse_utils.py#L1262-L1342](../thirdparty/flash-attention/flash_attn/cute/block_sparse_utils.py#L1262)

---

### Frame J: dKV epilogue and dQaccum store

#### J1. dKV epilogue
Reference: [flash_bwd_sm90.py#L1415](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py#L1415)

Two paths:
1. `qhead_per_kvhead == 1`: convert `acc_dK/acc_dV` to output dtype and store via TMA S2G.
2. GQA accumulation path: reduce-add fp32 accumulators into global accumulation buffers (`cpasync_reduce_bulk_add_f32`).

#### J2. dQaccum writer
Reference: [flash_bwd_sm90.py#L1562](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py#L1562)

- A dedicated producer warp consumes dQ fragments from consumer warpgroups using named barriers.
- Performs asynchronous reduce-add writes of fp32 dQaccum tiles to global memory.

Synchronization primitives used:
- [named_barrier.py:NamedBarrierBwd](../thirdparty/flash-attention/flash_attn/cute/named_barrier.py#L15)

---

### Frame K: Postprocess conversion kernel
Reference: [flash_bwd_postprocess.py#L201](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_postprocess.py#L201)

Responsibilities:
- Load fp32 accumulators (`dQaccum`, and optionally `dKaccum`/`dVaccum` from launcher side).
- Reduce/layout-adapt per tiled MMA mapping.
- Convert to destination dtype (bf16/fp16).
- Apply scale when required.

Launcher references for postprocess dispatch:
- [interface.py#L1124-L1251](../thirdparty/flash-attention/flash_attn/cute/interface.py#L1124)

## Data Layout and Thread-Value Mapping

### 1. Backward logical tiling direction
- Forward main loop fixed `m_block` and swept `n_block`.
- Backward main loop fixed `n_block` and sweeps contributing `m_block`s.

This matches gradient dependencies for `dK/dV` per KV tile.

### 2. CTA role partition
- Producer side: staged memory movement + dQaccum store helper.
- Consumer side: all heavy GEMM + pointwise differentiation.

Reference: [flash_bwd_sm90.py#L642-L683](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py#L642)

### 3. Swap/transposed semantics (`SdP_swapAB`)
- `AttentionMask` and score-mod index mapping use `swap_AB`/`transpose_indices` so coordinate semantics remain correct when MMA operand order is transposed.

References:
- [flash_bwd_sm90.py#L893-L901](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py#L893)
- [softmax.py#L382-L388](../thirdparty/flash-attention/flash_attn/cute/softmax.py#L382)

### 4. Shared-memory staging
- Q/LSE and dO/dPsum each have dedicated stage pipelines.
- P and dS share staged storage layouts for downstream GEMMs.
- dQaccum staging is structured for warpgroup-partitioned reduction.

## Optimized Data Movement and Compute Patterns
1. **Two async producer pipelines** for Q-side and dO-side statistics and tiles.
2. **WGMMA-heavy compute core** keeps the five dominant matrix contractions in tensor-core paths.
3. **Pointwise fusion window** computes softmax reconstruction and Jacobian contraction in-register.
4. **Dedicated dQaccum writer path** decouples reduction/store traffic from consumer math issue.
5. **Hook specialization** keeps no-hook kernels lean while still supporting custom score/mask logic.
6. **Block-sparse split loops** apply expensive masking/hook logic only on partial blocks.

## End-to-End Pseudocode (SM90 backward)

```python
# Host launcher
preprocess(out, dout, lse) -> dPsum, lse_log2, init dQaccum
main_sm90(q, k, v, dout, dPsum, lse_log2, dQaccum, dK_or_dKaccum, dV_or_dVaccum, hooks...)
postprocess(dQaccum -> dQ)
if gqa_accum:
    postprocess(dKaccum -> dK)
    postprocess(dVaccum -> dV)

# Main kernel per KV tile (n_block)
producer:
    stage K,V once
    for contributing m_block:
        stage Q,LSE and dO,dPsum

consumer for each m_block:
    S   = Q @ K^T
    dP  = dO @ V^T
    if score_mod:     S   = score_mod(S)
    if mask_mod/etc:  S   = mask(S)
    P   = exp2(S*scale_log2 - LSE)
    dS  = P * (dP - dPsum)
    if score_mod_bwd: dS  = score_mod_bwd(dS, S_pre)
    dV += P^T  @ dO
    dQ  = dS   @ K      -> accumulate/store via dQaccum channel
    dK += dS^T @ Q
```

## Per-Frame State Summary
| Frame | Before | After |
|---|---|---|
| A | raw backward tensors | validated config + allocated fp32 workspaces |
| B | no stats buffers | `dPsum`, `lse_log2`, initialized `dQaccum` |
| C | no compiled kernel | architecture/hook-specialized main kernel object cached |
| E/F | no staged tiles | staged Q/K/V/dO/stats for active tile |
| H | partial grads in regs | updated `acc_dK`, `acc_dV`, staged `dQaccum` |
| J | reg/shared partials | persisted dK/dV and reduced dQaccum global updates |
| K | fp32 accumulation buffers | final dtype output gradients |

## Notes on Lowest-Level Boundary
The deepest inspectable boundary in this repository-level trace is:
- `cute.compile(...)` (host lowering boundary) in [interface.py#L1071](../thirdparty/flash-attention/flash_attn/cute/interface.py#L1071)
- `@cute.kernel` runtime bodies in [flash_bwd_sm90.py#L517](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_sm90.py#L517), [flash_bwd_preprocess.py#L188](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_preprocess.py#L188), and [flash_bwd_postprocess.py#L278](../thirdparty/flash-attention/flash_attn/cute/flash_bwd_postprocess.py#L278)

Below that, CuTe/MLIR lowers to generated backend GPU code (PTX/CUBIN path outside this source-level trace).
