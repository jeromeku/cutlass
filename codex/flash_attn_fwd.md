# FlashAttention CuTeDSL Forward Trace (SM90)

## Scope
This document traces the SM90 forward kernel path starting from:
- [interface.py:_flash_attn_fwd](../thirdparty/flash-attention/flash_attn/cute/interface.py#L94)

and following all major frames into:
- [flash_fwd.py:FlashAttentionForwardSm90](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py#L1147)
- [flash_fwd.py:FlashAttentionForwardSm90.kernel](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py#L1530)
- producer/consumer loops, masking, score-mod, softmax, and epilogue.

Target focus:
1. Kernel data structures and launcher setup.
2. Helper functions and per-frame state transitions.
3. Data layout, thread-value mapping, and performance optimizations.

## Code Map
| Area | Link |
|---|---|
| Forward API entry + compile cache | [interface.py](../thirdparty/flash-attention/flash_attn/cute/interface.py#L94) |
| SM90 forward class | [flash_fwd.py](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py#L1147) |
| Base shared-memory/copy setup | [flash_fwd.py](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py#L207) |
| SM90 kernel body | [flash_fwd.py](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py#L1530) |
| Producer path (`load`) | [flash_fwd.py](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py#L1719) |
| Consumer path (`mma`) | [flash_fwd.py](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py#L1840) |
| N-block step (non-overlap) | [flash_fwd.py](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py#L2241) |
| N-block step (intra-WG overlap) | [flash_fwd.py](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py#L2300) |
| Score-mod inner application | [softmax.py](../thirdparty/flash-attention/flash_attn/cute/softmax.py#L343) |
| Masking logic | [mask.py](../thirdparty/flash-attention/flash_attn/cute/mask.py#L128) |
| Sequence metadata object | [seqlen_info.py](../thirdparty/flash-attention/flash_attn/cute/seqlen_info.py#L38) |
| Tile bound computation | [block_info.py](../thirdparty/flash-attention/flash_attn/cute/block_info.py#L24) |
| Tile scheduler types | [tile_scheduler.py](../thirdparty/flash-attention/flash_attn/cute/tile_scheduler.py#L57) |
| Pipeline wrappers | [pipeline.py](../thirdparty/flash-attention/flash_attn/cute/pipeline.py#L15) |
| Pack-GQA helper | [pack_gqa.py](../thirdparty/flash-attention/flash_attn/cute/pack_gqa.py#L11) |
| Named barriers | [named_barrier.py](../thirdparty/flash-attention/flash_attn/cute/named_barrier.py#L6) |
| WGMMA fragment partition helper | [quack/sm90_utils.py](../.venv/lib/python3.12/site-packages/quack/sm90_utils.py#L131) |
| Accumulator layout reshaping helpers | [quack/layout_utils.py](../.venv/lib/python3.12/site-packages/quack/layout_utils.py#L190) |
| TMA copy wrappers | [quack/copy_utils.py](../.venv/lib/python3.12/site-packages/quack/copy_utils.py#L440) |

## Key Functions Index
| Function | File | Purpose |
|---|---|---|
| `_flash_attn_fwd` | [interface.py](../thirdparty/flash-attention/flash_attn/cute/interface.py#L94) | Validates args, builds compile key, compiles/dispatches CuTe kernel. |
| `FlashAttentionForwardSm90.__call__` | [flash_fwd.py](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py#L1247) | Builds launch-time kernel params and launches `@cute.kernel`. |
| `FlashAttentionForwardSm90.kernel` | [flash_fwd.py](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py#L1530) | Block-level producer/consumer orchestration. |
| `FlashAttentionForwardSm90.load` | [flash_fwd.py](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py#L1719) | Producer: stage Q/K/V tiles into shared memory via TMA pipeline. |
| `FlashAttentionForwardSm90.mma` | [flash_fwd.py](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py#L1840) | Consumer: QK GEMM, mask/mod/softmax, PV GEMM, epilogue. |
| `Softmax.online_softmax/finalize` | [softmax.py](../thirdparty/flash-attention/flash_attn/cute/softmax.py#L53) | Streaming softmax across K/V tiles and final row normalization/LSE. |
| `AttentionMask.apply_mask` | [mask.py](../thirdparty/flash-attention/flash_attn/cute/mask.py#L128) | Applies seqlen/causal/local/flex mask_mod predicates on score tiles. |
| `apply_score_mod_inner` | [softmax.py](../thirdparty/flash-attention/flash_attn/cute/softmax.py#L343) | Vectorized score modification hook for flex attention. |

## Big Picture Call Chain (SM90 fwd)
1. `_flash_attn_fwd(...)` normalizes inputs and chooses architecture.
2. For `compute_capability == 9`, it instantiates `FlashAttentionForwardSm90(...)`.
3. `cute.compile(...)` lowers the Python DSL object to generated kernel code keyed by compile options.
4. Compiled callable is cached in `_flash_attn_fwd.compile_cache[compile_key]`.
5. Runtime invocation passes concrete tensors and scalar params.
6. Kernel launches with one producer warpgroup + N consumer warpgroups per block.

## Frame-By-Frame Trace

### Frame A: Host entry and compile cache (`interface.py`)
Reference: [interface.py#L94-L548](../thirdparty/flash-attention/flash_attn/cute/interface.py#L94)

**Before**
- Inputs are PyTorch tensors in `(B,S,H,D)` or varlen representations.
- No CuTe kernel object yet.

**Action**
- Validates shapes/dtypes/device, computes defaults (`softmax_scale`, block sizes).
- Decides causal/local/mask_mod mode and split-kv usage.
- Computes `compile_key` containing architecture + head dims + masking/score-mod hashes + layout mode flags.
- On miss, creates SM90 object and compiles it.

**After**
- A compiled CuTe callable exists (cache miss) or is reused (cache hit).
- Runtime call dispatches with concrete tensors.

Annotated snippet:

```python
# file: thirdparty/flash-attention/flash_attn/cute/interface.py
# link: ../thirdparty/flash-attention/flash_attn/cute/interface.py#L376
compile_key = (
    dtype, head_dim, head_dim_v,
    qhead_per_kvhead,
    causal,
    score_mod_hash,   # score_mod callable identity baked into codegen cache key
    mask_mod_hash,    # mask_mod callable identity baked into codegen cache key
    ...,
    m_block_size, n_block_size, q_stage, num_threads,
    is_split_kv, pack_gqa, compute_capability,
)

if compile_key not in _flash_attn_fwd.compile_cache:
    # SM90 kernel object materialization for Hopper
    fa_fwd = FlashAttentionForwardSm90(...)
    _flash_attn_fwd.compile_cache[compile_key] = cute.compile(
        fa_fwd, q_tensor, k_tensor, v_tensor, o_tensor, lse_tensor, ...
    )

# Runtime execution of compiled callable
_flash_attn_fwd.compile_cache[compile_key](q.detach(), k.detach(), v.detach(), ...)
```

Performance relevance:
- Compile-key granularity avoids re-lowering while still specializing for critical parameters.
- Hashing score/mask callables allows custom logic without wrong-kernel reuse.

---

### Frame B: SM90 object launch prep (`FlashAttentionForwardSm90.__call__`)
Reference: [flash_fwd.py#L1247-L1528](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py#L1247)

**Before**
- Input tensors are CuTe tensor wrappers but still in user-facing logical layout.
- No launch grid/block fixed yet.

**Action**
- Validates element types and optional metadata tensor types.
- Reorders layouts for kernel-friendly access (`layout_utils.select`).
- Derives warpgroup/thread allocation and register budgets.
- Builds SMEM layouts and TMA atoms for Q/K/V/O.
- Chooses scheduler class (`SingleTileScheduler`, `SingleTileLPTScheduler`, `SingleTileVarlenScheduler`).
- Computes `softmax_scale_log2` (and modified handling when score_mod exists).
- Launches `self.kernel(...).launch(grid=..., block=[self.num_threads,1,1], ...)`.

**After**
- Kernel launch is parameterized with concrete TMA descriptors, scheduler params, and closures.

Annotated snippet:

```python
# file: thirdparty/flash-attention/flash_attn/cute/flash_fwd.py
# link: ../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py#L1288
# Thread topology: 1 producer warpgroup + N consumer warpgroups
self.num_threads_per_warp_group = 128
self.num_mma_warp_groups = self.num_mma_threads // self.num_threads_per_warp_group
self.num_threads = self.num_threads_per_warp_group * (self.num_mma_warp_groups + 1)
self.num_producer_threads = 32

# TMA path toggles
self.use_tma_Q = self.arch >= 90 and not (self.pack_gqa and self.tile_m % self.qhead_per_kvhead != 0)
self.use_tma_O = self.arch >= 90 and mCuSeqlensQ is None and mSeqUsedQ is None and not self.pack_gqa

# Build TMA atoms for global<->shared traffic
tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(...)
tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(...)

# Launch
self.kernel(...).launch(grid=grid_dim, block=[self.num_threads, 1, 1], stream=stream)
```

Performance relevance:
- The dedicated producer warpgroup lets consumer warpgroups keep WGMMA issue rate high.
- TMA atomization removes per-thread pointer math from steady-state loads/stores.
- Register budgets (`setmaxregister_*` later) are tuned by number of MMA warpgroups.

---

### Frame C: Base shared-memory/copy structure setup (`_setup_attributes`)
Reference: [flash_fwd.py#L207-L302](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py#L207)

**Before**
- Tile sizes known, but copy tilings and shared-memory tensor views not instantiated.

**Action**
- Materializes SMEM layouts:
  - `sQ_layout: (tile_m, tile_hdim)`
  - `sK_layout: (tile_n, tile_hdim, num_stages)`
  - `sV_layout: (tile_n, tile_hdimv, num_stages)`
  - `sO_layout: (tile_m, tile_hdimv)`
- Builds tiled-copy descriptors:
  - async copy atoms for Q/K/V global->shared
  - universal copy atom for O shared->global fallback
- Creates thread/value layouts for each copy path.

**After**
- Kernel has deterministic thread-value mapping for all memory movement primitives.

Annotated snippet:

```python
# file: thirdparty/flash-attention/flash_attn/cute/flash_fwd.py
# link: ../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py#L214
self.sQ_layout = cute.tile_to_shape(..., (self.tile_m, self.tile_hdim), ...)
self.sK_layout = cute.tile_to_shape(..., (self.tile_n, self.tile_hdim, self.num_stages), ...)
self.sV_layout = cute.tile_to_shape(..., (self.tile_n, self.tile_hdimv, self.num_stages), ...)

atom_async_copy = cute.make_copy_atom(cpasync.CopyG2SOp(...), self.dtype, num_bits_per_copy=128)
self.gmem_tiled_copy_Q = cute.make_tiled_copy_tv(atom_async_copy, tQ_layout, vQKV_layout)
self.gmem_tiled_copy_K = cute.make_tiled_copy_tv(atom_async_copy, tK_layout, vQKV_layout)
self.gmem_tiled_copy_V = cute.make_tiled_copy_tv(atom_async_copy, tV_layout, vQKV_layout)
```

Performance relevance:
- Fixed 128-bit transactions maximize vectorized bandwidth.
- Staged K/V layouts provide ring-buffer pipeline slots for overlap.

---

### Frame D: Kernel entry and producer/consumer split (`kernel`)
Reference: [flash_fwd.py#L1530-L1717](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py#L1530)

**Before**
- CTA launched; no staged tiles yet.

**Action**
- Prefetches TMA descriptors (warp 0).
- Allocates shared storage and initializes barriers/pipelines.
- Creates shared tensor views (`sQ`, `sK`, `sV`, `sVt`, `sO`, optional `sP`).
- Builds `BlockInfo`, `SeqlenInfoCls`, `AttentionMaskCls`, `TileSchedulerCls` closures.
- Splits roles:
  - `warp_idx < 4`: producer path (`load`)
  - else: consumer path (`mma`)

**After**
- Producer begins K/V (and optionally Q) staging; consumers begin compute once data ready.

Annotated snippet:

```python
# file: thirdparty/flash-attention/flash_attn/cute/flash_fwd.py
# link: ../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py#L1660
if warp_idx < 4:  # Producer warps
    cute.arch.setmaxregister_decrease(self.num_producer_regs)
    self.load(...)
else:             # Consumer warps
    cute.arch.setmaxregister_increase(self.num_mma_regs)
    tidx = cute.arch.thread_idx()[0] - 128
    self.mma(..., tidx, ...)
```

Thread/value mapping note:
- Logical lane-to-work mapping is bifurcated by role. Producers minimize register footprint; consumers maximize accumulator residency.

---

### Frame E: Producer pipeline (`load`)
Reference: [flash_fwd.py#L1719-L1838](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py#L1719)

**Before**
- Tile scheduler yields `(m_block, head_idx, batch_idx)`.
- No staged KV slots for this tile yet.

**Action**
- Computes per-batch sequence offsets via `SeqlenInfoQK`.
- Builds local tiles of `mQ/mK/mV` for this tile.
- Builds TMA copy functions with barrier-aware wrappers (`copy_utils.tma_get_copy_fn`, `tma_producer_copy_fn`).
- Enqueues K/V (and first Q) into pipeline states.
- For overlap mode (`intra_wg_overlap=True`), K and V copies are staggered to overlap producer work.

**After**
- Pipeline slots contain staged K/V (and Q when enabled), synchronized via mbarriers.

Annotated snippet:

```python
# file: thirdparty/flash-attention/flash_attn/cute/flash_fwd.py
# link: ../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py#L1780
pipeline_k.producer_acquire(kv_producer_state, extra_tx_count=self.tma_copy_bytes["Q"] if self.use_tma_Q else 0)
if self.use_tma_Q:
    load_Q(tma_bar_ptr=pipeline_k.producer_get_barrier(kv_producer_state))
load_K(src_idx=n_block, producer_state=kv_producer_state)

# overlap mode: launch next K while prior V is still being issued
pipeline_k.producer_acquire(kv_producer_state_next)
load_K(src_idx=n_block_next, producer_state=kv_producer_state_next)
pipeline_v.producer_acquire(kv_producer_state_prev)
load_V(src_idx=n_block_prev, producer_state=kv_producer_state_prev)
```

Performance relevance:
- K and V use independent async pipelines (`pipeline_k`, `pipeline_v`) so traffic can overlap.
- `extra_tx_count` folds Q traffic into first pipeline stage without extra synchronization round.

---

### Frame F: Consumer compute loop (`mma`)
Reference: [flash_fwd.py#L1840-L2174](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py#L1840)

**Before**
- Shared-memory tiles become available from producer.
- Consumer thread has `tidx` in consumer domain.

**Action**
- Partitions WGMMA fragments for QK and PV.
- Creates softmax state object and per-tile mask/score-mod closures.
- Waits on Q barrier when Q not loaded by TMA.
- Executes n-block sweep in phases:
  - first masked/seqlen-sensitive block
  - causal/local masked region
  - unmasked steady-state region
  - optional local-left region
- Applies online softmax every n-block and rescales running `acc_O`.
- Finalizes softmax (row_sum -> LSE), rescales `acc_O`, writes O/LSE in epilogue.

**After**
- Output tile `O` and optional `LSE` are committed to global memory.

Annotated snippet:

```python
# file: thirdparty/flash-attention/flash_attn/cute/flash_fwd.py
# link: ../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py#L1964
mask = AttentionMaskCls(seqlen)
mask_fn = partial(mask.apply_mask, batch_idx=batch_idx, head_idx=head_idx, m_block=m_block, ...)

if self.score_mod is not None:
    score_mod_fn = partial(self.apply_score_mod, thr_mma_qk, batch_idx, head_idx, m_block, ...)

# Main loop over N tiles for fixed M tile
kv_consumer_state = mma_one_n_block(...)
...
row_scale = softmax.finalize(sink_val=sink_val)
softmax.rescale_O(acc_O, row_scale)
self.epilogue(acc_O, softmax.row_sum, mO, mLSE, ...)
```

Performance relevance:
- Online softmax avoids full score materialization across all K blocks.
- `acc_O` remains in registers across loop iterations, minimizing spill/load overhead.
- Masking is split into special-case regions to avoid unnecessary predicate work in fully valid blocks.

---

### Frame G: Single N-block micro-step (non-overlap vs overlap)
References:
- [mma_one_n_block](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py#L2241)
- [mma_one_n_block_intrawg_overlap](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py#L2300)

#### G1. Non-overlap (`mma_one_n_block`)
1. Wait K stage.
2. `acc_S = Q @ K^T`.
3. Score-mod + mask.
4. `online_softmax(acc_S)` and convert probabilities to FP16/BF16 fragment (`utils.cvt_f16`).
5. Wait V stage.
6. `acc_O += P @ V`.
7. Advance pipeline state.

#### G2. Intra-WG overlap (`mma_one_n_block_intrawg_overlap`)
1. Issue QK for current stage and PV for prior stage concurrently.
2. Use warpgroup wait groups and named barriers for ordering.
3. Post-process `acc_S` (mask/score_mod/softmax) while prior PV is completing.
4. Convert/store probabilities and continue.

Why overlap helps:
- It hides portions of PV latency under QK + post-QK scalar work.
- Better utilization when multiple consumer warpgroups share CTA resources.

---

### Frame H: Score-mod application path
References:
- [flash_fwd.py:apply_score_mod](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py#L2370)
- [softmax.py:apply_score_mod_inner](../thirdparty/flash-attention/flash_attn/cute/softmax.py#L343)

**Before**
- `acc_S` contains raw QK dot-product tile in FP32.

**Action**
- Builds identity coordinate tensor for score positions.
- Passes vectorized batches into user `score_mod` callback.
- Handles optional aux tensor wrapping with fast divmod helpers.
- Writes modified scores back into `acc_S` in-place.

**After**
- `acc_S` contains score-modified values consumed by mask + softmax.

Annotated snippet:

```python
# file: thirdparty/flash-attention/flash_attn/cute/softmax.py
# link: ../thirdparty/flash-attention/flash_attn/cute/softmax.py#L404
for i in range(0, n_vals, vec_size):
    score_vec[j] = score_tensor[i + j] * softmax_scale
    q_idx_vec[j], kv_idx_vec[j] = ...  # logical indices (pack-gqa aware)
    post_mod_scores = score_mod(score_ssa, batch_idx_ssa, head_idx_ssa, q_idx=q_idx_ssa, kv_idx=kv_idx_ssa, ...)
    score_tensor[i + j] = post_mod_scores[j]
```

Performance note:
- Vectorized callback invocation amortizes per-element index plumbing.

---

### Frame I: Masking path
Reference: [mask.py:AttentionMask.apply_mask](../thirdparty/flash-attention/flash_attn/cute/mask.py#L128)

Masking combines up to four constraints:
1. Sequence bounds (`mask_seqlen`).
2. Causal relation.
3. Local window (left/right).
4. Optional user `mask_mod`.

Key implementation details:
- Converts accumulator to M/N view using `layout_utils.reshape_acc_to_mn`.
- Uses thread-0 index tensor where possible (`t0ScS_mn`) so comparisons are compile-time-friendly.
- For some paths, uses R2P-style bitmask logic (`mask_r2p`) to cut scalar predicate overhead.

Performance note:
- Mask separation into dedicated loop regions avoids paying expensive checks in dense/full-valid blocks.

---

## Deep Dive: How `score_mod` and `mask_mod` change kernel execution

This section expands the exact control/data-flow impact of both hooks in SM90 forward.

### 1. Compile-time specialization impact

At launcher level (`_flash_attn_fwd`), both callables are hashed and included in the compile key:

```python
# file: thirdparty/flash-attention/flash_attn/cute/interface.py
# link: ../thirdparty/flash-attention/flash_attn/cute/interface.py#L318
score_mod_hash = utils.hash_callable(score_mod) if score_mod is not None else False
mask_mod_hash = utils.hash_callable(mask_mod) if mask_mod is not None else False
...
compile_key = (..., score_mod_hash, mask_mod_hash, ...)
```

Effect:
- Different hook logic forces a distinct compiled kernel variant.
- Hook-enabled kernels keep extra code paths live (`apply_score_mod`, `mask_mod` predicates, aux handling).
- Hook-disabled kernels remove those branches at compile time (`const_expr` conditions fold away).

### 2. Runtime ordering in the score path (`S` tile lifecycle)

For each `(m_block, n_block)` in the consumer loop:
1. Compute raw scores: `acc_S = Q @ K^T`.
2. If `score_mod` exists: mutate `acc_S` via `apply_score_mod_inner`.
3. Apply masking (`mask_mod` and/or causal/local/seqlen) through `AttentionMask.apply_mask`.
4. Run online softmax on the post-mod, post-mask scores.

Relevant frames:
- score mod callsite: [flash_fwd.py#L2267-L2269](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py#L2267)
- mask callsite: [flash_fwd.py#L2269-L2271](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py#L2269)

Important behavior:
- `mask_mod` can overwrite values with `-inf` after `score_mod`; masked entries do not contribute to softmax row sums.
- `score_mod` therefore matters only for entries surviving mask predicates.

### 3. `score_mod` dataflow details

Forward hook dispatch path:
- wrapper: [flash_fwd.py:apply_score_mod](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py#L2370)
- vectorized inner: [softmax.py:apply_score_mod_inner](../thirdparty/flash-attention/flash_attn/cute/softmax.py#L343)

Core mechanics inside `apply_score_mod_inner`:
- Builds logical `(q_idx, kv_idx)` from thread-fragment coordinates.
- Applies `softmax_scale` before calling user hook.
- Supports Pack-GQA logical-head reconstruction (`head_idx` remap when packed rows are used).
- Wraps indices with `FastDivmodDivisor` when aux tensors are present to avoid out-of-bounds indexing.

Net impact:
- Extra integer index plumbing and callback invocation in score hot loop.
- Potentially changed numerical range before softmax normalization.

### 4. `mask_mod` dataflow details

Mask hook dispatch path:
- entry: [mask.py:AttentionMask.apply_mask](../thirdparty/flash-attention/flash_attn/cute/mask.py#L128)
- flex mask_mod branch: [mask.py#L175-L234](../thirdparty/flash-attention/flash_attn/cute/mask.py#L175)

Core mechanics:
- Evaluates `mask_mod(batch, head, q_idx, kv_idx, seqlen_info, aux_tensors)` per score position.
- Combines that boolean with boundary checks (`mask_seqlen`, causal/local windows).
- Writes masked lanes to `-inf`.

Net impact:
- Changes both correctness domain (which tokens attend) and performance profile.
- Full blocks without `mask_mod` can stay on cheaper mask path; partial/masked blocks pay richer predicate path.

### 5. Block-sparse interaction

When block sparsity is enabled, execution splits into partial (masked) and full (unmasked) block lists:
- producer side: [block_sparse_utils.py:produce_block_sparse_loads](../thirdparty/flash-attention/flash_attn/cute/block_sparse_utils.py#L133)
- consumer side: [block_sparse_utils.py:consume_block_sparse_loads](../thirdparty/flash-attention/flash_attn/cute/block_sparse_utils.py#L295)

For full blocks, `mask_mod` is intentionally skipped; for partial blocks it is applied. This reduces predicate overhead and keeps dense regions fast.

### 6. Practical throughput implications

1. `score_mod` primarily adds compute and index overhead in the QK-to-softmax phase.
2. `mask_mod` primarily adds predicate/control overhead and may reduce useful arithmetic intensity by zeroing lanes.
3. Both hooks increase compile-variant count (cache pressure), but preserve runtime specialization.
4. When neither hook is present, the kernel follows the leanest execution path.

---

### Frame J: Softmax state machine
Reference: [softmax.py:Softmax](../thirdparty/flash-attention/flash_attn/cute/softmax.py#L19)

Per tile row state:
- `row_max` (running max across processed N tiles)
- `row_sum` (running sum of exponentials after re-scaling)

Per n-block update:
1. Compute row max from current `acc_S`.
2. Rebase with prior max (`row_scale = exp2((old-new)*scale_log2)`).
3. Update row sums.
4. Store exponentiated/shifted tile back into `acc_S` (becomes `P` tile).

Finalize:
- Warp-reduce row sums.
- Optionally add sink term.
- Convert row sums to natural-log LSE.
- Return reciprocal row scale for `acc_O` normalization.

---

### Frame K: Epilogue
Reference: [flash_fwd.py:epilogue](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py#L331)

**Before**
- `acc_O` is FP32 accumulated output tile; `softmax.row_sum` contains LSE-domain state.

**Action**
- Converts `acc_O` to output dtype in register fragment.
- Stores through shared memory (or TMA O path when enabled).
- Writes LSE to global memory, with pack-GQA aware addressing when needed.
- Writes O using vectorized copy with head-dim predicates for OOB-safe tails.

**After**
- Output tile persisted; optional LSE tile persisted.

Performance note:
- TMA S2G for O is used only when layout/mode allows it (`use_tma_O`).
- Otherwise, vectorized universal copies plus predicate masking handle tails.

## Data Layout and Thread-Value Mapping

### 1. Tensor layout transforms at launch
- `mQ/mO` are selected into layout order `[1,3,2,0]` for fixed-len or `[0,2,1]` for varlen.
- `mK/mV` similarly transformed.
- This makes sequence/head dimensions align with tiling and scheduler expectations.

Reference: [flash_fwd.py#L1281-L1287](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py#L1281)

### 2. CTA role partition
- Producer domain: first 128 threads (`warp_idx < 4` check is at warp granularity, i.e. 4 warps).
- Consumer domain: remaining threads (`tidx = thread_idx - 128`).

Reference: [flash_fwd.py#L1660-L1688](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py#L1660)

### 3. Warpgroup compute mapping
- `warp_group_idx = tidx // 128` among consumer threads.
- QK and PV fragments are partitioned per warpgroup via `sm90_utils.partition_fragment_ABC`.

References:
- [flash_fwd.py#L1872-L1888](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py#L1872)
- [quack/sm90_utils.py#L131](../.venv/lib/python3.12/site-packages/quack/sm90_utils.py#L131)

### 4. Shared-memory ring buffers
- K/V carry stage dimension (`num_stages`) and rotate by pipeline state index.
- Q/O are single-stage per tile.

Reference: [flash_fwd.py#L214-L232](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py#L214)

### 5. Pack-GQA mapping
- Logical query-head replication is flattened into packed Q row space.
- Helper computes per-thread pointers that reconstruct `(q_head, q_row)` from packed index.

Reference: [pack_gqa.py#L25-L42](../thirdparty/flash-attention/flash_attn/cute/pack_gqa.py#L25)

## Optimized Data Movement and Compute Patterns
1. **TMA bulk copies for Q/K/V/O**
   - K/V always use TMA on SM90 path; Q/O use TMA when legal by mode constraints.
   - Reduces instruction count and pointer arithmetic in steady state.
2. **Dual async pipelines (K and V)**
   - Independent producer/consumer states improve overlap flexibility.
3. **Producer/consumer warpgroup specialization**
   - Producer register footprint minimized (`setmaxregister_decrease`).
   - Consumer given larger register budget for accumulators and softmax state.
4. **Intra-warpgroup overlap mode**
   - Pipelines QK and PV in partially overlapped order to hide latency.
5. **Streaming softmax (online algorithm)**
   - Avoids storing full attention score matrix; only tile-local score fragments exist transiently.
6. **Predicate strategy for tails**
   - Compile-time friendly index tensors and optional R2P bitmasking reduce branch/predicate overhead.
7. **Fast float convert path (`cvt_f16x2`)**
   - Uses packed conversion to keep probability conversion vectorized.
8. **Scheduler selection by workload shape**
   - LPT and varlen schedulers improve load balance and cache locality.

## End-to-End Pseudocode (SM90 fwd)

```python
# Host frame
compiled = compile_cache.get_or_compile(
    key=(dtype, head_dim, causal, score_mod_hash, ...),
    kernel=FlashAttentionForwardSm90(...)
)
compiled(q, k, v, o, lse, ...)

# Device frame (per CTA)
init_shared_storage_and_pipelines()
if warpgroup_is_producer:
    for work_tile in scheduler:
        stage_QKV_via_TMA(work_tile, pipeline_k, pipeline_v)
else:
    for work_tile in scheduler:
        wait_for_Q_if_needed()
        init_softmax_state()
        for n_block in chosen_order(work_tile):
            S = QK_tile_gemm(n_block)           # FP32 accumulator
            S = apply_score_mod_if_any(S)
            S = apply_masks(S)
            P_tile, row_scale = online_softmax(S)
            acc_O = rescale(acc_O, row_scale)
            acc_O += PV_tile_gemm(P_tile, n_block)
        acc_O = normalize_with_final_row_sum(acc_O)
        store_O_and_optional_LSE(acc_O)
```

## Per-Frame State Summary
| Frame | Before | After |
|---|---|---|
| A | raw torch tensors | compiled callable selected/created |
| B | CuTe tensors + scalars | launch params + TMA descriptors + scheduler params |
| D | CTA started | producer and consumer roles activated |
| E | no staged KV slots | K/V pipeline stages filled for tile |
| F/G | accumulators uninitialized/partial | `acc_O` fully accumulated + normalized |
| K | output still in regs/smem | global `O` + optional `LSE` committed |

## Notes on Lowest-Level Boundary
The trace reaches the generated DSL kernel boundary at:
- `cute.compile(...)` in [interface.py#L498](../thirdparty/flash-attention/flash_attn/cute/interface.py#L498)
- `@cute.kernel` in [flash_fwd.py#L1530](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py#L1530)

From there, CuTe/MLIR lowers into backend GPU code. At source level in this repo, the deepest directly inspectable runtime control remains the kernel+helper DSL shown above.
