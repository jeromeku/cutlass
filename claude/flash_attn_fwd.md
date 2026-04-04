# Flash Attention SM90 Forward Kernel — Deep Trace

**Date:** 2026-02-14
**Scope:** `FlashAttentionForwardSm90` — launcher, data structures, TMA pipeline, WGMMA inner loop, softmax, epilogue
**Source root:** `thirdparty/flash-attention/flash_attn/cute/`

---

## 1. Architecture Overview

```
torch._inductor (CuteDSL backend)
  └─ CuteDSLTemplate.generate()
       └─ CuteDSLBenchmarkRequest.__init__()
            └─ cute.compile(FlashAttentionForwardSm90, ...)  ← JIT compile
                 └─ FlashAttentionForwardSm90.__call__()     ← host launcher
                      ├─ build TMA descriptors (Q/K/V/O)
                      ├─ choose TileScheduler
                      ├─ compute grid/block dims
                      └─ FlashAttentionForwardSm90.kernel()  ← GPU kernel
                           ├─ [Producer warp] TMA load K/V → smem (pipelined)
                           ├─ [Producer warp] TMA load Q   → smem
                           └─ [Consumer warpgroups]
                                ├─ WGMMA: S = Q × Kᵀ
                                ├─ score_mod / mask_mod
                                ├─ online softmax (Dao-Milakov)
                                ├─ fp16/bf16 convert P
                                ├─ WGMMA: O += P × V
                                └─ epilogue: normalize + TMA store O + write LSE
```

The SM90 kernel uses **producer/consumer warpgroup separation** (one producer warp + 3 consumer warpgroups = 384 threads total) with a **TMA async pipeline** for KV loading, and **WGMMA** (warpgroup MMA) for matrix multiplication.

---

## 2. Entry Point: `_flash_attn_fwd`

**File:** [interface.py#L94](../thirdparty/flash-attention/flash_attn/cute/interface.py#L94)

`_flash_attn_fwd` is the top-level Python launcher called from `FlashAttnFunc.forward` (the `torch.autograd.Function`) and directly from `flex_flash_attention.py` in Inductor.

### 2.1 Input Validation and Normalization

```python
# interface.py:140
q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
```
`maybe_contiguous` calls `.contiguous()` only if `stride(-1) != 1`, ensuring the innermost (head_dim) dimension is contiguous — required for 128-bit vectorized loads.

```python
# interface.py:216-218
alignment = 16 // q.element_size()          # fp16 → 8, bf16 → 8
assert head_dim % alignment == 0            # head_dim must be 8-aligned
```
Each 128-bit TMA transaction loads 8 fp16 elements; the alignment check enforces this at entry.

### 2.2 GQA Pack Decision

```python
# interface.py:224-225
if pack_gqa is None:
    pack_gqa = qhead_per_kvhead > 1         # auto-enable for GQA (e.g., Llama: 8 Q per KV)
```
**Pack GQA** fuses multiple Q heads into a single tile on the M dimension, eliminating redundant K/V loads. When `qhead_per_kvhead=8` and `tile_m=128`, each tile covers 128/(8)=16 token positions across all 8 Q heads simultaneously.

The stride manipulation that makes this work:

```python
# flash_fwd.py:1340-1354  (inside __call__)
shape_Q_packed = (
    (self.qhead_per_kvhead, mQ.shape[0]),   # (q_per_kv, seqlen_q) merged into M dim
    mQ.shape[1],                            # seqlen_k (unused for Q)
    mK.shape[2],                            # num_kv_heads
    *mQ.shape[3:],                          # head_dim
)
stride_Q_packed = (
    (mQ.stride[2], mQ.stride[0]),           # head stride, token stride interleaved
    mQ.stride[1],
    mQ.stride[2] * self.qhead_per_kvhead,   # stride across KV heads
    *mQ.stride[3:],
)
```
This is a **zero-copy view**: the same physical Q buffer is accessed with a new CuTe layout that maps `(q_head_idx, token_idx)` → flat address via `q_head_idx * stride[2] + token_idx * stride[0]`. No data movement; only metadata changes.

### 2.3 Compile Cache Key

```python
# interface.py:376-405
compile_key = (
    dtype,                # cutlass.Float16 or cutlass.BFloat16
    head_dim,             # e.g. 128
    head_dim_v,           # same or different (e.g., DeepSeek: 128/192)
    qhead_per_kvhead,     # 1 for MHA, >1 for GQA/MQA
    causal,               # bool
    score_mod_hash,       # hash of score_mod callable, or False
    mask_mod_hash,        # hash of mask_mod callable, or False
    use_block_sparsity,
    block_sparse_broadcast_pattern,
    aux_tensor_metadata,
    lse is None,          # whether to compute LSE
    cu_seqlens_q is None, # fixed vs. varlen mode
    cu_seqlens_k is None,
    seqused_q is None,
    seqused_k is None,
    page_table is not None,
    window_size_left is not None,
    window_size_right is not None,
    learnable_sink is not None,
    m_block_size,         # 128 (default SM90)
    n_block_size,         # 128 (non-causal) or 192 (hdim=128 non-causal)
    q_stage,              # 1 for SM90
    num_threads,          # 384 for SM90
    is_split_kv,
    pack_gqa,
    compute_capability,   # 9
    page_size not in [None, 128],
    q_subtile_factor,
)
```
This key is a 28-element tuple hashed for the `_flash_attn_fwd.compile_cache` dict. A **cache miss** triggers `cute.compile(fa_fwd, q_tensor, k_tensor, ...)` which JIT-compiles the Python kernel to PTX/CUBIN the first time; subsequent calls with the same config just call the cached compiled function.

### 2.4 TMA Tensor Construction

```python
# interface.py:424-432 (on cache miss)
q_tensor, k_tensor, v_tensor, o_tensor = [
    to_cute_tensor(t) for t in (q, k, v, out if not is_split_kv else out_partial)
]
```

`to_cute_tensor` ([cute_dsl_utils.py](../thirdparty/flash-attention/flash_attn/cute/cute_dsl_utils.py)) converts a PyTorch tensor to a `cutlass.cute.Tensor` by calling `from_dlpack`, annotating the tensor with CuTe strides and alignment info. The result is a `cute.Tensor` with:
- `.iterator`: a typed pointer (e.g., `cutlass.Float16*`) to device memory
- `.layout`: a `cute.Layout` encoding shape and strides (e.g., `(B,S,H,D):(S*H*D, H*D, D, 1)`)

### 2.5 FlashAttentionForwardSm90 Instantiation

```python
# interface.py:447-467
fa_fwd = FlashAttentionForwardSm90(
    dtype,                      # cutlass.Float16 or BFloat16
    head_dim,                   # e.g., 128
    head_dim_v,
    qhead_per_kvhead,
    is_causal=causal,
    is_local=local,
    pack_gqa=pack_gqa,
    tile_m=m_block_size,        # 128
    tile_n=n_block_size,        # 128 or 192
    num_stages=2,               # double-buffered KV pipeline
    num_threads=num_threads,    # 384 = 1 producer warp + 3 MMA warpgroups
    Q_in_regs=False,
    intra_wg_overlap=True,      # overlap QK and PV GEMMs
    mma_pv_is_rs=True,          # P lives in registers (not smem)
    mask_mod=mask_mod,
    score_mod=score_mod,
    has_aux_tensors=aux_tensors is not None,
)
```

Key SM90-specific flags:
- **`intra_wg_overlap=True`**: enables overlapping `QKᵀ` GEMM of tile N with `PV` GEMM of tile N-1 inside the same warpgroup
- **`mma_pv_is_rs=True`**: P (attention weights after softmax, fp16) lives in registers (RS = Register × Smem). WGMMA can consume operand A from registers directly, saving a smem write/read round-trip
- **`num_stages=2`**: two-stage async pipeline for K and V; while consumer processes stage 0, producer loads stage 1

### 2.6 cute.compile

```python
# interface.py:498-518
_flash_attn_fwd.compile_cache[compile_key] = cute.compile(
    fa_fwd,
    q_tensor, k_tensor, v_tensor, o_tensor,
    lse_tensor,
    softmax_scale,
    current_stream,
    cu_seqlens_q_tensor,
    cu_seqlens_k_tensor,
    ...
    options="--enable-tvm-ffi",
)
```
`cute.compile` inspects the Python `@cute.jit`-decorated `__call__` method of `fa_fwd`, traces it with the provided tensor type signatures, and emits MLIR → LLVM IR → PTX → CUBIN. The Python objects passed here serve as **type exemplars** for JIT specialization; on subsequent calls the compiled function takes the actual tensors.

---

## 3. Data Structures

### 3.1 Shared Memory Layout

**File:** [flash_fwd.py#L1162](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py#L1162)

For SM90, smem layouts use **swizzled** `ComposedLayout`s (base swizzle + tiled shape):

```python
def _get_smem_layout_atom(self):
    # warpgroup.make_smem_layout_atom wraps the hopper swizzle layout
    # that prevents bank conflicts for WGMMA 128B transactions
    sQ_layout_atom = warpgroup.make_smem_layout_atom(
        sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, self.dtype, self.tile_hdim),
        self.dtype,
    )
```

The SM90 WGMMA instruction reads from smem using 128-byte aligned, swizzled accesses. The swizzle pattern XOR-maps column indices so that simultaneous accesses from 4-warp WGMMA groups land on different smem banks. Without swizzle, 16 threads accessing consecutive columns all hit bank 0 (16-way conflict).

**SharedStorage struct** (SM90):
```
SharedStorageQKV {
    mbar_ptr:   Int64[2]            ← mbarriers for Q and O
    mbar_ptr_K: Int64[num_stages*2] ← mbarriers for K pipeline stages
    mbar_ptr_V: Int64[num_stages*2] ← mbarriers for V pipeline stages
    sV:   fp16[tile_n * tile_hdimv * num_stages]   ← 2-stage V buffer
    sQ:   fp16[tile_m * tile_hdim]                  ← Q (reused as sO in epilogue)
    sK:   fp16[tile_n * tile_hdim * num_stages]    ← 2-stage K buffer
    sP:   fp16[tile_m * tile_n]    ← P smem (only if mma_pv_is_rs=False)
}
```

Q and O **share the same smem buffer** (`sO = storage.sQ.get_tensor(sO_layout...)`). This is safe because: Q is read during the K-loop and never accessed again; the epilogue (O write) happens after all QK/PV GEMMs complete. Total smem ≈ 2×(128×128×2B)×2 stages for KV + 128×128×2B for Q = 64 KB + 32 KB = 96 KB, within H100's 228 KB/SM limit.

### 3.2 TMA Descriptors

```python
# flash_fwd.py:1399-1426
tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
    gmem_tiled_copy_Q,         # CopyBulkTensorTileG2SOp
    mQ,                        # global tensor (CuTe layout of full Q matrix)
    self.sQ_layout,            # smem tile shape and swizzle
    (self.tile_m, self.tile_hdim),   # tile extents
)
```

`make_tiled_tma_atom` creates a TMA descriptor that encodes:
- Source pointer + stride info (multi-dimensional)
- Tile dimensions and swizzle
- OOB clamping mode

The TMA hardware can execute a 2D (or higher) async copy in a **single PTX instruction** (`cp.async.bulk.tensor.2d.shared::cluster.global`), completely freeing the issuing warp while DMA hardware copies data. The kernel only needs to signal a **transaction barrier** and then `mbarrier.wait`.

### 3.3 Thread / Warpgroup Layout (SM90)

```
Thread index space (384 threads):
┌─────────────────────────────────────────────────────┐
│  Producer warp (warp 0, threads 0–31)               │
│    - Issues TMA loads for K and V (and Q if TMA_Q)  │
│    - Arrives on producer-side pipeline barriers     │
├─────────────────────────────────────────────────────┤
│  MMA Warpgroup 1 (warps 1–4, threads 32–159)        │
│    - WGMMA: S = Q × Kᵀ (QK GEMM)                   │
│    - WGMMA: O += P × V (PV GEMM)                   │
│    - Softmax, epilogue                              │
├─────────────────────────────────────────────────────┤
│  MMA Warpgroup 2 (warps 5–8, threads 160–287)       │
│    (same as WG1, working on different M rows)       │
├─────────────────────────────────────────────────────┤
│  MMA Warpgroup 3 (warps 9–12, threads 288–415? no…) │
│    Actually: num_threads=384 = 1*32 + 3*128 - wait  │
│    With num_mma_warp_groups=2: 32+256=288            │
│    With num_mma_warp_groups=3: 32+384=416... varies  │
└─────────────────────────────────────────────────────┘
```

The actual count: from `flash_fwd.py:1289-1292`:
```python
self.num_mma_threads = tiled_mma_qk.size          # = tile_m/64 * 128 = 256 for tile_m=128
self.num_threads_per_warp_group = 128
self.num_mma_warp_groups = self.num_mma_threads // 128   # = 2
self.num_threads = 128 * (2 + 1)                  # = 384: 2 MMA WGs + 1 producer WG
```
For `tile_m=128`, there are **2 MMA warpgroups** (256 MMA threads) + **1 producer warpgroup** (128 threads), with only the first 32 of the producer warpgroup doing the actual TMA issue. Total = 384 threads per block.

### 3.4 WGMMA Tiled MMA

```python
# flash_fwd.py:1187-1207
tiled_mma_qk = sm90_utils_basic.make_trivial_tiled_mma(
    self.dtype, self.dtype,
    warpgroup.OperandMajorMode.K,   # A operand K-major (row-major Q)
    warpgroup.OperandMajorMode.K,   # B operand K-major (row-major K transposed)
    Float32,                        # accumulator dtype
    atom_layout_mnk=(self.tile_m // 64, 1, 1),   # (2,1,1) for tile_m=128
    tiler_mn=(64, self.tile_n),      # WGMMA base tile: 64×tile_n
)
tiled_mma_pv = sm90_utils_basic.make_trivial_tiled_mma(
    self.dtype, self.dtype,
    warpgroup.OperandMajorMode.K,   # A (P): row-major
    warpgroup.OperandMajorMode.MN,  # B (Vᵀ): column-major (N-major)
    Float32,
    atom_layout_mnk=(self.tile_m // 64, 1, 1),
    tiler_mn=(64, self.tile_hdimv),
    a_source=warpgroup.OperandSource.RMEM,  # P from registers (mma_pv_is_rs=True)
)
```

**WGMMA** (SM90 Warpgroup Matrix Multiply-Accumulate) operates on 64×tile_n chunks. With `tile_m=128` and 2 warpgroups: each warpgroup handles 64 rows. The `atom_layout_mnk=(2,1,1)` means 2 atoms stacked in M (covering 128 rows total).

For the QK GEMM:
- A = Q tile from smem: `(64, head_dim)` in row-major swizzled layout
- B = Kᵀ tile from smem: `(head_dim, tile_n)` in row-major swizzled layout
- C = fp32 accumulator: `(64, tile_n)` in registers (partitioned per thread)

For the PV GEMM:
- A = P tile (fp16 attention weights): from **registers** when `mma_pv_is_rs=True`
- B = Vᵀ tile from smem: `(tile_n, head_dim_v)` column-major
- C = fp32 accumulator O: in registers

---

## 4. Frame-by-Frame: `FlashAttentionForwardSm90.__call__` (Host Launcher)

**File:** [flash_fwd.py#L1246](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py#L1246)

### Frame 1: Tensor layout transposition

```python
# flash_fwd.py:1280-1286
mQ, mO = [layout_utils.select(t, [1, 3, 2, 0]) for t in (mQ, mO)]
mK, mV = [layout_utils.select(t, [1, 3, 2, 0]) for t in (mK, mV)]
mLSE = layout_utils.select(mLSE, [2, 1, 0])
```

**Before:** `mQ.layout = (B, S_q, H, D) : (S_q*H*D, H*D, D, 1)` — PyTorch BSHD layout
**After:** `mQ.layout = (S_q, D, H, B) : (H*D, 1, D, S_q*H*D)` — CuTe "SDHB" permutation

This transposition is **zero-copy** (just metadata). The kernel addresses Q as `mQ[seq_idx, :, head_idx, batch_idx]`, which gives the D-dimensional slice for a given (seq, head, batch) triple. This matches the CuTe convention for the global memory tile: the first two modes (S,D) = (M,K) are the dimensions the MMA iterates over.

### Frame 2: Determine thread/warpgroup counts

```python
# flash_fwd.py:1289-1313
self.num_mma_threads = tiled_mma_qk.size              # 256 for tile_m=128
self.num_mma_warp_groups = 256 // 128                 # = 2
self.num_threads = 128 * (2 + 1)                      # = 384
self.num_producer_threads = 32                        # one warp for TMA
self.num_Q_load_threads = 256                         # MMA threads load Q if not TMA
self.use_tma_Q = True   # SM90 always uses TMA for Q (unless pack_gqa misalignment)
self.use_tma_O = True   # SM90 uses TMA for O (unless varlen)
```

Register budgeting (crucial for occupancy):
```python
self.num_mma_regs = 240    # for 2 MMA warpgroups
self.num_producer_regs = 24
```
H100 has 64K 32-bit registers per SM. With `240 * 256 + 24 * 128 = 64,512` registers, this fits in one SM-wide register file, enabling **maximum occupancy** (≥1 block/SM).

### Frame 3: TMA descriptor creation

```python
# flash_fwd.py:1399-1426
tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
    CopyBulkTensorTileG2SOp(), mQ, self.sQ_layout, (tile_m, tile_hdim))
tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
    CopyBulkTensorTileG2SOp(), mK, sK_layout_single_stage, (tile_n, tile_hdim), 1)
tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
    CopyBulkTensorTileG2SOp(), mV, sV_layout_single_stage, (tile_n, tile_hdimv), 1)
tma_atom_O, tma_tensor_O = cpasync.make_tiled_tma_atom(
    CopyBulkTensorTileS2GOp(), mO, self.sO_layout, (tile_m, tile_hdimv))
```

`tma_tensor_Q` is a special "TMA tensor" — it wraps the TMA descriptor alongside the global tensor; when passed to the kernel it becomes a **descriptor argument** whose physical representation is a pointer to the 128-byte TMA descriptor resident in SMEM after prefetch.

### Frame 4: TileScheduler selection

```python
# flash_fwd.py:1427-1434
if mCuSeqlensQ is not None or mSeqUsedQ is not None:
    TileScheduler = SingleTileVarlenScheduler   # variable-length sequences
else:
    TileScheduler = (
        SingleTileScheduler                     # simple: blockIdx.x = m_block
        if not self.is_causal or self.is_local
        else SingleTileLPTScheduler             # LPT = Longest-Processing-Time first
    )
```

**SingleTileScheduler**: `block_idx → (m_block, head_idx, batch_idx)` via integer division. Grid = `(ceil(seqlen_q/tile_m), num_head_kv, batch_size)`.

**SingleTileLPTScheduler** (for causal): Applies **L2 swizzle** to improve KV cache reuse. For causal attention, earlier M-blocks (higher sequence indices) have more K/V iterations — LPT orders blocks longest-first so fast SMs process short blocks while slow SMs are still working on long ones. The swizzle groups `swizzle = floor(50MB / (seqlen_k * (hdim + hdim_v) * elem_size))` heads per section, keeping each section's K/V in L2 cache ([tile_scheduler.py#L270](../thirdparty/flash-attention/flash_attn/cute/tile_scheduler.py#L270)).

### Frame 5: softmax_scale encoding

```python
# flash_fwd.py:1458-1467
if self.score_mod is None:
    softmax_scale_log2 = softmax_scale * LOG2_E   # = scale / log(2) for exp2()
    softmax_scale = None
else:
    softmax_scale_log2 = LOG2_E                    # just change-of-base
    softmax_scale = softmax_scale                  # applied explicitly before score_mod
```

The kernel uses `exp2(x * softmax_scale_log2)` instead of `exp(x * softmax_scale)` because CUDA's `__expf2` is faster than `__expf`. When `score_mod` is present, the `softmax_scale` is applied to the raw logits **before** `score_mod`, and `softmax_scale_log2` carries only `log₂(e)`.

### Frame 6: Kernel launch

```python
# flash_fwd.py:1487-1528
self.kernel(
    tma_tensor_Q, tma_tensor_K, tma_tensor_V, tma_tensor_O, mLSE,
    mCuSeqlensQ, mCuSeqlensK, mSeqUsedQ, mSeqUsedK,
    tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O,
    softmax_scale_log2, softmax_scale, ...,
    tile_sched_params, TileScheduler,
    SharedStorage, aux_tensors, fastdiv_mods,
).launch(
    grid=grid_dim,
    block=[self.num_threads, 1, 1],   # [384, 1, 1]
    stream=stream,
    min_blocks_per_mp=1,
)
```

Note: SM90 kernel omits explicit `smem=` parameter — smem size is encoded in the `SharedStorage.size_in_bytes()` call inside the launch descriptor. `min_blocks_per_mp=1` sets the occupancy hint to the driver.

---

## 5. Frame-by-Frame: `FlashAttentionForwardSm90.kernel` (GPU)

**File:** [flash_fwd.py#L1530](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py#L1530)

### Frame 7: Thread identification and TMA prefetch

```python
# flash_fwd.py:1569-1574
warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
if warp_idx == 0:
    for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O):
        if const_expr(tma_atom is not None):
            cpasync.prefetch_descriptor(tma_atom)
```

`cpasync.prefetch_descriptor(tma_atom)` issues `prefetch.tensormap.L2.shared::cta [addr]` — prefetches the 128-byte TMA descriptor into L2 cache. This runs in warp 0 (producer warp) while all other warps proceed concurrently. The TMA descriptor is needed by the producer warp when it later calls `cp.async.bulk.tensor`.

### Frame 8: Mbarrier / pipeline init

```python
# flash_fwd.py:1576-1610
smem = cutlass.utils.SmemAllocator()
storage = smem.allocate(SharedStorage)

# Q mbarrier: 1 arrival (producer warp TMA issue)
if not self.use_tma_Q:
    cute.arch.mbarrier_init(mbar_ptr_Q, self.num_Q_load_threads)

# K and V pipelines (2-stage double buffer)
pipeline_k = pipeline.PipelineTmaAsync.create(
    barrier_storage=storage.mbar_ptr_K.data_ptr(),
    num_stages=2,
    producer_group=CooperativeGroup(Agent.Thread),     # any thread can produce
    consumer_group=CooperativeGroup(Agent.Thread, 8),  # 8 warps = 2 WGs consume
    tx_count=self.tma_copy_bytes["K"],                 # bytes per stage (preset)
    defer_sync=True,
)
```

`PipelineTmaAsync` manages a circular ring of `num_stages=2` mbarriers for K (and separately for V). Each stage has a mbarrier with a **transaction count** (`tx_count`) pre-programmed: when TMA completes, it decrements the barrier to zero automatically, signaling consumers. Consumers call `pipeline_k.consumer_wait()` which spins until the barrier reaches zero.

### Frame 9: Smem tensor setup

```python
# flash_fwd.py:1615-1629
sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
sVt = layout_utils.transpose_view(sV)          # (tile_hdimv, tile_n, stages) for WGMMA-B
sO = storage.sQ.get_tensor(sO_layout, ...)     # reuses sQ smem buffer
```

`sVt` is critical: WGMMA for PV requires B in **column-major** (MN-major in CuTe terms). `transpose_view(sV)` creates a CuTe tensor pointing to the same smem bytes but with transposed layout: what was `(tile_n, tile_hdimv)` becomes `(tile_hdimv, tile_n)`. The WGMMA instruction sees a column-major B matrix, computing `P × V` as `P × (Vᵀ)ᵀ`.

### Frame 10: WGMMA tiled MMA setup (per-thread partitioning)

```python
# flash_fwd.py: (from agent output ~1860-1890)
thr_mma_qk = tiled_mma_qk.get_slice(tidx)
thr_mma_pv = tiled_mma_pv.get_slice(tidx)

# QK GEMM registers
tSrQ = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))    # Q fragment
tSrK = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK[..., 0]))

# PV GEMM registers
tOrVt = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sVt[..., 0]))
acc_O = cute.make_fragment(thr_mma_pv.partition_shape_C((tile_m, hdimv)), Float32)
acc_O.fill(0.0)

# P register fragment (fp16, lives in registers for mma_pv_is_rs=True)
tOrP = cute.make_fragment_like(thr_mma_pv.partition_A(sQ), dtype)
```

`get_slice(tidx)` computes which rows/columns of the tile this thread owns. For WGMMA with 128 threads per warpgroup, each thread owns `(tile_m // 128) * (tile_n // 8)` elements of the fp32 accumulator — for `tile_m=64, tile_n=128`: 1×16 = 16 fp32 values per thread.

### Frame 11: Producer loop — TMA issuing (warp 0)

From `flash_fwd.py` (producer section, ~line 1700 from agent output):
```python
# [warp_idx == 0: producer warp]
if warp_idx == 0:
    # Issue TMA load for Q (once, for this m_block)
    if const_expr(self.use_tma_Q):
        pipeline_q.producer_acquire(...)    # wait for mbar_ptr_Q to be available
        copy(tma_atom_Q, tma_tensor_Q[..., m_block], sQ)
        pipeline_q.producer_commit(...)     # signal TMA transaction count

    # Prologue: pre-load first KV stages before main loop
    for s in range(num_stages):
        pipeline_k.producer_acquire(kv_producer_state)
        copy(tma_atom_K, tma_tensor_K[..., n_block_max - 1 - s], sK[..., s])
        pipeline_k.producer_commit(kv_producer_state, ...)

        pipeline_v.producer_acquire(kv_producer_state)
        copy(tma_atom_V, tma_tensor_V[..., n_block_max - 1 - s], sV[..., s])
        pipeline_v.producer_commit(kv_producer_state, ...)
        kv_producer_state.advance()

    # Main loop: keep the pipeline full
    for n_block in range(n_block_min, n_block_max - num_stages):
        pipeline_k.producer_acquire(kv_producer_state)
        copy(tma_atom_K, tma_tensor_K[..., n_block], sK[..., stage])
        pipeline_k.producer_commit(kv_producer_state, ...)
        # same for V
        kv_producer_state.advance()
```

Each `copy(tma_atom_K, src, dst)` issues a single PTX `cp.async.bulk.tensor.2d.shared::cta.global` that DMA-copies an entire `(tile_n, head_dim)` tile = 128×128×2B = 32KB from HBM to smem **asynchronously**. The warp proceeds immediately; TMA completion is signaled by decrementing the associated mbarrier.

### Frame 12: Consumer main loop — QK GEMM + softmax + PV GEMM

**File:** [flash_fwd.py#L1939-2174](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py#L1939)

The consumer section (MMA warpgroups) processes tiles in a while-loop driven by the TileScheduler:

```python
# flash_fwd.py:1939-1993
while work_tile.is_valid_tile:
    m_block, head_idx, batch_idx, _ = work_tile.tile_idx
    seqlen = SeqlenInfoCls(batch_idx)       # actual seqlen for this batch element
    mask_fn = partial(mask.apply_mask, ...)  # causal / local / mask_mod
    score_mod_fn = partial(self.apply_score_mod, ...) if score_mod else None

    n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
    O_should_accumulate = False
```

`get_n_block_min_max` computes the KV block range this Q tile needs to attend to. For causal: `n_block_max = ceil((m_block+1)*tile_m / tile_n)`, `n_block_min = 0`. For local: `n_block_min = max(0, ceil((m_block*tile_m - window_left) / tile_n))`.

#### Frame 12a: `intra_wg_overlap` path (default, `self.intra_wg_overlap=True`)

The key innovation: **within one N-block iteration**, the QK GEMM of block N and the PV GEMM of block N-1 are **overlapped**:

```python
# flash_fwd.py:2300-2357 (mma_one_n_block_intrawg_overlap)
def mma_one_n_block_intrawg_overlap(self, smem_pipe_read, n_block, ...):
    smem_pipe_read_v = smem_pipe_read.clone()    # remember current V stage
    smem_pipe_read.advance()                     # advance K read pointer to next stage

    # Wait for K of stage N+1
    pipeline_k.consumer_wait(smem_pipe_read, ...)
    self.warp_scheduler_barrier_sync()           # sync warpgroups before QK

    # ─── Issue QK GEMM with wg_wait=-1 (start but don't wait) ───
    acc_S = mma_qk_fn(B_idx=smem_pipe_read.index, wg_wait=-1)
    #
    # ─── While QK is in-flight, issue PV GEMM of previous block ───
    pipeline_v.consumer_wait(smem_pipe_read_v, ...)
    mma_pv_fn(B_idx=smem_pipe_read_v.index, wg_wait=-1)  # O += P × V
    #
    self.warp_scheduler_barrier_arrive()
    warpgroup.wait_group(1)      # wait for QK to finish (PV may still be in-flight)
    pipeline_k.consumer_release(smem_pipe_read)

    # Apply score_mod and mask_mod to acc_S
    if score_mod_fn: score_mod_fn(acc_S, n_block=n_block, ...)
    if mask_fn:      mask_fn(acc_S=acc_S, n_block=n_block)

    # Online softmax on acc_S (row_max, row_sum update)
    row_scale = softmax.online_softmax(acc_S, check_inf=True)

    warpgroup.wait_group(0)      # now wait for PV to finish too
    pipeline_v.consumer_release(smem_pipe_read_v)

    # Convert fp32 acc_S → fp16 P (in registers)
    utils.cvt_f16(tOrP_acc, tOrP_cur)   # PTX-level vectorized fp32→fp16 cvt

    # Rescale O: O = O * row_scale (compensate for new row max)
    softmax.rescale_O(acc_O, row_scale)
```

**Overlap explanation**: `wg_wait=-1` issues WGMMA but does **not** insert `wgmma.wait_group`. `wg_wait=0` inserts `wgmma.wait_group 0` which stalls until **all** pending WGMMA groups complete. By issuing both QK and PV before any wait, the SM's MMA units execute both back-to-back without stalling on the pipeline.

#### Frame 12b: score_mod application

```python
# flash_fwd.py:2370-2402
def apply_score_mod(self, thr_mma_qk, batch_idx, head_idx, m_block, acc_S, n_block, ...):
    cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
    cS = cute.domain_offset((m_block * self.tile_m, n_block * self.tile_n), cS)
    tScS = thr_mma_qk.partition_C(cS)   # each thread gets its (q_idx, kv_idx) coordinates
    apply_score_mod_inner(
        acc_S, tScS, self.score_mod,
        batch_idx, head_idx, softmax_scale,
        self.vec_size, self.qk_acc_dtype,
        aux_tensors, fastdiv_mods, ...
    )
```

`apply_score_mod_inner` ([softmax.py](../thirdparty/flash-attention/flash_attn/cute/softmax.py)) iterates over each element of `acc_S`:
```python
for i in range(size(acc_S)):
    q_idx = tScS[i][0]           # actual sequence index for this element
    kv_idx = tScS[i][1]
    acc_S[i] = score_mod(acc_S[i], batch_idx, head_idx, q_idx, kv_idx, aux_tensors)
```
This runs in **registers** — no smem traffic. The user's `score_mod` callable (which was traced into an IR callable by Inductor) is inlined here directly.

#### Frame 12c: Online softmax (Dao-Milakov algorithm)

**File:** [softmax.py](../thirdparty/flash-attention/flash_attn/cute/softmax.py)

```python
def online_softmax(self, acc_S, is_first=False, check_inf=True):
    # 1. Find row max of this tile's S values
    row_max_new = reduce_max_rows(acc_S)   # per-row max across tile_n elements

    # 2. Compute correction factor for existing O accumulator
    row_scale = exp2(row_max - row_max_new)   # = exp(old_max - new_max) in base-2
    #    row_scale[i] ∈ (0, 1]: rescale factor to correct O for new max

    # 3. Update running row_max
    row_max = row_max_new

    # 4. Compute exp2(S - row_max) in-place
    for i in range(size(acc_S)):
        acc_S[i] = exp2(acc_S[i] * softmax_scale_log2 - row_max)
        # = exp2(s_ij / log(2) - m_i)   [ effectively: exp(s_ij - m_i * log(2)) ]

    # 5. Update row_sum
    row_sum = row_sum * row_scale + reduce_sum_rows(acc_S)
    # The row_scale corrects the old contributions; add new tile's contributions

    return row_scale    # caller uses this to rescale O: O = O * row_scale + P @ V
```

**State after each tile:** `row_max` = running max, `row_sum` = running softmax denominator (Σ exp(s_ij - m_i)).

In `rescale_O`: `acc_O = acc_O * row_scale` (broadcast row-wise). This implements the **numerically stable online softmax** of [Milakov & Gimelshein 2018], achieving FlashAttention's O(1)-memory property.

**Final normalization** (after all N blocks):
```python
# flash_fwd.py:2150-2151
row_scale = softmax.finalize(sink_val=sink_val)   # compute 1/row_sum
softmax.rescale_O(acc_O, row_scale)               # O = O / row_sum
# acc_O now holds the correctly normalized attention output O = softmax(QKᵀ) V
```

### Frame 13: Epilogue — O store + LSE write

**File:** [flash_fwd.py#L331](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py#L331)

```python
def epilogue(self, acc_O, lse, mO, mLSE, sO, seqlen, ...):
    # 1. Convert acc_O (fp32) → rO (fp16/bf16)
    rO = cute.make_fragment_like(acc_O, self.dtype)
    rO.store(acc_O.load().to(self.dtype))           # vectorized fp32→fp16 in registers

    # 2. Wait for all threads to finish reading V
    cute.arch.barrier(NamedBarrierFwd.Epilogue, ...)

    # 3. Copy rO from registers → smem sO (reusing sQ's smem buffer)
    smem_copy_atom_O = utils.get_smem_store_atom(arch=90, ...)  # STMATRIX atom
    smem_thr_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma_pv).get_slice(tidx)
    cute.copy(smem_copy_atom_O, smem_thr_copy_O.retile(rO), smem_thr_copy_O.partition_D(sO))

    # 4. TMA store: smem sO → global mO
    if self.use_tma_O:
        cute.arch.fence_view_async_shared()     # ensure smem writes visible to TMA
        cute.arch.barrier_arrive(NamedBarrierFwd.Epilogue, ...)
        gO = cute.local_tile(mO_cur, (tile_m, hdimv), (m_block, 0))
        store_O, _, _ = copy_utils.tma_get_copy_fn(tma_atom_O, 0, ..., sO, gO)
        if warp_idx == 4:  # epilogue warp in producer warpgroup
            cute.arch.barrier(NamedBarrierFwd.Epilogue, ...)   # wait for smem ready
            store_O()                                          # issue TMA store
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)   # wait for store done

    # 5. Write LSE to global memory
    if mLSE is not None:
        lse_log2 = row_sum (= log₂(Σ exp(s-m)) + m*log₂(e) = lse in nats / log(2))
        # Write per-row lse to mLSE[batch, head, seq_pos]
```

The epilogue uses a **named barrier** (`NamedBarrierFwd.Epilogue`) that only involves the epilogue threads (256 MMA threads + 32 producer threads for TMA store). Non-epilogue threads (producer warp except warp_idx==4) skip straight to `tile_scheduler.advance_to_next_work()`.

The LSE written to `mLSE` is the **log-sum-exp in base 2**: `lse_log2 = log₂(Σ exp(s_ij)) = log₂(row_sum * exp(row_max))`. The backward pass uses this to recompute the softmax without rematerializing the full attention matrix.

---

## 6. score_mod and mask_mod: Execution Flow Deep Dive

This section traces how user-provided `score_mod` and `mask_mod` callables thread through the entire kernel, from the Python interface down to per-element register operations.

### 6.1 What They Are

**`score_mod`** — a Python callable with signature:
```python
def score_mod(score, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, *aux_tensors) -> score
```
Receives the raw attention logit `S[b,h,q,k] = Q[b,h,q,:] · K[b,h,k,:]` (fp32, **after** scaling by `1/√d`) and returns a modified logit. The returned value replaces the original in the accumulator before softmax. Example: ALiBi adds a positional bias `-(q_idx - kv_idx) * slope`.

**`mask_mod`** — a Python callable with signature:
```python
def mask_mod(batch_idx, head_idx, q_idx, kv_idx, seqlen_info, *aux_tensors) -> bool
```
Returns `True` if the (q,k) pair should be **kept** (attend), `False` if it should be **masked** (set to −∞). Pure index function — no float value needed. Example: causal mask returns `q_idx >= kv_idx`.

### 6.2 Compile-Time Embedding

Both are passed as `cutlass.Constexpr` to the kernel class constructor:

```python
# interface.py:463-464
FlashAttentionForwardSm90(
    ...,
    mask_mod=mask_mod,    # Constexpr — baked into kernel binary
    score_mod=score_mod,  # Constexpr — baked into kernel binary
)
```

`cutlass.Constexpr` values are embedded at **JIT compile time** (inside `cute.compile`). The MLIR lowering sees these as compile-time constants, not runtime arguments. Each unique `(score_mod_hash, mask_mod_hash)` pair produces a distinct CUBIN. This is why the compile cache key includes `score_mod_hash` and `mask_mod_hash` — a different `score_mod` function is a different kernel.

**Consequence**: there is no runtime dispatch for these callbacks. The user's Python function is inlined into the PTX exactly once at compile time.

### 6.3 `vec_size` and Vectorization

```python
# flash_fwd.py:116-118  (FlashAttentionForwardBase.__init__)
self.vec_size: cutlass.Constexpr = getattr(
    score_mod, "__vec_size__", 1 if cutlass.const_expr(has_aux_tensors) else 2
)
```

`vec_size` controls how many logit elements are batched per `score_mod` call:
- **`vec_size=2`** (default): two consecutive `(q_idx, kv_idx)` pairs share the same KV index and differ only in that they are adjacent elements processed by the same PTX instruction. `score_mod` receives a 2-vector of scores, enabling the compiler to issue vectorized `FMUL2` and `FADD2` instructions.
- **`vec_size=1`**: forced when `aux_tensors` are present (random-access tensor reads), since gathering two different auxiliary tensor indices in one vectorized call is not guaranteed safe.
- A custom `score_mod` can declare `__vec_size__ = 4` (or any power-of-2) to request wider vectorization.

### 6.4 softmax_scale Interaction

When `score_mod` is present, the scale encoding changes:

```python
# flash_fwd.py:1459-1467 (__call__)
if self.score_mod is None:
    softmax_scale_log2 = softmax_scale * LOG2_E   # = 1/(√d * log(2))
    softmax_scale = None                           # no explicit scale in inner loop
else:
    softmax_scale_log2 = LOG2_E                    # = log₂(e): only change-of-base
    softmax_scale = softmax_scale                  # passed to apply_score_mod_inner
```

Without `score_mod`:
- Every logit is multiplied by `softmax_scale_log2` inside `online_softmax`:
  `exp2(s * scale_log2 - row_max)` = `exp(s / √d - row_max * log(2))`

With `score_mod`:
- `apply_score_mod_inner` multiplies `score * softmax_scale` **before** calling `score_mod` (line 406):
  ```python
  score_vec[j] = score_tensor[i + j] * softmax_scale   # scale first
  ```
- Then `score_mod(score_vec, ...)` transforms the scaled logit
- The result is stored back to `acc_S`
- `online_softmax` then applies only `* LOG2_E` (change of base), no additional scaling

This ensures `score_mod` always receives the logit in its "natural" domain (scaled by `1/√d`), matching the semantics users expect.

### 6.5 score_mod Execution Path (Frame-by-Frame)

#### Step 1: `apply_score_mod` wrapper (flash_fwd.py:2369)

```python
def apply_score_mod(self, thr_mma_qk, batch_idx, head_idx, m_block, acc_S, n_block, ...):
    # Build coordinate tensor: maps each accumulator element → (global_q_idx, global_kv_idx)
    cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
    cS = cute.domain_offset((m_block * self.tile_m, n_block * self.tile_n), cS)
    #    ^^ domain_offset shifts coordinates so element [r,c] of the tile
    #       maps to absolute position (m_block*tile_m + r, n_block*tile_n + c)
    tScS = thr_mma_qk.partition_C(cS)
    #    ^^ partition gives this thread's coordinates:
    #       tScS[i] = (global_q_idx, global_kv_idx) for the i-th element this thread owns
```

`thr_mma_qk.partition_C(cS)` is the critical step: it applies the same partitioning that was used to partition the fp32 accumulator `acc_S`, so `tScS[i]` is the logical (q,kv) coordinate for the i-th element of `acc_S[i]`. This is entirely compile-time — thread coordinates in the WGMMA accumulator layout are known at JIT time.

#### Step 2: `apply_score_mod_inner` (softmax.py:342)

```python
# softmax.py:404-469
for i in range(0, n_vals, vec_size):            # n_vals = elements per thread
    for j in range(vec_size):
        score_vec[j] = score_tensor[i+j] * softmax_scale  # scale in fp32

        # Extract (q_idx, kv_idx) from the coordinate tensor
        q_idx_vec[j] = floor_if_packed(index_tensor[i+j][0], qhead_per_kvhead)
        kv_idx_vec[j] = index_tensor[i+j][1]
        # For Pack-GQA: q_idx_packed = head_offset * seqlen + token_pos
        #   → q_idx_logical = q_idx_packed // qhead_per_kvhead (token position)
        #   → head_idx_for_mod = head_idx * qhead_per_kvhead + (q_idx_packed % qhead_per_kvhead)

    # Convert to SSA (MLIR Static Single Assignment form) for score_mod invocation
    score_ssa    = score_vec.load()
    kv_idx_ssa   = kv_idx_vec.load()
    q_idx_ssa    = q_idx_vec.load()
    head_idx_ssa = ...                           # scalar broadcast or per-element for Pack-GQA

    # ── CALL INTO USER'S score_mod ──
    post_mod_scores = score_mod(
        score_ssa, batch_idx_ssa, head_idx_ssa,
        q_idx=q_idx_ssa, kv_idx=kv_idx_ssa,
        seqlen_info=seqlen_info, aux_tensors=aux_args,
    )
    # score_mod is the user's @cute.jit function, inlined here at compile time

    score_vec.store(post_mod_scores)
    for j in range(vec_size):
        score_tensor[i+j] = score_vec[j]        # write back modified logits
```

**Register budget**: `score_vec`, `q_idx_vec`, `kv_idx_vec` are `cute.make_rmem_tensor` — small compile-time-sized register arrays. For `vec_size=2`, `n_vals=16` (typical): 3×2=6 extra registers per thread. No smem traffic.

**Aux tensors**: when `aux_tensors` is not None, `kv_idx_wrapped = kv_idx % seqlen_k` and `q_idx_wrapped = q_idx % seqlen_q` are computed using pre-built `FastDivmodDivisor` objects (multiply-shift trick, no integer division instruction). This prevents out-of-bounds reads when iterating tiles that extend beyond sequence length.

### 6.6 mask_mod Execution Path

`mask_mod` is called from `AttentionMask.apply_mask` ([mask.py#L128](../thirdparty/flash-attention/flash_attn/cute/mask.py#L128)) which runs in the `mask_fn` partial closure built in the consumer main loop.

#### Mask mode dispatch (compile-time, mask.py:162)

```python
if not mask_causal and not mask_local and mask_mod is None:
    # ── Path A: seqlen-only masking ──
    # Fast path: compare column indices against seqlen_k boundary
    # Uses R2P (Register-to-Predicate) bit-manipulation trick
    if mask_seqlen:
        mask_r2p(acc_S_mn, seqlenk_col_limit, arch=90)

elif not mask_causal and not mask_local and mask_mod is not None:
    # ── Path B: FlexAttention mask_mod ──
    # Per-element evaluation of user's mask function

else:  # mask_causal or mask_local
    # ── Path C: causal / sliding window ──
    # Row-dependent column limit: col_limit = q_idx + causal_offset
    mask_r2p(acc_S_mn[r, None], col_limit_right, arch=90, rank1=True)
```

All three dispatch branches are **compile-time selectors** (`const_expr`). The kernel binary for causal attention contains only Path C code; a FlexAttention kernel contains only Path B code.

#### Path B: mask_mod per-element evaluation (mask.py:189)

```python
# mask.py:189-233
for r in range_constexpr(nrow):                  # rows owned by this thread (compile-time count)
    local_row = tScS_mn[r, 0][ROW]               # local tile row index
    global_row_idx = local_row + m_block * tile_m # absolute q position
    # Pack-GQA: logical_q = global_row // qhead_per_kvhead
    # logical_head = head_idx * qhead_per_kvhead + (global_row % qhead_per_kvhead)
    row_for_mod = global_row_idx (or // qhead_per_kvhead)

    for col in range_constexpr(ncol):            # cols owned by this thread (compile-time count)
        global_col_idx = col_offset + n_block * tile_n + col_idx_local

        # Optional: wrap indices for aux_tensor OOB safety
        if wrap_aux_indices:
            row_for_mod = row_for_mod % seqlen_q
            col_for_mod = global_col_idx % seqlen_k

        # ── CALL INTO USER'S mask_mod ──
        mask_value = mask_mod(
            batch_idx_ssa, head_idx_ssa,
            q_idx_ssa, kv_idx_ssa,
            seqlen_info, aux_tensors,
        )
        # mask_value is a Boolean (SSA value)
        cond = Boolean(ssa_to_scalar(mask_value))

        # Apply: keep if True, mask if False
        if mask_seqlen:
            out_of_bounds = (row_for_seqlen >= seqlen_q) or (global_col_idx >= seqlen_k)
            if out_of_bounds:
                acc_S_mn[r, col] = -inf     # OOB always masked regardless of mask_mod
            else:
                acc_S_mn[r, col] = acc_S_mn[r, col] if cond else -inf
        else:
            acc_S_mn[r, col] = acc_S_mn[r, col] if cond else -inf
```

All iteration counts (`nrow`, `ncol`) are **compile-time constants** derived from the WGMMA accumulator layout — the compiler fully unrolls both loops. For a typical configuration (2 MMA warpgroups, `tile_m=128, tile_n=128`): each thread owns `(128/64/128*2) = 2` rows and `128/8 = 16` columns = 32 calls to `mask_mod` per thread per N-block, all fully unrolled.

#### The R2P trick (Path A and C)

`mask_r2p` ([mask.py#L16](../thirdparty/flash-attention/flash_attn/cute/mask.py#L16)) uses bit manipulation to set accumulator elements to −∞ using the SM90 R2P (Register-to-Predicate) instruction family:

```python
def mask_r2p(X, col_limit, arch=90, rank1=False):
    # Transform col_limit to R2P-friendly coordinate space
    # SM90: WGMMA acc columns are in pairs (0,1,8,9,16,17,...) not (0,1,2,3,...)
    col_limit_transformed = col_limit // 8 * 2 + min(col_limit % 8, 2)

    for s in range_constexpr(ceil_div(ncol, 24)):
        col_limit_right_s = max(col_limit_transformed - s * 24, 0)
        mask = (1 << col_limit_right_s) - 1   # bitmask: 1 for in-bounds cols
        for i in range_constexpr(min(24, ncol - s*24)):
            in_bound = Boolean(mask & (1 << i))   # compile-time boolean!
            X[r, s*24+i] = X[r, s*24+i] if in_bound else -inf
```

Since `col_limit` is a **runtime value** but the bitmask construction `(1 << col_limit_right_s) - 1` compiles to `PRMT`/`SHR` PTX instructions, the compiler generates optimal branch-free predicated store sequences. This is ~2–4× faster than a runtime loop with comparison for causal masking.

### 6.7 Ordering of score_mod and mask_mod in the Main Loop

Inside `mma_one_n_block` and `mma_one_n_block_intrawg_overlap`, the order is:

```
1. WGMMA: acc_S = Q × Kᵀ           (fp32 accumulator, raw logits)
2. score_mod(acc_S, ...)             ← FIRST: transform logits (optional)
3. mask_mod / causal / seqlen mask   ← SECOND: set masked positions to −∞
4. online_softmax(acc_S)             ← softmax on modified+masked logits
5. fp32→fp16 convert acc_S → tOrP
6. rescale_O(acc_O, row_scale)
7. WGMMA: acc_O += P × V
```

**Why score_mod before mask_mod?** Because `score_mod` may read aux_tensors at `(q_idx, kv_idx)` positions that include out-of-bounds elements (the tile may extend beyond seqlen). `mask_mod` with `mask_seqlen=True` then overwrites those positions to −∞ after the fact. If mask_mod ran first, score_mod would still overwrite the −∞ values with meaningful scores.

**Why mask_mod before softmax?** −∞ inputs to `exp2` produce 0 after exponentiation, correctly zeroing masked positions in the softmax denominator.

### 6.8 How mask_mod Enables Block-Sparse Tile Skipping

While `mask_mod` applies per-element masking within a tile, the Inductor-side block sparsity infrastructure ([flex_flash_attention.py](../thirdparty/pytorch/torch/_inductor/kernel/flex/flex_flash_attention.py)) pre-computes which entire KV tiles are fully masked at `create_block_mask()` time and passes them as `block_sparse_tensors`. The kernel then skips fully-masked tiles entirely in the N-block loop.

Inside the kernel, `consume_block_sparse_loads` ([block_sparse_utils.py](../thirdparty/flash-attention/flash_attn/cute/block_sparse_utils.py)) replaces the dense `for n_block in n_block_min..n_block_max` loop with iteration over only the non-empty tiles from the block sparse CSR lists. For tiles marked as "full" (all elements valid), `mask_mod` is also skipped entirely, saving the per-element evaluation overhead.

---

## 7. Warp Scheduler Barrier Protocol

For `use_scheduler_barrier=True` (2 MMA warpgroups, `tile_hdim ≤ 128`):

```
WG1 (warps 1-4): Processes rows [0, 63]
WG2 (warps 5-8): Processes rows [64, 127]

Each iteration of mma_one_n_block_intrawg_overlap:
  WG1.warp_scheduler_barrier_sync()   ← wait for WG2 to arrive
  WG1 issues QK GEMM (wg_wait=-1)
  WG1 issues PV GEMM (wg_wait=-1)
  WG1.warp_scheduler_barrier_arrive() ← signal WG2

  WG2.warp_scheduler_barrier_sync()   ← wait for WG1 to arrive
  WG2 issues QK GEMM (wg_wait=-1)
  WG2 issues PV GEMM (wg_wait=-1)
  WG2.warp_scheduler_barrier_arrive() ← signal WG1
```

Named barriers `WarpSchedulerWG1` and `WarpSchedulerWG2` ([named_barrier.py](../thirdparty/flash-attention/flash_attn/cute/named_barrier.py)) ensure that WGMMA operations from different warpgroups do not interleave at the hardware dispatch level, which would cause instruction-level deadlock (WGMMA requires that all warps in a warpgroup reach the same WGMMA instruction simultaneously).

---

## 7. Performance Optimization Summary

| Technique | Where | Effect |
|-----------|-------|--------|
| TMA async load K/V | Producer warp issues, consumer waits | Zero-overhead memory latency hiding; HBM bandwidth ≈ theoretical max |
| TMA async load Q | warp 0 issues once per M block | Q loaded without consuming MMA thread cycles |
| 2-stage double-buffered pipeline | `num_stages=2` K/V mbarriers | Overlaps HBM load of stage N+1 with GEMM of stage N |
| Intra-warpgroup overlap | `mma_one_n_block_intrawg_overlap` | QK GEMM of N overlaps with PV GEMM of N-1; ~2× MMA utilization |
| `mma_pv_is_rs=True` | P stays in registers | No smem write + fence + read for P; saves ~10–15% latency |
| Swizzled smem layouts | `warpgroup.make_smem_layout_atom` | Eliminates bank conflicts in 128B WGMMA smem reads |
| Q/O smem aliasing | `sO = storage.sQ.get_tensor(...)` | Saves 32KB smem; enables larger tile sizes |
| GQA packing | `pack_gqa=True` | Multiple Q heads per KV tile; reduces K/V load count by `qhead_per_kvhead` |
| LPT tile scheduling | `SingleTileLPTScheduler` (causal) | Balances SM load for causal workloads; improves SM occupancy |
| `exp2()` instead of `exp()` | `softmax_scale_log2 = scale * log₂e` | Single `__expf2` instruction vs. `__expf` (~2× faster on SM90) |
| Register-resident softmax state | `row_max`, `row_sum` in registers | Online softmax; no HBM for attention weights |
| Vectorized fp32→fp16 convert | `utils.cvt_f16` (PTX-level) | 2-element SIMD `.cvt.rn.f16x2.f32`; avoids scalar loop |

---

## 8. Key Functions Index

| Function | File | Line | Purpose |
|----------|------|------|---------|
| `_flash_attn_fwd` | [interface.py](../thirdparty/flash-attention/flash_attn/cute/interface.py) | 94 | Top-level launcher: validate inputs, build compile key, call cute.compile |
| `to_cute_tensor` | [cute_dsl_utils.py](../thirdparty/flash-attention/flash_attn/cute/cute_dsl_utils.py) | ~120 | Convert PyTorch tensor → CuTe typed tensor with layout metadata |
| `FlashAttentionForwardSm90.__init__` | [flash_fwd.py](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py) | 1150 | Store config; set `intra_wg_overlap`, `mma_pv_is_rs`, `num_stages` |
| `FlashAttentionForwardSm90.__call__` | [flash_fwd.py](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py) | 1246 | Host-side JIT: layout transpose, TMA descriptors, scheduler, launch kernel |
| `_get_smem_layout_atom` (SM90) | [flash_fwd.py](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py) | 1162 | Build swizzled WGMMA-compatible smem layouts for Q/K/V/O/P |
| `_get_tiled_mma` (SM90) | [flash_fwd.py](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py) | 1186 | Build WGMMA tiled_mma for QK (smem×smem) and PV (reg×smem) |
| `_get_shared_storage_cls` (SM90) | [flash_fwd.py](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py) | 1210 | Define smem struct with mbarriers + Q/K/V/O/P buffers |
| `FlashAttentionForwardSm90.kernel` | [flash_fwd.py](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py) | 1530 | GPU kernel: TMA prefetch, pipeline init, producer/consumer split |
| `mma_one_n_block_intrawg_overlap` | [flash_fwd.py](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py) | 2300 | Process one KV block: overlap QK(N) with PV(N-1), softmax, rescale-O |
| `mma_one_n_block` | [flash_fwd.py](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py) | 2241 | Non-overlapped fallback: QK → softmax → PV sequential |
| `first_half_block_overlap` | [flash_fwd.py](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py) | 2177 | First iteration: QK+softmax only (no prior PV to overlap with) |
| `last_half_block_overlap` | [flash_fwd.py](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py) | 2225 | Last iteration: PV only (no next QK to overlap with) |
| `apply_score_mod` | [flash_fwd.py](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py) | 2369 | Apply user score_mod callable to each element of acc_S with (b,h,q,kv) indices |
| `Softmax.online_softmax` | [softmax.py](../thirdparty/flash-attention/flash_attn/cute/softmax.py) | — | Milakov-style row-max + row-sum update; returns rescale factor |
| `Softmax.finalize` | [softmax.py](../thirdparty/flash-attention/flash_attn/cute/softmax.py) | — | Compute `1/row_sum`; store lse = `log₂(row_sum) + row_max/log₂e` |
| `epilogue` | [flash_fwd.py](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py) | 331 | acc_O→rO→sO→gmem via STMATRIX + TMA store; write LSE |
| `SingleTileLPTScheduler` | [tile_scheduler.py](../thirdparty/flash-attention/flash_attn/cute/tile_scheduler.py) | 253 | LPT tile ordering with L2 swizzle for causal workloads |
| `PipelineTmaAsync` | [pipeline.py](../thirdparty/flash-attention/flash_attn/cute/pipeline.py) | — | 2-stage TMA async pipeline with mbarrier producer/consumer protocol |

---

## 9. Code Map

| File | Role |
|------|------|
| [interface.py](../thirdparty/flash-attention/flash_attn/cute/interface.py) | Top-level launcher, compile cache, GQA pack decision |
| [flash_fwd.py](../thirdparty/flash-attention/flash_attn/cute/flash_fwd.py) | `FlashAttentionForwardBase`, `FlashAttentionForwardSm90` |
| [softmax.py](../thirdparty/flash-attention/flash_attn/cute/softmax.py) | Online softmax, `apply_score_mod_inner` |
| [pipeline.py](../thirdparty/flash-attention/flash_attn/cute/pipeline.py) | `PipelineTmaAsync`, mbarrier lifecycle |
| [tile_scheduler.py](../thirdparty/flash-attention/flash_attn/cute/tile_scheduler.py) | `SingleTileScheduler`, `SingleTileLPTScheduler`, `SingleTileVarlenScheduler` |
| [cute_dsl_utils.py](../thirdparty/flash-attention/flash_attn/cute/cute_dsl_utils.py) | `to_cute_tensor`, `to_cute_aux_tensor` |
| [block_info.py](../thirdparty/flash-attention/flash_attn/cute/block_info.py) | `BlockInfo.get_n_block_min_max` for causal/local bounds |
| [seqlen_info.py](../thirdparty/flash-attention/flash_attn/cute/seqlen_info.py) | `SeqlenInfoQK` for varlen batch offset computation |
| [pack_gqa.py](../thirdparty/flash-attention/flash_attn/cute/pack_gqa.py) | `PackGQA.load_Q`, `store_O`, `store_LSE` for multi-head packing |
| [named_barrier.py](../thirdparty/flash-attention/flash_attn/cute/named_barrier.py) | `NamedBarrierFwd` enum for warpgroup synchronization |
