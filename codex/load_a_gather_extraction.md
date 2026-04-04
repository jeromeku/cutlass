# `load_A_gather` Extraction Notes

## Recommendation

The best reusable extraction point is **not** another helper inside `grouped_gemm.py`. The clean boundary is a **standalone pre-gather CuTe kernel** in a new sibling module, launched from the existing Python orchestration layer before the grouped GEMM runs.

Suggested new file:

- `thirdparty/sonic-moe/sonicmoe/functional/a_gather.py`

Why this boundary is better:

- `load_A_gather` currently has a single call site inside the producer-side mainloop of `HopperWgmma_MoE_kernel`.
- The helper is tightly coupled to shared-memory staging, thread partitioning, and pipeline state prepared by the enclosing grouped GEMM kernel.
- `forward.py` and `backward.py` already own tensor conversion, stream selection, and `cute.compile(...)` caching for standalone CuTe kernels.
- `topk_softmax.py` is already the local pattern for a self-contained CuTe kernel module.

## Likely Integration Points

### 1. New standalone kernel module

- `thirdparty/sonic-moe/sonicmoe/functional/a_gather.py`

This should own a reusable CuTe kernel/class that materializes:

- forward / activation-grad form: `A_grouped[TK, K] = A[x_gather_idx, :]`
- weight-grad form: `A_grouped[M, TK] = A[:, x_gather_idx]`

### 2. Launch/orchestration sites

- `thirdparty/sonic-moe/sonicmoe/functional/forward.py`
- `thirdparty/sonic-moe/sonicmoe/functional/backward.py`

These files are the natural launch points because they already:

- convert torch tensors to CuTe tensors
- manage streams
- compile/cache CuTe modules

### 3. Downstream GEMM config flip points

- `thirdparty/sonic-moe/sonicmoe/functional/moe_config.py`

If the gather is materialized ahead of time, the grouped GEMM wrappers that currently instantiate `HopperWgmma_MoE_kernel(..., is_A_gather=True, ...)` are the places that would switch to the normal contiguous-A path.

## Why `load_A_gather` Is Hard To Extract Alone

`load_A_gather` closes over or depends on:

- `self.tile_M`, `self.tile_K`
- `self.tma_warp_id`
- `self.compute_weight_gradient`
- `self.num_load_A_threads`
- `self.prefetch_token_idx_size`
- `self.universal_copy_bits`
- `self.elem_pointer(...)`

It also depends on caller-prepared objects that are not generic on their own:

- `tmAIdx`, produced by `prefetch_gather_idx_for_A_when_vary_M(...)`
- `sAIdx_prefetch`, filled by `prefetch_gather_idx_for_A_when_vary_K(...)`
- `A_g2s_thr_copy`, built from `_make_tiled_copy_2D(...)`
- `tAsA`, a stage-specific destination slice into shared memory
- `tApA`, a predicate tensor derived from partitioned copy coordinates
- `M_offset`, `K_offset`, and `token_group_size`, which come from the grouped GEMM tile scheduler

## Tricky Dependencies

### Weight-gradient path

The weight-gradient case is the least standalone:

- it uses `sAIdx_prefetch` in shared memory
- it relies on `NamedBarrierGemm.Prolog`
- it requires `prefetch_token_idx_size` to be a multiple of both `tile_K` and `num_load_A_threads`

### Copy-layout coupling

The source and destination copy layout is implicit:

- `_make_tiled_copy_2D(...)` chooses the thread/value layout from `mA` layout, tile shape, and thread count
- `load_A_gather` assumes the resulting `A_g2s_thr_copy` and `tAsA` slicing scheme match exactly

### Path-dependent tensor shapes

The helper is really two different routines hidden behind one branch:

- non-weight-grad gathers rows from `mA`
- weight-grad gathers columns from transposed `mA`

The indexing/predicate tensors differ across those paths.

## Practical Conclusion

If the goal is a **truly reusable standalone CuTe kernel**, extract a separate `A` pre-gather kernel into a new file and invoke it from `forward.py` / `backward.py`, then run grouped GEMM on the materialized contiguous tensor.

If the goal is **minimum code churn with no extra global-memory materialization**, the smallest safe extraction unit is:

- `prefetch_gather_idx_for_A_when_vary_M(...)`
- `prefetch_gather_idx_for_A_when_vary_K(...)`
- `load_A_gather(...)`

and it should move together into a new helper file such as:

- `thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm_gather.py`

That option is reusable only inside grouped GEMM-style kernels, not as a standalone pre-processing kernel.

## Process Note

- I traced the single call site of `load_A_gather`, then followed the helper setup for `tmAIdx`, `sAIdx_prefetch`, `A_g2s_thr_copy`, `tAsA`, and `tApA`.
- I checked the orchestration layers (`forward.py`, `backward.py`, `moe_config.py`) to find where a standalone CuTe kernel could be compiled and launched cleanly.
- I also checked existing kernel organization (`topk_softmax.py`, `reduction_over_k_gather.py`, and QuACK gather-A wrappers) to choose a module boundary consistent with the repo.
- No delegation was used. Tools used: `rg`, `sed`, and `nl`.
