# `is_A_gather` full-mbarrier trace

Process:
- Traced the `is_A_gather` mainloop path in `thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py`.
- Followed the pipeline helpers into `thirdparty/quack/quack/pipeline.py`, `python/CuTeDSL/cutlass/pipeline/helpers.py`, and `python/CuTeDSL/cutlass/cute/arch/mbar.py`.
- Cross-checked the field-level state transitions against the NVIDIA PTX ISA `mbarrier` and `cp.async.mbarrier.arrive` sections.
- Delegation: none.

Let:
- `E = 1 + num_load_A_threads`
- `B = tma_copy_bytes = size_in_bytes(b_dtype, b_smem_layout)`

For the full barrier of one mainloop stage in the `is_A_gather` path:

1. `mbarrier.init`
   - Setup chooses `producer_group.size = E` and `tx_count = B` in
     `thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py:1436-1459`.
   - `MbarrierArray.__init__` sets `arrive_count = cg.size` and calls `mbarrier_init`
     in `python/CuTeDSL/cutlass/pipeline/helpers.py:177-196`.
   - `mbarrier_init` lowers to `nvvm.mbarrier_init_shared` in
     `python/CuTeDSL/cutlass/cute/arch/mbar.py:27-42`.
   - State after init: `phase=0`, `pending=E`, `expected=E`, `tx=0`.

2. TMA-side `arrive.expect_tx` for B
   - `PipelineTmaCpAsync.producer_acquire` only lets the TMA warp call
     `sync_object_full.arrive(...)` in `thirdparty/quack/quack/pipeline.py:130-152`.
   - For `PipelineOp.TmaLoad`, `MbarrierArray.arrive` lowers to `arrive_and_expect_tx`
     in `python/CuTeDSL/cutlass/pipeline/helpers.py:237-333`.
   - `mbarrier_arrive_and_expect_tx` lowers to NVVM `ARRIVE_EXPECT_TX` in
     `python/CuTeDSL/cutlass/cute/arch/mbar.py:54-91`.
   - State change: `phase` unchanged, `pending: E -> E-1`, `expected` unchanged at `E`,
     `tx: 0 -> B`.

3. B TMA launch
   - The TMA copy is issued with `tma_bar_ptr=producer_get_barrier(...)` in
     `thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py:1886-1894`.
   - The underlying SM90 bulk-copy form carries `mbarrier::complete_tx::bytes` in
     `include/cute/arch/copy_sm90_tma.hpp:1442-1451`.
   - No immediate state change at issue time; the barrier-visible effect happens on completion.

4. A `cp.async` launches
   - Gather A is loaded by per-thread `cute.copy(...)` calls in
     `thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py:578-604`,
     reached from `thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py:1904-1916`.
   - These do not mutate the full barrier yet.

5. `cp.async.mbarrier.arrive.noinc` for A
   - After issuing A copies, every participating gather producer thread calls
     `producer_cpasync_commit(...)` in
     `thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py:1930-1931`.
   - That lowers to `cp_async_mbarrier_arrive_noinc(...)` in
     `thirdparty/quack/quack/pipeline.py:154-159`, then
     `python/CuTeDSL/cutlass/cute/arch/mbar.py:254-268`.
   - Immediate state change: none to `phase`, `pending`, `expected`, or `tx`.
   - Effect: each thread registers one deferred asynchronous arrive-on to occur when its
     prior `cp.async` operations complete. Because `.noinc` is used, the pending count is
     not incremented at commit time; the future decrement is already accounted for by init.

6. Implicit A completion arrivals
   - When one producer thread's prior gather `cp.async` operations complete, the system performs
     one arrive-on on the same full barrier.
   - State change per completed producer thread: `pending -= 1`; `expected` unchanged; `tx`
     unchanged; `phase` unchanged unless this was the last outstanding condition for the phase.

7. Implicit B `complete_tx`
   - When the B TMA finishes, the system performs `complete_tx(B)` on the same barrier.
   - State change: `tx: B -> 0`; `pending` unchanged; `expected` unchanged; `phase` unchanged
     unless pending has already reached `0`.

8. Phase completion
   - The phase completes only when both conditions hold: `pending == 0` and `tx == 0`.
   - At that instant the barrier transitions atomically to the next phase and reinitializes
     `pending` from `expected`.
   - State change: `phase: p -> p ^ 1`, `pending: 0 -> E`, `expected: E`, `tx: 0`.

9. Consumer-side operations
   - `consumer_try_wait` / `consumer_wait` in
     `thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py:2101-2123`
     only read the full barrier through
     `python/CuTeDSL/cutlass/pipeline/sm90.py:241-260`,
     `python/CuTeDSL/cutlass/pipeline/helpers.py:342-350`,
     and `python/CuTeDSL/cutlass/cute/arch/mbar.py:135-181`.
   - They do not mutate full-barrier state.
   - `consumer_release` in
     `thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py:2132-2144`
     goes to `sync_object_empty.arrive(...)` in
     `python/CuTeDSL/cutlass/pipeline/sm90.py:262-264`, so it does not touch the full barrier.
