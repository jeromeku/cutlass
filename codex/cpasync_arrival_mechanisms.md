# CPAsync / TMA Arrival Mechanisms In `grouped_gemm`

## Process note

- Read `CPASYNC.md` to scope the question.
- Traced the producer and consumer paths in `grouped_gemm.py`, then followed the barrier implementation into `thirdparty/quack/quack/pipeline.py` and CuTeDSL's `pipeline/helpers.py`, `pipeline/sm90.py`, and `cute/arch/mbar.py`.
- Cross-checked the implementation against the PTX ISA docs for `mbarrier` and `cp.async.mbarrier.arrive`.
- Delegation: spawned one explorer agent to independently verify the `1 + num_load_A_threads` accounting and the role of `.noinc`.

## Code map

- [Mainloop pipeline setup](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L1436)
- [A gather load path](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L550)
- [Producer loop: TMA B, gather A, commit](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L1878)
- [Consumer wait/release on the same pipeline](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L2109)
- [Mixed TMA + cp.async pipeline wrapper](../thirdparty/quack/quack/pipeline.py#L54)
- [Generic TMA pipeline behavior](../python/CuTeDSL/cutlass/pipeline/sm90.py#L433)
- [How a TMA barrier arrive is lowered](../python/CuTeDSL/cutlass/pipeline/helpers.py#L237)
- [`cp.async.mbarrier.arrive.noinc` wrapper](../python/CuTeDSL/cutlass/cute/arch/mbar.py#L255)

## Key Functions Index

| Function | File | Purpose |
|----------|------|---------|
| `load_A_gather` | `thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py` | Issues per-thread `cp.async` loads for gathered A tiles |
| `PipelineTmaCpAsync.create` | `thirdparty/quack/quack/pipeline.py` | Builds a full/empty mbarrier pipeline whose producer side combines one TMA producer with many `cp.async` producer threads |
| `PipelineTmaCpAsync.producer_acquire` | `thirdparty/quack/quack/pipeline.py` | Waits for an empty stage and does the single TMA-side full-barrier arrive |
| `PipelineTmaCpAsync.producer_cpasync_commit` | `thirdparty/quack/quack/pipeline.py` | Associates the already-issued `cp.async` work with the stage barrier using `.noinc` |
| `MbarrierArray.arrive` | `python/CuTeDSL/cutlass/pipeline/helpers.py` | Maps a pipeline op type to the correct barrier primitive |
| `cp_async_mbarrier_arrive_noinc` | `python/CuTeDSL/cutlass/cute/arch/mbar.py` | Emits `cp.async.mbarrier.arrive.shared ... noinc=1` |

## Big picture

For the mainloop "full" barrier, a stage is usable by the consumer only when two things are both finished:

1. The barrier's pending arrival count has gone to zero.
2. The barrier's outstanding async transaction byte count has gone to zero.

That is exactly why the two code paths differ:

- **No gather A**: both A and B are TMA loads, so one TMA-side `arrive_and_expect_tx(...)` can account for the whole stage.
- **Gather A**: B is still TMA, but A is loaded by many per-thread `cp.async` instructions, so the stage needs one TMA arrival **plus** one future arrival for every `cp.async` producer thread.

## 1. What the pipeline initializes

CuTeDSL initializes each stage mbarrier with `arrive_count = producer_group.size` in [`MbarrierArray.__init__`](../python/CuTeDSL/cutlass/pipeline/helpers.py#L167). For TMA producers, a pipeline "arrive" lowers to `mbarrier.arrive.expect_tx` in [`MbarrierArray.arrive`](../python/CuTeDSL/cutlass/pipeline/helpers.py#L267).

That means the producer group size is not cosmetic. It is the number of arrivals the stage is expected to observe before the phase can complete.

## 2. Non-gather path: pure TMA accounting

When `is_A_gather` is false, the setup is:

- [`producer_group = CooperativeGroup(Thread)`](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L1443), so the full barrier is initialized with arrival count `1`.
- [`tx_count = sizeof(A tile) + sizeof(B tile)`](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L1427).
- The pipeline class is plain [`PipelineTmaAsync`](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L1448).

At runtime:

1. [`producer_acquire`](../python/CuTeDSL/cutlass/pipeline/sm90.py#L518) waits for the empty stage, then performs a single `arrive_and_expect_tx(tx_count)`.
2. The producer warp issues the TMA copy for B and the TMA copy for A, both tied to the same barrier in [`grouped_gemm.py`](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L1886) and [`grouped_gemm.py`](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L1918).
3. [`producer_commit`](../python/CuTeDSL/cutlass/pipeline/sm90.py#L540) is a noop because TMA completion is already tracked by the barrier transaction count.

So the full barrier for that stage is:

- pending arrivals: `1 -> 0` from the single TMA-side arrive
- tx bytes: `A_bytes + B_bytes -> 0` as the two TMA operations retire

No extra producer-side barrier operation is needed.

## 3. Gather path: one TMA arrival plus many `cp.async` arrivals

When `is_A_gather` is true, the setup changes in three important ways.

### 3.1 A is no longer TMA

`A_tiled_copy` is built from [`cpasync.CopyG2SOp`](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L2991), and [`load_A_gather`](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L551) issues those per-thread `cute.copy(...)` calls in [`grouped_gemm.py`](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L594) and [`grouped_gemm.py`](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L604).

### 3.2 The stage's static TMA tx count now covers only B

In the gather case, the pipeline is created with:

- [`producer_group = CooperativeGroup(Thread, 1 + num_load_A_threads)`](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L1437)
- [`tx_count = sizeof(B tile)`](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L1425)
- [`pipeline_class = PipelineTmaCpAsync`](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L1442)

That is the first clue to the design:

- the **single extra `1`** is for the TMA-side barrier arrive
- the **`num_load_A_threads`** term is for the `cp.async` producers that will also arrive on the same barrier
- only **B bytes** are known to the TMA `arrive_and_expect_tx(...)` path up front

### 3.3 Only the TMA warp does the explicit TMA arrive

`PipelineTmaCpAsync.producer_acquire` differs from plain `PipelineTmaAsync` here:

- it still waits on the empty barrier
- but it guards the full-barrier arrive behind `is_tma_warp` in [`pipeline.py`](../thirdparty/quack/quack/pipeline.py#L147)

That is necessary because `sync_object_full.arrive(...)` is still a TMA-style [`arrive_and_expect_tx`](../python/CuTeDSL/cutlass/pipeline/helpers.py#L267). If every gather-load thread did that call, the barrier would over-count both arrivals and TMA transaction bytes.

So on each stage:

1. The single TMA warp performs one `arrive_and_expect_tx(B_bytes)`.
2. That same warp issues the B TMA copy in [`grouped_gemm.py`](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L1886).
3. All gather-load threads issue their A-side `cp.async` copies in [`load_A_gather`](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L551).
4. All participating gather-load threads then call [`producer_cpasync_commit`](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L1930).

## 4. What `noinc` means here

[`producer_cpasync_commit`](../thirdparty/quack/quack/pipeline.py#L155) emits [`cp.async.mbarrier.arrive.shared ..., noinc=1`](../python/CuTeDSL/cutlass/cute/arch/mbar.py#L255).

For this kernel, `noinc` means:

- the `cp.async` operation is attached to the stage mbarrier
- its completion will still contribute one asynchronous arrive-on for that thread on the stage barrier
- the barrier tracks completion of the thread's prior `cp.async` operations through that future arrive-on, not through `tx-count`
- but the instruction does **not** add a new arrival to the barrier's expected pending-arrival count at commit time

That matches the PTX requirement for `.noinc`: the barrier must already have been initialized as if those `cp.async` arrivals are expected.

And that is exactly what this kernel does by choosing:

- `producer_group.size = 1 + num_load_A_threads`

So the stage is pre-accounted for:

- `1` arrival from the TMA-side `arrive_and_expect_tx(B_bytes)`
- `num_load_A_threads` future arrivals from the gathered-A `cp.async` producers

## 5. Why `noinc` is necessary in this design

It is necessary because the pipeline has chosen a **static producer-count model**.

The full barrier is initialized with all producer arrivals already included:

- one TMA producer arrival
- one arrival per `cp.async` producer thread

If the kernel used plain `cp.async.mbarrier.arrive` instead of `.noinc`, each `cp.async` commit would add another expected arrival dynamically. That would double-account the gathered-A producers relative to the pipeline's initialized `arrive_count`, and the stage could not drain in the intended `1 + num_load_A_threads` pattern.

So the combination is deliberate:

- `CooperativeGroup(Thread, 1 + num_load_A_threads)` says "this stage expects one TMA arrival and `num_load_A_threads` cp.async arrivals"
- `producer_acquire(..., is_tma_warp)` ensures the TMA-style `arrive_and_expect_tx` happens exactly once
- `producer_cpasync_commit(... noinc)` lets the `cp.async` path contribute its completion to the same barrier **without** inflating the expected-arrival count again

## 6. The clean mental model

Use this mental model for one stage:

### No gather A

- expected arrivals at init: `1`
- explicit producer arrive: `1 x TMA arrive_and_expect_tx(A_bytes + B_bytes)`
- async producers attached to barrier: `A TMA + B TMA`
- result: one producer arrival, all bytes tracked by TMA

### Gather A

- expected arrivals at init: `1 + num_load_A_threads`
- explicit producer arrive: `1 x TMA arrive_and_expect_tx(B_bytes)`
- async producers attached to barrier:
  - `B` via TMA `complete_tx(B_bytes)`
  - `A` via `num_load_A_threads` future per-thread `cp.async` arrive-ons
- extra barrier op after issuing A copies: `num_load_A_threads x cp.async.mbarrier.arrive.noinc`
- result: one TMA arrival plus one pre-accounted `cp.async` arrival per load thread

## 7. Direct answer to the question

`noinc` is necessary here because the barrier's expected producer arrivals are **already** baked into the pipeline at construction time through `producer_group.size = 1 + num_load_A_threads`.

In the gather-A path, the kernel wants the gathered-A `cp.async` completions to satisfy those pre-existing producer arrivals, not to create new ones. So it uses `cp.async.mbarrier.arrive.noinc`:

- **with gather A**: pre-account the `cp.async` producers in the barrier count, then use `.noinc`
- **without gather A**: there are no A-side `cp.async` producers at all, so plain TMA `arrive_and_expect_tx` is sufficient

## Sources

- [grouped_gemm.py](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py)
- [pipeline.py](../thirdparty/quack/quack/pipeline.py)
- [CuTeDSL `pipeline/helpers.py`](../python/CuTeDSL/cutlass/pipeline/helpers.py)
- [CuTeDSL `pipeline/sm90.py`](../python/CuTeDSL/cutlass/pipeline/sm90.py)
- [CuTeDSL `cute/arch/mbar.py`](../python/CuTeDSL/cutlass/cute/arch/mbar.py)
- <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier>
- <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-cp-async-mbarrier-arrive>
