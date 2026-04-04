# Gather-A Full `mbarrier` State Walkthrough

## Process note

- Re-read the PTX `mbarrier` and `cp.async.mbarrier.arrive` descriptions, then traced the gather-A producer path in `grouped_gemm.py`.
- Followed the full-barrier operations into `thirdparty/quack/quack/pipeline.py` and CuTeDSL's `pipeline/helpers.py` / `cute/arch/mbar.py`.
- Corrected one point from the earlier note: in this path, A-side `cp.async.mbarrier.arrive.noinc` affects the barrier through future asynchronous arrive-on operations, not through `tx-count`. The only `tx-count` contribution on this full barrier is B's TMA transfer.
- Delegation: spawned one explorer agent to independently sanity-check the gather-A barrier timeline.

## Code map

- [Mainloop pipeline setup for gather-A](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L1425)
- [Gather-A producer loop](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L1878)
- [A gather load implementation](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L551)
- [Mixed TMA + `cp.async` producer barrier ops](../thirdparty/quack/quack/pipeline.py#L130)
- [Full barrier initialization and TMA arrive lowering](../python/CuTeDSL/cutlass/pipeline/helpers.py#L166)
- [`mbarrier.arrive.expect_tx` wrapper](../python/CuTeDSL/cutlass/cute/arch/mbar.py#L54)
- [`cp.async.mbarrier.arrive.noinc` wrapper](../python/CuTeDSL/cutlass/cute/arch/mbar.py#L255)

## Key Functions Index

| Function | File | Purpose |
|----------|------|---------|
| `MbarrierArray.__init__` | `python/CuTeDSL/cutlass/pipeline/helpers.py` | Initializes each stage barrier with the producer-group arrival count |
| `MbarrierArray.arrive` | `python/CuTeDSL/cutlass/pipeline/helpers.py` | Lowers TMA producer arrival to `mbarrier.arrive.expect_tx` |
| `PipelineTmaCpAsync.producer_acquire` | `thirdparty/quack/quack/pipeline.py` | Performs the single TMA-side full-barrier arrive for B |
| `load_A_gather` | `thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py` | Issues the gathered A `cp.async` loads |
| `PipelineTmaCpAsync.producer_cpasync_commit` | `thirdparty/quack/quack/pipeline.py` | Emits `cp.async.mbarrier.arrive.noinc` once per A-loading thread |

## The PTX state we care about

For one full-barrier stage, I will describe the state as:

`(phase, expected, pending, tx)`

where:

- `phase` is the barrier's internal PTX phase number
- `expected` is the expected arrival count for the next phase
- `pending` is the current phase's remaining arrival count
- `tx` is the current phase's `tx-count`

For this gather-A path, define:

- `N = num_load_A_threads`
- `B = size_in_bytes(B tile)`
- `P = 1 + N`

Important: this is the **full barrier's PTX state**, not the software `PipelineState.phase` token used by the pipeline object for stage bookkeeping.

Because this code never uses `mbarrier.arrive_drop`, the barrier's `expected` count stays constant at `P` across phases.

## PTX rules that matter here

From the PTX ISA:

- `mbarrier.init(count)` sets phase to `0`, expected arrivals to `count`, pending arrivals to `count`, and on Hopper the `tx-count` starts at `0`.
- `mbarrier.arrive.expect_tx(txCount)` performs `expect_tx(txCount)` first, then an arrive-on with count `1`.
- `expect_tx` increases `tx-count` by `txCount`.
- An arrive-on decrements `pending` by its count.
- The current phase completes only when `pending == 0` **and** `tx == 0`.
- Completion atomically advances the barrier to the next phase and resets `pending = expected`.
- `cp.async.mbarrier.arrive.noinc` does **not** increment `pending` immediately. Instead, when all prior `cp.async` operations from that thread complete, the system later performs one arrive-on with count `1`.

## 1. Initialization

Gather-A setup chooses:

```python
if const_expr(self.is_A_gather):
    tma_copy_bytes = cute.size_in_bytes(self.b_dtype, b_smem_layout)
    mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread, 1 + self.num_load_A_threads
    )
```

from [grouped_gemm.py](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L1425).

That means the full barrier is initialized for:

- one TMA-side producer arrival for B
- `N` future `cp.async` arrivals from the A-loading threads

The actual init happens in:

```python
self.arrive_count = self.cg.size
...
cute.arch.mbarrier_init(self.get_barrier(index), self.arrive_count)
```

from [helpers.py](../python/CuTeDSL/cutlass/pipeline/helpers.py#L181).

So immediately after `mbarrier.init` for this stage:

- `phase = 0`
- `expected = P = 1 + N`
- `pending = P = 1 + N`
- `tx = 0`

State:

`S0 = (0, P, P, 0)`

## 2. The only explicit TMA-side barrier mutation: `mbarrier.arrive.expect_tx(B)`

The producer loop is:

```python
mainloop_pipeline.producer_acquire(
    mainloop_producer_state, peek_ab_empty_status, is_tma_warp=is_tma_warp
)

if is_tma_warp:
    cute.copy(
        tma_atom_b,
        ...,
        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_producer_state),
    )

self.load_A_gather(...)
mainloop_pipeline.producer_cpasync_commit(mainloop_producer_state)
```

from [grouped_gemm.py](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L1880).

`producer_acquire` is:

```python
if_generate(
    is_tma_warp,
    lambda: self.sync_object_full.arrive(state.index, self.producer_mask, loc=loc, ip=ip),
)
```

from [pipeline.py](../thirdparty/quack/quack/pipeline.py#L149).

And for a TMA producer, `arrive(...)` lowers to:

```python
self.arrive_and_expect_tx(index, self.tx_count, loc=loc, ip=ip)
```

from [helpers.py](../python/CuTeDSL/cutlass/pipeline/helpers.py#L267), which emits:

```python
nvvm.mbarrier_txn(... kind=nvvm.MBarrierTxnKind.ARRIVE_EXPECT_TX ...)
```

from [mbar.py](../python/CuTeDSL/cutlass/cute/arch/mbar.py#L84).

This is the only explicit barrier mutation done synchronously by the producer for B. Its PTX effect is:

1. `expect_tx(B)`:
   `tx: 0 -> B`
2. `arrive-on(count=1)`:
   `pending: P -> P - 1 = N`

The phase does **not** complete here because `pending = N > 0` and `tx = B > 0`.

State after `mbarrier.arrive.expect_tx(B)`:

`S1 = (0, P, N, B)`

## 3. Issuing the B TMA does not immediately change the barrier

The TMA B launch in [grouped_gemm.py](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L1886) receives the same barrier pointer, so the hardware knows which barrier should receive the eventual completion signal.

But at **issue time**, there is no additional barrier mutation beyond `S1`.

So immediately after launching the TMA:

`S2 = (0, P, N, B)`

The change comes later, at TMA completion, as an implicit `complete_tx(B)` event.

## 4. Issuing the gathered A `cp.async` loads also does not immediately change the barrier

`load_A_gather` issues one or more `cp.async` loads per participating thread:

```python
cute.copy(A_g2s_thr_copy, mA_cur_copy, tAsA[...])
```

from [grouped_gemm.py](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L594) and [grouped_gemm.py](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L604).

Those `cp.async` instructions do **not** mutate the full barrier when they are issued. They only become associated with the barrier when each thread later executes `cp.async.mbarrier.arrive.noinc`.

So after all A `cp.async` instructions have been issued, but before the commit-to-barrier step:

`S3 = (0, P, N, B)`

## 5. `cp.async.mbarrier.arrive.noinc`: no immediate count change

After issuing A's `cp.async` operations, each participating A-loading thread executes:

```python
cute.arch.cp_async_mbarrier_arrive_noinc(self.producer_get_barrier(state, ...))
```

from [pipeline.py](../thirdparty/quack/quack/pipeline.py#L155) and [mbar.py](../python/CuTeDSL/cutlass/cute/arch/mbar.py#L255).

PTX semantics here are the crucial point:

- the instruction makes the barrier track **all prior `cp.async` operations from that thread**
- with `.noinc`, it does **not** increment `pending` at commit time
- it schedules one future asynchronous arrive-on when that thread's prior `cp.async` operations have all completed

So the immediate barrier state after all `N` threads execute `cp.async.mbarrier.arrive.noinc` is still:

`S4 = (0, P, N, B)`

There is no immediate `pending` or `tx` change here.

If `.noinc` had **not** been used, each thread would first increment `pending` by `1`, and the later asynchronous arrive-on would decrement it by `1`, for a zero-net change. That is exactly what this kernel does **not** want, because `pending` was already pre-initialized to include those `N` future arrivals.

## 6. As A's `cp.async` work completes, the system performs future arrive-on operations

Now the A-side completions start to matter.

For each A-loading thread, once **all prior `cp.async` operations from that thread** have completed, the system performs one arrive-on with count `1` on the full barrier.

Each such completion changes the state like this:

- `pending: x -> x - 1`
- `tx` unchanged
- `phase` unchanged unless this was the last missing condition for phase completion

So after `k` of the `N` A-loading threads have fully completed their tracked `cp.async` work:

`Sk = (0, P, N - k, B_or_0)`

where `B_or_0` depends only on whether B's TMA has completed yet.

In particular:

- after the first A-thread completion: `(0, P, N - 1, B_or_0)`
- after the last A-thread completion: `(0, P, 0, B_or_0)`

The key point is that A contributes to the barrier only through **pending arrivals**, not through `tx-count`.

## 7. When B's TMA completes, the hardware performs `complete_tx(B)`

Because the TMA-side barrier operation was `arrive.expect_tx(B)`, B's completion eventually performs the matching implicit `complete_tx(B)`.

That changes the state like this:

- `tx: B -> 0`
- `pending` unchanged
- `phase` unchanged unless this was the last missing condition for phase completion

So if B completes before all A arrivals have happened:

- before: `(0, P, r, B)` for some `r > 0`
- after: `(0, P, r, 0)`

If all A arrivals were already done first:

- before: `(0, P, 0, B)`
- after: phase completion happens immediately

## 8. The phase completes when the **second** condition finally reaches zero

The barrier completes phase 0 only when both conditions are true:

- `pending == 0`
- `tx == 0`

That means the last event can be either:

### Case A: B finishes last

State just before B's completion:

`(0, P, 0, B)`

Then implicit `complete_tx(B)` makes:

- `tx: B -> 0`
- phase 0 complete
- atomic phase transition
- `pending` reset to `expected = P`

Final state:

`(1, P, P, 0)`

### Case B: the last A-thread completion arrives last

State just before the final A-thread's asynchronous arrive-on:

`(0, P, 1, 0)`

Then that final arrive-on makes:

- `pending: 1 -> 0`
- phase 0 complete
- atomic phase transition
- `pending` reset to `expected = P`

Final state:

`(1, P, P, 0)`

So no matter which side finishes last, the next visible barrier state is the same:

`Sdone = (1, P, P, 0)`

## 9. What `mbarrier.try_wait` / `wait` do here

The consumer side later does:

```python
return cute.arch.mbarrier_try_wait(self.get_barrier(index), phase, ...)
...
cute.arch.mbarrier_wait(self.get_barrier(index), phase, ...)
```

from [helpers.py](../python/CuTeDSL/cutlass/pipeline/helpers.py#L342).

These are observation operations, not producer-side state mutations:

- they test or wait for the specified phase to be complete
- they do not decrement `pending`
- they do not change `tx`

The important PTX rule is that at least one successful `try_wait` / `test_wait` must observe the phase completion before the barrier is used again in the next phase.

## 10. Condensed instruction-by-instruction timeline

For one gather-A stage:

1. `mbarrier.init(P)`
   state: `(0, P, P, 0)`
2. `mbarrier.arrive.expect_tx(B)` from the single TMA warp
   state: `(0, P, P - 1, B)` = `(0, P, N, B)`
3. issue TMA B
   state: unchanged
4. issue gathered-A `cp.async` instructions
   state: unchanged
5. `N x cp.async.mbarrier.arrive.noinc`
   immediate state: unchanged
6. implicit asynchronous arrive-on from A-thread completions
   each one: `pending -= 1`
7. implicit `complete_tx(B)` from B completion
   `tx -= B`
8. whichever of steps 6 or 7 satisfies the second missing completion condition triggers:
   phase `0 -> 1`, `pending := expected = P`, `tx := 0`

## Direct answer

For the gather-A full barrier, the only thing that ever changes `tx-count` is B's TMA path:

- `mbarrier.arrive.expect_tx(B)` sets `tx = B`
- TMA completion performs the matching `complete_tx(B)` and brings `tx` back to `0`

The A-side `cp.async` path does **not** contribute to `tx-count` here. Instead:

- each A-loading thread executes `cp.async.mbarrier.arrive.noinc`
- that schedules one future arrive-on for that thread when its prior `cp.async` work is done
- each such future arrive decrements `pending` by `1`

That is why the barrier is initialized with `P = 1 + num_load_A_threads`: one pending arrival is reserved for the TMA-side `arrive.expect_tx`, and the other `num_load_A_threads` pending arrivals are reserved for the later asynchronous arrive-on operations generated by A's `cp.async` completions.

## Sources

- [grouped_gemm.py](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py)
- [pipeline.py](../thirdparty/quack/quack/pipeline.py)
- [CuTeDSL `pipeline/helpers.py`](../python/CuTeDSL/cutlass/pipeline/helpers.py)
- [CuTeDSL `cute/arch/mbar.py`](../python/CuTeDSL/cutlass/cute/arch/mbar.py)
- <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier>
- <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-cp-async-mbarrier-arrive>
