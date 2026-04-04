# Mbarrier State Walkthrough for `PipelineTmaCpAsync`

How the mbarrier synchronizes mixed TMA (B matrix) + cp.async (gathered A matrix) producers
in the MoE grouped GEMM kernel (`thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py`).

## Mbarrier internal state

The mbarrier has four pieces of internal state (from the [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier)):

| Field | Meaning |
|-------|---------|
| `phase` | Current phase bit (0 or 1). Flips when the barrier completes. |
| `pendingCount` | Arrivals still needed. Set by `mbarrier.init`. Decremented by each arrival. |
| `expectedTxCount` | Total async-transaction bytes expected this phase. Accumulated by `arrive_and_expect_tx`. |
| `completedTxCount` | Bytes that have arrived from async operations (TMA copies linked to the barrier). |

**Completion condition** (both must hold):
1. `pendingCount == 0`
2. `completedTxCount >= expectedTxCount`

When both are satisfied, the phase flips and the barrier resets for the next phase.

## Concrete values for the gather-A path

From `grouped_gemm.py`:

```python
# line 1438: producer_group size
arrive_count = 1 + num_load_A_threads  # = 1 + 128 = 129

# line 1426: only B tile bytes (A uses cp.async, not TMA)
tma_copy_bytes = size_in_bytes(b_dtype, b_smem_layout)  # = B_bytes

# line 1458: passed to pipeline.create as tx_count
tx_count = tma_copy_bytes  # = B_bytes
```

The `1` accounts for the single TMA-side arrival (`arrive_and_expect_tx`).
The `128` accounts for the 128 cp.async producer threads that will each arrive via `cp.async.mbarrier.arrive.noinc`.

## Step-by-step state trace

### Step 0: `mbarrier.init(barrier, 129)`

Called during `MbarrierArray.__init__` → `mbarrier_init` ([helpers.py:224](../python/CuTeDSL/cutlass/pipeline/helpers.py#L224)).
Warp 0 initializes each stage's barrier.

| phase | pendingCount | expectedTxCount | completedTxCount |
|-------|-------------|-----------------|-----------------|
| 0 | **129** | 0 | 0 |

### Step 1: `mbarrier.arrive.expect_tx(barrier, B_bytes)`

`PipelineTmaCpAsync.producer_acquire` ([pipeline.py:131-152](../thirdparty/quack/quack/pipeline.py#L131)):
- All producer threads wait on the empty barrier.
- Only the TMA warp calls `sync_object_full.arrive()`.
- `arrive()` dispatches to `arrive_and_expect_tx` ([helpers.py:269](../python/CuTeDSL/cutlass/pipeline/helpers.py#L269)) because `op_type == TmaLoad`.
- Inside `arrive_and_expect_tx` ([helpers.py:330](../python/CuTeDSL/cutlass/pipeline/helpers.py#L330)): **`elect_one`** ensures exactly **1 thread** executes `mbarrier_arrive_and_expect_tx(barrier, B_bytes)`.

This PTX instruction does two things atomically:
1. **Arrival**: decrements `pendingCount` by 1.
2. **Set tx expectation**: adds `B_bytes` to `expectedTxCount`.

| phase | pendingCount | expectedTxCount | completedTxCount |
|-------|-------------|-----------------|-----------------|
| 0 | **128** | **B_bytes** | 0 |

### Step 2: TMA B copy issued

```python
# grouped_gemm.py:1887-1894
cute.copy(tma_atom_b, tBgB[...], tBsB[...],
          tma_bar_ptr=barrier, ...)
```

The TMA warp issues `cp.async.bulk.tensor` with the barrier pointer.
The TMA hardware begins an asynchronous bulk copy of the B tile from GMEM to SMEM.

**No immediate barrier state change.**
When the copy completes, the hardware will add `B_bytes` to `completedTxCount`.

| phase | pendingCount | expectedTxCount | completedTxCount |
|-------|-------------|-----------------|-----------------|
| 0 | 128 | B_bytes | 0 |

### Step 3: 128 threads issue cp.async loads for A

```python
# grouped_gemm.py:1905-1916
self.load_A_gather(mA, tmAIdx, ..., tAsA[..., stage], tApA, ...)
```

Each of the 128 producer threads issues per-thread `cp.async` instructions (GMEM→SMEM) via `cute.copy` with `CopyG2SOp`.
These are standard `cp.async.shared.global` instructions.
**They do not interact with the mbarrier at this point.**
They go into each thread's pending cp.async group.

| phase | pendingCount | expectedTxCount | completedTxCount |
|-------|-------------|-----------------|-----------------|
| 0 | 128 | B_bytes | 0 |

### Step 4: 128 × `cp.async.mbarrier.arrive.noinc(barrier)`

```python
# grouped_gemm.py:1931
mainloop_pipeline.producer_cpasync_commit(mainloop_producer_state)
```

Calls `PipelineTmaCpAsync.producer_cpasync_commit` ([pipeline.py:155-159](../thirdparty/quack/quack/pipeline.py#L155)):
```python
cute.arch.cp_async_mbarrier_arrive_noinc(barrier)
```

**Every producer thread** (all 128) executes the PTX instruction:
```
cp.async.mbarrier.arrive.shared.b64 [mbar], noinc=1;
```

This instruction does two things:
1. **Links** the calling thread's pending cp.async operations to the mbarrier.
   When those copies complete, they will produce **one arrival** on the barrier (per thread).
2. **`noinc`**: does **NOT** increment `pendingCount`.
   The arrival slots were pre-accounted by `mbarrier.init(129)`.

| phase | pendingCount | expectedTxCount | completedTxCount |
|-------|-------------|-----------------|-----------------|
| 0 | 128 | B_bytes | 0 |

No state change — the ops are linked but haven't completed yet.

### Step 5: Async completions (hardware-driven)

Two independent async paths are in flight:

**5a. TMA B completes.**
The TMA hardware finishes copying B into SMEM and reports to the barrier:
- `completedTxCount += B_bytes`

**5b. Each thread's cp.async completes.**
As each thread's linked cp.async operations finish, they automatically perform
**one arrival** per thread on the barrier:
- `pendingCount -= 1` (×128 total)

These happen asynchronously and in any order. Final state after all completions:

| phase | pendingCount | expectedTxCount | completedTxCount |
|-------|-------------|-----------------|-----------------|
| **1** (flipped) | **0** | B_bytes | **B_bytes** |

Both completion conditions met:
- `pendingCount == 0` ✓ (128 − 128 = 0)
- `completedTxCount >= expectedTxCount` ✓ (B_bytes ≥ B_bytes)

Phase flips. Consumer is unblocked.

## Summary diagram

```
mbarrier.init(129)
  pendingCount=129, expectedTx=0, completedTx=0

arrive_and_expect_tx(B_bytes)          [1 elected thread]
  pendingCount=128, expectedTx=B_bytes, completedTx=0

TMA B issued                           [1 TMA warp]
  (no immediate change)

128× cp.async A issued                 [128 threads]
  (no immediate change)

128× cp.async.mbarrier.arrive.noinc    [128 threads]
  (links cp.async to barrier, no state change due to noinc)

--- async completions (hardware-driven) ---

TMA B lands:     completedTx = B_bytes  (tx condition met ✓)
128× cp.async A: pendingCount → 0       (arrival condition met ✓)

→ phase flips, consumer unblocked
```

## Why `noinc` is necessary

Without `.noinc`, each of the 128 calls to `cp.async.mbarrier.arrive` would
**increment** `pendingCount` (add 1 expected arrival per call):

- After init: pendingCount = 129
- After `arrive_and_expect_tx`: pendingCount = 128
- After 128 × `cp.async.mbarrier.arrive` (no noinc): pendingCount = 128 + 128 = **256**
- After 128 cp.async completions arrive: pendingCount = 256 − 128 = **128** → barrier never completes

The kernel uses **static pre-accounting**:
- Initialize the barrier knowing exactly how many arrivals to expect: `1 + 128 = 129`.
- Use `.noinc` so that linking cp.async doesn't inflate `pendingCount`.
- The 128 cp.async completions then satisfy the 128 pre-initialized arrival slots.

### Why not use the non-noinc variant with `arrive_count = 1`?

If you initialized with `arrive_count = 1` and let each `cp.async.mbarrier.arrive` (no noinc)
dynamically add to `pendingCount`:

- After init: pendingCount = 1
- After `arrive_and_expect_tx`: pendingCount = 0, expectedTxCount = B_bytes
- **Race**: if TMA B completes before any `cp.async.mbarrier.arrive` call, then
  `pendingCount == 0` and `completedTxCount >= expectedTxCount` → barrier flips prematurely,
  before any A data has been linked or loaded.

The static pre-accounting pattern avoids this race entirely.

## Why A bytes are not in `expectedTxCount`

The A-side cp.async completion is tracked via the **arrival mechanism** (`pendingCount`),
not the tx-count mechanism (`expectedTxCount` / `completedTxCount`).

Each thread's cp.async completion produces one arrival.
The barrier doesn't need to know how many bytes each thread copies.

The tx-count mechanism is used only for the B-side TMA, which needs byte-level tracking because
TMA is a hardware unit that reports completion via byte counts, not thread arrivals.

```
                    Tracking mechanism
                    ──────────────────
B (TMA):            tx-count (expectedTxCount / completedTxCount)
                    + 1 arrival from arrive_and_expect_tx

A (cp.async):       arrivals only (128 pre-accounted in pendingCount)
                    linked via cp.async.mbarrier.arrive.noinc
```

## Comparison: non-gather path (pure TMA)

When `is_A_gather == False`, both A and B are TMA loads:

```python
# line 1428: A bytes + B bytes
tma_copy_bytes = size_in_bytes(a_dtype, a_smem_layout) + size_in_bytes(b_dtype, b_smem_layout)

# line 1445: single-thread producer group
producer_group = CooperativeGroup(Thread)  # arrive_count = 1
```

State trace:

```
mbarrier.init(1)
  pendingCount=1, expectedTx=0, completedTx=0

arrive_and_expect_tx(A_bytes + B_bytes)    [1 elected thread]
  pendingCount=0, expectedTx=A_bytes+B_bytes, completedTx=0

TMA A + TMA B issued
  (no immediate change)

TMA A lands: completedTx += A_bytes
TMA B lands: completedTx += B_bytes
  completedTx = A_bytes + B_bytes ≥ expectedTx ✓
  pendingCount = 0 ✓

→ phase flips, consumer unblocked
```

No `cp.async.mbarrier.arrive` is needed.
`producer_commit` is a NOP ([sm90.py:541](../python/CuTeDSL/cutlass/pipeline/sm90.py#L541)).

## Code references

| What | Where |
|------|-------|
| Pipeline setup (gather vs non-gather) | [grouped_gemm.py:1420-1460](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L1420) |
| Producer loop (TMA B + gather A + commit) | [grouped_gemm.py:1878-1937](../thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py#L1878) |
| `PipelineTmaCpAsync.producer_acquire` | [pipeline.py:131-152](../thirdparty/quack/quack/pipeline.py#L131) |
| `PipelineTmaCpAsync.producer_cpasync_commit` | [pipeline.py:155-159](../thirdparty/quack/quack/pipeline.py#L155) |
| `MbarrierArray.__init__` / `mbarrier_init` | [helpers.py:166-234](../python/CuTeDSL/cutlass/pipeline/helpers.py#L166) |
| `MbarrierArray.arrive_and_expect_tx` | [helpers.py:326-333](../python/CuTeDSL/cutlass/pipeline/helpers.py#L326) |
| `cp_async_mbarrier_arrive_noinc` | [mbar.py:255-268](../python/CuTeDSL/cutlass/cute/arch/mbar.py#L255) |
| PTX ISA: mbarrier | [docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier) |
| PTX ISA: cp.async.mbarrier.arrive | [docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-cp-async-mbarrier-arrive) |
