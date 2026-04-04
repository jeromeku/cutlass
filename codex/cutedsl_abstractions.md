# CuTe DSL Pipeline Abstractions vs CUTLASS/C++ Counterparts

## Scope

This note maps the CuTe DSL SM90 pipeline abstractions in:

- [sm90.py](../python/CuTeDSL/cutlass/pipeline/sm90.py#L37)

to the closest CUTLASS/C++ equivalents in:

- [sm90_pipeline.hpp](../include/cutlass/pipeline/sm90_pipeline.hpp#L171)
- [barrier.h](../include/cutlass/arch/barrier.h#L342)
- [cluster_sm90.hpp](../include/cute/arch/cluster_sm90.hpp#L48)
- [copy_sm90_tma.hpp](../include/cute/arch/copy_sm90_tma.hpp#L1225)

The main conclusion is:

1. The Python DSL does **not** inline PTX directly from `sm90.py`.
2. It lowers through `cutlass.pipeline.helpers` and `cute.arch.*` wrappers into NVVM/MLIR ops.
3. The CUTLASS/C++ side exposes the same hardware behavior through `cutlass::Pipeline*` classes and `cutlass::arch::*` / `cute::*` helpers, many of which do use inline PTX.
4. `PipelineProducer`, `PipelineConsumer`, and `ImmutableResourceHandle` are **DSL-only convenience layers**. There is no direct 1:1 C++ class with those names or semantics.

## Code Map

- [PipelineAsync / PipelineCpAsync / PipelineTmaAsync / PipelineTmaStore / PipelineOrder](../python/CuTeDSL/cutlass/pipeline/sm90.py#L37)
- [PipelineProducer / PipelineConsumer / ImmutableResourceHandle](../python/CuTeDSL/cutlass/pipeline/sm90.py#L888)
- [Pipeline helpers: Agent, CooperativeGroup, PipelineOp, SyncObject, MbarrierArray, TmaStoreFence, PipelineState](../python/CuTeDSL/cutlass/pipeline/helpers.py#L36)
- [DSL mbarrier wrappers](../python/CuTeDSL/cutlass/cute/arch/mbar.py#L27)
- [DSL NVVM wrappers for cluster and cp.async.bulk](../python/CuTeDSL/cutlass/cute/arch/nvvm_wrappers.py#L596)
- [CUTLASS `PipelineState`, `PipelineTmaAsync`, `PipelineTransactionAsync`, `PipelineAsync`, `OrderedSequenceBarrier`](../include/cutlass/pipeline/sm90_pipeline.hpp#L171)
- [CUTLASS `ClusterBarrier`, `ClusterTransactionBarrier`, cp.async barrier helpers](../include/cutlass/arch/barrier.h#L342)
- [CuTe cluster PTX helpers](../include/cute/arch/cluster_sm90.hpp#L48)
- [CuTe TMA store PTX helpers](../include/cute/arch/copy_sm90_tma.hpp#L1225)
- [Representative C++ `PipelineAsync` usage](../test/unit/pipeline/pipeline_async.cu#L72)
- [Representative C++ `PipelineTmaAsync` usage](../test/unit/pipeline/pipeline_tma_async.cu#L72)
- [Representative C++ `OrderedSequenceBarrier` usage](../test/unit/pipeline/sequence_barrier.cu#L72)

## Key Functions Index

| Function | File | Purpose |
|----------|------|---------|
| `PipelineAsync.create` | [sm90.py](../python/CuTeDSL/cutlass/pipeline/sm90.py#L152) | Builds the generic async producer/consumer pipeline from DSL sync objects. |
| `PipelineTmaAsync.create` | [sm90.py](../python/CuTeDSL/cutlass/pipeline/sm90.py#L433) | Builds the TMA-flavored pipeline and computes which consumers signal empties. |
| `MbarrierArray.arrive` | [helpers.py](../python/CuTeDSL/cutlass/pipeline/helpers.py#L236) | Dispatches to the right barrier-arrive primitive based on `PipelineOp`. |
| `mbarrier_wait` | [mbar.py](../python/CuTeDSL/cutlass/cute/arch/mbar.py#L135) | Lowers DSL waits to NVVM mbarrier wait ops. |
| `mbarrier_arrive` | [mbar.py](../python/CuTeDSL/cutlass/cute/arch/mbar.py#L212) | Lowers DSL arrives to NVVM mbarrier transaction ops. |
| `cp_async_mbarrier_arrive_noinc` | [mbar.py](../python/CuTeDSL/cutlass/cute/arch/mbar.py#L255) | Lowers cp.async completion signaling to NVVM. |
| `ClusterBarrier::wait/arrive` | [barrier.h](../include/cutlass/arch/barrier.h#L408) | CUTLASS inline PTX wrapper for basic barrier waits and arrives. |
| `ClusterTransactionBarrier::arrive_and_expect_tx` | [barrier.h](../include/cutlass/arch/barrier.h#L586) | CUTLASS inline PTX wrapper for transaction-counted TMA barriers. |
| `PipelineAsync::producer_commit` | [sm90_pipeline.hpp](../include/cutlass/pipeline/sm90_pipeline.hpp#L1115) | Generic C++ pipeline commit path for non-TMA producers. |
| `PipelineTmaAsync::producer_acquire` | [sm90_pipeline.hpp](../include/cutlass/pipeline/sm90_pipeline.hpp#L425) | TMA C++ pipeline acquire path that arms the transaction barrier. |

## Big Picture

There are three layers here:

1. **Python ergonomic layer**
   - `PipelineProducer`
   - `PipelineConsumer`
   - `ImmutableResourceHandle`
   - `PipelineAsync.create(...)` helpers

2. **Python hardware-abstraction layer**
   - `MbarrierArray`
   - `TmaStoreFence`
   - `cute.arch.mbarrier_*`
   - `cute.arch.cluster_*`
   - `cute.arch.cp_async_*`

3. **Lowering / target layer**
   - DSL side: NVVM MLIR ops such as `nvvm.mbarrier_txn`, `nvvm.mbarrier_try_wait_parity_shared`, `nvvm.cp_async_bulk_commit_group`, `nvvm.cluster_arrive`
   - C++ side: CUTLASS/CuTe wrappers that emit inline PTX such as:
     - `mbarrier.try_wait.parity.shared::cta.b64`
     - `mbarrier.arrive.shared::cta.b64`
     - `mbarrier.arrive.expect_tx.shared::cta.b64`
     - `cp.async.mbarrier.arrive.noinc.shared::cta.b64`
     - `cp.async.bulk.commit_group`
     - `cp.async.bulk.wait_group.read`
     - `barrier.cluster.arrive.relaxed.aligned`
     - `barrier.cluster.wait.aligned`

The Python and C++ paths are therefore conceptually equivalent, but not mechanically identical:

- Python goes through MLIR/NVVM ops.
- C++ goes through inline PTX helper wrappers.

## What Maps Cleanly vs What Does Not

### Clean 1:1-ish mappings

| DSL abstraction | Closest C++ counterpart | Notes |
|---|---|---|
| `PipelineState` | `cutlass::PipelineState<Stages>` | Same conceptual state: `(index, phase, count)`. |
| `PipelineAsync` | `cutlass::PipelineAsync<Stages>` | Generic async producer/consumer pipeline. |
| `PipelineTmaAsync` | `cutlass::PipelineTmaAsync<Stages>` | TMA producer + async consumer pipeline. |
| `PipelineTmaStore` | `cutlass::PipelineTmaStore<Stages>` | Producer-only TMA store pipeline. |
| `PipelineOrder` | `cutlass::OrderedSequenceBarrier<Depth, Length>` | Same ordered-stage/group sequencing abstraction. |
| `MbarrierArray` | `ClusterBarrier[]` or `ClusterTransactionBarrier[]` in `SharedStorage` | DSL runtime dispatches among barrier flavors; C++ types are explicit. |
| `TmaStoreFence` | `cute::tma_store_arrive()` / `cute::tma_store_wait<>()` via `PipelineTmaStore` | Same TMA-store fence semantics. |

### Not 1:1

| DSL abstraction | Closest C++ equivalent | Why it is not 1:1 |
|---|---|---|
| `PipelineProducer` | `(pipeline object, producer PipelineState)` | C++ keeps state explicit; there is no producer wrapper object. |
| `PipelineConsumer` | `(pipeline object, consumer PipelineState)` | Same issue on the consumer side. |
| `ImmutableResourceHandle` | A snapshot of `PipelineState`, plus optionally `producer_get_barrier(state)` | C++ relies on explicit state passing and tokens, not a persistent handle object. |
| `CooperativeGroup` | `Params` counts plus `ThreadCategory` / launch structure | The DSL models groups explicitly; C++ mostly encodes them as counts and role enums. |
| `PipelineOp` | Concrete pipeline class choice and/or selected arch helper | C++ usually picks the primitive statically, not via an enum-dispatch object. |
| `SyncObject` | No direct equivalent | C++ uses concrete barrier/fence classes, not an abstract polymorphic base. |

## Shared Base Abstractions

### `PipelineState`

DSL definition: [helpers.py](../python/CuTeDSL/cutlass/pipeline/helpers.py#L559)

```python
class PipelineState:
    def __init__(self, stages: int, count, index, phase):
        self._stages = stages
        self._count = count
        self._index = index
        self._phase = phase
```

C++ definition: [sm90_pipeline.hpp](../include/cutlass/pipeline/sm90_pipeline.hpp#L171)

```cpp
template<uint32_t Stages_>
struct PipelineState {
  int index_ = 0;
  uint32_t phase_ = 0;
  uint32_t count_ = 0;
};
```

This is a real 1:1 conceptual match.

Both track:

- `index`: current stage in the circular buffer
- `phase`: parity bit used by mbarrier waits
- `count`: total number of advances

The producer-start asymmetry also matches:

- DSL `make_pipeline_state(PipelineUserType.Producer, ...)` starts with `phase = 1`: [helpers.py](../python/CuTeDSL/cutlass/pipeline/helpers.py#L656)
- C++ `make_producer_start_state<Pipeline>()` starts with `phase = 1`: [sm90_pipeline.hpp](../include/cutlass/pipeline/sm90_pipeline.hpp#L252)

### `MbarrierArray`

DSL definition: [helpers.py](../python/CuTeDSL/cutlass/pipeline/helpers.py#L161)

This is the central Python abstraction that chooses which low-level barrier primitive to call based on `PipelineOp`.

Dispatch points:

- `PipelineOp.AsyncThread` -> `arrive_mbarrier`: [helpers.py](../python/CuTeDSL/cutlass/pipeline/helpers.py#L260)
- `PipelineOp.TmaLoad` -> `arrive_and_expect_tx`: [helpers.py](../python/CuTeDSL/cutlass/pipeline/helpers.py#L267)
- `PipelineOp.AsyncLoad` -> `cp_async_mbarrier_arrive_noinc`: [helpers.py](../python/CuTeDSL/cutlass/pipeline/helpers.py#L275)

The C++ side does not wrap these variants behind one runtime-dispatched object. Instead it uses explicit barrier classes:

- `cutlass::arch::ClusterBarrier`: [barrier.h](../include/cutlass/arch/barrier.h#L342)
- `cutlass::arch::ClusterTransactionBarrier`: [barrier.h](../include/cutlass/arch/barrier.h#L546)
- `cutlass::arch::cpasync_barrier_arrive[_noinc]`: [barrier.h](../include/cutlass/arch/barrier.h#L757)

So the closest mapping is:

```text
DSL MbarrierArray
  ~= "a Python-side dispatch shell over the same barrier primitives that C++ chooses statically"
```

### `TmaStoreFence`

DSL definition: [helpers.py](../python/CuTeDSL/cutlass/pipeline/helpers.py#L498)

```python
def arrive(self):
    cute.arch.cp_async_bulk_commit_group()

def wait(self):
    cute.arch.cp_async_bulk_wait_group(self.num_stages - 1, read=True)
```

The C++ equivalent is not a generic fence base class. It is the dedicated TMA-store path:

- `cutlass::PipelineTmaStore<Stages>`: [sm90_pipeline.hpp](../include/cutlass/pipeline/sm90_pipeline.hpp#L655)
- `cute::tma_store_arrive()`: [copy_sm90_tma.hpp](../include/cute/arch/copy_sm90_tma.hpp#L1223)
- `cute::tma_store_wait<Count>()`: [copy_sm90_tma.hpp](../include/cute/arch/copy_sm90_tma.hpp#L1245)

The underlying PTX is explicit in C++:

```cpp
asm volatile("cp.async.bulk.commit_group;");
asm volatile("cp.async.bulk.wait_group.read %0;");
```

## Class-by-Class Mapping

### `PipelineAsync`

DSL: [sm90.py](../python/CuTeDSL/cutlass/pipeline/sm90.py#L37)

C++: [sm90_pipeline.hpp](../include/cutlass/pipeline/sm90_pipeline.hpp#L1015)

This is the basic non-transaction-counted async pipeline.

#### Producer path

DSL:

- `producer_acquire(state)` waits on `sync_object_empty.wait(index, phase)`: [sm90.py](../python/CuTeDSL/cutlass/pipeline/sm90.py#L214)
- `producer_commit(state)` calls `sync_object_full.arrive(index, producer_mask)`: [sm90.py](../python/CuTeDSL/cutlass/pipeline/sm90.py#L236)

C++:

- `PipelineAsync::producer_acquire(state, token)`: [sm90_pipeline.hpp](../include/cutlass/pipeline/sm90_pipeline.hpp#L1109)
- `PipelineAsync::producer_commit(state)`: [sm90_pipeline.hpp](../include/cutlass/pipeline/sm90_pipeline.hpp#L1115)

Low-level instruction path:

- empty wait -> `mbarrier.try_wait.parity.shared::cta.b64`
- full arrive -> `mbarrier.arrive.shared::cta.b64` or cluster variant if remote

#### Consumer path

DSL:

- `consumer_wait(state)` waits on `sync_object_full.wait(index, phase)`: [sm90.py](../python/CuTeDSL/cutlass/pipeline/sm90.py#L241)
- `consumer_release(state)` arrives on empty barrier: [sm90.py](../python/CuTeDSL/cutlass/pipeline/sm90.py#L262)

C++:

- `PipelineAsync::consumer_wait(state, token)`: [sm90_pipeline.hpp](../include/cutlass/pipeline/sm90_pipeline.hpp#L1154)
- `PipelineAsync::consumer_release(state)`: [sm90_pipeline.hpp](../include/cutlass/pipeline/sm90_pipeline.hpp#L1159)

### `PipelineCpAsync`

DSL: [sm90.py](../python/CuTeDSL/cutlass/pipeline/sm90.py#L309)

There is **no dedicated `cutlass::PipelineCpAsync` class** in `sm90_pipeline.hpp`.

The closest C++ correspondence is:

- `cutlass::PipelineAsync<Stages>`
- plus a cp.async-specific producer commit helper such as:
  - `cutlass::arch::cpasync_barrier_arrive`: [barrier.h](../include/cutlass/arch/barrier.h#L757)
  - `cutlass::arch::cpasync_barrier_arrive_noinc`: [barrier.h](../include/cutlass/arch/barrier.h#L775)

You can see this pattern in C++ call sites such as:

- [sm90_mma_multistage_gmma_ss_warpspecialized.hpp](../include/cutlass/gemm/collective/sm90_mma_multistage_gmma_ss_warpspecialized.hpp#L291)
- [sm100_mma_array_warpspecialized_blockwise_scaling.hpp](../include/cutlass/gemm/collective/sm100_mma_array_warpspecialized_blockwise_scaling.hpp#L935)

Those call:

```cpp
pipeline.producer_commit(state, cutlass::arch::cpasync_barrier_arrive);
```

or:

```cpp
pipeline.producer_commit(state, cutlass::arch::cpasync_barrier_arrive_noinc);
```

That is the C++ analogue of the DSL choosing `PipelineOp.AsyncLoad` and routing `MbarrierArray.arrive(...)` to:

- `cute.arch.cp_async_mbarrier_arrive_noinc(...)`: [helpers.py](../python/CuTeDSL/cutlass/pipeline/helpers.py#L296)
- which lowers to `nvvm.cp_async_mbarrier_arrive_shared(..., noinc=True)`: [mbar.py](../python/CuTeDSL/cutlass/cute/arch/mbar.py#L255)
- which corresponds to inline PTX:
  - `cp.async.mbarrier.arrive.shared::cta.b64`
  - `cp.async.mbarrier.arrive.noinc.shared::cta.b64`

So `PipelineCpAsync` is best understood as:

```text
DSL convenience name
  ~= C++ PipelineAsync + cp.async-specific commit callback
```

### `PipelineTmaAsync`

DSL: [sm90.py](../python/CuTeDSL/cutlass/pipeline/sm90.py#L368)

C++: [sm90_pipeline.hpp](../include/cutlass/pipeline/sm90_pipeline.hpp#L271)

This is the closest true 1:1 mapping among the specialized classes.

The key semantic difference from `PipelineAsync` is that the producer arms a **transaction-counted full barrier**.

DSL producer acquire:

```python
self.sync_object_empty.wait(state.index, state.phase)
self.sync_object_full.arrive(state.index, self.producer_mask)
```

See [sm90.py](../python/CuTeDSL/cutlass/pipeline/sm90.py#L518).

Because `sync_object_full` was constructed with producer type `PipelineOp.TmaLoad`, that arrive dispatch becomes:

```text
MbarrierArray.arrive
-> arrive_and_expect_tx
-> cute.arch.mbarrier_arrive_and_expect_tx
-> nvvm.mbarrier_txn(... kind=ARRIVE_EXPECT_TX ...)
```

Relevant code:

- [helpers.py](../python/CuTeDSL/cutlass/pipeline/helpers.py#L267)
- [mbar.py](../python/CuTeDSL/cutlass/cute/arch/mbar.py#L55)

The C++ equivalent is explicit:

- `PipelineTmaAsync::producer_acquire(...)` calls `full_barrier_ptr_[stage].arrive_and_expect_tx(params_.transaction_bytes)`: [sm90_pipeline.hpp](../include/cutlass/pipeline/sm90_pipeline.hpp#L512)
- `ClusterTransactionBarrier::arrive_and_expect_tx(...)` emits inline PTX:
  - `mbarrier.arrive.expect_tx.shared::cta.b64`
  - or cluster variant `mbarrier.arrive.expect_tx.shared::cluster.b64`

See [barrier.h](../include/cutlass/arch/barrier.h#L586).

`PipelineTmaAsync.producer_commit(...)` is a no-op in the DSL: [sm90.py](../python/CuTeDSL/cutlass/pipeline/sm90.py#L541)

That matches the C++ intent:

- the TMA copy instruction itself completes the transaction count
- `PipelineTmaAsync::producer_commit(state, bytes)` is basically a test scaffolding path for non-real-TMA execution: [sm90_pipeline.hpp](../include/cutlass/pipeline/sm90_pipeline.hpp#L560)

The consumer empty-release signal scheduling also maps conceptually:

- DSL computes `dst_rank` and `is_signalling_thread` in `init_empty_barrier_arrive_signal`: [sm90.py](../python/CuTeDSL/cutlass/pipeline/sm90.py#L375)
- C++ computes `dst_blockid_` / `is_signaling_thread_` in the `PipelineTmaAsync` constructor using helper layouts: [sm90_pipeline.hpp](../include/cutlass/pipeline/sm90_pipeline.hpp#L343)

### `PipelineTmaStore`

DSL: [sm90.py](../python/CuTeDSL/cutlass/pipeline/sm90.py#L729)

C++: [sm90_pipeline.hpp](../include/cutlass/pipeline/sm90_pipeline.hpp#L655)

This is producer-only and does not use mbarriers on the producer side.

DSL path:

- `producer_acquire()` -> `TmaStoreFence.wait()` -> `cp_async_bulk_wait_group(..., read=True)`: [sm90.py](../python/CuTeDSL/cutlass/pipeline/sm90.py#L757), [helpers.py](../python/CuTeDSL/cutlass/pipeline/helpers.py#L514)
- `producer_commit()` -> `TmaStoreFence.arrive()` -> `cp_async_bulk_commit_group()`: [sm90.py](../python/CuTeDSL/cutlass/pipeline/sm90.py#L761), [helpers.py](../python/CuTeDSL/cutlass/pipeline/helpers.py#L510)

DSL lowering:

- `cute.arch.cp_async_bulk_commit_group` -> `nvvm.cp_async_bulk_commit_group`: [nvvm_wrappers.py](../python/CuTeDSL/cutlass/cute/arch/nvvm_wrappers.py#L596)
- `cute.arch.cp_async_bulk_wait_group` -> `nvvm.cp_async_bulk_wait_group`: [nvvm_wrappers.py](../python/CuTeDSL/cutlass/cute/arch/nvvm_wrappers.py#L606)

C++ path:

- `PipelineTmaStore::producer_acquire` -> `tma_store_wait<UnacquiredStages>()`: [sm90_pipeline.hpp](../include/cutlass/pipeline/sm90_pipeline.hpp#L675)
- `PipelineTmaStore::producer_commit` -> `tma_store_arrive()`: [sm90_pipeline.hpp](../include/cutlass/pipeline/sm90_pipeline.hpp#L681)

Actual PTX:

- `cp.async.bulk.commit_group`
- `cp.async.bulk.wait_group.read %0`

See [copy_sm90_tma.hpp](../include/cute/arch/copy_sm90_tma.hpp#L1225).

### `PipelineOrder`

DSL: [sm90.py](../python/CuTeDSL/cutlass/pipeline/sm90.py#L779)

C++: [sm90_pipeline.hpp](../include/cutlass/pipeline/sm90_pipeline.hpp#L1262) as `OrderedSequenceBarrier`

This is another clean conceptual mapping.

DSL:

- `wait()` -> `cute.arch.mbarrier_wait(...)`: [sm90.py](../python/CuTeDSL/cutlass/pipeline/sm90.py#L872)
- `arrive()` -> `cute.arch.mbarrier_arrive(...)` then `state.advance()`: [sm90.py](../python/CuTeDSL/cutlass/pipeline/sm90.py#L863)

C++:

- `OrderedSequenceBarrier::wait()`: [sm90_pipeline.hpp](../include/cutlass/pipeline/sm90_pipeline.hpp#L1332)
- `OrderedSequenceBarrier::arrive()`: [sm90_pipeline.hpp](../include/cutlass/pipeline/sm90_pipeline.hpp#L1337)

Representative usage:

- [sequence_barrier.cu](../test/unit/pipeline/sequence_barrier.cu#L92)

## Why `PipelineProducer` and `PipelineConsumer` Are DSL-Only Wrappers

DSL definitions:

- [PipelineProducer](../python/CuTeDSL/cutlass/pipeline/sm90.py#L935)
- [PipelineConsumer](../python/CuTeDSL/cutlass/pipeline/sm90.py#L1135)

These wrap:

- a pipeline object
- a mutable `PipelineState`
- a group

and expose ergonomic methods like:

- `acquire_and_advance()`
- `wait_and_advance()`
- `try_acquire()`
- `try_wait()`

There is no equivalent class in C++.

The corresponding C++ pattern is visible in:

- [pipeline_async.cu](../test/unit/pipeline/pipeline_async.cu#L98)
- [pipeline_tma_async.cu](../test/unit/pipeline/pipeline_tma_async.cu#L113)

Example:

```cpp
PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>();
pipeline.producer_acquire(smem_pipe_write);
pipeline.producer_commit(smem_pipe_write);
++smem_pipe_write;
```

That is the C++ equivalent of the DSL wrapper workflow:

```python
handle = producer.acquire_and_advance()
handle.commit()
```

The C++ version keeps the state explicit and passes it into the pipeline methods each time.

## Why `ImmutableResourceHandle` Exists in Python

DSL definitions:

- [base `ImmutableResourceHandle`](../python/CuTeDSL/cutlass/pipeline/sm90.py#L888)
- [producer handle subclass](../python/CuTeDSL/cutlass/pipeline/sm90.py#L975)
- [consumer handle subclass](../python/CuTeDSL/cutlass/pipeline/sm90.py#L1178)

This object snapshots the current `PipelineState` so that after `acquire_and_advance()` or `wait_and_advance()`, the caller can still refer to the stage that was just acquired/waited on even though the participant object has already advanced.

That is why:

- producer handle exposes `.barrier` and `.commit()`
- consumer handle exposes `.release()`

There is no C++ equivalent object because C++ leaves that responsibility with the caller:

- keep the old `PipelineState` if you still need it
- or fetch `producer_get_barrier(state)` directly
- or use a `ProducerToken` / `ConsumerToken` from try/finalize pairs

Relevant C++ pieces:

- `ProducerToken` / `ConsumerToken`: [sm90_pipeline.hpp](../include/cutlass/pipeline/sm90_pipeline.hpp#L160)
- `producer_get_barrier`: [sm90_pipeline.hpp](../include/cutlass/pipeline/sm90_pipeline.hpp#L1136), [sm90_pipeline.hpp](../include/cutlass/pipeline/sm90_pipeline.hpp#L457)

## Representative Dispatch Traces

### Trace A: `PipelineProducer.acquire_and_advance()` in the generic async pipeline

Python call chain:

```text
PipelineProducer.acquire_and_advance()
-> PipelineProducer.acquire()
-> PipelineAsync.producer_acquire(state)
-> sync_object_empty.wait(index, phase)
-> MbarrierArray.wait(index, phase)
-> cute.arch.mbarrier_wait(ptr, phase)
-> nvvm.mbarrier_try_wait_parity_shared(...)
```

Relevant code:

- [sm90.py](../python/CuTeDSL/cutlass/pipeline/sm90.py#L1051)
- [helpers.py](../python/CuTeDSL/cutlass/pipeline/helpers.py#L348)
- [mbar.py](../python/CuTeDSL/cutlass/cute/arch/mbar.py#L135)

C++ analogue:

```text
pipeline.producer_acquire(state)
-> ClusterBarrier::wait(...)
-> inline PTX mbarrier.try_wait.parity.shared::cta.b64
```

Relevant code:

- [sm90_pipeline.hpp](../include/cutlass/pipeline/sm90_pipeline.hpp#L1184)
- [barrier.h](../include/cutlass/arch/barrier.h#L408)

### Trace B: `handle.commit()` in the generic async pipeline

Python call chain:

```text
PipelineProducer.ImmutableResourceHandle.commit()
-> PipelineAsync.producer_commit(immutable_state)
-> sync_object_full.arrive(index, producer_mask)
-> MbarrierArray.arrive(...)
-> cute.arch.mbarrier_arrive(...)
-> nvvm.mbarrier_txn(... kind=ARRIVE ...)
```

Relevant code:

- [sm90.py](../python/CuTeDSL/cutlass/pipeline/sm90.py#L988)
- [helpers.py](../python/CuTeDSL/cutlass/pipeline/helpers.py#L236)
- [mbar.py](../python/CuTeDSL/cutlass/cute/arch/mbar.py#L212)

C++ analogue:

```text
pipeline.producer_commit(state)
-> ClusterBarrier::arrive(...)
-> inline PTX mbarrier.arrive.shared::cta.b64
```

Relevant code:

- [sm90_pipeline.hpp](../include/cutlass/pipeline/sm90_pipeline.hpp#L1192)
- [barrier.h](../include/cutlass/arch/barrier.h#L507)

### Trace C: `PipelineTmaAsync.producer_acquire()` in the TMA pipeline

Python call chain:

```text
PipelineTmaAsync.producer_acquire(state)
-> sync_object_empty.wait(index, phase)
-> sync_object_full.arrive(index, producer_mask)
-> MbarrierArray.arrive(...)
-> MbarrierArray.arrive_and_expect_tx(...)
-> cute.arch.mbarrier_arrive_and_expect_tx(...)
-> nvvm.mbarrier_txn(... kind=ARRIVE_EXPECT_TX ...)
```

Relevant code:

- [sm90.py](../python/CuTeDSL/cutlass/pipeline/sm90.py#L518)
- [helpers.py](../python/CuTeDSL/cutlass/pipeline/helpers.py#L327)
- [mbar.py](../python/CuTeDSL/cutlass/cute/arch/mbar.py#L55)

C++ analogue:

```text
PipelineTmaAsync::producer_acquire(state)
-> ClusterTransactionBarrier::arrive_and_expect_tx(bytes)
-> inline PTX mbarrier.arrive.expect_tx.shared::cta.b64
```

Relevant code:

- [sm90_pipeline.hpp](../include/cutlass/pipeline/sm90_pipeline.hpp#L512)
- [barrier.h](../include/cutlass/arch/barrier.h#L586)

### Trace D: `PipelineTmaStore.producer_commit()`

Python call chain:

```text
PipelineTmaStore.producer_commit()
-> TmaStoreFence.arrive()
-> cute.arch.cp_async_bulk_commit_group()
-> nvvm.cp_async_bulk_commit_group()
```

Relevant code:

- [sm90.py](../python/CuTeDSL/cutlass/pipeline/sm90.py#L761)
- [helpers.py](../python/CuTeDSL/cutlass/pipeline/helpers.py#L509)
- [nvvm_wrappers.py](../python/CuTeDSL/cutlass/cute/arch/nvvm_wrappers.py#L596)

C++ analogue:

```text
PipelineTmaStore::producer_commit(state)
-> cute::tma_store_arrive()
-> inline PTX cp.async.bulk.commit_group
```

Relevant code:

- [sm90_pipeline.hpp](../include/cutlass/pipeline/sm90_pipeline.hpp#L681)
- [copy_sm90_tma.hpp](../include/cute/arch/copy_sm90_tma.hpp#L1223)

## Low-Level Instruction Summary

| DSL wrapper | NVVM op / lowering | C++ helper | Inline PTX / hardware op |
|---|---|---|---|
| `cute.arch.mbarrier_wait` | `nvvm.mbarrier_try_wait_parity_shared` | `ClusterBarrier::wait` | `mbarrier.try_wait.parity.shared::cta.b64` spin loop |
| `cute.arch.mbarrier_try_wait` | `nvvm.mbarrier_wait_parity(... TRY)` | `ClusterBarrier::try_wait` | `mbarrier.try_wait.parity.shared::cta.b64` |
| `cute.arch.mbarrier_arrive` | `nvvm.mbarrier_txn(... ARRIVE ...)` | `ClusterBarrier::arrive` | `mbarrier.arrive.shared::cta.b64` or cluster variant |
| `cute.arch.mbarrier_arrive_and_expect_tx` | `nvvm.mbarrier_txn(... ARRIVE_EXPECT_TX ...)` | `ClusterTransactionBarrier::arrive_and_expect_tx` | `mbarrier.arrive.expect_tx.shared::cta.b64` or cluster variant |
| `cute.arch.cp_async_mbarrier_arrive_noinc` | `nvvm.cp_async_mbarrier_arrive_shared(... noinc=True)` | `cpasync_barrier_arrive_noinc` | `cp.async.mbarrier.arrive.noinc.shared::cta.b64` |
| `cute.arch.cp_async_bulk_commit_group` | `nvvm.cp_async_bulk_commit_group` | `cute::tma_store_arrive` | `cp.async.bulk.commit_group` |
| `cute.arch.cp_async_bulk_wait_group(..., read=True)` | `nvvm.cp_async_bulk_wait_group(..., read=True)` | `cute::tma_store_wait<Count>` | `cp.async.bulk.wait_group.read` |
| `cute.arch.cluster_arrive` | `nvvm.cluster_arrive` | `cute::cluster_arrive` | `barrier.cluster.arrive.aligned` |
| `cute.arch.cluster_arrive_relaxed` | `nvvm.cluster_arrive_relaxed` | `cute::cluster_arrive_relaxed` | `barrier.cluster.arrive.relaxed.aligned` |
| `cute.arch.cluster_wait` | `nvvm.cluster_wait` | `cute::cluster_wait` | `barrier.cluster.wait.aligned` |
| `cute.arch.mbarrier_init_fence` | `nvvm.fence_mbarrier_init` | `cutlass::arch::fence_barrier_init` | `fence.mbarrier_init.release.cluster` |

## Bottom Line

The most useful mental model is:

```text
Python DSL pipeline wrappers
  -> Python sync-object dispatch
  -> cute.arch barrier / cp.async / cluster wrappers
  -> NVVM MLIR ops

C++ CUTLASS pipelines
  -> cutlass::Pipeline* classes
  -> cutlass::arch::* and cute::* helpers
  -> inline PTX
```

And the highest-value correspondence rules are:

1. `PipelineState` is genuinely the same abstraction on both sides.
2. `PipelineAsync`, `PipelineTmaAsync`, `PipelineTmaStore`, and `PipelineOrder` each have close C++ analogues.
3. `PipelineCpAsync` is a DSL naming convenience; in C++ the same behavior is usually expressed as `PipelineAsync` plus a cp.async-specific commit callback.
4. `PipelineProducer`, `PipelineConsumer`, and `ImmutableResourceHandle` are Python ergonomics for a manual C++ pattern built out of:
   - explicit `PipelineState`
   - explicit pipeline method calls
   - optional `ProducerToken` / `ConsumerToken`
   - optional direct barrier pointer access via `producer_get_barrier`
