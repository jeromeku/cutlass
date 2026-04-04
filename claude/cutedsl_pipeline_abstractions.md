# CuTe DSL Pipeline Abstractions → CUTLASS C++ Mapping

## Overview

The CuTe DSL pipeline module (`cutlass.pipeline`) provides Python abstractions for warp-specialized producer-consumer patterns built on top of CUDA's `mbarrier` hardware. These map directly to the C++ pipeline classes in `include/cutlass/pipeline/sm90_pipeline.hpp`.

**Key files:**

| Layer | Python (CuTe DSL) | C++ (CUTLASS) |
|-------|-------------------|---------------|
| Pipeline | `python/CuTeDSL/cutlass/pipeline/sm90.py` | `include/cutlass/pipeline/sm90_pipeline.hpp` |
| Helpers/Base | `python/CuTeDSL/cutlass/pipeline/helpers.py` | (same hpp, plus `arch/barrier.h`) |
| Barriers | `helpers.py` → `MbarrierArray` | `include/cutlass/arch/barrier.h` → `ClusterBarrier` |

---

## 1. PipelineState — Circular Buffer Position Tracker

Tracks position within a multi-stage circular buffer using three values.

### Python (`helpers.py:559-678`)

```python
class PipelineState:
    _stages: int      # Number of pipeline stages
    _index:  Int32    # Current stage index [0, stages)
    _phase:  Int32    # Phase bit (0 or 1), toggles on wrap
    _count:  Int32    # Total operations count

    def advance(self):
        self._index += 1
        self._count += 1
        if self._index == self._stages:
            self._index = 0
            self._phase ^= 1    # Toggle phase on wrap
```

### C++ (`sm90_pipeline.hpp:171-250`)

```cpp
template<uint32_t Stages_>
struct PipelineState {
    int      index_ = 0;
    uint32_t phase_ = 0;
    uint32_t count_ = 0;

    CUTLASS_DEVICE void operator++() {
        ++index_; ++count_;
        if (index_ == Stages) {
            index_ = 0;
            phase_ ^= 1;
        }
    }
};
```

### Mapping

| Python | C++ | Notes |
|--------|-----|-------|
| `PipelineState._index` | `PipelineState::index_` | Stage index `[0, stages)` |
| `PipelineState._phase` | `PipelineState::phase_` | XOR-toggled on wraparound |
| `PipelineState._count` | `PipelineState::count_` | Monotonic operation counter |
| `PipelineState.advance()` | `PipelineState::operator++()` | Increment + conditional phase flip |
| `make_pipeline_state(Producer, N)` | `make_producer_start_state<Pipeline>()` | Producer starts at phase=**1** (buffers empty) |
| `make_pipeline_state(Consumer, N)` | default `PipelineState{}` | Consumer starts at phase=**0** |

**Why phase matters:** The mbarrier `wait` instruction takes a phase argument. A producer waiting on the empty barrier with phase=1 will block until the consumer flips that barrier's phase from 0→1 (by arriving). This is how the "full/empty" protocol works without explicit lock variables.

---

## 2. Barrier Abstractions

### Python: `MbarrierArray` (`helpers.py:161-387`)

Wraps a contiguous array of `mbarrier` objects in shared memory. Dispatches `arrive()` to the correct PTX instruction based on `PipelineOp` type.

```python
class MbarrierArray(SyncObject):
    barrier_storage: cute.Pointer   # Base smem pointer to mbarrier array
    num_stages: int
    op_type: PipelineOp             # AsyncThread, TmaLoad, TCGen05Mma, ...
    cg: CooperativeGroup            # Thread group that participates
    tx_count: int                   # Transaction bytes (TMA only)
    arrive_count: int               # Threads that must arrive

    def arrive(self, index, dst, cta_group): ...   # Dispatches by op_type
    def wait(self, index, phase): ...               # mbarrier.try_wait in a loop
    def try_wait(self, index, phase): ...           # Single mbarrier.try_wait
    def get_barrier(self, index): ...               # Pointer to barrier[index]
```

### C++: `ClusterBarrier` / `ClusterTransactionBarrier` (`arch/barrier.h:342-664`)

```cpp
struct ClusterBarrier {
    uint64_t barrier_;                             // mbarrier object in smem

    void init(uint32_t arrive_count);              // mbarrier.init
    void wait(uint32_t phase);                     // mbarrier.try_wait.parity in loop
    bool try_wait(uint32_t phase);                 // Single mbarrier.try_wait.parity
    bool test_wait(uint32_t phase, uint32_t pred); // Predicated test
    void arrive();                                 // mbarrier.arrive (local CTA)
    void arrive(uint32_t cta_id, uint32_t pred);   // mbarrier.arrive (remote, cluster-wide)
};

struct ClusterTransactionBarrier : ClusterBarrier {
    void arrive_and_expect_tx(uint32_t bytes);     // mbarrier.arrive.expect_tx
    void expect_transaction(uint32_t bytes);       // mbarrier.expect_tx (no arrive)
    void complete_transaction(uint32_t bytes, ...);// mbarrier.complete_tx
};
```

### Mapping

| Python `MbarrierArray` | C++ | PTX instruction |
|------------------------|-----|-----------------|
| `arrive(index)` with `AsyncThread` | `ClusterBarrier::arrive()` | `mbarrier.arrive` |
| `arrive(index)` with `TmaLoad` | `ClusterTransactionBarrier::arrive_and_expect_tx()` | `mbarrier.arrive.expect_tx` |
| `wait(index, phase)` | `ClusterBarrier::wait(phase)` | `mbarrier.try_wait.parity` in loop |
| `try_wait(index, phase)` | `ClusterBarrier::try_wait(phase)` | Single `mbarrier.try_wait.parity` |
| `get_barrier(index)` | Direct pointer arithmetic on array | `&full_barrier_[stage]` |
| `mbarrier_init()` | `ClusterBarrier::init(arrive_count)` | `mbarrier.init` |

---

## 3. PipelineAsync — Generic Producer-Consumer Pipeline

The general-purpose async pipeline using **two mbarrier arrays** (full/empty) for non-TMA workloads.

### Python (`sm90.py:36-306`)

```python
class PipelineAsync:
    sync_object_full:  MbarrierArray   # "Data is ready" barriers
    sync_object_empty: MbarrierArray   # "Slot is free" barriers
    num_stages: int

    @staticmethod
    def create(num_stages, producer_group, consumer_group, barrier_storage):
        # Allocates two mbarrier arrays from barrier_storage
        # Initializes all barriers, fences, syncs
        ...

    def producer_acquire(self, state, try_acquire_token):
        self.sync_object_empty.wait(state.index, state.phase)  # Wait for slot

    def producer_commit(self, state):
        self.sync_object_full.arrive(state.index, ...)         # Signal data ready

    def consumer_wait(self, state, try_wait_token):
        self.sync_object_full.wait(state.index, state.phase)   # Wait for data

    def consumer_release(self, state):
        self.sync_object_empty.arrive(state.index, ...)        # Signal slot free

    def make_participants(self) -> (PipelineProducer, PipelineConsumer): ...
```

### C++ (`sm90_pipeline.hpp:1015-1240`)

```cpp
template <int Stages_>
class PipelineAsync {
    using FullBarrier  = ClusterBarrier;
    using EmptyBarrier = ClusterBarrier;

    struct SharedStorage {
        FullBarrier  full_barrier_[Stages];
        EmptyBarrier empty_barrier_[Stages];
    };

    void producer_acquire(PipelineState state) {
        empty_barrier_ptr_[state.index()].wait(state.phase());
    }
    void producer_commit(PipelineState state) {
        full_barrier_ptr_[state.index()].arrive();
    }
    void consumer_wait(PipelineState state) {
        full_barrier_ptr_[state.index()].wait(state.phase());
    }
    void consumer_release(PipelineState state) {
        empty_barrier_ptr_[state.index()].arrive(dst_blockid_, pred);
    }
};
```

### State Transition Diagram

```
     full_barrier[i]                  empty_barrier[i]
     ───────────────                  ─────────────────

Producer                              Consumer
   │                                     │
   │  ┌──── acquire ────┐                │
   │  │ wait on empty[i] │◄──── release ──┤  arrive on empty[i]
   │  │ (phase match?)   │               │
   │  └────────┬─────────┘               │
   │           │ granted                  │
   │           ▼                          │
   │     write to smem                    │
   │           │                          │
   │  ┌──── commit ─────┐                │
   │  │ arrive full[i]  │────── wait ───►│  wait on full[i]
   │  └─────────────────┘                │  (phase match?)
   │                                     │
   │                                     ▼
   │                               read from smem
   │                                     │
   └─────────────────────────────────────┘
              (next iteration)
```

---

## 4. PipelineProducer / PipelineConsumer — High-Level Wrappers

These are the primary user-facing classes. They wrap a `PipelineAsync` + `PipelineState` and return **immutable handles** for safety.

### Python: `PipelineProducer` (`sm90.py:935-1133`)

```python
class PipelineProducer:
    __pipeline: PipelineAsync
    __state:    PipelineState
    __group:    CooperativeGroup

    class ImmutableResourceHandle:
        # Frozen snapshot of state at acquire time
        @property index:   Int32    # Stage index for smem addressing
        @property count:   Int32    # Operation count
        @property barrier: Pointer  # Pointer to full_barrier[index]
        def commit(self): ...       # Calls pipeline.producer_commit

    def acquire(self) -> ImmutableResourceHandle:
        pipeline.producer_acquire(self.__state, ...)
        return ImmutableResourceHandle(pipeline, state.clone())

    def advance(self):
        self.__state.advance()

    def acquire_and_advance(self) -> ImmutableResourceHandle:
        handle = self.acquire()
        self.advance()
        return handle

    def commit(self, handle=None):
        pipeline.producer_commit(handle.__state or self.__state)

    def tail(self):
        pipeline.producer_tail(self.__state)
```

### Python: `PipelineConsumer` (`sm90.py:1135-1309`)

```python
class PipelineConsumer:
    __pipeline: PipelineAsync
    __state:    PipelineState
    __group:    CooperativeGroup

    class ImmutableResourceHandle:
        @property index: Int32
        def release(self): ...    # Calls pipeline.consumer_release

    def wait(self) -> ImmutableResourceHandle:
        pipeline.consumer_wait(self.__state, ...)
        return ImmutableResourceHandle(pipeline, state.clone())

    def advance(self):
        self.__state.advance()

    def wait_and_advance(self) -> ImmutableResourceHandle:
        handle = self.wait()
        self.advance()
        return handle

    def release(self, handle=None):
        pipeline.consumer_release(handle.__state or self.__state)
```

### C++ Equivalent

There is **no direct C++ equivalent** for `PipelineProducer`/`PipelineConsumer` as separate classes. In C++, the producer/consumer role is encoded via a `ThreadCategory` enum and the raw `PipelineState` is managed manually by the caller:

```cpp
// C++ usage pattern (no wrapper class):
PipelineState state = make_producer_start_state<PipelineAsync<Stages>>();

for (int i = 0; i < N; ++i) {
    pipeline.producer_acquire(state);        // Wait for empty slot
    // ... write to smem[state.index()] ...
    pipeline.producer_commit(state);         // Signal data ready
    ++state;                                 // Advance (caller's responsibility)
}
pipeline.producer_tail(state);               // Drain remaining stages
```

### ImmutableResourceHandle — DSL-Only Abstraction

`ImmutableResourceHandle` (`sm90.py:888-932`) has **no C++ counterpart**. It exists in the DSL for two reasons:

1. **Safety**: Freezes the state at `acquire`/`wait` time, preventing the user from accidentally using a stale or advanced index for `commit`/`release`.
2. **MLIR codegen**: Its `__extract_mlir_values__` / `__new_from_mlir_values__` methods allow the handle's state to flow through the MLIR SSA value system, enabling the compiler to track which barrier a `commit()` or `release()` targets.

In C++, the caller is trusted to pass the correct `PipelineState` to each operation. The DSL adds this safety layer on top.

---

## 5. PipelineTmaAsync — TMA-Specialized Pipeline

For cases where the producer issues TMA (Tensor Memory Accelerator) loads rather than explicit stores.

### Python (`sm90.py:368-558`)

```python
class PipelineTmaAsync(PipelineAsync):
    is_signalling_thread: Boolean

    def producer_acquire(self, state, ...):
        self.sync_object_empty.wait(state.index, state.phase)
        # Immediately set transaction count on full barrier
        self.sync_object_full.arrive_and_expect_tx(state.index, tx_count)

    def producer_commit(self, state):
        pass  # NOP — TMA hardware signals the barrier on completion

    def consumer_release(self, state):
        # Only signalling thread arrives on empty barrier
        self.sync_object_empty.arrive(state.index, cond=is_signalling_thread)
```

### C++ (`sm90_pipeline.hpp:271-642`)

```cpp
template <int Stages_>
class PipelineTmaAsync {
    using FullBarrier  = ClusterTransactionBarrier;  // Note: Transaction variant
    using EmptyBarrier = ClusterBarrier;

    void producer_acquire(uint32_t stage, uint32_t phase) {
        empty_barrier_ptr_[stage].wait(phase);
        if (params_.is_leader) {
            full_barrier_ptr_[stage].arrive_and_expect_tx(params_.transaction_bytes);
        }
    }

    void producer_commit(uint32_t stage, uint32_t bytes) {
        // NOP for TMA — hardware signals barrier via complete_transaction
    }

    void consumer_release(uint32_t stage, uint32_t skip = false) {
        empty_barrier_ptr_[stage].arrive(dst_blockid_, is_signaling_thread_ & (!skip));
    }
};
```

### Key Differences from PipelineAsync

| Aspect | PipelineAsync | PipelineTmaAsync |
|--------|--------------|-----------------|
| Full barrier type | `ClusterBarrier` | `ClusterTransactionBarrier` |
| Producer commit | Explicit `arrive()` | **NOP** — TMA hardware signals |
| Acquire side-effect | None | Sets `expect_tx` on full barrier |
| Consumer release | All threads arrive | Only **signalling thread** arrives |

---

## 6. Other Pipeline Variants

| Python Class | C++ Equivalent | Use Case |
|-------------|---------------|----------|
| `PipelineCpAsync` (`sm90.py:309-365`) | N/A (folded into `PipelineAsync` usage) | `cp.async` (non-TMA) loads |
| `PipelineTmaStore` (`sm90.py:729-776`) | Epilogue store patterns | TMA stores, uses `TmaStoreFence` not mbarriers |
| `PipelineOrder` (`sm90.py:779-881`) | Order barriers in GEMM mainloops | Single mbarrier array for ordered execution |

---

## 7. Barrier Token / ArrivalToken Pattern

### Python

```python
# try_acquire returns a boolean token
token = pipeline.producer_try_acquire(state)
# Later, acquire checks the token:
pipeline.producer_acquire(state, try_acquire_token=token)
# If token was True (WaitDone), acquire returns immediately
```

### C++ (`sm90_pipeline.hpp:113-166`)

```cpp
enum class BarrierStatus : uint32_t {
    WaitAgain = 0u,
    WaitDone  = 1u,
};

class ProducerToken : public ArrivalToken { ... };
class ConsumerToken : public ArrivalToken { ... };

// Usage:
ProducerToken token = pipeline.producer_try_acquire(state);
pipeline.producer_acquire(state, token);  // Skips wait if token == WaitDone
```

This pattern enables **latency hiding**: issue a non-blocking `try_acquire`, do independent work, then finalize the acquire only blocking if the barrier wasn't ready.

---

## 8. Complete API Mapping Table

| DSL (Python) | C++ | PTX / Hardware |
|-------------|-----|----------------|
| `PipelineAsync.create()` | `PipelineAsync::init_barriers()` | `mbarrier.init` × 2×Stages |
| `PipelineProducer.acquire()` | `pipeline.producer_acquire(state)` | `mbarrier.try_wait.parity` on empty barrier |
| `PipelineProducer.commit()` | `pipeline.producer_commit(state)` | `mbarrier.arrive` on full barrier |
| `PipelineProducer.advance()` | `++state` (caller) | N/A (index arithmetic) |
| `PipelineProducer.acquire_and_advance()` | `acquire` + `++state` | Combined |
| `PipelineProducer.tail()` | `pipeline.producer_tail(state)` | Wait-loop draining empty barriers |
| `PipelineConsumer.wait()` | `pipeline.consumer_wait(state)` | `mbarrier.try_wait.parity` on full barrier |
| `PipelineConsumer.release()` | `pipeline.consumer_release(state)` | `mbarrier.arrive` on empty barrier |
| `PipelineConsumer.advance()` | `++state` (caller) | N/A (index arithmetic) |
| `PipelineConsumer.wait_and_advance()` | `wait` + `++state` | Combined |
| `ImmutableResourceHandle.index` | `state.index()` | Used for smem addressing |
| `ImmutableResourceHandle.commit()` | `pipeline.producer_commit(state)` | `mbarrier.arrive` on full barrier |
| `ImmutableResourceHandle.release()` | `pipeline.consumer_release(state)` | `mbarrier.arrive` on empty barrier |
| `PipelineProducer.try_acquire()` | `pipeline.producer_try_acquire(state)` | Single `mbarrier.try_wait.parity` |
| `PipelineConsumer.try_wait()` | `pipeline.consumer_try_wait(state)` | Single `mbarrier.try_wait.parity` |

---

## Process

Delegated two parallel research agents:
1. **Python agent** — read `pipeline/sm90.py` (1309 lines), `pipeline/helpers.py` (835 lines), and `pipeline/__init__.py`
2. **C++ agent** — read `pipeline/sm90_pipeline.hpp` (1340 lines) and `arch/barrier.h` (664 lines)

Then synthesized the mappings by matching method signatures, barrier types, and PTX instructions across the two codebases.
