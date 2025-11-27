# NVFP4 BlockScaled GEMM: Architecture Diagram & Flow

## System-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         HOST (CPU)                                   │
│                                                                       │
│  ptr_A, ptr_B (FP4/F6/F8 compressed)                               │
│  ptr_SFA, ptr_SFB (FP8 scale factors)                              │
│  Arguments → Params → TMA Descriptors                              │
└────────────┬────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    GLOBAL MEMORY (GMEM)                              │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────┐  ┌─────────┐     │
│  │  Matrix A    │  │  Matrix B    │  │ SFA (K) │  │ SFB (K) │     │
│  │ 128×256 (FP4)│  │ 128×256 (FP4)│  │ 64 elem │  │ 64 elem │     │
│  │   16 KB      │  │   16 KB      │  │ 256 B   │  │ 256 B   │     │
│  └──────────────┘  └──────────────┘  └─────────┘  └─────────┘     │
└────┬─────────────┬─────────────────┬─────────┬────────────────────┘
     │             │                 │         │
     │ TMA Load    │ TMA Load        │         │
     │ (16KB)      │ (16KB)          │         │
     │             │                 │ TMA Load│ TMA Load
     │             │                 │(256B)   │(256B)
     ▼             ▼                 ▼         ▼
┌─────────────────────────────────────────────────────────────────────┐
│              SHARED MEMORY (SMEM) - Per Stage                        │
│                                                                       │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───┐  ┌───┐      │
│  │ smem_A            │  │ smem_B            │  │SFA│  │SFB│      │
│  │ (MMA_M, MMA_K)    │  │ (MMA_N, MMA_K)    │  │   │  │   │      │
│  │ 128×256 layout    │  │ 128×256 layout    │  │32×4   │32×4│     │
│  │ Swizzled          │  │ Swizzled          │  │elem   │elem│     │
│  └───────────────────┘  └───────────────────┘  └───┘  └───┘      │
│                                                                       │
│  Stages: [0] [1] [2] ... [19]  (double-buffered pipeline)         │
└─────────────┬──────────────────────────────────┬──────────────────┘
              │                                  │
              │ (Producer signals FULL barrier) │
              │                                  │
              ▼                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│           UNIFIED MEMORY (TMEM) - Register-like                      │
│                                                                       │
│  Thread-Local TMEM Layout:                                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Accumulators C          (128×128) @ 2 Stages              │   │
│  │ [Main computation results, shared across K-iterations]    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────┐  ┌──────────────────────┐               │
│  │ tCtSFA               │  │ tCtSFB               │               │
│  │ (SFVecSize,MMA_NSF)  │  │ (SFVecSize,MMA_NSF)  │               │
│  │ e.g., (4, 16)        │  │ e.g., (4, 16)        │               │
│  │ 64 SF elements each  │  │ 64 SF elements each  │               │
│  └──────────────────────┘  └──────────────────────┘               │
│                                                                       │
│  Allocated via:                                                     │
│  - FrgTypeSFA, FrgTypeSFB (custom fragment types)                  │
│  - Uses TMEM allocation modes (ScaleFactorDuplicated)              │
│  - Register allocation per thread                                   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           │ UTCCP Copy
                           │ (SMEM→TMEM)
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│              UMMA INSTRUCTION (tcgen05.mma)                          │
│                                                                       │
│  Datapath:                                                          │
│                                                                       │
│  A[i,k] (FP4) ──────────────────────────────────────┐             │
│              Decompress & Expand                    │             │
│                         │                           │             │
│                         ▼                           │             │
│                    [float32]                        │             │
│                         │                           │             │
│  SFA[i] ──────────────►× (scale row)               │             │
│  (float)                │                           │             │
│                         ▼                           ▼             │
│                    [float32] ×╮                                   │
│                                 ├──► MUL+ACC ──► C[i,j]         │
│                    [float32] ◄─╯                                  │
│                         ▲                           ▲             │
│                         │                           │             │
│  SFB[j] ──────────────►× (scale col)               │             │
│  (float)                │                           │             │
│                         ▼                           │             │
│                    [float32]                        │             │
│                    Decompress & Expand             │             │
│                         │                           │             │
│  B[k,j] (FP4) ──────────────────────────────────────┘             │
│                                                                       │
│  ScaleOut Mode:                                                    │
│  - k_block=0: ScaleOut::Zero   → C = A*SFA @ B*SFB               │
│  - k_block>0: ScaleOut::One    → C += A*SFA @ B*SFB              │
└─────────────────────────────────────────────────────────────────────┘
                           │
                           │ Result stored in TMEM accumulators
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    EPILOGUE / OUTPUT                                 │
│                  (not covered in this trace)                         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Timing Diagram

```
Timeline for One K-Tile Iteration:

PRODUCER WARP (Thread 0 of each CTA):
┌─────────────────────────────────────────────────────┐
│ t0: TMA Load A → SMEM[stage 0]                      │
│     (async, hardware transfers data)                │
└─────────────────────────────────────────────────────┘
     ↓ (next instruction)
┌─────────────────────────────────────────────────────┐
│ t1: TMA Load B → SMEM[stage 0]                      │
│     (async)                                          │
└─────────────────────────────────────────────────────┘
     ↓
┌─────────────────────────────────────────────────────┐
│ t2: TMA Load SFA → SMEM[stage 0]                    │
│     (async, 256 bytes only)                         │
└─────────────────────────────────────────────────────┘
     ↓
┌─────────────────────────────────────────────────────┐
│ t3: TMA Load SFB → SMEM[stage 0]                    │
│     (async, 256 bytes only)                         │
│     All 4 TMA → Signal FULL_barrier[0]              │
└─────────────────────────────────────────────────────┘
     ↓ (hardware signal)
┌─────────────────────────────────────────────────────┐
│ t4: All threads blocked on FULL_barrier[0]          │
│                                                      │
│ ┌──────────────────────────────────────────────┐   │
│ │ CONSUMER WARP (All 32 threads):              │   │
│ │ t4: Wait for FULL_barrier[0] → RELEASED      │   │
│ └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
     ↓
┌─────────────────────────────────────────────────────┐
│ t5: Thread 0 issues UTCCP commands:                 │
│     copy(SMEM_SFA[0] → TMEM_tCtSFA)  [async]      │
│     copy(SMEM_SFB[0] → TMEM_tCtSFB)  [async]      │
└─────────────────────────────────────────────────────┘
     ↓ (doesn't wait, all threads proceed)
┌─────────────────────────────────────────────────────┐
│ t6-t7: All threads execute MMA loop:               │
│                                                      │
│  for (int k = 0; k < K_BLOCKS; ++k) {              │
│    // TMEM copy may still be in flight            │
│    // (but MMA instruction waits on register dep)  │
│    tcgen05.mma [C], A_desc[k], B_desc[k],         │
│                 SCALE_OUT, tCtSFA[k], tCtSFB[k]   │
│  }                                                   │
└─────────────────────────────────────────────────────┘
     ↓
┌─────────────────────────────────────────────────────┐
│ t8: Thread 0 signals EMPTY_barrier[0]              │
│     (tells producer stage 0 is now free)            │
└─────────────────────────────────────────────────────┘
     ↓
┌─────────────────────────────────────────────────────┐
│ t9: Producer (blocked on EMPTY) continues:         │
│     Loads next K-tile into stage 1                 │
│     TMA A, B, SFA, SFB → stage 1                   │
└─────────────────────────────────────────────────────┘

Key Overlaps:
├─ TMA transfers (A,B,SFA,SFB) overlap with each other
├─ UTCCP copy overlaps with MMA execution (async)
├─ Next producer iteration (stage N+1) overlaps with consumer (stage N)
└─ All hidden by MMA latency (no stalls in steady state)
```

---

## Memory Hierarchy & Scale Factor Flow

```
Layer 1: Global Memory (GMEM) - Device DRAM
┌────────────────────────────────────────────┐
│ Dense storage of A, B, SFA, SFB matrices   │
│ Stride-based access                        │
│ Bandwidth: ~2TB/s (SM100)                  │
└────────────────────────────────────────────┘
              │
              │ TMA (Hardware DMA)
              │ - Descriptor-based
              │ - Multi-box transfers
              │ - ~1000 cycles latency
              ▼
Layer 2: Shared Memory (SMEM) - Per-CTA Cache
┌────────────────────────────────────────────┐
│ [Stage 0]  [Stage 1]  ...  [Stage N]      │
│                                            │
│ Each Stage:                                │
│ ├─ smem_A: 128×256 (16 KB)                │
│ ├─ smem_B: 128×256 (16 KB)                │
│ ├─ smem_SFA: 32×4 (256 B)  ◄─ SCALE       │
│ ├─ smem_SFB: 32×4 (256 B)  ◄─ FACTORS    │
│ │                                         │
│ └─ Double/triple buffered                 │
│                                            │
│ Bandwidth: ~10TB/s (internal)              │
│ Latency: ~10-100 cycles                    │
└────────────────────────────────────────────┘
              │
              │ UTCCP (Hardware Async Copy)
              │ - Descriptor-based SMEM→TMEM
              │ - Single instruction per block
              │ - ~50 cycles latency
              ▼
Layer 3: Tensor Memory (TMEM) - High-BW Per-Thread
┌────────────────────────────────────────────┐
│ Thread-Local TMEM Registers (128KB per SM) │
│                                            │
│ Per-Thread Layout:                         │
│ ├─ Accumulator C: 128×128 (4 stages)      │
│ ├─ tCtSFA: (4,16)  [64 SF elements]       │
│ ├─ tCtSFB: (4,16)  [64 SF elements]       │
│ │                                         │
│ │ tCtSFA[i,j] → scales row i of A        │
│ │ tCtSFB[i,j] → scales column j of B     │
│ │                                         │
│ └─ Other thread temporaries                │
│                                            │
│ Bandwidth: ~100TB/s (theoretical)          │
│ Latency: 0 cycles (operand register)       │
└────────────────────────────────────────────┘
              │
              │ Register Operands (no movement)
              │
              ▼
Layer 4: MMA Execution Unit
┌────────────────────────────────────────────┐
│ UMMA Hardware Pipeline                     │
│                                            │
│ Fetch: A[i,k] from SMEM descriptor        │
│        B[k,j] from SMEM descriptor        │
│        SFA[i] from TMEM register          │
│        SFB[j] from TMEM register          │
│                                            │
│ Process:                                   │
│ 1. Decompress A[i,k], B[k,j]              │
│ 2. A[i,k] *= SFA[i]                       │
│ 3. B[k,j] *= SFB[j]                       │
│ 4. C[i,j] += A[i,k] · B[k,j]              │
│                                            │
│ Result: C[i,j] in TMEM accumulator        │
└────────────────────────────────────────────┘
```

---

## Scale Factor Blocking Structure

```
128-row Matrix Block:
┌──────────────────────────────────┐
│ Row 0-31   │ Row 32-63  │ Row 64-95  │ Row 96-127
│  ▲         │  ▲         │  ▲         │  ▲
│  │         │  │         │  │         │  │
│ SFA[0]     │ SFA[1]     │ SFA[2]     │ SFA[3]
│ (1 value)  │ (1 value)  │ (1 value)  │ (1 value)
│            │            │            │
│◄─ 32 SFs ─►│            │            │
│ (128/4)    │            │            │
└──────────────────────────────────┘

Each block of 32 elements compressed with:
  - SFVecSize = 4 or 8 (typical)
  - E.g., with SFVecSize=4: 128 elements ÷ 4 = 32 SFs needed
  - Organized as: (32 SFs) / 4 = 8 groups × 4 SFs

SMEM Layout for 64×K scale factors:
┌──────────────────────┐
│ SF Block (32×4)      │
│ ┌────┬────┬────┬────┐
│ │ g0 │ g1 │ g2 │ g3 │  ◄─ 4 "groups" of SFs
│ ├────┼────┼────┼────┤
│ │    │    │    │    │  ◄─ Rows 0-31 of A
│ │    │    │    │    │
│ │ 32 │ SF │    │    │
│ │ SFs│ per│    │    │
│ │    │row │    │    │
│ └────┴────┴────┴────┘
│
│ Dimensions: (32 columns) × (4 subblocks)
│            = 128 elements represented
│ Total: 32 SFs for 128 row elements
└──────────────────────────────────┘

In TMEM:
┌──────────────────────┐
│ tCtSFA (4×16)        │
│ ┌────┬────┬─...─┬──┐ 
│ │ SF │ SF │     │SF│
│ │ g0 │ g1 │ ... │g3│ = 4 groups (SFVecSize dim)
│ └────┴────┴─...─┴──┘
│  └──────┬──────┘
│    MMA blocks (16 total for K dimension)
└──────────────────────────────┘
```

---

## Special Case: N=192 CTA Handling

```
Problem: Tile N=192, but alignment granularity is 128

TileShape_SF padding solution:
┌─────────────────────────────────┐
│ Original: N = 192               │
│ Padded:   N_SF = 256            │
│ (ceil_div(192, 128) × 128)      │
└─────────────────────────────────┘

SMEM/GMEM Layout (SFB):
┌──────────────────────────────────────────┐
│ SFB with 256 columns (padded)            │
├──────────────┬──────────────┬────────────┤
│ Cols 0-127   │ Cols 128-255 │ Unused     │
│              │              │ (cols      │
│              │              │  192-256)  │
└──────────────┴──────────────┴────────────┘

CTA Partition:
┌─────────┬─────────┬─────────┬─────────┐
│ CTA[0]  │ CTA[1]  │ CTA[2]  │ CTA[3]  │
├─────────┼─────────┼─────────┼─────────┤
│ Col 0   │ Col 64  │ Col 128 │ Col 192 │
│ to      │ to      │ to      │ to      │
│ Col 63  │ Col 127 │ Col 191 │ Col 255 │
└─────────┴─────────┴─────────┴─────────┘

TMEM Offset Logic:
CTA even-index: No offset
  tCtSFB = base address (covers cols 0-128 or 128-192)

CTA odd-index: +2 words offset
  tCtSFB.data() = base + 2 (skips first 64 SFs)
  (each word ≈ 32 SFs, so +2 = 64 SFs = 64 columns)

Mapping:
├─ CTA[0] (even, N=64): SFB[0:64] ────────► cols 0-64
├─ CTA[1] (odd, N=64):  SFB[64:128] ──────► cols 64-128
├─ CTA[2] (even, N=128): SFB[0:64] + offset► cols 128-192
│                         Computed: tCtSFB.data() + 2
└─ CTA[3] (odd, N=128):  SFB[64:128] + offset ► cols 192-256
                         Computed: tCtSFB.data() + 2
```

---

## Barrier Synchronization Pattern

```
Dual Barrier System:

FULL Barrier: Signals "Data is Ready"
├─ Producer signals when all TMA operations complete
│  (A, B, SFA, SFB all written to SMEM)
│
└─ Consumer waits before reading
   (Ensures data consistency)

EMPTY Barrier: Signals "Data is Consumed"
├─ Consumer signals when MMA phase complete
│  (Read all data, wrote accumulators)
│
└─ Producer waits before overwriting
   (Ensures no RAW hazard)

State Machine per Stage:

        ┌─────────────┐
        │ FULL[0]=0   │
        │ EMPTY[0]=0  │
        └──────┬──────┘
               │
      Producer waits on
      EMPTY[0] phase flip
               │
               ▼
        ┌─────────────┐
        │ PRODUCER    │
        │ Writes A,B  │
        │ SFA,SFB to  │
        │ SMEM[0]     │
        └──────┬──────┘
               │
      TMA signals FULL[0]
      (4 arrivals needed)
               │
               ▼
        ┌─────────────────┐
        │ FULL[0] phase→1 │  FULL barrier flips
        │ Consumer wakes  │
        └──────┬──────────┘
               │
      Consumer reads
      SMEM[0] → TMEM
      Executes MMA
               │
               ▼
        ┌──────────────────┐
        │ CONSUMER signals │
        │ EMPTY[0] phase→1 │
        └──────┬───────────┘
               │
      Producer (waiting on
      EMPTY flip) continues
      with next stage
               │
               ▼
        ┌─────────────────┐
        │ FULL[1]=0       │
        │ EMPTY[1]=0      │  (next iteration, stage N+1)
        │ (pattern repeats)│
        └─────────────────┘

All 4 TMA operations → SAME barrier:
✓ Ensures atomicity
✓ No partial data states
✓ Scale factors synchronized with main data
```

---

## Summary: Complete Data Journey

```
┌─── GMEM: User-provided scale factor arrays
│   ptr_SFA: float8[M/128][K/SFVecSize]
│   ptr_SFB: float8[N/128][K/SFVecSize]
│
├─── TMA Setup: Descriptors created with:
│   - Start addresses from ptr_SFA, ptr_SFB
│   - Box dimensions from SmemLayoutSFA, SmemLayoutSFB
│   - Element type: uint16_t (for TMA)
│
├─── Load Phase: Producer thread issues TMA
│   copy(tma_load_sfa_, GMEM_offset, SMEM[stage])
│   copy(tma_load_sfb_, GMEM_offset, SMEM[stage])
│   Both arrive at FULL_barrier (same as A,B data)
│
├─── SMEM Storage: Double-buffered across Stages
│   smem_SFA[stage]: (32, 4) layout with swizzle
│   smem_SFB[stage]: (32, 4) layout with swizzle
│   Compact packing: ~256 bytes per stage
│
├─── SMEM→TMEM Copy: Once per K-iteration
│   Thread 0: UTCCP async copy (fire-and-forget)
│   tCtSFA ← smem_SFA[stage]
│   tCtSFB ← smem_SFB[stage]
│
├─── TMEM Resident: Per-thread registers
│   tCtSFA: (SFVecSize, MMA_NSF) = (4, 16) example
│   tCtSFB: (SFVecSize, MMA_NSF) = (4, 16) example
│   Ready for MMA operand
│
├─── MMA Execution: With scale factors
│   tiled_mma.with(accumulate_, tCtSFA[k], tCtSFB[k])
│   tcgen05.mma [C], A_desc[k], B_desc[k],
│               SCALE_OUT, idescE
│   (idescE contains scale factor enable bits)
│
└─── Hardware Scaling:
    Inside UMMA pipeline:
    A'[i,k] = A[i,k] × SFA[i]
    B'[k,j] = B[k,j] × SFB[j]
    C[i,j] += A'[i,k] · B'[k,j]
```

---

END OF ARCHITECTURE DIAGRAM
