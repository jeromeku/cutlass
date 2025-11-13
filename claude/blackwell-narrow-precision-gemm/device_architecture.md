# Device-Side Architecture and Execution

This document provides a detailed explanation of the device-side execution architecture for Blackwell narrow precision GEMM kernels.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Warp-Specialized Design](#warp-specialized-design)
3. [Memory Hierarchy](#memory-hierarchy)
4. [Pipeline Execution](#pipeline-execution)
5. [MMA Instruction Details](#mma-instruction-details)
6. [Epilogue Execution](#epilogue-execution)
7. [Synchronization Mechanisms](#synchronization-mechanisms)

---

## Architecture Overview

### High-Level Kernel Structure

```
┌──────────────────────────────────────────────────────────────┐
│                    CUDA Kernel Launch                        │
│                   (Grid of CTAs/Blocks)                       │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│              Single CTA (Threadblock)                        │
│  ┌────────────────────────────────────────────────────────┐  │
│  │          Warp Specialization Manager                   │  │
│  │  - Assigns roles to warps (Producer/Consumer/Epilogue)│  │
│  │  - Manages synchronization barriers                   │  │
│  └────────────────────────────────────────────────────────┘  │
│                            │                                  │
│              ┌─────────────┴─────────────┐                   │
│              ▼                           ▼                   │
│  ┌──────────────────────┐   ┌──────────────────────┐        │
│  │   Producer Warps     │   │   Consumer Warps     │        │
│  │  - Load from GMEM    │   │  - Execute MMA       │        │
│  │  - Store to TMEM/SMEM│   │  - Use TMEM data     │        │
│  │  - TMA async copies  │   │  - Accumulate results│        │
│  └──────────────────────┘   └──────────────────────┘        │
│              │                           │                   │
│              └─────────────┬─────────────┘                   │
│                            ▼                                  │
│              ┌──────────────────────┐                        │
│              │   Epilogue Warps     │                        │
│              │  - Load C from GMEM  │                        │
│              │  - Perform fusion    │                        │
│              │  - Store D to GMEM   │                        │
│              │  - Generate SF       │                        │
│              └──────────────────────┘                        │
└──────────────────────────────────────────────────────────────┘
```

### Key Architectural Features

1. **Warp Specialization**: Different warps perform different roles
2. **Tensor Memory (TMEM)**: Per-SM fast memory for data reuse
3. **Asynchronous Operations**: TMA and MMA operate concurrently
4. **Pipeline Execution**: Multiple stages in flight simultaneously

---

## Warp-Specialized Design

### Traditional vs. Warp-Specialized

**Traditional GEMM** (all warps do the same thing):
```
All Warps: Load A → Load B → Compute MMA → Epilogue → Write D
           └────────────── Sequential ──────────────┘
```

**Warp-Specialized GEMM** (Blackwell):
```
Producer Warps:  ┌─ Load A/SFA ─┐ ┌─ Load A/SFA ─┐ ┌─ Load A/SFA ─┐
                 │  Load B/SFB  │ │  Load B/SFB  │ │  Load B/SFB  │
                 └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
                        │                 │                 │
                    (Barrier)         (Barrier)         (Barrier)
                        │                 │                 │
Consumer Warps:         └─── Compute MMA ┴─── Compute MMA ┴─ ...
                            on TMEM data     on TMEM data

                        (Accumulator passes to Epilogue)

Epilogue Warps:         Load C → Fusion → Quantize → Write D/SFD
```

### Warp Role Assignment

**In a typical 1SM configuration** (128×128×256 tile):

| Role | # Warps | Responsibilities |
|------|---------|------------------|
| Producer | 2-4 | TMA loads from GMEM to TMEM |
| Consumer | 4-8 | MMA computations using TMEM |
| Epilogue | 2-4 | Epilogue fusion and output |

**Mapping in code**:
```cpp
// Pseudo-code showing warp role assignment
__device__ void kernel(...) {
  int warp_id = threadIdx.x / 32;
  int warp_role = get_warp_role(warp_id);  // From dispatch policy

  switch (warp_role) {
    case PRODUCER:
      producer_warp_main_loop();
      break;
    case CONSUMER:
      consumer_warp_main_loop();
      break;
    case EPILOGUE:
      epilogue_warp();
      break;
  }
}
```

### Benefits of Specialization

1. **Parallelism**: Producers and consumers run concurrently
2. **Latency Hiding**: Memory loads overlap with computation
3. **Resource Utilization**: Better register and SMEM usage
4. **Flexibility**: Can tune # of warps per role for different problem sizes

---

## Memory Hierarchy

### Memory Types and Usage

```
┌───────────────────────────────────────────────────────────────┐
│                    Global Memory (GMEM)                        │
│                                                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ A (FP4)  │  │ SFA (FP8)│  │ B (FP4)  │  │ SFB (FP8)│     │
│  │ M×K      │  │ (M/128)× │  │ N×K      │  │ (N/128)× │     │
│  │          │  │ (K/16)   │  │          │  │ (K/16)   │     │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │
└───────────────────────────────────────────────────────────────┘
       │              │              │              │
       │  TMA Copy    │  TMA Copy    │  TMA Copy    │  TMA Copy
       ▼              ▼              ▼              ▼
┌───────────────────────────────────────────────────────────────┐
│              Tensor Memory (TMEM) - Per SM                     │
│                      (Up to 1MB)                               │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Stage 0: A₀, SFA₀, B₀, SFB₀                            │ │
│  │  Stage 1: A₁, SFA₁, B₁, SFB₁                            │ │
│  │  Stage 2: A₂, SFA₂, B₂, SFB₂                            │ │
│  │  ...                                                     │ │
│  └──────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘
                            │
                            │ Read by Consumer Warps
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                  Register File (RF)                            │
│                                                                │
│  ┌────────────────────┐      ┌────────────────────┐          │
│  │  MMA Input Regs    │      │  Accumulator Regs  │          │
│  │  - A fragment      │      │  - FP32 accum      │          │
│  │  - B fragment      │      │  - 4×4×1 per       │          │
│  │  - SF values       │      │    thread          │          │
│  └────────────────────┘      └────────────────────┘          │
└───────────────────────────────────────────────────────────────┘
                            │
                            │ Passed to Epilogue
                            ▼
┌───────────────────────────────────────────────────────────────┐
│               Shared Memory (SMEM) - Per CTA                   │
│                    (Up to 232KB)                               │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Pipeline State (Barriers, Semaphores)                   │ │
│  │  Epilogue Scratch Space                                  │ │
│  │  Reduction Buffers                                       │ │
│  └──────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘
```

### Tensor Memory (TMEM) Details

**Key Characteristics**:
- **Size**: Up to 1MB per SM on Blackwell
- **Scope**: Shared across all CTAs on an SM
- **Access**: All warps in CTA can read
- **Management**: Software-managed (no hardware cache)
- **Purpose**: Staging area for MMA operands

**Usage Pattern**:
```
Producer: Write TMA → TMEM (Stage 0)
Consumer: Read TMEM (Stage 0) → MMA
Producer: Write TMA → TMEM (Stage 1)  (overlaps with above)
Consumer: Read TMEM (Stage 1) → MMA
...
```

**Size Calculation**:
```cpp
// Per stage TMEM usage
constexpr auto a_bytes_per_stage = (128 * 256 * 4) / 8;  // 16KB for A tile
constexpr auto sfa_bytes_per_stage = (128/128) * (256/16) * 1;  // 16 bytes for SFA
constexpr auto b_bytes_per_stage = (128 * 256 * 4) / 8;  // 16KB for B tile
constexpr auto sfb_bytes_per_stage = (128/128) * (256/16) * 1;  // 16 bytes for SFB

constexpr auto total_per_stage = a_bytes_per_stage + sfa_bytes_per_stage +
                                  b_bytes_per_stage + sfb_bytes_per_stage;
// ≈ 32KB per stage

// Number of stages that fit in TMEM (after subtracting epilogue carveout)
constexpr int max_stages = (1024*1024 - epilogue_smem - scheduler_smem) / total_per_stage;
// ≈ 20-30 stages possible
```

---

## Pipeline Execution

### Multi-Stage Pipeline

The kernel uses a **software-pipelined** execution model:

```
Time  │ Producer Warps          │ Consumer Warps        │ Epilogue
──────┼─────────────────────────┼───────────────────────┼─────────────
  0   │ Load Stage 0            │ (wait)                │ (wait)
  1   │ Load Stage 1            │ Compute Stage 0       │ (wait)
  2   │ Load Stage 2            │ Compute Stage 1       │ (wait)
  3   │ Load Stage 3            │ Compute Stage 2       │ (wait)
  4   │ Load Stage 4            │ Compute Stage 3       │ (wait)
 ...  │ ...                     │ ...                   │ (wait)
  N   │ (done)                  │ Compute Stage N-1     │ (wait)
 N+1  │ (done)                  │ Compute Stage N       │ Load C
 N+2  │ (done)                  │ (done)                │ Fusion + Write D
```

### Pipeline Stages

**Stage anatomy**:
```
┌─ Stage K ────────────────────────────────────────┐
│                                                   │
│  1. Producer: Issue TMA for A[k], SFA[k]         │
│  2. Producer: Issue TMA for B[k], SFB[k]         │
│  3. Producer: Wait for TMA complete               │
│  4. Producer: Signal barrier (data ready)         │
│                                                   │
│  5. Consumer: Wait on barrier                     │
│  6. Consumer: Load from TMEM to registers         │
│  7. Consumer: Execute MMA instructions            │
│  8. Consumer: Accumulate to RF                    │
│  9. Consumer: Signal barrier (stage done)         │
│                                                   │
└───────────────────────────────────────────────────┘
```

### Code Structure (Simplified)

```cpp
// Producer warp pseudo-code
__device__ void producer_main_loop() {
  for (int stage = 0; stage < num_stages; ++stage) {
    // Issue TMA for A and SFA
    tma_load_async(tma_desc_A, tmem_ptr_A[stage], gmem_ptr_A + stage * tile_A);
    tma_load_async(tma_desc_SFA, tmem_ptr_SFA[stage], gmem_ptr_SFA + stage * tile_SFA);

    // Issue TMA for B and SFB
    tma_load_async(tma_desc_B, tmem_ptr_B[stage], gmem_ptr_B + stage * tile_B);
    tma_load_async(tma_desc_SFB, tmem_ptr_SFB[stage], gmem_ptr_SFB + stage * tile_SFB);

    // Wait for TMAs to complete
    tma_wait();

    // Signal consumers that data is ready
    barrier_arrive(barrier_producers[stage]);
  }
}

// Consumer warp pseudo-code
__device__ void consumer_main_loop() {
  FragmentC accum = {0};  // Initialize accumulator

  for (int stage = 0; stage < num_stages; ++stage) {
    // Wait for producers to fill TMEM
    barrier_wait(barrier_producers[stage]);

    // Load A, SFA, B, SFB from TMEM to registers
    FragmentA frag_A = load_from_tmem(tmem_ptr_A[stage]);
    FragmentSFA frag_SFA = load_from_tmem(tmem_ptr_SFA[stage]);
    FragmentB frag_B = load_from_tmem(tmem_ptr_B[stage]);
    FragmentSFB frag_SFB = load_from_tmem(tmem_ptr_SFB[stage]);

    // Execute block-scaled MMA
    mma_blockscaled(accum, frag_A, frag_SFA, frag_B, frag_SFB);

    // Signal stage complete (for TMEM reuse)
    barrier_arrive(barrier_consumers[stage]);
  }

  // Pass accumulator to epilogue
  pass_to_epilogue(accum);
}
```

---

## MMA Instruction Details

### Block-Scaled MMA Instruction

**PTX Instruction**: `tcgen05.mma` with block scaling

**Format**:
```
tcgen05.mma.M128N128K256.f32.f4.f4.f32.u8.u8
  dst_accum,      // FP32 accumulator (output)
  src_A,          // FP4 data
  src_B,          // FP4 data
  src_accum,      // FP32 accumulator (input)
  src_SFA,        // FP8 scale factor for A
  src_SFB;        // FP8 scale factor for B
```

**Semantics**:
```
For each element (i, j) in the output tile:
  accum[i,j] += sum_k( (A[i,k] * SFA[i,k/16]) * (B[j,k] * SFB[j,k/16]) )
```

### MMA Tile Shapes

**Supported shapes** (for FP4×FP4):

| MMA Shape (M×N×K) | Threads per MMA | Output Elements per Thread | Notes |
|-------------------|-----------------|----------------------------|-------|
| 128×128×256 | 128 | 4×4×1 FP32 | Most common |
| 128×192×256 | 128 | 4×6×1 FP32 | Rectangular |
| 256×128×256 | 256 | 4×4×1 FP32 | 2SM mode |

**Thread layout** (128×128×256):
```
128 threads arranged as:
  - 4 threads in M dimension
  - 4 threads in N dimension
  - 8 warpgroups (4 threads each)

Each thread computes:
  - 4×4 output elements (FP32)
  - By accumulating 256 K-dimension elements
```

### Scale Factor Application

**Block dimensions**:
- Each scale factor covers 128 rows/cols × 16 K elements
- For 256-K MMA, need 256/16 = 16 scale factors per row/col

**Application timeline**:
```
K = 0..15:    Use SFA[0], SFB[0]
K = 16..31:   Use SFA[1], SFB[1]
K = 32..47:   Use SFA[2], SFB[2]
...
K = 240..255: Use SFA[15], SFB[15]
```

**Hardware implementation**:
- Scale factors loaded into special registers
- MMA instruction automatically applies correct SF for each K slice
- No explicit multiply needed in code

---

## Epilogue Execution

### Epilogue Flow

```
┌─────────────────────────────────────────────────────────────┐
│                   Epilogue Warps                            │
│                                                             │
│  1. Receive accumulator from consumer warps                │
│     - FP32 values in registers                             │
│     - Organized by thread layout                           │
│                                                             │
│  2. Load C matrix (if beta != 0)                          │
│     - From GMEM using TMA or vectorized loads              │
│     - Convert to compute type (FP32)                       │
│                                                             │
│  3. Perform fusion operation                               │
│     temp = alpha * accum + beta * C                        │
│                                                             │
│  4. Apply additional operations (if configured)            │
│     - Activation (ReLU, GELU, etc.)                        │
│     - Bias addition                                         │
│     - Other element-wise ops                               │
│                                                             │
│  5. Quantize output (if narrow precision output)           │
│     - Per-block quantization                               │
│     - Generate scale factors                               │
│     D_quantized, SF_D = quantize_blockwise(temp)           │
│                                                             │
│  6. Store D and SFD to GMEM                               │
│     - D: Quantized values                                  │
│     - SFD: Scale factors                                   │
│     - Use vectorized stores                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Block Quantization Algorithm

**For FP4 output** (with `LinCombBlockScaleFactor`):

```cpp
// Pseudo-code for block quantization
__device__ void quantize_block(
  float* input,          // 128×16 block of FP32 values
  float_e2m1_t* output,  // 128×16 block of FP4 values
  float_ue8m0_t* scale_factor)  // Single SF for block
{
  // 1. Find max absolute value in block
  float max_abs = 0.0f;
  for (int i = 0; i < 128*16; ++i) {
    max_abs = fmaxf(max_abs, fabsf(input[i]));
  }

  // 2. Compute scale factor
  // FP4 max representable value = 6.0
  // Add small epsilon to avoid division by zero
  float sf = (max_abs + 1e-6f) / 6.0f;
  *scale_factor = float_ue8m0_t(sf);

  // 3. Quantize each element
  for (int i = 0; i < 128*16; ++i) {
    float normalized = input[i] / sf;  // Scale to FP4 range
    output[i] = float_e2m1_t(normalized);  // Convert to FP4
  }
}
```

**Optimization details**:
- Max reduction uses warp shuffle intrinsics
- Quantization loop vectorized (process 4-8 elements per thread)
- Scale factors written coalesced to GMEM

### Epilogue Tile Shapes

**Auto-selected based on MMA tile**:

| MMA Tile | Epilogue Tile | Threads | Notes |
|----------|---------------|---------|-------|
| 128×128×256 | 64×64 or 128×64 | 128 | Depends on fusion complexity |
| 256×128×256 | 64×64 or 128×64 | 256 | May use subtiling |

---

## Synchronization Mechanisms

### Barrier Types

**1. Producer-Consumer Barriers**:
```cpp
// Producer signals data ready
barrier_arrive(barrier_stage_ready[stage]);

// Consumer waits for data
barrier_wait(barrier_stage_ready[stage]);
```

**2. Stage Completion Barriers**:
```cpp
// Consumer signals stage consumed (TMEM can be reused)
barrier_arrive(barrier_stage_done[stage]);

// Producer waits before overwriting TMEM
barrier_wait(barrier_stage_done[stage]);
```

**3. Epilogue Handoff Barrier**:
```cpp
// Consumer warps signal accumulator ready
barrier_arrive(barrier_mainloop_done);

// Epilogue warps wait for accumulator
barrier_wait(barrier_mainloop_done);
```

### Barrier Implementation

**Hardware Barriers** (SM100):
- Fast, low-overhead
- Support up to 32 barriers per CTA
- Arrival and wait are separate operations

**Usage in code**:
```cpp
// Initialize barrier for N threads
__shared__ barrier_t barrier;
if (threadIdx.x == 0) {
  init_barrier(&barrier, blockDim.x);
}
__syncthreads();

// Arrive at barrier
barrier_arrive(&barrier);

// Wait for all threads to arrive
barrier_wait(&barrier);
```

### Pipeline Synchronization Pattern

```
Stage 0: Producer │─ Load ─│─ Signal ─│
                           ↓
        Consumer  │─ Wait ─│─ Compute ─│─ Signal ─│
                                               ↓
                                          (TMEM reusable)

Stage 1: Producer │─ (Wait) ─│─ Load ─│─ Signal ─│
                                          ↓
        Consumer  │─────────────│─ Wait ─│─ Compute ─│─ Signal ─│
```

---

## Performance Characteristics

### Theoretical Peak Performance

**For SM100 with 128×128×256 tile**:

```
Operations per MMA: 2 * 128 * 128 * 256 = 8,388,608 FLOPs
Cycle time per MMA: ~256 cycles (pipelined)
FLOPs per cycle: 8,388,608 / 256 ≈ 32,768 FLOPs/cycle

With 128 SMs @ 2.0 GHz:
Peak = 32,768 * 128 * 2.0 = ~8.4 TFLOPS per SM100 GPU
```

**Achievable**: 70-90% of peak with good problem sizes and tuning

### Memory Bandwidth Requirements

**For 2048×2048×2048 GEMM**:

```
A matrix: 2048 * 2048 * 0.5 bytes (FP4) = 2 MB
SFA: (2048/128) * (2048/16) * 1 byte = 2 KB
B matrix: 2048 * 2048 * 0.5 bytes (FP4) = 2 MB
SFB: (2048/128) * (2048/16) * 1 byte = 2 KB
C matrix: 2048 * 2048 * 4 bytes (FP32) = 16 MB
D matrix: 2048 * 2048 * 2 bytes (BF16) = 8 MB

Total: ~28 MB transferred

Operations: 2 * 2048^3 = 17.2 GFLOPs
Arithmetic intensity: 17.2 GFLOPS / 28 MB ≈ 614 FLOPs/byte
```

**Compute-bound**: Arithmetic intensity >> memory bandwidth (good!)

---

## Summary

The Blackwell narrow precision GEMM architecture features:

1. **Warp specialization** for concurrent data movement and computation
2. **Tensor Memory** for efficient data staging
3. **Multi-stage pipelining** to hide latency
4. **Hardware MMA instructions** with built-in block scaling
5. **Flexible epilogue** with quantization and fusion

These features enable high efficiency for narrow precision workloads like LLM inference.

---

**Document Version**: 1.0
**Last Updated**: 2025-11-02
