# NVFP4 BlockScaled GEMM: Complete Analysis Index

This directory contains comprehensive frame-by-frame analysis of how scale factors are loaded and used in the NVFP4 blockscaled GEMM implementation on SM100.

## Documents

### 1. **NVFP4_SCALE_FACTOR_ANALYSIS_SUMMARY.md** ⭐ START HERE
**Quick reference guide** covering:
- How the `load()` function handles scale factors (TMA load)
- TMA descriptor setup for SFA and SFB
- TMEM layout and fragment organization
- SMEM-to-TMEM copy via UTCCP
- Scale factor integration in MMA instructions
- Special handling for N=192 CTAs
- Design insights and motivations

**Best for**: Quick understanding, implementation overview, debugging

### 2. **NVFP4_BLOCKSCALED_EXECUTION_TRACE.md** 📚 COMPLETE REFERENCE
**Deep execution trace** with detailed frame-by-frame analysis:
- Part 1: Data flow overview (GMEM→SMEM→TMEM→MMA)
- Part 2: Scale factor layout and organization (blocks, SMEM, TMEM)
- Part 3: TMA descriptor setup (configuration and differences from A/B)
- Part 4: Producer warp scale factor TMA load (transactions, barriers)
- Part 5: Consumer warp SMEM-to-TMEM copy (UTCCP setup and execution)
- Part 6: MMA instruction execution with scale factors
- Part 7: Detailed PTX instructions (actual assembly-level operations)
- Appendix: Memory layout examples

**Best for**: Deep understanding, PTX analysis, optimization work

---

## Key File References

### Core Implementation Files
| File | Purpose |
|------|---------|
| `/include/cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp` | Main GEMM collective implementation |
| `/include/cutlass/detail/sm100_blockscaled_layout.hpp` | Block scale layout configuration |
| `/include/cute/arch/mma_sm100_umma.hpp` | UMMA instruction definitions |
| `/include/cute/atom/mma_traits_sm100.hpp` | MMA traits and TMEM management |

### Key Code Sections

#### Load Function (Producer Thread)
- **Location**: Lines 875-922 in sm100_blockscaled_mma_warpspecialized.hpp
- **Purpose**: Issue 4 TMA load operations (A, B, SFA, SFB)
- **See**: NVFP4_SCALE_FACTOR_ANALYSIS_SUMMARY.md § 1

#### TMA Descriptor Setup
- **Location**: Lines 421-428, 540-546 (SFA); 430-437, 548-554 (SFB)
- **Purpose**: Configure descriptors for scale factor TMA loads
- **See**: NVFP4_SCALE_FACTOR_ANALYSIS_SUMMARY.md § 2

#### UTCCP Setup
- **Location**: Lines 832-850 in sm100_blockscaled_mma_warpspecialized.hpp
- **Purpose**: Initialize SMEM-to-TMEM copy operations
- **See**: NVFP4_BLOCKSCALED_EXECUTION_TRACE.md § PART 5

#### MMA with Scale Factors
- **Location**: Lines 1022-1032 (main loop); 1013-1032 (unrolled K loop)
- **Purpose**: Execute MMA instructions with tCtSFA and tCtSFB operands
- **See**: NVFP4_BLOCKSCALED_EXECUTION_TRACE.md § PART 6 & 7

---

## Understanding the Data Flow

### Quick Flow
```
GMEM [SFA/SFB data]
  ↓ (TMA Load)
SMEM [Buffered across stages]
  ↓ (UTCCP Copy)
TMEM [tCtSFA, tCtSFB registers]
  ↓ (Reference in MMA)
UMMA Hardware [Apply scaling]
  ↓
Output Accumulators [Scaled result]
```

### Timing Perspective
```
Load Phase (Producer Thread):
  TMA A → SMEM
  TMA B → SMEM
  TMA SFA → SMEM    (small, ~256B)
  TMA SFB → SMEM    (small, ~256B)
  All 4 arrive → Signal barrier

Compute Phase (Consumer Thread):
  Wait for barrier
  ↓
  UTCCP SFA: SMEM → TMEM (async, fire-and-forget)
  UTCCP SFB: SMEM → TMEM (async)
  ↓
  For each K block:
    MMA with tCtSFA[k], tCtSFB[k]
    (implicitly waits for TMEM via register dependency)
```

---

## Key Concepts Explained

### Scale Factor Blocking
- **Concept**: Group 128 data elements with 4 scale factors
- **Why**: Efficient use of 32-bit TMEM words and memory coalescing
- **Example**: 128-row matrix needs 128/SFVecSize SFs, stored as (32/4)×4 grid
- **See**: NVFP4_BLOCKSCALED_EXECUTION_TRACE.md § PART 2.1

### TMA vs UTCCP
- **TMA (Tensor Memory Accelerator)**: GMEM→SMEM, multi-threaded setup, asynchronous
- **UTCCP (Universal TMEM Copy Pipeline)**: SMEM→TMEM, single-thread operation, implicit wait
- **Why two?**: TMA optimized for large data, UTCCP for small register copy
- **See**: NVFP4_BLOCKSCALED_EXECUTION_TRACE.md § PART 4 & 5

### Barrier Synchronization
- **FULL barrier**: Signals data is loaded (producer arrives after TMA completes)
- **EMPTY barrier**: Signals data is consumed (consumer arrives after processing)
- **Key**: All 4 TMA operations signal same barrier → atomicity
- **See**: NVFP4_BLOCKSCALED_EXECUTION_TRACE.md § PART 4.3

### N=192 CTA Special Case
- **Problem**: Tile N=192, but UTCCP needs 128-alignment
- **Solution**: Pad TileShape_SF to 256, offset TMEM pointers for odd CTAs
- **Offset**: 2 words = 64 scale factors = 64 matrix columns
- **See**: NVFP4_SCALE_FACTOR_ANALYSIS_SUMMARY.md § 6

---

## Questions Answered

### Q: Where are scale factors loaded from?
**A**: Global memory (GMEM) via TMA to SMEM, one descriptor per load phase
- **Location**: Lines 913-914 in load() function
- **See**: NVFP4_SCALE_FACTOR_ANALYSIS_SUMMARY.md § 1

### Q: How are scale factors moved from SMEM to TMEM?
**A**: UTCCP (Universal TMEM Copy Pipeline) asynchronous copies, descriptor-based
- **Location**: Lines 1014-1015 in mma() function
- **See**: NVFP4_BLOCKSCALED_EXECUTION_TRACE.md § PART 5.3

### Q: How does MMA actually use scale factors?
**A**: Via tcgen05.mma instruction with scale factor operands in instruction descriptor
- Applies multiplicative scaling per row (SFA) and column (SFB)
- **See**: NVFP4_BLOCKSCALED_EXECUTION_TRACE.md § PART 6.2 & 7.3

### Q: Why are there separate TMA descriptors for scale factors?
**A**: Data volume mismatch (16KB vs 256B) requires different box sizes and scheduling
- **See**: NVFP4_SCALE_FACTOR_ANALYSIS_SUMMARY.md § 7

### Q: What does `tiled_mma.with()` do?
**A**: Creates MMA variant with scale factor operands and accumulation control
- **See**: NVFP4_SCALE_FACTOR_ANALYSIS_SUMMARY.md § 5

---

## For Implementing Similar Features

### If you need to understand:

1. **How to load auxiliary data alongside main operands**
   - See: NVFP4_BLOCKSCALED_EXECUTION_TRACE.md § PART 3 & 4

2. **How to synchronize auxiliary data with main computation**
   - See: NVFP4_BLOCKSCALED_EXECUTION_TRACE.md § PART 4.3 (barrier sharing)

3. **How to pass extra parameters to MMA instructions**
   - See: NVFP4_SCALE_FACTOR_ANALYSIS_SUMMARY.md § 5 (the `.with()` mechanism)

4. **How to handle layout padding for alignment requirements**
   - See: NVFP4_SCALE_FACTOR_ANALYSIS_SUMMARY.md § 6 (N=192 case)

5. **PTX-level MMA instruction format with extra operands**
   - See: NVFP4_BLOCKSCALED_EXECUTION_TRACE.md § PART 7.3

---

## Performance Characteristics

### Scale Factor Load Overhead
- **GMEM Bandwidth**: Negligible (256 bytes per stage vs 32KB data)
- **TMA Setup**: Same as A/B setup, no additional cost
- **Barrier Overhead**: Shared barrier → no additional synchronization

### UTCCP Copy Performance
- **Latency**: Asynchronous, overlaps with other computation
- **Throughput**: Single instruction per element, minimal
- **Bottleneck**: None (128 bytes << MMA latency)

### MMA Execution
- **Register Operands**: tCtSFA, tCtSFB are register-resident
- **Implicit Wait**: MMA waits on register dependency if UTCCP incomplete
- **Scaling Cost**: Built into UMMA hardware, no additional ALU cycles

---

## Related Documentation

- **CUDA C++ API Reference**: Consult CUTLASS header documentation
- **SM100 Architecture**: See CUDA Compute Architecture whitepaper
- **TMA**: Tensor Memory Accelerator specifications
- **UTCCP**: Universal TMEM Copy Pipeline specifications

---

Created with detailed frame-by-frame analysis of CUTLASS SM100 blockscaled GEMM implementation
