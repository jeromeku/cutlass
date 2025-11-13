# Blackwell SM100 GEMM Tutorial Analysis

## Overview

This documentation provides a comprehensive, pedagogical walkthrough of NVIDIA's Blackwell (SM100) GEMM examples from the CUTLASS library. The goal is to help developers understand the specialized call paths and data flow through heavily templated C++ code.

## Documentation Structure

### Core Documents

1. **[00-cpp-fundamentals.md](00-cpp-fundamentals.md)** - C++ Template Fundamentals
   - Template specialization
   - SFINAE and concepts
   - Compile-time computation
   - Type traits and meta-programming

2. **[01-architecture-overview.md](01-architecture-overview.md)** - SM100 Architecture Overview
   - Blackwell architecture features
   - UMMA (Unified Matrix Multiply Accumulate)
   - TMA (Tensor Memory Accelerator)
   - TMEM (Tensor Memory)
   - Cluster execution model

3. **[02-common-patterns.md](02-common-patterns.md)** - Common Code Patterns
   - Shared host setup code
   - Shared device kernel structure
   - Common helper functions
   - Tensor layout and partitioning concepts

### Example-Specific Analysis

4. **[03-multicast-gemm.md](03-multicast-gemm.md)** - Example 03: Multicast TMA
   - Complete line-by-line walkthrough
   - Multicast TMA patterns
   - 1SM MMA instructions
   - Barrier synchronization

5. **[04-2sm-gemm.md](04-2sm-gemm.md)** - Example 04: 2SM Instructions
   - Differences from Example 03
   - 2SM MMA instructions
   - 2SM TMA instructions
   - Peer CTA coordination

### Deep Dives

6. **[05-host-functions.md](05-host-functions.md)** - Host Function Deep Dive
   - `make_tiled_mma` - Creating tiled MMA operations
   - `UMMA::tile_to_mma_shape` - Layout transformation
   - `make_tma_atom` - TMA descriptor creation
   - `tma_atom.get_tma_tensor` - TMA tensor generation

7. **[06-device-functions.md](06-device-functions.md)** - Device Function Deep Dive
   - `tma_partition` - TMA partitioning
   - `create_tma_multicast_mask` - Multicast mask creation
   - `copy` - Asynchronous TMA copy
   - `gemm` - MMA execution
   - `make_tmem_copy` - TMEM copy creation
   - `thr_t2r_copy.partition_S/D` - Thread-level partitioning
   - `tiled_t2r_copy.get_slice` - Slice extraction
   - `tmem_allocator.allocate` - TMEM allocation
   - `cutlass::arch::umma_arrive_multicast_2x1SM` - Barrier arrival

### Minimal Examples

8. **[07-minimal-examples.md](07-minimal-examples.md)** - Isolated API Examples
   - Standalone code snippets for each API
   - How to test individual components
   - Debugging techniques

### Reference

9. **[08-type-reference.md](08-type-reference.md)** - Type and Template Reference
   - Complete list of template parameters
   - Type trait definitions
   - Compile-time constants

10. **[09-source-map.md](09-source-map.md)** - Source Code Map
    - Index of all relevant source files
    - Function/class locations
    - Include dependency graph

## Quick Start Guide

### For Beginners

Start with these documents in order:
1. [00-cpp-fundamentals.md](00-cpp-fundamentals.md) - Understand C++ templates
2. [01-architecture-overview.md](01-architecture-overview.md) - Learn SM100 architecture
3. [02-common-patterns.md](02-common-patterns.md) - Understand common patterns
4. [03-multicast-gemm.md](03-multicast-gemm.md) - Walk through first example

### For Experienced Developers

Jump to:
- [05-host-functions.md](05-host-functions.md) - See host API details
- [06-device-functions.md](06-device-functions.md) - See device API details
- [07-minimal-examples.md](07-minimal-examples.md) - Get working code snippets

### For Debugging

Use:
- [07-minimal-examples.md](07-minimal-examples.md) - Isolate component behavior
- [08-type-reference.md](08-type-reference.md) - Understand type errors
- [09-source-map.md](09-source-map.md) - Navigate source code

## Examples Covered

### Example 03: Multicast TMA GEMM
**File**: [examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu](../../examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu)

**Key Features**:
- 1SM tcgen05.mma instruction (128x256x16)
- Multicast TMA loads
- Cluster shape: 4x4x1
- TMEM accumulator
- Barrier synchronization across cluster

**Operations**: D = α·A·B + β·C
- A: 512×256 (F16, K-major/row-major)
- B: 1024×256 (F16, K-major/column-major)
- C, D: 512×1024 (F32, N-major/row-major)

### Example 04: 2SM GEMM
**File**: [examples/cute/tutorial/blackwell/04_mma_tma_2sm_sm100.cu](../../examples/cute/tutorial/blackwell/04_mma_tma_2sm_sm100.cu)

**Key Features**:
- 2SM tcgen05.mma instruction (256x256x16)
- 2SM Multicast TMA loads
- Cluster shape: 4x4x1 (with 2 peer CTAs per MMA)
- Peer/leader CTA coordination
- Enhanced barrier synchronization

**Operations**: Same as Example 03 (D = α·A·B + β·C)

## Key Concepts

### Template Specialization in CUTLASS

CUTLASS heavily uses C++ templates to achieve:
- **Zero-cost abstractions**: Templates compile to optimal code with no runtime overhead
- **Type safety**: Compile-time checking of dimensions, layouts, and data types
- **Flexibility**: Same codebase works for different data types, sizes, and architectures

### Memory Hierarchy

```
┌─────────────────────────────────────┐
│     Global Memory (GMEM)            │  ← A, B, C, D matrices
├─────────────────────────────────────┤
│          ▼ TMA Load ▼               │
├─────────────────────────────────────┤
│     Shared Memory (SMEM)            │  ← A, B tiles (with swizzling)
├─────────────────────────────────────┤
│          ▼ MMA Read ▼               │
├─────────────────────────────────────┤
│     Tensor Memory (TMEM)            │  ← Accumulator
├─────────────────────────────────────┤
│     ▼ TMEM Load to Register ▼       │
├─────────────────────────────────────┤
│     Registers (RMEM)                │  ← Epilogue computation
├─────────────────────────────────────┤
│          ▼ Store ▼                  │
├─────────────────────────────────────┤
│     Global Memory (GMEM)            │  ← D matrix result
└─────────────────────────────────────┘
```

### Execution Model

```
Grid
├── Cluster (4×4 CTAs)
│   ├── CTA (0,0) ─┬─ Peer CTAs for 2SM
│   ├── CTA (0,1) ─┘
│   ├── CTA (0,2) ─┬─ Peer CTAs for 2SM
│   ├── CTA (0,3) ─┘
│   ├── ...
│   └── CTA (3,3)
│       ├── Warp 0 (32 threads) ← Only warp 0 executes MMA
│       ├── Warp 1 (32 threads)
│       ├── Warp 2 (32 threads)
│       └── Warp 3 (32 threads)
```

### Data Flow

#### Mainloop (per K-tile)
1. **TMA Load**: GMEM → SMEM (async, multicasted)
2. **Barrier Wait**: Wait for TMA completion
3. **MMA**: SMEM → TMEM (accumulator)
4. **Barrier Signal**: MMA completion
5. **Barrier Wait**: Wait for MMA completion

#### Epilogue
1. **TMEM Load**: TMEM → RMEM
2. **GMEM Load**: GMEM (C) → RMEM
3. **AXPBY**: α·Acc + β·C in RMEM
4. **GMEM Store**: RMEM → GMEM (D)

## Terminology

| Term | Description |
|------|-------------|
| **UMMA** | Unified Matrix Multiply Accumulate - SM100 MMA instruction |
| **TMA** | Tensor Memory Accelerator - Hardware unit for async memory transfers |
| **TMEM** | Tensor Memory - On-chip memory for accumulator storage |
| **SMEM** | Shared Memory - Per-CTA on-chip memory |
| **RMEM** | Register Memory - Per-thread registers |
| **GMEM** | Global Memory - Device DRAM |
| **CTA** | Cooperative Thread Array - Thread block |
| **Cluster** | Group of CTAs that can synchronize |
| **Peer CTAs** | CTAs that collaborate on a 2SM instruction |
| **Leader CTA** | CTA that executes the 2SM instruction |
| **Multicast** | Broadcasting TMA loads to multiple CTAs |
| **Swizzle** | Memory layout transformation for bank conflict avoidance |
| **Descriptor** | Hardware-level pointer/layout information for TMA/MMA |
| **Atom** | Smallest logical unit of an operation |
| **Tiler** | Shape specification for partitioning |
| **Partition** | Division of tensor across threads/CTAs |
| **Tile** | Sub-block of a tensor |

## Conventions

### Naming Conventions

- **`m*`**: Tensors in global memory (e.g., `mA`, `mB`)
- **`g*`**: Thread-group level tensors (e.g., `gA`, `gB`)
- **`tC*`**: CTA-level tensors for MMA (e.g., `tCgA`, `tCsA`, `tCrA`)
- **`tD*`**: Thread-level tensors for epilogue (e.g., `tDgC`, `tDrC`)
- **`s*`**: Shared memory tensors (e.g., `sA`, `sB`)
- **`r*`**: Register tensors (e.g., `rAcc`)
- **`t*Acc`**: TMEM accumulator tensors (e.g., `tCtAcc`)

### Layout Notation

CuTe uses a compact notation for layouts:
- **Shape**: `(M, N, K)` - Dimensions
- **Stride**: `(strideM, strideN, strideK)` - Memory strides
- **Example**: `(512, 256):(256, _1)` means 512×256 with stride-256 in M-mode, stride-1 in N-mode

Compile-time constants:
- **`_1`, `_256`, etc.**: Compile-time integer constants
- **`Int<4>{}`**: Compile-time integer constant with value 4
- **`_`**: Wildcard/don't-care in pattern matching

## Navigation Tips

### Finding Definitions

All source code links in this documentation use the format:
```markdown
[filename.ext:line](path/to/filename.ext#Lline)
```

These links are clickable in VSCode when viewing as Markdown.

### Example Link Format
- Function definition: [mma_sm100_umma.hpp:97-120](../../include/cute/arch/mma_sm100_umma.hpp#L97-L120)
- Type definition: [copy_traits_sm100_tma.hpp:45](../../include/cute/atom/copy_traits_sm100_tma.hpp#L45)

### Exploring the Codebase

**Key directories**:
- `include/cute/arch/` - Architecture-specific implementations
- `include/cute/atom/` - Atomic operations and traits
- `include/cute/algorithm/` - High-level algorithms
- `include/cutlass/arch/` - CUTLASS architecture support
- `examples/cute/tutorial/blackwell/` - Tutorial examples

## Prerequisites

### Required Knowledge

- **C++17**: Templates, constexpr, auto, structured bindings
- **CUDA**: Kernels, thread hierarchy, memory hierarchy, synchronization
- **Linear Algebra**: Matrix multiplication, BLAS notation
- **GPU Architecture**: Basic understanding of SM, warp, registers

### Optional but Helpful

- **C++ Template Metaprogramming**: SFINAE, type traits, parameter packs
- **CuTe Library**: Tensor abstractions, layouts, algorithms
- **CUTLASS**: Tiling, partitioning, collective operations

## Building and Running

### Build Requirements

- CUDA Toolkit 12.0+
- CMake 3.18+
- C++17 compatible compiler
- Blackwell GPU (SM100, Compute Capability 10.0)

### Build Commands

```bash
# From CUTLASS root directory
mkdir build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS="100"
make 03_mma_tma_multicast_sm100
make 04_mma_tma_2sm_sm100
```

### Run Examples

```bash
# Example 03
./examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100 [M] [N] [K]

# Example 04
./examples/cute/tutorial/blackwell/04_mma_tma_2sm_sm100 [M] [N] [K]

# Default: M=512, N=1024, K=256
```

## Contributing to This Documentation

This documentation is meant to be a living guide. Improvements welcome:
- Clarifications for confusing sections
- Additional diagrams and visualizations
- More detailed examples
- Corrections for errors

## References

### Official Documentation

- [CUTLASS Documentation](https://github.com/NVIDIA/cutlass)
- [CuTe Documentation](https://github.com/NVIDIA/cutlass/tree/main/media/docs/cute)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Blackwell Architecture Whitepaper](https://www.nvidia.com/en-us/data-center/blackwell-architecture/)

### Related Tutorials

- CUTLASS GEMM examples for SM90
- CuTe tutorial series
- NVIDIA Developer Blog posts on Blackwell

---

**Note**: This documentation focuses on examples 03 and 04. Example 05 (with TMA epilogue) builds on these concepts but is not covered in detail here.
