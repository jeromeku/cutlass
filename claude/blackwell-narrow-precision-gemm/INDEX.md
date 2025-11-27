# Blackwell Narrow Precision GEMM Documentation Index

Complete documentation suite for understanding and using CUTLASS Blackwell narrow precision GEMM kernels.

## 📚 Documentation Structure

This documentation suite provides a comprehensive guide to CUTLASS narrow precision GEMM on Blackwell (SM100) architecture, organized as follows:

### 1. [README.md](./README.md) - Complete Walkthrough ⭐ **START HERE**

**Comprehensive end-to-end walkthrough** of the narrow precision GEMM example.

**What you'll learn**:
- Overview of the example and its features
- Line-by-line code walkthrough with file links
- Type system deep dive (FP4, FP6, scale factors)
- Host-side API execution flow
- CollectiveBuilder template instantiation and dispatch
- Block scaling configuration
- Device-side execution architecture
- Testing methodologies

**Best for**: Understanding the complete picture from user API to device execution

**Time to read**: 45-60 minutes

---

### 2. [quick_reference.md](./quick_reference.md) - Quick Lookup Guide

**Concise reference** for types, APIs, and common patterns.

**What's included**:
- Type reference table (FP4, FP6, scale factors)
- Key classes and function signatures
- Configuration parameters and constraints
- Common code patterns
- Troubleshooting guide
- File location cheat sheet

**Best for**: Quick lookups while coding, troubleshooting errors

**Time to read**: 10-15 minutes (use as reference)

---

### 3. [minimal_examples.md](./minimal_examples.md) - Runnable Examples

**Self-contained, minimal examples** for testing components in isolation.

**Includes**:
- Testing narrow precision type conversions
- Block scale layout computation
- Simple FP4 GEMM end-to-end
- Epilogue fusion configuration

**Best for**: Hands-on experimentation, building intuition, debugging

**Time to complete**: 30-45 minutes (compiling + running)

---

### 4. [device_architecture.md](./device_architecture.md) - Device Execution Deep Dive

**Detailed explanation** of device-side architecture and execution.

**Topics covered**:
- Architecture overview with diagrams
- Warp-specialized design
- Memory hierarchy (GMEM, TMEM, SMEM, RF)
- Pipeline execution model
- MMA instruction details
- Epilogue execution and quantization
- Synchronization mechanisms
- Performance analysis

**Best for**: Performance optimization, understanding hardware mapping

**Time to read**: 30-40 minutes

---

### 5. [EXECUTION_TRACE.md](./EXECUTION_TRACE.md) - Complete Frame-by-Frame Trace ⭐ **DEEP DIVE**

**Frame-by-frame execution trace** from host call to PTX instructions.

**What's included**:
- Host-side: gemm.run() → kernel launch (complete call stack)
- Device-side: Kernel entry → warp specialization
- Producer warp: TMA loads (fully unrolled)
- Consumer warp: MMA operations (down to PTX)
- Epilogue warp: Fusion, quantization, store
- Template instantiations for all types
- Memory traffic analysis
- Synchronization flow diagrams

**Best for**: Understanding the complete execution path, debugging, learning internals

**Time to read**: 60-90 minutes (comprehensive)

---

## 🚀 Getting Started

### For First-Time Users

**Recommended path**:

1. **Read the overview** (5 min)
   - [README.md - Overview section](./README.md#overview)
   - Understand what the example does and key features

2. **Explore types** (10 min)
   - [README.md - Narrow Precision Types](./README.md#narrow-precision-types-deep-dive)
   - [quick_reference.md - Type Reference](./quick_reference.md#type-reference)
   - Understand FP4, scale factors, block-scaled wrappers

3. **Run the example** (15 min)
   - Navigate to [examples/72_blackwell_narrow_precision_gemm/](../../examples/72_blackwell_narrow_precision_gemm/)
   - Build and run the example
   - Observe output

4. **Understand the host API** (20 min)
   - [README.md - Host-Side API Walkthrough](./README.md#host-side-api-walkthrough)
   - Follow execution from `main()` through initialization and kernel launch

5. **Explore device execution** (30 min)
   - [device_architecture.md](./device_architecture.md)
   - Understand warp specialization, TMEM, and MMA instructions

### For Experienced CUTLASS Users

**Fast track**:

1. **Review type system** (5 min)
   - [quick_reference.md - Type Reference](./quick_reference.md#type-reference)

2. **Check builder usage** (5 min)
   - [quick_reference.md - Code Snippets](./quick_reference.md#code-snippets)
   - [README.md - CollectiveBuilder Template Instantiation](./README.md#collectivebuilder-template-instantiation)

3. **Review supported configurations** (5 min)
   - [quick_reference.md - Configuration Parameters](./quick_reference.md#configuration-parameters)

4. **Start coding** (∞ time)
   - Use [quick_reference.md](./quick_reference.md) for lookups
   - Reference [minimal_examples.md](./minimal_examples.md) for patterns

### For Performance Engineers

**Optimization focus**:

1. **Understand device architecture** (30 min)
   - [device_architecture.md](./device_architecture.md)
   - Focus on warp specialization, TMEM usage, pipeline execution

2. **Review performance characteristics** (10 min)
   - [device_architecture.md - Performance Characteristics](./device_architecture.md#performance-characteristics)

3. **Explore tuning parameters** (15 min)
   - [quick_reference.md - Configuration Parameters](./quick_reference.md#configuration-parameters)
   - [quick_reference.md - Performance Optimization](./quick_reference.md#performance-optimization)

4. **Profile and iterate**
   - Use NSight Compute
   - Tune tile shapes, cluster sizes, stage counts

---

## 📖 Learning Paths by Role

### 🎓 Researcher / Student

**Goal**: Understand block-scaled arithmetic and narrow precision GEMM

**Path**:
1. [README.md - Overview](#overview) + [Narrow Precision Types](./README.md#narrow-precision-types-deep-dive)
2. [minimal_examples.md - Example 1](./minimal_examples.md#example-1-testing-narrow-precision-types) (run and modify)
3. [README.md - Block Scaling Configuration](./README.md#block-scaling-configuration)
4. [minimal_examples.md - Example 2](./minimal_examples.md#example-2-block-scale-layout-computation)
5. [device_architecture.md - MMA Instruction Details](./device_architecture.md#mma-instruction-details)

**Outcome**: Solid understanding of block scaling, FP4 arithmetic, and hardware support

---

### 💻 Application Developer

**Goal**: Integrate narrow precision GEMM into application

**Path**:
1. [README.md - Overview](./README.md#overview)
2. [quick_reference.md - Complete GEMM Setup](./quick_reference.md#complete-gemm-setup)
3. [minimal_examples.md - Example 3](./minimal_examples.md#example-3-simple-block-scaled-gemm) (modify for your types)
4. [quick_reference.md - Troubleshooting](./quick_reference.md#troubleshooting)
5. [README.md - Related Unit Tests](./README.md#related-unit-tests) (for validation)

**Outcome**: Working GEMM kernel integrated into your codebase

---

### ⚡ Performance Engineer

**Goal**: Maximize throughput for specific workload

**Path**:
1. [device_architecture.md](./device_architecture.md) (full read)
2. [quick_reference.md - Supported Tile Shapes](./quick_reference.md#supported-tile-shapes-sm100)
3. [device_architecture.md - Pipeline Execution](./device_architecture.md#pipeline-execution)
4. [quick_reference.md - Performance Optimization](./quick_reference.md#performance-optimization)
5. Profile with NSight Compute, iterate on configuration

**Outcome**: Optimal configuration for your problem size and hardware

---

### 🔧 Library Maintainer

**Goal**: Extend or customize CUTLASS for new types/operations

**Path**:
1. [README.md](./README.md) (full read)
2. [device_architecture.md](./device_architecture.md) (full read)
3. [README.md - CollectiveBuilder Template Instantiation](./README.md#collectivebuilder-template-instantiation) (understand dispatch)
4. Study builder implementations:
   - [include/cutlass/gemm/collective/builders/sm100_blockscaled_umma_builder.inl](../../include/cutlass/gemm/collective/builders/sm100_blockscaled_umma_builder.inl)
   - [include/cutlass/epilogue/collective/builders/sm100_builder.inl](../../include/cutlass/epilogue/collective/builders/sm100_builder.inl)
5. [README.md - Related Unit Tests](./README.md#related-unit-tests) (validation strategy)

**Outcome**: Ability to add new types, operations, or architectures

---

## 🔍 Finding Information

### By Topic

| Topic | Primary Document | Section |
|-------|------------------|---------|
| Type definitions (FP4, FP6, etc.) | [quick_reference.md](./quick_reference.md) | Type Reference |
| Block scaling concept | [README.md](./README.md) | Block Scaling Configuration |
| Host API usage | [README.md](./README.md) | Host-Side API Walkthrough |
| CollectiveBuilder | [README.md](./README.md) | CollectiveBuilder Template Instantiation |
| Device architecture | [device_architecture.md](./device_architecture.md) | All sections |
| Warp specialization | [device_architecture.md](./device_architecture.md) | Warp-Specialized Design |
| MMA instructions | [device_architecture.md](./device_architecture.md) | MMA Instruction Details |
| Epilogue fusion | [README.md](./README.md) | Host-Side API Walkthrough |
| Code examples | [minimal_examples.md](./minimal_examples.md) | All examples |
| Troubleshooting | [quick_reference.md](./quick_reference.md) | Troubleshooting |
| Performance tuning | [quick_reference.md](./quick_reference.md) | Performance Optimization |
| File locations | [quick_reference.md](./quick_reference.md) | File Locations Cheat Sheet |

### By Use Case

| Use Case | Where to Look |
|----------|---------------|
| I want to use FP4 GEMM in my code | [minimal_examples.md - Example 3](./minimal_examples.md#example-3-simple-block-scaled-gemm) |
| I'm getting a compilation error | [quick_reference.md - Troubleshooting](./quick_reference.md#troubleshooting) |
| I want to understand how block scaling works | [README.md - Block Scaling Configuration](./README.md#block-scaling-configuration) |
| I need to optimize performance | [device_architecture.md](./device_architecture.md) + [quick_reference.md - Performance](./quick_reference.md#performance-optimization) |
| I want to add a new data type | [README.md - Complete Walkthrough](./README.md) (understand full stack) |
| I need to validate my implementation | [README.md - Related Unit Tests](./README.md#related-unit-tests) |
| I want to test components in isolation | [minimal_examples.md](./minimal_examples.md) |

---

## 🛠️ Tools and Resources

### Building and Running

**Compile example**:
```bash
cd /path/to/cutlass/build
make 72b_blackwell_nvfp4_nvfp4_gemm
./examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm --m=2048 --n=2048 --k=2048
```

**Run unit tests**:
```bash
make test_unit_gemm_device_sm100_blockscaled_tensorop
./test/unit/gemm/device/test_unit_gemm_device_sm100_blockscaled_tensorop_nvf4_nvf4_bf16_bf16
```

### Profiling

**NSight Compute**:
```bash
ncu --set full --target-processes all \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed \
    ./72b_blackwell_nvfp4_nvfp4_gemm --m=2048 --n=2048 --k=2048 --iterations=10
```

**Key metrics to watch**:
- SM throughput (target: >80%)
- DRAM throughput (should be low - compute bound)
- Warp stall reasons
- Occupancy

### Additional Resources

**CUTLASS Documentation**:
- Main repo: https://github.com/NVIDIA/cutlass
- Documentation: https://github.com/NVIDIA/cutlass/tree/main/media/docs

**NVIDIA Documentation**:
- PTX ISA: https://docs.nvidia.com/cuda/parallel-thread-execution/
- CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/

**Related Papers**:
- "CUTLASS: Fast Linear Algebra in CUDA C++" (2020)
- "Microscaling Data Formats for Deep Learning" (2023)
- Blackwell Architecture Whitepaper (NVIDIA)

---

## 📝 Document Maintenance

**Version**: 1.0
**Last Updated**: 2025-11-02
**CUTLASS Branch**: blackwell-examples

### Change Log

- **2025-11-02**: Initial documentation suite created
  - Complete walkthrough (README.md)
  - Quick reference guide (quick_reference.md)
  - Minimal examples (minimal_examples.md)
  - Device architecture (device_architecture.md)
  - This index (INDEX.md)

### Contributing

Found an error or have a suggestion? Please file an issue with:
- Document name and section
- Description of issue or suggested improvement
- Your CUTLASS version and GPU architecture

---

## 🎯 Quick Start Checklist

Before diving in, ensure you have:

- [ ] CUDA Toolkit 12.8 or later installed
- [ ] Blackwell GPU (SM100, SM101, or SM103) available
- [ ] CUTLASS repository cloned
- [ ] Basic understanding of CUDA programming
- [ ] Familiarity with templated C++

**Ready?** Start with [README.md](./README.md)!

---

## 📊 Documentation Coverage Map

```
Blackwell Narrow Precision GEMM
│
├─ Concepts
│  ├─ Narrow precision types ─────────── README.md § Narrow Precision Types
│  ├─ Block scaling ──────────────────── README.md § Block Scaling Config
│  └─ Warp specialization ────────────── device_architecture.md § Warp-Spec
│
├─ Host-Side
│  ├─ API usage ──────────────────────── README.md § Host-Side API
│  ├─ CollectiveBuilder ───────────────── README.md § CollectiveBuilder
│  ├─ Configuration params ────────────── quick_reference.md § Config
│  └─ Code patterns ──────────────────── minimal_examples.md
│
├─ Device-Side
│  ├─ Architecture ───────────────────── device_architecture.md § Overview
│  ├─ Memory hierarchy ────────────────── device_architecture.md § Memory
│  ├─ Pipeline execution ──────────────── device_architecture.md § Pipeline
│  ├─ MMA instructions ────────────────── device_architecture.md § MMA
│  └─ Epilogue ───────────────────────── device_architecture.md § Epilogue
│
├─ Practical Guides
│  ├─ Quick reference ─────────────────── quick_reference.md
│  ├─ Minimal examples ────────────────── minimal_examples.md
│  ├─ Troubleshooting ─────────────────── quick_reference.md § Troubleshooting
│  └─ Unit tests ─────────────────────── README.md § Related Unit Tests
│
└─ This Index ────────────────────────── INDEX.md
```

---

**Happy coding! 🚀**
