# Type-Erased Float4: Visual Architecture

This document provides comprehensive visual diagrams illustrating the type-erased float4 system architecture.

## Overview: Static vs. Type-Erased Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TWO APPROACHES TO FLOAT4 TYPES                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STATIC (Compile-Time Format)          TYPE-ERASED (Runtime Format)        │
│  ═══════════════════════════════        ══════════════════════════════      │
│                                                                             │
│  nv_float4_t<float_e2m1_t>              type_erased_dynamic_nv_float4_t    │
│                                       = nv_float4_t<union { e2m1, ... }>   │
│                                                                             │
│  ┌───────────────────────────┐        ┌────────────────────────────────┐   │
│  │ Trait Struct              │        │ Trait Struct                   │   │
│  ├───────────────────────────┤        ├────────────────────────────────┤   │
│  │ DataType:                 │        │ DataType:                      │   │
│  │   float_e2m1_t            │        │   type_erased_dynamic_float4_t │   │
│  │   (concrete type)         │        │   (union type)                 │   │
│  │                           │        │                                │   │
│  │ ScaleFactorType:          │        │ ScaleFactorType:               │   │
│  │   float_ue4m3_t           │        │   float_ue4m3_t                │   │
│  └───────────────────────────┘        └────────────────────────────────┘   │
│           │                                     │                          │
│           │                                     │                          │
│           ▼                                     ▼                          │
│  ┌───────────────────────────┐        ┌────────────────────────────────┐   │
│  │ Concrete Type             │        │ Union (Type Erasure)           │   │
│  ├───────────────────────────┤        ├────────────────────────────────┤   │
│  │ struct float_e2m1_t {     │        │ union {                        │   │
│  │   uint8_t storage;        │        │   float_e2m1_t e2m1;           │   │
│  │   // 4 bits used          │        │   // future: float_e1m2_t      │   │
│  │ }                         │        │ }                              │   │
│  │                           │        │                                │   │
│  │ Format: KNOWN             │        │ Format: UNKNOWN                │   │
│  │   E2M1 at compile time ✓  │        │   Set at runtime ⏱            │   │
│  └───────────────────────────┘        └────────────────────────────────┘   │
│                                                  │                          │
│                                                  │                          │
│                                                  ▼                          │
│                                        ┌────────────────────────────────┐   │
│                                        │ Runtime Format Enum            │   │
│                                        ├────────────────────────────────┤   │
│                                        │ MXF8F6F4Format runtime_format  │   │
│                                        │   = MXF8F6F4Format::E2M1       │   │
│                                        │                                │   │
│                                        │ Passed as kernel argument!     │   │
│                                        └────────────────────────────────┘   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                TRADE-OFFS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STATIC                                 TYPE-ERASED                         │
│  • Fastest (no dispatch)                • ~1% slower (dispatch overhead)    │
│  • Smallest kernel                      • Slightly larger kernel            │
│  • Largest binary (N kernels)           • Smallest binary (1 kernel)        │
│  • Format fixed at compile time         • Format chosen at runtime          │
│  • No flexibility                       • Maximum flexibility               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Union Structure: Memory Layout

```
┌────────────────────────────────────────────────────────────────────────────┐
│                  type_erased_dynamic_float4_t UNION                        │
│                           Memory Layout                                    │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  C++ Definition:                                                           │
│  ──────────────                                                            │
│  union type_erased_dynamic_float4_t {                                      │
│      float_e2m1_t e2m1;         // Currently only member                   │
│      // Future:                                                            │
│      // float_e1m2_t e1m2;                                                 │
│      // float_ocp4_t ocp4;                                                 │
│  };                                                                        │
│                                                                            │
│  sizeof(union) = sizeof(largest member) = 1 byte (4 bits used)            │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                         Physical Memory (1 byte)                           │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │                      Single Byte Storage                             │ │
│  │  Bit: 7    6    5    4    3    2    1    0                          │ │
│  │      ┌────┬────┬────┬────┬────┬────┬────┬────┐                      │ │
│  │      │ 0  │ 0  │ 0  │ 0  │ S  │ E  │ E  │ M  │  ← e2m1 interpretation │ │
│  │      └────┴────┴────┴────┴────┴────┴────┴────┘                      │ │
│  │       unused bits        └─────────┬─────────┘                       │ │
│  │                                    │                                  │ │
│  │                                4 bits used                            │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                             ▲                                              │
│                             │                                              │
│                  All union members                                         │
│                  share this memory                                         │
│                             │                                              │
│  ┌──────────────────────────┴───────────────────────────────────────────┐ │
│  │                    Interpretations (Future)                          │ │
│  ├──────────────────────────────────────────────────────────────────────┤ │
│  │                                                                      │ │
│  │  As e2m1 (current):                                                  │ │
│  │  Bit: 3    2    1    0                                               │ │
│  │      ┌────┬────────┬────┐                                            │ │
│  │      │ S  │ E    E │ M  │  1 sign, 2 exp, 1 mantissa                │ │
│  │      └────┴────────┴────┘                                            │ │
│  │                                                                      │ │
│  │  As e1m2 (hypothetical):                                             │ │
│  │  Bit: 3    2    1    0                                               │ │
│  │      ┌────┬────┬─────────┐                                           │ │
│  │      │ S  │ E  │ M     M │  1 sign, 1 exp, 2 mantissa               │ │
│  │      └────┴────┴─────────┘                                           │ │
│  │                                                                      │ │
│  │  As ocp4 (hypothetical):                                             │ │
│  │  Bit: 3    2    1    0                                               │ │
│  │      ┌────┬────────┬────┐                                            │ │
│  │      │ S  │ E    E │ M  │  Different bias/encoding                  │ │
│  │      └────┴────────┴────┘                                            │ │
│  │                                                                      │ │
│  │  ⚠️  SAME BITS, different interpretations!                          │ │
│  │     Runtime enum determines which interpretation to use              │ │
│  │                                                                      │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘

Key Insight:
  The raw bits are IDENTICAL regardless of interpretation.
  The runtime format enum tells hardware HOW to decode those bits.
```

## Runtime Format Enum System

```
┌────────────────────────────────────────────────────────────────────────────┐
│               MXF8F6F4Format: Runtime Format Enumeration                   │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  enum class MXF8F6F4Format : uint8_t {                                     │
│      E4M3    = 0,  // FP8: 1 sign + 4 exp + 3 mantissa                    │
│      E5M2    = 1,  // FP8: 1 sign + 5 exp + 2 mantissa                    │
│      E2M3    = 3,  // FP6: 1 sign + 2 exp + 3 mantissa                    │
│      E3M2    = 4,  // FP6: 1 sign + 3 exp + 2 mantissa                    │
│      E2M1    = 5,  // FP4: 1 sign + 2 exp + 1 mantissa  ← Current         │
│      INVALID = 7   // For type-erased types at compile time               │
│  };                                                                        │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                        Usage Pattern                                       │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌─────────────────────────────┐                                          │
│  │  Host Code (CPU)            │                                          │
│  ├─────────────────────────────┤                                          │
│  │                             │                                          │
│  │  1. User selects format:    │                                          │
│  │     runtime_format = E2M1   │                                          │
│  │                             │                                          │
│  │  2. Pass to kernel args:    │                                          │
│  │     args.runtime_format_a   │                                          │
│  │       = MXF8F6F4Format::E2M1│                                          │
│  │                             │                                          │
│  │  3. Launch kernel:          │                                          │
│  │     gemm.run(args)          │                                          │
│  │                             │                                          │
│  └──────────────┬──────────────┘                                          │
│                 │                                                          │
│                 │ Kernel Launch                                            │
│                 ▼                                                          │
│  ┌─────────────────────────────┐                                          │
│  │  Device Code (GPU)          │                                          │
│  ├─────────────────────────────┤                                          │
│  │                             │                                          │
│  │  1. Kernel receives:        │                                          │
│  │     format = 5 (E2M1)       │                                          │
│  │                             │                                          │
│  │  2. Configure HW:           │                                          │
│  │     switch (format) {       │                                          │
│  │       case 5: /* E2M1 */    │                                          │
│  │         - exp_bias = 1      │                                          │
│  │         - mantissa_bits = 1 │                                          │
│  │         - exp_bits = 2      │                                          │
│  │     }                       │                                          │
│  │                             │                                          │
│  │  3. Issue PTX with format:  │                                          │
│  │     tcgen05.mma.format::5   │                                          │
│  │                             │                                          │
│  └─────────────────────────────┘                                          │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                    Compile-Time vs Runtime                                 │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Static Type:                          Type-Erased:                        │
│  ────────────                          ──────────────                      │
│  to_MXF8F6F4Format<float_e2m1_t>()     to_MXF8F6F4Format<union>()         │
│          ↓                                      ↓                          │
│    Returns: E2M1 (value 5)                Returns: INVALID (value 7)      │
│    Known at compile time ✓                Unknown at compile time ✗       │
│                                           Must use runtime argument ⏱      │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

## Complete Data Flow: Host to Hardware

```
┌────────────────────────────────────────────────────────────────────────────┐
│                     TYPE-ERASED DATA FLOW PIPELINE                         │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  STAGE 1: HOST MEMORY                                                      │
│  ──────────────────────                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  Matrix A: type_erased_dynamic_float4_t[M × K]                       │ │
│  │  ┌────┬────┬────┬────┬────┬─────────┬────┬────┐                     │ │
│  │  │0x45│0x67│0x89│0xAB│0xCD│   ...   │0xEF│0x12│  Raw bytes          │ │
│  │  └────┴────┴────┴────┴────┴─────────┴────┴────┘                     │ │
│  │     └┬┘ └┬┘                                                          │ │
│  │      │   └─ e3,e2 (4 bits each)                                      │ │
│  │      └───── e1,e0 (4 bits each)                                      │ │
│  │                                                                       │ │
│  │  Scale Factors: float_ue4m3_t[blocks]                                │ │
│  │  ┌────┬────┬────┬─────────┬────┐                                     │ │
│  │  │2.0 │4.0 │1.0 │   ...   │8.0 │  8-bit unsigned floats             │ │
│  │  └────┴────┴────┴─────────┴────┘                                     │ │
│  │                                                                       │ │
│  │  Runtime Format: MXF8F6F4Format::E2M1 (value 5)                      │ │
│  │                  ^^^^^^^^^^^^^^^^^^^^^^^^                             │ │
│  │                  Tells how to interpret data!                         │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│           │                                                                │
│           │ cudaMemcpy (no interpretation)                                 │
│           ▼                                                                │
│  STAGE 2: DEVICE GLOBAL MEMORY (GMEM)                                      │
│  ───────────────────────────────────────                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  Same raw bytes as host                                              │ │
│  │  Still type-erased (format in separate parameter)                    │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│           │                                                                │
│           │ TMA Load (cp.async.bulk.tensor.2d)                             │
│           │ - Uses runtime_format to configure data type                   │
│           │ - No interpretation, just memory transfer                      │
│           ▼                                                                │
│  STAGE 3: SHARED MEMORY (SMEM)                                             │
│  ─────────────────────────────                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  Tile data: type_erased_dynamic_float4_unpacksmem_t                  │ │
│  │  Unpacked layout for bank conflict avoidance                         │ │
│  │  Still type-erased!                                                  │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│           │                                                                │
│           │ SMEM → Register Load                                           │
│           ▼                                                                │
│  STAGE 4: REGISTERS                                                        │
│  ──────────────────                                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  Register A: type_erased_dynamic_float4_t[N]                         │ │
│  │  Raw 4-bit values, interpretation deferred                           │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│           │                                                                │
│           │ MMA Instruction                                                │
│           │ ⭐ KEY: Format interpretation happens HERE!                   │
│           ▼                                                                │
│  STAGE 5: TCGEN05 MMA HARDWARE                                             │
│  ──────────────────────────────────                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  MMA Descriptor:                                                     │ │
│  │    format_a = 5 (E2M1)  ← From runtime_format                       │ │
│  │    format_b = 5 (E2M1)                                               │ │
│  │                                                                      │ │
│  │  Hardware Decode Pipeline:                                          │ │
│  │  ┌────────────────────────────────────────────────────────────────┐ │ │
│  │  │ 1. Read raw bits: 0b0101                                       │ │ │
│  │  │                                                                │ │ │
│  │  │ 2. Apply E2M1 decoding (based on format=5):                   │ │ │
│  │  │    Sign: 0 (bit 3)                                             │ │ │
│  │  │    Exp:  10₂ = 2 (bits 2-1)                                    │ │ │
│  │  │    Mant: 1 (bit 0)                                             │ │ │
│  │  │                                                                │ │ │
│  │  │ 3. Compute float value:                                        │ │ │
│  │  │    = (-1)^sign × 1.mantissa × 2^(exp - bias)                  │ │ │
│  │  │    = (+1) × 1.1₂ × 2^(2 - 1)                                   │ │ │
│  │  │    = 1 × 1.5 × 2                                               │ │ │
│  │  │    = 3.0                                                       │ │ │
│  │  │                                                                │ │ │
│  │  │ 4. Apply scale factor: 3.0 × 2.0 = 6.0                        │ │ │
│  │  │                                                                │ │ │
│  │  │ 5. Accumulate: accum += 6.0 × (B value)                       │ │ │
│  │  └────────────────────────────────────────────────────────────────┘ │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│           │                                                                │
│           ▼                                                                │
│  STAGE 6: FP32 ACCUMULATOR                                                 │
│  ───────────────────────────                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  accum[0] = 128.5                                                    │ │
│  │  accum[1] = -64.25                                                   │ │
│  │  ...                                                                 │ │
│  │  High-precision results                                              │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘

Key Observations:
  1. Data remains type-erased through stages 1-4 (Host → GMEM → SMEM → Regs)
  2. Format interpretation happens ONLY at stage 5 (MMA hardware)
  3. Runtime format enum travels alongside data through all stages
  4. Hardware decoding configured by runtime_format value
  5. Zero overhead until actual computation (stage 5)
```

## Kernel Instantiation Comparison

```
┌────────────────────────────────────────────────────────────────────────────┐
│           KERNEL INSTANTIATION: STATIC vs TYPE-ERASED                      │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  SCENARIO: Support 3 FP4 formats (E2M1, E2M3, E1M2)                       │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                    STATIC TYPE APPROACH                                    │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  using GemmE2M1 = CollectiveBuilder<                                       │
│      nv_float4_t<float_e2m1_t>,  ← Concrete type                          │
│      ...                                                                   │
│  >::CollectiveOp;                                                          │
│                                                                            │
│  using GemmE2M3 = CollectiveBuilder<                                       │
│      nv_float4_t<float_e2m3_t>,  ← Different concrete type                │
│      ...                                                                   │
│  >::CollectiveOp;                                                          │
│                                                                            │
│  using GemmE1M2 = CollectiveBuilder<                                       │
│      nv_float4_t<float_e1m2_t>,  ← Yet another concrete type              │
│      ...                                                                   │
│  >::CollectiveOp;                                                          │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │                      RESULT: 3 KERNELS                               │ │
│  ├──────────────────────────────────────────────────────────────────────┤ │
│  │                                                                      │ │
│  │  gemm_kernel<GemmE2M1>  (512 KB)                                     │ │
│  │  gemm_kernel<GemmE2M3>  (512 KB)                                     │ │
│  │  gemm_kernel<GemmE1M2>  (512 KB)                                     │ │
│  │                                                                      │ │
│  │  Total Binary Size: 1536 KB                                         │ │
│  │                                                                      │ │
│  │  At Runtime:                                                         │ │
│  │    if (user_format == E2M1)      launch gemm_kernel<GemmE2M1>       │ │
│  │    else if (user_format == E2M3) launch gemm_kernel<GemmE2M3>       │ │
│  │    else if (user_format == E1M2) launch gemm_kernel<GemmE1M2>       │ │
│  │                                                                      │ │
│  │  ⚠️  Binary contains ALL kernels even if user only needs one!       │ │
│  │                                                                      │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                   TYPE-ERASED APPROACH                                     │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  using GemmTypeErased = CollectiveBuilder<                                 │
│      type_erased_dynamic_nv_float4_t,  ← Union type (all formats)         │
│      ...                                                                   │
│  >::CollectiveOp;                                                          │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │                      RESULT: 1 KERNEL                                │ │
│  ├──────────────────────────────────────────────────────────────────────┤ │
│  │                                                                      │ │
│  │  gemm_kernel<GemmTypeErased>  (520 KB)                               │ │
│  │                                ↑                                     │ │
│  │                  Slightly larger (dispatch overhead)                 │ │
│  │                                                                      │ │
│  │  Total Binary Size: 520 KB  (66% reduction!)                        │ │
│  │                                                                      │ │
│  │  At Runtime:                                                         │ │
│  │    args.runtime_format = user_format;  // E2M1, E2M3, or E1M2       │ │
│  │    launch gemm_kernel<GemmTypeErased>(args);                         │ │
│  │                                                                      │ │
│  │  Inside kernel:                                                      │ │
│  │    switch (args.runtime_format) {                                    │ │
│  │      case E2M1: /* configure for E2M1 */                             │ │
│  │      case E2M3: /* configure for E2M3 */                             │ │
│  │      case E1M2: /* configure for E1M2 */                             │ │
│  │    }                                                                 │ │
│  │                                                                      │ │
│  │  ✓  Binary contains single kernel that handles all formats          │ │
│  │                                                                      │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘

ANALYSIS:
  Static:      3 × 512 KB = 1536 KB, fastest execution
  Type-Erased: 1 × 520 KB =  520 KB, ~1% slower, 66% size reduction
```

## Hardware Instruction Dispatch

```
┌────────────────────────────────────────────────────────────────────────────┐
│              TCGEN05 MMA INSTRUCTION: FORMAT SELECTION                     │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  STATIC TYPE:                                                              │
│  ──────────────                                                            │
│  PTX Generated:                                                            │
│    tcgen05.mma.cta_group::2.kind::ab.tiled                                 │
│           .f16.f32.f32.f32                                                 │
│           .format::e2m1              ← Hardcoded at compile time!          │
│           {accum}, {a}, {b}, {c};                                          │
│                                                                            │
│  No runtime dispatch, direct instruction issue                             │
│  Latency: ~200 cycles (MMA compute time)                                   │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  TYPE-ERASED:                                                              │
│  ──────────────                                                            │
│  PTX Generated:                                                            │
│    // Load format from parameter                                           │
│    ld.param.u8 %format, [runtime_format_a];  // 1 cycle                    │
│                                                                            │
│    // Encode into MMA descriptor                                           │
│    or.b64 %desc, %base_desc, %format;        // 1 cycle                    │
│                                                                            │
│    // Predicated instruction issue                                         │
│    tcgen05.mma.cta_group::2.kind::ab.tiled                                 │
│           .f16.f32.f32.f32                                                 │
│           .format::%desc             ← Runtime format!                     │
│           {accum}, {a}, {b}, {c};                                          │
│                                                                            │
│  Runtime dispatch overhead: ~2 cycles                                      │
│  MMA compute time: ~200 cycles                                             │
│  Relative overhead: 2/200 = 1%                                             │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘

INSIGHT:
  The MMA instruction itself dominates execution time (~200 cycles).
  Format loading and encoding (~2 cycles) is negligible.
  Result: ~1% performance overhead for 66% binary size reduction.
```

## Summary: When to Use Each Approach

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          DECISION MATRIX                                   │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  USE STATIC TYPES IF:                                                      │
│  ═══════════════════════                                                   │
│  ✓ Format known at compile time                                            │
│  ✓ Absolute maximum performance required                                   │
│  ✓ Binary size not a concern                                               │
│  ✓ Supporting single format per binary                                     │
│  ✓ No runtime flexibility needed                                           │
│                                                                            │
│  EXAMPLE: Specialized kernel for production inference with fixed E2M1      │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  USE TYPE-ERASED IF:                                                       │
│  ══════════════════════                                                    │
│  ✓ Format chosen at runtime (user input, config, profiling)               │
│  ✓ Need to support multiple formats                                       │
│  ✓ Binary size is a concern                                               │
│  ✓ 1% performance overhead acceptable                                     │
│  ✓ Maximum flexibility required                                           │
│                                                                            │
│  EXAMPLE: General-purpose library supporting E2M1, E2M3, E1M2, etc.        │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│                            METRICS                                         │
│                                                                            │
│                Static          Type-Erased         Savings                 │
│  Performance:  100%            ~99%                -1%                     │
│  Binary Size:  N × kernel      1 × kernel          66% (N=3)              │
│  Flexibility:  Compile-time    Runtime             +++                    │
│  Complexity:   Simple          Medium              +                      │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

**See Also**:
- [09-type-erased-float4.md](09-type-erased-float4.md) - Complete type erasure documentation
- [10-type-erased-call-chains.md](10-type-erased-call-chains.md) - Detailed execution traces
- [08-design-patterns.md](08-design-patterns.md) - Design pattern analysis
