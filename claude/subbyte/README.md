# CUTLASS Sub-byte Floating Point Type System

A comprehensive guide to understanding the design and architecture of CUTLASS's sub-byte floating point types, specifically `float_e2m1_t` and `nv_float4_t`.

## Table of Contents

### Core Type System (Static Types)

1. [Overview](01-overview.md)
2. [Type Hierarchy](02-type-hierarchy.md)
3. [FpBitRepresentation - The Foundation](03-fpbitrepresentation.md)
4. [float_exmy_base - The CRTP Base](04-float-exmy-base.md)
5. [float_e2m1_t - The 4-bit Float](05-float-e2m1.md)
6. [nv_float4_t - The Scaling Wrapper](06-nv-float4.md)
7. [Complete Call Chain Examples](07-call-chains.md)
8. [Design Patterns and Insights](08-design-patterns.md)

### Type-Erased Types (Runtime Dispatch)

9. [type_erased_dynamic_float4_t - Union-Based Type Erasure](09-type-erased-float4.md)
10. [Type-Erased Call Chains - Runtime Format Selection](10-type-erased-call-chains.md)
11. [Type-Erased Visual Architecture - Comprehensive Diagrams](11-type-erased-diagrams.md)

---

## Quick Reference

### Type Definitions

**Static Type (Compile-Time Format)**:
```cpp
// In your code:
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;

// This expands to a struct containing:
// - DataType: float_e2m1_t (the actual 4-bit float)
// - ScaleFactorType: float_ue4m3_t (8-bit unsigned float for scaling)
```

**Type-Erased (Runtime Format)**:
```cpp
// In your code:
using ElementA = cutlass::type_erased_dynamic_nv_float4_t;
//             = nv_float4_t<type_erased_dynamic_float4_t>

// This expands to a struct containing:
// - DataType: type_erased_dynamic_float4_t (union of formats)
// - ScaleFactorType: float_ue4m3_t (8-bit unsigned float for scaling)

// Runtime format selection:
arguments.hw_info.runtime_format_a = cute::UMMA::MXF8F6F4Format::E2M1;
```

### Type Hierarchies

**Static Type Chain**:
```
nv_float4_t (wrapper/trait struct)
    └── Contains: float_e2m1_t as DataType
        └── Inherits from: float_exmy_base<E2M1, float_e2m1_t>
            └── Contains: FpBitRepresentation<uint8_t, 4, 2, 1, NONE>
                └── Fundamental bit manipulation and conversion logic
```

**Type-Erased Chain**:
```
type_erased_dynamic_nv_float4_t (wrapper/trait struct)
    └── Contains: type_erased_dynamic_float4_t as DataType (union)
        └── Union member: float_e2m1_t e2m1
            └── Inherits from: float_exmy_base<E2M1, float_e2m1_t>
                └── Contains: FpBitRepresentation<uint8_t, 4, 2, 1, NONE>
                    └── Fundamental bit manipulation and conversion logic
```

### File Locations

- [include/cutlass/float_subbyte.h](../../include/cutlass/float_subbyte.h) - Sub-byte float definitions
- [include/cutlass/exmy_base.h](../../include/cutlass/exmy_base.h) - CRTP base class
- [include/cutlass/float8.h](../../include/cutlass/float8.h) - FP8 types (including scale factors)

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Code Level                          │
│  nv_float4_t<float_e2m1_t>                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Trait struct (not a value type)                          │  │
│  │ - DataType = float_e2m1_t                                │  │
│  │ - ScaleFactorType = float_ue4m3_t                        │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Concrete Type Level                           │
│  float_e2m1_t : float_exmy_base<E2M1, float_e2m1_t>          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Storage: uint8_t (holds 4 bits)                          │  │
│  │ Constructors: float, int, double                         │  │
│  │ Conversions: to/from float                               │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   CRTP Base Class Level                         │
│  float_exmy_base<FpEncoding::E2M1, float_e2m1_t>             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Storage: uint8_t storage;                                │  │
│  │ BitRepresentation (computed type)                        │  │
│  │ Arithmetic operators: +, -, *, /                         │  │
│  │ Comparison operators: ==, !=, <, >, <=, >=               │  │
│  │ Conversion methods: convert_from_float, convert_to_float │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Bit Representation & Encoding Level                │
│  FpBitRepresentation<uint8_t, 4, 2, 1, NONE, true>           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Storage = uint8_t                                        │  │
│  │ NUM_BITS = 4                                             │  │
│  │ NUM_EXPONENT_BITS = 2                                    │  │
│  │ NUM_MANTISSA_BITS = 1                                    │  │
│  │ NAN_TYPE = NanInfEncoding::NONE                          │  │
│  │ IS_SIGNED = true                                         │  │
│  │                                                           │  │
│  │ Computed Constants:                                      │  │
│  │ - EXP_BIAS = 1                                           │  │
│  │ - MAX_EXP = 2                                            │  │
│  │ - MIN_EXP = 0                                            │  │
│  │ - MANTISSA_MASK = 0b1                                    │  │
│  │ - EXPONENT_MASK = 0b11                                   │  │
│  │                                                           │  │
│  │ Core Methods:                                            │  │
│  │ - sign_bit(), exponent_bits(), mantissa_bits()           │  │
│  │ - is_denorm(), is_nan(), is_inf()                        │  │
│  │ - convert() - the heart of type conversion               │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

Navigate to [01-overview.md](01-overview.md) to begin the detailed walkthrough.
