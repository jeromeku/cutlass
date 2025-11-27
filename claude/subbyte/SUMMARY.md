# Quick Reference Summary

## Core Types

### float_e2m1_t - 4-bit Floating Point

```cpp
struct float_e2m1_t : public float_exmy_base<E2M1, float_e2m1_t> {
    // Storage: uint8_t (4 bits used)
    // Format: 1 sign + 2 exponent + 1 mantissa
    // Range: ±[0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    // No Inf/NaN
};
```

**Bit Layout:**
```
Bit: 3   2 1   0
    ┌─┬─────┬─┐
    │S│ E E │M│
    └─┴─────┴─┘
```

**All 16 Values:**
| Bits | Value | Bits | Value |
|------|-------|------|-------|
| 0000 | +0.0  | 1000 | -0.0  |
| 0001 | +0.5  | 1001 | -0.5  |
| 0010 | +1.0  | 1010 | -1.0  |
| 0011 | +1.5  | 1011 | -1.5  |
| 0100 | +2.0  | 1100 | -2.0  |
| 0101 | +3.0  | 1101 | -3.0  |
| 0110 | +4.0  | 1110 | -4.0  |
| 0111 | +6.0  | 1111 | -6.0  |

### nv_float4_t - NVIDIA Block Scaling Wrapper

```cpp
template <class F4Type>
struct nv_float4_t {
    using DataType = F4Type;              // float_e2m1_t
    using ScaleFactorType = float_ue4m3_t; // 8-bit unsigned float
};
```

**NOT a value type!** Used for template metaprogramming:

```cpp
// ❌ Wrong:
nv_float4_t<float_e2m1_t> x;

// ✓ Correct:
using Element = nv_float4_t<float_e2m1_t>;
typename Element::DataType x(2.0f);
typename Element::ScaleFactorType scale(4.0f);
float real = float(x) * float(scale);  // 8.0
```

## Inheritance Hierarchy

```
nv_float4_t<float_e2m1_t>
    │
    └─ Trait struct (type package)
       ├─ DataType = float_e2m1_t
       └─ ScaleFactorType = float_ue4m3_t

float_e2m1_t
    │
    └─ Inherits: float_exmy_base<E2M1, float_e2m1_t>
       │
       ├─ Storage: uint8_t storage
       ├─ Operators: +, -, *, /, ==, !=, <, >, <=, >=
       ├─ Conversions: to/from float, int
       │
       └─ Uses: FpBitRepresentation<uint8_t, 4, 2, 1, NONE, true>
          │
          ├─ EXP_BIAS = 1
          ├─ MAX_EXP = 2
          ├─ MIN_EXP = 0
          ├─ MANTISSA_MASK = 0b1
          ├─ EXPONENT_MASK = 0b11
          │
          └─ Methods: sign_bit(), exponent(), mantissa(),
                      is_denorm(), convert()
```

## Key Characteristics

| Property | Value |
|----------|-------|
| Size | 4 bits (stored in uint8_t) |
| Sign bit | 1 |
| Exponent bits | 2 |
| Mantissa bits | 1 |
| Exponent bias | 1 |
| Has Inf | No |
| Has NaN | No |
| Has Denormals | Yes |
| Range (positive) | [0, 6.0] |
| Smallest positive | 0.5 (denormal) |
| Smallest normal | 1.0 |
| Largest | 6.0 |

## Common Operations

### Construction

```cpp
// From float
float_e2m1_t x(2.5f);  // Rounds to 2.0 or 3.0

// From int
float_e2m1_t y(3);     // 3.0 → 0b0101

// From bits
float_e2m1_t z = float_e2m1_t::bitcast(0b0100);  // 2.0
```

### Conversion

```cpp
float_e2m1_t x(1.5f);
float f = x;           // 1.5f
int i = int(x);        // 1
uint8_t bits = x.raw(); // 0b0011
```

### Arithmetic

```cpp
float_e2m1_t a(1.0f), b(0.5f);
float_e2m1_t sum = a + b;      // 1.5
float_e2m1_t diff = a - b;     // 0.5
float_e2m1_t prod = a * b;     // 0.5
float_e2m1_t quot = a / b;     // 2.0
```

**Note**: All arithmetic converts to float, computes, then converts back.

### Comparison

```cpp
float_e2m1_t a(1.0f), b(2.0f);
bool eq = (a == b);   // false
bool ne = (a != b);   // true
bool lt = (a < b);    // true
bool le = (a <= b);   // true
bool gt = (a > b);    // false
bool ge = (a >= b);   // false
```

### Bit Inspection

```cpp
float_e2m1_t x(3.0f);  // 0b0101

bool sign = x.signbit();        // false (0)
int exp_biased = x.exponent_biased(); // 2 (0b10)
int exp = x.exponent();         // 1 (2 - 1)
int mant = x.mantissa();        // 1 (0b1)

// Value = (-1)^sign × 1.mantissa × 2^exponent
//       = (-1)^0 × 1.1₂ × 2^1
//       = 1 × 1.5 × 2
//       = 3.0 ✓
```

## Conversion Example: 1.5f → E2M1

```
Step 1: IEEE-754 breakdown
    1.5f = 0x3FC00000
         = 0 01111111 10000000000000000000000
         = Sign:0 Exp:127 Mantissa:0.5
         = (+1) × 1.5 × 2^(127-127)

Step 2: Extract components
    Sign: 0
    Exponent: 127 - 127 = 0
    Significand: 1.5 (1.1₂ in binary)

Step 3: Adjust for E2M1
    Biased exponent: 0 + 1 = 1
    Mantissa: 1.5 → 1 bit → 1 (fractional .1 → 1)

Step 4: Assemble
    Bits: (0 << 3) | (1 << 1) | 1
        = 0b0000 | 0b0010 | 0b0001
        = 0b0011

Step 5: Verify
    0b0011 = S:0 E:01 M:1
           = (+1) × 1.1₂ × 2^(1-1)
           = 1.5 ✓
```

## Memory Layout

### Single Value

```cpp
float_e2m1_t x(2.0f);

Memory (1 byte):
┌────────────┐
│ 0000 0100  │  (4 bits used, 4 bits unused)
└────────────┘
     ↑
     0b0100 = 2.0
```

### Block-Scaled Format

```cpp
Block of 32 values:
┌─────────┬──────────────────────────────┐
│ Scale   │ Data (32 × 4-bit)           │
│ (8 bit) │ ┌──┬──┬──┬─────────┬──┬──┐ │
│ 0x90    │ │04│03│06│   ...   │02│05│ │
│ (4.0)   │ └──┴──┴──┴─────────┴──┴──┘ │
└─────────┴──────────────────────────────┘
  1 byte         16 bytes

Real values: [8.0, 6.0, 12.0, ..., 4.0, 10.0]
             (each data × 4.0)

Total: 17 bytes per 32 values
Compression vs FP16: 64 / 17 = 3.76x
```

## Design Patterns Used

1. **CRTP**: Static polymorphism without virtual functions
2. **Template Metaprogramming**: Type computation at compile time
3. **Constexpr**: All constants computed at compile time
4. **Trait Structs**: Type packages for metaprogramming
5. **Zero-State Types**: Static methods, no instance data
6. **Type Erasure**: Unions for runtime dispatch
7. **Operator Overloading**: Via type conversion
8. **Static Assertions**: Compile-time constraints

## File Locations

- **float_e2m1_t**: [include/cutlass/float_subbyte.h:79-100](../include/cutlass/float_subbyte.h#L79-L100)
- **nv_float4_t**: [include/cutlass/float_subbyte.h:506-513](../include/cutlass/float_subbyte.h#L506-L513)
- **float_exmy_base**: [include/cutlass/exmy_base.h:936-1211](../include/cutlass/exmy_base.h#L936-L1211)
- **FpBitRepresentation**: [include/cutlass/exmy_base.h:394-850](../include/cutlass/exmy_base.h#L394-L850)
- **float_ue4m3_t**: [include/cutlass/float8.h:1067+](../include/cutlass/float8.h#L1067)

## Type Selection Guide

**Use Static Types** (`nv_float4_t<float_e2m1_t>`) when:
- Format is known at compile time
- Maximum performance is critical
- Binary size is not a concern

**Use Type-Erased Types** (`type_erased_dynamic_nv_float4_t`) when:
- Format chosen at runtime (user input, config file)
- Need to support multiple formats in single binary
- Binary size reduction is important (~66% savings)
- Minimal performance overhead acceptable (~1%)

## Further Reading

### Core Type System
- [01-overview.md](01-overview.md) - Introduction and bit format
- [02-type-hierarchy.md](02-type-hierarchy.md) - Complete inheritance chain
- [03-fpbitrepresentation.md](03-fpbitrepresentation.md) - Bit-level foundation
- [04-float-exmy-base.md](04-float-exmy-base.md) - CRTP base class
- [05-float-e2m1.md](05-float-e2m1.md) - Concrete E2M1 type
- [06-nv-float4.md](06-nv-float4.md) - NVIDIA scaling wrapper
- [07-call-chains.md](07-call-chains.md) - Step-by-step examples
- [08-design-patterns.md](08-design-patterns.md) - Architecture insights

### Type-Erased System
- [09-type-erased-float4.md](09-type-erased-float4.md) - Union-based type erasure
- [10-type-erased-call-chains.md](10-type-erased-call-chains.md) - Runtime format selection

## Quick Reference: Value Lookup

Given bits, find value:

```cpp
uint8_t bits = /* 0b0000 to 0b1111 */;
float_e2m1_t x = float_e2m1_t::bitcast(bits);
float value = x;  // Get floating point value
```

| bits | value | | bits | value |
|------|-------|-|------|-------|
| 0x0  |  0.0  | | 0x8  | -0.0  |
| 0x1  |  0.5  | | 0x9  | -0.5  |
| 0x2  |  1.0  | | 0xA  | -1.0  |
| 0x3  |  1.5  | | 0xB  | -1.5  |
| 0x4  |  2.0  | | 0xC  | -2.0  |
| 0x5  |  3.0  | | 0xD  | -3.0  |
| 0x6  |  4.0  | | 0xE  | -4.0  |
| 0x7  |  6.0  | | 0xF  | -6.0  |

---

**Complete documentation available in [claude/subbyte/](./)**
