# FpBitRepresentation: The Foundation

[File: include/cutlass/exmy_base.h:394-850](../../include/cutlass/exmy_base.h#L394-L850)

## What Is This?

`FpBitRepresentation` is the **engine room** of CUTLASS's floating point system. It's a template that can represent **any** floating point format with custom bit layouts.

## Template Parameters

```cpp
template <
    class StorageType,        // What type holds the bits? (uint8_t, uint32_t, etc.)
    uint32_t NumBits,         // Total bits in the format
    uint32_t NumExpBits,      // Bits for exponent
    uint32_t NumMantissaBits, // Bits for mantissa
    NanInfEncoding Nan,       // How are NaN/Inf encoded?
    bool IsSigned = true      // Is there a sign bit?
>
struct FpBitRepresentation { /* ... */ };
```

### For `float_e2m1_t`, This Instantiates As:

```cpp
FpBitRepresentation<
    uint8_t,                  // Store in 1 byte
    4,                        // Use 4 bits total
    2,                        // 2 bits for exponent
    1,                        // 1 bit for mantissa
    NanInfEncoding::NONE,    // No NaN or Infinity
    true                      // Signed
>
```

## Compile-Time Computed Constants

### The Type Parameters

[Lines 401-406](../../include/cutlass/exmy_base.h#L401-L406)

```cpp
using Storage = StorageType;  // uint8_t

static constexpr bool IS_SIGNED = IsSigned;              // true
static constexpr NanInfEncoding NAN_TYPE = Nan;          // NONE
static constexpr bool HAS_INF = (NAN_TYPE == NanInfEncoding::IEEE_754);  // false
static constexpr bool HAS_NAN = (NAN_TYPE != NanInfEncoding::NONE);      // false
static constexpr bool HAS_DENORM = (NumMantissaBits > 0);                // true
```

**Visual Representation**:
```
E2M1 Format Properties:
┌─────────────────────────┐
│ HAS_INF:     false      │ ❌ No infinity representation
│ HAS_NAN:     false      │ ❌ No NaN representation
│ HAS_DENORM:  true       │ ✓ Supports denormal numbers
│ IS_SIGNED:   true       │ ✓ Has sign bit
└─────────────────────────┘
```

### Bit Layout Constants

[Lines 416-428](../../include/cutlass/exmy_base.h#L416-L428)

```cpp
static constexpr uint32_t NUM_BITS = NumBits;                    // 4
static constexpr uint32_t NUM_EXPONENT_BITS = NumExpBits;        // 2
static constexpr uint32_t NUM_MANTISSA_BITS = NumMantissaBits;   // 1

static constexpr Storage ONE = Storage(1);
static constexpr Storage ZERO = Storage(0);

// Bit masks
static constexpr Storage EXPONENT_MASK = (Storage(1) << Storage(NUM_EXPONENT_BITS)) - ONE;
//                                     = (1 << 2) - 1
//                                     = 0b11

static constexpr Storage MANTISSA_MASK = (Storage(1) << Storage(NUM_MANTISSA_BITS)) - ONE;
//                                     = (1 << 1) - 1
//                                     = 0b1

static constexpr Storage EXPONENT_SHIFT = Storage(NUM_MANTISSA_BITS);  // 1
static constexpr Storage SIGN_SHIFT = Storage(NUM_MANTISSA_BITS + NUM_EXPONENT_BITS);  // 3
```

**Visual Representation**:
```
Bit Position:     3       2   1       0
                ┌───┬─────────┬───┐
                │ S │   E E   │ M │
                └───┴─────────┴───┘
                  ▲       ▲       ▲
                  │       │       └─ Position 0: Mantissa
                  │       └───────── Positions 1-2: Exponent
                  └───────────────── Position 3: Sign

SIGN_SHIFT = 3     (shift by 3 to get to sign bit)
EXPONENT_SHIFT = 1 (shift by 1 to get to exponent bits)

MANTISSA_MASK = 0b0001  (isolate bit 0)
EXPONENT_MASK = 0b0011  (isolate bits 0-1, then shift by EXPONENT_SHIFT)
```

### Exponent Parameters

[Lines 432-434](../../include/cutlass/exmy_base.h#L432-L434)

```cpp
static constexpr int EXP_BIAS = detail::exponent_bias_cxx11<NUM_EXPONENT_BITS, NUM_MANTISSA_BITS>();
//                             = exponent_bias_cxx11<2, 1>()
//                             = (1 << (2-1)) - 1
//                             = (1 << 1) - 1
//                             = 1

static constexpr int MAX_EXP = detail::maximum_exponent_cxx11<NUM_EXPONENT_BITS, NUM_MANTISSA_BITS, NAN_TYPE>();
//                            = maximum_exponent_cxx11<2, 1, NONE>()
//                            = (1 << 2) - 1 - EXP_BIAS
//                            = 4 - 1 - 1
//                            = 2

static constexpr int MIN_EXP = detail::minimum_exponent_cxx11<NUM_EXPONENT_BITS, NUM_MANTISSA_BITS>();
//                            = minimum_exponent_cxx11<2, 1>()
//                            = 1 - EXP_BIAS
//                            = 1 - 1
//                            = 0
```

**What These Mean**:

```
Exponent Encoding for E2M1:
┌──────────────┬──────────────┬─────────────┬──────────────┐
│ Exp Bits (E) │ Biased Value │ Real Value  │ Interpretation│
├──────────────┼──────────────┼─────────────┼──────────────┤
│ 00           │ 0            │ 0 - 1 = -1  │ Denormal     │
│ 01           │ 1            │ 1 - 1 = 0   │ Normal       │
│ 10           │ 2            │ 2 - 1 = 1   │ Normal       │
│ 11           │ 3            │ 3 - 1 = 2   │ Normal (max) │
└──────────────┴──────────────┴─────────────┴──────────────┘

Real Exponent = Biased Exponent - EXP_BIAS

MIN_EXP = 0   (for normal numbers, when E=01)
MAX_EXP = 2   (when E=11)

Special case: E=00 → Denormal (exponent treated as 0, no hidden bit)
```

### Value Limits

[Lines 437-443](../../include/cutlass/exmy_base.h#L437-L443)

```cpp
static constexpr Storage MAX_POS_NORMAL_VAL = detail::max_pos_normal_value_cxx11<...>();
//   For E2M1: exp=11, mantissa=1
//   Bits: 0111 = (+1) × 1.1 × 2^2 = 1.5 × 4 = 6.0

static constexpr Storage MAX_POS_DENORMAL_VAL = detail::max_pos_denormal_value_cxx11<...>();
//   For E2M1: exp=00, mantissa=1
//   Bits: 0001 = (+1) × 0.1 × 2^0 = 0.5 × 1 = 0.5

static constexpr Storage MIN_POS_NORMAL_VAL = detail::min_pos_normal_value_cxx11<...>();
//   For E2M1: exp=01, mantissa=0
//   Bits: 0010 = (+1) × 1.0 × 2^0 = 1.0 × 1 = 1.0

static constexpr Storage MIN_POS_DENORMAL_VAL = detail::min_pos_denormal_value_cxx11<...>();
//   For E2M1: exp=00, mantissa=1
//   Bits: 0001 = (+1) × 0.1 × 2^0 = 0.5 × 1 = 0.5

static constexpr Storage MAX_VALUE = max_value_cxx11<...>();  // 0b0111 = 6.0
static constexpr Storage MIN_VALUE = min_value_cxx11<...>();  // 0b1111 = -6.0

static constexpr Storage INF_MASK = HAS_INF ? ... : MAX_VALUE;  // 0b0111 (no Inf, use max)
static constexpr Storage NAN_MASK = ...;                        // 0b0111 (no NaN, use max)
```

**Visual Number Line**:
```
E2M1 Value Range:
        Denormal │◄──── Normal Numbers ────►│
                 │                          │
    -6  -4  -3  -2  -1.5  -1  -0.5  0  0.5  1  1.5  2  3  4  6
    ●───●───●───●────●────●────●────●───●───●───●───●──●──●──●
    │                                   │       │           │
    MIN_VALUE                      MIN_POS   MIN_POS    MAX_VALUE
    -6.0                          DENORMAL  NORMAL      6.0
                                  0.5       1.0

All 16 representable values (including ±0):
Positive: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
Negative: -0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
```

## Core Methods

### Bit Extraction

[Lines 513-558](../../include/cutlass/exmy_base.h#L513-L558)

#### Extract Sign Bit

```cpp
template<typename T = Storage>
static CUTLASS_CONSTEXPR_IF_CXX17 T sign_bit(T flt) {
    if CUTLASS_CONSTEXPR_IF_CXX17 (!IS_SIGNED) {
        return T(0);
    }
    return static_cast<T>(flt >> T(SIGN_SHIFT));
    //     = flt >> 3
    //     Example: 0b1010 >> 3 = 0b0001 (extracts bit 3)
}
```

**Visual**:
```
Input:  0b1010
        ┌─┬──┬─┐
        │1│01│0│
        └─┴──┴─┘
         S EE M

flt >> 3:
        ┌────────┬─┐
        │00000000│1│  Result: 1 (negative)
        └────────┴─┘
```

#### Extract Exponent Bits

```cpp
static CUTLASS_CONSTEXPR_IF_CXX17 Storage exponent_bits(Storage flt) {
    if CUTLASS_CONSTEXPR_IF_CXX17 (NUM_EXPONENT_BITS == ZERO) {
        return ZERO;
    }
    return (flt >> NUM_MANTISSA_BITS) & EXPONENT_MASK;
    //    = (flt >> 1) & 0b11
}
```

**Visual**:
```
Input:  0b0101
        ┌─┬──┬─┐
        │0│10│1│
        └─┴──┴─┘
         S EE M

Step 1: flt >> 1
        ┌──────┬──┬─┐
        │000000│01│0│
        └──────┴──┴─┘

Step 2: & 0b11
        ┌──────┬──┐
        │000000│10│  Result: 2 (exponent bits)
        └──────┴──┘
```

#### Calculate Real Exponent

```cpp
static CUTLASS_CONSTEXPR_IF_CXX17 int exponent(Storage flt) {
    if CUTLASS_CONSTEXPR_IF_CXX17 (NUM_EXPONENT_BITS == ZERO) {
        return -int(EXP_BIAS);
    }

    if (HAS_DENORM && (exponent_bits(flt) == ZERO)) {
        return 1 - int(EXP_BIAS);  // Denormal: fixed exponent
        //   = 1 - 1 = 0
    }

    return int(flt >> NUM_MANTISSA_BITS & EXPONENT_MASK) - int(EXP_BIAS);
    //   = biased_exp - 1
}
```

**Example**:
```
Value: 0b0101 (representing 3.0)
       ┌─┬──┬─┐
       │0│10│1│
       └─┴──┴─┘
        S EE M

exponent_bits(0b0101) = 0b10 = 2
exponent(0b0101) = 2 - 1 = 1

Value = 1.1₂ × 2^1 = 1.5 × 2 = 3.0 ✓
```

#### Extract Mantissa

```cpp
static CUTLASS_CONSTEXPR_IF_CXX17 Storage mantissa_bits(Storage flt) {
    if CUTLASS_CONSTEXPR_IF_CXX17 (NUM_MANTISSA_BITS == ZERO) {
        return ZERO;
    }
    return (flt & MANTISSA_MASK);
    //    = flt & 0b1
}
```

**Visual**:
```
Input:  0b0101
        ┌─┬──┬─┐
        │0│10│1│
        └─┴──┴─┘
         S EE M

flt & 0b0001:
        ┌──────┬─┐
        │000000│1│  Result: 1 (mantissa bit is set)
        └──────┴─┘
```

### Classification Methods

[Lines 467-511](../../include/cutlass/exmy_base.h#L467-L511)

#### Is Denormal?

```cpp
static CUTLASS_CONSTEXPR_IF_CXX17 bool is_denorm(Storage flt) {
    if CUTLASS_CONSTEXPR_IF_CXX17 (!HAS_DENORM) {
        return false;
    }
    else if (exponent_bits(flt) == ZERO) {
        // Exponent bits are all 0s
        return true;
    }
    return false;
}
```

**Examples**:
```
0b0001 → exponent_bits = 00 → is_denorm = true  (0.5)
0b0010 → exponent_bits = 01 → is_denorm = false (1.0)
0b0101 → exponent_bits = 10 → is_denorm = false (3.0)
```

#### Is NaN / Infinity?

```cpp
static CUTLASS_CONSTEXPR_IF_CXX17 bool is_inf(Storage flt) {
    if CUTLASS_CONSTEXPR_IF_CXX17 (!HAS_INF) {
        return false;  // E2M1 has no infinity
    }
    // ... (for other formats with infinity)
}

static CUTLASS_CONSTEXPR_IF_CXX17 bool is_nan(Storage flt) {
    if CUTLASS_CONSTEXPR_IF_CXX17 (NAN_TYPE == NanInfEncoding::NONE) {
        return false;  // E2M1 has no NaN
    }
    // ... (for other formats with NaN)
}
```

For E2M1, both always return `false` - compile-time optimized out!

### Conversion: The Heart of the System

[Lines 678-845](../../include/cutlass/exmy_base.h#L678-L845)

This is the **most important method** - it converts between any two floating point formats.

```cpp
template <class SrcFpBits, class DstFpBits>
static typename DstFpBits::Storage convert(
    SrcFpBits src_encoding,
    typename SrcFpBits::Storage src_val,
    DstFpBits dst_encoding)
```

We'll dive deep into this in the next chapter, but here's the high-level algorithm:

```
┌─────────────────────────────────────────────────┐
│ Convert Algorithm (Simplified)                  │
├─────────────────────────────────────────────────┤
│                                                  │
│ 1. Extract src sign, exponent, significand      │
│    └─ Use bit masks and shifts                  │
│                                                  │
│ 2. Handle special cases:                        │
│    ├─ NaN → dst NaN                             │
│    ├─ Inf → dst Inf (or saturate)               │
│    └─ Zero → dst Zero                           │
│                                                  │
│ 3. Normalize denormals:                         │
│    └─ Shift mantissa left until hidden bit = 1  │
│        (adjusting exponent accordingly)         │
│                                                  │
│ 4. Check exponent range:                        │
│    ├─ Too large → Inf or MAX_VALUE              │
│    ├─ In range → Convert normally               │
│    └─ Too small → Denormal or underflow to 0    │
│                                                  │
│ 5. Convert mantissa:                            │
│    └─ Round to nearest even                     │
│        (considering guard, round, sticky bits)  │
│                                                  │
│ 6. Assemble result:                             │
│    └─ Pack sign | exp | mantissa                │
│                                                  │
└─────────────────────────────────────────────────┘
```

## Summary: What This Layer Provides

`FpBitRepresentation` is the **generic, reusable foundation** that:

1. **Defines the bit layout** of a floating point format
2. **Computes all constants** at compile time (zero runtime cost)
3. **Provides bit manipulation** primitives (extract sign, exp, mantissa)
4. **Implements classification** (is_nan, is_inf, is_denorm)
5. **Performs conversions** between any two formats

It's designed to be:
- **Generic**: Works for any ExMy format (E8M23, E5M2, E4M3, E2M1, etc.)
- **Efficient**: All operations are constexpr or inline
- **Type-safe**: Strong typing prevents mixing incompatible formats
- **Zero-overhead**: Compiles down to simple bit operations

---

**Next**: [04-float-exmy-base.md](04-float-exmy-base.md) - The CRTP base class that adds value semantics
