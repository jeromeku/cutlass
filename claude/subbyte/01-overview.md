# Overview: Sub-byte Floating Point Types in CUTLASS

## What Are We Looking At?

The code snippet from your experiment:

```cpp
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
```

This declares a **4-bit floating point type** with NVIDIA's block-wise scaling format. Let's unpack what this means.

## The Problem These Types Solve

### Why Sub-byte Floats?

1. **Memory Bandwidth**: Modern GPUs are often memory-bound. Using 4-bit instead of 16-bit floats reduces memory traffic by 4x.

2. **Tensor Core Efficiency**: NVIDIA's latest GPUs (Blackwell/Hopper) have specialized tensor cores that can compute with 4-bit and 6-bit floats.

3. **Model Compression**: AI models can often tolerate reduced precision, especially for inference.

### The Challenge

Standard C++ doesn't have sub-byte primitive types. You can't just declare a `float4_t` like you can `float` or `double`. The smallest addressable unit is typically a byte (8 bits).

### CUTLASS's Solution

CUTLASS implements sub-byte floats using:

1. **Bit-packed storage**: Store the value in a `uint8_t` but only use 4 bits
2. **Custom type system**: Define new types with proper semantics
3. **Conversion logic**: Implement IEEE-754-like conversion to/from standard floats
4. **Type traits**: Provide metadata for template metaprogramming

## The Two Main Components

### 1. `float_e2m1_t` - The Base Type

```
┌─────────────────────────────┐
│  float_e2m1_t (4 bits)     │
│  ┌─┬──┬─┐                  │
│  │S│EE│M│                  │
│  └─┴──┴─┘                  │
│   │  │  └─ 1 mantissa bit  │
│   │  └──── 2 exponent bits │
│   └─────── 1 sign bit      │
│                             │
│  Range: ±[0, 0.5, 1, 1.5,  │
│           2, 3, 4, 6]       │
│  No Inf/NaN                 │
└─────────────────────────────┘
```

- **E2M1** = 2 Exponent bits, 1 Mantissa bit
- Total: 4 bits (1 sign + 2 exp + 1 mantissa)
- Exponent bias: 1
- Supports denormals
- No infinity or NaN representations

### 2. `nv_float4_t` - The Scaling Wrapper

```
┌────────────────────────────────────────┐
│  nv_float4_t<float_e2m1_t>            │
│  ┌──────────────────────────────────┐  │
│  │ Not a value type!                │  │
│  │ It's a "trait struct"            │  │
│  │                                  │  │
│  │ using DataType = float_e2m1_t;   │  │
│  │ using ScaleFactorType =          │  │
│  │       float_ue4m3_t;             │  │
│  └──────────────────────────────────┘  │
└────────────────────────────────────────┘
```

This is a **template struct** that packages together:
- The data type (`float_e2m1_t`)
- The scale factor type (`float_ue4m3_t` - an 8-bit unsigned float)

It's used in NVIDIA's **block-wise scaling** format where:
- A group of values share a common scale factor
- Each value is stored as `real_value = data × scale_factor`

## The Format: E2M1

Let's understand the bit layout:

```
Bit position:  3   2 1   0
              ┌─┬─────┬─┐
              │S│ E E │M│
              └─┴─────┴─┘
               │   │   └─ Mantissa (1 bit)
               │   └───── Exponent (2 bits)
               └───────── Sign (1 bit)
```

### Value Interpretation

For a **normal** value:
```
value = (-1)^S × 1.M × 2^(E - bias)
      = (-1)^S × 1.M × 2^(E - 1)
```

For a **denormal** value (E = 0):
```
value = (-1)^S × 0.M × 2^(1 - bias)
      = (-1)^S × 0.M × 2^0
      = (-1)^S × 0.M
```

### All Possible Values

| Bits | S | E  | M | Type   | Value    | Decimal |
|------|---|----|---|--------|----------|---------|
| 0000 | 0 | 00 | 0 | Zero   | +0.0     | 0       |
| 0001 | 0 | 00 | 1 | Denorm | 0.1 × 2^0| 0.5     |
| 0010 | 0 | 01 | 0 | Normal | 1.0 × 2^0| 1.0     |
| 0011 | 0 | 01 | 1 | Normal | 1.1 × 2^0| 1.5     |
| 0100 | 0 | 10 | 0 | Normal | 1.0 × 2^1| 2.0     |
| 0101 | 0 | 10 | 1 | Normal | 1.1 × 2^1| 3.0     |
| 0110 | 0 | 11 | 0 | Normal | 1.0 × 2^2| 4.0     |
| 0111 | 0 | 11 | 1 | Normal | 1.1 × 2^2| 6.0     |
| 1000 | 1 | 00 | 0 | Zero   | -0.0     | -0      |
| 1001 | 1 | 00 | 1 | Denorm | -0.1×2^0 | -0.5    |
| 1010 | 1 | 01 | 0 | Normal | -1.0×2^0 | -1.0    |
| 1011 | 1 | 01 | 1 | Normal | -1.1×2^0 | -1.5    |
| 1100 | 1 | 10 | 0 | Normal | -1.0×2^1 | -2.0    |
| 1101 | 1 | 10 | 1 | Normal | -1.1×2^1 | -3.0    |
| 1110 | 1 | 11 | 0 | Normal | -1.0×2^2 | -4.0    |
| 1111 | 1 | 11 | 1 | Normal | -1.1×2^2 | -6.0    |

**Note**: In binary, 1.1 means 1 + 0.5 = 1.5, and 0.1 means 0.5.

### Key Properties

- **Range**: ±6.0 (no values beyond this)
- **No Infinity**: Overflow saturates to ±6.0
- **No NaN**: No representation for Not-a-Number
- **Satfinite behavior**: Operations saturate rather than producing Inf/NaN
- **16 distinct values**: Including ±0

## Why This Matters

When you write:

```cpp
float_e2m1_t x = float_e2m1_t(2.5f);
```

The following happens:
1. 2.5f is a 32-bit float (E8M23 format)
2. It must be converted to E2M1 format (4 bits)
3. The conversion:
   - Extracts sign, exponent, mantissa from float32
   - Rounds the mantissa to 1 bit
   - Adjusts the exponent for the new bias
   - Handles overflow (satfinite) and underflow (denormal)
   - Packs into 4 bits

We'll trace this **exact conversion process** in the following chapters.

## Design Philosophy

CUTLASS's approach uses several advanced C++ patterns:

1. **CRTP (Curiously Recurring Template Pattern)**: For static polymorphism
2. **Template Metaprogramming**: To compute types and constants at compile-time
3. **Type Erasure**: To unify different formats
4. **Trait Structs**: To package type information

All of this creates a **zero-overhead abstraction** - the generated code is as efficient as hand-written bit manipulation.

---

**Next**: [02-type-hierarchy.md](02-type-hierarchy.md) - Understanding the complete inheritance chain
