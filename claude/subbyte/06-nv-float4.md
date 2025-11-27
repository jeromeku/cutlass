# nv_float4_t: The NVIDIA Block Scaling Wrapper

[File: include/cutlass/float_subbyte.h:506-513](../../include/cutlass/float_subbyte.h#L506-L513)

## The Complete Definition

```cpp
template <class F4Type>
struct nv_float4_t
{
    static_assert(cute::is_same_v<F4Type, cutlass::float_e2m1_t>
                  || cute::is_same_v<F4Type, type_erased_dynamic_float4_t>
                  , "Only float_e2m1_t type_erased_dynamic_float4_t can have scale factors for NVFP4");

    using ScaleFactorType = cutlass::float_ue4m3_t;
    using DataType = F4Type;
};
```

## What Is This?

**Critical Understanding**: `nv_float4_t` is **NOT a value type**. It's a **trait struct** - a compile-time type package.

```
❌ WRONG: nv_float4_t<float_e2m1_t> myValue;
✓  RIGHT: typename nv_float4_t<float_e2m1_t>::DataType myValue;
```

## Purpose: NVIDIA's Block Scaling Format

NVIDIA's FP4 format (NVFP4) uses **block-wise scaling**:

```
Conceptual Memory Layout:
┌────────────────────────────────────────────┐
│  Block 0 (32 elements)                     │
│  ┌─────────┬────────────────────────────┐  │
│  │ Scale   │ Data (32 × 4-bit values)   │  │
│  │ (8-bit) │ E2M1 E2M1 E2M1 ... E2M1    │  │
│  └─────────┴────────────────────────────┘  │
├────────────────────────────────────────────┤
│  Block 1 (32 elements)                     │
│  ┌─────────┬────────────────────────────┐  │
│  │ Scale   │ Data (32 × 4-bit values)   │  │
│  │ (8-bit) │ E2M1 E2M1 E2M1 ... E2M1    │  │
│  └─────────┴────────────────────────────┘  │
├────────────────────────────────────────────┤
│  ...                                       │
└────────────────────────────────────────────┘

Real value = data × scale_factor

Example Block:
Scale factor: 2.0
Data values: [0.5, 1.0, 1.5, -0.5, ...]
Real values: [1.0, 2.0, 3.0, -1.0, ...]
```

### Why Block Scaling?

1. **Extended Range**: E2M1 alone has range ±6. With scaling, effective range becomes ±(6 × max_scale).

2. **Better Precision**: Scale factor concentrates precision where values cluster.

3. **Hardware Efficiency**: NVIDIA's tensor cores can compute `(data × scale)` efficiently.

4. **Compression**: Block scaling gives ~4:1 compression vs FP16, with acceptable accuracy.

## Type Members

### DataType

```cpp
using DataType = F4Type;  // float_e2m1_t
```

**Purpose**: The actual 4-bit value type.

**Example**:
```cpp
using Elem = nv_float4_t<float_e2m1_t>;
typename Elem::DataType x(2.0f);  // float_e2m1_t
```

### ScaleFactorType

```cpp
using ScaleFactorType = cutlass::float_ue4m3_t;
```

**Purpose**: The 8-bit unsigned float used for block scaling.

**What is float_ue4m3_t?**

[Defined in include/cutlass/float8.h:1067](../../include/cutlass/float8.h#L1067)

```cpp
struct float_ue4m3_t : public float_exmy_base<cutlass::detail::FpEncoding::UE4M3, float_ue4m3_t> {
    // UE4M3 = Unsigned E4M3
    // - No sign bit
    // - 4 exponent bits
    // - 3 mantissa bits
    // - Total: 8 bits
```

**Bit Layout of float_ue4m3_t**:
```
Bit Position: 7 6 5 4   3 2 1 0
             ┌─────────┬───────┐
             │  E E E E│ M M M │
             └─────────┴───────┘
              Exponent  Mantissa
              (4 bits)  (3 bits)

NO SIGN BIT - Always positive!

Exponent bias: (1 << (4-1)) - 1 = 7
Max exponent: depends on NaN encoding
Range: 0 to ~57344 (for E4M3 format)
```

**Why UE4M3 for scaling?**

1. **Always positive**: Scales are magnitudes, never negative
2. **Wide range**: 8 bits with 4 exp bits covers large dynamic range
3. **Reasonable precision**: 3 mantissa bits sufficient for scale factors
4. **Hardware support**: NVIDIA tensor cores support FP8 operations

### Comparison with MX Format

```
MX (Microscaling) Float:     NV Float:
┌───────────────────┐        ┌───────────────────┐
│ Scale: float_ue8m0│        │ Scale: float_ue4m3│
│ (8 exp, 0 mant)   │        │ (4 exp, 3 mant)   │
│ Data: float_e2m1  │        │ Data: float_e2m1  │
└───────────────────┘        └───────────────────┘

MX: Coarser scales, simpler hardware
NV: Finer scales, more flexible
```

## Template Parameter Constraints

```cpp
static_assert(cute::is_same_v<F4Type, cutlass::float_e2m1_t>
              || cute::is_same_v<F4Type, type_erased_dynamic_float4_t>
              , "Only float_e2m1_t type_erased_dynamic_float4_t can have scale factors for NVFP4");
```

**Allowed F4Type values**:
1. `cutlass::float_e2m1_t` - Concrete E2M1 type
2. `cutlass::type_erased_dynamic_float4_t` - Runtime-selectable format

**Disallowed**:
```cpp
nv_float4_t<float>           // ❌ Compile error
nv_float4_t<float_e2m3_t>    // ❌ Compile error (wrong format)
nv_float4_t<int>             // ❌ Compile error
```

## Usage in CUTLASS

### In Kernel Builders

```cpp
template <typename Element>
struct GemmConfiguration {
    using ElementA = typename Element::DataType;
    using ScaleA = typename Element::ScaleFactorType;
    // ...
};

// Instantiate with:
GemmConfiguration<nv_float4_t<float_e2m1_t>> config;
// config.ElementA = float_e2m1_t
// config.ScaleA = float_ue4m3_t
```

### In Type Dispatch

```cpp
template <typename T>
struct IsMxFloat : cute::false_type {};

template <typename F4>
struct IsMxFloat<mx_float4_t<F4>> : cute::true_type {};

template <typename T>
struct IsNvFloat : cute::false_type {};

template <typename F4>
struct IsNvFloat<nv_float4_t<F4>> : cute::true_type {};

// Usage:
if constexpr (IsNvFloat<ElementType>::value) {
    // Use NVIDIA block scaling
}
```

### In Memory Layout

```cpp
template <typename Element>
struct ScaledMemoryLayout {
    static constexpr int BLOCK_SIZE = 32;

    using DataType = typename Element::DataType;
    using ScaleType = typename Element::ScaleFactorType;

    struct Block {
        ScaleType scale;
        cute::array<DataType, BLOCK_SIZE> data;
    };
};

// Usage:
using Layout = ScaledMemoryLayout<nv_float4_t<float_e2m1_t>>;
Layout::Block block;
block.scale = float_ue4m3_t(2.0f);
block.data[0] = float_e2m1_t(1.5f);  // Real value: 1.5 × 2.0 = 3.0
```

## Related Types

### type_erased_dynamic_float4_t

[Lines 461-467](../../include/cutlass/float_subbyte.h#L461-L467)

```cpp
union type_erased_dynamic_float4_t {
    cutlass::float_e2m1_t e2m1;

    CUTLASS_HOST_DEVICE
    explicit operator cutlass::float_e2m1_t() const {
        return e2m1;
    }
};
```

**Purpose**: Runtime-selectable 4-bit format.

Currently only supports E2M1, but designed to be extensible:
```cpp
// Future extensions might include:
union type_erased_dynamic_float4_t {
    cutlass::float_e2m1_t e2m1;
    cutlass::some_future_fp4_format other_format;
    // ...
};
```

### type_erased_dynamic_nv_float4_t

[Line 515](../../include/cutlass/float_subbyte.h#L515)

```cpp
using type_erased_dynamic_nv_float4_t = nv_float4_t<type_erased_dynamic_float4_t>;
```

**Purpose**: NV float with runtime-selectable underlying format.

**Usage**:
```cpp
template <typename Element>
void kernel(Element::DataType* data, Element::ScaleFactorType* scales) {
    // Can be instantiated with either:
    // - nv_float4_t<float_e2m1_t> (compile-time known)
    // - type_erased_dynamic_nv_float4_t (runtime dispatch)
}
```

### mx_float4_t

[Lines 493-501](../../include/cutlass/float_subbyte.h#L493-L501)

```cpp
template <class F4Type>
struct mx_float4_t
{
    static_assert(cute::is_same_v<F4Type, cutlass::float_e2m1_t>
                  || cute::is_same_v<F4Type, type_erased_dynamic_float4_t>
                  , "Only float_e2m1_t type_erased_dynamic_float4_t can have scale factors for MXFP4");

    using ScaleFactorType = cutlass::float_ue8m0_t;  // ← Different scale type!
    using DataType = F4Type;
};
```

**Comparison**:
```
MX Format (Microsoft/OCP):
┌────────────────────────────┐
│ Scale: float_ue8m0_t       │ Power-of-2 only (8 exp, 0 mantissa)
│ Data:  float_e2m1_t        │
└────────────────────────────┘

NV Format (NVIDIA):
┌────────────────────────────┐
│ Scale: float_ue4m3_t       │ Arbitrary values (4 exp, 3 mantissa)
│ Data:  float_e2m1_t        │
└────────────────────────────┘
```

**float_ue8m0_t** (MX scale factor):
- 8 exponent bits, 0 mantissa bits
- Values are pure powers of 2: 2^exp
- Simpler hardware, coarser granularity

**float_ue4m3_t** (NV scale factor):
- 4 exponent bits, 3 mantissa bits
- Values can be non-power-of-2: mantissa × 2^exp
- More flexible, finer granularity

## Design Pattern: Trait Structs

`nv_float4_t` exemplifies the **trait struct** pattern:

```cpp
// NOT a value type - just a type package
template <typename F4Type>
struct nv_float4_t {
    using ScaleFactorType = ...;
    using DataType = ...;
    // No data members!
    // No constructors!
    // Just type aliases!
};

// Used in template metaprogramming:
template <typename Element>
void process() {
    using Data = typename Element::DataType;      // Extract types
    using Scale = typename Element::ScaleFactorType;

    Data d;      // Instantiate actual values
    Scale s;
}
```

## Example: Complete GEMM Element Type

```cpp
// Your code:
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;

// Expands to:
struct ElementA {
    using DataType = cutlass::float_e2m1_t;
    //   └─> Inherits from: float_exmy_base<E2M1, float_e2m1_t>
    //       └─> Uses: FpBitRepresentation<uint8_t, 4, 2, 1, NONE, true>
    //           └─> Storage: uint8_t (4 bits used)

    using ScaleFactorType = cutlass::float_ue4m3_t;
    //   └─> Inherits from: float_exmy_base<UE4M3, float_ue4m3_t>
    //       └─> Uses: FpBitRepresentation<uint8_t, 8, 4, 3, CANONICAL_ONLY, false>
    //           └─> Storage: uint8_t (8 bits, unsigned)
};

// In GEMM kernel:
// - Loads 32 float_e2m1_t values + 1 float_ue4m3_t scale
// - Computes: result += (A_data × A_scale) × (B_data × B_scale)
// - Accumulates in higher precision (FP16 or FP32)
```

## Memory Layout in Practice

```cpp
// Conceptual structure for a scaled tensor:
template <int ROWS, int COLS, int BLOCK_SIZE = 32>
struct ScaledTensor {
    using Element = nv_float4_t<float_e2m1_t>;
    using Data = typename Element::DataType;
    using Scale = typename Element::ScaleFactorType;

    static constexpr int NUM_BLOCKS = (ROWS * COLS + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cute::array<Scale, NUM_BLOCKS> scales;
    cute::array<Data, ROWS * COLS> data;

    // Actual value at (i, j):
    // = data[i * COLS + j] × scales[(i * COLS + j) / BLOCK_SIZE]
};
```

## Summary: The Role of nv_float4_t

`nv_float4_t` is a **type package** that:

1. **Bundles types** for NVIDIA's block scaling format
2. **Provides type traits** for template metaprogramming
3. **Enforces constraints** via static_assert
4. **Enables dispatch** based on element type
5. **Documents intent** - clearly indicates NVFP4 format

It's **NOT**:
- A value type you instantiate
- A container that holds data
- A wrapper around float_e2m1_t

It's a **compile-time type descriptor** used by CUTLASS's builder pattern and template machinery.

---

**Next**: [07-call-chains.md](07-call-chains.md) - Complete examples of conversions and operations
