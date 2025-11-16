# Type Hierarchy: The Complete Picture

## The Full Inheritance Chain

Let's visualize how `nv_float4_t<float_e2m1_t>` is constructed, layer by layer.

```
╔══════════════════════════════════════════════════════════════════╗
║  LAYER 0: User Declaration                                       ║
║  File: experiments/nvfp4_gemm.cu:95                             ║
╚══════════════════════════════════════════════════════════════════╝
                              │
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
                              │
                              ▼
╔══════════════════════════════════════════════════════════════════╗
║  LAYER 1: nv_float4_t - Trait Struct                           ║
║  File: include/cutlass/float_subbyte.h:506-513                 ║
╚══════════════════════════════════════════════════════════════════╝

template <class F4Type>
struct nv_float4_t {
    static_assert(cute::is_same_v<F4Type, cutlass::float_e2m1_t> || ...,
                  "Only float_e2m1_t can have scale factors for NVFP4");

    using ScaleFactorType = cutlass::float_ue4m3_t;  // 8-bit unsigned FP
    using DataType = F4Type;                          // float_e2m1_t
};

                              │
                              ▼
╔══════════════════════════════════════════════════════════════════╗
║  LAYER 2: float_e2m1_t - Concrete Type                         ║
║  File: include/cutlass/float_subbyte.h:79-100                  ║
╚══════════════════════════════════════════════════════════════════╝

struct float_e2m1_t : public float_exmy_base<
                               cutlass::detail::FpEncoding::E2M1,
                               float_e2m1_t> {

    using Base = float_exmy_base<cutlass::detail::FpEncoding::E2M1,
                                  float_e2m1_t>;

    float_e2m1_t() = default;

    CUTLASS_HOST_DEVICE
    explicit float_e2m1_t(float x) : Base(x) {}

    CUTLASS_HOST_DEVICE
    explicit float_e2m1_t(double x) : Base(float(x)) {}

    CUTLASS_HOST_DEVICE
    explicit float_e2m1_t(int x) : Base(x) {}

    CUTLASS_HOST_DEVICE
    float_e2m1_t(Base x) : Base(x) {}
};

                              │
                              ▼
╔══════════════════════════════════════════════════════════════════╗
║  LAYER 3: float_exmy_base - CRTP Base Class                    ║
║  File: include/cutlass/exmy_base.h:936-1211                    ║
╚══════════════════════════════════════════════════════════════════╝

template <detail::FpEncoding T, class Derived>
struct float_exmy_base {

    static constexpr detail::FpEncoding Encoding = T;  // E2M1

    using BitRepresentation = /* computed type - see Layer 4 */;
    using FP32BitRepresentation = /* E8M23 encoding */;
    using Storage = typename BitRepresentation::Storage;  // uint8_t

    // Data member - the actual storage!
    Storage storage;

    // Constructors
    float_exmy_base() = default;
    float_exmy_base(Storage s) : storage(s) {}
    explicit float_exmy_base(float x) { /* calls convert_from_float */ }
    explicit float_exmy_base(int x) { /* converts via float */ }

    // Conversion methods
    float_exmy_base convert_from_float(float const &flt) const;
    float convert_to_float(float_exmy_base const &x) const;

    // Operators (all implemented via float conversion)
    operator float() const;
    explicit operator int() const;
    friend bool operator==(float_exmy_base const &lhs,
                          float_exmy_base const &rhs);
    friend float_exmy_base operator+(float_exmy_base const &lhs,
                                     float_exmy_base const &rhs);
    // ... and many more

    // Bit manipulation helpers
    Storage &raw();
    Storage raw() const;
    bool signbit() const;
    int exponent_biased() const;
    int exponent() const;
    int mantissa() const;

    // Static helpers
    static bool isfinite(float_exmy_base flt);
    static bool isnan(float_exmy_base flt);
    static bool isinf(float_exmy_base flt);
    static bool isnormal(float_exmy_base flt);
    static float_exmy_base bitcast(Storage x);
};

                              │
                              ▼
╔══════════════════════════════════════════════════════════════════╗
║  LAYER 4: FpBitRepresentation - The Foundation                 ║
║  File: include/cutlass/exmy_base.h:394-850                     ║
╚══════════════════════════════════════════════════════════════════╝

// Instantiated as:
// FpBitRepresentation<uint8_t, 4, 2, 1, NanInfEncoding::NONE, true>

template <
    class StorageType,        // uint8_t
    uint32_t NumBits,         // 4
    uint32_t NumExpBits,      // 2
    uint32_t NumMantissaBits, // 1
    NanInfEncoding Nan,       // NONE
    bool IsSigned             // true
>
struct FpBitRepresentation {

    using Storage = StorageType;  // uint8_t

    // Compile-time computed constants
    static constexpr bool IS_SIGNED = true;
    static constexpr NanInfEncoding NAN_TYPE = NanInfEncoding::NONE;
    static constexpr bool HAS_INF = false;
    static constexpr bool HAS_NAN = false;
    static constexpr bool HAS_DENORM = true;  // NumMantissaBits > 0

    static constexpr uint32_t NUM_BITS = 4;
    static constexpr uint32_t NUM_EXPONENT_BITS = 2;
    static constexpr uint32_t NUM_MANTISSA_BITS = 1;

    // Bit masks
    static constexpr Storage EXPONENT_MASK = 0b11;     // (1 << 2) - 1
    static constexpr Storage MANTISSA_MASK = 0b1;      // (1 << 1) - 1
    static constexpr Storage EXPONENT_SHIFT = 1;       // NUM_MANTISSA_BITS
    static constexpr Storage SIGN_SHIFT = 3;           // NUM_MANTISSA + NUM_EXP

    // Exponent parameters
    static constexpr int EXP_BIAS = 1;           // (1 << (2-1)) - 1
    static constexpr int MAX_EXP = 2;            // (1 << 2) - 1 - bias
    static constexpr int MIN_EXP = 0;            // 1 - bias

    // Floating-point limits
    static constexpr Storage MAX_POS_NORMAL_VAL = 0b0111;    // 6.0
    static constexpr Storage MAX_POS_DENORMAL_VAL = 0b0001;  // 0.5
    static constexpr Storage MIN_POS_NORMAL_VAL = 0b0010;    // 1.0
    static constexpr Storage MIN_POS_DENORMAL_VAL = 0b0001;  // 0.5
    static constexpr Storage MAX_VALUE = 0b0111;             // 6.0
    static constexpr Storage MIN_VALUE = 0b1111;             // -6.0
    static constexpr Storage INF_MASK = 0b0111;              // Saturates to max
    static constexpr Storage NAN_MASK = 0b0111;              // Saturates to max

    // Bit extraction methods
    static Storage sign_bit(Storage flt);
    static Storage set_sign_bit(Storage flt, Storage sign);
    static Storage exponent_bits(Storage flt);
    static int exponent(Storage flt);
    static Storage mantissa_bits(Storage flt);

    // Classification methods
    static bool is_inf(Storage flt);        // Always false for E2M1
    static bool is_nan(Storage flt);        // Always false for E2M1
    static bool is_denorm(Storage flt);     // True when exp bits == 0
    static bool is_canonical_nan(Storage);  // Always false for E2M1

    // Conversion - THE HEART OF THE SYSTEM
    template <class SrcFpBits, class DstFpBits>
    static typename DstFpBits::Storage convert(
        SrcFpBits src_encoding,
        typename SrcFpBits::Storage src_val,
        DstFpBits dst_encoding);

private:
    // Helper methods for conversion
    static Storage make_fp_from_bits(Storage sign, Storage exp,
                                     Storage mantissa);
    static Storage nan_with_sign(Storage sign);
    static Storage inf_with_sign(Storage sign);
    static Storage significand(Storage flt);
    static Storage significand_hidden_bits(Storage significand);
    static Storage round_significand(Storage src, int shift_amount);
};
```

## Understanding the Layers

### Layer 1: `nv_float4_t` - The Trait

```cpp
template <class F4Type>
struct nv_float4_t {
    using ScaleFactorType = cutlass::float_ue4m3_t;
    using DataType = F4Type;
};
```

**Purpose**: Type packaging for NVIDIA's block-wise scaling format.

**Key Insight**: This is **NOT** a value type. You never instantiate it like:
```cpp
nv_float4_t<float_e2m1_t> myValue;  // DON'T do this!
```

Instead, it's used in template metaprogramming:
```cpp
template <typename Element>
struct KernelTraits {
    using DataType = typename Element::DataType;           // float_e2m1_t
    using ScaleType = typename Element::ScaleFactorType;   // float_ue4m3_t
};

KernelTraits<nv_float4_t<float_e2m1_t>> traits;
```

### Layer 2: `float_e2m1_t` - The Concrete Type

```cpp
struct float_e2m1_t : public float_exmy_base<
                               cutlass::detail::FpEncoding::E2M1,
                               float_e2m1_t> {
    using Base = float_exmy_base<...>;

    explicit float_e2m1_t(float x) : Base(x) {}
    // ... other constructors
};
```

**Purpose**: The actual 4-bit floating point type that holds values.

**Key Insight**: Uses **CRTP** (Curiously Recurring Template Pattern):
- Inherits from `float_exmy_base<E2M1, float_e2m1_t>`
- Passes itself as the second template parameter
- This allows the base class to call methods on the derived class

**Example Usage**:
```cpp
float_e2m1_t x(2.5f);  // Construct from float
float y = float(x);     // Convert back to float
float_e2m1_t z = x + x; // Arithmetic
```

### Layer 3: `float_exmy_base` - The CRTP Base

```cpp
template <detail::FpEncoding T, class Derived>
struct float_exmy_base {
    Storage storage;  // uint8_t for E2M1

    explicit float_exmy_base(float x) {
        storage = static_cast<Derived*>(this)->convert_from_float(x).storage;
    }

    operator float() const {
        return static_cast<const Derived*>(this)->convert_to_float(*this);
    }
};
```

**Purpose**: Provides common functionality for all ExMy format types.

**Key Insight**: The CRTP pattern allows:
1. **Static polymorphism**: No virtual functions, zero runtime overhead
2. **Type-specific behavior**: Each derived type can customize conversion
3. **Code reuse**: Arithmetic operators implemented once for all types

**Example of CRTP in action**:
```cpp
// In float_exmy_base constructor:
storage = static_cast<Derived*>(this)->convert_from_float(x).storage;
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          Calls the Derived class's method (float_e2m1_t)
```

### Layer 4: `FpBitRepresentation` - The Foundation

```cpp
template <class StorageType, uint32_t NumBits, uint32_t NumExpBits,
          uint32_t NumMantissaBits, NanInfEncoding Nan, bool IsSigned>
struct FpBitRepresentation {
    // All constants computed at compile time
    static constexpr int EXP_BIAS = exponent_bias_cxx11<...>();
    static constexpr Storage EXPONENT_MASK = (1 << NumExpBits) - 1;

    // The conversion engine
    template <class SrcFpBits, class DstFpBits>
    static typename DstFpBits::Storage convert(...);
};
```

**Purpose**: Generic bit-level representation and conversion for any ExMy format.

**Key Insight**: This is a **zero-state type** - all static members, no instance data:
- All operations are compile-time computed or static methods
- Supports any combination of exponent/mantissa bits
- Implements IEEE-754-style conversion algorithm

## Data Flow Example

Let's trace what happens when you write:

```cpp
float_e2m1_t x(2.5f);
```

### Call Chain:

```
1. float_e2m1_t constructor called
   ├─ float_e2m1_t(float x) : Base(x) {}
   │
   └─▶ 2. Calls float_exmy_base<E2M1, float_e2m1_t> constructor
        ├─ explicit float_exmy_base(float x)
        │
        └─▶ 3. Casts this to Derived* and calls convert_from_float
             ├─ storage = static_cast<Derived*>(this)->convert_from_float(x).storage;
             │
             └─▶ 4. Calls float_exmy_base::convert_from_float (default implementation)
                  ├─ FP32BitRepresentation::Storage fp32_bits =
                  │      FP32BitRepresentation::to_bits(2.5f);
                  │
                  └─▶ 5. Calls BitRepresentation::convert_from
                       ├─ BitRepresentation::convert_from(fp32_bits, FP32BitRepresentation{})
                       │
                       └─▶ 6. Calls the generic convert() method
                            ├─ convert(FP32BitRep, 0x40200000, E2M1BitRep)
                            │  // 0x40200000 = 2.5f in IEEE-754
                            │
                            └─▶ Result: 0b0011 = 1.5 (rounded down from 2.5)
```

### Memory Layout:

```
Before:
float x = 2.5f
┌──────────────────────────────────┐
│ 0 10000000 01000000000000000000000│ (32 bits)
└──────────────────────────────────┘
  S  E(8)    M(23)

After conversion:
float_e2m1_t result
┌───────────┐
│ 0 0 1 1 x │ (4 bits used, stored in uint8_t)
└───────────┘
  S E E M

Where 0011 = 1.5:
- Sign: 0 (positive)
- Exp: 01 (biased exp = 1, real exp = 0)
- Mantissa: 1 (0.5 in fractional)
- Value: (+1) × 1.1 × 2^0 = 1.5

Note: 2.5 can't be exactly represented, rounds to 1.5
```

## Type Computation at Compile Time

The magic happens with template metaprogramming:

```cpp
// Layer 3 computes the BitRepresentation type:
using BitRepresentation =
    #if (CUTLASS_CXX17_OR_LATER)
        decltype(detail::fp_encoding_selector<T>())
    #else
        typename detail::FpEncodingSelector<T>::type
    #endif
;

// For E2M1, this resolves to:
using BitRepresentation = FpBitRepresentation<
    uint8_t,                        // Storage
    4,                              // NumBits
    2,                              // NumExpBits
    1,                              // NumMantissaBits
    NanInfEncoding::NONE,          // Nan
    true                            // IsSigned
>;
```

All of this happens at **compile time** - zero runtime cost!

---

**Next**: [03-fpbitrepresentation.md](03-fpbitrepresentation.md) - Deep dive into the bit-level foundation
