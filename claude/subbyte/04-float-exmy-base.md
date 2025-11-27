# float_exmy_base: The CRTP Base Class

[File: include/cutlass/exmy_base.h:936-1211](../../include/cutlass/exmy_base.h#L936-L1211)

## What Is CRTP?

**CRTP** = Curiously Recurring Template Pattern

```cpp
template <detail::FpEncoding T, class Derived>
struct float_exmy_base {
    // Methods can call into Derived without virtual functions
    storage = static_cast<Derived*>(this)->convert_from_float(x).storage;
};

// Usage:
struct float_e2m1_t : public float_exmy_base<E2M1, float_e2m1_t> {
    //                 Pass itself as template param ──────────┘
};
```

**Why CRTP?**
- **Static polymorphism**: No vtables, no virtual function overhead
- **Customization**: Derived classes can override behavior
- **Zero cost**: All calls resolved at compile time

## The Template

```cpp
template <detail::FpEncoding T, class Derived>
struct float_exmy_base {
    // T = FpEncoding::E2M1 (for float_e2m1_t)
    // Derived = float_e2m1_t
```

### Type Aliases

[Lines 939-956](../../include/cutlass/exmy_base.h#L939-L956)

```cpp
static constexpr detail::FpEncoding Encoding = T;  // E2M1

// Compute BitRepresentation type at compile time
using BitRepresentation =
    #if (CUTLASS_CXX17_OR_LATER)
        decltype(detail::fp_encoding_selector<T>())
    #else
        typename detail::FpEncodingSelector<T>::type
    #endif
;
// For E2M1, this becomes:
// FpBitRepresentation<uint8_t, 4, 2, 1, NanInfEncoding::NONE, true>

// Similarly for FP32:
using FP32BitRepresentation =
    #if (CUTLASS_CXX17_OR_LATER)
        decltype(detail::fp_encoding_selector<FpEncoding::E8M23>())
    #else
        typename detail::FpEncodingSelector<FpEncoding::E8M23>::type
    #endif
;
// FpBitRepresentation<uint32_t, 32, 8, 23, NanInfEncoding::IEEE_754, true>

using Storage = typename BitRepresentation::Storage;  // uint8_t for E2M1
```

**Visual Representation**:

```
float_exmy_base<E2M1, float_e2m1_t>
│
├─ Encoding = E2M1
│
├─ BitRepresentation
│  └─ FpBitRepresentation<uint8_t, 4, 2, 1, NONE, true>
│     ├─ NUM_BITS = 4
│     ├─ NUM_EXPONENT_BITS = 2
│     ├─ NUM_MANTISSA_BITS = 1
│     ├─ EXP_BIAS = 1
│     └─ Storage = uint8_t
│
├─ FP32BitRepresentation
│  └─ FpBitRepresentation<uint32_t, 32, 8, 23, IEEE_754, true>
│     └─ For conversion to/from float
│
└─ Storage = uint8_t
```

## Data Member

[Lines 962-963](../../include/cutlass/exmy_base.h#L962-L963)

```cpp
Storage storage;  // uint8_t for E2M1
```

**This is it!** The only data member. For `float_e2m1_t`:
- `storage` is a `uint8_t` (8 bits)
- Only the lower 4 bits are used
- The upper 4 bits are typically zero

```
Memory layout of float_e2m1_t:
┌────────────────────────┐
│  uint8_t storage       │
│  ┌──────────┬───────┐  │
│  │ unused   │ E2M1  │  │
│  │ (4 bits) │(4 bits)│  │
│  │ 0000     │ SEEMM │  │
│  └──────────┴───────┘  │
└────────────────────────┘

Size: 1 byte (but only 4 bits meaningful)
```

## Constructors

[Lines 966-970](../../include/cutlass/exmy_base.h#L966-L970)

### Default Constructor

```cpp
float_exmy_base() = default;
```

Creates an uninitialized value (storage contains garbage).

### Storage Constructor

```cpp
CUTLASS_HOST_DEVICE
float_exmy_base(Storage s) : storage(s) {}
```

Direct bit-pattern construction:
```cpp
float_exmy_base<E2M1, float_e2m1_t> x(0b0101);  // 3.0
```

### Float Constructor (CRTP Magic!)

[Lines 1023-1026](../../include/cutlass/exmy_base.h#L1023-L1026)

```cpp
CUTLASS_HOST_DEVICE
explicit float_exmy_base<T, Derived>(float x) {
    storage = static_cast<Derived*>(this)->convert_from_float(x).storage;
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
              Calls the derived class's method!
}
```

**The CRTP Pattern in Action**:

```
User writes:
    float_e2m1_t x(2.5f);
                 │
                 ▼
    Calls: float_e2m1_t::float_e2m1_t(float x) : Base(x) {}
                 │
                 ▼
    Calls: float_exmy_base<E2M1, float_e2m1_t>::float_exmy_base(float x)
                 │
                 ▼
    Casts this pointer:
        static_cast<float_e2m1_t*>(this)
                 │
                 ▼
    Calls: float_e2m1_t::convert_from_float(x)
    (or default: float_exmy_base::convert_from_float)
                 │
                 ▼
    Returns: float_exmy_base with storage set
                 │
                 ▼
    Extracts: .storage field
```

### Integer Constructor

[Lines 1029-1032](../../include/cutlass/exmy_base.h#L1029-L1032)

```cpp
CUTLASS_HOST_DEVICE
explicit float_exmy_base<T, Derived>(int x) {
    storage = static_cast<Derived*>(this)->convert_from_float(float(x)).storage;
}
```

Converts through float:
```cpp
float_e2m1_t x(5);  // int → float → E2M1
```

## Conversion Methods

### convert_from_float

[Lines 1004-1009](../../include/cutlass/exmy_base.h#L1004-L1009)

```cpp
CUTLASS_HOST_DEVICE
float_exmy_base convert_from_float(float const &flt) const {
    FP32BitRepresentation::Storage fp32_bits = FP32BitRepresentation::to_bits(flt);
    float_exmy_base float_exmy;
    float_exmy.storage = BitRepresentation::convert_from(fp32_bits, FP32BitRepresentation{});
    return float_exmy;
}
```

**Step-by-step breakdown**:

```
Input: float flt = 2.5f

Step 1: Extract raw bits from float
    FP32BitRepresentation::to_bits(2.5f)
    ↓
    0x40200000
    ┌─┬────────┬──────────────────────┐
    │0│10000000│01000000000000000000000│
    └─┴────────┴──────────────────────┘
     S  E(128)  M(0.25)
    = (+1) × 1.25 × 2^(128-127) = 2.5 ✓

Step 2: Convert using FpBitRepresentation::convert_from
    BitRepresentation::convert_from(0x40200000, FP32BitRepresentation{})
    ↓
    Calls: convert(FP32BitRep, 0x40200000, E2M1BitRep)
    ↓
    [Complex conversion logic - see next section]
    ↓
    Returns: 0b0011
    ┌─┬──┬─┐
    │0│01│1│  = 1.5 (E2M1 can't represent 2.5, rounds down)
    └─┴──┴─┘

Step 3: Store result
    float_exmy.storage = 0b0011
    ↓
    Return float_exmy
```

### convert_to_float

[Lines 1012-1016](../../include/cutlass/exmy_base.h#L1012-L1016)

```cpp
CUTLASS_HOST_DEVICE
float convert_to_float(float_exmy_base<T, Derived> const &x) const {
    FP32BitRepresentation::Storage fp32_bits;
    fp32_bits = BitRepresentation::convert_to(x.storage, FP32BitRepresentation{});
    return detail::copy_bits<FP32BitRepresentation::Storage, float>(fp32_bits);
}
```

**Reverse process**:

```
Input: float_exmy_base x with storage = 0b0011 (1.5)

Step 1: Convert E2M1 → FP32
    BitRepresentation::convert_to(0b0011, FP32BitRepresentation{})
    ↓
    Calls: convert(E2M1BitRep, 0b0011, FP32BitRep)
    ↓
    [Conversion logic]
    ↓
    Returns: 0x3FC00000
    ┌─┬────────┬──────────────────────┐
    │0│01111111│10000000000000000000000│
    └─┴────────┴──────────────────────┘
     S  E(127)  M(0.5)
    = (+1) × 1.5 × 2^(127-127) = 1.5 ✓

Step 2: Copy bits to float
    copy_bits<uint32_t, float>(0x3FC00000)
    ↓
    Uses memcpy (type-safe bit reinterpretation)
    ↓
    Returns: 1.5f
```

### Conversion Operators

[Lines 1040-1049](../../include/cutlass/exmy_base.h#L1040-L1049)

```cpp
// To float
CUTLASS_HOST_DEVICE
operator float() const {
    return static_cast<const Derived*>(this)->convert_to_float(*this);
}

// To int
CUTLASS_HOST_DEVICE
explicit operator int() const {
    return int(static_cast<const Derived*>(this)->convert_to_float(*this));
}
```

**Usage**:
```cpp
float_e2m1_t x(1.5f);
float f = x;           // Implicit: calls operator float()
int i = int(x);        // Explicit: calls operator int() → truncates to 1
```

## Access Methods

[Lines 1052-1085](../../include/cutlass/exmy_base.h#L1052-L1085)

### Raw Access

```cpp
CUTLASS_HOST_DEVICE
Storage &raw() {
    return storage;
}

CUTLASS_HOST_DEVICE
Storage raw() const {
    return storage;
}
```

**Usage**:
```cpp
float_e2m1_t x(2.0f);
uint8_t bits = x.raw();  // Get bit pattern: 0b0100
```

### Component Extraction

```cpp
CUTLASS_HOST_DEVICE
bool signbit() const {
    return bool(BitRepresentation::sign_bit(storage));
}

CUTLASS_HOST_DEVICE
int exponent_biased() const {
    return int(BitRepresentation::exponent_bits(storage));
}

CUTLASS_HOST_DEVICE
int exponent() const {
    return int(BitRepresentation::exponent(storage));
}

CUTLASS_HOST_DEVICE
int mantissa() const {
    return int(BitRepresentation::mantissa_bits(storage));
}
```

**Example**:
```cpp
float_e2m1_t x(3.0f);  // 0b0101 = S:0 E:10 M:1

x.signbit()          → false (0)
x.exponent_biased()  → 2 (0b10)
x.exponent()         → 1 (2 - 1)
x.mantissa()         → 1 (0b1)

Value = (-1)^0 × 1.1 × 2^1 = 1.5 × 2 = 3.0 ✓
```

## Arithmetic Operators

[Lines 1098-1209](../../include/cutlass/exmy_base.h#L1098-L1209)

All arithmetic is **implemented via float conversion**:

### Comparison Operators

```cpp
CUTLASS_HOST_DEVICE
friend bool operator==(float_exmy_base const &lhs, float_exmy_base const &rhs) {
    return float(lhs) == float(rhs);
           ^^^^^^^^^^    ^^^^^^^^^^
           Convert to float, then compare
}

// Similarly: !=, <, <=, >, >=
```

**Example**:
```cpp
float_e2m1_t a(1.5f);  // 0b0011
float_e2m1_t b(2.0f);  // 0b0100

a == b  →  float(a) == float(b)  →  1.5f == 2.0f  →  false
a < b   →  float(a) < float(b)   →  1.5f < 2.0f   →  true
```

### Binary Arithmetic Operators

```cpp
CUTLASS_HOST_DEVICE
friend float_exmy_base operator+(float_exmy_base const &lhs,
                                 float_exmy_base const &rhs) {
    return float_exmy_base(float(lhs) + float(rhs));
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^
                          Convert, add, convert back
}

// Similarly: -, *, /
```

**Example**:
```cpp
float_e2m1_t a(1.0f);  // 0b0010
float_e2m1_t b(0.5f);  // 0b0001

a + b  →  float_exmy_base(float(a) + float(b))
       →  float_exmy_base(1.0f + 0.5f)
       →  float_exmy_base(1.5f)
       →  0b0011
```

**Trace of a + b**:
```
Step 1: Convert lhs to float
    float(a) → convert_to_float(a)
             → 0b0010 → 1.0f

Step 2: Convert rhs to float
    float(b) → convert_to_float(b)
             → 0b0001 → 0.5f

Step 3: FP32 addition
    1.0f + 0.5f = 1.5f

Step 4: Convert result back
    float_exmy_base(1.5f)
    → convert_from_float(1.5f)
    → 0b0011

Step 5: Return
    float_exmy_base with storage = 0b0011
```

### Unary Operators

```cpp
CUTLASS_HOST_DEVICE
friend float_exmy_base operator-(float_exmy_base const &lhs) {
    return float_exmy_base(-float(lhs));
}
```

**Example**:
```cpp
float_e2m1_t a(2.0f);   // 0b0100
float_e2m1_t b = -a;

-a  →  float_exmy_base(-float(a))
    →  float_exmy_base(-2.0f)
    →  0b1100  (sign bit flipped)
```

### Compound Assignment Operators

```cpp
CUTLASS_HOST_DEVICE
friend float_exmy_base &operator+=(float_exmy_base &lhs,
                                   float_exmy_base const &rhs) {
    lhs = float_exmy_base(float(lhs) + float(rhs));
    return lhs;
}

// Similarly: -=, *=, /=
```

### Increment/Decrement

```cpp
// Pre-increment
CUTLASS_HOST_DEVICE
friend float_exmy_base &operator++(float_exmy_base &lhs) {
    float tmp(lhs);
    ++tmp;
    lhs = float_exmy_base(tmp);
    return lhs;
}

// Post-increment
CUTLASS_HOST_DEVICE
friend float_exmy_base operator++(float_exmy_base &lhs, int) {
    float_exmy_base ret(lhs);
    float tmp(lhs);
    tmp++;
    lhs = float_exmy_base(tmp);
    return ret;
}

// Similarly: --, operator--(int)
```

**Example**:
```cpp
float_e2m1_t x(1.0f);  // 0b0010
++x;
// Converts to float: 1.0f
// Increments: 2.0f
// Converts back: 0b0100
```

## Static Utility Methods

[Lines 973-1001](../../include/cutlass/exmy_base.h#L973-L1001)

```cpp
CUTLASS_HOST_DEVICE
static bool isfinite(float_exmy_base flt) {
    return !BitRepresentation::is_inf(flt.storage);
    // For E2M1: always true (no infinity)
}

CUTLASS_HOST_DEVICE
static bool isnan(float_exmy_base flt) {
    return BitRepresentation::is_nan(flt.storage);
    // For E2M1: always false (no NaN)
}

CUTLASS_HOST_DEVICE
static bool isinf(float_exmy_base flt) {
    return BitRepresentation::is_inf(flt.storage);
    // For E2M1: always false (no infinity)
}

CUTLASS_HOST_DEVICE
static bool isnormal(float_exmy_base flt) {
    return !BitRepresentation::is_denorm(flt.storage);
}

CUTLASS_HOST_DEVICE
static float_exmy_base bitcast(Storage x) {
    float_exmy_base f;
    f.storage = x;
    return f;
}
```

**Usage**:
```cpp
float_e2m1_t x(0.5f);  // Denormal: 0b0001

float_e2m1_t::isnormal(x)   → false
float_e2m1_t::isfinite(x)   → true
float_e2m1_t::isnan(x)      → false
float_e2m1_t::isinf(x)      → false

float_e2m1_t y = float_e2m1_t::bitcast(0b0101);  // Direct bit pattern
```

## Summary: The Role of float_exmy_base

`float_exmy_base` provides:

1. **Value semantics**: Storage + constructors + conversions
2. **Arithmetic operations**: All standard operators via float conversion
3. **Type computation**: BitRepresentation and constants
4. **CRTP framework**: Derived classes can customize behavior
5. **Utility methods**: Classification, bit access, etc.

It sits between:
- **Below**: `FpBitRepresentation` (bit-level operations)
- **Above**: Concrete types like `float_e2m1_t` (user-facing API)

It's the **engine** that makes sub-byte floats behave like first-class numeric types.

---

**Next**: [05-float-e2m1.md](05-float-e2m1.md) - The concrete E2M1 type implementation
