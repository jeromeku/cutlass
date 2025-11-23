# Design Patterns and Architectural Insights

This chapter explores the **design patterns** and **architectural decisions** that make CUTLASS's sub-byte float system elegant and efficient.

## Pattern 1: CRTP (Curiously Recurring Template Pattern)

### The Pattern

```cpp
template <typename Encoding, typename Derived>
struct Base {
    void method() {
        static_cast<Derived*>(this)->derived_method();
    }
};

struct Concrete : public Base<E2M1, Concrete> {
    void derived_method() { /* implementation */ }
};
```

### In CUTLASS

```cpp
template <detail::FpEncoding T, class Derived>
struct float_exmy_base { /* ... */ };

struct float_e2m1_t : public float_exmy_base<E2M1, float_e2m1_t> {
    // Inherits everything, passes itself as template param
};
```

### Why CRTP?

```
┌─────────────────────────────────────────────────┐
│ Traditional Polymorphism (Virtual Functions)   │
├─────────────────────────────────────────────────┤
│ struct Base {                                   │
│     virtual void convert() = 0;                 │
│ };                                              │
│                                                  │
│ struct Derived : Base {                         │
│     void convert() override { /* ... */ }       │
│ };                                              │
│                                                  │
│ Runtime cost:                                   │
│ - vtable pointer in every object (+8 bytes)     │
│ - Indirect function call (slower)               │
│ - Prevents inlining                             │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ CRTP (Static Polymorphism)                      │
├─────────────────────────────────────────────────┤
│ template <typename Derived>                     │
│ struct Base {                                   │
│     void method() {                             │
│         static_cast<Derived*>(this)->method();  │
│     }                                            │
│ };                                              │
│                                                  │
│ struct Concrete : Base<Concrete> { /* ... */ }; │
│                                                  │
│ Compile-time resolution:                        │
│ - No vtable (+0 bytes)                          │
│ - Direct function call                          │
│ - Fully inlinable                               │
│ - Zero runtime overhead                         │
└─────────────────────────────────────────────────┘
```

### Benefits in CUTLASS

1. **Code reuse**: Common functionality in base class
2. **Customization**: Derived classes can override methods
3. **Type safety**: Each type is distinct at compile time
4. **Performance**: No runtime overhead whatsoever

### Example: Conversion Method Selection

```cpp
// In float_exmy_base constructor:
storage = static_cast<Derived*>(this)->convert_from_float(x).storage;
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          Resolved at compile time!

// For float_e2m1_t:
storage = static_cast<float_e2m1_t*>(this)->convert_from_float(x).storage;
                                           │
                                           └─ Uses default implementation
                                              (or custom if overridden)
```

## Pattern 2: Type Computation via Template Metaprogramming

### Compile-Time Type Selection

```cpp
#if (CUTLASS_CXX17_OR_LATER)
    // C++17: Use auto return type and constexpr if
    template<FpEncoding FpExMyCode>
    constexpr auto fp_encoding_selector() {
        if constexpr (FpExMyCode == FpEncoding::E2M1) {
            return FpBitRepresentation<uint8_t, 4, 2, 1, NanInfEncoding::NONE>{};
        }
        else if constexpr (FpExMyCode == FpEncoding::E8M23) {
            return FpBitRepresentation<uint32_t, 32, 8, 23, NanInfEncoding::IEEE_754>{};
        }
        // ... more formats
    }
#else
    // C++11: Use template specialization
    template <FpEncoding Code> struct FpEncodingSelector;

    template <>
    struct FpEncodingSelector<FpEncoding::E2M1> {
        using type = FpBitRepresentation<uint8_t, 4, 2, 1, NanInfEncoding::NONE>;
    };

    template <>
    struct FpEncodingSelector<FpEncoding::E8M23> {
        using type = FpBitRepresentation<uint32_t, 32, 8, 23, NanInfEncoding::IEEE_754>;
    };
#endif
```

### Usage

```cpp
// Computed at compile time, zero runtime cost!
using BitRep = decltype(fp_encoding_selector<E2M1>());
// or
using BitRep = typename FpEncodingSelector<E2M1>::type;

// Expands to:
// FpBitRepresentation<uint8_t, 4, 2, 1, NanInfEncoding::NONE>
```

### Benefits

1. **No runtime dispatch**: All type selection at compile time
2. **Type safety**: Impossible to mix incompatible formats
3. **Extensibility**: Add new formats by adding specializations
4. **Optimization**: Compiler can optimize knowing exact types

## Pattern 3: Constexpr Computation

### Compile-Time Constants

```cpp
template <uint32_t NumExpBits, uint32_t NumMantissaBits>
constexpr int exponent_bias_cxx11() {
    return (NumExpBits == 0) ?
        -1 * static_cast<int>(NumMantissaBits) :
        static_cast<int>((1 << (NumExpBits - 1))) - 1;
}

// Usage:
static constexpr int EXP_BIAS = exponent_bias_cxx11<2, 1>();
// Computed at compile time: EXP_BIAS = 1
```

### Benefits

```
Traditional Approach:
┌────────────────────────────┐
│ int compute_bias(int exp,  │
│                  int mant) │
│ {                          │
│     if (exp == 0)          │
│         return -mant;      │
│     return (1 << (exp-1))  │
│            - 1;            │
│ }                          │
│                            │
│ Runtime cost:              │
│ - Function call            │
│ - Branch misprediction     │
│ - Register allocation      │
└────────────────────────────┘

Constexpr Approach:
┌────────────────────────────┐
│ constexpr int bias =       │
│     exponent_bias<2, 1>(); │
│                            │
│ Compiled to:               │
│ mov eax, 1                 │
│                            │
│ Runtime cost: ZERO         │
│ - Computed at compile time │
│ - Inlined as constant      │
│ - No branches, no calls    │
└────────────────────────────┘
```

### All Constants Are Constexpr

```cpp
static constexpr Storage EXPONENT_MASK = (Storage(1) << NumExpBits) - 1;
static constexpr Storage MANTISSA_MASK = (Storage(1) << NumMantissaBits) - 1;
static constexpr int EXP_BIAS = exponent_bias_cxx11<...>();
static constexpr int MAX_EXP = maximum_exponent_cxx11<...>();
static constexpr Storage MAX_VALUE = max_value_cxx11<...>();
// All computed at compile time!
```

## Pattern 4: Trait Structs (Type Packages)

### The Pattern

```cpp
// Not a value type - just bundles type information
template <typename T>
struct TypeTraits {
    using DataType = /* ... */;
    using ScaleType = /* ... */;
    static constexpr int BLOCK_SIZE = /* ... */;
    // No data members, no constructors
};
```

### In CUTLASS

```cpp
template <class F4Type>
struct nv_float4_t {
    using ScaleFactorType = float_ue4m3_t;
    using DataType = F4Type;
};
```

### Usage Pattern

```cpp
// Extract types for use:
template <typename Element>
struct KernelConfig {
    using Data = typename Element::DataType;
    using Scale = typename Element::ScaleFactorType;

    void process(Data* data, Scale* scales) {
        Data d = data[0];
        Scale s = scales[0];
        float result = float(d) * float(s);
    }
};

// Instantiate:
KernelConfig<nv_float4_t<float_e2m1_t>> config;
```

### Benefits

1. **Compile-time polymorphism**: Different instantiations for different types
2. **Type bundling**: Related types travel together
3. **Zero overhead**: No runtime representation
4. **Self-documenting**: Clear intent (NVFP4 vs MXFP4)

## Pattern 5: Static Methods on Zero-State Types

### The Pattern

```cpp
struct Utility {
    // No data members - purely static
    static int compute(int x) { return x * 2; }
    static bool check(int x) { return x > 0; }
};

// Usage (no instantiation needed):
int result = Utility::compute(5);
```

### In CUTLASS: FpBitRepresentation

```cpp
template </* params */>
struct FpBitRepresentation {
    // NO DATA MEMBERS

    static Storage sign_bit(Storage flt) { /* ... */ }
    static Storage exponent_bits(Storage flt) { /* ... */ }
    static bool is_nan(Storage flt) { /* ... */ }
    // All static methods
};

// Usage:
using E2M1Bits = FpBitRepresentation<uint8_t, 4, 2, 1, NONE, true>;
bool is_denorm = E2M1Bits::is_denorm(0b0001);
```

### Why This Works

```cpp
// Traditional approach (with state):
struct FpConverter {
    int exp_bits;
    int mant_bits;

    FpConverter(int e, int m) : exp_bits(e), mant_bits(m) {}

    uint8_t convert(float x) { /* uses exp_bits, mant_bits */ }
};

FpConverter conv(2, 1);  // Create object
uint8_t result = conv.convert(1.5f);

Problems:
- Need to store exp_bits, mant_bits (8-16 bytes per object)
- Need constructor, need to pass object around
- Multiple instantiations waste memory
```

```cpp
// CUTLASS approach (zero-state):
template <int ExpBits, int MantBits>
struct FpConverter {
    // NO DATA MEMBERS

    static uint8_t convert(float x) {
        // ExpBits and MantBits available as constants
        return /* conversion using ExpBits, MantBits */;
    }
};

// Usage:
uint8_t result = FpConverter<2, 1>::convert(1.5f);

Benefits:
- Zero memory overhead
- No construction needed
- All information in type
- Can't accidentally modify state
```

## Pattern 6: Type Erasure with Unions

### The Pattern

```cpp
union TypeErased {
    ConcreteType1 type1;
    ConcreteType2 type2;

    operator ConcreteType1() const { return type1; }
    operator ConcreteType2() const { return type2; }
};
```

### In CUTLASS

```cpp
union type_erased_dynamic_float4_t {
    cutlass::float_e2m1_t e2m1;
    // Future: other 4-bit formats

    CUTLASS_HOST_DEVICE
    explicit operator cutlass::float_e2m1_t() const {
        return e2m1;
    }
};
```

### Use Case: Runtime Format Selection

```cpp
enum class FP4Format { E2M1, /* future formats */ };

template <typename Element>
void kernel(FP4Format format, void* data) {
    if (format == FP4Format::E2M1) {
        // Use Element::DataType = float_e2m1_t
        auto* typed_data = static_cast<typename Element::DataType*>(data);
        process_e2m1(typed_data);
    }
    // ... handle other formats
}

// Can be instantiated with:
// - nv_float4_t<float_e2m1_t> (compile-time format)
// - type_erased_dynamic_nv_float4_t (runtime dispatch)
```

## Pattern 7: Operator Overloading via Type Conversion

### The Pattern

```cpp
struct CustomType {
    operator float() const { /* convert to float */ }

    // All operators implemented via float
    friend CustomType operator+(CustomType a, CustomType b) {
        return CustomType(float(a) + float(b));
    }
};
```

### In CUTLASS

```cpp
// float_exmy_base provides conversion:
operator float() const {
    return convert_to_float(*this);
}

// All arithmetic via float conversion:
friend float_exmy_base operator+(
    float_exmy_base const &lhs,
    float_exmy_base const &rhs) {
    return float_exmy_base(float(lhs) + float(rhs));
}
```

### Why This Works

```
Option 1: Native Arithmetic (Complex)
┌────────────────────────────────────┐
│ float_e2m1_t operator+(            │
│     float_e2m1_t a,                │
│     float_e2m1_t b) {              │
│     // Need to:                    │
│     // - Extract signs, exps, mants│
│     // - Align exponents           │
│     // - Add mantissas             │
│     // - Normalize result          │
│     // - Handle overflow/underflow │
│     // - Round result              │
│     // ... hundreds of lines       │
│ }                                  │
│                                    │
│ Issues:                            │
│ - Complex logic                   │
│ - Hard to verify correctness      │
│ - Error-prone                     │
│ - Must implement for EVERY format │
└────────────────────────────────────┘

Option 2: Via Float Conversion (Simple)
┌────────────────────────────────────┐
│ float_e2m1_t operator+(            │
│     float_e2m1_t a,                │
│     float_e2m1_t b) {              │
│     return float_e2m1_t(           │
│         float(a) + float(b)        │
│     );                             │
│ }                                  │
│                                    │
│ Benefits:                          │
│ - Simple, obvious correctness     │
│ - Reuses float hardware           │
│ - Automatic for all formats       │
│ - Only pay conversion cost        │
│                                    │
│ Tradeoff:                          │
│ - Conversion overhead             │
│ - But: GPU float ops are fast!    │
└────────────────────────────────────┘
```

### Performance Consideration

For CUTLASS's use case (tensor operations):
- Most time spent in matrix multiply accumulate
- E2M1 values loaded, converted to FP32, used in FMA
- Conversion cost is unavoidable anyway
- Native E2M1 arithmetic would be slower than FP32!

## Pattern 8: Compile-Time Assertions

### Strategic Use of static_assert

```cpp
template <class F4Type>
struct nv_float4_t {
    static_assert(
        cute::is_same_v<F4Type, cutlass::float_e2m1_t> ||
        cute::is_same_v<F4Type, type_erased_dynamic_float4_t>,
        "Only float_e2m1_t type_erased_dynamic_float4_t can have scale factors for NVFP4"
    );
    // Compile error if instantiated with wrong type
};
```

### Benefits

1. **Catch errors early**: At compile time, not runtime
2. **Clear error messages**: Explains what's wrong
3. **Zero runtime cost**: No checks in generated code
4. **Self-documenting**: Constraints are explicit

### Example Error

```cpp
// Wrong usage:
using Bad = nv_float4_t<float>;

// Compiler error:
// error: static assertion failed: Only float_e2m1_t type_erased_dynamic_float4_t
//        can have scale factors for NVFP4
```

## Architectural Insights

### Layered Architecture

```
┌─────────────────────────────────────────────────┐
│ Application Layer                               │
│ - User code: using ElementA = nv_float4_t<...> │
└─────────────────────────────────────────────────┘
                     │
┌─────────────────────────────────────────────────┐
│ Trait Layer                                     │
│ - nv_float4_t: Type packages                   │
│ - sizeof_bits: Size metadata                   │
└─────────────────────────────────────────────────┘
                     │
┌─────────────────────────────────────────────────┐
│ Concrete Type Layer                             │
│ - float_e2m1_t: User-facing types              │
│ - Constructors, minimal logic                  │
└─────────────────────────────────────────────────┘
                     │
┌─────────────────────────────────────────────────┐
│ CRTP Base Layer                                 │
│ - float_exmy_base: Operators, conversions      │
│ - Storage, arithmetic via float                │
└─────────────────────────────────────────────────┘
                     │
┌─────────────────────────────────────────────────┐
│ Bit Representation Layer                        │
│ - FpBitRepresentation: Generic encoding        │
│ - Bit operations, conversion algorithm         │
└─────────────────────────────────────────────────┘
```

Each layer has a **clear responsibility**:
- **Trait**: Type information, compile-time dispatch
- **Concrete**: User interface, type identity
- **CRTP Base**: Shared functionality, customization points
- **Bit Rep**: Generic algorithms, bit manipulation

### Separation of Concerns

```
Type Identity        ←  float_e2m1_t (distinct type)
Value Semantics      ←  float_exmy_base (storage, operators)
Format Encoding      ←  FpBitRepresentation (bit layout)
Conversion Algorithm ←  convert() method (IEEE-754 algorithm)
```

Each component is:
- **Testable independently**
- **Reusable** for other formats
- **Understandable** in isolation

### Extensibility

Adding a new format (e.g., E3M2):

```cpp
// 1. Add encoding enum
enum class FpEncoding {
    // ... existing
    E3M2,  // ← New
};

// 2. Add selector specialization
template <>
struct FpEncodingSelector<FpEncoding::E3M2> {
    using type = FpBitRepresentation<uint8_t, 6, 3, 2, NanInfEncoding::NONE>;
};

// 3. Define concrete type
struct float_e3m2_t : public float_exmy_base<FpEncoding::E3M2, float_e3m2_t> {
    using Base = float_exmy_base<FpEncoding::E3M2, float_e3m2_t>;
    float_e3m2_t() = default;
    explicit float_e3m2_t(float x) : Base(x) {}
    // ... constructors
};

// Done! All functionality inherited.
```

## Summary: Why This Design Is Excellent

1. **Zero-overhead abstractions**: Compiles to optimal code
2. **Type safety**: Impossible to mix incompatible types
3. **Code reuse**: Generic algorithms work for all formats
4. **Extensibility**: New formats require minimal code
5. **Compile-time computation**: Constants computed at compile time
6. **Clear separation**: Each layer has distinct responsibility
7. **Self-documenting**: Types express intent clearly

This is **modern C++ template metaprogramming** at its finest.

---

**End of Documentation**

Return to [README.md](README.md) for the complete guide index.
