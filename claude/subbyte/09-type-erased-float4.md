# Type-Erased Float4: Runtime Format Selection

[File: include/cutlass/float_subbyte.h:461-467](../../include/cutlass/float_subbyte.h#L461-L467)

## Overview

`type_erased_dynamic_float4_t` is a **union-based type erasure** mechanism that enables **runtime selection** of 4-bit floating point formats without changing the kernel template signature.

## The Complete Definition

```cpp
union type_erased_dynamic_float4_t {
  cutlass::float_e2m1_t e2m1;

  CUTLASS_HOST_DEVICE
  explicit operator cutlass::float_e2m1_t() const {
    return e2m1;
  }
};

template <>
struct sizeof_bits<type_erased_dynamic_float4_t> {
  static constexpr int value = 4;
};
```

**Location**: [include/cutlass/float_subbyte.h:461-472](../../include/cutlass/float_subbyte.h#L461-L472)

## What Is Type Erasure?

**Type erasure** is a C++ pattern that hides concrete type information behind a common interface, enabling:
- Runtime polymorphism without virtual functions
- Single compiled kernel handling multiple data types
- Zero runtime overhead (union is compile-time construct)

```
┌─────────────────────────────────────────────────┐
│         Compile-Time Type System                │
├─────────────────────────────────────────────────┤
│                                                 │
│  template <typename ElementType>                │
│  void kernel(ElementType* data) {               │
│      // Single compiled kernel                  │
│  }                                              │
│                                                 │
│  Instantiated ONCE with:                        │
│  ElementType = type_erased_dynamic_float4_t     │
│                                                 │
├─────────────────────────────────────────────────┤
│         Runtime Format Dispatch                 │
├─────────────────────────────────────────────────┤
│                                                 │
│  MXF8F6F4Format runtime_format;  // enum        │
│  if (user_wants_e2m1) {                        │
│      runtime_format = MXF8F6F4Format::E2M1;    │
│  }                                              │
│  // Kernel receives format as runtime argument  │
│  kernel<<<...>>>(data, runtime_format);        │
│                                                 │
└─────────────────────────────────────────────────┘
```

## Architecture: Four-Layer Type System

```
Layer 4: User-Facing Trait Wrappers
┌────────────────────────────────────────────────────┐
│  type_erased_dynamic_nv_float4_t                   │
│  = nv_float4_t<type_erased_dynamic_float4_t>       │
│                                                    │
│  type_erased_dynamic_mx_float4_t                   │
│  = mx_float4_t<type_erased_dynamic_float4_t>       │
└─────────────────┬──────────────────────────────────┘
                  │
                  │ Uses as DataType
                  ↓
Layer 3: Type-Erased Union (Runtime Dispatch)
┌────────────────────────────────────────────────────┐
│  union type_erased_dynamic_float4_t {              │
│      float_e2m1_t e2m1;                            │
│      // Future: other 4-bit formats                │
│  }                                                 │
└─────────────────┬──────────────────────────────────┘
                  │
                  │ Contains
                  ↓
Layer 2: Concrete Float Types
┌────────────────────────────────────────────────────┐
│  struct float_e2m1_t                               │
│    : float_exmy_base<E2M1, float_e2m1_t>           │
│  {                                                 │
│      uint8_t storage;  // 4 bits used              │
│  }                                                 │
└─────────────────┬──────────────────────────────────┘
                  │
                  │ Inherits from
                  ↓
Layer 1: CRTP Base with All Operations
┌────────────────────────────────────────────────────┐
│  template <Encoding E, class Derived>              │
│  struct float_exmy_base {                          │
│      Storage storage;                              │
│      // All arithmetic, conversion, operators      │
│  }                                                 │
└────────────────────────────────────────────────────┘
```

## Key Design Question: Why Union?

**Problem**: How do you write a single GEMM kernel that can handle multiple 4-bit float formats (E2M1, future E1M2, etc.) without template explosion?

**Traditional Approach** (Template Specialization):
```cpp
// ❌ This creates MULTIPLE compiled kernels
template <typename ElementType>
__global__ void gemm_kernel(ElementType* A, ElementType* B, ...);

// Each instantiation = separate compiled kernel
gemm_kernel<float_e2m1_t><<<...>>>();  // Kernel 1
gemm_kernel<float_e1m2_t><<<...>>>();  // Kernel 2  (hypothetical)
// Problem: 2x binary size, 2x compilation time
```

**Type-Erased Approach** (Union + Runtime Enum):
```cpp
// ✓ Single compiled kernel
__global__ void gemm_kernel(
    type_erased_dynamic_float4_t* A,  // Same type signature
    type_erased_dynamic_float4_t* B,
    MXF8F6F4Format runtime_format) {  // Runtime dispatch

    // CUDA PTX uses runtime format to select instruction
    switch (runtime_format) {
        case MXF8F6F4Format::E2M1:
            // Process as E2M1
            break;
        // Future cases...
    }
}

// Single kernel handles all formats
gemm_kernel<<<...>>>(data_a, data_b, MXF8F6F4Format::E2M1);
```

**Benefits**:
1. **Single Binary**: One compiled kernel, not N kernels
2. **Runtime Flexibility**: Format chosen at launch time
3. **Zero Overhead**: Union has same size as largest member
4. **Forward Compatible**: Add new formats without recompiling user code

## The Union Structure

```cpp
union type_erased_dynamic_float4_t {
    cutlass::float_e2m1_t e2m1;  // Currently the only member
    // Future expansion:
    // cutlass::float_e1m2_t e1m2;
    // cutlass::float_ocp4_t ocp4;
};
```

### Memory Layout

```
Union Memory (4 bits in uint8_t):
┌────────────────┐
│  e2m1 field    │  All members share
│                │  the same memory
│  [future e1m2] │  location
│  [future ocp4] │
└────────────────┘
    4 bits total

At runtime, interpretation depends on MXF8F6F4Format enum:

If format == E2M1:
┌────────────────┐
│ 3  2 1  0      │
│┌─┬─────┬─┐     │
││S│ E E │M│     │
│└─┴─────┴─┘     │
└────────────────┘
  E2M1 format

If format == [future E1M2]:
┌────────────────┐
│ 3  2  1 0      │
│┌─┬──┬────┐     │
││S│E │M M │     │
│└─┴──┴────┘     │
└────────────────┘
  E1M2 format (hypothetical)
```

### Size Specialization

```cpp
template <>
struct sizeof_bits<type_erased_dynamic_float4_t> {
  static constexpr int value = 4;
};
```

**Purpose**: Tells CUTLASS's template machinery:
- This type occupies 4 bits (not 8)
- Memory allocation should pack 2 values per byte
- TMA (Tensor Memory Accelerator) should use 4-bit granularity

## Conversion Operator

```cpp
CUTLASS_HOST_DEVICE
explicit operator cutlass::float_e2m1_t() const {
    return e2m1;
}
```

**Purpose**: Allows explicit conversion to concrete type when format is known:

```cpp
type_erased_dynamic_float4_t erased_value;
// ... value loaded from memory ...

// When we know it's E2M1 at runtime:
float_e2m1_t concrete = static_cast<float_e2m1_t>(erased_value);
float f = float(concrete);  // Now can convert to float
```

**Why Explicit?**: Prevents accidental conversions when format might be wrong.

## The Companion: type_erased_dynamic_float4_unpacksmem_t

```cpp
namespace detail {

union type_erased_dynamic_float4_unpacksmem_t {
  cutlass::detail::float_e2m1_unpacksmem_t e2m1_unpacksmem;

  CUTLASS_HOST_DEVICE
  explicit operator cutlass::detail::float_e2m1_unpacksmem_t() const {
    return e2m1_unpacksmem;
  }
};

}  // namespace detail

template <>
struct sizeof_bits<detail::type_erased_dynamic_float4_unpacksmem_t> {
  static constexpr int value = 4;
};
```

**Location**: [include/cutlass/float_subbyte.h:535-555](../../include/cutlass/float_subbyte.h#L535-L555)

### Purpose: Memory Format Dispatch

CUTLASS needs different types for different memory layouts:

```
PACKED Memory (GMEM):        UNPACKED Shared Memory:
┌──┬──┬──┬──┬──┬──┬──┬──┐   ┌──────┬──────┬──────┬──────┐
│v0│v1│v2│v3│v4│v5│v6│v7│   │ v0   │ v1   │ v2   │ v3   │
└──┴──┴──┴──┴──┴──┴──┴──┘   └──────┴──────┴──────┴──────┘
 4 bits each, tightly packed   4 bits data + padding
 (global memory format)         (shared memory format)
```

**Usage**:
```cpp
// Type dispatch based on memory layout:
template <typename T>
auto select_copy_instruction() {
    if constexpr (is_same_v<T, type_erased_dynamic_float4_t>) {
        return use_packed_tma_load();    // For GMEM → Registers
    }
    else if constexpr (is_same_v<T, detail::type_erased_dynamic_float4_unpacksmem_t>) {
        return use_unpacked_smem_load(); // For SMEM → Registers
    }
}
```

## Runtime Format Enum: MXF8F6F4Format

```cpp
enum class MXF8F6F4Format : uint8_t {
  E4M3 = 0,     // 8-bit: 1 sign + 4 exp + 3 mantissa
  E5M2 = 1,     // 8-bit: 1 sign + 5 exp + 2 mantissa
  E2M3 = 3,     // 6-bit: 1 sign + 2 exp + 3 mantissa
  E3M2 = 4,     // 6-bit: 1 sign + 3 exp + 2 mantissa
  E2M1 = 5,     // 4-bit: 1 sign + 2 exp + 1 mantissa
  INVALID = 7   // Placeholder for type-erased types
};
```

**Location**: [include/cute/arch/mma_sm100_desc.hpp:168-175](../../include/cute/arch/mma_sm100_desc.hpp#L168-L175)

### Why INVALID for Type-Erased Types?

```cpp
template <class T>
CUTE_HOST_DEVICE constexpr auto
to_MXF8F6F4Format() {
  // Concrete types:
  if constexpr (is_same_v<T, float_e2m1_t>) { return MXF8F6F4Format::E2M1; }
  // ...

  // Type-erased types:
  if constexpr (is_same_v<T, type_erased_dynamic_float4_t>) {
    return MXF8F6F4Format::INVALID;  // ← Can't know at compile time!
  }
}
```

**Reason**: Type-erased types don't have a compile-time format. The actual format comes as a **runtime argument** to the kernel.

## Complete Type Hierarchy

```
┌──────────────────────────────────────────────────────────────────┐
│                  USER CODE: Kernel Instantiation                 │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  using ElementA = cutlass::type_erased_dynamic_nv_float4_t;     │
│                                                                  │
│  // Compiles to a single kernel                                 │
│  Gemm<ElementA> gemm;                                           │
│                                                                  │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            │ Type alias expansion
                            ↓
┌──────────────────────────────────────────────────────────────────┐
│              Layer 4: Trait Struct (Type Package)                │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  template <class F4Type>                                        │
│  struct nv_float4_t {                                           │
│      using ScaleFactorType = float_ue4m3_t;                     │
│      using DataType = F4Type;  ← type_erased_dynamic_float4_t   │
│                                                                  │
│      static_assert(is_same_v<F4Type, float_e2m1_t> ||          │
│                    is_same_v<F4Type, type_erased_..._t>,        │
│                    "Only E2M1 or type-erased allowed");         │
│  };                                                             │
│                                                                  │
│  Resulting type:                                                │
│  ┌─────────────────────────────────────────────┐               │
│  │ ScaleFactorType = float_ue4m3_t             │               │
│  │ DataType = type_erased_dynamic_float4_t     │               │
│  └─────────────────────────────────────────────┘               │
│                                                                  │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            │ DataType extracted
                            ↓
┌──────────────────────────────────────────────────────────────────┐
│               Layer 3: Type-Erased Union                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  union type_erased_dynamic_float4_t {                           │
│      float_e2m1_t e2m1;                                         │
│                                                                  │
│      explicit operator float_e2m1_t() const { return e2m1; }    │
│  };                                                             │
│                                                                  │
│  Memory Layout:                                                 │
│  ┌───────────────────┐                                          │
│  │   Shared Storage  │  All members overlay                    │
│  │   (4 bits)        │  same memory                            │
│  │                   │                                          │
│  │ ┌─e2m1────────┐   │                                          │
│  │ │ uint8_t     │   │  Currently 1 member                     │
│  │ │ (4 bits)    │   │  Extensible for future formats          │
│  │ └─────────────┘   │                                          │
│  └───────────────────┘                                          │
│                                                                  │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            │ Contains
                            ↓
┌──────────────────────────────────────────────────────────────────┐
│              Layer 2: Concrete Float Type                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  struct float_e2m1_t                                            │
│    : public float_exmy_base<E2M1, float_e2m1_t>                 │
│  {                                                              │
│      using Base = float_exmy_base<E2M1, float_e2m1_t>;         │
│                                                                  │
│      float_e2m1_t() = default;                                  │
│      explicit float_e2m1_t(float x) : Base(x) {}               │
│      explicit float_e2m1_t(int x) : Base(x) {}                 │
│      float_e2m1_t(Base x) : Base(x) {}                         │
│  };                                                             │
│                                                                  │
│  Storage:                                                       │
│  ┌────────────────┐                                             │
│  │ uint8_t storage│  Inherited from Base                        │
│  └────────────────┘                                             │
│                                                                  │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            │ Inherits from (CRTP)
                            ↓
┌──────────────────────────────────────────────────────────────────┐
│         Layer 1: CRTP Base (All Operations)                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  template <FpEncoding E, class Derived>                         │
│  struct float_exmy_base {                                       │
│      Storage storage;  // uint8_t                               │
│                                                                  │
│      // Construction                                            │
│      explicit float_exmy_base(float x);                         │
│      explicit float_exmy_base(int x);                           │
│                                                                  │
│      // Conversion                                              │
│      operator float() const;                                    │
│      operator int() const;                                      │
│                                                                  │
│      // Arithmetic (via float conversion)                       │
│      Derived operator+(Derived const&) const;                   │
│      Derived operator-(Derived const&) const;                   │
│      Derived operator*(Derived const&) const;                   │
│      Derived operator/(Derived const&) const;                   │
│                                                                  │
│      // Comparison                                              │
│      bool operator==(Derived const&) const;                     │
│      bool operator!=(Derived const&) const;                     │
│      bool operator<(Derived const&) const;                      │
│      // ...                                                     │
│                                                                  │
│      // Bit manipulation                                        │
│      uint8_t raw() const { return storage; }                    │
│      bool signbit() const;                                      │
│      int exponent() const;                                      │
│      int mantissa() const;                                      │
│  };                                                             │
│                                                                  │
│  Uses: FpBitRepresentation<uint8_t, 4, 2, 1, NONE, true>       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Usage Pattern: From User Code to Hardware

### Step 1: User Declares Type

```cpp
// User code
using ElementA = cutlass::type_erased_dynamic_nv_float4_t;
//             = nv_float4_t<type_erased_dynamic_float4_t>
```

### Step 2: Template Expansion

```cpp
// After template expansion:
struct ElementA_Expanded {
    using ScaleFactorType = cutlass::float_ue4m3_t;
    using DataType = cutlass::type_erased_dynamic_float4_t;
};
```

### Step 3: Kernel Instantiation

```cpp
template <typename Element>
__global__ void gemm_kernel(
    typename Element::DataType* A,           // type_erased_dynamic_float4_t*
    typename Element::ScaleFactorType* scaleA,  // float_ue4m3_t*
    MXF8F6F4Format runtime_format) {         // Runtime enum

    // Single compiled kernel
}
```

### Step 4: Runtime Dispatch (Host)

```cpp
// Host code: User chooses format at runtime
MXF8F6F4Format user_choice = MXF8F6F4Format::E2M1;

arguments.hw_info.runtime_format_a = user_choice;
gemm.run(arguments);  // Passes format to kernel
```

### Step 5: Hardware Instruction Selection (Device)

```cpp
// Inside kernel:
__device__ void load_and_compute(
    type_erased_dynamic_float4_t* data,
    MXF8F6F4Format format) {

    // CUDA PTX selects instruction based on format:
    asm volatile(
        "tcgen05.mma.cta_group::2.kind::ab.tiled.f16.f32.f32.f32"
        ".format::%0"  // ← Runtime format substitution
        " {%1}, {%2}, {%3}, {%4};"
        : "=r"(output)
        : "r"(format), "r"(data_a), "r"(data_b), "r"(accum)
    );
}
```

## Comparison: Static vs Type-Erased

### Static Type (Compile-Time Known)

```cpp
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
//               ─────────────────────^^^^^^^^^^^^^^^^^^^^^^^
//                                   Concrete type
```

**Properties**:
- Format known at compile time
- `to_MXF8F6F4Format()` returns `MXF8F6F4Format::E2M1`
- No runtime format argument needed
- Slightly smaller kernel (no format dispatch code)

**Use Case**: When you know the format ahead of time and don't need flexibility.

### Type-Erased (Runtime Dispatch)

```cpp
using ElementA = cutlass::type_erased_dynamic_nv_float4_t;
//             = nv_float4_t<type_erased_dynamic_float4_t>
//               ──────────────^^^^^^^^^^^^^^^^^^^^^^^^^────
//                             Union type
```

**Properties**:
- Format chosen at runtime
- `to_MXF8F6F4Format()` returns `MXF8F6F4Format::INVALID`
- Requires runtime format argument to kernel
- Single binary handles all formats

**Use Case**: When format depends on user input, dataset, or runtime profiling.

## Example: Runtime Format Selection

```cpp
// Complete example showing runtime format selection
int main(int argc, char** argv) {
    // User specifies format at runtime (e.g., command-line arg)
    std::string format_str = argv[1];  // "e2m1" or future "e1m2"

    // Convert to enum
    MXF8F6F4Format runtime_format;
    if (format_str == "e2m1") {
        runtime_format = MXF8F6F4Format::E2M1;
    }
    // Future: else if (format_str == "e1m2") { ... }

    // Single kernel type handles all formats
    using ElementA = cutlass::type_erased_dynamic_nv_float4_t;
    using Gemm = /* ... instantiated with ElementA ... */;

    Gemm gemm;
    auto args = gemm.to_underlying_arguments(problem_size, ...);

    // Pass runtime format to kernel
    args.hw_info.runtime_format_a = runtime_format;
    args.hw_info.runtime_format_b = runtime_format;

    // Single compiled kernel, runtime dispatch
    gemm.run(args);

    return 0;
}
```

## Design Patterns

### 1. Type Erasure via Union

**Pattern**: Use union to allow multiple types to share memory, with runtime dispatch determining interpretation.

```cpp
union TypeErased {
    ConcreteTypeA a;
    ConcreteTypeB b;
};
```

**Benefits**:
- Single binary
- Runtime flexibility
- Zero overhead (union size = max member size)

### 2. Dual Type System

**Pattern**: Maintain parallel types for different memory layouts.

```cpp
type_erased_dynamic_float4_t          // For GMEM (packed)
type_erased_dynamic_float4_unpacksmem_t  // For SMEM (unpacked)
```

**Benefits**:
- Correct PTX instruction selection
- Optimal memory layout for each memory hierarchy level
- Compile-time dispatch via template specialization

### 3. Trait Struct Wrapper

**Pattern**: Wrap type-erased union in trait struct for scale factor association.

```cpp
template <class F4Type>
struct nv_float4_t {
    using ScaleFactorType = float_ue4m3_t;
    using DataType = F4Type;  // Can be concrete or type-erased
};
```

**Benefits**:
- Consistent interface for both static and dynamic types
- Scale factor type automatically associated
- Compile-time validation via static_assert

### 4. Runtime Enum Companion

**Pattern**: Pair type-erased union with runtime enum for dispatch.

```cpp
union TypeErased { /* ... */ };
enum class FormatEnum { TYPE_A, TYPE_B, INVALID };
```

**Benefits**:
- Efficient runtime dispatch (integer comparison)
- Explicit handling of unknown/invalid states
- Can be passed as kernel argument

### 5. Explicit Conversion Only

**Pattern**: Provide only explicit conversion operators, never implicit.

```cpp
explicit operator float_e2m1_t() const { return e2m1; }
// No: operator float_e2m1_t() const { ... }  (implicit)
```

**Benefits**:
- Prevents accidental conversions when format might be wrong
- Forces programmer to acknowledge format assumption
- Compiler catches misuse at compile time

## Memory and Performance

### Size

```cpp
sizeof(type_erased_dynamic_float4_t) == sizeof(float_e2m1_t) == 1 byte
//                                                                (4 bits used)
```

Union size = size of largest member. Currently only one member, so 1 byte.

### Overhead

**Compile-time**: Zero. Union is a compile-time construct.

**Runtime**: Minimal. Format dispatch is typically:
```asm
// Single integer comparison in PTX
setp.eq.u8  %p, format, E2M1
@%p  tcgen05.mma.format::e2m1 ...
@!%p tcgen05.mma.format::other ...
```

### Binary Size

**Without Type Erasure**:
- N formats = N compiled kernels
- Binary size = N × kernel_size

**With Type Erasure**:
- N formats = 1 compiled kernel + small dispatch overhead
- Binary size = 1 × kernel_size + dispatch_code
- Savings: (N-1) × kernel_size - dispatch_code

For large kernels: **Massive binary size reduction**.

## Future Extensibility

The union design supports adding new 4-bit formats:

```cpp
union type_erased_dynamic_float4_t {
    cutlass::float_e2m1_t e2m1;        // Current
    cutlass::float_e1m2_t e1m2;        // Future
    cutlass::ocp_fp4_t ocp4;           // Hypothetical OCP format
};
```

**Adding a New Format**:
1. Define concrete type (e.g., `float_e1m2_t`)
2. Add member to union
3. Add enum value to `MXF8F6F4Format`
4. Update runtime dispatch in kernel
5. **No user code changes needed** (if using type-erased types)

## Limitations

### 1. Loss of Type Safety

```cpp
type_erased_dynamic_float4_t value = load_from_memory(...);

// Danger: No compile-time check that this is actually E2M1
float_e2m1_t concrete = static_cast<float_e2m1_t>(value);
```

**Mitigation**: Always pass runtime format enum alongside data.

### 2. Currently E2M1 Only

Union has only one member. Type erasure infrastructure is in place, but other formats not yet implemented.

### 3. Requires Runtime Format Argument

Kernels must accept `MXF8F6F4Format` argument, adding slight complexity.

### 4. Cannot Infer Format at Compile Time

```cpp
// This always returns INVALID:
auto format = to_MXF8F6F4Format<type_erased_dynamic_float4_t>();
// format == MXF8F6F4Format::INVALID
```

Must use runtime enum instead.

## Related Types

### type_erased_dynamic_float6_t

```cpp
union type_erased_dynamic_float6_t {
    cutlass::float_e2m3_t e2m3;
    cutlass::float_e3m2_t e3m2;
};
```

**Location**: Defined similarly in [include/cutlass/float_subbyte.h](../../include/cutlass/float_subbyte.h)

**Difference**: Has **two members** (E2M3 and E3M2), showing how union supports multiple formats.

### type_erased_dynamic_float8_t

```cpp
union type_erased_dynamic_float8_t {
    cutlass::float_e4m3_t e4m3;
    cutlass::float_e5m2_t e5m2;
};
```

**Location**: [include/cutlass/float8.h](../../include/cutlass/float8.h)

**Usage**: Same pattern for 8-bit float formats.

## Summary

`type_erased_dynamic_float4_t` is a **union-based type erasure** mechanism enabling:

1. **Runtime Format Selection**: Choose float format at kernel launch time
2. **Single Binary**: One compiled kernel instead of N per format
3. **Zero Overhead**: Union size equals largest member, no runtime cost
4. **Future Extensible**: Add new formats by adding union members
5. **Consistent Interface**: Works with same trait structs as static types

**Key Insight**: Type erasure moves type information from compile-time to runtime, trading some type safety for binary size reduction and runtime flexibility.

**Design Philosophy**: Optimize for the common case (single format at runtime) while supporting advanced use cases (multi-format support) through a unified interface.

---

**Next**: For complete call chain examples showing how type-erased values flow through the system, see [07-call-chains.md](07-call-chains.md).

**See Also**:
- [05-float-e2m1.md](05-float-e2m1.md) - The concrete type contained in the union
- [06-nv-float4.md](06-nv-float4.md) - The trait struct that wraps type-erased types
- [08-design-patterns.md](08-design-patterns.md) - Design patterns including type erasure
