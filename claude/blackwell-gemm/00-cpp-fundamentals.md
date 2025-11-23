# C++ Template Fundamentals for CUTLASS

## Introduction

CUTLASS makes extensive use of modern C++ template metaprogramming. This document provides the foundational knowledge needed to understand the templated code in the Blackwell GEMM examples.

## Table of Contents

1. [Function Templates](#function-templates)
2. [Class Templates](#class-templates)
3. [Template Specialization](#template-specialization)
4. [Non-Type Template Parameters](#non-type-template-parameters)
5. [Variadic Templates](#variadic-templates)
6. [Type Traits and SFINAE](#type-traits-and-sfinae)
7. [Compile-Time Constants](#compile-time-constants)
8. [Auto and Decltype](#auto-and-decltype)
9. [Structured Bindings](#structured-bindings)
10. [Common Patterns in CUTLASS](#common-patterns-in-cutlass)

---

## Function Templates

### Basic Function Template

```cpp
// Generic function that works with any type
template <class T>
T add(T a, T b) {
  return a + b;
}

// Usage
int x = add(1, 2);          // T = int
float y = add(1.0f, 2.0f);  // T = float
```

**In CUTLASS**: Functions like `make_tensor`, `make_layout`, `partition_A` are all templated.

### Example from CUTLASS

```cpp
// From include/cute/tensor.hpp (simplified)
template <class Pointer, class Layout>
CUTE_HOST_DEVICE
auto make_tensor(Pointer ptr, Layout layout) {
  return Tensor<Pointer, Layout>{ptr, layout};
}
```

**Usage in examples**:
```cpp
// examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu:428
Tensor mA = make_tensor(make_gmem_ptr(device_ptr_A), layout_A);
//         └─────┬────┘  └──────────┬─────────────┘  └────┬────┘
//            return       Pointer type (gmem_ptr)    Layout type
```

---

## Class Templates

### Basic Class Template

```cpp
// Container for any type
template <class T>
class Container {
  T value;
public:
  Container(T v) : value(v) {}
  T get() { return value; }
};

// Usage
Container<int> c1(42);
Container<float> c2(3.14f);
```

### Example from CUTLASS: Tensor

```cpp
// From include/cute/tensor.hpp (simplified)
template <class Engine, class Layout>
struct Tensor {
  Engine engine_;   // Stores pointer/data
  Layout layout_;   // Stores shape/stride information

  // Access operator
  template <class Coord>
  CUTE_HOST_DEVICE
  auto operator()(Coord const& coord) const {
    return engine_[layout_(coord)];
  }
};
```

**What does this mean?**
- `Tensor<gmem_ptr<float>, Layout<_512, _256>>` creates a tensor with:
  - `Engine` = `gmem_ptr<float>` (global memory pointer to float)
  - `Layout` = `Layout<_512, _256>` (shape information)

---

## Template Specialization

### Full Specialization

```cpp
// General template
template <class T>
struct TypeName {
  static const char* name() { return "unknown"; }
};

// Specialization for int
template <>
struct TypeName<int> {
  static const char* name() { return "int"; }
};

// Specialization for float
template <>
struct TypeName<float> {
  static const char* name() { return "float"; }
};
```

### Partial Specialization

```cpp
// General template
template <class T, int N>
struct Array {
  T data[N];
};

// Partial specialization for N=0
template <class T>
struct Array<T, 0> {
  // Empty, no storage needed
};
```

### Example from CUTLASS: MMA Instructions

The examples use different MMA instructions based on template parameters:

```cpp
// From include/cute/arch/mma_sm100_umma.hpp

// 1SM instruction (used in Example 03)
template <class a_type, class b_type, class c_type,
          int M, int N,
          UMMA::Major a_major, UMMA::Major b_major>
struct SM100_MMA_F16BF16_SS {  // SS = SMEM → SMEM
  // ... M must be 64 or 128
  // ... N must be 8-256 (multiple of 8)
};

// 2SM instruction (used in Example 04)
template <class a_type, class b_type, class c_type,
          int M, int N,
          UMMA::Major a_major, UMMA::Major b_major>
struct SM100_MMA_F16BF16_2x1SM_SS {  // 2x1SM = 2 SMs collaborate
  // ... M must be 128 or 256
  // ... Different constraints
};
```

**In Example 03** ([03_mma_tma_multicast_sm100.cu:448](../../examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu#L448)):
```cpp
TiledMMA tiled_mma = make_tiled_mma(
  SM100_MMA_F16BF16_SS<TypeA, TypeB, TypeC, 128, 256, UMMA::Major::K, UMMA::Major::K>{}
  //                   └──┬──┘ └──┬──┘ └──┬──┘ └─┬─┘ └─┬─┘ └───────────┬───────────┘
  //                    F16     F16    F32    M=128 N=256  Both K-major (col-major)
);
```

**In Example 04** ([04_mma_tma_2sm_sm100.cu:450](../../examples/cute/tutorial/blackwell/04_mma_tma_2sm_sm100.cu#L450)):
```cpp
TiledMMA tiled_mma = make_tiled_mma(
  SM100_MMA_F16BF16_2x1SM_SS<TypeA, TypeB, TypeC, 256, 256, UMMA::Major::K, UMMA::Major::K>{}
  //               └─┬──┘ Notice "2x1SM" and M=256 (double the M dimension)
  //                 │
  //            2SM instruction
);
```

**Key Point**: The template parameters select completely different hardware instructions at compile-time!

---

## Non-Type Template Parameters

Templates can take values, not just types:

```cpp
// Template with integer parameter
template <int N>
struct FixedArray {
  int data[N];  // Array size known at compile-time
};

// Usage
FixedArray<10> arr1;  // Array of 10 ints
FixedArray<20> arr2;  // Array of 20 ints (different type!)
```

### Example from CUTLASS: Compile-Time Dimensions

```cpp
// From include/cute/numeric/integral_constant.hpp
template <int N>
using Int = cute::integral_constant<int, N>;

// In the examples:
auto bK = tile_size<2>(tiled_mma) * Int<4>{};
//                                   └──┬──┘
//                               Compile-time constant 4

// This becomes a compile-time multiplication:
// If tile_size<2>(tiled_mma) = _16, then bK = _64 (known at compile-time)
```

**Why does this matter?**
- Compiler can optimize away loops
- No runtime overhead
- Enables template specialization based on values

---

## Variadic Templates

Templates that accept any number of arguments:

```cpp
// Function with variable number of arguments
template <class... Args>
void print(Args... args) {
  // Fold expression (C++17)
  (std::cout << ... << args) << '\n';
}

// Usage
print(1, 2, 3);
print("Hello", ' ', "World", '!');
```

### Example from CUTLASS: make_shape

```cpp
// From include/cute/numeric/int_tuple.hpp
template <class... Ts>
CUTE_HOST_DEVICE constexpr
auto make_shape(Ts const&... ts) {
  return cute::make_tuple(ts...);
}

// Usage in examples:
auto cluster_shape = make_shape(Int<4>{}, Int<4>{}, Int<1>{});
//                              └───┬──┘  └───┬──┘  └───┬──┘
//                                 M=4      N=4      K=1
// Result: Shape<_4, _4, _1>
```

---

## Type Traits and SFINAE

### Basic Type Traits

```cpp
#include <type_traits>

// Check if T is an integer type
template <class T>
void process(T value) {
  if constexpr (std::is_integral_v<T>) {
    // Integer-specific code
  } else {
    // Non-integer code
  }
}
```

### SFINAE (Substitution Failure Is Not An Error)

Used to enable/disable templates based on conditions:

```cpp
// Only enable for integer types
template <class T,
          std::enable_if_t<std::is_integral_v<T>, int> = 0>
void process_int(T value) {
  // Only available for integers
}
```

### Example from CUTLASS

```cpp
// From include/cute/atom/copy_traits.hpp
template <class... Args>
struct Copy_Traits {
  // Enabled only if certain conditions are met
  template <class TS, class SLayout, class TD, class DLayout>
  CUTE_HOST_DEVICE static constexpr auto
  copy(TS const& src, SLayout const& slayout,
       TD      & dst, DLayout const& dlayout) {
    // Implementation
  }
};
```

---

## Compile-Time Constants

### Why Compile-Time Constants?

In CUTLASS, many dimensions are known at compile-time. This allows:
1. **Better optimization**: Compiler can unroll loops, eliminate branches
2. **Template specialization**: Different code for different sizes
3. **Static assertions**: Catch errors at compile-time

### CuTe's Integral Constants

```cpp
// From include/cute/numeric/integral_constant.hpp

// Predefined constants
using _0  = Int<0>;
using _1  = Int<1>;
using _2  = Int<2>;
// ... up to _256

// Usage
constexpr _4 four;        // Compile-time constant
constexpr auto eight = _4{} * _2{};  // Also compile-time!
```

### In the Examples

```cpp
// examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu:467
auto bK = tile_size<2>(tiled_mma) * Int<4>{};
//        └───────────┬───────────┘   └──┬──┘
//        Returns _16              Compile-time 4
//
//        Result: _64 (= 16 * 4), known at compile-time!
```

**Static Assertions**:
```cpp
// examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu:476
if (not evenly_divides(shape(mma_tiler), tile_shape(tiled_mma))) {
  std::cerr << "The MMA Shape should evenly divide the MMA Tiler." << std::endl;
  return;
}
```

Since dimensions are compile-time, `evenly_divides` can be evaluated at compile-time!

---

## Auto and Decltype

### Auto Type Deduction

```cpp
auto x = 42;                    // x has type int
auto y = 3.14;                  // y has type double
auto z = std::vector<int>{};    // z has type std::vector<int>
```

### Why Auto in CUTLASS?

Template types can be very complex:

```cpp
// Without auto:
Tensor<ComposedLayout<ViewEngine<MakePointer<half_t, 0>, 1>,
       Layout<Shape<_128, _256>, Stride<_256, _1>>>> mA = ...;

// With auto:
auto mA = make_tensor(make_gmem_ptr(device_ptr_A), layout_A);
```

### Decltype

Gets the type of an expression:

```cpp
int x = 0;
decltype(x) y;  // y has type int

// Used in CUTLASS
template <class Layout>
using layout_type = decltype(make_layout(Layout{}));
```

### Example from Examples

```cpp
// examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu:517
using SMEMStorage = SharedStorage<TypeA, TypeB,
                                  decltype(sA_layout),
                                  decltype(sB_layout)>;
//                                └───────┬──────────┘
//                            Gets the type of sA_layout
```

---

## Structured Bindings

Decompose objects into their components (C++17):

```cpp
// Tuple/Pair decomposition
std::pair<int, float> p{42, 3.14f};
auto [a, b] = p;  // a=42, b=3.14f

// Struct decomposition
struct Point { int x; int y; };
Point pt{10, 20};
auto [x, y] = pt;  // x=10, y=20
```

### Example from CUTLASS

The examples use structured bindings extensively:

```cpp
// examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu:288
auto [tAgA, tAsA] = tma_partition(...);
//    └─┬─┘ └─┬─┘
//      │     └─ TMA SMEM tensor
//      └─ TMA GMEM tensor
```

**What's happening?**
`tma_partition` returns a `cute::tuple<GMemTensor, SMemTensor>`, which is decomposed into two variables.

---

## Common Patterns in CUTLASS

### Pattern 1: Type Extraction with `decltype`

```cpp
// Get the type of a complex expression
auto mma_tiler = make_shape(bM, bN, bK);
using MmaTiler = decltype(mma_tiler);

// Now MmaTiler can be used as a template parameter
template <class T>
void kernel(MmaTiler tiler) { ... }
```

### Pattern 2: Compile-Time Conditionals

```cpp
if constexpr (is_1sm_mma<MMA>::value) {
  // Code for 1SM MMA
} else if constexpr (is_2sm_mma<MMA>::value) {
  // Code for 2SM MMA
}
```

### Pattern 3: Perfect Forwarding

```cpp
template <class... Args>
auto forward_call(Args&&... args) {
  return function(std::forward<Args>(args)...);
}
```

This preserves the value category (lvalue/rvalue) of arguments.

### Pattern 4: Template Parameter Deduction

```cpp
// Compiler deduces template parameters from arguments
template <class T>
void process(T value);

process(42);    // T deduced as int
process(3.14);  // T deduced as double
```

### Pattern 5: Tag Dispatch

```cpp
// Different implementations based on type tags
template <class T>
void impl(T value, std::true_type /* is_integral */) {
  // Integer implementation
}

template <class T>
void impl(T value, std::false_type /* is_integral */) {
  // Non-integer implementation
}

template <class T>
void process(T value) {
  impl(value, std::is_integral<T>{});  // Dispatch based on type
}
```

**In CUTLASS**: Used extensively for selecting algorithms based on memory types, layouts, etc.

---

## Understanding Template Errors

Template errors can be intimidating. Here's how to read them:

### Example Error

```
error: no matching function for call to 'make_tensor(int*, Layout<Shape<_512>, Stride<_2>>)'
note: candidate template ignored: requirement 'is_supported_layout<Layout<Shape<_512>, Stride<_2>>>::value' was not satisfied
```

**How to read this**:
1. **Main error**: Can't find `make_tensor` for given arguments
2. **Note**: Tells you why - the layout doesn't satisfy `is_supported_layout`
3. **Fix**: Check layout requirements

### Tips for Template Debugging

1. **Break down complex expressions**:
   ```cpp
   // Instead of:
   auto x = foo(bar(baz(qux())));

   // Do:
   auto a = qux();
   auto b = baz(a);
   auto c = bar(b);
   auto x = foo(c);  // Easier to see where error occurs
   ```

2. **Use `static_assert` to check assumptions**:
   ```cpp
   static_assert(M == 128 || M == 256, "M must be 128 or 256");
   ```

3. **Print types with errors**:
   ```cpp
   template <class T>
   struct ShowType;  // Never defined

   ShowType<decltype(my_variable)> x;  // Error shows the type
   ```

---

## Exercises

To solidify understanding, try these exercises:

### Exercise 1: Basic Template Function

Create a template function that returns the maximum of two values:

```cpp
template <class T>
T max(T a, T b) {
  return (a > b) ? a : b;
}
```

### Exercise 2: Template Specialization

Specialize the `max` function for pointers to compare the pointed values:

```cpp
template <class T>
T* max(T* a, T* b) {
  return (*a > *b) ? a : b;
}
```

### Exercise 3: Compile-Time Constants

Create a compile-time factorial:

```cpp
template <int N>
struct Factorial {
  static constexpr int value = N * Factorial<N-1>::value;
};

template <>
struct Factorial<0> {
  static constexpr int value = 1;
};

// Usage
constexpr int fact5 = Factorial<5>::value;  // 120
```

---

## Summary

Key takeaways for understanding CUTLASS templates:

1. **Templates enable generic programming**: Same code works for different types
2. **Specialization selects different implementations**: Different hardware instructions based on parameters
3. **Compile-time constants enable optimization**: No runtime overhead
4. **Auto simplifies complex type names**: Makes code readable
5. **Type traits enable conditional compilation**: Different code paths for different types
6. **Understanding template errors is a skill**: Practice breaking down errors

With this foundation, you're ready to dive into the CUTLASS architecture and understand how the heavily templated code generates efficient GPU kernels.

---

## Next Steps

- Continue to [01-architecture-overview.md](01-architecture-overview.md) to learn about SM100 architecture
- Refer back to this document when encountering unfamiliar C++ patterns

