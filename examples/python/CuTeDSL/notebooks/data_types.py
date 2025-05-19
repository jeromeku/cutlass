# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from typing import List

import cutlass
import cutlass.cute as cute


# %% [markdown]
# ## Understanding data structure in CuTe DSL
#
# In most cases, data structures in CuTe DSL work the same as Python data structures with the notable difference that Python data structures in most cases are considered as static data which are interpreted by the DSL compiler embedded inside Python interpreter.
#
# To differentiate between compile-time and runtime values, CuTe DSL introduces primitive types that 
# represent dynamic values in JIT-compiled code.
#
# CuTe DSL provides a comprehensive set of primitive numeric types for representing dynamic values at 
# runtime. These types are formally defined within the CuTe DSL typing system:
#
# ### Integer Types
# - `Int8` - 8-bit signed integer
# - `Int16` - 16-bit signed integer  
# - `Int32` - 32-bit signed integer
# - `Int64` - 64-bit signed integer
# - `Int128` - 128-bit signed integer
# - `Uint8` - 8-bit unsigned integer
# - `Uint16` - 16-bit unsigned integer
# - `Uint32` - 32-bit unsigned integer
# - `Uint64` - 64-bit unsigned integer
# - `Uint128` - 128-bit unsigned integer
#
# ### Floating Point Types
# - `Float16` - 16-bit floating point
# - `Float32` - 32-bit floating point 
# - `Float64` - 64-bit floating point
# - `BFloat16` - Brain Floating Point format (16-bit)
# - `TFloat32` - Tensor Float32 format (reduced precision format used in tensor operations)
# - `Float8E4M3` - 8-bit floating point with 4-bit exponent and 3-bit mantissa
# - `Float8E5M2` - 8-bit floating point with 5-bit exponent and 2-bit mantissa
#
# These specialized types are designed to represent dynamic values in CuTe DSL code that will be 
# evaluated at runtime, in contrast to Python's built-in numeric types which are evaluated during 
# compilation.
#
# ### Example usage:
#
# ```python
# x = cutlass.Int32(5)        # Creates a 32-bit integer
# y = cutlass.Float32(3.14)   # Creates a 32-bit float
#
# @cute.jit
# def foo(a: cutlass.Int32):  # annotate `a` as 32-bit integer passed to jit function via ABI
#     ...
# ```
# To differentiate between compile-time and runtime values, CuTe DSL introduces primitive types that 
# represent dynamic values in JIT-compiled code.
#
# CuTe DSL provides a comprehensive set of primitive numeric types for representing dynamic values at 
# runtime. These types are formally defined within the CuTe DSL typing system:
#
# ### Integer Types
# - `Int8` - 8-bit signed integer
# - `Int16` - 16-bit signed integer  
# - `Int32` - 32-bit signed integer
# - `Int64` - 64-bit signed integer
# - `Int128` - 128-bit signed integer
# - `Uint8` - 8-bit unsigned integer
# - `Uint16` - 16-bit unsigned integer
# - `Uint32` - 32-bit unsigned integer
# - `Uint64` - 64-bit unsigned integer
# - `Uint128` - 128-bit unsigned integer
#
# ### Floating Point Types
# - `Float16` - 16-bit floating point
# - `Float32` - 32-bit floating point 
# - `Float64` - 64-bit floating point
# - `BFloat16` - Brain Floating Point format (16-bit)
# - `TFloat32` - Tensor Float32 format (reduced precision format used in tensor operations)
# - `Float8E4M3` - 8-bit floating point with 4-bit exponent and 3-bit mantissa
# - `Float8E5M2` - 8-bit floating point with 5-bit exponent and 2-bit mantissa
#
# These specialized types are designed to represent dynamic values in CuTe DSL code that will be 
# evaluated at runtime, in contrast to Python's built-in numeric types which are evaluated during 
# compilation.
#
# ### Example usage:
#
# ```python
# x = cutlass.Int32(5)        # Creates a 32-bit integer
# y = cutlass.Float32(3.14)   # Creates a 32-bit float
#
# @cute.jit
# def foo(a: cutlass.Int32):  # annotate `a` as 32-bit integer passed to jit function via ABI
#     ...
# ```

# %%
@cute.jit
def bar():
    a = cutlass.Float32(3.14)
    print("a(static) =", a)             # prints `a(static) = ?`
    cute.printf("a(dynamic) = {}", a)   # prints `a(dynamic) = 3.140000`

    b = cutlass.Int32(5)
    print("b(static) =", b)             # prints `b(static) = 5`
    cute.printf("b(dynamic) = {}", b)   # prints `b(dynamic) = 5`

bar()


# %% [markdown]
# ### Type Conversion API
#
# CUTLASS numeric types provide type conversion through the `to()` method available on all Numeric types. This allows you to convert between different numeric data types at runtime.
#
# Syntax:
#
# ```python
# new_value = value.to(target_type)
# ```
#
# The `to()` method supports conversion between:
# - Integer types (Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64)
# - Floating point types (Float16, Float32, Float64, BFloat16)
# - Mixed integer/floating point conversions
#
# Note that when converting from floating point to integer types, the decimal portion is truncated. When converting between types with different ranges, values may be clamped or lose precision if they exceed the target type's representable range.

# %%
@cute.jit
def type_conversion():
    # Convert from Int32 to Float32
    x = cutlass.Int32(42)
    y = x.to(cutlass.Float32)
    cute.printf("Int32({}) => Float32({})", x, y)

    # Convert from Float32 to Int32
    a = cutlass.Float32(3.14)
    b = a.to(cutlass.Int32)
    cute.printf("Float32({}) => Int32({})", a, b)

    # Convert from Int32 to Int8
    c = cutlass.Int32(127)
    d = c.to(cutlass.Int8)
    cute.printf("Int32({}) => Int8({})", c, d)

    # Convert from Int32 to Int8 with value exceeding Int8 range
    e = cutlass.Int32(300)
    f = e.to(cutlass.Int8)
    cute.printf("Int32({}) => Int8({}) (truncated due to range limitation)", e, f)

type_conversion()


# %% [markdown]
# ### Operator Overloading
#
# CUTLASS numeric types support Python's built-in operators, allowing you to write natural mathematical expressions. The operators work with both CUTLASS numeric types and Python native numeric types.
#
# Supported operators include:
# - Arithmetic: `+`, `-`, `*`, `/`, `//`, `%`, `**`
# - Comparison: `<`, `<=`, `==`, `!=`, `>=`, `>`
# - Bitwise: `&`, `|`, `^`, `<<`, `>>`
# - Unary: `-` (negation), `~` (bitwise NOT)

# %%
@cute.jit
def operator_demo():
    # Arithmetic operators
    a = cutlass.Int32(10)
    b = cutlass.Int32(3)
    cute.printf("a: Int32({}), b: Int32({})", a, b)

    x = cutlass.Float32(5.5)
    cute.printf("x: Float32({})", x)

    cute.printf("")

    sum_result = a + b
    cute.printf("a + b = {}", sum_result)

    y = x * 2  # Multiplying with Python native type
    cute.printf("x * 2 = {}", y)

    # Mixed type arithmetic (Int32 + Float32) that integer is converted into float32
    mixed_result = a + x
    cute.printf("a + x = {} (Int32 + Float32 promotes to Float32)", mixed_result)

    # Division with Int32 (note: integer division)
    div_result = a / b
    cute.printf("a / b = {}", div_result)

    # Float division
    float_div = x / cutlass.Float32(2.0)
    cute.printf("x / 2.0 = {}", float_div)

    # Comparison operators
    is_greater = a > b
    cute.printf("a > b = {}", is_greater)

    # Bitwise operators
    bit_and = a & b
    cute.printf("a & b = {}", bit_and)

    neg_a = -a
    cute.printf("-a = {}", neg_a)

    not_a = ~a
    cute.printf("~a = {}", not_a)

operator_demo()

