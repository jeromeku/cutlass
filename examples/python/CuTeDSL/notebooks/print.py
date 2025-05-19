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

# %% [markdown]
# # Printing with CuTe DSL
#
# This notebook demonstrates the different ways to print values in CuTe and explains the important distinction between static (compile-time) and dynamic (runtime) values.
#
# ## Key Concepts
# - Static values: Known at compile time
# - Dynamic values: Only known at runtime
# - Different printing methods for different scenarios
# - Layout representation in CuTe
# - Tensor visualization and formatting

# %%
import cutlass
import cutlass.cute as cute
import numpy as np


# %% [markdown]
# ## Print Example Function
#
# The `print_example` function demonstrates several important concepts:
#
# ### 1. Python's `print` vs CuTe's `cute.printf`
# - `print`: Can only show static values at compile time
# - `cute.printf`: Can display both static and dynamic values at runtime
#
# ### 2. Value Types
# - `a`: Dynamic `Int32` value (runtime)
# - `b`: Static `Constexpr[int]` value (compile-time)
#
# ### 3. Layout Printing
# Shows how layouts are represented differently in static vs dynamic contexts:
# - Static context: Unknown values shown as `?`
# - Dynamic context: Actual values displayed

# %%
@cute.jit
def print_example(a: cutlass.Int32, b: cutlass.Constexpr[int]):
    """
    Demonstrates different printing methods in CuTe and how they handle static vs dynamic values.

    This example shows:
    1. How Python's `print` function works with static values at compile time but can't show dynamic values
    2. How `cute.printf` can display both static and dynamic values at runtime
    3. The difference between types in static vs dynamic contexts
    4. How layouts are represented in both printing methods

    Args:
        a: A dynamic Int32 value that will be determined at runtime
        b: A static (compile-time constant) integer value
    """
    # Use Python `print` to print static information
    print(">>>", b)  # => 2
    # `a` is dynamic value
    print(">>>", a)  # => ?

    # Use `cute.printf` to print dynamic information
    cute.printf(">?? {}", a)  # => 8
    cute.printf(">?? {}", b)  # => 2

    print(">>>", type(a))  # => <class 'cutlass.Int32'>
    print(">>>", type(b))  # => <class 'int'>

    layout = cute.make_layout((a, b))
    print(">>>", layout)            # => (?,2):(1,?)
    cute.printf(">?? {}", layout)   # => (8,2):(1,8)


# %% [markdown]
# ## Compile and Run
#
# **Direct Compilation and Run**
#   - `print_example(cutlass.Int32(8), 2)`
#   - Compiles and runs in one step will execute both static and dynamic print
#     * `>>>` stands for static print
#     * `>??` stands for dynamic print

# %%
print_example(cutlass.Int32(8), 2)

# %% [markdown]
# ## Compile Function
#
# When compiles the function with `cute.compile(print_example, cutlass.Int32(8), 2)`, Python interpreter 
# traces code and only evaluate static expression and print static information.

# %%
print_example_compiled = cute.compile(print_example, cutlass.Int32(8), 2)

# %% [markdown]
# ## Call compiled function
#
# Only print out runtime information

# %%
print_example_compiled(cutlass.Int32(8))


# %% [markdown]
# ## Format String Example
#
# The `format_string_example` function shows an important limitation:
# - F-strings in CuTe are evaluated at compile time
# - This means dynamic values won't show their runtime values in f-strings
# - Use `cute.printf` when you need to see runtime values

# %%
@cute.jit
def format_string_example(a: cutlass.Int32, b: cutlass.Constexpr[int]):
    """
    Format string is evaluated at compile time.
    """
    print(f"a: {a}, b: {b}")

    layout = cute.make_layout((a, b))
    print(f"layout: {layout}")

print("Direct run output:")
format_string_example(cutlass.Int32(8), 2)

# %% [markdown]
# ## Printing Tensor Examples
#
# CuTe provides specialized functionality for printing tensors through the `print_tensor` operation. The `cute.print_tensor` takes the following parameter:
# - `Tensor` (required): A CuTe tensor object that you want to print. The tensor must support load and store operations
# - `verbose` (optional, default=False): A boolean flag that controls the level of detail in the output. When set to True, it will print indices details for each element in the tensor.
#
# Below example code shows the difference between verbose ON and OFF, and how to print a sub range of the given tensor.

# %%
from cutlass.cute.runtime import from_dlpack

@cute.jit
def print_tensor_basic(x : cute.Tensor):
    # Print the tensor
    print("Basic output:")
    cute.print_tensor(x)
    
@cute.jit
def print_tensor_verbose(x : cute.Tensor):
    # Print the tensor with verbose mode
    print("Verbose output:")
    cute.print_tensor(x, verbose=True)

@cute.jit
def print_tensor_slice(x : cute.Tensor, coord : tuple):
    # slice a 2D tensor from the 3D tensor
    sliced_data = cute.slice_(x, coord)
    y = cute.make_fragment(sliced_data.layout, sliced_data.element_type)
    # Convert to TensorSSA format by loading the sliced data into the fragment
    y.store(sliced_data.load())
    print("Slice output:")
    cute.print_tensor(y)


# %% [markdown]
# The default `cute.print_tensor` will output CuTe tensor with datatype, storage space, CuTe layout information, and print data in torch-style format.

# %%
def tensor_print_example1():
    shape = (4, 3, 2)
    
    # Creates [0,...,23] and reshape to (4, 3, 2)
    data = np.arange(24, dtype=np.float32).reshape(*shape) 
      
    print_tensor_basic(from_dlpack(data))

tensor_print_example1()


# %% [markdown]
# The verbosed print will show coodination details of each element in the tensor. The below example shows how we index element in a 2D 4x3 tensor space.

# %%
def tensor_print_example2():
    shape = (4, 3)
    
    # Creates [0,...,11] and reshape to (4, 3)
    data = np.arange(12, dtype=np.float32).reshape(*shape) 
      
    print_tensor_verbose(from_dlpack(data))

tensor_print_example2()


# %% [markdown]
# To print a subset elements in the given Tensor, we can use cute.slice_ to select a range of the given tensor, load them into register and then print the values with `cute.print_tensor`.

# %%
def tensor_print_example3():
    shape = (4, 3)
    
    # Creates [0,...,11] and reshape to (4, 3)
    data = np.arange(12, dtype=np.float32).reshape(*shape) 
      
    print_tensor_slice(from_dlpack(data), (None, 0))
    print_tensor_slice(from_dlpack(data), (1, None))

tensor_print_example3()
