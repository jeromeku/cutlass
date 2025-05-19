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
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

import numpy as np
import torch


# %% [markdown]
# # Introduction to the TensorSSA in CuTe DSL
#
# This tutorial introduces what is the `TensorSSA` and why we need it. We also give some examples to show how to use `TensorSSA`.
#
# ## What is TensorSSA
#
# `TensorSSA` is a Python class that represents a tensor value in Static Single Assignment (SSA) form within the CuTe DSL. You can think of it as a tensor residing in a (simulated) register.
#
# ## Why TensorSSA
#
# `TensorSSA` encapsulates the underlying MLIR tensor value into an object that's easier to manipulate in Python. By overloading numerous Python operators (like `+`, `-`, `*`, `/`, `[]`, etc.), it allows users to express tensor computations (primarily element-wise operations and reductions) in a more Pythonic way. These element-wise operations are then translated into optimized vectorization instructions.
#
# It's part of the CuTe DSL, serving as a bridge between the user-described computational logic and the lower-level MLIR IR, particularly for representing and manipulating register-level data.
#
# ## When to use TensorSSA
#
# `TensorSSA` is primarily used in the following scenarios:
#
# ### Load from memory and store to memory

# %%
@cute.jit
def load_and_store(res: cute.Tensor, a: cute.Tensor, b: cute.Tensor):
    """
    Load data from memory and store the result to memory.

    :param res: The destination tensor to store the result.
    :param a: The source tensor to be loaded.
    :param b: The source tensor to be loaded.
    """
    a_vec = a.load()
    print(f"a_vec: {a_vec}")      # prints `a_vec: vector<12xf32> o (3, 4)`
    b_vec = b.load()
    print(f"b_vec: {b_vec}")      # prints `b_vec: vector<12xf32> o (3, 4)`
    res.store(a_vec + b_vec)
    cute.print_tensor(res)

a = np.ones(12).reshape((3, 4)).astype(np.float32)
b = np.ones(12).reshape((3, 4)).astype(np.float32)
c = np.zeros(12).reshape((3, 4)).astype(np.float32)
load_and_store(from_dlpack(c), from_dlpack(a), from_dlpack(b))


# %% [markdown]
# ### Register-Level Tensor Operations
#
# When writing kernel logic, various computations, transformations, slicing, etc., are performed on data loaded into registers.

# %%
@cute.jit
def apply_slice(src: cute.Tensor, dst: cute.Tensor, indices: cutlass.Constexpr):
    """
    Apply slice operation on the src tensor and store the result to the dst tensor.

    :param src: The source tensor to be sliced.
    :param dst: The destination tensor to store the result.
    :param indices: The indices to slice the source tensor.
    """
    src_vec = src.load()
    dst_vec = src_vec[indices]
    print(f"{src_vec} -> {dst_vec}")
    if isinstance(dst_vec, cute.TensorSSA):
        dst.store(dst_vec)
        cute.print_tensor(dst)
    else:
        dst[0] = dst_vec
        cute.print_tensor(dst)

def slice_1():
    src_shape = (4, 2, 3)
    dst_shape = (4, 3)
    indices = (None, 1, None)

    """
    a:
    [[[ 0.  1.  2.]
      [ 3.  4.  5.]]

     [[ 6.  7.  8.]
      [ 9. 10. 11.]]

     [[12. 13. 14.]
      [15. 16. 17.]]

     [[18. 19. 20.]
      [21. 22. 23.]]]
    """
    a = np.arange(np.prod(src_shape)).reshape(*src_shape).astype(np.float32)
    dst = np.random.randn(*dst_shape).astype(np.float32)
    apply_slice(from_dlpack(a), from_dlpack(dst), indices)

slice_1()


# %%
def slice_2():
    src_shape = (4, 2, 3)
    dst_shape = (1,)
    indices = 10
    a = np.arange(np.prod(src_shape)).reshape(*src_shape).astype(np.float32)
    dst = np.random.randn(*dst_shape).astype(np.float32)
    apply_slice(from_dlpack(a), from_dlpack(dst), indices)

slice_2()


# %% [markdown]
# ## Arithmetic Operations
#
# As we mentioned earlier, there're many tensor operations whose operands are `TensorSSA`. And they are all element-wise operations. We give some examples below.
#
# ### Binary Operations
#
# For binary operations, the LHS operand is `TensorSSA` and the RHS operand can be either `TensorSSA` or `Numeric`. When the RHS is `Numeric`, it will be broadcast to a `TensorSSA`.

# %%
@cute.jit
def binary_op_1(res: cute.Tensor, a: cute.Tensor, b: cute.Tensor):
    a_vec = a.load()
    b_vec = b.load()

    add_res = a_vec + b_vec
    res.store(add_res)
    cute.print_tensor(res)        # prints [3.000000, 3.000000, 3.000000]

    sub_res = a_vec - b_vec
    res.store(sub_res)
    cute.print_tensor(res)        # prints [-1.000000, -1.000000, -1.000000]

    mul_res = a_vec * b_vec
    res.store(mul_res)
    cute.print_tensor(res)        # prints [2.000000, 2.000000, 2.000000]

    div_res = a_vec / b_vec
    res.store(div_res)
    cute.print_tensor(res)        # prints [0.500000, 0.500000, 0.500000]

    floor_div_res = a_vec // b_vec
    res.store(floor_div_res)
    cute.print_tensor(res)        # prints [0.000000, 0.000000, 0.000000]

    mod_res = a_vec % b_vec
    res.store(mod_res)
    cute.print_tensor(res)        # prints [1.000000, 1.000000, 1.000000]


a = np.empty((3,), dtype=np.float32)
a.fill(1.0)
b = np.empty((3,), dtype=np.float32)
b.fill(2.0)
res = np.empty((3,), dtype=np.float32)
binary_op_1(from_dlpack(res), from_dlpack(a), from_dlpack(b))


# %%
@cute.jit
def binary_op_2(res: cute.Tensor, a: cute.Tensor, c: cutlass.Constexpr):
    a_vec = a.load()

    add_res = a_vec + c
    res.store(add_res)
    cute.print_tensor(res)        # prints [3.000000, 3.000000, 3.000000]

    sub_res = a_vec - c
    res.store(sub_res)
    cute.print_tensor(res)        # prints [-1.000000, -1.000000, -1.000000]

    mul_res = a_vec * c
    res.store(mul_res)
    cute.print_tensor(res)        # prints [2.000000, 2.000000, 2.000000]

    div_res = a_vec / c
    res.store(div_res)
    cute.print_tensor(res)        # prints [0.500000, 0.500000, 0.500000]

    floor_div_res = a_vec // c
    res.store(floor_div_res)
    cute.print_tensor(res)        # prints [0.000000, 0.000000, 0.000000]

    mod_res = a_vec % c
    res.store(mod_res)
    cute.print_tensor(res)        # prints [1.000000, 1.000000, 1.000000]

a = np.empty((3,), dtype=np.float32)
a.fill(1.0)
c = 2.0
res = np.empty((3,), dtype=np.float32)
binary_op_2(from_dlpack(res), from_dlpack(a), c)


# %%
@cute.jit
def binary_op_3(res: cute.Tensor, a: cute.Tensor, b: cute.Tensor):
    a_vec = a.load()
    b_vec = b.load()

    gt_res = a_vec > b_vec
    res.store(gt_res)

    """
    ge_res = a_ >= b_   # [False, True, False]
    lt_res = a_ < b_    # [True, False, True]
    le_res = a_ <= b_   # [True, False, True]
    eq_res = a_ == b_   # [False, False, False]
    """

a = np.array([1, 2, 3], dtype=np.float32)
b = np.array([2, 1, 4], dtype=np.float32)
res = np.empty((3,), dtype=np.bool_)
binary_op_3(from_dlpack(res), from_dlpack(a), from_dlpack(b))
print(res)     # prints [False, True, False]


# %%
@cute.jit
def binary_op_4(res: cute.Tensor, a: cute.Tensor, b: cute.Tensor):
    a_vec = a.load()
    b_vec = b.load()

    xor_res = a_vec ^ b_vec
    res.store(xor_res)

    # or_res = a_vec | b_vec
    # res.store(or_res)     # prints [3, 2, 7]

    # and_res = a_vec & b_vec
    # res.store(and_res)      # prints [0, 2, 0]

a = np.array([1, 2, 3], dtype=np.int32)
b = np.array([2, 2, 4], dtype=np.int32)
res = np.empty((3,), dtype=np.int32)
binary_op_4(from_dlpack(res), from_dlpack(a), from_dlpack(b))
print(res)     # prints [3, 0, 7]


# %% [markdown]
# #### Unary Operations

# %%
@cute.jit
def unary_op_1(res: cute.Tensor, a: cute.Tensor):
    a_vec = a.load()

    sqrt_res = cute.math.sqrt(a_vec)
    res.store(sqrt_res)
    cute.print_tensor(res)        # prints [2.000000, 2.000000, 2.000000]

    sin_res = cute.math.sin(a_vec)
    res.store(sin_res)
    cute.print_tensor(res)        # prints [-0.756802, -0.756802, -0.756802]

    exp2_res = cute.math.exp2(a_vec)
    res.store(exp2_res)
    cute.print_tensor(res)        # prints [16.000000, 16.000000, 16.000000]

a = np.array([4.0, 4.0, 4.0], dtype=np.float32)
res = np.empty((3,), dtype=np.float32)
unary_op_1(from_dlpack(res), from_dlpack(a))


# %% [markdown]
# #### Reduction Operation
#
# The `TensorSSA`'s `reduce` method applies a specified reduction operation (`ReductionOp.ADD`, `ReductionOp.MUL`, `ReductionOp.MAX`, `ReductionOp.MIN`) starting with an initial value, and performs this reduction along the dimensions specified by the `reduction_profile.`. The result is typically a new `TensorSSA` with reduced dimensions or a scalar value if reduces across all axes.

# %%
@cute.jit
def reduction_op(a: cute.Tensor):
    """
    Apply reduction operation on the src tensor.

    :param src: The source tensor to be reduced.
    """
    a_vec = a.load()
    red_res = a_vec.reduce(
        cute.ReductionOp.ADD,
        0.0,
        reduction_profile=0
    )
    cute.printf(red_res)        # prints 21.000000

    red_res = a_vec.reduce(
        cute.ReductionOp.ADD,
        0.0,
        reduction_profile=(None, 1)
    )
    # We can't print the TensorSSA directly at this point, so we store it to a new Tensor and print it.
    res = cute.make_fragment(red_res.shape, cutlass.Float32)
    res.store(red_res)
    cute.print_tensor(res)        # prints [6.000000, 15.000000]

    red_res = a_vec.reduce(
        cute.ReductionOp.ADD,
        1.0,
        reduction_profile=(1, None)
    )
    res = cute.make_fragment(red_res.shape, cutlass.Float32)
    res.store(red_res)
    cute.print_tensor(res)        # prints [6.000000, 8.000000, 10.000000]


a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
reduction_op(from_dlpack(a))
