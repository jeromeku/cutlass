# ruff: noqa
from utils.logging import patch_cutlass_env

patch_cutlass_env(disable_cache=True, keep_ptx=True)
from cutlass.base_dsl.compiler import CompileCallable, KeepPTX, DumpDir, GenerateLineInfo
import os
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.runtime import _Tensor
import numpy as np
from cutlass.cutlass_dsl.cutlass import KernelLauncher
from cutlass.cutlass_dsl.cutlass import CuTeDSL
from cutlass.base_dsl.dsl import BaseDSL
from cutlass.cutlass_dsl.cuda_jit_executor import CudaDialectJitCompiledFunction
LaunchConfig = BaseDSL.LaunchConfig

@cute.kernel
def load_and_store(res: cute.Tensor, a: cute.Tensor, b: cute.Tensor):
    """
    Load data from memory and store the result to memory.

    :param res: The destination tensor to store the result.
    :param a: The source tensor to be loaded.
    :param b: The source tensor to be loaded.
    """
    a_vec = a.load()
    print(f"a_vec: {a_vec}")  # prints `a_vec: vector<12xf32> o (3, 4)`
    b_vec = b.load()
    print(f"b_vec: {b_vec}")  # prints `b_vec: vector<12xf32> o (3, 4)`
    res.store(a_vec + b_vec)
    cute.print_tensor(res)

@cute.kernel
def broadcast(res: cute.Tensor, a: cute.Tensor, c: cutlass.Constexpr):
    a_vec = a.load()
    print(f"a_vec: {a_vec}")  # prints `a_vec: vector<12xf32> o (3, 4)`
    print(f"c = {c}")  # prints `b_vec: vector<12xf32> o (3, 4)`
    res.store(a_vec + c)
    cute.print_tensor(res)

@cute.kernel
def unary_op_1(res: cute.Tensor, a: cute.Tensor):
    a_vec = a.load()

    # sqrt_res = cute.math.sqrt(a_vec)
    rqrt = cute.math.rsqrt(a_vec)
    # exp2_res = cute.math.exp2(a_vec)
    # cute.print_tensor(exp2_res)  # prints [16.000000, 16.000000, 16.000000]
    res.store(rqrt)


@cute.jit
def run_broadcast(res: cute.Tensor, a: cute.Tensor, c: cutlass.Constexpr):
    launcher: KernelLauncher = broadcast(res, a, c)
    launcher.launch(grid=[1, 1, 1], block=[1, 1, 1])

@cute.jit
def run_load_store(res: cute.Tensor, a: cute.Tensor, b: cute.Tensor):
    launcher: KernelLauncher = load_and_store(res, a, b)
    launcher.launch(grid=[1, 1, 1], block=[1, 1, 1])

@cute.jit
def run_unary(res: cute.Tensor, a: cute.Tensor):
    launcher: KernelLauncher = unary_op_1(res, a)
    launcher.launch(grid=[1, 1, 1], block=[1, 1, 1])

@cute.kernel
def reduction_op(a: cute.Tensor):
    """
    Apply reduction operation on the src tensor.

    :param src: The source tensor to be reduced.
    """
    a_vec = a.load()
    red_res = a_vec.reduce(cute.ReductionOp.ADD, 0.0, reduction_profile=0)
    cute.printf(red_res)  # prints 21.000000

    # red_res = a_vec.reduce(cute.ReductionOp.ADD, 0.0, reduction_profile=(None, 1))
    # cute.print_tensor(red_res)  # prints [6.000000, 15.000000]

    # red_res = a_vec.reduce(cute.ReductionOp.ADD, 1.0, reduction_profile=(1, None))
    # cute.print_tensor(red_res)  # prints [6.000000, 8.000000, 10.000000]
@cute.jit
def run_reduction(a: cute.Tensor):
    launcher: KernelLauncher = reduction_op(a)
    launcher.launch(grid=[1, 1, 1], block=[1, 1, 1])



@cute.kernel
def rmem_tensor():
    a = cute.make_rmem_tensor((1, 3), dtype=cutlass.Float32)
    a[0] = 0.0
    a[1] = 1.0
    a[2] = 2.0
    a_val = a.load()
    cute.print_tensor(a_val.broadcast_to((4, 3)))
    # # tensor(raw_ptr(0x00007ffe26625740: f32, rmem, align<32>) o (4,3):(1,4), data=
    # #    [[ 0.000000,  1.000000,  2.000000, ],
    # #     [ 0.000000,  1.000000,  2.000000, ],
    # #     [ 0.000000,  1.000000,  2.000000, ],
    # #     [ 0.000000,  1.000000,  2.000000, ]])

    # c = cute.make_rmem_tensor((4, 1), dtype=cutlass.Float32)
    # c[0] = 0.0
    # c[1] = 1.0
    # c[2] = 2.0
    # c[3] = 3.0
    # cute.print_tensor(a.load() + c.load())
    # tensor(raw_ptr(0x00007ffe26625780: f32, rmem, align<32>) o (4,3):(1,4), data=
    #        [[ 0.000000,  1.000000,  2.000000, ],
    #         [ 1.000000,  2.000000,  3.000000, ],
    #         [ 2.000000,  3.000000,  4.000000, ],
    #         [ 3.000000,  4.000000,  5.000000, ]])
@cute.jit
def run_rmem():
    launcher: KernelLauncher = rmem_tensor()
    launcher.launch(grid=[1, 1, 1], block=[1, 1, 1])
arr = np.arange(3, dtype=np.float32)
for i in range(len(arr)):
    print(f"{i}: {arr[i].view(np.uint32)}")
cute.compile(run_rmem)

# a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
# cute.compile(run_reduction, from_dlpack(a))

# reduction_op(from_dlpack(a))

# a_np = np.ones(12).reshape((3, 4)).astype(np.float16)
# b_np = np.ones(12).reshape((3, 4)).astype(np.float16)
# c_np = np.zeros(12).reshape((3, 4)).astype(np.float16)

# compiler: CompileCallable = cute.compile

# a, b, c = from_dlpack(a_np), from_dlpack(b_np), from_dlpack(c_np)

# kernel = broadcast
# constant = 1
# kernel_args = (c, a, cutlass.Float16(constant))
# args = (kernel, *kernel_args)
# #dumpdir="ssa_dump",
# # #with dump_mlir_pipeline(dump_dir=dump_dir, module_name="load_store"):

# unary_a = np.array([4.0, 4.0, 4.0], dtype=np.float32)
# res = np.empty((len(unary_a),), dtype=np.float32)
# compiled_fn: CudaDialectJitCompiledFunction = compiler(run_unary, from_dlpack(res), from_dlpack(unary_a))


# # # 1. Create an array of two float16s
# floats = np.array([constant, constant], dtype=np.float16)

# # 2. View the underlying memory as a single uint32
# # This works because two 16-bit slots = one 32-bit slot
# packed_uint32 = floats.view(np.uint32)[0]

# print(f"Packed uint32: {packed_uint32}")

# print(compiled_fn.artifacts.PTX)
# # %% [markdown]
# # ### Register-Level Tensor Operations
# #
# # When writing kernel logic, various computations, transformations, slicing, etc., are performed on data loaded into registers.

# # %%
# @cute.jit
# def apply_slice(src: cute.Tensor, dst: cute.Tensor, indices: cutlass.Constexpr):
#     """
#     Apply slice operation on the src tensor and store the result to the dst tensor.

#     :param src: The source tensor to be sliced.
#     :param dst: The destination tensor to store the result.
#     :param indices: The indices to slice the source tensor.
#     """
#     src_vec = src.load()
#     dst_vec = src_vec[indices]
#     print(f"{src_vec} -> {dst_vec}")
#     if cutlass.const_expr(isinstance(dst_vec, cute.TensorSSA)):
#         dst.store(dst_vec)
#         cute.print_tensor(dst)
#     else:
#         dst[0] = dst_vec
#         cute.print_tensor(dst)


# def slice_1():
#     src_shape = (4, 2, 3)
#     dst_shape = (4, 3)
#     indices = (None, 1, None)

#     """
#     a:
#     [[[ 0.  1.  2.]
#       [ 3.  4.  5.]]

#      [[ 6.  7.  8.]
#       [ 9. 10. 11.]]

#      [[12. 13. 14.]
#       [15. 16. 17.]]

#      [[18. 19. 20.]
#       [21. 22. 23.]]]
#     """
#     a = np.arange(np.prod(src_shape)).reshape(*src_shape).astype(np.float32)
#     dst = np.random.randn(*dst_shape).astype(np.float32)
#     apply_slice(from_dlpack(a), from_dlpack(dst), indices)


# slice_1()

# # %%
# def slice_2():
#     src_shape = (4, 2, 3)
#     dst_shape = (1,)
#     indices = 10
#     a = np.arange(np.prod(src_shape)).reshape(*src_shape).astype(np.float32)
#     dst = np.random.randn(*dst_shape).astype(np.float32)
#     apply_slice(from_dlpack(a), from_dlpack(dst), indices)


# slice_2()

# # %% [markdown]
# # ## Arithmetic Operations
# #
# # As we mentioned earlier, there're many tensor operations whose operands are `TensorSSA`. And they are all element-wise operations. We give some examples below.
# #
# # ### Binary Operations
# #
# # For binary operations, the LHS operand is `TensorSSA` and the RHS operand can be either `TensorSSA` or `Numeric`. When the RHS is `Numeric`, it will be broadcast to a `TensorSSA`.

# # %%
# @cute.jit
# def binary_op_1(res: cute.Tensor, a: cute.Tensor, b: cute.Tensor):
#     a_vec = a.load()
#     b_vec = b.load()

#     add_res = a_vec + b_vec
#     cute.print_tensor(add_res)  # prints [3.000000, 3.000000, 3.000000]

#     sub_res = a_vec - b_vec
#     cute.print_tensor(sub_res)  # prints [-1.000000, -1.000000, -1.000000]

#     mul_res = a_vec * b_vec
#     cute.print_tensor(mul_res)  # prints [2.000000, 2.000000, 2.000000]

#     div_res = a_vec / b_vec
#     cute.print_tensor(div_res)  # prints [0.500000, 0.500000, 0.500000]

#     floor_div_res = a_vec // b_vec
#     cute.print_tensor(res)  # prints [0.000000, 0.000000, 0.000000]

#     mod_res = a_vec % b_vec
#     cute.print_tensor(mod_res)  # prints [1.000000, 1.000000, 1.000000]


# a = np.empty((3,), dtype=np.float32)
# a.fill(1.0)
# b = np.empty((3,), dtype=np.float32)
# b.fill(2.0)
# res = np.empty((3,), dtype=np.float32)
# binary_op_1(from_dlpack(res), from_dlpack(a), from_dlpack(b))

# # %%
# @cute.jit
# def binary_op_2(res: cute.Tensor, a: cute.Tensor, c: cutlass.Constexpr):
#     a_vec = a.load()

#     add_res = a_vec + c
#     cute.print_tensor(add_res)  # prints [3.000000, 3.000000, 3.000000]

#     sub_res = a_vec - c
#     cute.print_tensor(sub_res)  # prints [-1.000000, -1.000000, -1.000000]

#     mul_res = a_vec * c
#     cute.print_tensor(mul_res)  # prints [2.000000, 2.000000, 2.000000]

#     div_res = a_vec / c
#     cute.print_tensor(div_res)  # prints [0.500000, 0.500000, 0.500000]

#     floor_div_res = a_vec // c
#     cute.print_tensor(floor_div_res)  # prints [0.000000, 0.000000, 0.000000]

#     mod_res = a_vec % c
#     cute.print_tensor(mod_res)  # prints [1.000000, 1.000000, 1.000000]


# a = np.empty((3,), dtype=np.float32)
# a.fill(1.0)
# c = 2.0
# res = np.empty((3,), dtype=np.float32)
# binary_op_2(from_dlpack(res), from_dlpack(a), c)

# # %%
# @cute.jit
# def binary_op_3(res: cute.Tensor, a: cute.Tensor, b: cute.Tensor):
#     a_vec = a.load()
#     b_vec = b.load()

#     gt_res = a_vec > b_vec
#     res.store(gt_res)

#     """
#     ge_res = a_ >= b_   # [False, True, False]
#     lt_res = a_ < b_    # [True, False, True]
#     le_res = a_ <= b_   # [True, False, True]
#     eq_res = a_ == b_   # [False, False, False]
#     """


# a = np.array([1, 2, 3], dtype=np.float32)
# b = np.array([2, 1, 4], dtype=np.float32)
# res = np.empty((3,), dtype=np.bool_)
# binary_op_3(from_dlpack(res), from_dlpack(a), from_dlpack(b))
# print(res)  # prints [False, True, False]

# # %%
# @cute.jit
# def binary_op_4(res: cute.Tensor, a: cute.Tensor, b: cute.Tensor):
#     a_vec = a.load()
#     b_vec = b.load()

#     xor_res = a_vec ^ b_vec
#     res.store(xor_res)

#     # or_res = a_vec | b_vec
#     # res.store(or_res)     # prints [3, 2, 7]

#     # and_res = a_vec & b_vec
#     # res.store(and_res)      # prints [0, 2, 0]


# a = np.array([1, 2, 3], dtype=np.int32)
# b = np.array([2, 2, 4], dtype=np.int32)
# res = np.empty((3,), dtype=np.int32)
# binary_op_4(from_dlpack(res), from_dlpack(a), from_dlpack(b))
# print(res)  # prints [3, 0, 7]

# # %% [markdown]
# # #### Unary Operations

# # %%
# @cute.jit
# def unary_op_1(res: cute.Tensor, a: cute.Tensor):
#     a_vec = a.load()

#     sqrt_res = cute.math.sqrt(a_vec)
#     cute.print_tensor(sqrt_res)  # prints [2.000000, 2.000000, 2.000000]

#     sin_res = cute.math.sin(a_vec)
#     res.store(sin_res)
#     cute.print_tensor(sin_res)  # prints [-0.756802, -0.756802, -0.756802]

#     exp2_res = cute.math.exp2(a_vec)
#     cute.print_tensor(exp2_res)  # prints [16.000000, 16.000000, 16.000000]


# a = np.array([4.0, 4.0, 4.0], dtype=np.float32)
# res = np.empty((3,), dtype=np.float32)
# unary_op_1(from_dlpack(res), from_dlpack(a))

# # %% [markdown]
# # #### Reduction Operation
# #
# # The `TensorSSA`'s `reduce` method applies a specified reduction operation (`ReductionOp.ADD`,
# # `ReductionOp.MUL`, `ReductionOp.MAX`, `ReductionOp.MIN`) starting with an initial value, and
# # performs this reduction along the dimensions specified by the `reduction_profile`. The result
# # is typically a new `TensorSSA` with reduced dimensions or a scalar value if it reduces across
# # all axes.

# # %%
# @cute.jit
# def reduction_op(a: cute.Tensor):
#     """
#     Apply reduction operation on the src tensor.

#     :param src: The source tensor to be reduced.
#     """
#     a_vec = a.load()
#     red_res = a_vec.reduce(cute.ReductionOp.ADD, 0.0, reduction_profile=0)
#     cute.printf(red_res)  # prints 21.000000

#     red_res = a_vec.reduce(cute.ReductionOp.ADD, 0.0, reduction_profile=(None, 1))
#     cute.print_tensor(red_res)  # prints [6.000000, 15.000000]

#     red_res = a_vec.reduce(cute.ReductionOp.ADD, 1.0, reduction_profile=(1, None))
#     cute.print_tensor(red_res)  # prints [6.000000, 8.000000, 10.000000]


# a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
# reduction_op(from_dlpack(a))

# # %% [markdown]
# # ## Broadcast
# #
# # `TensorSSA` supports broadcasting operations following NumPy's broadcasting rules. Broadcasting
# # allows you to perform operations on arrays of different shapes when certain conditions are met.
# # The key rules are:
# #
# # 1. Source shape is padded with 1's to match the rank of target shape
# # 2. The size in each mode of source shape must either be 1 or equal to target shape
# # 3. After broadcasting, all modes should match target shape
# #
# # Let's look at some examples of broadcasting in action:

# # %%
# import cutlass
# import cutlass.cute as cute


# @cute.jit
# def broadcast_examples():
#     a = cute.make_rmem_tensor((1, 3), dtype=cutlass.Float32)
#     a[0] = 0.0
#     a[1] = 1.0
#     a[2] = 2.0
#     a_val = a.load()
#     cute.print_tensor(a_val.broadcast_to((4, 3)))
#     # tensor(raw_ptr(0x00007ffe26625740: f32, rmem, align<32>) o (4,3):(1,4), data=
#     #    [[ 0.000000,  1.000000,  2.000000, ],
#     #     [ 0.000000,  1.000000,  2.000000, ],
#     #     [ 0.000000,  1.000000,  2.000000, ],
#     #     [ 0.000000,  1.000000,  2.000000, ]])

#     c = cute.make_rmem_tensor((4, 1), dtype=cutlass.Float32)
#     c[0] = 0.0
#     c[1] = 1.0
#     c[2] = 2.0
#     c[3] = 3.0
#     cute.print_tensor(a.load() + c.load())
#     # tensor(raw_ptr(0x00007ffe26625780: f32, rmem, align<32>) o (4,3):(1,4), data=
#     #        [[ 0.000000,  1.000000,  2.000000, ],
#     #         [ 1.000000,  2.000000,  3.000000, ],
#     #         [ 2.000000,  3.000000,  4.000000, ],
#     #         [ 3.000000,  4.000000,  5.000000, ]])


# broadcast_examples()

# # %% [markdown]
# # The examples above demonstrate two key broadcasting scenarios:
# #
# # 1. **Row Vector Broadcasting**: In the first example, we create a row vector `a` with shape
# #    (1, 3) containing values [0.0, 1.0, 2.0]. When we broadcast it to shape (4, 3), the values
# #    are repeated across the first dimension, resulting in:
# #    ```
# #    [[0.0, 1.0, 2.0],
# #     [0.0, 1.0, 2.0],
# #     [0.0, 1.0, 2.0],
# #     [0.0, 1.0, 2.0]]
# #    ```
# #    This demonstrates how a row vector can be broadcast to create multiple identical rows.
# #
# # 2. **Column Vector and Row Vector Addition**: In the second example, we have:
# #    - A row vector `a` with shape (1, 3) containing [0.0, 1.0, 2.0]
# #    - A column vector `c` with shape (4, 1) containing [0.0, 1.0, 2.0, 3.0]
# #
# #    When we add these together, both vectors are broadcast to shape (4, 3):
# #    - The row vector is broadcast vertically (4 times)
# #    - The column vector is broadcast horizontally (3 times)
# #
# #    The result is:
# #    ```
# #    [[0.0 + 0.0, 1.0 + 0.0, 2.0 + 0.0],
# #     [0.0 + 1.0, 1.0 + 1.0, 2.0 + 1.0],
# #     [0.0 + 2.0, 1.0 + 2.0, 2.0 + 2.0],
# #     [0.0 + 3.0, 1.0 + 3.0, 2.0 + 3.0]]
# #    ```
# #    =
# #    ```
# #    [[0.0, 1.0, 2.0],
# #     [1.0, 2.0, 3.0],
# #     [2.0, 3.0, 4.0],
# #     [3.0, 4.0, 5.0]]
# #    ```
# #
# # This demonstrates how `TensorSSA` can automatically handle broadcasting of both row and column
# # vectors in arithmetic operations, following the broadcasting rules where each dimension must
# # either be 1 or match the target size. The broadcasting is handled implicitly during operations,
# # making it easy to work with tensors of different shapes.
# #
