
# %%
import cutlass
import cutlass.cute as cute

# %% [markdown]
# ## Tensor
#
# A tensor in CuTe is created through the composition of two key components:
#
# 1. An **Engine** (E) - A random-access, pointer-like object that supports:
#    - Offset operation: `e + d → e` (offset engine by elements of a layout's codomain)
#    - Dereference operation: `*e → v` (dereference engine to produce value)
#
# 2. A **Layout** (L) - Defines the mapping from coordinates to offsets
#
# A tensor is formally defined as the composition of an engine E with a layout L, expressed as `T = E ∘ L`. When evaluating a tensor at coordinate c, it:
#
# 1. Maps the coordinate c to the codomain using the layout
# 2. Offsets the engine accordingly
# 3. Dereferences the result to obtain the tensor's value
#
# This can be expressed mathematically as:
#
# ```
# T(c) = (E ∘ L)(c) = *(E + L(c))
# ```
#
# ## Example Usage
#
# Here's a simple example of creating a tensor using pointer and layout `(8,5):(5,1)` and fill with ones:

# %%
@cute.jit
def create_tensor_from_ptr(ptr: cute.Pointer):
    layout = cute.make_layout((8, 5), stride=(5, 1))
    tensor = cute.make_tensor(ptr, layout)
    tensor.fill(1)
    cute.print_tensor(tensor)


# %% [markdown]
# This creates a tensor where:
# - The engine is a pointer
# - The layout with shape `(8, 5)` and stride `(5, 1)`
# - The resulting tensor can be evaluated using coordinates defined by the layout
#
# We can test this by allocating buffer with torch and run test with pointer to torch tensor

# %%
import cutlass.cute.runtime as cute_rt
import torch
from cutlass.torch import dtype as torch_dtype

a = torch.randn(8, 5, dtype=torch_dtype(cutlass.Float32))
ptr_a = cute_rt.make_ptr(cutlass.Float32, a.data_ptr())

# create_tensor_from_ptr(ptr_a)

# %% [markdown]
# ## DLPACK support 
#
# CuTe DSL is designed to support dlpack protocol natively. This offers easy integration with frameworks 
# supporting DLPack, e.g. torch, numpy, jax, tensorflow, etc.
#
# For more information, please refer to DLPACK project: https://github.com/dmlc/dlpack
#
# Calling `from_dlpack` can convert any tensor or ndarray object supporting `__dlpack__` and `__dlpack_device__`.
#

# %%
from cutlass.cute.runtime import from_dlpack


@cute.jit
def print_tensor_dlpack(src: cute.Tensor):
    print(src)
    cute.print_tensor(src)

@cute.jit
def print_tensor_torch(src):
    print(src)
    cute.print_tensor(src)


# %%
a = torch.randn(8, 5, dtype=torch_dtype(cutlass.Float32))

print_tensor_dlpack(from_dlpack(a))
print_tensor_torch(a)

# %%
import numpy as np

a = np.random.randn(8, 8).astype(np.float32)

print_tensor_dlpack(from_dlpack(a))


# %% [markdown]
# ## Tensor Evaluation Methods
#
# Tensors support two primary methods of evaluation:
#
# ### 1. Full Evaluation
# When applying the tensor evaluation with a complete coordinate c, it computes the offset, applies it to the engine, 
# and dereferences it to return the stored value. This is the straightforward case where you want to access 
# a specific element of the tensor.
#
# ### 2. Partial Evaluation (Slicing)
# When evaluating with an incomplete coordinate c = c' ⊕ c* (where c* represents the unspecified portion), 
# the result is a new tensor which is a slice of the original tensor with its engine offset to account for 
# the coordinates that were provided. This operation can be expressed as:
#
# ```
# T(c) = (E ∘ L)(c) = (E + L(c')) ∘ L(c*) = T'(c*)
# ```
#
# Slicing effectively reduces the dimensionality of the tensor, creating a sub-tensor that can be 
# further evaluated or manipulated.

# %%
@cute.jit
def tensor_access_item(a: cute.Tensor):
    # access data using linear index
    cute.printf("a[2] = {} (equivalent to a[{}])", a[2],
                cute.make_identity_tensor(a.layout.shape)[2])
    cute.printf("a[9] = {} (equivalent to a[{}])", a[9],
                cute.make_identity_tensor(a.layout.shape)[9])

    # access data using n-d coordinates, following two are equivalent
    cute.printf("a[2,0] = {}", a[2, 0])
    cute.printf("a[2,4] = {}", a[2, 4])
    cute.printf("a[(2,4)] = {}", a[2, 4])

    # assign value to tensor@(2,4)
    a[2,3] = 100.0
    a[2,4] = 101.0
    cute.printf("a[2,3] = {}", a[2,3])
    cute.printf("a[(2,4)] = {}", a[(2,4)])

@cute.kernel
def print_tensor_gpu(ptr: cute.Pointer):
    layout = cute.make_layout((8, 5), stride=(5, 1))
    tensor = cute.make_tensor(ptr, layout)

    tidx, _, _ = cute.arch.thread_idx()

    if tidx == 0:
        cute.print_tensor(tensor)


# Create a tensor with sequential data using torch
data = torch.arange(0, 8*5, dtype=torch.float32).reshape(8, 5)
tensor_access_item(from_dlpack(data))

print(data)


# %% [markdown]
# ### Tensor as memory view
#
# In CUDA programming, different memory spaces have different characteristics in terms of access speed, scope, and lifetime:
#
# - **generic**: Default memory space that can refer to any other memory space.
# - **global memory (gmem)**: Accessible by all threads across all blocks, but has higher latency.
# - **shared memory (smem)**: Accessible by all threads within a block, with much lower latency than global memory.
# - **register memory (rmem)**: Thread-private memory with the lowest latency, but limited capacity.
# - **tensor memory (tmem)**: Specialized memory introduced in NVIDIA Blackwell architecture for tensor operations.
#
# When creating tensors in CuTe, you can specify the memory space to optimize performance based on your access patterns.
#
# For more information on CUDA memory spaces, see the [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy).
#

# %% [markdown]
# ### Coordinate Tensor
#
# A coordinate tensor is a special type of tensor that maps coordinates to coordinates rather than to values. 
# The key distinction is that while regular tensors map coordinates to some value type (like numbers), 
# coordinate tensors map coordinates to other coordinates.
#
# For example, given a shape (4,4), a coordinate tensor using row-major layout would appear as:
#
# \begin{bmatrix} 
# (0,0) & (0,1) & (0,2) & (0,3) \\
# (1,0) & (1,1) & (1,2) & (1,3) \\
# (2,0) & (2,1) & (2,2) & (2,3) \\
# (3,0) & (3,1) & (3,2) & (3,3)
# \end{bmatrix}
#
# The same shape with a column-major layout would appear as:
#
# \begin{bmatrix}
# (0,0) & (1,0) & (2,0) & (3,0) \\
# (0,1) & (1,1) & (2,1) & (3,1) \\
# (0,2) & (1,2) & (2,2) & (3,2) \\
# (0,3) & (1,3) & (2,3) & (3,3)
# \end{bmatrix}
#
# The key points about coordinate tensors are:
# - Each element in the tensor is itself a coordinate tuple (i,j) rather than a scalar value
# - The coordinates map to themselves - so position (1,2) contains the coordinate (1,2)
# - The layout (row-major vs column-major) determines how these coordinate tuples are arranged in memory
#
# For example, coordinate tensors can be created using the `make_identity_tensor` utility:
#
# ```python
# coord_tensor = make_identity_tensor(layout.shape())
# ```
#
# This creates a tensor that maps each coordinate to itself, providing a reference point for understanding how other layouts transform these coordinates.

# %%
@cute.jit
def print_tensor_coord(a: cute.Tensor):
    coord_tensor = cute.make_identity_tensor(a.layout.shape)
    print(coord_tensor)

a = torch.randn(8,4, dtype=torch_dtype(cutlass.Float32))
print_tensor_coord(from_dlpack(a))
