# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: pythondsl_venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Using CuTe Layout Algebra With Python DSL
#
# Referencing the [01_layout.md](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/01_layout.md) and [02_layout_algebra.md](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/02_layout_algebra.md) documentation from CuTe C++, we summarize:
#
# In CuTe, a `Layout`:
# - is defined by a pair of `Shape` and `Stride`,
# - maps coordinates space(s) to an index space,
# - supports both static (compile-time) and dynamic (runtime) values.
#
# CuTe also provides a powerful set of operations—the *Layout Algebra*—for combining and manipulating layouts, including:
# - Layout composition: Functional composition of layouts,
# - Layout "divide": Splitting a layout into two component layouts,
# - Layout "product": Reproducing a layout according to another layout.
#
# In this notebook, we will demonstrate:
# 1. How to use CuTe’s key layout algebra operations with the Python DSL.
# 2. How static and dynamic layouts behave when printed or manipulated within the Python DSL.
#
# We use examples from [02_layout_algebra.md](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/02_layout_algebra.md) which we recommend to the reader for additional details.

# %%
import cutlass
import cutlass.cute as cute


# %% [markdown]
# ## Layout Algebra Operations
#
# These operations form the foundation of CuTe's layout manipulation capabilities, enabling:
# - Efficient data tiling and partitioning,
# - Separation of thread and data layouts with a canonical type to represent both,
# - Native description and manipulation of hierarchical tensors of threads and data crucial for tensor core programs,
# - Mixed static/dynamic layout transformations,
# - Seamless integration of layout algebra with tensor operations,
# - Expression of complex MMA and copies as canonical loops.

# %% [markdown]
# ### 1. Coalesce
#
# The `coalesce` operation simplifies a layout by flattening and combining modes when possible, without changing its size or behavior as a function on the integers.
#
# It ensures the post-conditions:
# - Preserve size: cute.size(layout) == cute.size(result),
# - Flattened: depth(result) <= 1,
# - Preserve functional: For all i, 0 <= i < cute.size(layout), layout(i) == result(i).
#
# #### Examples
#
# - Basic Coalesce Example :

# %%
@cute.jit
def coalesce_example():
    """
    Demonstrates coalesce operation flattening and combining modes
    """
    layout = cute.make_layout((2, (1, 6)), stride=(1, (cutlass.Int32(6), 2))) # Dynamic stride
    result = cute.coalesce(layout)

    print(">>> Original:", layout)
    cute.printf(">?? Original: {}", layout)
    print(">>> Coalesced:", result)
    cute.printf(">?? Coalesced: {}", result)

coalesce_example()


# %%
@cute.jit
def coalesce_post_conditions():
    """
    Demonstrates coalesce operation's 3 post-conditions:
    1. size(@a result) == size(@a layout)
    2. depth(@a result) <= 1
    3. for all i, 0 <= i < size(@a layout), @a result(i) == @a layout(i)
    """
    layout = cute.make_layout(
        ((2, (3, 4)), (3, 2), 1),
        stride=((4, (8, 24)), (2, 6), 12)
    )
    result = cute.coalesce(layout)

    print(">>> Original:", layout)
    print(">>> Coalesced:", result)

    print(">>> Checking post-conditions:")
    print(">>> 1. Checking size remains the same after the coalesce operation:")
    original_size = cute.size(layout)
    coalesced_size = cute.size(result)
    print(f"Original size: {original_size}, Coalesced size: {coalesced_size}")
    assert coalesced_size == original_size, \
            f"Size mismatch: original {original_size}, coalesced {coalesced_size}"
    
    print(">>> 2. Checking depth of coalesced layout <= 1:")
    depth = cute.depth(result)
    print(f"Depth of coalesced layout: {depth}")
    assert depth <= 1, f"Depth of coalesced layout should be <= 1, got {depth}"

    print(">>> 3. Checking layout functionality remains the same after the coalesce operation:")
    for i in range(original_size):
        original_value = layout(i)
        coalesced_value = result(i)
        print(f"Index {i}: original {original_value}, coalesced {coalesced_value}")
        assert coalesced_value == original_value, \
            f"Value mismatch at index {i}: original {original_value}, coalesced {coalesced_value}"

coalesce_post_conditions()


# %% [markdown]
# - By-mode Coalesce Example :

# %%
@cute.jit
def bymode_coalesce_example():
    """
    Demonstrates by-mode coalescing
    """
    layout = cute.make_layout((2, (1, 6)), stride=(1, (6, 2)))

    # Coalesce with mode-wise profile (1,1) = coalesce both modes
    result = cute.coalesce(layout, target_profile=(1, 1))
    
    # Print results
    print(">>> Original: ", layout)
    print(">>> Coalesced Result: ", result)

bymode_coalesce_example()


# %% [markdown]
# ### 2. Composition
#
# `Composition` of Layout `A` with Layout `B` creates a new layout `R = A ◦ B` where:
# - The shape of `B` is compatible with the shape of `R` so that all coordinates of `B` can also be used as coordinates of `R`,
# - `R(c) = A(B(c))` for all coordinates `c` in `B`'s domain.
#
# Layout composition is very useful for reshaping and reordering layouts.
#
# #### Examples
#
# - Basic Composition Example :

# %%
@cute.jit
def composition_example():
    """
    Demonstrates basic layout composition R = A ◦ B
    """
    A = cute.make_layout((6, 2), stride=(cutlass.Int32(8), 2)) # Dynamic stride
    B = cute.make_layout((4, 3), stride=(3, 1))
    R = cute.composition(A, B)

    # Print static and dynamic information
    print(">>> Layout A:", A)
    cute.printf(">?? Layout A: {}", A)
    print(">>> Layout B:", B) 
    cute.printf(">?? Layout B: {}", B)
    print(">>> Composition R = A ◦ B:", R)
    cute.printf(">?? Composition R: {}", R)

composition_example()


# %% [markdown]
# - Comparing Composition with static and dynamic layouts :
#
# In this case, the results may look different but are mathematically the same. The 1s in the shape don't affect the layout as a mathematical function on the integers. In the dynamic case, CuTe can not coalesce the dynamic size-1 modes to "simplify" the layout because it is not valid to do so for all possible dynamic values that parameter could realize at runtime.

# %%
@cute.jit
def composition_static_vs_dynamic_layout():
    """
    Shows difference between static and dynamic composition results
    """
    # Static version - using compile-time values
    A_static = cute.make_layout(
        (10, 2), 
        stride=(16, 4)
    )
    B_static = cute.make_layout(
        (5, 4), 
        stride=(1, 5)
    )
    R_static = cute.composition(A_static, B_static)

    # Static print shows compile-time info
    print(">>> Static composition:")
    print(">>> A_static: ", A_static)
    print(">>> B_static: ", B_static)
    print(">>> R_static: ", R_static)

    # Dynamic version - using runtime Int32 values
    A_dynamic = cute.make_layout(
        (cutlass.Int32(10), cutlass.Int32(2)),
        stride=(cutlass.Int32(16), cutlass.Int32(4))
    )
    B_dynamic = cute.make_layout(
        (cutlass.Int32(5), cutlass.Int32(4)),
        stride=(cutlass.Int32(1), cutlass.Int32(5))
    )
    R_dynamic = cute.composition(A_dynamic, B_dynamic)
    
    # Dynamic printf shows runtime values
    cute.printf(">?? Dynamic composition:")
    cute.printf(">?? A_dynamic: {}", A_dynamic)
    cute.printf(">?? B_dynamic: {}", B_dynamic)
    cute.printf(">?? R_dynamic: {}", R_dynamic)

composition_static_vs_dynamic_layout()


# %% [markdown]
# -  By-mode Composition Example :
#
# By-mode composition allows us to apply composition operations to individual modes of a layout. This is particularly useful when you want to manipulate specific modes layout independently (e.g. rows and columns).
#
# In the context of CuTe, by-mode composition is achieved by using a `Tiler`, which can be a layout or a tuple of layouts. The leaves of the `Tiler` tuple specify how the corresponding mode of the target layout should be composed, allowing for sublayouts to be treated independently.

# %%
@cute.jit
def bymode_composition_example():
    """
    Demonstrates by-mode composition using a tiler
    """
    # Define the original layout A
    A = cute.make_layout(
        (cutlass.Int32(12), (cutlass.Int32(4), cutlass.Int32(8))), 
        stride=(cutlass.Int32(59), (cutlass.Int32(13), cutlass.Int32(1)))
    )

    # Define the tiler for by-mode composition
    tiler = (3, 8) # Apply 3:1 to mode-0 and 8:1 to mode-1

    # Apply by-mode composition
    result = cute.composition(A, tiler)

    # Print static and dynamic information
    print(">>> Layout A:", A)
    cute.printf(">?? Layout A: {}", A)
    print(">>> Tiler:", tiler)
    cute.printf(">?? Tiler: {}", tiler)
    print(">>> By-mode Composition Result:", result)
    cute.printf(">?? By-mode Composition Result: {}", result)

bymode_composition_example()


# %% [markdown]
# ### 3. Division (Splitting into Tiles)
#
# The Division operation in CuTe is used to split a layout into tiles, which is particularly useful for partitioning data across threads or memory hierarchies.
#
# #### Examples :
#
# - Logical divide :
#
# When applied to two Layouts, `logical_divide` splits a layout into two modes -- the first mode contains the elements pointed to by the tiler, and the second mode contains the remaining elements.

# %%
@cute.jit
def logical_divide_1d_example():
    """
    Demonstrates 1D logical divide
    """
    # Define the original layout
    layout = cute.make_layout((4, 2, 3), stride=(2, 1, 8))  # (4,2,3):(2,1,8)
    
    # Define the tiler
    tiler = cute.make_layout(4, stride=2)  # Apply to layout 4:2
    
    # Apply logical divide
    result = cute.logical_divide(layout, tiler=tiler)
    
    # Print results
    print(">>> Layout:", layout)
    print(">>> Tiler :", tiler)
    print(">>> Logical Divide Result:", result)
    cute.printf(">?? Logical Divide Result: {}", result)

logical_divide_1d_example()


# %% [markdown]
# When applied to a Layout and a `Tiler` tuple, `logical_divide` applies itself to the leaves of the `Tiler`and the corresponding mode of the target Layout. This means that the sublayouts are split independently according to the layouts within the `Tiler`.

# %%
@cute.jit
def logical_divide_2d_example():
    """
    Demonstrates 2D logical divide :
    Layout Shape : (M, N, L, ...)
    Tiler Shape  : <TileM, TileN>
    Result Shape : ((TileM,RestM), (TileN,RestN), L, ...)
    """
    # Define the original layout
    layout = cute.make_layout((9, (4, 8)), stride=(59, (13, 1)))  # (9,(4,8)):(59,(13,1))
    
    # Define the tiler
    tiler = (cute.make_layout(3, stride=3),            # Apply to mode-0 layout 3:3
             cute.make_layout((2, 4), stride=(1, 8)))  # Apply to mode-1 layout (2,4):(1,8)
    
    # Apply logical divide
    result = cute.logical_divide(layout, tiler=tiler)
    
    # Print results
    print(">>> Layout:", layout)
    print(">>> Tiler :", tiler)
    print(">>> Logical Divide Result:", result)
    cute.printf(">?? Logical Divide Result: {}", result)

logical_divide_2d_example()


# %% [markdown]
# Zipped, tiled, and flat divide are flavors of `logical_divide` that potentially rearrange modes into more convenient forms.
#
# - Zipped Divide :

# %%
@cute.jit
def zipped_divide_example():
    """
    Demonstrates zipped divide :
    Layout Shape : (M, N, L, ...)
    Tiler Shape  : <TileM, TileN>
    Result Shape : ((TileM,TileN), (RestM,RestN,L,...))
    """
    # Define the original layout
    layout = cute.make_layout((9, (4, 8)), stride=(59, (13, 1)))  # (9,(4,8)):(59,(13,1))
    
    # Define the tiler
    tiler = (cute.make_layout(3, stride=3),            # Apply to mode-0 layout 3:3
             cute.make_layout((2, 4), stride=(1, 8)))  # Apply to mode-1 layout (2,4):(1,8)
    
    # Apply zipped divide
    result = cute.zipped_divide(layout, tiler=tiler)
    
    # Print results
    print(">>> Layout:", layout)
    print(">>> Tiler :", tiler)
    print(">>> Zipped Divide Result:", result)
    cute.printf(">?? Zipped Divide Result: {}", result)

zipped_divide_example()


# %% [markdown]
# - Tiled Divide :

# %%
@cute.jit
def tiled_divide_example():
    """
    Demonstrates tiled divide :
    Layout Shape : (M, N, L, ...)
    Tiler Shape  : <TileM, TileN>
    Result Shape : ((TileM,TileN), RestM, RestN, L, ...)
    """
    # Define the original layout
    layout = cute.make_layout((9, (4, 8)), stride=(59, (13, 1)))  # (9,(4,8)):(59,(13,1))
    
    # Define the tiler
    tiler = (cute.make_layout(3, stride=3),            # Apply to mode-0 layout 3:3
             cute.make_layout((2, 4), stride=(1, 8)))  # Apply to mode-1 layout (2,4):(1,8)
    
    # Apply tiled divide
    result = cute.tiled_divide(layout, tiler=tiler)
    
    # Print results
    print(">>> Layout:", layout)
    print(">>> Tiler :", tiler)
    print(">>> Tiled Divide Result:", result)
    cute.printf(">?? Tiled Divide Result: {}", result)

tiled_divide_example()


# %% [markdown]
# - Flat Divide :

# %%
@cute.jit
def flat_divide_example():
    """
    Demonstrates flat divide :
    Layout Shape : (M, N, L, ...)
    Tiler Shape  : <TileM, TileN>
    Result Shape : (TileM, TileN, RestM, RestN, L, ...)
    """
    # Define the original layout
    layout = cute.make_layout((9, (4, 8)), stride=(59, (13, 1)))  # (9,(4,8)):(59,(13,1))
    
    # Define the tiler
    tiler = (cute.make_layout(3, stride=3),            # Apply to mode-0 layout 3:3
             cute.make_layout((2, 4), stride=(1, 8)))  # Apply to mode-1 layout (2,4):(1,8)
    
    # Apply flat divide
    result = cute.flat_divide(layout, tiler=tiler)
    
    # Print results
    print(">>> Layout:", layout)
    print(">>> Tiler :", tiler)
    print(">>> Flat Divide Result:", result)
    cute.printf(">?? Flat Divide Result: {}", result)

flat_divide_example()


# %% [markdown]
# ### 4. Product (Reproducing a Tile)
#
# The Product operation in CuTe is used to reproduce one layout according to another layout. It creates a new layout where:
# - The first mode is the original layout A.
# - The second mode is a restrided layout B that points to the origin of a "unique replication" of A.
#
# This is particularly useful for repeating layouts of threads across a tile of data for creating "repeat" patterns.
#
# #### Examples
#
# - Logical Product :

# %%
@cute.jit
def logical_product_1d_example():
    """
    Demonstrates 1D logical product
    """
    # Define the original layout
    layout = cute.make_layout((2, 2), stride=(4, 1))  # (2,2):(4,1)
    
    # Define the tiler
    tiler = cute.make_layout(6, stride=1)  # Apply to layout 6:1
    
    # Apply logical product
    result = cute.logical_product(layout, tiler=tiler)
    
    # Print results
    print(">>> Layout:", layout)
    print(">>> Tiler :", tiler)
    print(">>> Logical Product Result:", result)
    cute.printf(">?? Logical Product Result: {}", result)

logical_product_1d_example()


# %% [markdown]
# - Blocked and Raked Product :
#   
#   - Blocked Product: Combines the modes of A and B in a block-like fashion, preserving the semantic meaning of the modes by reassociating them after the product.
#   - Raked Product: Combines the modes of A and B in an interleaved or "raked" fashion, creating a cyclic distribution of the tiles.

# %%
@cute.jit
def blocked_raked_product_example():
    """
    Demonstrates blocked and raked products
    """
    # Define the original layout
    layout = cute.make_layout((2, 5), stride=(5, 1))
    
    # Define the tiler
    tiler = cute.make_layout((3, 4), stride=(1, 3))
    
    # Apply blocked product
    blocked_result = cute.blocked_product(layout, tiler=tiler)

    # Apply raked product
    raked_result = cute.raked_product(layout, tiler=tiler)
    
    # Print results
    print(">>> Layout:", layout)
    print(">>> Tiler :", tiler)
    print(">>> Blocked Product Result:", blocked_result)
    print(">>> Raked Product Result:", raked_result)
    cute.printf(">?? Blocked Product Result: {}", blocked_result)
    cute.printf(">?? Raked Product Result: {}", raked_result)

blocked_raked_product_example()


# %% [markdown]
# - Zipped, tiled, and flat product :
#   
#   - Similar to divide operations, zipped, tiled, and flat product are flavors of `logical_product` that potentially rearrange modes into more convenient forms.

# %%
@cute.jit
def zipped_tiled_flat_product_example():
    """
    Demonstrates zipped, tiled, and flat products
    Layout Shape : (M, N, L, ...)
    Tiler Shape  : <TileM, TileN>

    zipped_product  : ((M,N), (TileM,TileN,L,...))
    tiled_product   : ((M,N), TileM, TileN, L, ...)
    flat_product    : (M, N, TileM, TileN, L, ...)
    """
    # Define the original layout
    layout = cute.make_layout((2, 5), stride=(5, 1))
    
    # Define the tiler
    tiler = cute.make_layout((3, 4), stride=(1, 3))

    # Apply zipped product
    zipped_result = cute.zipped_product(layout, tiler=tiler)
    
    # Apply tiled product
    tiled_result = cute.tiled_product(layout, tiler=tiler)
    
    # Apply flat product
    flat_result = cute.flat_product(layout, tiler=tiler)

    # Print results
    print(">>> Layout:", layout)
    print(">>> Tiler :", tiler)
    print(">>> Zipped Product Result:", zipped_result)
    print(">>> Tiled Product Result:", tiled_result)
    print(">>> Flat Product Result:", flat_result)
    cute.printf(">?? Zipped Product Result: {}", zipped_result)
    cute.printf(">?? Tiled Product Result: {}", tiled_result)
    cute.printf(">?? Flat Product Result: {}", flat_result)

zipped_tiled_flat_product_example()
