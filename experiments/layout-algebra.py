from cutlass._mlir import ir
import cutlass.cute as cute
from cutlass.utils import print_latex
# %%
@cute.jit
def logical_divide_1d_example(tiler: cute.Layout):
    """
    Demonstrates 1D logical divide
    """
    # Define the original layout
    layout = cute.make_layout((4, 2, 3), stride=(2, 1, 8))  # (4,2,3):(2,1,8)

    # Define the tiler
    
    # Apply logical divide
    result: cute.Layout = cute.logical_divide(layout, tiler=tiler)
    
    # Print results
    print(">>> Layout:", layout)
    print(">>> Tiler :", tiler)
    print(">>> Logical Divide Result:", result)
    cute.printf(">?? Logical Divide Result: {}", result)
    


@cute.jit
def run_logical_divide_1d():
    tiler = cute.make_layout(4, stride=2)  # Apply to layout 4:2

    logical_divide_1d_example(tiler)
#run_logical_divide_1d()
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
    layout = cute.make_layout(
        (9, (4, 8)), stride=(59, (13, 1))
    )  # (9,(4,8)):(59,(13,1))

    # Define the tiler
    tiler = (
        cute.make_layout(3, stride=3),  # Apply to mode-0 layout 3:3
        cute.make_layout((2, 4), stride=(1, 8)),
    )  # Apply to mode-1 layout (2,4):(1,8)

    # Apply logical divide
    result = cute.logical_divide(layout, tiler=tiler)

    # Print results
    # print(">>> Layout:", layout)
    # print(">>> Tiler :", tiler)
    # print(">>> Logical Divide Result:", result)
    # cute.printf(">?? Logical Divide Result: {}", result)
    print_latex(result)

logical_divide_2d_example()
#breakpoint()

# # %% [markdown]
# # Zipped, tiled, and flat divide are flavors of `logical_divide` that potentially rearrange modes into more convenient forms.
# # 
# # - Zipped Divide :

# # %%
# @cute.jit
# def zipped_divide_example():
#     """
#     Demonstrates zipped divide :
#     Layout Shape : (M, N, L, ...)
#     Tiler Shape  : <TileM, TileN>
#     Result Shape : ((TileM,TileN), (RestM,RestN,L,...))
#     """
#     # Define the original layout
#     layout = cute.make_layout(
#         (9, (4, 8)), stride=(59, (13, 1))
#     )  # (9,(4,8)):(59,(13,1))

#     # Define the tiler
#     tiler = (
#         cute.make_layout(3, stride=3),  # Apply to mode-0 layout 3:3
#         cute.make_layout((2, 4), stride=(1, 8)),
#     )  # Apply to mode-1 layout (2,4):(1,8)

#     # Apply zipped divide
#     result = cute.zipped_divide(layout, tiler=tiler)

#     # Print results
#     print(">>> Layout:", layout)
#     print(">>> Tiler :", tiler)
#     print(">>> Zipped Divide Result:", result)
#     cute.printf(">?? Zipped Divide Result: {}", result)

# breakpoint()
# zipped_divide_example()

# # %% [markdown]
# # - Tiled Divide :

# # %%
# @cute.jit
# def tiled_divide_example():
#     """
#     Demonstrates tiled divide :
#     Layout Shape : (M, N, L, ...)
#     Tiler Shape  : <TileM, TileN>
#     Result Shape : ((TileM,TileN), RestM, RestN, L, ...)
#     """
#     # Define the original layout
#     layout = cute.make_layout(
#         (9, (4, 8)), stride=(59, (13, 1))
#     )  # (9,(4,8)):(59,(13,1))

#     # Define the tiler
#     tiler = (
#         cute.make_layout(3, stride=3),  # Apply to mode-0 layout 3:3
#         cute.make_layout((2, 4), stride=(1, 8)),
#     )  # Apply to mode-1 layout (2,4):(1,8)

#     # Apply tiled divide
#     result = cute.tiled_divide(layout, tiler=tiler)

#     # Print results
#     print(">>> Layout:", layout)
#     print(">>> Tiler :", tiler)
#     print(">>> Tiled Divide Result:", result)
#     cute.printf(">?? Tiled Divide Result: {}", result)
#     breakpoint()
#     coord = ((0, None), 0, 0)
#     tile = cute.slice_(result, coord)
#     print(f"tiled divide tile: {tile}")
    
# breakpoint()
# tiled_divide_example()

# # %% [markdown]
# # - Flat Divide :

# # %%
# @cute.jit
# def flat_divide_example():
#     """
#     Demonstrates flat divide :
#     Layout Shape : (M, N, L, ...)
#     Tiler Shape  : <TileM, TileN>
#     Result Shape : (TileM, TileN, RestM, RestN, L, ...)
#     """
#     # Define the original layout
#     layout = cute.make_layout(
#         (9, (4, 8)), stride=(59, (13, 1))
#     )  # (9,(4,8)):(59,(13,1))

#     # Define the tiler
#     tiler = (
#         cute.make_layout(3, stride=3),  # Apply to mode-0 layout 3:3
#         cute.make_layout((2, 4), stride=(1, 8)),
#     )  # Apply to mode-1 layout (2,4):(1,8)

#     # Apply flat divide
#     result = cute.flat_divide(layout, tiler=tiler)
#     coord = (0, None, 0, 0)
#     tile1 = cute.slice_(result, coord)

#     # Print results
#     print(">>> Layout:", layout)
#     print(">>> Tiler :", tiler)
#     print(">>> Flat Divide Result:", result)
#     cute.printf(">?? Flat Divide Result: {}", result)
#     print(f"tile1 -> ${tile1}")

# breakpoint()
# flat_divide_example()

# # %% [markdown]
# # ### 4. Product (Reproducing a Tile)
# # 
# # The Product operation in CuTe is used to reproduce one layout according to another layout. It creates a new layout where:
# # - The first mode is the original layout A.
# # - The second mode is a restrided layout B that points to the origin of a "unique replication" of A.
# # 
# # This is particularly useful for repeating layouts of threads across a tile of data for creating "repeat" patterns.
# # 
# # #### Examples
# # 
# # - Logical Product :

# # %%
# @cute.jit
# def logical_product_1d_example():
#     """
#     Demonstrates 1D logical product
#     """
#     # Define the original layout
#     layout = cute.make_layout((2, 2), stride=(4, 1))  # (2,2):(4,1)

#     # Define the tiler
#     tiler = cute.make_layout(6, stride=1)  # Apply to layout 6:1

#     # Apply logical product
#     result = cute.logical_product(layout, tiler=tiler)

#     # Print results
#     print(">>> Layout:", layout)
#     print(">>> Tiler :", tiler)
#     print(">>> Logical Product Result:", result)
#     cute.printf(">?? Logical Product Result: {}", result)

# breakpoint()
# logical_product_1d_example()

# # %% [markdown]
# # - Blocked and Raked Product :
# #   
# #   - Blocked Product: Combines the modes of A and B in a block-like fashion, preserving the semantic meaning of the modes by reassociating them after the product.
# #   - Raked Product: Combines the modes of A and B in an interleaved or "raked" fashion, creating a cyclic distribution of the tiles.

# # %%
# @cute.jit
# def blocked_raked_product_example():
#     """
#     Demonstrates blocked and raked products
#     """
#     # Define the original layout
#     layout = cute.make_layout((2, 5), stride=(5, 1))

#     # Define the tiler
#     tiler = cute.make_layout((3, 4), stride=(1, 3))

#     # Apply blocked product
#     blocked_result = cute.blocked_product(layout, tiler=tiler)

#     # Apply raked product
#     raked_result = cute.raked_product(layout, tiler=tiler)

#     # Print results
#     print(">>> Layout:", layout)
#     print(">>> Tiler :", tiler)
#     print(">>> Blocked Product Result:", blocked_result)
#     print(">>> Raked Product Result:", raked_result)
#     cute.printf(">?? Blocked Product Result: {}", blocked_result)
#     cute.printf(">?? Raked Product Result: {}", raked_result)

# breakpoint()
# blocked_raked_product_example()

# # %% [markdown]
# # - Zipped, tiled, and flat product :
# #   
# #   - Similar to divide operations, zipped, tiled, and flat product are flavors of `logical_product` that potentially rearrange modes into more convenient forms.

# # %%
# @cute.jit
# def zipped_tiled_flat_product_example():
#     """
#     Demonstrates zipped, tiled, and flat products
#     Layout Shape : (M, N, L, ...)
#     Tiler Shape  : <TileM, TileN>

#     zipped_product  : ((M,N), (TileM,TileN,L,...))
#     tiled_product   : ((M,N), TileM, TileN, L, ...)
#     flat_product    : (M, N, TileM, TileN, L, ...)
#     """
#     # Define the original layout
#     layout = cute.make_layout((2, 5), stride=(5, 1))

#     # Define the tiler
#     tiler = cute.make_layout((3, 4), stride=(1, 3))

#     # Apply zipped product
#     zipped_result = cute.zipped_product(layout, tiler=tiler)

#     # Apply tiled product
#     tiled_result = cute.tiled_product(layout, tiler=tiler)

#     # Apply flat product
#     flat_result = cute.flat_product(layout, tiler=tiler)

#     # Print results
#     print(">>> Layout:", layout)
#     print(">>> Tiler :", tiler)
#     print(">>> Zipped Product Result:", zipped_result)
#     print(">>> Tiled Product Result:", tiled_result)
#     print(">>> Flat Product Result:", flat_result)
#     cute.printf(">?? Zipped Product Result: {}", zipped_result)
#     cute.printf(">?? Tiled Product Result: {}", tiled_result)
#     cute.printf(">?? Flat Product Result: {}", flat_result)

# breakpoint()
# zipped_tiled_flat_product_example()


