from functools import partial

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

M, N = 2048, 2048

a = torch.randn(M, N, device="cuda", dtype=torch.float16)
b = torch.randn(M, N, device="cuda", dtype=torch.float16)
c = torch.zeros(M, N, device="cuda", dtype=torch.float16)

a_ = from_dlpack(a, assumed_align=16)
b_ = from_dlpack(b, assumed_align=16)
c_ = from_dlpack(c, assumed_align=16)


# def benchmark(callable, *, num_warmups, num_iterations):
#     start_event = torch.cuda.Event(enable_timing=True)
#     end_event = torch.cuda.Event(enable_timing=True)

#     torch.cuda.synchronize()

#     for _ in range(num_warmups):
#         callable()

#     start_event.record(stream=torch.cuda.current_stream())
#     for _ in range(num_iterations):
#         callable()
#     end_event.record(stream=torch.cuda.current_stream())
#     torch.cuda.synchronize()

#     elapsed_time = start_event.elapsed_time(end_event)
#     avg_time = elapsed_time / num_iterations

#     print(f"Average execution time: {avg_time:.4f} ms")
#     print(f"Throughput: {(3 * a.numel() * 2) / (avg_time / 1000) / 1e9:.2f} GB/s")



@cute.kernel
def elementwise_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    tv_layout: cute.Layout
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    #--------------------------------
    # slice for thread-block level view
    #--------------------------------
    blk_coord = ((None, None), bidx)

    # logical coord -> address
    blkA = gA[blk_coord]  # (TileM, TileN) -> physical address
    blkB = gB[blk_coord]  # (TileM, TileN) -> physical address
    blkC = gC[blk_coord]  # (TileM, TileN) -> physical address

    #--------------------------------
    # compose for thread-index & value-index to physical mapping
    #--------------------------------
    # blockA:    (TileM, TileN) -> physical address
    # tv_layout: (tid, vid)     -> (TileM, TileN)
    # tidfrgA = blkA o tv_layout
    # tidfrgA:   (tid, vid) -> physical address
    tidfrgA = cute.composition(blkA, tv_layout)
    tidfrgB = cute.composition(blkB, tv_layout)
    tidfrgC = cute.composition(blkC, tv_layout)

    print(f"Composed with TV layout:")
    print(f"  tidfrgA: {tidfrgA.type}")

    #--------------------------------
    # slice for thread-level view
    #--------------------------------
    # `None` represent slice of the entire per-thread data
    thr_coord = (tidx, None)

    # slice for threads: vid -> address
    thrA = tidfrgA[thr_coord]  # (V) -> physical address
    thrB = tidfrgB[thr_coord]  # (V) -> physical address
    thrC = tidfrgC[thr_coord]  # (V) -> physical address

    thrC[None] = thrA.load() + thrB.load()


@cute.jit
def elementwise_add(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
):
    # mA layout: (M, N):(N, 1)
    # TV layout map thread & value index to (16, 256) logical tile
    #  - contiguous thread index maps to mode-1 because input layout is contiguous on
    #     mode-1 for coalesced load-store
    #  - each thread load 8 contiguous element each row and load 4 rows
    thr_layout = cute.make_layout((4, 32), stride=(32, 1))
    val_layout = cute.make_layout((4, 8), stride=(8, 1))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    print(f"Tiler: {tiler_mn}")
    print(f"TV Layout: {tv_layout}")

    gA = cute.zipped_divide(mA, tiler_mn)  # ((TileM, TileN), (RestM, RestN))
    gB = cute.zipped_divide(mB, tiler_mn)  # ((TileM, TileN), (RestM, RestN))
    gC = cute.zipped_divide(mC, tiler_mn)  # ((TileM, TileN), (RestM, RestN))

    print(f"Tiled Input Tensors:")
    print(f"  gA: {gA.type}")
    print(f"  gB: {gB.type}")
    print(f"  gC: {gC.type}")

    # Launch the kernel asynchronously
    # Async token(s) can also be specified as dependencies
    elementwise_add_kernel(
        gA, gB, gC, tv_layout
    ).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )

a = torch.randn(M, N, device="cuda", dtype=torch.float16)
b = torch.randn(M, N, device="cuda", dtype=torch.float16)
c = torch.zeros(M, N, device="cuda", dtype=torch.float16)

a_ = from_dlpack(a, assumed_align=16)
b_ = from_dlpack(b, assumed_align=16)
c_ = from_dlpack(c, assumed_align=16)

elementwise_add_ = cute.compile(elementwise_add, a_, b_, c_)

# elementwise_add_(a_, b_, c_)

# # verify correctness
# torch.testing.assert_close(c, a + b)

# %%
# benchmark(partial(elementwise_add_, a_, b_, c_), num_warmups=5, num_iterations=200)


# # %% [markdown]
# # ### Using Lambda Function
# #
# # CuTe DSL is built on top of Python. It can leverage Python to implement meta-programming to generate flexible kernels.
# # E.g. we can write kernel template that take custom binary operations to generate kernels for arbitrary binary operations.
# #
# #
# # ```python
# # @cute.jit
# # def elementwise_apply(
# #     op: cutlass.Constexpr,
# #     mA: cute.Tensor,
# #     mB: cute.Tensor,
# #     mC: cute.Tensor
# # ):
# #     ...
# #
# # ```

# # %%
# @cute.kernel
# def elementwise_apply_kernel(
#     op: cutlass.Constexpr,    # lambda function must be const expr to generate code at compile time
#     gA: cute.Tensor,
#     gB: cute.Tensor,
#     gC: cute.Tensor,
#     tv_layout: cute.Layout
# ):
#     tidx, _, _ = cute.arch.thread_idx()
#     bidx, _, _ = cute.arch.block_idx()

#     blk_coord = ((None, None), bidx)

#     # logical coord -> address
#     blkA = gA[blk_coord]  # (TileM, TileN) -> physical address
#     blkB = gB[blk_coord]  # (TileM, TileN) -> physical address
#     blkC = gC[blk_coord]  # (TileM, TileN) -> physical address

#     tidfrgA = cute.composition(blkA, tv_layout)
#     tidfrgB = cute.composition(blkB, tv_layout)
#     tidfrgC = cute.composition(blkC, tv_layout)

#     print(f"Composed with TV layout:")
#     print(f"  tidfrgA: {tidfrgA.type}")

#     thr_coord = (tidx, None)

#     # slice for threads: vid -> address
#     thrA = tidfrgA[thr_coord]  # (V) -> physical address
#     thrB = tidfrgB[thr_coord]  # (V) -> physical address
#     thrC = tidfrgC[thr_coord]  # (V) -> physical address

#     #--------------------------------
#     # apply custom operation
#     #--------------------------------
#     thrC[None] = op(thrA.load(), thrB.load())


# @cute.jit
# def elementwise_op(
#     op: cutlass.Constexpr,
#     mA: cute.Tensor,
#     mB: cute.Tensor,
#     mC: cute.Tensor,
# ):
#     # mA layout: (M, N):(N, 1)
#     # TV layout map thread & value index to (16, 256) logical tile
#     #  - contiguous thread index maps to mode-1 because input layout is contiguous on
#     #     mode-1 for coalesced load-store
#     #  - each thread load 8 contiguous element each row and load 4 rows
#     thr_layout = cute.make_layout((4, 32), stride=(32, 1))
#     val_layout = cute.make_layout((4, 8), stride=(8, 1))
#     tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
#     print(f"Tiler: {tiler_mn}")
#     print(f"TV Layout: {tv_layout}")

#     gA = cute.zipped_divide(mA, tiler_mn)  # ((TileM, TileN), (RestM, RestN))
#     gB = cute.zipped_divide(mB, tiler_mn)  # ((TileM, TileN), (RestM, RestN))
#     gC = cute.zipped_divide(mC, tiler_mn)  # ((TileM, TileN), (RestM, RestN))

#     print(f"Tiled Input Tensors:")
#     print(f"  gA: {gA.type}")
#     print(f"  gB: {gB.type}")
#     print(f"  gC: {gC.type}")

#     # Launch the kernel asynchronously
#     # Async token(s) can also be specified as dependencies
#     elementwise_apply_kernel(
#         op, gA, gB, gC, tv_layout
#     ).launch(
#         grid=[cute.size(gC, mode=[1]), 1, 1],
#         block=[cute.size(tv_layout, mode=[0]), 1, 1],
#     )

# a = torch.randn(M, N, device="cuda", dtype=torch.float16)
# b = torch.randn(M, N, device="cuda", dtype=torch.float16)
# c = torch.zeros(M, N, device="cuda", dtype=torch.float16)

# a_ = from_dlpack(a, assumed_align=16)
# b_ = from_dlpack(b, assumed_align=16)
# c_ = from_dlpack(c, assumed_align=16)

# from operator import mul

# elementwise_op(mul, a_, b_, c_)

# # verify correctness
# torch.testing.assert_close(c, mul(a, b))


# # %% [markdown]
# # Custom operators can be more complex. For example, here's a function that performs
# # multiplication followed by ReLU:

# # %%
# def mul_relu(a, b):
#     tmp = a * b
#     return cute.where(tmp > 0, tmp, cute.full_like(tmp, 0))


# # As we uses cute.where in customized operation, we need to create another relu function
# def mul_relu_ref(a, b):
#     tmp = a * b
#     return torch.relu(tmp)


# elementwise_op(mul_relu, a_, b_, c_)

# # verify correctness
# torch.testing.assert_close(c, mul_relu_ref(a, b))
