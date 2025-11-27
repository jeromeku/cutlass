"""Minimal CuTeDSL example: TMA load a swizzled 64x64 uint16 tile and dump the
physical shared-memory order back to GMEM for inspection."""

import argparse
import os

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack


TILE_SHAPE = (64, 64)
TILE_STRIDE = (64, 1)
THREADS_PER_CTA = 128


@cute.struct
class SharedStorage:
    # Single mbarrier to track the TMA load
    tma_bar: cute.struct.MemRange[cutlass.Int64, 1]


@cute.kernel
def tma_swizzle_kernel(
    load_atom: cute.CopyAtom,
    load_tma_tensor: cute.Tensor,
    store_atom: cute.CopyAtom,
    store_tma_tensor: cute.Tensor,
    g_out: cute.Tensor,
    smem_swizzled_layout: cute.ComposedLayout,
    smem_linear_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)

    # Allocate the swizzled SMEM tile that TMA will write into
    s_tile_swizzled = smem.allocate_tensor(
        element_type=cutlass.Uint16,
        layout=smem_swizzled_layout.outer,
        byte_alignment=128,
        swizzle=smem_swizzled_layout.inner,
    )
    # Reinterpret the same memory without swizzle so TMA store observes the
    # physical order of the swizzled tile.
    s_tile_linear = cute.make_tensor(
        cute.recast_ptr(s_tile_swizzled.iterator, swizzle_=None),
        smem_linear_layout,
    )

    # Descriptor prefetch + mbarrier init by warp 0
    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(storage.tma_bar.data_ptr(), 1)
            cpasync.prefetch_descriptor(load_atom)
            cpasync.prefetch_descriptor(store_atom)
    cute.arch.sync_threads()

    # Partition tensors for the TMA load
    s_part, g_part = cpasync.tma_partition(
        load_atom,
        0,
        cute.make_layout(1),
        s_tile_swizzled,
        load_tma_tensor,
    )

    bytes_to_copy = cute.size_in_bytes(cutlass.Uint16, smem_swizzled_layout)
    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(
                storage.tma_bar.data_ptr(), bytes_to_copy
            )
            cute.copy(
                load_atom,
                g_part,
                s_part,
                tma_bar_ptr=storage.tma_bar.data_ptr(),
            )

    cute.arch.mbarrier_wait(storage.tma_bar.data_ptr(), 0)
    cute.arch.sync_threads()

    # Partition tensors for the TMA store that dumps the physical SMEM order
    s_store_part, g_store_part = cpasync.tma_partition(
        store_atom,
        0,
        cute.make_layout(1),
        s_tile_linear,
        store_tma_tensor,
    )

    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.copy(store_atom, s_store_part, g_store_part)
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)


def run(swizzle_bits: int = 2):
    # Default to Hopper if the user did not choose an arch.
 #   os.environ.setdefault("CUTE_DSL_ARCH", "sm_90")

    device = "cuda:0"
    inp = (
        torch.arange(TILE_SHAPE[0] * TILE_SHAPE[1], dtype=torch.uint16, device=device)
        .reshape(TILE_SHAPE)
    )
    out = torch.empty_like(inp)

    g_in = from_dlpack(torch.utils.dlpack.to_dlpack(inp))
    g_out = from_dlpack(torch.utils.dlpack.to_dlpack(out))

    swizzle = cute.make_swizzle(swizzle_bits, 4, 3)
    layout = cute.make_layout(shape=TILE_SHAPE, stride=TILE_STRIDE)
    smem_swizzled = cute.make_composed_layout(swizzle, 0, layout)
    smem_linear = layout

    # TMA load: GMEM (row-major) -> SMEM (swizzled)
    load_atom, load_tma_tensor = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        g_in,
        smem_swizzled,
        TILE_SHAPE,
    )
    # TMA store: SMEM (physical order via linear layout) -> GMEM row-major
    store_atom, store_tma_tensor = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(),
        g_out,
        smem_linear,
        TILE_SHAPE,
    )

    tma_swizzle_kernel(
        load_atom,
        load_tma_tensor,
        store_atom,
        store_tma_tensor,
        g_out,
        smem_swizzled,
        smem_linear,
    ).launch(
        grid=(1, 1, 1),
        block=(1, 1, 1),
    )

    torch.cuda.synchronize()

    print("Row-major input (GMEM):")
    print(inp.cpu())
    print("\nSMEM physical order after swizzled TMA load (reshaped as 64x64):")
    print(out.cpu())


def parse():
    parser = argparse.ArgumentParser(
        description=(
            "TMA load a swizzled 64x64 uint16 tile and print the resulting SMEM order."
        )
    )
    parser.add_argument(
        "--swizzle_bits",
        type=int,
        default=2,
        help="swizzle strength (0=identity, 1=32B, 2=64B, 3=128B)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse()
    run(swizzle_bits=args.swizzle_bits)
