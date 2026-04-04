from __future__ import annotations

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from .cute_runtime_helpers import convert_torch_tensor_to_cute_tensor


class StandaloneGatherAKernel:
    """Standalone forward-style `load_A_gather` extraction tiled across CTAs.

    This models the `compute_weight_gradient == False` branch of
    `HopperWgmma_MoE_kernel.load_A_gather` in
    `thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py`.

    The logic is intentionally kept structurally close to the original:
    - the same thread decomposition over `(tile_M, tile_K)`
    - the same per-thread index prefetch helper
    - the same 128-bit vectorized gather/load pattern

    The one purposeful adaptation is the destination:
    instead of writing into the grouped-GEMM shared-memory staging tile,
    this standalone kernel writes into a normal output tensor so the tile
    contents can be inspected and tested directly.
    """

    def __init__(
        self,
        tile_M: int,
        tile_K: int,
        token_group_size: int,
        K_start: int,
        K_extent: int,
        num_load_A_threads: int,
        universal_copy_bits: int = 128,
    ):
        """Capture the static tile/config values used by the standalone gather kernel."""
        # These values are baked into the compiled CuTe kernel so the launch mirrors
        # the original grouped-GEMM helper's fixed CTA tile shape.
        self.tile_M = tile_M
        self.tile_K = tile_K
        self.token_group_size = token_group_size
        self.K_start = K_start
        self.K_extent = K_extent
        self.num_load_A_threads = num_load_A_threads
        self.universal_copy_bits = universal_copy_bits
        self.tma_warp_id = 0

    @dsl_user_op
    def elem_pointer(self, x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None) -> cute.Pointer:
        """Mirror the original helper that turns a tensor coordinate into an element pointer."""
        return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)

    @dsl_user_op
    def min_i32(self, a: int | cutlass.Int32, b: int | cutlass.Int32, *, loc=None, ip=None) -> cutlass.Int32:
        """Compute a runtime min exactly like the grouped GEMM helper does."""
        return cutlass.Int32(
            llvm.inline_asm(
                T.i32(),
                [cutlass.Int32(a).ir_value(loc=loc, ip=ip), cutlass.Int32(b).ir_value(loc=loc, ip=ip)],
                "min.s32 $0, $1, $2;",
                "=r,r,r",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @cute.jit
    def prefetch_gather_idx_for_A_when_vary_M(
        self, mAIdx: cute.Tensor, M_offset: int, M_boundary: int, copy_elems_per_thr_load: int
    ) -> cute.Tensor:
        """Prefetch the gather indices for the CTA's current M tile into registers."""
        M, K = self.tile_M, self.tile_K

        tidx, _, _ = cute.arch.thread_idx()
        tidx = tidx - self.tma_warp_id * cute.arch.WARP_SIZE

        # Match the original thread decomposition: threads are laid out so each one
        # owns one vector along K and iterates over a subset of rows in the CTA tile.
        stride_1_tile, other_tile = K, M
        threads_per_stride_1_dim = cutlass.const_expr(stride_1_tile // copy_elems_per_thr_load)
        num_other_dim_per_load = cutlass.const_expr(self.num_load_A_threads // threads_per_stride_1_dim)
        num_other_dim_per_thread = cutlass.const_expr(other_tile // num_other_dim_per_load)

        # Store the gathered source-row ids in registers so the later copy loop can
        # issue vector loads without rereading the index tensor from memory.
        tmAIdx = cute.make_rmem_tensor((num_other_dim_per_load,), dtype=mAIdx.element_type)

        for i in cutlass.range_constexpr(num_other_dim_per_thread):
            other_dim_offset = cutlass.const_expr(i * num_other_dim_per_load) + tidx // threads_per_stride_1_dim
            if other_dim_offset < M_boundary:
                M_i = M_offset + other_dim_offset
                tmAIdx[i] = mAIdx[M_i]

        return tmAIdx

    def _make_tiled_copy_2D(
        self,
        tensor: cute.Tensor,
        tile_shape_0: cutlass.Int32,
        tile_shape_1: cutlass.Int32,
        is_row_major: bool,
        threads_for_copy: int,
        universal_copy_bits: int,
    ) -> cute.TiledCopy:
        """Recreate the original 2D vectorized copy decomposition for gathered A tiles."""
        # The original kernel uses 128-bit vector copies to maximize memory bandwidth
        # on the row-major K dimension. This helper rebuilds that exact mapping.
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            tensor.element_type,
            num_bits_per_copy=universal_copy_bits,
        )
        copy_elems = universal_copy_bits // tensor.element_type.width
        shape_dim_1 = cute.size(tile_shape_1) // copy_elems
        thread_layout = cute.make_layout((threads_for_copy // shape_dim_1, shape_dim_1), stride=(shape_dim_1, 1))
        if not is_row_major:
            shape_dim_0 = cute.size(tile_shape_0) // copy_elems
            thread_layout = cute.make_layout((shape_dim_0, threads_for_copy // shape_dim_0), stride=(1, shape_dim_0))
        value_layout = cute.make_layout((1, copy_elems)) if is_row_major else cute.make_layout((copy_elems, 1))
        return cute.make_tiled_copy_tv(copy_atom, thread_layout, value_layout)

    @cute.jit
    def load_A_gather(
        self,
        mA: cute.Tensor,
        tmAIdx: cute.Tensor,
        mOut: cute.Tensor,
        A_thr_copy,
        M_offset: cutlass.Int32,
        M_boundary: cutlass.Int32,
        K_offset: cutlass.Int32,
        copy_elems_per_thr_load: cutlass.Int32,
    ):
        """Load one `(tile_M, tile_K)` gathered A tile from GMEM into the output tensor."""
        M, K = self.tile_M, self.tile_K

        tidx, _, _ = cute.arch.thread_idx()
        tidx = tidx - self.tma_warp_id * cute.arch.WARP_SIZE

        # Reuse the same per-thread ownership rule as the original helper so this
        # standalone kernel stays structurally comparable when studied in isolation.
        stride_1_tile, other_tile = K, M
        threads_per_stride_1_dim = cutlass.const_expr(stride_1_tile // copy_elems_per_thr_load)
        num_other_dim_per_load = cutlass.const_expr(self.num_load_A_threads // threads_per_stride_1_dim)

        for i in cutlass.range_constexpr(cute.ceil_div(other_tile, num_other_dim_per_load)):
            stride_1_dim_offset = (tidx % threads_per_stride_1_dim) * copy_elems_per_thr_load
            other_dim_offset = cutlass.const_expr(i * num_other_dim_per_load) + tidx // threads_per_stride_1_dim

            if other_dim_offset < M_boundary:
                MIdx = tmAIdx[i]
                KIdx = K_offset + stride_1_dim_offset

                # Read one vector from the gathered source row...
                src_ptr = self.elem_pointer(mA, (MIdx, KIdx)).align(self.universal_copy_bits // copy_elems_per_thr_load)
                src_tensor = cute.make_tensor(src_ptr, ((copy_elems_per_thr_load, 1), 1))

                # ...and place it into this CTA's slice of the global study/output tensor.
                dst_ptr = self.elem_pointer(mOut, (M_offset + other_dim_offset, KIdx - self.K_start)).align(
                    self.universal_copy_bits // copy_elems_per_thr_load
                )
                dst_tensor = cute.make_tensor(dst_ptr, ((copy_elems_per_thr_load, 1), 1))

                cute.copy(A_thr_copy, src_tensor, dst_tensor)

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mAIdx: cute.Tensor,
        mOut: cute.Tensor,
        stream: cuda.CUstream,
    ):
        """Launch enough CTAs to cover the requested gathered output rectangle."""
        # `grid.x` tiles over the token-group rows and `grid.y` tiles over the K range,
        # which is the same outer decomposition the original grouped-GEMM path uses.
        self.kernel(mA, mAIdx, mOut).launch(
            grid=[cute.ceil_div(self.token_group_size, self.tile_M), self.K_extent // self.tile_K, 1],
            block=[self.num_load_A_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mAIdx: cute.Tensor,
        mOut: cute.Tensor,
    ):
        """Compute the CTA-local gather tile and write it into the corresponding output slice."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        copy_elems_per_thr_load = self.universal_copy_bits // mA.element_type.width
        # Recover the same tile offsets that the grouped-GEMM producer loop would have
        # passed to `load_A_gather` for this CTA.
        M_offset = bidx * cutlass.const_expr(self.tile_M)
        M_boundary = self.min_i32(cutlass.const_expr(self.tile_M), self.token_group_size - M_offset)
        K_offset = self.K_start + bidy * cutlass.const_expr(self.tile_K)

        # Build the per-thread copy view once, then reuse it for the tile's gather loads.
        A_tiled_copy = self._make_tiled_copy_2D(
            mA,
            self.tile_M,
            self.tile_K,
            is_row_major=True,
            threads_for_copy=self.num_load_A_threads,
            universal_copy_bits=self.universal_copy_bits,
        )
        A_thr_copy = A_tiled_copy.get_slice(tidx)

        tmAIdx = self.prefetch_gather_idx_for_A_when_vary_M(
            mAIdx,
            M_offset,
            M_boundary,
            copy_elems_per_thr_load,
        )
        self.load_A_gather(
            mA,
            tmAIdx,
            mOut,
            A_thr_copy,
            M_offset,
            M_boundary,
            K_offset,
            copy_elems_per_thr_load,
        )


def gather_a_reference(
    A: torch.Tensor,
    A_idx: torch.Tensor,
    K_start: int = 0,
    K_extent: Optional[int] = None,
    token_group_size: Optional[int] = None,
) -> torch.Tensor:
    """Materialize the full gathered output region with a simple PyTorch reference path."""
    if token_group_size is None:
        token_group_size = A_idx.numel()
    if K_extent is None:
        K_extent = A.shape[1] - K_start

    if token_group_size == 0 or K_extent == 0:
        return torch.zeros((token_group_size, K_extent), device=A.device, dtype=A.dtype)

    gathered_rows = A_idx[:token_group_size].to(torch.long)
    return A.index_select(0, gathered_rows)[:, K_start : K_start + K_extent].contiguous()


def gather_a(
    A: torch.Tensor,
    A_idx: torch.Tensor,
    tile_M: int = 128,
    tile_K: int = 64,
    K_start: int = 0,
    K_extent: Optional[int] = None,
    token_group_size: Optional[int] = None,
    num_load_A_threads: Optional[int] = None,
) -> torch.Tensor:
    """Gather the full token-group slice using the original CTA tile shape decomposition."""
    # Keep the wrapper narrow and explicit so mismatches fail early instead of being
    # hidden inside CuTe compilation or runtime lowering.
    if A.device.type != "cuda":
        raise ValueError("gather_a requires a CUDA tensor")
    if A_idx.device != A.device:
        raise ValueError("A_idx must live on the same device as A")
    if A.dim() != 2:
        raise ValueError("A must be rank-2")
    if A_idx.dim() != 1:
        raise ValueError("A_idx must be rank-1")
    if A.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("Standalone gather_A currently supports float16 and bfloat16 only")
    if A_idx.dtype != torch.int32:
        raise TypeError("A_idx must be torch.int32")
    if A.stride(-1) != 1:
        raise ValueError("Standalone gather_A expects the source tensor to be row-major in K")

    if token_group_size is None:
        token_group_size = A_idx.numel()
    if K_extent is None:
        K_extent = A.shape[1] - K_start

    copy_elems = 128 // (A.element_size() * 8)
    if tile_K % copy_elems != 0:
        raise ValueError(f"tile_K must be divisible by the vector width ({copy_elems})")
    if K_start < 0:
        raise ValueError("K_start must be non-negative")
    if K_extent <= 0:
        raise ValueError("K_extent must be positive")
    if K_start + K_extent > A.shape[1]:
        raise ValueError("Requested K extent exceeds the source tensor width")
    if token_group_size > A_idx.numel():
        raise ValueError("token_group_size cannot exceed the available gather indices")
    if K_extent % tile_K != 0:
        raise ValueError("K_extent must be divisible by tile_K to match the original gather helper")

    if num_load_A_threads is None:
        num_load_A_threads = min(tile_M * tile_K // copy_elems, 128)
    if num_load_A_threads <= 0 or num_load_A_threads % cute.arch.WARP_SIZE != 0:
        raise ValueError("num_load_A_threads must be a positive multiple of the warp size")

    # The output is the fully materialized gathered rectangle, tiled over by CTAs.
    out = torch.zeros((token_group_size, K_extent), device=A.device, dtype=A.dtype)
    if token_group_size == 0:
        return out

    # Convert tensors into the CuTe runtime representation expected by `cute.compile`.
    stream_id = torch.cuda.current_stream(A.device).cuda_stream
    mA = convert_torch_tensor_to_cute_tensor(A.detach(), (0, 1), 1, 16, 8, stream=stream_id)
    mAIdx = convert_torch_tensor_to_cute_tensor(A_idx.detach(), (0,), 0, 4, 1, stream=stream_id)
    mOut = convert_torch_tensor_to_cute_tensor(out, (0, 1), 1, 16, 8, stream=stream_id)
    current_stream = cuda.CUstream(stream_id)

    compile_key = (
        A.dtype,
        tile_M,
        tile_K,
        K_start,
        K_extent,
        token_group_size,
        num_load_A_threads,
    )

    # Cache compiled variants by the static tile/config tuple, which mirrors how the
    # real kernel specializes on fixed tile shapes.
    if compile_key not in gather_a.compile_cache:
        kernel = StandaloneGatherAKernel(
            tile_M=tile_M,
            tile_K=tile_K,
            token_group_size=token_group_size,
            K_start=K_start,
            K_extent=K_extent,
            num_load_A_threads=num_load_A_threads,
        )
        gather_a.compile_cache[compile_key] = cute.compile(
            kernel,
            mA,
            mAIdx,
            mOut,
            current_stream,
        )

    gather_a.compile_cache[compile_key](mA, mAIdx, mOut, current_stream)
    return out


gather_a.compile_cache = {}
