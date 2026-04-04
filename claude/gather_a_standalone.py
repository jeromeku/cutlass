"""
Standalone gatherA kernel extracted from HopperWgmma_MoE_kernel.

Preserves the forward-pass (vary_M) gatherA path:
  - elem_pointer: coordinate-to-linear-pointer via crd2idx
  - min_i32: inline PTX min.s32
  - prefetch_gather_idx_for_A_when_vary_M: loads M-dim gather indices from GMEM into RMEM
  - load_A_gather (forward else branch): scattered cp.async loads using prefetched indices
  - _make_tiled_copy_2D: CopyG2SOp(cache_mode=GLOBAL) tiled copy construction

Simplifications vs original:
  - No producer/consumer pipeline: bare cp.async.commit_group + cp.async.wait_group
  - No TMA for B matrix
  - No WGMMA consumer, no epilogue
  - Single tile: one CTA processes one (tile_M, tile_K) tile
  - Single stage SMEM

Source: thirdparty/sonic-moe/sonicmoe/functional/grouped_gemm.py
"""

from typing import Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass import Int32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import warpgroup
from cutlass.cutlass_dsl import T, dsl_user_op


class GatherAStandalone:
    """Standalone gatherA kernel matching the forward-pass (vary_M) path.

    Parameters:
        tile_M: M-dimension tile size (number of rows to gather)
        tile_K: K-dimension tile size (contiguous columns per row)
        universal_copy_bits: bits per cp.async copy (default 128 = 16 bytes)
    """

    def __init__(self, tile_M: int = 128, tile_K: int = 64, universal_copy_bits: int = 128):
        self.tile_M = tile_M
        self.tile_K = tile_K
        self.universal_copy_bits = universal_copy_bits

        # Match original thread count formulas from grouped_gemm.py lines 194-210.
        # mma_warp_groups=1 (no ping-pong), so threads_per_cta = (1+1)*128 = 256
        self.mma_warp_groups = 1
        self.num_threads_per_warp_group = 128
        self.threads_per_cta = (self.mma_warp_groups + 1) * self.num_threads_per_warp_group  # 256
        self.tma_warp_id = self.mma_warp_groups * 4  # = 4 (warp index of first producer warp)

        # num_load_A_threads: min(tile_M * tile_K // 8, threads_per_cta - tma_warp_id * 32)
        # For BF16 (16-bit), universal_copy_bits=128 => 8 elements per load
        self.num_load_A_threads = min(
            tile_M * tile_K // 8,
            self.threads_per_cta - self.tma_warp_id * 32,
        )

        self.buffer_align_bytes = 1024

        # Will be set during __call__
        self.a_smem_layout_staged = None
        self.shared_storage = None
        self.token_group_size = None

    # -------------------------------------------------------------------------
    # Preserved device functions (lines 441-456 of grouped_gemm.py)
    # -------------------------------------------------------------------------

    @dsl_user_op
    def elem_pointer(self, x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None) -> cute.Pointer:
        """Convert coordinate to linear pointer via crd2idx. (line 441)"""
        return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)

    @dsl_user_op
    def min_i32(self, a, b, *, loc=None, ip=None) -> Int32:
        """Inline PTX min.s32. (line 444)"""
        return Int32(
            llvm.inline_asm(
                T.i32(),
                [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
                "min.s32 $0, $1, $2;",
                "=r,r,r",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    # -------------------------------------------------------------------------
    # Preserved: prefetch_gather_idx_for_A_when_vary_M (lines 458-483)
    # -------------------------------------------------------------------------

    @cute.jit
    def prefetch_gather_idx_for_A_when_vary_M(
        self, mAIdx: cute.Tensor, M_offset: int, M_boundary: int, copy_elems_per_thr_load: int
    ) -> cute.Tensor:
        M, K = self.tile_M, self.tile_K

        tidx, _, _ = cute.arch.thread_idx()
        tidx = tidx - self.tma_warp_id * cute.arch.WARP_SIZE

        stride_1_tile, other_tile = K, M

        threads_per_stride_1_dim = const_expr(stride_1_tile // copy_elems_per_thr_load)
        num_other_dim_per_load = const_expr(self.num_load_A_threads // threads_per_stride_1_dim)

        num_other_dim_per_thread = const_expr(other_tile // num_other_dim_per_load)
        tmAIdx = cute.make_rmem_tensor((num_other_dim_per_load,), dtype=mAIdx.element_type)

        for i in cutlass.range_constexpr(num_other_dim_per_thread):
            other_dim_offset = const_expr(i * num_other_dim_per_load) + tidx // threads_per_stride_1_dim

            if other_dim_offset < M_boundary:
                M_i = M_offset + other_dim_offset
                tmAIdx[i] = mAIdx[M_i]

        return tmAIdx

    # -------------------------------------------------------------------------
    # Preserved: load_A_gather forward else branch (lines 551-605)
    # -------------------------------------------------------------------------

    @cute.jit
    def load_A_gather(
        self,
        mA: cute.Tensor,
        tmAIdx: cute.Tensor,
        tAsA: cute.Tensor,
        tApA: cute.Tensor,
        A_g2s_thr_copy,
        K_offset: cutlass.Int32,
        copy_elems_per_thr_load: cutlass.Int32,
    ):
        M, K = self.tile_M, self.tile_K

        tidx, _, _ = cute.arch.thread_idx()
        tidx = tidx - self.tma_warp_id * cute.arch.WARP_SIZE

        # Forward path: stride_1 = K, other = M
        stride_1_tile, other_tile = K, M

        threads_per_stride_1_dim = const_expr(stride_1_tile // copy_elems_per_thr_load)
        num_other_dim_per_load = const_expr(self.num_load_A_threads // threads_per_stride_1_dim)

        for i in cutlass.range_constexpr(cute.ceil_div(other_tile, num_other_dim_per_load)):
            stride_1_dim_offset = (tidx % threads_per_stride_1_dim) * copy_elems_per_thr_load
            other_dim_offset = const_expr(i * num_other_dim_per_load) + tidx // threads_per_stride_1_dim

            # Forward else branch (lines 596-604)
            MIdx = tmAIdx[i]
            KIdx = K_offset + stride_1_dim_offset

            tPrAptr = self.elem_pointer(mA, (MIdx, KIdx)).align(
                self.universal_copy_bits // copy_elems_per_thr_load
            )
            mA_cur_copy = cute.make_tensor(tPrAptr, ((copy_elems_per_thr_load, 1), 1))
            cute.copy(A_g2s_thr_copy, mA_cur_copy, tAsA[None, i, None], pred=tApA[None, i, None])

    # -------------------------------------------------------------------------
    # Preserved: _make_tiled_copy_2D (lines 2981-3009)
    # -------------------------------------------------------------------------

    def _make_tiled_copy_2D(
        self,
        tensor: cute.Tensor,
        tile_shape_0,
        tile_shape_1,
        is_row_major: bool,
        threads_for_copy: int,
        universal_copy_bits: int,
    ) -> cute.TiledCopy:
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL),
            tensor.element_type,
            num_bits_per_copy=universal_copy_bits,
        )
        copy_elems = universal_copy_bits // tensor.element_type.width
        shape_dim_1 = cute.size(tile_shape_1) // copy_elems
        # thread layout for copy
        thread_layout = cute.make_layout((threads_for_copy // shape_dim_1, shape_dim_1), stride=(shape_dim_1, 1))
        if not is_row_major:
            shape_dim_0 = cute.size(tile_shape_0) // copy_elems
            thread_layout = cute.make_layout((shape_dim_0, threads_for_copy // shape_dim_0), stride=(1, shape_dim_0))
        # Value layout for copy
        value_layout = cute.make_layout((1, copy_elems)) if is_row_major else cute.make_layout((copy_elems, 1))
        return cute.make_tiled_copy_tv(copy_atom, thread_layout, value_layout)

    # -------------------------------------------------------------------------
    # SMEM layout construction (matching _make_smem_layouts, lines 2799-2817)
    # -------------------------------------------------------------------------

    def _make_a_smem_layout(self, a_dtype):
        """Create swizzled SMEM layout for A tensor (row-major = K-major)."""
        a_layout = cutlass.utils.LayoutEnum.ROW_MAJOR
        # For row-major A, K is the major mode
        a_major_mode_size = self.tile_K
        a_smem_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(a_layout, a_dtype, a_major_mode_size),
            a_dtype,
        )
        # Single stage: append 1 for the stage dimension
        a_smem_layout_staged = cute.tile_to_shape(
            a_smem_layout_atom,
            (self.tile_M, self.tile_K, 1),
            order=(0, 1, 2),  # K-major order
        )
        return a_smem_layout_staged

    # -------------------------------------------------------------------------
    # GPU device kernel
    # -------------------------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        A_tiled_copy: cute.TiledCopy,
        mA: cute.Tensor,       # (T, K) source matrix in GMEM
        mOut: cute.Tensor,     # (tile_M, tile_K) output in GMEM
        mAIdx: cute.Tensor,    # (num_tokens,) gather indices in GMEM
        a_smem_layout_staged: cute.ComposedLayout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # --- Allocate SMEM for one tile of A ---
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)

        # --- Only producer warps (warp >= tma_warp_id) execute the gather ---
        if warp_idx >= self.tma_warp_id:
            A_thr_copy_elems = const_expr(self.universal_copy_bits // mA.element_type.width)

            # Get tiled copy slice for this thread
            A_g2s_thr_copy = A_tiled_copy.get_slice(tidx - self.tma_warp_id * cute.arch.WARP_SIZE)

            # Create a dummy global tile for partitioning (to determine shapes)
            gA_mk = cute.local_tile(mA, (self.tile_M, self.tile_K), (0, None))
            tAgA = A_g2s_thr_copy.partition_S(gA_mk)

            # Create identity tensor for predicate construction
            mcA = cute.make_identity_tensor((self.token_group_size, mA.shape[1]))
            cA = cute.local_tile(mcA, (self.tile_M, self.tile_K), (0, None))

            # Partition SMEM destination and coordinate tensor
            tAsA = A_g2s_thr_copy.partition_D(sA)
            tAcA = A_g2s_thr_copy.partition_D(cA)

            # --- Build predicate tensor (lines 1791-1810) ---
            tApA = cute.make_rmem_tensor(
                cute.make_layout(
                    (
                        tAgA.shape[0][1],
                        cute.size(tAgA, mode=[1]),
                        cute.size(tAgA, mode=[2]),
                    ),
                    stride=(cute.size(tAgA, mode=[1]), 1, 0),
                ),
                cutlass.Boolean,
            )

            for rest_v in cutlass.range_constexpr(tApA.shape[0]):
                for m in cutlass.range_constexpr(tApA.shape[1]):
                    tApA[rest_v, m, 0] = cute.elem_less(
                        tAcA[(0, rest_v), m, 0, 0][0], self.token_group_size
                    )

            # --- Prefetch gather indices ---
            M_offset = cutlass.Int32(0)
            M_boundary = cute.arch.make_warp_uniform(
                self.min_i32(const_expr(self.tile_M), self.token_group_size - M_offset)
            )
            tmAIdx = self.prefetch_gather_idx_for_A_when_vary_M(
                mAIdx, M_offset, M_boundary, A_thr_copy_elems
            )

            # --- Scattered cp.async load ---
            K_offset = cutlass.Int32(0)
            self.load_A_gather(
                mA,
                tmAIdx,
                tAsA[None, None, None, 0],  # single stage, index 0
                tApA,
                A_g2s_thr_copy,
                K_offset,
                A_thr_copy_elems,
            )

            # --- Commit and wait for cp.async to complete ---
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)

        # --- Sync all threads before SMEM read ---
        cute.arch.sync_threads()

        # --- Copy SMEM -> GMEM output (all threads participate) ---
        copy_atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mOut.element_type)
        copy_elems = const_expr(self.universal_copy_bits // mOut.element_type.width)
        K_vecs = const_expr(self.tile_K // copy_elems)
        s2g_thr_layout = cute.make_layout(
            (const_expr(self.threads_per_cta // K_vecs), K_vecs),
            stride=(K_vecs, 1),
        )
        s2g_val_layout = cute.make_layout((1, copy_elems))
        s2g_copy = cute.make_tiled_copy_tv(copy_atom_store, s2g_thr_layout, s2g_val_layout)
        s2g_thr = s2g_copy.get_slice(tidx)

        sA_stage0 = cute.slice_(sA, (None, None, 0))
        thrS = s2g_thr.partition_S(sA_stage0)
        thrD = s2g_thr.partition_D(mOut)
        frag = cute.make_fragment_like(thrS)

        cute.copy(copy_atom_store, thrS, frag)
        cute.copy(copy_atom_store, frag, thrD)

    # -------------------------------------------------------------------------
    # Host wrapper
    # -------------------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,      # (T, K) BF16
        mOut: cute.Tensor,    # (tile_M, tile_K) BF16
        mAIdx: cute.Tensor,   # (num_tokens,) Int32
        stream: cuda.CUstream,
    ):
        a_dtype = mA.element_type

        # Create A tiled copy (GMEM -> SMEM via cp.async)
        A_tiled_copy = self._make_tiled_copy_2D(
            mA,
            self.tile_M,
            self.tile_K,
            True,  # row-major
            self.num_load_A_threads,
            self.universal_copy_bits,
        )

        # Create SMEM layout
        self.a_smem_layout_staged = self._make_a_smem_layout(a_dtype)

        # Define shared storage
        @cute.struct
        class SharedStorage:
            sA: cute.struct.Align[
                cute.struct.MemRange[a_dtype, cute.cosize(self.a_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch kernel: 1 CTA, threads_per_cta threads
        self.kernel(
            A_tiled_copy, mA, mOut, mAIdx, self.a_smem_layout_staged,
        ).launch(
            grid=[1, 1, 1],
            block=[self.threads_per_cta, 1, 1],
            stream=stream,
        )
