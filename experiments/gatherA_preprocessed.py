def load_A_gather(self, mA, tmAIdx, tAsA, tApA, A_g2s_thr_copy, K_offset, copy_elems_per_thr_load):
    import cutlass.base_dsl as __base_dsl__
    M, K = (self.tile_M, self.tile_K)
    tidx, _, _ = cute.arch.thread_idx()
    tidx = tidx - self.tma_warp_id * cute.arch.WARP_SIZE
    stride_1_tile, other_tile = (K, M)
    threads_per_stride_1_dim = const_expr(stride_1_tile // copy_elems_per_thr_load)
    num_other_dim_per_load = const_expr(self.num_load_A_threads // threads_per_stride_1_dim)
    __base_dsl__.ast_helpers.cf_symbol_check(cutlass.range_constexpr)
    for i in range(*__base_dsl__.ast_helpers.range_value_check(cute.ceil_div(other_tile, num_other_dim_per_load))):
        stride_1_dim_offset = tidx % threads_per_stride_1_dim * copy_elems_per_thr_load
        other_dim_offset = const_expr(i * num_other_dim_per_load) + tidx // threads_per_stride_1_dim
        MIdx = tmAIdx[i]
        KIdx = K_offset + stride_1_dim_offset
        tPrAptr = self.elem_pointer(mA, (MIdx, KIdx)).align(self.universal_copy_bits // copy_elems_per_thr_load)
        mA_cur_copy = cute.make_tensor(tPrAptr, ((copy_elems_per_thr_load, 1), 1))
        cute.copy(A_g2s_thr_copy, mA_cur_copy, tAsA[None, i, None], pred=tApA[None, i, None])