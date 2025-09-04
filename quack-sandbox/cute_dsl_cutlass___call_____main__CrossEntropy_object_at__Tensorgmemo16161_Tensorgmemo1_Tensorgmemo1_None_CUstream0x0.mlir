!copy_ldgsts = !cute.tiled_copy<!cute_nvgpu.atom.simt_async_copy<bf16, cache = always, 128 b>, layout_copy_tv = <"((8,16),(8,1)):((128,1),(16,1024))">, tiler_mn = <"[16:1;64:1]">>
!memref_gmem_bf16_ = !cute.memref<bf16, gmem, align<16>, "(?,16):(16,1)">
!memref_gmem_bf16_1 = !cute.memref<bf16, gmem, align<16>, "(16,64):(16,1)">
!memref_gmem_bf16_2 = !cute.memref<bf16, gmem, align<16>, "((8,1),1,1):((1,0),0,0)">
!memref_gmem_f32_ = !cute.memref<f32, gmem, "(?):(1)">
!memref_gmem_i64_ = !cute.memref<i64, gmem, "(?):(1)">
!memref_rmem_bf16_ = !cute.memref<bf16, rmem, align<16>, "((8,1),1,1):((1,0),0,0)">
!memref_rmem_bf16_1 = !cute.memref<bf16, rmem, align<16>, "(8):(1)">
!memref_rmem_bf16_2 = !cute.memref<bf16, rmem, align<16>, "(8,1):(1,0)">
!memref_rmem_f32_ = !cute.memref<f32, rmem, align<32>, "((8,1),1,1):((1,0),0,0)">
!memref_rmem_i8_ = !cute.memref<i8, rmem, align<32>, "(1,1,1):(1,0,1)">
!memref_smem_bf16_ = !cute.memref<bf16, smem, align<1024>, "(16,64):(64,1)">
!memref_smem_bf16_1 = !cute.memref<bf16, smem, align<16>, "((8,1),1,1):((1,0),0,0)">
!memref_smem_bf16_2 = !cute.memref<bf16, smem, align<16>, "(8):(1)">
!memref_smem_bf16_3 = !cute.memref<bf16, smem, align<16>, "(8,1):(1,0)">
!memref_smem_i64_ = !cute.memref<i64, smem, align<1024>, "(4,(1,1),1):(1,(0,0),0)">
!memref_smem_i64_1 = !cute.memref<i64, smem, align<1024>, "(4,(1,1)):(1,(0,0))">
#loop_unroll = #llvm.loop_unroll<full = true>
#loop_annotation = #llvm.loop_annotation<unroll = #loop_unroll>
module attributes {gpu.container_module} {
  gpu.module @kernels {
    func.func public @kernel_cutlass_kernel___main__CrossEntropy_object_at__tensorptrbf16gmemalign16o16161_tensorptri64gmemo1_tensorptrf32gmemo1_None_16_64_0(%arg0: !memref_gmem_bf16_, %arg1: !memref_gmem_i64_, %arg2: !memref_gmem_f32_, %arg3: !cute.layout<"((8,16),(8,1)):((128,1),(16,1024))">) attributes {cute.kernel, gpu.kernel, nvvm.reqntid = array<i32: 128, 1, 1>} {
      %iter = cute.get_iter(%arg0) : !memref_gmem_bf16_
      %iter_0 = cute.get_iter(%arg1) : !memref_gmem_i64_
      %iter_1 = cute.get_iter(%arg2) : !memref_gmem_f32_
      %iter_2 = cute.get_iter(%arg0) : !memref_gmem_bf16_
      %iter_3 = cute.get_iter(%arg1) : !memref_gmem_i64_
      %iter_4 = cute.get_iter(%arg2) : !memref_gmem_f32_
      %lay = cute.get_layout(%arg0) : !memref_gmem_bf16_
      %0 = cute.get_shape(%lay) : (!cute.layout<"(?,16):(16,1)">) -> !cute.shape<"(?,16)">
      %e0, %e1 = cute.get_leaves(%0) : !cute.shape<"(?,16)">
      %itup = cute.to_int_tuple(%e0) : !cute.shape<"?"> to !cute.int_tuple<"?">
      %1 = cute.get_scalars(%itup) : !cute.int_tuple<"?">
      %2 = cute.get_stride(%lay) : (!cute.layout<"(?,16):(16,1)">) -> !cute.stride<"(16,1)">
      %e0_5, %e1_6 = cute.get_leaves(%2) : !cute.stride<"(16,1)">
      %lay_7 = cute.get_layout(%arg1) : !memref_gmem_i64_
      %3 = cute.get_shape(%lay_7) : (!cute.layout<"(?):(1)">) -> !cute.shape<"(?)">
      %e0_8 = cute.get_leaves(%3) : !cute.shape<"(?)">
      %itup_9 = cute.to_int_tuple(%e0_8) : !cute.shape<"?"> to !cute.int_tuple<"?">
      %4 = cute.get_scalars(%itup_9) : !cute.int_tuple<"?">
      %5 = cute.get_stride(%lay_7) : (!cute.layout<"(?):(1)">) -> !cute.stride<"(1)">
      %e0_10 = cute.get_leaves(%5) : !cute.stride<"(1)">
      %lay_11 = cute.get_layout(%arg2) : !memref_gmem_f32_
      %6 = cute.get_shape(%lay_11) : (!cute.layout<"(?):(1)">) -> !cute.shape<"(?)">
      %e0_12 = cute.get_leaves(%6) : !cute.shape<"(?)">
      %itup_13 = cute.to_int_tuple(%e0_12) : !cute.shape<"?"> to !cute.int_tuple<"?">
      %7 = cute.get_scalars(%itup_13) : !cute.int_tuple<"?">
      %8 = cute.get_stride(%lay_11) : (!cute.layout<"(?):(1)">) -> !cute.stride<"(1)">
      %e0_14 = cute.get_leaves(%8) : !cute.stride<"(1)">
      %9 = cute.get_shape(%arg3) : (!cute.layout<"((8,16),(8,1)):((128,1),(16,1024))">) -> !cute.shape<"((8,16),(8,1))">
      %e0_15, %e1_16, %e2, %e3 = cute.get_leaves(%9) : !cute.shape<"((8,16),(8,1))">
      %10 = cute.get_stride(%arg3) : (!cute.layout<"((8,16),(8,1)):((128,1),(16,1024))">) -> !cute.stride<"((128,1),(16,1024))">
      %e0_17, %e1_18, %e2_19, %e3_20 = cute.get_leaves(%10) : !cute.stride<"((128,1),(16,1024))">
      %11 = nvvm.read.ptx.sreg.tid.x : i32
      %12 = nvvm.read.ptx.sreg.tid.y : i32
      %13 = nvvm.read.ptx.sreg.tid.z : i32
      %14 = nvvm.read.ptx.sreg.ctaid.x : i32
      %15 = nvvm.read.ptx.sreg.ctaid.y : i32
      %16 = nvvm.read.ptx.sreg.ctaid.z : i32
      %lay_21 = cute.get_layout(%arg0) : !memref_gmem_bf16_
      %17 = cute.get_shape(%lay_21) : (!cute.layout<"(?,16):(16,1)">) -> !cute.shape<"(?,16)">
      %e0_22, %e1_23 = cute.get_leaves(%17) : !cute.shape<"(?,16)">
      %itup_24 = cute.to_int_tuple(%e0_22) : !cute.shape<"?"> to !cute.int_tuple<"?">
      %18 = cute.get_scalars(%itup_24) : !cute.int_tuple<"?">
      %shape = cute.make_shape(%itup_24) : (!cute.int_tuple<"?">) -> !cute.shape<"(?,16)">
      %19 = cute.make_identity_tensor(%shape) : !cute.coord_tensor<"(0,0)", "(?,16):(1@0,1@1)">
      %iter_25 = cute.get_iter(%19) : !cute.coord_tensor<"(0,0)", "(?,16):(1@0,1@1)">
      %e0_26, %e1_27 = cute.get_leaves(%iter_25) : !cute.int_tuple<"(0,0)">
      %c16_i32 = arith.constant 16 : i32
      %20 = arith.muli %14, %c16_i32 : i32
      %21 = arith.extsi %20 : i32 to i64
      %lay_28 = cute.get_layout(%arg0) : !memref_gmem_bf16_
      %22 = cute.get_stride(%lay_28) : (!cute.layout<"(?,16):(16,1)">) -> !cute.stride<"(16,1)">
      %e0_29, %e1_30 = cute.get_leaves(%22) : !cute.stride<"(16,1)">
      %c16_i64 = arith.constant 16 : i64
      %23 = arith.muli %21, %c16_i64 : i64
      %c0_i64 = arith.constant 0 : i64
      %24 = arith.addi %23, %c0_i64 : i64
      %25 = arith.addi %24, %c0_i64 : i64
      %26 = cute.ptrtoint(%iter_2) : !cute.ptr<bf16, gmem, align<16>> to i64
      %27 = arith.muli %25, %c16_i64 : i64
      %c8_i64 = arith.constant 8 : i64
      %28 = arith.floordivsi %27, %c8_i64 : i64
      %29 = arith.addi %26, %28 : i64
      %iv = cute.assume(%29) : (i64) -> !cute.i64<divby 16>
      %30 = cute.inttoptr(%iv) : !cute.i64<divby 16> to !cute.ptr<bf16, gmem, align<16>>
      %lay_31 = cute.get_layout(%arg0) : !memref_gmem_bf16_
      %view = cute.make_view(%30, %lay_31) : !memref_gmem_bf16_
      %iter_32 = cute.get_iter(%view) : !memref_gmem_bf16_
      %shape_33 = cute.make_shape() : () -> !cute.shape<"(16,64)">
      %coord = cute.make_coord() : () -> !cute.coord<"(0,0)">
      %tiled_view = cute.local_tile(%view, %shape_33, %coord) : (!memref_gmem_bf16_, !cute.shape<"(16,64)">, !cute.coord<"(0,0)">) -> !memref_gmem_bf16_1
      %iter_34 = cute.get_iter(%tiled_view) : !memref_gmem_bf16_1
      %shape_35 = cute.make_shape() : () -> !cute.shape<"(16,64)">
      %coord_36 = cute.make_coord(%14) : (i32) -> !cute.coord<"(?,0)">
      %tiled_view_37 = cute.local_tile(%19, %shape_35, %coord_36) : (!cute.coord_tensor<"(0,0)", "(?,16):(1@0,1@1)">, !cute.shape<"(16,64)">, !cute.coord<"(?,0)">) -> !cute.coord_tensor<"(?{div=16},0)", "(16,64):(1@0,1@1)">
      %iter_38 = cute.get_iter(%tiled_view_37) : !cute.coord_tensor<"(?{div=16},0)", "(16,64):(1@0,1@1)">
      %e0_39, %e1_40 = cute.get_leaves(%iter_38) : !cute.int_tuple<"(?{div=16},0)">
      %31 = cute.get_scalars(%e0_39) : !cute.int_tuple<"?{div=16}">
      %smem_ptr = cute_nvgpu.arch.get_dyn_smem() : !cute.ptr<i8, smem, align<1024>>
      %shape_41 = cute.make_shape() : () -> !cute.shape<"(16,64)">
      %int_tuple = cute.make_int_tuple() : () -> !cute.int_tuple<"(1,0)">
      %lay_42 = cute.make_ordered_layout(%shape_41, %int_tuple) : (!cute.shape<"(16,64)">, !cute.int_tuple<"(1,0)">) -> !cute.layout<"(16,64):(64,1)">
      %coord_43 = cute.make_coord() : () -> !cute.coord<"0">
      %idx = cute.crd2idx(%coord_43, %lay_42) : (!cute.coord<"0">, !cute.layout<"(16,64):(64,1)">) -> !cute.int_tuple<"0">
      %e0_44 = cute.get_leaves(%idx) : !cute.int_tuple<"0">
      %cosz = cute.cosize(%lay_42) : (!cute.layout<"(16,64):(64,1)">) -> !cute.int_tuple<"1024">
      %e0_45 = cute.get_leaves(%cosz) : !cute.int_tuple<"1024">
      %int_tuple_46 = cute.make_int_tuple() : () -> !cute.int_tuple<"2048">
      %ptr = cute.add_offset(%smem_ptr, %int_tuple_46) : (!cute.ptr<i8, smem, align<1024>>, !cute.int_tuple<"2048">) -> !cute.ptr<i8, smem, align<1024>>
      %smem_size = cute_nvgpu.arch.get_dyn_smem_size() : i32
      %c2048_i32 = arith.constant 2048 : i32
      %32 = arith.cmpi uge, %smem_size, %c2048_i32 : i32
      cf.assert %32, "Allocation failed: shared memory allocation exceeds available memory set in kernel launch. Allocated bytes: 2048 bytes. Please reduce the allocation or set a larger smem size in kernel launch."
      %iter_47 = cute.recast_iter(%smem_ptr) : !cute.ptr<i8, smem, align<1024>> to !cute.ptr<bf16, smem, align<1024>>
      %view_48 = cute.make_view(%iter_47, %lay_42) : !memref_smem_bf16_
      %iter_49 = cute.get_iter(%view_48) : !memref_smem_bf16_
      %sz = cute.size(%arg3) <{mode = [0]}> : (!cute.layout<"((8,16),(8,1)):((128,1),(16,1024))">) -> !cute.int_tuple<"128">
      %e0_50 = cute.get_leaves(%sz) : !cute.int_tuple<"128">
      %33 = cute.get_shape(%arg3) : (!cute.layout<"((8,16),(8,1)):((128,1),(16,1024))">) -> !cute.shape<"((8,16),(8,1))">
      %e0_51, %e1_52, %e2_53, %e3_54 = cute.get_leaves(%33) : !cute.shape<"((8,16),(8,1))">
      %shape_55 = cute.make_shape() : () -> !cute.shape<"(4,(1,1),1)">
      %int_tuple_56 = cute.make_int_tuple() : () -> !cute.int_tuple<"(1,0,2)">
      %lay_57 = cute.make_ordered_layout(%shape_55, %int_tuple_56) : (!cute.shape<"(4,(1,1),1)">, !cute.int_tuple<"(1,0,2)">) -> !cute.layout<"(4,(1,1),1):(1,(0,0),0)">
      %coord_58 = cute.make_coord() : () -> !cute.coord<"0">
      %idx_59 = cute.crd2idx(%coord_58, %lay_57) : (!cute.coord<"0">, !cute.layout<"(4,(1,1),1):(1,(0,0),0)">) -> !cute.int_tuple<"0">
      %e0_60 = cute.get_leaves(%idx_59) : !cute.int_tuple<"0">
      %cosz_61 = cute.cosize(%lay_57) : (!cute.layout<"(4,(1,1),1):(1,(0,0),0)">) -> !cute.int_tuple<"4">
      %e0_62 = cute.get_leaves(%cosz_61) : !cute.int_tuple<"4">
      %int_tuple_63 = cute.make_int_tuple() : () -> !cute.int_tuple<"32">
      %ptr_64 = cute.add_offset(%ptr, %int_tuple_63) : (!cute.ptr<i8, smem, align<1024>>, !cute.int_tuple<"32">) -> !cute.ptr<i8, smem, align<32>>
      %smem_size_65 = cute_nvgpu.arch.get_dyn_smem_size() : i32
      %c2080_i32 = arith.constant 2080 : i32
      %34 = arith.cmpi uge, %smem_size_65, %c2080_i32 : i32
      cf.assert %34, "Allocation failed: shared memory allocation exceeds available memory set in kernel launch. Allocated bytes: 2080 bytes. Please reduce the allocation or set a larger smem size in kernel launch."
      %iter_66 = cute.recast_iter(%ptr) : !cute.ptr<i8, smem, align<1024>> to !cute.ptr<i64, smem, align<1024>>
      %view_67 = cute.make_view(%iter_66, %lay_57) : !memref_smem_i64_
      %iter_68 = cute.get_iter(%view_67) : !memref_smem_i64_
      %atom = cute.atom() : !cute_nvgpu.atom.simt_async_copy<bf16, cache = always, 128 b>
      %tile = cute.make_tile() : () -> !cute.tile<"[16:1;64:1]">
      %35 = cute.make_tiled_copy(%atom) : !copy_ldgsts
      %coord_69 = cute.make_coord(%11) : (i32) -> !cute.coord<"?">
      %src_partitioned = cute.tiled.copy.partition_S(%35, %tiled_view, %coord_69) : (!copy_ldgsts, !memref_gmem_bf16_1, !cute.coord<"?">) -> !memref_gmem_bf16_2
      %iter_70 = cute.get_iter(%src_partitioned) : !memref_gmem_bf16_2
      %coord_71 = cute.make_coord(%11) : (i32) -> !cute.coord<"?">
      %dst_partitioned = cute.tiled.copy.partition_D(%35, %view_48, %coord_71) : (!copy_ldgsts, !memref_smem_bf16_, !cute.coord<"?">) -> !memref_smem_bf16_1
      %iter_72 = cute.get_iter(%dst_partitioned) : !memref_smem_bf16_1
      %coord_73 = cute.make_coord(%11) : (i32) -> !cute.coord<"?">
      %src_partitioned_74 = cute.tiled.copy.partition_S(%35, %tiled_view_37, %coord_73) : (!copy_ldgsts, !cute.coord_tensor<"(?{div=16},0)", "(16,64):(1@0,1@1)">, !cute.coord<"?">) -> !cute.coord_tensor<"(?,?{div=8})", "((8,1),1,1):((1@1,0),0,0)">
      %iter_75 = cute.get_iter(%src_partitioned_74) : !cute.coord_tensor<"(?,?{div=8})", "((8,1),1,1):((1@1,0),0,0)">
      %e0_76, %e1_77 = cute.get_leaves(%iter_75) : !cute.int_tuple<"(?,?{div=8})">
      %36 = cute.get_scalars(%e0_76) : !cute.int_tuple<"?">
      %37 = cute.get_scalars(%e1_77) : !cute.int_tuple<"?{div=8}">
      %coord_78 = cute.make_coord() : () -> !cute.coord<"((0,_),_,_)">
      %slice = cute.slice(%src_partitioned_74, %coord_78) : !cute.coord_tensor<"(?,?{div=8})", "((8,1),1,1):((1@1,0),0,0)">, !cute.coord<"((0,_),_,_)">
      %iter_79 = cute.get_iter(%slice) : !cute.coord_tensor<"(?,?{div=8})", "(1,1,1):(0,0,0)">
      %e0_80, %e1_81 = cute.get_leaves(%iter_79) : !cute.int_tuple<"(?,?{div=8})">
      %38 = cute.get_scalars(%e0_80) : !cute.int_tuple<"?">
      %39 = cute.get_scalars(%e1_81) : !cute.int_tuple<"?{div=8}">
      %iter_82 = cute.get_iter(%slice) : !cute.coord_tensor<"(?,?{div=8})", "(1,1,1):(0,0,0)">
      %e0_83, %e1_84 = cute.get_leaves(%iter_82) : !cute.int_tuple<"(?,?{div=8})">
      %40 = cute.get_scalars(%e0_83) : !cute.int_tuple<"?">
      %41 = cute.get_scalars(%e1_84) : !cute.int_tuple<"?{div=8}">
      %42 = cute.make_fragment_like(%src_partitioned) <{elem_type = bf16}> : !memref_gmem_bf16_2 to !memref_rmem_bf16_
      %iter_85 = cute.get_iter(%42) : !memref_rmem_bf16_
      %iter_86 = cute.get_iter(%42) : !memref_rmem_bf16_
      %sz_87 = cute.size(%arg3) <{mode = [0]}> : (!cute.layout<"((8,16),(8,1)):((128,1),(16,1024))">) -> !cute.int_tuple<"128">
      %e0_88 = cute.get_leaves(%sz_87) : !cute.int_tuple<"128">
      %coord_89 = cute.make_coord() : () -> !cute.coord<"0">
      %slice_90 = cute.slice(%slice, %coord_89) : !cute.coord_tensor<"(?,?{div=8})", "(1,1,1):(0,0,0)">, !cute.coord<"0">
      %iter_91 = cute.get_iter(%slice_90) : !cute.coord_tensor<"(?,?{div=8})", "():()">
      %e0_92, %e1_93 = cute.get_leaves(%iter_91) : !cute.int_tuple<"(?,?{div=8})">
      %43 = cute.get_scalars(%e0_92) : !cute.int_tuple<"?">
      %44 = cute.get_scalars(%e1_93) : !cute.int_tuple<"?{div=8}">
      %iter_94 = cute.get_iter(%slice_90) : !cute.coord_tensor<"(?,?{div=8})", "():()">
      %e0_95, %e1_96 = cute.get_leaves(%iter_94) : !cute.int_tuple<"(?,?{div=8})">
      %45 = cute.get_scalars(%e0_95) : !cute.int_tuple<"?">
      %46 = cute.get_scalars(%e1_96) : !cute.int_tuple<"?{div=8}">
      %iter_97 = cute.get_iter(%slice_90) : !cute.coord_tensor<"(?,?{div=8})", "():()">
      %e0_98, %e1_99 = cute.get_leaves(%iter_97) : !cute.int_tuple<"(?,?{div=8})">
      %47 = cute.get_scalars(%e0_98) : !cute.int_tuple<"?">
      %48 = cute.get_scalars(%e1_99) : !cute.int_tuple<"?{div=8}">
      %49 = arith.cmpi slt, %47, %18 : i32
      %50 = arith.cmpi slt, %47, %18 : i32
      %coord_100 = cute.make_coord() : () -> !cute.coord<"0">
      %slice_101 = cute.slice(%slice, %coord_100) : !cute.coord_tensor<"(?,?{div=8})", "(1,1,1):(0,0,0)">, !cute.coord<"0">
      %iter_102 = cute.get_iter(%slice_101) : !cute.coord_tensor<"(?,?{div=8})", "():()">
      %e0_103, %e1_104 = cute.get_leaves(%iter_102) : !cute.int_tuple<"(?,?{div=8})">
      %51 = cute.get_scalars(%e0_103) : !cute.int_tuple<"?">
      %52 = cute.get_scalars(%e1_104) : !cute.int_tuple<"?{div=8}">
      %iter_105 = cute.get_iter(%slice_101) : !cute.coord_tensor<"(?,?{div=8})", "():()">
      %e0_106, %e1_107 = cute.get_leaves(%iter_105) : !cute.int_tuple<"(?,?{div=8})">
      %53 = cute.get_scalars(%e0_106) : !cute.int_tuple<"?">
      %54 = cute.get_scalars(%e1_107) : !cute.int_tuple<"?{div=8}">
      %iter_108 = cute.get_iter(%slice_101) : !cute.coord_tensor<"(?,?{div=8})", "():()">
      %e0_109, %e1_110 = cute.get_leaves(%iter_108) : !cute.int_tuple<"(?,?{div=8})">
      %55 = cute.get_scalars(%e0_109) : !cute.int_tuple<"?">
      %56 = cute.get_scalars(%e1_110) : !cute.int_tuple<"?{div=8}">
      %c0_i32 = arith.constant 0 : i32
      %57 = arith.cmpi eq, %56, %c0_i32 : i32
      %58 = arith.extui %50 : i1 to i32
      %59 = arith.cmpi ne, %58, %c0_i32 : i32
      %60 = arith.extui %50 : i1 to i32
      %61 = arith.extui %57 : i1 to i32
      %62 = arith.select %59, %61, %60 : i32
      %c0_i32_111 = arith.constant 0 : i32
      %63 = arith.cmpi ne, %62, %c0_i32_111 : i32
      %64 = scf.if %63 -> (i32) {
        %coord_392 = cute.make_coord(%e0_98) : (!cute.int_tuple<"?">) -> !cute.coord<"?">
        %206 = cute.memref.load(%arg1, %coord_392) : (!memref_gmem_i64_, !cute.coord<"?">) -> i64
        %207 = arith.trunci %206 : i64 to i32
        scf.yield %207 : i32
      } else {
        scf.yield %c0_i32_111 : i32
      }
      %coord_112 = cute.make_coord(%11) : (i32) -> !cute.coord<"?">
      %src_partitioned_113 = cute.tiled.copy.partition_S(%35, %tiled_view_37, %coord_112) : (!copy_ldgsts, !cute.coord_tensor<"(?{div=16},0)", "(16,64):(1@0,1@1)">, !cute.coord<"?">) -> !cute.coord_tensor<"(?,?{div=8})", "((8,1),1,1):((1@1,0),0,0)">
      %iter_114 = cute.get_iter(%src_partitioned_113) : !cute.coord_tensor<"(?,?{div=8})", "((8,1),1,1):((1@1,0),0,0)">
      %e0_115, %e1_116 = cute.get_leaves(%iter_114) : !cute.int_tuple<"(?,?{div=8})">
      %65 = cute.get_scalars(%e0_115) : !cute.int_tuple<"?">
      %66 = cute.get_scalars(%e1_116) : !cute.int_tuple<"?{div=8}">
      %sz_117 = cute.size(%src_partitioned_113) <{mode = [0, 1]}> : (!cute.coord_tensor<"(?,?{div=8})", "((8,1),1,1):((1@1,0),0,0)">) -> !cute.int_tuple<"1">
      %e0_118 = cute.get_leaves(%sz_117) : !cute.int_tuple<"1">
      %sz_119 = cute.size(%src_partitioned_113) <{mode = [1]}> : (!cute.coord_tensor<"(?,?{div=8})", "((8,1),1,1):((1@1,0),0,0)">) -> !cute.int_tuple<"1">
      %e0_120 = cute.get_leaves(%sz_119) : !cute.int_tuple<"1">
      %sz_121 = cute.size(%src_partitioned_113) <{mode = [2]}> : (!cute.coord_tensor<"(?,?{div=8})", "((8,1),1,1):((1@1,0),0,0)">) -> !cute.int_tuple<"1">
      %e0_122 = cute.get_leaves(%sz_121) : !cute.int_tuple<"1">
      %sz_123 = cute.size(%src_partitioned_113) <{mode = [2]}> : (!cute.coord_tensor<"(?,?{div=8})", "((8,1),1,1):((1@1,0),0,0)">) -> !cute.int_tuple<"1">
      %e0_124 = cute.get_leaves(%sz_123) : !cute.int_tuple<"1">
      %shape_125 = cute.make_shape() : () -> !cute.shape<"(1,1,1)">
      %stride = cute.make_stride() : () -> !cute.stride<"(1,0,1)">
      %lay_126 = cute.make_layout(%shape_125, %stride) : !cute.layout<"(1,1,1):(1,0,1)">
      %rmem = cute.memref.alloca(%lay_126) : !memref_rmem_i8_
      %iter_127 = cute.get_iter(%rmem) : !memref_rmem_i8_
      %iter_128 = cute.get_iter(%rmem) : !memref_rmem_i8_
      %lay_129 = cute.get_layout(%rmem) : !memref_rmem_i8_
      %67 = cute.get_shape(%lay_129) : (!cute.layout<"(1,1,1):(1,0,1)">) -> !cute.shape<"(1,1,1)">
      %e0_130, %e1_131, %e2_132 = cute.get_leaves(%67) : !cute.shape<"(1,1,1)">
      %lay_133 = cute.get_layout(%rmem) : !memref_rmem_i8_
      %68 = cute.get_shape(%lay_133) : (!cute.layout<"(1,1,1):(1,0,1)">) -> !cute.shape<"(1,1,1)">
      %e0_134, %e1_135, %e2_136 = cute.get_leaves(%68) : !cute.shape<"(1,1,1)">
      %coord_137 = cute.make_coord() : () -> !cute.coord<"((0,0),0,0)">
      %slice_138 = cute.slice(%src_partitioned_113, %coord_137) : !cute.coord_tensor<"(?,?{div=8})", "((8,1),1,1):((1@1,0),0,0)">, !cute.coord<"((0,0),0,0)">
      %iter_139 = cute.get_iter(%slice_138) : !cute.coord_tensor<"(?,?{div=8})", "():()">
      %e0_140, %e1_141 = cute.get_leaves(%iter_139) : !cute.int_tuple<"(?,?{div=8})">
      %69 = cute.get_scalars(%e0_140) : !cute.int_tuple<"?">
      %70 = cute.get_scalars(%e1_141) : !cute.int_tuple<"?{div=8}">
      %iter_142 = cute.get_iter(%slice_138) : !cute.coord_tensor<"(?,?{div=8})", "():()">
      %e0_143, %e1_144 = cute.get_leaves(%iter_142) : !cute.int_tuple<"(?,?{div=8})">
      %71 = cute.get_scalars(%e0_143) : !cute.int_tuple<"?">
      %72 = cute.get_scalars(%e1_144) : !cute.int_tuple<"?{div=8}">
      %iter_145 = cute.get_iter(%slice_138) : !cute.coord_tensor<"(?,?{div=8})", "():()">
      %e0_146, %e1_147 = cute.get_leaves(%iter_145) : !cute.int_tuple<"(?,?{div=8})">
      %73 = cute.get_scalars(%e0_146) : !cute.int_tuple<"?">
      %74 = cute.get_scalars(%e1_147) : !cute.int_tuple<"?{div=8}">
      %coord_148 = cute.make_coord(%e1_147) : (!cute.int_tuple<"?{div=8}">) -> !cute.coord<"?{div=8}">
      %coord_149 = cute.make_coord() : () -> !cute.coord<"16">
      %75 = cute.elem_less(%coord_148, %coord_149) : !cute.coord<"?{div=8}">, !cute.coord<"16">
      %76 = arith.extui %75 : i1 to i8
      %coord_150 = cute.make_coord() : () -> !cute.coord<"(0,0,0)">
      cute.memref.store(%rmem, %coord_150, %76) : (!memref_rmem_i8_, !cute.coord<"(0,0,0)">, i8) -> ()
      %77 = arith.cmpi slt, %47, %18 : i32
      scf.if %77 {
        cute.copy(%atom, %src_partitioned, %dst_partitioned, %rmem) : (!cute_nvgpu.atom.simt_async_copy<bf16, cache = always, 128 b>, !memref_gmem_bf16_2, !memref_smem_bf16_1, !memref_rmem_i8_)
      }
      nvvm.cp.async.commit.group
      nvvm.cp.async.wait.group 0
      %coord_151 = cute.make_coord() : () -> !cute.coord<"((_,0),0,0)">
      %slice_152 = cute.slice(%dst_partitioned, %coord_151) : !memref_smem_bf16_1, !cute.coord<"((_,0),0,0)">
      %iter_153 = cute.get_iter(%slice_152) : !memref_smem_bf16_2
      %iter_154 = cute.get_iter(%slice_152) : !memref_smem_bf16_2
      %78 = cute.make_fragment_like(%slice_152) <{elem_type = bf16}> : !memref_smem_bf16_2 to !memref_rmem_bf16_1
      %iter_155 = cute.get_iter(%78) : !memref_rmem_bf16_1
      %iter_156 = cute.get_iter(%78) : !memref_rmem_bf16_1
      %sz_157 = cute.size(%78) : (!memref_rmem_bf16_1) -> !cute.int_tuple<"8">
      %e0_158 = cute.get_leaves(%sz_157) : !cute.int_tuple<"8">
      %lay_159 = cute.get_layout(%78) : !memref_rmem_bf16_1
      %79 = cute.get_shape(%lay_159) : (!cute.layout<"(8):(1)">) -> !cute.shape<"(8)">
      %e0_160 = cute.get_leaves(%79) : !cute.shape<"(8)">
      %int_tuple_161 = cute.make_int_tuple() : () -> !cute.int_tuple<"(8)">
      %res = cute.tuple.product(%int_tuple_161) : (!cute.int_tuple<"(8)">) -> !cute.int_tuple<"8">
      %e0_162 = cute.get_leaves(%res) : !cute.int_tuple<"8">
      %cst = arith.constant dense<0xFF80> : vector<8xbf16>
      %coord_163 = cute.make_coord() : () -> !cute.coord<"_">
      %slice_164 = cute.slice(%78, %coord_163) : !memref_rmem_bf16_1, !cute.coord<"_">
      %iter_165 = cute.get_iter(%slice_164) : !memref_rmem_bf16_1
      %iter_166 = cute.get_iter(%slice_164) : !memref_rmem_bf16_1
      %lay_167 = cute.get_layout(%slice_164) : !memref_rmem_bf16_1
      %80 = cute.get_shape(%lay_167) : (!cute.layout<"(8):(1)">) -> !cute.shape<"(8)">
      %e0_168 = cute.get_leaves(%80) : !cute.shape<"(8)">
      %lay_169 = cute.get_layout(%slice_164) : !memref_rmem_bf16_1
      %81 = cute.get_shape(%lay_169) : (!cute.layout<"(8):(1)">) -> !cute.shape<"(8)">
      %e0_170 = cute.get_leaves(%81) : !cute.shape<"(8)">
      %int_tuple_171 = cute.make_int_tuple() : () -> !cute.int_tuple<"(8)">
      %sz_172 = cute.size(%int_tuple_171) : (!cute.int_tuple<"(8)">) -> !cute.int_tuple<"8">
      %e0_173 = cute.get_leaves(%sz_172) : !cute.int_tuple<"8">
      %int_tuple_174 = cute.make_int_tuple() : () -> !cute.int_tuple<"(8)">
      %sz_175 = cute.size(%int_tuple_174) : (!cute.int_tuple<"(8)">) -> !cute.int_tuple<"8">
      %e0_176 = cute.get_leaves(%sz_175) : !cute.int_tuple<"8">
      cute.memref.store_vec %cst, %slice_164, row_major : !memref_rmem_bf16_1
      %lay_177 = cute.get_layout(%dst_partitioned) : !memref_smem_bf16_1
      %82 = cute.get_shape(%lay_177) : (!cute.layout<"((8,1),1,1):((1,0),0,0)">) -> !cute.shape<"((8,1),1,1)">
      %e0_178, %e1_179, %e2_180, %e3_181 = cute.get_leaves(%82) : !cute.shape<"((8,1),1,1)">
      %lay_182 = cute.get_layout(%dst_partitioned) : !memref_smem_bf16_1
      %83 = cute.get_shape(%lay_182) : (!cute.layout<"((8,1),1,1):((1,0),0,0)">) -> !cute.shape<"((8,1),1,1)">
      %e0_183, %e1_184, %e2_185, %e3_186 = cute.get_leaves(%83) : !cute.shape<"((8,1),1,1)">
      %coord_187 = cute.make_coord() : () -> !cute.coord<"(0,0,0)">
      %84 = cute.memref.load(%rmem, %coord_187) : (!memref_rmem_i8_, !cute.coord<"(0,0,0)">) -> i8
      %c0_i8 = arith.constant 0 : i8
      %85 = arith.cmpi ne, %84, %c0_i8 : i8
      %false = arith.constant false
      %86 = arith.cmpi eq, %85, %false : i1
      scf.if %86 {
        %coord_392 = cute.make_coord() : () -> !cute.coord<"((_,0),_,0)">
        %slice_393 = cute.slice(%dst_partitioned, %coord_392) : !memref_smem_bf16_1, !cute.coord<"((_,0),_,0)">
        %iter_394 = cute.get_iter(%slice_393) : !memref_smem_bf16_3
        %iter_395 = cute.get_iter(%slice_393) : !memref_smem_bf16_3
        %lay_396 = cute.get_layout(%78) : !memref_rmem_bf16_1
        %206 = cute.get_shape(%lay_396) : (!cute.layout<"(8):(1)">) -> !cute.shape<"(8)">
        %e0_397 = cute.get_leaves(%206) : !cute.shape<"(8)">
        %lay_398 = cute.get_layout(%slice_393) : !memref_smem_bf16_3
        %207 = cute.get_shape(%lay_398) : (!cute.layout<"(8,1):(1,0)">) -> !cute.shape<"(8,1)">
        %e0_399, %e1_400 = cute.get_leaves(%207) : !cute.shape<"(8,1)">
        %lay_401 = cute.get_layout(%78) : !memref_rmem_bf16_1
        %lay_402 = cute.get_layout(%slice_393) : !memref_smem_bf16_3
        %rinv_403 = cute.right_inverse(%lay_402) : (!cute.layout<"(8,1):(1,0)">) -> !cute.layout<"8:1">
        %208 = cute.composition(%lay_401, %rinv_403) : (!cute.layout<"(8):(1)">, !cute.layout<"8:1">) -> !cute.layout<"8:1">
        %coalesce_404 = cute.coalesce(%208) : (!cute.layout<"8:1">) -> !cute.layout<"8:1">
        %209 = cute.get_shape(%coalesce_404) : (!cute.layout<"8:1">) -> !cute.shape<"8">
        %e0_405 = cute.get_leaves(%209) : !cute.shape<"8">
        %210 = cute.get_stride(%coalesce_404) : (!cute.layout<"8:1">) -> !cute.stride<"1">
        %e0_406 = cute.get_leaves(%210) : !cute.stride<"1">
        %211 = cute.get_shape(%coalesce_404) : (!cute.layout<"8:1">) -> !cute.shape<"8">
        %e0_407 = cute.get_leaves(%211) : !cute.shape<"8">
        %212 = cute.get_shape(%coalesce_404) : (!cute.layout<"8:1">) -> !cute.shape<"8">
        %e0_408 = cute.get_leaves(%212) : !cute.shape<"8">
        %213 = cute.composition(%rinv_403, %coalesce_404) : (!cute.layout<"8:1">, !cute.layout<"8:1">) -> !cute.layout<"8:1">
        %sz_409 = cute.size(%213) : (!cute.layout<"8:1">) -> !cute.int_tuple<"8">
        %e0_410 = cute.get_leaves(%sz_409) : !cute.int_tuple<"8">
        %lay_411 = cute.get_layout(%78) : !memref_rmem_bf16_1
        %lay_412 = cute.get_layout(%slice_393) : !memref_smem_bf16_3
        %div_413 = cute.logical_divide(%78, %213) : !memref_rmem_bf16_1, !cute.layout<"8:1">
        %iter_414 = cute.get_iter(%div_413) : !memref_rmem_bf16_2
        %iter_415 = cute.get_iter(%div_413) : !memref_rmem_bf16_2
        %div_416 = cute.logical_divide(%slice_393, %213) : !memref_smem_bf16_3, !cute.layout<"8:1">
        %iter_417 = cute.get_iter(%div_416) : !memref_smem_bf16_3
        %iter_418 = cute.get_iter(%div_416) : !memref_smem_bf16_3
        %shape_419 = cute.make_shape() : () -> !cute.shape<"8">
        %lay_420 = cute.make_layout(%shape_419) : !cute.layout<"8:1">
        %div_421 = cute.logical_divide(%div_413, %lay_420) : !memref_rmem_bf16_2, !cute.layout<"8:1">
        %iter_422 = cute.get_iter(%div_421) : !memref_rmem_bf16_2
        %iter_423 = cute.get_iter(%div_421) : !memref_rmem_bf16_2
        %shape_424 = cute.make_shape() : () -> !cute.shape<"8">
        %lay_425 = cute.make_layout(%shape_424) : !cute.layout<"8:1">
        %div_426 = cute.logical_divide(%div_416, %lay_425) : !memref_smem_bf16_3, !cute.layout<"8:1">
        %iter_427 = cute.get_iter(%div_426) : !memref_smem_bf16_3
        %iter_428 = cute.get_iter(%div_426) : !memref_smem_bf16_3
        %atom_429 = cute.atom() : !cute_nvgpu.atom.universal_copy<bf16, 128 b>
        cute.copy(%atom_429, %div_421, %div_426) : (!cute_nvgpu.atom.universal_copy<bf16, 128 b>, !memref_rmem_bf16_2, !memref_smem_bf16_3)
      }
      %lay_188 = cute.get_layout(%dst_partitioned) : !memref_smem_bf16_1
      %87 = cute.get_shape(%lay_188) : (!cute.layout<"((8,1),1,1):((1,0),0,0)">) -> !cute.shape<"((8,1),1,1)">
      %e0_189, %e1_190, %e2_191, %e3_192 = cute.get_leaves(%87) : !cute.shape<"((8,1),1,1)">
      %lay_193 = cute.get_layout(%42) : !memref_rmem_bf16_
      %88 = cute.get_shape(%lay_193) : (!cute.layout<"((8,1),1,1):((1,0),0,0)">) -> !cute.shape<"((8,1),1,1)">
      %e0_194, %e1_195, %e2_196, %e3_197 = cute.get_leaves(%88) : !cute.shape<"((8,1),1,1)">
      %lay_198 = cute.get_layout(%dst_partitioned) : !memref_smem_bf16_1
      %lay_199 = cute.get_layout(%42) : !memref_rmem_bf16_
      %rinv = cute.right_inverse(%lay_199) : (!cute.layout<"((8,1),1,1):((1,0),0,0)">) -> !cute.layout<"8:1">
      %89 = cute.composition(%lay_198, %rinv) : (!cute.layout<"((8,1),1,1):((1,0),0,0)">, !cute.layout<"8:1">) -> !cute.layout<"8:1">
      %coalesce = cute.coalesce(%89) : (!cute.layout<"8:1">) -> !cute.layout<"8:1">
      %90 = cute.get_shape(%coalesce) : (!cute.layout<"8:1">) -> !cute.shape<"8">
      %e0_200 = cute.get_leaves(%90) : !cute.shape<"8">
      %91 = cute.get_stride(%coalesce) : (!cute.layout<"8:1">) -> !cute.stride<"1">
      %e0_201 = cute.get_leaves(%91) : !cute.stride<"1">
      %92 = cute.get_shape(%coalesce) : (!cute.layout<"8:1">) -> !cute.shape<"8">
      %e0_202 = cute.get_leaves(%92) : !cute.shape<"8">
      %93 = cute.get_shape(%coalesce) : (!cute.layout<"8:1">) -> !cute.shape<"8">
      %e0_203 = cute.get_leaves(%93) : !cute.shape<"8">
      %94 = cute.composition(%rinv, %coalesce) : (!cute.layout<"8:1">, !cute.layout<"8:1">) -> !cute.layout<"8:1">
      %sz_204 = cute.size(%94) : (!cute.layout<"8:1">) -> !cute.int_tuple<"8">
      %e0_205 = cute.get_leaves(%sz_204) : !cute.int_tuple<"8">
      %lay_206 = cute.get_layout(%dst_partitioned) : !memref_smem_bf16_1
      %lay_207 = cute.get_layout(%42) : !memref_rmem_bf16_
      %div = cute.logical_divide(%dst_partitioned, %94) : !memref_smem_bf16_1, !cute.layout<"8:1">
      %iter_208 = cute.get_iter(%div) : !memref_smem_bf16_3
      %iter_209 = cute.get_iter(%div) : !memref_smem_bf16_3
      %div_210 = cute.logical_divide(%42, %94) : !memref_rmem_bf16_, !cute.layout<"8:1">
      %iter_211 = cute.get_iter(%div_210) : !memref_rmem_bf16_2
      %iter_212 = cute.get_iter(%div_210) : !memref_rmem_bf16_2
      %shape_213 = cute.make_shape() : () -> !cute.shape<"8">
      %lay_214 = cute.make_layout(%shape_213) : !cute.layout<"8:1">
      %div_215 = cute.logical_divide(%div, %lay_214) : !memref_smem_bf16_3, !cute.layout<"8:1">
      %iter_216 = cute.get_iter(%div_215) : !memref_smem_bf16_3
      %iter_217 = cute.get_iter(%div_215) : !memref_smem_bf16_3
      %shape_218 = cute.make_shape() : () -> !cute.shape<"8">
      %lay_219 = cute.make_layout(%shape_218) : !cute.layout<"8:1">
      %div_220 = cute.logical_divide(%div_210, %lay_219) : !memref_rmem_bf16_2, !cute.layout<"8:1">
      %iter_221 = cute.get_iter(%div_220) : !memref_rmem_bf16_2
      %iter_222 = cute.get_iter(%div_220) : !memref_rmem_bf16_2
      %atom_223 = cute.atom() : !cute_nvgpu.atom.universal_copy<bf16, 128 b>
      cute.copy(%atom_223, %div_215, %div_220) : (!cute_nvgpu.atom.universal_copy<bf16, 128 b>, !memref_smem_bf16_3, !memref_rmem_bf16_2)
      %lay_224 = cute.get_layout(%42) : !memref_rmem_bf16_
      %95 = cute.get_shape(%lay_224) : (!cute.layout<"((8,1),1,1):((1,0),0,0)">) -> !cute.shape<"((8,1),1,1)">
      %e0_225, %e1_226, %e2_227, %e3_228 = cute.get_leaves(%95) : !cute.shape<"((8,1),1,1)">
      %96 = cute.memref.load_vec %42, row_major : !memref_rmem_bf16_
      %lay_229 = cute.get_layout(%42) : !memref_rmem_bf16_
      %97 = cute.get_shape(%lay_229) : (!cute.layout<"((8,1),1,1):((1,0),0,0)">) -> !cute.shape<"((8,1),1,1)">
      %e0_230, %e1_231, %e2_232, %e3_233 = cute.get_leaves(%97) : !cute.shape<"((8,1),1,1)">
      %98 = arith.extf %96 : vector<8xbf16> to vector<8xf32>
      %99 = arith.cmpi slt, %47, %18 : i32
      %100 = arith.cmpi slt, %47, %18 : i32
      %coord_234 = cute.make_coord() : () -> !cute.coord<"0">
      %slice_235 = cute.slice(%slice, %coord_234) : !cute.coord_tensor<"(?,?{div=8})", "(1,1,1):(0,0,0)">, !cute.coord<"0">
      %iter_236 = cute.get_iter(%slice_235) : !cute.coord_tensor<"(?,?{div=8})", "():()">
      %e0_237, %e1_238 = cute.get_leaves(%iter_236) : !cute.int_tuple<"(?,?{div=8})">
      %101 = cute.get_scalars(%e0_237) : !cute.int_tuple<"?">
      %102 = cute.get_scalars(%e1_238) : !cute.int_tuple<"?{div=8}">
      %iter_239 = cute.get_iter(%slice_235) : !cute.coord_tensor<"(?,?{div=8})", "():()">
      %e0_240, %e1_241 = cute.get_leaves(%iter_239) : !cute.int_tuple<"(?,?{div=8})">
      %103 = cute.get_scalars(%e0_240) : !cute.int_tuple<"?">
      %104 = cute.get_scalars(%e1_241) : !cute.int_tuple<"?{div=8}">
      %iter_242 = cute.get_iter(%slice_235) : !cute.coord_tensor<"(?,?{div=8})", "():()">
      %e0_243, %e1_244 = cute.get_leaves(%iter_242) : !cute.int_tuple<"(?,?{div=8})">
      %105 = cute.get_scalars(%e0_243) : !cute.int_tuple<"?">
      %106 = cute.get_scalars(%e1_244) : !cute.int_tuple<"?{div=8}">
      %107 = arith.cmpi eq, %106, %c0_i32 : i32
      %108 = arith.extui %100 : i1 to i32
      %109 = arith.cmpi ne, %108, %c0_i32 : i32
      %110 = arith.extui %100 : i1 to i32
      %111 = arith.extui %107 : i1 to i32
      %112 = arith.select %109, %111, %110 : i32
      %113 = arith.cmpi ne, %112, %c0_i32_111 : i32
      %lay_245 = cute.get_layout(%view) : !memref_gmem_bf16_
      %114 = cute.get_shape(%lay_245) : (!cute.layout<"(?,16):(16,1)">) -> !cute.shape<"(?,16)">
      %e0_246, %e1_247 = cute.get_leaves(%114) : !cute.shape<"(?,16)">
      %itup_248 = cute.to_int_tuple(%e0_246) : !cute.shape<"?"> to !cute.int_tuple<"?">
      %115 = cute.get_scalars(%itup_248) : !cute.int_tuple<"?">
      %116 = cute.get_stride(%lay_245) : (!cute.layout<"(?,16):(16,1)">) -> !cute.stride<"(16,1)">
      %e0_249, %e1_250 = cute.get_leaves(%116) : !cute.stride<"(16,1)">
      %lay_251 = cute.get_layout(%view) : !memref_gmem_bf16_
      %117 = cute.get_shape(%lay_251) : (!cute.layout<"(?,16):(16,1)">) -> !cute.shape<"(?,16)">
      %e0_252, %e1_253 = cute.get_leaves(%117) : !cute.shape<"(?,16)">
      %itup_254 = cute.to_int_tuple(%e0_252) : !cute.shape<"?"> to !cute.int_tuple<"?">
      %118 = cute.get_scalars(%itup_254) : !cute.int_tuple<"?">
      %119 = cute.get_stride(%lay_251) : (!cute.layout<"(?,16):(16,1)">) -> !cute.stride<"(16,1)">
      %e0_255, %e1_256 = cute.get_leaves(%119) : !cute.stride<"(16,1)">
      %cst_257 = arith.constant 0.000000e+00 : f32
      %lay_258 = cute.get_layout(%view) : !memref_gmem_bf16_
      %120 = cute.get_shape(%lay_258) : (!cute.layout<"(?,16):(16,1)">) -> !cute.shape<"(?,16)">
      %e0_259, %e1_260 = cute.get_leaves(%120) : !cute.shape<"(?,16)">
      %itup_261 = cute.to_int_tuple(%e0_259) : !cute.shape<"?"> to !cute.int_tuple<"?">
      %121 = cute.get_scalars(%itup_261) : !cute.int_tuple<"?">
      %122 = cute.get_stride(%lay_258) : (!cute.layout<"(?,16):(16,1)">) -> !cute.stride<"(16,1)">
      %e0_262, %e1_263 = cute.get_leaves(%122) : !cute.stride<"(16,1)">
      %lay_264 = cute.get_layout(%view) : !memref_gmem_bf16_
      %123 = cute.get_shape(%lay_264) : (!cute.layout<"(?,16):(16,1)">) -> !cute.shape<"(?,16)">
      %e0_265, %e1_266 = cute.get_leaves(%123) : !cute.shape<"(?,16)">
      %itup_267 = cute.to_int_tuple(%e0_265) : !cute.shape<"?"> to !cute.int_tuple<"?">
      %124 = cute.get_scalars(%itup_267) : !cute.int_tuple<"?">
      %125 = cute.get_stride(%lay_264) : (!cute.layout<"(?,16):(16,1)">) -> !cute.stride<"(16,1)">
      %e0_268, %e1_269 = cute.get_leaves(%125) : !cute.stride<"(16,1)">
      %126:2 = scf.if %113 -> (!memref_gmem_bf16_, f32) {
        %iter_392 = cute.get_iter(%view) : !memref_gmem_bf16_
        %lay_393 = cute.get_layout(%view) : !memref_gmem_bf16_
        %206 = cute.get_shape(%lay_393) : (!cute.layout<"(?,16):(16,1)">) -> !cute.shape<"(?,16)">
        %e0_394, %e1_395 = cute.get_leaves(%206) : !cute.shape<"(?,16)">
        %itup_396 = cute.to_int_tuple(%e0_394) : !cute.shape<"?"> to !cute.int_tuple<"?">
        %207 = cute.get_scalars(%itup_396) : !cute.int_tuple<"?">
        %208 = cute.get_stride(%lay_393) : (!cute.layout<"(?,16):(16,1)">) -> !cute.stride<"(16,1)">
        %e0_397, %e1_398 = cute.get_leaves(%208) : !cute.stride<"(16,1)">
        %lay_399 = cute.get_layout(%view) : !memref_gmem_bf16_
        %209 = cute.get_shape(%lay_399) : (!cute.layout<"(?,16):(16,1)">) -> !cute.shape<"(?,16)">
        %e0_400, %e1_401 = cute.get_leaves(%209) : !cute.shape<"(?,16)">
        %itup_402 = cute.to_int_tuple(%e0_400) : !cute.shape<"?"> to !cute.int_tuple<"?">
        %210 = cute.get_scalars(%itup_402) : !cute.int_tuple<"?">
        %211 = cute.get_stride(%lay_399) : (!cute.layout<"(?,16):(16,1)">) -> !cute.stride<"(16,1)">
        %e0_403, %e1_404 = cute.get_leaves(%211) : !cute.stride<"(16,1)">
        %212 = arith.extsi %47 : i32 to i64
        %lay_405 = cute.get_layout(%arg0) : !memref_gmem_bf16_
        %213 = cute.get_stride(%lay_405) : (!cute.layout<"(?,16):(16,1)">) -> !cute.stride<"(16,1)">
        %e0_406, %e1_407 = cute.get_leaves(%213) : !cute.stride<"(16,1)">
        %c16_i64_408 = arith.constant 16 : i64
        %214 = arith.muli %212, %c16_i64_408 : i64
        %c0_i64_409 = arith.constant 0 : i64
        %215 = arith.addi %214, %c0_i64_409 : i64
        %216 = arith.addi %215, %c0_i64_409 : i64
        %217 = cute.ptrtoint(%iter_2) : !cute.ptr<bf16, gmem, align<16>> to i64
        %218 = arith.muli %216, %c16_i64_408 : i64
        %c8_i64_410 = arith.constant 8 : i64
        %219 = arith.floordivsi %218, %c8_i64_410 : i64
        %220 = arith.addi %217, %219 : i64
        %iv_411 = cute.assume(%220) : (i64) -> !cute.i64<divby 16>
        %221 = cute.inttoptr(%iv_411) : !cute.i64<divby 16> to !cute.ptr<bf16, gmem, align<16>>
        %lay_412 = cute.get_layout(%arg0) : !memref_gmem_bf16_
        %view_413 = cute.make_view(%221, %lay_412) : !memref_gmem_bf16_
        %iter_414 = cute.get_iter(%view_413) : !memref_gmem_bf16_
        %coord_415 = cute.make_coord(%64) : (i32) -> !cute.coord<"(0,?)">
        %222 = cute.memref.load(%view_413, %coord_415) : (!memref_gmem_bf16_, !cute.coord<"(0,?)">) -> bf16
        %223 = arith.extf %222 : bf16 to f32
        %lay_416 = cute.get_layout(%view_413) : !memref_gmem_bf16_
        %224 = cute.get_shape(%lay_416) : (!cute.layout<"(?,16):(16,1)">) -> !cute.shape<"(?,16)">
        %e0_417, %e1_418 = cute.get_leaves(%224) : !cute.shape<"(?,16)">
        %itup_419 = cute.to_int_tuple(%e0_417) : !cute.shape<"?"> to !cute.int_tuple<"?">
        %225 = cute.get_scalars(%itup_419) : !cute.int_tuple<"?">
        %226 = cute.get_stride(%lay_416) : (!cute.layout<"(?,16):(16,1)">) -> !cute.stride<"(16,1)">
        %e0_420, %e1_421 = cute.get_leaves(%226) : !cute.stride<"(16,1)">
        %lay_422 = cute.get_layout(%view_413) : !memref_gmem_bf16_
        %227 = cute.get_shape(%lay_422) : (!cute.layout<"(?,16):(16,1)">) -> !cute.shape<"(?,16)">
        %e0_423, %e1_424 = cute.get_leaves(%227) : !cute.shape<"(?,16)">
        %itup_425 = cute.to_int_tuple(%e0_423) : !cute.shape<"?"> to !cute.int_tuple<"?">
        %228 = cute.get_scalars(%itup_425) : !cute.int_tuple<"?">
        %229 = cute.get_stride(%lay_422) : (!cute.layout<"(?,16):(16,1)">) -> !cute.stride<"(16,1)">
        %e0_426, %e1_427 = cute.get_leaves(%229) : !cute.stride<"(16,1)">
        %lay_428 = cute.get_layout(%view_413) : !memref_gmem_bf16_
        %230 = cute.get_shape(%lay_428) : (!cute.layout<"(?,16):(16,1)">) -> !cute.shape<"(?,16)">
        %e0_429, %e1_430 = cute.get_leaves(%230) : !cute.shape<"(?,16)">
        %itup_431 = cute.to_int_tuple(%e0_429) : !cute.shape<"?"> to !cute.int_tuple<"?">
        %231 = cute.get_scalars(%itup_431) : !cute.int_tuple<"?">
        %232 = cute.get_stride(%lay_428) : (!cute.layout<"(?,16):(16,1)">) -> !cute.stride<"(16,1)">
        %e0_432, %e1_433 = cute.get_leaves(%232) : !cute.stride<"(16,1)">
        %lay_434 = cute.get_layout(%view_413) : !memref_gmem_bf16_
        %233 = cute.get_shape(%lay_434) : (!cute.layout<"(?,16):(16,1)">) -> !cute.shape<"(?,16)">
        %e0_435, %e1_436 = cute.get_leaves(%233) : !cute.shape<"(?,16)">
        %itup_437 = cute.to_int_tuple(%e0_435) : !cute.shape<"?"> to !cute.int_tuple<"?">
        %234 = cute.get_scalars(%itup_437) : !cute.int_tuple<"?">
        %235 = cute.get_stride(%lay_434) : (!cute.layout<"(?,16):(16,1)">) -> !cute.stride<"(16,1)">
        %e0_438, %e1_439 = cute.get_leaves(%235) : !cute.stride<"(16,1)">
        scf.yield %view_413, %223 : !memref_gmem_bf16_, f32
      } else {
        %iter_392 = cute.get_iter(%view) : !memref_gmem_bf16_
        %lay_393 = cute.get_layout(%view) : !memref_gmem_bf16_
        %206 = cute.get_shape(%lay_393) : (!cute.layout<"(?,16):(16,1)">) -> !cute.shape<"(?,16)">
        %e0_394, %e1_395 = cute.get_leaves(%206) : !cute.shape<"(?,16)">
        %itup_396 = cute.to_int_tuple(%e0_394) : !cute.shape<"?"> to !cute.int_tuple<"?">
        %207 = cute.get_scalars(%itup_396) : !cute.int_tuple<"?">
        %208 = cute.get_stride(%lay_393) : (!cute.layout<"(?,16):(16,1)">) -> !cute.stride<"(16,1)">
        %e0_397, %e1_398 = cute.get_leaves(%208) : !cute.stride<"(16,1)">
        %lay_399 = cute.get_layout(%view) : !memref_gmem_bf16_
        %209 = cute.get_shape(%lay_399) : (!cute.layout<"(?,16):(16,1)">) -> !cute.shape<"(?,16)">
        %e0_400, %e1_401 = cute.get_leaves(%209) : !cute.shape<"(?,16)">
        %itup_402 = cute.to_int_tuple(%e0_400) : !cute.shape<"?"> to !cute.int_tuple<"?">
        %210 = cute.get_scalars(%itup_402) : !cute.int_tuple<"?">
        %211 = cute.get_stride(%lay_399) : (!cute.layout<"(?,16):(16,1)">) -> !cute.stride<"(16,1)">
        %e0_403, %e1_404 = cute.get_leaves(%211) : !cute.stride<"(16,1)">
        %lay_405 = cute.get_layout(%view) : !memref_gmem_bf16_
        %212 = cute.get_shape(%lay_405) : (!cute.layout<"(?,16):(16,1)">) -> !cute.shape<"(?,16)">
        %e0_406, %e1_407 = cute.get_leaves(%212) : !cute.shape<"(?,16)">
        %itup_408 = cute.to_int_tuple(%e0_406) : !cute.shape<"?"> to !cute.int_tuple<"?">
        %213 = cute.get_scalars(%itup_408) : !cute.int_tuple<"?">
        %214 = cute.get_stride(%lay_405) : (!cute.layout<"(?,16):(16,1)">) -> !cute.stride<"(16,1)">
        %e0_409, %e1_410 = cute.get_leaves(%214) : !cute.stride<"(16,1)">
        %lay_411 = cute.get_layout(%view) : !memref_gmem_bf16_
        %215 = cute.get_shape(%lay_411) : (!cute.layout<"(?,16):(16,1)">) -> !cute.shape<"(?,16)">
        %e0_412, %e1_413 = cute.get_leaves(%215) : !cute.shape<"(?,16)">
        %itup_414 = cute.to_int_tuple(%e0_412) : !cute.shape<"?"> to !cute.int_tuple<"?">
        %216 = cute.get_scalars(%itup_414) : !cute.int_tuple<"?">
        %217 = cute.get_stride(%lay_411) : (!cute.layout<"(?,16):(16,1)">) -> !cute.stride<"(16,1)">
        %e0_415, %e1_416 = cute.get_leaves(%217) : !cute.stride<"(16,1)">
        %lay_417 = cute.get_layout(%view) : !memref_gmem_bf16_
        %218 = cute.get_shape(%lay_417) : (!cute.layout<"(?,16):(16,1)">) -> !cute.shape<"(?,16)">
        %e0_418, %e1_419 = cute.get_leaves(%218) : !cute.shape<"(?,16)">
        %itup_420 = cute.to_int_tuple(%e0_418) : !cute.shape<"?"> to !cute.int_tuple<"?">
        %219 = cute.get_scalars(%itup_420) : !cute.int_tuple<"?">
        %220 = cute.get_stride(%lay_417) : (!cute.layout<"(?,16):(16,1)">) -> !cute.stride<"(16,1)">
        %e0_421, %e1_422 = cute.get_leaves(%220) : !cute.stride<"(16,1)">
        %lay_423 = cute.get_layout(%view) : !memref_gmem_bf16_
        %221 = cute.get_shape(%lay_423) : (!cute.layout<"(?,16):(16,1)">) -> !cute.shape<"(?,16)">
        %e0_424, %e1_425 = cute.get_leaves(%221) : !cute.shape<"(?,16)">
        %itup_426 = cute.to_int_tuple(%e0_424) : !cute.shape<"?"> to !cute.int_tuple<"?">
        %222 = cute.get_scalars(%itup_426) : !cute.int_tuple<"?">
        %223 = cute.get_stride(%lay_423) : (!cute.layout<"(?,16):(16,1)">) -> !cute.stride<"(16,1)">
        %e0_427, %e1_428 = cute.get_leaves(%223) : !cute.stride<"(16,1)">
        scf.yield %view, %cst_257 : !memref_gmem_bf16_, f32
      }
      %iter_270 = cute.get_iter(%126#0) : !memref_gmem_bf16_
      %iter_271 = cute.get_iter(%126#0) : !memref_gmem_bf16_
      %lay_272 = cute.get_layout(%126#0) : !memref_gmem_bf16_
      %127 = cute.get_shape(%lay_272) : (!cute.layout<"(?,16):(16,1)">) -> !cute.shape<"(?,16)">
      %e0_273, %e1_274 = cute.get_leaves(%127) : !cute.shape<"(?,16)">
      %itup_275 = cute.to_int_tuple(%e0_273) : !cute.shape<"?"> to !cute.int_tuple<"?">
      %128 = cute.get_scalars(%itup_275) : !cute.int_tuple<"?">
      %129 = cute.get_stride(%lay_272) : (!cute.layout<"(?,16):(16,1)">) -> !cute.stride<"(16,1)">
      %e0_276, %e1_277 = cute.get_leaves(%129) : !cute.stride<"(16,1)">
      %lay_278 = cute.get_layout(%126#0) : !memref_gmem_bf16_
      %130 = cute.get_shape(%lay_278) : (!cute.layout<"(?,16):(16,1)">) -> !cute.shape<"(?,16)">
      %e0_279, %e1_280 = cute.get_leaves(%130) : !cute.shape<"(?,16)">
      %itup_281 = cute.to_int_tuple(%e0_279) : !cute.shape<"?"> to !cute.int_tuple<"?">
      %131 = cute.get_scalars(%itup_281) : !cute.int_tuple<"?">
      %132 = cute.get_stride(%lay_278) : (!cute.layout<"(?,16):(16,1)">) -> !cute.stride<"(16,1)">
      %e0_282, %e1_283 = cute.get_leaves(%132) : !cute.stride<"(16,1)">
      %133 = cute.get_shape(%arg3) : (!cute.layout<"((8,16),(8,1)):((128,1),(16,1024))">) -> !cute.shape<"((8,16),(8,1))">
      %e0_284, %e1_285, %e2_286, %e3_287 = cute.get_leaves(%133) : !cute.shape<"((8,16),(8,1))">
      %coord_288 = cute.make_coord() : () -> !cute.coord<"(_,_,0)">
      %slice_289 = cute.slice(%view_67, %coord_288) : !memref_smem_i64_, !cute.coord<"(_,_,0)">
      %iter_290 = cute.get_iter(%slice_289) : !memref_smem_i64_1
      %iter_291 = cute.get_iter(%slice_289) : !memref_smem_i64_1
      %cst_292 = arith.constant 0xFF800000 : f32
      %134 = vector.reduction <maximumf>, %98, %cst_292 : vector<8xf32> into f32
      %c-1_i32 = arith.constant -1 : i32
      %c1_i32 = arith.constant 1 : i32
      %c31_i32 = arith.constant 31 : i32
      %135 = nvvm.shfl.sync  bfly %c-1_i32, %134, %c1_i32, %c31_i32 : f32 -> f32
      %136 = nvvm.fmax %134, %135
      %c-1_i32_293 = arith.constant -1 : i32
      %c2_i32 = arith.constant 2 : i32
      %c31_i32_294 = arith.constant 31 : i32
      %137 = nvvm.shfl.sync  bfly %c-1_i32_293, %136, %c2_i32, %c31_i32_294 : f32 -> f32
      %138 = nvvm.fmax %136, %137
      %c-1_i32_295 = arith.constant -1 : i32
      %c4_i32 = arith.constant 4 : i32
      %c31_i32_296 = arith.constant 31 : i32
      %139 = nvvm.shfl.sync  bfly %c-1_i32_295, %138, %c4_i32, %c31_i32_296 : f32 -> f32
      %140 = nvvm.fmax %138, %139
      %cst_297 = arith.constant 1.44269502 : f32
      %141 = vector.broadcast %cst_297 : f32 to vector<8xf32>
      %142 = arith.mulf %98, %141 : vector<8xf32>
      %cst_298 = arith.constant 1.44269502 : f32
      %143 = arith.mulf %140, %cst_298 : f32
      %144 = vector.broadcast %143 : f32 to vector<8xf32>
      %145 = arith.subf %142, %144 : vector<8xf32>
      %shape_299 = cute.make_shape() : () -> !cute.shape<"((8,1),1,1)">
      %lay_300 = cute.make_layout(%shape_299) : !cute.layout<"((8,1),1,1):((1,0),0,0)">
      %rmem_301 = cute.memref.alloca(%lay_300) : !memref_rmem_f32_
      %iter_302 = cute.get_iter(%rmem_301) : !memref_rmem_f32_
      %iter_303 = cute.get_iter(%rmem_301) : !memref_rmem_f32_
      %lay_304 = cute.get_layout(%rmem_301) : !memref_rmem_f32_
      %146 = cute.get_shape(%lay_304) : (!cute.layout<"((8,1),1,1):((1,0),0,0)">) -> !cute.shape<"((8,1),1,1)">
      %e0_305, %e1_306, %e2_307, %e3_308 = cute.get_leaves(%146) : !cute.shape<"((8,1),1,1)">
      %lay_309 = cute.get_layout(%rmem_301) : !memref_rmem_f32_
      %147 = cute.get_shape(%lay_309) : (!cute.layout<"((8,1),1,1):((1,0),0,0)">) -> !cute.shape<"((8,1),1,1)">
      %e0_310, %e1_311, %e2_312, %e3_313 = cute.get_leaves(%147) : !cute.shape<"((8,1),1,1)">
      %int_tuple_314 = cute.make_int_tuple() : () -> !cute.int_tuple<"((8,1),1,1)">
      %sz_315 = cute.size(%int_tuple_314) : (!cute.int_tuple<"((8,1),1,1)">) -> !cute.int_tuple<"8">
      %e0_316 = cute.get_leaves(%sz_315) : !cute.int_tuple<"8">
      %int_tuple_317 = cute.make_int_tuple() : () -> !cute.int_tuple<"((8,1),1,1)">
      %sz_318 = cute.size(%int_tuple_317) : (!cute.int_tuple<"((8,1),1,1)">) -> !cute.int_tuple<"8">
      %e0_319 = cute.get_leaves(%sz_318) : !cute.int_tuple<"8">
      cute.memref.store_vec %145, %rmem_301, row_major : !memref_rmem_f32_
      %int_tuple_320 = cute.make_int_tuple() : () -> !cute.int_tuple<"((8,1),1,1)">
      %sz_321 = cute.size(%int_tuple_320) : (!cute.int_tuple<"((8,1),1,1)">) -> !cute.int_tuple<"8">
      %e0_322 = cute.get_leaves(%sz_321) : !cute.int_tuple<"8">
      %c8_i32 = arith.constant 8 : i32
      %c1_i32_323 = arith.constant 1 : i32
      scf.for %arg4 = %c0_i32_111 to %c8_i32 step %c1_i32_323  : i32 {
        %coord_392 = cute.make_coord(%arg4) : (i32) -> !cute.coord<"?">
        %206 = cute.memref.load(%rmem_301, %coord_392) : (!memref_rmem_f32_, !cute.coord<"?">) -> f32
        %207 = llvm.inline_asm has_side_effects asm_dialect = att "ex2.approx.ftz.f32 $0, $1;", "=f,f" %206 : (f32) -> f32
        %coord_393 = cute.make_coord(%arg4) : (i32) -> !cute.coord<"?">
        cute.memref.store(%rmem_301, %coord_393, %207) : (!memref_rmem_f32_, !cute.coord<"?">, f32) -> ()
      } {loop_annotation = #loop_annotation}
      %lay_324 = cute.get_layout(%rmem_301) : !memref_rmem_f32_
      %148 = cute.get_shape(%lay_324) : (!cute.layout<"((8,1),1,1):((1,0),0,0)">) -> !cute.shape<"((8,1),1,1)">
      %e0_325, %e1_326, %e2_327, %e3_328 = cute.get_leaves(%148) : !cute.shape<"((8,1),1,1)">
      %149 = cute.memref.load_vec %rmem_301, row_major : !memref_rmem_f32_
      %lay_329 = cute.get_layout(%rmem_301) : !memref_rmem_f32_
      %150 = cute.get_shape(%lay_329) : (!cute.layout<"((8,1),1,1):((1,0),0,0)">) -> !cute.shape<"((8,1),1,1)">
      %e0_330, %e1_331, %e2_332, %e3_333 = cute.get_leaves(%150) : !cute.shape<"((8,1),1,1)">
      %151 = vector.reduction <add>, %149, %cst_257 : vector<8xf32> into f32
      %c-1_i32_334 = arith.constant -1 : i32
      %c1_i32_335 = arith.constant 1 : i32
      %c31_i32_336 = arith.constant 31 : i32
      %152 = nvvm.shfl.sync  bfly %c-1_i32_334, %151, %c1_i32_335, %c31_i32_336 : f32 -> f32
      %153 = arith.addf %152, %151 : f32
      %c-1_i32_337 = arith.constant -1 : i32
      %c2_i32_338 = arith.constant 2 : i32
      %c31_i32_339 = arith.constant 31 : i32
      %154 = nvvm.shfl.sync  bfly %c-1_i32_337, %153, %c2_i32_338, %c31_i32_339 : f32 -> f32
      %155 = arith.addf %153, %154 : f32
      %c-1_i32_340 = arith.constant -1 : i32
      %c4_i32_341 = arith.constant 4 : i32
      %c31_i32_342 = arith.constant 31 : i32
      %156 = nvvm.shfl.sync  bfly %c-1_i32_340, %155, %c4_i32_341, %c31_i32_342 : f32 -> f32
      %157 = arith.addf %155, %156 : f32
      %lay_343 = cute.get_layout(%slice_289) : !memref_smem_i64_1
      %158 = cute.get_shape(%lay_343) : (!cute.layout<"(4,(1,1)):(1,(0,0))">) -> !cute.shape<"(4,(1,1))">
      %e0_344, %e1_345, %e2_346 = cute.get_leaves(%158) : !cute.shape<"(4,(1,1))">
      %coord_347 = cute.make_coord() : () -> !cute.coord<"0">
      %slice_348 = cute.slice(%slice, %coord_347) : !cute.coord_tensor<"(?,?{div=8})", "(1,1,1):(0,0,0)">, !cute.coord<"0">
      %iter_349 = cute.get_iter(%slice_348) : !cute.coord_tensor<"(?,?{div=8})", "():()">
      %e0_350, %e1_351 = cute.get_leaves(%iter_349) : !cute.int_tuple<"(?,?{div=8})">
      %159 = cute.get_scalars(%e0_350) : !cute.int_tuple<"?">
      %160 = cute.get_scalars(%e1_351) : !cute.int_tuple<"?{div=8}">
      %iter_352 = cute.get_iter(%slice_348) : !cute.coord_tensor<"(?,?{div=8})", "():()">
      %e0_353, %e1_354 = cute.get_leaves(%iter_352) : !cute.int_tuple<"(?,?{div=8})">
      %161 = cute.get_scalars(%e0_353) : !cute.int_tuple<"?">
      %162 = cute.get_scalars(%e1_354) : !cute.int_tuple<"?{div=8}">
      %iter_355 = cute.get_iter(%slice_348) : !cute.coord_tensor<"(?,?{div=8})", "():()">
      %e0_356, %e1_357 = cute.get_leaves(%iter_355) : !cute.int_tuple<"(?,?{div=8})">
      %163 = cute.get_scalars(%e0_356) : !cute.int_tuple<"?">
      %164 = cute.get_scalars(%e1_357) : !cute.int_tuple<"?{div=8}">
      %165 = arith.cmpi eq, %164, %c0_i32 : i32
      %coord_358 = cute.make_coord() : () -> !cute.coord<"0">
      %slice_359 = cute.slice(%slice, %coord_358) : !cute.coord_tensor<"(?,?{div=8})", "(1,1,1):(0,0,0)">, !cute.coord<"0">
      %iter_360 = cute.get_iter(%slice_359) : !cute.coord_tensor<"(?,?{div=8})", "():()">
      %e0_361, %e1_362 = cute.get_leaves(%iter_360) : !cute.int_tuple<"(?,?{div=8})">
      %166 = cute.get_scalars(%e0_361) : !cute.int_tuple<"?">
      %167 = cute.get_scalars(%e1_362) : !cute.int_tuple<"?{div=8}">
      %iter_363 = cute.get_iter(%slice_359) : !cute.coord_tensor<"(?,?{div=8})", "():()">
      %e0_364, %e1_365 = cute.get_leaves(%iter_363) : !cute.int_tuple<"(?,?{div=8})">
      %168 = cute.get_scalars(%e0_364) : !cute.int_tuple<"?">
      %169 = cute.get_scalars(%e1_365) : !cute.int_tuple<"?{div=8}">
      %iter_366 = cute.get_iter(%slice_359) : !cute.coord_tensor<"(?,?{div=8})", "():()">
      %e0_367, %e1_368 = cute.get_leaves(%iter_366) : !cute.int_tuple<"(?,?{div=8})">
      %170 = cute.get_scalars(%e0_367) : !cute.int_tuple<"?">
      %171 = cute.get_scalars(%e1_368) : !cute.int_tuple<"?{div=8}">
      %172 = arith.cmpi eq, %171, %c0_i32 : i32
      %173 = arith.cmpi slt, %47, %18 : i32
      %174 = arith.extui %172 : i1 to i32
      %175 = arith.cmpi ne, %174, %c0_i32 : i32
      %176 = arith.extui %172 : i1 to i32
      %177 = arith.extui %173 : i1 to i32
      %178 = arith.select %175, %177, %176 : i32
      %179 = arith.cmpi ne, %178, %c0_i32_111 : i32
      %coord_369 = cute.make_coord() : () -> !cute.coord<"0">
      %slice_370 = cute.slice(%slice, %coord_369) : !cute.coord_tensor<"(?,?{div=8})", "(1,1,1):(0,0,0)">, !cute.coord<"0">
      %iter_371 = cute.get_iter(%slice_370) : !cute.coord_tensor<"(?,?{div=8})", "():()">
      %e0_372, %e1_373 = cute.get_leaves(%iter_371) : !cute.int_tuple<"(?,?{div=8})">
      %180 = cute.get_scalars(%e0_372) : !cute.int_tuple<"?">
      %181 = cute.get_scalars(%e1_373) : !cute.int_tuple<"?{div=8}">
      %iter_374 = cute.get_iter(%slice_370) : !cute.coord_tensor<"(?,?{div=8})", "():()">
      %e0_375, %e1_376 = cute.get_leaves(%iter_374) : !cute.int_tuple<"(?,?{div=8})">
      %182 = cute.get_scalars(%e0_375) : !cute.int_tuple<"?">
      %183 = cute.get_scalars(%e1_376) : !cute.int_tuple<"?{div=8}">
      %iter_377 = cute.get_iter(%slice_370) : !cute.coord_tensor<"(?,?{div=8})", "():()">
      %e0_378, %e1_379 = cute.get_leaves(%iter_377) : !cute.int_tuple<"(?,?{div=8})">
      %184 = cute.get_scalars(%e0_378) : !cute.int_tuple<"?">
      %185 = cute.get_scalars(%e1_379) : !cute.int_tuple<"?{div=8}">
      %186 = arith.cmpi eq, %185, %c0_i32 : i32
      %coord_380 = cute.make_coord() : () -> !cute.coord<"0">
      %slice_381 = cute.slice(%slice, %coord_380) : !cute.coord_tensor<"(?,?{div=8})", "(1,1,1):(0,0,0)">, !cute.coord<"0">
      %iter_382 = cute.get_iter(%slice_381) : !cute.coord_tensor<"(?,?{div=8})", "():()">
      %e0_383, %e1_384 = cute.get_leaves(%iter_382) : !cute.int_tuple<"(?,?{div=8})">
      %187 = cute.get_scalars(%e0_383) : !cute.int_tuple<"?">
      %188 = cute.get_scalars(%e1_384) : !cute.int_tuple<"?{div=8}">
      %iter_385 = cute.get_iter(%slice_381) : !cute.coord_tensor<"(?,?{div=8})", "():()">
      %e0_386, %e1_387 = cute.get_leaves(%iter_385) : !cute.int_tuple<"(?,?{div=8})">
      %189 = cute.get_scalars(%e0_386) : !cute.int_tuple<"?">
      %190 = cute.get_scalars(%e1_387) : !cute.int_tuple<"?{div=8}">
      %iter_388 = cute.get_iter(%slice_381) : !cute.coord_tensor<"(?,?{div=8})", "():()">
      %e0_389, %e1_390 = cute.get_leaves(%iter_388) : !cute.int_tuple<"(?,?{div=8})">
      %191 = cute.get_scalars(%e0_389) : !cute.int_tuple<"?">
      %192 = cute.get_scalars(%e1_390) : !cute.int_tuple<"?{div=8}">
      %193 = arith.cmpi eq, %192, %c0_i32 : i32
      %194 = arith.cmpi slt, %47, %18 : i32
      %195 = arith.extui %193 : i1 to i32
      %196 = arith.cmpi ne, %195, %c0_i32 : i32
      %197 = arith.extui %193 : i1 to i32
      %198 = arith.extui %194 : i1 to i32
      %199 = arith.select %196, %198, %197 : i32
      %200 = arith.cmpi ne, %199, %c0_i32_111 : i32
      %201 = arith.extui %200 : i1 to i32
      %202 = arith.cmpi ne, %201, %c0_i32 : i32
      %203 = arith.extui %200 : i1 to i32
      %c1_i32_391 = arith.constant 1 : i32
      %204 = arith.select %202, %c1_i32_391, %203 : i32
      %205 = arith.cmpi ne, %204, %c0_i32_111 : i32
      scf.if %205 {
        %206 = llvm.inline_asm asm_dialect = att "lg2.approx.ftz.f32 $0, $1;", "=f,f" %157 : (f32) -> f32
        %cst_392 = arith.constant 0.693147182 : f32
        %207 = arith.mulf %206, %cst_392 : f32
        %208 = arith.addf %140, %207 : f32
        %209 = arith.subf %208, %126#1 : f32
        %coord_393 = cute.make_coord(%e0_98) : (!cute.int_tuple<"?">) -> !cute.coord<"?">
        cute.memref.store(%arg2, %coord_393, %209) : (!memref_gmem_f32_, !cute.coord<"?">, f32) -> ()
      }
      return
    }
  }
  func.func @cutlass___call_____main__CrossEntropy_object_at__Tensorgmemo16161_Tensorgmemo1_Tensorgmemo1_None_CUstream0x0(%arg0: !memref_gmem_bf16_, %arg1: !memref_gmem_i64_, %arg2: !memref_gmem_f32_, %arg3: !gpu.async.token) attributes {llvm.emit_c_interface} {
    %iter = cute.get_iter(%arg0) : !memref_gmem_bf16_
    %iter_0 = cute.get_iter(%arg1) : !memref_gmem_i64_
    %iter_1 = cute.get_iter(%arg2) : !memref_gmem_f32_
    %iter_2 = cute.get_iter(%arg0) : !memref_gmem_bf16_
    %iter_3 = cute.get_iter(%arg1) : !memref_gmem_i64_
    %iter_4 = cute.get_iter(%arg2) : !memref_gmem_f32_
    %lay = cute.get_layout(%arg0) : !memref_gmem_bf16_
    %0 = cute.get_shape(%lay) : (!cute.layout<"(?,16):(16,1)">) -> !cute.shape<"(?,16)">
    %e0, %e1 = cute.get_leaves(%0) : !cute.shape<"(?,16)">
    %itup = cute.to_int_tuple(%e0) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %1 = cute.get_scalars(%itup) : !cute.int_tuple<"?">
    %2 = cute.get_stride(%lay) : (!cute.layout<"(?,16):(16,1)">) -> !cute.stride<"(16,1)">
    %e0_5, %e1_6 = cute.get_leaves(%2) : !cute.stride<"(16,1)">
    %lay_7 = cute.get_layout(%arg1) : !memref_gmem_i64_
    %3 = cute.get_shape(%lay_7) : (!cute.layout<"(?):(1)">) -> !cute.shape<"(?)">
    %e0_8 = cute.get_leaves(%3) : !cute.shape<"(?)">
    %itup_9 = cute.to_int_tuple(%e0_8) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %4 = cute.get_scalars(%itup_9) : !cute.int_tuple<"?">
    %5 = cute.get_stride(%lay_7) : (!cute.layout<"(?):(1)">) -> !cute.stride<"(1)">
    %e0_10 = cute.get_leaves(%5) : !cute.stride<"(1)">
    %lay_11 = cute.get_layout(%arg2) : !memref_gmem_f32_
    %6 = cute.get_shape(%lay_11) : (!cute.layout<"(?):(1)">) -> !cute.shape<"(?)">
    %e0_12 = cute.get_leaves(%6) : !cute.shape<"(?)">
    %itup_13 = cute.to_int_tuple(%e0_12) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %7 = cute.get_scalars(%itup_13) : !cute.int_tuple<"?">
    %8 = cute.get_stride(%lay_11) : (!cute.layout<"(?):(1)">) -> !cute.stride<"(1)">
    %e0_14 = cute.get_leaves(%8) : !cute.stride<"(1)">
    %shape = cute.make_shape() : () -> !cute.shape<"2">
    %tile = cute.make_tile() : () -> !cute.tile<"8:1">
    %shp = cute.ceil_div(%shape, %tile) : !cute.shape<"2">, !cute.tile<"8:1">
    %e0_15 = cute.get_leaves(%shp) : !cute.shape<"1">
    %shape_16 = cute.make_shape() : () -> !cute.shape<"((8,16),(8,1))">
    %stride = cute.make_stride() : () -> !cute.stride<"((128,1),(16,1024))">
    %lay_17 = cute.make_layout(%shape_16, %stride) : !cute.layout<"((8,16),(8,1)):((128,1),(16,1024))">
    %sz = cute.size(%lay_17) <{mode = [0]}> : (!cute.layout<"((8,16),(8,1)):((128,1),(16,1024))">) -> !cute.int_tuple<"128">
    %e0_18 = cute.get_leaves(%sz) : !cute.int_tuple<"128">
    %lay_19 = cute.get_layout(%arg0) : !memref_gmem_bf16_
    %9 = cute.get_shape(%lay_19) : (!cute.layout<"(?,16):(16,1)">) -> !cute.shape<"(?,16)">
    %e0_20, %e1_21 = cute.get_leaves(%9) : !cute.shape<"(?,16)">
    %itup_22 = cute.to_int_tuple(%e0_20) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %10 = cute.get_scalars(%itup_22) : !cute.int_tuple<"?">
    %shape_23 = cute.make_shape(%itup_22) : (!cute.int_tuple<"?">) -> !cute.shape<"?">
    %tile_24 = cute.make_tile() : () -> !cute.tile<"16:1">
    %shp_25 = cute.ceil_div(%shape_23, %tile_24) : !cute.shape<"?">, !cute.tile<"16:1">
    %e0_26 = cute.get_leaves(%shp_25) : !cute.shape<"?">
    %itup_27 = cute.to_int_tuple(%e0_26) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %11 = cute.get_scalars(%itup_27) : !cute.int_tuple<"?">
    %shape_28 = cute.make_shape() : () -> !cute.shape<"(16,64)">
    %lay_29 = cute.make_layout(%shape_28) : !cute.layout<"(16,64):(1,16)">
    %cosz = cute.cosize(%lay_29) : (!cute.layout<"(16,64):(1,16)">) -> !cute.int_tuple<"1024">
    %e0_30 = cute.get_leaves(%cosz) : !cute.int_tuple<"1024">
    %lay_31 = cute.get_layout(%arg0) : !memref_gmem_bf16_
    %12 = cute.get_shape(%lay_31) : (!cute.layout<"(?,16):(16,1)">) -> !cute.shape<"(?,16)">
    %e0_32, %e1_33 = cute.get_leaves(%12) : !cute.shape<"(?,16)">
    %itup_34 = cute.to_int_tuple(%e0_32) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %13 = cute.get_scalars(%itup_34) : !cute.int_tuple<"?">
    %14 = cute.get_stride(%lay_31) : (!cute.layout<"(?,16):(16,1)">) -> !cute.stride<"(16,1)">
    %e0_35, %e1_36 = cute.get_leaves(%14) : !cute.stride<"(16,1)">
    %lay_37 = cute.get_layout(%arg1) : !memref_gmem_i64_
    %15 = cute.get_shape(%lay_37) : (!cute.layout<"(?):(1)">) -> !cute.shape<"(?)">
    %e0_38 = cute.get_leaves(%15) : !cute.shape<"(?)">
    %itup_39 = cute.to_int_tuple(%e0_38) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %16 = cute.get_scalars(%itup_39) : !cute.int_tuple<"?">
    %17 = cute.get_stride(%lay_37) : (!cute.layout<"(?):(1)">) -> !cute.stride<"(1)">
    %e0_40 = cute.get_leaves(%17) : !cute.stride<"(1)">
    %lay_41 = cute.get_layout(%arg2) : !memref_gmem_f32_
    %18 = cute.get_shape(%lay_41) : (!cute.layout<"(?):(1)">) -> !cute.shape<"(?)">
    %e0_42 = cute.get_leaves(%18) : !cute.shape<"(?)">
    %itup_43 = cute.to_int_tuple(%e0_42) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %19 = cute.get_scalars(%itup_43) : !cute.int_tuple<"?">
    %20 = cute.get_stride(%lay_41) : (!cute.layout<"(?):(1)">) -> !cute.stride<"(1)">
    %e0_44 = cute.get_leaves(%20) : !cute.stride<"(1)">
    %21 = cute.get_shape(%lay_17) : (!cute.layout<"((8,16),(8,1)):((128,1),(16,1024))">) -> !cute.shape<"((8,16),(8,1))">
    %e0_45, %e1_46, %e2, %e3 = cute.get_leaves(%21) : !cute.shape<"((8,16),(8,1))">
    %22 = cute.get_stride(%lay_17) : (!cute.layout<"((8,16),(8,1)):((128,1),(16,1024))">) -> !cute.stride<"((128,1),(16,1024))">
    %e0_47, %e1_48, %e2_49, %e3_50 = cute.get_leaves(%22) : !cute.stride<"((128,1),(16,1024))">
    %23 = arith.index_cast %11 : i32 to index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c2088_i32 = arith.constant 2088 : i32
    %24 = gpu.launch_func async [%arg3] @kernels::@kernel_cutlass_kernel___main__CrossEntropy_object_at__tensorptrbf16gmemalign16o16161_tensorptri64gmemo1_tensorptrf32gmemo1_None_16_64_0 blocks in (%23, %c1, %c1) threads in (%c128, %c1, %c1)  dynamic_shared_memory_size %c2088_i32 args(%arg0 : !memref_gmem_bf16_, %arg1 : !memref_gmem_i64_, %arg2 : !memref_gmem_f32_, %lay_17 : !cute.layout<"((8,16),(8,1)):((128,1),(16,1024))">)
    return
  }
}

