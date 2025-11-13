!memref_gmem_f16_ = !cute.memref<f16, gmem, align<16>, "(2048,2048):(2048,1)">
module attributes {gpu.container_module} {
  gpu.module @kernels {
    func.func public @kernel_cutlass_naive_elementwise_add_kernel_tensorptrf16gmemalign16o2048204820481_tensorptrf16gmemalign16o2048204820481_tensorptrf16gmemalign16o2048204820481_0(%arg0: !memref_gmem_f16_, %arg1: !memref_gmem_f16_, %arg2: !memref_gmem_f16_) attributes {cute.kernel, gpu.kernel, nvvm.reqntid = array<i32: 256, 1, 1>} {
      %iter = cute.get_iter(%arg0) : !memref_gmem_f16_
      %iter_0 = cute.get_iter(%arg1) : !memref_gmem_f16_
      %iter_1 = cute.get_iter(%arg2) : !memref_gmem_f16_
      %iter_2 = cute.get_iter(%arg0) : !memref_gmem_f16_
      %iter_3 = cute.get_iter(%arg1) : !memref_gmem_f16_
      %iter_4 = cute.get_iter(%arg2) : !memref_gmem_f16_
      %lay = cute.get_layout(%arg0) : !memref_gmem_f16_
      %0 = cute.get_shape(%lay) : (!cute.layout<"(2048,2048):(2048,1)">) -> !cute.shape<"(2048,2048)">
      %e0, %e1 = cute.get_leaves(%0) : !cute.shape<"(2048,2048)">
      %1 = cute.get_stride(%lay) : (!cute.layout<"(2048,2048):(2048,1)">) -> !cute.stride<"(2048,1)">
      %e0_5, %e1_6 = cute.get_leaves(%1) : !cute.stride<"(2048,1)">
      %lay_7 = cute.get_layout(%arg1) : !memref_gmem_f16_
      %2 = cute.get_shape(%lay_7) : (!cute.layout<"(2048,2048):(2048,1)">) -> !cute.shape<"(2048,2048)">
      %e0_8, %e1_9 = cute.get_leaves(%2) : !cute.shape<"(2048,2048)">
      %3 = cute.get_stride(%lay_7) : (!cute.layout<"(2048,2048):(2048,1)">) -> !cute.stride<"(2048,1)">
      %e0_10, %e1_11 = cute.get_leaves(%3) : !cute.stride<"(2048,1)">
      %lay_12 = cute.get_layout(%arg2) : !memref_gmem_f16_
      %4 = cute.get_shape(%lay_12) : (!cute.layout<"(2048,2048):(2048,1)">) -> !cute.shape<"(2048,2048)">
      %e0_13, %e1_14 = cute.get_leaves(%4) : !cute.shape<"(2048,2048)">
      %5 = cute.get_stride(%lay_12) : (!cute.layout<"(2048,2048):(2048,1)">) -> !cute.stride<"(2048,1)">
      %e0_15, %e1_16 = cute.get_leaves(%5) : !cute.stride<"(2048,1)">
      %6 = nvvm.read.ptx.sreg.tid.x : i32
      %7 = nvvm.read.ptx.sreg.tid.y : i32
      %8 = nvvm.read.ptx.sreg.tid.z : i32
      %9 = nvvm.read.ptx.sreg.ctaid.x : i32
      %10 = nvvm.read.ptx.sreg.ctaid.y : i32
      %11 = nvvm.read.ptx.sreg.ctaid.z : i32
      %12 = nvvm.read.ptx.sreg.ntid.x : i32
      %13 = nvvm.read.ptx.sreg.ntid.y : i32
      %14 = nvvm.read.ptx.sreg.ntid.z : i32
      %15 = arith.muli %9, %12 : i32
      %16 = arith.addi %15, %6 : i32
      %lay_17 = cute.get_layout(%arg0) : !memref_gmem_f16_
      %17 = cute.get_shape(%lay_17) : (!cute.layout<"(2048,2048):(2048,1)">) -> !cute.shape<"(2048,2048)">
      %e0_18, %e1_19 = cute.get_leaves(%17) : !cute.shape<"(2048,2048)">
      %c2048_i32 = arith.constant 2048 : i32
      %18 = arith.remsi %16, %c2048_i32 : i32
      %19 = arith.floordivsi %16, %c2048_i32 : i32
      %coord = cute.make_coord(%19, %18) : (i32, i32) -> !cute.coord<"(?,?)">
      %20 = cute.memref.load(%arg0, %coord) : (!memref_gmem_f16_, !cute.coord<"(?,?)">) -> f16
      %coord_20 = cute.make_coord(%19, %18) : (i32, i32) -> !cute.coord<"(?,?)">
      %21 = cute.memref.load(%arg1, %coord_20) : (!memref_gmem_f16_, !cute.coord<"(?,?)">) -> f16
      %22 = arith.addf %20, %21 : f16
      %coord_21 = cute.make_coord(%19, %18) : (i32, i32) -> !cute.coord<"(?,?)">
      cute.memref.store(%arg2, %coord_21, %22) : (!memref_gmem_f16_, !cute.coord<"(?,?)">, f16) -> ()
      return
    }
  }
  func.func @cutlass_naive_elementwise_add_Tensorgmemo2048204820481_Tensorgmemo2048204820481_Tensorgmemo2048204820481(%arg0: !memref_gmem_f16_, %arg1: !memref_gmem_f16_, %arg2: !memref_gmem_f16_) attributes {llvm.emit_c_interface} {
    %iter = cute.get_iter(%arg0) : !memref_gmem_f16_
    %iter_0 = cute.get_iter(%arg1) : !memref_gmem_f16_
    %iter_1 = cute.get_iter(%arg2) : !memref_gmem_f16_
    %iter_2 = cute.get_iter(%arg0) : !memref_gmem_f16_
    %iter_3 = cute.get_iter(%arg1) : !memref_gmem_f16_
    %iter_4 = cute.get_iter(%arg2) : !memref_gmem_f16_
    %lay = cute.get_layout(%arg0) : !memref_gmem_f16_
    %0 = cute.get_shape(%lay) : (!cute.layout<"(2048,2048):(2048,1)">) -> !cute.shape<"(2048,2048)">
    %e0, %e1 = cute.get_leaves(%0) : !cute.shape<"(2048,2048)">
    %1 = cute.get_stride(%lay) : (!cute.layout<"(2048,2048):(2048,1)">) -> !cute.stride<"(2048,1)">
    %e0_5, %e1_6 = cute.get_leaves(%1) : !cute.stride<"(2048,1)">
    %lay_7 = cute.get_layout(%arg1) : !memref_gmem_f16_
    %2 = cute.get_shape(%lay_7) : (!cute.layout<"(2048,2048):(2048,1)">) -> !cute.shape<"(2048,2048)">
    %e0_8, %e1_9 = cute.get_leaves(%2) : !cute.shape<"(2048,2048)">
    %3 = cute.get_stride(%lay_7) : (!cute.layout<"(2048,2048):(2048,1)">) -> !cute.stride<"(2048,1)">
    %e0_10, %e1_11 = cute.get_leaves(%3) : !cute.stride<"(2048,1)">
    %lay_12 = cute.get_layout(%arg2) : !memref_gmem_f16_
    %4 = cute.get_shape(%lay_12) : (!cute.layout<"(2048,2048):(2048,1)">) -> !cute.shape<"(2048,2048)">
    %e0_13, %e1_14 = cute.get_leaves(%4) : !cute.shape<"(2048,2048)">
    %5 = cute.get_stride(%lay_12) : (!cute.layout<"(2048,2048):(2048,1)">) -> !cute.stride<"(2048,1)">
    %e0_15, %e1_16 = cute.get_leaves(%5) : !cute.stride<"(2048,1)">
    %lay_17 = cute.get_layout(%arg0) : !memref_gmem_f16_
    %6 = cute.get_shape(%lay_17) : (!cute.layout<"(2048,2048):(2048,1)">) -> !cute.shape<"(2048,2048)">
    %e0_18, %e1_19 = cute.get_leaves(%6) : !cute.shape<"(2048,2048)">
    %lay_20 = cute.get_layout(%arg0) : !memref_gmem_f16_
    %7 = cute.get_shape(%lay_20) : (!cute.layout<"(2048,2048):(2048,1)">) -> !cute.shape<"(2048,2048)">
    %e0_21, %e1_22 = cute.get_leaves(%7) : !cute.shape<"(2048,2048)">
    %8 = cute.get_stride(%lay_20) : (!cute.layout<"(2048,2048):(2048,1)">) -> !cute.stride<"(2048,1)">
    %e0_23, %e1_24 = cute.get_leaves(%8) : !cute.stride<"(2048,1)">
    %lay_25 = cute.get_layout(%arg1) : !memref_gmem_f16_
    %9 = cute.get_shape(%lay_25) : (!cute.layout<"(2048,2048):(2048,1)">) -> !cute.shape<"(2048,2048)">
    %e0_26, %e1_27 = cute.get_leaves(%9) : !cute.shape<"(2048,2048)">
    %10 = cute.get_stride(%lay_25) : (!cute.layout<"(2048,2048):(2048,1)">) -> !cute.stride<"(2048,1)">
    %e0_28, %e1_29 = cute.get_leaves(%10) : !cute.stride<"(2048,1)">
    %lay_30 = cute.get_layout(%arg2) : !memref_gmem_f16_
    %11 = cute.get_shape(%lay_30) : (!cute.layout<"(2048,2048):(2048,1)">) -> !cute.shape<"(2048,2048)">
    %e0_31, %e1_32 = cute.get_leaves(%11) : !cute.shape<"(2048,2048)">
    %12 = cute.get_stride(%lay_30) : (!cute.layout<"(2048,2048):(2048,1)">) -> !cute.stride<"(2048,1)">
    %e0_33, %e1_34 = cute.get_leaves(%12) : !cute.stride<"(2048,1)">
    %c16384 = arith.constant 16384 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c0_i32 = arith.constant 0 : i32
    gpu.launch_func  @kernels::@kernel_cutlass_naive_elementwise_add_kernel_tensorptrf16gmemalign16o2048204820481_tensorptrf16gmemalign16o2048204820481_tensorptrf16gmemalign16o2048204820481_0 blocks in (%c16384, %c1, %c1) threads in (%c256, %c1, %c1)  dynamic_shared_memory_size %c0_i32 args(%arg0 : !memref_gmem_f16_, %arg1 : !memref_gmem_f16_, %arg2 : !memref_gmem_f16_)
    return
  }
}
