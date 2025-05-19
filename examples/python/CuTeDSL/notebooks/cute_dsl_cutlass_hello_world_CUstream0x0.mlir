module attributes {gpu.container_module} {
  gpu.module @kernels {
    func.func public @kernel_cutlass_hello_world_kernel_0() attributes {cute.kernel, gpu.kernel, nvvm.reqntid = array<i32: 1, 1, 1>} {
      cute.print("Hello world\0A", ) : 
      return
    }
  }
  func.func @cutlass_hello_world_CUstream0x0(%arg0: !gpu.async.token) attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = gpu.launch_func async [%arg0] @kernels::@kernel_cutlass_hello_world_kernel_0 blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)  dynamic_shared_memory_size %c0_i32 
    return
  }
}

