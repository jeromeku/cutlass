Please refer to this issue in pytorch: https://github.com/pytorch/pytorch/issues/152668, which is referring to this grouped gemm kernel implemented here:
https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/GroupMM.cu.

The underlying kernel it is calling originates from Cutlass:
./examples/57_hopper_grouped_gemm

Please examine the underlying cutlass kernel and suggest ways of fixing the M=0 hanging problem from **within** the kernel.