#!/bin/bash

set -euo pipefail

BUILD_TYPE="profiler_max"
BUILD_DIR="${BUILD_TYPE}_build"
rm -rf ${BUILD_DIR}

LIBRARY_INSTANTIATION_LEVEL=max #3310 instruction shape [0-3], mma multiplier [0-3,9], cluster shape [0-5], schedule {0,1}
KERNEL_PATTERN="cutlass3x_sm90_tensorop_gemm_f16_f16_f32_*"

CMD="cmake -B${BUILD_DIR} -S. \
  -DCUTLASS_NVCC_ARCHS="90a" \
  -DCUTLASS_LIBRARY_KERNELS=\"${KERNEL_PATTERN}\" \
  -DCUTLASS_LIBRARY_INSTANTIATION_LEVEL=${LIBRARY_INSTANTIATION_LEVEL} \
  -DCUTLASS_UNITY_BUILD_ENABLED=OFF \
  -DCMAKE_VERBOSE_MAKEFILE=ON \
  -DCUTLASS_ENABLE_CUBLAS=ON \
  -DCUTLASS_ENABLE_TESTS=OFF \
  -DCUTLASS_ENABLE_EXAMPLES=OFF\
  -GNinja \
  --debug-output"

echo $CMD
eval $CMD 2>&1 | tee _{BUILD_TYPE}.log