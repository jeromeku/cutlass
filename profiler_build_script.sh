#!/bin/bash

set -euo pipefail
COMMAND=$1


source utils.sh

CUTLASS_LIBRARY_OPERATIONS=gemm
LIBRARY_INSTANTIATION_LEVEL=max
BUILD_TYPE="${LIBRARY_INSTANTIATION_LEVEL}"

 #3310 instruction shape [0-3], mma multiplier [0-3,9], cluster shape [0-5], schedule {0,1}; "max" == 9992, see manifest.py init
KERNEL_NAME="cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16"
LAYOUT="tt"
KERNEL_PATTERN="${KERNEL_NAME}*_${LAYOUT}*"
BUILD_DIR="build_profiler_${BUILD_TYPE}_${KERNEL_NAME}_${LAYOUT}"

ENABLE_SM90_EXTENDED=ON
CUTLASS_PROFILER_DISABLE_REFERENCE=OFF
CUTLASS_BUILD_FOR_PROFILER_REGRESSIONS=OFF
UNITY_BUILD=ON
CUTLASS_GENERATOR_SCRIPT=custom_generator.py

LOG_FILE_STEM=${KERNEL_NAME}_${LAYOUT}

if [[ "${COMMAND}" == "configure" ]]; then
    CMD="cmake -B${BUILD_DIR} -S. \
    -DCUTLASS_NVCC_ARCHS="90a" \
    -DCUTLASS_ENABLE_SM90_EXTENDED_MMA_SHAPES=${ENABLE_SM90_EXTENDED} \
    -DCUTLASS_PROFILER_DISABLE_REFERENCE=${CUTLASS_PROFILER_DISABLE_REFERENCE} \
    -DCUTLASS_BUILD_FOR_PROFILER_REGRESSIONS=${CUTLASS_BUILD_FOR_PROFILER_REGRESSIONS} \
    -DCUTLASS_LIBRARY_OPERATIONS=${CUTLASS_LIBRARY_OPERATIONS} \
    -DCUTLASS_LIBRARY_KERNELS=\"${KERNEL_PATTERN}\" \
    -DCUTLASS_LIBRARY_INSTANTIATION_LEVEL=${LIBRARY_INSTANTIATION_LEVEL} \
    -DCUTLASS_UNITY_BUILD_ENABLED=${UNITY_BUILD} \
    -DCMAKE_VERBOSE_MAKEFILE=ON \
    -DCUTLASS_ENABLE_CUBLAS=ON \
    -DCUTLASS_ENABLE_TESTS=OFF \
    -DCUTLASS_ENABLE_EXAMPLES=OFF \
    -DCUTLASS_GENERATOR_SCRIPT=${CUTLASS_GENERATOR_SCRIPT} \
    -GNinja"

    echo $CMD
    rm -rf ${BUILD_DIR}
    START=$(date +%s)
    eval $CMD 2>&1 | tee ${LOG_FILE_STEM}.config.log
    END=$(date +%s)
    echo "Elapsed time: $((END - START)) seconds"

elif [[ "${COMMAND}" == "build" ]]; then
    BUILD_CMD="cmake --build ${BUILD_DIR} -j4 --target cutlass_profiler --verbose" 
    echo $BUILD_CMD

    START=$(date +%s)
    eval $BUILD_CMD 2>&1 | tee ${LOG_FILE_STEM}.build.log
    END=$(date +%s)

    echo "Elapsed time: $((END - START)) seconds"
else
    echo "Unrecognized command: ${COMMAND}"
    exit 1
fi