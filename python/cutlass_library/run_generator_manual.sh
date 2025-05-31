
#!/bin/bash
set -euo pipefail
#DEBUG_GENERATOR: Namespace(operations='all', build_dir='/home/jeromeku/kernels/cfx-article-src/external/cutlass/profiler_test_build', curr_build_dir='/home/jeromeku/kernels/cfx-article-src/external/cutlass/profiler_test_build/tools/library', generator_target='library', architectures='90a', kernels='cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16*_tn*', ignore_kernels='', exclude_kernels='', filter_by_cc='True', cuda_version='12.8.61', kernel_filter_file='', selected_kernel_list='/home/jeromeku/kernels/cfx-article-src/external/cutlass/profiler_test_build/tools/library/generated_kernels.txt', interface_dir=None, disable_full_archs_compilation=False, log_level=10, instantiation_level='0000', disable_cutlass_package_imports=True)
SCRIPT="custom_generator.py"
OPERATIONS='gemm'
INSTANTIATION_LEVEL='0000'
ARCHITECTURES='90a'
TYPES="f16_f16_f32_void_f16" # see cutlass/include/cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp#L126-L127, void -> Element_Source (C operand) => no load in epilogue if void
KERNELS="cutlass3x_sm90_tensorop_gemm_${TYPES}*"
# KERNELS="${KERNELS},*_spgemm_*"
BASE_DIR="generated"

if [[ "${SCRIPT}" == *"custom"* ]]; then
    BASE_DIR="custom_generated"
fi

BUILD_DIR="./${BASE_DIR}/${OPERATIONS}_${ARCHITECTURES}/${TYPES}/${INSTANTIATION_LEVEL}"
CURR_BUILD_DIR="${BUILD_DIR}/library"
mkdir -p ${CURR_BUILD_DIR}

echo "Outputs written to ${BUILD_DIR}..."

GENERATOR_TARGET='library'
#ignore_kernels='', exclude_kernels='', 
FILTER_BY_CC='True'
CUDA_VERSION='12.8.61'
#kernel_filter_file='', 
SELECTED_KERNEL_LIST="${BUILD_DIR}/generated_kernels.txt"
#, interface_dir=None, disable_full_archs_compilation=False, 
LOG_LEVEL=INFO

CMD="python ${SCRIPT} \
    --operations=${OPERATIONS} \
    --disable-cutlass-package-imports \
    --generator-target=library \
    --architectures=${ARCHITECTURES} \
    --build-dir=${BUILD_DIR} \
    --curr-build-dir=${CURR_BUILD_DIR} \
    --kernels=${KERNELS} \
    --filter-by-cc=${FILTER_BY_CC} \
    --cuda-version=${CUDA_VERSION} \
    --selected-kernel-list=${SELECTED_KERNEL_LIST} \
    --log-level=${LOG_LEVEL} \
    --instantiation-level=${INSTANTIATION_LEVEL}"

CMD="PYTHONPATH=`pwd` ${CMD}" 

echo $CMD
eval $CMD 2>&1 | tee ${BUILD_DIR}/gen.log