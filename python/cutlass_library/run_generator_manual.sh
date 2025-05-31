
#!/bin/bash
set -euo pipefail
#DEBUG_GENERATOR: Namespace(operations='all', build_dir='/home/jeromeku/kernels/cfx-article-src/external/cutlass/profiler_test_build', curr_build_dir='/home/jeromeku/kernels/cfx-article-src/external/cutlass/profiler_test_build/tools/library', generator_target='library', architectures='90a', kernels='cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16*_tn*', ignore_kernels='', exclude_kernels='', filter_by_cc='True', cuda_version='12.8.61', kernel_filter_file='', selected_kernel_list='/home/jeromeku/kernels/cfx-article-src/external/cutlass/profiler_test_build/tools/library/generated_kernels.txt', interface_dir=None, disable_full_archs_compilation=False, log_level=10, instantiation_level='0000', disable_cutlass_package_imports=True)

OPERATIONS='gemm'
BUILD_DIR='./generator_test'
CURR_BUILD_DIR="${BUILD_DIR}/library"

mkdir -p ${CURR_BUILD_DIR}

GENERATOR_TARGET='library'
ARCHITECTURES='90a'
KERNELS='cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f16*_tn*'
#ignore_kernels='', exclude_kernels='', 
FILTER_BY_CC='True'
CUDA_VERSION='12.8.61'
#kernel_filter_file='', 
SELECTED_KERNEL_LIST="${BUILD_DIR}/generated_kernels.txt"
#, interface_dir=None, disable_full_archs_compilation=False, 
LOG_LEVEL=DEBUG
INSTANTIATION_LEVEL='0000'

CMD="python generator.py \
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
eval $CMD