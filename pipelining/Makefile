# Write a Makefile for the cutlass_learn directory
# The cutlass repo is in the external directory: ../external/cutlass/include

# Set the CUTLASS_REPO_PATH to the path of the cutlass repo
CUTLASS_REPO_PATH = /home/jeromeku/cutlass
CUTLASS_HEADERS = ${CUTLASS_REPO_PATH}/include 
CUTLASS_TOOL_HEADERS = ${CUTLASS_REPO_PATH}/tools/util/include
CUTLASS_TEST_HEADERS = ${CUTLASS_REPO_PATH}/test/unit/common
CUDA_ROOT = /home/jeromeku/cuda-toolkit
CXX_FLAGS=--expt-relaxed-constexpr -std=c++17
DEBUG_FLAGS=-G -g -O0  #-lineinfo
CUTLASS_DEBUG_FLAGS=-DCUTLASS_DEBUG_TRACE_LEVEL=2
PROFILE_FLAGS=-lineinfo -g
CUDA_ARCH=compute_90a
CUDA_CODE=sm_90a
ARCH_FLAGS=--generate-code=arch=${CUDA_ARCH},code=${CUDA_CODE}
TMA_DEBUG_FLAG=-DPRINT_TMA_DESCRIPTOR
PIPELINE_TEST_FLAG=-DCUTLASS_UNIT_TEST_PIPELINE
.PHONY: all clean

all: clean

sequence_barrier: sequence_barrier.cu
	nvcc -DDEBUG_PINGPONG ${ARCH_FLAGS} ${TMA_DEBUG_FLAG} ${CUTLASS_DEBUG_FLAGS} ${CXX_FLAGS} -I${CUTLASS_HEADERS} -I${CUTLASS_TOOL_HEADERS} -I${CUTLASS_TEST_HEADERS} -I${CUDA_ROOT}/include -o $@ $<

pingpong_pipe: pipeline_tma_async_warp_specialized_persistent.cu
	nvcc ${ARCH_FLAGS} ${PIPELINE_TEST_FLAG} ${TMA_DEBUG_FLAG} ${CUTLASS_DEBUG_FLAGS} ${CXX_FLAGS} -I${CUTLASS_HEADERS} -I${CUTLASS_TOOL_HEADERS} -I${CUTLASS_TEST_HEADERS} -I${CUDA_ROOT}/include -o $@ $<

clean:
	rm -rf sequence_barrier pingpong_pipe