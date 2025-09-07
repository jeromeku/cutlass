
# Write a Makefile for the cutlass_learn directory
# The cutlass repo is in the external directory: ../external/cutlass/include

# Set the CUTLASS_REPO_PATH to the path of the cutlass repo
CUTLASS_REPO_PATH = /home/jerome/cutlass
CUTLASS_HEADERS = ${CUTLASS_REPO_PATH}/include 
CUTLASS_TOOL_HEADERS = ${CUTLASS_REPO_PATH}/tools/util/include
CUTLASS_TEST_HEADERS = ${CUTLASS_REPO_PATH}/test/unit/common
CUDA_ROOT = /home/jeromeku/cuda-toolkit
CXX_FLAGS=--expt-relaxed-constexpr -std=c++17
DEBUG_FLAGS=-G -g -O0  #-lineinfo
CUTLASS_DEBUG_FLAGS=-DCUTLASS_DEBUG_TRACE_LEVEL=2
PROFILE_FLAGS=-lineinfo -g
CUDA_ARCH=compute_100a
CUDA_CODE=sm_100a
ARCH_FLAGS=--generate-code=arch=${CUDA_ARCH},code=${CUDA_CODE}
TMA_DEBUG_FLAG=-DPRINT_TMA_DESCRIPTOR

.PHONY: all clean

all: clean tma_multicast_sm100

tma_multicast_sm100: tma_multicast_sm100.cu
	nvcc ${ARCH_FLAGS} ${TMA_DEBUG_FLAG} ${CUTLASS_DEBUG_FLAGS} ${CXX_FLAGS} -I${CUTLASS_HEADERS} -I${CUTLASS_TOOL_HEADERS} -I${CUTLASS_TEST_HEADERS} -DCUTLASS_ARCH_MMA_SM100_SUPPORTED -o $@ $<

multicast: multicast.cu
	nvcc ${ARCH_FLAGS} ${TMA_DEBUG_FLAG} ${CUTLASS_DEBUG_FLAGS} ${CXX_FLAGS} -I${CUTLASS_HEADERS} -I${CUTLASS_TOOL_HEADERS} -I${CUTLASS_TEST_HEADERS} -o $@ $<

cute_tile: cute_tile.cu
	nvcc ${ARCH_FLAGS} ${TMA_DEBUG_FLAG} ${CUTLASS_DEBUG_FLAGS} ${CXX_FLAGS} -I${CUTLASS_HEADERS} -o $@ $<

clean:
	rm -rf multicast