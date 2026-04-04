# Write a Makefile for the cutlass_learn directory
# The cutlass repo is in the external directory: ../external/cutlass/include

# Set the CUTLASS_REPO_PATH to the path of the cutlass repo
CUTLASS_REPO_PATH = /home/jeromeku/cutlass
CUTLASS_HEADERS = ${CUTLASS_REPO_PATH}/include 
CUTLASS_TOOL_HEADERS = ${CUTLASS_REPO_PATH}/tools/util/include
CUTLASS_TEST_HEADERS = ${CUTLASS_REPO_PATH}/test/unit/common
CXX_FLAGS=--expt-relaxed-constexpr -std=c++17
DEBUG_FLAGS=-G -g -O0  #-lineinfo
CUTLASS_DEBUG_FLAGS=-DCUTLASS_DEBUG_TRACE_LEVEL=2
PROFILE_FLAGS=-lineinfo -g
CUDA_ARCH=compute_90a
CUDA_CODE=sm_90a
ARCH_FLAGS=--generate-code=arch=${CUDA_ARCH},code=${CUDA_CODE}

.PHONY: all clean

all: clean layout_algebra

layout_algebra: layout_algebra.cu
	nvcc ${ARCH_FLAGS} ${CUTLASS_DEBUG_FLAGS} ${CXX_FLAGS} -I${CUTLASS_HEADERS} -I${CUTLASS_TOOL_HEADERS} -I${CUTLASS_TEST_HEADERS} -o $@ $<

clean:
	rm -rf layout_algebra