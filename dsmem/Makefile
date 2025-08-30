CUTLASS_REPO_PATH = /home/jeromeku/cutlass
CUTLASS_HEADERS = ${CUTLASS_REPO_PATH}/include 
CUTLASS_TOOL_HEADERS = ${CUTLASS_REPO_PATH}/tools/util/include
CUTLASS_TEST_HEADERS = ${CUTLASS_REPO_PATH}/test/unit/common
CUTLASS_DEBUG_FLAGS=-DCUTLASS_DEBUG_TRACE_LEVEL=2

CUDA_ROOT = /home/jeromeku/cuda-toolkit
CUDA_LIB_DIR=-L${CUDA_ROOT}/lib64/stubs/
CXX_FLAGS=--expt-relaxed-constexpr -std=c++17
DEBUG_FLAGS=-G -g -O0
PROFILE_FLAGS=-lineinfo -g
PTXAS_FLAGS=--ptxas-options=-v --ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage
CUDA_ARCH=compute_90a
CUDA_CODE=sm_90a
ARCH_FLAGS=--generate-code=arch=${CUDA_ARCH},code=${CUDA_CODE}
VERBOSE_FLAG=-DDEBUG
.PHONY: all clean

all: clean ptx_store

ptx_store: ptx_store.cu
	nvcc \
	${ARCH_FLAGS} \
	-I${CUDA_ROOT}/include \
	${CXX_FLAGS} \
	-I${CUTLASS_HEADERS} \
	-I${CUTLASS_TOOL_HEADERS} \
	-I${CUTLASS_TEST_HEADERS} \
	-o $@ $<

clean:
	rm -rf ptx_store