CUTLASS_REPO_PATH = /home/jeromeku/cutlass
CUTLASS_HEADERS = ${CUTLASS_REPO_PATH}/include 

CXX_FLAGS=--expt-relaxed-constexpr -std=c++17
CUDA_ARCH=compute_90a
CUDA_CODE=sm_90a
ARCH_FLAGS=--generate-code=arch=${CUDA_ARCH},code=${CUDA_CODE}

.PHONY: all clean

all: clean dsmem_demo

dsmem_demo: dsmem_demo.cu
	nvcc \
	${ARCH_FLAGS} \
	${CXX_FLAGS} \
	-I${CUTLASS_HEADERS} \
	-o $@ $<

clean:
	rm -rf dsmem_demo