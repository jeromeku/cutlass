// cudabind_shim.c — FORCES a relocation to cuInit/cuDeviceGet.
#include <cstddef>
#include <cuda.h>    // declares CUresult cuInit(unsigned)
#include <cuda_device_runtime_api.h>
#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int a[10];
    int* d;
    cudaMalloc((void *)(d), sizeof(int) * 10);
    cudaFree(nullptr);
    // Calling via a normal extern makes the linker emit a relocation.
    // // CUresult r = cuInit(0);
    // // if (r != CUDA_SUCCESS) { fprintf(stderr, "cuInit failed %d\n", (int)r); return (int)r; }
    // // CUdevice dev; 
    // // r = cuDeviceGet(&dev, 0);
    // if (r != CUDA_SUCCESS) { fprintf(stderr, "cuDeviceGet failed %d\n", (int)r); return (int)r; }
    return 0;
}
