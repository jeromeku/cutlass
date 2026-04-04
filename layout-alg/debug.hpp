#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>

#define PRINT_WITH_CONTEXT(label, expr) \
  do {  \
    printf("%s\n", label); \
    expr; \
    printf("\n"); \
  } while (0)
#define PRINT_TENSOR_META(label, tensor) PRINT_WITH_CONTEXT(label, print(tensor))
#define PRINT_TENSOR(label, tensor) PRINT_WITH_CONTEXT(label, print_tensor(tensor))

#define PRINT_THREAD0(expr) \
  do { \
    if(thread0()){ \
      expr; \
    } \
  } while (0)

#define CUDA_DRIVER_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t err__ = (call);                                                 \
    if (err__ != cudaSuccess) {                                                 \
      std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(err__)         \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;         \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                           \
  } while (0)

#define CUDA_RT_CHECK(call)                                                          \
  do {                                                                          \
    CUresult err__ = (call);                                                    \
    if (err__ != CUDA_SUCCESS) {                                                \
      const char* errStr;                                                       \
      cuGetErrorString(err__, &errStr);                                         \
      std::cerr << "CUDA Driver Error: " << (errStr ? errStr : "Unknown")      \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;         \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                           \
  } while (0)