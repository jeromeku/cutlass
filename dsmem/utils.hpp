#include <cuda_runtime.h>
#include <iostream>

#ifndef PRINT_CLUSTER_DELAY_CYCLES
#define PRINT_CLUSTER_DELAY_CYCLES 20000000ULL
#endif

// Print msg prefixed with cluster rank
// Sets delay based on cluster rank to prevent interleaved printing
#define PRINT_CLUSTER(expr)                                                                                    \
    do                                                                                                         \
    {                                                                                                          \
        if (threadIdx.x == 0 && threadIdx.y == 0)                                                              \
        {                                                                                                      \
            int _cluster_rank = cute::block_rank_in_cluster();                                                 \
            unsigned long long _delay_cycles = PRINT_CLUSTER_DELAY_CYCLES * (unsigned long long)_cluster_rank; \
            if (_delay_cycles)                                                                                 \
            {                                                                                                  \
                unsigned long long _start = clock64();                                                         \
                while (clock64() - _start < _delay_cycles)                                                     \
                { /* spin */                                                                                   \
                }                                                                                              \
            }                                                                                                  \
            printf("ClusterRank[%d]: ", _cluster_rank);                                                        \
            expr;                                                                                              \
            printf("\n");                                                                                      \
        }                                                                                                      \
    } while (0)

#define CUDA_CHECK(call)                                                  \
    do                                                                    \
    {                                                                     \
        cudaError_t error = call;                                         \
        if (error != cudaSuccess)                                         \
        {                                                                 \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__  \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)

// Function to check and print last CUDA error
void checkLastCudaError(const char *msg = "CUDA Error")
{
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cerr << msg << ": " << cudaGetErrorString(error) << std::endl;
    }
}
