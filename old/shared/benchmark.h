#ifndef __BENCHMARK_H__
#define __BENCHMARK_H__

#include <stdlib.h>
#include <stdint.h>

#ifdef CUDA_ENABLED
#define __makestring( testcase ) #testcase
#include <stdio.h>
#include <cuda.h>

#define cudaBenchmark( device, size, testcase )             \
    do {                                                    \
        cudaSetDevice((device));                            \
        cudaError_t __err = cudaGetLastError();             \
        if (cudaSuccess != __err) {                         \
            fprintf(stderr, "CUDA error: %s:%d: '%s'\n",    \
                    __FILE__, __LINE__,                     \
                    cudaGetErrorString(__err);              \
                   );                                       \
            exit(EXIT_FAILURE);                             \
        }                                                   \
        cudaDeviceSynchronize();                            \
        uint64_t __ts_start = usecs();                      \
        (testcase);                                         \
        cudaDeviceSynchronize();                            \
        uint64_t __ts_end = usecs();                        \
        PrintResult( size, __ts_end - __ts_start,           \
                __makestring(testcase) );                   \
    } while (0)
#undef __makestring
#endif


uint64_t usecs();

void PrintResult(size_t bytes, uint64_t usecs, const char* call);

#endif
