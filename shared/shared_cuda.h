#ifndef __SHARED_CUDA_H__
#define __SHARED_CUDA_H__

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>



#define cudaAssert(status)                          \
    do                                              \
    {                                               \
        if (cudaSuccess != (status))                \
        {                                           \
            fprintf(stderr,                         \
                    "%s = %d \"%s\" (%s:%d)\n",     \
                    #status, (status),              \
                    cudaGetErrorString((status)),   \
                    __FILE__, __LINE__);            \
            exit((status));                         \
        }                                           \
    }                                               \
    while (0)

#define cudaCheckError()                            \
    do                                              \
    {                                               \
        cudaError_t __err__ = cudaGetLastError();   \
        if (cudaSuccess != __err__)                 \
        {                                           \
            fprintf(stderr,                         \
                    "CUDA = %d \"%s\" (%s:%d)\n",   \
                    __err__,                        \
                    cudaGetErrorString(__err__),    \
                    __FILE__, __LINE__);            \
            exit(__err__);                          \
        }                                           \
    }                                               \
    while (0)



#ifdef __cplusplus
extern "C" {
#endif

void EnableP2P(int dev, int peer_dev);

void DisableP2P(int dev, int peer_dev);

void* GetGpuDevicePtr(const void* ptr);

int SetSyncMemops(void* gpu_ptr);

int GetGpuCount();

#ifdef __cplusplus
}
#endif

#endif
