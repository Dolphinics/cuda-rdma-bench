#include <cuda.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "shared_cuda.h"

int GetGpuCount()
{
    cudaError_t status;

    int device_count = 0;

    status = cudaGetDeviceCount(&device_count);

    if (cudaSuccess != status)
    {
        return -1;
    }

    return device_count;
}



void* GetGpuDevicePtr(const void* ptr)
{
    cudaPointerAttributes attrs;

    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);

    if (cudaSuccess != err)
    {
        return NULL;
    }

    return attrs.devicePointer;
}



int SetSyncMemops(void* ptr)
{
    uint32_t flag = 1;
    CUresult res = cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, (CUdeviceptr) ptr);

    if (CUDA_SUCCESS != res)
    {
        return -1;
    }

    return 0;
}
