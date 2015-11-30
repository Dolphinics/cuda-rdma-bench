#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <sisci_api.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
extern "C" {
#include "common.h"
#include "local.h"
#include "reporting.h"
}



__global__ void gpu_memset(void* buffer, size_t size, uint8_t value)
{
    const int num = gridDim.x * gridDim.y * blockDim.x * blockDim.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int pos = y * (gridDim.x * blockDim.x) + x;

    uint8_t* ptr = (uint8_t*) buffer;

    for (size_t i = pos * (size / num), n = (pos + 1) * (size / num); i < n && i < size; ++i)
    {
        ptr[i] = value;
    }
}



static __host__
void* get_dev_ptr(void* ptr)
{
    cudaPointerAttributes attrs;
    
    cudaError_t status = cudaPointerGetAttributes(&attrs, ptr);
    if (status != cudaSuccess)
    {
        log_error("Unexpected error: %s", cudaGetErrorString(status));
        exit(1);
    }

    unsigned flag = 1;
    CUresult result = cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, (CUdeviceptr) attrs.devicePointer);
    if (result != CUDA_SUCCESS)
    {
        log_error("Unexpected error when setting pointer attribute");
        exit(1);
    }

    return attrs.devicePointer;
}



static sci_callback_action_t notify_connection(void* arg, sci_local_segment_t segment, sci_segment_cb_reason_t reason, unsigned remote_node, unsigned adapter, sci_error_t status)
{
    if (reason == SCI_CB_CONNECT)
    {
        log_info("Got connection from remote cluster node %u on NTB adapter %u", remote_node, adapter);
    }

    return SCI_CALLBACK_CONTINUE;
}



static __host__
sci_local_segment_t create_segment(unsigned id, sci_desc_t sd, unsigned adapter, void* ptr, size_t size)
{
    sci_error_t err;
    sci_local_segment_t segment;

    SCICreateSegment(sd, &segment, id, size, &notify_connection, NULL, SCI_FLAG_USE_CALLBACK | SCI_FLAG_EMPTY, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Couldn't create segment: %s", SCIGetErrorString(err));
        exit(1);
    }

    void* dev_ptr = get_dev_ptr(ptr);
    
    SCIAttachPhysicalMemory(0, dev_ptr, 0, size, segment, SCI_FLAG_CUDA_BUFFER, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Couldn't attach physical memory: %s", SCIGetErrorString(err));
        exit(1);
    }

    SCIPrepareSegment(segment, adapter, 0, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Couldn't prepare segment: %s", SCIGetErrorString(err));
        exit(1);
    }

    return segment;
}



extern "C"
bufhandle_t create_gpu_buffer(sci_desc_t desc, unsigned adapter, int gpu, unsigned id, size_t size, unsigned flags)
{
    cudaError_t err;
    bufhandle_t handle;

    err = cudaSetDevice(gpu);
    if (err != cudaSuccess)
    {
        log_error("Couldn't set GPU: %s", cudaGetErrorString(err));
        exit(1);
    }
    handle.gpu_id = gpu;

    log_debug("Allocating buffer on GPU %d (%lu bytes)", gpu, size);
    if (!!(flags & cudaHostAllocMapped))
    {
        handle.hostmem = 1;
        err = cudaHostAlloc(&handle.buffer, size, flags);
    }
    else
    {
        handle.hostmem = 0;
        err = cudaMalloc(&handle.buffer, size);
    }
    if (err != cudaSuccess)
    {
        log_error("Couldn't allocate memory on GPU: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    handle.size = size;
    handle.segment = create_segment(id, desc, adapter, handle.buffer, size);

    uint8_t value = rand() & 255;

    dim3 grid;
    grid.x = 4;
    grid.y = 4;

    dim3 block;
    block.x = 4;
    block.y = 4;

    log_debug("Filling buffer with value %02x", value);
    gpu_memset<<<grid, block>>>(handle.buffer, size, value);

    cudaDeviceSynchronize();

    return handle;
}



extern "C"
void free_gpu_buffer(bufhandle_t handle)
{
    sci_error_t err;

    do
    {
        SCIRemoveSegment(handle.segment, 0, &err);
    }
    while (err != SCI_ERR_OK);

    cudaSetDevice(handle.gpu_id);
    if (handle.hostmem)
    {
        cudaFreeHost(handle.buffer);
    }
    else
    {
        cudaFree(handle.buffer);
    }
}



extern "C"
uint8_t validate_buffer(bufhandle_t handle)
{
    // TODO: Calculate actual checksum of higher and lower part of buffer

    uint8_t* ptr = (uint8_t*) handle.buffer;

    if (!handle.hostmem)
    {
        cudaError_t err;

        err = cudaHostAlloc(&ptr, handle.size, cudaHostAllocDefault);
        if (err != cudaSuccess)
        {
            log_error("Failed to allocate host buffer: %s", cudaGetErrorString(err));
            exit(1);
        }

        err = cudaMemcpy(ptr, handle.buffer, handle.size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            log_error("Failed to copy from device memory to host memory");
            exit(1);
        }
    }

    uint8_t value = ptr[0];
    for (size_t i = 1; i < handle.size; ++i)
    {
        if (ptr[i] != value)
        {
            log_error("Buffer is garbled");
            break;
        }
    }

    if (!handle.hostmem)
    {
        cudaFreeHost(ptr);
    }

    return value;
}
