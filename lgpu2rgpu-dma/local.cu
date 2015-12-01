#include <cuda.h>
#include <sisci_api.h>
#include <stdint.h>
#include <stdlib.h>
extern "C" {
#include "common.h"
#include "local.h"
#include "reporting.h"
}



__global__ void gpu_memset_kernel(void* buf, size_t len, uint8_t val)
{
    const int num = gridDim.x * gridDim.y * blockDim.x * blockDim.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int pos = y * (gridDim.x * blockDim.x) + x;

    uint8_t* ptr = (uint8_t*) buf;

    for (size_t i = pos * (len / num), n = (pos + 1) * (len / num); i < n && i < len; ++i)
    {
        ptr[i] = val;
    }

    __syncthreads();
}



/* Use CUDA kernels to set device buffer */
extern "C"
void gpu_memset(int gpu, void* ptr, size_t len, uint8_t val)
{
    cudaError_t err = cudaSetDevice(gpu);
    if (err != cudaSuccess)
    {
        log_error("Failed to set GPU: %s", cudaGetErrorString(err));
        exit(1);
    }

    dim3 grid;
    grid.x = 4;
    grid.y = 4;

    dim3 block;
    block.x = 4;
    block.y = 4;

    log_debug("Filling buffer with value %02x...", val);

    gpu_memset_kernel<<<grid, block>>>(ptr, len, val);

    cudaDeviceSynchronize();
}



/* Helper function to get the CUDA device pointer */
static __host__
void* phys_addr(void* ptr)
{
    cudaPointerAttributes attrs;

    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
    if (err != cudaSuccess)
    {
        log_error("Failed to get pointer attributes: %s", cudaGetErrorString(err));
        exit(1);
    }

    log_debug("CUDA device buffer %p has device ptr %p", ptr, attrs.devicePointer);
    return attrs.devicePointer;
}



/* Helper function to mark a device buffer as being used for RDMA */
static __host__
void set_sync_memops(void* phys_addr)
{
    unsigned flag = 1;
    
    CUresult err = cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, (CUdeviceptr) phys_addr);
    if (err != CUDA_SUCCESS)
    {
        log_error("Failed to set pointer attribute CU_POINTER_ATTRIBYTE_SYNC_MEMOPS");
        exit(1);
    }
}



/* Allocate device buffer */
extern "C"
void* make_gpu_buffer(int gpu, size_t len)
{
    cudaError_t err = cudaSetDevice(gpu);
    if (err != cudaSuccess)
    {
        log_error("Failed to set GPU: %s", cudaGetErrorString(err));
        exit(1);
    }

    void* buf = NULL;
    err = cudaMalloc(&buf, len);
    if (err != cudaSuccess)
    {
        log_error("Failed to allocate device memory: %s", cudaGetErrorString(err));
        exit(1);
    }

    log_debug("Allocated device buffer %p", buf);
    return buf;
}



/* Free device buffer */
extern "C"
void free_gpu_buffer(int gpu, void* ptr)
{
    log_debug("Freing buffer %p", ptr);
    cudaSetDevice(gpu);
    cudaFree(ptr);
}



/* Create local segment and attach device memory to it */
extern "C"
sci_local_segment_t make_local_segment(sci_desc_t sd, unsigned adapter, unsigned id, void* ptr, size_t len)
{
    sci_error_t err;
    sci_local_segment_t segment;

    SCICreateSegment(sd, &segment, id, len, NULL, NULL, SCI_FLAG_EMPTY, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to create segment: %s", SCIGetErrorString(err));
        exit(1);
    }

    void* phys_ptr = phys_addr(ptr);
    set_sync_memops(phys_ptr);

    SCIAttachPhysicalMemory(0, phys_ptr, 0, len, segment, SCI_FLAG_CUDA_BUFFER, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to attach physical memory: %s", SCIGetErrorString(err));
        exit(1);
    }

    SCIPrepareSegment(segment, adapter, 0, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to prepare segment: %s", SCIGetErrorString(err));
        exit(1);
    }

    log_ioaddr(segment);
    log_debug("Prepared segment %u (%p) for NTB adapter %u", id, ptr, adapter);
    return segment;
}



size_t validate_gpu_buffer(int gpu, void* ptr, size_t len, uint8_t val)
{
    cudaError_t err = cudaSetDevice(gpu);
    if (err != cudaSuccess)
    {
        log_error("Failed to set GPU: %s", cudaGetErrorString(err));
        exit(1);
    }

    cudaDeviceSynchronize();

    uint8_t* buf = NULL;
    err = cudaHostAlloc(&buf, len, cudaHostAllocDefault);
    if (err != cudaSuccess)
    {
        log_error("Failed to allocate host memory: %s", cudaGetErrorString(err));
        exit(1);
    }

    err = cudaMemcpy(buf, ptr, len, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        log_error("Failed to memcpy: %s", cudaGetErrorString(err));
        exit(1);
    }

    cudaDeviceSynchronize();

    size_t idx;
    val = buf[0];
    for (idx = 1; idx < len; ++idx)
    {
        if (buf[idx] != val)
        {
            log_debug("Byte %lu in GPU buffer differs (got %02x expected %02x)", idx, buf[idx], val);
            break;
        }
    }

    cudaFreeHost(buf);
    return idx;
}
