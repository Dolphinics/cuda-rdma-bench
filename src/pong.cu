#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <sisci_api.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

extern "C" {
#include "common.h"
}


/* Get the device pointer allocated by cudaMalloc */
__host__ void* getDevicePtr(const void* ptr)
{
    cudaPointerAttributes attrs;

    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);

    if (cudaSuccess != err)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        exit(err);
    }

    uint32_t flag = 1;
    CUresult res = cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, ((CUdeviceptr) attrs.devicePointer));

    if (CUDA_SUCCESS != res)
    {
        fprintf(stderr, "CU error: %d\n", res);
        exit(res);
    }

    return attrs.devicePointer;
}

__host__ sci_local_segment_t createSegment(uint32_t id, sci_desc_t sd, uint32_t adapter, void* ptr, size_t size)
{
    sci_error_t err;
    sci_local_segment_t segment;

    SCICreateSegment(sd, &segment, id, size, NULL, NULL, SCI_FLAG_EMPTY, &err);
    if (SCI_ERR_OK != err)
    {
        fprintf(stderr, "SISCI error: %s\n", sciGetErrorString(err));
        exit(err);
    }

    SCIAttachPhysicalMemory(0, getDevicePtr(ptr), 0, size, segment, SCI_FLAG_CUDA_BUFFER, &err);
    if (SCI_ERR_OK != err)
    {
        fprintf(stderr, "SISCI error: %s\n", sciGetErrorString(err));
        exit(err);
    }

    SCIPrepareSegment(segment, adapter, 0, &err);
    if (SCI_ERR_OK != err)
    {
        fprintf(stderr, "SISCI error: %s\n", sciGetErrorString(err));
        exit(err);
    }

    SCISetSegmentAvailable(segment, adapter, 0, &err);
    if (SCI_ERR_OK != err)
    {
        fprintf(stderr, "SISCI error: %s\n", sciGetErrorString(err));
        exit(err);
    }

    return segment;
}

__host__ sci_remote_segment_t connectSegment(uint32_t node_id, uint32_t seg_id, sci_desc_t sd, uint32_t adapter)
{
    sci_error_t err;
    sci_remote_segment_t segment;

    do
    {
        SCIConnectSegment(sd, &segment, node_id, seg_id, adapter, NULL, NULL, SCI_INFINITE_TIMEOUT, 0, &err);
    }
    while (SCI_ERR_OK != err);

    return segment;
}

__host__ volatile void* mapSegment(sci_remote_segment_t segment, size_t size)
{
    sci_error_t err;
    sci_map_t mapping;
    volatile void* ptr;

    ptr = SCIMapRemoteSegment(segment, &mapping, 0, size, NULL, 0, &err);
    if (SCI_ERR_OK != err)
    {
        fprintf(stderr, "SISCI error: %s\n", sciGetErrorString(err));
        exit(err);
    }

    return ptr;
}

extern "C"
void PongNode(sci_desc_t dev_desc, uint32_t adapter, uint32_t local_id, uint32_t remote_id, size_t seg_size)
{
    cudaError_t err;

    // Create some host memory and fill it with stuff
    uint8_t* host_mem = (uint8_t*) malloc(seg_size);
    if (NULL == host_mem)
    {
        perror("malloc");
        exit(errno);
    }

    fillMem(host_mem, seg_size);

    // Allocate device memory
    uint8_t* dev_mem;
    err = cudaMalloc(&dev_mem, seg_size);
    //err = cudaMallocHost(&dev_mem, seg_size);
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        exit(err);
    }

    // Copy host memory data to device memory
    err = cudaMemcpy(dev_mem, host_mem, seg_size, cudaMemcpyHostToDevice);
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        exit(err);
    }

    zeroMem(host_mem, seg_size);

    // Create local segment mapped to GPU memory
    const uint32_t cuda_id = SEGMENT_ID(local_id, 0, 0);
    sci_local_segment_t cuda_segment = createSegment(cuda_id, dev_desc, adapter, dev_mem, seg_size);

    // Connect to remote segment
    const uint32_t dma_id = SEGMENT_ID(remote_id, 0, 0);

    fprintf(stdout, "Trying to connect to remote segment ID = %u on node ID = %u\n", dma_id, remote_id);
    sci_remote_segment_t dma_segment = connectSegment(remote_id, dma_id, dev_desc, adapter);
    
    // Map remote segment into local memory
    volatile void* mapped_mem = mapSegment(dma_segment, seg_size);

    // Copy from GPU to remote memory
    err = cudaMemcpy((void*) mapped_mem, dev_mem, seg_size, cudaMemcpyDeviceToHost);
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        exit(err);
    }

    dumpMem((uint8_t*) mapped_mem, seg_size);

    // Zero out GPU memory
    err = cudaMemcpy(dev_mem, host_mem, seg_size, cudaMemcpyHostToDevice);
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        exit(err);
    }

    // Copy from remote memory to GPU memory
    err = cudaMemcpy(dev_mem, (void*) mapped_mem, seg_size, cudaMemcpyHostToDevice);
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        exit(err);
    }

    dumpMem(host_mem, seg_size);

    // Copy from GPU to local memory
    err = cudaMemcpy(host_mem, dev_mem, seg_size, cudaMemcpyDeviceToHost);
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        exit(err);
    }

    dumpMem(host_mem, seg_size);

    fprintf(stderr, "Device pointer: %16lx\n", (uint64_t) getDevicePtr(dev_mem));
    fprintf(stderr, "Physical addr : %16lx\n", getPhysAddr(cuda_segment));
    fprintf(stderr, "Local IO addr : %16lx\n", getLocalIOAddr(cuda_segment));
    fprintf(stderr, "Remote IO addr: %16lx\n", getRemoteIOAddr(dma_segment));
    fprintf(stderr, "host_mem addr : %16lx\n", (uint64_t) host_mem);

    //memcpy(getDevicePtr(dev_mem), host_mem, seg_size);
}

