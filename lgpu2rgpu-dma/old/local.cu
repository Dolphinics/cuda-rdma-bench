#include <cuda.h>
#include <stdint.h>
#include <stdlib.h>
#include "shared_cuda.h"
#include "shared_sci.h"
#include "local.h"

__global__ void MakeGPUSetNodeID(uint8_t* memPtr, uint32_t nodeId)
{
    *((uint32_t*) memPtr) = nodeId;
}

static __host__
sci_local_segment_t CreateSegment(uint32_t id, sci_desc_t sd, uint32_t adapter, void* ptr, size_t size)
{
    sci_error_t err;
    sci_local_segment_t segment;

    SCICreateSegment(sd, &segment, id, size, NULL, NULL, SCI_FLAG_EMPTY, &err);
    sciAssert(err);

    void* devPtr = GetGpuDevicePtr(ptr);
    SetSyncMemops(devPtr);
    // FIXME: Error handling

    SCIAttachPhysicalMemory(0, devPtr, 0, size, segment, SCI_FLAG_CUDA_BUFFER, &err);
    sciAssert(err);

    SCIPrepareSegment(segment, adapter, 0, &err);
    sciAssert(err);

    return segment;
}

extern "C"
local_t CreateLocalSegment(sci_desc_t sciDesc, uint32_t localNodeId, uint32_t adapterNo, int gpuId, size_t memSize)
{
    cudaError_t err;

    fprintf(stdout, "Local GPU: %d\n", gpuId);

    // Allocate device memory
    err = cudaSetDevice(gpuId);
    cudaAssert(err);

    uint8_t* memPtr;
    err = cudaMalloc(&memPtr, memSize);
    cudaAssert(err);

    // Create local segment
    const uint32_t segmentId = (localNodeId << 24);
    sci_local_segment_t segment = CreateSegment(segmentId, sciDesc, adapterNo, memPtr, memSize);

    uint32_t value = rand() & 255;
    printf("****** MY VALUE = %u *********\n", value);
    MakeGPUSetNodeID<<<1, 1>>>(memPtr, value);

    err = cudaDeviceSynchronize();
    cudaAssert(err);

    // Make descriptor type and return
    local_t descriptor;
    descriptor.segment = segment;
    descriptor.buffer = memPtr;
    descriptor.length = memSize;
    descriptor.device = gpuId;
    return descriptor;
}

extern "C"
void FreeLocalSegment(local_t h, uint32_t adapter)
{
    sci_error_t err;
    
    SCISetSegmentUnavailable(h.segment, adapter, SCI_FLAG_NOTIFY | SCI_FLAG_FORCE_DISCONNECT, &err);

    do
    {
        SCIRemoveSegment(h.segment, 0, &err);
    }
    while (SCI_ERR_OK != err);

    cudaSetDevice(h.device);
    cudaFree(h.buffer);
}

extern "C"
void DumpGPUMemory(local_t h)
{
    uint8_t* buf;
    cudaHostAlloc(&buf, h.length, cudaHostAllocDefault);
    cudaCheckError();

    cudaSetDevice(h.device);
    cudaCheckError();

    cudaMemcpy(buf, h.buffer, h.length, cudaMemcpyDeviceToHost);
    cudaCheckError();

    fprintf(stdout, "%u\n", *((unsigned*) buf));

    cudaFreeHost(buf);
}
