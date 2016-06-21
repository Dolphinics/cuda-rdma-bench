#include <cuda.h>
#include <memory>
#include <exception>
#include <stdexcept>
#include "hostbuf.h"


static void* createHostBuffer(size_t length, unsigned int flags)
{
    void* buffer;

    cudaError_t err = cudaHostAlloc(&buffer, length, flags);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return buffer;
}


HostBuffer::HostBuffer(size_t length, unsigned int flags)
    : length(length)
    , buffer(createHostBuffer(length, flags))
{
}


HostBuffer::~HostBuffer()
{
    cudaFreeHost(buffer);
}

