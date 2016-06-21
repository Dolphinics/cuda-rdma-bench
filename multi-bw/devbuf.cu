#include <cuda.h>
#include <memory>
#include <exception>
#include <stdexcept>
#include "devbuf.h"


static void* createDeviceBuffer(int device, size_t length)
{
    cudaError_t err;

    err = cudaSetDevice(device);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    void* buffer;
    err = cudaMalloc(&buffer, length);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return buffer;
}


DeviceBuffer::DeviceBuffer(int device, size_t length)
    : device(device)
    , length(length)
    , buffer(createDeviceBuffer(device, length))
{ 
}


DeviceBuffer::~DeviceBuffer()
{
    cudaFree(buffer);
}
