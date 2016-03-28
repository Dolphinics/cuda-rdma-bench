#include <cuda.h>
#include <memory>
#include <exception>
#include <stdexcept>
#include "devbuf.h"


struct DeviceBufferData
{
    int          device;
    size_t       length;
    void*        buffer;

    DeviceBufferData(int dev, size_t len);
    ~DeviceBufferData();
};


DeviceBufferData::DeviceBufferData(int dev, size_t len)
    : device(dev)
    , length(len)
    , buffer(NULL)
{
    cudaError_t err;

    err = cudaSetDevice(dev);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    err = cudaMalloc(&buffer, len);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}


DeviceBufferData::~DeviceBufferData()
{
    if (buffer != NULL)
    {
        cudaFree(buffer);
    }
    
    device = -1;
    length = 0;
    buffer = NULL;
}


DeviceBuffer::DeviceBuffer(int dev, size_t len)
    : pData(new DeviceBufferData(dev, len))
{
    device = pData->device;
    length = pData->length;
    buffer = pData->buffer;
}


DeviceBuffer::DeviceBuffer(const DeviceBuffer& rhs)
    : pData(rhs.pData)
{
    device = pData->device;
    length = pData->length;
    buffer = pData->buffer;
}


DeviceBuffer& DeviceBuffer::operator=(const DeviceBuffer& rhs)
{
    pData = rhs.pData;
    device = pData->device;
    length = pData->length;
    buffer = pData->buffer;
    return *this;
}
