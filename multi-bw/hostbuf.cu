#include <cuda.h>
#include <memory>
#include <exception>
#include <stdexcept>
#include "hostbuf.h"
#include <errno.h>
#include <string.h>


struct HostBufferData
{
    size_t  length;
    void*   buffer;
};


static void deleteData(HostBufferData* data)
{
    if (data->buffer != NULL)
    {
        cudaFreeHost(data->buffer);
    }

    delete data;
}


HostBuffer::HostBuffer(size_t len, unsigned int flags)
    : pData(new HostBufferData, deleteData)
    , length(pData->length)
    , buffer(pData->buffer)
{
    pData->buffer = NULL;
    pData->length = len;

    cudaError_t err;
    err = cudaHostAlloc(&pData->buffer, len, flags);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(strerror(errno));
    }
    
    length = len;
    buffer = pData->buffer;
}


HostBuffer::HostBuffer(const HostBuffer& rhs)
    : pData(rhs.pData)
    , length(pData->length)
    , buffer(pData->buffer)
{
}


HostBuffer& HostBuffer::operator=(const HostBuffer& rhs)
{
    pData = rhs.pData;
    length = pData->length;
    buffer = pData->buffer;
    return *this;
}
