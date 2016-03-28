#include <cuda.h>
#include <memory>
#include <exception>
#include <stdexcept>
#include "hostbuf.h"
#include <errno.h>
#include <string.h>


struct HostBufferData
{
    bool    sysmem;
    size_t  length;
    void*   buffer;
};


static void deleteData(HostBufferData* data)
{
    if (data->buffer != NULL)
    {
        if (data->sysmem)
        {
            free(data->buffer);
        }
        else
        {
            cudaFreeHost(data->buffer);
        }
    }

    delete data;
}


HostBuffer::HostBuffer(size_t len, unsigned int flags)
    : pData(new HostBufferData, deleteData)
    , length(pData->length)
    , buffer(pData->buffer)
{
    pData->sysmem = false;
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


HostBuffer::HostBuffer(size_t len)
    : pData(new HostBufferData, deleteData)
    , length(pData->length)
    , buffer(pData->buffer)
{
    pData->sysmem = true;
    pData->length = len;

    pData->buffer = (void*) malloc(len);
    if (pData->buffer == NULL)
    {
        throw std::runtime_error(strerror(errno));
    }
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
