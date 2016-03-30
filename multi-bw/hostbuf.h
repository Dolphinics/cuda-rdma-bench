#ifndef __HOST_BUFFER_H__
#define __HOST_BUFFER_H__

#include <memory>


struct HostBuffer
{
    const size_t length;
    void* const  buffer;

    HostBuffer(size_t length, unsigned int flags);
    ~HostBuffer();

    HostBuffer(const HostBuffer& other) = delete;
    HostBuffer& operator=(const HostBuffer& other) = delete;
};


typedef std::shared_ptr<HostBuffer> HostBufferPtr;

#endif
