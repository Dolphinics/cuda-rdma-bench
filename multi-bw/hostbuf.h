#ifndef __HOST_BUFFER_H__
#define __HOST_BUFFER_H__

#include <memory>
#include <tr1/memory>

struct HostBufferData;

class HostBuffer
{
    private:
        std::tr1::shared_ptr<HostBufferData> pData;

    public:
        size_t length;
        void*  buffer;

        HostBuffer(size_t length, unsigned int flags);
        HostBuffer(const HostBuffer& other);
        HostBuffer& operator=(const HostBuffer& other);
};

#endif
