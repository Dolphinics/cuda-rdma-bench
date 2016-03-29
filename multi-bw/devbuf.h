#ifndef __DEVICE_BUFFER_H__
#define __DEVICE_BUFFER_H__

#include <cuda.h>
#include <memory>
//#include <tr1/memory>

struct DeviceBufferData;

class DeviceBuffer
{
    private:
        //std::tr1::shared_ptr<DeviceBufferData> pData;
        std::shared_ptr<DeviceBufferData> pData;

    public:
        int          device;
        size_t       length;
        void*        buffer;

        DeviceBuffer(int device, size_t length);
        DeviceBuffer(const DeviceBuffer& other);
        DeviceBuffer& operator=(const DeviceBuffer& other);
};

#endif
