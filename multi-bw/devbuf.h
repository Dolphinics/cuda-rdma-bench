#ifndef __DEVICE_BUFFER_H__
#define __DEVICE_BUFFER_H__

#include <memory>


struct DeviceBuffer
{
    const int    device;
    const size_t length;
    void* const  buffer;

    DeviceBuffer(int device, size_t length);
    ~DeviceBuffer();

    DeviceBuffer(const DeviceBuffer& other) = delete;
    DeviceBuffer& operator=(const DeviceBuffer& other) = delete;
};


typedef std::shared_ptr<DeviceBuffer> DeviceBufferPtr;

#endif
