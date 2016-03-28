#ifndef __STREAM_HANDLING_H__
#define __STREAM_HANDLING_H__

#include <cuda.h>
#include <memory>
#include <tr1/memory>

typedef std::tr1::shared_ptr<cudaStream_t> streamPtr;


streamPtr retrieveStream(int device, bool shareDeviceStream, bool shareSingleStream);

#endif
