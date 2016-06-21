#ifndef __STREAM_HANDLING_H__
#define __STREAM_HANDLING_H__

#include <cuda.h>
#include <memory>

enum StreamSharingMode
{ 
    perTransfer,    // create a stream for every transfer
    perDevice,      // create a stream per device
    singleStream    // everyone use a single stream
};

typedef std::shared_ptr<cudaStream_t> StreamPtr;


StreamPtr retrieveStream(int cudaDevice, StreamSharingMode streamSharing);

#endif
