#ifndef __BENCHMARK_H__
#define __BENCHMARK_H__

#include <cuda.h>
#include "devbuf.h"
#include "hostbuf.h"
#include "stream.h"
#include "event.h"


struct TransferSpec
{
    DeviceBufferPtr             deviceBuffer;
    HostBufferPtr               hostBuffer;
    StreamPtr                   cudaStream;
    cudaMemcpyKind              direction;
    TimingDataPtr               cudaEvents;
};


void runBandwidthTest(const std::vector<TransferSpec>& transferSpecifications);

#endif
