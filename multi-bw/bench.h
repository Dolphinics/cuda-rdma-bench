#ifndef __BENCHMARK_H__
#define __BENCHMARK_H__

#include <cuda.h>
#include <vector>
#include "devbuf.h"
#include "hostbuf.h"


void benchmark(
        const std::vector<HostBuffer>& hostBuffers, 
        const std::vector<int>& cudaDevices, 
        const std::vector<cudaMemcpyKind>& transferModes,
        bool shareDeviceStream,
        bool shareGlobalStream
        );

#endif
