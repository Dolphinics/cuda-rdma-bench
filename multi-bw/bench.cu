#include <cuda.h>
#include <vector>
#include <exception>
#include <stdexcept>
#include <string>
#include <cstring>
#include <cstdio>
#include "bench.h"
#include "devbuf.h"
#include "hostbuf.h"
#include "stream.h"

using std::vector;
using std::runtime_error;
using std::string;



struct StreamData
{
    int         device;
    void*       buffer;
    size_t      length;
    streamPtr   stream;
    cudaEvent_t started;
    cudaEvent_t stopped;
    double      elapsed;
    double      bandwidth;
};


static string bytesToUnit(size_t size)
{
    char buffer[1024];
    const char* units[] = { "B  ", "KiB", "MiB", "GiB", "TiB" };
    size_t i = 0, n = sizeof(units) / sizeof(units[0]);

    double csize = (double) size;

    while (i < (n - 1) && csize >= 1024.0)
    {
        csize /= 1024.0;
        ++i;
    }

    snprintf(buffer, sizeof(buffer), "%.2f %s", csize, units[i]);
    return string(buffer);
}


static double usecsElapsed(const StreamData& data)
{
    float milliseconds = .0f;

    cudaError_t err = cudaEventElapsedTime(&milliseconds, data.started, data.stopped);
    if (err != cudaSuccess)
    {
        throw runtime_error(cudaGetErrorString(err));
    }

    return (double) milliseconds * 1000;
}


static void measureMemcpyBandwidth(void* hostBuffer, vector<StreamData>& streamData, cudaMemcpyKind kind)
{
    cudaError_t err;

    // Start transfers
    for (vector<StreamData>::iterator it = streamData.begin(); it != streamData.end(); ++it)
    {
        const void* src = kind == cudaMemcpyDeviceToHost ? it->buffer : hostBuffer;
        void* dst = kind == cudaMemcpyDeviceToHost ? hostBuffer : it->buffer;

        err = cudaEventRecord(it->started, *it->stream);
        if (err != cudaSuccess)
        {
            throw runtime_error(cudaGetErrorString(err));
        }

        err = cudaMemcpyAsync(dst, src, it->length, kind, *it->stream);
        if (err != cudaSuccess)
        {
            throw runtime_error(cudaGetErrorString(err));
        }

        err = cudaEventRecord(it->stopped, *it->stream);
        if (err != cudaSuccess)
        {
            throw runtime_error(cudaGetErrorString(err));
        }
    }

    // Synchronize events and record results
    for (vector<StreamData>::iterator it = streamData.begin(); it != streamData.end(); ++it)
    {
        err = cudaEventSynchronize(it->stopped);
        if (err != cudaSuccess)
        {
            throw runtime_error(cudaGetErrorString(err));
        }

        it->elapsed = usecsElapsed(*it);
        it->bandwidth = (double) it->length / it->elapsed;
    }
}


static void runBandwidthTest(const HostBuffer& hostBuffer, const vector<DeviceBuffer>& deviceBuffers, cudaMemcpyKind kind, bool shareDeviceStream, bool shareGlobalStream)
{
    cudaError_t err;

    // Create streams and events
    vector<StreamData> streamData;
    for (vector<DeviceBuffer>::const_iterator it = deviceBuffers.begin(); it != deviceBuffers.end(); ++it)
    {
        StreamData data;
        data.device = it->device;
        data.buffer = it->buffer;
        data.length = it->length;
        data.elapsed = 0;
        data.bandwidth = 0;

        data.stream = retrieveStream(it->device, shareDeviceStream, shareGlobalStream);

        err = cudaEventCreate(&data.started);
        if (err != cudaSuccess)
        {
            throw runtime_error(cudaGetErrorString(err));
        }

        err = cudaEventCreate(&data.stopped);
        if (err != cudaSuccess)
        {
            cudaEventDestroy(data.started);
            throw runtime_error(cudaGetErrorString(err));
        }

        streamData.push_back(data);
    }

    // Run measurements
    try
    {
        fprintf(stdout, "\n====================    %-14s   (%11s)    ====================\n",
                kind == cudaMemcpyDeviceToHost ? "DEVICE TO HOST" : "HOST TO DEVICE",
                bytesToUnit(hostBuffer.length).c_str()
               );

        measureMemcpyBandwidth(hostBuffer.buffer, streamData, kind);
    }
    catch (const runtime_error& e)
    {
        fprintf(stderr, "Unexpected error, aborting...\n");
    }

    // Print results and clean up
    for (vector<StreamData>::iterator it = streamData.begin(); it != streamData.end(); ++it)
    {
        // get device name
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, it->device);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "WARNING: %s\n", cudaGetErrorString(err));
            prop.name[0] = '\0';
        }

        // print results
        fprintf(stdout, "%4d %-25s %10s %8.0f µs %10.2f MiB/s\n",
                it->device, 
                prop.name, 
                bytesToUnit(it->length).c_str(), 
                it->elapsed,
                it->bandwidth
               );

        // clean up
        cudaEventDestroy(it->started);
        cudaEventDestroy(it->stopped);
    }
}


void benchmark(const vector<HostBuffer>& buffers, const vector<int>& devices, const vector<cudaMemcpyKind>& modes, bool shareDeviceStream, bool shareGlobalStream)
{
    cudaError_t err;

    for (vector<cudaMemcpyKind>::const_iterator kindIt = modes.begin(); kindIt != modes.end(); ++kindIt)
    {
        cudaMemcpyKind kind = *kindIt;

        for (vector<HostBuffer>::const_iterator bufIt = buffers.begin(); bufIt != buffers.end(); ++bufIt)
        {
            // Get host buffer
            const HostBuffer& buffer = *bufIt;

            // Create device buffers and synchronize devices
            vector<DeviceBuffer> deviceBuffers;
            for (vector<int>::const_iterator devIt = devices.begin(); devIt != devices.end(); ++devIt)
            {
                // TODO: Create StreamData here instead?

                int device = *devIt;

                // synchronize device
                err = cudaSetDevice(device);
                if (err != cudaSuccess)
                {
                    throw runtime_error(cudaGetErrorString(err));
                }

                err = cudaDeviceSynchronize();
                if (err != cudaSuccess)
                {
                    throw runtime_error(cudaGetErrorString(err));
                }

                // create device buffer
                deviceBuffers.push_back(DeviceBuffer(device, buffer.length));
            }

            // Run bandwidth test
            runBandwidthTest(buffer, deviceBuffers, kind, shareDeviceStream, shareGlobalStream);

            // TODO: Print results here?
        }
    }
    printf("\n");
}
