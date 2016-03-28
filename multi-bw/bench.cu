#include <cuda.h>
#include <vector>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <cstring>
#include "bench.h"
#include "devbuf.h"
#include "hostbuf.h"

using std::vector;
using std::runtime_error;
using std::string;


struct StreamData
{
    int          device;
    void*        buffer;
    size_t       length;
    cudaStream_t stream;
    cudaEvent_t  started;
    cudaEvent_t  stopped;
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

        err = cudaEventRecord(it->started, it->stream);
        if (err != cudaSuccess)
        {
            throw runtime_error(cudaGetErrorString(err));
        }

        err = cudaMemcpyAsync(dst, src, it->length, kind, it->stream);
        if (err != cudaSuccess)
        {
            throw runtime_error(cudaGetErrorString(err));
        }

        err = cudaEventRecord(it->stopped, it->stream);
        if (err != cudaSuccess)
        {
            throw runtime_error(cudaGetErrorString(err));
        }
    }

    // Print results
    size_t totalSize = 0;
    double totalTime = 0;
    for (vector<StreamData>::iterator it = streamData.begin(); it != streamData.end(); ++it)
    {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, it->device);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "WARNING: %s\n", cudaGetErrorString(err));
            prop.name[0] = '\0';
        }

        // make sure stream is done
        err = cudaEventSynchronize(it->stopped);
        if (err != cudaSuccess)
        {
            throw runtime_error(cudaGetErrorString(err));
        }

        double usecs = usecsElapsed(*it);
        double bandwidth = (double) it->length / usecs;

        totalSize += it->length;
        totalTime += usecs;

        fprintf(stdout, "%4d %-25s %10s %8.0f µs %10.2f MiB/s\n",
                it->device, 
                prop.name, 
                bytesToUnit(it->length).c_str(), 
                usecs,
                bandwidth
               );
    }
}


static void runBandwidthTest(const HostBuffer& hostBuffer, const vector<DeviceBuffer>& deviceBuffers, cudaMemcpyKind kind)
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
        
        err = cudaStreamCreate(&data.stream);
        if (err != cudaSuccess)
        {
            throw runtime_error(cudaGetErrorString(err));
        }

        err = cudaEventCreate(&data.started);
        if (err != cudaSuccess)
        {
            cudaStreamDestroy(data.stream);
            throw runtime_error(cudaGetErrorString(err));
        }

        err = cudaEventCreate(&data.stopped);
        if (err != cudaSuccess)
        {
            cudaStreamDestroy(data.stream);
            cudaEventDestroy(data.started);
            throw runtime_error(cudaGetErrorString(err));
        }

        streamData.push_back(data);
    }

    // Run measurements
    try
    {
        measureMemcpyBandwidth(hostBuffer.buffer, streamData, kind);
    }
    catch (const runtime_error& e)
    {
        fprintf(stderr, "Unexpected error, aborting...\n");
    }

    // Clean up
    for (vector<StreamData>::iterator it = streamData.begin(); it != streamData.end(); ++it)
    {
        cudaEventDestroy(it->started);
        cudaEventDestroy(it->stopped);
        cudaStreamDestroy(it->stream);
    }
}


void benchmark(const vector<HostBuffer>& buffers, const vector<int>& devices, const vector<cudaMemcpyKind>& modes)
{
    // TODO: Print out some info
    // #stream: x

    for (vector<cudaMemcpyKind>::const_iterator kindIt = modes.begin(); kindIt != modes.end(); ++kindIt)
    {
        cudaMemcpyKind kind = *kindIt;

        for (vector<HostBuffer>::const_iterator bufIt = buffers.begin(); bufIt != buffers.end(); ++bufIt)
        {
            // Get host buffer
            const HostBuffer& buffer = *bufIt;

            // Create device buffers
            vector<DeviceBuffer> deviceBuffers;
            for (vector<int>::const_iterator devIt = devices.begin(); devIt != devices.end(); ++devIt)
            {
                deviceBuffers.push_back(DeviceBuffer(*devIt, buffer.length));
            }

            // Run bandwidth test
            fprintf(stdout, "\n====================    %-14s   (%11s)    ====================\n",
                    kind == cudaMemcpyDeviceToHost ? "DEVICE TO HOST" : "HOST TO DEVICE",
                    bytesToUnit(buffer.length).c_str()
                    );
            runBandwidthTest(buffer, deviceBuffers, kind);
        }
    }
}
