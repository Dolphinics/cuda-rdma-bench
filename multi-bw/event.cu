#include <cuda.h>
#include <memory>
#include <exception>
#include <stdexcept>
#include "event.h"


static void deleteTimingData(TimingData* data)
{
    cudaEventDestroy(data->started);
    cudaEventDestroy(data->stopped);
    delete data;
}


TimingDataPtr createTimingData()
{
    cudaError_t err;

    TimingData* data = new TimingData;

    err = cudaEventCreate(&data->started);
    if (err != cudaSuccess)
    {
        delete data;
        throw std::runtime_error(cudaGetErrorString(err));
    }

    err = cudaEventCreate(&data->stopped);
    if (err != cudaSuccess)
    {
        cudaEventDestroy(data->started);
        delete data;
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return TimingDataPtr(data, &deleteTimingData);
}



double TimingData::usecs() const
{
    float milliseconds = .0f;

    cudaError_t err = cudaEventElapsedTime(&milliseconds, started, stopped);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return (double) milliseconds * 1000;
}
