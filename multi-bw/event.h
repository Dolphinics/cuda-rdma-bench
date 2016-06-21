#ifndef __TIMING_DATA_H__
#define __TIMING_DATA_H__

#include <cuda.h>
#include <memory>


struct TimingData
{
    cudaEvent_t started;
    cudaEvent_t stopped;

    double usecs() const;
};


typedef std::shared_ptr<TimingData> TimingDataPtr;


TimingDataPtr createTimingData();

#endif
