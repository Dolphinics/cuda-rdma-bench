#include <cuda.h>
#include <map>
#include <memory>
#include <exception>
#include <stdexcept>
#include "stream.h"


using std::runtime_error;
typedef std::map<int, streamPtr> streamMap;


static streamMap streams;


static void deleteStream(cudaStream_t* stream)
{
    cudaStreamDestroy(*stream);
    delete stream;
}


static streamPtr createStream()
{
    cudaStream_t* stream = new cudaStream_t;
    cudaError_t err = cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking);
    if (err != cudaSuccess)
    {
        delete stream;
        throw runtime_error(cudaGetErrorString(err));
    }

    return streamPtr(stream, &deleteStream);
}


streamPtr retrieveStream(int device, bool shareDeviceStream, bool shareSingleStream)
{
    if (shareSingleStream || shareDeviceStream)
    {
        if (shareSingleStream)
        {
            device = -1;
        }

        // Try to find entry in map
        streamMap::iterator lowerBound = streams.lower_bound(device);
        if (lowerBound != streams.end() && !(streams.key_comp()(device, lowerBound->first)))
        {
            return lowerBound->second;
        }

        streamPtr stream = createStream();
        streams.insert(lowerBound, streamMap::value_type(device, stream));
        return stream;
    }

    return createStream();
}

