#include <cuda.h>
#include <map>
#include <memory>
#include <exception>
#include <stdexcept>
#include "stream.h"

using std::runtime_error;
typedef std::map<int, StreamPtr> StreamMap;

static StreamMap streams;


static void deleteStream(cudaStream_t* stream)
{
    cudaStreamSynchronize(*stream);
    cudaStreamDestroy(*stream);
    delete stream;
}


static StreamPtr createStream()
{
    cudaStream_t* stream = new cudaStream_t;
    cudaError_t err = cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking);
    //cudaError_t err = cudaStreamCreateWithFlags(stream, cudaStreamDefault);
    if (err != cudaSuccess)
    {
        delete stream;
        throw runtime_error(cudaGetErrorString(err));
    }

    return StreamPtr(stream, &deleteStream);
}


StreamPtr retrieveStream(int device, StreamSharingMode sharing)
{
    if (sharing != perTransfer)
    {
        if (sharing == singleStream)
        {
            device = -1;
        }

        // Try to find stream in map
        StreamMap::iterator lowerBound = streams.lower_bound(device);
        if (lowerBound != streams.end() && !(streams.key_comp()(device, lowerBound->first)))
        {
            return lowerBound->second;
        }

        // Stream was not found in map, create it and return it
        StreamPtr stream = createStream();
        streams.insert(lowerBound, StreamMap::value_type(device, stream));
        return stream;
    }

    // Create a new stream every time
    return createStream();
}

