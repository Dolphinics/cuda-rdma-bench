#include <cuda.h>
#include <sisci_api.h>
#include "remote.h"
#include "shared_sci.h"

remote_t ConnectRemoteSegment(sci_desc_t sciDev, uint32_t remoteNodeId, uint32_t adapterNo)
{
    sci_error_t err;
    sci_remote_segment_t segment;

    do
    {
        SCIConnectSegment(sciDev, &segment, remoteNodeId, (remoteNodeId << 24), adapterNo, NULL, NULL, SCI_INFINITE_TIMEOUT, 0, &err);
    }
    while (SCI_ERR_OK != err);

    size_t memSize = SCIGetRemoteSegmentSize(segment);

    remote_t remote;
    remote.segment = segment;
    remote.length = memSize;
    return remote;
}

void FreeRemoteSegment(remote_t rseg)
{
    sci_error_t err;

    do
    {
        SCIDisconnectSegment(rseg.segment, 0, &err);
    }
    while (SCI_ERR_OK != err);
}
