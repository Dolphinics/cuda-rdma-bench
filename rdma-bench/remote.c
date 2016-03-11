#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <limits.h>
#include <sisci_api.h>
#include "simple_types.h"
#include "remote.h"
#include "log.h"
#include "util.h"


#ifndef CONNECT_TIMEOUT
#define CONNECT_TIMEOUT 5000
#endif


// Internal structure holding the remote segment descriptor and its state
struct remote_segment 
{
    unsigned             ro_mapped : 1, // SCIMapRemoteSegment w/ RO succeeded
                         rw_mapped : 1; // SCIMapRemoteSegment succeeded
    sci_desc_t           sci_d;         // SISCI descriptor
    unsigned             adapt_no;      // local adapter number
    unsigned             node_id;       // remote node identifier
    sci_remote_segment_t seg_d;         // remote segment descriptor
    unsigned             seg_id;        // remote segment identifier
    size_t               seg_sz;        // size of remote segment
    sci_map_t            ro_map;        // map descriptor for RO memory
    const volatile void* ro_ptr;        // pointer to mapped RO memory
    sci_map_t            rw_map;        // map descriptor for RW memory
    volatile void*       rw_ptr;        // pointer to mapped RW memory
    unsigned             fl_connect;    // flags for SCIConnectSegment
    unsigned             fl_map_ro;     // flags for SCIMapRemoteSegment
    unsigned             fl_map_rw;     // flags for SCIMapRemoteSegment
};


static void clear_handle(r_segment_t handle)
{
    handle->ro_mapped = 0;
    handle->rw_mapped = 0;

    handle->adapt_no = UINT_MAX;
    handle->node_id = 0;
    handle->seg_id = 0;
    handle->seg_sz = 0;
    handle->ro_ptr = NULL;
    handle->rw_ptr = NULL;
    handle->fl_connect = 0;
    handle->fl_map_ro = 0;
    handle->fl_map_rw = 0;
}


int ConnectRemoteSegment(r_segment_t* segment, unsigned adapterNo, unsigned remoteNodeId, unsigned segmentId)
{
    // FIXME: Return errnos for errors
    
    r_segment_t handle = (r_segment_t) malloc(sizeof(struct remote_segment));
    if (handle == NULL)
    {
        error("Failed to allocate handle: %s", strerror(errno));
        return -1;
    }

    clear_handle(handle);
    sci_error_t err;

    SCIOpen(&handle->sci_d, 0, &err);
    if (err != SCI_ERR_OK)
    {
        error("Failed to open descriptor: %s", GetErrorString(err));
        free(handle);
        return -1;
    }

    debug("Waiting for remote segment %u on node %u", segmentId, remoteNodeId);
    do
    {
        SCIConnectSegment(handle->sci_d, &handle->seg_d, remoteNodeId, segmentId, adapterNo, NULL, NULL, CONNECT_TIMEOUT, 0, &err);
    }
    while (err == SCI_ERR_TIMEOUT || err == SCI_ERR_NO_SUCH_SEGMENT);

    if (err != SCI_ERR_OK)
    {
        error("Failed to connect to remote segment: %s", GetErrorString(err));
        SCIClose(handle->sci_d, 0, &err);
        free(handle);
        return -1;
    }

    handle->adapt_no = adapterNo;
    handle->node_id = remoteNodeId;
    handle->seg_id = segmentId;
    handle->seg_sz = SCIGetRemoteSegmentSize(handle->seg_d);

    *segment = handle;
    debug("Connected to remote segment %u on node %u", segmentId, remoteNodeId);
    return 0;
}


int DisconnectRemoteSegment(r_segment_t segment)
{
    sci_error_t err;

    do
    {
        SCIDisconnectSegment(segment->seg_d, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

    if (err != SCI_ERR_OK)
    {
        error("Failed to disconnect remote segment: %s", GetErrorString(err));
    }

    SCIClose(segment->sci_d, 0, &err);
    if (err != SCI_ERR_OK)
    {
        error("Failed to close descriptor: %s", GetErrorString(err));
    }

    return 0;
}


size_t GetRemoteSegmentSize(r_segment_t segment)
{
    return segment->seg_sz;
}


void* GetRemoteSegmentPtr(r_segment_t segment)
{
    if (!segment->rw_mapped)
    {
        sci_error_t err;

        segment->rw_ptr = SCIMapRemoteSegment(segment->seg_d, &segment->rw_map, 0, segment->seg_sz, NULL, 0, &err);
        if (err != SCI_ERR_OK)
        {
            error("Failed to map remote segment: %s", GetErrorString(err));
            segment->rw_ptr = NULL;
        }
        else
        {
            debug("Mapped segment %u into virtual memory at address %p", segment->seg_id, segment->rw_ptr);
        }

        segment->rw_mapped = 1;
    }

    return (void*) segment->rw_ptr;
}


const void* GetRemoteSegmentPtrRO(r_segment_t segment)
{
    if (!segment->ro_mapped)
    {
        sci_error_t err;

        segment->ro_ptr = SCIMapRemoteSegment(segment->seg_d, &segment->ro_map, 0, segment->seg_sz, NULL, 0, &err);
        if (err != SCI_ERR_OK)
        {
            error("Failed to map remote segment: %s", GetErrorString(err));
            segment->rw_ptr = NULL;
        }
        else
        {
            debug("Mapped segment %u into virtual memory at address %p", segment->seg_id, segment->ro_ptr);
        }

        segment->ro_mapped = 1;
    }

    return (const void*) segment->ro_ptr;
}
