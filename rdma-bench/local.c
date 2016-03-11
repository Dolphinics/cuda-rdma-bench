#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <limits.h>
#include <sisci_api.h>
#include <cuda.h>
#include "simple_types.h"
#include "local.h"
#include "log.h"
#include "util.h"


// Export list entry
// Used to keep track of which adapters the segment is exported on
struct export
{
    unsigned adapt_no;  // the adapter the segment is exported on
    unsigned flags;     // the SISCI flags used for SCIPrepareSegment
    unsigned available; // has SCISetSegmentAvailable been called
};


// Internal structure holding the local segment descriptor and its state
struct local_segment
{
    unsigned            attached  : 1,  // SCICreateSegment succeeded
                        rw_mapped : 1,  // SCIMapLocalSegment w/ RO succeeded
                        ro_mapped : 1;  // SCIMapLocalSegment succeeded
    sci_desc_t          sci_d;          // SISCI descriptor
    sci_local_segment_t seg_d;          // local segment descriptor
    unsigned            seg_id;         // local segment identifier
    size_t              seg_sz;         // size of local segment
    unsigned            fl_create;      // additional flags passed do SCICreateSegment
    unsigned            fl_attach;      // additional flags passed to SCIAttachPhysicalMemory
    sci_map_t           ro_map;         // map descriptor for RO memory
    const void*         ro_ptr;         // pointer to mapped RO memory
    sci_map_t           rw_map;         // map descriptor for RW memory
    void*               rw_ptr;         // pointer to mapped RW memory
    struct export       exports[MAX_EXPORTS]; // export list
};


// Empty the internal structure
static void clear_handle(l_segment_t handle)
{
    handle->attached = 0;
    handle->rw_mapped = 0;
    handle->ro_mapped = 0;

    handle->seg_id = 0;
    handle->seg_sz = 0;
    handle->ro_ptr = NULL;
    handle->rw_ptr = NULL;

    handle->fl_create = 0;
    handle->fl_attach = 0;

    for (size_t i = 0; i < MAX_EXPORTS; ++i)
    {
        handle->exports[i].adapt_no = UINT_MAX;
        handle->exports[i].flags = 0;
        handle->exports[i].available = 0;
    }
}


int CreateLocalSegment(l_segment_t* segment, unsigned segmentId, unsigned flags)
{
    // FIXME: Return errnos for errors
    l_segment_t handle = (l_segment_t) malloc(sizeof(struct local_segment));
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

    handle->seg_id = segmentId;
    handle->fl_create = flags;

    *segment = handle;
    
    debug("Segment handle created for segment %u", segmentId);
    return 0;
}


int AllocSegmentMem(l_segment_t segment, size_t size)
{
    // FIXME: Return errnos for errors

    if (segment->attached)
    {
        error("Segment already alloc'd/attached");
        return -1;
    }

    sci_error_t err;

    SCICreateSegment(segment->sci_d, &segment->seg_d, segment->seg_id, size, NULL, NULL, segment->fl_create, &err);
    if (err != SCI_ERR_OK)
    {
        error("Failed to create segment: %s", GetErrorString(err));
        return -1;
    }

    segment->seg_sz = size;
    segment->fl_attach = 0;
    segment->attached = 1;

    // TODO: Log ioaddr of segment
    debug("Allocated memory for segment %u", segment->seg_id);
    return 0;
}


int AttachCudaMem(l_segment_t segment, void* devicePtr, size_t size)
{
    // FIXME: Return errnos for errors

    if (segment->attached)
    {
        error("Segment already alloc'd/attached");
        return -1;
    }

    unsigned flag = 1;
    CUresult res = cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, (CUdeviceptr) devicePtr);
    if (res != CUDA_SUCCESS)
    {
        warn("Failed to set pointer attribute CU_POINTER_ATTRIBUTE_SYNC_MEMOPS");
    }

    sci_error_t err;

    segment->fl_create |= SCI_FLAG_EMPTY;
    SCICreateSegment(segment->sci_d, &segment->seg_d, segment->seg_id, size, NULL, NULL, segment->fl_create, &err);
    if (err != SCI_ERR_OK)
    {
        error("Failed to create segment: %s", GetErrorString(err));
        return -1;
    }

    SCIAttachPhysicalMemory(0, devicePtr, 0, size, segment->seg_d, SCI_FLAG_CUDA_BUFFER, &err);
    if (err != SCI_ERR_OK)
    {
        error("Failed to attach physical memory: %s", GetErrorString(err));
        SCIRemoveSegment(segment->seg_d, 0, &err);
        return -1;
    }

    segment->seg_sz = size;
    segment->fl_attach = 0;
    segment->attached = 1;

    // TODO: Log ioaddr of segment
    debug("Attached memory for segment %u", segment->seg_id);
    return 0;
}


// Helper function to export a segment
static int export_segment(struct export* export, l_segment_t segment, unsigned adapterNo, unsigned flags)
{
    sci_error_t err;

    SCIPrepareSegment(segment->seg_d, adapterNo, flags, &err);
    if (err != SCI_ERR_OK)
    {
        error("Failed to prepare segment: %s", GetErrorString(err));
        return -1;
    }

    export->adapt_no = adapterNo;
    export->flags = flags;

    SCISetSegmentAvailable(segment->seg_d, adapterNo, 0, &err);
    if (err != SCI_ERR_OK)
    {
        error("Failed to set segment available: %s", GetErrorString(err));
        return -1;
    }

    export->available = 1;

    debug("Segment %u exported on adapter %u", segment->seg_id, adapterNo);
    return 0;
}


// Helper function to unexport a segment
static int unexport_segment(struct export* export, l_segment_t segment)
{
    if (!export->available)
    {
        debug("Segment %u was already unexported on adapter %u", segment->seg_id, export->adapt_no);
    }

    sci_error_t err;

    do
    {
        //SCISetSegmentUnavailable(segment->seg_d, export->adapt_no, SCI_FLAG_NOTIFY | SCI_FLAG_FORCE_DISCONNECT, &err);
        SCISetSegmentUnavailable(segment->seg_d, export->adapt_no, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

    if (err != SCI_ERR_OK)
    {
        error("Failed to set segment unavailable: %s", GetErrorString(err));
        return -1;
    }

    export->available = 0;

    debug("Segment %u unexported on adapter %u", segment->seg_id, export->adapt_no);
    return 0;
}


int ExportLocalSegment(l_segment_t segment, unsigned adapterNo, unsigned flags)
{
    // FIXME: Return errnos for errors
    
    if (!segment->attached)
    {
        error("Segment has not been alloc'd/attached");
        return -1;
    }

    for (size_t idx = 0; idx < MAX_EXPORTS; ++idx)
    {
        if (segment->exports[idx].adapt_no == UINT_MAX || segment->exports[idx].adapt_no == adapterNo)
        {
            return export_segment(&segment->exports[idx], segment, adapterNo, flags);
        }
    }

    error("Maximum number of exports for local segment %u reached", segment->seg_id);
    return -1;
}


int UnexportLocalSegment(l_segment_t segment, unsigned adapterNo)
{
    // FIXME: Return errnos for errors
    
    if (!segment->attached)
    {
        error("Segment has not been alloc'd/attached");
        return -1;
    }

    for (size_t idx = 0; idx < MAX_EXPORTS; ++idx)
    {
        if (segment->exports[idx].adapt_no == adapterNo)
        {
            return unexport_segment(&segment->exports[idx], segment);
        }
    }

    warn("Local segment %u was not exported on adapter %u", segment->seg_id, adapterNo);
    return -1;
}


int RemoveLocalSegment(l_segment_t segment)
{
    // FIXME: Return errnos for errors
    
    sci_error_t err;

    if (segment->ro_mapped)
    {
        do
        {
            SCIUnmapSegment(segment->ro_map, 0, &err);
        }
        while (err == SCI_ERR_BUSY);

        if (err != SCI_ERR_OK)
        {
            error("Failed to unmap segment: %s", GetErrorString(err));
        }
    }

    if (segment->rw_mapped)
    {
        do
        {
            SCIUnmapSegment(segment->rw_map, 0, &err);
        }
        while (err == SCI_ERR_BUSY);

        if (err != SCI_ERR_OK)
        {
            error("Failed to unmap segment: %s", GetErrorString(err));
        }
    }

    if (segment->attached)
    {
        for (size_t i = 0; i < MAX_EXPORTS; ++i)
        {
            if (segment->exports[i].adapt_no != UINT_MAX && segment->exports[i].available)
            {
                unexport_segment(&segment->exports[i], segment);
            }
        }

        do
        {
            SCIRemoveSegment(segment->seg_d, 0, &err);
        }
        while (err == SCI_ERR_BUSY);

        if (err != SCI_ERR_OK)
        {
            error("Failed to remove segment: %s", GetErrorString(err));
        }
    }

    SCIClose(segment->sci_d, 0, &err);
    if (err != SCI_ERR_OK)
    {
        error("Failed to close descriptor: %s", GetErrorString(err));
    }

    free(segment);
    return 0;
}


size_t GetLocalSegmentSize(l_segment_t segment)
{
    if (!segment->attached)
    {
        error("Segment has not been alloc'd/attached");
        return 0;
    }

    return segment->seg_sz;
}


void* GetLocalSegmentPtr(l_segment_t segment)
{
    if (!segment->attached)
    {
        error("Segment has not been alloc'd/attached");
        return NULL;
    }

    if (!segment->rw_mapped)
    {
        sci_error_t err;
        
        segment->rw_ptr = SCIMapLocalSegment(segment->seg_d, &segment->rw_map, 0, segment->seg_sz, NULL, 0, &err);
        if (err != SCI_ERR_OK)
        {
            error("Failed to map local segment: %s", GetErrorString(err));
            segment->rw_ptr = NULL;
        }
        else
        {
            debug("Mapped segment %u into virtual memory at address %p", segment->seg_id, segment->rw_ptr);
        }

        segment->rw_mapped = 1;
    }

    return segment->rw_ptr;
}


const void* GetLocalSegmentPtrRO(l_segment_t segment)
{
    if (!segment->attached)
    {
        error("Segment has not been alloc'd/attached");
        return NULL;
    }

    if (!segment->ro_mapped)
    {
        sci_error_t err;
        
        segment->ro_ptr = SCIMapLocalSegment(segment->seg_d, &segment->ro_map, 0, segment->seg_sz, NULL, SCI_FLAG_READONLY_MAP, &err);
        if (err != SCI_ERR_OK)
        {
            error("Failed to map local segment: %s", GetErrorString(err));
            segment->ro_ptr = NULL;
        }
        else
        {
            debug("Mapped segment %u into virtual memory at address %p", segment->seg_id, segment->ro_ptr);
        }

        segment->ro_mapped = 1;
    }

    return segment->ro_ptr;
}
