#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <limits.h>
#include <sisci_api.h>
#include "local.h"
#include "log.h"
#include "util.h"
#include <cuda.h>


enum 
{
    CREATE = 0,
    ATTACH,
    NUM_FLAGS
};


// Export list entry
struct export
{
    unsigned            adapter_no; // the adapter the segment is exported on
    unsigned            flags;      // the SISCI flags used for SCIPrepareSegment
    unsigned            available;  // has SCISetSegmentAvailable been called
};


// Internal structure holding the local segment descriptor and its state
struct local_segment
{
    unsigned            created     : 1,        // was the sci_desc_t descriptor opened?
                        attached    : 1,        // was SCICreateSegment successfylly called?
                        rw_mapped   : 1,        // was SCIMapLocalSegment with read-only called?
                        ro_mapped   : 1;        // was SCIMapLocalSegment without read-only called?

    sci_desc_t          sci_d;          
    unsigned            seg_id;
    sci_local_segment_t seg_d;
    unsigned            sci_flags[NUM_FLAGS];   // additional SISCI flags
    size_t              seg_sz;
    sci_map_t           ro_map;
    void*               ro_ptr;
    sci_map_t           rw_map;
    void*               rw_ptr;
    struct export       exports[MAX_EXPORTS];
};


static void clear_handle(l_segment_t handle)
{
    handle->created = 0;
    handle->attached = 0;
    handle->rw_mapped = 0;
    handle->ro_mapped = 0;

    handle->seg_id = 0;
    handle->seg_sz = 0;
    handle->ro_ptr = NULL;
    handle->rw_ptr = NULL;
    
    for (size_t i = 0; i < NUM_FLAGS; ++i)
    {
        handle->sci_flags[i] = 0;
    }

    for (size_t i = 0; i < MAX_EXPORTS; ++i)
    {
        handle->exports[i].adapter_no = UINT_MAX;
        handle->exports[i].flags = 0;
        handle->exports[i].available = 0;
    }
}


int CreateLocalSegment(l_segment_t* segment, unsigned segmentId, unsigned flags)
{
    *segment = NULL;

    l_segment_t handle = (l_segment_t) malloc(sizeof(struct local_segment));
    if (handle == NULL)
    {
        error("Failed to allocate handle: %s", strerror(errno));
        return -1;
    }

    sci_error_t err;

    SCIOpen(&handle->sci_d, 0, &err);
    if (err != SCI_ERR_OK)
    {
        error("Failed to open descriptor: %s", GetErrorString(err));
        free(handle);
        return -1;
    }

    clear_handle(handle);
    handle->seg_id = segmentId;
    handle->sci_flags[CREATE] = flags;
    handle->created = 1;

    *segment = handle;
    
    debug("Segment handle created for segment %u", segmentId);
    return 0;
}


int AllocSegmentMem(l_segment_t segment, size_t size, unsigned flags)
{
    if (!segment->created)
    {
        error("Segment descriptor not initialized");
        return -1;
    }
    else if (segment->attached)
    {
        error("Segment already alloc'd/attached");
        return -1;
    }
    else if (flags != 0)
    {
        debug("flags will not be used in AllocSegment");
        return -1;
    }

    sci_error_t err;

    SCICreateSegment(segment->sci_d, &segment->seg_d, segment->seg_id, size, NULL, NULL, segment->sci_flags[CREATE], &err);
    if (err != SCI_ERR_OK)
    {
        error("Failed to create segment: %s", GetErrorString(err));
        return -1;
    }

    segment->seg_sz = size;
    segment->sci_flags[ATTACH] = 0;
    segment->attached = 1;

    debug("Allocated memory for segment %u", segment->seg_id);
    return 0;
}


int AttachCudaMem(l_segment_t segment, void* devicePtr, size_t size, unsigned flags)
{
    if (!segment->created)
    {
        error("Segment descriptor not initialized");
        return -1;
    }
    else if (segment->attached)
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

    segment->sci_flags[CREATE] |= SCI_FLAG_EMPTY;
    SCICreateSegment(segment->sci_d, &segment->seg_d, segment->seg_id, size, NULL, NULL, segment->sci_flags[CREATE] | flags, &err);
    if (err != SCI_ERR_OK)
    {
        error("Failed to create segment: %s", GetErrorString(err));
        return -1;
    }

    SCIAttachPhysicalMemory(0, devicePtr, 0, size, segment->seg_d, SCI_FLAG_CUDA_BUFFER | flags, &err);
    if (err != SCI_ERR_OK)
    {
        error("Failed to attach physical memory: %s", GetErrorString(err));
        SCIRemoveSegment(segment->seg_d, 0, &err);
        return -1;
    }

    segment->seg_sz = size;
    segment->sci_flags[ATTACH] = flags;
    segment->attached = 1;

    debug("Attached memory for segment %u", segment->seg_id);
    return 0;
}


int ExportLocalSegment(l_segment_t segment, unsigned adapterNo, unsigned flags)
{
    if (!segment->attached)
    {
        error("Segment has not been alloc'd/attached");
        return -1;
    }

    size_t export_idx;

    for (export_idx = 0; export_idx < MAX_EXPORTS; ++export_idx)
    {
        if (segment->exports[export_idx].adapter_no == UINT_MAX || segment->exports[export_idx].adapter_no == adapterNo)
        {
            break;
        }
    }

    if (export_idx == MAX_EXPORTS)
    {
        error("Maximum number of exports for local segment %u reached", segment->seg_id);
        return -1;
    }

    if (segment->exports[export_idx].adapter_no == adapterNo)
    {
        debug("Segment %u was already exported on adapter %u", segment->seg_id, adapterNo);
    }

    sci_error_t err;

    SCIPrepareSegment(segment->seg_d, adapterNo, flags, &err);
    if (err != SCI_ERR_OK)
    {
        error("Failed to prepare segment: %s", GetErrorString(err));
        return -1;
    }

    struct export* export = &segment->exports[export_idx];
    export->adapter_no = adapterNo;
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


int UnexportLocalSegment(l_segment_t segment, unsigned adapterNo)
{
    if (!segment->attached)
    {
        error("Segment has not been alloc'd/attached");
        return -1;
    }

    size_t export_idx;

    for (export_idx = 0; export_idx < MAX_EXPORTS; ++export_idx)
    {
        if (segment->exports[export_idx].adapter_no == adapterNo)
        {
            break;
        }
    }

    if (export_idx == MAX_EXPORTS)
    {
        warn("Local segment %u was not exported on adapter %u", segment->seg_id, adapterNo);
        return -1;
    }

    if (!segment->exports[export_idx].available)
    {
        debug("Segment %u was already unexported on adapter %u", segment->seg_id, adapterNo);
        return 0;
    }

    sci_error_t err;

    do
    {
        SCISetSegmentUnavailable(segment->seg_d, adapterNo, SCI_FLAG_NOTIFY | SCI_FLAG_FORCE_DISCONNECT, &err);
    }
    while (err == SCI_ERR_BUSY);

    if (err != SCI_ERR_OK)
    {
        error("Failed to set segment unavailable: %s", GetErrorString(err));
        return -1;
    }

    debug("Segment %u unexported on adapter %u", segment->seg_id, adapterNo);
    return 0;
}


int RemoveLocalSegment(l_segment_t segment)
{
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
            if (segment->exports[i].adapter_no != UINT_MAX && segment->exports[i].available)
            {
                UnexportLocalSegment(segment, segment->exports[i].adapter_no);
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

    if (segment->created)
    {
        SCIClose(segment->sci_d, 0, &err);
        if (err != SCI_ERR_OK)
        {
            error("Failed to close descriptor: %s", GetErrorString(err));
        }
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
            return NULL;
        }

        segment->rw_mapped = 1;
        debug("Mapped segment %u into virtual memory at address %p", segment->seg_id, segment->rw_ptr);
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
            return NULL;
        }

        segment->rw_mapped = 1;
        debug("Mapped segment %u into virtual memory at address %p", segment->seg_id, segment->ro_ptr);
    }

    return segment->ro_ptr;
}
