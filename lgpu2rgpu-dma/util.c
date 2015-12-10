#include "util.h"
#include <sisci_api.h>
#include <string.h>
#include "reporting.h"
#include "gpu.h"
#include "bench.h"


/* SISCI error codes 
 *
 * Use this to lookup the correct index in the error string table.
 */
static unsigned error_codes[] = {
    SCI_ERR_OK,
    SCI_ERR_BUSY,
    SCI_ERR_FLAG_NOT_IMPLEMENTED,
    SCI_ERR_ILLEGAL_FLAG,
    SCI_ERR_NOSPC,
    SCI_ERR_API_NOSPC,
    SCI_ERR_HW_NOSPC,
    SCI_ERR_NOT_IMPLEMENTED,
    SCI_ERR_ILLEGAL_ADAPTERNO,
    SCI_ERR_NO_SUCH_ADAPTERNO,
    SCI_ERR_TIMEOUT,
    SCI_ERR_OUT_OF_RANGE,
    SCI_ERR_NO_SUCH_SEGMENT,
    SCI_ERR_ILLEGAL_NODEID,
    SCI_ERR_CONNECTION_REFUSED,
    SCI_ERR_SEGMENT_NOT_CONNECTED,
    SCI_ERR_SIZE_ALIGNMENT,
    SCI_ERR_OFFSET_ALIGNMENT,
    SCI_ERR_ILLEGAL_PARAMETER,
    SCI_ERR_MAX_ENTRIES,
    SCI_ERR_SEGMENT_NOT_PREPARED,
    SCI_ERR_ILLEGAL_ADDRESS,
    SCI_ERR_ILLEGAL_OPERATION,
    SCI_ERR_ILLEGAL_QUERY,
    SCI_ERR_SEGMENTID_USED,
    SCI_ERR_SYSTEM,
    SCI_ERR_CANCELLED,
    SCI_ERR_NOT_CONNECTED,
    SCI_ERR_NOT_AVAILABLE,
    SCI_ERR_INCONSISTENT_VERSIONS,
    SCI_ERR_COND_INT_RACE_PROBLEM,
    SCI_ERR_OVERFLOW,
    SCI_ERR_NOT_INITIALIZED,
    SCI_ERR_ACCESS,
    SCI_ERR_NOT_SUPPORTED,
    SCI_ERR_DEPRECATED,
    SCI_ERR_NO_SUCH_NODEID,
    SCI_ERR_NODE_NOT_RESPONDING,
    SCI_ERR_NO_REMOTE_LINK_ACCESS,
    SCI_ERR_NO_LINK_ACCESS,
    SCI_ERR_TRANSFER_FAILED,
    SCI_ERR_EWOULD_BLOCK,
    SCI_ERR_SEMAPHORE_COUNT_EXCEEDED,
    SCI_ERR_IRQL_ILLEGAL,
    SCI_ERR_REMOTE_BUSY,
    SCI_ERR_LOCAL_BUSY,
    SCI_ERR_ALL_BUSY
};


/* Corresponding error strings */
static const char* error_strings[] = {
    "OK",
    "Resource busy",
    "Flag option is not implemented",
    "Illegal flag option",
    "Out of local resources",
    "Out of local API resources",
    "Out of hardware resources",
    "Not implemented",
    "Illegal adapter number",
    "Adapter not found",
    "Operation timed out",
    "Out of range",
    "Segment ID not found",
    "Illegal node ID",
    "Connection to remote node is refused",
    "No connection to segment",
    "Size is not aligned",
    "Offset is not aligned",
    "Illegal function parameter",
    "Maximum possible physical mapping is exceeded",
    "Segment is not prepared",
    "Illegal address",
    "Illegal operation",
    "Illegal query operation",
    "Segment ID already used",
    "Could not get requested resource from the system",
    "Operation cancelled",
    "Host is not connected to remote host",
    "Operation not available",
    "Inconsistent driver version",
    "Out of local resources",
    "Host not initialized",
    "No local or remote access for requested operation",
    "Request not supported",
    "Function deprecated",
    "Node ID not found",
    "Node does not respond",
    "Remote link is not operational",
    "Local link is not operational",
    "Transfer failed",
    "Illegal interrupt line",
    "Remote host is busy",
    "Local host is busy",
    "System is busy"
};


static const char* bench_names[] = {
    "dma-push",
    "dma-push-global",
    "dma-pull",
    "dma-pull-global",
    "sciwrite",
    "scimemcpy-push",
    "scimemcpy-pull",
    "write",
    "read",
    "data-interrupt",
    NULL
};


static const char* bench_descriptions[] = {
    "use DIS DMA engine to push data to remote host",
    "use DIS DMA engine to push data to remote host",
    "use DIS DMA engine to pull data from remote host",
    "use DIS DMA engine to pull data from remote host",
    "use SCIMemWrite to write data to remote host",
    "use SCIMemCpy to write data to remote host",
    "use SCIMemCpy to read data from remote host",
    "use memcpy to write data to remote host",
    "use memcpy to read data from remote host",
    "use data interrupts to send data to remote host",
    NULL
};


bench_mode_t all_benchmarking_modes[] = {
    BENCH_SCI_DMA_PUSH_TO_REMOTE,
    BENCH_SCI_DMA_GLOBAL_PUSH_TO_REMOTE,
    BENCH_SCI_DMA_PULL_FROM_REMOTE,
    BENCH_SCI_DMA_GLOBAL_PULL_FROM_REMOTE,
    BENCH_SCI_WRITE_TO_REMOTE,
    BENCH_SCI_MEMCPY_TO_REMOTE,
    BENCH_SCI_MEMCPY_FROM_REMOTE,
    BENCH_WRITE_TO_REMOTE,
    BENCH_READ_FROM_REMOTE,
    BENCH_SCI_DATA_INTERRUPT,
    BENCH_DO_NOTHING
};



/* Lookup error string from SISCI error code */
const char* SCIGetErrorString(sci_error_t code)
{
    const size_t len = sizeof(error_codes) / sizeof(error_codes[0]);

    for (size_t idx = 0; idx < len; ++idx)
    {
        if (error_codes[idx] == code)
        {
            return error_strings[idx];
        }
    }

    return "Unknown error";
}


bench_mode_t bench_mode_from_name(const char* str)
{
    for (size_t i = 0; i < sizeof(all_benchmarking_modes) / sizeof(all_benchmarking_modes[0]) && bench_names[i] != NULL; ++i)
    {
        if (strcmp(str, bench_names[i]) == 0)
        {
            return all_benchmarking_modes[i];
        }
    }

    return BENCH_DO_NOTHING;
}


const char* bench_mode_name(bench_mode_t mode)
{
    for (size_t i = 0; i < sizeof(all_benchmarking_modes) / sizeof(all_benchmarking_modes[0]); ++i)
    {
        if (mode == all_benchmarking_modes[i])
        {
            return bench_names[i];
        }
    }

    return NULL;
}


const char* bench_mode_desc(bench_mode_t mode)
{
    for (size_t i = 0; i < sizeof(all_benchmarking_modes) / sizeof(all_benchmarking_modes[0]); ++i)
    {
        if (mode == all_benchmarking_modes[i])
        {
            return bench_descriptions[i];
        }
    }

    return NULL;
}


uint64_t local_ioaddr(sci_local_segment_t segment)
{
    sci_error_t err = SCI_ERR_OK;
    sci_query_local_segment_t query;

    query.subcommand = SCI_Q_LOCAL_SEGMENT_IOADDR;
    query.segment = segment;

    SCIQuery(SCI_Q_LOCAL_SEGMENT, &query, 0, &err);

    if (err != SCI_ERR_OK)
    {
        log_error("Failed to query local segment: %s", SCIGetErrorString(err));
        return 0;
    }

    return query.data.ioaddr;
}


uint64_t local_phaddr(sci_local_segment_t segment)
{
    sci_error_t err = SCI_ERR_OK;
    sci_query_local_segment_t query;

    query.subcommand = SCI_Q_LOCAL_SEGMENT_PHYS_ADDR;
    query.segment = segment;

    SCIQuery(SCI_Q_LOCAL_SEGMENT, &query, 0, &err);

    if (err != SCI_ERR_OK)
    {
        log_error("Failed to query local segment: %s", SCIGetErrorString(err));
        return 0;
    }

    return query.data.ioaddr;
}


uint64_t remote_ioaddr(sci_remote_segment_t segment)
{
    sci_error_t err = SCI_ERR_OK;
    sci_query_remote_segment_t query;

    query.subcommand = SCI_Q_REMOTE_SEGMENT_IOADDR;
    query.segment = segment;

    SCIQuery(SCI_Q_REMOTE_SEGMENT, &query, 0, &err);

    if (err != SCI_ERR_OK)
    {
        log_error("Failed to query remote segment: %s", SCIGetErrorString(err));
        return 0;
    }

    return query.data.ioaddr;
}


sci_error_t make_gpu_segment(sci_desc_t sd, unsigned adapter, unsigned id, sci_local_segment_t* segment, size_t size, int gpu, void** buf)
{
    sci_error_t err = SCI_ERR_OK;

    *buf = gpu_malloc(gpu, size);
    if (*buf == NULL)
    {
        log_error("Insufficient resources to allocate GPU buffer");
        return SCI_ERR_NOSPC;
    }

    SCICreateSegment(sd, segment, id, size, NULL, NULL, SCI_FLAG_EMPTY, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to create segment: %s", SCIGetErrorString(err));
        gpu_free(gpu, *buf);
        return err;
    }

    void* devptr = gpu_devptr(gpu, *buf);
    devptr_set_sync_memops(devptr);

    SCIAttachPhysicalMemory(0, devptr, 0, size, *segment, SCI_FLAG_CUDA_BUFFER, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to attach physical memory: %s", SCIGetErrorString(err));
        SCIRemoveSegment(*segment, 0, &err);
        gpu_free(gpu, *buf);
        return err;
    }

    SCIPrepareSegment(*segment, adapter, 0, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to prepare segment: %s", SCIGetErrorString(err));
        SCIRemoveSegment(*segment, 0, &err);
        gpu_free(gpu, *buf);
        return err;
    }

    log_debug("GPU segment %u created with IO addr 0x%lu", id, local_ioaddr(*segment));
    return SCI_ERR_OK;
}


sci_error_t make_ram_segment(sci_desc_t sd, unsigned adapter, unsigned id, sci_local_segment_t* segment, size_t size, sci_map_t* map, void** buf)
{
    sci_error_t err = SCI_ERR_OK;

    SCICreateSegment(sd, segment, id, size, NULL, NULL, 0, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to create segment: %s", SCIGetErrorString(err));
        return err;
    }

    *buf = SCIMapLocalSegment(*segment, map, 0, size, NULL, 0, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to map local segment: %s", SCIGetErrorString(err));
        SCIRemoveSegment(*segment, 0, &err);
        return err;
    }

    SCIPrepareSegment(*segment, adapter, 0, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to prepare segment: %s", SCIGetErrorString(err));
        SCIUnmapSegment(*map, 0, &err);
        SCIRemoveSegment(*segment, 0, &err);
        return err;
    }

    log_debug("RAM segment %u created with IO addr 0x%lu", id, local_ioaddr(*segment));
    return SCI_ERR_OK;
}


void free_gpu_segment(sci_local_segment_t segment, int gpu, void* buf)
{
    sci_error_t err = SCI_ERR_OK;

    do
    {
        SCIRemoveSegment(segment, 0, &err);
    }
    while (err == SCI_ERR_BUSY);
    
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to remove segment: %s", SCIGetErrorString(err));
    }

    gpu_free(gpu, buf);
}


void free_ram_segment(sci_local_segment_t segment, sci_map_t map)
{
    sci_error_t err;

    do
    {
        SCIUnmapSegment(map, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

    if (err != SCI_ERR_OK)
    {
        log_error("Failed to unmap segment: %s", SCIGetErrorString(err));
    }

    do
    {
        SCIRemoveSegment(segment, 0, &err);
    }
    while (err == SCI_ERR_BUSY);
    
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to remove segment: %s", SCIGetErrorString(err));
    }
}
