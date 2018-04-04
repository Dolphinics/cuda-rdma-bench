#include "util.h"
#include <sisci_api.h>
#include <string.h>
#include "common.h"
#include "reporting.h"
#include "gpu.h"
#include "bench.h"


static const char* bench_names[] = {
    "dma-push",
    "global-dma-push",
    "dma-pull",
    "global-dma-pull",
    "scimemwrite",
    "scimemcpy-write",
    "scimemcpy-read",
    "write",
    "read",
    NULL
};


static const char* bench_descriptions[] = {
    "use DMA to push data to remote host",
    "use DMA to push data to remote host (global)",
    "use DMA to pull data from remote host",
    "use DMA to pull data from remote host (global)",
    "use SCIMemWrite to write data to remote host",
    "use SCIMemCpy to write data to remote host",
    "use SCIMemCpy to read data from remote host",
    "use glibc memcpy / cudaMemcpy to write data to remote host",
    "use glibc memcpy / cudaMemcpy to read data from remote host",
    NULL
};


bench_mode_t all_benchmarking_modes[] = {
    BENCH_DMA_PUSH_TO_REMOTE,
    BENCH_DMA_PUSH_TO_REMOTE_G,
    BENCH_DMA_PULL_FROM_REMOTE,           
    BENCH_DMA_PULL_FROM_REMOTE_G,
    BENCH_SCIMEMWRITE_TO_REMOTE,          
    BENCH_SCIMEMCPY_TO_REMOTE,            
    BENCH_SCIMEMCPY_FROM_REMOTE,          
    BENCH_WRITE_TO_REMOTE,                
    BENCH_READ_FROM_REMOTE,               
    BENCH_DO_NOTHING
};


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


sci_error_t make_gpu_segment(sci_desc_t sd, unsigned adapter, unsigned id, sci_local_segment_t* segment, size_t size, const gpu_info_t* gpu, void** buf, int gl)
{
    unsigned flags = gl ? SCI_FLAG_DMA_GLOBAL : 0;
    sci_error_t err = SCI_ERR_OK, 
                tmp = SCI_ERR_OK;

    *buf = gpu_malloc(gpu->id, size);
    if (*buf == NULL)
    {
        log_error("Insufficient resources to allocate GPU buffer");
        return SCI_ERR_NOSPC;
    }

    SCICreateSegment(sd, segment, id, size, NULL, NULL, flags | SCI_FLAG_EMPTY, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to create segment: %s", SCIGetErrorString(err));
        gpu_free(gpu->id, *buf);
        return err;
    }

    void* devptr = gpu_devptr(gpu->id, *buf);
    devptr_set_sync_memops(devptr);

    SCIAttachPhysicalMemory(0, devptr, 0, size, *segment, SCI_FLAG_CUDA_BUFFER, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to attach physical memory: %s", SCIGetErrorString(err));
        SCIRemoveSegment(*segment, 0, &tmp);
        gpu_free(gpu->id, *buf);
        return err;
    }

    SCIPrepareSegment(*segment, adapter, 0, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to prepare segment: %s", SCIGetErrorString(err));
        SCIRemoveSegment(*segment, 0, &tmp);
        gpu_free(gpu->id, *buf);
        return err;
    }

    log_debug("GPU segment %x created with IO addr 0x%lx (size=%zu, global=%u)", 
            id, local_ioaddr(*segment), size, !!gl);
    return SCI_ERR_OK;
}


sci_error_t make_ram_segment(sci_desc_t sd, unsigned adapter, unsigned id, sci_local_segment_t* segment, size_t size, sci_map_t* map, void** buf, int gl)
{
    unsigned flags = gl ? SCI_FLAG_DMA_GLOBAL : 0;
    sci_error_t err, tmp;
    err = tmp = SCI_ERR_OK;

    SCICreateSegment(sd, segment, id, size, NULL, NULL, flags, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to create segment: %s", SCIGetErrorString(err));
        return err;
    }

    *buf = SCIMapLocalSegment(*segment, map, 0, size, NULL, 0, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to map local segment: %s", SCIGetErrorString(err));
        SCIRemoveSegment(*segment, 0, &tmp);
        return err;
    }

    SCIPrepareSegment(*segment, adapter, 0, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to prepare segment: %s", SCIGetErrorString(err));
        SCIUnmapSegment(*map, 0, &tmp);
        SCIRemoveSegment(*segment, 0, &tmp);
        return err;
    }

    log_debug("RAM segment %x created with IO addr 0x%lx (size=%zu, global=%u)", 
            id, local_ioaddr(*segment), size, !!gl);
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
    sci_error_t err = SCI_ERR_OK;

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

