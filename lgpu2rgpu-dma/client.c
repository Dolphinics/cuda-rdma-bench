#include <stdint.h>
#include <stdlib.h>
#include <sisci_api.h>
#include "translist.h"
#include "common.h"
#include "util.h"
#include "reporting.h"
#include "bench.h"
#include "gpu.h"
#include "ram.h"


static int verify_transfer(translist_desc_t* desc)
{
    sci_error_t err;
    sci_map_t remote_buf_map;

    volatile void* remote_ptr;
    remote_ptr = SCIMapRemoteSegment(desc->segment_remote, &remote_buf_map, 0, desc->segment_size, NULL, 0, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to map remote segment: %s", SCIGetErrorString(err));
        return -1;
    }

    log_info("Comparing local and remote memory...");
    size_t bytes;
    if (desc->local_gpu_info != NULL)
    {
        bytes = gpu_memcmp(desc->local_gpu_info->id, desc->buffer_ptr, remote_ptr, desc->segment_size);
    }
    else
    {
        bytes = ram_memcmp(desc->buffer_ptr, remote_ptr, desc->segment_size);
    }

    do
    {
        SCIUnmapSegment(remote_buf_map, 0, &err);
    }
    while (err == SCI_ERR_BUSY);
    
    if (err != SCI_ERR_OK)
    {
        log_error("Unexpected error: %s", SCIGetErrorString(err));
    }

    return bytes == desc->segment_size;
}


static size_t create_dma_vec(translist_t tl, dis_dma_vec_t* vec)
{
    size_t veclen = translist_size(tl);
    size_t total_size = 0;

    for (size_t i = 0; i < veclen; ++i)
    {
        translist_entry_t entry;

        translist_element(tl, i, &entry);
        
        vec[i].size = entry.size;
        vec[i].local_offset = entry.offset_local;
        vec[i].remote_offset = entry.offset_remote;
        vec[i].flags = 0;

        total_size += entry.size;
    }

    return total_size;
}


static void SCIMemWrite_wrapper(void* local_ptr, volatile void* remote_ptr, size_t len)
{
    sci_error_t err;

    SCIMemWrite(local_ptr, remote_ptr, len, 0, &err);
   
    if (err != SCI_ERR_OK)
    {
        log_error("SCIMemWrite failed");
    }
}


void pio(translist_t tl, translist_desc_t* td, unsigned flags, size_t repeat, result_t* result)
{
    sci_error_t err;
    sci_map_t remote_buf_map;
    volatile void* remote_buffer_ptr;

    // Map remote segment
    remote_buffer_ptr = SCIMapRemoteSegment(td->segment_remote, &remote_buf_map, 0, td->segment_size, NULL, 0, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to map remote segment");
        return;
    }

    size_t veclen = translist_size(tl);
    dis_dma_vec_t vec[veclen];

    // Find out which memcpy function to use
    void (*memcpy_func)(void*, volatile void*, size_t) = NULL;

    switch (flags)
    {
        case 0:
            memcpy_func = &ram_memcpy_local_to_remote;
            if (td->local_gpu_info != NULL)
            {
                gpu_prepare_memcpy(td->local_gpu_info->id);
                memcpy_func = &gpu_memcpy_local_to_remote;
            }
            break;

        case 1:
            memcpy_func = &ram_memcpy_remote_to_local;
            if (td->local_gpu_info != NULL)
            {
                gpu_prepare_memcpy(td->local_gpu_info->id);
                memcpy_func = &gpu_memcpy_remote_to_local;
            }
            break;

        case 2:
        default:
            memcpy_func = &SCIMemWrite_wrapper;
            if (td->local_gpu_info != NULL)
            {
                log_error("SCIMemWrite for local GPU is not implemented because cudaHostAlloc is not used");
                goto release;
            }
            break;
    }
        
    // Create transfer vector
    result->total_size = create_dma_vec(tl, vec);

    // Do PIO transfer
    uint8_t* local_ptr = (uint8_t*) td->buffer_ptr;
    volatile uint8_t* remote_ptr = (volatile uint8_t*) remote_buffer_ptr;

    uint64_t total_start = ts_usecs();
    for (size_t i = 0; i < repeat; ++i)
    {
        uint64_t start = ts_usecs();
        for (size_t i = 0; i < veclen; ++i)
        {
            memcpy_func(local_ptr + vec[i].local_offset, remote_ptr + vec[i].remote_offset, vec[i].size);
        }
        uint64_t end = ts_usecs();

        result->runtimes[i] = end - start;
        result->success_count++;
    }
    uint64_t total_end = ts_usecs();

    result->total_runtime = total_end - total_start;

release:
    // Release remote segment
    do
    {
        SCIUnmapSegment(remote_buf_map, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

    if (err != SCI_ERR_OK)
    {
        log_error("Failed to unmap remote segment");
    }
}


void dma(unsigned adapter, translist_t tl, translist_desc_t* tsd, unsigned flags, size_t repeat, result_t* result)
{
    sci_error_t err;
    sci_dma_queue_t q;
    size_t veclen = translist_size(tl);

    // Create DMA queue
    SCICreateDMAQueue(tsd->sisci_desc, &q, adapter, 1, 0, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to create DMA queue");
        return;
    }

    // Create DMA transfer vector
    dis_dma_vec_t vec[veclen];
    result->total_size = create_dma_vec(tl, vec);

    // Do DMA transfer
    log_debug("Performing DMA transfer of %lu-sized vector  %d times", veclen, repeat);
    result->total_runtime = 0;
    result->success_count = 0;

    uint64_t start = ts_usecs();
    for (size_t i = 0; i < repeat; ++i)
    {
        uint64_t before = ts_usecs();
        SCIStartDmaTransferVec(q, tsd->segment_local, tsd->segment_remote, veclen, vec, NULL, NULL, SCI_FLAG_DMA_WAIT | flags, &err);
        uint64_t after = ts_usecs();

        result->runtimes[i] = after - before;

        if (err != SCI_ERR_OK)
        {
            log_error("DMA transfer failed %s", SCIGetErrorString(err));
            result->runtimes[i] = 0;
            continue;
        }

        result->success_count++;
    }
    uint64_t end = ts_usecs();

    result->total_runtime = end - start;

    // Clean up and quit
    SCIRemoveDMAQueue(q, 0, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to remove DMA queue");
    }
}


int client(unsigned adapter, const bench_t* benchmark, result_t* result)
{
    translist_desc_t tl_desc = translist_desc(benchmark->transfer_list);

    // Fill local buffer with random byte
    uint8_t byte = random_byte_value();

    log_debug("Creating buffer and filling with random value %02x", byte);
    if (tl_desc.local_gpu_info != NULL)
    {
        gpu_memset(tl_desc.local_gpu_info->id, tl_desc.buffer_ptr, tl_desc.segment_size, byte);
    }
    else
    {
        ram_memset(tl_desc.buffer_ptr, tl_desc.segment_size, byte);
    }

    // Blank out benchmark results
    result->total_runtime = 0;
    result->success_count = 0;
    result->buffer_matches = 0;
    result->total_size = 0;
    for (size_t i = 0; i < benchmark->num_runs; ++i)
    {
        result->runtimes[i] = 0;
    }

    // Do benchmark
    unsigned fetch_data = 0;
    unsigned sci_flags = 0;

    log_info("Executing benchmark...");
    switch (benchmark->benchmark_mode)
    {
        case BENCH_DMA_PULL_FROM_REMOTE:
            sci_flags |= SCI_FLAG_DMA_READ;
            fetch_data = 1;
            /* intentional fall-through */
        case BENCH_DMA_PUSH_TO_REMOTE:
            dma(adapter, benchmark->transfer_list, &tl_desc, sci_flags, benchmark->num_runs, result);
            break;

        case BENCH_READ_FROM_REMOTE:
            fetch_data = 1;
            /* intentional fall-through */
        case BENCH_WRITE_TO_REMOTE:
            pio(benchmark->transfer_list, &tl_desc, fetch_data, benchmark->num_runs, result);
            break;
            
        case BENCH_SCIMEMWRITE_TO_REMOTE:
            pio(benchmark->transfer_list, &tl_desc, 2, benchmark->num_runs, result);
            break;

        case BENCH_DO_NOTHING:
            log_error("No benchmark type is set");
            return -1;

        default:
            log_error("%s is not yet supported", bench_mode_name(benchmark->benchmark_mode));
            return -2;

    }
    log_info("Benchmark complete, verifying transfer.");

    // Trigger remote interrupt to make remote host check its buffer
    if (!fetch_data)
    {
        sci_error_t err;

        SCITriggerInterrupt(tl_desc.validate, 0, &err);
        if (err != SCI_ERR_OK)
        {
            log_error("Failed to trigger remote interrupt");
        }
    }

    // Verify transfer by comparing local and remote buffer
    uint8_t value;
    if (tl_desc.local_gpu_info != NULL)
    {
        gpu_memcpy_buffer_to_local(tl_desc.local_gpu_info->id, tl_desc.buffer_ptr, &value, 1);
    }
    else
    {
        value = *((uint8_t*) tl_desc.buffer_ptr);
    }

    if (fetch_data)
    {
        report_buffer_change(stderr, byte, value);
    }
    
    if (verify_transfer(&tl_desc) != 1)
    {
        log_error("Local and remote buffers differ!!");
        result->buffer_matches = 0;
    }
    else
    {
        log_debug("Local and remote buffers are equal");
        result->buffer_matches = 1;
    }

    return 0;
}
