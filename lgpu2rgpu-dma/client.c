#include <stdlib.h>
#include <sisci_api.h>
#include <stdint.h>
#include "common.h"
#include "local.h"
#include "reporting.h"
#include <unistd.h>



static size_t size_factor = 0; // FIXME: Fix this ugly hack



sci_callback_action_t dma_complete(void* ready, sci_dma_queue_t queue, sci_error_t status)
{
    *((int*) ready) = 1;
    
    // TODO: check status

    return SCI_CALLBACK_CONTINUE;
}



void benchmark_one_way(sci_desc_t sd, unsigned adapter_id, sci_local_segment_t local, sci_remote_segment_t remote, size_t size, unsigned flags, unsigned repeat)
{
    sci_error_t err;
    sci_dma_queue_t q;

    SCICreateDMAQueue(sd, &q, adapter_id, 1, 0, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to create DMA queue: %s", SCIGetErrorString(err));
        exit(1);
    }

    uint64_t start, end;
    double megabytes_per_sec;

    uint64_t total_time = 0;

    for (unsigned i = 0; i < repeat; ++i)
    {
        int ready = 0;

        start = current_usecs();
        SCIStartDmaTransfer(q, local, remote, 0, size, 0, &dma_complete, &ready, SCI_FLAG_USE_CALLBACK | flags, &err);

        if (err != SCI_ERR_OK)
        {
            log_error("DMA transfer failed: %s", SCIGetErrorString(err));
            exit(1);
        }

        while (!ready);
        end = current_usecs();

        total_time += (end - start);
        megabytes_per_sec = (double) size / (double) (end - start);

        fprintf(stdout, "%4d %05.3f %-5s\n", i, megabytes_per_sec, size_factor == 1e6 ? "MB/s" : "MiB/s");
    }
    double total_megabytes_per_sec = (size * repeat) / (double) total_time;
    fprintf(stdout, "%4s %06.3f %-5s\n", "Acc.", total_megabytes_per_sec, size_factor == 1e6 ? "MB/s" : "MiB/s");

    SCIRemoveDMAQueue(q, 0, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to remove DMA queue: %s", SCIGetErrorString(err));
        exit(1);
    }
}



sci_remote_segment_t make_remote_buffer(sci_desc_t sd, unsigned segment_id, unsigned remote_node_id, unsigned adapter_id)
{
    sci_error_t err;
    sci_remote_segment_t segment;

    log_debug("Connecting to remote segment %u on remote cluster node %u", segment_id, remote_node_id);
    do
    {
        SCIConnectSegment(sd, &segment, remote_node_id, segment_id, adapter_id, NULL, NULL, SCI_INFINITE_TIMEOUT, 0, &err);
    }
    while (err != SCI_ERR_OK);

    return segment;
}



void run_client(client_args* args, size_t factor, unsigned repeat)
{
    sci_error_t err;

    sci_remote_segment_t r_segment = make_remote_buffer(args->desc, args->remote_segment_id, args->remote_node_id, args->adapter_id);

    size_t segment_size = SCIGetRemoteSegmentSize(r_segment);
    log_debug("Connected to segment %u on remote node %u (%lu bytes)", args->remote_segment_id, args->remote_node_id, segment_size);

    bufhandle_t bh;
    bh = create_gpu_buffer(args->desc, args->adapter_id, args->gpu_dev_id, args->local_segment_id, segment_size, args->gpu_mem_flags);

    size_factor = factor;

    sci_remote_interrupt_t trigger_irq;
    SCIConnectInterrupt(args->desc, &trigger_irq, args->remote_node_id, args->adapter_id, args->remote_segment_id, SCI_INFINITE_TIMEOUT, 0, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to connect to remote interrupt");
        exit(1);
    }

    if (!!(args->dma_flags & SCI_FLAG_DMA_READ))
    {
        log_debug("Buffer byte (before): %02x", validate_buffer(bh));
    }

    // Dump parameters
    fprintf(stdout, "dir: %4s -- mode: %4s -- global: %3s -- hmem: %3s -- size: %5.2f %-3s\n",
            args->dma_mode == DMA_TRANSFER_ONE_WAY ? "1way" : "2way", 
            !!(args->dma_flags & SCI_FLAG_DMA_READ) ? "pull" : "push", 
            !!(args->dma_flags & SCI_FLAG_DMA_GLOBAL) ? "yes" : "no",
            args->gpu_mem_flags != 0 ? "yes" : "no",
            (double) segment_size / (double) factor,
            factor == 1e6 ? "MB" : "MiB"
            );

    switch (args->dma_mode)
    {
        case DMA_TRANSFER_ONE_WAY:
            benchmark_one_way(args->desc, args->adapter_id, bh.segment, r_segment, segment_size, args->dma_flags, repeat);
            break;

        case DMA_TRANSFER_TWO_WAY:
            // TODO:
            break;

        default:
            log_error("Unknown DMA transfer mode, aborting.");
            break;
    }
    
    uint8_t byte = validate_buffer(bh);

    if (!!(args->dma_flags & SCI_FLAG_DMA_READ))
    {
        log_debug("Buffer byte (after) : %02x\n", byte);
    }
    else
    {
        SCITriggerInterrupt(trigger_irq, 0, &err);
    }

    free_gpu_buffer(bh);
    SCIDisconnectInterrupt(trigger_irq, 0, &err);
    do
    {
        SCIDisconnectSegment(r_segment, 0, &err);
    }
    while (err == SCI_ERR_BUSY);
}