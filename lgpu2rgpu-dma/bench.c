#include <stdlib.h>
#include <sisci_api.h>
#include <stdint.h>
#include "common.h"
#include "reporting.h"



uint64_t one_way(sci_desc_t sd, unsigned adapter, sci_local_segment_t local, sci_remote_segment_t remote, size_t size, unsigned flags, int repeat)
{
    sci_error_t err;
    sci_dma_queue_t q;

    SCICreateDMAQueue(sd, &q, adapter, 1, 0, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to create DMA queue: %s", SCIGetErrorString(err));
        exit(1);
    }

    uint32_t start, end;

    start = current_usecs();
    for (int i = 0; i < repeat; ++i)
    {
        SCIStartDmaTransfer(q, local, remote, 0, size, 0, NULL, NULL, SCI_FLAG_DMA_WAIT | flags, &err);
        if (err != SCI_ERR_OK)
        {
            log_error("Failed transfer! %s", SCIGetErrorString(err));
            break;
        }
    }
    end = current_usecs();
    
    SCIRemoveDMAQueue(q, 0, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to remove DMA queue: %s", SCIGetErrorString(err));
    }

    return end - start;
}



uint64_t benchmark(
        sci_desc_t sd, unsigned adapter, sci_local_segment_t local, sci_remote_segment_t remote, size_t size, 
        dma_mode_t mode, unsigned flags, int repeat)
{
    switch (mode)
    {
        case DMA_TRANSFER_ONE_WAY:
            return one_way(sd, adapter, local, remote, size, flags, repeat);

        case DMA_TRANSFER_TWO_WAY:
            // TODO: Implement this
            return 0;

        default:
            log_error("Unknown DMA transfer mode");
            return 0;
    }
    
}
