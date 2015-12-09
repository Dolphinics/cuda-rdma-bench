#include <stdlib.h>
#include <sisci_api.h>
#include <stdint.h>
#include "common.h"
#include "reporting.h"



// TODO: rename to sisci_dma and change dma_mode_t definition as well as make --mode=sisci --mode=cuda arguments
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

    dis_dma_vec_t vec[repeat];

    for (int i = 0; i < repeat; ++i)
    {
        vec[i].size = size / repeat;
        vec[i].local_offset = (size / repeat) * i;
        vec[i].remote_offset = (size / repeat) * i;
        vec[i].flags = 0;
    }

    uint64_t start = current_usecs();
    SCIStartDmaTransferVec(q, local, remote, repeat, vec, NULL, NULL,  SCI_FLAG_DMA_WAIT | flags, &err);
    uint64_t end = current_usecs();

    if (err != SCI_ERR_OK)
    {
        log_error("Failed transfer! %s", SCIGetErrorString(err));
        return 0;
    }
    
    SCIRemoveDMAQueue(q, 0, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to remove DMA queue: %s", SCIGetErrorString(err));
    }

    return end - start;
}



uint64_t benchmark(
        sci_desc_t sd, unsigned remote_node_id, unsigned adapter, 
        sci_local_segment_t local, sci_remote_segment_t remote, size_t size, 
        dma_mode_t mode, unsigned flags, int repeat)
{
    uint64_t usecs = 0;

    switch (mode)
    {
        case DMA_TRANSFER_ONE_WAY:
            usecs = one_way(sd, adapter, local, remote, size, flags, repeat);
            break;

        case DMA_TRANSFER_TWO_WAY:
            // TODO: Implement this
            log_error("two way transfer not implemented yet");
            break;

        default:
            log_error("Unknown DMA transfer mode");
            break;
    }

    return usecs;
}
