#include <errno.h>
#include <string.h>
#include <sisci_api.h>
#include <sisci_types.h>
#include <stdint.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include "simple_types.h"
#include "internal_types.h"
#include "dma.h"
#include "log.h"
#include "util.h"


struct callback_info
{
    trans_cb_t  user_func;  // pointer to user-supplied callback function
    void*       user_data;  // pointer to user-supplied callback data
    r_segment_t r_seg;      // remote segment handle
    l_segment_t l_seg;      // local segment handle
    uint64_t    ts_start;   // timestamp of transfer start
    size_t      bytes;      // number of bytes transferred
};


struct dma_vector
{
    size_t          size;                       // total transfer size
    size_t          length;                     // number of elements in vector
    dis_dma_vec_t   vector[DIS_DMA_MAX_VECLEN]; // vector elements
};


/* Handle callbacks from SISCI API */
static sci_callback_action_t callback(void* cbInfo, sci_dma_queue_t dmaQueue, sci_error_t status)
{
    uint64_t ts_now = 0; // TODO: get timestamp

    debug("DMA transfer complete");

    sci_error_t err;
    //SCIRemoveDMAQueue(dmaQueue, 0, &err);
    //if (err != SCI_ERR_OK)
    {
        //error("Failed to remove DMA queue: %s", GetErrorString(err));
    }

    if (status != SCI_ERR_OK)
    {
        warn("DMA transfer failed: %s", status);
    }

    struct callback_info* info = (struct callback_info*) cbInfo;
    if (info->user_func != NULL)
    {
        trans_status_t trans_status = status == SCI_ERR_OK ? TRANSFER_SUCCESS : TRANSFER_FAILURE;

        struct trans_info tinfo = {
            .transferStatus = trans_status,
            .remoteSegment = info->r_seg,
            .localSegment = info->l_seg,
            .bytesTransferred = info->bytes,
            .transferTime = ts_now - info->ts_start
        };
        
        info->user_func(trans_status, info->user_data, &tinfo);
    }

    free(info);
    //return SCI_CALLBACK_CANCEL;
    return SCI_CALLBACK_CONTINUE;
}



int CreateDmaVec(dma_vec_t* handle)
{
    dma_vec_t vector = (dma_vec_t) malloc(sizeof(struct dma_vector));
    if (vector == NULL)
    {
        error("Failed to allocate handle: %s", strerror(errno));
        return -errno;
    }

    vector->size = 0;
    vector->length = 0;
    *handle = vector;

    debug("Created DMA transfer vector handle");
    return 0;
}


int RemoveDmaVec(dma_vec_t handle)
{
    free(handle);
    return 0;
}


int AppendDmaVec(dma_vec_t vector, size_t loff, size_t roff, size_t size, unsigned flags)
{
    if (vector->length >= DIS_DMA_MAX_VECLEN)
    {
        warn("DMA transfer vector is already maximum size");
        return -1;
    }

    vector->vector[vector->length].size = size;
    vector->vector[vector->length].local_offset = loff;
    vector->vector[vector->length].remote_offset = roff;
    vector->vector[vector->length].flags = flags;

    vector->size += size;
    vector->length++;

    return 0;
}


int TransferDmaVec(dma_vec_t vector, unsigned adapt_no, l_segment_t lseg, r_segment_t rseg, unsigned flags, trans_cb_t cb, void* udata)
{
    sci_dma_queue_t queue;
    sci_error_t err;

    struct callback_info* info = (struct callback_info*) malloc(sizeof(struct callback_info));
    if (info == NULL)
    {
        error("Failed to create callback data: %s", strerror(errno));
        return -errno;
    }

    info->user_func = cb;
    info->user_data = udata;
    info->r_seg = rseg;
    info->l_seg = lseg;
    info->bytes = 0; // TODO
    info->ts_start = 0; // TODO

    SCICreateDMAQueue(lseg->sci_d, &queue, adapt_no, 1, 0, &err);
    if (err != SCI_ERR_OK)
    {
        free(info);
        error("Failed to create DMA queue: %s", GetErrorString(err));
        return -1;
    }

    debug("Executing DMA transfer");
    SCIStartDmaTransferVec(queue, lseg->seg_d, rseg->seg_d, vector->length, vector->vector, &callback, info, SCI_FLAG_USE_CALLBACK | flags, &err);
    if (err != SCI_ERR_OK)
    {
        error("Failed to start DMA transfer: %s", GetErrorString(err));
        free(info);
        SCIRemoveDMAQueue(queue, 0, &err);
    }

    return 0;
}


int DmaRead(unsigned adapt_no, l_segment_t lseg, size_t loff, r_segment_t rseg, size_t roff, size_t size, trans_cb_t cb, void* data)
{
    int err;
    dma_vec_t vector;

    if ((err = CreateDmaVec(&vector)) != 0)
    {
        return err;
    }

    if ((err = AppendDmaVec(vector, loff, roff, size, 0)) != 0)
    {
        RemoveDmaVec(vector);
        return err;
    }

    if ((err = TransferDmaVec(vector, adapt_no, lseg, rseg, SCI_FLAG_DMA_READ, cb, data)) != 0)
    {
        RemoveDmaVec(vector);
        return err;
    }

    RemoveDmaVec(vector);
    return 0;
}


int DmaWrite(unsigned adapt_no, l_segment_t lseg, size_t loff, r_segment_t rseg, size_t roff, size_t size, trans_cb_t cb, void* data)
{
    int err;
    dma_vec_t vector;

    if ((err = CreateDmaVec(&vector)) != 0)
    {
        return err;
    }

    if ((err = AppendDmaVec(vector, loff, roff, size, 0)) != 0)
    {
        RemoveDmaVec(vector);
        return err;
    }

    if ((err = TransferDmaVec(vector, adapt_no, lseg, rseg, 0, cb, data)) != 0)
    {
        RemoveDmaVec(vector);
        return err;
    }

    RemoveDmaVec(vector);
    return 0;
}
