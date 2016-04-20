#include <errno.h>
#include <string.h>
#include <sisci_api.h>
#include "simple_types.h"
#include "dma.h"
#include "log.h"

#ifndef MAX_DMA_VEC
#define DIS_DMA_MAX_VECLEN
#endif


// Internal wrapper for dis_dma_vec_t
struct dma_vector
{
    size_t          size;                   // total transfer size
    size_t          length;                 // number of vector entries
    dis_dma_vec_t   vector[MAX_DMA_VEC];    // vector entries
};


int CreateDmaVec(dma_vec_t* vec)
{
    dma_vec_t handle = (dma_vec_t) malloc(sizeof(struct dma_vector));
    if (handle == NULL)
    {
        error("Failed to allocate handle: %s", strerror(errno));
        return -1;
    }

    handle->size = handle->length = 0;
    *vec = handle;
    return 0;
}


int RemoveDmaVec(dma_vec_t vec)
{
    free(vec);
    return 0;
}


int AppendDmaVec(dma_vec_t vec, size_t loff, size_t roff, size_t size, unsigned flags)
{
    if (vec->length >= sizeof(vec->vector) / sizeof(vec->vector[0]))
    {
        warn("Already reached maximum number of entries");
        return -1;
    }

    vec->vector[vec->length].size = size;
    vec->vector[vec->length].local_offset = loff;
    vec->vector[vec->length].remote_offset = roff;
    vec->vector[vec->length].flags = flags;

    vec->size += size;

    return 0;
}


int TransferDmaVec(dma_vector_t vec, l_segment_t lseg, r_segment_t rseg, unsigned flags, trans_cb_t cb, void* data)
{
    return 0;
}


int DmaRead(l_segment_t lseg, size_t loff, r_segment_t rseg, size_t roff, size_t size, trans_cb_t cb, void* data)
{
    int err = 0;
    dma_vec_t vector;
    
    if ((err = CreateDmaVec(&vector)) != 0)
    {
        return err;
    }

    if ((err = AppendDmaVec(vector, loff, roff, size, 0)) != 0)
    {
        return err;
    }

    if ((err = TransferDmaVec(vector, lseg, rseg, SCI_FLAG_DMA_READ, cb, data)) != 0)
    {
        return err;
    }

    if ((err = RemoveDmaVec(vector)) != 0)
    {
        return err;
    }

    return 0;
}


int DmaWrite(l_segment_t lseg, size_t loff, r_segment_t rseg, size_t roff, size_t size, trans_cb_t cb, void* data)
{
    int err = 0;
    dma_vec_t vector;
    
    if ((err = CreateDmaVec(&vector)) != 0)
    {
        return err;
    }

    if ((err = AppendDmaVec(vector, loff, roff, size, 0)) != 0)
    {
        return err;
    }

    if ((err = TransferDmaVec(vector, lseg, rseg, 0, cb, data)) != 0)
    {
        return err;
    }

    if ((err = RemoveDmaVec(vector)) != 0)
    {
        return err;
    }

    return 0;
}
