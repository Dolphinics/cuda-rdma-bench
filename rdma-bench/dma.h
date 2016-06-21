#ifndef __DMA_H__
#define __DMA_H__
#ifdef __cplusplus
extern "C" {
#endif

#include "simple_types.h"


/**
 * \brief Simplied DMA read
 *
 * Start DMA transfer from remote segment to local segment.
 * Data is transferred asynchronously.
 *
 * \param[in]       adapterNo       local adapter number
 * \param[in]       localSegment    local segment descriptor handle
 * \param[in]       localOffset     offset in the local segment
 * \param[in]       remoteSegment   remote segment descriptor handle
 * \param[in]       remoteOffset    offset in the remote segment
 * \param[in]       size            total transfer size
 * \param[in]       callback        user-defined callback (can be \c NULL)
 * \param[in]       data            additional user-supplied callback data (can be \c NULL)
 *
 * \returns \c 0 on success
 *
 * \note For most architectures, doing a DMA read is slower than write.
 *
 * \note A successful return value does not mean that data was successfully transferred.
 *       Supply a callback function to check the status of the transfer.
 */
int DmaRead(unsigned adapterNo, l_segment_t localSegment, size_t localOffset, r_segment_t remoteSegment, size_t remoteOffset, size_t size, trans_cb_t callback, void* data);


/**
 * \brief Simplied DMA write
 *
 * Start DMA transfer from local segment to remote segment. 
 * Data is transferred asynchronously.
 *
 * \param[in]       adapterNo       local adapter number
 * \param[in]       localSegment    local segment descriptor handle
 * \param[in]       localOffset     offset in the local segment
 * \param[in]       remoteSegment   remote segment descriptor handle
 * \param[in]       remoteOffset    offset in the remote segment
 * \param[in]       size            total transfer size
 * \param[in]       callback        user-defined callback (can be \c NULL)
 * \param[in]       data            additional user-supplied callback data (can be \c NULL)
 *
 * \returns \c 0 on success
 *
 * \note A successful return value does not mean that data was successfully transferred.
 *       Supply a callback function to check the status of the transfer.
 */
int DmaWrite(unsigned adapterNo, l_segment_t localSegment, size_t localOffset, r_segment_t remoteSegment, size_t remoteOffset, size_t size, trans_cb_t callback, void* data);


/**
 * \brief Create DMA vector
 *
 * Create a DMA vector. A DMA vector can be used for transfers between
 * multiple local-remote segment pairs.
 * 
 * \param[out]      vector          DMA vector descriptor handle
 *
 * \returns \c 0 on success
 */
int CreateDmaVec(dma_vec_t* vector);


/**
 * \brief Remove DMA vector
 *
 * Remove and clean up a DMA vector. This function destroys the DMA 
 * vector descriptor.
 *
 * \param[in]       vector          DMA vector descriptor handle
 *
 * \returns \c 0 on success
 */
int RemoveDmaVec(dma_vec_t vector);


/**
 * \brief Append DMA vector entry
 *
 * Append a vector entry to the DMA vector.
 *
 * \param[in]       vector          DMA vector descriptor handle
 * \param[in]       localOffset     offset in the local segment
 * \param[in]       remoteOffset    offset in the remote segment
 * \param[in]       size            total transfer size
 * \param[in]       flags           additional flags for the \c dis_dma_vec_t entry
 *
 * \returns \c 0 on success
 */
int AppendDmaVec(dma_vec_t vector, size_t localOffset, size_t remoteOffset, size_t size, unsigned flags);


/**
 * \brief Transfer DMA vector 
 *
 * Transfer a DMA vector between a local-remote segment pair using the DMA
 * engine on the adapter where the remote segment is connected.
 *
 * \param[in]       vector          DMA vector descriptor handle
 * \param[in]       adapterNo       local adapter number
 * \param[in]       localSegment    local segment descriptor handle
 * \param[in]       remoteSegment   remote segment descriptor handle
 * \param[in]       flags           additional flags for \c SCIStartDmaTransferVec
 * \param[in]       callback        user-defined callback (can be \c NULL)
 * \param[in]       data            additional user-supplied callback data (can be \c NULL)
 *
 * \returns \c 0 on success
 *
 * \note A successful return value does not mean that data was successfully transferred.
 *       Supply a callback function to check the status of the transfer.
 */
int TransferDmaVec(dma_vec_t vector, unsigned adapterNo, l_segment_t localSegment, r_segment_t remoteSegment, unsigned flags, trans_cb_t callback, void* data);


#ifdef __cplusplus
}
#endif
#endif
