#ifndef __DMA_H__
#define __DMA_H__
#ifdef __cplusplus
extern "C" {
#endif

#include "types.h"


/**
 * \brief Create a DMA vector
 *
 * Create a DMA vector for transferring between a local and a remote segment.
 * 
 * \param[out]  vector          DMA vector descriptor handle
 * \param[in]   localSegment    local segment descriptor handle
 * \param[in]   remoteSegment   remote segment descriptor handle
 *
 * \returns \c 0 on success
 */
int CreateDmaVec(dma_vec_t* vector, l_segment_t localSegment, r_segment_t remoteSegment);


/**
 * \brief Remove a DMA vector
 *
 * Remove and clean up a DMA vector. This function destroys the DMA 
 * vector descriptor.
 *
 * \param[in]   vector          DMA vector descriptor handle
 *
 * \returns \c 0 on success
 */
int RemoveDmaVec(dma_vec_t vector);


/**
 * \brief Insert a DMA vector entry
 *
 * Insert a DMA vector entry into the DMA transfer vector.
 *
 * \param[in]   vector          DMA vector descriptor handle
 * \param[in]   localOff        offset in the local segment
 * \param[in]   remoteOff       offset in the remote segment
 * \param[in]   size            total transfer size
 * \param[in]   flags           additional flags for the \c dis_dma_vec_t entry
 *
 * \returns \c 0 on success
 */
int InsertDmaVecEntry(dma_vec_t vector, size_t localOff, size_t remoteOff, size_t size, unsigned flags);


/**
 * \brief Transfer DMA vector synchronously
 *
 * Transfer the DMA vector synchronously. This function blocks until the 
 * transfer is complete.
 *
 * \param[in]   vector          DMA vector descriptor handle
 * \param[in]   flags           additional flags for \c SCIStartDmaTransferVec
 *
 * \returns \c 0 on success
 */
int TransferDmaVecSync(dma_vec_t vector, unsigned flags);


/**
 * \brief Transfer DMA vector asynchronically
 *
 * Transfer the DMA vector asynchronically. This function does not block until
 * the transfer is complete and returns immediatly.
 *
 * \param[in]   vector          DMA vector descriptor handle
 * \param[in]   flags           additional flags for \c SCIStartDmaTransferVec
 *
 * \returns \c 0 on success
 */
int TransferDmaVec(dma_vec_t vector, unsigned flags);


/**
 * \brief Simplied DMA read
 *
 * Transfer data from remote segment to local segment. This function does not
 * block until transfer is complete and returns immediatly.
 *
 * \param[in]   localSegment    local segment descriptor handle
 * \param[in]   localOff        offset in the local segment
 * \param[in]   remoteSegment   remote segment descriptor handle
 * \param[in]   remoteOff       offset in the remote segment
 * \param[in]   size            total transfer size
 *
 * \returns \c 0 on success
 */
int DmaTransferRead(l_segment_t localSegment, size_t localOff, r_segment_t remoteSegment, size_t remoteOff, size_t size);


/**
 * \brief Simplied DMA write
 *
 * Transfer data from local segment to local segment. This function does not
 * block until transfer is complete and returns immediatly.
 *
 * \param[in]   localSegment    local segment descriptor handle
 * \param[in]   localOff        offset in the local segment
 * \param[in]   remoteSegment   remote segment descriptor handle
 * \param[in]   remoteOff       offset in the remote segment
 * \param[in]   size            total transfer size
 *
 * \returns \c 0 on success
 */
int DmaTransferWrite(l_segment_t localSegment, size_t localOff, r_segment_t remoteSegment, size_t remoteOff, size_t size);



#ifdef __cplusplus
}
#endif
#endif
