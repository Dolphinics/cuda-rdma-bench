#ifndef __SIMPLE_TYPES_H__
#define __SIMPLE_TYPES_H__

#include <stdint.h>
#include <stddef.h>


/**
 * \brief Local segment descriptor handle
 *
 * Handle for the internal local segment descriptor. 
 *
 * \note This type serves as a simplified handle type for the 
 *       \c sci_local_segment_t SISCI type.
 */
typedef struct local_segment* l_segment_t;


/**
 * \brief Remote segment descriptor handle
 *
 * Handle for the internal remote segment descriptor.
 *
 * \note This type serves as as simplified handle type for the
 *       \c sci_remote_segment_t SISCI type.
 */
typedef struct remote_segment* r_segment_t;


/**
 * \brief DMA vector descriptor handle
 *
 * Handle for the internal DMA vector descriptor. DMA vectors are reusable
 * between different local-remote segment pairs.
 *
 * \note This type serves as a simplified handle type for the
 *       \c sci_dma_queue_t and \c dis_dma_vec_t types.
 */
typedef struct dma_vector* dma_vec_t;


/**
 * \brief Transfer status
 *
 * Indicates whether a memory transfer was successful or not.
 */
typedef enum { TRANSFER_SUCCESS, TRANSFER_FAILURE } trans_status_t;


/**
 * \brief Transfer information
 *
 * Statistics and information about a DMA transfer.
 */
struct trans_info {
    trans_status_t  transferStatus; 
    r_segment_t     remoteSegment;
    l_segment_t     localSegment;
    size_t          bytesTransferred;
    uint64_t        transferTime;
};


/**
 * \brief Transfer comlpetion callback type
 *
 * User-implemented transfer callback function for checking the result of
 * a memory transfer.
 *
 * \param[in]       transferStatus  transfer status
 * \param[in,out]   userData        additional user-supplied callback data
 * \param[in]       transferInfo    additional transfer status
 */
typedef void (*trans_cb_t)(trans_status_t transferStatus, void* userData, const struct trans_info* transferInfo);


#endif
