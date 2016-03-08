#ifndef __SIMPLE_TYPES_H__
#define __SIMPLE_TYPES_H__


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
 * Handle for the internal DMA vector descriptor.
 *
 * \note This type serves as a simplified handle type for the
 *       \c sci_dma_queue_t and \c dis_dma_vec_t types.
 */
typedef struct dma_vector* dma_vec_t;


#endif
