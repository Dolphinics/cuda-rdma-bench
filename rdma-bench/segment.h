#ifndef __LOCAL_SEGMENT_H__
#define __LOCAL_SEGMENT_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>


/* Hosted segment descriptor handle */
typedef struct local_segment* lsegment_t;


/* Reserve and allocate a hosted memory segment and initialize the local 
 * segment descriptor.
 *
 * handle           - the segment descriptor reference
 * localAdapterNo   - the local SCI adapter identifier
 * id               - the hosted segment identifier
 * size             - size of the hosted segment
 * flags            - SISCI flags (usually 0)
 *
 * Returns 0 on success or a non-zero value on failure.
 */
int AllocHostedSegment(lsegment_t* handle, unsigned localAdapterNo, unsigned id, size_t size, unsigned flags);


/* Reserve, but do not allocate, a hosted memory segment and initialize 
 * the local segment descriptor.
 *
 * handle           - the segment descriptor reference
 * localAdapterNo   - the local SCI adapter identifier
 * id               - the hosted segment identifier
 * size             - size of the hosted segment
 * flags            - SISCI flags (usually 0)
 *
 * Returns 0 on success or a non-zero value on failure.
 */
int ReserveHostedSegment(lsegment_t* handle, unsigned localAdapterNo, unsigned id, size_t size, unsigned flags);


/* Free a hosted memory segment.
 *
 * handle           - the segment descriptor refernce
 *
 * No return value.
 */
void FreeHostedSegment(lsegment_t handle);


/* Attach physical memory, i.e. a CUDA device pointer, to the hosted 
 * memory segment.
 *
 * handle           - the segment descriptor reference
 * phys             - physical memory address
 * flags            - SISCI flags (usually SISI)
 *
 * Returns 0 on success or a non-zero value on failure.
 */
int AttachPhysMem(lsegment_t handle, void* phys, unsigned flags);


/* Attach virtual memory, i.e. malloc'd memory, to the hosted 
 * memory segment.
 *
 * handle           - the segment descriptor reference
 * virt             - virtual memory address
 * flags            - SISCI flags (usually 0)
 *
 * Returns 0 on success or a non-zero value on failure.
 */
int AttachVirtMem(lsegment_t handle, void* virt, unsigned flags);


/* Expose the hosted segment to the world.
 *
 * handle           - the segment descriptor reference
 *
 * Returns 0 on success or a non-zero value on failure.
 */
int SetHostedSegmentAvailable(lsegment_t handle);


/* Unexpose the hosted segment.
 *
 * handle           - the segment descriptor reference
 *
 * No return value.
 */
void SetHostedSegmentUnavailable(lsegment_t handle);


/* Get the associated memory pointer for a hosted memory segment.
 *
 * handle           - the segment descriptor reference
 *
 * Returns a memory pointer on success or NULL on failure.
 */
void* GetHostedSegmentMemPtr(lsegment_t handle);


#ifdef __cplusplus
}
#endif
#endif
