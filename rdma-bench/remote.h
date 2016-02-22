#ifndef __REMOTE_SEGMENT_H__
#define __REMOTE_SEGMENT_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>


/* Connected segment descriptor handle */
typedef struct remote_segment* rsegment_t;


/* Connect to a remote hosted segment.
 *
 * handle           - the segment descriptor reference
 * localAdapterNo   - the local SCI adapter identifier
 * remoteNodeId     - remote node identifier
 * id               - remote segment identifier
 *
 * Returns 0 on success or a non-zero value on failure.
 */
int CreateConnectedSegment(rsegment_t* handle, unsigned localAdapterNo, unsigned remoteNodeId, unsigned id);


/* Disconnect from a connected segment.
 *
 * handle           - the segment descriptor reference
 *
 * No return value.
 */
void ReleaseConnectedSegment(rsegment_t handle);


/* Get the size of the remote segment.
 *
 * handle           - the segment descriptor reference
 *
 * Returns the size of the connected segment.
 */
size_t GetConnectedSegmentSize(rsegment_t handle);


/* Get the associated memory pointer for a remote segment.
 *
 * handle           - the segment descriptor reference
 *
 * Returns a memory pointer on success or NULL on failure.
 */
void* GetConnectedSegmentMemPtr(rsegment_t handle);


#ifdef __cplusplus
extern "C" }
#endif
#endif
