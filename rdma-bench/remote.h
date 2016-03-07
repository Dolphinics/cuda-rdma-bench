#ifndef __REMOTE_SEGMENT_H__
#define __REMOTE_SEGMENT_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>


/**
 * \briev Remote segment descriptor handle
 *
 * Handle for the internal remote segment descriptor.
 *
 * \note This type serves as as simplified handle type for the
 *       \c sci_remote_segment_t SISCI type.
 */
typedef struct remote_segment* r_segment_t;


/**
 * \brief Connect to a remote segment
 *
 * Connect to a remote segment and initialize the remote segment descriptor.
 * This function will block until the remote segment becomes available.
 *
 * \param[out]  segment         segment descriptor handle
 * \param[in]   adapterNo       indentifier for the local adapter the remote node is connected to
 * \param[in]   remoteNodeId    remote node identifier
 * \param[in]   segmentId       unique remote segment identifier
 *
 * \returns \c 0 on success
 */
int ConnectRemoteSegment(r_segment_t* segment, unsigned adapterNo, unsigned remoteNodeId, unsigned segmentId);


/**
 * \brief Disconnect from a remote segment
 *
 * Disconnect from a remote segment and clean up local resources.
 * This function destroys the remote segment descriptor.
 *
 * \param[in]   segment         segment descriptor handle
 *
 * \returns \c 0 on success
 */
int DisconnectRemoteSegment(r_segment_t segment);


/**
 * \brief Get the size of remote segment
 *
 * Get the size of the remote segment.
 *
 * \param[in]   segment         segment descriptor handle
 *
 * \returns size of the segment in bytes or \c 0 on error
 */
size_t GetRemoteSegmentSize(r_segment_t segment);


/**
 * \brief Get pointer to remote segment memory
 *
 * Map remote segment memory into virtual memory and return a pointer to it.
 *
 * \param[in]   segment         segment descriptor handle
 *
 * \returns a memory mapped pointer or \c NULL on error
 */
void* GetRemoteSegmentPtr(r_segment_t segment);


/**
 * \brief Get read-only pointer to remote segment memory
 *
 * Map segment memory into read-only memory and return a pointer to it.
 *
 * \param[in]   segment         segment descriptor handle
 * 
 * \returns a memory mapped pointer or \c NULL on error
 */
const void* GetRemoteSegmentPtrRO(r_segment_t segment);

// TODO: Custom functions for broadcast ConnectBroadcastSegment or something, maybe even in a broadcast.h file

#ifdef __cplusplus
}
#endif
#endif
