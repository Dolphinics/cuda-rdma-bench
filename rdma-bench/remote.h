#ifndef __REMOTE_SEGMENT_H__
#define __REMOTE_SEGMENT_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>

typedef struct remote_segment* r_segment_t;


int ConnectRemoteSegment(r_segment_t* handle, unsigned adapterNo, unsigned remoteNode, unsigned segmentId);


void DisconnectRemoteSegment(r_segment_t handle);


size_t GetRemoteSegmentSize(r_segment_t handle);


void* GetRemoteSegmentPtr(r_segment_t handle);


#ifdef __cplusplus
extern "C" }
#endif
#endif
