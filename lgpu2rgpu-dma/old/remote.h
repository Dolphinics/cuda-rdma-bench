#ifndef __REMOTE_H__
#define __REMOTE_H__

#include <sisci_api.h>
#include <stdlib.h>
#include <stdint.h>

typedef struct {
    sci_remote_segment_t segment;
    size_t               length;
} remote_t;

#ifdef __cplusplus
extern "C" {
#endif

remote_t ConnectRemoteSegment(sci_desc_t sciDev, uint32_t remoteNodeId, uint32_t adapterNo);

void FreeRemoteSegment(remote_t handle);

#ifdef __cplusplus
}
#endif

#endif
