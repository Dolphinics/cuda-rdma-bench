#ifndef __LOCAL_H__
#define __LOCAL_H__

#include <sisci_api.h>
#include <stdlib.h>
#include <stdint.h>

typedef struct {
    sci_local_segment_t segment;
    void*               buffer;
    size_t              length;
    int                 device;
} local_t;

#ifdef __cplusplus
extern "C"
#endif
local_t CreateLocalSegment(sci_desc_t sciDesc, uint32_t localNodeId, uint32_t adapterNo, int gpuId, size_t memSize);

#ifdef __cplusplus
extern "C"
#endif
void FreeLocalSegment(local_t handle, uint32_t adapter);

#ifdef __cplusplus
extern "C"
#endif
void DumpGPUMemory(local_t handle);

#endif
