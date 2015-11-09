#ifndef __GPUDIRECT_COMMON_H__
#define __GPUDIRECT_COMMON_H__

#include <sisci_api.h>
#include <stdint.h>

#define NO_NODE_ID 0xffffffff

#define SEGMENT_ID(localNodeId, segmentNumber, offset) \
    (((localNodeId) << 16) | ((segmentNumber) << 8) | (offset))

#define SCIGetErrorString(error) sciGetErrorString((error))

const char* sciGetErrorString(sci_error_t error);

uint64_t getPhysAddr(sci_local_segment_t segment);

uint64_t getLocalIOAddr(sci_local_segment_t segment);

uint64_t getRemoteIOAddr(sci_remote_segment_t segment);

void dumpMem(const uint8_t* addr, size_t size);

void fillMem(uint8_t* addr, size_t size);

void zeroMem(uint8_t* addr, size_t size);

#endif
