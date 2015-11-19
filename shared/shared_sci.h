#ifndef __SHARED_SCI_H__
#define __SHARED_SCI_H__

#include <sisci_api.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define sciAssert(status)                           \
    do                                              \
    {                                               \
        if (SCI_ERR_OK != (status))                 \
        {                                           \
            fprintf(stderr,                         \
                    "%s = %d \"%s\" (%s:%d)\n",     \
                    #status, (status),              \
                    sciGetErrorString((status)),    \
                    __FILE__, __LINE__);            \
            exit((status));                         \
        }                                           \
    }                                               \
    while (0)

const char* sciGetErrorString(sci_error_t error);

uint64_t GetPhysAddr(sci_local_segment_t segment);

uint64_t GetLocalIOAddr(sci_local_segment_t segment);

uint64_t GetRemoteIOAddr(sci_remote_segment_t segment);

#endif
