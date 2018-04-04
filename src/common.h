#ifndef __COMMON_H__
#define __COMMON_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <limits.h>
#include "gpu.h"

typedef struct {
    unsigned    intr_no;
    int         global;
    size_t      size;
    unsigned    node_id;
    unsigned    segment_id;
    int         gpu;
    gpu_info_t  gpu_info;
} conn_info_t;


/* Some handy defines */
#define NO_NODE         UINT_MAX        // Indicate that no Dolphin cluster node is set
#define NO_GPU          -1              // Indicate that no CUDA device is selected
#define NO_ID           NO_NODE         // Indicate that no SISCI segment ID is set
#define ID_MASK_BITS    16              // Number of ID bits
#define ID_MASK         ((1ULL << ID_MASK_BITS)-1)  // Extract ID bits

#ifdef __cplusplus
}
#endif
#endif
