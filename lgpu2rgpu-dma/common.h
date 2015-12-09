#ifndef __COMMON_H__
#define __COMMON_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <limits.h>

/* Some handy defines */
#define NO_NODE     UINT_MAX        // Indicate that no Dolphin cluster node is set
#define NO_GPU      -1              // Indicate that no CUDA device is selected
#define NO_ID       NO_NODE         // Indicate that no SISCI segment ID is set

#ifndef MAX_SIZE
#define MAX_SIZE    (128 << 20)     // 128 MiB is the maximum size for DMA transfers
#endif

#ifdef __cplusplus
}
#endif
#endif
