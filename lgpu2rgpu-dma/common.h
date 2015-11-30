#ifndef __COMMON_H__
#define __COMMON_H__ 
#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <sisci_api.h>
#include <limits.h>
#include <stdint.h>


/* Some useful defines */
#define MAX_ID          (INT_MAX >> 1)
#define NO_ID           MAX_ID
#define MAX_NODE_ID     1024
#define NO_NODE_ID      MAX_ID
#define DEFAULT_REPEAT  5


/* Different DMA modes */
typedef enum {
    DMA_TRANSFER_ONE_WAY,   // Client pushes data
    DMA_TRANSFER_BOTH_WAYS, // Client pushes data and pulls data simultaneously
    DMA_TRANSFER_TWO_WAY,   // Client and server both push data
} dma_mode_t;
#define DMA_TRANSFER_DEFAULT DMA_TRANSFER_ONE_WAY

/* Arguments to the server */
typedef struct {
    sci_desc_t  desc;
    unsigned    adapter_id;
    unsigned    segment_id;
    size_t      segment_size;
    int         gpu_dev_id;
    unsigned    gpu_mem_flags;
    int         dma_mode;
    unsigned    dma_flags;
    const int*  keep_running;
} server_args;

/* Arguments to the client */
typedef struct {
    sci_desc_t  desc;
    unsigned    adapter_id;
    unsigned    remote_node_id;
    unsigned    remote_segment_id;
    unsigned    local_segment_id;
    int         gpu_dev_id;
    unsigned    gpu_mem_flags;
    dma_mode_t  dma_mode;
    unsigned    dma_flags;
} client_args;


void run_server(server_args* arguments);

void run_client(client_args* arguments, size_t factor, unsigned repeat);

uint64_t current_usecs();

#ifdef __cplusplus
}
#endif
#endif
