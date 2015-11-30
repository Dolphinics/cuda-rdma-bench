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



/* Different DMA transfer modes */
typedef enum {
    DMA_TRANSFER_ONE_WAY,   // Client only transfers data
    DMA_TRANSFER_TWO_WAY,   // Client and server both transfer data data
} dma_mode_t;



/* Run benchmarking server 
 * Doesn't return unless stop_benchmark_server() is called
 */
void server(sci_desc_t sd, unsigned adapter_no, int gpu_id, unsigned segment_id, size_t segment_size);



/* Stop the server */
void stop_server();



/* Do the benchmark */
uint64_t benchmark(
        sci_desc_t sd, unsigned adapter_no, 
        sci_local_segment_t local, sci_remote_segment_t remote, size_t size, 
        dma_mode_t mode, unsigned flags, int repeat
        );



/* Get current timestamp in microseconds */
uint64_t current_usecs();


#ifdef __cplusplus
}
#endif
#endif
