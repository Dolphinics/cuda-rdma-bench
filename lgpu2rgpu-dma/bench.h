#ifndef __BENCH_H__
#define __BENCH_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdlib.h>
#include <sisci_api.h>
#include "translist.h"


/* Benchmark different functions */
typedef enum {
    BENCH_DO_NOTHING = 0,                   // Dummy benchmark node
    BENCH_SCI_DMA_PUSH_TO_REMOTE,           // Use IX/PX DMA engine to push data to remote host
    BENCH_SCI_DMA_GLOBAL_PUSH_TO_REMOTE,    // Use IX/PX DMA engine to push data to remote host
    BENCH_SCI_DMA_PULL_FROM_REMOTE,         // Use IX/PX DMA engine to pull data from remote host
    BENCH_SCI_DMA_GLOBAL_PULL_FROM_REMOTE,  // Use IX/PX DMA engine to pull data from remote host
    BENCH_SCI_WRITE_TO_REMOTE,              // Use SISCI SCIMemWrite to write data to remote host
    BENCH_SCI_MEMCPY_TO_REMOTE,             // Use SISCI SCIMemCpy to write data to remote host
    BENCH_SCI_MEMCPY_FROM_REMOTE,           // Use SISCI SCIMemCpy to read data from remote host
    BENCH_WRITE_TO_REMOTE,                  // Use regular memcpy to write data to remote host
    BENCH_READ_FROM_REMOTE,                 // Use regular memcpy to read data from remote host
    BENCH_SCI_DATA_INTERRUPT                // Use IX/PX data interrupts to send data to a remote host
} bench_mode_t;


/* Get the current timestamp in microseconds (µs) */
uint64_t ts_usecs();


/* Run benchmarking server 
 * 
 * This will block until stop_server() is invoked asynchronously (i.e. from an signal handler)
 */
void server(unsigned adapter_no, int gpu_id, unsigned segment_id, size_t segment_size);


/* Stop the benchmarking server */
void stop_server();


/* Run benchmark */
void client(bench_mode_t benchmark_mode, translist_t transfer_list, int repeat, int iec_units);

#ifdef __cplusplus
}
#endif
#endif
