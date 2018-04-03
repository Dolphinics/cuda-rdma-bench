#ifndef __UTIL_H__
#define __UTIL_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <sisci_api.h>
#include <stdint.h>
#include <stdlib.h>
#include "bench.h"

extern bench_mode_t all_benchmarking_modes[];


/* Get a benchmark operation mode from its name */
bench_mode_t bench_mode_from_name(const char* name);


/* Get the name of a benchmark operation mode */
const char* bench_mode_name(bench_mode_t mode);


/* Get the benchmark operation mode description */
const char* bench_mode_desc(bench_mode_t mode);


/* Get the I/O address of a local segment */
uint64_t local_ioaddr(sci_local_segment_t segment);


/* Get the physical address of a local segment */
uint64_t local_phaddr(sci_local_segment_t segment);


/* Get the I/O address of a remote segment */
uint64_t remote_ioaddr(sci_remote_segment_t segment);


/* Create a local segment and attach it to GPU memory */
sci_error_t make_gpu_segment(sci_desc_t sd, unsigned adapter_no, unsigned segment_id, sci_local_segment_t* segment, size_t segment_size, const gpu_info_t* gpu_info, void** gpu_buf);


/* Create a local segment and map it RAM memory */
sci_error_t make_ram_segment(sci_desc_t sd, unsigned adapter_no, unsigned segment_id, sci_local_segment_t* segment, size_t segment_size, sci_map_t* map, void** ram_buf);


/* Release local segment attached to GPU memory */
void free_gpu_segment(sci_local_segment_t segment, int gpu_id, void* buf_ptr);


/* Release local segment mapped to RAM memory */
void free_ram_segment(sci_local_segment_t segment, sci_map_t buf_mapping);

#ifdef __cplusplus
}
#endif
#endif
