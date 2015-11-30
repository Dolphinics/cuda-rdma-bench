#ifndef __LOCAL_H__
#define __LOCAL_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <sisci_api.h>

typedef struct {
    sci_local_segment_t segment;
    void*               buffer;
    size_t              size;
    int                 hostmem;
    int                 gpu_id;
} bufhandle_t;

bufhandle_t create_gpu_buffer(sci_desc_t desc, unsigned adapter_id, int gpu_id, unsigned segment_id, size_t mem_size, unsigned mem_flags);

void free_gpu_buffer(bufhandle_t buffer_handle);

uint8_t validate_buffer(bufhandle_t buffer_handle);

#ifdef __cplusplus
}
#endif
#endif
