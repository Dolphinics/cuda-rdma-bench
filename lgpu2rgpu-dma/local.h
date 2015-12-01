#ifndef __LOCAL_H__
#define __LOCAL_H__
#ifdef __cplusplus
extern "C" {
#endif


#include <stdlib.h>
#include <stdint.h>
#include <sisci_api.h>



/* Allocate GPU memory */
void* make_gpu_buffer(int gpu_id, size_t buf_len);



/* Set all bytes in a GPU buffer */
void gpu_memset(int gpu_id, void* buf_ptr, size_t buf_len, uint8_t val);



/* Release a GPU buffer */
void free_gpu_buffer(int gpu_id, void* buf_ptr);



/* Make a local SISCI segment and attach the GPU buffer */
sci_local_segment_t make_local_segment(sci_desc_t desc, unsigned adapt_no, unsigned seg_id, void* buf_ptr, size_t buf_len);



/* Check that all bytes in the buffer contain the correct value
 * Returns buf_len on success and a value less than buf_len indicating 
 * wrong byte
 */
size_t validate_gpu_buffer(int gpu_id, void* buf_ptr, size_t buf_len, uint8_t expected_val);


#ifdef __cplusplus
}
#endif
#endif
