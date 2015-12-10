#ifndef __GPU_H__
#define __GPU_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <stdint.h>


/* Allocate GPU device buffer
 *
 * gpu_device_id        - local GPU to allocate device memory from
 * buf_size             - the size of the buffer
 *
 * Returns a CUDA device memory pointer on success or NULL on error
 */
void* gpu_malloc(int gpu_device_id, size_t buf_size);


/* Free GPU device buffer
 *
 * gpu_device_id        - local GPU where the device memory is allocated
 * buf_ptr              - pointer to the buffer that is to be free'd
 *
 * No return value
 */
void gpu_free(int gpu_device_id, void* buf_ptr);


/* Do memset on a GPU device buffer
 *
 * gpu                  - local GPU where the memory is allocated
 * buf                  - pointer to the buffer
 * len                  - size of the buffer
 * val                  - the value to set
 *
 * No return value
 */
void gpu_memset(int gpu, void* buf, size_t len, uint8_t val);


/* Do memcmp on a GPU device buffer and a RAM memory buffer
 *
 * gpu                  - local GPU where the device buffer is allocated
 * gpu_ptr              - pointer to the GPU buffer
 * ram_ptr              - pointer to the RAM buffer
 * len                  - size of the buffer
 *
 * Returns the byte position where the first difference is encountered
 */
size_t gpu_memcmp(int gpu, void* gpu_ptr, volatile void* remote_ptr, size_t len);


//void gpu_copy_remote_to_local(int gpu, void* gpu_ptr, volatile void* remote_ptr, size_t len);

//void gpu_copy_local_to_remote(int gpu, void* gpu_ptr, volatile void* remote_ptr, size_t len);

void gpu_memcpy_buffer_to_local(int gpu, void* gpu_ptr, void* local_ptr, size_t len);


/* Get the device pointer for a device buffer
 *
 * Translates a CUDA device memory pointer into a device pointer
 *
 * gpu_device_id        - local GPU where the device memory is allocated
 * buf_ptr              - pointer to the CUDA device buffer
 *
 * Returns the devicePointer associated with the CUDA memory pointer
 */
void* gpu_devptr(int gpu_device_id, void* buf_ptr);


/* Set the CU_POINTER_ATTRIBUTE_SYNC_MEMOPS flag on a device pointer
 *
 * dev_ptr              - the device pointer
 *
 * No return value
 */
void devptr_set_sync_memops(void* dev_ptr);


#ifdef __cplusplus
}
#endif
#endif
