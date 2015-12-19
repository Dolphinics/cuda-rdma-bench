#ifndef __TRANSFER_LIST_H__
#define __TRANSFER_LIST_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <sisci_api.h>
#include <stdlib.h>
#include "gpu.h"

#ifndef MAX_TRANSLIST_SIZE
#define MAX_TRANSLIST_SIZE DIS_DMA_MAX_VECLEN
#endif

#ifndef TRANSLIST_TIMEOUT_MS
#define TRANSLIST_TIMEOUT_MS 5000
#endif


/* Define transfer list handle */
typedef struct transfer_list* translist_t;


/* Define transfer list entry */
typedef struct {
    size_t  offset_local;   // offset in local segment
    size_t  offset_remote;  // offset in remote segment
    size_t  size;           // total number of bytes to transfer
} translist_entry_t;


/* Define transfer list descriptor */
typedef struct {
    sci_desc_t                  sisci_desc;         // SISCI API descriptor
    sci_local_segment_t         segment_local;      // local segment handle
    sci_remote_segment_t        segment_remote;     // remote segment handle
    size_t                      segment_size;       // size of the segment
    sci_remote_data_interrupt_t interrupt;          // data interrupt
    sci_remote_interrupt_t      validate;           // remote validation interrupt
    void*                       buffer_ptr;         // pointer to the local buffer
    const gpu_info_t*           local_gpu_info;     // local GPU description
    const gpu_info_t*           remote_gpu_info;    // remote GPU description
} translist_desc_t;


/* Create transfer list handle
 *
 * handle           - handle reference
 * local_adapter    - local Dolphin NTB adapter number
 * local_segment    - local SISCI segment ID
 * remote_node      - remote Dolphin cluster node
 * remote_segment   - remote SISCI segment ID
 * gpu_id           - local CUDA device ID (or NO_GPU)
 *
 * Returns 0 on success or a negative value on error
 */
int translist_create(translist_t* handle, unsigned local_adapter, unsigned local_segment, unsigned remote_node, unsigned remote_segment, int gpu_id);


/* Get transfer list descriptor
 *
 * handle           - transfer list handle
 *
 * Returns a transfer list descriptor
 */
translist_desc_t translist_desc(translist_t handle);


/* Get transfer list size
 *
 * handle           - transfer list handle
 *
 * Returns the number of transfers in the list
 */
size_t translist_size(translist_t handle);


/* Insert transfer instruction into transfer list
 *
 * handle           - transfer list handle
 * local_offset     - offset in local SISCI segment
 * remote_offset    - offset in remote SISCI segment
 * size             - number of bytes to transfer
 *
 * Returns 0 on success and a negative value on error
 */
int translist_insert(translist_t handle, size_t local_offset, size_t remote_offset, size_t size);


/* Get specific transfer list entry
 *
 * handle           - transfer list handle
 * entry_idx        - the list index of the desired entry
 * entry_ptr        - pointer to buffer where the entry should be copied to
 *
 * Returns 0 on success and a negative value on error
 */
int translist_element(translist_t handle, size_t entry_idx, translist_entry_t* entry_ptr);


/* Clean up and free resources allocated by a transfer list
 *
 * handle           - transfer list handle to clean up after
 *
 * No return value
 */
void translist_delete(translist_t handle);

#ifdef __cplusplus
}
#endif
#endif
