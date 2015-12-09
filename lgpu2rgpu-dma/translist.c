#include <errno.h>
#include <stdlib.h>
#include <sisci_api.h>
#include "translist.h"
#include "reporting.h"
#include "util.h"
#include "common.h"
#include "gpu.h"


/* Transfer list handle declaration */
struct transfer_list
{
    sci_desc_t              sd;
    unsigned int            local_adapter_no;
    unsigned int            local_node_id;
    unsigned int            local_segment_id;
    sci_local_segment_t     local_segment;
    unsigned int            remote_node_id;
    unsigned int            remote_segment_id;
    sci_remote_segment_t    remote_segment;
    size_t                  segment_size;
    int                     gpu_device_id;
    void*                   local_buf_ptr;
    sci_map_t               buf_mapping;
    size_t                  entry_list_size;
    translist_entry_t       entry_list[0];
};



int translist_create(translist_t* handle, unsigned adapter, unsigned local_segment, unsigned remote_node, unsigned remote_segment, int gpu)
{
    sci_error_t err = SCI_ERR_OK;
    translist_t list;

    // Allocate memory for a list
    list = (translist_t) malloc(sizeof(struct transfer_list) + sizeof(translist_entry_t) * MAX_TRANSLIST_SIZE);
    
    if (list == NULL)
    {
        log_error("Insufficient resources to allocate transfer list");
        return -ENOMEM;
    }

    // Open SISCI API descriptor and extract local cluster node ID
    SCIOpen(&list->sd, 0, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to open descriptor: %s", SCIGetErrorString(err));
        goto error_free;
    }

    list->local_adapter_no = adapter;
    list->local_segment_id = local_segment;
    
    SCIGetLocalNodeId(adapter, &list->local_node_id, 0, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Unexpected error when retrieving local cluster node ID");
        goto error_close;
    }
    log_debug("Local cluster node ID: %u", list->local_node_id);

    // Try to connect to remote segment
    list->remote_node_id = remote_node;
    list->remote_segment_id = remote_segment;

    log_debug("Trying to connect to remote segment %u on cluster node %u", remote_segment, remote_node);
    do
    {
        SCIConnectSegment(list->sd, &list->remote_segment, remote_node, remote_node, adapter, NULL, NULL, 50, 0, &err);
    }
    while (err == SCI_ERR_TIMEOUT || err == SCI_ERR_NO_SUCH_SEGMENT);
    
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to connect to remote segment: %s", SCIGetErrorString(err));
        goto error_close;
    }
    log_info("Connected to remote segment %u on cluster node %u", remote_segment, remote_node);

    // Create local buffer and segment
    list->segment_size = SCIGetRemoteSegmentSize(list->remote_segment);

    list->gpu_device_id = gpu;
    if (gpu != NO_GPU)
    {
        err = make_gpu_segment(list->sd, adapter, local_segment, &list->local_segment, list->segment_size, gpu, &list->local_buf_ptr);
    }
    else
    {
        err = make_ram_segment(list->sd, adapter, local_segment, &list->local_segment, list->segment_size, &list->buf_mapping, &list->local_buf_ptr);
    }

    if (err != SCI_ERR_OK)
    {
        log_error("Unexpected error when creating local segment");
        goto error_disconnect;
    }

    // Initialise transfer entry list
    list->entry_list_size = 0;
    *handle = list;

    return 0;

error_disconnect:
    SCIDisconnectSegment(list->remote_segment, 0, &err);
error_close:
    SCIClose(list->sd, 0, &err);
error_free:
    free(list);

    return -EBADFD;
}



void translist_delete(translist_t handle)
{
    sci_error_t err = SCI_ERR_OK;

    if (handle->entry_list_size > 0)
    {
        log_warn("Transfer list is not empty");
    }

    if (handle->gpu_device_id != NO_GPU)
    {
        free_gpu_segment(handle->local_segment, handle->gpu_device_id, handle->local_buf_ptr);
    }
    else
    {
        free_ram_segment(handle->local_segment, handle->buf_mapping);
    }

    do
    {
        SCIDisconnectSegment(handle->remote_segment, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

    if (err != SCI_ERR_OK)
    {
        log_error("Failed to disconnect remote segment: %s", SCIGetErrorString(err));
    }

    SCIClose(handle->sd, 0, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to close descriptor: %s", SCIGetErrorString(err));
    }

    free(handle);
}



translist_desc_t translist_desc(translist_t handle)
{
    translist_desc_t info;

    info.sisci_desc = handle->sd;
    info.segment_local = handle->local_segment;
    info.segment_remote = handle->remote_segment;
    info.segment_size = handle->segment_size;
    info.gpu_device_id = handle->gpu_device_id;
    info.buffer_ptr = handle->local_buf_ptr;

    return info;
}



size_t translist_size(translist_t handle)
{
    return handle->entry_list_size;
}



int translist_element(translist_t handle, size_t idx, translist_entry_t* ptr)
{
    if (idx >= handle->entry_list_size)
    {
        log_error("No such element in transfer list");
        return -EINVAL;
    }

    *ptr = handle->entry_list[idx];
    return 0;
}



int translist_insert(translist_t handle, size_t local_offs, size_t remote_offs, size_t size)
{
    if (handle->entry_list_size >= MAX_TRANSLIST_SIZE)
    {
        log_error("Transfer list is already at maximum size");
        return -ENOMEM;
    }

    if (size > MAX_SIZE)
    {
        log_error("Transfer size is too large");
        return -EINVAL;
    }
    else if (size > handle->segment_size)
    {
        log_error("Transfer size is larger than segment size");
        return -EINVAL;
    }
    else if (local_offs > handle->segment_size || local_offs + size > handle->segment_size)
    {
        log_error("Local offset is too large");
        return -EINVAL;
    }
    else if (remote_offs > handle->segment_size || remote_offs + size > handle->segment_size)
    {
        log_error("Remote offset is too large");
        return -EINVAL;
    }

    size_t idx = handle->entry_list_size++;

    handle->entry_list[idx].offset_local = local_offs;
    handle->entry_list[idx].offset_remote = remote_offs;
    handle->entry_list[idx].size = size;
    
    return 0;
}
