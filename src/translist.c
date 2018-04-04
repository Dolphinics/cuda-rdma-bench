#include <errno.h>
#include <stdlib.h>
#include <sisci_api.h>
#include <pthread.h>
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
    size_t                  remote_segment_size;
    int                     remote_segment_global;
    size_t                  segment_size;
    unsigned int            validate_irq_no;
    sci_remote_interrupt_t  validate_irq;
    int                     local_gpu_id;
    int                     remote_gpu_id;
    void*                   local_buf_ptr;
    gpu_info_t              local_gpu_info;
    gpu_info_t              remote_gpu_info;
    sci_map_t               buf_mapping;
    size_t                  entry_list_size;
    translist_entry_t       entry_list[MAX_TRANSLIST_SIZE];
};


static sci_error_t load_remote_info(unsigned adapter, unsigned node, unsigned id, translist_t list)
{
    sci_error_t err = SCI_ERR_OK;
    sci_error_t status = SCI_ERR_OK;
    sci_desc_t sd;
    sci_remote_segment_t segment;
    sci_map_t map;
    volatile conn_info_t* ci;

    SCIOpen(&sd, 0, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to open descriptor: %s", SCIGetErrorString(err));
        return err;
    }

    log_debug("Retrieving remote buffer info %x on node %u...", node, id);
    do
    {
        SCIConnectSegment(sd, &segment, node, id & ID_MASK, adapter, NULL, NULL, TRANSLIST_TIMEOUT_MS, 0, &err);
    }
    while (err == SCI_ERR_TIMEOUT || err == SCI_ERR_NO_SUCH_SEGMENT);

    if (err != SCI_ERR_OK)
    {
        log_warn("Failed to retrieve remote buffer info");
        status = err;
        goto close;
    }

    ci = (volatile conn_info_t*) SCIMapRemoteSegment(segment, &map, 0, sizeof(conn_info_t), NULL, SCI_FLAG_READONLY_MAP, &err);
    if (err != SCI_ERR_OK)
    {
        log_warn("Failed to retrieve remote buffer info");
        status = err;
        goto disconnect;
    }

    list->remote_node_id = ci->node_id;
    list->remote_segment_id = ci->segment_id;
    list->remote_segment_size = ci->size;
    list->remote_segment_global = ci->global;
    list->remote_gpu_id = ci->gpu;
    list->remote_gpu_info = ci->gpu_info;
    list->validate_irq_no = ci->intr_no;

    if (list->remote_gpu_id != NO_GPU)
    {
        log_info("Remote buffer is GPU memory");
    }
    else
    {
        log_info("Remote buffer is RAM memory");
    }

    SCIUnmapSegment(map, 0, &err);

disconnect:
    do
    {
        SCIDisconnectSegment(segment, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

close:
    SCIClose(sd, 0, &err);

    return status;
}


static sci_error_t connect_remote_segment(translist_t list)
{
    sci_error_t err = SCI_ERR_OK;

    unsigned remote_segment = list->remote_segment_id;
    unsigned remote_node = list->remote_node_id;
    unsigned adapter = list->local_adapter_no;

    log_debug("Trying to connect to remote segment %x on node %u", remote_segment, remote_node);
    do
    {
        SCIConnectSegment(list->sd, &list->remote_segment, remote_node, remote_segment, adapter, NULL, NULL, TRANSLIST_TIMEOUT_MS, 0, &err);
    }
    while (err == SCI_ERR_TIMEOUT || err == SCI_ERR_NO_SUCH_SEGMENT);

    if (err != SCI_ERR_OK)
    {
        log_error("Failed to connect to remote segment: %s", SCIGetErrorString(err));
        return err;
    }
    log_info("Connected to remote segment %x on node %u", remote_segment, remote_node);
    log_debug("Remote segment %x on node %u has size %zu", 
            remote_segment, remote_node, SCIGetRemoteSegmentSize(list->remote_segment));

    return SCI_ERR_OK;
}


static sci_error_t connect_remote_irq(translist_t list)
{
    sci_error_t err = SCI_ERR_OK;

    unsigned remote_node = list->remote_node_id;
    unsigned remote_irq_no = list->validate_irq_no;
    unsigned adapter = list->local_adapter_no;

    log_debug("Trying to connect to remote interrupt %u on node %u", remote_irq_no, remote_node);
    do
    {
        SCIConnectInterrupt(list->sd, &list->validate_irq, remote_node, adapter, remote_irq_no, TRANSLIST_TIMEOUT_MS, 0, &err);
    }
    while (err == SCI_ERR_TIMEOUT || /*err == SCI_ERR_NO_SUCH_INTNO ||*/ err == SCI_ERR_NO_SUCH_SEGMENT); // FIXME: This is a bug in the SISCI API

    if (err != SCI_ERR_OK)
    {
        log_error("Failed to connect to remote interrupt: %s", SCIGetErrorString(err));
        return err;
    }

    return SCI_ERR_OK;
}





int translist_create(translist_t* handle, unsigned adapter, unsigned local_segment, size_t lsize, unsigned remote_node, unsigned remote_segment, int gpu, int gl)
{
    sci_error_t err = SCI_ERR_OK;
    translist_t list;
    
    // Allocate memory for a list
    list = (translist_t) malloc(sizeof(struct transfer_list));
    
    if (list == NULL)
    {
        log_error("Insufficient resources to allocate transfer list");
        return -ENOMEM;
    }

    list->local_gpu_id = gpu;
    list->remote_gpu_id = NO_GPU;
    if (gpu != NO_GPU && gpu_info(gpu, &list->local_gpu_info) != 1)
    {
        log_error("Failed to get GPU info");
        goto error_free;
    }

    err = load_remote_info(adapter, remote_node, remote_segment, list);
    if (err != SCI_ERR_OK)
    {
        goto error_free;
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
        log_error("Unexpected error when retrieving local node ID");
        goto error_close;
    }
    log_debug("Local node ID: %u", list->local_node_id);

    // Try to connect to remote segment
    err = connect_remote_irq(list);
    if (err != SCI_ERR_OK)
    {
        goto error_close;
    }
    
    err = connect_remote_segment(list);
    if (err != SCI_ERR_OK)
    {
        goto error_disconnect_irq;
    }

    // Figure out transfer list size
    if (lsize == 0 || list->remote_segment_size < lsize)
    {
        log_warn("Transfer segment size exceeds remote segment size, using remote segment size");
        list->segment_size = list->remote_segment_size;
    }
    else
    {
        list->segment_size = lsize;
    }

    // Create local buffer and segment
    if (gpu != NO_GPU)
    {
        err = make_gpu_segment(list->sd, adapter, local_segment, &list->local_segment, list->segment_size, &list->local_gpu_info, &list->local_buf_ptr, gl);
    }
    else
    {
        err = make_ram_segment(list->sd, adapter, local_segment, &list->local_segment, list->segment_size, &list->buf_mapping, &list->local_buf_ptr, gl);
    }

    if (err != SCI_ERR_OK)
    {
        log_error("Unexpected error when creating local segment");
        goto error_disconnect_segment;
    }

    // Initialise transfer entry list
    list->entry_list_size = 0;
    *handle = list;

    return 0;

error_disconnect_segment:
    SCIDisconnectSegment(list->remote_segment, 0, &err);
error_disconnect_irq:
    SCIDisconnectInterrupt(list->validate_irq, 0, &err);
error_close:
    SCIClose(list->sd, 0, &err);
error_free:
    free(list);

    return -EBADFD;
}


void translist_delete(translist_t handle)
{
    sci_error_t err = SCI_ERR_OK;

    if (handle->local_gpu_id != NO_GPU)
    {
        free_gpu_segment(handle->local_segment, handle->local_gpu_id, handle->local_buf_ptr);
    }
    else
    {
        free_ram_segment(handle->local_segment, handle->buf_mapping);
    }

    SCIDisconnectInterrupt(handle->validate_irq, 0, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to disconnect remote interrupt: %s", SCIGetErrorString(err));
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
    info.validate = handle->validate_irq;
    info.global = handle->remote_segment_global;
    info.buffer_ptr = handle->local_buf_ptr;
    info.local_gpu_info = info.remote_gpu_info = NULL;

    if (handle->local_gpu_id != NO_GPU)
    {
        info.local_gpu_info = &handle->local_gpu_info;
    }

    if (handle->remote_gpu_id != NO_GPU)
    {
        info.remote_gpu_info = &handle->remote_gpu_info;
    }

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

    if (size > handle->segment_size)
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
