#include <stdint.h>
#include <stdlib.h>
#include <sisci_api.h>
#include <pthread.h>
#include <string.h>
#include <signal.h>
#include "reporting.h"
#include "common.h"
#include "util.h"
#include "bench.h"
#include "gpu.h"
#include "ram.h"

#ifdef __GNUC__
#define UNUSED(x) x __attribute__((unused))
#else
#define UNUSED(x) x
#endif


/* Buffer info */
typedef struct {
    const gpu_info_t*   gpu;
    void*               ptr;
    size_t              len;
    uint8_t             val;
} buf_info_t;


/* Should we keep running? */
static volatile int keep_running = 1;
static pthread_cond_t queue = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;


void stop_server()
{
    log_info("Stopping server...");

    pthread_mutex_lock(&lock);
    keep_running = 0;
    pthread_cond_signal(&queue);
    pthread_mutex_unlock(&lock);
}


static sci_callback_action_t validate_buffer(void* buf_info, sci_local_interrupt_t UNUSED(irq), sci_error_t status)
{
    if (status == SCI_ERR_OK)
    {
        buf_info_t* bi = (buf_info_t*) buf_info;
        uint8_t byte;

        if (bi->ptr == NULL && bi->len == 0)
        {
            log_error("Interrupt callback called before segment was initialized");
            return SCI_CALLBACK_CANCEL;
        }

        if (bi->gpu != NULL)
        {
            gpu_memcpy_buffer_to_local(bi->gpu->id, bi->ptr, &byte, 1);
        }
        else
        {
            byte = *((uint8_t*) bi->ptr);
        }

        report_buffer_change(stdout, bi->val, byte);

        bi->val = byte;
    }

    return SCI_CALLBACK_CONTINUE;
}


static void run_server(unsigned adapter, sci_local_segment_t ci_segment, const conn_info_t* ci, buf_info_t* bi)
{
    sci_error_t err;
    sci_desc_t sd;

    // Create SISCI descriptor
    SCIOpen(&sd, 0, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to open SISCI descriptor");
        return;
    }

    // Create transfer segment
    sci_local_segment_t segment;
    sci_map_t mapping;
    void* buffer;

    uint8_t byte = random_byte_value();
    log_debug("Creating buffer and filling with random value %02x", byte);
    if (ci->gpu != NO_GPU)
    {
        err = make_gpu_segment(sd, adapter, ci->segment_id, &segment, ci->size, &ci->gpu_info, &buffer, ci->global);
        gpu_memset(ci->gpu, buffer, ci->size, byte);
    }
    else
    {
        err = make_ram_segment(sd, adapter, ci->segment_id, &segment, ci->size, &mapping, &buffer, ci->global);
        ram_memset(buffer, ci->size, byte);
    }

    if (err != SCI_ERR_OK)
    {
        log_error("Failed to create segment");
        goto close_desc;
    }

    // Set transfer segment available
    SCISetSegmentAvailable(segment, adapter, 0, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to set transfer segment available: %s", SCIGetErrorString(err));
        goto free_segment;
    }

    bi->gpu = ci->gpu != NO_GPU ? &ci->gpu_info : NULL;
    bi->ptr = buffer;
    bi->len = ci->size;
    bi->val = byte;

    // Set connection info segment available
    SCISetSegmentAvailable(ci_segment, adapter, 0, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to set buffer info segment available: %s", SCIGetErrorString(err));
        goto free_segment;
    }

    signal(SIGINT, (sig_t) &stop_server);
    signal(SIGTERM, (sig_t) &stop_server);
    signal(SIGPIPE, (sig_t) &stop_server);

    // Run until we're killed
    log_info("Running server...");
    pthread_mutex_lock(&lock);
    while (keep_running)
    {
        pthread_cond_wait(&queue, &lock);
    }
    pthread_mutex_unlock(&lock);
    log_info("Server stopped");

    // Do clean up
    SCISetSegmentUnavailable(ci_segment, adapter, 0, &err);
    SCISetSegmentUnavailable(segment, adapter, 0, &err);

free_segment:
    if (ci->gpu != NO_GPU)
    {
        free_gpu_segment(segment, ci->gpu, buffer);
    }
    else
    {
        free_ram_segment(segment, mapping);
    }

close_desc:
    SCIClose(sd, 0, &err);
}


void server(unsigned adapter, int gpu, unsigned id, size_t size, int global)
{
    sci_error_t err = SCI_ERR_OK;
    sci_desc_t sd;

    SCIOpen(&sd, 0, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to initialize SISCI descriptor: %s", SCIGetErrorString(err));
        return;
    }

    unsigned local_node = 0;
    SCIGetLocalNodeId(adapter, &local_node, 0, &err);
    if (err != SCI_ERR_OK)
    {
        SCIClose(sd, 0, &err);
        return;
    }

    // Create connection info segment
    sci_local_segment_t gi_segment;
    sci_map_t gi_mapping;
    conn_info_t* conn_info;

    err = make_ram_segment(sd, adapter, id & ID_MASK, &gi_segment, sizeof(conn_info_t), &gi_mapping, (void**) &conn_info, 0);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to create buffer info segment");
        SCIClose(sd, 0, &err);
        return;
    }

    conn_info->intr_no = 0;
    conn_info->global = global;
    conn_info->size = size;
    conn_info->node_id = local_node;
    conn_info->segment_id = ((unsigned long long) id) << ID_MASK_BITS;
    conn_info->gpu = gpu;

    memset(&conn_info->gpu_info, 0xff, sizeof(gpu_info_t));
    conn_info->gpu_info.id = NO_GPU;

    // Get local GPU information
    if (gpu != NO_GPU && gpu_info(gpu, &conn_info->gpu_info) != 1)
    {
        log_error("Failed to get GPU info, aborting...");
        goto leave;
    }

    // Create interrupt to trigger validation of the buffer
    sci_local_interrupt_t validate_irq;
    buf_info_t buf_info = { .gpu = NULL, .ptr = NULL, .len = 0, .val = 0 };

    SCICreateInterrupt(sd, &validate_irq, adapter, &conn_info->intr_no, &validate_buffer, &buf_info, SCI_FLAG_USE_CALLBACK, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to create interrupt: %s", SCIGetErrorString(err));
        goto leave;
    }
    log_debug("Validation IRQ %u", conn_info->intr_no);

    // Run server
    run_server(adapter, gi_segment, conn_info, &buf_info);

    do
    {
        SCIRemoveInterrupt(validate_irq, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

leave:
    free_ram_segment(gi_segment, gi_mapping);
    SCIClose(sd, 0, &err);
}
