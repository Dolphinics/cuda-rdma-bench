#include <stdint.h>
#include <stdlib.h>
#include <sisci_api.h>
#include <pthread.h>
#include <string.h>
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
static pthread_cond_t signal = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;


void stop_server()
{
    log_info("Stopping server...");

    pthread_mutex_lock(&lock);
    keep_running = 0;
    pthread_cond_signal(&signal);
    pthread_mutex_unlock(&lock);
}


static sci_callback_action_t validate_buffer(void* buf_info, sci_local_interrupt_t UNUSED(irq), sci_error_t status)
{
    if (status == SCI_ERR_OK)
    {
        buf_info_t* bi = (buf_info_t*) buf_info;
        uint8_t byte;

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


static void run_server(unsigned adapter, const gpu_info_t* gpu, unsigned id, size_t size, int global)
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

    // Create local memory buffer and local segment
    sci_local_segment_t segment;
    sci_map_t mapping;
    void* buffer;

    uint8_t byte = random_byte_value();
    log_debug("Creating buffer and filling with random value %02x", byte);
    if (gpu != NULL)
    {
        err = make_gpu_segment(sd, adapter, id & ID_MASK, &segment, size, gpu, &buffer, global);
        gpu_memset(gpu->id, buffer, size, byte);
    }
    else
    {
        err = make_ram_segment(sd, adapter, id & ID_MASK, &segment, size, &mapping, &buffer, global);
        ram_memset(buffer, size, byte);
    }

    if (err != SCI_ERR_OK)
    {
        log_error("Failed to create segment");
        goto close_desc;
    }

    // Create interrupt to trigger validation of the buffer
    sci_local_interrupt_t validate_irq;
    unsigned validate_irq_no = id;
    buf_info_t info = { .gpu = gpu, .ptr = buffer, .len = size, .val = byte };

    SCICreateInterrupt(sd, &validate_irq, adapter, &validate_irq_no, &validate_buffer, &info, SCI_FLAG_FIXED_INTNO | SCI_FLAG_USE_CALLBACK, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to create interrupt");
        goto free_segment;
    }

    // Set local segment available
    SCISetSegmentAvailable(segment, adapter, 0, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to set segment available");
        goto remove_irq;
    }

    // Run until we're killed
    log_info("Running server...");
    pthread_mutex_lock(&lock);
    while (keep_running)
    {
        pthread_cond_wait(&signal, &lock);
    }
    pthread_mutex_unlock(&lock);
    log_info("Server stopped");

    // Do clean up
    SCISetSegmentUnavailable(segment, adapter, SCI_FLAG_NOTIFY | SCI_FLAG_FORCE_DISCONNECT, &err);

remove_irq:
    do
    {
        SCIRemoveInterrupt(validate_irq, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

free_segment:
    if (gpu != NULL)
    {
        free_gpu_segment(segment, gpu->id, buffer);
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
        log_error("Failed to initialize SISCI descriptor");
        return;
    }

    // Create GPU info segment
    sci_local_segment_t gi_segment;
    sci_map_t gi_mapping;
    gpu_info_t* local_gpu;

    unsigned bufinfo_id = (('G' ^ 'P' ^ 'U') << ID_MASK_BITS) | id;
    err = make_ram_segment(sd, adapter, bufinfo_id, &gi_segment, sizeof(gpu_info_t), &gi_mapping, (void**) &local_gpu, 0);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to create buffer info segment");
        SCIClose(sd, 0, &err);
        return;
    }
    memset(local_gpu, sizeof(gpu_info_t), 0xff);
    local_gpu->id = NO_GPU;

    // Get local GPU information
    if (gpu != NO_GPU && gpu_info(gpu, local_gpu) != 1)
    {
        log_error("Failed to get GPU info, aborting...");
        free_ram_segment(gi_segment, gi_mapping);
        SCIClose(sd, 0, &err);
        return;
    }

    SCISetSegmentAvailable(gi_segment, adapter, 0, &err);

    // Run server
    run_server(adapter, gpu != NO_GPU ? local_gpu : NULL, id, size, global);

    SCISetSegmentUnavailable(gi_segment, adapter, 0, &err);

    free_ram_segment(gi_segment, gi_mapping);
    SCIClose(sd, 0, &err);
}
