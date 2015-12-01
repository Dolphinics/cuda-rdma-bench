#include <stdlib.h>
#include <sisci_api.h>
#include <stdint.h>
#include "common.h"
#include "local.h"
#include "reporting.h"



/* Buffer information */
struct bufinfo
{
    int     gpu;
    void*   ptr;
    size_t  len;
};



/* Indicate whether or not the server should keep running */
static int keep_running = 1;



/* Stop the server */
void stop_server()
{
    keep_running = 0;
}



/* Interrupt handler 
 * Trigger a buffer validation
 */
sci_callback_action_t trigger_validate_buffer(void* buf_info, sci_local_interrupt_t irq, sci_error_t status)
{
    // TODO: Implement a mechanism to get the expected byte using data interrupts instead
    struct bufinfo* info = (struct bufinfo*) buf_info;
    log_debug("Validating GPU buffer after DMA transfer...");
    size_t last_correct_byte = validate_gpu_buffer(info->gpu, info->ptr, info->len, 0);
    if (last_correct_byte != info->len)
    {
        log_error("Buffer is garbled, last correct byte is %lu but buffer size is %lu", last_correct_byte, info->len);
    }
    else
    {
        log_info("Buffer is valid after DMA transfer");
    }
    return SCI_CALLBACK_CONTINUE;
}



void server(sci_desc_t sd, unsigned adapter, int gpu, unsigned id, size_t size)
{
    sci_error_t err;

    // Make GPU buffer and fill all bytes with a random value
    void* buf = make_gpu_buffer(gpu, size);
    uint8_t val = rand() & 255;
    gpu_memset(gpu, buf, size, val);

    // Make local segment
    sci_local_segment_t segment = make_local_segment(sd, adapter, id, buf, size);

    // Trigger buffer validation on interrupt
    sci_local_interrupt_t validate_irq;
    unsigned validate_irq_no = id;

    struct bufinfo info = {
        .gpu = gpu, .ptr = buf, .len = size
    };

    SCICreateInterrupt(sd, &validate_irq, adapter, &validate_irq_no, &trigger_validate_buffer, &info, SCI_FLAG_FIXED_INTNO | SCI_FLAG_USE_CALLBACK, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to create buffer validation trigger: %s", SCIGetErrorString(err));
        exit(1);
    }

    // Create data interrupt for message passing
    sci_local_data_interrupt_t msg_irq;
    unsigned msg_irq_no = id;

    // TODO: Create data interrupt

    // Set segment available so we're good to go
    SCISetSegmentAvailable(segment, adapter, 0, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to set segment available: %s", SCIGetErrorString(err));
        exit(1);
    }

    // Keep running until SIGINT is raised
    while (keep_running);

    // Make segment unavailable to prevent more connections
    SCISetSegmentUnavailable(segment, adapter, SCI_FLAG_NOTIFY | SCI_FLAG_FORCE_DISCONNECT, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to set segment unavailable, but continuing: %s", SCIGetErrorString(err));
    }

    // Remove interrupts
    // TODO: remove data interrupt
    do
    {
        SCIRemoveInterrupt(validate_irq, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

    SCIRemoveSegment(segment, 0, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to remove segment, but continuing");
    }

    free_gpu_buffer(gpu, buf);
}
