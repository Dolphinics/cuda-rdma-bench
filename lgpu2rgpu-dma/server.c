#include <stdlib.h>
#include <sisci_api.h>
#include <stdint.h>
#include "common.h"
#include "local.h"
#include "reporting.h"



sci_callback_action_t dma_transfer(void* buff_info, sci_local_data_interrupt_t irq, void* client_data, unsigned length, sci_error_t status)
{
    
    return SCI_CALLBACK_CONTINUE;
}



sci_callback_action_t print_buffer_byte(void* buff_info, sci_local_interrupt_t irq, sci_error_t status)
{
    bufhandle_t* bh = (bufhandle_t*) buff_info;

    uint8_t byte_value = validate_buffer(*bh);

    fprintf(stdout, "Buffer byte (trigger): %02x\n", byte_value);

    return SCI_CALLBACK_CONTINUE;
}



void run_server(server_args* args)
{
    sci_error_t err;

    bufhandle_t bh;
    bh = create_gpu_buffer(args->desc, args->adapter_id, args->gpu_dev_id, args->segment_id, args->segment_size, args->gpu_mem_flags);

    fprintf(stdout, "Buffer byte (initial): %02x\n", validate_buffer(bh));

    sci_local_data_interrupt_t data_irq;
    unsigned data_irq_no = args->segment_id;

    SCICreateDataInterrupt(args->desc, &data_irq, args->adapter_id, &data_irq_no, &dma_transfer, &bh, SCI_FLAG_FIXED_INTNO | SCI_FLAG_USE_CALLBACK, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to create data interrupt: %s", SCIGetErrorString(err));
        exit(1);
    }

    sci_local_interrupt_t trigger_irq;
    unsigned trigger_irq_no = args->segment_id;
    
    SCICreateInterrupt(args->desc, &trigger_irq, args->adapter_id, &trigger_irq_no, &print_buffer_byte, &bh, SCI_FLAG_FIXED_INTNO | SCI_FLAG_USE_CALLBACK, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to create interrupt: %s", SCIGetErrorString(err));
        exit(1);
    }

    SCISetSegmentAvailable(bh.segment, args->adapter_id, 0, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Couldn't export segment: %s", SCIGetErrorString(err));
        exit(1);
    }

    while (*args->keep_running);

    SCISetSegmentUnavailable(bh.segment, args->adapter_id, SCI_FLAG_NOTIFY | SCI_FLAG_FORCE_DISCONNECT, &err);

    do
    {
        SCIRemoveDataInterrupt(data_irq, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

    do
    {
        SCIRemoveInterrupt(trigger_irq, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

    free_gpu_buffer(bh);
}
