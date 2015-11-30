#include <sisci_api.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include "dma.h"
#include "remote.h"
#include "local.h"
#include "shared_sci.h"

uint64_t GetCurrentMicros()
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}


double DMAPush(sci_desc_t sciDev, uint32_t adapt, local_t local, size_t loff, remote_t remote, size_t roff, int repeat)
{
    sci_error_t err;
    sci_dma_queue_t queue;

    SCICreateDMAQueue(sciDev, &queue, adapt, 1, 0, &err);
    sciAssert(err);

    uint64_t start, end;

    start = GetCurrentMicros();

    fprintf(stdout, "Starting push transfer...");
    fflush(stdout);

    for (int i = 0; i < repeat; ++i)
    {
        SCIStartDmaTransfer(queue, local.segment, remote.segment, loff, local.length - loff, roff, NULL, NULL, 0/*SCI_FLAG_DMA_WAIT*/, &err);
        sciAssert(err);

        sleep(2);

        SCIWaitForDMAQueue(queue, SCI_INFINITE_TIMEOUT, 0, &err);
        sciAssert(err);
    }

    fprintf(stdout, " DONE\n");

    end = GetCurrentMicros();

    SCIRemoveDMAQueue(queue, 0, &err);
    sciAssert(err);

    double megabytesPerSecond = (double) (local.length - loff) / (double) (end - start) / (double) repeat;

    return megabytesPerSecond;
}


double DMAPull(sci_desc_t sciDev, uint32_t adapt, local_t local, size_t loff, remote_t remote, size_t roff, int repeat)
{
    sci_error_t err;
    sci_dma_queue_t queue;

    SCICreateDMAQueue(sciDev, &queue, adapt, 1, 0, &err);
    sciAssert(err);

    uint64_t start, end;

    start = GetCurrentMicros();

    // FIXME: DEBUG
#if 0
#define DEBUG
    sci_desc_t sciDev2;
    SCIOpen(&sciDev2, 0, &err);
    sciAssert(err);

    sci_local_segment_t segment;
    SCICreateSegment(sciDev2, &segment, 1, local.length - loff, NULL, NULL, 0, &err);
    sciAssert(err);

    SCIPrepareSegment(segment, adapt, 0, &err);
    sciAssert(err);

    sci_map_t map;
    void* addr = SCIMapLocalSegment(segment, &map, 0, remote.length, NULL, 0, &err);
    sciAssert(err);
    
    uint32_t value = rand() & 255;
    printf("******** RAM value: %u\n", value);
    *((uint32_t*) addr) = value;
#endif
    // FIXME: DEBUG


    fprintf(stdout, "Starting push transfer...");
    fflush(stdout);

    uint32_t flags = SCI_FLAG_DMA_GLOBAL;

    for (int i = 0; i < repeat; ++i)
    {
        SCIStartDmaTransfer(queue, local.segment, remote.segment, loff, local.length - loff, roff, NULL, NULL, flags | SCI_FLAG_DMA_READ, &err);
       
        //SCIStartDmaTransfer(queue, segment, remote.segment, loff, local.length - loff, roff, NULL, NULL, flags | SCI_FLAG_DMA_READ, &err);
        
        
        //SCIStartDmaTransfer(queue, local.segment, remote.segment, loff, local.length - loff, roff, NULL, NULL, flags, &err);

       //SCIStartDmaTransfer(queue, segment, remote.segment, loff, local.length - loff, roff, NULL, NULL, flags, &err);
       //
        

        sciAssert(err);

        SCIWaitForDMAQueue(queue, SCI_INFINITE_TIMEOUT, 0, &err);
        sciAssert(err);

    }

    fprintf(stdout, " DONE\n");
#ifdef DEBUG
    printf("*addr = %u\n", *((unsigned*) addr));
#endif

    end = GetCurrentMicros();

    SCIRemoveDMAQueue(queue, 0, &err);
    sciAssert(err);

    DumpGPUMemory(local);

    double megabytesPerSecond = (double) (local.length - loff) / (double) (end - start) / (double) repeat;

    return megabytesPerSecond;
}
