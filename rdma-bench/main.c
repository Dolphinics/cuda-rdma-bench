#include <stdio.h>
#include <sisci_api.h>
#include <signal.h>
#include "local.h"
#include "remote.h"
#include "util.h"
#include "log.h"
#include "dma.h"

static int keep_running = 1;

static void stop_server()
{
    keep_running = 0;
}


static void cb(trans_status_t success)
{
    printf("done\n");
    keep_running = 0;
}



void dump_memory(void* ptr, size_t len)
{
    for (size_t i = 0; i < len; ++i)
    {
        if (i % 16 == 0)
        {
            printf("\n");
        }

        printf("%02x ", *((uint8_t*) ptr + i));
    }
    printf("\n");
}


int client(unsigned node)
{
    r_segment_t rsegment;

    if (ConnectRemoteSegment(&rsegment, 0, node, 10) != 0)
    {
        return 1;
    }

    l_segment_t lsegment;
    if (CreateLocalSegment(&lsegment, 11, 0) != 0)
    {
        return 1;
    }

    if (AllocSegmentMem(lsegment, 4096) != 0)
    {
        RemoveLocalSegment(lsegment);
        return 1;
    }
    
    memset(GetLocalSegmentPtr(lsegment), 'a', 4096);

    int* ptr = (int*) GetLocalSegmentPtr(lsegment);
    if (ptr == NULL)
    {
        RemoveLocalSegment(lsegment);
        return 1;
    }

    *ptr = 0xb00bbabe;
    printf("%x\n", *ptr);

//    dump_memory(GetLocalSegmentPtr(lsegment), 4096);
    printf("\n\n\n\n\n\n\n");

    DmaRead(0, lsegment, 0, rsegment, 0, 4, (trans_cb_t) &cb, NULL);

    while (keep_running)
    {
    }

//    dump_memory(GetLocalSegmentPtr(lsegment), 4096);

    DisconnectRemoteSegment(rsegment);
    RemoveLocalSegment(lsegment);
    return 0;
}


int server()
{
    l_segment_t segment;

    if (CreateLocalSegment(&segment, 10, 0) != 0)
    {
        return 1;
    }

    if (AllocSegmentMem(segment, 4096) != 0)
    {
        RemoveLocalSegment(segment);
        return 1;
    }
    memset(GetLocalSegmentPtr(segment), 'c', 4096);

    uint32_t* ptr = (uint32_t*) GetLocalSegmentPtr(segment);
    if (ptr == NULL)
    {
        RemoveLocalSegment(segment);
        return 1;
    }

    *ptr = 0xdeadbeef;

    ExportLocalSegment(segment, 0, 0);

    while (keep_running)
    {
    }
    
    RemoveLocalSegment(segment);
    return 0;
}


int main(int argc, char** argv)
{
    sci_error_t err;
    SCIInitialize(0, &err);

    verbosity = 4;

    if (argc > 1)
    {
        unsigned remoteNode = GetNodeIdByName(argv[1]);
        client(remoteNode);
    }
    else
    {
        signal(SIGINT, (sig_t) &stop_server);
        signal(SIGTERM, (sig_t) &stop_server);
        server();
    }

    SCITerminate();

    return 0;
}
