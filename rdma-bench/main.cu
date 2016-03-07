#include <stdio.h>
#include <sisci_api.h>
#include <signal.h>
#include "local.h"
#include "remote.h"
#include "util.h"
#include "log.h"

static int keep_running = 1;

static void stop_server()
{
    keep_running = 0;
}


int client(unsigned node)
{
    r_segment_t segment;

    if (ConnectRemoteSegment(&segment, 0, node, 10))
    {
        return 1;
    }

    DisconnectRemoteSegment(segment);
    return 0;
}


int server()
{
    l_segment_t segment;

    if (CreateLocalSegment(&segment, 10, 0) != 0)
    {
        return 1;
    }

    if (AllocSegmentMem(segment, 4) != 0)
    {
        RemoveLocalSegment(segment);
        return 1;
    }

    int* ptr = (int*) GetLocalSegmentPtr(segment);
    if (ptr == NULL)
    {
        RemoveLocalSegment(segment);
        return 1;
    }

    *ptr = 0xdeadbeef;

    ExportLocalSegment(segment, 0, 0);

    while (keep_running)
    {
        printf("%x\n", *ptr);
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
