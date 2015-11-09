#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>
#include <sisci_api.h>
#include "common.h"

#ifdef PING

extern void PingNode(
        sci_desc_t device_desc,    
        uint32_t local_adapter_no, 
        uint32_t remote_node_id,
        size_t segment_size
        );
#else

extern void PongNode(
        sci_desc_t device_desc, 
        uint32_t local_adapter_no,
        uint32_t local_node_id,
        uint32_t remote_node_id,
        size_t segment_size
        );

#endif


#if 0
/* Check if user is root */
static bool IsRoot(char** envp)
{
    for (int idx = 0; envp[idx] != NULL; ++idx) 
    {
        int len = strlen(envp[idx]);
        int pos;

        for (pos = 0; pos < len && envp[idx][pos] != '='; ++pos);

        if (pos != len)
        {
            envp[idx][pos] = '\0';
            if (strcmp(&envp[idx][0], "USER") == 0)
            {
                return strcmp(&envp[idx][pos+1], "root") == 0;
            }
        }
    }

    return false;
}
#endif

int main(int argc, char** argv/*, char** envp*/)
{
    uint32_t remote_node_id = NO_NODE_ID;
    uint32_t local_adapter = 0;
    uint32_t local_node_id = NO_NODE_ID;
    sci_desc_t dev_desc;
    size_t segment_size = 0;

    // Parse program arguments
    int opt, optidx;
    char* strptr;

    
#ifdef PING
    const char* opts = ":ha:s:";
#else
    const char* opts = ":hr:a:s:";
#endif
    while ((opt = getopt_long(argc, argv, opts, NULL, &optidx)) != -1)
    {
        switch (opt)
        {
            case ':': // missing value
                fprintf(stderr, "Option -%c requires a value\n", optopt);
                goto give_usage;

            case '?': // unknown flag
                fprintf(stderr, "Unknown option: -%c\n", optopt);
                goto give_usage;

            case 'h': // show help
                goto give_usage;

#ifndef PING
            case 'r': // set remote node ID
                strptr = NULL;
                remote_node_id = strtoul(optarg, &strptr, 0);
                if (strptr == NULL || *strptr != '\0')
                {
                    fprintf(stderr, "Option -r requires a remote node ID\n");
                    goto give_usage;
                }
                break;
#endif

            case 's': // set segment size
                strptr = NULL;
                segment_size = strtoul(optarg, &strptr, 0);
                if (strptr == NULL || *strptr != '\0')
                {
                    fprintf(stderr, "Option -s requires a valid segment size\n");
                    goto give_usage;
                } 
                break;

            case 'a': // set local adapter number
                strptr = NULL;
                local_adapter = strtoul(optarg, &strptr, 0);
                if (strptr == NULL || *strptr != '\0')
                {
                    fprintf(stderr, "Option -a requires a valid adapter number\n");
                    goto give_usage;
                }
                break;
        }
    }

#ifndef PING
    if (NO_NODE_ID == remote_node_id)
    {
        fprintf(stderr, "Remote node ID is required!\n");
        goto give_usage;
    }
#endif

    if (0 == segment_size)
    {
        fprintf(stderr, "Segment size is required!\n");
        goto give_usage;
    }

    sci_error_t err;

    // Initalize SISCI API
    SCIInitialize(0, &err);
    if (SCI_ERR_OK != err)
    {
        fprintf(stderr, "Initialization failed, aborting\n");
        return 2;
    }

    // Open SISCI device descriptor
    SCIOpen(&dev_desc, 0, &err);
    if (SCI_ERR_OK != err)
    {
        fprintf(stderr, "Failed to open device descriptor\n");
        return 2;
    }

    // Get local node ID
    SCIGetLocalNodeId(local_adapter, &local_node_id, 0, &err);
    if (SCI_ERR_OK != err)
    {
        fprintf(stderr, "Could not find the local adapter %u\n", local_adapter);
        return 2;
    }

    // Probe remote end to check if it's up
    if (remote_node_id != NO_NODE_ID)
    {
        SCIProbeNode(dev_desc, local_adapter, remote_node_id, 0, &err);
        switch (err)
        {
            case SCI_ERR_NO_LINK_ACCESS:
                fprintf(stderr, "No link access on adapter %u\n", local_adapter);
                return 2;
            case SCI_ERR_NO_REMOTE_LINK_ACCESS:
                fprintf(stderr, "Node %u unreachable through adapter %u\n", remote_node_id, local_adapter);
                return 2;
            case SCI_ERR_OK:
                break;
            default:
                fprintf(stderr, "Unexpected error\n");
                return 2;
        }
    }

    // Run test
#ifdef PING
    PingNode(dev_desc, local_adapter, local_node_id, segment_size);
#else
    PongNode(dev_desc, local_adapter, local_node_id, remote_node_id, segment_size);
#endif

    // Clean up
    SCIClose(dev_desc, 0, &err);
    SCITerminate();

    return 0;

give_usage:
    fprintf(stderr, 
#ifdef PING
            "Usage: %s -s <segment size> [-a <local adapter no>]\n"
#else
            "Usage: %s -s <segment size> [-a <local adapter no>] -r <remote node id>\n"
#endif
            "\n"
            "  -s\tsegment size in bytes\n"
            "  -a\tlocal adapter number\n"
#ifndef PING
            "  -r\tremote node id\n"
#endif
            "\n",
            argv[0]);

    return 1;
}
