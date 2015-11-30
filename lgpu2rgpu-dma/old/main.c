#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <stdint.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>
#include "shared_cuda.h"
#include "shared_sci.h"
#include "local.h"
#include "remote.h"
#include "dma.h"

#define MAX_SIZE_NUM 24

static int run = 0;

static void terminate(int signal)
{
    if (signal == SIGINT)
    {
        run = 0;
    }
}

int main(int argc, char** argv)
{
    // Parameters
    int gpu = 0;
    int mode = 0;
    size_t factor = 1e6;
    size_t size = 0;
    size_t loff = 0;
    size_t roff = 0;
    int gpu_count;
    int remote_node = -1;
    unsigned adapter = 0;

    gpu_count = GetGpuCount();
    if (gpu_count < 1)
    {
        fprintf(stderr, "No CUDA GPUs detected\n");
        return 1;
    }

    // Parse arguments
    struct option opts[] = {
        { .name = "gpu", .has_arg = 1, .flag = NULL, .val = 1 },
        { .name = "size", .has_arg = 1, .flag = NULL, .val = 2 },
        { .name = "roff", .has_arg = 1, .flag = NULL, .val = 3 },
        { .name = "loff", .has_arg = 1, .flag = NULL, .val = 4 },
        { .name = "node", .has_arg = 1, .flag = NULL, .val = 5 },
        { .name = "push", .has_arg = 0, .flag = NULL, .val = 6 },
        { .name = "pull", .has_arg = 0, .flag = NULL, .val = 7 },
        { .name = "help", .has_arg = 0, .flag = NULL, .val = 'h' }
    };
    int opt, optidx;
    char* strptr;

    // TODO: implement -l and -m -p (hostAlloc)
    while ((opt = getopt_long(argc, argv, "-:ha:i", opts, &optidx)) != -1)
    {
        switch (opt)
        {
            case ':': // missing value
                // FIXME: I'm not sure this works for both --long and -s options
                fprintf(stderr, "Argument %s requires an argument\n", argv[optind - 1]);
                goto giveUsage;

            case '?': // unknown flag
                // FIXME: I'm not sure this works for both --long and -s options
                fprintf(stderr, "Unknown argument or option: %s\n", argv[optind - 1]);
                goto giveUsage;

            case 'h': // show help
                goto giveUsage;

            case 1: // set local GPU
                strptr = NULL;
                gpu = strtol(optarg, &strptr, 0);
                if (strptr == NULL || *strptr != '\0' || gpu < 0 || gpu >= gpu_count)
                {
                    fprintf(stderr, "Argument --gpu requires a valid CUDA GPU number\n");
                    return 1;
                }
                break;

            case 2: // set segment size
                strptr = NULL;
                size = strtoul(optarg, &strptr, 0);
                if (strptr == NULL || *strptr != '\0' || size == 0)
                {
                    fprintf(stderr, "Argument --size requires a valid memory size\n");
                    return 1;
                }
                break;

            case 3: // set remote offset
                strptr = NULL;
                roff = strtoul(optarg, &strptr, 0);
                if (strptr == NULL || *strptr != '\0')
                {
                    fprintf(stderr, "Argument --roff requires a valid memory offset\n");
                    return 1;
                }
                break;

            case 4: // set local offset
                strptr = NULL;
                loff = strtoul(optarg, &strptr, 0);
                if (strptr == NULL || *strptr != '\0')
                {
                    fprintf(stderr, "Argument --loff requires a valid memory offset\n");
                    return 1;
                }
                break;

            case 5: // set remote node id
                strptr = NULL;
                remote_node = strtol(optarg, &strptr, 0);
                if (strptr == NULL || *strptr != '\0' || remote_node < 0)
                {
                    fprintf(stderr, "Argument --node requires a valid cluster node ID\n");
                    return 1;
                }
                break;

            case 6: // set push
                mode |= 1;
                break;

            case 7: // set pull
                mode |= 2;
                break;

            case 'a': // set local adapter number
                strptr = NULL;
                adapter = strtoul(optarg, &strptr, 0);
                if (strptr == NULL || *strptr != '\0')
                {
                    fprintf(stderr, "Option -a requires a valid adapter number\n");
                    return 1;
                }
                break;

            case 'i': // use 1024 instead of 1000
                factor = 1 << 20;
                break;
        }
    }

    // Do some sanity checking
    if (size == 0)
    {
        fprintf(stderr, "No memory chunk size specified!\n");
        goto giveUsage;
    }

    if (roff >= size || loff >= size)
    {
        fprintf(stderr, "Memory offsets are larger than memory chunk!!\n");
        return 1;
    }

    if (mode != 0 && remote_node < 0)
    {
        fprintf(stderr, "Transfer mode is set, but no remote cluster ID is specified!\n");
        goto giveUsage;
    }

    srand(time(NULL));

    sci_error_t err;
    
    // Initialize SISCI API
    SCIInitialize(0, &err);
    sciAssert(err);

    sci_desc_t desc;
    SCIOpen(&desc, 0, &err);
    sciAssert(err);

    unsigned local_node = GetLocalNodeID(adapter);

    local_t local = CreateLocalSegment(desc, local_node, adapter, gpu, size * factor);

    if (mode == 0)
    {
        run = 1;
        signal(SIGINT, &terminate);
    }

    SCISetSegmentAvailable(local.segment, adapter, 0, &err);
    sciAssert(err);


    if (mode != 0 && remote_node >= 0)
    {
        remote_t remote = ConnectRemoteSegment(desc, remote_node, adapter);

        loff *= factor;
        roff *= factor;

        if ((remote.length - roff) < (local.length - loff))
        {
            fprintf(stderr, "Remote segment is smaller than local segment, aborting....\n");
            run = 0;
        }
        else
        {
            if (!!(mode & 1))
            {
                double push;
                push = DMAPush(desc, adapter, local, loff, remote, roff, 3);
            }

            if (!!(mode & 2))
            {
                double pull;
                pull = DMAPull(desc, adapter, local, loff, remote, roff, 15);
                fprintf(stdout, "PULL: %05.3f %s\n", pull, factor == 1e6 ? "MB/s" : "MiB/s");
            }

            //fprintf(stdout, "PUSH: %05.3f %s\n", push, factor == 1e6 ? "MB/s" : "MiB/s");
            
        }

        FreeRemoteSegment(remote);
    }

    while (run);

    DumpGPUMemory(local);
    FreeLocalSegment(local, adapter);

    SCIClose(desc, 0, &err);
    SCITerminate();

    return 0;

giveUsage:
    fprintf(stderr,
            "Usage: %s --node=<remote node> --push --size=<size> [--roff=<size>] [--loff=<size>] [-a <number>] [--gpu=<device no>] [-i]\n"
            "   or: %s --node=<remote node> --pull --size=<size> [--roff=<size>] [--loff=<size>] [-a <number>] [--gpu=<device no>] [-i]\n"
            "   or: %s --size=<size> [-a <number>] [--gpu=<device no>] [-i]\n"
            "   or: %s -l\n"
            "\nArguments\n"
            "  --size=<size>        memory chunk size in MB (or MiB if -i is set)\n"
            "  --roff=<size>        offset in MB (or MiB if -i is set) in remote GPU memory\n"
            "  --loff=<size>        offset in MB (or MiB if -i is set) in local GPU memory\n"
            "  --node=<remote node> remote cluster node ID\n"
            "  --pull               pull data from remote GPU memory\n"
            "  --push               push data to remote GPU memory\n"
            "  --gpu=<device no>   specify CUDA GPU (use -l to list valid local devices)\n"
            "\nOptions\n"
            "   -a <number>         local SISCI adapter number\n"
            "   -i                  use IEC units (1024) instead of SI units (1000)\n"
            "   -l                  list local CUDA GPUs and quit\n"
            "\nBuild date: %s %s\n",
            argv[0], 
            argv[0],
            argv[0],
            argv[0],
            __DATE__, __TIME__);

    return 1;
}
