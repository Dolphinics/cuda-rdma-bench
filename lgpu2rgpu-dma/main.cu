#include <cuda.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <stdint.h>
#include <sisci_api.h>
#include <signal.h>
#include <time.h>
extern "C" {
#include "reporting.h"
#include "common.h"
}



/* Program parameters */
static unsigned dma_flags = 0;
static dma_mode_t dma_mode = DMA_TRANSFER_ONE_WAY;
static size_t size_factor = 1e6;
static size_t size_count = 0;
static unsigned local_segment_id = NO_ID;
static unsigned remote_segment_id = NO_ID;
static unsigned remote_node_id = NO_NODE_ID;
static unsigned local_node_id = NO_NODE_ID;
static unsigned adapter_no = 0;
static int gpu_device_count = 0;
static int gpu_device_id = NO_ID;
static unsigned repeat = DEFAULT_REPEAT;
static unsigned mem_flags = 0;
static int verbosity = 0;



/* Program options */
static struct option options[] = {
    { .name = "local-id", .has_arg = 1, .flag = NULL, .val = 2 },
    { .name = "remote-id", .has_arg = 1, .flag = NULL, .val = 3 },
    { .name = "remote-node", .has_arg = 1, .flag = NULL, .val = 4 },
    { .name = "size", .has_arg = 1, .flag = NULL, .val = 5 },
    { .name = "gpu", .has_arg = 1, .flag = NULL, .val = 6 },
    { .name = "both-ways", .has_arg = 0, .flag = NULL, .val = 8 },
    { .name = "both-ways-separate", .has_arg = 0, .flag = NULL, .val = 9 },
    { .name = "pull", .has_arg = 0, .flag = NULL, .val = 7 },
    { .name = "info", .has_arg = 0, .flag = NULL, .val = 'h' },
    { .name = "help", .has_arg = 0, .flag = NULL, .val = 'h' },
    { .name = NULL, .has_arg = 0, .flag = NULL, .val = 0 }
};



/* Should the server keep running? */
static int keep_running = 1;



static void stop_program()
{
    log_info("Stopping server...");
    keep_running = 0;
}




uint64_t current_usecs()
{
    struct timespec ts;
    if (clock_gettime(CLOCK_REALTIME, &ts) < 0)
    {
        log_error("Failed to get realtime timestamp");
    }
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}




/* List local GPUs */
static void list_gpu_devices()
{
    cudaError_t err;

    fprintf(stderr, "Devices\n");
    fprintf(stderr, "  %3s %-20s %-9s %-8s %3s %7s %3s %4s   %-13s\n",
            "id", "name", "io addr", "map hmem", "tcc", "unified", "L1", "rdma", "glob mem size");

    for (int i = 0; i < gpu_device_count; ++i)
    {
        cudaDeviceProp prop;

        err = cudaGetDeviceProperties(&prop, i);
        if (err != cudaSuccess)
        {
            log_error("Unexpected error: %s", cudaGetErrorString(err));
            exit('h');
        }

        fprintf(stderr, "  %3d %-20s %02x:%02x.%-3x",
                i, prop.name, prop.pciBusID, prop.pciDomainID, prop.pciDeviceID);
        
        fprintf(stderr, " %8s", prop.canMapHostMemory ? "yes" : "no");
        fprintf(stderr, " %3s", prop.tccDriver ? "yes" : "no");
        fprintf(stderr, " %7s", prop.unifiedAddressing ? "yes" : "no");
        fprintf(stderr, " %3s", prop.globalL1CacheSupported ? "yes" : "no");
        fprintf(stderr, " %4s", prop.major >= 5 ? "yes" : "no");

        fprintf(stderr, "   %9.02f %-3s", prop.totalGlobalMem / (double) size_factor, 
                size_factor == 1e6 ? "MB" : "MiB");

        fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
}



/* Parse program arguments */
static void parse_args(int argc, char** argv)
{
    int opt, idx;
    char* str;

    while ((opt = getopt_long(argc, argv, "-:ha:igr:pmwv", options, &idx)) != -1)
    {
        switch (opt)
        {
            case ':': // missing value
                log_error("%s %s requires a value", (argv[optind - 1][1] != '-') ? "Option" : "Argument", argv[optind - 1]);
                goto give_usage;

            case '?': // unknown option or argument
                log_error("Unknown %s: %s", (argv[optind - 1][1] != '-') ? "option" : "argument", argv[optind - 1]);
                goto give_usage;

            case 'h': // show help
                list_gpu_devices();
                // TODO: List NTB adapters
                goto give_usage;

            case 'a':
                str = NULL;
                adapter_no = strtoul(optarg, &str, 10);
                if (str == NULL || *str != '\0' || adapter_no == 0)
                {
                    log_error("Option -a must be a valid NTB adapter number");
                    exit('a');
                }
                break;

            case 'i': // use MiBs (1024s) instead of MBs (1000s)
                size_factor = 1 << 20;
                break;

            case 'g': // do global DMA transfer
                dma_flags |= SCI_FLAG_DMA_GLOBAL;
                break;

            case 7: // pull data instead of pushing it
                dma_flags |= SCI_FLAG_DMA_READ;
                break;

            case 8: // set transfer mode
                if (dma_mode != DMA_TRANSFER_DEFAULT)
                {
                    log_warn("Argument --both-ways-separate will be ignored");
                }
                dma_mode = DMA_TRANSFER_BOTH_WAYS;
                break;

            case 9: // set transfer mode
                if (dma_mode != DMA_TRANSFER_DEFAULT)
                {
                    log_warn("Argument --both-ways will be ignored");
                }
                dma_mode = DMA_TRANSFER_TWO_WAY;
                break;

            case 'r': // set number of times to repeat for benchmarking granularity
                str = NULL;
                repeat = strtoul(optarg, &str, 10);
                if (str == NULL || *str != '\0' || repeat == 0)
                {
                    log_error("Option -r must be atleast 1");
                    exit('r');
                }
                break;

            case 'p':
                mem_flags |= cudaHostAllocMapped;
                mem_flags |= cudaHostAllocPortable;
                break;

            case 'm':
                mem_flags |= cudaHostAllocMapped;
                break;

            case 'w':
                mem_flags |= cudaHostAllocMapped;
                mem_flags |= cudaHostAllocWriteCombined;
                break;

            case 2: // set local segment ID
                str = NULL;
                local_segment_id = strtoul(optarg, &str, 0);
                if (str == NULL || *str != '\0' || local_segment_id >= MAX_ID)
                {
                    log_error("Argument --local-id must be between 0 and %u", MAX_ID);
                    exit(2);
                }
                break;

            case 3: // set remote segment ID
                str = NULL;
                remote_segment_id = strtoul(optarg, &str, 0);
                if (str == NULL || *str != '\0' || remote_segment_id >= MAX_ID)
                {
                    log_error("Argument --remote-id must be between 0 and %u", MAX_ID);
                    exit(3);
                }
                break;

            case 4: // set remote node ID
                str = NULL;
                remote_node_id = strtoul(optarg, &str, 10);
                if (str == NULL || *str != '\0' || remote_node_id >= MAX_NODE_ID)
                {
                    log_error("Argument --remote-node must be a valid cluster node ID");
                    exit(4);
                }
                break;

            case 5: // set segment size
                str = NULL;
                size_count = strtoul(optarg, &str, 0);
                if (str == NULL || *str != '\0' || size_count == 0)
                {
                    log_error("Argument --size must be at least 1");
                    exit(5);
                }
                break;

            case 6: // set local GPU
                str = NULL;
                gpu_device_id = strtol(optarg, &str, 10);
                if (str == NULL || *str != '\0' || gpu_device_id < 0 || gpu_device_id >= gpu_device_count)
                {
                    log_error("Argument --gpu-id must be a valid GPU ID");
                    exit(6);
                }
                break;

            case 'v': // increase verbosity level
                set_verbosity(++verbosity);
                break;
        }
    }

    // Find out local node ID
    // We have to do this here because we need to know which adapter to use
    log_info("Using local NTB adapter %u", adapter_no);
    
    sci_error_t err;
    SCIGetLocalNodeId(adapter_no, (unsigned*) &local_node_id, 0, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Unexpected error: %s", SCIGetErrorString(err));
        exit(1);
    }

    // Do some sanity checking
    if (gpu_device_id == NO_ID)
    {
        log_error("No local GPU is specified!");
        goto give_usage;
    }

    if (size_count == 0 && remote_node_id == NO_NODE_ID)
    {
        log_error("Either --size or --remote-node are required!");
        goto give_usage;
    }

    if (local_node_id == remote_node_id)
    {
        log_warn("Remote cluster node ID is the same as the local node ID");
    }

    if (local_segment_id == NO_ID)
    {
        local_segment_id = local_node_id;
        log_warn("No local segment ID is specified, using %d", local_node_id);
    }

    if (remote_node_id != NO_NODE_ID)
    {
        log_info("Running in client mode");

        if (remote_segment_id == NO_ID)
        {
            remote_segment_id = remote_node_id;
            log_warn("No remote segment ID is specified, using %d", remote_node_id);
        }

        if (size_count != 0)
        {
            log_warn("Argument --size has no effect in client mode");
        }
    }
    else
    {
        log_info("Running in server mode. Connect using node ID %d and segment ID %d", local_node_id, local_segment_id);

        if (dma_mode != DMA_TRANSFER_DEFAULT)
        {
            log_warn("Arguments --both-ways or --both-ways-separate have no effect in server mode");
        }

        if (!!(dma_flags & SCI_FLAG_DMA_GLOBAL))
        {
            log_warn("Option -g has no effect in server mode");
        }
    }

    return;

give_usage:
    fprintf(stderr, 
            "Usage: %s --gpu=<gpu id> --size=<size> [-i] [-pmw] [-a <adapter no>] [--local-id=<number>]\n"
            "   or: %s --gpu=<gpu id> [-pmw] --remote-node=<node id> [-a <adapter no>] [--remote-id=<number>] [--pull] [-g] [--both-ways|--both-ways-separate]\n"
            "   or: %s --info\n"
            "\nDescription\n"
            "    Copy memory from a local NVIDIA GPU to a remote NVIDIA GPU across a NTB link.\n"
            "\nArguments\n"
            "  --gpu=<gpu id>           specify which local GPU to use\n"
            "  --size=<size>            memory segment size in MB (or MiB if -i is set)\n"
            "  --remote-node=<node id>  remote cluster node ID\n"
            "  --local-id=<number>      number identifying the local memory segment\n"
            "  --remote-id=<number>     number identifying the memory segment on a remote host\n"
            "  --pull                   do DMA transfer from remote to local memory (default is local to remote)\n"
            "  --info                   list GPUs and NTB adapters and quit\n"
            "\nOptions\n"
            "   -i                      use IEC units (1024) instead of SI units (1000)\n"
            "   -p                      use cudaHostAllocPortable flag (sets -m implicitly)\n"
            "   -m                      use cudaHostAllocMapped flag\n"
            "   -w                      use cudaHostAllocWriteCombined flag (sets -m implicitly)\n"
            "   -a <adapter no>         local NTB adapter number (defaults to 0)\n"
            "   -g                      do global DMA transfer\n"
            "   -r <number>             number of times to repeat (default is %d)\n"
            "   -v                      increase verbosity level\n"
            , argv[0], argv[0], argv[0], DEFAULT_REPEAT);
    
    exit(1);
}



int main(int argc, char** argv)
{
    set_verbosity(verbosity);

    // Get number of GPUs
    cudaError_t cu_err = cudaGetDeviceCount(&gpu_device_count);

    if (cu_err != cudaSuccess)
    {
        log_error("No CUDA enabled local GPUs found!");
        exit(1);
    }

    // Initialize the SISCI API
    sci_error_t sci_err;
    sci_desc_t sci_desc;

    SCIInitialize(0, &sci_err);
    if (sci_err != SCI_ERR_OK)
    {
        log_error("Failed to initialize SISCI API!");
        exit(1);
    }

    SCIOpen(&sci_desc, 0, &sci_err);
    if (sci_err != SCI_ERR_OK)
    {
        log_error("Unexpected error: %s", SCIGetErrorString(sci_err));
        exit(1);
    }
    

    // Parse program arguments and set parameters
    parse_args(argc, argv);

    srand(current_usecs() / 1000);

    // Run as server or client?
    if (remote_node_id == NO_ID)
    {
        // FIXME: This is the non-recommended way of doing this, 
        //        but seems to be the only semi-portable way...
        signal(SIGINT, (sig_t) &stop_program);

        server_args args = {
            .desc = sci_desc,
            .adapter_id = adapter_no,
            .segment_id = local_segment_id,
            .segment_size = size_count * size_factor,
            .gpu_dev_id = gpu_device_id,
            .gpu_mem_flags = mem_flags,
            .dma_mode = dma_mode,
            .dma_flags = dma_flags,
            .keep_running = &keep_running
        };

        run_server(&args);
    }
    else
    {
        client_args args = {
            .desc = sci_desc,
            .adapter_id = adapter_no,
            .remote_node_id = remote_node_id,
            .remote_segment_id = remote_segment_id,
            .local_segment_id = local_segment_id,
            .gpu_dev_id = gpu_device_id,
            .gpu_mem_flags = mem_flags,
            .dma_mode = dma_mode,
            .dma_flags = dma_flags,
        };

        run_client(&args, size_factor, repeat);
    }

    SCIClose(sci_desc, 0, &sci_err);
    SCITerminate();

    exit(0);
}
