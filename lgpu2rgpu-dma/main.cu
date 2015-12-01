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
#include "local.h"
#include "reporting.h"
#include "common.h"
}



/* Program parameters */
static int dma_global = 0;
static int dma_pull = 0;
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
static int repeat = DEFAULT_REPEAT;



/* Program options */
static struct option options[] = {
    { .name = "local-id", .has_arg = 1, .flag = NULL, .val = 2 },
    { .name = "remote-id", .has_arg = 1, .flag = NULL, .val = 3 },
    { .name = "remote-node", .has_arg = 1, .flag = NULL, .val = 4 },
    { .name = "size", .has_arg = 1, .flag = NULL, .val = 5 },
    { .name = "gpu", .has_arg = 1, .flag = NULL, .val = 6 },
    { .name = "both-ways", .has_arg = 0, .flag = NULL, .val = 8 },
    { .name = "pull", .has_arg = 0, .flag = NULL, .val = 7 },
    { .name = "info", .has_arg = 0, .flag = NULL, .val = 'h' },
    { .name = "help", .has_arg = 0, .flag = NULL, .val = 'h' },
    { .name = NULL, .has_arg = 0, .flag = NULL, .val = 0 }
};



/* Signal handler to stop the benchmarking server */
static void stop_program()
{
    log_info("Stopping server...");
    stop_server();
}



/* Get current timestamp in microseconds */
extern "C"
uint64_t current_usecs()
{
    struct timespec ts;
    if (clock_gettime(CLOCK_REALTIME, &ts) < 0)
    {
        log_error("Failed to get realtime timestamp");
    }
    return ts.tv_sec * 1e6L + ts.tv_nsec / 1e3L;
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

    while ((opt = getopt_long(argc, argv, "-:ha:igr:v", options, &idx)) != -1)
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
                dma_global = 1;
                break;

            case 7: // pull data instead of pushing it
                dma_pull = 1;
                break;

            case 8: // set transfer mode
                dma_mode = DMA_TRANSFER_TWO_WAY;
                break;

            case 'r': // set number of times to repeat for benchmarking granularity
                str = NULL;
                repeat = strtol(optarg, &str, 10);
                if (str == NULL || *str != '\0' || repeat <= 0)
                {
                    log_error("Option -r must be at least 1");
                    exit('r');
                }
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
                ++verbosity;
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

    log_info("Using local GPU %u", gpu_device_id);

    if (size_count == 0 && remote_node_id == NO_NODE_ID)
    {
        log_error("Either --size or --remote-node are required!");
        goto give_usage;
    }

    if ((size_count * size_factor) >= MAX_SEGMENT_SIZE)
    {
        log_error("Maximum segment size is %lu MiB", MAX_SEGMENT_SIZE / (1 << 20));
        exit(1);
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

        if (dma_mode != DMA_TRANSFER_ONE_WAY)
        {
            log_warn("Argument --both-ways have no effect in server mode");
        }

        if (!!(dma_global))
        {
            log_warn("Option -g has no effect in server mode");
        }

        if (repeat != DEFAULT_REPEAT)
        {
            log_warn("Option -r has no effect in server mode");
        }
    }

    return;

give_usage:
    fprintf(stderr, 
            "Usage: %s --gpu=<gpu id> --size=<size> [-i] [-a <adapter no>] [--local-id=<number>]\n"
            "   or: %s --gpu=<gpu id> --remote-node=<node id> [-a <adapter no>] [--remote-id=<number>] [--pull] [-g] [--both-ways]\n"
            "   or: %s --info\n"
            "\nDescription\n"
            "    Copy memory from a local NVIDIA GPU to a remote NVIDIA GPU across a NTB link.\n"
            "\nArguments\n"
            "  --gpu=<gpu id>           specify which local GPU to use\n"
            "  --size=<size>            memory segment size in MB (or MiB if -i is set)\n"
            "  --remote-node=<node id>  remote cluster node ID\n"
            "  --local-id=<number>      number identifying the local memory segment\n"
            "  --remote-id=<number>     number identifying the memory segment on a remote host\n"
            "  --pull                   pull data from remote hode instead of pushing it\n"
            "  --both-ways              do DMA transfer in both directions simultaneously\n"
            "  --info                   list GPUs and NTB adapters and quit\n"
            "\nOptions\n"
            "   -i                      use IEC units (1024) instead of SI units (1000)\n"
            "   -a <adapter no>         local NTB adapter number (defaults to 0)\n"
            "   -g                      do global DMA transfer\n"
            "   -r <number>             number of times to repeat (default is %d)\n"
            "   -v                      increase verbosity level\n"
            , argv[0], argv[0], argv[0], DEFAULT_REPEAT);
    
    exit(1);
}



/* Run benchmarking client */
void client(sci_desc_t sd)
{
    sci_error_t err;
    sci_remote_segment_t remote_segment;

    // Connect to remote segment
    log_debug("Trying to connect to remote segment %u on remote node %u", remote_segment_id, remote_node_id);
    do
    {
        SCIConnectSegment(sd, &remote_segment, remote_node_id, remote_segment_id, adapter_no, NULL, NULL, SCI_INFINITE_TIMEOUT, 0, &err);
    }
    while (err != SCI_ERR_OK);

    // Retrieve segment size
    size_t remote_segment_size = SCIGetRemoteSegmentSize(remote_segment);
    log_info("Connected to segment %u (%.2f %s) on remote node %u", 
            remote_segment_id, remote_segment_size / (double) size_factor, size_factor == 1e6 ? "MB" : "MiB", remote_node_id);

    // Connect to remote validation IRQ
    sci_remote_interrupt_t validate_irq;

    log_debug("Connecting to remote validation interrupt");
    do
    {
        SCIConnectInterrupt(sd, &validate_irq, remote_node_id, adapter_no, remote_segment_id, SCI_INFINITE_TIMEOUT, 0, &err);
    }
    while (err != SCI_ERR_OK);

    // Create local GPU buffer
    void* buf = make_gpu_buffer(gpu_device_id, remote_segment_size);
    uint8_t val = rand() & 255;
    gpu_memset(gpu_device_id, buf, remote_segment_size, val);

    sci_local_segment_t local_segment = make_local_segment(sd, adapter_no, local_segment_id, buf, remote_segment_size);

    // Do benchmarks
    unsigned dma_flags = 0;
    dma_flags |= dma_pull ? SCI_FLAG_DMA_READ : 0;
    dma_flags |= dma_global ? SCI_FLAG_DMA_GLOBAL : 0;
    
    log_info("Starting benchmark...");
    uint64_t usecs = benchmark(sd, remote_node_id, adapter_no, local_segment, remote_segment, remote_segment_size, dma_mode, dma_flags, repeat);
    double megabytes_per_second = (remote_segment_size * repeat) / (double) usecs;

    fprintf(stdout, "%5.3f %-5s\n", megabytes_per_second, size_factor == 1e6 ? "MB/s" : "MiB/s");

    log_debug("Validating buffer after transfer...");
    if (!dma_pull)
    {
        SCITriggerInterrupt(validate_irq, 0, &err);
    }
    else
    {
        size_t last_byte = validate_gpu_buffer(gpu_device_id, buf, remote_segment_size, 0); // FIXME: pass in expected byte
        if (last_byte != remote_segment_size)
        {
            log_error("Buffer is garbled, last correct byte is %lu but buffer size is %lu", last_byte, remote_segment_size);
        }
        else
        {
            log_info("Buffer is valid after DMA transfer");
        }
    }
    
    // Clean up
    SCIDisconnectInterrupt(validate_irq, 0, &err);
    do
    {
        SCIDisconnectSegment(remote_segment, 0, &err);
    }
    while (err == SCI_ERR_BUSY);
    free_gpu_buffer(gpu_device_id, buf);
}



int main(int argc, char** argv)
{
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

        server(sci_desc, adapter_no, gpu_device_id, local_segment_id, size_factor * size_count);
    }
    else
    {
        client(sci_desc);
    }

    SCIClose(sci_desc, 0, &sci_err);
    SCITerminate();

    exit(0);
}
