#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <sisci_api.h>
#include <signal.h>
#include "translist.h"
#include "reporting.h"
#include "common.h"
#include "gpu.h"
#include "util.h"
#include "bench.h"


/* Program options */
static struct option options[] = {
    { .name = "adapter", .has_arg = 1, .flag = NULL, .val = 'a' },
    { .name = "remote-node", .has_arg = 1, .flag = NULL, .val = 'n' },
    { .name = "local-id", .has_arg = 1, .flag = NULL, .val = 'l' },
    { .name = "remote-id", .has_arg = 1, .flag = NULL, .val = 'r' },
    { .name = "size", .has_arg = 1, .flag = NULL, .val = 's' },
    { .name = "gpu", .has_arg = 1, .flag = NULL, .val = 'g' },
    { .name = "type", .has_arg = 1, .flag = NULL, .val = 'b' },
    { .name = "bench", .has_arg = 1, .flag = NULL, .val = 'b' },
    { .name = "benchmark", .has_arg = 1, .flag = NULL, .val = 'b' },
    { .name = "count", .has_arg = 1, .flag = NULL, .val = 'c' },
    { .name = "verbose", .has_arg = 0, .flag = NULL, .val = 'v' },
    { .name = "iec", .has_arg = 0, .flag = NULL, .val = 'i' },
    { .name = "help", .has_arg = 0, .flag = NULL, .val = 'h' },
    { .name = NULL, .has_arg = 0, .flag = NULL, .val = 0 }
};


/* List supported benchmarking modes */
static void list_bench_modes()
{
    fprintf(stderr, "Benchmarking operations\n");
    fprintf(stderr, "  %-18s  %-56s\n", "name", "explanation");

    const bench_mode_t* mode = all_benchmarking_modes;

    while (*mode != BENCH_DO_NOTHING)
    {
        fprintf(stderr, "  %-18s  %-56s\n", bench_mode_name(*mode), bench_mode_desc(*mode));
        ++mode;
    }
    fprintf(stderr, "\n");
}


/* List local GPUs */
static void list_gpu_devices(size_t size_factor)
{
    cudaError_t err;

    int gpu_device_count = 0;
    err = cudaGetDeviceCount(&gpu_device_count);
    if (err != cudaSuccess)
    {
        log_error("Unexpected error: %s", cudaGetErrorString(err));
        gpu_device_count = 0;
    }

    fprintf(stderr, "Devices\n");
    fprintf(stderr, "  %2s %-20s %-9s %-8s %3s %7s %3s %4s   %-13s\n",
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

        int rdma = !strncmp("Tesla", prop.name, 5) || !strncmp("Quadro", prop.name, 6);

        fprintf(stderr, "  %2d %-20s %02x:%02x.%-3x",
                i, prop.name, prop.pciBusID, prop.pciDomainID, prop.pciDeviceID);
        
        fprintf(stderr, " %8s", prop.canMapHostMemory ? "yes" : "no");
        fprintf(stderr, " %3s", prop.tccDriver ? "yes" : "no");
        fprintf(stderr, " %7s", prop.unifiedAddressing ? "yes" : "no");
        fprintf(stderr, " %3s", prop.globalL1CacheSupported ? "yes" : "no");
        fprintf(stderr, " %4s", rdma ? "yes" : "no");

        fprintf(stderr, "   %9.02f %-3s", prop.totalGlobalMem / (double) size_factor, 
                size_factor == 1e6 ? "MB" : "MiB");

        fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
}


/* Retrieve local cluster node ID */
static unsigned get_local_node_id(unsigned adapter_no)
{
    sci_error_t err = SCI_ERR_OK;
    unsigned node_id = NO_NODE;

    sci_desc_t desc;
    SCIOpen(&desc, 0, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("%s", SCIGetErrorString(err));
        exit(1);
    }

    SCIGetLocalNodeId(adapter_no, &node_id, 0, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("%s", SCIGetErrorString(err));
        exit(1);
    }

    SCIClose(desc, 0, &err);

    return node_id;
}


/* Give program usage */
static void give_usage(const char* progname)
{
    fprintf(stderr,
            "Usage: %s --size=<size>\n"
            "   or: %s --remote-node=<node id> --mode=<benchmark type>\n"
            "\nDescription\n"
            "    Benchmark how long it takes to transfer memory between a local and a\n"
            "    remote segment across an NTB link.\n"
            "\nServer arguments\n"
            "  --size=<size>            memory size in MB (or MiB if --iec is set)\n"
            "\nClient arguments\n"
            "  --remote-node=<node id>  remote cluster node ID\n"
            "  --bench=<bencmark type>  specify benchmarking operation\n"
            "  --count=<number>         number of times to repeat test\n"
            "\nOptional arguments (both client and server)\n"
            "  --adapter=<adapter no>   local host adapter card number (defaults to 0)\n"
            "  --local-id=<segment id>  number identifying the local segment\n"
            "  --remote-id=<segment id> number identifying the remote segment\n"
            "  --gpu=<gpu id>           specify a local GPU to use\n"
            "  --verbose                increase verbosity level\n"
            "  --iec                    use IEC units (1024) instead of SI units (1000)\n"
            "  --help                   show list of local GPUs and benchmarking operations\n"
            , progname, progname);
}


int main(int argc, char** argv)
{
    /* Program parameters */
    unsigned local_adapter = 0;
    unsigned remote_node_id = NO_NODE;
    unsigned local_node_id = NO_NODE;
    unsigned local_segment_id = NO_ID;
    unsigned remote_segment_id = NO_ID;
    int local_gpu_id = NO_GPU;

    size_t local_segment_count = 0;
    size_t local_segment_factor = 1e6;

    int repeat_count = 1;
    bench_mode_t mode = BENCH_DO_NOTHING;

    /* Do shortcut */
    if (argc < 2)
    {
        give_usage(argv[0]);
        exit(1);
    }

    /* Parse program arguments */
    int opt, idx;
    char* str;

    while ((opt = getopt_long(argc, argv, "-:a:n:l:r:s:g:m:c:vih", options, &idx)) != -1)
    {
        switch (opt)
        {
            case ':': // missing value
                log_error("Option %s requires a value", argv[optind-1]);
                give_usage(argv[0]);
                exit(':');

            case '?': // unknown option
                log_error("Unknown option: %s", argv[optind-1]);
                give_usage(argv[0]);
                exit('?');

            case 'h': // show help
                list_gpu_devices(local_segment_factor);
                list_bench_modes();
                fprintf(stderr, "\n");
                give_usage(argv[0]);
                exit('h');

            case 'a': // set local adapter number
                str = NULL;
                local_adapter = strtoul(optarg, &str, 10);
                if (str == NULL || *str != '\0')
                {
                    log_error("Argument %s must be a valid adapter number", argv[optind-1]);
                    exit('a');
                }
                break;

            case 'n': // set remote node
                str = NULL;
                remote_node_id = strtoul(optarg, &str, 10);
                if (str == NULL || *str != '\0' || remote_node_id == NO_NODE)
                {
                    log_error("Argument %s must be a valid cluster node ID", argv[optind-1]);
                    exit('n');
                }
                break;

            case 'l': // set local segment ID
                str = NULL;
                local_segment_id = strtoul(optarg, &str, 0);
                if (str == NULL || *str != '\0')
                {
                    log_error("Argument %s must be a valid segment ID", argv[optind-1]);
                    exit('l');
                }
                break;

            case 'r': // set remote segment ID
                str = NULL;
                remote_segment_id = strtoul(optarg, &str, 0);
                if (str == NULL || *str != '\0')
                {
                    log_error("Argument %s must be a valid segment ID", argv[optind-1]);
                    exit('r');
                }
                break;

            case 's': // set segment size
                str = NULL;
                local_segment_count = strtoul(optarg, &str, 0);
                if (str == NULL || *str != '\0' || local_segment_count == 0 || local_segment_count >= MAX_SIZE || local_segment_count * local_segment_factor >= MAX_SIZE)
                {
                    log_error("Argument %s must be a valid segment size in %s", argv[optind-1], local_segment_factor == 1e3 ? "MB" : "MiB");
                    exit('s');
                }
                break;

            case 'g': // set local GPU
                str = NULL;
                local_gpu_id = strtol(optarg, &str, 10);
                if (str == NULL || *str != '\0' || local_gpu_id < 0)
                {
                    log_error("Argument %s must be a valid GPU number", argv[optind-1]);
                    exit('g');
                }
                break;

            case 'b': // set benchmark mode
                mode = bench_mode_from_name(optarg);
                if (mode == BENCH_DO_NOTHING)
                {
                    log_error("Argument %s must be a valid benchmarking mode", argv[optind-1]);
                    exit('m');
                }
                break;

            case 'c': // set repeat count
                str = NULL;
                repeat_count = strtol(optarg, &str, 10);
                if (str == NULL || *str != '\0' || repeat_count <= 0)
                {
                    log_error("Argument %s must be at least 1", argv[optind-1]);
                    exit('c');
                }
                break;

            case 'v': // increase verbosity
                ++verbosity;
                break;

            case 'i': // use IEC units instead of SI
                log_debug("Using IEC units");
                local_segment_factor = 1 << 20;
                break;
        }
    }

    /* Sanity checking */
    if (remote_node_id == NO_NODE && local_segment_count == 0)
    {
        log_error("Either segment size or remote node ID must be specified");
        give_usage(argv[0]);
        exit(1);
    }
    if (remote_node_id != NO_NODE && mode == BENCH_DO_NOTHING)
    {
        log_error("No benchmarking operation is specified");
        give_usage(argv[0]);
        exit(1);
    }

    /* Get number of CUDA enabled GPUs */
    if (local_gpu_id != NO_GPU)
    {
        int gpu_count;
        if (cudaGetDeviceCount(&gpu_count) != cudaSuccess)
        {
            log_warn("Failed to initialize CUDA, setting GPU will not work");
            gpu_count = 0;
        }

        if (local_gpu_id >= gpu_count)
        {
            log_error("Invalid local GPU selected");
            exit(1);
        }
    }

    /* Initialize SISCI API */
    sci_error_t err = SCI_ERR_OK;
    SCIInitialize(0, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("%s", SCIGetErrorString(err));
        exit(1);
    }

    local_node_id = get_local_node_id(local_adapter);
    log_debug("Local node ID %u (adapter %u)", local_node_id, local_adapter);

    if (local_gpu_id != NO_GPU)
    {
        log_info("GPU segment is specified");
    }

    // Run as client or server
    if (remote_node_id == NO_NODE)
    {
        if (local_segment_count >= MAX_SIZE || local_segment_factor * local_segment_count >= MAX_SIZE)
        {
            log_error("Segment size is too large");
            exit(1);
        }
        log_info("Segment size is set to %lu %s", local_segment_factor * local_segment_count, local_segment_factor == 1e6 ? "MB" : "MiB");

        if (mode != BENCH_DO_NOTHING)
        {
            log_warn("Setting benchmarking operation has no effect in server mode");
        }

        if (repeat_count != 1)
        {
            log_warn("Setting repeat count has no effect in server mode");
        }

        if (local_segment_id == NO_ID)
        {
            log_warn("No local segment ID specified, using node ID...");
            local_segment_id = local_node_id;
        }
        log_info("Connect to node %u with segment ID %u", local_node_id, local_segment_id);

        signal(SIGINT, (sig_t) &stop_server);
        signal(SIGTERM, (sig_t) &stop_server);
        server(local_adapter, local_gpu_id, local_segment_id, local_segment_count * local_segment_factor);
    }
    else
    {
        if (local_segment_count != 0)
        {
            log_warn("Setting segment size has no effect in client mode");
        }

        if (local_segment_id == NO_ID)
        {
            log_warn("No local segment ID specified, using node ID...");
            local_segment_id = local_node_id;
        }
        if (remote_segment_id == NO_ID)
        {
            log_warn("No remote segment ID specified using node ID...");
            remote_segment_id = remote_node_id;
        }

        log_info("Initializing transfer list...");

        translist_t ts;
        if (translist_create(&ts, local_adapter, local_segment_id, remote_node_id, remote_segment_id, local_gpu_id) != 0)
        {
            log_error("Unexpected error when creating transfer list, aborting...");
            exit(1);
        }

        translist_desc_t tsd = translist_descriptor(ts);
        log_info("Remote segment size %lu %s", tsd.segment_size, local_segment_factor == 1e6 ? "MB" : "MiB");

        // TODO: Make client accept many --size arguments and server only one (big)
        // --transfer=size:offset
        translist_insert(ts, 0, 0, tsd.segment_size);

        client(mode, ts, repeat_count, local_segment_factor != 1e6);

        translist_delete(ts);
    }

    SCITerminate();
    exit(0);
}
