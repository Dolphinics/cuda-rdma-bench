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
    { .name = "type", .has_arg = 1, .flag = NULL, .val = 't' },
    { .name = "bench", .has_arg = 1, .flag = NULL, .val = 't' },
    { .name = "benchmark", .has_arg = 1, .flag = NULL, .val = 't' },
    { .name = "test", .has_arg = 1, .flag = NULL, .val = 't' },
    { .name = "count", .has_arg = 1, .flag = NULL, .val = 'c' },
    { .name = "verbose", .has_arg = 0, .flag = NULL, .val = 'v' },
    { .name = "iec", .has_arg = 0, .flag = NULL, .val = 'i' },
    { .name = "vec", .has_arg = 1, .flag = NULL, .val = 'V'},
    { .name = "len", .has_arg = 1, .flag = NULL, .val = 'L' },
    { .name = "help", .has_arg = 0, .flag = NULL, .val = 'h' },
    { .name = NULL, .has_arg = 0, .flag = NULL, .val = 0 }
};


/* List supported benchmark types */
static void list_bench_modes()
{
    fprintf(stderr, "Benchmark types\n");
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
    // TODO rewrite this to use gpu_info instead and extend gpu_info_t
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
                size_factor == 1e3 ? "kB" : "KiB");

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
            "   or: %s --type=<benchmark type> --remote-node=<node id>\n"
            "\nDescription\n"
            "    Benchmark how long it takes to transfer memory between a local and a\n"
            "    remote segment across an NTB link.\n"
            "\nServer mode arguments\n"
            "  --size=<size>            memory size in kB (or KiB if --iec is set)\n"
            "\nClient mode arguments\n"
            "  --type=<bencmark type>   specify benchmark type\n"
            "  --remote-node=<node id>  remote node identifier\n"
            "  --remote-id=<segment id> number identifying the remote segment\n"
            "  --count=<number>         number of times to repeat test (defaults to 1)\n"
            "\nDMA vector options (client mode)\n"
            "  --vec=<number>           divide segment into a number of DMA vector elements (defaults to 1)\n"
            "  --len=<number>           repeat the entire vector a number of times (defaults to 1)\n"
            "\nOptional arguments (both client and server mode)\n"
            "  --adapter=<adapter no>   local host adapter card number (defaults to 0)\n"
            "  --local-id=<segment id>  number identifying the local segment\n"
            "  --gpu=<gpu id>           specify a local GPU (if not given, buffer is allocated in RAM)\n"
            "  --verbose                increase verbosity level\n"
            "  --iec                    use IEC units (1024) instead of SI units (1000)\n"
            "  --help                   show list of local GPUs and benchmark types\n"
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
    size_t local_segment_factor = 1e3;

    size_t vec_div = 1;
    size_t vec_len = 1;

    size_t repeat_count = 1;
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

    while ((opt = getopt_long(argc, argv, "-:a:n:l:r:s:g:m:c:viV:L:h", options, &idx)) != -1)
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
                    log_error("Argument --adapter must be a valid adapter number");
                    exit('a');
                }
                break;

            case 'n': // set remote node
                str = NULL;
                remote_node_id = strtoul(optarg, &str, 10);
                if (str == NULL || *str != '\0' || remote_node_id == NO_NODE)
                {
                    log_error("Argument --remote-node must be a valid node identifier");
                    exit('n');
                }
                break;

            case 'l': // set local segment ID
                str = NULL;
                local_segment_id = strtoul(optarg, &str, 0);
                if (str == NULL || *str != '\0')
                {
                    log_error("Argument --local-id must be a valid segment identifier");
                    exit('l');
                }
                break;

            case 'r': // set remote segment ID
                str = NULL;
                remote_segment_id = strtoul(optarg, &str, 0);
                if (str == NULL || *str != '\0')
                {
                    log_error("Argument --remote-id must be a valid segment identifier");
                    exit('r');
                }
                break;

            case 's': // set segment size
                str = NULL;
                local_segment_count = strtoul(optarg, &str, 0);
                if (str == NULL || *str != '\0' || local_segment_count == 0)
                {
                    log_error("Argument --size must be a valid segment size in %s", local_segment_factor == 1e3 ? "kB" : "KiB");
                    exit('s');
                }
                break;

            case 'g': // set local GPU
                str = NULL;
                local_gpu_id = strtol(optarg, &str, 10);
                if (str == NULL || *str != '\0' || local_gpu_id < 0)
                {
                    log_error("Argument --gpu must be a valid GPU number");
                    exit('g');
                }
                break;

            case 't': // set benchmark type
                mode = bench_mode_from_name(optarg);
                if (mode == BENCH_DO_NOTHING)
                {
                    log_error("Argument --type must be a valid benchmark type, see --help for a list of valid types");
                    exit('m');
                }
                break;

            case 'c': // set repeat count
                str = NULL;
                repeat_count = strtoul(optarg, &str, 10);
                if (str == NULL || *str != '\0' || repeat_count == 0)
                {
                    log_error("Argument --count must be at least 1");
                    exit('c');
                }
                break;

            case 'v': // increase verbosity
                ++verbosity;
                break;

            case 'i': // use IEC units instead of SI
                log_debug("Using IEC units");
                local_segment_factor = 1 << 10;
                break;

            case 'V': // set DMA vector element length
                str = NULL;
                vec_div = strtoul(optarg, &str, 0);
                if (str == NULL || *str != '\0' || vec_div == 0)
                {
                    log_error("Argument --vec must be at least 1");
                    exit('V');
                }
                break;

            case 'L': // set DMA vector length
                str = NULL;
                vec_len = strtoul(optarg, &str, 10);
                if (str == NULL || *str != '\0' || vec_len == 0)
                {
                    log_error("Argument --len must be at least 1");
                    exit('L');
                }
                break;
        }
    }

    /* Sanity checking */
    if (remote_node_id == NO_NODE && local_segment_count == 0)
    {
        log_error("Either segment size or remote node identifier must be specified");
        give_usage(argv[0]);
        exit(1);
    }
    if (remote_node_id != NO_NODE && mode == BENCH_DO_NOTHING)
    {
        log_error("No benchmark type is specified");
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

    /* Run as client or server */
    if (remote_node_id == NO_NODE)
    {
        log_info("Segment size is set to %lu %s", local_segment_count, local_segment_factor == 1e3 ? "kB" : "KiB");

        if (mode != BENCH_DO_NOTHING)
        {
            log_warn("Setting benchmark type has no effect in server mode");
        }

        if (repeat_count != 1)
        {
            log_warn("Setting benchmark repeat count has no effect in server mode");
        }

        if (vec_len != 1 || vec_div != 1)
        {
            log_warn("DMA vector options have no effect in server mode");
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

        if (mode == BENCH_SCIMEMWRITE_TO_REMOTE && local_gpu_id != NO_GPU)
        {
            log_error("%s specified, but GPU buffer is selected", bench_mode_name(mode));
            SCITerminate();
            exit(1);
        }

        if ((vec_div != 1 || vec_len != 1) && !BENCH_IS_DMA(mode))
        {
            log_warn("DMA vector options have no effect when benchmark type is not DMA");
            vec_div = vec_len = 1;
        }

        log_info("Initializing transfer list...");

        translist_t ts;
        if (translist_create(&ts, local_adapter, local_segment_id, remote_node_id, remote_segment_id, local_gpu_id) != 0)
        {
            log_error("Unexpected error when creating transfer list, aborting...");
            SCITerminate();
            exit(1);
        }

        translist_desc_t tsd = translist_desc(ts);
        size_t segment_size = tsd.segment_size;
        log_info("Remote segment size %.2f %s", segment_size / (double) local_segment_factor, local_segment_factor == 1e3 ? "kB" : "KiB");

        if (vec_div >= segment_size || segment_size / vec_div == 0)
        {
            log_error("Number of DMA vector entries is larger than segment size");
            translist_delete(ts);
            SCITerminate();
            exit(1);
        }
    
        /* Fill transfer list */
        size_t entry_size = segment_size / vec_div;

        for (size_t k = 0; k < vec_len; ++k)
        {
            for (size_t v = 0; v < vec_div; ++v)
            {
                if (translist_insert(ts, v * entry_size, v * entry_size, entry_size) != 0)
                {
                    log_error("Failed to create transfer list");
                    break;
                }
            }
        }

        /* Create and run benchmark */
        bench_t bench_conf = {
            .benchmark_mode = mode,
            .num_runs = repeat_count,
            .transfer_list = ts
        };

        result_t* result = (result_t*) malloc(sizeof(result_t) + sizeof(uint64_t) * repeat_count);
        if (result == NULL)
        {
            log_error("Out of resources");
            translist_delete(ts);
            SCITerminate();
            exit(1);
        }

        if (client(local_adapter, &bench_conf, result) == 0)
        {
            log_info("Total runtime is %.2f s", result->total_runtime / 1e6l);
            log_info("Avg bandwidth is %.2f %-5s", (double) (result->total_size * repeat_count) / (double) result->total_runtime, local_segment_factor == 1e3 ? "MB/s" : "MiB/s");
            report_summary(stdout, &bench_conf, result, local_segment_factor != 1e3);
            report_bandwidth(stdout, &bench_conf, result, local_segment_factor != 1e3);
        }
        else
        {
            log_warn("Benchmark failed, skipping results");
        }

        free(result);
        translist_delete(ts);
    }

    SCITerminate();
    exit(0);
}
