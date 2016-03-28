#include <cuda.h>
#include <getopt.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <vector>
#include <exception>
#include <stdexcept>
#include "devbuf.h"
#include "hostbuf.h"
#include "bench.h"

using namespace std;


// Number of available CUDA devices
static int deviceCount = 0;

// What devices to use for the bandwidth test
static vector<int> devices;

// Different host buffers to use for the bandwidth test
static vector<HostBuffer> buffers;

// Different copy modes to use for the bandwidth test
static vector<cudaMemcpyKind> modes;


static void showUsage(const char* fname)
{
    fprintf(stderr, "Usage: %s --device=<gpu>... --size=<size>...\n" 
            "\nDescription\n"
            "    As the CUDA samples bandwidthTest might not be able to fully utilize a PCIe,\n"
            "    this programs starts multiple cudaMemcpyAsync transfers in order to measure\n"
            "    the maximum bandwidth.\n"
            "\nArguments\n"
            "  --size=<size>    transfer size in bytes\n"
            "  --device=<gpu>   specify GPU to use for transfer\n"
            "\nOptional arguments\n"
            "  --dtoh           specify device to host transfer (GPU to RAM)\n"
            "  --htod           specify host to device transfer (RAM to GPU)\n" 
            "  --pinned         page-lock host memory and map into CUDA address space\n" 
            "  --wc             allocate write-combined host memory\n" 
            "  --malloc         use system memory (malloc) instead of cudaHostAlloc\n"
            "  --verify         verify transfer by checking data\n"
            "  --list           list available CUDA GPUs\n"
            "  --help           show this help\n"
            "\nNOTE: Each argument can be given multiple times to test different transfers\n",
            fname);
}


static void listDevices()
{
    cudaError_t err;

    fprintf(stderr, "Available devices\n");
    for (int i = 0; i < deviceCount; ++i)
    {
        cudaDeviceProp prop;

        err = cudaGetDeviceProperties(&prop, i);
        if (err != cudaSuccess)
        {
            throw runtime_error(cudaGetErrorString(err));
        }

        // FIXME: Check compute compatibility

        fprintf(stderr, "  %2d %-25s %02x:%02x.%-3x\n",
                i, prop.name, prop.pciBusID, prop.pciDomainID, prop.pciDeviceID);
    }
    fprintf(stderr, "\n");
}


static void parseArguments(int argc, char** argv)
{
    vector<size_t> sizes;
    unsigned int flags = cudaHostAllocDefault;
    bool useSystemMemory = false;

    // Define program arguments
    option opts[] = {
        { .name = "device", .has_arg = 1, .flag = NULL, .val = 'd' },
        { .name = "size", .has_arg = 1, .flag = NULL, .val = 's' },
        { .name = "dtoh", .has_arg = 0, .flag = NULL, .val = cudaMemcpyDeviceToHost },
        { .name = "htod", .has_arg = 0, .flag = NULL, .val = cudaMemcpyHostToDevice },
        { .name = "mapped", .has_arg = 0, .flag = NULL, .val = 'p' },
        { .name = "pinned", .has_arg = 0, .flag = NULL, .val = 'p' },
        { .name = "write-combined", .has_arg = 0, .flag = NULL, .val = 'c' },
        { .name = "wc", .has_arg = 0, .flag = NULL, .val = 'c' },
        { .name = "malloc", .has_arg = 0, .flag = NULL, .val = 'm' },
        { .name = "verify", .has_arg = 0, .flag = NULL, .val = 'v' },
        { .name = "list", .has_arg = 0, .flag = NULL, .val = 'l' },
        { .name = "help", .has_arg = 0, .flag = NULL, .val = 'h' },
        { .name = NULL, .has_arg = 0, .flag = NULL, .val = 0 }
    };

    // Parse arguments
    int opt, idx;
    while ((opt = getopt_long(argc, argv, "-:d:s:vlhmpwc", opts, &idx)) != -1)
    {
        switch (opt)
        {
            case ':': // missing value
                fprintf(stderr, "Option %s requires a value\n", argv[optind-1]);
                throw 1;

            case '?': // unknown option
                fprintf(stderr, "Unknown option: %s\n", argv[optind-1]);
                throw 1;
    
            case 'd': // append device to device list
                {
                    char* str = NULL;
                    int device = strtol(optarg, &str, 10);
                    if (str == NULL || *str != '\0' || device < 0 || device >= deviceCount)
                    {
                        throw "Argument --device must be a valid GPU";
                    }
                    devices.push_back(device);
                }
                break;

            case 's': // append transfer size to size list
                {
                    char* str = NULL;
                    size_t size = strtoull(optarg, &str, 0);
                    if (str == NULL || *str != '\0' || size == 0)
                    {
                        throw "Argument --size must be a valid byte count";
                    }
                    sizes.push_back(size);
                }
                break;

            case cudaMemcpyDeviceToHost: // device to host
                modes.push_back(cudaMemcpyDeviceToHost);
                break;

            case cudaMemcpyHostToDevice: // host to device
                modes.push_back(cudaMemcpyHostToDevice);
                break;

            case 'p': // pinned and mapped memory
                flags |= cudaHostAllocMapped;
                break;

            case 'c': // write combined memory
            case 'w':
                flags |= cudaHostAllocWriteCombined;
                break;

            case 'm': // use malloc instead of 
                useSystemMemory = true;
                break;

            case 'v': // verify transfer
                throw "Verify transfer is not implemented\n";

            case 'l': // list devices
                listDevices();
                throw 0;

            case 'h': // show help
                showUsage(argv[0]);
                throw 0;
        }
    }

    if (modes.empty())
    {
        modes.push_back(cudaMemcpyHostToDevice);
        modes.push_back(cudaMemcpyDeviceToHost);
    }

    if (sizes.empty())
    {
        fprintf(stderr, "NOTE: No size argument given, using default size 32 MiB\n");
        sizes.push_back(32 << 20);
    }

    if (devices.empty())
    {
        fprintf(stderr, "NOTE: No devices specified, using all devices\n");
        for (int i = 0; i < deviceCount; ++i)
        {
            devices.push_back(i);
        }
    }

    if (flags != cudaHostAllocDefault && useSystemMemory)
    {
        fprintf(stderr, "NOTE: System memory allocation (malloc) does not support different memory types\n");
    }

    // Create host buffers
    for (vector<size_t>::const_iterator sizeIt = sizes.begin(); sizeIt != sizes.end(); ++sizeIt)
    {
        const size_t size = *sizeIt;

        if (useSystemMemory)
        {
            buffers.push_back(HostBuffer(size));
        }
        else
        {
            buffers.push_back(HostBuffer(size, flags));
        }
    }
}



int main(int argc, char** argv)
{
    // Find maximum GPU count
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Unexpected error: %s\n", cudaGetErrorString(err));
        return 'd';
    }

    // Parse program arguments
    try 
    {
        parseArguments(argc, argv);
    }
    catch (const runtime_error& e)
    {
        fprintf(stderr, "Unexpected error: %s\n", e.what());
        return 1;
    }
    catch (const int e) // FIXME: Hack
    {
        return e;
    }
    catch (const char* e) // FIXME: Hack
    {
        fprintf(stderr, "%s\n", e);
        return 1;
    }

    // Run bandwidth benchmark
    try
    {
        benchmark(buffers, devices, modes);
    }
    catch (const runtime_error& e)
    {
        fprintf(stderr, "Unexpected error: %s\n", e.what());
        return 1;
    }

    // Reset devices to supress warnings from cuda-memcheck
    for (vector<int>::const_iterator deviceIt = devices.begin(); deviceIt != devices.end(); ++deviceIt)
    {
        cudaSetDevice(*deviceIt);
        cudaDeviceReset();
    }

    return 0;
}
