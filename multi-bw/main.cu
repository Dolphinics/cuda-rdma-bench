#include <cuda.h>
#include <getopt.h>
#include <string>
#include <cstdlib>
#include <cstring>
#include <strings.h>
#include <cstdio>
#include <vector>
#include <exception>
#include <stdexcept>
#include "devbuf.h"
#include "hostbuf.h"
#include "bench.h"
#include "event.h"

using namespace std;


static void showUsage(const char* fname)
{
    fprintf(stderr, "Usage: %s --transfer=<transfer specs>... [--streams=<mode>] [--list] [--help]\n" 
            "\nDescription\n"
            "    As the CUDA samples bandwidthTest might not be able to fully utilize the bus,\n"
            "    this programs starts multiple cudaMemcpyAsync transfers using multiple streams\n"
            "    in order to measure the maximum bandwidth.\n"
            "\nProgram options\n"
            "  --streams=<mode>      stream modes for transfers\n"
            "  --list               list available CUDA devices and quit\n"
            "  --help               show this help text and quit\n"
            "\nStream modes\n"
            "  per-transfer         one stream per transfer (default)\n"
            "  per-device           transfers to the same device share streams\n"
            "  only-one             all transfers share the same single stream\n"
            "\nTransfer specification format\n"
            "    <device>[:<direction>][:<size>][:<memory options>...]\n"
            "\nTransfer specification arguments\n"
            "  <device>             CUDA device to use for transfer\n"
            "  <direction>          transfer directions\n"
            "  <size>               transfer size in bytes (default is 32 MiB)\n"
            "  <memory options>     memory allocation options\n"
            "\nTransfer directions\n"
            "  HtoD                 host to device transfer (RAM to GPU)\n"
            "  DtoH                 device to host transfer (GPU to RAM)\n"
            "  both                 first HtoD then DtoH (default)\n"
            "  reverse              first DtoH then HtoD\n"
            "\nMemory options format\n"
            "   opt1,opt2,opt3,...\n"
            "\nMemory options\n"
            "  mapped               map host memory into CUDA address space\n"
            "  managed              allocate managed memory on the device\n"
            "  wc                   allocate write-combined memory on the host\n"
            "\nExample runs\n"
            "  %s --run=all\n"
            "  %s --run=0:both:1048576\n"
            "  %s --run=0:DtoH:1024:wc,mapped --run=1:HtoD:4096 --run=0:DtoH:128:wc --streams=per-device\n"
            ,
            fname, fname, fname, fname
           );
}


static void listDevices()
{
    cudaError_t err;

    int deviceCount = 0;
    err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        throw runtime_error(cudaGetErrorString(err));
    }

    fprintf(stderr, "\n%3s   %-20s   %-9s   %-8s   %-8s   %-8s   %3s\n",
            "ID", "Device name", "IO addr", "Managed", "Unified", "Mappable", "#");
    fprintf(stderr, "-----------------------------------------------------------------------------\n");
    for (int i = 0; i < deviceCount; ++i)
    {
        cudaDeviceProp prop;

        err = cudaGetDeviceProperties(&prop, i);
        if (err != cudaSuccess)
        {
            throw runtime_error(cudaGetErrorString(err));
        }

        if (prop.computeMode == cudaComputeModeProhibited)
        {
            continue;
        }

        fprintf(stderr, "%3d   %-20s   %02x:%02x.%-3x   %8s   %8s   %8s   %3d\n",
                i, prop.name, prop.pciBusID, prop.pciDeviceID, prop.pciDomainID,
                prop.managedMemory ? "yes" : "no", 
                prop.unifiedAddressing ? "yes" : "no",
                prop.canMapHostMemory ? "yes" : "no",
                prop.asyncEngineCount);
    }
    fprintf(stderr, "\n");
}


static bool isValidDevice(int device)
{
    cudaDeviceProp prop;

    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess)
    {
        return false;
    }

    if (prop.computeMode == cudaComputeModeProhibited)
    {
        return false;
    }

    return true;
}


static void parseTransferSpecification(vector<TransferSpec>& transferSpecs, char* specStr)
{
    const char* delim = ":,";
    int device = -1;
    size_t size = 0;
    unsigned int hostAllocFlags = cudaHostAllocDefault;
    unsigned int deviceAllocFlags = 0;
    bool useManagedDeviceMem = false;
    vector<cudaMemcpyKind> directions;

    // First token must be device
    char* strptr = NULL;
    char* token = strtok(specStr, delim);

    device = strtol(token, &strptr, 10);
    if (strptr == NULL || *strptr != '\0' || !isValidDevice(device))
    {
        fprintf(stderr, "Invalid transfer specification: '%s' is not a valid device\n", token);
        throw 3;
    }

    // The remaining of the transfer specification may be in arbitrary order
    // because we want to be nice
    while ((token = strtok(NULL, delim)) != NULL)
    {
        if (strcasecmp("dtoh", token) == 0 && directions.empty())
        {
            directions.push_back(cudaMemcpyDeviceToHost);
        }
        else if (strcasecmp("htod", token) == 0 && directions.empty())
        {
            directions.push_back(cudaMemcpyHostToDevice);
        }
        else if (strcasecmp("both", token) == 0 && directions.empty())
        {
            directions.push_back(cudaMemcpyHostToDevice);
            directions.push_back(cudaMemcpyDeviceToHost);
        }
        else if (strcasecmp("reverse", token) == 0 && directions.empty())
        {
            directions.push_back(cudaMemcpyDeviceToHost);
            directions.push_back(cudaMemcpyHostToDevice);
        }
        else if (strcasecmp("mapped", token) == 0)
        {
            hostAllocFlags |= cudaHostAllocMapped;
        }
        else if (strcasecmp("write-combined", token) == 0 || strcasecmp("wc", token) == 0)
        {
            hostAllocFlags |= cudaHostAllocWriteCombined;
        }
        else if (strcasecmp("managed", token) == 0)
        {
            useManagedDeviceMem = true;
        }
        else if (size == 0)
        {
            strptr = NULL;
            size = strtoul(token, &strptr, 0);
            if (strptr == NULL || *strptr != '\0')
            {
                size = 0;
            }
        }
    }

    // Insert default values if necessary
    if (directions.empty())
    {
        directions.push_back(cudaMemcpyDeviceToHost);
        directions.push_back(cudaMemcpyHostToDevice);
    }
    if (size == 0)
    {
        size = 32 << 20;
    }

    // Try to allocate buffers and create transfer specification
    for (cudaMemcpyKind transferMode : directions)
    {
        TransferSpec spec;
        spec.deviceBuffer = DeviceBufferPtr(new DeviceBuffer(device, size)); // FIXME: Managed memory
        spec.hostBuffer = HostBufferPtr(new HostBuffer(size, hostAllocFlags));
        spec.direction = transferMode;

        transferSpecs.push_back(spec);
    }
}


static void parseArguments(int argc, char** argv, StreamSharingMode& streamMode, vector<TransferSpec>& transferSpecs)
{
    // Define program arguments
    const option opts[] = {
        { .name = "transfer", .has_arg = 1, .flag = NULL, .val = 't' },
        { .name = "streams", .has_arg = 1, .flag = NULL, .val = 's' },
        { .name = "list", .has_arg = 0, .flag = NULL, .val = 'l' },
        { .name = "help", .has_arg = 0, .flag = NULL, .val = 'h' },
        { .name = NULL, .has_arg = 0, .flag = NULL, .val = 0 }
    };

    // Parse arguments
    int opt, idx;
    while ((opt = getopt_long(argc, argv, "-:t:s:lh", opts, &idx)) != -1)
    {
        switch (opt)
        {
            case ':': // missing value
                fprintf(stderr, "Option %s requires a value\n", argv[optind-1]);
                throw 1;

            case '?': // unknown option
                fprintf(stderr, "Unknown option: %s\n", argv[optind-1]);
                throw 1;

            case 't': // transfer specification
                parseTransferSpecification(transferSpecs, optarg);
                break;

            case 's': // stream sharing mode
                if (strcasecmp("per-transfer", optarg) == 0)
                {
                    streamMode = perTransfer;
                }
                else if (strcasecmp("per-device", optarg) == 0 || strcasecmp("per-gpu", optarg) == 0)
                {
                    streamMode = perDevice;
                }
                else if (strcasecmp("only-one", optarg) == 0 || strcasecmp("single", optarg) == 0)
                {
                    streamMode = singleStream;
                }
                else
                {
                    fprintf(stderr, "Unknown stream mode: %s\n", optarg);
                    throw 2;
                }
                break;

            case 'l': // list devices
                listDevices();
                throw 0;

            case 'h': // show help
                showUsage(argv[0]);
                throw 0;
        }
    }
}



int main(int argc, char** argv)
{
    // Parse program arguments
    StreamSharingMode streamMode = perTransfer;
    vector<TransferSpec> transferSpecs;

    try 
    {
        parseArguments(argc, argv, streamMode, transferSpecs);
    }
    catch (const runtime_error& e)
    {
        fprintf(stderr, "Unexpected error: %s\n", e.what());
        return 1;
    }
    catch (const int e) 
    {
        return e;
    }

    try
    {
        // Check number of available GPUs
        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        if (err != cudaSuccess)
        {
            throw runtime_error(cudaGetErrorString(err));
        }

        if (deviceCount == 0)
        {
            fprintf(stderr, "No CUDA capable devices found!\n");
            return 1;
        }

        // No transfer specifications?
        if (transferSpecs.empty())
        {
            for (int device = 0; device < deviceCount; ++device)
            {
                if (isValidDevice(device))
                {
                    char buffer[64];
                    snprintf(buffer, sizeof(buffer), "%d", device);

                    parseTransferSpecification(transferSpecs, buffer);
                }
            }
        }

        // Create streams and timing events
        for (TransferSpec& spec : transferSpecs)
        {
            spec.cudaStream = retrieveStream(spec.deviceBuffer->device, streamMode);
            spec.cudaEvents = createTimingData();
        }

        // Run bandwidth test
        runBandwidthTest(transferSpecs);
    }
    catch (const runtime_error& e)
    {
        fprintf(stderr, "Unexpected error: %s\n", e.what());
        return 1;
    }

    return 0;
}
