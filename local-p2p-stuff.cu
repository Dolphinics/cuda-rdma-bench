#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

#define cudaCheckError()                                \
    do {                                                \
        cudaError_t __err__ = cudaGetLastError();       \
        if (__err__ != cudaSuccess) {                   \
            fprintf(stderr, "CUDA error %s:%d: %s\n",  \
                __FILE__, __LINE__,                     \
                cudaGetErrorString(__err__));           \
            exit(1);                                    \
        }                                               \
    } while (0)

#define STRINGIFY(s) #s

static size_t factor = 1000L;

void EnableP2P(int dev, int peer)
{
    cudaSetDevice(dev);
    
    int access = 0;
    cudaDeviceCanAccessPeer(&access, dev, peer);
    cudaCheckError();

    if (access)
    {
        cudaDeviceEnablePeerAccess(peer, 0);
        cudaCheckError();
    }
}

void DisableP2P(int dev, int peer)
{
    cudaSetDevice(dev);

    int access = 0;
    cudaDeviceCanAccessPeer(&access, dev, peer);
    cudaCheckError();

    if (access)
    {
        cudaDeviceDisablePeerAccess(peer);
        cudaCheckError();
    }
}

void ConfigureP2P(int ctl, int src, int dst, int bidirect, int useP2P)
{
    int dev1, dev2;

    if (ctl == src)
    {
        dev1 = src;
        dev2 = dst;
    }
    else
    {
        dev1 = dst;
        dev2 = src;
    }

    if (useP2P)
    {
        EnableP2P(dev1, dev2);
        if (bidirect)
        {
            EnableP2P(dev2, dev1);
        }
    }
    else
    {
        DisableP2P(dev1, dev2);
        if (bidirect)
        {
            DisableP2P(dev2, dev1);
        }
    }
}


void AllocHostAndDevBufs(int dev, uint8_t** devptr, uint8_t** hostptr, size_t size, unsigned type, cudaStream_t* stream)
{
    cudaSetDevice(dev);
    cudaCheckError();

    cudaHostAlloc((void**) hostptr, size, type);
    cudaCheckError();

    if (!!(type & cudaHostAllocMapped))
    {
        cudaHostGetDevicePointer((void**) devptr, (void*) *hostptr, 0);
        cudaCheckError();
    }
    else
    {
        cudaMalloc((void**) devptr, size);
        cudaCheckError();
    }

    cudaStreamCreate(stream);
    cudaCheckError();
}

void MeasureBandwidth(int ctlDev, int srcDev, int dstDev, size_t memSize, int bidirectional, int p2p, unsigned memType, int repeat, int verify)
{
    float time_ms;
    double time_s, gigabytes;

    uint8_t *srcBuf, *srcPtr;
    cudaStream_t srcStream;
    AllocHostAndDevBufs(srcDev, &srcPtr, &srcBuf, memSize, memType, &srcStream);

    uint8_t *dstBuf, *dstPtr;
    cudaStream_t dstStream;
    AllocHostAndDevBufs(dstDev, &dstPtr, &dstBuf, memSize, memType, &dstStream);

    ConfigureP2P(ctlDev, srcDev, dstDev, bidirectional, p2p);

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaCheckError();
    cudaEventCreate(&stop);
    cudaCheckError();

    if (verify)
    {
        for (size_t i = 0; i < memSize; ++i)
        {
            srcBuf[i] = rand() & 255;
        }

        if (!(memType & cudaHostAllocMapped))
        {
            cudaSetDevice(srcDev);
            cudaDeviceSynchronize();
            cudaCheckError();

            cudaEventRecord(start);
            for (int i = 0; i < repeat; ++i)
            {
                cudaMemcpyAsync(srcPtr, srcBuf, memSize, cudaMemcpyHostToDevice);
            }
            cudaEventRecord(stop);

            cudaDeviceSynchronize();
            cudaCheckError();

            cudaEventElapsedTime(&time_ms, start, stop);
            time_s = time_ms / (double) 1e3;
            gigabytes = (memSize * repeat) / (double) (factor * factor * factor);

            printf("Host to device  : %6.02f %s\n", gigabytes / time_s, factor == 1024L ? "GiB/s" : "GB/s");
        }
        else
        {
            printf("Host to device  : memory is mapped\n");
        }
    }

    cudaSetDevice(ctlDev);
    cudaDeviceSynchronize();
    cudaCheckError();
    cudaEventRecord(start);

    // cudaMemcpyPeerAsync will fall back to cudaMemcpyAsync when p2p is disabled
    for (int i = 0; i < repeat; ++i)
    {
        cudaMemcpyPeerAsync(dstPtr, dstDev, srcPtr, srcDev, memSize, dstStream);
        if (bidirectional)
        {
            cudaMemcpyPeerAsync(srcPtr, srcDev, dstPtr, dstDev, memSize, srcStream);
        }
    }

    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaCheckError();

    cudaEventElapsedTime(&time_ms, start, stop);
    time_s = time_ms / (double) 1e3;
    gigabytes = (memSize * repeat) / (double) (factor * factor * factor);

    printf("Device to device: %6.02f %s\n", gigabytes / time_s, factor == 1024L ? "GiB/s" : "GB/s");

    if (verify)
    {
        if (!(memType & cudaHostAllocMapped))
        {
            cudaSetDevice(srcDev);
            cudaDeviceSynchronize();
            cudaCheckError();

            cudaEventRecord(start);
            for (int i = 0; i < repeat; ++i)
            {
                cudaMemcpyAsync(dstBuf, srcPtr, memSize, cudaMemcpyDeviceToHost);
            }
            cudaEventRecord(stop);

            cudaDeviceSynchronize();
            cudaCheckError();

            cudaEventElapsedTime(&time_ms, start, stop);
            time_s = time_ms / (double) 1e3;
            gigabytes = (memSize * repeat) / (double) (factor * factor * factor);

            printf("Device to host  : %6.02f %s\n", gigabytes / time_s, factor == 1024L ? "GiB/s" : "GB/s");
        }
        else
        {
            printf("Device to host  : memory is mapped\n");
        }

        size_t i;
        for (i = 0; i < memSize && srcBuf[i] == dstBuf[i]; ++i);

        if (i != memSize)
        {
            printf("\n ***** Data was NOT transfered properly! Byte %ld differs! *****\n", i);
        }
    }

    cudaFreeHost(srcBuf);
    cudaFreeHost(dstBuf);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaSetDevice(srcDev);
    cudaFree(srcPtr);
    cudaStreamDestroy(srcStream);

    cudaSetDevice(dstDev);
    cudaFree(dstPtr);
    cudaStreamDestroy(dstStream);
}

void ListDevices()
{
    int count;

    cudaGetDeviceCount(&count);
    cudaCheckError();

    int** peerAccessMatrix = (int**) malloc(sizeof(int*) * count);

    for (int i = 0; i < count; ++i)
    {
        cudaDeviceProp prop;

        cudaGetDeviceProperties(&prop, i);
        cudaCheckError();

        printf("%3d %-16s  %02x:%02x.%x\n",
            i, prop.name, prop.pciBusID, prop.pciDomainID, prop.pciDeviceID);

        peerAccessMatrix[i] = (int*) malloc(sizeof(int) * count);

        for (int j = 0; j < count; ++j)
        {
            if (i != j)
            {
                cudaSetDevice(i);
                cudaCheckError();

                cudaDeviceCanAccessPeer(&peerAccessMatrix[i][j], j, 0);
                cudaCheckError();
            }
        }
    }

    printf("\nP2P");
    for (int i = 0; i < count; ++i)
    {
        printf(" %3d", i);
    }
    printf("\n");

    for (int i = 0; i < count; ++i)
    {
        printf("%3d", i);
        for (int j = 0; j < count; ++j)
        {
            if (i == j)
            {
                printf("   -");
            }
            else if (peerAccessMatrix[i][j])
            {
                printf("   y");
            }
            else
            {
                printf("   n");
            }
        }
        printf("\n");

        free(peerAccessMatrix[i]);
    }
    free(peerAccessMatrix);
}

int main(int argc, char** argv)
{
    // Parameters
    size_t size = 0;
    int srcDevice = -1;
    int dstDevice = -1;
    int bidirectional = 0;
    int usePeer2Peer = 0;
    int oppositeDevice = 0;
    int verify = 0;
    int repeat = 5;
    unsigned memtype = cudaHostAllocDefault;

    // Get device count
    int devCount = 0;
    cudaError_t err;
    err = cudaGetDeviceCount(&devCount);
    switch (err)
    {
        case cudaErrorNoDevice:
            fprintf(stderr, "No CUDA capable device detected!\n");
            return 1;

        default:
            cudaCheckError();
            break;
    }

    // Parse command line options
    struct option opts[] = {
        { .name = "srcdev", .has_arg = 1, .flag = NULL, 1 },
        { .name = "dstdev", .has_arg = 1, .flag = NULL, 2 },
        { .name = "size", .has_arg = 1, .flag = NULL, 3 },
        { .name = "peer", .has_arg = 0, .flag = NULL, 4 },
        { .name = "help", .has_arg = 0, .flag = NULL, 'h' },
    };
    int opt, optidx;
    char* strptr;

    while ((opt = getopt_long(argc, argv, "-:hbpiolvmwkr:", opts, &optidx)) != -1)
    {
        switch (opt)
        {
            case ':': // missing value
                fprintf(stderr, "Argument %s requires an argument\n", argv[optind - 1]);
                goto giveUsage;

            case '?': // unknown flag
                fprintf(stderr, "Unknown option: -%c\n", optopt);
                goto giveUsage;

            case 1: // set source device
                strptr = NULL;
                srcDevice = strtoul(optarg, &strptr, 0);
                if (strptr == NULL || *strptr != '\0' || srcDevice >= devCount)
                {
                    fprintf(stderr, "Argument --srcdev requires a valid CUDA device number\n");
                    goto giveUsage;
                }
                else if (dstDevice == srcDevice)
                {
                    fprintf(stderr, "NOTE!! Source device is equal to destination device!\n");
                }
                break;

            case 2: // set destination device
                strptr = NULL;
                dstDevice = strtoul(optarg, &strptr, 0);
                if (strptr == NULL || *strptr != '\0' || dstDevice >= devCount)
                {
                    fprintf(stderr, "Argument --dstdev requires a valid CUDA device number\n");
                    goto giveUsage;
                }
                else if (dstDevice == srcDevice)
                {
                    fprintf(stderr, "NOTE!! Destination device is equal to source device!\n");
                }
                break;

            case 3: // set memory chunk size 
                strptr = NULL;
                size = strtoul(optarg, &strptr, 0);
                if (strptr == NULL || *strptr != '\0' || size == 0)
                {
                    fprintf(stderr, "Argument --size requires a valid memory size\n");
                    return 1;
                }
                break;

            case 4: // enable peer-to-peer
                usePeer2Peer = 1;
                break;

            case 'v': // verify transfer 
                verify = 1;
                break;

            case 'b': // bidirectional benchmark (if p2p is supported+enabled, this will show the difference)
                bidirectional = 1;
                break;

            case 'p':
                memtype |= cudaHostAllocPortable;
                break;

            case 'm':
                memtype |= cudaHostAllocMapped;
                break;

            case 'w':
                memtype |= cudaHostAllocWriteCombined;
                break;

            case 'o': // reverse cudaSetDevice
                oppositeDevice = 1;
                break;

            case 'r': // set number of times to repeat
                strptr = NULL;
                repeat = strtoul(optarg, &strptr, 0);
                if (strptr == NULL || *strptr != '\0' || repeat <= 0 || repeat > 1000)
                {
                    fprintf(stderr, "Option -r requires a valid number between 1 and 1000\n");
                    return 1;
                }
                break;

            case 'i': // use MiBs instead of MBs
                factor = 1024L;
                break;

            case 'l': // list CUDA enabled devices
                ListDevices();
                return 0;

            case 'h': // show help
                goto giveUsage;
        }
    }

    // Verify program arguments
    if (srcDevice < 0 || dstDevice < 0 || size == 0)
    {
        fprintf(stderr, "Missing required arguments!!\n");
        goto giveUsage;
    }

    // Calculate chunk size
    size = size * factor * factor;

    // If 
    if (!!(memtype & cudaHostAllocMapped))
    {
        cudaSetDevice(srcDevice);
        cudaSetDeviceFlags(cudaDeviceMapHost);
        cudaCheckError();
        cudaSetDevice(dstDevice);
        cudaSetDeviceFlags(cudaDeviceMapHost);
        cudaCheckError();
    }
    
    // Allocate host buffer
    MeasureBandwidth(
            oppositeDevice ? dstDevice : srcDevice, 
            srcDevice, 
            dstDevice, 
            size, 
            bidirectional, 
            usePeer2Peer,
            memtype,
            repeat,
            verify
    );

    return 0;

giveUsage:
    fprintf(stderr, 
            "Usage: %s --srcdev=<device no> --dstdev=<device no> --size=<size> [--peer] [options]\n"
            "\nArguments\n"
            "  --srcdev=<device no>  CUDA device to copy data from\n"
            "  --dstdev=<device no>  CUDA device to copy data to\n"
            "  --size=<size>         memory chunk size in MB (or MiB if -i is set)\n"
            "  --peer                enable p2p is possible\n"
            "\nOptions\n"
            "   -v                   verify transfer by copying memory from device and comparing\n"
            "   -b                   transfer memory in both directions simultaneously\n"
            "   -p                   use cudaHostAllocPortable flag\n"
            "   -m                   use cudaHostAllocMapped flag\n"
            "   -w                   use cudaHostAllocWriteCombined flag\n"
            "   -o                   make cudaSetDevice set opposite device (pull instead of push)\n"
            "   -r <number>          number of times to repeat (default is 5)\n"
            "   -i                   use IEC units (1024) instead of SI units (1000)\n"
            "   -l                   list CUDA devices and quit\n"
            "\nBuild date: %s %s\n"
            , argv[0], __DATE__, __TIME__);

    return 1;
}