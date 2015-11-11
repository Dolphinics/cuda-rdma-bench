#include <cuda.h>
#include <stdio.h>
#include <getopt.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>


#define check( status ) \
    do { \
        if ((status) != cudaSuccess) { \
            fprintf(stderr, "%s\n", cudaGetErrorString((status))); \
            exit((status)); \
        } \
    } while (0)

#define test( dev, testcase ) \
    do { \
        check( cudaDeviceSynchronize() ); \
        check( cudaSetDevice((dev)) ); \
        check( cudaDeviceSynchronize() ); \
        start = usecs(); \
        check( (testcase) ); \
        check( cudaDeviceSynchronize() ); \
        end = usecs(); \
    } while (0) 


static int verbosity = 0;


__host__ void GetDeviceInfo(int dev, cudaDeviceProp* prop)
{
    cudaError_t err = cudaGetDeviceProperties(prop, dev);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "%s\n", cudaGetErrorString(err));
        exit(1);
    }
}


__host__ void GetDeviceName(int dev, char* name, size_t len)
{
    if (dev < 0)
    {
        strncpy(name, "host", len);
        return;
    }

    cudaDeviceProp prop;
    GetDeviceInfo(dev, &prop);

    strncpy(name, prop.name, len);
}


__host__ void PrintMeasurement(const char* func_name, const char* mem_type, int src, int dst, size_t size, uint64_t usec)
{

    fprintf(stdout, "%-16s :: %-28s :: %2d -> %2d :: %12.3f MB/s",
            func_name,
            mem_type,
            src,
            dst,
            ((double) size) / ((double) usec)
            );

    if (verbosity > 0)
    {
        char src_name[256];
        char dst_name[256];

        GetDeviceName(src, src_name, sizeof(src_name));
        GetDeviceName(dst, dst_name, sizeof(dst_name));

        fprintf(stdout, " :: %lu MB copied in %lu µs from '%s' to '%s'", 
                size / 1000000L,
                usec, src_name, dst_name);
    }

    fprintf(stdout, "\n");
}


__host__ void* AllocDeviceMemory(size_t size, int device)
{
    cudaError_t err;
    void* ptr = NULL;

    err = cudaSetDevice(device);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "%s\n", cudaGetErrorString(err));
        exit(1);
    }

    err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "%s\n", cudaGetErrorString(err));
        exit(1);
    }

    return ptr;
}


__host__ void* AllocHostMemory(size_t size, unsigned int flags)
{
    cudaError_t err;
    void* ptr = NULL;

    err = cudaHostAlloc(&ptr, size, flags);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "%s\n", cudaGetErrorString(err));
        exit(1);
    }

    return ptr;
}


inline uint64_t usecs()
{
    struct timespec ts;
    
    if (clock_gettime(CLOCK_REALTIME, &ts) != 0)
    {
        fprintf(stderr, "Something is wrong...\n");
        exit(1);
    }

    return (ts.tv_sec * 1000000L) + (ts.tv_nsec / 1000);
}


__host__ void MeasureMemcpy(size_t size, int dev_src, int dev_dst)
{
    void* mem_dst = AllocDeviceMemory(size, dev_dst);
    void* mem_src = AllocDeviceMemory(size, dev_src);
    void* mem_host = NULL;

    uint64_t end, start;


    // HOST TO DEVICE
    fprintf(stdout, "\nHOST TO DEVICE\n");

    mem_host = malloc(size);
    test( dev_dst, cudaMemcpy(mem_dst, mem_host, size, cudaMemcpyHostToDevice) );
    free(mem_host);
    PrintMeasurement("cudaMemcpy", "malloc'd", -1, dev_dst, size, end - start);

    mem_host = AllocHostMemory(size, cudaHostAllocDefault);
    test( dev_dst, cudaMemcpy(mem_dst, mem_host, size, cudaMemcpyHostToDevice) );
    cudaFreeHost(mem_host);
    PrintMeasurement("cudaMemcpy", "cudaHostAllocDefault", -1, dev_dst, size, end - start);

    mem_host = AllocHostMemory(size, cudaHostAllocPortable);
    test( dev_dst, cudaMemcpy(mem_dst, mem_host, size, cudaMemcpyHostToDevice) );
    cudaFreeHost(mem_host);
    PrintMeasurement("cudaMemcpy", "cudaHostAllocPortable", -1, dev_dst, size, end - start);

    mem_host = AllocHostMemory(size, cudaHostAllocMapped);
    test( dev_dst, cudaMemcpy(mem_dst, mem_host, size, cudaMemcpyHostToDevice) );
    cudaFreeHost(mem_host);
    PrintMeasurement("cudaMemcpy", "cudaHostAllocMapped", -1, dev_dst, size, end - start);

    mem_host = AllocHostMemory(size, cudaHostAllocWriteCombined);
    test( dev_dst, cudaMemcpy(mem_dst, mem_host, size, cudaMemcpyHostToDevice) );
    cudaFreeHost(mem_host);
    PrintMeasurement("cudaMemcpy", "cudaHostAllocWriteCombined", -1, dev_dst, size, end - start);


    // DEVICE TO HOST
    fprintf(stdout, "\nDEVICE TO HOST\n");

    mem_host = malloc(size);
    test( dev_dst, cudaMemcpy(mem_host, mem_dst, size, cudaMemcpyDeviceToHost) );
    free(mem_host);
    PrintMeasurement("cudaMemcpy", "malloc'd", dev_dst, -1, size, end - start);

    mem_host = AllocHostMemory(size, cudaHostAllocDefault);
    test( dev_dst, cudaMemcpy(mem_host, mem_dst, size, cudaMemcpyDeviceToHost) );
    cudaFreeHost(mem_host);
    PrintMeasurement("cudaMemcpy", "cudaHostAllocDefault", dev_dst, -1, size, end - start);

    mem_host = AllocHostMemory(size, cudaHostAllocPortable);
    test( dev_dst, cudaMemcpy(mem_host, mem_dst, size, cudaMemcpyDeviceToHost) );
    cudaFreeHost(mem_host);
    PrintMeasurement("cudaMemcpy", "cudaHostAllocPortable", dev_dst, -1, size, end - start);

    mem_host = AllocHostMemory(size, cudaHostAllocMapped);
    test( dev_dst, cudaMemcpy(mem_host, mem_dst, size, cudaMemcpyDeviceToHost) );
    cudaFreeHost(mem_host);
    PrintMeasurement("cudaMemcpy", "cudaHostAllocMapped", dev_dst, -1, size, end - start);

    mem_host = AllocHostMemory(size, cudaHostAllocWriteCombined);
    test( dev_dst, cudaMemcpy(mem_host, mem_dst, size, cudaMemcpyDeviceToHost) );
    cudaFreeHost(mem_host);
    PrintMeasurement("cudaMemcpy", "cudaHostAllocWriteCombined", dev_dst, -1, size, end - start);


    // DEVICE TO DEVICE
    fprintf(stdout, "\nDEVICE TO DEVICE\n");

    test( dev_src, cudaMemcpy(mem_src, mem_dst, size, cudaMemcpyDeviceToDevice) );
    PrintMeasurement("cudaMemcpy", "cudaMalloc", dev_src, dev_dst, size, end - start);

    test( dev_src, cudaMemcpy(mem_dst, mem_src, size, cudaMemcpyDeviceToDevice) );
    PrintMeasurement("cudaMemcpy", "cudaMalloc", dev_dst, dev_src, size, end - start);
   
    test( dev_dst, cudaMemcpy(mem_src, mem_dst, size, cudaMemcpyDeviceToDevice) );
    PrintMeasurement("cudaMemcpy", "cudaMalloc", dev_src, dev_dst, size, end - start);

    test( dev_dst, cudaMemcpy(mem_dst, mem_src, size, cudaMemcpyDeviceToDevice) );
    PrintMeasurement("cudaMemcpy", "cudaMalloc", dev_dst, dev_src, size, end - start);

    test( dev_src, cudaMemcpyPeer(mem_src, dev_src, mem_dst, dev_dst, size) );
    PrintMeasurement("cudaMemcpyPeer", "cudaMalloc", dev_src, dev_dst, size, end - start);

    test( dev_src, cudaMemcpyPeer(mem_dst, dev_dst, mem_src, dev_src, size) );
    PrintMeasurement("cudaMemcpyPeer", "cudaMalloc", dev_dst, dev_src, size, end - start);

    test( dev_dst, cudaMemcpyPeer(mem_src, dev_src, mem_dst, dev_dst, size) );
    PrintMeasurement("cudaMemcpyPeer", "cudaMalloc", dev_src, dev_dst, size, end - start);

    test( dev_dst, cudaMemcpyPeer(mem_dst, dev_dst, mem_src, dev_src, size) );
    PrintMeasurement("cudaMemcpyPeer", "cudaMalloc", dev_dst, dev_src, size, end - start);

#define INSANITY
#ifdef INSANITY
    test( dev_src, cudaMemcpyPeer(mem_src, dev_dst, mem_dst, dev_src, size) );
    PrintMeasurement("cudaMemcpyPeer", "cudaMalloc", dev_src, dev_dst, size, end - start);

    test( dev_src, cudaMemcpyPeer(mem_dst, dev_src, mem_src, dev_dst, size) );
    PrintMeasurement("cudaMemcpyPeer", "cudaMalloc", dev_src, dev_dst, size, end - start);

    test( dev_dst, cudaMemcpyPeer(mem_src, dev_dst, mem_dst, dev_src, size) );
    PrintMeasurement("cudaMemcpyPeer", "cudaMalloc", dev_src, dev_dst, size, end - start);

    test( dev_dst, cudaMemcpyPeer(mem_dst, dev_src, mem_src, dev_dst, size) );
    PrintMeasurement("cudaMemcpyPeer", "cudaMalloc", dev_src, dev_dst, size, end - start);
#endif

    fprintf(stdout, "\n");

    // Release resources
    cudaFree(mem_src);
    cudaFree(mem_dst);
}
    

__host__ void ListDevices()
{
    int dev_count;
    int curr_dev;

    cudaError_t err;

    err = cudaGetDeviceCount(&dev_count);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "%s\n", cudaGetErrorString(err));
        exit(1);
    }

    fprintf(stdout, "%d %s: N/A [N/A]\n", -1, "Host");
    for (curr_dev = 0; curr_dev < dev_count; ++curr_dev)
    {
        cudaDeviceProp devProperties;

        GetDeviceInfo(curr_dev, &devProperties);

        fprintf(stdout, "%d %s: %02x:%02x.%x [%s]\n", 
                curr_dev, devProperties.name, 
                devProperties.pciBusID, devProperties.pciDomainID, devProperties.pciDeviceID,
                devProperties.computeMode != cudaComputeModeProhibited ? "enabled" : "disabled"
                );
    }
}


int main(int argc, char** argv)
{
    int opt, optidx;
    char* strptr;

    int dev_src = 0;
    int dev_dst = 0;
    size_t size = 0;
    int dev_count;

    cudaError_t err = cudaGetDeviceCount(&dev_count);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "%s\n", cudaGetErrorString(err));
        return 1;
    }

    while ((opt = getopt_long(argc, argv, ":hs:lf:t:v", NULL, &optidx)) != -1)
    {
        switch (opt)
        {
            case ':': // missing value
                fprintf(stderr, "Option -%c requires a value\n", optopt);
                goto giveUsage;

            case '?': // unknown flag
                fprintf(stderr, "Unknown option: -%c\n", optopt);
                goto giveUsage;

            case 'h': // show help
                goto giveUsage;

            case 'l': // list CUDA enabled devices
                ListDevices();
                return 0;

            case 's': // set memory size
                strptr = NULL;
                size = strtoul(optarg, &strptr, 0);
                if (strptr == NULL || *strptr != '\0' || size == 0)
                {
                    fprintf(stderr, "Option -s requires a valid memory size\n");
                    goto giveUsage;
                }
                break;

            case 'f': // set source device
                strptr = NULL;
                dev_src = strtoul(optarg, &strptr, 0);
                if (strptr == NULL || *strptr != '\0')
                {
                    fprintf(stderr, "Option -f requires a valid CUDA device number\n");
                    goto giveUsage;
                }
                break;

            case 't': // set destination device
                strptr = NULL;
                dev_dst = strtoul(optarg, &strptr, 0);
                if (strptr == NULL || *strptr != '\0')
                {
                    fprintf(stderr, "Option -t requires a valid CUDA device number\n");
                    goto giveUsage;
                }
                break;

            case 'v': // increase verbosity level
                ++verbosity;
                break;
        }
    }

    if (size == 0)
    {
        fprintf(stderr, "Memory size is a required argument\n");
        goto giveUsage;
    }

    if (dev_src >= dev_count || dev_src < 0)
    {
        fprintf(stderr, "Invalid device number: %d\n", dev_src);
        return 1;
    }

    if (dev_dst >= dev_count || dev_src < 0)
    {
        fprintf(stderr, "Invalid device number: %d\n", dev_dst);
        return 1;
    }

    MeasureMemcpy(size * 1000000L, dev_src, dev_dst);

    return 0;

giveUsage:
    fprintf(stderr, 
            "Usage: %s -l\n"
            "Usage: %s -s <size> [-t <to device>] [-f <from device>] [-v]\n"
            "\n"
            "  -l\tlist CUDA enabled devices and quit\n"
            "  -s\tsize in megabytes\n"
            "  -f\tsource device (defaults to 0)\n"
            "  -t\tdestination device (defaults to 0)\n"
            "  -v\tbe more verbose\n"
            "\n",
            argv[0], argv[0]);
    return 1;
}
