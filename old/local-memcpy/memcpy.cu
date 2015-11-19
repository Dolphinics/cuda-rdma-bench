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

#define test( dev, func, mem_type, from, to, testcase ) \
    do { \
        uint64_t __end, __start; \
        check( cudaDeviceSynchronize() ); \
        check( cudaSetDevice((dev)) ); \
        check( cudaDeviceSynchronize() ); \
        __start = usecs(); \
        check( (testcase) ); \
        check( cudaDeviceSynchronize() ); \
        __end = usecs(); \
        PrintMeasurement( (func), (mem_type), (dev), (from), (to), size, __end - __start, stringify(testcase)); \
    } while (0) 

#define stringify(s) #s

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

    fprintf(stdout, "%2d %-12s\n", -1, "Host");
    for (curr_dev = 0; curr_dev < dev_count; ++curr_dev)
    {
        cudaDeviceProp devProperties;

        GetDeviceInfo(curr_dev, &devProperties);

        fprintf(stdout, "%2d %-12s: %02x:%02x.%x [%s]\n", 
                curr_dev, devProperties.name, 
                devProperties.pciBusID, devProperties.pciDomainID, devProperties.pciDeviceID,
                devProperties.computeMode != cudaComputeModeProhibited ? "enabled" : "disabled"
                );

        if (verbosity > 1)
        {
            int i;
            for (i = 0; i < dev_count; ++i)
            {
                int access = 0;
                err = cudaDeviceCanAccessPeer(&access, curr_dev, i);
                if (err != cudaSuccess)
                {
                    exit(1);
                }
                fprintf(stdout, "\tcan access device %2d: %s\n", i, access ? "yes" : "no");
            }
        }
    }
}


__host__ void PrintMeasurement(const char* func_name, const char* mem_type, int ctl, int src, int dst, size_t size, uint64_t usec, const char* s)
{

    fprintf(stdout, "%-16s :: %-21s",
            func_name,
            mem_type);

    if (verbosity > 1)
    {
        char ctl_name[256];
        char src_name[256];
        char dst_name[256];

        GetDeviceName(ctl, ctl_name, sizeof(ctl_name));
        GetDeviceName(src, src_name, sizeof(src_name));
        GetDeviceName(dst, dst_name, sizeof(dst_name));

        fprintf(stdout, " :: ctl=%d %-12s",
                ctl, ctl_name);

        fprintf(stdout, " :: src=%d %-12s -> dst=%d %-12s",
                src, src_name, dst, dst_name);
    }
    else
    {
        if (verbosity > 0)
        {
            fprintf(stdout, " :: ctl=%2d", ctl);
            fprintf(stdout, " :: src=%2d -> dst=%2d",
                    src,
                    dst);
        }
        else
        {
            fprintf(stdout, " :: %2d -> %2d",
                    src,
                    dst);
        }
    }
           
    fprintf(stdout, " :: %12.3f MB/s",
            ((double) size) / ((double) usec)
            );

    if (verbosity > 2)
    {
        fprintf(stdout, "\n\t");
        if (verbosity > 3)
        {
            fprintf(stdout, "%s", s);
        }
        fprintf(stdout, "\t--\t%lu MB copied in %lu µs\n", 
                size / 1000000L,
                usec);
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

    // HOST TO DEVICE
    fprintf(stdout, "\nHOST TO DEVICE\n");

    mem_host = malloc(size);
    test( dev_dst, "cudaMemcpy", "malloc'd", -1, dev_dst, cudaMemcpy(mem_dst, mem_host, size, cudaMemcpyHostToDevice) );
    free(mem_host);

    mem_host = AllocHostMemory(size, cudaHostAllocDefault);
    test( dev_dst, "cudaMemcpy", "cudaHostAllocDefault", -1, dev_dst, cudaMemcpy(mem_dst, mem_host, size, cudaMemcpyHostToDevice) );
    cudaFreeHost(mem_host);

    mem_host = AllocHostMemory(size, cudaHostAllocPortable);
    test( dev_dst, "cudaMemcpy", "cudaHostAllocPortable", -1, dev_dst, cudaMemcpy(mem_dst, mem_host, size, cudaMemcpyHostToDevice) );
    cudaFreeHost(mem_host);

    mem_host = AllocHostMemory(size, cudaHostAllocMapped);
    test( dev_dst, "cudaMemcpy", "cudaHostAllocMapped", -1, dev_dst, cudaMemcpy(mem_dst, mem_host, size, cudaMemcpyHostToDevice) );
    cudaFreeHost(mem_host);

    mem_host = AllocHostMemory(size, cudaHostAllocWriteCombined);
    test( dev_dst, "cudaMemcpy", "cudaHostAllocWriteCom", -1, dev_dst, cudaMemcpy(mem_dst, mem_host, size, cudaMemcpyHostToDevice) );
    cudaFreeHost(mem_host);


    // DEVICE TO HOST
    fprintf(stdout, "\nDEVICE TO HOST\n");

    mem_host = malloc(size);
    test( dev_dst, "cudaMemcpy", "malloc'd", dev_dst, -1, cudaMemcpy(mem_host, mem_dst, size, cudaMemcpyDeviceToHost) );
    free(mem_host);

    mem_host = AllocHostMemory(size, cudaHostAllocDefault);
    test( dev_dst, "cudaMemcpy", "cudaHostAllocDefault", dev_dst, -1, cudaMemcpy(mem_host, mem_dst, size, cudaMemcpyDeviceToHost) );
    cudaFreeHost(mem_host);

    mem_host = AllocHostMemory(size, cudaHostAllocPortable);
    test( dev_dst, "cudaMemcpy", "cudaHostAllocPortable", dev_dst, -1, cudaMemcpy(mem_host, mem_dst, size, cudaMemcpyDeviceToHost) );
    cudaFreeHost(mem_host);

    mem_host = AllocHostMemory(size, cudaHostAllocMapped);
    test( dev_dst, "cudaMemcpy", "cudaHostAllocMapped", dev_dst, -1, cudaMemcpy(mem_host, mem_dst, size, cudaMemcpyDeviceToHost) );
    cudaFreeHost(mem_host);

    mem_host = AllocHostMemory(size, cudaHostAllocWriteCombined);
    test( dev_dst, "cudaMemcpy", "cudaHostAllocWriteCom", dev_dst, -1, cudaMemcpy(mem_host, mem_dst, size, cudaMemcpyDeviceToHost) );
    cudaFreeHost(mem_host);



    // DEVICE TO DEVICE
    fprintf(stdout, "\nDEVICE TO DEVICE\n");

    test( dev_src, "cudaMemcpy", "cudaMalloc", dev_src, dev_dst, cudaMemcpy(mem_src, mem_dst, size, cudaMemcpyDeviceToDevice) );

    test( dev_src, "cudaMemcpy", "cudaMalloc", dev_dst, dev_src, cudaMemcpy(mem_dst, mem_src, size, cudaMemcpyDeviceToDevice) );
   
    test( dev_dst, "cudaMemcpy", "cudaMalloc", dev_src, dev_dst, cudaMemcpy(mem_src, mem_dst, size, cudaMemcpyDeviceToDevice) );

    test( dev_dst, "cudaMemcpy", "cudaMalloc", dev_dst, dev_src, cudaMemcpy(mem_dst, mem_src, size, cudaMemcpyDeviceToDevice) );

    test( dev_src, "cudaMemcpyPeer", "cudaMalloc", dev_src, dev_dst, cudaMemcpyPeer(mem_src, dev_src, mem_dst, dev_dst, size) );

    test( dev_src, "cudaMemcpyPeer", "cudaMalloc", dev_dst, dev_src, cudaMemcpyPeer(mem_dst, dev_dst, mem_src, dev_src, size) );

    test( dev_dst, "cudaMemcpyPeer", "cudaMalloc", dev_src, dev_dst, cudaMemcpyPeer(mem_src, dev_src, mem_dst, dev_dst, size) );

    test( dev_dst, "cudaMemcpyPeer", "cudaMalloc", dev_dst, dev_src, cudaMemcpyPeer(mem_dst, dev_dst, mem_src, dev_src, size) );



    // Use opposite devices
    test( dev_src, "cudaMemcpyPeer*", "cudaMalloc", dev_dst, dev_src, cudaMemcpyPeer(mem_src, dev_dst, mem_dst, dev_src, size) );

    test( dev_src, "cudaMemcpyPeer*", "cudaMalloc", dev_src, dev_dst, cudaMemcpyPeer(mem_dst, dev_src, mem_src, dev_dst, size) );

    test( dev_dst, "cudaMemcpyPeer*", "cudaMalloc", dev_dst, dev_src, cudaMemcpyPeer(mem_src, dev_dst, mem_dst, dev_src, size) );

    test( dev_dst, "cudaMemcpyPeer*", "cudaMalloc", dev_src, dev_dst, cudaMemcpyPeer(mem_dst, dev_src, mem_src, dev_dst, size) );

    // Release resources
    cudaFree(mem_src);
    cudaFree(mem_dst);

    // Print device list
    fprintf(stdout, "\nDEVICE LEGEND\n");
    ListDevices();
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
                verbosity = 100;
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
