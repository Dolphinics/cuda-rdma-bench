#include "benchmark.h"
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

uint64_t usecs()
{
    struct timespec ts;

    if (clock_gettime(CLOCK_REALTIME, &ts) != 0)
    {
        fprintf(stderr, "Failed to sample realtime clock\n");
        exit(1);
    }

    return (ts.tv_sec * 1000000L) + (ts.tv_nsec / 1000);
}

void PrintResult(size_t bytes, uint64_t usecs, const char* callstr)
{
    fprintf(stdout, "%12.3f MB/s -- %s\n",
            ((double) size) / ((double) usecs),
            callstr
           );
}
