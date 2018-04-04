#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "ram.h"
#include "reporting.h"
#include "bench.h"


int ram_memcmp(void* local, volatile void* remote, size_t len)
{
    log_debug("Comparing local RAM memory %p to remote memory %p", local, remote);
    return memcmp(local, (void*) remote, len);
}


void ram_memset(void* buf, size_t len, uint8_t val)
{
    log_debug("Filling buffer with value %02x...", val);
    memset(buf, val, len);
}


uint64_t ram_memcpy_remote_to_local(volatile void* dst, volatile void* src, size_t len, int clear)
{
    if (clear)
    {
        memset((void*) dst, 0, len);
    }

    uint64_t before = ts_usecs();
    memcpy((void*) dst, (void*) src, len);
    uint64_t after = ts_usecs();

    return after - before;
}


uint64_t ram_memcpy_local_to_remote(volatile void* src, volatile void* dst, size_t len, int clear)
{
    uint64_t before = ts_usecs();
    memcpy((void*) dst, (void*) src, len);
    uint64_t after = ts_usecs();

    if (clear)
    {
        volatile uint8_t* ptr = src;
        for (size_t i = 0; i < len; ++i)
        {
            ptr[i] = ptr[i] + 1;
        }
    }

    return after - before;
}
