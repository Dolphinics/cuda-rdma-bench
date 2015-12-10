#include <stdlib.h>
#include <stdint.h>
#include "ram.h"
#include "reporting.h"


size_t ram_memcmp(void* local, volatile void* remote, size_t len)
{
    uint8_t* loc_ptr = (uint8_t*) local;
    volatile uint8_t* rem_ptr = (volatile uint8_t*) remote;

    size_t idx;

    log_debug("Comparing local RAM memory %p to remote memory %p", local, remote);
    for (idx = 0; idx < len; ++idx)
    {
        if (loc_ptr[idx] != rem_ptr[idx])
        {
            log_debug("Byte %lu differs (%02x %02x)", idx, loc_ptr[idx], rem_ptr[idx]);
            break;
        }
    }

    return idx;
}


void ram_memset(void* buf, size_t len, uint8_t val)
{
    uint8_t* ptr = (uint8_t*) buf;

    for (size_t i = 0; i < len; ++i)
    {
        ptr[i] = val;
    }
}


void ram_memcpy_remote_to_local(void* local, volatile void* remote, size_t len)
{
    uint8_t* dst = (uint8_t*) local;
    volatile uint8_t* src = (volatile uint8_t*) remote;

    log_debug("Copying to local RAM memory %p from remote memory %p", local, remote);
    for (size_t i = 0; i < len; ++i)
    {
        dst[i] = src[i];
    }
}


void ram_memcpy_local_to_remote(void* local, volatile void* remote, size_t len)
{
    uint8_t* src = (uint8_t*) local;
    volatile uint8_t* dst = (volatile uint8_t*) remote;

    log_debug("Copying from local RAM memory %p to remote memory %p", local, remote);
    for (size_t i = 0; i < len; ++i)
    {
        dst[i] = src[i];
    }
}
