#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "ram.h"
#include "reporting.h"


int ram_memcmp(void* local, volatile void* remote, size_t len)
{
    log_debug("Comparing local RAM memory %p to remote memory %p", local, remote);
    return memcmp(local, (void*) remote, len);
}


void ram_memset(void* buf, size_t len, uint8_t val)
{
    log_debug("Filling buffer with value %02x...", val);
    memset(buf, len, val);
}


void ram_memcpy_remote_to_local(void* dst, volatile void* src, size_t len)
{
    memcpy(dst, (void*) src, len);
}


void ram_memcpy_local_to_remote(void* src, volatile void* dst, size_t len)
{
    memcpy((void*) dst, src, len);
}
