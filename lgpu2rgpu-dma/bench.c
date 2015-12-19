#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include "reporting.h"


uint64_t ts_usecs()
{
    struct timespec ts;
    if (clock_gettime(CLOCK_REALTIME, &ts) < 0)
    {
        return 0;
    }
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}


uint8_t random_byte_value()
{
    srand(ts_usecs());
    return (rand() & 0xff) + 1; // should never be 0x00
}

