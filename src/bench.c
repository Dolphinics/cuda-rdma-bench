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
    uint8_t byte = 0;
    srand(ts_usecs());

    do
    {
        byte = rand() & 0xff;
    }
    while (byte == 0x00 || byte == 0xff);

    return byte;
}

