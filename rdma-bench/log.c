#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <sisci_api.h>
#include "log.h"

#define BUFLEN 1024


unsigned int verbosity = 0;



static inline void output(const char* hdr, const char* msg, size_t len)
{
    fwrite(hdr, 9, 1, stderr);
    fwrite(msg, len, 1, stderr);
    fwrite("\n", 1, 1, stderr);
    fflush(stderr);
}


void error(const char* frmt, ...)
{
    va_list args;
    char buf[BUFLEN];
    size_t len;

    va_start(args, frmt);
    len = vsnprintf(buf, sizeof(buf), frmt, args);
    va_end(args);

    output("ERROR  : ", buf, len);
}


void warn(const char* frmt, ...)
{
    if (verbosity < 1)
    {
        return;
    }

    va_list args;
    char buf[BUFLEN];
    size_t len;

    va_start(args, frmt);
    len = vsnprintf(buf, sizeof(buf), frmt, args);
    va_end(args);

    output("WARNING: ", buf, len);
}


void info(const char* frmt, ...)
{
    if (verbosity < 2)
    {
        return;
    }

    va_list args;
    char buf[BUFLEN];
    size_t len;

    va_start(args, frmt);
    len = vsnprintf(buf, sizeof(buf), frmt, args);
    va_end(args);

    output("INFO   : ", buf, len);
}


void debug(const char* frmt, ...)
{
    if (verbosity < 3)
    {
        return;
    }

    va_list args;
    char buf[BUFLEN];
    size_t len;

    va_start(args, frmt);
    len = vsnprintf(buf, sizeof(buf), frmt, args);
    va_end(args);

    output("DEBUG  : ", buf, len);
}


