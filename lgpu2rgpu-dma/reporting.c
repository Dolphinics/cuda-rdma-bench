#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <sisci_api.h>
#include "reporting.h"


#define BUFLEN 1024



/* Verbosity level 
 *
 * 0 = errors only,
 * 1 = warnings and errors
 * 2 = warnings, errors and informatives
 * 3 = everything above + debug
 */
unsigned verbosity = 0;



void log_info(const char* frmt, ...)
{
    if (verbosity >= 2)
    {
        va_list args;
        char buff[BUFLEN];
        size_t size;

        va_start(args, frmt);
        size = vsnprintf(buff, sizeof(buff), frmt, args);
        va_end(args);

        fwrite("INFO   : ", 9, 1, stderr);
        fwrite(buff, size, 1, stderr);
        fwrite("\n", 1, 1, stderr);
        fflush(stderr);
    }
}



void log_warn(const char* frmt, ...)
{
    if (verbosity >= 1)
    {
        va_list args;
        char buff[BUFLEN];
        size_t size;

        va_start(args, frmt);
        size = vsnprintf(buff, sizeof(buff), frmt, args);
        va_end(args);

        fwrite("WARNING: ", 9, 1, stderr);
        fwrite(buff, size, 1, stderr);
        fwrite("\n", 1, 1, stderr);
        fflush(stderr);
    }
}



void log_error(const char* frmt, ...)
{
    va_list args;
    char buff[BUFLEN];
    size_t size;

    va_start(args, frmt);
    size = vsnprintf(buff, sizeof(buff), frmt, args);
    va_end(args);

    fwrite("ERROR  : ", 9, 1, stderr);
    fwrite(buff, size, 1, stderr);
    fwrite("\n", 1, 1, stderr);
    fflush(stderr);
}



void log_debug(const char* frmt, ...)
{
    if (verbosity >= 3)
    {
        va_list args;
        char buff[BUFLEN];
        size_t size;

        va_start(args, frmt);
        size = vsnprintf(buff, sizeof(buff), frmt, args);
        va_end(args);

        fwrite("DEBUG  : ", 9, 1, stderr);
        fwrite(buff, size, 1, stderr);
        fwrite("\n", 1, 1, stderr);
        fflush(stderr);
    }

}
