#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sisci_api.h>
#include "reporting.h"
#include "translist.h"
#include "common.h"
#include "gpu.h"
#include "util.h"


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


void report_bandwidth(FILE* fp, bench_mode_t mode, translist_t tl, size_t num, double* runs, double total, int iec)
{
    translist_desc_t td = translist_desc(tl);
    gpu_info_t info;
    
    fprintf(fp, "========= BENCHMARK =========\n");
    fprintf(fp, "reps: %lu\n", num);
    fprintf(fp, "type: %s\n", bench_mode_name(mode));
    fprintf(fp, "size: %lu bytes\n", td.segment_size);
    if (td.gpu_device_id != NO_GPU)
    {
        gpu_info(td.gpu_device_id, &info);
        fprintf(fp, "gpu : %s (%02x:%02x:%x)\n", info.name, info.domain, info.bus, info.device);
    }
    else
    {
        fprintf(fp, "gpu : N/A\n");
    }
    fprintf(fp, "len : %lu vector entries\n", translist_size(tl));

    fprintf(fp, "========= BANDWIDTH =========\n");
    for (size_t run = 0; run < num; ++run)
    {
        fprintf(fp, "%3lu %16.3f %-5s\n", run + 1, runs[run], iec ? "MiB/s" : "MB/s");        
    }
    fprintf(fp, "avg %16.3f %-5s\n", total, iec ? "MiB/s" : "MB/s");
}
