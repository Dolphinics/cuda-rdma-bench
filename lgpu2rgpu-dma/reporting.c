#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
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



void report_buffer_change(FILE* fp, uint8_t old, uint8_t new)
{
    log_debug("Value before transfer was %02x", old);
    log_debug("Value after transfer was  %02x", new);

    if (old != new && new != 0x00)
    {
        fprintf(fp, "******* DATA RECEIVED SUCCESSFULLY *******\n");
    }
    else
    {
        fprintf(fp, "******* DATA NOT RECEIVED PROPERLY *******\n");
    }
}



void report_summary(FILE* fp, const bench_t* test, const result_t* result, int iec)
{
    translist_desc_t td = translist_desc(test->transfer_list);

    fprintf(fp, "===============   BENCHMARK   ===============\n");
    fprintf(fp, "benchmark type: %s\n", bench_mode_name(test->benchmark_mode));
    fprintf(fp, "overall status: %4s\n", result->success_count == test->num_runs && result->buffer_matches ? "pass" : "fail");
    fprintf(fp, "buffers match : %-3s\n", result->buffer_matches ? "yes" : "no");
    fprintf(fp, "segment size  : %.3lf %-3s\n", (double) td.segment_size / (iec ? 1<<20 : 1e6), iec ? "MiB" : "MB");
    fprintf(fp, "repetitions   : %lu\n", test->num_runs);
    fprintf(fp, "success runs  : %lu\n", result->success_count);

    fprintf(fp, "transfer size : %.3f %-3s x %lu\n", (double) result->total_size / (iec ? 1<<20 : 1e6), iec ? "MiB" : "MB", test->num_runs);

    size_t ts = translist_size(test->transfer_list);
    translist_entry_t te;
    translist_element(test->transfer_list, 0, &te);

    fprintf(fp, "transfer units: %lu x %.3f %s\n",
            ts, te.size / (iec ? 1<<20 : 1e6), iec ? "MiB" : "MB" );
    
    if (td.local_gpu_info != NULL)
    {
        fprintf(fp, "local memory  : gpu\n");
        fprintf(fp, "local gpu     : #%d %s (local ioaddr %02x:%02x.%x)\n", 
                td.local_gpu_info->id, td.local_gpu_info->name, 
                td.local_gpu_info->domain, td.local_gpu_info->bus, td.local_gpu_info->device);
    }
    else
    {
        fprintf(fp, "local memory  : ram\n");
        fprintf(fp, "local gpu     : not applicable\n");
    }

    if (td.remote_gpu_info != NULL)
    {
        fprintf(fp, "remote memory : gpu\n");
        fprintf(fp, "remote gpu    : #%d %s (remote ioaddr %02x:%02x.%x)\n", 
                td.remote_gpu_info->id, td.remote_gpu_info->name, 
                td.remote_gpu_info->domain, td.remote_gpu_info->bus, td.remote_gpu_info->device);
    }
    else
    {
        fprintf(fp, "remote memory : ram\n");
        fprintf(fp, "remote gpu    : not applicable\n");
    }
}



void report_bandwidth(FILE* fp, const bench_t* test, const result_t* result, int iec)
{
    fprintf(fp, "===============   BANDWIDTH   ===============\n");

    const char* bw_unit = iec ? "MiB/s" : "MB/s";
    const char* mb_unit = iec ? "MiB" : "MB";
    
    double megabytes_per_sec;
    
    fprintf(fp, "%3s %-13s %-10s %-17s\n", 
            "#", "segment size", "latency", "bandwidth");

    for (size_t i = 0; i < test->num_runs; ++i)
    {
        size_t n = translist_size(test->transfer_list);
        size_t size = 0;
        for (size_t e = 0; e < n; ++e)
        {
            translist_entry_t entry;
            translist_element(test->transfer_list, e, &entry);
            size += entry.size;
        }

        if (result->runtimes[i] != 0)
        {
            megabytes_per_sec = (double) size / (double) result->runtimes[i];

            fprintf(fp, "%3lu %7.2f %-3s %7lu µs %11.3f %-5s\n", 
                    i + 1, (double) size / (iec ? 1<<20 : 1e6), mb_unit, result->runtimes[i], megabytes_per_sec, bw_unit);
        }
        else
        {
            fprintf(fp, "%3lu %-13s %-10s %-17s\n", i + 1, "---", "---", "---");
        }
    }

    double total_size = result->total_size;
    total_size *= test->num_runs;

    megabytes_per_sec = total_size / result->total_runtime;
    fprintf(fp, "avg %7.2f %-3s %7lu µs %11.3f %-5s\n",
            total_size / (iec ? 1<<20 : 1e6), mb_unit, result->total_runtime, megabytes_per_sec, bw_unit);
}
