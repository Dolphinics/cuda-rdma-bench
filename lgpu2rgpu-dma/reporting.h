#ifndef __REPORTING_H__
#define __REPORTING_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stdarg.h>
#include <stdio.h>
#include "translist.h"
#include <stdlib.h>

extern unsigned verbosity;

void log_info(const char* format, ...);

void log_warn(const char* format, ...);

void log_error(const char* format, ...);

void log_debug(const char* format, ...);

void report_bandwidth(FILE* file, translist_t trans_list, size_t num_runs, double* runs, double avg, int iec_units);

#ifdef __cplusplus
}
#endif
#endif
