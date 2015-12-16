#ifndef __REPORTING_H__
#define __REPORTING_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "translist.h"
#include "bench.h"

extern unsigned verbosity;

void log_info(const char* format, ...);

void log_warn(const char* format, ...);

void log_error(const char* format, ...);

void log_debug(const char* format, ...);

void report_buffer_change(FILE* file, uint8_t old_value, uint8_t new_value);

void report_summary(FILE* file, const bench_t* benchmark, const result_t* result, int iec_units);

void report_bandwidth(FILE* file, const bench_t* benchmark, const result_t* result, int iec_units);

#ifdef __cplusplus
}
#endif
#endif
