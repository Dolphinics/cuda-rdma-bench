#ifndef __REPORTING_H__
#define __REPORTING_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <sisci_api.h>
#include <stdarg.h>
#include <stdio.h>

extern unsigned verbosity;

void log_info(const char* format, ...);

void log_warn(const char* format, ...);

void log_error(const char* format, ...);

void log_debug(const char* format, ...);

const char* SCIGetErrorString(sci_error_t error);

#ifdef __cplusplus
}
#endif
#endif
