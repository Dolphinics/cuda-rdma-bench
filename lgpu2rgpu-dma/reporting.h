#ifndef __REPORTING_H__
#define __REPORTING_H__
#ifdef __cplusplus
extern "C" {
#endif


#include <sisci_api.h>
#include <stdarg.h>
#include <stdio.h>


const char* SCIGetErrorString(sci_error_t error);



void set_verbosity(int level);

//void set_logfile(FILE* file);

void log_info(const char* format, ...);

void log_warn(const char* format, ...);

void log_error(const char* format, ...);

void log_debug(const char* format, ...);


#ifdef __cplusplus
}
#endif
#endif
