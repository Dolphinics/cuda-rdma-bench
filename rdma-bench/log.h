#ifndef __LOGGING_H__
#define __LOGGING_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stdarg.h>

// TODO: Rename to ReportError, ReportWarning ReportDebug etc
// TODO: make 0 swallow everything
// TODO: make reportFile variable, rename verbosity to LogLevel or something

/* Verbosity level for reporting
 *
 *  0 = errors only
 *  1 = warnings,
 *  2 = informational
 *  3 = debug information
 */
extern unsigned int verbosity;


/* Log error */
void error(const char* format, ...);


/* Log warning */
void warn(const char* format, ...);


/* Log informational */
void info(const char* format, ...);


/* Debug output */
void debug(const char* format, ...);

#ifdef __cplusplus
}
#endif
#endif
