#ifndef __UTIL_H__
#define __UTIL_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <sisci_api.h>


/**
 * \brief Get a node ID by hostname
 *
 * Lookup node identifier by hostname.
 *
 * \param[in]   hostname    hostname or IP address of node
 *
 * \returns node ID of the host
 */
unsigned GetNodeIdByName(const char* hostname);


/**
 * \brief Get a description of a SISCI API error
 *
 * Get a description of a SISCI API error code.
 *
 * \param[in]   code    the error code returned by the SISCI API
 *
 * \returns a textual description
 *
 * \note this should be regarded as the SISCI equivalent of \c strerror 
 *       or \c cudaGetErrorString
 */
const char* GetErrorString(sci_error_t code);

#define GetErrStr(code) GetErrorString((code))


#ifdef __cplusplus
}
#endif
#endif
