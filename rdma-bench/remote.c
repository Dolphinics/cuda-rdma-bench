#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <limits.h>
#include <sisci_api.h>
#include "remote.h"
#include "log.h"
#include "util.h"


// Internal structure holding the remote segment descriptor and its state
struct remote_segment 
{
    unsigned             connected : 1, // SCIConnectSegment succeeded
                         ro_mapped : 1, // SCIMapRemoteSegment w/ RO succeeded
                         rw_mapped : 1; // SCIMapRemoteSegment succeeded
    sci_desc_t           sci_d;         // SISCI descriptor
    unsigned             adapt_no;      // local adapter number
    unsigned             node_id;       // remote node identifier
    sci_remote_segment_t seg_d;         // remote segment descriptor
    unsigned             seg_id;        // remote segment identifier
    size_t               seg_sz;        // size of remote segment
    sci_map_t            ro_map;        // map descriptor for RO memory
    const volatile void* ro_ptr;        // pointer to mapped RO memory
    sci_map_t            rw_map;        // map descriptor for RW memory
    volatile void*       rw_ptr;        // pointer to mapped RW memory
    unsigned             fl_connect;    // flags for SCIConnectSegment
    unsigned             fl_map_ro;     // flags for SCIMapRemoteSegment
    unsigned             fl_map_rw;     // flags for SCIMapRemoteSegment
};
