#ifndef __INTERNAL_TYPES_H__
#define __INTERNAL_TYPES_H__

#include "simple_types.h"
#include <stdint.h>
#include <stddef.h>


#ifndef MAX_EXPORTS
#define MAX_EXPORTS 16
#endif


// Export list entry
// Used to keep track of which adapters the segment is exported on
struct export
{
    unsigned adapt_no;  // the adapter the segment is exported on
    unsigned flags;     // the SISCI flags used for SCIPrepareSegment
    unsigned available; // has SCISetSegmentAvailable been called
};


// Internal structure holding the local segment descriptor and its state
struct local_segment
{
    unsigned            attached  : 1,  // SCICreateSegment succeeded
                        rw_mapped : 1,  // SCIMapLocalSegment w/ RO succeeded
                        ro_mapped : 1;  // SCIMapLocalSegment succeeded
    sci_desc_t          sci_d;          // SISCI descriptor
    sci_local_segment_t seg_d;          // local segment descriptor
    unsigned            seg_id;         // local segment identifier
    size_t              seg_sz;         // size of local segment
    unsigned            fl_create;      // additional flags passed do SCICreateSegment
    unsigned            fl_attach;      // additional flags passed to SCIAttachPhysicalMemory
    sci_map_t           ro_map;         // map descriptor for RO memory
    const void*         ro_ptr;         // pointer to mapped RO memory
    sci_map_t           rw_map;         // map descriptor for RW memory
    void*               rw_ptr;         // pointer to mapped RW memory
    struct export       exports[MAX_EXPORTS]; // export list
};



// Internal structure holding the remote segment descriptor and its state
struct remote_segment 
{
    unsigned             ro_mapped : 1, // SCIMapRemoteSegment w/ RO succeeded
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

#endif
