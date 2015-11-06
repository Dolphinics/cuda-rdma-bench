#include <cuda.h>
#include <stdint.h>
#include <sisci_api.h>
#include <stdio.h>

extern "C" {
#include "common.h"
}

__host__ sci_local_segment_t createSegment(uint32_t id, sci_desc_t sd, uint32_t adapter, size_t size)
{
    sci_error_t err;
    sci_local_segment_t segment;

    SCICreateSegment(sd, &segment, id, size, NULL, NULL, 0, &err);
    if (SCI_ERR_OK != err)
    {
        fprintf(stderr, "SISCI error: %s\n", sciGetErrorString(err));
        exit(err);
    }

    SCIPrepareSegment(segment, adapter, 0, &err);
    if (SCI_ERR_OK != err)
    {
        fprintf(stderr, "SISCI error: %s\n", sciGetErrorString(err));
        exit(err);
    }

    SCISetSegmentAvailable(segment, adapter, 0, &err);
    if (SCI_ERR_OK != err)
    {
        fprintf(stderr, "SISCI error: %s\n", sciGetErrorString(err));
        exit(err);
    }

    return segment;
}

extern "C"
void PingNode(sci_desc_t dev_desc, uint32_t adapter, uint32_t local_id, size_t seg_size)
{
    const uint32_t segment_id = SEGMENT_ID(local_id, 0, 0);
    sci_local_segment_t segment = createSegment(segment_id, dev_desc, adapter, seg_size);

    fprintf(stderr, "Created segment with ID = %u, local node ID = %u\n", segment_id, local_id);
    while (1);
}
