#include <stdio.h>
#include <sisci_api.h>
#include "local.h"
#include "log.h"


int main(int argc, char** argv)
{
    sci_error_t err;
    SCIInitialize(0, &err);

    verbosity = 4;

    l_segment_t segment;

    if (CreateLocalSegment(&segment, 10, 0) != 0)
    {
        return 1;
    }

    if (AllocSegmentMem(segment, 4096, 0) != 0)
    {
        RemoveLocalSegment(segment);
        return 1;
    }

    int* ptr = (int*) GetLocalSegmentPtr(segment);
    if (ptr == NULL)
    {
        RemoveLocalSegment(segment);
        return 1;
    }

    *ptr = 0xdeadbeef;

    printf("%x\n", *ptr);
    
    RemoveLocalSegment(segment);
    SCITerminate();

    return 0;
}
