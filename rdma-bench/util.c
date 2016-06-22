#include <sisci_api.h>
#include <stdlib.h>
#include "util.h"
#include "log.h"


int GetNodeIdByName(const char* hostname)
{
    int node_id = 0;
    char* strptr = NULL;

    warn("GetNodeIdByName is not fully implemented");

    node_id = strtol(hostname, &strptr, 0);
    if (strptr == NULL || *strptr != '\0' || node_id <= 0)
    {
        error("Invalid node ID: %s", hostname);
        return -1;
    }

    return node_id;
}
