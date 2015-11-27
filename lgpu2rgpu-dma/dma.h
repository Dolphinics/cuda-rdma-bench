#ifndef __DMA_H__
#define __DMA_H__

#include <sisci_api.h>
#include <stdint.h>
#include "local.h"
#include "remote.h"

double DMAPush(sci_desc_t dev, uint32_t adapterNo, local_t local, size_t loff, remote_t remote, size_t roff, int repeat);

double DMAPull(sci_desc_t dev, uint32_t adapterNo, local_t local, size_t loff, remote_t remote, size_t roff, int repeat);

#endif
