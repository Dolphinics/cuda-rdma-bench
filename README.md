GPUDirect + Dolphin ICS SISCI
=============================

This repo contains a bunch of benchmarking programs aimed at testing different
configurations of local and remote GPUs connected across a PCIe NTB link.

_TODO_: Better description

Repo structure
--------------
`lgpu2lgpu-p2p` test bandwidth of device to device memory transfer on a local machine with CUDA only.
`lgpu2lgpu-dma` test bandwidth of device to device memory transfer on a local machine using SISCI DMA.
`lram2rgpu-memcpy` test bandwidth of memcpy from local RAM to a remote GPU using SISCI.

_TODO_: Fill in the blanks


Notes
-----
NVIDIA Quadro and Tesla only.
