GPUDirect + Dolphin ICS SISCI
=============================

Utility for benchmarking bandwidth when using SISCI in combination with
[NVIDIA GPUDirect RDMA](https://developer.nvidia.com/gpudirect)
Can be used to measure performance from/to CUDA GPUs and RAM using various
transfer methods.

Requirements
-----------
* Linux
* NVIDIA Quadro or Tesla GPU (for GPUDirect RDMA support)
* [Dolphin Interconnect Solutions](http://dolphinics.com) software stack and
  supported hardware.

Example
--------
Benchmark data transfer from RAM on node A to 30MB GPU buffer on node B.

### Client
```
./lgpu2rgpu-dma --remote-node=8 --type=dma-push
===============   BENCHMARK   ===============
benchmark type: dma-push
overall status: pass
buffers match : yes
segment size  : 32.002 MB 
repetitions   : 1
success runs  : 1
transfer size : 32.002 MB  x 1
transfer units: 1 x 32.002 MB
local memory  : ram
local gpu     : not applicable
remote memory : gpu
remote gpu    : #0 Tesla K40c (remote ioaddr 08:00.0)
===============   BANDWIDTH   ===============
  # segment size  latency    bandwidth        
  1   32.00 MB     3127 µs   10234.106 MB/s 
avg   32.00 MB     3127 µs   10234.106 MB/s 
```

### Server
```
./lgpu2rgpu-dma --size=32 --gpu=0
******* DATA RECEIVED SUCCESSFULLY *******
```

Benchmark types
---------------
* `dma-push`            use DMA to push data to remote host
* `dma-pull`            use DMA to pull data from remote host
* `scimemwrite`         use SCIMemWrite to write data to remote host
* `write`               use glibc memcpy / cudaMemcpy to write data to remote host
* `read`                use glibc memcpy / cudaMemcpy to read data from remote host

Usage
-----
```
Usage: ./lgpu2rgpu-dma --size=<size>
   or: ./lgpu2rgpu-dma --type=<benchmark type> --remote-node=<node id>

Description
    Benchmark how long it takes to transfer memory between a local and a
    remote segment across an NTB link.

Server mode arguments
  --size=<size>            memory size in MB (or MiB if --iec is set)

Client mode arguments
  --type=<bencmark type>   specify benchmark type
  --remote-node=<node id>  remote node identifier
  --remote-id=<segment id> number identifying the remote segment
  --count=<number>         number of times to repeat test (defaults to 1)

DMA vector options (client mode)
  --vec=<number>           divide segment into a number of DMA vector elements (defaults to 1)
  --len=<number>           repeat the entire vector a number of times (defaults to 1)

Optional arguments (both client and server mode)
  --adapter=<adapter no>   local host adapter card number (defaults to 0)
  --local-id=<segment id>  number identifying the local segment
  --gpu=<gpu id>           specify a local GPU (if not given, buffer is allocated in RAM)
  --verbose                increase verbosity level
  --iec                    use IEC units (1024) instead of SI units (1000)
  --help                   show list of local GPUs and benchmark types
```
