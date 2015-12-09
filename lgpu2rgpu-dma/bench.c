#include <stdint.h>
#include <stdlib.h>
#include <sisci_api.h>
#include <pthread.h>
#include <time.h>
#include "translist.h"
#include "common.h"
#include "util.h"
#include "reporting.h"
#include "bench.h"


/* Should we keep running? */
static volatile int keep_running = 1;
static pthread_cond_t signal = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;


uint64_t ts_usecs()
{
    struct timespec ts;
    if (clock_gettime(CLOCK_REALTIME, &ts) < 0)
    {
        return 0;
    }
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}


void stop_server()
{
    log_info("Stopping server...");

    pthread_mutex_lock(&lock);
    keep_running = 0;
    pthread_cond_signal(&signal);
    pthread_mutex_unlock(&lock);
}


void server(unsigned adapter, int gpu, unsigned id, size_t size)
{
    sci_error_t err;
    sci_desc_t sd;

    SCIOpen(&sd, 0, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to open SISCI descriptor");
        return;
    }

    sci_local_segment_t segment;
    sci_map_t mapping;
    void* buffer;

    if (gpu != NO_GPU)
    {
        err = make_gpu_segment(sd, adapter, id, &segment, size, gpu, &buffer);
    }
    else
    {
        err = make_ram_segment(sd, adapter, id, &segment, size, &mapping, &buffer);
    }

    if (err != SCI_ERR_OK)
    {
        log_error("Failed to create segment");
        goto close_desc;
    }

    SCISetSegmentAvailable(segment, adapter, 0, &err);
    if (err != SCI_ERR_OK)
    {
        log_error("Failed to set segment available");
        goto free_segment;
    }

    log_info("Running server...");
    pthread_mutex_lock(&lock);
    while (keep_running)
    {
        pthread_cond_wait(&signal, &lock);
    }
    pthread_mutex_unlock(&lock);
    log_info("Server stopped");

    SCISetSegmentUnavailable(segment, adapter, SCI_FLAG_NOTIFY | SCI_FLAG_FORCE_DISCONNECT, &err);

free_segment:
    if (gpu != NO_GPU)
    {
        free_gpu_segment(segment, gpu, buffer);
    }
    else
    {
        free_ram_segment(segment, mapping);
    }
close_desc:
    SCIClose(sd, 0, &err);
}


void client(bench_mode_t mode, translist_t tl, int repeat, int use_iec)
{
    translist_desc_t tl_desc = translist_desc(tl);

    // Do benchmark
    switch (mode)
    {
        case BENCH_SCI_DATA_INTERRUPT:
            log_error("%s is not yet supported", bench_mode_name(mode));
            break;

        default:
        case BENCH_DO_NOTHING:
            log_error("No benchmarking operation is set");
            break;
    }

    // Verify transfer
    // TODO: Map remote segment and do memcmp
}
