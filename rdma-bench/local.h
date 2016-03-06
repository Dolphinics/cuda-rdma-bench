#ifndef __LOCAL_SEGMENT_H__
#define __LOCAL_SEGMENT_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <stdint.h>

// FIXME: What about Shared segments?


/**
 * Maximum number of exports of a local segment.
 */
#define MAX_EXPORTS 16

/**
 * \brief Local segment descriptor handle
 *
 * Handle for the internal local segment descriptor. 
 *
 * \note This type serves as a simplified handle type for the 
 *       \c sci_local_segment_t SISCI type.
 */
typedef struct local_segment* l_segment_t;


/**
 * \brief Create a local segment
 *
 *  Create a local segment and initialize the local segment descriptor.
 *
 * \param[out]  segment     segment descriptor handle
 * \param[in]   segmentId   unique (for this node) segment identifier
 * \param[in]   flags       additional flags for \c SCICreateSegment
 *
 * \returns \c 0 on success
 */
int CreateLocalSegment(l_segment_t* segment, unsigned segmentId, unsigned flags);


/**
 * \brief Remove a local segment
 *
 * Remove and clean up a local segment. This function destroys the local
 * segment descriptor.
 *
 * \param[in]    segment     segment descriptor handle
 *
 * \returns \c 0 on success
 */
int RemoveLocalSegment(l_segment_t segment);


/**
 * \brief Allocate segment memory
 *
 * Allocate a RAM memory range for the local segment.
 *
 * \param[in]   segment     segment descriptor handle
 * \param[in]   size        size of the memory range
 *
 * \returns \c 0 on success
 */
int AllocSegmentMem(l_segment_t segment, size_t size);


/**
 * \brief Attach physical address range to a local segment
 *
 * Attach a physical address range to the local segment.
 *
 * \param[in]   segment     segment descriptor handle
 * \param[in]   addr        start address of the physical address range
 * \param[in]   size        size of the address range
 * \param[in]   flags       additional flags for \c SCIAttachPhysicalMemory
 *
 * \returns \c 0 on success
 */
int AttachPhysAddr(l_segment_t segment, uintptr_t addr, size_t size, unsigned flags);


/**
 * \brief Attach virtual memory to a local segment
 *
 * Attach an already allocated RAM memory range for the local segment.
 *
 * \param[in]   segment     segment descriptor handle
 * \param[in]   ptr         pointer to the start of the virtual address range
 * \param[in]   size        size of the memory range
 *
 * \returns \c 0 on success
 */
int AttachVirtMem(l_segment_t segment, void* ptr, size_t size);


/**
 * \brief Attach GPU memory to a local segment
 *
 * Attach a range of GPU memory to the local segment.
 *
 * \param[in]   segment     segment descriptor handle
 * \param[in]   devicePtr   a CUDA \c devicePointer 
 * \param[in]   size        size of the memory range
 *
 * \returns \c 0 on success
 *
 * \note A valid value for \c devicePtr can be retrieved 
 *       using \c cudaPointerGetAttributes
 *
 * \note GPU memory must be allocated using either \c cudaMalloc or
 *       \c cudaHostAlloc (with the \c cudaHostAllocMapped flag set).
 *
 * \note SISCI driver must be built with CUDA support.
 */
int AttachCudaMem(l_segment_t segment, void* cudaPtr, size_t size);


/**
 * \brief Export local segment
 *
 * Export a local segment on a specific adapter so that remote nodes can 
 * connect to it.
 *
 * \param[in]   segment     segment descriptor handle
 * \param[in]   adapterNo   identifier for the local adapter
 * \param[in]   flags       additional flags for \c SCIPrepareSegment
 *
 * \returns \c 0 on success
 */
int ExportLocalSegment(l_segment_t segment, unsigned adapterNo, unsigned flags);


/**
 * \brief Unexport local segment
 *
 * Make a previously exported local segment unavailable for remote nodes on
 * a specific adapter.
 * 
 * \param[in]   segment     segment descriptor handle
 * \param[in]   adapterNo   identifier for the local adapter
 *
 * \returns \c 0 on success
 */
int UnexportLocalSegment(l_segment_t segment, unsigned adapterNo);


/**
 * \brief Get pointer to local segment memory
 *
 * Map segment memory into virtual memory and return a pointer to it.
 *
 * \param[in]   segment     segment descriptor handle
 *
 * \returns a memory mapped pointer or \c NULL on error
 */
void* GetLocalSegmentPtr(l_segment_t segment);


/**
 * \breif Get read-only pointer to local segment memory
 *
 * Map segment memory into read-only memory and return a pointer to it.
 *
 * \param[in]   segment     segment descriptor handle
 * 
 * \returns a memory mapped pointer or \c NULL on error
 */
const void* GetLocalSegmentPtrRO(l_segment_t segment);


/**
 * \brief Get the size of local segment memory
 *
 * Get the size of the address range associated with the local segment.
 *
 * \param[in]   segment     segment descriptor handle
 *
 * \returns size of the segment in bytes or \c 0 on error
 */
size_t GetLocalSegmentSize(l_segment_t segment);


#ifdef __cplusplus
}
#endif
#endif
