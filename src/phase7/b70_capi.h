#pragma once

/**
 * Shooting Brake Phase 7 — C ABI for the B70 routed-MoE provider.
 *
 * A thin C-compatible wrapper around the Phase 1 C++ B70Provider so it can be
 * called from Python via ctypes. All pointers are plain C pointers; the
 * ``hidden`` buffer is raw FP16 (IEEE half) bytes.
 *
 * Return convention: 0 = ok, >0 = busy/pending, <0 = error.
 */

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
#else
#include <stddef.h>
#include <stdint.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** Opaque provider handle. */
typedef void sb_b70_provider_t;

/** Stable C representation of B70Provider::health(). */
typedef struct sb_b70_health {
  uint64_t generation;
  uint64_t dispatches;
  uint64_t allocations;
  uint64_t last_error_bytes;
  uint32_t loaded;
  uint32_t pending;
  uint32_t stopped;
  uint32_t reserved;
} sb_b70_health_t;

#ifdef __cplusplus
static_assert(sizeof(sb_b70_health_t) == 48);
#endif
/** Create a new provider instance. Returns NULL on failure. */
sb_b70_provider_t* sb_b70_create(void);

/**
 * Load an SBEXP001 NVFP4 or SBINT401 int4 bank and select resident experts.
 *
 * @param provider       Handle from sb_b70_create.
 * @param bank_path      Path to the expert bank.
 * @param generation     Placement generation id.
 * @param resident_experts  Source expert IDs resident on the B70. For
 *                          SBINT401, a non-empty array must exactly match the
 *                          bank's explicit source-ID map. NULL/count==0 adopts
 *                          the bank-defined resident set.
 * @param resident_count Length of resident_experts.
 * @param max_batch      Maximum tokens per dispatch (M).
 * @return 0 on success, <0 on error.
 */
int sb_b70_load(sb_b70_provider_t* provider, const char* bank_path,
                uint64_t generation,
                const int32_t* resident_experts, size_t resident_count,
                size_t max_batch);

/**
 * Submit one MoE dispatch (non-blocking until sb_b70_take).
 *
 * @param provider    Handle.
 * @param generation  Must match the generation passed to sb_b70_load.
 * @param sequence    Caller-chosen monotonic id for this dispatch.
 * @param layer       Absolute layer index in [0, capability.num_layers).
 * @param hidden_fp16 Pointer to M * capability.hidden IEEE half values.
 * @param ids         Pointer to M * topk int32 values. Each must be in
 *                    [-1, resident_count). -1 means "skip this route".
 * @param weights     Pointer to M * topk float32 values.
 * @param M           Number of token rows.
 * @return 0 on success, >0 if busy, <0 on error.
 */
int sb_b70_issue(sb_b70_provider_t* provider, uint64_t generation,
                 uint64_t sequence, size_t layer,
                 const void* hidden_fp16,
                 const int32_t* ids, const float* weights, size_t M);

/**
 * Collect the result of a completed dispatch (blocks until ready).
 *
 * @param output          Pointer to M * 2048 float32 values (caller-allocated).
 * @param output_elements Must equal M * 2048.
 * @return 0 on success, <0 on error.
 */
int sb_b70_take(sb_b70_provider_t* provider, uint64_t generation,
                uint64_t sequence, float* output, size_t output_elements);

/** Number of resident experts per layer. */
size_t sb_b70_num_resident(sb_b70_provider_t* provider);

/**
 * Free and total bytes on the B70 the provider selected.
 *
 * The second card exists for capacity, so its occupancy is the number that
 * decides whether a given expert bank fits. Until this existed it was the
 * one resource in the system inferred from arithmetic rather than measured:
 * the 5090's VRAM, host DRAM and the KV cache were all reported, while the
 * B70's was computed as experts x bytes-per-expert and assumed.
 *
 * Returns 0 on success, -1 if the provider is null, not loaded, or the
 * runtime does not expose the free-memory aspect. Outputs are untouched on
 * failure. Either pointer may be null.
 */
int sb_b70_device_memory(sb_b70_provider_t* provider,
                         size_t* free_bytes, size_t* total_bytes);

/**
 * Snapshot provider health and copy its diagnostic string.
 *
 * @param health          Required output structure.
 * @param last_error      Optional caller-owned output buffer.
 * @param last_error_size Size of last_error in bytes. Passing NULL and zero
 *                        queries the required size through
 *                        health->last_error_bytes.
 * @return 0 on success, -1 for invalid arguments, -2 when last_error is too
 *         small. On -2 the buffer contains a NUL-terminated truncation and
 *         last_error_bytes names the full required size including NUL.
 */
int sb_b70_health(sb_b70_provider_t* provider, sb_b70_health_t* health,
                  char* last_error, size_t last_error_size);

/**
 * Native signal-driven poller.
 *
 * Tier 3 makes the CUDA side dispatch to the B70 entirely from within a
 * captured CUDA graph: the stream writes the batch size M to a
 * host-mapped signal flag (cuStreamWriteValue32) and then blocks on a
 * completion flag (cuStreamWaitValue32). Something on the host must
 * notice the signal and drive the provider.
 *
 * That watcher cannot live in Python. A yielding poll loop costs ~55us
 * per wakeup on the reference host — more than the B70 kernel itself,
 * and paid once per layer per token — while a spinning one holds the
 * GIL and starves the engine thread. This poller runs on a native
 * thread and never touches the interpreter.
 *
 * Failure semantics: the CUDA-side wait cannot time out, so the poller
 * raises each layer's completion flag even when its dispatch fails,
 * and records the failure for the caller to collect via
 * sb_b70_poll_error_count. Leaving a flag unset wedges the device
 * permanently and makes the process unkillable.
 */

/** Opaque poller handle. */
typedef void sb_b70_poller_t;

/**
 * Create a poller bound to @p provider. The provider must outlive it.
 * Returns NULL on failure.
 */
sb_b70_poller_t* sb_b70_poll_create(sb_b70_provider_t* provider,
                                    uint64_t generation);

/**
 * Register one layer's flags and pinned buffers.
 *
 * Safe to call while the poller is running. All pointers must remain
 * valid until sb_b70_poll_destroy.
 *
 * @param layer      Absolute layer index in [0, capability.num_layers).
 * @param signal     Host-mapped flag; the CUDA stream writes M here.
 * @param completion Host-mapped flag; set to 1 when the result is ready.
 * @param hidden     Pinned FP16 [max_batch, capability.hidden] activation.
 * @param ids        Pinned int32 [max_batch, topk] compact B70 slots.
 * @param weights    Pinned float32 [max_batch, topk] routing weights.
 * @param output     Pinned float32 [max_batch, capability.hidden] result.
 * @param topk       Routes per token.
 * @return 0 on success, <0 on error.
 */
int sb_b70_poll_register(sb_b70_poller_t* poller, size_t layer,
                         volatile uint32_t* signal,
                         volatile uint32_t* completion,
                         const void* hidden, const int32_t* ids,
                         const float* weights, float* output,
                         size_t topk);

/** Start the polling thread. Returns 0 on success, <0 on error. */
int sb_b70_poll_start(sb_b70_poller_t* poller);

/** Stop the polling thread and join it. */
void sb_b70_poll_stop(sb_b70_poller_t* poller);

/**
 * Atomically zero all counters. Safe while running; one in-flight dispatch
 * may straddle the reset boundary.
 */
void sb_b70_poll_reset(sb_b70_poller_t* poller);

/** Total dispatches served since start. */
uint64_t sb_b70_poll_dispatch_count(sb_b70_poller_t* poller);

/** Total rows (sum of M) served since start. */
uint64_t sb_b70_poll_row_count(sb_b70_poller_t* poller);

/**
 * Dispatch count in one M bucket.
 *
 * Buckets 0-6 are M=1, M=2, M=3-4, M=5-8, M=9-16, M=17-32, and M>32.
 * An out-of-range bucket returns 0.
 */
uint64_t sb_b70_poll_m_bucket_count(sb_b70_poller_t* poller, size_t bucket);

/** Number of failed dispatches; nonzero means results are untrustworthy. */
uint64_t sb_b70_poll_error_count(sb_b70_poller_t* poller);

/** Accumulated service time in nanoseconds, for exposed-wait accounting. */
uint64_t sb_b70_poll_service_ns(sb_b70_poller_t* poller);

/**
 * Accumulated profiled device-queue time in nanoseconds.
 *
 * Zero unless SHOOTING_BRAKE_B70_PROFILE=1. This spans the first queued
 * input copy through completion of the output copy. Level Zero copy engines
 * need not share a profiling timebase, so the raw span can underflow and is
 * not a valid decomposition boundary unless the caller first establishes
 * that the endpoint clocks are comparable. When they are, service - total
 * isolates host-side issue/take work.
 */
uint64_t sb_b70_poll_total_ns(sb_b70_poller_t* poller);

/**
 * Accumulated on-device kernel time in nanoseconds.
 *
 * Zero unless SHOOTING_BRAKE_B70_PROFILE=1, which puts the provider's queue
 * in profiling mode. Paired with sb_b70_poll_total_ns this splits profiled
 * device time into the kernel itself and the device-queue remainder.
 * Profiling adds two marker submissions per dispatch, so quantify its
 * perturbation with a separate unprofiled run.
 */
uint64_t sb_b70_poll_kernel_ns(sb_b70_poller_t* poller);

/** Stop if running, then release the handle. */
void sb_b70_poll_destroy(sb_b70_poller_t* poller);

/** Shut down and release device resources. */
void sb_b70_shutdown(sb_b70_provider_t* provider);

/** Destroy the provider handle. */
void sb_b70_destroy(sb_b70_provider_t* provider);

#ifdef __cplusplus
}
#endif
