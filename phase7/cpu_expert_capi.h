#pragma once

/**
 * Shooting Brake — C ABI for the CPU DDR5 expert tier.
 *
 * The third residency tier, below CUDA (hot) and B70 (warm). Experts that
 * fire rarely live here: their weights sit in hugepage-backed host DRAM and
 * are computed on CPU cores, freeing both GPUs' VRAM for hotter experts and
 * KV cache.
 *
 * Cost model (measured, not estimated). A decode-shaped expert pass (M=1)
 * reads every weight exactly once and reuses none of them, so it is a pure
 * DRAM stream: latency is ``expert_bytes / achievable_bandwidth`` and compute
 * never enters the picture. For the qualified model (hidden=2048,
 * intermediate=768 -- 9.0 MiB of bf16 per expert) on the reference host
 * (Core Ultra 9 285K, DDR5), phase7/cpu_expert_bench.py measures:
 *
 *      1 thread   -> 1340 us  @  7.0 GB/s
 *      2 threads  ->  690 us  @ 13.7 GB/s
 *      8 threads  ->  287 us  @ 32.8 GB/s
 *     12 threads  ->  195 us  @ 48.3 GB/s
 *
 * Two things follow. Thread count is not a tuning knob but a requirement --
 * single-threaded, this tier is nearly 7x slower and unusable. And the
 * achieved ceiling is ~48 GB/s, well under DDR5's theoretical peak, so plan
 * against the measured figure.
 *
 * The B70 reads the same weights from ~450 GB/s VRAM in roughly 40us, making
 * this tier ~5x slower. That ratio is the whole reason it is the *cold* tier:
 * placement must keep frequently-routed experts off it. A hot expert parked
 * here is a latency bug, not a capacity win.
 *
 * Weight layout follows PyTorch ``nn.Linear`` (``[out_features, in_features]``)
 * so no transpose is needed at load time, and each output row's dot product
 * walks contiguous memory -- the right access pattern for the M=1 GEMV that
 * dominates decode.
 *
 * Storage is bf16, accumulation is fp32. bf16 -> fp32 is a 16-bit shift, so
 * the conversion is free relative to the DRAM read it rides along with.
 *
 * Return convention: 0 = ok, <0 = error.
 */

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/** Opaque CPU expert host handle. */
typedef void sb_cpu_host_t;

/**
 * Create a CPU expert host.
 *
 * Reserves a hugepage-backed arena sized for @p max_experts experts. The
 * arena is virtual until touched, so oversizing costs address space, not
 * resident pages.
 *
 * @param num_layers   Total MoE layers in the model (arena index bound).
 * @param num_experts  Routed experts per layer (arena index bound).
 * @param hidden       Model hidden size.
 * @param intermediate Expert FFN intermediate size.
 * @param max_experts  Number of (layer, expert) pairs to reserve space for.
 * @param num_threads  Worker threads for FFN compute. 0 selects a default.
 * @return Handle, or NULL on failure.
 */
sb_cpu_host_t* sb_cpu_create(size_t num_layers, size_t num_experts,
                             size_t hidden, size_t intermediate,
                             size_t max_experts, size_t num_threads);

/**
 * Load one expert's bf16 weights into the arena.
 *
 * Copies are made; the caller's buffers may be freed immediately after.
 * Re-loading an already-resident (layer, expert) overwrites it in place
 * and does not consume additional arena space.
 *
 * @param gate_bf16 bf16 [intermediate, hidden].
 * @param up_bf16   bf16 [intermediate, hidden].
 * @param down_bf16 bf16 [hidden, intermediate].
 * @return 0 on success, <0 on error (arena exhausted, bad index).
 */
int sb_cpu_load_expert(sb_cpu_host_t* host, size_t layer, size_t expert,
                       const void* gate_bf16, const void* up_bf16,
                       const void* down_bf16);

/** 1 if (layer, expert) is CPU-resident, 0 otherwise. */
int sb_cpu_has_expert(sb_cpu_host_t* host, size_t layer, size_t expert);

/**
 * Run one expert's SwiGLU FFN over M token rows.
 *
 *   g = x @ gate^T ; u = x @ up^T ; y = (silu(g) * u) @ down^T
 *
 * @param input_bf16 bf16 [M, hidden].
 * @param output_f32 fp32 [M, hidden]; overwritten, not accumulated.
 * @return 0 on success, <0 if the expert is not resident.
 */
int sb_cpu_expert_forward(sb_cpu_host_t* host, size_t layer, size_t expert,
                          const void* input_bf16, float* output_f32, size_t M);

/**
 * Run a full routed-expert batch and return the routing-weighted partial.
 *
 * Mirrors the B70 provider's contract so the two tiers join identically on
 * the CUDA side: @p output_f32 is zero-initialised, every resident route
 * contributes ``weight * FFN(token)``, and routes whose expert is not
 * CPU-resident are skipped -- the caller's partition owns that decision, and
 * a route landing here by mistake must not silently vanish, so it is counted
 * in sb_cpu_skipped_routes.
 *
 * @param hidden_bf16 bf16 [M, hidden].
 * @param expert_ids  int32 [M, topk] global expert ids; -1 skips the route.
 * @param weights     fp32 [M, topk] routing weights.
 * @param output_f32  fp32 [M, hidden] accumulated weighted partial.
 * @return 0 on success, <0 on error.
 */
int sb_cpu_moe_forward(sb_cpu_host_t* host, size_t layer,
                       const void* hidden_bf16,
                       const int32_t* expert_ids, const float* weights,
                       size_t M, size_t topk, float* output_f32);

/** Bytes actually committed in the arena. */
size_t sb_cpu_arena_used(sb_cpu_host_t* host);

/** Bytes reserved for the arena. */
size_t sb_cpu_arena_capacity(sb_cpu_host_t* host);

/** Number of resident (layer, expert) pairs. */
size_t sb_cpu_resident_count(sb_cpu_host_t* host);

/** Routes dropped because their expert was not resident; must stay 0. */
uint64_t sb_cpu_skipped_routes(sb_cpu_host_t* host);

/**
 * Signal-driven poller ("all-out mode" only).
 *
 * Mirrors the B70 poller so both tiers dispatch identically: the CUDA stream
 * writes the batch size M into a host-mapped signal flag
 * (cuStreamWriteValue32) and then blocks on a completion flag
 * (cuStreamWaitValue32), and a native thread on the host notices the signal
 * and runs the work. Because every step on the CUDA side is a stream
 * operation, the whole round trip is captured by torch.cuda.graph().
 *
 * The watcher cannot live in Python: a yielding poll loop costs ~55us per
 * wakeup and a spinning one holds the GIL and starves the engine thread.
 *
 * Unlike the B70 poller this one backs off to a short sleep rather than
 * spinning hot. The service time here is ~195us, so a few microseconds of
 * wakeup latency is noise, and a spinning watcher would steal a core from
 * the FFN thread pool that actually needs it.
 *
 * Failure semantics are identical and non-negotiable: cuStreamWaitValue32
 * has no timeout, so the completion flag is raised even when the dispatch
 * fails. Leaving it unset wedges the device permanently and makes the
 * process unkillable. Failures are counted for the caller to collect.
 */

/** Opaque poller handle. */
typedef void sb_cpu_poller_t;

/**
 * Create a poller bound to @p host. The host must outlive it.
 * Returns NULL on failure.
 */
sb_cpu_poller_t* sb_cpu_poll_create(sb_cpu_host_t* host);

/**
 * Register one layer's flags and pinned buffers.
 *
 * Safe to call while the poller is running. All pointers must remain valid
 * until sb_cpu_poll_destroy.
 *
 * @param layer      Absolute layer index.
 * @param signal     Host-mapped flag; the CUDA stream writes M here.
 * @param completion Host-mapped flag; set to 1 when the result is ready.
 * @param hidden     Pinned bf16 [max_batch, hidden] activation.
 * @param ids        Pinned int32 [max_batch, topk] global expert ids.
 * @param weights    Pinned float32 [max_batch, topk] routing weights.
 * @param output     Pinned float32 [max_batch, hidden] result target.
 * @param topk       Routes per token.
 * @return 0 on success, <0 on error.
 */
int sb_cpu_poll_register(sb_cpu_poller_t* poller, size_t layer,
                         volatile uint32_t* signal,
                         volatile uint32_t* completion,
                         const void* hidden, const int32_t* ids,
                         const float* weights, float* output, size_t topk);

/** Start the polling thread. Returns 0 on success, <0 on error. */
int sb_cpu_poll_start(sb_cpu_poller_t* poller);

/** Stop the polling thread and join it. */
void sb_cpu_poll_stop(sb_cpu_poller_t* poller);

/** Total dispatches served since start. */
uint64_t sb_cpu_poll_dispatch_count(sb_cpu_poller_t* poller);

/** Number of failed dispatches; nonzero means results are untrustworthy. */
uint64_t sb_cpu_poll_error_count(sb_cpu_poller_t* poller);

/** Accumulated service time in nanoseconds, for exposed-wait accounting. */
uint64_t sb_cpu_poll_service_ns(sb_cpu_poller_t* poller);

/** Stop if running, then release the handle. */
void sb_cpu_poll_destroy(sb_cpu_poller_t* poller);

/** Release the arena and join worker threads. */
void sb_cpu_destroy(sb_cpu_host_t* host);

#ifdef __cplusplus
}
#endif
