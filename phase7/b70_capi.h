#pragma once

/**
 * Shooting Brake Phase 7 — C ABI for the B70 NVFP4 MoE provider.
 *
 * A thin C-compatible wrapper around the Phase 1 C++ B70Provider so it can be
 * called from Python via ctypes. All pointers are plain C pointers; the
 * ``hidden`` buffer is raw FP16 (IEEE half) bytes.
 *
 * Return convention: 0 = ok, >0 = busy/pending, <0 = error.
 */

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/** Opaque provider handle. */
typedef void sb_b70_provider_t;

/** Create a new provider instance. Returns NULL on failure. */
sb_b70_provider_t* sb_b70_create(void);

/**
 * Load the NVFP4 expert bank and select resident experts.
 *
 * @param provider       Handle from sb_b70_create.
 * @param bank_path      Path to phase1/expert_bank.bin.
 * @param generation     Placement generation id.
 * @param resident_experts  Array of global expert IDs that are B70-owned,
 *                          per layer (same set for all 32 NVFP4 layers).
 *                          If NULL or count==0, loads all 256 experts/layer.
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
 * @param layer       Absolute layer index (0-31 for NVFP4 layers).
 * @param hidden_fp16 Pointer to M * 2048 IEEE half values (row-major).
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

/** Shut down and release device resources. */
void sb_b70_shutdown(sb_b70_provider_t* provider);

/** Destroy the provider handle. */
void sb_b70_destroy(sb_b70_provider_t* provider);

#ifdef __cplusplus
}
#endif
