/* b70_tier.h — B70 expert tier interface for qwen36 */
#ifndef B70_TIER_H
#define B70_TIER_H

#include <stdint.h>

/* Returns 1 if B70 tier is active */
int  b70_tier_on(void);

/* Init: dlopen libb70_moe.so and allocate the configured compact expert bank.
 * Ownership is assigned later from the shared heat-ordered placement plan. */
int  b70_tier_init(int nl, int ne, int D, int Ih, int topk, int egs);

/* Convert one Colibri signed-S4 expert to llm-scaler IPEX K-major:
 * qweight [K/8,N] with offset-binary marlin-shuffled nibbles and
 * scale [K/group,N] in FP16, preserving the original group size. */
int b70_convert_expert_s4(const uint8_t *gate, const uint8_t *up,
                          const uint8_t *down, const float *gate_scales,
                          const float *up_scales, const float *down_scales,
                          int hidden, int intermediate, int group_size,
                          uint32_t *gate_up_out, uint16_t *gate_up_scales_out,
                          uint32_t *down_out, uint16_t *down_scales_out);

/* Convert and upload one resident expert without dequantizing its weights. */
void b70_tier_note(int layer, int eid,
                   const uint8_t *g4, const uint8_t *u4, const uint8_t *d4,
                   const float *gs, const float *us, const float *ds);

/* Claim/query compact B70 ownership after hotter CUDA placements are reserved. */
int b70_tier_claim(int layer, int eid);
int b70_tier_capacity(void);
int b70_tier_is_resident(int layer, int eid);

/* During qt_issue: enqueue ready B70 experts and set their mask bits. */
void b70_tier_record(int layer, const int *eids, const float *weights,
                     int K, const float *x, uint32_t *mask);

/* Dispatch and accumulate; returns route bits that require CPU recomputation. */
uint32_t b70_tier_collect(const float *val, int K, float *out, int D);
void b70_tier_stats(void);
void b70_tier_shutdown(void);

#endif
