#ifndef COLIBRI_QWEN36_CUDA_H
#define COLIBRI_QWEN36_CUDA_H

#include <stddef.h>
#include <stdint.h>
#include "backend_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const void *weights;
    const float *scales;
    int fmt, input, output, group_size;
} QwenCudaWeight;

typedef struct {
    int is_attention;
    QwenCudaWeight q, k, v, o, router;
    QwenCudaWeight shared_gate, shared_up, shared_down;
    QwenCudaWeight dn_qkv, dn_z, dn_b, dn_a, dn_out;
    const float *input_norm, *post_norm;
    const float *q_norm, *k_norm, *router_bias, *shared_gate_weight;
    const float *dn_conv, *dn_dt_bias, *dn_a_log, *dn_norm;
} QwenCudaLayerWeights;

typedef struct {
    int hidden, layers, vocab;
    int q_heads, kv_heads, head_dim, q_head_dim, k_head_dim, rotary_dim;
    int experts, topk, groups, topk_groups, shared_intermediate;
    int dn_value_heads, dn_key_heads, dn_key_dim, dn_value_dim;
    int dn_conv_kernel, dn_conv_dim;
    float rms_eps, rope_theta;
} QwenCudaConfig;

typedef int (*QwenCudaCpuRoutes)(void *opaque, int layer, const int *eids,
                                 const float *weights, int K, uint32_t mask,
                                 const float *activation, float *partial);
typedef void (*QwenCudaRouteCommit)(void *opaque, int layer, const int *eids,
                                    const float *weights, int K);

typedef struct QwenCudaState QwenCudaState;

/* Creates all dense handles, persistent recurrent state and fixed execution
 * scratch. CUDA must be requested through COLI_CUDA=1. `initial_kv_capacity`
 * is reserved before the expert tier computes its remaining-memory budget. */
QwenCudaState *qwen_cuda_create(const QwenCudaConfig *cfg,
        const QwenCudaLayerWeights *layers,
        const QwenCudaWeight *embedding,const QwenCudaWeight *lm_head,
        const float *final_norm,int initial_kv_capacity,
        QwenCudaCpuRoutes cpu_routes,QwenCudaRouteCommit route_commit,
        void *opaque);
void qwen_cuda_destroy(QwenCudaState *state);
int qwen_cuda_ready(const QwenCudaState *state);
size_t qwen_cuda_reserved_bytes(const QwenCudaState *state);

int qwen_cuda_reset(QwenCudaState *state);
int qwen_cuda_ensure_kv(QwenCudaState *state,int capacity);
/* One whole-step transaction. On success, logits_host contains the last row and
 * greedy_id is selected on the GPU. Failure rolls device state back internally. */
int qwen_cuda_step(QwenCudaState *state,const int *ids,int S,int pos_base,
                   float *logits_host,int *greedy_id);

/* Emergency failover mirrors. Export is valid only after qwen_cuda_step has
 * rolled back a failed transaction or at a committed boundary. */
int qwen_cuda_export_state(QwenCudaState *state,float **k_host,float **v_host,
                           float **dn_rec_host,float **dn_conv_host,
                           int kv_len,int host_kv_capacity);
int qwen_cuda_import_state(QwenCudaState *state,float **k_host,float **v_host,
                           float **dn_rec_host,float **dn_conv_host,
                           int kv_len,int host_kv_capacity);
int qwen_cuda_kv_len(const QwenCudaState *state);
void qwen_cuda_disable_request(QwenCudaState *state);
int qwen_cuda_request_enabled(const QwenCudaState *state);

#ifdef __cplusplus
}
#endif
#endif
