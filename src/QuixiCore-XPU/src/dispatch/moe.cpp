// Dispatch layer for the moe family (native only).

#include "quixicore/xpu/ops.hpp"

#include "moe/moe_route/moe_route_kernel.hpp"
#include "moe/int4_moe/int4_moe_kernel.hpp"

#include "moe/nvfp4_moe/nvfp4_moe_kernel.hpp"

namespace quixicore::xpu::ops {

void moe_route_topk(sycl::queue& q, const void* router_logits, int* expert_ids,
                    float* expert_weights, std::size_t n_tokens,
                    std::size_t n_experts, int k, DType dt, Variant variant,
                    bool blocking) {
  (void)variant;
  sycl::event ev = kernels::moe_route_topk_sycl(q, router_logits, expert_ids,
                                                expert_weights, n_tokens,
                                                n_experts, k, dt);
  if (blocking) ev.wait();
}

void nvfp4_moe_fused(sycl::queue &q, const void *hidden, const int *topk_ids,
                     const float *topk_weights, const void *w13, const void *w13_scales,
                     const float *w13_global_scales, const void *w2, const void *w2_scales,
                     const float *w2_global_scales, float *out_f32, std::size_t M, std::size_t E,
                     std::size_t top_k, std::size_t K, std::size_t I, DType act_dt,
                     bool multiply_router_weight, Variant variant, bool blocking) {
  (void)variant;
  const sycl::event zeroed = q.memset(out_f32, 0, M * K * sizeof(float));
  sycl::event event = kernels::nvfp4_moe_fused_sycl(
      q, hidden, topk_ids, topk_weights, w13, w13_scales, w13_global_scales, w2, w2_scales,
      w2_global_scales, out_f32, M, E, top_k, K, I, multiply_router_weight, act_dt, zeroed);
  if (blocking)
    event.wait();
}

void nvfp4_moe_split(sycl::queue &q, const void *hidden, const int *topk_ids,
                     const float *topk_weights, const void *w13, const void *w13_scales,
                     const float *w13_global_scales, const void *w2, const void *w2_scales,
                     const float *w2_global_scales, float *scratch_f32, float *out_f32,
                     std::size_t M, std::size_t E, std::size_t top_k, std::size_t K, std::size_t I,
                     DType act_dt, bool multiply_router_weight, Variant variant, bool blocking) {
  (void)variant;
  const sycl::event zeroed = q.memset(out_f32, 0, M * K * sizeof(float));
  sycl::event event = kernels::nvfp4_moe_split_sycl(
      q, hidden, topk_ids, topk_weights, w13, w13_scales, w13_global_scales, w2, w2_scales,
      w2_global_scales, scratch_f32, out_f32, M, E, top_k, K, I, multiply_router_weight, act_dt,
      zeroed);
  if (blocking)
    event.wait();
}
void int4_moe_split(
    sycl::queue &q, const void *hidden, const int *topk_ids,
    const float *topk_weights, const std::int32_t *gate_qweight,
    const half_t *gate_scales, const std::int32_t *up_qweight,
    const half_t *up_scales, const std::int32_t *down_qweight,
    const half_t *down_scales, float *scratch_f32, float *out_f32,
    std::size_t expert_stride_bytes, std::size_t group_size, std::size_t M,
    std::size_t E, std::size_t top_k, std::size_t K, std::size_t I,
    DType act_dt, bool multiply_router_weight, Variant variant,
    bool blocking) {
  (void)variant;
  const sycl::event zeroed = q.memset(out_f32, 0, M * K * sizeof(float));
  sycl::event event = kernels::int4_moe_split_sycl(
      q, hidden, topk_ids, topk_weights, gate_qweight, gate_scales,
      up_qweight, up_scales, down_qweight, down_scales, scratch_f32, out_f32,
      expert_stride_bytes, group_size, M, E, top_k, K, I,
      multiply_router_weight, act_dt, zeroed);
  if (blocking)
    event.wait();
}


std::size_t nvfp4_moe_grouped_workspace_bytes(std::size_t M, std::size_t E,
                                              std::size_t top_k, std::size_t I,
                                              DType act_dt) {
  return kernels::nvfp4_moe_grouped_workspace_bytes(M, E, top_k, I, act_dt);
}

bool nvfp4_moe_grouped_profitable(std::size_t M, std::size_t E,
                                  std::size_t top_k, DType act_dt) {
  (void)M;
  (void)E;
  (void)top_k;
  (void)act_dt;
  // Always false, on purpose.
  //
  // The original heuristic returned true at >= 8 routes per expert, reasoning
  // that a grouped block stops being mostly padding there. No winning
  // crossover has actually been measured: at the one shape benchmarked on a
  // B70 (M=2048, E=256, top_k=8, K=2048, I=512, f16 -- 64 routes per expert)
  // the grouped kernel lost to nvfp4_moe_split. Other shapes and a larger row
  // block might behave differently; none have been tried.
  //
  // Until a crossover is measured, this returns false rather than routing
  // traffic on an untested guess -- a comment warning callers off does not
  // stop callers. The grouped path stays reachable by explicit call for
  // development and benchmarking. See the STATUS note in ops.hpp.
  return false;
}

void nvfp4_moe_grouped(sycl::queue &q, const void *hidden, const int *topk_ids,
                       const float *topk_weights, const void *w13,
                       const void *w13_scales, const float *w13_global_scales,
                       const void *w2, const void *w2_scales,
                       const float *w2_global_scales, void *workspace,
                       float *out_f32, std::size_t M, std::size_t E,
                       std::size_t top_k, std::size_t K, std::size_t I,
                       DType act_dt, bool multiply_router_weight,
                       Variant variant, bool blocking) {
  if (!kernels::nvfp4_moe_grouped_supported(act_dt)) {
    // f32 activations have no DPAS operand type. Falling back keeps the entry
    // point total rather than making every caller special-case a dtype.
    nvfp4_moe_fused(q, hidden, topk_ids, topk_weights, w13, w13_scales,
                    w13_global_scales, w2, w2_scales, w2_global_scales,
                    out_f32, M, E, top_k, K, I, act_dt,
                    multiply_router_weight, variant, blocking);
    return;
  }
  (void)variant;
  const sycl::event zeroed = q.memset(out_f32, 0, M * K * sizeof(float));
  sycl::event event = kernels::nvfp4_moe_grouped_sycl(
      q, hidden, topk_ids, topk_weights, w13, w13_scales, w13_global_scales, w2,
      w2_scales, w2_global_scales, workspace, out_f32, M, E, top_k, K, I,
      multiply_router_weight, act_dt, zeroed);
  if (blocking)
    event.wait();
}

}  // namespace quixicore::xpu::ops
