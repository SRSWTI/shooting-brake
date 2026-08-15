#pragma once

#include <cstddef>
#include <cstdint>


#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

// Device events for attributing one split call. The production and profiled
// entry points share the same implementation; profiling only copies the events
// returned by the two submissions.
struct Int4MoeSplitStages {
  sycl::event gate_up;
  sycl::event down;
};

// Decode-oriented per-route MoE for AutoGPTQ int4 weights. Within each expert,
// qweight is [K/8,N] int32 and scales are [K/group_size,N] fp16. The six
// pointers address their respective planes in expert 0;
// `expert_stride_bytes` advances any plane to the next AoS expert record.
// The effective zero point is the format constant 8; no qzeros pointer is
// accepted. Scratch is fp32
// [M*top_k,I] and holds the post-SwiGLU route activations.
sycl::event int4_moe_split_sycl(
    sycl::queue &q, const void *hidden, const int *topk_ids,
    const float *topk_weights, const std::int32_t *gate_qweight,
    const half_t *gate_scales, const std::int32_t *up_qweight,
    const half_t *up_scales, const std::int32_t *down_qweight,
    const half_t *down_scales, float *scratch_f32, float *out_f32,
    std::size_t expert_stride_bytes, std::size_t group_size, std::size_t M,
    std::size_t E, std::size_t top_k, std::size_t K, std::size_t I,
    bool multiply_router_weight, DType act_dt,
    const sycl::event &output_ready);

sycl::event int4_moe_split_profiled_sycl(
    sycl::queue &q, const void *hidden, const int *topk_ids,
    const float *topk_weights, const std::int32_t *gate_qweight,
    const half_t *gate_scales, const std::int32_t *up_qweight,
    const half_t *up_scales, const std::int32_t *down_qweight,
    const half_t *down_scales, float *scratch_f32, float *out_f32,
    std::size_t expert_stride_bytes, std::size_t group_size, std::size_t M,
    std::size_t E, std::size_t top_k, std::size_t K, std::size_t I,
    bool multiply_router_weight, DType act_dt,
    const sycl::event &output_ready, Int4MoeSplitStages *stages);
// Benchmark-only down-projection geometry experiment. Each lane owns 2 or 4
// adjacent-transaction output columns and reuses each activated fp32 vector
// load across them. The gate/up stage is identical to int4_moe_split_sycl.
sycl::event int4_moe_split_down_wide_sycl(
    sycl::queue &q, const void *hidden, const int *topk_ids,
    const float *topk_weights, const std::int32_t *gate_qweight,
    const half_t *gate_scales, const std::int32_t *up_qweight,
    const half_t *up_scales, const std::int32_t *down_qweight,
    const half_t *down_scales, float *scratch_f32, float *out_f32,
    std::size_t expert_stride_bytes, std::size_t group_size, std::size_t M,
    std::size_t E, std::size_t top_k, std::size_t K, std::size_t I,
    bool multiply_router_weight, DType act_dt, int outputs_per_lane,
    const sycl::event &output_ready, Int4MoeSplitStages *stages = nullptr);



} // namespace quixicore::xpu::kernels
