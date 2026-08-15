#pragma once

#include <cstddef>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

sycl::event nvfp4_moe_fused_sycl(sycl::queue &q, const void *hidden, const int *topk_ids,
                                 const float *topk_weights, const void *w13, const void *w13_scales,
                                 const float *w13_global_scales, const void *w2,
                                 const void *w2_scales, const float *w2_global_scales,
                                 float *out_f32, std::size_t M, std::size_t E, std::size_t top_k,
                                 std::size_t K, std::size_t I, bool multiply_router_weight,
                                 DType act_dt, const sycl::event &output_ready);

sycl::event nvfp4_moe_split_sycl(sycl::queue &q, const void *hidden, const int *topk_ids,
                                 const float *topk_weights, const void *w13, const void *w13_scales,
                                 const float *w13_global_scales, const void *w2,
                                 const void *w2_scales, const float *w2_global_scales,
                                 float *scratch_f32, float *out_f32, std::size_t M, std::size_t E,
                                 std::size_t top_k, std::size_t K, std::size_t I,
                                 bool multiply_router_weight, DType act_dt,
                                 const sycl::event &output_ready);

// Grouped form: routes are sorted by expert, then each expert's rows run as a
// DPAS GEMM whose dequantized weight tiles are reused across 32 rows. Wins
// once tokens actually share experts (roughly M*top_k/E >= 8); at decode
// widths the per-route kernels above are still the right shape.
//
// `workspace` must be at least `nvfp4_moe_grouped_workspace_bytes(...)` and
// need not be initialized. Returns an empty event for act_dt the DPAS path
// cannot express (see `nvfp4_moe_grouped_supported`).
sycl::event nvfp4_moe_grouped_sycl(
    sycl::queue &q, const void *hidden, const int *topk_ids,
    const float *topk_weights, const void *w13, const void *w13_scales,
    const float *w13_global_scales, const void *w2, const void *w2_scales,
    const float *w2_global_scales, void *workspace, float *out_f32,
    std::size_t M, std::size_t E, std::size_t top_k, std::size_t K,
    std::size_t I, bool multiply_router_weight, DType act_dt,
    const sycl::event &output_ready);

// Per-stage events from one grouped call, for attributing its cost. Populated
// only by nvfp4_moe_grouped_profiled_sycl; the production entry above does not
// pay for it.
//
// Every submission the call makes is here, housekeeping included -- a clear or
// join that turns out to be expensive must not get misattributed to a GEMM.
// Note the caller zeroes out_f32 before entry, so that cost sits outside.
struct Nvfp4MoeGroupedStages {
  sycl::event clear_counts;
  sycl::event clear_sorted;
  sycl::event histogram;
  sycl::event scan;
  sycl::event join;
  sycl::event scatter;
  sycl::event gate_up;
  sycl::event down;
};

// Benchmark-only. Same submissions, same order, same dependencies as
// nvfp4_moe_grouped_sycl -- it is literally the same code path with the events
// handed back -- so a profiling-enabled queue can time each stage without the
// measurement diverging from what production runs. Requires the queue to have
// been created with enable_profiling.
sycl::event nvfp4_moe_grouped_profiled_sycl(
    sycl::queue &q, const void *hidden, const int *topk_ids,
    const float *topk_weights, const void *w13, const void *w13_scales,
    const float *w13_global_scales, const void *w2, const void *w2_scales,
    const float *w2_global_scales, void *workspace, float *out_f32,
    std::size_t M, std::size_t E, std::size_t top_k, std::size_t K,
    std::size_t I, bool multiply_router_weight, DType act_dt,
    const sycl::event &output_ready, Nvfp4MoeGroupedStages *stages);

std::size_t nvfp4_moe_grouped_workspace_bytes(std::size_t M, std::size_t E,
                                              std::size_t top_k, std::size_t I,
                                              DType act_dt);

bool nvfp4_moe_grouped_supported(DType act_dt);

} // namespace quixicore::xpu::kernels
