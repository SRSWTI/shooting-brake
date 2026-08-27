#pragma once
// Grouped NVFP4 MoE for the B70 prefill path.
//
// Kept in its own translation unit on purpose: the implementation drags in
// cute/cutlass-sycl and needs three SPIR-V extensions on the command line, and
// none of that belongs in b70_provider.cpp's compile.
//
// Why it exists: the per-route GEMV reads an expert's whole 5.06 MiB once PER
// ROUTE. At M=2560 routes over 85 resident experts that is ~30x the traffic the
// weights require, and the GEMV is already at 437 GB/s of the card's ~510 -- so
// the waste is bytes, not rate. This reads each expert once per layer.
//
// Measured standalone at r15 geometry on real bank bytes, 2026-08-21:
// 2.355 ms/layer = 216 us/token against 1,705 us/token for the GEMV path in
// vLLM, i.e. ~7.9x on the B70 GEMM leg.
//
// Decode is NOT a target: at M=1 each token already touches its experts once,
// so there is no amplification to remove.

#include <sycl/sycl.hpp>

#include <cstdint>

namespace sb::xe2 {

// Returns false if the shape is one this path does not serve, in which case the
// caller must fall back to the GEMV. Never throws across the boundary.
bool grouped_moe_nvfp4(sycl::queue& q,
                       const sycl::half* act_src,      // [M, hidden]
                       const std::int32_t* ids,        // [M, top_k]
                       const float* route_w,           // [M, top_k]
                       const std::uint8_t* w13,        // [E, 2I, H/2]
                       const std::uint8_t* s13,        // [E, 2I, H/16] e4m3
                       const float* alpha13,           // [E]
                       const std::uint8_t* w2,         // [E, H, I/2]
                       const std::uint8_t* s2,         // [E, H, I/16] e4m3
                       const float* alpha2,            // [E]
                       sycl::half* g_act, float* g_mid,
                       sycl::half* g_gated, float* g_outr,
                       const sycl::half* bias13, const sycl::half* bias2,
                       std::int32_t* hist, std::int32_t* offs,
                       std::int32_t* cursor, std::int32_t* rows,
                       std::int32_t* slot_row, std::int32_t* slot_exp,
                       float* slot_w,
                       std::int32_t* slot_of,  // [M*top_k] route -> slot, -1 if unowned
                       std::int32_t* atom,
                       float* out,                     // [M, hidden]
                       int M, int experts, int top_k, int hidden, int inter,
                       int group_size,
                       int rows_cap  // g_* chunk allocation capacity, routes
                       ) noexcept;

}  // namespace sb::xe2
