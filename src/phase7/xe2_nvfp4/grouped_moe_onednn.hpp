#pragma once
// oneDNN grouped-GEMM backend for the B70 NVFP4 prefill path.
//
// Kept in its own translation unit: it drags in dnnl.hpp and links libdnnl,
// none of which belongs in grouped_moe.cpp's cutlass compile. Selected at
// runtime with SB_GROUPED_BACKEND=onednn (default remains the native cute
// pipeline); any internal failure disarms the backend for the process and
// the caller's native launch takes over, so the provider contract is
// unchanged.
//
// Why it exists: NVFP4's group-16 scales pin the native cute pipeline to
// tile_k=16 (see grouped_moe.cpp), capping it at 2.355 ms/layer. oneDNN's
// grouped_gemm:micro:m_axis runs the same bank at 1.01 ms/layer at the
// production 30 rows/expert fill (gate-2 probe, 2026-08-25, this silicon).

#include <sycl/sycl.hpp>

#include <cstdint>

namespace sb::xe2 {

// True when SB_GROUPED_BACKEND=onednn and no runtime failure has disarmed
// the backend. Cheap; safe to call per dispatch.
bool onednn_grouped_armed() noexcept;

// One grouped GEMM leg: [rows, K] x [E, K, N] -> [rows, N], where the
// device-resident `offs_ends` (inclusive cumulative row ends, length
// `experts`) defines the per-expert rows -- no host sync. `act`/`out` are
// the provider's gathered chunk views with allocation capacity `rows_cap`
// rows; rows past the last offset are neither read nor written.
//
// `wgt` aliases the bank's packed plane [E, N, K/2] (E2M1 pairs, low nibble
// first) zero-copy. `scl_bank` is the bank's [E, N, K/16] e4m3 plane; the
// backend keeps a one-time repacked canonical copy per plane (oneDNN does
// not honor strided scale descriptors -- gate-2 probe).
//
// Submits asynchronously onto `q` (in-order), like the native launches.
// Returns false without touching `out` when the call cannot be served.
bool onednn_grouped_gemm(sycl::queue& q, const sycl::half* act,
                         const std::uint8_t* wgt,
                         const std::uint8_t* scl_bank, float* out,
                         const std::int32_t* offs_ends, int rows_cap,
                         int experts, int K, int N) noexcept;

}  // namespace sb::xe2
