#include "grouped_moe.hpp"

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

#include "cutlass/kernel_hardware_info.h"
#include "xe_2/gemm_xe2_policy.hpp"
#include "xe_2/grouped_gemm_xe2.hpp"
#include "grouped_moe_onednn.hpp"

using namespace cute;

namespace sb::xe2 {
namespace {

template <typename, typename, typename, char, char, class, MoE::A_DTYPE,
          MoE::B_DTYPE>
class GroupedName;

// Faithful copy of the vendored MoEGEMMLauncher minus its torch dependency.
// Divergence here would silently change what we measured, so it is kept
// identical: same ranges, same sub_group_size<16>/grf_size<256>, same entry.
template <char layoutA, char layoutB, class policy, MoE::A_DTYPE ADT,
          MoE::B_DTYPE BDT, typename ElementA, typename ElementB,
          typename ElementS, typename ElementBI, typename ElementD>
void launch(sycl::queue& q, const ElementA* act, const ElementB* wgt,
            const ElementS* scl, const ElementBI* bias, ElementD* out,
            int gemm_n, int gemm_k, const int* rows_per_expert, int num_experts,
            int group_size, std::int32_t* atomic_buffer) {
  using ElementA_non_CV = cutlass::platform::remove_cv_t<ElementA>;
  auto op = XE_DPAS_TT<8, float, ElementA_non_CV>{};
  using WGTile = typename policy::WGTile;
  using SGLayout = typename policy::SGLayout;
  using MMA = typename TiledMMAHelper<MMA_Atom<decltype(op)>, Layout<WGTile>,
                                      SGLayout>::TiledMMA;
  auto mma = MMA{};
  const int sm =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  const auto tpw = size(mma);
  sycl::range<3> local(1, 1, tpw);
  sycl::range<3> global(1, sm * 512 / tpw, 1);
  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  syclex::properties props{syclex::sub_group_size<16>, intelex::grf_size<256>};
  using CopyA = typename policy::GmemTiledCopyA;
  using CopyB = typename policy::GmemTiledCopyB;
  using CopyD = typename policy::GmemTiledCopyD;
  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<std::int32_t, 1> lm(sycl::range<1>(1), cgh);
    cgh.parallel_for<GroupedName<ElementA, ElementB, ElementD, layoutA, layoutB,
                                 policy, ADT, BDT>>(
        sycl::nd_range<3>{global * local, local}, props, [=](auto) {
          MoE::MoEGEMM<ADT, BDT, CopyA, CopyB, CopyD, layoutA, layoutB, 'R'>(
              act, wgt, scl, bias, out, mma, rows_per_expert, num_experts,
              group_size, gemm_n, gemm_k, atomic_buffer, lm);
        });
  });
}

using HalfT = cutlass::half_t;
using E2M1 = cutlass::float_e2m1_t;

// layoutB must be 'C'. The kernel INVERTS it -- actual_layout_of_B is
// LayoutKindB ^ ('R' ^ 'C') -- so passing 'R' makes B column-major, which puts
// two adjacent N values in one packed byte instead of two K values. Verified
// 2026-08-21: with 'R' every weight decoded to the low nibble; with 'C' the
// probe output alternates correctly. Our bank is [N, K/2] with K contiguous.
constexpr char kLayoutA = 'R';
constexpr char kLayoutB = 'C';

// tile_k must equal group_size for NVFP4. The scale reload is gated on
// `k_tile * tile_k % group_size == 0`, so a tile_k of 32 spanning two
// 16-element groups loads only the first group's scale and applies it to all
// 32 values -- wrong, and silently so. Costs 4.6% against the tile_k=32 tile.
//
// WHICH tile_k=16 policy wins is M-dependent, and the crossover sits inside
// the range SHOOTING_BRAKE_B70_MAX_BATCH moves us across. Measured 2026-08-21,
// E=85, N=1024, K=3072, group 16, this silicon:
//
//   rows/expert | m_32_k16 (32x64x16) | k16 (128x256x16)
//   ------------|---------------------|------------------
//        30     | 429 GB/s   <- wins  | 122 GB/s
//       120     | 116 GB/s            | 164 GB/s  <- wins
//
// A 128-row tile swallows an expert's whole row set in one pass once
// rows/expert clears ~64; below that it runs mostly padding. Pinning either
// one statically throws away ~1.4x at the end it is wrong for -- which is what
// shipping m_32_k16 did the moment MAX_BATCH went 256 -> 2048.
using PolicySmallM = MoE::w4a16_policy_m_32_k16;
// The big-tile arm never actually got a big tile: k16_d32 lost 7% when it
// was pinned GLOBALLY (pre-dispatch-branch, so it also ran the 30-row
// fills it is wrong for). With the >=64 rows/expert branch below it only
// ever sees fat fills, where the probe table above says the 128-row tile
// wins 1.41x. SB_GROUPED_BIGM selects the arm for the re-bench:
//   m32 (default, shipped) | m64 | m64n128 | m128 | d32
using PolicyBigM = MoE::w4a16_policy_m_32_k16;
using PolicyBigM64 = MoE::w4a16_policy_m_64_k16;
using PolicyBigM64N128 = MoE::w4a16_policy_m_64_n128_k16;
using PolicyBigM128 = MoE::w4a16_policy_m_128_k16;
using PolicyBigD32 = MoE::w4a16_policy_k16_d32;
enum class BigMArm { kM32, kM64, kM64N128, kM128, kD32 };
inline BigMArm bigm_arm() {
  static const BigMArm arm = [] {
    const char* v = std::getenv("SB_GROUPED_BIGM");
    if (v == nullptr) return BigMArm::kM32;
    if (std::strcmp(v, "m64") == 0) return BigMArm::kM64;
    if (std::strcmp(v, "m64n128") == 0) return BigMArm::kM64N128;
    if (std::strcmp(v, "m128") == 0) return BigMArm::kM128;
    if (std::strcmp(v, "d32") == 0) return BigMArm::kD32;
    return BigMArm::kM32;
  }();
  return arm;
}
constexpr int kBigTileRowsPerExpert = 64;

// Only the policy varies between the two GEMMs, so bind the rest once.
template <class policy>
inline void launch_nvfp4(sycl::queue& q, const sycl::half* a,
                         const std::uint8_t* w, const std::uint8_t* s,
                         const sycl::half* bias, float* d, int n, int k,
                         const std::int32_t* rows, int e, int gs,
                         std::int32_t* atom) {
  launch<kLayoutA, kLayoutB, policy, MoE::A_DTYPE::BITS16,
         MoE::B_DTYPE::NVFP4, HalfT, E2M1, std::uint8_t, HalfT, float>(
      q, reinterpret_cast<const HalfT*>(a), reinterpret_cast<const E2M1*>(w), s,
      reinterpret_cast<const HalfT*>(bias), d, n, k, rows, e, gs, atom);
}

// The big-tile arm is env-selected (SB_GROUPED_BIGM) for the re-bench;
// unused arms only JIT if selected, so the default costs nothing extra.
inline void launch_nvfp4_big(sycl::queue& q, const sycl::half* a,
                             const std::uint8_t* w, const std::uint8_t* s,
                             const sycl::half* bias, float* d, int n, int k,
                             const std::int32_t* rows, int e, int gs,
                             std::int32_t* atom) {
  switch (bigm_arm()) {
    case BigMArm::kM64:
      launch_nvfp4<PolicyBigM64>(q, a, w, s, bias, d, n, k, rows, e, gs, atom);
      return;
    case BigMArm::kM64N128:
      launch_nvfp4<PolicyBigM64N128>(q, a, w, s, bias, d, n, k, rows, e, gs,
                                     atom);
      return;
    case BigMArm::kM128:
      launch_nvfp4<PolicyBigM128>(q, a, w, s, bias, d, n, k, rows, e, gs,
                                  atom);
      return;
    case BigMArm::kD32:
      launch_nvfp4<PolicyBigD32>(q, a, w, s, bias, d, n, k, rows, e, gs, atom);
      return;
    default:
      launch_nvfp4<PolicyBigM>(q, a, w, s, bias, d, n, k, rows, e, gs, atom);
      return;
  }
}

// MXFP4 arm: same E2M1 payload, e8m0 scales at group 32 (tile_k=32 legal
// again -- the whole speed argument). Scales are DERIVED from the bank's
// e4m3/16 planes and are approximate: e8m0 has no mantissa and one scale
// covers two 16-blocks, so this arm is quality-gated before any default
// flip; the bank itself is untouched.
template <class policy>
inline void launch_mxfp4(sycl::queue& q, const sycl::half* a,
                         const std::uint8_t* w, const std::uint8_t* s_e8m0,
                         const sycl::half* bias, float* d, int n, int k,
                         const std::int32_t* rows, int e,
                         std::int32_t* atom) {
  launch<kLayoutA, kLayoutB, policy, MoE::A_DTYPE::BITS16,
         MoE::B_DTYPE::MXFP4, HalfT, E2M1, std::uint8_t, HalfT, float>(
      q, reinterpret_cast<const HalfT*>(a), reinterpret_cast<const E2M1*>(w),
      s_e8m0, reinterpret_cast<const HalfT*>(bias), d, n, k, rows, e,
      /*group_size=*/32, atom);
}

using PolicyMxSmallM = MoE::w4a16_policy_m_32;
using PolicyMxBigM = MoE::w4a16_policy_m_32;  // d32 128-tile: measure first

// Which backend serves the GEMM legs. Parsed once; the oneDNN arm keeps its
// own disarm latch on top of this.
enum class Backend { kNative, kOnednn, kMxfp4 };

Backend backend() {
  static const Backend b = [] {
    const char* v = std::getenv("SB_GROUPED_BACKEND");
    if (v != nullptr && std::strcmp(v, "mxfp4") == 0) {
      std::fprintf(stderr, "[sb.grouped] backend=mxfp4 armed\n");
      return Backend::kMxfp4;
    }
    if (v != nullptr && std::strcmp(v, "onednn") == 0) {
      return Backend::kOnednn;  // grouped_moe_onednn.cpp logs its own arm
    }
    return Backend::kNative;
  }();
  return b;
}

// Bank e4m3/16 scale plane -> derived e8m0/32 plane, [E, N, K/32] u8,
// computed once per plane on the caller's in-order queue and cached for the
// life of the process (exactly the bank's lifetime). Geometric mean of the
// two covered e4m3 scales, rounded to the nearest power of two: the
// log-symmetric choice, so the two halves of a 32-block split the error.
const std::uint8_t* mxfp4_scale_plane(sycl::queue& q,
                                      const std::uint8_t* bank_e4m3, int E,
                                      int N, int kg16) {
  static std::mutex mu;
  static std::unordered_map<const void*, std::uint8_t*> cache;
  std::lock_guard<std::mutex> lock(mu);
  auto it = cache.find(bank_e4m3);
  if (it != cache.end()) {
    return it->second;
  }
  const int kg32 = kg16 / 2;
  std::uint8_t* plane = sycl::malloc_device<std::uint8_t>(
      static_cast<std::size_t>(E) * N * kg32, q);
  if (plane == nullptr) {
    return nullptr;  // caller falls back to the native launch
  }
  q.parallel_for(
      sycl::range<3>(static_cast<std::size_t>(E), static_cast<std::size_t>(N),
                     static_cast<std::size_t>(kg32)),
      [=](sycl::id<3> idx) {
        const std::size_t e = idx[0], n = idx[1], g = idx[2];
        const std::uint8_t* src = bank_e4m3 + (e * N + n) * kg16 + 2 * g;
        const auto dec = [](std::uint8_t v) {
          const int ex = (v >> 3) & 0xF;
          const int mn = v & 0x7;
          const float mag =
              ex == 0 ? sycl::ldexp(static_cast<float>(mn) / 8.0f, -6)
                      : sycl::ldexp(1.0f + static_cast<float>(mn) / 8.0f,
                                    ex - 7);
          // Scales are magnitudes; a sign bit here would be checkpoint
          // corruption and clamping it is the safe read.
          return sycl::fmax(mag, 1e-8f);
        };
        const float g0 = sycl::log2(dec(src[0]));
        const float g1 = sycl::log2(dec(src[1]));
        const int ex = static_cast<int>(sycl::round((g0 + g1) * 0.5f)) + 127;
        plane[(e * N + n) * kg32 + g] =
            static_cast<std::uint8_t>(sycl::clamp(ex, 0, 254));
      });
  cache.emplace(bank_e4m3, plane);
  return plane;
}

}  // namespace

bool grouped_moe_nvfp4(sycl::queue& q, const sycl::half* act_src,
                       const std::int32_t* ids, const float* route_w,
                       const std::uint8_t* w13, const std::uint8_t* s13,
                       const float* alpha13, const std::uint8_t* w2,
                       const std::uint8_t* s2, const float* alpha2,
                       sycl::half* g_act, float* g_mid,
                       sycl::half* g_gated, float* g_outr,
                       const sycl::half* bias13, const sycl::half* bias2,
                       std::int32_t* hist, std::int32_t* offs,
                       std::int32_t* cursor, std::int32_t* rows,
                       std::int32_t* slot_row, std::int32_t* slot_exp,
                       float* slot_w, std::int32_t* slot_of,
                       std::int32_t* atom, float* out, int M,
                       int experts, int top_k, int hidden, int inter,
                       int group_size, int rows_cap) noexcept try {
  if (M <= 0 || experts <= 0 || top_k <= 0 || group_size != 16) {
    return false;
  }
  const int routes = M * top_k;
  const int H = hidden;
  const int I = inter;
  const int E = experts;

  // 1. histogram of routes per resident expert. Routes not owned by this card
  //    arrive as -1 from the caller's compaction and are skipped.
  q.memset(hist, 0, (E + 1) * sizeof(std::int32_t));
  q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::range<1>(routes), [=](sycl::id<1> i) {
      const int e = ids[i];
      if (e >= 0 && e < E) {
        sycl::atomic_ref<std::int32_t, sycl::memory_order::relaxed,
                         sycl::memory_scope::device>(hist[e])++;
      }
    });
  });

  // 2. exclusive prefix sum. One thread: E is ~85, and this measured 0.003 ms,
  //    so a parallel scan would be more code for no time.
  q.submit([&](sycl::handler& h) {
    h.single_task([=]() {
      int acc = 0;
      for (int e = 0; e < E; ++e) {
        offs[e] = acc;
        acc += hist[e];
        cursor[e] = offs[e];
        rows[e] = hist[e];
      }
      offs[E] = acc;
    });
  });

  // 3. scatter each route into its expert's slot range, recording the
  //    inverse map (route -> slot) for the reduce-mode output stage.
  //    Unowned routes (-1 ids) and overflow slots stay -1.
  q.memset(slot_of, 0xFF, static_cast<std::size_t>(routes) * sizeof(std::int32_t));
  q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::range<1>(routes), [=](sycl::id<1> i) {
      const int e = ids[i];
      if (e < 0 || e >= E) {
        return;
      }
      const int slot =
          sycl::atomic_ref<std::int32_t, sycl::memory_order::relaxed,
                           sycl::memory_scope::device>(cursor[e])++;
      if (slot >= routes) {
        return;
      }
      slot_row[slot] = static_cast<int>(i) / top_k;
      slot_exp[slot] = e;
      slot_w[slot] = route_w[i];
      slot_of[i] = slot;
    });
  });

  // 4. gather activations into expert-major order.
  q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::range<2>(routes, H), [=](sycl::id<2> ij) {
      const int s = static_cast<int>(ij[0]);
      const int c = static_cast<int>(ij[1]);
      // Only offs[E] slots were populated: routes this card does not own
      // arrive as -1 and are skipped by the histogram. Without this bound the
      // gather reads UNINITIALISED slot_row and indexes out of bounds, and the
      // scatter then writes garbage to arbitrary output rows -- which is
      // exactly how this produced NaN logprobs on first boot.
      if (s >= offs[E]) {
        return;
      }
      g_act[static_cast<std::size_t>(s) * H + c] =
          act_src[static_cast<std::size_t>(slot_row[s]) * H + c];
    });
  });

  // 5. w13: [routes, H] x [E, 2I, H] -> [routes, 2I]
  // Backend dispatch, SB_GROUPED_BACKEND:
  //   onednn -- grouped micro kernel over the untouched bank (bit-exact vs
  //             native, 1.01 vs 2.355 ms/layer at 30 rows/expert; gate-2
  //             probe + A/B verify, 2026-08-25).
  //   mxfp4  -- native cute mainloop at group 32 / tile_k 32 over DERIVED
  //             e8m0 scales (approximate; quality-gated before any default).
  // Any refusal falls back to the native launch below within the same
  // dispatch, so numerics never depend on a flag being healthy. offs+1 is
  // the inclusive-ends view oneDNN's grouped encoding expects; rows_cap is
  // the g_* chunk allocation capacity and keeps the cached primitive's
  // shape process-constant.
  const Backend be = backend();
  const bool use_onednn =
      be == Backend::kOnednn && onednn_grouped_armed() && rows_cap >= routes;
  const std::uint8_t* s13_mx =
      be == Backend::kMxfp4 && group_size == 16
          ? mxfp4_scale_plane(q, s13, E, 2 * I, H / 16)
          : nullptr;
  bool leg_done = false;
  if (use_onednn) {
    leg_done = onednn_grouped_gemm(q, g_act, w13, s13, g_mid, offs + 1,
                                   rows_cap, E, H, 2 * I);
  } else if (s13_mx != nullptr) {
    // The cute mainloop distributes tiles through `atom`; it must be zeroed
    // for the mxfp4 launch exactly as for the native one.
    q.memset(atom, 0, sizeof(std::int32_t));
    launch_mxfp4<PolicyMxSmallM>(q, g_act, w13, s13_mx, bias13, g_mid, 2 * I,
                                 H, rows, E, atom);
    leg_done = true;
  }
  if (!leg_done) {
    q.memset(atom, 0, sizeof(std::int32_t));
    // Rows per expert has to be estimated on the host: the true count is
    // offs[E], which lives on the device and reading it would cost a sync.
    // The /2 is the two-card split this rig ships -- a different card count
    // wants this re-derived, not inherited. At M=512 this yields 30 (small
    // tile wins), at M=2048 it yields 120 (big tile wins), matching both
    // measured points.
    const bool big_tile = (M * top_k) / (2 * E) >= kBigTileRowsPerExpert;
    if (big_tile) {
      launch_nvfp4_big(q, g_act, w13, s13, bias13, g_mid, 2 * I, H,
                       rows, E, group_size, atom);
    } else {
      launch_nvfp4<PolicySmallM>(q, g_act, w13, s13, bias13, g_mid, 2 * I, H,
                                 rows, E, group_size, atom);
    }
  }

  // 6. SwiGLU. alpha13 lands here rather than in the kernel: it is constant per
  //    expert, so it factors out of the dot product entirely.
  q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::range<2>(routes, I), [=](sycl::id<2> ij) {
      const int s = static_cast<int>(ij[0]);
      const int c = static_cast<int>(ij[1]);
      if (s >= offs[E]) {
        return;
      }
      const float sc = alpha13[slot_exp[s]];
      const float a = g_mid[static_cast<std::size_t>(s) * 2 * I + c] * sc;
      const float b =
          g_mid[static_cast<std::size_t>(s) * 2 * I + I + c] * sc;
      const float silu = a / (1.0f + sycl::exp(-a));
      g_gated[static_cast<std::size_t>(s) * I + c] =
          static_cast<sycl::half>(silu * b);
    });
  });

  // 7. w2: [routes, I] x [E, H, I] -> [routes, H]
  const std::uint8_t* s2_mx =
      be == Backend::kMxfp4 && group_size == 16
          ? mxfp4_scale_plane(q, s2, E, H, I / 16)
          : nullptr;
  leg_done = false;
  if (use_onednn) {
    leg_done = onednn_grouped_gemm(q, g_gated, w2, s2, g_outr, offs + 1,
                                   rows_cap, E, I, H);
  } else if (s2_mx != nullptr) {
    q.memset(atom, 0, sizeof(std::int32_t));
    launch_mxfp4<PolicyMxSmallM>(q, g_gated, w2, s2_mx, bias2, g_outr, H, I,
                                 rows, E, atom);
    leg_done = true;
  }
  if (!leg_done) {
    q.memset(atom, 0, sizeof(std::int32_t));
    const bool big_tile = (M * top_k) / (2 * E) >= kBigTileRowsPerExpert;
    if (big_tile) {
      launch_nvfp4_big(q, g_gated, w2, s2, bias2, g_outr, H, I, rows,
                       E, group_size, atom);
    } else {
      launch_nvfp4<PolicySmallM>(q, g_gated, w2, s2, bias2, g_outr, H, I,
                                 rows, E, group_size, atom);
    }
  }

  // 8. output stage, two modes:
  //    atomic (default) -- weighted atomic scatter onto a zeroed output.
  //      Float atomics make the summation order run-dependent, which is
  //      the measured run-to-run first-bit churn at partial M.
  //    reduce (SB_GROUPED_SCATTER=reduce) -- per-token segmented reduce
  //      over the inverse map: no memset, no atomics, fixed k order, so
  //      the output is bit-stable across runs. Also drops ~140 MiB of
  //      traffic per layer at M=2048 (24 memset + the RMW read leg).
  //    alpha2 lands here in both modes: constant per expert, so it
  //    factors out of the dot product.
  static const bool kReduceScatter = [] {
    const char* v = std::getenv("SB_GROUPED_SCATTER");
    return v != nullptr && std::strcmp(v, "reduce") == 0;
  }();
  if (kReduceScatter) {
    const int tk = top_k;
    q.submit([&](sycl::handler& h) {
      h.parallel_for(sycl::range<2>(M, H), [=](sycl::id<2> ij) {
        const int t = static_cast<int>(ij[0]);
        const int c = static_cast<int>(ij[1]);
        float acc = 0.0f;
        for (int k = 0; k < tk; ++k) {
          const int s = slot_of[static_cast<std::size_t>(t) * tk + k];
          // A zero router weight would NOT be enough: 0 * NaN is NaN, so
          // unowned routes are skipped by index rather than scaled away.
          if (s >= 0) {
            acc += g_outr[static_cast<std::size_t>(s) * H + c] * slot_w[s] *
                   alpha2[slot_exp[s]];
          }
        }
        out[static_cast<std::size_t>(t) * H + c] = acc;
      });
    });
    return true;
  }
  q.memset(out, 0, static_cast<std::size_t>(M) * H * sizeof(float));
  q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::range<2>(routes, H), [=](sycl::id<2> ij) {
      const int s = static_cast<int>(ij[0]);
      const int c = static_cast<int>(ij[1]);
      // A zero router weight would NOT be enough here: 0 * NaN is NaN, so an
      // invalid row must be skipped rather than scaled away.
      if (s >= offs[E]) {
        return;
      }
      const float v =
          g_outr[static_cast<std::size_t>(s) * H + c] * slot_w[s] * alpha2[slot_exp[s]];
      sycl::atomic_ref<float, sycl::memory_order::relaxed,
                       sycl::memory_scope::device>(
          out[static_cast<std::size_t>(slot_row[s]) * H + c]) += v;
    });
  });
  return true;
} catch (...) {
  // The caller owns the fallback decision; never let an exception cross into
  // the provider's dispatch path.
  return false;
}

}  // namespace sb::xe2
