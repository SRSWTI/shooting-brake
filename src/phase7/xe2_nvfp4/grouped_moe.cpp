#include "grouped_moe.hpp"

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include "cutlass/kernel_hardware_info.h"
#include "xe_2/gemm_xe2_policy.hpp"
#include "xe_2/grouped_gemm_xe2.hpp"

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
using PolicyBigM = MoE::w4a16_policy_m_32_k16;  // measured: k16_d32 lost 7% in situ
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
                       float* slot_w, std::int32_t* atom, float* out, int M,
                       int experts, int top_k, int hidden, int inter,
                       int group_size) noexcept try {
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

  // 3. scatter each route into its expert's slot range.
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
  q.memset(atom, 0, sizeof(std::int32_t));
  // Rows per expert has to be estimated on the host: the true count is
  // offs[E], which lives on the device and reading it would cost a sync. The
  // /2 is the two-card split this rig ships -- a different card count wants
  // this re-derived, not inherited. At M=512 this yields 30 (small tile wins),
  // at M=2048 it yields 120 (big tile wins), matching both measured points.
  const bool big_tile = (M * top_k) / (2 * E) >= kBigTileRowsPerExpert;
  if (big_tile) {
    launch_nvfp4<PolicyBigM>(q, g_act, w13, s13, bias13, g_mid, 2 * I, H, rows,
                             E, group_size, atom);
  } else {
    launch_nvfp4<PolicySmallM>(q, g_act, w13, s13, bias13, g_mid, 2 * I, H,
                               rows, E, group_size, atom);
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
  q.memset(atom, 0, sizeof(std::int32_t));
  if (big_tile) {
    launch_nvfp4<PolicyBigM>(q, g_gated, w2, s2, bias2, g_outr, H, I, rows, E,
                             group_size, atom);
  } else {
    launch_nvfp4<PolicySmallM>(q, g_gated, w2, s2, bias2, g_outr, H, I, rows, E,
                               group_size, atom);
  }

  // 8. weighted scatter back to token rows. alpha2 applied here for the same
  //    reason as alpha13. The output is fully written, not accumulated onto
  //    stale contents, so it must be zeroed first.
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
