// Grouped (batch-shaped) routed MoE for ModelOpt NVFP4 weights.
//
// WHY THIS EXISTS
//
// nvfp4_moe.sycl.cpp launches one work-group per (token, expert) route and
// walks that expert's whole w13/w2 for a single token, so an expert touched by
// n tokens has its weights read n times. That is the right shape at M=1 --
// decode reads each of the 8 routed experts exactly once and runs near the
// card's memory-bandwidth floor -- and the wrong shape for everything else.
// Measured on the 35B at subset:16:8, prefill costs ~27 us per token per
// layer at *every* prompt length from 894 to 57,342 tokens, and a chunk sweep
// (benchmarks/results/chunk) shows collapsing 25 dispatches into 2 moves it
// only 5.4%: the round trips are not the cost, the redundant weight reads are.
//
// At M tokens with top_k=8 over E=256 experts, weight traffic per layer is
// M*top_k*expert_bytes for the per-route kernel against
// distinct_experts*expert_bytes here -- 64x less at M=2048, where every expert
// is touched by ~64 tokens.
//
// SHAPE
//
// Standard grouped-GEMM MoE, the same structure vLLM's fused_moe_kernel and
// llm-scaler's moe_prefill_int4 use:
//
//   1. histogram routes per expert
//   2. exclusive scan -> each expert owns a block-aligned row range
//   3. scatter routes into that range (a permutation of [M*top_k])
//   4. gate/up GEMM + SwiGLU over each expert's rows -> activated[rows, I]
//   5. down GEMM over the same rows, weighted scatter-add into out[token, K]
//
// Steps 4 and 5 are DPAS (Xe tensor engine) via SYCL joint_matrix, tiled the
// way kernels/quantization/w4a16_gemm does it: one subgroup owns a TM x TN
// output tile, loops the reduction dim in TK steps, and stages the DEQUANTIZED
// weight tile into SLM once per step. Each subgroup carries kRowTiles A tiles
// against the same staged weight tile, so one dequant feeds
// kRowTiles*TM = 32 rows.
//
// TK=16 is exactly the NVFP4 block-scale group size, so a weight tile needs one
// E4M3 scale per column and never straddles a scale boundary -- the awkward
// part of int4 g128 simply does not arise here.
//
// WHAT THIS DOES NOT CHANGE
//
// Decode. At M=1 there is nothing to amortize (8 routes, 8 distinct experts),
// and the per-route kernel is already near the bandwidth floor, so the
// dispatch layer keeps sending small M there. This kernel earns its keep from
// roughly M*top_k/E >= 8, i.e. M >= 256 at the 35B's geometry.

#include "moe/nvfp4_moe/nvfp4_moe_kernel.hpp"

#include "nvfp4_dequant.hpp"

#include <cstdint>

#include <sycl/ext/oneapi/matrix/matrix.hpp>

namespace quixicore::xpu::kernels {
namespace {

using namespace sycl::ext::oneapi::experimental::matrix;

// DPAS tile geometry. Matches w4a16_gemm, which was verified bit-exact against
// a CPU reference on this hardware before being trusted.
constexpr int TM = 8;  // DPAS tile rows
constexpr int TN = 16; // DPAS tile cols
constexpr int TK = 16; // DPAS tile depth == NVFP4 block-scale group
constexpr int SG = 16; // subgroup width

// A tiles per subgroup. Each staged (dequantized) weight tile is reused across
// kRowTiles*TM rows, so this is the weight-amortization factor. 4 costs 8
// fp32 accumulators = 256 B/lane of GRF, comfortable in large-GRF mode.
constexpr int kRowTiles = 4;
constexpr int kBlockRows = TM * kRowTiles; // rows per scheduling block

using AtomicI32 =
    sycl::atomic_ref<std::int32_t, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>;
using AtomicF32 =
    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>;

inline std::size_t align_up(std::size_t value, std::size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

inline float silu(float value) { return value / (1.0f + sycl::exp(-value)); }

// Byte offsets of each region inside the caller-provided workspace. Computed
// identically by the size query and the launcher, so the two cannot drift.
struct WorkspacePlan {
  std::size_t counts;       // int32[E]      routes per expert
  std::size_t cursor;       // int32[E]      scatter write cursor
  std::size_t offsets;      // int32[E + 1]  block-aligned row start per expert
  std::size_t num_blocks;   // int32[1]
  std::size_t block_expert; // int32[max_blocks]
  std::size_t sorted;       // int32[padded_rows]  route id, or -1 for padding
  std::size_t activated;    // T[padded_rows * I]
  std::size_t padded_rows;
  std::size_t max_blocks;
  std::size_t total_bytes;
};

WorkspacePlan plan_workspace(std::size_t M, std::size_t E, std::size_t top_k,
                             std::size_t I, std::size_t act_elem_bytes) {
  constexpr std::size_t kAlign = 256;
  // Each expert's range is padded up to a whole block, so the worst case adds
  // kBlockRows-1 rows per expert on top of the real routes.
  const std::size_t routes = M * top_k;
  WorkspacePlan p{};
  p.padded_rows = routes + E * (kBlockRows - 1);
  p.max_blocks = p.padded_rows / kBlockRows + 1;

  std::size_t at = 0;
  auto take = [&](std::size_t bytes) {
    const std::size_t here = at;
    at = align_up(at + bytes, kAlign);
    return here;
  };
  p.counts = take(E * sizeof(std::int32_t));
  p.cursor = take(E * sizeof(std::int32_t));
  p.offsets = take((E + 1) * sizeof(std::int32_t));
  p.num_blocks = take(sizeof(std::int32_t));
  p.block_expert = take(p.max_blocks * sizeof(std::int32_t));
  p.sorted = take(p.padded_rows * sizeof(std::int32_t));
  p.activated = take(p.padded_rows * I * act_elem_bytes);
  p.total_bytes = at;
  return p;
}

// --- stage 1-3: group routes by expert -------------------------------------

class Nvfp4MoeHistogram;
class Nvfp4MoeScan;
class Nvfp4MoeScatter;

// Routes per expert. Invalid ids (padding, or an expert this provider does not
// hold) are dropped here and therefore never reach a GEMM.
sycl::event launch_histogram(sycl::queue &q, const int *topk_ids,
                             std::int32_t *counts, std::size_t routes,
                             std::size_t E, const sycl::event &wait_on) {
  return q.submit([&](sycl::handler &h) {
    h.depends_on(wait_on);
    h.parallel_for<Nvfp4MoeHistogram>(
        sycl::range<1>(routes), [=](sycl::id<1> idx) {
          const int expert = topk_ids[idx];
          if (expert < 0 || static_cast<std::size_t>(expert) >= E)
            return;
          AtomicI32(counts[expert]).fetch_add(1);
        });
  });
}

// Exclusive scan over experts plus the block table. E is 256 at the 35B and
// 122B, so a single work-item is both fast enough (sub-microsecond) and
// trivially deterministic; a parallel scan would buy nothing measurable and
// would need a second pass to build the block list.
sycl::event launch_scan(sycl::queue &q, const std::int32_t *counts,
                        std::int32_t *cursor, std::int32_t *offsets,
                        std::int32_t *num_blocks, std::int32_t *block_expert,
                        std::size_t E, const sycl::event &wait_on) {
  return q.submit([&](sycl::handler &h) {
    h.depends_on(wait_on);
    h.single_task<Nvfp4MoeScan>([=]() {
      std::int32_t row = 0;
      std::int32_t blocks = 0;
      for (std::size_t e = 0; e < E; ++e) {
        const std::int32_t count = counts[e];
        offsets[e] = row;
        cursor[e] = row;
        const std::int32_t nblocks =
            (count + kBlockRows - 1) / kBlockRows;
        for (std::int32_t b = 0; b < nblocks; ++b)
          block_expert[blocks + b] = static_cast<std::int32_t>(e);
        blocks += nblocks;
        row += nblocks * kBlockRows;
      }
      offsets[E] = row;
      num_blocks[0] = blocks;
    });
  });
}

// Place each route in its expert's range. Order within a range is whatever the
// atomics produce; that is immaterial because every row is computed
// independently and the only cross-row combination is the commutative
// scatter-add in stage 5, which the per-route kernel performs too.
sycl::event launch_scatter(sycl::queue &q, const int *topk_ids,
                           std::int32_t *cursor, std::int32_t *sorted,
                           std::size_t routes, std::size_t E,
                           const sycl::event &wait_on) {
  return q.submit([&](sycl::handler &h) {
    h.depends_on(wait_on);
    h.parallel_for<Nvfp4MoeScatter>(
        sycl::range<1>(routes), [=](sycl::id<1> idx) {
          const int expert = topk_ids[idx];
          if (expert < 0 || static_cast<std::size_t>(expert) >= E)
            return;
          const std::int32_t slot = AtomicI32(cursor[expert]).fetch_add(1);
          sorted[slot] = static_cast<std::int32_t>(idx);
        });
  });
}

// --- shared tile staging ----------------------------------------------------

// Stage one weight tile [TK][TN] into SLM, transposed so that
// bs[kk*TN + nn] = W[n0+nn][k0+kk] -- the row-major layout joint_matrix use::b
// expects. Out-of-range lanes write zero, which makes any N or K shape correct
// without a separate remainder kernel.
//
// The tile holds the true e2m1 * e4m3 product. The per-expert global scale is
// NOT folded in; it is applied to the fp32 accumulator in the epilogue. Both
// halves of that split are load-bearing:
//
//  * Undo the decoders' 2^-22 here (kGlobalScaleFixup, an exact power of two).
//    Left in, an fp16 tile lands in subnormals where the product's 6
//    significant bits do not fit -- 3% error at block scale 0x3B, 18% at 0x2D,
//    and 1.1e-3 end-to-end against a bound of 1.7e-5. This is an EXPONENT
//    problem, not a mantissa one: bf16 carries fp32's 8 exponent bits and is
//    bit-identical either way, so an fp16-free test would never see it.
//    Undone, the product spans [2^-9 * 0.5, 448 * 6] = [1e-3, 2688], entirely
//    within fp16 normals.
//  * Keep the global scale OUT. e2m1 contributes 1 mantissa bit and e4m3
//    contributes 3, so their product needs at most 6 significant bits -- fewer
//    than fp16's 11 or bf16's 8, making the staged tile EXACT. An arbitrary
//    fp32 scale would round every weight, and for this model would also
//    overflow: ~8e4 against fp16's 65504 max.
//
// So the only rounding in this kernel is the activation operand.
//
// `rows` points at the first weight row of the expert, `scales` at its first
// scale row. Because TK equals the NVFP4 group size and k0 is a multiple of
// TK, every element of a tile column shares one block scale.
template <typename T>
inline void stage_weight_tile(int lane, const std::uint8_t *rows,
                              const std::uint8_t *scales,
                              std::size_t row_bytes, std::size_t scale_stride,
                              std::size_t n0, std::size_t n_limit,
                              std::size_t k0, std::size_t k_limit, T *bs) {
  for (int e = lane; e < TK * TN; e += SG) {
    const std::size_t kk = static_cast<std::size_t>(e) / TN;
    const std::size_t nn = static_cast<std::size_t>(e) % TN;
    const std::size_t gk = k0 + kk;
    const std::size_t gn = n0 + nn;
    float value = 0.0f;
    if (gk < k_limit && gn < n_limit) {
      const std::uint8_t *row = rows + gn * row_bytes;
      const float scale =
          nvfp4::decode_e4m3(scales[gn * scale_stride + (gk / 16)]);
      value = nvfp4::decode_packed_element(row, gk) * scale *
              nvfp4::kGlobalScaleFixup;
    }
    bs[e] = static_cast<T>(value);
  }
}

// --- stage 4: gate/up GEMM + SwiGLU ----------------------------------------

template <typename T> class Nvfp4MoeGateUp;

// out: activated[row, 0..I) = silu(x . w13_gate[e]) * (x . w13_up[e])
//
// Gate row i and up row i of w13 are rows i and i+I of the same tensor, so one
// subgroup accumulates both against a single staged A tile. That halves the
// activation traffic versus writing gate and up to scratch and reducing them in
// a second pass, and it is why this kernel needs no [rows, 2I] scratch at all.
template <typename T>
sycl::event launch_gate_up(sycl::queue &q, const T *hidden,
                           const std::uint8_t *w13, const std::uint8_t *s13,
                           const float *w13_global, const std::int32_t *sorted,
                           const std::int32_t *block_expert,
                           const std::int32_t *num_blocks, T *activated,
                           std::size_t top_k, std::size_t K, std::size_t I,
                           std::size_t max_blocks, const sycl::event &wait_on) {
  const std::size_t two_i = 2 * I;
  const std::size_t w13_expert_stride = two_i * (K / 2);
  const std::size_t s13_expert_stride = two_i * (K / 16);
  const std::size_t row_bytes = K / 2;
  const std::size_t scale_stride = K / 16;
  const std::size_t ntiles = (I + TN - 1) / TN;
  const std::size_t ktiles = (K + TK - 1) / TK;

  return q.submit([&](sycl::handler &h) {
    h.depends_on(wait_on);
    sycl::local_accessor<T, 1> as(sycl::range<1>(kRowTiles * TM * TK), h);
    sycl::local_accessor<T, 1> bs_gate(sycl::range<1>(TK * TN), h);
    sycl::local_accessor<T, 1> bs_up(sycl::range<1>(TK * TN), h);
    sycl::local_accessor<float, 1> cs(sycl::range<1>(2 * kRowTiles * TM * TN),
                                      h);
    h.parallel_for<Nvfp4MoeGateUp<T>>(
        sycl::nd_range<2>(sycl::range<2>(max_blocks, ntiles * SG),
                          sycl::range<2>(1, SG)),
        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
          const std::size_t block = it.get_group(0);
          // max_blocks is an upper bound computed on the host; the real count
          // is only known on device. Over-launching and exiting is what keeps
          // this path free of a host round trip (and so graph-capturable).
          if (block >= static_cast<std::size_t>(num_blocks[0]))
            return;

          const int lane = static_cast<int>(it.get_local_id(1));
          const sycl::sub_group sg = it.get_sub_group();
          const std::size_t expert =
              static_cast<std::size_t>(block_expert[block]);
          const std::size_t row0 = block * kBlockRows;
          const std::size_t n0 = it.get_group(1) * TN;

          const std::uint8_t *rows = w13 + expert * w13_expert_stride;
          const std::uint8_t *scales = s13 + expert * s13_expert_stride;
          // Raw, un-fixed-up: stage_weight_tile already undid the decoders'
          // 2^-22, so the accumulator is in true weight units.
          const float global = w13_global[expert];

          joint_matrix<sycl::sub_group, T, use::a, TM, TK, layout::row_major>
              ma[kRowTiles];
          joint_matrix<sycl::sub_group, T, use::b, TK, TN, layout::row_major>
              mb_gate, mb_up;
          joint_matrix<sycl::sub_group, float, use::accumulator, TM, TN>
              acc_gate[kRowTiles], acc_up[kRowTiles];
#pragma unroll
          for (int r = 0; r < kRowTiles; ++r) {
            joint_matrix_fill(sg, acc_gate[r], 0.0f);
            joint_matrix_fill(sg, acc_up[r], 0.0f);
          }

          for (std::size_t kt = 0; kt < ktiles; ++kt) {
            const std::size_t k0 = kt * TK;
            // A tiles: gather each row's token through the permutation.
            for (int e = lane; e < kRowTiles * TM * TK; e += SG) {
              const std::size_t tile = static_cast<std::size_t>(e) / (TM * TK);
              const std::size_t rest = static_cast<std::size_t>(e) % (TM * TK);
              const std::size_t mm = rest / TK;
              const std::size_t kk = rest % TK;
              const std::int32_t route = sorted[row0 + tile * TM + mm];
              const std::size_t gk = k0 + kk;
              T value = T(0.0f);
              if (route >= 0 && gk < K) {
                const std::size_t token =
                    static_cast<std::size_t>(route) / top_k;
                value = hidden[token * K + gk];
              }
              as[e] = value;
            }
            stage_weight_tile<T>(lane, rows, scales, row_bytes, scale_stride,
                                 n0, I, k0, K, &bs_gate[0]);
            stage_weight_tile<T>(lane, rows, scales, row_bytes, scale_stride,
                                 I + n0, two_i, k0, K, &bs_up[0]);
            it.barrier(sycl::access::fence_space::local_space);

            joint_matrix_load(
                sg, mb_gate,
                bs_gate.template get_multi_ptr<sycl::access::decorated::no>(),
                TN);
            joint_matrix_load(
                sg, mb_up,
                bs_up.template get_multi_ptr<sycl::access::decorated::no>(),
                TN);
#pragma unroll
            for (int r = 0; r < kRowTiles; ++r) {
              joint_matrix_load(
                  sg, ma[r],
                  as.template get_multi_ptr<sycl::access::decorated::no>() +
                      r * TM * TK,
                  TK);
              joint_matrix_mad(sg, acc_gate[r], ma[r], mb_gate, acc_gate[r]);
              joint_matrix_mad(sg, acc_up[r], ma[r], mb_up, acc_up[r]);
            }
            it.barrier(sycl::access::fence_space::local_space);
          }

#pragma unroll
          for (int r = 0; r < kRowTiles; ++r) {
            joint_matrix_store(
                sg, acc_gate[r],
                cs.template get_multi_ptr<sycl::access::decorated::no>() +
                    r * TM * TN,
                TN, layout::row_major);
            joint_matrix_store(
                sg, acc_up[r],
                cs.template get_multi_ptr<sycl::access::decorated::no>() +
                    (kRowTiles + r) * TM * TN,
                TN, layout::row_major);
          }
          it.barrier(sycl::access::fence_space::local_space);

          for (int e = lane; e < kRowTiles * TM * TN; e += SG) {
            const std::size_t tile = static_cast<std::size_t>(e) / (TM * TN);
            const std::size_t rest = static_cast<std::size_t>(e) % (TM * TN);
            const std::size_t mm = rest / TN;
            const std::size_t nn = rest % TN;
            const std::size_t row = row0 + tile * TM + mm;
            const std::size_t col = n0 + nn;
            if (col >= I || sorted[row] < 0)
              continue;
            // Global scale lands here, on the fp32 accumulator, so the staged
            // weight tiles stay exact and in fp16 range.
            const float gate = cs[e] * global;
            const float up = cs[kRowTiles * TM * TN + e] * global;
            activated[row * I + col] = static_cast<T>(silu(gate) * up);
          }
        });
  });
}

// --- stage 5: down GEMM + weighted scatter-add -----------------------------

template <typename T> class Nvfp4MoeDown;

// out[token, 0..K) += router_weight * (activated[row, 0..I) . w2[e])
//
// Rows of one expert are contiguous in the permuted order, so unlike stage 4
// the A operand needs no gather. The scatter-add is atomic because a token's
// top_k routes land in different experts and therefore different blocks.
template <typename T>
sycl::event launch_down(sycl::queue &q, const T *activated,
                        const std::uint8_t *w2, const std::uint8_t *s2,
                        const float *w2_global, const std::int32_t *sorted,
                        const std::int32_t *block_expert,
                        const std::int32_t *num_blocks,
                        const float *topk_weights, float *out, std::size_t top_k,
                        std::size_t K, std::size_t I, std::size_t max_blocks,
                        bool multiply_router_weight,
                        const sycl::event &wait_on) {
  const std::size_t w2_expert_stride = K * (I / 2);
  const std::size_t s2_expert_stride = K * (I / 16);
  const std::size_t row_bytes = I / 2;
  const std::size_t scale_stride = I / 16;
  const std::size_t ntiles = (K + TN - 1) / TN;
  const std::size_t ktiles = (I + TK - 1) / TK;

  return q.submit([&](sycl::handler &h) {
    h.depends_on(wait_on);
    sycl::local_accessor<T, 1> as(sycl::range<1>(kRowTiles * TM * TK), h);
    sycl::local_accessor<T, 1> bs(sycl::range<1>(TK * TN), h);
    sycl::local_accessor<float, 1> cs(sycl::range<1>(kRowTiles * TM * TN), h);
    h.parallel_for<Nvfp4MoeDown<T>>(
        sycl::nd_range<2>(sycl::range<2>(max_blocks, ntiles * SG),
                          sycl::range<2>(1, SG)),
        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
          const std::size_t block = it.get_group(0);
          if (block >= static_cast<std::size_t>(num_blocks[0]))
            return;

          const int lane = static_cast<int>(it.get_local_id(1));
          const sycl::sub_group sg = it.get_sub_group();
          const std::size_t expert =
              static_cast<std::size_t>(block_expert[block]);
          const std::size_t row0 = block * kBlockRows;
          const std::size_t n0 = it.get_group(1) * TN;

          const std::uint8_t *rows = w2 + expert * w2_expert_stride;
          const std::uint8_t *scales = s2 + expert * s2_expert_stride;
          const float global = w2_global[expert];

          joint_matrix<sycl::sub_group, T, use::a, TM, TK, layout::row_major>
              ma[kRowTiles];
          joint_matrix<sycl::sub_group, T, use::b, TK, TN, layout::row_major>
              mb;
          joint_matrix<sycl::sub_group, float, use::accumulator, TM, TN>
              acc[kRowTiles];
#pragma unroll
          for (int r = 0; r < kRowTiles; ++r)
            joint_matrix_fill(sg, acc[r], 0.0f);

          for (std::size_t kt = 0; kt < ktiles; ++kt) {
            const std::size_t k0 = kt * TK;
            for (int e = lane; e < kRowTiles * TM * TK; e += SG) {
              const std::size_t tile = static_cast<std::size_t>(e) / (TM * TK);
              const std::size_t rest = static_cast<std::size_t>(e) % (TM * TK);
              const std::size_t mm = rest / TK;
              const std::size_t kk = rest % TK;
              const std::size_t row = row0 + tile * TM + mm;
              const std::size_t gk = k0 + kk;
              as[e] = (sorted[row] >= 0 && gk < I)
                          ? activated[row * I + gk]
                          : T(0.0f);
            }
            stage_weight_tile<T>(lane, rows, scales, row_bytes, scale_stride,
                                 n0, K, k0, I, &bs[0]);
            it.barrier(sycl::access::fence_space::local_space);

            joint_matrix_load(
                sg, mb,
                bs.template get_multi_ptr<sycl::access::decorated::no>(), TN);
#pragma unroll
            for (int r = 0; r < kRowTiles; ++r) {
              joint_matrix_load(
                  sg, ma[r],
                  as.template get_multi_ptr<sycl::access::decorated::no>() +
                      r * TM * TK,
                  TK);
              joint_matrix_mad(sg, acc[r], ma[r], mb, acc[r]);
            }
            it.barrier(sycl::access::fence_space::local_space);
          }

#pragma unroll
          for (int r = 0; r < kRowTiles; ++r)
            joint_matrix_store(
                sg, acc[r],
                cs.template get_multi_ptr<sycl::access::decorated::no>() +
                    r * TM * TN,
                TN, layout::row_major);
          it.barrier(sycl::access::fence_space::local_space);

          for (int e = lane; e < kRowTiles * TM * TN; e += SG) {
            const std::size_t tile = static_cast<std::size_t>(e) / (TM * TN);
            const std::size_t rest = static_cast<std::size_t>(e) % (TM * TN);
            const std::size_t mm = rest / TN;
            const std::size_t nn = rest % TN;
            const std::size_t row = row0 + tile * TM + mm;
            const std::size_t col = n0 + nn;
            const std::int32_t route = sorted[row];
            if (col >= K || route < 0)
              continue;
            const std::size_t token = static_cast<std::size_t>(route) / top_k;
            // Router weight and the per-expert global scale both fold into
            // this one fp32 multiply (see stage_weight_tile).
            const float weight =
                multiply_router_weight ? topk_weights[route] : 1.0f;
            AtomicF32(out[token * K + col]).fetch_add(cs[e] * global * weight);
          }
        });
  });
}

// --- driver -----------------------------------------------------------------

template <typename T>
sycl::event dispatch_grouped(sycl::queue &q, const void *hidden,
                             const int *topk_ids, const float *topk_weights,
                             const void *w13, const void *w13_scales,
                             const float *w13_global_scales, const void *w2,
                             const void *w2_scales, const float *w2_global_scales,
                             void *workspace, float *out_f32, std::size_t M,
                             std::size_t E, std::size_t top_k, std::size_t K,
                             std::size_t I, bool multiply_router_weight,
                             const sycl::event &output_ready,
                             Nvfp4MoeGroupedStages *stages) {
  const WorkspacePlan plan = plan_workspace(M, E, top_k, I, sizeof(T));
  auto *base = static_cast<std::uint8_t *>(workspace);
  auto *counts = reinterpret_cast<std::int32_t *>(base + plan.counts);
  auto *cursor = reinterpret_cast<std::int32_t *>(base + plan.cursor);
  auto *offsets = reinterpret_cast<std::int32_t *>(base + plan.offsets);
  auto *num_blocks = reinterpret_cast<std::int32_t *>(base + plan.num_blocks);
  auto *block_expert =
      reinterpret_cast<std::int32_t *>(base + plan.block_expert);
  auto *sorted = reinterpret_cast<std::int32_t *>(base + plan.sorted);
  auto *activated = reinterpret_cast<T *>(base + plan.activated);
  const std::size_t routes = M * top_k;

  // 0xFF bytes == -1 int32: every padded row starts invalid, so a row a route
  // never claims is skipped by both GEMMs.
  sycl::event cleared_counts =
      q.memset(counts, 0, E * sizeof(std::int32_t));
  sycl::event cleared_sorted = q.submit([&](sycl::handler &h) {
    h.depends_on(output_ready);
    h.memset(sorted, 0xFF, plan.padded_rows * sizeof(std::int32_t));
  });

  sycl::event hist =
      launch_histogram(q, topk_ids, counts, routes, E, cleared_counts);
  sycl::event scan = launch_scan(q, counts, cursor, offsets, num_blocks,
                                 block_expert, E, hist);
  sycl::event join = q.submit([&](sycl::handler &h) {
    h.depends_on({scan, cleared_sorted});
    h.single_task([]() {}); // join point for the two prerequisites
  });
  sycl::event scatter =
      launch_scatter(q, topk_ids, cursor, sorted, routes, E, join);

  sycl::event gate_up = launch_gate_up<T>(
      q, static_cast<const T *>(hidden), static_cast<const std::uint8_t *>(w13),
      static_cast<const std::uint8_t *>(w13_scales), w13_global_scales, sorted,
      block_expert, num_blocks, activated, top_k, K, I, plan.max_blocks,
      scatter);

  sycl::event down = launch_down<T>(
      q, activated, static_cast<const std::uint8_t *>(w2),
      static_cast<const std::uint8_t *>(w2_scales), w2_global_scales, sorted,
      block_expert, num_blocks, topk_weights, out_f32, top_k, K, I,
      plan.max_blocks, multiply_router_weight, gate_up);

  if (stages != nullptr) {
    stages->clear_counts = cleared_counts;
    stages->clear_sorted = cleared_sorted;
    stages->histogram = hist;
    stages->scan = scan;
    stages->join = join;
    stages->scatter = scatter;
    stages->gate_up = gate_up;
    stages->down = down;
  }
  return down;
}

} // namespace

std::size_t nvfp4_moe_grouped_workspace_bytes(std::size_t M, std::size_t E,
                                              std::size_t top_k, std::size_t I,
                                              DType act_dt) {
  const std::size_t elem = act_dt == DType::bf16 ? sizeof(bf16_t)
                                                 : sizeof(half_t);
  return plan_workspace(M, E, top_k, I, elem).total_bytes;
}

namespace {

sycl::event grouped_by_dtype(
    sycl::queue &q, const void *hidden, const int *topk_ids,
    const float *topk_weights, const void *w13, const void *w13_scales,
    const float *w13_global_scales, const void *w2, const void *w2_scales,
    const float *w2_global_scales, void *workspace, float *out_f32,
    std::size_t M, std::size_t E, std::size_t top_k, std::size_t K,
    std::size_t I, bool multiply_router_weight, DType act_dt,
    const sycl::event &output_ready, Nvfp4MoeGroupedStages *stages) {
  switch (act_dt) {
  case DType::f16:
    return dispatch_grouped<half_t>(
        q, hidden, topk_ids, topk_weights, w13, w13_scales, w13_global_scales,
        w2, w2_scales, w2_global_scales, workspace, out_f32, M, E, top_k, K, I,
        multiply_router_weight, output_ready, stages);
  case DType::bf16:
    return dispatch_grouped<bf16_t>(
        q, hidden, topk_ids, topk_weights, w13, w13_scales, w13_global_scales,
        w2, w2_scales, w2_global_scales, workspace, out_f32, M, E, top_k, K, I,
        multiply_router_weight, output_ready, stages);
  default:
    // f32 activations have no DPAS operand type. The dispatch layer routes
    // those to the per-route kernel rather than silently changing precision.
    return {};
  }
}

} // namespace

sycl::event nvfp4_moe_grouped_sycl(
    sycl::queue &q, const void *hidden, const int *topk_ids,
    const float *topk_weights, const void *w13, const void *w13_scales,
    const float *w13_global_scales, const void *w2, const void *w2_scales,
    const float *w2_global_scales, void *workspace, float *out_f32,
    std::size_t M, std::size_t E, std::size_t top_k, std::size_t K,
    std::size_t I, bool multiply_router_weight, DType act_dt,
    const sycl::event &output_ready) {
  return grouped_by_dtype(q, hidden, topk_ids, topk_weights, w13, w13_scales,
                          w13_global_scales, w2, w2_scales, w2_global_scales,
                          workspace, out_f32, M, E, top_k, K, I,
                          multiply_router_weight, act_dt, output_ready,
                          /*stages=*/nullptr);
}

sycl::event nvfp4_moe_grouped_profiled_sycl(
    sycl::queue &q, const void *hidden, const int *topk_ids,
    const float *topk_weights, const void *w13, const void *w13_scales,
    const float *w13_global_scales, const void *w2, const void *w2_scales,
    const float *w2_global_scales, void *workspace, float *out_f32,
    std::size_t M, std::size_t E, std::size_t top_k, std::size_t K,
    std::size_t I, bool multiply_router_weight, DType act_dt,
    const sycl::event &output_ready, Nvfp4MoeGroupedStages *stages) {
  return grouped_by_dtype(q, hidden, topk_ids, topk_weights, w13, w13_scales,
                          w13_global_scales, w2, w2_scales, w2_global_scales,
                          workspace, out_f32, M, E, top_k, K, I,
                          multiply_router_weight, act_dt, output_ready, stages);
}

bool nvfp4_moe_grouped_supported(DType act_dt) {
  return act_dt == DType::f16 || act_dt == DType::bf16;
}

} // namespace quixicore::xpu::kernels
