// Decode-oriented split routed MoE for AutoGPTQ grouped int4 weights.
//
// The checkpoint layout is already the layout this kernel wants: qweight is
// [K/8,N], so lane n of a subgroup reads qweight[kpack*N+n] while neighboring
// lanes read neighboring int32 words. One word stays in a register for all
// eight MACs it encodes; there is no repack, transpose, or zero-point load.

#include "moe/int4_moe/int4_moe_kernel.hpp"

#include <cstdint>
#include <vector>

namespace quixicore::xpu::kernels {
namespace {

constexpr std::size_t kValuesPerWord = 8;
// auto-gptq stores symmetric qzeros as (zero_point - 1). The checkpoint words
// are 0x77777777, therefore the effective zero point is 8, not 7.
constexpr int kAutoGptqZeroPoint = 8;
constexpr int kSG = 32;
constexpr int kGateReductionSubgroups = 8;
constexpr int kDownReductionSubgroups = 4;


inline float silu(float value) {
  return value / (1.0f + sycl::exp(-value));
}

inline int decode_nibble(std::uint32_t word, int slot) {
  return static_cast<int>((word >> (4 * slot)) & 0x0fu) -
         kAutoGptqZeroPoint;
}

template <typename T> class Int4MoeGateUpKernel;
template <typename T> class Int4MoeDownKernel;
template <typename T, int OutputsPerLane> class Int4MoeDownWideKernel;


// One work-group owns 32 adjacent output features; its eight subgroups split
// the reduction groups. At K=3072/I=1024, a gate/up work-item reads one eighth
// of an output feature: 384 qweight + 12 scale bytes. Identical activation
// vectors are serviced through cache/coalescing (~24 physical bytes/work-item),
// then 8 bytes of partials are stored in 2 KiB SLM: ~428 bytes/work-item.
// Four local routes expose 1024 subgroups in 128 work-groups. The WG uses only
// 2 KiB SLM and two fp32 accumulators per lane, so B70 occupancy should be
// scheduling-limited rather than register- or SLM-limited. Every qweight
// transaction remains across 32 adjacent N values and every word feeds 8 MACs.
template <typename T>
sycl::event launch_gate_up(
    sycl::queue &q, const T *hidden, const int *topk_ids,
    const std::int32_t *gate_qweight, const half_t *gate_scales,
    const std::int32_t *up_qweight, const half_t *up_scales,
    float *scratch, std::size_t expert_stride_bytes, std::size_t group_size,
    std::size_t pairs, std::size_t E, std::size_t top_k, std::size_t K,
    std::size_t I) {
  const std::size_t tiles = (I + kSG - 1) / kSG;
  constexpr int kWG = kSG * kGateReductionSubgroups;

  return q.submit([&](sycl::handler &handler) {
    sycl::local_accessor<float, 1> partials(
        sycl::range<1>(2 * kGateReductionSubgroups * kSG), handler);
    handler.parallel_for<Int4MoeGateUpKernel<T>>(
        sycl::nd_range<2>(sycl::range<2>(pairs, tiles * kWG),
                          sycl::range<2>(1, kWG)),
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(kSG)]] {
          const std::size_t pair = item.get_global_id(0);
          const std::size_t tile = item.get_group(1);
          const sycl::sub_group subgroup = item.get_sub_group();
          const int subgroup_id =
              static_cast<int>(subgroup.get_group_linear_id());
          const int lane = static_cast<int>(subgroup.get_local_linear_id());
          const std::size_t n = tile * kSG + static_cast<std::size_t>(lane);
          const int expert_id = topk_ids[pair];
          const bool valid = n < I && expert_id >= 0 &&
                             static_cast<std::size_t>(expert_id) < E;

          float gate_acc = 0.0f;
          float up_acc = 0.0f;
          if (valid) {
            const std::size_t expert_offset =
                static_cast<std::size_t>(expert_id) * expert_stride_bytes;
            const std::int32_t *gate_q =
                reinterpret_cast<const std::int32_t *>(
                    reinterpret_cast<const std::uint8_t *>(gate_qweight) +
                    expert_offset);
            const std::int32_t *up_q = reinterpret_cast<const std::int32_t *>(
                reinterpret_cast<const std::uint8_t *>(up_qweight) +
                expert_offset);
            const half_t *gate_s = reinterpret_cast<const half_t *>(
                reinterpret_cast<const std::uint8_t *>(gate_scales) +
                expert_offset);
            const half_t *up_s = reinterpret_cast<const half_t *>(
                reinterpret_cast<const std::uint8_t *>(up_scales) +
                expert_offset);
            const T *input = hidden + (pair / top_k) * K;
            const std::size_t groups = K / group_size;
            const std::size_t words_per_group =
                group_size / kValuesPerWord;
            for (std::size_t group = static_cast<std::size_t>(subgroup_id);
                 group < groups; group += kGateReductionSubgroups) {
              const float gate_scale =
                  static_cast<float>(gate_s[group * I + n]);
              const float up_scale =
                  static_cast<float>(up_s[group * I + n]);
              for (std::size_t word_in_group = 0;
                   word_in_group < words_per_group; ++word_in_group) {
                const std::size_t packed_k =
                    group * words_per_group + word_in_group;
                const std::uint32_t gate_word =
                    static_cast<std::uint32_t>(gate_q[packed_k * I + n]);
                const std::uint32_t up_word =
                    static_cast<std::uint32_t>(up_q[packed_k * I + n]);
                const sycl::vec<T, kValuesPerWord> xv =
                    *reinterpret_cast<const sycl::vec<T, kValuesPerWord> *>(
                        input + packed_k * kValuesPerWord);
#pragma unroll
                for (int slot = 0;
                     slot < static_cast<int>(kValuesPerWord); ++slot) {
                  const float x = static_cast<float>(xv[slot]);
                  gate_acc = sycl::fma(
                      static_cast<float>(decode_nibble(gate_word, slot)) *
                          gate_scale,
                      x, gate_acc);
                  up_acc = sycl::fma(
                      static_cast<float>(decode_nibble(up_word, slot)) *
                          up_scale,
                      x, up_acc);
                }
              }
            }
          }

          const std::size_t partial_base =
              static_cast<std::size_t>(subgroup_id) * 2 * kSG +
              static_cast<std::size_t>(lane);
          partials[partial_base] = gate_acc;
          partials[partial_base + kSG] = up_acc;
          sycl::group_barrier(item.get_group());

          if (subgroup_id == 0 && valid) {
            float gate = 0.0f;
            float up = 0.0f;
#pragma unroll
            for (int split = 0; split < kGateReductionSubgroups; ++split) {
              const std::size_t base =
                  static_cast<std::size_t>(split) * 2 * kSG +
                  static_cast<std::size_t>(lane);
              gate += partials[base];
              up += partials[base + kSG];
            }
            scratch[pair * I + n] = silu(gate) * up;
          }
        });
  });
}

// Each down work-item reads one quarter of an output feature at I=1024:
// 128 qweight + 4 scale bytes plus ~32 physical activation bytes and one fp32
// partial in 512 B SLM. Four local routes expose 1536 subgroups in 384 WGs.
// The subgroup lanes still consume adjacent qweight/scales, and each loaded
// int32 supplies eight MACs.
template <typename T>
sycl::event launch_down(
    sycl::queue &q, const int *topk_ids, const float *topk_weights,
    const std::int32_t *down_qweight, const half_t *down_scales,
    const float *scratch, float *output, std::size_t expert_stride_bytes,
    std::size_t group_size, std::size_t pairs, std::size_t E,
    std::size_t top_k, std::size_t K, std::size_t I,
    bool multiply_router_weight, const sycl::event &gate_up_ready,
    const sycl::event &output_ready) {
  const std::size_t tiles = (K + kSG - 1) / kSG;
  constexpr int kWG = kSG * kDownReductionSubgroups;

  return q.submit([&](sycl::handler &handler) {
    handler.depends_on(std::vector<sycl::event>{gate_up_ready, output_ready});
    sycl::local_accessor<float, 1> partials(
        sycl::range<1>(kDownReductionSubgroups * kSG), handler);
    handler.parallel_for<Int4MoeDownKernel<T>>(
        sycl::nd_range<2>(sycl::range<2>(pairs, tiles * kWG),
                          sycl::range<2>(1, kWG)),
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(kSG)]] {
          const std::size_t pair = item.get_global_id(0);
          const std::size_t tile = item.get_group(1);
          const sycl::sub_group subgroup = item.get_sub_group();
          const int subgroup_id =
              static_cast<int>(subgroup.get_group_linear_id());
          const int lane = static_cast<int>(subgroup.get_local_linear_id());
          const std::size_t n = tile * kSG + static_cast<std::size_t>(lane);
          const int expert_id = topk_ids[pair];
          const bool valid = n < K && expert_id >= 0 &&
                             static_cast<std::size_t>(expert_id) < E;

          float accumulator = 0.0f;
          if (valid) {
            const std::size_t expert_offset =
                static_cast<std::size_t>(expert_id) * expert_stride_bytes;
            const std::int32_t *down_q =
                reinterpret_cast<const std::int32_t *>(
                    reinterpret_cast<const std::uint8_t *>(down_qweight) +
                    expert_offset);
            const half_t *down_s = reinterpret_cast<const half_t *>(
                reinterpret_cast<const std::uint8_t *>(down_scales) +
                expert_offset);
            const float *activation = scratch + pair * I;
            const std::size_t groups = I / group_size;
            const std::size_t words_per_group =
                group_size / kValuesPerWord;
            for (std::size_t group = static_cast<std::size_t>(subgroup_id);
                 group < groups; group += kDownReductionSubgroups) {
              const float scale =
                  static_cast<float>(down_s[group * K + n]);
              for (std::size_t word_in_group = 0;
                   word_in_group < words_per_group; ++word_in_group) {
                const std::size_t packed_k =
                    group * words_per_group + word_in_group;
                const std::uint32_t word =
                    static_cast<std::uint32_t>(down_q[packed_k * K + n]);
                const sycl::vec<float, kValuesPerWord> xv =
                    *reinterpret_cast<const sycl::vec<float, kValuesPerWord> *>(
                        activation + packed_k * kValuesPerWord);
#pragma unroll
                for (int slot = 0;
                     slot < static_cast<int>(kValuesPerWord); ++slot) {
                  accumulator = sycl::fma(
                      static_cast<float>(decode_nibble(word, slot)) * scale,
                      xv[slot], accumulator);
                }
              }
            }
          }

          partials[static_cast<std::size_t>(subgroup_id) * kSG +
                   static_cast<std::size_t>(lane)] = accumulator;
          sycl::group_barrier(item.get_group());

          if (subgroup_id == 0 && valid) {
            float value = 0.0f;
#pragma unroll
            for (int split = 0; split < kDownReductionSubgroups; ++split) {
              value += partials[static_cast<std::size_t>(split) * kSG +
                                static_cast<std::size_t>(lane)];
            }
            const float router_weight =
                multiply_router_weight ? topk_weights[pair] : 1.0f;
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                out(output[(pair / top_k) * K + n]);
            out.fetch_add(router_weight * value);
          }
        });
  });
}
// Geometry experiment for down's short I=1024 reduction. A lane retains
// multiple output accumulators, so one fp32 activation vector load feeds 2 or
// 4 independently coalesced qweight transactions. Qweight/scale bytes and the
// four-way reduction tree are unchanged; work-groups, barriers, and activation
// loads per output fall by OutputsPerLane. The cost is proportionally longer
// live ranges and 2x/4x fewer work-groups, which is why both shapes remain
// benchmark-only until measured on B70.
template <typename T, int OutputsPerLane>
sycl::event launch_down_wide(
    sycl::queue &q, const int *topk_ids, const float *topk_weights,
    const std::int32_t *down_qweight, const half_t *down_scales,
    const float *scratch, float *output, std::size_t expert_stride_bytes,
    std::size_t group_size, std::size_t pairs, std::size_t E,
    std::size_t top_k, std::size_t K, std::size_t I,
    bool multiply_router_weight, const sycl::event &gate_up_ready,
    const sycl::event &output_ready) {
  const std::size_t columns_per_group = kSG * OutputsPerLane;
  const std::size_t tiles =
      (K + columns_per_group - 1) / columns_per_group;
  constexpr int kWG = kSG * kDownReductionSubgroups;

  return q.submit([&](sycl::handler &handler) {
    handler.depends_on(std::vector<sycl::event>{gate_up_ready, output_ready});
    sycl::local_accessor<float, 1> partials(
        sycl::range<1>(kDownReductionSubgroups * OutputsPerLane * kSG),
        handler);
    handler.parallel_for<Int4MoeDownWideKernel<T, OutputsPerLane>>(
        sycl::nd_range<2>(sycl::range<2>(pairs, tiles * kWG),
                          sycl::range<2>(1, kWG)),
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(kSG)]] {
          const std::size_t pair = item.get_global_id(0);
          const std::size_t first_n =
              item.get_group(1) * columns_per_group;
          const sycl::sub_group subgroup = item.get_sub_group();
          const int subgroup_id =
              static_cast<int>(subgroup.get_group_linear_id());
          const int lane = static_cast<int>(subgroup.get_local_linear_id());
          const int expert_id = topk_ids[pair];
          const bool expert_valid =
              expert_id >= 0 && static_cast<std::size_t>(expert_id) < E;

          float accumulators[OutputsPerLane] = {};
          if (expert_valid) {
            const std::size_t expert_offset =
                static_cast<std::size_t>(expert_id) * expert_stride_bytes;
            const std::int32_t *down_q =
                reinterpret_cast<const std::int32_t *>(
                    reinterpret_cast<const std::uint8_t *>(down_qweight) +
                    expert_offset);
            const half_t *down_s = reinterpret_cast<const half_t *>(
                reinterpret_cast<const std::uint8_t *>(down_scales) +
                expert_offset);
            const float *activation = scratch + pair * I;
            const std::size_t groups = I / group_size;
            const std::size_t words_per_group =
                group_size / kValuesPerWord;
            for (std::size_t group = static_cast<std::size_t>(subgroup_id);
                 group < groups; group += kDownReductionSubgroups) {
              float scales[OutputsPerLane] = {};
#pragma unroll
              for (int output_index = 0; output_index < OutputsPerLane;
                   ++output_index) {
                const std::size_t n =
                    first_n + static_cast<std::size_t>(output_index) * kSG +
                    static_cast<std::size_t>(lane);
                if (n < K)
                  scales[output_index] =
                      static_cast<float>(down_s[group * K + n]);
              }
              for (std::size_t word_in_group = 0;
                   word_in_group < words_per_group; ++word_in_group) {
                const std::size_t packed_k =
                    group * words_per_group + word_in_group;
                const sycl::vec<float, kValuesPerWord> xv =
                    *reinterpret_cast<const sycl::vec<float, kValuesPerWord> *>(
                        activation + packed_k * kValuesPerWord);
                std::uint32_t words[OutputsPerLane] = {};
#pragma unroll
                for (int output_index = 0; output_index < OutputsPerLane;
                     ++output_index) {
                  const std::size_t n =
                      first_n + static_cast<std::size_t>(output_index) * kSG +
                      static_cast<std::size_t>(lane);
                  if (n < K)
                    words[output_index] = static_cast<std::uint32_t>(
                        down_q[packed_k * K + n]);
                }
#pragma unroll
                for (int slot = 0;
                     slot < static_cast<int>(kValuesPerWord); ++slot) {
#pragma unroll
                  for (int output_index = 0;
                       output_index < OutputsPerLane; ++output_index) {
                    accumulators[output_index] = sycl::fma(
                        static_cast<float>(
                            decode_nibble(words[output_index], slot)) *
                            scales[output_index],
                        xv[slot], accumulators[output_index]);
                  }
                }
              }
            }
          }

#pragma unroll
          for (int output_index = 0; output_index < OutputsPerLane;
               ++output_index) {
            const std::size_t partial =
                (static_cast<std::size_t>(subgroup_id) * OutputsPerLane +
                 static_cast<std::size_t>(output_index)) *
                    kSG +
                static_cast<std::size_t>(lane);
            partials[partial] = accumulators[output_index];
          }
          sycl::group_barrier(item.get_group());

          if (subgroup_id == 0 && expert_valid) {
            const float router_weight =
                multiply_router_weight ? topk_weights[pair] : 1.0f;
#pragma unroll
            for (int output_index = 0; output_index < OutputsPerLane;
                 ++output_index) {
              const std::size_t n =
                  first_n + static_cast<std::size_t>(output_index) * kSG +
                  static_cast<std::size_t>(lane);
              if (n >= K)
                continue;
              float value = 0.0f;
#pragma unroll
              for (int split = 0; split < kDownReductionSubgroups; ++split) {
                value += partials[
                    (static_cast<std::size_t>(split) * OutputsPerLane +
                     static_cast<std::size_t>(output_index)) *
                        kSG +
                    static_cast<std::size_t>(lane)];
              }
              sycl::atomic_ref<float, sycl::memory_order::relaxed,
                               sycl::memory_scope::device,
                               sycl::access::address_space::global_space>
                  out(output[(pair / top_k) * K + n]);
              out.fetch_add(router_weight * value);
            }
          }
        });
  });
}

template <typename T, int OutputsPerLane>
sycl::event split_down_wide_typed(
    sycl::queue &q, const T *hidden, const int *topk_ids,
    const float *topk_weights, const std::int32_t *gate_qweight,
    const half_t *gate_scales, const std::int32_t *up_qweight,
    const half_t *up_scales, const std::int32_t *down_qweight,
    const half_t *down_scales, float *scratch_f32, float *out_f32,
    std::size_t expert_stride_bytes, std::size_t group_size, std::size_t M,
    std::size_t E, std::size_t top_k, std::size_t K, std::size_t I,
    bool multiply_router_weight, const sycl::event &output_ready,
    Int4MoeSplitStages *stages) {
  const std::size_t pairs = M * top_k;
  const sycl::event gate_up = launch_gate_up(
      q, hidden, topk_ids, gate_qweight, gate_scales, up_qweight, up_scales,
      scratch_f32, expert_stride_bytes, group_size, pairs, E, top_k, K, I);
  const sycl::event down = launch_down_wide<T, OutputsPerLane>(
      q, topk_ids, topk_weights, down_qweight, down_scales, scratch_f32,
      out_f32, expert_stride_bytes, group_size, pairs, E, top_k, K, I,
      multiply_router_weight, gate_up, output_ready);
  if (stages != nullptr) {
    stages->gate_up = gate_up;
    stages->down = down;
  }
  return down;
}

template <typename T>
sycl::event dispatch_down_wide_typed(
    sycl::queue &q, const T *hidden, const int *topk_ids,
    const float *topk_weights, const std::int32_t *gate_qweight,
    const half_t *gate_scales, const std::int32_t *up_qweight,
    const half_t *up_scales, const std::int32_t *down_qweight,
    const half_t *down_scales, float *scratch_f32, float *out_f32,
    std::size_t expert_stride_bytes, std::size_t group_size, std::size_t M,
    std::size_t E, std::size_t top_k, std::size_t K, std::size_t I,
    bool multiply_router_weight, int outputs_per_lane,
    const sycl::event &output_ready, Int4MoeSplitStages *stages) {
  if (outputs_per_lane == 2)
    return split_down_wide_typed<T, 2>(
        q, hidden, topk_ids, topk_weights, gate_qweight, gate_scales,
        up_qweight, up_scales, down_qweight, down_scales, scratch_f32,
        out_f32, expert_stride_bytes, group_size, M, E, top_k, K, I,
        multiply_router_weight, output_ready, stages);
  return split_down_wide_typed<T, 4>(
      q, hidden, topk_ids, topk_weights, gate_qweight, gate_scales,
      up_qweight, up_scales, down_qweight, down_scales, scratch_f32, out_f32,
      expert_stride_bytes, group_size, M, E, top_k, K, I,
      multiply_router_weight, output_ready, stages);
}


template <typename T>
sycl::event split_typed(
    sycl::queue &q, const T *hidden, const int *topk_ids,
    const float *topk_weights, const std::int32_t *gate_qweight,
    const half_t *gate_scales, const std::int32_t *up_qweight,
    const half_t *up_scales, const std::int32_t *down_qweight,
    const half_t *down_scales, float *scratch_f32, float *out_f32,
    std::size_t expert_stride_bytes, std::size_t group_size, std::size_t M,
    std::size_t E, std::size_t top_k, std::size_t K, std::size_t I,
    bool multiply_router_weight, const sycl::event &output_ready,
    Int4MoeSplitStages *stages) {
  const std::size_t pairs = M * top_k;
  const sycl::event gate_up = launch_gate_up(
      q, hidden, topk_ids, gate_qweight, gate_scales, up_qweight, up_scales,
      scratch_f32, expert_stride_bytes, group_size, pairs, E, top_k, K, I);
  const sycl::event down = launch_down<T>(
      q, topk_ids, topk_weights, down_qweight, down_scales, scratch_f32,
      out_f32, expert_stride_bytes, group_size, pairs, E, top_k, K, I,
      multiply_router_weight, gate_up, output_ready);
  if (stages != nullptr) {
    stages->gate_up = gate_up;
    stages->down = down;
  }
  return down;
}

sycl::event dispatch_split(
    sycl::queue &q, const void *hidden, const int *topk_ids,
    const float *topk_weights, const std::int32_t *gate_qweight,
    const half_t *gate_scales, const std::int32_t *up_qweight,
    const half_t *up_scales, const std::int32_t *down_qweight,
    const half_t *down_scales, float *scratch_f32, float *out_f32,
    std::size_t expert_stride_bytes, std::size_t group_size, std::size_t M,
    std::size_t E, std::size_t top_k, std::size_t K, std::size_t I,
    bool multiply_router_weight, DType act_dt,
    const sycl::event &output_ready, Int4MoeSplitStages *stages) {
  switch (act_dt) {
  case DType::f32:
    return split_typed(q, static_cast<const float *>(hidden), topk_ids,
                       topk_weights, gate_qweight, gate_scales, up_qweight,
                       up_scales, down_qweight, down_scales, scratch_f32,
                       out_f32, expert_stride_bytes, group_size, M, E, top_k,
                       K, I, multiply_router_weight, output_ready, stages);
  case DType::f16:
    return split_typed(q, static_cast<const half_t *>(hidden), topk_ids,
                       topk_weights, gate_qweight, gate_scales, up_qweight,
                       up_scales, down_qweight, down_scales, scratch_f32,
                       out_f32, expert_stride_bytes, group_size, M, E, top_k,
                       K, I, multiply_router_weight, output_ready, stages);
  case DType::bf16:
    return split_typed(q, static_cast<const bf16_t *>(hidden), topk_ids,
                       topk_weights, gate_qweight, gate_scales, up_qweight,
                       up_scales, down_qweight, down_scales, scratch_f32,
                       out_f32, expert_stride_bytes, group_size, M, E, top_k,
                       K, I, multiply_router_weight, output_ready, stages);
  }
  return {};
}

} // namespace

sycl::event int4_moe_split_sycl(
    sycl::queue &q, const void *hidden, const int *topk_ids,
    const float *topk_weights, const std::int32_t *gate_qweight,
    const half_t *gate_scales, const std::int32_t *up_qweight,
    const half_t *up_scales, const std::int32_t *down_qweight,
    const half_t *down_scales, float *scratch_f32, float *out_f32,
    std::size_t expert_stride_bytes, std::size_t group_size, std::size_t M,
    std::size_t E, std::size_t top_k, std::size_t K, std::size_t I,
    bool multiply_router_weight, DType act_dt,
    const sycl::event &output_ready) {
  return dispatch_split(q, hidden, topk_ids, topk_weights, gate_qweight,
                        gate_scales, up_qweight, up_scales, down_qweight,
                        down_scales, scratch_f32, out_f32, expert_stride_bytes,
                        group_size, M, E, top_k, K, I, multiply_router_weight,
                        act_dt, output_ready, nullptr);
}

sycl::event int4_moe_split_profiled_sycl(
    sycl::queue &q, const void *hidden, const int *topk_ids,
    const float *topk_weights, const std::int32_t *gate_qweight,
    const half_t *gate_scales, const std::int32_t *up_qweight,
    const half_t *up_scales, const std::int32_t *down_qweight,
    const half_t *down_scales, float *scratch_f32, float *out_f32,
    std::size_t expert_stride_bytes, std::size_t group_size, std::size_t M,
    std::size_t E, std::size_t top_k, std::size_t K, std::size_t I,
    bool multiply_router_weight, DType act_dt,
    const sycl::event &output_ready, Int4MoeSplitStages *stages) {
  return dispatch_split(q, hidden, topk_ids, topk_weights, gate_qweight,
                        gate_scales, up_qweight, up_scales, down_qweight,
                        down_scales, scratch_f32, out_f32, expert_stride_bytes,
                        group_size, M, E, top_k, K, I, multiply_router_weight,
                        act_dt, output_ready, stages);
}
sycl::event int4_moe_split_down_wide_sycl(
    sycl::queue &q, const void *hidden, const int *topk_ids,
    const float *topk_weights, const std::int32_t *gate_qweight,
    const half_t *gate_scales, const std::int32_t *up_qweight,
    const half_t *up_scales, const std::int32_t *down_qweight,
    const half_t *down_scales, float *scratch_f32, float *out_f32,
    std::size_t expert_stride_bytes, std::size_t group_size, std::size_t M,
    std::size_t E, std::size_t top_k, std::size_t K, std::size_t I,
    bool multiply_router_weight, DType act_dt, int outputs_per_lane,
    const sycl::event &output_ready, Int4MoeSplitStages *stages) {
  switch (act_dt) {
  case DType::f32:
    return dispatch_down_wide_typed(
        q, static_cast<const float *>(hidden), topk_ids, topk_weights,
        gate_qweight, gate_scales, up_qweight, up_scales, down_qweight,
        down_scales, scratch_f32, out_f32, expert_stride_bytes, group_size, M,
        E, top_k, K, I, multiply_router_weight, outputs_per_lane,
        output_ready, stages);
  case DType::f16:
    return dispatch_down_wide_typed(
        q, static_cast<const half_t *>(hidden), topk_ids, topk_weights,
        gate_qweight, gate_scales, up_qweight, up_scales, down_qweight,
        down_scales, scratch_f32, out_f32, expert_stride_bytes, group_size, M,
        E, top_k, K, I, multiply_router_weight, outputs_per_lane,
        output_ready, stages);
  case DType::bf16:
    return dispatch_down_wide_typed(
        q, static_cast<const bf16_t *>(hidden), topk_ids, topk_weights,
        gate_qweight, gate_scales, up_qweight, up_scales, down_qweight,
        down_scales, scratch_f32, out_f32, expert_stride_bytes, group_size, M,
        E, top_k, K, I, multiply_router_weight, outputs_per_lane,
        output_ready, stages);
  }
  return {};
}


} // namespace quixicore::xpu::kernels
