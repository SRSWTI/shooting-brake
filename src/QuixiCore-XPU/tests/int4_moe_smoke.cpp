// Independent CPU-reference correctness test for AutoGPTQ int4 routed MoE.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/ops.hpp"
#include "quixicore/xpu/runtime.hpp"
#include "moe/int4_moe/int4_moe_kernel.hpp"


namespace {

using namespace quixicore::xpu;

constexpr std::size_t kM = 2;
constexpr std::size_t kE = 4;
constexpr std::size_t kTopK = 4;
constexpr std::size_t kK = 256;
constexpr std::size_t kI = 128;
constexpr std::size_t kGroupSize = 128;
constexpr int kZeroPoint = 8;

struct BankLayout {
  std::size_t gate_q;
  std::size_t gate_s;
  std::size_t up_q;
  std::size_t up_s;
  std::size_t down_q;
  std::size_t down_s;
  std::size_t stride;
};

BankLayout bank_layout() {
  const std::size_t gate_q_bytes = (kK / 8) * kI * sizeof(std::int32_t);
  const std::size_t gate_s_bytes = (kK / kGroupSize) * kI * sizeof(half_t);
  const std::size_t down_q_bytes = (kI / 8) * kK * sizeof(std::int32_t);
  const std::size_t down_s_bytes = (kI / kGroupSize) * kK * sizeof(half_t);
  BankLayout p{};
  p.gate_q = 0;
  p.gate_s = p.gate_q + gate_q_bytes;
  p.up_q = p.gate_s + gate_s_bytes;
  p.up_s = p.up_q + gate_q_bytes;
  p.down_q = p.up_s + gate_s_bytes;
  p.down_s = p.down_q + down_q_bytes;
  p.stride = p.down_s + down_s_bytes;
  return p;
}

template <typename T>
T *plane(std::uint8_t *bank, std::size_t expert, std::size_t offset,
         std::size_t stride) {
  return reinterpret_cast<T *>(bank + expert * stride + offset);
}

template <typename T>
const T *plane(const std::uint8_t *bank, std::size_t expert,
               std::size_t offset, std::size_t stride) {
  return reinterpret_cast<const T *>(bank + expert * stride + offset);
}

// Expert 0 has 80% exact-zero nibbles (a sparse late-layer-like block), expert
// 1 has 7% (a dense early-layer-like block), and the remaining experts sit in
// between. The hash depends on k and n so every packed word is heterogeneous.
int fixture_nibble(std::size_t expert, std::size_t projection,
                   std::size_t k, std::size_t n) {
  const std::uint32_t h = static_cast<std::uint32_t>(
      k * 2246822519u + n * 3266489917u + projection * 668265263u +
      expert * 374761393u);
  const unsigned percentile = (h >> 8) % 100u;
  const unsigned zero_percent = expert == 0 ? 80u : (expert == 1 ? 7u : 38u);
  if (percentile < zero_percent)
    return kZeroPoint;
  int value = static_cast<int>((h >> 19) & 0x0fu);
  if (value == kZeroPoint)
    value = (value + 3) & 0x0f;
  return value;
}

void fill_qweight(std::int32_t *dst, std::size_t expert,
                  std::size_t projection, std::size_t reduction,
                  std::size_t output, std::size_t &zeros,
                  std::size_t &elements) {
  for (std::size_t packed_k = 0; packed_k < reduction / 8; ++packed_k) {
    for (std::size_t n = 0; n < output; ++n) {
      std::uint32_t word = 0;
      for (int slot = 0; slot < 8; ++slot) {
        const int nibble = fixture_nibble(expert, projection,
                                          packed_k * 8 + slot, n);
        word |= static_cast<std::uint32_t>(nibble) << (4 * slot);
        zeros += nibble == kZeroPoint ? 1 : 0;
        ++elements;
      }
      dst[packed_k * output + n] = static_cast<std::int32_t>(word);
    }
  }
}

void fill_scales(half_t *dst, std::size_t count, std::size_t expert,
                 std::size_t projection, std::size_t &negative) {
  // Deliberately non-power-of-two values with nonzero mantissas. The negative
  // entry exercises the checkpoint's legal signed-scale case.
  constexpr float values[] = {0.0137f, -0.0213f, 0.0379f, 0.00917f,
                              0.0286f};
  for (std::size_t i = 0; i < count; ++i) {
    const float value = values[(i * 7 + expert * 3 + projection) % std::size(values)];
    dst[i] = static_cast<half_t>(value);
    negative += value < 0.0f ? 1 : 0;
  }
}

float dequant(const std::int32_t *qweight, const half_t *scales,
              std::size_t reduction_index, std::size_t output_index,
              std::size_t output) {
  const std::uint32_t word = static_cast<std::uint32_t>(
      qweight[(reduction_index / 8) * output + output_index]);
  const int nibble = static_cast<int>(
      (word >> (4 * (reduction_index % 8))) & 0x0fu);
  return static_cast<float>(nibble - kZeroPoint) *
         static_cast<float>(scales[(reduction_index / kGroupSize) * output +
                                   output_index]);
}

template <typename T>
bool run_case(sycl::queue &q, DType dtype) {
  const BankLayout p = bank_layout();
  std::uint8_t *bank = sycl::malloc_shared<std::uint8_t>(kE * p.stride, q);
  T *hidden = sycl::malloc_shared<T>(kM * kK, q);
  int *ids = sycl::malloc_shared<int>(kM * kTopK, q);
  float *router = sycl::malloc_shared<float>(kM * kTopK, q);
  float *scratch = sycl::malloc_shared<float>(kM * kTopK * kI, q);
  float *output = sycl::malloc_shared<float>(kM * kK, q);
  float *wide2 = sycl::malloc_shared<float>(kM * kK, q);
  float *wide4 = sycl::malloc_shared<float>(kM * kK, q);


  std::vector<std::size_t> zeros(kE, 0), elements(kE, 0);
  std::size_t negative_scales = 0;
  for (std::size_t expert = 0; expert < kE; ++expert) {
    fill_qweight(plane<std::int32_t>(bank, expert, p.gate_q, p.stride),
                 expert, 0, kK, kI, zeros[expert], elements[expert]);
    fill_scales(plane<half_t>(bank, expert, p.gate_s, p.stride),
                (kK / kGroupSize) * kI, expert, 0, negative_scales);
    fill_qweight(plane<std::int32_t>(bank, expert, p.up_q, p.stride),
                 expert, 1, kK, kI, zeros[expert], elements[expert]);
    fill_scales(plane<half_t>(bank, expert, p.up_s, p.stride),
                (kK / kGroupSize) * kI, expert, 1, negative_scales);
    fill_qweight(plane<std::int32_t>(bank, expert, p.down_q, p.stride),
                 expert, 2, kI, kK, zeros[expert], elements[expert]);
    fill_scales(plane<half_t>(bank, expert, p.down_s, p.stride),
                (kI / kGroupSize) * kK, expert, 2, negative_scales);
  }

  for (std::size_t i = 0; i < kM * kK; ++i) {
    const float value =
        0.031f * (static_cast<float>((i * 1103515245u >> 17) & 31u) - 15.5f) /
        15.5f;
    hidden[i] = static_cast<T>(value);
  }
  const int route_ids[kM * kTopK] = {0, 1, 3, -1, 1, 0, 3,
                                     static_cast<int>(kE)};
  const float route_weights[kM * kTopK] = {0.31f, 0.27f, 0.19f, 0.23f,
                                            0.29f, 0.25f, 0.21f, 0.25f};
  std::copy(std::begin(route_ids), std::end(route_ids), ids);
  std::copy(std::begin(route_weights), std::end(route_weights), router);

  ops::int4_moe_split(
      q, hidden, ids, router,
      plane<std::int32_t>(bank, 0, p.gate_q, p.stride),
      plane<half_t>(bank, 0, p.gate_s, p.stride),
      plane<std::int32_t>(bank, 0, p.up_q, p.stride),
      plane<half_t>(bank, 0, p.up_s, p.stride),
      plane<std::int32_t>(bank, 0, p.down_q, p.stride),
      plane<half_t>(bank, 0, p.down_s, p.stride), scratch, output, p.stride,
      kGroupSize, kM, kE, kTopK, kK, kI, dtype, true, Variant::sycl, true);
  const sycl::event zero2 = q.memset(wide2, 0, kM * kK * sizeof(float));
  kernels::int4_moe_split_down_wide_sycl(
      q, hidden, ids, router,
      plane<std::int32_t>(bank, 0, p.gate_q, p.stride),
      plane<half_t>(bank, 0, p.gate_s, p.stride),
      plane<std::int32_t>(bank, 0, p.up_q, p.stride),
      plane<half_t>(bank, 0, p.up_s, p.stride),
      plane<std::int32_t>(bank, 0, p.down_q, p.stride),
      plane<half_t>(bank, 0, p.down_s, p.stride), scratch, wide2, p.stride,
      kGroupSize, kM, kE, kTopK, kK, kI, true, dtype, 2, zero2)
      .wait();
  const sycl::event zero4 = q.memset(wide4, 0, kM * kK * sizeof(float));
  kernels::int4_moe_split_down_wide_sycl(
      q, hidden, ids, router,
      plane<std::int32_t>(bank, 0, p.gate_q, p.stride),
      plane<half_t>(bank, 0, p.gate_s, p.stride),
      plane<std::int32_t>(bank, 0, p.up_q, p.stride),
      plane<half_t>(bank, 0, p.up_s, p.stride),
      plane<std::int32_t>(bank, 0, p.down_q, p.stride),
      plane<half_t>(bank, 0, p.down_s, p.stride), scratch, wide4, p.stride,
      kGroupSize, kM, kE, kTopK, kK, kI, true, dtype, 4, zero4)
      .wait();


  std::vector<float> reference(kM * kK, 0.0f);
  std::vector<float> activated(kI);
  for (std::size_t m = 0; m < kM; ++m) {
    for (std::size_t route = 0; route < kTopK; ++route) {
      const std::size_t pair = m * kTopK + route;
      const int expert_id = ids[pair];
      if (expert_id < 0 || static_cast<std::size_t>(expert_id) >= kE)
        continue;
      const std::size_t expert = static_cast<std::size_t>(expert_id);
      const auto *gate_q =
          plane<std::int32_t>(bank, expert, p.gate_q, p.stride);
      const auto *gate_s = plane<half_t>(bank, expert, p.gate_s, p.stride);
      const auto *up_q = plane<std::int32_t>(bank, expert, p.up_q, p.stride);
      const auto *up_s = plane<half_t>(bank, expert, p.up_s, p.stride);
      const auto *down_q =
          plane<std::int32_t>(bank, expert, p.down_q, p.stride);
      const auto *down_s = plane<half_t>(bank, expert, p.down_s, p.stride);

      for (std::size_t n = 0; n < kI; ++n) {
        float gate = 0.0f;
        float up = 0.0f;
        for (std::size_t k = 0; k < kK; ++k) {
          const float x = static_cast<float>(hidden[m * kK + k]);
          gate = std::fma(dequant(gate_q, gate_s, k, n, kI), x, gate);
          up = std::fma(dequant(up_q, up_s, k, n, kI), x, up);
        }
        activated[n] = (gate / (1.0f + std::exp(-gate))) * up;
      }
      for (std::size_t n = 0; n < kK; ++n) {
        float acc = 0.0f;
        for (std::size_t i = 0; i < kI; ++i)
          acc = std::fma(dequant(down_q, down_s, i, n, kK), activated[i], acc);
        reference[m * kK + n] += router[pair] * acc;
      }
    }
  }

  double peak = 0.0;
  double max_abs = 0.0;
  double wide2_max_abs = 0.0;
  double wide4_max_abs = 0.0;
  for (float ref : reference)
    peak = std::max(peak, std::abs(static_cast<double>(ref)));
  std::size_t nonfinite = 0;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    if (!std::isfinite(output[i])) {
      ++nonfinite;
      continue;
    }
    const double abs_error =
        std::abs(static_cast<double>(output[i] - reference[i]));
    max_abs = std::max(max_abs, abs_error);
    wide2_max_abs = std::max(
        wide2_max_abs,
        std::abs(static_cast<double>(wide2[i] - reference[i])));
    wide4_max_abs = std::max(
        wide4_max_abs,
        std::abs(static_cast<double>(wide4[i] - reference[i])));
  }
  // Peak-normalized max error is stable at outputs crossing zero, where an
  // elementwise relative error is undefined and a tiny absolute difference
  // otherwise reports an arbitrarily large ratio.
  const double max_relative = max_abs / std::max(peak, 1e-12);
  const double wide2_relative = wide2_max_abs / std::max(peak, 1e-12);
  const double wide4_relative = wide4_max_abs / std::max(peak, 1e-12);

  const double sparse_fraction =
      static_cast<double>(zeros[0]) / static_cast<double>(elements[0]);
  const double dense_fraction =
      static_cast<double>(zeros[1]) / static_cast<double>(elements[1]);
  const bool empty_expert =
      std::find(std::begin(route_ids), std::end(route_ids), 2) ==
      std::end(route_ids);
  // Kernel and oracle both accumulate fp32, but the kernel uses an eight-way
  // gate/up and four-way down reduction tree. The allowance covers that tree,
  // device-vs-libm exp, and nondeterministic fp32 atomic route summation. It is
  // over 12x the measured ~4.1e-7 peak-relative error yet still two orders
  // below one fp16 input ulp; the input is already rounded identically.
  constexpr double kRelativeBound = 5e-6;
  const bool fixture_ok = sparse_fraction > 0.75 && sparse_fraction < 0.85 &&
                          dense_fraction > 0.04 && dense_fraction < 0.10 &&
                          negative_scales != 0 && empty_expert;
  const bool ok = fixture_ok && nonfinite == 0 &&
                  max_relative <= kRelativeBound &&
                  wide2_relative <= kRelativeBound &&
                  wide4_relative <= kRelativeBound;
  std::cout << "int4_moe_split dtype=" << dtype_name(dtype)
            << " max_relative=" << max_relative << " bound=" << kRelativeBound
            << " max_abs=" << max_abs << " reference_peak=" << peak
            << " down2_relative=" << wide2_relative
            << " down4_relative=" << wide4_relative
            << " sparse_zero_fraction=" << sparse_fraction
            << " dense_zero_fraction=" << dense_fraction
            << " negative_scales=" << negative_scales
            << " empty_expert=" << empty_expert << (ok ? "  ok" : "  FAIL")
            << '\n';

  sycl::free(bank, q);
  sycl::free(hidden, q);
  sycl::free(ids, q);
  sycl::free(router, q);
  sycl::free(scratch, q);
  sycl::free(output, q);
  sycl::free(wide2, q);
  sycl::free(wide4, q);
  return ok;
}

} // namespace

int main() {
  const auto devices = quixicore::xpu::gpu_devices();
  if (devices.empty()) {
    std::cout << "no SYCL GPU device; skipping int4_moe_smoke\n";
    return 0;
  }
  sycl::queue q = quixicore::xpu::make_gpu_queue();
  int failures = 0;
  failures += run_case<quixicore::xpu::half_t>(q,
                                               quixicore::xpu::DType::f16)
                  ? 0
                  : 1;
  failures += run_case<quixicore::xpu::bf16_t>(q,
                                               quixicore::xpu::DType::bf16)
                  ? 0
                  : 1;
  return failures == 0 ? 0 : 1;
}
