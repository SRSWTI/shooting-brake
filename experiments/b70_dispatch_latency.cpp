// Decode-dispatch latency decomposition for the B70, with no model involved.
//
// Measured on this box (hybrid-subset-16-8.json + perf/harness/xpu_bench.cpp):
//
//   B70 MoE kernel, M=1, top_k 8 ......  52.8 us
//   Dispatch service time ............. 186.1 us
//   ------------------------------------------
//   Overhead (launch/signal/copy) ..... 133.3 us   <- 72%, never measured
//
// Everything here targets that 133 us. A decode dispatch is tiny -- 4 KiB of
// activations out, 8 KiB of fp32 partials back -- so it is pure latency, and
// latency is reproducible without a single model weight.
//
// Modes:
//   real-kernel  production NVFP4 split kernel latency by idle duty cycle.
//   int4-graph production-geometry seven-submit eager vs command-graph replay.
//   launch      empty kernel submit + wait. Pure launch cost.
//   copy        H2D/D2H round trip across payload sizes. Latency floor vs bandwidth.
//   dispatch    H2D + trivial kernel + D2H + wait. The current per-layer shape.
//   environment  dispatch latency by host-buffer provenance and idle gap.
//   persistent  one long-lived kernel spinning on a host doorbell. The proposal.
//   queues      N concurrent dispatches on N queues. Multi-queue headroom.
//
// The persistent kernel ALWAYS bounds its spin so it cannot wedge the device.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>
#include <limits>

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <cuda_runtime_api.h>
#include "quixicore/xpu/ops.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double us_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}

struct Stats {
  double median, p10, p90, min;
};

Stats summarize(std::vector<double> &v) {
  std::sort(v.begin(), v.end());
  auto at = [&](double q) { return v[std::min(v.size() - 1, static_cast<std::size_t>(q * v.size()))]; };
  return Stats{at(0.50), at(0.10), at(0.90), v.front()};
}

void report(const char *label, Stats s, const char *unit = "us") {
  std::printf("  %-34s median %8.2f %s   p10 %8.2f   p90 %8.2f   min %8.2f\n",
              label, s.median, unit, s.p10, s.p90, s.min);
}

// Decode shape: hidden 2048 f16 out, 2048 fp32 partials back, per layer.
constexpr std::size_t kActBytes = 2048 * sizeof(sycl::half);
constexpr std::size_t kOutBytes = 2048 * sizeof(float);

// ---------------------------------------------------------------- launch ---

void bench_launch(sycl::queue &q, int iters) {
  std::printf("\n[launch] empty kernel submit + wait\n");
  for (int i = 0; i < 200; ++i) q.single_task([]() {}).wait();

  std::vector<double> submit_wait, submit_only;
  submit_wait.reserve(iters);
  submit_only.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    auto t0 = Clock::now();
    sycl::event e = q.single_task([]() {});
    submit_only.push_back(us_since(t0));
    e.wait();
    submit_wait.push_back(us_since(t0));
  }
  report("submit only (no wait)", summarize(submit_only));
  report("submit + wait", summarize(submit_wait));
}

// ------------------------------------------------------------------ copy ---

void bench_copy(sycl::queue &q, int iters) {
  std::printf("\n[copy] pinned host <-> device round trip by payload size\n");
  const std::size_t sizes[] = {1024, 4096, 12288, 65536, 262144, 1048576, 4194304};
  for (std::size_t bytes : sizes) {
    void *host = sycl::malloc_host(bytes, q);
    void *dev = sycl::malloc_device(bytes, q);
    std::memset(host, 0xA5, bytes);
    for (int i = 0; i < 50; ++i) { q.memcpy(dev, host, bytes).wait(); }

    std::vector<double> rt;
    rt.reserve(iters);
    for (int i = 0; i < iters; ++i) {
      auto t0 = Clock::now();
      q.memcpy(dev, host, bytes).wait();
      q.memcpy(host, dev, bytes).wait();
      rt.push_back(us_since(t0));
    }
    Stats s = summarize(rt);
    const double gbps = (2.0 * bytes) / (s.median * 1e-6) / 1e9;
    std::printf("  %8zu B  round trip median %8.2f us   min %8.2f   -> %6.2f GB/s\n",
                bytes, s.median, s.min, gbps);
    sycl::free(host, q);
    sycl::free(dev, q);
  }
}

// -------------------------------------------------------------- dispatch ---

// One decode layer as the provider does it today: copy activations in, launch
// a kernel, copy partials out, wait.
void bench_dispatch(sycl::queue &q, int iters) {
  std::printf("\n[dispatch] H2D %zu B + kernel + D2H %zu B + wait (one decode layer)\n",
              kActBytes, kOutBytes);
  void *h_act = sycl::malloc_host(kActBytes, q);
  void *d_act = sycl::malloc_device(kActBytes, q);
  float *h_out = static_cast<float *>(sycl::malloc_host(kOutBytes, q));
  float *d_out = sycl::malloc_device<float>(kOutBytes / sizeof(float), q);
  std::memset(h_act, 0x3C, kActBytes);

  auto one = [&] {
    q.memcpy(d_act, h_act, kActBytes);
    q.parallel_for(sycl::range<1>(2048), [=](sycl::id<1> i) { d_out[i] = static_cast<float>(i[0]); });
    q.memcpy(h_out, d_out, kOutBytes);
    q.wait();
  };
  for (int i = 0; i < 100; ++i) one();

  std::vector<double> rt;
  rt.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    auto t0 = Clock::now();
    one();
    rt.push_back(us_since(t0));
  }
  Stats s = summarize(rt);
  report("full dispatch (chained, one wait)", s);
  std::printf("    -> 16 layers/token: %6.2f us    48 layers/token: %6.2f us\n",
              s.median * 16, s.median * 48);

  sycl::free(h_act, q); sycl::free(d_act, q);
  sycl::free(h_out, q); sycl::free(d_out, q);
}

// ----------------------------------------------------------- environment ---

struct EnvironmentStats {
  double median, p95;
};

EnvironmentStats summarize_environment(std::vector<double> &v) {
  std::sort(v.begin(), v.end());
  auto at = [&](double q) {
    return v[std::min(v.size() - 1,
                      static_cast<std::size_t>(q * v.size()))];
  };
  return EnvironmentStats{at(0.50), at(0.95)};
}

void spin_for_us(int idle_us) {
  const auto until = Clock::now() + std::chrono::microseconds(idle_us);
  while (Clock::now() < until) {
  }
}

enum class HostProvenance {
  sycl_host,          // sycl::malloc_host in the B70's context
  cuda_pinned,        // cudaHostAlloc — production pin_memory=True provenance
  cuda_pinned_reg,    // cudaHostAlloc + prepare_for_device_copy (the 1b fix)
  malloc_reg,         // plain malloc + prepare_for_device_copy
};

const char *provenance_name(HostProvenance p) {
  switch (p) {
  case HostProvenance::sycl_host:       return "sycl::malloc_host";
  case HostProvenance::cuda_pinned:     return "cudaHostAlloc";
  case HostProvenance::cuda_pinned_reg: return "cudaHostAlloc+reg";
  case HostProvenance::malloc_reg:      return "malloc+reg";
  }
  return "?";
}

EnvironmentStats bench_environment_cell(sycl::queue &q, int iters,
                                        HostProvenance prov, int idle_us) {
  namespace syclex = sycl::ext::oneapi::experimental;
  const bool cuda_pinned = prov == HostProvenance::cuda_pinned ||
                           prov == HostProvenance::cuda_pinned_reg;
  const bool registered = prov == HostProvenance::cuda_pinned_reg ||
                          prov == HostProvenance::malloc_reg;
  void *h_act = nullptr;
  float *h_out = nullptr;
  if (cuda_pinned) {
    cudaError_t status =
        cudaHostAlloc(&h_act, kActBytes, cudaHostAllocPortable);
    if (status == cudaSuccess) {
      status = cudaHostAlloc(reinterpret_cast<void **>(&h_out), kOutBytes,
                             cudaHostAllocPortable);
    }
    if (status != cudaSuccess) {
      if (h_act != nullptr) cudaFreeHost(h_act);
      throw std::runtime_error(
          std::string("cudaHostAlloc failed: ") + cudaGetErrorString(status));
    }
  } else if (prov == HostProvenance::malloc_reg) {
    h_act = std::malloc(kActBytes);
    h_out = static_cast<float *>(std::malloc(kOutBytes));
    if (h_act == nullptr || h_out == nullptr) throw std::bad_alloc();
  } else {
    h_act = sycl::malloc_host(kActBytes, q);
    h_out = sycl::malloc_host<float>(kOutBytes / sizeof(float), q);
  }
  if (registered) {
    // The 1b primitive: make an externally-allocated range DMA-able from
    // this SYCL context. Same call vllm-xpu-kernels' xpu_host_register wraps.
    syclex::prepare_for_device_copy(h_act, kActBytes, q.get_context());
    syclex::prepare_for_device_copy(h_out, kOutBytes, q.get_context());
  }
  void *d_act = sycl::malloc_device(kActBytes, q);
  float *d_out = sycl::malloc_device<float>(kOutBytes / sizeof(float), q);
  std::memset(h_act, 0x3C, kActBytes);

  auto one = [&] {
    q.memcpy(d_act, h_act, kActBytes);
    q.parallel_for(sycl::range<1>(2048), [=](sycl::id<1> i) {
      d_out[i] = static_cast<float>(i[0]);
    });
    q.memcpy(h_out, d_out, kOutBytes);
    q.wait();
  };
  for (int i = 0; i < 100; ++i) one();

  std::vector<double> rt;
  rt.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    spin_for_us(idle_us);
    const auto t0 = Clock::now();
    one();
    rt.push_back(us_since(t0));
  }

  sycl::free(d_act, q);
  sycl::free(d_out, q);
  if (registered) {
    syclex::release_from_device_copy(h_act, q.get_context());
    syclex::release_from_device_copy(h_out, q.get_context());
  }
  if (cuda_pinned) {
    cudaFreeHost(h_act);
    cudaFreeHost(h_out);
  } else if (prov == HostProvenance::malloc_reg) {
    std::free(h_act);
    std::free(h_out);
  } else {
    sycl::free(h_act, q);
    sycl::free(h_out, q);
  }
  return summarize_environment(rt);
}

void bench_environment(sycl::queue &q, int iters) {
  std::printf(
      "\n[environment] full dispatch by host provenance and idle gap\n");
  std::printf("  %-18s %10s %12s\n", "host buffer", "gap", "latency");
  for (HostProvenance prov :
       {HostProvenance::sycl_host, HostProvenance::cuda_pinned,
        HostProvenance::cuda_pinned_reg, HostProvenance::malloc_reg}) {
    for (int idle_us : {0, 180}) {
      const EnvironmentStats s =
          bench_environment_cell(q, iters, prov, idle_us);
      std::printf("  %-18s %7d us   median %8.2f us   p95 %8.2f us\n",
                  provenance_name(prov), idle_us, s.median, s.p95);
    }
  }
}

// ----------------------------------------------------------- real-kernel ---

int read_act_freq() {
  const char *path = std::getenv("SHOOTING_BRAKE_B70_ACT_FREQ");
  if (path == nullptr) {
    path = "/sys/class/drm/card2/device/tile0/gt0/freq0/act_freq";
  }
  std::ifstream input(path);
  int value = -1;
  return input >> value ? value : -1;
}

double median_frequency(std::vector<int> &values) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  return static_cast<double>(values[values.size() / 2]);
}

void bench_real_kernel(sycl::device dev, int iters) {
  constexpr std::size_t M = 1;
  constexpr std::size_t E = 8;
  constexpr std::size_t top_k = 8;
  constexpr std::size_t K = 2048;
  constexpr std::size_t I = 512;
  constexpr std::size_t pairs = M * top_k;
  sycl::queue q(
      dev, {sycl::property::queue::in_order{},
            sycl::property::queue::enable_profiling{}});

  auto *hidden = sycl::malloc_device<sycl::half>(M * K, q);
  auto *expert_ids = sycl::malloc_device<int>(pairs, q);
  auto *router_weights = sycl::malloc_device<float>(pairs, q);
  void *w13 = sycl::malloc_device(E * 2 * I * K / 2, q);
  void *w13_scales = sycl::malloc_device(E * 2 * I * K / 16, q);
  auto *w13_global = sycl::malloc_device<float>(E, q);
  void *w2 = sycl::malloc_device(E * K * I / 2, q);
  void *w2_scales = sycl::malloc_device(E * K * I / 16, q);
  auto *w2_global = sycl::malloc_device<float>(E, q);
  auto *scratch = sycl::malloc_device<float>(pairs * 2 * I, q);
  auto *output = sycl::malloc_device<float>(M * K, q);

  std::vector<sycl::half> host_hidden(M * K);
  for (std::size_t i = 0; i < host_hidden.size(); ++i) {
    host_hidden[i] = static_cast<sycl::half>(
        0.05f * (static_cast<float>((i * 1103515245u >> 17) & 7u) - 3.5f) /
        3.5f);
  }
  const int host_ids[pairs] = {0, 1, 2, 3, 4, 5, 6, 7};
  q.memcpy(hidden, host_hidden.data(), M * K * sizeof(sycl::half)).wait();
  q.memcpy(expert_ids, host_ids, pairs * sizeof(int)).wait();
  q.fill(router_weights, 1.0f / static_cast<float>(top_k), pairs).wait();

  const std::size_t w13_bytes = E * 2 * I * K / 2;
  const std::size_t w2_bytes = E * K * I / 2;
  std::vector<std::uint8_t> pack(std::max(w13_bytes, w2_bytes));
  for (std::size_t i = 0; i < pack.size(); ++i) {
    pack[i] =
        static_cast<std::uint8_t>((i * 71u + (i >> 5)) & 0x7Fu) | 0x11u;
  }
  q.memcpy(w13, pack.data(), w13_bytes).wait();
  q.memcpy(w2, pack.data(), w2_bytes).wait();

  const std::size_t s13_bytes = E * 2 * I * K / 16;
  const std::size_t s2_bytes = E * K * I / 16;
  static constexpr std::uint8_t kScales[4] = {0x3Bu, 0x2Du, 0x35u, 0x43u};
  std::vector<std::uint8_t> scales(std::max(s13_bytes, s2_bytes));
  for (std::size_t i = 0; i < scales.size(); ++i) {
    scales[i] = kScales[i & 3];
  }
  q.memcpy(w13_scales, scales.data(), s13_bytes).wait();
  q.memcpy(w2_scales, scales.data(), s2_bytes).wait();
  q.fill(w13_global, 0.02f, E).wait();
  q.fill(w2_global, 0.03f, E).wait();

  auto submit = [&] {
    sycl::event begin = q.single_task([] {});
    quixicore::xpu::ops::nvfp4_moe_split(
        q, hidden, expert_ids, router_weights, w13, w13_scales, w13_global,
        w2, w2_scales, w2_global, scratch, output, M, E, top_k, K, I,
        quixicore::xpu::DType::f16, true, quixicore::xpu::Variant::sycl,
        false);
    sycl::event end = q.single_task([] {});
    end.wait_and_throw();
    const auto start =
        begin.get_profiling_info<sycl::info::event_profiling::command_end>();
    const auto stop =
        end.get_profiling_info<sycl::info::event_profiling::command_start>();
    return static_cast<double>(stop - start) * 1.0e-3;
  };

  std::printf(
      "\n[real-kernel] nvfp4_moe_split M=1 E=8 top_k=8 K=2048 I=512 f16\n");
  std::printf("  %8s %20s %20s %10s\n", "gap", "kernel median", "act_freq median",
              "samples");
  for (int idle_us : {0, 50, 100, 200, 500, 1000}) {
    for (int i = 0; i < 500; ++i) {
      spin_for_us(idle_us);
      submit();
    }

    std::atomic<bool> sampling{true};
    std::vector<int> frequencies;
    std::thread sampler([&] {
      auto next = Clock::now();
      while (sampling.load(std::memory_order_relaxed)) {
        const int value = read_act_freq();
        if (value >= 0) frequencies.push_back(value);
        next += std::chrono::milliseconds(5);
        std::this_thread::sleep_until(next);
      }
    });

    std::vector<double> kernel_us;
    kernel_us.reserve(iters);
    for (int i = 0; i < iters; ++i) {
      spin_for_us(idle_us);
      kernel_us.push_back(submit());
    }
    sampling.store(false, std::memory_order_relaxed);
    sampler.join();
    const Stats stats = summarize(kernel_us);
    std::printf("  %5d us   %12.2f us   %12.0f MHz   %10zu\n", idle_us,
                stats.median, median_frequency(frequencies),
                frequencies.size());
  }

  sycl::free(hidden, q);
  sycl::free(expert_ids, q);
  sycl::free(router_weights, q);
  sycl::free(w13, q);
  sycl::free(w13_scales, q);
  sycl::free(w13_global, q);
  sycl::free(w2, q);
  sycl::free(w2_scales, q);
  sycl::free(w2_global, q);
  sycl::free(scratch, q);
  sycl::free(output, q);
}

// ----------------------------------------------------------- int4-graph ---

struct GraphStats {
  double median;
  double p95;
};

GraphStats summarize_graph(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  auto at = [&](double q) {
    return values[std::min(
        values.size() - 1,
        static_cast<std::size_t>(std::ceil(q * values.size()) - 1.0))];
  };
  return {at(0.50), at(0.95)};
}

constexpr std::size_t kGraphM = 1;
constexpr std::size_t kGraphE = 8;
constexpr std::size_t kGraphTopK = 8;
constexpr std::size_t kGraphK = 3072;
constexpr std::size_t kGraphI = 1024;
constexpr std::size_t kGraphGroup = 128;
constexpr std::size_t kGraphQBytes = kGraphK * kGraphI / 2;
constexpr std::size_t kGraphScaleBytes =
    (kGraphK / kGraphGroup) * kGraphI * sizeof(sycl::half);

struct Int4GraphLayout {
  std::size_t gate_q = 0;
  std::size_t gate_s = gate_q + kGraphQBytes;
  std::size_t up_q = gate_s + kGraphScaleBytes;
  std::size_t up_s = up_q + kGraphQBytes;
  std::size_t down_q = up_s + kGraphScaleBytes;
  std::size_t down_s = down_q + kGraphQBytes;
  std::size_t stride = down_s + kGraphScaleBytes;
};

template <typename T>
T *graph_plane(std::uint8_t *bank, std::size_t expert, std::size_t offset,
               std::size_t stride) {
  return reinterpret_cast<T *>(bank + expert * stride + offset);
}

template <typename T>
const T *graph_plane(const std::uint8_t *bank, std::size_t expert,
                     std::size_t offset, std::size_t stride) {
  return reinterpret_cast<const T *>(bank + expert * stride + offset);
}

void fill_graph_qweight(std::int32_t *destination, std::size_t expert,
                        std::size_t projection, std::size_t reduction,
                        std::size_t output) {
  for (std::size_t packed_k = 0; packed_k < reduction / 8; ++packed_k) {
    for (std::size_t n = 0; n < output; ++n) {
      std::uint32_t word = 0;
      for (std::size_t slot = 0; slot < 8; ++slot) {
        const std::uint32_t hash = static_cast<std::uint32_t>(
            (packed_k * 8 + slot) * 2246822519u + n * 3266489917u +
            expert * 374761393u + projection * 668265263u);
        const int nibble = 5 + static_cast<int>((hash >> 19) % 7u);
        word |= static_cast<std::uint32_t>(nibble) << (4 * slot);
      }
      destination[packed_k * output + n] = static_cast<std::int32_t>(word);
    }
  }
}

void fill_graph_scales(sycl::half *destination, std::size_t count,
                       std::size_t expert, std::size_t projection) {
  constexpr float values[] = {0.0067f, -0.0043f, 0.0091f, 0.0039f,
                              -0.0077f};
  for (std::size_t index = 0; index < count; ++index) {
    destination[index] = static_cast<sycl::half>(
        values[(index * 7 + expert * 3 + projection) % std::size(values)]);
  }
}

float graph_dequant(const std::int32_t *qweight, const sycl::half *scales,
                    std::size_t reduction_index, std::size_t output_index,
                    std::size_t output_size) {
  const std::uint32_t word = static_cast<std::uint32_t>(
      qweight[(reduction_index / 8) * output_size + output_index]);
  const int nibble = static_cast<int>(
      (word >> (4 * (reduction_index % 8))) & 0x0fu);
  return static_cast<float>(nibble - 8) *
         static_cast<float>(
             scales[(reduction_index / kGraphGroup) * output_size +
                    output_index]);
}

std::vector<float> int4_graph_cpu_reference(
    const std::uint8_t *bank, const Int4GraphLayout &layout,
    const sycl::half *hidden, const int *ids, const float *weights) {
  std::vector<float> reference(kGraphK, 0.0f);
  std::vector<float> activated(kGraphI);
  // Worst-case host weight/reference residency in this synthetic graph test is
  // 38,928,384 bank bytes plus under 64 KiB of dispatch/oracle buffers. The
  // loop reuses one activation vector and cannot approach the 4 GiB ceiling.
  for (std::size_t route = 0; route < kGraphTopK; ++route) {
    if (ids[route] < 0 ||
        ids[route] >= static_cast<int>(kGraphE)) {
      continue;
    }
    const std::size_t expert = static_cast<std::size_t>(ids[route]);
    const auto *gate_q = graph_plane<std::int32_t>(
        bank, expert, layout.gate_q, layout.stride);
    const auto *gate_s = graph_plane<sycl::half>(
        bank, expert, layout.gate_s, layout.stride);
    const auto *up_q = graph_plane<std::int32_t>(
        bank, expert, layout.up_q, layout.stride);
    const auto *up_s = graph_plane<sycl::half>(
        bank, expert, layout.up_s, layout.stride);
    const auto *down_q = graph_plane<std::int32_t>(
        bank, expert, layout.down_q, layout.stride);
    const auto *down_s = graph_plane<sycl::half>(
        bank, expert, layout.down_s, layout.stride);
    for (std::size_t n = 0; n < kGraphI; ++n) {
      float gate = 0.0f;
      float up = 0.0f;
      for (std::size_t k = 0; k < kGraphK; ++k) {
        const float input = static_cast<float>(hidden[k]);
        gate = std::fma(
            graph_dequant(gate_q, gate_s, k, n, kGraphI), input, gate);
        up = std::fma(
            graph_dequant(up_q, up_s, k, n, kGraphI), input, up);
      }
      activated[n] = (gate / (1.0f + std::exp(-gate))) * up;
    }
    for (std::size_t n = 0; n < kGraphK; ++n) {
      float down = 0.0f;
      for (std::size_t k = 0; k < kGraphI; ++k) {
        down = std::fma(
            graph_dequant(down_q, down_s, k, n, kGraphK), activated[k],
            down);
      }
      reference[n] = std::fma(weights[route], down, reference[n]);
    }
  }
  return reference;
}

void bench_int4_graph(sycl::device dev, int iters) {
  namespace graph = sycl::ext::oneapi::experimental;
  using ModifiableGraph =
      graph::command_graph<graph::graph_state::modifiable>;

  sycl::context context(dev);
  sycl::queue q(
      context, dev, {sycl::property::queue::in_order{},
                     sycl::property::queue::enable_profiling{}});
  sycl::queue graph_q(
      context, dev, {sycl::property::queue::in_order{},
                     sycl::property::queue::enable_profiling{}});
  if (!dev.has(sycl::aspect::ext_oneapi_graph)) {
    throw std::runtime_error("selected B70 lacks ext_oneapi_graph");
  }

  const Int4GraphLayout layout;
  static_assert(kGraphQBytes == 1572864);
  static_assert(kGraphScaleBytes == 49152);
  if (layout.stride != 4866048) {
    throw std::runtime_error("synthetic int4 expert stride is not 4,866,048");
  }

  std::vector<std::uint8_t> host_bank(kGraphE * layout.stride);
  for (std::size_t expert = 0; expert < kGraphE; ++expert) {
    fill_graph_qweight(
        graph_plane<std::int32_t>(host_bank.data(), expert, layout.gate_q,
                                  layout.stride),
        expert, 0, kGraphK, kGraphI);
    fill_graph_scales(
        graph_plane<sycl::half>(host_bank.data(), expert, layout.gate_s,
                                layout.stride),
        (kGraphK / kGraphGroup) * kGraphI, expert, 0);
    fill_graph_qweight(
        graph_plane<std::int32_t>(host_bank.data(), expert, layout.up_q,
                                  layout.stride),
        expert, 1, kGraphK, kGraphI);
    fill_graph_scales(
        graph_plane<sycl::half>(host_bank.data(), expert, layout.up_s,
                                layout.stride),
        (kGraphK / kGraphGroup) * kGraphI, expert, 1);
    fill_graph_qweight(
        graph_plane<std::int32_t>(host_bank.data(), expert, layout.down_q,
                                  layout.stride),
        expert, 2, kGraphI, kGraphK);
    fill_graph_scales(
        graph_plane<sycl::half>(host_bank.data(), expert, layout.down_s,
                                layout.stride),
        (kGraphI / kGraphGroup) * kGraphK, expert, 2);
  }

  auto *device_bank =
      sycl::malloc_device<std::uint8_t>(host_bank.size(), q);
  auto *device_hidden = sycl::malloc_device<sycl::half>(kGraphK, q);
  auto *device_ids = sycl::malloc_device<int>(kGraphTopK, q);
  auto *device_weights = sycl::malloc_device<float>(kGraphTopK, q);
  auto *device_scratch =
      sycl::malloc_device<float>(kGraphTopK * kGraphI, q);
  auto *device_output = sycl::malloc_device<float>(kGraphK, q);
  if (device_bank == nullptr || device_hidden == nullptr ||
      device_ids == nullptr || device_weights == nullptr ||
      device_scratch == nullptr || device_output == nullptr) {
    throw std::bad_alloc();
  }
  q.memcpy(device_bank, host_bank.data(), host_bank.size()).wait_and_throw();

  constexpr std::size_t hidden_bytes = kGraphK * sizeof(sycl::half);
  constexpr std::size_t ids_bytes = kGraphTopK * sizeof(int);
  constexpr std::size_t weights_bytes = kGraphTopK * sizeof(float);
  constexpr std::size_t output_bytes = kGraphK * sizeof(float);
  constexpr std::size_t ids_offset = hidden_bytes;
  constexpr std::size_t weights_offset = ids_offset + ids_bytes;
  constexpr std::size_t output_offset = weights_offset + weights_bytes;
  constexpr std::size_t arena_bytes = output_offset + output_bytes;
  auto *host_arena = sycl::malloc_host<std::uint8_t>(arena_bytes, q);
  if (host_arena == nullptr) {
    throw std::bad_alloc();
  }
  auto *host_hidden =
      reinterpret_cast<sycl::half *>(host_arena);
  auto *host_ids = reinterpret_cast<int *>(host_arena + ids_offset);
  auto *host_weights =
      reinterpret_cast<float *>(host_arena + weights_offset);
  auto *host_output =
      reinterpret_cast<float *>(host_arena + output_offset);

  auto set_hidden = [&](int salt) {
    for (std::size_t index = 0; index < kGraphK; ++index) {
      const int centered =
          static_cast<int>((index * 17 + static_cast<std::size_t>(salt)) % 29) -
          14;
      host_hidden[index] = static_cast<sycl::half>(
          0.015625f * static_cast<float>(centered) / 14.0f);
    }
  };
  auto set_routes = [&](int active_routes) {
    for (int route = 0; route < static_cast<int>(kGraphTopK); ++route) {
      host_ids[route] = route < active_routes ? route : -1;
      host_weights[route] =
          route < active_routes ? 1.0f / static_cast<float>(active_routes)
                                : 0.0f;
    }
  };

  auto submit_seven = [&](sycl::queue &queue) {
    queue.memcpy(device_hidden, host_hidden, hidden_bytes);
    queue.memcpy(device_ids, host_ids, ids_bytes);
    queue.memcpy(device_weights, host_weights, weights_bytes);
    quixicore::xpu::ops::int4_moe_split(
        queue, device_hidden, device_ids, device_weights,
        graph_plane<std::int32_t>(device_bank, 0, layout.gate_q,
                                  layout.stride),
        graph_plane<sycl::half>(device_bank, 0, layout.gate_s, layout.stride),
        graph_plane<std::int32_t>(device_bank, 0, layout.up_q, layout.stride),
        graph_plane<sycl::half>(device_bank, 0, layout.up_s, layout.stride),
        graph_plane<std::int32_t>(device_bank, 0, layout.down_q,
                                  layout.stride),
        graph_plane<sycl::half>(device_bank, 0, layout.down_s, layout.stride),
        device_scratch, device_output, layout.stride, kGraphGroup, kGraphM,
        kGraphE, kGraphTopK, kGraphK, kGraphI,
        quixicore::xpu::DType::f16, true, quixicore::xpu::Variant::sycl,
        false);
    queue.memcpy(host_output, device_output, output_bytes);
  };
  auto eager = [&] {
    submit_seven(q);
    q.wait_and_throw();
  };

  set_hidden(3);
  set_routes(8);
  const auto construct_start = Clock::now();
  ModifiableGraph modifiable(graph_q);
  const double construct_us = us_since(construct_start);
  const auto record_start = Clock::now();
  modifiable.begin_recording(graph_q);
  submit_seven(graph_q);
  modifiable.end_recording(graph_q);
  const double record_us = us_since(record_start);
  const auto finalize_start = Clock::now();
  auto executable = modifiable.finalize(sycl::property_list{
      graph::property::graph::enable_profiling{}});
  const double finalize_us = us_since(finalize_start);

  // Change every host input after capture. Correct CPU results below therefore
  // prove replay dereferences the same stable arena at replay time instead of
  // replaying stale captured contents.
  set_hidden(11);
  const std::array<int, 3> route_counts{1, 4, 8};
  double eager_graph_max_abs = 0.0;
  double eager_graph_peak = 0.0;
  std::vector<float> graph_output_k8;
  struct Row {
    int routes;
    GraphStats eager_wall;
    GraphStats graph_wall;
    GraphStats kernel_device;
    GraphStats graph_device_total;
  };
  std::vector<Row> rows;
  rows.reserve(route_counts.size());

  for (int active_routes : route_counts) {
    set_routes(active_routes);
    eager();
    std::vector<float> eager_output(host_output, host_output + kGraphK);
    graph_q.ext_oneapi_graph(executable).wait_and_throw();
    std::vector<float> graph_output(host_output, host_output + kGraphK);
    for (std::size_t index = 0; index < kGraphK; ++index) {
      eager_graph_max_abs =
          std::max(eager_graph_max_abs,
                   std::abs(static_cast<double>(eager_output[index]) -
                            static_cast<double>(graph_output[index])));
      eager_graph_peak =
          std::max(eager_graph_peak,
                   std::abs(static_cast<double>(eager_output[index])));
    }
    if (active_routes == 8) {
      graph_output_k8 = graph_output;
    }

    for (int warmup = 0; warmup < 50; ++warmup) {
      eager();
      graph_q.ext_oneapi_graph(executable).wait_and_throw();
    }
    std::vector<double> eager_wall;
    std::vector<double> graph_wall;
    std::vector<double> graph_device_total;
    eager_wall.reserve(iters);
    graph_wall.reserve(iters);
    graph_device_total.reserve(iters);
    for (int iteration = 0; iteration < iters; ++iteration) {
      auto start = Clock::now();
      eager();
      eager_wall.push_back(us_since(start));

      start = Clock::now();
      sycl::event replay = graph_q.ext_oneapi_graph(executable);
      replay.wait_and_throw();
      graph_wall.push_back(us_since(start));
      const auto device_start =
          replay.get_profiling_info<
              sycl::info::event_profiling::command_start>();
      const auto device_stop =
          replay.get_profiling_info<
              sycl::info::event_profiling::command_end>();
      graph_device_total.push_back(
          static_cast<double>(device_stop - device_start) * 1.0e-3);
    }

    q.memcpy(device_hidden, host_hidden, hidden_bytes);
    q.memcpy(device_ids, host_ids, ids_bytes);
    q.memcpy(device_weights, host_weights, weights_bytes).wait_and_throw();
    auto measure_kernel = [&] {
      sycl::event begin = q.single_task([] {});
      quixicore::xpu::ops::int4_moe_split(
          q, device_hidden, device_ids, device_weights,
          graph_plane<std::int32_t>(device_bank, 0, layout.gate_q,
                                    layout.stride),
          graph_plane<sycl::half>(device_bank, 0, layout.gate_s,
                                  layout.stride),
          graph_plane<std::int32_t>(device_bank, 0, layout.up_q,
                                    layout.stride),
          graph_plane<sycl::half>(device_bank, 0, layout.up_s,
                                  layout.stride),
          graph_plane<std::int32_t>(device_bank, 0, layout.down_q,
                                    layout.stride),
          graph_plane<sycl::half>(device_bank, 0, layout.down_s,
                                  layout.stride),
          device_scratch, device_output, layout.stride, kGraphGroup, kGraphM,
          kGraphE, kGraphTopK, kGraphK, kGraphI,
          quixicore::xpu::DType::f16, true, quixicore::xpu::Variant::sycl,
          false);
      sycl::event end = q.single_task([] {});
      end.wait_and_throw();
      const auto start =
          begin.get_profiling_info<
              sycl::info::event_profiling::command_end>();
      const auto stop =
          end.get_profiling_info<
              sycl::info::event_profiling::command_start>();
      return static_cast<double>(stop - start) * 1.0e-3;
    };
    for (int warmup = 0; warmup < 50; ++warmup) {
      measure_kernel();
    }
    std::vector<double> kernel_device;
    kernel_device.reserve(iters);
    for (int iteration = 0; iteration < iters; ++iteration) {
      kernel_device.push_back(measure_kernel());
    }
    rows.push_back({active_routes, summarize_graph(std::move(eager_wall)),
                    summarize_graph(std::move(graph_wall)),
                    summarize_graph(std::move(kernel_device)),
                    summarize_graph(std::move(graph_device_total))});
  }

  set_routes(8);
  const std::vector<float> cpu_reference = int4_graph_cpu_reference(
      host_bank.data(), layout, host_hidden, host_ids, host_weights);
  double cpu_max_abs = 0.0;
  double cpu_peak = 0.0;
  for (std::size_t index = 0; index < kGraphK; ++index) {
    cpu_max_abs =
        std::max(cpu_max_abs,
                 std::abs(static_cast<double>(graph_output_k8[index]) -
                          static_cast<double>(cpu_reference[index])));
    cpu_peak =
        std::max(cpu_peak, std::abs(static_cast<double>(cpu_reference[index])));
  }
  const double cpu_max_relative =
      cpu_max_abs / std::max(cpu_peak, 1.0e-30);
  const double eager_graph_max_relative =
      eager_graph_max_abs / std::max(eager_graph_peak, 1.0e-30);

  std::printf(
      "\n[int4-graph] seven-command eager chain vs one command-graph replay\n");
  std::printf(
      "  graph_supported=1 stable_host_arena=%p stable_device_hidden=%p "
      "stable_device_output=%p\n",
      static_cast<void *>(host_arena), static_cast<void *>(device_hidden),
      static_cast<void *>(device_output));
  std::printf(
      "  graph_construct_us=%.3f record_seven_commands_us=%.3f "
      "finalize_us=%.3f projected_32_bucket_finalize_ms=%.3f\n",
      construct_us, record_us, finalize_us, finalize_us * 32.0e-3);
  std::printf(
      "  M_contract captured_M=1 dynamic_host_contents=yes dynamic_M=no "
      "required_decode_graph_buckets=32 "
      "reason=captured_copy_sizes_and_kernel_ranges\n");
  std::printf(
      "  %3s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s\n",
      "k", "A_wall_p50", "A_wall_p95", "B_wall_p50", "B_wall_p95",
      "kernel_p50", "kernel_p95", "AminusK_p50", "BminusK_p50",
      "B_device_p50", "B_device_p95");
  for (const Row &row : rows) {
    std::printf(
        "  %3d %12.3f %12.3f %12.3f %12.3f %12.3f %12.3f %12.3f "
        "%12.3f %12.3f %12.3f\n",
        row.routes, row.eager_wall.median, row.eager_wall.p95,
        row.graph_wall.median, row.graph_wall.p95,
        row.kernel_device.median, row.kernel_device.p95,
        row.eager_wall.median - row.kernel_device.median,
        row.graph_wall.median - row.kernel_device.median,
        row.graph_device_total.median, row.graph_device_total.p95);
  }
  std::printf(
      "  correctness eager_graph_max_relative=%.9e "
      "eager_graph_max_abs=%.9e graph_cpu_max_relative=%.9e "
      "bound=5.000000000e-05 graph_cpu_max_abs=%.9e reference_peak=%.9e\n",
      eager_graph_max_relative, eager_graph_max_abs, cpu_max_relative,
      cpu_max_abs, cpu_peak);
  if (!std::isfinite(eager_graph_max_relative) ||
      eager_graph_max_relative > 5.0e-5 ||
      !std::isfinite(cpu_max_relative) || cpu_max_relative > 5.0e-5) {
    throw std::runtime_error("int4 graph replay correctness check failed");
  }

  sycl::free(host_arena, q);
  sycl::free(device_output, q);
  sycl::free(device_scratch, q);
  sycl::free(device_weights, q);
  sycl::free(device_ids, q);
  sycl::free(device_hidden, q);
  sycl::free(device_bank, q);
}

// ------------------------------------------------------------ persistent ---

// One kernel, launched once, spinning on a host-visible doorbell. The host
// rings it and polls for completion -- no per-layer submit, no CPU in the
// signalling path. Spin is ALWAYS bounded so the device cannot wedge.
void bench_persistent(sycl::queue &q, int iters) {
  std::printf("\n[persistent] long-lived kernel + host doorbell\n");
  auto *ctl = sycl::malloc_host<std::uint32_t>(4, q);   // [0]=doorbell [1]=done [2]=exit
  auto *h_act = static_cast<sycl::half *>(sycl::malloc_host(kActBytes, q));
  auto *h_out = sycl::malloc_host<float>(kOutBytes / sizeof(float), q);
  ctl[0] = 0; ctl[1] = 0; ctl[2] = 0;
  std::memset(h_act, 0x3C, kActBytes);

  // Escape hatch: the kernel exits after this many spin iterations no matter
  // what, so a missed doorbell degrades to a slow test, never a hung GPU.
  constexpr std::uint64_t kMaxSpins = 40'000'000'000ull;

  sycl::event worker = q.single_task([=]() {
    using Ref = sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                 sycl::memory_scope::system,
                                 sycl::access::address_space::global_space>;
    Ref bell(ctl[0]), done(ctl[1]), quit(ctl[2]);
    std::uint32_t seen = 0;
    for (std::uint64_t spin = 0; spin < kMaxSpins; ++spin) {
      if (quit.load() != 0u) break;
      const std::uint32_t cur = bell.load();
      if (cur != seen) {
        seen = cur;
        float acc = 0.0f;
        for (int i = 0; i < 2048; ++i) acc += static_cast<float>(h_act[i]);
        h_out[0] = acc;
        done.store(cur);
      }
    }
  });

  using Ref = sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                               sycl::memory_scope::system,
                               sycl::access::address_space::global_space>;
  Ref bell(ctl[0]), done(ctl[1]), quit(ctl[2]);

  // Warm up and confirm the doorbell is actually observed before timing.
  bool alive = false;
  for (int i = 0; i < 100; ++i) {
    const std::uint32_t want = bell.load() + 1;
    bell.store(want);
    auto t0 = Clock::now();
    while (done.load() != want && us_since(t0) < 200000.0) {}
    if (done.load() == want) alive = true; else break;
  }
  if (!alive) {
    std::printf("  doorbell never acknowledged -- system-scope atomics on host USM\n"
                "  are not working on this stack; persistent-kernel path is not viable as written\n");
    quit.store(1);
    worker.wait();
    sycl::free(ctl, q); sycl::free(h_act, q); sycl::free(h_out, q);
    return;
  }

  std::vector<double> rt;
  rt.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    const std::uint32_t want = bell.load() + 1;
    auto t0 = Clock::now();
    bell.store(want);
    while (done.load() != want && us_since(t0) < 200000.0) {}
    rt.push_back(us_since(t0));
  }
  Stats s = summarize(rt);
  report("doorbell round trip", s);
  std::printf("    -> 16 layers/token: %6.2f us    48 layers/token: %6.2f us\n",
              s.median * 16, s.median * 48);

  quit.store(1);
  worker.wait();
  sycl::free(ctl, q); sycl::free(h_act, q); sycl::free(h_out, q);
}

// ---------------------------------------------------------------- queues ---

void bench_queues(sycl::device dev, int iters, int nq) {
  std::printf("\n[queues] %d concurrent dispatches on %d in-order queues\n", nq, nq);
  std::vector<sycl::queue> qs;
  std::vector<void *> h_act, d_act, d_out, h_out;
  for (int i = 0; i < nq; ++i) {
    qs.emplace_back(dev, sycl::property::queue::in_order{});
    h_act.push_back(sycl::malloc_host(kActBytes, qs[i]));
    d_act.push_back(sycl::malloc_device(kActBytes, qs[i]));
    d_out.push_back(sycl::malloc_device(kOutBytes, qs[i]));
    h_out.push_back(sycl::malloc_host(kOutBytes, qs[i]));
    std::memset(h_act[i], 0x3C, kActBytes);
  }
  auto wave = [&] {
    for (int i = 0; i < nq; ++i) {
      auto *o = static_cast<float *>(d_out[i]);
      qs[i].memcpy(d_act[i], h_act[i], kActBytes);
      qs[i].parallel_for(sycl::range<1>(2048), [=](sycl::id<1> k) { o[k] = static_cast<float>(k[0]); });
      qs[i].memcpy(h_out[i], d_out[i], kOutBytes);
    }
    for (int i = 0; i < nq; ++i) qs[i].wait();
  };
  for (int i = 0; i < 50; ++i) wave();

  std::vector<double> rt;
  rt.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    auto t0 = Clock::now();
    wave();
    rt.push_back(us_since(t0));
  }
  Stats s = summarize(rt);
  report("wave of concurrent dispatches", s);
  std::printf("    -> per dispatch: %6.2f us  (serial would be %d x single)\n",
              s.median / nq, nq);
  for (int i = 0; i < nq; ++i) {
    sycl::free(h_act[i], qs[i]); sycl::free(d_act[i], qs[i]);
    sycl::free(d_out[i], qs[i]); sycl::free(h_out[i], qs[i]);
  }
}

} // namespace

int main(int argc, char **argv) {
  std::string mode = argc > 1 ? argv[1] : "all";
  const int iters = argc > 2 ? std::stoi(argv[2]) : 2000;

  sycl::device dev;
  try {
    dev = sycl::device(sycl::gpu_selector_v);
  } catch (const sycl::exception &e) {
    std::fprintf(stderr, "no SYCL GPU: %s\n", e.what());
    return 1;
  }
  sycl::queue q(dev, sycl::property::queue::in_order{});
  std::printf("device : %s\n", dev.get_info<sycl::info::device::name>().c_str());
  std::printf("iters  : %d\n", iters);

  if (mode == "all" || mode == "launch") bench_launch(q, iters);
  if (mode == "all" || mode == "copy") bench_copy(q, std::min(iters, 500));
  if (mode == "all" || mode == "dispatch") bench_dispatch(q, iters);
  if (mode == "environment") bench_environment(q, iters);
  if (mode == "real-kernel") bench_real_kernel(dev, iters);
  if (mode == "int4-graph") bench_int4_graph(dev, iters);
  if (mode == "all" || mode == "persistent") bench_persistent(q, iters);
  if (mode == "all" || mode == "queues") {
    for (int nq : {2, 4, 8}) bench_queues(dev, std::min(iters, 500), nq);
  }
  std::printf("\ndone\n");
  return 0;
}
