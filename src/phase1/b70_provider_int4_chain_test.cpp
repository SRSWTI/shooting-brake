#include "b70_provider.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

using shooting_brake::phase1::B70Provider;
using shooting_brake::phase1::DispatchResult;
using shooting_brake::phase1::ProviderConfig;
using shooting_brake::phase1::ProviderStatus;

constexpr std::string_view kBankPath = "src/phase1/expert_bank_int4.bin";
constexpr std::uint64_t kGeneration = 71;
constexpr std::size_t kTopK = 8;
constexpr std::size_t kHidden = 3072;
constexpr std::size_t kIntermediate = 1024;
constexpr std::size_t kGroupSize = 128;
constexpr std::size_t kLayer = 0;
constexpr std::size_t kTimingWarmups = 10;
constexpr std::size_t kTimingSamples = 50;
constexpr double kRelativeBound = 5.0e-5;
constexpr double kPartitionRelativeBound = 5.0e-6;

#pragma pack(push, 1)
struct Int4BankHeaderPrefix {
  char magic[8];
  std::uint32_t version;
  std::uint32_t data_offset;
  std::uint32_t num_layers;
  std::uint32_t source_num_layers;
  std::uint32_t experts_per_layer;
  std::uint32_t source_experts_per_layer;
  std::uint32_t resident_set_shared_across_layers;
  std::uint32_t hidden;
  std::uint32_t moe_intermediate;
  std::uint32_t group_size;
  std::uint32_t bits;
  std::uint32_t zero_point;
  std::uint32_t reserved0;
  std::uint32_t reserved1;
  std::uint32_t gate_q_offset;
  std::uint32_t gate_q_size;
  std::uint32_t gate_s_offset;
  std::uint32_t gate_s_size;
  std::uint32_t up_q_offset;
  std::uint32_t up_q_size;
  std::uint32_t up_s_offset;
  std::uint32_t up_s_size;
  std::uint32_t down_q_offset;
  std::uint32_t down_q_size;
  std::uint32_t down_s_offset;
  std::uint32_t down_s_size;
  std::uint64_t expert_stride_bytes;
  std::uint64_t layer_stride_bytes;
};
#pragma pack(pop)

static_assert(sizeof(Int4BankHeaderPrefix) == 128);

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

void require(const bool condition, const std::string& message) {
  if (!condition) {
    fail(message);
  }
}

const char* status_name(const ProviderStatus status) {
  switch (status) {
    case ProviderStatus::ok:
      return "ok";
    case ProviderStatus::busy:
      return "busy";
    case ProviderStatus::not_loaded:
      return "not_loaded";
    case ProviderStatus::invalid_argument:
      return "invalid_argument";
    case ProviderStatus::generation_mismatch:
      return "generation_mismatch";
    case ProviderStatus::sequence_mismatch:
      return "sequence_mismatch";
    case ProviderStatus::device_error:
      return "device_error";
    case ProviderStatus::shutdown:
      return "shutdown";
  }
  return "unknown";
}

void require_status(const ProviderStatus actual, const ProviderStatus expected,
                    const std::string_view operation) {
  if (actual != expected) {
    std::ostringstream message;
    message << operation << ": expected " << status_name(expected) << ", got "
            << status_name(actual);
    fail(message.str());
  }
}

void pread_exact(const int fd, void* destination, const std::size_t bytes,
                 const std::uint64_t file_offset) {
  auto* output = static_cast<std::uint8_t*>(destination);
  std::size_t done = 0;
  while (done < bytes) {
    const ssize_t result = ::pread(
        fd, output + done, bytes - done,
        static_cast<off_t>(file_offset + static_cast<std::uint64_t>(done)));
    if (result < 0 && errno == EINTR) {
      continue;
    }
    if (result <= 0) {
      std::ostringstream message;
      message << "pread failed/short at offset " << file_offset + done
              << " after " << done << "/" << bytes << " bytes";
      fail(message.str());
    }
    done += static_cast<std::size_t>(result);
  }
}

struct ParsedBank {
  int fd = -1;
  Int4BankHeaderPrefix header{};
  std::vector<std::int32_t> source_expert_ids;

  ParsedBank() = default;
  ParsedBank(const ParsedBank&) = delete;
  ParsedBank& operator=(const ParsedBank&) = delete;
  ParsedBank(ParsedBank&& other) noexcept
      : fd(other.fd),
        header(other.header),
        source_expert_ids(std::move(other.source_expert_ids)) {
    other.fd = -1;
  }
  ~ParsedBank() {
    if (fd >= 0) {
      ::close(fd);
    }
  }
};

ParsedBank parse_bank() {
  ParsedBank bank;
  bank.fd = ::open(std::string(kBankPath).c_str(), O_RDONLY | O_CLOEXEC);
  if (bank.fd < 0) {
    fail("cannot open real int4 bank at " + std::string(kBankPath));
  }
  pread_exact(bank.fd, &bank.header, sizeof(bank.header), 0);
  const auto& h = bank.header;
  require(std::memcmp(h.magic, "SBINT401", 8) == 0,
          "real bank magic is not SBINT401");
  require(h.version == 2 && h.bits == 4 && h.zero_point == 8 &&
              h.group_size == kGroupSize,
          "real bank quantization header is not v2 int4 zp=8 group=128");
  require(h.hidden == kHidden && h.moe_intermediate == kIntermediate,
          "real bank geometry is not hidden=3072 intermediate=1024");
  require(h.experts_per_layer == 126 && h.source_experts_per_layer == 180 &&
              h.num_layers == 48,
          "real bank layer/expert geometry is not 48 x 126 of 180");
  require(h.expert_stride_bytes == 4866048,
          "real bank expert stride is not 4,866,048 bytes");
  bank.source_expert_ids.resize(h.experts_per_layer);
  pread_exact(bank.fd, bank.source_expert_ids.data(),
              bank.source_expert_ids.size() * sizeof(std::int32_t),
              sizeof(Int4BankHeaderPrefix));
  require(bank.source_expert_ids.front() == 54 &&
              bank.source_expert_ids.back() == 179,
          "real bank source expert map is not 54..179");
  for (std::size_t index = 0; index < bank.source_expert_ids.size(); ++index) {
    require(bank.source_expert_ids[index] ==
                static_cast<std::int32_t>(54 + index),
            "real bank source expert map is not contiguous 54..179");
  }
  return bank;
}

struct RssPeaks {
  std::uint64_t anon_kib = 0;
  std::uint64_t file_kib = 0;
  std::uint64_t total_kib = 0;
  std::uint64_t samples = 0;
};

class RssSampler {
 public:
  RssSampler() : thread_([this] { run(); }) {}
  RssSampler(const RssSampler&) = delete;
  RssSampler& operator=(const RssSampler&) = delete;
  ~RssSampler() { stop(); }

  RssPeaks stop() {
    bool expected = false;
    if (stopped_.compare_exchange_strong(expected, true)) {
      thread_.join();
    }
    return {peak_anon_kib_.load(), peak_file_kib_.load(),
            peak_total_kib_.load(), samples_.load()};
  }

 private:
  static void update_peak(std::atomic<std::uint64_t>& peak,
                          const std::uint64_t value) {
    std::uint64_t observed = peak.load(std::memory_order_relaxed);
    while (observed < value &&
           !peak.compare_exchange_weak(observed, value,
                                       std::memory_order_relaxed)) {
    }
  }

  void sample() {
    std::FILE* input = std::fopen("/proc/self/status", "r");
    if (input == nullptr) {
      return;
    }
    std::uint64_t anon = 0;
    std::uint64_t file = 0;
    std::uint64_t rss = 0;
    char line[256];
    while (std::fgets(line, sizeof(line), input) != nullptr) {
      unsigned long long value = 0;
      if (std::sscanf(line, "RssAnon: %llu kB", &value) == 1) {
        anon = value;
      } else if (std::sscanf(line, "RssFile: %llu kB", &value) == 1) {
        file = value;
      } else if (std::sscanf(line, "VmRSS: %llu kB", &value) == 1) {
        rss = value;
      }
    }
    std::fclose(input);
    update_peak(peak_anon_kib_, anon);
    update_peak(peak_file_kib_, file);
    update_peak(peak_total_kib_, rss);
    samples_.fetch_add(1, std::memory_order_relaxed);
  }

  void run() {
    while (!stopped_.load(std::memory_order_relaxed)) {
      sample();
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    sample();
  }

  std::atomic<bool> stopped_{false};
  std::atomic<std::uint64_t> peak_anon_kib_{0};
  std::atomic<std::uint64_t> peak_file_kib_{0};
  std::atomic<std::uint64_t> peak_total_kib_{0};
  std::atomic<std::uint64_t> samples_{0};
  std::thread thread_;
};

void require_diagnostic_contains(const std::string& diagnostic,
                                 const std::initializer_list<std::string_view> values,
                                 const std::string_view case_name) {
  for (const std::string_view value : values) {
    if (diagnostic.find(value) == std::string::npos) {
      fail(std::string(case_name) + " rejection diagnostic lacks '" +
           std::string(value) + "': " + diagnostic);
    }
  }
}

void reject_bad_map(const std::vector<std::int32_t>& resident_experts,
                    const std::initializer_list<std::string_view> diagnostic_values,
                    const std::string_view case_name) {
  B70Provider provider;
  ProviderConfig config;
  config.max_batch = 1;
  config.top_k = kTopK;
  config.generation = kGeneration;
  config.resident_experts = resident_experts;
  require_status(provider.load(std::string(kBankPath), config),
                 ProviderStatus::invalid_argument, case_name);
  const std::string diagnostic = provider.health().last_error;
  require_diagnostic_contains(diagnostic, diagnostic_values, case_name);
  std::cout << "map_rejection case=" << case_name << " diagnostic=\""
            << diagnostic << "\"\n";
}

float dequant(const std::int32_t* qweight, const sycl::half* scales,
              const std::size_t reduction_index,
              const std::size_t output_index, const std::size_t output_size) {
  const std::uint32_t word = static_cast<std::uint32_t>(
      qweight[(reduction_index / 8) * output_size + output_index]);
  const int nibble = static_cast<int>(
      (word >> (4 * (reduction_index % 8))) & 0x0fU);
  return static_cast<float>(nibble - 8) *
         static_cast<float>(scales[(reduction_index / kGroupSize) * output_size +
                                  output_index]);
}

std::vector<float> cpu_reference(
    const ParsedBank& bank, const std::vector<sycl::half>& hidden,
    const std::array<std::int32_t, kTopK>& compact_ids,
    const std::array<float, kTopK>& route_weights) {
  const auto& h = bank.header;
  std::vector<float> reference(kHidden, 0.0F);
  std::vector<float> activated(kIntermediate);
  std::vector<std::uint8_t> record(
      static_cast<std::size_t>(h.expert_stride_bytes));

  // Worst-case resident bytes owned by this CPU oracle are 4,901,432 bytes
  // (4.675 MiB): one 4,866,048-byte expert record, hidden, actual/reference
  // outputs, activation, routes, and the 126-ID header map. The loop below
  // reuses that one record for all eight routes; it can never retain eight
  // experts (38.9 MiB), much less materialise the 27.41 GiB bank.
  for (std::size_t route = 0; route < kTopK; ++route) {
    const std::size_t compact_expert =
        static_cast<std::size_t>(compact_ids[route]);
    const std::uint64_t expert_offset =
        static_cast<std::uint64_t>(h.data_offset) +
        static_cast<std::uint64_t>(kLayer) * h.layer_stride_bytes +
        compact_expert * h.expert_stride_bytes;
    pread_exact(bank.fd, record.data(), record.size(), expert_offset);

    const auto* gate_q = reinterpret_cast<const std::int32_t*>(
        record.data() + h.gate_q_offset);
    const auto* gate_s = reinterpret_cast<const sycl::half*>(
        record.data() + h.gate_s_offset);
    const auto* up_q = reinterpret_cast<const std::int32_t*>(
        record.data() + h.up_q_offset);
    const auto* up_s = reinterpret_cast<const sycl::half*>(
        record.data() + h.up_s_offset);
    const auto* down_q = reinterpret_cast<const std::int32_t*>(
        record.data() + h.down_q_offset);
    const auto* down_s = reinterpret_cast<const sycl::half*>(
        record.data() + h.down_s_offset);

    for (std::size_t output = 0; output < kIntermediate; ++output) {
      float gate = 0.0F;
      float up = 0.0F;
      for (std::size_t reduction = 0; reduction < kHidden; ++reduction) {
        const float input = static_cast<float>(hidden[reduction]);
        gate = std::fma(dequant(gate_q, gate_s, reduction, output,
                                kIntermediate),
                        input, gate);
        up = std::fma(
            dequant(up_q, up_s, reduction, output, kIntermediate), input, up);
      }
      activated[output] = (gate / (1.0F + std::exp(-gate))) * up;
    }

    for (std::size_t output = 0; output < kHidden; ++output) {
      float down = 0.0F;
      for (std::size_t reduction = 0; reduction < kIntermediate; ++reduction) {
        down = std::fma(
            dequant(down_q, down_s, reduction, output, kHidden),
            activated[reduction], down);
      }
      reference[output] =
          std::fma(route_weights[route], down, reference[output]);
    }
  }
  return reference;
}

struct ErrorSummary {
  double max_abs = 0.0;
  double reference_peak = 0.0;
  double max_relative = 0.0;
  std::size_t nonfinite = 0;
};

ErrorSummary compare(const std::vector<float>& actual,
                     const std::vector<float>& reference) {
  ErrorSummary result;
  for (const float value : reference) {
    result.reference_peak =
        std::max(result.reference_peak, std::abs(static_cast<double>(value)));
  }
  for (std::size_t index = 0; index < reference.size(); ++index) {
    if (!std::isfinite(actual[index])) {
      ++result.nonfinite;
      continue;
    }
    result.max_abs = std::max(
        result.max_abs,
        std::abs(static_cast<double>(actual[index]) - reference[index]));
  }
  // Peak normalization is well-defined at output elements crossing zero and
  // is the same correctness metric used by the kernel's independent smoke
  // test. The 5e-5 bound covers fp32 reduction-tree/atomic ordering while
  // remaining >19x tighter than one fp16 input ulp; wrong expert/plane/zero
  // point errors are orders of magnitude larger.
  result.max_relative =
      result.max_abs / std::max(result.reference_peak, 1.0e-30);
  return result;
}
struct PartitionSummary {
  double max_abs = 0.0;
  double max_relative = 0.0;
  std::size_t all_skip_nonzero = 0;
};

PartitionSummary verify_route_partitions(
    B70Provider& provider, const std::vector<sycl::half>& hidden,
    const std::array<std::int32_t, kTopK>& full_ids,
    const std::array<float, kTopK>& full_weights,
    const std::vector<float>& full_output, std::uint64_t& sequence) {
  PartitionSummary summary;
  double full_peak = 0.0;
  for (const float value : full_output) {
    full_peak = std::max(full_peak, std::abs(static_cast<double>(value)));
  }
  constexpr std::array<std::size_t, kTopK> kAlternatingOrder{
      0, 2, 4, 6, 1, 3, 5, 7};
  for (std::size_t left_count = 0; left_count <= kTopK; ++left_count) {
    std::array<bool, kTopK> assigned_left{};
    for (std::size_t index = 0; index < left_count; ++index) {
      assigned_left[kAlternatingOrder[index]] = true;
    }
    std::array<std::int32_t, kTopK> left_ids{};
    std::array<std::int32_t, kTopK> right_ids{};
    std::array<float, kTopK> left_weights{};
    std::array<float, kTopK> right_weights{};
    left_ids.fill(-1);
    right_ids.fill(-1);
    left_weights.fill(0.0F);
    right_weights.fill(0.0F);
    for (std::size_t route = 0; route < kTopK; ++route) {
      auto& ids = assigned_left[route] ? left_ids : right_ids;
      auto& weights = assigned_left[route] ? left_weights : right_weights;
      ids[route] = full_ids[route];
      weights[route] = full_weights[route];
    }

    std::vector<float> left_output(kHidden);
    std::vector<float> right_output(kHidden);
    require_status(
        provider.issue(kGeneration, sequence, kLayer, hidden.data(),
                       left_ids.data(), left_weights.data(), 1),
        ProviderStatus::ok, "partition-left issue");
    DispatchResult left_result;
    require_status(provider.take(kGeneration, sequence, left_output.data(),
                                 left_output.size(), &left_result),
                   ProviderStatus::ok, "partition-left take");
    ++sequence;
    require_status(
        provider.issue(kGeneration, sequence, kLayer, hidden.data(),
                       right_ids.data(), right_weights.data(), 1),
        ProviderStatus::ok, "partition-right issue");
    DispatchResult right_result;
    require_status(provider.take(kGeneration, sequence, right_output.data(),
                                 right_output.size(), &right_result),
                   ProviderStatus::ok, "partition-right take");
    ++sequence;

    if (left_count == 0) {
      for (const float value : left_output) {
        require(std::isfinite(value),
                "all--1 route dispatch produced a non-finite output");
        summary.all_skip_nonzero += value != 0.0F ? 1 : 0;
      }
    }
    for (std::size_t output = 0; output < kHidden; ++output) {
      const float combined = left_output[output] + right_output[output];
      require(std::isfinite(combined),
              "partitioned route dispatch produced a non-finite output");
      summary.max_abs =
          std::max(summary.max_abs,
                   std::abs(static_cast<double>(combined) -
                            static_cast<double>(full_output[output])));
    }
  }
  summary.max_relative =
      summary.max_abs / std::max(full_peak, 1.0e-30);
  return summary;
}


struct TimingSummary {
  double wall_median_us = 0.0;
  double wall_p95_us = 0.0;
  double kernel_median_us = 0.0;
  double kernel_p95_us = 0.0;
};

double percentile(std::vector<double> values, const double fraction) {
  std::sort(values.begin(), values.end());
  const std::size_t index = static_cast<std::size_t>(
      std::ceil(fraction * static_cast<double>(values.size())) - 1.0);
  return values[std::min(index, values.size() - 1)];
}

TimingSummary measure_dispatch(
    B70Provider& provider, const std::vector<sycl::half>& hidden,
    const std::array<std::int32_t, kTopK>& compact_ids,
    const std::array<float, kTopK>& weights, std::vector<float>& output,
    std::uint64_t& sequence) {
  std::vector<double> wall;
  std::vector<double> kernel;
  wall.reserve(kTimingSamples);
  kernel.reserve(kTimingSamples);
  for (std::size_t iteration = 0;
       iteration < kTimingWarmups + kTimingSamples; ++iteration) {
    const auto start = std::chrono::steady_clock::now();
    require_status(provider.issue(kGeneration, sequence, kLayer, hidden.data(),
                                  compact_ids.data(), weights.data(), 1),
                   ProviderStatus::ok, "timed issue");
    DispatchResult result;
    require_status(provider.take(kGeneration, sequence, output.data(),
                                 output.size(), &result),
                   ProviderStatus::ok, "timed take");
    const auto stop = std::chrono::steady_clock::now();
    require(result.kernel == "split", "int4 dispatch did not report split");
    require(result.kernel_us > 0.0,
            "device kernel timing is zero; run with profiling enabled");
    if (iteration >= kTimingWarmups) {
      wall.push_back(std::chrono::duration<double, std::micro>(stop - start)
                         .count());
      kernel.push_back(result.kernel_us);
    }
    ++sequence;
  }
  return {percentile(wall, 0.50), percentile(wall, 0.95),
          percentile(kernel, 0.50), percentile(kernel, 0.95)};
}
std::array<TimingSummary, kTopK + 1> measure_route_sweep(
    B70Provider& provider, const std::vector<sycl::half>& hidden,
    std::vector<float>& output, std::uint64_t& sequence) {
  std::array<TimingSummary, kTopK + 1> sweep{};
  for (std::size_t active_routes = 0; active_routes <= kTopK;
       ++active_routes) {
    std::array<std::int32_t, kTopK> ids{};
    std::array<float, kTopK> weights{};
    ids.fill(-1);
    weights.fill(0.0F);
    for (std::size_t route = 0; route < active_routes; ++route) {
      ids[route] = static_cast<std::int32_t>(route);
      weights[route] = 1.0F / static_cast<float>(active_routes);
    }
    sweep[active_routes] =
        measure_dispatch(provider, hidden, ids, weights, output, sequence);
  }
  return sweep;
}


void validate_capability(const B70Provider& provider) {
  const auto capability = provider.capability();
  require(capability.backend == "quixicore-xpu-int4",
          "capability backend is not quixicore-xpu-int4");
  require(capability.supported_hidden_sizes ==
              std::vector<std::uint32_t>{kHidden},
          "capability hidden sizes are not exactly [3072]");
  require(capability.supported_intermediate_sizes ==
              std::vector<std::uint32_t>{kIntermediate},
          "capability intermediate sizes are not exactly [1024]");
  require(capability.kernel_families ==
              std::vector<std::string>{"int4_moe_split"},
          "capability kernel families are not exactly [int4_moe_split]");
  require(capability.num_layers == 48 && capability.experts_per_layer == 180 &&
              capability.num_resident_experts == 48 * 126,
          "capability layer/source/resident expert geometry is wrong");

  require(capability.source_expert_ids.size() == 126 &&
              capability.source_expert_ids.front() == 54 &&
              capability.source_expert_ids.back() == 179,
          "capability source expert ID map is not ordered 54..179");
  for (std::size_t index = 0;
       index < capability.source_expert_ids.size(); ++index) {
    require(capability.source_expert_ids[index] ==
                static_cast<std::int32_t>(54 + index),
            "capability source expert ID map differs from the bank header");
  }
}
}  // namespace

int main() {
  try {
    RssSampler rss_sampler;
    ParsedBank bank = parse_bank();

    std::vector<std::int32_t> wrong_full(bank.source_expert_ids.size());
    std::iota(wrong_full.begin(), wrong_full.end(), 0);
    reject_bad_map(wrong_full, {"126", "0", "125", "54", "179"},
                   "wrong-full-map");

    std::vector<std::int32_t> subset = bank.source_expert_ids;
    subset.pop_back();
    reject_bad_map(subset, {"125", "54", "178", "126", "179"},
                   "correct-map-subset");

    ProviderConfig config;
    config.max_batch = 1;
    config.top_k = kTopK;
    config.generation = kGeneration;
    config.resident_experts = bank.source_expert_ids;

    B70Provider provider;
    require_status(provider.load(std::string(kBankPath), config),
                   ProviderStatus::ok, "load exact int4 map");
    validate_capability(provider);

    std::vector<sycl::half> hidden(kHidden);
    for (std::size_t index = 0; index < hidden.size(); ++index) {
      const int centered = static_cast<int>((index * 17 + 5) % 29) - 14;
      hidden[index] = static_cast<sycl::half>(
          0.015625F * static_cast<float>(centered) / 14.0F);
    }
    const std::array<std::int32_t, kTopK> compact_ids{0, 1, 2, 3, 4, 5, 6, 7};
    const std::array<float, kTopK> route_weights{0.19F, 0.17F, 0.15F, 0.14F,
                                                0.12F, 0.10F, 0.08F, 0.05F};
    std::vector<float> actual(kHidden,
                              std::numeric_limits<float>::quiet_NaN());

    auto out_of_range_ids = compact_ids;
    out_of_range_ids[0] = 126;
    require_status(
        provider.issue(kGeneration, 1, kLayer, hidden.data(),
                       out_of_range_ids.data(), route_weights.data(), 1),
        ProviderStatus::invalid_argument, "out-of-range compact route");
    require(!provider.health().pending,
            "rejected compact route left the provider pending");
    require(provider.health().last_error.find(
                "outside the configured compact range") != std::string::npos,
            "compact-route rejection diagnostic does not name its range");

    std::uint64_t sequence = 1;
    require_status(provider.issue(kGeneration, sequence, kLayer, hidden.data(),
                                  compact_ids.data(), route_weights.data(), 1),
                   ProviderStatus::ok, "correctness issue");
    DispatchResult correctness_result;
    require_status(provider.take(kGeneration, sequence, actual.data(),
                                 actual.size(), &correctness_result),
                   ProviderStatus::ok, "correctness take");
    ++sequence;

    const std::vector<float> reference =
        cpu_reference(bank, hidden, compact_ids, route_weights);
    const ErrorSummary errors = compare(actual, reference);
    require(errors.nonfinite == 0, "provider output contains non-finite values");
    require(errors.max_relative <= kRelativeBound,
            "provider output exceeds peak-relative correctness bound");
    const PartitionSummary partitions = verify_route_partitions(
        provider, hidden, compact_ids, route_weights, actual, sequence);
    require(partitions.all_skip_nonzero == 0,
            "all--1 route dispatch did not produce exact zero output");
    require(partitions.max_relative <= kPartitionRelativeBound,
            "partitioned route outputs do not reconstruct full dispatch");


    const auto route_sweep =
        measure_route_sweep(provider, hidden, actual, sequence);
    provider.shutdown();
    const RssPeaks rss = rss_sampler.stop();

    std::cout << std::setprecision(9) << std::scientific
              << "correctness max_relative=" << errors.max_relative
              << " bound=" << kRelativeBound << " max_abs=" << errors.max_abs
              << " reference_peak=" << errors.reference_peak
              << " nonfinite=" << errors.nonfinite << '\n';
    std::cout << "partition_sum max_relative=" << partitions.max_relative
              << " bound=" << kPartitionRelativeBound
              << " max_abs=" << partitions.max_abs
              << " all_skip_nonzero=" << partitions.all_skip_nonzero << '\n';
    std::cout << std::fixed << std::setprecision(3)
              << "route_sweep M=1 warmups=" << kTimingWarmups
              << " samples=" << kTimingSamples << '\n';
    std::cout << "active_routes issue_take_wall_median_us "
                 "issue_take_wall_p95_us device_kernel_median_us "
                 "device_kernel_p95_us\n";
    for (std::size_t active_routes = 0; active_routes <= kTopK;
         ++active_routes) {
      const TimingSummary& timing = route_sweep[active_routes];
      std::cout << active_routes << ' ' << timing.wall_median_us << ' '
                << timing.wall_p95_us << ' ' << timing.kernel_median_us << ' '
                << timing.kernel_p95_us << '\n';
    }
    std::cout << "rss sampled_peak_anon_kib=" << rss.anon_kib
              << " sampled_peak_file_kib=" << rss.file_kib
              << " sampled_peak_total_kib=" << rss.total_kib
              << " samples=" << rss.samples << '\n';
    std::cout << "cpu_reference_worst_case_resident_bytes=4901432"
              << " cpu_reference_bank_bytes_pread="
              << kTopK * bank.header.expert_stride_bytes << '\n';
    std::cout << "int4_provider_chain PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "int4_provider_chain FAIL: " << error.what() << '\n';
    return 1;
  }
}
