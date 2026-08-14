#include "b70_provider.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using shooting_brake::phase1::B70Provider;
using shooting_brake::phase1::DispatchResult;
using shooting_brake::phase1::ProviderConfig;
using shooting_brake::phase1::ProviderStatus;
#ifdef SHOOTING_BRAKE_ENABLE_TEST_FAULTS
using shooting_brake::phase1::ProviderTestFault;
#endif

constexpr std::size_t kLayers = 32;
constexpr std::size_t kExpertsPerLayer = 256;
constexpr std::size_t kHiddenSize = 2048;
constexpr std::size_t kIntermediateSize = 512;
constexpr std::size_t kTopK = 8;
constexpr std::size_t kMaxBatch = 128;
constexpr std::uint64_t kGeneration = 17;
constexpr float kAtol = 1.0e-6F;
constexpr float kRtol = 1.0e-2F;
constexpr std::string_view kBankPath = "src/phase1/expert_bank.bin";
constexpr std::string_view kGoldenPath = "src/phase1/golden_reference.bin";

class TestFailure final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

[[noreturn]] void fail(const std::string& message) {
  throw TestFailure(message);
}

void require(bool condition, const std::string& message) {
  if (!condition) {
    fail(message);
  }
}

const char* status_name(ProviderStatus status) {
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

void require_status(ProviderStatus actual, ProviderStatus expected,
                    std::string_view operation) {
  if (actual == expected) {
    return;
  }
  std::ostringstream message;
  message << operation << ": expected status " << status_name(expected)
          << ", got " << status_name(actual);
  fail(message.str());
}


struct GoldenReference {
  std::vector<sycl::half> hidden;
  std::vector<float> output;
};

GoldenReference load_golden_reference(std::string_view path) {
  std::ifstream input(std::string(path), std::ios::binary | std::ios::ate);
  if (!input) {
    fail("golden load: cannot open " + std::string(path));
  }

  const std::streampos end = input.tellg();
  if (end < 0) {
    fail("golden load: cannot determine file size for " + std::string(path));
  }
  const auto file_size = static_cast<std::size_t>(end);
  constexpr std::size_t expected_size =
      3 * sizeof(std::uint32_t) + kHiddenSize * sizeof(sycl::half) +
      sizeof(std::int32_t) + sizeof(float) + kHiddenSize * sizeof(float);
  if (file_size != expected_size) {
    std::ostringstream message;
    message << "golden load: size mismatch for " << path << ": got "
            << file_size << " bytes, expected " << expected_size;
    fail(message.str());
  }

  std::vector<std::byte> bytes(file_size);
  input.seekg(0, std::ios::beg);
  input.read(reinterpret_cast<char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!input || static_cast<std::size_t>(input.gcount()) != bytes.size()) {
    fail("golden load: short read from " + std::string(path));
  }

  std::size_t offset = 0;
  auto take = [&](void* destination, std::size_t size) {
    if (size > bytes.size() - offset) {
      fail("golden load: truncated payload in " + std::string(path));
    }
    std::memcpy(destination, bytes.data() + offset, size);
    offset += size;
  };

  std::uint32_t hidden_size = 0;
  std::uint32_t intermediate_size = 0;
  std::uint32_t top_k = 0;
  take(&hidden_size, sizeof(hidden_size));
  take(&intermediate_size, sizeof(intermediate_size));
  take(&top_k, sizeof(top_k));
  require(hidden_size == kHiddenSize,
          "golden load: hidden size is not 2048");
  require(intermediate_size == kIntermediateSize,
          "golden load: intermediate size is not 512");
  require(top_k == 1, "golden load: reference top_k is not 1");

  GoldenReference golden;
  golden.hidden.resize(kHiddenSize);
  golden.output.resize(kHiddenSize);
  take(golden.hidden.data(), golden.hidden.size() * sizeof(sycl::half));

  std::int32_t expert_id = -1;
  float expert_weight = 0.0F;
  take(&expert_id, sizeof(expert_id));
  take(&expert_weight, sizeof(expert_weight));
  require(expert_id == 0, "golden load: reference expert ID is not 0");
  require(expert_weight == 1.0F,
          "golden load: reference expert weight is not 1");

  take(golden.output.data(), golden.output.size() * sizeof(float));
  require(offset == bytes.size(), "golden load: unconsumed trailing payload");
  for (std::size_t col = 0; col < golden.output.size(); ++col) {
    if (!std::isfinite(golden.output[col])) {
      std::ostringstream message;
      message << "golden load: non-finite expected output at column " << col;
      fail(message.str());
    }
  }
  return golden;
}

struct HostBatch {
  std::size_t M;
  std::vector<sycl::half> hidden;
  std::vector<std::int32_t> ids;
  std::vector<float> weights;
  std::vector<float> output;
};

HostBatch make_batch(const GoldenReference& golden, std::size_t M,
                     bool duplicate_top8) {
  HostBatch batch{M,
                  std::vector<sycl::half>(M * kHiddenSize),
                  std::vector<std::int32_t>(M * kTopK, -1),
                  std::vector<float>(M * kTopK, 0.0F),
                  std::vector<float>(M * kHiddenSize,
                                     std::numeric_limits<float>::quiet_NaN())};
  constexpr std::array<float, kTopK> duplicate_weights{
      0.03F, 0.07F, 0.11F, 0.13F, 0.17F, 0.19F, 0.23F, 0.07F};

  for (std::size_t row = 0; row < M; ++row) {
    std::copy(golden.hidden.begin(), golden.hidden.end(),
              batch.hidden.begin() + static_cast<std::ptrdiff_t>(row * kHiddenSize));
    if (duplicate_top8) {
      for (std::size_t route = 0; route < kTopK; ++route) {
        batch.ids[row * kTopK + route] = 0;
        batch.weights[row * kTopK + route] = duplicate_weights[route];
      }
    } else {
      batch.ids[row * kTopK] = 0;
      batch.weights[row * kTopK] = 1.0F;
    }
  }
  return batch;
}

struct ErrorSummary {
  float max_abs = 0.0F;
  float max_rel = 0.0F;
  std::size_t bad = 0;
  std::size_t nonfinite = 0;
  std::size_t first_bad_row = 0;
  std::size_t first_bad_col = 0;
  float first_actual = 0.0F;
  float first_expected = 0.0F;
  float first_tolerance = 0.0F;
};

ErrorSummary compare_to_golden(const HostBatch& batch,
                               const GoldenReference& golden) {
  ErrorSummary summary;
  bool have_first_bad = false;
  for (std::size_t row = 0; row < batch.M; ++row) {
    for (std::size_t col = 0; col < kHiddenSize; ++col) {
      const float actual = batch.output[row * kHiddenSize + col];
      const float expected = golden.output[col];
      const float tolerance = kAtol + kRtol * std::abs(expected);
      if (!std::isfinite(actual)) {
        ++summary.nonfinite;
        if (!have_first_bad) {
          have_first_bad = true;
          summary.first_bad_row = row;
          summary.first_bad_col = col;
          summary.first_actual = actual;
          summary.first_expected = expected;
          summary.first_tolerance = tolerance;
        }
        continue;
      }

      const float abs_error = std::abs(actual - expected);
      summary.max_abs = std::max(summary.max_abs, abs_error);
      if (expected != 0.0F) {
        summary.max_rel =
            std::max(summary.max_rel, abs_error / std::abs(expected));
      }
      if (abs_error > tolerance) {
        ++summary.bad;
        if (!have_first_bad) {
          have_first_bad = true;
          summary.first_bad_row = row;
          summary.first_bad_col = col;
          summary.first_actual = actual;
          summary.first_expected = expected;
          summary.first_tolerance = tolerance;
        }
      }
    }
  }
  return summary;
}

void verify_dispatch_result(const DispatchResult& result, std::uint64_t sequence,
                            std::size_t M, std::string_view expected_kernel) {
  require(result.generation == kGeneration,
          "dispatch result: generation does not match request");
  require(result.sequence == sequence,
          "dispatch result: sequence does not match request");
  require(result.M == M, "dispatch result: M does not match request");
  if (result.kernel != expected_kernel) {
    std::ostringstream message;
    message << "dispatch result: M=" << M << " expected kernel "
            << expected_kernel << ", got " << result.kernel;
    fail(message.str());
  }
  require(std::isfinite(result.kernel_us) && result.kernel_us >= 0.0,
          "dispatch result: kernel_us is not finite and nonnegative");
  require(std::isfinite(result.total_us) && result.total_us >= 0.0,
          "dispatch result: total_us is not finite and nonnegative");
}

void print_and_require_correct(std::string_view case_name,
                               const DispatchResult& result,
                               const ErrorSummary& errors) {
  std::cout << "M=" << std::setw(3) << result.M << " case=" << case_name
            << " kernel=" << result.kernel << " kernel_us=" << std::fixed
            << std::setprecision(3) << result.kernel_us
            << " total_us=" << result.total_us << " max_abs="
            << std::scientific << errors.max_abs << " max_rel="
            << errors.max_rel << " bad=" << std::dec << errors.bad
            << " nonfinite=" << errors.nonfinite << '\n';

  if (errors.bad != 0 || errors.nonfinite != 0) {
    std::ostringstream message;
    message << "correctness " << case_name << " M=" << result.M << ": "
            << errors.bad << " out-of-tolerance and " << errors.nonfinite
            << " non-finite elements; first failure row="
            << errors.first_bad_row << " col=" << errors.first_bad_col
            << " actual=" << errors.first_actual
            << " expected=" << errors.first_expected
            << " tolerance=" << errors.first_tolerance;
    fail(message.str());
  }
}

DispatchResult finish_dispatch(B70Provider& provider, HostBatch& batch,
                               std::uint64_t sequence,
                               const GoldenReference& golden,
                               std::string_view case_name,
                               std::string_view expected_kernel) {
  DispatchResult result{};
  require_status(provider.take(kGeneration, sequence, batch.output.data(),
                               batch.output.size(), &result),
                 ProviderStatus::ok, "take " + std::string(case_name));
  verify_dispatch_result(result, sequence, batch.M, expected_kernel);
  print_and_require_correct(case_name, result,
                            compare_to_golden(batch, golden));
  return result;
}

void issue_dispatch(B70Provider& provider, const HostBatch& batch,
                    std::uint64_t sequence, std::string_view case_name) {
  require_status(provider.issue(kGeneration, sequence, 0, batch.hidden.data(),
                                batch.ids.data(), batch.weights.data(), batch.M),
                 ProviderStatus::ok, "issue " + std::string(case_name));
}

void test_loaded_capability_and_health(B70Provider& provider,
                                       std::uint64_t* allocation_baseline) {
  const auto capability = provider.capability();
  require(capability.protocol_version == 1,
          "capability: protocol_version is not 1");
  require(capability.backend == "quixicore-xpu-nvfp4",
          "capability: backend is not quixicore-xpu-nvfp4");
  require(!capability.device_name.empty(), "capability: device_name is empty");
  require(capability.device_name.find("B70") != std::string::npos ||
              capability.device_name.find("Arc") != std::string::npos,
          "capability: selected device name contains neither B70 nor Arc");
  require(capability.device_memory_total_bytes > 0,
          "capability: total device memory is zero");
  require(capability.device_memory_available_bytes > 0,
          "capability: available device memory is zero");
  require(capability.device_memory_available_bytes <
              capability.device_memory_total_bytes,
          "capability: available memory does not account for resident buffers");
  require(capability.supported_hidden_sizes.size() == 1 &&
              capability.supported_hidden_sizes[0] == kHiddenSize,
          "capability: supported_hidden_sizes is not exactly [2048]");
  require(capability.supported_intermediate_sizes.size() == 1 &&
              capability.supported_intermediate_sizes[0] ==
                  kIntermediateSize,
          "capability: supported_intermediate_sizes is not exactly [512]");
  require(capability.supported_topk.size() == 1 &&
              capability.supported_topk[0] == kTopK,
          "capability: supported_topk is not exactly [8]");
  require(capability.kernel_families.size() == 2 &&
              capability.kernel_families[0] == "nvfp4_moe_split" &&
              capability.kernel_families[1] == "nvfp4_moe_fused",
          "capability: kernel_families is not exactly "
          "[nvfp4_moe_split, nvfp4_moe_fused]");
  require(capability.num_resident_experts ==
              kLayers * kExpertsPerLayer,
          "capability: num_resident_experts is not 8192");
  require(capability.max_batch_remote == kMaxBatch,
          "capability: max_batch_remote is not 128");
  require(capability.health_heartbeat_interval_ms == 1000,
          "capability: health heartbeat interval is not 1000 ms");
  require(capability.num_layers == kLayers,
          "capability: num_layers is not 32");
  require(capability.experts_per_layer == kExpertsPerLayer,
          "capability: experts_per_layer is not 256");

  const auto health = provider.health();
  require(health.loaded, "health after load: loaded is false");
  require(!health.pending, "health after load: pending is true");
  require(!health.stopped, "health after load: stopped is true");
  require(health.generation == kGeneration,
          "health after load: generation does not match config");
  require(health.dispatches == 0,
          "health after load: dispatches is not zero");
  require(health.allocations > 0,
          "health after load: allocations is zero");
  require(health.last_error.empty(),
          "health after load: last_error is not empty: " + health.last_error);
  *allocation_baseline = health.allocations;
}

void test_validation_and_single_flight(B70Provider& provider,
                                       const GoldenReference& golden,
                                       std::uint64_t sequence) {
  HostBatch batch = make_batch(golden, 1, false);

  require_status(provider.issue(kGeneration + 1, sequence, 0,
                                batch.hidden.data(), batch.ids.data(),
                                batch.weights.data(), batch.M),
                 ProviderStatus::generation_mismatch,
                 "issue wrong generation");
  require_status(provider.issue(kGeneration, sequence, kLayers,
                                batch.hidden.data(), batch.ids.data(),
                                batch.weights.data(), batch.M),
                 ProviderStatus::invalid_argument, "issue invalid layer");

  batch.ids[0] = static_cast<std::int32_t>(kExpertsPerLayer);
  require_status(provider.issue(kGeneration, sequence, 0, batch.hidden.data(),
                                batch.ids.data(), batch.weights.data(), batch.M),
                 ProviderStatus::invalid_argument,
                 "issue out-of-range expert ID 256");
  batch.ids[0] = -2;
  require_status(provider.issue(kGeneration, sequence, 0, batch.hidden.data(),
                                batch.ids.data(), batch.weights.data(), batch.M),
                 ProviderStatus::invalid_argument,
                 "issue out-of-range expert ID -2");
  batch.ids[0] = 0;

  const auto after_rejections = provider.health();
  require(!after_rejections.pending,
          "health after rejected issues: pending is true");
  require(after_rejections.dispatches == 0,
          "health after rejected issues: dispatches changed");

  issue_dispatch(provider, batch, sequence, "single-route");
  require(provider.health().pending,
          "health after accepted issue: pending is false");
  require_status(provider.issue(kGeneration, sequence + 1, 0,
                                batch.hidden.data(), batch.ids.data(),
                                batch.weights.data(), batch.M),
                 ProviderStatus::busy, "issue while request pending");

  DispatchResult wrong_result{};
  require_status(provider.take(kGeneration + 1, sequence, batch.output.data(),
                               batch.output.size(), &wrong_result),
                 ProviderStatus::generation_mismatch,
                 "take wrong generation");
  require(provider.health().pending,
          "health after wrong-generation take: pending was cleared");

  require_status(provider.take(kGeneration, sequence + 1, batch.output.data(),
                               batch.output.size(), &wrong_result),
                 ProviderStatus::sequence_mismatch,
                 "take wrong sequence");
  require(provider.health().pending,
          "health after wrong-sequence take: pending was cleared");

  finish_dispatch(provider, batch, sequence, golden, "single-route", "split");
  require(!provider.health().pending,
          "health after successful take: pending is true");
}

void run_repeated_row_cases(B70Provider& provider,
                            const GoldenReference& golden,
                            std::uint64_t* next_sequence) {
  constexpr std::array<std::size_t, 7> batch_sizes{1, 2, 4, 8, 16, 32, 128};
  for (const std::size_t M : batch_sizes) {
    HostBatch batch = make_batch(golden, M, false);
    const std::uint64_t sequence = (*next_sequence)++;
    issue_dispatch(provider, batch, sequence, "single-route");
    finish_dispatch(provider, batch, sequence, golden, "single-route",
                    M <= 32 ? "split" : "fused");
  }
}

void run_duplicate_top8_case(B70Provider& provider,
                             const GoldenReference& golden,
                             std::uint64_t sequence) {
  constexpr std::size_t M = 8;
  HostBatch batch = make_batch(golden, M, true);
  issue_dispatch(provider, batch, sequence, "duplicate-top8");
  finish_dispatch(provider, batch, sequence, golden, "duplicate-top8",
                  "split");
}

void test_invalid_resident_lists() {
  const auto require_invalid_load =
      [](std::vector<std::int32_t> resident_experts,
         std::string_view case_name) {
        B70Provider provider;
        ProviderConfig config;
        config.max_batch = kMaxBatch;
        config.top_k = kTopK;
        config.generation = kGeneration;
        config.resident_experts = std::move(resident_experts);
        require_status(provider.load(std::string(kBankPath), config),
                       ProviderStatus::invalid_argument, case_name);
        require(!provider.health().loaded,
                std::string(case_name) + ": provider became loaded");
      };

  require_invalid_load({0, 0}, "load duplicate resident expert");
  require_invalid_load({-1, 0}, "load negative resident expert");
  require_invalid_load(
      {0, static_cast<std::int32_t>(kExpertsPerLayer)},
      "load resident expert beyond model geometry");
}

void test_compact_resident_gather(const GoldenReference& golden) {
  constexpr std::uint64_t kCompactSequence = 20000;
  constexpr std::size_t kCompactExpertsPerLayer = 2;
  constexpr std::int32_t kCanonicalZeroLocalSlot = 1;

  B70Provider provider;
  ProviderConfig config;
  config.max_batch = kMaxBatch;
  config.top_k = kTopK;
  config.generation = kGeneration;
  config.resident_experts = {1, 0};
  require_status(provider.load(std::string(kBankPath), config),
                 ProviderStatus::ok, "compact resident load [1,0]");

  const auto capability = provider.capability();
  require(capability.num_resident_experts ==
              kLayers * kCompactExpertsPerLayer,
          "compact capability: num_resident_experts is not 64");
  require(capability.experts_per_layer == kExpertsPerLayer,
          "compact capability: experts_per_layer changed from 256");
  const std::uint64_t allocation_baseline = provider.health().allocations;
  require(allocation_baseline > 0,
          "compact health: allocation baseline is zero");

  HostBatch batch = make_batch(golden, 1, false);
  batch.ids[0] = static_cast<std::int32_t>(kCompactExpertsPerLayer);
  require_status(provider.issue(kGeneration, kCompactSequence, 0,
                                batch.hidden.data(), batch.ids.data(),
                                batch.weights.data(), batch.M),
                 ProviderStatus::invalid_argument,
                 "compact issue local ID beyond residency");
  require(provider.health().allocations == allocation_baseline,
          "compact rejected issue changed allocation count");

  batch.ids[0] = kCanonicalZeroLocalSlot;
  require_status(provider.issue(kGeneration, kCompactSequence, 0,
                                batch.hidden.data(), batch.ids.data(),
                                batch.weights.data(), batch.M),
                 ProviderStatus::ok,
                 "compact issue canonical expert 0 through local slot 1");
  finish_dispatch(provider, batch, kCompactSequence, golden,
                  "compact-[1,0]-canonical-0", "split");

  const auto final_health = provider.health();
  require(final_health.dispatches == 1,
          "compact health: accepted dispatch count is not one");
  require(final_health.allocations == allocation_baseline,
          "compact allocation invariant changed after dispatch");
}

#ifdef SHOOTING_BRAKE_ENABLE_TEST_FAULTS
void test_sequence_bound_copyout_fault(const GoldenReference& golden) {
  constexpr std::uint64_t kPassSequence = 10000;
  constexpr std::uint64_t kFaultSequence = 10001;
  constexpr float kOutputSentinel = 12345.25F;

  B70Provider provider;
  require_status(
      provider.arm_test_fault(
          ProviderTestFault::after_kernel_before_copyout, kFaultSequence),
      ProviderStatus::not_loaded, "arm fault before load");

  ProviderConfig config;
  config.max_batch = kMaxBatch;
  config.top_k = kTopK;
  config.generation = kGeneration;
  require_status(provider.load(std::string(kBankPath), config),
                 ProviderStatus::ok, "fault-test load");
  const std::uint64_t allocation_baseline = provider.health().allocations;

  require_status(
      provider.arm_test_fault(
          ProviderTestFault::after_kernel_before_copyout, 0),
      ProviderStatus::invalid_argument, "arm fault with zero sequence");
  require_status(
      provider.arm_test_fault(
          ProviderTestFault::after_kernel_before_copyout, kFaultSequence),
      ProviderStatus::ok, "arm sequence-bound copyout fault");
  require_status(
      provider.arm_test_fault(
          ProviderTestFault::after_kernel_before_copyout, kFaultSequence + 1),
      ProviderStatus::busy, "arm second fault while one is armed");

  HostBatch passing_batch = make_batch(golden, 1, false);
  issue_dispatch(provider, passing_batch, kPassSequence,
                 "fault-sequence-nonmatch");
  finish_dispatch(provider, passing_batch, kPassSequence, golden,
                  "fault-sequence-nonmatch", "split");
  require(provider.health().allocations == allocation_baseline,
          "fault sequence nonmatch changed allocation count");

  HostBatch fault_batch = make_batch(golden, 1, false);
  std::fill(fault_batch.output.begin(), fault_batch.output.end(),
            kOutputSentinel);
  DispatchResult result;
  result.generation = std::numeric_limits<std::uint64_t>::max();
  result.sequence = std::numeric_limits<std::uint64_t>::max() - 1;
  result.M = std::numeric_limits<std::size_t>::max();
  result.kernel = "fault-result-sentinel";
  result.kernel_us = -1.0;
  result.total_us = -2.0;

  const auto require_sentinels = [&]() {
    require(std::all_of(fault_batch.output.begin(), fault_batch.output.end(),
                        [](const float value) {
                          return value == kOutputSentinel;
                        }),
            "fault take modified caller output");
    require(result.generation ==
                    std::numeric_limits<std::uint64_t>::max() &&
                result.sequence ==
                    std::numeric_limits<std::uint64_t>::max() - 1 &&
                result.M == std::numeric_limits<std::size_t>::max() &&
                result.kernel == "fault-result-sentinel" &&
                result.kernel_us == -1.0 && result.total_us == -2.0,
            "fault take modified caller dispatch result");
  };

  issue_dispatch(provider, fault_batch, kFaultSequence,
                 "after-kernel-before-copyout-fault");
  require_status(
      provider.arm_test_fault(
          ProviderTestFault::after_kernel_before_copyout, kFaultSequence + 1),
      ProviderStatus::busy, "arm fault while dispatch is pending");
  require_status(provider.take(kGeneration + 1, kFaultSequence,
                               fault_batch.output.data(),
                               fault_batch.output.size(), &result),
                 ProviderStatus::generation_mismatch,
                 "fault take wrong generation");
  require_sentinels();
  require_status(provider.take(kGeneration, kFaultSequence + 1,
                               fault_batch.output.data(),
                               fault_batch.output.size(), &result),
                 ProviderStatus::sequence_mismatch,
                 "fault take wrong sequence");
  require_sentinels();
  require_status(provider.take(kGeneration, kFaultSequence,
                               fault_batch.output.data(),
                               fault_batch.output.size(), &result),
                 ProviderStatus::device_error, "fault take exact sequence");
  require_sentinels();
  require_status(provider.take(kGeneration, kFaultSequence,
                               fault_batch.output.data(),
                               fault_batch.output.size(), &result),
                 ProviderStatus::sequence_mismatch,
                 "fault take after terminal retirement");
  require_sentinels();

  const auto fault_health = provider.health();
  require(fault_health.loaded, "fault health: provider is not loaded");
  require(!fault_health.pending,
          "fault health: terminal dispatch was not retired");
  require(fault_health.dispatches == 2,
          "fault health: expected two accepted dispatches");
  require(fault_health.allocations == allocation_baseline,
          "fault health: allocation count changed after load");
  require(fault_health.last_error ==
              "injected device error after kernel before copyout",
          "fault health: deterministic device error does not match");
  require_status(
      provider.arm_test_fault(
          ProviderTestFault::after_kernel_before_copyout, kFaultSequence + 1),
      ProviderStatus::ok, "arm fault after terminal retirement");

  provider.shutdown();
  const auto shutdown_health = provider.health();
  require(!shutdown_health.loaded,
          "fault cleanup: provider is still loaded");
  require(!shutdown_health.pending,
          "fault cleanup: pending dispatch was not cleared");
  require(shutdown_health.stopped,
          "fault cleanup: provider was not stopped");
  require(shutdown_health.allocations == allocation_baseline,
          "fault cleanup: allocation count changed");
  require_status(
      provider.arm_test_fault(
          ProviderTestFault::after_kernel_before_copyout, kFaultSequence),
      ProviderStatus::shutdown, "arm fault after cleanup");
}
#endif

}  // namespace

int main() {
  try {
    static_assert(sizeof(sycl::half) == 2,
                  "golden_reference.bin requires 16-bit sycl::half");

    const GoldenReference golden = load_golden_reference(kGoldenPath);
    B70Provider provider;
    ProviderConfig config;
    config.max_batch = kMaxBatch;
    config.top_k = kTopK;
    config.generation = kGeneration;

    const ProviderStatus load_status =
        provider.load(std::string(kBankPath), config);
    if (load_status != ProviderStatus::ok) {
      const auto health = provider.health();
      std::ostringstream message;
      message << "load " << kBankPath << ": expected status ok, got "
              << status_name(load_status);
      if (!health.last_error.empty()) {
        message << "; last_error=" << health.last_error;
      }
      fail(message.str());
    }

    std::uint64_t allocation_baseline = 0;
    test_loaded_capability_and_health(provider, &allocation_baseline);

    std::uint64_t next_sequence = 100;
    test_validation_and_single_flight(provider, golden, next_sequence++);
    run_repeated_row_cases(provider, golden, &next_sequence);
    run_duplicate_top8_case(provider, golden, next_sequence++);

    const auto final_health = provider.health();
    require(final_health.loaded,
            "health after dispatches: loaded is false");
    require(!final_health.pending,
            "health after dispatches: pending is true");
    require(!final_health.stopped,
            "health after dispatches: stopped is true");
    require(final_health.generation == kGeneration,
            "health after dispatches: generation changed");
    require(final_health.dispatches == 9,
            "health after dispatches: expected 9 successful dispatches");
    require(final_health.allocations == allocation_baseline,
            "allocation invariant: allocation count changed after load");
    require(final_health.last_error.empty(),
            "health after dispatches: last_error is not empty: " +
                final_health.last_error);

    provider.shutdown();
    provider.shutdown();
    const auto shutdown_health = provider.health();
    require(!shutdown_health.loaded,
            "health after shutdown: loaded is true");
    require(!shutdown_health.pending,
            "health after shutdown: pending is true");
    require(shutdown_health.stopped,
            "health after shutdown: stopped is false");
    require(shutdown_health.generation == kGeneration,
            "health after shutdown: generation changed");
    require(shutdown_health.dispatches == 9,
            "health after shutdown: dispatch count changed");
    require(shutdown_health.allocations == allocation_baseline,
            "health after shutdown: allocation count changed");

    HostBatch rejected = make_batch(golden, 1, false);
    require_status(provider.issue(kGeneration, next_sequence, 0,
                                  rejected.hidden.data(), rejected.ids.data(),
                                  rejected.weights.data(), rejected.M),
                   ProviderStatus::shutdown, "issue after shutdown");

    test_invalid_resident_lists();
    test_compact_resident_gather(golden);

#ifdef SHOOTING_BRAKE_ENABLE_TEST_FAULTS
    test_sequence_bound_copyout_fault(golden);
#endif

    std::cout << "Phase-1 PASS\n";
    return 0;
  } catch (const TestFailure& error) {
    std::cerr << "Phase-1 FAIL: " << error.what() << '\n';
    return 1;
  } catch (const std::exception& error) {
    std::cerr << "Phase-1 FAIL: unexpected exception: " << error.what()
              << '\n';
    return 1;
  } catch (...) {
    std::cerr << "Phase-1 FAIL: unexpected non-standard exception\n";
    return 1;
  }
}
