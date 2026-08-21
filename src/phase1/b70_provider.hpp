#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

namespace shooting_brake::phase1 {

enum class ProviderStatus {
  ok,
  busy,
  not_loaded,
  invalid_argument,
  generation_mismatch,
  sequence_mismatch,
  device_error,
  shutdown,
};

#ifdef SHOOTING_BRAKE_ENABLE_TEST_FAULTS
enum class ProviderTestFault : std::uint32_t {
  after_kernel_before_copyout = 1,
};
#endif

struct ProviderConfig {
  std::size_t max_batch = 128;
  std::size_t top_k = 8;
  std::uint64_t generation = 1;
  std::vector<std::int32_t> resident_experts;
  // Empty preserves the legacy first-B70 selection. A decimal string selects
  // the zero-based B70 index; any other string is matched as a PCI BDF.
  std::string device_selector;
};

struct Capability {
  std::uint32_t protocol_version = 1;
  std::string device_name;
  std::uint32_t device_index = 0;
  std::string device_pci_bdf;
  std::uint64_t device_memory_total_bytes = 0;
  std::uint64_t device_memory_available_bytes = 0;
  std::string backend;
  std::vector<std::uint32_t> supported_hidden_sizes;
  std::vector<std::uint32_t> supported_intermediate_sizes;
  std::vector<std::uint32_t> supported_topk;
  std::uint32_t num_resident_experts = 0;
  // Result wire width. When true the provider copies fp16 out of
  // take() and the caller MUST supply fp16 storage; the pair is
  // published rather than inferred from the environment on both sides,
  // because a mismatch is silent garbage rather than an error.
  bool output_fp16 = false;
  std::uint32_t max_batch_remote = 0;
  std::vector<std::string> kernel_families;
  std::uint32_t health_heartbeat_interval_ms = 1000;
  std::uint32_t num_layers = 0;
  std::uint32_t experts_per_layer = 0;
  // SBINT401 compact slot -> source expert ID map. Empty for SBEXP001, whose
  // resident selection remains caller-defined.
  std::vector<std::int32_t> source_expert_ids;
};

struct Health {
  bool loaded = false;
  bool pending = false;
  bool stopped = false;
  std::uint64_t generation = 0;
  std::uint64_t dispatches = 0;
  std::uint64_t allocations = 0;
  std::string last_error;
};

struct DispatchResult {
  std::uint64_t generation = 0;
  std::uint64_t sequence = 0;
  std::size_t M = 0;
  std::string kernel;
  double kernel_us = 0.0;
  double total_us = 0.0;
};

class B70Provider final {
 public:
  B70Provider();
  ~B70Provider();

  B70Provider(const B70Provider&) = delete;
  B70Provider& operator=(const B70Provider&) = delete;
  B70Provider(B70Provider&&) = delete;
  B70Provider& operator=(B70Provider&&) = delete;

  ProviderStatus load(const std::string& bank_path,
                      const ProviderConfig& config = {});
  Capability capability() const;
  Health health() const;

  ProviderStatus issue(std::uint64_t generation, std::uint64_t sequence,
                       std::size_t layer, const sycl::half* hidden,
                       const std::int32_t* ids, const float* weights,
                       std::size_t M);
  ProviderStatus take(std::uint64_t generation, std::uint64_t sequence,
                      float* output, std::size_t output_elements,
                      DispatchResult* result);
#ifdef SHOOTING_BRAKE_ENABLE_TEST_FAULTS
  ProviderStatus arm_test_fault(ProviderTestFault fault,
                                std::uint64_t sequence);
#endif

  // Free and total bytes on the selected B70, read from the provider's own
  // device handle rather than a second lookup -- so the occupancy reported
  // is guaranteed to be the card the expert bank actually loaded onto,
  // which matters on a host carrying more than one Intel GPU.
  //
  // Uses the ext_intel_free_memory aspect. Returns false when the runtime
  // does not expose it, leaving both outputs untouched: occupancy is
  // reporting, never a reason to fail a dispatch.
  bool device_memory(std::size_t* free_bytes,
                     std::size_t* total_bytes) const noexcept;

  // Registers an externally-pinned host range (e.g. a torch pin_memory
  // staging buffer, which lives in the CUDA caching host allocator) with
  // this provider's SYCL context via prepare_for_device_copy, so doorbell
  // H2D/D2H copies from it run as direct DMA instead of staged (H2D) or
  // synchronous (D2H) pageable copies. Registration is context-scoped:
  // CUDA pinning is invisible to the B70's Level Zero context.
  // Measured on this box (experiments/b70_dispatch_latency environment,
  // clock-pinned): cudaHostAlloc 29.3 -> 20.5 us per dispatch at the
  // production 180 us duty cycle.
  //
  // Requires load() to have succeeded (needs the context). Fail-open:
  // returns false on any failure and the range simply stays pageable
  // from the B70's side. Ranges are released at shutdown(); the memory
  // itself must outlive the provider's use of it either way.
  bool register_host_range(const void* ptr, std::size_t bytes) noexcept;

  void shutdown() noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace shooting_brake::phase1
