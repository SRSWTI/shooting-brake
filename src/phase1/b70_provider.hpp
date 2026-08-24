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

  // Doorbell 2.0 (SHOOTING_BRAKE_B70_CS_DOORBELL=1): enqueue one layer's full
  // dispatch as a command-streamer chain on the in-order queue --
  //   MI_SEMAPHORE_WAIT(signal==M) -> WRITE(signal=0) -> H2D from the pinned
  //   ring -> kernels -> narrow -> D2H to the ring -> CCS marker ->
  //   WRITE(completion=1)
  // -- so the host leaves the dispatch critical path entirely. Probes
  // (experiments/b70_cs_doorbell_probe, b70_sycl_zex_interop_probe) measured
  // the bracket at 7.9-8.7 us against the 61 us host handshake, and caught
  // the two hazards this signature encodes: the signal self-clear happens on
  // the CS (no host clear race), and the completion write is ordered behind a
  // CCS marker kernel because the D2H may ride the BCS.
  //
  // Serves the GEMV/split shapes only (M <= 32); larger M returns
  // invalid_argument and the caller falls back to classic issue/take, which
  // keeps prefill on the pipelined path. Fire-and-forget: no take(), no
  // pending bookkeeping; the CUDA side observes completion directly.
  // Blocks until every enqueued CS chain for the current step has drained
  // its SYCL portion. The poller MUST call this after enqueueing a step's
  // chains: its scan loop otherwise races the chains for the very signals
  // they wait on -- the classic sweep steals-and-clears a later layer's
  // signal before that layer's MI_SEMAPHORE_WAIT fires, deadlocking the
  // in-order queue (found via experiments/b70_cs_harness.py, step 1 layer 14,
  // poller parked in take()'s wait_and_throw). Safe resume window: by the
  // time the SYCL tail drains, every chain has consumed its signal, so the
  // sweep finds nothing to steal; the next step's signal0 cannot arrive
  // before the last completion because the consumer is sequential.
  ProviderStatus cs_step_fence();

  // Doorbell 2.0 "ride-along" (mode 2, SHOOTING_BRAKE_B70_CS_DOORBELL=2):
  // the WAIT/WRITE brackets append onto the SYCL queue's OWN immediate
  // command list (fetched fresh per call), so brackets, copies, and kernels
  // ride ONE in-order list -- zero cross-runtime event seams. Marker kernels
  // join raw list-order into SYCL's event chain across the BCS/CCS split
  // (probe #2: full chain 7.9-8.7 us). Requires one-list-per-queue
  // semantics: the V1 UR adapter, or a V2 that has not rotated the list.
  ProviderStatus issue_cs_ride(std::uint64_t generation, std::size_t layer,
                               const sycl::half* ring_hidden,
                               const std::int32_t* ring_ids,
                               const float* ring_weights, float* ring_output,
                               std::size_t M, volatile std::uint32_t* signal,
                               volatile std::uint32_t* completion);

  // Baked decode chain (mode 3, SHOOTING_BRAKE_B70_CS_DOORBELL=3): each layer
  // gets ONE pre-recorded raw command list -- WAIT(signal==1) ->
  // WRITE(signal=0) -> H2D x3 -> gate_up -> fused w2 epilogue (direct fp16)
  // -> D2H -> WRITE(completion=1) -- recorded once at registration, replayed
  // per step with a single ExecuteCommandLists. No live appends (the 80-95 us
  // per-bracket trap, kill-bench 22), no UR adapter involvement, no event
  // seams (kill-bench 23). M=1 decode and out_fp16 only; everything else
  // stays on the classic sweep.
  ProviderStatus baked_record_layer(std::uint64_t generation,
                                    std::size_t layer,
                                    const sycl::half* ring_hidden,
                                    const std::int32_t* ring_ids,
                                    const float* ring_weights,
                                    float* ring_output,
                                    volatile std::uint32_t* signal,
                                    volatile std::uint32_t* completion);
  ProviderStatus baked_execute_step();

  ProviderStatus issue_cs_chain(std::uint64_t generation, std::size_t layer,
                                const sycl::half* ring_hidden,
                                const std::int32_t* ring_ids,
                                const float* ring_weights, float* ring_output,
                                std::size_t M, volatile std::uint32_t* signal,
                                volatile std::uint32_t* completion);
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
