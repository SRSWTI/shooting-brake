#pragma once

#include "ring_protocol.hpp"

#include <cstdint>

namespace shooting_brake::phase2 {

enum class RingStatus : std::uint32_t {
  ok = 0,
  would_block,
  invalid_argument,
  not_open,
  system_error,
  protocol_mismatch,
  layout_mismatch,
  generation_mismatch,
  sequence_exhausted,
  state_mismatch,
  invalid_descriptor,
  invalid_payload,
  stale_completion,
  cancelled,
  device_failure,
  quarantined,
  deadline_expired,
  generation_dead,
};
const char* status_message(RingStatus status) noexcept;

template <typename T>
struct PayloadSpan final {
  T* data = nullptr;
  std::uint32_t count = 0;
  T& operator[](std::uint32_t index) const noexcept { return data[index]; }
  explicit operator bool() const noexcept { return data != nullptr; }
};

struct RingIdentity final {
  std::uint64_t ring_generation = 0;
  std::uint64_t provider_generation = 0;
  std::uint64_t placement_generation = 0;
  std::uint64_t weight_generation = 0;
  Nonce provider_nonce{};
  Nonce ring_nonce{};
  Fingerprint placement_sha256{};
  Fingerprint weight_sha256{};
  std::uint64_t provider_pid = 0;
};

struct RequestSpec final {
  std::uint64_t request_seq = 0;
  std::uint64_t scheduler_step = 0;
  std::uint64_t deadline_monotonic_ns = 0;
  std::uint32_t layer = 0;
  std::uint32_t num_batch_tokens = 0;
  std::uint32_t num_staged_tokens = 0;
  std::uint32_t num_routes = 0;
};

struct RingTicket final {
  std::uint64_t request_seq = 0;
  std::uint64_t request_generation = 0;
  std::uint64_t output_buffer_version = 0;
  std::uint32_t slot = 0;
};

struct RequestPayload final {
  PayloadSpan<std::uint16_t> activation_fp16;
  PayloadSpan<std::int32_t> canonical_ids;
  PayloadSpan<float> routing_weights;
  PayloadSpan<std::uint8_t> remote_mask;
  PayloadSpan<std::uint32_t> token_row_map;
  PayloadSpan<std::uint16_t> canonical_route_positions;
};
struct ConstRequestPayload final {
  PayloadSpan<const std::uint16_t> activation_fp16;
  PayloadSpan<const std::int32_t> canonical_ids;
  PayloadSpan<const float> routing_weights;
  PayloadSpan<const std::uint8_t> remote_mask;
  PayloadSpan<const std::uint32_t> token_row_map;
  PayloadSpan<const std::uint16_t> canonical_route_positions;
};
struct CompletionPayload final {
  PayloadSpan<float> output_fp32;
  PayloadSpan<std::uint8_t> route_status;
  PayloadSpan<std::uint8_t> token_status;
};
struct ConstCompletionPayload final {
  PayloadSpan<const float> output_fp32;
  PayloadSpan<const std::uint8_t> route_status;
  PayloadSpan<const std::uint8_t> token_status;
};
struct CompletionTiming final {
  std::uint64_t kernel_ns = 0;
  std::uint64_t provider_total_ns = 0;
  std::uint64_t completion_monotonic_ns = 0;
};
struct RequestMetadata final {
  std::uint64_t scheduler_step = 0;
  std::uint64_t deadline_monotonic_ns = 0;
  std::uint32_t layer = 0;
  std::uint32_t num_batch_tokens = 0;
  std::uint32_t num_staged_tokens = 0;
  std::uint32_t num_routes = 0;
};

struct ProviderClaim final {
  RingTicket ticket{};
  RequestMetadata metadata{};
  ConstRequestPayload request{};
  CompletionPayload completion{};
  bool valid = false;
};

// Move-only owner of one sealed memfd mapping and two nonblocking eventfd hints.
// State words remain authoritative. All hot operations are bounded/noexcept and
// allocate nothing; typed payload views are returned only after exact descriptor
// validation.
class SharedRing final {
 public:
  SharedRing() noexcept = default;
  ~SharedRing() noexcept;
  SharedRing(const SharedRing&) = delete;
  SharedRing& operator=(const SharedRing&) = delete;
  SharedRing(SharedRing&& other) noexcept;
  SharedRing& operator=(SharedRing&& other) noexcept;

  static RingStatus create(const RingIdentity& identity, SharedRing* out,
                           const char** detail = nullptr) noexcept;
  static RingStatus attach(int mapping_fd, int request_eventfd,
                           int completion_eventfd,
                           const RingIdentity& expected, SharedRing* out,
                           const char** detail = nullptr) noexcept;

  bool is_open() const noexcept { return mapping_ != nullptr; }
  int fd() const noexcept { return mapping_fd_; }
  int request_eventfd() const noexcept { return request_eventfd_; }
  int completion_eventfd() const noexcept { return completion_eventfd_; }
  std::uint64_t mapping_size() const noexcept { return mapping_size_; }
  const RingIdentity& identity() const noexcept { return identity_; }

  // Startup/shutdown control-plane hook for registering or unregistering the
  // exact physical mapping (for example cudaHostRegister/cudaHostUnregister).
  // No pointer is retained in the wire ABI and this is never a token-path API.
  using MappingVisitor =
      RingStatus (*)(void* mapping, std::uint64_t bytes,
                     void* context) noexcept;
  RingStatus visit_mapping(MappingVisitor visitor, void* context) noexcept;

  RingStatus host_begin(const RequestSpec& spec, RingTicket* ticket,
                        RequestPayload* payload) noexcept;
  RingStatus host_publish(const RingTicket& ticket) noexcept;
  RingStatus host_cancel(const RingTicket& ticket) noexcept;
  RingStatus host_timeout(const RingTicket& ticket) noexcept;
  RingStatus host_consume(const RingTicket& ticket,
                          ConstCompletionPayload* payload,
                          CompletionCode* completion,
                          ErrorCode* error) noexcept;
  RingStatus host_completion_timing(const RingTicket& ticket,
                                    CompletionTiming* timing) const noexcept;
  RingStatus host_reclaim(const RingTicket& ticket) noexcept;

  RingStatus provider_claim(std::uint64_t now_monotonic_ns,
                            ProviderClaim* claim) noexcept;
  RingStatus provider_complete(const ProviderClaim& claim,
                               CompletionCode completion, ErrorCode error,
                               std::uint64_t kernel_ns = 0,
                               std::uint64_t total_ns = 0,
                               std::uint64_t completion_ns = 0) noexcept;

  RingStatus teardown_generation(std::uint64_t expected_ring_generation,
                                 std::uint64_t expected_provider_generation) noexcept;
  RingStatus drain_request_notifications() noexcept;
  RingStatus drain_completion_notifications() noexcept;
  SlotState slot_state(std::uint32_t slot) const noexcept;
  bool slot_quarantined(std::uint32_t slot) const noexcept;

 private:
  SharedRing(int mapping_fd, int request_eventfd, int completion_eventfd,
             void* mapping, const RingIdentity& identity) noexcept;
  void reset() noexcept;

  int mapping_fd_ = -1;
  int request_eventfd_ = -1;
  int completion_eventfd_ = -1;
  void* mapping_ = nullptr;
  std::uint64_t mapping_size_ = 0;
  RingIdentity identity_{};
  std::uint64_t next_host_seq_ = 1;
  std::uint64_t next_provider_seq_ = 1;
  bool provider_busy_ = false;
  bool dead_ = false;
};

}  // namespace shooting_brake::phase2
