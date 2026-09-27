#include "shared_ring.hpp"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include <sys/mman.h>

namespace sb = shooting_brake::phase2;

namespace {

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

void require(bool condition, const char* message) {
  if (!condition) {
    fail(message);
  }
}

void require_status(sb::RingStatus actual, sb::RingStatus expected,
                    const char* operation) {
  if (actual != expected) {
    fail(std::string(operation) + ": expected " + sb::status_message(expected) +
         ", got " + sb::status_message(actual));
  }
}

sb::RingIdentity identity(std::uint64_t ring_generation,
                          std::uint64_t provider_generation) {
  sb::RingIdentity value{};
  value.ring_generation = ring_generation;
  value.provider_generation = provider_generation;
  value.placement_generation = 29;
  value.weight_generation = 31;
  value.provider_pid = 501;
  for (std::uint32_t index = 0; index < sb::kNonceBytes; ++index) {
    value.provider_nonce.bytes[index] =
        static_cast<std::uint8_t>(0x10U + index);
    value.ring_nonce.bytes[index] = static_cast<std::uint8_t>(0x80U + index);
  }
  for (std::uint32_t index = 0; index < sb::kFingerprintBytes; ++index) {
    value.placement_sha256.bytes[index] =
        static_cast<std::uint8_t>(index * 3U + 5U);
    value.weight_sha256.bytes[index] =
        static_cast<std::uint8_t>(index * 5U + 9U);
  }
  return value;
}

sb::SharedRing create_ring(std::uint64_t ring_generation = 1,
                           std::uint64_t provider_generation = 1) {
  sb::SharedRing ring;
  const char* detail = nullptr;
  require_status(sb::SharedRing::create(
                     identity(ring_generation, provider_generation), &ring,
                     &detail),
                 sb::RingStatus::ok,
                 detail == nullptr ? "SharedRing::create" : detail);
  return ring;
}

struct Request {
  sb::RingTicket ticket{};
  sb::RequestPayload payload{};
};

Request begin_request(sb::SharedRing& ring) {
  sb::RequestSpec spec{};
  spec.request_seq = 1;
  spec.scheduler_step = 1001;
  spec.deadline_monotonic_ns = std::numeric_limits<std::uint64_t>::max();
  spec.layer = 0;
  spec.num_batch_tokens = 1;
  spec.num_staged_tokens = 1;
  spec.num_routes = 1;

  Request request;
  require_status(ring.host_begin(spec, &request.ticket, &request.payload),
                 sb::RingStatus::ok, "host_begin");
  require(request.ticket.slot == 0, "sequence 1 did not select slot 0");
  require(ring.slot_state(0) == sb::SlotState::host_writing,
          "host_begin did not publish HOST_WRITING");

  request.payload.token_row_map[0] = 0;
  request.payload.remote_mask[0] = 1;
  request.payload.canonical_ids[0] = 0;
  request.payload.routing_weights[0] = 1.0F;
  request.payload.canonical_route_positions[0] = 0;
  return request;
}

void publish(sb::SharedRing& ring, const Request& request) {
  require_status(ring.host_publish(request.ticket), sb::RingStatus::ok,
                 "host_publish");
  require(ring.slot_state(request.ticket.slot) == sb::SlotState::request_ready,
          "host_publish did not publish REQUEST_READY");
}

void fill_terminal(sb::ProviderClaim& claim, bool contributed) {
  for (std::uint32_t index = 0; index < claim.completion.route_status.count;
       ++index) {
    claim.completion.route_status[index] =
        claim.request.remote_mask[index] == 0U
            ? sb::wire(sb::RouteStatus::not_remote)
            : sb::wire(contributed ? sb::RouteStatus::contributed
                                   : sb::RouteStatus::not_contributed);
  }
  for (std::uint32_t row = 0; row < claim.completion.token_status.count; ++row) {
    claim.completion.token_status[row] =
        sb::wire(contributed ? sb::TokenStatus::contributed
                             : sb::TokenStatus::not_contributed);
  }
  if (contributed) {
    for (std::uint32_t index = 0; index < claim.completion.output_fp32.count;
         ++index) {
      claim.completion.output_fp32[index] = 1.0F;
    }
  }
}

sb::ProviderClaim claim(sb::SharedRing& ring, sb::RingStatus expected) {
  sb::ProviderClaim result{};
  require_status(ring.provider_claim(1, &result), expected, "provider_claim");
  require(result.valid, "provider_claim did not return a closable claim");
  require(ring.slot_state(result.ticket.slot) ==
              sb::SlotState::provider_running,
          "provider_claim did not publish PROVIDER_RUNNING");
  return result;
}

void consume_and_reclaim(sb::SharedRing& ring, const sb::RingTicket& ticket,
                         sb::RingStatus expected, bool output_expected) {
  sb::ConstCompletionPayload output{};
  sb::CompletionCode completion = sb::CompletionCode::unset;
  sb::ErrorCode error = sb::ErrorCode::internal;
  require_status(ring.host_consume(ticket, &output, &completion, &error),
                 expected, "host_consume");
  require(static_cast<bool>(output.output_fp32) == output_expected,
          "host_consume output exposure disagreed with terminal status");
  require(ring.slot_state(ticket.slot) == sb::SlotState::host_reading,
          "host_consume did not acquire HOST_READING");
  require_status(ring.host_reclaim(ticket), sb::RingStatus::ok,
                 "host_reclaim");
  require(ring.slot_state(ticket.slot) == sb::SlotState::free,
          "host_reclaim did not publish FREE");
}

void success_path() {
  sb::SharedRing ring = create_ring();
  Request request = begin_request(ring);
  publish(ring, request);
  sb::ProviderClaim provider = claim(ring, sb::RingStatus::ok);
  fill_terminal(provider, true);
  require_status(ring.provider_complete(provider, sb::CompletionCode::ok_all,
                                        sb::ErrorCode::none),
                 sb::RingStatus::ok, "provider_complete success");
  require(ring.slot_state(request.ticket.slot) == sb::SlotState::response_ready,
          "provider_complete did not publish RESPONSE_READY");
  consume_and_reclaim(ring, request.ticket, sb::RingStatus::ok, true);
}

void cancel_before_claim() {
  sb::SharedRing ring = create_ring();
  Request request = begin_request(ring);
  publish(ring, request);
  require_status(ring.host_cancel(request.ticket), sb::RingStatus::cancelled,
                 "host_cancel before claim");
  sb::ProviderClaim provider = claim(ring, sb::RingStatus::cancelled);
  fill_terminal(provider, false);
  require_status(ring.provider_complete(provider, sb::CompletionCode::cancelled,
                                        sb::ErrorCode::cancelled),
                 sb::RingStatus::ok, "provider_complete cancellation");
  consume_and_reclaim(ring, request.ticket, sb::RingStatus::quarantined,
                      false);
}

void cancel_after_claim() {
  sb::SharedRing ring = create_ring();
  Request request = begin_request(ring);
  publish(ring, request);
  sb::ProviderClaim provider = claim(ring, sb::RingStatus::ok);
  require_status(ring.host_cancel(request.ticket), sb::RingStatus::quarantined,
                 "host_cancel after claim");
  require(ring.slot_quarantined(request.ticket.slot),
          "post-claim cancellation did not quarantine slot");
  fill_terminal(provider, true);
  require_status(ring.provider_complete(provider, sb::CompletionCode::ok_all,
                                        sb::ErrorCode::none),
                 sb::RingStatus::ok, "provider_complete late success");
  consume_and_reclaim(ring, request.ticket, sb::RingStatus::quarantined,
                      false);
}

void stale_completion() {
  sb::SharedRing ring = create_ring();
  Request request = begin_request(ring);
  publish(ring, request);
  sb::ProviderClaim provider = claim(ring, sb::RingStatus::ok);
  fill_terminal(provider, true);
  require_status(ring.provider_complete(provider, sb::CompletionCode::ok_all,
                                        sb::ErrorCode::none),
                 sb::RingStatus::ok, "provider_complete before stale mutation");

  void* mapping = ::mmap(nullptr, sb::kMappingBytes, PROT_READ | PROT_WRITE,
                         MAP_SHARED, ring.fd(), 0);
  require(mapping != MAP_FAILED, "mmap for stale completion mutation failed");
  auto* completion = reinterpret_cast<sb::CompletionDescriptor*>(
      static_cast<std::uint8_t*>(mapping) + sb::kCompletionDescriptorOffset);
  ++completion[request.ticket.slot].provider_generation;
  require(::munmap(mapping, sb::kMappingBytes) == 0,
          "munmap after stale completion mutation failed");

  consume_and_reclaim(ring, request.ticket, sb::RingStatus::stale_completion,
                      false);
}

void retirement_and_replacement() {
  sb::SharedRing old_ring = create_ring(1, 1);
  Request request = begin_request(old_ring);
  publish(old_ring, request);
  static_cast<void>(claim(old_ring, sb::RingStatus::ok));
  require_status(old_ring.teardown_generation(1, 1), sb::RingStatus::ok,
                 "teardown_generation");

  sb::ConstCompletionPayload output{};
  sb::CompletionCode completion = sb::CompletionCode::unset;
  sb::ErrorCode error = sb::ErrorCode::internal;
  require_status(old_ring.host_consume(request.ticket, &output, &completion,
                                       &error),
                 sb::RingStatus::generation_dead,
                 "host_consume retired generation");
  require(!output.output_fp32,
          "retired generation exposed output from a claimed slot");
  require_status(old_ring.host_reclaim(request.ticket),
                 sb::RingStatus::generation_dead,
                 "host_reclaim retired generation");

  sb::SharedRing replacement = create_ring(2, 2);
  Request replacement_request = begin_request(replacement);
  publish(replacement, replacement_request);
  require(replacement.slot_state(0) == sb::SlotState::request_ready,
          "fresh generation did not accept sequence 1");
}

}  // namespace

int main() {
  try {
    success_path();
    cancel_before_claim();
    cancel_after_claim();
    stale_completion();
    retirement_and_replacement();
    std::cout
        << "{\"successAndSafeReuseTest\":\"matched\","
           "\"preclaimCancellationTombstoneTest\":\"matched\","
           "\"postclaimDeadlineQuarantineTest\":\"matched-cancel-variant\","
           "\"staleCompletionNeverExposesOutputTest\":\"matched\","
           "\"retirementRequiresFreshGenerationTest\":\"matched\"}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ring_correspondence: " << error.what() << '\n';
    return 1;
  }
}
