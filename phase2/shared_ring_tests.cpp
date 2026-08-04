#include "shared_ring.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include <signal.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

namespace sb = shooting_brake::phase2;
namespace {

class Failure final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};
[[noreturn]] void fail(const std::string& text) { throw Failure(text); }
void require(bool value, const char* text) { if (!value) fail(text); }
void status(sb::RingStatus actual, sb::RingStatus expected, const char* where) {
  if (actual != expected)
    fail(std::string(where) + ": expected " + sb::status_message(expected) +
         ", got " + sb::status_message(actual));
}

sb::RingIdentity make_identity(std::uint64_t ring = 11,
                               std::uint64_t provider = 17) {
  sb::RingIdentity id{};
  id.ring_generation = ring;
  id.provider_generation = provider;
  id.placement_generation = 23;
  id.weight_generation = 29;
  id.provider_pid = static_cast<std::uint64_t>(::getpid());
  for (std::uint32_t i = 0; i < sb::kNonceBytes; ++i) {
    id.provider_nonce.bytes[i] = static_cast<std::uint8_t>(i * 7U + 1U);
    id.ring_nonce.bytes[i] = static_cast<std::uint8_t>(i * 11U + 3U);
  }
  for (std::uint32_t i = 0; i < sb::kFingerprintBytes; ++i) {
    id.placement_sha256.bytes[i] = static_cast<std::uint8_t>(i * 3U + 5U);
    id.weight_sha256.bytes[i] = static_cast<std::uint8_t>(i * 5U + 9U);
  }
  return id;
}

sb::SharedRing make_ring(const sb::RingIdentity& id = make_identity()) {
  sb::SharedRing ring;
  const char* detail = nullptr;
  const sb::RingStatus result = sb::SharedRing::create(id, &ring, &detail);
  if (result != sb::RingStatus::ok)
    fail(std::string("create: ") + sb::status_message(result) +
         (detail ? std::string(" (") + detail + ")" : ""));
  return ring;
}

struct RawMap final {
  void* p = nullptr;
  RawMap(const RawMap&) = delete;
  RawMap& operator=(const RawMap&) = delete;
  RawMap() = default;
  RawMap(RawMap&& other) noexcept : p(other.p) { other.p = nullptr; }
  RawMap& operator=(RawMap&&) = delete;
  ~RawMap() {
    if (p) {
      static_cast<void>(::munmap(p, sb::kMappingBytes));
    }
  }
};
RawMap raw_map(int fd) {
  RawMap map;
  map.p = ::mmap(nullptr, sb::kMappingBytes, PROT_READ | PROT_WRITE,
                 MAP_SHARED, fd, 0);
  if (map.p == MAP_FAILED) {
    map.p = nullptr;
    fail("raw mmap failed");
  }
  return map;
}
sb::RequestDescriptor* raw_request(void* p, std::uint32_t slot) {
  return reinterpret_cast<sb::RequestDescriptor*>(
             static_cast<std::uint8_t*>(p) + sb::kRequestDescriptorOffset) + slot;
}
sb::CompletionDescriptor* raw_completion(void* p, std::uint32_t slot) {
  return reinterpret_cast<sb::CompletionDescriptor*>(
             static_cast<std::uint8_t*>(p) + sb::kCompletionDescriptorOffset) + slot;
}

sb::RingTicket begin(sb::SharedRing& ring, std::uint64_t sequence,
                     std::uint32_t full_m, std::uint32_t staged_m,
                     sb::RequestPayload* payload,
                     std::uint64_t deadline =
                         std::numeric_limits<std::uint64_t>::max()) {
  sb::RequestSpec spec{};
  spec.request_seq = sequence;
  spec.scheduler_step = sequence * 13U;
  spec.deadline_monotonic_ns = deadline;
  spec.layer = static_cast<std::uint32_t>(sequence % 32U);
  spec.num_batch_tokens = full_m;
  spec.num_staged_tokens = staged_m;
  spec.num_routes = staged_m;
  sb::RingTicket ticket{};
  status(ring.host_begin(spec, &ticket, payload), sb::RingStatus::ok,
         "host_begin");
  require(ticket.slot == (sequence - 1U) % sb::kRingSlots,
          "request_seq did not select exact deterministic slot");
  return ticket;
}

void fill_request(sb::RequestPayload& p, std::uint32_t full_m,
                  std::uint32_t staged_m, std::uint32_t row_bias = 0) {
  for (std::uint32_t row = 0; row < staged_m; ++row) {
    const std::uint32_t original = row + row_bias;
    require(original < full_m, "test row map outside full M");
    p.token_row_map[row] = original;
    for (std::uint32_t h = 0; h < sb::kHiddenSize; ++h)
      p.activation_fp16[row * sb::kHiddenSize + h] =
          static_cast<std::uint16_t>((original + 1U) * 17U + h % 251U);
    const std::uint32_t x = row * sb::kTopK;
    p.remote_mask[x] = 1U;
    p.canonical_ids[x] = static_cast<std::int32_t>((original * 7U) % 256U);
    p.routing_weights[x] = 0.25F + row * 0.01F;
    p.canonical_route_positions[x] = 0U;
  }
}

float expected(const sb::ConstRequestPayload& p, std::uint32_t row,
               std::uint32_t hidden) {
  float sum = 0;
  for (std::uint32_t j = 0; j < sb::kTopK; ++j) {
    const std::uint32_t x = row * sb::kTopK + j;
    if (p.remote_mask[x])
      sum += p.routing_weights[x] * static_cast<float>(p.canonical_ids[x] + 1);
  }
  return sum + p.activation_fp16[row * sb::kHiddenSize + hidden] / 65536.0F;
}

void fill_status(sb::ProviderClaim& claim, sb::CompletionCode code) {
  const bool success = code == sb::CompletionCode::ok_all;
  const bool ambiguous = code == sb::CompletionCode::ambiguous;
  for (std::uint32_t i = 0; i < claim.completion.route_status.count; ++i) {
    claim.completion.route_status[i] =
        claim.request.remote_mask[i] == 0U
            ? sb::wire(sb::RouteStatus::not_remote)
            : (success ? sb::wire(sb::RouteStatus::contributed)
                       : (ambiguous ? sb::wire(sb::RouteStatus::unknown)
                                    : sb::wire(sb::RouteStatus::not_contributed)));
  }
  const std::uint8_t token =
      success ? sb::wire(sb::TokenStatus::contributed)
              : (ambiguous ? sb::wire(sb::TokenStatus::unknown)
                           : sb::wire(sb::TokenStatus::not_contributed));
  for (std::uint32_t i = 0; i < claim.completion.token_status.count; ++i)
    claim.completion.token_status[i] = token;
}

void execute(sb::ProviderClaim& claim) {
  for (std::uint32_t row = 0; row < claim.request.token_row_map.count; ++row)
    for (std::uint32_t h = 0; h < sb::kHiddenSize; ++h)
      claim.completion.output_fp32[row * sb::kHiddenSize + h] =
          expected(claim.request, row, h);
  fill_status(claim, sb::CompletionCode::ok_all);
}

void provider_once(sb::SharedRing& provider, std::uint64_t now = 1) {
  sb::ProviderClaim claim{};
  const sb::RingStatus result = provider.provider_claim(now, &claim);
  if (result == sb::RingStatus::ok) {
    execute(claim);
    status(provider.provider_complete(claim, sb::CompletionCode::ok_all,
                                      sb::ErrorCode::none),
           sb::RingStatus::ok, "complete OK_ALL");
  } else if (result == sb::RingStatus::cancelled) {
    fill_status(claim, sb::CompletionCode::cancelled);
    status(provider.provider_complete(claim, sb::CompletionCode::cancelled,
                                      sb::ErrorCode::cancelled),
           sb::RingStatus::ok, "complete CANCELLED");
  } else if (result == sb::RingStatus::deadline_expired) {
    fill_status(claim, sb::CompletionCode::deadline_exceeded);
    status(provider.provider_complete(claim,
                                      sb::CompletionCode::deadline_exceeded,
                                      sb::ErrorCode::deadline),
           sb::RingStatus::ok, "complete DEADLINE_EXCEEDED");
  } else {
    fail(std::string("provider_once: ") + sb::status_message(result));
  }
}

void consume_ok(sb::SharedRing& host, const sb::RingTicket& ticket,
                std::uint32_t staged_m) {
  sb::ConstCompletionPayload payload{};
  sb::CompletionCode code{};
  sb::ErrorCode error{};
  status(host.host_consume(ticket, &payload, &code, &error),
         sb::RingStatus::ok, "consume OK_ALL");
  require(code == sb::CompletionCode::ok_all && error == sb::ErrorCode::none,
          "successful completion code/error mismatch");
  require(payload.output_fp32.count == staged_m * sb::kHiddenSize,
          "successful output active extent mismatch");
  status(host.host_reclaim(ticket), sb::RingStatus::ok, "reclaim");
}

void test_layout() {
  require(sizeof(sb::RingHeader) == 4096 && alignof(sb::RingHeader) == 4096,
          "RingHeader ABI changed");
  require(sizeof(sb::SlotPublication) == 64, "publication ABI changed");
  require(sizeof(sb::RequestDescriptor) == 320 &&
              sizeof(sb::CompletionDescriptor) == 320,
          "descriptor ABI changed");
  require(sb::kSlotPublicationOffset +
                  sb::kRingSlots * sizeof(sb::SlotPublication) <=
              sb::kRequestDescriptorOffset,
          "publication and request tables overlap");
  require(sb::kCompletionDescriptorOffset +
                  sb::kRingSlots * sizeof(sb::CompletionDescriptor) <=
              sb::kPayloadOffset,
          "descriptor and payload tables overlap");
  require(sb::kActivationRelativeOffset + sb::kActivationBytes <=
              sb::kExpertIdsRelativeOffset,
          "activation/ID overlap");
  require(sb::kExpertIdsRelativeOffset + sb::kExpertIdsBytes <=
              sb::kRoutingWeightsRelativeOffset,
          "ID/weight overlap");
  require(sb::kRoutingWeightsRelativeOffset + sb::kRoutingWeightsBytes <=
              sb::kRemoteMaskRelativeOffset,
          "weight/mask overlap");
  require(sb::kRemoteMaskRelativeOffset + sb::kRemoteMaskBytes <=
              sb::kTokenRowMapRelativeOffset,
          "mask/token-map overlap");
  require(sb::kTokenRowMapRelativeOffset + sb::kTokenRowMapBytes <=
              sb::kRoutePositionRelativeOffset,
          "token/route-position overlap");
  require(sb::kRoutePositionRelativeOffset + sb::kRoutePositionBytes <=
              sb::kOutputRelativeOffset,
          "route-position/output overlap");
  require(sb::kOutputRelativeOffset + sb::kOutputBytes <=
              sb::kRouteStatusRelativeOffset,
          "output/route-status overlap");
  require(sb::kRouteStatusRelativeOffset + sb::kRouteStatusBytes <=
              sb::kTokenStatusRelativeOffset,
          "route/token-status overlap");
  require(sb::kPayloadOffset + sb::kRingSlots * sb::kPayloadStride ==
              sb::kMappingBytes,
          "eight disjoint slot payloads do not exactly fill mapping");
}

void test_states_visibility_and_notifications() {
  sb::SharedRing ring = make_ring();
  sb::RequestPayload p{};
  const sb::RingTicket ticket = begin(ring, 1, 4, 2, &p);
  require(ring.slot_state(ticket.slot) == sb::SlotState::host_writing,
          "missing HOST_WRITING");
  fill_request(p, 4, 2, 1);
  status(ring.host_publish(ticket), sb::RingStatus::ok, "publish");
  require(ring.slot_state(ticket.slot) == sb::SlotState::request_ready,
          "missing REQUEST_READY");
  status(ring.drain_request_notifications(), sb::RingStatus::ok,
         "drain request eventfd");
  sb::ProviderClaim claim{};
  status(ring.provider_claim(1, &claim), sb::RingStatus::ok, "claim");
  require(ring.slot_state(ticket.slot) == sb::SlotState::provider_running,
          "missing PROVIDER_RUNNING");
  require(claim.metadata.layer == 1U &&
              claim.metadata.num_batch_tokens == 4U &&
              claim.metadata.num_staged_tokens == 2U &&
              claim.metadata.num_routes == 2U,
          "validated provider metadata mismatch");
  require(claim.request.token_row_map[0] == 1U &&
              claim.request.token_row_map[1] == 2U &&
              claim.request.canonical_ids[0] == 7,
          "acquire did not expose request writes");
  execute(claim);
  status(ring.provider_complete(claim, sb::CompletionCode::ok_all,
                                sb::ErrorCode::none),
         sb::RingStatus::ok, "complete");
  require(ring.slot_state(ticket.slot) == sb::SlotState::response_ready,
          "missing RESPONSE_READY");
  status(ring.drain_completion_notifications(), sb::RingStatus::ok,
         "drain completion eventfd");
  consume_ok(ring, ticket, 2);
  require(ring.slot_state(ticket.slot) == sb::SlotState::free,
          "missing final FREE");
}

void test_invalid_payload_and_bounds() {
  sb::SharedRing ring = make_ring();
  sb::RequestSpec bad{};
  bad.request_seq = 1;
  bad.deadline_monotonic_ns = 1;
  bad.num_batch_tokens = 128;
  bad.num_staged_tokens = 129;
  bad.num_routes = 1;
  sb::RingTicket unused{};
  sb::RequestPayload p{};
  status(ring.host_begin(bad, &unused, &p), sb::RingStatus::invalid_argument,
         "reject M overflow");

  const sb::RingTicket ticket = begin(ring, 1, 2, 1, &p);
  fill_request(p, 2, 1);
  p.remote_mask[1] = 2;
  status(ring.host_publish(ticket), sb::RingStatus::invalid_payload,
         "reject mask");
  p.remote_mask[1] = 0;
  p.canonical_ids[0] = -1;
  status(ring.host_publish(ticket), sb::RingStatus::invalid_payload,
         "reject enabled ID");
  p.canonical_ids[0] = 0;
  p.token_row_map[0] = 2;
  status(ring.host_publish(ticket), sb::RingStatus::invalid_payload,
         "reject row map");
  p.token_row_map[0] = 0;
  p.canonical_route_positions[0] = 7;
  status(ring.host_publish(ticket), sb::RingStatus::invalid_payload,
         "reject canonical route position");
  p.canonical_route_positions[0] = 0;
  status(ring.host_publish(ticket), sb::RingStatus::ok,
         "publish restored request");

  RawMap raw = raw_map(ring.fd());
  raw_request(raw.p, ticket.slot)->activation_offset =
      std::numeric_limits<std::uint64_t>::max();
  sb::ProviderClaim claim{};
  status(ring.provider_claim(1, &claim), sb::RingStatus::invalid_descriptor,
         "reject exact offset/overflow");
  status(ring.provider_complete(claim, sb::CompletionCode::protocol_error,
                                sb::ErrorCode::bad_bounds),
         sb::RingStatus::ok, "close malformed claim");
  sb::ConstCompletionPayload output{};
  sb::CompletionCode code{};
  sb::ErrorCode error{};
  status(ring.host_consume(ticket, &output, &code, &error),
         sb::RingStatus::stale_completion,
         "malformed completion must not expose output");
  require(!output.output_fp32, "invalid descriptor exposed output");
  status(ring.host_reclaim(ticket), sb::RingStatus::ok,
         "reclaim invalid completion");
}

void test_backpressure_and_reclamation() {
  sb::SharedRing ring = make_ring();
  sb::RingTicket tickets[sb::kRingSlots]{};
  for (std::uint32_t i = 0; i < sb::kRingSlots; ++i) {
    sb::RequestPayload p{};
    tickets[i] = begin(ring, i + 1U, 1, 1, &p);
    fill_request(p, 1, 1);
    status(ring.host_publish(tickets[i]), sb::RingStatus::ok,
           "fill ring");
  }
  sb::RequestSpec next{};
  next.request_seq = 9;
  next.deadline_monotonic_ns = std::numeric_limits<std::uint64_t>::max();
  next.num_batch_tokens = next.num_staged_tokens = next.num_routes = 1;
  sb::RequestPayload p{};
  sb::RingTicket t9{};
  status(ring.host_begin(next, &t9, &p), sb::RingStatus::would_block,
         "full ring backpressure");
  provider_once(ring);
  sb::ConstCompletionPayload out{};
  sb::CompletionCode code{};
  sb::ErrorCode error{};
  status(ring.host_consume(tickets[0], &out, &code, &error),
         sb::RingStatus::ok, "consume first");
  status(ring.host_begin(next, &t9, &p), sb::RingStatus::would_block,
         "HOST_READING blocks reuse");
  status(ring.host_reclaim(tickets[0]), sb::RingStatus::ok, "free first");
  status(ring.host_begin(next, &t9, &p), sb::RingStatus::ok,
         "reuse only after reclaim");
  fill_request(p, 1, 1);
  status(ring.host_publish(t9), sb::RingStatus::ok, "publish reused");
  for (std::uint32_t i = 1; i < sb::kRingSlots; ++i) {
    provider_once(ring);
    consume_ok(ring, tickets[i], 1);
  }
  provider_once(ring);
  consume_ok(ring, t9, 1);
}

void test_cancel_timeout_quarantine() {
  sb::SharedRing ring = make_ring();
  sb::RequestPayload p{};
  sb::RingTicket cancel = begin(ring, 1, 1, 1, &p);
  fill_request(p, 1, 1);
  status(ring.host_publish(cancel), sb::RingStatus::ok, "publish cancel");
  status(ring.host_cancel(cancel), sb::RingStatus::cancelled,
         "preclaim cancel tombstone");
  require(ring.slot_state(cancel.slot) == sb::SlotState::request_ready,
          "published cancel created sequence hole");
  provider_once(ring);
  sb::ConstCompletionPayload out{};
  sb::CompletionCode code{};
  sb::ErrorCode error{};
  status(ring.host_consume(cancel, &out, &code, &error),
         sb::RingStatus::quarantined,
         "closed cancellation lane does not commit");
  status(ring.host_reclaim(cancel), sb::RingStatus::ok, "reclaim cancel");

  sb::RingTicket late = begin(ring, 2, 1, 1, &p);
  fill_request(p, 1, 1);
  status(ring.host_publish(late), sb::RingStatus::ok, "publish late");
  sb::ProviderClaim claim{};
  status(ring.provider_claim(1, &claim), sb::RingStatus::ok, "claim late");
  status(ring.host_timeout(late), sb::RingStatus::quarantined,
         "postclaim timeout quarantine");
  require(ring.slot_quarantined(late.slot), "timeout did not quarantine");
  execute(claim);
  status(ring.provider_complete(claim, sb::CompletionCode::ok_all,
                                sb::ErrorCode::none),
         sb::RingStatus::ok, "late complete");
  status(ring.host_consume(late, &out, &code, &error),

         sb::RingStatus::quarantined, "late output cannot commit");
  require(!out.output_fp32, "quarantined output exposed");
  status(ring.host_reclaim(late), sb::RingStatus::ok, "reclaim late");
}

void test_ambiguous_device_failure() {
  sb::SharedRing ring = make_ring();
  sb::RequestPayload request{};
  const sb::RingTicket ticket = begin(ring, 1, 2, 2, &request);
  fill_request(request, 2, 2);
  status(ring.host_publish(ticket), sb::RingStatus::ok,
         "publish ambiguous failure");
  sb::ProviderClaim claim{};
  status(ring.provider_claim(1, &claim), sb::RingStatus::ok,
         "claim ambiguous failure");
  fill_status(claim, sb::CompletionCode::ambiguous);
  status(ring.provider_complete(claim, sb::CompletionCode::ambiguous,
                                sb::ErrorCode::core_device),
         sb::RingStatus::ok, "complete ambiguous failure");
  sb::ConstCompletionPayload output{};
  sb::CompletionCode code{};
  sb::ErrorCode error{};
  status(ring.host_consume(ticket, &output, &code, &error),
         sb::RingStatus::device_failure,
         "ambiguous output is wholly uncommittable");
  require(!output.output_fp32, "ambiguous failure exposed output");
  require(code == sb::CompletionCode::ambiguous &&
              error == sb::ErrorCode::core_device,
          "ambiguous completion identity mismatch");
  status(ring.host_reclaim(ticket), sb::RingStatus::ok,
         "reclaim ambiguous failure");
}

void test_stale_completion() {
  sb::SharedRing ring = make_ring();
  sb::RequestPayload p{};
  const sb::RingTicket ticket = begin(ring, 1, 1, 1, &p);
  fill_request(p, 1, 1);
  status(ring.host_publish(ticket), sb::RingStatus::ok, "publish stale");
  provider_once(ring);
  RawMap raw = raw_map(ring.fd());
  raw_completion(raw.p, ticket.slot)->provider_generation++;
  sb::ConstCompletionPayload out{};
  sb::CompletionCode code{};
  sb::ErrorCode error{};
  status(ring.host_consume(ticket, &out, &code, &error),
         sb::RingStatus::stale_completion, "reject stale generation");
  require(!out.output_fp32, "stale completion exposed output");
  status(ring.host_reclaim(ticket), sb::RingStatus::ok, "reclaim stale");
}

void test_fork_delayed_writer() {
  const sb::RingIdentity id = make_identity();
  sb::SharedRing host = make_ring(id);
  sb::RequestPayload p{};
  const sb::RingTicket ticket = begin(host, 1, 8, 4, &p);
  fill_request(p, 8, 2);
  const pid_t child = ::fork();
  require(child >= 0, "fork failed");
  if (child == 0) {
    sb::SharedRing provider;
    if (sb::SharedRing::attach(host.fd(), host.request_eventfd(),
                               host.completion_eventfd(), id, &provider) !=
        sb::RingStatus::ok)
      _exit(20);
    for (std::uint32_t i = 0; i < 10000; ++i) {
      sb::ProviderClaim claim{};
      const sb::RingStatus result = provider.provider_claim(1, &claim);
      if (result == sb::RingStatus::ok) {
        if (claim.request.token_row_map[3] != 3U ||
            claim.request.canonical_ids[3U * sb::kTopK] != 21)
          _exit(21);
        execute(claim);
        _exit(provider.provider_complete(claim, sb::CompletionCode::ok_all,
                                         sb::ErrorCode::none) ==
                      sb::RingStatus::ok
                  ? 0 : 22);
      }
      if (result != sb::RingStatus::would_block) _exit(23);
      ::usleep(50);
    }
    _exit(24);
  }
  ::usleep(2000);
  fill_request(p, 8, 4);
  status(host.host_publish(ticket), sb::RingStatus::ok, "delayed publish");
  int ws = 0;
  require(::waitpid(child, &ws, 0) == child && WIFEXITED(ws) &&
              WEXITSTATUS(ws) == 0,
          "fork provider missed release/acquire writes");
  consume_ok(host, ticket, 4);
}

void test_child_kill_replacement() {
  const sb::RingIdentity old_id = make_identity(41, 43);
  sb::SharedRing old = make_ring(old_id);
  sb::RequestPayload p{};
  const sb::RingTicket ticket = begin(old, 1, 1, 1, &p);
  fill_request(p, 1, 1);
  status(old.host_publish(ticket), sb::RingStatus::ok, "publish kill");
  int pipefd[2]{};
  require(::pipe(pipefd) == 0, "pipe failed");
  const pid_t child = ::fork();
  require(child >= 0, "fork kill failed");
  if (child == 0) {
    ::close(pipefd[0]);
    sb::SharedRing provider;
    if (sb::SharedRing::attach(old.fd(), old.request_eventfd(),
                               old.completion_eventfd(), old_id, &provider) !=
        sb::RingStatus::ok)
      _exit(30);
    sb::ProviderClaim claim{};
    if (provider.provider_claim(1, &claim) != sb::RingStatus::ok) _exit(31);
    const char ready = 'R';
    if (::write(pipefd[1], &ready, 1) != 1) _exit(32);
    for (;;) ::pause();
  }
  ::close(pipefd[1]);
  char ready = 0;
  require(::read(pipefd[0], &ready, 1) == 1 && ready == 'R',
          "child did not claim");
  ::close(pipefd[0]);
  require(::kill(child, SIGKILL) == 0, "kill failed");
  int ws = 0;
  require(::waitpid(child, &ws, 0) == child && WIFSIGNALED(ws),
          "child was not killed");
  require(old.slot_state(ticket.slot) == sb::SlotState::provider_running,
          "provider death reused claimed slot");
  status(old.teardown_generation(old_id.ring_generation,
                                 old_id.provider_generation),
         sb::RingStatus::ok, "destructive teardown");
  status(old.host_reclaim(ticket), sb::RingStatus::generation_dead,
         "dead generation forbids reclamation");

  sb::SharedRing replacement = make_ring(make_identity(47, 53));
  sb::RequestPayload replacement_payload{};
  const sb::RingTicket replacement_ticket =
      begin(replacement, 1, 1, 1, &replacement_payload);
  fill_request(replacement_payload, 1, 1);
  status(replacement.host_publish(replacement_ticket), sb::RingStatus::ok,
         "replacement publish");
  provider_once(replacement);
  consume_ok(replacement, replacement_ticket, 1);
}

void test_cross_process_generation_retirement() {
  const sb::RingIdentity id = make_identity(59, 61);
  sb::SharedRing host = make_ring(id);
  int ready_pipe[2]{};
  int continue_pipe[2]{};
  require(::pipe(ready_pipe) == 0 && ::pipe(continue_pipe) == 0,
          "retirement pipes failed");
  const pid_t child = ::fork();
  require(child >= 0, "retirement fork failed");
  if (child == 0) {
    ::close(ready_pipe[0]);
    ::close(continue_pipe[1]);
    sb::SharedRing provider;
    if (sb::SharedRing::attach(host.fd(), host.request_eventfd(),
                               host.completion_eventfd(), id, &provider) !=
        sb::RingStatus::ok) {
      _exit(40);
    }
    const char ready = 'R';
    if (::write(ready_pipe[1], &ready, 1) != 1) {
      _exit(41);
    }
    char proceed = 0;
    if (::read(continue_pipe[0], &proceed, 1) != 1 || proceed != 'G') {
      _exit(42);
    }
    sb::ProviderClaim claim{};
    _exit(provider.provider_claim(1, &claim) ==
                  sb::RingStatus::generation_dead
              ? 0
              : 43);
  }
  ::close(ready_pipe[1]);
  ::close(continue_pipe[0]);
  char ready = 0;
  require(::read(ready_pipe[0], &ready, 1) == 1 && ready == 'R',
          "retirement child did not attach");
  ::close(ready_pipe[0]);
  status(host.teardown_generation(id.ring_generation, id.provider_generation),
         sb::RingStatus::ok, "publish shared retirement");
  const char proceed = 'G';
  require(::write(continue_pipe[1], &proceed, 1) == 1,
          "retirement release failed");
  ::close(continue_pipe[1]);
  int child_status = 0;
  require(::waitpid(child, &child_status, 0) == child &&
              WIFEXITED(child_status) && WEXITSTATUS(child_status) == 0,
          "attached peer did not acquire shared retirement");

  sb::SharedRing rejected_attach;
  status(sb::SharedRing::attach(host.fd(), host.request_eventfd(),
                                host.completion_eventfd(), id,
                                &rejected_attach),
         sb::RingStatus::generation_dead,
         "attach rejects retired generation");
  sb::RequestSpec spec{};
  spec.request_seq = 1;
  spec.deadline_monotonic_ns = 1;
  spec.num_batch_tokens = spec.num_staged_tokens = spec.num_routes = 1;
  sb::RingTicket ticket{};
  sb::RequestPayload payload{};
  status(host.host_begin(spec, &ticket, &payload),
         sb::RingStatus::generation_dead,
         "host rejects retired generation");
}

void test_wraparound() {
  sb::SharedRing ring = make_ring();
  const std::uint64_t iterations =
      std::getenv("SB_RING_LONG_STRESS") ? 2000000ULL : 50000ULL;
  for (std::uint64_t sequence = 1; sequence <= iterations; ++sequence) {
    sb::RequestPayload p{};
    const sb::RingTicket ticket = begin(ring, sequence, 1, 1, &p);
    fill_request(p, 1, 1);
    status(ring.host_publish(ticket), sb::RingStatus::ok, "stress publish");
    provider_once(ring);
    consume_ok(ring, ticket, 1);
  }
}

template <typename Function>
void run(const char* name, Function function) {
  function();
  std::cout << "PASS " << name << '\n';
}

}  // namespace

int main() {
  try {
    std::cout << "Phase-2 protocol verification only: deterministic CPU/fork "
                 "MAP_SHARED tests, not the later real CUDA/B70 DMA/provider "
                 "acceptance gate.\n";
    run("exact wire layout and disjoint slots", test_layout);
    run("state publication, visibility, eventfd hints",
        test_states_visibility_and_notifications);
    run("bounds, overflow, masks, IDs, maps", test_invalid_payload_and_bounds);
    run("full ring backpressure and safe reuse", test_backpressure_and_reclamation);
    run("cancellation, timeout, quarantine", test_cancel_timeout_quarantine);
    run("all-or-nothing ambiguous device failure",
        test_ambiguous_device_failure);
    run("stale completion rejection", test_stale_completion);
    run("delayed writer across fork", test_fork_delayed_writer);
    run("provider kill and generation replacement", test_child_kill_replacement);
    run("cross-process generation retirement",
        test_cross_process_generation_retirement);
    run("deterministic wraparound stress", test_wraparound);
    std::cout << "All Phase-2 protocol-only tests passed.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
  }
}
