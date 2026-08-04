#include "shared_ring.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

#include <fcntl.h>
#include <linux/memfd.h>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace shooting_brake::phase2 {
namespace {

std::uint32_t load32(const std::uint32_t* word, int order) noexcept {
  return __atomic_load_n(word, order);
}
void store32(std::uint32_t* word, std::uint32_t value, int order) noexcept {
  __atomic_store_n(word, value, order);
}
bool cas32(std::uint32_t* word, std::uint32_t* expected, std::uint32_t desired,
           int success, int failure) noexcept {
  return __atomic_compare_exchange_n(word, expected, desired, false, success,
                                     failure);
}

bool zeroed(const void* data, std::size_t bytes) noexcept {
  const auto* p = static_cast<const std::uint8_t*>(data);
  for (std::size_t i = 0; i < bytes; ++i) {
    if (p[i] != 0U) return false;
  }
  return true;
}
bool equal(const Fingerprint& a, const Fingerprint& b) noexcept {
  return std::memcmp(a.bytes, b.bytes, sizeof(a.bytes)) == 0;
}
bool equal(const Nonce& a, const Nonce& b) noexcept {
  return std::memcmp(a.bytes, b.bytes, sizeof(a.bytes)) == 0;
}
bool valid_identity(const RingIdentity& id) noexcept {
  return id.ring_generation != 0U && id.provider_generation != 0U &&
         id.placement_generation != 0U && id.weight_generation != 0U;
}

std::uint8_t* base(void* mapping) noexcept {
  return static_cast<std::uint8_t*>(mapping);
}
const std::uint8_t* base(const void* mapping) noexcept {
  return static_cast<const std::uint8_t*>(mapping);
}
RingHeader* ring_header(void* mapping) noexcept {
  return static_cast<RingHeader*>(mapping);
}
const RingHeader* ring_header(const void* mapping) noexcept {
  return static_cast<const RingHeader*>(mapping);
}

std::uint64_t load64(const std::uint64_t* word, int order) noexcept {
  return __atomic_load_n(word, order);
}

bool generation_retired(const void* mapping) noexcept {
  return mapping != nullptr &&
         (load64(&ring_header(mapping)->flags, __ATOMIC_ACQUIRE) &
          kRingFlagDead) != 0;
}
SlotPublication* publication(void* mapping, std::uint32_t slot) noexcept {
  return reinterpret_cast<SlotPublication*>(base(mapping) +
                                             kSlotPublicationOffset) + slot;
}
RequestDescriptor* request_descriptor(void* mapping,
                                      std::uint32_t slot) noexcept {
  return reinterpret_cast<RequestDescriptor*>(base(mapping) +
                                               kRequestDescriptorOffset) + slot;
}
CompletionDescriptor* completion_descriptor(void* mapping,
                                            std::uint32_t slot) noexcept {
  return reinterpret_cast<CompletionDescriptor*>(base(mapping) +
                                                  kCompletionDescriptorOffset) + slot;
}
std::uint64_t payload_base(std::uint32_t slot) noexcept {
  return kPayloadOffset + static_cast<std::uint64_t>(slot) * kPayloadStride;
}

template <typename T>
T* at(void* mapping, std::uint64_t offset) noexcept {
  return reinterpret_cast<T*>(base(mapping) + offset);
}
template <typename T>
const T* at(const void* mapping, std::uint64_t offset) noexcept {
  return reinterpret_cast<const T*>(base(mapping) + offset);
}

std::uint32_t rotr(std::uint32_t value, std::uint32_t bits) noexcept {
  return (value >> bits) | (value << (32U - bits));
}
struct Sha256 final {
  std::uint32_t h[8] = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U,
                        0xa54ff53aU, 0x510e527fU, 0x9b05688cU,
                        0x1f83d9abU, 0x5be0cd19U};
  std::uint8_t block[64]{};
  std::uint32_t used = 0;
  std::uint64_t bytes = 0;
};
constexpr std::uint32_t kShaK[64] = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
void sha_block(Sha256* s) noexcept {
  std::uint32_t w[64]{};
  for (std::uint32_t i = 0; i < 16; ++i) {
    w[i] = (static_cast<std::uint32_t>(s->block[i * 4]) << 24U) |
           (static_cast<std::uint32_t>(s->block[i * 4 + 1]) << 16U) |
           (static_cast<std::uint32_t>(s->block[i * 4 + 2]) << 8U) |
           s->block[i * 4 + 3];
  }
  for (std::uint32_t i = 16; i < 64; ++i) {
    const std::uint32_t a =
        rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
    const std::uint32_t b =
        rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + a + w[i - 7] + b;
  }
  std::uint32_t a = s->h[0], b = s->h[1], c = s->h[2], d = s->h[3];
  std::uint32_t e = s->h[4], f = s->h[5], g = s->h[6], h = s->h[7];
  for (std::uint32_t i = 0; i < 64; ++i) {
    const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
    const std::uint32_t ch = (e & f) ^ ((~e) & g);
    const std::uint32_t t1 = h + s1 + ch + kShaK[i] + w[i];
    const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
    const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    const std::uint32_t t2 = s0 + maj;
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  s->h[0] += a;
  s->h[1] += b;
  s->h[2] += c;
  s->h[3] += d;
  s->h[4] += e;
  s->h[5] += f;
  s->h[6] += g;
  s->h[7] += h;
}
void sha_update(Sha256* s, const void* data, std::uint32_t count) noexcept {
  const auto* p = static_cast<const std::uint8_t*>(data);
  s->bytes += count;
  while (count != 0U) {
    const std::uint32_t take =
        std::min<std::uint32_t>(count, 64U - s->used);
    std::memcpy(s->block + s->used, p, take);
    s->used += take;
    p += take;
    count -= take;
    if (s->used == 64U) {
      sha_block(s);
      s->used = 0;
    }
  }
}
void sha_le32(Sha256* s, std::uint32_t value) noexcept {
  const std::uint8_t b[4] = {
      static_cast<std::uint8_t>(value),
      static_cast<std::uint8_t>(value >> 8U),
      static_cast<std::uint8_t>(value >> 16U),
      static_cast<std::uint8_t>(value >> 24U)};
  sha_update(s, b, 4);
}
void sha_le16(Sha256* s, std::uint16_t value) noexcept {
  const std::uint8_t b[2] = {static_cast<std::uint8_t>(value),
                             static_cast<std::uint8_t>(value >> 8U)};
  sha_update(s, b, 2);
}
Fingerprint sha_finish(Sha256* s) noexcept {
  const std::uint64_t bits = s->bytes * 8U;
  const std::uint8_t one = 0x80U;
  sha_update(s, &one, 1);
  const std::uint8_t zero = 0;
  while (s->used != 56U)
    sha_update(s, &zero, 1);
  std::uint8_t length[8]{};
  for (std::uint32_t i = 0; i < 8; ++i)
    length[7U - i] = static_cast<std::uint8_t>(bits >> (i * 8U));
  sha_update(s, length, 8);
  Fingerprint result{};
  for (std::uint32_t i = 0; i < 8; ++i) {
    result.bytes[i * 4] = static_cast<std::uint8_t>(s->h[i] >> 24U);
    result.bytes[i * 4 + 1] = static_cast<std::uint8_t>(s->h[i] >> 16U);
    result.bytes[i * 4 + 2] = static_cast<std::uint8_t>(s->h[i] >> 8U);
    result.bytes[i * 4 + 3] = static_cast<std::uint8_t>(s->h[i]);
  }
  return result;
}

Fingerprint route_hash(const RequestDescriptor& d,
                       const ConstRequestPayload& p) noexcept {
  Sha256 sha{};
  constexpr std::uint8_t prefix[9] = {'S', 'B', 'R', 'R', 'O',
                                      'U', 'T', 'E', '2'};
  sha_update(&sha, prefix, 9);
  sha_le32(&sha, d.layer);
  sha_le32(&sha, d.num_batch_tokens);
  sha_le32(&sha, d.num_staged_tokens);
  sha_le32(&sha, d.topk);
  sha_le32(&sha, d.num_routes);
  for (std::uint32_t r = 0; r < d.num_staged_tokens; ++r) {
    sha_le32(&sha, p.token_row_map[r]);
    for (std::uint32_t j = 0; j < kTopK; ++j) {
      const std::uint32_t x = r * kTopK + j;
      sha_update(&sha, &p.remote_mask[x], 1);
      sha_le16(&sha, p.canonical_route_positions[x]);
      if (p.remote_mask[x] != 0U) {
        sha_le32(&sha, static_cast<std::uint32_t>(p.canonical_ids[x]));
        std::uint32_t bits = 0;
        std::memcpy(&bits, &p.routing_weights[x], 4);
        sha_le32(&sha, bits);
      }
    }
  }
  return sha_finish(&sha);
}

RingStatus validate_header(const void* mapping, const RingIdentity& expected,
                           const char** detail) noexcept {
  const auto bad = [detail](RingStatus s, const char* m) {
    if (detail)
      *detail = m;
    return s;
  };
  const RingHeader& h = *ring_header(mapping);
  if (std::memcmp(h.magic, kRingMagic, 8) != 0 ||
      h.abi_major != kAbiMajor ||
      h.abi_minor != kAbiMinor)
    return bad(RingStatus::protocol_mismatch,
               "ring magic or ABI version mismatch");
  if (h.header_bytes != kHeaderBytes ||
      h.endian_tag != kEndianTag ||
      h.cache_line_bytes != 64 ||
      h.mapping_bytes != kMappingBytes ||
      h.num_slots != kRingSlots ||
      h.max_batch_tokens != kMaxBatchTokens ||
      h.max_staged_tokens != kMaxStagedTokens ||
      h.max_routes != kMaxRoutes ||
      h.hidden_size != kHiddenSize ||
      h.topk != kTopK)
    return bad(RingStatus::layout_mismatch,
               "ring fixed capacity/header field mismatch");
  if (h.slot_publication_offset != kSlotPublicationOffset ||
      h.slot_publication_stride != sizeof(SlotPublication) ||
      h.request_descriptor_offset != kRequestDescriptorOffset ||
      h.request_descriptor_stride != sizeof(RequestDescriptor) ||
      h.completion_descriptor_offset != kCompletionDescriptorOffset ||
      h.completion_descriptor_stride != sizeof(CompletionDescriptor) ||
      h.payload_offset != kPayloadOffset ||
      h.payload_stride != kPayloadStride)
    return bad(RingStatus::layout_mismatch,
               "ring table offset or stride mismatch");
  if (h.activation_relative_offset != kActivationRelativeOffset ||
      h.expert_ids_relative_offset != kExpertIdsRelativeOffset ||
      h.routing_weights_relative_offset != kRoutingWeightsRelativeOffset ||
      h.remote_mask_relative_offset != kRemoteMaskRelativeOffset ||
      h.token_row_map_relative_offset != kTokenRowMapRelativeOffset ||
      h.route_position_relative_offset != kRoutePositionRelativeOffset ||
      h.output_relative_offset != kOutputRelativeOffset ||
      h.route_status_relative_offset != kRouteStatusRelativeOffset ||
      h.token_status_relative_offset != kTokenStatusRelativeOffset)
    return bad(RingStatus::layout_mismatch,
               "ring payload relative offset mismatch");
  if (h.activation_capacity_bytes != kActivationBytes ||
      h.expert_ids_capacity_bytes != kExpertIdsBytes ||
      h.routing_weights_capacity_bytes != kRoutingWeightsBytes ||
      h.remote_mask_capacity_bytes != kRemoteMaskBytes ||
      h.token_row_map_capacity_bytes != kTokenRowMapBytes ||
      h.route_position_capacity_bytes != kRoutePositionBytes ||
      h.output_capacity_bytes != kOutputBytes ||
      h.route_status_capacity_bytes != kRouteStatusBytes ||
      h.token_status_capacity_bytes != kTokenStatusBytes ||
      (load64(&h.flags, __ATOMIC_ACQUIRE) & ~kKnownRingFlags) != 0 ||
      h.placement_manifest_offset != 0 ||
      h.placement_manifest_bytes != 0 ||
      !zeroed(h.reserved, sizeof(h.reserved)))
    return bad(RingStatus::layout_mismatch,
               "ring capacity, flags, manifest, or reserved bytes mismatch");
  if ((load64(&h.flags, __ATOMIC_ACQUIRE) & kRingFlagDead) != 0) {
    return bad(RingStatus::generation_dead,
               "ring generation is already retired");
  }
  if (h.ring_generation != expected.ring_generation ||
      h.provider_generation != expected.provider_generation ||
      h.placement_generation != expected.placement_generation ||
      h.weight_generation != expected.weight_generation ||
      !equal(h.provider_nonce, expected.provider_nonce) ||
      !equal(h.ring_nonce, expected.ring_nonce) ||
      !equal(h.placement_sha256, expected.placement_sha256) ||
      !equal(h.weight_sha256, expected.weight_sha256) ||
      h.provider_pid != expected.provider_pid)
    return bad(RingStatus::generation_mismatch,
               "ring/provider/placement/weight identity mismatch");
  return RingStatus::ok;
}

ConstRequestPayload const_payload(const void* mapping,
                                  const RequestDescriptor& d) noexcept {
  const std::uint32_t positions = d.num_staged_tokens * kTopK;
  return {{at<std::uint16_t>(mapping, d.activation_offset),
           d.num_staged_tokens * kHiddenSize},
          {at<std::int32_t>(mapping, d.expert_ids_offset), positions},
          {at<float>(mapping, d.routing_weights_offset), positions},
          {at<std::uint8_t>(mapping, d.remote_mask_offset), positions},
          {at<std::uint32_t>(mapping, d.token_row_map_offset),
           d.num_staged_tokens},
          {at<std::uint16_t>(mapping, d.route_position_offset), positions}};
}
RequestPayload mutable_payload(void* mapping,
                               const RequestDescriptor& d) noexcept {
  const std::uint32_t positions = d.num_staged_tokens * kTopK;
  return {{at<std::uint16_t>(mapping, d.activation_offset),
           d.num_staged_tokens * kHiddenSize},
          {at<std::int32_t>(mapping, d.expert_ids_offset), positions},
          {at<float>(mapping, d.routing_weights_offset), positions},
          {at<std::uint8_t>(mapping, d.remote_mask_offset), positions},
          {at<std::uint32_t>(mapping, d.token_row_map_offset),
           d.num_staged_tokens},
          {at<std::uint16_t>(mapping, d.route_position_offset), positions}};
}
CompletionPayload mutable_completion(void* mapping,
                                     const RequestDescriptor& d) noexcept {
  const std::uint64_t pb = payload_base(d.ring_slot);
  const std::uint32_t positions = d.num_staged_tokens * kTopK;
  return {{at<float>(mapping, pb + kOutputRelativeOffset),
           d.num_staged_tokens * kHiddenSize},
          {at<std::uint8_t>(mapping, pb + kRouteStatusRelativeOffset),
           positions},
          {at<std::uint8_t>(mapping, pb + kTokenStatusRelativeOffset),
           d.num_staged_tokens}};
}
ConstCompletionPayload const_completion(
    const void* mapping, const CompletionDescriptor& d) noexcept {
  return {{at<float>(mapping, d.output_offset), d.output_bytes / 4U},
          {at<std::uint8_t>(mapping, d.route_status_offset),
           d.route_status_bytes},
          {at<std::uint8_t>(mapping, d.token_status_offset),
           d.token_status_bytes}};
}

RingStatus validate_request_descriptor(const RequestDescriptor& d,
                                       const RingIdentity& id,
                                       std::uint32_t slot) noexcept {
  if (d.abi_major != kAbiMajor || d.abi_minor != kAbiMinor ||
      d.descriptor_bytes != sizeof(d))
    return RingStatus::protocol_mismatch;
  if (d.ring_slot != slot || slot >= kRingSlots || d.flags != 0 ||
      d.reserved0 != 0 || !zeroed(d.reserved, sizeof(d.reserved)))
    return RingStatus::invalid_descriptor;
  if (d.request_seq == 0 || d.ring_generation != id.ring_generation ||
      d.provider_generation != id.provider_generation ||
      d.placement_generation != id.placement_generation ||
      d.weight_generation != id.weight_generation ||
      d.request_generation == 0 || d.output_buffer_version == 0 ||
      !equal(d.placement_sha256, id.placement_sha256) ||
      !equal(d.weight_sha256, id.weight_sha256))
    return RingStatus::generation_mismatch;
  if (d.layer >= 32 || d.num_batch_tokens == 0 ||
      d.num_batch_tokens > kMaxBatchTokens || d.num_staged_tokens == 0 ||
      d.num_staged_tokens > kMaxStagedTokens ||
      d.num_staged_tokens > d.num_batch_tokens || d.num_routes == 0 ||
      d.num_routes > d.num_staged_tokens * kTopK ||
      d.hidden_size != kHiddenSize || d.topk != kTopK)
    return RingStatus::invalid_descriptor;
  if (d.activation_dtype != wire(DType::fp16) ||
      d.routing_weight_dtype != wire(DType::fp32) ||
      d.output_dtype != wire(DType::fp32))
    return RingStatus::invalid_descriptor;
  const std::uint64_t pb = payload_base(slot);
  const std::uint32_t rows = d.num_staged_tokens;
  const std::uint32_t pos = rows * kTopK;
  if (d.activation_offset != pb + kActivationRelativeOffset ||
      d.expert_ids_offset != pb + kExpertIdsRelativeOffset ||
      d.routing_weights_offset != pb + kRoutingWeightsRelativeOffset ||
      d.remote_mask_offset != pb + kRemoteMaskRelativeOffset ||
      d.token_row_map_offset != pb + kTokenRowMapRelativeOffset ||
      d.route_position_offset != pb + kRoutePositionRelativeOffset ||
      d.token_status_offset != pb + kTokenStatusRelativeOffset)
    return RingStatus::invalid_descriptor;
  if (d.activation_bytes != rows * kHiddenSize * 2U ||
      d.expert_ids_bytes != pos * 4U ||
      d.routing_weights_bytes != pos * 4U || d.remote_mask_bytes != pos ||
      d.token_row_map_bytes != rows * 4U ||
      d.route_position_bytes != pos * 2U || d.token_status_bytes != rows)
    return RingStatus::invalid_descriptor;
  if (pb > kMappingBytes || kPayloadStride > kMappingBytes - pb)
    return RingStatus::invalid_descriptor;
  return RingStatus::ok;
}
RingStatus validate_request_payload(const void* mapping,
                                    const RequestDescriptor& d) noexcept {
  const ConstRequestPayload p = const_payload(mapping, d);
  std::uint32_t count = 0, previous = 0;
  for (std::uint32_t r = 0; r < d.num_staged_tokens; ++r) {
    const std::uint32_t row = p.token_row_map[r];
    if (row >= d.num_batch_tokens || (r != 0 && row <= previous))
      return RingStatus::invalid_payload;
    previous = row;
    std::uint32_t row_count = 0;
    for (std::uint32_t j = 0; j < kTopK; ++j) {
      const std::uint32_t x = r * kTopK + j;
      const std::uint8_t mask = p.remote_mask[x];
      if (mask > 1)
        return RingStatus::invalid_payload;
      if (mask != 0) {
        if (p.canonical_route_positions[x] != j ||
            p.canonical_ids[x] < 0 || p.canonical_ids[x] >= 256 ||
            !std::isfinite(p.routing_weights[x]))
          return RingStatus::invalid_payload;
        ++count;
        ++row_count;
      } else if (p.canonical_route_positions[x] != 0xffffU)
        return RingStatus::invalid_payload;
    }
    if (row_count == 0)
      return RingStatus::invalid_payload;
  }
  if (count != d.num_routes)
    return RingStatus::invalid_payload;
  return equal(route_hash(d, p), d.route_subset_sha256)
             ? RingStatus::ok
             : RingStatus::invalid_payload;
}

bool ticket_matches(const SlotPublication& p, const RingTicket& t) noexcept {
  return p.request_seq == t.request_seq &&
         p.request_generation == t.request_generation &&
         p.output_buffer_version == t.output_buffer_version;
}
void signal_hint(int fd) noexcept {
  const std::uint64_t one = 1;
  const ssize_t ignored = ::write(fd, &one, sizeof(one));
  static_cast<void>(ignored);
}
RingStatus drain_hint(int fd) noexcept {
  if (fd < 0)
    return RingStatus::not_open;
  std::uint64_t value = 0;
  for (;;) {
    const ssize_t n = ::read(fd, &value, sizeof(value));
    if (n == sizeof(value))
      continue;
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
      return RingStatus::ok;
    return RingStatus::system_error;
  }
}

void fill_completion_identity(CompletionDescriptor* c,
                              const RequestDescriptor& r,
                              const RingIdentity& id, CompletionCode code,
                              ErrorCode error, std::uint64_t kernel,
                              std::uint64_t total,
                              std::uint64_t done) noexcept {
  std::memset(c, 0, sizeof(*c));
  c->abi_major = kAbiMajor;
  c->abi_minor = kAbiMinor;
  c->descriptor_bytes = sizeof(*c);
  c->request_seq = r.request_seq;
  c->ring_generation = r.ring_generation;
  c->request_generation = r.request_generation;
  c->provider_generation = r.provider_generation;
  c->placement_generation = r.placement_generation;
  c->weight_generation = r.weight_generation;
  c->scheduler_step = r.scheduler_step;
  c->deadline_monotonic_ns = r.deadline_monotonic_ns;
  c->ring_slot = r.ring_slot;
  c->layer = r.layer;
  c->num_batch_tokens = r.num_batch_tokens;
  c->num_staged_tokens = r.num_staged_tokens;
  c->num_routes = r.num_routes;
  c->hidden_size = r.hidden_size;
  c->topk = r.topk;
  c->activation_dtype = r.activation_dtype;
  c->routing_weight_dtype = r.routing_weight_dtype;
  c->output_dtype = r.output_dtype;
  c->completion_code = wire(code);
  c->error_code = wire(error);
  const std::uint64_t pb = payload_base(r.ring_slot);
  c->output_offset = pb + kOutputRelativeOffset;
  c->route_status_offset = pb + kRouteStatusRelativeOffset;
  c->token_status_offset = pb + kTokenStatusRelativeOffset;
  c->route_status_bytes = r.num_staged_tokens * kTopK;
  c->token_status_bytes = r.num_staged_tokens;
  c->output_bytes =
      (code == CompletionCode::ok_all || code == CompletionCode::ok_partial)
          ? r.num_staged_tokens * kHiddenSize * 4U
          : 0;
  c->latency_kernel_ns = kernel;
  c->latency_total_ns = total;
  c->completion_monotonic_ns = done;
  c->output_buffer_version = r.output_buffer_version;
  c->provider_nonce = id.provider_nonce;
  c->placement_sha256 = r.placement_sha256;
  c->route_subset_sha256 = r.route_subset_sha256;
  c->weight_sha256 = r.weight_sha256;
}
RingStatus validate_completion_descriptor(const CompletionDescriptor& c,
                                          const RequestDescriptor& r,
                                          const RingIdentity& id) noexcept {
  if (c.abi_major != kAbiMajor || c.abi_minor != kAbiMinor ||
      c.descriptor_bytes != sizeof(c) ||
      !zeroed(c.reserved, sizeof(c.reserved)))
    return RingStatus::stale_completion;
  if (c.request_seq != r.request_seq ||
      c.ring_generation != r.ring_generation ||
      c.request_generation != r.request_generation ||
      c.provider_generation != r.provider_generation ||
      c.placement_generation != r.placement_generation ||
      c.weight_generation != r.weight_generation ||
      c.scheduler_step != r.scheduler_step ||
      c.deadline_monotonic_ns != r.deadline_monotonic_ns ||
      c.ring_slot != r.ring_slot || c.layer != r.layer ||
      c.num_batch_tokens != r.num_batch_tokens ||
      c.num_staged_tokens != r.num_staged_tokens ||
      c.num_routes != r.num_routes || c.hidden_size != r.hidden_size ||
      c.topk != r.topk || c.activation_dtype != r.activation_dtype ||
      c.routing_weight_dtype != r.routing_weight_dtype ||
      c.output_dtype != r.output_dtype ||
      c.output_buffer_version != r.output_buffer_version ||
      !equal(c.provider_nonce, id.provider_nonce) ||
      !equal(c.placement_sha256, r.placement_sha256) ||
      !equal(c.route_subset_sha256, r.route_subset_sha256) ||
      !equal(c.weight_sha256, r.weight_sha256))
    return RingStatus::stale_completion;
  const std::uint64_t pb = payload_base(r.ring_slot);
  if (c.output_offset != pb + kOutputRelativeOffset ||
      c.route_status_offset != pb + kRouteStatusRelativeOffset ||
      c.token_status_offset != pb + kTokenStatusRelativeOffset ||
      c.route_status_bytes != r.num_staged_tokens * kTopK ||
      c.token_status_bytes != r.num_staged_tokens)
    return RingStatus::stale_completion;
  if (c.completion_code > wire(CompletionCode::protocol_error) ||
      c.completion_code == 0)
    return RingStatus::stale_completion;
  const bool committable =
      c.completion_code == wire(CompletionCode::ok_all) ||
      c.completion_code == wire(CompletionCode::ok_partial);
  if (c.output_bytes !=
      (committable ? r.num_staged_tokens * kHiddenSize * 4U : 0U))
    return RingStatus::stale_completion;
  return RingStatus::ok;
}
RingStatus validate_terminal(const void* mapping, const RequestDescriptor& r,
                             const CompletionDescriptor& c) noexcept {
  const auto in = const_payload(mapping, r);
  const auto out = const_completion(mapping, c);
  const CompletionCode code = static_cast<CompletionCode>(c.completion_code);
  if (code == CompletionCode::ok_partial)
    return RingStatus::invalid_payload;
  const std::uint8_t remote_expected =
      code == CompletionCode::ok_all
          ? wire(RouteStatus::contributed)
          : (code == CompletionCode::ambiguous
                 ? wire(RouteStatus::unknown)
                 : wire(RouteStatus::not_contributed));
  const std::uint8_t token_expected =
      code == CompletionCode::ok_all
          ? wire(TokenStatus::contributed)
          : (code == CompletionCode::ambiguous
                 ? wire(TokenStatus::unknown)
                 : wire(TokenStatus::not_contributed));
  for (std::uint32_t i = 0; i < out.route_status.count; ++i) {
    const std::uint8_t expected =
        in.remote_mask[i] ? remote_expected : wire(RouteStatus::not_remote);
    if (out.route_status[i] != expected)
      return RingStatus::invalid_payload;
  }
  for (std::uint32_t i = 0; i < out.token_status.count; ++i)
    if (out.token_status[i] != token_expected)
      return RingStatus::invalid_payload;
  if (code == CompletionCode::ok_all) {
    if (c.error_code != wire(ErrorCode::none))
      return RingStatus::invalid_payload;
    for (std::uint32_t i = 0; i < out.output_fp32.count; ++i)
      if (!std::isfinite(out.output_fp32[i]))
        return RingStatus::invalid_payload;
  } else if (c.error_code == wire(ErrorCode::none))
    return RingStatus::invalid_payload;
  return RingStatus::ok;
}

}  // namespace

const char* status_message(RingStatus s) noexcept {
  switch (s) {
    case RingStatus::ok:
      return "operation completed";
    case RingStatus::would_block:
      return "authoritative selected slot is not eligible";
    case RingStatus::invalid_argument:
      return "argument violates protocol-v2 contract";
    case RingStatus::not_open:
      return "ring mapping or notification fd is not open";
    case RingStatus::system_error:
      return "Linux memfd/eventfd/mmap/fd operation failed";
    case RingStatus::protocol_mismatch:
      return "SBRING02 ABI version mismatch";
    case RingStatus::layout_mismatch:
      return "4096-byte header or fixed layout mismatch";
    case RingStatus::generation_mismatch:
      return "ring/provider/placement/weight identity mismatch";
    case RingStatus::sequence_exhausted:
      return "sequence or buffer version exhausted; replace ring";
    case RingStatus::state_mismatch:
      return "ticket does not own the required slot state";
    case RingStatus::invalid_descriptor:
      return "descriptor exact offset, extent, shape, dtype, or reserved invariant failed";
    case RingStatus::invalid_payload:
      return "canonical route, mask, map, fingerprint, status, or finite-value invariant failed";
    case RingStatus::stale_completion:
      return "completion did not exactly echo immutable request identity";
    case RingStatus::cancelled:
      return "published request is a cancellation tombstone";
    case RingStatus::device_failure:
      return "provider output is not committable";
    case RingStatus::quarantined:
      return "post-claim lane is closed until completion or teardown";
    case RingStatus::deadline_expired:
      return "published request is a deadline tombstone";
    case RingStatus::generation_dead:
      return "old mapping generation is destructively retired";
  }
  return "unknown ring status";
}

SharedRing::SharedRing(int m, int r, int c, void* p,
                       const RingIdentity& id) noexcept
    : mapping_fd_(m),
      request_eventfd_(r),
      completion_eventfd_(c),
      mapping_(p),
      mapping_size_(kMappingBytes),
      identity_(id) {}
SharedRing::~SharedRing() noexcept {
  reset();
}
SharedRing::SharedRing(SharedRing&& o) noexcept
    : mapping_fd_(std::exchange(o.mapping_fd_, -1)),
      request_eventfd_(std::exchange(o.request_eventfd_, -1)),
      completion_eventfd_(std::exchange(o.completion_eventfd_, -1)),
      mapping_(std::exchange(o.mapping_, nullptr)),
      mapping_size_(std::exchange(o.mapping_size_, 0)),
      identity_(o.identity_),
      next_host_seq_(o.next_host_seq_),
      next_provider_seq_(o.next_provider_seq_),
      provider_busy_(o.provider_busy_),
      dead_(o.dead_) {
  o.provider_busy_ = false;
  o.dead_ = false;
}
SharedRing& SharedRing::operator=(SharedRing&& o) noexcept {
  if (this != &o) {
    reset();
    mapping_fd_ = std::exchange(o.mapping_fd_, -1);
    request_eventfd_ = std::exchange(o.request_eventfd_, -1);
    completion_eventfd_ = std::exchange(o.completion_eventfd_, -1);
    mapping_ = std::exchange(o.mapping_, nullptr);
    mapping_size_ = std::exchange(o.mapping_size_, 0);
    identity_ = o.identity_;
    next_host_seq_ = o.next_host_seq_;
    next_provider_seq_ = o.next_provider_seq_;
    provider_busy_ = o.provider_busy_;
    dead_ = o.dead_;
    o.provider_busy_ = false;
    o.dead_ = false;
  }
  return *this;
}
void SharedRing::reset() noexcept {
  if (mapping_)
    static_cast<void>(::munmap(mapping_, kMappingBytes));
  if (mapping_fd_ >= 0)
    static_cast<void>(::close(mapping_fd_));
  if (request_eventfd_ >= 0)
    static_cast<void>(::close(request_eventfd_));
  if (completion_eventfd_ >= 0)
    static_cast<void>(::close(completion_eventfd_));
  mapping_fd_ = request_eventfd_ = completion_eventfd_ = -1;
  mapping_ = nullptr;
  mapping_size_ = 0;
  identity_ = {};
  next_host_seq_ = next_provider_seq_ = 1;
  provider_busy_ = dead_ = false;
}

RingStatus SharedRing::visit_mapping(MappingVisitor visitor,
                                     void* context) noexcept {
  if (mapping_ == nullptr) {
    return RingStatus::not_open;
  }
  if (visitor == nullptr) {
    return RingStatus::invalid_argument;
  }
  return visitor(mapping_, mapping_size_, context);
}

RingStatus SharedRing::create(const RingIdentity& id, SharedRing* out,
                              const char** detail) noexcept {
  if (detail)
    *detail = nullptr;
  if (!out || out->is_open() || !valid_identity(id)) {
    if (detail)
      *detail =
          "create requires closed output and four nonzero generations";
    return RingStatus::invalid_argument;
  }
  const int m = static_cast<int>(::syscall(
      SYS_memfd_create, "shooting-brake-ring-v2",
      MFD_CLOEXEC | MFD_ALLOW_SEALING));
  if (m < 0) {
    if (detail)
      *detail = "memfd_create failed";
    return RingStatus::system_error;
  }
  if (::ftruncate(m, kMappingBytes) != 0 ||
      ::fcntl(m, F_ADD_SEALS, F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL) !=
          0) {
    ::close(m);
    if (detail)
      *detail = "memfd sizing or sealing failed";
    return RingStatus::system_error;
  }
  void* p =
      ::mmap(nullptr, kMappingBytes, PROT_READ | PROT_WRITE, MAP_SHARED, m, 0);
  if (p == MAP_FAILED) {
    ::close(m);
    if (detail)
      *detail = "MAP_SHARED mmap failed";
    return RingStatus::system_error;
  }
  const int r = ::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  const int c = ::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  if (r < 0 || c < 0) {
    if (r >= 0)
      ::close(r);
    if (c >= 0)
      ::close(c);
    ::munmap(p, kMappingBytes);
    ::close(m);
    if (detail)
      *detail = "nonblocking eventfd creation failed";
    return RingStatus::system_error;
  }
  std::memset(p, 0, kMappingBytes);
  RingHeader& h = *ring_header(p);
  std::memcpy(h.magic, kRingMagic, 8);
  h.abi_major = kAbiMajor;
  h.abi_minor = kAbiMinor;
  h.header_bytes = kHeaderBytes;
  h.endian_tag = kEndianTag;
  h.cache_line_bytes = 64;
  h.mapping_bytes = kMappingBytes;
  h.ring_generation = id.ring_generation;
  h.num_slots = kRingSlots;
  h.max_batch_tokens = kMaxBatchTokens;
  h.max_staged_tokens = kMaxStagedTokens;
  h.max_routes = kMaxRoutes;
  h.hidden_size = kHiddenSize;
  h.topk = kTopK;
  h.slot_publication_offset = kSlotPublicationOffset;
  h.slot_publication_stride = sizeof(SlotPublication);
  h.request_descriptor_offset = kRequestDescriptorOffset;
  h.request_descriptor_stride = sizeof(RequestDescriptor);
  h.completion_descriptor_offset = kCompletionDescriptorOffset;
  h.completion_descriptor_stride = sizeof(CompletionDescriptor);
  h.payload_offset = kPayloadOffset;
  h.payload_stride = kPayloadStride;
  h.activation_relative_offset = kActivationRelativeOffset;
  h.expert_ids_relative_offset = kExpertIdsRelativeOffset;
  h.routing_weights_relative_offset = kRoutingWeightsRelativeOffset;
  h.remote_mask_relative_offset = kRemoteMaskRelativeOffset;
  h.token_row_map_relative_offset = kTokenRowMapRelativeOffset;
  h.route_position_relative_offset = kRoutePositionRelativeOffset;
  h.output_relative_offset = kOutputRelativeOffset;
  h.route_status_relative_offset = kRouteStatusRelativeOffset;
  h.activation_capacity_bytes = kActivationBytes;
  h.expert_ids_capacity_bytes = kExpertIdsBytes;
  h.routing_weights_capacity_bytes = kRoutingWeightsBytes;
  h.remote_mask_capacity_bytes = kRemoteMaskBytes;
  h.token_row_map_capacity_bytes = kTokenRowMapBytes;
  h.route_position_capacity_bytes = kRoutePositionBytes;
  h.output_capacity_bytes = kOutputBytes;
  h.route_status_capacity_bytes = kRouteStatusBytes;
  h.provider_generation = id.provider_generation;
  h.placement_generation = id.placement_generation;
  h.weight_generation = id.weight_generation;
  h.provider_nonce = id.provider_nonce;
  h.ring_nonce = id.ring_nonce;
  h.placement_sha256 = id.placement_sha256;
  h.weight_sha256 = id.weight_sha256;
  h.provider_pid = id.provider_pid;
  h.token_status_relative_offset = kTokenStatusRelativeOffset;
  h.token_status_capacity_bytes = kTokenStatusBytes;
  for (std::uint32_t i = 0; i < kRingSlots; ++i)
    store32(&publication(p, i)->state, wire(SlotState::free),
            __ATOMIC_RELAXED);
  *out = SharedRing(m, r, c, p, id);
  return RingStatus::ok;
}
RingStatus SharedRing::attach(int mfd, int rfd, int cfd,
                              const RingIdentity& id, SharedRing* out,
                              const char** detail) noexcept {
  if (detail)
    *detail = nullptr;
  if (mfd < 0 || rfd < 0 || cfd < 0 || !out || out->is_open() ||
      !valid_identity(id))
    return RingStatus::invalid_argument;
  struct stat st{};
  if (::fstat(mfd, &st) != 0 ||
      st.st_size != static_cast<off_t>(kMappingBytes)) {
    if (detail)
      *detail = "mapping fd has wrong exact size";
    return RingStatus::layout_mismatch;
  }
  const int seals = ::fcntl(mfd, F_GET_SEALS);
  constexpr int required_seals =
      F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;
  if (seals < 0 || (seals & required_seals) != required_seals) {
    if (detail) {
      *detail = "mapping fd is not sealed against grow, shrink, and reseal";
    }
    return RingStatus::layout_mismatch;
  }
  const int m = ::dup(mfd), r = ::dup(rfd), c = ::dup(cfd);
  if (m < 0 || r < 0 || c < 0) {
    if (m >= 0)
      ::close(m);
    if (r >= 0)
      ::close(r);
    if (c >= 0)
      ::close(c);
    return RingStatus::system_error;
  }
  void* p =
      ::mmap(nullptr, kMappingBytes, PROT_READ | PROT_WRITE, MAP_SHARED, m, 0);
  if (p == MAP_FAILED) {
    ::close(m);
    ::close(r);
    ::close(c);
    return RingStatus::system_error;
  }
  const RingStatus v = validate_header(p, id, detail);
  if (v != RingStatus::ok) {
    ::munmap(p, kMappingBytes);
    ::close(m);
    ::close(r);
    ::close(c);
    return v;
  }
  *out = SharedRing(m, r, c, p, id);
  return RingStatus::ok;
}

RingStatus SharedRing::host_begin(const RequestSpec& s, RingTicket* ticket,
                                  RequestPayload* payload) noexcept {
  if (!mapping_)
    return RingStatus::not_open;
  if (dead_ || generation_retired(mapping_)) {
    return RingStatus::generation_dead;
  }
  if (!ticket || !payload || s.request_seq == 0 ||
      s.request_seq != next_host_seq_ || s.deadline_monotonic_ns == 0 ||
      s.layer >= 32 || s.num_batch_tokens == 0 ||
      s.num_batch_tokens > kMaxBatchTokens || s.num_staged_tokens == 0 ||
      s.num_staged_tokens > kMaxStagedTokens ||
      s.num_staged_tokens > s.num_batch_tokens || s.num_routes == 0 ||
      s.num_routes > s.num_staged_tokens * kTopK)
    return RingStatus::invalid_argument;
  // Only one unpublished reservation may exist. This lets the host abandon
  // HOST_WRITING without creating an invisible hole in the contiguous SPSC
  // request sequence; already-published slots may still fill the ring.
  for (std::uint32_t candidate = 0; candidate < kRingSlots; ++candidate) {
    if (load32(&publication(mapping_, candidate)->state, __ATOMIC_ACQUIRE) ==
        wire(SlotState::host_writing)) {
      return RingStatus::would_block;
    }
  }
  const std::uint32_t slot =
      static_cast<std::uint32_t>((s.request_seq - 1U) % kRingSlots);
  SlotPublication& pub = *publication(mapping_, slot);
  std::uint32_t expected = wire(SlotState::free);
  if (!cas32(&pub.state, &expected, wire(SlotState::host_writing),
             __ATOMIC_ACQUIRE, __ATOMIC_RELAXED))
    return RingStatus::would_block;
  const std::uint64_t rg = pub.request_generation + 1U,
                      ov = pub.output_buffer_version + 1U;
  if (rg == 0 || ov == 0 ||
      s.request_seq == std::numeric_limits<std::uint64_t>::max()) {
    store32(&pub.state, wire(SlotState::free), __ATOMIC_RELEASE);
    return RingStatus::sequence_exhausted;
  }
  std::memset(&pub.cancel_flags, 0, sizeof(pub.cancel_flags));
  std::memset(pub.reserved, 0, sizeof(pub.reserved));
  pub.request_seq = s.request_seq;
  pub.request_generation = rg;
  pub.output_buffer_version = ov;
  RequestDescriptor& d = *request_descriptor(mapping_, slot);
  std::memset(&d, 0, sizeof(d));
  d.abi_major = kAbiMajor;
  d.abi_minor = kAbiMinor;
  d.descriptor_bytes = sizeof(d);
  d.request_seq = s.request_seq;
  d.ring_generation = identity_.ring_generation;
  d.request_generation = rg;
  d.provider_generation = identity_.provider_generation;
  d.placement_generation = identity_.placement_generation;
  d.weight_generation = identity_.weight_generation;
  d.scheduler_step = s.scheduler_step;
  d.deadline_monotonic_ns = s.deadline_monotonic_ns;
  d.ring_slot = slot;
  d.layer = s.layer;
  d.num_batch_tokens = s.num_batch_tokens;
  d.num_staged_tokens = s.num_staged_tokens;
  d.num_routes = s.num_routes;
  d.hidden_size = kHiddenSize;
  d.topk = kTopK;
  d.activation_dtype = wire(DType::fp16);
  d.routing_weight_dtype = wire(DType::fp32);
  d.output_dtype = wire(DType::fp32);
  const std::uint64_t pb = payload_base(slot);
  d.activation_offset = pb + kActivationRelativeOffset;
  d.expert_ids_offset = pb + kExpertIdsRelativeOffset;
  d.routing_weights_offset = pb + kRoutingWeightsRelativeOffset;
  d.remote_mask_offset = pb + kRemoteMaskRelativeOffset;
  d.token_row_map_offset = pb + kTokenRowMapRelativeOffset;
  d.route_position_offset = pb + kRoutePositionRelativeOffset;
  d.output_buffer_version = ov;
  d.activation_bytes = s.num_staged_tokens * kHiddenSize * 2U;
  d.expert_ids_bytes = s.num_staged_tokens * kTopK * 4U;
  d.routing_weights_bytes = d.expert_ids_bytes;
  d.remote_mask_bytes = s.num_staged_tokens * kTopK;
  d.token_row_map_bytes = s.num_staged_tokens * 4U;
  d.route_position_bytes = s.num_staged_tokens * kTopK * 2U;
  d.placement_sha256 = identity_.placement_sha256;
  d.weight_sha256 = identity_.weight_sha256;
  d.token_status_offset = pb + kTokenStatusRelativeOffset;
  d.token_status_bytes = s.num_staged_tokens;
  const RingStatus descriptor_status =
      validate_request_descriptor(d, identity_, slot);
  if (descriptor_status != RingStatus::ok) {
    store32(&pub.state, wire(SlotState::free), __ATOMIC_RELEASE);
    return descriptor_status;
  }
  RequestPayload p = mutable_payload(mapping_, d);
  std::memset(p.activation_fp16.data, 0, p.activation_fp16.count * 2U);
  std::fill_n(p.canonical_ids.data, p.canonical_ids.count, -1);
  std::memset(p.routing_weights.data, 0, p.routing_weights.count * 4U);
  std::memset(p.remote_mask.data, 0, p.remote_mask.count);
  std::memset(p.token_row_map.data, 0, p.token_row_map.count * 4U);
  std::fill_n(p.canonical_route_positions.data,
              p.canonical_route_positions.count,
              static_cast<std::uint16_t>(0xffffU));
  CompletionPayload cp = mutable_completion(mapping_, d);
  std::memset(cp.output_fp32.data, 0, cp.output_fp32.count * 4U);
  std::memset(cp.route_status.data, 0, cp.route_status.count);
  std::memset(cp.token_status.data, 0, cp.token_status.count);
  *ticket = {s.request_seq, rg, ov, slot};
  *payload = p;
  ++next_host_seq_;
  return RingStatus::ok;
}
RingStatus SharedRing::host_publish(const RingTicket& t) noexcept {
  if (mapping_ == nullptr) {
    return RingStatus::not_open;
  }
  if (dead_ || generation_retired(mapping_)) {
    return RingStatus::generation_dead;
  }
  if (t.slot >= kRingSlots) {
    return RingStatus::invalid_argument;
  }
  SlotPublication& p = *publication(mapping_, t.slot);
  if (!ticket_matches(p, t) ||
      load32(&p.state, __ATOMIC_ACQUIRE) !=
          wire(SlotState::host_writing)) {
    return RingStatus::state_mismatch;
  }
  RequestDescriptor& d = *request_descriptor(mapping_, t.slot);
  RingStatus validation = validate_request_descriptor(d, identity_, t.slot);
  if (validation != RingStatus::ok) {
    return validation;
  }
  d.route_subset_sha256 = route_hash(d, const_payload(mapping_, d));
  validation = validate_request_payload(mapping_, d);
  if (validation != RingStatus::ok) {
    return validation;
  }
  store32(&p.state, wire(SlotState::request_ready), __ATOMIC_RELEASE);
  signal_hint(request_eventfd_);
  return RingStatus::ok;
}
RingStatus SharedRing::host_cancel(const RingTicket& t) noexcept {
  if (!mapping_)
    return RingStatus::not_open;
  if (dead_ || generation_retired(mapping_)) {
    return RingStatus::generation_dead;
  }
  if (t.slot >= kRingSlots)
    return RingStatus::invalid_argument;
  SlotPublication& p = *publication(mapping_, t.slot);
  if (!ticket_matches(p, t))
    return RingStatus::state_mismatch;
  const std::uint32_t state = load32(&p.state, __ATOMIC_ACQUIRE);
  if (state == wire(SlotState::host_writing)) {
    store32(&p.state, wire(SlotState::free), __ATOMIC_RELEASE);
    if (next_host_seq_ == t.request_seq + 1U)
      next_host_seq_ = t.request_seq;
    return RingStatus::cancelled;
  }
  if (state == wire(SlotState::request_ready)) {
    __atomic_fetch_or(&p.cancel_flags, kCancelRequested, __ATOMIC_RELEASE);
    signal_hint(request_eventfd_);
    return RingStatus::cancelled;
  }
  if (state == wire(SlotState::provider_running)) {
    __atomic_fetch_or(&p.cancel_flags, kCancelRequested, __ATOMIC_RELEASE);
    return RingStatus::quarantined;
  }
  if (state == wire(SlotState::response_ready)) {
    __atomic_fetch_or(&p.cancel_flags, kCancelRequested, __ATOMIC_RELEASE);
    return RingStatus::quarantined;
  }
  return RingStatus::state_mismatch;
}
RingStatus SharedRing::host_timeout(const RingTicket& t) noexcept {
  if (!mapping_)
    return RingStatus::not_open;
  if (dead_ || generation_retired(mapping_)) {
    return RingStatus::generation_dead;
  }
  if (t.slot >= kRingSlots)
    return RingStatus::invalid_argument;
  SlotPublication& p = *publication(mapping_, t.slot);
  if (!ticket_matches(p, t))
    return RingStatus::state_mismatch;
  const std::uint32_t state = load32(&p.state, __ATOMIC_ACQUIRE);
  if (state == wire(SlotState::host_writing)) {
    store32(&p.state, wire(SlotState::free), __ATOMIC_RELEASE);
    if (next_host_seq_ == t.request_seq + 1U)
      next_host_seq_ = t.request_seq;
    return RingStatus::deadline_expired;
  }
  if (state == wire(SlotState::request_ready)) {
    __atomic_fetch_or(&p.cancel_flags, kDeadlineClosed, __ATOMIC_RELEASE);
    signal_hint(request_eventfd_);
    return RingStatus::deadline_expired;
  }
  if (state == wire(SlotState::provider_running) ||
      state == wire(SlotState::response_ready)) {
    __atomic_fetch_or(&p.cancel_flags, kDeadlineClosed, __ATOMIC_RELEASE);
    return RingStatus::quarantined;
  }
  return RingStatus::state_mismatch;
}

RingStatus SharedRing::provider_claim(std::uint64_t now,
                                      ProviderClaim* claim) noexcept {
  if (claim)
    *claim = {};
  if (!mapping_)
    return RingStatus::not_open;
  if (dead_ || generation_retired(mapping_)) {
    return RingStatus::generation_dead;
  }
  if (!claim)
    return RingStatus::invalid_argument;
  if (provider_busy_)
    return RingStatus::would_block;
  const std::uint32_t slot =
      static_cast<std::uint32_t>((next_provider_seq_ - 1U) % kRingSlots);
  SlotPublication& p = *publication(mapping_, slot);
  if (load32(&p.state, __ATOMIC_ACQUIRE) != wire(SlotState::request_ready))
    return RingStatus::would_block;
  std::uint32_t expected = wire(SlotState::request_ready);
  if (!cas32(&p.state, &expected, wire(SlotState::provider_running),
             __ATOMIC_ACQUIRE, __ATOMIC_RELAXED))
    return RingStatus::would_block;
  claim->ticket = {p.request_seq, p.request_generation,
                   p.output_buffer_version, slot};
  claim->valid = true;
  provider_busy_ = true;
  const RequestDescriptor& d = *request_descriptor(mapping_, slot);
  RingStatus v = validate_request_descriptor(d, identity_, slot);
  if (v == RingStatus::ok &&
      (p.request_seq != next_provider_seq_ ||
       d.request_seq != next_provider_seq_ ||
       !ticket_matches(p, claim->ticket)))
    v = RingStatus::invalid_descriptor;
  if (v == RingStatus::ok)
    v = validate_request_payload(mapping_, d);
  if (v != RingStatus::ok)
    return v;
  claim->metadata = {d.scheduler_step, d.deadline_monotonic_ns, d.layer,
                     d.num_batch_tokens, d.num_staged_tokens, d.num_routes};
  claim->request = const_payload(mapping_, d);
  claim->completion = mutable_completion(mapping_, d);
  const std::uint32_t flags = load32(&p.cancel_flags, __ATOMIC_ACQUIRE);
  if ((flags & ~kKnownCancelFlags) != 0)
    return RingStatus::invalid_descriptor;
  if (flags & kCancelRequested)
    return RingStatus::cancelled;
  if ((flags & kDeadlineClosed) || now >= d.deadline_monotonic_ns)
    return RingStatus::deadline_expired;
  return RingStatus::ok;
}
RingStatus SharedRing::provider_complete(const ProviderClaim& claim,
                                         CompletionCode code, ErrorCode error,
                                         std::uint64_t kernel,
                                         std::uint64_t total,
                                         std::uint64_t done) noexcept {
  if (!mapping_)
    return RingStatus::not_open;
  if (dead_ || generation_retired(mapping_)) {
    return RingStatus::generation_dead;
  }
  if (!claim.valid || !provider_busy_ || claim.ticket.slot >= kRingSlots ||
      code == CompletionCode::unset || code == CompletionCode::ok_partial)
    return RingStatus::invalid_argument;
  SlotPublication& p = *publication(mapping_, claim.ticket.slot);
  if (!ticket_matches(p, claim.ticket) ||
      load32(&p.state, __ATOMIC_ACQUIRE) != wire(SlotState::provider_running))
    return RingStatus::state_mismatch;
  const RequestDescriptor& d =
      *request_descriptor(mapping_, claim.ticket.slot);
  RingStatus v = validate_request_descriptor(d, identity_, claim.ticket.slot);
  if (v == RingStatus::ok)
    v = validate_request_payload(mapping_, d);
  const bool malformed = v != RingStatus::ok;
  if (malformed && code != CompletionCode::protocol_error &&
      code != CompletionCode::rejected)
    return v;
  CompletionDescriptor& c =
      *completion_descriptor(mapping_, claim.ticket.slot);
  fill_completion_identity(&c, d, identity_, code, error, kernel, total, done);
  if (!malformed) {
    v = validate_terminal(mapping_, d, c);
    if (v != RingStatus::ok)
      return v;
  }
  store32(&p.state, wire(SlotState::response_ready), __ATOMIC_RELEASE);
  signal_hint(completion_eventfd_);
  provider_busy_ = false;
  if (next_provider_seq_ == std::numeric_limits<std::uint64_t>::max()) {
    static_cast<void>(__atomic_fetch_or(&ring_header(mapping_)->flags,
                                        kRingFlagDead, __ATOMIC_RELEASE));
    dead_ = true;
  } else {
    ++next_provider_seq_;
  }
  return RingStatus::ok;
}

RingStatus SharedRing::host_consume(const RingTicket& ticket,
                                    ConstCompletionPayload* payload,
                                    CompletionCode* code,
                                    ErrorCode* error) noexcept {
  if (payload != nullptr) {
    *payload = {};
  }
  if (mapping_ == nullptr) {
    return RingStatus::not_open;
  }
  if (dead_ || generation_retired(mapping_)) {
    return RingStatus::generation_dead;
  }
  if (ticket.slot >= kRingSlots || payload == nullptr || code == nullptr ||
      error == nullptr) {
    return RingStatus::invalid_argument;
  }
  SlotPublication& slot = *publication(mapping_, ticket.slot);
  if (!ticket_matches(slot, ticket)) {
    return RingStatus::state_mismatch;
  }
  std::uint32_t expected = wire(SlotState::response_ready);
  if (!cas32(&slot.state, &expected, wire(SlotState::host_reading),
             __ATOMIC_ACQUIRE, __ATOMIC_ACQUIRE)) {
    return RingStatus::would_block;
  }
  const RequestDescriptor& request =
      *request_descriptor(mapping_, ticket.slot);
  const CompletionDescriptor& completion =
      *completion_descriptor(mapping_, ticket.slot);
  RingStatus validation =
      validate_request_descriptor(request, identity_, ticket.slot);
  if (validation != RingStatus::ok) {
    return RingStatus::stale_completion;
  }
  validation =
      validate_completion_descriptor(completion, request, identity_);
  if (validation != RingStatus::ok) {
    return validation;
  }
  *code = static_cast<CompletionCode>(completion.completion_code);
  *error = static_cast<ErrorCode>(completion.error_code);
  validation = validate_terminal(mapping_, request, completion);
  if (validation != RingStatus::ok) {
    return validation;
  }
  const std::uint32_t flags =
      load32(&slot.cancel_flags, __ATOMIC_ACQUIRE);
  if (flags != 0) {
    return RingStatus::quarantined;
  }
  if (*code != CompletionCode::ok_all) {
    if (*code == CompletionCode::cancelled) {
      return RingStatus::cancelled;
    }
    if (*code == CompletionCode::deadline_exceeded) {
      return RingStatus::deadline_expired;
    }
    return RingStatus::device_failure;
  }
  *payload = const_completion(mapping_, completion);
  return RingStatus::ok;
}

RingStatus SharedRing::host_completion_timing(
    const RingTicket& ticket, CompletionTiming* timing) const noexcept {
  if (timing != nullptr) {
    *timing = {};
  }
  if (mapping_ == nullptr) {
    return RingStatus::not_open;
  }
  if (dead_ || generation_retired(mapping_)) {
    return RingStatus::generation_dead;
  }
  if (ticket.slot >= kRingSlots || timing == nullptr) {
    return RingStatus::invalid_argument;
  }
  const SlotPublication& slot = *publication(mapping_, ticket.slot);
  if (!ticket_matches(slot, ticket) ||
      load32(&slot.state, __ATOMIC_ACQUIRE) !=
          wire(SlotState::host_reading)) {
    return RingStatus::state_mismatch;
  }
  const RequestDescriptor& request =
      *request_descriptor(mapping_, ticket.slot);
  const CompletionDescriptor& completion =
      *completion_descriptor(mapping_, ticket.slot);
  RingStatus validation =
      validate_request_descriptor(request, identity_, ticket.slot);
  if (validation != RingStatus::ok) {
    return RingStatus::stale_completion;
  }
  validation =
      validate_completion_descriptor(completion, request, identity_);
  if (validation != RingStatus::ok) {
    return validation;
  }
  timing->kernel_ns = completion.latency_kernel_ns;
  timing->provider_total_ns = completion.latency_total_ns;
  timing->completion_monotonic_ns = completion.completion_monotonic_ns;
  return RingStatus::ok;
}

RingStatus SharedRing::host_reclaim(const RingTicket& ticket) noexcept {
  if (mapping_ == nullptr) {
    return RingStatus::not_open;
  }
  if (dead_ || generation_retired(mapping_)) {
    return RingStatus::generation_dead;
  }
  if (ticket.slot >= kRingSlots) {
    return RingStatus::invalid_argument;
  }
  SlotPublication& slot = *publication(mapping_, ticket.slot);
  if (!ticket_matches(slot, ticket) ||
      load32(&slot.state, __ATOMIC_ACQUIRE) !=
          wire(SlotState::host_reading)) {
    return RingStatus::state_mismatch;
  }
  store32(&slot.state, wire(SlotState::free), __ATOMIC_RELEASE);
  return RingStatus::ok;
}

RingStatus SharedRing::teardown_generation(std::uint64_t ring_generation,
                                           std::uint64_t provider_generation)
    noexcept {
  if (mapping_ == nullptr) {
    return RingStatus::not_open;
  }
  if (ring_generation != identity_.ring_generation ||
      provider_generation != identity_.provider_generation) {
    return RingStatus::generation_mismatch;
  }
  static_cast<void>(__atomic_fetch_or(&ring_header(mapping_)->flags,
                                      kRingFlagDead, __ATOMIC_RELEASE));
  dead_ = true;
  signal_hint(request_eventfd_);
  signal_hint(completion_eventfd_);
  return RingStatus::ok;
}

RingStatus SharedRing::drain_request_notifications() noexcept {
  if (mapping_ == nullptr) {
    return RingStatus::not_open;
  }
  if (dead_ || generation_retired(mapping_)) {
    return RingStatus::generation_dead;
  }
  return drain_hint(request_eventfd_);
}

RingStatus SharedRing::drain_completion_notifications() noexcept {
  if (mapping_ == nullptr) {
    return RingStatus::not_open;
  }
  if (dead_ || generation_retired(mapping_)) {
    return RingStatus::generation_dead;
  }
  return drain_hint(completion_eventfd_);
}

SlotState SharedRing::slot_state(std::uint32_t slot) const noexcept {
  if (mapping_ == nullptr || slot >= kRingSlots) {
    return SlotState::free;
  }
  return static_cast<SlotState>(
      load32(&publication(mapping_, slot)->state, __ATOMIC_ACQUIRE));
}

bool SharedRing::slot_quarantined(std::uint32_t slot) const noexcept {
  if (mapping_ == nullptr || slot >= kRingSlots) {
    return false;
  }
  const SlotPublication& selected = *publication(mapping_, slot);
  return load32(&selected.cancel_flags, __ATOMIC_ACQUIRE) != 0 &&
         load32(&selected.state, __ATOMIC_ACQUIRE) >=
             wire(SlotState::provider_running);
}

}  // namespace shooting_brake::phase2
