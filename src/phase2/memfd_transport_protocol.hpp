#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace shooting_brake::phase2 {

inline constexpr std::uint64_t kProtocolMagic = 0x534250324d454d46ULL;  // SBP2MEMF
inline constexpr std::uint32_t kProtocolVersion = 1;
inline constexpr std::uint32_t kHeaderBytes = 4096;
inline constexpr std::uint64_t kPayloadCapacity = 1ULL << 20;
inline constexpr std::uint64_t kRequestOffset = kHeaderBytes;
inline constexpr std::uint64_t kResponseOffset =
    kRequestOffset + kPayloadCapacity;
inline constexpr std::uint64_t kMappingBytes =
    kResponseOffset + kPayloadCapacity;
inline constexpr std::uint32_t kBootstrapFdCount = 3;
inline constexpr std::uint32_t kCacheLineBytes = 64;

inline constexpr std::uint64_t kBootstrapMagic = 0x5342503246445331ULL;
inline constexpr std::uint64_t kBootstrapAckMagic = 0x5342503241434b31ULL;

inline constexpr std::uint32_t kRequiredTransferSizes[] = {
    4096U, 8192U, 524288U, 1048576U};

enum class RequestState : std::uint32_t {
  idle = 0,
  ready = 1,
  processing = 2,
  shutdown = 3,
};

enum class CompletionState : std::uint32_t {
  idle = 0,
  complete = 1,
  failed = 2,
  shutdown = 3,
};

enum class CompletionStatus : std::uint32_t {
  ok = 0,
  bad_layout = 1,
  bad_state = 2,
  bad_sequence = 3,
  bad_extent = 4,
  device_failure = 5,
};

struct alignas(kCacheLineBytes) HeaderPrefix {
  std::uint64_t magic;
  std::uint32_t version;
  std::uint32_t header_bytes;
  std::uint64_t mapping_bytes;
  std::uint64_t request_offset;
  std::uint64_t request_capacity;
  std::uint64_t response_offset;
  std::uint64_t response_capacity;
  std::uint32_t max_transfer_bytes;
  std::uint32_t flags;
};

// The publishing state is written last with release ordering and consumed first
// with acquire ordering. All other fields in the line belong to that publication.
struct alignas(kCacheLineBytes) RequestPublication {
  std::atomic<std::uint64_t> sequence;
  std::atomic<std::uint32_t> state;
  std::uint32_t payload_bytes;
  std::uint32_t pattern_seed;
  std::uint32_t flags;
  std::uint64_t iteration_tag;
  std::uint8_t reserved[32];
};

struct alignas(kCacheLineBytes) CompletionPublication {
  std::atomic<std::uint64_t> sequence;
  std::atomic<std::uint32_t> state;
  std::uint32_t status;
  std::uint32_t payload_bytes;
  std::uint32_t reserved0;
  std::uint64_t provider_h2d_ns;
  std::uint64_t provider_d2h_ns;
  std::uint64_t provider_total_ns;
  std::uint8_t reserved[16];
};

struct alignas(kCacheLineBytes) ProbeHeader {
  HeaderPrefix prefix;
  RequestPublication request;
  CompletionPublication completion;
  std::uint8_t reserved[kHeaderBytes - 3 * kCacheLineBytes];
};

struct BootstrapMessage {
  std::uint64_t magic;
  std::uint32_t version;
  std::uint32_t fd_count;
  std::uint64_t mapping_bytes;
};

struct BootstrapAck {
  std::uint64_t magic;
  std::uint32_t status;
  std::uint32_t reserved;
};

constexpr bool range_within(const std::uint64_t offset,
                            const std::uint64_t extent,
                            const std::uint64_t total) noexcept {
  return offset <= total && extent <= total - offset;
}

constexpr bool ranges_disjoint(const std::uint64_t first_offset,
                               const std::uint64_t first_extent,
                               const std::uint64_t second_offset,
                               const std::uint64_t second_extent) noexcept {
  return first_offset <= second_offset
             ? first_extent <= second_offset - first_offset
             : second_extent <= first_offset - second_offset;
}

inline bool valid_layout(const ProbeHeader& header,
                         const std::uint64_t mapped_bytes) noexcept {
  const HeaderPrefix& p = header.prefix;
  if (p.magic != kProtocolMagic || p.version != kProtocolVersion ||
      p.header_bytes != kHeaderBytes || p.mapping_bytes != mapped_bytes ||
      p.mapping_bytes != kMappingBytes || p.request_offset != kRequestOffset ||
      p.request_capacity != kPayloadCapacity ||
      p.response_offset != kResponseOffset ||
      p.response_capacity != kPayloadCapacity ||
      p.max_transfer_bytes != kPayloadCapacity || p.flags != 0) {
    return false;
  }
  if (p.request_offset < p.header_bytes ||
      p.response_offset < p.header_bytes ||
      !range_within(p.request_offset, p.request_capacity, p.mapping_bytes) ||
      !range_within(p.response_offset, p.response_capacity, p.mapping_bytes) ||
      !ranges_disjoint(p.request_offset, p.request_capacity,
                       p.response_offset, p.response_capacity)) {
    return false;
  }
  return true;
}

static_assert(std::atomic<std::uint64_t>::is_always_lock_free);
static_assert(std::atomic<std::uint32_t>::is_always_lock_free);
static_assert(sizeof(std::atomic<std::uint64_t>) == sizeof(std::uint64_t));
static_assert(sizeof(std::atomic<std::uint32_t>) == sizeof(std::uint32_t));
static_assert(std::is_standard_layout_v<HeaderPrefix>);
static_assert(std::is_standard_layout_v<RequestPublication>);
static_assert(std::is_standard_layout_v<CompletionPublication>);
static_assert(std::is_standard_layout_v<ProbeHeader>);
static_assert(sizeof(HeaderPrefix) == kCacheLineBytes);
static_assert(sizeof(RequestPublication) == kCacheLineBytes);
static_assert(sizeof(CompletionPublication) == kCacheLineBytes);
static_assert(sizeof(ProbeHeader) == kHeaderBytes);
static_assert(alignof(ProbeHeader) == kCacheLineBytes);
static_assert(offsetof(ProbeHeader, request) == kCacheLineBytes);
static_assert(offsetof(ProbeHeader, completion) == 2 * kCacheLineBytes);
static_assert(sizeof(BootstrapMessage) == 24);
static_assert(sizeof(BootstrapAck) == 16);

}  // namespace shooting_brake::phase2
