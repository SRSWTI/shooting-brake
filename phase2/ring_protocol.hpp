#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace shooting_brake::phase2 {

#if !defined(__linux__) || !defined(__x86_64__) || \
    !defined(__BYTE_ORDER__) || (__BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__)
#error "Phase-2 ring ABI requires Linux x86_64 little-endian"
#endif

inline constexpr std::uint16_t kAbiMajor = 2;
inline constexpr std::uint16_t kAbiMinor = 0;
inline constexpr std::uint32_t kRingSlots = 8;
inline constexpr std::uint32_t kMaxBatchTokens = 128;
inline constexpr std::uint32_t kMaxStagedTokens = 128;
inline constexpr std::uint32_t kHiddenSize = 2048;
inline constexpr std::uint32_t kTopK = 8;
inline constexpr std::uint32_t kMaxRoutes = kMaxStagedTokens * kTopK;
inline constexpr std::uint32_t kEndianTag = 0x01020304U;
inline constexpr std::uint32_t kCacheLineBytes = 64;
inline constexpr std::uint32_t kHeaderBytes = 4096;
inline constexpr std::uint32_t kDescriptorBytes = 320;
inline constexpr std::uint8_t kRingMagic[8] = {'S', 'B', 'R', 'I', 'N', 'G', '0', '2'};
inline constexpr std::uint32_t kFingerprintBytes = 32;
inline constexpr std::uint32_t kNonceBytes = 16;

// Wire records contain fixed-width integers only; enums are API-side values.
enum class SlotState : std::uint32_t {
  free = 0,
  host_writing = 1,
  request_ready = 2,
  provider_running = 3,
  response_ready = 4,
  host_reading = 5,
};
enum class DType : std::uint16_t { fp32 = 0, fp16 = 1, bf16 = 2 };
enum class CompletionCode : std::uint16_t {
  unset = 0,
  ok_all = 1,
  ok_partial = 2,
  rejected = 3,
  deadline_exceeded = 4,
  cancelled = 5,
  execution_failed = 6,
  ambiguous = 7,
  provider_draining = 8,
  protocol_error = 9,
};
enum class ErrorCode : std::uint32_t {
  none = 0,
  bad_descriptor = 1,
  bad_generation = 2,
  bad_bounds = 3,
  bad_dtype = 4,
  bad_route = 5,
  bad_placement = 6,
  deadline = 7,
  cancelled = 8,
  core_busy = 9,
  core_not_loaded = 10,
  core_generation = 11,
  core_sequence = 12,
  core_device = 13,
  core_shutdown = 14,
  internal = 15,
};
enum class RouteStatus : std::uint8_t {
  not_remote = 0,
  contributed = 1,
  not_contributed = 2,
  unknown = 3,
};
enum class TokenStatus : std::uint8_t {
  not_remote = 0,
  contributed = 1,
  not_contributed = 2,
  unknown = 3,
};

inline constexpr std::uint32_t kCancelRequested = 1U;
inline constexpr std::uint32_t kDeadlineClosed = 2U;
inline constexpr std::uint32_t kKnownCancelFlags =
    kCancelRequested | kDeadlineClosed;
inline constexpr std::uint64_t kRingFlagDead = 1ULL;
inline constexpr std::uint64_t kKnownRingFlags = kRingFlagDead;

inline constexpr std::uint32_t wire(SlotState value) noexcept {
  return static_cast<std::uint32_t>(value);
}
inline constexpr std::uint16_t wire(DType value) noexcept {
  return static_cast<std::uint16_t>(value);
}
inline constexpr std::uint16_t wire(CompletionCode value) noexcept {
  return static_cast<std::uint16_t>(value);
}
inline constexpr std::uint32_t wire(ErrorCode value) noexcept {
  return static_cast<std::uint32_t>(value);
}
inline constexpr std::uint8_t wire(RouteStatus value) noexcept {
  return static_cast<std::uint8_t>(value);
}
inline constexpr std::uint8_t wire(TokenStatus value) noexcept {
  return static_cast<std::uint8_t>(value);
}

struct Fingerprint final { std::uint8_t bytes[32]; };
struct Nonce final { std::uint8_t bytes[16]; };

struct alignas(64) SlotPublication final {
  std::uint32_t state;
  std::uint32_t cancel_flags;
  std::uint64_t request_seq;
  std::uint64_t request_generation;
  std::uint64_t output_buffer_version;
  std::uint8_t reserved[32];
};

struct alignas(64) RequestDescriptor final {
  std::uint16_t abi_major;
  std::uint16_t abi_minor;
  std::uint32_t descriptor_bytes;
  std::uint64_t request_seq;
  std::uint64_t ring_generation;
  std::uint64_t request_generation;
  std::uint64_t provider_generation;
  std::uint64_t placement_generation;
  std::uint64_t weight_generation;
  std::uint64_t scheduler_step;
  std::uint64_t deadline_monotonic_ns;
  std::uint32_t ring_slot;
  std::uint32_t layer;
  std::uint32_t num_batch_tokens;
  std::uint32_t num_staged_tokens;
  std::uint32_t num_routes;
  std::uint32_t hidden_size;
  std::uint32_t topk;
  std::uint16_t activation_dtype;
  std::uint16_t routing_weight_dtype;
  std::uint16_t output_dtype;
  std::uint16_t flags;
  std::uint32_t reserved0;
  std::uint64_t activation_offset;
  std::uint64_t expert_ids_offset;
  std::uint64_t routing_weights_offset;
  std::uint64_t remote_mask_offset;
  std::uint64_t token_row_map_offset;
  std::uint64_t route_position_offset;
  std::uint64_t output_buffer_version;
  std::uint32_t activation_bytes;
  std::uint32_t expert_ids_bytes;
  std::uint32_t routing_weights_bytes;
  std::uint32_t remote_mask_bytes;
  std::uint32_t token_row_map_bytes;
  std::uint32_t route_position_bytes;
  Fingerprint placement_sha256;
  Fingerprint route_subset_sha256;
  Fingerprint weight_sha256;
  // Protocol-v2 Shooting Brake extension required by the fixed token-status
  // region. It occupies bytes that were reserved in the reviewed base record.
  std::uint64_t token_status_offset;
  std::uint32_t token_status_bytes;
  std::uint8_t reserved[20];
};

struct alignas(64) CompletionDescriptor final {
  std::uint16_t abi_major;
  std::uint16_t abi_minor;
  std::uint32_t descriptor_bytes;
  std::uint64_t request_seq;
  std::uint64_t ring_generation;
  std::uint64_t request_generation;
  std::uint64_t provider_generation;
  std::uint64_t placement_generation;
  std::uint64_t weight_generation;
  std::uint64_t scheduler_step;
  std::uint64_t deadline_monotonic_ns;
  std::uint32_t ring_slot;
  std::uint32_t layer;
  std::uint32_t num_batch_tokens;
  std::uint32_t num_staged_tokens;
  std::uint32_t num_routes;
  std::uint32_t hidden_size;
  std::uint32_t topk;
  std::uint16_t activation_dtype;
  std::uint16_t routing_weight_dtype;
  std::uint16_t output_dtype;
  std::uint16_t completion_code;
  std::uint32_t error_code;
  std::uint64_t output_offset;
  std::uint64_t route_status_offset;
  std::uint32_t output_bytes;
  std::uint32_t route_status_bytes;
  std::uint64_t latency_kernel_ns;
  std::uint64_t latency_total_ns;
  std::uint64_t completion_monotonic_ns;
  std::uint64_t output_buffer_version;
  Nonce provider_nonce;
  Fingerprint placement_sha256;
  Fingerprint route_subset_sha256;
  Fingerprint weight_sha256;
  std::uint64_t token_status_offset;
  std::uint32_t token_status_bytes;
  std::uint8_t reserved[28];
};

struct alignas(4096) RingHeader final {
  std::uint8_t magic[8];
  std::uint16_t abi_major;
  std::uint16_t abi_minor;
  std::uint32_t header_bytes;
  std::uint32_t endian_tag;
  std::uint32_t cache_line_bytes;
  std::uint64_t mapping_bytes;
  std::uint64_t ring_generation;
  std::uint32_t num_slots;
  std::uint32_t max_batch_tokens;
  std::uint32_t max_staged_tokens;
  std::uint32_t max_routes;
  std::uint32_t hidden_size;
  std::uint32_t topk;
  std::uint64_t slot_publication_offset;
  std::uint64_t slot_publication_stride;
  std::uint64_t request_descriptor_offset;
  std::uint64_t request_descriptor_stride;
  std::uint64_t completion_descriptor_offset;
  std::uint64_t completion_descriptor_stride;
  std::uint64_t payload_offset;
  std::uint64_t payload_stride;
  std::uint64_t activation_relative_offset;
  std::uint64_t expert_ids_relative_offset;
  std::uint64_t routing_weights_relative_offset;
  std::uint64_t remote_mask_relative_offset;
  std::uint64_t token_row_map_relative_offset;
  std::uint64_t route_position_relative_offset;
  std::uint64_t output_relative_offset;
  std::uint64_t route_status_relative_offset;
  std::uint64_t activation_capacity_bytes;
  std::uint64_t expert_ids_capacity_bytes;
  std::uint64_t routing_weights_capacity_bytes;
  std::uint64_t remote_mask_capacity_bytes;
  std::uint64_t token_row_map_capacity_bytes;
  std::uint64_t route_position_capacity_bytes;
  std::uint64_t output_capacity_bytes;
  std::uint64_t route_status_capacity_bytes;
  std::uint64_t provider_generation;
  std::uint64_t placement_generation;
  std::uint64_t weight_generation;
  Nonce provider_nonce;
  Nonce ring_nonce;
  Fingerprint placement_sha256;
  Fingerprint weight_sha256;
  std::uint64_t placement_manifest_offset;
  std::uint64_t placement_manifest_bytes;
  std::uint64_t provider_pid;
  std::uint64_t flags;
  std::uint64_t token_status_relative_offset;
  std::uint64_t token_status_capacity_bytes;
  std::uint8_t reserved[3672];
};

constexpr std::uint64_t align_up(std::uint64_t value,
                                 std::uint64_t alignment) noexcept {
  return (value + alignment - 1U) & ~(alignment - 1U);
}

inline constexpr std::uint64_t kSlotPublicationOffset = kHeaderBytes;
inline constexpr std::uint64_t kRequestDescriptorOffset =
    kSlotPublicationOffset + kRingSlots * sizeof(SlotPublication);
inline constexpr std::uint64_t kCompletionDescriptorOffset =
    kRequestDescriptorOffset + kRingSlots * sizeof(RequestDescriptor);
inline constexpr std::uint64_t kPayloadOffset = align_up(
    kCompletionDescriptorOffset + kRingSlots * sizeof(CompletionDescriptor),
    4096U);
inline constexpr std::uint64_t kActivationBytes =
    static_cast<std::uint64_t>(kMaxStagedTokens) * kHiddenSize * 2U;
inline constexpr std::uint64_t kExpertIdsBytes =
    static_cast<std::uint64_t>(kMaxRoutes) * 4U;
inline constexpr std::uint64_t kRoutingWeightsBytes = kExpertIdsBytes;
inline constexpr std::uint64_t kRemoteMaskBytes = kMaxRoutes;
inline constexpr std::uint64_t kTokenRowMapBytes = kMaxStagedTokens * 4U;
inline constexpr std::uint64_t kRoutePositionBytes = kMaxRoutes * 2U;
inline constexpr std::uint64_t kOutputBytes =
    static_cast<std::uint64_t>(kMaxStagedTokens) * kHiddenSize * 4U;
inline constexpr std::uint64_t kRouteStatusBytes = kMaxRoutes;
inline constexpr std::uint64_t kTokenStatusBytes = kMaxStagedTokens;
inline constexpr std::uint64_t kActivationRelativeOffset = 0;
inline constexpr std::uint64_t kExpertIdsRelativeOffset =
    align_up(kActivationRelativeOffset + kActivationBytes, 64U);
inline constexpr std::uint64_t kRoutingWeightsRelativeOffset =
    align_up(kExpertIdsRelativeOffset + kExpertIdsBytes, 64U);
inline constexpr std::uint64_t kRemoteMaskRelativeOffset =
    align_up(kRoutingWeightsRelativeOffset + kRoutingWeightsBytes, 64U);
inline constexpr std::uint64_t kTokenRowMapRelativeOffset =
    align_up(kRemoteMaskRelativeOffset + kRemoteMaskBytes, 64U);
inline constexpr std::uint64_t kRoutePositionRelativeOffset =
    align_up(kTokenRowMapRelativeOffset + kTokenRowMapBytes, 64U);
inline constexpr std::uint64_t kOutputRelativeOffset =
    align_up(kRoutePositionRelativeOffset + kRoutePositionBytes, 64U);
inline constexpr std::uint64_t kRouteStatusRelativeOffset =
    align_up(kOutputRelativeOffset + kOutputBytes, 64U);
inline constexpr std::uint64_t kTokenStatusRelativeOffset =
    align_up(kRouteStatusRelativeOffset + kRouteStatusBytes, 64U);
inline constexpr std::uint64_t kPayloadStride =
    align_up(kTokenStatusRelativeOffset + kTokenStatusBytes, 4096U);
inline constexpr std::uint64_t kMappingBytes =
    kPayloadOffset + kRingSlots * kPayloadStride;

static_assert(sizeof(Fingerprint) == 32);
static_assert(offsetof(Fingerprint, bytes) == 0);
static_assert(sizeof(Nonce) == 16);
static_assert(offsetof(Nonce, bytes) == 0);
static_assert(sizeof(SlotPublication) == 64 && alignof(SlotPublication) == 64);
static_assert(offsetof(SlotPublication, state) == 0);
static_assert(offsetof(SlotPublication, cancel_flags) == 4);
static_assert(offsetof(SlotPublication, request_seq) == 8);
static_assert(offsetof(SlotPublication, request_generation) == 16);
static_assert(offsetof(SlotPublication, output_buffer_version) == 24);
static_assert(offsetof(SlotPublication, reserved) == 32);

static_assert(sizeof(RequestDescriptor) == 320 &&
              alignof(RequestDescriptor) == 64);
static_assert(offsetof(RequestDescriptor, abi_major) == 0);
static_assert(offsetof(RequestDescriptor, abi_minor) == 2);
static_assert(offsetof(RequestDescriptor, descriptor_bytes) == 4);
static_assert(offsetof(RequestDescriptor, request_seq) == 8);
static_assert(offsetof(RequestDescriptor, ring_generation) == 16);
static_assert(offsetof(RequestDescriptor, request_generation) == 24);
static_assert(offsetof(RequestDescriptor, provider_generation) == 32);
static_assert(offsetof(RequestDescriptor, placement_generation) == 40);
static_assert(offsetof(RequestDescriptor, weight_generation) == 48);
static_assert(offsetof(RequestDescriptor, scheduler_step) == 56);
static_assert(offsetof(RequestDescriptor, deadline_monotonic_ns) == 64);
static_assert(offsetof(RequestDescriptor, ring_slot) == 72);
static_assert(offsetof(RequestDescriptor, layer) == 76);
static_assert(offsetof(RequestDescriptor, num_batch_tokens) == 80);
static_assert(offsetof(RequestDescriptor, num_staged_tokens) == 84);
static_assert(offsetof(RequestDescriptor, num_routes) == 88);
static_assert(offsetof(RequestDescriptor, hidden_size) == 92);
static_assert(offsetof(RequestDescriptor, topk) == 96);
static_assert(offsetof(RequestDescriptor, activation_dtype) == 100);
static_assert(offsetof(RequestDescriptor, routing_weight_dtype) == 102);
static_assert(offsetof(RequestDescriptor, output_dtype) == 104);
static_assert(offsetof(RequestDescriptor, flags) == 106);
static_assert(offsetof(RequestDescriptor, reserved0) == 108);
static_assert(offsetof(RequestDescriptor, activation_offset) == 112);
static_assert(offsetof(RequestDescriptor, expert_ids_offset) == 120);
static_assert(offsetof(RequestDescriptor, routing_weights_offset) == 128);
static_assert(offsetof(RequestDescriptor, remote_mask_offset) == 136);
static_assert(offsetof(RequestDescriptor, token_row_map_offset) == 144);
static_assert(offsetof(RequestDescriptor, route_position_offset) == 152);
static_assert(offsetof(RequestDescriptor, output_buffer_version) == 160);
static_assert(offsetof(RequestDescriptor, activation_bytes) == 168);
static_assert(offsetof(RequestDescriptor, expert_ids_bytes) == 172);
static_assert(offsetof(RequestDescriptor, routing_weights_bytes) == 176);
static_assert(offsetof(RequestDescriptor, remote_mask_bytes) == 180);
static_assert(offsetof(RequestDescriptor, token_row_map_bytes) == 184);
static_assert(offsetof(RequestDescriptor, route_position_bytes) == 188);
static_assert(offsetof(RequestDescriptor, placement_sha256) == 192);
static_assert(offsetof(RequestDescriptor, route_subset_sha256) == 224);
static_assert(offsetof(RequestDescriptor, weight_sha256) == 256);
static_assert(offsetof(RequestDescriptor, token_status_offset) == 288);
static_assert(offsetof(RequestDescriptor, token_status_bytes) == 296);
static_assert(offsetof(RequestDescriptor, reserved) == 300);

static_assert(sizeof(CompletionDescriptor) == 320 &&
              alignof(CompletionDescriptor) == 64);
static_assert(offsetof(CompletionDescriptor, abi_major) == 0);
static_assert(offsetof(CompletionDescriptor, abi_minor) == 2);
static_assert(offsetof(CompletionDescriptor, descriptor_bytes) == 4);
static_assert(offsetof(CompletionDescriptor, request_seq) == 8);
static_assert(offsetof(CompletionDescriptor, ring_generation) == 16);
static_assert(offsetof(CompletionDescriptor, request_generation) == 24);
static_assert(offsetof(CompletionDescriptor, provider_generation) == 32);
static_assert(offsetof(CompletionDescriptor, placement_generation) == 40);
static_assert(offsetof(CompletionDescriptor, weight_generation) == 48);
static_assert(offsetof(CompletionDescriptor, scheduler_step) == 56);
static_assert(offsetof(CompletionDescriptor, deadline_monotonic_ns) == 64);
static_assert(offsetof(CompletionDescriptor, ring_slot) == 72);
static_assert(offsetof(CompletionDescriptor, layer) == 76);
static_assert(offsetof(CompletionDescriptor, num_batch_tokens) == 80);
static_assert(offsetof(CompletionDescriptor, num_staged_tokens) == 84);
static_assert(offsetof(CompletionDescriptor, num_routes) == 88);
static_assert(offsetof(CompletionDescriptor, hidden_size) == 92);
static_assert(offsetof(CompletionDescriptor, topk) == 96);
static_assert(offsetof(CompletionDescriptor, activation_dtype) == 100);
static_assert(offsetof(CompletionDescriptor, routing_weight_dtype) == 102);
static_assert(offsetof(CompletionDescriptor, output_dtype) == 104);
static_assert(offsetof(CompletionDescriptor, completion_code) == 106);
static_assert(offsetof(CompletionDescriptor, error_code) == 108);
static_assert(offsetof(CompletionDescriptor, output_offset) == 112);
static_assert(offsetof(CompletionDescriptor, route_status_offset) == 120);
static_assert(offsetof(CompletionDescriptor, output_bytes) == 128);
static_assert(offsetof(CompletionDescriptor, route_status_bytes) == 132);
static_assert(offsetof(CompletionDescriptor, latency_kernel_ns) == 136);
static_assert(offsetof(CompletionDescriptor, latency_total_ns) == 144);
static_assert(offsetof(CompletionDescriptor, completion_monotonic_ns) == 152);
static_assert(offsetof(CompletionDescriptor, output_buffer_version) == 160);
static_assert(offsetof(CompletionDescriptor, provider_nonce) == 168);
static_assert(offsetof(CompletionDescriptor, placement_sha256) == 184);
static_assert(offsetof(CompletionDescriptor, route_subset_sha256) == 216);
static_assert(offsetof(CompletionDescriptor, weight_sha256) == 248);
static_assert(offsetof(CompletionDescriptor, token_status_offset) == 280);
static_assert(offsetof(CompletionDescriptor, token_status_bytes) == 288);
static_assert(offsetof(CompletionDescriptor, reserved) == 292);

static_assert(sizeof(RingHeader) == 4096 && alignof(RingHeader) == 4096);
static_assert(offsetof(RingHeader, magic) == 0);
static_assert(offsetof(RingHeader, abi_major) == 8);
static_assert(offsetof(RingHeader, abi_minor) == 10);
static_assert(offsetof(RingHeader, header_bytes) == 12);
static_assert(offsetof(RingHeader, endian_tag) == 16);
static_assert(offsetof(RingHeader, cache_line_bytes) == 20);
static_assert(offsetof(RingHeader, mapping_bytes) == 24);
static_assert(offsetof(RingHeader, ring_generation) == 32);
static_assert(offsetof(RingHeader, num_slots) == 40);
static_assert(offsetof(RingHeader, max_batch_tokens) == 44);
static_assert(offsetof(RingHeader, max_staged_tokens) == 48);
static_assert(offsetof(RingHeader, max_routes) == 52);
static_assert(offsetof(RingHeader, hidden_size) == 56);
static_assert(offsetof(RingHeader, topk) == 60);
static_assert(offsetof(RingHeader, slot_publication_offset) == 64);
static_assert(offsetof(RingHeader, slot_publication_stride) == 72);
static_assert(offsetof(RingHeader, request_descriptor_offset) == 80);
static_assert(offsetof(RingHeader, request_descriptor_stride) == 88);
static_assert(offsetof(RingHeader, completion_descriptor_offset) == 96);
static_assert(offsetof(RingHeader, completion_descriptor_stride) == 104);
static_assert(offsetof(RingHeader, payload_offset) == 112);
static_assert(offsetof(RingHeader, payload_stride) == 120);
static_assert(offsetof(RingHeader, activation_relative_offset) == 128);
static_assert(offsetof(RingHeader, expert_ids_relative_offset) == 136);
static_assert(offsetof(RingHeader, routing_weights_relative_offset) == 144);
static_assert(offsetof(RingHeader, remote_mask_relative_offset) == 152);
static_assert(offsetof(RingHeader, token_row_map_relative_offset) == 160);
static_assert(offsetof(RingHeader, route_position_relative_offset) == 168);
static_assert(offsetof(RingHeader, output_relative_offset) == 176);
static_assert(offsetof(RingHeader, route_status_relative_offset) == 184);
static_assert(offsetof(RingHeader, activation_capacity_bytes) == 192);
static_assert(offsetof(RingHeader, expert_ids_capacity_bytes) == 200);
static_assert(offsetof(RingHeader, routing_weights_capacity_bytes) == 208);
static_assert(offsetof(RingHeader, remote_mask_capacity_bytes) == 216);
static_assert(offsetof(RingHeader, token_row_map_capacity_bytes) == 224);
static_assert(offsetof(RingHeader, route_position_capacity_bytes) == 232);
static_assert(offsetof(RingHeader, output_capacity_bytes) == 240);
static_assert(offsetof(RingHeader, route_status_capacity_bytes) == 248);
static_assert(offsetof(RingHeader, provider_generation) == 256);
static_assert(offsetof(RingHeader, placement_generation) == 264);
static_assert(offsetof(RingHeader, weight_generation) == 272);
static_assert(offsetof(RingHeader, provider_nonce) == 280);
static_assert(offsetof(RingHeader, ring_nonce) == 296);
static_assert(offsetof(RingHeader, placement_sha256) == 312);
static_assert(offsetof(RingHeader, weight_sha256) == 344);
static_assert(offsetof(RingHeader, placement_manifest_offset) == 376);
static_assert(offsetof(RingHeader, placement_manifest_bytes) == 384);
static_assert(offsetof(RingHeader, provider_pid) == 392);
static_assert(offsetof(RingHeader, flags) == 400);
static_assert(offsetof(RingHeader, token_status_relative_offset) == 408);
static_assert(offsetof(RingHeader, token_status_capacity_bytes) == 416);
static_assert(offsetof(RingHeader, reserved) == 424);
static_assert(kPayloadOffset == 12288);
static_assert(kPayloadStride == 1589248);
static_assert(kMappingBytes == 12726272);
static_assert(std::is_standard_layout_v<RingHeader>);
static_assert(std::is_trivially_copyable_v<RequestDescriptor>);
static_assert(__atomic_always_lock_free(sizeof(std::uint32_t), nullptr),
              "cross-process state needs always-lock-free uint32 atomics");
static_assert(__atomic_always_lock_free(sizeof(std::uint64_t), nullptr),
              "cross-process retirement needs always-lock-free uint64 atomics");

}  // namespace shooting_brake::phase2
