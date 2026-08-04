#pragma once

#include "shared_ring.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace shooting_brake::phase2::b70_control {

inline constexpr std::array<char, 8> kBootstrapMagic{
    'S', 'B', 'R', 'B', 'O', 'O', 'T', '2'};
inline constexpr std::array<char, 8> kStartupMagic{
    'S', 'B', 'R', 'R', 'E', 'A', 'D', '2'};
inline constexpr std::array<char, 8> kControlMagic{
    'S', 'B', 'R', 'C', 'T', 'R', 'L', '2'};
inline constexpr std::array<char, 8> kFaultAckMagic{
    'S', 'B', 'R', 'F', 'A', 'C', 'K', '2'};
inline constexpr std::array<char, 8> kShutdownMagic{
    'S', 'B', 'R', 'D', 'O', 'N', 'E', '2'};
inline constexpr std::uint16_t kProtocolMajor = 1;
inline constexpr std::uint16_t kProtocolMinor = 0;
inline constexpr std::uint32_t kNoProviderStatus =
    std::numeric_limits<std::uint32_t>::max();

enum class Command : std::uint32_t {
  shutdown = 1,
  arm_test_fault = 2,
};

enum class FaultPoint : std::uint32_t {
  none = 0,
  after_kernel_before_copyout = 1,
};

enum class FaultArgument : std::uint32_t {
  none = 0,
};

enum class FaultAckStatus : std::uint32_t {
  armed = 0,
  unsupported = 1,
  invalid_request = 2,
  provider_rejected = 3,
};

struct Bootstrap final {
  char magic[8];
  std::uint16_t major;
  std::uint16_t minor;
  std::uint32_t bytes;
  std::uint64_t mapping_bytes;
  RingIdentity identity;
  std::uint8_t reserved[32];
};

struct StartupReply final {
  char magic[8];
  std::uint16_t major;
  std::uint16_t minor;
  std::uint32_t bytes;
  std::uint32_t startup_status;
  std::uint32_t provider_status;
  std::uint64_t allocation_baseline;
  std::uint8_t reserved[32];
};

struct Control final {
  char magic[8];
  std::uint16_t major;
  std::uint16_t minor;
  std::uint32_t bytes;
  Command command;
  FaultPoint fault_point;
  FaultArgument fault_argument;
  std::uint32_t reserved0;
  std::uint64_t sequence;
  std::uint8_t reserved[16];
};

struct FaultAck final {
  char magic[8];
  std::uint16_t major;
  std::uint16_t minor;
  std::uint32_t bytes;
  FaultAckStatus status;
  std::uint32_t provider_status;
  FaultPoint fault_point;
  FaultArgument fault_argument;
  std::uint64_t sequence;
  std::uint8_t reserved[24];
};

struct ShutdownReply final {
  char magic[8];
  std::uint16_t major;
  std::uint16_t minor;
  std::uint32_t bytes;
  std::uint32_t status;
  std::uint32_t reserved0;
  std::uint64_t dispatches;
  std::uint64_t allocation_baseline;
  std::uint64_t allocation_final;
  std::uint8_t reserved[32];
};

static_assert(sizeof(Command) == sizeof(std::uint32_t));
static_assert(sizeof(FaultPoint) == sizeof(std::uint32_t));
static_assert(sizeof(FaultArgument) == sizeof(std::uint32_t));
static_assert(sizeof(FaultAckStatus) == sizeof(std::uint32_t));
static_assert(std::is_trivially_copyable_v<Bootstrap>);
static_assert(std::is_trivially_copyable_v<StartupReply>);
static_assert(std::is_trivially_copyable_v<Control>);
static_assert(std::is_trivially_copyable_v<FaultAck>);
static_assert(std::is_trivially_copyable_v<ShutdownReply>);
static_assert(sizeof(Bootstrap) == 192);
static_assert(sizeof(StartupReply) == 64);
static_assert(sizeof(Control) == 56);
static_assert(sizeof(FaultAck) == 64);
static_assert(sizeof(ShutdownReply) == 80);
static_assert(offsetof(Control, command) == 16);
static_assert(offsetof(Control, fault_point) == 20);
static_assert(offsetof(Control, fault_argument) == 24);
static_assert(offsetof(Control, sequence) == 32);

}  // namespace shooting_brake::phase2::b70_control
