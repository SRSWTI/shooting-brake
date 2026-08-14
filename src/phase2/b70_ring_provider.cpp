#include "b70_ring_control.hpp"
#include "../phase1/b70_provider.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

namespace sb1 = shooting_brake::phase1;
namespace sb2 = shooting_brake::phase2;
namespace ctl = shooting_brake::phase2::b70_control;
namespace {

static_assert(sizeof(sycl::half) == sizeof(std::uint16_t));
constexpr int kWaitMs = 250;

class Fd final {
 public:
  Fd() noexcept = default;
  explicit Fd(int fd) noexcept : fd_(fd) {}
  ~Fd() noexcept {
    if (fd_ >= 0) {
      static_cast<void>(::close(fd_));
    }
  }
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
  Fd(Fd&& other) noexcept : fd_(other.release()) {}
  Fd& operator=(Fd&& other) noexcept {
    if (this != &other) {
      reset(other.release());
    }
    return *this;
  }
  int get() const noexcept { return fd_; }
  int release() noexcept {
    const int result = fd_;
    fd_ = -1;
    return result;
  }
  void reset(int fd = -1) noexcept {
    if (fd_ >= 0) {
      static_cast<void>(::close(fd_));
    }
    fd_ = fd;
  }

 private:
  int fd_ = -1;
};

class SocketPath final {
 public:
  explicit SocketPath(std::string path) : path_(std::move(path)) {}
  ~SocketPath() noexcept {
    if (!path_.empty()) {
      static_cast<void>(::unlink(path_.c_str()));
    }
  }
  const char* c_str() const noexcept { return path_.c_str(); }
  void unlink_now() noexcept {
    if (!path_.empty()) {
      static_cast<void>(::unlink(path_.c_str()));
      path_.clear();
    }
  }

 private:
  std::string path_;
};

struct Options final {
  std::string socket_path;
  std::string bank_path = "src/phase1/expert_bank.bin";
  std::uint64_t expected_placement_generation = 0;
  std::uint64_t expected_weight_generation = 0;
  sb2::Fingerprint expected_placement_sha256{};
  sb2::Fingerprint expected_weight_sha256{};
  bool quiet = false;
  std::vector<std::int32_t> resident_experts;
};

bool all_zero(const void* data, std::size_t bytes) noexcept {
  const auto* values = static_cast<const std::uint8_t*>(data);
  for (std::size_t index = 0; index < bytes; ++index) {
    if (values[index] != 0U) {
      return false;
    }
  }
  return true;
}

bool exact_magic(const char* actual, const std::array<char, 8>& expected) noexcept {
  return std::memcmp(actual, expected.data(), expected.size()) == 0;
}

void copy_magic(char* destination, const std::array<char, 8>& source) noexcept {
  std::memcpy(destination, source.data(), source.size());
}

std::uint64_t monotonic_ns() noexcept {
  timespec value{};
  if (::clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
    return 0;
  }
  return static_cast<std::uint64_t>(value.tv_sec) * 1'000'000'000ULL +
         static_cast<std::uint64_t>(value.tv_nsec);
}

bool parse_generation(std::string_view text, std::uint64_t* value) noexcept {
  if (text.empty()) {
    return false;
  }
  const char* const begin = text.data();
  const char* const end = begin + text.size();
  const auto result = std::from_chars(begin, end, *value, 10);
  return result.ec == std::errc{} && result.ptr == end && *value != 0U;
}

bool parse_resident_experts(
    std::string_view text,
    std::vector<std::int32_t>* resident_experts) {
  if (text.empty()) {
    return false;
  }
  std::array<bool, 256> seen{};
  resident_experts->clear();
  resident_experts->reserve(seen.size());
  std::size_t begin = 0;
  while (begin < text.size()) {
    const std::size_t end = text.find(',', begin);
    const std::size_t token_end =
        end == std::string_view::npos ? text.size() : end;
    if (token_end == begin) {
      return false;
    }
    const std::string_view token = text.substr(begin, token_end - begin);
    std::uint32_t canonical = 0;
    const auto parsed =
        std::from_chars(token.data(), token.data() + token.size(), canonical, 10);
    if (parsed.ec != std::errc{} ||
        parsed.ptr != token.data() + token.size() ||
        canonical >= seen.size() || seen[canonical]) {
      return false;
    }
    seen[canonical] = true;
    resident_experts->push_back(static_cast<std::int32_t>(canonical));
    if (end == std::string_view::npos) {
      break;
    }
    begin = end + 1U;
    if (begin == text.size()) {
      return false;
    }
  }
  return !resident_experts->empty();
}

int hex_nibble(char value) noexcept {
  if (value >= '0' && value <= '9') {
    return value - '0';
  }
  if (value >= 'a' && value <= 'f') {
    return value - 'a' + 10;
  }
  if (value >= 'A' && value <= 'F') {
    return value - 'A' + 10;
  }
  return -1;
}

bool parse_fingerprint(std::string_view text,
                       sb2::Fingerprint* fingerprint) noexcept {
  if (text.size() != 2U * sizeof(fingerprint->bytes)) {
    return false;
  }
  for (std::size_t index = 0; index < sizeof(fingerprint->bytes); ++index) {
    const int high = hex_nibble(text[index * 2U]);
    const int low = hex_nibble(text[index * 2U + 1U]);
    if (high < 0 || low < 0) {
      return false;
    }
    fingerprint->bytes[index] =
        static_cast<std::uint8_t>((high << 4) | low);
  }
  return true;
}

std::uint32_t rotate_right(std::uint32_t value, std::uint32_t bits) noexcept {
  return (value >> bits) | (value << (32U - bits));
}

struct Sha256 final {
  std::uint32_t state[8] = {
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
  std::uint8_t block[64]{};
  std::size_t used = 0;
  std::uint64_t bytes = 0;
};

constexpr std::uint32_t kSha256Constants[64] = {
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

void sha256_block(Sha256* sha) noexcept {
  std::uint32_t words[64]{};
  for (std::uint32_t index = 0; index < 16; ++index) {
    words[index] =
        (static_cast<std::uint32_t>(sha->block[index * 4U]) << 24U) |
        (static_cast<std::uint32_t>(sha->block[index * 4U + 1U]) << 16U) |
        (static_cast<std::uint32_t>(sha->block[index * 4U + 2U]) << 8U) |
        sha->block[index * 4U + 3U];
  }
  for (std::uint32_t index = 16; index < 64; ++index) {
    const std::uint32_t sigma0 =
        rotate_right(words[index - 15U], 7U) ^
        rotate_right(words[index - 15U], 18U) ^
        (words[index - 15U] >> 3U);
    const std::uint32_t sigma1 =
        rotate_right(words[index - 2U], 17U) ^
        rotate_right(words[index - 2U], 19U) ^
        (words[index - 2U] >> 10U);
    words[index] =
        words[index - 16U] + sigma0 + words[index - 7U] + sigma1;
  }

  std::uint32_t a = sha->state[0];
  std::uint32_t b = sha->state[1];
  std::uint32_t c = sha->state[2];
  std::uint32_t d = sha->state[3];
  std::uint32_t e = sha->state[4];
  std::uint32_t f = sha->state[5];
  std::uint32_t g = sha->state[6];
  std::uint32_t h = sha->state[7];
  for (std::uint32_t index = 0; index < 64; ++index) {
    const std::uint32_t sum1 =
        rotate_right(e, 6U) ^ rotate_right(e, 11U) ^ rotate_right(e, 25U);
    const std::uint32_t choose = (e & f) ^ ((~e) & g);
    const std::uint32_t temporary1 =
        h + sum1 + choose + kSha256Constants[index] + words[index];
    const std::uint32_t sum0 =
        rotate_right(a, 2U) ^ rotate_right(a, 13U) ^ rotate_right(a, 22U);
    const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
    const std::uint32_t temporary2 = sum0 + majority;
    h = g;
    g = f;
    f = e;
    e = d + temporary1;
    d = c;
    c = b;
    b = a;
    a = temporary1 + temporary2;
  }
  sha->state[0] += a;
  sha->state[1] += b;
  sha->state[2] += c;
  sha->state[3] += d;
  sha->state[4] += e;
  sha->state[5] += f;
  sha->state[6] += g;
  sha->state[7] += h;
}

void sha256_update(Sha256* sha, const void* data, std::size_t bytes) noexcept {
  const auto* source = static_cast<const std::uint8_t*>(data);
  sha->bytes += bytes;
  while (bytes != 0U) {
    const std::size_t chunk = std::min(bytes, sizeof(sha->block) - sha->used);
    std::memcpy(sha->block + sha->used, source, chunk);
    sha->used += chunk;
    source += chunk;
    bytes -= chunk;
    if (sha->used == sizeof(sha->block)) {
      sha256_block(sha);
      sha->used = 0;
    }
  }
}

void sha256_little_endian_u32(Sha256* sha, std::uint32_t value) noexcept {
  const std::uint8_t bytes[4] = {
      static_cast<std::uint8_t>(value),
      static_cast<std::uint8_t>(value >> 8U),
      static_cast<std::uint8_t>(value >> 16U),
      static_cast<std::uint8_t>(value >> 24U)};
  sha256_update(sha, bytes, sizeof(bytes));
}

sb2::Fingerprint sha256_finish(Sha256* sha) noexcept {
  const std::uint64_t bit_count = sha->bytes * 8U;
  const std::uint8_t one = 0x80U;
  sha256_update(sha, &one, 1U);
  const std::uint8_t zero = 0U;
  while (sha->used != 56U) {
    sha256_update(sha, &zero, 1U);
  }
  std::uint8_t length[8]{};
  for (std::uint32_t index = 0; index < 8U; ++index) {
    length[7U - index] =
        static_cast<std::uint8_t>(bit_count >> (index * 8U));
  }
  sha256_update(sha, length, sizeof(length));

  sb2::Fingerprint result{};
  for (std::uint32_t index = 0; index < 8U; ++index) {
    result.bytes[index * 4U] =
        static_cast<std::uint8_t>(sha->state[index] >> 24U);
    result.bytes[index * 4U + 1U] =
        static_cast<std::uint8_t>(sha->state[index] >> 16U);
    result.bytes[index * 4U + 2U] =
        static_cast<std::uint8_t>(sha->state[index] >> 8U);
    result.bytes[index * 4U + 3U] =
        static_cast<std::uint8_t>(sha->state[index]);
  }
  return result;
}

sb2::Fingerprint placement_fingerprint(
    const std::vector<std::int32_t>& resident_experts) noexcept {
  constexpr std::uint8_t kPlacementMagic[8] = {
      'S', 'B', 'P', 'L', 'A', 'C', '0', '1'};
  const std::uint32_t resident_count =
      resident_experts.empty()
          ? 256U
          : static_cast<std::uint32_t>(resident_experts.size());
  Sha256 sha{};
  sha256_update(&sha, kPlacementMagic, sizeof(kPlacementMagic));
  sha256_little_endian_u32(&sha, resident_count);
  for (std::uint32_t local = 0; local < resident_count; ++local) {
    const std::uint32_t canonical =
        resident_experts.empty()
            ? local
            : static_cast<std::uint32_t>(resident_experts[local]);
    sha256_little_endian_u32(&sha, canonical);
  }
  return sha256_finish(&sha);
}

bool bank_fingerprint(int fd, sb2::Fingerprint* fingerprint) noexcept {
  Sha256 sha{};
  std::array<std::uint8_t, 64U * 1024U> buffer{};
  for (;;) {
    const ssize_t bytes = ::read(fd, buffer.data(), buffer.size());
    if (bytes > 0) {
      sha256_update(&sha, buffer.data(), static_cast<std::size_t>(bytes));
      continue;
    }
    if (bytes == 0) {
      *fingerprint = sha256_finish(&sha);
      return true;
    }
    if (errno != EINTR) {
      return false;
    }
  }
}

using ExpertMap = std::array<std::int32_t, 256>;

ExpertMap make_expert_map(
    const std::vector<std::int32_t>& resident_experts) noexcept {
  ExpertMap map{};
  map.fill(-1);
  if (resident_experts.empty()) {
    for (std::size_t canonical = 0; canonical < map.size(); ++canonical) {
      map[canonical] = static_cast<std::int32_t>(canonical);
    }
  } else {
    for (std::size_t local = 0; local < resident_experts.size(); ++local) {
      map[static_cast<std::size_t>(resident_experts[local])] =
          static_cast<std::int32_t>(local);
    }
  }
  return map;
}

bool same_fingerprint(const sb2::Fingerprint& left,
                      const sb2::Fingerprint& right) noexcept {
  return std::memcmp(left.bytes, right.bytes, sizeof(left.bytes)) == 0;
}

bool matches_trusted_identity(const Options& options,
                              const sb2::RingIdentity& identity) noexcept {
  return identity.placement_generation ==
             options.expected_placement_generation &&
         identity.weight_generation == options.expected_weight_generation &&
         same_fingerprint(identity.placement_sha256,
                          options.expected_placement_sha256) &&
         same_fingerprint(identity.weight_sha256,
                          options.expected_weight_sha256);
}

bool parse_options(int argc, char** argv, Options* options) {
  bool saw_socket = false;
  bool saw_bank = false;
  bool saw_placement_generation = false;
  bool saw_weight_generation = false;
  bool saw_placement_sha256 = false;
  bool saw_weight_sha256 = false;
  bool saw_quiet = false;
  bool saw_resident_experts = false;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--quiet") {
      if (saw_quiet) {
        return false;
      }
      saw_quiet = true;
      options->quiet = true;
      continue;
    }
    if (argument != "--socket" && argument != "--bank" &&
        argument != "--resident-experts" &&
        argument != "--expected-placement-generation" &&
        argument != "--expected-weight-generation" &&
        argument != "--expected-placement-sha256" &&
        argument != "--expected-weight-sha256") {
      return false;
    }
    if (index + 1 >= argc || argv[index + 1][0] == '\0') {
      return false;
    }
    const std::string_view value(argv[++index]);
    if (argument == "--socket") {
      if (saw_socket) {
        return false;
      }
      saw_socket = true;
      options->socket_path = value;
    } else if (argument == "--bank") {
      if (saw_bank) {
        return false;
      }
      saw_bank = true;
      options->bank_path = value;
    } else if (argument == "--resident-experts") {
      if (saw_resident_experts ||
          !parse_resident_experts(value, &options->resident_experts)) {
        return false;
      }
      saw_resident_experts = true;
    } else if (argument == "--expected-placement-generation") {
      if (saw_placement_generation ||
          !parse_generation(value, &options->expected_placement_generation)) {
        return false;
      }
      saw_placement_generation = true;
    } else if (argument == "--expected-weight-generation") {
      if (saw_weight_generation ||
          !parse_generation(value, &options->expected_weight_generation)) {
        return false;
      }
      saw_weight_generation = true;
    } else if (argument == "--expected-placement-sha256") {
      if (saw_placement_sha256 ||
          !parse_fingerprint(value, &options->expected_placement_sha256)) {
        return false;
      }
      saw_placement_sha256 = true;
    } else {
      if (saw_weight_sha256 ||
          !parse_fingerprint(value, &options->expected_weight_sha256)) {
        return false;
      }
      saw_weight_sha256 = true;
    }
  }
  return saw_socket && saw_placement_generation && saw_weight_generation &&
         saw_placement_sha256 && saw_weight_sha256 &&
         options->socket_path.size() < sizeof(sockaddr_un::sun_path);
}

Fd make_listener(const char* path) noexcept {
  Fd socket(::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0));
  if (socket.get() < 0) {
    return {};
  }
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  const std::size_t path_bytes = std::strlen(path);
  if (path_bytes == 0 || path_bytes >= sizeof(address.sun_path)) {
    return {};
  }
  std::memcpy(address.sun_path, path, path_bytes + 1U);
  const socklen_t address_bytes = static_cast<socklen_t>(
      offsetof(sockaddr_un, sun_path) + path_bytes + 1U);
  if (::bind(socket.get(), reinterpret_cast<const sockaddr*>(&address),
             address_bytes) != 0 ||
      ::listen(socket.get(), 1) != 0) {
    return {};
  }
  return socket;
}

bool is_seqpacket(int fd) noexcept {
  int type = 0;
  socklen_t bytes = sizeof(type);
  return ::getsockopt(fd, SOL_SOCKET, SO_TYPE, &type, &bytes) == 0 &&
         bytes == sizeof(type) && type == SOCK_SEQPACKET;
}

bool is_eventfd(int fd) noexcept {
  char target[64]{};
  char path[64]{};
  const int path_bytes = std::snprintf(path, sizeof(path), "/proc/self/fd/%d", fd);
  if (path_bytes <= 0 || static_cast<std::size_t>(path_bytes) >= sizeof(path)) {
    return false;
  }
  const ssize_t target_bytes = ::readlink(path, target, sizeof(target));
  constexpr std::string_view expected = "anon_inode:[eventfd]";
  if (target_bytes != static_cast<ssize_t>(expected.size()) ||
      std::memcmp(target, expected.data(), expected.size()) != 0) {
    return false;
  }
  const int status_flags = ::fcntl(fd, F_GETFL);
  const int descriptor_flags = ::fcntl(fd, F_GETFD);
  return status_flags >= 0 && (status_flags & O_NONBLOCK) != 0 &&
         descriptor_flags >= 0 && (descriptor_flags & FD_CLOEXEC) != 0;
}

bool eventfds_are_distinct(int request_fd, int completion_fd) noexcept {
  const std::uint64_t one = 1U;
  if (::write(request_fd, &one, sizeof(one)) !=
      static_cast<ssize_t>(sizeof(one))) {
    return false;
  }
  pollfd completion{completion_fd, POLLIN, 0};
  const int poll_status = ::poll(&completion, 1, 0);
  std::uint64_t value = 0;
  const bool restored =
      ::read(request_fd, &value, sizeof(value)) ==
          static_cast<ssize_t>(sizeof(value)) &&
      value == one;
  return poll_status == 0 && completion.revents == 0 && restored;
}

bool is_sealed_mapping(int fd) noexcept {
  struct stat status {};
  if (::fstat(fd, &status) != 0 || !S_ISREG(status.st_mode) ||
      status.st_size != static_cast<off_t>(sb2::kMappingBytes)) {
    return false;
  }
  const int seals = ::fcntl(fd, F_GET_SEALS);
  constexpr int required = F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;
  return seals >= 0 && (seals & required) == required;
}

bool receive_bootstrap(int socket, ctl::Bootstrap* bootstrap,
                       std::array<Fd, 3>* descriptors) noexcept {
  std::array<int, 3> received{-1, -1, -1};
  std::array<std::byte, CMSG_SPACE(sizeof(received))> control{};
  iovec payload{bootstrap, sizeof(*bootstrap)};
  msghdr message{};
  message.msg_iov = &payload;
  message.msg_iovlen = 1;
  message.msg_control = control.data();
  message.msg_controllen = control.size();
  const ssize_t bytes = ::recvmsg(socket, &message, MSG_CMSG_CLOEXEC | MSG_TRUNC);
  if (bytes != static_cast<ssize_t>(sizeof(*bootstrap)) ||
      (message.msg_flags & (MSG_TRUNC | MSG_CTRUNC)) != 0) {
    return false;
  }

  cmsghdr* rights = nullptr;
  std::size_t header_count = 0;
  for (cmsghdr* header = CMSG_FIRSTHDR(&message); header != nullptr;
       header = CMSG_NXTHDR(&message, header)) {
    ++header_count;
    rights = header;
  }
  if (header_count != 1 || rights == nullptr || rights->cmsg_level != SOL_SOCKET ||
      rights->cmsg_type != SCM_RIGHTS ||
      rights->cmsg_len != CMSG_LEN(sizeof(received))) {
    return false;
  }
  std::memcpy(received.data(), CMSG_DATA(rights), sizeof(received));
  for (std::size_t index = 0; index < received.size(); ++index) {
    (*descriptors)[index].reset(received[index]);
  }
  return received[0] >= 0 && received[1] >= 0 && received[2] >= 0 &&
         received[0] != received[1] && received[0] != received[2] &&
         received[1] != received[2];
}

template <typename T>
bool send_packet(int socket, const T& packet) noexcept {
  const ssize_t bytes = ::send(socket, &packet, sizeof(packet), MSG_NOSIGNAL);
  return bytes == static_cast<ssize_t>(sizeof(packet));
}

bool receive_control(int socket, ctl::Control* control) noexcept {
  const ssize_t bytes = ::recv(socket, control, sizeof(*control), MSG_TRUNC);
  return bytes == static_cast<ssize_t>(sizeof(*control));
}

sb2::ErrorCode provider_error(sb1::ProviderStatus status) noexcept {
  switch (status) {
    case sb1::ProviderStatus::busy:
      return sb2::ErrorCode::core_busy;
    case sb1::ProviderStatus::not_loaded:
      return sb2::ErrorCode::core_not_loaded;
    case sb1::ProviderStatus::invalid_argument:
      return sb2::ErrorCode::bad_route;
    case sb1::ProviderStatus::generation_mismatch:
      return sb2::ErrorCode::core_generation;
    case sb1::ProviderStatus::sequence_mismatch:
      return sb2::ErrorCode::core_sequence;
    case sb1::ProviderStatus::device_error:
      return sb2::ErrorCode::core_device;
    case sb1::ProviderStatus::shutdown:
      return sb2::ErrorCode::core_shutdown;
    case sb1::ProviderStatus::ok:
      return sb2::ErrorCode::none;
  }
  return sb2::ErrorCode::internal;
}

sb2::CompletionCode provider_completion(sb1::ProviderStatus status) noexcept {
  switch (status) {
    case sb1::ProviderStatus::busy:
    case sb1::ProviderStatus::invalid_argument:
    case sb1::ProviderStatus::generation_mismatch:
    case sb1::ProviderStatus::sequence_mismatch:
      return sb2::CompletionCode::rejected;
    case sb1::ProviderStatus::shutdown:
      return sb2::CompletionCode::provider_draining;
    case sb1::ProviderStatus::not_loaded:
    case sb1::ProviderStatus::device_error:
      return sb2::CompletionCode::execution_failed;
    case sb1::ProviderStatus::ok:
      return sb2::CompletionCode::ok_all;
  }
  return sb2::CompletionCode::execution_failed;
}

void fill_statuses(sb2::ProviderClaim& claim, bool contributed) noexcept {
  const std::uint8_t remote = sb2::wire(contributed
                                            ? sb2::RouteStatus::contributed
                                            : sb2::RouteStatus::not_contributed);
  for (std::uint32_t index = 0; index < claim.completion.route_status.count;
       ++index) {
    claim.completion.route_status[index] =
        claim.request.remote_mask[index] != 0U
            ? remote
            : sb2::wire(sb2::RouteStatus::not_remote);
  }
  const std::uint8_t token =
      sb2::wire(contributed ? sb2::TokenStatus::contributed
                            : sb2::TokenStatus::not_contributed);
  for (std::uint32_t row = 0; row < claim.completion.token_status.count; ++row) {
    claim.completion.token_status[row] = token;
  }
}

bool validate_claim(const sb2::ProviderClaim& claim) noexcept {
  const std::uint32_t rows = claim.request.token_row_map.count;
  const std::uint32_t positions = rows * sb2::kTopK;
  if (claim.ticket.request_seq == 0U || claim.metadata.layer >= 32U ||
      claim.metadata.num_batch_tokens == 0U ||
      claim.metadata.num_batch_tokens > sb2::kMaxBatchTokens ||
      claim.metadata.num_staged_tokens != rows || rows == 0U ||
      rows > sb2::kMaxStagedTokens || rows > claim.metadata.num_batch_tokens ||
      claim.metadata.num_routes == 0U ||
      claim.metadata.num_routes > positions ||
      !claim.request.activation_fp16 || !claim.request.canonical_ids ||
      !claim.request.routing_weights || !claim.request.remote_mask ||
      !claim.request.token_row_map ||
      !claim.request.canonical_route_positions ||
      !claim.completion.output_fp32 || !claim.completion.route_status ||
      !claim.completion.token_status ||
      claim.request.activation_fp16.count != rows * sb2::kHiddenSize ||
      claim.request.canonical_ids.count != positions ||
      claim.request.routing_weights.count != positions ||
      claim.request.remote_mask.count != positions ||
      claim.request.canonical_route_positions.count != positions ||
      claim.completion.output_fp32.count != rows * sb2::kHiddenSize ||
      claim.completion.route_status.count != positions ||
      claim.completion.token_status.count != rows) {
    return false;
  }

  std::uint32_t remote_count = 0;
  std::uint32_t previous_token_row = 0;
  for (std::uint32_t row = 0; row < rows; ++row) {
    const std::uint32_t token_row = claim.request.token_row_map[row];
    if (token_row >= claim.metadata.num_batch_tokens ||
        (row != 0U && token_row <= previous_token_row)) {
      return false;
    }
    previous_token_row = token_row;
    std::uint32_t row_remote_count = 0;
    for (std::uint32_t route = 0; route < sb2::kTopK; ++route) {
      const std::uint32_t index = row * sb2::kTopK + route;
      const bool remote = claim.request.remote_mask[index] == 1U;
      if (claim.request.remote_mask[index] > 1U ||
          (remote && (claim.request.canonical_route_positions[index] != route ||
                      claim.request.canonical_ids[index] < 0 ||
                      claim.request.canonical_ids[index] >= 256 ||
                      !std::isfinite(claim.request.routing_weights[index]))) ||
          (!remote &&
           claim.request.canonical_route_positions[index] != 0xffffU)) {
        return false;
      }
      if (remote) {
        ++remote_count;
        ++row_remote_count;
      }
    }
    if (row_remote_count == 0U) {
      return false;
    }
  }
  return remote_count == claim.metadata.num_routes;
}

std::uint64_t timing_ns(double microseconds) noexcept {
  if (!std::isfinite(microseconds) || microseconds <= 0.0) {
    return 0;
  }
  constexpr double maximum =
      static_cast<double>(std::numeric_limits<std::uint64_t>::max());
  const double nanoseconds = microseconds * 1000.0;
  return nanoseconds >= maximum
             ? std::numeric_limits<std::uint64_t>::max()
             : static_cast<std::uint64_t>(nanoseconds + 0.5);
}

bool complete_failure(sb2::SharedRing& ring, sb2::ProviderClaim& claim,
                      sb2::CompletionCode completion,
                      sb2::ErrorCode error) noexcept {
  if (claim.request.remote_mask.data != nullptr &&
      claim.completion.route_status.data != nullptr &&
      claim.completion.token_status.data != nullptr &&
      claim.request.remote_mask.count == claim.completion.route_status.count &&
      claim.request.remote_mask.count <= sb2::kMaxRoutes &&
      claim.request.token_row_map.count == claim.completion.token_status.count &&
      claim.request.token_row_map.count <= sb2::kMaxStagedTokens) {
    fill_statuses(claim, false);
  }
  return ring.provider_complete(claim, completion, error, 0, 0,
                                monotonic_ns()) == sb2::RingStatus::ok;
}

int serve(sb2::SharedRing& ring, sb1::B70Provider& provider,
          const ExpertMap& canonical_to_local, int control_socket,
          std::uint64_t allocation_baseline, bool log_requests) {
  std::vector<sycl::half> hidden(sb2::kMaxStagedTokens * sb2::kHiddenSize);
  std::vector<std::int32_t> local_ids(sb2::kMaxRoutes, -1);
  std::vector<float> local_weights(sb2::kMaxRoutes, 0.0F);
  sb1::DispatchResult dispatch{};
  dispatch.kernel.reserve(16);
  char timing_line[256]{};

  for (;;) {
    sb2::ProviderClaim claim{};
    const sb2::RingStatus claim_status =
        ring.provider_claim(monotonic_ns(), &claim);
    if (claim_status == sb2::RingStatus::ok) {
      const std::uint32_t rows = claim.request.token_row_map.count;
      if (!validate_claim(claim)) {
        if (!complete_failure(ring, claim, sb2::CompletionCode::protocol_error,
                              sb2::ErrorCode::bad_route)) {
          return 30;
        }
        continue;
      }

      std::memcpy(hidden.data(), claim.request.activation_fp16.data,
                  static_cast<std::size_t>(rows) * sb2::kHiddenSize *
                      sizeof(sycl::half));
      bool finite_activations = true;
      for (std::size_t index = 0;
           index < static_cast<std::size_t>(rows) * sb2::kHiddenSize; ++index) {
        if (!std::isfinite(static_cast<float>(hidden[index]))) {
          finite_activations = false;
          break;
        }
      }
      if (!finite_activations) {
        if (!complete_failure(ring, claim, sb2::CompletionCode::rejected,
                              sb2::ErrorCode::bad_bounds)) {
          return 31;
        }
        continue;
      }

      const std::uint32_t positions = rows * sb2::kTopK;
      std::fill_n(local_ids.data(), positions, -1);
      std::fill_n(local_weights.data(), positions, 0.0F);
      bool missing_resident_expert = false;
      for (std::uint32_t index = 0; index < positions; ++index) {
        if (claim.request.remote_mask[index] != 0U) {
          const std::int32_t canonical = claim.request.canonical_ids[index];
          const std::int32_t local =
              canonical_to_local[static_cast<std::size_t>(canonical)];
          if (local < 0) {
            missing_resident_expert = true;
            break;
          }
          local_ids[index] = local;
          local_weights[index] = claim.request.routing_weights[index];
        }
      }
      if (missing_resident_expert) {
        if (!complete_failure(ring, claim, sb2::CompletionCode::rejected,
                              sb2::ErrorCode::bad_route)) {
          return 31;
        }
        continue;
      }

      const sb1::ProviderStatus issue = provider.issue(
          ring.identity().provider_generation, claim.ticket.request_seq,
          claim.metadata.layer, hidden.data(), local_ids.data(),
          local_weights.data(), rows);
      if (issue != sb1::ProviderStatus::ok) {
        if (!complete_failure(ring, claim, provider_completion(issue),
                              provider_error(issue))) {
          return 32;
        }
        continue;
      }

      const sb1::ProviderStatus take = provider.take(
          ring.identity().provider_generation, claim.ticket.request_seq,
          claim.completion.output_fp32.data, claim.completion.output_fp32.count,
          &dispatch);
      if (take != sb1::ProviderStatus::ok) {
        if (!complete_failure(ring, claim, provider_completion(take),
                              provider_error(take))) {
          return 33;
        }
        continue;
      }

      bool finite_output = true;
      for (std::uint32_t index = 0;
           index < claim.completion.output_fp32.count; ++index) {
        if (!std::isfinite(claim.completion.output_fp32[index])) {
          finite_output = false;
          break;
        }
      }
      if (!finite_output) {
        if (!complete_failure(ring, claim,
                              sb2::CompletionCode::execution_failed,
                              sb2::ErrorCode::core_device)) {
          return 34;
        }
        continue;
      }

      fill_statuses(claim, true);
      const std::uint64_t kernel_ns = timing_ns(dispatch.kernel_us);
      const std::uint64_t total_ns = timing_ns(dispatch.total_us);
      if (ring.provider_complete(claim, sb2::CompletionCode::ok_all,
                                 sb2::ErrorCode::none, kernel_ns, total_ns,
                                 monotonic_ns()) != sb2::RingStatus::ok) {
        return 34;
      }
      if (log_requests) {
        const int line_bytes = std::snprintf(
            timing_line, sizeof(timing_line),
            "Phase-2 request seq=%llu M=%zu kernel=%s kernel_us=%.3f "
            "total_us=%.3f\n",
            static_cast<unsigned long long>(dispatch.sequence), dispatch.M,
            dispatch.kernel.c_str(), dispatch.kernel_us, dispatch.total_us);
        if (line_bytes > 0) {
          static_cast<void>(::write(
              STDOUT_FILENO, timing_line,
              static_cast<std::size_t>(line_bytes) < sizeof(timing_line)
                  ? static_cast<std::size_t>(line_bytes)
                  : sizeof(timing_line) - 1U));
        }
      }
      continue;
    }

    if (claim.valid) {
      sb2::CompletionCode completion = sb2::CompletionCode::protocol_error;
      sb2::ErrorCode error = sb2::ErrorCode::internal;
      if (claim_status == sb2::RingStatus::cancelled) {
        completion = sb2::CompletionCode::cancelled;
        error = sb2::ErrorCode::cancelled;
      } else if (claim_status == sb2::RingStatus::deadline_expired) {
        completion = sb2::CompletionCode::deadline_exceeded;
        error = sb2::ErrorCode::deadline;
      } else if (claim_status == sb2::RingStatus::generation_mismatch) {
        error = sb2::ErrorCode::bad_generation;
      } else if (claim_status == sb2::RingStatus::invalid_descriptor ||
                 claim_status == sb2::RingStatus::protocol_mismatch) {
        error = sb2::ErrorCode::bad_descriptor;
      } else if (claim_status == sb2::RingStatus::invalid_payload) {
        error = sb2::ErrorCode::bad_route;
      }
      if (!complete_failure(ring, claim, completion, error)) {
        return 35;
      }
      continue;
    }
    if (claim_status != sb2::RingStatus::would_block) {
      return 36;
    }

    pollfd waits[2]{{control_socket, POLLIN | POLLHUP | POLLERR, 0},
                    {ring.request_eventfd(), POLLIN | POLLERR, 0}};
    const int poll_status = ::poll(waits, 2, kWaitMs);
    if (poll_status < 0) {
      if (errno == EINTR) {
        continue;
      }
      return 37;
    }
    if ((waits[1].revents & POLLIN) != 0 &&
        ring.drain_request_notifications() != sb2::RingStatus::ok) {
      return 38;
    }
    if ((waits[1].revents & (POLLERR | POLLNVAL)) != 0) {
      return 39;
    }
    if ((waits[0].revents & POLLIN) != 0) {
      ctl::Control control{};
      if (!receive_control(control_socket, &control) ||
          !exact_magic(control.magic, ctl::kControlMagic) ||
          control.major != ctl::kProtocolMajor ||
          control.minor != ctl::kProtocolMinor ||
          control.bytes != sizeof(control) || control.reserved0 != 0U ||
          !all_zero(control.reserved, sizeof(control.reserved))) {
        return 40;
      }
      if (control.command == ctl::Command::shutdown) {
        if (control.fault_point != ctl::FaultPoint::none ||
            control.fault_argument != ctl::FaultArgument::none ||
            control.sequence != 0U) {
          return 40;
        }
        break;
      }
      if (control.command != ctl::Command::arm_test_fault) {
        return 40;
      }

      ctl::FaultAck acknowledgment{};
      copy_magic(acknowledgment.magic, ctl::kFaultAckMagic);
      acknowledgment.major = ctl::kProtocolMajor;
      acknowledgment.minor = ctl::kProtocolMinor;
      acknowledgment.bytes = sizeof(acknowledgment);
      acknowledgment.provider_status = ctl::kNoProviderStatus;
      acknowledgment.fault_point = control.fault_point;
      acknowledgment.fault_argument = control.fault_argument;
      acknowledgment.sequence = control.sequence;
      const bool valid_fault =
          control.fault_point ==
              ctl::FaultPoint::after_kernel_before_copyout &&
          control.fault_argument == ctl::FaultArgument::none &&
          control.sequence != 0U;
      if (!valid_fault) {
        acknowledgment.status = ctl::FaultAckStatus::invalid_request;
      } else {
#if defined(SHOOTING_BRAKE_ENABLE_TEST_FAULTS)
        const sb1::ProviderStatus arm_status = provider.arm_test_fault(
            sb1::ProviderTestFault::after_kernel_before_copyout,
            control.sequence);
        acknowledgment.provider_status =
            static_cast<std::uint32_t>(arm_status);
        acknowledgment.status =
            arm_status == sb1::ProviderStatus::ok
                ? ctl::FaultAckStatus::armed
                : ctl::FaultAckStatus::provider_rejected;
#else
        acknowledgment.status = ctl::FaultAckStatus::unsupported;
#endif
      }
      if (!send_packet(control_socket, acknowledgment)) {
        return 40;
      }
      continue;
    }
    if ((waits[0].revents & (POLLHUP | POLLERR | POLLNVAL)) != 0) {
      return 41;
    }
  }

  const sb1::Health final_health = provider.health();
  ctl::ShutdownReply reply{};
  copy_magic(reply.magic, ctl::kShutdownMagic);
  reply.major = ctl::kProtocolMajor;
  reply.minor = ctl::kProtocolMinor;
  reply.bytes = sizeof(reply);
  reply.status = final_health.loaded && !final_health.pending &&
                         !final_health.stopped &&
                         final_health.allocations == allocation_baseline
                     ? 0U
                     : 1U;
  reply.dispatches = final_health.dispatches;
  reply.allocation_baseline = allocation_baseline;
  reply.allocation_final = final_health.allocations;
  return send_packet(control_socket, reply) && reply.status == 0U ? 0 : 42;
}

int run(const Options& options) {
  const ExpertMap canonical_to_local =
      make_expert_map(options.resident_experts);
  const sb2::Fingerprint configured_placement =
      placement_fingerprint(options.resident_experts);
  SocketPath socket_path(options.socket_path);
  Fd listener = make_listener(socket_path.c_str());
  if (listener.get() < 0) {
    std::fprintf(stderr, "b70_ring_provider: cannot bind SOCK_SEQPACKET socket: %s\n",
                 std::strerror(errno));
    return 10;
  }
  Fd connection(::accept4(listener.get(), nullptr, nullptr, SOCK_CLOEXEC));
  if (connection.get() < 0 || !is_seqpacket(connection.get())) {
    std::fprintf(stderr, "b70_ring_provider: accept failed\n");
    return 11;
  }
  listener.reset();
  socket_path.unlink_now();

  ctl::Bootstrap bootstrap{};
  std::array<Fd, 3> descriptors{};
  if (!receive_bootstrap(connection.get(), &bootstrap, &descriptors) ||
      !exact_magic(bootstrap.magic, ctl::kBootstrapMagic) ||
      bootstrap.major != ctl::kProtocolMajor ||
      bootstrap.minor != ctl::kProtocolMinor ||
      bootstrap.bytes != sizeof(bootstrap) ||
      bootstrap.mapping_bytes != sb2::kMappingBytes ||
      !all_zero(bootstrap.reserved, sizeof(bootstrap.reserved)) ||
      bootstrap.identity.ring_generation == 0U ||
      bootstrap.identity.provider_generation == 0U ||
      bootstrap.identity.placement_generation == 0U ||
      bootstrap.identity.weight_generation == 0U ||
      bootstrap.identity.provider_pid != static_cast<std::uint64_t>(::getpid()) ||
      !matches_trusted_identity(options, bootstrap.identity) ||
      !is_sealed_mapping(descriptors[0].get()) ||
      !is_eventfd(descriptors[1].get()) || !is_eventfd(descriptors[2].get()) ||
      !eventfds_are_distinct(descriptors[1].get(), descriptors[2].get())) {
    std::fprintf(stderr, "b70_ring_provider: invalid bootstrap packet or fds\n");
    return 12;
  }

  if (!same_fingerprint(configured_placement,
                        options.expected_placement_sha256)) {
    std::fprintf(
        stderr,
        "b70_ring_provider: resident experts do not match placement SHA-256\n");
    return 15;
  }
  Fd verified_bank(::open(options.bank_path.c_str(), O_RDONLY | O_CLOEXEC));
  if (verified_bank.get() < 0) {
    std::fprintf(stderr, "b70_ring_provider: cannot open expert bank: %s\n",
                 std::strerror(errno));
    return 16;
  }
  sb2::Fingerprint actual_bank_fingerprint{};
  if (!bank_fingerprint(verified_bank.get(), &actual_bank_fingerprint)) {
    std::fprintf(stderr, "b70_ring_provider: cannot hash expert bank: %s\n",
                 std::strerror(errno));
    return 16;
  }
  if (!same_fingerprint(actual_bank_fingerprint,
                        options.expected_weight_sha256)) {
    std::fprintf(stderr,
                 "b70_ring_provider: expert bank does not match weight SHA-256\n");
    return 17;
  }
  const std::string verified_bank_path =
      "/proc/self/fd/" + std::to_string(verified_bank.get());

  sb2::SharedRing ring;
  const char* attach_detail = nullptr;
  const sb2::RingStatus attach = sb2::SharedRing::attach(
      descriptors[0].get(), descriptors[1].get(), descriptors[2].get(),
      bootstrap.identity, &ring, &attach_detail);
  if (attach != sb2::RingStatus::ok) {
    std::fprintf(stderr, "b70_ring_provider: attach failed: %s%s%s\n",
                 sb2::status_message(attach), attach_detail == nullptr ? "" : ": ",
                 attach_detail == nullptr ? "" : attach_detail);
    return 13;
  }

  sb1::B70Provider provider;
  sb1::ProviderConfig config{};
  config.max_batch = sb2::kMaxStagedTokens;
  config.top_k = sb2::kTopK;
  config.generation = bootstrap.identity.provider_generation;
  config.resident_experts = options.resident_experts;
  const sb1::ProviderStatus load = provider.load(verified_bank_path, config);

  ctl::StartupReply startup{};
  copy_magic(startup.magic, ctl::kStartupMagic);
  startup.major = ctl::kProtocolMajor;
  startup.minor = ctl::kProtocolMinor;
  startup.bytes = sizeof(startup);
  startup.startup_status = load == sb1::ProviderStatus::ok ? 0U : 1U;
  startup.provider_status = static_cast<std::uint32_t>(load);
  std::uint64_t allocation_baseline = 0;
  if (load == sb1::ProviderStatus::ok) {
    const sb1::Health health = provider.health();
    if (!health.loaded || health.pending || health.stopped ||
        health.generation != bootstrap.identity.provider_generation ||
        health.allocations == 0U) {
      startup.startup_status = 2U;
    } else {
      allocation_baseline = health.allocations;
      startup.allocation_baseline = allocation_baseline;
    }
  }
  if (!send_packet(connection.get(), startup) || startup.startup_status != 0U) {
    return 14;
  }

  const int service_status =
      serve(ring, provider, canonical_to_local, connection.get(),
            allocation_baseline, !options.quiet);
  provider.shutdown();
  return service_status;
}

}  // namespace

int main(int argc, char** argv) {
  static_cast<void>(::signal(SIGPIPE, SIG_IGN));
  Options options;
  if (!parse_options(argc, argv, &options)) {
    std::fprintf(
        stderr,
        "usage: b70_ring_provider --socket PATH [--bank PATH] "
        "[--resident-experts ID,ID,...] "
        "--expected-placement-generation N --expected-weight-generation N "
        "--expected-placement-sha256 HEX --expected-weight-sha256 HEX "
        "[--quiet]\n");
    return 2;
  }
  try {
    return run(options);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "b70_ring_provider: %s\n", error.what());
    return 70;
  } catch (...) {
    std::fprintf(stderr, "b70_ring_provider: unknown failure\n");
    return 71;
  }
}
