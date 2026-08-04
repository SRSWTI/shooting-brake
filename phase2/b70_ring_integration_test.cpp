#include "b70_ring_control.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

namespace sb = shooting_brake::phase2;
namespace ctl = shooting_brake::phase2::b70_control;
namespace {

constexpr std::uint32_t kIntermediateSize = 512;
constexpr float kAtol = 1.0e-6F;
constexpr float kRtol = 1.0e-2F;
constexpr int kPollSliceMs = 250;
constexpr std::uint64_t kOperationTimeoutNs = 120'000'000'000ULL;
constexpr std::uint64_t kExpectedPlacementGeneration = 29;
constexpr std::uint64_t kExpectedWeightGeneration = 31;

inline constexpr sb::Fingerprint kExpectedPlacementSha256{{
    0x1aU, 0xa9U, 0x92U, 0x90U, 0x92U, 0x9cU, 0xb6U, 0xd7U,
    0x38U, 0xf1U, 0xd0U, 0xdeU, 0xe4U, 0xf6U, 0x06U, 0x57U,
    0xb2U, 0xbeU, 0xcbU, 0x93U, 0x6bU, 0x51U, 0x90U, 0x30U,
    0x04U, 0xffU, 0x87U, 0xf4U, 0xadU, 0xa0U, 0x96U, 0xe9U}};
inline constexpr sb::Fingerprint kExpectedWeightSha256{{
    0x0cU, 0xe6U, 0x37U, 0x7bU, 0xa3U, 0xc9U, 0x84U, 0x8dU,
    0xa4U, 0x2bU, 0x60U, 0x63U, 0x57U, 0x4eU, 0xa8U, 0x84U,
    0x05U, 0x2dU, 0x2eU, 0x0fU, 0x5eU, 0x60U, 0x5dU, 0x86U,
    0xd1U, 0x68U, 0x4aU, 0x1eU, 0x58U, 0x26U, 0xe8U, 0xdbU}};

sb::Fingerprint mismatched(sb::Fingerprint fingerprint) noexcept {
  fingerprint.bytes[0] ^= 0xffU;
  return fingerprint;
}


class TestFailure final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

[[noreturn]] void fail(const std::string& message) { throw TestFailure(message); }

void require(bool condition, const std::string& message) {
  if (!condition) {
    fail(message);
  }
}

void require_status(sb::RingStatus actual, sb::RingStatus expected,
                    std::string_view operation) {
  if (actual != expected) {
    fail(std::string(operation) + ": expected " + sb::status_message(expected) +
         ", got " + sb::status_message(actual));
  }
}

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

class TemporaryDirectory final {
 public:
  TemporaryDirectory() {
    std::array<char, 64> pattern{};
    constexpr std::string_view base = "/tmp/shooting-brake-b70-XXXXXX";
    std::memcpy(pattern.data(), base.data(), base.size());
    char* created = ::mkdtemp(pattern.data());
    if (created == nullptr) {
      fail("mkdtemp failed: " + std::string(std::strerror(errno)));
    }
    path_ = created;
  }
  ~TemporaryDirectory() noexcept {
    if (!socket_path_.empty()) {
      static_cast<void>(::unlink(socket_path_.c_str()));
    }
    if (!path_.empty()) {
      static_cast<void>(::rmdir(path_.c_str()));
    }
  }
  std::string socket_path() {
    socket_path_ = path_ + "/ring.sock";
    return socket_path_;
  }

 private:
  std::string path_;
  std::string socket_path_;
};

class ChildProcess final {
 public:
  explicit ChildProcess(pid_t pid) noexcept : pid_(pid) {}
  ~ChildProcess() noexcept {
    if (pid_ > 0 && !reaped_) {
      static_cast<void>(::kill(pid_, SIGKILL));
      int status = 0;
      while (::waitpid(pid_, &status, 0) < 0 && errno == EINTR) {
      }
    }
  }
  ChildProcess(const ChildProcess&) = delete;
  ChildProcess& operator=(const ChildProcess&) = delete;
  pid_t pid() const noexcept { return pid_; }
  bool poll_exit(int* status) noexcept {
    if (reaped_) {
      *status = status_;
      return true;
    }
    const pid_t result = ::waitpid(pid_, &status_, WNOHANG);
    if (result == pid_) {
      reaped_ = true;
      *status = status_;
      return true;
    }
    return false;
  }
  int wait_bounded() {
    const std::uint64_t deadline = monotonic_ns() + kOperationTimeoutNs;
    for (;;) {
      int status = 0;
      if (poll_exit(&status)) {
        return status;
      }
      if (monotonic_ns() >= deadline) {
        fail("provider did not exit after clean shutdown");
      }
      timespec pause{0, 20'000'000};
      while (::nanosleep(&pause, &pause) != 0 && errno == EINTR) {
      }
    }
  }

 private:
  static std::uint64_t monotonic_ns() noexcept {
    timespec value{};
    if (::clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
      return 0;
    }
    return static_cast<std::uint64_t>(value.tv_sec) * 1'000'000'000ULL +
           static_cast<std::uint64_t>(value.tv_nsec);
  }

  pid_t pid_ = -1;
  bool reaped_ = false;
  int status_ = 0;
};

struct GoldenReference final {
  std::vector<std::uint16_t> hidden;
  std::vector<float> output;
};

enum class Routes { zero, single, duplicate_top8 };

struct RequestCase final {
  std::uint32_t staged_tokens = 1;
  std::uint32_t batch_tokens = 1;
  Routes routes = Routes::single;
  float single_weight = 1.0F;
  bool sparse_row_map = false;
};

struct Pending final {
  sb::RingTicket ticket{};
  std::uint32_t staged_tokens = 0;
  Routes routes = Routes::zero;
  float output_scale = 0.0F;
  std::uint64_t publication_monotonic_ns = 0;
};

struct RequestTiming final {
  std::uint64_t ring_wall_ns = 0;
  std::uint64_t kernel_ns = 0;
  std::uint64_t provider_total_ns = 0;
};
std::uint64_t monotonic_ns() noexcept {
  timespec value{};
  if (::clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
    return 0;
  }
  return static_cast<std::uint64_t>(value.tv_sec) * 1'000'000'000ULL +
         static_cast<std::uint64_t>(value.tv_nsec);
}

bool exact_magic(const char* actual, const std::array<char, 8>& expected) noexcept {
  return std::memcmp(actual, expected.data(), expected.size()) == 0;
}

bool all_zero(const void* data, std::size_t bytes) noexcept {
  const auto* values = static_cast<const std::uint8_t*>(data);
  for (std::size_t index = 0; index < bytes; ++index) {
    if (values[index] != 0U) {
      return false;
    }
  }
  return true;
}

void copy_magic(char* destination, const std::array<char, 8>& source) noexcept {
  std::memcpy(destination, source.data(), source.size());
}

std::string executable_directory() {
  std::array<char, 4096> path{};
  const ssize_t bytes = ::readlink("/proc/self/exe", path.data(), path.size());
  if (bytes <= 0 || bytes == static_cast<ssize_t>(path.size())) {
    fail("cannot resolve integration test executable path");
  }
  std::string result(path.data(), static_cast<std::size_t>(bytes));
  const std::size_t slash = result.find_last_of('/');
  if (slash == std::string::npos) {
    fail("integration test executable has no parent directory");
  }
  result.resize(slash);
  return result;
}

GoldenReference load_golden(const std::string& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    fail("cannot open golden fixture " + path);
  }
  const std::streampos end = input.tellg();
  constexpr std::size_t expected_bytes =
      3U * sizeof(std::uint32_t) + sb::kHiddenSize * sizeof(std::uint16_t) +
      sizeof(std::int32_t) + sizeof(float) +
      sb::kHiddenSize * sizeof(float);
  if (end < 0 || static_cast<std::size_t>(end) != expected_bytes) {
    fail("golden fixture has wrong exact size");
  }
  std::vector<std::byte> bytes(expected_bytes);
  input.seekg(0, std::ios::beg);
  input.read(reinterpret_cast<char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!input || static_cast<std::size_t>(input.gcount()) != bytes.size()) {
    fail("golden fixture short read");
  }
  std::size_t offset = 0;
  const auto take = [&](void* destination, std::size_t count) {
    if (count > bytes.size() - offset) {
      fail("golden fixture truncated payload");
    }
    std::memcpy(destination, bytes.data() + offset, count);
    offset += count;
  };
  std::uint32_t hidden_size = 0;
  std::uint32_t intermediate_size = 0;
  std::uint32_t top_k = 0;
  take(&hidden_size, sizeof(hidden_size));
  take(&intermediate_size, sizeof(intermediate_size));
  take(&top_k, sizeof(top_k));
  require(hidden_size == sb::kHiddenSize &&
              intermediate_size == kIntermediateSize && top_k == 1U,
          "golden fixture header mismatch");
  GoldenReference golden;
  golden.hidden.resize(sb::kHiddenSize);
  golden.output.resize(sb::kHiddenSize);
  take(golden.hidden.data(), golden.hidden.size() * sizeof(std::uint16_t));
  std::int32_t expert = -1;
  float weight = 0.0F;
  take(&expert, sizeof(expert));
  take(&weight, sizeof(weight));
  require(expert == 0 && weight == 1.0F,
          "golden fixture route is not expert 0 at weight 1");
  take(golden.output.data(), golden.output.size() * sizeof(float));
  require(offset == bytes.size(), "golden fixture has trailing data");
  for (float value : golden.output) {
    require(std::isfinite(value), "golden fixture contains non-finite output");
  }
  return golden;
}

sb::RingIdentity make_identity(
    pid_t provider_pid,
    const sb::Fingerprint& placement_sha256 = kExpectedPlacementSha256,
    const sb::Fingerprint& weight_sha256 = kExpectedWeightSha256) noexcept {
  sb::RingIdentity identity{};
  identity.ring_generation = 101;
  identity.provider_generation = 73;
  identity.placement_generation = kExpectedPlacementGeneration;
  identity.weight_generation = kExpectedWeightGeneration;
  identity.provider_pid = static_cast<std::uint64_t>(provider_pid);
  for (std::size_t index = 0; index < sizeof(identity.provider_nonce.bytes);
       ++index) {
    identity.provider_nonce.bytes[index] =
        static_cast<std::uint8_t>(0x10U + index);
    identity.ring_nonce.bytes[index] = static_cast<std::uint8_t>(0x80U + index);
  }
  identity.placement_sha256 = placement_sha256;
  identity.weight_sha256 = weight_sha256;
  return identity;
}

std::string fingerprint_hex(const sb::Fingerprint& fingerprint) {
  constexpr char kHex[] = "0123456789abcdef";
  std::string result(2U * sizeof(fingerprint.bytes), '0');
  for (std::size_t index = 0; index < sizeof(fingerprint.bytes); ++index) {
    result[index * 2U] = kHex[fingerprint.bytes[index] >> 4U];
    result[index * 2U + 1U] = kHex[fingerprint.bytes[index] & 0x0fU];
  }
  return result;
}

pid_t spawn_provider(
    const std::string& provider_path, const std::string& socket_path,
    const std::string& bank_path,
    const sb::Fingerprint& placement_sha256 = kExpectedPlacementSha256,
    const sb::Fingerprint& weight_sha256 = kExpectedWeightSha256) {
  const std::string placement_generation =
      std::to_string(kExpectedPlacementGeneration);
  const std::string weight_generation =
      std::to_string(kExpectedWeightGeneration);
  const std::string placement_sha256_hex = fingerprint_hex(placement_sha256);
  const std::string weight_sha256_hex = fingerprint_hex(weight_sha256);
  const pid_t child = ::fork();
  if (child < 0) {
    fail("fork failed: " + std::string(std::strerror(errno)));
  }
  if (child == 0) {
    ::execl(
        provider_path.c_str(), provider_path.c_str(), "--socket",
        socket_path.c_str(), "--bank", bank_path.c_str(),
        "--expected-placement-generation", placement_generation.c_str(),
        "--expected-weight-generation", weight_generation.c_str(),
        "--expected-placement-sha256", placement_sha256_hex.c_str(),
        "--expected-weight-sha256", weight_sha256_hex.c_str(), "--quiet",
        static_cast<char*>(nullptr));
    _exit(127);
  }
  return child;
}

Fd connect_provider(const std::string& path, ChildProcess& child) {
  const std::uint64_t deadline = monotonic_ns() + kOperationTimeoutNs;
  for (;;) {
    Fd socket(::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0));
    if (socket.get() < 0) {
      fail("cannot create bootstrap socket");
    }
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    require(path.size() < sizeof(address.sun_path), "bootstrap path too long");
    std::memcpy(address.sun_path, path.c_str(), path.size() + 1U);
    const socklen_t bytes = static_cast<socklen_t>(
        offsetof(sockaddr_un, sun_path) + path.size() + 1U);
    if (::connect(socket.get(), reinterpret_cast<const sockaddr*>(&address),
                  bytes) == 0) {
      return socket;
    }
    const int connect_error = errno;
    int child_status = 0;
    if (child.poll_exit(&child_status)) {
      fail("provider exited before bootstrap, wait status=" +
           std::to_string(child_status));
    }
    if (connect_error != ENOENT && connect_error != ECONNREFUSED) {
      fail("bootstrap connect failed: " +
           std::string(std::strerror(connect_error)));
    }
    if (monotonic_ns() >= deadline) {
      fail("timed out waiting for provider bootstrap socket");
    }
    timespec pause{0, 20'000'000};
    while (::nanosleep(&pause, &pause) != 0 && errno == EINTR) {
    }
  }
}

void send_bootstrap(int socket, const sb::SharedRing& ring) {
  ctl::Bootstrap bootstrap{};
  copy_magic(bootstrap.magic, ctl::kBootstrapMagic);
  bootstrap.major = ctl::kProtocolMajor;
  bootstrap.minor = ctl::kProtocolMinor;
  bootstrap.bytes = sizeof(bootstrap);
  bootstrap.mapping_bytes = ring.mapping_size();
  bootstrap.identity = ring.identity();

  std::array<int, 3> descriptors{ring.fd(), ring.request_eventfd(),
                                 ring.completion_eventfd()};
  std::array<std::byte, CMSG_SPACE(sizeof(descriptors))> control{};
  iovec payload{&bootstrap, sizeof(bootstrap)};
  msghdr message{};
  message.msg_iov = &payload;
  message.msg_iovlen = 1;
  message.msg_control = control.data();
  message.msg_controllen = control.size();
  cmsghdr* header = CMSG_FIRSTHDR(&message);
  require(header != nullptr, "cannot construct SCM_RIGHTS header");
  header->cmsg_level = SOL_SOCKET;
  header->cmsg_type = SCM_RIGHTS;
  header->cmsg_len = CMSG_LEN(sizeof(descriptors));
  std::memcpy(CMSG_DATA(header), descriptors.data(), sizeof(descriptors));
  const ssize_t sent = ::sendmsg(socket, &message, MSG_NOSIGNAL);
  require(sent == static_cast<ssize_t>(sizeof(bootstrap)),
          "bootstrap sendmsg was not exact size");
}

void check_startup_identity_rejection(
    const std::string& provider_path, const std::string& bank_path,
    const sb::Fingerprint& placement_sha256,
    const sb::Fingerprint& weight_sha256, int expected_exit,
    std::string_view operation) {
  TemporaryDirectory temporary;
  const std::string socket_path = temporary.socket_path();
  ChildProcess child(spawn_provider(provider_path, socket_path, bank_path,
                                      placement_sha256, weight_sha256));
  const sb::RingIdentity identity =
      make_identity(child.pid(), placement_sha256, weight_sha256);
  sb::SharedRing ring;
  const char* create_detail = nullptr;
  require_status(sb::SharedRing::create(identity, &ring, &create_detail),
                 sb::RingStatus::ok,
                 create_detail == nullptr ? operation : create_detail);
  Fd connection = connect_provider(socket_path, child);
  send_bootstrap(connection.get(), ring);

  const int wait_status = child.wait_bounded();
  require(WIFEXITED(wait_status) &&
              WEXITSTATUS(wait_status) == expected_exit,
          std::string(operation) +
              ": provider did not reject before bank load");
  std::byte unexpected{};
  const ssize_t received =
      ::recv(connection.get(), &unexpected, sizeof(unexpected), MSG_DONTWAIT);
  require(received == 0,
          std::string(operation) +
              ": provider emitted a startup packet after identity rejection");
}

void wait_readable(int fd, std::string_view operation) {
  const std::uint64_t deadline = monotonic_ns() + kOperationTimeoutNs;
  for (;;) {
    pollfd wait{fd, POLLIN | POLLHUP | POLLERR, 0};
    const int status = ::poll(&wait, 1, kPollSliceMs);
    if (status < 0) {
      if (errno == EINTR) {
        continue;
      }
      fail(std::string(operation) + ": poll failed");
    }
    if ((wait.revents & POLLIN) != 0) {
      return;
    }
    if ((wait.revents & (POLLHUP | POLLERR | POLLNVAL)) != 0) {
      fail(std::string(operation) + ": peer socket closed");
    }
    if (monotonic_ns() >= deadline) {
      fail(std::string(operation) + ": timed out");
    }
  }
}

template <typename T>
T receive_exact(int socket, std::string_view operation) {
  wait_readable(socket, operation);
  T packet{};
  const ssize_t bytes = ::recv(socket, &packet, sizeof(packet), MSG_TRUNC);
  require(bytes == static_cast<ssize_t>(sizeof(packet)),
          std::string(operation) + ": wrong packet size");
  return packet;
}

template <typename T>
void send_exact(int socket, const T& packet, std::string_view operation) {
  const ssize_t bytes = ::send(socket, &packet, sizeof(packet), MSG_NOSIGNAL);
  require(bytes == static_cast<ssize_t>(sizeof(packet)),
          std::string(operation) + ": short send");
}

float route_scale(const RequestCase& request) noexcept {
  if (request.routes == Routes::single) {
    return request.single_weight;
  }
  if (request.routes == Routes::duplicate_top8) {
    constexpr std::array<float, sb::kTopK> weights{
        0.03F, 0.07F, 0.11F, 0.13F, 0.17F, 0.19F, 0.23F, 0.07F};
    float sum = 0.0F;
    for (float weight : weights) {
      sum += weight;
    }
    return sum;
  }
  return 0.0F;
}

bool publish_request(sb::SharedRing& ring, const GoldenReference& golden,
                     std::uint64_t sequence, const RequestCase& request,
                     Pending* pending) {
  const std::uint32_t routes_per_row =
      request.routes == Routes::zero
          ? 0U
          : (request.routes == Routes::single ? 1U : sb::kTopK);
  const std::uint32_t num_routes = request.staged_tokens * routes_per_row;
  if (num_routes == 0U) {
    return false;
  }

  require(request.staged_tokens > 0U &&
              request.staged_tokens <= sb::kMaxStagedTokens &&
              request.batch_tokens >= request.staged_tokens &&
              request.batch_tokens <= sb::kMaxBatchTokens,
          "invalid integration request shape");
  sb::RequestSpec spec{};
  spec.request_seq = sequence;
  spec.scheduler_step = 1000U + sequence;
  spec.deadline_monotonic_ns = monotonic_ns() + kOperationTimeoutNs;
  spec.layer = 0;
  spec.num_batch_tokens = request.batch_tokens;
  spec.num_staged_tokens = request.staged_tokens;
  spec.num_routes = num_routes;

  sb::RequestPayload payload{};
  sb::RingTicket ticket{};
  require_status(ring.host_begin(spec, &ticket, &payload), sb::RingStatus::ok,
                 "host_begin valid B70 request");
  for (std::uint32_t row = 0; row < request.staged_tokens; ++row) {
    std::memcpy(payload.activation_fp16.data + row * sb::kHiddenSize,
                golden.hidden.data(),
                golden.hidden.size() * sizeof(std::uint16_t));
    payload.token_row_map[row] = request.sparse_row_map ? row * 2U : row;
    for (std::uint32_t route = 0; route < sb::kTopK; ++route) {
      const std::uint32_t index = row * sb::kTopK + route;
      const bool remote = request.routes == Routes::duplicate_top8 || route == 0U;
      payload.remote_mask[index] = remote ? 1U : 0U;
      payload.canonical_ids[index] = remote ? 0 : -1;
      payload.routing_weights[index] = 0.0F;
      payload.canonical_route_positions[index] =
          remote ? static_cast<std::uint16_t>(route) : 0xffffU;
    }
    if (request.routes == Routes::single) {
      payload.routing_weights[row * sb::kTopK] = request.single_weight;
    } else {
      constexpr std::array<float, sb::kTopK> weights{
          0.03F, 0.07F, 0.11F, 0.13F, 0.17F, 0.19F, 0.23F, 0.07F};
      for (std::uint32_t route = 0; route < sb::kTopK; ++route) {
        payload.routing_weights[row * sb::kTopK + route] = weights[route];
      }
    }
  }
  pending->ticket = ticket;
  pending->staged_tokens = request.staged_tokens;
  pending->routes = request.routes;
  pending->output_scale = route_scale(request);
  pending->publication_monotonic_ns = monotonic_ns();
  require_status(ring.host_publish(ticket), sb::RingStatus::ok,
                 "host_publish valid B70 request");
  
  return true;
}

void verify_numerics(const sb::ConstCompletionPayload& completion,
                     const GoldenReference& golden, const Pending& pending,
                     bool report) {
  require(completion.output_fp32.count ==
              pending.staged_tokens * sb::kHiddenSize,
          "completion output extent mismatch");
  float max_abs = 0.0F;
  float max_rel = 0.0F;
  for (std::uint32_t row = 0; row < pending.staged_tokens; ++row) {
    for (std::uint32_t column = 0; column < sb::kHiddenSize; ++column) {
      const float actual =
          completion.output_fp32[row * sb::kHiddenSize + column];
      const float expected = golden.output[column] * pending.output_scale;
      require(std::isfinite(actual), "completion contains non-finite output");
      const float absolute = std::abs(actual - expected);
      const float tolerance = kAtol + kRtol * std::abs(expected);
      max_abs = std::max(max_abs, absolute);
      if (expected != 0.0F) {
        max_rel = std::max(max_rel, absolute / std::abs(expected));
      }
      if (absolute > tolerance) {
        fail("numerical mismatch at seq=" +
             std::to_string(pending.ticket.request_seq) + " row=" +
             std::to_string(row) + " column=" + std::to_string(column) +
             " actual=" + std::to_string(actual) +
             " expected=" + std::to_string(expected));
      }
    }
  }
  if (report) {
    std::printf("Phase-2 verified seq=%llu M=%u max_abs=%.8e max_rel=%.8e\n",
                static_cast<unsigned long long>(pending.ticket.request_seq),
                pending.staged_tokens, static_cast<double>(max_abs),
                static_cast<double>(max_rel));
  }
}

void verify_statuses(const sb::ConstCompletionPayload& completion,
                     const Pending& pending) {
  const std::uint32_t positions = pending.staged_tokens * sb::kTopK;
  require(completion.route_status.count == positions,
          "completion route-status extent mismatch");
  require(completion.token_status.count == pending.staged_tokens,
          "completion token-status extent mismatch");
  for (std::uint32_t row = 0; row < pending.staged_tokens; ++row) {
    for (std::uint32_t route = 0; route < sb::kTopK; ++route) {
      const bool remote = pending.routes == Routes::duplicate_top8 || route == 0U;
      const std::uint8_t expected =
          sb::wire(remote ? sb::RouteStatus::contributed
                          : sb::RouteStatus::not_remote);
      require(completion.route_status[row * sb::kTopK + route] == expected,
              "route status mismatch at seq=" +
                  std::to_string(pending.ticket.request_seq) + " row=" +
                  std::to_string(row) + " route=" + std::to_string(route));
    }
    require(completion.token_status[row] ==
                sb::wire(sb::TokenStatus::contributed),
            "token status mismatch at seq=" +
                std::to_string(pending.ticket.request_seq) + " row=" +
                std::to_string(row));
  }
}

RequestTiming consume_request(sb::SharedRing& ring,
                              const GoldenReference& golden,
                              const Pending& pending, int provider_socket,
                              bool report = true) {
  const std::uint64_t deadline = monotonic_ns() + kOperationTimeoutNs;
  for (;;) {
    sb::ConstCompletionPayload completion{};
    sb::CompletionCode completion_code = sb::CompletionCode::unset;
    sb::ErrorCode error = sb::ErrorCode::internal;
    const sb::RingStatus status = ring.host_consume(
        pending.ticket, &completion, &completion_code, &error);
    if (status == sb::RingStatus::ok) {
      const std::uint64_t observed_ns = monotonic_ns();
      require(completion_code == sb::CompletionCode::ok_all &&
                  error == sb::ErrorCode::none,
              "successful host_consume returned wrong terminal identity");
      verify_numerics(completion, golden, pending, report);
      verify_statuses(completion, pending);
      sb::CompletionTiming timing{};
      require_status(ring.host_completion_timing(pending.ticket, &timing),
                     sb::RingStatus::ok,
                     "host_completion_timing completed B70 request");
      require(observed_ns >= pending.publication_monotonic_ns,
              "completion observation preceded request publication");
      require_status(ring.host_reclaim(pending.ticket), sb::RingStatus::ok,
                     "host_reclaim completed B70 request");
      return {observed_ns - pending.publication_monotonic_ns,
              timing.kernel_ns, timing.provider_total_ns};
    }
    if (status != sb::RingStatus::would_block) {
      fail("host_consume seq=" + std::to_string(pending.ticket.request_seq) +
           ": " + sb::status_message(status) + ", completion=" +
           std::to_string(static_cast<unsigned>(completion_code)) +
           ", error=" + std::to_string(static_cast<unsigned>(error)));
    }

    pollfd waits[2]{{ring.completion_eventfd(), POLLIN | POLLERR, 0},
                    {provider_socket, POLLHUP | POLLERR, 0}};
    const int poll_status = ::poll(waits, 2, kPollSliceMs);
    if (poll_status < 0) {
      if (errno == EINTR) {
        continue;
      }
      fail("completion poll failed");
    }
    if ((waits[0].revents & POLLIN) != 0) {
      require_status(ring.drain_completion_notifications(), sb::RingStatus::ok,
                     "drain completion eventfd");
    }
    if ((waits[0].revents & (POLLERR | POLLNVAL)) != 0 ||
        (waits[1].revents & (POLLHUP | POLLERR | POLLNVAL)) != 0) {
      fail("provider connection failed while waiting for completion");
    }
    if (monotonic_ns() >= deadline) {
      fail("timed out waiting for B70 completion seq=" +
           std::to_string(pending.ticket.request_seq));
    }
  }
}
struct Percentiles final {
  double p50_us = 0.0;
  double p95_us = 0.0;
  double p99_us = 0.0;
};

Percentiles summarize(std::vector<std::uint64_t>* samples) {
  require(samples != nullptr && !samples->empty(),
          "cannot summarize an empty timing series");
  std::sort(samples->begin(), samples->end());
  const auto percentile = [&](std::size_t numerator) {
    const std::size_t rank =
        (samples->size() * numerator + 99U) / 100U;
    return static_cast<double>((*samples)[rank - 1U]) / 1000.0;
  };
  return {percentile(50U), percentile(95U), percentile(99U)};
}

void print_percentiles(std::uint32_t staged_tokens, std::string_view metric,
                       const Percentiles& values) {
  std::printf(
      "phase2_benchmark M=%u metric=%.*s p50_us=%.3f p95_us=%.3f "
      "p99_us=%.3f\n",
      staged_tokens, static_cast<int>(metric.size()), metric.data(),
      values.p50_us, values.p95_us, values.p99_us);
}

std::uint64_t benchmark_ring(sb::SharedRing& ring,
                             const GoldenReference& golden,
                             int provider_socket,
                             std::uint64_t first_sequence) {
  constexpr std::size_t kWarmupIterations = 8;
  constexpr std::size_t kMeasuredIterations = 100;
  constexpr std::array<std::uint32_t, 4> kShapes{1U, 8U, 32U, 128U};
  std::uint64_t sequence = first_sequence;

  for (const std::uint32_t staged_tokens : kShapes) {
    std::vector<std::uint64_t> ring_wall;
    std::vector<std::uint64_t> kernel;
    std::vector<std::uint64_t> provider_total;
    std::vector<std::uint64_t> process_overhead;
    ring_wall.reserve(kMeasuredIterations);
    kernel.reserve(kMeasuredIterations);
    provider_total.reserve(kMeasuredIterations);
    process_overhead.reserve(kMeasuredIterations);

    for (std::size_t iteration = 0;
         iteration < kWarmupIterations + kMeasuredIterations; ++iteration) {
      Pending pending{};
      const RequestCase request{staged_tokens, staged_tokens, Routes::single,
                                1.0F, false};
      require(publish_request(ring, golden, sequence, request, &pending),
              "benchmark request was not published");
      ++sequence;
      const RequestTiming timing =
          consume_request(ring, golden, pending, provider_socket, false);
      require(timing.kernel_ns > 0U &&
                  timing.provider_total_ns >= timing.kernel_ns &&
                  timing.ring_wall_ns >= timing.provider_total_ns,
              "benchmark returned inconsistent timing stages");
      if (iteration < kWarmupIterations) {
        continue;
      }
      ring_wall.push_back(timing.ring_wall_ns);
      kernel.push_back(timing.kernel_ns);
      provider_total.push_back(timing.provider_total_ns);
      process_overhead.push_back(timing.ring_wall_ns -
                                 timing.provider_total_ns);
    }

    print_percentiles(staged_tokens, "publication_to_observation",
                      summarize(&ring_wall));
    print_percentiles(staged_tokens, "provider_total",
                      summarize(&provider_total));
    print_percentiles(staged_tokens, "kernel", summarize(&kernel));
    print_percentiles(staged_tokens, "ring_process_overhead",
                      summarize(&process_overhead));
  }
  return sequence;
}


void check_stale_sequence(sb::SharedRing& ring, std::uint64_t stale_sequence) {
  sb::RequestSpec stale{};
  stale.request_seq = stale_sequence;
  stale.scheduler_step = 1;
  stale.deadline_monotonic_ns = monotonic_ns() + kOperationTimeoutNs;
  stale.layer = 0;
  stale.num_batch_tokens = 1;
  stale.num_staged_tokens = 1;
  stale.num_routes = 1;
  sb::RingTicket ticket{};
  sb::RequestPayload payload{};
  require_status(ring.host_begin(stale, &ticket, &payload),
                 sb::RingStatus::invalid_argument,
                 "public API rejects stale request sequence");
}

void run_integration() {
  const std::string phase2_directory = executable_directory();
  const std::size_t slash = phase2_directory.find_last_of('/');
  require(slash != std::string::npos, "cannot locate repository root");
  const std::string repository_root = phase2_directory.substr(0, slash);
  const std::string provider_path = phase2_directory + "/b70_ring_provider";
  const std::string bank_path = repository_root + "/phase1/expert_bank.bin";
  const std::string golden_path = repository_root + "/phase1/golden_reference.bin";
  const GoldenReference golden = load_golden(golden_path);

  check_startup_identity_rejection(
      provider_path, bank_path, mismatched(kExpectedPlacementSha256),
      kExpectedWeightSha256, 15, "placement SHA-256 mismatch");
  check_startup_identity_rejection(
      provider_path, bank_path, kExpectedPlacementSha256,
      mismatched(kExpectedWeightSha256), 17, "weight SHA-256 mismatch");

  TemporaryDirectory temporary;
  const std::string socket_path = temporary.socket_path();
  ChildProcess child(spawn_provider(provider_path, socket_path, bank_path));

  const sb::RingIdentity identity = make_identity(child.pid());
  sb::SharedRing ring;
  const char* create_detail = nullptr;
  require_status(sb::SharedRing::create(identity, &ring, &create_detail),
                 sb::RingStatus::ok,
                 create_detail == nullptr ? "SharedRing::create"
                                          : create_detail);

  sb::RingIdentity stale_provider_identity = identity;
  ++stale_provider_identity.provider_generation;
  sb::SharedRing stale_provider_attach;
  const char* attach_detail = nullptr;
  require_status(sb::SharedRing::attach(
                     ring.fd(), ring.request_eventfd(), ring.completion_eventfd(),
                     stale_provider_identity, &stale_provider_attach,
                     &attach_detail),
                 sb::RingStatus::generation_mismatch,
                 "public API rejects stale provider generation");
  require(!stale_provider_attach.is_open(),
          "stale-provider-generation attach exposed an open mapping");

  sb::RingIdentity stale_placement_identity = identity;
  ++stale_placement_identity.placement_generation;
  sb::SharedRing stale_placement_attach;
  require_status(sb::SharedRing::attach(
                     ring.fd(), ring.request_eventfd(), ring.completion_eventfd(),
                     stale_placement_identity, &stale_placement_attach,
                     &attach_detail),
                 sb::RingStatus::generation_mismatch,
                 "public API rejects stale placement generation");
  require(!stale_placement_attach.is_open(),
          "stale-placement-generation attach exposed an open mapping");

  sb::RingIdentity stale_weight_identity = identity;
  ++stale_weight_identity.weight_generation;
  sb::SharedRing stale_weight_attach;
  require_status(sb::SharedRing::attach(
                     ring.fd(), ring.request_eventfd(), ring.completion_eventfd(),
                     stale_weight_identity, &stale_weight_attach, &attach_detail),
                 sb::RingStatus::generation_mismatch,
                 "public API rejects stale weight generation");
  require(!stale_weight_attach.is_open(),
          "stale-weight-generation attach exposed an open mapping");

  sb::RingIdentity wrong_pid_identity = identity;
  ++wrong_pid_identity.provider_pid;
  sb::SharedRing wrong_pid_attach;
  require_status(sb::SharedRing::attach(
                     ring.fd(), ring.request_eventfd(), ring.completion_eventfd(),
                     wrong_pid_identity, &wrong_pid_attach, &attach_detail),
                 sb::RingStatus::generation_mismatch,
                 "public API rejects mismatched provider PID identity");
  require(!wrong_pid_attach.is_open(),
          "wrong-provider-PID attach exposed an open mapping");

  Fd connection = connect_provider(socket_path, child);
  send_bootstrap(connection.get(), ring);
  const ctl::StartupReply startup =
      receive_exact<ctl::StartupReply>(connection.get(), "provider startup");
  require(exact_magic(startup.magic, ctl::kStartupMagic) &&
              startup.major == ctl::kProtocolMajor &&
              startup.minor == ctl::kProtocolMinor &&
              startup.bytes == sizeof(startup) && startup.startup_status == 0U &&
              startup.provider_status == 0U && startup.allocation_baseline > 0U &&
              all_zero(startup.reserved, sizeof(startup.reserved)),
          "provider startup reply failed exact validation");

  Pending ignored{};
  const RequestCase zero_remote{1, 1, Routes::zero, 0.0F, false};
  require(!publish_request(ring, golden, 1, zero_remote, &ignored),
          "zero-remote client request was published");
  require(ring.slot_state(0) == sb::SlotState::free,
          "zero-remote client request changed ring slot state");

  Pending single{};
  require(publish_request(ring, golden, 1,
                          RequestCase{1, 1, Routes::single, 1.0F, false},
                          &single),
          "M=1 golden request was not published");
  check_stale_sequence(ring, 1);
  consume_request(ring, golden, single, connection.get());

  Pending duplicate{};
  require(publish_request(ring, golden, 2,
                          RequestCase{8, 16, Routes::duplicate_top8, 0.0F, true},
                          &duplicate),
          "M=8 duplicate-top8 request was not published");
  consume_request(ring, golden, duplicate, connection.get());

  std::array<Pending, sb::kRingSlots> queued{};
  constexpr std::array<float, sb::kRingSlots> queue_weights{
      0.25F, 0.50F, 0.75F, 1.00F, 0.125F, 0.375F, 0.625F, 0.875F};
  for (std::uint32_t index = 0; index < sb::kRingSlots; ++index) {
    const std::uint64_t sequence = 3U + index;
    require(publish_request(
                ring, golden, sequence,
                RequestCase{1, 2, Routes::single, queue_weights[index], true},
                &queued[index]),
            "queued request was not published");
    require(queued[index].ticket.slot ==
                static_cast<std::uint32_t>((sequence - 1U) % sb::kRingSlots),
            "queued request used non-deterministic ring slot");
  }

  sb::RequestSpec blocked{};
  blocked.request_seq = 11;
  blocked.scheduler_step = 1011;
  blocked.deadline_monotonic_ns = monotonic_ns() + kOperationTimeoutNs;
  blocked.layer = 0;
  blocked.num_batch_tokens = 1;
  blocked.num_staged_tokens = 1;
  blocked.num_routes = 1;
  sb::RingTicket blocked_ticket{};
  sb::RequestPayload blocked_payload{};
  require_status(ring.host_begin(blocked, &blocked_ticket, &blocked_payload),
                 sb::RingStatus::would_block,
                 "ring forbids reuse of a live wrapped slot");

  for (const Pending& pending : queued) {
    consume_request(ring, golden, pending, connection.get());
  }

  Pending wrapped{};
  require(publish_request(ring, golden, 11,
                          RequestCase{1, 1, Routes::single, 1.0F, false},
                          &wrapped),
          "wrapped request was not published after reclamation");
  require(wrapped.ticket.slot == queued[0].ticket.slot,
          "wraparound did not reuse the reclaimed deterministic slot");
  require(wrapped.ticket.request_generation >
              queued[0].ticket.request_generation &&
              wrapped.ticket.output_buffer_version >
                  queued[0].ticket.output_buffer_version,
          "wraparound did not advance slot generations");
  consume_request(ring, golden, wrapped, connection.get());
  const std::uint64_t next_sequence =
      benchmark_ring(ring, golden, connection.get(), 12U);
  const std::uint64_t expected_dispatches = next_sequence - 1U;

  ctl::Control shutdown{};
  copy_magic(shutdown.magic, ctl::kControlMagic);
  shutdown.major = ctl::kProtocolMajor;
  shutdown.minor = ctl::kProtocolMinor;
  shutdown.bytes = sizeof(shutdown);
  shutdown.command = ctl::Command::shutdown;
  send_exact(connection.get(), shutdown, "clean shutdown request");
  const ctl::ShutdownReply shutdown_reply =
      receive_exact<ctl::ShutdownReply>(connection.get(),
                                        "clean shutdown reply");
  require(exact_magic(shutdown_reply.magic, ctl::kShutdownMagic) &&
              shutdown_reply.major == ctl::kProtocolMajor &&
              shutdown_reply.minor == ctl::kProtocolMinor &&
              shutdown_reply.bytes == sizeof(shutdown_reply) &&
              shutdown_reply.status == 0U && shutdown_reply.reserved0 == 0U &&
              shutdown_reply.dispatches == expected_dispatches &&
              shutdown_reply.allocation_baseline == startup.allocation_baseline &&
              shutdown_reply.allocation_final == startup.allocation_baseline &&
              all_zero(shutdown_reply.reserved,
                       sizeof(shutdown_reply.reserved)),
          "provider shutdown health/allocation invariant failed");
  connection.reset();

  const int wait_status = child.wait_bounded();
  require(WIFEXITED(wait_status) && WEXITSTATUS(wait_status) == 0,
          "provider did not exit successfully after clean shutdown");
  std::printf("Phase-2 B70 ring PASS\n");
}

}  // namespace

int main() {
  static_cast<void>(::signal(SIGPIPE, SIG_IGN));
  try {
    run_integration();
    return 0;
  } catch (const TestFailure& error) {
    std::fprintf(stderr, "Phase-2 B70 ring FAIL: %s\n", error.what());
    return 1;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "Phase-2 B70 ring ERROR: %s\n", error.what());
    return 2;
  } catch (...) {
    std::fprintf(stderr, "Phase-2 B70 ring ERROR: unknown failure\n");
    return 3;
  }
}
