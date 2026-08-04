#include "math_cases.hpp"
#include "reference_fixture.hpp"

#include "../phase1/b70_provider.hpp"
#include "../phase2/b70_ring_control.hpp"
#include "../phase2/shared_ring.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <poll.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

namespace sb1 = shooting_brake::phase1;
namespace sb2 = shooting_brake::phase2;
namespace ctl = shooting_brake::phase2::b70_control;
namespace sb3 = shooting_brake::phase3;
namespace {

constexpr int kPollSliceMs = 250;
constexpr std::uint64_t kOperationTimeoutNs = 120'000'000'000ULL;
constexpr std::uint64_t kPlacementGeneration = 29U;
constexpr std::uint64_t kWeightGeneration = 31U;
constexpr float kPoison = -0x1.5a5a5ap+17F;

class TestFailure final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

[[noreturn]] void fail(const std::string& message) { throw TestFailure(message); }

void require(bool condition, std::string_view message) {
  if (!condition) {
    fail(std::string(message));
  }
}

void require_status(sb2::RingStatus actual, sb2::RingStatus expected,
                    std::string_view operation) {
  if (actual != expected) {
    fail(std::string(operation) + ": expected " + sb2::status_message(expected) +
         ", got " + sb2::status_message(actual));
  }
}

std::uint64_t monotonic_ns() noexcept {
  timespec value{};
  if (::clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
    return 0;
  }
  return static_cast<std::uint64_t>(value.tv_sec) * 1'000'000'000ULL +
         static_cast<std::uint64_t>(value.tv_nsec);
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

bool exact_magic(const char* actual,
                 const std::array<char, 8>& expected) noexcept {
  return std::memcmp(actual, expected.data(), expected.size()) == 0;
}

void copy_magic(char* destination,
                const std::array<char, 8>& source) noexcept {
  std::memcpy(destination, source.data(), source.size());
}

class Fd final {
 public:
  Fd() noexcept = default;
  explicit Fd(int fd) noexcept : fd_(fd) {}
  ~Fd() noexcept { reset(); }
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
  int release() noexcept { return std::exchange(fd_, -1); }
  void reset(int replacement = -1) noexcept {
    if (fd_ >= 0) {
      static_cast<void>(::close(fd_));
    }
    fd_ = replacement;
  }

 private:
  int fd_ = -1;
};

class TemporaryDirectory final {
 public:
  TemporaryDirectory() {
    std::array<char, 64> pattern{};
    constexpr std::string_view base = "/tmp/shooting-brake-phase3-XXXXXX";
    std::memcpy(pattern.data(), base.data(), base.size());
    char* created = ::mkdtemp(pattern.data());
    if (created == nullptr) {
      fail("mkdtemp failed: " + std::string(std::strerror(errno)));
    }
    path_ = created;
    socket_path_ = path_ + "/provider.sock";
  }
  ~TemporaryDirectory() noexcept {
    if (!socket_path_.empty()) {
      static_cast<void>(::unlink(socket_path_.c_str()));
    }
    if (!path_.empty()) {
      static_cast<void>(::rmdir(path_.c_str()));
    }
  }
  TemporaryDirectory(const TemporaryDirectory&) = delete;
  TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;
  const std::string& socket_path() const noexcept { return socket_path_; }

 private:
  std::string path_;
  std::string socket_path_;
};

class ChildProcess final {
 public:
  ChildProcess() noexcept = default;
  explicit ChildProcess(pid_t pid) noexcept : pid_(pid) {}
  ~ChildProcess() noexcept {
    if (pid_ > 0 && !reaped_) {
      static_cast<void>(::kill(pid_, SIGKILL));
      int ignored = 0;
      while (::waitpid(pid_, &ignored, 0) < 0 && errno == EINTR) {
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
        fail("provider child did not exit before deadline");
      }
      timespec pause{0, 20'000'000};
      while (::nanosleep(&pause, &pause) != 0 && errno == EINTR) {
      }
    }
  }

 private:
  pid_t pid_ = -1;
  bool reaped_ = false;
  int status_ = 0;
};

constexpr sb2::Fingerprint placement_fingerprint() noexcept {
  return {{0x4bU, 0x37U, 0x93U, 0x0cU, 0x7aU, 0x8bU, 0x3fU, 0x32U,
           0xbdU, 0xfeU, 0x7eU, 0xc3U, 0x40U, 0xbeU, 0xdfU, 0x45U,
           0x62U, 0xe0U, 0x04U, 0xc2U, 0x7eU, 0x5fU, 0x9fU, 0xa4U,
           0xe5U, 0x52U, 0x07U, 0xf2U, 0xfbU, 0x0cU, 0x95U, 0xa0U}};
}

int hex_nibble(char value) noexcept {
  if (value >= '0' && value <= '9') {
    return value - '0';
  }
  return value - 'a' + 10;
}

sb2::Fingerprint bank_fingerprint() noexcept {
  constexpr std::string_view hash =
      "0ce6377ba3c9848da42b6063574ea884052d2e0f5e605d86d1684a1e5826e8db";
  sb2::Fingerprint result{};
  for (std::size_t index = 0; index < sizeof(result.bytes); ++index) {
    result.bytes[index] = static_cast<std::uint8_t>(
        (hex_nibble(hash[index * 2U]) << 4) |
        hex_nibble(hash[index * 2U + 1U]));
  }
  return result;
}

void verify_runtime_bank(const std::string& path) {
  constexpr std::uint64_t expected_bytes = 14'495'580'220ULL;
  const std::array<std::uint8_t, 32> digest =
      sb3::sha256_file(path, expected_bytes);
  const sb2::Fingerprint expected = bank_fingerprint();
  require(std::memcmp(digest.data(), expected.bytes, digest.size()) == 0,
          "runtime expert bank SHA256 does not match reference fixture");
}

std::string fingerprint_hex(const sb2::Fingerprint& fingerprint) {
  constexpr char digits[] = "0123456789abcdef";
  std::string result(sizeof(fingerprint.bytes) * 2U, '0');
  for (std::size_t index = 0; index < sizeof(fingerprint.bytes); ++index) {
    result[index * 2U] = digits[fingerprint.bytes[index] >> 4U];
    result[index * 2U + 1U] = digits[fingerprint.bytes[index] & 0x0fU];
  }
  return result;
}

sb2::RingIdentity make_identity(pid_t pid) noexcept {
  sb2::RingIdentity identity{};
  identity.ring_generation = 301U;
  identity.provider_generation = 73U;
  identity.placement_generation = kPlacementGeneration;
  identity.weight_generation = kWeightGeneration;
  identity.provider_pid = static_cast<std::uint64_t>(pid);
  identity.placement_sha256 = placement_fingerprint();
  identity.weight_sha256 = bank_fingerprint();
  for (std::size_t index = 0; index < sizeof(identity.provider_nonce.bytes);
       ++index) {
    identity.provider_nonce.bytes[index] =
        static_cast<std::uint8_t>(0x10U + index);
    identity.ring_nonce.bytes[index] =
        static_cast<std::uint8_t>(0x90U + index);
  }
  return identity;
}

pid_t spawn_provider(const std::string& provider_path,
                     const std::string& socket_path,
                     const std::string& bank_path) {
  const std::string placement_generation =
      std::to_string(kPlacementGeneration);
  const std::string weight_generation = std::to_string(kWeightGeneration);
  const std::string placement_sha256 =
      fingerprint_hex(placement_fingerprint());
  const std::string weight_sha256 = fingerprint_hex(bank_fingerprint());
  const pid_t pid = ::fork();
  if (pid < 0) {
    fail("fork provider failed: " + std::string(std::strerror(errno)));
  }
  if (pid == 0) {
    ::execl(provider_path.c_str(), provider_path.c_str(), "--socket",
            socket_path.c_str(), "--bank", bank_path.c_str(),
            "--expected-placement-generation", placement_generation.c_str(),
            "--expected-weight-generation", weight_generation.c_str(),
            "--expected-placement-sha256", placement_sha256.c_str(),
            "--expected-weight-sha256", weight_sha256.c_str(),
            "--resident-experts", "255,0,7,63,127,191,254,1", "--quiet",
            static_cast<char*>(nullptr));
    _exit(127);
  }
  return pid;
}

Fd connect_provider(const std::string& path, ChildProcess& child) {
  const std::uint64_t deadline = monotonic_ns() + kOperationTimeoutNs;
  for (;;) {
    Fd socket(::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0));
    if (socket.get() < 0) {
      fail("cannot create provider control socket");
    }
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    require(path.size() < sizeof(address.sun_path), "provider socket path too long");
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
      fail("provider connect failed: " +
           std::string(std::strerror(connect_error)));
    }
    if (monotonic_ns() >= deadline) {
      fail("provider bootstrap connection timed out");
    }
    timespec pause{0, 20'000'000};
    while (::nanosleep(&pause, &pause) != 0 && errno == EINTR) {
    }
  }
}

void send_bootstrap(int socket, const sb2::SharedRing& ring) {
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
  require(header != nullptr, "cannot form bootstrap SCM_RIGHTS message");
  header->cmsg_level = SOL_SOCKET;
  header->cmsg_type = SCM_RIGHTS;
  header->cmsg_len = CMSG_LEN(sizeof(descriptors));
  std::memcpy(CMSG_DATA(header), descriptors.data(), sizeof(descriptors));
  require(::sendmsg(socket, &message, MSG_NOSIGNAL) ==
              static_cast<ssize_t>(sizeof(bootstrap)),
          "bootstrap send was not exact");
}

void wait_readable(int fd, std::string_view operation) {
  const std::uint64_t deadline = monotonic_ns() + kOperationTimeoutNs;
  for (;;) {
    pollfd wait{fd, POLLIN | POLLHUP | POLLERR, 0};
    const int result = ::poll(&wait, 1, kPollSliceMs);
    if (result < 0) {
      if (errno == EINTR) {
        continue;
      }
      fail(std::string(operation) + ": poll failed");
    }
    if ((wait.revents & POLLIN) != 0) {
      return;
    }
    if ((wait.revents & (POLLHUP | POLLERR | POLLNVAL)) != 0) {
      fail(std::string(operation) + ": peer closed");
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
          std::string(operation) + ": packet size mismatch");
  return packet;
}

template <typename T>
void send_exact(int socket, const T& packet, std::string_view operation) {
  require(::send(socket, &packet, sizeof(packet), MSG_NOSIGNAL) ==
              static_cast<ssize_t>(sizeof(packet)),
          std::string(operation) + ": short send");
}

struct RawWireCheck final {
  const sb2::RingTicket* ticket = nullptr;
  const sb2::RingIdentity* identity = nullptr;
  const sb3::StagedCase* staged = nullptr;
  bool poison = false;
  bool expect_success = false;
  bool ok = false;
};

sb2::RingStatus inspect_wire(void* mapping, std::uint64_t bytes,
                             void* opaque) noexcept {
  auto* check = static_cast<RawWireCheck*>(opaque);
  if (mapping == nullptr || check == nullptr || check->ticket == nullptr ||
      check->identity == nullptr || check->staged == nullptr ||
      bytes != sb2::kMappingBytes || check->ticket->slot >= sb2::kRingSlots) {
    return sb2::RingStatus::invalid_argument;
  }
  auto* base = static_cast<std::byte*>(mapping);
  const std::uint32_t slot = check->ticket->slot;
  auto* completion = reinterpret_cast<sb2::CompletionDescriptor*>(
      base + sb2::kCompletionDescriptorOffset +
      static_cast<std::uint64_t>(slot) * sizeof(sb2::CompletionDescriptor));
  float* output = reinterpret_cast<float*>(
      base + sb2::kPayloadOffset +
      static_cast<std::uint64_t>(slot) * sb2::kPayloadStride +
      sb2::kOutputRelativeOffset);
  if (check->poison) {
    std::fill_n(output,
                static_cast<std::size_t>(check->staged->rows) *
                    sb2::kHiddenSize,
                kPoison);
    check->ok = true;
    return sb2::RingStatus::ok;
  }
  const bool identity_ok =
      completion->request_seq == check->ticket->request_seq &&
      completion->request_generation == check->ticket->request_generation &&
      completion->output_buffer_version ==
          check->ticket->output_buffer_version &&
      completion->ring_generation == check->identity->ring_generation &&
      completion->provider_generation == check->identity->provider_generation &&
      completion->placement_generation ==
          check->identity->placement_generation &&
      completion->weight_generation == check->identity->weight_generation &&
      completion->layer == check->staged->layer &&
      completion->num_batch_tokens == check->staged->full_batch &&
      completion->num_staged_tokens == check->staged->rows &&
      completion->num_routes == check->staged->remote_routes &&
      std::memcmp(completion->provider_nonce.bytes,
                  check->identity->provider_nonce.bytes,
                  sizeof(completion->provider_nonce.bytes)) == 0 &&
      std::memcmp(completion->placement_sha256.bytes,
                  check->identity->placement_sha256.bytes,
                  sizeof(completion->placement_sha256.bytes)) == 0 &&
      std::memcmp(completion->weight_sha256.bytes,
                  check->identity->weight_sha256.bytes,
                  sizeof(completion->weight_sha256.bytes)) == 0;
  if (!identity_ok) {
    return sb2::RingStatus::stale_completion;
  }
  const auto* route_status = reinterpret_cast<const std::uint8_t*>(
      base + completion->route_status_offset);
  const auto* token_status = reinterpret_cast<const std::uint8_t*>(
      base + completion->token_status_offset);
  for (std::size_t index = 0; index < check->staged->mask().size(); ++index) {
    const std::uint8_t expected = check->staged->mask()[index] != 0U
                                      ? sb2::wire(check->expect_success
                                                      ? sb2::RouteStatus::contributed
                                                      : sb2::RouteStatus::not_contributed)
                                      : sb2::wire(sb2::RouteStatus::not_remote);
    if (route_status[index] != expected) {
      return sb2::RingStatus::invalid_payload;
    }
  }
  for (std::uint32_t row = 0; row < check->staged->rows; ++row) {
    const std::uint8_t expected =
        sb2::wire(check->expect_success ? sb2::TokenStatus::contributed
                                       : sb2::TokenStatus::not_contributed);
    if (token_status[row] != expected) {
      return sb2::RingStatus::invalid_payload;
    }
  }
  if (!check->expect_success) {
    if (completion->output_bytes != 0U) {
      return sb2::RingStatus::invalid_payload;
    }
    for (std::size_t index = 0;
         index < static_cast<std::size_t>(check->staged->rows) *
                     sb2::kHiddenSize;
         ++index) {
      if (std::bit_cast<std::uint32_t>(output[index]) !=
          std::bit_cast<std::uint32_t>(kPoison)) {
        return sb2::RingStatus::invalid_payload;
      }
    }
  }
  check->ok = true;
  return sb2::RingStatus::ok;
}

class CompletionLease final {
 public:
  CompletionLease(sb2::SharedRing& ring, sb2::RingTicket ticket) noexcept
      : ring_(&ring), ticket_(ticket) {}
  ~CompletionLease() noexcept {
    if (consumed_) {
      static_cast<void>(ring_->host_reclaim(ticket_));
    }
  }
  CompletionLease(const CompletionLease&) = delete;
  CompletionLease& operator=(const CompletionLease&) = delete;
  const sb2::RingTicket& ticket() const noexcept { return ticket_; }
  void mark_consumed() noexcept { consumed_ = true; }
  void reclaim() {
    require_status(ring_->host_reclaim(ticket_), sb2::RingStatus::ok,
                   "host_reclaim completion lease");
    consumed_ = false;
  }

 private:
  sb2::SharedRing* ring_;
  sb2::RingTicket ticket_{};
  bool consumed_ = false;
};

class WireProviderSession final {
 public:
  WireProviderSession(const std::string& provider_path,
                      const std::string& bank_path)
      : child_(spawn_provider(provider_path, temporary_.socket_path(), bank_path)),
        identity_(make_identity(child_.pid())) {
    const char* detail = nullptr;
    require_status(sb2::SharedRing::create(identity_, &ring_, &detail),
                   sb2::RingStatus::ok,
                   detail == nullptr ? "SharedRing::create" : detail);
    connection_ = connect_provider(temporary_.socket_path(), child_);
    send_bootstrap(connection_.get(), ring_);
    startup_ = receive_exact<ctl::StartupReply>(connection_.get(),
                                                "provider startup");
    require(exact_magic(startup_.magic, ctl::kStartupMagic) &&
                startup_.major == ctl::kProtocolMajor &&
                startup_.minor == ctl::kProtocolMinor &&
                startup_.bytes == sizeof(startup_) &&
                startup_.startup_status == 0U && startup_.provider_status == 0U &&
                startup_.allocation_baseline > 0U &&
                all_zero(startup_.reserved, sizeof(startup_.reserved)),
            "provider startup reply validation failed");
  }
  ~WireProviderSession() noexcept = default;
  WireProviderSession(const WireProviderSession&) = delete;
  WireProviderSession& operator=(const WireProviderSession&) = delete;

  std::uint64_t next_sequence() const noexcept { return next_sequence_; }

  void arm_fault(std::uint64_t sequence) {
    ctl::Control control{};
    copy_magic(control.magic, ctl::kControlMagic);
    control.major = ctl::kProtocolMajor;
    control.minor = ctl::kProtocolMinor;
    control.bytes = sizeof(control);
    control.command = ctl::Command::arm_test_fault;
    control.fault_point = ctl::FaultPoint::after_kernel_before_copyout;
    control.fault_argument = ctl::FaultArgument::none;
    control.sequence = sequence;
    send_exact(connection_.get(), control, "arm provider fault");
    const ctl::FaultAck ack =
        receive_exact<ctl::FaultAck>(connection_.get(), "provider fault ack");
    require(exact_magic(ack.magic, ctl::kFaultAckMagic) &&
                ack.major == ctl::kProtocolMajor &&
                ack.minor == ctl::kProtocolMinor && ack.bytes == sizeof(ack) &&
                ack.status == ctl::FaultAckStatus::armed &&
                ack.provider_status ==
                    static_cast<std::uint32_t>(sb1::ProviderStatus::ok) &&
                ack.fault_point == control.fault_point &&
                ack.fault_argument == control.fault_argument &&
                ack.sequence == sequence &&
                all_zero(ack.reserved, sizeof(ack.reserved)),
            "fault acknowledgment failed exact validation");
  }

  void execute_success(const sb3::StagedCase& staged,
                       const sb3::OracleResult& oracle,
                       float* exact_copy = nullptr) {
    require(staged.rows > 0U && staged.rows == oracle.rows,
            "success case has inconsistent staged/oracle rows");
    CompletionLease lease = publish(staged);
    sb2::ConstCompletionPayload completion{};
    sb2::CompletionCode code = sb2::CompletionCode::unset;
    sb2::ErrorCode error = sb2::ErrorCode::internal;
    wait_completion(lease.ticket(), &completion, &code, &error,
                    sb2::RingStatus::ok);
    lease.mark_consumed();
    require(code == sb2::CompletionCode::ok_all &&
                error == sb2::ErrorCode::none,
            "successful completion returned wrong code/error");
    require(completion.output_fp32.count == staged.rows * sb2::kHiddenSize &&
                completion.route_status.count ==
                    staged.rows * sb2::kTopK &&
                completion.token_status.count == staged.rows,
            "successful completion exposed wrong exact extents");
    for (std::uint32_t row = 0; row < staged.rows; ++row) {
      for (std::uint32_t column = 0; column < sb2::kHiddenSize; ++column) {
        const std::size_t index =
            static_cast<std::size_t>(row) * sb2::kHiddenSize + column;
        const double actual = completion.output_fp32[index];
        const double expected = oracle.expected[index];
        require(std::isfinite(actual), "completion contains non-finite output");
        if (std::abs(actual - expected) >
            sb3::output_budget(oracle, row, column)) {
          fail("wire numerical mismatch at sequence=" +
               std::to_string(lease.ticket().request_seq) + " row=" +
               std::to_string(row) + " column=" + std::to_string(column));
        }
        if (exact_copy != nullptr) {
          exact_copy[index] = completion.output_fp32[index];
        }
      }
    }
    RawWireCheck raw{&lease.ticket(), &identity_, &staged, false, true, false};
    require_status(ring_.visit_mapping(inspect_wire, &raw), sb2::RingStatus::ok,
                   "inspect successful completion wire identity/status");
    require(raw.ok, "successful completion raw inspection did not run");
    lease.reclaim();
  }

  void execute_failure(const sb3::StagedCase& staged, bool arm = true) {
    require(staged.rows > 0U, "fault case has no staged rows");
    const std::uint64_t sequence = next_sequence_;
    if (arm) {
      arm_fault(sequence);
    }
    CompletionLease lease = publish(staged, true);

    std::array<float, sb2::kHiddenSize> client_poison{};
    client_poison.fill(kPoison);
    sb2::ConstCompletionPayload completion{
        {client_poison.data(), static_cast<std::uint32_t>(client_poison.size())},
        {}, {}};
    sb2::CompletionCode code = sb2::CompletionCode::unset;
    sb2::ErrorCode error = sb2::ErrorCode::internal;
    wait_completion(lease.ticket(), &completion, &code, &error,
                    sb2::RingStatus::device_failure);
    lease.mark_consumed();
    require(code == sb2::CompletionCode::execution_failed &&
                error == sb2::ErrorCode::core_device &&
                !completion.output_fp32 && !completion.route_status &&
                !completion.token_status,
            "device failure exposed payload or wrong terminal identity");
    for (float value : client_poison) {
      require(std::bit_cast<std::uint32_t>(value) ==
                  std::bit_cast<std::uint32_t>(kPoison),
              "failed client path modified poison output");
    }
    RawWireCheck raw{&lease.ticket(), &identity_, &staged, false, false, false};
    require_status(ring_.visit_mapping(inspect_wire, &raw), sb2::RingStatus::ok,
                   "inspect failed completion wire status/poison");
    require(raw.ok, "failed completion raw inspection did not run");
    lease.reclaim();
  }

  void prove_zero_remote_bypass(const sb3::StagedCase& staged) {
    require(staged.rows == 0U && staged.remote_routes == 0U,
            "zero-remote materialization was not empty");
    for (std::uint32_t slot = 0; slot < sb2::kRingSlots; ++slot) {
      require(ring_.slot_state(slot) == sb2::SlotState::free,
              "zero-remote bypass observed a published ring slot");
    }
  }

  void shutdown(std::uint64_t expected_dispatches) {
    ctl::Control control{};
    copy_magic(control.magic, ctl::kControlMagic);
    control.major = ctl::kProtocolMajor;
    control.minor = ctl::kProtocolMinor;
    control.bytes = sizeof(control);
    control.command = ctl::Command::shutdown;
    control.fault_point = ctl::FaultPoint::none;
    control.fault_argument = ctl::FaultArgument::none;
    send_exact(connection_.get(), control, "provider shutdown");
    const ctl::ShutdownReply reply = receive_exact<ctl::ShutdownReply>(
        connection_.get(), "provider shutdown reply");
    require(exact_magic(reply.magic, ctl::kShutdownMagic) &&
                reply.major == ctl::kProtocolMajor &&
                reply.minor == ctl::kProtocolMinor &&
                reply.bytes == sizeof(reply) && reply.status == 0U &&
                reply.reserved0 == 0U &&
                reply.dispatches == expected_dispatches &&
                reply.allocation_baseline == startup_.allocation_baseline &&
                reply.allocation_final == startup_.allocation_baseline &&
                all_zero(reply.reserved, sizeof(reply.reserved)),
            "shutdown dispatch/allocation invariant failed");
    connection_.reset();
    const int status = child_.wait_bounded();
    require(WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "provider did not exit cleanly");
  }

 private:
  CompletionLease publish(const sb3::StagedCase& staged,
                          bool poison_output = false) {
    sb2::RequestSpec spec{};
    spec.request_seq = next_sequence_++;
    spec.scheduler_step = 100'000U + spec.request_seq;
    spec.deadline_monotonic_ns = monotonic_ns() + kOperationTimeoutNs;
    spec.layer = staged.layer;
    spec.num_batch_tokens = staged.full_batch;
    spec.num_staged_tokens = staged.rows;
    spec.num_routes = staged.remote_routes;
    sb2::RingTicket ticket{};
    sb2::RequestPayload payload{};
    require_status(ring_.host_begin(spec, &ticket, &payload),
                   sb2::RingStatus::ok, "host_begin mathematics request");
    require(payload.activation_fp16.count == staged.activations().size() &&
                payload.canonical_ids.count ==
                    staged.canonical_ids().size() &&
                payload.routing_weights.count ==
                    staged.route_weights().size() &&
                payload.remote_mask.count == staged.mask().size() &&
                payload.token_row_map.count == staged.row_map().size() &&
                payload.canonical_route_positions.count ==
                    staged.positions().size(),
            "request payload exact extents mismatch");
    std::memcpy(payload.activation_fp16.data, staged.activations().data(),
                staged.activations().size_bytes());
    std::memcpy(payload.canonical_ids.data, staged.canonical_ids().data(),
                staged.canonical_ids().size_bytes());
    std::memcpy(payload.routing_weights.data, staged.route_weights().data(),
                staged.route_weights().size_bytes());
    std::memcpy(payload.remote_mask.data, staged.mask().data(),
                staged.mask().size_bytes());
    std::memcpy(payload.token_row_map.data, staged.row_map().data(),
                staged.row_map().size_bytes());
    std::memcpy(payload.canonical_route_positions.data,
                staged.positions().data(), staged.positions().size_bytes());
    if (poison_output) {
      RawWireCheck poison{&ticket, &identity_, &staged, true, false, false};
      require_status(ring_.visit_mapping(inspect_wire, &poison),
                     sb2::RingStatus::ok, "poison failure output region");
      require(poison.ok, "failure output poison was not installed");
    }
    require_status(ring_.host_publish(ticket), sb2::RingStatus::ok,
                   "host_publish mathematics request");
    return CompletionLease(ring_, ticket);
  }

  void wait_completion(const sb2::RingTicket& ticket,
                       sb2::ConstCompletionPayload* completion,
                       sb2::CompletionCode* code, sb2::ErrorCode* error,
                       sb2::RingStatus expected) {
    const std::uint64_t deadline = monotonic_ns() + kOperationTimeoutNs;
    for (;;) {
      const sb2::RingStatus status =
          ring_.host_consume(ticket, completion, code, error);
      if (status == expected) {
        return;
      }
      if (status != sb2::RingStatus::would_block) {
        fail("host_consume returned " +
             std::string(sb2::status_message(status)) + " instead of " +
             sb2::status_message(expected));
      }
      pollfd waits[2]{{ring_.completion_eventfd(), POLLIN | POLLERR, 0},
                      {connection_.get(), POLLHUP | POLLERR, 0}};
      const int result = ::poll(waits, 2, kPollSliceMs);
      if (result < 0) {
        if (errno == EINTR) {
          continue;
        }
        fail("completion poll failed");
      }
      if ((waits[0].revents & POLLIN) != 0) {
        require_status(ring_.drain_completion_notifications(),
                       sb2::RingStatus::ok,
                       "drain completion notifications");
      }
      if ((waits[0].revents & (POLLERR | POLLNVAL)) != 0 ||
          (waits[1].revents & (POLLHUP | POLLERR | POLLNVAL)) != 0) {
        fail("provider connection failed while awaiting completion");
      }
      if (monotonic_ns() >= deadline) {
        fail("mathematics request timed out");
      }
    }
  }

  // Destruction is reverse declaration order: connection, ring, child, temp.
  TemporaryDirectory temporary_;
  ChildProcess child_;
  sb2::RingIdentity identity_{};
  sb2::SharedRing ring_;
  Fd connection_;
  ctl::StartupReply startup_{};
  std::uint64_t next_sequence_ = 1U;
};

void run_trusted_bootstrap_negative(const std::string& provider_path,
                                    const std::string& bank_path,
                                    bool stale_placement) {
  TemporaryDirectory temporary;
  ChildProcess child(
      spawn_provider(provider_path, temporary.socket_path(), bank_path));
  sb2::RingIdentity identity = make_identity(child.pid());
  if (stale_placement) {
    ++identity.placement_generation;
  } else {
    ++identity.weight_generation;
  }
  sb2::SharedRing ring;
  const char* detail = nullptr;
  require_status(sb2::SharedRing::create(identity, &ring, &detail),
                 sb2::RingStatus::ok,
                 detail == nullptr ? "create stale bootstrap ring" : detail);
  Fd connection = connect_provider(temporary.socket_path(), child);
  send_bootstrap(connection.get(), ring);
  connection.reset();
  const int status = child.wait_bounded();
  require(WIFEXITED(status) && WEXITSTATUS(status) == 12,
          stale_placement
              ? "provider accepted stale trusted placement bootstrap"
              : "provider accepted stale trusted weight bootstrap");
}

void run_direct_shape_negatives(const std::string& bank_path) {
  sb1::B70Provider provider;
  sb1::ProviderConfig config{};
  config.max_batch = 128U;
  config.top_k = sb2::kTopK;
  config.generation = 911U;
  require(provider.load(bank_path, config) == sb1::ProviderStatus::ok,
          "direct-negative provider load failed");
  const sb1::Health baseline = provider.health();
  require(baseline.loaded && !baseline.pending && baseline.allocations > 0U,
          "direct-negative provider baseline is invalid");
  std::vector<sycl::half> hidden(129U * sb2::kHiddenSize);
  std::vector<std::int32_t> ids(129U * sb2::kTopK, -1);
  std::vector<float> weights(129U * sb2::kTopK, 0.0F);
  require(provider.issue(config.generation, 1U, 0U, hidden.data(), ids.data(),
                         weights.data(), 0U) ==
              sb1::ProviderStatus::invalid_argument,
          "direct provider accepted M=0");
  require(provider.issue(config.generation, 1U, 0U, hidden.data(), ids.data(),
                         weights.data(), 129U) ==
              sb1::ProviderStatus::invalid_argument,
          "direct provider accepted M=129");
  const sb1::Health final = provider.health();
  require(final.dispatches == baseline.dispatches &&
              final.allocations == baseline.allocations && !final.pending,
          "direct shape rejection changed dispatch/allocation state");
  provider.shutdown();
}

void report_artifact_matrix(const sb3::ReferenceFixture& fixture) {
  const std::vector<sb3::ArtifactMetrics> matrix =
      sb3::compute_artifact_metrics(fixture);
  require(matrix.size() ==
              sb3::kReferenceLayers * sb3::kReferenceExperts + 1U,
          "artifact matrix plus aggregate extent mismatch");
  std::printf("Phase-3 BF16-source-vs-NVFP4 artifact matrix BEGIN\n");
  for (const sb3::ArtifactMetrics& metric : matrix) {
    require(std::isfinite(metric.max_absolute) && std::isfinite(metric.rms) &&
                std::isfinite(metric.source_rms) &&
                std::isfinite(metric.relative_rms) &&
                std::isfinite(metric.cosine) && metric.source_rms > 0.0 &&
                metric.relative_rms <= 0.18 && metric.cosine >= 0.98,
            "artifact matrix exceeds frozen relative-RMSE/cosine gate");
    if (metric.expert < 0) {
      std::printf(
          "phase3_artifact aggregate max_abs=%.12e rms=%.12e "
          "source_rms=%.12e relative_rms=%.12e cosine=%.12e\n",
          metric.max_absolute, metric.rms, metric.source_rms,
          metric.relative_rms, metric.cosine);
    } else {
      std::printf(
          "phase3_artifact layer=%u expert=%d max_abs=%.12e rms=%.12e "
          "source_rms=%.12e relative_rms=%.12e cosine=%.12e\n",
          metric.layer, metric.expert, metric.max_absolute, metric.rms,
          metric.source_rms, metric.relative_rms, metric.cosine);
    }
  }
  std::printf("Phase-3 BF16-source-vs-NVFP4 artifact matrix END\n");
}

void run(const std::string& executable_dir) {
  const std::size_t slash = executable_dir.find_last_of('/');
  require(slash != std::string::npos, "cannot locate repository root");
  const std::string root = executable_dir.substr(0, slash);
  const std::string fixture_path = executable_dir + "/reference_fixture.bin";
  const std::string bank_path = root + "/phase1/expert_bank.bin";
  const std::string provider_path = executable_dir + "/b70_ring_provider";
  const std::string fault_provider_path =
      executable_dir + "/b70_ring_provider_fault";

  const sb3::ReferenceFixture fixture =
      sb3::ReferenceFixture::open(fixture_path);
  std::printf("Phase-3 fixture schema/identity/finite/uniqueness PASS\n");
  report_artifact_matrix(fixture);
  verify_runtime_bank(bank_path);
  std::printf("Phase-3 runtime expert-bank size/SHA256 binding PASS\n");

  run_trusted_bootstrap_negative(provider_path, bank_path, true);
  run_trusted_bootstrap_negative(provider_path, bank_path, false);
  std::printf("Phase-3 trusted placement/weight bootstrap negatives PASS\n");

  std::uint64_t dispatches = 0;
  {
    WireProviderSession session(provider_path, bank_path);
    const sb3::StagedCase zero =
        sb3::stage_remote_routes(sb3::make_zero_remote_case(), fixture);
    session.prove_zero_remote_bypass(zero);
    std::printf("Phase-3 zero-remote no-publication/zero-dispatch PASS\n");

    const std::vector<sb3::CanonicalCase> sweep = sb3::make_one_route_sweep();
    require(sweep.size() == 128U, "one-route sweep extent mismatch");
    for (const sb3::CanonicalCase& test_case : sweep) {
      const sb3::StagedCase staged =
          sb3::stage_remote_routes(test_case, fixture);
      const sb3::OracleResult oracle =
          sb3::compute_nvfp4_oracle(test_case, fixture);
      session.execute_success(staged, oracle);
      ++dispatches;
    }
    std::printf("Phase-3 canonical one-route wire sweep M=1..128 PASS\n");

    const sb3::CanonicalCase all_remote = sb3::make_all_remote_case();
    require(sb3::small_route_is_sensitivity_checked(all_remote, fixture),
            "2^-12 route is not observable beyond the comparison budget");
    session.execute_success(sb3::stage_remote_routes(all_remote, fixture),
                            sb3::compute_nvfp4_oracle(all_remote, fixture));
    ++dispatches;
    std::printf("Phase-3 all-remote duplicate/unsorted/boundary/2^-12 M=4 PASS\n");

    const sb3::CanonicalCase mixed = sb3::make_mixed_case(false);
    const sb3::CanonicalCase paired = sb3::make_mixed_case(true);
    require(sb3::remote_materialization_equal(mixed, paired, fixture),
            "local-only route perturbation changed remote materialization");
    std::array<float, 4U * sb2::kHiddenSize> first_partial{};
    std::array<float, 4U * sb2::kHiddenSize> second_partial{};
    const sb3::OracleResult mixed_oracle =
        sb3::compute_nvfp4_oracle(mixed, fixture);
    session.execute_success(sb3::stage_remote_routes(mixed, fixture),
                            mixed_oracle, first_partial.data());
    ++dispatches;
    session.execute_success(sb3::stage_remote_routes(paired, fixture),
                            mixed_oracle, second_partial.data());
    ++dispatches;
    for (std::size_t index = 0; index < first_partial.size(); ++index) {
      const std::size_t row = index / sb2::kHiddenSize;
      const std::size_t column = index % sb2::kHiddenSize;
      require(std::abs(static_cast<double>(first_partial[index]) -
                       static_cast<double>(second_partial[index])) <=
                  2.0 * sb3::output_budget(mixed_oracle, row, column),
              "changing only local routes changed the remote partial beyond "
              "the numerical comparison budget");
    }
    std::printf(
        "Phase-3 mixed sparse-row/interleaved ownership invariance Mremote=4 "
        "PASS\n");
    session.shutdown(dispatches);
  }
  std::printf("Phase-3 process-ring identity/status/allocation accounting PASS\n");

  {
    WireProviderSession split_fault(fault_provider_path, bank_path);
    const std::vector<sb3::CanonicalCase> sweep =
        sb3::make_one_route_sweep();
    split_fault.arm_fault(2U);
    split_fault.execute_success(
        sb3::stage_remote_routes(sweep.front(), fixture),
        sb3::compute_nvfp4_oracle(sweep.front(), fixture));
    const sb3::CanonicalCase test_case = sb3::make_all_remote_case();
    split_fault.execute_failure(
        sb3::stage_remote_routes(test_case, fixture), false);
    split_fault.shutdown(2U);
  }
  std::printf("Phase-3 sequence-bound split-kernel failure/poison PASS\n");

  {
    WireProviderSession fused_fault(fault_provider_path, bank_path);
    const std::vector<sb3::CanonicalCase> sweep =
        sb3::make_one_route_sweep();
    fused_fault.arm_fault(2U);
    fused_fault.execute_success(
        sb3::stage_remote_routes(sweep.front(), fixture),
        sb3::compute_nvfp4_oracle(sweep.front(), fixture));
    const sb3::CanonicalCase& test_case = sweep.back();
    fused_fault.execute_failure(
        sb3::stage_remote_routes(test_case, fixture), false);
    fused_fault.shutdown(2U);
  }
  std::printf("Phase-3 sequence-bound fused-kernel failure/poison PASS\n");

  run_direct_shape_negatives(bank_path);
  std::printf("Phase-3 direct-provider unsupported M=0/M=129 PASS\n");
  std::printf("Phase-3 provider mathematics PASS\n");
}

std::string executable_directory() {
  std::array<char, 4096> path{};
  const ssize_t bytes = ::readlink("/proc/self/exe", path.data(), path.size());
  if (bytes <= 0 || bytes == static_cast<ssize_t>(path.size())) {
    fail("cannot resolve provider_math_test executable path");
  }
  std::string result(path.data(), static_cast<std::size_t>(bytes));
  const std::size_t slash = result.find_last_of('/');
  require(slash != std::string::npos, "test executable has no parent");
  result.resize(slash);
  return result;
}

}  // namespace

int main() {
  static_cast<void>(::signal(SIGPIPE, SIG_IGN));
  try {
    run(executable_directory());
    return 0;
  } catch (const TestFailure& error) {
    std::fprintf(stderr, "Phase-3 provider mathematics FAIL: %s\n", error.what());
    return 1;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "Phase-3 provider mathematics ERROR: %s\n", error.what());
    return 2;
  } catch (...) {
    std::fprintf(stderr, "Phase-3 provider mathematics ERROR: unknown failure\n");
    return 3;
  }
}
