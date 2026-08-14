#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "memfd_transport_protocol.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <iostream>
#include <linux/memfd.h>
#include <new>
#include <poll.h>
#include <signal.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <thread>
#include <utility>
#include <unistd.h>

namespace sb = shooting_brake::phase2;

namespace {

constexpr std::size_t kColdIterations = 8;
constexpr std::size_t kWarmupIterations = 16;
constexpr std::size_t kWarmIterations = 100;
constexpr int kThreadsPerBlock = 256;

struct Config {
  std::string provider_path = "src/phase2/memfd_transport_provider";
  std::string socket_path;
  int cuda_device = 0;
  int timeout_ms = 30000;
  bool accept_only = false;
};

class UniqueFd {
 public:
  UniqueFd() noexcept = default;
  explicit UniqueFd(const int fd) noexcept : fd_(fd) {}
  ~UniqueFd() {
    if (fd_ >= 0) {
      ::close(fd_);
    }
  }
  UniqueFd(const UniqueFd&) = delete;
  UniqueFd& operator=(const UniqueFd&) = delete;
  UniqueFd(UniqueFd&& other) noexcept : fd_(other.release()) {}
  int get() const noexcept { return fd_; }
  int release() noexcept {
    const int result = fd_;
    fd_ = -1;
    return result;
  }
  void reset(const int fd = -1) noexcept {
    if (fd_ >= 0) {
      ::close(fd_);
    }
    fd_ = fd;
  }

 private:
  int fd_ = -1;
};

class Mapping {
 public:
  Mapping() noexcept = default;
  ~Mapping() { reset(); }
  Mapping(const Mapping&) = delete;
  Mapping& operator=(const Mapping&) = delete;
  void reset(void* const address = MAP_FAILED,
             const std::size_t bytes = 0) noexcept {
    if (address_ != MAP_FAILED) {
      ::munmap(address_, bytes_);
    }
    address_ = address;
    bytes_ = bytes;
  }
  void* get() const noexcept { return address_; }

 private:
  void* address_ = MAP_FAILED;
  std::size_t bytes_ = 0;
};

struct Timing {
  double cuda_d2h_ms = 0.0;
  double ipc_provider_ms = 0.0;
  double cuda_h2d_ms = 0.0;
  double provider_h2d_ms = 0.0;
  double provider_d2h_ms = 0.0;
  double provider_device_total_ms = 0.0;
};

[[noreturn]] void throw_errno(const char* const operation) {
  throw std::runtime_error(std::string(operation) + ": " +
                           std::strerror(errno));
}

void check_cuda(const cudaError_t result, const char* const operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(result));
  }
}

int parse_bounded_int(const std::string_view text, const int minimum,
                      const int maximum, const char* const name) {
  int value = 0;
  const char* const begin = text.data();
  const char* const end = begin + text.size();
  const auto parsed = std::from_chars(begin, end, value);
  if (parsed.ec != std::errc{} || parsed.ptr != end || value < minimum ||
      value > maximum) {
    throw std::runtime_error(std::string("invalid ") + name);
  }
  return value;
}

Config parse_config(const int argc, char** argv) {
  Config config;
  bool provider_seen = false;
  bool socket_seen = false;
  bool cuda_seen = false;
  bool timeout_seen = false;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    auto require_value = [&](const char* const option) -> std::string_view {
      if (index + 1 >= argc) {
        throw std::runtime_error(std::string(option) + " requires a value");
      }
      return argv[++index];
    };
    if (argument == "--provider") {
      if (provider_seen) {
        throw std::runtime_error("duplicate --provider");
      }
      provider_seen = true;
      config.provider_path = require_value("--provider");
    } else if (argument == "--socket") {
      if (socket_seen) {
        throw std::runtime_error("duplicate --socket");
      }
      socket_seen = true;
      config.socket_path = require_value("--socket");
    } else if (argument == "--cuda-device") {
      if (cuda_seen) {
        throw std::runtime_error("duplicate --cuda-device");
      }
      cuda_seen = true;
      config.cuda_device = parse_bounded_int(
          require_value("--cuda-device"), 0, 1023, "CUDA device index");
    } else if (argument == "--timeout-ms") {
      if (timeout_seen) {
        throw std::runtime_error("duplicate --timeout-ms");
      }
      timeout_seen = true;
      config.timeout_ms = parse_bounded_int(
          require_value("--timeout-ms"), 1, 600000, "timeout");
    } else if (argument == "--accept-only") {
      if (config.accept_only) {
        throw std::runtime_error("duplicate --accept-only");
      }
      config.accept_only = true;
    } else if (argument == "--help") {
      std::cout
          << "usage: src/phase2/memfd_transport_host [--provider PATH] "
             "[--socket PATH] [--cuda-device N] [--timeout-ms N] "
             "[--accept-only]\n"
          << "Default behavior spawns one provider per fresh runtime. "
             "With --accept-only, run the printed provider command once; "
             "its --sessions count covers every benchmark runtime.\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + std::string(argument));
    }
  }
  if (config.accept_only && provider_seen) {
    throw std::runtime_error("--provider cannot be combined with --accept-only");
  }
  if (config.provider_path.empty()) {
    throw std::runtime_error("provider path cannot be empty");
  }
  if (config.socket_path.empty()) {
    config.socket_path = "/tmp/shooting-brake-phase2-" +
                         std::to_string(static_cast<long long>(::getpid())) +
                         ".sock";
  }
  if (config.socket_path.size() >= sizeof(sockaddr_un::sun_path)) {
    throw std::runtime_error("Unix socket path is too long");
  }
  return config;
}

void wait_readable(const int fd, const int timeout_ms,
                   const char* const description) {
  pollfd item{fd, POLLIN, 0};
  int result;
  do {
    result = ::poll(&item, 1, timeout_ms);
  } while (result < 0 && errno == EINTR);
  if (result < 0) {
    throw_errno("poll");
  }
  if (result == 0) {
    throw std::runtime_error(std::string("timed out waiting for ") +
                             description);
  }
  if ((item.revents & POLLIN) == 0) {
    throw std::runtime_error(std::string("control failure while waiting for ") +
                             description);
  }
}

void signal_eventfd(const int fd) {
  const std::uint64_t value = 1;
  ssize_t result;
  do {
    result = ::write(fd, &value, sizeof(value));
  } while (result < 0 && errno == EINTR);
  if (result != static_cast<ssize_t>(sizeof(value))) {
    if (result < 0) {
      throw_errno("write request eventfd");
    }
    throw std::runtime_error("short request eventfd write");
  }
}

void wait_eventfd(const int fd, const int timeout_ms) {
  wait_readable(fd, timeout_ms, "provider completion");
  std::uint64_t value = 0;
  ssize_t result;
  do {
    result = ::read(fd, &value, sizeof(value));
  } while (result < 0 && errno == EINTR);
  if (result != static_cast<ssize_t>(sizeof(value))) {
    if (result < 0) {
      throw_errno("read completion eventfd");
    }
    throw std::runtime_error("short completion eventfd read");
  }
  if (value != 1) {
    throw std::runtime_error("coalesced or invalid completion eventfd signal");
  }
}

__global__ void fill_pattern_kernel(std::uint8_t* const destination,
                                    const std::uint64_t bytes,
                                    const std::uint32_t seed) {
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < bytes;
       index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    destination[index] = static_cast<std::uint8_t>(
        (index * 131ULL + static_cast<std::uint64_t>(seed) * 17ULL +
         (index >> 8U)) &
        0xffULL);
  }
}

std::uint8_t expected_byte(const std::uint64_t index,
                           const std::uint32_t seed) noexcept {
  return static_cast<std::uint8_t>(
      (index * 131ULL + static_cast<std::uint64_t>(seed) * 17ULL +
       (index >> 8U)) &
      0xffULL);
}

class TransportProbe {
 public:
  explicit TransportProbe(Config config) : config_(std::move(config)) {}
  ~TransportProbe() {
    best_effort_provider_stop();
    cleanup_cuda();
    terminate_child_if_needed();
    control_.reset();
    listener_.reset();
    if (socket_bound_) {
      ::unlink(config_.socket_path.c_str());
    }
  }
  TransportProbe(const TransportProbe&) = delete;
  TransportProbe& operator=(const TransportProbe&) = delete;

  void start() {
    create_shared_region();
    create_listener();
    if (!config_.accept_only) {
      spawn_provider();
    } else {
      std::cout << "control_socket=" << config_.socket_path << std::endl;
    }
    accept_provider();
    send_bootstrap();
    receive_bootstrap_ack();
    provider_ready_ = true;
    initialize_cuda();
  }

  Timing run_iteration(const std::uint32_t bytes) {
    if (!provider_ready_ || bytes == 0 || bytes > sb::kPayloadCapacity) {
      throw std::runtime_error("invalid iteration precondition");
    }
    if (!sb::valid_layout(*header_, sb::kMappingBytes)) {
      throw std::runtime_error("host shared layout changed unexpectedly");
    }
    if (header_->request.state.load(std::memory_order_acquire) !=
            static_cast<std::uint32_t>(sb::RequestState::idle) ||
        header_->completion.state.load(std::memory_order_acquire) !=
            static_cast<std::uint32_t>(sb::CompletionState::idle)) {
      throw std::runtime_error("attempt to reuse a live publication");
    }

    const std::uint32_t seed = static_cast<std::uint32_t>(
        sequence_ ^ (static_cast<std::uint64_t>(bytes) << 7U));
    const unsigned int blocks = static_cast<unsigned int>(
        (static_cast<std::uint64_t>(bytes) + kThreadsPerBlock - 1) /
        kThreadsPerBlock);
    fill_pattern_kernel<<<blocks, kThreadsPerBlock, 0, stream_>>>(
        cuda_source_, bytes, seed);
    check_cuda(cudaGetLastError(), "launch fill_pattern_kernel");
    check_cuda(cudaStreamSynchronize(stream_), "synchronize pattern kernel");

    auto* const base = static_cast<std::uint8_t*>(mapping_.get());
    auto* const request_payload = base + header_->prefix.request_offset;
    const auto* const response_payload =
        base + header_->prefix.response_offset;

    Timing timing;
    check_cuda(cudaEventRecord(d2h_start_, stream_), "record CUDA D2H start");
    check_cuda(cudaMemcpyAsync(request_payload, cuda_source_, bytes,
                               cudaMemcpyDeviceToHost, stream_),
               "CUDA device-to-shared copy");
    check_cuda(cudaEventRecord(d2h_end_, stream_), "record CUDA D2H end");
    check_cuda(cudaEventSynchronize(d2h_end_), "synchronize CUDA D2H");
    float elapsed_ms = 0.0F;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, d2h_start_, d2h_end_),
               "measure CUDA D2H");
    timing.cuda_d2h_ms = elapsed_ms;

    const auto ipc_start = std::chrono::steady_clock::now();
    header_->request.payload_bytes = bytes;
    header_->request.pattern_seed = seed;
    header_->request.flags = 0;
    header_->request.iteration_tag = sequence_;
    header_->request.sequence.store(sequence_, std::memory_order_relaxed);
    header_->request.state.store(
        static_cast<std::uint32_t>(sb::RequestState::ready),
        std::memory_order_release);
    signal_eventfd(request_event_.get());
    wait_eventfd(completion_event_.get(), config_.timeout_ms);

    const auto completion_state = static_cast<sb::CompletionState>(
        header_->completion.state.load(std::memory_order_acquire));
    const auto ipc_end = std::chrono::steady_clock::now();
    timing.ipc_provider_ms =
        std::chrono::duration<double, std::milli>(ipc_end - ipc_start).count();

    const std::uint64_t completion_sequence =
        header_->completion.sequence.load(std::memory_order_relaxed);
    const auto completion_status = static_cast<sb::CompletionStatus>(
        header_->completion.status);
    if (completion_state != sb::CompletionState::complete ||
        completion_status != sb::CompletionStatus::ok ||
        completion_sequence != sequence_ ||
        header_->completion.payload_bytes != bytes) {
      throw std::runtime_error("provider returned invalid completion metadata");
    }
    if (header_->request.state.load(std::memory_order_acquire) !=
        static_cast<std::uint32_t>(sb::RequestState::idle)) {
      throw std::runtime_error("provider completed before releasing request");
    }
    timing.provider_h2d_ms =
        static_cast<double>(header_->completion.provider_h2d_ns) / 1.0e6;
    timing.provider_d2h_ms =
        static_cast<double>(header_->completion.provider_d2h_ns) / 1.0e6;
    timing.provider_device_total_ms =
        static_cast<double>(header_->completion.provider_total_ns) / 1.0e6;
    header_->completion.state.store(
        static_cast<std::uint32_t>(sb::CompletionState::idle),
        std::memory_order_release);

    check_cuda(cudaEventRecord(h2d_start_, stream_), "record CUDA H2D start");
    check_cuda(cudaMemcpyAsync(cuda_destination_, response_payload, bytes,
                               cudaMemcpyHostToDevice, stream_),
               "CUDA shared-to-device copy");
    check_cuda(cudaEventRecord(h2d_end_, stream_), "record CUDA H2D end");
    check_cuda(cudaEventSynchronize(h2d_end_), "synchronize CUDA H2D");
    elapsed_ms = 0.0F;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, h2d_start_, h2d_end_),
               "measure CUDA H2D");
    timing.cuda_h2d_ms = elapsed_ms;

    check_cuda(cudaMemcpyAsync(verification_buffer_, cuda_destination_, bytes,
                               cudaMemcpyDeviceToHost, stream_),
               "copy CUDA destination for verification");
    check_cuda(cudaStreamSynchronize(stream_),
               "synchronize CUDA verification copy");
    for (std::uint64_t index = 0; index < bytes; ++index) {
      if (verification_buffer_[index] != expected_byte(index, seed)) {
        throw std::runtime_error(
            "byte mismatch after CUDA-shared-B70-shared-CUDA round trip at " +
            std::to_string(index));
      }
    }

    ++sequence_;
    return timing;
  }

  void shutdown() {
    if (!provider_ready_ || shutdown_complete_) {
      return;
    }
    if (header_->request.state.load(std::memory_order_acquire) !=
            static_cast<std::uint32_t>(sb::RequestState::idle) ||
        header_->completion.state.load(std::memory_order_acquire) !=
            static_cast<std::uint32_t>(sb::CompletionState::idle)) {
      throw std::runtime_error("cannot shut down with a live publication");
    }
    header_->request.payload_bytes = 0;
    header_->request.pattern_seed = 0;
    header_->request.flags = 0;
    header_->request.iteration_tag = sequence_;
    header_->request.sequence.store(sequence_, std::memory_order_relaxed);
    header_->request.state.store(
        static_cast<std::uint32_t>(sb::RequestState::shutdown),
        std::memory_order_release);
    signal_eventfd(request_event_.get());
    wait_eventfd(completion_event_.get(), config_.timeout_ms);
    const auto state = static_cast<sb::CompletionState>(
        header_->completion.state.load(std::memory_order_acquire));
    if (state != sb::CompletionState::shutdown ||
        header_->completion.sequence.load(std::memory_order_relaxed) !=
            sequence_ ||
        static_cast<sb::CompletionStatus>(header_->completion.status) !=
            sb::CompletionStatus::ok) {
      throw std::runtime_error("provider rejected clean shutdown");
    }
    header_->completion.state.store(
        static_cast<std::uint32_t>(sb::CompletionState::idle),
        std::memory_order_release);
    ++sequence_;
    shutdown_complete_ = true;
    provider_ready_ = false;
    wait_for_child_exit();
  }

 private:
  void create_shared_region() {
    memfd_.reset(::memfd_create("shooting-brake-phase2",
                                MFD_CLOEXEC | MFD_ALLOW_SEALING));
    if (memfd_.get() < 0) {
      throw_errno("memfd_create");
    }
    if (::ftruncate(memfd_.get(), static_cast<off_t>(sb::kMappingBytes)) != 0) {
      throw_errno("ftruncate memfd");
    }
    if (::fcntl(memfd_.get(), F_ADD_SEALS,
                F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL) != 0) {
      throw_errno("seal memfd extent");
    }
    void* const mapped =
        ::mmap(nullptr, sb::kMappingBytes, PROT_READ | PROT_WRITE, MAP_SHARED,
               memfd_.get(), 0);
    if (mapped == MAP_FAILED) {
      throw_errno("mmap memfd");
    }
    mapping_.reset(mapped, sb::kMappingBytes);
    std::memset(mapped, 0, sb::kMappingBytes);
    header_ = ::new (mapped) sb::ProbeHeader{};
    header_->prefix = sb::HeaderPrefix{
        sb::kProtocolMagic,
        sb::kProtocolVersion,
        sb::kHeaderBytes,
        sb::kMappingBytes,
        sb::kRequestOffset,
        sb::kPayloadCapacity,
        sb::kResponseOffset,
        sb::kPayloadCapacity,
        static_cast<std::uint32_t>(sb::kPayloadCapacity),
        0};
    header_->request.sequence.store(0, std::memory_order_relaxed);
    header_->request.state.store(
        static_cast<std::uint32_t>(sb::RequestState::idle),
        std::memory_order_relaxed);
    header_->completion.sequence.store(0, std::memory_order_relaxed);
    header_->completion.state.store(
        static_cast<std::uint32_t>(sb::CompletionState::idle),
        std::memory_order_relaxed);
    if (!sb::valid_layout(*header_, sb::kMappingBytes)) {
      throw std::runtime_error("internal shared layout initialization failed");
    }

    request_event_.reset(::eventfd(0, EFD_CLOEXEC));
    completion_event_.reset(::eventfd(0, EFD_CLOEXEC));
    if (request_event_.get() < 0 || completion_event_.get() < 0) {
      throw_errno("eventfd");
    }
  }

  void create_listener() {
    struct stat existing {};
    if (::lstat(config_.socket_path.c_str(), &existing) == 0) {
      throw std::runtime_error("control socket path already exists");
    }
    if (errno != ENOENT) {
      throw_errno("lstat control socket");
    }
    listener_.reset(
        ::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0));
    if (listener_.get() < 0) {
      throw_errno("socket");
    }
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    std::memcpy(address.sun_path, config_.socket_path.c_str(),
                config_.socket_path.size() + 1);
    const socklen_t address_bytes = static_cast<socklen_t>(
        offsetof(sockaddr_un, sun_path) + config_.socket_path.size() + 1);
    if (::bind(listener_.get(), reinterpret_cast<sockaddr*>(&address),
               address_bytes) != 0) {
      throw_errno("bind control socket");
    }
    socket_bound_ = true;
    if (::listen(listener_.get(), 1) != 0) {
      throw_errno("listen control socket");
    }
  }

  void spawn_provider() {
    child_pid_ = ::fork();
    if (child_pid_ < 0) {
      throw_errno("fork provider");
    }
    if (child_pid_ == 0) {
      ::execl(config_.provider_path.c_str(), config_.provider_path.c_str(),
              "--socket", config_.socket_path.c_str(),
              static_cast<char*>(nullptr));
      std::fprintf(stderr, "exec %s: %s\n", config_.provider_path.c_str(),
                   std::strerror(errno));
      ::_exit(127);
    }
  }

  void accept_provider() {
    wait_readable(listener_.get(), config_.timeout_ms, "provider connection");
    int accepted;
    do {
      accepted = ::accept4(listener_.get(), nullptr, nullptr, SOCK_CLOEXEC);
    } while (accepted < 0 && errno == EINTR);
    if (accepted < 0) {
      throw_errno("accept provider");
    }
    control_.reset(accepted);
    listener_.reset();
  }

  void send_bootstrap() {
    const sb::BootstrapMessage bootstrap{sb::kBootstrapMagic,
                                         sb::kProtocolVersion,
                                         sb::kBootstrapFdCount,
                                         sb::kMappingBytes};
    iovec payload{const_cast<sb::BootstrapMessage*>(&bootstrap),
                  sizeof(bootstrap)};
    std::array<std::byte,
               CMSG_SPACE(sizeof(int) * sb::kBootstrapFdCount)>
        control{};
    msghdr message{};
    message.msg_iov = &payload;
    message.msg_iovlen = 1;
    message.msg_control = control.data();
    message.msg_controllen = control.size();
    cmsghdr* const cmsg = CMSG_FIRSTHDR(&message);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int) * sb::kBootstrapFdCount);
    const std::array<int, sb::kBootstrapFdCount> handles{
        memfd_.get(), request_event_.get(), completion_event_.get()};
    std::memcpy(CMSG_DATA(cmsg), handles.data(), sizeof(handles));

    ssize_t sent;
    do {
      sent = ::sendmsg(control_.get(), &message, MSG_NOSIGNAL);
    } while (sent < 0 && errno == EINTR);
    if (sent != static_cast<ssize_t>(sizeof(bootstrap))) {
      if (sent < 0) {
        throw_errno("sendmsg bootstrap");
      }
      throw std::runtime_error("short bootstrap packet");
    }
  }

  void receive_bootstrap_ack() {
    wait_readable(control_.get(), config_.timeout_ms,
                  "provider bootstrap acknowledgement");
    sb::BootstrapAck ack{};
    ssize_t received;
    do {
      received = ::recv(control_.get(), &ack, sizeof(ack), 0);
    } while (received < 0 && errno == EINTR);
    if (received != static_cast<ssize_t>(sizeof(ack)) ||
        ack.magic != sb::kBootstrapAckMagic || ack.status != 0 ||
        ack.reserved != 0) {
      throw std::runtime_error("provider bootstrap validation failed");
    }
  }

  void initialize_cuda() {
    check_cuda(cudaSetDevice(config_.cuda_device), "select CUDA device");
    cuda_selected_ = true;
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, config_.cuda_device),
               "query CUDA device");
    if (std::string(properties.name) != "NVIDIA GeForce RTX 5090") {
      throw std::runtime_error(
          "selected CUDA device is not the required NVIDIA GeForce RTX 5090");
    }
    check_cuda(cudaHostRegister(mapping_.get(), sb::kMappingBytes,
                                cudaHostRegisterPortable),
               "register shared memfd mapping with CUDA");
    cuda_registered_ = true;
    check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
               "create CUDA stream");
    check_cuda(cudaEventCreate(&d2h_start_), "create D2H start event");
    check_cuda(cudaEventCreate(&d2h_end_), "create D2H end event");
    check_cuda(cudaEventCreate(&h2d_start_), "create H2D start event");
    check_cuda(cudaEventCreate(&h2d_end_), "create H2D end event");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&cuda_source_),
                          sb::kPayloadCapacity),
               "allocate fixed CUDA source buffer");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&cuda_destination_),
                          sb::kPayloadCapacity),
               "allocate fixed CUDA destination buffer");
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&verification_buffer_),
                             sb::kPayloadCapacity, cudaHostAllocPortable),
               "allocate fixed verification buffer");
    std::cout << "cuda_device=\"" << properties.name
              << "\" registered_shared_bytes=" << sb::kMappingBytes << '\n';
  }

  void cleanup_cuda() noexcept {
    if (cuda_destination_ != nullptr) {
      cudaFree(cuda_destination_);
      cuda_destination_ = nullptr;
    }
    if (cuda_source_ != nullptr) {
      cudaFree(cuda_source_);
      cuda_source_ = nullptr;
    }
    if (verification_buffer_ != nullptr) {
      cudaFreeHost(verification_buffer_);
      verification_buffer_ = nullptr;
    }
    if (h2d_end_ != nullptr) {
      cudaEventDestroy(h2d_end_);
      h2d_end_ = nullptr;
    }
    if (h2d_start_ != nullptr) {
      cudaEventDestroy(h2d_start_);
      h2d_start_ = nullptr;
    }
    if (d2h_end_ != nullptr) {
      cudaEventDestroy(d2h_end_);
      d2h_end_ = nullptr;
    }
    if (d2h_start_ != nullptr) {
      cudaEventDestroy(d2h_start_);
      d2h_start_ = nullptr;
    }
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
      stream_ = nullptr;
    }
    if (cuda_registered_) {
      cudaHostUnregister(mapping_.get());
      cuda_registered_ = false;
    }
    if (cuda_selected_) {
      const cudaError_t ignored = cudaDeviceReset();
      (void)ignored;
      cuda_selected_ = false;
    }
  }

  void best_effort_provider_stop() noexcept {
    if (!provider_ready_ || shutdown_complete_ || header_ == nullptr) {
      return;
    }
    if (header_->request.state.load(std::memory_order_acquire) !=
            static_cast<std::uint32_t>(sb::RequestState::idle) ||
        header_->completion.state.load(std::memory_order_acquire) !=
            static_cast<std::uint32_t>(sb::CompletionState::idle)) {
      return;
    }
    header_->request.payload_bytes = 0;
    header_->request.pattern_seed = 0;
    header_->request.flags = 0;
    header_->request.iteration_tag = sequence_;
    header_->request.sequence.store(sequence_, std::memory_order_relaxed);
    header_->request.state.store(
        static_cast<std::uint32_t>(sb::RequestState::shutdown),
        std::memory_order_release);
    const std::uint64_t value = 1;
    const ssize_t ignored = ::write(request_event_.get(), &value, sizeof(value));
    (void)ignored;
    provider_ready_ = false;
  }

  void wait_for_child_exit() {
    if (child_pid_ <= 0) {
      return;
    }
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(config_.timeout_ms);
    int status = 0;
    while (std::chrono::steady_clock::now() < deadline) {
      const pid_t result = ::waitpid(child_pid_, &status, WNOHANG);
      if (result == child_pid_) {
        child_pid_ = -1;
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
          throw std::runtime_error("provider exited unsuccessfully");
        }
        return;
      }
      if (result < 0 && errno != EINTR) {
        throw_errno("waitpid provider");
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    throw std::runtime_error("provider did not exit after shutdown");
  }

  void terminate_child_if_needed() noexcept {
    if (child_pid_ <= 0) {
      return;
    }
    int status = 0;
    if (::waitpid(child_pid_, &status, WNOHANG) == child_pid_) {
      child_pid_ = -1;
      return;
    }
    ::kill(child_pid_, SIGTERM);
    for (int attempt = 0; attempt < 100; ++attempt) {
      if (::waitpid(child_pid_, &status, WNOHANG) == child_pid_) {
        child_pid_ = -1;
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ::kill(child_pid_, SIGKILL);
    while (::waitpid(child_pid_, &status, 0) < 0 && errno == EINTR) {
    }
    child_pid_ = -1;
  }

  Config config_;
  UniqueFd memfd_;
  UniqueFd request_event_;
  UniqueFd completion_event_;
  Mapping mapping_;
  sb::ProbeHeader* header_ = nullptr;
  UniqueFd listener_;
  UniqueFd control_;
  pid_t child_pid_ = -1;
  bool socket_bound_ = false;
  bool provider_ready_ = false;
  bool shutdown_complete_ = false;
  bool cuda_registered_ = false;
  bool cuda_selected_ = false;
  std::uint64_t sequence_ = 1;

  cudaStream_t stream_ = nullptr;
  cudaEvent_t d2h_start_ = nullptr;
  cudaEvent_t d2h_end_ = nullptr;
  cudaEvent_t h2d_start_ = nullptr;
  cudaEvent_t h2d_end_ = nullptr;
  std::uint8_t* cuda_source_ = nullptr;
  std::uint8_t* cuda_destination_ = nullptr;
  std::uint8_t* verification_buffer_ = nullptr;
};

template <std::size_t Count>
void collect_samples(TransportProbe& probe, const std::uint32_t bytes,
                     std::array<Timing, Count>& samples) {
  for (std::size_t index = 0; index < Count; ++index) {
    samples[index] = probe.run_iteration(bytes);
  }
}

template <std::size_t Count>
double percentile(const std::array<Timing, Count>& samples,
                  const double Timing::* const member, const double p) {
  std::array<double, Count> sorted{};
  for (std::size_t index = 0; index < Count; ++index) {
    sorted[index] = samples[index].*member;
  }
  std::sort(sorted.begin(), sorted.end());
  const std::size_t rank = std::max<std::size_t>(
      1, static_cast<std::size_t>(std::ceil(p * Count)));
  return sorted[std::min(rank, Count) - 1];
}

template <std::size_t Count>
void print_metric(const char* const metric,
                  const std::array<Timing, Count>& samples,
                  const double Timing::* const member) {
  std::cout << " metric=" << metric << " p50_ms="
            << percentile(samples, member, 0.50) << " p95_ms="
            << percentile(samples, member, 0.95) << " p99_ms="
            << percentile(samples, member, 0.99) << '\n';
}

template <std::size_t Count>
void print_stats(const std::uint32_t bytes, const char* const phase,
                 const std::array<Timing, Count>& samples) {
  std::cout << "size_bytes=" << bytes << " phase=" << phase
            << " samples=" << Count << '\n';
  print_metric("cuda_d2h", samples, &Timing::cuda_d2h_ms);
  print_metric("ipc_provider_round_trip", samples,
               &Timing::ipc_provider_ms);
  print_metric("cuda_h2d", samples, &Timing::cuda_h2d_ms);
  print_metric("provider_sycl_h2d", samples, &Timing::provider_h2d_ms);
  print_metric("provider_sycl_d2h", samples, &Timing::provider_d2h_ms);
  print_metric("provider_device_total", samples,
               &Timing::provider_device_total_ms);
}

int run(const Config& config) {
  constexpr std::size_t kRequiredProviderSessions =
      (sizeof(sb::kRequiredTransferSizes) /
       sizeof(sb::kRequiredTransferSizes[0])) *
      (kColdIterations + 1);
  if (config.accept_only) {
    std::cout << "external_provider_command=\"" << config.provider_path
              << " --socket " << config.socket_path << " --sessions "
              << kRequiredProviderSessions << "\"\n";
  }
  for (const std::uint32_t bytes : sb::kRequiredTransferSizes) {
    std::array<Timing, kColdIterations> cold{};
    std::array<Timing, kWarmIterations> warm{};

    for (std::size_t index = 0; index < cold.size(); ++index) {
      TransportProbe cold_probe(config);
      cold_probe.start();
      cold[index] = cold_probe.run_iteration(bytes);
      cold_probe.shutdown();
    }

    TransportProbe warm_probe(config);
    warm_probe.start();
    for (std::size_t index = 0; index < kWarmupIterations; ++index) {
      (void)warm_probe.run_iteration(bytes);
    }
    collect_samples(warm_probe, bytes, warm);
    warm_probe.shutdown();

    print_stats(bytes, "fresh_runtime_cold", cold);
    print_stats(bytes, "warm", warm);
    std::cout << "byte_correctness=PASS size_bytes=" << bytes
              << " cold_runtime_restarts=" << cold.size()
              << " warm_verified_iterations="
              << (kWarmupIterations + kWarmIterations) << '\n';
  }

  std::cout << "transport_probe=PASS\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(parse_config(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "memfd_transport_host: " << error.what() << '\n';
    return 1;
  }
}
