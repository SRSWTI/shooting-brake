#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "memfd_transport_protocol.hpp"

#include <sycl/sycl.hpp>

#include <array>
#include <cerrno>
#include <chrono>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <iostream>
#include <mutex>
#include <optional>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>

namespace sb = shooting_brake::phase2;

namespace {

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
  UniqueFd& operator=(UniqueFd&& other) noexcept {
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
  Mapping(void* const address, const std::size_t bytes) noexcept
      : address_(address), bytes_(bytes) {}
  ~Mapping() {
    if (address_ != MAP_FAILED) {
      ::munmap(address_, bytes_);
    }
  }
  Mapping(const Mapping&) = delete;
  Mapping& operator=(const Mapping&) = delete;
  void* get() const noexcept { return address_; }

 private:
  void* address_ = MAP_FAILED;
  std::size_t bytes_ = 0;
};

struct BootstrapHandles {
  sb::BootstrapMessage message{};
  std::array<UniqueFd, sb::kBootstrapFdCount> fds{};
};

[[noreturn]] void throw_errno(const char* const operation) {
  throw std::runtime_error(std::string(operation) + ": " +
                           std::strerror(errno));
}

UniqueFd connect_seqpacket(const std::string& path) {
  if (path.empty() || path.size() >= sizeof(sockaddr_un::sun_path)) {
    throw std::runtime_error("Unix socket path is empty or too long");
  }

  UniqueFd socket_fd(
      ::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0));
  if (socket_fd.get() < 0) {
    throw_errno("socket");
  }

  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
  constexpr int kAttempts = 200;
  for (int attempt = 0; attempt < kAttempts; ++attempt) {
    if (::connect(socket_fd.get(), reinterpret_cast<sockaddr*>(&address),
                  sizeof(address)) == 0) {
      return socket_fd;
    }
    if (errno != ENOENT && errno != ECONNREFUSED && errno != EINTR) {
      throw_errno("connect");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
  }
  throw std::runtime_error("timed out connecting to host control socket");
}

BootstrapHandles receive_bootstrap(const int socket_fd) {
  BootstrapHandles result;
  std::array<std::byte, CMSG_SPACE(sizeof(int) * sb::kBootstrapFdCount)>
      control{};
  iovec payload{&result.message, sizeof(result.message)};
  msghdr message{};
  message.msg_iov = &payload;
  message.msg_iovlen = 1;
  message.msg_control = control.data();
  message.msg_controllen = control.size();

  ssize_t received;
  do {
    received = ::recvmsg(socket_fd, &message, MSG_CMSG_CLOEXEC);
  } while (received < 0 && errno == EINTR);
  if (received < 0) {
    throw_errno("recvmsg");
  }
  if (received != static_cast<ssize_t>(sizeof(result.message)) ||
      (message.msg_flags & (MSG_TRUNC | MSG_CTRUNC)) != 0) {
    throw std::runtime_error("invalid bootstrap packet size or truncation");
  }
  if (result.message.magic != sb::kBootstrapMagic ||
      result.message.version != sb::kProtocolVersion ||
      result.message.fd_count != sb::kBootstrapFdCount ||
      result.message.mapping_bytes != sb::kMappingBytes) {
    throw std::runtime_error("invalid bootstrap metadata");
  }

  bool found_rights = false;
  for (cmsghdr* cmsg = CMSG_FIRSTHDR(&message); cmsg != nullptr;
       cmsg = CMSG_NXTHDR(&message, cmsg)) {
    if (found_rights || cmsg->cmsg_level != SOL_SOCKET ||
        cmsg->cmsg_type != SCM_RIGHTS ||
        cmsg->cmsg_len != CMSG_LEN(sizeof(int) * sb::kBootstrapFdCount)) {
      throw std::runtime_error("invalid bootstrap ancillary data");
    }
    std::array<int, sb::kBootstrapFdCount> raw_fds{};
    std::memcpy(raw_fds.data(), CMSG_DATA(cmsg), sizeof(raw_fds));
    for (std::size_t index = 0; index < raw_fds.size(); ++index) {
      result.fds[index].reset(raw_fds[index]);
    }
    found_rights = true;
  }
  if (!found_rights) {
    throw std::runtime_error("bootstrap did not contain SCM_RIGHTS handles");
  }
  return result;
}

void send_ack(const int socket_fd, const std::uint32_t status) {
  const sb::BootstrapAck ack{sb::kBootstrapAckMagic, status, 0};
  ssize_t sent;
  do {
    sent = ::send(socket_fd, &ack, sizeof(ack), MSG_NOSIGNAL);
  } while (sent < 0 && errno == EINTR);
  if (sent != static_cast<ssize_t>(sizeof(ack))) {
    if (sent < 0) {
      throw_errno("send bootstrap acknowledgement");
    }
    throw std::runtime_error("short bootstrap acknowledgement");
  }
}

void validate_memfd(const int fd) {
  struct stat status {};
  if (::fstat(fd, &status) != 0) {
    throw_errno("fstat memfd");
  }
  if (!S_ISREG(status.st_mode) || status.st_size < 0 ||
      static_cast<std::uint64_t>(status.st_size) != sb::kMappingBytes) {
    throw std::runtime_error("memfd has wrong type or exact extent");
  }
  const int open_flags = ::fcntl(fd, F_GETFL);
  if (open_flags < 0) {
    throw_errno("fcntl F_GETFL");
  }
  if ((open_flags & O_ACCMODE) != O_RDWR) {
    throw std::runtime_error("memfd is not read/write");
  }
  const int seals = ::fcntl(fd, F_GET_SEALS);
  if (seals < 0) {
    throw_errno("fcntl F_GET_SEALS");
  }
  if ((seals & (F_SEAL_GROW | F_SEAL_SHRINK)) !=
      (F_SEAL_GROW | F_SEAL_SHRINK)) {
    throw std::runtime_error("memfd extent is not sealed");
  }
}


sycl::device select_b70() {
  std::optional<sycl::device> selected;
  for (const sycl::platform& platform : sycl::platform::get_platforms()) {
    if (platform.get_backend() != sycl::backend::ext_oneapi_level_zero) {
      continue;
    }
    for (const sycl::device& device : platform.get_devices()) {
      if (!device.is_gpu() ||
          device.get_info<sycl::info::device::vendor_id>() != 0x8086U) {
        continue;
      }
      const std::string name = device.get_info<sycl::info::device::name>();
      if (name != "Intel(R) Arc(TM) Pro B70 Graphics") {
        continue;
      }
      if (selected) {
        throw std::runtime_error(
            "multiple Intel B70 GPUs were exposed through Level Zero");
      }
      selected.emplace(device);
    }
  }
  if (!selected) {
    throw std::runtime_error(
        "no Intel vendor 0x8086 B70 GPU was exposed through Level Zero");
  }
  return std::move(*selected);
}

std::uint64_t read_eventfd(const int fd) {
  std::uint64_t value = 0;
  ssize_t result;
  do {
    result = ::read(fd, &value, sizeof(value));
  } while (result < 0 && errno == EINTR);
  if (result != static_cast<ssize_t>(sizeof(value))) {
    if (result < 0) {
      throw_errno("read request eventfd");
    }
    throw std::runtime_error("short request eventfd read");
  }
  if (value != 1) {
    throw std::runtime_error("coalesced or invalid request eventfd signal");
  }
  return value;
}

bool wait_for_request_or_host_close(const int request_eventfd,
                                    const int control_socket) {
  std::array<pollfd, 2> items{{
      {request_eventfd, POLLIN, 0},
      {control_socket, POLLIN, 0},
  }};
  for (;;) {
    int result;
    do {
      result = ::poll(items.data(), items.size(), -1);
    } while (result < 0 && errno == EINTR);
    if (result < 0) {
      throw_errno("poll request/control handles");
    }
    if ((items[0].revents & POLLIN) != 0) {
      read_eventfd(request_eventfd);
      return true;
    }
    if ((items[0].revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
      throw std::runtime_error("request eventfd became invalid");
    }
    if ((items[1].revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
      return false;
    }
    if ((items[1].revents & POLLIN) != 0) {
      std::byte unexpected{};
      const ssize_t received =
          ::recv(control_socket, &unexpected, sizeof(unexpected),
                 MSG_PEEK | MSG_DONTWAIT);
      if (received == 0) {
        return false;
      }
      if (received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        continue;
      }
      if (received < 0) {
        throw_errno("peek control socket");
      }
      throw std::runtime_error(
          "unexpected payload on control-only Unix socket");
    }
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
      throw_errno("write completion eventfd");
    }
    throw std::runtime_error("short completion eventfd write");
  }
}

void publish_completion(sb::ProbeHeader& header, const int completion_eventfd,
                        const std::uint64_t sequence,
                        const sb::CompletionState state,
                        const sb::CompletionStatus status,
                        const std::uint32_t payload_bytes,
                        const std::uint64_t h2d_ns,
                        const std::uint64_t d2h_ns,
                        const std::uint64_t total_ns) {
  header.completion.status = static_cast<std::uint32_t>(status);
  header.completion.payload_bytes = payload_bytes;
  header.completion.reserved0 = 0;
  header.completion.provider_h2d_ns = h2d_ns;
  header.completion.provider_d2h_ns = d2h_ns;
  header.completion.provider_total_ns = total_ns;
  std::memset(header.completion.reserved, 0,
              sizeof(header.completion.reserved));
  header.completion.sequence.store(sequence, std::memory_order_relaxed);
  header.completion.state.store(static_cast<std::uint32_t>(state),
                                std::memory_order_release);
  signal_eventfd(completion_eventfd);
}

std::uint64_t profiling_duration_ns(const sycl::event& event) {
  const std::uint64_t start =
      event.get_profiling_info<sycl::info::event_profiling::command_start>();
  const std::uint64_t end =
      event.get_profiling_info<sycl::info::event_profiling::command_end>();
  if (end < start) {
    throw std::runtime_error("SYCL profiling timestamps regressed");
  }
  return end - start;
}

int run_provider(const std::string& socket_path) {
  UniqueFd control = connect_seqpacket(socket_path);
  BootstrapHandles bootstrap = receive_bootstrap(control.get());
  const int memfd = bootstrap.fds[0].get();
  const int request_eventfd = bootstrap.fds[1].get();
  const int completion_eventfd = bootstrap.fds[2].get();
  bool startup_acknowledged = false;

  try {
    validate_memfd(memfd);
    void* const mapped =
        ::mmap(nullptr, sb::kMappingBytes, PROT_READ | PROT_WRITE, MAP_SHARED,
               memfd, 0);
    if (mapped == MAP_FAILED) {
      throw_errno("mmap memfd");
    }
    Mapping mapping(mapped, sb::kMappingBytes);
    auto* const header = static_cast<sb::ProbeHeader*>(mapping.get());
    if (!sb::valid_layout(*header, sb::kMappingBytes)) {
      throw std::runtime_error("shared probe header failed exact validation");
    }
    if (header->request.state.load(std::memory_order_acquire) !=
            static_cast<std::uint32_t>(sb::RequestState::idle) ||
        header->completion.state.load(std::memory_order_acquire) !=
            static_cast<std::uint32_t>(sb::CompletionState::idle) ||
        header->request.sequence.load(std::memory_order_relaxed) != 0 ||
        header->completion.sequence.load(std::memory_order_relaxed) != 0) {
      throw std::runtime_error("shared publications are not initially idle");
    }

    const sycl::device device = select_b70();
    const std::string device_name =
        device.get_info<sycl::info::device::name>();
    const sycl::context context(device);
    std::mutex async_mutex;
    std::exception_ptr async_error;
    auto async_handler = [&](sycl::exception_list errors) {
      std::lock_guard<std::mutex> lock(async_mutex);
      if (!async_error) {
        for (const std::exception_ptr& error : errors) {
          async_error = error;
          break;
        }
      }
    };
    sycl::queue queue(
        context, device, async_handler,
        sycl::property_list{sycl::property::queue::enable_profiling{}});
    auto* const device_buffer =
        sycl::malloc_device<std::uint8_t>(sb::kPayloadCapacity, queue);
    if (device_buffer == nullptr) {
      throw std::runtime_error("failed to allocate fixed B70 device buffer");
    }

    struct DeviceBufferGuard {
      std::uint8_t* pointer;
      sycl::context context;
      sycl::queue* queue;
      ~DeviceBufferGuard() noexcept {
        if (pointer == nullptr) {
          return;
        }
        try {
          queue->wait();
          sycl::free(pointer, context);
        } catch (...) {
        }
      }
    } device_guard{device_buffer, context, &queue};

    send_ack(control.get(), 0);
    startup_acknowledged = true;
    std::cout << "provider_device=\"" << device_name
              << "\" vendor=0x8086 fixed_device_bytes="
              << sb::kPayloadCapacity << '\n';

    std::uint64_t expected_sequence = 1;
    bool running = true;
    bool clean_shutdown = false;
    while (running) {
      if (!wait_for_request_or_host_close(request_eventfd, control.get())) {
        break;
      }

      if (!sb::valid_layout(*header, sb::kMappingBytes)) {
        const std::uint64_t sequence =
            header->request.sequence.load(std::memory_order_relaxed);
        header->request.state.store(
            static_cast<std::uint32_t>(sb::RequestState::idle),
            std::memory_order_release);
        publish_completion(*header, completion_eventfd, sequence,
                           sb::CompletionState::failed,
                           sb::CompletionStatus::bad_layout, 0, 0, 0, 0);
        break;
      }

      if (header->completion.state.load(std::memory_order_acquire) !=
          static_cast<std::uint32_t>(sb::CompletionState::idle)) {
        throw std::runtime_error("host reused a live completion publication");
      }

      const auto request_state = static_cast<sb::RequestState>(
          header->request.state.load(std::memory_order_acquire));
      const std::uint64_t sequence =
          header->request.sequence.load(std::memory_order_relaxed);
      const std::uint32_t payload_bytes = header->request.payload_bytes;
      const std::uint32_t request_flags = header->request.flags;

      if (request_state != sb::RequestState::ready &&
          request_state != sb::RequestState::shutdown) {
        publish_completion(*header, completion_eventfd, sequence,
                           sb::CompletionState::failed,
                           sb::CompletionStatus::bad_state, 0, 0, 0, 0);
        break;
      }

      header->request.state.store(
          static_cast<std::uint32_t>(sb::RequestState::processing),
          std::memory_order_release);

      if (sequence != expected_sequence) {
        header->request.state.store(
            static_cast<std::uint32_t>(sb::RequestState::idle),
            std::memory_order_release);
        publish_completion(*header, completion_eventfd, sequence,
                           sb::CompletionState::failed,
                           sb::CompletionStatus::bad_sequence, 0, 0, 0, 0);
        break;
      }

      if (request_state == sb::RequestState::shutdown) {
        if (payload_bytes != 0 || request_flags != 0) {
          header->request.state.store(
              static_cast<std::uint32_t>(sb::RequestState::idle),
              std::memory_order_release);
          publish_completion(*header, completion_eventfd, sequence,
                             sb::CompletionState::failed,
                             sb::CompletionStatus::bad_extent, 0, 0, 0, 0);
          break;
        }
        header->request.state.store(
            static_cast<std::uint32_t>(sb::RequestState::idle),
            std::memory_order_release);
        publish_completion(*header, completion_eventfd, sequence,
                           sb::CompletionState::shutdown,
                           sb::CompletionStatus::ok, 0, 0, 0, 0);
        clean_shutdown = true;
        running = false;
        continue;
      }

      const sb::HeaderPrefix& layout = header->prefix;
      const bool valid_extent =
          request_flags == 0 && payload_bytes != 0 &&
          payload_bytes <= layout.max_transfer_bytes &&
          payload_bytes <= layout.request_capacity &&
          payload_bytes <= layout.response_capacity &&
          sb::range_within(layout.request_offset, payload_bytes,
                           layout.mapping_bytes) &&
          sb::range_within(layout.response_offset, payload_bytes,
                           layout.mapping_bytes) &&
          sb::ranges_disjoint(layout.request_offset, payload_bytes,
                              layout.response_offset, payload_bytes);
      if (!valid_extent) {
        header->request.state.store(
            static_cast<std::uint32_t>(sb::RequestState::idle),
            std::memory_order_release);
        publish_completion(*header, completion_eventfd, sequence,
                           sb::CompletionState::failed,
                           sb::CompletionStatus::bad_extent, 0, 0, 0, 0);
        break;
      }

      auto* const base = static_cast<std::uint8_t*>(mapping.get());
      const auto* const request_payload = base + layout.request_offset;
      auto* const response_payload = base + layout.response_offset;

      try {
        {
          std::lock_guard<std::mutex> lock(async_mutex);
          async_error = nullptr;
        }
        const sycl::event to_device =
            queue.memcpy(device_buffer, request_payload, payload_bytes);
        sycl::event from_device = queue.submit([&](sycl::handler& cgh) {
          cgh.depends_on(to_device);
          cgh.memcpy(response_payload, device_buffer, payload_bytes);
        });
        from_device.wait_and_throw();
        {
          std::lock_guard<std::mutex> lock(async_mutex);
          if (async_error) {
            std::rethrow_exception(async_error);
          }
        }

        const std::uint64_t h2d_ns = profiling_duration_ns(to_device);
        const std::uint64_t d2h_ns = profiling_duration_ns(from_device);
        const std::uint64_t first_start =
            to_device.get_profiling_info<
                sycl::info::event_profiling::command_start>();
        const std::uint64_t last_end =
            from_device.get_profiling_info<
                sycl::info::event_profiling::command_end>();
        if (last_end < first_start) {
          throw std::runtime_error("SYCL total profiling timestamps regressed");
        }

        header->request.state.store(
            static_cast<std::uint32_t>(sb::RequestState::idle),
            std::memory_order_release);
        publish_completion(*header, completion_eventfd, sequence,
                           sb::CompletionState::complete,
                           sb::CompletionStatus::ok, payload_bytes, h2d_ns,
                           d2h_ns, last_end - first_start);
        ++expected_sequence;
      } catch (const std::exception& error) {
        std::cerr << "provider copy failure: " << error.what() << '\n';
        try {
          queue.wait_and_throw();
        } catch (const std::exception& drain_error) {
          std::cerr << "provider queue drain failure: " << drain_error.what()
                    << '\n';
        } catch (...) {
          std::cerr << "provider queue drain failed with a non-standard error\n";
        }
        header->request.state.store(
            static_cast<std::uint32_t>(sb::RequestState::idle),
            std::memory_order_release);
        publish_completion(*header, completion_eventfd, sequence,
                           sb::CompletionState::failed,
                           sb::CompletionStatus::device_failure, 0, 0, 0, 0);
        break;
      }
    }
    return clean_shutdown ? 0 : 1;
  } catch (...) {
    if (!startup_acknowledged) {
      try {
        send_ack(control.get(), 1);
      } catch (...) {
      }
    }
    throw;
  }
}

struct Options {
  std::string socket_path;
  std::uint32_t sessions = 1;
};

Options parse_options(const int argc, char** argv) {
  Options options;
  bool sessions_seen = false;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--socket") {
      if (!options.socket_path.empty() || index + 1 >= argc) {
        throw std::runtime_error("--socket requires exactly one path");
      }
      options.socket_path = argv[++index];
    } else if (argument == "--sessions") {
      if (sessions_seen || index + 1 >= argc) {
        throw std::runtime_error("--sessions requires exactly one count");
      }
      sessions_seen = true;
      const std::string_view text(argv[++index]);
      const char* const begin = text.data();
      const char* const end = begin + text.size();
      const auto parsed =
          std::from_chars(begin, end, options.sessions);
      if (parsed.ec != std::errc{} || parsed.ptr != end ||
          options.sessions == 0 || options.sessions > 1024) {
        throw std::runtime_error(
            "--sessions must be an integer in the range 1..1024");
      }
    } else if (argument == "--help") {
      std::cout
          << "usage: src/phase2/memfd_transport_provider --socket PATH "
             "[--sessions N]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + std::string(argument));
    }
  }
  if (options.socket_path.empty()) {
    throw std::runtime_error("--socket PATH is required");
  }
  return options;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    for (std::uint32_t session = 0; session < options.sessions; ++session) {
      if (run_provider(options.socket_path) != 0) {
        return 1;
      }
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "memfd_transport_provider: " << error.what() << '\n';
    return 1;
  }
}
