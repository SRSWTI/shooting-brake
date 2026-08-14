#pragma once
// Shared protocol + percentile helpers for the warm-run benchmark.
// Handshake: host creates a memfd (payload) + 2 eventfds (go/done), sends
// all three over SCM_RIGHTS once. Then N iterations reuse the same buffer —
// this matches how the real phase2 ring is warmed (handoff once, dispatch
// repeatedly), not a fresh mmap/fd per request.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <vector>

namespace bench {

constexpr int kFdCount = 3;  // memfd, go_eventfd, done_eventfd

// Shared sweep parameters — both host_bench and provider_bench must loop
// identically since there is no dynamic "how many more rounds" message.
inline constexpr std::size_t kBenchSizes[] = {4096, 65536, 1048576, 4194304};
inline constexpr int kBenchWarmup = 20;
inline constexpr int kBenchMeasured = 200;
inline constexpr std::size_t kBenchMaxBytes = 4194304;

inline void send_fds(const int socket_fd, const int fds[kFdCount]) {
  char dummy = 'x';
  iovec iov{&dummy, 1};
  char control[CMSG_SPACE(sizeof(int) * kFdCount)] = {};
  msghdr msg{};
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_control = control;
  msg.msg_controllen = sizeof(control);
  cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  cmsg->cmsg_len = CMSG_LEN(sizeof(int) * kFdCount);
  std::memcpy(CMSG_DATA(cmsg), fds, sizeof(int) * kFdCount);
  if (::sendmsg(socket_fd, &msg, 0) < 0) {
    std::perror("sendmsg");
    std::exit(1);
  }
}

inline void recv_fds(const int socket_fd, int fds[kFdCount]) {
  char dummy;
  iovec iov{&dummy, 1};
  char control[CMSG_SPACE(sizeof(int) * kFdCount)] = {};
  msghdr msg{};
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_control = control;
  msg.msg_controllen = sizeof(control);
  if (::recvmsg(socket_fd, &msg, 0) <= 0) {
    std::perror("recvmsg");
    std::exit(1);
  }
  cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
  if (!cmsg || cmsg->cmsg_type != SCM_RIGHTS) {
    std::fprintf(stderr, "no fds received\n");
    std::exit(1);
  }
  std::memcpy(fds, CMSG_DATA(cmsg), sizeof(int) * kFdCount);
}

inline void efd_signal(const int fd) {
  const std::uint64_t one = 1;
  if (::write(fd, &one, sizeof(one)) != sizeof(one)) {
    std::perror("eventfd write");
    std::exit(1);
  }
}

inline void efd_wait(const int fd) {
  std::uint64_t value = 0;
  if (::read(fd, &value, sizeof(value)) != sizeof(value)) {
    std::perror("eventfd read");
    std::exit(1);
  }
}

struct Percentiles {
  double p50, p95, p99, mean, min, max;
};

inline Percentiles compute_percentiles(std::vector<double> samples_ms) {
  std::sort(samples_ms.begin(), samples_ms.end());
  const std::size_t n = samples_ms.size();
  auto at = [&](double frac) {
    std::size_t idx = static_cast<std::size_t>(frac * (n - 1));
    return samples_ms[idx];
  };
  double sum = 0.0;
  for (double v : samples_ms) sum += v;
  return Percentiles{at(0.50), at(0.95), at(0.99), sum / n, samples_ms.front(),
                     samples_ms.back()};
}

inline double gbps(std::size_t bytes, double ms) {
  return (bytes / 1e9) / (ms / 1e3);
}

}  // namespace bench
