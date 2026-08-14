// Minimal standalone probe for the memfd -> mmap -> cudaHostRegister leg of
// the cross-vendor host-staged transport chain. See README.md. Not the
// qualified transport — that is src/phase2/memfd_transport_host.cu.

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <linux/memfd.h>
#include <string>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {

constexpr std::size_t kBytes = 4ULL * 1024 * 1024;  // 4 MiB probe payload
constexpr std::uint32_t kSeed = 0xA5A5A5A5u;

using Clock = std::chrono::steady_clock;
double ms_since(const Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void check_cuda(const cudaError_t rc, const char* op) {
  if (rc != cudaSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", op, cudaGetErrorString(rc));
    std::exit(1);
  }
}

int memfd_create_checked(const char* name) {
  const int fd = static_cast<int>(::syscall(SYS_memfd_create, name, 0u));
  if (fd < 0) {
    std::perror("memfd_create");
    std::exit(1);
  }
  return fd;
}

__global__ void fill_pattern(std::uint8_t* dst, std::size_t n, std::uint32_t seed) {
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    dst[i] = static_cast<std::uint8_t>((i * 131u + seed * 17u + (i >> 8)) & 0xffu);
  }
}

void send_fd(const int socket_fd, const int fd_to_send) {
  char dummy = 'x';
  iovec iov{&dummy, 1};
  char control[CMSG_SPACE(sizeof(int))] = {};
  msghdr msg{};
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_control = control;
  msg.msg_controllen = sizeof(control);
  cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  cmsg->cmsg_len = CMSG_LEN(sizeof(int));
  std::memcpy(CMSG_DATA(cmsg), &fd_to_send, sizeof(int));
  if (::sendmsg(socket_fd, &msg, 0) < 0) {
    std::perror("sendmsg");
    std::exit(1);
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string socket_path = "/tmp/pinned-staging-probe.sock";
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--socket" && i + 1 < argc) {
      socket_path = argv[++i];
    }
  }

  // 1. memfd_create + ftruncate + mmap(MAP_SHARED)
  const auto t_setup = Clock::now();
  const int fd = memfd_create_checked("pinned-staging-probe");
  if (::ftruncate(fd, static_cast<off_t>(kBytes)) != 0) {
    std::perror("ftruncate");
    return 1;
  }
  void* const shared = ::mmap(nullptr, kBytes, PROT_READ | PROT_WRITE,
                               MAP_SHARED, fd, 0);
  if (shared == MAP_FAILED) {
    std::perror("mmap");
    return 1;
  }
  std::printf("[host] memfd+mmap setup: %.3f ms\n", ms_since(t_setup));

  // 2. cudaHostRegister on the mmap'd region. Match the proven pattern in
  //    src/phase2/memfd_transport_host.cu:640-641 exactly: cudaHostRegisterPortable,
  //    NOT cudaHostRegisterMapped+cudaHostGetDevicePointer. The Mapped variant
  //    creates a separate device-side BAR-style aliasing mapping that does not
  //    reliably alias the memfd's page-cache pages for a second process that
  //    mmaps the same fd fresh — that mismatch is what produced all-zero reads
  //    on the provider side in a real run of this probe. With UVA (default on
  //    Linux/single-GPU), the plain host pointer is already valid directly in
  //    cudaMemcpy calls once pinned; no separate device pointer is needed.
  const auto t_register = Clock::now();
  check_cuda(cudaHostRegister(shared, kBytes, cudaHostRegisterPortable),
             "cudaHostRegister");
  std::printf("[host] cudaHostRegister: %.3f ms\n", ms_since(t_register));

  // 3. GPU write into the pinned region — this is the "5090 -> host DRAM" hop.
  cudaStream_t stream;
  check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
  std::uint8_t* device_buf = nullptr;
  check_cuda(cudaMalloc(&device_buf, kBytes), "cudaMalloc");

  const auto t_write = Clock::now();
  fill_pattern<<<256, 256, 0, stream>>>(device_buf, kBytes, kSeed);
  // A silent launch failure (e.g. PTX/arch JIT mismatch) leaves device_buf
  // untouched; the D2H copy below would then "succeed" while faithfully
  // copying garbage/zeros. Catch that here instead of debugging it via a
  // data mismatch three hops downstream.
  check_cuda(cudaGetLastError(), "fill_pattern launch");
  check_cuda(cudaMemcpyAsync(shared, device_buf, kBytes,
                             cudaMemcpyDeviceToHost, stream),
             "cudaMemcpyAsync D2H(pinned)");
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  std::printf("[host] GPU->pinned-host write (%zu bytes): %.3f ms (%.1f GB/s)\n",
              kBytes, ms_since(t_write),
              (kBytes / 1e9) / (ms_since(t_write) / 1e3));

  // 4. Hand the fd to the provider process over SCM_RIGHTS, same mechanism
  //    src/phase2/memfd_transport_host.cu uses for the qualified transport.
  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);
  const int sock = ::socket(AF_UNIX, SOCK_SEQPACKET, 0);
  if (sock < 0 || ::connect(sock, reinterpret_cast<sockaddr*>(&addr),
                            sizeof(addr)) != 0) {
    std::perror("connect to provider_side (start it first)");
    return 1;
  }

  const auto t_handoff = Clock::now();
  send_fd(sock, fd);
  char ack = 0;
  ::recv(sock, &ack, 1, 0);
  std::printf("[host] fd handoff + provider ack: %.3f ms\n", ms_since(t_handoff));

  std::printf("[host] total (setup..ack): %.3f ms\n", ms_since(t_setup));

  ::close(sock);
  check_cuda(cudaHostUnregister(shared), "cudaHostUnregister");
  ::munmap(shared, kBytes);
  ::close(fd);
  cudaFree(device_buf);
  cudaStreamDestroy(stream);
  return 0;
}
