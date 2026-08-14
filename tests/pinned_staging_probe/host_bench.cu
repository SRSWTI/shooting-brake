// Warm-run benchmark, host/CUDA side. See README.md "Warm benchmark" section.
//
// Measures two things per size, after warmup:
//   1. Full route: GPU write -> memfd -> eventfd signal -> [provider copies
//      to B70 and back] -> eventfd ack. This is the real two-hop path.
//   2. Isolated 5090<->host reference: same GPU write, but into a private
//      pinned buffer never touched by the provider. No second GPU of either
//      vendor exists on this box, so this is the best available proxy for
//      "what one hop of true P2P would cost on this exact PCIe topology" —
//      it is NOT a measured P2P number, see README caveats.

#include "bench_common.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fcntl.h>
#include <linux/memfd.h>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <unistd.h>

namespace {

constexpr auto& kSizes = bench::kBenchSizes;
constexpr int kWarmup = bench::kBenchWarmup;
constexpr int kMeasured = bench::kBenchMeasured;
constexpr std::size_t kMaxBytes = bench::kBenchMaxBytes;

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

__global__ void fill_pattern(std::uint8_t* dst, std::size_t n, std::uint32_t seed) {
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    dst[i] = static_cast<std::uint8_t>((i * 131u + seed * 17u + (i >> 8)) & 0xffu);
  }
}

void print_row(const char* label, std::size_t bytes, const bench::Percentiles& p) {
  std::printf("%-28s %8zu B  p50=%8.3f ms (%6.1f GB/s)  p95=%8.3f ms  "
              "p99=%8.3f ms  min=%8.3f max=%8.3f\n",
              label, bytes, p.p50, bench::gbps(bytes, p.p50), p.p95, p.p99,
              p.min, p.max);
}

}  // namespace

int main(int argc, char** argv) {
  std::string socket_path = "/tmp/pinned-staging-bench.sock";
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--socket" && i + 1 < argc) {
      socket_path = argv[++i];
    }
  }

  // Shared memfd (the two-hop path's buffer).
  const int memfd = static_cast<int>(::syscall(SYS_memfd_create, "bench", 0));
  if (::ftruncate(memfd, static_cast<off_t>(kMaxBytes)) != 0) {
    std::perror("ftruncate");
    return 1;
  }
  void* const shared = ::mmap(nullptr, kMaxBytes, PROT_READ | PROT_WRITE,
                              MAP_SHARED, memfd, 0);
  check_cuda(cudaHostRegister(shared, kMaxBytes, cudaHostRegisterPortable),
             "cudaHostRegister(shared)");

  // Private pinned buffer for the isolated single-hop reference (never
  // shared with the provider process).
  void* private_pinned = nullptr;
  check_cuda(cudaMallocHost(&private_pinned, kMaxBytes), "cudaMallocHost");

  const int go_efd = ::eventfd(0, 0);
  const int done_efd = ::eventfd(0, 0);

  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);
  const int sock = ::socket(AF_UNIX, SOCK_SEQPACKET, 0);
  if (sock < 0 || ::connect(sock, reinterpret_cast<sockaddr*>(&addr),
                            sizeof(addr)) != 0) {
    std::perror("connect to provider_bench (start it first)");
    return 1;
  }
  const int fds[bench::kFdCount] = {memfd, go_efd, done_efd};
  bench::send_fds(sock, fds);
  char ack = 0;
  ::recv(sock, &ack, 1, 0);

  cudaStream_t stream;
  check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
  std::uint8_t* device_buf = nullptr;
  check_cuda(cudaMalloc(&device_buf, kMaxBytes), "cudaMalloc");

  std::printf("\n== warm two-hop route: 5090 write -> memfd -> eventfd -> "
              "[B70 copy round trip] -> eventfd ack ==\n");
  for (const std::size_t bytes : kSizes) {
    for (int i = 0; i < kWarmup; ++i) {
      fill_pattern<<<256, 256, 0, stream>>>(device_buf, bytes,
                                            0x1000u + i);
      check_cuda(cudaGetLastError(), "fill_pattern");
      check_cuda(cudaMemcpyAsync(shared, device_buf, bytes,
                                cudaMemcpyDeviceToHost, stream),
                "memcpy warmup");
      check_cuda(cudaStreamSynchronize(stream), "sync warmup");
      bench::efd_signal(go_efd);
      bench::efd_wait(done_efd);
    }
    std::vector<double> samples;
    samples.reserve(kMeasured);
    for (int i = 0; i < kMeasured; ++i) {
      const auto t0 = Clock::now();
      fill_pattern<<<256, 256, 0, stream>>>(device_buf, bytes,
                                            0x2000u + i);
      check_cuda(cudaGetLastError(), "fill_pattern");
      check_cuda(cudaMemcpyAsync(shared, device_buf, bytes,
                                cudaMemcpyDeviceToHost, stream),
                "memcpy measured");
      check_cuda(cudaStreamSynchronize(stream), "sync measured");
      bench::efd_signal(go_efd);
      bench::efd_wait(done_efd);
      samples.push_back(ms_since(t0));
    }
    print_row("two-hop total", bytes, bench::compute_percentiles(samples));
  }

  std::printf("\n== isolated single-hop reference: 5090<->private-pinned-host "
              "(no provider involved; proxy for one P2P hop on this exact "
              "PCIe topology) ==\n");
  for (const std::size_t bytes : kSizes) {
    for (int i = 0; i < kWarmup; ++i) {
      check_cuda(cudaMemcpyAsync(private_pinned, device_buf, bytes,
                                cudaMemcpyDeviceToHost, stream),
                "memcpy warmup ref");
      check_cuda(cudaStreamSynchronize(stream), "sync warmup ref");
    }
    std::vector<double> samples;
    samples.reserve(kMeasured);
    for (int i = 0; i < kMeasured; ++i) {
      const auto t0 = Clock::now();
      check_cuda(cudaMemcpyAsync(private_pinned, device_buf, bytes,
                                cudaMemcpyDeviceToHost, stream),
                "memcpy measured ref");
      check_cuda(cudaStreamSynchronize(stream), "sync measured ref");
      samples.push_back(ms_since(t0));
    }
    print_row("5090<->host D2H only", bytes, bench::compute_percentiles(samples));
  }

  // Sentinel: signal a zero-byte round with go_efd so the provider's fixed
  // loop count naturally ends (provider runs the identical kSizes/kWarmup/
  // kMeasured loop compiled from the same bench_common.hpp constants).
  ::close(sock);
  check_cuda(cudaHostUnregister(shared), "cudaHostUnregister");
  ::munmap(shared, kMaxBytes);
  ::close(memfd);
  ::close(go_efd);
  ::close(done_efd);
  cudaFreeHost(private_pinned);
  cudaFree(device_buf);
  cudaStreamDestroy(stream);
  return 0;
}
