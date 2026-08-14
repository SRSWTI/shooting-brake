// Warm-run benchmark, Level Zero/B70 side. See README.md "Warm benchmark".
// Loop structure mirrors host_bench.cu exactly via the shared constants in
// bench_common.hpp — there is no dynamic round-count message.

#include "bench_common.hpp"

#include <level_zero/ze_api.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(const Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void check_ze(const ze_result_t rc, const char* op) {
  if (rc != ZE_RESULT_SUCCESS) {
    std::fprintf(stderr, "%s failed: 0x%x\n", op, static_cast<unsigned>(rc));
    std::exit(1);
  }
}

void print_row(const char* label, std::size_t bytes,
              const bench::Percentiles& p) {
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

  check_ze(zeInit(0), "zeInit");
  ze_driver_handle_t driver = nullptr;
  uint32_t count = 1;
  check_ze(zeDriverGet(&count, &driver), "zeDriverGet");
  ze_device_handle_t device = nullptr;
  count = 1;
  check_ze(zeDeviceGet(driver, &count, &device), "zeDeviceGet");
  ze_device_properties_t props{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
  check_ze(zeDeviceGetProperties(device, &props), "zeDeviceGetProperties");
  std::printf("[provider] device=\"%s\"\n", props.name);

  ze_context_desc_t ctx_desc{ZE_STRUCTURE_TYPE_CONTEXT_DESC};
  ze_context_handle_t context = nullptr;
  check_ze(zeContextCreate(driver, &ctx_desc, &context), "zeContextCreate");

  uint32_t group_count = 0;
  zeDeviceGetCommandQueueGroupProperties(device, &group_count, nullptr);
  uint32_t copy_ordinal = 0;
  {
    std::vector<ze_command_queue_group_properties_t> groups(
        group_count, {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES});
    zeDeviceGetCommandQueueGroupProperties(device, &group_count,
                                           groups.data());
    for (uint32_t i = 0; i < group_count; ++i) {
      if (groups[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) {
        copy_ordinal = i;
        break;
      }
    }
  }
  ze_command_queue_desc_t cq_desc{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                  nullptr, copy_ordinal, 0, 0,
                                  ZE_COMMAND_QUEUE_MODE_DEFAULT,
                                  ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  ze_command_list_handle_t cmdlist = nullptr;
  check_ze(zeCommandListCreateImmediate(context, device, &cq_desc, &cmdlist),
           "zeCommandListCreateImmediate");

  ze_event_pool_desc_t pool_desc{ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr,
                                 ZE_EVENT_POOL_FLAG_HOST_VISIBLE, 4};
  ze_event_pool_handle_t pool = nullptr;
  check_ze(zeEventPoolCreate(context, &pool_desc, 1, &device, &pool),
           "zeEventPoolCreate");
  auto make_event = [&](uint32_t idx) {
    ze_event_desc_t d{ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr, idx,
                      ZE_EVENT_SCOPE_FLAG_HOST, ZE_EVENT_SCOPE_FLAG_HOST};
    ze_event_handle_t e = nullptr;
    check_ze(zeEventCreate(pool, &d, &e), "zeEventCreate");
    return e;
  };
  ze_event_handle_t h2d_event = make_event(0);
  ze_event_handle_t d2h_event = make_event(1);
  ze_event_handle_t ref_h2d_event = make_event(2);
  ze_event_handle_t ref_d2h_event = make_event(3);

  ze_device_mem_alloc_desc_t dev_desc{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
  void* device_buf = nullptr;
  check_ze(zeMemAllocDevice(context, &dev_desc, bench::kBenchMaxBytes, 0,
                            device, &device_buf),
           "zeMemAllocDevice");
  ze_host_mem_alloc_desc_t host_desc{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
  void* readback_buf = nullptr;
  check_ze(zeMemAllocHost(context, &host_desc, bench::kBenchMaxBytes, 0,
                          &readback_buf),
           "zeMemAllocHost(readback)");
  // Private pinned buffer for the isolated B70<->host reference (never
  // shared with the CUDA process's memfd).
  void* private_host = nullptr;
  check_ze(zeMemAllocHost(context, &host_desc, bench::kBenchMaxBytes, 0,
                          &private_host),
           "zeMemAllocHost(private)");

  ::unlink(socket_path.c_str());
  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);
  const int listener = ::socket(AF_UNIX, SOCK_SEQPACKET, 0);
  ::bind(listener, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
  ::listen(listener, 1);
  std::printf("[provider] listening on %s\n", socket_path.c_str());
  const int conn = ::accept(listener, nullptr, nullptr);

  int fds[bench::kFdCount];
  bench::recv_fds(conn, fds);
  const int memfd = fds[0];
  const int go_efd = fds[1];
  const int done_efd = fds[2];
  char ack = 1;
  ::send(conn, &ack, 1, 0);

  void* const mapped = ::mmap(nullptr, bench::kBenchMaxBytes,
                              PROT_READ | PROT_WRITE, MAP_SHARED, memfd, 0);
  if (mapped == MAP_FAILED) {
    std::perror("mmap");
    return 1;
  }

  std::printf("\n== warm two-hop route: provider side (B70 copy round trip) "
              "==\n");
  for (const std::size_t bytes : bench::kBenchSizes) {
    for (int i = 0; i < bench::kBenchWarmup; ++i) {
      bench::efd_wait(go_efd);
      check_ze(zeCommandListAppendMemoryCopy(cmdlist, device_buf, mapped,
                                            bytes, h2d_event, 0, nullptr),
               "H2D warmup");
      check_ze(zeEventHostSynchronize(h2d_event, UINT64_MAX), "sync H2D");
      check_ze(zeEventHostReset(h2d_event), "reset H2D");
      check_ze(zeCommandListAppendMemoryCopy(cmdlist, readback_buf,
                                            device_buf, bytes, d2h_event, 0,
                                            nullptr),
               "D2H warmup");
      check_ze(zeEventHostSynchronize(d2h_event, UINT64_MAX), "sync D2H");
      check_ze(zeEventHostReset(d2h_event), "reset D2H");
      bench::efd_signal(done_efd);
    }
    std::vector<double> h2d_samples, d2h_samples, total_samples;
    h2d_samples.reserve(bench::kBenchMeasured);
    d2h_samples.reserve(bench::kBenchMeasured);
    total_samples.reserve(bench::kBenchMeasured);
    for (int i = 0; i < bench::kBenchMeasured; ++i) {
      bench::efd_wait(go_efd);
      const auto t0 = Clock::now();
      check_ze(zeCommandListAppendMemoryCopy(cmdlist, device_buf, mapped,
                                            bytes, h2d_event, 0, nullptr),
               "H2D measured");
      check_ze(zeEventHostSynchronize(h2d_event, UINT64_MAX), "sync H2D");
      const double t_h2d = ms_since(t0);
      check_ze(zeEventHostReset(h2d_event), "reset H2D");
      const auto t1 = Clock::now();
      check_ze(zeCommandListAppendMemoryCopy(cmdlist, readback_buf,
                                            device_buf, bytes, d2h_event, 0,
                                            nullptr),
               "D2H measured");
      check_ze(zeEventHostSynchronize(d2h_event, UINT64_MAX), "sync D2H");
      const double t_d2h = ms_since(t1);
      check_ze(zeEventHostReset(d2h_event), "reset D2H");
      bench::efd_signal(done_efd);
      h2d_samples.push_back(t_h2d);
      d2h_samples.push_back(t_d2h);
      total_samples.push_back(t_h2d + t_d2h);
    }
    print_row("  host(mmap)->B70", bytes, bench::compute_percentiles(h2d_samples));
    print_row("  B70->host(readback)", bytes, bench::compute_percentiles(d2h_samples));
    print_row("  provider-side total", bytes,
             bench::compute_percentiles(total_samples));
  }

  std::printf("\n== isolated single-hop reference: B70<->private-pinned-host "
              "(no shared memfd; proxy for one P2P hop) ==\n");
  for (const std::size_t bytes : bench::kBenchSizes) {
    for (int i = 0; i < bench::kBenchWarmup; ++i) {
      check_ze(zeCommandListAppendMemoryCopy(cmdlist, device_buf,
                                            private_host, bytes,
                                            ref_h2d_event, 0, nullptr),
               "ref warmup");
      check_ze(zeEventHostSynchronize(ref_h2d_event, UINT64_MAX), "sync ref");
      check_ze(zeEventHostReset(ref_h2d_event), "reset ref");
    }
    std::vector<double> samples;
    samples.reserve(bench::kBenchMeasured);
    for (int i = 0; i < bench::kBenchMeasured; ++i) {
      const auto t0 = Clock::now();
      check_ze(zeCommandListAppendMemoryCopy(cmdlist, device_buf,
                                            private_host, bytes,
                                            ref_d2h_event, 0, nullptr),
               "ref measured");
      check_ze(zeEventHostSynchronize(ref_d2h_event, UINT64_MAX), "sync ref");
      check_ze(zeEventHostReset(ref_d2h_event), "reset ref");
      samples.push_back(ms_since(t0));
    }
    print_row("B70<->host H2D only", bytes, bench::compute_percentiles(samples));
  }

  ::munmap(mapped, bench::kBenchMaxBytes);
  zeMemFree(context, private_host);
  zeMemFree(context, readback_buf);
  zeMemFree(context, device_buf);
  zeCommandListDestroy(cmdlist);
  zeContextDestroy(context);
  ::close(memfd);
  ::close(go_efd);
  ::close(done_efd);
  ::close(conn);
  ::close(listener);
  return 0;
}
