// Minimal standalone probe for the Level Zero side of the memfd -> mmap ->
// B70 copy leg. See README.md. Not the qualified transport — that is
// src/phase2/memfd_transport_provider.cpp (SYCL, real B70 dispatch).
//
// FIX (confirmed against the working phase2 provider): Level Zero does NOT
// need an external-memory "import" of the shared fd at all.
// zeCommandListAppendMemoryCopy() accepts a plain mmap'd host pointer as its
// source directly — the driver walks that virtual address like any other
// host buffer via the IOMMU. zeMemAllocHost + ZE_EXTERNAL_MEMORY_TYPE_FLAG_*
// import is for a different case (re-wrapping memory another Level Zero/
// dma-buf-aware *driver* allocated), and rejects an anonymous memfd with
// ZE_RESULT_ERROR_INVALID_ARGUMENT, which is exactly what was observed.
// src/phase2/memfd_transport_provider.cpp:368-407 does precisely this: plain
// mmap(), then sycl queue.memcpy() straight off that pointer. This file is
// the same fix expressed in raw Level Zero instead of SYCL.

#include <level_zero/ze_api.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <vector>
#include <unistd.h>

namespace {

constexpr std::size_t kBytes = 4ULL * 1024 * 1024;
constexpr std::uint32_t kSeed = 0xA5A5A5A5u;

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

int recv_fd(const int socket_fd) {
  char dummy;
  iovec iov{&dummy, 1};
  char control[CMSG_SPACE(sizeof(int))] = {};
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
    std::fprintf(stderr, "no fd received\n");
    std::exit(1);
  }
  int fd = -1;
  std::memcpy(&fd, CMSG_DATA(cmsg), sizeof(int));
  return fd;
}

ze_event_pool_handle_t make_event_pool(ze_context_handle_t context,
                                       ze_device_handle_t device,
                                       uint32_t count) {
  ze_event_pool_desc_t desc{ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr,
                            ZE_EVENT_POOL_FLAG_HOST_VISIBLE, count};
  ze_event_pool_handle_t pool = nullptr;
  check_ze(zeEventPoolCreate(context, &desc, 1, &device, &pool),
           "zeEventPoolCreate");
  return pool;
}

ze_event_handle_t make_event(ze_event_pool_handle_t pool, uint32_t index) {
  ze_event_desc_t desc{ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr, index,
                       ZE_EVENT_SCOPE_FLAG_HOST, ZE_EVENT_SCOPE_FLAG_HOST};
  ze_event_handle_t event = nullptr;
  check_ze(zeEventCreate(pool, &desc, &event), "zeEventCreate");
  return event;
}

}  // namespace

int main(int argc, char** argv) {
  std::string socket_path = "/tmp/pinned-staging-probe.sock";
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--socket" && i + 1 < argc) {
      socket_path = argv[++i];
    }
  }

  check_ze(zeInit(0), "zeInit");
  uint32_t driver_count = 0;
  check_ze(zeDriverGet(&driver_count, nullptr), "zeDriverGet(count)");
  if (driver_count == 0) {
    std::fprintf(stderr, "no Level Zero drivers found\n");
    return 1;
  }
  ze_driver_handle_t driver = nullptr;
  driver_count = 1;
  check_ze(zeDriverGet(&driver_count, &driver), "zeDriverGet");

  uint32_t device_count = 0;
  check_ze(zeDeviceGet(driver, &device_count, nullptr), "zeDeviceGet(count)");
  if (device_count == 0) {
    std::fprintf(stderr, "no Level Zero devices found\n");
    return 1;
  }
  ze_device_handle_t device = nullptr;
  device_count = 1;
  check_ze(zeDeviceGet(driver, &device_count, &device), "zeDeviceGet");

  ze_device_properties_t device_props{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
  check_ze(zeDeviceGetProperties(device, &device_props),
           "zeDeviceGetProperties");
  std::printf("[provider] device=\"%s\" vendorId=0x%x\n", device_props.name,
              device_props.vendorId);

  ze_context_desc_t ctx_desc{ZE_STRUCTURE_TYPE_CONTEXT_DESC};
  ze_context_handle_t context = nullptr;
  check_ze(zeContextCreate(driver, &ctx_desc, &context), "zeContextCreate");

  // Discover a command-queue group that supports copy (compute groups
  // usually support copy too; ordinal 0 is the common case).
  uint32_t group_count = 0;
  check_ze(zeDeviceGetCommandQueueGroupProperties(device, &group_count,
                                                  nullptr),
           "zeDeviceGetCommandQueueGroupProperties(count)");
  uint32_t copy_ordinal = 0;
  {
    std::vector<ze_command_queue_group_properties_t> groups(
        group_count, {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES});
    check_ze(zeDeviceGetCommandQueueGroupProperties(device, &group_count,
                                                    groups.data()),
             "zeDeviceGetCommandQueueGroupProperties");
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
  check_ze(
      zeCommandListCreateImmediate(context, device, &cq_desc, &cmdlist),
      "zeCommandListCreateImmediate");

  ze_event_pool_handle_t event_pool = make_event_pool(context, device, 2);
  ze_event_handle_t h2d_event = make_event(event_pool, 0);
  ze_event_handle_t d2h_event = make_event(event_pool, 1);

  ze_device_mem_alloc_desc_t dev_desc{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
  void* device_buf = nullptr;
  check_ze(zeMemAllocDevice(context, &dev_desc, kBytes, 0, device,
                            &device_buf),
           "zeMemAllocDevice");

  ze_host_mem_alloc_desc_t readback_desc{
      ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
  void* readback_buf = nullptr;
  check_ze(zeMemAllocHost(context, &readback_desc, kBytes, 0, &readback_buf),
           "zeMemAllocHost(plain, no import)");

  // Socket bring-up + fd handoff.
  ::unlink(socket_path.c_str());
  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);
  const int listener = ::socket(AF_UNIX, SOCK_SEQPACKET, 0);
  ::bind(listener, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
  ::listen(listener, 1);
  std::printf("[provider] listening on %s\n", socket_path.c_str());
  const int conn = ::accept(listener, nullptr, nullptr);
  const int shared_fd = recv_fd(conn);

  // THE FIX: plain mmap of the shared fd, no Level Zero import step.
  const auto t_map = Clock::now();
  void* const mapped =
      ::mmap(nullptr, kBytes, PROT_READ | PROT_WRITE, MAP_SHARED, shared_fd, 0);
  if (mapped == MAP_FAILED) {
    std::perror("mmap");
    return 1;
  }
  std::printf("[provider] plain mmap of shared fd: %.3f ms\n", ms_since(t_map));

  // host(mmap'd) -> B70 device memory
  const auto t_h2d = Clock::now();
  check_ze(zeCommandListAppendMemoryCopy(cmdlist, device_buf, mapped, kBytes,
                                        h2d_event, 0, nullptr),
           "zeCommandListAppendMemoryCopy(H2D)");
  check_ze(zeEventHostSynchronize(h2d_event, UINT64_MAX),
           "zeEventHostSynchronize(H2D)");
  std::printf("[provider] host(mmap)->B70 copy (%zu bytes): %.3f ms (%.1f GB/s)\n",
              kBytes, ms_since(t_h2d),
              (kBytes / 1e9) / (ms_since(t_h2d) / 1e3));

  // B70 device memory -> fresh host buffer, to verify the round trip
  // independent of the original mmap region.
  const auto t_d2h = Clock::now();
  check_ze(zeCommandListAppendMemoryCopy(cmdlist, readback_buf, device_buf,
                                        kBytes, d2h_event, 0, nullptr),
           "zeCommandListAppendMemoryCopy(D2H)");
  check_ze(zeEventHostSynchronize(d2h_event, UINT64_MAX),
           "zeEventHostSynchronize(D2H)");
  std::printf("[provider] B70->host copy (%zu bytes): %.3f ms (%.1f GB/s)\n",
              kBytes, ms_since(t_d2h),
              (kBytes / 1e9) / (ms_since(t_d2h) / 1e3));

  bool ok = true;
  const auto* bytes = static_cast<const std::uint8_t*>(readback_buf);
  for (std::size_t i = 0; i < kBytes; i += 4096) {
    const std::uint8_t expected =
        static_cast<std::uint8_t>((i * 131u + kSeed * 17u + (i >> 8)) & 0xffu);
    if (bytes[i] != expected) {
      ok = false;
      std::fprintf(stderr, "mismatch at byte %zu: got 0x%02x want 0x%02x\n", i,
                   bytes[i], expected);
      break;
    }
  }
  std::printf("[provider] end-to-end pattern verify: %s\n", ok ? "PASS" : "FAIL");

  char ack = ok ? 1 : 0;
  ::send(conn, &ack, 1, 0);

  ::munmap(mapped, kBytes);
  zeMemFree(context, readback_buf);
  zeMemFree(context, device_buf);
  zeEventDestroy(h2d_event);
  zeEventDestroy(d2h_event);
  zeEventPoolDestroy(event_pool);
  zeCommandListDestroy(cmdlist);
  zeContextDestroy(context);
  ::close(shared_fd);
  ::close(conn);
  ::close(listener);
  return ok ? 0 : 1;
}
