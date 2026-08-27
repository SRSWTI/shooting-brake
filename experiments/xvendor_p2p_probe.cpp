// Kill-bench #8, arm 8g: cross-vendor P2P via the dma-buf handshake, running
// the OPPOSITE direction from the killed 8b/8c arms.
//
//   8b/8c (KILLED): NVIDIA exports, Intel imports.
//       cuMemGetHandleForAddressRange(DMA_BUF_FD) -> CUDA_ERROR_NOT_SUPPORTED.
//       Gate is byte-identical at 595.71.05 and 610.57.04, so no driver
//       version or community fork opens it.
//
//   8g (this probe): Intel exports, NVIDIA imports.
//       Level Zero allocates B70 device memory with a DMA_BUF export flag and
//       hands out an fd (zeMemGetAllocProperties + ze_external_memory_export_fd_t).
//       CUDA imports it (cuImportExternalMemory, OPAQUE_FD) and maps it to a
//       CUdeviceptr. If that succeeds, the 5090's copy engine can PULL the
//       doorbell result straight out of B70 VRAM over its own Gen5 x16 link:
//       one PCIe hop, no host-DRAM bounce, only documented APIs.
//
// Both directions are needed for a verdict on "can these two vendors share
// memory at all on this box". This one costs no driver change and touches no
// raw BAR mapping.
//
// What we measure if the handshake survives:
//   * correctness: B70 writes a pattern, CUDA reads it through the import.
//   * pull latency at 12 KiB (the M=1 doorbell payload shape).
//   * pull bandwidth at 32 MiB.
//   * baseline for the same bytes: CUDA H2D from pinned host, i.e. the second
//     leg of today's two-hop return path. The pull must beat B70-D2H + H2D
//     combined to be worth wiring in.
//
// Build:
//   g++ -O2 -o experiments/xvendor_p2p_probe experiments/xvendor_p2p_probe.cpp \
//       -I/usr/local/cuda-13/include -lcuda -lze_loader
//
// DVFS caveat inherited from kill-bench rule 7: B70-side numbers want pinned
// clocks before quoting absolutes; ratios measured back-to-back are the
// trustworthy part.

#include <cuda.h>
#include <level_zero/ze_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_stage_failures = 0;

void report_kill(const char* stage, const char* what, long code) {
  std::fprintf(stderr, "KILL at %s: %s (code %ld)\n", stage, what, code);
  std::printf("{\"verdict\":\"KILLED\",\"stage\":\"%s\",\"reason\":\"%s\","
              "\"code\":%ld}\n", stage, what, code);
  ++g_stage_failures;
}

[[noreturn]] void die(const char* stage, const char* what, long code) {
  report_kill(stage, what, code);
  std::exit(1);
}

void cu_check(CUresult r, const char* stage, const char* what) {
  if (r == CUDA_SUCCESS) return;
  const char* name = nullptr;
  cuGetErrorName(r, &name);
  die(stage, (std::string(what) + ": " + (name ? name : "?")).c_str(),
      static_cast<long>(r));
}

void ze_check(ze_result_t r, const char* stage, const char* what) {
  if (r == ZE_RESULT_SUCCESS) return;
  die(stage, what, static_cast<long>(r));
}

double now_us() {
  return std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

struct Stats {
  double mean_us, p50_us, p99_us;
};

// One synchronous CUDA copy per iteration: the production shape, where the
// 5090 must have the bytes in VRAM before the next layer proceeds.
Stats time_cuda_copy(CUdeviceptr dst, CUdeviceptr src, size_t bytes, int iters,
                     const char* stage) {
  std::vector<double> s;
  s.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    const double t0 = now_us();
    cu_check(cuMemcpyDtoD(dst, src, bytes), stage, "cuMemcpyDtoD");
    cu_check(cuCtxSynchronize(), stage, "sync");
    s.push_back(now_us() - t0);
  }
  std::sort(s.begin(), s.end());
  double sum = 0;
  for (double v : s) sum += v;
  return {sum / s.size(), s[s.size() / 2], s[static_cast<size_t>(s.size() * 0.99)]};
}

Stats time_h2d(CUdeviceptr dst, const void* host_src, size_t bytes, int iters,
               const char* stage) {
  std::vector<double> s;
  s.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    const double t0 = now_us();
    cu_check(cuMemcpyHtoD(dst, host_src, bytes), stage, "cuMemcpyHtoD");
    cu_check(cuCtxSynchronize(), stage, "sync");
    s.push_back(now_us() - t0);
  }
  std::sort(s.begin(), s.end());
  double sum = 0;
  for (double v : s) sum += v;
  return {sum / s.size(), s[s.size() / 2], s[static_cast<size_t>(s.size() * 0.99)]};
}

}  // namespace

int main() {
  constexpr size_t kBytes = 32ull << 20;
  constexpr uint32_t kPattern = 0xB70D1BEEu;

  // ---- Level Zero: find the B70, check export caps -----------------------
  ze_check(zeInit(ZE_INIT_FLAG_GPU_ONLY), "8g", "zeInit");
  uint32_t n_drv = 0;
  ze_check(zeDriverGet(&n_drv, nullptr), "8g", "zeDriverGet count");
  std::vector<ze_driver_handle_t> drvs(n_drv);
  ze_check(zeDriverGet(&n_drv, drvs.data()), "8g", "zeDriverGet");

  ze_device_handle_t dev = nullptr;
  ze_driver_handle_t drv = nullptr;
  std::string bdf;
  for (auto d_drv : drvs) {
    uint32_t n_dev = 0;
    if (zeDeviceGet(d_drv, &n_dev, nullptr) != ZE_RESULT_SUCCESS) continue;
    std::vector<ze_device_handle_t> devs(n_dev);
    zeDeviceGet(d_drv, &n_dev, devs.data());
    for (auto d : devs) {
      ze_device_properties_t p{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
      if (zeDeviceGetProperties(d, &p) != ZE_RESULT_SUCCESS) continue;
      if (p.type != ZE_DEVICE_TYPE_GPU) continue;
      if (std::strstr(p.name, "B70") == nullptr) continue;
      ze_pci_ext_properties_t pci{ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES};
      char b[32] = "????";
      if (zeDevicePciGetPropertiesExt(d, &pci) == ZE_RESULT_SUCCESS)
        std::snprintf(b, sizeof(b), "%04x:%02x:%02x.%x", pci.address.domain,
                      pci.address.bus, pci.address.device, pci.address.function);
      std::fprintf(stderr, "[8g] B70 '%s' at %s\n", p.name, b);
      if (dev == nullptr || std::strcmp(b, "0000:15:00.0") == 0) {
        dev = d; drv = d_drv; bdf = b;
      }
    }
  }
  if (dev == nullptr) die("8g", "no B70 Level Zero device", -1);
  std::fprintf(stderr, "[8g] using B70 %s (Gen4 preferred)\n", bdf.c_str());

  ze_device_external_memory_properties_t ext{
      ZE_STRUCTURE_TYPE_DEVICE_EXTERNAL_MEMORY_PROPERTIES};
  ze_check(zeDeviceGetExternalMemoryProperties(dev, &ext), "8g",
           "external memory properties");
  std::fprintf(stderr,
               "[8g] B70 export types alloc=0x%x  import types alloc=0x%x\n",
               ext.memoryAllocationExportTypes,
               ext.memoryAllocationImportTypes);
  const bool can_export =
      (ext.memoryAllocationExportTypes & ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF) != 0;
  std::fprintf(stderr, "[8g] DMA_BUF export advertised: %s\n",
               can_export ? "YES" : "NO");
  if (!can_export)
    die("8g", "B70 does not advertise DMA_BUF export for allocations", 0);

  ze_context_handle_t ctx;
  ze_context_desc_t ctx_desc{ZE_STRUCTURE_TYPE_CONTEXT_DESC};
  ze_check(zeContextCreate(drv, &ctx_desc, &ctx), "8g", "zeContextCreate");

  // Allocate B70 device memory WITH the dma-buf export flag attached.
  ze_external_memory_export_desc_t exp_desc{
      ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_DESC, nullptr,
      ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF};
  ze_device_mem_alloc_desc_t alloc_desc{
      ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, &exp_desc, 0, 0};
  void* b70_buf = nullptr;
  ze_check(zeMemAllocDevice(ctx, &alloc_desc, kBytes, 4096, dev, &b70_buf),
           "8g", "zeMemAllocDevice(exportable)");

  // Retrieve the fd for that allocation.
  ze_external_memory_export_fd_t exp_fd{
      ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_FD, nullptr,
      ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF, 0};
  ze_memory_allocation_properties_t mem_props{
      ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES, &exp_fd};
  ze_check(zeMemGetAllocProperties(ctx, b70_buf, &mem_props, nullptr), "8g",
           "zeMemGetAllocProperties(export fd)");
  if (exp_fd.fd <= 0)
    die("8g", "Level Zero returned no usable dma-buf fd", exp_fd.fd);
  std::fprintf(stderr, "[8g] EXPORT OK: B70 dma-buf fd=%d for %zu MiB of "
               "B70 VRAM\n", exp_fd.fd, kBytes >> 20);

  // Stamp the pattern from the B70 side so any readback proves provenance.
  ze_command_queue_desc_t q{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0,
                            0, ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
                            ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  ze_command_list_handle_t list;
  ze_check(zeCommandListCreateImmediate(ctx, dev, &q, &list), "8g",
           "immediate command list");
  ze_check(zeCommandListAppendMemoryFill(list, b70_buf, &kPattern,
                                         sizeof(kPattern), kBytes, nullptr, 0,
                                         nullptr),
           "8g", "B70-side pattern fill");
  std::fprintf(stderr, "[8g] B70 stamped 0x%08x across its buffer\n", kPattern);

  // ---- CUDA: import the foreign fd ---------------------------------------
  cu_check(cuInit(0), "8g", "cuInit");
  int n_cu = 0;
  cu_check(cuDeviceGetCount(&n_cu), "8g", "cuDeviceGetCount");
  CUdevice cu_dev = -1;
  char cu_name[256] = {0};
  for (int i = 0; i < n_cu; ++i) {
    CUdevice d;
    cu_check(cuDeviceGet(&d, i), "8g", "cuDeviceGet");
    cu_check(cuDeviceGetName(cu_name, sizeof(cu_name), d), "8g", "name");
    if (std::strstr(cu_name, "5090")) { cu_dev = d; break; }
  }
  if (cu_dev < 0) die("8g", "no RTX 5090", -1);
  CUcontext cu_ctx;
  cu_check(cuCtxCreate(&cu_ctx, nullptr, 0, cu_dev), "8g", "cuCtxCreate");
  std::fprintf(stderr, "[8g] CUDA device: %s\n", cu_name);

  CUDA_EXTERNAL_MEMORY_HANDLE_DESC h{};
  h.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
  h.handle.fd = exp_fd.fd;
  h.size = kBytes;
  CUexternalMemory ext_mem{};
  const CUresult import_rc = cuImportExternalMemory(&ext_mem, &h);
  if (import_rc != CUDA_SUCCESS) {
    const char* nm = nullptr;
    cuGetErrorName(import_rc, &nm);
    std::fprintf(stderr,
                 "[8g] IMPORT REFUSED: cuImportExternalMemory = %s. NVIDIA's "
                 "OPAQUE_FD accepts only its own exports, not a foreign "
                 "dma-buf.\n", nm ? nm : "?");
    die("8g", "cuImportExternalMemory rejected the Level Zero dma-buf fd",
        static_cast<long>(import_rc));
  }
  std::fprintf(stderr, "[8g] IMPORT OK: CUDA accepted the B70 dma-buf\n");

  CUDA_EXTERNAL_MEMORY_BUFFER_DESC bd{};
  bd.offset = 0;
  bd.size = kBytes;
  CUdeviceptr imported = 0;
  cu_check(cuExternalMemoryGetMappedBuffer(&imported, ext_mem, &bd), "8g",
           "cuExternalMemoryGetMappedBuffer");
  std::fprintf(stderr, "[8g] mapped B70 VRAM at CUdeviceptr 0x%llx\n",
               static_cast<unsigned long long>(imported));

  // ---- correctness: does the 5090 actually see the B70's bytes? ----------
  std::vector<uint32_t> host(kBytes / 4, 0);
  cu_check(cuMemcpyDtoH(host.data(), imported, kBytes), "8g",
           "readback through import");
  size_t bad = 0;
  for (uint32_t v : host)
    if (v != kPattern) ++bad;
  if (bad != 0) {
    std::fprintf(stderr, "[8g] MISMATCH: %zu/%zu words wrong\n", bad,
                 host.size());
    die("8g", "imported mapping does not read back the B70's pattern",
        static_cast<long>(bad));
  }
  std::fprintf(stderr,
               "[8g] CORRECTNESS OK: all %zu words the 5090 read match the "
               "pattern the B70 wrote -- cross-vendor shared mapping is real\n",
               host.size());

  // ---- numbers: pull vs today's second hop -------------------------------
  CUdeviceptr dst = 0;
  cu_check(cuMemAlloc(&dst, kBytes), "8g", "5090 destination alloc");
  void* pinned = nullptr;
  cu_check(cuMemHostAlloc(&pinned, kBytes, CU_MEMHOSTALLOC_PORTABLE), "8g",
           "pinned host baseline alloc");
  std::memset(pinned, 0x5A, kBytes);

  struct Row { const char* path; size_t bytes; Stats s; };
  std::vector<Row> rows;
  const size_t sizes[] = {12288, 65536};
  constexpr int kWarm = 100, kIters = 1000;
  for (size_t sz : sizes) {
    time_cuda_copy(dst, imported, sz, kWarm, "8g");
    rows.push_back({"pull_b70vram_to_5090", sz,
                    time_cuda_copy(dst, imported, sz, kIters, "8g")});
    time_h2d(dst, pinned, sz, kWarm, "8g");
    rows.push_back({"h2d_pinnedhost_to_5090", sz,
                    time_h2d(dst, pinned, sz, kIters, "8g")});
  }

  auto bandwidth_pull = [&]() {
    const int iters = 16;
    cu_check(cuMemcpyDtoD(dst, imported, kBytes), "8g", "bw warm");
    cu_check(cuCtxSynchronize(), "8g", "bw warm sync");
    const double t0 = now_us();
    for (int i = 0; i < iters; ++i)
      cu_check(cuMemcpyDtoD(dst, imported, kBytes), "8g", "bw copy");
    cu_check(cuCtxSynchronize(), "8g", "bw sync");
    return (static_cast<double>(kBytes) * iters) / ((now_us() - t0) * 1e-6) / 1e9;
  };
  auto bandwidth_h2d = [&]() {
    const int iters = 16;
    cu_check(cuMemcpyHtoD(dst, pinned, kBytes), "8g", "bw warm");
    cu_check(cuCtxSynchronize(), "8g", "bw warm sync");
    const double t0 = now_us();
    for (int i = 0; i < iters; ++i)
      cu_check(cuMemcpyHtoD(dst, pinned, kBytes), "8g", "bw copy");
    cu_check(cuCtxSynchronize(), "8g", "bw sync");
    return (static_cast<double>(kBytes) * iters) / ((now_us() - t0) * 1e-6) / 1e9;
  };
  const double bw_pull = bandwidth_pull();
  const double bw_h2d = bandwidth_h2d();

  std::fprintf(stderr, "\n%-26s %9s %10s %10s %10s\n", "path", "bytes",
               "mean us", "p50 us", "p99 us");
  for (const auto& r : rows)
    std::fprintf(stderr, "%-26s %9zu %10.2f %10.2f %10.2f\n", r.path, r.bytes,
                 r.s.mean_us, r.s.p50_us, r.s.p99_us);
  std::fprintf(stderr, "bandwidth 32 MiB: pull=%.2f GB/s  h2d=%.2f GB/s\n",
               bw_pull, bw_h2d);

  std::printf("{\"verdict\":\"SURVIVED\",\"b70\":\"%s\",\"cuda\":\"%s\","
              "\"correctness_words\":%zu,\"rows\":[",
              bdf.c_str(), cu_name, host.size());
  for (size_t i = 0; i < rows.size(); ++i)
    std::printf("%s{\"path\":\"%s\",\"bytes\":%zu,\"mean_us\":%.3f,"
                "\"p50_us\":%.3f,\"p99_us\":%.3f}",
                i ? "," : "", rows[i].path, rows[i].bytes, rows[i].s.mean_us,
                rows[i].s.p50_us, rows[i].s.p99_us);
  std::printf("],\"bw_pull_gbps\":%.3f,\"bw_h2d_gbps\":%.3f}\n", bw_pull,
              bw_h2d);
  return 0;
}
