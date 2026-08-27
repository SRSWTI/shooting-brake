// Kill-bench #8: RBAR cross-vendor direct write probe.
//
// Question: can the Arc Pro B70 DMA directly into RTX 5090 VRAM through the
// kernel's vendor-neutral dma-buf interconnect?
//
//   8b. CUDA exports a 5090 device allocation as a dma-buf fd
//       (cuMemGetHandleForAddressRange, open kernel modules only).
//   8c. Level Zero imports that fd as a device allocation on the B70
//       (ze_external_memory_import_fd_t, DMA_BUF flag).
//   8d. The B70 copy engine writes a pattern through the imported pointer;
//       CUDA reads it back and verifies bytes. The nvidia exporter maps BAR1
//       pages for the importer's device -- there is no silent host-bounce
//       mode -- so byte-correct == direct.
//   8e. Latency (4 B flag, 12 KiB decode payload, 64 KiB) and bandwidth
//       (64 MiB) of B70->imported(5090) vs B70->pinned-host (today's
//       doorbell return hop), both on the copy ordinal and compute ordinal.
//
// Build:
//   g++ -O2 -o experiments/rbar_dmabuf_probe experiments/rbar_dmabuf_probe.cpp \
//       -I/usr/local/cuda-13/include -lcuda -lze_loader
//
// DVFS caveat: B70 latency numbers swing 2.6x with cold clocks
// (kill-bench rule 7 lineage). Every timed section runs a long warm loop and
// the BAR-vs-host comparison shares one clock state, so the RATIO is the
// trustworthy number; absolutes want a pinned-clock rerun before quoting.

#include <cuda.h>
#include <level_zero/ze_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

[[noreturn]] void die(const char* stage, const char* what, long code) {
  std::fprintf(stderr, "KILL at %s: %s (code %ld)\n", stage, what, code);
  std::printf("{\"verdict\":\"KILLED\",\"stage\":\"%s\",\"reason\":\"%s\",\"code\":%ld}\n",
              stage, what, code);
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

struct TimedStats {
  double mean_us;
  double p50_us;
  double p99_us;
};

// Times one synchronous-immediate-list copy per iteration: submit + execute +
// completion, the same shape as a production doorbell leg.
TimedStats time_copies(ze_command_list_handle_t list, void* dst,
                       const void* src, size_t bytes, int iters,
                       const char* stage) {
  std::vector<double> samples;
  samples.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    const double t0 = now_us();
    ze_check(zeCommandListAppendMemoryCopy(list, dst, src, bytes, nullptr, 0,
                                           nullptr),
             stage, "append memory copy");
    samples.push_back(now_us() - t0);
  }
  std::sort(samples.begin(), samples.end());
  double sum = 0;
  for (double s : samples) sum += s;
  return {sum / samples.size(), samples[samples.size() / 2],
          samples[static_cast<size_t>(samples.size() * 0.99)]};
}

}  // namespace

int main() {
  // ---- CUDA side: find the 5090, allocate, export ------------------------
  cu_check(cuInit(0), "8b", "cuInit");
  int n_cuda = 0;
  cu_check(cuDeviceGetCount(&n_cuda), "8b", "cuDeviceGetCount");
  CUdevice cu_dev = -1;
  char cu_name[256] = {0};
  for (int i = 0; i < n_cuda; ++i) {
    CUdevice d;
    cu_check(cuDeviceGet(&d, i), "8b", "cuDeviceGet");
    cu_check(cuDeviceGetName(cu_name, sizeof(cu_name), d), "8b",
             "cuDeviceGetName");
    if (std::strstr(cu_name, "5090") != nullptr) {
      cu_dev = d;
      break;
    }
  }
  if (cu_dev < 0) die("8b", "no RTX 5090 CUDA device found", -1);
  std::fprintf(stderr, "[8b] CUDA device: %s\n", cu_name);

  int dmabuf_supported = 0;
  cu_check(cuDeviceGetAttribute(&dmabuf_supported,
                                CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, cu_dev),
           "8b", "query DMA_BUF_SUPPORTED");
  std::fprintf(stderr, "[8b] CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = %d\n",
               dmabuf_supported);
  // NVIDIA documents this attribute as advisory ("applications should check").
  // On 595.84 it reports 0 while the RM gate it maps to
  // (NV2080_CTRL_GPU_INFO_INDEX_DMABUF_CAPABILITY) is compile-time
  // CONFIG_DMA_SHARED_BUFFER, which this kernel has. So do NOT trust the
  // attribute: attempt the export and let the syscall be the verdict.
  if (!dmabuf_supported)
    std::fprintf(stderr,
                 "[8b] attribute says unsupported -- attempting export "
                 "anyway, the call is the real gate\n");

  CUcontext cu_ctx;
  cu_check(cuCtxCreate(&cu_ctx, nullptr, 0, cu_dev), "8b", "cuCtxCreate");

  constexpr size_t kRegionBytes = 64ull << 20;  // 64 MiB test window
  CUdeviceptr d_region = 0;
  cu_check(cuMemAlloc(&d_region, kRegionBytes), "8b", "cuMemAlloc 64 MiB");
  if (d_region % 4096 != 0)
    die("8b", "cuMemAlloc region not page-aligned; needs VMM fallback", 0);
  cu_check(cuMemsetD32(d_region, 0xA5A5A5A5u, kRegionBytes / 4), "8b",
           "pattern-A fill");
  cu_check(cuCtxSynchronize(), "8b", "ctx sync after fill");

  int dmabuf_fd = -1;
  cu_check(cuMemGetHandleForAddressRange(
               &dmabuf_fd, d_region, kRegionBytes,
               CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0),
           "8b", "cuMemGetHandleForAddressRange(DMA_BUF_FD)");
  std::fprintf(stderr, "[8b] EXPORT OK: dma-buf fd=%d for 64 MiB of 5090 VRAM\n",
               dmabuf_fd);

  // ---- Level Zero side: find the B70, report caps, import ----------------
  ze_check(zeInit(ZE_INIT_FLAG_GPU_ONLY), "8c", "zeInit");
  uint32_t n_drivers = 0;
  ze_check(zeDriverGet(&n_drivers, nullptr), "8c", "zeDriverGet count");
  std::vector<ze_driver_handle_t> drivers(n_drivers);
  ze_check(zeDriverGet(&n_drivers, drivers.data()), "8c", "zeDriverGet");

  ze_device_handle_t ze_dev = nullptr;
  ze_driver_handle_t ze_drv = nullptr;
  std::string ze_bdf;
  for (auto drv : drivers) {
    uint32_t n_dev = 0;
    if (zeDeviceGet(drv, &n_dev, nullptr) != ZE_RESULT_SUCCESS) continue;
    std::vector<ze_device_handle_t> devs(n_dev);
    zeDeviceGet(drv, &n_dev, devs.data());
    for (auto d : devs) {
      ze_device_properties_t props{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
      if (zeDeviceGetProperties(d, &props) != ZE_RESULT_SUCCESS) continue;
      if (props.type != ZE_DEVICE_TYPE_GPU) continue;
      if (std::strstr(props.name, "B70") == nullptr) continue;
      ze_pci_ext_properties_t pci{ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES};
      char bdf[32] = "????";
      if (zeDevicePciGetPropertiesExt(d, &pci) == ZE_RESULT_SUCCESS) {
        std::snprintf(bdf, sizeof(bdf), "%04x:%02x:%02x.%x",
                      pci.address.domain, pci.address.bus, pci.address.device,
                      pci.address.function);
      }
      std::fprintf(stderr, "[8c] found B70 '%s' at %s\n", props.name, bdf);
      // Prefer the Gen4 card (0000:15:00.0); fall back to any B70.
      if (ze_dev == nullptr || std::strcmp(bdf, "0000:15:00.0") == 0) {
        ze_dev = d;
        ze_drv = drv;
        ze_bdf = bdf;
      }
    }
  }
  if (ze_dev == nullptr) die("8c", "no B70 Level Zero device found", -1);
  std::fprintf(stderr, "[8c] using B70 at %s\n", ze_bdf.c_str());

  ze_device_external_memory_properties_t ext{
      ZE_STRUCTURE_TYPE_DEVICE_EXTERNAL_MEMORY_PROPERTIES};
  ze_check(zeDeviceGetExternalMemoryProperties(ze_dev, &ext), "8c",
           "external memory properties");
  std::fprintf(stderr,
               "[8c] B70 import types: alloc=0x%x imageIn=0x%x  "
               "export types: alloc=0x%x\n",
               ext.memoryAllocationImportTypes, ext.imageImportTypes,
               ext.memoryAllocationExportTypes);
  const bool import_ok =
      (ext.memoryAllocationImportTypes & ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF) != 0;
  if (!import_ok)
    die("8c", "B70 does not advertise DMA_BUF import for allocations", 0);

  ze_context_handle_t ze_ctx;
  ze_context_desc_t ctx_desc{ZE_STRUCTURE_TYPE_CONTEXT_DESC};
  ze_check(zeContextCreate(ze_drv, &ctx_desc, &ze_ctx), "8c", "zeContextCreate");

  ze_external_memory_import_fd_t import_fd{
      ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD, nullptr,
      ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF, dmabuf_fd};
  ze_device_mem_alloc_desc_t dev_alloc{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
                                       &import_fd, 0, 0};
  void* imported = nullptr;
  ze_result_t import_result =
      zeMemAllocDevice(ze_ctx, &dev_alloc, kRegionBytes, 4096, ze_dev, &imported);
  if (import_result != ZE_RESULT_SUCCESS)
    die("8c", "zeMemAllocDevice(dma-buf import) rejected the fd",
        static_cast<long>(import_result));
  std::fprintf(stderr, "[8c] IMPORT OK: B70 pointer %p over 5090 VRAM\n",
               imported);

  // ---- Command lists: copy ordinal + compute ordinal ---------------------
  uint32_t n_groups = 0;
  ze_check(zeDeviceGetCommandQueueGroupProperties(ze_dev, &n_groups, nullptr),
           "8d", "queue group count");
  std::vector<ze_command_queue_group_properties_t> groups(
      n_groups, {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES});
  ze_check(zeDeviceGetCommandQueueGroupProperties(ze_dev, &n_groups,
                                                  groups.data()),
           "8d", "queue group props");
  uint32_t copy_ordinal = UINT32_MAX, compute_ordinal = UINT32_MAX;
  for (uint32_t i = 0; i < n_groups; ++i) {
    const bool has_compute =
        groups[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE;
    const bool has_copy =
        groups[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY;
    if (has_compute && compute_ordinal == UINT32_MAX) compute_ordinal = i;
    if (has_copy && !has_compute && copy_ordinal == UINT32_MAX)
      copy_ordinal = i;
    std::fprintf(stderr, "[8d] queue group %u: compute=%d copy=%d queues=%u\n",
                 i, has_compute, has_copy, groups[i].numQueues);
  }
  if (compute_ordinal == UINT32_MAX) die("8d", "no compute ordinal", -1);
  const bool have_ce = copy_ordinal != UINT32_MAX;

  auto make_sync_list = [&](uint32_t ordinal) {
    ze_command_queue_desc_t q{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr,
                              ordinal, 0, 0, ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
                              ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
    ze_command_list_handle_t list;
    ze_check(zeCommandListCreateImmediate(ze_ctx, ze_dev, &q, &list), "8d",
             "immediate list");
    return list;
  };
  ze_command_list_handle_t list_ce =
      make_sync_list(have_ce ? copy_ordinal : compute_ordinal);
  ze_command_list_handle_t list_cc = make_sync_list(compute_ordinal);

  // ---- B70-local source + pinned-host comparison target ------------------
  void* b70_src = nullptr;
  ze_device_mem_alloc_desc_t plain_dev{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
  ze_check(zeMemAllocDevice(ze_ctx, &plain_dev, kRegionBytes, 4096, ze_dev,
                            &b70_src),
           "8d", "B70 local source alloc");
  const uint32_t kPatternB = 0x5B70CAFEu;
  ze_check(zeCommandListAppendMemoryFill(list_cc, b70_src, &kPatternB,
                                         sizeof(kPatternB), kRegionBytes,
                                         nullptr, 0, nullptr),
           "8d", "pattern-B fill on B70");

  void* host_target = nullptr;
  ze_host_mem_alloc_desc_t host_desc{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
  ze_check(zeMemAllocHost(ze_ctx, &host_desc, kRegionBytes, 4096, &host_target),
           "8e", "pinned host target alloc");

  // ---- 8d: correctness ----------------------------------------------------
  ze_check(zeCommandListAppendMemoryCopy(list_ce, imported, b70_src,
                                         kRegionBytes, nullptr, 0, nullptr),
           "8d", "B70 -> imported(5090) full-region write");

  std::vector<uint32_t> readback(kRegionBytes / 4);
  cu_check(cuMemcpyDtoH(readback.data(), d_region, kRegionBytes), "8d",
           "CUDA readback");
  size_t bad = 0;
  for (uint32_t v : readback)
    if (v != kPatternB) ++bad;
  if (bad != 0) {
    std::fprintf(stderr, "[8d] MISMATCH: %zu of %zu words wrong\n", bad,
                 readback.size());
    die("8d", "pattern did not land in 5090 VRAM", static_cast<long>(bad));
  }
  std::fprintf(stderr,
               "[8d] CORRECTNESS OK: all %zu words in 5090 VRAM match the "
               "B70's pattern -- the DMA is direct by construction\n",
               readback.size());

  // ---- 8e: latency + bandwidth, imported vs host, CE vs compute ----------
  struct Row {
    const char* path;
    const char* engine;
    size_t bytes;
    TimedStats stats;
  };
  std::vector<Row> rows;
  const size_t sizes[] = {4, 12288, 65536};  // flag, M=1 fp32 payload, chunk
  constexpr int kWarm = 200, kIters = 2000;

  auto bench_path = [&](const char* path, void* dst,
                        ze_command_list_handle_t list, const char* engine) {
    for (size_t sz : sizes) {
      time_copies(list, dst, b70_src, sz, kWarm, "8e");  // warm
      rows.push_back({path, engine, sz, time_copies(list, dst, b70_src, sz,
                                                    kIters, "8e")});
    }
  };
  bench_path("b70_to_5090_bar", imported, list_ce, have_ce ? "copy" : "compute");
  bench_path("b70_to_pinned_host", host_target, list_ce,
             have_ce ? "copy" : "compute");
  bench_path("b70_to_5090_bar_compute", imported, list_cc, "compute");

  auto bandwidth = [&](void* dst, ze_command_list_handle_t list) {
    const int iters = 8;
    time_copies(list, dst, b70_src, kRegionBytes, 2, "8e");  // warm
    const double t0 = now_us();
    for (int i = 0; i < iters; ++i)
      ze_check(zeCommandListAppendMemoryCopy(list, dst, b70_src, kRegionBytes,
                                             nullptr, 0, nullptr),
               "8e", "bandwidth copy");
    return (static_cast<double>(kRegionBytes) * iters) /
           ((now_us() - t0) * 1e-6) / 1e9;  // GB/s
  };
  const double bw_bar = bandwidth(imported, list_ce);
  const double bw_host = bandwidth(host_target, list_ce);

  // Reverse: B70 READS 5090 VRAM (non-posted, expect ugly).
  TimedStats rev = time_copies(list_ce, b70_src, imported, 65536, 200, "8e");

  // ---- report -------------------------------------------------------------
  std::fprintf(stderr, "\n%-26s %-8s %9s %10s %10s %10s\n", "path", "engine",
               "bytes", "mean us", "p50 us", "p99 us");
  for (const auto& r : rows)
    std::fprintf(stderr, "%-26s %-8s %9zu %10.2f %10.2f %10.2f\n", r.path,
                 r.engine, r.bytes, r.stats.mean_us, r.stats.p50_us,
                 r.stats.p99_us);
  std::fprintf(stderr, "bandwidth 64 MiB: bar=%.2f GB/s host=%.2f GB/s\n",
               bw_bar, bw_host);
  std::fprintf(stderr, "reverse read 64 KiB (5090->B70 pull): mean %.2f us\n",
               rev.mean_us);

  std::printf("{\"verdict\":\"SURVIVED\",\"b70\":\"%s\",\"dmabuf_fd\":%d,", 
              ze_bdf.c_str(), dmabuf_fd);
  std::printf("\"correctness_words\":%zu,\"rows\":[", readback.size());
  for (size_t i = 0; i < rows.size(); ++i)
    std::printf("%s{\"path\":\"%s\",\"engine\":\"%s\",\"bytes\":%zu,"
                "\"mean_us\":%.3f,\"p50_us\":%.3f,\"p99_us\":%.3f}",
                i ? "," : "", rows[i].path, rows[i].engine, rows[i].bytes,
                rows[i].stats.mean_us, rows[i].stats.p50_us,
                rows[i].stats.p99_us);
  std::printf("],\"bw_bar_gbps\":%.3f,\"bw_host_gbps\":%.3f,"
              "\"reverse_read_64k_mean_us\":%.3f}\n",
              bw_bar, bw_host, rev.mean_us);

  close(dmabuf_fd);
  return 0;
}
