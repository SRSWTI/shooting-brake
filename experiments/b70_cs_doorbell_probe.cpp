// Doorbell 2.0 silicon probe: can the B70's command streamer replace the host
// poller?
//
// WHY THIS EXISTS
// ---------------
// Decode pays a measured ~61 us/layer handshake: CUDA writes a host flag, a
// pinned host thread spins on it, then submits SYCL work to the B70. Over 47
// layers that is ~2.9 ms of the 12.1 ms ITL -- the largest single attackable
// term left (kill-bench-level-up 15/17).
//
// The hypothesis (vendor recon, 2026-08-24): a pre-recorded Level-Zero command
// list can MI_SEMAPHORE_WAIT on a host memory word and run entirely on the
// command streamer -- no host thread in the loop, in either direction:
//
//   producer (5090/host) --store--> [word A] <--poll-- B70 command streamer
//   B70 CS  --PIPE_CONTROL write--> [word B] <--spin-- consumer (5090/host)
//
// Mechanism evidence (all vendored):
//   zexCommandListAppendWaitOnMemory emits a true Xe2 MI_SEMAPHORE_WAIT and
//   BMG instantiates the family    [compute-runtime cmdlist_hw.inl:4856-4935,
//                                   xe2_hpg_core/bmg/cmdlist_bmg.cpp:7-19]
//   zexCommandListAppendWriteToMemory on CCS is a stalling PIPE_CONTROL
//   post-sync write; HOST scope requests DC flush          [same, 4948-4998]
//   The watched word must be an L0-known allocation: arbitrary host pointers
//   risk a SILENT internal host copy that never observes external writes
//   [device.cpp:1807-1822]. The airtight import is zeMemAllocHost +
//   ze_external_memmap_sysmem_ext_desc_t: page-aligned, VA-preserving,
//   heap memory supported                          [ze_api.h:16505-16553]
//
// WHAT THIS MEASURES
// ------------------
//   Phase HOST: host store -> CS wakes -> CS writes B -> host sees B.
//   Phase CUDA (--cuda): the SAME page cudaHostRegister'd; the 5090 DMA-writes
//   the doorbell word. Proves cross-runtime coherence end to end.
//
// The number to beat is 61 us. Anything at or under ~15 us round-trip makes
// the full per-layer pre-recorded WAIT -> kernel -> WRITE chain worth building.
//
// Build:
//   source /opt/intel/oneapi/setvars.sh --force
//   icpx -O2 -std=c++20 experiments/b70_cs_doorbell_probe.cpp \
//        -I vendor/compute-runtime/level_zero/include \
//        -o experiments/b70_cs_doorbell_probe -lze_loader -ldl
// Run:
//   ./experiments/b70_cs_doorbell_probe 0000:15:00.0 [--cuda]

#include <level_zero/ze_api.h>
#include <level_zero/driver_experimental/zex_cmdlist.h>
#include <level_zero/driver_experimental/zex_common.h>

#include <dlfcn.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#define ZE_CHECK(call)                                                     \
  do {                                                                     \
    ze_result_t r_ = (call);                                               \
    if (r_ != ZE_RESULT_SUCCESS) {                                         \
      std::fprintf(stderr, "FAIL %s:%d %s -> 0x%x\n", __FILE__, __LINE__,  \
                   #call, r_);                                             \
      return 1;                                                            \
    }                                                                      \
  } while (0)

using WaitFn = ze_result_t (*)(zex_command_list_handle_t,
                               zex_wait_on_mem_desc_t*, void*, uint32_t,
                               zex_event_handle_t);
using WriteFn = ze_result_t (*)(zex_command_list_handle_t,
                                zex_write_to_mem_desc_t*, void*, uint64_t);

static double now_us() {
  return std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

static void report(const char* label, std::vector<double>& v) {
  std::sort(v.begin(), v.end());
  auto pct = [&](double p) { return v[size_t(p * (v.size() - 1))]; };
  std::printf("%-22s n=%zu  min=%6.2f us  p50=%6.2f us  p90=%6.2f us  "
              "p99=%6.2f us\n",
              label, v.size(), v.front(), pct(0.5), pct(0.9), pct(0.99));
}

int main(int argc, char** argv) {
  const std::string want_bdf = argc > 1 ? argv[1] : "0000:15:00.0";
  const bool use_cuda =
      argc > 2 && std::string(argv[2]) == "--cuda";
  constexpr int kIters = 2000;
  constexpr int kWarmup = 100;

  ZE_CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));

  // -- device by BDF ---------------------------------------------------------
  uint32_t nd = 0;
  ZE_CHECK(zeDriverGet(&nd, nullptr));
  std::vector<ze_driver_handle_t> drivers(nd);
  ZE_CHECK(zeDriverGet(&nd, drivers.data()));
  ze_driver_handle_t driver{};
  ze_device_handle_t device{};
  for (auto d : drivers) {
    uint32_t nn = 0;
    zeDeviceGet(d, &nn, nullptr);
    std::vector<ze_device_handle_t> devs(nn);
    zeDeviceGet(d, &nn, devs.data());
    for (auto dev : devs) {
      ze_pci_ext_properties_t pci{ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES};
      if (zeDevicePciGetPropertiesExt(dev, &pci) != ZE_RESULT_SUCCESS)
        continue;
      char bdf[32];
      std::snprintf(bdf, sizeof bdf, "%04x:%02x:%02x.%x", pci.address.domain,
                    pci.address.bus, pci.address.device, pci.address.function);
      if (want_bdf == bdf) {
        driver = d;
        device = dev;
      }
    }
  }
  if (!device) {
    std::fprintf(stderr, "device %s not found\n", want_bdf.c_str());
    return 1;
  }
  std::printf("device: %s\n", want_bdf.c_str());

  ze_context_handle_t ctx{};
  ze_context_desc_t cdesc{ZE_STRUCTURE_TYPE_CONTEXT_DESC};
  ZE_CHECK(zeContextCreate(driver, &cdesc, &ctx));

  // -- the shared page: L0 host USM FIRST, CUDA registers it second ----------
  // Measured on this driver (sweep below, kept for regression): every
  // action/scope combination on an external-sysmem IMPORTED page returns
  // INVALID_ARGUMENT for appendWaitOnMemory, while plain host USM is accepted.
  // So the shared word must be born as L0 host USM; the CUDA side registers
  // the same VA afterwards (cudaHostRegister on already-pinned memory).
  const size_t page = size_t(sysconf(_SC_PAGESIZE));
  ze_host_mem_alloc_desc_t hdesc{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
  void* raw = nullptr;
  ZE_CHECK(zeMemAllocHost(ctx, &hdesc, page, page, &raw));
  std::memset(raw, 0, page);
  void* mapped = raw;  // freed via zeMemFree at exit
  std::printf("L0 host-USM page at %p\n", raw);

  // doorbell word A and completion word B on separate cache lines
  auto* A = reinterpret_cast<volatile uint32_t*>(raw);
  auto* B = reinterpret_cast<volatile uint32_t*>(
      static_cast<char*>(raw) + 128);

  // -- zex entry points -------------------------------------------------------
  WaitFn zex_wait{};
  WriteFn zex_write{};
  ZE_CHECK(zeDriverGetExtensionFunctionAddress(
      driver, "zexCommandListAppendWaitOnMemory",
      reinterpret_cast<void**>(&zex_wait)));
  ZE_CHECK(zeDriverGetExtensionFunctionAddress(
      driver, "zexCommandListAppendWriteToMemory",
      reinterpret_cast<void**>(&zex_write)));
  std::printf("zex entry points resolved\n");

  // -- pre-recorded list: WAIT(A==1, acquire) ; WRITE(B=1, host scope) -------
  // This is the shape of one production layer minus the kernel: the wait is
  // the doorbell, the write is the completion. Reused across iterations.
  ze_command_queue_desc_t qdesc{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
  qdesc.ordinal = 0;  // COMPUTE|COPY group (recon: group 0, numQueues=1)
  qdesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  ze_command_queue_handle_t queue{};
  ZE_CHECK(zeCommandQueueCreate(ctx, device, &qdesc, &queue));

  ze_command_list_desc_t ldesc{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
  ldesc.commandQueueGroupOrdinal = 0;
  ze_command_list_handle_t list{};
  ZE_CHECK(zeCommandListCreate(ctx, device, &ldesc, &list));

  zex_wait_on_mem_desc_t wdesc{};
  wdesc.actionFlag = ZEX_WAIT_ON_MEMORY_FLAG_EQUAL;
  wdesc.waitScope = ZEX_MEM_ACTION_SCOPE_FLAG_HOST;
  ze_result_t wres =
      zex_wait(reinterpret_cast<zex_command_list_handle_t>(list), &wdesc,
               const_cast<uint32_t*>(A), 1u, nullptr);
  if (wres != ZE_RESULT_SUCCESS) {
    // The installed driver's validation may differ from the vendored source.
    // Diagnose: sweep action x scope on the imported page, then retry on a
    // plain (non-imported) host-USM word to isolate the allocation kind.
    std::printf("first wait append rejected (0x%x); sweeping...\n", wres);
    for (uint32_t act : {1u, 2u, 4u, 8u, 16u, 32u}) {
      for (uint32_t scope : {0u, 2u, 4u, 6u}) {
        zex_wait_on_mem_desc_t d{};
        d.actionFlag = act;
        d.waitScope = scope;
        ze_result_t r =
            zex_wait(reinterpret_cast<zex_command_list_handle_t>(list), &d,
                     const_cast<uint32_t*>(A), 1u, nullptr);
        std::printf("  imported page: act=%2u scope=%u -> 0x%x\n", act, scope,
                    r);
        if (r == ZE_RESULT_SUCCESS) {
          wdesc = d;
          wres = r;
          goto found;
        }
      }
    }
    {
      void* husm = nullptr;
      ze_host_mem_alloc_desc_t hd{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
      if (zeMemAllocHost(ctx, &hd, 64, 64, &husm) == ZE_RESULT_SUCCESS) {
        zex_wait_on_mem_desc_t d{};
        d.actionFlag = ZEX_WAIT_ON_MEMORY_FLAG_EQUAL;
        d.waitScope = 0;
        ze_result_t r =
            zex_wait(reinterpret_cast<zex_command_list_handle_t>(list), &d,
                     husm, 1u, nullptr);
        std::printf("  plain host USM: act=1 scope=0 -> 0x%x\n", r);
        zeMemFree(ctx, husm);
      }
    }
    std::fprintf(stderr, "no accepted wait combination -- report and stop\n");
    return 4;
  }
found:
  std::printf("wait accepted: act=%u scope=%u\n", wdesc.actionFlag,
              wdesc.waitScope);
  zex_write_to_mem_desc_t wrdesc{};
  wrdesc.writeScope = ZEX_MEM_ACTION_SCOPE_FLAG_HOST;  // DC flush -> host sees
  ZE_CHECK(zex_write(reinterpret_cast<zex_command_list_handle_t>(list), &wrdesc,
                     const_cast<uint32_t*>(B), 1ull));
  ZE_CHECK(zeCommandListClose(list));
  std::printf("pre-recorded WAIT->WRITE list closed\n");

  // -- phase HOST: host store rings the doorbell ------------------------------
  {
    std::vector<double> rt;
    rt.reserve(kIters);
    for (int i = 0; i < kWarmup + kIters; ++i) {
      *A = 0;
      *B = 0;
      std::atomic_thread_fence(std::memory_order_seq_cst);
      ZE_CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
      // Let the CS actually reach the semaphore poll before ringing: without
      // this the measurement includes submission, which production prerecords.
      std::this_thread::sleep_for(std::chrono::microseconds(150));
      const double t0 = now_us();
      reinterpret_cast<volatile std::atomic<uint32_t>*>(A)->store(
          1, std::memory_order_release);
      while (*B != 1u) {
      }
      const double t1 = now_us();
      ZE_CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));
      if (i >= kWarmup) rt.push_back(t1 - t0);
    }
    report("HOST ring -> CS -> B", rt);
  }

  // -- phase CUDA: the 5090 rings the doorbell over PCIe ----------------------
  if (use_cuda) {
    void* cudart = dlopen("libcudart.so.13", RTLD_NOW | RTLD_GLOBAL);
    if (!cudart) cudart = dlopen("libcudart.so.12", RTLD_NOW | RTLD_GLOBAL);
    if (!cudart) cudart = dlopen("libcudart.so", RTLD_NOW | RTLD_GLOBAL);
    if (!cudart) {
      std::fprintf(stderr, "cudart unavailable: %s\n", dlerror());
      return 3;
    }
    // cudaHostRegister of L0 host USM fails on this stack (file-backed VMA),
    // so the composition is reversed: CUDA ALLOCATES the pinned page and the
    // CS waits on it directly. NEO resolves unknown host pointers through
    // allocateMemoryFromHostPtr, which pins page-locked memory directly --
    // but may SILENTLY fall back to an internal host copy that never sees
    // CUDA's writes (device.cpp:1807). So this arm VERIFIES end to end with
    // a bounded spin: B arriving proves the CS watched the real page.
    auto host_alloc = reinterpret_cast<int (*)(void**, size_t, unsigned)>(
        dlsym(cudart, "cudaHostAlloc"));
    auto memset_fn = reinterpret_cast<int (*)(void*, int, size_t)>(
        dlsym(cudart, "cudaMemset"));
    auto get_dev_ptr = reinterpret_cast<int (*)(void**, void*, unsigned)>(
        dlsym(cudart, "cudaHostGetDevicePointer"));
    auto sync_fn = reinterpret_cast<int (*)()>(
        dlsym(cudart, "cudaDeviceSynchronize"));
    void* cpage = nullptr;
    // cudaHostAllocMapped(0x02) | cudaHostAllocPortable(0x01)
    if (host_alloc(&cpage, page, 0x03) != 0) {
      std::fprintf(stderr, "cudaHostAlloc failed\n");
      return 3;
    }
    std::memset(cpage, 0, page);
    auto* cA = reinterpret_cast<volatile uint32_t*>(cpage);
    auto* cB = reinterpret_cast<volatile uint32_t*>(
        static_cast<char*>(cpage) + 128);
    void* dA = nullptr;
    if (get_dev_ptr(&dA, const_cast<uint32_t*>(cA), 0) != 0) {
      std::fprintf(stderr, "cudaHostGetDevicePointer failed\n");
      return 3;
    }

    ze_command_list_handle_t clist{};
    ZE_CHECK(zeCommandListCreate(ctx, device, &ldesc, &clist));
    ze_result_t cw =
        zex_wait(reinterpret_cast<zex_command_list_handle_t>(clist), &wdesc,
                 const_cast<uint32_t*>(cA), 1u, nullptr);
    ze_result_t cwr =
        zex_write(reinterpret_cast<zex_command_list_handle_t>(clist), &wrdesc,
                  const_cast<uint32_t*>(cB), 1ull);
    std::printf("CUDA-page appends: wait=0x%x write=0x%x\n", cw, cwr);
    if (cw != ZE_RESULT_SUCCESS || cwr != ZE_RESULT_SUCCESS) {
      std::fprintf(stderr, "CS refuses the CUDA-pinned page -- cross-runtime "
                           "arm blocked; host-relay remains the fallback\n");
      return 5;
    }
    ZE_CHECK(zeCommandListClose(clist));

    std::vector<double> rt;
    rt.reserve(kIters);
    int timeouts = 0;
    for (int i = 0; i < kWarmup + kIters; ++i) {
      *cA = 0;
      *cB = 0;
      std::atomic_thread_fence(std::memory_order_seq_cst);
      ZE_CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &clist, nullptr));
      std::this_thread::sleep_for(std::chrono::microseconds(150));
      const double t0 = now_us();
      memset_fn(dA, 1, 1);  // 5090 DMA write: word becomes 0x00000001 (LE)
      // Bounded spin: a silent-copy fallback would never deliver B. That is
      // the trap this arm exists to detect -- fail loudly, not hang.
      while (*cB != 1u) {
        if (now_us() - t0 > 2e5) {
          ++timeouts;
          reinterpret_cast<volatile std::atomic<uint32_t>*>(cA)->store(
              1, std::memory_order_release);  // host rescue so the list retires
          break;
        }
      }
      const double t1 = now_us();
      ZE_CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));
      sync_fn();
      if (i >= kWarmup && *cB == 1u && t1 - t0 < 2e5) rt.push_back(t1 - t0);
    }
    if (timeouts) {
      std::printf("CUDA arm: %d/%d TIMEOUTS -- silent-copy trap CONFIRMED on "
                  "CUDA-pinned pages; cross-runtime doorbell needs another "
                  "path\n", timeouts, kWarmup + kIters);
    }
    if (!rt.empty()) report("CUDA ring -> CS -> B", rt);
    ZE_CHECK(zeCommandListDestroy(clist));
  }

  std::printf("\nbaseline to beat: 61 us host-poller handshake "
              "(x47 layers = 2.9 ms/token)\n");
  zeCommandListDestroy(list);
  zeCommandQueueDestroy(queue);
  zeMemFree(ctx, mapped);
  zeContextDestroy(ctx);
  return 0;
}
