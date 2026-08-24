// Probe #2 for Doorbell 2.0: can zex waits/writes ride the provider's own
// in-order SYCL queue?
//
// The integration design that avoids kernel-handle extraction entirely:
// pre-enqueue, one layer ahead, on the SAME in-order SYCL queue the provider
// already uses:
//
//   [zex WAIT(signal_L == seq)]  <- parks the CCS only after L-1's work drains
//   [H2D ring -> device]         <- normal SYCL memcpy
//   [kernels]                    <- normal SYCL submissions (quixicore/grouped)
//   [D2H device -> ring]
//   [zex WRITE(completion_L)]    <- PIPE_CONTROL host write, CUDA spin sees it
//
// For that to work, sycl::get_native on the queue must yield an L0 IMMEDIATE
// command list (append-executes-in-order), and zex appends on it must be
// accepted and correctly ordered against SYCL submissions.
//
// This probe answers, on the real device with the provider's queue shape:
//   1. does get_native return a command LIST (immediate) or QUEUE?
//   2. are zexCommandListAppendWaitOnMemory/WriteToMemory accepted on it?
//   3. ordering: WAIT(A==1) -> kernel(writes X) -> WRITE(B=1): does B arrive
//      only after the kernel's effect is host-visible? (release semantics)
//   4. round-trip latency of ring -> [WAIT->kernel->WRITE] -> host sees B,
//      i.e. the production critical path with a real (tiny) kernel inside.
//
// Build:
//   icpx -fsycl -O2 -std=c++20 experiments/b70_sycl_zex_interop_probe.cpp \
//        -I vendor/compute-runtime/level_zero/include \
//        -o experiments/b70_sycl_zex_interop_probe -lze_loader
// Run:  ONEAPI_DEVICE_SELECTOR=level_zero:* ./experiments/b70_sycl_zex_interop_probe 0000:15:00.0

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <level_zero/driver_experimental/zex_cmdlist.h>
#include <level_zero/driver_experimental/zex_common.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <variant>
#include <vector>

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

int main(int argc, char** argv) {
  const std::string want_bdf = argc > 1 ? argv[1] : "0000:15:00.0";

  // -- the provider's queue shape: in-order SYCL queue on the chosen B70 -----
  sycl::device dev;
  bool found = false;
  for (auto& d : sycl::device::get_devices(sycl::info::device_type::gpu)) {
    if (d.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
    // match by BDF via L0 PCI properties
    auto zdev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(d);
    ze_pci_ext_properties_t pci{ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES};
    if (zeDevicePciGetPropertiesExt(zdev, &pci) != ZE_RESULT_SUCCESS) continue;
    char bdf[32];
    std::snprintf(bdf, sizeof bdf, "%04x:%02x:%02x.%x", pci.address.domain,
                  pci.address.bus, pci.address.device, pci.address.function);
    if (want_bdf == bdf) {
      dev = d;
      found = true;
      break;
    }
  }
  if (!found) {
    std::fprintf(stderr, "B70 %s not found via SYCL\n", want_bdf.c_str());
    return 1;
  }
  sycl::queue q(dev, sycl::property::queue::in_order{});
  std::printf("SYCL in-order queue on %s\n", want_bdf.c_str());

  // -- (1) what does get_native give us? --------------------------------------
  auto native = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q);
  ze_command_list_handle_t ilist{};
  if (auto* pl = std::get_if<ze_command_list_handle_t>(&native)) {
    ilist = *pl;
    std::printf("get_native(queue) -> IMMEDIATE COMMAND LIST (%p)\n",
                (void*)ilist);
  } else {
    std::printf("get_native(queue) -> command QUEUE handle: immediate lists "
                "are OFF; the ride-along design needs "
                "UR_L0_USE_IMMEDIATE_COMMANDLISTS=1 -- probe BLOCKED\n");
    return 2;
  }

  auto zctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
      q.get_context());
  ze_driver_handle_t zdrv{};
  {
    uint32_t nd = 1;
    zeDriverGet(&nd, &zdrv);
  }
  WaitFn zex_wait{};
  WriteFn zex_write{};
  if (zeDriverGetExtensionFunctionAddress(
          zdrv, "zexCommandListAppendWaitOnMemory",
          reinterpret_cast<void**>(&zex_wait)) != ZE_RESULT_SUCCESS ||
      zeDriverGetExtensionFunctionAddress(
          zdrv, "zexCommandListAppendWriteToMemory",
          reinterpret_cast<void**>(&zex_write)) != ZE_RESULT_SUCCESS) {
    std::fprintf(stderr, "zex entry points missing\n");
    return 1;
  }

  // flags live in L0 host USM of THIS context (probe #1: accepted for wait);
  // in production they are the CUDA-pinned pages (probe #1: also accepted).
  auto* flags = sycl::malloc_host<uint32_t>(64, q);  // A=flags[0], B=flags[32]
  volatile uint32_t* A = flags;
  volatile uint32_t* B = flags + 32;
  auto* dx = sycl::malloc_device<uint32_t>(1024, q);
  auto* hx = sycl::malloc_host<uint32_t>(1024, q);

  zex_wait_on_mem_desc_t wdesc{};
  wdesc.actionFlag = ZEX_WAIT_ON_MEMORY_FLAG_EQUAL;
  wdesc.waitScope = ZEX_MEM_ACTION_SCOPE_FLAG_HOST;
  zex_write_to_mem_desc_t wrdesc{};
  wrdesc.writeScope = ZEX_MEM_ACTION_SCOPE_FLAG_HOST;

  // -- (2)+(3) acceptance and ordering, one shot ------------------------------
  *A = 0;
  *B = 0;
  ze_result_t r1 = zex_wait(
      reinterpret_cast<zex_command_list_handle_t>(ilist), &wdesc,
      const_cast<uint32_t*>(A), 1u, nullptr);
  q.parallel_for(sycl::range<1>(1024), [=](sycl::id<1> i) {
    dx[i] = uint32_t(i) * 7u + 1u;
  });
  q.memcpy(hx, dx, 1024 * sizeof(uint32_t));
  // Cross-engine hazard (measured: payload invisible at B without this): the
  // memcpy may ride the BCS while zex commands order only within the CCS
  // immediate list. A marker kernel joins SYCL's own cross-engine event chain
  // on the CCS; the WRITE after it is then correctly ordered.
  q.single_task([]() {});
  ze_result_t r2 = zex_write(
      reinterpret_cast<zex_command_list_handle_t>(ilist), &wrdesc,
      const_cast<uint32_t*>(B), 1ull);
  std::printf("zex on immediate list: wait=0x%x write=0x%x\n", r1, r2);
  if (r1 != ZE_RESULT_SUCCESS || r2 != ZE_RESULT_SUCCESS) return 3;

  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  if (*B != 0) {
    std::printf("ORDERING VIOLATION: B fired before the doorbell\n");
    return 3;
  }
  reinterpret_cast<volatile std::atomic<uint32_t>*>(A)->store(
      1, std::memory_order_release);
  while (*B != 1u) {
  }
  // release check: kernel effect must be host-visible once B is
  bool ok = true;
  for (int i = 0; i < 1024; ++i) ok &= (hx[i] == uint32_t(i) * 7u + 1u);
  q.wait();
  std::printf("ordering + payload visibility: %s\n",
              ok ? "CORRECT" : "BROKEN -- payload not visible at B");
  if (!ok) return 3;

  // -- (4) production-shaped latency loop -------------------------------------
  // ring -> [WAIT -> kernel -> D2H -> WRITE] -> host sees B, pre-enqueued.
  constexpr int kIters = 1000, kWarmup = 100;
  std::vector<double> rt;
  rt.reserve(kIters);
  for (int i = 0; i < kWarmup + kIters; ++i) {
    *A = 0;
    *B = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    zex_wait(reinterpret_cast<zex_command_list_handle_t>(ilist), &wdesc,
             const_cast<uint32_t*>(A), 1u, nullptr);
    q.parallel_for(sycl::range<1>(1024),
                   [=](sycl::id<1> j) { dx[j] = uint32_t(j) + uint32_t(i); });
    q.memcpy(hx, dx, 1024 * sizeof(uint32_t));
    q.single_task([]() {});  // CCS marker: orders WRITE after the BCS copy
    zex_write(reinterpret_cast<zex_command_list_handle_t>(ilist), &wrdesc,
              const_cast<uint32_t*>(B), 1ull);
    std::this_thread::sleep_for(std::chrono::microseconds(120));
    const double t0 = now_us();
    reinterpret_cast<volatile std::atomic<uint32_t>*>(A)->store(
        1, std::memory_order_release);
    while (*B != 1u) {
    }
    const double t1 = now_us();
    q.wait();
    if (i >= kWarmup) rt.push_back(t1 - t0);
  }
  std::sort(rt.begin(), rt.end());
  auto pct = [&](double p) { return rt[size_t(p * (rt.size() - 1))]; };
  std::printf("ring -> [WAIT->kernel->D2H->WRITE] -> B:  min=%5.2f  "
              "p50=%5.2f  p90=%5.2f  p99=%5.2f us   (vs 61 us poller)\n",
              rt.front(), pct(0.5), pct(0.9), pct(0.99));

  sycl::free(flags, q);
  sycl::free(dx, q);
  sycl::free(hx, q);
  return 0;
}
