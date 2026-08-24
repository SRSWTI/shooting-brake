// Kernel-isolated A/B: quixicore nvfp4_moe_split (SYCL reference) vs the
// baked gate_up + w2 epilogue launched RAW from extracted handles -- same
// device-resident inputs, no doorbells, no pinned rings, no H2D. Splits the
// harness parity failure into kernel/arg bugs vs provider list plumbing.
//
// Build:
//   icpx -fsycl -O2 -std=c++20 experiments/b70_baked_kernel_ab.cpp \
//     -I src/QuixiCore-XPU/include -L src/QuixiCore-XPU/build-sycl \
//     -lquixicore_xpu_ops -lquixicore_xpu -lze_loader \
//     -o experiments/b70_baked_kernel_ab
#include <dlfcn.h>
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>

#include "quixicore/xpu/ops.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace qx = quixicore::xpu;

int main() {
  constexpr std::size_t K = 3072, I = 1024, TOPK = 10, E = 4, M = 1;
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order{});
  std::printf("device: %s\n",
              q.get_device().get_info<sycl::info::device::name>().c_str());

  std::mt19937 rng(11);
  std::uniform_int_distribution<int> byte(0, 255);
  std::uniform_real_distribution<float> uf(-1.0f, 1.0f);
  auto up8 = [&](std::size_t n) {
    std::vector<std::uint8_t> h(n);
    for (auto &b : h) b = static_cast<std::uint8_t>(byte(rng) & 0x77);
    auto *p = sycl::malloc_device<std::uint8_t>(n, q);
    q.memcpy(p, h.data(), n).wait();
    return p;
  };
  auto *w13 = up8(E * 2 * I * (K / 2));
  auto *s13 = up8(E * 2 * I * (K / 16));
  auto *w2 = up8(E * K * (I / 2));
  auto *s2 = up8(E * K * (I / 16));
  std::vector<float> hg(E);
  for (auto &v : hg) v = std::abs(uf(rng)) * 1e-6f + 1e-7f;
  auto *w13g = sycl::malloc_device<float>(E, q);
  q.memcpy(w13g, hg.data(), E * sizeof(float)).wait();
  for (auto &v : hg) v = std::abs(uf(rng)) * 1e-6f + 1e-7f;
  auto *w2g = sycl::malloc_device<float>(E, q);
  q.memcpy(w2g, hg.data(), E * sizeof(float)).wait();
  std::vector<sycl::half> hh(K);
  for (auto &v : hh) v = static_cast<sycl::half>(uf(rng));
  auto *hidden = sycl::malloc_device<sycl::half>(K, q);
  q.memcpy(hidden, hh.data(), K * sizeof(sycl::half)).wait();
  std::vector<int> hid(TOPK);
  for (std::size_t i = 0; i < TOPK; ++i) hid[i] = static_cast<int>(i % E);
  auto *ids = sycl::malloc_device<int>(TOPK, q);
  q.memcpy(ids, hid.data(), TOPK * sizeof(int)).wait();
  std::vector<float> hw(TOPK);
  for (auto &v : hw) v = std::abs(uf(rng)) * 0.2f;
  auto *rw = sycl::malloc_device<float>(TOPK, q);
  q.memcpy(rw, hw.data(), TOPK * sizeof(float)).wait();
  auto *scratch = sycl::malloc_device<float>(TOPK * 2 * I, q);
  auto *out_f32 = sycl::malloc_device<float>(K, q);
  auto *out16 = sycl::malloc_device<sycl::half>(K, q);

  // -- reference: the shipped split path ---------------------------------------
  qx::ops::nvfp4_moe_split(q, hidden, ids, rw, w13, s13, w13g, w2, s2, w2g,
                           scratch, out_f32, M, E, TOPK, K, I, qx::DType::f16,
                           true, qx::Variant::sycl, true);
  std::vector<float> ref(K);
  q.memcpy(ref.data(), out_f32, K * sizeof(float)).wait();
  std::vector<float> scratch_ref(TOPK * 2 * I);
  q.memcpy(scratch_ref.data(), scratch, scratch_ref.size() * 4).wait();

  // -- baked kernels, raw launch ------------------------------------------------
  void *gu{}; void *w2k{};
  if (!qx::ops::nvfp4_moe_baked_handles(q, I, &gu, &w2k)) {
    std::printf("handle extraction FAILED\n");
    return 2;
  }
  q.memset(scratch, 0, TOPK * 2 * I * 4).wait();
  q.memset(out16, 0, K * sizeof(sycl::half)).wait();
  const auto zgu = reinterpret_cast<ze_kernel_handle_t>(gu);
  const auto zw2 = reinterpret_cast<ze_kernel_handle_t>(w2k);
  ze_kernel_properties_t p1{ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES};
  ze_kernel_properties_t p2{ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES};
  zeKernelGetProperties(zgu, &p1);
  zeKernelGetProperties(zw2, &p2);
  std::printf("numKernelArgs: gate_up=%u (want 11) w2=%u (want 12)\n",
              p1.numKernelArgs, p2.numKernelArgs);
  ze_result_t zr = ZE_RESULT_SUCCESS;
  auto ok = [&](ze_result_t r) {
    if (zr == ZE_RESULT_SUCCESS) zr = r;
  };
  auto sp = [&](ze_kernel_handle_t k, std::uint32_t i, const void *p) {
    ok(zeKernelSetArgumentValue(k, i, sizeof(void *), &p));
  };
  auto su = [&](ze_kernel_handle_t k, std::uint32_t i, std::uint32_t v) {
    ok(zeKernelSetArgumentValue(k, i, sizeof(std::uint32_t), &v));
  };
  const std::uint32_t gu_tiles = (I + 7) / 8;
  const std::uint32_t w2_tiles = (K + 31) / 32;  // R=4: 8 subgroups x 4 rows
  sp(zgu, 0, hidden); sp(zgu, 1, ids); sp(zgu, 2, w13); sp(zgu, 3, s13);
  sp(zgu, 4, w13g); sp(zgu, 5, scratch);
  su(zgu, 6, E); su(zgu, 7, K); su(zgu, 8, I); su(zgu, 9, TOPK);
  su(zgu, 10, gu_tiles);
  ok(zeKernelSetGroupSize(zgu, 256, 1, 1));
  sp(zw2, 0, ids); sp(zw2, 1, rw); sp(zw2, 2, w2); sp(zw2, 3, s2);
  sp(zw2, 4, w2g); sp(zw2, 5, scratch); sp(zw2, 6, out16);
  su(zw2, 7, E); su(zw2, 8, K); su(zw2, 9, TOPK); su(zw2, 10, w2_tiles);
  su(zw2, 11, 1u);
  ok(zeKernelSetGroupSize(zw2, 256, 1, 1));
  std::printf("arg set: %s (0x%x)\n", zr == ZE_RESULT_SUCCESS ? "ok" : "FAIL",
              zr);

  auto zctx =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
  auto zdev =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
  ze_command_queue_desc_t qd{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
  qd.ordinal = 0;
  qd.mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS;
  ze_command_queue_handle_t zq{};
  ze_command_list_handle_t list{};
  zeCommandQueueCreate(zctx, zdev, &qd, &zq);
  ze_command_list_desc_t ld{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
  zeCommandListCreate(zctx, zdev, &ld, &list);
  ze_group_count_t g1{static_cast<std::uint32_t>(TOPK) * gu_tiles, 1, 1};
  ze_group_count_t g2{w2_tiles, 1, 1};
  ok(zeCommandListAppendLaunchKernel(list, zgu, &g1, nullptr, 0, nullptr));
  ok(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));
  ok(zeCommandListAppendLaunchKernel(list, zw2, &g2, nullptr, 0, nullptr));
  ok(zeCommandListClose(list));
  ok(zeCommandQueueExecuteCommandLists(zq, 1, &list, nullptr));
  std::printf("execute: %s (0x%x)\n", zr == ZE_RESULT_SUCCESS ? "ok" : "FAIL",
              zr);

  std::vector<float> scratch_baked(TOPK * 2 * I);
  q.memcpy(scratch_baked.data(), scratch, scratch_baked.size() * 4).wait();
  std::vector<sycl::half> got(K);
  q.memcpy(got.data(), out16, K * sizeof(sycl::half)).wait();

  // -- stage 1: gate_up scratch parity ------------------------------------------
  float srel = 0.0f;
  for (std::size_t i = 0; i < scratch_ref.size(); ++i) {
    const float d = std::abs(scratch_ref[i] - scratch_baked[i]);
    const float m = std::max(std::abs(scratch_ref[i]), 1e-3f);
    srel = std::max(srel, d / m);
  }
  std::printf("stage1 gate_up scratch max-rel: %.3e -> %s\n", srel,
              srel < 1e-4f ? "MATCH" : "MISMATCH");
  // -- stage 2: final output parity ---------------------------------------------
  float orel = 0.0f;
  std::size_t worst = 0;
  for (std::size_t i = 0; i < K; ++i) {
    const float d = std::abs(ref[i] - static_cast<float>(got[i]));
    const float m = std::max(std::abs(ref[i]), 1e-3f);
    if (d / m > orel) { orel = d / m; worst = i; }
  }
  std::printf("stage2 output max-rel: %.3e (row %zu: ref=%f got=%f) -> %s\n",
              orel, worst, ref[worst], static_cast<float>(got[worst]),
              orel < 2e-2f ? "MATCH" : "MISMATCH");

  // -- stage 3: PROVIDER topology -- in-list copies from CUDA-pinned rings ----
  // Harness/production rings are pinned by the NVIDIA driver (torch
  // pin_memory == cudaHostAlloc). Raw L0 copies from foreign-pinned host VAs
  // are the documented "silent-copy trap" candidate (device.cpp:1807 lore).
  // Poison the device staging, then restore it via IN-LIST H2D from pinned
  // rings and D2H back out -- exactly what the baked provider lists do.
  void *cudart = dlopen("libcudart.so.13", RTLD_NOW | RTLD_GLOBAL);
  if (!cudart) cudart = dlopen("libcudart.so.12", RTLD_NOW | RTLD_GLOBAL);
  if (!cudart) cudart = dlopen("libcudart.so", RTLD_NOW | RTLD_GLOBAL);
  if (cudart == nullptr) {
    std::printf("stage3 SKIPPED: libcudart unavailable\n");
    return (srel < 1e-4f && orel < 2e-2f) ? 0 : 1;
  }
  auto *cuda_host_alloc = reinterpret_cast<int (*)(void **, std::size_t,
                                                   unsigned)>(
      dlsym(cudart, "cudaHostAlloc"));
  void *ring_h{}; void *ring_i{}; void *ring_w{}; void *ring_o{};
  if (cuda_host_alloc == nullptr ||
      cuda_host_alloc(&ring_h, K * sizeof(sycl::half), 0) != 0 ||
      cuda_host_alloc(&ring_i, TOPK * sizeof(int), 0) != 0 ||
      cuda_host_alloc(&ring_w, TOPK * sizeof(float), 0) != 0 ||
      cuda_host_alloc(&ring_o, K * sizeof(sycl::half), 0) != 0) {
    std::printf("stage3 SKIPPED: cudaHostAlloc failed\n");
    return (srel < 1e-4f && orel < 2e-2f) ? 0 : 1;
  }
  std::memcpy(ring_h, hh.data(), K * sizeof(sycl::half));
  std::memcpy(ring_i, hid.data(), TOPK * sizeof(int));
  std::memcpy(ring_w, hw.data(), TOPK * sizeof(float));
  std::memset(ring_o, 0, K * sizeof(sycl::half));
  // Poison the staging so stale-correct data cannot mask a broken copy.
  q.memset(const_cast<sycl::half *>(hidden), 0xAA, K * sizeof(sycl::half));
  q.memset(const_cast<int *>(ids), 0xAA, TOPK * sizeof(int));
  q.memset(const_cast<float *>(rw), 0xAA, TOPK * sizeof(float));
  q.memset(scratch, 0, TOPK * 2 * I * 4);
  q.memset(out16, 0, K * sizeof(sycl::half)).wait();
  ze_command_list_handle_t list3{};
  zeCommandListCreate(zctx, zdev, &ld, &list3);
  ok(zeCommandListAppendMemoryCopy(list3, const_cast<sycl::half *>(hidden),
                                   ring_h, K * sizeof(sycl::half), nullptr, 0,
                                   nullptr));
  ok(zeCommandListAppendMemoryCopy(list3, const_cast<int *>(ids), ring_i,
                                   TOPK * sizeof(int), nullptr, 0, nullptr));
  ok(zeCommandListAppendMemoryCopy(list3, const_cast<float *>(rw), ring_w,
                                   TOPK * sizeof(float), nullptr, 0, nullptr));
  ok(zeCommandListAppendBarrier(list3, nullptr, 0, nullptr));
  ok(zeCommandListAppendLaunchKernel(list3, zgu, &g1, nullptr, 0, nullptr));
  ok(zeCommandListAppendBarrier(list3, nullptr, 0, nullptr));
  ok(zeCommandListAppendLaunchKernel(list3, zw2, &g2, nullptr, 0, nullptr));
  ok(zeCommandListAppendBarrier(list3, nullptr, 0, nullptr));
  ok(zeCommandListAppendMemoryCopy(list3, ring_o, out16,
                                   K * sizeof(sycl::half), nullptr, 0,
                                   nullptr));
  ok(zeCommandListClose(list3));
  ok(zeCommandQueueExecuteCommandLists(zq, 1, &list3, nullptr));
  std::printf("stage3 build+execute: %s (0x%x)\n",
              zr == ZE_RESULT_SUCCESS ? "ok" : "FAIL", zr);
  const auto *ro = static_cast<const sycl::half *>(ring_o);
  float prel = 0.0f;
  std::size_t pworst = 0;
  for (std::size_t i = 0; i < K; ++i) {
    const float d = std::abs(ref[i] - static_cast<float>(ro[i]));
    const float m = std::max(std::abs(ref[i]), 1e-3f);
    if (d / m > prel) { prel = d / m; pworst = i; }
  }
  std::printf("stage3 pinned-ring topology max-rel: %.3e (row %zu: ref=%f "
              "got=%f) -> %s\n",
              prel, pworst, ref[pworst], static_cast<float>(ro[pworst]),
              prel < 2e-2f ? "MATCH -- plumbing exonerated"
                           : "MISMATCH -- pinned-ring copies are the bug");
  return (srel < 1e-4f && orel < 2e-2f && prel < 2e-2f) ? 0 : 1;
}
