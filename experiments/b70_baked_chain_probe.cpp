// Baked-chain probe: can a PRE-RECORDED Level-Zero command list carry the
// whole decode chain -- WAIT(doorbell) -> gate_up kernel -> fused w2 epilogue
// (rowmajor, no atomics, direct fp16) -> D2H -> WRITE(completion) -- and
// replay it at hardware speed?
//
// This is the shape the 8 us doorbell probe actually measured, and the shape
// live appends could not reach (80-95 us per bracket, kill-bench §22). The
// three blockers it settles, in order:
//   1. HANDLES: named SYCL functor kernels -> kernel_bundle -> get_native
//      yields ze_kernel_handle_t. (yes/no)
//   2. ARG ABI: functor fields become kernel args in declaration order;
//      zeKernelSetArgumentValue by field index reproduces the SYCL launch
//      BIT-EXACTLY. (identity gate; also counts args empirically first)
//   3. SPEED: closed-list replay of the full chain, host ring -> completion,
//      vs the classic poller's 120 us/layer reference.
//
// Kernels are 1D on purpose: SYCL<->L0 dimension mapping is a known trap;
// dim0==X is the only mapping nobody argues about. The math is quixicore's
// nvfp4 row-dot verbatim (E2M1 pairs, E4M3 block scales, +4194304.0f global).
// The w2 epilogue here is ALSO the fusion prototype: one work-group owns
// output rows, loops the token's routes with register accumulators, writes
// each row once as fp16 -- no atomics, no zero-fill, no narrow pass.
//
// Build (same toolchain as the provider):
//   source /opt/intel/oneapi/setvars.sh
//   icpx -fsycl -O2 -std=c++20 experiments/b70_baked_chain_probe.cpp \
//     -I vendor/compute-runtime/level_zero/include -o experiments/b70_baked_chain_probe \
//     -lze_loader
// Run:  ./experiments/b70_baked_chain_probe 0000:15:00.0

#include <level_zero/ze_api.h>
#include <level_zero/driver_experimental/zex_cmdlist.h>
#include <level_zero/driver_experimental/zex_common.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int kSG = 32;
constexpr int kSubgroups = 8;
constexpr int kWG = kSG * kSubgroups;
constexpr std::size_t kK = 3072;   // hidden
constexpr std::size_t kI = 1024;   // intermediate
constexpr std::size_t kTopK = 10;
constexpr std::size_t kE = 16;     // synthetic expert count
using WVec = sycl::vec<std::uint32_t, 4>;

inline sycl::vec<float, 2> decode_e2m1_pair(std::uint32_t word) {
  const std::uint32_t nibbles = word & 0x000f000fu;
  const std::uint32_t halves =
      ((nibbles & 0x00080008u) << 12) | ((nibbles & 0x00070007u) << 9);
  return sycl::bit_cast<sycl::vec<sycl::half, 2>>(halves)
      .template convert<float>();
}

inline float decode_e4m3_raw(std::uint8_t v) {
  const auto bits =
      static_cast<std::uint16_t>(((v & 0x80u) << 8) | ((v & 0x7fu) << 7));
  return static_cast<float>(sycl::bit_cast<sycl::half>(bits));
}

inline float row_dot(const sycl::sub_group &sg, const std::uint8_t *w,
                     const std::uint8_t *bs, float gs, const float *x,
                     std::size_t K) {
  const WVec *packed = reinterpret_cast<const WVec *>(w);
  const int lane = static_cast<int>(sg.get_local_linear_id());
  float partial = 0.0f;
  for (std::size_t c = lane; c < K / 32; c += kSG) {
    const WVec vals = packed[c];
    const float s0 = decode_e4m3_raw(bs[2 * c]) * gs;
    const float s1 = decode_e4m3_raw(bs[2 * c + 1]) * gs;
    const float *xv = x + c * 32;
    float words[4];
#pragma unroll
    for (int wi = 0; wi < 4; ++wi) {
      const std::uint32_t word = vals[wi];
      float lo = 0.0f, hi = 0.0f;
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        const sycl::vec<float, 2> d = decode_e2m1_pair(word >> (4 * p));
        lo += d[0] * xv[wi * 8 + p];
        hi += d[1] * xv[wi * 8 + p + 4];
      }
      words[wi] = lo + hi;
    }
    partial += (words[0] + words[1]) * s0 + (words[2] + words[3]) * s1;
  }
  return sycl::reduce_over_group(sg, partial, sycl::plus<float>());
}

inline float silu(float v) { return v / (1.0f + sycl::exp(-v)); }

// ---- functor kernels: fields ARE the kernel args, in declaration order ----

struct GateUpBaked {
  const float *hidden;          // arg 0
  const int *ids;               // arg 1
  const std::uint8_t *w13;      // arg 2
  const std::uint8_t *s13;      // arg 3
  const float *w13_global;      // arg 4
  float *scratch;               // arg 5
  // 1D: group = pair * row_tiles + tile; row = tile*kSubgroups + subgroup_id.
  [[sycl::reqd_sub_group_size(kSG)]] void
  operator()(sycl::nd_item<1> item) const {
    constexpr std::size_t row_tiles = kI / kSubgroups;
    const std::size_t group = item.get_group(0);
    const std::size_t pair = group / row_tiles;
    const std::size_t tile = group % row_tiles;
    const int expert = ids[pair];
    if (expert < 0 || static_cast<std::size_t>(expert) >= kE) return;
    const std::size_t token = pair / kTopK;
    const sycl::sub_group sg = item.get_sub_group();
    const std::size_t row = tile * kSubgroups + sg.get_group_linear_id();
    if (row >= kI) return;
    const std::size_t e = static_cast<std::size_t>(expert);
    const float gs = w13_global[e] * 4194304.0f;
    const std::uint8_t *ew = w13 + e * (2 * kI) * (kK / 2);
    const std::uint8_t *es = s13 + e * (2 * kI) * (kK / 16);
    const float gate = row_dot(sg, ew + row * (kK / 2), es + row * (kK / 16),
                               gs, hidden + token * kK, kK);
    const float up =
        row_dot(sg, ew + (row + kI) * (kK / 2), es + (row + kI) * (kK / 16),
                gs, hidden + token * kK, kK);
    if (sg.get_local_linear_id() == 0) {
      scratch[pair * 2 * kI + row] = gate;
      scratch[pair * 2 * kI + row + kI] = up;
    }
  }
};

// Fused w2 epilogue: rowmajor, register accumulation across the token's
// routes, single fp16 write per output element. No atomics, no zero-fill,
// no narrow pass. Also the fusion prototype for the split path.
struct W2EpilogueBaked {
  const int *ids;               // arg 0
  const float *weights;         // arg 1
  const std::uint8_t *w2;       // arg 2
  const std::uint8_t *s2;       // arg 3
  const float *w2_global;       // arg 4
  const float *scratch;         // arg 5
  sycl::half *out16;            // arg 6
  // 1D over row tiles (M=1): group = row tile; each subgroup owns one row.
  [[sycl::reqd_sub_group_size(kSG)]] void
  operator()(sycl::nd_item<1> item) const {
    const sycl::sub_group sg = item.get_sub_group();
    const std::size_t row =
        item.get_group(0) * kSubgroups + sg.get_group_linear_id();
    if (row >= kK) return;
    auto &activated = *sycl::ext::oneapi::group_local_memory<float[kI]>(
        item.get_group());
    const int thread = static_cast<int>(item.get_local_linear_id());
    float acc = 0.0f;
    for (std::size_t pair = 0; pair < kTopK; ++pair) {
      const int expert = ids[pair];
      sycl::group_barrier(item.get_group());
      if (expert >= 0 && static_cast<std::size_t>(expert) < kE) {
        const float *gu = scratch + pair * 2 * kI;
        for (std::size_t i = thread; i < kI; i += kWG)
          activated[i] = silu(gu[i]) * gu[i + kI];
      }
      sycl::group_barrier(item.get_group());
      if (expert < 0 || static_cast<std::size_t>(expert) >= kE) continue;
      const std::size_t e = static_cast<std::size_t>(expert);
      const float gs = w2_global[e] * 4194304.0f;
      const float v = row_dot(
          sg, w2 + e * kK * (kI / 2) + row * (kI / 2),
          s2 + e * kK * (kI / 16) + row * (kI / 16), gs, &activated[0], kI);
      acc += weights[pair] * v;
    }
    if (sg.get_local_linear_id() == 0)
      out16[row] = static_cast<sycl::half>(acc);
  }
};

double now_us() {
  return std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

#define ZE_CHECK(call)                                                         \
  do {                                                                         \
    ze_result_t r_ = (call);                                                   \
    if (r_ != ZE_RESULT_SUCCESS) {                                             \
      std::printf("FAIL %s -> 0x%x\n", #call, r_);                             \
      return 3;                                                                \
    }                                                                          \
  } while (0)

} // namespace

int main(int argc, char **argv) {
  const std::string bdf = argc > 1 ? argv[1] : "0000:15:00.0";
  sycl::device dev;
  bool found = false;
  for (const auto &d : sycl::device::get_devices(sycl::info::device_type::gpu)) {
    if (d.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
    auto zd = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(d);
    ze_pci_ext_properties_t pci{ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES};
    if (zeDevicePciGetPropertiesExt(zd, &pci) != ZE_RESULT_SUCCESS) continue;
    char buf[32];
    std::snprintf(buf, sizeof buf, "%04x:%02x:%02x.%x", pci.address.domain,
                  pci.address.bus, pci.address.device, pci.address.function);
    if (bdf == buf) { dev = d; found = true; break; }
  }
  if (!found) { std::printf("device %s not found\n", bdf.c_str()); return 2; }
  sycl::queue q(dev, sycl::property::queue::in_order{});

  // -- synthetic model data, device USM ---------------------------------------
  std::mt19937 rng(7);
  std::uniform_int_distribution<int> byte(0, 255);
  std::uniform_real_distribution<float> uf(-1.0f, 1.0f);
  const std::size_t w13_bytes = kE * 2 * kI * (kK / 2);
  const std::size_t s13_bytes = kE * 2 * kI * (kK / 16);
  const std::size_t w2_bytes = kE * kK * (kI / 2);
  const std::size_t s2_bytes = kE * kK * (kI / 16);
  auto up8 = [&](std::size_t n) {
    std::vector<std::uint8_t> h(n);
    for (auto &b : h) b = static_cast<std::uint8_t>(byte(rng) & 0x77);
    auto *p = sycl::malloc_device<std::uint8_t>(n, q);
    q.memcpy(p, h.data(), n).wait();
    return p;
  };
  auto *w13 = up8(w13_bytes); auto *s13 = up8(s13_bytes);
  auto *w2 = up8(w2_bytes);   auto *s2 = up8(s2_bytes);
  std::vector<float> hg(kE);
  for (auto &v : hg) v = std::abs(uf(rng)) * 1e-6f + 1e-7f;
  auto *w13g = sycl::malloc_device<float>(kE, q);
  q.memcpy(w13g, hg.data(), kE * sizeof(float)).wait();
  for (auto &v : hg) v = std::abs(uf(rng)) * 1e-6f + 1e-7f;
  auto *w2g = sycl::malloc_device<float>(kE, q);
  q.memcpy(w2g, hg.data(), kE * sizeof(float)).wait();
  std::vector<float> hh(kK);
  for (auto &v : hh) v = uf(rng);
  auto *hidden = sycl::malloc_device<float>(kK, q);
  q.memcpy(hidden, hh.data(), kK * sizeof(float)).wait();
  std::vector<int> hid(kTopK);
  for (std::size_t i = 0; i < kTopK; ++i) hid[i] = static_cast<int>((i * 3) % kE);
  auto *ids = sycl::malloc_device<int>(kTopK, q);
  q.memcpy(ids, hid.data(), kTopK * sizeof(int)).wait();
  std::vector<float> hw(kTopK, 0.1f);
  auto *rw = sycl::malloc_device<float>(kTopK, q);
  q.memcpy(rw, hw.data(), kTopK * sizeof(float)).wait();
  auto *scratch = sycl::malloc_device<float>(kTopK * 2 * kI, q);
  auto *out_ref = sycl::malloc_device<sycl::half>(kK, q);
  auto *out_baked = sycl::malloc_device<sycl::half>(kK, q);

  const std::size_t gu_groups = kTopK * (kI / kSubgroups);
  const std::size_t w2_groups = kK / kSubgroups;
  const GateUpBaked gu_f{hidden, ids, w13, s13, w13g, scratch};
  const W2EpilogueBaked w2_ref{ids, rw, w2, s2, w2g, scratch, out_ref};
  const W2EpilogueBaked w2_bak{ids, rw, w2, s2, w2g, scratch, out_baked};

  // -- reference: plain SYCL submission ---------------------------------------
  q.parallel_for(sycl::nd_range<1>(gu_groups * kWG, kWG), gu_f);
  q.parallel_for(sycl::nd_range<1>(w2_groups * kWG, kWG), w2_ref);
  q.wait();
  std::vector<sycl::half> ref(kK);
  q.memcpy(ref.data(), out_ref, kK * sizeof(sycl::half)).wait();

  // -- blocker 1: native kernel handles ---------------------------------------
  auto ctx = q.get_context();
  auto kid_gu = sycl::get_kernel_id<GateUpBaked>();
  auto kid_w2 = sycl::get_kernel_id<W2EpilogueBaked>();
  auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      ctx, {kid_gu, kid_w2});
  auto zk_gu = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
      bundle.get_kernel(kid_gu));
  auto zk_w2 = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
      bundle.get_kernel(kid_w2));
  std::printf("handles: gate_up=%p w2=%p\n", (void *)zk_gu, (void *)zk_w2);
  if (zk_gu == nullptr || zk_w2 == nullptr) return 3;

  // -- blocker 2: arg ABI. Count args empirically, then set by field order. ---
  auto count_args = [](ze_kernel_handle_t k) {
    ze_kernel_properties_t props{ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES};
    if (zeKernelGetProperties(k, &props) == ZE_RESULT_SUCCESS)
      return static_cast<int>(props.numKernelArgs);
    return -1;
  };
  std::printf("arg counts: gate_up=%d (fields 6) w2=%d (fields 7)\n",
              count_args(zk_gu), count_args(zk_w2));
  auto set_ptr = [](ze_kernel_handle_t k, std::uint32_t i, const void *p) {
    return zeKernelSetArgumentValue(k, i, sizeof(void *), &p);
  };
  ZE_CHECK(set_ptr(zk_gu, 0, hidden)); ZE_CHECK(set_ptr(zk_gu, 1, ids));
  ZE_CHECK(set_ptr(zk_gu, 2, w13));    ZE_CHECK(set_ptr(zk_gu, 3, s13));
  ZE_CHECK(set_ptr(zk_gu, 4, w13g));   ZE_CHECK(set_ptr(zk_gu, 5, scratch));
  ZE_CHECK(zeKernelSetGroupSize(zk_gu, kWG, 1, 1));
  ZE_CHECK(set_ptr(zk_w2, 0, ids));    ZE_CHECK(set_ptr(zk_w2, 1, rw));
  ZE_CHECK(set_ptr(zk_w2, 2, w2));     ZE_CHECK(set_ptr(zk_w2, 3, s2));
  ZE_CHECK(set_ptr(zk_w2, 4, w2g));    ZE_CHECK(set_ptr(zk_w2, 5, scratch));
  ZE_CHECK(set_ptr(zk_w2, 6, out_baked));
  ZE_CHECK(zeKernelSetGroupSize(zk_w2, kWG, 1, 1));

  // -- bake the chain ----------------------------------------------------------
  auto zctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
  auto zdev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);
  ze_driver_handle_t zdrv =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
          ctx.get_platform());
  using WaitFn = ze_result_t (*)(zex_command_list_handle_t,
                                 zex_wait_on_mem_desc_t *, void *,
                                 std::uint32_t, zex_event_handle_t);
  using WriteFn = ze_result_t (*)(zex_command_list_handle_t,
                                  zex_write_to_mem_desc_t *, void *,
                                  std::uint64_t);
  WaitFn zex_wait{}; WriteFn zex_write{};
  ZE_CHECK(zeDriverGetExtensionFunctionAddress(
      zdrv, "zexCommandListAppendWaitOnMemory",
      reinterpret_cast<void **>(&zex_wait)));
  ZE_CHECK(zeDriverGetExtensionFunctionAddress(
      zdrv, "zexCommandListAppendWriteToMemory",
      reinterpret_cast<void **>(&zex_write)));

  volatile std::uint32_t *sig{}; volatile std::uint32_t *comp{};
  {
    ze_host_mem_alloc_desc_t hd{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
    void *p{};
    ZE_CHECK(zeMemAllocHost(zctx, &hd, 256, 64, &p));
    std::memset(p, 0, 256);
    sig = static_cast<volatile std::uint32_t *>(p);
    comp = sig + 32;
  }
  sycl::half *host_out{};
  {
    ze_host_mem_alloc_desc_t hd{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
    void *p{};
    ZE_CHECK(zeMemAllocHost(zctx, &hd, kK * sizeof(sycl::half), 64, &p));
    host_out = static_cast<sycl::half *>(p);
  }

  ze_command_queue_desc_t qd{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
  qd.ordinal = 0;
  qd.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  ze_command_queue_handle_t zq{};
  ZE_CHECK(zeCommandQueueCreate(zctx, zdev, &qd, &zq));
  ze_command_list_desc_t ld{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
  ld.commandQueueGroupOrdinal = 0;
  ze_command_list_handle_t list{};
  ZE_CHECK(zeCommandListCreate(zctx, zdev, &ld, &list));

  zex_wait_on_mem_desc_t wd{};
  wd.actionFlag = ZEX_WAIT_ON_MEMORY_FLAG_EQUAL;
  zex_write_to_mem_desc_t wr{};
  wr.writeScope = ZEX_MEM_ACTION_SCOPE_FLAG_HOST;
  ze_group_count_t gc_gu{static_cast<std::uint32_t>(gu_groups), 1, 1};
  ze_group_count_t gc_w2{static_cast<std::uint32_t>(w2_groups), 1, 1};
  auto *xl = reinterpret_cast<zex_command_list_handle_t>(list);
  ZE_CHECK(zex_wait(xl, &wd, const_cast<std::uint32_t *>(sig), 1u, nullptr));
  ZE_CHECK(zex_write(xl, &wr, const_cast<std::uint32_t *>(sig), 0ull));
  ZE_CHECK(zeCommandListAppendLaunchKernel(list, zk_gu, &gc_gu, nullptr, 0,
                                           nullptr));
  ZE_CHECK(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));
  ZE_CHECK(zeCommandListAppendLaunchKernel(list, zk_w2, &gc_w2, nullptr, 0,
                                           nullptr));
  ZE_CHECK(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));
  ZE_CHECK(zeCommandListAppendMemoryCopy(list, host_out, out_baked,
                                         kK * sizeof(sycl::half), nullptr, 0,
                                         nullptr));
  ZE_CHECK(zex_write(xl, &wr, const_cast<std::uint32_t *>(comp), 1ull));
  ZE_CHECK(zeCommandListClose(list));
  std::printf("baked list closed: WAIT -> gate_up -> w2_epilogue -> D2H -> "
              "WRITE\n");

  // -- replay loop: identity + timing ------------------------------------------
  constexpr int kWarm = 20, kIters = 300;
  std::vector<double> rt;
  rt.reserve(kIters);
  bool bit_exact = true;
  for (int i = 0; i < kWarm + kIters; ++i) {
    *comp = 0;
    q.memset(out_baked, 0, kK * sizeof(sycl::half)).wait();
    ZE_CHECK(zeCommandQueueExecuteCommandLists(zq, 1, &list, nullptr));
    const double t0 = now_us();
    *sig = 1u;
    while (*comp != 1u) {
      if (now_us() - t0 > 2e6) { std::printf("TIMEOUT iter %d\n", i); return 4; }
    }
    const double t1 = now_us();
    if (i >= kWarm) rt.push_back(t1 - t0);
    if (i == kWarm) {
      for (std::size_t j = 0; j < kK; ++j) {
        if (std::memcmp(&host_out[j], &ref[j], sizeof(sycl::half)) != 0) {
          bit_exact = false;
          std::printf("MISMATCH row %zu: baked=%f ref=%f\n", j,
                      static_cast<float>(host_out[j]),
                      static_cast<float>(ref[j]));
          break;
        }
      }
      std::printf("identity gate vs SYCL launch: %s\n",
                  bit_exact ? "BIT-EXACT" : "FAILED");
      if (!bit_exact) return 5;
    }
  }
  std::sort(rt.begin(), rt.end());
  std::printf("baked chain replay (ring -> gate_up+w2+D2H -> completion):\n"
              "  n=%d p50=%.1f us p90=%.1f us min=%.1f max=%.1f\n",
              kIters, rt[rt.size() / 2], rt[rt.size() * 9 / 10], rt.front(),
              rt.back());

  // -- reference timing: same work, SYCL submit + host wait per iteration ------
  std::vector<double> st;
  st.reserve(kIters);
  for (int i = 0; i < kWarm + kIters; ++i) {
    const double t0 = now_us();
    q.parallel_for(sycl::nd_range<1>(gu_groups * kWG, kWG), gu_f);
    q.parallel_for(sycl::nd_range<1>(w2_groups * kWG, kWG), w2_ref);
    q.memcpy(host_out, out_ref, kK * sizeof(sycl::half));
    q.wait();
    const double t1 = now_us();
    if (i >= kWarm) st.push_back(t1 - t0);
  }
  std::sort(st.begin(), st.end());
  std::printf("SYCL submit+wait reference (same kernels, host in loop):\n"
              "  n=%d p50=%.1f us p90=%.1f us min=%.1f\n", kIters,
              st[st.size() / 2], st[st.size() * 9 / 10], st.front());
  std::printf("VERDICT: baked=%.1f us vs live-SYCL=%.1f us vs classic-poller "
              "~120 us/layer\n", rt[rt.size() / 2], st[st.size() / 2]);
  zeCommandListDestroy(list);
  zeCommandQueueDestroy(zq);
  return 0;
}
