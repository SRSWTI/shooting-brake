// Full grouped MoE layer on REAL bank bytes: the number that matters.
//
// Everything before this measured one GEMM in isolation. A serving layer is
// more than that: route sorting, an activation gather, w13, SwiGLU, w2, then a
// weighted scatter back. This runs all of it at r15 geometry with weights read
// out of the actual 54 GiB bank, so the per-token figure is directly
// comparable to the 1,705 us/token measured in production.
//
// Deliberately NOT in the provider yet. Proving the op standalone means a
// failure here is a failure of the op, not of 1,638 lines of provider it would
// otherwise be tangled with. Integration is a smaller, safer step afterwards.
//
// Layout arithmetic that makes this possible without repacking: the kernel
// addresses expert e at `e * gemm_n * gemm_k / 2` and its scales at
// `e * gemm_n * gemm_k / group_size`. For w13 that is e*3,145,728 and
// e*393,216; for w2, e*1,572,864. Those are byte-for-byte our bank's
// g_w13_bytes / g_s13_bytes / g_w2_bytes strides, so the bank feeds the kernel
// directly.
//
// Correctness note: this measures TIME. The arithmetic was gated separately by
// xe2_nvfp4_verify (E4M3 decode, nibble order, layout char, alpha semantics).
// The SwiGLU and scatter here are straightforward but unverified against a
// reference; they are timed, not trusted.

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <random>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "cutlass/kernel_hardware_info.h"
#include "xe_2/gemm_xe2_policy.hpp"
#include "xe_2/grouped_gemm_xe2.hpp"

using namespace cute;

namespace {

template <typename, typename, typename, char, char, class, MoE::A_DTYPE,
          MoE::B_DTYPE>
class MoEName;

template <char layoutA, char layoutB, class policy, MoE::A_DTYPE ADT,
          MoE::B_DTYPE BDT, typename ElementA, typename ElementB,
          typename ElementS, typename ElementBI, typename ElementD>
void launch(sycl::queue& q, const ElementA* act, const ElementB* wgt,
            const ElementS* scl, const ElementBI* bias, ElementD* out,
            int gemm_n, int gemm_k, const int* rows_per_expert, int num_experts,
            int group_size, int32_t* atomic_buffer) {
  using ElementA_non_CV = cutlass::platform::remove_cv_t<ElementA>;
  auto op = XE_DPAS_TT<8, float, ElementA_non_CV>{};
  using WGTile = typename policy::WGTile;
  using SGLayout = typename policy::SGLayout;
  using MMA = typename TiledMMAHelper<MMA_Atom<decltype(op)>, Layout<WGTile>,
                                      SGLayout>::TiledMMA;
  auto mma = MMA{};
  int sm = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  auto tpw = size(mma);
  sycl::range<3> local(1, 1, tpw);
  sycl::range<3> global(1, sm * 512 / tpw, 1);
  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  syclex::properties props{syclex::sub_group_size<16>, intelex::grf_size<256>};
  using CopyA = typename policy::GmemTiledCopyA;
  using CopyB = typename policy::GmemTiledCopyB;
  using CopyD = typename policy::GmemTiledCopyD;
  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<int32_t, 1> lm(sycl::range<1>(1), cgh);
    cgh.parallel_for<MoEName<ElementA, ElementB, ElementD, layoutA, layoutB,
                             policy, ADT, BDT>>(
        sycl::nd_range<3>{global * local, local}, props, [=](auto) {
          MoE::MoEGEMM<ADT, BDT, CopyA, CopyB, CopyD, layoutA, layoutB, 'R'>(
              act, wgt, scl, bias, out, mma, rows_per_expert, num_experts,
              group_size, gemm_n, gemm_k, atomic_buffer, lm);
        });
  });
}

struct Bank {
  int layers, experts, hidden, inter;
  std::uint64_t w13_b, s13_b, w2_b, s2_b, record, hdr;
  const std::uint8_t* base;
  std::size_t bytes;
};

Bank map_bank(const char* path) {
  int fd = ::open(path, O_RDONLY);
  if (fd < 0) { printf("cannot open %s\n", path); std::exit(1); }
  struct stat st{};
  ::fstat(fd, &st);
  auto* p = static_cast<const std::uint8_t*>(
      ::mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0));
  if (p == MAP_FAILED) { printf("mmap failed\n"); std::exit(1); }
  Bank b{};
  b.base = p;
  b.bytes = st.st_size;
  b.hdr = 60;  // 8s + 5*u32 + 4*u64 = 8+20+32 = 60
  std::memcpy(&b.layers, p + 8, 4);
  std::memcpy(&b.experts, p + 12, 4);
  std::memcpy(&b.hidden, p + 16, 4);
  std::memcpy(&b.inter, p + 20, 4);
  std::memcpy(&b.w13_b, p + 28, 8);
  std::memcpy(&b.s13_b, p + 36, 8);
  std::memcpy(&b.w2_b, p + 44, 8);
  std::memcpy(&b.s2_b, p + 52, 8);
  b.record = b.w13_b + b.s13_b + b.w2_b + b.s2_b + 8;
  return b;
}

int arg_int(int c, char** v, const char* f, int d) {
  for (int i = 1; i + 1 < c; ++i)
    if (!std::strcmp(v[i], f)) return std::atoi(v[i + 1]);
  return d;
}
const char* arg_str(int c, char** v, const char* f, const char* d) {
  for (int i = 1; i + 1 < c; ++i)
    if (!std::strcmp(v[i], f)) return v[i + 1];
  return d;
}

}  // namespace

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char* path = arg_str(argc, argv, "--bank",
                             "src/phase1/expert_bank_jota_118b_r15.bin");
  const int E = arg_int(argc, argv, "--experts", 85);   // resident per card
  const int tokens = arg_int(argc, argv, "--tokens", 512);
  const int top_k = arg_int(argc, argv, "--top-k", 10);
  const int cards = arg_int(argc, argv, "--cards", 2);
  const int iters = arg_int(argc, argv, "--iters", 10);
  const int layer = arg_int(argc, argv, "--layer", 0);

  Bank bk = map_bank(path);
  const int H = bk.hidden, I = bk.inter, gs = 16;
  const int M = tokens * top_k / cards;   // routes landing on this card

  printf("bank    : %s\n", path);
  printf("geometry: layers=%d experts=%d H=%d I=%d record=%lu\n",
         bk.layers, bk.experts, H, I, (unsigned long)bk.record);
  printf("cell    : %d resident experts, M=%d routes (%d tok x top%d / %d cards)\n",
         E, M, tokens, top_k, cards);
  printf("          %.1f rows/expert\n\n", double(M) / E);

  sycl::queue q{sycl::gpu_selector_v};
  printf("device  : %s\n\n", q.get_device().get_info<sycl::info::device::name>().c_str());

  // ---- upload one layer's resident experts, exactly as the provider does ----
  auto* w13 = sycl::malloc_device<std::uint8_t>(size_t(E) * bk.w13_b, q);
  auto* s13 = sycl::malloc_device<std::uint8_t>(size_t(E) * bk.s13_b, q);
  auto* w2  = sycl::malloc_device<std::uint8_t>(size_t(E) * bk.w2_b, q);
  auto* s2  = sycl::malloc_device<std::uint8_t>(size_t(E) * bk.s2_b, q);
  std::vector<float> a13(E), a2(E);
  {
    auto t0 = std::chrono::steady_clock::now();
    for (int e = 0; e < E; ++e) {
      const std::uint8_t* r =
          bk.base + bk.hdr + (size_t(layer) * bk.experts + e) * bk.record;
      q.memcpy(w13 + size_t(e) * bk.w13_b, r, bk.w13_b);
      q.memcpy(s13 + size_t(e) * bk.s13_b, r + bk.w13_b, bk.s13_b);
      q.memcpy(w2 + size_t(e) * bk.w2_b, r + bk.w13_b + bk.s13_b, bk.w2_b);
      q.memcpy(s2 + size_t(e) * bk.s2_b,
               r + bk.w13_b + bk.s13_b + bk.w2_b, bk.s2_b);
      std::memcpy(&a13[e], r + bk.record - 8, 4);
      std::memcpy(&a2[e], r + bk.record - 4, 4);
    }
    q.wait();
    double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
    double gib = double(E) * bk.record / (1u << 30);
    printf("uploaded %.2f GiB in %.0f ms (%.1f GB/s)\n", gib, ms,
           gib * 1.073741824 / (ms * 1e-3));
  }
  printf("alpha13[0]=%.6g alpha2[0]=%.6g\n\n", a13[0], a2[0]);

  // ---- buffers ----
  auto* act_src = sycl::malloc_device<cutlass::half_t>(size_t(tokens) * H, q);
  auto* act   = sycl::malloc_device<cutlass::half_t>(size_t(M) * H, q);
  auto* mid   = sycl::malloc_device<cutlass::half_t>(size_t(M) * 2 * I, q);
  auto* gated = sycl::malloc_device<cutlass::half_t>(size_t(M) * I, q);
  auto* outr  = sycl::malloc_device<cutlass::half_t>(size_t(M) * H, q);
  auto* final_out = sycl::malloc_device<float>(size_t(tokens) * H, q);
  auto* bias13 = sycl::malloc_device<cutlass::half_t>(size_t(E) * 2 * I, q);
  auto* bias2  = sycl::malloc_device<cutlass::half_t>(size_t(E) * H, q);
  auto* rows   = sycl::malloc_device<int>(E, q);
  auto* atom   = sycl::malloc_device<int32_t>(1, q);
  auto* alpha13 = sycl::malloc_device<float>(E, q);
  auto* alpha2  = sycl::malloc_device<float>(E, q);
  // routing
  auto* ids   = sycl::malloc_device<int>(size_t(tokens) * top_k, q);
  auto* rw    = sycl::malloc_device<float>(size_t(tokens) * top_k, q);
  auto* hist  = sycl::malloc_device<int>(E + 1, q);
  auto* offs  = sycl::malloc_device<int>(E + 1, q);
  auto* cursor = sycl::malloc_device<int>(E, q);
  auto* slot_row = sycl::malloc_device<int>(M, q);   // slot -> source token
  auto* slot_w   = sycl::malloc_device<float>(M, q); // slot -> route weight
  auto* slot_exp = sycl::malloc_device<int>(M, q);   // slot -> expert

  q.memset(bias13, 0, size_t(E) * 2 * I * sizeof(cutlass::half_t)).wait();
  q.memset(bias2, 0, size_t(E) * H * sizeof(cutlass::half_t)).wait();
  q.memcpy(alpha13, a13.data(), E * sizeof(float)).wait();
  q.memcpy(alpha2, a2.data(), E * sizeof(float)).wait();

  // Routing that mirrors what Bench 17 measured on natural prose: near-uniform
  // with max/mean about 1.63, NOT perfectly even. An even split flatters the
  // work queue and would overstate this.
  std::mt19937 rng(7);
  std::vector<int> h_ids(size_t(tokens) * top_k);
  std::vector<float> h_rw(size_t(tokens) * top_k);
  {
    std::vector<double> wgt(E);
    for (int e = 0; e < E; ++e) wgt[e] = 1.0 + 0.63 * (double(rng() % 1000) / 1000.0);
    std::discrete_distribution<int> pick(wgt.begin(), wgt.end());
    for (int t = 0; t < tokens; ++t)
      for (int k = 0; k < top_k; ++k) {
        h_ids[size_t(t) * top_k + k] = pick(rng);
        h_rw[size_t(t) * top_k + k] = 1.0f / top_k;
      }
  }
  q.memcpy(ids, h_ids.data(), h_ids.size() * sizeof(int)).wait();
  q.memcpy(rw, h_rw.data(), h_rw.size() * sizeof(float)).wait();

  std::vector<cutlass::half_t> h_act(size_t(tokens) * H);
  for (auto& v : h_act) v = cutlass::half_t(0.01f * float(int(rng() % 200) - 100));
  q.memcpy(act_src, h_act.data(), h_act.size() * sizeof(cutlass::half_t)).wait();

  const int total_routes = tokens * top_k;
  const int route_cap = M;

  auto one_layer = [&]() {
    // 1. histogram over the routes this card owns
    q.memset(hist, 0, (E + 1) * sizeof(int));
    q.submit([&](sycl::handler& h) {
      h.parallel_for(sycl::range<1>(route_cap), [=](sycl::id<1> i) {
        int e = ids[i];
        if (e >= 0 && e < E)
          sycl::atomic_ref<int, sycl::memory_order::relaxed,
                           sycl::memory_scope::device>(hist[e])++;
      });
    });
    // 2. exclusive prefix sum (E is 85; one thread is cheaper than a scan)
    q.submit([&](sycl::handler& h) {
      h.single_task([=]() {
        int acc = 0;
        for (int e = 0; e < E; ++e) { offs[e] = acc; acc += hist[e]; cursor[e] = offs[e]; }
        offs[E] = acc;
      });
    });
    // 3. scatter routes into expert-major slots, and gather activations
    q.submit([&](sycl::handler& h) {
      h.parallel_for(sycl::range<1>(route_cap), [=](sycl::id<1> i) {
        int e = ids[i];
        if (e < 0 || e >= E) return;
        int slot = sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device>(cursor[e])++;
        if (slot >= route_cap) return;
        slot_row[slot] = int(i) / top_k;
        slot_w[slot] = rw[i];
        slot_exp[slot] = e;
      });
    });
    q.submit([&](sycl::handler& h) {
      h.parallel_for(sycl::range<2>(route_cap, H), [=](sycl::id<2> ij) {
        int s = int(ij[0]), c = int(ij[1]);
        act[size_t(s) * H + c] = act_src[size_t(slot_row[s]) * H + c];
      });
    });
    // 4. w13 grouped GEMM -> [routes, 2I]
    q.memset(atom, 0, sizeof(int32_t));
    launch<'R', 'C', MoE::w4a16_policy_m_32_k16, MoE::A_DTYPE::BITS16,
           MoE::B_DTYPE::NVFP4, cutlass::half_t, cutlass::float_e2m1_t,
           std::uint8_t, cutlass::half_t, cutlass::half_t>(
        q, act, reinterpret_cast<const cutlass::float_e2m1_t*>(w13), s13,
        bias13, mid, 2 * I, H, rows, E, gs, atom);
    // 5. SwiGLU: gated = silu(gate) * up, with alpha13 folded in
    q.submit([&](sycl::handler& h) {
      h.parallel_for(sycl::range<2>(route_cap, I), [=](sycl::id<2> ij) {
        int s = int(ij[0]), c = int(ij[1]);
        float a = float(mid[size_t(s) * 2 * I + c]);
        float b = float(mid[size_t(s) * 2 * I + I + c]);
        float sc = alpha13[slot_exp[s]];
        a *= sc; b *= sc;
        float silu = a / (1.0f + sycl::exp(-a));
        gated[size_t(s) * I + c] = cutlass::half_t(silu * b);
      });
    });
    // 6. w2 grouped GEMM -> [routes, H]
    q.memset(atom, 0, sizeof(int32_t));
    launch<'R', 'C', MoE::w4a16_policy_m_32_k16, MoE::A_DTYPE::BITS16,
           MoE::B_DTYPE::NVFP4, cutlass::half_t, cutlass::float_e2m1_t,
           std::uint8_t, cutlass::half_t, cutlass::half_t>(
        q, gated, reinterpret_cast<const cutlass::float_e2m1_t*>(w2), s2,
        bias2, outr, H, I, rows, E, gs, atom);
    // 7. weighted scatter back to token rows, alpha2 applied here
    q.memset(final_out, 0, size_t(tokens) * H * sizeof(float));
    q.submit([&](sycl::handler& h) {
      h.parallel_for(sycl::range<2>(route_cap, H), [=](sycl::id<2> ij) {
        int s = int(ij[0]), c = int(ij[1]);
        int row = slot_row[s];
        float v = float(outr[size_t(s) * H + c]) * slot_w[s] * alpha2[slot_exp[s]];
        sycl::atomic_ref<float, sycl::memory_order::relaxed,
                         sycl::memory_scope::device>(
            final_out[size_t(row) * H + c]) += v;
      });
    });
  };

  // rows_per_expert must match the histogram the scatter produced
  q.submit([&](sycl::handler& h) {
    h.single_task([=]() { for (int e = 0; e < E; ++e) rows[e] = hist[e]; });
  }).wait();

  try {
    one_layer(); q.wait_and_throw();
    q.submit([&](sycl::handler& h) {
      h.single_task([=]() { for (int e = 0; e < E; ++e) rows[e] = hist[e]; });
    }).wait();
    one_layer(); q.wait_and_throw();

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) one_layer();
    q.wait_and_throw();
    double ms_pipelined = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count() / iters;

    // Same work, but drained after every layer. If this matches the pipelined
    // figure, back-to-back layers cost nothing extra and the shortfall is in
    // the stages themselves. If it is FASTER, the gap is a queueing artifact.
    auto t0b = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) { one_layer(); q.wait_and_throw(); }
    double ms_drained = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0b).count() / iters;
    printf("\n=== gap experiment ===\n");
    printf("  pipelined (1 wait / %d layers) : %.3f ms/layer\n", iters, ms_pipelined);
    printf("  drained   (1 wait / layer)     : %.3f ms/layer\n", ms_drained);
    printf("  attributed stage sum           : see below\n");
    // Headline is the DRAINED figure: the provider's doorbell drains every
    // layer (issue -> take waits before the next issue), so pipelined timing
    // would describe a regime production never runs in.
    double ms = ms_drained;

    printf("=== full grouped MoE layer, real bank weights ===\n");
    printf("per layer      : %.3f ms\n", ms);
    printf("47 layers      : %.1f ms per %d-token chunk\n", ms * 47, tokens);
    printf("per token      : %.1f us/token\n", ms * 47 * 1000.0 / tokens);
    printf("\nproduction     : 1705.0 us/token (measured, Bench 22)\n");
    double sp = 1705.0 / (ms * 47 * 1000.0 / tokens);
    printf("B70 GEMM leg   : %.1fx\n", sp);
    double e2e = 1.0 / (0.10 + 0.90 / sp);
    printf("end-to-end     : ~%.1fx  (B70 at 90%% of TTFT)\n", e2e);
    printf("prefill        : 1705 -> ~%.0f us/token\n", 1705.0 / e2e);
    printf("8K cold TTFT   : 13.8 s -> ~%.1f s\n", 13.8 / e2e);

    // Stage attribution. Waits between stages serialise the pipeline, so the
    // sum slightly exceeds the fused figure above -- it is for apportioning
    // blame, not for quoting.
    auto stage = [&](const char* name, auto&& fn) {
      fn(); q.wait_and_throw();
      auto s0 = std::chrono::steady_clock::now();
      for (int i = 0; i < iters; ++i) fn();
      q.wait_and_throw();
      double sms = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - s0).count() / iters;
      printf("  %-22s %8.3f ms\n", name, sms);
      return sms;
    };
    printf("\n=== stage attribution ===\n");
    double t_hist = stage("histogram", [&]{
      q.memset(hist, 0, (E + 1) * sizeof(int));
      q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(route_cap), [=](sycl::id<1> i) {
          int e = ids[i];
          if (e >= 0 && e < E)
            sycl::atomic_ref<int, sycl::memory_order::relaxed,
                             sycl::memory_scope::device>(hist[e])++;
        });
      });
    });
    double t_pfx = stage("prefix sum (single_task)", [&]{
      q.submit([&](sycl::handler& h) {
        h.single_task([=]() {
          int acc = 0;
          for (int e = 0; e < E; ++e) { offs[e] = acc; acc += hist[e]; cursor[e] = offs[e]; }
          offs[E] = acc;
        });
      });
    });
    double t_rscat = stage("route scatter", [&]{
      q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(route_cap), [=](sycl::id<1> i) {
          int e = ids[i];
          if (e < 0 || e >= E) return;
          int slot = sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                      sycl::memory_scope::device>(cursor[e])++;
          if (slot >= route_cap) return;
          slot_row[slot] = int(i) / top_k;
          slot_w[slot] = rw[i];
          slot_exp[slot] = e;
        });
      });
    });
    double t_gather = stage("gather activations", [&]{
      q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>(route_cap, H), [=](sycl::id<2> ij) {
          int s = int(ij[0]), c = int(ij[1]);
          act[size_t(s) * H + c] = act_src[size_t(slot_row[s]) * H + c];
        });
      });
    });
    double t_g1 = stage("GEMM w13", [&]{
      q.memset(atom, 0, sizeof(int32_t));
      launch<'R', 'C', MoE::w4a16_policy_m_32_k16, MoE::A_DTYPE::BITS16,
             MoE::B_DTYPE::NVFP4, cutlass::half_t, cutlass::float_e2m1_t,
             std::uint8_t, cutlass::half_t, cutlass::half_t>(
          q, act, reinterpret_cast<const cutlass::float_e2m1_t*>(w13), s13,
          bias13, mid, 2 * I, H, rows, E, gs, atom);
    });
    double t_silu = stage("SwiGLU", [&]{
      q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>(route_cap, I), [=](sycl::id<2> ij) {
          int s = int(ij[0]), c = int(ij[1]);
          float a = float(mid[size_t(s) * 2 * I + c]);
          float b = float(mid[size_t(s) * 2 * I + I + c]);
          float sc = alpha13[slot_exp[s]];
          a *= sc; b *= sc;
          gated[size_t(s) * I + c] =
              cutlass::half_t((a / (1.0f + sycl::exp(-a))) * b);
        });
      });
    });
    double t_g2 = stage("GEMM w2", [&]{
      q.memset(atom, 0, sizeof(int32_t));
      launch<'R', 'C', MoE::w4a16_policy_m_32_k16, MoE::A_DTYPE::BITS16,
             MoE::B_DTYPE::NVFP4, cutlass::half_t, cutlass::float_e2m1_t,
             std::uint8_t, cutlass::half_t, cutlass::half_t>(
          q, gated, reinterpret_cast<const cutlass::float_e2m1_t*>(w2), s2,
          bias2, outr, H, I, rows, E, gs, atom);
    });
    double t_zero = stage("memset final_out", [&]{
      q.memset(final_out, 0, size_t(tokens) * H * sizeof(float));
    });
    double t_scat = stage("atomic scatter", [&]{
      q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>(route_cap, H), [=](sycl::id<2> ij) {
          int s = int(ij[0]), c = int(ij[1]);
          float v = float(outr[size_t(s) * H + c]) * slot_w[s] * alpha2[slot_exp[s]];
          sycl::atomic_ref<float, sycl::memory_order::relaxed,
                           sycl::memory_scope::device>(
              final_out[size_t(slot_row[s]) * H + c]) += v;
        });
      });
    });
    double gemms = t_g1 + t_g2;
    double support = t_hist + t_pfx + t_rscat + t_gather + t_silu + t_zero + t_scat;
    printf("  %-22s %8.3f ms  (%.0f%%)\n", "GEMMs", gemms,
           gemms / (gemms + support) * 100.0);
    printf("  %-22s %8.3f ms  (%.0f%%)\n", "support (mine)", support,
           support / (gemms + support) * 100.0);
    printf("\nif support were free: %.1f us/token -> %.1fx leg\n",
           gemms * 47 * 1000.0 / tokens, 1705.0 / (gemms * 47 * 1000.0 / tokens));
  } catch (sycl::exception const& e) {
    printf("THREW: %s\n", e.what());
    return 1;
  }
  return 0;
}
