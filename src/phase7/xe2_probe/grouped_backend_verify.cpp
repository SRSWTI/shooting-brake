// A/B gate for grouped_moe_nvfp4 backends: runs the FULL production entry
// (histogram -> gather -> GEMM1 -> SwiGLU+alpha13 -> GEMM2 -> weighted
// scatter+alpha2) on a deterministic fixture and dumps the [M, H] fp32
// output. Run once per backend, then compare:
//
//   SYCL_UR_USE_LEVEL_ZERO_V2=0 ./grouped_backend_verify dump /tmp/native.bin 2048
//   SB_GROUPED_BACKEND=onednn SYCL_UR_USE_LEVEL_ZERO_V2=0 \
//       ./grouped_backend_verify dump /tmp/onednn.bin 2048
//   ./grouped_backend_verify compare /tmp/native.bin /tmp/onednn.bin
//
// The fixture mirrors production: E=85, H=3072, I=1024, top_k=10, and 50% of
// routes marked -1 (non-resident, the two-card compaction), so empty and
// thin experts both occur. Backends differ only in DPAS accumulation order,
// so the gate is a small relative tolerance on fp32 outputs.

#include <sycl/sycl.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "grouped_moe.hpp"

namespace {

constexpr int kE = 85, kH = 3072, kI = 1024, kTopK = 10, kGroup = 16;

template <typename T>
T* dev(sycl::queue& q, std::size_t n) {
  T* p = sycl::malloc_device<T>(n, q);
  if (!p) { std::fprintf(stderr, "alloc failed (%zu)\n", n); std::exit(2); }
  return p;
}

int run_dump(const char* path, int M, int bench = 0) {
  sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
  std::fprintf(stderr, "device: %s, M=%d\n",
               q.get_device().get_info<sycl::info::device::name>().c_str(), M);
  const int routes = M * kTopK;
  std::mt19937 rng(0xB70D);
  std::uniform_int_distribution<int> byte(0, 255);

  // Bank planes in production layout.
  const std::size_t w13_b = std::size_t(kE) * 2 * kI * kH / 2;
  const std::size_t s13_b = std::size_t(kE) * 2 * kI * kH / kGroup;
  const std::size_t w2_b = std::size_t(kE) * kH * kI / 2;
  const std::size_t s2_b = std::size_t(kE) * kH * kI / kGroup;
  std::vector<std::uint8_t> h_w13(w13_b), h_s13(s13_b), h_w2(w2_b), h_s2(s2_b);
  for (auto& v : h_w13) v = static_cast<std::uint8_t>(byte(rng));
  for (auto& v : h_w2) v = static_cast<std::uint8_t>(byte(rng));
  // Moderate, mantissa-rich e4m3 scales (0.28..0.81): the native kernel's
  // store path saturates at f16 range, so the fixture must stay well inside
  // the checkpoint's operating envelope (hot scales -> inf, seen rev 1).
  constexpr std::uint8_t kScales[4] = {0x31u, 0x2Du, 0x35u, 0x29u};
  for (std::size_t i = 0; i < h_s13.size(); ++i) h_s13[i] = kScales[i & 3];
  for (std::size_t i = 0; i < h_s2.size(); ++i) h_s2[i] = kScales[(i + 1) & 3];

  std::vector<float> h_a13(kE), h_a2(kE);
  for (int e = 0; e < kE; ++e) {
    h_a13[e] = 0.01f + 0.0001f * static_cast<float>(e);
    h_a2[e] = 0.02f + 0.0002f * static_cast<float>(e);
  }
  std::vector<sycl::half> h_act(std::size_t(M) * kH);
  for (auto& v : h_act) {
    v = static_cast<sycl::half>(
        (static_cast<float>(byte(rng)) - 127.5f) / 1024.f);
  }
  // Routing: skewed over experts, 50% non-resident (-1), incl. hot expert 0.
  std::vector<std::int32_t> h_ids(routes);
  std::vector<float> h_rw(routes);
  for (int i = 0; i < routes; ++i) {
    if (i & 1) {
      h_ids[i] = -1;
      h_rw[i] = 0.0f;
    } else {
      h_ids[i] = (i % 7 == 0) ? 0 : (i * 37 % kE);
      h_rw[i] = 1.0f / kTopK;
    }
  }

  auto* w13 = dev<std::uint8_t>(q, w13_b);
  auto* s13 = dev<std::uint8_t>(q, s13_b);
  auto* w2 = dev<std::uint8_t>(q, w2_b);
  auto* s2 = dev<std::uint8_t>(q, s2_b);
  auto* a13 = dev<float>(q, kE);
  auto* a2 = dev<float>(q, kE);
  auto* act = dev<sycl::half>(q, std::size_t(M) * kH);
  auto* ids = dev<std::int32_t>(q, routes);
  auto* rw = dev<float>(q, routes);
  auto* g_act = dev<sycl::half>(q, std::size_t(routes) * kH);
  auto* g_mid = dev<float>(q, std::size_t(routes) * 2 * kI);
  auto* g_gated = dev<sycl::half>(q, std::size_t(routes) * kI);
  auto* g_outr = dev<float>(q, std::size_t(routes) * kH);
  auto* bias13 = dev<sycl::half>(q, std::size_t(kE) * 2 * kI);
  auto* bias2 = dev<sycl::half>(q, std::size_t(kE) * kH);
  auto* hist = dev<std::int32_t>(q, kE + 1);
  auto* offs = dev<std::int32_t>(q, kE + 1);
  auto* cursor = dev<std::int32_t>(q, kE);
  auto* rows = dev<std::int32_t>(q, kE);
  auto* slot_row = dev<std::int32_t>(q, routes);
  auto* slot_exp = dev<std::int32_t>(q, routes);
  auto* slot_w = dev<float>(q, routes);
  auto* slot_of = dev<std::int32_t>(q, routes);
  auto* atom = dev<std::int32_t>(q, 1);
  auto* out = dev<float>(q, std::size_t(M) * kH);

  q.memcpy(w13, h_w13.data(), w13_b);
  q.memcpy(s13, h_s13.data(), s13_b);
  q.memcpy(w2, h_w2.data(), w2_b);
  q.memcpy(s2, h_s2.data(), s2_b);
  q.memcpy(a13, h_a13.data(), kE * sizeof(float));
  q.memcpy(a2, h_a2.data(), kE * sizeof(float));
  q.memcpy(act, h_act.data(), h_act.size() * sizeof(sycl::half));
  q.memcpy(ids, h_ids.data(), routes * sizeof(std::int32_t));
  q.memcpy(rw, h_rw.data(), routes * sizeof(float));
  q.memset(bias13, 0, std::size_t(kE) * 2 * kI * sizeof(sycl::half));
  q.memset(bias2, 0, std::size_t(kE) * kH * sizeof(sycl::half));
  q.wait();
  const bool ok = sb::xe2::grouped_moe_nvfp4(
      q, act, ids, rw, w13, s13, a13, w2, s2, a2, g_act, g_mid, g_gated,
      g_outr, bias13, bias2, hist, offs, cursor, rows, slot_row, slot_exp,
      slot_w, slot_of, atom, out, M, kE, kTopK, kH, kI, kGroup, routes);
  q.wait();
  if (!ok) {
    std::fprintf(stderr, "grouped_moe_nvfp4 returned false\n");
    return 2;
  }
  // Stage probes: how far does finiteness survive? Bounded by the real slot
  // count (offs[E]); rows past it are legitimately uninitialized.
  {
    std::int32_t n_slots = 0;
    q.memcpy(&n_slots, offs + kE, sizeof(n_slots)).wait();
    std::vector<std::int32_t> h_rows(kE);
    q.memcpy(h_rows.data(), rows, kE * sizeof(std::int32_t)).wait();
    auto stat = [&](const char* name, const auto* p, std::size_t row_elems) {
      using T = std::remove_cv_t<std::remove_pointer_t<decltype(p)>>;
      const std::size_t elems = std::size_t(n_slots) * row_elems;
      std::vector<T> h(elems);
      q.memcpy(h.data(), p, elems * sizeof(T)).wait();
      std::size_t bad = 0, bad_rows = 0, first_bad_row = SIZE_MAX;
      double peak = 0;
      for (std::size_t s = 0; s < std::size_t(n_slots); ++s) {
        std::size_t row_bad = 0;
        for (std::size_t c = 0; c < row_elems; ++c) {
          const float f = static_cast<float>(h[s * row_elems + c]);
          if (!std::isfinite(f)) ++row_bad;
          else peak = std::max(peak, std::fabs(static_cast<double>(f)));
        }
        bad += row_bad;
        if (row_bad) {
          ++bad_rows;
          if (first_bad_row == SIZE_MAX) first_bad_row = s;
        }
      }
      std::fprintf(stderr,
                   "  %-8s slots=%d nonfinite=%zu bad_rows=%zu first_bad=%zd "
                   "peak=%.4e\n",
                   name, n_slots, bad, bad_rows,
                   first_bad_row == SIZE_MAX ? -1 : (ssize_t)first_bad_row,
                   peak);
    };
    stat("g_mid", g_mid, std::size_t(2) * kI);
    stat("g_gated", g_gated, kI);
    stat("g_outr", g_outr, kH);
    // Cumulative row ends per expert, to map bad rows onto experts.
    std::fprintf(stderr, "  rows[e]: ");
    int acc = 0;
    for (int e = 0; e < kE; ++e) {
      acc += h_rows[e];
      if (e < 8 || e == kE - 1) std::fprintf(stderr, "%d ", acc);
    }
    std::fprintf(stderr, "\n");
  }
  std::vector<float> h_out(std::size_t(M) * kH);
  q.memcpy(h_out.data(), out, h_out.size() * sizeof(float)).wait();

  std::size_t nonzero = 0, nonfinite = 0;
  double l1 = 0;
  for (float v : h_out) {
    if (!std::isfinite(v)) ++nonfinite;
    else if (v != 0.0f) { ++nonzero; l1 += std::fabs(v); }
  }
  std::fprintf(stderr, "out: nonzero=%zu nonfinite=%zu l1=%.6e\n", nonzero,
               nonfinite, l1);
  if (nonfinite || !nonzero) return 2;

  std::FILE* f = std::fopen(path, "wb");
  if (!f || std::fwrite(h_out.data(), sizeof(float), h_out.size(), f) !=
                h_out.size()) {
    std::fprintf(stderr, "dump write failed: %s\n", path);
    return 2;
  }
  std::fclose(f);
  std::printf("dumped %zu floats to %s\n", h_out.size(), path);

  if (bench > 0) {
    // Wall-clock the WHOLE production entry (all eight stages), the number
    // that maps to ms/layer in the provider. Backend chosen by env, so a
    // native-vs-onednn delta here also proves the flag engaged.
    for (int i = 0; i < 5; ++i) {
      sb::xe2::grouped_moe_nvfp4(q, act, ids, rw, w13, s13, a13, w2, s2, a2,
                                 g_act, g_mid, g_gated, g_outr, bias13, bias2,
                                 hist, offs, cursor, rows, slot_row, slot_exp,
                                 slot_w, slot_of, atom, out, M, kE, kTopK, kH,
                                 kI, kGroup, routes);
    }
    q.wait();
    double best_ms = 1e30;
    for (int rep = 0; rep < 3; ++rep) {
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < bench; ++i) {
        sb::xe2::grouped_moe_nvfp4(q, act, ids, rw, w13, s13, a13, w2, s2, a2,
                                   g_act, g_mid, g_gated, g_outr, bias13,
                                   bias2, hist, offs, cursor, rows, slot_row,
                                   slot_exp, slot_w, slot_of, atom, out, M, kE,
                                   kTopK, kH, kI, kGroup, routes);
      }
      q.wait();
      const auto t1 = std::chrono::steady_clock::now();
      best_ms = std::min(
          best_ms,
          std::chrono::duration<double, std::milli>(t1 - t0).count() / bench);
    }
    std::printf("bench M=%d: %.4f ms/layer (all stages)\n", M, best_ms);
  }
  return 0;
}

int run_compare(const char* pa, const char* pb) {
  auto load = [](const char* p) {
    std::FILE* f = std::fopen(p, "rb");
    if (!f) { std::fprintf(stderr, "open failed: %s\n", p); std::exit(2); }
    std::fseek(f, 0, SEEK_END);
    const long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<float> v(static_cast<std::size_t>(bytes) / sizeof(float));
    if (std::fread(v.data(), 1, bytes, f) != static_cast<std::size_t>(bytes))
      std::exit(2);
    std::fclose(f);
    return v;
  };
  const auto a = load(pa), b = load(pb);
  if (a.size() != b.size() || a.empty()) {
    std::fprintf(stderr, "size mismatch %zu vs %zu\n", a.size(), b.size());
    return 2;
  }
  double max_rel = 0, mean_rel = 0, dot = 0, na = 0, nb = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double denom = std::max(1e-3, std::fabs(static_cast<double>(a[i])));
    const double rel = std::fabs(static_cast<double>(b[i]) - a[i]) / denom;
    max_rel = std::max(max_rel, rel);
    mean_rel += rel;
    dot += static_cast<double>(a[i]) * b[i];
    na += static_cast<double>(a[i]) * a[i];
    nb += static_cast<double>(b[i]) * b[i];
  }
  mean_rel /= static_cast<double>(a.size());
  const double cosine = dot / std::max(1e-30, std::sqrt(na) * std::sqrt(nb));
  const bool pass = max_rel < 5e-2 && cosine > 0.9999;
  std::printf("compare: max_rel=%.3e mean_rel=%.3e cosine=%.8f -> %s\n",
              max_rel, mean_rel, cosine, pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc >= 3 && std::strcmp(argv[1], "dump") == 0) {
    return run_dump(argv[2], argc > 3 ? std::atoi(argv[3]) : 2048);
  }
  if (argc >= 3 && std::strcmp(argv[1], "bench") == 0) {
    return run_dump("/tmp/grouped_bench_scratch.bin", std::atoi(argv[2]), 20);
  }
  if (argc == 4 && std::strcmp(argv[1], "compare") == 0) {
    return run_compare(argv[2], argv[3]);
  }
  std::fprintf(stderr,
               "usage: %s dump <out.bin> [M] | %s compare <a.bin> <b.bin>\n",
               argv[0], argv[0]);
  return 2;
}
