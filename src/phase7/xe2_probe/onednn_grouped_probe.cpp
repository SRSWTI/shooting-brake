// Gate-2 numerics + perf probe: oneDNN grouped matmul against the Shooting
// Brake NVFP4 bank layout, before any provider integration.
//
// Findings encoded as checks (rev 2 after first run):
//   1. nibble   -- oneDNN reads our packed [E, N, K/2] planes low-nibble-first,
//                  matching the bank (ModelOpt/OCP). Weights alias zero-copy.
//   2. scales   -- a strided (acb) scale md is NOT honored: the primitive
//                  assumes dense canonical [E, K/16, N]. The provider must
//                  keep repacked scale planes (~2.35 GB/card, one-time).
//   3. offsets  -- grouped md row dim may be an upper bound; the device
//                  offsets buffer defines the real work (sync-free provider).
//   4. dst dtype -- impl dispatch is dtype-sensitive; sweep f16 vs f32 dst
//                  and print impl_info to chase the benchdnn-class numbers
//                  (0.970 ms/layer @ fill 30).
//
// Build: src/phase7/xe2_probe/build_onednn_probe.sh (oneAPI sourced).
// Run:   LD_LIBRARY_PATH=vendor/oneDNN/build/src ./onednn_grouped_probe

#include <sycl/sycl.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"

namespace {

using dnnl::memory;
using dt = memory::data_type;
using tag = memory::format_tag;

// ---- reference decoders -----------------------------------------------

constexpr float kE2M1[16] = {0.0f, 0.5f,  1.0f,  1.5f,  2.0f,  3.0f,
                             4.0f, 6.0f,  -0.0f, -0.5f, -1.0f, -1.5f,
                             -2.0f, -3.0f, -4.0f, -6.0f};

float decode_e4m3(std::uint8_t v) {
  const int s = v >> 7;
  const int e = (v >> 3) & 0xF;
  const int m = v & 0x7;
  if (e == 0xF && m == 0x7) return std::nanf("");
  const float x = e == 0 ? std::ldexp(static_cast<float>(m) / 8.0f, -6)
                         : std::ldexp(1.0f + static_cast<float>(m) / 8.0f,
                                      e - 7);
  return s ? -x : x;
}

// Nonzero-mantissa e4m3 scales; constant powers of two hide narrowing bugs.
constexpr std::uint8_t kScaleBytes[4] = {0x3Bu, 0x2Du, 0x35u, 0x43u};

// ---- oracle -------------------------------------------------------------

void oracle(const std::vector<sycl::half>& src,      // [rows_total, K]
            const std::vector<std::uint8_t>& wgt,    // [E, N, K/2]
            const std::vector<std::uint8_t>& scl,    // [E, N, K/16]
            const std::vector<int>& rows_per_e, int K, int N, bool low_first,
            std::vector<float>& dst) {               // [rows_total, N]
  const int E = static_cast<int>(rows_per_e.size());
  std::size_t row0 = 0;
  for (int e = 0; e < E; ++e) {
    for (int r = 0; r < rows_per_e[e]; ++r) {
      const std::size_t m = row0 + r;
      for (int n = 0; n < N; ++n) {
        double acc = 0.0;
        const std::uint8_t* wrow =
            wgt.data() + (static_cast<std::size_t>(e) * N + n) * (K / 2);
        const std::uint8_t* srow =
            scl.data() + (static_cast<std::size_t>(e) * N + n) * (K / 16);
        for (int k = 0; k < K; ++k) {
          const std::uint8_t byte = wrow[k / 2];
          const bool high = (k & 1) ^ (low_first ? 0 : 1);
          const int nib = high ? (byte >> 4) & 0xF : byte & 0xF;
          acc += static_cast<double>(static_cast<float>(src[m * K + k])) *
                 kE2M1[nib] * decode_e4m3(srow[k / 16]);
        }
        dst[m * N + n] = static_cast<float>(acc);
      }
    }
    row0 += rows_per_e[e];
  }
}

double max_rel_err(const float* got, const std::vector<float>& ref,
                   std::size_t count) {
  double worst = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    const double denom = std::max(1.0, std::fabs(static_cast<double>(ref[i])));
    worst = std::max(worst,
                     std::fabs(static_cast<double>(got[i]) - ref[i]) / denom);
  }
  return worst;
}

// ---- oneDNN runner -------------------------------------------------------

struct Runner {
  sycl::queue& q;
  dnnl::engine eng;
  dnnl::stream strm;

  explicit Runner(sycl::queue& queue)
      : q(queue),
        eng(dnnl::sycl_interop::make_engine(q.get_device(), q.get_context())),
        strm(dnnl::sycl_interop::make_stream(eng, q)) {}

  struct Bound {
    dnnl::matmul prim;
    std::unordered_map<int, dnnl::memory> args;
    std::string impl;
  };

  // src f16 grouped [rows_md, K]; wei f4 aliasing packed [E, N, K/2];
  // scl e4m3 dense canonical [E, K/16, N] unless scale_acb; dst grouped
  // [rows_md, N] of dst_dt.
  Bound bind(const sycl::half* src, const std::uint8_t* wgt,
             const std::uint8_t* scl, void* dst, std::int32_t* offs_dev,
             int rows_md, int E, int K, int N, dt dst_dt, bool scale_acb) {
    auto src_md = memory::desc::grouped({rows_md, K}, dt::f16, 0, E);
    auto dst_md = memory::desc::grouped({rows_md, N}, dst_dt, 0, E);
    // Bank plane [E, N, K] with K contiguous == acb on logical {E, K, N}.
    auto wei_md = memory::desc({E, K, N}, dt::f4_e2m1, tag::acb);

    dnnl::primitive_attr attr;
    attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1) | (1 << 2), {16, 1},
                    dt::f8_e4m3, false);

    auto pd = dnnl::matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr);
    auto scl_md = memory::desc({E, K / 16, N}, dt::f8_e4m3,
                               scale_acb ? tag::acb : tag::abc);

    Bound b{dnnl::matmul(pd), {}, pd.impl_info_str()};
    b.args.emplace(DNNL_ARG_SRC,
                   dnnl::memory(src_md, eng,
                                {const_cast<sycl::half*>(src),
                                 static_cast<void*>(offs_dev)}));
    b.args.emplace(DNNL_ARG_WEIGHTS,
                   dnnl::memory(wei_md, eng, const_cast<std::uint8_t*>(wgt)));
    b.args.emplace(DNNL_ARG_DST,
                   dnnl::memory(dst_md, eng, {dst, static_cast<void*>(offs_dev)}));
    b.args.emplace(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
                   dnnl::memory(scl_md, eng, const_cast<std::uint8_t*>(scl)));
    return b;
  }

  void exec(Bound& b) { b.prim.execute(strm, b.args); }
};

// Fixture in bank layout. Deterministic.
struct Fixture {
  std::vector<sycl::half> src;
  std::vector<std::uint8_t> wgt;      // [E, N, K/2]
  std::vector<std::uint8_t> scl;      // [E, N, K/16]  (bank order)
  std::vector<std::uint8_t> scl_abc;  // dense canonical [E, K/16, N]
  std::vector<int> rows;
  std::vector<std::int32_t> offs;     // inclusive cumulative ends, length E
  int rows_total = 0;
  int K, N;

  Fixture(std::vector<int> rows_per_e, int k, int n, unsigned seed)
      : rows(std::move(rows_per_e)), K(k), N(n) {
    const int E = static_cast<int>(rows.size());
    offs.resize(E);
    for (int e = 0; e < E; ++e) {
      rows_total += rows[e];
      offs[e] = rows_total;
    }
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> byte(0, 255);
    src.resize(static_cast<std::size_t>(rows_total) * K);
    for (auto& v : src) {
      v = static_cast<sycl::half>(
          (static_cast<float>(byte(rng)) - 127.5f) / 64.0f);
    }
    wgt.resize(static_cast<std::size_t>(E) * N * (K / 2));
    for (auto& v : wgt) v = static_cast<std::uint8_t>(byte(rng));
    scl.resize(static_cast<std::size_t>(E) * N * (K / 16));
    for (std::size_t i = 0; i < scl.size(); ++i) scl[i] = kScaleBytes[i & 3];
    const int KG = K / 16;
    scl_abc.resize(scl.size());
    for (int e = 0; e < E; ++e)
      for (int n2 = 0; n2 < N; ++n2)
        for (int g = 0; g < KG; ++g)
          scl_abc[(static_cast<std::size_t>(e) * KG + g) * N + n2] =
              scl[(static_cast<std::size_t>(e) * N + n2) * KG + g];
  }
};

struct DeviceCopy {
  sycl::half* src;
  std::uint8_t* wgt;
  std::uint8_t* scl;
  std::uint8_t* scl_abc;
  void* dst;  // sized for f32; also fits f16
  std::int32_t* offs;
  std::size_t dst_elems;

  DeviceCopy(sycl::queue& q, const Fixture& f, int rows_md) {
    dst_elems = static_cast<std::size_t>(rows_md) * f.N;
    src = sycl::malloc_device<sycl::half>(
        static_cast<std::size_t>(rows_md) * f.K, q);
    wgt = sycl::malloc_device<std::uint8_t>(f.wgt.size(), q);
    scl = sycl::malloc_device<std::uint8_t>(f.scl.size(), q);
    scl_abc = sycl::malloc_device<std::uint8_t>(f.scl_abc.size(), q);
    dst = sycl::malloc_device<float>(dst_elems, q);
    offs = sycl::malloc_device<std::int32_t>(f.offs.size(), q);
    q.memcpy(src, f.src.data(), f.src.size() * sizeof(sycl::half));
    q.memcpy(wgt, f.wgt.data(), f.wgt.size());
    q.memcpy(scl, f.scl.data(), f.scl.size());
    q.memcpy(scl_abc, f.scl_abc.data(), f.scl_abc.size());
    q.memcpy(offs, f.offs.data(), f.offs.size() * sizeof(std::int32_t));
    q.fill(reinterpret_cast<float*>(dst), -1.0f, dst_elems).wait();
  }
  void free(sycl::queue& q) {
    sycl::free(src, q); sycl::free(wgt, q); sycl::free(scl, q);
    sycl::free(scl_abc, q); sycl::free(dst, q); sycl::free(offs, q);
  }
};

int check(const char* name, bool ok, double err) {
  std::printf("%-34s %s  (max rel err %.3e)\n", name, ok ? "PASS" : "FAIL",
              err);
  return ok ? 0 : 1;
}

}  // namespace

int main() {
  sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
  std::printf("device: %s\n",
              q.get_device().get_info<sycl::info::device::name>().c_str());
  Runner run(q);
  int failures = 0;

  // ---- correctness, small geometry, empty group included -----------------
  {
    Fixture f({7, 0, 19, 3}, /*K=*/128, /*N=*/64, 0x5EED);
    const int E = static_cast<int>(f.rows.size());
    const int rows_md = f.rows_total;

    std::vector<float> ref_low(static_cast<std::size_t>(f.rows_total) * f.N);
    std::vector<float> ref_high(ref_low.size());
    oracle(f.src, f.wgt, f.scl, f.rows, f.K, f.N, true, ref_low);
    oracle(f.src, f.wgt, f.scl, f.rows, f.K, f.N, false, ref_high);

    DeviceCopy d(q, f, rows_md);
    std::vector<float> got(ref_low.size());

    // Canonical scales, f32 dst: nibble convention + baseline correctness.
    {
      auto b = run.bind(d.src, d.wgt, d.scl_abc, d.dst, d.offs, rows_md, E,
                        f.K, f.N, dt::f32, false);
      run.exec(b);
      q.wait();
      q.memcpy(got.data(), d.dst, got.size() * sizeof(float)).wait();
      const double e_low = max_rel_err(got.data(), ref_low, got.size());
      const double e_high = max_rel_err(got.data(), ref_high, got.size());
      std::printf("impl: %s | low-first err %.3e, high-first err %.3e\n",
                  b.impl.c_str(), e_low, e_high);
      failures += check("canonical scales, f32 dst", e_low < 5e-3, e_low);
    }

    // f16 dst correctness (candidate g_mid dtype for the fast impl).
    {
      q.fill(reinterpret_cast<float*>(d.dst), -1.0f, d.dst_elems).wait();
      auto b = run.bind(d.src, d.wgt, d.scl_abc, d.dst, d.offs, rows_md, E,
                        f.K, f.N, dt::f16, false);
      run.exec(b);
      q.wait();
      std::vector<sycl::half> got16(got.size());
      q.memcpy(got16.data(), d.dst, got16.size() * sizeof(sycl::half)).wait();
      for (std::size_t i = 0; i < got.size(); ++i)
        got[i] = static_cast<float>(got16[i]);
      const double e = max_rel_err(got.data(), ref_low, got.size());
      std::printf("impl (f16 dst): %s\n", b.impl.c_str());
      failures += check("canonical scales, f16 dst", e < 2e-2, e);
    }

    // Strided acb scale md: expected NOT honored (documents the repack need).
    {
      q.fill(reinterpret_cast<float*>(d.dst), -1.0f, d.dst_elems).wait();
      auto b = run.bind(d.src, d.wgt, d.scl, d.dst, d.offs, rows_md, E, f.K,
                        f.N, dt::f32, true);
      run.exec(b);
      q.wait();
      q.memcpy(got.data(), d.dst, got.size() * sizeof(float)).wait();
      const double e = max_rel_err(got.data(), ref_low, got.size());
      std::printf("strided acb scale md honored: %s (err %.3e)\n",
                  e < 5e-3 ? "YES (zero-copy scales!)" : "no (repack at load)",
                  e);
    }

    // Offsets as upper bound with canonical scales.
    {
      const int rows_pad = rows_md * 2;
      DeviceCopy d2(q, f, rows_pad);
      auto b = run.bind(d2.src, d2.wgt, d2.scl_abc, d2.dst, d2.offs, rows_pad,
                        E, f.K, f.N, dt::f32, false);
      run.exec(b);
      q.wait();
      std::vector<float> got2(d2.dst_elems);
      q.memcpy(got2.data(), d2.dst, got2.size() * sizeof(float)).wait();
      const double e = max_rel_err(got2.data(), ref_low, ref_low.size());
      bool tail_untouched = true;
      for (std::size_t i = ref_low.size(); i < got2.size(); ++i) {
        if (got2[i] != -1.0f) { tail_untouched = false; break; }
      }
      failures += check("offsets as upper bound", e < 5e-3, e);
      failures += check("padded tail untouched", tail_untouched, 0.0);
      d2.free(q);
    }
    d.free(q);
  }

  // ---- full-geometry perf, cached primitive, dst dtype sweep -------------
  {
    const int E = 85;
    struct Shape { const char* name; int K, N; };
    const Shape gemms[2] = {{"GEMM1 K=3072 N=2048", 3072, 2048},
                            {"GEMM2 K=1024 N=3072", 1024, 3072}};
    for (dt dst_dt : {dt::f16, dt::f32}) {
      const char* dname = dst_dt == dt::f16 ? "f16" : "f32";
      for (int fill : {30, 120}) {
        double layer_ms = 0.0;
        for (const auto& g : gemms) {
          Fixture f(std::vector<int>(E, fill), g.K, g.N, 0xB70 + fill);
          DeviceCopy d(q, f, f.rows_total);
          auto b = run.bind(d.src, d.wgt, d.scl_abc, d.dst, d.offs,
                            f.rows_total, E, g.K, g.N, dst_dt, false);
          for (int i = 0; i < 10; ++i) run.exec(b);
          q.wait();
          double best_ms = 1e30;
          for (int rep = 0; rep < 3; ++rep) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 20; ++i) run.exec(b);
            q.wait();
            const auto t1 = std::chrono::steady_clock::now();
            best_ms = std::min(
                best_ms,
                std::chrono::duration<double, std::milli>(t1 - t0).count() /
                    20);
          }
          const double tflops = 2.0 * f.rows_total * g.K * g.N / best_ms / 1e9;
          std::printf("perf dst=%s fill=%-3d %s: %.4f ms  %.1f TFLOP/s  [%s]\n",
                      dname, fill, g.name, best_ms, tflops, b.impl.c_str());
          layer_ms += best_ms;
          d.free(q);
        }
        std::printf("perf dst=%s fill=%-3d layer total: %.4f ms (native 2.355)\n",
                    dname, fill, layer_ms);
      }
    }
  }

  std::printf(failures ? "RESULT: %d FAILURE(S)\n" : "RESULT: ALL PASS\n",
              failures);
  return failures ? 1 : 0;
}
