// Measure achievable host<->B70 PCIe bandwidth directly.
//
// sysfs reports the card's internal switch downstream port and the GPU at
// 2.5 GT/s x1 (PCIe Gen1 x1, ~250 MB/s theoretical per direction) while the
// card's upstream port runs 16 GT/s x4. If that is real and not a sysfs
// artifact, every host<->device copy is capped near 200 MB/s, which is the
// first-order explanation for prefill costing ~480 us/token: each token
// crosses this link once per B70-active layer.
//
// Gen1 x1   ~0.25 GB/s/dir      Gen3 x4  ~3.9 GB/s/dir
// Gen4 x4   ~7.9 GB/s/dir       Gen4 x8  ~15.8 GB/s/dir
//
// Build:
//   source /opt/intel/oneapi/setvars.sh
//   icpx -fsycl -O2 -o experiments/b70_pcie_bw experiments/b70_pcie_bw.cpp
//
// Run:
//   ./experiments/b70_pcie_bw [size_mib] [iters]

#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

double elapsed_s(std::chrono::steady_clock::time_point a,
                 std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

// Median rather than mean: a single scheduling hiccup on a shared card should
// not decide the verdict, and the question is what the link sustains.
double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  // "sweep" walks transfer sizes from a single dispatch payload up to a bulk
  // weight move. The dispatch path sends ~12 KiB per token per layer; the
  // streaming path moves ~0.41 GiB per layer. Same link, four orders of
  // magnitude apart in message size, and only the large end sees peak GB/s.
  const bool sweep = argc > 1 && std::string(argv[1]) == "sweep";
  const size_t mib = (!sweep && argc > 1) ? std::stoul(argv[1]) : 64;
  const int iters = argc > 2 ? std::stoi(argv[2]) : 10;
  const size_t bytes = sweep ? (64ull << 20) : mib * 1024ull * 1024ull;

  sycl::device dev;
  try {
    dev = sycl::device(sycl::gpu_selector_v);
  } catch (const sycl::exception& e) {
    std::fprintf(stderr, "no SYCL GPU: %s\n", e.what());
    return 1;
  }
  sycl::queue q(dev, sycl::property::queue::in_order{});

  std::printf("device      : %s\n", dev.get_info<sycl::info::device::name>().c_str());

  // Pinned (host USM) staging on both sides, matching what the provider does:
  // the dispatch path copies between pinned host pages and device memory.
  void* host = sycl::malloc_host(bytes, q);
  void* devm = sycl::malloc_device(bytes, q);
  if (!host || !devm) {
    std::fprintf(stderr, "allocation failed (%zu MiB)\n", mib);
    return 1;
  }
  std::memset(host, 0xA5, bytes);

  // Warm up: first copy pays mapping and any lazy driver init.
  q.memcpy(devm, host, bytes).wait();
  q.memcpy(host, devm, bytes).wait();
  if (sweep) {
    // 12 KiB is one token's dispatch payload for a 2048-hidden model
    // (4 KiB fp16 in + 8 KiB fp32 out); 1.5 MiB is a 128-row chunk.
    const size_t sizes[] = {4ull << 10,  12ull << 10,  64ull << 10,
                            256ull << 10, 1536ull << 10, 8ull << 20,
                            32ull << 20, 64ull << 20};
    std::printf("\n%12s %14s %14s %12s\n", "size", "H2D GB/s", "D2H GB/s",
                "H2D us");
    for (size_t sz : sizes) {
      const int n = sz < (1ull << 20) ? 200 : 20;
      q.memcpy(devm, host, sz).wait();
      std::vector<double> a, b;
      for (int i = 0; i < n; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        q.memcpy(devm, host, sz).wait();
        auto t1 = std::chrono::steady_clock::now();
        q.memcpy(host, devm, sz).wait();
        auto t2 = std::chrono::steady_clock::now();
        a.push_back(sz / elapsed_s(t0, t1) / 1e9);
        b.push_back(sz / elapsed_s(t1, t2) / 1e9);
      }
      const double ha = median(a);
      std::printf("%10zu B %14.3f %14.3f %12.1f\n", sz, ha, median(b),
                  sz / (ha * 1e9) * 1e6);
    }
    sycl::free(host, q);
    sycl::free(devm, q);
    return 0;
  }


  std::vector<double> h2d, d2h;
  for (int i = 0; i < iters; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    q.memcpy(devm, host, bytes).wait();
    auto t1 = std::chrono::steady_clock::now();
    q.memcpy(host, devm, bytes).wait();
    auto t2 = std::chrono::steady_clock::now();
    h2d.push_back(bytes / elapsed_s(t0, t1) / 1e9);
    d2h.push_back(bytes / elapsed_s(t1, t2) / 1e9);
  }

  const double h = median(h2d), d = median(d2h);
  std::printf("H2D  median : %7.3f GB/s   (max %7.3f)\n", h,
              *std::max_element(h2d.begin(), h2d.end()));
  std::printf("D2H  median : %7.3f GB/s   (max %7.3f)\n", d,
              *std::max_element(d2h.begin(), d2h.end()));
  std::printf("\nreference   : Gen1 x1 ~0.25 | Gen3 x4 ~3.9 | Gen4 x4 ~7.9 GB/s per direction\n");
  const double best = h > d ? h : d;
  if (best < 0.4)
    std::printf("verdict     : consistent with PCIe Gen1 x1 -- link is NOT trained up\n");
  else if (best < 5.0)
    std::printf("verdict     : below Gen4 x4; link is degraded but above Gen1 x1\n");
  else
    std::printf("verdict     : consistent with Gen4 x4 or better\n");

  sycl::free(host, q);
  sycl::free(devm, q);
  return 0;
}
