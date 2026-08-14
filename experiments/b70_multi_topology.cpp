// Two-B70 topology probe. Decides the multi-device architecture.
//
// Measured PCI topology on this box (sysfs max_link_speed - hard caps, not
// idle downtraining):
//
//   5090   01:00.0 -> 00:01.1                       Gen5 x16, own CPU lanes
//   B70-A  15:00.0 -> 0a:08.0  Gen4 x4   (original)
//   B70-B  11:00.0 -> 0a:04.0  Gen3 x4   (newly added)
//                       both -> 09:00.0  Gen4 x4   SHARED uplink to CPU
//
// Intel VRAM doubled to 64 GB and local Intel bandwidth doubled (~608 GB/s per
// card), but host<->Intel bandwidth did NOT increase. Three questions:
//
//   Q1 Does each card reach its own link ceiling alone?        (asymmetry)
//   Q2 Do concurrent transfers contend on shared 09:00.0?      (aggregate cap)
//   Q3 Can the two B70s peer directly through the switch?      (the big one)
//
// Q3 caveat: the in-tree journal (intel-xpu/vllm-xpu/b70_ai_things) found peer
// access returns false with IOMMU on, and Puget measured PCIe RxErr plus
// "Engine reset: engine_class=bcs" attempting P2P on a 4x B70 box. The
// capability QUERY is always safe; the peer COPY is opt-in via argv.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <sycl/sycl.hpp>

namespace {

using Clock = std::chrono::steady_clock;

double secs_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

constexpr std::size_t kXfer = 64ull << 20; // 64 MiB, well past the latency floor
constexpr int kIters = 8;

double h2d_gbps(sycl::queue &q, void *host, void *dev) {
  for (int i = 0; i < 3; ++i) q.memcpy(dev, host, kXfer).wait();
  std::vector<double> r;
  for (int i = 0; i < kIters; ++i) {
    auto t0 = Clock::now();
    q.memcpy(dev, host, kXfer).wait();
    r.push_back(kXfer / secs_since(t0) / 1e9);
  }
  return median(r);
}

} // namespace

int main(int argc, char **argv) {
  // Peer COPY is opt-in. The query below is harmless either way.
  const bool try_peer_copy = argc > 1 && std::string(argv[1]) == "p2p";

  std::vector<sycl::device> gpus;
  for (const auto &p : sycl::platform::get_platforms())
    for (const auto &d : p.get_devices(sycl::info::device_type::gpu))
      if (d.get_backend() == sycl::backend::ext_oneapi_level_zero)
        gpus.push_back(d);

  std::printf("=== Level Zero GPUs ===\n");
  for (std::size_t i = 0; i < gpus.size(); ++i)
    std::printf("  [%zu] %-38s  vram %.1f GiB\n", i,
                gpus[i].get_info<sycl::info::device::name>().c_str(),
                gpus[i].get_info<sycl::info::device::global_mem_size>() / 1073741824.0);
  std::fflush(stdout);

  if (gpus.size() < 2) {
    std::printf("\nfewer than 2 Level Zero GPUs visible -- nothing to test\n");
    return 0;
  }

  sycl::queue qa(gpus[0], sycl::property::queue::in_order{});
  sycl::queue qb(gpus[1], sycl::property::queue::in_order{});

  void *ha = sycl::malloc_host(kXfer, qa);
  void *hb = sycl::malloc_host(kXfer, qb);
  void *da = sycl::malloc_device(kXfer, qa);
  void *db = sycl::malloc_device(kXfer, qb);
  std::memset(ha, 0x5A, kXfer);
  std::memset(hb, 0xA5, kXfer);

  // --- Q1 ------------------------------------------------------------------
  std::printf("\n=== Q1: host->device, one card at a time ===\n");
  const double a_alone = h2d_gbps(qa, ha, da);
  const double b_alone = h2d_gbps(qb, hb, db);
  std::printf("  dev0 alone : %6.2f GB/s\n", a_alone);
  std::printf("  dev1 alone : %6.2f GB/s\n", b_alone);
  std::printf("  (Gen4 x4 ~7.9 theoretical, Gen3 x4 ~3.9 theoretical)\n");
  std::fflush(stdout);

  // --- Q2 ------------------------------------------------------------------
  std::printf("\n=== Q2: host->device, BOTH concurrently (shared 09:00.0) ===\n");
  {
    for (int i = 0; i < 2; ++i) {
      qa.memcpy(da, ha, kXfer).wait();
      qb.memcpy(db, hb, kXfer).wait();
    }
    std::vector<double> agg;
    for (int i = 0; i < kIters; ++i) {
      auto t0 = Clock::now();
      std::thread ta([&] { qa.memcpy(da, ha, kXfer).wait(); });
      std::thread tb([&] { qb.memcpy(db, hb, kXfer).wait(); });
      ta.join();
      tb.join();
      agg.push_back(2.0 * kXfer / secs_since(t0) / 1e9);
    }
    const double both = median(agg);
    std::printf("  aggregate            : %6.2f GB/s\n", both);
    std::printf("  sum if independent   : %6.2f GB/s\n", a_alone + b_alone);
    std::printf("  -> %s\n", both < 0.75 * (a_alone + b_alone)
                                 ? "CONTENDING on the shared uplink"
                                 : "scaling roughly independently");
    std::fflush(stdout);
  }

  // --- Q3 ------------------------------------------------------------------
  std::printf("\n=== Q3: B70 <-> B70 peer access ===\n");
  bool peer01 = false;
  try {
    peer01 = gpus[0].ext_oneapi_can_access_peer(
        gpus[1], sycl::ext::oneapi::peer_access::access_supported);
    const bool peer10 = gpus[1].ext_oneapi_can_access_peer(
        gpus[0], sycl::ext::oneapi::peer_access::access_supported);
    std::printf("  dev0 -> dev1 : %s\n", peer01 ? "YES" : "no");
    std::printf("  dev1 -> dev0 : %s\n", peer10 ? "YES" : "no");
  } catch (const sycl::exception &e) {
    std::printf("  peer query threw: %s\n", e.what());
  }
  std::fflush(stdout);

  if (peer01 && try_peer_copy) {
    try {
      gpus[0].ext_oneapi_enable_peer_access(gpus[1]);
      for (int i = 0; i < 3; ++i) qa.memcpy(db, da, kXfer).wait();
      std::vector<double> r;
      for (int i = 0; i < kIters; ++i) {
        auto t0 = Clock::now();
        qa.memcpy(db, da, kXfer).wait();
        r.push_back(kXfer / secs_since(t0) / 1e9);
      }
      std::printf("  P2P dev0->dev1 direct : %6.2f GB/s\n", median(r));
      gpus[0].ext_oneapi_disable_peer_access(gpus[1]);
    } catch (const sycl::exception &e) {
      std::printf("  P2P copy FAILED: %s\n", e.what());
    }
  } else if (peer01) {
    std::printf("  (peer copy not attempted -- rerun with 'p2p' to try it)\n");
  }
  std::fflush(stdout);

  // Baseline P2P has to beat: device -> host -> device, what we do today.
  {
    for (int i = 0; i < 2; ++i) {
      qa.memcpy(ha, da, kXfer).wait();
      qb.memcpy(db, ha, kXfer).wait();
    }
    std::vector<double> r;
    for (int i = 0; i < kIters; ++i) {
      auto t0 = Clock::now();
      qa.memcpy(ha, da, kXfer).wait();
      qb.memcpy(db, ha, kXfer).wait();
      r.push_back(kXfer / secs_since(t0) / 1e9);
    }
    std::printf("  via host DRAM (today) : %6.2f GB/s  <- the number to beat\n", median(r));
  }

  sycl::free(ha, qa);
  sycl::free(hb, qb);
  sycl::free(da, qa);
  sycl::free(db, qb);
  std::printf("\ndone\n");
  return 0;
}
