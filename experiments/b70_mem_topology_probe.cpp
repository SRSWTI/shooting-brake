/*
 * b70_mem_topology_probe.cpp — per-card transport rates and the host-memory
 * cost of device residency.
 *
 * Two questions this rig could not answer from sysfs or from the existing
 * single-device bandwidth probe:
 *
 * 1. **Is the pair asymmetric, and by how much?** `b70_pcie_bw` measures
 *    whichever device the default selector picks, so it has only ever
 *    characterised one card. Serving splits 204 remote experts across both,
 *    and CUDA waits on the slower one every layer, so the per-card rate is
 *    the number that sets prefill cost. sysfs is useless here: both cards
 *    report an ASPM-downtrained `2.5 GT/s x1` at idle *and* under sustained
 *    load, so the link generation has to be inferred from achieved rate.
 *
 * 2. **Does device residency cost host RAM 1:1?** Booting the 99B server
 *    consumes ~47 GiB of host RAM in two step-drops that match the ~24.3 GiB
 *    of expert weights each card holds -- yet the serving process's RSS is
 *    ~1.2 GiB and page cache is ~1.3 GiB, so the memory is not in the
 *    process address space and not file-backed. If plain device USM
 *    allocation reproduces the step, the backing is the driver's, it is
 *    structural, and it is what makes a 44-56 GiB host-resident prefill bank
 *    impossible on a 59 GiB box. If it does not, the cost belongs to the
 *    provider's upload path and is ours to remove.
 *
 * The allocation ladder writes to every byte it allocates: an untouched USM
 * allocation can be lazily backed, which would answer question 2 with a
 * comfortable lie.
 *
 * Build:
 *   source /opt/intel/oneapi/setvars.sh
 *   icpx -fsycl -O2 -o experiments/b70_mem_topology_probe \
 *       experiments/b70_mem_topology_probe.cpp
 *
 * Run (server must be down; it owns both cards):
 *   ./experiments/b70_mem_topology_probe --alloc-gib 24 --bw-mib 256
 */

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using clock_t_ = std::chrono::steady_clock;

double seconds_since(clock_t_::time_point start) {
  return std::chrono::duration<double>(clock_t_::now() - start).count();
}

// MemAvailable is the kernel's own estimate of what a new allocation can
// take without swapping, which is exactly the quantity a prefill bank would
// have to fit inside.
double mem_available_gib() {
  std::FILE* f = std::fopen("/proc/meminfo", "r");
  if (f == nullptr) {
    return -1.0;
  }
  char key[64];
  unsigned long long value = 0;
  double out = -1.0;
  while (std::fscanf(f, "%63s %llu kB\n", key, &value) == 2) {
    if (std::strcmp(key, "MemAvailable:") == 0) {
      out = static_cast<double>(value) / (1024.0 * 1024.0);
      break;
    }
  }
  std::fclose(f);
  return out;
}

std::string pci_address(const sycl::device& dev) {
#ifdef SYCL_EXT_INTEL_DEVICE_INFO
  try {
    if (dev.has(sycl::aspect::ext_intel_pci_address)) {
      return dev.get_info<sycl::ext::intel::info::device::pci_address>();
    }
  } catch (const sycl::exception&) {
  }
#endif
  return "unknown";
}

double free_device_gib(const sycl::device& dev) {
  try {
    if (dev.has(sycl::aspect::ext_intel_free_memory)) {
      return static_cast<double>(
                 dev.get_info<sycl::ext::intel::info::device::free_memory>()) /
             (1024.0 * 1024.0 * 1024.0);
    }
  } catch (const sycl::exception&) {
  }
  return -1.0;
}

struct Card {
  sycl::device device;
  std::string bdf;
  std::string name;
};

std::vector<Card> discover() {
  std::vector<Card> cards;
  for (const auto& platform : sycl::platform::get_platforms()) {
    for (const auto& dev : platform.get_devices(sycl::info::device_type::gpu)) {
      const std::string name = dev.get_info<sycl::info::device::name>();
      if (name.find("Arc") == std::string::npos &&
          name.find("Intel") == std::string::npos) {
        continue;                       // skip non-Intel GPUs (the 5090)
      }
      // One physical card appears once per SYCL platform (Level Zero and
      // OpenCL both enumerate it). Keying on the PCI address keeps one queue
      // per card, which is what "concurrent" has to mean for a link test.
      const std::string bdf = pci_address(dev);
      const bool seen = std::any_of(
          cards.begin(), cards.end(),
          [&bdf](const Card& c) { return c.bdf == bdf && bdf != "unknown"; });
      if (seen) {
        continue;
      }
      cards.push_back({dev, bdf, name});
    }
  }
  return cards;
}
// One direction, median of `iters`, on pinned host USM so the measurement is
// of the link and not of pageable-copy bookkeeping.
double bandwidth_gib_s(sycl::queue& q, void* host, void* device, std::size_t bytes,
                       int iters, bool to_device) {
  std::vector<double> rates;
  rates.reserve(static_cast<std::size_t>(iters));
  for (int i = 0; i < iters; ++i) {
    const auto start = clock_t_::now();
    if (to_device) {
      q.memcpy(device, host, bytes).wait();
    } else {
      q.memcpy(host, device, bytes).wait();
    }
    const double elapsed = seconds_since(start);
    if (elapsed > 0.0) {
      rates.push_back(static_cast<double>(bytes) / elapsed / 1e9);
    }
  }
  if (rates.empty()) {
    return 0.0;
  }
  std::sort(rates.begin(), rates.end());
  return rates[rates.size() / 2];
}

void report_bandwidth(Card& card, std::size_t bw_mib, int iters) {
  sycl::queue q{card.device, sycl::property::queue::in_order()};
  const std::size_t bytes = bw_mib * 1024ULL * 1024ULL;
  void* host = sycl::malloc_host(bytes, q);
  void* dev = sycl::malloc_device(bytes, q);
  if (host == nullptr || dev == nullptr) {
    std::printf("    allocation failed for %zu MiB\n", bw_mib);
    sycl::free(host, q);
    sycl::free(dev, q);
    return;
  }
  std::memset(host, 0xA5, bytes);
  q.memcpy(dev, host, bytes).wait();     // warm the path

  const double h2d = bandwidth_gib_s(q, host, dev, bytes, iters, true);
  const double d2h = bandwidth_gib_s(q, host, dev, bytes, iters, false);
  std::printf("    %5zu MiB : H2D %6.3f GB/s   D2H %6.3f GB/s\n", bw_mib, h2d,
              d2h);
  sycl::free(host, q);
  sycl::free(dev, q);
}

// Allocate device memory in steps, touching every byte, and watch host
// MemAvailable. A 1:1 slope means device residency is charged to host RAM.
void report_allocation_ladder(Card& card, double alloc_gib, double step_gib) {
  sycl::queue q{card.device, sycl::property::queue::in_order()};
  const double base_host = mem_available_gib();
  const double base_dev = free_device_gib(card.device);
  std::printf("    baseline: host MemAvailable %.2f GiB, device free %.2f GiB\n",
              base_host, base_dev);

  std::vector<void*> blocks;
  const std::size_t step_bytes =
      static_cast<std::size_t>(step_gib * 1024.0 * 1024.0 * 1024.0);
  double allocated = 0.0;
  while (allocated + step_gib <= alloc_gib + 1e-9) {
    void* p = sycl::malloc_device(step_bytes, q);
    if (p == nullptr) {
      std::printf("    device allocation refused at %.1f GiB\n", allocated);
      break;
    }
    q.memset(p, 0x5A, step_bytes).wait();   // force real backing
    blocks.push_back(p);
    allocated += step_gib;
    const double host_now = mem_available_gib();
    std::printf("    device +%5.1f GiB -> host MemAvailable %6.2f GiB "
                "(host delta %+6.2f)   device free %6.2f GiB\n",
                allocated, host_now, host_now - base_host,
                free_device_gib(card.device));
  }
  for (void* p : blocks) {
    sycl::free(p, q);
  }
  const double after = mem_available_gib();
  std::printf("    after free: host MemAvailable %.2f GiB (delta vs baseline "
              "%+.2f)\n", after, after - base_host);
}

// Both cards copying at once: the aggregate the shared uplink actually
// sustains, which is what a two-lane dispatch contends for.
void report_concurrent(std::vector<Card>& cards, std::size_t bw_mib, int iters) {
  if (cards.size() < 2) {
    return;
  }
  std::vector<sycl::queue> queues;
  std::vector<void*> hosts, devs;
  const std::size_t bytes = bw_mib * 1024ULL * 1024ULL;
  for (auto& card : cards) {
    queues.emplace_back(card.device, sycl::property::queue::in_order());
    void* h = sycl::malloc_host(bytes, queues.back());
    void* d = sycl::malloc_device(bytes, queues.back());
    std::memset(h, 0x33, bytes);
    hosts.push_back(h);
    devs.push_back(d);
  }
  std::vector<double> aggregate;
  for (int i = 0; i < iters; ++i) {
    const auto start = clock_t_::now();
    for (std::size_t c = 0; c < queues.size(); ++c) {
      queues[c].memcpy(devs[c], hosts[c], bytes);
    }
    for (auto& q : queues) {
      q.wait();
    }
    const double elapsed = seconds_since(start);
    if (elapsed > 0.0) {
      aggregate.push_back(static_cast<double>(bytes * queues.size()) / elapsed /
                          1e9);
    }
  }
  std::sort(aggregate.begin(), aggregate.end());
  std::printf("  concurrent H2D both cards, %zu MiB each: aggregate %.3f GB/s\n",
              bw_mib, aggregate.empty() ? 0.0 : aggregate[aggregate.size() / 2]);
  for (std::size_t c = 0; c < queues.size(); ++c) {
    sycl::free(hosts[c], queues[c]);
    sycl::free(devs[c], queues[c]);
  }
}

}  // namespace

int main(int argc, char** argv) {
  double alloc_gib = 0.0;
  double step_gib = 4.0;
  std::size_t bw_mib = 256;
  int iters = 7;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() { return (i + 1 < argc) ? argv[++i] : nullptr; };
    if (arg == "--alloc-gib") {
      if (const char* v = next()) alloc_gib = std::atof(v);
    } else if (arg == "--step-gib") {
      if (const char* v = next()) step_gib = std::atof(v);
    } else if (arg == "--bw-mib") {
      if (const char* v = next()) bw_mib = std::strtoull(v, nullptr, 10);
    } else if (arg == "--iters") {
      if (const char* v = next()) iters = std::atoi(v);
    } else {
      std::printf("usage: %s [--alloc-gib N] [--step-gib N] [--bw-mib N] "
                  "[--iters N]\n", argv[0]);
      return 2;
    }
  }

  std::vector<Card> cards = discover();
  if (cards.empty()) {
    std::printf("no Intel GPU found\n");
    return 1;
  }
  std::printf("host MemAvailable at start: %.2f GiB\n\n", mem_available_gib());
  for (std::size_t i = 0; i < cards.size(); ++i) {
    std::printf("card %zu: %s  bdf=%s  device_free=%.2f GiB\n", i,
                cards[i].name.c_str(), cards[i].bdf.c_str(),
                free_device_gib(cards[i].device));
  }

  for (std::size_t i = 0; i < cards.size(); ++i) {
    std::printf("\n== card %zu (%s) transport\n", i, cards[i].bdf.c_str());
    for (std::size_t mib : {std::size_t{1}, std::size_t{16}, bw_mib}) {
      report_bandwidth(cards[i], mib, iters);
    }
  }

  std::printf("\n== concurrent transport\n");
  report_concurrent(cards, bw_mib, iters);

  if (alloc_gib > 0.0) {
    for (std::size_t i = 0; i < cards.size(); ++i) {
      std::printf("\n== card %zu (%s) residency cost ladder\n", i,
                  cards[i].bdf.c_str());
      report_allocation_ladder(cards[i], alloc_gib, step_gib);
    }
  }
  std::printf("\nhost MemAvailable at end: %.2f GiB\n", mem_available_gib());
  return 0;
}
