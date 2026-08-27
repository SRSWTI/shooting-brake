/*
 * b70_hostflag_spin_probe.cpp — Kill-bench #4: Xe2 device-side host-flag spin.
 *
 * Question: can a resident B70 kernel spin on a USM-host word and observe a
 * host write in < 20 us, with forward progress intact, while the device is
 * otherwise busy?
 *
 * This is the prerequisite for removing the host poller from the doorbell
 * loop (the NIC-interrupt model). If it holds, the persistent-kernel doorbell
 * is on the table and Bench 8's BAR work becomes a pool-A lever (2.99 ms/step,
 * ~25%). If it does not, Bench 8 collapses to a ~2% transport lever.
 *
 * Instrument: host/device ping-pong on two USM-host words. Host writes seq to
 * `to_dev`; the resident kernel spins until it sees it and writes seq to
 * `to_host`; host spins until it sees that. One host clock, no cross-clock
 * correlation needed. Reported RTT is the honest end-to-end doorbell number;
 * one-way visibility is ~RTT/2 (the device->host leg is a posted write to
 * host DRAM, the cheaper of the two directions).
 *
 * COHERENCE NOTE (learned the hard way, first run): a plain
 * atomic_ref::load() on USM-host memory compiles to a cacheable load. The
 * device latches the flag line on its first read and never observes another
 * host store -- seq 1 completes, seq 2 hangs forever. Both directions here
 * therefore use read-modify-write atomics (fetch_add(0) / exchange), which
 * are emitted as real atomic instructions and go to the coherent point. That
 * is the load-bearing detail for any device-side doorbell on this stack.
 *
 * Arms:
 *   1. idle device
 *   2. contended -- responder goes resident first, then a saturating compute
 *      kernel lands on a second queue. Tests starvation and forward progress,
 *      which is the arm that actually decides the persistent-kernel design.
 *
 * SAFETY: three independent escapes, because a wedged xe means
 * `modprobe -r xe && modprobe xe`.
 *   - host writes an ABORT sentinel the responder checks every iteration
 *   - device-side spin guard as a backstop if the host dies outright
 *   - host wall-clock timeout per round trip
 *
 * Build:
 *   /opt/intel/oneapi/2026.1/bin/icpx -O3 -fsycl \
 *       experiments/b70_hostflag_spin_probe.cpp -lze_loader \
 *       -o experiments/b70_hostflag_spin_probe
 * Run:
 *   LD_LIBRARY_PATH=/opt/intel/oneapi/2026.1/lib ./b70_hostflag_spin_probe [iters] [dev]
 */
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <sycl/ext/intel/experimental/cache_control_properties.hpp>
#include <immintrin.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>

namespace {

// Sentinel telling a resident responder to exit immediately.
constexpr uint32_t kAbort = 0xFFFFFFFFu;

// Backstop only -- the host ABORT sentinel is the primary escape. Sized
// generously because we do NOT yet know the device spin rate; the probe now
// measures it (max spin iters per wait) instead of guessing. The first draft
// guessed 200M assuming EU-local speed and cost one 120 s hang; the second
// guessed 4M and tripped on a routine host scheduling stall.
constexpr uint64_t kDeviceSpinGuard = 100'000'000ULL;

// Device-reported exit reason.
constexpr uint32_t kStatusDone = 1u;
constexpr uint32_t kStatusAbort = 2u;
constexpr uint32_t kStatusGuard = 3u;

// Host ceiling on one round trip before declaring the responder dead.
constexpr double kHostTimeoutSec = 2.0;

using AtomRef = sycl::atomic_ref<uint32_t, sycl::memory_order::acq_rel,
                                 sycl::memory_scope::system,
                                 sycl::access::address_space::global_space>;

// An uncached read port. Device loads through this pointer bypass L1/L2/L3,
// so the resident kernel keeps observing host stores indefinitely instead of
// freezing once the GPU cache takes ownership of the flag line.
namespace iexp = sycl::ext::intel::experimental;
namespace oexp = sycl::ext::oneapi::experimental;
using UncachedProps = decltype(oexp::properties{
    iexp::read_hint<iexp::cache_control<iexp::cache_mode::uncached,
                                        iexp::cache_level::L1,
                                        iexp::cache_level::L2,
                                        iexp::cache_level::L3>>});
using UncachedFlag = oexp::annotated_ptr<uint32_t, UncachedProps>;

double now_sec() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return double(ts.tv_sec) + double(ts.tv_nsec) * 1e-9;
}

struct Stats {
  double min = 0, p50 = 0, p90 = 0, p99 = 0, max = 0;
  size_t n = 0;
};

Stats summarize(std::vector<double> us) {
  if (us.empty()) return {};
  std::sort(us.begin(), us.end());
  auto pick = [&](double q) { return us[size_t(q * double(us.size() - 1) + 0.5)]; };
  return Stats{us.front(), pick(0.50), pick(0.90), pick(0.99), us.back(), us.size()};
}

void print_stats(const char* label, const Stats& s) {
  std::printf("  %-22s n=%-6zu min %7.2f  p50 %7.2f  p90 %7.2f  p99 %7.2f  max %9.2f   (us)\n",
              label, s.n, s.min, s.p50, s.p90, s.p99, s.max);
}

void queue_group_recon(const sycl::device& dev) {
  auto zdev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);
  uint32_t count = 0;
  if (zeDeviceGetCommandQueueGroupProperties(zdev, &count, nullptr) != ZE_RESULT_SUCCESS) {
    std::printf("  (zeDeviceGetCommandQueueGroupProperties failed)\n");
    return;
  }
  std::vector<ze_command_queue_group_properties_t> props(count);
  for (auto& p : props) {
    p = ze_command_queue_group_properties_t{};
    p.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
  }
  zeDeviceGetCommandQueueGroupProperties(zdev, &count, props.data());

  for (uint32_t i = 0; i < count; ++i) {
    const auto f = props[i].flags;
    std::printf("  queue group %u: numQueues=%-3u  %s%s%s%s\n", i, props[i].numQueues,
                (f & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) ? "COMPUTE " : "",
                (f & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) ? "COPY " : "",
                (f & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COOPERATIVE_KERNELS)
                    ? "COOPERATIVE_KERNELS " : "",
                (f & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_METRICS) ? "METRICS " : "");
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);

  const uint32_t iters = (argc > 1) ? uint32_t(std::atoi(argv[1])) : 2000;
  const int dev_index = (argc > 2) ? std::atoi(argv[2]) : 0;
  const uint32_t warmup = 200;

  std::vector<sycl::device> gpus;
  for (const auto& p : sycl::platform::get_platforms()) {
    if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
    for (const auto& d : p.get_devices())
      if (d.is_gpu()) gpus.push_back(d);
  }
  if (gpus.empty()) {
    std::fprintf(stderr, "no Level Zero GPU found\n");
    return 1;
  }
  if (dev_index >= int(gpus.size())) {
    std::fprintf(stderr, "device index %d out of range (%zu found)\n", dev_index, gpus.size());
    return 1;
  }
  sycl::device dev = gpus[dev_index];

  std::printf("== Bench 4: Xe2 device-side host-flag spin probe ==\n");
  std::printf("device: %s   (%zu L0 GPU(s), index %d)\n",
              dev.get_info<sycl::info::device::name>().c_str(), gpus.size(), dev_index);
  std::printf("  EUs=%u  max_wg=%zu  usm_host=%s\n",
              dev.get_info<sycl::info::device::max_compute_units>(),
              dev.get_info<sycl::info::device::max_work_group_size>(),
              dev.has(sycl::aspect::usm_host_allocations) ? "yes" : "NO");

  bool sys_scope = false;
  for (auto s : dev.get_info<sycl::info::device::atomic_memory_scope_capabilities>())
    if (s == sycl::memory_scope::system) sys_scope = true;
  std::printf("  atomic system scope: %s\n", sys_scope ? "yes" : "NO");
  queue_group_recon(dev);

  if (!dev.has(sycl::aspect::usm_host_allocations)) {
    std::printf("\nKILL: no USM host allocations; device cannot see host memory.\n");
    return 2;
  }

  sycl::queue spin_q(dev, sycl::property::queue::in_order{});

  // Each flag gets its OWN cache line. With to_dev and to_host adjacent in
  // one line, the device's atomic write to to_host takes ownership of the
  // line that also holds to_dev and writes back its stale copy, silently
  // eating the host's next request. That false sharing froze the responder
  // after O(100) round trips regardless of how the poll was expressed --
  // atomic load, fetch_add, compare_exchange, and uncached load all died the
  // same way, because the bug was on the WRITE side all along.
  constexpr size_t kLine = 256;  // >= any plausible line size on either side
  uint8_t* flags = sycl::malloc_host<uint8_t>(kLine * 4, spin_q);
  if (!flags) {
    std::fprintf(stderr, "malloc_host failed\n");
    return 1;
  }
  std::memset(flags, 0, kLine * 4);
  uint32_t* to_dev = reinterpret_cast<uint32_t*>(flags + 0 * kLine);
  uint32_t* to_host = reinterpret_cast<uint32_t*>(flags + 1 * kLine);
  uint32_t* status = reinterpret_cast<uint32_t*>(flags + 2 * kLine);
  uint32_t* maxguard = reinterpret_cast<uint32_t*>(flags + 3 * kLine);

  // One resident kernel serves every round trip of an arm, seq 1..total.
  //
  // Poll discipline (see COHERENCE NOTE): the device must READ to_dev without
  // writing it. A plain load caches and never refreshes; fetch_add(0) reads
  // coherently but is a read-MODIFY-WRITE, and its write-back races the host's
  // next store -- the device rewrites the stale seq, the host's write is lost,
  // and both sides wait forever (observed: random stalls at seq 193 / 386).
  // compare_exchange performs NO write on mismatch, so it polls coherently
  // under strict single-writer discipline, and returns the observed value.
  double arm_t0 = 0.0;
  auto launch_responder = [&](uint32_t total) {
    arm_t0 = now_sec();
    return spin_q.single_task([=]() {
      // Uncached read port on the request word. A 32-bit aligned load is
      // single-copy atomic on this hardware, so no atomic op is needed to
      // poll -- and an uncached load is the ONLY thing that keeps observing
      // host stores once L3 has taken the line.
      UncachedFlag req{to_dev};
      uint32_t reason = kStatusDone;
      uint64_t worst = 0;
      for (uint32_t seq = 1; seq <= total; ++seq) {
        uint64_t guard = 0;
        bool stop = false;
        for (;;) {
          const uint32_t observed = *req;
          if (observed == seq) break;
          if (observed == kAbort) { reason = kStatusAbort; stop = true; break; }
          if (++guard > kDeviceSpinGuard) { reason = kStatusGuard; stop = true; break; }
        }
        if (guard > worst) worst = guard;
        if (stop) break;
        AtomRef out(*to_host);
        out.exchange(seq);
      }
      AtomRef mg(*maxguard);
      mg.exchange(uint32_t(worst > 0xFFFFFFFFULL ? 0xFFFFFFFFULL : worst));
      AtomRef st(*status);
      st.exchange(reason);
    });
  };

  // Publish a host store all the way to memory. The B70's reads appear to be
  // issued No-Snoop: they do not probe the CPU caches, so a store sitting
  // Modified in L1/L2 is invisible to the device until the line happens to be
  // evicted naturally. That is exactly the observed failure -- works for a
  // random number of round trips, then freezes forever. clflush + sfence make
  // publication deterministic instead of luck.
  auto publish = [&](uint32_t* p, uint32_t v) {
    __atomic_store_n(p, v, __ATOMIC_RELEASE);
    _mm_clflush(p);
    _mm_sfence();
  };

  auto abort_responder = [&]() { publish(to_dev, kAbort); };

  // Drives sequences [from, to] inclusive. Numbering is continuous across
  // warmup and timed phases -- the responder counts monotonically.
  auto pingpong = [&](uint32_t from, uint32_t to, std::vector<double>* rec) -> bool {
    for (uint32_t seq = from; seq <= to; ++seq) {
      const double t0 = now_sec();
      publish(to_dev, seq);
      while (__atomic_load_n(to_host, __ATOMIC_ACQUIRE) != seq) {
        if (now_sec() - t0 > kHostTimeoutSec) {
          std::printf("  !! responder stopped at seq %u after %.3f ms resident\n",
                      seq, (now_sec() - arm_t0) * 1e3);
          abort_responder();
          return false;
        }
#if defined(__x86_64__)
        __builtin_ia32_pause();
#endif
      }
      if (rec) rec->push_back((now_sec() - t0) * 1e6);
    }
    return true;
  };

  auto reset_flags = [&]() {
    __atomic_store_n(to_host, 0u, __ATOMIC_RELEASE);
    __atomic_store_n(status, 0u, __ATOMIC_RELEASE);
    __atomic_store_n(maxguard, 0u, __ATOMIC_RELEASE);
    __atomic_store_n(to_dev, 0u, __ATOMIC_RELEASE);
  };

  auto report_exit = [&](const char* arm) {
    const uint32_t s = __atomic_load_n(status, __ATOMIC_ACQUIRE);
    const uint32_t g = __atomic_load_n(maxguard, __ATOMIC_ACQUIRE);
    const char* why = (s == kStatusDone)    ? "completed all sequences"
                      : (s == kStatusAbort) ? "host ABORT sentinel"
                      : (s == kStatusGuard) ? "DEVICE SPIN GUARD tripped"
                                            : "kernel never reported status";
    std::printf("  %s responder exit: %s  (peak spin iters in one wait: %u)\n", arm, why, g);
  };

  const uint32_t total = warmup + iters;

  // ---------------- arm 1: idle device ----------------
  std::printf("\n-- arm 1: idle device (%u timed RTTs, %u warmup) --\n", iters, warmup);
  reset_flags();
  auto ev1 = launch_responder(total);

  std::vector<double> warm_us, idle_us;
  bool ok1 = pingpong(1, warmup, &warm_us);
  if (ok1) ok1 = pingpong(warmup + 1, total, &idle_us);
  ev1.wait();
  report_exit("arm 1");

  if (!warm_us.empty()) print_stats("warmup RTT", summarize(warm_us));
  Stats idle = summarize(idle_us);
  if (!idle_us.empty()) print_stats("idle RTT", idle);

  // ---------------- arm 2: contended ----------------
  std::printf("\n-- arm 2: contended (responder resident, then saturating kernel) --\n");
  reset_flags();
  auto ev2 = launch_responder(total);

  bool ok2 = pingpong(1, warmup, nullptr);  // let the responder go resident

  // Created only now, so arm 1 runs with the responder as the ONLY context on
  // the engine. If the freeze is GuC timeslicing between contexts
  // (timeslice_duration_us = 1000), arm 1 should now survive indefinitely.
  sycl::queue busy_q(dev, sycl::property::queue::in_order{});
  const size_t busy_items = 1u << 20;
  float* scratch = sycl::malloc_device<float>(busy_items, busy_q);
  sycl::event ev_busy;
  bool busy_launched = false;
  if (scratch && ok2) {
    ev_busy = busy_q.parallel_for(sycl::range<1>(busy_items), [=](sycl::id<1> i) {
      float acc = float(i[0]) * 1e-6f;
      for (int k = 0; k < 4096; ++k) acc = sycl::fma(acc, 1.0000001f, 1e-7f);
      scratch[i[0]] = acc;
    });
    busy_launched = true;
  }

  std::vector<double> busy_us;
  if (ok2) ok2 = pingpong(warmup + 1, total, &busy_us);

  if (busy_launched) ev_busy.wait();
  if (scratch) sycl::free(scratch, busy_q);
  ev2.wait();
  report_exit("arm 2");

  Stats busy = summarize(busy_us);
  if (!busy_us.empty()) print_stats("contended RTT", busy);

  sycl::free(flags, spin_q);

  // ---------------- verdict ----------------
  std::printf("\n== verdict ==\n");
  if (!ok1) {
    std::printf("KILL: no coherent forward progress on an IDLE device.\n");
    return 3;
  }
  if (!ok2) {
    std::printf("KILL: responder starved under contention -- a resident doorbell\n"
                "      kernel cannot coexist with production compute on this stack.\n");
    return 4;
  }
  const double ow_idle = idle.p50 / 2.0;
  const double ow_busy = busy.p50 / 2.0;
  std::printf("  one-way visibility (p50 RTT/2):  idle %.2f us   contended %.2f us\n",
              ow_idle, ow_busy);
  std::printf("  tail (p99 RTT):                  idle %.2f us   contended %.2f us\n",
              idle.p99, busy.p99);
  const bool killed = ow_idle > 20.0 || ow_busy > 20.0;
  std::printf("  kill condition (>20 us one-way): %s\n", killed ? "HIT -> KILLED" : "not hit");
  if (!killed)
    std::printf("\n  Bench 4 survives. Persistent-kernel doorbell is viable; Bench 8's\n"
                "  BAR work is a pool-A lever, not a transport rounding error.\n");
  return killed ? 5 : 0;
}
