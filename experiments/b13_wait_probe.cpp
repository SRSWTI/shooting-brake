/*
 * b13_wait_probe.cpp — Kill-bench #13 step 3: blocking wait vs spin.
 *
 * Where we are:
 *   step 1: the doorbell MECHANISM costs 4.79 us; production spends ~39 us.
 *           cuStreamWaitValue32 eliminated (1.19 us).
 *   step 2: the poller's 48-flag sweep costs 0.11 us. Eliminated.
 *   => the ~34 us is inside issue()/take() on the host.
 *
 * The suspect, src/phase1/b70_provider.cpp:1396 (inside take()):
 *
 *     impl_->copy_out->wait_and_throw();
 *
 * SYCL's blocking wait. Intel's L0 runtime spins for a short window, then
 * sleeps the thread and waits for an interrupt. Interrupt wakeup on Linux is
 * tens of microseconds -- which is exactly the size of the unexplained gap.
 *
 * The poller is a DEDICATED native thread that already spins
 * (SB_SPIN_HINT in b70_capi.cpp). It has no reason to ever sleep. If a spin
 * wait is materially faster than the blocking wait at the production duty
 * cycle, the fix is a few lines in take().
 *
 * Arms, at the production doorbell shape (12 KiB H2D, kernel, 12 KiB D2H):
 *   A. blocking : event.wait()                      <- what production does
 *   B. spin     : poll SYCL event status
 *   C. spin-L0  : poll zeEventQueryStatus on the native handle (lowest cost)
 *
 * Defaults to device index 1 = the Gen4 B70 (0000:15:00.0), which is what
 * production serves on. Index 0 is the Gen3 card -- do not measure there.
 *
 * Build:
 *   /opt/intel/oneapi/2026.1/bin/icpx -O3 -fsycl \
 *       experiments/b13_wait_probe.cpp -lze_loader -o experiments/b13_wait_probe
 */
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>

#if defined(__x86_64__)
#include <immintrin.h>
#define SB_SPIN_HINT() _mm_pause()
#else
#define SB_SPIN_HINT() ((void)0)
#endif

namespace {

double now_sec() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return double(ts.tv_sec) + double(ts.tv_nsec) * 1e-9;
}

struct Stats { double mean=0,min=0,p50=0,p90=0,p99=0,max=0; size_t n=0; };

Stats summarize(std::vector<double> v) {
  if (v.empty()) return {};
  double sum = 0;
  for (double x : v) sum += x;
  std::sort(v.begin(), v.end());
  auto pick=[&](double q){ return v[size_t(q*double(v.size()-1)+0.5)]; };
  return Stats{sum/double(v.size()), v.front(), pick(0.50), pick(0.90),
               pick(0.99), v.back(), v.size()};
}

void print_stats(const char* label, const Stats& s) {
  // MEAN is the statistic that matters: a step sums 48 dispatches, so a fat
  // tail costs real time even when the median looks fine.
  std::printf("  %-34s n=%-5zu MEAN %7.2f | p50 %7.2f  p90 %7.2f  p99 %8.2f  max %9.2f\n",
              label, s.n, s.mean, s.p50, s.p90, s.p99, s.max);
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  const int iters  = (argc > 1) ? std::atoi(argv[1]) : 3000;
  const int devidx = (argc > 2) ? std::atoi(argv[2]) : 1;   // Gen4 default
  const int warmup = 300;

  std::vector<sycl::device> gpus;
  for (const auto& p : sycl::platform::get_platforms()) {
    if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
    for (const auto& d : p.get_devices()) if (d.is_gpu()) gpus.push_back(d);
  }
  if (gpus.empty() || devidx >= int(gpus.size())) {
    std::fprintf(stderr, "no such L0 GPU (found %zu)\n", gpus.size());
    return 1;
  }
  sycl::device dev = gpus[devidx];
  sycl::queue q(dev, sycl::property::queue::in_order{});

  std::printf("== Bench 13 step 3: blocking wait vs spin ==\n");
  std::printf("device: %s  (index %d — %s)\n\n",
              dev.get_info<sycl::info::device::name>().c_str(), devidx,
              devidx == 1 ? "Gen4, production card" : "Gen3, NOT production");

  // Production doorbell payload: 12 KiB hidden in, 12 KiB result out.
  const size_t kBytes = 12 * 1024;
  auto* h_in  = sycl::malloc_host<uint8_t>(kBytes, q);
  auto* h_out = sycl::malloc_host<uint8_t>(kBytes, q);
  auto* d_buf = sycl::malloc_device<uint8_t>(kBytes, q);
  if (!h_in || !h_out || !d_buf) { std::fprintf(stderr, "alloc failed\n"); return 1; }
  for (size_t i = 0; i < kBytes; ++i) h_in[i] = uint8_t(i);

  // One dispatch = H2D + a trivial kernel + D2H, matching the doorbell shape.
  auto submit = [&]() {
    q.memcpy(d_buf, h_in, kBytes);
    q.parallel_for(sycl::range<1>(256), [=](sycl::id<1> i) {
      d_buf[i[0]] = uint8_t(d_buf[i[0]] + 1);
    });
    return q.memcpy(h_out, d_buf, kBytes);
  };

  std::vector<double> block_us, spin_us, zespin_us, submit_us, total_us;

  // ---- arm A: blocking wait (what take() does today) ----
  for (int i = 0; i < warmup + iters; ++i) {
    auto ev = submit();
    const double t0 = now_sec();
    ev.wait();
    const double t1 = now_sec();
    if (i >= warmup) block_us.push_back((t1 - t0) * 1e6);
  }

  // ---- arm B: spin on the SYCL event status, and time SUBMISSION too ----
  // Production's service_ns spans issue()+take(), i.e. submission AND wait.
  // Timing only the wait would miss the enqueue cost entirely.
  for (int i = 0; i < warmup + iters; ++i) {
    const double ts0 = now_sec();
    auto ev = submit();
    const double ts1 = now_sec();
    while (ev.get_info<sycl::info::event::command_execution_status>() !=
           sycl::info::event_command_status::complete) {
      SB_SPIN_HINT();
    }
    const double t1 = now_sec();
    if (i >= warmup) {
      submit_us.push_back((ts1 - ts0) * 1e6);
      spin_us.push_back((t1 - ts1) * 1e6);
      total_us.push_back((t1 - ts0) * 1e6);
    }
  }

  // ---- arm C: spin on the native L0 event ----
  bool have_ze = false;
  for (int i = 0; i < warmup + iters; ++i) {
    auto ev = submit();
    ze_event_handle_t zev = nullptr;
    try {
      zev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ev);
      have_ze = true;
    } catch (...) { break; }
    const double t0 = now_sec();
    while (zeEventQueryStatus(zev) != ZE_RESULT_SUCCESS) SB_SPIN_HINT();
    const double t1 = now_sec();
    if (i >= warmup) zespin_us.push_back((t1 - t0) * 1e6);
    ev.wait();  // keep SYCL's bookkeeping consistent
  }

  std::printf("-- host time spent waiting for one dispatch (us) --\n");
  const Stats A = summarize(block_us), B = summarize(spin_us);
  print_stats("A. blocking  ev.wait()", A);
  print_stats("B. spin on SYCL event status", B);
  Stats C{};
  if (have_ze && !zespin_us.empty()) {
    C = summarize(zespin_us);
    print_stats("C. spin on zeEventQueryStatus", C);
  }

  std::printf("\n-- where the host time actually goes, per dispatch (us) --\n");
  const Stats S = summarize(submit_us), T = summarize(total_us);
  print_stats("D. SUBMIT (enqueue 3 ops)", S);
  print_stats("E. submit + spin-wait TOTAL", T);

  std::printf("\n== verdict ==\n");
  const double best_spin = (C.n && C.mean < B.mean) ? C.mean : B.mean;
  const double saved = A.mean - best_spin;
  std::printf("  blocking MEAN %.2f us  vs  best spin MEAN %.2f us  =>  %.2f us/dispatch\n",
              A.mean, best_spin, saved);
  std::printf("  unexplained gap from step 1: ~34 us/dispatch\n");
  std::printf("  at 48 dispatches/step: %.2f ms/step  (%.1f%% of a 12.35 ms step)\n",
              saved * 48.0 / 1000.0, saved * 48.0 / 1000.0 / 12.35 * 100.0);
  if (saved > 20.0)
    std::printf("\n  => BLOCKING WAIT IS THE BUG. Replace wait_and_throw() in take()\n"
                "     with a spin; the poller thread is dedicated and already spins.\n");
  else if (saved > 5.0)
    std::printf("\n  => blocking costs real time but not all 34 us. Worth fixing,\n"
                "     keep looking for the remainder.\n");
  else
    std::printf("\n  => blocking is NOT the bug. The cost is elsewhere in issue()/take()\n"
                "     — suspect the H2D submission path or per-dispatch allocation.\n");

  sycl::free(h_in, q); sycl::free(h_out, q); sycl::free(d_buf, q);
  return 0;
}
