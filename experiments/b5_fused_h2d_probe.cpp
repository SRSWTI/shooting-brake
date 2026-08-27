/*
 * b5_fused_h2d_probe.cpp — Kill-bench #5, measured arm.
 *
 * Bench 5 hypothesis was [estimate]: fusing the doorbell's three H2D copies
 * into one contiguous staged record saves 3-6 us/dispatch. Bench 13 step 3
 * measured enqueue cost at ~2.4 us PER OPERATION, which predicts ~4.8 us for
 * dropping two of the three. This probe measures it directly instead.
 *
 * Production, src/phase1/b70_provider.cpp:1232-1237 (M=1 decode):
 *   queue->memcpy(impl_->hidden,  hidden,  M*3072*sizeof(half)); // 6144 B
 *   queue->memcpy(impl_->ids,     ids,     M*8*sizeof(int32));   //   32 B
 *   queue->memcpy(impl_->weights, weights, M*8*sizeof(float));   //   32 B
 *
 * Two of the three transfers are 32 BYTES and still pay a full enqueue.
 * The bytes are irrelevant; the submission is the cost.
 *
 * Arms (identical device work, identical bytes moved):
 *   A. today  : 3 separate H2D enqueues + kernel + D2H
 *   B. fused  : 1 contiguous H2D enqueue + kernel + D2H
 *
 * Reports MEAN as the headline: a decode step sums 48 dispatches, so the
 * mean is the statistic that maps to ITL. Medians hide the sleep tail
 * (Bench 13 step 3: mean ran 58% above median on the blocking wait).
 *
 * Defaults to device index 1 = Gen4 B70 (0000:15:00.0), the production card.
 *
 * Build:
 *   /opt/intel/oneapi/2026.1/bin/icpx -O3 -fsycl \
 *       experiments/b5_fused_h2d_probe.cpp -o experiments/b5_fused_h2d_probe
 */
#include <sycl/sycl.hpp>

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

// Production decode geometry (88B / 122B share it): hidden 3072, top_k 8.
constexpr size_t kHidden = 3072;
constexpr size_t kTopK = 8;
constexpr size_t kHiddenBytes = kHidden * sizeof(uint16_t);  // half  -> 6144
constexpr size_t kIdsBytes = kTopK * sizeof(int32_t);        //       ->   32
constexpr size_t kWeightsBytes = kTopK * sizeof(float);      //       ->   32
constexpr size_t kFusedBytes = kHiddenBytes + kIdsBytes + kWeightsBytes;
constexpr size_t kOutBytes = kHidden * sizeof(float);        //       -> 12288

double now_sec() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return double(ts.tv_sec) + double(ts.tv_nsec) * 1e-9;
}

struct Stats { double mean=0,min=0,p50=0,p90=0,p99=0,max=0; size_t n=0; };

Stats summarize(std::vector<double> v) {
  if (v.empty()) return {};
  double s = 0; for (double x : v) s += x;
  std::sort(v.begin(), v.end());
  auto pick=[&](double q){ return v[size_t(q*double(v.size()-1)+0.5)]; };
  return Stats{s/double(v.size()), v.front(), pick(0.50), pick(0.90),
               pick(0.99), v.back(), v.size()};
}

void row(const char* label, const Stats& s) {
  std::printf("  %-30s MEAN %7.3f | p50 %7.3f  p90 %7.3f  p99 %7.3f  max %8.3f\n",
              label, s.mean, s.p50, s.p90, s.p99, s.max);
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  const int iters  = (argc > 1) ? std::atoi(argv[1]) : 5000;
  const int devidx = (argc > 2) ? std::atoi(argv[2]) : 1;
  const int warmup = 500;

  std::vector<sycl::device> gpus;
  for (const auto& p : sycl::platform::get_platforms()) {
    if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
    for (const auto& d : p.get_devices()) if (d.is_gpu()) gpus.push_back(d);
  }
  if (gpus.empty() || devidx >= int(gpus.size())) {
    std::fprintf(stderr, "no such L0 GPU (found %zu)\n", gpus.size());
    return 1;
  }
  sycl::queue q(gpus[devidx], sycl::property::queue::in_order{});

  std::printf("== Bench 5: fused vs split H2D at production decode shape ==\n");
  std::printf("device: %s  (index %d — %s)\n",
              gpus[devidx].get_info<sycl::info::device::name>().c_str(), devidx,
              devidx == 1 ? "Gen4, production card" : "Gen3, NOT production");
  std::printf("payload: hidden %zu B + ids %zu B + weights %zu B = %zu B; out %zu B\n\n",
              kHiddenBytes, kIdsBytes, kWeightsBytes, kFusedBytes, kOutBytes);

  // Host staging. Arm A mirrors production: three separate pinned buffers.
  // Arm B: one contiguous record, the three fields laid end to end.
  auto* h_hidden  = sycl::malloc_host<uint8_t>(kHiddenBytes, q);
  auto* h_ids     = sycl::malloc_host<uint8_t>(kIdsBytes, q);
  auto* h_weights = sycl::malloc_host<uint8_t>(kWeightsBytes, q);
  auto* h_fused   = sycl::malloc_host<uint8_t>(kFusedBytes, q);
  auto* h_out     = sycl::malloc_host<uint8_t>(kOutBytes, q);

  // Device side. Arm B's three fields are offsets into ONE allocation, which
  // is what makes a single memcpy legal.
  auto* d_hidden  = sycl::malloc_device<uint8_t>(kHiddenBytes, q);
  auto* d_ids     = sycl::malloc_device<uint8_t>(kIdsBytes, q);
  auto* d_weights = sycl::malloc_device<uint8_t>(kWeightsBytes, q);
  auto* d_fused   = sycl::malloc_device<uint8_t>(kFusedBytes, q);
  auto* d_out     = sycl::malloc_device<uint8_t>(kOutBytes, q);

  if (!h_hidden || !h_ids || !h_weights || !h_fused || !h_out ||
      !d_hidden || !d_ids || !d_weights || !d_fused || !d_out) {
    std::fprintf(stderr, "allocation failed\n");
    return 1;
  }
  for (size_t i = 0; i < kFusedBytes; ++i) h_fused[i] = uint8_t(i);
  for (size_t i = 0; i < kHiddenBytes; ++i) h_hidden[i] = uint8_t(i);

  // Same device work in both arms so only the submission count differs.
  auto kernel_and_out = [&](uint8_t* src) {
    q.parallel_for(sycl::range<1>(256), [=](sycl::id<1> i) {
      d_out[i[0]] = uint8_t(src[i[0]] + 1);
    });
    return q.memcpy(h_out, d_out, kOutBytes);
  };

  std::vector<double> a_sub, a_tot, b_sub, b_tot;

  for (int i = 0; i < warmup + iters; ++i) {
    // ---- arm A: three separate H2D enqueues (production today) ----
    {
      const double t0 = now_sec();
      q.memcpy(d_hidden, h_hidden, kHiddenBytes);
      q.memcpy(d_ids, h_ids, kIdsBytes);
      q.memcpy(d_weights, h_weights, kWeightsBytes);
      auto ev = kernel_and_out(d_hidden);
      const double t1 = now_sec();
      while (ev.get_info<sycl::info::event::command_execution_status>() !=
             sycl::info::event_command_status::complete) SB_SPIN_HINT();
      const double t2 = now_sec();
      if (i >= warmup) { a_sub.push_back((t1-t0)*1e6); a_tot.push_back((t2-t0)*1e6); }
    }
    // ---- arm B: one fused H2D enqueue ----
    {
      const double t0 = now_sec();
      q.memcpy(d_fused, h_fused, kFusedBytes);
      auto ev = kernel_and_out(d_fused);
      const double t1 = now_sec();
      while (ev.get_info<sycl::info::event::command_execution_status>() !=
             sycl::info::event_command_status::complete) SB_SPIN_HINT();
      const double t2 = now_sec();
      if (i >= warmup) { b_sub.push_back((t1-t0)*1e6); b_tot.push_back((t2-t0)*1e6); }
    }
  }

  const Stats AS = summarize(a_sub), AT = summarize(a_tot);
  const Stats BS = summarize(b_sub), BT = summarize(b_tot);

  std::printf("-- submission cost, host side (us) --\n");
  row("A. 3 H2D + kernel + D2H", AS);
  row("B. 1 H2D + kernel + D2H", BS);
  std::printf("\n-- full dispatch, submit + spin-wait (us) --\n");
  row("A. split H2D", AT);
  row("B. fused H2D", BT);

  const double saved_sub = AS.mean - BS.mean;
  const double saved_tot = AT.mean - BT.mean;
  std::printf("\n== verdict ==\n");
  std::printf("  submission saved : %.3f us/dispatch\n", saved_sub);
  std::printf("  end-to-end saved : %.3f us/dispatch\n", saved_tot);
  std::printf("  per enqueue      : %.3f us  (2 enqueues removed)\n", saved_sub / 2.0);
  std::printf("  at 48 dispatches/step: %.3f ms/step = %.2f%% of a 12.35 ms step\n",
              saved_tot * 48.0 / 1000.0, saved_tot * 48.0 / 1000.0 / 12.35 * 100.0);
  std::printf("  Bench 5 kill condition was < 2 us/dispatch: %s\n",
              saved_tot < 2.0 ? "HIT -> KILLED" : "not hit -> SURVIVES");

  sycl::free(h_hidden,q); sycl::free(h_ids,q); sycl::free(h_weights,q);
  sycl::free(h_fused,q); sycl::free(h_out,q);
  sycl::free(d_hidden,q); sycl::free(d_ids,q); sycl::free(d_weights,q);
  sycl::free(d_fused,q); sycl::free(d_out,q);
  return 0;
}
