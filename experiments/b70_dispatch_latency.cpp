// Decode-dispatch latency decomposition for the B70, with no model involved.
//
// Measured on this box (hybrid-subset-16-8.json + perf/harness/xpu_bench.cpp):
//
//   B70 MoE kernel, M=1, top_k 8 ......  52.8 us
//   Dispatch service time ............. 186.1 us
//   ------------------------------------------
//   Overhead (launch/signal/copy) ..... 133.3 us   <- 72%, never measured
//
// Everything here targets that 133 us. A decode dispatch is tiny -- 4 KiB of
// activations out, 8 KiB of fp32 partials back -- so it is pure latency, and
// latency is reproducible without a single model weight.
//
// Modes:
//   launch      empty kernel submit + wait. Pure launch cost.
//   copy        H2D/D2H round trip across payload sizes. Latency floor vs bandwidth.
//   dispatch    H2D + trivial kernel + D2H + wait. The current per-layer shape.
//   persistent  one long-lived kernel spinning on a host doorbell. The proposal.
//   queues      N concurrent dispatches on N queues. Multi-queue headroom.
//
// The persistent kernel ALWAYS bounds its spin so it cannot wedge the device.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

namespace {

using Clock = std::chrono::steady_clock;

double us_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}

struct Stats {
  double median, p10, p90, min;
};

Stats summarize(std::vector<double> &v) {
  std::sort(v.begin(), v.end());
  auto at = [&](double q) { return v[std::min(v.size() - 1, static_cast<std::size_t>(q * v.size()))]; };
  return Stats{at(0.50), at(0.10), at(0.90), v.front()};
}

void report(const char *label, Stats s, const char *unit = "us") {
  std::printf("  %-34s median %8.2f %s   p10 %8.2f   p90 %8.2f   min %8.2f\n",
              label, s.median, unit, s.p10, s.p90, s.min);
}

// Decode shape: hidden 2048 f16 out, 2048 fp32 partials back, per layer.
constexpr std::size_t kActBytes = 2048 * sizeof(sycl::half);
constexpr std::size_t kOutBytes = 2048 * sizeof(float);

// ---------------------------------------------------------------- launch ---

void bench_launch(sycl::queue &q, int iters) {
  std::printf("\n[launch] empty kernel submit + wait\n");
  for (int i = 0; i < 200; ++i) q.single_task([]() {}).wait();

  std::vector<double> submit_wait, submit_only;
  submit_wait.reserve(iters);
  submit_only.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    auto t0 = Clock::now();
    sycl::event e = q.single_task([]() {});
    submit_only.push_back(us_since(t0));
    e.wait();
    submit_wait.push_back(us_since(t0));
  }
  report("submit only (no wait)", summarize(submit_only));
  report("submit + wait", summarize(submit_wait));
}

// ------------------------------------------------------------------ copy ---

void bench_copy(sycl::queue &q, int iters) {
  std::printf("\n[copy] pinned host <-> device round trip by payload size\n");
  const std::size_t sizes[] = {1024, 4096, 12288, 65536, 262144, 1048576, 4194304};
  for (std::size_t bytes : sizes) {
    void *host = sycl::malloc_host(bytes, q);
    void *dev = sycl::malloc_device(bytes, q);
    std::memset(host, 0xA5, bytes);
    for (int i = 0; i < 50; ++i) { q.memcpy(dev, host, bytes).wait(); }

    std::vector<double> rt;
    rt.reserve(iters);
    for (int i = 0; i < iters; ++i) {
      auto t0 = Clock::now();
      q.memcpy(dev, host, bytes).wait();
      q.memcpy(host, dev, bytes).wait();
      rt.push_back(us_since(t0));
    }
    Stats s = summarize(rt);
    const double gbps = (2.0 * bytes) / (s.median * 1e-6) / 1e9;
    std::printf("  %8zu B  round trip median %8.2f us   min %8.2f   -> %6.2f GB/s\n",
                bytes, s.median, s.min, gbps);
    sycl::free(host, q);
    sycl::free(dev, q);
  }
}

// -------------------------------------------------------------- dispatch ---

// One decode layer as the provider does it today: copy activations in, launch
// a kernel, copy partials out, wait.
void bench_dispatch(sycl::queue &q, int iters) {
  std::printf("\n[dispatch] H2D %zu B + kernel + D2H %zu B + wait (one decode layer)\n",
              kActBytes, kOutBytes);
  void *h_act = sycl::malloc_host(kActBytes, q);
  void *d_act = sycl::malloc_device(kActBytes, q);
  float *h_out = static_cast<float *>(sycl::malloc_host(kOutBytes, q));
  float *d_out = sycl::malloc_device<float>(kOutBytes / sizeof(float), q);
  std::memset(h_act, 0x3C, kActBytes);

  auto one = [&] {
    q.memcpy(d_act, h_act, kActBytes);
    q.parallel_for(sycl::range<1>(2048), [=](sycl::id<1> i) { d_out[i] = static_cast<float>(i[0]); });
    q.memcpy(h_out, d_out, kOutBytes);
    q.wait();
  };
  for (int i = 0; i < 100; ++i) one();

  std::vector<double> rt;
  rt.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    auto t0 = Clock::now();
    one();
    rt.push_back(us_since(t0));
  }
  Stats s = summarize(rt);
  report("full dispatch (chained, one wait)", s);
  std::printf("    -> 16 layers/token: %6.2f us    48 layers/token: %6.2f us\n",
              s.median * 16, s.median * 48);

  sycl::free(h_act, q); sycl::free(d_act, q);
  sycl::free(h_out, q); sycl::free(d_out, q);
}

// ------------------------------------------------------------ persistent ---

// One kernel, launched once, spinning on a host-visible doorbell. The host
// rings it and polls for completion -- no per-layer submit, no CPU in the
// signalling path. Spin is ALWAYS bounded so the device cannot wedge.
void bench_persistent(sycl::queue &q, int iters) {
  std::printf("\n[persistent] long-lived kernel + host doorbell\n");
  auto *ctl = sycl::malloc_host<std::uint32_t>(4, q);   // [0]=doorbell [1]=done [2]=exit
  auto *h_act = static_cast<sycl::half *>(sycl::malloc_host(kActBytes, q));
  auto *h_out = sycl::malloc_host<float>(kOutBytes / sizeof(float), q);
  ctl[0] = 0; ctl[1] = 0; ctl[2] = 0;
  std::memset(h_act, 0x3C, kActBytes);

  // Escape hatch: the kernel exits after this many spin iterations no matter
  // what, so a missed doorbell degrades to a slow test, never a hung GPU.
  constexpr std::uint64_t kMaxSpins = 40'000'000'000ull;

  sycl::event worker = q.single_task([=]() {
    using Ref = sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                 sycl::memory_scope::system,
                                 sycl::access::address_space::global_space>;
    Ref bell(ctl[0]), done(ctl[1]), quit(ctl[2]);
    std::uint32_t seen = 0;
    for (std::uint64_t spin = 0; spin < kMaxSpins; ++spin) {
      if (quit.load() != 0u) break;
      const std::uint32_t cur = bell.load();
      if (cur != seen) {
        seen = cur;
        float acc = 0.0f;
        for (int i = 0; i < 2048; ++i) acc += static_cast<float>(h_act[i]);
        h_out[0] = acc;
        done.store(cur);
      }
    }
  });

  using Ref = sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                               sycl::memory_scope::system,
                               sycl::access::address_space::global_space>;
  Ref bell(ctl[0]), done(ctl[1]), quit(ctl[2]);

  // Warm up and confirm the doorbell is actually observed before timing.
  bool alive = false;
  for (int i = 0; i < 100; ++i) {
    const std::uint32_t want = bell.load() + 1;
    bell.store(want);
    auto t0 = Clock::now();
    while (done.load() != want && us_since(t0) < 200000.0) {}
    if (done.load() == want) alive = true; else break;
  }
  if (!alive) {
    std::printf("  doorbell never acknowledged -- system-scope atomics on host USM\n"
                "  are not working on this stack; persistent-kernel path is not viable as written\n");
    quit.store(1);
    worker.wait();
    sycl::free(ctl, q); sycl::free(h_act, q); sycl::free(h_out, q);
    return;
  }

  std::vector<double> rt;
  rt.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    const std::uint32_t want = bell.load() + 1;
    auto t0 = Clock::now();
    bell.store(want);
    while (done.load() != want && us_since(t0) < 200000.0) {}
    rt.push_back(us_since(t0));
  }
  Stats s = summarize(rt);
  report("doorbell round trip", s);
  std::printf("    -> 16 layers/token: %6.2f us    48 layers/token: %6.2f us\n",
              s.median * 16, s.median * 48);

  quit.store(1);
  worker.wait();
  sycl::free(ctl, q); sycl::free(h_act, q); sycl::free(h_out, q);
}

// ---------------------------------------------------------------- queues ---

void bench_queues(sycl::device dev, int iters, int nq) {
  std::printf("\n[queues] %d concurrent dispatches on %d in-order queues\n", nq, nq);
  std::vector<sycl::queue> qs;
  std::vector<void *> h_act, d_act, d_out, h_out;
  for (int i = 0; i < nq; ++i) {
    qs.emplace_back(dev, sycl::property::queue::in_order{});
    h_act.push_back(sycl::malloc_host(kActBytes, qs[i]));
    d_act.push_back(sycl::malloc_device(kActBytes, qs[i]));
    d_out.push_back(sycl::malloc_device(kOutBytes, qs[i]));
    h_out.push_back(sycl::malloc_host(kOutBytes, qs[i]));
    std::memset(h_act[i], 0x3C, kActBytes);
  }
  auto wave = [&] {
    for (int i = 0; i < nq; ++i) {
      auto *o = static_cast<float *>(d_out[i]);
      qs[i].memcpy(d_act[i], h_act[i], kActBytes);
      qs[i].parallel_for(sycl::range<1>(2048), [=](sycl::id<1> k) { o[k] = static_cast<float>(k[0]); });
      qs[i].memcpy(h_out[i], d_out[i], kOutBytes);
    }
    for (int i = 0; i < nq; ++i) qs[i].wait();
  };
  for (int i = 0; i < 50; ++i) wave();

  std::vector<double> rt;
  rt.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    auto t0 = Clock::now();
    wave();
    rt.push_back(us_since(t0));
  }
  Stats s = summarize(rt);
  report("wave of concurrent dispatches", s);
  std::printf("    -> per dispatch: %6.2f us  (serial would be %d x single)\n",
              s.median / nq, nq);
  for (int i = 0; i < nq; ++i) {
    sycl::free(h_act[i], qs[i]); sycl::free(d_act[i], qs[i]);
    sycl::free(d_out[i], qs[i]); sycl::free(h_out[i], qs[i]);
  }
}

} // namespace

int main(int argc, char **argv) {
  std::string mode = argc > 1 ? argv[1] : "all";
  const int iters = argc > 2 ? std::stoi(argv[2]) : 2000;

  sycl::device dev;
  try {
    dev = sycl::device(sycl::gpu_selector_v);
  } catch (const sycl::exception &e) {
    std::fprintf(stderr, "no SYCL GPU: %s\n", e.what());
    return 1;
  }
  sycl::queue q(dev, sycl::property::queue::in_order{});
  std::printf("device : %s\n", dev.get_info<sycl::info::device::name>().c_str());
  std::printf("iters  : %d\n", iters);

  if (mode == "all" || mode == "launch") bench_launch(q, iters);
  if (mode == "all" || mode == "copy") bench_copy(q, std::min(iters, 500));
  if (mode == "all" || mode == "dispatch") bench_dispatch(q, iters);
  if (mode == "all" || mode == "persistent") bench_persistent(q, iters);
  if (mode == "all" || mode == "queues") {
    for (int nq : {2, 4, 8}) bench_queues(dev, std::min(iters, 500), nq);
  }
  std::printf("\ndone\n");
  return 0;
}
