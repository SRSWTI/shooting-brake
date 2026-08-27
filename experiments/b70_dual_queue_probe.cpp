// Can one B70 run two SYCL queues concurrently, and do they actually overlap?
//
// WHY THIS EXISTS
// ---------------
// Intra-dispatch pipelining needs H2D of chunk B to run under the kernel of
// chunk A. That requires either two in-order queues or one out-of-order queue
// with event chains. Neither is exercised anywhere in this repo: the provider
// uses exactly one in-order queue.
//
// It is not a given. We are pinned to SYCL_UR_USE_LEVEL_ZERO_V2=0 because the
// V2 adapter segfaults on a plain USM memcpy -- null function pointer inside
// ur_command_list_manager::isGraphCaptureActive, oneAPI 2026.1 (Bench 23). A
// stack that crashes on a single memcpy has not earned an assumption about two
// concurrent queues.
//
// Three questions, in order of how badly a "no" hurts:
//   1. CORRECTNESS. Do two queues on one device produce right answers at all?
//   2. OVERLAP. Does copy-on-q2 actually run under kernel-on-q1, or does the
//      driver serialise them anyway? Overlap is the entire point; correct-but-
//      serialised means the design buys nothing.
//   3. FALLBACK. If two in-order queues fail, does one out-of-order queue with
//      explicit event dependencies overlap instead?
//
// Shapes mirror one B70 chunk at MAX_BATCH=2048 split two ways: 1024 tokens x
// 3072 hidden, fp16 -> 6.3 MiB per H2D, which is the transfer we need to hide.
//
// Build: source /opt/intel/oneapi/setvars.sh --force
//        icpx -fsycl -O2 -o b70_dual_queue_probe b70_dual_queue_probe.cpp
#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using clk = std::chrono::steady_clock;

double ms_since(clk::time_point t0) {
  return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

// Deliberately memory-bound and long enough to hide a 6.3 MiB copy behind.
// A trivial kernel would finish before the copy started and prove nothing.
sycl::event busy_kernel(sycl::queue& q, float* buf, std::size_t n, int iters,
                        const std::vector<sycl::event>& deps = {}) {
  return q.submit([&](sycl::handler& h) {
    if (!deps.empty()) h.depends_on(deps);
    h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
      float v = buf[i];
      for (int k = 0; k < iters; ++k) v = v * 1.0000001f + 1e-6f;
      buf[i] = v;
    });
  });
}

struct Arm {
  std::string name;
  double wall_ms;
  bool correct;
  std::string note;
};

}  // namespace

int main(int argc, char** argv) {
  const int iters = argc > 1 ? std::atoi(argv[1]) : 300;
  const std::size_t tokens = 1024, hidden = 3072;
  const std::size_t copy_elems = tokens * hidden;              // fp16 pair
  const std::size_t work_elems = 4u << 20;                     // 16 MiB fp32

  sycl::device dev;
  try {
    dev = sycl::device(sycl::gpu_selector_v);
  } catch (const std::exception& e) {
    std::printf("no gpu: %s\n", e.what());
    return 1;
  }
  std::printf("device : %s\n", dev.get_info<sycl::info::device::name>().c_str());
  std::printf("shape  : copy %.2f MiB x2, kernel %.0f MiB x %d iters\n\n",
              copy_elems * 2.0 / (1024 * 1024),
              work_elems * 4.0 / (1024 * 1024), iters);

  sycl::context ctx(dev);
  auto* h_a = sycl::malloc_host<sycl::half>(copy_elems, ctx);
  auto* h_b = sycl::malloc_host<sycl::half>(copy_elems, ctx);
  auto* d_a = sycl::malloc_device<sycl::half>(copy_elems, dev, ctx);
  auto* d_b = sycl::malloc_device<sycl::half>(copy_elems, dev, ctx);
  auto* w_a = sycl::malloc_device<float>(work_elems, dev, ctx);
  auto* w_b = sycl::malloc_device<float>(work_elems, dev, ctx);
  if (!h_a || !h_b || !d_a || !d_b || !w_a || !w_b) {
    std::printf("allocation failed\n");
    return 1;
  }
  for (std::size_t i = 0; i < copy_elems; ++i) {
    h_a[i] = static_cast<sycl::half>(1.0f);
    h_b[i] = static_cast<sycl::half>(2.0f);
  }

  auto check = [&](sycl::queue& q) {
    // Round-trip both copies and confirm the payloads did not cross.
    std::vector<sycl::half> back_a(copy_elems), back_b(copy_elems);
    q.memcpy(back_a.data(), d_a, copy_elems * sizeof(sycl::half)).wait();
    q.memcpy(back_b.data(), d_b, copy_elems * sizeof(sycl::half)).wait();
    return static_cast<float>(back_a[0]) == 1.0f &&
           static_cast<float>(back_a[copy_elems - 1]) == 1.0f &&
           static_cast<float>(back_b[0]) == 2.0f &&
           static_cast<float>(back_b[copy_elems - 1]) == 2.0f;
  };

  std::vector<Arm> arms;

  // ---- Arm 1: one in-order queue, everything serial. The baseline to beat.
  {
    sycl::queue q(ctx, dev, sycl::property::queue::in_order{});
    busy_kernel(q, w_a, work_elems, iters).wait();     // warm JIT
    auto t0 = clk::now();
    q.memcpy(d_a, h_a, copy_elems * sizeof(sycl::half));
    busy_kernel(q, w_a, work_elems, iters);
    q.memcpy(d_b, h_b, copy_elems * sizeof(sycl::half));
    busy_kernel(q, w_b, work_elems, iters);
    q.wait_and_throw();
    double w = ms_since(t0);
    arms.push_back({"1 in-order queue (serial)", w, check(q), "baseline"});
  }

  // ---- Arm 2: two in-order queues. The design under test.
  {
    sycl::queue q1(ctx, dev, sycl::property::queue::in_order{});
    sycl::queue q2(ctx, dev, sycl::property::queue::in_order{});
    busy_kernel(q1, w_a, work_elems, iters).wait();
    busy_kernel(q2, w_b, work_elems, iters).wait();
    auto t0 = clk::now();
    q1.memcpy(d_a, h_a, copy_elems * sizeof(sycl::half));
    busy_kernel(q1, w_a, work_elems, iters);
    q2.memcpy(d_b, h_b, copy_elems * sizeof(sycl::half));
    busy_kernel(q2, w_b, work_elems, iters);
    q1.wait_and_throw();
    q2.wait_and_throw();
    double w = ms_since(t0);
    arms.push_back({"2 in-order queues", w, check(q1), "the design"});
  }

  // ---- Arm 3: one out-of-order queue, explicit event chains. The fallback.
  {
    sycl::queue q(ctx, dev);
    busy_kernel(q, w_a, work_elems, iters).wait();
    auto t0 = clk::now();
    auto ca = q.memcpy(d_a, h_a, copy_elems * sizeof(sycl::half));
    auto ka = busy_kernel(q, w_a, work_elems, iters, {ca});
    auto cb = q.memcpy(d_b, h_b, copy_elems * sizeof(sycl::half));
    auto kb = busy_kernel(q, w_b, work_elems, iters, {cb});
    ka.wait();
    kb.wait();
    double w = ms_since(t0);
    arms.push_back({"1 out-of-order queue + events", w, check(q), "fallback"});
  }

  std::printf("%-34s %10s %10s  %s\n", "arm", "wall ms", "correct", "note");
  double base = arms.front().wall_ms;
  for (auto& a : arms) {
    std::printf("%-34s %10.2f %10s  %s (%.2fx vs serial)\n", a.name.c_str(),
                a.wall_ms, a.correct ? "yes" : "NO", a.note.c_str(),
                base / a.wall_ms);
  }

  // A concurrent arm that merely matches serial is CORRECT and USELESS: the
  // driver ran them back to back. Only a real speedup proves overlap.
  double best_conc = std::min(arms[1].wall_ms, arms[2].wall_ms);
  bool all_ok = arms[0].correct && arms[1].correct && arms[2].correct;
  std::printf("\nverdict: ");
  if (!all_ok) {
    std::printf("CORRECTNESS FAILURE -- pipelining is not available this way\n");
  } else if (base / best_conc > 1.15) {
    std::printf("overlap CONFIRMED (%.2fx) -- pipelining has headroom\n",
                base / best_conc);
  } else {
    std::printf("correct but SERIALISED (%.2fx) -- driver is not overlapping;\n"
                "         pipelining would buy nothing through this path\n",
                base / best_conc);
  }

  sycl::free(h_a, ctx); sycl::free(h_b, ctx);
  sycl::free(d_a, ctx); sycl::free(d_b, ctx);
  sycl::free(w_a, ctx); sycl::free(w_b, ctx);
  return 0;
}
