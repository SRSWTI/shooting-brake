/*
 * b13_doorbell_flag_probe.cpp — Kill-bench #13: split the 38 us host leg.
 *
 * window_decomposition.json attributes ~39 us/dispatch to a "host leg" and
 * calls it "the largest unattributed decode item ... the poller already
 * spins, so 39 us is anomalous and unexplained". Bench 4 measured the B70
 * doing the equivalent host<->device flag handoff in 5.3 us round trip.
 * This probe asks where the 5090's half goes.
 *
 * Production Tier 3 doorbell (src/phase7/b70_capi.cpp):
 *   CUDA graph : cuStreamWriteValue32(signal, M)
 *                cuStreamWaitValue32(completion, 1)
 *   host thread: spin on signal[0] -> issue()/take() -> completion[0] = 1
 *
 * issue()/take() are already accounted as device time. Everything else is
 * the two flag handoffs, which this reproduces with NO model, NO bank and
 * NO B70 -- just the 5090 and one host thread -- and times separately:
 *
 *   A) graph launch -> host observes signal      (GPU write visibility)
 *   B) host writes completion -> replay retires  (cuStreamWaitValue32 wakeup)
 *
 * B is the prime suspect: cuStreamWaitValue32 parks the GPU front-end on a
 * host-mapped word, and its polling interval is undocumented.
 *
 * Flags sit on separate cache lines. Bench 4 learned that the hard way: two
 * flags sharing a line let one side's write clobber the other's.
 *
 * Build:
 *   g++ -O2 -o experiments/b13_doorbell_flag_probe \
 *       experiments/b13_doorbell_flag_probe.cpp \
 *       -I/usr/local/cuda-13/include -lcuda
 */
#include <cuda.h>

#include <algorithm>
#include <atomic>
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

constexpr size_t kLine = 256;         // one cache line per flag, see Bench 4
constexpr double kTimeoutSec = 5.0;

void cu_check(CUresult r, const char* what) {
  if (r == CUDA_SUCCESS) return;
  const char* name = nullptr;
  cuGetErrorName(r, &name);
  std::fprintf(stderr, "FATAL %s: %s (%d)\n", what, name ? name : "?", int(r));
  std::exit(1);
}

double now_sec() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return double(ts.tv_sec) + double(ts.tv_nsec) * 1e-9;
}

struct Stats {
  double min = 0, p50 = 0, p90 = 0, p99 = 0, max = 0;
  size_t n = 0;
};

Stats summarize(std::vector<double> v) {
  if (v.empty()) return {};
  std::sort(v.begin(), v.end());
  auto pick = [&](double q) { return v[size_t(q * double(v.size() - 1) + 0.5)]; };
  return Stats{v.front(), pick(0.50), pick(0.90), pick(0.99), v.back(), v.size()};
}

void print_stats(const char* label, const Stats& s) {
  std::printf("  %-40s n=%-5zu min %7.2f  p50 %7.2f  p90 %7.2f  p99 %8.2f  max %9.2f\n",
              label, s.n, s.min, s.p50, s.p90, s.p99, s.max);
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  const int iters = (argc > 1) ? std::atoi(argv[1]) : 2000;
  const int warmup = 200;

  cu_check(cuInit(0), "cuInit");
  CUdevice dev;
  cu_check(cuDeviceGet(&dev, 0), "cuDeviceGet");
  char name[128];
  cu_check(cuDeviceGetName(name, sizeof(name), dev), "cuDeviceGetName");
  CUcontext ctx;
  cu_check(cuCtxCreate(&ctx, nullptr, 0, dev), "cuCtxCreate");  // v4 signature

  std::printf("== Bench 13: doorbell flag handoff probe ==\n");
  std::printf("device: %s\n", name);

  int can_map = 0;
  cuDeviceGetAttribute(&can_map, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, dev);
  std::printf("  can_map_host_memory=%d\n\n", can_map);

  // Host-mapped flags, one cache line apart.
  void* raw = nullptr;
  cu_check(cuMemHostAlloc(&raw, kLine * 2, CU_MEMHOSTALLOC_DEVICEMAP),
           "cuMemHostAlloc");
  volatile uint32_t* signal = reinterpret_cast<volatile uint32_t*>(
      static_cast<char*>(raw) + 0 * kLine);
  volatile uint32_t* completion = reinterpret_cast<volatile uint32_t*>(
      static_cast<char*>(raw) + 1 * kLine);
  *signal = 0;
  *completion = 0;

  CUdeviceptr d_signal = 0, d_completion = 0;
  cu_check(cuMemHostGetDevicePointer(&d_signal, (void*)signal, 0), "getdevptr signal");
  cu_check(cuMemHostGetDevicePointer(&d_completion, (void*)completion, 0),
           "getdevptr completion");

  CUstream stream;
  cu_check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate");

  // Capture exactly the production shape: write the signal, then park.
  cu_check(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_RELAXED), "beginCapture");
  cu_check(cuStreamWriteValue32(stream, d_signal, 1, 0), "writeValue");
  cu_check(cuStreamWaitValue32(stream, d_completion, 1, CU_STREAM_WAIT_VALUE_EQ),
           "waitValue");
  CUgraph graph;
  cu_check(cuStreamEndCapture(stream, &graph), "endCapture");
  CUgraphExec exec;
  cu_check(cuGraphInstantiate(&exec, graph, 0), "graphInstantiate");

  std::vector<double> a_us, b_us, rtt_us;
  a_us.reserve(iters); b_us.reserve(iters); rtt_us.reserve(iters);

  bool timed_out = false;
  for (int i = 0; i < warmup + iters && !timed_out; ++i) {
    const bool timed = i >= warmup;
    *signal = 0;
    *completion = 0;
    std::atomic_signal_fence(std::memory_order_seq_cst);

    const double t_launch = now_sec();
    cu_check(cuGraphLaunch(exec, stream), "graphLaunch");

    // --- leg A: GPU's writeValue becomes visible to this host thread ---
    while (*signal != 1) {
      if (now_sec() - t_launch > kTimeoutSec) {
        std::printf("  !! timeout waiting for signal at iter %d\n", i);
        timed_out = true;
        break;
      }
      SB_SPIN_HINT();
    }
    if (timed_out) break;
    const double t_seen = now_sec();

    // --- leg B: our completion write releases cuStreamWaitValue32 ---
    *completion = 1;
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const double t_wrote = now_sec();

    cu_check(cuStreamSynchronize(stream), "streamSync");
    const double t_done = now_sec();

    if (timed) {
      a_us.push_back((t_seen - t_launch) * 1e6);
      b_us.push_back((t_done - t_wrote) * 1e6);
      rtt_us.push_back((t_done - t_launch) * 1e6);
    }
  }

  if (!a_us.empty()) {
    std::printf("-- per-dispatch flag handoff (us) --\n");
    print_stats("A: launch -> host sees signal", summarize(a_us));
    print_stats("B: host writes -> waitValue releases", summarize(b_us));
    print_stats("A+B: full doorbell round trip", summarize(rtt_us));

    const Stats A = summarize(a_us), B = summarize(b_us), R = summarize(rtt_us);
    std::printf("\n== verdict ==\n");
    std::printf("  production host leg (measured elsewhere): ~39 us/dispatch\n");
    std::printf("  this probe's equivalent round trip:       %.2f us (p50)\n", R.p50);
    std::printf("  split: signal visibility %.2f us | waitValue wakeup %.2f us\n",
                A.p50, B.p50);
    std::printf("  B70 host<->device floor (Bench 4):        5.28 us RTT\n");
    if (B.p50 > A.p50 * 2)
      std::printf("\n  => cuStreamWaitValue32 wakeup DOMINATES. The fix is on the\n"
                  "     completion path, not the signal path.\n");
    else if (A.p50 > B.p50 * 2)
      std::printf("\n  => signal visibility DOMINATES. The poller is slow to see the\n"
                  "     GPU's write; suspect the 48-flag linear sweep.\n");
    else
      std::printf("\n  => both legs comparable; the cost is the mechanism itself.\n");
  }

  cuGraphExecDestroy(exec);
  cuGraphDestroy(graph);
  cuStreamDestroy(stream);
  cuMemFreeHost(raw);
  cuCtxDestroy(ctx);
  return 0;
}
