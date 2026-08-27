/*
 * b13_sweep_probe.cpp — Kill-bench #13 step 2: is the 48-flag sweep the 34 us?
 *
 * Step 1 (b13_doorbell_flag_probe) proved the doorbell MECHANISM costs 4.79 us
 * while production spends ~39 us. ~34 us is software. Leading suspect: the
 * poller's steady-state loop in src/phase7/b70_capi.cpp
 *
 *     for (const PollLayer& entry : snapshot) {
 *       const uint32_t M = entry.signal[0];     // host-mapped, GPU-written
 *       if (M == 0) continue;
 *       ...
 *     }
 *
 * Two properties make this a suspect:
 *   1. Every iteration reads a host-mapped word the GPU writes, so each read
 *      is a coherency miss, not a cache hit.
 *   2. Production allocates one flag PER LAYER via separate
 *      alloc_host_mapped_flag() calls, so the 48 flags land on 48 different
 *      pages -> a TLB miss per read as well.
 *   3. After serving layer k the for-loop finishes and the while-loop
 *      restarts the sweep at layer 0, so the next layer to fire (k+1) is
 *      found only after re-reading layers 0..k.
 *
 * This probe replicates that exactly -- N separately-allocated host-mapped
 * flags, swept linearly -- and times it. No model, no bank, no B70.
 *
 * Arms:
 *   A. cold sweep: read all N flags, none written by the GPU recently
 *   B. post-write sweep: GPU writes one flag, then sweep (the real case)
 *   C. per-position detect latency: how long to find a flag at index k
 *
 * Build:
 *   g++ -O2 -o experiments/b13_sweep_probe experiments/b13_sweep_probe.cpp \
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

void cu_check(CUresult r, const char* what) {
  if (r == CUDA_SUCCESS) return;
  const char* n = nullptr;
  cuGetErrorName(r, &n);
  std::fprintf(stderr, "FATAL %s: %s (%d)\n", what, n ? n : "?", int(r));
  std::exit(1);
}

double now_sec() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return double(ts.tv_sec) + double(ts.tv_nsec) * 1e-9;
}

double p50(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
double p99(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[size_t(0.99 * double(v.size() - 1))];
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  const int n_layers = (argc > 1) ? std::atoi(argv[1]) : 48;
  const int iters = (argc > 2) ? std::atoi(argv[2]) : 2000;

  cu_check(cuInit(0), "cuInit");
  CUdevice dev;
  cu_check(cuDeviceGet(&dev, 0), "cuDeviceGet");
  CUcontext ctx;
  cu_check(cuCtxCreate(&ctx, nullptr, 0, dev), "cuCtxCreate");
  char name[128];
  cuDeviceGetName(name, sizeof(name), dev);

  std::printf("== Bench 13 step 2: poller sweep cost ==\n");
  std::printf("device: %s   layers=%d\n\n", name, n_layers);

  // Replicate production: ONE SEPARATE host-mapped allocation per layer,
  // exactly as alloc_host_mapped_flag() is called per RoutedExperts layer.
  std::vector<volatile uint32_t*> host_flags(n_layers);
  std::vector<CUdeviceptr> dev_flags(n_layers);
  for (int i = 0; i < n_layers; ++i) {
    void* p = nullptr;
    cu_check(cuMemHostAlloc(&p, 4096, CU_MEMHOSTALLOC_DEVICEMAP), "hostAlloc");
    host_flags[i] = reinterpret_cast<volatile uint32_t*>(p);
    *host_flags[i] = 0;
    cu_check(cuMemHostGetDevicePointer(&dev_flags[i], p, 0), "getdevptr");
  }

  CUstream stream;
  cu_check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "streamCreate");

  // --- arm A: cost of one full sweep with nothing pending -----------------
  std::vector<double> cold;
  for (int it = 0; it < iters; ++it) {
    const double t0 = now_sec();
    uint32_t acc = 0;
    for (int i = 0; i < n_layers; ++i) acc |= host_flags[i][0];
    const double t1 = now_sec();
    if (acc == 0xDEADBEEF) std::printf("");  // keep the loop
    if (it >= 200) cold.push_back((t1 - t0) * 1e6);
  }
  std::printf("  A. full sweep of %d flags, none GPU-written:\n", n_layers);
  std::printf("       p50 %6.2f us   p99 %6.2f us   -> %.1f ns per flag\n\n",
              p50(cold), p99(cold), p50(cold) * 1000.0 / n_layers);

  // --- arm B: sweep right after the GPU writes one flag -------------------
  // This is the production case: the line is dirty in the GPU's view and the
  // CPU's copy has been invalidated by the inbound PCIe write.
  std::vector<double> hot;
  for (int it = 0; it < iters; ++it) {
    const int target = it % n_layers;
    for (int i = 0; i < n_layers; ++i) *host_flags[i] = 0;
    cu_check(cuStreamWriteValue32(stream, dev_flags[target], 1, 0), "writeValue");
    cu_check(cuStreamSynchronize(stream), "sync");

    const double t0 = now_sec();
    uint32_t acc = 0;
    for (int i = 0; i < n_layers; ++i) acc |= host_flags[i][0];
    const double t1 = now_sec();
    if (acc == 0xDEADBEEF) std::printf("");
    if (it >= 200) hot.push_back((t1 - t0) * 1e6);
  }
  std::printf("  B. full sweep immediately after a GPU flag write:\n");
  std::printf("       p50 %6.2f us   p99 %6.2f us   -> %.1f ns per flag\n\n",
              p50(hot), p99(hot), p50(hot) * 1000.0 / n_layers);

  // --- arm C: detect latency by position ----------------------------------
  // The poller restarts at layer 0 after every dispatch, so a layer at
  // index k is found only after k wasted reads.
  std::printf("  C. scan cost to reach a flag at index k (restart-at-0 model):\n");
  for (int k : {0, 8, 16, 24, 32, 40, 47}) {
    if (k >= n_layers) continue;
    std::vector<double> pos;
    for (int it = 0; it < 500; ++it) {
      const double t0 = now_sec();
      uint32_t acc = 0;
      for (int i = 0; i <= k; ++i) acc |= host_flags[i][0];
      const double t1 = now_sec();
      if (acc == 0xDEADBEEF) std::printf("");
      if (it >= 50) pos.push_back((t1 - t0) * 1e6);
    }
    std::printf("       k=%2d : %6.2f us\n", k, p50(pos));
  }

  const double per_sweep = p50(hot);
  const double avg_wasted = per_sweep * 0.5;  // mean position over a pass
  std::printf("\n== verdict ==\n");
  std::printf("  one full sweep      : %.2f us\n", per_sweep);
  std::printf("  mean wasted scan    : %.2f us/dispatch (restart-at-0, avg position)\n",
              avg_wasted);
  std::printf("  unexplained gap     : ~34 us/dispatch (Bench 13 step 1)\n");
  if (avg_wasted > 15.0)
    std::printf("  => the sweep EXPLAINS most of the gap. Fix: index the signal,\n"
                "     or resume the sweep where it left off.\n");
  else if (avg_wasted > 5.0)
    std::printf("  => the sweep explains a MEANINGFUL SLICE but not all of it.\n"
                "     Keep looking for the remainder in issue()/take().\n");
  else
    std::printf("  => the sweep is CHEAP and is NOT the 34 us. The cost is inside\n"
                "     issue()/take() host work or CPU contention. Look there.\n");

  for (int i = 0; i < n_layers; ++i) cuMemFreeHost((void*)host_flags[i]);
  cuStreamDestroy(stream);
  cuCtxDestroy(ctx);
  return 0;
}
