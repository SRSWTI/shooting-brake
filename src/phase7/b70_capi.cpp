#include "b70_capi.h"

#include "b70_provider.hpp"

#include <atomic>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <thread>
#include <vector>

#include <pthread.h>
#include <sched.h>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define SB_SPIN_HINT() _mm_pause()
#else
#define SB_SPIN_HINT() ((void)0)
#endif

using shooting_brake::phase1::B70Provider;
using shooting_brake::phase1::ProviderConfig;
using shooting_brake::phase1::ProviderStatus;

namespace {
int status_to_int(ProviderStatus s) noexcept {
  switch (s) {
    case ProviderStatus::ok:
      return 0;
    case ProviderStatus::busy:
      return 1;
    default:
      return -1;
  }
}

// One recorded dispatch window on the host CLOCK_MONOTONIC (steady_clock)
// timeline, for merging with a torch-profiler capture taken in the same
// process. t0 brackets issue(), t1 the completed take(); kernel/total are
// the provider's Level Zero event timings (0 unless B70_PROFILE=1).
struct TraceEntry {
  uint64_t t0_ns;
  uint64_t t1_ns;
  uint64_t kernel_ns;
  uint64_t total_ns;
  uint32_t layer;
  uint32_t M;
};
static_assert(sizeof(TraceEntry) == 40);

constexpr size_t kTraceCapacity = 1u << 16;  // ~2.6 MiB, ~13 decode steps/48L

// One registered layer's flags and pinned buffers. Buffers are owned by
// the caller (they are CUDA pinned allocations) and must outlive the
// poller.
struct PollLayer {
  size_t layer;
  volatile uint32_t* signal;
  volatile uint32_t* completion;
  const sycl::half* hidden;
  const int32_t* ids;
  const float* weights;
  float* output;
};

class B70Poller {
 public:
  B70Poller(B70Provider* provider, uint64_t generation)
      : provider_(provider), generation_(generation) {}

  B70Provider* provider() const { return provider_; }

  ~B70Poller() { stop(); }

  // Safe to call while running: the sweep takes a snapshot under the
  // same mutex, and layers are only ever appended.
  void add(const PollLayer& layer) {
    std::lock_guard<std::mutex> guard(mutex_);
    layers_.push_back(layer);
    // Release-publish the new size; the sweep acquires it to decide
    // whether to re-snapshot.
    layer_count_.store(layers_.size(), std::memory_order_release);
  }

  int start(int pin_cpu = -1) {
    if (running_.exchange(true)) return 0;  // already running
    try {
      thread_ = std::thread([this, pin_cpu] {
        if (pin_cpu >= 0) pin_to_cpu(pin_cpu);
        loop();
      });
    } catch (...) {
      running_ = false;
      return -1;
    }
    return 0;
  }

  // Best-effort affinity: a failed pin degrades to the scheduler's choice,
  // which is today's behaviour. It must never fail the start — an unpinned
  // poller serves correctly, just with cross-core migration jitter.
  static void pin_to_cpu(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0) {
      std::fprintf(stderr,
                   "[sb_b70] poller CPU pin to %d failed; running unpinned\n",
                   cpu);
    }
  }

  void stop() {
    if (!running_.exchange(false)) return;
    if (thread_.joinable()) thread_.join();
  }
  void reset() {
    dispatches_.store(0, std::memory_order_relaxed);
    rows_.store(0, std::memory_order_relaxed);
    for (auto& bucket : m_histogram_) {
      bucket.store(0, std::memory_order_relaxed);
    }
    errors_.store(0, std::memory_order_relaxed);
    service_ns_.store(0, std::memory_order_relaxed);
    total_ns_.store(0, std::memory_order_relaxed);
    kernel_ns_.store(0, std::memory_order_relaxed);
  }


  uint64_t dispatch_count() const { return dispatches_.load(); }
  uint64_t error_count() const { return errors_.load(); }
  uint64_t row_count() const { return rows_.load(); }
  uint64_t m_bucket_count(size_t bucket) const {
    if (bucket >= m_histogram_.size()) return 0;
    return m_histogram_[bucket].load();
  }
  uint64_t service_ns() const { return service_ns_.load(); }
  // Nonzero only under SHOOTING_BRAKE_B70_PROFILE=1; the provider does not
  // timestamp commands otherwise.
  uint64_t kernel_ns() const { return kernel_ns_.load(); }
  // Nonzero only under SHOOTING_BRAKE_B70_PROFILE=1. This spans the profiled
  // device queue from the first input copy through the completed output copy.
  // On Level Zero the endpoint copies can run on engines whose profiling
  // clocks are not comparable, so this raw span can underflow and must not be
  // used for decomposition without first establishing a common timebase.
  uint64_t total_ns() const { return total_ns_.load(); }

 private:
  void loop() {
    std::vector<PollLayer> snapshot;
    size_t known = 0;
    uint64_t sequence = 0;
    bool cs_disabled = false;
    // Fixed once the bank is loaded, and the poller only starts after
    // that, so read it once rather than per dispatch on the hot path.
    const size_t hidden = hidden_size();
    const uint32_t total_layers =
        static_cast<uint32_t>(provider_->capability().num_layers);
    static const bool cs_env = [] {
      const char* v = std::getenv("SHOOTING_BRAKE_B70_CS_DOORBELL");
      return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();

    while (running_.load(std::memory_order_relaxed)) {
      // Refresh only when a new layer appeared, so the steady-state
      // sweep touches no lock.
      if (layer_count_.load(std::memory_order_acquire) != known) {
        std::lock_guard<std::mutex> guard(mutex_);
        snapshot = layers_;
        known = snapshot.size();
      }

      // Doorbell 2.0 (SHOOTING_BRAKE_B70_CS_DOORBELL=1): once every layer is
      // registered, decode-shaped steps (M <= 32) ride command-streamer
      // chains and this thread leaves the dispatch critical path entirely --
      // the probes measured the CS bracket at ~8 us against this loop's 61 us
      // round trip. Prefill/mixed steps (M > 32) and the registration warmup
      // stay on the classic sweep below. Any chain failure latches classic
      // mode permanently; the failed step is finished by the sweep, which
      // spins the remaining layers' untouched signals.
      if (cs_env && !cs_disabled && !snapshot.empty() &&
          known == total_layers) {
        const uint32_t M = snapshot[0].signal[0];
        if (M != 0 && M <= 32) {
          bool ok = true;
          for (const PollLayer& entry : snapshot) {
            if (provider_->issue_cs_chain(
                    generation_, entry.layer, entry.hidden, entry.ids,
                    entry.weights, entry.output, M, entry.signal,
                    entry.completion) !=
                shooting_brake::phase1::ProviderStatus::ok) {
              ok = false;
              cs_disabled = true;
              errors_.fetch_add(1);
              break;
            }
          }
          if (ok) {
            sequence += snapshot.size();
            dispatches_.fetch_add(snapshot.size());
            rows_.fetch_add(static_cast<uint64_t>(M) * snapshot.size());
            m_histogram_[m_bucket(M)].fetch_add(snapshot.size());
            continue;
          }
        }
      }

      for (const PollLayer& entry : snapshot) {
        // The signal's VALUE is the batch size M; 0 means idle.
        const uint32_t M = entry.signal[0];
        if (M == 0) continue;

        // Clear before dispatching so the next graph replay can signal
        // this layer again while we work.
        entry.signal[0] = 0;
        std::atomic_thread_fence(std::memory_order_seq_cst);

        const auto t0 = std::chrono::steady_clock::now();
        ++sequence;
        shooting_brake::phase1::DispatchResult result;
        ProviderStatus status = provider_->issue(
            generation_, sequence, entry.layer, entry.hidden, entry.ids,
            entry.weights, M);
        if (status == ProviderStatus::ok) {
          status = provider_->take(generation_, sequence, entry.output,
                                   static_cast<size_t>(M) * hidden,
                                   &result);
        }
        if (status != ProviderStatus::ok) errors_.fetch_add(1);

        const auto t1 = std::chrono::steady_clock::now();
        service_ns_.fetch_add(static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0)
                .count()));

        // Lock-free single-producer trace ring on the host steady_clock
        // (CLOCK_MONOTONIC) timeline; readers snapshot via
        // sb_b70_poll_trace_snapshot and merge with a same-process torch
        // profiler capture.
        {
          const uint64_t slot = trace_head_.load(std::memory_order_relaxed);
          TraceEntry& te = trace_[slot % kTraceCapacity];
          te.t0_ns = static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  t0.time_since_epoch()).count());
          te.t1_ns = static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  t1.time_since_epoch()).count());
          te.kernel_ns = static_cast<uint64_t>(result.kernel_us * 1000.0);
          te.total_ns = static_cast<uint64_t>(result.total_us * 1000.0);
          te.layer = static_cast<uint32_t>(entry.layer);
          te.M = M;
          trace_head_.store(slot + 1, std::memory_order_release);
        }

        // The provider only fills these timings on a profiled queue; they
        // are 0 otherwise. Keeping both makes the dispatch self-describing:
        // service_ns - total_ns is host-side issue/take work, total_ns -
        // kernel_ns is device-queue work outside the kernel, and kernel_ns
        // is the on-device kernel itself. Guessing those splits from the
        // outside is how you end up optimising the wrong part.
        if (result.total_us > 0.0) {
          total_ns_.fetch_add(
              static_cast<uint64_t>(result.total_us * 1000.0));
        }
        if (result.kernel_us > 0.0) {
          kernel_ns_.fetch_add(
              static_cast<uint64_t>(result.kernel_us * 1000.0));
        }

        // Dispatch count alone cannot distinguish M=1 decode from a prefill
        // dispatch that processes hundreds of rows. Keep both the exact row
        // sum and coarse shape so cumulative snapshots remain interpretable.
        rows_.fetch_add(M);
        m_histogram_[m_bucket(M)].fetch_add(1);

        // Always release the waiter, even on failure: the CUDA side is
        // parked in cuStreamWaitValue32, which has no timeout.
        std::atomic_thread_fence(std::memory_order_seq_cst);
        entry.completion[0] = 1;
        dispatches_.fetch_add(1);
      }

      SB_SPIN_HINT();
    }
  }

  static size_t m_bucket(uint32_t M) {
    if (M <= 2) return M - 1;
    if (M <= 4) return 2;
    if (M <= 8) return 3;
    if (M <= 16) return 4;
    if (M <= 32) return 5;
    return 6;
  }

  // Hidden size of the loaded bank, not a constant. This was 2048 — the
  // 35B's — and it is the length this poller declares for the output
  // buffer when it calls `take`. On any other geometry the provider sees
  // a length that disagrees with what it produced and rejects every
  // dispatch: 47 failures on the 122B, one per layer, with nothing else
  // wrong, and invisible to any 35B test because 2048 is correct there.
  // Read from the provider's own capability, which reports the geometry
  // adopted from the bank header.
  size_t hidden_size() const {
    const auto sizes = provider_->capability().supported_hidden_sizes;
    return sizes.empty() ? 0 : static_cast<size_t>(sizes.front());
  }

  B70Provider* provider_;
  uint64_t generation_;
  std::vector<PollLayer> layers_;
  std::mutex mutex_;
  std::atomic<size_t> layer_count_{0};
  std::atomic<bool> running_{false};
  std::atomic<uint64_t> dispatches_{0};
  std::atomic<uint64_t> rows_{0};
  std::array<std::atomic<uint64_t>, 7> m_histogram_{};
  std::atomic<uint64_t> errors_{0};
  std::atomic<uint64_t> service_ns_{0};
  std::atomic<uint64_t> total_ns_{0};
  std::atomic<uint64_t> kernel_ns_{0};
  std::thread thread_;

  std::vector<TraceEntry> trace_{kTraceCapacity};
  std::atomic<uint64_t> trace_head_{0};

 public:
  // Copies the most recent `capacity` entries (oldest first) into `out`;
  // returns the count copied. Racy by design against the producer -- the
  // newest slot may tear -- so the copy skips the in-flight slot.
  size_t trace_snapshot(TraceEntry* out, size_t capacity) const {
    const uint64_t head = trace_head_.load(std::memory_order_acquire);
    const uint64_t available = std::min<uint64_t>(head, kTraceCapacity);
    const uint64_t n = std::min<uint64_t>(available, capacity);
    for (uint64_t i = 0; i < n; ++i) {
      out[i] = trace_[(head - n + i) % kTraceCapacity];
    }
    return static_cast<size_t>(n);
  }

};
}  // namespace

extern "C" {

size_t sb_b70_abi_version(void) { return 2; }

sb_b70_provider_t* sb_b70_create(void) {
  try {
    return reinterpret_cast<sb_b70_provider_t*>(new B70Provider());
  } catch (...) {
    return nullptr;
  }
}

int sb_b70_load(sb_b70_provider_t* provider, const char* bank_path,
                uint64_t generation,
                const int32_t* resident_experts, size_t resident_count,
                size_t max_batch, const char* device_selector) {
  return sb_b70_load_v2(provider, bank_path, generation, resident_experts,
                        resident_count, max_batch, device_selector,
                        /*top_k=*/8);
}

int sb_b70_load_v2(sb_b70_provider_t* provider, const char* bank_path,
                   uint64_t generation,
                   const int32_t* resident_experts, size_t resident_count,
                   size_t max_batch, const char* device_selector,
                   size_t top_k) {
  if (!provider || !bank_path) return -1;
  auto* p = reinterpret_cast<B70Provider*>(provider);

  ProviderConfig config;
  config.generation = generation;
  config.max_batch = max_batch ? max_batch : 128;
  config.top_k = top_k;
  if (resident_experts && resident_count > 0) {
    config.resident_experts.assign(resident_experts,
                                   resident_experts + resident_count);
  }
  if (device_selector) {
    config.device_selector = device_selector;
  }

  return status_to_int(p->load(bank_path, config));
}

int sb_b70_issue(sb_b70_provider_t* provider, uint64_t generation,
                 uint64_t sequence, size_t layer,
                 const void* hidden_fp16,
                 const int32_t* ids, const float* weights, size_t M) {
  if (!provider) return -1;
  auto* p = reinterpret_cast<B70Provider*>(provider);

  return status_to_int(p->issue(
      generation, sequence, layer,
      reinterpret_cast<const sycl::half*>(hidden_fp16),
      ids, weights, M));
}

int sb_b70_take(sb_b70_provider_t* provider, uint64_t generation,
                uint64_t sequence, float* output, size_t output_elements) {
  if (!provider) return -1;
  auto* p = reinterpret_cast<B70Provider*>(provider);
  shooting_brake::phase1::DispatchResult result;
  return status_to_int(
      p->take(generation, sequence, output, output_elements, &result));
}

int sb_b70_out_fp16(sb_b70_provider_t* provider) {
  if (!provider) return -1;
  auto* p = reinterpret_cast<B70Provider*>(provider);
  return p->capability().output_fp16 ? 1 : 0;
}

int sb_b70_register_host(sb_b70_provider_t* provider, const void* ptr,
                         const size_t bytes) {
  // Imports a caller-owned host range into the device runtime
  // (prepare_for_device_copy) so H2D/D2H DMA directly instead of staging.
  // The poller rings already get this at registration; this export extends it
  // to the issue/take prefill path, whose buffers were previously CUDA-pinned
  // but Level-Zero-unknown. Unregistered on provider teardown.
  if (provider == nullptr || ptr == nullptr || bytes == 0) return -1;
  auto* p = reinterpret_cast<B70Provider*>(provider);
  return p->register_host_range(ptr, bytes) ? 0 : -1;
}

size_t sb_b70_num_resident(sb_b70_provider_t* provider) {
  if (!provider) return 0;
  auto* p = reinterpret_cast<B70Provider*>(provider);
  return p->capability().num_resident_experts;
}

int sb_b70_device_memory(sb_b70_provider_t* provider,
                         size_t* free_bytes, size_t* total_bytes) {
  if (!provider) return -1;
  auto* p = reinterpret_cast<B70Provider*>(provider);
  return p->device_memory(free_bytes, total_bytes) ? 0 : -1;
}

int sb_b70_health(sb_b70_provider_t* provider, sb_b70_health_t* health,
                  char* last_error, size_t last_error_size) {
  if (!provider || !health ||
      (last_error == nullptr && last_error_size != 0)) {
    return -1;
  }

  const auto snapshot = reinterpret_cast<B70Provider*>(provider)->health();
  health->generation = snapshot.generation;
  health->dispatches = snapshot.dispatches;
  health->allocations = snapshot.allocations;
  health->last_error_bytes = snapshot.last_error.size() + 1;
  health->loaded = snapshot.loaded ? 1u : 0u;
  health->pending = snapshot.pending ? 1u : 0u;
  health->stopped = snapshot.stopped ? 1u : 0u;
  health->reserved = 0;

  if (last_error == nullptr) {
    return 0;
  }
  if (last_error_size == 0) {
    return -2;
  }

  const size_t payload =
      std::min(snapshot.last_error.size(), last_error_size - 1);
  std::memcpy(last_error, snapshot.last_error.data(), payload);
  last_error[payload] = '\0';
  return last_error_size < health->last_error_bytes ? -2 : 0;
}

sb_b70_poller_t* sb_b70_poll_create(sb_b70_provider_t* provider,
                                    uint64_t generation) {
  if (!provider) return nullptr;
  try {
    return reinterpret_cast<sb_b70_poller_t*>(
        new B70Poller(reinterpret_cast<B70Provider*>(provider), generation));
  } catch (...) {
    return nullptr;
  }
}

int sb_b70_poll_register(sb_b70_poller_t* poller, size_t layer,
                         volatile uint32_t* signal,
                         volatile uint32_t* completion,
                         const void* hidden, const int32_t* ids,
                         const float* weights, float* output,
                         size_t topk) {
  if (!poller || !signal || !completion || !hidden || !ids || !weights ||
      !output || topk == 0) {
    return -1;
  }
  PollLayer entry;
  entry.layer = layer;
  entry.signal = signal;
  entry.completion = completion;
  entry.hidden = reinterpret_cast<const sycl::half*>(hidden);
  entry.ids = ids;
  entry.weights = weights;
  entry.output = output;
  // 1b: make the CUDA-pinned staging buffers DMA-able from the B70's SYCL
  // context. torch pin_memory=True registers them with the CUDA caching
  // host allocator only; from the Arc's side they are pageable, forcing a
  // staged H2D and a synchronous D2H on every doorbell round trip.
  // Measured (experiments/b70_dispatch_latency environment, clock-pinned):
  // 29.3 -> 20.5 us per dispatch at the production 180 us duty cycle.
  // Fail-open: an unregistered buffer still works, just slower.
  // Kill switch: SHOOTING_BRAKE_B70_XPU_REGISTER=0.
  static const bool xpu_register = [] {
    const char* flag = std::getenv("SHOOTING_BRAKE_B70_XPU_REGISTER");
    return flag == nullptr || std::strcmp(flag, "0") != 0;
  }();
  try {
    auto* wrapped = reinterpret_cast<B70Poller*>(poller);
    auto* prov = wrapped->provider();
    const auto cap = prov->capability();
    if (cap.supported_topk.size() != 1 ||
        cap.supported_topk.front() != topk) {
      return -1;
    }
    if (xpu_register) {
      const size_t max_batch = cap.max_batch_remote;
      const size_t hidden_elems = cap.supported_hidden_sizes.empty()
          ? 0
          : cap.supported_hidden_sizes.front();
      if (max_batch != 0 && hidden_elems != 0) {
        const bool ok =
            prov->register_host_range(
                hidden, max_batch * hidden_elems * sizeof(sycl::half)) &&
            prov->register_host_range(
                ids, max_batch * topk * sizeof(int32_t)) &&
            prov->register_host_range(
                weights, max_batch * topk * sizeof(float)) &&
            prov->register_host_range(
                output, max_batch * hidden_elems * sizeof(float));
        static std::atomic<bool> warned{false};
        if (!ok && !warned.exchange(true)) {
          std::fprintf(stderr,
                       "[sb_b70] XPU host registration unavailable; doorbell "
                       "staging stays pageable from the B70 side\n");
        }
      }
    }
    wrapped->add(entry);
  } catch (...) {
    return -1;
  }
  return 0;
}

int sb_b70_poll_start(sb_b70_poller_t* poller, int pin_cpu) {
  if (!poller) return -1;
  return reinterpret_cast<B70Poller*>(poller)->start(pin_cpu);
}

void sb_b70_poll_stop(sb_b70_poller_t* poller) {
  if (!poller) return;
  reinterpret_cast<B70Poller*>(poller)->stop();
}

void sb_b70_poll_reset(sb_b70_poller_t* poller) {
  if (!poller) return;
  reinterpret_cast<B70Poller*>(poller)->reset();
}

uint64_t sb_b70_poll_dispatch_count(sb_b70_poller_t* poller) {
  if (!poller) return 0;
  return reinterpret_cast<B70Poller*>(poller)->dispatch_count();
}

uint64_t sb_b70_poll_error_count(sb_b70_poller_t* poller) {
  if (!poller) return 0;
  return reinterpret_cast<B70Poller*>(poller)->error_count();
}

uint64_t sb_b70_poll_row_count(sb_b70_poller_t* poller) {
  if (!poller) return 0;
  return reinterpret_cast<B70Poller*>(poller)->row_count();
}

uint64_t sb_b70_poll_m_bucket_count(sb_b70_poller_t* poller, size_t bucket) {
  if (!poller) return 0;
  return reinterpret_cast<B70Poller*>(poller)->m_bucket_count(bucket);
}

uint64_t sb_b70_poll_service_ns(sb_b70_poller_t* poller) {
  if (!poller) return 0;
  return reinterpret_cast<B70Poller*>(poller)->service_ns();
}

uint64_t sb_b70_poll_total_ns(sb_b70_poller_t* poller) {
  if (!poller) return 0;
  return reinterpret_cast<B70Poller*>(poller)->total_ns();
}

uint64_t sb_b70_poll_kernel_ns(sb_b70_poller_t* poller) {
  if (!poller) return 0;
  return reinterpret_cast<B70Poller*>(poller)->kernel_ns();
}

size_t sb_b70_poll_trace_snapshot(sb_b70_poller_t* poller, void* out,
                                  size_t capacity_entries) {
  if (!poller || !out || capacity_entries == 0) return 0;
  return reinterpret_cast<B70Poller*>(poller)->trace_snapshot(
      reinterpret_cast<TraceEntry*>(out), capacity_entries);
}

void sb_b70_clock_reference(uint64_t* monotonic_ns, uint64_t* realtime_ns) {
  if (monotonic_ns) {
    *monotonic_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
  }
  if (realtime_ns) {
    *realtime_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
  }
}

void sb_b70_poll_destroy(sb_b70_poller_t* poller) {
  if (!poller) return;
  delete reinterpret_cast<B70Poller*>(poller);
}

void sb_b70_shutdown(sb_b70_provider_t* provider) {
  if (!provider) return;
  auto* p = reinterpret_cast<B70Provider*>(provider);
  p->shutdown();
}

void sb_b70_destroy(sb_b70_provider_t* provider) {
  if (!provider) return;
  delete reinterpret_cast<B70Provider*>(provider);
}

}  // extern "C"
