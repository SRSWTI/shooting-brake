#include "b70_capi.h"

#include "b70_provider.hpp"

#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <new>
#include <thread>
#include <vector>

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
  size_t topk;
};

class B70Poller {
 public:
  B70Poller(B70Provider* provider, uint64_t generation)
      : provider_(provider), generation_(generation) {}

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

  int start() {
    if (running_.exchange(true)) return 0;  // already running
    try {
      thread_ = std::thread([this] { loop(); });
    } catch (...) {
      running_ = false;
      return -1;
    }
    return 0;
  }

  void stop() {
    if (!running_.exchange(false)) return;
    if (thread_.joinable()) thread_.join();
  }

  uint64_t dispatch_count() const { return dispatches_.load(); }
  uint64_t error_count() const { return errors_.load(); }
  uint64_t service_ns() const { return service_ns_.load(); }
  // Nonzero only under SHOOTING_BRAKE_B70_PROFILE=1; the provider does not
  // timestamp commands otherwise.
  uint64_t kernel_ns() const { return kernel_ns_.load(); }

 private:
  void loop() {
    std::vector<PollLayer> snapshot;
    size_t known = 0;
    uint64_t sequence = 0;

    while (running_.load(std::memory_order_relaxed)) {
      // Refresh only when a new layer appeared, so the steady-state
      // sweep touches no lock.
      if (layer_count_.load(std::memory_order_acquire) != known) {
        std::lock_guard<std::mutex> guard(mutex_);
        snapshot = layers_;
        known = snapshot.size();
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
                                   static_cast<size_t>(M) * kHidden, &result);
        }
        if (status != ProviderStatus::ok) errors_.fetch_add(1);

        service_ns_.fetch_add(static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t0)
                .count()));

        // The provider only fills kernel_us on a profiled queue; it is 0
        // otherwise. Accumulating it here is what makes the round trip
        // self-describing: service_ns is the whole dispatch, so
        // service_ns - kernel_ns is submission and synchronisation
        // overhead, which is the part engineering can remove. Guessing that
        // split from the outside is how you end up optimising the wrong
        // half.
        if (result.kernel_us > 0.0) {
          kernel_ns_.fetch_add(
              static_cast<uint64_t>(result.kernel_us * 1000.0));
        }

        // Always release the waiter, even on failure: the CUDA side is
        // parked in cuStreamWaitValue32, which has no timeout.
        std::atomic_thread_fence(std::memory_order_seq_cst);
        entry.completion[0] = 1;
        dispatches_.fetch_add(1);
      }

      SB_SPIN_HINT();
    }
  }

  static constexpr size_t kHidden = 2048;

  B70Provider* provider_;
  uint64_t generation_;
  std::vector<PollLayer> layers_;
  std::mutex mutex_;
  std::atomic<size_t> layer_count_{0};
  std::atomic<bool> running_{false};
  std::atomic<uint64_t> dispatches_{0};
  std::atomic<uint64_t> errors_{0};
  std::atomic<uint64_t> service_ns_{0};
  std::atomic<uint64_t> kernel_ns_{0};
  std::thread thread_;

};
}  // namespace

extern "C" {

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
                size_t max_batch) {
  if (!provider || !bank_path) return -1;
  auto* p = reinterpret_cast<B70Provider*>(provider);

  ProviderConfig config;
  config.generation = generation;
  config.max_batch = max_batch ? max_batch : 128;
  config.top_k = 8;
  if (resident_experts && resident_count > 0) {
    config.resident_experts.assign(resident_experts,
                                   resident_experts + resident_count);
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

size_t sb_b70_num_resident(sb_b70_provider_t* provider) {
  if (!provider) return 0;
  auto* p = reinterpret_cast<B70Provider*>(provider);
  return p->capability().num_resident_experts;
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
  entry.topk = topk;
  try {
    reinterpret_cast<B70Poller*>(poller)->add(entry);
  } catch (...) {
    return -1;
  }
  return 0;
}

int sb_b70_poll_start(sb_b70_poller_t* poller) {
  if (!poller) return -1;
  return reinterpret_cast<B70Poller*>(poller)->start();
}

void sb_b70_poll_stop(sb_b70_poller_t* poller) {
  if (!poller) return;
  reinterpret_cast<B70Poller*>(poller)->stop();
}

uint64_t sb_b70_poll_dispatch_count(sb_b70_poller_t* poller) {
  if (!poller) return 0;
  return reinterpret_cast<B70Poller*>(poller)->dispatch_count();
}

uint64_t sb_b70_poll_error_count(sb_b70_poller_t* poller) {
  if (!poller) return 0;
  return reinterpret_cast<B70Poller*>(poller)->error_count();
}

uint64_t sb_b70_poll_service_ns(sb_b70_poller_t* poller) {
  if (!poller) return 0;
  return reinterpret_cast<B70Poller*>(poller)->service_ns();
}

uint64_t sb_b70_poll_kernel_ns(sb_b70_poller_t* poller) {
  if (!poller) return 0;
  return reinterpret_cast<B70Poller*>(poller)->kernel_ns();
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
