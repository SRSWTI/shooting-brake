// Immediate-zero B70 provider used only by the CUDA critical-path control.
//
// Build:
//   c++ -O3 -std=c++20 -fPIC -shared -pthread \
//     benchmarks/dummy_b70_provider.cpp -o benchmarks/libsb_b70_dummy.so
//
// Select explicitly with SHOOTING_BRAKE_B70_LIB=<absolute path>.  This library
// implements the complete ctypes ABI consumed by b70_binding.py, validates the
// canonical SBINT401 v2 header, preserves the native signal/completion protocol,
// and writes an all-zero remote partial.  Generated tokens are intentionally
// wrong.  The control is timing-only and must never be used for correctness.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define SB_DUMMY_SPIN_HINT() _mm_pause()
#else
#define SB_DUMMY_SPIN_HINT() ((void)0)
#endif

extern "C" {

typedef void sb_b70_provider_t;
typedef void sb_b70_poller_t;

typedef struct sb_b70_health {
  std::uint64_t generation;
  std::uint64_t dispatches;
  std::uint64_t allocations;
  std::uint64_t last_error_bytes;
  std::uint32_t loaded;
  std::uint32_t pending;
  std::uint32_t stopped;
  std::uint32_t reserved;
} sb_b70_health_t;

}  // extern "C"

static_assert(sizeof(sb_b70_health_t) == 48);

namespace {

constexpr std::array<char, 8> kMagic{'S', 'B', 'I', 'N', 'T', '4', '0', '1'};
constexpr std::uint32_t kVersion = 2;
constexpr std::size_t kPrefixBytes = 128;

std::uint32_t load_u32_le(const unsigned char* p) {
  return static_cast<std::uint32_t>(p[0]) |
         (static_cast<std::uint32_t>(p[1]) << 8) |
         (static_cast<std::uint32_t>(p[2]) << 16) |
         (static_cast<std::uint32_t>(p[3]) << 24);
}

std::uint64_t steady_ns() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

struct DummyProvider {
  std::uint64_t generation = 0;
  std::uint64_t dispatches = 0;
  std::size_t layers = 0;
  std::size_t resident_per_layer = 0;
  std::size_t hidden = 0;
  std::size_t max_batch = 0;
  bool loaded = false;
  bool stopped = false;
  std::string error;
  std::mutex mutex;

  int reject(std::string message) {
    error = std::move(message);
    return -1;
  }

  int load(const char* bank_path, std::uint64_t requested_generation,
           std::size_t requested_resident, std::size_t requested_max_batch) {
    if (!bank_path || requested_max_batch == 0) {
      return reject("dummy provider requires a bank path and nonzero max_batch");
    }
    std::ifstream input(bank_path, std::ios::binary);
    if (!input) return reject("dummy provider could not open the int4 bank");
    std::array<unsigned char, kPrefixBytes> prefix{};
    input.read(reinterpret_cast<char*>(prefix.data()), prefix.size());
    if (input.gcount() != static_cast<std::streamsize>(prefix.size())) {
      return reject("dummy provider found a truncated int4 header");
    }
    if (!std::equal(kMagic.begin(), kMagic.end(), prefix.begin())) {
      return reject("dummy provider accepts only canonical SBINT401 banks");
    }
    const std::uint32_t version = load_u32_le(prefix.data() + 8);
    const std::uint32_t num_layers = load_u32_le(prefix.data() + 16);
    const std::uint32_t experts_per_layer = load_u32_le(prefix.data() + 24);
    const std::uint32_t hidden_size = load_u32_le(prefix.data() + 36);
    const std::uint32_t intermediate = load_u32_le(prefix.data() + 40);
    const std::uint32_t group_size = load_u32_le(prefix.data() + 44);
    const std::uint32_t bits = load_u32_le(prefix.data() + 48);
    const std::uint32_t zero_point = load_u32_le(prefix.data() + 52);
    if (version != kVersion || num_layers == 0 || experts_per_layer == 0 ||
        hidden_size == 0 || intermediate == 0 || group_size != 128 ||
        bits != 4 || zero_point != 8) {
      return reject("dummy provider rejected unsupported SBINT401 geometry");
    }
    if (requested_resident != 0 && requested_resident != experts_per_layer) {
      return reject("dummy provider resident set disagrees with bank header");
    }
    generation = requested_generation;
    layers = num_layers;
    resident_per_layer = experts_per_layer;
    hidden = hidden_size;
    max_batch = requested_max_batch;
    dispatches = 0;
    loaded = true;
    stopped = false;
    error.clear();
    return 0;
  }
};

struct PollLayer {
  std::size_t layer;
  volatile std::uint32_t* signal;
  volatile std::uint32_t* completion;
  float* output;
  std::size_t topk;
};

struct TraceRecord {
  std::uint64_t sequence;
  std::size_t layer;
  std::uint32_t rows;
  std::uint64_t signal_observed_ns;
  std::uint64_t output_zeroed_ns;
  std::uint64_t completion_written_ns;
};

class DummyPoller {
 public:
  DummyPoller(DummyProvider* provider, std::uint64_t generation)
      : provider_(provider), generation_(generation) {
    const char* path = std::getenv("SHOOTING_BRAKE_DUMMY_TIMELINE");
    if (path && path[0]) {
      trace_path_ = path;
      traces_.reserve(1U << 20);
    }
  }

  ~DummyPoller() {
    stop();
    flush_trace();
  }

  void add(PollLayer entry) {
    std::lock_guard<std::mutex> guard(mutex_);
    layers_.push_back(entry);
    layer_count_.store(layers_.size(), std::memory_order_release);
  }

  int start() {
    if (generation_ != provider_->generation || !provider_->loaded) return -1;
    if (running_.exchange(true)) return 0;
    try {
      thread_ = std::thread([this] { loop(); });
    } catch (...) {
      running_.store(false);
      return -1;
    }
    return 0;
  }

  void stop() {
    if (!running_.exchange(false)) return;
    if (thread_.joinable()) thread_.join();
  }

  void reset() {
    dispatches_.store(0, std::memory_order_relaxed);
    rows_.store(0, std::memory_order_relaxed);
    errors_.store(0, std::memory_order_relaxed);
    service_ns_.store(0, std::memory_order_relaxed);
    for (auto& bucket : m_histogram_) bucket.store(0, std::memory_order_relaxed);
  }

  std::uint64_t dispatches() const { return dispatches_.load(); }
  std::uint64_t rows() const { return rows_.load(); }
  std::uint64_t errors() const { return errors_.load(); }
  std::uint64_t service_ns() const { return service_ns_.load(); }
  std::uint64_t bucket(std::size_t index) const {
    return index < m_histogram_.size() ? m_histogram_[index].load() : 0;
  }

 private:
  static std::size_t bucket_for(std::uint32_t rows) {
    if (rows <= 2) return rows - 1;
    if (rows <= 4) return 2;
    if (rows <= 8) return 3;
    if (rows <= 16) return 4;
    if (rows <= 32) return 5;
    return 6;
  }

  void loop() {
    std::vector<PollLayer> snapshot;
    std::size_t known = 0;
    std::uint64_t sequence = 0;
    while (running_.load(std::memory_order_relaxed)) {
      if (layer_count_.load(std::memory_order_acquire) != known) {
        std::lock_guard<std::mutex> guard(mutex_);
        snapshot = layers_;
        known = snapshot.size();
      }
      for (const PollLayer& entry : snapshot) {
        const std::uint32_t rows = entry.signal[0];
        if (rows == 0) continue;
        const std::uint64_t observed = steady_ns();
        entry.signal[0] = 0;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        ++sequence;
        if (rows > provider_->max_batch || entry.layer >= provider_->layers) {
          errors_.fetch_add(1, std::memory_order_relaxed);
        } else {
          std::memset(entry.output, 0,
                      static_cast<std::size_t>(rows) * provider_->hidden * sizeof(float));
        }
        const std::uint64_t zeroed = steady_ns();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        entry.completion[0] = 1;
        const std::uint64_t completed = steady_ns();
        service_ns_.fetch_add(completed - observed, std::memory_order_relaxed);
        rows_.fetch_add(rows, std::memory_order_relaxed);
        m_histogram_[bucket_for(rows)].fetch_add(1, std::memory_order_relaxed);
        dispatches_.fetch_add(1, std::memory_order_release);
        {
          std::lock_guard<std::mutex> guard(provider_->mutex);
          ++provider_->dispatches;
        }
        if (!trace_path_.empty() && traces_.size() < traces_.capacity()) {
          traces_.push_back({sequence, entry.layer, rows, observed, zeroed, completed});
        }
      }
      SB_DUMMY_SPIN_HINT();
    }
  }

  void flush_trace() {
    if (trace_path_.empty() || trace_flushed_) return;
    trace_flushed_ = true;
    std::ofstream output(trace_path_, std::ios::out | std::ios::trunc);
    if (!output) return;
    output << "sequence,layer,rows,signal_observed_ns,output_zeroed_ns,completion_written_ns\n";
    for (const TraceRecord& row : traces_) {
      output << row.sequence << ',' << row.layer << ',' << row.rows << ','
             << row.signal_observed_ns << ',' << row.output_zeroed_ns << ','
             << row.completion_written_ns << '\n';
    }
  }

  DummyProvider* provider_;
  std::uint64_t generation_;
  std::vector<PollLayer> layers_;
  std::mutex mutex_;
  std::atomic<std::size_t> layer_count_{0};
  std::atomic<bool> running_{false};
  std::thread thread_;
  std::atomic<std::uint64_t> dispatches_{0};
  std::atomic<std::uint64_t> rows_{0};
  std::atomic<std::uint64_t> errors_{0};
  std::atomic<std::uint64_t> service_ns_{0};
  std::array<std::atomic<std::uint64_t>, 7> m_histogram_{};
  std::string trace_path_;
  std::vector<TraceRecord> traces_;
  bool trace_flushed_ = false;
};

}  // namespace

extern "C" {

sb_b70_provider_t* sb_b70_create(void) {
  return reinterpret_cast<sb_b70_provider_t*>(new (std::nothrow) DummyProvider());
}

int sb_b70_load(sb_b70_provider_t* handle, const char* bank_path,
                std::uint64_t generation, const std::int32_t*,
                std::size_t resident_count, std::size_t max_batch) {
  if (!handle) return -1;
  return reinterpret_cast<DummyProvider*>(handle)->load(
      bank_path, generation, resident_count, max_batch);
}

int sb_b70_issue(sb_b70_provider_t* handle, std::uint64_t generation,
                 std::uint64_t, std::size_t layer, const void*,
                 const std::int32_t*, const float*, std::size_t rows) {
  if (!handle) return -1;
  auto* provider = reinterpret_cast<DummyProvider*>(handle);
  if (!provider->loaded || provider->generation != generation ||
      layer >= provider->layers || rows == 0 || rows > provider->max_batch) return -1;
  return 0;
}

int sb_b70_take(sb_b70_provider_t* handle, std::uint64_t generation,
                std::uint64_t, float* output, std::size_t output_elements) {
  if (!handle || !output) return -1;
  auto* provider = reinterpret_cast<DummyProvider*>(handle);
  if (!provider->loaded || provider->generation != generation ||
      output_elements % provider->hidden != 0) return -1;
  std::memset(output, 0, output_elements * sizeof(float));
  {
    std::lock_guard<std::mutex> guard(provider->mutex);
    ++provider->dispatches;
  }
  return 0;
}

std::size_t sb_b70_num_resident(sb_b70_provider_t* handle) {
  if (!handle) return 0;
  const auto* provider = reinterpret_cast<DummyProvider*>(handle);
  return provider->layers * provider->resident_per_layer;
}

int sb_b70_device_memory(sb_b70_provider_t*, std::size_t*, std::size_t*) {
  return -1;
}

int sb_b70_health(sb_b70_provider_t* handle, sb_b70_health_t* health,
                  char* last_error, std::size_t last_error_size) {
  if (!handle || !health || (!last_error && last_error_size != 0)) return -1;
  auto* provider = reinterpret_cast<DummyProvider*>(handle);
  std::lock_guard<std::mutex> guard(provider->mutex);
  health->generation = provider->generation;
  health->dispatches = provider->dispatches;
  health->allocations = 0;
  health->last_error_bytes = provider->error.size() + 1;
  health->loaded = provider->loaded ? 1U : 0U;
  health->pending = 0;
  health->stopped = provider->stopped ? 1U : 0U;
  health->reserved = 0;
  if (!last_error) return 0;
  if (last_error_size == 0) return -2;
  const std::size_t copied = std::min(provider->error.size(), last_error_size - 1);
  std::memcpy(last_error, provider->error.data(), copied);
  last_error[copied] = '\0';
  return last_error_size < health->last_error_bytes ? -2 : 0;
}

sb_b70_poller_t* sb_b70_poll_create(sb_b70_provider_t* handle,
                                    std::uint64_t generation) {
  if (!handle) return nullptr;
  return reinterpret_cast<sb_b70_poller_t*>(new (std::nothrow) DummyPoller(
      reinterpret_cast<DummyProvider*>(handle), generation));
}

int sb_b70_poll_register(sb_b70_poller_t* handle, std::size_t layer,
                         volatile std::uint32_t* signal,
                         volatile std::uint32_t* completion, const void*,
                         const std::int32_t*, const float*, float* output,
                         std::size_t topk) {
  if (!handle || !signal || !completion || !output || topk == 0) return -1;
  reinterpret_cast<DummyPoller*>(handle)->add(
      {layer, signal, completion, output, topk});
  return 0;
}

int sb_b70_poll_start(sb_b70_poller_t* handle) {
  return handle ? reinterpret_cast<DummyPoller*>(handle)->start() : -1;
}
void sb_b70_poll_stop(sb_b70_poller_t* handle) {
  if (handle) reinterpret_cast<DummyPoller*>(handle)->stop();
}
void sb_b70_poll_reset(sb_b70_poller_t* handle) {
  if (handle) reinterpret_cast<DummyPoller*>(handle)->reset();
}
std::uint64_t sb_b70_poll_dispatch_count(sb_b70_poller_t* handle) {
  return handle ? reinterpret_cast<DummyPoller*>(handle)->dispatches() : 0;
}
std::uint64_t sb_b70_poll_row_count(sb_b70_poller_t* handle) {
  return handle ? reinterpret_cast<DummyPoller*>(handle)->rows() : 0;
}
std::uint64_t sb_b70_poll_error_count(sb_b70_poller_t* handle) {
  return handle ? reinterpret_cast<DummyPoller*>(handle)->errors() : 0;
}
std::uint64_t sb_b70_poll_service_ns(sb_b70_poller_t* handle) {
  return handle ? reinterpret_cast<DummyPoller*>(handle)->service_ns() : 0;
}
std::uint64_t sb_b70_poll_total_ns(sb_b70_poller_t*) { return 0; }
std::uint64_t sb_b70_poll_kernel_ns(sb_b70_poller_t*) { return 0; }
std::uint64_t sb_b70_poll_m_bucket_count(sb_b70_poller_t* handle,
                                         std::size_t bucket) {
  return handle ? reinterpret_cast<DummyPoller*>(handle)->bucket(bucket) : 0;
}
void sb_b70_poll_destroy(sb_b70_poller_t* handle) {
  delete reinterpret_cast<DummyPoller*>(handle);
}

void sb_b70_shutdown(sb_b70_provider_t* handle) {
  if (!handle) return;
  auto* provider = reinterpret_cast<DummyProvider*>(handle);
  provider->loaded = false;
  provider->stopped = true;
}
void sb_b70_destroy(sb_b70_provider_t* handle) {
  delete reinterpret_cast<DummyProvider*>(handle);
}

}  // extern "C"
