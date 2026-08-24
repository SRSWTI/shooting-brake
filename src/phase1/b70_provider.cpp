#include "b70_provider.hpp"

#include "grouped_moe.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>
#include <sstream>
#include <string>
#include <utility>

#if defined(__x86_64__)
#include <immintrin.h>
#define SB_PROVIDER_SPIN_HINT() _mm_pause()
#else
#define SB_PROVIDER_SPIN_HINT() ((void)0)
#endif

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "quixicore/xpu/ops.hpp"
#include "quixicore/xpu/runtime.hpp"

namespace shooting_brake::phase1 {
namespace {

// Expert-bank geometry is adopted at load time. The NVFP4 and int4 banks
// describe different source expert counts and resident layouts. The NVFP4
// expert-ID domain has a fixed upper bound; routing row width is runtime
// ProviderConfig state.
constexpr std::size_t kNvfp4ExpertsPerLayer = 256;
constexpr std::size_t kInt4Alignment = 4096;
constexpr std::uint32_t kInt4Version = 2;

enum class BankFormat {
  nvfp4,
  int4,
};

std::size_t g_layers = 0;
std::size_t g_hidden = 0;
std::size_t g_intermediate = 0;
std::size_t g_source_experts_per_layer = 0;
std::size_t g_total_experts = 0;
std::size_t g_w13_bytes = 0;
std::size_t g_s13_bytes = 0;
std::size_t g_w2_bytes = 0;
std::size_t g_s2_bytes = 0;
std::size_t g_expert_bytes = 0;

// Derives every dependent NVFP4 size from the header. Keeping this function's
// arithmetic unchanged preserves the existing SBEXP001 reader and upload
// contract.
//
// experts_per_layer is adopted from the bank rather than pinned to 256:
// the 99B carries 205. kNvfp4ExpertsPerLayer only bounds the source expert-ID
// domain and its resident-set bitmap; ProviderConfig::top_k sets row width.
bool adopt_nvfp4_bank_geometry(std::size_t layers,
                               std::size_t experts_per_layer,
                               std::size_t hidden,
                               std::size_t intermediate,
                               std::uint64_t w13, std::uint64_t s13,
                               std::uint64_t w2, std::uint64_t s2) noexcept {
  if (layers == 0 || hidden == 0 || intermediate == 0 ||
      experts_per_layer == 0 ||
      experts_per_layer > kNvfp4ExpertsPerLayer || hidden % 16 != 0 ||
      intermediate % 16 != 0) {
    return false;
  }
  const std::size_t w13_bytes = 2 * intermediate * (hidden / 2);
  const std::size_t s13_bytes = 2 * intermediate * (hidden / 16);
  const std::size_t w2_bytes = hidden * (intermediate / 2);
  const std::size_t s2_bytes = hidden * (intermediate / 16);
  if (w13 != w13_bytes || s13 != s13_bytes || w2 != w2_bytes ||
      s2 != s2_bytes) {
    return false;
  }
  g_layers = layers;
  g_hidden = hidden;
  g_intermediate = intermediate;
  g_source_experts_per_layer = experts_per_layer;
  g_total_experts = layers * experts_per_layer;
  g_w13_bytes = w13_bytes;
  g_s13_bytes = s13_bytes;
  g_w2_bytes = w2_bytes;
  g_s2_bytes = s2_bytes;
  g_expert_bytes =
      w13_bytes + s13_bytes + w2_bytes + s2_bytes + 2 * sizeof(float);
  return true;
}

bool adopt_int4_bank_geometry(const std::size_t layers,
                              const std::size_t source_experts_per_layer,
                              const std::size_t hidden,
                              const std::size_t intermediate) noexcept {
  if (layers == 0 || source_experts_per_layer == 0 || hidden == 0 ||
      intermediate == 0 || hidden % 8 != 0 || intermediate % 8 != 0) {
    return false;
  }
  g_layers = layers;
  g_hidden = hidden;
  g_intermediate = intermediate;
  g_source_experts_per_layer = source_experts_per_layer;
  g_total_experts = 0;
  g_w13_bytes = 0;
  g_s13_bytes = 0;
  g_w2_bytes = 0;
  g_s2_bytes = 0;
  g_expert_bytes = 0;
  return true;
}

// Per-dispatch profiling costs real latency: a profiled Level Zero
// queue timestamps every command, and the kernel_us figure additionally
// needs two empty marker kernels bracketing the real one, each a full
// submission. Off by default; the isolated Phase 1 tests turn it on.
bool profiling_requested() noexcept {
  const char* value = std::getenv("SHOOTING_BRAKE_B70_PROFILE");
  return value != nullptr && value[0] == '1' && value[1] == '\0';
}

bool int4_requested() noexcept {
  const char* value = std::getenv("SHOOTING_BRAKE_B70_INT4");
  return value != nullptr && value[0] == '1' && value[1] == '\0';
}

// Grouped prefill: read each resident expert ONCE per layer instead of once
// per route. OFF by default because it is new; the per-route GEMV stays the
// only path that has served production traffic.
//
// Measured 2026-08-21 standalone at r15 geometry on real bank bytes
// (src/phase7/xe2_probe/xe2_grouped_moe): 2.355 ms/layer = 216 us/token
// against the 1,705 us/token the GEMV path measures in vLLM, i.e. ~7.9x on
// the B70 GEMM leg and ~4.7x end-to-end with B70 at 86-92% of TTFT.
//
// Decode deliberately keeps the GEMV: at M=1 each token already touches its
// experts once, so there is no read amplification to remove and grouping would
// only add permutation work.
bool grouped_requested() noexcept {
  const char* value = std::getenv("SHOOTING_BRAKE_B70_GROUPED");
  return value != nullptr && value[0] == '1' && value[1] == '\0';
}

// take() can spin on the completion event before falling back to the
// blocking wait. OFF by default — it was MEASURED AND IT DOES NOTHING.
//
// The synthetic probe said it should win: at the production decode shape a
// blocking wait means 9.46 us vs 6.37 us spinning, i.e. 3.09 us/dispatch,
// which at 48 dispatches/step predicted ~0.15 ms/step (~1.2% ITL)
// [experiments/b13_wait_probe.cpp].
//
// End-to-end on the 88B it delivered nothing (2026-08-19, same build, same
// boot config, 4 ITL runs per arm, Gen4 B70, split:54):
//
//     spin OFF : 11.5098 ms ITL      spin ON : 11.5130 ms ITL
//
// +0.003 ms, against a run-to-run spread of +-0.06 ms. Noise.
//
// The probe over-predicted because it submitted and waited IMMEDIATELY,
// which maximises the chance of landing in the runtime's sleep path. In
// production the poller does issue(), bookkeeping, then take(), and by then
// the wait behaves differently. Lesson: a synthetic probe's timing pattern
// is part of what it measures.
//
// Kept behind a flag rather than deleted, per kill-bench rule 3 — the
// negative result is the product, and this stops the idea being re-proposed.
// Set SHOOTING_BRAKE_B70_SPIN_WAIT=1 to re-enable and re-measure.
bool spin_wait_enabled() noexcept {
  const char* value = std::getenv("SHOOTING_BRAKE_B70_SPIN_WAIT");
  return value != nullptr && value[0] == '1' && value[1] == '\0';
}

#pragma pack(push, 1)
struct ExpertBankHeader {
  char magic[8];
  std::uint32_t num_layers;
  std::uint32_t experts_per_layer;
  std::uint32_t hidden_size;
  std::uint32_t intermediate_size;
  std::uint32_t reserved;
  std::uint64_t w13_bytes;
  std::uint64_t s13_bytes;
  std::uint64_t w2_bytes;
  std::uint64_t s2_bytes;
};

// Authoritative definition:
// src/phase4/src/shooting_brake_vllm/int4_bank_format.py
// Any field-order, size, or semantic change there requires a version bump and
// a matching update here. The explicit duplication is intentional: a
// Python-only source of truth cannot protect this C++ mmap reader from drift.
struct Int4BankHeaderPrefix {
  char magic[8];
  std::uint32_t version;
  std::uint32_t data_offset;
  std::uint32_t num_layers;
  std::uint32_t source_num_layers;
  std::uint32_t experts_per_layer;
  std::uint32_t source_experts_per_layer;
  std::uint32_t resident_set_shared_across_layers;
  std::uint32_t hidden;
  std::uint32_t moe_intermediate;
  std::uint32_t group_size;
  std::uint32_t bits;
  std::uint32_t zero_point;
  std::uint32_t reserved0;
  std::uint32_t reserved1;
  std::uint32_t gate_q_offset;
  std::uint32_t gate_q_size;
  std::uint32_t gate_s_offset;
  std::uint32_t gate_s_size;
  std::uint32_t up_q_offset;
  std::uint32_t up_q_size;
  std::uint32_t up_s_offset;
  std::uint32_t up_s_size;
  std::uint32_t down_q_offset;
  std::uint32_t down_q_size;
  std::uint32_t down_s_offset;
  std::uint32_t down_s_size;
  std::uint64_t expert_stride_bytes;
  std::uint64_t layer_stride_bytes;
};
#pragma pack(pop)

static_assert(sizeof(ExpertBankHeader) == 60,
              "the packed expert-bank header must be 60 bytes");
static_assert(sizeof(Int4BankHeaderPrefix) == 128,
              "the packed SBINT401 v2 fixed prefix must be 128 bytes");

std::uint64_t expected_nvfp4_file_bytes() noexcept {
  return sizeof(ExpertBankHeader) +
         static_cast<std::uint64_t>(g_total_experts) * g_expert_bytes;
}

std::uint64_t nvfp4_weight_bytes_per_expert() noexcept {
  return g_w13_bytes + g_s13_bytes + g_w2_bytes + g_s2_bytes +
         2 * sizeof(float);
}

std::uint64_t persistent_device_bytes(
    const BankFormat format, const std::size_t max_batch,
    const std::size_t top_k, const std::size_t resident_experts,
    const std::uint64_t int4_expert_stride = 0) {
  const std::size_t scratch_intermediates =
      format == BankFormat::int4 ? 1 : 2;
  const std::uint64_t bytes_per_batch_row =
      g_hidden * sizeof(sycl::half) +
      top_k * sizeof(std::int32_t) +
      top_k * sizeof(float) +
      top_k * scratch_intermediates * g_intermediate * sizeof(float) +
      g_hidden * sizeof(float);
  const std::uint64_t bytes_per_expert =
      format == BankFormat::int4 ? int4_expert_stride
                                 : nvfp4_weight_bytes_per_expert();
  const std::uint64_t resident_weight_bytes =
      static_cast<std::uint64_t>(g_layers) * resident_experts *
      bytes_per_expert;
  return resident_weight_bytes +
         static_cast<std::uint64_t>(max_batch) * bytes_per_batch_row;
}

std::string errno_message(const char* operation) {
  const int saved_errno = errno;
  return std::string(operation) + ": " + std::strerror(saved_errno);
}

std::string lowercase(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

struct SelectedDevice {
  sycl::device device;
  std::string name;
  std::string pci_bdf;
  std::size_t index = 0;
};

SelectedDevice select_b70(const std::string& selector) {
  std::vector<SelectedDevice> level_zero_devices;
  std::vector<SelectedDevice> fallback_devices;

  for (const auto& platform : sycl::platform::get_platforms()) {
    const bool level_zero =
        platform.get_backend() == sycl::backend::ext_oneapi_level_zero;
    for (const auto& device : platform.get_devices()) {
      if (!device.is_gpu() ||
          device.get_info<sycl::info::device::vendor_id>() != 0x8086u) {
        continue;
      }

      std::string name = device.get_info<sycl::info::device::name>();
      if (lowercase(name).find("b70") == std::string::npos) {
        continue;
      }
      if (!device.has(sycl::aspect::ext_intel_pci_address)) {
        throw std::runtime_error("B70 device '" + name +
                                 "' does not expose its PCI address");
      }
      std::string pci_bdf =
          device.get_info<sycl::ext::intel::info::device::pci_address>();
      pci_bdf = lowercase(std::move(pci_bdf));
      auto& destination =
          level_zero ? level_zero_devices : fallback_devices;
      destination.push_back(
          SelectedDevice{device, std::move(name), std::move(pci_bdf), 0});
    }
  }

  std::vector<SelectedDevice>& devices =
      level_zero_devices.empty() ? fallback_devices : level_zero_devices;
  for (std::size_t index = 0; index < devices.size(); ++index) {
    devices[index].index = index;
  }
  if (devices.empty()) {
    throw std::runtime_error(
        "no Intel GPU whose device name contains B70 was found");
  }

  const auto available_devices = [&devices]() {
    std::ostringstream message;
    for (std::size_t index = 0; index < devices.size(); ++index) {
      if (index != 0) {
        message << ", ";
      }
      message << index << "=" << devices[index].pci_bdf;
    }
    return message.str();
  };
  if (selector.empty()) {
    return std::move(devices.front());
  }

  const bool numeric = std::all_of(
      selector.begin(), selector.end(),
      [](const unsigned char c) { return std::isdigit(c) != 0; });
  if (numeric) {
    std::size_t requested_index = 0;
    const auto result = std::from_chars(
        selector.data(), selector.data() + selector.size(), requested_index);
    if (result.ec == std::errc{} &&
        result.ptr == selector.data() + selector.size() &&
        requested_index < devices.size()) {
      return std::move(devices[requested_index]);
    }
  } else {
    const std::string requested_bdf = lowercase(selector);
    const auto match = std::find_if(
        devices.begin(), devices.end(), [&requested_bdf](const auto& device) {
          return device.pci_bdf == requested_bdf;
        });
    if (match != devices.end()) {
      return std::move(*match);
    }
  }

  throw std::invalid_argument(
      "B70 device selector '" + selector +
      "' did not match a discovered device; available: " +
      available_devices());
}


}  // namespace

struct B70Provider::Impl {
  mutable std::mutex mutex;
  mutable std::mutex async_mutex;

  ProviderConfig config;
  Capability capability;
  Health health;

  int bank_fd = -1;
  void* bank_mapping = MAP_FAILED;
  std::size_t bank_mapping_bytes = 0;
  BankFormat bank_format = BankFormat::nvfp4;
  Int4BankHeaderPrefix int4_header{};
  std::vector<std::int32_t> int4_source_expert_ids;

  std::optional<sycl::queue> queue;
  std::string async_error;

  // Host ranges imported into the queue's context via
  // prepare_for_device_copy; released in release_resources_locked while
  // the context still exists.
  std::vector<std::pair<void*, std::size_t>> registered_host_ranges;

  // SBEXP001 uses the six legacy SoA allocations below. SBINT401 uses one
  // contiguous AoS allocation so the six expert-0 plane pointers can share
  // the header's expert stride without repacking.
  std::uint8_t* int4_records = nullptr;
  std::uint8_t* w13 = nullptr;
  std::uint8_t* s13 = nullptr;
  std::uint8_t* w2 = nullptr;
  std::uint8_t* s2 = nullptr;
  float* w13_global = nullptr;
  float* w2_global = nullptr;

  sycl::half* hidden = nullptr;
  std::int32_t* ids = nullptr;
  float* weights = nullptr;
  float* scratch = nullptr;
  float* output = nullptr;
  std::uint8_t* upload_staging = nullptr;
  float* copyout_staging = nullptr;

  // fp16 result wire (SHOOTING_BRAKE_B70_OUT_FP16=1). Both B70s hang off one
  // shared Gen4 x4 uplink -- 6.44 GB/s aggregate concurrent against 9.7 solo,
  // measured by experiments/b70_mem_topology_probe -- so the result copy is
  // priced in PCIe, not VRAM. At M=2048 the fp32 output is 25.2 MiB per card
  // per layer against 12.6 MiB of inbound activations, which makes it the
  // largest single term in prefill: 175 of 262 us/token.
  sycl::half* out16 = nullptr;
  sycl::half* copyout_staging16 = nullptr;
  bool out_fp16 = false;

  // Grouped prefill path (SHOOTING_BRAKE_B70_GROUPED=1). The per-route GEMV
  // reads each expert's 5.06 MiB once PER ROUTE; at M=2560 over 85 resident
  // experts that is ~30x more traffic than the weights require, and the split
  // kernel is already at 437 GB/s of the card's ~510, so the waste is bytes
  // rather than rate. These buffers hold the expert-major permutation that
  // lets each expert be read once instead.
  //
  // Sized off max_batch * top_k, the worst case where every route is distinct.
  sycl::half* g_act = nullptr;    // [routes, hidden]   gathered, expert-major
  float* g_mid = nullptr;         // [routes, 2*inter]  w13 output, fp32:
                                  // the un-scaled dot product overflows fp16
  sycl::half* g_gated = nullptr;  // [routes, inter]    after SwiGLU
  float* g_outr = nullptr;        // [routes, hidden]   w2 output, fp32
  sycl::half* g_bias13 = nullptr; // zeros; the kernel applies bias uncondit.
  sycl::half* g_bias2 = nullptr;
  std::int32_t* g_hist = nullptr;    // [experts+1] routes per expert
  std::int32_t* g_offs = nullptr;    // [experts+1] exclusive prefix sum
  std::int32_t* g_cursor = nullptr;  // [experts]   scatter write cursor
  std::int32_t* g_rows = nullptr;    // [experts]   rows_per_expert for the GEMM
  std::int32_t* g_slot_row = nullptr;  // [routes] slot -> source token
  std::int32_t* g_slot_exp = nullptr;  // [routes] slot -> expert
  float* g_slot_w = nullptr;           // [routes] slot -> router weight
  std::int32_t* g_atomic = nullptr;    // grouped GEMM persistent work counter
  bool grouped_ready = false;
  // Pipelining geometry (SHOOTING_BRAKE_B70_PIPELINE=<nchunks>, default 1).
  // Phase 1 is plumbing only: chunks run sequentially on the one in-order
  // queue, and nchunks=1 reproduces the old layout exactly -- chunk 0's view
  // IS the whole pool at offset zero. The second queue is Phase 2.
  int pipeline_chunks = 1;
  std::size_t pl_chunk_M = 0;       // row capacity per chunk
  std::size_t pl_chunk_routes = 0;  // route capacity per chunk
  std::size_t pl_experts = 0;       // stride basis for hist/offs/cursor/rows

  std::optional<sycl::event> dispatch_begin;
  std::optional<sycl::event> kernel_begin;
  std::optional<sycl::event> kernel_end;
  std::optional<sycl::event> copy_out;
  bool profiling = false;
  bool spin_wait = true;

  std::uint64_t pending_generation = 0;
  std::uint64_t pending_sequence = 0;
  std::size_t pending_M = 0;
  bool pending_split = false;
  std::string pending_error;

#ifdef SHOOTING_BRAKE_ENABLE_TEST_FAULTS
  std::optional<ProviderTestFault> armed_test_fault;
  std::uint64_t armed_test_fault_sequence = 0;
#endif

  Impl() {
    capability.protocol_version = 1;
    capability.backend = "quixicore-xpu-nvfp4";
  }

  void set_error(const std::string& message) noexcept {
    try {
      health.last_error = message;
    } catch (...) {
    }
  }

  void set_current_exception(const char* prefix) noexcept {
    try {
      throw;
    } catch (const std::exception& error) {
      try {
        set_error(std::string(prefix) + error.what());
      } catch (...) {
        set_error(prefix);
      }
    } catch (...) {
      set_error(prefix);
    }
  }

  void capture_async_errors(sycl::exception_list errors) noexcept {
    try {
      std::lock_guard<std::mutex> lock(async_mutex);
      for (const std::exception_ptr& error : errors) {
        try {
          std::rethrow_exception(error);
        } catch (const std::exception& exception) {
          if (!async_error.empty()) {
            async_error += "; ";
          }
          async_error += exception.what();
        } catch (...) {
          if (!async_error.empty()) {
            async_error += "; ";
          }
          async_error += "unknown asynchronous SYCL error";
        }
      }
    } catch (...) {
    }
  }

  std::string consume_async_error() {
    std::lock_guard<std::mutex> lock(async_mutex);
    std::string result = std::move(async_error);
    async_error.clear();
    return result;
  }

  template <typename T>
  T* allocate_device(const std::size_t count) {
    ++health.allocations;
    T* pointer = sycl::malloc_device<T>(count, *queue);
    if (pointer == nullptr) {
      throw std::bad_alloc();
    }
    return pointer;
  }

  template <typename T>
  T* allocate_host(const std::size_t count) {
    ++health.allocations;
    T* pointer = sycl::malloc_host<T>(count, *queue);
    if (pointer == nullptr) {
      throw std::bad_alloc();
    }
    return pointer;
  }

  void retire_pending_locked() noexcept {
    health.pending = false;
    pending_generation = 0;
    pending_sequence = 0;
    pending_M = 0;
    pending_split = false;
    pending_error.clear();
    dispatch_begin.reset();
    kernel_begin.reset();
    kernel_end.reset();
    copy_out.reset();
  }

  void release_resources_locked(const bool mark_stopped) noexcept {
    if (queue) {
      try {
        queue->wait_and_throw();
      } catch (...) {
      }

      auto free_pointer = [this](auto*& pointer) noexcept {
        if (pointer == nullptr) {
          return;
        }
        auto* saved = pointer;
        pointer = nullptr;
        try {
          sycl::free(saved, *queue);
        } catch (...) {
        }
      };

      free_pointer(upload_staging);
      free_pointer(copyout_staging);
      free_pointer(out16);
      free_pointer(copyout_staging16);
      free_pointer(output);
      free_pointer(scratch);
      free_pointer(weights);
      free_pointer(ids);
      free_pointer(hidden);
      free_pointer(int4_records);
      free_pointer(w2_global);
      free_pointer(w13_global);
      free_pointer(s2);
      free_pointer(w2);
      free_pointer(s13);
      free_pointer(w13);
      free_pointer(g_atomic);
      free_pointer(g_slot_w);
      free_pointer(g_slot_exp);
      free_pointer(g_slot_row);
      free_pointer(g_rows);
      free_pointer(g_cursor);
      free_pointer(g_offs);
      free_pointer(g_hist);
      free_pointer(g_bias2);
      free_pointer(g_bias13);
      free_pointer(g_outr);
      free_pointer(g_gated);
      free_pointer(g_mid);
      free_pointer(g_act);

      for (auto& range : registered_host_ranges) {
        try {
          sycl::ext::oneapi::experimental::release_from_device_copy(
              range.first, queue->get_context());
        } catch (...) {
        }
      }
    }
    registered_host_ranges.clear();

    dispatch_begin.reset();
    kernel_begin.reset();
    kernel_end.reset();
    copy_out.reset();
    queue.reset();
    try {
      std::lock_guard<std::mutex> async_lock(async_mutex);
      async_error.clear();
    } catch (...) {
    }

    if (bank_mapping != MAP_FAILED) {
      ::munmap(bank_mapping, bank_mapping_bytes);
      bank_mapping = MAP_FAILED;
      bank_mapping_bytes = 0;
    }
    if (bank_fd >= 0) {
      ::close(bank_fd);
      bank_fd = -1;
    }

    health.loaded = false;
    health.pending = false;
    if (mark_stopped) {
      health.stopped = true;
    }
    pending_generation = 0;
    pending_sequence = 0;
    pending_M = 0;
    pending_split = false;
    pending_error.clear();
#ifdef SHOOTING_BRAKE_ENABLE_TEST_FAULTS
    armed_test_fault.reset();
    armed_test_fault_sequence = 0;
#endif
  }

  ProviderStatus reject_load_locked(const ProviderStatus status,
                                    const std::string& message) noexcept {
    set_error(message);
    release_resources_locked(false);
    return status;
  }

  void upload_plane(const std::uint8_t* records, const std::size_t record_offset,
                    const std::size_t bytes_per_expert, std::uint8_t* destination,
                    const std::size_t layer) {
    const std::size_t resident_count = config.resident_experts.size();
    const std::size_t source_layer = layer * g_source_experts_per_layer;
    const std::size_t destination_layer = layer * resident_count;
    for (std::size_t local_expert = 0; local_expert < resident_count;
         ++local_expert) {
      const std::size_t canonical_expert =
          static_cast<std::size_t>(config.resident_experts[local_expert]);
      const std::uint8_t* source =
          records + (source_layer + canonical_expert) * g_expert_bytes +
          record_offset;
      std::memcpy(upload_staging + local_expert * bytes_per_expert, source,
                  bytes_per_expert);
    }
    queue
        ->memcpy(destination + destination_layer * bytes_per_expert,
                 upload_staging, resident_count * bytes_per_expert)
        .wait_and_throw();
  }

  void upload_globals(const std::uint8_t* records,
                      const std::size_t record_offset, float* destination,
                      const std::size_t layer) {
    const std::size_t resident_count = config.resident_experts.size();
    const std::size_t source_layer = layer * g_source_experts_per_layer;
    const std::size_t destination_layer = layer * resident_count;
    for (std::size_t local_expert = 0; local_expert < resident_count;
         ++local_expert) {
      const std::size_t canonical_expert =
          static_cast<std::size_t>(config.resident_experts[local_expert]);
      const std::uint8_t* source =
          records + (source_layer + canonical_expert) * g_expert_bytes +
          record_offset;
      std::memcpy(upload_staging + local_expert * sizeof(float), source,
                  sizeof(float));
    }
    queue
        ->memcpy(destination + destination_layer, upload_staging,
                 resident_count * sizeof(float))
        .wait_and_throw();
  }
};

B70Provider::B70Provider() : impl_(std::make_unique<Impl>()) {}

B70Provider::~B70Provider() { shutdown(); }

ProviderStatus B70Provider::load(const std::string& bank_path,
                                 const ProviderConfig& config) {
  try {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    if (impl_->health.stopped) {
      return ProviderStatus::shutdown;
    }
    if (impl_->health.loaded || impl_->health.pending) {
      impl_->set_error("the provider is already loaded");
      return ProviderStatus::busy;
    }
    if (bank_path.empty()) {
      impl_->set_error("bank_path must not be empty");
      return ProviderStatus::invalid_argument;
    }
    if (config.max_batch == 0 ||
        config.max_batch > std::numeric_limits<std::uint32_t>::max() ||
        config.top_k == 0 ||
        config.top_k > std::numeric_limits<std::uint32_t>::max() ||
        config.generation == 0) {
      impl_->set_error(
          "config requires 0 < max_batch <= UINT32_MAX, "
          "0 < top_k <= UINT32_MAX, and generation > 0");
      return ProviderStatus::invalid_argument;
    }

    impl_->bank_fd = ::open(bank_path.c_str(), O_RDONLY | O_CLOEXEC);
    if (impl_->bank_fd < 0) {
      return impl_->reject_load_locked(ProviderStatus::device_error,
                                       errno_message("open expert bank"));
    }

    struct stat file_stat {};
    if (::fstat(impl_->bank_fd, &file_stat) != 0) {
      return impl_->reject_load_locked(ProviderStatus::device_error,
                                       errno_message("fstat expert bank"));
    }
    if (!S_ISREG(file_stat.st_mode) || file_stat.st_size < 0 ||
        static_cast<std::uint64_t>(file_stat.st_size) <
            sizeof(ExpertBankHeader)) {
      return impl_->reject_load_locked(
          ProviderStatus::invalid_argument,
          "expert bank is not a regular file large enough to hold a header");
    }

    impl_->bank_mapping_bytes = static_cast<std::size_t>(file_stat.st_size);
    impl_->bank_mapping =
        ::mmap(nullptr, impl_->bank_mapping_bytes, PROT_READ, MAP_PRIVATE,
               impl_->bank_fd, 0);
    if (impl_->bank_mapping == MAP_FAILED) {
      impl_->bank_mapping_bytes = 0;
      return impl_->reject_load_locked(ProviderStatus::device_error,
                                       errno_message("mmap expert bank"));
    }

    const auto* bank_bytes =
        static_cast<const std::uint8_t*>(impl_->bank_mapping);
    const bool is_nvfp4 = std::memcmp(bank_bytes, "SBEXP001", 8) == 0;
    const bool is_int4 = std::memcmp(bank_bytes, "SBINT401", 8) == 0;
    if (!is_nvfp4 && !is_int4) {
      std::string found(reinterpret_cast<const char*>(bank_bytes), 8);
      for (char& c : found) {
        if (!std::isprint(static_cast<unsigned char>(c))) {
          c = '?';
        }
      }
      return impl_->reject_load_locked(
          ProviderStatus::invalid_argument,
          "expert bank magic '" + found +
              "' is unsupported; expected 'SBEXP001' or 'SBINT401'");
    }

    std::vector<std::int32_t> resident_experts;
    std::uint64_t int4_expert_stride = 0;
    const std::uint8_t* bank_records = nullptr;
    if (is_nvfp4) {
      impl_->bank_format = BankFormat::nvfp4;
      ExpertBankHeader header{};
      std::memcpy(&header, bank_bytes, sizeof(header));
      if (header.reserved != 0) {
        return impl_->reject_load_locked(
            ProviderStatus::invalid_argument,
            "expert bank header does not match the Phase-1 NVFP4 B70 contract");
      }
      if (!adopt_nvfp4_bank_geometry(
              header.num_layers, header.experts_per_layer, header.hidden_size,
              header.intermediate_size, header.w13_bytes, header.s13_bytes,
              header.w2_bytes, header.s2_bytes)) {
        return impl_->reject_load_locked(
            ProviderStatus::invalid_argument,
            "expert bank header declares an inconsistent or unsupported "
            "geometry");
      }
      if (static_cast<std::uint64_t>(file_stat.st_size) !=
          expected_nvfp4_file_bytes()) {
        return impl_->reject_load_locked(
            ProviderStatus::invalid_argument,
            "expert bank size does not match the geometry its header declares");
      }

      if (config.resident_experts.size() > kNvfp4ExpertsPerLayer) {
        return impl_->reject_load_locked(
            ProviderStatus::invalid_argument,
            "config.resident_experts contains more than 256 entries");
      }
      std::array<bool, kNvfp4ExpertsPerLayer> seen_resident_experts{};
      for (const std::int32_t canonical_expert : config.resident_experts) {
        if (canonical_expert < 0 ||
            canonical_expert >=
                static_cast<std::int32_t>(g_source_experts_per_layer)) {
          return impl_->reject_load_locked(
              ProviderStatus::invalid_argument,
              "config.resident_experts contains an ID outside the bank's "
              "expert range");
        }
        const std::size_t expert_index =
            static_cast<std::size_t>(canonical_expert);
        if (seen_resident_experts[expert_index]) {
          return impl_->reject_load_locked(
              ProviderStatus::invalid_argument,
              "config.resident_experts contains a duplicate ID");
        }
        seen_resident_experts[expert_index] = true;
      }
      resident_experts = config.resident_experts;
      if (resident_experts.empty()) {
        resident_experts.reserve(g_source_experts_per_layer);
        for (std::size_t expert = 0; expert < g_source_experts_per_layer;
             ++expert) {
          resident_experts.push_back(static_cast<std::int32_t>(expert));
        }
      }
      bank_records = bank_bytes + sizeof(ExpertBankHeader);
    } else {
      if (impl_->bank_mapping_bytes < sizeof(Int4BankHeaderPrefix)) {
        return impl_->reject_load_locked(
            ProviderStatus::invalid_argument,
            "SBINT401 bank is shorter than the expected 128-byte v2 prefix");
      }
      Int4BankHeaderPrefix header{};
      std::memcpy(&header, bank_bytes, sizeof(header));
      if (header.version != kInt4Version) {
        std::ostringstream message;
        message << "SBINT401 bank version " << header.version
                << " is unsupported; expected version " << kInt4Version;
        return impl_->reject_load_locked(ProviderStatus::invalid_argument,
                                         message.str());
      }
      if (!int4_requested()) {
        return impl_->reject_load_locked(
            ProviderStatus::invalid_argument,
            "SBINT401 bank requires explicit opt-in with "
            "SHOOTING_BRAKE_B70_INT4=1");
      }
      if (header.reserved0 != 0 || header.reserved1 != 0) {
        return impl_->reject_load_locked(
            ProviderStatus::invalid_argument,
            "SBINT401 v2 reserved header fields must be zero");
      }
      if (header.num_layers == 0 ||
          header.num_layers > header.source_num_layers ||
          header.experts_per_layer == 0 ||
          header.source_experts_per_layer == 0 ||
          header.resident_set_shared_across_layers != 1) {
        return impl_->reject_load_locked(
            ProviderStatus::invalid_argument,
            "SBINT401 v2 declares invalid layer, expert, or resident-set "
            "geometry");
      }
      if (header.group_size != 128) {
        std::ostringstream message;
        message << "SBINT401 group_size=" << header.group_size
                << " is unsupported; int4_bank_format.py requires 128";
        return impl_->reject_load_locked(ProviderStatus::invalid_argument,
                                         message.str());
      }
      if (header.hidden % header.group_size != 0 ||
          header.moe_intermediate % header.group_size != 0) {
        return impl_->reject_load_locked(
            ProviderStatus::invalid_argument,
            "SBINT401 hidden and moe_intermediate dimensions must both be "
            "divisible by group_size");
      }
      if (header.bits != 4) {
        std::ostringstream message;
        message << "SBINT401 bits=" << header.bits
                << " is unsupported; int4_bank_format.py requires 4";
        return impl_->reject_load_locked(ProviderStatus::invalid_argument,
                                         message.str());
      }
      if (header.zero_point != 8) {
        std::ostringstream message;
        message
            << "SBINT401 zero_point=" << header.zero_point
            << " is unsupported; required value is 8 per int4_bank_format.py "
               "and the AutoGPTQ zeros-1 convention (stored qzero nibble 7 "
               "means effective zero point 8)";
        return impl_->reject_load_locked(ProviderStatus::invalid_argument,
                                         message.str());
      }
      if (!adopt_int4_bank_geometry(
              header.num_layers, header.source_experts_per_layer,
              header.hidden, header.moe_intermediate)) {
        return impl_->reject_load_locked(
            ProviderStatus::invalid_argument,
            "SBINT401 v2 declares inconsistent or unsupported geometry");
      }

      const std::uint64_t ids_end =
          sizeof(Int4BankHeaderPrefix) +
          static_cast<std::uint64_t>(header.experts_per_layer) *
              sizeof(std::int32_t);
      const std::uint64_t expected_data_offset =
          ((ids_end + kInt4Alignment - 1) / kInt4Alignment) *
          kInt4Alignment;
      if (header.data_offset != expected_data_offset ||
          header.data_offset % kInt4Alignment != 0 ||
          header.data_offset > impl_->bank_mapping_bytes) {
        std::ostringstream message;
        message << "SBINT401 data_offset=" << header.data_offset
                << " is invalid; expected " << expected_data_offset;
        return impl_->reject_load_locked(ProviderStatus::invalid_argument,
                                         message.str());
      }

      const std::array<std::uint32_t, 6> plane_offsets{
          header.gate_q_offset, header.gate_s_offset, header.up_q_offset,
          header.up_s_offset,   header.down_q_offset, header.down_s_offset};
      const std::array<std::uint32_t, 6> plane_sizes{
          header.gate_q_size, header.gate_s_size, header.up_q_size,
          header.up_s_size,   header.down_q_size, header.down_s_size};
      const std::uint64_t matrix_elements =
          static_cast<std::uint64_t>(header.hidden) *
          header.moe_intermediate;
      const std::uint64_t qweight_bytes = matrix_elements / 2;
      const std::uint64_t scale_bytes =
          (matrix_elements / header.group_size) * sizeof(sycl::half);
      const std::array<std::uint64_t, 6> expected_plane_sizes{
          qweight_bytes, scale_bytes, qweight_bytes,
          scale_bytes,   qweight_bytes, scale_bytes};
      std::uint64_t running_offset = 0;
      for (std::size_t plane = 0; plane < plane_offsets.size(); ++plane) {
        if (plane_offsets[plane] != running_offset ||
            plane_sizes[plane] != expected_plane_sizes[plane] ||
            plane_offsets[plane] % kInt4Alignment != 0 ||
            plane_sizes[plane] == 0 ||
            plane_sizes[plane] % kInt4Alignment != 0) {
          std::ostringstream message;
          message << "SBINT401 plane " << plane
                  << " has offset/size " << plane_offsets[plane] << "/"
                  << plane_sizes[plane] << "; expected " << running_offset
                  << "/" << expected_plane_sizes[plane]
                  << " with 4096-byte alignment";
          return impl_->reject_load_locked(ProviderStatus::invalid_argument,
                                           message.str());
        }
        running_offset += plane_sizes[plane];
      }
      if (header.expert_stride_bytes != running_offset ||
          header.expert_stride_bytes % kInt4Alignment != 0 ||
          header.layer_stride_bytes !=
              static_cast<std::uint64_t>(header.experts_per_layer) *
                  header.expert_stride_bytes) {
        return impl_->reject_load_locked(
            ProviderStatus::invalid_argument,
            "SBINT401 expert/layer stride does not match its six AoS planes");
      }
      const std::uint64_t expected_file_size =
          static_cast<std::uint64_t>(header.data_offset) +
          static_cast<std::uint64_t>(header.num_layers) *
              header.layer_stride_bytes;
      if (static_cast<std::uint64_t>(file_stat.st_size) !=
          expected_file_size) {
        std::ostringstream message;
        message << "SBINT401 file size=" << file_stat.st_size
                << " does not match header-derived size="
                << expected_file_size;
        return impl_->reject_load_locked(ProviderStatus::invalid_argument,
                                         message.str());
      }

      resident_experts.resize(header.experts_per_layer);
      std::memcpy(resident_experts.data(),
                  bank_bytes + sizeof(Int4BankHeaderPrefix),
                  resident_experts.size() * sizeof(std::int32_t));
      std::int32_t previous = -1;
      for (const std::int32_t source_expert : resident_experts) {
        if (source_expert <= previous || source_expert < 0 ||
            source_expert >=
                static_cast<std::int32_t>(header.source_experts_per_layer)) {
          return impl_->reject_load_locked(
              ProviderStatus::invalid_argument,
              "SBINT401 source expert IDs must be strictly increasing, "
              "unique, and inside the source expert geometry");
        }
        previous = source_expert;
      }
      for (std::uint64_t offset = ids_end; offset < header.data_offset;
           ++offset) {
        if (bank_bytes[offset] != 0) {
          return impl_->reject_load_locked(
              ProviderStatus::invalid_argument,
              "SBINT401 header alignment padding must be zero");
        }
      }
      if (!config.resident_experts.empty() &&
          config.resident_experts != resident_experts) {
        const auto summarize_ids =
            [](const std::vector<std::int32_t>& ids) {
              std::ostringstream summary;
              summary << "size=" << ids.size() << " ids=[";
              const std::size_t prefix = std::min<std::size_t>(3, ids.size());
              for (std::size_t index = 0; index < prefix; ++index) {
                if (index != 0) {
                  summary << ",";
                }
                summary << ids[index];
              }
              if (ids.size() > 6) {
                summary << ",...,";
              } else if (ids.size() > prefix) {
                summary << ",";
              }
              const std::size_t suffix =
                  ids.size() > 6 ? ids.size() - 3 : prefix;
              for (std::size_t index = suffix; index < ids.size(); ++index) {
                if (index != suffix) {
                  summary << ",";
                }
                summary << ids[index];
              }
              summary << "]";
              return summary.str();
            };
        return impl_->reject_load_locked(
            ProviderStatus::invalid_argument,
            "config.resident_experts does not exactly match the SBINT401 "
            "source expert ID map: config {" +
                summarize_ids(config.resident_experts) + "}, bank {" +
                summarize_ids(resident_experts) + "}");
      }

      impl_->bank_format = BankFormat::int4;
      impl_->int4_header = header;
      impl_->int4_source_expert_ids = resident_experts;
      int4_expert_stride = header.expert_stride_bytes;
      bank_records = bank_bytes + header.data_offset;
    }

    if (config.top_k > g_source_experts_per_layer) {
      return impl_->reject_load_locked(
          ProviderStatus::invalid_argument,
          "config.top_k exceeds the bank's source expert count");
    }

    const std::size_t scratch_intermediates =
        impl_->bank_format == BankFormat::int4 ? 1 : 2;
    const auto checked_multiply = [](const std::size_t lhs,
                                     const std::size_t rhs,
                                     std::size_t* product) noexcept {
      if (lhs != 0 &&
          rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        return false;
      }
      *product = lhs * rhs;
      return true;
    };
    std::size_t scratch_elements = config.max_batch;
    std::size_t scratch_bytes = 0;
    if (!checked_multiply(scratch_elements, config.top_k,
                          &scratch_elements) ||
        !checked_multiply(scratch_elements, scratch_intermediates,
                          &scratch_elements) ||
        !checked_multiply(scratch_elements, g_intermediate,
                          &scratch_elements) ||
        !checked_multiply(scratch_elements, sizeof(float), &scratch_bytes)) {
      return impl_->reject_load_locked(
          ProviderStatus::invalid_argument,
          "config.max_batch and top_k overflow the scratch-buffer size");
    }

    std::optional<SelectedDevice> selected_holder;
    try {
      selected_holder.emplace(select_b70(config.device_selector));
    } catch (const std::invalid_argument& error) {
      return impl_->reject_load_locked(ProviderStatus::invalid_argument,
                                       error.what());
    }
    const SelectedDevice selected = std::move(*selected_holder);
    sycl::async_handler async_handler =
        [state = impl_.get()](sycl::exception_list errors) noexcept {
          state->capture_async_errors(std::move(errors));
        };
    impl_->profiling = profiling_requested();
    impl_->spin_wait = spin_wait_enabled();
    sycl::property_list queue_properties =
        impl_->profiling
            ? sycl::property_list{sycl::property::queue::in_order(),
                                  sycl::property::queue::enable_profiling()}
            : sycl::property_list{sycl::property::queue::in_order()};
    impl_->queue.emplace(selected.device, std::move(async_handler),
                         queue_properties);

    impl_->config.max_batch = config.max_batch;
    impl_->config.top_k = config.top_k;
    impl_->config.generation = config.generation;
    impl_->config.resident_experts = std::move(resident_experts);
    impl_->config.device_selector = config.device_selector;
    impl_->health.generation = config.generation;
    impl_->capability.device_name = selected.name;
    impl_->capability.device_index =
        static_cast<std::uint32_t>(selected.index);
    impl_->capability.device_pci_bdf = selected.pci_bdf;
    impl_->capability.device_memory_total_bytes =
        static_cast<std::uint64_t>(
            selected.device.get_info<sycl::info::device::global_mem_size>());
    impl_->capability.backend =
        impl_->bank_format == BankFormat::int4 ? "quixicore-xpu-int4"
                                               : "quixicore-xpu-nvfp4";
    impl_->capability.supported_hidden_sizes = {
        static_cast<std::uint32_t>(g_hidden)};
    impl_->capability.supported_intermediate_sizes = {
        static_cast<std::uint32_t>(g_intermediate)};
    impl_->capability.supported_topk = {
        static_cast<std::uint32_t>(config.top_k)};
    const std::size_t resident_experts_per_layer =
        impl_->config.resident_experts.size();
    const std::size_t resident_experts_total =
        g_layers * resident_experts_per_layer;
    if (resident_experts_total >
        std::numeric_limits<std::uint32_t>::max()) {
      return impl_->reject_load_locked(
          ProviderStatus::invalid_argument,
          "bank resident expert count exceeds the capability ABI");
    }
    impl_->capability.num_resident_experts =
        static_cast<std::uint32_t>(resident_experts_total);
    impl_->capability.output_fp16 = impl_->out_fp16;
    impl_->capability.max_batch_remote =
        static_cast<std::uint32_t>(config.max_batch);
    impl_->capability.kernel_families =
        impl_->bank_format == BankFormat::int4
            ? std::vector<std::string>{"int4_moe_split"}
            : std::vector<std::string>{"nvfp4_moe_split",
                                       "nvfp4_moe_fused"};
    impl_->capability.health_heartbeat_interval_ms = 1000;
    impl_->capability.num_layers = static_cast<std::uint32_t>(g_layers);
    impl_->capability.experts_per_layer =
        static_cast<std::uint32_t>(g_source_experts_per_layer);
    impl_->capability.source_expert_ids =
        impl_->bank_format == BankFormat::int4
            ? impl_->int4_source_expert_ids
            : std::vector<std::int32_t>{};

    const std::uint64_t required_bytes = persistent_device_bytes(
        impl_->bank_format, config.max_batch, config.top_k,
        resident_experts_per_layer, int4_expert_stride);
    const std::uint64_t device_bytes =
        impl_->capability.device_memory_total_bytes;
    if (device_bytes != 0 && required_bytes > device_bytes) {
      const std::uint64_t bytes_per_expert =
          impl_->bank_format == BankFormat::int4
              ? int4_expert_stride
              : nvfp4_weight_bytes_per_expert();
      const std::uint64_t per_index =
          static_cast<std::uint64_t>(g_layers) * bytes_per_expert;
      std::ostringstream message;
      message << "placement needs " << (required_bytes >> 20)
              << " MiB on the B70 but the device has " << (device_bytes >> 20)
              << " MiB; at " << g_layers << " layers each resident expert "
              << "index costs " << (per_index >> 20) << " MiB, so at most "
              << (per_index ? device_bytes / per_index : 0)
              << " experts per layer fit";
      return impl_->reject_load_locked(ProviderStatus::device_error,
                                       message.str());
    }

    if (impl_->bank_format == BankFormat::int4) {
      impl_->int4_records = impl_->allocate_device<std::uint8_t>(
          static_cast<std::size_t>(impl_->int4_header.num_layers) *
          static_cast<std::size_t>(impl_->int4_header.layer_stride_bytes));
    } else {
      impl_->w13 = impl_->allocate_device<std::uint8_t>(
          resident_experts_total * g_w13_bytes);
      impl_->s13 = impl_->allocate_device<std::uint8_t>(
          resident_experts_total * g_s13_bytes);
      impl_->w2 = impl_->allocate_device<std::uint8_t>(
          resident_experts_total * g_w2_bytes);
      impl_->s2 = impl_->allocate_device<std::uint8_t>(
          resident_experts_total * g_s2_bytes);
      impl_->w13_global =
          impl_->allocate_device<float>(resident_experts_total);
      impl_->w2_global =
          impl_->allocate_device<float>(resident_experts_total);
    }

    impl_->hidden =
        impl_->allocate_device<sycl::half>(config.max_batch * g_hidden);
    impl_->ids = impl_->allocate_device<std::int32_t>(
        config.max_batch * config.top_k);
    impl_->weights =
        impl_->allocate_device<float>(config.max_batch * config.top_k);
    impl_->scratch = impl_->allocate_device<float>(scratch_elements);
    impl_->output =
        impl_->allocate_device<float>(config.max_batch * g_hidden);
    impl_->copyout_staging =
        impl_->allocate_host<float>(config.max_batch * g_hidden);

    // Narrow result wire. Values here are the final scattered MoE output with
    // the per-expert alpha already applied, so they sit at activation scale --
    // fp16's 65,504 ceiling is not in play. That is specifically NOT true of
    // the GEMM accumulators upstream, where an unscaled dot product over
    // K=3072 reaches ~1e6 and saturating it to Inf was the NaN that cost a
    // boot to find. Those stay fp32.
    {
      const char* v = std::getenv("SHOOTING_BRAKE_B70_OUT_FP16");
      impl_->out_fp16 = v != nullptr && v[0] == '1' && v[1] == '\0';
    }
    if (impl_->out_fp16) {
      impl_->out16 =
          impl_->allocate_device<sycl::half>(config.max_batch * g_hidden);
      impl_->copyout_staging16 =
          impl_->allocate_host<sycl::half>(config.max_batch * g_hidden);
    }

    // Grouped prefill scratch. Only allocated when the flag is on and only
    // for NVFP4 -- the int4 bank has a different record layout and its own
    // kernel, so grouping it is a separate exercise.
    if (grouped_requested() && impl_->bank_format != BankFormat::int4) {
      {
        const char* v = std::getenv("SHOOTING_BRAKE_B70_PIPELINE");
        long n = (v != nullptr) ? std::strtol(v, nullptr, 10) : 1;
        if (n < 1) n = 1;
        if (n > 8) n = 8;
        impl_->pipeline_chunks = static_cast<int>(n);
      }
      const std::size_t nchunks =
          static_cast<std::size_t>(impl_->pipeline_chunks);
      impl_->pl_chunk_M = (config.max_batch + nchunks - 1) / nchunks;
      impl_->pl_chunk_routes = impl_->pl_chunk_M * config.top_k;
      impl_->pl_experts = resident_experts_per_layer;
      // Route-scaled scratch is pooled with a leading [nchunks] dimension
      // (one pool + views, not nchunks pointer sets: a missed pointer is
      // silent garbage, a missed index is the same bug in one place). Views
      // scale down with M/nchunks, so the pool total stays flat; only the
      // tiny expert-indexed arrays and the work counter genuinely duplicate.
      const std::size_t routes = nchunks * impl_->pl_chunk_routes;
      const std::size_t experts = resident_experts_per_layer;
      impl_->g_act = impl_->allocate_device<sycl::half>(routes * g_hidden);
      impl_->g_mid =
          impl_->allocate_device<float>(routes * 2 * g_intermediate);
      impl_->g_gated =
          impl_->allocate_device<sycl::half>(routes * g_intermediate);
      impl_->g_outr = impl_->allocate_device<float>(routes * g_hidden);
      // The mainloop applies bias unconditionally, so a null pointer faults
      // inside the kernel and surfaces as a bare SIGSEGV with no diagnostic.
      // Biases are constant zeros, read-only, and therefore shared across
      // chunks rather than pooled.
      impl_->g_bias13 =
          impl_->allocate_device<sycl::half>(experts * 2 * g_intermediate);
      impl_->g_bias2 = impl_->allocate_device<sycl::half>(experts * g_hidden);
      impl_->g_hist =
          impl_->allocate_device<std::int32_t>(nchunks * (experts + 1));
      impl_->g_offs =
          impl_->allocate_device<std::int32_t>(nchunks * (experts + 1));
      impl_->g_cursor = impl_->allocate_device<std::int32_t>(nchunks * experts);
      impl_->g_rows = impl_->allocate_device<std::int32_t>(nchunks * experts);
      impl_->g_slot_row = impl_->allocate_device<std::int32_t>(routes);
      impl_->g_slot_exp = impl_->allocate_device<std::int32_t>(routes);
      impl_->g_slot_w = impl_->allocate_device<float>(routes);
      impl_->g_atomic = impl_->allocate_device<std::int32_t>(nchunks);
      impl_->queue
          ->memset(impl_->g_bias13, 0,
                   experts * 2 * g_intermediate * sizeof(sycl::half))
          .wait();
      impl_->queue
          ->memset(impl_->g_bias2, 0, experts * g_hidden * sizeof(sycl::half))
          .wait();
      impl_->grouped_ready = true;
    }

    if (impl_->bank_format != BankFormat::int4) {
      const std::size_t layer_staging_bytes =
          resident_experts_per_layer * g_w13_bytes;
      impl_->upload_staging =
          impl_->allocate_host<std::uint8_t>(layer_staging_bytes);
    }

    if (impl_->bank_format == BankFormat::int4) {
      constexpr std::size_t kUploadChunkBytes = 32 * 1024 * 1024;
      static_assert(kUploadChunkBytes % kInt4Alignment == 0);
      const std::size_t payload_bytes =
          static_cast<std::size_t>(impl_->int4_header.num_layers) *
          static_cast<std::size_t>(impl_->int4_header.layer_stride_bytes);
      // No anonymous bounce buffer: each 32 MiB, page-aligned mmap slice is
      // copied straight to device USM. The event is waited before MADV_DONTNEED
      // releases that slice, bounding weight-source file RSS to 32 MiB plus
      // kernel readahead and adding zero weight bytes to anonymous RSS.
      for (std::size_t offset = 0; offset < payload_bytes;
           offset += kUploadChunkBytes) {
        const std::size_t chunk_bytes =
            std::min(kUploadChunkBytes, payload_bytes - offset);
        const std::uint8_t* source = bank_records + offset;
        impl_->queue
            ->memcpy(impl_->int4_records + offset, source, chunk_bytes)
            .wait_and_throw();
        if (::madvise(const_cast<std::uint8_t*>(source), chunk_bytes,
                      MADV_DONTNEED) != 0) {
          throw std::runtime_error(errno_message(
              "madvise SBINT401 chunk after completed device upload"));
        }
      }
    } else {
      const std::size_t kS13Offset = g_w13_bytes;
      const std::size_t kW2Offset = kS13Offset + g_s13_bytes;
      const std::size_t kS2Offset = kW2Offset + g_w2_bytes;
      const std::size_t kW13GlobalOffset = kS2Offset + g_s2_bytes;
      const std::size_t kW2GlobalOffset = kW13GlobalOffset + sizeof(float);
      for (std::size_t layer = 0; layer < g_layers; ++layer) {
        impl_->upload_plane(bank_records, 0, g_w13_bytes, impl_->w13, layer);
        impl_->upload_plane(bank_records, kS13Offset, g_s13_bytes, impl_->s13,
                            layer);
        impl_->upload_plane(bank_records, kW2Offset, g_w2_bytes, impl_->w2,
                            layer);
        impl_->upload_plane(bank_records, kS2Offset, g_s2_bytes, impl_->s2,
                            layer);
        impl_->upload_globals(bank_records, kW13GlobalOffset,
                              impl_->w13_global, layer);
        impl_->upload_globals(bank_records, kW2GlobalOffset,
                              impl_->w2_global, layer);
      }
    }

    if (impl_->upload_staging != nullptr) {
      sycl::free(impl_->upload_staging, *impl_->queue);
      impl_->upload_staging = nullptr;
    }

    const std::uint64_t owned_device_bytes = persistent_device_bytes(
        impl_->bank_format, config.max_batch, config.top_k,
        resident_experts_per_layer, int4_expert_stride);
    impl_->capability.device_memory_available_bytes =
        impl_->capability.device_memory_total_bytes > owned_device_bytes
            ? impl_->capability.device_memory_total_bytes - owned_device_bytes
            : 0;

    const std::string asynchronous_error = impl_->consume_async_error();
    if (!asynchronous_error.empty()) {
      throw std::runtime_error("weight upload failed asynchronously: " +
                               asynchronous_error);
    }

    impl_->health.loaded = true;
    impl_->health.pending = false;
    impl_->health.last_error.clear();
    return ProviderStatus::ok;
  } catch (...) {
    try {
      std::lock_guard<std::mutex> lock(impl_->mutex);
      impl_->set_current_exception("provider load failed: ");
      impl_->release_resources_locked(false);
    } catch (...) {
    }
    return ProviderStatus::device_error;
  }
}

Capability B70Provider::capability() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->capability;
}

Health B70Provider::health() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  Health result = impl_->health;
  try {
    std::lock_guard<std::mutex> async_lock(impl_->async_mutex);
    if (!impl_->async_error.empty()) {
      result.last_error = impl_->async_error;
    }
  } catch (...) {
  }
  return result;
}

ProviderStatus B70Provider::issue(const std::uint64_t generation,
                                  const std::uint64_t sequence,
                                  const std::size_t layer,
                                  const sycl::half* hidden,
                                  const std::int32_t* ids,
                                  const float* weights,
                                  const std::size_t M) {
  try {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    if (impl_->health.stopped) {
      return ProviderStatus::shutdown;
    }
    if (!impl_->health.loaded || !impl_->queue) {
      return ProviderStatus::not_loaded;
    }
    if (impl_->health.pending) {
      return ProviderStatus::busy;
    }
    if (generation != impl_->config.generation) {
      return ProviderStatus::generation_mismatch;
    }
    if (hidden == nullptr || ids == nullptr || weights == nullptr || M == 0 ||
        M > impl_->config.max_batch || layer >= g_layers) {
      impl_->set_error("issue received an invalid pointer, layer, or batch shape");
      return ProviderStatus::invalid_argument;
    }

    const std::size_t route_elements = M * impl_->config.top_k;
    const std::size_t resident_experts = impl_->config.resident_experts.size();
    for (std::size_t index = 0; index < route_elements; ++index) {
      if (ids[index] < -1 ||
          ids[index] >= static_cast<std::int32_t>(resident_experts)) {
        impl_->set_error(
            "issue received an expert ID outside the configured compact range");
        return ProviderStatus::invalid_argument;
      }
      if (!std::isfinite(weights[index])) {
        impl_->set_error("issue received a non-finite routing weight");
        return ProviderStatus::invalid_argument;
      }
    }

    sycl::event first = impl_->queue->memcpy(
        impl_->hidden, hidden, M * g_hidden * sizeof(sycl::half));
    impl_->queue->memcpy(impl_->ids, ids,
                         route_elements * sizeof(std::int32_t));
    impl_->queue->memcpy(impl_->weights, weights,
                         route_elements * sizeof(float));
    if (impl_->profiling) {
      impl_->dispatch_begin.emplace(std::move(first));
      impl_->kernel_begin.emplace(impl_->queue->single_task([] {}));
    }

    bool use_split = true;
    if (impl_->bank_format == BankFormat::int4) {
      const auto& header = impl_->int4_header;
      const std::uint8_t* layer_records =
          impl_->int4_records + layer * header.layer_stride_bytes;
      // The kernel intentionally folds the symmetric zero point 8 into its
      // inner loop. load() validated header.zero_point loudly; making it a
      // runtime subtrahend would add work to every eight-nibble decode.
      quixicore::xpu::ops::int4_moe_split(
          *impl_->queue, impl_->hidden, impl_->ids, impl_->weights,
          reinterpret_cast<const std::int32_t*>(
              layer_records + header.gate_q_offset),
          reinterpret_cast<const quixicore::xpu::half_t*>(
              layer_records + header.gate_s_offset),
          reinterpret_cast<const std::int32_t*>(
              layer_records + header.up_q_offset),
          reinterpret_cast<const quixicore::xpu::half_t*>(
              layer_records + header.up_s_offset),
          reinterpret_cast<const std::int32_t*>(
              layer_records + header.down_q_offset),
          reinterpret_cast<const quixicore::xpu::half_t*>(
              layer_records + header.down_s_offset),
          impl_->scratch, impl_->output, header.expert_stride_bytes,
          header.group_size, M, resident_experts, impl_->config.top_k, g_hidden,
          g_intermediate, quixicore::xpu::DType::f16, true,
          quixicore::xpu::Variant::sycl, false);
    } else {
      const std::size_t first_expert = layer * resident_experts;
      const std::uint8_t* layer_w13 =
          impl_->w13 + first_expert * g_w13_bytes;
      const std::uint8_t* layer_s13 =
          impl_->s13 + first_expert * g_s13_bytes;
      const std::uint8_t* layer_w2 =
          impl_->w2 + first_expert * g_w2_bytes;
      const std::uint8_t* layer_s2 =
          impl_->s2 + first_expert * g_s2_bytes;
      const float* layer_w13_global = impl_->w13_global + first_expert;
      const float* layer_w2_global = impl_->w2_global + first_expert;

      use_split = M <= 32;
      // Grouped prefill takes the large-M arm when armed. Any refusal falls
      // through to the fused GEMV, so a shape this path does not serve
      // degrades in speed rather than in correctness.
      bool grouped_done = false;
      if (!use_split && impl_->grouped_ready) {
        // Chunked by token: each token's top_k routes live entirely inside
        // its own chunk, so chunks write disjoint output rows and the maths
        // is unchanged. The kernel zeroes exactly its own [Mc, H] output
        // window, resets its own scratch, and slot_row is chunk-relative on
        // a chunk-offset output pointer. Phase 1: sequential, one queue.
        const std::size_t nchunks =
            static_cast<std::size_t>(impl_->pipeline_chunks);
        const std::size_t chunk_rows = (M + nchunks - 1) / nchunks;
        const std::size_t er = impl_->pl_experts;
        const std::size_t cr = impl_->pl_chunk_routes;
        // Views are strided off the load-time expert count; a mismatch here
        // means the config changed under us -- refuse and take the GEMV.
        grouped_done = resident_experts == er;
        for (std::size_t c = 0; c < nchunks && grouped_done; ++c) {
          const std::size_t r0 = c * chunk_rows;
          if (r0 >= M) {
            break;
          }
          const std::size_t mc = std::min(chunk_rows, M - r0);
          grouped_done = sb::xe2::grouped_moe_nvfp4(
              *impl_->queue, impl_->hidden + r0 * g_hidden,
              impl_->ids + r0 * impl_->config.top_k,
              impl_->weights + r0 * impl_->config.top_k, layer_w13, layer_s13,
              layer_w13_global, layer_w2, layer_s2, layer_w2_global,
              impl_->g_act + c * cr * g_hidden,
              impl_->g_mid + c * cr * 2 * g_intermediate,
              impl_->g_gated + c * cr * g_intermediate,
              impl_->g_outr + c * cr * g_hidden, impl_->g_bias13,
              impl_->g_bias2, impl_->g_hist + c * (er + 1),
              impl_->g_offs + c * (er + 1), impl_->g_cursor + c * er,
              impl_->g_rows + c * er, impl_->g_slot_row + c * cr,
              impl_->g_slot_exp + c * cr, impl_->g_slot_w + c * cr,
              impl_->g_atomic + c, impl_->output + r0 * g_hidden,
              static_cast<int>(mc), static_cast<int>(resident_experts),
              static_cast<int>(impl_->config.top_k),
              static_cast<int>(g_hidden), static_cast<int>(g_intermediate),
              16);
        }
        // A refusal mid-loop falls through to the fused GEMV below, which
        // fully rewrites output[0, M*H) -- correct after partial chunk
        // success, merely slower.
      }
      if (grouped_done) {
        // The op wrote impl_->output; nothing further for this layer.
      } else if (use_split) {
        quixicore::xpu::ops::nvfp4_moe_split(
            *impl_->queue, impl_->hidden, impl_->ids, impl_->weights, layer_w13,
            layer_s13, layer_w13_global, layer_w2, layer_s2, layer_w2_global,
            impl_->scratch, impl_->output, M, resident_experts,
            impl_->config.top_k,
            g_hidden, g_intermediate, quixicore::xpu::DType::f16, true,
            quixicore::xpu::Variant::sycl, false);
      } else {
        quixicore::xpu::ops::nvfp4_moe_fused(
            *impl_->queue, impl_->hidden, impl_->ids, impl_->weights, layer_w13,
            layer_s13, layer_w13_global, layer_w2, layer_s2, layer_w2_global,
            impl_->output, M, resident_experts, impl_->config.top_k, g_hidden,
            g_intermediate, quixicore::xpu::DType::f16, true,
            quixicore::xpu::Variant::sycl, false);
      }
    }

    impl_->pending_error.clear();
    if (impl_->profiling) {
      impl_->kernel_end.emplace(impl_->queue->single_task([] {}));
    }

    // Enqueue the result copy now rather than in take(). The queue is
    // in-order, so it cannot start before the kernel finishes, and
    // submitting it here keeps its latency off the critical path —
    // take() would otherwise wait for the kernel, only then submit the
    // copy, and wait again.
    if (impl_->out_fp16) {
      // Narrow on the device. Widening on the host instead would put ~4 ms of
      // CPU conversion on take()'s critical path and hand the PCIe saving
      // straight back; this pass is ~0.08 ms of VRAM traffic.
      const std::size_t n = static_cast<std::size_t>(M) * g_hidden;
      const float* src = impl_->output;
      sycl::half* dst = impl_->out16;
      impl_->queue->parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        dst[i] = static_cast<sycl::half>(src[i]);
      });
      impl_->copy_out.emplace(impl_->queue->memcpy(
          impl_->copyout_staging16, impl_->out16, n * sizeof(sycl::half)));
    } else {
      impl_->copy_out.emplace(impl_->queue->memcpy(
          impl_->copyout_staging, impl_->output,
          M * g_hidden * sizeof(float)));
    }
#ifdef SHOOTING_BRAKE_ENABLE_TEST_FAULTS
    if (impl_->armed_test_fault &&
        *impl_->armed_test_fault ==
            ProviderTestFault::after_kernel_before_copyout &&
        impl_->armed_test_fault_sequence == sequence) {
      impl_->pending_error =
          "injected device error after kernel before copyout";
      impl_->armed_test_fault.reset();
      impl_->armed_test_fault_sequence = 0;
    }
#endif
    impl_->pending_generation = generation;
    impl_->pending_sequence = sequence;
    impl_->pending_M = M;
    impl_->pending_split = use_split;
    impl_->health.last_error.clear();
    impl_->health.pending = true;
    ++impl_->health.dispatches;
    return ProviderStatus::ok;
  } catch (...) {
    try {
      std::lock_guard<std::mutex> lock(impl_->mutex);
      impl_->set_current_exception("provider issue failed: ");
      if (impl_->queue) {
        try {
          impl_->queue->wait_and_throw();
        } catch (...) {
        }
      }
      try {
        const std::string asynchronous_error = impl_->consume_async_error();
        if (!asynchronous_error.empty()) {
          impl_->set_error("provider issue failed asynchronously: " +
                           asynchronous_error);
        }
      } catch (...) {
      }
      impl_->health.pending = false;
      impl_->pending_M = 0;
    } catch (...) {
    }
    return ProviderStatus::device_error;
  }
}

ProviderStatus B70Provider::take(const std::uint64_t generation,
                                 const std::uint64_t sequence, float* output,
                                 const std::size_t output_elements,
                                 DispatchResult* result) {
  try {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    if (impl_->health.stopped) {
      return ProviderStatus::shutdown;
    }
    if (!impl_->health.loaded || !impl_->queue) {
      return ProviderStatus::not_loaded;
    }
    if (!impl_->health.pending) {
      return ProviderStatus::sequence_mismatch;
    }
    if (generation != impl_->pending_generation) {
      return ProviderStatus::generation_mismatch;
    }
    if (sequence != impl_->pending_sequence) {
      return ProviderStatus::sequence_mismatch;
    }
    if (!impl_->pending_error.empty()) {
      impl_->set_error(impl_->pending_error);
      impl_->retire_pending_locked();
      return ProviderStatus::device_error;
    }

    const std::size_t required_elements = impl_->pending_M * g_hidden;
    if (output == nullptr || result == nullptr ||
        output_elements < required_elements) {
      impl_->set_error("take received an invalid output buffer or result pointer");
      return ProviderStatus::invalid_argument;
    }

    // One wait. The queue is in-order and issue() already enqueued the
    // H2D copies, the kernel, and the result copy, so waiting on the
    // last of them covers the whole dispatch.
    //
    // Spin first, block second. Intel's runtime spins for a short window
    // inside wait_and_throw() and then SLEEPS the thread, so completion
    // costs an interrupt wakeup. Measured at the production decode shape on
    // the Gen4 B70 [experiments/b13_wait_probe.cpp, 3000 iters]:
    //
    //   blocking wait_and_throw() : mean 9.46 us  (p50 8.35, p90 17.87)
    //   spin on event status      : mean 6.37 us  (p50 6.33, p90  6.76)
    //
    // 3.09 us/dispatch, and the blocking path is bimodal -- the p90 is where
    // the sleep shows up. At 48 dispatches/step that is 0.15 ms/step.
    // take() runs on the dedicated poller thread, which already spins in its
    // outer loop (b70_capi.cpp), so it must never sleep here.
    //
    // The spin is BOUNDED and falls through to the blocking wait: a wedged
    // device must not burn a core forever, and wait_and_throw() is still the
    // only thing that surfaces asynchronous exceptions. After a successful
    // spin it observes an already-complete event and returns immediately.
    if (impl_->spin_wait) {
      constexpr int kMaxSpins = 200000;  // ~2 orders of magnitude over p99
      for (int spins = 0; spins < kMaxSpins; ++spins) {
        if (impl_->copy_out
                ->get_info<sycl::info::event::command_execution_status>() ==
            sycl::info::event_command_status::complete) {
          break;
        }
        SB_PROVIDER_SPIN_HINT();
      }
    }
    impl_->copy_out->wait_and_throw();
    const std::string asynchronous_error = impl_->consume_async_error();
    if (!asynchronous_error.empty()) {
      impl_->pending_error =
          "provider dispatch failed asynchronously: " + asynchronous_error;
      impl_->set_error(impl_->pending_error);
      impl_->retire_pending_locked();
      return ProviderStatus::device_error;
    }

    DispatchResult completed_result;
    completed_result.generation = impl_->pending_generation;
    completed_result.sequence = impl_->pending_sequence;
    completed_result.M = impl_->pending_M;
    completed_result.kernel = impl_->pending_split ? "split" : "fused";
    if (impl_->profiling) {
      const auto kernel_start =
          impl_->kernel_begin->get_profiling_info<
              sycl::info::event_profiling::command_end>();
      const auto kernel_stop =
          impl_->kernel_end->get_profiling_info<
              sycl::info::event_profiling::command_start>();
      const auto total_start =
          impl_->dispatch_begin->get_profiling_info<
              sycl::info::event_profiling::command_start>();
      const auto total_stop =
          impl_->copy_out->get_profiling_info<
              sycl::info::event_profiling::command_end>();
      completed_result.kernel_us =
          static_cast<double>(kernel_stop - kernel_start) * 1.0e-3;
      completed_result.total_us =
          static_cast<double>(total_stop - total_start) * 1.0e-3;
    }
    *result = std::move(completed_result);

    impl_->health.last_error.clear();
    impl_->retire_pending_locked();

    // The caller's buffer element width follows the wire: sb_b70_out_fp16()
    // publishes which to allocate. Getting that pair wrong is silent garbage,
    // which is why the provider states its choice instead of both sides
    // reading the environment and hoping they agree.
    std::memcpy(output,
                impl_->out_fp16
                    ? static_cast<const void*>(impl_->copyout_staging16)
                    : static_cast<const void*>(impl_->copyout_staging),
                required_elements *
                    (impl_->out_fp16 ? sizeof(sycl::half) : sizeof(float)));
    return ProviderStatus::ok;
  } catch (...) {
    try {
      std::lock_guard<std::mutex> lock(impl_->mutex);
      impl_->set_current_exception("provider take failed: ");
      impl_->retire_pending_locked();
    } catch (...) {
    }
    return ProviderStatus::device_error;
  }
}

#ifdef SHOOTING_BRAKE_ENABLE_TEST_FAULTS
ProviderStatus B70Provider::arm_test_fault(const ProviderTestFault fault,
                                           const std::uint64_t sequence) {
  try {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    if (impl_->health.stopped) {
      return ProviderStatus::shutdown;
    }
    if (!impl_->health.loaded || !impl_->queue) {
      return ProviderStatus::not_loaded;
    }
    if (impl_->health.pending || impl_->armed_test_fault) {
      return ProviderStatus::busy;
    }
    if (fault != ProviderTestFault::after_kernel_before_copyout ||
        sequence == 0) {
      impl_->set_error("invalid provider test fault or sequence");
      return ProviderStatus::invalid_argument;
    }

    impl_->armed_test_fault = fault;
    impl_->armed_test_fault_sequence = sequence;
    impl_->health.last_error.clear();
    return ProviderStatus::ok;
  } catch (...) {
    try {
      std::lock_guard<std::mutex> lock(impl_->mutex);
      impl_->set_current_exception("arming provider test fault failed: ");
    } catch (...) {
    }
    return ProviderStatus::device_error;
  }
}
#endif

bool B70Provider::device_memory(std::size_t* free_bytes,
                                std::size_t* total_bytes) const noexcept {
  try {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->queue) {
      return false;
    }
    const sycl::device device = impl_->queue->get_device();
    if (!device.has(sycl::aspect::ext_intel_free_memory)) {
      return false;
    }
    if (free_bytes) {
      *free_bytes =
          device.get_info<sycl::ext::intel::info::device::free_memory>();
    }
    if (total_bytes) {
      *total_bytes = device.get_info<sycl::info::device::global_mem_size>();
    }
    return true;
  } catch (...) {
    // Occupancy is reporting. A runtime that refuses the query must not be
    // able to take down a serving process through a telemetry call.
    return false;
  }
}

bool B70Provider::register_host_range(const void* ptr,
                                      const std::size_t bytes) noexcept {
  try {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->health.loaded || !impl_->queue || ptr == nullptr ||
        bytes == 0) {
      return false;
    }
    void* mutable_ptr = const_cast<void*>(ptr);
    sycl::ext::oneapi::experimental::prepare_for_device_copy(
        mutable_ptr, bytes, impl_->queue->get_context());
    impl_->registered_host_ranges.emplace_back(mutable_ptr, bytes);
    return true;
  } catch (...) {
    return false;
  }
}

void B70Provider::shutdown() noexcept {
  try {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->health.stopped) {
      return;
    }
    impl_->release_resources_locked(true);
  } catch (...) {
  }
}

}  // namespace shooting_brake::phase1
