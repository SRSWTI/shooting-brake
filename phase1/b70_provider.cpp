#include "b70_provider.hpp"

#include <algorithm>
#include <array>
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

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "quixicore/xpu/ops.hpp"
#include "quixicore/xpu/runtime.hpp"

namespace shooting_brake::phase1 {
namespace {

// Expert-bank geometry. Runtime, not compile-time: the bank file names
// the shape it was built for, and hard-coding it here is how a provider
// silently serves a different model's weights. Set once by
// `adopt_bank_geometry` before any dispatch, from the header of the bank
// being loaded.
//
// Two values stay compile-time because they size stack arrays and buffers
// before the bank is even opened, and because both qualified models share
// them. `require_qualified_config` on the Python side asserts them against
// the model's own config and names this provider as the reason, so a third
// model cannot reach here with a different routing width.
constexpr std::size_t kExpertsPerLayer = 256;
constexpr std::size_t kTopK = 8;

std::size_t g_layers = 0;
std::size_t g_hidden = 0;
std::size_t g_intermediate = 0;
std::size_t g_total_experts = 0;
std::size_t g_w13_bytes = 0;
std::size_t g_s13_bytes = 0;
std::size_t g_w2_bytes = 0;
std::size_t g_s2_bytes = 0;
std::size_t g_expert_bytes = 0;

// Derives every dependent size from the three the header carries. Returns
// false when the header's own arithmetic disagrees with its dimensions,
// which catches a truncated or foreign file before anything is mapped
// into a kernel argument.
bool adopt_bank_geometry(std::size_t layers, std::size_t experts_per_layer,
                         std::size_t hidden, std::size_t intermediate,
                         std::uint64_t w13, std::uint64_t s13,
                         std::uint64_t w2, std::uint64_t s2) noexcept {
  if (layers == 0 || hidden == 0 || intermediate == 0 ||
      experts_per_layer != kExpertsPerLayer || hidden % 16 != 0 ||
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
  g_total_experts = layers * experts_per_layer;
  g_w13_bytes = w13_bytes;
  g_s13_bytes = s13_bytes;
  g_w2_bytes = w2_bytes;
  g_s2_bytes = s2_bytes;
  g_expert_bytes =
      w13_bytes + s13_bytes + w2_bytes + s2_bytes + 2 * sizeof(float);
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
#pragma pack(pop)

static_assert(sizeof(ExpertBankHeader) == 60,
              "the packed expert-bank header must be 60 bytes");
// Functions rather than constants: every term depends on the geometry the
// bank header supplies, so none of them exists until `adopt_bank_geometry`
// has run.
std::uint64_t expected_file_bytes() noexcept {
  return sizeof(ExpertBankHeader) +
         static_cast<std::uint64_t>(g_total_experts) * g_expert_bytes;
}

std::uint64_t persistent_weight_bytes_per_expert() noexcept {
  return g_w13_bytes + g_s13_bytes + g_w2_bytes + g_s2_bytes +
         2 * sizeof(float);
}

std::uint64_t persistent_device_bytes(const std::size_t max_batch,
                                      const std::size_t resident_experts) {
  const std::uint64_t bytes_per_batch_row =
      g_hidden * sizeof(sycl::half) +
      kTopK * sizeof(std::int32_t) +
      kTopK * sizeof(float) +
      kTopK * 2 * g_intermediate * sizeof(float) +
      g_hidden * sizeof(float);
  const std::uint64_t resident_weight_bytes =
      static_cast<std::uint64_t>(g_layers) * resident_experts *
      persistent_weight_bytes_per_expert();
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
};

SelectedDevice select_b70() {
  std::optional<SelectedDevice> selected;
  bool selected_level_zero = false;

  for (const auto& platform : sycl::platform::get_platforms()) {
    const bool level_zero =
        platform.get_backend() == sycl::backend::ext_oneapi_level_zero;
    for (const auto& device : platform.get_devices()) {
      if (!device.is_gpu() ||
          device.get_info<sycl::info::device::vendor_id>() != 0x8086u) {
        continue;
      }

      const std::string name = device.get_info<sycl::info::device::name>();
      if (lowercase(name).find("b70") == std::string::npos) {
        continue;
      }
      if (!selected || (level_zero && !selected_level_zero)) {
        selected.emplace(SelectedDevice{device, name});
        selected_level_zero = level_zero;
      }
    }
  }

  if (!selected) {
    throw std::runtime_error(
        "no Intel GPU whose device name contains B70 was found");
  }
  return std::move(*selected);
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

  std::optional<sycl::queue> queue;
  std::string async_error;

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

  std::optional<sycl::event> dispatch_begin;
  std::optional<sycl::event> kernel_begin;
  std::optional<sycl::event> kernel_end;
  std::optional<sycl::event> copy_out;
  bool profiling = false;

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
      free_pointer(output);
      free_pointer(scratch);
      free_pointer(weights);
      free_pointer(ids);
      free_pointer(hidden);
      free_pointer(w2_global);
      free_pointer(w13_global);
      free_pointer(s2);
      free_pointer(w2);
      free_pointer(s13);
      free_pointer(w13);
    }

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
    const std::size_t source_layer = layer * kExpertsPerLayer;
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
    const std::size_t source_layer = layer * kExpertsPerLayer;
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
        config.top_k != kTopK || config.generation == 0) {
      impl_->set_error(
          "config requires 0 < max_batch <= UINT32_MAX, top_k == 8, and "
          "generation > 0");
      return ProviderStatus::invalid_argument;
    }
    if (config.max_batch >
        std::numeric_limits<std::size_t>::max() /
            (kTopK * 2 * g_intermediate * sizeof(float))) {
      impl_->set_error("config.max_batch overflows the scratch-buffer size");
      return ProviderStatus::invalid_argument;
    }

    if (config.resident_experts.size() > kExpertsPerLayer) {
      impl_->set_error(
          "config.resident_experts contains more than 256 entries");
      return ProviderStatus::invalid_argument;
    }
    std::array<bool, kExpertsPerLayer> seen_resident_experts{};
    for (const std::int32_t canonical_expert : config.resident_experts) {
      if (canonical_expert < 0 ||
          canonical_expert >=
              static_cast<std::int32_t>(kExpertsPerLayer)) {
        impl_->set_error(
            "config.resident_experts contains an ID outside [0, 256)");
        return ProviderStatus::invalid_argument;
      }
      const std::size_t expert_index =
          static_cast<std::size_t>(canonical_expert);
      if (seen_resident_experts[expert_index]) {
        impl_->set_error(
            "config.resident_experts contains a duplicate ID");
        return ProviderStatus::invalid_argument;
      }
      seen_resident_experts[expert_index] = true;
    }

    std::vector<std::int32_t> resident_experts = config.resident_experts;
    if (resident_experts.empty()) {
      resident_experts.reserve(kExpertsPerLayer);
      for (std::size_t expert = 0; expert < kExpertsPerLayer; ++expert) {
        resident_experts.push_back(static_cast<std::int32_t>(expert));
      }
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
    // Size cannot be checked yet: the expected size is a function of the
    // geometry, and the geometry lives in the header. Only rule out a file
    // too small to contain one.
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

    ExpertBankHeader header {};
    std::memcpy(&header, impl_->bank_mapping, sizeof(header));
    if (std::memcmp(header.magic, "SBEXP001", sizeof(header.magic)) != 0 ||
        header.reserved != 0) {
      return impl_->reject_load_locked(
          ProviderStatus::invalid_argument,
          "expert bank header does not match the Phase-1 NVFP4 B70 contract");
    }
    // The bank declares its own shape and this provider adopts it, rather
    // than the reverse. `adopt_bank_geometry` still rejects a header whose
    // record sizes disagree with its own dimensions, so a truncated or
    // foreign file cannot install a geometry that would then be used to
    // compute kernel arguments.
    if (!adopt_bank_geometry(header.num_layers, header.experts_per_layer,
                             header.hidden_size, header.intermediate_size,
                             header.w13_bytes, header.s13_bytes,
                             header.w2_bytes, header.s2_bytes)) {
      return impl_->reject_load_locked(
          ProviderStatus::invalid_argument,
          "expert bank header declares an inconsistent or unsupported "
          "geometry");
    }
    if (static_cast<std::uint64_t>(file_stat.st_size) !=
        expected_file_bytes()) {
      return impl_->reject_load_locked(
          ProviderStatus::invalid_argument,
          "expert bank size does not match the geometry its header declares");
    }

    const SelectedDevice selected = select_b70();
    sycl::async_handler async_handler =
        [state = impl_.get()](sycl::exception_list errors) noexcept {
          state->capture_async_errors(std::move(errors));
        };
    // Profiling is opt-in. Level Zero timestamps every command on a
    // profiled queue, and the two marker kernels that make kernel_us
    // meaningful are themselves full submissions — together a
    // significant share of dispatch latency at decode batch sizes,
    // where the whole point is to finish inside the concurrent CUDA
    // expert window. Set SHOOTING_BRAKE_B70_PROFILE=1 to get the
    // per-dispatch kernel/total breakdown back.
    impl_->profiling = profiling_requested();
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
    impl_->health.generation = config.generation;
    impl_->capability.device_name = selected.name;
    impl_->capability.device_memory_total_bytes =
        static_cast<std::uint64_t>(
            selected.device.get_info<sycl::info::device::global_mem_size>());
    impl_->capability.supported_hidden_sizes = {
        static_cast<std::uint32_t>(g_hidden)};
    impl_->capability.supported_intermediate_sizes = {
        static_cast<std::uint32_t>(g_intermediate)};
    impl_->capability.supported_topk = {static_cast<std::uint32_t>(kTopK)};
    const std::size_t resident_experts_per_layer =
        impl_->config.resident_experts.size();
    const std::size_t resident_experts_total =
        g_layers * resident_experts_per_layer;
    impl_->capability.num_resident_experts =
        static_cast<std::uint32_t>(resident_experts_total);
    impl_->capability.max_batch_remote =
        static_cast<std::uint32_t>(config.max_batch);
    impl_->capability.kernel_families = {
        "nvfp4_moe_split", "nvfp4_moe_fused"};
    impl_->capability.health_heartbeat_interval_ms = 1000;
    impl_->capability.num_layers = static_cast<std::uint32_t>(g_layers);
    impl_->capability.experts_per_layer =
        static_cast<std::uint32_t>(kExpertsPerLayer);

    // Refuse a placement that cannot fit before allocating any of it.
    // `persistent_device_bytes` already computed this figure, but only
    // afterwards, to report free memory — so an over-committed placement
    // surfaced as a raw SYCL allocation failure with the engine half up.
    // At the 122B's geometry one resident expert index costs ~238 MiB
    // across 47 layers, so the ceiling is easy to cross by accident and
    // the message needs to say how many would fit.
    const std::uint64_t required_bytes =
        persistent_device_bytes(config.max_batch, resident_experts_per_layer);
    const std::uint64_t device_bytes =
        impl_->capability.device_memory_total_bytes;
    if (device_bytes != 0 && required_bytes > device_bytes) {
      const std::uint64_t per_index =
          static_cast<std::uint64_t>(g_layers) *
          persistent_weight_bytes_per_expert();
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

    impl_->hidden =
        impl_->allocate_device<sycl::half>(config.max_batch * g_hidden);
    impl_->ids = impl_->allocate_device<std::int32_t>(config.max_batch * kTopK);
    impl_->weights = impl_->allocate_device<float>(config.max_batch * kTopK);
    impl_->scratch = impl_->allocate_device<float>(
        config.max_batch * kTopK * 2 * g_intermediate);
    impl_->output =
        impl_->allocate_device<float>(config.max_batch * g_hidden);
    impl_->copyout_staging =
        impl_->allocate_host<float>(config.max_batch * g_hidden);

    const std::size_t layer_staging_bytes =
        resident_experts_per_layer * g_w13_bytes;
    impl_->upload_staging =
        impl_->allocate_host<std::uint8_t>(layer_staging_bytes);

    const auto* records = static_cast<const std::uint8_t*>(impl_->bank_mapping) +
                          sizeof(ExpertBankHeader);
    const std::size_t kS13Offset = g_w13_bytes;
    const std::size_t kW2Offset = kS13Offset + g_s13_bytes;
    const std::size_t kS2Offset = kW2Offset + g_w2_bytes;
    const std::size_t kW13GlobalOffset = kS2Offset + g_s2_bytes;
    const std::size_t kW2GlobalOffset = kW13GlobalOffset + sizeof(float);

    for (std::size_t layer = 0; layer < g_layers; ++layer) {
      impl_->upload_plane(records, 0, g_w13_bytes, impl_->w13, layer);
      impl_->upload_plane(records, kS13Offset, g_s13_bytes, impl_->s13, layer);
      impl_->upload_plane(records, kW2Offset, g_w2_bytes, impl_->w2, layer);
      impl_->upload_plane(records, kS2Offset, g_s2_bytes, impl_->s2, layer);
      impl_->upload_globals(records, kW13GlobalOffset, impl_->w13_global, layer);
      impl_->upload_globals(records, kW2GlobalOffset, impl_->w2_global, layer);
    }

    sycl::free(impl_->upload_staging, *impl_->queue);
    impl_->upload_staging = nullptr;

    const std::uint64_t owned_device_bytes = persistent_device_bytes(
        config.max_batch, resident_experts_per_layer);
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

    const std::size_t route_elements = M * kTopK;
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

    const std::size_t first_expert = layer * resident_experts;
    const std::uint8_t* layer_w13 = impl_->w13 + first_expert * g_w13_bytes;
    const std::uint8_t* layer_s13 = impl_->s13 + first_expert * g_s13_bytes;
    const std::uint8_t* layer_w2 = impl_->w2 + first_expert * g_w2_bytes;
    const std::uint8_t* layer_s2 = impl_->s2 + first_expert * g_s2_bytes;
    const float* layer_w13_global = impl_->w13_global + first_expert;
    const float* layer_w2_global = impl_->w2_global + first_expert;

    const bool use_split = M <= 32;
    if (use_split) {
      quixicore::xpu::ops::nvfp4_moe_split(
          *impl_->queue, impl_->hidden, impl_->ids, impl_->weights, layer_w13,
          layer_s13, layer_w13_global, layer_w2, layer_s2, layer_w2_global,
          impl_->scratch, impl_->output, M, resident_experts, kTopK,
          g_hidden, g_intermediate, quixicore::xpu::DType::f16, true,
          quixicore::xpu::Variant::sycl, false);
    } else {
      quixicore::xpu::ops::nvfp4_moe_fused(
          *impl_->queue, impl_->hidden, impl_->ids, impl_->weights, layer_w13,
          layer_s13, layer_w13_global, layer_w2, layer_s2, layer_w2_global,
          impl_->output, M, resident_experts, kTopK, g_hidden,
          g_intermediate, quixicore::xpu::DType::f16, true,
          quixicore::xpu::Variant::sycl, false);
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
    impl_->copy_out.emplace(impl_->queue->memcpy(
        impl_->copyout_staging, impl_->output,
        M * g_hidden * sizeof(float)));
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

    std::memcpy(output, impl_->copyout_staging,
                required_elements * sizeof(float));
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
