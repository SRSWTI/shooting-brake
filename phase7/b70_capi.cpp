#include "b70_capi.h"

#include "b70_provider.hpp"

#include <cstring>
#include <new>

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
