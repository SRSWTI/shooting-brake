#pragma once

#include "reference_fixture.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace shooting_brake::phase3 {

enum class RouteOwner : std::uint8_t { local, remote };

struct CanonicalRoute final {
  std::int32_t global_expert = -1;
  float weight = 0.0F;
  RouteOwner owner = RouteOwner::local;
};

struct CanonicalToken final {
  std::uint32_t original_row = 0;
  std::uint32_t input_index = 0;
  std::array<CanonicalRoute, kReferenceTopK> routes{};
};

struct CanonicalCase final {
  std::string name;
  std::uint32_t layer = 0;
  std::uint32_t full_batch = 0;
  std::vector<CanonicalToken> tokens;
};

struct StagedCase final {
  std::uint32_t layer = 0;
  std::uint32_t full_batch = 0;
  std::uint32_t rows = 0;
  std::uint32_t remote_routes = 0;
  std::array<std::uint16_t, 128U * kReferenceHidden> activation_fp16{};
  std::array<std::int32_t, 128U * kReferenceTopK> canonical_global_ids{};
  std::array<float, 128U * kReferenceTopK> weights{};
  std::array<std::uint8_t, 128U * kReferenceTopK> remote_mask{};
  std::array<std::uint32_t, 128U> token_row_map{};
  std::array<std::uint16_t, 128U * kReferenceTopK> route_positions{};

  std::span<const std::uint16_t> activations() const noexcept;
  std::span<const std::int32_t> canonical_ids() const noexcept;
  std::span<const float> route_weights() const noexcept;
  std::span<const std::uint8_t> mask() const noexcept;
  std::span<const std::uint32_t> row_map() const noexcept;
  std::span<const std::uint16_t> positions() const noexcept;
};

struct OracleResult final {
  std::uint32_t rows = 0;
  std::vector<double> expected;
  std::vector<double> weighted_magnitude;
  std::vector<double> sum_abs_weights;
  std::vector<std::uint32_t> remote_route_counts;
};

struct ArtifactMetrics final {
  std::uint32_t layer = 0;
  std::int32_t expert = -1;
  double max_absolute = 0.0;
  double rms = 0.0;
  double source_rms = 0.0;
  double relative_rms = 0.0;
  double cosine = 0.0;
};

std::vector<CanonicalCase> make_one_route_sweep();
CanonicalCase make_all_remote_case();
CanonicalCase make_mixed_case(bool changed_local_routes);
CanonicalCase make_zero_remote_case();

StagedCase stage_remote_routes(const CanonicalCase& test_case,
                               const ReferenceFixture& fixture);
OracleResult compute_nvfp4_oracle(const CanonicalCase& test_case,
                                  const ReferenceFixture& fixture);
bool remote_materialization_equal(const CanonicalCase& left,
                                  const CanonicalCase& right,
                                  const ReferenceFixture& fixture);
bool small_route_is_sensitivity_checked(const CanonicalCase& test_case,
                                        const ReferenceFixture& fixture);
double output_budget(const OracleResult& oracle, std::size_t row,
                     std::size_t column) noexcept;
std::vector<ArtifactMetrics> compute_artifact_metrics(
    const ReferenceFixture& fixture);

}  // namespace shooting_brake::phase3
