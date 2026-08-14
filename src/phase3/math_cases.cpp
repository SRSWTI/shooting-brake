#include "math_cases.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace shooting_brake::phase3 {
namespace {

constexpr std::array<std::int32_t, kReferenceExperts> kExperts{
    0, 1, 7, 63, 127, 191, 254, 255};

[[noreturn]] void invalid_case(const std::string& detail) {
  throw std::invalid_argument("invalid canonical Phase-3 case: " + detail);
}

CanonicalToken base_token(std::uint32_t row, std::uint32_t input) {
  CanonicalToken token{};
  token.original_row = row;
  token.input_index = input;
  for (CanonicalRoute& route : token.routes) {
    route = {-1, 0.0F, RouteOwner::local};
  }
  return token;
}

void validate_remote_ownership(std::uint32_t layer, std::int32_t global,
                               const ReferenceFixture& fixture) {
  static_cast<void>(fixture.layer_index(layer));
  static_cast<void>(fixture.expert_index(global));
  if (global < 0 || global >= 256) {
    invalid_case("remote global expert is outside the provider placement");
  }
  // The wire remains canonical-global. The current qualified full 256-expert
  // bank has an identity global-to-local map, but that provider-private
  // translation is centralized in the server rather than encoded here.
}

void validate_case(const CanonicalCase& test_case) {
  if (test_case.name.empty() || test_case.full_batch == 0U ||
      test_case.full_batch > 128U || test_case.tokens.size() != test_case.full_batch ||
      test_case.layer >= 32U) {
    invalid_case("name, shape, or layer is invalid");
  }
  std::uint32_t previous_row = 0;
  for (std::size_t token_index = 0; token_index < test_case.tokens.size();
       ++token_index) {
    const CanonicalToken& token = test_case.tokens[token_index];
    if (token.original_row >= test_case.full_batch ||
        (token_index != 0U && token.original_row <= previous_row) ||
        token.input_index >= kReferenceInputs) {
      invalid_case("token row map or input index is invalid");
    }
    previous_row = token.original_row;
    double weight_sum = 0.0;
    for (const CanonicalRoute& route : token.routes) {
      if (!std::isfinite(route.weight) || route.weight < 0.0F) {
        invalid_case("canonical post-top-k route weight is invalid");
      }
      weight_sum += route.weight;
      if (route.global_expert < 0 || route.global_expert >= 256) {
        invalid_case("canonical route expert is invalid");
      }
    }
    if (std::abs(weight_sum - 1.0) > 1.0e-5) {
      invalid_case(test_case.name + " row " + std::to_string(token_index) +
                   " post-top-k weight sum " + std::to_string(weight_sum) +
                   " is not normalized");
    }
  }
}

std::size_t staged_rows(const CanonicalCase& test_case) noexcept {
  std::size_t rows = 0;
  for (const CanonicalToken& token : test_case.tokens) {
    if (std::any_of(token.routes.begin(), token.routes.end(),
                    [](const CanonicalRoute& route) {
                      return route.owner == RouteOwner::remote;
                    })) {
      ++rows;
    }
  }
  return rows;
}

}  // namespace

std::span<const std::uint16_t> StagedCase::activations() const noexcept {
  return {activation_fp16.data(),
          static_cast<std::size_t>(rows) * kReferenceHidden};
}
std::span<const std::int32_t> StagedCase::canonical_ids() const noexcept {
  return {canonical_global_ids.data(),
          static_cast<std::size_t>(rows) * kReferenceTopK};
}
std::span<const float> StagedCase::route_weights() const noexcept {
  return {weights.data(), static_cast<std::size_t>(rows) * kReferenceTopK};
}
std::span<const std::uint8_t> StagedCase::mask() const noexcept {
  return {remote_mask.data(),
          static_cast<std::size_t>(rows) * kReferenceTopK};
}
std::span<const std::uint32_t> StagedCase::row_map() const noexcept {
  return {token_row_map.data(), rows};
}
std::span<const std::uint16_t> StagedCase::positions() const noexcept {
  return {route_positions.data(),
          static_cast<std::size_t>(rows) * kReferenceTopK};
}

std::vector<CanonicalCase> make_one_route_sweep() {
  std::vector<CanonicalCase> cases;
  cases.reserve(128U);
  for (std::uint32_t count = 1; count <= 128U; ++count) {
    CanonicalCase test_case;
    test_case.name = "one-route-M=" + std::to_string(count);
    test_case.layer = 0U;
    test_case.full_batch = count;
    test_case.tokens.reserve(count);
    for (std::uint32_t row = 0; row < count; ++row) {
      CanonicalToken token = base_token(row, row % kReferenceInputs);
      constexpr std::array<std::int32_t, kReferenceTopK> local_ids{
          2, 3, 4, 5, 6, 8, 9, 10};
      for (std::size_t route = 0; route < kReferenceTopK; ++route) {
        token.routes[route] = {
            local_ids[route], 0.125F, RouteOwner::local};
      }
      const std::size_t position = (static_cast<std::size_t>(row) * 3U) %
                                   kReferenceTopK;
      token.routes[position] = {
          kExperts[row % kExperts.size()], 0.125F, RouteOwner::remote};
      test_case.tokens.push_back(token);
    }
    cases.push_back(std::move(test_case));
  }
  return cases;
}

CanonicalCase make_all_remote_case() {
  CanonicalCase test_case;
  test_case.name = "all-remote-semantic-M=4";
  test_case.layer = 0U;
  test_case.full_batch = 4U;
  constexpr std::array<std::array<std::int32_t, kReferenceTopK>, 4> ids{{
      {{255, 0, 7, 7, 127, 1, 254, 63}},
      {{63, 191, 0, 255, 1, 127, 7, 254}},
      {{254, 7, 191, 0, 255, 63, 127, 1}},
      {{1, 255, 63, 191, 0, 254, 7, 7}},
  }};
  constexpr std::array<std::array<float, kReferenceTopK>, 4> weights{{
      {{0.25F, 0.125F, 0.1875F, 0.0625F, 0.21875F, 0.09375F,
        0.000244140625F, 0.062255859375F}},
      {{0.03125F, 0.21875F, 0.0625F, 0.40625F, 0.09375F, 0.0078125F,
        0.15625F, 0.0234375F}},
      {{0.09375F, 0.28125F, 0.0234375F, 0.046875F, 0.34375F, 0.140625F,
        0.0546875F, 0.015625F}},
      {{0.203125F, 0.0390625F, 0.3203125F, 0.01171875F, 0.02734375F,
        0.171875F, 0.0859375F, 0.140625F}},
  }};
  for (std::uint32_t row = 0; row < 4U; ++row) {
    CanonicalToken token = base_token(row, row);
    for (std::size_t position = 0; position < kReferenceTopK; ++position) {
      token.routes[position] = {
          ids[row][position], weights[row][position], RouteOwner::remote};
    }
    test_case.tokens.push_back(token);
  }
  return test_case;
}

CanonicalCase make_mixed_case(bool changed_local_routes) {
  CanonicalCase test_case;
  test_case.name = changed_local_routes ? "mixed-local-perturbed-Mremote=4"
                                        : "mixed-local-baseline-Mremote=4";
  test_case.layer = 31U;
  test_case.full_batch = 10U;
  constexpr std::array<std::uint32_t, 4> remote_rows{0U, 2U, 5U, 9U};
  constexpr std::array<std::array<std::uint32_t, 3>, 4> remote_positions{{
      {{1U, 4U, 7U}}, {{0U, 3U, 6U}}, {{2U, 5U, 7U}}, {{1U, 3U, 6U}},
  }};
  constexpr std::array<std::array<std::int32_t, 3>, 4> remote_ids{{
      {{255, 7, 0}}, {{63, 255, 1}}, {{254, 63, 127}}, {{0, 191, 254}},
  }};
  constexpr std::array<std::array<float, 3>, 4> remote_weights{{
      {{0.3125F, 0.046875F, 0.015625F}},
      {{0.1875F, 0.09375F, 0.0078125F}},
      {{0.03125F, 0.40625F, 0.0625F}},
      {{0.140625F, 0.0234375F, 0.265625F}},
  }};

  for (std::uint32_t row = 0; row < test_case.full_batch; ++row) {
    CanonicalToken token = base_token(row, (row * 5U + 1U) % kReferenceInputs);
    constexpr std::array<std::int32_t, kReferenceTopK> baseline_local_ids{
        2, 3, 4, 5, 6, 8, 9, 10};
    constexpr std::array<std::int32_t, kReferenceTopK> changed_local_ids{
        11, 12, 13, 14, 15, 16, 17, 18};
    for (std::size_t position = 0; position < kReferenceTopK; ++position) {
      token.routes[position] = {
          changed_local_routes ? changed_local_ids[position]
                               : baseline_local_ids[position],
          0.0F, RouteOwner::local};
    }
    const auto found =
        std::find(remote_rows.begin(), remote_rows.end(), row);
    if (found != remote_rows.end()) {
      const std::size_t remote_row =
          static_cast<std::size_t>(found - remote_rows.begin());
      for (std::size_t route = 0; route < 3U; ++route) {
        const std::size_t position = remote_positions[remote_row][route];
        token.routes[position] = {remote_ids[remote_row][route],
                                  remote_weights[remote_row][route],
                                  RouteOwner::remote};
      }
    }
    double remote_sum = 0.0;
    double local_factor_sum = 0.0;
    for (std::size_t position = 0; position < kReferenceTopK; ++position) {
      if (token.routes[position].owner == RouteOwner::remote) {
        remote_sum += token.routes[position].weight;
      } else {
        local_factor_sum +=
            changed_local_routes ? static_cast<double>(kReferenceTopK - position)
                                 : static_cast<double>(position + 1U);
      }
    }
    const double local_total = 1.0 - remote_sum;
    for (std::size_t position = 0; position < kReferenceTopK; ++position) {
      if (token.routes[position].owner == RouteOwner::local) {
        const double factor =
            changed_local_routes ? static_cast<double>(kReferenceTopK - position)
                                 : static_cast<double>(position + 1U);
        token.routes[position].weight =
            static_cast<float>(local_total * factor / local_factor_sum);
      }
    }
    test_case.tokens.push_back(token);
  }
  return test_case;
}

CanonicalCase make_zero_remote_case() {
  CanonicalCase test_case;
  test_case.name = "zero-remote-bypass";
  test_case.layer = 31U;
  test_case.full_batch = 4U;
  for (std::uint32_t row = 0; row < test_case.full_batch; ++row) {
    CanonicalToken token = base_token(row, row);
    for (std::size_t position = 0; position < kReferenceTopK; ++position) {
      token.routes[position] = {
          static_cast<std::int32_t>((row * 31U + position) % 256U),
          0.125F, RouteOwner::local};
    }
    test_case.tokens.push_back(token);
  }
  return test_case;
}

StagedCase stage_remote_routes(const CanonicalCase& test_case,
                               const ReferenceFixture& fixture) {
  validate_case(test_case);
  StagedCase staged;
  staged.layer = test_case.layer;
  staged.full_batch = test_case.full_batch;
  staged.canonical_global_ids.fill(-1);
  staged.weights.fill(0.0F);
  staged.remote_mask.fill(0U);
  staged.route_positions.fill(0xffffU);

  for (const CanonicalToken& token : test_case.tokens) {
    const bool has_remote = std::any_of(
        token.routes.begin(), token.routes.end(), [](const CanonicalRoute& route) {
          return route.owner == RouteOwner::remote;
        });
    if (!has_remote) {
      continue;
    }
    if (staged.rows >= 128U) {
      invalid_case("staged row capacity exceeded");
    }
    const std::size_t row = staged.rows++;
    const std::span<const std::uint16_t> hidden =
        fixture.hidden_bits(token.input_index);
    std::memcpy(staged.activation_fp16.data() + row * kReferenceHidden,
                hidden.data(), hidden.size_bytes());
    staged.token_row_map[row] = token.original_row;
    for (std::size_t position = 0; position < kReferenceTopK; ++position) {
      const CanonicalRoute& route = token.routes[position];
      const std::size_t index = row * kReferenceTopK + position;
      if (route.owner == RouteOwner::remote) {
        validate_remote_ownership(test_case.layer, route.global_expert,
                                  fixture);
        staged.canonical_global_ids[index] = route.global_expert;
        staged.weights[index] = route.weight;
        staged.remote_mask[index] = 1U;
        staged.route_positions[index] = static_cast<std::uint16_t>(position);
        ++staged.remote_routes;
      }
    }
  }
  if (staged.rows != staged_rows(test_case)) {
    invalid_case("staged row accounting mismatch");
  }
  return staged;
}

OracleResult compute_nvfp4_oracle(const CanonicalCase& test_case,
                                  const ReferenceFixture& fixture) {
  validate_case(test_case);
  OracleResult oracle;
  oracle.rows = static_cast<std::uint32_t>(staged_rows(test_case));
  oracle.expected.assign(static_cast<std::size_t>(oracle.rows) *
                             kReferenceHidden,
                         0.0);
  oracle.weighted_magnitude.assign(oracle.expected.size(), 0.0);
  oracle.sum_abs_weights.assign(oracle.rows, 0.0);
  oracle.remote_route_counts.assign(oracle.rows, 0U);
  const std::size_t layer = fixture.layer_index(test_case.layer);

  std::size_t output_row = 0;
  for (const CanonicalToken& token : test_case.tokens) {
    const bool has_remote = std::any_of(
        token.routes.begin(), token.routes.end(), [](const CanonicalRoute& route) {
          return route.owner == RouteOwner::remote;
        });
    if (!has_remote) {
      continue;
    }
    for (const CanonicalRoute& route : token.routes) {
      if (route.owner != RouteOwner::remote) {
        continue;
      }
      const double alpha = static_cast<double>(route.weight);
      const std::span<const double> expert = fixture.nvfp4_output(
          layer, token.input_index, fixture.expert_index(route.global_expert));
      oracle.sum_abs_weights[output_row] += std::abs(alpha);
      ++oracle.remote_route_counts[output_row];
      for (std::size_t column = 0; column < kReferenceHidden; ++column) {
        const double product = alpha * expert[column];
        const std::size_t index = output_row * kReferenceHidden + column;
        oracle.expected[index] += product;
        oracle.weighted_magnitude[index] += std::abs(product);
      }
    }
    ++output_row;
  }
  return oracle;
}

bool remote_materialization_equal(const CanonicalCase& left,
                                  const CanonicalCase& right,
                                  const ReferenceFixture& fixture) {
  const StagedCase a = stage_remote_routes(left, fixture);
  const StagedCase b = stage_remote_routes(right, fixture);
  if (a.layer != b.layer || a.full_batch != b.full_batch || a.rows != b.rows ||
      a.remote_routes != b.remote_routes) {
    return false;
  }
  return a.activations().size_bytes() == b.activations().size_bytes() &&
         std::memcmp(a.activations().data(), b.activations().data(),
                     a.activations().size_bytes()) == 0 &&
         std::memcmp(a.canonical_ids().data(), b.canonical_ids().data(),
                     a.canonical_ids().size_bytes()) == 0 &&
         std::memcmp(a.route_weights().data(), b.route_weights().data(),
                     a.route_weights().size_bytes()) == 0 &&
         std::memcmp(a.mask().data(), b.mask().data(), a.mask().size_bytes()) == 0 &&
         std::memcmp(a.row_map().data(), b.row_map().data(),
                     a.row_map().size_bytes()) == 0 &&
         std::memcmp(a.positions().data(), b.positions().data(),
                     a.positions().size_bytes()) == 0;
}

bool small_route_is_sensitivity_checked(const CanonicalCase& test_case,
                                        const ReferenceFixture& fixture) {
  CanonicalCase without_small_route = test_case;
  std::size_t target_row = std::numeric_limits<std::size_t>::max();
  std::size_t staged_row = 0;
  float alpha = 0.0F;
  std::uint32_t target_input = 0;
  std::int32_t target_expert = -1;
  for (CanonicalToken& token : without_small_route.tokens) {
    const bool has_remote = std::any_of(
        token.routes.begin(), token.routes.end(), [](const CanonicalRoute& route) {
          return route.owner == RouteOwner::remote;
        });
    if (!has_remote) {
      continue;
    }
    for (CanonicalRoute& route : token.routes) {
      if (route.owner == RouteOwner::remote &&
          std::bit_cast<std::uint32_t>(route.weight) ==
              std::bit_cast<std::uint32_t>(0.000244140625F)) {
        if (target_row != std::numeric_limits<std::size_t>::max()) {
          return false;
        }
        target_row = staged_row;
        alpha = route.weight;
        target_input = token.input_index;
        target_expert = route.global_expert;
        route.owner = RouteOwner::local;
      }
    }
    ++staged_row;
  }
  if (target_row == std::numeric_limits<std::size_t>::max()) {
    return false;
  }
  const OracleResult full = compute_nvfp4_oracle(test_case, fixture);
  const OracleResult without =
      compute_nvfp4_oracle(without_small_route, fixture);
  constexpr double unit_roundoff = 0x1p-24;
  constexpr double gamma_two =
      2.0 * unit_roundoff / (1.0 - 2.0 * unit_roundoff);
  const std::span<const double> target_output = fixture.nvfp4_output(
      fixture.layer_index(test_case.layer), target_input,
      fixture.expert_index(target_expert));
  for (std::size_t column = 0; column < kReferenceHidden; ++column) {
    const std::size_t index = target_row * kReferenceHidden + column;
    const double delta = std::abs(full.expected[index] - without.expected[index]);
    const double contribution =
        std::abs(static_cast<double>(alpha) * target_output[column]);
    const double budget =
        1.0e-6 * std::abs(static_cast<double>(alpha)) +
        (1.0e-2 + gamma_two) * contribution;
    if (delta > budget) {
      return true;
    }
  }
  return false;
}

double output_budget(const OracleResult& oracle, std::size_t row,
                     std::size_t column) noexcept {
  if (row >= oracle.rows || column >= kReferenceHidden) {
    return 0.0;
  }
  constexpr double unit_roundoff = 0x1p-24;
  const double operations =
      2.0 * static_cast<double>(oracle.remote_route_counts[row]);
  const double gamma =
      operations * unit_roundoff / (1.0 - operations * unit_roundoff);
  const std::size_t index = row * kReferenceHidden + column;
  return 1.0e-6 * oracle.sum_abs_weights[row] +
         (1.0e-2 + gamma) * oracle.weighted_magnitude[index];
}

std::vector<ArtifactMetrics> compute_artifact_metrics(
    const ReferenceFixture& fixture) {
  std::vector<ArtifactMetrics> metrics;
  metrics.reserve(kReferenceLayers * kReferenceExperts + 1U);
  long double aggregate_squared = 0.0L;
  long double aggregate_source_squared = 0.0L;
  long double aggregate_nvfp4_squared = 0.0L;
  long double aggregate_dot = 0.0L;
  std::size_t aggregate_count = 0;
  ArtifactMetrics aggregate;
  aggregate.layer = std::numeric_limits<std::uint32_t>::max();
  aggregate.expert = -1;

  for (std::size_t layer = 0; layer < kReferenceLayers; ++layer) {
    for (std::size_t expert = 0; expert < kReferenceExperts; ++expert) {
      ArtifactMetrics metric;
      metric.layer = fixture.layer_ids()[layer];
      metric.expert = fixture.expert_ids()[expert];
      long double squared = 0.0L;
      long double source_squared = 0.0L;
      long double nvfp4_squared = 0.0L;
      long double dot = 0.0L;
      std::size_t count = 0;
      for (std::size_t input = 0; input < kReferenceInputs; ++input) {
        const std::span<const double> nvfp4 =
            fixture.nvfp4_output(layer, input, expert);
        const std::span<const double> source =
            fixture.source_output(layer, input, expert);
        for (std::size_t column = 0; column < kReferenceHidden; ++column) {
          const double difference = nvfp4[column] - source[column];
          metric.max_absolute =
              std::max(metric.max_absolute, std::abs(difference));
          aggregate.max_absolute =
              std::max(aggregate.max_absolute, std::abs(difference));
          const long double difference_squared =
              static_cast<long double>(difference) * difference;
          const long double source_square =
              static_cast<long double>(source[column]) * source[column];
          const long double nvfp4_square =
              static_cast<long double>(nvfp4[column]) * nvfp4[column];
          const long double product =
              static_cast<long double>(nvfp4[column]) * source[column];
          squared += difference_squared;
          source_squared += source_square;
          nvfp4_squared += nvfp4_square;
          dot += product;
          aggregate_squared += difference_squared;
          aggregate_source_squared += source_square;
          aggregate_nvfp4_squared += nvfp4_square;
          aggregate_dot += product;
          ++count;
          ++aggregate_count;
        }
      }
      if (source_squared == 0.0L || nvfp4_squared == 0.0L) {
        throw std::runtime_error(
            "artifact metric has a zero source or NVFP4 norm");
      }
      metric.rms = std::sqrt(static_cast<double>(squared / count));
      metric.source_rms =
          std::sqrt(static_cast<double>(source_squared / count));
      metric.relative_rms = metric.rms / metric.source_rms;
      metric.cosine = static_cast<double>(
          dot / std::sqrt(nvfp4_squared * source_squared));
      metrics.push_back(metric);
    }
  }
  if (aggregate_source_squared == 0.0L ||
      aggregate_nvfp4_squared == 0.0L || aggregate_count == 0U) {
    throw std::runtime_error("aggregate artifact metric has a zero norm");
  }
  aggregate.rms =
      std::sqrt(static_cast<double>(aggregate_squared / aggregate_count));
  aggregate.source_rms = std::sqrt(
      static_cast<double>(aggregate_source_squared / aggregate_count));
  aggregate.relative_rms = aggregate.rms / aggregate.source_rms;
  aggregate.cosine = static_cast<double>(
      aggregate_dot /
      std::sqrt(aggregate_nvfp4_squared * aggregate_source_squared));
  metrics.push_back(aggregate);
  return metrics;
}

}  // namespace shooting_brake::phase3
