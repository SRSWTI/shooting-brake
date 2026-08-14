#include "reference_fixture.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace shooting_brake::phase3 {
namespace {

constexpr std::array<char, 8> kMagic{'S', 'B', 'P', '3', 'R', 'F', '0', '1'};
constexpr std::string_view kSourceSnapshot =
    "995ad96eacd98c81ed38be0c5b274b04031597b0";
constexpr std::string_view kNvfp4Snapshot =
    "739af1e7aac320af1682ed1e0cce369af4c5265d";
constexpr std::string_view kBankSha256 =
    "0ce6377ba3c9848da42b6063574ea884052d2e0f5e605d86d1684a1e5826e8db";
constexpr std::string_view kManifestSha256 =
    "320fad67387d36509947a691fa269d5a55dfb08f0cd7da6434868a6861bff2fa";
constexpr std::string_view kFixtureSha256 =
    "3ebac16d0f09907cee4718ac1054d21939e420eabaf76ebe79c75fa5d0132606";
constexpr std::array<std::uint32_t, kReferenceLayers> kLayerIds{0U, 31U};
constexpr std::array<std::int32_t, kReferenceExperts> kExpertIds{
    0, 1, 7, 63, 127, 191, 254, 255};

[[noreturn]] void invalid(const std::string& detail) {
  throw std::runtime_error("invalid Phase-3 reference fixture: " + detail);
}

std::string errno_message(const char* operation) {
  return std::string(operation) + ": " + std::strerror(errno);
}

constexpr std::uint64_t align64(std::uint64_t value) noexcept {
  return (value + 63U) & ~std::uint64_t{63U};
}

bool all_zero(const void* data, std::size_t bytes) noexcept {
  const auto* values = static_cast<const std::uint8_t*>(data);
  for (std::size_t index = 0; index < bytes; ++index) {
    if (values[index] != 0U) {
      return false;
    }
  }
  return true;
}

constexpr std::uint8_t hex_digit(char value) noexcept {
  return value >= '0' && value <= '9'
             ? static_cast<std::uint8_t>(value - '0')
             : static_cast<std::uint8_t>(value - 'a' + 10);
}

bool exact_hex(const std::uint8_t* actual, std::string_view expected) noexcept {
  if (expected.size() != 64U) {
    return false;
  }
  for (std::size_t index = 0; index < 32U; ++index) {
    const std::uint8_t value = static_cast<std::uint8_t>(
        (hex_digit(expected[index * 2U]) << 4U) |
        hex_digit(expected[index * 2U + 1U]));
    if (actual[index] != value) {
      return false;
    }
  }
  return true;
}

void require_region(std::uint64_t actual, std::uint64_t expected,
                    std::string_view name) {
  if (actual != expected || (actual & 63U) != 0U) {
    invalid(std::string(name) + " offset/alignment mismatch");
  }
}

void validate_unique_ids(std::span<const std::uint32_t> values,
                         std::string_view name) {
  for (std::size_t left = 0; left < values.size(); ++left) {
    for (std::size_t right = left + 1U; right < values.size(); ++right) {
      if (values[left] == values[right]) {
        invalid(std::string(name) + " contains duplicate IDs");
      }
    }
  }
}

void validate_unique_ids(std::span<const std::int32_t> values,
                         std::string_view name) {
  for (std::size_t left = 0; left < values.size(); ++left) {
    for (std::size_t right = left + 1U; right < values.size(); ++right) {
      if (values[left] == values[right]) {
        invalid(std::string(name) + " contains duplicate IDs");
      }
    }
  }
}

constexpr std::uint32_t rotate_right(std::uint32_t value,
                                     std::uint32_t bits) noexcept {
  return (value >> bits) | (value << (32U - bits));
}

struct Sha256 final {
  std::uint32_t state[8] = {
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
  std::uint8_t block[64]{};
  std::uint32_t used = 0;
  std::uint64_t bytes = 0;
};

constexpr std::uint32_t kShaConstants[64] = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

void sha256_block(Sha256* sha) noexcept {
  std::uint32_t words[64]{};
  for (std::uint32_t index = 0; index < 16U; ++index) {
    words[index] =
        (static_cast<std::uint32_t>(sha->block[index * 4U]) << 24U) |
        (static_cast<std::uint32_t>(sha->block[index * 4U + 1U]) << 16U) |
        (static_cast<std::uint32_t>(sha->block[index * 4U + 2U]) << 8U) |
        sha->block[index * 4U + 3U];
  }
  for (std::uint32_t index = 16U; index < 64U; ++index) {
    const std::uint32_t first =
        rotate_right(words[index - 15U], 7U) ^
        rotate_right(words[index - 15U], 18U) ^
        (words[index - 15U] >> 3U);
    const std::uint32_t second =
        rotate_right(words[index - 2U], 17U) ^
        rotate_right(words[index - 2U], 19U) ^
        (words[index - 2U] >> 10U);
    words[index] =
        words[index - 16U] + first + words[index - 7U] + second;
  }
  std::uint32_t a = sha->state[0];
  std::uint32_t b = sha->state[1];
  std::uint32_t c = sha->state[2];
  std::uint32_t d = sha->state[3];
  std::uint32_t e = sha->state[4];
  std::uint32_t f = sha->state[5];
  std::uint32_t g = sha->state[6];
  std::uint32_t h = sha->state[7];
  for (std::uint32_t index = 0; index < 64U; ++index) {
    const std::uint32_t sigma_one =
        rotate_right(e, 6U) ^ rotate_right(e, 11U) ^ rotate_right(e, 25U);
    const std::uint32_t choose = (e & f) ^ ((~e) & g);
    const std::uint32_t temp_one =
        h + sigma_one + choose + kShaConstants[index] + words[index];
    const std::uint32_t sigma_zero =
        rotate_right(a, 2U) ^ rotate_right(a, 13U) ^ rotate_right(a, 22U);
    const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
    const std::uint32_t temp_two = sigma_zero + majority;
    h = g;
    g = f;
    f = e;
    e = d + temp_one;
    d = c;
    c = b;
    b = a;
    a = temp_one + temp_two;
  }
  sha->state[0] += a;
  sha->state[1] += b;
  sha->state[2] += c;
  sha->state[3] += d;
  sha->state[4] += e;
  sha->state[5] += f;
  sha->state[6] += g;
  sha->state[7] += h;
}

void sha256_update(Sha256* sha, const void* data,
                   std::uint32_t count) noexcept {
  const auto* input = static_cast<const std::uint8_t*>(data);
  sha->bytes += count;
  while (count != 0U) {
    const std::uint32_t take =
        std::min<std::uint32_t>(count, 64U - sha->used);
    std::memcpy(sha->block + sha->used, input, take);
    sha->used += take;
    input += take;
    count -= take;
    if (sha->used == 64U) {
      sha256_block(sha);
      sha->used = 0U;
    }
  }
}

std::array<std::uint8_t, 32> sha256_finish(Sha256* sha) noexcept {
  const std::uint64_t bit_count = sha->bytes * 8U;
  const std::uint8_t one = 0x80U;
  const std::uint8_t zero = 0U;
  sha256_update(sha, &one, 1U);
  while (sha->used != 56U) {
    sha256_update(sha, &zero, 1U);
  }
  std::uint8_t length[8]{};
  for (std::uint32_t index = 0; index < 8U; ++index) {
    length[7U - index] =
        static_cast<std::uint8_t>(bit_count >> (index * 8U));
  }
  sha256_update(sha, length, sizeof(length));
  std::array<std::uint8_t, 32> digest{};
  for (std::uint32_t index = 0; index < 8U; ++index) {
    digest[index * 4U] =
        static_cast<std::uint8_t>(sha->state[index] >> 24U);
    digest[index * 4U + 1U] =
        static_cast<std::uint8_t>(sha->state[index] >> 16U);
    digest[index * 4U + 2U] =
        static_cast<std::uint8_t>(sha->state[index] >> 8U);
    digest[index * 4U + 3U] =
        static_cast<std::uint8_t>(sha->state[index]);
  }
  return digest;
}

}  // namespace
std::array<std::uint8_t, 32> sha256_file(const std::string& path,
                                         std::uint64_t expected_bytes) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    throw std::runtime_error(errno_message("open SHA256 input"));
  }
  struct stat status {};
  if (::fstat(fd, &status) != 0 || !S_ISREG(status.st_mode) ||
      status.st_size < 0 ||
      static_cast<std::uint64_t>(status.st_size) != expected_bytes) {
    static_cast<void>(::close(fd));
    throw std::runtime_error("SHA256 input exact size/type mismatch: " + path);
  }
  Sha256 sha;
  std::array<std::uint8_t, 64U * 1024U> buffer{};
  std::uint64_t consumed = 0;
  while (consumed < expected_bytes) {
    const std::size_t requested = static_cast<std::size_t>(
        std::min<std::uint64_t>(buffer.size(), expected_bytes - consumed));
    const ssize_t bytes = ::read(fd, buffer.data(), requested);
    if (bytes < 0 && errno == EINTR) {
      continue;
    }
    if (bytes <= 0) {
      static_cast<void>(::close(fd));
      throw std::runtime_error("SHA256 input read failed or truncated: " +
                               path);
    }
    sha256_update(&sha, buffer.data(), static_cast<std::uint32_t>(bytes));
    consumed += static_cast<std::uint64_t>(bytes);
  }
  if (::close(fd) != 0) {
    throw std::runtime_error(errno_message("close SHA256 input"));
  }
  return sha256_finish(&sha);
}


ReferenceFixture::ReferenceFixture(int fd, void* mapping,
                                   std::size_t bytes) noexcept
    : fd_(fd), mapping_(mapping), bytes_(bytes) {
  const auto* base = static_cast<const std::byte*>(mapping_);
  header_ = reinterpret_cast<const ReferenceFixtureHeader*>(base);
  layer_ids_ = reinterpret_cast<const std::uint32_t*>(
      base + header_->layer_ids_offset);
  expert_ids_ = reinterpret_cast<const std::int32_t*>(
      base + header_->expert_ids_offset);
  hidden_fp16_ = reinterpret_cast<const std::uint16_t*>(
      base + header_->hidden_fp16_offset);
  nvfp4_outputs_ = reinterpret_cast<const double*>(
      base + header_->nvfp4_outputs_offset);
  source_outputs_ = reinterpret_cast<const double*>(
      base + header_->source_outputs_offset);
}

ReferenceFixture::~ReferenceFixture() noexcept { reset(); }

ReferenceFixture::ReferenceFixture(ReferenceFixture&& other) noexcept
    : fd_(std::exchange(other.fd_, -1)),
      mapping_(std::exchange(other.mapping_, nullptr)),
      bytes_(std::exchange(other.bytes_, 0)),
      header_(std::exchange(other.header_, nullptr)),
      layer_ids_(std::exchange(other.layer_ids_, nullptr)),
      expert_ids_(std::exchange(other.expert_ids_, nullptr)),
      hidden_fp16_(std::exchange(other.hidden_fp16_, nullptr)),
      nvfp4_outputs_(std::exchange(other.nvfp4_outputs_, nullptr)),
      source_outputs_(std::exchange(other.source_outputs_, nullptr)) {}

ReferenceFixture& ReferenceFixture::operator=(ReferenceFixture&& other) noexcept {
  if (this != &other) {
    reset();
    fd_ = std::exchange(other.fd_, -1);
    mapping_ = std::exchange(other.mapping_, nullptr);
    bytes_ = std::exchange(other.bytes_, 0);
    header_ = std::exchange(other.header_, nullptr);
    layer_ids_ = std::exchange(other.layer_ids_, nullptr);
    expert_ids_ = std::exchange(other.expert_ids_, nullptr);
    hidden_fp16_ = std::exchange(other.hidden_fp16_, nullptr);
    nvfp4_outputs_ = std::exchange(other.nvfp4_outputs_, nullptr);
    source_outputs_ = std::exchange(other.source_outputs_, nullptr);
  }
  return *this;
}

void ReferenceFixture::reset() noexcept {
  if (mapping_ != nullptr) {
    static_cast<void>(::munmap(mapping_, bytes_));
  }
  if (fd_ >= 0) {
    static_cast<void>(::close(fd_));
  }
  fd_ = -1;
  mapping_ = nullptr;
  bytes_ = 0;
  header_ = nullptr;
  layer_ids_ = nullptr;
  expert_ids_ = nullptr;
  hidden_fp16_ = nullptr;
  nvfp4_outputs_ = nullptr;
  source_outputs_ = nullptr;
}

ReferenceFixture ReferenceFixture::open(const std::string& path) {
  constexpr std::uint64_t kExpectedFixtureBytes = 4'227'456ULL;
  const std::array<std::uint8_t, 32> fixture_digest =
      sha256_file(path, kExpectedFixtureBytes);
  if (!exact_hex(fixture_digest.data(), kFixtureSha256)) {
    invalid("whole-file SHA256 mismatch");
  }
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    throw std::runtime_error(errno_message("open reference fixture"));
  }
  struct stat status {};
  if (::fstat(fd, &status) != 0) {
    const std::string detail = errno_message("fstat reference fixture");
    static_cast<void>(::close(fd));
    throw std::runtime_error(detail);
  }
  if (!S_ISREG(status.st_mode) || status.st_size < 0 ||
      static_cast<std::uint64_t>(status.st_size) <
          sizeof(ReferenceFixtureHeader) ||
      static_cast<std::uint64_t>(status.st_size) >
          std::numeric_limits<std::size_t>::max()) {
    static_cast<void>(::close(fd));
    invalid("not a bounded regular file");
  }
  const std::size_t bytes = static_cast<std::size_t>(status.st_size);
  void* mapping = ::mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
  if (mapping == MAP_FAILED) {
    const std::string detail = errno_message("mmap reference fixture");
    static_cast<void>(::close(fd));
    throw std::runtime_error(detail);
  }

  const auto cleanup_invalid = [&](const std::string& detail) -> void {
    static_cast<void>(::munmap(mapping, bytes));
    static_cast<void>(::close(fd));
    invalid(detail);
  };
  const auto* header = static_cast<const ReferenceFixtureHeader*>(mapping);
  if (std::memcmp(header->magic, kMagic.data(), kMagic.size()) != 0 ||
      header->version != 1U || header->header_bytes != 256U ||
      header->endian_tag != 0x01020304U ||
      header->hidden != kReferenceHidden ||
      header->intermediate != kReferenceIntermediate ||
      header->topk != kReferenceTopK ||
      header->num_layers != kReferenceLayers ||
      header->num_experts != kReferenceExperts ||
      header->num_inputs != kReferenceInputs || header->reserved0 != 0U) {
    cleanup_invalid("header identity or geometry mismatch");
  }
  if (std::memcmp(header->source_snapshot, kSourceSnapshot.data(), 40U) != 0 ||
      std::memcmp(header->nvfp4_snapshot, kNvfp4Snapshot.data(), 40U) != 0 ||
      !exact_hex(header->bank_sha256, kBankSha256) ||
      !exact_hex(header->nvfp4_manifest_sha256, kManifestSha256) ||
      !all_zero(header->reserved, sizeof(header->reserved))) {
    cleanup_invalid("snapshot, fingerprint, or reserved header mismatch");
  }

  constexpr std::uint64_t layer_bytes =
      kReferenceLayers * sizeof(std::uint32_t);
  constexpr std::uint64_t expert_bytes =
      kReferenceExperts * sizeof(std::int32_t);
  constexpr std::uint64_t hidden_bytes =
      kReferenceInputs * kReferenceHidden * sizeof(std::uint16_t);
  constexpr std::uint64_t output_elements =
      static_cast<std::uint64_t>(kReferenceLayers) * kReferenceInputs *
      kReferenceExperts * kReferenceHidden;
  constexpr std::uint64_t output_bytes = output_elements * sizeof(double);
  const std::uint64_t expected_layers = align64(sizeof(ReferenceFixtureHeader));
  const std::uint64_t expected_experts = align64(expected_layers + layer_bytes);
  const std::uint64_t expected_hidden = align64(expected_experts + expert_bytes);
  const std::uint64_t expected_nvfp4 = align64(expected_hidden + hidden_bytes);
  const std::uint64_t expected_source = align64(expected_nvfp4 + output_bytes);
  const std::uint64_t expected_file = expected_source + output_bytes;
  if (header->file_bytes != bytes || header->file_bytes != expected_file) {
    cleanup_invalid("exact file size mismatch");
  }
  try {
    require_region(header->layer_ids_offset, expected_layers, "layer IDs");
    require_region(header->expert_ids_offset, expected_experts, "expert IDs");
    require_region(header->hidden_fp16_offset, expected_hidden, "hidden FP16");
    require_region(header->nvfp4_outputs_offset, expected_nvfp4,
                   "NVFP4 outputs");
    require_region(header->source_outputs_offset, expected_source,
                   "source outputs");
  } catch (...) {
    static_cast<void>(::munmap(mapping, bytes));
    static_cast<void>(::close(fd));
    throw;
  }

  const auto* base = static_cast<const std::byte*>(mapping);
  if (!all_zero(base + expected_layers + layer_bytes,
                static_cast<std::size_t>(
                    expected_experts - (expected_layers + layer_bytes))) ||
      !all_zero(base + expected_experts + expert_bytes,
                static_cast<std::size_t>(
                    expected_hidden - (expected_experts + expert_bytes)))) {
    cleanup_invalid("nonzero region-alignment padding");
  }
  const auto* layers = reinterpret_cast<const std::uint32_t*>(
      base + header->layer_ids_offset);
  const auto* experts = reinterpret_cast<const std::int32_t*>(
      base + header->expert_ids_offset);
  if (!std::equal(kLayerIds.begin(), kLayerIds.end(), layers) ||
      !std::equal(kExpertIds.begin(), kExpertIds.end(), experts)) {
    cleanup_invalid("fixed layer/expert ID set mismatch");
  }
  try {
    validate_unique_ids({layers, kReferenceLayers}, "layer IDs");
    validate_unique_ids({experts, kReferenceExperts}, "expert IDs");
  } catch (...) {
    static_cast<void>(::munmap(mapping, bytes));
    static_cast<void>(::close(fd));
    throw;
  }

  const auto* hidden = reinterpret_cast<const std::uint16_t*>(
      base + header->hidden_fp16_offset);
  for (std::uint64_t index = 0;
       index < static_cast<std::uint64_t>(kReferenceInputs) * kReferenceHidden;
       ++index) {
    if ((hidden[index] & 0x7c00U) == 0x7c00U) {
      cleanup_invalid("hidden FP16 region contains non-finite values");
    }
  }
  const auto* nvfp4 = reinterpret_cast<const double*>(
      base + header->nvfp4_outputs_offset);
  const auto* source = reinterpret_cast<const double*>(
      base + header->source_outputs_offset);
  for (std::uint64_t index = 0; index < output_elements; ++index) {
    if (!std::isfinite(nvfp4[index]) || !std::isfinite(source[index])) {
      cleanup_invalid("per-expert output region contains non-finite values");
    }
  }
  return ReferenceFixture(fd, mapping, bytes);
}

std::span<const std::uint32_t> ReferenceFixture::layer_ids() const noexcept {
  return {layer_ids_, kReferenceLayers};
}

std::span<const std::int32_t> ReferenceFixture::expert_ids() const noexcept {
  return {expert_ids_, kReferenceExperts};
}

std::span<const std::uint16_t> ReferenceFixture::hidden_bits(
    std::size_t input) const {
  if (input >= kReferenceInputs) {
    throw std::out_of_range("reference input index");
  }
  return {hidden_fp16_ + input * kReferenceHidden, kReferenceHidden};
}

std::span<const double> ReferenceFixture::nvfp4_output(
    std::size_t layer, std::size_t input, std::size_t expert) const {
  if (layer >= kReferenceLayers || input >= kReferenceInputs ||
      expert >= kReferenceExperts) {
    throw std::out_of_range("NVFP4 reference index");
  }
  const std::size_t offset =
      ((layer * kReferenceInputs + input) * kReferenceExperts + expert) *
      kReferenceHidden;
  return {nvfp4_outputs_ + offset, kReferenceHidden};
}

std::span<const double> ReferenceFixture::source_output(
    std::size_t layer, std::size_t input, std::size_t expert) const {
  if (layer >= kReferenceLayers || input >= kReferenceInputs ||
      expert >= kReferenceExperts) {
    throw std::out_of_range("source reference index");
  }
  const std::size_t offset =
      ((layer * kReferenceInputs + input) * kReferenceExperts + expert) *
      kReferenceHidden;
  return {source_outputs_ + offset, kReferenceHidden};
}

std::size_t ReferenceFixture::layer_index(std::uint32_t layer) const {
  for (std::size_t index = 0; index < kReferenceLayers; ++index) {
    if (layer_ids_[index] == layer) {
      return index;
    }
  }
  throw std::out_of_range("fixture does not contain requested layer");
}

std::size_t ReferenceFixture::expert_index(std::int32_t expert) const {
  for (std::size_t index = 0; index < kReferenceExperts; ++index) {
    if (expert_ids_[index] == expert) {
      return index;
    }
  }
  throw std::out_of_range("fixture does not contain requested expert");
}

}  // namespace shooting_brake::phase3
