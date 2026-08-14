#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace shooting_brake::phase3 {

inline constexpr std::uint32_t kReferenceHidden = 2048;
inline constexpr std::uint32_t kReferenceIntermediate = 512;
inline constexpr std::uint32_t kReferenceTopK = 8;
inline constexpr std::uint32_t kReferenceLayers = 2;
inline constexpr std::uint32_t kReferenceExperts = 8;
inline constexpr std::uint32_t kReferenceInputs = 8;

#pragma pack(push, 1)
struct ReferenceFixtureHeader final {
  char magic[8];
  std::uint32_t version;
  std::uint32_t header_bytes;
  std::uint32_t endian_tag;
  std::uint32_t hidden;
  std::uint32_t intermediate;
  std::uint32_t topk;
  std::uint32_t num_layers;
  std::uint32_t num_experts;
  std::uint32_t num_inputs;
  std::uint32_t reserved0;
  std::uint64_t layer_ids_offset;
  std::uint64_t expert_ids_offset;
  std::uint64_t hidden_fp16_offset;
  std::uint64_t nvfp4_outputs_offset;
  std::uint64_t source_outputs_offset;
  std::uint64_t file_bytes;
  char source_snapshot[40];
  char nvfp4_snapshot[40];
  std::uint8_t bank_sha256[32];
  std::uint8_t nvfp4_manifest_sha256[32];
  std::uint8_t reserved[16];
};
#pragma pack(pop)

static_assert(sizeof(ReferenceFixtureHeader) == 256);

std::array<std::uint8_t, 32> sha256_file(const std::string& path,
                                         std::uint64_t expected_bytes);

class ReferenceFixture final {
 public:
  ReferenceFixture() noexcept = default;
  ~ReferenceFixture() noexcept;
  ReferenceFixture(const ReferenceFixture&) = delete;
  ReferenceFixture& operator=(const ReferenceFixture&) = delete;
  ReferenceFixture(ReferenceFixture&& other) noexcept;
  ReferenceFixture& operator=(ReferenceFixture&& other) noexcept;

  static ReferenceFixture open(const std::string& path);

  const ReferenceFixtureHeader& header() const noexcept { return *header_; }
  std::span<const std::uint32_t> layer_ids() const noexcept;
  std::span<const std::int32_t> expert_ids() const noexcept;
  std::span<const std::uint16_t> hidden_bits(std::size_t input) const;
  std::span<const double> nvfp4_output(std::size_t layer_index,
                                      std::size_t input,
                                      std::size_t expert_index) const;
  std::span<const double> source_output(std::size_t layer_index,
                                       std::size_t input,
                                       std::size_t expert_index) const;
  std::size_t layer_index(std::uint32_t layer) const;
  std::size_t expert_index(std::int32_t expert) const;
  std::size_t file_bytes() const noexcept { return bytes_; }

 private:
  ReferenceFixture(int fd, void* mapping, std::size_t bytes) noexcept;
  void reset() noexcept;

  int fd_ = -1;
  void* mapping_ = nullptr;
  std::size_t bytes_ = 0;
  const ReferenceFixtureHeader* header_ = nullptr;
  const std::uint32_t* layer_ids_ = nullptr;
  const std::int32_t* expert_ids_ = nullptr;
  const std::uint16_t* hidden_fp16_ = nullptr;
  const double* nvfp4_outputs_ = nullptr;
  const double* source_outputs_ = nullptr;
};

}  // namespace shooting_brake::phase3
