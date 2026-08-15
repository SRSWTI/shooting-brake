// NVFP4 element decoders, shared by the grouped MoE kernels.
//
// Deliberately NOT included by nvfp4_moe.sycl.cpp, which keeps its own copies.
// That kernel is the numerical oracle the grouped path is validated against
// (tests/xpu_ops_smoke.cpp), and an oracle that shares its dequant with the
// code under test cannot catch a wrong dequant -- both sides would be wrong
// identically and the comparison would pass. The duplication is the point.

#pragma once

#include <cstdint>

#include <sycl/sycl.hpp>

namespace quixicore::xpu::kernels::nvfp4 {

// Both decoders below reinterpret the source bits as fp16 WITHOUT rebiasing the
// exponent, which is what makes them branchless -- and what makes them return
// values scaled by a constant: 2^-14 for e2m1, 2^-8 for e4m3. Their product is
// therefore the true weight times 2^-22, and this constant is what undoes that.
//
// It is a property of these decoders, NOT of the NVFP4 format: ModelOpt's
// dequant is plain `nibble * block_scale * scaling_factor_2`
// (modelopt/torch/quantization/qtensor/nvfp4_tensor.py:399-405).
//
// Apply it BEFORE narrowing to an fp16/bf16 matrix operand. A tile left at
// 2^-22 lands in fp16 subnormals, where the true product's 6 significant bits
// do not fit: measured 3% error for block scale 0x3B, 18% for 0x2D. Undone
// first, the same tile is exact.
inline constexpr float kGlobalScaleFixup = 4194304.0f; // 2^22

// One E2M1 nibble -> float. [s|ee|m] maps onto the top of an fp16 pattern:
// sign to bit 15, the 2-bit exponent and 1-bit mantissa to bits 12..9, which
// is exactly fp16's layout for this dynamic range.
inline float decode_e2m1(std::uint32_t nibble) {
  const std::uint32_t bits =
      ((nibble & 0x8u) << 12) | ((nibble & 0x7u) << 9);
  return static_cast<float>(
      sycl::bit_cast<sycl::half>(static_cast<std::uint16_t>(bits)));
}

// One E4M3 block scale byte -> float, via the fp16 bit pattern (sign to bit
// 15, exponent+mantissa shifted into place). E4M3's bias matches fp16's, so
// no exponent rebias is needed.
inline float decode_e4m3(std::uint8_t value) {
  const auto bits = static_cast<std::uint16_t>(
      ((value & 0x80u) << 8) | ((value & 0x7fu) << 7));
  return static_cast<float>(sycl::bit_cast<sycl::half>(bits));
}

// Select nibble `k` of a packed NVFP4 row: two values per byte, low nibble
// holds the even k. Matches the bank layout produced by src/phase1/extract_experts.
inline float decode_packed_element(const std::uint8_t *row, std::size_t k) {
  const std::uint8_t byte = row[k >> 1];
  const std::uint32_t nibble = (k & 1u) ? (byte >> 4) : (byte & 0x0fu);
  return decode_e2m1(nibble);
}

} // namespace quixicore::xpu::kernels::nvfp4
