#pragma once

#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"

namespace MoE {
using namespace cute;

class xe_gemm_policy_base {
 public:
  using WGTile = Shape<_256, _256, _32>;
  using SGLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;

  // Copy can be turned for better performance
  using GmemTiledCopyA = void;  // same as make_block_2d_copy_A
  using GmemTiledCopyB = void;  // same as make_block_2d_copy_B
  using GmemTiledCopyD = void;  // same as make_block_2d_copy_D
};

class w16a16_policy : public xe_gemm_policy_base {
 public:
  using GmemTiledCopyD = XE_STORE_2D<16, 8, 32>;
};

class w16a16_policy_n_128 : public xe_gemm_policy_base {
 public:
  using WGTile = Shape<_256, _128, _32>;
  using SGLayout = Layout<Shape<_8, _2, _1>, Stride<_2, _1, _0>>;
};

class w16a16_policy_n_64 : public xe_gemm_policy_base {
 public:
  using WGTile = Shape<_256, _64, _32>;
  using SGLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _0>>;
};

class w16a16_policy_m_8 : public xe_gemm_policy_base {
 public:
  using WGTile = Shape<_8, _64, _32>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
};

class w16a16_policy_m_16 : public xe_gemm_policy_base {
 public:
  using WGTile = Shape<_16, _64, _32>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
};

class w16a16_policy_m_32 : public xe_gemm_policy_base {
 public:
  using WGTile = Shape<_32, _64, _32>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
};

class w8a16_policy : public xe_gemm_policy_base {
 public:
  using WGTile = Shape<_128, _128, _16>;
  using SGLayout = Layout<Shape<_4, _2, _1>, Stride<_2, _1, _0>>;

  using GmemTiledCopyD = XE_STORE_2D<16, 8, 32>;
};

class w8a16_policy_m_8 : public xe_gemm_policy_base {
 public:
  using WGTile = Shape<_8, _64, _32>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
};

class w8a16_policy_m_16 : public xe_gemm_policy_base {
 public:
  using WGTile = Shape<_16, _64, _32>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
};

class w8a16_policy_m_32 : public xe_gemm_policy_base {
 public:
  using WGTile = Shape<_32, _64, _32>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
};

class w4a16_policy : public xe_gemm_policy_base {
 public:
  using WGTile = Shape<_128, _256, _32>;
  using SGLayout = Layout<Shape<_4, _8, _1>, Stride<_8, _1, _0>>;

  using GmemTiledCopyD = XE_STORE_2D<16, 8, 32>;
};

class w4a16_policy_m_8 : public xe_gemm_policy_base {
 public:
  using WGTile = Shape<_8, _64, _32>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
};

class w4a16_policy_m_16 : public xe_gemm_policy_base {
 public:
  using WGTile = Shape<_16, _64, _32>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
};

class w4a16_policy_m_32 : public xe_gemm_policy_base {
 public:
  using WGTile = Shape<_32, _64, _32>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
};

// NVFP4 additions: tile_k = 16 so that tile_k == group_size for NVFP4's
// 16-element block scales. That makes the existing scale-reload gate
// (`k_tile * tile_k % group_size == 0`) fire once per tile with exactly one
// scale group in flight, which is the whole reason these exist -- it avoids
// touching the mainloop at all. Halving tile_k does halve the work per K
// iteration, so these must be measured against the tile_k=32 variants rather
// than assumed equivalent.
class w4a16_policy_k16 : public xe_gemm_policy_base {
 public:
  using WGTile = Shape<_128, _256, _16>;
  using SGLayout = Layout<Shape<_4, _8, _1>, Stride<_8, _1, _0>>;

  using GmemTiledCopyD = XE_STORE_2D<16, 8, 32>;
};

class w4a16_policy_m_32_k16 : public xe_gemm_policy_base {
 public:
  using WGTile = Shape<_32, _64, _16>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
};

class w4a16_policy_m_16_k16 : public xe_gemm_policy_base {
 public:
  using WGTile = Shape<_16, _64, _16>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
};

}  // namespace MoE