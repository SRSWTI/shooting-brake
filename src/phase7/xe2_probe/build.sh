#!/usr/bin/env bash
# Build the Xe2 grouped-MoE-GEMM speed probe.
#
# The three non-obvious requirements, each of which fails the build on its own:
#  1. CUTLASS_ENABLE_SYCL + SYCL_INTEL_TARGET -- without them cute/config.hpp
#     reaches for <cuda_runtime_api.h> and stops.
#  2. The spirv-ext list -- the mainloop uses split barriers, 2D block IO and
#     DPAS. Without the extensions llvm-spirv exits 18 with
#     "RequiresExtension: SPV_INTEL_split_barrier", which does not name a file
#     or line and is not obviously a flag problem.
#  3. cutlass-sycl from .deps, NOT vendor/cutlass -- the latter is upstream
#     NVIDIA CUTLASS and has no Xe DPAS atoms.
# No `set -u`: sourcing oneAPI setvars.sh references unset vars and aborts.
# The serve recipes in benchmarks/ carry the same warning for the same reason.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
D="$(cd ../../../vendor/intel-xpu/vllm-xpu/vllm-xpu-kernels && pwd)"
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
icpx -fsycl -std=c++20 -O2 -o xe2_grouped_probe xe2_grouped_probe.cpp \
  -DCUTLASS_ENABLE_HEADERS_ONLY -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET \
  -DCUTLASS_VERSIONS_GENERATED -DVLLM_XPU_ENABLE_XE2 \
  -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate \
  -I "$D/csrc/xpu/grouped_gemm" -I "$D/.deps/cutlass-sycl-src/include" \
  -I "$D/.deps/cutlass-sycl-src/tools/util/include" -I "$D"
echo "built: $(pwd)/xe2_grouped_probe"
# Run with SYCL_UR_USE_LEVEL_ZERO_V2=0 -- oneAPI 2026.1's Level-Zero V2 adapter
# segfaults on a plain USM memcpy (null fn ptr in isGraphCaptureActive).
