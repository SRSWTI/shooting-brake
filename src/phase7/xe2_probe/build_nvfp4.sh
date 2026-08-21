#!/usr/bin/env bash
# Build the probe against OUR FORK of the kernel (src/phase7/xe2_nvfp4), not
# the vendored tree. Include order does the swap: our xe_2/ shadows theirs.
#
# The fork adds, so far:
#   - a group_size == 16 instantiation of the 4-bit mainloop
#   - tile_k = 16 policies, so tile_k == group_size for NVFP4's 16-element
#     block scales and the existing scale-reload gate stays correct
#
# Run the result with --group-size 16 to exercise the NVFP4 path shape, and
# with SYCL_UR_USE_LEVEL_ZERO_V2=0 -- oneAPI 2026.1's Level-Zero V2 adapter
# segfaults on a plain USM memcpy (null fn ptr in isGraphCaptureActive).
#
# No `set -u`: sourcing oneAPI setvars.sh references unset vars and aborts.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
V="$(cd ../../../vendor/intel-xpu/vllm-xpu/vllm-xpu-kernels && pwd)"
FORK="$(cd ../xe2_nvfp4 && pwd)"
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
icpx -fsycl -std=c++20 -O2 -o xe2_grouped_probe_nvfp4 xe2_grouped_probe.cpp \
  -DXE2_NVFP4_FORK \
  -DCUTLASS_ENABLE_HEADERS_ONLY -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET \
  -DCUTLASS_VERSIONS_GENERATED -DVLLM_XPU_ENABLE_XE2 \
  -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate \
  -I "$FORK" \
  -I "$V/csrc/xpu/grouped_gemm" -I "$V/.deps/cutlass-sycl-src/include" \
  -I "$V/.deps/cutlass-sycl-src/tools/util/include" -I "$V"
echo "built: $(pwd)/xe2_grouped_probe_nvfp4"
