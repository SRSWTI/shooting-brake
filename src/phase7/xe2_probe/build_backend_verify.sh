#!/usr/bin/env bash
# Builds the grouped-backend A/B verifier against the phase7 objects
# (native fork + oneDNN TU). Requires `make -C src/phase7 b70` to have run
# so the two .o files exist. No `set -u`: setvars references unset vars.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
P7="$(cd .. && pwd)"
ONEDNN="$(cd ../../../vendor/oneDNN && pwd)"
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
icpx -fsycl -std=gnu++20 -O2 -Wall \
  -I "$P7/xe2_nvfp4" \
  -o grouped_backend_verify grouped_backend_verify.cpp \
  "$P7/grouped_moe.o" "$P7/grouped_moe_onednn.o" \
  -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate \
  -L "$ONEDNN/build/src" -ldnnl \
  -Wl,--disable-new-dtags -Wl,-rpath,"$ONEDNN/build/src"
echo "built: $(pwd)/grouped_backend_verify"
