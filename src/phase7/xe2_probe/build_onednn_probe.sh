#!/usr/bin/env bash
# Builds the gate-2 oneDNN grouped-NVFP4 probe against the vendor/oneDNN
# build tree. oneAPI must be sourced. Run the probe with the fresh lib pinned:
#   LD_LIBRARY_PATH=$ONEDNN/build/src ./onednn_grouped_probe
set -euo pipefail
cd "$(dirname "$0")"
ONEDNN=../../../vendor/oneDNN
icpx -fsycl -std=gnu++20 -O2 -Wall \
  -I"$ONEDNN/include" -I"$ONEDNN/build/include" \
  -o onednn_grouped_probe onednn_grouped_probe.cpp \
  -L"$ONEDNN/build/src" -ldnnl
echo "built: $(pwd)/onednn_grouped_probe"
