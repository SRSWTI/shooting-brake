#!/usr/bin/env bash
# Builds the standalone device-USM host-shadow reproducer.
# Self-contained: SYCL only, no project headers, no oneDNN, no checkpoint.
# No `set -u`: oneAPI setvars references unset vars.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ -z "${SETVARS_COMPLETED}" ]]; then
  source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
fi

icpx -fsycl -std=gnu++17 -O2 -Wall -Wextra \
  -o device_usm_host_shadow device_usm_host_shadow.cpp

echo "built: $(pwd)/device_usm_host_shadow"
echo
echo "usage:"
echo "  ./device_usm_host_shadow --help"
echo "  ./device_usm_host_shadow --gib 24 --chunk 2 --device 0"
echo
echo "note: run with the inference server stopped, on an otherwise idle card."
