#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SCRIPT_DIR}/guidellm}"
PYTHON_SITE="${PYTHON_SITE:-${PROJECT_DIR}/.venv/lib/python3.13/site-packages}"
CUDA_ROOT="${PYTHON_SITE}/nvidia/cu13"

if [[ ! -x "${PROJECT_DIR}/.venv/bin/vllm" ]]; then
  printf 'Error: vLLM is not installed at %s/.venv/bin/vllm\n' "${PROJECT_DIR}" >&2
  exit 1
fi
if [[ ! -x "${CUDA_ROOT}/bin/nvcc" ]]; then
  printf 'Error: CUDA 13 compiler not found at %s/bin/nvcc\n' "${CUDA_ROOT}" >&2
  exit 1
fi

# NVIDIA's pip CUDA layout may omit linker-compatible aliases required by JITs.
if [[ ! -e "${CUDA_ROOT}/lib64" ]]; then
  ln -s lib "${CUDA_ROOT}/lib64"
fi
if [[ ! -e "${CUDA_ROOT}/lib/libcudart.so" ]]; then
  ln -s libcudart.so.13 "${CUDA_ROOT}/lib/libcudart.so"
fi

export CUDA_HOME="${CUDA_ROOT}"
export PATH="${CUDA_ROOT}/bin:${PATH}"

cd "${PROJECT_DIR}"
exec .venv/bin/vllm serve srswti/axe-superveloce-jota-118b-r15-nvfp4 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 160000 \
  --max-num-seqs 6 \
  --port 8016 \
  --enable-auto-tool-choice \
  --tool-call-parser poolside_v1 \
  --reasoning-parser poolside_v1 \
  --enable-mfu-metrics \
  --no-enable-prefix-caching
