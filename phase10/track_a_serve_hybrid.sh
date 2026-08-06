#!/usr/bin/env bash
# Shooting Brake — Track A: launch the hybrid vLLM server.
#
# Serves Qwen3.6-35B-A3B-NVFP4 with the Shooting Brake hybrid MoE path
# active: B70-owned experts compute on the Intel Arc Pro B70 via the
# QuixiCore NVFP4 kernel, CUDA-owned experts on the RTX 5090, dispatch
# inside a normal CUDA graph. The 5090 stays the CUDA state owner
# (scheduler, attention, KV cache, LM head); the B70 only computes
# routed-expert partials.
#
# max_model_len is the load-bearing knob: with VRAM surgery freeing
# ~6.5 GiB of expert weights, the hybrid gets ~4.2x the KV cache of
# all-CUDA and can therefore serve contexts (up to 131072) that all-CUDA
# cannot.  Lower it for a strict head-to-head with the Phase 0 baseline.
#
# Usage:
#   bash phase10/track_a_serve_hybrid.sh                 # 131k, subset:16:8
#   MAX_MODEL_LEN=32768 bash phase10/track_a_serve_hybrid.sh
#   PLACEMENT=split:128 PORT=8001 bash phase10/track_a_serve_hybrid.sh
#
# Env overrides (defaults shown):
#   MODEL=unsloth/Qwen3.6-35B-A3B-NVFP4
#   MAX_MODEL_LEN=131072         # native context ceiling is 262144
#   MAX_NUM_SEQS=64              # all-CUDA caps at 83 (GDN state cache)
#   GPU_MEM_UTIL=0.90
#   PLACEMENT=subset:16:8        # offload policy, see track_b_offload_sweep.sh
#   PORT=8000
#
# Stop with:  pkill -INT -f 'vllm serve'
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="${MODEL:-unsloth/Qwen3.6-35B-A3B-NVFP4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
PLACEMENT="${PLACEMENT:-subset:16:8}"
PORT="${PORT:-8000}"

# oneAPI runtime must be in the server process before SYCL loads.
# shellcheck disable=SC1091
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
export PATH="$REPO_ROOT/.venv/bin:${PATH:-}"

# Adapter switches.  VLLM_PLUGINS loads the out-of-tree adapter; the
# SHOOTING_BRAKE_* vars select the hybrid path and the offload policy.
export VLLM_PLUGINS=shooting_brake_vllm
export SHOOTING_BRAKE_PHASE4=all-cuda
export SHOOTING_BRAKE_MODEL="$MODEL"
export SHOOTING_BRAKE_PLACEMENT="$PLACEMENT"
export SHOOTING_BRAKE_HYBRID=1
export SHOOTING_BRAKE_B70_DEVICE=1
export SHOOTING_BRAKE_VRAM_SURGERY=1
export SHOOTING_BRAKE_B70_GRAPH=1
export SHOOTING_BRAKE_B70_STATS=1
# Bank and provider .so resolve relative to the repo root.
export SHOOTING_BRAKE_B70_BANK="$REPO_ROOT/phase1/expert_bank.bin"
export SHOOTING_BRAKE_B70_LIB="$REPO_ROOT/phase7/libsb_b70_provider.so"

echo "[track_a] serving $MODEL hybrid ($PLACEMENT) max_model_len=$MAX_MODEL_LEN on :$PORT"

exec "$REPO_ROOT/.venv/bin/vllm" serve "$MODEL" \
  --host 0.0.0.0 --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --dtype bfloat16 \
  --trust-remote-code \
  --disable-log-requests
