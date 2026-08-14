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
#   bash benchmarks/serve_hybrid.sh                 # 131k, subset:16:8
#   MAX_MODEL_LEN=32768 bash benchmarks/serve_hybrid.sh
#   PLACEMENT=split:128 PORT=8001 bash benchmarks/serve_hybrid.sh
#   PREFILL_STREAM=1 bash benchmarks/serve_hybrid.sh   # 2.2x at long context
#
# Env overrides (defaults shown):
#   MODEL=unsloth/Qwen3.6-35B-A3B-NVFP4
#   MAX_MODEL_LEN=131072         # native context ceiling is 262144
#   MAX_NUM_SEQS=64              # all-CUDA caps at 83 (GDN state cache)
#   GPU_MEM_UTIL=0.90
#   PLACEMENT=subset:16:8        # offload policy, see track_b_offload_sweep.sh
#   PORT=8000
#   PREFILL_STREAM=0             # see below
#
# PREFILL_STREAM=1 sends the B70-owned expert weights from a host-DRAM mirror
# to the 5090 once per forward, instead of shipping every prompt token to the
# B70 once per active layer. Measured 2026-08-13 on this machine, 10/10
# requests per cell, `benchmarks/matrix/longctx/RESULT.md`:
#
#     ctx      TTFT  63.6s -> 26.7s      e2e  8.5 -> 18.7 tok/s   (127,000)
#              TTFT  31.0s -> 11.2s      e2e 16.5 -> 38.8 tok/s   ( 65,536)
#
# Decode ITL and KV capacity are unchanged (the mirror lives in host DRAM, not
# 5090 VRAM); it costs ~4-6 GiB of host RAM. Default is 0 only because short
# prompts have not been re-measured with it on -- SHOOTING_BRAKE_B70_STREAM_T
# (default 1024 tokens) is supposed to keep them on the dispatch path, but that
# has not been verified here. Turn it on for long-context serving.
#
# Stop with:  pkill -INT -f 'vllm serve'
#
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# oneAPI's setvars.sh terminates the sourcing shell under `set -e`, so
# source it BEFORE enabling strict mode. The SYCL runtime must be live in
# this process before the B70 provider loads (lazily, on first forward).
# shellcheck disable=SC1091
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
export PATH="$REPO_ROOT/.venv/bin:${PATH:-}"

# Strict mode only for our own logic below.
set -euo pipefail

MODEL="${MODEL:-unsloth/Qwen3.6-35B-A3B-NVFP4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
PLACEMENT="${PLACEMENT:-subset:16:8}"
PORT="${PORT:-8000}"

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
export SHOOTING_BRAKE_B70_PREFILL_STREAM="${PREFILL_STREAM:-0}"
# Bank and provider .so resolve relative to the repo root.
export SHOOTING_BRAKE_B70_BANK="$REPO_ROOT/phase1/expert_bank.bin"
export SHOOTING_BRAKE_B70_LIB="$REPO_ROOT/phase7/libsb_b70_provider.so"

echo "[track_a] serving $MODEL hybrid ($PLACEMENT) max_model_len=$MAX_MODEL_LEN on :$PORT" \
     "prefill=$([ "${PREFILL_STREAM:-0}" = 1 ] && echo stream || echo dispatch)"

exec "$REPO_ROOT/.venv/bin/vllm" serve "$MODEL" \
  --host 0.0.0.0 --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --dtype bfloat16 \
  --trust-remote-code \
  --enable-mfu-metrics \
  --kv-cache-metrics
