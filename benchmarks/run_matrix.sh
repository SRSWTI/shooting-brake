#!/usr/bin/env bash
# Shooting Brake — Track A: run the guidellm SLO matrix against a live
# hybrid server (started by track_a_serve_hybrid.sh).
#
# What this measures.  guidellm drives the server's OpenAI endpoint at
# controlled offered load and reports the serving-grade metrics that
# matter in production:
#
#   * synchronous profile  — one request at a time; pure latency floor
#                            (TTFT, inter-token latency, decode tok/s).
#   * concurrent profile   — fixed concurrency (rates 1,2,3,4,5,6);
#                            throughput and ITL tail under load.
#   * sweep profile        — interpolates rate from baseline to peak to
#                            trace the throughput/latency curve and the
#                            point the SLO breaks.
#
# The matrix sweeps prompt length (context) against concurrency.  At
# short contexts both all-CUDA and hybrid serve; the interesting cells
# are the long ones, where all-CUDA cannot admit the request and the
# hybrid's 4.2x KV cache is the only reason it runs at all.
#
# Compare contexts <= 8192 against the existing Phase 0 all-CUDA
# baseline (already recorded).  Above 8192 there is no baseline to
# compare against — that gap is the capacity result.
#
# Thermal pacing: the wrapper loops one (context, profile) cell at a
# time so cooldowns land exactly where they should — a longer rest when
# moving to a new context length, a shorter one between the three
# profiles within a context.  Pair with a power cap:
#   bash benchmarks/gpu_power.sh cap 575
#
# Prerequisites:
#   1. Power cap set:  bash benchmarks/gpu_power.sh cap 575
#   2. Server up:      bash benchmarks/serve_hybrid.sh  (own shell)
#   3. Server ready:   curl -s http://127.0.0.1:8000/health
#
# Usage:
#   bash benchmarks/run_matrix.sh
#   CONTEXTS=1024,4096,8192 COOLDOWN_CTX=60 bash benchmarks/run_matrix.sh
#
# Env overrides (defaults shown):
#   TARGET=http://127.0.0.1:8000
#   MODEL=unsloth/Qwen3.6-35B-A3B-NVFP4
#   CONTEXTS=1024,4096,8192,16384,32768,65536,98304,127000
#   RATES=1,2,3,4,5,6
#   PROFILES=synchronous,concurrent,sweep
#   OUTPUT_TOKENS=512
#   MAX_REQUESTS=20            # per cell; sweep needs >= 10
#   MAX_SECONDS=180            # scaled up automatically for long contexts
#   COOLDOWN_CTX=120           # seconds to idle between context lengths
#   COOLDOWN_PROFILE=15        # seconds to idle between profiles
#   OUTPUT_ROOT=$PWD/bench-matrix/hybrid_131k_c6
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MATRIX="$REPO_ROOT/benchmarks/matrix_runner.py"

TARGET="${TARGET:-http://127.0.0.1:8000}"
METRICS="${TARGET}/metrics"
MODEL="${MODEL:-unsloth/Qwen3.6-35B-A3B-NVFP4}"
CONTEXTS="${CONTEXTS:-1024,4096,8192,16384,32768,65536,98304,127000}"
RATES="${RATES:-1,2,3,4,5,6}"
PROFILES="${PROFILES:-synchronous,concurrent,sweep}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-512}"
MAX_REQUESTS="${MAX_REQUESTS:-20}"
MAX_SECONDS="${MAX_SECONDS:-180}"
COOLDOWN_CTX="${COOLDOWN_CTX:-120}"
COOLDOWN_PROFILE="${COOLDOWN_PROFILE:-15}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/bench-matrix/hybrid_131k_c6}"

# Preflight: server healthy and exposing Prometheus metrics.
if ! curl -sf "$TARGET/health" >/dev/null 2>&1; then
  echo "[track_a] server not healthy at $TARGET/health — start it first:" >&2
  echo "[track_a]   bash benchmarks/serve_hybrid.sh" >&2
  exit 1
fi
if ! curl -sf "$METRICS" 2>/dev/null | grep -q '^vllm:'; then
  echo "[track_a] no vllm: metrics at $METRICS — server still starting?" >&2
  exit 1
fi

IFS=',' read -ra CTX_LIST <<< "$CONTEXTS"
IFS=',' read -ra PROFILE_LIST <<< "$PROFILES"
n_ctx="${#CTX_LIST[@]}"

echo "[track_a] matrix -> $OUTPUT_ROOT"
echo "[track_a] ${n_ctx} contexts x ${#PROFILE_LIST[@]} profiles; out=$OUTPUT_TOKENS tok"
echo "[track_a] cooldown: ${COOLDOWN_CTX}s between contexts, ${COOLDOWN_PROFILE}s between profiles"

for ci in "${!CTX_LIST[@]}"; do
  ctx="${CTX_LIST[$ci]}"
  for pi in "${!PROFILE_LIST[@]}"; do
    profile="${PROFILE_LIST[$pi]}"
    echo "[track_a] ($(date +%H:%M:%S)) context=$ctx profile=$profile"
    "$REPO_ROOT/.venv/bin/python" "$MATRIX" \
      --target "$TARGET" \
      --metrics-url "$METRICS" \
      --model "$MODEL" \
      --contexts "$ctx" \
      --concurrent-rates "$RATES" \
      --profiles "$profile" \
      --sweep-steps 3 \
      --output-tokens "$OUTPUT_TOKENS" \
      --max-requests "$MAX_REQUESTS" \
      --max-seconds "$MAX_SECONDS" \
      --warmup 0 --cooldown 0 \
      --sample-interval 5 \
      --output-root "$OUTPUT_ROOT" \
      --skip-existing

    # Short rest between profiles within a context (not after the last
    # profile of the last context).
    if ! [[ "$pi" -eq "$(( ${#PROFILE_LIST[@]} - 1 ))" && "$ci" -eq "$(( n_ctx - 1 ))" ]]; then
      echo "[track_a] cooling ${COOLDOWN_PROFILE}s..."
      sleep "$COOLDOWN_PROFILE"
    fi
  done

  # Longer rest when moving to a new context length (not after the last).
  if [[ "$ci" -ne $(( n_ctx - 1 )) ]]; then
    echo "[track_a] context boundary — cooling ${COOLDOWN_CTX}s..."
    sleep "$COOLDOWN_CTX"
  fi
done

echo "[track_a] completed output_root=$OUTPUT_ROOT"
