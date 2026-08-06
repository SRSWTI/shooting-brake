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
# The matrix sweeps prompt length (contexts) against concurrency.  At
# short contexts both all-CUDA and hybrid serve; the interesting cells
# are the long ones, where all-CUDA cannot admit the request and the
# hybrid's 4.2x KV cache is the only reason it runs at all.
#
# Compare contexts <= 8192 against the existing Phase 0 all-CUDA
# baseline (already recorded).  Above 8192 there is no baseline to
# compare against — that gap is the capacity result.
#
# Prerequisites:
#   1. Server up:   bash phase10/track_a_serve_hybrid.sh   (in another shell)
#   2. Server ready: curl -s http://127.0.0.1:8000/health && curl -s http://127.0.0.1:8000/metrics | grep -c vllm
#
# Usage:
#   bash phase10/track_a_guidellm_matrix.sh
#   CONTEXTS=1024,4096,8192 RATES=1,4 OUTPUT_ROOT=./bench-matrix/smoke \
#     bash phase10/track_a_guidellm_matrix.sh
#
# Env overrides (defaults shown):
#   TARGET=http://127.0.0.1:8000
#   MODEL=unsloth/Qwen3.6-35B-A3B-NVFP4
#   CONTEXTS=1024,4096,8192,16384,32768,65536,98304,127000
#   RATES=1,2,3,4,5,6
#   OUTPUT_TOKENS=512
#   MAX_REQUESTS=20            # per cell; sweep needs >= 10
#   MAX_SECONDS=180            # scaled up automatically for long contexts
#   OUTPUT_ROOT=$PWD/bench-matrix/hybrid_131k_c6
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MATRIX="$REPO_ROOT/benchmarks-vllm/guidellm/scripts/run_vllm_matrix.py"

TARGET="${TARGET:-http://127.0.0.1:8000}"
METRICS="${TARGET}/metrics"
MODEL="${MODEL:-unsloth/Qwen3.6-35B-A3B-NVFP4}"
CONTEXTS="${CONTEXTS:-1024,4096,8192,16384,32768,65536,98304,127000}"
RATES="${RATES:-1,2,3,4,5,6}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-512}"
MAX_REQUESTS="${MAX_REQUESTS:-20}"
MAX_SECONDS="${MAX_SECONDS:-180}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/bench-matrix/hybrid_131k_c6}"

# Preflight: server healthy and exposing Prometheus metrics.  guidellm's
# matrix scraper depends on vllm: metrics being present.
if ! curl -sf "$TARGET/health" >/dev/null 2>&1; then
  echo "[track_a] server not healthy at $TARGET/health — start it first:" >&2
  echo "[track_a]   bash phase10/track_a_serve_hybrid.sh" >&2
  exit 1
fi
if ! curl -sf "$METRICS" 2>/dev/null | grep -q '^vllm:'; then
  echo "[track_a] no vllm: metrics at $METRICS — server still starting?" >&2
  exit 1
fi

echo "[track_a] matrix -> $OUTPUT_ROOT"
echo "[track_a] contexts=$CONTEXTS  rates=$RATES  out=$OUTPUT_TOKENS tok"

exec "$REPO_ROOT/.venv/bin/python" "$MATRIX" \
  --target "$TARGET" \
  --metrics-url "$METRICS" \
  --model "$MODEL" \
  --contexts "$CONTEXTS" \
  --concurrent-rates "$RATES" \
  --profiles synchronous,concurrent,sweep \
  --sweep-steps 3 \
  --output-tokens "$OUTPUT_TOKENS" \
  --max-requests "$MAX_REQUESTS" \
  --max-seconds "$MAX_SECONDS" \
  --warmup 0 --cooldown 0 \
  --sample-interval 5 \
  --output-root "$OUTPUT_ROOT" \
  --skip-existing
