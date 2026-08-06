#!/usr/bin/env bash
# Shooting Brake — Track B: in-process offload sweep.
#
# What this measures.  Track A holds the architecture fixed and sweeps
# load.  Track B holds the load fixed and sweeps the architecture: how
# much expert capacity to move to the B70, and in how many layers.
#
# Each B70-active layer costs one dispatch per token (a fixed overhead
# that does not shrink with the number of routes it sends), while the
# total number of offloaded experts decides how much CUDA VRAM is freed
# for KV cache.  So two knobs matter:
#
#   * active layers  — how many of the 32 NVFP4 layers own B70 experts.
#                      Fewer active layers = fewer dispatches = lower
#                      decode latency, for the same freed VRAM.
#   * experts/layer  — how many experts each active layer keeps on CUDA.
#                      Fewer CUDA experts = more freed VRAM = more KV,
#                      but a larger share of each token's routes hit B70.
#
# "subset:<active>:<cuda_per_layer>" concentrates the offload; "split:N"
# spreads it across all 32 layers.  This script runs each placement as a
# fresh process (the adapter reads the policy at class-construction time)
# and collects tok/s, ITL, KV capacity, B70 route share, and dispatch
# service time, then summarizes the tradeoff curve.
#
# No server is launched — this uses the in-process harness directly, so
# the GPU is held only for the duration of each run.
#
# Usage:
#   bash phase10/track_b_offload_sweep.sh
#   PLACEMENTS="split:128 subset:16:8" bash phase10/track_b_offload_sweep.sh
#
# Env overrides:
#   PLACEMENTS   space-separated offload policies (default: the curve below)
#   DECODE_TOKENS=400  CONCURRENCY="1 8 32"
#   OUT_DIR=$PWD/phase10/results/offload
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# A spread across the two knobs: active-layer count rises 8 -> 16 -> 24,
# and split:128 is the all-32-layers baseline policy.
PLACEMENTS="${PLACEMENTS:-subset:8:8 subset:16:8 subset:24:64 split:128}"
DECODE_TOKENS="${DECODE_TOKENS:-400}"
CONCURRENCY="${CONCURRENCY:-1 8 32}"
OUT_DIR="${OUT_DIR:-$PWD/phase10/results/offload}"
mkdir -p "$OUT_DIR"

export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export PATH="$REPO_ROOT/.venv/bin:${PATH:-}"
# oneAPI is needed even in-process: the B70 provider loads SYCL.
# shellcheck disable=SC1091
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true

run_one () {
  local config="$1" placement="$2" out="$3"
  if [[ -f "$out" ]]; then
    echo "[track_b] cached $config $placement -> $out"
    return
  fi
  echo "[track_b] running $config ${placement:-(n/a)}"
  "$REPO_ROOT/.venv/bin/python" phase10/benchmark.py \
    --config "$config" \
    --placement "${placement:-split:128}" \
    --out "$out" \
    --trials 2 \
    --decode-tokens "$DECODE_TOKENS" \
    --batch-tokens 128 \
    --concurrency $CONCURRENCY \
    --context-lengths 2048
}

# Baseline first — same adapter, all-CUDA placement, no B70.
run_one all-cuda "" "$OUT_DIR/all-cuda.json"

for p in $PLACEMENTS; do
  slug="$(echo "$p" | tr ':' '-')"
  run_one hybrid "$p" "$OUT_DIR/hybrid-${slug}.json"
done

echo "[track_b] summarizing -> $OUT_DIR"
"$REPO_ROOT/.venv/bin/python" phase10/track_b_summarize.py --dir "$OUT_DIR"
