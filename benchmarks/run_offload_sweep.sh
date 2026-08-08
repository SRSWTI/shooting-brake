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
#   bash benchmarks/run_offload_sweep.sh
#   PLACEMENTS="split:128 subset:16:8" bash benchmarks/run_offload_sweep.sh
#
# Env overrides:
#   PLACEMENTS   space-separated offload policies (default: the curve below)
#   DECODE_TOKENS=400  CONCURRENCY="1 8 32"
#   OUT_DIR=$PWD/benchmarks/results/offload
#
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# oneAPI's setvars.sh terminates the sourcing shell under `set -e`, so
# source it BEFORE enabling strict mode. The B70 provider loads SYCL.
# shellcheck disable=SC1091
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
export PATH="$REPO_ROOT/.venv/bin:${PATH:-}"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

# Strict mode only for our own logic below.
set -euo pipefail

# A spread across the two knobs: active-layer count rises 8 -> 16 -> 24,
# and split:128 is the all-32-layers baseline policy.
# ${VAR-default} rather than ${VAR:-default}: an explicitly empty PLACEMENTS
# means "run none of these", which is how a single tier gets measured on its
# own. The colon form would silently substitute the default instead.
PLACEMENTS="${PLACEMENTS-subset:8:8 subset:16:8 subset:24:64 split:128}"
# Three-tier placements ("allout:<layers>:<cuda>:<cpu>"). Empty by default:
# the cold tier is a deliberate opt-in, so a plain sweep never pays its cost
# without being asked.
ALLOUT_PLACEMENTS="${ALLOUT_PLACEMENTS-}"
DECODE_TOKENS="${DECODE_TOKENS:-400}"
CONCURRENCY="${CONCURRENCY:-1 8 32}"
# Smoke-test knobs: TRIALS=1 with a couple of context lengths gives the
# shape of the tradeoff in minutes rather than the full curve in hours.
TRIALS="${TRIALS:-2}"
CONTEXT_LENGTHS="${CONTEXT_LENGTHS:-2048}"
# Engine admission cap. The offload tradeoff is visible at short context,
# so 8192 is a fast default; raise to 32768/131072 to also see the KV
# capacity difference at long context (slower cells).
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
# Admission width. Must be at least the largest concurrency being swept, or
# the extra requests queue instead of running and the cell measures nothing
# new. All-CUDA additionally caps out near 83: GDN allocates one Mamba cache
# block per decode sequence, and the baseline has the least VRAM spare for
# them, so keeping this under that ceiling keeps every config comparable.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
# Seconds to idle the GPU between runs, so the card is not pinned at the
# power cap for the whole sweep. Pair with gpu_power.sh cap <watts>.
REST_BETWEEN="${REST_BETWEEN:-15}"
OUT_DIR="${OUT_DIR:-$PWD/benchmarks/results/offload}"
mkdir -p "$OUT_DIR"

run_one () {
  local config="$1" placement="$2" out="$3"
  if [[ -f "$out" ]]; then
    echo "[track_b] cached $config $placement -> $out"
    return
  fi
  echo "[track_b] running $config ${placement:-(n/a)}"
  "$REPO_ROOT/.venv/bin/python" benchmarks/offload_benchmark.py \
    --config "$config" \
    --placement "${placement:-split:128}" \
    --out "$out" \
    --max-model-len "$MAX_MODEL_LEN" \
    --trials "$TRIALS" \
    --decode-tokens "$DECODE_TOKENS" \
    --batch-tokens 128 \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --concurrency $CONCURRENCY \
    --context-lengths $CONTEXT_LENGTHS
  # Let the card cool before the next process grabs the GPU.
  sleep "$REST_BETWEEN"
}

# Baseline first — same adapter, all-CUDA placement, no B70.
run_one all-cuda "" "$OUT_DIR/all-cuda.json"

for p in $PLACEMENTS; do
  slug="$(echo "$p" | tr ':' '-')"
  run_one hybrid "$p" "$OUT_DIR/hybrid-${slug}.json"
done

# Three-tier runs, if any were asked for. Kept separate from PLACEMENTS
# because the cold tier is opt-in: it needs SHOOTING_BRAKE_ALL_OUT=1, which
# run_one sets only for the all-out config, so an "allout:" policy listed
# under PLACEMENTS would be refused rather than quietly downgraded.
for p in $ALLOUT_PLACEMENTS; do
  slug="$(echo "$p" | tr ':' '-')"
  run_one all-out "$p" "$OUT_DIR/allout-${slug}.json"
done

echo "[track_b] summarizing -> $OUT_DIR"
"$REPO_ROOT/.venv/bin/python" benchmarks/offload_summarize.py --dir "$OUT_DIR"
