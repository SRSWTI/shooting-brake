#!/usr/bin/env bash
# Shooting Brake — post-hoc vs pre-emptive VRAM surgery, same placement.
#
# Why this exists.  Post-hoc surgery allocates all 256 experts per layer,
# loads them, then slices ownership away.  That works only while the whole
# expert bank transiently fits in VRAM: 13.5 GiB for the 35B, 59.5 GiB for
# the 122B against a 32 GiB card.  Pre-emptive allocation never creates the
# offloaded experts, so peak equals steady state.
#
# What this proves.  The 35B is the only model where BOTH strategies can
# run, which makes it the correctness oracle: same placement, same weights,
# same kernel, so the two must agree token-for-token — this is a stricter
# bar than compare.py's all-CUDA-vs-hybrid, which tolerates forks between
# two different NVFP4 kernels.
#
# The only figure expected to move is cuda_memory.load_peak_allocated_gib,
# sampled during the post-load hook.  Throughput, KV tokens and the
# end-of-run peak are all expected UNCHANGED: post-hoc surgery already
# completes before vLLM's profiling pass, so KV sizing sees the same freed
# memory either way, and the end-of-run high-water mark describes the KV
# cache in both legs.
#
#   agreement + lower load peak -> pre-emptive is correct and does its job
#   token or logprob divergence -> the compact layout addresses wrong experts
#   equal load peak             -> allocation was not actually subsetted
# Usage:
#   bash benchmarks/run_preemptive_parity.sh
#
# Env overrides:
#   PLACEMENT=subset:16:8   DECODE_TOKENS=512   TRIALS=2
#   OUT_DIR=$PWD/benchmarks/results/preemptive
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# oneAPI's setvars.sh terminates the sourcing shell under `set -e`, so
# source it BEFORE enabling strict mode. The B70 provider loads SYCL.
# shellcheck disable=SC1091
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
export PATH="$REPO_ROOT/.venv/bin:${PATH:-}"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

set -euo pipefail

PLACEMENT="${PLACEMENT-subset:16:8}"
DECODE_TOKENS="${DECODE_TOKENS-512}"
TRIALS="${TRIALS-2}"
MAX_NUM_SEQS="${MAX_NUM_SEQS-64}"
CONTEXT="${CONTEXT-1024 2048 4096 8192}"
# Must exceed the longest prompt plus the decode budget, else the top of
# the sweep is clamped and shows up as a capacity regression.
MAX_MODEL_LEN="${MAX_MODEL_LEN-16384}"
OUT_DIR="${OUT_DIR-$REPO_ROOT/benchmarks/results/preemptive}"
mkdir -p "$OUT_DIR"

# A stale VLLM::EngineCore from an interrupted run keeps its whole KV cache
# reserved, and vLLM then fails ~2 minutes in with an engine-init error that
# says nothing about the real cause. Fail in one second instead. Note that
# killing a run started with setsid needs the process *group*
# (`kill -TERM -$PGID`); `pkill -f offload_benchmark` leaves the EngineCore
# child holding the memory.
preflight() {
  python - <<'PY'
import sys, torch
free, total = torch.cuda.mem_get_info()
free_gib, total_gib = free / 2**30, total / 2**30
if free_gib < 28.0:
    sys.exit(
        f"only {free_gib:.2f}/{total_gib:.2f} GiB free on cuda:0 — a stale "
        "engine is probably still holding VRAM. Try: pkill -9 -f VLLM::EngineCore"
    )
print(f"preflight ok: {free_gib:.2f}/{total_gib:.2f} GiB free")
PY
}

run_one() {
  local label="$1"; shift
  local out="$OUT_DIR/$label.json"
  echo "=== $label (placement=$PLACEMENT) ==="
  preflight
  # Each run is a fresh process: the adapter reads its configuration at
  # class-construction time, so a second engine in the same interpreter
  # would silently inherit the first run's placement.
  #
  # Output goes to a per-leg log rather than through `tail`: a truncated
  # tail hides the traceback whenever the engine fails to start, which is
  # exactly when the output is needed.
  local log="$OUT_DIR/$label.log"
  # CONTEXT is a space-separated list, so it must stay unquoted here;
  # quoted, argparse receives one argv entry like "1024 2048" and its
  # type=int rejects it. MAX_MODEL_LEN has to cover the longest prompt plus
  # the decode budget, or the top of the sweep is clamped and reads as a
  # capacity regression that never happened.
  # shellcheck disable=SC2086
  if ! python benchmarks/offload_benchmark.py \
    --config hybrid \
    --placement "$PLACEMENT" \
    --out "$out" \
    --trials "$TRIALS" \
    --decode-tokens "$DECODE_TOKENS" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --context-lengths $CONTEXT \
    --max-model-len "$MAX_MODEL_LEN" \
    "$@" > "$log" 2>&1
  then
    echo "FAILED — root cause from $log:"
    grep -E "Error|error:|raise |ValueError|RuntimeError" "$log" | tail -12
    exit 1
  fi
  tail -6 "$log"
  echo
}

run_one posthoc
run_one preemptive --preemptive

python benchmarks/compare_preemptive.py \
  --posthoc "$OUT_DIR/posthoc.json" \
  --preemptive "$OUT_DIR/preemptive.json"
