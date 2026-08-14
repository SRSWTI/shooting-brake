#!/usr/bin/env bash
# Drive prefill_cliff_probe.py across three arms, one process each.
#
# Arms, in order:
#   1. hybrid subset:16:8      -- the Track A configuration, in-process
#   2. all-cuda                -- control: attention's own N^2 growth, no B70
#   3. hybrid + prefill-stream -- the suspected fix: move B70-owned weights
#                                 to the 5090 once per forward instead of
#                                 dispatching every token to the B70
#
# Each arm is a fresh process because the adapter reads placement at
# class-construction time, and the GPU is held only for that arm's duration.
#
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
export PATH="$REPO_ROOT/.venv/bin:${PATH:-}"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

OUT_DIR="${OUT_DIR:-$REPO_ROOT/benchmarks/results/prefill_probe}"
LENGTHS="${LENGTHS:-8192 32768 65536}"
CONCURRENCY="${CONCURRENCY:-1 4 8}"
TRIALS="${TRIALS:-2}"
DECODE_TOKENS="${DECODE_TOKENS:-64}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
REST="${REST:-20}"
mkdir -p "$OUT_DIR"

run_arm () {
  local label="$1" config="$2" placement="$3"
  shift 3
  local out="$OUT_DIR/${label}.json"
  if [[ -f "$out" ]]; then
    echo "[probe] cached $label"
    return
  fi
  echo "=== [probe] $label ($(date +%H:%M:%S)) ==="
  ./.venv/bin/python benchmarks/prefill_cliff_probe.py \
    --config "$config" --placement "$placement" \
    --lengths $LENGTHS --concurrency $CONCURRENCY \
    --trials "$TRIALS" --decode-tokens "$DECODE_TOKENS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --out "$out" "$@"
  echo "=== [probe] $label rc=$? ==="
  sleep "$REST"
}

run_arm hybrid-subset-16-8 hybrid subset:16:8
run_arm all-cuda all-cuda ""
run_arm hybrid-subset-16-8-stream hybrid subset:16:8 --prefill-stream

echo "=== [probe] ALL ARMS DONE ($(date +%H:%M:%S)) ==="
