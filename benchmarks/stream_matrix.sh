#!/usr/bin/env bash
# Sweep B70 prefill dispatch vs streaming across prompt length and concurrency.
#
# One process per mode: the adapter reads its switches at layer construction,
# and the arena is populated at weight-load time, so a mode cannot be changed
# inside a live engine.
#
# setvars.sh calls exit internally, so it is sourced before set -e; it also
# strips the venv from PATH, so the venv is re-prepended after.
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
set -euo pipefail

cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH"

export VLLM_PLUGINS=shooting_brake_vllm
export SHOOTING_BRAKE_B70_BANK="$PWD/phase1/expert_bank.bin"
export SHOOTING_BRAKE_B70_LIB="$PWD/phase7/libsb_b70_provider.so"
export SHOOTING_BRAKE_CPU_LIB="$PWD/phase7/libsb_cpu_expert.so"

# Bare-form defaults: an explicitly empty MODES means "none", not "all".
MODES=${MODES-all-cuda dispatch stream}
LENGTHS=${LENGTHS-256 512 1024 2048 4096}
CONCURRENCY=${CONCURRENCY-1 4 16 64}
PLACEMENT=${PLACEMENT:-subset:16:8}
TRIALS=${TRIALS:-2}
DECODE_TOKENS=${DECODE_TOKENS:-32}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-80}
# Threshold 1 forces streaming at every length, so the crossover is located
# from measurement. Production picks a threshold from the result.
THRESHOLD=${THRESHOLD:-1}
OUT=${OUT:-benchmarks/results/stream/matrix.json}

python benchmarks/stream_matrix.py \
  --modes $MODES \
  --lengths $LENGTHS \
  --concurrency $CONCURRENCY \
  --placement "$PLACEMENT" \
  --threshold "$THRESHOLD" \
  --trials "$TRIALS" \
  --decode-tokens "$DECODE_TOKENS" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --out "$OUT"

echo "results in $(dirname "$OUT")"
