#!/usr/bin/env bash
# Sweep the B70 prefill chunk size against TTFT.
#
# One process per chunk size: the pinned staging buffers are sized from
# SHOOTING_BRAKE_B70_MAX_BATCH when the layer is constructed, so the value
# cannot be changed inside a live engine.
#
# setvars.sh calls exit internally, so it must be sourced before set -e, and it
# strips the venv from PATH, so the venv is re-prepended after.
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
set -euo pipefail

cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH"

OUT=${OUT:-benchmarks/results/chunk}
CHUNKS=${CHUNKS-128 256 512 1024 2048}
PLACEMENT=${PLACEMENT:-subset:16:8}
LENGTHS=${LENGTHS:-1536 4096}
TRIALS=${TRIALS:-3}

mkdir -p "$OUT"

for c in $CHUNKS; do
  echo "=== chunk $c (placement $PLACEMENT) ==="
  SHOOTING_BRAKE_B70_MAX_BATCH="$c" \
  SHOOTING_BRAKE_PLACEMENT="$PLACEMENT" \
    python src/phase7/prefill_chunk_bench.py \
      --trials "$TRIALS" \
      --context-lengths $LENGTHS \
      --out "$OUT/chunk-$c.json"
done

echo "results in $OUT"
