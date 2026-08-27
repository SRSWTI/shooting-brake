#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
WORK_DIR="${SB_DRAFTER_WORK_DIR:-$ROOT/experiments/drafter_train/work}"
CHECKPOINT="${SB_DRAFTER_CHECKPOINT:-$WORK_DIR/export}"
CORPUS="${SB_DRAFTER_GATE_CORPUS:-$HOME/sb_corpus_big.txt}"
EVIDENCE="$WORK_DIR/gate"
BASE_URL="http://127.0.0.1:8017"
# Baseline arm: the banked pool minus the 512 MiB OOM headroom.
KV_BYTES=10936647680
# Candidate arm: the drafter is a THIRD resident on the 5090 (1.96 GiB of
# weights plus its own graphs/activations). MEASURED 2026-08-26: booting
# the candidate at the baseline pool OOMs during engine init with 469 MiB
# free (model 16.77 GiB + KV 10.19 GiB + drafter + workspace > 31.36 GiB).
# 7.45 GiB still exceeds the 6.64 GiB a single 131072-token request needs
# (serve_jota_r15_dual.sh), so every ladder rung still fits; KV size sets
# capacity, not per-token latency, so the ITL comparison stays fair.
KV_BYTES_SPEC="${SB_KV_BYTES_SPEC:-8000000000}"

if [[ ! -s "$CHECKPOINT/config.json" ]] || \
   [[ ! -s "$CHECKPOINT/model.safetensors" && \
      ! -s "$CHECKPOINT/model.safetensors.index.json" ]]; then
  echo "ERROR: no exported Laguna checkpoint at $CHECKPOINT; run train.sh first." >&2
  exit 2
fi
if [[ ! -s "$CORPUS" ]]; then
  echo "ERROR: gate corpus not found at $CORPUS" >&2
  exit 2
fi
mkdir -p "$EVIDENCE"
rm -f \
  "$EVIDENCE/baseline.json" \
  "$EVIDENCE/candidate.json" \
  "$EVIDENCE/ladder.json" \
  "$EVIDENCE/summary.json"
unset SB_HIDDEN_CAPTURE_DIR
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Both arms run the same max_model_len so the comparison is apples-to-apples
# (see the ceiling arithmetic on KV_BYTES_SPEC above).
export SB_MML="${SB_MML:-98304}"
export SB_KV_BYTES="$KV_BYTES"
unset SB_SPEC
echo "[1/6] booting the no-speculation baseline"
./serve_production.sh

echo "[2/6] capturing the 120-prompt baseline top-logprob quality sweep"
"$PYTHON" experiments/drafter_train/gate.py capture \
  --url "$BASE_URL" --corpus "$CORPUS" --output "$EVIDENCE/baseline.json"

printf -v SB_SPEC '{"model":"%s","num_speculative_tokens":15,"method":"dflash"}' "$CHECKPOINT"
export SB_SPEC
export SB_KV_BYTES="$KV_BYTES_SPEC"
echo "[3/6] booting native Laguna DFlash (KV=$KV_BYTES_SPEC, trimmed for the drafter's residency)"
./serve_production.sh

echo "[4/6] capturing the 120-prompt candidate top-logprob quality sweep"
"$PYTHON" experiments/drafter_train/gate.py capture \
  --url "$BASE_URL" --corpus "$CORPUS" --output "$EVIDENCE/candidate.json"

echo "[5/6] running the 1K -> 98K acceptance/effective-decode ladder"
"$PYTHON" benchmarks/decode_ladder_probe.py \
  --url "$BASE_URL" \
  --model shooting-brake-jota-r15 \
  --corpus "$CORPUS" \
  --contexts 1024,8192,16384,32768,65536,98304 \
  --out-tokens 512 \
  --label jota-r15-dflash \
  --json-out "$EVIDENCE/ladder.json"

echo "[6/6] validating quality, acceptance, and effective decode gates"
"$PYTHON" experiments/drafter_train/gate.py summarize \
  --baseline "$EVIDENCE/baseline.json" \
  --candidate "$EVIDENCE/candidate.json" \
  --ladder "$EVIDENCE/ladder.json" \
  --output "$EVIDENCE/summary.json"
