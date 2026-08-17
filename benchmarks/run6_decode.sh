#!/usr/bin/env bash
# run6, decode + saturation grids on the SAME config as the F_matrix cells.
#
# Why: F_matrix covers TTFT/ITL/TPOT/out-tok/s across 8 contexts x C={1..6,10},
# but it has no 128-token decode-shaped grid and no throughput profile. The
# PRO matrix reports a `sweep`/throughput peak (798 out tok/s @ ctx_1024), so
# a like-for-like peak row needs OUR throughput profile measured on run6's
# config -- not run4's (pre-register-DMA, pre-KV-levers) numbers.
#
# Config is byte-identical to run6_finish.sh: BANK_REGISTER=1,
# --kv-cache-memory=2.9e9 (209,715 KV tokens), B70_MAX_BATCH=256 (set by
# serve_88b_128k.sh under BANK_REGISTER), dual served-model-name.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
export PATH="$PWD/.venv/bin:$PATH"   # JIT extension builds invoke bare `ninja`

ROOT=benchmarks/results/run6_final
LOG=$ROOT/server_decode.log
PY=.venv/bin/python

pkill -f "vllm serve srswti/axe-superveloce" 2>/dev/null || true
sleep 5

SHOOTING_BRAKE_BANK_REGISTER=1 \
SB_EXTRA_ARGS='--served-model-name shooting-brake-88b srswti/axe-superveloce-88b-nvfp4a16 --kv-cache-memory=2900000000' \
  setsid nohup bash benchmarks/serve_88b_128k.sh > "$LOG" 2>&1 &
disown

for _ in $(seq 90); do
  curl -sf http://127.0.0.1:8016/health >/dev/null && break
  grep -q "EngineCore failed" "$LOG" && { echo BOOT_FAILED; exit 1; }
  sleep 10
done
curl -sf http://127.0.0.1:8016/health >/dev/null || { echo BOOT_FAILED; exit 1; }
echo BOOTED
grep -E "GPU KV cache size|Maximum concurrency" "$LOG" | sed 's/.*INFO [0-9:-]* //'

# Warm past the one-time pin-eviction thrash window before measuring.
for i in 1 2; do
  curl -sf -X POST http://127.0.0.1:8016/v1/completions \
    -H 'Content-Type: application/json' \
    -d "$(jq -n --rawfile p "$ROOT/../prefill_profile/prompt_8k.txt" \
        '{model:"shooting-brake-88b",prompt:$p,max_tokens:4,temperature:0}')" \
    >/dev/null && echo "warm $i ok" || echo "warm $i FAILED"
done
curl -sf http://127.0.0.1:8016/health >/dev/null || {
  echo "WARM_KILLED_SERVER; engine root cause:"
  grep "core.py:1351" "$LOG" | grep -E "Error" | tail -2 | cut -c1-200
  exit 1
}
for _ in $(seq 40); do
  psi=$(awk -F'avg10=' '/^some/{split($2,a," ");print a[1]}' /proc/pressure/memory)
  awk -v v="$psi" 'BEGIN{exit !(v<0.1)}' && break
  sleep 5
done
echo "psi=$psi"

$PY benchmarks/bench_88b.py --root "$ROOT/matrix" \
  --grids A_decode,D_saturation --skip-existing 2>&1 | tail -12
