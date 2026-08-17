#!/usr/bin/env bash
# run6 finish: boot the final config, then complete the F_matrix tail.
#
# Why this exists: the first F_matrix pass landed 6/8 cells (1024..65536) and
# then the engine died mid-ctx_98304 with no CUDA OOM and no Python traceback
# (signature of a host-side SIGKILL; suspect = 27.4 GiB pinned bank + ~30 GiB
# server RSS on a 59 GiB box). Mitigation applied before this run: evicted the
# 29.9 GiB b12x bank page cache (its kernel lost the bake-off; nothing maps it).
#
# Cells re-run rather than resumed:
#   ctx_1024  -- two prior passes measured 2.24 s: prompts <=1024 tokens take
#                the doorbell dispatch whenever M <= SHOOTING_BRAKE_B70_MAX_BATCH
#                (routed_experts.py:2130 gates _prefill_forward_offloaded, home
#                of the Marlin branch). The first "fix" set
#                SHOOTING_BRAKE_B70_STREAM_T -- a dead knob on this config.
#                serve_88b_128k.sh now defaults B70_MAX_BATCH=256 under
#                BANK_REGISTER=1, which is the real crossover knob.
#   ctx_98304, ctx_127000 -- failed on a dead server (the ctx_1024 C=10 rung
#                OOM'd the engine at KV=3.07e9: concurrency transients need
#                more slack than C<=6 -- 186 MiB free vs a 128 MiB alloc).
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
export PATH="$PWD/.venv/bin:$PATH"   # JIT extension builds invoke bare `ninja`

ROOT=benchmarks/results/run6_final
LOG=$ROOT/server_v6.log
PY=.venv/bin/python

pkill -f "vllm serve srswti/axe-superveloce" 2>/dev/null || true
sleep 5

rm -rf "$ROOT/matrix/F_matrix/ctx_1024" "$ROOT/matrix/F_matrix/ctx_98304" \
       "$ROOT/matrix/F_matrix/ctx_127000"

# Explicit KV sizing instead of utilization: the boot profiler's estimate
# varies ~50 MiB per boot and runtime wants slack it never charges. Ladder of
# measured outcomes: util 0.92 -> 235,929 tok served 6 cells but a sibling
# boot's 239,674 OOM'd warm #1; explicit 3.34e9 (243,419 tok) OOM'd warm #1;
# explicit 3.07e9 (220,949 tok) survived warms + C<=6 but OOM'd at C=10.
# 2.90e9 ~= 211,350 tok (+23% over run5's 172K, ~1.61x seats) leaves ~600 MiB.
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

# Record what the boot actually chose, so the matrix numbers carry their config.
grep -E "GPU KV cache size|Maximum concurrency" "$LOG" | sed 's/.*INFO [0-9:-]* //'
SRV_PID=$(pgrep -f "vllm serve srswti/axe-superveloce" | head -1)
tr '\0' '\n' < "/proc/$SRV_PID/environ" \
  | grep -E "SHOOTING_BRAKE_B70_MAX_BATCH|SHOOTING_BRAKE_BANK_REGISTER" || true

# Warm past the one-time pin-eviction thrash window before measuring.
for i in 1 2; do
  curl -sf -X POST http://127.0.0.1:8016/v1/completions \
    -H 'Content-Type: application/json' \
    -d "$(jq -n --rawfile p "$ROOT/../prefill_profile/prompt_8k.txt" \
        '{model:"shooting-brake-88b",prompt:$p,max_tokens:4,temperature:0}')" \
    >/dev/null && echo "warm $i ok" || echo "warm $i FAILED"
done
# A failed warm means the engine died on first prefill (the OOM class) --
# abort loudly with the root cause instead of limping into the matrix.
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

# 8K spot check on the fixed threshold: guards against the threshold edit
# perturbing the >=4096 rows, which already stream and must not move.
$PY benchmarks/cuda_decode_profile.py measure \
  --target http://127.0.0.1:8016 --model shooting-brake-88b \
  --prompt "$(cat "$ROOT/../prefill_profile/prompt_8k.txt")" \
  --warmup-tokens 4 --decode-steps 64 --label run6-threshold-8k \
  --out "$ROOT/ttft_8k_threshold.json" >/dev/null 2>&1 || true
$PY -c "
import json
d = json.load(open('$ROOT/ttft_8k_threshold.json'))['measured']
print('8K ttft:', round(d['ttft_us']/1e6, 3), 's | itl:',
      round(d['decode_interval_median_us']/1e3, 2), 'ms')" \
  || echo "spot check unavailable (non-fatal)"

$PY benchmarks/bench_88b.py --root "$ROOT/matrix" \
  --grids F_matrix --skip-existing 2>&1 | tail -8
echo "=== F_matrix cells:"
for d in "$ROOT"/matrix/F_matrix/*/; do
  n=$(basename "$d")
  [ -f "$d/report.json" ] && echo "$n COMPLETE" || echo "$n INCOMPLETE"
done
