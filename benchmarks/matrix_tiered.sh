#!/usr/bin/env bash
# Drive the benchmark matrix one context tier at a time, restarting the server
# between tiers.
#
# WHY THIS EXISTS
# ---------------
# A single long matrix_runner invocation cannot finish on this box. Device USM on
# the B70s costs host RAM 1:1 -- ~24.34 GiB per card, ~48.7 GiB for the pair,
# measured by experiments/b70_mem_topology_probe and unaffected by every NEO
# debug key tried -- against 59.4 GiB total. That leaves ~5 GiB for everything
# else, and on 2026-08-22 an hour of GuideLLM consumed it monotonically:
#
#   02:54  mem 2.9G  swap 1.2G
#   03:14  mem 2.0G  swap 0.0G   <- swap exhausted
#   03:34  mem 0.5G  swap 0.0G
#   03:45  server and runner both gone, no error in either log, 8/80 cells
#
# Restarting between tiers returns the accumulation to zero. --skip-existing
# makes it resumable, so a death costs one tier rather than the whole run.
#
# Tiers ascend so the cheap, most-used contexts bank first: an interrupted run
# still leaves a useful surface.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

ROOT="${MATRIX_ROOT:-$PWD/bench-matrix/jota_r15_c6}"
LOG="${MATRIX_LOG:-/tmp/sb_matrix_tiered.log}"
CONTEXTS="${MATRIX_CONTEXTS:-1024 4096 8192 16384 32768 65536 98304 127000}"
MIN_MEM_GIB="${MATRIX_MIN_MEM_GIB:-3}"

say() { printf '%s %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$LOG"; }
mem_gib() { awk '/^MemAvailable:/ {printf "%.1f", $2/1048576}' /proc/meminfo; }

stop_server() {
  pkill -9 -f 'vllm serve' 2>/dev/null
  pkill -9 -f 'VLLM::EngineCore' 2>/dev/null
  sleep 4
  for _ in $(seq 1 45); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0)
    [ "${u:-0}" -lt 2000 ] && break
    sleep 2
  done
}

start_server() {
  # Same knobs that produced 430 us/token and cleared Bench 26's quality gate.
  # MNS=6 is required or --concurrent-rates 5,6 is a fiction: the server would
  # run 4 and queue the rest while GuideLLM reports 6.
  SHOOTING_BRAKE_B70_GROUPED=1 \
  SHOOTING_BRAKE_B70_MAX_BATCH=2048 \
  SHOOTING_BRAKE_B70_OUT_FP16=1 \
  SB_MNBT=2048 SB_MNS=6 SYCL_UR_USE_LEVEL_ZERO_V2=0 \
    setsid nohup benchmarks/serve_jota_r15_dual.sh > /tmp/sb_matrix_serve.log 2>&1 < /dev/null &
  disown
  for _ in $(seq 1 100); do
    sleep 5
    curl -fsS --max-time 5 http://127.0.0.1:8017/health >/dev/null 2>&1 && return 0
  done
  return 1
}

cells() { find "$ROOT" -name report.json 2>/dev/null | wc -l | tr -d ' '; }

mkdir -p "$ROOT"
say "=== tiered matrix start | root=$ROOT | banked=$(cells) ==="

for ctx in $CONTEXTS; do
  stop_server
  say "tier ctx=$ctx: mem after server stop $(mem_gib)G"
  if ! start_server; then
    say "tier ctx=$ctx: SERVER FAILED TO START -- stopping"
    exit 1
  fi
  say "tier ctx=$ctx: server up, mem $(mem_gib)G, banked $(cells)"

  HF_HUB_OFFLINE=1 .venv/bin/python benchmarks/matrix_runner.py \
    --target http://127.0.0.1:8017 \
    --metrics-url http://127.0.0.1:8017/metrics \
    --model shooting-brake-jota-r15 \
    --tokenizer-model srswti/axe-superveloce-jota-118b-r15-nvfp4 \
    --contexts "$ctx" \
    --concurrent-rates 1,2,3,4,5,6 \
    --profiles synchronous,concurrent,sweep \
    --sweep-steps 3 \
    --output-tokens 512 \
    --max-requests 20 \
    --max-seconds 180 \
    --warmup 0 --cooldown 0 \
    --sample-interval 5 \
    --outputs json,csv \
    --max-errors 60 \
    --skip-existing \
    --output-root "$ROOT" >> "$LOG" 2>&1
  rc=$?
  say "tier ctx=$ctx: runner exit $rc, banked $(cells), mem $(mem_gib)G"

  m=$(mem_gib)
  if awk -v m="$m" -v t="$MIN_MEM_GIB" 'BEGIN {exit !(m < t)}'; then
    say "tier ctx=$ctx: mem ${m}G below ${MIN_MEM_GIB}G -- restart will reclaim it"
  fi
done

stop_server
say "=== tiered matrix done | banked=$(cells) cells ==="
