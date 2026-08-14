#!/usr/bin/env bash
# Shooting Brake — chain Track B behind a running Track A matrix.
#
# Waits for the Track A wrapper process to exit, audits the matrix output
# tree for completeness, and only then tears the server down and starts
# the full Track B offload sweep.  Both tracks need the GPU exclusively,
# so the teardown gate is load-bearing: it waits for the vLLM process to
# die AND for the B70 to release its memory before Track B allocates.
#
# The audit is the safety interlock.  A Ctrl-C'd matrix also ends the
# watched PID, and chaining a multi-hour sweep behind a half-finished
# baseline would waste the night, so an incomplete matrix aborts the
# chain and spawns an agent to diagnose instead.
#
# Usage:
#   setsid nohup bash benchmarks/watch_and_chain.sh <PID> >/dev/null 2>&1 &
#
# Env overrides (defaults shown):
#   WATCH_PID              (required as $1 or this)
#   MATRIX_ROOT=$PWD/bench-matrix/hybrid_131k_c6
#   MODEL=unsloth/Qwen3.6-35B-A3B-NVFP4
#   EXPECT_CONTEXTS=1024,4096,8192,16384,32768,65536,98304,127000
#   TRACKB_OUT=$PWD/benchmarks/results/offload_full
#   TRACKB_CONTEXT_LENGTHS="2048 8192 32768 131072"
#   TRACKB_CONCURRENCY="1 4 8 16 32 64"
#   TRACKB_MAX_MODEL_LEN=131072
#   TRACKB_TRIALS=3
#   AGENT=jesco            (set AGENT=none to skip agent spawns)
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WATCH_PID="${1:-${WATCH_PID:-}}"
if [[ -z "$WATCH_PID" ]]; then
  echo "usage: bash benchmarks/watch_and_chain.sh <pid-of-run_matrix.sh>" >&2
  exit 2
fi

MATRIX_ROOT="${MATRIX_ROOT:-$REPO_ROOT/bench-matrix/hybrid_131k_c6}"
MODEL="${MODEL:-unsloth/Qwen3.6-35B-A3B-NVFP4}"
EXPECT_CONTEXTS="${EXPECT_CONTEXTS:-1024,4096,8192,16384,32768,65536,98304,127000}"
TRACKB_OUT="${TRACKB_OUT:-$REPO_ROOT/benchmarks/results/offload_full}"
TRACKB_CONTEXT_LENGTHS="${TRACKB_CONTEXT_LENGTHS:-2048 8192 32768 131072}"
TRACKB_CONCURRENCY="${TRACKB_CONCURRENCY:-1 4 8 16 32 64}"
TRACKB_MAX_MODEL_LEN="${TRACKB_MAX_MODEL_LEN:-131072}"
TRACKB_TRIALS="${TRACKB_TRIALS:-3}"
# Idle seconds after the B70 frees memory, before Track B allocates. A knob
# so the chain can be rehearsed end-to-end without waiting out the real rest.
SETTLE_SECONDS="${SETTLE_SECONDS:-60}"
AGENT="${AGENT:-jesco}"

RUN_ID="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="$REPO_ROOT/bench-matrix/chain/$RUN_ID"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/chain.log"
ln -sfn "$LOG_DIR" "$REPO_ROOT/bench-matrix/chain/latest"

log () { echo "[chain $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# Spawn a headless agent turn. Non-fatal: the chain's value is the data,
# and an agent failure must never take the sweep down with it.
spawn_agent () {
  local label="$1" prompt="$2"
  if [[ "$AGENT" == "none" ]]; then
    log "agent spawn skipped (AGENT=none): $label"
    return 0
  fi
  log "spawning agent: $label"
  "$AGENT" -p --auto-approve --cwd "$REPO_ROOT" "$prompt" \
    >"$LOG_DIR/agent_${label}.log" 2>&1 \
    || log "agent $label exited nonzero (see agent_${label}.log)"
}

# Resident B70 memory in MiB, or empty when it cannot be read.
#
# Parses `xpu-smi stats -j`, NOT the human table: the table truncates
# "current: 14380" to "cur..." at the terminal column width, so a field-split
# parse silently yields empty -- which the caller would read as "card is
# free" and start Track B on top of a still-loaded server.
b70_mib () {
  xpu-smi stats -d 0 -j 2>/dev/null | "$REPO_ROOT/.venv/bin/python" -c '
import json, sys
try:
    d = json.load(sys.stdin)
    tiles = d["memory"]["used_mib"]
    print(int(round(max(t["current"] for t in tiles.values()))))
except Exception:
    pass
'
}

log "watching pid=$WATCH_PID  matrix_root=$MATRIX_ROOT"
log "log dir: $LOG_DIR"

# 1. Wait for Track A. Polling kill -0 rather than `wait`: the matrix is
#    not this shell's child, so `wait` cannot see it.
while kill -0 "$WATCH_PID" 2>/dev/null; do
  sleep 30
done
log "track A wrapper pid=$WATCH_PID exited"

# Let the last cell's json/csv/html writers flush before auditing.
sleep 20

# 2. Audit. This is the gate, and it is deliberately strict.
AUDIT_LOG="$LOG_DIR/audit.log"
"$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/benchmarks/chain_audit.py" \
  --output-root "$MATRIX_ROOT" \
  --model "$MODEL" \
  --contexts "$EXPECT_CONTEXTS" \
  2>&1 | tee "$AUDIT_LOG" | tee -a "$LOG"
audit_rc="${PIPESTATUS[0]}"

if [[ "$audit_rc" -ne 0 ]]; then
  log "AUDIT FAILED — not chaining Track B (server left running)"
  spawn_agent audit_failure "The Track A SLO matrix run just ended INCOMPLETE and the \
automatic Track B chain was aborted by benchmarks/chain_audit.py. Read \
$AUDIT_LOG for which cells are missing or failed, then investigate: for each bad \
cell read the guidellm_stderr.log and run_manifest.json under \
$MATRIX_ROOT/${MODEL//\//__}/ctx_<n>/<profile>/, check whether the vLLM \
server (pgrep -af 'vllm serve' to find it) is still alive and healthy at \
http://127.0.0.1:8000/health, and check dmesg/nvidia-smi/xpu-smi for OOM or device \
faults. Do NOT start Track B and do NOT kill the server. Write your findings and the \
exact resume command to $LOG_DIR/DIAGNOSIS.md."
  exit 1
fi

log "audit clean — tearing down the hybrid server"

# 3. Teardown. SIGINT first (vLLM's clean shutdown path), escalate only
#    if it refuses, then wait for the B70 to actually release memory:
#    Track B allocates on the same card and starting early means an OOM
#    hours into the night.
pkill -INT -f 'vllm serve' 2>/dev/null || true
for _ in $(seq 1 60); do
  pgrep -f 'vllm serve' >/dev/null 2>&1 || break
  sleep 5
done
if pgrep -f 'vllm serve' >/dev/null 2>&1; then
  log "server ignored SIGINT after 300s — sending SIGKILL"
  pkill -KILL -f 'vllm serve' 2>/dev/null || true
  sleep 20
fi

# Unreadable telemetry is NOT treated as "free". Track B allocates the whole
# card, so the failure mode of guessing wrong is an OOM hours into an
# unattended run; fall back to a fixed conservative wait instead.
released=0
for _ in $(seq 1 60); do
  mib="$(b70_mib)"
  if [[ -z "$mib" ]]; then
    log "WARNING: cannot read B70 memory via xpu-smi — falling back to a 180s wait"
    sleep 180
    released=1
    break
  fi
  if [[ "$mib" -lt 2000 ]]; then
    log "B70 released memory (${mib} MiB resident)"
    released=1
    break
  fi
  sleep 10
done
if [[ "$released" -eq 0 ]]; then
  log "WARNING: B70 still at $(b70_mib) MiB after 600s — starting Track B anyway"
fi
log "B70 now at $(b70_mib) MiB; settling ${SETTLE_SECONDS}s before Track B"
sleep "$SETTLE_SECONDS"

# 4. Track B. Runs in the foreground of this detached script, so the
#    whole chain lives or dies as one process tree.
log "starting Track B -> $TRACKB_OUT"
log "  contexts='$TRACKB_CONTEXT_LENGTHS' concurrency='$TRACKB_CONCURRENCY' \
max_model_len=$TRACKB_MAX_MODEL_LEN trials=$TRACKB_TRIALS"

CONTEXT_LENGTHS="$TRACKB_CONTEXT_LENGTHS" \
CONCURRENCY="$TRACKB_CONCURRENCY" \
MAX_MODEL_LEN="$TRACKB_MAX_MODEL_LEN" \
TRIALS="$TRACKB_TRIALS" \
OUT_DIR="$TRACKB_OUT" \
  bash "$REPO_ROOT/benchmarks/run_offload_sweep.sh" \
  >"$LOG_DIR/track_b.log" 2>&1
trackb_rc=$?
log "Track B exited rc=$trackb_rc"

# 5. Analysis. The sweep's own summarizer already ran inside the script;
#    the agent turn is for the cross-track reading that no script can do.
spawn_agent report "The automated benchmark chain finished. Track A (guidellm SLO matrix, \
placement subset:16:8) completed all cells under $MATRIX_ROOT, and Track B (offload \
sweep) then ran to $TRACKB_OUT with exit code $trackb_rc; its stdout is \
$LOG_DIR/track_b.log and the chain log is $LOG. Write $LOG_DIR/REPORT.md covering: (1) \
Track B per-placement decode tok/s, ITL p50/p99, KV max_tokens, B70 route share and \
dispatch service time, as a table across all-cuda / subset:8:8 / subset:16:8 / \
subset:24:64 / split:128; (2) the capacity-frontier result per context length, \
specifically which contexts all-cuda cannot admit at all; (3) Track A's long-context \
cells (65536/98304/127000) read from report.json in each cell dir, and whether decode \
throughput holds flat across prompt length as benchmarks/README.md claims; (4) any cell \
or placement that errored, with the evidence. Ground every number in a file path. Do not \
re-run benchmarks and do not modify the result artifacts. Leave the GPU power cap as it \
is."

log "chain complete — report in $LOG_DIR"
exit "$trackb_rc"
