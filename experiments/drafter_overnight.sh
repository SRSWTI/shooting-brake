#!/usr/bin/env bash
# ==============================================================================
# Shooting Brake -- overnight drafter pipeline (corpus -> capture -> train ->
# acceptance -> production restore).
#
# Design rules, in order of precedence:
#   1. FAIL TOWARD A HEALTHY PRODUCTION SERVER. Every exit path (success,
#      stage failure, signal) ends by booting the production unit.
#   2. Stages are IDEMPOTENT and RESUMABLE: each records a marker in
#      $STATE_DIR; a rerun skips completed stages, so a crash loses at most
#      one stage of progress.
#   3. Long-lived children run as systemd USER UNITS (linger is enabled on
#      this box), so nothing dies with a terminal session -- the failure
#      mode that killed the stack twice on 2026-08-25.
#   4. Every stage prints measured numbers; nothing is claimed unmeasured.
#
# Stages:
#   1 corpus      wait for >= MIN_RECORDS corpus records (gate on the FILE,
#                 never on the generator process -- see incident note below)
#   2 capture     reboot serving with the aux-hidden capture hook armed and
#                 replay a disk-bounded subset through drafter_capture.py
#   3 warmstart   convert poolside/Laguna-S-2.1-DFlash for SpecForge (CPU)
#   4 train       experiments/drafter_train/train.sh on the idle 5090
#   5 acceptance  experiments/drafter_train/acceptance.sh (boots its own
#                 baseline + candidate arms, gates acceptance/TPOT/quality)
#   6 restore     production serving unit (also runs on ANY failure)
#
# Incident notes (2026-08-25):
#   - Gate on record count, not `pgrep datagen`: a dead generator once
#     triggered an early capture boot that raced a production boot.
#   - Host RAM is the scarce resource (47.4 GiB driver shadow on a 59.4 GiB
#     box): stages 3-5 only run with serving STOPPED.
#
# Usage:
#   systemd-run --user --collect --unit sb-pipeline \
#     --working-directory="$PWD" \
#     bash -c 'exec experiments/drafter_overnight.sh >> /tmp/sb_pipeline.log 2>&1'
# ==============================================================================
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
export PATH="$PWD/.venv/bin:$PATH"

# ---- configuration -----------------------------------------------------------
CORPUS=benchmarks/results/drafter_corpus/pilot.jsonl
SUBSET=benchmarks/results/drafter_corpus/capture_subset.jsonl
CAPDIR="${SB_CAPTURE_DIR:-$HOME/sb_hidden_capture}"
STATE_DIR="${SB_PIPELINE_STATE:-$HOME/sb_pipeline_state}"
# MEASURED 2026-08-25: one training checkpoint is 15 GB and the export is
# 2.3 GB, on the same volume as the captures. 45 GB of capture (~590
# records) leaves ~18 GB headroom against the 82 GB free; the plan calls
# for a pilot-then-scale anyway.
BUDGET_GB="${SB_CAPTURE_BUDGET_GB:-45}"
# Sized to the CAPTURE BUDGET, not to the full corpus: the 45 GB budget
# admits ~590 records, so waiting for 2900 would burn the night generating
# data this pilot cannot capture. 900 gives the subset selector real choice
# with margin. Generation keeps running until the pipeline stops it, so a
# later, larger capture cycle can use whatever accumulated.
MIN_RECORDS="${SB_CAPTURE_MIN_RECORDS:-900}"
CORPUS_DEADLINE_H="${SB_CORPUS_DEADLINE_H:-9}"
KV_BYTES=10936647680  # banked pool minus 512 MiB OOM headroom (pending ratification)
SERVE_ENV="--kv-cache-memory=${KV_BYTES} --enable-prefix-caching"

mkdir -p "$STATE_DIR" "$CAPDIR"
log() { echo "[pipeline $(date +%F' '%T)] $*"; }
mark() { touch "$STATE_DIR/$1.done"; log "stage $1 DONE"; }
done_already() { [ -f "$STATE_DIR/$1.done" ]; }

# /tmp IS A 30 GB tmpfs ON THIS BOX -- files written there consume HOST RAM,
# which is the scarce resource (47.4 GiB driver shadow on 59.4 GiB). Bulk
# scratch in /tmp caused every "OOM" on 2026-08-25, including a killed
# checkpoint save. Refuse to start a GPU stage while tmpfs is bloated.
require_lean_tmpfs() {
    local used_gb
    used_gb=$(df -BG --output=used /tmp | tail -1 | tr -dc 0-9)
    if ((used_gb > 5)); then
        log "REFUSING: /tmp (tmpfs, = host RAM) holds ${used_gb} GB; free it first"
        return 1
    fi
    log "tmpfs check ok (${used_gb} GB in /tmp)"
    return 0
}

# ---- serving helpers -----------------------------------------------------------
stop_serving() {
  systemctl --user stop sb-serve sb-serve-capture sb-serve-prod 2>/dev/null
  pkill -TERM -f 'vllm serve' 2>/dev/null; sleep 3
  pkill -9 -f 'VLLM::EngineCore' 2>/dev/null; sleep 2
}

# boot_serving <unit-suffix> [overrides-json]
# Boots serve_jota_r15_dual.sh as an independent user unit and waits for
# health. Overrides (applied inside the engine by the plugin's
# _apply_file_env_overrides) arm experiment flags without touching the env.
boot_serving() {
  local suffix="$1" overrides="${2:-}"
  if [ -n "$overrides" ]; then
    printf '%s' "$overrides" > /tmp/sb_env_overrides.json
  else
    rm -f /tmp/sb_env_overrides.json
  fi
  systemd-run --user --collect --unit "sb-serve-$suffix" \
    --working-directory="$PWD" \
    --setenv=HF_HUB_OFFLINE=1 \
    --setenv=SB_EXTRA_ARGS="$SERVE_ENV" \
    bash -c "exec benchmarks/serve_jota_r15_dual.sh >> /tmp/sb_serve_${suffix}.log 2>&1" \
    || return 1
  for _ in $(seq 1 60); do
    curl -fsS -m 3 http://127.0.0.1:8017/health >/dev/null 2>&1 && return 0
    sleep 10
  done
  log "boot sb-serve-$suffix: health check never passed"
  return 1
}

restore_production() {
  # Idempotent: if a server is already answering (e.g. the operator's own
  # sb-serve unit, or we exited before touching serving), leave it alone.
  # Booting a second one races the first for port 8017 and the GPU.
  if curl -fsS -m 3 http://127.0.0.1:8017/health >/dev/null 2>&1; then
    log "serving already healthy; no restore needed"
    return 0
  fi
  log "restoring production serving"
  stop_serving
  rm -f /tmp/sb_env_overrides.json
  if boot_serving prod; then
    log "production RESTORED (unit sb-serve-prod)"
  else
    log "PRODUCTION RESTORE FAILED -- operator needed"
  fi
}
trap 'log "trap: exiting"; restore_production' EXIT

# ---- stage 1: corpus -----------------------------------------------------------
if ! done_already corpus; then
  deadline=$((SECONDS + CORPUS_DEADLINE_H * 3600))
  while :; do
    n=$(wc -l < "$CORPUS" 2>/dev/null || echo 0)
    ((n >= MIN_RECORDS)) && { log "corpus complete: $n records"; break; }
    ((SECONDS >= deadline)) && { log "corpus deadline: proceeding with $n"; break; }
    sleep 120
  done
  n=$(wc -l < "$CORPUS" 2>/dev/null || echo 0)
  if ((n < 500)); then
    log "only $n records -- refusing to burn the night on a tiny pilot"
    exit 1
  fi
  # Give in-flight generation a grace window, then stop it for good.
  for _ in $(seq 1 30); do
    systemctl --user is-active --quiet sb-datagen || break
    sleep 60
  done
  systemctl --user stop sb-datagen 2>/dev/null
  mark corpus
fi

# ---- stage 2: capture ----------------------------------------------------------
if ! done_already capture; then
  require_lean_tmpfs || exit 1
  # Size the subset to the disk budget from real per-record token counts.
  .venv/bin/python - "$CORPUS" "$SUBSET" "$BUDGET_GB" <<'EOF'
import json, sys
corpus, subset, budget_gb = sys.argv[1], sys.argv[2], float(sys.argv[3])
budget, per_tok = budget_gb * 1e9, 6 * 3072 * 2  # 6 slices x H=3072 x bf16
total = kept = 0
with open(corpus) as src, open(subset, "w") as dst:
    for line in src:
        if not line.strip():
            continue
        r = json.loads(line)
        toks = int(r.get("prompt_token_count", 0)) + int(
            r.get("usage", {}).get("completion_tokens", 0))
        if toks <= 0:
            continue
        if total + toks * per_tok > budget:
            break
        total += toks * per_tok
        kept += 1
        dst.write(line)
print(f"[pipeline] capture subset: {kept} records, ~{total/1e9:.1f} GB")
EOF
  free_gb=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
  log "disk free ${free_gb} GB, capture budget ${BUDGET_GB} GB"
  ((free_gb < BUDGET_GB + 10)) && { log "insufficient disk"; exit 1; }

  stop_serving
  boot_serving capture "{\"SB_HIDDEN_CAPTURE_DIR\": \"$CAPDIR\", \"VLLM_DISABLE_COMPILE_CACHE\": \"1\"}" \
    || exit 1
  log "capture serving up; replaying subset"
  .venv/bin/python experiments/drafter_capture.py \
    --corpus "$SUBSET" --capture-dir "$CAPDIR" --concurrency 1 --retries 8
  rc=$?
  log "capture rc=$rc: $(ls "$CAPDIR" | wc -l) files, $(du -sh "$CAPDIR" | cut -f1)"
  systemctl --user stop sb-serve-capture 2>/dev/null
  stop_serving
  ((rc == 0)) || exit 1
  mark capture
fi

# ---- stage 3: warm-start conversion (CPU, serving down) -----------------------
if ! done_already warmstart; then
  stop_serving
  ws_out="experiments/drafter_train/work/warmstart"
  if [ -f "$ws_out/.complete" ]; then
    log "warm start already converted"
  else
    mkdir -p "$ws_out"
    .venv/bin/python experiments/drafter_train/prepare_warmstart.py \
      --output "$ws_out" || exit 1
    touch "$ws_out/.complete"
  fi
  mark warmstart
fi

# ---- stage 4: training (one 5090, serving down) --------------------------------
if ! done_already train; then
  require_lean_tmpfs || exit 1
  stop_serving
  log "training start ($(ls "$CAPDIR" | wc -l) capture files)"
  CUDA_VISIBLE_DEVICES=0 ./experiments/drafter_train/train.sh \
    >> /tmp/sb_train.log 2>&1
  rc=$?
  log "training rc=$rc (log: /tmp/sb_train.log, tail follows)"
  tail -5 /tmp/sb_train.log | sed 's/^/[pipeline][train] /'
  ((rc == 0)) || exit 1
  mark train
fi

# ---- stage 5: acceptance gate ---------------------------------------------------
if ! done_already acceptance; then
  stop_serving
  log "acceptance gate start"
  CUDA_VISIBLE_DEVICES=0 ./experiments/drafter_train/acceptance.sh \
    >> /tmp/sb_acceptance.log 2>&1
  rc=$?
  log "acceptance rc=$rc (log: /tmp/sb_acceptance.log, summary follows)"
  tail -15 /tmp/sb_acceptance.log | sed 's/^/[pipeline][gate] /'
  ((rc == 0)) && mark acceptance || log "acceptance FAILED -- drafter stays unshipped"
fi

log "pipeline complete"
exit 0
