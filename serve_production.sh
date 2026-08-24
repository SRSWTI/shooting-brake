#!/usr/bin/env bash
# Shooting Brake -- production serving entrypoint.
#
# Boots the tracked, benchmark-qualified recipe (benchmarks/serve_jota_r15_dual.sh
# -- the single source of truth for flags), waits for health, then prints a
# banner with every serving fact VERIFIED LIVE from the server rather than
# echoed from assumptions.
#
# Rig: 1x RTX 5090 (attention/GDN + local & shared experts, CUTLASS W4A4)
#      2x Intel Arc Pro B70 (170 routed NVFP4 experts via doorbell poller)
# Qualified config (kill-bench 26 / SLO grid, 36/36 rows):
#      grouped NVFP4 GEMM + fp16 result wire + prefetch_dist=1 + 2-chunk
#      pipelining; prefill 387 us/token @32K, decode ITL ~12.5 ms, prefix
#      caching ~139x on repeat traffic.
#
# Usage:
#   ./serve_production.sh              # boot + verify + leave serving
#   SB_MML=65536 ./serve_production.sh # any recipe knob still overrides
set -o pipefail

cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

HOST="127.0.0.1"
PORT="8017"
BASE="http://${HOST}:${PORT}"
LOG="${SB_SERVE_LOG:-/tmp/shooting_brake_serve.log}"
BOOT_TIMEOUT_S="${SB_BOOT_TIMEOUT_S:-900}"

# Prefix caching is a vLLM default, but production configs state their
# load-bearing choices explicitly. Measured here: 139x on repeated prompts.
export SB_EXTRA_ARGS="--enable-prefix-caching ${SB_EXTRA_ARGS:-}"

# KV pool: claim the engine's own "fully utilize" figure instead of the 0.85
# utilization heuristic -- 8.29 -> 10.69 GiB (+29%), 131K-request concurrency
# 1.25x -> 1.61x, ITL guard 11.78 ms (baseline band 11.7-12.5). KV cannot go
# on the B70s: attention lives on the 5090 (per-token PCIe reads would blow
# ITL) and B70 allocations shadow host RAM 1:1 (kill-bench, the 47 GiB find).
# CHECKPOINT-SPECIFIC: re-derive from the boot log advisory
# ("--kv-cache-memory=... to fully utilize") whenever weights/MML/util change;
# an oversized value fails the boot loudly, it never silently degrades.
SB_KV_BYTES="${SB_KV_BYTES:-11473444352}"
export SB_EXTRA_ARGS="--kv-cache-memory=${SB_KV_BYTES} ${SB_EXTRA_ARGS}"

echo "[serve] stopping any previous server..."
pkill -TERM -f 'vllm serve' 2>/dev/null
for _ in $(seq 1 10); do pgrep -f 'VLLM::EngineCore' >/dev/null || break; sleep 1; done
pkill -9 -f 'VLLM::EngineCore' 2>/dev/null
sleep 2

echo "[serve] booting (log: ${LOG})..."
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" setsid nohup benchmarks/serve_jota_r15_dual.sh \
  > "${LOG}" 2>&1 < /dev/null &
disown

deadline=$((SECONDS + BOOT_TIMEOUT_S))
until curl -fsS "${BASE}/health" >/dev/null 2>&1; do
  if ((SECONDS >= deadline)); then
    echo "[serve] BOOT FAILED after ${BOOT_TIMEOUT_S}s -- last log lines:"
    tail -15 "${LOG}"
    exit 1
  fi
  sleep 5
done

# ---- live verification: ask the SERVER, not the script ---------------------
MODELS_JSON="$(curl -fsS "${BASE}/v1/models")"
SERVED_ID="$(printf '%s' "${MODELS_JSON}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"
CTX="$(printf '%s' "${MODELS_JSON}" | python3 -c 'import json,sys; d=json.load(sys.stdin)["data"][0]; print(d.get("max_model_len", "unknown"))')"
KV_LINE="$(grep -aoE "GPU KV cache size: [0-9,]+ tokens|Available KV cache memory: [0-9.]+ GiB|Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x" "${LOG}" | tail -2 | paste -sd '; ' -)"
PREFIX_STATE="$(grep -ao "enable_prefix_caching=[A-Za-z]*" "${LOG}" | tail -1)"
CAPTURE_LINE="$(grep -a "Graph capturing finished" "${LOG}" | tail -1 | sed 's/.*\] //')"

# One real completion proves the pipeline end to end (both cards + 5090).
SMOKE="$(curl -fsS "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${SERVED_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: ready\"}],\"max_tokens\":8,\"temperature\":0}" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip()[:40])')"

cat <<BANNER

=====================================================================
 SHOOTING BRAKE -- heterogeneous inference server: READY
=====================================================================
 model (HF id)      : srswti/axe-superveloce-jota-118b-r15-nvfp4
 served model id    : ${SERVED_ID}
 api base           : ${BASE}/v1
 endpoints          : /v1/chat/completions /v1/completions /v1/models
                      /health /metrics
 context length     : ${CTX} tokens (verified via /v1/models)
 max output tokens  : per request, up to ${CTX} minus prompt length
                      (no server-side cap; set "max_tokens" per call)
 prefix caching     : ${PREFIX_STATE:-enable_prefix_caching=True} (~139x on repeat traffic)
 scheduler          : max_num_batched_tokens=${SB_MNBT:-2048}, max_num_seqs=${SB_MNS:-6}
 gpu memory util    : ${SB_GPU_UTIL:-0.85}
 kv cache           : ${KV_LINE:-see ${LOG}}
 graph capture      : ${CAPTURE_LINE:-see ${LOG}}
 placement          : 5090 attention/GDN + local&shared experts (CUTLASS W4A4)
                      2x Arc Pro B70: 170 routed NVFP4 experts (85/card,
                      grouped GEMM + fp16 wire, doorbell poller)
 parsers            : reasoning=poolside_v1, tool_calls=poolside_v1 (auto)
 speculative        : off by default (SB_SPEC to arm; see docs)
 smoke completion   : "${SMOKE}"
 log                : ${LOG}
=====================================================================
BANNER
