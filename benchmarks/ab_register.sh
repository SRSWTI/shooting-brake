#!/usr/bin/env bash
# Register-path A/B, PHASE 1 (historical): microbench trio + first boot.
#
# SUPERSEDED for the numerics gate by ab_register_phase2.sh — this script's
# prompt_logprobs gate is a recorded dead end: >1K-token prompts OOM-kill
# the engine in vLLM's prompt-logprobs path, <1024-token prompts never
# engage the streamer, and bit-equality across boots does not exist (see
# docs/next-steps-88b.md, run5 section). The microbench trio steps here
# remain the canonical way to re-run floor_{register,compute,overlap}.
#
# Everything lands under benchmarks/results/run5_88b_register/.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
# JIT extension builds (B70 graph path) invoke bare `ninja`; it lives in the
# venv, which non-activated shells don't have on PATH.
export PATH="$PWD/.venv/bin:$PATH"
PY=.venv/bin/python
OUT=benchmarks/results/run5_88b_register
PROMPT8K=benchmarks/results/prefill_profile/prompt_8k.txt
TARGET=http://127.0.0.1:8016
mkdir -p "$OUT"

if pgrep -f "bench_88b.py --root benchmarks/results/run4_88b_bank" >/dev/null \
   || pgrep -f "guidellm run" >/dev/null; then
  echo "run4 suite still running -- refusing to touch the GPU"; exit 1
fi

echo "=== [1/4] baseline logprobs from live flag-off server"
if curl -sf "$TARGET/health" >/dev/null; then
  $PY benchmarks/logprob_capture.py capture --max-chars 4000 \
    --target "$TARGET" --prompt-file "$PROMPT8K" \
    --out "$OUT/logprobs_off.json"
else
  echo "no live server; will need a flag-off boot for the baseline arm"
  NEED_OFF_ARM=1
fi

echo "=== [2/4] stopping server, running microbench trio"
pkill -f "vllm serve srswti/axe-superveloce-88b-nvfp4a16" || true
for i in $(seq 30); do
  pgrep -f "vllm serve srswti/axe-superveloce-88b-nvfp4a16" >/dev/null || break
  sleep 2
done
sleep 5  # let the driver release

if [ -n "$NEED_OFF_ARM" ]; then
  echo "--- flag-off boot for baseline logprobs"
  bash benchmarks/serve_88b_128k.sh >"$OUT/server_off.log" 2>&1 &
  for i in $(seq 180); do curl -sf "$TARGET/health" >/dev/null && break; sleep 5; done
  $PY benchmarks/logprob_capture.py capture --max-chars 4000 \
    --target "$TARGET" --prompt-file "$PROMPT8K" --out "$OUT/logprobs_off.json"
  pkill -f "vllm serve srswti/axe-superveloce-88b-nvfp4a16" || true
  sleep 10
fi

for mode in register compute overlap; do
  $PY benchmarks/prefill_floor_bench.py --mode "$mode" \
    --out "$OUT/floor_${mode}.json"
done

echo "=== [3/4] launching register-path server"
SHOOTING_BRAKE_BANK_REGISTER=1 bash benchmarks/serve_88b_128k.sh \
  >"$OUT/server_register.log" 2>&1 &
for i in $(seq 180); do
  curl -sf "$TARGET/health" >/dev/null && break
  sleep 5
done
curl -sf "$TARGET/health" >/dev/null || { echo "server failed to boot"; exit 1; }
grep -m1 "Registered .* bank page cache" "$OUT/server_register.log" || true

echo "=== [4/4] TTFT spot + logprob gate + matrix cells"
$PY benchmarks/cuda_decode_profile.py measure \
  --target "$TARGET" --model shooting-brake-88b \
  --prompt "$(cat "$PROMPT8K")" --decode-steps 64 \
  --label register-prefill-8k --out "$OUT/ttft_8k.json"

$PY benchmarks/logprob_capture.py capture --max-chars 4000 \
  --target "$TARGET" --prompt-file "$PROMPT8K" --out "$OUT/logprobs_on.json"
$PY benchmarks/logprob_capture.py diff \
  --a "$OUT/logprobs_on.json" --b "$OUT/logprobs_off.json" \
  --out "$OUT/logprob_ab.json"

$PY benchmarks/bench_88b.py --root "$OUT/matrix" \
  --cells B_context/ctx_8192,B_context/ctx_32768 2>&1 \
  | tee "$OUT/suite_run.log"

echo "=== done; server left running with SHOOTING_BRAKE_BANK_REGISTER=1"
echo "results: $OUT/{floor_*,ttft_8k,logprob_ab}.json + matrix/"
