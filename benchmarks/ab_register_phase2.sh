#!/usr/bin/env bash
# A/B phase 2, with the lessons from phase 1 baked in:
#   - the full-bank pin evicts ~9 GiB of server working set on first prefill;
#     warm the server and wait for memory pressure to settle BEFORE measuring
#   - prompt_logprobs over the API cannot gate the streamer (OOM >1K tokens,
#     no streamer <1024 tokens); gate on generated-token logprobs instead
#   - gate threshold is the cross-boot envelope (0.49 bug signature), not
#     bit-equality: different boots autotune different reduction orders
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
export PATH="$PWD/.venv/bin:$PATH"
PY=.venv/bin/python
OUT=benchmarks/results/run5_88b_register
PROMPT8K=benchmarks/results/prefill_profile/prompt_8k.txt
TARGET=http://127.0.0.1:8016

boot() {  # $1 = extra env assignment or empty, $2 = logfile
  pkill -f "vllm serve srswti/axe-superveloce-88b-nvfp4a16" || true
  for i in $(seq 30); do
    pgrep -f "vllm serve srswti/axe-superveloce" >/dev/null || break; sleep 2
  done
  sleep 5
  env $1 bash benchmarks/serve_88b_128k.sh >"$2" 2>&1 &
  for i in $(seq 180); do curl -sf "$TARGET/health" >/dev/null && return 0; sleep 5; done
  echo "boot failed: $2"; exit 1
}

warm() {  # fire 8K prefills to flush the pin-eviction tail, wait for psi calm
  for i in 1 2; do
    curl -sf -X POST "$TARGET/v1/completions" -H 'Content-Type: application/json' \
      -d "$(jq -n --rawfile p "$PROMPT8K" '{model:"shooting-brake-88b",prompt:$p,max_tokens:4,temperature:0}')" \
      >/dev/null || true
  done
  for i in $(seq 60); do
    avg10=$(awk -F'avg10=' '/^some/{split($2,a," ");print a[1]}' /proc/pressure/memory)
    awk -v v="$avg10" 'BEGIN{exit !(v<0.1)}' && break
    sleep 5
  done
  echo "memory psi avg10=$avg10 after warm"
}

echo "=== register arm"
boot "SHOOTING_BRAKE_BANK_REGISTER=1" "$OUT/server_register2.log"
warm
$PY benchmarks/cuda_decode_profile.py measure --target "$TARGET" \
  --model shooting-brake-88b --prompt "$(cat "$PROMPT8K")" \
  --warmup-tokens 4 --decode-steps 64 \
  --label register-prefill-8k --out "$OUT/ttft_8k.json" >/dev/null
$PY benchmarks/logprob_capture.py capture --gen-tokens 64 \
  --target "$TARGET" --prompt-file "$PROMPT8K" --out "$OUT/genlp_on.json"
$PY benchmarks/bench_88b.py --root "$OUT/matrix" \
  --cells B_context/ctx_8192,B_context/ctx_32768 2>&1 | tee "$OUT/suite_run.log"

echo "=== flag-off arm (gate baseline)"
boot "" "$OUT/server_off2.log"
warm
$PY benchmarks/logprob_capture.py capture --gen-tokens 64 \
  --target "$TARGET" --prompt-file "$PROMPT8K" --out "$OUT/genlp_off.json"
$PY benchmarks/logprob_capture.py diff --a "$OUT/genlp_on.json" \
  --b "$OUT/genlp_off.json" --gate 0.49 --out "$OUT/genlp_ab.json"

echo "=== phase 2 done; flag-off server left running"
