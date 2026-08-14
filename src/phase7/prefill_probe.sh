#!/usr/bin/env bash
# Measure prefill quality across expert-placement configurations.
#
# Prompt logprobs are the instrument. They are produced during prefill, one
# per prompt position, so they read that pass directly -- unlike sampled
# token ids, which only reveal damage severe enough to flip an argmax. That
# distinction is not academic: the pre-Phase-6 prefill path emitted
# byte-identical tokens while losing 0.49 nats/token.
#
#   all-cuda       ground truth: every expert on CUDA, no surgery
#   prefix-broken  archived pre-Phase-6 result, kept as the regression floor
#   phase6         subset:16:8 through the Phase 6 prefill path
#   allout         allout:16:8:8 -- adds the cold tier, weights streamed
#
# phase6 and allout must both land at all-cuda's logprob. Landing near
# prefix-broken instead means offloaded routes are still being dropped.
#
# oneAPI's setvars.sh calls exit internally, so it must be sourced before
# strict mode; it also strips the venv from PATH, so PATH is restored after.
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PATH="$REPO/.venv/bin:$PATH"

OUT_DIR="${SB_PROBE_DIR:-/tmp/sb_prefill_probe}"
mkdir -p "$OUT_DIR"

export VLLM_PLUGINS=shooting_brake_vllm
export SHOOTING_BRAKE_PHASE4=all-cuda
export SHOOTING_BRAKE_MODEL=unsloth/Qwen3.6-35B-A3B-NVFP4
export SHOOTING_BRAKE_B70_BANK="$REPO/phase1/expert_bank.bin"
export SHOOTING_BRAKE_B70_LIB="$REPO/phase7/libsb_b70_provider.so"
export SHOOTING_BRAKE_CPU_LIB="$REPO/phase7/libsb_cpu_expert.so"

run_case() {
  local label="$1"; shift
  local out="$OUT_DIR/$label.json"
  if [[ -f "$out" && "${SB_SKIP_EXISTING:-1}" == "1" ]]; then
    echo "== $label: cached"
    return 0
  fi
  echo "== $label: running"
  (
    export SB_LABEL="$label" SB_OUT="$out"
    for kv in "$@"; do export "${kv?}"; done
    python src/phase7/prefill_probe.py >"$OUT_DIR/$label.log" 2>&1
  ) || { echo "!! $label FAILED — tail of log:"; tail -25 "$OUT_DIR/$label.log"; return 1; }
  echo "   ok"
}

# Features shared by every offloaded case. Kept identical across them so the
# only thing that varies is the placement policy.
HYBRID=(
  SHOOTING_BRAKE_HYBRID=1
  SHOOTING_BRAKE_B70_DEVICE=1
  SHOOTING_BRAKE_B70_GRAPH=1
  SHOOTING_BRAKE_VRAM_SURGERY=1
)

# The all-cuda case sets no hybrid variables at all; SHOOTING_BRAKE_PLACEMENT
# is never exported by this script, and the plugin defaults to "all-cuda".
# Passing it empty is not the same thing — the policy parser rejects "".
run_case all-cuda
run_case phase6 "${HYBRID[@]}" SHOOTING_BRAKE_PLACEMENT=subset:16:8
# Same placement as phase6, but B70-owned routes are computed on the 5090
# from streamed weights instead of dispatched to the B70. The threshold is
# forced to 1 because this probe's prompt is ~138 tokens, far below the
# production default -- without it the case would silently measure the
# dispatch path again and report a meaningless pass.
run_case phase6-stream "${HYBRID[@]}" SHOOTING_BRAKE_PLACEMENT=subset:16:8 \
                SHOOTING_BRAKE_B70_PREFILL_STREAM=1 \
                SHOOTING_BRAKE_B70_STREAM_T=1
run_case allout "${HYBRID[@]}" SHOOTING_BRAKE_PLACEMENT=allout:16:8:8 \
                SHOOTING_BRAKE_ALL_OUT=1 SHOOTING_BRAKE_CPU_VERIFY=1

echo
python - "$OUT_DIR" <<'PY'
import json, pathlib, sys

d = pathlib.Path(sys.argv[1])
cases = {}
for name in ("all-cuda", "prefix-broken", "phase6", "allout"):
    p = d / f"{name}.json"
    if p.exists():
        cases[name] = json.loads(p.read_text())
if not cases:
    sys.exit("no probe results")

base = cases.get("all-cuda")
print(f"{'case':10} {'prompt':>6} {'sum logprob':>12} {'mean':>8} "
      f"{'vs base':>10}  tokens")
print("-" * 82)
for name, c in cases.items():
    if base is None:
        delta = "?"
    else:
        d = c["prompt_logprob_mean"] - base["prompt_logprob_mean"]
        delta = f"{d:+.4f}"
    tok = "identical" if base and c["token_ids"] == base["token_ids"] else (
        "differs" if base else "?"
    )
    print(f"{name:10} {c['prompt_tokens']:>6} "
          f"{c['prompt_logprob_sum']:>12.3f} "
          f"{c['prompt_logprob_mean']:>8.4f} {delta:>10}  {tok}")

print()
print("prompt_logprob_mean is computed during prefill, so it reads that pass")
print("directly. A path that drops routed-expert output lands far below the")
print("all-cuda baseline even when the sampled tokens happen to match.")
PY
