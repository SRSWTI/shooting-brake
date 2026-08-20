#!/usr/bin/env bash
# Laguna jota-118b-r15 NVFP4 on dual B70, with thinking and tools enabled.
#
# Deltas from serve_jota_r20_dual.sh, and why:
#   MODEL          r15: 218 experts instead of r20's 205 (less REAP pruning).
#                  Every integration mechanism is shared; only the bank size
#                  (50.65 vs 47.63 GiB) and the remote split (161 = 81/80 vs
#                  151 = 76/75) change.
#   PLACEMENT      IDENTICAL string. round(218 * 0.2634...) == 57 local, so
#                  the same default lands on a sane split for both models.
#   reasoning      --reasoning-parser poolside_v1. Laguna is Poolside's
#                  architecture and its chat_template.jinja sets
#                  `enable_thinking | default(true)` -- thinking is ON unless
#                  the caller opts out. WITHOUT a reasoning parser the raw
#                  `</think>` tag leaks into content (measured on r20); WITH
#                  it, the trace lands in `reasoning_content` and `content`
#                  stays clean.
#   tools          --enable-auto-tool-choice --tool-call-parser poolside_v1.
#                  The template carries `tools` and `tool_calls` blocks.
#   max-model-len  131072. KV at L=57 is ~10.3 GiB and Laguna costs ~50.2
#                  KiB/token, so ~215K tokens fit; 128K leaves ~1.6x
#                  concurrency. Laguna's positional limit is 1,048,576 (YaRN
#                  factor 128 over an 8192 base), so this is a memory choice,
#                  not an architectural one. Raise SB_MML if you would rather
#                  have one very long sequence than several long ones.
#   max-num-seqs   4, matching the 99B recipe.
# No `set -u`: sourcing oneAPI setvars.sh references unset vars and would abort.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
source /opt/intel/oneapi/setvars.sh --force >/dev/null
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH="$PWD/src/phase4/src:$PYTHONPATH"
export VLLM_PLUGINS=shooting_brake_vllm
export SHOOTING_BRAKE_PHASE4=all-cuda
export SHOOTING_BRAKE_MODEL=srswti/axe-superveloce-jota-118b-r15-nvfp4
export SHOOTING_BRAKE_PLACEMENT="${SHOOTING_BRAKE_PLACEMENT:-fractional:2:0.2634146341463415}"
export SHOOTING_BRAKE_HYBRID=1
export SHOOTING_BRAKE_B70_DEVICE=1
export SHOOTING_BRAKE_B70_GRAPH=1
export SHOOTING_BRAKE_PREEMPTIVE_SURGERY=1
export SHOOTING_BRAKE_VRAM_SURGERY=1
export SHOOTING_BRAKE_B70_PREFILL_STREAM=0
export SHOOTING_BRAKE_PREFILL_MARLIN=0
# Would hide the second card. Cards are selected per provider by PCI BDF.
unset ZE_AFFINITY_MASK
export SHOOTING_BRAKE_B70_BANK="$PWD/src/phase1/expert_bank_jota_118b_r15.bin"
export SHOOTING_BRAKE_B70_BANKS="${SHOOTING_BRAKE_B70_BANKS:-$PWD/src/phase1/expert_bank_jota_118b_r15.bin,$PWD/src/phase1/expert_bank_jota_118b_r15.bin}"
# Device index -> BDF. Order is load-bearing: index 0 is the Gen4 card. BDFs,
# never enumeration indices -- index order once silently picked the Gen3 card
# and cost 31% ITL for weeks.
export SHOOTING_BRAKE_B70_SELECTORS="${SHOOTING_BRAKE_B70_SELECTORS:-0000:15:00.0,0000:11:00.0}"
export SHOOTING_BRAKE_B70_POLL_CPUS="${SHOOTING_BRAKE_B70_POLL_CPUS:-5,6}"
export SHOOTING_BRAKE_B70_LIB="$PWD/src/phase7/libsb_b70_provider.so"
export SHOOTING_BRAKE_B70_MAX_BATCH="${SHOOTING_BRAKE_B70_MAX_BATCH:-256}"
# ROUTE_TRACE stages topk_ids D2H inside the routed-expert forward, which is
# not capture-safe: with it set, graph capture dies with
# cudaErrorStreamCaptureUnsupported before the engine ever serves.
unset SHOOTING_BRAKE_B70_PROFILE SHOOTING_BRAKE_B70_STATS SHOOTING_BRAKE_ROUTE_TRACE
unset VLLM_USE_BREAKABLE_CUDAGRAPH
export SHOOTING_BRAKE_EXPECT_ARM=doorbell
export SHOOTING_BRAKE_B70_TRACE_DUMP="${SHOOTING_BRAKE_B70_TRACE_DUMP:-/tmp/sb_r15_trace.json}"
# sm_120 W4A4 MoE: FlashInfer's CUTLASS MoE tuner wedges on this shape, and
# skipping those ops reaches a heuristic that faults with a misaligned
# address. Pin to vLLM's in-tree CUTLASS experts (kill-bench 16).
export VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS="trtllm::fused_moe::gemm1,trtllm::fused_moe::gemm2"
echo "=== launching jota-r15 dual-B70 $(date '+%H:%M:%S')"
exec .venv/bin/vllm serve srswti/axe-superveloce-jota-118b-r15-nvfp4 \
  --served-model-name shooting-brake-jota-r15 \
  --host 0.0.0.0 --port 8017 \
  --trust-remote-code \
  --moe-backend "${SB_MOE_BACKEND:-cutlass}" \
  --max-model-len "${SB_MML:-131072}" \
  --max-num-batched-tokens "${SB_MNBT:-2048}" \
  --gpu-memory-utilization "${SB_GPU_UTIL:-0.90}" \
  --max-num-seqs "${SB_MNS:-4}" \
  --reasoning-parser poolside_v1 \
  --enable-auto-tool-choice \
  --tool-call-parser poolside_v1 \
  --default-chat-template-kwargs "${SB_TEMPLATE_KWARGS:-{\"enable_thinking\": true, \"thinking\": true\}}" ${SB_EXTRA_ARGS}
