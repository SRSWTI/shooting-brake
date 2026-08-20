#!/usr/bin/env bash
# First serve recipe for Laguna: jota-118b-r20 NVFP4, one monolithic 47-row
# bank, two B70s.
#
# Deltas from serve_99b_dual.sh, and why:
#   MODEL                srswti/axe-superveloce-jota-118b-r20-nvfp4.
#                        LagunaForCausalLM, NOT a Qwen variant: 47 sparse MoE
#                        layers plus a DENSE MLP at layer 0, sliding-window
#                        attention (512) instead of GDN, per-head attention
#                        gating, top_k 10. It is not IsHybrid, so vLLM
#                        prefix-caches it -- the entire reason this model is
#                        being brought up (142x on a repeated 6,023-token
#                        prompt, where the 99B managed 1.007x).
#   PLACEMENT            SAME fraction as the 99B. r20 keeps 205 experts, so
#                        0.2634... is still 54 local / 151 remote split 76/75,
#                        and the resident-set arithmetic is unchanged. That
#                        equality is why r20 was chosen over r15 (218).
#   B70_BANKS            expert_bank_jota_118b_r20.bin, the same monolithic
#                        file twice. 47 ROWS, not 48: bank row 0 holds MODEL
#                        LAYER 1, because layer 0 is dense and owns no
#                        experts. The absolute ids live in the qualified
#                        spec's bank_layer_ids and every native crossing
#                        translates through bank_row_for_model_layer().
#                        Byte-validated against the checkpoint through that
#                        map by src/phase1/validate_expert_bank.py.
#   top_k                10, carried to the provider by sb_b70_load_v2. The
#                        frozen v1 entry point would silently serve width 8.
#   reasoning-parser     DROPPED. qwen3's parser does not describe Laguna's
#                        output and an unmatched parser is a response-shaping
#                        risk for no benefit on a correctness boot.
#   max-model-len        32768 first boot, as the 99B's was. r20's dense
#                        footprint is 3.18 GiB against the 99B's 10.24, so at
#                        L=54 there is far more KV headroom -- but its KV is
#                        1.70x per token (8 kv heads x 128 vs 2 x 256), so
#                        measure before raising anything.
#   PREFILL_MARLIN       0. No NVFP4 Marlin prefill bank exists for any model
#                        yet, so prefill runs the chunked doorbell dispatch.
#                        TTFT is expected to be poor; this recipe is for
#                        correctness and decode, not prefill numbers.
# No `set -u`: sourcing oneAPI setvars.sh references unset vars and would abort.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
source /opt/intel/oneapi/setvars.sh --force >/dev/null
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH="$PWD/src/phase4/src:$PYTHONPATH"
export VLLM_PLUGINS=shooting_brake_vllm
export SHOOTING_BRAKE_PHASE4=all-cuda
export SHOOTING_BRAKE_MODEL=srswti/axe-superveloce-jota-118b-r20-nvfp4
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
export SHOOTING_BRAKE_B70_BANK="$PWD/src/phase1/expert_bank_jota_118b_r20.bin"
export SHOOTING_BRAKE_B70_BANKS="${SHOOTING_BRAKE_B70_BANKS:-$PWD/src/phase1/expert_bank_jota_118b_r20.bin,$PWD/src/phase1/expert_bank_jota_118b_r20.bin}"
# Device index -> BDF. Order is load-bearing: index 0 is the Gen4 card. BDFs,
# never enumeration indices -- index order once silently picked the Gen3 card
# and cost 31% ITL for weeks.
export SHOOTING_BRAKE_B70_SELECTORS="${SHOOTING_BRAKE_B70_SELECTORS:-0000:15:00.0,0000:11:00.0}"
export SHOOTING_BRAKE_B70_POLL_CPUS="${SHOOTING_BRAKE_B70_POLL_CPUS:-5,6}"
export SHOOTING_BRAKE_B70_LIB="$PWD/src/phase7/libsb_b70_provider.so"
export SHOOTING_BRAKE_B70_MAX_BATCH="${SHOOTING_BRAKE_B70_MAX_BATCH:-256}"
# ROUTE_TRACE stages topk_ids D2H inside the routed-expert forward, which is
# not capture-safe: with it set, graph capture dies with
# cudaErrorStreamCaptureUnsupported before the engine ever serves. This arm
# asserts `doorbell` (captured graphs), so unset it rather than let a stray
# export from an eager calibration run kill the boot.
unset SHOOTING_BRAKE_B70_PROFILE SHOOTING_BRAKE_B70_STATS SHOOTING_BRAKE_ROUTE_TRACE
unset VLLM_USE_BREAKABLE_CUDAGRAPH
export SHOOTING_BRAKE_EXPECT_ARM=doorbell
export SHOOTING_BRAKE_B70_TRACE_DUMP="${SHOOTING_BRAKE_B70_TRACE_DUMP:-/tmp/sb_r20_trace.json}"
# sm_120 W4A4 MoE facts, measured 2026-08-19 (kill-bench 16) and re-confirmed
# for r20 in the CPU-offload gate: FlashInfer's CUTLASS MoE backend wedges in
# one tuner tactic on this shape; skipping those ops reaches a heuristic that
# faults with a misaligned address. Pin to vLLM's in-tree CUTLASS experts.
export VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS="trtllm::fused_moe::gemm1,trtllm::fused_moe::gemm2"
echo "=== launching jota-r20 dual-B70 $(date '+%H:%M:%S')"
exec .venv/bin/vllm serve srswti/axe-superveloce-jota-118b-r20-nvfp4 \
  --served-model-name shooting-brake-jota-r20 \
  --host 127.0.0.1 --port 8017 \
  --trust-remote-code \
  --moe-backend "${SB_MOE_BACKEND:-cutlass}" \
  --max-model-len "${SB_MML:-32768}" \
  --max-num-batched-tokens "${SB_MNBT:-2048}" \
  --gpu-memory-utilization "${SB_GPU_UTIL:-0.85}" \
  --max-num-seqs "${SB_MNS:-4}" ${SB_EXTRA_ARGS}
