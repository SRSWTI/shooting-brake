#!/usr/bin/env bash
# First dual-B70 serve recipe: 99B NVFP4, one monolithic bank, two cards.
#
# Deltas from serve_88b_128k.sh, and why:
#   MODEL/PLACEMENT      99B, fractional:2:<1/205>: 1 expert stays on CUDA
#                        (the fused-partial dummy-slot rule), 204 split
#                        102/102 across both B70s.
#   B70_INT4             GONE — the 99B decode bank is NVFP4 (SBEXP001).
#   ZE_AFFINITY_MASK     UNSET, deliberately: it would hide the second
#                        card. Cards are selected per provider by PCI BDF.
#   B70_SELECTORS        device 0 -> 0000:15:00.0 (Gen4 x4),
#                        device 1 -> 0000:11:00.0 (Gen3 x4). BDFs, never
#                        enumeration indices — index order once silently
#                        picked the Gen3 card and cost 31% ITL for weeks.
#   B70_BANKS            the SAME monolithic bank twice: SBEXP001 loads
#                        take a per-card resident list, so there are no
#                        per-split bank files in the NVFP4 world.
#   B70_POLL_CPUS        one pinned core per poller (8 logical CPUs total;
#                        two spinning pollers must never share one).
#   PREFILL_MARLIN       0 — no NVFP4 Marlin prefill bank exists yet, so
#                        prefill runs the chunked doorbell dispatch. TTFT
#                        will be poor; this recipe is for decode gates and
#                        first-boot correctness, not prefill numbers.
#   max-model-len        32768 first boot; raise only after KV sizing is
#                        measured on THIS model (dense footprint differs
#                        from the 88B).
# No `set -u`: sourcing oneAPI setvars.sh references unset vars and would abort.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
source /opt/intel/oneapi/setvars.sh --force >/dev/null
export PYTHONPATH="$PWD/src/phase4/src:$PYTHONPATH"
export VLLM_PLUGINS=shooting_brake_vllm
export SHOOTING_BRAKE_PHASE4=all-cuda
export SHOOTING_BRAKE_MODEL=srswti/axe-superveloce-99b-nvfp4
export SHOOTING_BRAKE_PLACEMENT="fractional:2:0.0048780487804878"
export SHOOTING_BRAKE_HYBRID=1
export SHOOTING_BRAKE_B70_DEVICE=1
export SHOOTING_BRAKE_B70_GRAPH=1
export SHOOTING_BRAKE_PREEMPTIVE_SURGERY=1
export SHOOTING_BRAKE_VRAM_SURGERY=1
export SHOOTING_BRAKE_B70_PREFILL_STREAM=0
export SHOOTING_BRAKE_PREFILL_MARLIN=0
unset ZE_AFFINITY_MASK
export SHOOTING_BRAKE_B70_BANK="$PWD/src/phase1/expert_bank_99b.bin"
export SHOOTING_BRAKE_B70_BANKS="$PWD/src/phase1/expert_bank_99b.bin,$PWD/src/phase1/expert_bank_99b.bin"
export SHOOTING_BRAKE_B70_SELECTORS="0000:15:00.0,0000:11:00.0"
export SHOOTING_BRAKE_B70_POLL_CPUS="${SHOOTING_BRAKE_B70_POLL_CPUS:-5,6}"
export SHOOTING_BRAKE_B70_LIB="$PWD/src/phase7/libsb_b70_provider.so"
export SHOOTING_BRAKE_B70_MAX_BATCH="${SHOOTING_BRAKE_B70_MAX_BATCH:-256}"
unset SHOOTING_BRAKE_B70_PROFILE VLLM_USE_BREAKABLE_CUDAGRAPH
# sm_120 W4A4 MoE facts, measured on first boot (2026-08-19, kill-bench 16):
#   * FlashInfer's trtllm-gen fused-moe tactics HANG this GeForce Blackwell
#     (one tuner profile ran 25 min at 100% GPU); its untuned heuristic
#     tactic faults with a misaligned address. Skip those ops in the tuner
#     AND pin the MoE backend to vLLM's in-tree CUTLASS experts, which
#     produced the first correct 99B tokens.
export VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS="trtllm::fused_moe::gemm1,trtllm::fused_moe::gemm2"
echo "=== launching 99B dual-B70 $(date '+%H:%M:%S')"
exec .venv/bin/vllm serve srswti/axe-superveloce-99b-nvfp4 \
  --served-model-name shooting-brake-99b \
  --host 127.0.0.1 --port 8017 \
  --trust-remote-code \
  --moe-backend cutlass \
  --max-model-len "${SB_MML:-32768}" \
  --max-num-batched-tokens "${SB_MNBT:-2048}" \
  --gpu-memory-utilization "${SB_GPU_UTIL:-0.85}" \
  --max-num-seqs "${SB_MNS:-4}" \
  --reasoning-parser qwen3 ${SB_EXTRA_ARGS}
