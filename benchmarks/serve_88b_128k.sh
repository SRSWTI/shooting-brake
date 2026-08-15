#!/usr/bin/env bash
# 128K-context variant of serve_88b.sh. The baseline script is left untouched so
# the 60-65 tok/s measurement stays reproducible at its own configuration.
#
# Deltas from serve_88b.sh, and why:
#   --max-model-len       32768 -> 131072   the point of this run
#   --max-num-batched-tokens 256 -> 8192    at 256 a 128K prefill is 512 chunks
#                                           x 48 layers = 24,576 dispatches
#   --max-num-seqs           4 -> SB_MNS    KV supports ~456 short seqs, but
#                                           GDN/mamba per-seq state lives
#                                           outside the KV budget and OOM'd
#                                           profiling at 64 with the marlin
#                                           streamer resident; default 16
#                                           (knee was C~16), override SB_MNS.
#   B70_MAX_BATCH           256 -> 2048     4 sub-chunks per 8192-token chunk.
#                                           Not 8192: prefill_chunk_bench.py
#                                           measured 25->2 dispatches as a 5.1%
#                                           TTFT move, so the extra 315 MB of
#                                           B70 staging buys ~nothing.
# No `set -u`: sourcing oneAPI setvars.sh references unset vars and would abort.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
source /opt/intel/oneapi/setvars.sh --force >/dev/null
export PYTHONPATH="$PWD/src/phase4/src:$PYTHONPATH"
export VLLM_PLUGINS=shooting_brake_vllm
export SHOOTING_BRAKE_PHASE4=all-cuda
export SHOOTING_BRAKE_MODEL=srswti/axe-superveloce-88b-nvfp4a16
export SHOOTING_BRAKE_PLACEMENT=split:54
export SHOOTING_BRAKE_HYBRID=1
export SHOOTING_BRAKE_B70_INT4=1
export SHOOTING_BRAKE_B70_DEVICE=1
export SHOOTING_BRAKE_B70_GRAPH=1
export SHOOTING_BRAKE_PREEMPTIVE_SURGERY=1
export SHOOTING_BRAKE_VRAM_SURGERY=1
export SHOOTING_BRAKE_B70_PREFILL_STREAM=0
export SHOOTING_BRAKE_B70_MAX_BATCH=2048
export SHOOTING_BRAKE_B70_BANK="$PWD/src/phase1/expert_bank_int4.bin"
export SHOOTING_BRAKE_B70_LIB="$PWD/src/phase7/libsb_b70_provider.so"
# Measured wins, promoted to defaults (both overridable):
#   ZE_AFFINITY_MASK=1        pins Level Zero to the Gen4-x4 B70 (0000:15:00.0).
#                             select_b70() otherwise takes the first-enumerated
#                             Gen3-x4 card: decode ITL 16.52 -> 11.31 ms (-31%),
#                             8K TTFT 21.86 -> 18.36 s on the dispatch path.
#   SHOOTING_BRAKE_PREFILL_MARLIN=1  streams int4 experts to the 5090 for
#                             prefill (fused_marlin_moe); 8K TTFT 3.07 s vs
#                             21.86 s dispatch. Decode path untouched.
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-1}"
export SHOOTING_BRAKE_PREFILL_MARLIN="${SHOOTING_BRAKE_PREFILL_MARLIN:-1}"
unset SHOOTING_BRAKE_B70_PROFILE VLLM_USE_BREAKABLE_CUDAGRAPH
echo "=== launching 128K $(date '+%H:%M:%S')"
exec .venv/bin/vllm serve srswti/axe-superveloce-88b-nvfp4a16 \
  --served-model-name shooting-brake-88b \
  --host 127.0.0.1 --port 8016 \
  --trust-remote-code --language-model-only \
  --max-model-len 131072 --max-num-batched-tokens "${SB_MNBT:-8192}" \
  --gpu-memory-utilization 0.90 --max-num-seqs "${SB_MNS:-16}" \
  --reasoning-parser qwen3
