#!/usr/bin/env bash
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
export SHOOTING_BRAKE_B70_MAX_BATCH=256
export SHOOTING_BRAKE_B70_BANK="$PWD/src/phase1/expert_bank_int4.bin"
export SHOOTING_BRAKE_B70_LIB="$PWD/src/phase7/libsb_b70_provider.so"
unset SHOOTING_BRAKE_B70_PROFILE VLLM_USE_BREAKABLE_CUDAGRAPH
echo "=== launching $(date '+%H:%M:%S')"
exec .venv/bin/vllm serve srswti/axe-superveloce-88b-nvfp4a16 \
  --served-model-name shooting-brake-88b \
  --host 127.0.0.1 --port 8016 \
  --trust-remote-code --language-model-only \
  --max-model-len 32768 --max-num-batched-tokens 256 \
  --gpu-memory-utilization 0.90 --max-num-seqs 4 \
  --reasoning-parser qwen3
