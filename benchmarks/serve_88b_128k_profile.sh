#!/usr/bin/env bash
# Profiling variant of serve_88b_128k.sh — for ATTRIBUTION ONLY, never for
# performance numbers: B70 queue profiling adds two marker submissions per
# dispatch and the torch profiler adds its own overhead (a prior session
# measured profiling markers moving decode 63 -> 37.5 tok/s). Wall-clock truth
# comes from the unprofiled Grid B run (TTFT @ 8K = 21.86 s).
#
# Deltas from serve_88b_128k.sh:
#   SHOOTING_BRAKE_B70_PROFILE=1   provider queue in profiling mode ->
#                                  sb_b70_poll_kernel_ns is real; poller logs
#                                  totals on stop
#   --profiler-config.*            this vLLM gates /start_profile behind
#                                  ProfilerConfig (config/profiler.py), NOT the
#                                  old VLLM_TORCH_PROFILER_DIR env var. Stack
#                                  tracing off: it multiplies trace size and
#                                  distorts an 8K prefill.
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
export SHOOTING_BRAKE_B70_PROFILE=1
export SB_TRACE_DIR="$PWD/benchmarks/results/prefill_profile/traces"
mkdir -p "$SB_TRACE_DIR"
unset VLLM_USE_BREAKABLE_CUDAGRAPH
echo "=== launching 128K PROFILE $(date '+%H:%M:%S')"
exec .venv/bin/vllm serve srswti/axe-superveloce-88b-nvfp4a16 \
  --served-model-name shooting-brake-88b \
  --host 127.0.0.1 --port 8016 \
  --trust-remote-code --language-model-only \
  --max-model-len 131072 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 --max-num-seqs 64 \
  --reasoning-parser qwen3 \
  --profiler-config.profiler=torch \
  --profiler-config.torch_profiler_dir="$SB_TRACE_DIR" \
  --profiler-config.torch_profiler_with_stack=false
