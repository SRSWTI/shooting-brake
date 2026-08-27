#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
CAPTURE_DIR="${SB_DRAFTER_CAPTURE_DIR:-$HOME/sb_hidden_capture}"
WORK_DIR="${SB_DRAFTER_WORK_DIR:-$ROOT/experiments/drafter_train/work}"
WARM_SOURCE="${SB_DRAFTER_WARM_START:-poolside/Laguna-S-2.1-DFlash}"
CONFIG="${SB_DRAFTER_CONFIG:-$ROOT/experiments/drafter_train/pilot.yaml}"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python environment not found or not executable: $PYTHON" >&2
  exit 2
fi
if [[ -z "$WARM_SOURCE" ]]; then
  echo "ERROR: SB_DRAFTER_WARM_START must name the required poolside warm start or its local mirror." >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  echo "ERROR: training is restricted to one RTX 5090; CUDA_VISIBLE_DEVICES must name exactly one device." >&2
  exit 2
fi

# The first GPU smoke OOM'd with 2.93 GiB reserved-but-unallocated: the
# DFlash objective allocates chunk-sized logits repeatedly, which fragments
# the caching allocator. Expandable segments reclaim that band.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if pgrep -f 'vllm serve|VLLM::EngineCore' >/dev/null; then
  echo "ERROR: stop the serving stack before training (vLLM is still running)." >&2
  exit 2
fi
# Exclusivity: refuse only on REAL competitors. GNOME desktop residents
# (gnome-remote-desktop ~504 MiB, ptyxis, control-center) always hold small
# compute contexts on this box, so "zero compute apps" is unachievable --
# that literal check would have refused every unattended run (2026-08-25).
if pgrep -f 'vllm serve|VLLM::EngineCore' >/dev/null 2>&1; then
  echo "ERROR: vLLM serving is still running; stop it before training." >&2
  exit 2
fi
big_consumers="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits \
  | awk -F', *' '$2 > 1024 {print $1}' | wc -l)"
if [[ "$big_consumers" -ne 0 ]]; then
  echo "ERROR: a process holds >1 GiB on the RTX 5090; training requires the card." >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2
  exit 2
fi

mkdir -p "$WORK_DIR"
export PYTHONPATH="$ROOT/experiments/drafter_train:$ROOT/vendor/SpecForge${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "[1/4] validating and staging capture records"
"$PYTHON" experiments/drafter_train/adapter.py \
  --input "$CAPTURE_DIR" --output "$WORK_DIR/features"

if [[ ! -s "$WORK_DIR/warmstart/config.json" ]] || \
   [[ ! -s "$WORK_DIR/warmstart/model.safetensors" && \
      ! -s "$WORK_DIR/warmstart/model.safetensors.index.json" ]]; then
  echo "[2/4] converting the required poolside warm start from the offline cache"
  "$PYTHON" experiments/drafter_train/prepare_warmstart.py \
    --source "$WARM_SOURCE" --output "$WORK_DIR/warmstart"
else
  echo "[2/4] reusing converted warm start at $WORK_DIR/warmstart"
fi

TRAIN_OVERRIDES=(
  "data.hidden_states_path=$WORK_DIR/features"
  "model.draft_model_config=$WORK_DIR/warmstart/config.json"
  "data.cache_dir=$WORK_DIR/cache"
  "output_dir=$WORK_DIR/checkpoints"
)
if [[ -s "$WORK_DIR/checkpoints/jota-r15-dflash-latest/training_state.pt" ]]; then
  echo "[3/4] resuming one-GPU Laguna DFlash from the latest complete checkpoint"
  # Overrides are STRING-typed in SpecForge (schema.py apply_overrides), so
  # "=null" cannot unset a field: pilot.yaml keeps both null and each branch
  # sets exactly the one it means.
  TRAIN_OVERRIDES+=(
    "training.resume_from=$WORK_DIR/checkpoints"
  )
else
  echo "[3/4] training one-GPU Laguna DFlash from the mandatory warm start"
  TRAIN_OVERRIDES+=(
    "model.draft_checkpoint_path=$WORK_DIR/warmstart"
  )
fi
"$PYTHON" experiments/drafter_train/train_runner.py train --config "$CONFIG" \
  "${TRAIN_OVERRIDES[@]}"

echo "[4/4] exporting the latest checkpoint to native vLLM Laguna layout"
"$PYTHON" experiments/drafter_train/export_laguna.py \
  --checkpoint "$WORK_DIR/checkpoints" \
  --warm-source "$WARM_SOURCE" \
  --output "$WORK_DIR/export"

echo "DONE: $WORK_DIR/export"
