#!/bin/bash
# Shooting Brake Phase 0 — CUDA benchmark environment
# Source this before running benchmark_cuda.py
# Usage: source src/phase0/env.sh && python src/phase0/benchmark_cuda.py --mode graph

# Limit parallel compilation to avoid OOM during JIT (nvcc/CUTLASS kernels are RAM-heavy)
export MAX_JOBS=16
export NVCC_THREADS=1
export CMAKE_BUILD_PARALLEL_LEVEL=16
export TORCHINDUCTOR_COMPILE_THREADS=4

# Model path (HF cache auto-resolves)
export MODEL="unsloth/Qwen3.6-35B-A3B-NVFP4"
