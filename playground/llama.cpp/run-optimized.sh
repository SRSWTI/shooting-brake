#!/usr/bin/env bash
# Recommended llama-server runner for this box (4x Xeon 6252N + RTX 3090).
# Branch: multisocket-cpu-moe. See README-OPTIMIZATIONS.md for every choice below.
#
# Serving shape: single user + agent fanout -> config D from PARALLEL-MATRIX.md:
#   --parallel 4 --kv-unified -c 131072  (4 concurrent streams, one shared KV pool;
#   batch-4 aggregate 42.8 tok/s, single-stream ~24 tok/s, prefix cache survives
#   slot reassignment).
#
# MTP speculative decoding (-md sidecar) is NOT enabled: the 10.4 GB draft + its
# 2.4 GB compute buffer do not fit VRAM at this context/parallel config
# (measured; see README-OPTIMIZATIONS.md section 3 for the math and options).
set -euo pipefail

BIN=${BIN:-/home/pleb/llama.cpp/build-publish/bin/llama-server}
MODEL=${MODEL:-/home/pleb/models/DeepSeek-V4-Flash-0731-ggmlorg/DeepSeek-V4-Flash-0731-MXFP4.gguf}

# --- accepted optimization envs (each one measured; see README journey table) ---
export GGML_EXPERT_SHARD=1          # expert-home NUMA sharding + fastload path
export GGML_EXPERT_REPLICAS=1       # two-home replication (+147 GB RAM, +10-14% decode)
export GGML_EXPERT_PIN=1            # cudaHostRegister primaries after NUMA placement
export GGML_CUDA_REGISTER_HOST=1    # required for the register wrapper to act
export GGML_OP_OFFLOAD_MIN_BATCH=2048  # sub-2048-tok ubatches stay on CPU (TTFT 29s -> 7.7s)
# GGML_PIN_CORE defaults on; GGML_FASTLOAD_THREADS defaults 16.

# numa_balancing fights explicit placement during load; harmless in steady decode.
if [ "$(cat /proc/sys/kernel/numa_balancing 2>/dev/null)" = "1" ]; then
    echo "note: kernel.numa_balancing is enabled; consider: sudo sysctl kernel.numa_balancing=0" >&2
fi

exec numactl --interleave=all "$BIN" \
    -m "$MODEL" \
    --alias deepseek-v4-flash \
    --api-key 59e4d4cd30efa68f69a5bfbd66f8293c \
    --host 0.0.0.0 --port 8080 \
    -ngl 99 --n-cpu-moe 43 \
    --numa distribute -t 96 -tb 96 \
    --no-mmap \
    -c 131072 -b 4096 -ub 4096 \
    --parallel 4 --kv-unified \
    "$@"
