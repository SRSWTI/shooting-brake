#!/usr/bin/env python3
"""Smoke decode benchmark: all-cuda vs hybrid+B70+surgery.

Mirrors the Phase 0 benchmark_cuda.py methodology:
  - Single decode (batch=1, 512 tokens): per-token speed, TTFT, ITL
  - Batched decode (8 requests, 256 tokens): aggregate throughput

Usage:
  # all-cuda
  python src/phase8/smoke_benchmark.py

  # hybrid+surgery
  SHOOTING_BRAKE_PLACEMENT=split:128 \
  SHOOTING_BRAKE_HYBRID=1 SHOOTING_BRAKE_B70_DEVICE=1 \
  SHOOTING_BRAKE_VRAM_SURGERY=1 \
  python src/phase8/smoke_benchmark.py
"""

import json
import os
import time

from vllm import LLM, SamplingParams
from vllm.plugins import load_general_plugins

load_general_plugins()

MODEL = "unsloth/Qwen3.6-35B-A3B-NVFP4"
MAX_MODEL_LEN = 8192
GPU_MEM_UTIL = 0.90
MAX_NUM_SEQS = 64

DECODE_PROMPT = (
    "Write a detailed essay about the history of computing, "
    "from Charles Babbage to modern AI. Cover at least five key milestones."
)

BATCH_PROMPTS = [
    "Explain how a transformer neural network processes text, step by step.",
    "Describe the water cycle in detail.",
    "Write a short story about a robot learning to paint.",
    "What are the key differences between renewable and non-renewable energy?",
    "Explain the concept of recursion in programming with an example.",
    "Describe the process of photosynthesis.",
    "What caused the fall of the Roman Empire?",
    "Explain quantum entanglement for a general audience.",
]


def main():
    llm = LLM(
        model=MODEL, enforce_eager=True, tensor_parallel_size=1,
        pipeline_parallel_size=1, gpu_memory_utilization=GPU_MEM_UTIL,
        max_model_len=MAX_MODEL_LEN, max_num_seqs=MAX_NUM_SEQS,
    )

    sampling = SamplingParams(temperature=0.0)

    # --- Warmup (also triggers VRAM surgery on first forward) ---
    print("Warmup...")
    llm.generate(["Hello"], SamplingParams(temperature=0.0, max_tokens=4),
                 use_tqdm=False)

    # --- Single decode ---
    print(f"\n{'='*60}")
    print("DECODE (batch=1, 512 tokens)")
    print(f"{'='*60}")

    sp = sampling.clone()
    sp.max_tokens = 512
    sp.ignore_eos = True

    t0 = time.perf_counter()
    outputs = llm.generate([DECODE_PROMPT], sp, use_tqdm=False)
    elapsed = time.perf_counter() - t0

    out = outputs[0].outputs[0]
    n = len(out.token_ids)
    tok_s = n / elapsed
    itl_ms = (elapsed / max(n, 1)) * 1000

    print(f"  Tokens:     {n}")
    print(f"  Throughput: {tok_s:.2f} tok/s")
    print(f"  Wall:       {elapsed:.3f} s")
    print(f"  Mean ITL:   {itl_ms:.1f} ms")

    # --- Batched decode ---
    print(f"\n{'='*60}")
    print("BATCHED DECODE (8 requests, 256 tokens)")
    print(f"{'='*60}")

    sp2 = sampling.clone()
    sp2.max_tokens = 256
    sp2.ignore_eos = True

    t0 = time.perf_counter()
    outputs = llm.generate(BATCH_PROMPTS, sp2, use_tqdm=False)
    elapsed = time.perf_counter() - t0

    total = sum(len(o.outputs[0].token_ids) for o in outputs)
    agg = total / elapsed

    print(f"  Total tokens:     {total}")
    print(f"  Aggregate:        {agg:.2f} tok/s")
    print(f"  Wall:             {elapsed:.3f} s")

    # --- Config summary ---
    placement = os.environ.get("SHOOTING_BRAKE_PLACEMENT", "all-cuda")
    surgery = os.environ.get("SHOOTING_BRAKE_VRAM_SURGERY", "0")
    print(f"\n{'='*60}")
    print(f"Config: placement={placement} surgery={surgery}")


if __name__ == "__main__":
    main()
