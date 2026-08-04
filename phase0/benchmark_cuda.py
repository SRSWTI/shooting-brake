#!/usr/bin/env python3
"""
Shooting Brake Phase 0 — All-CUDA vLLM Baseline Benchmark

Runs the fixed correctness prompts and performance workload from phase0/
against the NVFP4 model on the RTX 5090.

Usage:
    source .venv/bin/activate
    python phase0/benchmark_cuda.py --mode eager
    python phase0/benchmark_cuda.py --mode graph
"""

import argparse
import json
import time
import sys
import os
from pathlib import Path

# ─── Config ────────────────────────────────────────────────────────────────
MODEL = "unsloth/Qwen3.6-35B-A3B-NVFP4"
MAX_MODEL_LEN = 8192
GPU_MEM_UTIL = 0.90
MAX_NUM_SEQS = 64  # limited by Mamba cache blocks on 32 GiB VRAM

PHASE0_DIR = Path(__file__).parent
PROMPTS_FILE = PHASE0_DIR / "correctness_prompts.jsonl"


def load_correctness_prompts():
    prompts = []
    with open(PROMPTS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def check_correctness(output_text, expected_contains):
    """Check if output contains expected substrings (case-insensitive)."""
    text_lower = output_text.lower()
    for expected in expected_contains:
        if expected.lower() not in text_lower:
            return False, expected
    return True, None


def run_correctness(llm, sampling):
    """Run the fixed correctness prompt set."""
    prompts = load_correctness_prompts()
    print(f"\n{'='*60}")
    print(f"CORRECTNESS CHECK ({len(prompts)} prompts)")
    print(f"{'='*60}")

    results = []
    all_pass = True
    for p in prompts:
        sp = sampling.clone()
        sp.max_tokens = p["max_tokens"]
        outputs = llm.generate([p["prompt"]], sp, use_tqdm=False)
        text = outputs[0].outputs[0].text

        passed, missing = check_correctness(text, p["expected_contains"])
        status = "PASS" if passed else f"FAIL (missing: {missing})"
        if not passed:
            all_pass = False

        print(f"  [{p['id']}] {status}")
        print(f"    Q: {p['prompt'][:80]}...")
        print(f"    A: {text[:120]}...")
        results.append({"id": p["id"], "passed": passed, "output": text[:200]})

    print(f"\nCorrectness: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return results, all_pass


def run_decode_workload(llm, sampling):
    """Single-request decode: 512 tokens, batch=1."""
    print(f"\n{'='*60}")
    print("DECODE WORKLOAD (batch=1, 512 tokens)")
    print(f"{'='*60}")

    sp = sampling.clone()
    sp.max_tokens = 512

    prompt = ("Write a detailed essay about the history of computing, "
              "from Charles Babbage to modern AI. Cover at least five key milestones.")

    t0 = time.perf_counter()
    outputs = llm.generate([prompt], sp, use_tqdm=False)
    elapsed = time.perf_counter() - t0

    req = outputs[0]
    out = req.outputs[0]
    n_tokens = len(out.token_ids)
    tok_s = n_tokens / elapsed

    # vLLM RequestMetrics: TTFT and per-token timing
    ttft_ms = 0.0
    mean_itl_ms = (elapsed / max(n_tokens, 1)) * 1000
    if hasattr(req, 'metrics') and req.metrics:
        m = req.metrics
        if hasattr(m, 'time_to_first_token') and m.time_to_first_token is not None:
            ttft_ms = m.time_to_first_token * 1000
        elif hasattr(m, 'first_token_time') and hasattr(m, 'arrival_time'):
            if m.first_token_time and m.arrival_time:
                ttft_ms = (m.first_token_time - m.arrival_time) * 1000
        if hasattr(m, 'time_per_output_token') and m.time_per_output_token is not None and n_tokens > 1:
            mean_itl_ms = m.time_per_output_token * 1000
        elif hasattr(m, 'first_token_time') and hasattr(m, 'last_token_time'):
            if m.first_token_time and m.last_token_time and n_tokens > 1:
                mean_itl_ms = ((m.last_token_time - m.first_token_time) / (n_tokens - 1)) * 1000

    print(f"  Output tokens: {n_tokens}")
    print(f"  Throughput:    {tok_s:.2f} tok/s")
    print(f"  Wall time:     {elapsed:.3f} s")
    print(f"  TTFT:          {ttft_ms:.1f} ms")
    print(f"  Mean ITL:      {mean_itl_ms:.1f} ms")
    print(f"  Output preview: {out.text[:150]}...")

    return {
        "workload": "decode",
        "output_tokens": n_tokens,
        "throughput_tok_s": round(tok_s, 2),
        "wall_time_s": round(elapsed, 3),
        "ttft_ms": round(ttft_ms, 1),
        "mean_itl_ms": round(mean_itl_ms, 1),
    }


def run_batched_decode(llm, sampling):
    """8 concurrent requests, 256 tokens each."""
    print(f"\n{'='*60}")
    print("BATCHED DECODE WORKLOAD (8 requests, 256 tokens each)")
    print(f"{'='*60}")

    sp = sampling.clone()
    sp.max_tokens = 256

    prompts = [
        "Explain how a transformer neural network processes text, step by step.",
        "Describe the water cycle in detail.",
        "Write a short story about a robot learning to paint.",
        "What are the key differences between renewable and non-renewable energy?",
        "Explain the concept of recursion in programming with an example.",
        "Describe the process of photosynthesis.",
        "What caused the fall of the Roman Empire?",
        "Explain quantum entanglement for a general audience.",
    ]

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sp, use_tqdm=False)
    elapsed = time.perf_counter() - t0

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    agg_tok_s = total_tokens / elapsed

    # Collect per-request metrics from RequestOutput (not CompletionOutput)
    ttfts = []
    itls = []
    for req in outputs:
        out = req.outputs[0]
        n = len(out.token_ids)
        if hasattr(req, 'metrics') and req.metrics:
            m = req.metrics
            ttft = 0.0
            if hasattr(m, 'time_to_first_token') and m.time_to_first_token is not None:
                ttft = m.time_to_first_token * 1000
            elif hasattr(m, 'first_token_time') and hasattr(m, 'arrival_time'):
                if m.first_token_time and m.arrival_time:
                    ttft = (m.first_token_time - m.arrival_time) * 1000
            ttfts.append(ttft)
            if hasattr(m, 'time_per_output_token') and m.time_per_output_token is not None and n > 1:
                itls.append(m.time_per_output_token * 1000)
            elif hasattr(m, 'first_token_time') and hasattr(m, 'last_token_time'):
                if m.first_token_time and m.last_token_time and n > 1:
                    itls.append(((m.last_token_time - m.first_token_time) / (n - 1)) * 1000)
            else:
                ttfts.append(0.0)
        else:
            ttfts.append(0.0)

    mean_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0
    mean_itl_val = sum(itls) / len(itls) if itls else 0.0
    p99_itl_val = sorted(itls)[int(len(itls) * 0.99)] if len(itls) > 0 else 0.0

    print(f"  Total output tokens: {total_tokens}")
    print(f"  Aggregate throughput: {agg_tok_s:.2f} tok/s")
    print(f"  Wall time: {elapsed:.3f} s")
    print(f"  Mean TTFT: {mean_ttft:.1f} ms")
    print(f"  Mean ITL:  {mean_itl_val:.1f} ms")
    print(f"  P99 ITL:   {p99_itl_val:.1f} ms")

    return {
        "workload": "batched_decode",
        "total_output_tokens": total_tokens,
        "aggregate_throughput_tok_s": round(agg_tok_s, 2),
        "wall_time_s": round(elapsed, 3),
        "mean_ttft_ms": round(mean_ttft, 1),
        "mean_itl_ms": round(mean_itl_val, 1),
        "p99_itl_ms": round(p99_itl_val, 1),
    }


def run_prefill_workload(llm, sampling):
    """Long-prompt prefill benchmark."""
    print(f"\n{'='*60}")
    print("PREFILL WORKLOAD (long prompt, 128 output tokens)")
    print(f"{'='*60}")

    sp = sampling.clone()
    sp.max_tokens = 128

    prompt = """Summarize the following text in one paragraph:

The Industrial Revolution was a period of major industrialization and innovation that took place during the late 1700s and early 1800s. The Industrial Revolution began in Great Britain and quickly spread throughout the world. The use of new basic materials, primarily iron and steel, new energy sources such as coal and the steam engine, new machines like the spinning jenny and the power loom that increased production, the factory system, and new transportation technologies like railways and steamboats were all key developments. These changes had profound effects on social and cultural conditions, leading to urbanization as people moved from farms to cities, changes in family structure, and the emergence of new social classes. The Industrial Revolution marked a major turning point in history; almost every aspect of daily life was influenced in some way. In particular, average income and population began to exhibit unprecedented sustained growth. In the words of Nobel Prize winner Robert E. Lucas Jr., 'For the first time in history, the living standards of the masses of ordinary people have begun to undergo sustained growth. Nothing remotely like this economic behavior has happened before.'"""

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    input_ids = tok.encode(prompt)
    n_input = len(input_ids)

    t0 = time.perf_counter()
    outputs = llm.generate([prompt], sp, use_tqdm=False)
    elapsed = time.perf_counter() - t0

    req = outputs[0]
    out = req.outputs[0]
    n_output = len(out.token_ids)

    # TTFT approximates prefill time
    ttft = 0.0
    if hasattr(req, 'metrics') and req.metrics:
        m = req.metrics
        if hasattr(m, 'time_to_first_token') and m.time_to_first_token is not None:
            ttft = m.time_to_first_token * 1000
        elif hasattr(m, 'first_token_time') and hasattr(m, 'arrival_time'):
            if m.first_token_time and m.arrival_time:
                ttft = (m.first_token_time - m.arrival_time) * 1000

    prefill_tok_s = n_input / (ttft / 1000) if ttft > 0 else 0

    print(f"  Input tokens:  {n_input}")
    print(f"  Output tokens: {n_output}")
    print(f"  TTFT (prefill): {ttft:.1f} ms")
    print(f"  Prefill throughput: {prefill_tok_s:.0f} tok/s")
    print(f"  Output: {out.text[:150]}...")

    return {
        "workload": "prefill",
        "input_tokens": n_input,
        "output_tokens": n_output,
        "ttft_ms": round(ttft, 1),
        "prefill_throughput_tok_s": round(prefill_tok_s, 0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["eager", "graph"], default="eager")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    enforce_eager = (args.mode == "eager")

    print(f"Shooting Brake Phase 0 — CUDA Baseline")
    print(f"Model: {args.model}")
    print(f"Mode:  {args.mode} (enforce_eager={enforce_eager})")
    print(f"VRAM:  RTX 5090 32 GiB")
    print()

    from vllm import LLM, SamplingParams

    print("Loading model...")
    t0 = time.perf_counter()
    llm = LLM(
        model=args.model,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_num_seqs=MAX_NUM_SEQS,
        enforce_eager=enforce_eager,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.1f}s")

    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1)

    # Warmup
    print("Warming up...")
    llm.generate(["Hello"], sampling.clone(), use_tqdm=False)

    all_results = {
        "model": args.model,
        "mode": args.mode,
        "load_time_s": round(load_time, 1),
        "gpu": "RTX 5090",
    }

    # 1. Correctness
    correctness, all_pass = run_correctness(llm, sampling)
    all_results["correctness"] = {"all_pass": all_pass, "details": correctness}

    # 2. Decode
    all_results["decode"] = run_decode_workload(llm, sampling)

    # 3. Batched decode
    all_results["batched_decode"] = run_batched_decode(llm, sampling)

    # 4. Prefill
    all_results["prefill"] = run_prefill_workload(llm, sampling)

    # Summary
    print(f"\n{'='*60}")
    print("BASELINE SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Mode:  {args.mode}")
    print(f"Correctness: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    d = all_results["decode"]
    b = all_results["batched_decode"]
    p = all_results["prefill"]
    print(f"Decode:     {d['throughput_tok_s']:.2f} tok/s | TTFT {d['ttft_ms']:.0f} ms | ITL {d['mean_itl_ms']:.1f} ms")
    print(f"Batched(8): {b['aggregate_throughput_tok_s']:.2f} tok/s | TTFT {b['mean_ttft_ms']:.0f} ms | ITL {b['mean_itl_ms']:.1f} ms")
    print(f"Prefill:    {p['prefill_throughput_tok_s']:.0f} tok/s | TTFT {p['ttft_ms']:.0f} ms")

    # Save
    outfile = PHASE0_DIR / f"baseline_cuda_{args.mode}.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {outfile}")



if __name__ == "__main__":
    main()
