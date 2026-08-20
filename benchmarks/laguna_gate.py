#!/usr/bin/env python3
"""Architecture gate for LagunaForCausalLM NVFP4, with ZERO Shooting Brake involvement.

Answers four questions that do not require any plugin work, in order of value:

  1. Does the checkpoint load in stock vLLM at all?  (validates NVFP4
     compressed-tensors loading, `gating="per-head"` g_proj sizing at
     [heads, hidden/2], the 47-sparse + 1-dense MLP schedule, and the clean
     `model.layers.N.mlp.experts.N.*` naming.)
  2. Does it produce coherent, finite-logprob greedy output?
  3. Is prefix caching actually ENABLED?  The 99B reports
     `enable_prefix_caching=False` because `Qwen3_5MoeForCausalLM` is
     `is_hybrid=True` with `supports_mamba_prefix_caching=False`.
     `LagunaForCausalLM` is not `IsHybrid`, so it should report True.
  4. What is the measured cold-vs-warm TTFT ratio on an identical long
     prompt?  This is the deciding number for adopting Laguna: on the 99B
     the same 5,524-token prompt three times cost 9.788 / 9.729 / 9.723 s
     -- zero amortisation, structurally.

Routed experts run on the 5090 here via CPU offload, NOT on the B70s, so the
absolute latencies are meaningless and deliberately not reported as such.
The cold/warm RATIO is valid because both arms traverse the identical slow
path -- that is the whole point of measuring a ratio rather than a wall time.

Usage:
  .venv/bin/python benchmarks/laguna_gate.py \
      --model srswti/axe-superveloce-jota-118b-r20-nvfp4 \
      --cpu-offload-gib 26 --prompt-tokens 4096 --json-out /tmp/laguna_gate.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path


def build_prompt(corpus: Path | None, target_tokens: int) -> str:
    """Prefix-cacheable prose. Real text, not repeated tokens: a repeated word
    tokenises differently by count and silently perturbs prefix boundaries --
    a bug that already contaminated one prefill ladder in this repo."""
    if corpus is not None and corpus.exists():
        words = corpus.read_text(errors="ignore").split()
    else:
        words = []
        for pat in ("docs/*.md", "README.md", "benchmarks/*.py"):
            for f in sorted(Path(".").glob(pat)):
                words.extend(f.read_text(errors="ignore").split())
                if len(words) > target_tokens * 2:
                    break
            if len(words) > target_tokens * 2:
                break
    if not words:
        raise SystemExit("no corpus text found; pass --corpus")
    # ~0.75 words/token for prose; overshoot then let the caller report actual
    return " ".join(words[: max(1, int(target_tokens * 0.78))])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cpu-offload-gib", type=float, default=26.0)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--prompt-tokens", type=int, default=4096)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--corpus", type=Path, default=None)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    result: dict = {"model": args.model, "checks": {}}

    t0 = time.perf_counter()
    llm = LLM(
        model=args.model,
        enforce_eager=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.max_model_len,
        cpu_offload_gb=args.cpu_offload_gib,
        trust_remote_code=True,
        moe_backend="cutlass",
        enable_prefix_caching=True,
    )
    result["boot_s"] = round(time.perf_counter() - t0, 2)
    result["checks"]["loads"] = True
    print(f"=== LOADED in {result['boot_s']}s ===", flush=True)

    # --- Q3: is prefix caching on? read it from the live config, not the log ---
    cfg = llm.llm_engine.vllm_config
    result["enable_prefix_caching"] = bool(cfg.cache_config.enable_prefix_caching)
    result["is_hybrid"] = bool(cfg.model_config.is_hybrid)
    result["sliding_window"] = cfg.model_config.get_sliding_window()
    result["checks"]["prefix_caching_enabled"] = result["enable_prefix_caching"]
    print(f"prefix_caching={result['enable_prefix_caching']} "
          f"is_hybrid={result['is_hybrid']} window={result['sliding_window']}", flush=True)

    # --- Q2: correctness on short deterministic prompts ---
    gate = llm.generate(
        ["The integer after 41 is", "Water is composed of hydrogen and"],
        SamplingParams(temperature=0.0, max_tokens=16, logprobs=1),
        use_tqdm=False,
    )
    texts, finite = [], True
    for o in gate:
        c = o.outputs[0]
        texts.append(c.text)
        finite &= all(math.isfinite(next(iter(d.values())).logprob) for d in c.logprobs)
    result["gate_texts"] = texts
    result["checks"]["logprobs_finite"] = finite
    result["checks"]["says_42"] = "42" in texts[0]
    for t in texts:
        print("TEXT:", repr(t), flush=True)

    # --- Q4: cold vs warm on an IDENTICAL long prompt ---
    prompt = build_prompt(args.corpus, args.prompt_tokens)
    sp = SamplingParams(temperature=0.0, max_tokens=1)  # max_tokens=1 -> TTFT proxy
    runs = []
    for i in range(args.repeats):
        t = time.perf_counter()
        out = llm.generate([prompt], sp, use_tqdm=False)
        dt = time.perf_counter() - t
        runs.append({"run": i, "wall_s": round(dt, 4),
                     "prompt_tokens": len(out[0].prompt_token_ids)})
        print(f"run {i}: {dt:.4f}s  ptok={runs[-1]['prompt_tokens']}", flush=True)
    result["repeat_runs"] = runs
    cold, warm = runs[0]["wall_s"], min(r["wall_s"] for r in runs[1:]) if len(runs) > 1 else None
    if warm:
        result["cold_s"], result["warm_s"] = cold, warm
        result["speedup"] = round(cold / warm, 3)
        # The 99B measured 1.007x here (9.788 -> 9.723). Anything >1.5x is a
        # real cache hit; >5x means the prefill is genuinely being skipped.
        result["checks"]["prefix_cache_effective"] = (cold / warm) > 1.5
        print(f"cold={cold:.4f}s warm={warm:.4f}s speedup={result['speedup']}x", flush=True)

    failed = [k for k, v in result["checks"].items() if not v]
    result["verdict"] = "PASS" if not failed else f"FAIL: {failed}"
    print(f"=== {result['verdict']} ===", flush=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2))
        print(f"wrote {args.json_out}", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
