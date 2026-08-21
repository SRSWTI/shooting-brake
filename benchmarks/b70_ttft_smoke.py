#!/usr/bin/env python3
"""Cold/warm TTFT smoke across 1K..32K against a live server.

The standing acceptance check for a prefill change. Every number here is
measured on one clock, through HTTP, on exact token counts -- prompts are
binary-searched against the real tokenizer rather than estimated, because a
loose target silently changes the workload between runs and invalidates the
comparison (that defect cost us Bench 20's per-step wobble).

Cold vs warm is the whole point: a unique random header per prompt breaks the
block hash chain, so the cold pass genuinely misses. The same string sent twice
is a full prefix-cache hit. Reporting only one of the two hides which half of
the system moved.

Usage:
  benchmarks/b70_ttft_smoke.py --label mb2048 --json-out results.json
  benchmarks/b70_ttft_smoke.py --contexts 1024,8192 --baseline 1705
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer


def mint(tok, words, target: int, tag: str) -> str:
    """A prompt of EXACTLY target tokens (or the closest reachable below).

    Binary search over a prefix of one fixed word pool. Monotone in prefix
    length, so it converges; an estimate-and-correct loop re-slices different
    corpus regions with different token density and oscillates instead.
    """
    head = f"Doc {tag} ref {random.randrange(1 << 40):#x}. "
    pool = words[: target * 2 + 256]
    if len(pool) < 8:
        raise SystemExit(f"corpus too small for a {target}-token prompt")
    lo, hi, best = 1, len(pool), None
    while lo <= hi:
        mid = (lo + hi) // 2
        text = head + " ".join(pool[:mid])
        n = len(tok(text, add_special_tokens=False)["input_ids"])
        if n == target:
            return text
        if n < target:
            best, lo = text, mid + 1
        else:
            hi = mid - 1
    if best is None:
        raise SystemExit(f"could not reach {target} tokens")
    return best


def ttft(url: str, model: str, prompt: str, timeout: float) -> tuple[float, int]:
    """Wall time to a 1-token completion == prefill + one decode step."""
    t0 = time.perf_counter()
    r = requests.post(
        f"{url}/v1/completions",
        json={"model": model, "prompt": prompt, "temperature": 0.0, "max_tokens": 1},
        timeout=timeout,
    )
    wall = time.perf_counter() - t0
    r.raise_for_status()
    return wall, r.json()["usage"]["prompt_tokens"]


def gate(url: str, model: str, timeout: float) -> dict:
    """Correctness before speed. A fast wrong kernel is worse than no kernel.

    The counting prompt has a forced continuation, so a broken GEMM shows up as
    wrong tokens rather than plausible text -- and NaN logits surface as an
    HTTP 400 ('Out of range float values are not JSON compliant') rather than
    quietly as an early EOS, which is how the fp16 overflow first hid.
    """
    prompt = "Count upward. " + " ".join(f"{i}," for i in range(1, 60)) + " "
    r = requests.post(
        f"{url}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "temperature": 0.0,
            "max_tokens": 12,
            "logprobs": 1,
        },
        timeout=timeout,
    )
    if r.status_code != 200:
        return {"ok": False, "status": r.status_code, "body": r.text[:200]}
    c = r.json()["choices"][0]
    lps = [x for x in c["logprobs"]["token_logprobs"] if x is not None]
    return {
        "ok": bool(c["text"].strip()) and all(math.isfinite(x) for x in lps),
        "status": 200,
        "text": c["text"],
        "logprobs_finite": all(math.isfinite(x) for x in lps),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8017")
    ap.add_argument("--model", default="shooting-brake-jota-r15")
    ap.add_argument("--tokenizer", default="srswti/axe-superveloce-jota-118b-r15-nvfp4")
    ap.add_argument("--corpus", type=Path, default=Path.home() / "sb_corpus_big.txt")
    ap.add_argument("--contexts", default="1024,8192,16384,32768")
    ap.add_argument("--label", default="run")
    ap.add_argument(
        "--baseline",
        type=float,
        default=1705.0,
        help="us/token to compare against; 1705 is the pre-grouped per-route GEMV",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--json-out", type=Path, default=None)
    a = ap.parse_args()

    random.seed(a.seed)
    contexts = [int(x) for x in a.contexts.split(",") if x.strip()]

    g = gate(a.url, a.model, a.timeout)
    print(f"correctness gate: {'PASS' if g['ok'] else 'FAIL'}  {g}")
    if not g["ok"]:
        print("refusing to report speed for an incorrect server")
        return 1

    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    words = a.corpus.read_text(errors="ignore").split()

    print(
        f"\n{'ctx':>7} {'ptok':>7} {'cold':>9} {'us/tok':>8} "
        f"{'vs base':>8} {'warm':>8} {'speedup':>8}"
    )
    cells = []
    for ctx in contexts:
        prompt = mint(tok, words, ctx, f"{a.label}-{ctx}")
        cold_s, ptok = ttft(a.url, a.model, prompt, a.timeout)
        warm_s, _ = ttft(a.url, a.model, prompt, a.timeout)
        us = cold_s / ptok * 1e6
        cells.append(
            {
                "context": ctx,
                "prompt_tokens": ptok,
                "cold_s": cold_s,
                "warm_s": warm_s,
                "us_per_token": us,
                "vs_baseline": a.baseline / us,
                "warm_speedup": cold_s / warm_s if warm_s > 0 else None,
            }
        )
        print(
            f"{ctx:>7} {ptok:>7} {cold_s:>8.3f}s {us:>8.0f} "
            f"{a.baseline / us:>7.2f}x {warm_s:>7.3f}s {cold_s / warm_s:>7.1f}x"
        )

    tot = sum(c["cold_s"] for c in cells)
    tok_tot = sum(c["prompt_tokens"] for c in cells)
    agg_us = tot / tok_tot * 1e6
    print(
        f"\ntotal cold {tot:.1f}s over {tok_tot} tokens -> {agg_us:.0f} us/token "
        f"= {a.baseline / agg_us:.2f}x baseline ({a.baseline:.0f} us/token)"
    )

    if a.json_out:
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        prev = []
        if a.json_out.exists():
            try:
                prev = json.loads(a.json_out.read_text())
            except json.JSONDecodeError:
                prev = []
        prev.append(
            {
                "label": a.label,
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "baseline_us_per_token": a.baseline,
                "gate": g,
                "cells": cells,
                "aggregate_us_per_token": agg_us,
                "aggregate_speedup": a.baseline / agg_us,
            }
        )
        a.json_out.write_text(json.dumps(prev, indent=2) + "\n")
        print(f"appended {a.json_out} ({len(prev)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
