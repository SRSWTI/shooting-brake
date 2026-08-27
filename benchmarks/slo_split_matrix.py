#!/usr/bin/env python3
"""SLO matrix for an A/B over server configuration.

Sweeps context x concurrency, issuing N requests per cell against a live
OpenAI-compatible endpoint, and records the metrics an SLO is actually written
against: TTFT, inter-token latency percentiles, end-to-end latency, and
per-cell throughput.

Every request carries a unique prefix so nothing is served from the prefix
cache. That is deliberate: prefill is the wire-bound phase, and a warm cache
would hide exactly the effect under test.

Usage:
  benchmarks/slo_split_matrix.py --label split_95_75 \
      --json-out benchmarks/results/split_ab/split_95_75.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from pathlib import Path

import aiohttp


def build_prompt(corpus: str, target_tokens: int, tokenizer, nonce: str) -> str:
    """Slice the corpus to approximately `target_tokens`, prefixed by a nonce."""
    head = f"[{nonce}] "
    # Start from a generous character estimate, then trim by token count.
    approx = corpus[: max(1, target_tokens * 6)]
    ids = tokenizer(head + approx, add_special_tokens=False)["input_ids"]
    if len(ids) < target_tokens:
        reps = (target_tokens // max(1, len(ids))) + 2
        approx = (corpus * reps)[: target_tokens * 8]
        ids = tokenizer(head + approx, add_special_tokens=False)["input_ids"]
    ids = ids[:target_tokens]
    return tokenizer.decode(ids)


async def one_request(session, url, model, prompt, max_tokens, timeout):
    """Stream one completion, returning TTFT / ITLs / e2e."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    stamps: list[float] = []
    start = time.perf_counter()
    text_len = 0
    try:
        async with session.post(
            f"{url}/v1/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            if resp.status != 200:
                return {"ok": False, "error": f"HTTP {resp.status}: {(await resp.text())[:200]}"}
            async for raw in resp.content:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                try:
                    chunk = json.loads(body)
                except json.JSONDecodeError:
                    continue
                piece = chunk.get("choices", [{}])[0].get("text", "")
                if piece:
                    stamps.append(time.perf_counter())
                    text_len += len(piece)
    except asyncio.TimeoutError:
        return {"ok": False, "error": f"timeout after {timeout}s"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    if not stamps:
        return {"ok": False, "error": "no tokens streamed"}

    ttft = stamps[0] - start
    itls = [b - a for a, b in zip(stamps, stamps[1:])]
    return {
        "ok": True,
        "ttft_s": ttft,
        "e2e_s": stamps[-1] - start,
        "tokens": len(stamps),
        "chars": text_len,
        "itl_p50_ms": 1e3 * statistics.median(itls) if itls else None,
        "itl_p99_ms": 1e3 * (sorted(itls)[int(0.99 * (len(itls) - 1))] if itls else 0.0)
        if itls
        else None,
        "tpot_ms": 1e3 * statistics.fmean(itls) if itls else None,
    }


async def run_cell(url, model, prompts, max_tokens, timeout):
    """Issue all prompts for one cell concurrently."""
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn) as session:
        t0 = time.perf_counter()
        results = await asyncio.gather(
            *(one_request(session, url, model, p, max_tokens, timeout) for p in prompts)
        )
        wall = time.perf_counter() - t0
    return results, wall


def summarize(results, wall):
    ok = [r for r in results if r.get("ok")]
    if not ok:
        return {"ok": 0, "failed": len(results),
                "errors": [r.get("error") for r in results][:3]}
    tok = sum(r["tokens"] for r in ok)
    return {
        "ok": len(ok),
        "failed": len(results) - len(ok),
        "errors": [r.get("error") for r in results if not r.get("ok")][:3],
        "ttft_s_min": min(r["ttft_s"] for r in ok),
        "ttft_s_mean": statistics.fmean(r["ttft_s"] for r in ok),
        "ttft_s_max": max(r["ttft_s"] for r in ok),
        "tpot_ms_mean": statistics.fmean(r["tpot_ms"] for r in ok if r["tpot_ms"]),
        "itl_p50_ms": statistics.fmean(r["itl_p50_ms"] for r in ok if r["itl_p50_ms"]),
        "itl_p99_ms": max(r["itl_p99_ms"] for r in ok if r["itl_p99_ms"]),
        "e2e_s_mean": statistics.fmean(r["e2e_s"] for r in ok),
        "tokens_total": tok,
        "wall_s": wall,
        "cell_tok_per_s": tok / wall if wall > 0 else None,
    }


async def main_async(a) -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    corpus = Path(a.corpus).read_text(errors="replace")
    rng = random.Random(a.seed)

    contexts = [int(x) for x in a.contexts.split(",") if x.strip()]
    concurrencies = [int(x) for x in a.concurrency.split(",") if x.strip()]

    out = {
        "label": a.label,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "url": a.url,
        "model": a.model,
        "requests_per_cell": a.requests,
        "max_tokens": a.max_tokens,
        "cells": [],
    }

    for ctx in contexts:
        for conc in concurrencies:
            # `requests` rounds, each issuing `conc` genuinely-concurrent
            # requests. Every prompt is unique so no round is served warm.
            print(f"  ctx={ctx:<7} conc={conc:<2} rounds={a.requests} ...",
                  end="", flush=True)
            results: list[dict] = []
            wall = 0.0
            for rnd in range(a.requests):
                prompts = [
                    build_prompt(
                        corpus, ctx, tokenizer,
                        f"{a.label}-{ctx}-{conc}-r{rnd}-s{i}-{rng.randrange(1 << 30)}")
                    for i in range(conc)
                ]
                got, w = await run_cell(a.url, a.model, prompts, a.max_tokens,
                                        a.timeout)
                results.extend(got)
                wall += w
            summary = summarize(results, wall)
            summary.update({"context": ctx, "concurrency": conc,
                            "rounds": a.requests})
            out["cells"].append(summary)
            if summary["ok"]:
                print(f" ttft {summary['ttft_s_mean']:.3f}s  tpot {summary['tpot_ms_mean']:.1f}ms"
                      f"  itl_p99 {summary['itl_p99_ms']:.0f}ms  {summary['cell_tok_per_s']:.1f} tok/s")
            else:
                print(f" FAILED  {summary.get('errors')}")
    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json_out).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {a.json_out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8017")
    ap.add_argument("--model", default="shooting-brake-jota-r15")
    ap.add_argument("--tokenizer", default="srswti/axe-superveloce-jota-118b-r15-nvfp4")
    ap.add_argument("--corpus", default=str(Path.home() / "sb_corpus_big.txt"))
    ap.add_argument("--contexts", default="1024,8192,32768")
    ap.add_argument("--concurrency", default="1,2,4")
    ap.add_argument("--requests", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--label", default="run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()
    return asyncio.run(main_async(a))


if __name__ == "__main__":
    sys.exit(main())
