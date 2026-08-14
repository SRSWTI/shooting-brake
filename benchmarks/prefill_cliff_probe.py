#!/usr/bin/env python3
"""Prefill/decode cost across (context length x concurrency), repeated.

Purpose: settle where the Track A TTFT cliff comes from. Track A (live
server, subset:16:8) prefills at ~20k tok/s up to 32,768 prompt tokens and
~2.05k tok/s from 65,536 on -- a 10x step, then linear scaling again.
Track B's in-process arms show no step at all: ~2.3k tok/s at *every*
length including 1,543 tokens. Same placement, same provider, so at most
one of those two is measuring what it claims.

What this runs, for each (length, concurrency) cell, `--trials` times:
  * `concurrency` identical requests issued together
  * TTFT per request (prefill cost), ITL p50/p99 (decode cost),
    aggregate output tok/s
  * B70 poller counters diffed across the cell, so a cell that dispatches
    to the B70 can be distinguished from one that does not

Two deliberate differences from offload_benchmark's context sweep:

  * `ignore_eos=True`. That harness omits it, so greedy decode on a
    repetitive synthetic prompt stops at EOS -- 12 of its 20 context-sweep
    points returned exactly 8 tokens instead of 400, and the derived
    tok/s divides ~7 tokens by a few ms of streaming, which is how an arm
    routing 96% of its experts over PCIe reported 387 tok/s. Forcing the
    full token budget makes the decode numbers mean something.
  * No batched/frontier phases. They dominate runtime and answer a
    different question.

Usage:
  ./.venv/bin/python benchmarks/prefill_cliff_probe.py \
      --config hybrid --placement subset:16:8 --max-model-len 131072 \
      --lengths 8192 32768 65536 --concurrency 1 4 8 --trials 2 \
      --out benchmarks/results/prefill_probe/hybrid-subset-16-8.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from offload_benchmark import (  # noqa: E402
    _context_prompt,
    _pct,
    apply_config_env,
    build_engine,
    worker_stats,
)


@dataclass
class Timing:
    ttft_s: float = 0.0
    total_s: float = 0.0
    output_tokens: int = 0
    prompt_tokens: int = 0
    itls_s: list[float] = field(default_factory=list)


async def stream_one(engine: Any, prompt: str, max_tokens: int) -> Timing:
    """One request, every token timestamped. ignore_eos, so the token
    budget is what was asked for rather than wherever the model stopped."""
    from vllm import SamplingParams

    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    t = Timing()
    start = time.perf_counter()
    previous = start
    seen = 0

    async for output in engine.generate(prompt, params, request_id=str(uuid.uuid4())):
        now = time.perf_counter()
        produced = len(output.outputs[0].token_ids)
        if produced <= seen:
            continue
        if seen == 0:
            t.ttft_s = now - start
        else:
            # A streaming chunk can carry several tokens; charge the gap
            # evenly across them rather than to the last one.
            gap = (now - previous) / (produced - seen)
            t.itls_s.extend([gap] * (produced - seen))
        seen = produced
        previous = now
        t.prompt_tokens = len(output.prompt_token_ids or [])

    t.total_s = time.perf_counter() - start
    t.output_tokens = seen
    return t


def _poller(stats: list[dict[str, Any]]) -> dict[str, float]:
    for w in stats or []:
        p = w.get("poller") or {}
        if p:
            return {
                "dispatches": p.get("dispatches", 0) or 0,
                "service_mean_us": p.get("service_mean_us", 0.0) or 0.0,
                "errors": p.get("errors", 0) or 0,
            }
    return {"dispatches": 0, "service_mean_us": 0.0, "errors": 0}


def _kv_tokens(stats: list[dict[str, Any]]) -> int | None:
    for w in stats or []:
        kv = (w.get("kv_cache") or {}).get("max_tokens")
        if kv:
            return kv
    return None


async def run(args: argparse.Namespace) -> dict[str, Any]:
    engine = build_engine(args.max_num_seqs, args.max_model_len)
    cells: list[dict[str, Any]] = []
    try:
        # Warm up: the first request pays lazy provider init and graph
        # capture, which would otherwise be charged to the first cell.
        await stream_one(engine, "Hello", 4)
        kv = _kv_tokens(await worker_stats(engine))
        print(f"[probe] kv_cache max_tokens={kv:,}" if kv else "[probe] kv unknown",
              flush=True)

        for length in args.lengths:
            prompt = _context_prompt(length)
            for conc in args.concurrency:
                for trial in range(args.trials):
                    before = _poller(await worker_stats(engine))
                    t0 = time.perf_counter()
                    timings = await asyncio.gather(*(
                        stream_one(engine, prompt, args.decode_tokens)
                        for _ in range(conc)
                    ))
                    wall = time.perf_counter() - t0
                    after = _poller(await worker_stats(engine))

                    ttfts = [t.ttft_s for t in timings]
                    itls = [v for t in timings for v in t.itls_s]
                    ptok = timings[0].prompt_tokens
                    out_tok = sum(t.output_tokens for t in timings)
                    disp = after["dispatches"] - before["dispatches"]
                    row = {
                        "target_prompt_tokens": length,
                        "actual_prompt_tokens": ptok,
                        "concurrency": conc,
                        "trial": trial,
                        "wall_s": wall,
                        "ttft_p50_s": _pct(ttfts, 0.50),
                        "ttft_min_s": min(ttfts),
                        "ttft_max_s": max(ttfts),
                        # Per-request prefill rate: the first request's own
                        # prompt over its own TTFT, not the aggregate, so
                        # concurrency does not inflate it.
                        "prefill_tok_per_s_min_ttft": ptok / min(ttfts) if min(ttfts) else 0.0,
                        "us_per_token_min_ttft": min(ttfts) * 1e6 / ptok if ptok else 0.0,
                        "output_tokens_total": out_tok,
                        "output_tok_per_s": out_tok / wall if wall else 0.0,
                        "itl_p50_ms": _pct(itls, 0.50) * 1e3,
                        "itl_p99_ms": _pct(itls, 0.99) * 1e3,
                        "kv_fits": (kv // (ptok + args.decode_tokens)) if kv else None,
                        "b70_dispatches": disp,
                        "b70_dispatches_per_prompt_token": (
                            disp / (ptok * conc) if ptok else 0.0
                        ),
                        "b70_service_mean_us": after["service_mean_us"],
                        "b70_errors": after["errors"] - before["errors"],
                    }
                    cells.append(row)
                    print(
                        f"  {ptok:>7,} tok x{conc:<2} t{trial}: "
                        f"TTFT p50 {row['ttft_p50_s']:>7.2f}s  "
                        f"prefill {row['prefill_tok_per_s_min_ttft']:>7,.0f} tok/s  "
                        f"ITL p50 {row['itl_p50_ms']:>6.2f}ms  "
                        f"out {row['output_tok_per_s']:>7.1f} tok/s  "
                        f"disp {disp:>8,}",
                        flush=True,
                    )
    finally:
        stats = await worker_stats(engine)

    return {
        "config": args.config,
        "placement": args.placement if args.config != "all-cuda" else "all-cuda",
        "prefill_stream": args.prefill_stream,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "decode_tokens": args.decode_tokens,
        "trials": args.trials,
        "kv_max_tokens": _kv_tokens(stats),
        "cells": cells,
        "workers": stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True,
                        choices=("all-cuda", "hybrid", "all-out"))
    parser.add_argument("--placement", default="subset:16:8")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--lengths", type=int, nargs="+", default=[8192, 32768, 65536])
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--prefill-stream", action="store_true")
    args = parser.parse_args()

    apply_config_env(args.config, args.placement, prefill_stream=args.prefill_stream)
    print(f"[probe] config={args.config} placement={args.placement} "
          f"prefill_stream={args.prefill_stream} max_model_len={args.max_model_len} "
          f"lengths={args.lengths} concurrency={args.concurrency} trials={args.trials}",
          flush=True)

    started = time.time()
    result = asyncio.run(run(args))
    result["wall_s"] = time.time() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"[probe] wrote {args.out} ({result['wall_s']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
