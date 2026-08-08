#!/usr/bin/env python3
"""Where does streaming B70 experts beat dispatching them?

Two ways to compute an offloaded routed-expert partial during prefill, with
different cost curves against the same work:

* **dispatch** — send tokens to the B70, leave the weights there. Cost scales
  with token count (~26.9 us/token/layer measured).
* **stream** — send the weights to the 5090 once per layer, compute there.
  Cost is flat in token count (0.41 GiB/layer over Gen5 x16).

Flat versus linear means there is a crossover, and the crossover is the whole
question: below it dispatch wins and streaming wastes PCIe on a handful of
routes, above it dispatch loses and streaming amortises. The analytic estimate
is ~311 tokens, but it assumes every route reaches a distinct expert and
ignores what else is contending for the CUDA stream. This measures it instead.

The grid is (mode x prompt length x concurrency) because all three interact:

* **prompt length** sets tokens-per-forward directly, which is the x-axis of
  the crossover.
* **concurrency** changes what the streaming transfer competes with. Streaming
  occupies the CUDA stream, so under load it contends with work that dispatch
  would have overlapped with instead. A crossover measured at concurrency 1
  does not transfer to a loaded server, and that is exactly the case a serving
  system runs in.
* **mode** includes all-cuda as the ceiling, so every cell reads as a fraction
  of what the 5090 could do alone rather than only against the other offload
  mode.

Reported per cell: TTFT (the prefill metric this targets), output throughput,
and ITL (to confirm decode is untouched -- streaming must not regress it).

Usage:
    ./benchmarks/stream_matrix.sh
    MODES="dispatch stream" LENGTHS="512 2048" ./benchmarks/stream_matrix.sh
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from offload_benchmark import (  # noqa: E402
    apply_config_env,
    build_engine,
    reset_stats,
    stream_many,
    stream_one,
    worker_stats,
    _context_prompt,
)

#: Mode -> extra environment on top of the shared hybrid configuration.
#: all-cuda is the ceiling; the two offload modes differ only in the flag,
#: which is the point -- placement, surgery and graph mode are held constant
#: so the comparison isolates the prefill strategy.
MODES: dict[str, dict[str, str]] = {
    "all-cuda": {},
    "dispatch": {},
    "stream": {
        "SHOOTING_BRAKE_B70_PREFILL_STREAM": "1",
    },
}


def percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


async def cell(
    engine, prompt: str, concurrency: int, decode_tokens: int, trials: int
) -> dict[str, float]:
    """One (length, concurrency) point.

    Prompts are made distinct per request. Identical prompts would let prefix
    caching serve every request after the first, which measures the cache
    rather than the prefill path under test.
    """
    ttfts: list[float] = []
    itls: list[float] = []
    tps: list[float] = []
    for t in range(trials):
        prompts = [f"{prompt}\n(variant {t}.{i})" for i in range(concurrency)]
        timings, wall = await stream_many(engine, prompts, decode_tokens)
        ttfts += [x.ttft_s * 1000 for x in timings]
        for x in timings:
            itls += [d * 1000 for d in x.itls_s]
        tps.append(sum(len(x.itls_s) + 1 for x in timings) / wall)
    return {
        "ttft_ms_p50": percentile(ttfts, 0.50),
        "ttft_ms_p99": percentile(ttfts, 0.99),
        "ttft_ms_mean": statistics.mean(ttfts),
        "itl_ms_p50": percentile(itls, 0.50),
        "output_tok_per_s": statistics.mean(tps),
        "requests": concurrency * trials,
    }


async def run_mode(mode: str, args: argparse.Namespace) -> dict:
    env = dict(MODES[mode])
    placement = "all-cuda" if mode == "all-cuda" else args.placement
    config = "all-cuda" if mode == "all-cuda" else "hybrid"
    apply_config_env(config, placement)
    os.environ.update(env)
    if mode != "stream":
        # apply_config_env clears adapter vars, but the streaming flag is not
        # one of them; drop it explicitly so a stray export cannot turn the
        # dispatch baseline into a second streaming run wearing its label.
        os.environ.pop("SHOOTING_BRAKE_B70_PREFILL_STREAM", None)
    os.environ["SHOOTING_BRAKE_B70_STREAM_T"] = str(args.threshold)

    result: dict = {
        "mode": mode,
        "placement": placement,
        "threshold": args.threshold,
        "cells": [],
    }
    engine = build_engine(args.max_num_seqs, args.max_model_len)
    try:
        for length in args.lengths:
            prompt = _context_prompt(length)
            # Warm this shape before timing it: the first prefill at a new
            # length pays graph capture and allocator growth, which would
            # otherwise be attributed to whichever mode ran first.
            await stream_one(engine, prompt, 4)
            for conc in args.concurrency:
                await reset_stats(engine)
                t0 = time.monotonic()
                row = await cell(
                    engine, prompt, conc, args.decode_tokens, args.trials
                )
                row["target_prompt_tokens"] = length
                row["concurrency"] = conc
                row["wall_s"] = time.monotonic() - t0
                stats = await worker_stats(engine)
                if stats:
                    w = stats[0]
                    row["cpu_stream"] = w.get("cpu_stream")
                    row["poller"] = w.get("poller")
                result["cells"].append(row)
                print(
                    f"  {mode:9} len={length:<6} conc={conc:<4} "
                    f"TTFT p50 {row['ttft_ms_p50']:8.1f} ms  "
                    f"ITL {row['itl_ms_p50']:6.2f} ms  "
                    f"{row['output_tok_per_s']:7.1f} tok/s",
                    flush=True,
                )
        result["workers"] = await worker_stats(engine)
    finally:
        engine.shutdown()
    return result


async def run(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for mode in args.modes:
        if mode not in MODES:
            raise SystemExit(f"unknown mode: {mode}")
        print(f"== {mode} ==", flush=True)
        res = await run_mode(mode, args)
        (out.parent / f"{out.stem}-{mode}.json").write_text(
            json.dumps(res, indent=2)
        )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--modes", nargs="+", default=["all-cuda", "dispatch", "stream"])
    p.add_argument("--lengths", type=int, nargs="+",
                   default=[256, 512, 1024, 2048, 4096])
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16, 64])
    p.add_argument("--placement", default="subset:16:8")
    p.add_argument("--threshold", type=int, default=1,
                   help="B70 stream threshold; 1 forces streaming at every "
                        "length so the crossover can be located from data "
                        "rather than assumed")
    p.add_argument("--decode-tokens", type=int, default=32)
    p.add_argument("--trials", type=int, default=2)
    p.add_argument("--max-num-seqs", type=int, default=80)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--out", default="benchmarks/results/stream/matrix.json")
    return asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
