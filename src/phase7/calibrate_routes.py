#!/usr/bin/env python3
"""Measure whether Qwen3.6-35B's expert routing is actually skewed.

The README's core claim -- "a small set of hot experts handles most tokens,
and the rest sit idle" -- has never been checked against this model, and every
placement policy in the repo is positional rather than frequency-based. This
run either substantiates the claim with numbers or retires it.

Method:

* All-CUDA mode. Routing is computed by the router on the 5090 before any
  dispatch decision, so the histogram is placement-independent; all-CUDA is
  simply the fastest way to push tokens through, and it involves zero B70
  round trips.
* ``SHOOTING_BRAKE_ROUTE_STATS=1`` turns on a device-side scatter_add per
  layer (route_stats.py). Counters are zeroed *after* warmup, so vLLM's
  profiling pass and graph-capture dummy batches do not pollute the table.
* Corpus is NeelNanda/pile-10k (cached locally): diverse real text. A
  repeated synthetic sentence would manufacture skew; a single domain would
  understate the tail. Documents are truncated, then a short decode runs per
  document so the table reflects both prefill routing (corpus tokens) and
  decode routing (model-generated tokens), in roughly the mix serving sees.
* The counter lives in the EngineCore worker process. The CSV is written
  from inside the worker via collective_rpc(dump_route_histogram); the
  driver then loads it back and prints the Lucebox-style skew analysis.

Usage:
    ./phase7/calibrate_routes.sh                 # defaults: 2000 docs
    DOCS=10000 ./phase7/calibrate_routes.sh      # full corpus
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from offload_benchmark import (  # noqa: E402
    apply_config_env,
    build_engine,
    reset_stats,
    stream_many,
    stream_one,
)


def load_corpus(n_docs: int, max_chars: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("NeelNanda/pile-10k", split="train")
    docs = []
    for row in ds:
        text = row["text"].strip()
        if len(text) < 200:  # skip fragments; they carry few routes
            continue
        docs.append(text[:max_chars])
        if len(docs) >= n_docs:
            break
    return docs


async def run(args: argparse.Namespace) -> int:
    from shooting_brake_vllm.telemetry import dump_route_histogram

    apply_config_env("all-cuda", "all-cuda")
    os.environ["SHOOTING_BRAKE_ROUTE_STATS"] = "1"
    os.environ["SHOOTING_BRAKE_ROUTE_STATS_OUT"] = args.out

    docs = load_corpus(args.docs, args.max_chars)
    print(f"corpus: {len(docs)} documents, <= {args.max_chars} chars each",
          flush=True)

    engine = build_engine(args.max_num_seqs, args.max_model_len)
    try:
        # Warm up, then zero: drops the profiling pass and capture dummies.
        await stream_one(engine, docs[0], 8)
        await reset_stats(engine)

        t0 = time.monotonic()
        done = 0
        for start in range(0, len(docs), args.batch):
            chunk = docs[start:start + args.batch]
            await stream_many(engine, chunk, args.decode_tokens)
            done += len(chunk)
            el = time.monotonic() - t0
            print(f"  {done}/{len(docs)} docs  ({el:5.0f}s, "
                  f"{done / el:5.1f} docs/s)", flush=True)

        paths = await engine.collective_rpc(dump_route_histogram)
        print(f"histogram written: {paths}", flush=True)
    finally:
        engine.shutdown()

    # Analysis runs on the CSV, in this process, after the engine is gone.
    from shooting_brake_vllm.route_stats import analyze, format_analysis, load_csv

    counts, top_k = load_csv(args.out)
    print()
    print(format_analysis(analyze(counts, top_k)), flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--docs", type=int, default=int(os.environ.get("DOCS", "2000")))
    p.add_argument("--max-chars", type=int, default=6000,
                   help="~1500 tokens per document")
    p.add_argument("--decode-tokens", type=int, default=32)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--out", default="benchmarks/results/route_stats.csv")
    return asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
