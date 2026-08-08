#!/usr/bin/env python3
"""Does B70 prefill chunking actually explain the prefill gap?

The hypothesis on record is that offloaded prefill is slow because
``_b70_prefill_partial`` splits the prompt into ``SHOOTING_BRAKE_B70_MAX_BATCH``
sized chunks, and every chunk re-reads the layer's whole expert working set out
of B70 VRAM. Under that hypothesis TTFT should fall roughly in proportion to
the chunk count, because the weight traffic -- not the activation traffic --
dominates.

That is a mechanism, not a measurement, and the chunk size is an environment
variable. So it is directly testable with no code change: sweep the chunk size,
measure TTFT at a fixed prompt length, and see whether the curve bends the way
the hypothesis says it must.

The alternative hypotheses this separates:

  * chunking dominates      -> TTFT falls ~linearly in chunk count, then flattens
                               once the prompt fits one dispatch
  * per-dispatch overhead   -> TTFT falls, but only by chunks x ~150us, which is
                               far too small to explain a 12x gap
  * B70 is simply slow at   -> TTFT is flat in chunk size; the gap is kernel
    prefill shapes             throughput and no buffer change will help

Run under the same environment as the benchmarks (oneAPI sourced, venv on PATH).
Each chunk size needs a fresh process because the pinned buffers are sized from
the variable at construction time.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from offload_benchmark import (  # noqa: E402
    apply_config_env,
    build_engine,
    stream_one,
    _context_prompt,
    worker_stats,
)


async def measure(engine, prompt: str, trials: int) -> dict[str, float]:
    # Warm on the real prompt, not a token stub. The first long prefill pays
    # one-time costs (graph capture for the shape, allocator growth) that
    # otherwise land entirely on the first measured length and read as a
    # chunk-size effect. An earlier revision of this probe warmed with a
    # 4-token prompt and reported the 1153-token point at 6010 tok/s against
    # 28556 for the 3123-token point -- an artefact, not a measurement.
    await stream_one(engine, prompt, 1)
    ttfts = []
    tokens = 0
    for _ in range(trials):
        t = await stream_one(engine, prompt, 1)
        ttfts.append(t.ttft_s)
        tokens = t.prompt_tokens
    mean = statistics.mean(ttfts)
    return {
        "prompt_tokens": tokens,
        "ttft_ms": mean * 1000.0,
        "ttft_ms_min": min(ttfts) * 1000.0,
        "prefill_tok_per_s": tokens / mean,
    }


async def run(args: argparse.Namespace) -> int:
    chunk = int(os.environ.get("SHOOTING_BRAKE_B70_MAX_BATCH", "128"))
    placement = os.environ.get("SHOOTING_BRAKE_PLACEMENT", "all-cuda")
    # The adapter needs HYBRID/B70_DEVICE/VRAM_SURGERY/B70_GRAPH together, not
    # just a placement string, and reads them at class-construction time.
    # Setting the placement alone silently yields an all-CUDA run wearing a
    # hybrid label: the first version of this probe did exactly that and
    # reproduced the all-CUDA baseline TTFT to 0.1 ms.
    config = "all-cuda" if placement == "all-cuda" else (
        "all-out" if placement.startswith("allout:") else "hybrid"
    )
    apply_config_env(config, placement)
    os.environ["SHOOTING_BRAKE_B70_MAX_BATCH"] = str(chunk)
    engine = build_engine(args.max_num_seqs, args.max_model_len)
    try:
        out: dict[str, object] = {
            "chunk_size": chunk,
            "placement": os.environ.get("SHOOTING_BRAKE_PLACEMENT", "all-cuda"),
            "points": [],
        }
        for length in args.context_lengths:
            prompt = _context_prompt(length)
            row = await measure(engine, prompt, args.trials)
            # How many B70 dispatches this prompt costs per offloaded layer.
            row["dispatches_per_layer"] = -(-row["prompt_tokens"] // chunk)
            row["target_tokens"] = length
            out["points"].append(row)
            print(
                f"chunk={chunk:<5} prompt={row['prompt_tokens']:<6} "
                f"dispatches/layer={row['dispatches_per_layer']:<4} "
                f"TTFT={row['ttft_ms']:8.1f} ms  "
                f"{row['prefill_tok_per_s']:9.0f} tok/s",
                flush=True,
            )
        out["workers"] = await worker_stats(engine)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(out, indent=2))
        return 0
    finally:
        engine.shutdown()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--max-num-seqs", type=int, default=16)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument(
        "--context-lengths", type=int, nargs="+", default=[1536, 4096],
    )
    p.add_argument("--out", default="")
    return asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
