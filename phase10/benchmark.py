"""Phase 10 — controlled benchmark: all-CUDA vLLM vs Shooting Brake hybrid.

Runs one configuration per process (the two differ only by environment)
and writes a JSON result file.  ``compare.py`` runs both and diffs them.

Measured, per plan.md "Phase 10 — Controlled production benchmark":

  * output and request throughput, single-stream and batched
  * TTFT and inter-token-latency percentiles, from real per-token
    arrival times (streamed through ``AsyncLLM``, not inferred by
    dividing a wall-clock span)
  * prefill throughput
  * CUDA/B70 route shares, per layer
  * B70 dispatch count and mean service time
  * CUDA memory: allocated, reserved, free
  * generated-token agreement across configurations

Both configurations load the same checkpoint and run the same prompt
matrix at temperature 0, so token sequences must match exactly.  A
mismatch is a correctness failure, not a benchmark artifact.

Usage::

    python phase10/benchmark.py --config all-cuda --out results/all-cuda.json
    python phase10/benchmark.py --config hybrid   --out results/hybrid.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MODEL = "unsloth/Qwen3.6-35B-A3B-NVFP4"

# Fixed prompt matrix: short and long prefills, and prompts that pull
# different regions of the expert space, so route shares are not an
# artifact of a single topic.
CORRECTNESS_PROMPTS: tuple[str, ...] = (
    "State the integer after 41.",
    "Write a Python function that reverses a linked list.",
    "Explain why the sky appears blue, in two sentences.",
    "Translate to French: The weather is pleasant today.",
    "What is the derivative of x^3 * sin(x)?",
    "List three causes of the fall of the Western Roman Empire.",
    "Summarize the difference between TCP and UDP.",
    "Write a haiku about winter mountains.",
)

DECODE_PROMPT = (
    "Write a detailed technical explanation of how a modern CPU pipeline works."
)

# Long prefill, to separate prefill cost from decode.
PREFILL_PROMPT = (
    "Consider the following system description, then answer the question "
    "that follows.\n\n"
    + (
        "A distributed inference service routes requests across a "
        "heterogeneous pool of accelerators. Each accelerator holds a "
        "subset of model parameters resident in its local memory. The "
        "scheduler must decide, for every request, which accelerator "
        "computes which portion of the model, subject to memory capacity, "
        "interconnect bandwidth, and tail-latency constraints. "
    )
    * 24
    + "\n\nQuestion: what dominates the critical path?"
)


@dataclass
class StreamTiming:
    """Arrival-time record for one streamed request."""

    prompt_tokens: int = 0
    output_tokens: int = 0
    ttft_s: float = 0.0
    total_s: float = 0.0
    itls_s: list[float] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    text: str = ""

    @property
    def decode_tok_per_s(self) -> float:
        """Decode rate, prefill excluded."""
        decode_s = self.total_s - self.ttft_s
        n = self.output_tokens - 1
        return n / decode_s if decode_s > 0 and n > 0 else 0.0


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * q), len(ordered) - 1)]


def _latency_summary(timings: list[StreamTiming]) -> dict[str, float]:
    itls = [v for t in timings for v in t.itls_s]
    ttfts = [t.ttft_s for t in timings]
    return {
        "itl_mean_ms": statistics.mean(itls) * 1e3 if itls else 0.0,
        "itl_p50_ms": _pct(itls, 0.50) * 1e3,
        "itl_p95_ms": _pct(itls, 0.95) * 1e3,
        "itl_p99_ms": _pct(itls, 0.99) * 1e3,
        "itl_max_ms": max(itls) * 1e3 if itls else 0.0,
        "ttft_mean_ms": statistics.mean(ttfts) * 1e3 if ttfts else 0.0,
        "ttft_p95_ms": _pct(ttfts, 0.95) * 1e3,
    }


# --- engine ------------------------------------------------------------


# Every adapter switch the benchmark controls. Both configurations clear
# all of them and then set only what they need, so an inherited value
# from the caller's shell cannot silently enable the hybrid path in the
# baseline — which reads as a plausible but wrong all-CUDA number.
ADAPTER_VARS = (
    "SHOOTING_BRAKE_HYBRID",
    "SHOOTING_BRAKE_B70_DEVICE",
    "SHOOTING_BRAKE_VRAM_SURGERY",
    "SHOOTING_BRAKE_B70_GRAPH",
    "SHOOTING_BRAKE_B70_STATS",
    "SHOOTING_BRAKE_PLACEMENT",
)


def apply_config_env(config: str, placement: str) -> None:
    """Make the environment exactly describe ``config``.

    The adapter reads these at class-construction time, so this must run
    before the engine is built.
    """
    for name in ADAPTER_VARS:
        os.environ.pop(name, None)

    env = {
        "VLLM_PLUGINS": "shooting_brake_vllm",
        "SHOOTING_BRAKE_PHASE4": "all-cuda",
        "SHOOTING_BRAKE_MODEL": MODEL,
        # collective_rpc ships the telemetry callable to the workers.
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        "SHOOTING_BRAKE_B70_STATS": "1",
    }
    if config == "hybrid":
        env.update({
            "SHOOTING_BRAKE_PLACEMENT": placement,
            "SHOOTING_BRAKE_HYBRID": "1",
            "SHOOTING_BRAKE_B70_DEVICE": "1",
            "SHOOTING_BRAKE_VRAM_SURGERY": "1",
            "SHOOTING_BRAKE_B70_GRAPH": "1",
        })
    elif config == "all-cuda":
        # The adapter stays installed, with an all-CUDA placement: no
        # surgery, no B70, no Tier 3. This isolates the hybrid path
        # rather than the presence of the plugin.
        env["SHOOTING_BRAKE_PLACEMENT"] = "all-cuda"
    else:
        raise SystemExit(f"unknown config: {config}")
    os.environ.update(env)


def build_engine(max_num_seqs: int) -> Any:
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.plugins import load_general_plugins
    from vllm.v1.engine.async_llm import AsyncLLM

    load_general_plugins()
    args = AsyncEngineArgs(
        model=MODEL,
        enforce_eager=False,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_model_len=8192,
        max_num_seqs=max_num_seqs,
        dtype="bfloat16",
        trust_remote_code=True,
        disable_log_stats=True,
    )
    return AsyncLLM.from_engine_args(args)


async def stream_one(
    engine: Any, prompt: str, max_tokens: int
) -> StreamTiming:
    """Drive one request and timestamp every token as it arrives."""
    from vllm import SamplingParams

    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    timing = StreamTiming()
    start = time.perf_counter()
    previous = start
    seen = 0

    async for output in engine.generate(
        prompt, params, request_id=str(uuid.uuid4())
    ):
        now = time.perf_counter()
        completion = output.outputs[0]
        produced = len(completion.token_ids)
        if produced <= seen:
            continue
        if seen == 0:
            timing.ttft_s = now - start
        else:
            # One chunk can carry several tokens; charge the interval
            # evenly across them rather than to the last one.
            gap = (now - previous) / (produced - seen)
            timing.itls_s.extend([gap] * (produced - seen))
        seen = produced
        previous = now
        timing.token_ids = list(completion.token_ids)
        timing.text = completion.text
        timing.prompt_tokens = len(output.prompt_token_ids or ())

    timing.total_s = time.perf_counter() - start
    timing.output_tokens = seen
    return timing


async def stream_many(
    engine: Any, prompts: list[str], max_tokens: int
) -> tuple[list[StreamTiming], float]:
    """Run prompts concurrently; returns timings and wall-clock span."""
    start = time.perf_counter()
    timings = await asyncio.gather(
        *(stream_one(engine, p, max_tokens) for p in prompts)
    )
    return list(timings), time.perf_counter() - start


# --- workloads ---------------------------------------------------------


async def run_correctness(engine: Any) -> list[dict[str, Any]]:
    """Greedy-decode the prompt matrix; token ids are the comparison key."""
    timings, _ = await stream_many(engine, list(CORRECTNESS_PROMPTS), 32)
    return [
        {"prompt": prompt, "token_ids": t.token_ids, "text": t.text}
        for prompt, t in zip(CORRECTNESS_PROMPTS, timings, strict=True)
    ]


async def run_single_stream(
    engine: Any, trials: int, max_tokens: int
) -> dict[str, Any]:
    """One request at a time — the latency-sensitive case."""
    timings = [
        await stream_one(engine, DECODE_PROMPT, max_tokens)
        for _ in range(trials)
    ]
    rates = [t.output_tokens / t.total_s for t in timings]
    summary = {
        "trials": trials,
        "output_tokens": timings[0].output_tokens,
        "tok_per_s_mean": statistics.mean(rates),
        "tok_per_s_min": min(rates),
        "tok_per_s_max": max(rates),
        "decode_tok_per_s_mean": statistics.mean(
            t.decode_tok_per_s for t in timings
        ),
    }
    summary.update(_latency_summary(timings))
    return summary


async def run_batched(
    engine: Any, concurrencies: list[int], max_tokens: int
) -> list[dict[str, Any]]:
    """Throughput and tail latency as a function of in-flight requests."""
    rows = []
    for n in concurrencies:
        prompts = [f"{DECODE_PROMPT} (variant {i})" for i in range(n)]
        timings, wall = await stream_many(engine, prompts, max_tokens)
        total_out = sum(t.output_tokens for t in timings)
        row = {
            "concurrency": n,
            "wall_s": wall,
            "total_output_tokens": total_out,
            "output_tok_per_s": total_out / wall,
            "requests_per_s": n / wall,
        }
        row.update(_latency_summary(timings))
        rows.append(row)
    return rows


async def run_prefill(engine: Any, trials: int) -> dict[str, Any]:
    """Prefill throughput: long prompt, one output token."""
    timings = [
        await stream_one(engine, PREFILL_PROMPT, 1) for _ in range(trials)
    ]
    n_tokens = timings[0].prompt_tokens
    ttfts = [t.ttft_s for t in timings]
    return {
        "prompt_tokens": n_tokens,
        "ttft_s_mean": statistics.mean(ttfts),
        "prefill_tok_per_s": n_tokens / statistics.mean(ttfts),
    }


# --- worker telemetry --------------------------------------------------


async def worker_stats(engine: Any) -> list[dict[str, Any]]:
    from shooting_brake_vllm.telemetry import collect_worker_stats

    return await engine.collective_rpc(collect_worker_stats)


async def reset_stats(engine: Any) -> None:
    from shooting_brake_vllm.telemetry import reset_worker_stats

    await engine.collective_rpc(reset_worker_stats)


# --- driver ------------------------------------------------------------


async def run(args: argparse.Namespace) -> dict[str, Any]:
    engine = build_engine(args.max_num_seqs)
    try:
        # Warm up: the first request pays graph capture and lazy init.
        await stream_one(engine, DECODE_PROMPT, 16)
        await reset_stats(engine)

        result: dict[str, Any] = {
            "config": args.config,
            "model": MODEL,
            "placement": os.environ.get("SHOOTING_BRAKE_PLACEMENT", "all-cuda"),
            "max_num_seqs": args.max_num_seqs,
        }
        result["correctness"] = await run_correctness(engine)
        result["single_stream"] = await run_single_stream(
            engine, args.trials, args.decode_tokens
        )
        result["batched"] = await run_batched(
            engine, args.concurrency, args.batch_tokens
        )
        result["prefill"] = await run_prefill(engine, args.trials)
        result["workers"] = await worker_stats(engine)
        return result
    finally:
        engine.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, choices=("all-cuda", "hybrid"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--decode-tokens", type=int, default=400)
    parser.add_argument("--batch-tokens", type=int, default=128)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument(
        "--concurrency", type=int, nargs="+", default=[1, 4, 16, 32, 64]
    )
    parser.add_argument(
        "--placement", default="split:128",
        help="placement policy for the hybrid config",
    )
    args = parser.parse_args()

    apply_config_env(args.config, args.placement)
    result = asyncio.run(run(args))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    single = result["single_stream"]
    print(f"\n=== {args.config} ===")
    print(
        f"single-stream: {single['tok_per_s_mean']:.1f} tok/s  "
        f"ITL p50 {single['itl_p50_ms']:.2f} ms  "
        f"p99 {single['itl_p99_ms']:.2f} ms  "
        f"TTFT {single['ttft_mean_ms']:.1f} ms"
    )
    for row in result["batched"]:
        print(
            f"concurrency {row['concurrency']:>3}: "
            f"{row['output_tok_per_s']:>8.1f} tok/s  "
            f"ITL p50 {row['itl_p50_ms']:>6.2f} ms"
        )
    print(f"prefill: {result['prefill']['prefill_tok_per_s']:.0f} tok/s")
    for worker in result["workers"]:
        routes = worker.get("routes")
        if routes:
            print(f"routes: B70 share {routes['b70_share'] * 100:.1f}%")
        poller = worker.get("poller")
        if poller:
            print(
                f"poller: {poller['dispatches']} dispatches, "
                f"service mean {poller['service_mean_us']:.1f} us, "
                f"{poller['errors']} errors"
            )
        kv = worker["kv_cache"]
        mem = worker["cuda_memory"]
        print(
            f"kv cache: {kv['max_tokens']:,} tokens "
            f"({kv['num_gpu_blocks']:,} blocks); "
            f"{mem['allocated_gib']:.2f} GiB allocated"
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
