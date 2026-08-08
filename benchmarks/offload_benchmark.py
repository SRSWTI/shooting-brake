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

    python benchmarks/offload_benchmark.py --config all-cuda --out results/all-cuda.json
    python benchmarks/offload_benchmark.py --config hybrid   --out results/hybrid.json
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
    #: Per-position prompt logprobs, populated only when requested. This is
    #: the quality signal that token ids cannot provide: it is produced
    #: during prefill and degrades continuously, where a sampled id only
    #: changes once damage flips an argmax.
    prompt_logprobs: list[float] = field(default_factory=list)

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
    "SHOOTING_BRAKE_ALL_OUT",
    "SHOOTING_BRAKE_B70_PREFILL_STREAM",
)


def apply_config_env(
    config: str, placement: str, prefill_stream: bool = False,
) -> None:
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
    if config in ("hybrid", "all-out"):
        env.update({
            "SHOOTING_BRAKE_PLACEMENT": placement,
            "SHOOTING_BRAKE_HYBRID": "1",
            "SHOOTING_BRAKE_B70_DEVICE": "1",
            "SHOOTING_BRAKE_VRAM_SURGERY": "1",
            "SHOOTING_BRAKE_B70_GRAPH": "1",
        })
        # The cold tier is refused unless asked for explicitly, so a
        # mistyped placement degrades to an error rather than to a
        # silently B70-only run wearing an all-out label.
        if config == "all-out":
            env["SHOOTING_BRAKE_ALL_OUT"] = "1"
    elif config == "all-cuda":
        # The adapter stays installed, with an all-CUDA placement: no
        # surgery, no B70, no Tier 3. This isolates the hybrid path
        # rather than the presence of the plugin.
        env["SHOOTING_BRAKE_PLACEMENT"] = "all-cuda"
    else:
        raise SystemExit(f"unknown config: {config}")
    # Passed explicitly rather than inherited. It is listed in ADAPTER_VARS
    # and therefore cleared above, so an exported value from the calling
    # shell cannot turn a dispatch baseline into a streaming run wearing the
    # baseline's label -- the failure mode that makes a comparison table
    # quietly meaningless.
    if prefill_stream:
        if config == "all-cuda":
            raise SystemExit(
                "prefill streaming needs an offloaded placement; "
                "there is nothing to stream in all-cuda"
            )
        env["SHOOTING_BRAKE_B70_PREFILL_STREAM"] = "1"
    os.environ.update(env)


def build_engine(max_num_seqs: int, max_model_len: int) -> Any:
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.plugins import load_general_plugins
    from vllm.v1.engine.async_llm import AsyncLLM

    load_general_plugins()
    args = AsyncEngineArgs(
        model=MODEL,
        enforce_eager=False,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        dtype="bfloat16",
        trust_remote_code=True,
        disable_log_stats=True,
    )
    return AsyncLLM.from_engine_args(args)


async def stream_one(
    engine: Any, prompt: str, max_tokens: int,
    prompt_logprobs: bool = False,
) -> StreamTiming:
    """Drive one request and timestamp every token as it arrives."""
    from vllm import SamplingParams

    params = SamplingParams(
        temperature=0.0, max_tokens=max_tokens,
        # 0 = the sampled token's own logprob at each prompt position, which
        # is all the quality metric needs. Requesting a top-k list instead
        # would cost far more to serialise for no extra signal.
        prompt_logprobs=0 if prompt_logprobs else None,
    )
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
        if prompt_logprobs and getattr(output, "prompt_logprobs", None):
            # Position 0 has no prediction behind it, so it carries no
            # logprob and is skipped rather than counted as zero.
            timing.prompt_logprobs = [
                next(iter(d.values())).logprob
                for d in output.prompt_logprobs[1:] if d
            ]

    timing.total_s = time.perf_counter() - start
    timing.output_tokens = seen
    return timing


async def stream_many(
    engine: Any, prompts: list[str], max_tokens: int,
    prompt_logprobs: bool = False,
) -> tuple[list[StreamTiming], float]:
    """Run prompts concurrently; returns timings and wall-clock span."""
    start = time.perf_counter()
    timings = await asyncio.gather(
        *(stream_one(engine, p, max_tokens, prompt_logprobs)
          for p in prompts)
    )
    return list(timings), time.perf_counter() - start


# --- workloads ---------------------------------------------------------


async def run_correctness(engine: Any) -> list[dict[str, Any]]:
    """Greedy decode plus prompt logprobs — quality, not just agreement.

    Token ids alone are a weak instrument, and this project has the receipt:
    a prefill path that dropped every offloaded route still emitted
    byte-identical tokens, because damage has to flip an argmax before a
    sampled id moves. It cost 0.49 nats/token and went unnoticed through a
    full benchmark cycle.

    ``prompt_logprobs`` is produced during prefill, one value per prompt
    position, so it reads that pass directly and moves continuously with the
    damage rather than only at a threshold. Recording it here means every
    run carries a quality number instead of needing a separate probe to
    notice a tier has gone quietly wrong.
    """
    timings, _ = await stream_many(
        engine, list(CORRECTNESS_PROMPTS), 32, prompt_logprobs=True,
    )
    rows = []
    for prompt, t in zip(CORRECTNESS_PROMPTS, timings, strict=True):
        plp = t.prompt_logprobs
        rows.append({
            "prompt": prompt,
            "token_ids": t.token_ids,
            "text": t.text,
            "prompt_tokens": t.prompt_tokens,
            "prompt_logprob_sum": sum(plp) if plp else None,
            "prompt_logprob_mean": (sum(plp) / len(plp)) if plp else None,
        })
    return rows


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


def _context_prompt(target_tokens: int) -> str:
    """A prompt close to ``target_tokens`` after tokenization.

    Repeats a neutral sentence; the count is calibrated so the result
    lands near the target without going over ``max_model_len``.
    """
    # ~13 tokens per sentence; keep a 64-token headroom for the question.
    sentences = max(1, (target_tokens - 64) // 13)
    body = (
        "The quick brown fox jumps over the lazy dog. "
    ) * sentences
    return (
        body
        + "\nIgnore the text above. Question: what is 7 multiplied by 6? "
        "Answer with the number only."
    )


async def run_context_sweep(
    engine: Any, lengths: list[int], decode_tokens: int
) -> list[dict[str, Any]]:
    """Single-stream decode latency and throughput vs prompt length.

    This is where the hybrid's KV-cache win should matter most: longer
    contexts consume more KV, so the same request is more expensive on
    the all-CUDA baseline, which has 4.2x less KV headroom.
    """
    rows = []
    for length in lengths:
        prompt = _context_prompt(length)
        timing = await stream_one(engine, prompt, decode_tokens)
        rows.append({
            "target_prompt_tokens": length,
            "actual_prompt_tokens": timing.prompt_tokens,
            "output_tokens": timing.output_tokens,
            "decode_tok_per_s": timing.decode_tok_per_s,
            "ttft_ms": timing.ttft_s * 1e3,
            "itl_p50_ms": _pct(timing.itls_s, 0.50) * 1e3,
            "itl_p99_ms": _pct(timing.itls_s, 0.99) * 1e3,
        })
    return rows


async def run_capacity_frontier(
    engine: Any, lengths: list[int], decode_tokens: int,
    kv_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """Throughput and latency across the prompt-length x concurrency grid.

    Every (length, wave) point is recorded, not just the largest wave that
    completed. The intermediate points are the grid the offload tradeoff
    actually lives on -- concurrency at one prompt length and prompt length at
    concurrency one each show a slice, and the interesting behaviour is where
    they cross.

    ``max_completed_wave`` reads as completion, not as fit: vLLM preempts
    rather than rejecting, so a wave whose KV footprint exceeds capacity still
    finishes and still reports a wave number. Throughput and ITL are where
    exhaustion actually shows up.

    Waves are capped by measured KV capacity when it is known. Past that
    ceiling every request is preempted, so the point measures the scheduler
    rather than the placement -- and it is ruinously expensive to collect,
    because each wave re-prefills every prompt. At 32k context the full wave
    list is 6.8M prompt tokens per length, which at the hybrid's measured
    prefill rate is most of an hour for a result that says nothing.
    """
    waves = [1, 4, 8, 12, 16, 24, 32, 48, 64]
    rows = []
    for length in lengths:
        prompt = _context_prompt(length)
        # One wave past the ceiling is kept deliberately: the first point that
        # does not fit is the one that shows what exhaustion costs.
        if kv_tokens:
            fits = max(1, kv_tokens // max(length + decode_tokens, 1))
            usable = [n for n in waves if n <= fits]
            beyond = [n for n in waves if n > fits][:1]
            wave_list = usable + beyond
        else:
            wave_list = waves
        points: list[dict[str, Any]] = []
        max_ok = 0
        for n in wave_list:
            prompts = [prompt] * n
            try:
                timings, wall = await stream_many(
                    engine, prompts, decode_tokens
                )
            except Exception:
                break
            if any(t.output_tokens == 0 for t in timings):
                break
            total_out = sum(t.output_tokens for t in timings)
            itls = [v for t in timings for v in t.itls_s]
            max_ok = n
            points.append({
                "concurrent_requests": n,
                "actual_prompt_tokens": timings[0].prompt_tokens,
                "total_output_tokens": total_out,
                "output_tok_per_s": total_out / wall,
                "ttft_p50_ms": _pct([t.ttft_s for t in timings], 0.50) * 1e3,
                "itl_p50_ms": _pct(itls, 0.50) * 1e3,
                "itl_p99_ms": _pct(itls, 0.99) * 1e3,
            })
        rows.append({
            "target_prompt_tokens": length,
            "max_completed_wave": max_ok,
            "points": points,
        })
    return rows


# --- worker telemetry --------------------------------------------------


async def worker_stats(engine: Any) -> list[dict[str, Any]]:
    from shooting_brake_vllm.telemetry import collect_worker_stats

    return await engine.collective_rpc(collect_worker_stats)


async def reset_stats(engine: Any) -> None:
    from shooting_brake_vllm.telemetry import reset_worker_stats

    await engine.collective_rpc(reset_worker_stats)


# --- driver ------------------------------------------------------------


async def run(args: argparse.Namespace) -> dict[str, Any]:
    engine = build_engine(args.max_num_seqs, args.max_model_len)
    try:
        # Warm up: the first request pays graph capture and lazy init.
        await stream_one(engine, DECODE_PROMPT, 16)
        await reset_stats(engine)

        result: dict[str, Any] = {
            "config": args.config,
            "model": MODEL,
            "placement": os.environ.get("SHOOTING_BRAKE_PLACEMENT", "all-cuda"),
            "max_num_seqs": args.max_num_seqs,
            "max_model_len": args.max_model_len,
        }
        result["correctness"] = await run_correctness(engine)

        # Poller counters accumulate across every phase, so a single figure
        # at the end averages decode dispatches together with prefill ones
        # that touch far more experts -- which makes the kernel-vs-overhead
        # split unreadable for the case that matters. Snapshot around the
        # single-stream phase to get decode alone.
        await reset_stats(engine)
        result["single_stream"] = await run_single_stream(
            engine, args.trials, args.decode_tokens
        )
        result["workers_decode_only"] = await worker_stats(engine)
        # Prefill is measured before the batched phase, not after. TTFT is
        # sensitive to KV-cache occupancy, so running it downstream of a
        # concurrency sweep made the figure depend on how wide that sweep
        # went: all-CUDA prefill read 22282 tok/s after a conc<=8 phase and
        # 6083 after conc<=64, on a code path neither run touched. Ordering
        # it here keeps prefill comparable across runs.
        result["prefill"] = await run_prefill(engine, args.trials)
        result["batched"] = await run_batched(
            engine, args.concurrency, args.batch_tokens
        )
        result["context_sweep"] = await run_context_sweep(
            engine, args.context_lengths, args.decode_tokens
        )
        # Read KV capacity before the frontier so it can skip waves that
        # cannot fit. Each wave re-prefills every prompt, so an uncapped list
        # at long context costs hours to measure preemption.
        kv_tokens = None
        for w in (result.get("workers_decode_only") or []):
            kv_tokens = (w.get("kv_cache") or {}).get("max_tokens") or kv_tokens
        result["capacity_frontier"] = await run_capacity_frontier(
            engine, args.context_lengths, args.decode_tokens,
            kv_tokens=kv_tokens,
        )
        result["workers"] = await worker_stats(engine)
        return result
    finally:
        engine.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True,
        choices=("all-cuda", "hybrid", "all-out"),
        help="all-cuda baseline, hybrid (CUDA+B70), or all-out "
             "(CUDA+B70+CPU DRAM); all-out additionally requires an "
             "'allout:...' placement",
    )
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
    parser.add_argument(
        "--prefill-stream", action="store_true",
        help="compute B70-owned routes on the 5090 from streamed weights "
             "during prefill, above SHOOTING_BRAKE_B70_STREAM_T tokens per "
             "forward. Decode is unaffected; pair a run with and without "
             "it to read the effect.",
    )
    parser.add_argument(
        "--context-lengths", type=int, nargs="+",
        default=[512, 2048, 4096, 8192],
        help="prompt lengths for the context sweep and capacity frontier",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=8192,
        help="vLLM admission cap; the model supports 262144 natively. "
             "Raise it to exercise the hybrid's KV capacity at long context.",
    )
    args = parser.parse_args()

    apply_config_env(args.config, args.placement, args.prefill_stream)
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
    print("context sweep (single-stream decode by prompt length):")
    for row in result["context_sweep"]:
        print(
            f"  {row['actual_prompt_tokens']:>5} tok prompt: "
            f"{row['decode_tok_per_s']:>6.1f} tok/s  "
            f"TTFT {row['ttft_ms']:>6.1f} ms  "
            f"ITL p50 {row['itl_p50_ms']:>5.2f}  "
            f"p99 {row['itl_p99_ms']:>5.2f} ms"
        )
    print("capacity frontier (largest concurrent wave that completed):")
    for row in result["capacity_frontier"]:
        n = row.get("concurrent_requests", 0)
        if n:
            print(
                f"  {row['target_prompt_tokens']:>5} tok prompt: "
                f"{n:>3} concurrent  "
                f"{row['output_tok_per_s']:>8.1f} tok/s  "
                f"ITL p50 {row['itl_p50_ms']:>6.2f}  "
                f"p99 {row['itl_p99_ms']:>6.2f} ms"
            )
        else:
            print(
                f"  {row['target_prompt_tokens']:>5} tok prompt: none"
            )
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
