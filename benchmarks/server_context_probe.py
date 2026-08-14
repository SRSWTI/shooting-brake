#!/usr/bin/env python3
"""Bracket the TTFT cliff against a live server, sampling PCIe link state.

Two questions, one run.

1. Where exactly does prefill fall off? Track A measured ~20k tok/s prefill
   at and below 32,768 prompt tokens and ~2.05k tok/s at and above 65,536 --
   a 10x step with nothing measured in between. This walks the gap in five
   steps at fixed output length, several repeats, at more than one
   concurrency, so the shape of the transition is measured rather than
   inferred from its endpoints.

2. Is the B70 link actually negotiated where it should be? sysfs reports the
   card's internal downstream port (0000:11:01.0) and the GPU (0000:12:00.0)
   at 2.5 GT/s x1 while the chain above them runs 16 GT/s x4. Intel dGPUs
   downtrain when idle, so an idle reading proves nothing; this samples the
   link continuously *while the B70 is computing* and reports the best state
   observed. A link that stays at Gen1 x1 under load caps the interconnect at
   ~250 MB/s, which would explain the prefill cost directly.

`ignore_eos` is set: greedy decode on a repetitive synthetic prompt otherwise
stops early (offload_benchmark hit exactly 8 tokens on 12 of 20 points), and a
decode rate over 7 tokens is noise.

Usage:
  ./.venv/bin/python benchmarks/server_context_probe.py \
      --lengths 8192 32768 40960 49152 57344 65536 \
      --concurrency 1 4 --repeats 2 --max-tokens 32 \
      --out benchmarks/results/server_probe/bracket.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import threading
import time
from pathlib import Path
from typing import Any

import httpx

# The card's own upstream port, its internal downstream port, the GPU, and
# the 5090 for comparison. Sampled together so a change can be attributed.
PCI_WATCH = {
    "b70_gpu": "0000:12:00.0",
    "b70_downstream_port": "0000:11:01.0",
    "b70_upstream_port": "0000:10:00.0",
    "rtx5090": "0000:01:00.0",
}

# One neutral sentence, repeated. ~0.75 tokens/word for this tokenizer, so the
# repeat count is calibrated below against the server's reported usage.
_SENTENCE = (
    "The quick brown fox jumps over the lazy dog while the engineer "
    "measures interconnect latency between two accelerators. "
)


def _read(path: str) -> str:
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return ""


def pcie_state() -> dict[str, dict[str, Any]]:
    out = {}
    for label, bdf in PCI_WATCH.items():
        base = f"/sys/bus/pci/devices/{bdf}"
        speed = _read(f"{base}/current_link_speed").split()[0:1]
        width = _read(f"{base}/current_link_width")
        out[label] = {
            "speed_gts": float(speed[0]) if speed else None,
            "width": int(width) if width.isdigit() else None,
        }
    return out


class PcieSampler(threading.Thread):
    """Samples link state until stopped; keeps the best state seen.

    'Best' = highest speed x width product, because the question is whether
    the link is *capable* of training up under load, and any single sample can
    catch it mid-downtrain.
    """

    def __init__(self, interval: float = 0.25) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        # NOT `_stop`: threading.Thread uses that name for an internal method.
        self._done = threading.Event()
        self.samples: list[dict[str, Any]] = []

    def run(self) -> None:
        while not self._done.is_set():
            self.samples.append({"t": time.time(), **pcie_state()})
            self._done.wait(self.interval)

    def stop(self) -> dict[str, Any]:
        self._done.set()
        self.join(timeout=2.0)
        best: dict[str, Any] = {}
        for label in PCI_WATCH:
            seen = [
                s[label] for s in self.samples
                if s[label]["speed_gts"] and s[label]["width"]
            ]
            if not seen:
                best[label] = None
                continue
            top = max(seen, key=lambda v: v["speed_gts"] * v["width"])
            distinct = sorted({(v["speed_gts"], v["width"]) for v in seen})
            best[label] = {
                "best_speed_gts": top["speed_gts"],
                "best_width": top["width"],
                "states_seen": [f"{s} GT/s x{w}" for s, w in distinct],
                "n_samples": len(seen),
            }
        return best


def build_prompt(target_tokens: int) -> str:
    # Rough first cut; the server reports actual prompt_tokens and the caller
    # records that, so no calibration loop is needed for a length *bracket*.
    words_per_token = 0.75
    n_words = int(target_tokens * words_per_token)
    words = _SENTENCE.split()
    reps = n_words // len(words) + 1
    return " ".join((words * reps)[:n_words])


async def one_request(
    client: httpx.AsyncClient, target: str, model: str, prompt: str, max_tokens: int
) -> dict[str, Any]:
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    ttft = None
    itls: list[float] = []
    n_tokens = 0
    prompt_tokens = 0
    start = time.perf_counter()
    previous = start
    async with client.stream("POST", f"{target}/v1/completions", json=body) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload.strip() == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                prompt_tokens = chunk["usage"].get("prompt_tokens", 0) or prompt_tokens
            choices = chunk.get("choices") or []
            if not choices or not choices[0].get("text"):
                continue
            now = time.perf_counter()
            if ttft is None:
                ttft = now - start
            else:
                itls.append(now - previous)
            previous = now
            n_tokens += 1
    total = time.perf_counter() - start
    return {
        "ttft_s": ttft if ttft is not None else float("nan"),
        "total_s": total,
        "output_tokens": n_tokens,
        "prompt_tokens": prompt_tokens,
        "itls_s": itls,
    }


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(q * len(ordered)), len(ordered) - 1)
    return ordered[idx]


async def run(args: argparse.Namespace) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    limits = httpx.Limits(max_connections=64, max_keepalive_connections=64)
    timeout = httpx.Timeout(args.timeout, connect=10.0)
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        for length in args.lengths:
            prompt = build_prompt(length)
            for conc in args.concurrency:
                for rep in range(args.repeats):
                    sampler = PcieSampler()
                    sampler.start()
                    t0 = time.perf_counter()
                    results = await asyncio.gather(*(
                        one_request(client, args.target, args.model, prompt, args.max_tokens)
                        for _ in range(conc)
                    ), return_exceptions=True)
                    wall = time.perf_counter() - t0
                    pcie = sampler.stop()

                    errors = [repr(r) for r in results if isinstance(r, Exception)]
                    ok = [r for r in results if not isinstance(r, Exception)]
                    if not ok:
                        cells.append({
                            "target_prompt_tokens": length, "concurrency": conc,
                            "repeat": rep, "errors": errors, "wall_s": wall,
                            "pcie": pcie,
                        })
                        print(f"  {length:>7,} x{conc} r{rep}: ALL FAILED {errors[:1]}",
                              flush=True)
                        continue

                    ttfts = [r["ttft_s"] for r in ok]
                    itls = [v for r in ok for v in r["itls_s"]]
                    ptok = max(r["prompt_tokens"] for r in ok)
                    out_tok = sum(r["output_tokens"] for r in ok)
                    row = {
                        "target_prompt_tokens": length,
                        "actual_prompt_tokens": ptok,
                        "concurrency": conc,
                        "repeat": rep,
                        "wall_s": wall,
                        "ttft_min_s": min(ttfts),
                        "ttft_p50_s": _pct(ttfts, 0.5),
                        "ttft_max_s": max(ttfts),
                        # Fastest request's own prompt over its own TTFT: the
                        # cleanest per-request prefill rate at this concurrency.
                        "prefill_tok_per_s": ptok / min(ttfts) if min(ttfts) else 0.0,
                        "us_per_token": min(ttfts) * 1e6 / ptok if ptok else 0.0,
                        "itl_p50_ms": _pct(itls, 0.5) * 1e3,
                        "itl_p99_ms": _pct(itls, 0.99) * 1e3,
                        "output_tokens_total": out_tok,
                        "output_tok_per_s": out_tok / wall if wall else 0.0,
                        "errors": errors,
                        "pcie": pcie,
                    }
                    cells.append(row)
                    b70 = pcie.get("b70_gpu") or {}
                    print(
                        f"  {ptok:>7,} tok x{conc} r{rep}: "
                        f"TTFT min {row['ttft_min_s']:>7.2f}s  "
                        f"prefill {row['prefill_tok_per_s']:>7,.0f} tok/s  "
                        f"{row['us_per_token']:>6.1f} us/tok  "
                        f"ITL p50 {row['itl_p50_ms']:>6.2f}ms  "
                        f"B70 link best {b70.get('best_speed_gts')} GT/s "
                        f"x{b70.get('best_width')}",
                        flush=True,
                    )
    return {"target": args.target, "model": args.model,
            "max_tokens": args.max_tokens, "cells": cells,
            "pcie_static": pcie_state()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="unsloth/Qwen3.6-35B-A3B-NVFP4")
    parser.add_argument("--lengths", type=int, nargs="+",
                        default=[8192, 32768, 40960, 49152, 57344, 65536])
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    print(f"[probe] target={args.target} lengths={args.lengths} "
          f"concurrency={args.concurrency} repeats={args.repeats} "
          f"max_tokens={args.max_tokens}", flush=True)
    print(f"[probe] pcie at start: {json.dumps(pcie_state())}", flush=True)

    started = time.time()
    result = asyncio.run(run(args))
    result["wall_s"] = time.time() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"[probe] wrote {args.out} ({result['wall_s']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
