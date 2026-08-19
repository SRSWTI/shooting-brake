#!/usr/bin/env python3
"""Separate the two terms of B70 prefill cost, and say which flag can move them.

Prefill on this rig is a chunked doorbell: vLLM hands the routed-expert
partial a batch of M tokens, the plugin splits it at
``SHOOTING_BRAKE_B70_MAX_BATCH`` and rings each card once per chunk per
layer. Total B70 service for T prompt tokens is therefore

    service(T, C) = ceil(T / C) * (a + b * C) * num_layers

where ``a`` is the fixed per-dispatch cost (doorbell + staging + poller
wakeup) and ``b`` the marginal per-token cost (the GEMV weight re-read).
Only ``a`` responds to the chunk cap: the C-dependent share of the total is
``(T/C)*a / ((T/C)*a + b*T)``. Fitting a and b is thus the whole question
of whether raising MAX_BATCH is worth a reboot -- and it is answerable from
one trace, without one.

Two modes, deliberately separate because they fail differently:

  trace   Fit a and b per physical card from the native dispatch trace
          (``SHOOTING_BRAKE_B70_TRACE_DUMP``, one file per device). Groups
          entries by their real M, so a wrapped ring or a mixed
          decode/prefill history stays interpretable. Reports the
          MAX_BATCH ceiling directly. No server load, no GPU use.

  ladder  Measure true streamed TTFT against prompt length on a live
          server, then report what fraction the trace model accounts for.
          Prompts are sampled without replacement from a large vocabulary
          so that no two requests share a prefix: with prefix caching on
          (the default) a shared prefix silently converts prefill into a
          cache hit and the slope collapses. The prefix-cache counters are
          read before and after and recorded, so the claim is checkable
          rather than asserted.

Usage:
  .venv/bin/python benchmarks/b70_prefill_cost.py trace \
      --trace /tmp/sb_99b_trace.json --devices 2 \
      --json-out benchmarks/results/b70_gemv_audit/99b_prefill_cost.json

  .venv/bin/python benchmarks/b70_prefill_cost.py ladder \
      --url http://127.0.0.1:8017 --model shooting-brake-99b \
      --tokens 1024,2048,4096,8192 --repeats 2 \
      --json-out benchmarks/results/b70_gemv_audit/99b_prefill_ladder.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import requests

NUM_LAYERS = 48


# --------------------------------------------------------------------------
# trace mode
# --------------------------------------------------------------------------

def _device_trace_paths(trace: str, devices: int) -> list[Path]:
    """Per-device dump paths, matching B70Poller's suffix rule.

    The poller writes ``<stem>.device<N><ext>``; both pollers writing one
    undecorated path is the collision this suffix exists to prevent, so a
    bare path is accepted only as a fallback for pre-fix artifacts.
    """
    p = Path(trace)
    out = [p.with_name(f"{p.stem}.device{i}{p.suffix}") for i in range(devices)]
    missing = [q for q in out if not q.exists()]
    if missing and p.exists():
        return [p]
    if missing:
        raise SystemExit(f"missing per-device traces: {[str(q) for q in missing]}")
    return out


def _fit_affine(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares ``y = a + b*x``; explicit so numpy stays optional."""
    n = len(xs)
    if n < 2:
        raise SystemExit("need >= 2 distinct M values to separate a from b")
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0.0:
        raise SystemExit("all dispatches share one M; cannot separate a from b")
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, max(0, int(q * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def analyse_trace(path: Path, tail: int | None) -> dict:
    entries = json.loads(path.read_text())["entries"]
    if tail:
        entries = entries[-tail:]
    by_m: dict[int, list[float]] = {}
    for e in entries:
        by_m.setdefault(int(e["M"]), []).append((e["t1_ns"] - e["t0_ns"]) / 1e3)

    classes = []
    for m in sorted(by_m):
        v = sorted(by_m[m])
        classes.append({
            "M": m,
            "n": len(v),
            "p50_us": round(statistics.median(v), 2),
            "p90_us": round(_percentile(v, 0.90), 2),
            "us_per_token": round(statistics.median(v) / m, 3),
        })

    a, b = _fit_affine([float(c["M"]) for c in classes],
                       [c["p50_us"] for c in classes])
    resid = [{"M": c["M"], "pred_us": round(a + b * c["M"], 1),
              "meas_us": c["p50_us"],
              "err_pct": round((a + b * c["M"] - c["p50_us"]) / c["p50_us"] * 100, 3)}
             for c in classes]
    return {
        "path": str(path),
        "entries": len(entries),
        "layers": NUM_LAYERS,
        "classes": classes,
        "fit": {"fixed_us_per_dispatch": round(a, 2),
                "marginal_us_per_token": round(b, 4)},
        "fit_residuals": resid,
    }


def chunk_cap_ceiling(a: float, b: float, tokens: list[int],
                      caps: list[int]) -> list[dict]:
    """What each chunk cap costs, and the most any cap change can return."""
    rows = []
    for T in tokens:
        base = None
        for C in caps:
            n = -(-T // C)
            total_s = n * (a + b * C) * NUM_LAYERS / 1e6
            fixed_s = n * a * NUM_LAYERS / 1e6
            base = total_s if base is None else base
            rows.append({
                "prompt_tokens": T,
                "chunk_cap": C,
                "chunks": n,
                "b70_service_s": round(total_s, 4),
                "fixed_share_pct": round(fixed_s / total_s * 100, 3),
                "delta_vs_first_cap_pct": round((total_s - base) / base * 100, 3),
            })
    return rows


# --------------------------------------------------------------------------
# ladder mode
# --------------------------------------------------------------------------

def _prefix_cache_counters(url: str) -> dict[str, float]:
    """vLLM's prefix-cache counters; absence is recorded, not fatal."""
    try:
        text = requests.get(f"{url}/metrics", timeout=10).text
    except requests.RequestException as exc:
        return {"error": repr(exc)}
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for key in ("vllm:prefix_cache_queries_total",
                    "vllm:prefix_cache_hits_total"):
            if line.startswith(key):
                try:
                    out[key] = out.get(key, 0.0) + float(line.split()[-1])
                except ValueError:
                    pass
    return out


class PromptMint:
    """Unique, prefix-disjoint prompts of a requested token length.

    Words are drawn without replacement from a large pool and the first
    word of every prompt is unique, so no two prompts share a token-0
    prefix and prefix caching cannot turn a prefill into a lookup. Length
    is calibrated against the server's own tokenizer count (returned in
    ``usage.prompt_tokens``) rather than assumed from word count.
    """

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._pool = [f"q{i}x" for i in range(200_000)]
        self._rng.shuffle(self._pool)
        self._cursor = 0
        self._tokens_per_word = 3.0     # refined from the first measurement

    def _take(self, n: int) -> list[str]:
        if self._cursor + n > len(self._pool):
            self._rng.shuffle(self._pool)
            self._cursor = 0
        out = self._pool[self._cursor:self._cursor + n]
        self._cursor += n
        return out

    def observe(self, words: int, tokens: int) -> None:
        if words > 0 and tokens > 0:
            self._tokens_per_word = tokens / words

    def make(self, target_tokens: int) -> tuple[str, int]:
        words = max(1, round(target_tokens / self._tokens_per_word))
        return " ".join(self._take(words)), words


def stream_ttft(url: str, model: str, prompt: str,
                max_tokens: int, timeout: float) -> tuple[float, float, dict]:
    """Return (ttft_s, wall_s, usage) from a real SSE stream.

    TTFT is the arrival of the first chunk carrying generated text, which
    is the only definition that matches what a client experiences; a
    non-streamed request cannot distinguish prefill from decode.
    """
    payload = {"model": model, "prompt": prompt, "temperature": 0.0,
               "max_tokens": max_tokens, "stream": True,
               "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    ttft = float("nan")
    usage: dict = {}
    with requests.post(f"{url}/v1/completions", json=payload,
                       stream=True, timeout=timeout) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            body = raw[6:]
            if body == "[DONE]":
                break
            chunk = json.loads(body)
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if choices and choices[0].get("text") and ttft != ttft:
                ttft = time.perf_counter() - t0
    return ttft, time.perf_counter() - t0, usage


def run_ladder(args) -> dict:
    mint = PromptMint(seed=args.seed)
    before = _prefix_cache_counters(args.url)

    # One warm request so JIT/allocator state is not billed to the ladder.
    warm, _ = mint.make(64)
    stream_ttft(args.url, args.model, warm, 1, args.timeout)

    samples = []
    for target in args.tokens:
        for rep in range(args.repeats):
            prompt, words = mint.make(target)
            ttft, wall, usage = stream_ttft(args.url, args.model, prompt,
                                            args.max_tokens, args.timeout)
            pt = int(usage.get("prompt_tokens", 0))
            mint.observe(words, pt)
            samples.append({"target_tokens": target, "rep": rep,
                            "prompt_tokens": pt,
                            "ttft_s": round(ttft, 6),
                            "wall_s": round(wall, 6),
                            "us_per_prompt_token": round(ttft / pt * 1e6, 2) if pt else None})
            print(f"  target={target:>6} actual={pt:>6} ttft={ttft:>8.3f}s "
                  f"wall={wall:>8.3f}s {ttft/max(pt,1)*1e6:>7.1f} us/token", flush=True)

    after = _prefix_cache_counters(args.url)
    dq = after.get("vllm:prefix_cache_queries_total", 0) - \
        before.get("vllm:prefix_cache_queries_total", 0)
    dh = after.get("vllm:prefix_cache_hits_total", 0) - \
        before.get("vllm:prefix_cache_hits_total", 0)

    grouped: dict[int, list[float]] = {}
    for s in samples:
        if s["prompt_tokens"]:
            grouped.setdefault(s["prompt_tokens"], []).append(s["ttft_s"])
    pts = sorted(grouped)
    a_l, b_l = _fit_affine([float(p) for p in pts],
                           [statistics.median(grouped[p]) for p in pts])
    return {
        "samples": samples,
        "prefix_cache": {"queries_delta": dq, "hits_delta": dh,
                         "hit_rate_pct": round(dh / dq * 100, 3) if dq else None},
        "fit": {"intercept_s": round(a_l, 4),
                "marginal_us_per_token": round(b_l * 1e6, 3),
                "marginal_us_per_token_per_layer": round(b_l * 1e6 / NUM_LAYERS, 4)},
    }


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    t = sub.add_parser("trace", help="fit a/b from the native dispatch trace")
    t.add_argument("--trace", default="/tmp/sb_99b_trace.json")
    t.add_argument("--devices", type=int, default=2)
    t.add_argument("--tail", type=int, default=None,
                   help="use only the last N entries (a wrapped ring mixes regimes)")
    t.add_argument("--tokens", default="1024,8192,32768")
    t.add_argument("--caps", default="256,512,1024,2048")
    t.add_argument("--json-out", type=Path, default=None)

    l = sub.add_parser("ladder", help="streamed TTFT vs prompt length, live server")
    l.add_argument("--url", default="http://127.0.0.1:8017")
    l.add_argument("--model", required=True)
    l.add_argument("--tokens", default="1024,2048,4096,8192")
    l.add_argument("--repeats", type=int, default=2)
    l.add_argument("--max-tokens", type=int, default=1)
    l.add_argument("--timeout", type=float, default=900.0)
    l.add_argument("--seed", type=int, default=0)
    l.add_argument("--json-out", type=Path, default=None)

    args = ap.parse_args()
    ints = lambda s: [int(x) for x in str(s).split(",") if x.strip()]

    if args.mode == "trace":
        per_device = [analyse_trace(p, args.tail)
                      for p in _device_trace_paths(args.trace, args.devices)]
        for d in per_device:
            f = d["fit"]
            print(f"{d['path']}: {d['entries']} entries")
            for c in d["classes"]:
                print(f"   M={c['M']:>5} n={c['n']:>6} p50={c['p50_us']:>10.1f}us "
                      f"{c['us_per_token']:>8.2f} us/token")
            print(f"   fit: {f['fixed_us_per_dispatch']:.1f} us fixed + "
                  f"{f['marginal_us_per_token']:.3f} us/token")
            worst = max(abs(r["err_pct"]) for r in d["fit_residuals"])
            print(f"   max fit error {worst:.3f}%\n")

        # The card that finishes last sets the layer's cost: CUDA waits on both.
        a = max(d["fit"]["fixed_us_per_dispatch"] for d in per_device)
        b = max(d["fit"]["marginal_us_per_token"] for d in per_device)
        rows = chunk_cap_ceiling(a, b, ints(args.tokens), ints(args.caps))
        print(f"worst-card model: a={a:.1f} us, b={b:.3f} us/token")
        for r in rows:
            print(f"   T={r['prompt_tokens']:>6} C={r['chunk_cap']:>5}: "
                  f"{r['chunks']:>4} chunks  {r['b70_service_s']:>8.3f} s  "
                  f"fixed {r['fixed_share_pct']:>6.3f}%  "
                  f"vs C={ints(args.caps)[0]}: {r['delta_vs_first_cap_pct']:+.3f}%")
        result = {"kind": "b70_prefill_cost_trace", "per_device": per_device,
                  "worst_card": {"fixed_us_per_dispatch": round(a, 2),
                                 "marginal_us_per_token": round(b, 4)},
                  "chunk_cap_scan": rows}
    else:
        args.tokens = ints(args.tokens)
        result = {"kind": "b70_prefill_cost_ladder", "url": args.url,
                  "model": args.model, **run_ladder(args)}
        f = result["fit"]
        pc = result["prefix_cache"]
        print(f"\nfit: {f['marginal_us_per_token']:.1f} us/token "
              f"({f['marginal_us_per_token_per_layer']:.2f} us/token/layer) "
              f"+ {f['intercept_s']*1e3:.0f} ms")
        print(f"prefix cache hit rate during ladder: {pc['hit_rate_pct']}% "
              f"(queries +{pc['queries_delta']}, hits +{pc['hits_delta']})")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
