#!/usr/bin/env python3
"""One instrument for the whole 99B dual-B70 serving surface.

Every other probe in this tree answers one question. This one exists because
the questions are coupled: TTFT depends on chunked-prefill scheduling, which
depends on how many doorbells a batch costs, which depends on how the experts
are split across two cards of different PCIe generations, which changes which
card gates the layer -- and none of that is visible if you measure request
latency alone or dispatch service alone.

Per cell (context x concurrency x output tokens) it records, on one clock:

  client        streamed TTFT, per-token ITL series, end-to-end latency,
                server-reported prompt/output token counts
  per card      dispatch count, M histogram, service percentiles, the
                inter-dispatch gap (the 5090-side cost between doorbells),
                and full 48-layer sweep durations reconstructed from the
                layer sequence
  machine       host MemAvailable/SwapFree, both B70 act_freq, 5090 memory,
                and vLLM's own KV/queue gauges

Attribution is exact rather than inferred: the native trace stamps
``CLOCK_MONOTONIC`` (verified against ``time.monotonic_ns()``), so a cell
selects precisely the dispatches whose start falls inside its own window.
The trace ring is snapshotted by a background thread every ~5 s, so each
cell waits for a flush before reading -- without that wait the tail of a
cell's own work is missing and service percentiles skew fast.

Two properties are load-bearing and easy to get wrong:

* **Prompts must not share a prefix.** With prefix caching on (the default)
  a shared prefix turns prefill into a cache lookup and the TTFT slope
  collapses. Words are drawn without replacement and every prompt starts
  with a unique token.
* **Prompt length must be measured, not assumed.** Length is calibrated
  against the server's own ``usage.prompt_tokens``, because a synthetic
  vocabulary's tokens-per-word is a property of the tokenizer.

Usage:
  .venv/bin/python benchmarks/b70_matrix_probe.py \
      --model shooting-brake-99b --url http://127.0.0.1:8017 \
      --cells 1024x1x64,8192x1x64,8192x2x64 \
      --label even-102-102 \
      --json-out benchmarks/results/b70_gemv_audit/99b_matrix.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import requests

NUM_LAYERS = 48
TRACE_FLUSH_S = 6.5          # dump thread period is 5 s; leave margin
CARD_FREQ = {
    "gen4_0000:15:00.0": "/sys/bus/pci/devices/0000:15:00.0/tile0/gt0/freq0/act_freq",
    "gen3_0000:11:00.0": "/sys/bus/pci/devices/0000:11:00.0/tile0/gt0/freq0/act_freq",
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    return s[min(len(s) - 1, max(0, int(q * (len(s) - 1))))]


def summarise(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 3),
        "p50": round(statistics.median(values), 3),
        "p90": round(pct(values, 0.90), 3),
        "p99": round(pct(values, 0.99), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def read_int(path: str) -> int | None:
    try:
        return int(Path(path).read_text().split()[0])
    except (OSError, ValueError):
        return None


def host_mem() -> dict:
    out = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key = line.split(":")[0]
        if key in ("MemAvailable", "SwapFree", "MemFree", "Cached"):
            out[key + "_GiB"] = round(int(line.split()[1]) / 2**20, 3)
    return out


def nvidia_mem() -> dict:
    try:
        raw = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,clocks.sm",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15).stdout.strip()
        used, total, sm = (int(x) for x in raw.split(","))
        return {"used_MiB": used, "total_MiB": total, "sm_clock_MHz": sm}
    except Exception as exc:                                   # noqa: BLE001
        return {"error": repr(exc)}


def vllm_gauges(url: str) -> dict:
    """A small, stable subset of vLLM's Prometheus gauges."""
    wanted = ("vllm:num_requests_running", "vllm:num_requests_waiting",
              "vllm:gpu_cache_usage_perc", "vllm:gpu_prefix_cache_hit_rate",
              "vllm:prompt_tokens_total", "vllm:generation_tokens_total")
    try:
        text = requests.get(f"{url}/metrics", timeout=10).text
    except requests.RequestException as exc:
        return {"error": repr(exc)}
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for key in wanted:
            if line.startswith(key):
                try:
                    out[key] = float(line.split()[-1])
                except ValueError:
                    pass
    return out


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------

class PromptMint:
    """Prompts calibrated to the server's tokenizer.

    ``corpus`` selects what the tokens *are*, which is not cosmetic: measured
    2026-08-19, random-token prompts and real prose route to anti-correlated
    expert distributions (per-expert correlation -0.21). Random tokens put
    55.3% of route mass on experts [103..204] while prose puts 48.1% there, so
    a dual-card split "optimised" against synthetic prompts is 6.4% *worse*
    than an even split on prose. Synthetic prompts remain correct for
    total-cost questions -- top-k is 8 per token either way -- but any
    per-card, split, or balance claim must come from a real corpus.

    Prefix disjointness is preserved in both modes: prose is sliced at
    non-overlapping offsets, random prompts draw words without replacement.
    """

    def __init__(self, seed: int = 0, pool: int = 400_000,
                 corpus: str | None = None) -> None:
        self._rng = random.Random(seed)
        self._pool = [f"z{i}q" for i in range(pool)]
        self._rng.shuffle(self._pool)
        self._cursor = 0
        self.tokens_per_word = 3.0
        self._corpus: list[str] | None = None
        if corpus:
            text = Path(corpus).read_text(errors="ignore")
            self._corpus = text.split()
            if len(self._corpus) < 512:
                raise SystemExit(f"corpus {corpus} too small: {len(self._corpus)} words")
            self.tokens_per_word = 1.4        # prose tokenises far denser

    def _take(self, n: int) -> list[str]:
        source = self._corpus if self._corpus is not None else self._pool
        if self._cursor + n > len(source):
            if self._corpus is None:
                self._rng.shuffle(source)
            self._cursor = 0
        chunk = source[self._cursor:self._cursor + n]
        self._cursor += n
        return chunk

    def observe(self, words: int, tokens: int) -> None:
        if words > 0 and tokens > 0:
            self.tokens_per_word = tokens / words

    def make(self, target_tokens: int) -> tuple[str, int]:
        words = max(1, round(target_tokens / self.tokens_per_word))
        return " ".join(self._take(words)), words


# --------------------------------------------------------------------------
# one streamed request
# --------------------------------------------------------------------------

@dataclass
class RequestResult:
    prompt_tokens: int = 0
    output_tokens: int = 0
    ttft_s: float = float("nan")
    e2e_s: float = float("nan")
    itl_ms: list[float] = field(default_factory=list)
    error: str | None = None


def one_request(url: str, model: str, prompt: str, max_tokens: int,
                timeout: float) -> RequestResult:
    res = RequestResult()
    payload = {"model": model, "prompt": prompt, "temperature": 0.0,
               "max_tokens": max_tokens, "stream": True,
               "stream_options": {"include_usage": True}}
    arrivals: list[float] = []
    t0 = time.perf_counter()
    try:
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
                    res.prompt_tokens = int(chunk["usage"].get("prompt_tokens", 0))
                    res.output_tokens = int(chunk["usage"].get("completion_tokens", 0))
                choices = chunk.get("choices") or []
                if choices and choices[0].get("text"):
                    arrivals.append(time.perf_counter())
    except Exception as exc:                                   # noqa: BLE001
        res.error = repr(exc)
    res.e2e_s = time.perf_counter() - t0
    if arrivals:
        res.ttft_s = arrivals[0] - t0
        res.itl_ms = [(b - a) * 1e3 for a, b in zip(arrivals, arrivals[1:])]
    return res


# --------------------------------------------------------------------------
# trace attribution
# --------------------------------------------------------------------------

def device_trace_paths(trace: str, devices: int) -> list[Path]:
    p = Path(trace)
    return [p.with_name(f"{p.stem}.device{i}{p.suffix}") for i in range(devices)]


def analyse_window(path: Path, t_start_ns: int, t_end_ns: int) -> dict:
    """Dispatch statistics for exactly the dispatches inside a cell."""
    if not path.exists():
        return {"error": f"missing {path}"}
    doc = json.loads(path.read_text())
    sel = [e for e in doc["entries"] if t_start_ns <= e["t0_ns"] <= t_end_ns]
    if not sel:
        return {"dispatches": 0, "note": "no dispatches in window"}
    sel.sort(key=lambda e: e["t0_ns"])

    service_us = [(e["t1_ns"] - e["t0_ns"]) / 1e3 for e in sel]
    gaps_us = [(b["t0_ns"] - a["t1_ns"]) / 1e3 for a, b in zip(sel, sel[1:])
               if 0 <= (b["t0_ns"] - a["t1_ns"]) / 1e3 < 5000]
    by_m: dict[int, list[float]] = {}
    for e, svc in zip(sel, service_us):
        by_m.setdefault(int(e["M"]), []).append(svc)

    # A full sweep is a run of layer indices climbing 0 -> 47; its duration is
    # the per-token cost of this card's whole share of the model.
    sweeps_ms: list[float] = []
    start_ns, prev_layer = None, None
    for e in sel:
        layer = int(e["layer"])
        if prev_layer is None or layer <= prev_layer:
            start_ns = e["t0_ns"]
        if layer == NUM_LAYERS - 1 and start_ns is not None:
            sweeps_ms.append((e["t1_ns"] - start_ns) / 1e6)
            start_ns = None
        prev_layer = layer

    return {
        "dispatches": len(sel),
        "layers_seen": len({int(e["layer"]) for e in sel}),
        "service_us": summarise(service_us),
        "inter_dispatch_gap_us": summarise(gaps_us),
        "sweep_48layer_ms": summarise(sweeps_ms),
        "by_M": {str(m): {"n": len(v), "p50_us": round(statistics.median(v), 2),
                          "us_per_token": round(statistics.median(v) / m, 3)}
                 for m, v in sorted(by_m.items())},
        "window_utilisation_pct": round(
            sum(service_us) / max(t_end_ns - t_start_ns, 1) * 1e3 * 100, 2),
    }


# --------------------------------------------------------------------------
# a cell
# --------------------------------------------------------------------------

def run_cell(args, mint: PromptMint, ctx: int, conc: int, out_tok: int) -> dict:
    prompts = []
    for _ in range(conc):
        prompt, words = mint.make(ctx)
        prompts.append((prompt, words))

    gauges_before = vllm_gauges(args.url)
    time.sleep(1.0)                       # let prior cell's dispatches drain
    t_start = time.monotonic_ns()
    wall0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=conc) as pool:
        futures = [pool.submit(one_request, args.url, args.model, p, out_tok,
                               args.timeout) for p, _ in prompts]
        results = [f.result() for f in futures]
    wall = time.perf_counter() - wall0
    t_end = time.monotonic_ns()

    for (_, words), r in zip(prompts, results):
        mint.observe(words, r.prompt_tokens)

    time.sleep(TRACE_FLUSH_S)
    per_device = {f"device{i}": analyse_window(p, t_start, t_end)
                  for i, p in enumerate(device_trace_paths(args.trace, args.devices))}

    ok = [r for r in results if r.error is None and r.output_tokens]
    ttfts = [r.ttft_s for r in ok if r.ttft_s == r.ttft_s]
    itls = [v for r in ok for v in r.itl_ms]
    prompt_toks = [r.prompt_tokens for r in ok]
    out_toks = sum(r.output_tokens for r in ok)

    cell = {
        "label": args.label,
        "target": {"context_tokens": ctx, "concurrency": conc,
                   "output_tokens": out_tok},
        "actual": {
            "requests_ok": len(ok), "requests_err": len(results) - len(ok),
            "errors": [r.error for r in results if r.error][:3],
            "prompt_tokens_median": statistics.median(prompt_toks) if prompt_toks else None,
            "output_tokens_total": out_toks,
            "wall_s": round(wall, 4),
            "aggregate_output_tok_per_s": round(out_toks / wall, 2) if wall else None,
        },
        "ttft_s": summarise(ttfts),
        "itl_ms": summarise(itls),
        "e2e_s": summarise([r.e2e_s for r in ok]),
        "per_device": per_device,
        "machine": {
            "host": host_mem(),
            "nvidia": nvidia_mem(),
            "b70_act_freq_MHz": {k: read_int(v) for k, v in CARD_FREQ.items()},
            "vllm_before": gauges_before,
            "vllm_after": vllm_gauges(args.url),
        },
    }

    # Derived: which card gated the layer, and what share of TTFT it explains.
    svc = {d: v.get("service_us", {}).get("p50") for d, v in per_device.items()
           if isinstance(v, dict)}
    if all(isinstance(x, (int, float)) for x in svc.values()) and svc:
        slow = max(svc, key=lambda k: svc[k])
        cell["derived"] = {
            "gating_device": slow,
            "service_p50_ratio": round(max(svc.values()) / max(min(svc.values()), 1e-9), 4),
        }
    return cell


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8017")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cells", required=True,
                    help="comma list of CTXxCONCxOUT, e.g. 1024x1x64,8192x2x32")
    ap.add_argument("--label", default="unlabelled")
    ap.add_argument("--trace", default="/tmp/sb_99b_trace.json")
    ap.add_argument("--devices", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--corpus", default=None,
                    help="text file of real prose; omit for random tokens. "
                         "REQUIRED for any per-card/split/balance claim -- "
                         "random tokens route anti-correlated to prose.")
    ap.add_argument("--warmup", action="store_true",
                    help="one small request first so JIT/allocator is not billed")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    cells = []
    for spec in args.cells.split(","):
        if not spec.strip():
            continue
        ctx, conc, out = (int(x) for x in spec.lower().split("x"))
        cells.append((ctx, conc, out))

    mint = PromptMint(seed=args.seed, corpus=args.corpus)
    if args.warmup:
        p, w = mint.make(64)
        r = one_request(args.url, args.model, p, 4, args.timeout)
        mint.observe(w, r.prompt_tokens)
        print(f"warmup: {r.prompt_tokens} prompt tokens, ttft={r.ttft_s*1e3:.0f} ms")

    out = []
    for ctx, conc, out_tok in cells:
        print(f"\n=== cell ctx={ctx} conc={conc} out={out_tok}", flush=True)
        cell = run_cell(args, mint, ctx, conc, out_tok)
        out.append(cell)
        a, t, i = cell["actual"], cell["ttft_s"], cell["itl_ms"]
        print(f"  ok={a['requests_ok']}/{a['requests_ok']+a['requests_err']} "
              f"ptok={a['prompt_tokens_median']} "
              f"TTFT p50={t.get('p50')}s ITL p50={i.get('p50')}ms "
              f"agg={a['aggregate_output_tok_per_s']} tok/s", flush=True)
        for dev, d in cell["per_device"].items():
            if d.get("dispatches"):
                print(f"    {dev}: n={d['dispatches']:>6} "
                      f"svc p50={d['service_us']['p50']:>9.1f}us "
                      f"gap p50={d['inter_dispatch_gap_us'].get('p50')}us "
                      f"sweep p50={d['sweep_48layer_ms'].get('p50')}ms "
                      f"util={d['window_utilisation_pct']}%", flush=True)
        if a["errors"]:
            print(f"    errors: {a['errors']}", flush=True)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        prev = []
        if args.json_out.exists():
            try:
                prev = json.loads(args.json_out.read_text())
            except json.JSONDecodeError:
                prev = []
        args.json_out.write_text(json.dumps(prev + out, indent=2) + "\n")
        print(f"\nappended {len(out)} cells -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
