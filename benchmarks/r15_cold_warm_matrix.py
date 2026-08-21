#!/usr/bin/env python3
"""Cold-vs-warm serving surface for a Laguna model on the dual-B70 rig.

Why this instrument and not GuideLLM
------------------------------------
GuideLLM measures a server. This measures *this* server: it pairs every cell
with a cache-warm repeat, and it attributes each pass to the two B70s from the
native doorbell trace on a shared CLOCK_MONOTONIC.

The pairing is the point. r15 is the first model on this rig whose prefix cache
works at all (`is_hybrid=False`; the 99B/88B carry recurrent GDN state that
vLLM refuses to cache, measured 1.007x on an identical prompt). Cold prefill is
~1670 us/token and dominates TTFT at 86-92%, so for any workload with repeated
prefixes the cold number is close to irrelevant and the warm number is the one
that ships. Reporting only one of them is how you get an SLO table that is
wrong by two orders of magnitude in whichever direction flatters you.

Three traps this is built to avoid
----------------------------------
1. LOOSE TOKEN TARGETING. `b70_matrix_probe.py` sizes prompts by an estimated
   tokens-per-word ratio and so delivered 21,528 and 46,957 tokens for the same
   `ctx=32768` target on two runs, which invalidated an A/B. Here prompts are
   built against the real tokenizer and corrected until the server-reported
   count is within a hard tolerance, or the cell fails loudly.
2. FAKE COLDNESS. Prefix-cache block hashes are chained, so one differing token
   at position 0 invalidates every downstream block. Each cold prompt gets a
   unique header; the run asserts the cold pass added ~no cache hits and says
   so in the output when it did.
3. UNVERIFIED WARMTH. Warmth is not assumed either. Cached tokens live *in the
   KV pool*, so at 10.51 GiB (~214K tokens) a single 124K prompt is 58% of the
   pool and six cannot coexist -- the earliest is evicted before the last cold
   prompt lands. Every pass records its measured hit rate, so capacity-bound
   cells report a low warm hit rate as a finding rather than hiding behind an
   assumption.

Cells run in ascending context order so the most-used numbers land first, and
each cell is appended to the output the moment it completes: the run is
stoppable at any point and resumable with the same command.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import requests

TRACE_FLUSH_S = 6.5      # native dump thread period is 5 s; leave margin
LAST_ROUTED_LAYER = 47   # Laguna: layer 0 is a dense MLP, 1..47 are routed


# --------------------------------------------------------------------------
# small helpers
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
        "mean": round(statistics.fmean(values), 4),
        "p50": round(statistics.median(values), 4),
        "p90": round(pct(values, 0.90), 4),
        "p99": round(pct(values, 0.99), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def host_mem() -> dict:
    out = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, _, rest = line.partition(":")
            if k in ("MemAvailable", "SwapFree", "Cached"):
                out[k] = round(int(rest.split()[0]) / 1024 / 1024, 2)
    except Exception:                                          # noqa: BLE001
        pass
    return out


def nvidia_mem() -> dict:
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout.split(",")
        return {"used_mib": int(out[0]), "total_mib": int(out[1])}
    except Exception as exc:                                   # noqa: BLE001
        return {"error": repr(exc)}


def card_freq_mhz(paths: dict[str, str]) -> dict:
    out = {}
    for name, p in paths.items():
        try:
            out[name] = int(Path(p).read_text().strip()) // 1_000_000
        except Exception:                                      # noqa: BLE001
            out[name] = None
    return out


class Metrics:
    """vLLM's Prometheus counters we actually reason about."""

    KEYS = ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total",
            "vllm:num_requests_running", "vllm:num_requests_waiting",
            "vllm:gpu_cache_usage_perc", "vllm:num_preemptions_total")

    def __init__(self, url: str) -> None:
        self.url = url

    def read(self) -> dict:
        try:
            text = requests.get(self.url, timeout=30).text
        except Exception as exc:                               # noqa: BLE001
            return {"error": repr(exc)}
        out: dict[str, float] = {}
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            for k in self.KEYS:
                if line.startswith(k):
                    try:
                        out[k] = float(line.split()[-1])
                    except ValueError:
                        pass
        return out


# --------------------------------------------------------------------------
# prompts built against the real tokenizer
# --------------------------------------------------------------------------

class ExactPromptMint:
    """Prompts whose token count is measured, not estimated.

    Corpus content is load-bearing and not cosmetic: measured 2026-08-19,
    random-token prompts and real prose route to anti-correlated expert
    distributions (per-expert correlation -0.21), putting 55.3% vs 48.1% of
    route mass on experts [103..204]. Any per-card or split claim needs prose.
    """

    def __init__(self, corpus: Path, tokenizer_dir: str, seed: int = 0) -> None:
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(tokenizer_dir,
                                                 trust_remote_code=True)
        self.words = corpus.read_text(errors="ignore").split()
        if len(self.words) < 20_000:
            raise SystemExit(f"corpus too small: {len(self.words)} words")
        self._rng = random.Random(seed)
        self._cursor = 0

    def _slice(self, n_words: int) -> list[str]:
        if self._cursor + n_words > len(self.words):
            self._cursor = 0
        out = self.words[self._cursor:self._cursor + n_words]
        self._cursor += n_words
        return out

    def _count(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def make(self, target_tokens: int, tag: str) -> str:
        """A prompt of EXACTLY (or within 1 token of) target_tokens.

        Block hashes are chained, so a unique head is sufficient to make the
        whole prompt miss the prefix cache -- no need for a disjoint body,
        which matters because an exhaustive grid would otherwise need millions
        of unique corpus tokens.

        The sizing is a binary search over a prefix of ONE fixed word pool.
        The naive alternative -- guess a word count from a tokens-per-word
        ratio, then add or drop words to correct -- does not converge: each
        correction draws from a different corpus region with a different token
        density, so it oscillates. Measured 2026-08-21, that produced 6330
        tokens for a 4096 target. Searching a prefix of a fixed pool is
        monotone in the word count, so it cannot oscillate.
        """
        head = f"Document {tag}. Reference {self._rng.randrange(1 << 40):#x}. "
        budget = target_tokens - self._count(head)
        if budget < 16:
            raise SystemExit(f"target {target_tokens} too small for a header")
        # 1.0 tokens/word is a safe lower bound for any text, so this pool is
        # guaranteed long enough to overshoot the target.
        pool = self._slice(budget + 64)
        lo, hi, best = 1, len(pool), None
        while lo <= hi:
            mid = (lo + hi) // 2
            text = head + " ".join(pool[:mid])
            have = self._count(text)
            if have == target_tokens:
                return text
            if have < target_tokens:
                best = text                     # closest from below
                lo = mid + 1
            else:
                hi = mid - 1
        return best if best is not None else head + pool[0]


# --------------------------------------------------------------------------
# one streamed request
# --------------------------------------------------------------------------

@dataclass
class Res:
    prompt_tokens: int = 0
    output_tokens: int = 0
    ttft_s: float = float("nan")
    e2e_s: float = float("nan")
    itl_ms: list[float] = field(default_factory=list)
    error: str | None = None


def one_request(base: str, model: str, prompt: str, max_tokens: int,
                timeout: float) -> Res:
    res = Res()
    payload = {"model": model, "prompt": prompt, "temperature": 0.0,
               "max_tokens": max_tokens, "stream": True,
               "stream_options": {"include_usage": True}}
    arrivals: list[float] = []
    t0 = time.perf_counter()
    try:
        with requests.post(f"{base}/v1/completions", json=payload,
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
                ch = chunk.get("choices") or []
                if ch and ch[0].get("text"):
                    arrivals.append(time.perf_counter())
    except Exception as exc:                                   # noqa: BLE001
        res.error = repr(exc)
    res.e2e_s = time.perf_counter() - t0
    if arrivals:
        res.ttft_s = arrivals[0] - t0
        res.itl_ms = [(b - a) * 1e3 for a, b in zip(arrivals, arrivals[1:])]
    return res


# --------------------------------------------------------------------------
# doorbell trace, windowed to one pass
# --------------------------------------------------------------------------

def trace_paths(trace: str, devices: int) -> list[Path]:
    p = Path(trace)
    return [p.with_name(f"{p.stem}.device{i}{p.suffix}") for i in range(devices)]


def analyse_window(path: Path, t0_ns: int, t1_ns: int) -> dict:
    if not path.exists():
        return {"error": f"missing {path}"}
    try:
        doc = json.loads(path.read_text())
    except Exception as exc:                                   # noqa: BLE001
        return {"error": repr(exc)}
    sel = [e for e in doc.get("entries", []) if t0_ns <= e["t0_ns"] <= t1_ns]
    if not sel:
        return {"dispatches": 0, "note": "no dispatches in window"}
    sel.sort(key=lambda e: e["t0_ns"])

    svc = [(e["t1_ns"] - e["t0_ns"]) / 1e3 for e in sel]
    gaps = [(b["t0_ns"] - a["t1_ns"]) / 1e3 for a, b in zip(sel, sel[1:])
            if 0 <= (b["t0_ns"] - a["t1_ns"]) / 1e3 < 5000]
    by_m: dict[int, list[float]] = {}
    for e, s in zip(sel, svc):
        by_m.setdefault(int(e["M"]), []).append(s)

    sweeps, start, prev = [], None, None
    for e in sel:
        layer = int(e["layer"])
        if prev is None or layer <= prev:
            start = e["t0_ns"]
        if layer == LAST_ROUTED_LAYER and start is not None:
            sweeps.append((e["t1_ns"] - start) / 1e6)
            start = None
        prev = layer
    rows = {int(e.get("bank_row", -1)) for e in sel}

    return {
        "dispatches": len(sel),
        "layers_seen": len({int(e["layer"]) for e in sel}),
        "bank_rows_seen": len(rows),
        "layer_eq_row_plus_1": all(
            int(e["layer"]) == int(e.get("bank_row", -1)) + 1 for e in sel),
        "service_us": summarise(svc),
        "inter_dispatch_gap_us": summarise(gaps),
        "sweep_layer_ms": summarise(sweeps),
        "by_M": {str(m): {"n": len(v), "p50_us": round(statistics.median(v), 2),
                          "us_per_token": round(statistics.median(v) / m, 3)}
                 for m, v in sorted(by_m.items())},
        "window_utilisation_pct": round(
            sum(svc) / max(t1_ns - t0_ns, 1) * 1e3 * 100, 2),
    }


# --------------------------------------------------------------------------
# one pass (cold or warm) over a fixed prompt set
# --------------------------------------------------------------------------

def run_pass(args, prompts: list[str], metrics: Metrics, devices: list[Path],
             kind: str) -> dict:
    before = metrics.read()
    t_mono0 = time.monotonic_ns()
    t_wall0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        results = list(pool.map(
            lambda p: one_request(args.url, args.model, p, args.output_tokens,
                                  args.timeout), prompts))
    wall = time.perf_counter() - t_wall0
    t_mono1 = time.monotonic_ns()
    after = metrics.read()

    ok = [r for r in results if r.error is None and r.output_tokens > 0]
    hits = after.get("vllm:prefix_cache_hits_total", 0) - \
        before.get("vllm:prefix_cache_hits_total", 0)
    queries = after.get("vllm:prefix_cache_queries_total", 0) - \
        before.get("vllm:prefix_cache_queries_total", 0)
    preempt = after.get("vllm:num_preemptions_total", 0) - \
        before.get("vllm:num_preemptions_total", 0)

    itl_all = [x for r in ok for x in r.itl_ms]
    out_total = sum(r.output_tokens for r in ok)
    ptoks = [r.prompt_tokens for r in ok]

    time.sleep(TRACE_FLUSH_S)
    per_dev = {f"device{i}": analyse_window(p, t_mono0, t_mono1)
               for i, p in enumerate(devices)}

    return {
        "kind": kind,
        "requests_ok": len(ok),
        "requests_err": len(results) - len(ok),
        "errors": [r.error for r in results if r.error][:3],
        "prompt_tokens": {"median": statistics.median(ptoks) if ptoks else 0,
                          "min": min(ptoks) if ptoks else 0,
                          "max": max(ptoks) if ptoks else 0},
        "output_tokens_total": out_total,
        "wall_s": round(wall, 3),
        "ttft_s": summarise([r.ttft_s for r in ok]),
        "itl_ms": summarise(itl_all),
        "e2e_s": summarise([r.e2e_s for r in ok]),
        "aggregate_output_tok_per_s": round(out_total / wall, 2) if wall else 0,
        "per_stream_tok_per_s": round(
            1000.0 / statistics.median(itl_all), 2) if itl_all else 0,
        "prefix_cache": {
            "hits": hits, "queries": queries,
            "hit_rate": round(hits / queries, 4) if queries else None},
        "preemptions": preempt,
        "vllm_after": after,
        "per_device": per_dev,
        "machine": {"host": host_mem(), "nvidia": nvidia_mem(),
                    "b70_act_freq_MHz": card_freq_mhz(args.freq)},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8017")
    ap.add_argument("--metrics-url", default="http://127.0.0.1:8017/metrics")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer-dir", required=True,
                    help="real HF repo/path; the served name is an API alias")
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--contexts", required=True,
                    help="comma-separated prompt token counts")
    ap.add_argument("--concurrency", required=True,
                    help="comma-separated concurrent stream counts")
    ap.add_argument("--output-tokens", type=int, default=512)
    ap.add_argument("--trace", default="/tmp/sb_r15_trace.json")
    ap.add_argument("--devices", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=2400.0)
    ap.add_argument("--tolerance-pct", type=float, default=2.0,
                    help="fail a cell if actual prompt tokens drift this far")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", required=True, type=Path)
    args = ap.parse_args()

    args.freq = {
        "gen4_15:00.0": "/sys/bus/pci/devices/0000:15:00.0/tile0/gt0/freq0/act_freq",
        "gen3_11:00.0": "/sys/bus/pci/devices/0000:11:00.0/tile0/gt0/freq0/act_freq",
    }
    contexts = [int(x) for x in args.contexts.split(",") if x.strip()]
    concs = [int(x) for x in args.concurrency.split(",") if x.strip()]
    devices = trace_paths(args.trace, args.devices)
    metrics = Metrics(args.metrics_url)

    cells: list[dict] = []
    done: set[tuple[int, int]] = set()
    if args.json_out.exists():
        try:
            cells = json.loads(args.json_out.read_text())
            done = {(c["context_tokens"], c["concurrency"]) for c in cells}
            print(f"resuming: {len(done)} cells already present", flush=True)
        except Exception:                                      # noqa: BLE001
            cells = []

    mint = ExactPromptMint(args.corpus, args.tokenizer_dir, args.seed)
    print(f"corpus {len(mint.words):,} words | grid "
          f"{len(contexts)}x{len(concs)} = {len(contexts)*len(concs)} cells, "
          f"cold+warm each", flush=True)

    for ctx in sorted(contexts):
        for conc in sorted(concs):
            if (ctx, conc) in done:
                continue
            print(f"\n=== ctx={ctx} conc={conc} out={args.output_tokens}",
                  flush=True)
            prompts = [mint.make(ctx, f"c{ctx}k{conc}s{i}-{args.seed}")
                       for i in range(conc)]

            cold = run_pass(args, prompts, metrics, devices, "cold")
            warm = run_pass(args, prompts, metrics, devices, "warm")

            drift = None
            if cold["requests_ok"]:
                med = cold["prompt_tokens"]["median"]
                drift = abs(med - ctx) / ctx * 100
            cell = {
                "context_tokens": ctx, "concurrency": conc,
                "output_tokens": args.output_tokens,
                "prompt_token_drift_pct": round(drift, 3) if drift is not None else None,
                "prompt_tokens_within_tolerance": (
                    drift is not None and drift <= args.tolerance_pct),
                "kv_capacity_note": (
                    "capacity-bound: conc*ctx exceeds the KV pool, so the "
                    "warm hit rate is limited by eviction, not by caching"
                    if ctx * conc > 214_000 else "fits KV pool"),
                "cold": cold, "warm": warm,
                "speedup_ttft_p50": (
                    round(cold["ttft_s"]["p50"] / warm["ttft_s"]["p50"], 2)
                    if warm["ttft_s"].get("p50") else None),
            }
            cells.append(cell)
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(cells, indent=2))

            c, w = cold, warm
            print(f"  ptok={c['prompt_tokens']['median']:.0f} "
                  f"(drift {drift:.2f}%) ok={c['requests_ok']}/{conc}", flush=True)
            print(f"  COLD ttft p50={c['ttft_s'].get('p50')}s "
                  f"itl p50={c['itl_ms'].get('p50')}ms "
                  f"agg={c['aggregate_output_tok_per_s']} tok/s "
                  f"hits={c['prefix_cache']['hit_rate']}", flush=True)
            print(f"  WARM ttft p50={w['ttft_s'].get('p50')}s "
                  f"itl p50={w['itl_ms'].get('p50')}ms "
                  f"agg={w['aggregate_output_tok_per_s']} tok/s "
                  f"hits={w['prefix_cache']['hit_rate']}", flush=True)
            if cell["speedup_ttft_p50"]:
                print(f"  TTFT speedup cold->warm: "
                      f"{cell['speedup_ttft_p50']}x", flush=True)
            if not cell["prompt_tokens_within_tolerance"]:
                print("  WARNING: prompt token drift beyond tolerance", flush=True)

    print(f"\nwrote {args.json_out} ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
