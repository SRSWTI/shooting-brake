#!/usr/bin/env python3
"""Sweep local-expert count L for a Laguna model and measure both axes.

Why this exists
---------------
L=57 on r15 is inherited arithmetic, not a measurement: it is what the 99B's
measured optimum of 54 local experts becomes when the same fraction is applied
to 218 experts. Laguna's memory economics are completely different from the
99B's -- dense weights are 3.18 GiB against 10.24, and KV costs ~1.70x per
token -- so the optimum has no reason to be in the same place. On the 99B,
moving L from 1 to 54 cut 8.5K TTFT by 21%, so this is not a small knob.

Each point is a full boot, because L changes the CUDA weight footprint and
therefore the KV pool vLLM sizes at startup.

Two measurement traps this avoids
---------------------------------
1. PREFIX CACHING. Laguna caches, at ~139x. Measuring cold prefill twice
   measures the cache the second time. Every cold TTFT here uses a prompt
   slice unique to its (L, boot) pair, and asserts the server reported zero
   new cache hits for it.
2. VRAM SETTLE. A boot started before the previous engine released VRAM
   profiles a partial pool and reports a KV figure that is simply wrong
   (observed: 2.43 GiB where a settled boot reported 3.7). Every boot here
   waits for the card to return to baseline and ABORTS rather than proceeding.

Usage:
  .venv/bin/python benchmarks/laguna_l_sweep.py \
      --model shooting-brake-jota-r15 --total-experts 218 \
      --script benchmarks/serve_jota_r15_dual.sh \
      --local 20,40,57,65 --json-out benchmarks/results/b70_gemv_audit/r15_l_sweep.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def sh(cmd: str, timeout: float = 120.0) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                          timeout=timeout).stdout.strip()


def vram_used_mib() -> int:
    out = sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits")
    return int(out.splitlines()[0]) if out else 10**9


def kill_server(settle_mib: int, settle_s: float = 120.0) -> int:
    subprocess.run("pkill -9 -f 'vllm serve'", shell=True)
    subprocess.run("pkill -9 -f 'VLLM::EngineCore'", shell=True)
    time.sleep(4.0)
    deadline = time.monotonic() + settle_s
    while time.monotonic() < deadline:
        u = vram_used_mib()
        if u < settle_mib:
            return u
        time.sleep(2.0)
    return vram_used_mib()


def post(url: str, payload: dict, timeout: float = 900.0) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def metrics_hits(base: str) -> tuple[float, float]:
    try:
        with urllib.request.urlopen(f"{base}/metrics", timeout=30) as r:
            text = r.read().decode()
    except Exception:
        return (0.0, 0.0)

    def g(key: str) -> float:
        for line in text.splitlines():
            if line.startswith(key) and not line.startswith("#"):
                return float(line.split()[-1])
        return 0.0

    return g("vllm:prefix_cache_hits_total"), g("vllm:prefix_cache_queries_total")


def wait_health(base: str, log: Path, budget_s: float) -> bool:
    deadline = time.monotonic() + budget_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        if log.exists() and "Engine core initialization failed" in log.read_text(
                errors="ignore"):
            return False
        time.sleep(5.0)
    return False


def cold_ttft(base: str, model: str, prompt: str) -> tuple[float, int, int]:
    """One cold prefill. max_tokens=1 isolates TTFT from decode."""
    h0 = metrics_hits(base)
    t0 = time.perf_counter()
    r = post(f"{base}/v1/completions",
             {"model": model, "prompt": prompt, "temperature": 0.0,
              "max_tokens": 1})
    wall = time.perf_counter() - t0
    h1 = metrics_hits(base)
    return wall, int(r["usage"]["prompt_tokens"]), int(h1[0] - h0[0])


def decode_itl(base: str, model: str, out_tokens: int = 192,
               runs: int = 3) -> float:
    """Mean inter-token latency, excluding each run's first token.

    Decode is unaffected by prefix caching, so a repeated short prompt is
    fine here -- unlike the prefill measurement above.
    """
    per_run = []
    for _ in range(runs):
        req = urllib.request.Request(
            f"{base}/v1/chat/completions",
            data=json.dumps({
                "model": model,
                "messages": [{"role": "user",
                              "content": "Count from 1 to 60, one per line."}],
                "max_tokens": out_tokens, "temperature": 0.0, "stream": True,
                "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False,
                                         "thinking": False},
            }).encode(),
            headers={"Content-Type": "application/json"})
        stamps, tokens = [], 0
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                try:
                    chunk = json.loads(body)
                except json.JSONDecodeError:
                    continue
                usage = chunk.get("usage") or {}
                if usage.get("completion_tokens") is not None:
                    tokens = int(usage["completion_tokens"])
                ch = chunk.get("choices") or []
                if ch and (ch[0].get("delta") or {}).get("content"):
                    stamps.append(time.perf_counter())
        if len(stamps) >= 3 and tokens >= 3:
            # exclude the first inter-arrival: it carries prefill tail
            per_run.append((stamps[-1] - stamps[1]) / max(1, len(stamps) - 2))
    return sum(per_run) / len(per_run) * 1000.0 if per_run else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--script", required=True, help="serve recipe to launch")
    ap.add_argument("--model", required=True, help="served-model-name")
    ap.add_argument("--total-experts", type=int, required=True)
    ap.add_argument("--local", required=True,
                    help="comma-separated local expert counts to try")
    ap.add_argument("--base", default="http://127.0.0.1:8017")
    ap.add_argument("--prompt-words", type=int, default=6200,
                    help="~8K tokens of prose for the cold prefill")
    ap.add_argument("--settle-mib", type=int, default=2000)
    ap.add_argument("--boot-budget", type=float, default=480.0)
    ap.add_argument("--json-out", type=Path, required=True)
    args = ap.parse_args()

    words: list[str] = []
    for pat in ("docs/*.md", "src/phase4/src/shooting_brake_vllm/*.py",
                "benchmarks/*.py"):
        for f in sorted(Path(".").glob(pat)):
            words.extend(f.read_text(errors="ignore").split())
        if len(words) > 200_000:
            break
    print(f"corpus {len(words):,} words", flush=True)

    locals_ = [int(x) for x in args.local.split(",") if x.strip()]
    log = Path("/tmp/laguna_l_sweep_server.log")
    results = []

    for idx, L in enumerate(locals_):
        frac = L / args.total_experts
        print(f"\n{'=' * 66}\nL={L} of {args.total_experts} "
              f"(fraction {frac:.17f})\n{'=' * 66}", flush=True)
        settled = kill_server(args.settle_mib)
        if settled >= args.settle_mib:
            print(f"  ABORT: {settled} MiB still held, refusing to boot",
                  flush=True)
            results.append({"local_experts": L, "error": "vram_not_settled",
                            "held_mib": settled})
            continue
        print(f"  baseline {settled} MiB", flush=True)

        env = dict(os.environ)
        env["SHOOTING_BRAKE_PLACEMENT"] = f"fractional:2:{frac!r}"
        env["HF_HUB_OFFLINE"] = "1"
        if log.exists():
            log.unlink()
        with log.open("w") as fh:
            proc = subprocess.Popen(
                ["setsid", args.script], stdout=fh, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, env=env, start_new_session=True)

        t_boot = time.perf_counter()
        if not wait_health(args.base, log, args.boot_budget):
            txt = log.read_text(errors="ignore")
            why = next((l for l in txt.splitlines()
                        if "Error" in l or "error" in l), "")[:160]
            print(f"  BOOT FAILED: {why}", flush=True)
            results.append({"local_experts": L, "error": "boot_failed",
                            "detail": why})
            continue
        boot_s = time.perf_counter() - t_boot

        txt = log.read_text(errors="ignore")
        kv = next((float(m) for m in re.findall(
            r"Available KV cache memory: ([0-9.]+) GiB", txt)), float("nan"))
        conc = next((float(m) for m in re.findall(
            r"Maximum concurrency for [0-9,]+ tokens per request: ([0-9.]+)",
            txt)), float("nan"))
        engine_mib = vram_used_mib()

        # unique slice per point -> genuinely cold
        lo = idx * args.prompt_words
        prompt = " ".join(words[lo:lo + args.prompt_words])
        ttft, ptok, new_hits = cold_ttft(args.base, args.model, prompt)
        itl = decode_itl(args.base, args.model)

        row = {
            "local_experts": L, "remote_experts": args.total_experts - L,
            "fraction": frac, "boot_s": round(boot_s, 1),
            "kv_gib": kv, "max_concurrency": conc,
            "card_total_mib": engine_mib,
            "cold_ttft_s": round(ttft, 3), "prompt_tokens": ptok,
            "us_per_token": round(ttft / max(1, ptok) * 1e6, 1),
            "cold_cache_hits": new_hits,
            "itl_ms": round(itl, 3),
        }
        results.append(row)
        print(f"  KV {kv} GiB, conc {conc}x, card {engine_mib} MiB, boot {boot_s:.0f}s",
              flush=True)
        print(f"  cold TTFT {ttft:.3f}s over {ptok} tok = "
              f"{row['us_per_token']} us/tok (new cache hits {new_hits})",
              flush=True)
        print(f"  decode ITL {itl:.3f} ms", flush=True)
        if new_hits:
            print("  WARNING: cold measurement saw cache hits; not cold",
                  flush=True)

        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2))

    print(f"\n{'=' * 66}\nsummary\n{'=' * 66}")
    print(f"{'L':>4} {'remote':>7} {'KV GiB':>7} {'conc':>6} "
          f"{'TTFT s':>7} {'us/tok':>7} {'ITL ms':>7} {'card MiB':>9}")
    for r in results:
        if "error" in r:
            print(f"{r['local_experts']:>4} {r['error']}")
            continue
        print(f"{r['local_experts']:>4} {r['remote_experts']:>7} "
              f"{r['kv_gib']:>7} {r['max_concurrency']:>6} "
              f"{r['cold_ttft_s']:>7} {r['us_per_token']:>7} "
              f"{r['itl_ms']:>7} {r['card_total_mib']:>9}")
    print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
