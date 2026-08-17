"""Minimal streaming ITL probe for the DVFS pin A/B on the serving B70.

Sends N short-prompt streaming completions, measures TTFT and mean
inter-token latency from SSE chunk arrival times, and concurrently samples
the serving B70's act_freq (PCODE-resolved; cur_freq only shows GuC's
request and hides SLPC overrides).

Not a GuideLLM replacement: one shape (128-in / --out-tokens out, C=1),
built to answer exactly one question -- does pinning min_freq on card3
move production decode ITL?

Usage:
  .venv/bin/python benchmarks/b70_itl_probe.py --label baseline
  .venv/bin/python benchmarks/b70_itl_probe.py --label pinned2800
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import requests

URL = "http://127.0.0.1:8016/v1/chat/completions"
MODEL = "shooting-brake-88b"
FREQ = "/sys/class/drm/card3/device/tile0/gt0/freq0/act_freq"  # serving B70

PROMPT = (
    "You are auditing a heterogeneous inference system that streams "
    "27.4 GiB of int4 expert weights from host page cache to an RTX 5090 "
    "during prefill and dispatches 126 remote experts to an Intel Arc Pro "
    "B70 over a doorbell protocol during decode. Explain, in careful "
    "detail and without omitting any step, how the per-layer overlap of "
    "the CUDA partial and the B70 round trip bounds the inter-token "
    "latency, and why the maximum of the two legs rather than their sum "
    "is the correct cost model for one decoder layer under this design. "
    "Then describe what changes at batch sizes above one."
)


def one_request(out_tokens: int):
    t0 = time.perf_counter()
    first = None
    gaps = []
    prev = None
    with requests.post(
        URL,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": out_tokens,
            "temperature": 0.0,
            "stream": True,
            "ignore_eos": True,
        },
        stream=True,
        timeout=600,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith(b"data:"):
                continue
            if line == b"data: [DONE]":
                break
            now = time.perf_counter()
            if first is None:
                first = now
            elif prev is not None:
                gaps.append(now - prev)
            prev = now
    ttft = first - t0
    itl = statistics.mean(gaps) if gaps else float("nan")
    return ttft, itl, len(gaps) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out-tokens", type=int, default=256)
    ap.add_argument("--json-out", default=str(
        Path(__file__).resolve().parent / "results/b70_gemv_audit/itl_probe.json"))
    args = ap.parse_args()

    sampler = subprocess.Popen(
        ["bash", "-c",
         f"while true; do cat {FREQ} 2>/dev/null; sleep 0.005; done"],
        stdout=subprocess.PIPE, text=True)

    rows = []
    for i in range(args.n):
        ttft, itl, chunks = one_request(args.out_tokens)
        rows.append({"ttft_s": ttft, "itl_ms": itl * 1e3, "chunks": chunks})
        print(f"req {i}: ttft {ttft:.3f}s  itl {itl*1e3:.2f}ms  chunks {chunks}")

    sampler.terminate()
    out, _ = sampler.communicate()
    freqs = [int(x) for x in out.split() if x.strip().isdigit()]
    active = [f for f in freqs if f > 0]

    steady = rows[1:] if len(rows) > 1 else rows  # first request warms
    summary = {
        "label": args.label,
        "n": args.n,
        "out_tokens": args.out_tokens,
        "itl_ms_mean_excl_first": statistics.mean(r["itl_ms"] for r in steady),
        "ttft_s_mean_excl_first": statistics.mean(r["ttft_s"] for r in steady),
        "requests": rows,
        "act_freq": {
            "samples": len(freqs),
            "active_min": min(active) if active else 0,
            "active_max": max(active) if active else 0,
            "active_median": int(statistics.median(active)) if active else 0,
            "idle_fraction": round(1 - len(active) / len(freqs), 3) if freqs else None,
        },
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "requests"}, indent=1))

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out_path.read_text()) if out_path.exists() else []
    existing.append(summary)
    out_path.write_text(json.dumps(existing, indent=1))
    print("appended:", out_path)


if __name__ == "__main__":
    main()
