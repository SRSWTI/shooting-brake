"""Minimal streaming ITL probe for the DVFS pin A/B on the serving B70.

Sends N short-prompt streaming completions, measures TTFT and mean
inter-token latency from SSE chunk arrival times, and concurrently samples
the serving B70's act_freq (PCODE-resolved; cur_freq only shows GuC's
request and hides SLPC overrides).

Not a GuideLLM replacement: one shape (128-in / --out-tokens out, C=1),
built to measure warmed decode ITL while sampling every serving B70.

Usage:
  .venv/bin/python benchmarks/b70_itl_probe.py --label baseline
  .venv/bin/python benchmarks/b70_itl_probe.py --label 99b-graph \
    --url http://127.0.0.1:8017/v1/chat/completions \
    --model shooting-brake-99b \
    --freq-path /sys/bus/pci/devices/0000:15:00.0/tile0/gt0/freq0/act_freq \
    --freq-path /sys/bus/pci/devices/0000:11:00.0/tile0/gt0/freq0/act_freq
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from pathlib import Path

import requests

DEFAULT_URL = "http://127.0.0.1:8016/v1/chat/completions"
DEFAULT_MODEL = "shooting-brake-88b"
DEFAULT_FREQ_PATHS = (
    "/sys/class/drm/card3/device/tile0/gt0/freq0/act_freq",
)

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

class FrequencySampler:
    """Sample one or more B70 clocks without shelling out."""

    def __init__(self, paths: list[Path], interval_s: float = 0.005) -> None:
        self.paths = paths
        self.interval_s = interval_s
        self.samples: dict[str, list[int]] = {str(path): [] for path in paths}
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="b70-frequency-sampler"
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        while not self._stop.is_set():
            for path in self.paths:
                try:
                    self.samples[str(path)].append(int(path.read_text().strip()))
                except (OSError, ValueError):
                    continue
            self._stop.wait(self.interval_s)


def summarize_frequencies(freqs: list[int]) -> dict[str, int | float | None]:
    active = [freq for freq in freqs if freq > 0]
    return {
        "samples": len(freqs),
        "active_min": min(active) if active else 0,
        "active_max": max(active) if active else 0,
        "active_median": int(statistics.median(active)) if active else 0,
        "idle_fraction": (
            round(1 - len(active) / len(freqs), 3) if freqs else None
        ),
    }


def one_request(url: str, model: str, out_tokens: int):
    t0 = time.perf_counter()
    first = None
    gaps = []
    prev = None
    with requests.post(
        url,
        json={
            "model": model,
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
    if first is None:
        raise RuntimeError("stream completed without a data chunk")
    ttft = first - t0
    itl = statistics.mean(gaps) if gaps else float("nan")
    return ttft, itl, len(gaps) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out-tokens", type=int, default=256)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument(
        "--freq-path",
        action="append",
        dest="freq_paths",
        help="act_freq sysfs path; repeat once per serving B70",
    )
    ap.add_argument("--json-out", default=str(
        Path(__file__).resolve().parent / "results/b70_gemv_audit/itl_probe.json"))
    args = ap.parse_args()

    freq_paths = [
        Path(path) for path in (args.freq_paths or DEFAULT_FREQ_PATHS)
    ]
    missing = [str(path) for path in freq_paths if not path.is_file()]
    if missing:
        ap.error(f"frequency path(s) not found: {', '.join(missing)}")

    sampler = FrequencySampler(freq_paths)
    rows = []
    sampler.start()
    try:
        for i in range(args.n):
            ttft, itl, chunks = one_request(
                args.url, args.model, args.out_tokens
            )
            rows.append({
                "ttft_s": ttft,
                "itl_ms": itl * 1e3,
                "chunks": chunks,
            })
            print(
                f"req {i}: ttft {ttft:.3f}s  "
                f"itl {itl * 1e3:.2f}ms  chunks {chunks}"
            )
    finally:
        sampler.stop()

    steady = rows[1:] if len(rows) > 1 else rows
    summary = {
        "label": args.label,
        "url": args.url,
        "model": args.model,
        "n": args.n,
        "out_tokens": args.out_tokens,
        "itl_ms_mean_excl_first": statistics.mean(
            row["itl_ms"] for row in steady
        ),
        "ttft_s_mean_excl_first": statistics.mean(
            row["ttft_s"] for row in steady
        ),
        "requests": rows,
        "act_freq": {
            path: summarize_frequencies(freqs)
            for path, freqs in sampler.samples.items()
        },
    }
    print(json.dumps(
        {key: value for key, value in summary.items() if key != "requests"},
        indent=1,
    ))

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out_path.read_text()) if out_path.exists() else []
    existing.append(summary)
    out_path.write_text(json.dumps(existing, indent=1))
    print("appended:", out_path)


if __name__ == "__main__":
    main()
