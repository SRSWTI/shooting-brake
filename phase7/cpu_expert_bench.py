#!/usr/bin/env python3
"""Cost model check for the CPU DDR5 expert tier.

A decode-shaped expert pass (M=1) reads every weight exactly once and reuses
none of them, so it is a pure DRAM stream and its latency is
``expert_bytes / achievable_bandwidth`` — compute never enters the picture.
This measures that floor at real model dimensions and reports the implied
bandwidth, which is what tells us whether the tier behaves as predicted and
where thread scaling stops paying.

Run::

    make -C phase7 cpu
    .venv/bin/python phase7/cpu_expert_bench.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                       "phase4" / "src"))

from shooting_brake_vllm.cpu_expert_host import CpuExpertHost  # noqa: E402

# The qualified model. Shooting Brake targets Qwen3.6-35B-A3B-NVFP4; larger
# models are a placement question, not a kernel one, so they are out of scope
# here until this tier is wired into the forward path.
SHAPES = [
    ("Qwen3.6-35B-A3B", 2048, 768),
]
THREAD_COUNTS = [1, 2, 4, 8, 12]
TOPK = 8


def bench(hidden: int, inter: int, threads: int, lib: Path,
          iters: int = 60) -> tuple[float, float]:
    """Returns (median microseconds per expert, implied GB/s).

    Median, not mean: this competes for DRAM bandwidth and cores with
    anything else on the box (a live vLLM server above all), and a handful of
    descheduled iterations otherwise drag the average far off the floor this
    is trying to find.
    """
    host = CpuExpertHost(
        num_layers=1, num_experts=1, hidden=hidden, intermediate=inter,
        max_experts=1, num_threads=threads, lib_path=lib,
    )
    gen = torch.Generator().manual_seed(7)
    gate = (torch.randn(inter, hidden, generator=gen) * 0.05).bfloat16()
    up = (torch.randn(inter, hidden, generator=gen) * 0.05).bfloat16()
    down = (torch.randn(hidden, inter, generator=gen) * 0.05).bfloat16()
    host.load_expert(0, 0, gate, up, down)

    x = (torch.randn(1, hidden, generator=gen) * 0.5).bfloat16()
    for _ in range(5):
        host.expert_forward(0, 0, x)  # warm the arena's first-touch faults

    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        host.expert_forward(0, 0, x)
        samples.append(time.perf_counter() - t0)
    samples.sort()

    per_call_s = samples[len(samples) // 2]
    weight_bytes = 3 * hidden * inter * 2  # gate + up + down, bf16
    host.close()
    return per_call_s * 1e6, weight_bytes / per_call_s / 1e9


def main() -> int:
    lib = Path(__file__).resolve().parent / "libsb_cpu_expert.so"
    if not lib.is_file():
        print(f"missing {lib}; run: make -C phase7 cpu")
        return 1

    for label, hidden, inter in SHAPES:
        weight_mib = 3 * hidden * inter * 2 / 2**20
        print(f"\n== {label} — hidden={hidden} inter={inter} "
              f"({weight_mib:.1f} MiB/expert bf16) ==")
        print(f"  {'threads':>8}  {'us/expert':>10}  {'GB/s':>7}  "
              f"{'us/token (top-8)':>17}")
        for threads in THREAD_COUNTS:
            us, gbs = bench(hidden, inter, threads, lib)
            # Worst case for a layer: every one of the token's routes lands
            # on this tier. Real placement should make that vanishingly rare.
            print(f"  {threads:>8}  {us:>10.1f}  {gbs:>7.1f}  "
                  f"{us * TOPK:>17.0f}")

    print("\nRead: us/expert is the per-activation cost when a routed expert "
          "lives in DRAM.\nCompare against the B70's ~40us for the same work. "
          "The right-hand column is\nthe worst case where all top-8 routes of "
          "one layer are CPU-resident — the\nnumber frequency-aware placement "
          "exists to keep you far away from.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
