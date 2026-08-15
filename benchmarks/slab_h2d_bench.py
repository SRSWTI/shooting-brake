#!/usr/bin/env python3
"""H2D transfer ceiling for one SBINT401 layer slab.

Measures the three phases of the proposed prefill streaming pipeline
separately, because they bound different designs:

  1. stage   : mmap (page-cached) -> pinned staging, CPU memcpy, wall clock
  2. h2d     : pinned staging -> CUDA, measured with CUDA events
  3. e2e     : stage + h2d serialized -- the naive un-overlapped pipeline
  4. pageable: mmap -> CUDA directly (CUDA's internal staging), for reference

The pinned-H2D figure alone is an UPPER BOUND on any pipeline, not a pipeline
result: a real double-buffered stream runs at min(stage, h2d) at best.

A slab is one layer's remote experts: 126 experts x expert_stride (4,866,048 B)
= 584.7 MiB, read from the real bank at its real offset, so page-cache
behaviour and alignment match production. The mmap is explicitly faulted before
timing and the fault pass is reported (cold vs warm).

Usage:
  .venv/bin/python benchmarks/slab_h2d_bench.py \
      --bank src/phase1/expert_bank_int4.bin \
      --out benchmarks/results/slab_h2d/slab_h2d.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch

EXPERT_STRIDE = 4_866_048  # SBINT401 v2, validated by the provider
REMOTE_EXPERTS = 126       # split:54 -> experts 54..179 on the B70
HEADER_SKIP = 4096         # bank data_offset alignment


def numa_snapshot() -> dict:
    nodes = sorted(
        p.name for p in Path("/sys/devices/system/node").glob("node[0-9]*")
    ) if Path("/sys/devices/system/node").exists() else []
    return {
        "nodes": nodes,
        "cpu_affinity_count": len(os.sched_getaffinity(0)),
        "comment": (
            "single-socket Ryzen 9950X3D; one NUMA node expected, so no "
            "cross-node placement to control for"
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--iters", type=int, default=9)
    args = ap.parse_args()

    assert torch.cuda.is_available()
    slab = REMOTE_EXPERTS * EXPERT_STRIDE

    mm = np.memmap(args.bank, dtype=np.uint8, mode="r")
    src = mm[HEADER_SKIP: HEADER_SKIP + slab]

    # Explicit fault-in, timed, so cold-vs-warm is recorded rather than hidden.
    t0 = time.perf_counter()
    _ = int(np.sum(src[:: 4096], dtype=np.int64))
    fault_cold_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    _ = int(np.sum(src[:: 4096], dtype=np.int64))
    fault_warm_s = time.perf_counter() - t0

    src_t = torch.from_numpy(np.asarray(src))
    pinned = torch.empty(slab, dtype=torch.uint8, pin_memory=True)
    dev = torch.empty(slab, dtype=torch.uint8, device="cuda")
    gib = slab / 2**30

    def wall(fn, sync: bool):
        ts = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            fn()
            if sync:
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        return ts

    # 1. stage: mmap -> pinned (pure CPU memcpy; no sync needed)
    stage_ts = wall(lambda: pinned.copy_(src_t), sync=False)

    # 2. h2d: pinned -> device, CUDA events around only the copy
    ev_a, ev_b = torch.cuda.Event(True), torch.cuda.Event(True)
    h2d_ts = []
    for _ in range(args.iters):
        torch.cuda.synchronize()
        ev_a.record()
        dev.copy_(pinned, non_blocking=True)
        ev_b.record()
        torch.cuda.synchronize()
        h2d_ts.append(ev_a.elapsed_time(ev_b) / 1000.0)

    # 3. e2e: stage then h2d, serialized (the un-overlapped pipeline)
    def e2e():
        pinned.copy_(src_t)
        dev.copy_(pinned, non_blocking=True)
    e2e_ts = wall(e2e, sync=True)

    # 4. pageable reference: mmap tensor -> device directly
    pageable_ts = wall(lambda: dev.copy_(src_t, non_blocking=True), sync=True)

    def summarize(ts):
        med = statistics.median(ts)
        return {
            "median_s": med, "min_s": min(ts), "max_s": max(ts),
            "gib_per_s_median": gib / med, "raw_s": ts,
        }

    result = {
        "bank": str(args.bank),
        "slab_bytes": slab,
        "slab_mib": slab / 2**20,
        "experts": REMOTE_EXPERTS,
        "expert_stride_bytes": EXPERT_STRIDE,
        "iters": args.iters,
        "fault_pass": {"cold_s": fault_cold_s, "warm_s": fault_warm_s},
        "numa": numa_snapshot(),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "phases": {
            "stage_mmap_to_pinned": summarize(stage_ts),
            "h2d_pinned_to_device_cuda_events": summarize(h2d_ts),
            "e2e_serialized": summarize(e2e_ts),
            "pageable_mmap_to_device": summarize(pageable_ts),
        },
        "derived": {
            "forward_48_layers_gib": 48 * gib,
            "forward_s_e2e_serialized": 48 * statistics.median(e2e_ts),
            "forward_s_pageable": 48 * statistics.median(pageable_ts),
            "forward_s_overlap_bound": 48 * max(
                statistics.median(stage_ts), statistics.median(h2d_ts)
            ),
            "note": (
                "overlap_bound = min(stage, h2d) rate governs a double-buffered "
                "pipeline; h2d alone is an upper bound no pipeline reaches"
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    p = result["phases"]
    print(f"slab {result['slab_mib']:.1f} MiB, {args.iters} iters, "
          f"fault cold {fault_cold_s*1000:.0f} ms / warm {fault_warm_s*1000:.1f} ms")
    for k, v in p.items():
        print(f"  {k:36s} {v['gib_per_s_median']:6.2f} GiB/s")
    dv = result["derived"]
    print(f"  48-layer forward: e2e {dv['forward_s_e2e_serialized']:.2f} s | "
          f"pageable {dv['forward_s_pageable']:.2f} s | "
          f"overlap bound {dv['forward_s_overlap_bound']:.2f} s")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
