#!/usr/bin/env python3
"""Shooting Brake — Track B: summarize the offload sweep.

Reads every ``benchmark.py`` result JSON in a directory and prints the
tradeoff curve the sweep exists to map: as more expert capacity moves to
the B70 (and into fewer layers), how do decode speed, KV capacity, B70
route share, and dispatch service time move?

The "good" region is the top-left of the speed-vs-capacity curve — the
configuration that frees the most VRAM (most KV) at the least latency
cost.  Because each B70-active layer pays a fixed per-token dispatch,
concentrating the same offload into fewer layers (``subset``) usually
dominates spreading it across all 32 (``split``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        print(f"  warn: could not read {path.name}: {exc}", file=sys.stderr)
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir", type=Path, default=Path("phase10/results/offload"),
        help="directory of benchmark.py result JSON files",
    )
    args = parser.parse_args()

    rows = []
    for path in sorted(args.dir.glob("*.json")):
        if path.name in {"comparison.json", "matrix_config.json"}:
            continue
        data = _load(path)
        if not data:
            continue
        single = data.get("single_stream", {})
        worker = (data.get("workers") or [{}])[0]
        kv = worker.get("kv_cache", {})
        routes = worker.get("routes", {})
        poller = worker.get("poller", {})
        rows.append({
            "file": path.stem,
            "placement": data.get("placement", "?"),
            "tok_per_s": single.get("tok_per_s_mean", float("nan")),
            "itl_p50_ms": single.get("itl_p50_ms", float("nan")),
            "kv_tokens": kv.get("max_tokens", 0),
            "b70_share": routes.get("b70_share", 0.0),
            "service_us": poller.get("service_mean_us", float("nan")),
        })

    if not rows:
        print(f"no result JSON files in {args.dir}", file=sys.stderr)
        return 1

    # Baseline (all-cuda) for relative figures, if present.
    base = next((r for r in rows if r["b70_share"] == 0.0), None)
    base_tps = base["tok_per_s"] if base else float("nan")
    base_kv = base["kv_tokens"] if base else 1

    print(
        f"\n{'placement':<14} {'tok/s':>7} {'%base':>6} "
        f"{'ITLp50':>7} {'KV tok':>9} {'KVx':>5} "
        f"{'B70%':>5} {'svc_us':>7}"
    )
    print("-" * 64)
    for r in sorted(rows, key=lambda x: x["tok_per_s"], reverse=True):
        rel = (r["tok_per_s"] / base_tps * 100) if base_tps else float("nan")
        kvx = (r["kv_tokens"] / base_kv) if base_kv else float("nan")
        print(
            f"{r['placement']:<14} {r['tok_per_s']:>7.1f} {rel:>5.0f}% "
            f"{r['itl_p50_ms']:>6.2f}m {r['kv_tokens']:>9,} {kvx:>4.2f}x "
            f"{r['b70_share']*100:>4.0f}% {r['service_us']:>6.1f}"
        )

    if base:
        print(
            f"\nbaseline: all-cuda {base_tps:.1f} tok/s, "
            f"{base['kv_tokens']:,} KV tokens"
        )
    best = max(rows, key=lambda x: x["kv_tokens"])
    print(
        f"most capacity: {best['placement']} -> {best['kv_tokens']:,} KV tokens "
        f"at {best['tok_per_s']:.1f} tok/s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
