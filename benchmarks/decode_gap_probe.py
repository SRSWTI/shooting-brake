#!/usr/bin/env python3
"""Decompose the decode inter-dispatch gap by LAYER INDEX -- zero server changes.

Decode ITL is 48 x ~64 us of B70 doorbell service plus 47 x ~211 us of
5090-side gap (69.7% of ITL). Before optimizing the gap, decide what it IS:

  - If per-layer gaps are UNIFORM across layer types, the gap is
    orchestration (graph segment replay, doorbell handshake, glue) and the
    levers are FULL capture / submission latency / scheduler overlap.
  - If gaps TRACK the layer type (full-attention layers cost visibly more
    than linear-attention layers), the gap is compute and the levers are
    kernel work, not plumbing.

Laguna interleaves linear_attention and full_attention layers
(`config.json: layer_types`), which makes the architecture itself the
instrument: the two layer classes alternate through the same orchestration,
so the DIFFERENCE isolates compute while the FLOOR isolates plumbing.

Method: stream one short-prompt completion (pure decode), wait for the trace
ring to flush, then reconstruct per-step layer sweeps from the per-device
doorbell traces (M=1 entries, host CLOCK_MONOTONIC) and report per-layer-index
gap medians annotated with layer type.

Usage (server must be up, trace dump enabled -- the serve script default):
  .venv/bin/python benchmarks/decode_gap_probe.py --out-tokens 96
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics as st
import time
from pathlib import Path

import requests


def layer_types(model_cache_glob: str) -> list[str]:
    hits = glob.glob(model_cache_glob)
    if not hits:
        return []
    cfg = json.loads((Path(sorted(hits)[-1]) / "config.json").read_text())
    return list(cfg.get("layer_types", []))


def decode_steps(entries: list[dict]) -> list[list[dict]]:
    """Group M=1 dispatches into layer sweeps (a new step starts when the
    layer index does not increase)."""
    dec = [e for e in entries if e.get("M") == 1]
    steps: list[list[dict]] = []
    cur: list[dict] = []
    last = -1
    for e in dec:
        if e["layer"] <= last and cur:
            steps.append(cur)
            cur = []
        cur.append(e)
        last = e["layer"]
    if cur:
        steps.append(cur)
    return [s for s in steps if len(s) >= 40]  # full sweeps only


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8017")
    ap.add_argument("--model", default="shooting-brake-jota-r15")
    ap.add_argument("--out-tokens", type=int, default=96)
    ap.add_argument("--trace-glob", default="/tmp/sb_r15_trace.device*.json")
    ap.add_argument(
        "--config-glob",
        default=str(Path.home() / ".cache/huggingface/hub/"
                    "models--srswti--axe-superveloce-jota-118b-r15-nvfp4/"
                    "snapshots/*"),
    )
    a = ap.parse_args()

    types = layer_types(a.config_glob)
    # bank row i holds model layer i+1 (layer 0 is the dense MLP, no MoE row)
    bank_type = types[1:] if types else []

    t0 = time.perf_counter()
    r = requests.post(
        f"{a.url}/v1/completions",
        json={"model": a.model, "prompt": "Count upward slowly: 1, 2, 3,",
              "max_tokens": a.out_tokens, "temperature": 0.0},
        timeout=600,
    )
    r.raise_for_status()
    wall = time.perf_counter() - t0
    print(f"decode request: {a.out_tokens} tokens in {wall:.2f}s "
          f"(~{wall/a.out_tokens*1e3:.2f} ms/token incl. prefill)")
    time.sleep(6)  # trace ring flushes every ~5 s

    for path in sorted(glob.glob(a.trace_glob)):
        entries = json.loads(Path(path).read_text())["entries"]
        steps = decode_steps(entries)
        if not steps:
            print(f"{path}: no full decode sweeps in ring")
            continue
        steps = steps[-min(len(steps), 64):]
        nlayers = max(len(s) for s in steps)
        print(f"\n=== {Path(path).name}: {len(steps)} decode sweeps, "
              f"{nlayers} layers ===")
        svc_all, gap_all, gap_by_type = [], [], {}
        rows = []
        for li in range(nlayers - 1):
            gaps, svcs = [], []
            for s in steps:
                if li + 1 >= len(s):
                    continue
                svcs.append((s[li]["t1_ns"] - s[li]["t0_ns"]) / 1e3)
                gaps.append((s[li + 1]["t0_ns"] - s[li]["t1_ns"]) / 1e3)
            if not gaps:
                continue
            g, v = st.median(gaps), st.median(svcs)
            svc_all.append(v)
            gap_all.append(g)
            # the NEXT layer's compute fills this gap, so attribute the gap
            # to the type of layer li+1
            t = bank_type[li + 1] if li + 1 < len(bank_type) else "?"
            gap_by_type.setdefault(t, []).append(g)
            rows.append((li, t, v, g))
        # compact: print every 4th row plus extremes
        for li, t, v, g in rows[::4]:
            print(f"  L{li:>2} -> next={t:<17} svc={v:7.1f} us  gap={g:7.1f} us")
        print(f"\n  svc  p50={st.median(svc_all):7.1f} us  "
              f"sum={sum(svc_all)/1e3:6.2f} ms")
        print(f"  gap  p50={st.median(gap_all):7.1f} us  "
              f"sum={sum(gap_all)/1e3:6.2f} ms")
        for t, gs in sorted(gap_by_type.items()):
            print(f"  gap into {t:<17} p50={st.median(gs):7.1f} us  n={len(gs)}")
        lin = gap_by_type.get("linear_attention", [])
        full = gap_by_type.get("full_attention", [])
        if lin and full:
            d = st.median(full) - st.median(lin)
            floor = st.median(lin)
            print(f"\n  VERDICT ({Path(path).name}): attention compute adds "
                  f"{d:.0f} us on full-attention layers; the {floor:.0f} us "
                  f"floor on linear layers is orchestration "
                  f"(replay + handshake + router/MoE glue).")
            print(f"  orchestration share of gap-sum: "
                  f"{floor*len(gap_all)/sum(gap_all)*100:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
