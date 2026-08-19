#!/usr/bin/env python3
"""Byte-exactness gate for an SBEXP001 NVFP4 bank against its checkpoint.

The NVFP4-native bank's whole value proposition is ZERO transcode error —
bank bytes ARE checkpoint bytes. That makes the correctness gate exact
rather than statistical: for sampled (layer, expert) records, rebuild the
expected record from the checkpoint tensors (the extractor's own layout:
[gate;up] packed, [gate;up] scales, down packed, down scales, then the two
reciprocal global multipliers) and compare byte-for-byte.

Corners plus deterministic random interior samples. A pass proves the
extractor's indexing, fusion, and reciprocal convention for exactly the
records the B70 will serve; any transcode bug that varies by position is
caught at the corners, any uniform one everywhere.

Usage:
    .venv/bin/python src/phase1/validate_99b_bank.py \
        --bank src/phase1/expert_bank_99b.bin \
        --model-dir <hf snapshot dir> [--samples 12]
"""

from __future__ import annotations

import argparse
import random
import struct
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

HEADER = struct.Struct("<8sIIIIIQQQQ")
MAGIC = b"SBEXP001"
GSCALE = struct.Struct("<ff")


def index_shards(model_dir: Path) -> dict[str, str]:
    index: dict[str, str] = {}
    for path in sorted(model_dir.glob("model*.safetensors")):
        if "mtp" in path.name.lower():
            continue
        with safe_open(str(path), framework="pt") as f:
            for key in f.keys():
                index[key] = str(path)
    if not index:
        raise SystemExit(f"no shards under {model_dir}")
    return index


def expected_record(
    index: dict[str, str], layer: int, expert: int,
) -> bytes:
    prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}"
    wanted: dict[str, list[str]] = defaultdict(list)
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for suffix in ("weight_packed", "weight_scale",
                       "weight_global_scale"):
            key = f"{prefix}.{proj}.{suffix}"
            shard = index.get(key)
            if shard is None:
                raise SystemExit(f"missing tensor: {key}")
            wanted[shard].append(key)
    t: dict[str, torch.Tensor] = {}
    for shard, keys in wanted.items():
        with safe_open(shard, framework="pt") as f:
            for key in keys:
                t[key] = f.get_tensor(key)

    gate_g = t[f"{prefix}.gate_proj.weight_global_scale"].item()
    up_g = t[f"{prefix}.up_proj.weight_global_scale"].item()
    down_g = t[f"{prefix}.down_proj.weight_global_scale"].item()
    if gate_g != up_g:
        raise SystemExit(
            f"gate/up global scale mismatch at layer={layer} expert={expert}"
        )

    w13 = torch.cat(
        [t[f"{prefix}.gate_proj.weight_packed"],
         t[f"{prefix}.up_proj.weight_packed"]], dim=0,
    ).contiguous()
    s13 = torch.cat(
        [t[f"{prefix}.gate_proj.weight_scale"],
         t[f"{prefix}.up_proj.weight_scale"]], dim=0,
    ).contiguous()
    return b"".join((
        w13.numpy().tobytes(),
        s13.view(torch.uint8).numpy().tobytes(),
        t[f"{prefix}.down_proj.weight_packed"].numpy().tobytes(),
        t[f"{prefix}.down_proj.weight_scale"]
            .view(torch.uint8).numpy().tobytes(),
        GSCALE.pack(1.0 / gate_g, 1.0 / down_g),
    ))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=12,
                    help="random interior samples on top of the 4 corners")
    args = ap.parse_args()

    raw = args.bank.open("rb")
    (magic, layers, experts, hidden, inter, _res,
     w13_b, s13_b, w2_b, s2_b) = HEADER.unpack(raw.read(HEADER.size))
    if magic != MAGIC:
        raise SystemExit(f"{args.bank} is not an SBEXP001 bank")
    record_bytes = w13_b + s13_b + w2_b + s2_b + GSCALE.size
    expected_size = HEADER.size + layers * experts * record_bytes
    actual_size = args.bank.stat().st_size
    if actual_size != expected_size:
        raise SystemExit(
            f"bank is {actual_size} bytes, header describes {expected_size}"
        )
    print(f"bank: {layers} layers x {experts} experts, "
          f"hidden {hidden}, intermediate {inter}, "
          f"{record_bytes / 2**20:.2f} MiB/expert")

    index = index_shards(args.model_dir)

    rng = random.Random(99)
    corners = [(0, 0), (0, experts - 1), (layers - 1, 0),
               (layers - 1, experts - 1)]
    interior = {
        (rng.randrange(layers), rng.randrange(experts))
        for _ in range(args.samples)
    }
    checks = corners + sorted(interior - set(corners))

    failures = 0
    for layer, expert in checks:
        offset = HEADER.size + (layer * experts + expert) * record_bytes
        raw.seek(offset)
        actual = raw.read(record_bytes)
        expected = expected_record(index, layer, expert)
        ok = actual == expected
        print(f"  layer {layer:2d} expert {expert:3d}: "
              f"{'EXACT' if ok else 'MISMATCH'}")
        if not ok:
            failures += 1
            # Name the first differing region for diagnosis.
            regions = (("w13", 0, w13_b), ("s13", w13_b, w13_b + s13_b),
                       ("w2", w13_b + s13_b, w13_b + s13_b + w2_b),
                       ("s2", w13_b + s13_b + w2_b, record_bytes - 8),
                       ("gscale", record_bytes - 8, record_bytes))
            for name, lo, hi in regions:
                if actual[lo:hi] != expected[lo:hi]:
                    print(f"    first mismatch region: {name}")
                    break
    raw.close()

    if failures:
        print(f"FAIL: {failures}/{len(checks)} records differ")
        return 1
    print(f"PASS: {len(checks)} records byte-exact against the checkpoint")
    return 0


if __name__ == "__main__":
    sys.exit(main())
