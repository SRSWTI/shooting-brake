#!/usr/bin/env python3
"""Byte-exactness gate for an SBEXP001 NVFP4 bank against its checkpoint.

The NVFP4-native bank's whole value proposition is ZERO transcode error --
bank bytes ARE checkpoint bytes. That makes the correctness gate exact rather
than statistical: for sampled records, rebuild the expected record from the
checkpoint tensors (the extractor's own layout: [gate;up] packed, [gate;up]
scales, down packed, down scales, then the two reciprocal global multipliers)
and compare byte-for-byte.

WHY THIS TOOL EXISTS SEPARATELY FROM THE SIZE CHECK
---------------------------------------------------
The bank is stored COMPACTLY. Bank row `i` holds the checkpoint's `i`-th
SPARSE layer, which is model layer `i` only when the sparse run starts at 0.
Laguna models carry a dense MLP at layer 0, so their row 0 holds model
layer 1.

A bank built one layer out of step is exactly the right length, so the header
self-consistency check at the top of `main` CANNOT see it. Nothing downstream
can either: the provider happily serves row `i`, the kernels run, the
logprobs are finite, and the model emits plausible wrong tokens.

So this tool derives the row -> model-layer mapping INDEPENDENTLY, by
rediscovering the checkpoint's sparse layer set itself rather than importing
`extract_experts.discover_shape`. That independence is the entire point: a
mapping bug shared with the writer would otherwise be invisible here.

Corners plus deterministic interior samples. The FIRST and LAST rows are
always checked, because those are where an off-by-one is unambiguous.

Usage:
    .venv/bin/python src/phase1/validate_expert_bank.py \
        --bank src/phase1/expert_bank_jota_118b_r20.bin \
        --model-dir <hf snapshot dir> [--samples 12]
"""

from __future__ import annotations

import argparse
import random
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

HEADER = struct.Struct("<8sIIIIIQQQQ")
MAGIC = b"SBEXP001"
GSCALE = struct.Struct("<ff")

# Independent of the extractor's pattern, deliberately.
EXPERT_KEY = re.compile(
    r"^(?P<root>.+)\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?P<suffix>.+)$"
)


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


def discover_layout(index: dict[str, str]) -> tuple[str, tuple[int, ...]]:
    """Rediscover the expert key root and the NVFP4 sparse layer list.

    A layer counts as NVFP4-routed when it carries ``weight_packed`` expert
    tensors; FP8 layers have experts but no packed planes and are never
    banked.
    """
    roots: set[str] = set()
    packed: set[int] = set()
    seen: set[int] = set()
    for key in index:
        m = EXPERT_KEY.match(key)
        if not m:
            continue
        roots.add(m.group("root"))
        layer = int(m.group("layer"))
        seen.add(layer)
        if m.group("suffix").endswith("weight_packed"):
            packed.add(layer)
    if not packed:
        raise SystemExit("no NVFP4 expert layers found in the checkpoint")
    if len(roots) != 1:
        raise SystemExit(f"expected one expert key root, found {sorted(roots)}")
    layers = tuple(sorted(packed))
    contiguous = tuple(range(layers[0], layers[-1] + 1))
    if layers != contiguous:
        raise SystemExit(
            "checkpoint sparse layers are not contiguous: "
            f"{layers}, missing {sorted(set(contiguous) - packed)}"
        )
    return next(iter(roots)), layers


def expected_record(
    index: dict[str, str], key_root: str, model_layer: int, expert: int,
) -> bytes:
    prefix = f"{key_root}.layers.{model_layer}.mlp.experts.{expert}"
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
            f"gate/up global scale mismatch at model layer={model_layer} "
            f"expert={expert}"
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
                    help="deterministic interior samples on top of the corners")
    ap.add_argument("--expect-first-model-layer", type=int, default=None,
                    help="assert bank row 0 holds this model layer "
                         "(0 for Qwen, 1 for Laguna). Optional but recommended "
                         "in scripts: it pins the mapping the operator INTENDS "
                         "rather than only the one the checkpoint implies.")
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
    print(f"bank: {layers} rows x {experts} experts, "
          f"hidden {hidden}, intermediate {inter}, "
          f"{record_bytes / 2**20:.2f} MiB/expert")

    index = index_shards(args.model_dir)
    key_root, sparse_layers = discover_layout(index)

    # The cross-check the size check cannot do: the bank's row count must
    # equal the number of sparse layers the CHECKPOINT actually has.
    if len(sparse_layers) != layers:
        raise SystemExit(
            f"bank declares {layers} rows but the checkpoint has "
            f"{len(sparse_layers)} NVFP4 sparse layers "
            f"({sparse_layers[0]}..{sparse_layers[-1]}) -- the bank was built "
            "from a different model or a different layer selection"
        )
    row_to_layer = {row: layer for row, layer in enumerate(sparse_layers)}
    print(f"checkpoint: key root {key_root!r}, {len(sparse_layers)} sparse "
          f"layers {sparse_layers[0]}..{sparse_layers[-1]}")
    print(f"mapping: bank row 0 -> model layer {row_to_layer[0]}, "
          f"row {layers - 1} -> model layer {row_to_layer[layers - 1]}")
    if args.expect_first_model_layer is not None:
        if row_to_layer[0] != args.expect_first_model_layer:
            raise SystemExit(
                f"row 0 maps to model layer {row_to_layer[0]}, operator "
                f"expected {args.expect_first_model_layer}"
            )
        print(f"  operator assertion satisfied: row 0 is model layer "
              f"{args.expect_first_model_layer}")

    rng = random.Random(99)
    # First and last rows always, at both expert extremes: an off-by-one in
    # the mapping is unambiguous exactly there.
    corners = [(0, 0), (0, experts - 1), (layers - 1, 0),
               (layers - 1, experts - 1)]
    interior = {
        (rng.randrange(layers), rng.randrange(experts))
        for _ in range(args.samples)
    }
    checks = corners + sorted(interior - set(corners))

    failures = 0
    for row, expert in checks:
        model_layer = row_to_layer[row]
        offset = HEADER.size + (row * experts + expert) * record_bytes
        raw.seek(offset)
        actual = raw.read(record_bytes)
        expected = expected_record(index, key_root, model_layer, expert)
        ok = actual == expected
        print(f"  row {row:2d} (model layer {model_layer:2d}) "
              f"expert {expert:3d}: {'EXACT' if ok else 'MISMATCH'}")
        if not ok:
            failures += 1
            regions = (("w13", 0, w13_b), ("s13", w13_b, w13_b + s13_b),
                       ("w2", w13_b + s13_b, w13_b + s13_b + w2_b),
                       ("s2", w13_b + s13_b + w2_b, record_bytes - 8),
                       ("gscale", record_bytes - 8, record_bytes))
            for name, lo, hi in regions:
                if actual[lo:hi] != expected[lo:hi]:
                    print(f"    first mismatch region: {name}")
                    break
            # A shift shows up as "this row matches a DIFFERENT model layer".
            for delta in (-1, 1):
                neighbour = row_to_layer.get(row + delta)
                if neighbour is None:
                    continue
                try:
                    alt = expected_record(index, key_root, neighbour, expert)
                except SystemExit:
                    continue
                if actual == alt:
                    print(f"    *** row {row} actually holds model layer "
                          f"{neighbour}: the bank is shifted by {delta:+d} ***")
                    break
    raw.close()

    if failures:
        print(f"FAIL: {failures}/{len(checks)} records differ")
        return 1
    print(f"PASS: {len(checks)} records byte-exact against the checkpoint")
    return 0


if __name__ == "__main__":
    sys.exit(main())
