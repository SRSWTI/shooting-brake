#!/usr/bin/env python3
"""Compare the two VRAM-surgery strategies by hashing the weights.

Token-level comparison can no longer answer "does the compact layout hold
the right experts". Measured on the 35B: two runs of *identical* code and
configuration disagree by ~0.11 nats/token and share only 4/8 sequences,
because B70 partials are accumulated asynchronously and the CUDA kernel's
reductions are not order-stable. Any difference smaller than that is
invisible, and 0.11 nats is not a small budget.

Weights do not have that problem. They are fixed once loading ends, so a
digest is exact: post-hoc surgery slices a fully-loaded bank down to the
CUDA-owned experts, pre-emptive never allocates the rest, and the two must
arrive at identical bytes or one of them is addressing the wrong experts.

Only the load is exercised — the digests are written from
``process_weights_after_loading`` — so each leg costs a model load rather
than a full benchmark.

Usage:
  python benchmarks/weight_digest.py --placement subset:16:8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))


async def _load_only(preemptive: bool, digest_path: Path, placement: str) -> None:
    """Build an engine, let it load, and shut it down."""
    from offload_benchmark import apply_config_env, build_engine

    apply_config_env("hybrid", placement, False, preemptive)
    os.environ["SHOOTING_BRAKE_WEIGHT_DIGEST"] = str(digest_path)
    engine = build_engine(max_num_seqs=8, max_model_len=2048)
    try:
        pass  # loading is the whole experiment
    finally:
        shutdown = getattr(engine, "shutdown", None)
        if shutdown is not None:
            shutdown()


def _read(path: Path) -> dict[int, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    return {int(r["layer"]): r for r in rows}


def compare(posthoc: Path, preemptive: Path) -> int:
    a, b = _read(posthoc), _read(preemptive)
    if a.keys() != b.keys():
        print(f"layer sets differ: {sorted(a)} vs {sorted(b)}")
        return 1

    # `w2_alpha_vec` is diagnostic, not a field to diff: it is expected to
    # differ whenever the activation scale does, and a raw list comparison
    # would both mis-classify it and print 241 floats. It is analysed
    # separately below, where the *shape* of the difference is the answer.
    fields = [
        k for k in next(iter(a.values()))
        if k not in ("layer", "num_experts", "w2_alpha_vec")
    ]
    differing: dict[str, list[int]] = {f: [] for f in fields}
    for layer in sorted(a):
        if a[layer]["num_experts"] != b[layer]["num_experts"]:
            print(
                f"layer {layer}: expert count differs "
                f"{a[layer]['num_experts']} vs {b[layer]['num_experts']}"
            )
            return 1
        for f in fields:
            if a[layer][f] != b[layer][f]:
                differing[f].append(layer)

    # The weights, block scales, per-expert weight scales and remap answer
    # "does the compact layout address the right experts". Only the two
    # per-layer activation scalars answer a different question — what range
    # the kernel quantizes activations over — and they legitimately differ:
    # `prepare_nvfp4_moe_layer_for_fi_or_cutlass` reduces them with a
    # whole-tensor `.max()`, so post-hoc reduces over all 256 experts'
    # scales while pre-emptive only ever loaded the CUDA-owned ones. That
    # is upstream's own semantics for a partial expert bank: every
    # expert-parallel rank reduces over what it holds.
    ACTIVATION = {"a1_gscale", "a2_gscale"}
    # `qc_w*_alpha` are per-expert weight global scales and are normally
    # addressing-sensitive. Kernel-format conversion can fold the
    # activation scalar into them, so a difference there is excused only
    # when it tracks the corresponding activation scalar exactly, layer for
    # layer. Anything else is a wrong-expert bug wearing a scale's name.
    # `w2_weight_scale_2` is the layer-attribute alias of `qc_w2_alpha`
    # (post-hoc surgery assigns one from the other), so it folds identically.
    FOLDED = {
        "qc_w1_alpha": "a1_gscale",
        "qc_w2_alpha": "a2_gscale",
        "w13_weight_scale_2": "a1_gscale",
        "w2_weight_scale_2": "a2_gscale",
    }

    print(f"{'field':22} {'layers differing':>16}   detail")
    print("-" * 78)
    addressing_clean = True
    activation_differs = []
    for f in fields:
        layers = differing[f]
        if not layers:
            print(f"{f:22} {'0':>16}   identical")
            continue
        sample = layers[0]
        detail = f"e.g. layer {sample}: {a[sample][f]} vs {b[sample][f]}"
        excused = f in ACTIVATION or (
            f in FOLDED and layers == differing.get(FOLDED[f])
        )
        if excused:
            activation_differs.append(f)
            print(f"{f:22} {len(layers):>16}   [scale] {detail}")
        else:
            addressing_clean = False
            print(f"{f:22} {len(layers):>16}   [ADDRESSING] {detail}")

    # The discriminating test. If the per-expert alpha differs only because
    # the activation scalar was folded in, the elementwise ratio between
    # the two strategies is one constant per layer, equal to the a2 ratio.
    # A wrong-expert bug permutes values instead, and the ratios scatter —
    # which min/max alone cannot reveal, since a permutation can preserve
    # the extremes.
    scattered = []
    for layer in sorted(a):
        va, vb = a[layer].get("w2_alpha_vec"), b[layer].get("w2_alpha_vec")
        if not va or not vb or len(va) != len(vb):
            continue
        ratios = [x / y for x, y in zip(va, vb) if y]
        if not ratios:
            continue
        spread = max(ratios) - min(ratios)
        # fp32 round-trip through JSON leaves ~1e-5 of relative slop.
        if spread / max(ratios) > 1e-4:
            scattered.append((layer, spread))
    if scattered:
        addressing_clean = False
        worst = max(scattered, key=lambda t: t[1])
        print(
            f"\n  w2 alpha ratio is NOT constant in {len(scattered)} layers "
            f"(worst layer {worst[0]}, spread {worst[1]:.2e}) — the per-expert "
            "values were permuted, not scaled"
        )
    else:
        print(
            "\n  w2 alpha ratio constant within fp32 rounding in every layer "
            "— a scalar fold, not a permutation"
        )
    print()
    if not addressing_clean:
        print(
            "DIVERGENT — the compact layout holds different expert weights "
            "than the sliced one. This is an addressing bug, not a "
            "calibration difference."
        )
        return 1
    if activation_differs:
        print(
            "WEIGHTS IDENTICAL — every expert weight, block scale and the "
            "global->local remap match byte for byte, so compaction "
            "addresses the right experts.\n"
            f"Activation scale differs ({', '.join(activation_differs)}): "
            "the reduction covers only the CUDA-owned experts, which is "
            "what upstream computes on an expert-parallel rank and the only "
            "thing available on a model too large to load whole."
        )
        return 0
    print(
        "IDENTICAL — pre-emptive allocation produces byte-for-byte the "
        "same expert weights, scales and remap as post-hoc surgery."
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--placement", default="subset:16:8")
    ap.add_argument(
        "--out-dir", type=Path,
        default=Path(__file__).resolve().parent / "results" / "digest",
    )
    ap.add_argument(
        "--leg", choices=("posthoc", "preemptive"),
        help="run one leg (a fresh process per leg is required: the adapter "
             "reads its configuration at class-construction time)",
    )
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    posthoc = args.out_dir / "posthoc.jsonl"
    preemptive = args.out_dir / "preemptive.jsonl"

    if args.leg:
        target = posthoc if args.leg == "posthoc" else preemptive
        target.unlink(missing_ok=True)
        asyncio.run(_load_only(args.leg == "preemptive", target, args.placement))
        print(f"wrote {target} ({len(target.read_text().splitlines())} layers)")
        return 0
    if args.compare:
        return compare(posthoc, preemptive)
    ap.error("pass --leg posthoc, --leg preemptive, or --compare")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
