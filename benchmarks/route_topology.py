#!/usr/bin/env python3
"""Simulate two-B70 expert placements from a Shooting Brake route trace.

The trace format is owned by ``shooting_brake_vllm.route_stats`` and parsed by
``route_locality``.  This tool is CPU-only.  Counts are evaluated per trace
record: one token row at one routed layer invocation.

Subcommands:

* ``generate`` writes a production-format synthetic trace.  ``uniform`` makes
  each of the top-k remote occurrences choose either half independently, then
  chooses a unique expert uniformly within that half.  Its card count is
  exactly Binomial(top_k, 0.5), while every expert has the same marginal
  probability.  ``id-clustered`` chooses one half per record and puts every
  route in it.
* ``analyze`` compares a contiguous ID split, replicated occurrence dealing,
  and the current single-card placement, then sweeps every contiguous remote
  boundary.

The device-kernel curve is evaluated record by record at its measured integer
knots.  Route counts are integers, so this simulator performs no interpolation.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from route_locality import (
    TraceHeader,
    analyze_trace as analyze_locality,
    format_report as format_locality_report,
    read_trace,
    write_trace,
)

KERNEL_US = np.asarray([8.1, 30.8, 32.2, 33.6, 32.6, 37.1, 56.6, 72.3, 79.9])
DEFAULT_NUM_EXPERTS = 180
DEFAULT_TOP_K = 8
DEFAULT_CUDA_EXPERTS = 12
DEFAULT_SPLIT_BOUNDARY = 96
DEFAULT_CURRENT_CUDA_EXPERTS = 54


def _validate_geometry(
    header: TraceHeader,
    *,
    cuda_experts: int,
    split_boundary: int,
    current_cuda_experts: int,
) -> None:
    if header.top_k >= len(KERNEL_US):
        raise ValueError(
            f"top_k={header.top_k} exceeds the measured curve's maximum k={len(KERNEL_US) - 1}"
        )
    if not 0 <= cuda_experts < split_boundary < header.num_experts:
        raise ValueError(
            "contiguous geometry must satisfy 0 <= cuda_experts < "
            "split_boundary < num_experts"
        )
    if not 0 <= current_cuda_experts < header.num_experts:
        raise ValueError("current CUDA share must be inside the expert ID range")


def select_single_row_steps(records: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Keep groups with exactly one row, the observable concurrency-1 proxy.

    The v1 trace has no prefill/decode bit.  Multi-row groups are certainly not
    concurrency-1 decode and are excluded.  A one-row prefill cannot be
    distinguished and must be prevented by capture procedure.
    """
    if not len(records):
        return records, 0, 0
    keys = np.empty(len(records), dtype=[("layer", "<u2"), ("step", "<u8")])
    keys["layer"] = records["layer"]
    keys["step"] = records["step"]
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    keep = counts[inverse] == 1
    kept_groups = int(np.count_nonzero(counts == 1))
    excluded_groups = int(np.count_nonzero(counts != 1))
    return records[keep], kept_groups, excluded_groups


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q, method="linear"))


def count_summary(values: np.ndarray) -> dict[str, object]:
    if values.ndim != 1 or not len(values):
        raise ValueError("cannot summarize an empty or non-vector count array")
    histogram = np.bincount(values, minlength=int(values.max()) + 1)
    return {
        "mean": float(values.mean()),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "max": int(values.max()),
        "zero_fraction": float(np.mean(values == 0)),
        "histogram": histogram,
    }


def topology_summary(name: str, card_a: np.ndarray, card_b: np.ndarray | None) -> dict[str, object]:
    if card_a.dtype.kind not in "iu" or (card_b is not None and card_b.dtype.kind not in "iu"):
        raise TypeError("route counts must be integer arrays")
    result: dict[str, object] = {"name": name, "card_a": count_summary(card_a)}
    if card_b is None:
        critical_routes = card_a
        critical_kernel = KERNEL_US[card_a]
        result["card_b"] = None
        result["any_card_zero_fraction"] = float(np.mean(card_a == 0))
        result["both_cards_zero_fraction"] = float(np.mean(card_a == 0))
    else:
        if card_a.shape != card_b.shape:
            raise ValueError("card count arrays have different shapes")
        critical_routes = np.maximum(card_a, card_b)
        critical_kernel = np.maximum(KERNEL_US[card_a], KERNEL_US[card_b])
        result["card_b"] = count_summary(card_b)
        result["any_card_zero_fraction"] = float(np.mean((card_a == 0) | (card_b == 0)))
        result["both_cards_zero_fraction"] = float(np.mean((card_a == 0) & (card_b == 0)))
    result["critical_routes"] = count_summary(critical_routes)
    result["mean_critical_kernel_us"] = float(critical_kernel.mean())
    result["critical_kernel_p50_us"] = _percentile(critical_kernel, 50)
    result["critical_kernel_p95_us"] = _percentile(critical_kernel, 95)
    result["critical_kernel_p99_us"] = _percentile(critical_kernel, 99)
    result["critical_kernel_max_us"] = float(critical_kernel.max())
    return result


def simulate_topologies(
    header: TraceHeader,
    records: np.ndarray,
    *,
    cuda_experts: int = DEFAULT_CUDA_EXPERTS,
    split_boundary: int = DEFAULT_SPLIT_BOUNDARY,
    current_cuda_experts: int = DEFAULT_CURRENT_CUDA_EXPERTS,
) -> list[dict[str, object]]:
    _validate_geometry(
        header,
        cuda_experts=cuda_experts,
        split_boundary=split_boundary,
        current_cuda_experts=current_cuda_experts,
    )
    if not len(records):
        raise ValueError("trace contains no selected layer-step records")
    ids = records["experts"]
    contiguous_a = np.count_nonzero(
        (ids >= cuda_experts) & (ids < split_boundary), axis=1
    ).astype(np.int16)
    contiguous_b = np.count_nonzero(ids >= split_boundary, axis=1).astype(np.int16)
    remote = np.count_nonzero(ids >= cuda_experts, axis=1).astype(np.int16)
    replicated_a = ((remote + 1) // 2).astype(np.int16)
    replicated_b = (remote // 2).astype(np.int16)
    current_remote = np.count_nonzero(ids >= current_cuda_experts, axis=1).astype(np.int16)
    return [
        topology_summary("contiguous ID split", contiguous_a, contiguous_b),
        topology_summary("replicated occurrence split", replicated_a, replicated_b),
        topology_summary(
            f"current single-card split:{current_cuda_experts}", current_remote, None
        ),
    ]


def _uniform_global_expected_max(
    *, top_k: int, num_experts: int, cuda_experts: int, split_boundary: int
) -> float:
    """Exact E[max(A,B)] for a uniform top-k subset of all experts."""
    a_experts = split_boundary - cuda_experts
    b_experts = num_experts - split_boundary
    denominator = math.comb(num_experts, top_k)
    total = 0
    for local in range(top_k + 1):
        if local > cuda_experts:
            continue
        for a_count in range(top_k - local + 1):
            b_count = top_k - local - a_count
            if a_count > a_experts or b_count > b_experts:
                continue
            ways = (
                math.comb(cuda_experts, local)
                * math.comb(a_experts, a_count)
                * math.comb(b_experts, b_count)
            )
            total += ways * max(a_count, b_count)
    return total / denominator


def sweep_boundaries(
    header: TraceHeader,
    records: np.ndarray,
    *,
    cuda_experts: int,
) -> list[dict[str, float | int]]:
    """Evaluate every A/B boundary while holding the CUDA prefix fixed."""
    ids = records["experts"]
    local = np.count_nonzero(ids < cuda_experts, axis=1).astype(np.int16)
    remote = header.top_k - local
    rows: list[dict[str, float | int]] = []
    for boundary in range(cuda_experts + 1, header.num_experts):
        card_a = np.count_nonzero(
            (ids >= cuda_experts) & (ids < boundary), axis=1
        ).astype(np.int16)
        card_b = remote - card_a
        critical = np.maximum(card_a, card_b)
        critical_kernel = np.maximum(KERNEL_US[card_a], KERNEL_US[card_b])
        uniform = _uniform_global_expected_max(
            top_k=header.top_k,
            num_experts=header.num_experts,
            cuda_experts=cuda_experts,
            split_boundary=boundary,
        )
        rows.append(
            {
                "boundary": boundary,
                "a_experts": boundary - cuda_experts,
                "b_experts": header.num_experts - boundary,
                "mean_max": float(critical.mean()),
                "uniform_mean_max": uniform,
                "above_uniform": float(critical.mean()) - uniform,
                "mean_critical_kernel_us": float(critical_kernel.mean()),
                "a_zero_fraction": float(np.mean(card_a == 0)),
                "b_zero_fraction": float(np.mean(card_b == 0)),
            }
        )
    return rows


def _format_histogram(summary: dict[str, object]) -> str:
    histogram = summary["histogram"]
    assert isinstance(histogram, np.ndarray)
    total = int(histogram.sum())
    return " ".join(
        f"{count}:{int(frequency)}({100.0 * int(frequency) / total:.2f}%)"
        for count, frequency in enumerate(histogram)
        if frequency
    )


def _format_card(label: str, summary: dict[str, object]) -> list[str]:
    return [
        f"  {label}: mean={summary['mean']:.5f} p50={summary['p50']:.2f} "
        f"p95={summary['p95']:.2f} p99={summary['p99']:.2f} "
        f"max={summary['max']} zero={100.0 * summary['zero_fraction']:.3f}%",
        f"    histogram routes:count(percent) {_format_histogram(summary)}",
    ]


def format_topology(summary: dict[str, object]) -> str:
    lines = [f"--- {summary['name']} ---"]
    card_a = summary["card_a"]
    assert isinstance(card_a, dict)
    lines.extend(_format_card("card A", card_a))
    card_b = summary["card_b"]
    if isinstance(card_b, dict):
        lines.extend(_format_card("card B", card_b))
    critical = summary["critical_routes"]
    assert isinstance(critical, dict)
    lines.extend(_format_card("max(A,B)" if card_b is not None else "active routes", critical))
    lines.append(
        f"  E[max(cardA,cardB)]={critical['mean']:.5f} routes; "
        f"critical device kernel mean={summary['mean_critical_kernel_us']:.3f} us "
        f"p50={summary['critical_kernel_p50_us']:.1f} "
        f"p95={summary['critical_kernel_p95_us']:.1f} "
        f"p99={summary['critical_kernel_p99_us']:.1f} "
        f"max={summary['critical_kernel_max_us']:.1f} us"
    )
    lines.append(
        f"  any-card-zero={100.0 * summary['any_card_zero_fraction']:.3f}% "
        f"both-cards-zero={100.0 * summary['both_cards_zero_fraction']:.3f}%"
    )
    return "\n".join(lines)


def format_sweep(
    rows: Sequence[dict[str, float | int]], *, selected_boundary: int
) -> str:
    if not rows:
        raise ValueError("empty boundary sweep")
    best = min(rows, key=lambda row: float(row["mean_max"]))
    lines = [
        "=== Contiguous boundary sweep (CUDA prefix held fixed) ===",
        "uniform E[max] is the exact without-replacement null over all expert IDs",
        f"minimum observed E[max]={best['mean_max']:.5f} at boundary={best['boundary']} "
        f"(A={best['a_experts']} experts, B={best['b_experts']} experts)",
        "boundary A_n B_n Emax uniform_Emax delta critical_kernel_us A_zero% B_zero% selected",
    ]
    for row in rows:
        lines.append(
            f"{row['boundary']:>8} {row['a_experts']:>3} {row['b_experts']:>3} "
            f"{row['mean_max']:.5f} {row['uniform_mean_max']:.5f} "
            f"{row['above_uniform']:+.5f} {row['mean_critical_kernel_us']:.3f} "
            f"{100.0 * float(row['a_zero_fraction']):.3f} "
            f"{100.0 * float(row['b_zero_fraction']):.3f} "
            f"{'*' if int(row['boundary']) == selected_boundary else ''}"
        )
    return "\n".join(lines)


def _synthetic_records(
    *,
    mode: str,
    tokens: int,
    layers: int,
    top_k: int,
    cuda_experts: int,
    split_boundary: int,
    num_experts: int,
    seed: int,
) -> Iterable[tuple[int, int, int, Sequence[int]]]:
    rng = random.Random(seed)
    pool_a = range(cuda_experts, split_boundary)
    pool_b = range(split_boundary, num_experts)
    if len(pool_a) < top_k or len(pool_b) < top_k:
        raise ValueError("each synthetic remote half must contain at least top_k experts")
    for step in range(tokens):
        for layer in range(layers):
            if mode == "uniform":
                a_count = sum(rng.getrandbits(1) for _ in range(top_k))
                experts = rng.sample(pool_a, a_count) + rng.sample(pool_b, top_k - a_count)
            elif mode == "id-clustered":
                experts = rng.sample(pool_a if rng.getrandbits(1) else pool_b, top_k)
            else:
                raise ValueError(f"unsupported synthetic mode {mode!r}")
            rng.shuffle(experts)
            yield step, layer, 0, experts


def generate_trace(args: argparse.Namespace) -> None:
    if args.tokens <= 0 or args.layers <= 0:
        raise ValueError("tokens and layers must be positive")
    if not 0 <= args.cuda_experts < args.split_boundary < args.experts:
        raise ValueError("invalid synthetic topology geometry")
    records = _synthetic_records(
        mode=args.mode,
        tokens=args.tokens,
        layers=args.layers,
        top_k=args.top_k,
        cuda_experts=args.cuda_experts,
        split_boundary=args.split_boundary,
        num_experts=args.experts,
        seed=args.seed,
    )
    target = write_trace(
        args.output,
        records,
        num_layers=args.layers,
        num_experts=args.experts,
        top_k=args.top_k,
    )
    print(
        f"wrote {target}: mode={args.mode} tokens={args.tokens} layers={args.layers} "
        f"records={args.tokens * args.layers} seed={args.seed}"
    )
    if args.mode == "uniform":
        analytic = sum(
            math.comb(args.top_k, count)
            * max(count, args.top_k - count)
            for count in range(args.top_k + 1)
        ) / (2**args.top_k)
        print(f"analytic Binomial(top_k, 0.5) E[max]={analytic:.5f}")


def select_last_step_runs(
    records: np.ndarray, run_count: int
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Keep the last ``run_count`` consecutive global-step runs.

    In a strictly sequential concurrency-1 capture, each multi-token prefill
    is removed by :func:`select_single_row_steps`, leaving one consecutive
    single-row run per request.  Selecting the tail also discards server
    startup/profile forwards.  This is not valid for continuous batching.
    """
    if run_count <= 0:
        raise ValueError("last-sequences must be positive")
    steps = np.unique(records["step"])
    if not len(steps):
        raise ValueError("cannot select sequence runs from an empty record set")
    breaks = np.flatnonzero(steps[1:] != steps[:-1] + 1) + 1
    step_runs = np.split(steps, breaks)
    if len(step_runs) < run_count:
        raise ValueError(
            f"trace contains only {len(step_runs)} consecutive single-row step "
            f"runs, cannot select the last {run_count}"
        )
    selected_runs = step_runs[-run_count:]
    selected_steps = np.concatenate(selected_runs)
    selected = records[np.isin(records["step"], selected_steps)]
    bounds = [(int(run[0]), int(run[-1])) for run in selected_runs]
    return selected, bounds


def analyze_trace_file(args: argparse.Namespace) -> None:
    header, all_records = read_trace(args.trace)
    if args.include_multirow:
        records = all_records
        kept_groups = len(np.unique(all_records[["layer", "step"]])) if len(all_records) else 0
        excluded_groups = 0
        selection = "all rows (including multi-row forwards)"
    else:
        records, kept_groups, excluded_groups = select_single_row_steps(all_records)
        selection = "single-row (concurrency-1 decode proxy)"
    if not len(records):
        raise ValueError(
            "no records remain after selection; use --include-multirow only if "
            "per-row prefill/continuous-batch analysis is intentional"
        )
    selected_runs: list[tuple[int, int]] = []
    if args.last_sequences is not None:
        if args.include_multirow:
            raise ValueError("--last-sequences cannot be combined with --include-multirow")
        records, selected_runs = select_last_step_runs(records, args.last_sequences)
        selection += f", last {args.last_sequences} consecutive step runs"
    summaries = simulate_topologies(
        header,
        records,
        cuda_experts=args.cuda_experts,
        split_boundary=args.split_boundary,
        current_cuda_experts=args.current_cuda_experts,
    )
    print("=== Route topology simulation ===")
    print(
        f"trace={args.trace} dimensions={header.num_layers} layers x "
        f"{header.num_experts} experts top_k={header.top_k}"
    )
    print(
        f"selection={selection}; selected records={len(records)} "
        f"groups={kept_groups}; excluded multi-row groups={excluded_groups}"
    )
    if selected_runs:
        print(
            "selected sequence step ranges="
            + ",".join(f"{first}:{last}" for first, last in selected_runs)
        )
    print(
        f"candidate geometry CUDA=[0,{args.cuda_experts}) "
        f"A=[{args.cuda_experts},{args.split_boundary}) "
        f"B=[{args.split_boundary},{header.num_experts}); "
        f"baseline CUDA=[0,{args.current_cuda_experts})"
    )
    print(
        "cost model: k->device-kernel-us "
        + ", ".join(f"{k}:{cost:.1f}" for k, cost in enumerate(KERNEL_US))
    )
    print("interpolation: none (every simulated route count is an integer measured knot)")
    print(
        "zero-route note: fractions identify dispatches where an all--1 host-side "
        "skip can avoid the approximately 158 us round trip; that host saving is "
        "not added to the device-only kernel estimates"
    )
    print()
    for summary in summaries:
        print(format_topology(summary))
        print()
    if not args.no_sweep:
        rows = sweep_boundaries(header, records, cuda_experts=args.cuda_experts)
        print(format_sweep(rows, selected_boundary=args.split_boundary))
        print()
    if args.with_locality:
        locality = analyze_locality(
            header,
            records,
            expert_bytes=args.expert_bytes,
        )
        print("=== Same-record temporal locality and cache analysis ===")
        print(
            "The following report consumes exactly the selected topology records "
            "shown above, not a second parse or a broader prefill-inclusive set."
        )
        print(format_locality_report(locality))
        print()
    print("=== Real-trace capture requirement ===")
    print(
        "Use strictly sequential concurrency-1 requests with multi-token prompts. "
        "The v1 layout has no phase or request ID. Multi-row `(layer,step)` groups "
        "are identifiable prefill and are removed; their resulting step gaps split "
        "the remaining decode records into request runs. `--last-sequences 10` keeps "
        "the ten tail runs and discards startup/profile forwards. A one-token prefill "
        "or continuous batching is not identifiable from v1 and is unsupported."
    )
    print(
        "The written fields `(step,layer,row,expert IDs)` are sufficient for that "
        "strictly sequential capture. They are not sufficient to attribute rows to "
        "requests under continuous batching because row slots can be reused."
    )
    print(
        "Capture at least 12,000 decode tokens total across at least 10 independent, "
        "diverse sequences (1,200 forced tokens each). This is about 576,000 records "
        "for 48 routed layers and gives a worst-case independent-token Hoeffding "
        "half-width below 0.1 route at 95%; multiple sequences remain necessary for "
        "correlation and prompt diversity. One sequence characterizes only itself."
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="write a synthetic production-format trace")
    generate.add_argument("output", type=Path)
    generate.add_argument("--mode", choices=("uniform", "id-clustered"), required=True)
    generate.add_argument("--tokens", type=int, default=100_000)
    generate.add_argument("--layers", type=int, default=1)
    generate.add_argument("--experts", type=int, default=DEFAULT_NUM_EXPERTS)
    generate.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    generate.add_argument("--cuda-experts", type=int, default=DEFAULT_CUDA_EXPERTS)
    generate.add_argument("--split-boundary", type=int, default=DEFAULT_SPLIT_BOUNDARY)
    generate.add_argument("--seed", type=int, default=0xB70)
    generate.set_defaults(func=generate_trace)

    analyze = subparsers.add_parser("analyze", help="simulate placements from a route trace")
    analyze.add_argument("trace", type=Path)
    analyze.add_argument("--cuda-experts", type=int, default=DEFAULT_CUDA_EXPERTS)
    analyze.add_argument("--split-boundary", type=int, default=DEFAULT_SPLIT_BOUNDARY)
    analyze.add_argument(
        "--current-cuda-experts", type=int, default=DEFAULT_CURRENT_CUDA_EXPERTS
    )
    analyze.add_argument(
        "--include-multirow",
        action="store_true",
        help="include prefill/continuous-batch rows instead of selecting single-row forwards",
    )
    analyze.add_argument(
        "--last-sequences",
        type=int,
        default=None,
        help=(
            "after single-row filtering, keep the last N consecutive global-step "
            "runs; use for strictly sequential captures to remove startup forwards"
        ),
    )
    analyze.add_argument(
        "--with-locality",
        action="store_true",
        help="append route_locality.py's report over exactly the same selected records",
    )
    analyze.add_argument(
        "--expert-bytes",
        type=int,
        default=None,
        help="expert payload bytes passed to the side-by-side locality report",
    )
    analyze.add_argument("--no-sweep", action="store_true")
    analyze.set_defaults(func=analyze_trace_file)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
