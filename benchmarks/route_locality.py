#!/usr/bin/env python3
"""Analyze temporal expert locality in a Shooting Brake route trace.

The input format is emitted when ``SHOOTING_BRAKE_ROUTE_TRACE=<path>`` is set.
It is a 24-byte little-endian ``<8sHHHHII`` header followed by packed records;
see ``shooting_brake_vllm.route_stats`` for the authoritative layout.  This
analyzer intentionally has no torch, vLLM, model, or GPU dependency.

Rows are treated as sequence slots.  A temporal comparison is made only when
one layer has records for the same row at exactly consecutive step indices.
This avoids joining across a row that disappeared for one or more forwards.
Continuous batching can reuse a row for a new request without exposing that
identity in ``topk_ids``; a runner that needs request-identity-perfect results
must keep row assignment stable for the measured interval.
"""

from __future__ import annotations

import argparse
import math
import random
import struct
import sys
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

TRACE_MAGIC = b"SBRTv1\0\0"
TRACE_VERSION = 1
TRACE_HEADER = struct.Struct("<8sHHHHII")
WINDOWS = (1, 2, 4, 8, 16, 32)
CACHE_CAPACITIES = (8, 16, 32, 64, 128)
_BASELINE_TOKENS = 100_000


@dataclass(frozen=True)
class TraceHeader:
    """Dimensions encoded in a route-trace header."""

    top_k: int
    num_layers: int
    num_experts: int
    record_size: int


def record_dtype(top_k: int) -> np.dtype:
    """Return the packed NumPy dtype for one trace record."""
    return np.dtype(
        [
            ("step", "<u8"),
            ("layer", "<u2"),
            ("row", "<u4"),
            ("experts", "<i4", (top_k,)),
        ]
    )


def read_trace(path: str | Path) -> tuple[TraceHeader, np.ndarray]:
    """Read and validate a binary route trace."""
    source = Path(path)
    with source.open("rb") as stream:
        raw = stream.read(TRACE_HEADER.size)
        if len(raw) != TRACE_HEADER.size:
            raise ValueError(f"{source}: truncated trace header")
        magic, version, header_size, rec_size, top_k, layers, experts = TRACE_HEADER.unpack(raw)
        if magic != TRACE_MAGIC:
            raise ValueError(f"{source}: bad route-trace magic {magic!r}")
        if version != TRACE_VERSION:
            raise ValueError(f"{source}: unsupported route-trace version {version}")
        if header_size != TRACE_HEADER.size:
            raise ValueError(f"{source}: header size {header_size}, expected {TRACE_HEADER.size}")
        if not top_k or not layers or not experts or top_k > experts:
            raise ValueError(
                f"{source}: invalid dimensions top_k={top_k}, layers={layers}, experts={experts}"
            )
        dtype = record_dtype(top_k)
        if rec_size != dtype.itemsize:
            raise ValueError(f"{source}: record size {rec_size}, expected {dtype.itemsize}")
        payload_bytes = source.stat().st_size - header_size
        if payload_bytes % rec_size:
            raise ValueError(
                f"{source}: truncated final record "
                f"({payload_bytes} payload bytes is not divisible by {rec_size})"
            )
        records = np.fromfile(stream, dtype=dtype)

    if records.size:
        if int(records["layer"].max()) >= layers:
            raise ValueError(f"{source}: record has out-of-range layer index")
        ids = records["experts"]
        if int(ids.min()) < 0 or int(ids.max()) >= experts:
            raise ValueError(f"{source}: record has out-of-range expert id")
        sorted_ids = np.sort(ids, axis=1)
        if top_k > 1 and bool(np.any(sorted_ids[:, 1:] == sorted_ids[:, :-1])):
            raise ValueError(f"{source}: a top-k record contains duplicate expert ids")
        order = np.lexsort((records["row"], records["step"], records["layer"]))
        ordered = records[order]
        if len(ordered) > 1:
            duplicate = (
                (ordered["layer"][1:] == ordered["layer"][:-1])
                & (ordered["step"][1:] == ordered["step"][:-1])
                & (ordered["row"][1:] == ordered["row"][:-1])
            )
            if bool(np.any(duplicate)):
                raise ValueError(f"{source}: duplicate (layer, step, row) record")

    return TraceHeader(top_k, layers, experts, rec_size), records


def write_trace(
    path: str | Path,
    records: Iterable[tuple[int, int, int, Sequence[int]]],
    *,
    num_layers: int,
    num_experts: int,
    top_k: int,
    block_records: int = 4096,
) -> Path:
    """Write synthetic or converted records in the production trace format.

    This helper is for offline generation and tests; the live hook uses its own
    pinned, asynchronous writer in ``route_stats.py``.
    """
    if not (0 < top_k <= num_experts and num_layers > 0 and block_records > 0):
        raise ValueError("invalid trace dimensions")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    dtype = record_dtype(top_k)
    header = TRACE_HEADER.pack(
        TRACE_MAGIC,
        TRACE_VERSION,
        TRACE_HEADER.size,
        dtype.itemsize,
        top_k,
        num_layers,
        num_experts,
    )
    block = np.empty(block_records, dtype=dtype)
    used = 0
    with target.open("wb") as stream:
        stream.write(header)
        for step, layer, row, expert_ids in records:
            ids = tuple(int(expert) for expert in expert_ids)
            if len(ids) != top_k:
                raise ValueError(f"record has {len(ids)} experts, expected {top_k}")
            if not 0 <= layer < num_layers or step < 0 or row < 0:
                raise ValueError("negative or out-of-range record coordinate")
            if len(set(ids)) != top_k or any(expert < 0 or expert >= num_experts for expert in ids):
                raise ValueError("expert ids must be a unique in-range top-k set")
            block["step"][used] = step
            block["layer"][used] = layer
            block["row"][used] = row
            block["experts"][used] = ids
            used += 1
            if used == len(block):
                block.tofile(stream)
                used = 0
        if used:
            block[:used].tofile(stream)
    return target


def uniform_jaccard(top_k: int, num_experts: int) -> float:
    """Exact expected Jaccard of two independent uniform top-k subsets."""
    denominator = math.comb(num_experts, top_k)
    expected = 0.0
    minimum = max(0, 2 * top_k - num_experts)
    for intersection in range(minimum, top_k + 1):
        ways = math.comb(top_k, intersection) * math.comb(num_experts - top_k, top_k - intersection)
        union = 2 * top_k - intersection
        expected += (ways / denominator) * (intersection / union)
    return expected


def uniform_working_set(
    top_k: int,
    num_experts: int,
    window: int,
) -> float:
    """Exact expected union size across ``window`` uniform top-k subsets."""
    absent_probability = (1.0 - top_k / num_experts) ** window
    return num_experts * (1.0 - absent_probability)


@cache
def _uniform_lru_rates(
    top_k: int,
    num_experts: int,
    capacities: tuple[int, ...],
) -> tuple[float, ...]:
    """Deterministic Monte Carlo LRU null for uniform top-k routing.

    The no-replacement constraint within one token materially lowers the
    small-cache null below the common ``capacity / E`` approximation.  The
    simulation therefore draws uniform top-k *sets*, in random order, and uses
    a fixed seed and 100,000 tokens.  It is shared by every layer and cached.
    """
    rng = random.Random((num_experts << 16) ^ (top_k << 8) ^ 0x5B17)
    caches = [OrderedDict() for _ in capacities]
    hits = [0] * len(capacities)
    requests = _BASELINE_TOKENS * top_k
    population = range(num_experts)
    for _ in range(_BASELINE_TOKENS):
        for expert in rng.sample(population, top_k):
            for index, (capacity, entries) in enumerate(zip(capacities, caches)):
                if expert in entries:
                    hits[index] += 1
                    entries.move_to_end(expert)
                else:
                    entries[expert] = None
                    if len(entries) > capacity:
                        entries.popitem(last=False)
    return tuple(hit / requests for hit in hits)


def uniform_baseline(
    top_k: int,
    num_experts: int,
    *,
    windows: tuple[int, ...] = WINDOWS,
    capacities: tuple[int, ...] = CACHE_CAPACITIES,
) -> dict:
    """Return chance locality for the trace's exact ``k`` and ``E``."""
    rates = _uniform_lru_rates(top_k, num_experts, capacities)
    return {
        "jaccard": uniform_jaccard(top_k, num_experts),
        "working_set": {
            window: uniform_working_set(top_k, num_experts, window) for window in windows
        },
        "lru_hit_rate": dict(zip(capacities, rates)),
        "lru_method": (f"fixed-seed uniform top-k simulation, {_BASELINE_TOKENS} tokens"),
    }


def _consecutive_runs(layer_records: np.ndarray) -> list[np.ndarray]:
    """Split one layer into same-row runs with exactly consecutive steps."""
    if not len(layer_records):
        return []
    order = np.lexsort((layer_records["step"], layer_records["row"]))
    sequenced = layer_records[order]
    runs: list[np.ndarray] = []
    start = 0
    for index in range(1, len(sequenced)):
        if (
            sequenced["row"][index] != sequenced["row"][index - 1]
            or sequenced["step"][index] != sequenced["step"][index - 1] + 1
        ):
            runs.append(sequenced[start:index])
            start = index
    runs.append(sequenced[start:])
    return runs


def _working_set_sums(
    runs: Sequence[np.ndarray],
    windows: tuple[int, ...],
) -> dict[int, tuple[int, int]]:
    totals = {window: [0, 0] for window in windows}
    for run in runs:
        route_sets = [set(map(int, experts)) for experts in run["experts"]]
        for window in windows:
            if len(route_sets) < window:
                continue
            counts: dict[int, int] = {}
            distinct = 0
            for index, experts in enumerate(route_sets):
                for expert in experts:
                    old = counts.get(expert, 0)
                    counts[expert] = old + 1
                    if old == 0:
                        distinct += 1
                if index >= window:
                    for expert in route_sets[index - window]:
                        remaining = counts[expert] - 1
                        if remaining:
                            counts[expert] = remaining
                        else:
                            del counts[expert]
                            distinct -= 1
                if index + 1 >= window:
                    totals[window][0] += distinct
                    totals[window][1] += 1
    return {window: (values[0], values[1]) for window, values in totals.items()}


def _lru_counts(
    layer_records: np.ndarray,
    capacities: tuple[int, ...],
) -> dict[int, tuple[int, int]]:
    """Return ``capacity -> (hits, requests)`` in forward/row order."""
    order = np.lexsort((layer_records["row"], layer_records["step"]))
    routed = layer_records[order]
    caches = [OrderedDict() for _ in capacities]
    hits = [0] * len(capacities)
    for experts in routed["experts"]:
        for raw_expert in experts:
            expert = int(raw_expert)
            for index, (capacity, entries) in enumerate(zip(capacities, caches)):
                if expert in entries:
                    hits[index] += 1
                    entries.move_to_end(expert)
                else:
                    entries[expert] = None
                    if len(entries) > capacity:
                        entries.popitem(last=False)
    requests = len(routed) * routed["experts"].shape[1]
    return {capacity: (hits[index], requests) for index, capacity in enumerate(capacities)}


def _layer_metrics(
    layer: int,
    records: np.ndarray,
    baseline: dict,
    windows: tuple[int, ...],
    capacities: tuple[int, ...],
    expert_bytes: int | None,
) -> dict:
    runs = _consecutive_runs(records)
    jaccard_sum = 0.0
    comparisons = 0
    for run in runs:
        for index in range(1, len(run)):
            previous = set(map(int, run["experts"][index - 1]))
            current = set(map(int, run["experts"][index]))
            jaccard_sum += len(previous & current) / len(previous | current)
            comparisons += 1
    jaccard = jaccard_sum / comparisons if comparisons else None

    working_counts = _working_set_sums(runs, windows)
    working_set = {}
    for window, (total, count) in working_counts.items():
        mean = total / count if count else None
        chance = baseline["working_set"][window]
        working_set[window] = {
            "mean_distinct": mean,
            "windows": count,
            "chance_distinct": chance,
            "reuse_above_chance_experts": (chance - mean if mean is not None else None),
        }

    lru = {}
    for capacity, (hits, requests) in _lru_counts(records, capacities).items():
        misses = requests - hits
        rate = hits / requests if requests else 0.0
        row = {
            "requests": requests,
            "hits": hits,
            "misses": misses,
            "hit_rate": rate,
            "chance_hit_rate": baseline["lru_hit_rate"][capacity],
            "reuse_above_chance": (rate - baseline["lru_hit_rate"][capacity]),
            "round_trips_without_cache": requests,
            "round_trips_with_cache": misses,
            "round_trips_saved": hits,
            "round_trip_reduction": rate,
            "expert_payloads_fetched": misses,
            "expert_payloads_saved": hits,
            "byte_reduction": rate,
        }
        if expert_bytes is not None:
            row["bytes_fetched"] = misses * expert_bytes
            row["bytes_saved"] = hits * expert_bytes
        lru[capacity] = row

    return {
        "layer": layer,
        "records": len(records),
        "jaccard": jaccard,
        "jaccard_comparisons": comparisons,
        "chance_jaccard": baseline["jaccard"],
        "reuse_above_chance": (jaccard - baseline["jaccard"] if jaccard is not None else None),
        "working_set": working_set,
        "lru": lru,
        "_jaccard_sum": jaccard_sum,
        "_working_counts": working_counts,
    }


def analyze_trace(
    header: TraceHeader,
    records: np.ndarray,
    *,
    windows: tuple[int, ...] = WINDOWS,
    capacities: tuple[int, ...] = CACHE_CAPACITIES,
    expert_bytes: int | None = None,
) -> dict:
    """Compute per-layer and aggregate locality metrics."""
    if expert_bytes is not None and expert_bytes <= 0:
        raise ValueError("expert_bytes must be positive")
    baseline = uniform_baseline(
        header.top_k,
        header.num_experts,
        windows=windows,
        capacities=capacities,
    )
    layers = []
    for layer in range(header.num_layers):
        selected = records[records["layer"] == layer]
        if len(selected):
            layers.append(
                _layer_metrics(
                    layer,
                    selected,
                    baseline,
                    windows,
                    capacities,
                    expert_bytes,
                )
            )

    comparisons = sum(row["jaccard_comparisons"] for row in layers)
    jaccard_sum = sum(row["_jaccard_sum"] for row in layers)
    aggregate_jaccard = jaccard_sum / comparisons if comparisons else None
    aggregate_working = {}
    for window in windows:
        total = sum(row["_working_counts"][window][0] for row in layers)
        count = sum(row["_working_counts"][window][1] for row in layers)
        mean = total / count if count else None
        chance = baseline["working_set"][window]
        aggregate_working[window] = {
            "mean_distinct": mean,
            "windows": count,
            "chance_distinct": chance,
            "reuse_above_chance_experts": (chance - mean if mean is not None else None),
        }

    aggregate_lru = {}
    for capacity in capacities:
        hits = sum(row["lru"][capacity]["hits"] for row in layers)
        requests = sum(row["lru"][capacity]["requests"] for row in layers)
        misses = requests - hits
        rate = hits / requests if requests else 0.0
        aggregate_lru[capacity] = {
            "requests": requests,
            "hits": hits,
            "misses": misses,
            "hit_rate": rate,
            "chance_hit_rate": baseline["lru_hit_rate"][capacity],
            "reuse_above_chance": rate - baseline["lru_hit_rate"][capacity],
            "round_trips_without_cache": requests,
            "round_trips_with_cache": misses,
            "round_trips_saved": hits,
            "round_trip_reduction": rate,
            "expert_payloads_fetched": misses,
            "expert_payloads_saved": hits,
            "byte_reduction": rate,
        }
        if expert_bytes is not None:
            aggregate_lru[capacity]["bytes_fetched"] = misses * expert_bytes
            aggregate_lru[capacity]["bytes_saved"] = hits * expert_bytes

    aggregate = {
        "layer": "all",
        "records": sum(row["records"] for row in layers),
        "jaccard": aggregate_jaccard,
        "jaccard_comparisons": comparisons,
        "chance_jaccard": baseline["jaccard"],
        "reuse_above_chance": (
            aggregate_jaccard - baseline["jaccard"] if aggregate_jaccard is not None else None
        ),
        "working_set": aggregate_working,
        "lru": aggregate_lru,
    }
    for row in layers:
        del row["_jaccard_sum"]
        del row["_working_counts"]
    return {
        "top_k": header.top_k,
        "num_experts": header.num_experts,
        "num_layers": header.num_layers,
        "active_layers": len(layers),
        "records": len(records),
        "expert_bytes": expert_bytes,
        "baseline": baseline,
        "layers": layers,
        "aggregate": aggregate,
    }


def _metric(value: float | None, width: int = 7) -> str:
    return f"{value:{width}.4f}" if value is not None else f"{'n/a':>{width}}"


def _bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024.0 or unit == "TiB":
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    raise AssertionError("unreachable")


def format_report(summary: dict) -> str:
    """Format every required metric as a human-readable report."""
    lines = [
        "=== Route locality analysis ===",
        (
            f"{summary['records']} records; {summary['active_layers']}/"
            f"{summary['num_layers']} layers; top-{summary['top_k']} of "
            f"{summary['num_experts']} experts"
        ),
        (
            "Rows are sequence slots; comparisons require the same row on "
            "exactly consecutive per-layer steps."
        ),
        "",
        "--- Consecutive-token top-k Jaccard ---",
        f"uniform chance = {summary['baseline']['jaccard']:.6f}",
        (f"{'layer':>5} {'pairs':>10} {'observed':>9} {'chance':>9} {'above':>9}"),
    ]
    for row in [*summary["layers"], summary["aggregate"]]:
        label = str(row["layer"])
        lines.append(
            f"{label:>5} {row['jaccard_comparisons']:>10} "
            f"{_metric(row['jaccard'], 9)} "
            f"{row['chance_jaccard']:>9.4f} "
            f"{_metric(row['reuse_above_chance'], 9)}"
        )

    lines += [
        "",
        "--- Distinct-expert working set (observed/chance) ---",
        f"{'layer':>5} " + " ".join(f"w{window:>2}".rjust(15) for window in WINDOWS),
    ]
    for row in [*summary["layers"], summary["aggregate"]]:
        cells = []
        for window in WINDOWS:
            metric = row["working_set"][window]
            observed = metric["mean_distinct"]
            cells.append(
                f"{observed:.2f}/{metric['chance_distinct']:.2f}"
                if observed is not None
                else f"n/a/{metric['chance_distinct']:.2f}"
            )
        lines.append(f"{row['layer']!s:>5} " + " ".join(f"{cell:>15}" for cell in cells))
    lines += [
        "Lower than chance means a smaller working set from temporal reuse.",
        "",
        "--- LRU expert-weight cache ---",
        "Uniform null: " + summary["baseline"]["lru_method"],
        (
            "Each miss implies one expert-weight round trip and one equal-sized "
            "expert payload fetched."
        ),
        (
            f"{'layer':>5} {'S':>4} {'requests':>10} {'hit%':>8} "
            f"{'chance%':>8} {'above pp':>9} {'trips':>10} {'saved':>10} "
            f"{'bytes fetch/save':>19}"
        ),
    ]
    expert_bytes = summary["expert_bytes"]
    for row in [*summary["layers"], summary["aggregate"]]:
        for capacity in CACHE_CAPACITIES:
            metric = row["lru"][capacity]
            if expert_bytes is None:
                byte_text = (
                    f"{metric['expert_payloads_fetched']}/"
                    f"{metric['expert_payloads_saved']} payloads"
                )
            else:
                byte_text = f"{_bytes(metric['bytes_fetched'])}/{_bytes(metric['bytes_saved'])}"
            lines.append(
                f"{row['layer']!s:>5} {capacity:>4} "
                f"{metric['requests']:>10} {100 * metric['hit_rate']:>7.2f}% "
                f"{100 * metric['chance_hit_rate']:>7.2f}% "
                f"{100 * metric['reuse_above_chance']:>+8.2f} "
                f"{metric['round_trips_with_cache']:>10} "
                f"{metric['round_trips_saved']:>10} {byte_text:>19}"
            )
    lines += [
        "",
        (
            "Trip reduction and byte reduction both equal the hit rate under "
            "the equal-sized expert-weight model."
        ),
    ]
    if expert_bytes is None:
        lines.append(
            "Byte counts are normalized expert payloads; pass --expert-bytes "
            "for absolute byte totals."
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="binary route trace")
    parser.add_argument(
        "--expert-bytes",
        type=int,
        default=None,
        help=(
            "bytes in one layer's expert weights; enables absolute byte "
            "fetch/save totals (ratios do not require it)"
        ),
    )
    args = parser.parse_args(argv)
    try:
        header, records = read_trace(args.trace)
        summary = analyze_trace(
            header,
            records,
            expert_bytes=args.expert_bytes,
        )
    except (OSError, ValueError) as exc:
        print(f"route_locality: {exc}", file=sys.stderr)
        return 2
    print(format_report(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
