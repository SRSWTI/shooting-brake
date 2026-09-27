"""Analyse Nsight graph-node captures alongside existing native service records.

Device activity unions are not summed across overlapping engines. Native service
includes dispatch and transfers, not just Intel computation. Uncovered time is
reported without assigning it to an unmeasured cause.
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path


def merge_intervals(intervals):
    result = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("negative activity interval")
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def duration(intervals):
    return sum(end - start for start, end in merge_intervals(intervals))


def intersection_duration(left, right):
    left, right = merge_intervals(left), merge_intervals(right)
    i = j = total = 0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def category(name):
    if "GroupProblemShape" in name:
        return "grouped_expert_gemm"
    if "cutlass::device_kernel" in name and "Gemm" in name:
        return "dense_gemm_unattributed_module"
    if "flash_fwd" in name:
        return "attention"
    if "cvt_fp16_to_fp4" in name:
        return "activation_fp4_quantization"
    if "moeTopK" in name:
        return "router_topk"
    if "gemvx" in name:
        return "gemv_unattributed_module"
    if "elementwise" in name:
        return "elementwise"
    if "norm" in name.lower():
        return "normalization"
    return "other_kernel"


def analyse(database: Path, worker_path: Path):
    worker = json.loads(worker_path.read_text())
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        markers = {
            row["text"]: row
            for row in connection.execute(
                "SELECT start,end,text FROM NVTX_EVENTS WHERE text LIKE 'sb.clock.%'"
            )
        }
        clock_bounds = []
        for sample in worker["clock_start"] + worker["clock_stop"]:
            marker = markers[sample["nvtx_name"]]
            # Both monotonic sample and NVTX event bracket the same host operation.
            clock_bounds.append(
                (
                    marker["start"] - sample["host_after_ns"],
                    marker["end"] - sample["host_before_ns"],
                )
            )
        lower = max(bound[0] for bound in clock_bounds)
        upper = min(bound[1] for bound in clock_bounds)
        if lower > upper:
            raise ValueError("clock correlation intervals do not intersect")
        offset = (lower + upper) // 2
        steps = []
        for row in connection.execute(
            "SELECT start,end,text FROM NVTX_EVENTS WHERE text LIKE 'execute_%' ORDER BY start"
        ):
            match = re.match(r"execute_(\d+)_context_(\d+).*_generation_(\d+)", row["text"])
            if not match:
                raise ValueError(f"unrecognised model-step annotation: {row['text']}")
            tokens, context, generation = map(int, match.groups())
            steps.append(
                {
                    "cpu_start": row["start"],
                    "cpu_end": row["end"],
                    "scheduled_tokens": tokens,
                    "phase": "mixed"
                    if context and generation
                    else "prefill"
                    if context
                    else "decode",
                    "kernels": [],
                    "copies": [],
                    "categories": defaultdict(list),
                    "native": defaultdict(list),
                }
            )
        starts = [step["cpu_start"] for step in steps]
        orphan = defaultdict(list)
        for table, kind, names in (
            ("CUPTI_ACTIVITY_KIND_KERNEL", "kernels", "JOIN StringIds s ON s.id=a.demangledName"),
            ("CUPTI_ACTIVITY_KIND_MEMCPY", "copies", ""),
        ):
            name_column = "s.value" if kind == "kernels" else "'memcpy'"
            query = f"SELECT a.start,a.end,r.start AS api_start,{name_column} AS name FROM {table} a LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=a.correlationId {names} ORDER BY a.start"
            for row in connection.execute(query):
                interval = (row["start"], row["end"])
                index = (
                    bisect.bisect_right(starts, row["api_start"]) - 1
                    if row["api_start"] is not None
                    else -1
                )
                if index < 0 or row["api_start"] > steps[index]["cpu_end"]:
                    orphan[kind].append(interval)
                    continue
                step = steps[index]
                step[kind].append(interval)
                if kind == "kernels":
                    step["categories"][category(row["name"])].append(interval)
        gpu_steps = []
        for step in steps:
            activities = step["kernels"] + step["copies"]
            if activities:
                step["gpu_start"] = min(start for start, end in activities)
                step["gpu_end"] = max(end for start, end in activities)
                gpu_steps.append(step)
        gpu_steps.sort(key=lambda step: step["gpu_start"])
        gpu_starts = [step["gpu_start"] for step in gpu_steps]
        unmatched_native = 0
        for device, record in worker["devices"].items():
            for entry in record["entries"]:
                interval = (entry["t0_ns"] + offset, entry["t1_ns"] + offset)
                index = bisect.bisect_right(gpu_starts, interval[0]) - 1
                if index < 0 or interval[1] > gpu_steps[index]["gpu_end"] + (upper - lower):
                    unmatched_native += 1
                else:
                    gpu_steps[index]["native"][device].append(interval)
        phases = defaultdict(list)
        for step in gpu_steps:
            span = step["gpu_end"] - step["gpu_start"]
            cuda = step["kernels"] + step["copies"]
            native = [interval for intervals in step["native"].values() for interval in intervals]
            native_union = duration(native)
            overlap = intersection_duration(cuda, native)
            phases[step["phase"]].append(
                {
                    "scheduled_tokens": step["scheduled_tokens"],
                    "gpu_activity_span_ms": span / 1e6,
                    "cuda_kernel_union_ms": duration(step["kernels"]) / 1e6,
                    "cuda_copy_union_ms": duration(step["copies"]) / 1e6,
                    "native_service_union_ms": native_union / 1e6,
                    "native_service_without_cuda_activity_ms": (native_union - overlap) / 1e6,
                    "uncovered_span_ms": max(0, span - duration(cuda + native)) / 1e6,
                    "per_device_service_ms": {
                        device: duration(intervals) / 1e6
                        for device, intervals in step["native"].items()
                    },
                    "kernel_categories_ms": {
                        name: duration(intervals) / 1e6
                        for name, intervals in step["categories"].items()
                    },
                }
            )
        summary = {}
        for phase, records in phases.items():
            numeric_keys = [
                key for key, value in records[0].items() if isinstance(value, (int, float))
            ]
            summary[phase] = {
                "steps": len(records),
                **{
                    key: {
                        "median": statistics.median(row[key] for row in records),
                        "mean": statistics.mean(row[key] for row in records),
                    }
                    for key in numeric_keys
                },
            }
            names = {name for row in records for name in row["kernel_categories_ms"]}
            summary[phase]["mean_kernel_categories_ms"] = {
                name: statistics.mean(row["kernel_categories_ms"].get(name, 0) for row in records)
                for name in sorted(names)
            }
        return {
            "label": worker["label"],
            "database": str(database),
            "worker": str(worker_path),
            "clock_offset_ns": offset,
            "clock_offset_uncertainty_ns": upper - lower,
            "unmatched_native_entries": unmatched_native,
            "unattributed_cuda_activity_ms": {
                kind: duration(intervals) / 1e6 for kind, intervals in orphan.items()
            },
            "summary": summary,
            "steps": dict(phases),
            "limitations": [
                "Native service is a host-observed envelope, not Intel kernel time.",
                "Activity without concurrent CUDA activity is not by itself proof of critical-path causality.",
                "CPU-range correlation excludes CUDA work submitted outside execute_model; it is reported unattributed.",
                "Profiled timings are not unprofiled serving performance.",
            ],
        }
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyse(args.database, args.worker)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as destination:
        json.dump(result, destination, indent=2, allow_nan=False)
    print(json.dumps({key: value for key, value in result.items() if key != "steps"}, indent=2))


if __name__ == "__main__":
    main()
