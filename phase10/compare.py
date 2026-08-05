"""Phase 10 — run both configurations and diff them.

Each configuration needs its own process: the adapter reads its
environment at class-construction time, and only one engine can own the
GPU at a time.  This script runs them in sequence and prints the
comparison required by plan.md's Phase 10 gate.

Usage::

    python phase10/compare.py --out-dir phase10/results
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

CONFIGS = ("all-cuda", "hybrid")


def run_config(
    config: str, out: Path, extra: list[str], python: str
) -> dict[str, Any]:
    """Run one configuration in a fresh process."""
    here = Path(__file__).resolve().parent
    cmd = [
        python, str(here / "benchmark.py"),
        "--config", config, "--out", str(out), *extra,
    ]
    print(f"\n{'=' * 70}\nrunning {config}\n{'=' * 70}", flush=True)
    result = subprocess.run(cmd, cwd=here.parent, env=os.environ.copy())
    if result.returncode != 0:
        raise SystemExit(f"{config} benchmark failed ({result.returncode})")
    return json.loads(out.read_text())


def compare_correctness(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> tuple[int, int]:
    """Token-level agreement across the prompt matrix.

    Temperature is 0 in both runs, so every sequence must match exactly.
    """
    print(f"\n{'=' * 70}\ncorrectness: generated-token agreement\n{'=' * 70}")
    matches = 0
    rows = list(
        zip(baseline["correctness"], candidate["correctness"], strict=True)
    )
    for base, cand in rows:
        same = base["token_ids"] == cand["token_ids"]
        matches += same
        mark = "ok  " if same else "DIFF"
        print(f"  [{mark}] {base['prompt'][:58]}")
        if not same:
            print(f"         all-cuda: {base['token_ids'][:12]}")
            print(f"         hybrid  : {cand['token_ids'][:12]}")
    print(f"\n  {matches}/{len(rows)} sequences identical")
    return matches, len(rows)


def _delta(candidate: float, baseline: float) -> str:
    """Candidate as a percentage of baseline, with sign."""
    if baseline == 0:
        return "n/a"
    ratio = candidate / baseline
    return f"{ratio * 100:5.1f}% of baseline ({(ratio - 1) * 100:+.1f}%)"


def compare_throughput(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> None:
    print(f"\n{'=' * 70}\nsingle-stream latency and throughput\n{'=' * 70}")
    b, c = baseline["single_stream"], candidate["single_stream"]
    print(f"  {'metric':<22} {'all-cuda':>12} {'hybrid':>12}   relative")
    for key, label, unit in (
        ("tok_per_s_mean", "output", "tok/s"),
        ("decode_tok_per_s_mean", "decode only", "tok/s"),
        ("itl_p50_ms", "ITL p50", "ms"),
        ("itl_p95_ms", "ITL p95", "ms"),
        ("itl_p99_ms", "ITL p99", "ms"),
        ("itl_max_ms", "ITL max", "ms"),
        ("ttft_mean_ms", "TTFT mean", "ms"),
    ):
        print(
            f"  {label + ' (' + unit + ')':<22} {b[key]:>12.2f} "
            f"{c[key]:>12.2f}   {_delta(c[key], b[key])}"
        )

    print(f"\n{'=' * 70}\nbatched throughput\n{'=' * 70}")
    print(
        f"  {'conc':>5} {'all-cuda':>12} {'hybrid':>12}   relative"
        f"      {'ITL p50 base':>13} {'ITL p50 hyb':>12}"
    )
    for row_b, row_c in zip(
        baseline["batched"], candidate["batched"], strict=True
    ):
        print(
            f"  {row_b['concurrency']:>5} "
            f"{row_b['output_tok_per_s']:>12.1f} "
            f"{row_c['output_tok_per_s']:>12.1f}   "
            f"{_delta(row_c['output_tok_per_s'], row_b['output_tok_per_s'])}"
            f"      {row_b['itl_p50_ms']:>13.2f} {row_c['itl_p50_ms']:>12.2f}"
        )

    print(f"\n{'=' * 70}\nprefill\n{'=' * 70}")
    pb, pc = baseline["prefill"], candidate["prefill"]
    print(
        f"  {pb['prompt_tokens']} prompt tokens: "
        f"{pb['prefill_tok_per_s']:.0f} vs {pc['prefill_tok_per_s']:.0f} tok/s "
        f"  {_delta(pc['prefill_tok_per_s'], pb['prefill_tok_per_s'])}"
    )


def compare_capacity(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> None:
    """The reason the hybrid path exists: freed VRAM."""
    print(f"\n{'=' * 70}\ncapacity and routing\n{'=' * 70}")
    for label, result in (("all-cuda", baseline), ("hybrid", candidate)):
        for worker in result["workers"]:
            mem = worker["cuda_memory"]
            print(
                f"  [{label:<8}] {mem['allocated_gib']:6.2f} GiB allocated, "
                f"{mem['reserved_gib']:6.2f} GiB reserved, "
                f"{mem['free_gib']:6.2f} GiB free"
            )
            routes = worker.get("routes")
            if routes:
                print(
                    f"  [{label:<8}] routes: {routes['b70']:,} of "
                    f"{routes['total']:,} to B70 "
                    f"({routes['b70_share'] * 100:.1f}%)"
                )
                per_layer = routes.get("per_layer_b70") or {}
                if per_layer:
                    active = sorted(
                        (int(k) for k, v in per_layer.items() if v), key=int
                    )
                    print(
                        f"  [{label:<8}] B70-active layers: "
                        f"{len(active)} ({min(active)}..{max(active)})"
                    )
            poller = worker.get("poller")
            if poller:
                print(
                    f"  [{label:<8}] poller: {poller['dispatches']:,} "
                    f"dispatches, service mean "
                    f"{poller['service_mean_us']:.1f} us, "
                    f"{poller['errors']} errors"
                )

    base_free = baseline["workers"][0]["cuda_memory"]["free_gib"]
    cand_free = candidate["workers"][0]["cuda_memory"]["free_gib"]
    print(f"\n  VRAM freed by hybrid: {cand_free - base_free:+.2f} GiB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("phase10/results"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--only", choices=CONFIGS, help="run one configuration and stop"
    )
    parser.add_argument(
        "bench_args", nargs="*",
        help="extra arguments forwarded to benchmark.py",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    configs = (args.only,) if args.only else CONFIGS
    results = {
        config: run_config(
            config, args.out_dir / f"{config}.json", args.bench_args,
            args.python,
        )
        for config in configs
    }
    if len(results) < 2:
        return 0

    baseline, candidate = results["all-cuda"], results["hybrid"]
    matches, total = compare_correctness(baseline, candidate)
    compare_throughput(baseline, candidate)
    compare_capacity(baseline, candidate)

    summary = args.out_dir / "comparison.json"
    summary.write_text(json.dumps({
        "token_agreement": {"matched": matches, "total": total},
        "single_stream": {
            "all_cuda_tok_per_s": baseline["single_stream"]["tok_per_s_mean"],
            "hybrid_tok_per_s": candidate["single_stream"]["tok_per_s_mean"],
        },
        "batched": [
            {
                "concurrency": rb["concurrency"],
                "all_cuda_tok_per_s": rb["output_tok_per_s"],
                "hybrid_tok_per_s": rc["output_tok_per_s"],
            }
            for rb, rc in zip(
                baseline["batched"], candidate["batched"], strict=True
            )
        ],
    }, indent=2))
    print(f"\nwrote {summary}")

    if matches != total:
        print("\nFAIL: generated tokens differ between configurations")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
