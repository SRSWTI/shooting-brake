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


def _common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def compare_correctness(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Generated-token agreement across the prompt matrix.

    Both runs decode greedily, so a *bit-identical* B70 would reproduce
    the baseline token for token.  It does not: the B70's NVFP4 kernel
    and CUDA's differ in the last bits, and greedy decoding turns any
    near-tie into a hard fork, after which the sequences separate for
    good.  So exact-match count alone is misleading — the divergence
    index says how long the two agree, and the text says whether the
    hybrid output is still a correct answer or is degenerate.
    """
    print(f"\n{'=' * 70}\ncorrectness: generated-token agreement\n{'=' * 70}")
    rows = list(
        zip(baseline["correctness"], candidate["correctness"], strict=True)
    )
    exact = 0
    prefixes: list[int] = []
    for base, cand in rows:
        shared = _common_prefix(base["token_ids"], cand["token_ids"])
        same = base["token_ids"] == cand["token_ids"]
        exact += same
        prefixes.append(shared)
        mark = "ok  " if same else f"fork@{shared:<3}"
        print(f"  [{mark}] {base['prompt'][:56]}")
        if not same:
            print(f"         all-cuda: {base['text'][:88]!r}")
            print(f"         hybrid  : {cand['text'][:88]!r}")
    mean_prefix = sum(prefixes) / len(prefixes)
    print(
        f"\n  {exact}/{len(rows)} sequences identical; "
        f"mean agreement before divergence: {mean_prefix:.1f} tokens"
    )
    return {
        "exact": exact,
        "total": len(rows),
        "mean_common_prefix": mean_prefix,
        "common_prefix": prefixes,
    }


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
    """The reason the hybrid path exists: KV cache capacity.

    Free VRAM is not the metric — vLLM allocates up to
    ``gpu_memory_utilization`` either way, so both configurations end
    with a similar free figure. What the freed expert weights actually
    buy is KV cache, and therefore concurrent requests and context.
    """
    print(f"\n{'=' * 70}\ncapacity and routing\n{'=' * 70}")
    for label, result in (("all-cuda", baseline), ("hybrid", candidate)):
        for worker in result["workers"]:
            mem = worker["cuda_memory"]
            kv = worker["kv_cache"]
            print(
                f"  [{label:<8}] weights+state {mem['allocated_gib']:6.2f} GiB, "
                f"KV cache {kv['max_tokens']:>9,} tokens "
                f"({kv['num_gpu_blocks']:,} blocks)"
            )
            routes = worker.get("routes")
            if routes:
                print(
                    f"  [{label:<8}] routes: {routes['b70']:,} of "
                    f"{routes['total']:,} to B70 "
                    f"({routes['b70_share'] * 100:.1f}%)"
                )
                # All-CUDA reports every layer with a zero count, so
                # filter before taking a range.
                active = sorted(
                    int(layer)
                    for layer, count in (routes.get("per_layer_b70") or {}).items()
                    if count
                )
                if active:
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

    base_kv = baseline["workers"][0]["kv_cache"]["max_tokens"]
    cand_kv = candidate["workers"][0]["kv_cache"]["max_tokens"]
    if base_kv:
        print(
            f"\n  KV capacity: {base_kv:,} -> {cand_kv:,} tokens "
            f"({cand_kv / base_kv:.2f}x)"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("phase10/results"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--only", choices=CONFIGS, help="run one configuration and stop"
    )
    parser.add_argument(
        "--from-results", action="store_true",
        help="re-report from saved JSON without re-running the benchmarks",
    )
    parser.add_argument(
        "bench_args", nargs="*",
        help="extra arguments forwarded to benchmark.py",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    configs = (args.only,) if args.only else CONFIGS
    if args.from_results:
        results = {
            config: json.loads((args.out_dir / f"{config}.json").read_text())
            for config in configs
        }
    else:
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
    agreement = compare_correctness(baseline, candidate)
    compare_throughput(baseline, candidate)
    compare_capacity(baseline, candidate)

    base_kv = baseline["workers"][0]["kv_cache"]["max_tokens"]
    cand_kv = candidate["workers"][0]["kv_cache"]["max_tokens"]
    summary = args.out_dir / "comparison.json"
    summary.write_text(json.dumps({
        "token_agreement": agreement,
        "kv_cache_tokens": {"all_cuda": base_kv, "hybrid": cand_kv},
        "single_stream": {
            "all_cuda_tok_per_s": baseline["single_stream"]["tok_per_s_mean"],
            "hybrid_tok_per_s": candidate["single_stream"]["tok_per_s_mean"],
            "all_cuda_itl_p50_ms": baseline["single_stream"]["itl_p50_ms"],
            "hybrid_itl_p50_ms": candidate["single_stream"]["itl_p50_ms"],
        },
        "batched": [
            {
                "concurrency": rb["concurrency"],
                "all_cuda_tok_per_s": rb["output_tok_per_s"],
                "hybrid_tok_per_s": rc["output_tok_per_s"],
                "all_cuda_itl_p50_ms": rb["itl_p50_ms"],
                "hybrid_itl_p50_ms": rc["itl_p50_ms"],
            }
            for rb, rc in zip(
                baseline["batched"], candidate["batched"], strict=True
            )
        ],
    }, indent=2))
    print(f"\nwrote {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
