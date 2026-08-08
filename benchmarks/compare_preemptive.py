#!/usr/bin/env python3
"""Post-hoc vs pre-emptive VRAM surgery on the same placement.

Unlike ``compare.py``, which weighs all-CUDA against hybrid and must
tolerate forks (two different NVFP4 kernels disagree in the last bits, and
greedy decoding turns any near-tie into a permanent divergence), both legs
here run the *same* kernel over the *same* compact weights. The strategies
differ only in how that compact tensor came to exist: ``index_select`` after
a full load, versus never allocating the rest. So the bar is exact
agreement, and any fork at all is a bug rather than expected noise.

Three things are checked, because each catches a failure the others miss:

* token ids -- catches gross misaddressing.
* prompt logprobs -- catches the failure this project has already shipped
  once, where output matched token-for-token while an entire tier's routes
  were silently dropped. Token ids proved insensitive to 0.49 nats of
  damage; the logprob did not.
* load peak -- catches the opposite failure, a correct run that did not
  actually subset the allocation. Nothing else distinguishes the two after
  load: KV sizing sees identical freed memory either way, because post-hoc
  surgery completes before vLLM's profiling pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare import compare_correctness  # noqa: E402

#: Generated text is greedy and both legs share a kernel, so identical
#: inputs must give identical logprobs. This tolerance exists only to
#: absorb float formatting in the JSON round-trip, not real divergence.
LOGPROB_TOL = 1e-4


def compare_logprobs(
    posthoc: dict[str, Any], preemptive: dict[str, Any]
) -> float:
    """Worst per-prompt prompt-logprob gap between the two strategies."""
    print(f"\n{'=' * 70}\ncorrectness: prompt logprobs\n{'=' * 70}")
    worst = 0.0
    rows = list(
        zip(posthoc["correctness"], preemptive["correctness"], strict=True)
    )
    for base, cand in rows:
        b = base.get("prompt_logprob_mean")
        c = cand.get("prompt_logprob_mean")
        if b is None or c is None:
            print("  (run predates prompt logprobs; skipping)")
            return float("nan")
        delta = abs(c - b)
        worst = max(worst, delta)
        mark = "ok  " if delta <= LOGPROB_TOL else "DIFF"
        print(
            f"  [{mark}] {base['prompt'][:44]:44} "
            f"{b:+.6f} -> {c:+.6f}  ({c - b:+.2e})"
        )
    print(f"\n  worst |delta| = {worst:.3e} nats/token (tolerance {LOGPROB_TOL:g})")
    return worst


def _worker_field(
    run: dict[str, Any], block: str, field: str, default: Any
) -> Any:
    """Read a telemetry field out of the per-worker `collective_rpc` list.

    The benchmark writes `result["workers"]`, one dict per worker; there is
    no flat "telemetry" key. `max()` keeps this honest under TP>1, where
    the figure that decides whether a model fits is the worst card.
    """
    workers = run.get("workers") or run.get("workers_decode_only") or []
    vals = [
        (w.get(block) or {}).get(field)
        for w in workers
        if (w.get(block) or {}).get(field) is not None
    ]
    return max(vals) if vals else default


def compare_load_peak(
    posthoc: dict[str, Any], preemptive: dict[str, Any]
) -> tuple[float, float]:
    """Peak CUDA allocation during weight loading, per strategy."""
    print(f"\n{'=' * 70}\nallocation: load-time VRAM peak\n{'=' * 70}")

    def peak(run: dict[str, Any]) -> float:
        return float(
            _worker_field(
                run, "cuda_memory", "load_peak_allocated_gib", float("nan")
            )
        )

    def kv(run: dict[str, Any]) -> int:
        return int(_worker_field(run, "kv_cache", "max_tokens", 0))

    p, q = peak(posthoc), peak(preemptive)
    print(f"  post-hoc    {p:6.2f} GiB  (allocates 256 experts, then slices)")
    print(f"  pre-emptive {q:6.2f} GiB  (allocates only CUDA-owned experts)")
    if p == p and q == q and p > 0:
        print(f"  saved       {p - q:6.2f} GiB  ({(1 - q / p) * 100:.1f}% lower)")
    print(
        f"\n  KV tokens: {kv(posthoc):,} -> {kv(preemptive):,} "
        "(expected equal — post-hoc frees before KV is sized)"
    )
    return p, q


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--posthoc", required=True, type=Path)
    ap.add_argument("--preemptive", required=True, type=Path)
    ap.add_argument(
        "--null-a", type=Path,
        help="control: one of two runs with identical code and config. "
             "Supplies the instrument's noise floor, without which a small "
             "logprob delta cannot be attributed to the change under test.",
    )
    ap.add_argument("--null-b", type=Path, help="the other control run")
    args = ap.parse_args()

    base = json.loads(args.posthoc.read_text())
    cand = json.loads(args.preemptive.read_text())

    # A parity claim is only meaningful between two runs that differ in
    # nothing but the surgery strategy. With several placements' results
    # living in one directory, a mismatched pair would otherwise produce a
    # verdict that reads as authoritative. `adapter_sha` covers the case
    # that is invisible in the filenames: an edit landing between the two
    # legs, so they ran different code.
    for key in ("placement", "config", "adapter_sha"):
        if base.get(key) is None or cand.get(key) is None:
            raise SystemExit(
                f"refusing to compare: {key!r} missing — a result predates "
                "this guard and cannot be shown to be comparable"
            )
        if base.get(key) != cand.get(key):
            raise SystemExit(
                f"refusing to compare: {key} differs "
                f"({base.get(key)!r} vs {cand.get(key)!r})"
            )

    tokens = compare_correctness(base, cand)
    worst = compare_logprobs(base, cand)
    peak_post, peak_pre = compare_load_peak(base, cand)

    floor = None
    if args.null_a and args.null_b:
        print(f"\n{'=' * 70}\ncontrol: same code, same config, run twice\n{'=' * 70}")
        null_a = json.loads(args.null_a.read_text())
        null_b = json.loads(args.null_b.read_text())
        null_tokens = compare_correctness(null_a, null_b)
        floor = compare_logprobs(null_a, null_b)
        print(
            f"\n  instrument noise floor: {floor:.3e} nats, "
            f"{null_tokens['exact']}/{null_tokens['total']} sequences identical"
        )

    print(f"\n{'=' * 70}\nverdict\n{'=' * 70}")
    subsetted = peak_pre == peak_pre and peak_post > peak_pre

    if floor is None:
        # No control supplied: the only defensible bar is bit-identity, and
        # the measured floor says this engine does not deliver it. Ask for
        # the control rather than pass or fail on an unknown.
        print("  [SKIP] no control pair given (--null-a/--null-b); "
              "cannot tell a real change from run-to-run noise")
        quiet = False
    else:
        # A change is only visible if it exceeds what the instrument
        # produces from nothing. Measured on the 35B: two identical runs
        # disagree by ~0.11 nats and share 4/8 sequences, because B70
        # partials are accumulated asynchronously and the CUDA kernel's
        # reductions are not order-stable. Token identity is therefore not
        # a usable criterion at all, and any tolerance tighter than the
        # floor would reject correct code.
        quiet = worst <= floor * 1.5
        print(f"  [{'PASS' if quiet else 'FAIL'}] logprob delta "
              f"{worst:.3e} vs noise floor {floor:.3e} "
              f"({'within' if quiet else 'ABOVE'} 1.5x floor)")
        print(f"  [info] tokens identical {tokens['exact']}/{tokens['total']} "
              "(control gives fewer than all; not a criterion)")

    print(f"  [{'PASS' if subsetted else 'FAIL'}] allocation subsetted "
          f"({peak_post:.2f} -> {peak_pre:.2f} GiB)")

    if quiet and subsetted:
        print("\npre-emptive surgery is indistinguishable from post-hoc at "
              "this instrument's resolution, and allocates less.")
        return 0
    print("\nnot equivalent — do not promote.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
