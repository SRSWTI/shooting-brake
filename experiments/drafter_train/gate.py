#!/usr/bin/env python3
"""Live HTTP instruments for drafter quality and latency acceptance."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

QUALITY_COUNT = 120
QUALITY_TOP_LOGPROBS = 5
QUALITY_MAX_JSD = 0.10
QUALITY_PASS_PCT = 90.0
LADDER_LABEL = "jota-r15-dflash"
# MEASURED 2026-08-26: the drafter is a third resident on the 5090 and
# 131072-token context needs 9.64 GiB of KV (vLLM's own figure) on top of
# the 16.77 GiB model + 1.96 GiB drafter + workspace -- it does not fit in
# 31.36 GiB. The candidate boots at max_model_len 98304, so BOTH arms run
# the same six rungs and the comparison stays apples-to-apples. Restoring
# the 127000 rung requires freeing 5090 VRAM (e.g. cuda_fraction 0.22 ->
# 0.10 moves ~26 experts to the B70s, ~5.7 GB) -- its own gated change.
LADDER_CONTEXTS = (1024, 8192, 16384, 32768, 65536, 98304)

import requests



def model_id(base_url: str) -> str:
    response = requests.get(f"{base_url}/v1/models", timeout=30)
    response.raise_for_status()
    return response.json()["data"][0]["id"]


def post_json(url: str, payload: dict[str, Any], timeout: float = 600) -> dict:
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def quality_prompts(corpus: Path, count: int = QUALITY_COUNT) -> list[str]:
    words = corpus.expanduser().read_text(errors="ignore").split()
    width = 230
    if len(words) < width + count:
        raise ValueError(f"quality corpus {corpus} is too short")
    rng = random.Random(20260826)
    starts = rng.sample(range(0, len(words) - width), count)
    return [" ".join(words[start : start + width]) for start in starts]


def capture(base_url: str, corpus: Path, output: Path) -> dict:
    model = model_id(base_url)
    quality = []
    for index, prompt in enumerate(quality_prompts(corpus)):
        result = post_json(
            f"{base_url}/v1/completions",
            {
                "model": model,
                "prompt": prompt,
                "temperature": 0,
                "max_tokens": 1,
                "logprobs": QUALITY_TOP_LOGPROBS,
            },
        )
        choice = result["choices"][0]
        top_logprobs = (choice.get("logprobs") or {}).get("top_logprobs") or []
        if len(top_logprobs) != 1 or not isinstance(top_logprobs[0], dict):
            raise ValueError(
                f"quality prompt {index}: server did not return one top-logprob distribution"
            )
        distribution = {}
        for token, value in top_logprobs[0].items():
            logprob = float(value)
            if not math.isfinite(logprob):
                raise ValueError(f"quality prompt {index}: non-finite logprob")
            distribution[str(token)] = logprob
        if not distribution:
            raise ValueError(f"quality prompt {index}: empty top-logprob distribution")
        quality.append({"index": index, "top_logprobs": distribution})

    payload = {
        "schema": "drafter_quality_sweep_v1",
        "model": model,
        "top_logprobs": QUALITY_TOP_LOGPROBS,
        "quality": quality,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"captured {len(quality)} top-logprob quality prompts -> {output}")
    return payload


def _probabilities(logprobs: dict[str, Any]) -> tuple[dict[str, float], float]:
    probabilities = {str(token): math.exp(float(value)) for token, value in logprobs.items()}
    total = sum(probabilities.values())
    tail = max(0.0, 1.0 - total)
    normalizer = total + tail
    if not math.isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError("quality distribution has no finite probability mass")
    return (
        {token: probability / normalizer for token, probability in probabilities.items()},
        tail / normalizer,
    )


def _js_divergence(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_probabilities, left_tail = _probabilities(left)
    right_probabilities, right_tail = _probabilities(right)
    divergence = 0.0
    for token in left_probabilities.keys() | right_probabilities.keys():
        p = left_probabilities.get(token, 0.0)
        q = right_probabilities.get(token, 0.0)
        midpoint = 0.5 * (p + q)
        if p:
            divergence += 0.5 * p * math.log(p / midpoint)
        if q:
            divergence += 0.5 * q * math.log(q / midpoint)
    tail_midpoint = 0.5 * (left_tail + right_tail)
    if left_tail:
        divergence += 0.5 * left_tail * math.log(left_tail / tail_midpoint)
    if right_tail:
        divergence += 0.5 * right_tail * math.log(right_tail / tail_midpoint)
    return divergence


def compare(baseline_path: Path, candidate_path: Path) -> dict:
    baseline = json.loads(baseline_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    for name, payload in (("baseline", baseline), ("candidate", candidate)):
        if payload.get("schema") != "drafter_quality_sweep_v1":
            raise ValueError(f"{name} quality evidence has an unsupported schema")
        if payload.get("top_logprobs") != QUALITY_TOP_LOGPROBS:
            raise ValueError(f"{name} quality evidence has the wrong top-logprob depth")
        if len(payload.get("quality") or []) != QUALITY_COUNT:
            raise ValueError(f"{name} quality evidence must contain {QUALITY_COUNT} prompts")

    divergences = []
    for expected_index, (left, right) in enumerate(
        zip(baseline["quality"], candidate["quality"], strict=True)
    ):
        if left.get("index") != expected_index or right.get("index") != expected_index:
            raise ValueError("quality evidence prompt indices are not aligned")
        divergences.append(
            _js_divergence(left["top_logprobs"], right["top_logprobs"])
        )
    ordered = sorted(divergences)
    passing = sum(value <= QUALITY_MAX_JSD for value in divergences)
    return {
        "metric": "top_logprob_jensen_shannon_divergence_nats",
        "quality_passing": passing,
        "quality_total": len(divergences),
        "quality_pct": 100.0 * passing / len(divergences),
        "max_jsd": max(divergences),
        "mean_jsd": sum(divergences) / len(divergences),
        "p95_jsd": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "per_prompt_jsd": divergences,
    }


def _ladder_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"{path}: ladder JSON must be a list")
    rows = [row for row in payload if isinstance(row, dict) and row.get("label") == LADDER_LABEL]
    if len(rows) != len(LADDER_CONTEXTS):
        raise ValueError(
            f"{path}: expected {len(LADDER_CONTEXTS)} {LADDER_LABEL!r} rows, got {len(rows)}"
        )
    contexts = []
    required = {
        "label",
        "ctx",
        "prompt_tokens",
        "out_tokens",
        "ttft_s",
        "tpot_ms",
        "chunk_gap_p50_ms",
        "decode_tok_s",
        "wall_s",
        "drafted",
        "accepted",
        "acceptance_pct",
    }
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"{path}: ladder row {index} is missing {sorted(missing)}")
        if type(row["ctx"]) is not int:
            raise ValueError(f"{path}: ladder row {index} ctx must be an integer")
        contexts.append(row["ctx"])
        for key in ("prompt_tokens", "out_tokens", "drafted", "accepted"):
            if type(row[key]) is not int or row[key] < 0:
                raise ValueError(f"{path}: ladder row {index} {key} must be a non-negative integer")
        if row["prompt_tokens"] == 0 or row["out_tokens"] < 2:
            raise ValueError(f"{path}: ladder row {index} has no measurable decode span")
        for key in ("ttft_s", "tpot_ms", "chunk_gap_p50_ms", "decode_tok_s", "wall_s"):
            if (
                not isinstance(row[key], (int, float))
                or not math.isfinite(float(row[key]))
                or float(row[key]) <= 0.0
            ):
                raise ValueError(f"{path}: ladder row {index} {key} must be finite and positive")
        if row["accepted"] > row["drafted"]:
            raise ValueError(f"{path}: ladder row {index} accepted exceeds drafted")
        if row["drafted"]:
            acceptance_pct = row["acceptance_pct"]
            expected = round(100.0 * row["accepted"] / row["drafted"], 1)
            if (
                not isinstance(acceptance_pct, (int, float))
                or not math.isfinite(float(acceptance_pct))
                or abs(float(acceptance_pct) - expected) > 0.05
            ):
                raise ValueError(f"{path}: ladder row {index} acceptance_pct disagrees with counters")
        elif row["acceptance_pct"] is not None:
            raise ValueError(f"{path}: ladder row {index} zero drafted requires null acceptance_pct")
    if tuple(contexts) != LADDER_CONTEXTS:
        raise ValueError(f"{path}: ladder contexts are {contexts}, expected {list(LADDER_CONTEXTS)}")
    return rows


def summarize(baseline: Path, candidate: Path, ladder: Path, output: Path) -> int:
    quality = compare(baseline, candidate)
    rows = _ladder_rows(ladder)
    drafted = sum(row["drafted"] for row in rows)
    accepted = sum(row["accepted"] for row in rows)
    acceptance = 100.0 * accepted / drafted if drafted else 0.0
    worst_tpot = max(float(row["tpot_ms"]) for row in rows)
    every_rung_accepts = all(
        row["drafted"] > 0 and 100.0 * row["accepted"] / row["drafted"] >= 60.0
        for row in rows
    )
    every_rung_is_fast = all(float(row["tpot_ms"]) <= 6.0 for row in rows)
    gates = {
        "quality_sweep_ge_90pct_with_jsd_le_0.10": quality["quality_pct"]
        >= QUALITY_PASS_PCT,
        "aggregate_acceptance_ge_60pct": acceptance >= 60.0,
        "every_ladder_rung_acceptance_ge_60pct": every_rung_accepts,
        "every_ladder_rung_effective_decode_le_6ms": every_rung_is_fast,
    }
    passed = all(gates.values())
    summary = {
        "schema": "drafter_acceptance_summary_v1",
        "passed": passed,
        "gates": gates,
        "quality": quality,
        "aggregate_acceptance_pct": round(acceptance, 2),
        "worst_effective_decode_ms_per_token": round(worst_tpot, 3),
        "drafted": drafted,
        "accepted": accepted,
        "ladder": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print("=" * 72)
    print("DRAFTER ACCEPTANCE: " + ("PASS" if passed else "FAIL"))
    print(
        f"quality={quality['quality_passing']}/{quality['quality_total']} "
        f"({quality['quality_pct']:.1f}%, floor {QUALITY_PASS_PCT:.1f}% at "
        f"JSD <= {QUALITY_MAX_JSD:.2f} nats); p95 JSD={quality['p95_jsd']:.4f}"
    )
    print(
        f"acceptance={acceptance:.2f}% ({accepted}/{drafted}, target >=60%); "
        f"worst effective decode={worst_tpot:.3f} ms/tok (target <=6.0)"
    )
    for name, ok in gates.items():
        print(f"  {'PASS' if ok else 'FAIL'} {name}")
    print(f"evidence: {output}")
    print("=" * 72)
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    capture_parser = sub.add_parser("capture")
    capture_parser.add_argument("--url", default="http://127.0.0.1:8017")
    capture_parser.add_argument("--corpus", type=Path, default=Path.home() / "sb_corpus_big.txt")
    capture_parser.add_argument("--output", type=Path, required=True)
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--baseline", type=Path, required=True)
    summary_parser.add_argument("--candidate", type=Path, required=True)
    summary_parser.add_argument("--ladder", type=Path, required=True)
    summary_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "capture":
        capture(args.url, args.corpus, args.output)
        return 0
    return summarize(args.baseline, args.candidate, args.ladder, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
