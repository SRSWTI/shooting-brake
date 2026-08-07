#!/usr/bin/env python3
"""Benchmark native prefix caching with a large, stable OpenAI tool schema.

The benchmark is deliberately self-contained: it talks to a running
``dflash_server``, uses ``usage.timings`` for backend-only measurements, and
checks both sides of the cache identity contract:

* later turns with byte-identical tools must restore the system/tool prefix;
* changing one tool must miss instead of restoring incompatible KV state.

Example:

    python3 server/scripts/benchmark_tool_prefix_cache.py \
        --url http://127.0.0.1:8080 --json-out /tmp/tool-prefix.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cold/warm native tool-prefix cache benchmark",
    )
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="dflash")
    parser.add_argument("--warm-turns", type=int, default=3)
    parser.add_argument("--tool-count", type=int, default=24)
    parser.add_argument("--params-per-tool", type=int, default=8)
    parser.add_argument("--description-words", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--min-speedup",
        type=float,
        default=3.0,
        help="Required cold / median-warm prefill speedup (default: 3.0)",
    )
    parser.add_argument(
        "--timing-only",
        action="store_true",
        help=(
            "Accept a pre-PR server without cache telemetry. This measures the "
            "main-branch baseline but cannot validate cache identity."
        ),
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.warm_turns < 1:
        parser.error("--warm-turns must be at least 1")
    for name in ("tool_count", "params_per_tool", "description_words", "max_tokens"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.min_speedup <= 0:
        parser.error("--min-speedup must be positive")
    return args


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


def get_json(url: str, timeout: float) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def make_tools(
    count: int,
    params_per_tool: int,
    description_words: int,
    *,
    mutation: str = "",
) -> list[dict[str, Any]]:
    filler = " ".join(
        f"constraint{index % 11}" for index in range(description_words)
    )
    tools: list[dict[str, Any]] = []
    for tool_index in range(count):
        properties = {}
        required = []
        for param_index in range(params_per_tool):
            name = f"argument_{param_index:02d}"
            properties[name] = {
                "type": "string",
                "description": (
                    f"Input {param_index} for operation {tool_index}; {filler}."
                ),
            }
            required.append(name)
        description = f"Deterministic benchmark operation {tool_index}; {filler}."
        if mutation and tool_index == 0:
            description += f" Mutation marker: {mutation}."
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": f"benchmark_operation_{tool_index:02d}",
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                        "additionalProperties": False,
                    },
                },
            }
        )
    return tools


def make_messages(benchmark_id: str, turn: int) -> list[dict[str, str]]:
    messages = [
        {
            "role": "system",
            "content": (
                "This is a deterministic prefix-cache benchmark. "
                "Do not call tools. Reply with the single word OK. "
                f"Benchmark id: {benchmark_id}."
            ),
        }
    ]
    for prior in range(1, turn):
        messages.append(
            {"role": "user", "content": f"Benchmark turn {prior}. Reply OK."}
        )
        messages.append({"role": "assistant", "content": "OK"})
    messages.append(
        {"role": "user", "content": f"Benchmark turn {turn}. Reply OK."}
    )
    return messages


def extract_result(
    response: dict[str, Any],
    turn: str,
    wall_ms: float,
    *,
    require_cache_telemetry: bool,
) -> dict[str, Any]:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise RuntimeError(f"{turn}: response has no usage object")
    timings = usage.get("timings")
    if not isinstance(timings, dict):
        raise RuntimeError(f"{turn}: response has no usage.timings object")
    required = {
        "prefill_ms",
        "decode_ms",
        "cache_hit",
        "cached_prefix_tokens",
        "prefilled_tokens",
        "effective_prompt_tokens",
    }
    missing = sorted(required.difference(timings))
    if missing and require_cache_telemetry:
        raise RuntimeError(
            f"{turn}: server lacks cache telemetry {missing}; build this PR's "
            "dflash_server before running the benchmark"
        )
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"{turn}: response has no choices")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise RuntimeError(f"{turn}: response choice is not an object")
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError(f"{turn}: response choice has no message object")
    output_text = message.get("content")
    if not isinstance(output_text, str):
        raise RuntimeError(f"{turn}: response message has no text content")
    return {
        "turn": turn,
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "prefill_ms": float(timings["prefill_ms"]),
        "decode_ms": float(timings["decode_ms"]),
        "cache_hit": (
            bool(timings["cache_hit"]) if "cache_hit" in timings else None
        ),
        "cached_prefix_tokens": (
            int(timings["cached_prefix_tokens"])
            if "cached_prefix_tokens" in timings
            else None
        ),
        "prefilled_tokens": (
            int(timings["prefilled_tokens"])
            if "prefilled_tokens" in timings
            else None
        ),
        "effective_prompt_tokens": (
            int(timings["effective_prompt_tokens"])
            if "effective_prompt_tokens" in timings
            else None
        ),
        "wall_ms": round(wall_ms, 1),
        "output_text": output_text.strip(),
        "finish_reason": first_choice.get("finish_reason"),
    }


def run_request(
    args: argparse.Namespace,
    tools: list[dict[str, Any]],
    benchmark_id: str,
    turn: int,
    label: str,
) -> dict[str, Any]:
    payload = {
        "model": args.model,
        "messages": make_messages(benchmark_id, turn),
        "tools": tools,
        "tool_choice": "none",
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    started = time.perf_counter()
    response = post_json(
        f"{args.url.rstrip('/')}/v1/chat/completions",
        payload,
        args.timeout,
    )
    wall_ms = (time.perf_counter() - started) * 1000.0
    return extract_result(
        response,
        label,
        wall_ms,
        require_cache_telemetry=not args.timing_only,
    )


def main() -> int:
    args = parse_args()
    benchmark_id = uuid.uuid4().hex
    tools = make_tools(
        args.tool_count,
        args.params_per_tool,
        args.description_words,
    )
    props_before = get_json(f"{args.url.rstrip('/')}/props", min(args.timeout, 10.0))

    rows = [run_request(args, tools, benchmark_id, 1, "cold")]
    for turn in range(2, args.warm_turns + 2):
        rows.append(run_request(args, tools, benchmark_id, turn, f"warm-{turn - 1}"))

    mutated_tools = make_tools(
        args.tool_count,
        args.params_per_tool,
        args.description_words,
        mutation=uuid.uuid4().hex,
    )
    mutation = run_request(
        args,
        mutated_tools,
        benchmark_id,
        args.warm_turns + 2,
        "changed-tools-control",
    )
    rows.append(mutation)
    props_after = get_json(f"{args.url.rstrip('/')}/props", min(args.timeout, 10.0))

    cold = rows[0]
    warm = rows[1:-1]
    measured_prefill_ms = [row["prefill_ms"] for row in [cold, *warm]]
    if any(value <= 0 for value in measured_prefill_ms):
        raise RuntimeError(
            "cold and warm prefill timings must all be positive; got "
            f"{measured_prefill_ms}"
        )
    median_warm_ms = statistics.median(row["prefill_ms"] for row in warm)
    speedup = cold["prefill_ms"] / median_warm_ms

    checks = {"speedup_meets_threshold": speedup >= args.min_speedup}
    if not args.timing_only:
        checks = {
            "all_outputs_nonempty": all(row["output_text"] for row in rows),
            "outputs_are_stable": all(
                row["output_text"] == cold["output_text"] for row in warm
            ),
            "cold_is_miss": not cold["cache_hit"],
            "all_warm_hit": all(row["cache_hit"] for row in warm),
            "warm_cached_tokens_positive": all(
                row["cached_prefix_tokens"] > 0 for row in warm
            ),
            "warm_prefills_less_than_cold": all(
                row["prefilled_tokens"] < cold["prefilled_tokens"] for row in warm
            ),
            "changed_tools_is_miss": not mutation["cache_hit"],
            "token_accounting_is_exact": all(
                row["cached_prefix_tokens"] + row["prefilled_tokens"]
                == row["effective_prompt_tokens"]
                for row in rows
            ),
            **checks,
        }
    passed = all(checks.values())
    result = {
        "schema": 1,
        "benchmark": "native-tool-prefix-cache",
        "benchmark_id": benchmark_id,
        "config": {
            "url": args.url,
            "model": args.model,
            "warm_turns": args.warm_turns,
            "tool_count": args.tool_count,
            "params_per_tool": args.params_per_tool,
            "description_words": args.description_words,
            "max_tokens": args.max_tokens,
            "min_speedup": args.min_speedup,
            "timing_only": args.timing_only,
        },
        "server_before": props_before,
        "server_after": props_after,
        "requests": rows,
        "cold_prefill_ms": cold["prefill_ms"],
        "median_warm_prefill_ms": median_warm_ms,
        "prefill_speedup": round(speedup, 2),
        "checks": checks,
        "passed": passed,
    }

    print(
        "turn                     raw_prompt  effective  cached  prefilled  "
        "prefill_ms  cache_hit"
    )
    for row in rows:
        cached = (
            str(row["cached_prefix_tokens"])
            if row["cached_prefix_tokens"] is not None
            else "-"
        )
        prefilled = (
            str(row["prefilled_tokens"])
            if row["prefilled_tokens"] is not None
            else "-"
        )
        effective = (
            str(row["effective_prompt_tokens"])
            if row["effective_prompt_tokens"] is not None
            else "-"
        )
        cache_hit = (
            str(row["cache_hit"]).lower()
            if row["cache_hit"] is not None
            else "-"
        )
        print(
            f"{row['turn']:<24} {row['prompt_tokens']:>10} "
            f"{effective:>10} {cached:>7} {prefilled:>10} "
            f"{row['prefill_ms']:>10.1f}  {cache_hit}"
        )
    print(
        f"\ncold / median-warm prefill: {speedup:.2f}x "
        f"(required {args.min_speedup:.2f}x)"
    )
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n{'PASS' if passed else 'FAIL'}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, urllib.error.URLError, TimeoutError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
