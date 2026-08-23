#!/usr/bin/env python3

from __future__ import annotations

import argparse
import zlib
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests
from prometheus_client.parser import text_string_to_metric_families


@dataclass
class RunConfig:
    target: str
    metrics_url: str
    model: str
    tokenizer_model: str | None
    output_root: Path
    contexts: list[int]
    concurrent_rates: list[int]
    sweep_steps: int
    output_tokens: int
    max_seconds: float | None
    max_requests: int | None
    sample_interval: float
    outputs: list[str]
    request_format: str
    random_seed: int
    rampup: float
    warmup: str | None
    cooldown: str | None
    max_errors: int
    disable_console: bool
    skip_existing: bool
    profiles: list[str]


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Run a GuideLLM benchmark matrix against a live vLLM server while "
            "scraping the server's Prometheus /metrics endpoint in parallel."
        )
    )
    parser.add_argument("--target", default="http://localhost:8080")
    parser.add_argument("--metrics-url", default="http://localhost:8080/metrics")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--tokenizer-model",
        help="Tokenizer repo/path when the served model name is only an API alias.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/root/testing/bench-matrix/prod_gpu"),
    )
    parser.add_argument(
        "--contexts",
        default="1024,4096,8192,16384,32768",
        help="Comma-separated prompt token lengths.",
    )
    parser.add_argument(
        "--concurrent-rates",
        default="1,2,4,8,16",
        help="Comma-separated concurrent profile rates.",
    )
    parser.add_argument(
        "--sweep-steps",
        type=int,
        default=6,
        help="GuideLLM sweep step count, including synchronous and throughput anchors.",
    )
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--sample-interval", type=float, default=5.0)
    parser.add_argument(
        "--outputs",
        default="json,csv,html",
        help="Comma-separated GuideLLM output formats.",
    )
    parser.add_argument("--request-format", default="/v1/chat/completions")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--rampup", type=float, default=10.0)
    parser.add_argument("--warmup", default="0.1")
    parser.add_argument("--cooldown", default="0.1")
    parser.add_argument("--max-errors", type=int, default=5)
    parser.add_argument(
        "--disable-console",
        action="store_true",
        default=True,
        help="Disable GuideLLM console output. Enabled by default for cleaner logs.",
    )
    parser.add_argument(
        "--enable-console",
        dest="disable_console",
        action="store_false",
        help="Enable GuideLLM console output.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Skip cells whose run_manifest.json already reports return_code=0.",
    )
    parser.add_argument(
        "--profiles",
        default="synchronous,concurrent,sweep",
        help="Comma-separated list of profiles to run (synchronous, concurrent, sweep).",
    )

    args = parser.parse_args()
    return RunConfig(
        target=args.target,
        metrics_url=args.metrics_url,
        model=args.model,
        tokenizer_model=args.tokenizer_model,
        output_root=args.output_root,
        contexts=_parse_int_list(args.contexts),
        concurrent_rates=_parse_int_list(args.concurrent_rates),
        sweep_steps=args.sweep_steps,
        output_tokens=args.output_tokens,
        max_seconds=args.max_seconds,
        max_requests=args.max_requests,
        sample_interval=args.sample_interval,
        outputs=_parse_str_list(args.outputs),
        request_format=args.request_format,
        random_seed=args.random_seed,
        rampup=args.rampup,
        warmup=args.warmup,
        cooldown=args.cooldown,
        max_errors=args.max_errors,
        disable_console=args.disable_console,
        skip_existing=args.skip_existing,
        profiles=_parse_str_list(args.profiles),
    )


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _slugify_model_name(model: str) -> str:
    return model.replace("/", "__")


def _metric_key(name: str, labels: dict[str, Any]) -> str:
    if not labels:
        return name
    label_str = ",".join(f"{key}={labels[key]}" for key in sorted(labels))
    return f"{name}{{{label_str}}}"


def _parse_prometheus_metrics(payload: str) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for family in text_string_to_metric_families(payload):
        for sample in family.samples:
            key = _metric_key(sample.name, sample.labels)
            try:
                parsed[key] = float(sample.value)
            except (TypeError, ValueError):
                continue
    return parsed


def _scrape_metrics(stop_event: threading.Event, metrics_url: str, output_path: Path, interval: float) -> None:
    session = requests.Session()
    with output_path.open("a", encoding="utf-8") as handle:
        while not stop_event.is_set():
            started = time.time()
            record: dict[str, Any] = {"timestamp": started}
            try:
                response = session.get(metrics_url, timeout=10)
                response.raise_for_status()
                record["status"] = response.status_code
                record["metrics"] = _parse_prometheus_metrics(response.text)
            except Exception as exc:  # noqa: BLE001
                record["error"] = str(exc)

            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()

            elapsed = time.time() - started
            remaining = interval - elapsed
            if remaining > 0:
                stop_event.wait(remaining)


def _guidellm_repo_root(script_path: Path) -> Path:
    return script_path.resolve().parents[1]


def _cell_seed(config: "RunConfig", profile: str, context_len: int) -> int:
    """Distinct synthetic prompts per cell, deterministically.

    zlib.crc32 rather than hash(): PYTHONHASHSEED randomises str hashing per
    process, which would make a --skip-existing resume generate different
    prompts for a cell than the original run did.
    """
    key = f"{config.random_seed}|{profile}|{context_len}|{config.output_tokens}"
    return zlib.crc32(key.encode()) & 0x7FFF_FFFF


def _guidellm_command(
    repo_root: Path,
    config: RunConfig,
    profile: str,
    context_len: int,
    output_dir: Path,
) -> list[str]:
    # guidellm's CLI moved to config-based ``kind=...`` options. Every
    # section (backend / profile / data / constraint / output) takes a
    # ``kind`` plus inline key=value fields; per-section tuning that the
    # CLI can't reach goes through ``--override``.
    trim_transients = config.max_requests is None or config.max_requests >= 12
    warmup = config.warmup if trim_transients else "0"
    cooldown = config.cooldown if trim_transients else "0"
    profile_spec = (
        f"kind={profile},warmup={warmup},cooldown={cooldown},"
        f"rampup_duration={config.rampup}"
    )
    command = [
        sys.executable, "-m", "guidellm", "run",
        "--backend",
        (
            "kind=openai_http,"
            f"target={config.target},"
            f"model={config.model},"
            f"request_format={config.request_format}"
        ),
    ]
    if config.tokenizer_model is not None:
        command += [
            "--tokenizer",
            (
                "kind=huggingface_auto,"
                f"model={config.tokenizer_model},"
                "load_kwargs.trust_remote_code=true"
            ),
        ]
    command += [
        "--profile", profile_spec,
        "--data",
        (
            "kind=synthetic_text,"
            f"prompt_tokens={context_len},"
            f"output_tokens={config.output_tokens}"
        ),
        # Per-CELL seed, not per-run. GuideLLM restarts its synthetic generator
        # in every subprocess, so a constant seed makes cell N's sample K
        # byte-identical to cell M's sample K -- and with prefix caching on, every
        # cell after the first measures a cache HIT instead of prefill. Measured
        # 2026-08-22: ctx_1024/C=1 read TTFT min 63 / median 89 / max 126 ms on a
        # 1,066-token prompt whose cold cost is ~430 ms. The whole 24-cell grid's
        # TTFT column was warm-path. bench_88b.py:cell_seed already guarded this;
        # matrix_runner did not.
        "--seed", f"kind=static,value={_cell_seed(config, profile, context_len)}",
    ]

    # Per-profile load shape. concurrent runs each rate as a fixed
    # concurrency; sweep interpolates across strategies.
    if profile == "concurrent":
        command += [
            "--override", "profile.streams",
            ",".join(str(r) for r in config.concurrent_rates),
        ]
    elif profile == "sweep":
        command += ["--override", "profile.sweep_size", str(config.sweep_steps)]

    # Stop conditions. max_seconds is scaled by context length so large
    # cells get enough wall time for valid statistics.
    if config.max_seconds is not None:
        if context_len > 128_000:
            scaled = config.max_seconds * 8
        elif context_len > 32_000:
            scaled = config.max_seconds * 4
        elif context_len > 8_000:
            scaled = config.max_seconds * 2
        else:
            scaled = config.max_seconds
        command += ["--constraint", f"kind=max_duration,seconds={int(scaled)}"]
    if config.max_requests is not None and (
        profile != "sweep" or config.max_requests >= 10
    ):
        command += ["--constraint", f"kind=max_requests,count={config.max_requests}"]
    command += ["--constraint", f"kind=max_errors,count={config.max_errors}"]

    # One --output per requested format; each needs a full file path.
    for fmt in config.outputs:
        command += ["--output", f"kind={fmt},path={output_dir / f'report.{fmt}'}"]

    if config.disable_console:
        command.append("--disable-console")
    return command


def _run_profile(repo_root: Path, config: RunConfig, profile: str, context_len: int, base_dir: Path) -> None:
    profile_dir = base_dir / profile
    profile_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = profile_dir / "vllm_metrics.jsonl"
    command_path = profile_dir / "guidellm_command.json"
    stdout_path = profile_dir / "guidellm_stdout.log"
    stderr_path = profile_dir / "guidellm_stderr.log"

    command = _guidellm_command(repo_root, config, profile, context_len, profile_dir)
    command_path.write_text(json.dumps({"command": command}, indent=2), encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")

    stop_event = threading.Event()
    scraper = threading.Thread(
        target=_scrape_metrics,
        args=(stop_event, config.metrics_url, metrics_path, config.sample_interval),
        daemon=True,
    )
    scraper.start()

    started = time.time()
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_handle:
            result = subprocess.run(
                command,
                cwd=repo_root,
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                check=False,
            )
    finally:
        stop_event.set()
        scraper.join(timeout=max(config.sample_interval * 2, 5.0))

    manifest = {
        "profile": profile,
        "context_tokens": context_len,
        "started_at": started,
        "finished_at": time.time(),
        "return_code": result.returncode,
        "metrics_path": str(metrics_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    (profile_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"GuideLLM profile '{profile}' failed for context {context_len}. "
            f"See {stderr_path}."
        )


def _preflight_metrics(metrics_url: str) -> None:
    response = requests.get(metrics_url, timeout=10)
    response.raise_for_status()
    parsed = _parse_prometheus_metrics(response.text)
    if not any(name.startswith("vllm:") for name in parsed):
        raise RuntimeError(f"No vLLM metrics found at {metrics_url}")


def main() -> int:
    config = parse_args()
    repo_root = _guidellm_repo_root(Path(__file__))

    _preflight_metrics(config.metrics_url)

    model_dir = config.output_root / _slugify_model_name(config.model)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "matrix_config.json").write_text(
        json.dumps(asdict(config), indent=2, default=str, sort_keys=True),
        encoding="utf-8",
    )

    for context_len in config.contexts:
        context_dir = model_dir / f"ctx_{context_len}"
        context_dir.mkdir(parents=True, exist_ok=True)
        for profile in config.profiles:
            if config.skip_existing:
                manifest_path = context_dir / profile / "run_manifest.json"
                if manifest_path.exists():
                    try:
                        cached = json.loads(manifest_path.read_text(encoding="utf-8"))
                        if cached.get("return_code") == 0:
                            print(
                                f"[run_vllm_matrix] skip (already done) context={context_len} profile={profile}",
                                flush=True,
                            )
                            continue
                    except Exception:  # noqa: BLE001
                        pass
            print(f"[run_vllm_matrix] context={context_len} profile={profile}", flush=True)
            _run_profile(repo_root, config, profile, context_len, context_dir)

    print(f"[run_vllm_matrix] completed output_root={model_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())