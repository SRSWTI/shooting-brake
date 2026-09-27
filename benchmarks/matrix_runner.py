#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
import zlib
from dataclasses import asdict, dataclass
from importlib import metadata
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
    server_manifest: Path | None = None
    seed_mode: str = "per-cell"
    duration_mode: str = "scaled"


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
        "--server-manifest",
        type=Path,
        help="JSON-object server identity/config supplied by the operator; no server properties are inferred.",
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
    parser.add_argument(
        "--duration-mode",
        choices=("scaled", "fixed"),
        default="scaled",
        help="Scale duration by context, or use max-seconds unchanged in every cell.",
    )
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--sample-interval", type=float, default=5.0)
    parser.add_argument(
        "--outputs",
        default="json,csv,html",
        help="Comma-separated GuideLLM output formats.",
    )
    parser.add_argument("--request-format", default="/v1/chat/completions")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--seed-mode",
        choices=("per-cell", "fixed"),
        default="per-cell",
        help="Derive seeds per cell, or reuse random-seed. Disable server prefix caching for cold fixed-seed runs.",
    )
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
        help="Skip successful cells only when their stored run identity matches this invocation.",
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
        server_manifest=args.server_manifest,
        seed_mode=args.seed_mode,
        duration_mode=args.duration_mode,
    )


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _slugify_model_name(model: str) -> str:
    return model.replace("/", "__")


def _identity_json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError("expected a JSON object")
        # Reject non-JSON NaN/Infinity values, including overflowed numbers.
        _identity_json(value)
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Cannot read JSON object from {path}: {exc}") from exc
    return value


def _load_server_manifest(path: Path | None) -> dict[str, Any] | None:
    return None if path is None else _read_json_object(path)


def _client_identity() -> dict[str, Any]:
    # These are local client distributions, NOT remote serving-stack versions.
    packages = sorted(
        (
            {"name": dist.metadata["Name"], "version": dist.version}
            for dist in metadata.distributions()
        ),
        key=lambda item: (item["name"] or "", item["version"]),
    )
    return {
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "installed_packages": packages,
    }


def _run_identity(config: RunConfig, runner_path: Path) -> dict[str, Any]:
    server_manifest = _load_server_manifest(config.server_manifest)
    benchmark_config = asdict(config)
    for key in ("output_root", "skip_existing", "server_manifest"):
        del benchmark_config[key]
    return {
        "schema_version": 1,
        "benchmark_config": benchmark_config,
        "client": _client_identity(),
        "runner_source_sha256": hashlib.sha256(runner_path.read_bytes()).hexdigest(),
        "server": {
            "source": "user_supplied" if server_manifest is not None else "not_provided",
            "manifest": server_manifest,
        },
    }


def _resume_error(path: Path, reason: str) -> RuntimeError:
    return RuntimeError(
        f"Refusing to reuse {path}: {reason}. "
        "Use a new output directory with --output-root; existing evidence was not overwritten."
    )


def _resume_cells(config: RunConfig, model_dir: Path, identity_sha256: str) -> set[tuple[int, str]]:
    completed: set[tuple[int, str]] = set()
    for context_len in config.contexts:
        for profile in config.profiles:
            profile_dir = model_dir / f"ctx_{context_len}" / profile
            if not profile_dir.exists():
                continue
            if not profile_dir.is_dir():
                raise _resume_error(profile_dir, "cell path is not a directory")
            manifest_path = profile_dir / "run_manifest.json"
            if not manifest_path.exists():
                if any(profile_dir.iterdir()):
                    raise _resume_error(profile_dir, "existing cell artifacts have no provenance")
                continue
            try:
                manifest = _read_json_object(manifest_path)
            except RuntimeError as exc:
                raise _resume_error(manifest_path, str(exc)) from exc
            if (
                manifest.get("run_identity_sha256") != identity_sha256
                or manifest.get("profile") != profile
                or manifest.get("context_tokens") != context_len
                or "return_code" not in manifest
                or (
                    manifest["return_code"] is not None and type(manifest["return_code"]) is not int
                )
            ):
                raise _resume_error(
                    manifest_path, "cell identity is missing, incompatible, or malformed"
                )
            if manifest["return_code"] == 0:
                completed.add((context_len, profile))
    return completed


def _prepare_run(config: RunConfig, runner_path: Path) -> tuple[Path, str, set[tuple[int, str]]]:
    # Validate every cell before any network activity or modification of evidence.
    identity = _run_identity(config, runner_path)
    identity_json = _identity_json(identity)
    identity_sha256 = hashlib.sha256(identity_json.encode("utf-8")).hexdigest()
    model_dir = config.output_root / _slugify_model_name(config.model)
    identity_path = model_dir / "run_identity.json"
    identity_exists = identity_path.exists()
    if model_dir.exists() and not model_dir.is_dir():
        raise _resume_error(model_dir, "model path is not a directory")
    if identity_exists:
        try:
            stored = _read_json_object(identity_path)
        except RuntimeError as exc:
            raise _resume_error(identity_path, str(exc)) from exc
        if _identity_json(stored) != identity_json:
            raise _resume_error(
                identity_path, "workload, client, runner, or supplied server identity changed"
            )
    elif model_dir.exists() and any(model_dir.iterdir()):
        raise _resume_error(model_dir, "existing run has no run_identity.json provenance")
    completed = _resume_cells(config, model_dir, identity_sha256)

    model_dir.mkdir(parents=True, exist_ok=True)
    if not identity_exists:
        # Exclusive creation: an identity is never replaced by a resumed invocation.
        try:
            with identity_path.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(identity, indent=2, sort_keys=True, allow_nan=False))
        except FileExistsError as exc:
            raise _resume_error(
                identity_path, "another invocation created the run identity"
            ) from exc
    config_path = model_dir / "matrix_config.json"
    if not config_path.exists():
        with config_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(config), indent=2, default=str, sort_keys=True))
    return model_dir, identity_sha256, completed


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


def _scrape_metrics(
    stop_event: threading.Event, metrics_url: str, output_path: Path, interval: float
) -> None:
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


def _cell_seed(config: RunConfig, profile: str, context_len: int) -> int:
    """Choose deterministic per-cell prompts or an explicitly fixed seed.

    zlib.crc32 rather than hash(): PYTHONHASHSEED randomises str hashing per
    process, which would make a --skip-existing resume generate different
    prompts for a cell than the original run did.
    """
    if config.seed_mode == "fixed":
        return config.random_seed
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
        f"kind={profile},warmup={warmup},cooldown={cooldown},rampup_duration={config.rampup}"
    )
    command = [
        sys.executable,
        "-m",
        "guidellm",
        "run",
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
        "--profile",
        profile_spec,
        "--data",
        (f"kind=synthetic_text,prompt_tokens={context_len},output_tokens={config.output_tokens}"),
        # By default, derive a per-cell seed. GuideLLM restarts its synthetic
        # generator in every subprocess, so a fixed seed can reuse identical
        # prompts across cells. Fixed mode requires explicit cache isolation
        # for cold comparisons; with prefix caching on, otherwise every
        # cell after the first measures a cache HIT instead of prefill. Measured
        # 2026-08-22: ctx_1024/C=1 read TTFT min 63 / median 89 / max 126 ms on a
        # 1,066-token prompt whose cold cost is ~430 ms. The whole 24-cell grid's
        # TTFT column was warm-path. bench_88b.py:cell_seed already guarded this;
        # matrix_runner did not.
        "--seed",
        f"kind=static,value={_cell_seed(config, profile, context_len)}",
    ]

    # Per-profile load shape. concurrent runs each rate as a fixed
    # concurrency; sweep interpolates across strategies.
    if profile == "concurrent":
        command += [
            "--override",
            "profile.streams",
            ",".join(str(r) for r in config.concurrent_rates),
        ]
    elif profile == "sweep":
        command += ["--override", "profile.sweep_size", str(config.sweep_steps)]

    # Scale durations by default so long-context cells collect enough samples;
    # fixed mode reproduces a common measurement window across all cells.
    if config.max_seconds is not None:
        if config.duration_mode == "fixed":
            scaled = config.max_seconds
        elif context_len > 128_000:
            scaled = config.max_seconds * 8
        elif context_len > 32_000:
            scaled = config.max_seconds * 4
        elif context_len > 8_000:
            scaled = config.max_seconds * 2
        else:
            scaled = config.max_seconds
        command += ["--constraint", f"kind=max_duration,seconds={int(scaled)}"]
    if config.max_requests is not None and (profile != "sweep" or config.max_requests >= 10):
        command += ["--constraint", f"kind=max_requests,count={config.max_requests}"]
    command += ["--constraint", f"kind=max_errors,count={config.max_errors}"]

    # One --output per requested format; each needs a full file path.
    for fmt in config.outputs:
        command += ["--output", f"kind={fmt},path={output_dir / f'report.{fmt}'}"]

    if config.disable_console:
        command.append("--disable-console")
    return command


def _run_profile(
    repo_root: Path,
    config: RunConfig,
    profile: str,
    context_len: int,
    base_dir: Path,
    identity_sha256: str,
) -> None:
    profile_dir = base_dir / profile
    profile_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = profile_dir / "vllm_metrics.jsonl"
    command_path = profile_dir / "guidellm_command.json"
    stdout_path = profile_dir / "guidellm_stdout.log"
    stderr_path = profile_dir / "guidellm_stderr.log"

    command = _guidellm_command(repo_root, config, profile, context_len, profile_dir)
    started = time.time()
    manifest = {
        "profile": profile,
        "context_tokens": context_len,
        "run_identity_sha256": identity_sha256,
        "started_at": started,
        "finished_at": None,
        "return_code": None,
        "metrics_path": str(metrics_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    # Establish provenance before starting a cell so interrupted attempts can resume.
    (profile_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
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

    try:
        with (
            stdout_path.open("w", encoding="utf-8") as stdout_handle,
            stderr_path.open("w", encoding="utf-8") as stderr_handle,
        ):
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

    manifest.update(finished_at=time.time(), return_code=result.returncode)
    (profile_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"GuideLLM profile '{profile}' failed for context {context_len}. See {stderr_path}."
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

    # Identity validation and the supplied server snapshot must precede HTTP requests.

    model_dir, identity_sha256, completed = _prepare_run(config, Path(__file__))
    _preflight_metrics(config.metrics_url)

    for context_len in config.contexts:
        context_dir = model_dir / f"ctx_{context_len}"
        context_dir.mkdir(parents=True, exist_ok=True)
        for profile in config.profiles:
            if config.skip_existing and (context_len, profile) in completed:
                print(
                    f"[run_vllm_matrix] skip (already done) context={context_len} profile={profile}",
                    flush=True,
                )
                continue
            print(f"[run_vllm_matrix] context={context_len} profile={profile}", flush=True)
            _run_profile(repo_root, config, profile, context_len, context_dir, identity_sha256)

    print(f"[run_vllm_matrix] completed output_root={model_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
