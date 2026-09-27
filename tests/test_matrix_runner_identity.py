from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

_RUNNER = Path(__file__).parents[1] / "benchmarks" / "matrix_runner.py"
_SPEC = importlib.util.spec_from_file_location("matrix_runner_identity", _RUNNER)
assert _SPEC is not None and _SPEC.loader is not None
matrix_runner = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = matrix_runner
_SPEC.loader.exec_module(matrix_runner)


def _config(tmp_path, **changes):
    config = matrix_runner.RunConfig(
        target="http://127.0.0.1:1",
        metrics_url="http://127.0.0.1:1/metrics",
        model="test/model",
        tokenizer_model="test/tokenizer",
        output_root=tmp_path / "results",
        contexts=[128, 256],
        concurrent_rates=[1, 3],
        sweep_steps=3,
        output_tokens=16,
        max_seconds=2.0,
        max_requests=4,
        sample_interval=0.25,
        outputs=["json"],
        request_format="/v1/chat/completions",
        random_seed=71,
        rampup=0.0,
        warmup=None,
        cooldown=None,
        max_errors=1,
        disable_console=True,
        skip_existing=False,
        profiles=["synchronous", "concurrent"],
    )
    return replace(config, **changes)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _evidence(root):
    # Bytes alone would miss an identical rewrite of supposedly immutable evidence.
    return {
        str(path.relative_to(root)): (
            path.read_bytes() if path.is_file() else None,
            path.stat().st_mtime_ns,
        )
        for path in root.rglob("*")
    }


def _cell(model_dir, identity, context=128, profile="synchronous", return_code=0):
    path = model_dir / f"ctx_{context}" / profile / "run_manifest.json"
    _write_json(
        path,
        {
            "profile": profile,
            "context_tokens": context,
            "run_identity_sha256": identity,
            "return_code": return_code,
        },
    )
    (path.parent / "guidellm_stdout.log").write_bytes(b"original evidence\n")
    return path


def _assert_rejected_unchanged(config, root, runner_path=_RUNNER):
    before = _evidence(root)
    with pytest.raises(RuntimeError, match="Refusing to reuse"):
        matrix_runner._prepare_run(config, runner_path)
    assert _evidence(root) == before


@pytest.mark.parametrize("with_manifest", [False, True])
def test_fresh_identity_and_matching_resume_preserve_evidence(tmp_path, with_manifest):
    config = _config(tmp_path)
    supplied = {"model": "served-checkpoint", "engine": {"dtype": "float16"}}
    if with_manifest:
        manifest = tmp_path / "server.json"
        _write_json(manifest, supplied)
        config = replace(config, server_manifest=manifest)

    model_dir, identity_hash, completed = matrix_runner._prepare_run(config, _RUNNER)
    assert model_dir == config.output_root / "test__model"
    assert completed == set()
    identity = json.loads((model_dir / "run_identity.json").read_text())
    assert identity == matrix_runner._run_identity(config, _RUNNER)
    assert (
        identity_hash
        == hashlib.sha256(matrix_runner._identity_json(identity).encode("utf-8")).hexdigest()
    )
    assert identity["runner_source_sha256"] == hashlib.sha256(_RUNNER.read_bytes()).hexdigest()
    assert identity["client"] == matrix_runner._client_identity()
    assert identity["server"]["manifest"] == (supplied if with_manifest else None)
    assert (
        json.loads((model_dir / "matrix_config.json").read_text())["random_seed"]
        == config.random_seed
    )

    _cell(model_dir, identity_hash)
    before = _evidence(config.output_root)
    resumed = matrix_runner._prepare_run(replace(config, skip_existing=True), _RUNNER)
    assert resumed == (model_dir, identity_hash, {(128, "synchronous")})
    assert _evidence(config.output_root) == before


def test_identity_ignores_operational_paths_and_manifest_serialization(tmp_path):
    first_manifest = tmp_path / "first-server.json"
    second_manifest = tmp_path / "second-server.json"
    first_manifest.write_text('{"model":"checkpoint","engine":{"tp":1,"dtype":"fp16"}}')
    second_manifest.write_text('{\n "engine": {"dtype": "fp16", "tp": 1}, "model": "checkpoint"\n}')
    config = _config(tmp_path, server_manifest=first_manifest)
    model_dir, identity_hash, _ = matrix_runner._prepare_run(config, _RUNNER)
    _cell(model_dir, identity_hash)
    before = _evidence(config.output_root)

    operational = replace(config, skip_existing=True, server_manifest=second_manifest)
    assert matrix_runner._run_identity(operational, _RUNNER) == matrix_runner._run_identity(
        config, _RUNNER
    )
    assert matrix_runner._prepare_run(operational, _RUNNER)[2] == {(128, "synchronous")}
    relocated = replace(operational, output_root=tmp_path / "other-results")
    other_dir, other_hash, completed = matrix_runner._prepare_run(relocated, _RUNNER)
    assert other_dir != model_dir
    assert other_hash == identity_hash
    assert completed == set()
    assert _evidence(config.output_root) == before


@pytest.mark.parametrize(
    "changes",
    [
        {"random_seed": 72},
        {"output_tokens": 17},
        {"contexts": [128, 512]},
        {"concurrent_rates": [1, 4]},
        {"seed_mode": "fixed"},
        {"duration_mode": "fixed"},
    ],
    ids=["seed", "output-length", "contexts", "concurrency", "seed-policy", "duration-policy"],
)
def test_changed_workload_rejects_without_overwriting(tmp_path, changes):
    config = _config(tmp_path)
    model_dir, identity_hash, _ = matrix_runner._prepare_run(config, _RUNNER)
    _cell(model_dir, identity_hash)
    _assert_rejected_unchanged(replace(config, skip_existing=True, **changes), config.output_root)


def test_changed_server_contents_reject_without_overwriting(tmp_path):
    manifest = tmp_path / "server.json"
    _write_json(manifest, {"checkpoint": "first"})
    config = _config(tmp_path, server_manifest=manifest)
    model_dir, identity_hash, _ = matrix_runner._prepare_run(config, _RUNNER)
    _cell(model_dir, identity_hash)
    _write_json(manifest, {"checkpoint": "second"})
    _assert_rejected_unchanged(replace(config, skip_existing=True), config.output_root)


def test_changed_runner_source_rejects_without_overwriting(tmp_path):
    runner_copy = tmp_path / "runner.py"
    runner_copy.write_bytes(_RUNNER.read_bytes())
    config = _config(tmp_path)
    model_dir, identity_hash, _ = matrix_runner._prepare_run(config, runner_copy)
    _cell(model_dir, identity_hash)
    runner_copy.write_bytes(runner_copy.read_bytes() + b"\n# Changed runner revision\n")
    _assert_rejected_unchanged(replace(config, skip_existing=True), config.output_root, runner_copy)


def test_stored_older_client_identity_rejects_without_overwriting(tmp_path):
    config = _config(tmp_path)
    model_dir, identity_hash, _ = matrix_runner._prepare_run(config, _RUNNER)
    _cell(model_dir, identity_hash)
    identity_path = model_dir / "run_identity.json"
    identity = json.loads(identity_path.read_text())
    current_client = matrix_runner._client_identity()
    assert identity["client"] == current_client
    identity["client"]["python_version"] += " (older environment)"
    assert identity["client"] != current_client
    _write_json(identity_path, identity)
    _assert_rejected_unchanged(replace(config, skip_existing=True), config.output_root)


@pytest.mark.parametrize(
    "contents",
    [
        '{"unfinished":',
        "[]",
        "null",
        '"server"',
        '{"x":NaN}',
        '{"x":Infinity}',
        '{"x":-Infinity}',
        '{"nested":{"x":1e999}}',
    ],
    ids=[
        "invalid-json",
        "array",
        "null",
        "string",
        "nan",
        "infinity",
        "negative-infinity",
        "overflow",
    ],
)
def test_invalid_server_manifest_rejects_before_creating_run(tmp_path, contents):
    manifest = tmp_path / "server.json"
    manifest.write_text(contents, encoding="utf-8")
    config = _config(tmp_path, server_manifest=manifest)
    before = _evidence(tmp_path)
    with pytest.raises(RuntimeError, match="Cannot read JSON object"):
        matrix_runner._prepare_run(config, _RUNNER)
    assert not config.output_root.exists()
    assert _evidence(tmp_path) == before


def test_legacy_run_without_identity_rejects_without_overwriting(tmp_path):
    config = _config(tmp_path, skip_existing=True)
    model_dir = config.output_root / "test__model"
    _write_json(model_dir / "matrix_config.json", {"model": config.model})
    _cell(model_dir, "legacy-unverified")
    _assert_rejected_unchanged(config, config.output_root)
    assert not (model_dir / "run_identity.json").exists()


@pytest.mark.parametrize(
    "field", ["profile", "context_tokens", "run_identity_sha256", "return_code"]
)
def test_missing_required_cell_field_rejects_without_overwriting(tmp_path, field):
    config = _config(tmp_path, skip_existing=True)
    model_dir, identity_hash, _ = matrix_runner._prepare_run(config, _RUNNER)
    path = _cell(model_dir, identity_hash)
    cell = json.loads(path.read_text())
    del cell[field]
    _write_json(path, cell)
    _assert_rejected_unchanged(config, config.output_root)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("profile", "sweep"),
        ("context_tokens", 256),
        ("run_identity_sha256", "different-run"),
        ("return_code", "0"),
        ("return_code", False),
        ("return_code", 0.0),
        ("return_code", {}),
    ],
)
def test_incompatible_cell_metadata_rejects_without_overwriting(tmp_path, field, value):
    config = _config(tmp_path, skip_existing=True)
    model_dir, identity_hash, _ = matrix_runner._prepare_run(config, _RUNNER)
    path = _cell(model_dir, identity_hash)
    cell = json.loads(path.read_text())
    cell[field] = value
    _write_json(path, cell)
    _assert_rejected_unchanged(config, config.output_root)


@pytest.mark.parametrize("contents", ['{"unfinished":', "[]", '{"return_code":NaN}'])
def test_malformed_cell_json_rejects_without_overwriting(tmp_path, contents):
    config = _config(tmp_path, skip_existing=True)
    model_dir, identity_hash, _ = matrix_runner._prepare_run(config, _RUNNER)
    path = _cell(model_dir, identity_hash)
    path.write_text(contents, encoding="utf-8")
    _assert_rejected_unchanged(config, config.output_root)


def test_orphan_cell_artifacts_reject_without_overwriting(tmp_path):
    config = _config(tmp_path, skip_existing=True)
    model_dir, identity_hash, _ = matrix_runner._prepare_run(config, _RUNNER)
    path = _cell(model_dir, identity_hash)
    path.unlink()
    _assert_rejected_unchanged(config, config.output_root)


def test_resume_only_returns_successful_cells_as_completed(tmp_path):
    config = _config(tmp_path, skip_existing=True)
    model_dir, identity_hash, _ = matrix_runner._prepare_run(config, _RUNNER)
    _cell(model_dir, identity_hash, context=128, profile="synchronous", return_code=0)
    _cell(model_dir, identity_hash, context=128, profile="concurrent", return_code=1)
    _cell(model_dir, identity_hash, context=256, profile="synchronous", return_code=None)
    (model_dir / "ctx_256" / "concurrent").mkdir(parents=True)
    before = _evidence(config.output_root)
    expected = {(128, "synchronous")}
    assert matrix_runner._resume_cells(config, model_dir, identity_hash) == expected
    assert matrix_runner._prepare_run(config, _RUNNER) == (model_dir, identity_hash, expected)
    assert _evidence(config.output_root) == before


def test_cli_rejects_legacy_output_before_unreachable_network_target(tmp_path):
    config = _config(tmp_path, skip_existing=True)
    model_dir = config.output_root / "test__model"
    _write_json(model_dir / "matrix_config.json", {"model": config.model})
    before = _evidence(config.output_root)
    # Reserve a local port without listening: no server or HTTP test double runs.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reserved:
        reserved.bind(("127.0.0.1", 0))
        target = f"http://127.0.0.1:{reserved.getsockname()[1]}"
        env = os.environ.copy()
        env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1"
        result = subprocess.run(
            [
                sys.executable,
                str(_RUNNER),
                "--model",
                config.model,
                "--output-root",
                str(config.output_root),
                "--target",
                target,
                "--metrics-url",
                f"{target}/metrics",
                "--skip-existing",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=env,
        )
    assert result.returncode != 0
    assert "Refusing to reuse" in result.stderr
    assert "no run_identity.json provenance" in result.stderr
    assert "ConnectionError" not in result.stderr
    assert "_preflight_metrics" not in result.stderr
    assert _evidence(config.output_root) == before
    assert not (model_dir / "run_identity.json").exists()


@pytest.mark.parametrize("mode", ["fixed", "scaled"])
@pytest.mark.parametrize(
    ("context", "multiplier"),
    [(8000, 1), (8001, 2), (32000, 2), (32001, 4), (128000, 4), (128001, 8)],
)
def test_duration_policy_preserves_requested_window(tmp_path, mode, context, multiplier):
    config = _config(tmp_path, max_seconds=7, duration_mode=mode)
    command = matrix_runner._guidellm_command(tmp_path, config, "synchronous", context, tmp_path)
    durations = [value for value in command if value.startswith("kind=max_duration,")]
    expected = config.max_seconds * (1 if mode == "fixed" else multiplier)
    assert durations == [f"kind=max_duration,seconds={int(expected)}"]


def test_seed_policies_are_explicit_and_reproducible(tmp_path):
    config = _config(tmp_path, random_seed=137, seed_mode="fixed")
    cells = [("synchronous", 1024), ("concurrent", 1024), ("synchronous", 32768)]
    assert {matrix_runner._cell_seed(config, *cell) for cell in cells} == {137}
    per_cell = replace(config, seed_mode="per-cell")
    first = [matrix_runner._cell_seed(per_cell, *cell) for cell in cells]
    assert len(set(first)) == len(cells)
    assert first == [matrix_runner._cell_seed(per_cell, *cell) for cell in cells]
    changed = replace(per_cell, random_seed=138)
    assert first != [matrix_runner._cell_seed(changed, *cell) for cell in cells]
