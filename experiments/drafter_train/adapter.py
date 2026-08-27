#!/usr/bin/env python3
"""Validate Shooting Brake captures and stage zero-copy SpecForge inputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch

FORMAT = "shooting_brake_dflash_v1"
TARGET_LAYER_IDS = (1, 10, 19, 29, 38, 47)
HIDDEN_SIZE = 3072
SPECFORGE_FORMAT = "specforge_hidden_states_v1"
SPECFORGE_FEATURE_KEYS = ("input_ids", "loss_mask", "hidden_states")
# Must equal data.max_length in pilot.yaml -- asserted by
# tests/test_adapter.py so the two can never drift.
MAX_LENGTH_DEFAULT = 3072


def capture_paths(source: Path) -> Iterator[Path]:
    """Yield capture files without materializing a dataset-sized path list."""

    if not source.is_dir():
        raise ValueError(f"capture source is not a directory: {source}")
    found = False
    with os.scandir(source) as entries:
        for entry in entries:
            if not entry.name.endswith(".pt") or not entry.is_file():
                continue
            found = True
            yield Path(entry.path)
    if not found:
        raise ValueError(f"no .pt capture records found in {source}")


def load_record(path: Path) -> dict[str, Any]:
    record = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(record, dict):
        raise ValueError(f"{path}: expected a mapping")
    return record


def validate_record(record: dict[str, Any], path: Path) -> dict[str, int]:
    required = {
        "id",
        "format",
        "input_ids",
        "loss_mask",
        "response_start",
        "hidden_states",
        "hidden_states_by_layer",
        "target_layer_ids",
    }
    missing = required - record.keys()
    if missing:
        raise ValueError(f"{path}: missing keys {sorted(missing)}")
    if record["format"] != FORMAT:
        raise ValueError(
            f"{path}: format is {record['format']!r}, expected {FORMAT!r}"
        )
    if not isinstance(record["id"], str) or not record["id"]:
        raise ValueError(f"{path}: id must be a non-empty string")

    tensor_keys = (
        "input_ids",
        "loss_mask",
        "hidden_states",
        "hidden_states_by_layer",
        "target_layer_ids",
    )
    non_tensors = [key for key in tensor_keys if not torch.is_tensor(record[key])]
    if non_tensors:
        raise ValueError(f"{path}: expected tensors for keys {non_tensors}")

    tokens = record["input_ids"]
    loss_mask = record["loss_mask"]
    flat = record["hidden_states"]
    by_layer = record["hidden_states_by_layer"]
    layer_ids_tensor = record["target_layer_ids"]
    if layer_ids_tensor.dtype != torch.int64 or layer_ids_tensor.shape != (
        len(TARGET_LAYER_IDS),
    ):
        raise ValueError(f"{path}: target_layer_ids must be int64 [6]")
    layer_ids = tuple(int(value) for value in layer_ids_tensor.tolist())
    if layer_ids != TARGET_LAYER_IDS:
        raise ValueError(
            f"{path}: target_layer_ids={layer_ids}, expected {TARGET_LAYER_IDS}"
        )
    if tokens.ndim != 1 or tokens.dtype != torch.int32:
        raise ValueError(
            f"{path}: input_ids must be int32 [T], got "
            f"{tokens.dtype} {tuple(tokens.shape)}"
        )
    length = int(tokens.shape[0])
    if loss_mask.shape != (length,) or loss_mask.dtype != torch.bool:
        raise ValueError(f"{path}: loss_mask must be bool [T]")
    if by_layer.shape != (length, len(TARGET_LAYER_IDS), HIDDEN_SIZE):
        raise ValueError(
            f"{path}: hidden_states_by_layer must be [T,6,3072], got {tuple(by_layer.shape)}"
        )
    if flat.shape != (length, len(TARGET_LAYER_IDS) * HIDDEN_SIZE):
        raise ValueError(
            f"{path}: hidden_states must be [T,18432], got {tuple(flat.shape)}"
        )
    if flat.dtype != torch.bfloat16 or by_layer.dtype != torch.bfloat16:
        raise ValueError(f"{path}: hidden states must be bfloat16")
    if (
        flat.data_ptr() != by_layer.data_ptr()
        or flat.storage_offset() != by_layer.storage_offset()
        or not flat.is_contiguous()
        or not by_layer.is_contiguous()
    ):
        raise ValueError(
            f"{path}: hidden_states must be the contiguous flattened view of "
            "hidden_states_by_layer without a tensor copy"
        )

    response_start = record["response_start"]
    if isinstance(response_start, bool) or not isinstance(response_start, int):
        raise ValueError(f"{path}: response_start must be an integer")
    if not 0 <= response_start < length:
        raise ValueError(f"{path}: response_start={response_start} outside [0,{length})")
    if bool(loss_mask[:response_start].any()) or not bool(
        loss_mask[response_start:].all()
    ):
        raise ValueError(f"{path}: loss_mask does not match response_start")
    supervised_tokens = length - response_start
    if supervised_tokens < 2:
        raise ValueError(f"{path}: fewer than two consecutive supervised tokens")
    for start in range(0, length, 256):
        if not bool(torch.isfinite(flat[start : start + 256]).all()):
            raise ValueError(f"{path}: hidden_states contains non-finite values")
    return {
        "tokens": length,
        "supervised_tokens": supervised_tokens,
        "bytes": path.stat().st_size,
    }


def _stage_link(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        if not destination.samefile(source):
            raise ValueError(f"{destination} already points at a different capture")
        return
    try:
        os.link(source, destination)
    except OSError:
        destination.symlink_to(source.resolve())


def stage_capture(
    source: Path, output: Path, max_length: int = MAX_LENGTH_DEFAULT
) -> dict[str, Any]:
    """Validate and link captures one at a time into a SpecForge feature tree.

    ``max_length`` must match ``data.max_length`` in the training config.
    SpecForge truncates every sample to that window BEFORE computing the
    loss (hidden_states_data.py normalize_offline_sample), so a record whose
    prompt alone exceeds it loses its entire supervised region and aborts the
    run with "require two consecutive supervised tokens". Four of 808 pilot
    records hit exactly that (prompts of 3,139-5,903 tokens, 2026-08-26).
    Such records are skipped and accounted, not fatal: they are unusable for
    THIS window, and a wider window is a config decision, not a data fault.
    """

    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = 0
    tokens = 0
    supervised_tokens = 0
    captured_bytes = 0
    skipped_truncated: list[str] = []
    for path in capture_paths(source):
        record = load_record(path)
        try:
            row = validate_record(record, path)
            # Supervised region must survive truncation to max_length.
            usable = max(0, min(int(record["input_ids"].shape[0]), max_length)
                         - int(record["response_start"]))
            record_id = str(record["id"])
        finally:
            # Do not retain mmap-backed tensors while walking the next record.
            del record
        if usable < 2:
            skipped_truncated.append(record_id)
            continue
        destination = output / f"{path.stem}.ckpt"
        _stage_link(path, destination)
        records += 1
        tokens += row["tokens"]
        supervised_tokens += row["supervised_tokens"]
        captured_bytes += row["bytes"]

    manifest = {
        "format": FORMAT,
        "specforge_format": SPECFORGE_FORMAT,
        "feature_keys": list(SPECFORGE_FEATURE_KEYS),
        "target_layer_ids": list(TARGET_LAYER_IDS),
        "hidden_size": HIDDEN_SIZE,
        "max_length": max_length,
        "records": records,
        "tokens": tokens,
        "supervised_tokens": supervised_tokens,
        "captured_bytes": captured_bytes,
        "skipped_truncated": skipped_truncated,
        "source": str(source),
        "output": str(output),
    }
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(output / "manifest.json")
    return manifest


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path.home() / "sb_hidden_capture")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-length",
        type=int,
        default=MAX_LENGTH_DEFAULT,
        help="must match data.max_length in the training config",
    )
    args = parser.parse_args(argv)
    manifest = stage_capture(args.input, args.output, args.max_length)
    skipped = manifest["skipped_truncated"]
    print(
        f"staged {manifest['records']} records / {manifest['tokens']} tokens "
        f"({manifest['supervised_tokens']} supervised) at {manifest['output']}"
    )
    if skipped:
        print(
            f"skipped {len(skipped)} record(s) whose supervised region falls "
            f"outside max_length={args.max_length}: {skipped[:3]}"
            f"{' ...' if len(skipped) > 3 else ''}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
