#!/usr/bin/env python3
"""Export a SpecForge checkpoint in vLLM's native Laguna DFlash layout."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

try:
    from .prepare_warmstart import (
        checkpoint_keys,
        expected_native_keys,
        expected_training_keys,
        save_sharded,
        validate_source_config,
    )
except ImportError:  # Direct script execution.
    from prepare_warmstart import (
        checkpoint_keys,
        expected_native_keys,
        expected_training_keys,
        save_sharded,
        validate_source_config,
    )


def resolve_source(source: str) -> Path:
    path = Path(source).expanduser()
    if path.is_dir():
        return path.resolve()
    return Path(
        snapshot_download(
            source,
            allow_patterns=[
                "config.json",
                "*.safetensors",
                "*.safetensors.index.json",
            ],
        )
    )


def resolve_training_state(checkpoint: Path) -> Path:
    checkpoint = checkpoint.expanduser().resolve()
    if checkpoint.is_file():
        return checkpoint
    direct = checkpoint / "training_state.pt"
    if direct.is_file():
        return direct

    latest = sorted(checkpoint.glob("*-latest/training_state.pt"))
    if len(latest) == 1:
        return latest[0].resolve()
    if len(latest) > 1:
        raise ValueError(
            f"multiple SpecForge checkpoint lineages under {checkpoint}: {latest}"
        )
    candidates = sorted(checkpoint.glob("*-step*/training_state.pt"))
    if not candidates:
        raise ValueError(f"no SpecForge training_state.pt found under {checkpoint}")
    return candidates[-1]


def validate_config(config: dict) -> None:
    # Export intentionally restores the checked-in poolside config byte-for-field
    # rather than adapting the SpecForge Qwen3 training config in place.
    validate_source_config(config)


def iter_native_tensors(
    source_keys: set[str],
    trained: dict[str, torch.Tensor],
    config: dict,
) -> Iterator[tuple[str, torch.Tensor]]:
    q_width = int(config["num_attention_heads"]) * int(config["head_dim"])
    kv_width = int(config["num_key_value_heads"]) * int(config["head_dim"])
    hidden_size = int(config["hidden_size"])

    for key in sorted(source_keys):
        if key.endswith(".self_attn.qkv_proj.weight"):
            prefix = key.removesuffix("qkv_proj.weight")
            names = [prefix + projection + ".weight" for projection in ("q_proj", "k_proj", "v_proj")]
            parts = [trained.pop(name, None) for name in names]
            if any(part is None for part in parts):
                missing = [name for name, part in zip(names, parts) if part is None]
                raise ValueError(f"trained checkpoint lacks QKV tensors: {missing}")
            expected_shapes = (
                (q_width, hidden_size),
                (kv_width, hidden_size),
                (kv_width, hidden_size),
            )
            for name, part, expected_shape in zip(names, parts, expected_shapes):
                if tuple(part.shape) != expected_shape:
                    raise ValueError(
                        f"trained tensor {name} shape {tuple(part.shape)} != {expected_shape}"
                    )
            yield key, torch.cat(parts, dim=0)
            continue

        tensor = trained.pop(key, None)
        if tensor is None:
            raise ValueError(f"trained checkpoint lacks required Laguna tensor {key}")
        yield key, tensor

    if trained:
        raise ValueError(
            "trained checkpoint has tensors outside the native Laguna schema: "
            f"{sorted(trained)}"
        )


def export(checkpoint: Path, warm_source: str, output: Path) -> Path:
    state_path = resolve_training_state(checkpoint)
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"{state_path}: checkpoint payload is not a mapping")
    if state.get("strategy") != "dflash":
        raise ValueError(
            f"{state_path}: strategy={state.get('strategy')!r}, expected 'dflash'"
        )
    trained = state.get("draft_state_dict")
    if not isinstance(trained, dict) or not trained:
        raise ValueError(f"{state_path}: missing draft_state_dict")
    trained_keys = set(trained)
    expected_trained = expected_training_keys()
    if trained_keys != expected_trained:
        raise ValueError(
            "SpecForge Laguna tensor key mismatch: "
            f"missing={sorted(expected_trained - trained_keys)}, "
            f"unexpected={sorted(trained_keys - expected_trained)}"
        )

    source = resolve_source(warm_source)
    config = json.loads((source / "config.json").read_text())
    validate_config(config)
    source_keys = checkpoint_keys(source)
    expected_source = expected_native_keys()
    if source_keys != expected_source:
        raise ValueError(
            "poolside Laguna tensor key mismatch: "
            f"missing={sorted(expected_source - source_keys)}, "
            f"unexpected={sorted(source_keys - expected_source)}"
        )

    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    weight_map = save_sharded(iter_native_tensors(source_keys, trained, config), output)
    if set(weight_map) != source_keys:
        raise ValueError("exported Laguna checkpoint key set differs from poolside reference")

    temporary_config = output / "config.json.tmp"
    temporary_config.write_text(json.dumps(config, indent=2) + "\n")
    temporary_config.replace(output / "config.json")
    print(f"vLLM-loadable Laguna checkpoint: {output}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--warm-source", default="poolside/Laguna-S-2.1-DFlash")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.checkpoint, args.warm_source, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
