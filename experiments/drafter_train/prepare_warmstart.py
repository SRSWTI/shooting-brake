#!/usr/bin/env python3
"""Convert poolside's native Laguna DFlash checkpoint for SpecForge training."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

TARGET_LAYER_IDS = [1, 10, 19, 29, 38, 47]
ARCHITECTURE = "LagunaDFlashDraftModel"
MAX_SHARD_SIZE = 512 * 1024 * 1024


def resolve_checkpoint(source: str) -> Path:
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


def validate_source_config(config: dict) -> None:
    expected = {
        "architectures": ["DFlashLagunaForCausalLM"],
        "model_type": "laguna",
        "hidden_size": 3072,
        "intermediate_size": 12288,
        "head_dim": 128,
        "num_attention_heads": 72,
        "num_key_value_heads": 8,
        "num_hidden_layers": 6,
        "vocab_size": 100352,
        "draft_vocab_size": 100352,
        "sliding_window": 512,
        "gating": "per-head",
        "attention_bias": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"poolside Laguna config {key}={config.get(key)!r}, expected {value!r}"
            )
    if config.get("layer_types") != ["sliding_attention"] * 6:
        raise ValueError("poolside Laguna DFlash must have six sliding-attention layers")
    if config.get("eagle_aux_hidden_state_layer_ids") != [2, 11, 20, 30, 39, 48]:
        raise ValueError("poolside Laguna capture-point layer ids are not target ids + 1")
    dflash = config.get("dflash_config") or {}
    expected_dflash = {
        "block_size": 16,
        "mask_token_id": 12,
        "num_target_layers": 48,
        "target_layer_ids": TARGET_LAYER_IDS,
        "causal": True,
    }
    if dflash != expected_dflash:
        raise ValueError(
            f"poolside Laguna dflash_config={dflash!r}, expected {expected_dflash!r}"
        )


def build_config(source_config: dict) -> dict:
    validate_source_config(source_config)
    config = dict(source_config)
    config.update(
        architectures=[ARCHITECTURE],
        model_type="qwen3",
        attention_dropout=0.0,
        attention_bias=False,
        use_cache=True,
        use_sliding_window=True,
        tie_word_embeddings=False,
        max_window_layers=6,
        block_size=16,
        num_target_layers=48,
        dflash_config={
            "block_size": 16,
            "mask_token_id": 12,
            "num_target_layers": 48,
            "target_layer_ids": TARGET_LAYER_IDS,
            "causal": True,
        },
    )
    config.pop("auto_map", None)
    # vLLM capture-point ids are target_layer_ids + 1. SpecForge consumes the
    # raw target layer outputs named by dflash_config.target_layer_ids.
    config.pop("eagle_aux_hidden_state_layer_ids", None)
    return config


def expected_native_keys() -> set[str]:
    keys = {"fc.weight", "hidden_norm.weight", "norm.weight"}
    keys.update(f"aux_hidden_norms.{index}.weight" for index in range(6))
    for index in range(6):
        prefix = f"layers.{index}."
        keys.update(
            {
                prefix + "input_layernorm.weight",
                prefix + "post_attention_layernorm.weight",
                prefix + "self_attn.qkv_proj.weight",
                prefix + "self_attn.o_proj.weight",
                prefix + "self_attn.g_proj.weight",
                prefix + "self_attn.q_norm.weight",
                prefix + "self_attn.k_norm.weight",
                prefix + "mlp.gate_proj.weight",
                prefix + "mlp.up_proj.weight",
                prefix + "mlp.down_proj.weight",
            }
        )
    return keys


def expected_training_keys() -> set[str]:
    keys = expected_native_keys()
    for index in range(6):
        fused = f"layers.{index}.self_attn.qkv_proj.weight"
        keys.remove(fused)
        keys.update(
            {
                f"layers.{index}.self_attn.q_proj.weight",
                f"layers.{index}.self_attn.k_proj.weight",
                f"layers.{index}.self_attn.v_proj.weight",
            }
        )
    return keys


def checkpoint_weight_files(checkpoint: Path) -> list[Path]:
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text())
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"invalid safetensors index: {index_path}")
        filenames = list(dict.fromkeys(weight_map.values()))
        files = [checkpoint / filename for filename in filenames]
    else:
        single = checkpoint / "model.safetensors"
        files = [single] if single.is_file() else sorted(checkpoint.glob("model-*.safetensors"))
    if not files or any(not path.is_file() for path in files):
        raise ValueError(f"no complete model safetensors found under {checkpoint}")
    return files


def checkpoint_keys(checkpoint: Path) -> set[str]:
    keys: set[str] = set()
    for path in checkpoint_weight_files(checkpoint):
        with safe_open(path, framework="pt", device="cpu") as handle:
            overlap = keys.intersection(handle.keys())
            if overlap:
                raise ValueError(f"duplicate checkpoint tensors: {sorted(overlap)}")
            keys.update(handle.keys())
    return keys


def iter_checkpoint_tensors(checkpoint: Path) -> Iterator[tuple[str, torch.Tensor]]:
    for path in checkpoint_weight_files(checkpoint):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                yield key, handle.get_tensor(key)


def iter_training_tensors(
    checkpoint: Path, config: dict
) -> Iterator[tuple[str, torch.Tensor]]:
    q_width = int(config["num_attention_heads"]) * int(config["head_dim"])
    kv_width = int(config["num_key_value_heads"]) * int(config["head_dim"])
    for key, tensor in iter_checkpoint_tensors(checkpoint):
        if not key.endswith(".self_attn.qkv_proj.weight"):
            yield key, tensor
            continue
        expected_shape = (q_width + 2 * kv_width, int(config["hidden_size"]))
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{key} shape {tuple(tensor.shape)} does not match {expected_shape}"
            )
        prefix = key.removesuffix("qkv_proj.weight")
        query, key_weight, value = tensor.split((q_width, kv_width, kv_width), dim=0)
        # Safetensors rejects tensors sharing one backing allocation. Cloning
        # also lets the large fused source tensor be released before a shard flush.
        yield prefix + "q_proj.weight", query.clone()
        yield prefix + "k_proj.weight", key_weight.clone()
        yield prefix + "v_proj.weight", value.clone()


def save_sharded(
    tensors: Iterable[tuple[str, torch.Tensor]], output: Path
) -> dict[str, str]:
    staged: list[tuple[Path, tuple[str, ...]]]= []
    shard: dict[str, torch.Tensor] = {}
    shard_size = 0
    total_size = 0

    def flush() -> None:
        nonlocal shard, shard_size
        if not shard:
            return
        path = output / f".model-{len(staged) + 1:05d}.safetensors.tmp"
        save_file(shard, path, metadata={"format": "pt"})
        staged.append((path, tuple(shard)))
        shard = {}
        shard_size = 0

    seen: set[str] = set()
    for key, tensor in tensors:
        if key in seen:
            raise ValueError(f"duplicate output tensor {key}")
        seen.add(key)
        tensor_size = tensor.numel() * tensor.element_size()
        if shard and shard_size + tensor_size > MAX_SHARD_SIZE:
            flush()
        shard[key] = tensor.contiguous()
        shard_size += tensor_size
        total_size += tensor_size
    flush()
    if not staged:
        raise ValueError("refusing to save an empty warm-start checkpoint")

    for path in output.glob("model-*.safetensors"):
        path.unlink()
    for name in ("model.safetensors", "model.safetensors.index.json"):
        path = output / name
        if path.exists():
            path.unlink()

    count = len(staged)
    weight_map: dict[str, str] = {}
    for index, (temporary, keys) in enumerate(staged, start=1):
        filename = (
            "model.safetensors"
            if count == 1
            else f"model-{index:05d}-of-{count:05d}.safetensors"
        )
        temporary.replace(output / filename)
        weight_map.update(dict.fromkeys(keys, filename))
    if count > 1:
        temporary_index = output / "model.safetensors.index.json.tmp"
        temporary_index.write_text(
            json.dumps(
                {"metadata": {"total_size": total_size}, "weight_map": weight_map},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        temporary_index.replace(output / "model.safetensors.index.json")
    return weight_map


def convert(source: str, output: Path) -> Path:
    checkpoint = resolve_checkpoint(source)
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_config = json.loads((checkpoint / "config.json").read_text())
    config = build_config(source_config)

    source_keys = checkpoint_keys(checkpoint)
    expected_source = expected_native_keys()
    if source_keys != expected_source:
        raise ValueError(
            "poolside Laguna tensor key mismatch: "
            f"missing={sorted(expected_source - source_keys)}, "
            f"unexpected={sorted(source_keys - expected_source)}"
        )
    weight_map = save_sharded(iter_training_tensors(checkpoint, config), output)
    expected_output = expected_training_keys()
    if set(weight_map) != expected_output:
        raise ValueError(
            "converted Laguna tensor key mismatch: "
            f"missing={sorted(expected_output - set(weight_map))}, "
            f"unexpected={sorted(set(weight_map) - expected_output)}"
        )

    temporary_config = output / "config.json.tmp"
    temporary_config.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    temporary_config.replace(output / "config.json")
    print(f"warm start ready: {output}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="poolside/Laguna-S-2.1-DFlash")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    convert(args.source, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
