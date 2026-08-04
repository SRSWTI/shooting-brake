#!/usr/bin/env python3
"""Validate the Phase-1 golden and expert bank with compressed-tensors' NVFP4 oracle."""

import json
import os
import struct
from pathlib import Path

import torch
from compressed_tensors.compressors.nvfp4 import NVFP4PackedCompressor
from compressed_tensors.quantization import QuantizationConfig
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent.parent
PHASE1 = ROOT / "phase1"
MODEL_SNAPSHOT = "739af1e7aac320af1682ed1e0cce369af4c5265d"
DEFAULT_MODEL_DIR = (
    Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    / "hub"
    / "models--unsloth--Qwen3.6-35B-A3B-NVFP4"
    / "snapshots"
    / MODEL_SNAPSHOT
)
MODEL_DIR = Path(os.environ.get("SB_NVFP4_MODEL_DIR", DEFAULT_MODEL_DIR))
PREFIX = "model.language_model.layers.0.mlp.experts.0"
HEADER = struct.Struct("<8sIIIIIQQQQ")
K = 2048
I = 512
ATOL = 2e-6
RMSE_LIMIT = 5e-7


def load_tensor(shards: list[Path], suffix: str) -> torch.Tensor:
    key = f"{PREFIX}.{suffix}"
    for shard in shards:
        with safe_open(str(shard), framework="pt") as tensors:
            if key in tensors.keys():
                return tensors.get_tensor(key)
    raise KeyError(key)


def official_weight(
    shards: list[Path], projection: str, scheme
) -> torch.Tensor:
    state = {
        "weight_packed": load_tensor(shards, f"{projection}.weight_packed"),
        "weight_scale": load_tensor(shards, f"{projection}.weight_scale"),
        "weight_global_scale": load_tensor(
            shards, f"{projection}.weight_global_scale"
        ),
    }
    return NVFP4PackedCompressor.decompress(state, scheme)["weight"].float()


def main() -> None:
    model_dir = MODEL_DIR
    if not model_dir.is_dir():
        raise RuntimeError(f"checkpoint snapshot not found: {model_dir}")
    shards = sorted(model_dir.glob("model*.safetensors"))

    config = json.loads((model_dir / "config.json").read_text())
    quant_config = QuantizationConfig.model_validate(config["quantization_config"])
    scheme = quant_config.config_groups["group_1"]

    gate_packed = load_tensor(shards, "gate_proj.weight_packed").contiguous()
    up_packed = load_tensor(shards, "up_proj.weight_packed").contiguous()
    down_packed = load_tensor(shards, "down_proj.weight_packed").contiguous()
    gate_scale = load_tensor(shards, "gate_proj.weight_scale").contiguous()
    up_scale = load_tensor(shards, "up_proj.weight_scale").contiguous()
    down_scale = load_tensor(shards, "down_proj.weight_scale").contiguous()
    gate_global = load_tensor(shards, "gate_proj.weight_global_scale").item()
    up_global = load_tensor(shards, "up_proj.weight_global_scale").item()
    down_global = load_tensor(shards, "down_proj.weight_global_scale").item()
    if gate_global != up_global:
        raise AssertionError("gate/up global scales differ")

    bank = (PHASE1 / "expert_bank.bin").open("rb")
    header_bytes = bank.read(HEADER.size)
    magic, layers, experts, hidden, intermediate, _, w13_b, s13_b, w2_b, s2_b = (
        HEADER.unpack(header_bytes)
    )
    if (magic, layers, experts, hidden, intermediate) != (
        b"SBEXP001",
        32,
        256,
        K,
        I,
    ):
        raise AssertionError("unexpected expert-bank header")
    expected_record = w13_b + s13_b + w2_b + s2_b + 8
    if (PHASE1 / "expert_bank.bin").stat().st_size != (
        HEADER.size + layers * experts * expected_record
    ):
        raise AssertionError("expert-bank size does not match header")

    expected_w13 = torch.cat([gate_packed, up_packed], dim=0).numpy().tobytes()
    expected_s13 = (
        torch.cat([gate_scale, up_scale], dim=0)
        .contiguous()
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )
    expected_w2 = down_packed.numpy().tobytes()
    expected_s2 = down_scale.view(torch.uint8).numpy().tobytes()
    for name, expected, size in (
        ("w13", expected_w13, w13_b),
        ("w13_scales", expected_s13, s13_b),
        ("w2", expected_w2, w2_b),
        ("w2_scales", expected_s2, s2_b),
    ):
        actual = bank.read(size)
        if actual != expected:
            raise AssertionError(f"expert-0 {name} bytes differ from checkpoint")
    bank_w13_global, bank_w2_global = struct.unpack("<ff", bank.read(8))
    bank.close()
    if bank_w13_global != torch.tensor(1.0 / gate_global).float().item():
        raise AssertionError("expert-0 w13 global multiplier is wrong")
    if bank_w2_global != torch.tensor(1.0 / down_global).float().item():
        raise AssertionError("expert-0 w2 global multiplier is wrong")

    golden = (PHASE1 / "golden_reference.bin").read_bytes()
    golden_k, golden_i, golden_topk = struct.unpack_from("<III", golden, 0)
    if (golden_k, golden_i, golden_topk) != (K, I, 1):
        raise AssertionError("unexpected golden header")
    hidden = torch.frombuffer(
        bytearray(golden), dtype=torch.float16, count=K, offset=12
    ).float()
    output_offset = 12 + K * 2 + 8
    expected_output = torch.frombuffer(
        bytearray(golden), dtype=torch.float32, count=K, offset=output_offset
    ).clone()

    gate = official_weight(shards, "gate_proj", scheme)
    up = official_weight(shards, "up_proj", scheme)
    down = official_weight(shards, "down_proj", scheme)
    activation = torch.nn.functional.silu(gate @ hidden) * (up @ hidden)
    official_output = down @ activation
    error = (official_output - expected_output).abs()
    max_abs = error.max().item()
    rmse = error.square().mean().sqrt().item()
    if max_abs > ATOL or rmse > RMSE_LIMIT:
        raise AssertionError(
            f"compressed-tensors oracle mismatch: max_abs={max_abs:.3e}, "
            f"rmse={rmse:.3e}"
        )

    print(
        "PASS compressed-tensors oracle: "
        f"bank bytes exact, max_abs={max_abs:.3e}, rmse={rmse:.3e}"
    )


if __name__ == "__main__":
    main()
