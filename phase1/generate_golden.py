#!/usr/bin/env python3
"""
Shooting Brake Phase 1 — Generate golden MoE reference for correctness validation.

Uses NVIDIA ModelOpt's E2M1 table and compressed-tensors' dequantization formula:
  weight = e2m1_value * (block_scale / global_scale)

Computes reference from fp16 input (same as what B70 kernel receives).
"""

import glob
import os
import struct
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

MODEL_SNAPSHOT = "739af1e7aac320af1682ed1e0cce369af4c5265d"
DEFAULT_MODEL_DIR = (
    Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    / "hub"
    / "models--unsloth--Qwen3.6-35B-A3B-NVFP4"
    / "snapshots"
    / MODEL_SNAPSHOT
)
MODEL_DIR = Path(os.environ.get("SB_NVFP4_MODEL_DIR", DEFAULT_MODEL_DIR))

OUTPUT = Path(__file__).parent / "golden_reference.bin"

K = 2048
I = 512

# NVIDIA ModelOpt E2M1 magnitude table (index = 3-bit unsigned part of nibble)
# _E2M1_MAGNITUDE = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)


def unpack_e2m1(packed_uint8):
    """Unpack uint8 packed E2M1 nibbles → float32.
    Each byte = 2 nibbles (low nibble = even index, high = odd).
    """
    low = (packed_uint8 & 0x0F).long()
    high = ((packed_uint8 >> 4) & 0x0F).long()
    out = torch.stack([low, high], dim=-1).flatten(-2)
    return E2M1_TABLE[out]


def decode_e4m3(uint8_arr):
    """Decode E4M3 bytes to float32. E4M3: 1 sign + 4 exp (bias 7) + 3 mant."""
    bits = uint8_arr.astype(np.uint32)
    sign = (bits >> 7) & 1
    exp = (bits >> 3) & 0xF
    mant = bits & 0x7
    result = np.zeros_like(bits, dtype=np.float32)
    # Subnormal (exp=0, mant>0): value = mant/8 * 2^(-6)
    sub = (exp == 0) & (mant != 0)
    result[sub] = (mant[sub].astype(np.float32) / 8.0) * (2.0 ** -6)
    # Normal (exp>0): value = 2^(exp-7) * (1 + mant/8)
    norm = exp > 0
    result[norm] = (1.0 + mant[norm].astype(np.float32) / 8.0) * \
                    np.power(2.0, exp[norm].astype(np.float32) - 7.0)
    result[sign == 1] = -result[sign == 1]
    return torch.from_numpy(result)


def dequantize_weight(packed, scales_f8, global_scale_val):
    """Dequantize NVFP4 weight: weight = e2m1 * (block_scale / global_scale).
    
    compressed-tensors forward_helpers.py:256: scale = scale / global_scale
    """
    e2m1_vals = unpack_e2m1(packed)  # [N_out, N_in]
    block_scales = decode_e4m3(scales_f8.view(torch.uint8).numpy())  # [N_out, N_in/16]
    
    N_out, N_in = e2m1_vals.shape
    # Each block scale covers 16 consecutive weights
    scales_expanded = block_scales.unsqueeze(-1).expand(-1, -1, 16).reshape(N_out, N_in)
    
    # Dequantize: weight = e2m1 * (block_scale / global_scale)
    return e2m1_vals * (scales_expanded / global_scale_val)


def main():
    print("Generating golden MoE reference (correct NVFP4 dequant)...")
    
    shard_paths = sorted(glob.glob(f"{MODEL_DIR}/model*.safetensors"))
    prefix = "model.language_model.layers.0.mlp.experts.0"
    
    tensors = {}
    needed = [
        "gate_proj.weight_packed", "gate_proj.weight_scale", "gate_proj.weight_global_scale",
        "up_proj.weight_packed", "up_proj.weight_scale", "up_proj.weight_global_scale",
        "down_proj.weight_packed", "down_proj.weight_scale", "down_proj.weight_global_scale",
    ]
    for suffix in needed:
        key = f"{prefix}.{suffix}"
        for sp in shard_paths:
            with safe_open(sp, framework="pt") as f:
                if key in f.keys():
                    tensors[suffix] = f.get_tensor(key)
                    break
    
    gg = tensors["gate_proj.weight_global_scale"].item()
    dg = tensors["down_proj.weight_global_scale"].item()
    print(f"Global scales: gate={gg}, down={dg}")
    
    gate_w = dequantize_weight(tensors["gate_proj.weight_packed"],
                                tensors["gate_proj.weight_scale"], gg)
    up_w = dequantize_weight(tensors["up_proj.weight_packed"],
                              tensors["up_proj.weight_scale"], gg)
    down_w = dequantize_weight(tensors["down_proj.weight_packed"],
                                tensors["down_proj.weight_scale"], dg)
    
    print(f"gate_w: {gate_w.shape}, range [{gate_w.min():.4f}, {gate_w.max():.4f}]")
    print(f"up_w:   {up_w.shape}, range [{up_w.min():.4f}, {up_w.max():.4f}]")
    print(f"down_w: {down_w.shape}, range [{down_w.min():.4f}, {down_w.max():.4f}]")
    
    # Create deterministic fp16 input FIRST (advisory: compute reference from fp16)
    torch.manual_seed(42)
    hidden_fp16 = (torch.randn(K, dtype=torch.float32) * 0.1).to(torch.float16)
    hidden_f32 = hidden_fp16.to(torch.float32)  # compute reference from fp16 input
    
    # MoE: y = down(silu(gate @ x) * (up @ x))
    gate_out = gate_w @ hidden_f32    # [I]
    up_out = up_w @ hidden_f32        # [I]
    act = torch.nn.functional.silu(gate_out) * up_out
    output = down_w @ act             # [K]
    
    print(f"\nGolden output: range [{output.min():.6f}, {output.max():.6f}]")
    print(f"  mean={output.mean():.6f}, std={output.std():.6f}")
    print(f"  first 8: {[f'{v:.6f}' for v in output[:8].tolist()]}")
    
    with open(OUTPUT, "wb") as f:
        f.write(struct.pack("<III", K, I, 1))  # header: K, I, topk=1
        f.write(hidden_fp16.numpy().tobytes())  # [K] fp16
        f.write(struct.pack("<i", 0))   # expert 0
        f.write(struct.pack("<f", 1.0)) # weight 1.0
        f.write(output.numpy().tobytes())  # [K] fp32
    
    print(f"\nSaved to {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
