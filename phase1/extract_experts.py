#!/usr/bin/env python3
"""
Shooting Brake Phase 1 — Extract NVFP4 expert weights from unsloth safetensors.

Optimized: opens each shard once, batch-loads all needed tensors.
Writes to .tmp then atomically renames on success.

Only layers 0-31 (NVFP4). Layers 32-39 use FP8 and stay on CUDA.
"""

import glob
import os
import struct
import sys
import time
from pathlib import Path

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

OUTPUT = Path(__file__).parent / "expert_bank.bin"
TMP_OUTPUT = OUTPUT.with_suffix(".bin.tmp")

K = 2048
I = 512
NUM_LAYERS = 32
EXPERTS_PER_LAYER = 256
GLOBAL_SCALE_DIVISOR = 4194304.0

HEADER_FMT = "<8sIIIIIQQQQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
MAGIC = b"SBEXP001"


def main():
    shard_paths = sorted(glob.glob(f"{MODEL_DIR}/model*.safetensors"))
    print(f"Shards: {len(shard_paths)}")

    # Pass 1: build key→shard index (fast, just reads headers)
    shard_index = {}
    for sp in shard_paths:
        with safe_open(sp, framework="pt") as f:
            for key in f.keys():
                shard_index[key] = sp

    # Collect all keys we need
    needed = set()
    for layer in range(NUM_LAYERS):
        for exp in range(EXPERTS_PER_LAYER):
            p = f"model.language_model.layers.{layer}.mlp.experts.{exp}"
            for proj in ("gate_proj", "up_proj", "down_proj"):
                needed.add(f"{p}.{proj}.weight_packed")
                needed.add(f"{p}.{proj}.weight_scale")
                needed.add(f"{p}.{proj}.weight_global_scale")

    # Group needed keys by shard
    shard_keys = {sp: [] for sp in shard_paths}
    for key in needed:
        if key in shard_index:
            shard_keys[shard_index[key]].append(key)
        else:
            print(f"WARNING: key not found: {key}")

    # Pass 2: load all needed tensors from each shard (one open per shard)
    tensors = {}
    for sp in shard_paths:
        keys_in_shard = shard_keys[sp]
        if not keys_in_shard:
            continue
        print(f"  Loading {len(keys_in_shard)} tensors from {sp.split('/')[-1]}...")
        with safe_open(sp, framework="pt") as f:
            for key in keys_in_shard:
                tensors[key] = f.get_tensor(key)

    print(f"  Total tensors loaded: {len(tensors)}")

    # Calculate sizes
    w13_bytes = 2 * I * (K // 2)
    s13_bytes = 2 * I * (K // 16)
    w2_bytes = K * (I // 2)
    s2_bytes = K * (I // 16)
    expert_size = w13_bytes + s13_bytes + w2_bytes + s2_bytes + 8
    total_experts = NUM_LAYERS * EXPERTS_PER_LAYER
    total_bytes = HEADER_SIZE + total_experts * expert_size

    print(f"\nPer-expert: {expert_size} bytes ({expert_size/1024/1024:.2f} MiB)")
    print(f"Total: {total_bytes/1024/1024/1024:.2f} GiB ({total_experts} experts)")

    # Write expert bank
    print(f"\nWriting to {TMP_OUTPUT}...")
    t0 = time.perf_counter()

    with open(TMP_OUTPUT, "wb") as out:
        out.write(struct.pack(HEADER_FMT, MAGIC, NUM_LAYERS, EXPERTS_PER_LAYER,
                              K, I, 0, w13_bytes, s13_bytes, w2_bytes, s2_bytes))

        for layer in range(NUM_LAYERS):
            lt0 = time.perf_counter()
            for exp in range(EXPERTS_PER_LAYER):
                p = f"model.language_model.layers.{layer}.mlp.experts.{exp}"

                g = tensors[f"{p}.gate_proj.weight_packed"]
                u = tensors[f"{p}.up_proj.weight_packed"]
                gs = tensors[f"{p}.gate_proj.weight_scale"]
                us = tensors[f"{p}.up_proj.weight_scale"]
                d = tensors[f"{p}.down_proj.weight_packed"]
                ds = tensors[f"{p}.down_proj.weight_scale"]
                gg = tensors[f"{p}.gate_proj.weight_global_scale"]
                dg = tensors[f"{p}.down_proj.weight_global_scale"]

                # Assert gate/up globals match (verified for all 8192 experts)
                ug = tensors[f"{p}.up_proj.weight_global_scale"]
                assert gg.item() == ug.item(), f"gate/up global mismatch at layer={layer} exp={exp}"

                w13 = torch.cat([g, u], dim=0).contiguous()
                w13_scales = torch.cat([gs, us], dim=0).contiguous()
                # Kernel's *4194304 cancels E2M1 decode (2^-14) + E4M3 decode (2^-8)
                # So param is the raw dequant multiplier from compressed-tensors:
                # weight = e2m1 * (block_scale / global_scale)  →  param = 1/global_scale
                w13_global = 1.0 / gg.item()
                w2_global = 1.0 / dg.item()

                record_start = out.tell()
                out.write(w13.numpy().tobytes())
                out.write(w13_scales.view(torch.uint8).numpy().tobytes())
                out.write(d.numpy().tobytes())
                out.write(ds.view(torch.uint8).numpy().tobytes())
                out.write(struct.pack("<ff", w13_global, w2_global))
                assert out.tell() - record_start == expert_size
            elapsed = time.perf_counter() - lt0
            pct = (layer + 1) / NUM_LAYERS * 100
            print(f"  Layer {layer:2d}/{NUM_LAYERS} ({pct:5.1f}%) — {elapsed:.1f}s")

    total_elapsed = time.perf_counter() - t0
    actual_size = TMP_OUTPUT.stat().st_size

    if actual_size != total_bytes:
        print(f"FATAL: Size mismatch: {actual_size} vs {total_bytes}")
        TMP_OUTPUT.unlink()
        sys.exit(1)

    # Atomic rename
    os.replace(TMP_OUTPUT, OUTPUT)
    print(f"\nDone in {total_elapsed:.1f}s")
    print(f"Output: {OUTPUT} ({actual_size/1024/1024/1024:.2f} GiB)")
    print("Header validated, atomic rename complete.")


if __name__ == "__main__":
    main()
