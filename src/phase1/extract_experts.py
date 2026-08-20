#!/usr/bin/env python3
"""Shooting Brake Phase 1 — build the B70 expert bank from a checkpoint.

The B70 provider does not read HF checkpoints. It mmaps a bank file in its
own NVFP4 layout, so every model needs one built once, offline.

Shape is *discovered*, not declared. The dimensions come from the model's
own config.json and the NVFP4 layer set is probed from the shards, because
a Qwen MoE checkpoint quantizes a tail of layers to FP8 (8 layers on the
35B, 1 on the 122B) and those stay on CUDA. Hard-coding either is how a
bank silently ends up describing a different model than the one loaded.

Memory is bounded to one layer. The previous version read every needed
tensor into a dict before writing, which costs the size of the whole bank:
fine at 13.5 GiB (35B), fatal at 59.5 GiB (122B) against 54 GiB of RAM.
Layers are independent records, so streaming one at a time caps residency
at experts x per-expert bytes regardless of model size.

Writes to .tmp and renames atomically, so an interrupted run never leaves a
truncated bank that would load and produce garbage.

Usage:
  python src/phase1/extract_experts.py                     # the qualified 35B
  python src/phase1/extract_experts.py --model-dir DIR --out FILE
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

DEFAULT_MODEL_DIR = (
    Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    / "hub"
    / "models--unsloth--Qwen3.6-35B-A3B-NVFP4"
    / "snapshots"
    / "739af1e7aac320af1682ed1e0cce369af4c5265d"
)

HEADER_FMT = "<8sIIIIIQQQQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
MAGIC = b"SBEXP001"

#: Per-expert trailer: the two fp32 dequant multipliers, w13 then w2.
GSCALE_FMT = "<ff"
GSCALE_BYTES = struct.calcsize(GSCALE_FMT)

EXPERT_KEY = re.compile(
    r"^(?P<root>.+)\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\."
)


class BankShape:
    """Dimensions and layer set for one checkpoint's expert bank."""

    def __init__(
        self,
        hidden: int,
        intermediate: int,
        experts: int,
        nvfp4_layers: list[int],
        key_root: str,
    ) -> None:
        self.hidden = hidden
        self.intermediate = intermediate
        self.experts = experts
        self.nvfp4_layers = nvfp4_layers
        self.key_root = key_root

        # NVFP4 packs two 4-bit values per byte and one scale per 16
        # elements, so both dimensions must divide evenly or the record
        # layout silently truncates.
        if hidden % 16 or intermediate % 16:
            raise SystemExit(
                f"hidden={hidden} and intermediate={intermediate} must both "
                "be multiples of 16 for the NVFP4 block layout"
            )

        self.w13_bytes = 2 * intermediate * (hidden // 2)
        self.s13_bytes = 2 * intermediate * (hidden // 16)
        self.w2_bytes = hidden * (intermediate // 2)
        self.s2_bytes = hidden * (intermediate // 16)
        self.expert_bytes = (
            self.w13_bytes + self.s13_bytes
            + self.w2_bytes + self.s2_bytes + GSCALE_BYTES
        )

    @property
    def num_layers(self) -> int:
        return len(self.nvfp4_layers)

    @property
    def total_bytes(self) -> int:
        return HEADER_SIZE + self.num_layers * self.experts * self.expert_bytes

    def header(self) -> bytes:
        return struct.pack(
            HEADER_FMT, MAGIC, self.num_layers, self.experts,
            self.hidden, self.intermediate, 0,
            self.w13_bytes, self.s13_bytes, self.w2_bytes, self.s2_bytes,
        )

    def describe(self) -> str:
        return (
            f"  layers      {self.num_layers} NVFP4 "
            f"({self.nvfp4_layers[0]}..{self.nvfp4_layers[-1]})\n"
            f"  bank rows   0..{self.num_layers - 1} -> model layers "
            f"{self.nvfp4_layers}\n"
            f"  key root    {self.key_root}\n"
            f"  experts     {self.experts}/layer\n"
            f"  hidden      {self.hidden}\n"
            f"  intermediate{self.intermediate:>5}\n"
            f"  per expert  {self.expert_bytes/2**20:.2f} MiB\n"
            f"  bank total  {self.total_bytes/2**30:.2f} GiB"
        )


def read_config(model_dir: Path) -> dict:
    cfg = json.loads((model_dir / "config.json").read_text())
    # Multimodal Qwen configs nest the language model's dimensions.
    return cfg.get("text_config", cfg)


def index_shards(model_dir: Path) -> tuple[list[str], dict[str, str]]:
    """Map every tensor key to the shard holding it.

    MTP shards are excluded: the speculative-decode head carries its own
    expert tensors under the same naming, and folding them into the bank
    would append a phantom layer.
    """
    shards = sorted(
        p for p in glob.glob(f"{model_dir}/model*.safetensors")
        if "mtp" not in Path(p).name.lower()
    )
    if not shards:
        raise SystemExit(f"no safetensors shards under {model_dir}")
    index: dict[str, str] = {}
    for path in shards:
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                index[key] = path
    return shards, index


def discover_shape(model_dir: Path, index: dict[str, str]) -> BankShape:
    """Derive bank dimensions from the config and the shard key set."""
    cfg = read_config(model_dir)
    hidden = int(cfg["hidden_size"])
    intermediate = int(cfg["moe_intermediate_size"])
    experts = int(cfg["num_experts"])

    # An NVFP4 expert layer is one whose experts carry `weight_packed`.
    # FP8 layers carry a plain `weight` instead and are CUDA-only.
    packed: set[int] = set()
    seen: set[int] = set()
    roots: set[str] = set()
    for key in index:
        m = EXPERT_KEY.match(key)
        if not m:
            continue
        roots.add(m.group("root"))
        layer = int(m.group("layer"))
        seen.add(layer)
        if key.endswith(".weight_packed"):
            packed.add(layer)

    nvfp4 = sorted(packed)
    if not nvfp4:
        raise SystemExit("no NVFP4 expert layers found; is this an FP8 checkpoint?")
    if len(roots) != 1:
        raise SystemExit(
            f"expected one expert tensor key root, found {sorted(roots)}"
        )
    key_root = next(iter(roots))

    # The provider addresses compact bank rows directly. Source layers may
    # start above zero when the absolute ids are recorded externally, but a
    # gap would still shift every layer above it onto the wrong weights.
    expected = list(range(nvfp4[0], nvfp4[-1] + 1))
    if nvfp4 != expected:
        missing = sorted(set(expected) - packed)
        raise SystemExit(
            "NVFP4 layers contain gaps; compact bank rows require one "
            f"contiguous source-layer run: layers={nvfp4}, missing={missing}"
        )
    fp8 = sorted(seen - packed)
    print(f"Discovered {len(nvfp4)} NVFP4 expert layers; {len(fp8)} FP8 "
          f"(CUDA-only): {fp8 if len(fp8) <= 12 else f'{fp8[:6]}...'}")
    if nvfp4[0] != 0:
        print(
            f"Compact source run starts at model layer {nvfp4[0]}; "
            "bank row 0 uses that layer (not model layer 0)"
        )
    return BankShape(hidden, intermediate, experts, nvfp4, key_root)


def load_layer(
    layer: int,
    shape: BankShape,
    index: dict[str, str],
) -> dict[str, torch.Tensor]:
    """Read one layer's expert tensors, grouped by shard to open each once."""
    wanted: dict[str, list[str]] = defaultdict(list)
    for exp in range(shape.experts):
        prefix = f"{shape.key_root}.layers.{layer}.mlp.experts.{exp}"
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for suffix in ("weight_packed", "weight_scale", "weight_global_scale"):
                key = f"{prefix}.{proj}.{suffix}"
                shard = index.get(key)
                if shard is None:
                    raise SystemExit(f"missing tensor: {key}")
                wanted[shard].append(key)

    tensors: dict[str, torch.Tensor] = {}
    for shard, keys in wanted.items():
        with safe_open(shard, framework="pt") as f:
            for key in keys:
                tensors[key] = f.get_tensor(key)
    return tensors


def write_layer(
    out,
    layer: int,
    shape: BankShape,
    tensors: dict[str, torch.Tensor],
) -> None:
    for exp in range(shape.experts):
        p = f"{shape.key_root}.layers.{layer}.mlp.experts.{exp}"
        gate = tensors[f"{p}.gate_proj.weight_packed"]
        up = tensors[f"{p}.up_proj.weight_packed"]
        gate_s = tensors[f"{p}.gate_proj.weight_scale"]
        up_s = tensors[f"{p}.up_proj.weight_scale"]
        down = tensors[f"{p}.down_proj.weight_packed"]
        down_s = tensors[f"{p}.down_proj.weight_scale"]
        gate_g = tensors[f"{p}.gate_proj.weight_global_scale"]
        up_g = tensors[f"{p}.up_proj.weight_global_scale"]
        down_g = tensors[f"{p}.down_proj.weight_global_scale"]

        # w13 is stored fused with a single global scale, which is only
        # meaningful if both halves share one.
        if gate_g.item() != up_g.item():
            raise SystemExit(
                f"gate/up global scale mismatch at layer={layer} expert={exp}: "
                f"{gate_g.item()} vs {up_g.item()}; the fused w13 record "
                "cannot represent two different global scales"
            )

        # Bank convention is [gate, up]. vLLM's FlashInfer path reorders its
        # own copy to [up, gate] after loading; consumers that read this
        # file must not assume the VRAM ordering.
        w13 = torch.cat([gate, up], dim=0).contiguous()
        w13_scales = torch.cat([gate_s, up_s], dim=0).contiguous()

        # The kernel's *4194304 cancels E2M1 decode (2^-14) and E4M3 decode
        # (2^-8), so the stored parameter is the raw dequant multiplier:
        # weight = e2m1 * (block_scale / global_scale) -> param = 1/global.
        start = out.tell()
        out.write(w13.numpy().tobytes())
        out.write(w13_scales.view(torch.uint8).numpy().tobytes())
        out.write(down.numpy().tobytes())
        out.write(down_s.view(torch.uint8).numpy().tobytes())
        out.write(struct.pack(
            GSCALE_FMT, 1.0 / gate_g.item(), 1.0 / down_g.item(),
        ))
        written = out.tell() - start
        if written != shape.expert_bytes:
            raise SystemExit(
                f"record size mismatch at layer={layer} expert={exp}: "
                f"wrote {written}, expected {shape.expert_bytes}"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model-dir", type=Path,
        default=Path(os.environ.get("SB_NVFP4_MODEL_DIR", DEFAULT_MODEL_DIR)),
        help="HF snapshot directory holding config.json and the shards",
    )
    ap.add_argument(
        "--out", type=Path, default=Path(__file__).parent / "expert_bank.bin",
        help="destination bank file",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="report the discovered shape and size, then stop",
    )
    args = ap.parse_args()

    if not args.model_dir.is_dir():
        raise SystemExit(f"model dir not found: {args.model_dir}")

    shards, index = index_shards(args.model_dir)
    print(f"Shards: {len(shards)}")
    shape = discover_shape(args.model_dir, index)
    print(shape.describe())

    if args.dry_run:
        return 0

    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting {tmp} ...")
    t0 = time.perf_counter()

    with open(tmp, "wb") as out:
        out.write(shape.header())
        # Compact row `position` holds absolute model layer
        # `shape.nvfp4_layers[position]`.
        for position, layer in enumerate(shape.nvfp4_layers):
            lt0 = time.perf_counter()
            tensors = load_layer(layer, shape, index)
            write_layer(out, layer, shape, tensors)
            del tensors  # bound residency to one layer
            print(
                f"  bank row {position:2d} <- model layer {layer:2d} "
                f"({(position + 1) / shape.num_layers * 100:5.1f}%) — "
                f"{time.perf_counter() - lt0:.1f}s"
            )

    actual = tmp.stat().st_size
    if actual != shape.total_bytes:
        tmp.unlink()
        raise SystemExit(
            f"FATAL size mismatch: wrote {actual}, expected {shape.total_bytes}"
        )

    os.replace(tmp, args.out)
    print(
        f"\nDone in {time.perf_counter() - t0:.1f}s — "
        f"{args.out} ({actual/2**30:.2f} GiB)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
