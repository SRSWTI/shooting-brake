#!/usr/bin/env python3
"""Build the B12x expert bank (bank v2) from the native NVFP4 checkpoint.

Reads srswti/axe-superveloce-88b-nvfp4a16's per-expert
``gate_proj/up_proj/down_proj.{weight,weight_scale,weight_scale_2}``
tensors DIRECTLY (no GPTQ re-encode: this is a fidelity upgrade over the
int4 bank) and writes the SBB12X01 arena consumed by the b12x prefill
streamer / flashinfer B12xMoEWrapper:

  w1  = cat(gate, up) packed fp4     [E, 2I, H/2]  verbatim checkpoint bytes
  w2  = down packed fp4              [E, H, I/2]   verbatim checkpoint bytes
  sf1 = mma_layout(block_scale * weight_scale_2)   e4m3, baked, alpha := 1
  sf2 = mma_layout(block_scale * weight_scale_2)   e4m3, baked, alpha := 1

The bake-in is the same math vLLM's FlashInferB12xExperts does at load
(one e4m3 round-trip); doing it at build time makes the runtime a pure
memcpy. Scale conversion runs on CPU (verified). RSS is bounded per layer.

Usage:
  .venv/bin/python src/phase1/build_b12x_bank.py \
      --out src/phase1/expert_bank_int4.bin.b12x \
      --validate-layers 2
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase4/src"))

from extract_experts_int4 import (  # noqa: E402
    DEFAULT_NVFP4_MODEL_DIR,
    index_shards,
    peak_rss_mib,
    tensor_key,
)
from shooting_brake_vllm.b12x_bank_format import (  # noqa: E402
    ALIGNMENT,
    make_header,
)

EXPERT_ID_BASE = 54     # remote experts 54..179, provider resident order
REMOTE_EXPERTS = 126
NUM_LAYERS = 48
HIDDEN = 3072
INTERMEDIATE = 1024
GROUP = 16


class ShardCache:
    def __init__(self, model_dir: Path, index: dict[str, str]) -> None:
        self.model_dir = model_dir
        self.index = index
        self._handles: dict[str, object] = {}

    def get(self, key: str) -> torch.Tensor:
        shard = self.index[key]
        h = self._handles.get(shard)
        if h is None:
            h = safe_open(self.model_dir / shard, framework="pt", device="cpu")
            self._handles[shard] = h
        return h.get_tensor(key)


E4M3_MIN_SUBNORMAL = 2.0 ** -9  # 0.001953125


def load_layer_planes(
    cache: ShardCache, layer: int, sf_convert,
) -> tuple[torch.Tensor, ...]:
    """Return (w1, w2, sf1_mma, sf2_mma, alpha1, alpha2) for one layer.

    Scale design (measured constraint, benchmarks/results/b12x_poc/):
    baking weight_scale_2 (~1e-5) into e4m3 block scales flushes 86.5% of
    them to zero (products < e4m3 min subnormal 1.95e-3) -- the naive bake
    (which vLLM's FlashInferB12xExperts also does) serves zeros. Instead:

      sf1 rows carry block_scale * (proj_s2 / max(gate_s2, up_s2)) -- an
          O(1) ratio, e4m3-safe; alpha1[e] = max(gate_s2, up_s2) exact f32.
      sf2 is VERBATIM checkpoint block scales; alpha2[e] = down_s2 exact.

    The kernel applies alpha as the per-expert weight-dequant multiplier
    (pre-activation), so the factorization is exact. The runtime MUST pass
    input_global_scale=1.0 (flashinfer >= 0.6.18): without it w1_alpha
    doubles as the FC1 activation-quant scale and ~1e-5 re-zeros
    everything from the activation side.
    """
    e, h_, i_ = REMOTE_EXPERTS, HIDDEN, INTERMEDIATE
    w1 = torch.empty(e, 2 * i_, h_ // 2, dtype=torch.uint8)
    w2 = torch.empty(e, h_, i_ // 2, dtype=torch.uint8)
    s1 = torch.empty(e, 2 * i_, h_ // GROUP, dtype=torch.float32)
    s2 = torch.empty(e, h_, i_ // GROUP, dtype=torch.float32)
    alpha1 = torch.empty(e, dtype=torch.float32)
    alpha2 = torch.empty(e, dtype=torch.float32)

    for slot, expert in enumerate(range(EXPERT_ID_BASE, EXPERT_ID_BASE + e)):
        def t(proj: str, suffix: str) -> torch.Tensor:
            return cache.get(tensor_key(layer, expert, proj, suffix))

        g_w, u_w, d_w = t("gate_proj", "weight"), t("up_proj", "weight"), t("down_proj", "weight")
        assert g_w.shape == (i_, h_ // 2) and g_w.dtype == torch.uint8, g_w.shape
        assert d_w.shape == (h_, i_ // 2) and d_w.dtype == torch.uint8, d_w.shape
        w1[slot, :i_] = g_w
        w1[slot, i_:] = u_w
        w2[slot] = d_w

        g_s2 = float(t("gate_proj", "weight_scale_2"))
        u_s2 = float(t("up_proj", "weight_scale_2"))
        d_s2 = float(t("down_proj", "weight_scale_2"))
        a1 = max(g_s2, u_s2)
        alpha1[slot] = a1
        alpha2[slot] = d_s2
        s1[slot, :i_] = t("gate_proj", "weight_scale").float() * (g_s2 / a1)
        s1[slot, i_:] = t("up_proj", "weight_scale").float() * (u_s2 / a1)
        s2[slot] = t("down_proj", "weight_scale").float()

    # Build-time telemetry: ratio bake must not reintroduce the underflow.
    nz = s1 != 0
    under = float(((s1 < E4M3_MIN_SUBNORMAL) & nz).float().mean())
    if under > 0.005:
        raise SystemExit(
            f"layer {layer}: {under:.2%} of ratio-baked sf1 values below the "
            f"e4m3 subnormal floor -- scale design assumption violated"
        )

    s1_e4m3 = s1.to(torch.float8_e4m3fn)
    s2_e4m3 = s2.to(torch.float8_e4m3fn)
    sf1 = sf_convert(
        s1_e4m3.reshape(e * 2 * i_, h_ // GROUP), m=2 * i_, k=h_, num_groups=e
    ).contiguous()
    sf2 = sf_convert(
        s2_e4m3.reshape(e * h_, i_ // GROUP), m=h_, k=i_, num_groups=e
    ).contiguous()
    return w1, w2, sf1, sf2, alpha1, alpha2


def write_plane(out, tensor: torch.Tensor) -> None:
    payload = memoryview(tensor.view(torch.uint8).numpy()).cast("B")
    out.write(payload)
    pad = -len(payload) % ALIGNMENT
    if pad:
        out.write(b"\x00" * pad)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nvfp4-model-dir", type=Path, default=DEFAULT_NVFP4_MODEL_DIR)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", type=int, default=NUM_LAYERS)
    ap.add_argument("--validate-layers", type=int, default=0,
                    help="After writing, recompute N layers and bit-compare.")
    args = ap.parse_args()

    from vllm.utils.flashinfer import flashinfer_convert_sf_to_mma_layout as sf_convert

    _, index = index_shards(args.nvfp4_model_dir)
    cache = ShardCache(args.nvfp4_model_dir, index)

    # Learn the swizzled shapes once from a real conversion of layer 0.
    t0 = time.perf_counter()
    planes = load_layer_planes(cache, 0, sf_convert)
    w1, w2, sf1, sf2 = planes[:4]
    hdr = make_header(
        num_layers=args.layers, experts=REMOTE_EXPERTS, hidden=HIDDEN,
        intermediate=INTERMEDIATE, group_size=GROUP,
        expert_id_base=EXPERT_ID_BASE,
        sf1_shape=tuple(sf1.shape), sf2_shape=tuple(sf2.shape),
    )
    print(f"layer stride {hdr.layer_stride_bytes / 2**20:.1f} MiB, "
          f"bank {args.layers * hdr.layer_stride_bytes / 2**30:.1f} GiB, "
          f"sf1 {tuple(sf1.shape)}, sf2 {tuple(sf2.shape)}")

    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    with open(tmp, "wb") as out:
        out.write(hdr.to_bytes())
        for layer in range(args.layers):
            if layer > 0:
                planes = load_layer_planes(cache, layer, sf_convert)
            for plane in planes:
                write_plane(out, plane)
            print(f"layer {layer:2d} written  "
                  f"({time.perf_counter() - t0:6.1f}s, rss {peak_rss_mib():.0f} MiB)",
                  flush=True)
    os.replace(tmp, args.out)
    print(f"-> {args.out} ({args.out.stat().st_size / 2**30:.2f} GiB)")

    if args.validate_layers:
        import numpy as np
        mm = np.memmap(args.out, dtype=np.uint8, mode="r")
        for layer in range(args.validate_layers):
            planes = load_layer_planes(cache, layer, sf_convert)
            base = hdr.data_offset + layer * hdr.layer_stride_bytes
            for name, plane, off, size in zip(
                ("w1", "w2", "sf1", "sf2", "alpha1", "alpha2"), planes,
                hdr.plane_offsets, hdr.plane_sizes,
            ):
                got = np.asarray(mm[base + off: base + off + size])
                want = np.frombuffer(
                    memoryview(plane.view(torch.uint8).numpy()).cast("B"),
                    dtype=np.uint8,
                )
                if not np.array_equal(got, want):
                    raise SystemExit(f"VALIDATE FAIL layer {layer} plane {name}")
            print(f"layer {layer} validate ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
