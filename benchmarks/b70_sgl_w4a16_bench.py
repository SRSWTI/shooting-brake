"""Arm B of the B70 int4 GEMV bandwidth audit: sgl-kernel-xpu W4A16 grouped MoE.

Times sgl_kernel.fused_experts(use_int4_w4a16=True) -- prepare + scatter +
grouped GEMM1 + silu_and_mul + grouped GEMM2 + combine -- at our production
remote-expert geometry (E=126, hidden=3072, intermediate=1024, g128, symmetric
int4), against the same weight-bytes accounting as the Arm A run of
quixicore's int4_moe_split (benchmarks/results/b70_gemv_audit/gemv_bw_audit.json).

Methodology mirrors Arm A exactly:
  * incompressible weights (random signed nibbles, random per-group scales) --
    constant-fill fixtures measured up to 111% of DRAM peak via Xe2 memory
    compression before this was fixed;
  * routing rotates across 31 disjoint 4-expert sets per call so the timed
    loop streams the resident bank instead of hitting the 18.6 MiB L2 (their
    own bench reuses one routing per window, which at M=1 nearly fits L2);
  * sustained clocks: long warmup + batched XPU-event windows, median of 5.
    Cold-process small-M numbers swing 2.6x on this card (DVFS).

Correctness-gates against the vendored torch_naive_moe reference before any
timing (quality gate before speed -- the campaign's process rule).

Run (idle B70, not the serving card):
  ZE_AFFINITY_MASK=0 .venv-xpu/bin/python benchmarks/b70_sgl_w4a16_bench.py
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

# Must be set before torch loads Level Zero.
os.environ.setdefault("ZE_AFFINITY_MASK", "0")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vendor/intel-xpu/vllm-xpu/sgl-kernel-xpu/tests"))

import sgl_kernel  # noqa: F401, E402 -- registers torch.ops.sgl_kernel
from sgl_kernel import fused_experts  # noqa: E402

# Production remote-expert geometry (srswti/axe-superveloce-88b-nvfp4a16,
# B70-resident 126 of 180 experts).
E, HIDDEN, INTER, GROUP = 126, 3072, 1024, 128
TOPK = 4  # valid routes per token == Arm A's spread_local_half fixture
EXPERT_BYTES = (
    2 * INTER * HIDDEN // 2          # w1 packed
    + HIDDEN * INTER // 2            # w2 packed
    + 2 * INTER * (HIDDEN // GROUP) * 2  # w1 scales bf16
    + HIDDEN * (INTER // GROUP) * 2      # w2 scales bf16
)
assert EXPERT_BYTES == 4_866_048, EXPERT_BYTES  # == Arm A / bank stride
MEASURED_CEILING_GBPS = 599.26  # membw kernel, incompressible, this card


def pack_int4_codes(codes: torch.Tensor) -> torch.Tensor:
    # Vendored convention (tests/test_moe_gemm.py): low nibble = even k.
    nib = torch.bitwise_and(codes.to(torch.int16), 0xF)
    return (nib[..., 0::2] | (nib[..., 1::2] << 4)).to(torch.uint8)


def build_weights(seed: int, backend: str = "sgl"):
    """Random symmetric int4 weights + random per-group bf16 scales.

    Built per-expert to bound host RAM (a full [E,N,K] int16 + float chain
    peaks >10 GB and this box serves a model). Returns device tensors plus a
    small per-expert CPU dequant closure for the correctness gate.
    """
    gen = torch.Generator().manual_seed(seed)
    w1 = torch.empty(E, 2 * INTER, HIDDEN // 2, dtype=torch.uint8)
    w2 = torch.empty(E, HIDDEN, INTER // 2, dtype=torch.uint8)
    w1_scale = (
        torch.rand(E, 2 * INTER, HIDDEN // GROUP, generator=gen) * 0.02 + 0.005
    ).to(torch.bfloat16)
    w2_scale = (
        torch.rand(E, HIDDEN, INTER // GROUP, generator=gen) * 0.02 + 0.005
    ).to(torch.bfloat16)
    codes_kept: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    keep = set(range(2 * TOPK))  # experts used by the correctness gate
    for e in range(E):
        c1 = torch.randint(-8, 8, (2 * INTER, HIDDEN), generator=gen, dtype=torch.int16)
        c2 = torch.randint(-8, 8, (HIDDEN, INTER), generator=gen, dtype=torch.int16)
        if backend == "vllm":
            # vLLM XpuFusedMoe packing: uint4 codes 0..15 with zero-point 8,
            # dequant (code - 8) * scale; the constructor folds the zp in
            # place. Same logical weights as the sgl signed-s4 encoding.
            w1[e] = pack_int4_codes(c1 + 8)
            w2[e] = pack_int4_codes(c2 + 8)
        else:
            w1[e] = pack_int4_codes(c1)
            w2[e] = pack_int4_codes(c2)
        if e in keep:
            codes_kept[e] = (c1, c2)

    def dequant(e: int):
        c1, c2 = codes_kept[e]
        d1 = c1.float() * w1_scale[e].float().repeat_interleave(GROUP, dim=1)
        d2 = c2.float() * w2_scale[e].float().repeat_interleave(GROUP, dim=1)
        return d1.to(torch.bfloat16), d2.to(torch.bfloat16)

    dev = "xpu"
    return (
        w1.to(dev), w2.to(dev),
        w1_scale.to(dev), w2_scale.to(dev),
        dequant,
    )


def naive_moe_ref(x, topk_ids, topk_weights, dequant):
    """Vendored torch_naive_moe, restricted to the touched experts, CPU fp32."""
    M = x.shape[0]
    xr = x.float()
    out = torch.zeros(M, HIDDEN)
    for m in range(M):
        for j in range(topk_ids.shape[1]):
            e = int(topk_ids[m, j])
            d1, d2 = dequant(e)
            g = xr[m] @ d1.float().t()
            act = F.silu(g[:INTER]) * g[INTER:]
            out[m] += float(topk_weights[m, j]) * (act @ d2.float().t())
    return out


def rotating_routes(n_sets: int, M: int, device):
    """Disjoint expert sets per call, Arm A's spread pattern: all pairs in one
    dispatch are distinct experts and consecutive calls touch fresh banks."""
    ids, wts = [], []
    for s in range(n_sets):
        idm = torch.empty(M, TOPK, dtype=torch.int32)
        for m in range(M):
            for j in range(TOPK):
                idm[m, j] = (s * TOPK + m * TOPK + j) % E
        ids.append(idm.to(device))
        wts.append(torch.full((M, TOPK), 1.0 / TOPK, dtype=torch.float32).to(device))
    return ids, wts


def make_run_fn(backend, w1, w2, w1_scale, w2_scale):
    if backend == "vllm":
        import vllm_xpu_kernels._moe_C  # noqa: F401
        from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe

        # Constructor folds the uint4 zero point in place -- build ONCE.
        moe = XpuFusedMoe(
            w13=w1, w13_scales=w1_scale, w13_bias=None,
            w2=w2, w2_scales=w2_scale, w2_bias=None,
            n_experts_per_token=TOPK, activation="silu", num_experts=E,
        )

        def run(x, ids, wts):
            output = torch.empty_like(x)
            moe.apply(output=output, hidden_states=x,
                      topk_weights=wts, topk_ids=ids)
            return output

        return run

    def run(x, ids, wts):
        return fused_experts(
            x, w1, w2, wts, ids,
            use_int4_w4a16=True,
            w1_scale=w1_scale, w2_scale=w2_scale,
            activation="silu",
        )

    return run


def run_cell(M, run_fn, warmup, windows, inner):
    device = "xpu"
    ids, wts = rotating_routes(31, M, device)
    x = (torch.randn(M, HIDDEN) * 0.025).to(torch.bfloat16).to(device)

    call_i = 0

    def once():
        nonlocal call_i
        r = call_i % len(ids)
        call_i += 1
        return run_fn(x, ids[r], wts[r])

    for _ in range(warmup):
        once()
    torch.xpu.synchronize()

    samples = []
    for _ in range(windows):
        e0 = torch.xpu.Event(enable_timing=True)
        e1 = torch.xpu.Event(enable_timing=True)
        e0.record()
        for _ in range(inner):
            once()
        e1.record()
        torch.xpu.synchronize()
        samples.append(e0.elapsed_time(e1) / inner)  # ms/call
    samples.sort()
    med = samples[len(samples) // 2]
    pairs = M * TOPK
    weight_bytes = pairs * EXPERT_BYTES
    gbps = weight_bytes / (med * 1e-3) / 1e9
    return {
        "M": M, "valid_routes": pairs, "median_ms": med,
        "min_ms": samples[0], "max_ms": samples[-1],
        "weight_mib": weight_bytes / 2**20, "weight_gbps": gbps,
        "pct_of_measured_ceiling": round(gbps / MEASURED_CEILING_GBPS * 100, 1),
        "windows": windows, "inner": inner, "warmup": warmup,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["sgl", "vllm"], default="sgl")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--tokens", type=int, nargs="*", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--warmup", type=int, default=300)
    args = ap.parse_args()
    if args.json_out is None:
        args.json_out = str(ROOT / "benchmarks/results/b70_gemv_audit" /
                            ("sgl_w4a16.json" if args.backend == "sgl"
                             else "vllm_xpu_w4a16.json"))

    assert torch.xpu.is_available()
    dev_name = torch.xpu.get_device_name(0)
    print(f"device: {dev_name}  ZE_AFFINITY_MASK={os.environ.get('ZE_AFFINITY_MASK')}")

    w1, w2, w1_scale, w2_scale, dequant = build_weights(seed=0, backend=args.backend)
    run_fn = make_run_fn(args.backend, w1, w2, w1_scale, w2_scale)

    # -- correctness gate (before any timing) ------------------------------
    Mg = 2
    gids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32)
    gwts = torch.tensor([[0.4, 0.3, 0.2, 0.1], [0.25, 0.25, 0.25, 0.25]])
    xg = (torch.randn(Mg, HIDDEN, generator=torch.Generator().manual_seed(7)) * 0.05)
    got = run_fn(
        xg.to(torch.bfloat16).to("xpu"),
        gids.to("xpu"),
        gwts.to(torch.float32).to("xpu"),
    ).cpu().float()
    ref = naive_moe_ref(xg, gids, gwts, dequant)
    rel = (got - ref).norm() / ref.norm()
    cos = F.cosine_similarity(got.flatten(), ref.flatten(), dim=0)
    print(f"correctness: rel_l2={rel:.4e} cosine={cos:.6f}")
    assert cos > 0.999, f"{args.backend} w4a16 output does not match the naive reference"

    results = {
        "device": dev_name,
        "backend": args.backend,
        "geometry": {"E": E, "hidden": HIDDEN, "intermediate": INTER,
                     "group": GROUP, "topk_valid": TOPK,
                     "expert_bytes": EXPERT_BYTES},
        "measured_ceiling_gbps": MEASURED_CEILING_GBPS,
        "correctness": {"rel_l2": float(rel), "cosine": float(cos)},
        "cells": [],
    }
    for M in args.tokens:
        inner = 400 if M <= 4 else 100
        cell = run_cell(M, run_fn, args.warmup, 5, inner)
        results["cells"].append(cell)
        print(f"M={M:3d}  {cell['median_ms']*1e3:8.1f} us  "
              f"{cell['weight_gbps']:6.1f} GB/s  "
              f"{cell['pct_of_measured_ceiling']:5.1f}% of ceiling")

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print("saved:", out)


if __name__ == "__main__":
    main()
