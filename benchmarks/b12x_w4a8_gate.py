#!/usr/bin/env python3
"""Kernel round 2, quality gate: b12x w4a8_nvfp4 against two oracles.

Three arms per layer, each scored with per-row cosine + relative L2 against
the same routed inputs:

  A  b12x w4a8_nvfp4 kernel (GPU; skipped under --cpu-only)
  B  b12x's own w4a8 reference (`moe_reference_w4a8_mx`) fed from OUR bank v2
     planes -- weights, MMA-swizzled scales, ratio-baked alphas
  C  exact fp32 oracle: checkpoint dequant (`dequantize_nvfp4`, the repo's
     validated reader) + fp32 SwiGLU MoE

  A-vs-B isolates our wiring (their in-repo gate for this is cos > 0.998).
  B-vs-C prices w4a8's intrinsic quality ceiling per layer -- pure torch,
         CPU-runnable while the GPU serves; this is the number the W4A4
         bake-off never had until too late.
  A-vs-C is the total serving delta, the one the firsttok gate cares about.

Layer choice matters: the residual-underflow probe
(`benchmarks/results/run6_final/w4a8_residual_probe.json`) found down_proj
flush on layers 0 (6.2%), 1 (4.3%), 41 (0.4%); everything else is exact.
Default layers cover both populations.

Usage:
  .venv/bin/python benchmarks/b12x_w4a8_gate.py --cpu-only \
      --bank src/phase1/expert_bank_int4.bin.b12x --m 256 --layers 0,24
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "b12x"))

from b12x_bank_poc import (  # noqa: E402  (repo bench module, not a package)
    load_bank_layer,
    reference_expert_mats,
)

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src" / "phase4" / "src")
)
from shooting_brake_vllm.b12x_bank_format import (  # noqa: E402
    default_b12x_bank_path,
    read_b12x_bank_header,
)

TOP_K = 8


def routed_inputs(m: int, experts: int, hidden: int, seed: int, device: str):
    """RMSNorm-scale activations (rms ~= 1) and a uniform expert routing."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(m, hidden, generator=g, dtype=torch.float32)
    ids = torch.stack(
        [torch.randperm(experts, generator=g)[:TOP_K] for _ in range(m)]
    ).to(torch.int32)
    w = torch.rand(m, TOP_K, generator=g, dtype=torch.float32)
    w = w / w.sum(-1, keepdim=True)
    return x.to(device), ids.to(device), w.to(device)


def scores(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a, b = a.float().flatten(), b.float().flatten()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    rel = ((a - b).norm() / b.norm().clamp_min(1e-12)).item()
    return {"cosine": round(cos, 6), "rel_l2": round(rel, 6)}


# -- swizzle adapters ------------------------------------------------------
# Bank v2 stores sf planes in flashinfer's MMA emit order (E LAST); b12x
# expects the TRT-LLM arrangement [E, rp/128, c/4, 32, 4, 4] flattened.
# Both verified byte-exact round-trip against _unswizzle_block_scales_batched
# (NaN caveat: compare bytes, never e4m3 floats -- 0x7F/0xFF are NaN).
_INV_MAP_CACHE: dict[tuple[int, int, int], torch.Tensor] = {}


def _converter_inverse_map(e: int, rows: int, cols: int) -> torch.Tensor:
    key = (e, rows, cols)
    if key not in _INV_MAP_CACHE:
        from vllm.utils.flashinfer import (
            flashinfer_convert_sf_to_mma_layout as sf_convert,
        )
        idx = torch.arange(e * rows * cols, dtype=torch.int32).reshape(
            e * rows, cols)
        _INV_MAP_CACHE[key] = sf_convert(
            idx, m=rows, k=cols * 16, num_groups=e
        ).reshape(-1).to(torch.long)
    return _INV_MAP_CACHE[key]


def bank_sf_to_b12x(plane: torch.Tensor, e: int, rows: int, cols: int,
                    swap_halves: bool = False) -> torch.Tensor:
    """flashinfer-swizzled sf plane -> b12x-layout flat bytes [E, -1].

    swap_halves reorders logical rows [gate;up] -> [up;gate]: bank v2 stacks
    gate first (b12x "w31"); the kernel's safe layout is up-first ("w13") --
    its w31 handling is the in-place-swap hazard, so production pre-swaps.
    """
    inv = _converter_inverse_map(e, rows, cols)
    logical = torch.empty(e * rows * cols, dtype=torch.uint8)
    logical[inv] = plane.reshape(-1).cpu().view(torch.uint8)
    logical = logical.reshape(e, rows, cols)
    if swap_halves:
        half = rows // 2
        logical = torch.cat([logical[:, half:], logical[:, :half]], dim=1)
    rp, cp = (rows + 127) // 128 * 128, (cols + 3) // 4 * 4
    pad = torch.zeros(e, rp, cp, dtype=torch.uint8)
    pad[:, :rows, :cols] = logical
    s = pad.reshape(e, rp // 128, 4, 32, cp // 4, 4)
    return s.permute(0, 1, 4, 3, 2, 5).contiguous().reshape(e, -1).to(
        plane.device)


def arm_b_their_oracle(v: dict, x, ids, w, hidden: int, inter: int):
    """moe_reference_w4a8_mx on our bank planes (their w4a8 approximation)."""
    from b12x.moe.fused_moe._impl import _derive_w4a8_weight_grids
    from b12x.moe._shared.kernels.reference import moe_reference_w4a8_mx

    e = v["alpha1"].shape[0]
    sf1_b = bank_sf_to_b12x(v["sf1"], e, 2 * inter, hidden // 16)
    sf2_b = bank_sf_to_b12x(v["sf2"], e, hidden, inter // 16)
    w13_mx, w13_res = _derive_w4a8_weight_grids(sf1_b, 2 * inter, hidden)
    w2_mx, w2_res = _derive_w4a8_weight_grids(sf2_b, hidden, inter)
    return moe_reference_w4a8_mx(
        x.float(),
        v["w1"].view(torch.uint8), w13_mx,
        w13_res.view(torch.float8_e4m3fn), v["alpha1"].float(),
        v["w2"].view(torch.uint8), w2_mx,
        w2_res.view(torch.float8_e4m3fn), v["alpha2"].float(),
        ids, w.float(), e, hidden, inter,
        activation="silu",
        # Bank v2 stacks gate rows 0..I-1, up rows I..2I-1 -- b12x calls that
        # order "w31" (their default "w13" is up-first; passing the wrong one
        # computes silu(up)*gate and scored 0.87 cosine on layer 24).
        # PRODUCTION NOTE: the kernel path's w31 handling is the in-place
        # row-swap hazard (_W13_NORMALIZED_STORAGES) -- bank v3 must emit
        # up-first planes so serving runs the safe "w13" layout.
        w13_layout="w31",
    )


def arm_c_exact(model_dir: Path, layer: int, x, ids, w,
                hidden: int, inter: int, expert_id_base: int):
    """fp32 SwiGLU MoE over the checkpoint's exact dequantized weights."""
    used = sorted(set(ids.flatten().tolist()))
    mats = {
        s: reference_expert_mats(
            model_dir, layer, s + expert_id_base, hidden, inter
        )
        for s in used
    }
    xf, wf = x.float().cpu(), w.float().cpu()
    idc = ids.cpu()
    y = torch.zeros(x.shape[0], hidden, dtype=torch.float32)
    for tok in range(x.shape[0]):
        for j in range(TOP_K):
            wg, wu, wd = mats[int(idc[tok, j])]
            act = torch.nn.functional.silu(xf[tok] @ wg) * (xf[tok] @ wu)
            y[tok] += wf[tok, j] * (act @ wd)
    return y


def arm_a_kernel(v: dict, x, ids, w, hidden: int, inter: int):
    """b12x w4a8_nvfp4 through the serving prepare/run path (GPU)."""
    sys.path.insert(
        0, str(Path(__file__).resolve().parents[1] / "vendor" / "b12x")
    )
    from tests._reference.helpers import (
        prepare_tp_moe_fp4_experts,
        run_tp_moe_fp4,
    )
    from b12x.moe.fused_moe._impl import clear_tp_moe_caches

    clear_tp_moe_caches()
    e = v["alpha1"].shape[0]
    ones = torch.ones(e, device=x.device, dtype=torch.float32)
    xb = x.to(torch.bfloat16)
    half = inter
    w1_up_first = torch.cat(
        [v["w1"][:, half:], v["w1"][:, :half]], dim=1
    ).contiguous()
    experts = prepare_tp_moe_fp4_experts(
        a=xb,
        a1_gscale=ones,           # our bank carries no input_scale; alpha is
        w1_fp4=w1_up_first,       # the full weight-side multiplier already
        w1_blockscale=bank_sf_to_b12x(
            v["sf1"], e, 2 * inter, hidden // 16, swap_halves=True),
        w1_alphas=v["alpha1"].float(),
        a2_gscale=ones,
        w2_fp4=v["w2"],
        w2_blockscale=bank_sf_to_b12x(
            v["sf2"], e, hidden, inter // 16),
        w2_alphas=v["alpha2"].float(),
        quant_mode="w4a8_nvfp4",
    )
    out = run_tp_moe_fp4(
        a=xb,
        experts=experts,
        topk_weights=w,
        topk_ids=ids,
        input_scales_static=True,
        quant_mode="w4a8_nvfp4",
    )
    torch.cuda.synchronize()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", type=Path,
                    default=Path(default_b12x_bank_path(
                        "src/phase1/expert_bank_int4.bin")))
    ap.add_argument("--nvfp4-model-dir", type=Path, default=None)
    ap.add_argument("--layers", default="0,1,24,41,47",
                    help="comma list; 0/1/41 are the residual-flush layers")
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu-only", action="store_true",
                    help="skip arm A (kernel); B-vs-C runs on CPU")
    ap.add_argument("--out", type=Path,
                    default=Path("benchmarks/results/run6_final/w4a8_gate.json"))
    args = ap.parse_args()

    if args.nvfp4_model_dir is None:
        from b12x_bank_poc import DEFAULT_NVFP4_MODEL_DIR
        args.nvfp4_model_dir = DEFAULT_NVFP4_MODEL_DIR

    device = "cpu" if args.cpu_only else "cuda"
    h = read_b12x_bank_header(args.bank)
    hidden, inter = h.hidden, h.moe_intermediate
    result: dict[str, object] = {
        "kind": "w4a8_quality_gate",
        "bank": str(args.bank),
        "m": args.m,
        "device": device,
        "layers": {},
    }

    for layer in (int(s) for s in args.layers.split(",")):
        v = load_bank_layer(args.bank, h, layer, device)
        e = v["alpha1"].shape[0]
        x, ids, w = routed_inputs(args.m, e, hidden, args.seed + layer, device)

        t0 = time.perf_counter()
        y_b = arm_b_their_oracle(v, x, ids, w, hidden, inter)
        y_c = arm_c_exact(args.nvfp4_model_dir, layer, x, ids, w,
                          hidden, inter, h.expert_id_base)
        row: dict[str, object] = {
            "B_vs_C_intrinsic_w4a8": scores(y_b.cpu(), y_c),
            "oracle_seconds": round(time.perf_counter() - t0, 1),
        }
        if not args.cpu_only:
            y_a = arm_a_kernel(v, x, ids, w, hidden, inter)
            row["A_vs_B_wiring"] = scores(y_a.cpu(), y_b.cpu())
            row["A_vs_C_total"] = scores(y_a.cpu(), y_c)
        result["layers"][str(layer)] = row
        print(f"layer {layer}: {json.dumps(row)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
