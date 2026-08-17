#!/usr/bin/env python3
"""B12x bank fidelity PoC: real bank planes through the real kernel vs CPU oracle.

Mirrors marlin_prefill_poc.py for bank v2. The reference is dequantized
from the CHECKPOINT (extract_experts_int4.dequantize_nvfp4 -- the repo's
validated NVFP4 reader), never from the bank, so a pass validates the
whole build chain: gate/up stacking, weight_scale_2 bake-in, MMA swizzle,
and on-disk byte layout. A swizzle or offset bug reads as cosine ~0, not
as a small delta.

Expected error model: b12x is W4A4 (in-kernel BF16->FP4 activation quant)
while the fp32 reference is exact, so the bound is LOOSER than the Marlin
PoC's W4A16 bound (0.999): cosine >= 0.99, rel_l2 <= 0.15. The serve-level
firsttok envelope gate remains the quality arbiter.

Usage (GPU must be free -- planes need ~700 MiB):
  .venv/bin/python benchmarks/b12x_bank_poc.py \
      --bank src/phase1/expert_bank_int4.bin.b12x \
      --out benchmarks/results/b12x_poc/poc.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src/phase1"))
sys.path.insert(0, str(REPO / "src/phase4/src"))

from extract_experts_int4 import (  # noqa: E402
    DEFAULT_NVFP4_MODEL_DIR,
    dequantize_nvfp4,
    index_shards,
    tensor_key,
)
from shooting_brake_vllm.b12x_bank_format import read_b12x_bank_header  # noqa: E402

TOP_K = 8


def load_bank_layer(bank: Path, h, layer: int, device: str):
    """Device tensors for one layer's planes, shaped as the streamer views them."""
    mm = np.memmap(bank, dtype=np.uint8, mode="r")
    base = h.data_offset + layer * h.layer_stride_bytes
    e, k, i = h.experts_per_layer, h.hidden, h.moe_intermediate

    def plane(idx: int, dtype: torch.dtype, shape: tuple[int, ...]):
        off, size = h.plane_offsets[idx], h.plane_sizes[idx]
        t = torch.from_numpy(np.asarray(mm[base + off: base + off + size]).copy())
        return t.to(device).view(dtype).view(*shape)

    return {
        "w1": plane(0, torch.uint8, (e, 2 * i, k // 2)),
        "w2": plane(1, torch.uint8, (e, k, i // 2)),
        "sf1": plane(2, torch.float8_e4m3fn, h.sf1_shape),
        "sf2": plane(3, torch.float8_e4m3fn, h.sf2_shape),
        "alpha1": plane(4, torch.float32, (e,)),
        "alpha2": plane(5, torch.float32, (e,)),
    }


def reference_expert_mats(model_dir: Path, layer: int, expert: int,
                          hidden: int, intermediate: int):
    """(W_gate, W_up, W_down) fp32 from the checkpoint, each [in, out]
    (dequantize_nvfp4 returns the transposed logical [K, N] layout)."""
    from safetensors import safe_open
    _, index = index_shards(model_dir)
    handles: dict[str, object] = {}

    def t(proj: str, suffix: str) -> torch.Tensor:
        key = tensor_key(layer, expert, proj, suffix)
        shard = index[key]
        hd = handles.get(shard)
        if hd is None:
            hd = safe_open(model_dir / shard, framework="pt", device="cpu")
            handles[shard] = hd
        return hd.get_tensor(key)

    def dq(proj: str, k: int, n: int) -> torch.Tensor:
        return dequantize_nvfp4(
            t(proj, "weight"), t(proj, "weight_scale"),
            t(proj, "weight_scale_2"), k, n,
        ).float()

    # gate/up: packed [I, H/2] -> k=hidden, n=intermediate -> [H, I]
    # down:    packed [H, I/2] -> k=intermediate, n=hidden -> [I, H]
    return (dq("gate_proj", hidden, intermediate),
            dq("up_proj", hidden, intermediate),
            dq("down_proj", intermediate, hidden))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--nvfp4-model-dir", type=Path, default=DEFAULT_NVFP4_MODEL_DIR)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--m", type=int, default=48)
    ap.add_argument("--experts", type=int, default=16,
                    help="sampled expert slots for the oracle")
    ap.add_argument("--bench-m", type=int, default=8192)
    ap.add_argument("--iters", type=int, default=7)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sf-mode", choices=("bank", "direct"), default="bank",
                    help="bank: contiguous swizzled bytes from the bank file; "
                         "direct: recompute scale factors from the checkpoint "
                         "and pass the converter's raw (non-contiguous) view "
                         "-- discriminates whether the kernel is strides-aware "
                         "or reads the converter's underlying linear storage")
    args = ap.parse_args()

    assert torch.cuda.is_available()
    dev = "cuda"
    h = read_b12x_bank_header(args.bank)
    e, k, i = h.experts_per_layer, h.hidden, h.moe_intermediate
    v = load_bank_layer(args.bank, h, args.layer, dev)
    if args.sf_mode == "direct":
        # The bank stores the swizzled scales MATERIALIZED (contiguous in
        # logical order). The converter's native output is a non-contiguous
        # VIEW over linear storage. Reconstruct that native form by copying
        # the bank's logical values into a fresh converter view: the copy
        # writes the underlying linear buffer, giving the exact tensor the
        # converter would have returned.
        from vllm.utils.flashinfer import (
            flashinfer_convert_sf_to_mma_layout as _conv,
        )

        def to_direct(sf_bank: torch.Tensor, m_: int, k_: int):
            lin = torch.empty(e * m_ * (k_ // 16), dtype=torch.float8_e4m3fn,
                              device=dev)
            view = _conv(lin.reshape(e * m_, k_ // 16), m=m_, k=k_,
                         num_groups=e)
            view.copy_(sf_bank)
            return view

        v["sf1"] = to_direct(v["sf1"], 2 * i, k)
        v["sf2"] = to_direct(v["sf2"], k, i)

    from flashinfer.fused_moe import B12xMoEWrapper
    wrapper = B12xMoEWrapper(
        num_experts=e, top_k=TOP_K, hidden_size=k, intermediate_size=i,
        use_cuda_graph=True, max_num_tokens=max(args.m, args.bench_m),
        num_local_experts=e, activation="silu",
    )
    ones = torch.ones(e, device=dev, dtype=torch.float32)

    def run(x, ids, w):
        # Validated wiring per flashinfer's own
        # test_input_global_scale_decoupled_weight_alpha: exact scale_2
        # rides in w1_alpha/w2_alpha (fp32); block scales carry only the
        # gate/up RATIO bake; input_global_scale=1 puts FC1 input on the
        # raw e2m1 grid (correct for rms~1 RMSNorm'd hidden states);
        # fc2 input quant is per-block dynamic with unit per-expert scale.
        return wrapper.run(
            x=x, w1_weight=v["w1"], w1_weight_sf=v["sf1"],
            w1_alpha=v["alpha1"], w2_alpha=v["alpha2"],
            input_global_scale=ones, fc2_input_scale=ones,
            w2_weight=v["w2"], w2_weight_sf=v["sf2"],
            token_selected_experts=ids, token_final_scales=w,
        )

    # -- correctness: m tokens routed among a sampled expert subset ----------
    torch.manual_seed(42)
    slots = torch.randperm(e)[: args.experts].tolist()
    # RMSNorm-scale activations (rms ~= 1): FC1 input quant is a STATIC
    # e2m1 grid at x*gs (b12x_zero_bisect.py measured it) -- values well
    # below 0.25 all round to zero, so an unphysically small test vector
    # zeroes the kernel while real hidden states quantize fine.
    x = torch.randn(args.m, k, dtype=torch.float32)
    ids = torch.stack([
        torch.tensor(np.random.default_rng(s).choice(slots, TOP_K, replace=False))
        for s in range(args.m)
    ]).to(torch.int32)
    w = torch.rand(args.m, TOP_K, dtype=torch.float32)
    w = w / w.sum(-1, keepdim=True)

    y_kernel = run(x.to(dev, torch.bfloat16), ids.to(dev), w.to(dev)).float().cpu()

    mats = {s: reference_expert_mats(args.nvfp4_model_dir, args.layer,
                                     s + h.expert_id_base, k, i)
            for s in sorted(set(ids.flatten().tolist()))}
    y_ref = torch.zeros(args.m, k, dtype=torch.float32)
    for tok in range(args.m):
        xt = x[tok]
        for j in range(TOP_K):
            wg, wu, wd = mats[int(ids[tok, j])]
            act = torch.nn.functional.silu(xt @ wg) * (xt @ wu)
            y_ref[tok] += float(w[tok, j]) * (act @ wd)

    cos = float(torch.nn.functional.cosine_similarity(
        y_kernel.flatten(), y_ref.flatten(), dim=0))
    rel_l2 = float((y_kernel - y_ref).norm() / y_ref.norm())

    # -- speed with REAL weights (synthetic-arm cross-check) -----------------
    xb = torch.randn(args.bench_m, k, device=dev, dtype=torch.bfloat16)
    idb = torch.randint(0, e, (args.bench_m, TOP_K), device=dev, dtype=torch.int32)
    wb = torch.rand(args.bench_m, TOP_K, device=dev, dtype=torch.float32)
    run(xb, idb, wb); torch.cuda.synchronize()
    ev_a, ev_b = torch.cuda.Event(True), torch.cuda.Event(True)
    ts = []
    for _ in range(args.iters):
        torch.cuda.synchronize()
        ev_a.record()
        run(xb, idb, wb)
        ev_b.record()
        torch.cuda.synchronize()
        ts.append(ev_a.elapsed_time(ev_b))

    result = {
        "bank": str(args.bank), "layer": args.layer,
        "correctness": {
            "m": args.m, "expert_subset": sorted(slots),
            "cosine": cos, "rel_l2": rel_l2,
            "bound_cosine": 0.99, "bound_rel_l2": 0.15,
            "pass": cos >= 0.99 and rel_l2 <= 0.15,
            "note": "W4A4 in-kernel activation quant vs exact fp32 oracle; "
                    "swizzle/offset bugs read as cosine ~0",
        },
        "bench": {
            "m": args.bench_m,
            "kernel_ms_median_real_weights": statistics.median(ts),
            "synthetic_arm_ms": 2.70,
        },
        "gpu": torch.cuda.get_device_name(0),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["correctness"], indent=2))
    print(f"kernel {statistics.median(ts):.2f} ms @ M={args.bench_m} "
          f"(synthetic arm was 2.70)")
    print(f"-> {args.out}")
    if not result["correctness"]["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
