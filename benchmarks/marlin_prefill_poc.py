#!/usr/bin/env python3
"""Proof of concept: prefill the B70-resident experts on the 5090 via Marlin.

The design under test: during prefill, stream a layer's int4 expert planes from
the mmap'd SBINT401 bank to the 5090, repack to Marlin format on-GPU, and run
vLLM's existing ``fused_marlin_moe`` -- the same Marlin machinery the 54 local
experts already use. No novel GEMM. The bank's quantization contract
(AutoGPTQ sym int4, group 128, zero_point 8) is exactly Marlin's ``uint4b8``.

Gates, in order:
  1. CORRECTNESS: fused_marlin_moe output vs an independent CPU fp32 dequant
     reference on real bank weights. Bound: rel_l2 <= 2e-2 (bf16 output),
     cosine >= 0.999. Routing is confined to a 16-expert subset so the CPU
     reference stays tractable; grouping/sorting is still exercised.
  2. SPEED at prefill shape: M=8192 rows, top-8 routing over all 126 experts.
     Reports per-layer H2D (pageable, from the real mmap), repack, and kernel
     medians, and the 48-layer projection.

Memory rule: the bank is mmap'd and sliced per layer; nothing eager-loads it.

Usage:
  .venv/bin/python benchmarks/marlin_prefill_poc.py \
      --bank src/phase1/expert_bank_int4.bin \
      --out benchmarks/results/marlin_poc/poc.json
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/phase4/src"))
from shooting_brake_vllm.int4_bank_format import read_int4_bank_header  # noqa: E402

from vllm import _custom_ops as ops  # noqa: E402
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (  # noqa: E402
    fused_marlin_moe,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (  # noqa: E402
    marlin_moe_permute_scales,
)
from vllm.scalar_type import scalar_types  # noqa: E402


def load_layer_planes(mm: np.memmap, hdr, layer: int):
    """Slice one layer's expert planes out of the mmap. Returns numpy views."""
    e = hdr.experts_per_layer
    k, i = hdr.hidden, hdr.moe_intermediate
    base = hdr.data_offset + layer * hdr.layer_stride_bytes
    shapes = {
        "gate_qweight": (k // 8, i), "gate_scales": (k // 128, i),
        "up_qweight": (k // 8, i), "up_scales": (k // 128, i),
        "down_qweight": (i // 8, k), "down_scales": (i // 128, k),
    }
    dtypes = {"qweight": np.int32, "scales": np.float16}
    out = {name: [] for name in shapes}
    from shooting_brake_vllm.int4_bank_format import PLANE_NAMES
    for slot in range(e):
        ebase = base + slot * hdr.expert_stride_bytes
        for name, off, size in zip(PLANE_NAMES, hdr.plane_offsets, hdr.plane_sizes):
            dt = dtypes["qweight" if "qweight" in name else "scales"]
            shape = shapes[name]
            n_items = shape[0] * shape[1]
            arr = np.frombuffer(
                mm, dtype=dt, count=n_items, offset=ebase + off
            ).reshape(shape)
            out[name].append(arr)
    return {name: np.stack(v) for name, v in out.items()}


def dequant_ref(qweight: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """AutoGPTQ K-major int4 -> fp32. w[k,n] = (nib - 8) * scales[k//128, n]."""
    kp, n = qweight.shape
    q = qweight.view(np.uint32)
    w = np.empty((kp * 8, n), dtype=np.float32)
    for j in range(8):
        nib = ((q >> np.uint32(4 * j)) & np.uint32(0xF)).astype(np.float32) - 8.0
        w[j::8] = 0.0  # placeholder, overwritten below
        w[np.arange(kp) * 8 + j] = nib
    ks = np.repeat(scales.astype(np.float32), 128, axis=0)
    return w * ks


def cpu_reference(planes, x: np.ndarray, topk_ids: np.ndarray,
                  topk_w: np.ndarray, experts: list[int]) -> np.ndarray:
    """fp32 SwiGLU MoE over the routed experts only."""
    deq = {}
    for eid in experts:
        g = dequant_ref(planes["gate_qweight"][eid], planes["gate_scales"][eid])
        u = dequant_ref(planes["up_qweight"][eid], planes["up_scales"][eid])
        d = dequant_ref(planes["down_qweight"][eid], planes["down_scales"][eid])
        deq[eid] = (g, u, d)
    m, k = x.shape
    y = np.zeros((m, k), dtype=np.float32)
    for r in range(m):
        for j in range(topk_ids.shape[1]):
            eid = int(topk_ids[r, j])
            g, u, d = deq[eid]
            hg = x[r] @ g
            hu = x[r] @ u
            h = (hg / (1.0 + np.exp(-hg))) * hu
            y[r] += topk_w[r, j] * (h @ d)
    return y


def to_gpu_marlin(planes, device="cuda", act_dtype=torch.bfloat16):
    """H2D raw planes, fuse w13, repack to marlin, permute scales.

    Returns (marlin tensors dict, timings dict).
    """
    t = {}
    ev = lambda: torch.cuda.Event(enable_timing=True)  # noqa: E731
    a, b = ev(), ev()

    a.record()
    gate_q = torch.from_numpy(planes["gate_qweight"]).to(device, non_blocking=True)
    up_q = torch.from_numpy(planes["up_qweight"]).to(device, non_blocking=True)
    down_q = torch.from_numpy(planes["down_qweight"]).to(device, non_blocking=True)
    gate_s = torch.from_numpy(planes["gate_scales"]).to(device, non_blocking=True)
    up_s = torch.from_numpy(planes["up_scales"]).to(device, non_blocking=True)
    down_s = torch.from_numpy(planes["down_scales"]).to(device, non_blocking=True)
    b.record(); torch.cuda.synchronize()
    t["h2d_ms"] = a.elapsed_time(b)

    # w13: concat gate,up along N. SiluAndMul convention: first half gate.
    a.record()
    w13_q = torch.cat((gate_q, up_q), dim=2).contiguous()
    # Marlin dispatches on the ACTIVATION dtype and requires scales to match it.
    # fp16 scales with bf16 activations run but write zeros -- silently.
    w13_s = torch.cat((gate_s, up_s), dim=2).to(act_dtype).contiguous()
    w2_q = down_q.contiguous()
    w2_s = down_s.to(act_dtype).contiguous()
    e = w13_q.shape[0]
    empty_idx = torch.empty((e, 0), dtype=torch.int32, device=device)
    m13 = ops.gptq_marlin_moe_repack(
        w13_q, empty_idx, w13_q.shape[1] * 8, w13_q.shape[2], 4)
    m2 = ops.gptq_marlin_moe_repack(
        w2_q, empty_idx, w2_q.shape[1] * 8, w2_q.shape[2], 4)
    s13 = marlin_moe_permute_scales(
        s=w13_s, size_k=w13_q.shape[1] * 8, size_n=w13_s.shape[2], group_size=128)
    s2 = marlin_moe_permute_scales(
        s=w2_s, size_k=w2_q.shape[1] * 8, size_n=w2_s.shape[2], group_size=128)
    b.record(); torch.cuda.synchronize()
    t["repack_ms"] = a.elapsed_time(b)
    return {"w13": m13, "w2": m2, "s13": s13, "s2": s2}, t


def run_marlin(mw, x, topk_w, topk_ids, num_experts):
    return fused_marlin_moe(
        x, mw["w13"], mw["w2"], None, None, mw["s13"], mw["s2"],
        topk_w, topk_ids, quant_type_id=scalar_types.uint4b8.id,
        global_num_experts=num_experts,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--bench-m", type=int, default=8192)
    ap.add_argument("--iters", type=int, default=7)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    hdr = read_int4_bank_header(args.bank)
    mm = np.memmap(args.bank, dtype=np.uint8, mode="r")
    planes = load_layer_planes(mm, hdr, args.layer)
    e, k, i = hdr.experts_per_layer, hdr.hidden, hdr.moe_intermediate
    print(f"bank: {e} experts/layer, hidden {k}, intermediate {i}, "
          f"layer {args.layer}")

    mw, t_prep = to_gpu_marlin(planes)
    print(f"H2D {t_prep['h2d_ms']:.1f} ms, repack+permute {t_prep['repack_ms']:.1f} ms")

    # ---- Gate 1: correctness vs CPU fp32 reference -------------------------
    rng = np.random.default_rng(2928)
    m_ref, topk = 48, 8
    subset = sorted(rng.choice(e, size=16, replace=False).tolist())
    x_np = (rng.standard_normal((m_ref, k)) * 0.05).astype(np.float32)
    ids_np = rng.choice(subset, size=(m_ref, topk)).astype(np.int32)
    w_np = rng.random((m_ref, topk)).astype(np.float32)
    w_np /= w_np.sum(axis=1, keepdims=True)

    y_ref = cpu_reference(planes, x_np, ids_np, w_np, subset)

    x_gpu = torch.from_numpy(x_np).to("cuda", torch.bfloat16)
    y_gpu = run_marlin(
        mw, x_gpu, torch.from_numpy(w_np).cuda(),
        torch.from_numpy(ids_np).cuda(), e,
    ).float().cpu().numpy()

    diff = y_gpu - y_ref
    rel_l2 = float(np.linalg.norm(diff) / (np.linalg.norm(y_ref) + 1e-30))
    cos = float((y_gpu * y_ref).sum() /
                (np.linalg.norm(y_gpu) * np.linalg.norm(y_ref) + 1e-30))
    print(f"correctness: rel_l2 {rel_l2:.3e}  cosine {cos:.6f}")
    ok = rel_l2 <= 2e-2 and cos >= 0.999
    if not ok:
        raise SystemExit(f"CORRECTNESS GATE FAILED: rel_l2={rel_l2}, cos={cos}")

    # ---- Gate 2: speed at prefill shape ------------------------------------
    mb = args.bench_m
    xb = torch.randn(mb, k, dtype=torch.bfloat16, device="cuda") * 0.05
    idb = torch.from_numpy(rng.integers(0, e, size=(mb, topk)).astype(np.int32)).cuda()
    wb = torch.full((mb, topk), 1.0 / topk, dtype=torch.float32, device="cuda")

    def bench(fn):
        fn(); torch.cuda.synchronize()
        ts = []
        ev_a, ev_b = torch.cuda.Event(True), torch.cuda.Event(True)
        for _ in range(args.iters):
            ev_a.record(); fn(); ev_b.record(); torch.cuda.synchronize()
            ts.append(ev_a.elapsed_time(ev_b))
        return statistics.median(ts)

    kernel_ms = bench(lambda: run_marlin(mw, xb, wb, idb, e))

    # per-layer streaming cost, re-measured end to end (H2D + repack)
    stream_ms = []
    for _ in range(3):
        t0 = time.perf_counter()
        mw2, _ = to_gpu_marlin(planes)
        stream_ms.append((time.perf_counter() - t0) * 1000)
        del mw2
        torch.cuda.empty_cache()
    stream_med = statistics.median(stream_ms)

    per_layer_ms = max(stream_med, kernel_ms)  # pipelined bound
    serial_ms = stream_med + kernel_ms

    result = {
        "bank": str(args.bank), "layer": args.layer,
        "experts": e, "hidden": k, "intermediate": i,
        "correctness": {"rel_l2": rel_l2, "cosine": cos,
                        "bound_rel_l2": 2e-2, "bound_cosine": 0.999,
                        "m": m_ref, "expert_subset": subset},
        "bench": {
            "m": mb, "iters": args.iters,
            "kernel_ms_median": kernel_ms,
            "stream_h2d_repack_ms_median": stream_med,
            "per_layer_serial_ms": serial_ms,
            "per_layer_pipelined_bound_ms": per_layer_ms,
            "prefill_48_layers_serial_s": 48 * serial_ms / 1000,
            "prefill_48_layers_pipelined_s": 48 * per_layer_ms / 1000,
            "today_b70_dispatch_8k_s": 21.86,
        },
        "gpu": torch.cuda.get_device_name(0),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    b = result["bench"]
    print(f"\nM={mb}: kernel {kernel_ms:.1f} ms | stream(H2D+repack) {stream_med:.1f} ms")
    print(f"48-layer MoE component: serial {b['prefill_48_layers_serial_s']:.2f} s | "
          f"pipelined {b['prefill_48_layers_pipelined_s']:.2f} s | "
          f"today's B70 path @8K: 21.86 s")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
