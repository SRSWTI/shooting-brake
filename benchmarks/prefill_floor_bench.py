#!/usr/bin/env python3
"""Prefill microbench trio: compute floor, overlap interference, full-bank pin.

Gates the endgame table in the prefill-parity plan; each mode has a kill
condition. DO NOT run while a serving suite is live -- these use the GPU's
copy engines and SMs and will taint both results.

Modes:
  compute   48-layer fused_marlin_moe floor with weights RESIDENT (zero
            streaming). Pins the compute column of the endgame table.
            Kill: >~2 s @ M=8192 -> the short-ctx story dies (32K parity
            survives on stream-once regardless).
  overlap   Registered-mmap H2D ring (production geometry) concurrent with
            fused kernels. Decides TTFT = max(stream, compute) vs -> sum.
            Kill: >20% mutual degradation -> endgame table shifts right.
  register  cudaHostRegister the FULL bank plane region. Measures the
            one-time startup tax (projection: ~7.8 s), memory pressure on
            the 59 GiB box, and spot-checks DMA rate across the bank.
            Kill: OOM/thrash -> register per-layer working set instead.

Usage (after the suite completes):
  .venv/bin/python benchmarks/prefill_floor_bench.py --mode compute \
      --out benchmarks/results/prefill_floor/compute.json
  .venv/bin/python benchmarks/prefill_floor_bench.py --mode overlap \
      --out benchmarks/results/prefill_floor/overlap.json
  .venv/bin/python benchmarks/prefill_floor_bench.py --mode register \
      --out benchmarks/results/prefill_floor/register.json
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import statistics
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
DEFAULT_INT4_BANK = str(REPO / "src/phase1/expert_bank_int4.bin")

NUM_LAYERS = 48
TOP_K = 8


def _bank(args):
    import sys
    sys.path.insert(0, str(REPO / "src/phase4/src"))
    from shooting_brake_vllm.marlin_bank_format import (
        default_marlin_bank_path,
        read_marlin_bank_header,
    )
    path = default_marlin_bank_path(args.int4_bank)
    return path, read_marlin_bank_header(path)


def _register_mmap(path: str, h, layers: int) -> tuple[mmap.mmap, torch.Tensor, float]:
    """MAP_PRIVATE|PROT_WRITE over O_RDONLY fd + cudaHostRegister.

    Same recipe as MarlinPrefillStreamer._open_bank_source; returns
    (mmap, uint8 tensor over `layers` leading layers, register_seconds).
    """
    data_len = layers * h.layer_stride_bytes
    map_len = h.data_offset + data_len
    fd = os.open(path, os.O_RDONLY)
    try:
        mm = mmap.mmap(fd, map_len, flags=mmap.MAP_PRIVATE,
                       prot=mmap.PROT_READ | mmap.PROT_WRITE)
    finally:
        os.close(fd)
    mm.madvise(mmap.MADV_WILLNEED)
    base = torch.frombuffer(
        memoryview(mm)[h.data_offset: h.data_offset + data_len],
        dtype=torch.uint8,
    )
    t0 = time.perf_counter()
    status = int(torch.cuda.cudart().cudaHostRegister(base.data_ptr(), data_len, 0))
    reg_s = time.perf_counter() - t0
    if status != 0:
        raise SystemExit(f"cudaHostRegister failed: cudaError {status}")
    return mm, base, reg_s


def _views(arena: torch.Tensor, h, act_dtype=torch.bfloat16):
    e, k, i = h.experts_per_layer, h.hidden, h.moe_intermediate
    offs, sizes = h.plane_offsets, h.plane_sizes

    def cut(idx, dtype, shape):
        return arena[offs[idx]: offs[idx] + sizes[idx]].view(dtype).view(*shape)

    return {
        "m13": cut(0, torch.int32, (e, k // 16, 4 * i)),
        "m2": cut(1, torch.int32, (e, i // 16, 2 * k)),
        "s13": cut(2, act_dtype, (e, k // 128, 2 * i)),
        "s2": cut(3, act_dtype, (e, i // 128, k)),
    }


def _moe_call(x, v, ids, weights, e):
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
        fused_marlin_moe,
    )
    from vllm.scalar_type import scalar_types
    return fused_marlin_moe(
        x, v["m13"], v["m2"], None, None, v["s13"], v["s2"],
        weights, ids, quant_type_id=scalar_types.uint4b8.id,
        global_num_experts=e,
    )


def _routing(m: int, e: int, device):
    torch.manual_seed(42)
    ids = torch.stack([
        torch.randperm(e, device=device)[:TOP_K] for _ in range(m)
    ]).to(torch.int32)
    w = torch.rand(m, TOP_K, device=device, dtype=torch.float32)
    return ids, w / w.sum(-1, keepdim=True)


def _thp_vmstat() -> dict:
    """THP split/collapse counters: a pin BLOCKS collapse (named kernel
    outcome), so deltas here during register/DMA give a direct empirical
    read on whether THP promotion ever races the registration."""
    keys = ("thp_split_page", "thp_split_page_failed", "thp_collapse_alloc",
            "thp_collapse_alloc_failed", "thp_fault_alloc")
    out = {}
    for line in open("/proc/vmstat"):
        k, _, v = line.partition(" ")
        if k in keys:
            out[k] = int(v)
    return out


def _meminfo() -> dict:
    out = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":")
        if k in ("MemTotal", "MemFree", "MemAvailable", "Cached", "Unevictable", "Mlocked"):
            out[k + "_gib"] = int(v.split()[0]) / 2**20
    return out


def mode_compute(args, path, h) -> dict:
    """48-layer MoE floor, weights resident, M sweep for the tier decision."""
    dev = "cuda"
    e = h.experts_per_layer
    mm, base, reg_s = _register_mmap(path, h, layers=1)
    arena = torch.empty(h.layer_stride_bytes, dtype=torch.uint8, device=dev)
    arena.copy_(base[: h.layer_stride_bytes])
    torch.cuda.synchronize()
    v = _views(arena, h)

    sweep = {}
    for m in args.m_sweep:
        x = torch.randn(m, h.hidden, device=dev, dtype=torch.bfloat16)
        ids, w = _routing(m, e, dev)
        _moe_call(x, v, ids, w, e)  # warmup + allocator
        torch.cuda.synchronize()
        ev_a, ev_b = torch.cuda.Event(True), torch.cuda.Event(True)
        ts = []
        for _ in range(args.iters):
            torch.cuda.synchronize()
            ev_a.record()
            for _ in range(NUM_LAYERS):
                _moe_call(x, v, ids, w, e)
            ev_b.record()
            torch.cuda.synchronize()
            ts.append(ev_a.elapsed_time(ev_b) / 1000.0)
        med = statistics.median(ts)
        sweep[str(m)] = {
            "moe_48layer_s_median": med,
            "moe_per_layer_ms": med / NUM_LAYERS * 1e3,
            "raw_s": ts,
        }
        del x
    torch.cuda.cudart().cudaHostUnregister(base.data_ptr())
    return {
        "register_1layer_s": reg_s,
        "sweep": sweep,
        "note": (
            "MoE floor only: same-layer weights reused 48x (identical shapes, "
            "identical compute; cache effects negligible at 585 MiB/layer). "
            "Add non-MoE busy (~0.30 s @ 8K, attribution_8k.json: attention "
            "0.081 + gdn 0.070 + norms 0.022 + unattributed_gemm 0.149) for "
            "the full compute column."
        ),
        "kill_condition": "moe_48layer + 0.30 > ~2 s at M=8192",
    }


def mode_overlap(args, path, h) -> dict:
    """Production-geometry ring H2D concurrent with fused kernels."""
    dev = "cuda"
    e = h.experts_per_layer
    layers = min(NUM_LAYERS, args.overlap_layers)
    mm, base, reg_s = _register_mmap(path, h, layers=layers)
    stride = h.layer_stride_bytes
    gib_total = layers * stride / 2**30

    arenas = [torch.empty(stride, dtype=torch.uint8, device=dev) for _ in range(2)]
    v = _views(arenas[0], h)  # kernels read slot 0; copies alternate slots
    copy_stream = torch.cuda.Stream()
    m = args.m_sweep[-1] if args.m_sweep else 8192
    x = torch.randn(m, h.hidden, device=dev, dtype=torch.bfloat16)
    ids, w = _routing(m, e, dev)

    def h2d_all():
        with torch.cuda.stream(copy_stream):
            for layer in range(layers):
                arenas[layer % 2].copy_(
                    base[layer * stride: (layer + 1) * stride], non_blocking=True
                )

    def kernels_all():
        for _ in range(layers):
            _moe_call(x, v, ids, w, e)

    def timed(fn) -> float:
        ts = []
        for _ in range(args.iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        return statistics.median(ts)

    _moe_call(x, v, ids, w, e); h2d_all(); torch.cuda.synchronize()  # warmup

    stream_alone = timed(h2d_all)
    compute_alone = timed(kernels_all)

    def both():
        h2d_all()
        kernels_all()

    together = timed(both)
    ideal = max(stream_alone, compute_alone)
    torch.cuda.cudart().cudaHostUnregister(base.data_ptr())
    return {
        "layers": layers, "m": m, "gib_streamed": gib_total,
        "register_s": reg_s,
        "stream_alone_s": stream_alone,
        "stream_alone_gib_per_s": gib_total / stream_alone,
        "compute_alone_s": compute_alone,
        "together_s": together,
        "ideal_max_s": ideal,
        "overlap_efficiency": ideal / together,
        "degradation_vs_max": together / ideal - 1.0,
        "kill_condition": "degradation_vs_max > 0.20",
    }


def mode_register(args, path, h) -> dict:
    """Full-bank pin: startup tax, memory pressure, DMA spot checks."""
    before = _meminfo()
    thp_before = _thp_vmstat()
    mm, base, reg_s = _register_mmap(path, h, layers=NUM_LAYERS)
    after = _meminfo()
    stride = h.layer_stride_bytes
    gib = NUM_LAYERS * stride / 2**30

    dev_buf = torch.empty(stride, dtype=torch.uint8, device="cuda")
    ev_a, ev_b = torch.cuda.Event(True), torch.cuda.Event(True)
    spots = {}
    for layer in range(0, NUM_LAYERS, 8):
        src = base[layer * stride: (layer + 1) * stride]
        dev_buf.copy_(src, non_blocking=True)  # warm
        torch.cuda.synchronize()
        ev_a.record()
        dev_buf.copy_(src, non_blocking=True)
        ev_b.record()
        torch.cuda.synchronize()
        s = ev_a.elapsed_time(ev_b) / 1000.0
        spots[str(layer)] = {"s": s, "gib_per_s": (stride / 2**30) / s}

    thp_after = _thp_vmstat()
    t0 = time.perf_counter()
    torch.cuda.cudart().cudaHostUnregister(base.data_ptr())
    unreg_s = time.perf_counter() - t0
    return {
        "bank_gib": gib,
        "register_s": reg_s,
        "register_gib_per_s": gib / reg_s,
        "unregister_s": unreg_s,
        "meminfo_before": before,
        "meminfo_after": after,
        "available_delta_gib": before["MemAvailable_gib"] - after["MemAvailable_gib"],
        "thp_vmstat_delta": {
            k: thp_after[k] - thp_before[k] for k in thp_after
        },
        "h2d_spot_checks": spots,
        "kill_condition": "OOM, MemAvailable < ~8 GiB after pin, or spot rate << 52.8",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("compute", "overlap", "register"), required=True)
    ap.add_argument("--int4-bank", default=DEFAULT_INT4_BANK)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--iters", type=int, default=7)
    ap.add_argument("--m-sweep", type=int, nargs="+",
                    default=[2048, 8192, 16384, 32768])
    ap.add_argument("--overlap-layers", type=int, default=48)
    args = ap.parse_args()

    assert torch.cuda.is_available()
    path, h = _bank(args)
    result = {
        "kind": f"prefill_floor_{args.mode}",
        "bank": path,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        **{"compute": mode_compute, "overlap": mode_overlap,
           "register": mode_register}[args.mode](args, path, h),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items()
                      if not isinstance(v, dict) or len(str(v)) < 400}, indent=2))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
