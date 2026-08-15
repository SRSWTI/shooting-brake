#!/usr/bin/env python3
"""Build the pre-repacked Marlin bank (SBMARL01) from the int4 bank (SBINT401).

This is the ONE home of the AutoGPTQ->Marlin transform. It performs, offline
and once, exactly what the first marlin_prefill streamer did per layer at
runtime (27 ms/layer, 378 repack calls, ~600 MiB of transients inside vLLM's
memory-profiling forward):

    per expert:  torch.ops._C.gptq_marlin_repack(gate|up|down qweight)
    per plane:   marlin_moe_permute_scales(scales.to(act_dtype))

and writes the results as one contiguous [m13|m2|s13|s2] arena per layer, so
the runtime becomes a single H2D memcpy into device views. Scales are stored
already converted to the serving activation dtype (bf16): Marlin requires
scales dtype == activation dtype (it returns silent zeros otherwise), and the
runtime path always performed this exact fp16->bf16 conversion, so the bank
is bit-identical to what the server computed on the fly.

Memory rule: the int4 bank is mmap'd (page cache only); host transients are
one ~585 MiB D2H buffer per layer; hard abort if anon RSS exceeds 4 GiB.

Usage:
    .venv/bin/python src/phase1/build_marlin_bank.py \
        --int4-bank src/phase1/expert_bank_int4.bin \
        [--out <path>] [--scales-dtype bf16] [--validate-layers 0,23,47]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase4", "src"))

from shooting_brake_vllm.int4_bank_format import read_int4_bank_header  # noqa: E402
from shooting_brake_vllm.marlin_bank_format import (  # noqa: E402
    SCALES_DTYPE_BF16,
    SCALES_DTYPE_FP16,
    MarlinBankHeader,
    default_marlin_bank_path,
    plane_geometry,
)

RSS_ABORT_BYTES = 4 << 30

_PLANE_ORDER = (
    "gate_qweight", "gate_scales", "up_qweight",
    "up_scales", "down_qweight", "down_scales",
)


def _rss_bytes() -> int:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def _rss_guard() -> None:
    # ANON rss is what the hard rule protects (the historical OOM was 55.5
    # GiB of anonymous memory). File-backed pages from the mmap'd bank are
    # reclaimable and additionally trimmed via madvise below.
    rss = _anon_rss_bytes()
    if rss > RSS_ABORT_BYTES:
        raise SystemExit(
            f"ABORT: anon RSS {rss / 2**30:.2f} GiB exceeds the 4 GiB rule"
        )


def _anon_rss_bytes() -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("RssAnon:"):
                return int(line.split()[1]) * 1024
    return _rss_bytes()


class _Int4LayerReader:
    """Strided per-plane views over one mmap'd int4-bank layer."""

    def __init__(self, bank_path: str) -> None:
        self.hdr = read_int4_bank_header(bank_path)
        self._mm = np.memmap(bank_path, dtype=np.uint8, mode="r")
        h = self.hdr
        k, i = h.hidden, h.moe_intermediate
        self._geom = {
            "gate_qweight": (torch.int32, (k // 8, i)),
            "gate_scales": (torch.float16, (k // 128, i)),
            "up_qweight": (torch.int32, (k // 8, i)),
            "up_scales": (torch.float16, (k // 128, i)),
            "down_qweight": (torch.int32, (i // 8, k)),
            "down_scales": (torch.float16, (i // 128, k)),
        }
        self._offs = dict(zip(_PLANE_ORDER, h.plane_offsets))

    def slab(self, layer: int) -> torch.Tensor:
        h = self.hdr
        off = h.data_offset + layer * h.layer_stride_bytes
        view = np.asarray(self._mm[off: off + h.layer_stride_bytes])
        return torch.from_numpy(view)

    def discard(self, layer: int) -> None:
        """madvise(DONTNEED) the consumed layer's pages (4096-aligned)."""
        import mmap as _mmap

        h = self.hdr
        off = h.data_offset + layer * h.layer_stride_bytes
        self._mm._mmap.madvise(_mmap.MADV_DONTNEED, off, h.layer_stride_bytes)

    def plane(self, slab: torch.Tensor, name: str) -> torch.Tensor:
        dtype, (rows, cols) = self._geom[name]
        item = torch.tensor([], dtype=dtype).element_size()
        return torch.as_strided(
            slab.view(dtype),
            (self.hdr.experts_per_layer, rows, cols),
            (self.hdr.expert_stride_bytes // item, cols, 1),
            storage_offset=self._offs[name] // item,
        )


def _repack_layer_into(
    reader: _Int4LayerReader,
    slab_dev: torch.Tensor,
    act_dtype: torch.dtype,
    views: dict[str, torch.Tensor],
) -> None:
    """AutoGPTQ planes (device slab) -> Marlin planes (device arena views).

    Split-repacking gate and up separately into halves of the fused m13 is
    BIT-EXACT to repacking the concatenation (verified on real bank planes:
    qweights and permuted scales both torch.equal), because Marlin tiles N in
    64-column blocks and I=1024 is a multiple.
    """
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (  # local: needs CUDA ctx
        marlin_moe_permute_scales,
    )
    h = reader.hdr
    e, k, i = h.experts_per_layer, h.hidden, h.moe_intermediate
    gate_q = reader.plane(slab_dev, "gate_qweight")
    up_q = reader.plane(slab_dev, "up_qweight")
    down_q = reader.plane(slab_dev, "down_qweight")
    perm = torch.empty(0, dtype=torch.int32, device=slab_dev.device)
    half = 2 * i
    m13, m2, s13, s2 = views["m13"], views["m2"], views["s13"], views["s2"]
    for ex in range(e):
        m13[ex, :, :half] = torch.ops._C.gptq_marlin_repack(
            gate_q[ex], perm, k, i, 4, False)
        m13[ex, :, half:] = torch.ops._C.gptq_marlin_repack(
            up_q[ex], perm, k, i, 4, False)
        m2[ex] = torch.ops._C.gptq_marlin_repack(
            down_q[ex], perm, i, k, 4, False)
    s13[:, :, :i] = marlin_moe_permute_scales(
        s=reader.plane(slab_dev, "gate_scales").to(act_dtype).contiguous(),
        size_k=k, size_n=i, group_size=h.group_size)
    s13[:, :, i:] = marlin_moe_permute_scales(
        s=reader.plane(slab_dev, "up_scales").to(act_dtype).contiguous(),
        size_k=k, size_n=i, group_size=h.group_size)
    s2.copy_(marlin_moe_permute_scales(
        s=reader.plane(slab_dev, "down_scales").to(act_dtype).contiguous(),
        size_k=i, size_n=k, group_size=h.group_size))


def _arena_views(
    arena: torch.Tensor, e: int, k: int, i: int, act_dtype: torch.dtype,
    offsets: tuple[int, ...], sizes: tuple[int, ...],
) -> dict[str, torch.Tensor]:
    def cut(idx: int, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
        o, s = offsets[idx], sizes[idx]
        return arena[o: o + s].view(dtype).view(*shape)

    return {
        "m13": cut(0, torch.int32, (e, k // 16, 4 * i)),
        "m2": cut(1, torch.int32, (e, i // 16, 2 * k)),
        "s13": cut(2, act_dtype, (e, k // 128, 2 * i)),
        "s2": cut(3, act_dtype, (e, i // 128, k)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--int4-bank", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--scales-dtype", choices=("bf16", "fp16"), default="bf16")
    ap.add_argument("--validate-layers", default="0,23,47")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_path = args.out or default_marlin_bank_path(args.int4_bank)
    act_dtype = torch.bfloat16 if args.scales_dtype == "bf16" else torch.float16
    dtype_code = (
        SCALES_DTYPE_BF16 if act_dtype is torch.bfloat16 else SCALES_DTYPE_FP16
    )
    dev = torch.device(args.device)

    reader = _Int4LayerReader(args.int4_bank)
    h = reader.hdr
    e, k, i = h.experts_per_layer, h.hidden, h.moe_intermediate
    offsets, sizes = plane_geometry(e, k, i)
    stride = offsets[-1] + sizes[-1]
    hdr = MarlinBankHeader(
        num_layers=h.num_layers,
        hidden=k,
        moe_intermediate=i,
        group_size=h.group_size,
        bits=h.bits,
        zero_point=h.zero_point,
        scales_dtype=dtype_code,
        plane_offsets=offsets,
        plane_sizes=sizes,
        layer_stride_bytes=stride,
        source_expert_ids=tuple(h.source_expert_ids),
    )
    hdr.validate()
    total = hdr.data_offset + h.num_layers * stride
    print(
        f"int4 bank: {e} experts/layer x {h.num_layers} layers "
        f"(K={k}, I={i}, g={h.group_size}, zp={h.zero_point})\n"
        f"marlin bank -> {out_path}  ({total / 2**30:.2f} GiB, "
        f"{stride / 2**20:.1f} MiB/layer, scales {args.scales_dtype})"
    )

    slab_dev = torch.empty(h.layer_stride_bytes, dtype=torch.uint8, device=dev)
    arena = torch.empty(stride, dtype=torch.uint8, device=dev)
    views = _arena_views(arena, e, k, i, act_dtype, offsets, sizes)
    host = torch.empty(stride, dtype=torch.uint8)  # one D2H buffer, reused

    fd = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.pwrite(fd, hdr.pack(), 0)
        t0 = time.perf_counter()
        for layer in range(h.num_layers):
            slab_dev.copy_(reader.slab(layer), non_blocking=False)
            reader.discard(layer)
            _repack_layer_into(reader, slab_dev, act_dtype, views)
            host.copy_(arena)
            os.pwrite(fd, memoryview(host.numpy()), hdr.data_offset + layer * stride)
            _rss_guard()
            if layer % 8 == 0 or layer == h.num_layers - 1:
                el = time.perf_counter() - t0
                print(f"  layer {layer:2d}/{h.num_layers}  {el:6.1f}s  "
                      f"anon rss {_anon_rss_bytes() / 2**30:.2f} GiB", flush=True)
        os.fsync(fd)
    finally:
        os.close(fd)
    build_s = time.perf_counter() - t0

    # -- validation: file bytes == fresh repack, bit for bit -----------------
    check_layers = [int(x) for x in args.validate_layers.split(",") if x != ""]
    mm_out = np.memmap(out_path, dtype=np.uint8, mode="r")
    fresh = torch.empty_like(arena)
    fresh_views = _arena_views(fresh, e, k, i, act_dtype, offsets, sizes)
    for layer in check_layers:
        off = hdr.data_offset + layer * stride
        slab_dev.copy_(reader.slab(layer), non_blocking=False)
        _repack_layer_into(reader, slab_dev, act_dtype, fresh_views)
        file_bytes = torch.from_numpy(
            np.asarray(mm_out[off: off + stride])
        ).to(dev)
        if not torch.equal(file_bytes, fresh):
            n_bad = int((file_bytes != fresh).sum())
            raise SystemExit(
                f"VALIDATION FAILED: layer {layer} differs from fresh repack "
                f"({n_bad} bytes)"
            )
        print(f"  validate layer {layer:2d}: bit-exact vs fresh repack")
        _rss_guard()

    print(f"DONE  build {build_s:.1f}s  "
          f"validated layers {check_layers} bit-exact  -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
