"""On-disk ABI for the B12x expert bank (bank v2).

Per layer, one contiguous arena of four 4096-aligned planes:

  w1      uint8            [E, 2*intermediate, hidden/2]   packed fp4 (e2m1),
                           rows 0..I-1 = gate_proj, I..2I-1 = up_proj
  w2      uint8            [E, hidden, intermediate/2]     packed fp4
  sf1     float8_e4m3fn    MMA-swizzled block scales for w1 (6-D, shape in
                           header; produced by flashinfer_convert_sf_to_mma_layout)
  sf2     float8_e4m3fn    MMA-swizzled block scales for w2

Scale convention: the checkpoint's per-projection ``weight_scale_2`` global
multiplier is BAKED into the e4m3 block scales at build time (exact
per-row semantics, one e4m3 round-trip -- identical to what vLLM's
FlashInferB12xExperts.process_weights_after_loading does at load), so the
runtime passes ``alpha = 1.0`` per expert. fc2_input_scale is uniform 1.0
(W4A16 checkpoint; b12x quantizes FC2 inputs dynamically per block).

Weights are verbatim checkpoint bytes (no transform); scales are the only
build-time computation. ``sf*_shape`` are stored so the runtime never
re-derives the swizzle.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass

MAGIC = b"SBB12X01"
VERSION = 1
ALIGNMENT = 4096

# magic + 8 u32 (version, num_layers, experts_per_layer, hidden,
# moe_intermediate, group_size, expert_id_base, pad)
# + 12 u32 (sf1_shape[6], sf2_shape[6])
# + 14 u64 (6x plane offset, 6x plane size, layer_stride, data_offset)
#
# Planes: w1, w2, sf1, sf2, alpha1, alpha2. The alpha planes exist because
# the naive scale bake (block_scale * weight_scale_2 -> e4m3) UNDERFLOWS
# for ModelOpt checkpoints: measured 86.5% of products flush to zero
# (scale_2 = amax/(6*448) ~ 1e-5; products max 4.6e-3 < e4m3 min subnormal
# 1.95e-3). This is also a live bug in vLLM's FlashInferB12xExperts
# load-time bake. Instead: w13 block scales carry only the RATIO
# proj_s2 / max(gate_s2, up_s2) (O(1), e4m3-safe) with
# alpha1[e] = max(gate_s2, up_s2) fp32; w2 blocks are verbatim with
# alpha2[e] = down_s2. The wrapper applies alpha pre-activation (per-expert
# weight-dequant multiplier), so the factorization is exact.
HEADER_FMT = "<8s8I12I14Q"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
assert HEADER_SIZE == 200, HEADER_SIZE


def _align_up(v: int, a: int = ALIGNMENT) -> int:
    return (v + a - 1) // a * a


@dataclass(frozen=True)
class B12xBankHeader:
    num_layers: int
    experts_per_layer: int
    hidden: int
    moe_intermediate: int
    group_size: int
    expert_id_base: int
    sf1_shape: tuple[int, ...]
    sf2_shape: tuple[int, ...]
    plane_offsets: tuple[int, ...]
    plane_sizes: tuple[int, ...]
    layer_stride_bytes: int
    data_offset: int

    def to_bytes(self) -> bytes:
        blob = struct.pack(
            HEADER_FMT, MAGIC, VERSION,
            self.num_layers, self.experts_per_layer, self.hidden,
            self.moe_intermediate, self.group_size, self.expert_id_base, 0,
            *self.sf1_shape, *self.sf2_shape,
            *self.plane_offsets, *self.plane_sizes,
            self.layer_stride_bytes, self.data_offset,
        )
        return blob + b"\x00" * (self.data_offset - len(blob))


def plane_geometry(
    experts: int, hidden: int, intermediate: int,
    sf1_shape: tuple[int, ...], sf2_shape: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(offsets, sizes) of w1/w2/sf1/sf2/alpha1/alpha2 in one layer arena."""
    import math
    sizes = (
        experts * 2 * intermediate * (hidden // 2),   # w1 uint8
        experts * hidden * (intermediate // 2),       # w2 uint8
        math.prod(sf1_shape),                         # sf1 e4m3 (1 B/elem)
        math.prod(sf2_shape),                         # sf2 e4m3
        experts * 4,                                  # alpha1 f32
        experts * 4,                                  # alpha2 f32
    )
    offsets, off = [], 0
    for s in sizes:
        offsets.append(off)
        off += _align_up(s)
    return tuple(offsets), sizes


def layer_stride(sizes: tuple[int, ...]) -> int:
    return sum(_align_up(s) for s in sizes)


def make_header(
    num_layers: int, experts: int, hidden: int, intermediate: int,
    group_size: int, expert_id_base: int,
    sf1_shape: tuple[int, ...], sf2_shape: tuple[int, ...],
) -> B12xBankHeader:
    offsets, sizes = plane_geometry(experts, hidden, intermediate,
                                    sf1_shape, sf2_shape)
    return B12xBankHeader(
        num_layers=num_layers, experts_per_layer=experts, hidden=hidden,
        moe_intermediate=intermediate, group_size=group_size,
        expert_id_base=expert_id_base,
        sf1_shape=tuple(sf1_shape), sf2_shape=tuple(sf2_shape),
        plane_offsets=offsets, plane_sizes=sizes,
        layer_stride_bytes=layer_stride(sizes),
        data_offset=_align_up(HEADER_SIZE),
    )


def parse_header(data: bytes | memoryview) -> B12xBankHeader:
    fields = struct.unpack_from(HEADER_FMT, data, 0)
    magic, version = fields[0], fields[1]
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic!r}, want {MAGIC!r}")
    if version != VERSION:
        raise ValueError(f"bad version {version}, want {VERSION}")
    (num_layers, experts, hidden, intermediate, group_size,
     expert_id_base, _pad) = fields[2:9]
    sf1_shape = tuple(fields[9:15])
    sf2_shape = tuple(fields[15:21])
    plane_offsets = tuple(fields[21:27])
    plane_sizes = tuple(fields[27:33])
    stride, data_offset = fields[33], fields[34]
    return B12xBankHeader(
        num_layers=num_layers, experts_per_layer=experts, hidden=hidden,
        moe_intermediate=intermediate, group_size=group_size,
        expert_id_base=expert_id_base,
        sf1_shape=sf1_shape, sf2_shape=sf2_shape,
        plane_offsets=plane_offsets, plane_sizes=plane_sizes,
        layer_stride_bytes=stride, data_offset=data_offset,
    )


def read_b12x_bank_header(path: str | os.PathLike) -> B12xBankHeader:
    """Header-only pread; never maps plane data."""
    fd = os.open(path, os.O_RDONLY)
    try:
        return parse_header(os.pread(fd, HEADER_SIZE, 0))
    finally:
        os.close(fd)


def default_b12x_bank_path(int4_bank_path: str) -> str:
    env = os.environ.get("SHOOTING_BRAKE_B12X_BANK")
    return env if env else int4_bank_path + ".b12x"
