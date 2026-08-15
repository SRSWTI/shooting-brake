"""Canonical on-disk ABI for the pre-repacked Marlin expert bank.

``SBMARL01`` is the int4 bank (``SBINT401``) transformed offline into exactly
the tensors ``fused_marlin_moe`` consumes: per layer, one contiguous arena

    [ m13 | m2 | s13 | s2 ]

where every plane is the *bit-exact* output of the runtime repack this bank
replaces (``torch.ops._C.gptq_marlin_repack`` on the AutoGPTQ planes, plus
``marlin_moe_permute_scales`` with the scales pre-converted to the serving
activation dtype). The runtime therefore does a single H2D memcpy per layer
into a device arena whose ``m13/m2/s13/s2`` tensors are views -- no repack,
no per-layer transients, nothing for a compiler to trace.

Plane shapes (E experts, hidden K, intermediate I, all int4 g128 sym zp8):

    m13 : [E, K/16, 4*I] int32   fused gate|up Marlin qweight
    m2  : [E, I/16, 2*K] int32   down Marlin qweight
    s13 : [E, K/128, 2*I] act    fused gate|up permuted scales
    s2  : [E, I/128,  K ] act    down permuted scales

The transform lives in ONE place: ``src/phase1/build_marlin_bank.py``. The
streamer only validates this header and copies bytes.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass

MAGIC = b"SBMARL01"
VERSION = 1
ALIGNMENT = 4096
PLANE_NAMES = ("m13", "m2", "s13", "s2")

# 0 = float16, 1 = bfloat16 -- the dtype the scales were converted to at
# build time, which MUST equal the serving activation dtype (Marlin returns
# silent zeros on mismatch; learned the hard way).
SCALES_DTYPE_FP16 = 0
SCALES_DTYPE_BF16 = 1

# magic + 10 u32 (version, num_layers, experts_per_layer, hidden,
# moe_intermediate, group_size, bits, zero_point, scales_dtype, pad)
# + 10 u64 (4x plane offset, 4x plane size, layer_stride, data_offset)
HEADER_PREFIX_FMT = "<8s10I10Q"
HEADER_PREFIX_SIZE = struct.calcsize(HEADER_PREFIX_FMT)
assert HEADER_PREFIX_SIZE == 128, HEADER_PREFIX_SIZE


def _align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def plane_geometry(
    experts_per_layer: int, hidden: int, moe_intermediate: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(offsets, sizes) of m13/m2/s13/s2 within one layer arena."""
    e, k, i = experts_per_layer, hidden, moe_intermediate
    sizes = (
        e * (k // 16) * (4 * i) * 4,
        e * (i // 16) * (2 * k) * 4,
        e * (k // 128) * (2 * i) * 2,
        e * (i // 128) * k * 2,
    )
    offsets: list[int] = []
    off = 0
    for s in sizes:
        assert off % ALIGNMENT == 0, (off, s)
        offsets.append(off)
        off += s
    assert off % ALIGNMENT == 0, off
    return tuple(offsets), sizes


@dataclass(frozen=True)
class MarlinBankHeader:
    num_layers: int
    hidden: int
    moe_intermediate: int
    group_size: int
    bits: int
    zero_point: int
    scales_dtype: int
    plane_offsets: tuple[int, int, int, int]
    plane_sizes: tuple[int, int, int, int]
    layer_stride_bytes: int
    source_expert_ids: tuple[int, ...]

    @property
    def experts_per_layer(self) -> int:
        return len(self.source_expert_ids)

    @property
    def data_offset(self) -> int:
        return _align_up(HEADER_PREFIX_SIZE + 4 * self.experts_per_layer)

    def validate(self) -> None:
        offs, sizes = plane_geometry(
            self.experts_per_layer, self.hidden, self.moe_intermediate
        )
        if self.plane_offsets != offs or self.plane_sizes != sizes:
            raise ValueError(
                f"marlin bank plane geometry mismatch: header "
                f"{self.plane_offsets}/{self.plane_sizes}, derived {offs}/{sizes}"
            )
        if self.layer_stride_bytes != offs[-1] + sizes[-1]:
            raise ValueError(
                f"layer_stride_bytes {self.layer_stride_bytes} != "
                f"{offs[-1] + sizes[-1]}"
            )
        if self.bits != 4 or self.zero_point != 8:
            raise ValueError(
                f"expected int4 sym zp8 (AutoGPTQ zeros-1 convention, see "
                f"int4_bank_format.py); got bits={self.bits} "
                f"zp={self.zero_point}"
            )
        if self.scales_dtype not in (SCALES_DTYPE_FP16, SCALES_DTYPE_BF16):
            raise ValueError(f"unknown scales_dtype {self.scales_dtype}")
        ids = self.source_expert_ids
        if not ids or any(b <= a for a, b in zip(ids, ids[1:])):
            raise ValueError("source_expert_ids must be strictly ascending")

    def pack(self) -> bytes:
        prefix = struct.pack(
            HEADER_PREFIX_FMT,
            MAGIC,
            VERSION,
            self.num_layers,
            self.experts_per_layer,
            self.hidden,
            self.moe_intermediate,
            self.group_size,
            self.bits,
            self.zero_point,
            self.scales_dtype,
            0,
            *self.plane_offsets,
            *self.plane_sizes,
            self.layer_stride_bytes,
            self.data_offset,
        )
        ids = struct.pack(f"<{self.experts_per_layer}i", *self.source_expert_ids)
        blob = prefix + ids
        return blob + b"\x00" * (self.data_offset - len(blob))


def parse_marlin_bank_header(data: bytes | bytearray | memoryview) -> MarlinBankHeader:
    (
        magic, version, num_layers, experts_per_layer, hidden,
        moe_intermediate, group_size, bits, zero_point, scales_dtype, _pad,
        o0, o1, o2, o3, s0, s1, s2, s3, layer_stride, data_offset,
    ) = struct.unpack_from(HEADER_PREFIX_FMT, data, 0)
    if magic != MAGIC:
        raise ValueError(f"not a marlin bank (magic {magic!r})")
    if version != VERSION:
        raise ValueError(f"marlin bank version {version} != {VERSION}")
    ids = struct.unpack_from(
        f"<{experts_per_layer}i", data, HEADER_PREFIX_SIZE
    )
    hdr = MarlinBankHeader(
        num_layers=num_layers,
        hidden=hidden,
        moe_intermediate=moe_intermediate,
        group_size=group_size,
        bits=bits,
        zero_point=zero_point,
        scales_dtype=scales_dtype,
        plane_offsets=(o0, o1, o2, o3),
        plane_sizes=(s0, s1, s2, s3),
        layer_stride_bytes=layer_stride,
        source_expert_ids=ids,
    )
    if hdr.data_offset != data_offset:
        raise ValueError(
            f"data_offset {data_offset} != derived {hdr.data_offset}"
        )
    hdr.validate()
    return hdr


def read_marlin_bank_header(path: str | os.PathLike[str]) -> MarlinBankHeader:
    """Header-only read via pread; never maps or loads plane data."""
    fd = os.open(path, os.O_RDONLY)
    try:
        head = os.pread(fd, HEADER_PREFIX_SIZE, 0)
        if len(head) < HEADER_PREFIX_SIZE:
            raise ValueError(f"{path}: truncated header")
        experts = struct.unpack_from("<I", head, 8 + 8)[0]
        blob = os.pread(fd, HEADER_PREFIX_SIZE + 4 * experts, 0)
        hdr = parse_marlin_bank_header(blob)
        expect = hdr.data_offset + hdr.num_layers * hdr.layer_stride_bytes
        actual = os.fstat(fd).st_size
        if actual != expect:
            raise ValueError(
                f"{path}: file size {actual} != header-implied {expect}"
            )
        return hdr
    finally:
        os.close(fd)


def default_marlin_bank_path(int4_bank_path: str) -> str:
    env = os.environ.get("SHOOTING_BRAKE_MARLIN_BANK")
    return env if env else int4_bank_path + ".marlin"
