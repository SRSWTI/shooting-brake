"""Canonical on-disk ABI for Shooting Brake AutoGPTQ int4 expert banks.

ABI discipline: any field order, field size, alignment rule, or field semantic
change MUST increment ``VERSION``. Readers reject every unknown version before
interpreting version-specific fields.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from pathlib import Path

MAGIC = b"SBINT401"
VERSION = 2
ALIGNMENT = 4096
GROUP_SIZE = 128
BITS = 4
ZERO_POINT = 8
RESIDENT_SET_SHARED_ACROSS_LAYERS = 1
PLANE_NAMES = (
    "gate_qweight",
    "gate_scales",
    "up_qweight",
    "up_scales",
    "down_qweight",
    "down_scales",
)

# magic; 14 metadata u32; six (offset,size) u32 pairs; two stride u64.
HEADER_PREFIX_FMT = "<8s26I2Q"
HEADER_PREFIX = struct.Struct(HEADER_PREFIX_FMT)
HEADER_PREFIX_SIZE = HEADER_PREFIX.size
# Python's import-time equivalent of static_assert(sizeof(HeaderPrefix) == 128).
assert HEADER_PREFIX_SIZE == 128, (
    f"int4 bank header prefix ABI changed: {HEADER_PREFIX_SIZE} != 128"
)


def _align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class Int4BankHeader:
    """Parsed v2 header, including the explicit compact-slot to source-ID map."""

    num_layers: int
    source_num_layers: int
    source_experts_per_layer: int
    hidden: int
    moe_intermediate: int
    group_size: int
    bits: int
    zero_point: int
    plane_offsets: tuple[int, int, int, int, int, int]
    plane_sizes: tuple[int, int, int, int, int, int]
    expert_stride_bytes: int
    layer_stride_bytes: int
    source_expert_ids: tuple[int, ...]
    resident_set_shared_across_layers: int = RESIDENT_SET_SHARED_ACROSS_LAYERS

    @property
    def version(self) -> int:
        return VERSION

    @property
    def experts_per_layer(self) -> int:
        return len(self.source_expert_ids)

    @property
    def data_offset(self) -> int:
        return _align_up(HEADER_PREFIX_SIZE + 4 * self.experts_per_layer)

    @property
    def plane_fields(self) -> tuple[int, ...]:
        fields: list[int] = []
        for offset, size in zip(self.plane_offsets, self.plane_sizes):
            fields.extend((offset, size))
        return tuple(fields)

    def validate(self) -> None:
        if self.num_layers <= 0 or self.num_layers > self.source_num_layers:
            raise ValueError(
                f"invalid layer geometry: resident={self.num_layers}, "
                f"source={self.source_num_layers}"
            )
        if self.source_experts_per_layer <= 0 or self.experts_per_layer <= 0:
            raise ValueError("expert counts must be positive")
        if self.resident_set_shared_across_layers != RESIDENT_SET_SHARED_ACROSS_LAYERS:
            raise ValueError(
                "v2 int4 banks require one resident expert set shared across all layers"
            )
        if self.source_expert_ids != tuple(sorted(set(self.source_expert_ids))):
            raise ValueError("source_expert_ids must be strictly increasing and unique")
        if (
            self.source_expert_ids[0] < 0
            or self.source_expert_ids[-1] >= self.source_experts_per_layer
        ):
            raise ValueError(
                f"source_expert_ids must be in 0..{self.source_experts_per_layer - 1}"
            )
        if self.hidden <= 0 or self.moe_intermediate <= 0:
            raise ValueError("hidden and moe_intermediate must be positive")
        if (self.group_size, self.bits, self.zero_point) != (
            GROUP_SIZE,
            BITS,
            ZERO_POINT,
        ):
            raise ValueError(
                "unsupported int4 quantization contract: "
                f"group_size={self.group_size}, bits={self.bits}, "
                f"zero_point={self.zero_point}; expected "
                f"{GROUP_SIZE}, {BITS}, {ZERO_POINT}"
            )
        if len(self.plane_offsets) != 6 or len(self.plane_sizes) != 6:
            raise ValueError("the v2 int4 bank must contain exactly six planes")
        running = 0
        for name, offset, size in zip(
            PLANE_NAMES, self.plane_offsets, self.plane_sizes
        ):
            if offset != running:
                raise ValueError(
                    f"{name} offset={offset} is not contiguous expected={running}"
                )
            if offset % ALIGNMENT or size <= 0 or size % ALIGNMENT:
                raise ValueError(
                    f"{name} offset/size must be positive and {ALIGNMENT}-byte aligned: "
                    f"offset={offset}, size={size}"
                )
            if offset > 0xFFFFFFFF or size > 0xFFFFFFFF:
                raise ValueError(f"{name} offset/size exceeds uint32")
            running += size
        if self.data_offset % ALIGNMENT:
            raise ValueError(
                f"data_offset={self.data_offset} is not {ALIGNMENT}-byte aligned"
            )
        if self.expert_stride_bytes != running:
            raise ValueError(
                f"expert_stride_bytes={self.expert_stride_bytes}, expected {running}"
            )
        if self.expert_stride_bytes % ALIGNMENT:
            raise ValueError(
                f"expert_stride_bytes={self.expert_stride_bytes} is not "
                f"{ALIGNMENT}-byte aligned"
            )
        expected_layer = self.experts_per_layer * self.expert_stride_bytes
        if self.layer_stride_bytes != expected_layer:
            raise ValueError(
                f"layer_stride_bytes={self.layer_stride_bytes}, expected {expected_layer}"
            )

    def to_bytes(self) -> bytes:
        self.validate()
        prefix = HEADER_PREFIX.pack(
            MAGIC,
            VERSION,
            self.data_offset,
            self.num_layers,
            self.source_num_layers,
            self.experts_per_layer,
            self.source_experts_per_layer,
            self.resident_set_shared_across_layers,
            self.hidden,
            self.moe_intermediate,
            self.group_size,
            self.bits,
            self.zero_point,
            0,
            0,
            *self.plane_fields,
            self.expert_stride_bytes,
            self.layer_stride_bytes,
        )
        expert_ids = struct.pack(
            f"<{self.experts_per_layer}i", *self.source_expert_ids
        )
        return prefix + expert_ids + bytes(
            self.data_offset - len(prefix) - len(expert_ids)
        )


def parse_int4_bank_header(data: bytes | bytearray | memoryview) -> Int4BankHeader:
    """Parse and validate a complete variable-size v2 header."""
    view = memoryview(data)
    if len(view) < HEADER_PREFIX_SIZE:
        raise ValueError(
            f"int4 bank is shorter than the {HEADER_PREFIX_SIZE}-byte header prefix"
        )
    fields = HEADER_PREFIX.unpack(view[:HEADER_PREFIX_SIZE])
    magic = fields[0]
    version = fields[1]
    data_offset = fields[2]
    experts_per_layer = fields[5]
    if magic != MAGIC:
        raise ValueError(f"bad int4 bank magic: {magic!r}, expected {MAGIC!r}")
    if version != VERSION:
        raise ValueError(
            f"unsupported int4 bank version {version}; this reader supports "
            f"version {VERSION} only"
        )
    expected_data_offset = _align_up(HEADER_PREFIX_SIZE + 4 * experts_per_layer)
    if data_offset != expected_data_offset:
        raise ValueError(
            f"data_offset={data_offset}, expected {expected_data_offset} "
            f"for {experts_per_layer} resident experts"
        )
    if data_offset % ALIGNMENT:
        raise ValueError(
            f"data_offset={data_offset} is not {ALIGNMENT}-byte aligned"
        )
    if len(view) < data_offset:
        raise ValueError(f"truncated int4 header: {len(view)}/{data_offset} bytes")
    if fields[13] != 0 or fields[14] != 0:
        raise ValueError("reserved int4 header fields must be zero")

    plane_fields = fields[15:27]
    plane_offsets = tuple(plane_fields[0::2])
    plane_sizes = tuple(plane_fields[1::2])
    ids_end = HEADER_PREFIX_SIZE + 4 * experts_per_layer
    source_expert_ids = struct.unpack(
        f"<{experts_per_layer}i", view[HEADER_PREFIX_SIZE:ids_end]
    )
    if any(view[ids_end:data_offset]):
        raise ValueError("int4 header alignment padding must be zero")

    header = Int4BankHeader(
        num_layers=fields[3],
        source_num_layers=fields[4],
        source_experts_per_layer=fields[6],
        resident_set_shared_across_layers=fields[7],
        hidden=fields[8],
        moe_intermediate=fields[9],
        group_size=fields[10],
        bits=fields[11],
        zero_point=fields[12],
        plane_offsets=plane_offsets,
        plane_sizes=plane_sizes,
        expert_stride_bytes=fields[27],
        layer_stride_bytes=fields[28],
        source_expert_ids=source_expert_ids,
    )
    header.validate()
    return header


def read_int4_bank_header(path: str | os.PathLike[str]) -> Int4BankHeader:
    """Read only the variable-size header from an int4 bank file."""
    bank_path = Path(path)
    with bank_path.open("rb") as bank:
        prefix = bank.read(HEADER_PREFIX_SIZE)
        if len(prefix) != HEADER_PREFIX_SIZE:
            raise ValueError(
                f"int4 bank is shorter than the {HEADER_PREFIX_SIZE}-byte header prefix"
            )
        fields = HEADER_PREFIX.unpack(prefix)
        data_offset = fields[2]
        if data_offset < HEADER_PREFIX_SIZE or data_offset % ALIGNMENT:
            raise ValueError(f"invalid int4 data_offset: {data_offset}")
        remainder = bank.read(data_offset - HEADER_PREFIX_SIZE)
    return parse_int4_bank_header(prefix + remainder)


__all__ = [
    "ALIGNMENT",
    "BITS",
    "GROUP_SIZE",
    "HEADER_PREFIX",
    "HEADER_PREFIX_FMT",
    "HEADER_PREFIX_SIZE",
    "Int4BankHeader",
    "MAGIC",
    "PLANE_NAMES",
    "RESIDENT_SET_SHARED_ACROSS_LAYERS",
    "VERSION",
    "ZERO_POINT",
    "parse_int4_bank_header",
    "read_int4_bank_header",
]
