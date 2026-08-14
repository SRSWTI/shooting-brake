"""CPU-only memory-mapped readers for Shooting Brake expert banks.

The legacy :class:`ExpertBank` reader retains the NVFP4 arena layout used by
the existing single-B70 datapath.  :class:`Int4ExpertBank` reads the separate
GPTQ-int4 routed-expert checkpoint format; it exposes packed qweights and fp16
group scales verbatim and never decodes the dense checkpoint's NVFP4 weights.
"""

from __future__ import annotations
from bisect import bisect_left

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from .int4_bank_format import (
    MAGIC as INT4_MAGIC,
    VERSION as INT4_VERSION,
    read_int4_bank_header,
)

HEADER_FMT = "<8sIIIIIQQQQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
MAGIC = b"SBEXP001"

#: The two fp32 dequant multipliers that close each expert record.
GSCALE_BYTES = 2 * 4

#: Semantic name paired with the canonical ``int4_bank_format`` ABI.
INT4_FORMAT = "gptq-int4-group128"


@dataclass(frozen=True)
class ExpertPlanes:
    """One expert's NVFP4 weights, exactly as the arena wants them.

    ``gate``/``up``/``down`` are packed uint8 (two nibbles per byte);
    ``*_sf`` are linear-order e4m3 block scales as raw bytes. The two
    scalars are the record's own ``1 / weight_global_scale`` values, before
    any activation fold.
    """

    gate: np.ndarray
    up: np.ndarray
    down: np.ndarray
    gate_sf: np.ndarray
    up_sf: np.ndarray
    down_sf: np.ndarray
    w13_inv_global: float
    w2_inv_global: float


class ExpertBank:
    """Memory-mapped reader for one expert bank file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with self.path.open("rb") as f:
            raw = f.read(HEADER_SIZE)
        if len(raw) != HEADER_SIZE:
            raise ValueError(f"{self.path} is too small to hold a bank header")
        (magic, self.layers, self.experts_per_layer, self.hidden,
         self.intermediate, _reserved, self.w13_bytes, self.s13_bytes,
         self.w2_bytes, self.s2_bytes) = struct.unpack(HEADER_FMT, raw)
        if magic != MAGIC:
            raise ValueError(f"{self.path} is not a Shooting Brake expert bank")

        self.expert_bytes = (
            self.w13_bytes + self.s13_bytes + self.w2_bytes + self.s2_bytes
            + GSCALE_BYTES
        )
        expected = HEADER_SIZE + self.layers * self.experts_per_layer * self.expert_bytes
        actual = self.path.stat().st_size
        if actual != expected:
            raise ValueError(
                f"{self.path} is {actual} bytes but its header describes "
                f"{expected}; the bank is truncated or from another build"
            )
        # Whole-file mapping, read-only. Pages are faulted in on access, so
        # a 59 GiB bank costs address space rather than resident memory, and
        # only the experts actually read are ever touched.
        self._map = np.memmap(self.path, dtype=np.uint8, mode="r")

    def close(self) -> None:
        self._map = None  # type: ignore[assignment]

    def _record(self, layer: int, expert: int) -> np.ndarray:
        if not (0 <= layer < self.layers):
            raise IndexError(f"layer {layer} outside bank's 0..{self.layers - 1}")
        if not (0 <= expert < self.experts_per_layer):
            raise IndexError(f"expert {expert} outside 0..{self.experts_per_layer - 1}")
        start = HEADER_SIZE + (
            (layer * self.experts_per_layer + expert) * self.expert_bytes
        )
        return self._map[start:start + self.expert_bytes]

    def expert(self, layer: int, expert: int) -> ExpertPlanes:
        """Slice one expert's record into the arena's plane layout."""
        rec = self._record(layer, expert)
        inter, hidden = self.intermediate, self.hidden

        off = 0
        w13 = rec[off:off + self.w13_bytes].reshape(2 * inter, hidden // 2)
        off += self.w13_bytes
        s13 = rec[off:off + self.s13_bytes].reshape(2 * inter, hidden // 16)
        off += self.s13_bytes
        w2 = rec[off:off + self.w2_bytes].reshape(hidden, inter // 2)
        off += self.w2_bytes
        s2 = rec[off:off + self.s2_bytes].reshape(hidden, inter // 16)
        off += self.s2_bytes
        w13_inv, w2_inv = struct.unpack("<ff", rec[off:off + GSCALE_BYTES].tobytes())

        # The bank stores [gate, up]; the arena wants them separately. No
        # swap: unlike VRAM, this ordering never went through
        # `reorder_w1w3_to_w3w1`.
        return ExpertPlanes(
            gate=w13[:inter], up=w13[inter:],
            down=w2,
            gate_sf=s13[:inter], up_sf=s13[inter:],
            down_sf=s2,
            w13_inv_global=w13_inv, w2_inv_global=w2_inv,
        )


@dataclass(frozen=True)
class Int4ExpertPlanes:
    """One routed expert in checkpoint-native GPTQ layout.

    qweights are int32 ``[K/8, N]`` and scales are fp16 ``[K/128, N]``.
    Dequantization is ``(nibble - 8) * scale``; qzeros are intentionally not
    stored because the bank builder verifies every source word is 0x77777777.
    """

    gate_qweight: np.ndarray
    gate_scales: np.ndarray
    up_qweight: np.ndarray
    up_scales: np.ndarray
    down_qweight: np.ndarray
    down_scales: np.ndarray


class Int4ExpertBank:
    """Read-only mapping of an ``SBINT401`` routed-expert bank."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        header = read_int4_bank_header(self.path)
        self.header = header
        self.version = INT4_VERSION
        self.data_offset = header.data_offset
        self.layers = header.num_layers
        self.source_layers = header.source_num_layers
        self.experts_per_layer = header.experts_per_layer
        self.source_experts_per_layer = header.source_experts_per_layer
        self.source_expert_ids = header.source_expert_ids
        self.resident_set_shared_across_layers = (
            header.resident_set_shared_across_layers
        )
        self.hidden = header.hidden
        self.intermediate = header.moe_intermediate
        self.group_size = header.group_size
        self.bits = header.bits
        self.zero_point = header.zero_point
        self._planes = tuple(zip(
            header.plane_offsets, header.plane_sizes, strict=True
        ))
        self.expert_bytes = header.expert_stride_bytes
        self.layer_bytes = header.layer_stride_bytes

        if self.hidden % self.group_size or self.intermediate % self.group_size:
            raise ValueError(
                "int4 bank dimensions must be divisible by group_size"
            )
        expected_sizes = (
            self.hidden // 8 * self.intermediate * 4,
            self.hidden // self.group_size * self.intermediate * 2,
            self.hidden // 8 * self.intermediate * 4,
            self.hidden // self.group_size * self.intermediate * 2,
            self.intermediate // 8 * self.hidden * 4,
            self.intermediate // self.group_size * self.hidden * 2,
        )
        for index, ((offset, size), expected_size) in enumerate(
            zip(self._planes, expected_sizes, strict=True)
        ):
            if size != expected_size:
                raise ValueError(
                    f"int4 plane {index} has {size} bytes, expected {expected_size}"
                )
            if offset + size > self.expert_bytes:
                raise ValueError(f"int4 plane {index} exceeds its expert record")
        expected_file_bytes = self.data_offset + self.layers * self.layer_bytes
        actual_file_bytes = self.path.stat().st_size
        if actual_file_bytes != expected_file_bytes:
            raise ValueError(
                f"{self.path} is {actual_file_bytes} bytes but its header "
                f"describes {expected_file_bytes}"
            )
        self._map = np.memmap(self.path, dtype=np.uint8, mode="r")

    def close(self) -> None:
        self._map = None  # type: ignore[assignment]

    def _compact_expert(self, expert: int) -> int:
        compact = bisect_left(self.source_expert_ids, expert)
        if (
            compact == self.experts_per_layer
            or self.source_expert_ids[compact] != expert
        ):
            raise IndexError(f"expert {expert} is not resident in {self.path}")
        return compact

    def _record(self, layer: int, expert: int) -> np.ndarray:
        if not 0 <= layer < self.layers:
            raise IndexError(f"layer {layer} outside bank's 0..{self.layers - 1}")
        compact = self._compact_expert(expert)
        start = (
            self.data_offset
            + layer * self.layer_bytes
            + compact * self.expert_bytes
        )
        return self._map[start:start + self.expert_bytes]

    @staticmethod
    def _view(
        record: np.ndarray,
        plane: tuple[int, int],
        dtype: str,
        shape: tuple[int, int],
    ) -> np.ndarray:
        offset, size = plane
        return record[offset:offset + size].view(dtype).reshape(shape)

    def expert(self, layer: int, expert: int) -> Int4ExpertPlanes:
        record = self._record(layer, expert)
        hidden, inter, group = self.hidden, self.intermediate, self.group_size
        return Int4ExpertPlanes(
            gate_qweight=self._view(
                record, self._planes[0], "<i4", (hidden // 8, inter)
            ),
            gate_scales=self._view(
                record, self._planes[1], "<f2", (hidden // group, inter)
            ),
            up_qweight=self._view(
                record, self._planes[2], "<i4", (hidden // 8, inter)
            ),
            up_scales=self._view(
                record, self._planes[3], "<f2", (hidden // group, inter)
            ),
            down_qweight=self._view(
                record, self._planes[4], "<i4", (inter // 8, hidden)
            ),
            down_scales=self._view(
                record, self._planes[5], "<f2", (inter // group, hidden)
            ),
        )


def open_expert_bank(path: str | Path) -> ExpertBank | Int4ExpertBank:
    """Open a bank by on-disk magic while preserving legacy reader behavior."""
    with Path(path).open("rb") as file:
        magic = file.read(8)
    if magic == MAGIC:
        return ExpertBank(path)
    if magic == INT4_MAGIC:
        return Int4ExpertBank(path)
    raise ValueError(f"{path} is not a supported Shooting Brake expert bank")


def global_scale_divisor(activation_gscale: float) -> float:
    """The factor between the bank's scalar and the arena's.

    The arena reconstructs ``e2m1(q) * e4m3(sf) * gscale``, and the gscale
    it has always been given is the quant config's ``alpha_or_gscale`` --
    which is ``1 / weight_global_scale`` *divided by* the activation global
    scale, because the CUDA kernel it mirrors quantises activations too.
    Measured on the 35B at layer 16: 54.744x for w13 against a1 = 54.75,
    and 154.02x for w2 against a2 = 154.0.

    The bank stores the undivided value, so the fold has to be reapplied
    here for the host tiers to stay commensurate with the CUDA partials
    they are summed with.

    Note that under pre-emptive allocation this scalar is a reduction over
    the *CUDA-resident* experts, while the arena holds the others. That is
    deliberate: the whole pipeline shares one activation scale, so matching
    it is what keeps the partials addable. It does mean the arena's contents
    depend on the placement, which a byte-for-byte comparison against a
    post-hoc reference cannot detect.
    """
    if activation_gscale <= 0.0:
        raise ValueError(
            f"activation global scale must be positive, got {activation_gscale}"
        )
    return float(activation_gscale)
