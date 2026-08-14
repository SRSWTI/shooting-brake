"""Read expert records out of the Phase-1 NVFP4 bank.

The host tiers used to source their weights from CUDA VRAM, by
``index_select``-ing the layer's tensors with global expert ids. That is
only correct while VRAM still holds every expert, which pre-emptive
allocation deliberately breaks: the offloaded experts are never created
there, so the same call either indexes out of bounds or silently addresses
a different expert. The bank always holds all of them, so it is the source
that survives.

Reading the bank is also *less* work than reading VRAM, because the bank is
written from the raw checkpoint while the VRAM tensors have been through
``prepare_nvfp4_moe_layer_for_fi_or_cutlass``:

* ``w13`` is ``[gate, up]`` here, and ``[up, gate]`` in VRAM on FlashInfer
  backends — no half-swap is needed.
* block scales are linear here, and swizzled in VRAM — no un-swizzling.

One thing the bank does *not* carry is the activation global scale, and the
arena's per-plane scalar has it folded in. See ``global_scale_divisor``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HEADER_FMT = "<8sIIIIIQQQQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
MAGIC = b"SBEXP001"

#: The two fp32 dequant multipliers that close each expert record.
GSCALE_BYTES = 2 * 4


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
