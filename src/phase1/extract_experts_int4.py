#!/usr/bin/env python3
"""Build and validate the Shooting Brake AutoGPTQ int4 routed-expert bank.

Only ``model.language_model.layers.*.mlp.experts.*`` is considered.  The vision
and shared-expert tensors are deliberately outside that namespace.  Input is
streamed one layer at a time, grouped by safetensors shard, so residency is
bounded by one routed-expert layer rather than the whole checkpoint.

The v2 header has a fixed 128-byte little-endian prefix
(``<8s26I2Q``), followed by ``experts_per_layer`` explicit int32 source
expert IDs and zero padding to the next 4096-byte boundary:

=======  ====  ========  ================================
offset   size  C type    field
=======  ====  ========  ================================
0        8     char[8]   magic (``SBINT401``)
8        4     uint32    version
12       4     uint32    data_offset
16       4     uint32    num_layers
20       4     uint32    source_num_layers
24       4     uint32    experts_per_layer (resident)
28       4     uint32    source_experts_per_layer
32       4     uint32    resident_set_shared_across_layers
36       4     uint32    hidden
40       4     uint32    moe_intermediate
44       4     uint32    group_size
48       4     uint32    bits
52       4     uint32    zero_point
56       8     2*uint32  reserved (zero)
64       8     2*uint32  gate_qweight offset,size
72       8     2*uint32  gate_scales offset,size
80       8     2*uint32  up_qweight offset,size
88       8     2*uint32  up_scales offset,size
96       8     2*uint32  down_qweight offset,size
104      8     2*uint32  down_scales offset,size
112      8     uint64    expert_stride_bytes
120      8     uint64    layer_stride_bytes
128      4*E   int32[E]  source_expert_ids
128+4E  pad    uint8[]   zeros through ``data_offset`` (4096-byte aligned)
=======  ====  ========  ================================

Payload starts at ``data_offset``.  Records are AoS and directly indexed as
``data_offset + layer * layer_stride_bytes + compact_expert *
expert_stride_bytes``. ``source_expert_ids[compact_expert]`` is the model's
global expert ID. The explicit shared-set flag is 1; a future per-layer-varying
format must use a new version rather than being silently misread. Within each
record the six planes occur in header order at the explicit offsets above.
Qweight is copied verbatim in AutoGPTQ's ``[K/8, N]`` K-major layout; scales
are copied verbatim as fp16 ``[K/128, N]``. Qzeros is not stored: every word is
asserted to be ``0x77777777``, whose AutoGPTQ ``zeros-1`` convention means the
effective zero point in this header is 8.

``--layers N`` selects a prefix of layers. ``--experts`` accepts a prefix
count, ``START:STOP[:STEP]``, or a comma-separated list. All forms resolve to
the explicit sorted source-ID list stored in the header.
``--dry-run`` performs the same reads, zero checks, and writes to ``/dev/null``
without creating a bank, making it useful for measuring full-build peak RSS.

Examples::

  python src/phase1/extract_experts_int4.py --layers 2 --experts 8 \
      --out /tmp/int4-bank.bin --validate --nvfp4-model-dir /path/to/nvfp4
  /usr/bin/time -v python src/phase1/extract_experts_int4.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import BinaryIO

PHASE4_SRC = Path(__file__).resolve().parents[1] / "phase4" / "src"
if str(PHASE4_SRC) not in sys.path:
    sys.path.insert(0, str(PHASE4_SRC))

from shooting_brake_vllm.int4_bank_format import (  # noqa: E402
    ALIGNMENT,
    BITS,
    GROUP_SIZE,
    Int4BankHeader,
    ZERO_POINT,
    read_int4_bank_header,
)

import torch
from safetensors import safe_open

HF_HUB = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
DEFAULT_MODEL_DIR = (
    HF_HUB
    / "models--srswti--axe-superveloce-88b-int4"
    / "snapshots"
    / "ef995883990337362a07074885149e3e51c3fed8"
)
DEFAULT_NVFP4_MODEL_DIR = (
    HF_HUB
    / "models--srswti--axe-superveloce-88b-nvfp4a16"
    / "snapshots"
    / "88dedc71dc874f2c5727cd4329e694c27ec7963d"
)

QZERO_WORD = 0x77777777      # AutoGPTQ v1 / auto-round: stores zero_point - 1
QZERO_WORD_V2 = 0x88888888   # GPTQModel v2: stores zero_point directly
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
SUFFIXES = ("qweight", "scales", "qzeros")
EXPERT_KEY = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.(qweight|scales|qzeros)$"
)
SAMPLE_RE = re.compile(r"^(\d+):(\d+):(gate|up|down)(?:_proj)?$")
E2M1_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


class BankShape:
    """Checkpoint geometry and compact resident expert selection."""

    def __init__(
        self,
        hidden: int,
        intermediate: int,
        source_layers: int,
        source_experts: int,
        num_layers: int,
        expert_ids: list[int],
    ) -> None:
        self.hidden = hidden
        self.intermediate = intermediate
        self.source_layers = source_layers
        self.source_experts = source_experts
        self.num_layers = num_layers
        self.expert_ids = expert_ids
        self.experts = len(expert_ids)

        if hidden % GROUP_SIZE or intermediate % GROUP_SIZE:
            raise SystemExit(
                f"hidden={hidden} and moe_intermediate={intermediate} must both "
                f"be divisible by int4 group_size={GROUP_SIZE}"
            )
        if not 1 <= num_layers <= source_layers:
            raise SystemExit(
                f"--layers must be in 1..{source_layers}, got {num_layers}"
            )
        if not expert_ids:
            raise SystemExit("--experts selected no experts")
        if expert_ids != sorted(set(expert_ids)):
            raise SystemExit(
                "--experts must be strictly increasing with no duplicates"
            )
        if expert_ids[0] < 0 or expert_ids[-1] >= source_experts:
            raise SystemExit(
                f"--experts IDs must be in 0..{source_experts - 1}, got "
                f"{expert_ids[0]}..{expert_ids[-1]}"
            )
        self._expert_positions = {
            source_expert: position
            for position, source_expert in enumerate(expert_ids)
        }

        gate_q = self.qweight_bytes(hidden, intermediate)
        gate_s = self.scale_bytes(hidden, intermediate)
        down_q = self.qweight_bytes(intermediate, hidden)
        down_s = self.scale_bytes(intermediate, hidden)
        self.plane_sizes = (gate_q, gate_s, gate_q, gate_s, down_q, down_s)
        self.expert_stride = sum(self.plane_sizes)
        self.layer_stride = self.experts * self.expert_stride

        if any(offset % ALIGNMENT for offset in self.plane_offsets):
            raise SystemExit(
                f"int4 plane offsets are not {ALIGNMENT}-byte aligned: "
                f"{self.plane_offsets}"
            )
        if self.expert_stride % ALIGNMENT or self.layer_stride % ALIGNMENT:
            raise SystemExit("int4 expert/layer strides violate bank alignment")
        if any(value > 0xFFFFFFFF for value in (*self.plane_offsets, *self.plane_sizes)):
            raise SystemExit("int4 plane offset/size exceeds the uint32 header ABI")

    @staticmethod
    def qweight_bytes(k: int, n: int) -> int:
        return (k // 8) * n * torch.int32.itemsize

    @staticmethod
    def scale_bytes(k: int, n: int) -> int:
        return (k // GROUP_SIZE) * n * torch.float16.itemsize

    @property
    def plane_offsets(self) -> tuple[int, ...]:
        offsets: list[int] = []
        running = 0
        for size in self.plane_sizes:
            offsets.append(running)
            running += size
        return tuple(offsets)


    @property
    def bank_header(self) -> Int4BankHeader:
        return Int4BankHeader(
            num_layers=self.num_layers,
            source_num_layers=self.source_layers,
            source_experts_per_layer=self.source_experts,
            hidden=self.hidden,
            moe_intermediate=self.intermediate,
            group_size=GROUP_SIZE,
            bits=BITS,
            zero_point=ZERO_POINT,
            plane_offsets=self.plane_offsets,
            plane_sizes=self.plane_sizes,
            expert_stride_bytes=self.expert_stride,
            layer_stride_bytes=self.layer_stride,
            source_expert_ids=tuple(self.expert_ids),
        )

    @property
    def data_offset(self) -> int:
        return self.bank_header.data_offset

    @property
    def total_bytes(self) -> int:
        return self.data_offset + self.num_layers * self.layer_stride

    def header(self) -> bytes:
        return self.bank_header.to_bytes()

    def describe(self) -> str:
        if self.experts <= 12:
            selection = ",".join(map(str, self.expert_ids))
        else:
            selection = (
                ",".join(map(str, self.expert_ids[:6]))
                + ",...,"
                + ",".join(map(str, self.expert_ids[-3:]))
            )
        return (
            f"  layers       {self.num_layers}/{self.source_layers} "
            f"(prefix 0..{self.num_layers - 1})\n"
            f"  experts      {self.experts}/{self.source_experts}/layer "
            f"(source IDs {selection})\n"
            f"  hidden       {self.hidden}\n"
            f"  intermediate {self.intermediate}\n"
            f"  quantization int4 g{GROUP_SIZE}, zero_point={ZERO_POINT}\n"
            f"  data offset  {self.data_offset} bytes\n"
            f"  per expert   {self.expert_stride / 2**20:.2f} MiB\n"
            f"  bank total   {self.total_bytes / 2**30:.2f} GiB"
        )

    def projection_shape(self, projection: str) -> tuple[int, int]:
        if projection in ("gate_proj", "up_proj"):
            return self.hidden, self.intermediate
        if projection == "down_proj":
            return self.intermediate, self.hidden
        raise ValueError(f"unknown projection: {projection}")

    def projection_plane_indices(self, projection: str) -> tuple[int, int]:
        if projection == "gate_proj":
            return 0, 1
        if projection == "up_proj":
            return 2, 3
        if projection == "down_proj":
            return 4, 5
        raise ValueError(f"unknown projection: {projection}")

    def expert_position(self, source_expert: int) -> int:
        try:
            return self._expert_positions[source_expert]
        except KeyError as exc:
            raise ValueError(f"source expert {source_expert} is not resident") from exc


Sample = tuple[int, int, str]


def read_config(model_dir: Path) -> tuple[dict, dict]:
    cfg = json.loads((model_dir / "config.json").read_text())
    return cfg.get("text_config", cfg), cfg.get("quantization_config", {})


def index_shards(model_dir: Path) -> tuple[list[str], dict[str, str]]:
    """Read the intact HF index without loading any tensor payload."""
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise SystemExit(f"missing safetensors index: {index_path}")
    document = json.loads(index_path.read_text())
    weight_map = document.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise SystemExit(f"invalid or empty weight_map in {index_path}")

    index: dict[str, str] = {}
    shards: set[str] = set()
    for key, filename in weight_map.items():
        path = model_dir / filename
        if not path.is_file():
            raise SystemExit(f"index names a missing shard: {path}")
        index[key] = str(path)
        shards.add(str(path))
    return sorted(shards), index


def _quant_field(quant_cfg: dict, name: str, default=None):
    value = quant_cfg.get(name, default)
    return value


def parse_expert_ids(spec: str | None, source_experts: int) -> list[int]:
    """Parse COUNT, START:STOP[:STEP], or a comma-separated source-ID list."""
    if spec is None:
        return list(range(source_experts))
    if spec.isdecimal():
        count = int(spec)
        if not 1 <= count <= source_experts:
            raise SystemExit(
                f"--experts count must be in 1..{source_experts}, got {count}"
            )
        return list(range(count))
    if "," in spec:
        parts = spec.split(",")
        if any(not part.isdecimal() for part in parts):
            raise SystemExit(
                f"--experts comma-list must contain non-negative integer IDs: {spec}"
            )
        return [int(part) for part in parts]

    parts = spec.split(":")
    if len(parts) not in (2, 3) or any(not part.isdecimal() for part in parts):
        raise SystemExit(
            "--experts must be COUNT, START:STOP[:STEP], or comma-list; "
            "for example 8, 0:180:2, or 0,4,9"
        )
    start, stop = int(parts[0]), int(parts[1])
    step = int(parts[2]) if len(parts) == 3 else 1
    if step <= 0 or stop > source_experts:
        raise SystemExit(
            f"--experts range must have STEP>0 and STOP<={source_experts}, got {spec}"
        )
    selected = list(range(start, stop, step))
    if not selected:
        raise SystemExit(f"--experts range selects no experts: {spec}")
    return selected


def discover_shape(
    model_dir: Path,
    index: dict[str, str],
    layer_limit: int | None,
    expert_spec: str | None,
) -> BankShape:
    """Validate the checkpoint contract and derive the selected bank shape."""
    cfg, quant_cfg = read_config(model_dir)
    hidden = int(cfg["hidden_size"])
    intermediate = int(cfg["moe_intermediate_size"])
    configured_layers = int(cfg["num_hidden_layers"])
    configured_experts = int(cfg["num_experts"])

    bits = int(_quant_field(quant_cfg, "bits", -1))
    group_size = int(_quant_field(quant_cfg, "group_size", -1))
    packing = str(_quant_field(quant_cfg, "packing_format", ""))
    symmetric = bool(_quant_field(quant_cfg, "sym", False))
    quant_method = str(_quant_field(quant_cfg, "quant_method", ""))
    desc_act = bool(_quant_field(quant_cfg, "desc_act", False))
    # Two admitted checkpoint families, one bank contract:
    #   auto-round (88B: packing_format "auto_round:auto_gptq"), and plain
    #   first-party GPTQ (122B: quant_method "gptq", desc_act False -- g_idx
    #   tensors exist but must be the trivial ramp; spot-asserted at load).
    auto_round_ok = packing == "auto_round:auto_gptq"
    plain_gptq_ok = quant_method == "gptq" and not desc_act
    if (bits, group_size, symmetric) != (BITS, GROUP_SIZE, True) or not (
        auto_round_ok or plain_gptq_ok
    ):
        raise SystemExit(
            "checkpoint quantization does not match the int4 bank contract: "
            f"bits={bits}, group_size={group_size}, sym={symmetric}, "
            f"packing_format={packing!r}, quant_method={quant_method!r}, "
            f"desc_act={desc_act}"
        )

    seen_layers: set[int] = set()
    seen_experts: set[int] = set()
    for key in index:
        match = EXPERT_KEY.match(key)
        if match:
            seen_layers.add(int(match.group(1)))
            seen_experts.add(int(match.group(2)))
    expected_layers = set(range(configured_layers))
    expected_experts = set(range(configured_experts))
    if seen_layers != expected_layers:
        raise SystemExit(
            "routed-expert layers in index do not match config: "
            f"found={sorted(seen_layers)}, expected=0..{configured_layers - 1}"
        )
    if seen_experts != expected_experts:
        raise SystemExit(
            "routed expert IDs in index do not match config: "
            f"found={sorted(seen_experts)}, expected=0..{configured_experts - 1}"
        )

    return BankShape(
        hidden,
        intermediate,
        configured_layers,
        configured_experts,
        configured_layers if layer_limit is None else layer_limit,
        parse_expert_ids(expert_spec, configured_experts),
    )


def tensor_key(layer: int, expert: int, projection: str, suffix: str) -> str:
    return (
        f"model.language_model.layers.{layer}.mlp.experts.{expert}."
        f"{projection}.{suffix}"
    )


def load_layer(
    layer: int,
    shape: BankShape,
    index: dict[str, str],
) -> dict[str, torch.Tensor]:
    """Read one selected layer, grouping keys so every shard is opened once."""
    wanted: dict[str, list[str]] = defaultdict(list)
    for expert in shape.expert_ids:
        for projection in PROJECTIONS:
            for suffix in SUFFIXES:
                key = tensor_key(layer, expert, projection, suffix)
                shard = index.get(key)
                if shard is None:
                    raise SystemExit(f"missing tensor: {key}")
                wanted[shard].append(key)

    tensors: dict[str, torch.Tensor] = {}
    for shard, keys in wanted.items():
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in keys:
                tensors[key] = handle.get_tensor(key)
    return tensors


def _require_tensor(
    tensor: torch.Tensor,
    key: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    if tuple(tensor.shape) != shape or tensor.dtype != dtype:
        raise SystemExit(
            f"unexpected tensor contract for {key}: shape={tuple(tensor.shape)} "
            f"dtype={tensor.dtype}, expected shape={shape} dtype={dtype}"
        )
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise SystemExit(f"{key} must be a contiguous CPU tensor; refusing to repack it")


def assert_qzeros(qzeros: torch.Tensor, key: str, expected_shape: tuple[int, int]) -> None:
    """Reject any stored zero point whose effective value is not 8.

    Two writer conventions exist for symmetric 4-bit, both meaning
    effective zero_point=8 and therefore byte-identical qweight semantics:

    * AutoGPTQ v1 / auto-round stores ``zero_point - 1``: 0x77777777 (88B).
    * GPTQModel v2 (first-party Qwen 122B) removed the off-by-one and
      stores the zero point directly: 0x88888888.

    Every word in a tensor must match ONE of the two; anything else (or a
    mix) means an asymmetric or foreign layout and the bank must refuse.
    The cross-format NVFP4 check (cosine > 0.97 band) independently
    confirms the interpretation on real weights.
    """
    _require_tensor(qzeros, key, expected_shape, torch.int32)
    for word in (QZERO_WORD, QZERO_WORD_V2):
        if bool((qzeros == word).all().item()):
            return
    mismatch = (qzeros != QZERO_WORD) & (qzeros != QZERO_WORD_V2)
    flat_index = int(mismatch.flatten().nonzero()[0].item())
    actual = int(qzeros.flatten()[flat_index].item()) & 0xFFFFFFFF
    raise SystemExit(
        f"FATAL qzeros contract violation in {key}: word[{flat_index}]="
        f"0x{actual:08x}, expected uniform 0x{QZERO_WORD:08x} (v1) or "
        f"0x{QZERO_WORD_V2:08x} (v2); this bank format requires effective "
        f"zero_point={ZERO_POINT}"
    )


def assert_g_idx_trivial(
    index: dict[str, str],
    shape: BankShape,
) -> None:
    """Spot-assert that g_idx (when present) is the identity ramp.

    desc_act=False promises no activation-order permutation, but the bank
    copies qweight verbatim -- a non-trivial g_idx would silently permute
    the K dimension out from under both the B70 kernel and Marlin. One
    tensor per projection suffices: desc_act is a config-global property.
    """
    layer = shape.layer_ids[0] if hasattr(shape, "layer_ids") else 0
    expert = shape.expert_ids[0]
    for projection in PROJECTIONS:
        key = tensor_key(layer, expert, projection, "g_idx")
        shard = index.get(key)
        if shard is None:
            return  # auto-round checkpoints carry no g_idx at all
        with safe_open(shard, framework="pt", device="cpu") as handle:
            g_idx = handle.get_tensor(key)
        k = int(g_idx.numel())
        expected = torch.arange(k, dtype=g_idx.dtype) // GROUP_SIZE
        if not torch.equal(g_idx, expected):
            raise SystemExit(
                f"FATAL g_idx contract violation in {key}: not the trivial "
                f"ramp despite desc_act=False; refusing to copy qweight "
                f"verbatim under an activation-order permutation"
            )
    print("g_idx spot check: trivial ramp on all three projections")


class CountingDevNull:
    """A counted /dev/null sink that faults one byte from every input page."""

    def __init__(self) -> None:
        self._out: BinaryIO | None = None
        self._position = 0
        self._page_probe = 0

    def __enter__(self) -> "CountingDevNull":
        self._out = open(os.devnull, "wb")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        assert self._out is not None
        self._out.close()
        self._out = None

    def write(self, payload) -> int:
        assert self._out is not None
        # Linux /dev/null may accept an iovec without faulting its user pages.
        # Probe one byte per page so --dry-run measures the resident working set
        # of a real bank write rather than only the tensor mapping metadata.
        self._page_probe ^= sum(memoryview(payload)[::4096]) & 0xFF
        written = self._out.write(payload)
        self._position += written
        return written

    def tell(self) -> int:
        return self._position


def _write_tensor_bytes(out: BinaryIO, tensor: torch.Tensor, key: str) -> int:
    """Write a contiguous CPU tensor through the buffer protocol, without repacking."""
    payload = memoryview(tensor.numpy()).cast("B")
    written = out.write(payload)
    if written != payload.nbytes:
        raise SystemExit(f"short write for {key}: wrote {written}/{payload.nbytes} bytes")
    return written


def write_layer(
    out: BinaryIO,
    layer: int,
    shape: BankShape,
    tensors: dict[str, torch.Tensor],
) -> None:
    for expert in shape.expert_ids:
        start = out.tell()
        for projection in PROJECTIONS:
            k, n = shape.projection_shape(projection)
            qkey = tensor_key(layer, expert, projection, "qweight")
            skey = tensor_key(layer, expert, projection, "scales")
            zkey = tensor_key(layer, expert, projection, "qzeros")
            qweight = tensors[qkey]
            scales = tensors[skey]
            qzeros = tensors[zkey]

            _require_tensor(qweight, qkey, (k // 8, n), torch.int32)
            _require_tensor(scales, skey, (k // GROUP_SIZE, n), torch.float16)
            assert_qzeros(qzeros, zkey, (k // GROUP_SIZE, n // 8))
            _write_tensor_bytes(out, qweight, qkey)
            _write_tensor_bytes(out, scales, skey)

        record_bytes = out.tell() - start
        if record_bytes != shape.expert_stride:
            raise SystemExit(
                f"record size mismatch at layer={layer} expert={expert}: "
                f"wrote {record_bytes}, expected {shape.expert_stride}"
            )
def layer_zero_nibble_fraction(
    layer: int,
    shape: BankShape,
    tensors: dict[str, torch.Tensor],
) -> tuple[int, int]:
    """Count packed int4 zeros for one resident routed-expert layer."""
    zero_nibbles = 0
    total_nibbles = 0
    for expert in shape.expert_ids:
        for projection in PROJECTIONS:
            key = tensor_key(layer, expert, projection, "qweight")
            qweight = tensors[key]
            packed = qweight.view(torch.uint8)
            zero_nibbles += int(((packed & 0xF) == ZERO_POINT).sum().item())
            zero_nibbles += int(((packed >> BITS) == ZERO_POINT).sum().item())
            total_nibbles += packed.numel() * 2
    return zero_nibbles, total_nibbles


def write_sparsity_csv(
    path: Path,
    rows: list[tuple[int, int, int]],
) -> None:
    """Atomically publish a per-layer packed-zero profile."""
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path.with_suffix(path.suffix + ".tmp")
    try:
        with target.open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(("layer", "zero_nibbles", "total_nibbles", "zero_fraction"))
            for layer, zero_nibbles, total_nibbles in rows:
                writer.writerow(
                    (layer, zero_nibbles, total_nibbles, f"{zero_nibbles / total_nibbles:.9f}")
                )
        os.replace(target, path)
    except BaseException:
        target.unlink(missing_ok=True)
        raise




def dequantize_int4(qweight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Return logical ``[K, N]`` fp32 weights: ``(nibble - 8) * scale``.

    ``qweight`` must be int32 ``[K/8, N]``.  Nibble ``k % 8`` occupies bits
    ``4*(k%8)..4*(k%8)+3`` of word row ``k//8``.  ``scales`` is fp16
    ``[K/128, N]`` and is intentionally not absolutized: negative scales are
    valid checkpoint data.
    """
    if qweight.dtype != torch.int32 or qweight.ndim != 2:
        raise ValueError("qweight must be a rank-2 int32 tensor")
    if scales.dtype != torch.float16 or scales.ndim != 2:
        raise ValueError("scales must be a rank-2 float16 tensor")
    k = qweight.shape[0] * 8
    n = qweight.shape[1]
    if tuple(scales.shape) != (k // GROUP_SIZE, n):
        raise ValueError(
            f"scales shape {tuple(scales.shape)} does not match qweight "
            f"shape {tuple(qweight.shape)} for group_size={GROUP_SIZE}"
        )

    shifts = (torch.arange(8, dtype=torch.int32) * BITS).view(8, 1, 1)
    nibbles = (
        ((qweight.unsqueeze(0) >> shifts) & 0xF)
        .permute(1, 0, 2)
        .reshape(k, n)
    )
    expanded_scales = scales.float().repeat_interleave(GROUP_SIZE, dim=0)
    return (nibbles.float() - ZERO_POINT) * expanded_scales


def _read_bank_projection(
    bank_path: Path,
    shape: BankShape,
    layer: int,
    expert: int,
    projection: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    qplane, splane = shape.projection_plane_indices(projection)
    k, n = shape.projection_shape(projection)
    compact_expert = shape.expert_position(expert)
    record = (
        shape.data_offset
        + layer * shape.layer_stride
        + compact_expert * shape.expert_stride
    )
    qoffset = record + shape.plane_offsets[qplane]
    soffset = record + shape.plane_offsets[splane]

    with bank_path.open("rb") as bank:
        bank.seek(qoffset)
        qraw = bank.read(shape.plane_sizes[qplane])
        bank.seek(soffset)
        sraw = bank.read(shape.plane_sizes[splane])
    if len(qraw) != shape.plane_sizes[qplane] or len(sraw) != shape.plane_sizes[splane]:
        raise SystemExit(
            f"truncated bank record at layer={layer} expert={expert} projection={projection}"
        )
    qweight = torch.frombuffer(bytearray(qraw), dtype=torch.int32).reshape(k // 8, n)
    scales = torch.frombuffer(bytearray(sraw), dtype=torch.float16).reshape(
        k // GROUP_SIZE, n
    )
    return qweight, scales


def _validate_header(bank_path: Path, shape: BankShape) -> None:
    try:
        actual_header = read_int4_bank_header(bank_path)
    except ValueError as exc:
        raise SystemExit(f"invalid int4 bank header in {bank_path}: {exc}") from exc
    expected_header = shape.bank_header
    if actual_header != expected_header:
        raise SystemExit(
            f"bank header does not match selected checkpoint geometry:\n"
            f"  actual={actual_header}\n"
            f"  expected={expected_header}"
        )
    actual_size = bank_path.stat().st_size
    if actual_size != shape.total_bytes:
        raise SystemExit(
            f"bank size mismatch: {actual_size} bytes, expected {shape.total_bytes}"
        )


def parse_sample(value: str) -> Sample:
    match = SAMPLE_RE.match(value)
    if not match:
        raise argparse.ArgumentTypeError(
            "sample must be LAYER:EXPERT:gate|up|down (for example 0:0:gate)"
        )
    return int(match.group(1)), int(match.group(2)), f"{match.group(3)}_proj"


def default_samples(shape: BankShape) -> list[Sample]:
    return [
        (0, shape.expert_ids[0], "gate_proj"),
        (
            shape.num_layers // 2,
            shape.expert_ids[shape.experts // 2],
            "up_proj",
        ),
        (shape.num_layers - 1, shape.expert_ids[-1], "down_proj"),
    ]


def validate_samples(shape: BankShape, samples: list[Sample]) -> None:
    for layer, expert, projection in samples:
        if not 0 <= layer < shape.num_layers:
            raise SystemExit(
                f"sample layer {layer} is outside selected bank 0..{shape.num_layers - 1}"
            )
        if expert not in shape.expert_ids:
            raise SystemExit(
                f"sample expert {expert} is not resident; selected source IDs are "
                f"{shape.expert_ids}"
            )
        if projection not in PROJECTIONS:
            raise SystemExit(f"invalid sample projection: {projection}")


def _load_projection(
    model_dir: Path,
    index: dict[str, str],
    layer: int,
    expert: int,
    projection: str,
    suffixes: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for suffix in suffixes:
        key = tensor_key(layer, expert, projection, suffix)
        shard = index.get(key)
        if shard is None:
            raise SystemExit(f"missing tensor: {key}")
        with safe_open(shard, framework="pt", device="cpu") as handle:
            result[suffix] = handle.get_tensor(key)
    return result


def validate_bank_against_shards(
    bank_path: Path,
    model_dir: Path,
    shape: BankShape,
    index: dict[str, str],
    samples: list[Sample],
) -> None:
    """Prove sampled bank dequantization is bit-exact with original shards."""
    del model_dir  # the absolute shard index is the authority after discovery
    _validate_header(bank_path, shape)
    print("\nBank-vs-shard validation:")
    for layer, expert, projection in samples:
        bank_q, bank_s = _read_bank_projection(
            bank_path, shape, layer, expert, projection
        )
        shard = _load_projection(
            Path(), index, layer, expert, projection, ("qweight", "scales")
        )
        shard_q = shard["qweight"]
        shard_s = shard["scales"]
        q_bytes_equal = torch.equal(bank_q.view(torch.uint8), shard_q.view(torch.uint8))
        s_bytes_equal = torch.equal(bank_s.view(torch.uint8), shard_s.view(torch.uint8))
        if not q_bytes_equal or not s_bytes_equal:
            raise SystemExit(
                f"bank payload differs from shard at layer={layer} expert={expert} "
                f"projection={projection}: qweight={q_bytes_equal}, scales={s_bytes_equal}"
            )
        bank_dequant = dequantize_int4(bank_q, bank_s)
        shard_dequant = dequantize_int4(shard_q, shard_s)
        if not torch.equal(bank_dequant.view(torch.uint8), shard_dequant.view(torch.uint8)):
            raise SystemExit(
                f"dequant is not bit-exact at layer={layer} expert={expert} "
                f"projection={projection}"
            )
        print(
            f"  layer={layer} expert={expert} projection={projection}: "
            "qweight bytes exact, scales bytes exact, dequant bit-exact"
        )
    print(f"Bank-vs-shard validation PASSED ({len(samples)} sampled tensors)")


def dequantize_nvfp4(
    packed: torch.Tensor,
    block_scales: torch.Tensor,
    global_scale: torch.Tensor,
    k: int,
    n: int,
    convention: str = "multiplier",
) -> torch.Tensor:
    """Decode linear NVFP4 into logical fp32 ``[K, N]`` weights.

    Two global-scale conventions exist and confusing them corrupts every
    weight by global_scale**2 (code-verified in both sources):

    * ``multiplier`` -- ModelOpt W4A16 (88B): ``weight_scale_2`` stores
      amax/(6*448), a direct multiplier: ``w = codes * scale * gs``.
    * ``divisor`` -- llm-compressor/compressed-tensors (122B):
      ``weight_global_scale`` was MULTIPLIED into the e4m3 block scales at
      quantization time (compressed_tensors/quantization/utils/helpers.py:102,
      ``scales = global_scale * scales``), so dequant divides it back out:
      ``w = codes * scale / gs``.
    """
    if packed.dtype != torch.uint8 or tuple(packed.shape) != (n, k // 2):
        raise ValueError(
            f"unexpected NVFP4 packed shape/dtype: {tuple(packed.shape)} {packed.dtype}; "
            f"expected {(n, k // 2)} torch.uint8"
        )
    if block_scales.numel() != n * (k // 16):
        raise ValueError(
            f"unexpected NVFP4 block-scale size: {tuple(block_scales.shape)}; "
            f"expected {n * (k // 16)} elements"
        )
    if global_scale.numel() != 1:
        raise ValueError(f"NVFP4 global scale must be scalar, got {global_scale.shape}")
    if convention not in ("multiplier", "divisor"):
        raise ValueError(f"unknown NVFP4 global-scale convention: {convention}")

    low = (packed & 0xF).long()
    high = (packed >> 4).long()
    codes = torch.stack((low, high), dim=-1).reshape(n, k)
    values = E2M1_TABLE[codes]
    scales = block_scales.reshape(n, k // 16).float().repeat_interleave(16, dim=1)
    gs = global_scale.float()
    factor = gs if convention == "multiplier" else 1.0 / gs
    return (values * scales * factor).T.contiguous()


def _nvfp4_key(layer: int, expert: int, projection: str, suffix: str) -> str:
    return tensor_key(layer, expert, projection, suffix)


def cross_validate_nvfp4(
    nvfp4_model_dir: Path,
    int4_index: dict[str, str],
    shape: BankShape,
    samples: list[Sample],
) -> None:
    """Report independent NVFP4/int4 agreement without weakening int4 validity."""
    _, nv_index = index_shards(nvfp4_model_dir)
    print("\nCross-format int4 vs NVFP4 validation:")
    all_expected = True
    reader_suspect = False
    for layer, expert, projection in samples:
        int4 = _load_projection(
            Path(), int4_index, layer, expert, projection, ("qweight", "scales")
        )
        int4_weight = dequantize_int4(int4["qweight"], int4["scales"])

        # Variant detection: ModelOpt ships weight/weight_scale/weight_scale_2
        # (multiplier); compressed-tensors ships weight_packed/weight_scale/
        # weight_global_scale (divisor). Probe the index rather than trusting
        # config shape.
        modelopt = _nvfp4_key(layer, expert, projection, "weight_scale_2") in nv_index
        if modelopt:
            suffixes = ("weight", "weight_scale", "weight_scale_2")
            convention = "multiplier"
        else:
            suffixes = ("weight_packed", "weight_scale", "weight_global_scale")
            convention = "divisor"
        packed_key, scale_key, gs_key = suffixes
        if _nvfp4_key(layer, expert, projection, packed_key) not in nv_index:
            # Mixed-precision checkpoints keep some layers off NVFP4 (the
            # 122B stores layer 47's experts as FP8). Nothing to cross
            # against; the bit-exact bank validation already covered it.
            print(
                f"  layer={layer} expert={expert} projection={projection}: "
                f"no NVFP4 counterpart (mixed-precision layer) — SKIPPED"
            )
            continue
        nv: dict[str, torch.Tensor] = {}
        for suffix in suffixes:
            key = _nvfp4_key(layer, expert, projection, suffix)
            shard = nv_index.get(key)
            if shard is None:
                raise SystemExit(f"missing NVFP4 tensor: {key}")
            with safe_open(shard, framework="pt", device="cpu") as handle:
                nv[suffix] = handle.get_tensor(key)

        k, n = shape.projection_shape(projection)
        raw_scale = nv["weight_scale"].float()
        floor_fraction = (raw_scale == 2.0**-9).float().mean().item()
        ceiling_fraction = (raw_scale == 448.0).float().mean().item()
        nonfinite_fraction = (~torch.isfinite(raw_scale)).float().mean().item()
        negative_fraction = (raw_scale < 0).float().mean().item()
        sample_reader_suspect = (
            tuple(nv["weight_scale"].shape) != (n, k // 16)
            or nonfinite_fraction > 0.0
            or negative_fraction > 0.0
            or (floor_fraction > 0.25 and ceiling_fraction > 0.01)
        )
        reader_suspect |= sample_reader_suspect

        try:
            nv_weight = dequantize_nvfp4(
                nv[packed_key], nv[scale_key], nv[gs_key], k, n,
                convention=convention,
            )
        except (ValueError, RuntimeError) as exc:
            print(
                f"  layer={layer} expert={expert} projection={projection}: "
                f"NVFP4 READER SUSPECT ({exc})"
            )
            reader_suspect = True
            all_expected = False
            continue

        delta = int4_weight - nv_weight
        relative_l2 = (
            torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(nv_weight)
        ).item()
        cosine = torch.nn.functional.cosine_similarity(
            int4_weight.flatten(), nv_weight.flatten(), dim=0
        ).item()
        expected = 0.10 <= relative_l2 <= 0.20 and cosine > 0.97
        all_expected &= expected
        status = "CONFIRMS LAYOUT" if expected else "POSSIBLE LAYOUT ERROR"
        if sample_reader_suspect:
            status = "NVFP4 READER SUSPECT; INT4 RESULT UNAFFECTED"
        print(
            f"  layer={layer} expert={expert} projection={projection}: "
            f"relative_L2={relative_l2:.6f}, cosine={cosine:.6f}, "
            f"scale_floor_fraction={floor_fraction:.6f}, "
            f"scale_448_fraction={ceiling_fraction:.6f} — {status}"
        )

    if reader_suspect:
        print(
            "Cross-format WARNING: NVFP4 scale storage looks swizzled or otherwise "
            "implausible. The NVFP4 reader is suspect; this does not fail the "
            "independent bit-exact int4 bank validation."
        )
    elif all_expected:
        print(
            f"Cross-format check CONFIRMS the int4 layout on {len(samples)} sampled tensors "
            "(all relative L2 in 10-20% and cosine > 0.97)."
        )
    else:
        print(
            "Cross-format POSSIBLE LAYOUT ERROR: at least one sample is outside "
            "relative L2 10-20% or cosine > 0.97. The bit-exact int4 validation "
            "remains authoritative for bank losslessness."
        )


def peak_rss_mib() -> float:
    # Linux reports ru_maxrss in KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(os.environ.get("SB_INT4_MODEL_DIR", DEFAULT_MODEL_DIR)),
        help="AutoGPTQ int4 HF snapshot directory",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "expert_bank_int4.bin",
        help="destination bank file",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=None,
        metavar="N",
        help="build the prefix of N layers (default: all)",
    )
    parser.add_argument(
        "--experts",
        type=str,
        default=None,
        metavar="COUNT|RANGE|LIST",
        help="resident expert prefix, START:STOP[:STEP], or sorted comma-list",
    )
    parser.add_argument(
        "--sparsity-csv",
        type=Path,
        default=None,
        help="write per-layer fraction of packed int4 nibbles equal to zero_point",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="stream and validate the full selection to /dev/null; create no bank",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="validate sampled bank tensors against the original int4 shards",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate an existing --out without rebuilding it",
    )
    parser.add_argument(
        "--sample",
        action="append",
        type=parse_sample,
        default=None,
        metavar="L:E:PROJ",
        help="validation sample; repeatable (default: first/middle/last)",
    )
    parser.add_argument(
        "--nvfp4-model-dir",
        type=Path,
        default=None,
        help="also compare samples with the independently quantized NVFP4 checkpoint",
    )
    args = parser.parse_args()

    if args.dry_run and args.validate_only:
        parser.error("--dry-run and --validate-only are mutually exclusive")
    if args.validate_only:
        args.validate = True
    if args.nvfp4_model_dir is not None:
        args.validate = True
    if not args.model_dir.is_dir():
        raise SystemExit(f"model dir not found: {args.model_dir}")
    if args.nvfp4_model_dir is not None and not args.nvfp4_model_dir.is_dir():
        raise SystemExit(f"NVFP4 model dir not found: {args.nvfp4_model_dir}")

    shards, index = index_shards(args.model_dir)
    print(f"Shards: {len(shards)}")
    shape = discover_shape(args.model_dir, index, args.layers, args.experts)
    print(shape.describe())
    assert_g_idx_trivial(index, shape)
    samples = args.sample if args.sample is not None else default_samples(shape)
    validate_samples(shape, samples)

    sparsity_rows: list[tuple[int, int, int]] = []

    if not args.validate_only:
        target = Path(os.devnull) if args.dry_run else args.out.with_suffix(args.out.suffix + ".tmp")
        if not args.dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
        action = "Dry-run streaming to /dev/null" if args.dry_run else f"Writing {target}"
        print(f"\n{action} ...")
        started = time.perf_counter()
        try:
            stream = CountingDevNull() if args.dry_run else target.open("wb")
            with stream as out:
                out.write(shape.header())
                for layer in range(shape.num_layers):
                    layer_started = time.perf_counter()
                    tensors = load_layer(layer, shape, index)
                    if args.sparsity_csv is not None:
                        zero_nibbles, total_nibbles = layer_zero_nibble_fraction(
                            layer, shape, tensors
                        )
                        sparsity_rows.append((layer, zero_nibbles, total_nibbles))
                    write_layer(out, layer, shape, tensors)
                    del tensors
                    print(
                        f"  layer {layer:2d} "
                        f"({(layer + 1) / shape.num_layers * 100:5.1f}%) — "
                        f"{time.perf_counter() - layer_started:.1f}s"
                    )
                final_size = out.tell()
        except BaseException:
            if not args.dry_run:
                target.unlink(missing_ok=True)
            raise

        if final_size != shape.total_bytes:
            if not args.dry_run:
                target.unlink(missing_ok=True)
            raise SystemExit(
                f"FATAL size mismatch: wrote {final_size}, expected {shape.total_bytes}"
            )
        elapsed = time.perf_counter() - started
        if args.dry_run:
            print(
                f"\nDry run complete in {elapsed:.1f}s — streamed "
                f"{final_size / 2**30:.2f} GiB, peak RSS {peak_rss_mib():.1f} MiB"
            )
        else:
            os.replace(target, args.out)
            print(
                f"\nDone in {elapsed:.1f}s — {args.out} "
                f"({final_size / 2**30:.2f} GiB), peak RSS {peak_rss_mib():.1f} MiB"
            )
            if args.sparsity_csv is not None:
                write_sparsity_csv(args.sparsity_csv, sparsity_rows)
                print(f"Sparsity CSV: {args.sparsity_csv}")

    if args.validate and not args.dry_run:
        validate_bank_against_shards(args.out, args.model_dir, shape, index, samples)
    if args.nvfp4_model_dir is not None:
        cross_validate_nvfp4(args.nvfp4_model_dir, index, shape, samples)

    print(f"Peak RSS: {peak_rss_mib():.1f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
