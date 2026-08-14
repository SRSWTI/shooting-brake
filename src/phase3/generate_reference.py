#!/usr/bin/env python3
"""Generate the independent Phase 3 BF16/NVFP4 expert-reference fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Sequence

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent.parent
PHASE3 = ROOT / "phase3"
OUTPUT = PHASE3 / "reference_fixture.bin"
TMP_OUTPUT = PHASE3 / "reference_fixture.bin.tmp"
BANK_PATH = ROOT / "phase1" / "expert_bank.bin"

SOURCE_REPOSITORY = "models--Qwen--Qwen3.6-35B-A3B"
SOURCE_SNAPSHOT = "995ad96eacd98c81ed38be0c5b274b04031597b0"
NVFP4_REPOSITORY = "models--unsloth--Qwen3.6-35B-A3B-NVFP4"
NVFP4_SNAPSHOT = "739af1e7aac320af1682ed1e0cce369af4c5265d"
BANK_SHA256 = "0ce6377ba3c9848da42b6063574ea884052d2e0f5e605d86d1684a1e5826e8db"
NVFP4_MANIFEST_SHA256 = (
    "320fad67387d36509947a691fa269d5a55dfb08f0cd7da6434868a6861bff2fa"
)
FIXTURE_SHA256 = "3ebac16d0f09907cee4718ac1054d21939e420eabaf76ebe79c75fa5d0132606"
SOURCE_INDEX_TOTAL_BYTES = 71_903_645_408
NVFP4_INDEX_TOTAL_BYTES = 26_473_821_704
BANK_FILE_BYTES = 14_495_580_220

HIDDEN = 2048
INTERMEDIATE = 512
TOPK = 8
BANK_LAYERS = 32
BANK_EXPERTS = 256
LAYERS = (0, 31)
EXPERTS = (0, 1, 7, 63, 127, 191, 254, 255)
NUM_INPUTS = 8
BLOCK_SIZE = 16
ALIGNMENT = 64

# Frozen from the physical Phase 3 calibration matrix:
# worst expert relative RMSE=0.1683012879458, minimum cosine=0.985919468279;
# aggregate relative RMSE=0.1579548618065, cosine=0.987528585785.
MAX_RELATIVE_RMSE = 0.18
MIN_COSINE = 0.98

BANK_HEADER = struct.Struct("<8sIIIIIQQQQ")
FIXTURE_HEADER = struct.Struct("<8s10I6Q40s40s32s32s16s")
FIXTURE_MAGIC = b"SBP3RF01"
FIXTURE_VERSION = 1
FIXTURE_HEADER_BYTES = 256
ENDIAN_TAG = 0x01020304

W13_BYTES = 2 * INTERMEDIATE * (HIDDEN // 2)
S13_BYTES = 2 * INTERMEDIATE * (HIDDEN // BLOCK_SIZE)
W2_BYTES = HIDDEN * (INTERMEDIATE // 2)
S2_BYTES = HIDDEN * (INTERMEDIATE // BLOCK_SIZE)
EXPERT_RECORD_BYTES = W13_BYTES + S13_BYTES + W2_BYTES + S2_BYTES + 8

E2M1 = np.asarray(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=np.float64,
)


def default_snapshot(repository: str, snapshot: str) -> Path:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    return hf_home / "hub" / repository / "snapshots" / snapshot


def source_model_dir() -> Path:
    return Path(
        os.environ.get(
            "SB_SOURCE_MODEL_DIR",
            default_snapshot(SOURCE_REPOSITORY, SOURCE_SNAPSHOT),
        )
    )


def nvfp4_model_dir() -> Path:
    return Path(
        os.environ.get(
            "SB_NVFP4_MODEL_DIR",
            default_snapshot(NVFP4_REPOSITORY, NVFP4_SNAPSHOT),
        )
    )


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    buffer = bytearray(16 * 1024 * 1024)
    view = memoryview(buffer)
    with path.open("rb", buffering=0) as stream:
        while True:
            count = stream.readinto(buffer)
            if not count:
                break
            digest.update(view[:count])
    return digest.hexdigest()


def verify_file_hash(path: Path, expected_size: int, expected_digest: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"required file not found: {path}")
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"{path}: file size {actual_size} does not match frozen {expected_size}"
        )
    actual_digest = sha256_file(path)
    if actual_digest != expected_digest:
        raise RuntimeError(
            f"{path}: SHA256 {actual_digest} does not match frozen {expected_digest}"
        )


def canonical_shard_manifest(model_dir: Path) -> tuple[bytes, list[tuple[Path, str]]]:
    shards = sorted(model_dir.glob("model-*.safetensors"), key=lambda path: path.name)
    if not shards:
        raise RuntimeError(f"NVFP4 shard manifest: no model-*.safetensors in {model_dir}")
    entries: list[tuple[Path, str]] = []
    manifest = bytearray()
    for shard in shards:
        if not shard.is_file():
            raise RuntimeError(f"NVFP4 shard manifest: missing shard {shard}")
        shard_digest = sha256_file(shard)
        entries.append((shard, shard_digest))
        manifest.extend(f"{shard_digest}  {shard.name}\n".encode("ascii"))
    return bytes(manifest), entries


def verify_nvfp4_manifest(model_dir: Path) -> None:
    manifest, entries = canonical_shard_manifest(model_dir)
    actual = hashlib.sha256(manifest).hexdigest()
    if actual != NVFP4_MANIFEST_SHA256:
        raise RuntimeError(
            "NVFP4 shard manifest SHA256 "
            f"{actual} does not match frozen {NVFP4_MANIFEST_SHA256}"
        )
    print(
        f"Verified NVFP4 manifest {actual} over {len(entries)} canonical shards",
        flush=True,
    )


class TensorStore:
    """Indexed, lazily opened safetensors snapshot."""

    def __init__(self, root: Path, label: str, expected_total_bytes: int) -> None:
        self.root = root
        self.label = label
        if not root.is_dir():
            raise RuntimeError(f"{label} snapshot directory not found: {root}")
        index_path = root / "model.safetensors.index.json"
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"{label} invalid tensor index {index_path}: {error}") from error
        total_size = index.get("metadata", {}).get("total_size")
        if int(total_size) != expected_total_bytes:
            raise RuntimeError(
                f"{label} index total_size {total_size!r} does not match "
                f"frozen {expected_total_bytes}"
            )
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise RuntimeError(f"{label} tensor index has no weight_map")
        self.weight_map: dict[str, str] = weight_map
        self.stack = ExitStack()
        self.handles: dict[str, object] = {}

    def close(self) -> None:
        self.stack.close()

    def __enter__(self) -> TensorStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _handle(self, key: str):
        file_name = self.weight_map.get(key)
        if not isinstance(file_name, str):
            raise RuntimeError(f"{self.label} tensor missing: {key}")
        if Path(file_name).name != file_name:
            raise RuntimeError(f"{self.label} tensor {key}: unsafe shard path {file_name!r}")
        shard = self.root / file_name
        if not shard.is_file():
            raise RuntimeError(f"{self.label} tensor {key}: shard not found: {shard}")
        handle = self.handles.get(file_name)
        if handle is None:
            handle = self.stack.enter_context(
                safe_open(str(shard), framework="pt", device="cpu")
            )
            self.handles[file_name] = handle
        return handle

    def tensor(self, key: str) -> torch.Tensor:
        try:
            return self._handle(key).get_tensor(key)
        except Exception as error:
            raise RuntimeError(f"{self.label} tensor {key}: load failed: {error}") from error

    def tensor_slice(self, key: str, index: int) -> torch.Tensor:
        try:
            return self._handle(key).get_slice(key)[index]
        except Exception as error:
            raise RuntimeError(
                f"{self.label} tensor {key}[{index}]: slice failed: {error}"
            ) from error


def require_tensor(
    tensor: torch.Tensor,
    expected_shape: tuple[int, ...],
    expected_dtype: torch.dtype,
    context: str,
) -> None:
    if tuple(tensor.shape) != expected_shape:
        raise RuntimeError(
            f"{context}: shape {tuple(tensor.shape)} does not match {expected_shape}"
        )
    if tensor.dtype != expected_dtype:
        raise RuntimeError(
            f"{context}: dtype {tensor.dtype} does not match {expected_dtype}"
        )
    if not tensor.is_contiguous():
        raise RuntimeError(f"{context}: tensor is not C-contiguous")


def tensor_raw_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def decode_bfloat16(tensor: torch.Tensor, context: str) -> np.ndarray:
    if tensor.dtype != torch.bfloat16:
        raise RuntimeError(f"{context}: expected BF16, got {tensor.dtype}")
    words = (
        tensor.detach()
        .cpu()
        .contiguous()
        .view(torch.uint16)
        .numpy()
        .astype(np.uint32, copy=False)
    )
    fp32_bits = np.left_shift(words, np.uint32(16))
    values = fp32_bits.view(np.float32).astype(np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{context}: BF16 tensor contains non-finite values")
    return values


def decode_e2m1(packed: np.ndarray, logical_columns: int, context: str) -> np.ndarray:
    if packed.dtype != np.uint8 or packed.ndim != 2:
        raise RuntimeError(f"{context}: packed E2M1 must be a rank-2 uint8 tensor")
    if packed.shape[1] * 2 != logical_columns:
        raise RuntimeError(
            f"{context}: packed width {packed.shape[1]} does not encode "
            f"{logical_columns} logical columns"
        )
    nibbles = np.empty((packed.shape[0], logical_columns), dtype=np.uint8)
    nibbles[:, 0::2] = packed & np.uint8(0x0F)
    nibbles[:, 1::2] = packed >> np.uint8(4)
    return E2M1[nibbles]


def decode_e4m3fn(scale_bytes: np.ndarray, context: str) -> np.ndarray:
    if scale_bytes.dtype != np.uint8 or scale_bytes.ndim != 2:
        raise RuntimeError(f"{context}: E4M3FN scales must be rank-2 raw uint8")
    reserved = (scale_bytes & np.uint8(0x7F)) == np.uint8(0x7F)
    if reserved.any():
        row, column = np.argwhere(reserved)[0]
        value = int(scale_bytes[row, column])
        raise RuntimeError(
            f"{context}: reserved/NaN E4M3FN encoding 0x{value:02x} "
            f"at row={row}, scale={column}"
        )

    bits = scale_bytes.astype(np.uint16)
    sign = (bits >> 7) != 0
    exponent = ((bits >> 3) & 0x0F).astype(np.int16)
    mantissa = (bits & 0x07).astype(np.int16)
    values = np.empty(scale_bytes.shape, dtype=np.float64)
    subnormal = exponent == 0
    values[subnormal] = np.ldexp(mantissa[subnormal].astype(np.float64), -9)
    normal = ~subnormal
    values[normal] = np.ldexp(
        (8 + mantissa[normal]).astype(np.float64), exponent[normal] - 10
    )
    values[sign] = -values[sign]
    if not np.isfinite(values).all() or (values < 0.0).any():
        row, column = np.argwhere(~np.isfinite(values) | (values < 0.0))[0]
        raise RuntimeError(
            f"{context}: scale at row={row}, scale={column} is not finite/nonnegative"
        )
    return values


@dataclass
class NVFP4Projection:
    name: str
    packed: np.ndarray
    scale_bytes: np.ndarray
    global_bits: bytes
    global_value: float
    weight: np.ndarray


def load_nvfp4_projection(
    store: TensorStore,
    layer: int,
    expert: int,
    projection: str,
    output_rows: int,
    input_columns: int,
) -> NVFP4Projection:
    prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}"
    context = f"layer={layer} expert={expert} tensor={projection}"
    packed_tensor = store.tensor(f"{prefix}.weight_packed")
    scale_tensor = store.tensor(f"{prefix}.weight_scale")
    global_tensor = store.tensor(f"{prefix}.weight_global_scale")
    require_tensor(
        packed_tensor,
        (output_rows, input_columns // 2),
        torch.uint8,
        f"{context}.weight_packed",
    )
    require_tensor(
        scale_tensor,
        (output_rows, input_columns // BLOCK_SIZE),
        torch.float8_e4m3fn,
        f"{context}.weight_scale",
    )
    if global_tensor.numel() != 1 or global_tensor.dtype != torch.float32:
        raise RuntimeError(
            f"{context}.weight_global_scale: expected one float32, got "
            f"shape={tuple(global_tensor.shape)} dtype={global_tensor.dtype}"
        )

    packed = packed_tensor.numpy()
    scale_bytes = scale_tensor.view(torch.uint8).numpy()
    global_bits = tensor_raw_bytes(global_tensor)
    if len(global_bits) != 4:
        raise RuntimeError(f"{context}.weight_global_scale: expected exactly four bytes")
    global_value = struct.unpack("<f", global_bits)[0]
    if not math.isfinite(global_value) or global_value <= 0.0:
        raise RuntimeError(
            f"{context}.weight_global_scale: value {global_value!r} is not finite/positive"
        )

    values = decode_e2m1(packed, input_columns, f"{context}.weight_packed")
    scales = decode_e4m3fn(scale_bytes, f"{context}.weight_scale")
    values *= np.repeat(scales, BLOCK_SIZE, axis=1)
    values /= np.float64(global_value)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{context}: decoded weight contains non-finite values")
    return NVFP4Projection(
        projection,
        packed,
        scale_bytes,
        global_bits,
        global_value,
        values,
    )


def load_source_expert(
    store: TensorStore, layer: int, expert: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prefix = f"model.language_model.layers.{layer}.mlp.experts"
    gate_up_key = f"{prefix}.gate_up_proj"
    down_key = f"{prefix}.down_proj"
    gate_up_tensor = store.tensor_slice(gate_up_key, expert)
    down_tensor = store.tensor_slice(down_key, expert)
    require_tensor(
        gate_up_tensor,
        (2 * INTERMEDIATE, HIDDEN),
        torch.bfloat16,
        f"layer={layer} expert={expert} tensor=source.gate_up_proj",
    )
    require_tensor(
        down_tensor,
        (HIDDEN, INTERMEDIATE),
        torch.bfloat16,
        f"layer={layer} expert={expert} tensor=source.down_proj",
    )
    gate_up = decode_bfloat16(
        gate_up_tensor,
        f"layer={layer} expert={expert} tensor=source.gate_up_proj",
    )
    down = decode_bfloat16(
        down_tensor,
        f"layer={layer} expert={expert} tensor=source.down_proj",
    )
    return gate_up[:INTERMEDIATE], gate_up[INTERMEDIATE:], down


def compare_region(
    bank: BinaryIO,
    offset: int,
    expected: bytes,
    layer: int,
    expert: int,
    tensor: str,
    packed_e2m1: bool = False,
) -> None:
    bank.seek(offset)
    actual = bank.read(len(expected))
    if len(actual) != len(expected):
        raise RuntimeError(
            f"layer={layer} expert={expert} tensor={tensor}: bank record is truncated"
        )
    if actual == expected:
        return
    mismatch = next(index for index, pair in enumerate(zip(actual, expected)) if pair[0] != pair[1])
    detail = ""
    if packed_e2m1:
        actual_byte = actual[mismatch]
        expected_byte = expected[mismatch]
        detail = (
            f"; logical even/odd nibbles actual=({actual_byte & 15},"
            f"{actual_byte >> 4}) expected=({expected_byte & 15},"
            f"{expected_byte >> 4})"
        )
    raise RuntimeError(
        f"layer={layer} expert={expert} tensor={tensor}: byte mismatch at "
        f"region byte {mismatch}: bank=0x{actual[mismatch]:02x} "
        f"artifact=0x{expected[mismatch]:02x}{detail}"
    )


def audit_bank_header(bank: BinaryIO) -> None:
    raw = bank.read(BANK_HEADER.size)
    if len(raw) != BANK_HEADER.size:
        raise RuntimeError("expert bank: truncated packed header")
    fields = BANK_HEADER.unpack(raw)
    expected = (
        b"SBEXP001",
        BANK_LAYERS,
        BANK_EXPERTS,
        HIDDEN,
        INTERMEDIATE,
        0,
        W13_BYTES,
        S13_BYTES,
        W2_BYTES,
        S2_BYTES,
    )
    if fields != expected:
        raise RuntimeError(f"expert bank: header {fields!r} does not match {expected!r}")


def reciprocal_float32_bits(global_value: float) -> bytes:
    reciprocal = np.float32(np.float64(1.0) / np.float64(global_value))
    return struct.pack("<f", float(reciprocal))


def audit_bank_record(
    bank: BinaryIO,
    layer: int,
    expert: int,
    gate: NVFP4Projection,
    up: NVFP4Projection,
    down: NVFP4Projection,
) -> None:
    record = BANK_HEADER.size + (layer * BANK_EXPERTS + expert) * EXPERT_RECORD_BYTES
    offset = record
    compare_region(
        bank,
        offset,
        gate.packed.tobytes(),
        layer,
        expert,
        "bank.w13.gate_packed",
        True,
    )
    offset += gate.packed.nbytes
    compare_region(
        bank,
        offset,
        up.packed.tobytes(),
        layer,
        expert,
        "bank.w13.up_packed",
        True,
    )
    offset += up.packed.nbytes
    compare_region(
        bank,
        offset,
        gate.scale_bytes.tobytes(),
        layer,
        expert,
        "bank.w13.gate_scale_raw_e4m3fn",
    )
    offset += gate.scale_bytes.nbytes
    compare_region(
        bank,
        offset,
        up.scale_bytes.tobytes(),
        layer,
        expert,
        "bank.w13.up_scale_raw_e4m3fn",
    )
    offset += up.scale_bytes.nbytes
    compare_region(
        bank,
        offset,
        down.packed.tobytes(),
        layer,
        expert,
        "bank.w2.down_packed",
        True,
    )
    offset += down.packed.nbytes
    compare_region(
        bank,
        offset,
        down.scale_bytes.tobytes(),
        layer,
        expert,
        "bank.w2.down_scale_raw_e4m3fn",
    )
    offset += down.scale_bytes.nbytes
    compare_region(
        bank,
        offset,
        reciprocal_float32_bits(gate.global_value),
        layer,
        expert,
        "bank.w13.float32_reciprocal_bits",
    )
    offset += 4
    compare_region(
        bank,
        offset,
        reciprocal_float32_bits(down.global_value),
        layer,
        expert,
        "bank.w2.float32_reciprocal_bits",
    )
    offset += 4
    if offset != record + EXPERT_RECORD_BYTES:
        raise RuntimeError(
            f"layer={layer} expert={expert} tensor=bank.record: internal size mismatch"
        )


def deterministic_hidden() -> np.ndarray:
    columns = np.arange(HIDDEN, dtype=np.int64)
    rows = np.empty((NUM_INPUTS, HIDDEN), dtype=np.float64)
    for row in range(NUM_INPUTS):
        multiplier = 37 + 22 * row
        quadratic = (row + 3) * ((columns * columns + 17 * columns) % 257)
        integers = (
            multiplier * (columns + 1) + quadratic + 97 * row * row + 11
        ) % 2047 - 1023
        signs = np.where(((columns // (row + 3)) + row) % 2 == 0, 1, -1)
        rows[row] = signs * integers / np.float64(2048)
    hidden = rows.astype("<f2")
    for left in range(NUM_INPUTS):
        if np.count_nonzero(hidden[left]) < HIDDEN // 2:
            raise RuntimeError(f"deterministic input row {left} is unexpectedly trivial")
        for right in range(left):
            if np.array_equal(hidden[left].view("<u2"), hidden[right].view("<u2")):
                raise RuntimeError(
                    f"deterministic input rows {left} and {right} have identical FP16 bits"
                )
    if not np.isfinite(hidden).all():
        raise RuntimeError("deterministic FP16 inputs contain non-finite values")
    return hidden


def stable_silu_in_place(values: np.ndarray) -> None:
    positive = values >= 0.0
    negative = ~positive
    values[positive] /= 1.0 + np.exp(-values[positive])
    exp_values = np.exp(values[negative])
    values[negative] *= exp_values / (1.0 + exp_values)


def expert_forward(
    hidden: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    context: str,
) -> np.ndarray:
    gate_values = hidden @ gate.T
    up_values = hidden @ up.T
    stable_silu_in_place(gate_values)
    gate_values *= up_values
    output = gate_values @ down.T
    if output.shape != (NUM_INPUTS, HIDDEN) or not np.isfinite(output).all():
        raise RuntimeError(f"{context}: expert output is wrong-shaped or non-finite")
    return output


@dataclass(frozen=True)
class Metrics:
    relative_rmse: float
    cosine: float


def quality_metrics(actual: np.ndarray, reference: np.ndarray, context: str) -> Metrics:
    actual_flat = actual.reshape(-1)
    reference_flat = reference.reshape(-1)
    difference = actual_flat - reference_flat
    reference_norm = float(np.linalg.norm(reference_flat))
    actual_norm = float(np.linalg.norm(actual_flat))
    if reference_norm == 0.0 or actual_norm == 0.0:
        raise RuntimeError(f"{context}: cannot compute quality metric with zero norm")
    relative_rmse = float(np.linalg.norm(difference) / reference_norm)
    cosine = float(np.dot(actual_flat, reference_flat) / (actual_norm * reference_norm))
    if not math.isfinite(relative_rmse) or not math.isfinite(cosine):
        raise RuntimeError(f"{context}: quality metric is non-finite")
    return Metrics(relative_rmse, cosine)


def report_and_check_quality(
    nvfp4_outputs: np.ndarray,
    source_outputs: np.ndarray,
    calibrate: bool,
) -> None:
    failures: list[str] = []
    for layer_index, layer in enumerate(LAYERS):
        for expert_index, expert in enumerate(EXPERTS):
            context = f"layer={layer} expert={expert}"
            metrics = quality_metrics(
                nvfp4_outputs[layer_index, :, expert_index, :],
                source_outputs[layer_index, :, expert_index, :],
                context,
            )
            print(
                f"QUALITY {context} relative_rmse={metrics.relative_rmse:.12e} "
                f"cosine={metrics.cosine:.12f}"
            )
            if not calibrate:
                if metrics.relative_rmse > MAX_RELATIVE_RMSE:
                    failures.append(
                        f"{context} relative_rmse={metrics.relative_rmse:.12e} "
                        f"> {MAX_RELATIVE_RMSE:.12e}"
                    )
                if metrics.cosine < MIN_COSINE:
                    failures.append(
                        f"{context} cosine={metrics.cosine:.12f} < {MIN_COSINE:.12f}"
                    )

    aggregate = quality_metrics(nvfp4_outputs, source_outputs, "aggregate")
    print(
        f"QUALITY aggregate relative_rmse={aggregate.relative_rmse:.12e} "
        f"cosine={aggregate.cosine:.12f}"
    )
    if calibrate:
        print("CALIBRATION quality thresholds intentionally not applied")
        return
    if aggregate.relative_rmse > MAX_RELATIVE_RMSE:
        failures.append(
            f"aggregate relative_rmse={aggregate.relative_rmse:.12e} "
            f"> {MAX_RELATIVE_RMSE:.12e}"
        )
    if aggregate.cosine < MIN_COSINE:
        failures.append(
            f"aggregate cosine={aggregate.cosine:.12f} < {MIN_COSINE:.12f}"
        )
    if failures:
        raise RuntimeError("BF16-vs-NVFP4 quality threshold failure: " + "; ".join(failures))


def fixture_layout() -> tuple[int, int, int, int, int]:
    layer_ids_offset = align_up(FIXTURE_HEADER_BYTES)
    expert_ids_offset = align_up(layer_ids_offset + len(LAYERS) * 4)
    hidden_fp16_offset = align_up(expert_ids_offset + len(EXPERTS) * 4)
    hidden_bytes = NUM_INPUTS * HIDDEN * 2
    nvfp4_outputs_offset = align_up(hidden_fp16_offset + hidden_bytes)
    outputs_bytes = len(LAYERS) * NUM_INPUTS * len(EXPERTS) * HIDDEN * 8
    source_outputs_offset = align_up(nvfp4_outputs_offset + outputs_bytes)
    file_bytes = source_outputs_offset + outputs_bytes
    return (
        layer_ids_offset,
        expert_ids_offset,
        hidden_fp16_offset,
        nvfp4_outputs_offset,
        source_outputs_offset,
        file_bytes,
    )


def write_at(stream: BinaryIO, offset: int, payload: bytes, name: str) -> None:
    position = stream.tell()
    if position > offset:
        raise RuntimeError(f"fixture {name}: offset {offset} overlaps byte {position}")
    stream.write(bytes(offset - position))
    if stream.tell() != offset:
        raise RuntimeError(f"fixture {name}: failed to reach offset {offset}")
    stream.write(payload)


def expected_header(layout: Sequence[int]) -> bytes:
    (
        layer_ids_offset,
        expert_ids_offset,
        hidden_fp16_offset,
        nvfp4_outputs_offset,
        source_outputs_offset,
        file_bytes,
    ) = layout
    return FIXTURE_HEADER.pack(
        FIXTURE_MAGIC,
        FIXTURE_VERSION,
        FIXTURE_HEADER_BYTES,
        ENDIAN_TAG,
        HIDDEN,
        INTERMEDIATE,
        TOPK,
        len(LAYERS),
        len(EXPERTS),
        NUM_INPUTS,
        0,
        layer_ids_offset,
        expert_ids_offset,
        hidden_fp16_offset,
        nvfp4_outputs_offset,
        source_outputs_offset,
        file_bytes,
        SOURCE_SNAPSHOT.encode("ascii"),
        NVFP4_SNAPSHOT.encode("ascii"),
        bytes.fromhex(BANK_SHA256),
        bytes.fromhex(NVFP4_MANIFEST_SHA256),
        bytes(16),
    )


def validate_fixture(path: Path, layout: Sequence[int]) -> None:
    if FIXTURE_HEADER.size != FIXTURE_HEADER_BYTES:
        raise RuntimeError(
            f"fixture packed header is {FIXTURE_HEADER.size}, expected {FIXTURE_HEADER_BYTES}"
        )
    expected = expected_header(layout)
    with path.open("rb") as stream:
        actual = stream.read(FIXTURE_HEADER_BYTES)
    if actual != expected:
        raise RuntimeError(f"fixture {path}: packed header validation failed")
    file_bytes = layout[-1]
    actual_size = path.stat().st_size
    if actual_size != file_bytes:
        raise RuntimeError(
            f"fixture {path}: file size {actual_size} does not match header {file_bytes}"
        )
    for name, offset in zip(
        (
            "layer_ids",
            "expert_ids",
            "hidden_fp16",
            "nvfp4_outputs",
            "source_outputs",
        ),
        layout[:-1],
    ):
        if offset % ALIGNMENT != 0:
            raise RuntimeError(f"fixture {name}: offset {offset} is not 64-byte aligned")


def write_fixture(
    hidden_fp16: np.ndarray,
    nvfp4_outputs: np.ndarray,
    source_outputs: np.ndarray,
) -> None:
    layout = fixture_layout()
    layer_ids = np.asarray(LAYERS, dtype="<u4").tobytes()
    expert_ids = np.asarray(EXPERTS, dtype="<i4").tobytes()
    hidden_bytes = np.ascontiguousarray(hidden_fp16, dtype="<f2").tobytes()
    nvfp4_bytes = np.ascontiguousarray(nvfp4_outputs, dtype="<f8").tobytes()
    source_bytes = np.ascontiguousarray(source_outputs, dtype="<f8").tobytes()
    with TMP_OUTPUT.open("wb") as stream:
        stream.write(expected_header(layout))
        write_at(stream, layout[0], layer_ids, "layer_ids")
        write_at(stream, layout[1], expert_ids, "expert_ids")
        write_at(stream, layout[2], hidden_bytes, "hidden_fp16")
        write_at(stream, layout[3], nvfp4_bytes, "nvfp4_outputs")
        write_at(stream, layout[4], source_bytes, "source_outputs")
        if stream.tell() != layout[5]:
            raise RuntimeError(
                f"fixture temporary file ended at {stream.tell()}, expected {layout[5]}"
            )
        stream.flush()
        os.fsync(stream.fileno())
    validate_fixture(TMP_OUTPUT, layout)
    os.replace(TMP_OUTPUT, OUTPUT)
    validate_fixture(OUTPUT, layout)
    actual_sha256 = sha256_file(OUTPUT)
    if actual_sha256 != FIXTURE_SHA256:
        raise RuntimeError(
            f"fixture SHA256 {actual_sha256} does not match frozen {FIXTURE_SHA256}"
        )
    print(f"Verified fixture SHA256 {actual_sha256}")
    print(f"Wrote {OUTPUT} ({layout[5]} bytes) atomically")


def validate_snapshot_configs(source_dir: Path, nvfp4_dir: Path) -> None:
    try:
        source_config = json.loads((source_dir / "config.json").read_text("utf-8"))
        nvfp4_config = json.loads((nvfp4_dir / "config.json").read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"snapshot config validation failed: {error}") from error
    text = source_config.get("text_config", {})
    source_contract = (
        text.get("hidden_size"),
        text.get("moe_intermediate_size"),
        text.get("num_experts"),
        text.get("num_experts_per_tok"),
        text.get("dtype"),
    )
    expected_source = (HIDDEN, INTERMEDIATE, BANK_EXPERTS, TOPK, "bfloat16")
    if source_contract != expected_source:
        raise RuntimeError(
            f"source snapshot config {source_contract!r} does not match {expected_source!r}"
        )
    group = nvfp4_config.get("quantization_config", {}).get("config_groups", {}).get("group_1", {})
    weights = group.get("weights", {})
    nvfp4_contract = (
        group.get("format"),
        weights.get("num_bits"),
        weights.get("group_size"),
        weights.get("scale_dtype"),
        weights.get("type"),
    )
    expected_nvfp4 = (
        "nvfp4-pack-quantized",
        4,
        BLOCK_SIZE,
        "torch.float8_e4m3fn",
        "float",
    )
    if nvfp4_contract != expected_nvfp4:
        raise RuntimeError(
            f"NVFP4 snapshot config {nvfp4_contract!r} does not match {expected_nvfp4!r}"
        )


def generate(calibrate: bool) -> None:
    if sys.byteorder != "little":
        raise RuntimeError("generator requires a little-endian host for raw tensor audit")
    source_dir = source_model_dir()
    nvfp4_dir = nvfp4_model_dir()
    validate_snapshot_configs(source_dir, nvfp4_dir)

    print(f"Verifying full bank SHA256: {BANK_PATH}", flush=True)
    verify_file_hash(BANK_PATH, BANK_FILE_BYTES, BANK_SHA256)
    print(f"Verified bank SHA256 {BANK_SHA256}", flush=True)
    print(f"Verifying canonical NVFP4 shard manifest: {nvfp4_dir}", flush=True)
    verify_nvfp4_manifest(nvfp4_dir)

    hidden_fp16 = deterministic_hidden()
    hidden_float64 = hidden_fp16.astype(np.float64)
    output_shape = (len(LAYERS), NUM_INPUTS, len(EXPERTS), HIDDEN)
    nvfp4_outputs = np.empty(output_shape, dtype=np.float64)
    source_outputs = np.empty(output_shape, dtype=np.float64)

    with TensorStore(
        source_dir, "BF16 source", SOURCE_INDEX_TOTAL_BYTES
    ) as source_store, TensorStore(
        nvfp4_dir, "NVFP4 artifact", NVFP4_INDEX_TOTAL_BYTES
    ) as nvfp4_store, BANK_PATH.open("rb") as bank:
        audit_bank_header(bank)
        for layer_index, layer in enumerate(LAYERS):
            for expert_index, expert in enumerate(EXPERTS):
                context = f"layer={layer} expert={expert}"
                source_gate, source_up, source_down = load_source_expert(
                    source_store, layer, expert
                )
                gate = load_nvfp4_projection(
                    nvfp4_store,
                    layer,
                    expert,
                    "gate_proj",
                    INTERMEDIATE,
                    HIDDEN,
                )
                up = load_nvfp4_projection(
                    nvfp4_store,
                    layer,
                    expert,
                    "up_proj",
                    INTERMEDIATE,
                    HIDDEN,
                )
                down = load_nvfp4_projection(
                    nvfp4_store,
                    layer,
                    expert,
                    "down_proj",
                    HIDDEN,
                    INTERMEDIATE,
                )
                if gate.global_bits != up.global_bits:
                    raise RuntimeError(
                        f"{context} tensor=gate/up.weight_global_scale: raw float32 "
                        f"bits differ gate={gate.global_bits.hex()} up={up.global_bits.hex()}"
                    )
                audit_bank_record(bank, layer, expert, gate, up, down)
                source_outputs[layer_index, :, expert_index, :] = expert_forward(
                    hidden_float64,
                    source_gate,
                    source_up,
                    source_down,
                    f"{context} tensor=BF16-source",
                )
                nvfp4_outputs[layer_index, :, expert_index, :] = expert_forward(
                    hidden_float64,
                    gate.weight,
                    up.weight,
                    down.weight,
                    f"{context} tensor=NVFP4-artifact",
                )
                print(f"Audited and computed {context}", flush=True)

    report_and_check_quality(nvfp4_outputs, source_outputs, calibrate)
    write_fixture(hidden_fp16, nvfp4_outputs, source_outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Phase 3 source/NVFP4 float64 expert references after "
            "byte-exact artifact validation"
        )
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="report quality metrics without applying BF16-vs-NVFP4 thresholds",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        generate(args.calibrate)
    except Exception as error:
        try:
            TMP_OUTPUT.unlink(missing_ok=True)
        except OSError:
            pass
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
