"""Replay captured real-model routes on one native provider, without vLLM.

Extracts one bank row verbatim into a temporary one-layer bank. The output
records nonfinite values separately from first-use wall time; that time is
not a steady-state kernel benchmark. No serving library is overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src/phase4/src")]

from shooting_brake_vllm.b70_binding import B70ProviderClient

from src.phase1.extract_experts import GSCALE_BYTES, HEADER_FMT, HEADER_SIZE, MAGIC


def extract_row(source: Path, destination: Path, row: int) -> tuple[list, str]:
    with source.open("rb") as src, destination.open("xb") as dst:
        header = list(struct.unpack(HEADER_FMT, src.read(HEADER_SIZE)))
        if header[0] != MAGIC or not 0 <= row < header[1]:
            raise ValueError("expected an NVFP4 bank and an in-range bank row")
        layer_bytes = header[2] * (sum(header[6:10]) + GSCALE_BYTES)
        src.seek(HEADER_SIZE + row * layer_bytes)
        single_header = header.copy()
        single_header[1] = 1
        encoded_header = struct.pack(HEADER_FMT, *single_header)
        dst.write(encoded_header)
        digest = hashlib.sha256(encoded_header)
        buffer = bytearray(min(16 << 20, layer_bytes))
        view = memoryview(buffer)
        remaining = layer_bytes
        while remaining:
            count = src.readinto(view[: min(len(view), remaining)])
            if not count:
                raise ValueError("truncated expert bank row")
            dst.write(view[:count])
            digest.update(view[:count])
            remaining -= count
    return header, digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--bank-row", type=int, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--selector", required=True)
    parser.add_argument("--resident-start", type=int, required=True)
    parser.add_argument("--resident-count", type=int, required=True)
    parser.add_argument("--grouped", choices=(0, 1), type=int, default=1)
    parser.add_argument("--pipeline", type=int, default=2)
    parser.add_argument("--wire", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--backend", choices=("native", "onednn"), default="native")
    parser.add_argument("--rows", type=int)
    parser.add_argument("--max-batch", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    array_path = args.output.with_suffix(".npy")
    if args.output.exists() or array_path.exists():
        raise FileExistsError("refusing to overwrite replay evidence")
    capture = torch.load(args.capture, map_location="cpu", weights_only=True)
    rows = args.rows if args.rows is not None else capture["actual_x"].shape[0]
    if not 0 < rows <= min(args.max_batch, capture["actual_x"].shape[0]):
        raise ValueError("rows must fit the capture and max batch")
    x = np.ascontiguousarray(capture["actual_x"][:rows].to(torch.float16).numpy())
    global_ids = capture["actual_global_ids"][:rows].numpy()
    weights = np.ascontiguousarray(capture["actual_weights"][:rows].float().numpy())
    if not np.isfinite(x).all() or not np.isfinite(weights).all():
        raise ValueError("replay input or routing weights are nonfinite")
    start = args.resident_start
    end = start + args.resident_count
    ids = np.where((global_ids >= start) & (global_ids < end), global_ids - start, -1)
    ids = np.ascontiguousarray(ids, dtype=np.int32)
    os.environ["SHOOTING_BRAKE_B70_GROUPED"] = str(args.grouped)
    os.environ["SHOOTING_BRAKE_B70_PIPELINE"] = str(args.pipeline)
    os.environ["SHOOTING_BRAKE_B70_OUT_FP16"] = "1" if args.wire == "fp16" else "0"
    os.environ["SB_GROUPED_BACKEND"] = args.backend
    os.environ["SYCL_UR_USE_LEVEL_ZERO_V2"] = "0"
    with tempfile.TemporaryDirectory(prefix="sb-captured-bank-") as temporary:
        bank = Path(temporary) / "single-layer.bin"
        header, bank_digest = extract_row(args.bank, bank, args.bank_row)
        if not 0 <= start < end <= header[2] or x.shape[1] != header[3]:
            raise ValueError("resident set or capture geometry does not match bank")
        provider = B70ProviderClient(args.library.resolve())
        try:
            provider.load(
                bank,
                top_k=ids.shape[1],
                resident_experts=np.arange(start, end, dtype=np.int32),
                max_batch=args.max_batch,
                device_selector=args.selector,
            )
            expected_dtype = np.dtype(np.float16 if args.wire == "fp16" else np.float32)
            if provider.out_dtype != expected_dtype:
                raise RuntimeError("provider output capability disagrees with requested wire")
            started = time.perf_counter_ns()
            output = provider.dispatch(0, x, ids, weights)
            elapsed_ms = (time.perf_counter_ns() - started) / 1e6
            health = asdict(provider.health)
        finally:
            provider.shutdown()
    finite = np.isfinite(output)
    with args.library.open("rb") as handle:
        library_digest = hashlib.file_digest(handle, "sha256").hexdigest()
    result = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "model_layer": capture["layer"],
        "single_layer_bank_sha256": bank_digest,
        "library_sha256": library_digest,
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "active_routes": int((ids >= 0).sum()),
        "first_row_global_routes": global_ids[0].tolist(),
        "first_row_local_routes": ids[0].tolist(),
        "nan_count": int(np.isnan(output).sum()),
        "positive_inf_count": int(np.isposinf(output).sum()),
        "negative_inf_count": int(np.isneginf(output).sum()),
        "nonfinite_rows": np.flatnonzero(~finite.all(axis=1)).tolist(),
        "finite_abs_max": float(np.abs(output[finite]).max()) if finite.any() else None,
        "dispatch_wall_ms_including_first_use": elapsed_ms,
        "health": health,
        "passed": bool(finite.all()) and not health["last_error"],
        "array": str(array_path),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(array_path, output, allow_pickle=False)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
