#!/usr/bin/env python3
"""Offline CPU oracle for an opt-in real-layer hybrid aggregation capture.

The runtime capture is produced by setting ``SHOOTING_BRAKE_AGGREGATION_CAPTURE``
to a file path during one synchronous eager inference.  This program performs
no accelerator work: it dequantizes the canonical SBINT401 bank on CPU and
checks the provider's int4 partial for the all-B70 and straddling cases.  It
also proves that each captured hybrid result is exactly the sum of its two
captured tier partials and that the inactive tier is zero in the pure cases.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from shooting_brake_vllm.expert_bank import Int4ExpertBank, Int4ExpertPlanes


def _dequant(qweight: np.ndarray, scales: np.ndarray, group_size: int) -> torch.Tensor:
    """Decode K-major GPTQ int4 with constant zero point 8."""
    packed = torch.from_numpy(np.array(qweight, copy=True)).to(torch.int64)
    packed.bitwise_and_(0xFFFFFFFF)
    shifts = (torch.arange(8, dtype=torch.int64) * 4).view(1, 8, 1)
    nibble = ((packed.unsqueeze(1) >> shifts) & 0xF).reshape(
        packed.shape[0] * 8, packed.shape[1]
    )
    scale = torch.from_numpy(np.array(scales, copy=True)).float()
    return (nibble.float() - 8.0) * scale.repeat_interleave(group_size, dim=0)


def _expert_forward(x: torch.Tensor, planes: Int4ExpertPlanes, group_size: int) -> torch.Tensor:
    gate = _dequant(planes.gate_qweight, planes.gate_scales, group_size)
    up = _dequant(planes.up_qweight, planes.up_scales, group_size)
    hidden = F.silu(x @ gate) * (x @ up)
    del gate, up
    down = _dequant(planes.down_qweight, planes.down_scales, group_size)
    return hidden @ down


def _cpu_b70_partial(
    bank: Int4ExpertBank,
    layer: int,
    x: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    x = x.float()
    weights = weights.float()
    result = torch.zeros(x.shape[0], bank.hidden, dtype=torch.float32)
    cache: dict[tuple[int, int], torch.Tensor] = {}
    for row in range(ids.shape[0]):
        for route in range(ids.shape[1]):
            expert = int(ids[row, route])
            try:
                bank.expert(layer, expert)
            except IndexError:
                continue
            key = (row, expert)
            if key not in cache:
                cache[key] = _expert_forward(
                    x[row : row + 1], bank.expert(layer, expert), bank.group_size
                )[0]
            result[row].add_(cache[key], alpha=float(weights[row, route]))
    return result


def _error(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    reference = reference.float()
    actual = actual.float()
    delta = actual - reference
    denom = torch.linalg.vector_norm(reference).clamp_min(torch.finfo(torch.float32).tiny)
    return {
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "rel_l2": float(torch.linalg.vector_norm(delta) / denom),
        "cosine": float(F.cosine_similarity(reference.flatten(), actual.flatten(), dim=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--max-relative-l2", type=float, default=5e-3)
    parser.add_argument("--min-cosine", type=float, default=0.9999)
    args = parser.parse_args()

    capture = torch.load(args.capture, map_location="cpu", weights_only=True)
    if capture.get("format") != "shooting-brake-int4-aggregation-v1":
        raise RuntimeError(f"unsupported capture format: {capture.get('format')!r}")
    bank = Int4ExpertBank(args.bank)
    try:
        layer = int(capture["layer"])
        x = capture["actual_x"]
        reports: dict[str, object] = {}
        for name in ("all_cuda", "all_b70", "straddling"):
            case = capture["cases"][name]
            cuda_partial = case["cuda_partial"].float()
            b70_partial = case["b70_partial"].float()
            hybrid = case["hybrid_routed"].float()
            sum_error = _error(cuda_partial + b70_partial, hybrid)
            if sum_error["max_abs"] != 0.0:
                raise RuntimeError(f"{name}: saved hybrid is not the exact saved partial sum")
            if name == "all_cuda":
                inactive_max = float(b70_partial.abs().max())
                if inactive_max != 0.0:
                    raise RuntimeError(f"all_cuda: B70 partial is nonzero ({inactive_max})")
                reports[name] = {
                    "inactive_b70_max_abs": inactive_max,
                    "sum": sum_error,
                }
                continue

            cpu_b70 = _cpu_b70_partial(
                bank, layer, x, case["global_ids"], case["weights"]
            )
            provider_error = _error(cpu_b70, b70_partial)
            if provider_error["rel_l2"] > args.max_relative_l2:
                raise RuntimeError(
                    f"{name}: provider rel_l2={provider_error['rel_l2']:.6g} "
                    f"> {args.max_relative_l2:.6g}"
                )
            if provider_error["cosine"] < args.min_cosine:
                raise RuntimeError(
                    f"{name}: provider cosine={provider_error['cosine']:.9g} "
                    f"< {args.min_cosine:.9g}"
                )
            inactive_max = None
            if name == "all_b70":
                inactive_max = float(cuda_partial.abs().max())
                if inactive_max != 0.0:
                    raise RuntimeError(
                        f"all_b70: compact CUDA partial is nonzero ({inactive_max})"
                    )
            reports[name] = {
                "provider_vs_cpu_int4": provider_error,
                "inactive_cuda_max_abs": inactive_max,
                "sum": sum_error,
            }
        print(json.dumps({"status": "PASS", "layer": layer, "cases": reports}, indent=2))
    finally:
        bank.close()


if __name__ == "__main__":
    main()
