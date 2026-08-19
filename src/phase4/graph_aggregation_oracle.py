#!/usr/bin/env python3
"""Controlled A -> B -> A oracle for the production B70 graph doorbell path.

This is a live-GPU gate. It invokes the production `_b70_issue_graph` and
`_b70_take_graph` methods inside a small CUDA graph, uses the native poller and
real SBINT401 provider, and compares the raw FP32 remote result with the
independent CPU int4 oracle. The captured graph also adds a known CUDA partial.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from shooting_brake_vllm.b70_poller import get_b70_poller
from shooting_brake_vllm.expert_bank import Int4ExpertBank
from shooting_brake_vllm.partition import DispatchBufferGeometry
from shooting_brake_vllm.placement import SplitPolicy, build_placement
from shooting_brake_vllm.routed_experts import (
    HybridRoutedExperts,
    _B70Lane,
    _build_b70_slot_map,
)
from shooting_brake_vllm.stream_signal import alloc_host_mapped_flag
from int4_aggregation_oracle import _cpu_b70_partial, _error


def fixture(name: str, hidden: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    index = torch.arange(hidden, dtype=torch.float32)
    if name == "A":
        x = (torch.sin(index * 0.013) * 0.75).to(torch.bfloat16).reshape(1, -1)
        global_ids = torch.arange(54, 62, dtype=torch.int32).reshape(1, -1)
        weights = torch.tensor(
            [[0.04, 0.07, 0.09, 0.11, 0.14, 0.16, 0.18, 0.21]],
            dtype=torch.float32,
        )
        cuda = (torch.cos(index * 0.007) * 0.5).to(torch.bfloat16).reshape(1, -1)
    elif name == "B":
        x = (torch.cos(index * 0.019 + 0.7) * 1.25).to(torch.bfloat16).reshape(1, -1)
        global_ids = torch.arange(70, 78, dtype=torch.int32).reshape(1, -1)
        weights = torch.tensor(
            [[0.22, 0.18, 0.15, 0.13, 0.11, 0.09, 0.07, 0.05]],
            dtype=torch.float32,
        )
        cuda = (torch.sin(index * 0.011 + 0.4) * 0.8).to(torch.bfloat16).reshape(1, -1)
    else:
        raise ValueError(name)
    return x, global_ids, weights, cuda


def peak_relative(reference: torch.Tensor, actual: torch.Tensor) -> float:
    denominator = float(reference.abs().max())
    if denominator == 0.0:
        raise RuntimeError("CPU remote reference is identically zero")
    return float((actual.float() - reference.float()).abs().max()) / denominator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument(
        "--cycles", type=int, default=100,
        help="Alternating A/B replays after the readable A->B->A case. Three "
             "replays cannot exercise a flag-ordering race; this loop can.",
    )
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--max-peak-relative", type=float, default=5e-5)
    args = parser.parse_args()
    if args.max_peak_relative <= 0:
        raise RuntimeError("--max-peak-relative must be positive")

    os.environ["SHOOTING_BRAKE_B70_BANK"] = str(args.bank)
    bank = Int4ExpertBank(args.bank)
    if (bank.hidden, bank.intermediate, bank.group_size) != (3072, 1024, 128):
        raise RuntimeError(
            "controlled graph oracle requires bank geometry 3072/1024/group128, "
            f"got {bank.hidden}/{bank.intermediate}/group{bank.group_size}"
        )
    if tuple(bank.source_expert_ids) != tuple(range(54, 180)):
        raise RuntimeError("controlled graph oracle requires resident source IDs 54..179")
    if not 0 <= args.layer < bank.layers:
        raise RuntimeError(f"layer {args.layer} outside bank")

    placement = build_placement(
        SplitPolicy(54), num_layers=48, num_experts=180,
        b70_capable=frozenset(range(48)),
    )
    max_batch = 1
    geometry = DispatchBufferGeometry(max_batch=max_batch, hidden_size=3072, top_k=8)
    lane = _B70Lane(
        device_index=0,
        slot_map=_build_b70_slot_map(placement, 0),
        pinned_hidden=torch.empty(
            geometry.hidden_shape, dtype=torch.float16, pin_memory=True,
        ),
        pinned_output=torch.empty(
            geometry.hidden_shape, dtype=torch.float32, pin_memory=True,
        ),
        pinned_ids=torch.empty(
            geometry.route_shape, dtype=torch.int32, pin_memory=True,
        ),
        pinned_weights=torch.empty(
            geometry.route_shape, dtype=torch.float32, pin_memory=True,
        ),
        dev_fp32=torch.empty(
            geometry.hidden_shape, dtype=torch.float32, device="cuda",
        ),
        dev_bf16=torch.empty(
            geometry.hidden_shape, dtype=torch.bfloat16, device="cuda",
        ),
    )
    lane.signal_host, lane.signal_dev = alloc_host_mapped_flag(0)
    lane.completion_host, lane.completion_dev = alloc_host_mapped_flag(0)
    layer = SimpleNamespace(
        _b70_max_batch=max_batch,
        hidden_size=3072,
        shooting_brake_placement=placement,
        _dispatch_geometry=geometry,
        _b70_lanes=(lane,),
    )

    poller = get_b70_poller(placement)
    poller.register_layer(
        layer_idx=args.layer,
        signal_host=lane.signal_host,
        completion_host=lane.completion_host,
        pinned_hidden=lane.pinned_hidden,
        pinned_ids=lane.pinned_ids,
        pinned_weights=lane.pinned_weights,
        pinned_output=lane.pinned_output,
    )
    poller.start()

    x_static = torch.empty((1, 3072), dtype=torch.bfloat16, device="cuda")
    ids_static = torch.empty((1, 8), dtype=torch.int32, device="cuda")
    weights_static = torch.empty((1, 8), dtype=torch.float32, device="cuda")
    cuda_static = torch.empty((1, 3072), dtype=torch.bfloat16, device="cuda")
    combined_static = torch.empty_like(cuda_static)

    fixtures: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {
        name: fixture(name, 3072) for name in ("A", "B")
    }
    references: dict[str, torch.Tensor] = {}
    for name, (x, global_ids, weights, _) in fixtures.items():
        references[name] = _cpu_b70_partial(
            bank, args.layer, x.to(torch.float16), global_ids, weights,
        )

    x_a, global_a, weights_a, cuda_a = fixtures["A"]
    x_static.copy_(x_a)
    ids_static.copy_((global_a - 54).to(torch.int32))
    weights_static.copy_(weights_a)
    cuda_static.copy_(cuda_a)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        HybridRoutedExperts._b70_issue_graph(
            layer, lane, x_static, ids_static, weights_static,
        )
        remote_bf16 = HybridRoutedExperts._b70_take_graph(layer, lane, 1)
        torch.add(cuda_static, remote_bf16, out=combined_static)
    torch.cuda.synchronize()

    observations: list[dict[str, object]] = []
    raw_by_step: list[torch.Tensor] = []
    try:
        for step, name in enumerate(("A", "B", "A")):
            x, global_ids, weights, cuda_partial = fixtures[name]
            x_static.copy_(x)
            ids_static.copy_((global_ids - 54).to(torch.int32))
            weights_static.copy_(weights)
            cuda_static.copy_(cuda_partial)
            graph.replay()
            torch.cuda.synchronize()

            raw_remote = lane.dev_fp32.detach().cpu().clone()
            remote_bf16_cpu = lane.dev_bf16.detach().cpu().clone()
            combined = combined_static.detach().cpu().clone()
            expected_chain = torch.add(
                cuda_static, lane.dev_bf16[:1],
            ).detach().cpu()
            if not torch.equal(combined, expected_chain):
                raise RuntimeError(
                    f"step {step} fixture {name}: graph-path CUDA+B70 addition differs"
                )
            peak_rel = peak_relative(references[name], raw_remote)
            if peak_rel > args.max_peak_relative:
                raise RuntimeError(
                    f"step {step} fixture {name}: B70 raw peak-relative error "
                    f"{peak_rel:.9g} > {args.max_peak_relative:.9g}"
                )
            raw_by_step.append(raw_remote)
            observations.append({
                "step": step,
                "fixture": name,
                "raw_remote_vs_cpu": _error(references[name], raw_remote),
                "combined_vs_fp32_cpu_oracle": _error(
                    cuda_partial.float() + references[name], combined.float(),
                ),
                "raw_remote_peak_relative": peak_rel,
                "combine_exact": True,
            })

        # Fixture separation must be large relative to tolerance, otherwise
        # "matched its own reference" is not a discriminating statement.
        ref_separation = float((references["A"] - references["B"]).abs().max())
        separation = float((raw_by_step[0] - raw_by_step[1]).abs().max())
        if separation == 0.0:
            raise RuntimeError("fixtures A and B produced identical remote partials")

        # Staleness is DISCRIMINATION, not bitwise identity. The kernel
        # accumulates routes with fp32 atomics, so replay order varies and two
        # correct runs of the same fixture are not bit-identical. Requiring
        # max_abs == 0 would fail on a healthy system, which is exactly what it
        # did. What a stale read actually looks like is a step whose output
        # resembles the PREVIOUS fixture's reference more than its own.
        discrimination: list[dict[str, object]] = []
        for step, name in enumerate(("A", "B", "A")):
            other = "B" if name == "A" else "A"
            d_own = float((raw_by_step[step] - references[name]).abs().max())
            d_other = float((raw_by_step[step] - references[other]).abs().max())
            discrimination.append({
                "step": step, "fixture": name,
                "max_abs_vs_own_reference": d_own,
                "max_abs_vs_other_reference": d_other,
                "ratio_other_over_own": (d_other / d_own) if d_own else float("inf"),
            })
            if d_own >= d_other:
                raise RuntimeError(
                    f"step {step} fixture {name}: output is no closer to its own "
                    f"reference ({d_own:.6g}) than to {other}'s ({d_other:.6g}) "
                    "— stale or mis-selected buffer"
                )

        # Two correct runs of A each sit within the peak-relative bound of the
        # same reference, so they must sit within roughly twice that of each
        # other. Report the observed value rather than demanding zero.
        repeat_error = _error(raw_by_step[0], raw_by_step[2])
        repeat_peak_rel = peak_relative(raw_by_step[0], raw_by_step[2])
        if repeat_peak_rel > 2.0 * args.max_peak_relative:
            raise RuntimeError(
                f"A repeat differs by peak-relative {repeat_peak_rel:.9g}, above "
                f"2x the {args.max_peak_relative:.9g} bound; not attributable to "
                "atomic accumulation order"
            )
        # A->B->A above is the minimal readable case. Flag ordering and
        # completion reset are RACES, and a race that only fires occasionally
        # will pass three replays. Alternate many times and check every single
        # replay against its own reference, so a one-in-fifty stale read is a
        # failure rather than a rounding anecdote.
        stress = {"cycles": 0, "worst_own_max_abs": 0.0,
                  "min_discrimination_ratio": float("inf"),
                  "combine_exact_failures": 0}
        for cycle in range(args.cycles):
            name = "A" if cycle % 2 == 0 else "B"
            other = "B" if name == "A" else "A"
            x, global_ids, weights, cuda_partial = fixtures[name]
            x_static.copy_(x)
            ids_static.copy_((global_ids - 54).to(torch.int32))
            weights_static.copy_(weights)
            cuda_static.copy_(cuda_partial)
            graph.replay()
            torch.cuda.synchronize()

            raw = lane.dev_fp32.detach().cpu().clone()
            d_own = float((raw - references[name]).abs().max())
            d_other = float((raw - references[other]).abs().max())
            ratio = (d_other / d_own) if d_own else float("inf")
            stress["cycles"] = cycle + 1
            stress["worst_own_max_abs"] = max(stress["worst_own_max_abs"], d_own)
            stress["min_discrimination_ratio"] = min(
                stress["min_discrimination_ratio"], ratio
            )
            if not torch.equal(
                combined_static.detach().cpu(),
                torch.add(cuda_static, lane.dev_bf16[:1]).detach().cpu(),
            ):
                stress["combine_exact_failures"] += 1
            if d_own >= d_other:
                raise RuntimeError(
                    f"cycle {cycle} fixture {name}: stale or mis-selected buffer "
                    f"(own {d_own:.6g} >= other {d_other:.6g})"
                )
            if peak_relative(references[name], raw) > args.max_peak_relative:
                raise RuntimeError(
                    f"cycle {cycle} fixture {name}: exceeded peak-relative bound"
                )
        if stress["combine_exact_failures"]:
            raise RuntimeError(
                f"{stress['combine_exact_failures']} of {stress['cycles']} "
                "replays had a non-exact graph CUDA+B70 addition"
            )

        print(json.dumps({
            "status": "PASS",
            "stress": stress,
            "layer": args.layer,
            "sequence": ["A", "B", "A"],
            "max_peak_relative_bound": args.max_peak_relative,
            "a_b_max_abs_separation": separation,
            "a_repeat": repeat_error,
            "a_repeat_peak_relative": repeat_peak_rel,
            "reference_separation_max_abs": ref_separation,
            "discrimination": discrimination,
            "steps": observations,
            "validated": [
                "graph result freshness",
                "signal/completion/reset ordering",
                "static output buffer selection",
                "graph CUDA+B70 addition",
            ],
        }, indent=2))
    finally:
        poller.stop()
        bank.close()


if __name__ == "__main__":
    main()
