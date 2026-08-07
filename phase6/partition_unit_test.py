#!/usr/bin/env python3
"""Phase-6a unit test: runtime route partitioning + invariants.

Validates the partition logic on synthetic (topk_ids, topk_weights) tensors
without vLLM. Covers:
  * correct CUDA/B70 classification against the placement device-map;
  * covering + disjoint invariants (exactly one owner per valid route);
  * B70-only-in-capable-layers constraint;
  * weight-preservation (original tensors unmodified);
  * padding (-1) handling;
  * negatives.

Runs with torch only (no vLLM import needed beyond the package config/placement
which are pure-Python).
"""

from __future__ import annotations

import torch

from shooting_brake_vllm.partition import (
    PartitionError,
    RoutePartition,
    build_device_map,
    partition_routes,
    validate_partition,
)
from shooting_brake_vllm.placement import (
    AllCudaPolicy,
    InterleavedPolicy,
    SplitPolicy,
    build_placement,
)

NL, NE = 40, 256
CAPABLE = frozenset(range(32))


def ok(label: str, cond: bool) -> None:
    if not cond:
        raise AssertionError(f"FAIL {label}")
    print(f"  ok  {label}")


def expect_raise(label: str, fn) -> None:
    try:
        fn()
    except PartitionError:
        print(f"  ok  {label} (rejected)")
    else:
        raise AssertionError(f"FAIL {label}: expected rejection")


def main() -> None:
    split = build_placement(
        SplitPolicy(128), num_layers=NL, num_experts=NE, b70_capable=CAPABLE
    )
    dm = build_device_map(split)
    ok("device_map shape", dm.shape == (NL, NE))

    # ---- correct classification ---------------------------------------
    ids = torch.tensor(
        [[0, 1, 127, 128, 129, 200, 255, 50],
         [10, 20, 130, 140, 150, 160, 170, 180]]
    )
    wts = torch.rand(2, 8)

    part = partition_routes(ids, wts, dm[5], layer=5)
    validate_partition(part, CAPABLE)
    # experts 0-127 CUDA, 128-255 B70 in capable layers
    ok("layer 5 cuda=6", part.num_cuda_routes() == 6)
    ok("layer 5 b70=10", part.num_b70_routes() == 10)
    ok("layer 5 remote", part.has_remote())
    # spot-check row 0: 0,1,127,50 CUDA; 128,129,200,255 B70
    ok("row0 cuda ids", ids[0][part.cuda_mask[0]].tolist() == [0, 1, 127, 50])
    ok("row0 b70 ids", ids[0][part.b70_mask[0]].tolist() == [128, 129, 200, 255])

    # ---- FP8 layer 35: all CUDA ----------------------------------------
    part35 = partition_routes(ids, wts, dm[35], layer=35)
    validate_partition(part35, CAPABLE)
    ok("layer 35 b70=0", part35.num_b70_routes() == 0)
    ok("layer 35 cuda=16", part35.num_cuda_routes() == 16)

    # ---- weight-preserving --------------------------------------------
    wts_before = wts.clone()
    _ = partition_routes(ids, wts, dm[5], layer=5)
    ok("weights unchanged", torch.equal(wts, wts_before))

    # ---- all-cuda placement: zero B70 everywhere ----------------------
    dm_ac = build_device_map(
        build_placement(AllCudaPolicy(), num_layers=NL, num_experts=NE, b70_capable=CAPABLE)
    )
    part_ac = partition_routes(ids, wts, dm_ac[5], layer=5)
    validate_partition(part_ac, CAPABLE)
    ok("all-cuda b70=0", part_ac.num_b70_routes() == 0)

    # ---- interleaved placement: experts 1,3,5,7,... B70 ---------------
    dm_il = build_device_map(
        build_placement(InterleavedPolicy(2), num_layers=NL, num_experts=NE, b70_capable=CAPABLE)
    )
    part_il = partition_routes(ids, wts, dm_il[5], layer=5)
    validate_partition(part_il, CAPABLE)
    ok("interleaved has remote", part_il.has_remote())

    # ---- padding (-1) handling ----------------------------------------
    ids_pad = torch.tensor([[0, 5, -1, -1, 128, 200, -1, -1]])
    wts_pad = torch.rand(1, 8)
    part_pad = partition_routes(ids_pad, wts_pad, dm[5], layer=5)
    validate_partition(part_pad, CAPABLE)
    ok("padding valid=4", part_pad.num_valid_routes() == 4)
    ok("padding cuda=2", part_pad.num_cuda_routes() == 2)   # 0, 5
    ok("padding b70=2", part_pad.num_b70_routes() == 2)     # 128, 200

    # ---- negatives ----------------------------------------------------
    # These build a RoutePartition directly because the device map cannot
    # produce them: a gather assigns each route exactly one code. They exist
    # to prove validate_partition still catches the states a hand-written or
    # future policy-driven partition could reach.
    zeros = torch.zeros(2, 8, dtype=torch.bool)
    ones = torch.ones(2, 8, dtype=torch.bool)

    # double-owned: CUDA and B70 both claim every route
    fake = RoutePartition(
        layer=5, topk_ids=ids, topk_weights=wts,
        cuda_mask=ones, b70_mask=ones, cpu_mask=zeros, valid_mask=ones,
    )
    expect_raise("double-owned rejected", lambda: validate_partition(fake, CAPABLE))

    # uncovered: valid routes that no tier claims
    fake2 = RoutePartition(
        layer=5, topk_ids=ids, topk_weights=wts,
        cuda_mask=zeros, b70_mask=zeros, cpu_mask=zeros, valid_mask=ones,
    )
    expect_raise("uncovered rejected", lambda: validate_partition(fake2, CAPABLE))

    # B70 in non-capable layer
    fake3 = RoutePartition(
        layer=35, topk_ids=ids, topk_weights=wts,
        cuda_mask=zeros, b70_mask=ones, cpu_mask=zeros, valid_mask=ones,
    )
    expect_raise("B70 in FP8 layer rejected", lambda: validate_partition(fake3, CAPABLE))

    # CPU tier is held to the same layer restriction as the B70. Without
    # this, an all-out placement could put expert compute on the CPU in a
    # layer no offload path was validated against.
    fake4 = RoutePartition(
        layer=35, topk_ids=ids, topk_weights=wts,
        cuda_mask=zeros, b70_mask=zeros, cpu_mask=ones, valid_mask=ones,
    )
    expect_raise("CPU in FP8 layer rejected", lambda: validate_partition(fake4, CAPABLE))

    # A clean three-way split validates: each tier claims a disjoint third.
    third = torch.zeros(2, 8, dtype=torch.bool)
    third[:, 0:3] = True
    mid = torch.zeros(2, 8, dtype=torch.bool)
    mid[:, 3:6] = True
    last = torch.zeros(2, 8, dtype=torch.bool)
    last[:, 6:8] = True
    ok3 = RoutePartition(
        layer=5, topk_ids=ids, topk_weights=wts,
        cuda_mask=third, b70_mask=mid, cpu_mask=last, valid_mask=ones,
    )
    validate_partition(ok3, CAPABLE)
    ok("three-way split validates", True)
    ok("three-way counts", (ok3.num_cuda_routes(), ok3.num_b70_routes(),
                            ok3.num_cpu_routes()) == (6, 6, 4))

    print("\nPhase-6a partition unit-test PASS")


if __name__ == "__main__":
    main()
