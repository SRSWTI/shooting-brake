#!/usr/bin/env python3
"""Phase-5 placement-manifest gate.

Proves the compact-ownership invariants for the qualified Qwen3.6 model:

* every routed expert has exactly one normal-path owner (CUDA or B70);
* CUDA+B70 ownership fully covers all 40 layers x 256 experts (10,240 total);
* FP8 layers 32-39 are CUDA-forced (not in the NVFP4 bank);
* every B70-owned expert is physically present in the Phase-1 NVFP4 bank;
* the manifest is versioned (``generation``) and round-trips to JSON, so a
  future predictive/speculative offloader can swap it at coarse boundaries.

This runs without vLLM/torch: the placement module is pure Python and the
qualified model dims are constructed directly.
"""

from __future__ import annotations

import struct
from pathlib import Path

from shooting_brake_vllm.config import (
    QUALIFIED_BANK_EXPERTS_PER_LAYER,
    QUALIFIED_BANK_LAYERS,
    QUALIFIED_EXPERTS,
    QUALIFIED_LAYERS,
    QualifiedModel,
)
from shooting_brake_vllm.placement import (
    Device,
    Placement,
    PlacementError,
    AllCudaPolicy,
    SplitPolicy,
    InterleavedPolicy,
    LayerSubsetPolicy,
    build_placement,
    build_for_qualified,
    policy_from_name,
    validate_placement,
    b70_bank_covers,
)

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "phase1" / "expert_bank.bin"
HEADER_FMT = "<8sIIIIIQQQQ"


def qualified() -> QualifiedModel:
    return QualifiedModel(
        model="unsloth/Qwen3.6-35B-A3B-NVFP4",
        architecture="Qwen3_5MoeForConditionalGeneration",
        hidden_size=2048,
        num_layers=QUALIFIED_LAYERS,
        num_experts=QUALIFIED_EXPERTS,
        top_k=8,
        moe_intermediate_size=512,
    )


def read_bank_header() -> tuple[int, int]:
    """Return (num_layers, experts_per_layer) recorded in the real bank."""
    with open(BANK, "rb") as f:
        magic, num_layers, experts_per_layer = struct.unpack(
            HEADER_FMT, f.read(struct.calcsize(HEADER_FMT))
        )[:3]
    assert magic == b"SBEXP001", f"bad bank magic {magic!r}"
    return num_layers, experts_per_layer


def expect_ok(label: str, cond: bool) -> None:
    if not cond:
        raise AssertionError(f"FAIL {label}")
    print(f"  ok  {label}")


def expect_raise(label: str, fn) -> None:
    try:
        fn()
    except (PlacementError, ValueError):
        print(f"  ok  {label} (correctly rejected)")
    else:
        raise AssertionError(f"FAIL {label}: expected rejection")


def main() -> None:
    q = qualified()
    nl, ne = q.num_layers, q.num_experts
    capable = q.b70_capable_layers
    total = nl * ne
    print(f"qualified model: {nl} layers x {ne} experts = {total} routed experts")
    print(f"  B70-capable (NVFP4 bank): layers 0-{q.bank_layers - 1}")
    print(f"  CUDA-forced (FP8): layers {q.bank_layers}-{nl - 1}")

    # ---- cross-reference the real bank header --------------------------
    bank_layers, bank_experts = read_bank_header()
    expect_ok(
        f"bank header matches config ({bank_layers}x{bank_experts})",
        bank_layers == q.bank_layers and bank_experts == q.bank_experts_per_layer,
    )

    # ---- every built-in policy validates + bank covers -----------------
    policies = {
        "all-cuda": AllCudaPolicy(),
        "split:128": SplitPolicy(128),
        "split:96": SplitPolicy(96),
        "interleaved:2": InterleavedPolicy(2),
        "interleaved:4": InterleavedPolicy(4),
    }
    for name, policy in policies.items():
        p = build_placement(policy, num_layers=nl, num_experts=ne, b70_capable=capable)
        validate_placement(p)
        covers = b70_bank_covers(
            p, bank_layers=q.bank_layers, bank_experts_per_layer=q.bank_experts_per_layer
        )
        expect_ok(
            f"{name}: cuda={p.cuda_count()} b70={p.b70_count()} "
            f"sum={p.cuda_count() + p.b70_count()} bank_covers={covers}",
            p.cuda_count() + p.b70_count() == total and covers,
        )

    # all-cuda has zero B70
    p_all = build_placement(
        AllCudaPolicy(), num_layers=nl, num_experts=ne, b70_capable=capable
    )
    expect_ok("all-cuda: zero B70 owners", p_all.b70_count() == 0)

    # split:128 -> 32 capable layers x 128 B70 = 4096 B70
    p_split = build_placement(
        SplitPolicy(128), num_layers=nl, num_experts=ne, b70_capable=capable
    )
    expect_ok("split:128 -> 4096 B70", p_split.b70_count() == 32 * 128)
    # FP8 layers 32-39 are CUDA-forced
    expect_ok(
        "FP8 layers fully CUDA",
        all(
            p_split.owners[layer][e].device is Device.CUDA
            for layer in range(q.bank_layers, nl)
            for e in range(ne)
        ),
    )

    # ---- subset: same capacity, fewer B70-active layers -----------------
    # A B70-active layer costs one dispatch per token, and that cost is
    # fixed regardless of how many routes it carries.  subset trades
    # experts-per-layer against active-layer count to hold capacity while
    # paying the fixed cost fewer times.
    p_subset = build_placement(
        LayerSubsetPolicy(active_layers=16, cuda_per_layer=8),
        num_layers=nl, num_experts=ne, b70_capable=capable,
    )
    validate_placement(p_subset)
    active = p_subset.b70_active_layers()
    expect_ok("subset:16:8 -> 16 active layers", len(active) == 16)
    expect_ok(
        "subset active layers are the last capable ones",
        active == tuple(range(16, 32)),
    )
    expect_ok(
        "subset:16:8 -> 16 x 248 B70", p_subset.b70_count() == 16 * (ne - 8)
    )
    expect_ok(
        "subset leaves early capable layers all-CUDA",
        all(
            p_subset.owners[layer][e].device is Device.CUDA
            for layer in range(16)
            for e in range(ne)
        ),
    )
    expect_ok(
        "subset inactive layer reports is_b70_active False",
        not p_subset.is_b70_active(0) and p_subset.is_b70_active(31),
    )
    # Every active layer must offload the same expert ids: the provider
    # uploads one resident set for the whole bank.
    resident = tuple(
        e for e in range(ne)
        if p_subset.owners[active[0]][e].device is Device.B70
    )
    expect_ok(
        "subset active layers share one resident set",
        all(
            tuple(
                e for e in range(ne)
                if p_subset.owners[layer][e].device is Device.B70
            ) == resident
            for layer in active
        ),
    )
    bank_layers, bank_experts = read_bank_header()
    expect_ok(
        "subset covered by bank",
        b70_bank_covers(
            p_subset,
            bank_layers=bank_layers,
            bank_experts_per_layer=bank_experts,
        ),
    )

    # ---- generation bump (swappable seam) ------------------------------
    p_gen2 = build_placement(
        SplitPolicy(96), num_layers=nl, num_experts=ne, b70_capable=capable, generation=2
    )
    expect_ok("generation bump preserved", p_gen2.generation == 2)

    # ---- manifest round-trip -------------------------------------------
    rt = Placement.from_manifest(p_split.to_manifest())
    expect_ok("manifest JSON round-trip identical", rt == p_split)

    # ---- build_for_qualified end-to-end --------------------------------
    for spec in ("all-cuda", "split:128", "interleaved:4"):
        p = build_for_qualified(q, policy_name=spec)
        validate_placement(p)
        print(f"  ok  build_for_qualified({spec}) validated")
    # policy_from_name consistency
    expect_ok(
        "policy_from_name(split:128) == SplitPolicy(128)",
        policy_from_name("split:128").cuda_per_layer == 128,
    )

    # ---- negatives -----------------------------------------------------
    # B70 ownership in an FP8 layer must fail
    bad_rows = list(p_split.owners)
    bad_fp8 = list(bad_rows[35])
    bad_fp8[0] = bad_fp8[0].__class__(Device.B70, 0)
    bad_rows[35] = tuple(bad_fp8)
    bad = Placement(
        generation=1, num_layers=nl, num_experts=ne,
        owners=tuple(bad_rows), b70_capable_layers=capable, policy_name="bad-fp8",
    )
    expect_raise("B70 in FP8 layer rejected", lambda: validate_placement(bad))

    # non-dense slots must fail
    bad2_rows = list(p_all.owners)
    bad2_row = list(bad2_rows[0])
    bad2_row[5] = bad2_row[5].__class__(Device.CUDA, 999)
    bad2_rows[0] = tuple(bad2_row)
    bad2 = Placement(
        generation=1, num_layers=nl, num_experts=ne,
        owners=tuple(bad2_rows), b70_capable_layers=capable, policy_name="bad-slot",
    )
    expect_raise("non-dense CUDA slot rejected", lambda: validate_placement(bad2))

    # unknown policy must fail
    expect_raise("unknown policy rejected", lambda: policy_from_name("nonsense"))

    print("\nPhase-5 compact expert-ownership gate PASS")


if __name__ == "__main__":
    main()
