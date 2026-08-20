#!/usr/bin/env python3
"""Gate for the prefill DRAM mirror's selection, slot map, and sizing.

The mirror duplicates a subset of B70-resident experts into host DRAM so
prefill can compute them on the 5090 while decode keeps using the doorbell.
Three properties decide whether it helps or silently wastes memory, and all
three are checked here without a GPU:

1. **Balance.** A layer's B70 leg costs ``max(dev0, dev1)``. Mirroring only
   one card's experts lowers that card and leaves the maximum untouched, so
   the selection must draw evenly from both cards or the whole feature buys
   nothing.
2. **Slot map.** Downstream masking uses the CPU tier's convention: ``-1``
   means "owned elsewhere". A mirrored expert must map to a dense arena slot
   and every other expert to ``-1``, or the streamer reads the wrong weights.
3. **Sizing.** Residency already costs host DRAM 1:1, so the arena has to be
   refused *before* allocation. Overcommitting this box measured a 25x
   collapse in host memory throughput, which would look like a slow streamer
   rather than memory pressure.

Run: .venv/bin/python src/phase6/prefill_mirror_unit_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase4" / "src"))

from shooting_brake_vllm.placement import (  # noqa: E402
    Device,
    ExpertOwner,
    Placement,
)
from shooting_brake_vllm import prefill_mirror as pm  # noqa: E402

FAILURES: list[str] = []


def check(label: str, ok: bool) -> None:
    print(f"{'ok' if ok else 'FAIL'} {label}")
    if not ok:
        FAILURES.append(label)


def make_placement(
    num_layers: int = 4, num_experts: int = 205, local: int = 54,
    devices: int = 2,
) -> Placement:
    """The production shape: remote experts split evenly across ``devices``,
    then the local tail on CUDA -- matching FractionalRemotePolicy, which
    hands CUDA the high ids.
    """
    remote = num_experts - local
    per_device = [remote // devices] * devices
    for i in range(remote % devices):
        per_device[i] += 1

    rows = []
    for _ in range(num_layers):
        row: list[ExpertOwner] = []
        expert = 0
        for d in range(devices):
            for slot in range(per_device[d]):
                row.append(ExpertOwner(Device.B70, slot, d))
                expert += 1
        for slot in range(local):
            row.append(ExpertOwner(Device.CUDA, slot))
            expert += 1
        assert expert == num_experts
        rows.append(tuple(row))
    return Placement(
        generation=1, num_layers=num_layers, num_experts=num_experts,
        owners=tuple(rows), b70_capable_layers=frozenset(range(num_layers)),
        policy_name="test",
    )


def device_of(placement: Placement, layer: int, expert: int) -> int:
    return placement.owners[layer][expert].device_index


def main() -> int:
    p = make_placement()
    n_remote = p.num_experts - 54

    # -- 1. balance ------------------------------------------------------
    ids = pm.select_mirror_ids(p, 0, 60)
    check("selects exactly the requested count", len(ids) == 60)
    check("selection is unique", len(set(ids)) == len(ids))
    check("selection is ascending", list(ids) == sorted(ids))
    check(
        "every selected expert is B70-owned",
        all(p.owners[0][e].device is Device.B70 for e in ids),
    )
    per_dev = {0: 0, 1: 0}
    for e in ids:
        per_dev[device_of(p, 0, e)] += 1
    check(
        f"balanced across cards ({per_dev[0]} vs {per_dev[1]})",
        abs(per_dev[0] - per_dev[1]) <= 1,
    )

    # An odd count still cannot skew by more than one.
    odd = pm.select_mirror_ids(p, 0, 37)
    skew = {0: 0, 1: 0}
    for e in odd:
        skew[device_of(p, 0, e)] += 1
    check(
        f"odd count stays balanced ({skew[0]} vs {skew[1]})",
        abs(skew[0] - skew[1]) <= 1 and len(odd) == 37,
    )

    # -- 2. degenerate requests -----------------------------------------
    check("zero request selects nothing", pm.select_mirror_ids(p, 0, 0) == ())
    check(
        "negative request selects nothing",
        pm.select_mirror_ids(p, 0, -5) == (),
    )
    everything = pm.select_mirror_ids(p, 0, n_remote + 100)
    check(
        f"over-request clamps to the {n_remote} remote experts",
        len(everything) == n_remote,
    )
    check(
        "over-request never takes a CUDA expert",
        all(p.owners[0][e].device is Device.B70 for e in everything),
    )

    # A single-card placement must still work (and is trivially balanced).
    solo = make_placement(devices=1)
    solo_ids = pm.select_mirror_ids(solo, 0, 10)
    check("single-card placement selects", len(solo_ids) == 10)

    # An all-CUDA layer has nothing to mirror.
    all_cuda = Placement(
        generation=1, num_layers=1, num_experts=8,
        owners=((tuple(ExpertOwner(Device.CUDA, s) for s in range(8))),),
        b70_capable_layers=frozenset(),
        policy_name="all-cuda",
    )
    check(
        "all-CUDA layer mirrors nothing",
        pm.select_mirror_ids(all_cuda, 0, 4) == (),
    )

    # -- 3. slot map -----------------------------------------------------
    slots = pm.build_slot_map(ids, p.num_experts)
    check("slot map covers every expert", len(slots) == p.num_experts)
    check(
        "mirrored experts get dense slots 0..N-1",
        sorted(slots[e] for e in ids) == list(range(len(ids))),
    )
    check(
        "non-mirrored experts are -1",
        all(slots[e] == -1 for e in range(p.num_experts) if e not in set(ids)),
    )
    check(
        "slot order follows selection order",
        all(slots[e] == i for i, e in enumerate(ids)),
    )
    try:
        pm.build_slot_map((999,), p.num_experts)
        check("out-of-range expert rejected", False)
    except ValueError:
        check("out-of-range expert rejected", True)

    # -- 4. sizing arithmetic -------------------------------------------
    # NVFP4: half a byte per weight plus one e4m3 scale per 16, three planes.
    hidden, inter, layers = 3072, 1024, 48
    plane = hidden * inter
    expect_per_expert = 3 * (plane // 2 + plane // 16)
    got = pm.arena_gib(1, layers, hidden, inter)
    check(
        f"per-expert arena size ({got * 2**30 / layers:.0f} B/layer)",
        abs(got - expect_per_expert * layers / 2**30) < 1e-9,
    )
    check(
        "arena size scales linearly in experts",
        abs(pm.arena_gib(64, layers, hidden, inter)
            - 64 * pm.arena_gib(1, layers, hidden, inter)) < 1e-9,
    )
    # Production geometry: one expert is ~0.2373 GiB across 48 layers.
    one = pm.arena_gib(1, layers, hidden, inter)
    check(f"one expert is {one:.4f} GiB over 48 layers", 0.23 < one < 0.245)

    # -- 5. budget guard -------------------------------------------------
    have = pm.host_available_gib()
    check(f"MemAvailable readable ({have:.2f} GiB)", have > 0)
    fits = pm.validate_mirror_budget(1, layers, hidden, inter)
    check("a one-expert mirror is accepted", fits > 0)
    try:
        pm.validate_mirror_budget(100000, layers, hidden, inter)
        check("oversized mirror refused", False)
    except RuntimeError as exc:
        check("oversized mirror refused", "host DRAM" in str(exc))
    cap = pm.max_mirror_experts(layers, hidden, inter)
    check(f"max_mirror_experts reports a bound ({cap})", cap >= 0)
    if cap > 0:
        try:
            pm.validate_mirror_budget(cap, layers, hidden, inter)
            check("the reported bound itself fits", True)
        except RuntimeError:
            check("the reported bound itself fits", False)
        try:
            pm.validate_mirror_budget(cap + 8, layers, hidden, inter)
            check("one past the bound is refused", False)
        except RuntimeError:
            check("one past the bound is refused", True)

    # -- 6. every layer is selected independently ------------------------
    check(
        "selection is stable across layers",
        all(pm.select_mirror_ids(p, l, 60) == ids
            for l in range(p.num_layers)),
    )

    print()
    if FAILURES:
        print(f"prefill-mirror unit-test FAIL ({len(FAILURES)}): {FAILURES}")
        return 1
    print("prefill-mirror unit-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
