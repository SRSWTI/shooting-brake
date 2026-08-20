"""Prefill-only DRAM mirror of a subset of B70-resident experts.

Why a mirror and not a placement tier: decode needs every remote expert
resident on a B70 -- dispatch beats streaming by ~9x at M=1 -- while prefill
wants those same experts on the 5090, because the B70's per-route GEMV
re-reads each expert ~10x at M=256 (measured reuse 9.99x, kill-bench 18 §2).
The placement model gives each expert exactly one owner, so it cannot express
"B70 for decode, DRAM for prefill". This module names the subset that is
*duplicated* into host DRAM and computed on the 5090 during prefill only.
Placement, decode, and the doorbell path are untouched.

Sizing is a hard constraint, all measured 2026-08-20 (kill-bench 18):

* B70 device residency costs host DRAM **1:1** and the cost is committed, not
  accounted -- holding 24 GiB on each card consumed 48.17 GiB of host, and a
  follow-on host allocation ran at 0.27 GiB/s (swap thrash).
* So a mirrored expert costs DRAM **twice**: 0.2373 GiB of residency shadow
  (so decode can reach it) plus its own arena copy.
* At L=54 local experts the shadow is 36.1 GiB of 59.44 and MemAvailable is
  15.29 GiB. The existing all-or-nothing streamer sizes its arena for *every*
  B70-owned expert, which needs L>=102 => 34.3 GiB of CUDA weights on a 31.84
  GiB card. It can never fit. Hence: a subset.

Balance is load-bearing and easy to get wrong. A layer's B70 leg costs
``max(dev0, dev1)``, so mirroring only one card's experts lowers that card's
share and leaves the maximum untouched -- buying nothing. The subset is
therefore drawn evenly from each card, round-robin over per-device lists.

The subset is chosen at runtime, so more host RAM later means a larger
mirror from the same source bank with no rebuild: raise
``SHOOTING_BRAKE_PREFILL_MIRROR`` and the arena grows.
"""

from __future__ import annotations

import os
from pathlib import Path

from .placement import Device, Placement

_ENV = "SHOOTING_BRAKE_PREFILL_MIRROR"
_ENV_HEADROOM = "SHOOTING_BRAKE_PREFILL_MIRROR_HEADROOM_GIB"

#: Default DRAM left unclaimed after the arena. The engine, CUDA's pinned
#: staging, and the page cache all live here; at 2.78 GiB free the box was
#: already swapping (kill-bench 17 §8), so the guard is not decorative.
DEFAULT_HEADROOM_GIB = 3.0


def mirror_expert_count() -> int:
    """Experts per layer to mirror into DRAM. ``0`` disables the mirror."""
    raw = os.environ.get(_ENV, "0")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{_ENV} must be an integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{_ENV} must be >= 0, got {value}")
    return value


def mirror_enabled() -> bool:
    return mirror_expert_count() > 0


def headroom_gib() -> float:
    return float(os.environ.get(_ENV_HEADROOM, DEFAULT_HEADROOM_GIB))


def host_available_gib() -> float:
    """``MemAvailable``, the kernel's own estimate of what a new allocation
    can take without swapping -- which is exactly the question here."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    raise RuntimeError("MemAvailable missing from /proc/meminfo")


def select_mirror_ids(
    placement: Placement, layer: int, count: int,
) -> tuple[int, ...]:
    """Global expert ids to mirror for ``layer``, balanced across B70 cards.

    Round-robin over per-device lists so each card sheds the same number of
    experts. Returns ascending ids -- the arena is keyed by compact slot, so
    the on-disk order of the selection does not need to be contiguous, but a
    stable order keeps slot maps reproducible across boots.
    """
    if count <= 0:
        return ()
    by_device: dict[int, list[int]] = {}
    for expert, owner in enumerate(placement.owners[layer]):
        if owner.device is Device.B70:
            by_device.setdefault(owner.device_index, []).append(expert)
    if not by_device:
        return ()

    picked: list[int] = []
    devices = sorted(by_device)
    cursors = {d: 0 for d in devices}
    while len(picked) < count:
        progressed = False
        for d in devices:
            if len(picked) >= count:
                break
            ids = by_device[d]
            i = cursors[d]
            if i < len(ids):
                picked.append(ids[i])
                cursors[d] = i + 1
                progressed = True
        if not progressed:
            break                     # every card exhausted
    return tuple(sorted(picked))


def build_slot_map(
    mirror_ids: tuple[int, ...], num_experts: int,
) -> list[int]:
    """``global expert id -> arena slot``, ``-1`` for everything else.

    Same shape and convention as the CPU tier's ``_cpu_id_map``, so the
    downstream masking (`-1` means "owned elsewhere") is unchanged.
    """
    slots = [-1] * num_experts
    for slot, expert in enumerate(mirror_ids):
        if not 0 <= expert < num_experts:
            raise ValueError(
                f"mirror expert id {expert} outside [0, {num_experts})"
            )
        slots[expert] = slot
    return slots


def arena_gib(
    count: int, num_layers: int, hidden: int, intermediate: int,
) -> float:
    """Host DRAM the arena will occupy, in GiB.

    NVFP4 packing, three planes per expert: half a byte per weight plus one
    e4m3 block scale per 16 weights -- the same arithmetic the arena itself
    uses, kept here so sizing can be checked before any allocation.
    """
    plane = hidden * intermediate
    per_expert = 3 * (plane // 2 + plane // 16)
    return count * num_layers * per_expert / 2**30


def validate_mirror_budget(
    count: int, num_layers: int, hidden: int, intermediate: int,
) -> float:
    """Raise unless the arena fits in ``MemAvailable`` minus headroom.

    Refusing up front beats discovering it by swapping: the measured penalty
    for overcommitting this box is a 25x collapse in host memory throughput,
    which would be attributed to the streamer rather than to memory pressure.
    """
    want = arena_gib(count, num_layers, hidden, intermediate)
    have = host_available_gib()
    room = have - headroom_gib()
    if want > room:
        raise RuntimeError(
            f"prefill mirror of {count} experts/layer needs {want:.2f} GiB of "
            f"host DRAM but only {room:.2f} GiB is safely available "
            f"(MemAvailable {have:.2f} GiB minus {headroom_gib():.2f} GiB "
            f"headroom). Lower {_ENV}, raise the local-expert count so B70 "
            f"residency releases its 1:1 shadow, or add host RAM."
        )
    return want


def max_mirror_experts(
    num_layers: int, hidden: int, intermediate: int,
) -> int:
    """Largest mirror that fits right now, for reporting and for --auto."""
    plane = hidden * intermediate
    per_expert_gib = 3 * (plane // 2 + plane // 16) * num_layers / 2**30
    room = host_available_gib() - headroom_gib()
    return max(0, int(room / per_expert_gib))


__all__ = [
    "arena_gib",
    "build_slot_map",
    "headroom_gib",
    "host_available_gib",
    "max_mirror_experts",
    "mirror_enabled",
    "mirror_expert_count",
    "select_mirror_ids",
    "validate_mirror_budget",
]
