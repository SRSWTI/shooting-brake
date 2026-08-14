"""Versioned expert ownership for CUDA, remote accelerators, and CPU.

The manifest records exactly one normal-path owner for every
``(layer, global_expert)``.  A remote owner carries an explicit device index,
so placement is not coupled to the current dispatch implementation's
single-B70 limitation.  Existing policies continue to target B70 device zero;
multi-device placement is opt-in through :class:`ExpertGroupPolicy` or
:class:`FractionalRemotePolicy`.

The ownership table is immutable during an inference step and swappable at
coarse request/generation boundaries.  It is deliberately separate from
vLLM's expert-parallel ranks: Arc devices are external accelerators, not fake
vLLM ranks.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol


class Device(str, Enum):
    CUDA = "cuda"
    B70 = "b70"
    #: Host DRAM, computed on CPU cores. Reachable only in "all-out mode"
    #: (``SHOOTING_BRAKE_ALL_OUT=1``); every default policy leaves it unused,
    #: which is what keeps the no-normal-path-CPU-compute invariant intact
    #: outside that mode. Roughly 5x the B70's per-expert cost, so it is the
    #: cold tier and a latency bug for any expert that actually fires --
    #: see ``src/phase7/cpu_expert_capi.h`` for the measured cost model.
    CPU = "cpu"


@dataclass(frozen=True, order=True)
class DeviceTarget:
    """A concrete device, including its index within a device kind."""

    device: Device
    index: int = 0

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"device index must be non-negative, got {self.index}")
        if self.device in (Device.CUDA, Device.CPU) and self.index != 0:
            raise ValueError(
                f"{self.device.value} device index {self.index} is not supported"
            )

    @property
    def label(self) -> str:
        return f"{self.device.value}:{self.index}"


@dataclass(frozen=True)
class DeviceCapacity:
    """Resident expert-weight budget for one concrete device."""

    target: DeviceTarget
    capacity_bytes: int
    bytes_per_expert: int

    def __post_init__(self) -> None:
        if self.capacity_bytes < 0:
            raise ValueError("capacity_bytes must be non-negative")
        if self.bytes_per_expert < 1:
            raise ValueError("bytes_per_expert must be positive")

    def to_dict(self) -> dict:
        return {
            "device": self.target.device.value,
            "device_index": self.target.index,
            "capacity_bytes": self.capacity_bytes,
            "bytes_per_expert": self.bytes_per_expert,
        }

    @classmethod
    def from_dict(cls, value: dict) -> DeviceCapacity:
        return cls(
            target=DeviceTarget(
                Device(value["device"]), int(value.get("device_index", 0))
            ),
            capacity_bytes=int(value["capacity_bytes"]),
            bytes_per_expert=int(value["bytes_per_expert"]),
        )


@dataclass(frozen=True)
class ExpertOwner:
    """The single normal-path owner of one ``(layer, global_expert)``."""

    device: Device
    slot: int  # compact device-local slot, dense from 0
    device_index: int = 0

    def __post_init__(self) -> None:
        DeviceTarget(self.device, self.device_index)

    @property
    def target(self) -> DeviceTarget:
        return DeviceTarget(self.device, self.device_index)

    def to_dict(self) -> dict:
        result = {"device": self.device.value, "slot": self.slot}
        # Preserve byte-for-byte v1 manifests for every existing policy.
        if self.device_index:
            result["device_index"] = self.device_index
        return result

    @classmethod
    def from_dict(cls, d: dict) -> ExpertOwner:
        return cls(
            device=Device(d["device"]),
            slot=int(d["slot"]),
            device_index=int(d.get("device_index", 0)),
        )


@dataclass(frozen=True)
class Placement:
    """Immutable ownership table for every routed expert in the model.

    ``owners[layer][global_expert]`` is complete by construction.  Remote
    devices are distinguished by :attr:`ExpertOwner.device_index`.
    ``device_capacities`` is optional so legacy placements retain their
    behaviour; when supplied, validation rejects resident expert weights that
    exceed any concrete device's budget.
    """

    generation: int
    num_layers: int
    num_experts: int
    owners: tuple[tuple[ExpertOwner, ...], ...]
    b70_capable_layers: frozenset[int]
    policy_name: str = ""
    device_capacities: tuple[DeviceCapacity, ...] = ()

    # -- introspection ---------------------------------------------------

    def count(self, device: Device) -> int:
        return sum(
            1 for layer in self.owners for owner in layer if owner.device is device
        )

    def count_target(self, target: DeviceTarget) -> int:
        return sum(
            1 for layer in self.owners for owner in layer
            if owner.target == target
        )

    def remote_device_indices(self) -> tuple[int, ...]:
        return tuple(sorted({
            owner.device_index
            for layer in self.owners
            for owner in layer
            if owner.device is Device.B70
        }))

    def cuda_count(self) -> int:
        return self.count(Device.CUDA)

    def b70_count(self) -> int:
        return self.count(Device.B70)

    def is_b70_capable(self, layer: int) -> bool:
        return layer in self.b70_capable_layers

    def layer_b70_count(self, layer: int) -> int:
        return sum(
            1 for owner in self.owners[layer] if owner.device is Device.B70
        )

    def b70_active_layers(self) -> tuple[int, ...]:
        """Layers that actually own B70 experts.

        Distinct from :attr:`b70_capable_layers`, which only says the
        expert bank covers that layer. A policy may concentrate B70
        ownership into a subset of capable layers — every active layer
        costs a dispatch per token, so the count matters for latency,
        while total B70-owned experts is what determines freed VRAM.
        """
        return tuple(
            layer for layer in range(self.num_layers)
            if self.layer_b70_count(layer)
        )

    def is_b70_active(self, layer: int) -> bool:
        return self.layer_b70_count(layer) > 0

    def cpu_count(self) -> int:
        return self.count(Device.CPU)

    def layer_cpu_count(self, layer: int) -> int:
        return sum(
            1 for owner in self.owners[layer] if owner.device is Device.CPU
        )

    def cpu_active_layers(self) -> tuple[int, ...]:
        """Layers that actually own CPU-tier experts.

        Empty for every default policy — the CPU tier only appears under
        an all-out placement. Same latency reasoning as
        :meth:`b70_active_layers`, but the per-dispatch cost is ~5x
        higher, so the count matters correspondingly more.
        """
        return tuple(
            layer for layer in range(self.num_layers)
            if self.layer_cpu_count(layer)
        )

    def is_cpu_active(self, layer: int) -> bool:
        return self.layer_cpu_count(layer) > 0

    def is_cpu_expert(self, layer: int, expert: int) -> bool:
        return self.owners[layer][expert].device is Device.CPU

    def cpu_expert_ids(self, layer: int) -> tuple[int, ...]:
        """Global expert ids this layer keeps in host DRAM."""
        return tuple(
            e for e, owner in enumerate(self.owners[layer])
            if owner.device is Device.CPU
        )

    def b70_expert_ids(self, layer: int) -> tuple[int, ...]:
        """Global expert ids this layer keeps on the B70.

        Used by the prefill weight streamer, which needs a host-side copy of
        exactly these experts: at prefill the B70's kernel costs more than
        moving its weights to the 5090, so the same arena that backs the CPU
        tier also holds the B70 set. Distinct from the provider's own compact
        slot numbering, which is per-layer and dense.
        """
        return tuple(
            e for e, owner in enumerate(self.owners[layer])
            if owner.device is Device.B70
        )

    def cuda_expert_ids(self, layer: int) -> tuple[int, ...]:
        """Global expert ids this layer keeps in CUDA VRAM, ascending.

        The compact local id of a CUDA expert is its position in this
        tuple, which is what makes ``index_select`` on a sorted id list and
        a from-scratch compact allocation produce identical layouts. Both
        surgery strategies read ownership from here so they cannot drift.
        """
        return tuple(
            e for e, owner in enumerate(self.owners[layer])
            if owner.device is Device.CUDA
        )

    # -- (de)serialization for data-driven swapping ----------------------

    SCHEMA = "shooting-brake.placement.v1"
    MULTI_DEVICE_SCHEMA = "shooting-brake.placement.v2"

    def to_manifest(self) -> dict:
        multi_device = bool(self.device_capacities) or any(
            owner.device_index
            for layer in self.owners
            for owner in layer
        )
        result = {
            "schema": self.MULTI_DEVICE_SCHEMA if multi_device else self.SCHEMA,
            "generation": self.generation,
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "b70_capable_layers": sorted(self.b70_capable_layers),
            "policy": self.policy_name,
            "owners": [
                [owner.to_dict() for owner in layer] for layer in self.owners
            ],
        }
        if self.device_capacities:
            result["device_capacities"] = [
                capacity.to_dict() for capacity in self.device_capacities
            ]
        return result

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_manifest()), encoding="utf-8")

    @classmethod
    def from_manifest(cls, manifest: dict) -> Placement:
        supported = (cls.SCHEMA, cls.MULTI_DEVICE_SCHEMA)
        if manifest.get("schema") not in supported:
            raise ValueError(f"unsupported schema {manifest.get('schema')!r}")
        owners = tuple(
            tuple(ExpertOwner.from_dict(o) for o in layer)
            for layer in manifest["owners"]
        )
        placement = cls(
            generation=int(manifest["generation"]),
            num_layers=int(manifest["num_layers"]),
            num_experts=int(manifest["num_experts"]),
            owners=owners,
            b70_capable_layers=frozenset(manifest["b70_capable_layers"]),
            policy_name=str(manifest.get("policy", "")),
            device_capacities=tuple(
                DeviceCapacity.from_dict(value)
                for value in manifest.get("device_capacities", ())
            ),
        )
        validate_placement(placement)
        return placement

    @classmethod
    def from_json(cls, path: str | Path) -> Placement:
        return cls.from_manifest(json.loads(Path(path).read_text(encoding="utf-8")))


class PlacementPolicy(Protocol):
    """Produces an owner table from the model geometry.

    A predictive / speculative offloader implements this interface and is
    free to consult access-frequency statistics, KV-cache pressure, or any
    other signal. The only contract is the returned table; the hot path never
    sees the policy.
    """

    name: str

    def assign(
        self,
        num_layers: int,
        num_experts: int,
        b70_capable: frozenset[int],
    ) -> tuple[tuple[ExpertOwner, ...], ...]:
        """Return ``owners[layer][global_expert]`` for every layer."""
        ...


# --------------------------------------------------------------------------
# Built-in policies
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AllCudaPolicy:
    """Every expert stays on CUDA. The Phase-4-equivalent degenerate baseline."""

    name: str = "all-cuda"

    def assign(
        self,
        num_layers: int,
        num_experts: int,
        b70_capable: frozenset[int],
    ) -> tuple[tuple[ExpertOwner, ...], ...]:
        return tuple(
            tuple(ExpertOwner(Device.CUDA, e) for e in range(num_experts))
            for _ in range(num_layers)
        )


@dataclass(frozen=True)
class ExpertGroup:
    """Half-open global expert-id range assigned to one concrete device."""

    start: int
    stop: int
    target: DeviceTarget

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop <= self.start:
            raise ValueError(
                f"expert group must be a non-empty half-open range, "
                f"got [{self.start},{self.stop})"
            )


def validate_expert_groups(
    groups: Sequence[ExpertGroup], num_experts: int,
) -> None:
    """Reject gaps, overlap, and ranges outside ``[0, num_experts)``."""
    if num_experts < 1:
        raise PlacementError("num_experts must be positive")
    covered = [False] * num_experts
    for group in groups:
        if group.stop > num_experts:
            raise PlacementError(
                f"expert group [{group.start},{group.stop}) exceeds "
                f"expert count {num_experts}"
            )
        for expert in range(group.start, group.stop):
            if covered[expert]:
                raise PlacementError(
                    f"expert {expert} is assigned by more than one group"
                )
            covered[expert] = True
    missing = [expert for expert, assigned in enumerate(covered) if not assigned]
    if missing:
        raise PlacementError(
            f"expert groups do not cover {len(missing)} experts; "
            f"first missing expert is {missing[0]}"
        )


@dataclass(frozen=True)
class ExpertGroupPolicy:
    """Apply explicit, complete expert ranges to every capable layer.

    For example, three groups can express ``[0,60) -> b70:0``,
    ``[60,120) -> b70:1``, and ``[120,180) -> cuda:0`` without relying on
    the expert count being a power of two or evenly divisible.
    """

    groups: tuple[ExpertGroup, ...]
    name: str = "expert-groups"

    def assign(
        self,
        num_layers: int,
        num_experts: int,
        b70_capable: frozenset[int],
    ) -> tuple[tuple[ExpertOwner, ...], ...]:
        validate_expert_groups(self.groups, num_experts)
        target_by_expert: list[DeviceTarget | None] = [None] * num_experts
        for group in self.groups:
            target_by_expert[group.start:group.stop] = [
                group.target
            ] * (group.stop - group.start)

        slots: dict[DeviceTarget, int] = {}
        owners: list[ExpertOwner] = []
        for target in target_by_expert:
            assert target is not None
            slot = slots.get(target, 0)
            owners.append(ExpertOwner(target.device, slot, target.index))
            slots[target] = slot + 1
        grouped_row = tuple(owners)
        all_cuda = tuple(
            ExpertOwner(Device.CUDA, expert) for expert in range(num_experts)
        )
        return tuple(
            grouped_row if layer in b70_capable else all_cuda
            for layer in range(num_layers)
        )


@dataclass(frozen=True)
class FractionalRemotePolicy:
    """Keep a fraction on CUDA and balance the rest across N B70 devices."""

    remote_device_indices: tuple[int, ...]
    cuda_fraction: float
    name: str = field(init=False, default="fractional-remote")

    def __post_init__(self) -> None:
        if not self.remote_device_indices:
            raise ValueError("at least one remote device is required")
        if len(set(self.remote_device_indices)) != len(self.remote_device_indices):
            raise ValueError("remote device indices must be unique")
        for index in self.remote_device_indices:
            DeviceTarget(Device.B70, index)
        if not 0.0 <= self.cuda_fraction <= 1.0:
            raise ValueError("cuda_fraction must be between zero and one")
        object.__setattr__(
            self,
            "name",
            f"fractional-remote:devices={','.join(map(str, self.remote_device_indices))}"
            f",cuda_fraction={self.cuda_fraction}",
        )

    def groups(self, num_experts: int) -> tuple[ExpertGroup, ...]:
        cuda_n = round(num_experts * self.cuda_fraction)
        remote_n = num_experts - cuda_n
        quotient, remainder = divmod(remote_n, len(self.remote_device_indices))
        groups: list[ExpertGroup] = []
        start = 0
        for position, index in enumerate(self.remote_device_indices):
            count = quotient + (1 if position < remainder else 0)
            if count:
                groups.append(
                    ExpertGroup(
                        start,
                        start + count,
                        DeviceTarget(Device.B70, index),
                    )
                )
                start += count
        if cuda_n:
            groups.append(
                ExpertGroup(start, num_experts, DeviceTarget(Device.CUDA))
            )
        return tuple(groups)

    def assign(
        self,
        num_layers: int,
        num_experts: int,
        b70_capable: frozenset[int],
    ) -> tuple[tuple[ExpertOwner, ...], ...]:
        return ExpertGroupPolicy(self.groups(num_experts), self.name).assign(
            num_layers, num_experts, b70_capable
        )


@dataclass(frozen=True)
class SplitPolicy:
    """First ``cuda_per_layer`` experts of each B70-capable layer stay on CUDA;
    the remainder move to B70. FP8 (non-capable) layers are all-CUDA.

    This is the simplest non-trivial static placement and a reasonable default:
    it keeps the low-id experts hot on CUDA and offloads the cold tail to B70.
    """

    cuda_per_layer: int
    name: str = field(init=False, default="split")

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name", f"split:cuda_per_layer={self.cuda_per_layer}"
        )

    def assign(
        self,
        num_layers: int,
        num_experts: int,
        b70_capable: frozenset[int],
    ) -> tuple[tuple[ExpertOwner, ...], ...]:
        rows: list[tuple[ExpertOwner, ...]] = []
        for layer in range(num_layers):
            if layer in b70_capable:
                cuda_n = max(0, min(self.cuda_per_layer, num_experts))
                row = tuple(
                    ExpertOwner(Device.CUDA, e) for e in range(cuda_n)
                ) + tuple(
                    ExpertOwner(Device.B70, e - cuda_n)
                    for e in range(cuda_n, num_experts)
                )
            else:
                row = tuple(
                    ExpertOwner(Device.CUDA, e) for e in range(num_experts)
                )
            rows.append(row)
        return tuple(rows)


@dataclass(frozen=True)
class InterleavedPolicy:
    """Every ``period``-th expert in a B70-capable layer goes to B70; the rest
    stay on CUDA. Spreads B70 ownership evenly across the expert id space.
    """

    period: int
    name: str = field(init=False, default="interleaved")

    def __post_init__(self) -> None:
        if self.period < 2:
            raise ValueError("InterleavedPolicy.period must be >= 2")
        object.__setattr__(self, "name", f"interleaved:period={self.period}")

    def assign(
        self,
        num_layers: int,
        num_experts: int,
        b70_capable: frozenset[int],
    ) -> tuple[tuple[ExpertOwner, ...], ...]:
        rows: list[tuple[ExpertOwner, ...]] = []
        for layer in range(num_layers):
            if layer in b70_capable:
                cuda_slot = 0
                b70_slot = 0
                owners: list[ExpertOwner] = []
                for e in range(num_experts):
                    if e % self.period == self.period - 1:
                        owners.append(ExpertOwner(Device.B70, b70_slot))
                        b70_slot += 1
                    else:
                        owners.append(ExpertOwner(Device.CUDA, cuda_slot))
                        cuda_slot += 1
                rows.append(tuple(owners))
            else:
                rows.append(
                    tuple(
                        ExpertOwner(Device.CUDA, e) for e in range(num_experts)
                    )
                )
        return tuple(rows)


@dataclass(frozen=True)
class LayerSubsetPolicy:
    """Concentrate B70 ownership into the last ``active_layers`` capable
    layers; every other layer stays entirely on CUDA.

    Motivation is latency, not capacity. Each B70-active layer costs one
    dispatch per token, and that dispatch has a fixed overhead that does
    not shrink with the number of routes. Spreading a fixed number of
    offloaded experts across all 32 capable layers therefore pays the
    fixed cost 32 times; concentrating the same experts into K layers
    pays it K times, for the same freed VRAM.

    The last layers are chosen because early layers' routing is the more
    load-bearing, and because it keeps the active set contiguous.

    Every active layer offloads the same expert ids, which the provider
    requires: one resident set is uploaded for the whole bank.
    """

    active_layers: int
    cuda_per_layer: int
    name: str = field(init=False, default="subset")

    def __post_init__(self) -> None:
        if self.active_layers < 1:
            raise ValueError("LayerSubsetPolicy.active_layers must be >= 1")
        object.__setattr__(
            self, "name",
            f"subset:active_layers={self.active_layers}"
            f",cuda_per_layer={self.cuda_per_layer}",
        )

    def assign(
        self,
        num_layers: int,
        num_experts: int,
        b70_capable: frozenset[int],
    ) -> tuple[tuple[ExpertOwner, ...], ...]:
        active = set(sorted(b70_capable)[-self.active_layers:])
        all_cuda = tuple(
            ExpertOwner(Device.CUDA, e) for e in range(num_experts)
        )
        cuda_n = max(0, min(self.cuda_per_layer, num_experts))
        offloaded = tuple(
            ExpertOwner(Device.CUDA, e) for e in range(cuda_n)
        ) + tuple(
            ExpertOwner(Device.B70, e - cuda_n)
            for e in range(cuda_n, num_experts)
        )
        return tuple(
            offloaded if layer in active else all_cuda
            for layer in range(num_layers)
        )


@dataclass(frozen=True)
class AllOutPolicy:
    """Three-tier placement: CUDA hot, B70 warm, host DRAM cold.

    The only policy that assigns :attr:`Device.CPU`, and therefore the only
    one that puts expert matrix compute on the CPU in the normal path. That
    is a deliberate, opt-in exception to the default contract -- see
    ``SHOOTING_BRAKE_ALL_OUT`` -- and it exists to fit models whose expert
    bank exceeds combined 5090 + B70 VRAM.

    Within each active layer the expert id space is cut into three ranges::

        [0, cuda_n)                  -> CUDA   (hot)
        [cuda_n, num_experts-cpu_n)  -> B70    (warm)
        [num_experts-cpu_n, ...)     -> CPU    (cold)

    CPU takes the *high* ids so the B70 range stays contiguous from
    ``cuda_n``, which keeps B70 compact slots identical to
    :class:`LayerSubsetPolicy` at the same ``cuda_per_layer``. That makes
    the two directly comparable: switching a run from ``subset`` to
    ``allout`` moves only the tail experts and changes nothing else.

    Sizing ``cpu_n`` is a latency decision, not a capacity one. Each
    CPU-resident expert that fires costs ~195us against the B70's ~40us, so
    this range should hold experts that almost never route. Until
    frequency-calibrated placement lands, high expert ids are a proxy for
    "cold" no better than chance -- treat any ``cpu_n`` above a handful as a
    capacity experiment rather than a serving configuration.
    """

    active_layers: int
    cuda_per_layer: int
    cpu_per_layer: int
    name: str = field(init=False, default="allout")

    def __post_init__(self) -> None:
        if self.active_layers < 1:
            raise ValueError("AllOutPolicy.active_layers must be >= 1")
        if self.cpu_per_layer < 0:
            raise ValueError("AllOutPolicy.cpu_per_layer must be >= 0")
        object.__setattr__(
            self, "name",
            f"allout:active_layers={self.active_layers}"
            f",cuda_per_layer={self.cuda_per_layer}"
            f",cpu_per_layer={self.cpu_per_layer}",
        )

    def assign(
        self,
        num_layers: int,
        num_experts: int,
        b70_capable: frozenset[int],
    ) -> tuple[tuple[ExpertOwner, ...], ...]:
        active = set(sorted(b70_capable)[-self.active_layers:])
        all_cuda = tuple(
            ExpertOwner(Device.CUDA, e) for e in range(num_experts)
        )
        cuda_n = max(0, min(self.cuda_per_layer, num_experts))
        cpu_n = max(0, min(self.cpu_per_layer, num_experts - cuda_n))
        b70_end = num_experts - cpu_n

        offloaded = tuple(
            ExpertOwner(Device.CUDA, e) for e in range(cuda_n)
        ) + tuple(
            ExpertOwner(Device.B70, e - cuda_n) for e in range(cuda_n, b70_end)
        ) + tuple(
            ExpertOwner(Device.CPU, e - b70_end)
            for e in range(b70_end, num_experts)
        )
        return tuple(
            offloaded if layer in active else all_cuda
            for layer in range(num_layers)
        )


# --------------------------------------------------------------------------
# Construction + validation
# --------------------------------------------------------------------------


def build_placement(
    policy: PlacementPolicy,
    *,
    num_layers: int,
    num_experts: int,
    b70_capable: frozenset[int],
    generation: int = 1,
    device_capacities: tuple[DeviceCapacity, ...] = (),
) -> Placement:
    owners = policy.assign(num_layers, num_experts, b70_capable)
    placement = Placement(
        generation=generation,
        num_layers=num_layers,
        num_experts=num_experts,
        owners=owners,
        b70_capable_layers=b70_capable,
        policy_name=getattr(policy, "name", policy.__class__.__name__),
        device_capacities=device_capacities,
    )
    validate_placement(placement)
    return placement


class PlacementError(RuntimeError):
    """A Phase-5 ownership invariant was violated."""


def validate_placement(placement: Placement) -> None:
    """Assert complete ownership, valid device slots, and capacity budgets.

    Slots are dense independently for each concrete ``(device, index)``.
    This distinction is required when two B70s each own slot zero.
    """
    p = placement
    if p.num_layers < 1 or p.num_experts < 1:
        raise PlacementError("placement dimensions must be positive")
    invalid_capable = sorted(
        layer for layer in p.b70_capable_layers
        if not 0 <= layer < p.num_layers
    )
    if invalid_capable:
        raise PlacementError(
            f"B70-capable layers outside model bounds: {invalid_capable}"
        )
    # (1) shape + coverage
    if len(p.owners) != p.num_layers:
        raise PlacementError(
            f"owner table has {len(p.owners)} layers, expected {p.num_layers}"
        )
    for layer_idx, layer in enumerate(p.owners):
        if len(layer) != p.num_experts:
            raise PlacementError(
                f"layer {layer_idx} has {len(layer)} owners, "
                f"expected {p.num_experts}"
            )
        for owner in layer:
            if not isinstance(owner, ExpertOwner):
                raise PlacementError(f"layer {layer_idx} has a non-owner entry")

    # (2) compact slots per concrete device: dense [0, count), no dups/gaps
    for layer_idx, layer in enumerate(p.owners):
        seen: dict[DeviceTarget, set[int]] = {}
        for owner in layer:
            target = owner.target
            slots = seen.setdefault(target, set())
            if owner.slot < 0:
                raise PlacementError(
                    f"layer {layer_idx}: negative {target.label} slot {owner.slot}"
                )
            if owner.slot in slots:
                raise PlacementError(
                    f"layer {layer_idx}: duplicate {target.label} "
                    f"slot {owner.slot}"
                )
            slots.add(owner.slot)
        for target, slots in seen.items():
            if slots != set(range(len(slots))):
                raise PlacementError(
                    f"layer {layer_idx}: {target.label} slots are not a dense "
                    f"[0,{len(slots)}) range (got {sorted(slots)})"
                )

    # (3) + (4) CUDA-forced / offload-only-in-capable constraint.
    # CPU is held to the same layer restriction as B70 because both source
    # routed weights from an offline expert bank. Allowing either outside the
    # bank-covered set would create an owner with no weights.
    for layer_idx, layer in enumerate(p.owners):
        capable = layer_idx in p.b70_capable_layers
        for exp_idx, owner in enumerate(layer):
            if owner.device is Device.B70 and not capable:
                raise PlacementError(
                    f"layer {layer_idx} expert {exp_idx}: B70 ownership in a "
                    "non-B70-capable (FP8 / CUDA-forced) layer"
                )
            if owner.device is Device.CPU and not capable:
                raise PlacementError(
                    f"layer {layer_idx} expert {exp_idx}: CPU ownership in a "
                    "non-B70-capable (FP8 / CUDA-forced) layer"
                )

    # (5) Optional resident-weight budgets.  Counts span layers because each
    # layer has distinct expert weights resident on the target device.
    limits: dict[DeviceTarget, DeviceCapacity] = {}
    for capacity in p.device_capacities:
        if capacity.target in limits:
            raise PlacementError(
                f"duplicate capacity for {capacity.target.label}"
            )
        limits[capacity.target] = capacity
    for target, capacity in limits.items():
        resident_experts = p.count_target(target)
        required = resident_experts * capacity.bytes_per_expert
        if required > capacity.capacity_bytes:
            raise PlacementError(
                f"{target.label} expert weights require {required} bytes "
                f"({resident_experts} experts x {capacity.bytes_per_expert}) "
                f"but capacity is {capacity.capacity_bytes} bytes"
            )


def b70_bank_covers(
    placement: Placement,
    *,
    bank_layers: int,
    bank_experts_per_layer: int,
    bank_source_expert_ids: tuple[int, ...] = (),
) -> bool:
    """Whether the bank contains every source expert assigned off CUDA.

    SBINT401 banks carry an explicit sparse source-id list; legacy NVFP4
    banks are contiguous prefixes and therefore retain the count fallback.
    Device-local compact slots are never treated as source expert ids.
    """
    bank_layer_set = set(range(bank_layers))
    resident_ids = (
        set(bank_source_expert_ids)
        if bank_source_expert_ids
        else set(range(bank_experts_per_layer))
    )
    for layer_idx, layer in enumerate(placement.owners):
        for expert_idx, owner in enumerate(layer):
            if owner.device in (Device.B70, Device.CPU):
                if layer_idx not in bank_layer_set:
                    return False
                if expert_idx not in resident_ids:
                    return False
    return placement.b70_capable_layers <= bank_layer_set



# --------------------------------------------------------------------------
# Policy selection + adapter entry points
# --------------------------------------------------------------------------


def all_out_enabled() -> bool:
    """Whether "all-out mode" is armed.

    Default Shooting Brake never puts expert matrix compute on the CPU --
    that is a normative invariant across ``docs/``. All-out mode is the
    single, explicit exception, and it must be switched on deliberately:
    a placement string alone is not enough, because a stray env var or a
    copied command line should not be able to move expert compute onto the
    CPU behind the operator's back.
    """
    return os.environ.get("SHOOTING_BRAKE_ALL_OUT") == "1"


def policy_from_name(name: str) -> PlacementPolicy:
    """Parse a placement-policy spec of the form ``<family>[:<arg>...]``.

    Supported:
      * ``all-cuda``            -> :class:`AllCudaPolicy`
      * ``split:<N>``           -> :class:`SplitPolicy`(cuda_per_layer=N)
      * ``interleaved:<N>``     -> :class:`InterleavedPolicy`(period=N)
      * ``subset:<K>:<N>``      -> :class:`LayerSubsetPolicy`(K layers,
                                   cuda_per_layer=N)
      * ``allout:<K>:<N>:<C>``  -> :class:`AllOutPolicy`(K layers,
                                   cuda_per_layer=N, cpu_per_layer=C);
                                   requires ``SHOOTING_BRAKE_ALL_OUT=1``
      * ``fractional:<N>:<F>``  -> balance non-CUDA experts over B70 devices
                                   ``0..N-1``, keeping fraction ``F`` on CUDA

    This is the swappable seam a future predictive / speculative offloader
    plugs into: implement :class:`PlacementPolicy` and register a name here.
    """
    if name == "all-cuda":
        return AllCudaPolicy()
    if name.startswith("split:"):
        return SplitPolicy(cuda_per_layer=int(name.split(":", 1)[1]))
    if name.startswith("interleaved:"):
        return InterleavedPolicy(period=int(name.split(":", 1)[1]))
    if name.startswith("fractional:"):
        parts = name.split(":")
        if len(parts) != 3:
            raise PlacementError(
                "fractional policy needs "
                f"'fractional:<remote_devices>:<cuda_fraction>', got {name!r}"
            )
        device_count = int(parts[1])
        if device_count < 1:
            raise PlacementError("fractional policy needs at least one device")
        return FractionalRemotePolicy(
            remote_device_indices=tuple(range(device_count)),
            cuda_fraction=float(parts[2]),
        )
    if name.startswith("subset:"):
        parts = name.split(":")
        if len(parts) != 3:
            raise PlacementError(
                f"subset policy needs 'subset:<layers>:<cuda_per_layer>', "
                f"got {name!r}"
            )
        return LayerSubsetPolicy(
            active_layers=int(parts[1]), cuda_per_layer=int(parts[2])
        )
    if name.startswith("allout:"):
        parts = name.split(":")
        if len(parts) != 4:
            raise PlacementError(
                "allout policy needs "
                f"'allout:<layers>:<cuda_per_layer>:<cpu_per_layer>', "
                f"got {name!r}"
            )
        policy = AllOutPolicy(
            active_layers=int(parts[1]),
            cuda_per_layer=int(parts[2]),
            cpu_per_layer=int(parts[3]),
        )
        # Fail loudly rather than silently downgrading to a B70-only split:
        # an operator who asked for the CPU tier and did not get it would
        # otherwise read the resulting numbers as the tier's performance.
        if policy.cpu_per_layer > 0 and not all_out_enabled():
            raise PlacementError(
                f"placement {name!r} assigns {policy.cpu_per_layer} experts "
                "per layer to the CPU tier, but all-out mode is off. "
                "Set SHOOTING_BRAKE_ALL_OUT=1 to enable CPU expert compute "
                "on the normal path, or use a 'subset:' policy instead."
            )
        return policy
    raise PlacementError(f"unknown placement policy: {name!r}")


def build_for_qualified(
    qualified_model: "object",
    policy_name: str = "all-cuda",
    generation: int = 1,
) -> Placement:
    """Build + validate a placement from a :class:`QualifiedModel`.

    ``qualified_model`` must expose ``num_layers``, ``num_experts``,
    ``b70_capable_layers``, ``bank_layers`` and ``bank_experts_per_layer``
    (all provided by :class:`shooting_brake_vllm.config.QualifiedModel`).
    """
    policy = policy_from_name(policy_name)
    placement = build_placement(
        policy,
        num_layers=qualified_model.num_layers,
        num_experts=qualified_model.num_experts,
        b70_capable=qualified_model.b70_capable_layers,
        generation=generation,
    )
    if not b70_bank_covers(
        placement,
        bank_layers=qualified_model.bank_layers,
        bank_experts_per_layer=qualified_model.bank_experts_per_layer,
        bank_source_expert_ids=getattr(
            qualified_model, "bank_source_expert_ids", (),
        ),
    ):
        raise PlacementError(
            "placement assigns B70/CPU experts the selected bank does not hold"
        )
    return placement