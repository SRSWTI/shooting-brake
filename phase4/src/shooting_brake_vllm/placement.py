"""Phase-5 compact expert ownership: a versioned, swappable placement manifest.

The manifest records, for every ``(layer, global_expert)`` in the qualified
Qwen3.6 model, exactly one normal-path owner: the RTX 5090 (CUDA) or the Arc
Pro B70. It is the single source of truth that Phase 6 will consume to
partition a token's top-k routes across the two devices.

Design goals
------------
* **Immutable per inference step.** Moving a ~1.7 MiB expert mid-step would
  blow the decode latency budget, so ownership is frozen for the hot path.
* **Swappable at coarse boundaries.** The manifest is plain data with a
  ``generation`` id. A future predictive / speculative offloader can compute a
  new placement, bump the generation, and swap the manifest between requests or
  generations without touching the per-step contract. The hot path never
  inspects ``generation`` for routing decisions -- it only uses it to reject
  stale references (mirroring the Phase-2 protocol's placement-generation id).
* **Not a fake EP rank.** vLLM's ``ExpertMapManager`` distributes experts over
  tensor/expert-parallel ranks. The B70 is a separate process with its own
  QuixiCore kernels, not a vLLM rank, so we own the map ourselves.

Layer structure of the qualified model
---------------------------------------
All 40 layers are MoE (256 routed experts each). Layers 0-31 are NVFP4 and are
extracted into the Phase-1 B70 bank (8192 experts); they are ``b70_capable``.
Layers 32-39 are FP8 and are not in the bank; their experts are CUDA-forced.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol


class Device(str, Enum):
    CUDA = "cuda"
    B70 = "b70"


@dataclass(frozen=True)
class ExpertOwner:
    """The single normal-path owner of one (layer, global_expert)."""

    device: Device
    slot: int  # compact device-local slot, dense from 0

    def to_dict(self) -> dict:
        return {"device": self.device.value, "slot": self.slot}

    @classmethod
    def from_dict(cls, d: dict) -> ExpertOwner:
        return cls(device=Device(d["device"]), slot=int(d["slot"]))


@dataclass(frozen=True)
class Placement:
    """Immutable ownership table for every routed expert in the model.

    Attributes:
        generation: monotonic version; bumped whenever the manifest is swapped.
        num_layers: total number of MoE layers (40 for the qualified model).
        num_experts: routed experts per layer (256).
        owners: ``owners[layer][global_expert]`` -> :class:`ExpertOwner`.
        b70_capable_layers: absolute layer indices whose experts exist in the
            NVFP4 B70 bank and may therefore be B70-owned (0..31).
        policy_name: human-readable name of the policy that produced this.
    """

    generation: int
    num_layers: int
    num_experts: int
    owners: tuple[tuple[ExpertOwner, ...], ...]
    b70_capable_layers: frozenset[int]
    policy_name: str = ""

    # -- introspection ---------------------------------------------------

    def count(self, device: Device) -> int:
        return sum(
            1 for layer in self.owners for owner in layer if owner.device is device
        )

    def cuda_count(self) -> int:
        return self.count(Device.CUDA)

    def b70_count(self) -> int:
        return self.count(Device.B70)

    def is_b70_capable(self, layer: int) -> bool:
        return layer in self.b70_capable_layers

    # -- (de)serialization for data-driven swapping ----------------------

    SCHEMA = "shooting-brake.placement.v1"

    def to_manifest(self) -> dict:
        return {
            "schema": self.SCHEMA,
            "generation": self.generation,
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "b70_capable_layers": sorted(self.b70_capable_layers),
            "policy": self.policy_name,
            "owners": [
                [owner.to_dict() for owner in layer] for layer in self.owners
            ],
        }

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_manifest()), encoding="utf-8")

    @classmethod
    def from_manifest(cls, manifest: dict) -> Placement:
        assert manifest["schema"] == cls.SCHEMA, f"unsupported schema {manifest['schema']!r}"
        owners = tuple(
            tuple(ExpertOwner.from_dict(o) for o in layer)
            for layer in manifest["owners"]
        )
        return cls(
            generation=int(manifest["generation"]),
            num_layers=int(manifest["num_layers"]),
            num_experts=int(manifest["num_experts"]),
            owners=owners,
            b70_capable_layers=frozenset(manifest["b70_capable_layers"]),
            policy_name=str(manifest.get("policy", "")),
        )

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
) -> Placement:
    owners = policy.assign(num_layers, num_experts, b70_capable)
    placement = Placement(
        generation=generation,
        num_layers=num_layers,
        num_experts=num_experts,
        owners=owners,
        b70_capable_layers=b70_capable,
        policy_name=getattr(policy, "name", policy.__class__.__name__),
    )
    validate_placement(placement)
    return placement


class PlacementError(RuntimeError):
    """A Phase-5 ownership invariant was violated."""


def validate_placement(placement: Placement) -> None:
    """Assert the Phase-5 gate invariants:

    1. exactly one owner per expert, full coverage, no gaps;
    2. compact, gap-free, duplicate-free device-local slots per (layer, device);
    3. experts in non-B70-capable layers are CUDA-forced;
    4. B70-owned experts only live in B70-capable layers.
    """
    p = placement
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

    # (2) compact slots per (layer, device): dense [0, count) with no dups/gaps
    for layer_idx, layer in enumerate(p.owners):
        seen: dict[Device, set[int]] = {Device.CUDA: set(), Device.B70: set()}
        counts: dict[Device, int] = {Device.CUDA: 0, Device.B70: 0}
        for owner in layer:
            counts[owner.device] += 1
            if owner.slot in seen[owner.device]:
                raise PlacementError(
                    f"layer {layer_idx}: duplicate {owner.device.value} "
                    f"slot {owner.slot}"
                )
            seen[owner.device].add(owner.slot)
        for device in (Device.CUDA, Device.B70):
            slots = seen[device]
            n = counts[device]
            if slots != set(range(n)):
                raise PlacementError(
                    f"layer {layer_idx}: {device.value} slots are not a dense "
                    f"[0,{n}) range (got {sorted(slots)})"
                )

    # (3) + (4) CUDA-forced / B70-only-in-capable constraint
    for layer_idx, layer in enumerate(p.owners):
        capable = layer_idx in p.b70_capable_layers
        for exp_idx, owner in enumerate(layer):
            if owner.device is Device.B70 and not capable:
                raise PlacementError(
                    f"layer {layer_idx} expert {exp_idx}: B70 ownership in a "
                    "non-B70-capable (FP8 / CUDA-forced) layer"
                )


def b70_bank_covers(
    placement: Placement,
    *,
    bank_layers: int,
    bank_experts_per_layer: int,
) -> bool:
    """Whether the Phase-1 NVFP4 bank physically holds every B70-owned expert.

    The bank is laid out as ``bank_layers * bank_experts_per_layer`` contiguous
    NVFP4 records (layers 0..bank_layers-1). Any B70-owned expert must fall
    inside that range. This is an explicit cross-check that the placement never
    promises the B70 an expert it does not have.
    """
    bank_layer_set = set(range(bank_layers))
    for layer_idx, layer in enumerate(placement.owners):
        for owner in layer:
            if owner.device is Device.B70:
                if layer_idx not in bank_layer_set:
                    return False
                if owner.slot >= bank_experts_per_layer:
                    return False
    # b70_capable_layers must also be within the bank
    return placement.b70_capable_layers <= bank_layer_set



# --------------------------------------------------------------------------
# Policy selection + adapter entry points
# --------------------------------------------------------------------------


def policy_from_name(name: str) -> PlacementPolicy:
    """Parse a placement-policy spec of the form ``<family>[:<arg>=<value>]``.

    Supported:
      * ``all-cuda``                  -> :class:`AllCudaPolicy`
      * ``split:<N>``                 -> :class:`SplitPolicy`(cuda_per_layer=N)
      * ``interleaved:<N>``           -> :class:`InterleavedPolicy`(period=N)

    This is the swappable seam a future predictive / speculative offloader
    plugs into: implement :class:`PlacementPolicy` and register a name here.
    """
    if name == "all-cuda":
        return AllCudaPolicy()
    if name.startswith("split:"):
        return SplitPolicy(cuda_per_layer=int(name.split(":", 1)[1]))
    if name.startswith("interleaved:"):
        return InterleavedPolicy(period=int(name.split(":", 1)[1]))
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
    ):
        raise PlacementError(
            "placement assigns B70 experts the Phase-1 bank does not hold"
        )
    return placement