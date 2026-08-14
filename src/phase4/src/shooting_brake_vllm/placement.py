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
    all_devices = (Device.CUDA, Device.B70, Device.CPU)
    for layer_idx, layer in enumerate(p.owners):
        seen: dict[Device, set[int]] = {d: set() for d in all_devices}
        counts: dict[Device, int] = {d: 0 for d in all_devices}
        for owner in layer:
            counts[owner.device] += 1
            if owner.slot in seen[owner.device]:
                raise PlacementError(
                    f"layer {layer_idx}: duplicate {owner.device.value} "
                    f"slot {owner.slot}"
                )
            seen[owner.device].add(owner.slot)
        for device in all_devices:
            slots = seen[device]
            n = counts[device]
            if slots != set(range(n)):
                raise PlacementError(
                    f"layer {layer_idx}: {device.value} slots are not a dense "
                    f"[0,{n}) range (got {sorted(slots)})"
                )

    # (3) + (4) CUDA-forced / offload-only-in-capable constraint.
    # The CPU tier is held to the same layer restriction as the B70: it
    # sources weights from the same NVFP4 layers, and allowing it to spread
    # further would let a placement quietly put expert compute on the CPU in
    # layers no offload policy was ever validated against.
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

    This is the swappable seam a future predictive / speculative offloader
    plugs into: implement :class:`PlacementPolicy` and register a name here.
    """
    if name == "all-cuda":
        return AllCudaPolicy()
    if name.startswith("split:"):
        return SplitPolicy(cuda_per_layer=int(name.split(":", 1)[1]))
    if name.startswith("interleaved:"):
        return InterleavedPolicy(period=int(name.split(":", 1)[1]))
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
    ):
        raise PlacementError(
            "placement assigns B70 experts the Phase-1 bank does not hold"
        )
    return placement