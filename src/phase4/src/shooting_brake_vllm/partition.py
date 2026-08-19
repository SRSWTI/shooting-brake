"""Phase-6a runtime route partitioning.

Splits a router step's ``(topk_ids, topk_weights)`` into CUDA-local,
B70-remote, and CPU-remote subsets using the Phase-5 placement manifest.

The CPU subset is empty under every default placement; it only appears in
all-out mode. See :class:`shooting_brake_vllm.placement.AllOutPolicy`.

The partition classifies routes and never rewrites them, so the route sets
partition the original top-k without touching weights:

    sum_all w_k E_k(x) == sum_cuda w_k E_k(x)
                        + sum_b70  w_k E_k(x)
                        + sum_cpu  w_k E_k(x)

which is the same identity vLLM's expert-parallel all-reduce relies on, and
the reason :func:`validate_partition` insists every valid route has exactly
one owner: a double-counted route inflates its contribution, an uncovered
one silently drops it, and neither shows up as an error downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .placement import Device, Placement


class PartitionError(RuntimeError):
    """A Phase-6a route-partition invariant was violated at runtime."""


@dataclass(frozen=True)
class DispatchBufferGeometry:
    """Model-derived shapes shared by CUDA staging and the native poller."""

    max_batch: int
    hidden_size: int
    top_k: int

    def __post_init__(self) -> None:
        if self.max_batch < 1:
            raise ValueError("max_batch must be positive")
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")

    @property
    def hidden_shape(self) -> tuple[int, int]:
        return (self.max_batch, self.hidden_size)

    @property
    def route_shape(self) -> tuple[int, int]:
        return (self.max_batch, self.top_k)


def validate_dispatch_buffer_shapes(
    geometry: DispatchBufferGeometry,
    *,
    pinned_hidden: torch.Tensor,
    pinned_ids: torch.Tensor,
    pinned_weights: torch.Tensor,
    pinned_output: torch.Tensor,
) -> None:
    """Fail before poller registration when either side uses stale geometry."""
    expected = {
        "pinned_hidden": geometry.hidden_shape,
        "pinned_ids": geometry.route_shape,
        "pinned_weights": geometry.route_shape,
        "pinned_output": geometry.hidden_shape,
    }
    observed = {
        "pinned_hidden": tuple(pinned_hidden.shape),
        "pinned_ids": tuple(pinned_ids.shape),
        "pinned_weights": tuple(pinned_weights.shape),
        "pinned_output": tuple(pinned_output.shape),
    }
    wrong = {
        name: (observed[name], shape)
        for name, shape in expected.items()
        if observed[name] != shape
    }
    if wrong:
        detail = ", ".join(
            f"{name}={got}, expected {want}"
            for name, (got, want) in wrong.items()
        )
        raise PartitionError(f"dispatch staging geometry mismatch: {detail}")


def validate_cuda_dummy_slot_placement(placement: Placement) -> None:
    """Require one real CUDA expert wherever remote routes use dummy slot 0.

    The current fused CUDA call always runs, even when every route in a batch
    is remote. Non-CUDA routes therefore use zero-weight local slot 0. A layer
    with no CUDA experts has no row zero and needs a separate no-CUDA fast path,
    which does not exist yet.
    """
    for layer in range(placement.num_layers):
        cuda_count = len(placement.cuda_expert_ids(layer))
        offloaded_count = (
            placement.layer_b70_count(layer) + placement.layer_cpu_count(layer)
        )
        if offloaded_count and cuda_count == 0:
            raise PartitionError(
                f"layer {layer} offloads {offloaded_count} experts but keeps "
                "zero CUDA experts; compact dispatch requires a real CUDA "
                "expert at dummy local slot 0 until a no-CUDA fast path exists"
            )


def build_cuda_expert_maps(
    placement: Placement,
    layer: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return checked global→local and local→global CUDA expert maps.

    CUDA surgery compacts experts in ascending global-id order.  The returned
    global map uses ``-1`` for every non-CUDA owner so an offloaded global id
    can never be mistaken for a valid compact slot.
    """
    if not 0 <= layer < placement.num_layers:
        raise PartitionError(
            f"layer {layer} outside placement's 0..{placement.num_layers - 1}"
        )
    local_to_global = torch.tensor(
        placement.cuda_expert_ids(layer), dtype=torch.long,
    )
    global_to_local = torch.full(
        (placement.num_experts,), -1, dtype=torch.long,
    )
    for local_id, global_id in enumerate(local_to_global.tolist()):
        owner = placement.owners[layer][global_id]
        if owner.slot != local_id:
            raise PartitionError(
                f"layer {layer} CUDA expert {global_id} has placement slot "
                f"{owner.slot}, but surgery compacts it to {local_id}"
            )
        global_to_local[global_id] = local_id

    if local_to_global.numel():
        round_trip = global_to_local[local_to_global]
        expected = torch.arange(local_to_global.numel(), dtype=torch.long)
        if not torch.equal(round_trip, expected):
            raise PartitionError(
                f"layer {layer} CUDA compaction map does not round-trip"
            )
    return global_to_local, local_to_global


def compact_cuda_routes(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    global_to_local: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Translate global router ids for a compact CUDA tensor.

    Non-CUDA routes become dummy local slot zero *and* receive zero weight.
    The sentinel remains ``-1`` in ``global_to_local`` so ownership is tested
    before the dummy is introduced; a B70 global id is never handed to the
    compact CUDA tensor.
    """
    if topk_ids.shape != topk_weights.shape:
        raise PartitionError(
            f"route id shape {tuple(topk_ids.shape)} != weight shape "
            f"{tuple(topk_weights.shape)}"
        )
    if global_to_local.ndim != 1:
        raise PartitionError("CUDA global-to-local map must be one-dimensional")
    safe_global_ids = topk_ids.long().clamp(min=0)
    local_ids = global_to_local.to(topk_ids.device)[safe_global_ids]
    cuda_mask = (local_ids >= 0) & (topk_ids >= 0)
    cuda_weights = topk_weights * cuda_mask.to(topk_weights.dtype)
    # Preserve the router's id dtype: the map is int64 for indexing, but
    # downstream kernels (vLLM CUTLASS fp4 experts) require the ids in the
    # exact dtype the router produced — silently widening int32 -> int64
    # here cost a boot ("expected scalar type Int but found Long").
    return (
        local_ids.clamp(min=0).to(topk_ids.dtype),
        cuda_weights,
        cuda_mask,
    )


_DEVICE_CODE = {Device.CUDA: 0, Device.B70: 1, Device.CPU: 2}


def build_device_map(placement: Placement) -> torch.Tensor:
    """``[num_layers, num_experts]`` int8 CPU tensor: 0=CUDA, 1=B70, 2=CPU.

    Built on CPU; the caller moves the relevant layer row to the activation
    device once and caches it. Code 2 only ever appears under an all-out
    placement; every default policy produces a map of 0s and 1s.
    """
    rows = [
        [_DEVICE_CODE[owner.device] for owner in layer]
        for layer in placement.owners
    ]
    return torch.tensor(rows, dtype=torch.int8)


@dataclass
class RoutePartition:
    """The CUDA/B70/CPU split of one router step for one layer.

    All tensors are ``[M, topk]`` on the activation device. ``topk_ids`` and
    ``topk_weights`` are the **original, unmodified** router outputs — the
    partition classifies routes, it never rewrites them.
    """

    layer: int
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    cuda_mask: torch.Tensor  # bool, True where the route is CUDA-owned
    b70_mask: torch.Tensor  # bool, True where the route is B70-owned
    cpu_mask: torch.Tensor  # bool, True where the route is CPU-owned
    valid_mask: torch.Tensor  # bool, True where topk_ids >= 0 (non-padding)

    def has_remote(self) -> bool:
        return bool(self.b70_mask.any())

    def has_cpu(self) -> bool:
        return bool(self.cpu_mask.any())

    def num_b70_routes(self) -> int:
        return int(self.b70_mask.sum())

    def num_cpu_routes(self) -> int:
        return int(self.cpu_mask.sum())

    def num_cuda_routes(self) -> int:
        return int(self.cuda_mask.sum())

    def num_valid_routes(self) -> int:
        return int(self.valid_mask.sum())


def partition_routes(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    device_map_layer: torch.Tensor,
    layer: int,
) -> RoutePartition:
    """Classify each route by owner via a single device-map gather.

    ``device_map_layer[expert_id]`` is 0 for CUDA, 1 for B70, 2 for CPU.
    Padding entries (``topk_ids < 0``) are clamped before indexing and
    excluded from every mask.
    """
    valid = topk_ids >= 0
    safe_ids = topk_ids.clamp(min=0)
    route_devices = device_map_layer[safe_ids]  # [M, topk] gather
    return RoutePartition(
        layer=layer,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        cuda_mask=(route_devices == 0) & valid,
        b70_mask=(route_devices == 1) & valid,
        cpu_mask=(route_devices == 2) & valid,
        valid_mask=valid,
    )


def validate_partition(
    part: RoutePartition,
    b70_capable_layers: frozenset[int],
) -> None:
    """Assert the Phase-6a invariants:

    1. **covering + disjoint**: every valid route has exactly one owner;
    2. **offload-only-in-capable**: B70 and CPU routes appear only in
       B70-capable layers;
    3. **weight-preserving**: the partition did not modify ``topk_weights``.
    """
    # (1) exactly one owner per valid route (0 = uncovered, >1 = double-owned).
    # This is the invariant that makes the three partials summable: the
    # identity sum_all == sum_cuda + sum_b70 + sum_cpu holds only if each
    # route lands in exactly one term.
    owner_count = (
        part.cuda_mask.long() + part.b70_mask.long() + part.cpu_mask.long()
    )
    if part.valid_mask.any() and (owner_count[part.valid_mask] != 1).any():
        raise PartitionError(
            f"layer {part.layer}: route with invalid ownership "
            "(uncovered or double-owned)"
        )

    # (2) offloaded routes only in capable layers (FP8 layers 32-39 all-CUDA)
    if part.has_remote() and part.layer not in b70_capable_layers:
        raise PartitionError(
            f"layer {part.layer}: B70 routes in a non-B70-capable layer"
        )
    if part.has_cpu() and part.layer not in b70_capable_layers:
        raise PartitionError(
            f"layer {part.layer}: CPU routes in a non-B70-capable layer"
        )

    # (3) weight-preserving: masks do not alter weights (structural — the
    # parent receives the original tensor). We assert the weights tensor is
    # finite as a basic sanity check.
    if not torch.isfinite(part.topk_weights[part.valid_mask]).all():
        raise PartitionError(
            f"layer {part.layer}: non-finite routing weights"
        )
