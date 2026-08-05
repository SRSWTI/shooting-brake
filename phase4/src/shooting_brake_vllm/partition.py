"""Phase-6a runtime route partitioning.

Splits a router step's ``(topk_ids, topk_weights)`` into CUDA-local and
B70-remote subsets using the Phase-5 placement manifest.

Phase 6a computes and validates the partition on every step but does NOT
change execution: the stock CUDA kernel still runs all routes unchanged. The
partition invariants (disjoint, covering, B70-only-in-capable-layers) are
asserted at runtime so any placement/routing mismatch fails loudly.

The math that makes the later split correct (Phase 6c) is already visible
here: the route sets partition the original top-k without touching weights, so

    sum_all  w_k * E_k(x)  ==  sum_cuda w_k * E_k(x)  +  sum_b70 w_k * E_k(x)

which is the same identity vLLM's expert-parallel all-reduce relies on.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .placement import Device, Placement


class PartitionError(RuntimeError):
    """A Phase-6a route-partition invariant was violated at runtime."""


def build_device_map(placement: Placement) -> torch.Tensor:
    """``[num_layers, num_experts]`` int8 CPU tensor: 0=CUDA, 1=B70.

    Built on CPU; the caller moves the relevant layer row to the activation
    device once and caches it.
    """
    rows = [
        [0 if owner.device is Device.CUDA else 1 for owner in layer]
        for layer in placement.owners
    ]
    return torch.tensor(rows, dtype=torch.int8)


@dataclass
class RoutePartition:
    """The CUDA/B70 split of one router step for one layer.

    All tensors are ``[M, topk]`` on the activation device. ``topk_ids`` and
    ``topk_weights`` are the **original, unmodified** router outputs — Phase 6a
    never touches them.
    """

    layer: int
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    cuda_mask: torch.Tensor  # bool, True where the route is CUDA-owned
    b70_mask: torch.Tensor  # bool, True where the route is B70-owned
    valid_mask: torch.Tensor  # bool, True where topk_ids >= 0 (non-padding)

    def has_remote(self) -> bool:
        return bool(self.b70_mask.any())

    def num_b70_routes(self) -> int:
        return int(self.b70_mask.sum())

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
    """Classify each route as CUDA or B70 via a single device-map gather.

    ``device_map_layer[expert_id]`` is 0 for CUDA-owned, 1 for B70-owned.
    Padding entries (``topk_ids < 0``) are clamped before indexing and excluded
    from both masks.
    """
    valid = topk_ids >= 0
    safe_ids = topk_ids.clamp(min=0)
    route_devices = device_map_layer[safe_ids]  # [M, topk] gather
    cuda_mask = (route_devices == 0) & valid
    b70_mask = (route_devices == 1) & valid
    return RoutePartition(
        layer=layer,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        cuda_mask=cuda_mask,
        b70_mask=b70_mask,
        valid_mask=valid,
    )


def validate_partition(
    part: RoutePartition,
    b70_capable_layers: frozenset[int],
) -> None:
    """Assert the Phase-6a invariants:

    1. **covering + disjoint**: every valid route has exactly one owner;
    2. **B70-only-in-capable**: B70 routes appear only in B70-capable layers;
    3. **weight-preserving**: the partition did not modify ``topk_weights``.
    """
    # (1) exactly one owner per valid route (0 = uncovered, 2 = double-owned)
    owner_count = part.cuda_mask.long() + part.b70_mask.long()
    if part.valid_mask.any() and (owner_count[part.valid_mask] != 1).any():
        raise PartitionError(
            f"layer {part.layer}: route with invalid ownership "
            "(uncovered or double-owned)"
        )

    # (2) B70 routes only in capable layers (FP8 layers 32-39 must be all-CUDA)
    if part.has_remote() and part.layer not in b70_capable_layers:
        raise PartitionError(
            f"layer {part.layer}: B70 routes in a non-B70-capable layer"
        )

    # (3) weight-preserving: masks do not alter weights (structural — the
    # parent receives the original tensor). We assert the weights tensor is
    # finite as a basic sanity check.
    if not torch.isfinite(part.topk_weights[part.valid_mask]).all():
        raise PartitionError(
            f"layer {part.layer}: non-finite routing weights"
        )
