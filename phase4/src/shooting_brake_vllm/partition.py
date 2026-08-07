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
