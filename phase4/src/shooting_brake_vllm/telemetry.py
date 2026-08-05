"""Worker-side telemetry collection for Shooting Brake.

The adapter runs inside vLLM's EngineCore worker process, so its
counters are not reachable from the driver that submits requests.
:func:`collect_worker_stats` is written to be invoked through
``LLM.collective_rpc`` from the driver, which executes it in every
worker and returns the results.

Nothing here runs on the decode path: the device-side route counters are
accumulated inside the captured graph and only *read* from here, between
engine steps.
"""

from __future__ import annotations

from typing import Any

import torch


def _iter_hybrid_layers(model: Any) -> list[Any]:
    """Every HybridRoutedExperts instance reachable from the model."""
    from .routed_experts import HybridRoutedExperts

    return [
        module for module in model.modules()
        if isinstance(module, HybridRoutedExperts)
    ]


def collect_worker_stats(worker: Any) -> dict[str, Any]:
    """Snapshot adapter counters for one worker.

    Args:
        worker: vLLM worker instance, supplied by ``collective_rpc``.

    Returns:
        Route shares, B70 poller counters, and CUDA memory. Keys are
        absent rather than zero when a subsystem is inactive, so a
        caller can tell "not enabled" from "enabled but idle".
    """
    stats: dict[str, Any] = {}

    model = worker.model_runner.model
    layers = _iter_hybrid_layers(model)
    stats["hybrid_layers"] = len(layers)

    # --- route shares ---------------------------------------------------
    # Counters live on device and are summed here, one host sync per
    # call, never on the decode path.
    b70_routes = 0
    total_routes = 0
    per_layer: dict[int, int] = {}
    for layer in layers:
        counter = getattr(layer, "_route_counter", None)
        if counter is None:
            continue
        b70, total = (int(v) for v in counter.tolist())
        b70_routes += b70
        total_routes += total
        idx = getattr(layer, "_layer_idx", None)
        if idx is not None:
            per_layer[idx] = b70
    if total_routes:
        stats["routes"] = {
            "b70": b70_routes,
            "total": total_routes,
            "b70_share": b70_routes / total_routes,
            "per_layer_b70": per_layer,
        }

    # --- B70 poller -----------------------------------------------------
    poller = next(
        (
            layer._b70_poller for layer in layers
            if getattr(layer, "_b70_poller", None) is not None
        ),
        None,
    )
    if poller is not None:
        dispatches = poller.dispatch_count
        stats["poller"] = {
            "dispatches": dispatches,
            "errors": poller.error_count,
            "service_mean_us": poller.service_mean_us,
        }

    # --- memory ---------------------------------------------------------
    free_b, total_b = torch.cuda.mem_get_info()
    stats["cuda_memory"] = {
        "allocated_gib": torch.cuda.memory_allocated() / 2**30,
        "reserved_gib": torch.cuda.memory_reserved() / 2**30,
        "free_gib": free_b / 2**30,
        "total_gib": total_b / 2**30,
    }
    return stats


def reset_worker_stats(worker: Any) -> None:
    """Zero the device route counters, for per-phase measurement."""
    for layer in _iter_hybrid_layers(worker.model_runner.model):
        counter = getattr(layer, "_route_counter", None)
        if counter is not None:
            counter.zero_()


__all__ = ["collect_worker_stats", "reset_worker_stats"]
