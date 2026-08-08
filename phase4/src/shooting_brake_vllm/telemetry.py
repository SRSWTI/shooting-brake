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
    cpu_routes = 0
    total_routes = 0
    per_layer: dict[int, int] = {}
    per_layer_cpu: dict[int, int] = {}
    for layer in layers:
        counter = getattr(layer, "_route_counter", None)
        if counter is None:
            continue
        # [b70, total, cpu]; the third slot predates all-out mode being
        # switched on, so tolerate a two-wide counter from an older layer.
        values = [int(v) for v in counter.tolist()]
        b70, total = values[0], values[1]
        cpu = values[2] if len(values) > 2 else 0
        b70_routes += b70
        cpu_routes += cpu
        total_routes += total
        idx = getattr(layer, "_layer_idx", None)
        if idx is not None:
            per_layer[idx] = b70
            if cpu:
                per_layer_cpu[idx] = cpu
    if total_routes:
        stats["routes"] = {
            "b70": b70_routes,
            "cpu": cpu_routes,
            "total": total_routes,
            "b70_share": b70_routes / total_routes,
            "cpu_share": cpu_routes / total_routes,
            "per_layer_b70": per_layer,
            "per_layer_cpu": per_layer_cpu,
        }

    # --- pollers ---------------------------------------------------------
    poller = next(
        (
            layer._b70_poller for layer in layers
            if getattr(layer, "_b70_poller", None) is not None
        ),
        None,
    )
    if poller is not None:
        # kernel_mean_us is 0 unless SHOOTING_BRAKE_B70_PROFILE=1. When set,
        # service - kernel is the submission and synchronisation overhead:
        # the removable half of the round trip, as opposed to the kernel's
        # own B70 VRAM-bandwidth floor.
        stats["poller"] = {
            "dispatches": poller.dispatch_count,
            "errors": poller.error_count,
            "service_mean_us": poller.service_mean_us,
            "kernel_mean_us": poller.kernel_mean_us,
        }

    cpu_poller = next(
        (
            layer._cpu_poller for layer in layers
            if getattr(layer, "_cpu_poller", None) is not None
        ),
        None,
    )
    if cpu_poller is not None:
        # Service time here is the number to watch: it is ~5x the B70's for
        # the same work, so a rising mean means placement has put a
        # frequently-routed expert on the cold tier.
        stats["cpu_poller"] = {
            "dispatches": cpu_poller.dispatch_count,
            "errors": cpu_poller.error_count,
            "service_mean_us": cpu_poller.service_mean_us,
        }
        host = next(
            (
                layer._cpu_host for layer in layers
                if getattr(layer, "_cpu_host", None) is not None
            ),
            None,
        )
        if host is not None:
            stats["cpu_arena"] = {
                "resident_experts": host.resident_count,
                "used_gib": host.arena_used_bytes / 2**30,
                # Must stay 0: a skipped route is a dropped contribution,
                # which is a wrong answer rather than a slow one.
                "skipped_routes": host.skipped_routes,
            }

        # Prefill weight streaming is otherwise invisible: it bypasses the
        # poller entirely, so cpu_poller counts only the decode dispatches.
        # Without this, a run that streamed gigabytes to the 5090 looks
        # identical to one that streamed nothing.
        try:
            from .cpu_stream import _streamer_singleton
        except ImportError:
            _streamer_singleton = None
        if _streamer_singleton is not None:
            stats["cpu_stream"] = dict(_streamer_singleton.stats)

    # --- capacity -------------------------------------------------------
    # KV cache size is the metric that matters: it is what the freed
    # expert VRAM buys, and it caps how many requests can be in flight.
    # "Free VRAM" says nothing, because vLLM allocates up to
    # gpu_memory_utilization either way.
    cache_config = worker.vllm_config.cache_config
    num_blocks = cache_config.num_gpu_blocks or 0
    stats["kv_cache"] = {
        "num_gpu_blocks": num_blocks,
        "block_size": cache_config.block_size,
        "max_tokens": num_blocks * cache_config.block_size,
    }

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
