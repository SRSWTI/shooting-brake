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

    # Arena and streamer are reported independently of the CPU poller. Both
    # were nested under it, which was right when only the cold tier used
    # them: no poller meant no arena. B70 prefill streaming broke that
    # assumption -- it populates the same arena and drives the same streamer
    # on a hybrid placement that has no CPU poller at all, so the nesting
    # silently reported nothing for exactly the configuration the feature
    # runs in. A run that streamed gigabytes to the 5090 looked identical to
    # one that streamed none.
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

    try:
        from .cpu_stream import _streamer_singleton
    except ImportError:
        _streamer_singleton = None
    if _streamer_singleton is not None:
        stats["cpu_stream"] = dict(_streamer_singleton.stats)

    # --- route histogram --------------------------------------------------
    # Present only under SHOOTING_BRAKE_ROUTE_STATS=1. The full table goes
    # to CSV via dump_route_histogram; this block is the liveness check --
    # it confirms counting is happening without shipping 40x256 ints
    # through every stats call.
    from . import route_stats

    hist = route_stats.peek()
    if hist is not None:
        counts, tokens = hist.to_cpu()
        stats["route_histogram"] = {
            "layers_observed": int((tokens > 0).sum()),
            "tokens_per_layer_max": int(tokens.max()),
            "total_routes": int(counts.sum()),
        }

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

    # Function-local, matching `_iter_hybrid_layers`: a module-scope import
    # would pull routed_experts' vLLM dependencies into the driver process,
    # which imports this module only for `collect_worker_stats`.
    from .routed_experts import load_peak_gib

    free_b, total_b = torch.cuda.mem_get_info()
    stats["cuda_memory"] = {
        "allocated_gib": torch.cuda.memory_allocated() / 2**30,
        "reserved_gib": torch.cuda.memory_reserved() / 2**30,
        # Process high-water marks, read at end of run: they describe the
        # KV cache, which fills to `gpu_memory_utilization x total` however
        # the experts were allocated. NOT how the surgery strategies differ.
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        # This one is. Sampled during the post-load hook, before the KV
        # cache exists, so it still describes the weights: post-hoc surgery
        # carries the whole expert bank through that moment, pre-emptive
        # allocation never materialises the offloaded part. On a model
        # where both strategies run it is the only figure that moves — KV
        # sizing sees identical freed memory either way, because post-hoc
        # surgery completes before vLLM's profiling pass.
        "load_peak_allocated_gib": load_peak_gib(),
        "free_gib": free_b / 2**30,
        "total_gib": total_b / 2**30,
    }

    # B70 VRAM. The second card is bought for capacity, so its occupancy is
    # the number that decides whether an expert bank fits -- and it was the
    # only resource here inferred rather than measured, while the 5090's
    # VRAM, host DRAM and the KV cache were all reported. Read through the
    # provider's own device handle, so it is guaranteed to describe the card
    # the bank actually loaded onto rather than whichever Intel GPU a second
    # lookup happens to enumerate first.
    from .routed_experts import _b70_provider_singleton

    if _b70_provider_singleton is not None:
        mem = _b70_provider_singleton.device_memory
        if mem is not None:
            free_b70, total_b70 = mem
            stats["b70_memory"] = {
                "used_gib": (total_b70 - free_b70) / 2**30,
                "free_gib": free_b70 / 2**30,
                "total_gib": total_b70 / 2**30,
                "resident_per_layer": _b70_provider_singleton.resident_per_layer,
            }

    # Host DRAM. Untracked until now because the first two tiers do not use
    # it for weights, but the cold tier holds its whole expert bank here and
    # B70 prefill streaming adds a second copy of the B70 bank. On the 35B
    # that is under a gigabyte; at 122B scale it is the constraint that
    # decides whether the model loads at all, so it belongs beside VRAM
    # rather than in a footnote. VmHWM is the peak, which is what a capacity
    # decision needs -- current RSS understates a run that has already
    # freed a transient staging copy.
    try:
        fields = {}
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(("VmRSS:", "VmHWM:")):
                    k, v = line.split(":", 1)
                    fields[k] = int(v.split()[0]) / 2**20  # kB -> GiB
        if fields:
            stats["host_memory"] = {
                "rss_gib": fields.get("VmRSS", 0.0),
                "peak_rss_gib": fields.get("VmHWM", 0.0),
            }
    except OSError:
        pass

    # Board power. The project's claim is capacity per dollar, and running
    # cost is part of that: a second card that idles at 40 W changes the
    # arithmetic differently than one that draws 200 W. Sampled, not
    # integrated -- a single reading between steps, not an energy total.
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        stats["cuda_power"] = {
            "watts": pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0,
            "limit_watts": pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0,
            "temperature_c": pynvml.nvmlDeviceGetTemperature(h, 0),
        }
    except Exception:
        # pynvml missing or the driver refuses the query: power is a nice
        # extra, never a reason to lose the rest of the snapshot.
        pass
    return stats


def reset_worker_stats(worker: Any) -> None:
    """Zero the device route counters, for per-phase measurement."""
    for layer in _iter_hybrid_layers(worker.model_runner.model):
        counter = getattr(layer, "_route_counter", None)
        if counter is not None:
            counter.zero_()
    from . import route_stats

    hist = route_stats.peek()
    if hist is not None:
        hist.counts.zero_()
        hist.tokens.zero_()


def dump_route_histogram(worker: Any) -> str | None:
    """Write the route histogram CSV from inside the worker.

    Invoked through ``collective_rpc`` at the end of a calibration run. The
    counter lives in the worker process, so the driver cannot reach it any
    other way; the return value is the path, for the driver to report.
    """
    from . import route_stats

    out = route_stats.dump()
    return str(out) if out is not None else None


__all__ = [
    "collect_worker_stats",
    "dump_route_histogram",
    "reset_worker_stats",
]
