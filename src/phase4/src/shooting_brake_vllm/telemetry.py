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

_provider_health_baseline: dict[int, tuple[int, int]] = {}



def _provider_health_stats(
    health: Any,
    baseline: tuple[int, int] | None,
) -> dict[str, Any]:
    """Convert monotonic native health into an auditable scoped count."""
    generation = int(health.generation)
    raw_dispatches = int(health.dispatches)
    stats: dict[str, Any] = {
        "available": True,
        "generation_raw": generation,
        "generation_baseline": baseline[0] if baseline is not None else None,
        "dispatches_raw": raw_dispatches,
        "dispatches_baseline": baseline[1] if baseline is not None else None,
        "dispatches_delta": None,
        "last_error": health.last_error,
    }
    if baseline is None:
        return stats
    baseline_generation, baseline_dispatches = baseline
    if generation != baseline_generation:
        stats["available"] = False
        stats["reason"] = (
            "provider generation changed after health baseline "
            f"({baseline_generation} -> {generation})"
        )
        return stats
    if raw_dispatches < baseline_dispatches:
        stats["available"] = False
        stats["reason"] = (
            "provider dispatch counter reset after health baseline "
            f"({baseline_dispatches} -> {raw_dispatches})"
        )
        return stats
    stats["dispatches_delta"] = raw_dispatches - baseline_dispatches
    return stats


def _iter_hybrid_layers(model: Any) -> list[Any]:
    """Every HybridRoutedExperts instance reachable from the model."""
    from .routed_experts import HybridRoutedExperts

    return [
        module for module in model.modules()
        if isinstance(module, HybridRoutedExperts)
    ]


def _collect_eager_partition(layers: list[Any]) -> dict[str, int]:
    """Aggregate synchronous-dispatch liveness without a device sync."""
    steps = 0
    remote_steps = 0
    expected_remote_steps = 0
    remote_routes = 0
    for layer in layers:
        partition_stats = getattr(
            layer, "shooting_brake_partition_stats", None
        )
        if partition_stats is None:
            continue
        layer_steps = partition_stats.get("steps", 0)
        steps += layer_steps
        remote_steps += partition_stats.get("remote_steps", 0)
        remote_routes += partition_stats.get("remote_routes", 0)
        placement = getattr(layer, "shooting_brake_placement", None)
        layer_idx = getattr(layer, "_layer_idx", None)
        if (
            placement is not None
            and layer_idx is not None
            and placement.is_b70_active(layer_idx)
        ):
            expected_remote_steps += layer_steps
    return {
        "steps": steps,
        "remote_steps": remote_steps,
        "expected_remote_steps": expected_remote_steps,
        "expected_provider_dispatches": expected_remote_steps,
        "all_cuda_route_steps": expected_remote_steps - remote_steps,
        "remote_routes": remote_routes,
    }


def _collect_arm_contract(layers: list[Any]) -> dict[str, Any] | None:
    """Aggregate the opt-in benchmark-arm contract without device work."""
    configured = [
        layer for layer in layers
        if getattr(layer, "shooting_brake_arm_config", None) is not None
    ]
    if not configured:
        return None
    if len(configured) != len(layers):
        raise RuntimeError(
            "benchmark arm contract is missing from one or more hybrid layers: "
            f"configured={len(configured)} total={len(layers)}"
        )
    config = configured[0].shooting_brake_arm_config
    if any(layer.shooting_brake_arm_config != config for layer in configured):
        raise RuntimeError("benchmark arm configuration differs across layers")

    modes: dict[str, int] = {}
    branches: dict[str, int] = {}
    observed_layers = 0
    observations: list[dict[str, Any]] = []
    for layer in configured:
        layer_observations = sorted(layer._arm_runtime_observations)
        if layer_observations:
            observed_layers += 1
        for mode, branch, stream_capturing in layer_observations:
            modes[mode] = modes.get(mode, 0) + 1
            branches[branch] = branches.get(branch, 0) + 1
            observations.append({
                "layer": layer.layer_name,
                "runtime_mode": mode,
                "stream_capturing": stream_capturing,
                "branch": branch,
            })
    return {
        "config": dict(config),
        "total_layers": len(layers),
        "observed_layers": observed_layers,
        "runtime_mode_layer_counts": modes,
        "branch_layer_counts": branches,
        "observations": observations,
    }


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
    arm_contract = _collect_arm_contract(layers)
    if arm_contract is not None:
        stats["benchmark_arm"] = arm_contract

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

    # Eager/synchronous arm counters. These are ordinary Python integers,
    # so reading them here adds no device synchronization.
    stats["eager_partition"] = _collect_eager_partition(layers)

    # --- pollers ---------------------------------------------------------
    # One poller per physical card. Top-level keys aggregate across cards
    # (back-compat with every reader of stats["poller"]); "per_device"
    # carries each card separately — per-card service time is exactly the
    # number the dual-B70 bring-up must watch (a card whose kernel goes
    # latency-bound shows up here first).
    from .b70_poller import _pollers as _b70_poller_registry

    if _b70_poller_registry:
        per_device: dict[str, Any] = {}
        total_dispatches = 0
        total_errors = 0
        total_rows = 0
        service_us_sum = 0.0
        kernel_us_sum = 0.0
        histogram_sum: dict[str, int] = {}
        for device_index in sorted(_b70_poller_registry):
            p = _b70_poller_registry[device_index]
            n = p.dispatch_count
            # Profiling metrics are 0 unless SHOOTING_BRAKE_B70_PROFILE=1.
            per_device[str(device_index)] = {
                "dispatches": n,
                "errors": p.error_count,
                "rows": p.row_count,
                "m_histogram": p.m_histogram,
                "service_mean_us": p.service_mean_us,
                "kernel_mean_us": p.kernel_mean_us,
                "kernel_mean_us_per_row": p.kernel_mean_us_per_row,
            }
            total_dispatches += n
            total_errors += p.error_count
            total_rows += p.row_count
            service_us_sum += p.service_mean_us * n
            kernel_us_sum += p.kernel_mean_us * n
            for label, count in p.m_histogram.items():
                histogram_sum[label] = histogram_sum.get(label, 0) + count
        stats["poller"] = {
            "available": True,
            "dispatches": total_dispatches,
            "errors": total_errors,
            "rows": total_rows,
            "m_histogram": histogram_sum,
            "service_mean_us": (
                service_us_sum / total_dispatches if total_dispatches else 0.0
            ),
            "kernel_mean_us": (
                kernel_us_sum / total_dispatches if total_dispatches else 0.0
            ),
            "kernel_mean_us_per_row": (
                kernel_us_sum / total_rows if total_rows else 0.0
            ),
            "per_device": per_device,
        }
    else:
        stats["poller"] = {
            "available": False,
            "reason": "unavailable on this arm",
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
    # For hybrid attention/Mamba models a block is a combined page carrying
    # both attention KV and recurrent state. Multiplying block count by its
    # attention token width overstates capacity, so expose the authoritative
    # raw scheduler values without inventing a max-token figure.
    cache_config = worker.vllm_config.cache_config
    stats["kv_cache"] = {
        "num_gpu_blocks": cache_config.num_gpu_blocks or 0,
        "block_size": cache_config.block_size,
        "max_tokens": "unavailable for hybrid cache",
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

    # B70 VRAM, one entry per card. Occupancy is the number that decides
    # whether an expert bank fits -- and it was the only resource here
    # inferred rather than measured, while the 5090's VRAM, host DRAM and
    # the KV cache were all reported. Read through each provider's own
    # device handle, so it is guaranteed to describe the card that bank
    # actually loaded onto rather than whichever Intel GPU a second lookup
    # happens to enumerate first.
    from .routed_experts import _b70_providers

    b70_memory: dict[str, Any] = {}
    for device_index in sorted(_b70_providers):
        provider = _b70_providers[device_index]
        mem = provider.device_memory
        if mem is None:
            continue
        free_b70, total_b70 = mem
        b70_memory[str(device_index)] = {
            "used_gib": (total_b70 - free_b70) / 2**30,
            "free_gib": free_b70 / 2**30,
            "total_gib": total_b70 / 2**30,
            "resident_per_layer": provider.resident_per_layer,
        }
    if b70_memory:
        stats["b70_memory"] = b70_memory

    if not _b70_providers:
        stats["synchronous_provider"] = {
            "available": False,
            "reason": "unavailable on this arm",
        }
    else:
        provider_stats: dict[str, Any] = {}
        for device_index in sorted(_b70_providers):
            health = getattr(_b70_providers[device_index], "health", None)
            if health is None:
                provider_stats[str(device_index)] = {
                    "available": False,
                    "reason": "native health ABI unavailable",
                }
            else:
                provider_stats[str(device_index)] = _provider_health_stats(
                    health, _provider_health_baseline.get(device_index),
                )
        stats["synchronous_provider"] = provider_stats

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
    """Zero route and native poller counters, for per-phase measurement."""
    layers = _iter_hybrid_layers(worker.model_runner.model)
    for layer in layers:
        counter = getattr(layer, "_route_counter", None)
        if counter is not None:
            counter.zero_()
        partition_stats = getattr(
            layer, "shooting_brake_partition_stats", None
        )
        if partition_stats is not None:
            for key in tuple(partition_stats):
                partition_stats[key] = 0

    # The historical 186.05 us "decode" service figure was contaminated by
    # warmup and correctness because this reset used to omit the native
    # poller. Reset each card's poller once, not once per layer.
    from .b70_poller import _pollers as _b70_poller_registry

    for poller in _b70_poller_registry.values():
        poller.reset()

    # Provider health is monotonic. Snapshot rather than resetting it: a
    # before/after subtraction cannot silently relabel a cumulative count as
    # a scoped one, and both absolute values remain available for audit.
    from .routed_experts import _b70_providers

    _provider_health_baseline.clear()
    for device_index, provider in _b70_providers.items():
        health = provider.health
        _provider_health_baseline[device_index] = (
            int(health.generation),
            int(health.dispatches),
        )

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
