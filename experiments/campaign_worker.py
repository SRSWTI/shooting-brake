"""Experiment-only vLLM worker extension for bounded profile windows.

Load with --worker-extension-cls experiments.campaign_worker.CampaignWorkerExtension
on a loopback-only development server. Reuses vLLM's configured profiler and
Shooting Brake's existing telemetry/native trace ABI; installs no hot-path hooks.
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import time
from pathlib import Path


class CampaignWorkerExtension:
    def campaign_snapshot(self) -> dict:
        from shooting_brake_vllm.telemetry import collect_worker_stats

        return collect_worker_stats(self)

    def _campaign_clock_samples(self, label: str) -> list[dict]:
        import torch
        from shooting_brake_vllm.b70_poller import _pollers

        if not _pollers:
            raise RuntimeError("no native B70 pollers available for clock correlation")
        library = next(iter(_pollers.values()))._lib
        samples = []
        for index in range(5):
            name = f"sb.clock.{label}.{index}"
            monotonic = ctypes.c_uint64()
            realtime = ctypes.c_uint64()
            with torch.cuda.nvtx.range(name):
                before = time.monotonic_ns()
                library.sb_b70_clock_reference(ctypes.byref(monotonic), ctypes.byref(realtime))
                after = time.monotonic_ns()
            samples.append(
                {
                    "nvtx_name": name,
                    "monotonic_ns": monotonic.value,
                    "realtime_ns": realtime.value,
                    "host_before_ns": before,
                    "host_after_ns": after,
                    "within_host_bounds": before <= monotonic.value <= after,
                }
            )
        return samples

    def campaign_start(self, label: str, output_root: str, tracing: str = "0") -> dict:
        import torch

        if getattr(self, "_campaign_active", None) is not None:
            raise RuntimeError("a campaign window is already active")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
            raise ValueError("campaign label must be a simple filename component")
        if tracing not in ("0", "1"):
            raise ValueError("tracing must be 0 or 1")
        root = Path(output_root).resolve()
        destination = root / f"{label}.worker.json"
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite {destination}")
        torch.cuda.synchronize()
        before = self.campaign_snapshot()
        torch.cuda.reset_peak_memory_stats()
        if tracing == "1":
            self.profile(is_start=True, profile_prefix=label)
            if self.profiler is None or not self.profiler._running:
                raise RuntimeError("vLLM profiler did not start")
        try:
            clocks = self._campaign_clock_samples(f"{label}.start")
        except Exception:
            if tracing == "1":
                self.profile(is_start=False)
                self.profiler = None
            raise
        start = time.monotonic_ns()
        self._campaign_active = {
            "label": label,
            "destination": str(destination),
            "tracing": tracing == "1",
            "provider_profiling": os.environ.get("SHOOTING_BRAKE_B70_PROFILE") == "1",
            "start_monotonic_ns": start,
            "clock_start": clocks,
            "before": before,
        }
        return {"label": label, "start_monotonic_ns": start, "destination": str(destination)}

    def campaign_stop(self) -> dict:
        import torch
        from shooting_brake_vllm.b70_poller import _pollers

        state = getattr(self, "_campaign_active", None)
        if state is None:
            raise RuntimeError("no campaign window is active")
        torch.cuda.synchronize()
        end = time.monotonic_ns()
        try:
            clocks = self._campaign_clock_samples(f"{state['label']}.stop")
        finally:
            if state["tracing"]:
                self.profile(is_start=False)
                # vLLM annotates steps whenever a profiler object exists, even
                # after stop. Restore the uninstrumented annotation branch.
                self.profiler = None
        after = self.campaign_snapshot()
        devices = {}
        errors = []
        for device, poller in sorted(_pollers.items()):
            entries = [
                entry
                for entry in poller.trace_snapshot()
                if entry["t0_ns"] >= state["start_monotonic_ns"] and entry["t1_ns"] <= end
            ]
            old = state["before"]["poller"]["per_device"][str(device)]
            new = after["poller"]["per_device"][str(device)]
            dispatches = new["dispatches"] - old["dispatches"]
            rows = new["rows"] - old["rows"]
            native_errors = new["errors"] - old["errors"]
            if len(entries) != dispatches:
                errors.append(
                    f"device {device}: trace covers {len(entries)}/{dispatches} dispatches"
                )
            if sum(entry["M"] for entry in entries) != rows:
                errors.append(f"device {device}: trace row count disagrees with counters")
            if native_errors:
                errors.append(f"device {device}: {native_errors} native errors")
            devices[str(device)] = {
                "dispatches": dispatches,
                "rows": rows,
                "native_errors": native_errors,
                "trace_capacity": 1 << 16,
                "entries": entries,
                "timing_contract": (
                    "t0/t1 are host service bounds after signal observation; "
                    "kernel/total are raw device spans only when provider profiling is enabled; "
                    "cross-copy-engine total spans require independent clock validation"
                ),
            }
        if not all(item["within_host_bounds"] for item in state["clock_start"] + clocks):
            errors.append("native monotonic clock is outside sampled host bounds")
        result = {
            **state,
            "end_monotonic_ns": end,
            "clock_stop": clocks,
            "after": after,
            "devices": devices,
            "passed_trace_coverage": not errors,
            "errors": errors,
            "worker_pid": os.getpid(),
        }
        destination = Path(state["destination"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
        self._campaign_active = None
        return {
            "destination": str(destination),
            "passed_trace_coverage": not errors,
            "errors": errors,
            "devices": {
                device: {key: value[key] for key in ("dispatches", "rows", "native_errors")}
                for device, value in devices.items()
            },
        }
