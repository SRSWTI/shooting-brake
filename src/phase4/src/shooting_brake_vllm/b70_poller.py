"""Host-side watcher that drives the B70 for Tier 3 graph dispatch.

Tier 3 dispatches to the B70 from inside a captured CUDA graph: the
stream writes the batch size M to a host-mapped signal flag
(``cuStreamWriteValue32``), then parks on a completion flag
(``cuStreamWaitValue32``).  Something on the host has to notice the
signal and drive the provider.

That watcher cannot live in Python.  Measured on the reference host, a
yielding poll loop costs ~55 us per wakeup — more than the B70 kernel
itself, and paid once per layer per token — while a spinning one holds
the GIL and starves the engine thread outright.  The loop therefore runs
on a native thread inside ``libsb_b70_provider.so``
(``sb_b70_poll_*``); this module is only the handle that owns it.

There is exactly one physical B70 and one SYCL queue, so a single poller
serves every NVFP4 layer.  Layers register as their real index becomes
known, which is their first forward.
"""

from __future__ import annotations

import ctypes
import json
import os
import struct
import threading
from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


_M_BUCKET_LABELS = ("1", "2", "3-4", "5-8", "9-16", "17-32", ">32")


class B70Poller:
    """Handle for the native polling thread.

    Failure semantics: the CUDA-side wait cannot time out, so the native
    loop raises a layer's completion flag even when its dispatch fails —
    leaving it unset wedges the device permanently and makes the process
    unkillable.  A failed dispatch therefore leaves stale data in the
    output buffer and is reported through :attr:`error_count`, which
    callers must check.  Exact failed-route recovery is Phase 9.
    """

    def __init__(self, provider: Any, generation: int = 1) -> None:
        self._provider = provider
        self._lib = provider.lib
        self._handle = ctypes.c_void_p(
            self._lib.sb_b70_poll_create(provider.handle, generation)
        )
        if not self._handle:
            raise RuntimeError("sb_b70_poll_create returned NULL")
        self._started = False
        self._layers = 0
        self._lib.sb_b70_poll_trace_snapshot.restype = ctypes.c_size_t
        self._lib.sb_b70_poll_trace_snapshot.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
        ]
        self._lib.sb_b70_clock_reference.restype = None
        self._lib.sb_b70_clock_reference.argtypes = [
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
        ]
        # Opt-in decode-trace dump: SHOOTING_BRAKE_B70_TRACE_DUMP=<path>
        # snapshots the native per-dispatch trace ring to <path> every few
        # seconds (atomic replace), for merging with a same-process torch
        # profiler capture on the shared CLOCK_MONOTONIC timeline.
        self._trace_dump_path = os.environ.get(
            "SHOOTING_BRAKE_B70_TRACE_DUMP"
        )
        self._trace_thread: threading.Thread | None = None
        self._trace_stop = threading.Event()

    def register_layer(
        self,
        layer_idx: int,
        signal_host: int,
        completion_host: int,
        pinned_hidden: torch.Tensor,
        pinned_ids: torch.Tensor,
        pinned_weights: torch.Tensor,
        pinned_output: torch.Tensor,
    ) -> None:
        """Register one layer's flags and pinned buffers.

        Safe to call while the poller runs.  The caller must keep every
        buffer alive for the poller's lifetime — the native side stores
        raw pointers and never takes a reference.

        Args:
            layer_idx: Absolute layer index (0-31), indexes the B70 bank.
            signal_host: Host view of the signal flag; the CUDA stream
                writes M here to request a dispatch.
            completion_host: Host view of the completion flag; set to 1
                once the result is in ``pinned_output``.
            pinned_hidden: FP16 [max_batch, hidden] activation.
            pinned_ids: Int32 [max_batch, topk] compact B70 slots, -1 skips.
            pinned_weights: FP32 [max_batch, topk] routing weights.
            pinned_output: FP32 [max_batch, hidden] result target.
        """
        status = self._lib.sb_b70_poll_register(
            self._handle,
            ctypes.c_size_t(layer_idx),
            ctypes.c_void_p(signal_host),
            ctypes.c_void_p(completion_host),
            ctypes.c_void_p(pinned_hidden.data_ptr()),
            ctypes.cast(pinned_ids.data_ptr(), ctypes.POINTER(ctypes.c_int32)),
            ctypes.cast(
                pinned_weights.data_ptr(), ctypes.POINTER(ctypes.c_float)
            ),
            ctypes.cast(
                pinned_output.data_ptr(), ctypes.POINTER(ctypes.c_float)
            ),
            ctypes.c_size_t(pinned_ids.shape[1]),
        )
        if status != 0:
            raise RuntimeError(
                f"sb_b70_poll_register failed for layer {layer_idx} "
                f"(status={status})"
            )
        self._layers += 1

    def start(self) -> None:
        if self._started:
            return
        if self._lib.sb_b70_poll_start(self._handle) != 0:
            raise RuntimeError("sb_b70_poll_start failed")
        self._started = True
        if self._trace_dump_path and self._trace_thread is None:
            self._trace_thread = threading.Thread(
                target=self._trace_dump_loop, daemon=True,
                name="sb-b70-trace-dump",
            )
            self._trace_thread.start()
            logger.info(
                "Shooting Brake B70 trace dump enabled -> %s",
                self._trace_dump_path,
            )

    def reset(self) -> None:
        """Zero native timing, shape, dispatch, and error counters."""
        self._lib.sb_b70_poll_reset(self._handle)

    def trace_snapshot(self) -> list[dict]:
        """Most recent per-dispatch windows from the native trace ring.

        Timestamps are host CLOCK_MONOTONIC nanoseconds -- directly
        comparable with a torch-profiler capture from this process.
        """
        capacity = 1 << 16
        entry_bytes = 40
        buf = (ctypes.c_uint8 * (capacity * entry_bytes))()
        n = int(self._lib.sb_b70_poll_trace_snapshot(
            self._handle, buf, ctypes.c_size_t(capacity)))
        entries = []
        for i in range(n):
            t0, t1, kernel, total, layer, m = struct.unpack_from(
                "<QQQQII", buf, i * entry_bytes)
            entries.append({
                "t0_ns": t0, "t1_ns": t1, "kernel_ns": kernel,
                "total_ns": total, "layer": layer, "M": m,
            })
        return entries

    def _trace_dump_loop(self) -> None:
        """Periodically snapshot the ring to the dump path (atomic replace).

        Tracing must never hurt serving: every failure is swallowed and the
        loop simply tries again next period.
        """
        path = self._trace_dump_path
        while not self._trace_stop.wait(5.0):
            try:
                monotonic = ctypes.c_uint64()
                realtime = ctypes.c_uint64()
                self._lib.sb_b70_clock_reference(
                    ctypes.byref(monotonic), ctypes.byref(realtime))
                payload = {
                    "clock_monotonic_ns": monotonic.value,
                    "clock_realtime_ns": realtime.value,
                    "entries": self.trace_snapshot(),
                }
                tmp = f"{path}.tmp"
                with open(tmp, "w") as f:
                    json.dump(payload, f)
                os.replace(tmp, path)
            except Exception:
                pass

    def stop(self) -> None:
        if not self._started:
            return
        self._trace_stop.set()
        self._lib.sb_b70_poll_stop(self._handle)
        self._started = False
        self.report()

    def report(self) -> None:
        """Log dispatch counts and mean service time."""
        logger.info(
            "Shooting Brake Tier 3 B70 poller: %d dispatches over %d layers, "
            "service mean=%.1fus, %d errors",
            self.dispatch_count, self._layers,
            self.service_mean_us, self.error_count,
        )

    @property
    def started(self) -> bool:
        return self._started

    @property
    def dispatch_count(self) -> int:
        return int(self._lib.sb_b70_poll_dispatch_count(self._handle))

    @property
    def row_count(self) -> int:
        """Total rows served: the sum of M over every dispatch."""
        return int(self._lib.sb_b70_poll_row_count(self._handle))

    @property
    def m_histogram(self) -> dict[str, int]:
        """Dispatch counts bucketed by M, to expose mixed decode/prefill data."""
        return {
            label: int(
                self._lib.sb_b70_poll_m_bucket_count(self._handle, bucket)
            )
            for bucket, label in enumerate(_M_BUCKET_LABELS)
        }

    @property
    def error_count(self) -> int:
        """Failed dispatches.  Nonzero means results are untrustworthy."""
        return int(self._lib.sb_b70_poll_error_count(self._handle))

    @property
    def service_mean_us(self) -> float:
        """Mean issue+take time the poller itself observed.

        A lower bound on what Tier 3 adds per layer: the CUDA side's
        exposed wait is this plus signal-detection latency, minus
        whatever overlaps the concurrent CUDA expert compute.
        """
        n = self.dispatch_count
        if n == 0:
            return 0.0
        return int(self._lib.sb_b70_poll_service_ns(self._handle)) / n / 1000.0

    @property
    def total_mean_us(self) -> float:
        """Mean profiled device-queue time per dispatch.

        Zero unless ``SHOOTING_BRAKE_B70_PROFILE=1``. This spans the first
        input copy through completion of the output copy. Level Zero copy
        engines need not share a profiling timebase, so this raw span can
        underflow. It is diagnostic only unless endpoint-clock comparability
        has been established; only then does its difference from
        :attr:`service_mean_us` isolate host-side issue/take work.
        """
        n = self.dispatch_count
        if n == 0:
            return 0.0
        return int(self._lib.sb_b70_poll_total_ns(self._handle)) / n / 1000.0

    @property
    def kernel_mean_us(self) -> float:
        """Mean on-device kernel time per dispatch.

        Zero unless ``SHOOTING_BRAKE_B70_PROFILE=1``; the provider only
        timestamps commands on a profiled queue.

        When :attr:`total_mean_us` has a valid timebase, their difference is
        device-queue work outside the kernel. Profiling adds two marker
        submissions per dispatch, so quantify its perturbation with a separate
        unprofiled run.
        """
        n = self.dispatch_count
        if n == 0:
            return 0.0
        return int(self._lib.sb_b70_poll_kernel_ns(self._handle)) / n / 1000.0

    @property
    def kernel_mean_us_per_row(self) -> float:
        """Mean profiled kernel time per input row across all dispatches."""
        rows = self.row_count
        if rows == 0:
            return 0.0
        return (
            int(self._lib.sb_b70_poll_kernel_ns(self._handle))
            / rows
            / 1000.0
        )

    def __del__(self) -> None:
        try:
            if self._handle:
                self._lib.sb_b70_poll_destroy(self._handle)
                self._handle = ctypes.c_void_p(0)
        except Exception:  # noqa: BLE001 - interpreter teardown
            pass


# --- Process-wide singleton -------------------------------------------
#
# One physical B70, one SYCL queue: a single poller serves every layer.

_poller_singleton: B70Poller | None = None


def get_b70_poller(placement: Any) -> B70Poller:
    """Return the shared poller, creating it on first use.

    The thread is NOT started here — the caller starts it once the first
    layer has registered.
    """
    global _poller_singleton
    if _poller_singleton is None:
        from .routed_experts import _get_b70_provider

        _poller_singleton = B70Poller(_get_b70_provider(placement))
    return _poller_singleton


__all__ = ["B70Poller", "get_b70_poller"]
