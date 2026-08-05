"""B70 polling thread for Tier 3 graph-compatible dispatch.

Runs as a daemon thread, polling host-mapped signal flags.  When a flag
is set by the CUDA stream (via ``cuStreamWriteValue32``), the thread:

  1. Reads activation + routing data from pinned host buffers
  2. Translates global expert IDs → B70 compact slots
  3. Dispatches to the B70 provider via :class:`B70ProviderClient`
     (the same in-process ctypes binding used by the eager Tier-1/2
     path — proven working, correct ABI signatures)
  4. Writes the result to the pinned output buffer
  5. Sets the completion flag (detected by ``cuStreamWaitValue32``)

The thread handles ALL NVFP4 layers sequentially.  Each layer has its
own signal/completion flag pair.  The polling loop checks all layers in
round-robin order.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
from vllm.logger import init_logger

from .stream_signal import read_flag, write_flag_host

logger = init_logger(__name__)


class B70PollerThread(threading.Thread):
    """Daemon thread that polls signal flags and dispatches to B70.

    Each NVFP4 layer registers its signal/completion flag pair and
    pinned buffers via :meth:`register_layer`.  The polling loop
    checks all layers in round-robin.

    The B70 provider (SYCL/oneAPI, `libsb_b70_provider.so`) is loaded
    lazily on the FIRST call to :meth:`run`, via the same
    ``_get_b70_provider`` singleton the eager Tier-1/2 dispatch path
    uses — this guarantees identical, ABI-correct loading logic and a
    single shared provider instance across every layer's poller
    registration (no duplicate/conflicting `sb_b70_create` handles).
    """

    def __init__(self, placement: Any) -> None:
        super().__init__(daemon=True, name="B70Poller")
        self._placement = placement
        self._layers: tuple[dict[str, Any], ...] = ()
        self._layers_lock = threading.Lock()
        self._running = False
        self._stop_event = threading.Event()
        self._dispatch_count = 0
        self._error: BaseException | None = None

    def register_layer(
        self,
        layer_idx: int,
        signal_host: int,
        completion_host: int,
        pinned_hidden: np.ndarray,
        pinned_ids: np.ndarray,
        pinned_weights: np.ndarray,
        pinned_output: np.ndarray,
    ) -> None:
        """Register a layer's buffers for polling.

        Shapes are read from the arrays themselves; only the active
        ``[:M]`` prefix is dispatched, where M arrives as the signal
        flag's value.

        Args:
            layer_idx: NVFP4 layer index (0-31), indexes the B70 bank.
            signal_host: Host view of the signal flag; the CUDA stream
                writes M here to request a dispatch.
            completion_host: Host view of the completion flag; set to 1
                once the result is in ``pinned_output``.
            pinned_hidden: FP16 [max_batch, hidden] activation (pinned).
            pinned_ids: Int32 [max_batch, topk] compact B70 slots, -1 skips.
            pinned_weights: FP32 [max_batch, topk] routing weights.
            pinned_output: FP32 [max_batch, hidden] result target.
        """
        entry = {
            "layer_idx": layer_idx,
            "signal": signal_host,
            "completion": completion_host,
            "hidden": pinned_hidden,
            "ids": pinned_ids,
            "weights": pinned_weights,
            "output": pinned_output,
        }
        # Layers register lazily (first forward of each layer), which can
        # happen while the poll loop is already running.  Swapping in a
        # fresh tuple makes the loop's snapshot read atomic — it never
        # observes a partially built list.
        with self._layers_lock:
            self._layers = (*self._layers, entry)

    def start(self) -> None:
        self._running = True
        super().start()

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        self.join(timeout=5.0)

    @property
    def dispatch_count(self) -> int:
        return self._dispatch_count

    @property
    def error(self) -> BaseException | None:
        """First fault seen by the poller, if any.

        Results produced after a fault are NOT trustworthy: the
        completion flag is raised regardless so the CUDA side cannot
        hang, which means a failed dispatch leaves stale data in the
        output buffer.  Exact failed-route recovery is Phase 9.
        """
        return self._error

    def run(self) -> None:
        # Lazy, same-singleton load: identical code path as the eager
        # Tier-1/2 dispatch (`_get_b70_provider`), correct ABI signatures.
        from .routed_experts import _get_b70_provider

        try:
            provider = _get_b70_provider(self._placement)
        except BaseException as exc:  # noqa: BLE001 - see _fail_open
            self._fail_open(exc, "B70 provider load failed")
            return

        while self._running:
            layers = self._layers  # atomic snapshot; see register_layer
            pending = False
            for entry in layers:
                if not self._running:
                    break
                # The signal VALUE is the batch size M; 0 means idle.
                # The CUDA side bakes M into the captured graph, so no
                # extra host transfer is needed to size the dispatch.
                M = read_flag(entry["signal"])
                if M == 0:
                    continue
                pending = True

                # Reset immediately so the next replay can signal again.
                write_flag_host(entry["signal"], 0)

                try:
                    seq = provider.issue(
                        entry["layer_idx"],
                        entry["hidden"][:M],
                        entry["ids"][:M],
                        entry["weights"][:M],
                    )
                    provider.take(seq, M, output=entry["output"][:M])
                except BaseException as exc:  # noqa: BLE001 - see below
                    # Record and keep serving.  The completion flag is
                    # still raised below: the GPU is blocked in
                    # cuStreamWaitValue32, which has no timeout, so
                    # leaving it unset wedges the device permanently and
                    # the process stops responding to SIGKILL.
                    self._record_error(
                        exc, f"B70 dispatch failed "
                             f"(layer={entry['layer_idx']}, M={M})",
                    )

                write_flag_host(entry["completion"], 1)
                self._dispatch_count += 1

            # Idle back-off only when this sweep found nothing, so a
            # burst of layer signals is drained without sleeping.
            if not pending:
                time.sleep(0.00001)  # 10μs

    # -- fault containment ------------------------------------------------
    #
    # The CUDA side waits on the completion flag inside a captured graph.
    # That wait cannot time out and cannot be interrupted, so any poller
    # fault MUST still release every waiter — otherwise the GPU hangs and
    # the process becomes unkillable.  Errors are recorded and surfaced
    # via :attr:`error`; exact failed-route recovery is Phase 9.

    def _record_error(self, exc: BaseException, context: str) -> None:
        if self._error is None:
            self._error = exc
        logger.exception("Shooting Brake Tier 3: %s", context)

    def _fail_open(self, exc: BaseException, context: str) -> None:
        """Record a fatal fault, then release every present and future
        waiter so the CUDA side unblocks instead of hanging forever."""
        self._record_error(exc, context)
        self._running = False
        while True:
            layers = self._layers
            for entry in layers:
                write_flag_host(entry["completion"], 1)
            # Layers registered after the fault would otherwise wait on a
            # dead thread; keep releasing until the engine tears down.
            time.sleep(0.001)
            if self._stop_event.is_set():
                return


# --- Process-wide singleton -------------------------------------------
#
# There is exactly ONE physical B70.  A single SYCL queue serializes all
# dispatches anyway, so one poller thread serves every NVFP4 layer:
# 32 spinning threads would only add scheduler contention and fight over
# the same device.  Layers register into the shared instance as their
# real layer index becomes known (first forward).

_poller_singleton: B70PollerThread | None = None
_poller_lock = threading.Lock()


def get_b70_poller(placement: Any) -> B70PollerThread:
    """Return the shared poller thread, creating it on first use.

    The thread is NOT started here — the caller starts it once all
    registrations for the first forward pass have landed.
    """
    global _poller_singleton
    with _poller_lock:
        if _poller_singleton is None:
            _poller_singleton = B70PollerThread(placement)
        return _poller_singleton


__all__ = ["B70PollerThread", "get_b70_poller"]
