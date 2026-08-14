"""Routing frequency counters and an opt-in token-level route trace.

The frequency counter is enabled by ``SHOOTING_BRAKE_ROUTE_STATS=1`` and
writes the existing aggregate CSV.  The locality trace is independent: set
``SHOOTING_BRAKE_ROUTE_TRACE=<path>`` to record the global, pre-placement
``topk_ids`` for every routed row.  An unset or empty trace path is off.

The trace is a packed little-endian binary stream.  Its 24-byte header is
``<8sHHHHII``:

* magic ``b"SBRTv1\\0\\0"``;
* format version (``uint16``, currently 1);
* header size (``uint16``, 24);
* record size (``uint16``, ``14 + 4 * top_k``);
* ``top_k`` (``uint16``);
* number of model layers and experts (two ``uint32`` values).

Records continue to EOF and have layout ``<QHI`` followed by ``top_k``
little-endian ``int32`` expert ids: monotonic per-layer step index (``uint64``),
layer index (``uint16``), row index within that forward's batch (``uint32``),
then the global expert ids.  There is deliberately no placement information:
routing precedes placement.  A step is one invocation of a given routed-expert
layer; equal step values across layers describe corresponding forwards.

The tracer allocates a ring of pinned D2H staging tensors and a binary output
block once, then reuses them.  ``observe`` schedules a non-blocking copy of the
already-existing ``topk_ids`` and later copies completed staging slots into the
preallocated output block.  It never allocates per decode step.  A larger than
previously seen prefill batch causes a one-time high-water-mark resize after
draining the ring.  Full blocks are written as they fill; the partial tail is
finalized at clean process exit.

Placement in this project has always been positional: ``LayerSubsetPolicy``
keeps experts ``0..cuda_per_layer-1`` on CUDA and sends the rest to the B70.
The aggregate histogram measures skew; it cannot answer whether consecutive
tokens in one row reuse experts.  The binary trace supplies the missing time
and row dimensions for ``benchmarks/route_locality.py``.
"""

from __future__ import annotations

import atexit
import logging
import os
import struct
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

_ENV = "SHOOTING_BRAKE_ROUTE_STATS"
_OUT_ENV = "SHOOTING_BRAKE_ROUTE_STATS_OUT"

_TRACE_ENV = "SHOOTING_BRAKE_ROUTE_TRACE"
_TRACE_MAGIC = b"SBRTv1\0\0"
_TRACE_VERSION = 1
_TRACE_HEADER = struct.Struct("<8sHHHHII")
_TRACE_STAGE_SLOTS = 8
_TRACE_BLOCK_RECORDS = 65_536


class RouteCounter:
    """Device-resident ``[n_layer, n_expert]`` activation counts.

    One instance is shared by every layer; :meth:`observe` is called with the
    global (pre-remap) ``topk_ids`` so the table describes the model's routing
    rather than any particular placement's view of it.
    """

    def __init__(self, num_layers: int, num_experts: int, device: torch.device):
        self.num_layers = num_layers
        self.num_experts = num_experts
        # int64: a long calibration run at 8 top-k over millions of tokens
        # will overflow int32 per-expert on a hot expert.
        #
        # inference_mode(False): construction happens lazily on the first
        # forward, which vLLM runs under torch.inference_mode(). A tensor
        # created there is an "inference tensor", and reset_worker_stats --
        # a collective_rpc handler running between steps, outside inference
        # mode -- cannot zero_() one. Allocating with inference mode
        # suspended yields normal tensors, writable from both contexts.
        with torch.inference_mode(False):
            self.counts = torch.zeros(
                num_layers, num_experts, dtype=torch.int64, device=device,
            )
            self.tokens = torch.zeros(
                num_layers, dtype=torch.int64, device=device,
            )
        self.top_k = 0

    def observe(self, layer_idx: int, topk_ids: torch.Tensor) -> None:
        """Accumulate one forward's routes for ``layer_idx``.

        ``topk_ids`` is ``[num_tokens, top_k]`` of global expert ids. Both
        updates are device-side and shape-static, so this is safe to call
        under CUDA graph capture.
        """
        if layer_idx >= self.num_layers:
            return
        if not self.top_k:
            self.top_k = int(topk_ids.shape[1])
        flat = topk_ids.reshape(-1).to(torch.int64)
        self.counts[layer_idx].scatter_add_(
            0, flat, torch.ones_like(flat),
        )
        self.tokens[layer_idx] += topk_ids.shape[0]

    # -- readout ---------------------------------------------------------
    # Everything below syncs, and is called once at the end of calibration.

    def to_cpu(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.counts.cpu(), self.tokens.cpu()

    def save_csv(self, path: str | Path) -> Path:
        counts, tokens = self.to_cpu()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# route histogram: n_layer={self.num_layers} "
            f"n_expert={self.num_experts} top_k={self.top_k}",
            "layer,expert,count",
        ]
        for il in range(self.num_layers):
            row = counts[il]
            if not int(row.sum()):
                continue
            for ie in range(self.num_experts):
                c = int(row[ie])
                if c:
                    lines.append(f"{il},{ie},{c}")
        out.write_text("\n".join(lines) + "\n")
        return out


class RouteTrace:
    """Buffered token-level route writer.

    One process-wide instance is shared by all layers.  Each layer owns an
    independent step counter because layers call the shared writer in order;
    this keeps corresponding model forwards on the same step number without a
    host-side synchronization or a separate begin-step hook.
    """

    def __init__(
        self,
        path: str | Path,
        num_layers: int,
        num_experts: int,
        topk_ids: torch.Tensor,
    ) -> None:
        if topk_ids.ndim != 2 or topk_ids.shape[1] <= 0:
            raise ValueError(
                "topk_ids must have shape [rows, top_k], got "
                f"{tuple(topk_ids.shape)}"
            )
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.top_k = int(topk_ids.shape[1])
        self.device = topk_ids.device
        self._record_dtype = np.dtype([
            ("step", "<u8"),
            ("layer", "<u2"),
            ("row", "<u4"),
            ("experts", "<i4", (self.top_k,)),
        ])
        record_size = self._record_dtype.itemsize
        header = _TRACE_HEADER.pack(
            _TRACE_MAGIC,
            _TRACE_VERSION,
            _TRACE_HEADER.size,
            record_size,
            self.top_k,
            num_layers,
            num_experts,
        )
        self._file = self.path.open("wb")
        self._file.write(header)
        self._records = np.empty(
            _TRACE_BLOCK_RECORDS, dtype=self._record_dtype,
        )
        self._record_count = 0
        self._steps = [0] * num_layers
        self._closed = False
        self._next_slot = 0
        self._is_cuda = self.device.type == "cuda"
        self._allocate_staging(max(128, int(topk_ids.shape[0])))
        atexit.register(self.close)

    def _allocate_staging(self, rows: int) -> None:
        """Allocate or replace the fixed staging ring at a new high water."""
        self._stage_rows = rows
        self._staging = [
            torch.empty(
                rows,
                self.top_k,
                dtype=torch.int32,
                device="cpu",
                pin_memory=self._is_cuda,
            )
            for _ in range(_TRACE_STAGE_SLOTS)
        ]
        self._staging_np = [tensor.numpy() for tensor in self._staging]
        self._events = (
            [torch.cuda.Event() for _ in range(_TRACE_STAGE_SLOTS)]
            if self._is_cuda
            else [None] * _TRACE_STAGE_SLOTS
        )
        self._pending = [False] * _TRACE_STAGE_SLOTS
        self._pending_rows = [0] * _TRACE_STAGE_SLOTS
        self._pending_steps = [0] * _TRACE_STAGE_SLOTS
        self._pending_layers = [0] * _TRACE_STAGE_SLOTS
        self._row_indices = np.arange(rows, dtype="<u4")
        self._next_slot = 0

    def _write_block(self) -> None:
        if self._record_count:
            self._records[:self._record_count].tofile(self._file)
            self._record_count = 0

    def _append_slot(self, slot: int) -> None:
        """Memcpy one completed staging slot into output blocks."""
        rows = self._pending_rows[slot]
        source = self._staging_np[slot]
        offset = 0
        while offset < rows:
            space = len(self._records) - self._record_count
            take = min(space, rows - offset)
            dst = self._records[
                self._record_count:self._record_count + take
            ]
            dst["step"] = self._pending_steps[slot]
            dst["layer"] = self._pending_layers[slot]
            dst["row"] = self._row_indices[offset:offset + take]
            dst["experts"] = source[offset:offset + take]
            self._record_count += take
            offset += take
            if self._record_count == len(self._records):
                self._write_block()
        self._pending[slot] = False

    def _finish_slot(self, slot: int, wait: bool) -> bool:
        if not self._pending[slot]:
            return True
        event = self._events[slot]
        if event is not None:
            if wait:
                event.synchronize()
            elif not event.query():
                return False
        self._append_slot(slot)
        return True

    def _drain_ready(self) -> None:
        for slot in range(_TRACE_STAGE_SLOTS):
            self._finish_slot(slot, wait=False)

    def _drain_all(self) -> None:
        for slot in range(_TRACE_STAGE_SLOTS):
            self._finish_slot(slot, wait=True)

    def observe(self, layer_idx: int, topk_ids: torch.Tensor) -> None:
        """Schedule one forward's existing route ids for buffered output."""
        if self._closed:
            raise RuntimeError("route trace is already closed")
        if not 0 <= layer_idx < self.num_layers:
            raise ValueError(f"layer index {layer_idx} is out of range")
        if topk_ids.ndim != 2 or int(topk_ids.shape[1]) != self.top_k:
            raise ValueError(
                f"topk_ids shape changed from [rows, {self.top_k}] to "
                f"{tuple(topk_ids.shape)}"
            )
        if topk_ids.device != self.device:
            raise ValueError(
                f"topk_ids device changed from {self.device} to "
                f"{topk_ids.device}"
            )

        rows = int(topk_ids.shape[0])
        if rows > self._stage_rows:
            self._drain_all()
            self._allocate_staging(rows)
        self._drain_ready()
        slot = self._next_slot
        self._finish_slot(slot, wait=True)

        step = self._steps[layer_idx]
        self._steps[layer_idx] = step + 1
        self._staging[slot][:rows].copy_(
            topk_ids, non_blocking=self._is_cuda,
        )
        self._pending_rows[slot] = rows
        self._pending_steps[slot] = step
        self._pending_layers[slot] = layer_idx
        self._pending[slot] = True
        if self._is_cuda:
            self._events[slot].record(torch.cuda.current_stream(self.device))
        else:
            self._append_slot(slot)
        self._next_slot = (slot + 1) % _TRACE_STAGE_SLOTS

    def flush(self) -> None:
        """Wait for pending D2H copies and make every record durable."""
        if self._closed:
            return
        self._drain_all()
        self._write_block()
        self._file.flush()

    def close(self) -> None:
        """Finalize the stream; safe to call more than once."""
        if self._closed:
            return
        try:
            self.flush()
        finally:
            self._file.close()
            self._closed = True


def load_csv(path: str | Path) -> tuple[torch.Tensor, int]:
    """Read a histogram written by :meth:`RouteCounter.save_csv`.

    Returns the ``[n_layer, n_expert]`` count table and ``top_k``.
    """
    text = Path(path).read_text().splitlines()
    n_layer = n_expert = top_k = 0
    for line in text:
        if line.startswith("#"):
            for tok in line.replace("#", " ").split():
                if "=" in tok:
                    k, _, v = tok.partition("=")
                    if k == "n_layer":
                        n_layer = int(v)
                    elif k == "n_expert":
                        n_expert = int(v)
                    elif k == "top_k":
                        top_k = int(v)
            break
    if not (n_layer and n_expert):
        raise ValueError(f"{path}: missing or malformed header")
    counts = torch.zeros(n_layer, n_expert, dtype=torch.int64)
    for line in text:
        if line.startswith("#") or line.startswith("layer,"):
            continue
        if not line.strip():
            continue
        il, ie, c = line.split(",")
        counts[int(il), int(ie)] = int(c)
    return counts, top_k


def analyze(counts: torch.Tensor, top_k: int) -> dict:
    """StreamMoE-style skew summary.

    ``n50``/``n80``/``n90`` are the number of experts needed to cover that
    percentage of a layer's activations. Against a uniform router they would
    sit at 50/80/90% of ``n_expert``; well below that is the skew the README
    assumes and the knapsack would exploit.
    """
    n_layer, n_expert = counts.shape
    layers = []
    for il in range(n_layer):
        row = counts[il]
        total = int(row.sum())
        if not total:
            layers.append(None)
            continue
        srt, _ = torch.sort(row, descending=True)
        cum = torch.cumsum(srt, 0)
        unique = int((srt > 0).sum())

        def cover(pct: int) -> int:
            need = (total * pct + 99) // 100
            idx = int(torch.searchsorted(cum, torch.tensor(need)))
            return min(idx + 1, n_expert)

        def top(n: int) -> float:
            n = min(n, n_expert)
            return 100.0 * float(cum[n - 1]) / total

        layers.append({
            "layer": il,
            "unique": unique,
            "total": total,
            "top10_pct": top(10),
            "top30_pct": top(30),
            "top60_pct": top(60),
            "n50": cover(50),
            "n80": cover(80),
            "n90": cover(90),
        })

    live = [row for row in layers if row]
    summary = {
        "n_layer": n_layer,
        "n_expert": n_expert,
        "top_k": top_k,
        "active_layers": len(live),
        "layers": layers,
    }
    if live:
        summary["mean_n80"] = sum(r["n80"] for r in live) / len(live)
        summary["mean_top10_pct"] = sum(r["top10_pct"] for r in live) / len(live)
        # The honest null hypothesis: a uniform router needs 80% of experts to
        # cover 80% of traffic. Skew is how far below that the real n80 sits.
        summary["uniform_n80"] = 0.8 * n_expert
        summary["skew_ratio"] = summary["uniform_n80"] / summary["mean_n80"]
    return summary


def format_analysis(summary: dict) -> str:
    lines = [
        "=== Expert frequency analysis ===",
        f"{summary['n_layer']} layers x {summary['n_expert']} experts, "
        f"top-{summary['top_k']} routing "
        f"({summary['active_layers']} layers observed)",
        "",
        f"{'layer':>5} {'unique':>7} {'top10':>7} {'top30':>7} {'top60':>7} "
        f"{'n50':>5} {'n80':>5} {'n90':>5}",
    ]
    for row in summary["layers"]:
        if not row:
            continue
        lines.append(
            f"{row['layer']:>5} {row['unique']:>7} "
            f"{row['top10_pct']:>6.1f}% {row['top30_pct']:>6.1f}% "
            f"{row['top60_pct']:>6.1f}% "
            f"{row['n50']:>5} {row['n80']:>5} {row['n90']:>5}"
        )
    if "mean_n80" in summary:
        lines += [
            "",
            "--- summary ---",
            f"experts covering 80% of routes: {summary['mean_n80']:.1f} per layer "
            f"(uniform router would need {summary['uniform_n80']:.0f})",
            f"skew ratio: {summary['skew_ratio']:.2f}x "
            f"(1.0 = no skew, higher = hotter tail)",
            f"top 10 experts hold {summary['mean_top10_pct']:.1f}% of routes "
            f"(uniform: {100.0 * 10 / summary['n_expert']:.1f}%)",
        ]
    return "\n".join(lines)


# -- process-wide singleton ----------------------------------------------

_counter: RouteCounter | None = None
_trace: RouteTrace | None = None
_checked = False


def enabled() -> bool:
    return os.environ.get(_ENV) == "1"


def trace_enabled() -> bool:
    """Whether a non-empty route-trace output path was requested."""
    return bool(os.environ.get(_TRACE_ENV))


def get_counter(
    num_layers: int, num_experts: int, device: torch.device,
) -> RouteCounter | None:
    """The shared counter, or ``None`` when the flag is off.

    Construction is lazy so that an off run allocates nothing at all.
    """
    global _counter, _checked
    if not enabled():
        return None
    if _counter is None:
        _counter = RouteCounter(num_layers, num_experts, device)
        if not _checked:
            _checked = True
            logger.warning(
                "Shooting Brake: route statistics enabled (%d layers x %d "
                "experts). This adds a device-side scatter_add per layer.",
                num_layers, num_experts,
            )
    return _counter


def get_trace(
    num_layers: int,
    num_experts: int,
    topk_ids: torch.Tensor,
) -> RouteTrace | None:
    """The shared trace writer, or ``None`` when its path is unset."""
    global _trace
    target = os.environ.get(_TRACE_ENV)
    if not target:
        return None
    if _trace is None:
        _trace = RouteTrace(target, num_layers, num_experts, topk_ids)
        logger.warning(
            "Shooting Brake: token route trace enabled at %s "
            "(%d layers x %d experts, top-%d)",
            _trace.path,
            num_layers,
            num_experts,
            _trace.top_k,
        )
    elif (
        _trace.num_layers != num_layers
        or _trace.num_experts != num_experts
        or _trace.top_k != int(topk_ids.shape[1])
    ):
        raise ValueError("route trace model dimensions changed after creation")
    return _trace


def peek_trace() -> RouteTrace | None:
    """The route writer if one exists, without creating it."""
    return _trace


def flush_trace() -> Path | None:
    """Synchronize pending copies and flush the trace's partial output block."""
    if _trace is None:
        return None
    _trace.flush()
    return _trace.path


def peek() -> RouteCounter | None:
    """The counter if one exists, without creating it."""
    return _counter


def dump(path: str | Path | None = None) -> Path | None:
    """Write the histogram, defaulting to ``SHOOTING_BRAKE_ROUTE_STATS_OUT``."""
    if _counter is None:
        return None
    target = path or os.environ.get(_OUT_ENV) or "route_stats.csv"
    out = _counter.save_csv(target)
    logger.warning("Shooting Brake: route histogram written to %s", out)
    return out
