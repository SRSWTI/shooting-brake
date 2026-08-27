# Decode seam tracer -- decomposes the 5090-side per-token cost.
#
# The campaign's biggest unexplained number is ~7.9 ms/token of 5090-side
# orchestration (12.1 ms ITL minus ~2.5 ms Arc service minus ~2 ms tail).
# Because PIECEWISE breakable graphs re-enter Python for every routed layer
# (`HybridMoERunner._forward_impl` is an eager break), we can timestamp the
# seam itself without changing execution semantics.
#
# Per traced seam call this records:
#   host ns:  t_enter, t_issued (all lanes rung + local MoE enqueued),
#             t_taken (all takes enqueued), t_exit
#   CUDA evs: e_enter, e_pre_take, e_post_take, e_exit
#
# GPU elapsed(e_pre_take -> e_post_take) ~= stream parked on the Arc
# completion words (doorbell + Arc service as seen by the 5090) plus the
# small H2D result copy. elapsed(e_exit[L] -> e_enter[L+1]) is the
# attention/router segment between seams. Host deltas measure Python +
# enqueue cost, which the GPU timeline cannot see.
#
# Enable with SB_SEAM_TRACE=1. Sampling: every SB_SEAM_TRACE_EVERY-th
# decode step (default 4) up to SB_SEAM_TRACE_TOKENS sampled steps
# (default 400), then dumps SB_SEAM_TRACE_OUT
# (default /tmp/sb_seam_trace.npz) once and disarms. Also dumped at exit.
# Tracing is skipped during stream capture: recording events on a
# capturing stream would bake them into the graph.

from __future__ import annotations

import atexit
import os
import time

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_MAX_SEAM_M = 32  # decode-shaped dispatches only; prefill would drown decode


class _SeamRecord:
    __slots__ = ("layer", "m", "t", "ev")

    def __init__(self, layer: int, m: int) -> None:
        self.layer = layer
        self.m = m
        self.t = [0, 0, 0, 0]
        self.ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]


class SeamTracer:
    def __init__(self) -> None:
        self.every = max(1, int(os.environ.get("SB_SEAM_TRACE_EVERY", "4")))
        self.max_tokens = int(os.environ.get("SB_SEAM_TRACE_TOKENS", "400"))
        self.out = os.environ.get("SB_SEAM_TRACE_OUT", "/tmp/sb_seam_trace.npz")
        self.records: list[_SeamRecord] = []
        self.steps: list[tuple[int, int, int]] = []  # (t_ns, wall_ns, M)
        self._step_seen = 0
        self._tracing_step = False
        self._sampled = 0
        self._dumped = False
        self._steps_via_patch = False
        self._last_layer = 1 << 30
        logger.info(
            "[seam-trace] armed: every=%d max_tokens=%d out=%s",
            self.every, self.max_tokens, self.out,
        )
        atexit.register(self.dump)

    # -- step boundary (called from the Worker.execute_model wrapper) ----
    def step_begin(self, m_hint: int) -> bool:
        """Marks a step boundary; returns whether this step is traced."""
        self._steps_via_patch = True
        return self._advance_step(m_hint)

    def _advance_step(self, m_hint: int) -> bool:
        if self._dumped or m_hint > _MAX_SEAM_M:
            self._tracing_step = False
            return False
        self._step_seen += 1
        self._tracing_step = (self._step_seen % self.every) == 0
        if self._tracing_step:
            self._sampled += 1
            if self._sampled == 1:
                logger.info("[seam-trace] first sampled step")
        return self._tracing_step

    def step_end(self, t0_ns: int, m_hint: int) -> None:
        self.steps.append((t0_ns, time.perf_counter_ns() - t0_ns, m_hint))
        if self._sampled >= self.max_tokens and not self._dumped:
            self.dump()

    # -- seam points (called from _hybrid_forward_modular) ---------------
    def begin(self, layer: int, m: int) -> _SeamRecord | None:
        if self._dumped or m > _MAX_SEAM_M:
            return None
        if not self._steps_via_patch and layer <= self._last_layer:
            # The Worker step patch never engaged (vLLM version drift):
            # a non-increasing layer index marks a new decode step, so
            # sampling and the dump threshold work standalone.
            self._advance_step(m)
            if self._sampled >= self.max_tokens:
                self.dump()
        self._last_layer = layer
        if (
            not self._tracing_step
            or torch.cuda.is_current_stream_capturing()
        ):
            return None
        rec = _SeamRecord(layer, m)
        rec.ev[0].record()
        rec.t[0] = time.perf_counter_ns()
        self.records.append(rec)
        return rec

    @staticmethod
    def mark(rec: _SeamRecord | None, point: int) -> None:
        if rec is not None:
            rec.ev[point].record()
            rec.t[point] = time.perf_counter_ns()

    # -- output -----------------------------------------------------------
    def dump(self) -> None:
        if self._dumped or not self.records:
            return
        self._dumped = True
        torch.cuda.synchronize()
        n = len(self.records)
        layer = np.array([r.layer for r in self.records], dtype=np.int32)
        m = np.array([r.m for r in self.records], dtype=np.int32)
        host = np.array([r.t for r in self.records], dtype=np.int64)
        gpu = np.zeros((n, 3), dtype=np.float64)  # ms between event pairs
        gap = np.zeros(n, dtype=np.float64)  # exit[i-1] -> enter[i]
        for i, r in enumerate(self.records):
            try:
                gpu[i, 0] = r.ev[0].elapsed_time(r.ev[1])  # enqueue+local
                gpu[i, 1] = r.ev[1].elapsed_time(r.ev[2])  # arc wait + H2D
                gpu[i, 2] = r.ev[2].elapsed_time(r.ev[3])  # adds
                if i > 0 and layer[i] > layer[i - 1]:
                    gap[i] = self.records[i - 1].ev[3].elapsed_time(r.ev[0])
            except RuntimeError:
                gpu[i, :] = np.nan  # unrecorded pair (partial step)
        steps = np.array(self.steps, dtype=np.int64)
        np.savez(
            self.out,
            layer=layer, m=m, host_ns=host, gpu_ms=gpu, gap_ms=gap,
            steps=steps,
        )
        logger.info(
            "[seam-trace] dumped %d seam records / %d steps -> %s",
            n, len(self.steps), self.out,
        )


# The EngineCore child is spawned with a rebuilt environment that drops
# most custom vars (observed 2026-08-25: only the last 4 of 25 SB_*/
# SHOOTING_* entries survived into /proc/<EngineCore>/environ, while the
# API server had all 25). A file marker survives any env curation, so the
# tracer arms on EITHER the env flag or /tmp/sb_seam_trace.arm.
_ENABLED = (
    os.environ.get("SB_SEAM_TRACE") == "1"
    or os.path.exists("/tmp/sb_seam_trace.arm")
)
_TRACER: SeamTracer | None = None


def seam_tracer() -> SeamTracer | None:
    """Singleton, or None when disabled. The disabled path is one global
    check -- safe to call once per layer per token."""
    global _TRACER
    if not _ENABLED:
        return None
    if _TRACER is None:
        _TRACER = SeamTracer()
    return _TRACER
