"""Phase 6 — stream cold expert weights from DRAM to the 5090 for prefill.

The CPU tier computes its own FFN on CPU cores, which is the right call for
decode and the wrong one for prefill.

At M=1 an expert pass reads every weight once and reuses nothing, so it is a
pure DRAM stream: ~9 MiB at the measured ~48 GB/s, ~195 us, and the
arithmetic is free by comparison. Shipping the same weights to a GPU instead
costs about the same, because the transfer reads that identical 9 MiB. The
two designs tie.

They stop tying as soon as tokens accumulate. Weight traffic is fixed per
expert, but arithmetic grows with the token count, so the CPU crosses from
bandwidth-bound to compute-bound and then falls off a cliff: at M=2048 a
layer's cold experts need roughly 4.8 GMAC, which is tens of milliseconds on
CPU cores against ~1.3 ms to stream the same weights to a GPU that finishes
the GEMM in microseconds. Hence the threshold in ``stream_threshold`` --
below it, compute where the weights already are; above it, move the weights.

Streaming targets the 5090 and only the 5090. The B70 hangs off a PCIe Gen3
x4 chipset link measured at ~3.9 GB/s, so staging one 9 MiB expert into it
costs ~2.3 ms against ~150 us for the 5090's Gen5 x16 -- 16x worse, and worse
than not streaming at all. The B70 keeps the experts it owns and never
receives streamed ones.

Transfers are pipelined against compute through a small ring of device slots.
The pipeline is deliberately shallow: the work is transfer-bound, so extra
slots buy nothing beyond keeping the copy engine fed, and each one costs a
full expert's worth of VRAM.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

try:
    from vllm.logger import init_logger
except ImportError:  # standalone use, outside a vLLM process
    import logging

    def init_logger(name: str):  # type: ignore[misc]
        return logging.getLogger(name)

logger = init_logger(__name__)

#: Token count above which prefill streams instead of computing on CPU.
#:
#: Measured (phase7/stream_crossover_bench.py, 8 cold experts of the
#: qualified model's shape, routing sampled as top-8 of 256):
#:
#:       M     CPU cores      streamed    speedup
#:       4        1.17 ms       0.72 ms      1.6x
#:      32        5.31 ms       1.04 ms      5.1x
#:     128       14.08 ms       1.62 ms      8.7x
#:    2048      128.50 ms       1.63 ms     78.8x
#:
#: Streaming is flat because its cost is the fixed 72 MiB weight transfer,
#: independent of M; the CPU curve is linear in routed tokens. Isolated,
#: they cross at M=4.
#:
#: The default stays at 128 regardless, because the isolated comparison
#: misses where each cost lands. The CPU tier runs on host cores and is
#: issued early to hide under CUDA compute, so at decode its ~195 us can
#: cost nothing at all. Streaming occupies the CUDA stream itself and is
#: therefore always on the critical path. Below the decode/prefill boundary
#: the cheap-looking option is the one that consumes a resource nothing else
#: wants; above it, no amount of overlap rescues a 128 ms CPU pass.
#:
#: Lower this only to benchmark the CPU side at prefill sizes.
DEFAULT_STREAM_THRESHOLD = 128

#: Device slots in the transfer/compute ring. Three is enough to keep the
#: copy engine fed while one slot is being read; more only helps if compute
#: ever rivals transfer, which the measurements above say it never does.
#: Each slot costs one expert of VRAM (~9 MiB for the qualified model).
DEFAULT_SLOTS = 3


def stream_threshold() -> int:
    """Token count at or above which prefill streams instead of computing."""
    return int(
        os.environ.get(
            "SHOOTING_BRAKE_CPU_STREAM_T", str(DEFAULT_STREAM_THRESHOLD)
        )
    )


class ExpertStreamer:
    """Computes cold-expert routes on CUDA by streaming weights from DRAM.

    One instance is shared by every layer: the ring and its slots are reused
    across layers because layers run strictly in sequence, so peak VRAM is
    ``slots * expert_bytes`` rather than a per-layer cost.
    """

    def __init__(
        self,
        host: object,
        hidden: int,
        intermediate: int,
        *,
        slots: int = DEFAULT_SLOTS,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self._host = host
        self.hidden = int(hidden)
        self.intermediate = int(intermediate)
        self.device = torch.device(device)
        self.dtype = dtype

        self._plane = self.hidden * self.intermediate
        self._slots = max(1, int(slots))
        # Packed NVFP4: half a byte per weight plus one e4m3 block scale per
        # 16, three planes per expert.
        self._block_bytes = 3 * (self._plane // 2 + self._plane // 16)

        # Ring of device-side slots holding the arena's packed bytes verbatim,
        # plus the events that keep a copy from overwriting a slot still being
        # read. uint8 because these are packed bytes, not weights: the unpack
        # happens after the transfer, which is exactly what makes the transfer
        # 3.56x smaller.
        self._ring = torch.empty(
            self._slots, self._block_bytes, dtype=torch.uint8,
            device=self.device,
        )
        self._copy_stream = torch.cuda.Stream(device=self.device)
        self._copy_done = [torch.cuda.Event() for _ in range(self._slots)]
        self._compute_done = [torch.cuda.Event() for _ in range(self._slots)]
        for ev in self._compute_done:
            # Recorded up front so the first pass through the ring has
            # something to wait on rather than a special case.
            ev.record()

        # E2M1 magnitude ladder, indexed by nibble: sign in bit 3, magnitude
        # in bits 0-2. Resident on device so the unpack is a gather rather
        # than a host round trip.
        mags = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32,
        )
        self._e2m1 = torch.cat((mags, -mags)).to(self.device)

        # Host-side arena views and per-expert global scales, built once and
        # reused. Rebuilding either per call would allocate on the prefill
        # path.
        self._blocks: dict[tuple[int, int], torch.Tensor] = {}
        self._gscales: dict[tuple[int, int], tuple[torch.Tensor, ...]] = {}

        self.experts_streamed = 0
        self.bytes_streamed = 0
        self.calls = 0

    # -- internals --------------------------------------------------------

    def _block(self, layer: int, expert: int) -> torch.Tensor:
        key = (layer, expert)
        blk = self._blocks.get(key)
        if blk is None:
            blk = self._host.expert_block(layer, expert)  # type: ignore[attr-defined]
            self._blocks[key] = blk
        return blk

    def _gscale(self, layer: int, expert: int) -> tuple[torch.Tensor, ...]:
        """Per-plane global scales as device scalars, cached.

        Cached as tensors rather than floats because the dequantizer takes a
        device scalar: building one per call meant a fresh allocation plus a
        tiny H2D copy for every plane of every expert of every layer, which
        showed up as a measurable prefill regression -- hundreds of
        micro-transfers per step, each dwarfed by its own launch overhead.
        """
        key = (layer, expert)
        gs = self._gscales.get(key)
        if gs is None:
            gs = tuple(
                torch.tensor(v, dtype=torch.float32, device=self.device)
                for v in self._host.expert_gscales(layer, expert)  # type: ignore[attr-defined]
            )
            self._gscales[key] = gs
        return gs

    def _views(
        self, slot: int, gscales: tuple[float, float, float]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unpack a ring slot into dequantized gate, up and down.

        The slot holds the arena's packed bytes verbatim:
        ``gate_q | gate_sf | up_q | up_sf | down_q | down_sf``. Unpacking on
        the GPU is the point of streaming packed rather than dequantized
        data -- the transfer carries 3.56x fewer bytes, and the 5090 absorbs
        the decode.

        Layout matches the arena: ``gate`` and ``up`` are
        ``[intermediate, hidden]`` and ``down`` is ``[hidden, intermediate]``.
        """
        flat = self._ring[slot]
        out = []
        off = 0
        for (rows, cols), gs in zip(
            ((self.intermediate, self.hidden),
             (self.intermediate, self.hidden),
             (self.hidden, self.intermediate)),
            gscales,
        ):
            nq = rows * cols // 2
            nsf = rows * cols // 16
            q = flat[off:off + nq].view(rows, cols // 2)
            sf = flat[off + nq:off + nq + nsf].view(rows, cols // 16)
            off += nq + nsf
            out.append(self._dequant(q, sf, gs))
        return out[0], out[1], out[2]

    def _dequant(
        self, q: torch.Tensor, sf: torch.Tensor, gscale: torch.Tensor
    ) -> torch.Tensor:
        """packed NVFP4 -> dtype: e2m1(nib) * e4m3(sf) * gscale.

        Delegates to vLLM's fused Triton dequantizer rather than expressing
        the unpack as torch ops. Not a style preference: the elementwise form
        materialises several full-size intermediates, and measured on this
        path it cost more than the 3.56x of PCIe traffic that packing saves,
        turning a transfer win into a net loss. The fused kernel makes one
        pass.

        ``swizzle=False`` is the matching entry point -- the arena stores
        linear block scales, having undone the checkpoint's swizzle at load
        time.
        """
        from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (  # noqa: E501
            dequantize_to_dtype,
        )

        return dequantize_to_dtype(
            q, sf, gscale, self.dtype, block_size=16, swizzle=False,
        )

    @staticmethod
    def _group_routes(
        cpu_ids: torch.Tensor, topk_weights: torch.Tensor
    ) -> tuple[list[int], torch.Tensor, torch.Tensor, list[int]]:
        """Bucket routes by expert.

        Returns the expert ids present, the per-route token rows and routing
        weights sorted so each expert's routes are contiguous, and the bucket
        boundaries. Sorting on device keeps this to a single host sync for
        the expert list, which prefill can afford and decode could not.
        """
        topk = cpu_ids.shape[1]
        flat = cpu_ids.reshape(-1)
        sel = torch.nonzero(flat >= 0, as_tuple=True)[0]
        if sel.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=cpu_ids.device)
            return [], empty, empty.float(), empty

        eids = flat[sel]
        order = torch.argsort(eids)
        sel = sel[order]
        eids = eids[order]

        rows = torch.div(sel, topk, rounding_mode="floor")
        weights = topk_weights.reshape(-1)[sel]

        uniq, counts = torch.unique_consecutive(eids, return_counts=True)
        bounds = torch.cumsum(counts, 0)
        # The only sync in the call: which experts to stream cannot be known
        # device-side without launching a kernel per possible expert.
        return uniq.tolist(), rows, weights, bounds.tolist()

    # -- public API -------------------------------------------------------

    def forward(
        self,
        layer_idx: int,
        x: torch.Tensor,
        cpu_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Routing-weighted partial for this layer's cold routes, on CUDA.

        Mirrors the contract of the CPU poller and the B70 provider so all
        three partials sum: the result covers exactly the routes selected by
        ``cpu_ids``, weighted, and is zero where there are none.

        Args:
            layer_idx: absolute layer index, keying the arena.
            x: ``[M, hidden]`` activation on CUDA.
            cpu_ids: ``[M, topk]`` global expert ids for cold routes, -1
                elsewhere.
            topk_weights: ``[M, topk]`` routing weights, unmodified.

        Returns:
            ``[M, hidden]`` partial in ``x``'s dtype.
        """
        out = torch.zeros(
            x.shape[0], self.hidden, dtype=torch.float32, device=x.device,
        )
        experts, rows, weights, bounds = self._group_routes(
            cpu_ids, topk_weights
        )
        if not experts:
            return out.to(x.dtype)

        self.calls += 1
        n = len(experts)
        compute_stream = torch.cuda.current_stream()

        def issue(i: int) -> None:
            """Start expert i's transfer into its ring slot."""
            slot = i % self._slots
            # Do not overwrite a slot whose previous tenant is still being
            # read. Without this the ring silently corrupts under load.
            self._copy_stream.wait_event(self._compute_done[slot])
            with torch.cuda.stream(self._copy_stream):
                self._ring[slot].copy_(
                    self._block(layer_idx, experts[i]), non_blocking=True,
                )
            self._copy_done[slot].record(self._copy_stream)

        for i in range(min(self._slots, n)):
            issue(i)

        for i in range(n):
            slot = i % self._slots
            compute_stream.wait_event(self._copy_done[slot])

            begin = 0 if i == 0 else bounds[i - 1]
            end = bounds[i]
            r = rows[begin:end]
            w = weights[begin:end].to(torch.float32).unsqueeze(1)

            gate, up, down = self._views(slot, self._gscale(layer_idx, experts[i]))
            xs = x.index_select(0, r)
            h = F.silu(xs @ gate.t()) * (xs @ up.t())
            y = h @ down.t()
            out.index_add_(0, r, y.to(torch.float32) * w)

            self._compute_done[slot].record(compute_stream)

            nxt = i + self._slots
            if nxt < n:
                issue(nxt)

        self.experts_streamed += n
        self.bytes_streamed += n * self._block_bytes
        return out.to(x.dtype)

    # -- introspection ----------------------------------------------------

    @property
    def stats(self) -> dict[str, object]:
        return {
            "calls": self.calls,
            "experts_streamed": self.experts_streamed,
            "gib_streamed": self.bytes_streamed / 2**30,
            "slots": self._slots,
            "slot_bytes": 3 * self._plane * self._ring.element_size(),
            "arena_pinned": bool(getattr(self._host, "arena_pinned", False)),
        }


#: Process-wide, because the ring is reused across layers. Layers run
#: strictly in sequence, so one ring serves all of them and peak VRAM stays
#: ``slots * expert_bytes`` instead of scaling with layer count.
_streamer_singleton: ExpertStreamer | None = None


def get_streamer(
    host: object, hidden: int, intermediate: int, **kwargs: object
) -> ExpertStreamer:
    """Return the shared expert streamer, creating it on first use.

    Pins the arena here rather than at load time: pinning covers only the
    committed prefix, so it has to happen after the last expert is loaded.
    First streaming use is the earliest point where that is guaranteed.
    """
    global _streamer_singleton
    if _streamer_singleton is None:
        pinned = host.pin_arena()  # type: ignore[attr-defined]
        _streamer_singleton = ExpertStreamer(
            host, hidden, intermediate, **kwargs,  # type: ignore[arg-type]
        )
        logger.warning(
            "Shooting Brake: prefill weight streaming active — %d slots x "
            "%.1f MiB, arena %s. Host-resident experts above the tier's "
            "threshold are computed on the 5090 from streamed weights "
            "(cold tier above M=%d; B70 prefill has its own threshold).",
            _streamer_singleton._slots,
            _streamer_singleton._block_bytes / 2**20,
            "pinned (DMA)" if pinned else "unpinned (staged copies)",
            stream_threshold(),
        )
    return _streamer_singleton
