"""Prefill the B70-resident experts on the 5090 via Marlin, streaming int4.

Why this exists: the B70's MoE kernel is a decode kernel - per-route GEMV,
``O(M x top_k)`` weight reads. At prefill shapes it re-reads each expert
hundreds of times, and a profiled 8K prefill measured the 5090 idle 97%
waiting on it. This module streams each layer's remote experts from the
pre-repacked Marlin bank (host DRAM, page-cached) to the 5090 and runs
vLLM's grouped ``fused_marlin_moe`` there - the same grouped path the 54
local experts already use. Decode is untouched: the B70 doorbell keeps
serving the regime it wins (dispatch beats streaming ~9x at M=1).

The bank (``SBMARL01``, built by ``src/phase1/build_marlin_bank.py``) holds
per layer one contiguous ``[m13|m2|s13|s2]`` arena - the bit-exact output of
the repack this module used to do at runtime (validated bit-exact at build).
The runtime is therefore a single H2D memcpy per layer into a ring of device
arenas whose Marlin tensors are views; layer N+1's copy overlaps layer N's
fused kernel. No repack, no per-layer transients, nothing to trace.

Opt-in via ``SHOOTING_BRAKE_PREFILL_MARLIN=1``; default behaviour unchanged.
``SHOOTING_BRAKE_MARLIN_BANK`` overrides the bank path (default: the int4
bank path + ``.marlin``). ``SHOOTING_BRAKE_MARLIN_ARENAS`` sets the ring
size (default 2; 1 saves 584.7 MiB of VRAM at the cost of H2D/kernel
overlap). ``SHOOTING_BRAKE_BANK_REGISTER=1`` pins the bank's mmap'd page
cache with ``cudaHostRegister`` so the per-layer H2D runs as true async DMA
at the measured 53.9 GiB/s ceiling instead of the 18.5 GiB/s pageable path
(probe: benchmarks/results/slab_h2d/hostregister_probe.json). Correctness
is gated black-box (CPU-oracle PoC + live prompt-logprob A/B); an in-situ
verify cannot live inside vLLM's compiled forward.
"""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
    fused_marlin_moe,
)
from vllm.scalar_type import scalar_types

from .marlin_bank_format import (
    SCALES_DTYPE_BF16,
    default_marlin_bank_path,
    read_marlin_bank_header,
)

logger = init_logger(__name__)


class MarlinPrefillStreamer:
    """Ring-buffered arena loader over the pre-repacked Marlin bank."""

    def __init__(
        self,
        int4_bank_path: str,
        act_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
    ) -> None:
        path = default_marlin_bank_path(int4_bank_path)
        if not os.path.exists(path):
            raise RuntimeError(
                f"Marlin bank not found at {path}. Build it once:\n"
                f"  .venv/bin/python src/phase1/build_marlin_bank.py "
                f"--int4-bank {int4_bank_path}\n"
                f"(or point SHOOTING_BRAKE_MARLIN_BANK at an existing bank)"
            )
        self._hdr = read_marlin_bank_header(path)
        h = self._hdr
        want = (
            torch.bfloat16 if h.scales_dtype == SCALES_DTYPE_BF16
            else torch.float16
        )
        if act_dtype != want:
            raise RuntimeError(
                f"Marlin bank {path} stores {want} scales but activations are "
                f"{act_dtype}; Marlin silently returns zeros on dtype "
                f"mismatch. Rebuild with build_marlin_bank.py "
                f"--scales-dtype {'bf16' if act_dtype == torch.bfloat16 else 'fp16'}"
            )
        self._base_t = self._open_bank_source(path, h)
        self.device = torch.device(device)
        self.act_dtype = act_dtype
        self._e = h.experts_per_layer
        self._num_layers = h.num_layers

        n_slots = max(1, int(os.environ.get("SHOOTING_BRAKE_MARLIN_ARENAS", "2")))
        e, k, i = self._e, h.hidden, h.moe_intermediate
        offs, sizes = h.plane_offsets, h.plane_sizes

        def views(arena: torch.Tensor) -> dict[str, torch.Tensor]:
            def cut(idx: int, dtype: torch.dtype, shape: tuple[int, ...]):
                return arena[offs[idx]: offs[idx] + sizes[idx]].view(dtype).view(*shape)

            return {
                "m13": cut(0, torch.int32, (e, k // 16, 4 * i)),
                "m2": cut(1, torch.int32, (e, i // 16, 2 * k)),
                "s13": cut(2, act_dtype, (e, k // 128, 2 * i)),
                "s2": cut(3, act_dtype, (e, i // 128, k)),
            }

        self._arenas = [
            torch.empty(h.layer_stride_bytes, dtype=torch.uint8, device=self.device)
            for _ in range(n_slots)
        ]
        self._views = [views(a) for a in self._arenas]
        self._slot_layer = [-1] * n_slots
        self._copy_stream = torch.cuda.Stream(device=self.device)
        self._copy_done = [torch.cuda.Event() for _ in range(n_slots)]
        # A slot may not be overwritten while the compute stream's fused
        # kernel still reads it: compute records "consumed", the copy waits.
        self._consumed = [torch.cuda.Event() for _ in range(n_slots)]
        for ev in self._consumed:
            ev.record()

        self.layers_streamed = 0
        self.bytes_streamed = 0
        logger.info(
            "Marlin bank %s: %d experts/layer x %d layers, %d-arena ring "
            "(%.1f MiB each)",
            path, e, h.num_layers, n_slots,
            h.layer_stride_bytes / 2**20,
        )

    # -- staging --------------------------------------------------------------

    def _open_bank_source(self, path: str, h) -> torch.Tensor:
        """uint8 view over the bank's plane data, pageable or pinned.

        Delegates to the shared BankSource (bank_source.py) -- pageable
        np.memmap by default, cudaHostRegister'd page cache under
        SHOOTING_BRAKE_BANK_REGISTER=1 (53.9 GiB/s measured,
        benchmarks/results/slab_h2d/hostregister_probe.json). Constructed
        lazily in the worker, post-fork; hard-fails on register error.
        """
        from .bank_source import BankSource
        data_len = h.num_layers * h.layer_stride_bytes
        register = os.environ.get("SHOOTING_BRAKE_BANK_REGISTER", "0") == "1"
        self._source = BankSource(path, h.data_offset, data_len, register)
        return self._source.tensor

    def _layer_cpu(self, layer: int) -> torch.Tensor:
        stride = self._hdr.layer_stride_bytes
        return self._base_t[layer * stride: (layer + 1) * stride]

    def prefetch(self, layer: int) -> None:
        """Start layer's arena H2D on the copy stream (idempotent)."""
        if not (0 <= layer < self._num_layers):
            return
        slot = layer % len(self._arenas)
        if self._slot_layer[slot] == layer:
            return
        self._copy_stream.wait_event(self._consumed[slot])
        with torch.cuda.stream(self._copy_stream):
            self._arenas[slot].copy_(self._layer_cpu(layer), non_blocking=True)
        self._copy_done[slot].record(self._copy_stream)
        self._slot_layer[slot] = layer

    # -- the partial -----------------------------------------------------------

    def partial(
        self,
        layer_idx: int,
        x: torch.Tensor,
        b70_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Routing-weighted partial for this layer's B70-owned routes.

        ``b70_ids`` are compact bank slots with -1 for routes owned elsewhere.
        Bank slot order equals provider resident order (both are the header's
        ``source_expert_ids`` in ascending order), so the ids index the bank
        tensors directly. -1 routes are neutralised by zeroing their router
        weight and clamping the id - a zero weight contributes zero.
        """
        mask = b70_ids >= 0
        if not bool(mask.any()):
            return torch.zeros_like(x)

        slot = layer_idx % len(self._arenas)
        self.prefetch(layer_idx)
        torch.cuda.current_stream().wait_event(self._copy_done[slot])
        v = self._views[slot]

        weights = torch.where(
            mask, topk_weights, torch.zeros_like(topk_weights)
        ).to(torch.float32)
        ids = b70_ids.clamp_min(0).to(torch.int32)

        # M-tiling: _fused_marlin_moe with None caches torch.empty's
        # ~2*M*top_k*max(2N,K) bytes of GEMM scratch per call -- 1.6+ GiB at
        # M=32768. Tiling token rows caps scratch at tile size while the
        # arena (the expensive part: 584.7 MiB streamed once per layer per
        # step) is reused by every tile. Numerically identical: the kernel
        # is per-token independent (no atomics, ordered fp32 reduction,
        # per-row top-k weighted sum). This is what makes MNBT=32K+
        # stream-once prefill affordable: one step = one bank pass.
        tile = max(1, int(os.environ.get("SHOOTING_BRAKE_PREFILL_TILE", "8192")))
        m = x.shape[0]
        if m <= tile:
            y = fused_marlin_moe(
                x.contiguous(), v["m13"], v["m2"], None, None, v["s13"], v["s2"],
                weights, ids, quant_type_id=scalar_types.uint4b8.id,
                global_num_experts=self._e,
            )
        else:
            y = torch.empty_like(x)
            for lo in range(0, m, tile):
                hi = min(lo + tile, m)
                y[lo:hi] = fused_marlin_moe(
                    x[lo:hi].contiguous(), v["m13"], v["m2"], None, None,
                    v["s13"], v["s2"], weights[lo:hi], ids[lo:hi],
                    quant_type_id=scalar_types.uint4b8.id,
                    global_num_experts=self._e,
                ).to(x.dtype)
        # The kernel is enqueued; the event lands after it in stream order,
        # releasing this slot for the copy stream. Then start the next
        # layer's H2D so it overlaps this layer's kernel + everything else.
        self._consumed[slot].record()
        self._slot_layer[slot] = -1
        self.prefetch(layer_idx + 1)

        self.layers_streamed += 1
        self.bytes_streamed += self._hdr.layer_stride_bytes
        return y.to(x.dtype)

    @property
    def stats(self) -> dict[str, object]:
        return {
            "layers_streamed": self.layers_streamed,
            "gib_streamed": self.bytes_streamed / 2**30,
        }


# -- custom op boundary --------------------------------------------------------
#
# The streamer does mmap reads, cross-stream H2D, and CUDA events. Inside
# vLLM's compiled forward, dynamo/AOT functionalization traces through all of
# it and materializes copies of every tensor touched -- the memory-profiling
# forward then peaks GiBs higher than eager and the engine OOMs at boot (six
# launches). @torch.compiler.disable did not break out of vLLM's compile
# pipeline; the idiomatic boundary -- the one vLLM itself uses for every
# complex kernel -- is a registered custom op: the compiler sees one opaque
# node with a shape rule and never looks inside.

_STREAMER = None  # MarlinPrefillStreamer | B12xPrefillStreamer


def _get_streamer(bank_path: str, act_dtype: torch.dtype):
    """Select the prefill streamer once per worker.

    SHOOTING_BRAKE_PREFILL_B12X=1 serves the custom op with the B12x
    (bank v2, native-FP4 tensor core) streamer instead of Marlin. Same
    partial() contract, same ring/overlap semantics; W4A4 numerics --
    gated by the firsttok envelope before any serving claim.
    """
    global _STREAMER
    if _STREAMER is None:
        if os.environ.get("SHOOTING_BRAKE_PREFILL_B12X", "0") == "1":
            from .b12x_prefill import B12xPrefillStreamer
            _STREAMER = B12xPrefillStreamer(bank_path, act_dtype=act_dtype)
        else:
            _STREAMER = MarlinPrefillStreamer(bank_path, act_dtype=act_dtype)
    return _STREAMER


def _marlin_prefill_partial_impl(
    x: torch.Tensor,
    b70_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    layer_idx: int,
    bank_path: str,
) -> torch.Tensor:
    return _get_streamer(bank_path, x.dtype).partial(
        layer_idx, x, b70_ids, topk_weights,
    )


def _marlin_prefill_partial_fake(
    x: torch.Tensor,
    b70_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    layer_idx: int,
    bank_path: str,
) -> torch.Tensor:
    return torch.empty_like(x)


from vllm.utils.torch_utils import direct_register_custom_op  # noqa: E402

direct_register_custom_op(
    op_name="shooting_brake_marlin_prefill",
    op_func=_marlin_prefill_partial_impl,
    mutates_args=[],
    fake_impl=_marlin_prefill_partial_fake,
)


def marlin_prefill_partial(
    x: torch.Tensor,
    b70_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    layer_idx: int,
    bank_path: str,
) -> torch.Tensor:
    """Compiled-graph-safe entry point."""
    return torch.ops.vllm.shooting_brake_marlin_prefill(
        x, b70_ids, topk_weights, layer_idx, bank_path,
    )
