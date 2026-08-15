"""Prefill the B70-resident experts on the 5090 via Marlin, streaming int4.

Why this exists: the B70's MoE kernel is a decode kernel — per-route GEMV,
``O(M x top_k)`` weight reads. At prefill shapes it re-reads each expert
hundreds of times, and a profiled 8K prefill measured the 5090 idle 97% of a
27.7 s span waiting for it (`benchmarks/results/prefill_profile/`). The same
work through vLLM's ``fused_marlin_moe`` on the 5090 takes 6.6 ms per layer at
M=8192 (`benchmarks/results/marlin_poc/poc.json`).

What it does per layer, during prefill only:

  1. one contiguous H2D of the layer's raw SBINT401 slab (126 experts,
     584.7 MiB) from the mmap'd bank — page-cached host DRAM, measured
     18.5 GiB/s on this box; the B70 is not involved,
  2. reinterpret the slab in-place on GPU (strided views over the six planes),
  3. ``gptq_marlin_moe_repack`` + ``marlin_moe_permute_scales`` (~27 ms),
  4. ``fused_marlin_moe`` with the bank's quant contract, which is exactly
     Marlin's ``uint4b8``: AutoGPTQ sym int4, group 128, zero-point 8.

The next layer's slab H2D is prefetched on a side stream while the current
layer computes, double-buffered.

Consistency: decode keeps the B70 doorbell path untouched, and prefill uses
the SAME int4 weights the B70 serves — the served model is numerically
unchanged (kernel rounding aside). Streaming the checkpoint's NVFP4 experts
instead would make the same expert change quantization between prefill and
decode; rejected for that reason.

Pitfall recorded so nobody reintroduces it: Marlin dispatches on the
ACTIVATION dtype and requires scales in that dtype. fp16 scales with bf16
activations do not error — they return zeros.

Opt-in via ``SHOOTING_BRAKE_PREFILL_MARLIN=1``; default behaviour unchanged.
Correctness is gated black-box (CPU-oracle PoC + live prompt-logprob A/B);
an in-situ verify cannot live inside vLLM's compiled forward.
"""

from __future__ import annotations

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
    fused_marlin_moe,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_moe_permute_scales,
)
from vllm.scalar_type import scalar_types

from .int4_bank_format import read_int4_bank_header

logger = init_logger(__name__)


class MarlinPrefillStreamer:
    """Streams one layer of int4 experts at a time; never holds two repacked."""

    def __init__(
        self,
        bank_path: str,
        act_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
    ) -> None:
        self._hdr = read_int4_bank_header(bank_path)
        h = self._hdr
        self._mm = np.memmap(bank_path, dtype=np.uint8, mode="r")
        self.device = torch.device(device)
        self.act_dtype = act_dtype
        self._e = h.experts_per_layer
        self._k = h.hidden
        self._i = h.moe_intermediate
        self._num_layers = h.num_layers

        # ONE device slab, not two: vLLM's memory-profiling forward runs this
        # path and its peak must fit physical VRAM alongside weights + the
        # profiling activations -- a second 584.7 MiB buffer OOM'd the engine
        # at 31.25/31.36 GiB. With a single buffer, layer N+1's H2D still
        # overlaps layer N's fused kernel (copy waits only on the repack,
        # which is what reads the slab), just not the repack itself.
        self._slab_dev = torch.empty(
            h.layer_stride_bytes, dtype=torch.uint8, device=self.device
        )
        self._copy_stream = torch.cuda.Stream(device=self.device)
        self._copy_done = torch.cuda.Event()
        # The slab may not be overwritten while the compute stream's repack
        # still reads it: compute records "consumed", the copy stream waits.
        self._consumed = torch.cuda.Event()
        self._consumed.record()
        self._buffered_layer = -1

        # Plane geometry in int32/f16 units for GPU-side reinterpretation.
        kp, i, k = self._k // 8, self._i, self._k
        self._plane_geom = {
            "gate_qweight": (torch.int32, (kp, i)),
            "gate_scales": (torch.float16, (self._k // 128, i)),
            "up_qweight": (torch.int32, (kp, i)),
            "up_scales": (torch.float16, (self._k // 128, i)),
            "down_qweight": (torch.int32, (i // 8, k)),
            "down_scales": (torch.float16, (i // 128, k)),
        }
        # Persistent Marlin output buffers, allocated ONCE. Every repack
        # writes into slices of these; per-layer transient allocation is what
        # OOM'd the engine's memory-profiling forward three launches running.
        # Split-repacking gate and up separately into halves of the fused
        # buffer is BIT-EXACT to repacking the concatenation (verified on real
        # bank planes: qweights and permuted scales both torch.equal), because
        # Marlin tiles N in 64-column blocks and I=1024 is a multiple.
        e, k, i = self._e, self._k, self._i
        dev = self.device
        self._m13 = torch.empty((e, k // 16, 2 * i * 2), dtype=torch.int32, device=dev)
        self._m2 = torch.empty((e, i // 16, k * 2), dtype=torch.int32, device=dev)
        self._s13 = torch.empty((e, k // 128, 2 * i), dtype=act_dtype, device=dev)
        self._s2 = torch.empty((e, i // 128, k), dtype=act_dtype, device=dev)
        self._empty_idx = torch.empty((e, 0), dtype=torch.int32, device=dev)

        self.layers_streamed = 0
        self.bytes_streamed = 0
        self._verified = False

    # -- staging ------------------------------------------------------------

    def _slab_cpu(self, layer: int) -> torch.Tensor:
        h = self._hdr
        off = h.data_offset + layer * h.layer_stride_bytes
        view = np.asarray(self._mm[off: off + h.layer_stride_bytes])
        return torch.from_numpy(view)

    def prefetch(self, layer: int) -> None:
        """Start layer's slab H2D on the copy stream (idempotent).

        MUST only be called after the current occupant's repack has recorded
        ``_consumed``; the wait below is what keeps the copy from overwriting
        a slab the compute stream is still reading.
        """
        if not (0 <= layer < self._num_layers):
            return
        if self._buffered_layer == layer:
            return
        self._copy_stream.wait_event(self._consumed)
        with torch.cuda.stream(self._copy_stream):
            self._slab_dev.copy_(self._slab_cpu(layer), non_blocking=True)
        self._copy_done.record(self._copy_stream)
        self._buffered_layer = layer

    def _plane(self, slab: torch.Tensor, name: str, plane_off: int) -> torch.Tensor:
        """Strided [E, rows, cols] view of one plane across all experts."""
        dtype, (rows, cols) = self._plane_geom[name]
        item = torch.tensor([], dtype=dtype).element_size()
        base = slab.view(dtype)
        stride_e = self._hdr.expert_stride_bytes // item
        return torch.as_strided(
            base, (self._e, rows, cols), (stride_e, cols, 1),
            storage_offset=plane_off // item,
        )

    def _repack(self, slab: torch.Tensor):
        """Repack the slab's planes into the persistent Marlin buffers.

        No concatenation, no per-layer output allocation: each per-expert
        plane inside the slab is already contiguous, so it feeds
        ``_C.gptq_marlin_repack`` directly, and the result lands in a slice of
        the preallocated fused buffer. The only transients are the repack
        op's per-expert output (~3 MiB) and two 6 MiB scale copies.
        """
        h = self._hdr
        offs = dict(zip(
            ("gate_qweight", "gate_scales", "up_qweight",
             "up_scales", "down_qweight", "down_scales"),
            h.plane_offsets,
        ))
        gate_q = self._plane(slab, "gate_qweight", offs["gate_qweight"])
        up_q = self._plane(slab, "up_qweight", offs["up_qweight"])
        down_q = self._plane(slab, "down_qweight", offs["down_qweight"])
        gate_s = self._plane(slab, "gate_scales", offs["gate_scales"])
        up_s = self._plane(slab, "up_scales", offs["up_scales"])
        down_s = self._plane(slab, "down_scales", offs["down_scales"])

        k, i = self._k, self._i
        perm = self._empty_idx[0]
        half = 2 * i  # int32 columns per repacked [K, I] half
        for e in range(self._e):
            # SiluAndMul convention: first half gate, second half up.
            self._m13[e, :, :half] = torch.ops._C.gptq_marlin_repack(
                gate_q[e], perm, k, i, 4, False)
            self._m13[e, :, half:] = torch.ops._C.gptq_marlin_repack(
                up_q[e], perm, k, i, 4, False)
            self._m2[e] = torch.ops._C.gptq_marlin_repack(
                down_q[e], perm, i, k, 4, False)

        self._s13[:, :, :i] = marlin_moe_permute_scales(
            s=gate_s.to(self.act_dtype).contiguous(), size_k=k, size_n=i,
            group_size=h.group_size)
        self._s13[:, :, i:] = marlin_moe_permute_scales(
            s=up_s.to(self.act_dtype).contiguous(), size_k=k, size_n=i,
            group_size=h.group_size)
        self._s2.copy_(marlin_moe_permute_scales(
            s=down_s.to(self.act_dtype).contiguous(), size_k=i, size_n=k,
            group_size=h.group_size))
        return self._m13, self._m2, self._s13, self._s2

    # -- the partial ---------------------------------------------------------

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
        ``source_expert_ids`` in ascending order), so the ids index the
        repacked tensors directly. -1 routes are neutralised by zeroing their
        router weight and clamping the id — a zero weight contributes zero.
        """
        mask = b70_ids >= 0
        if not bool(mask.any()):
            return torch.zeros_like(x)

        if self._buffered_layer != layer_idx:
            self.prefetch(layer_idx)
        torch.cuda.current_stream().wait_event(self._copy_done)

        m13, m2, s13, s2 = self._repack(self._slab_dev)
        self._consumed.record()  # repack outputs are contiguous copies
        self._buffered_layer = -1
        # Next layer's H2D starts once the repack has drained (the copy
        # stream waits on _consumed) and overlaps this layer's fused kernel.
        self.prefetch(layer_idx + 1)

        weights = torch.where(
            mask, topk_weights, torch.zeros_like(topk_weights)
        ).to(torch.float32)
        ids = b70_ids.clamp_min(0).to(torch.int32)

        y = fused_marlin_moe(
            x.contiguous(), m13, m2, None, None, s13, s2,
            weights, ids, quant_type_id=scalar_types.uint4b8.id,
            global_num_experts=self._e,
        )
        self.layers_streamed += 1
        self.bytes_streamed += self._hdr.layer_stride_bytes
        return y.to(x.dtype)

    @property
    def stats(self) -> dict[str, object]:
        return {
            "layers_streamed": self.layers_streamed,
            "gib_streamed": self.bytes_streamed / 2**30,
        }


# -- custom op boundary ------------------------------------------------------
#
# The streamer does mmap reads, cross-stream H2D, CUDA events, and in-place
# reuse of persistent buffers. Inside vLLM's compiled forward, dynamo/AOT
# functionalization traces through all of it and materializes copies of every
# tensor touched -- the memory-profiling forward then peaks GiBs higher than
# eager and the engine OOMs at boot (six launches). @torch.compiler.disable
# did not break out of vLLM's compile pipeline; the idiomatic boundary -- the
# one vLLM itself uses for every complex kernel -- is a registered custom op:
# the compiler sees one opaque node with a shape rule and never looks inside.

_STREAMER: MarlinPrefillStreamer | None = None


def _get_streamer(bank_path: str, act_dtype: torch.dtype) -> MarlinPrefillStreamer:
    global _STREAMER
    if _STREAMER is None:
        _STREAMER = MarlinPrefillStreamer(bank_path, act_dtype=act_dtype)
        logger.info(
            "Shooting Brake Marlin prefill: streaming %d experts/layer from "
            "the int4 bank to CUDA (opt-in, decode unaffected)",
            _STREAMER._e,
        )
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
