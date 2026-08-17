"""Prefill the B70-resident experts on the 5090 via FlashInfer B12x.

Bank v2 (SBB12X01) drop-in successor to the Marlin prefill streamer:
native-FP4 tensor cores via ``flashinfer.fused_moe.B12xMoEWrapper``
(measured 2.70 ms/layer @ M=8192 vs Marlin's 7.07 --
benchmarks/results/run5_88b_register/floor_b12x.json), weights sourced
from the checkpoint's own NVFP4 planes (no GPTQ re-encode), scales
pre-baked and pre-swizzled at build time so the runtime is a pure
memcpy into a ring of device arenas.

Opt-in via ``SHOOTING_BRAKE_PREFILL_B12X=1`` (requires
``SHOOTING_BRAKE_PREFILL_MARLIN=1`` to route prefill through the custom
op at all; the flag selects WHICH streamer serves it).
``SHOOTING_BRAKE_B12X_BANK`` overrides the bank path (default: int4 bank
path + ``.b12x``). ``SHOOTING_BRAKE_BANK_REGISTER=1`` pins the bank page
cache (see bank_source.py). ``SHOOTING_BRAKE_PREFILL_TILE`` caps
per-call token rows (b12x quantizes activations in-kernel; W4A4 math --
numerics differ from Marlin's W4A16 and MUST pass the firsttok envelope
gate before serving).
"""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger

from .b12x_bank_format import (
    default_b12x_bank_path,
    read_b12x_bank_header,
)
from .bank_source import BankSource

logger = init_logger(__name__)

TOP_K = 8


class B12xPrefillStreamer:
    """Ring-buffered arena loader over the pre-swizzled B12x bank."""

    def __init__(
        self,
        int4_bank_path: str,
        act_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
    ) -> None:
        path = default_b12x_bank_path(int4_bank_path)
        if not os.path.exists(path):
            raise RuntimeError(
                f"B12x bank not found at {path}. Build it once:\n"
                f"  .venv/bin/python src/phase1/build_b12x_bank.py "
                f"--out {path} --validate-layers 2\n"
                f"(or point SHOOTING_BRAKE_B12X_BANK at an existing bank)"
            )
        if act_dtype != torch.bfloat16:
            raise RuntimeError(
                f"B12xMoEWrapper takes bf16 hidden states; activations are "
                f"{act_dtype}"
            )
        h = self._hdr = read_b12x_bank_header(path)
        self.device = torch.device(device)
        self._e = h.experts_per_layer
        self._num_layers = h.num_layers
        self._tile = max(1, int(os.environ.get("SHOOTING_BRAKE_PREFILL_TILE",
                                               "8192")))

        data_len = h.num_layers * h.layer_stride_bytes
        register = os.environ.get("SHOOTING_BRAKE_BANK_REGISTER", "0") == "1"
        self._source = BankSource(path, h.data_offset, data_len, register)
        self._base_t = self._source.tensor

        n_slots = max(1, int(os.environ.get("SHOOTING_BRAKE_B12X_ARENAS", "2")))
        offs, sizes = h.plane_offsets, h.plane_sizes
        e, k, i = self._e, h.hidden, h.moe_intermediate

        def views(arena: torch.Tensor) -> dict[str, torch.Tensor]:
            def cut(idx: int, dtype: torch.dtype, shape: tuple[int, ...]):
                return (arena[offs[idx]: offs[idx] + sizes[idx]]
                        .view(dtype).view(*shape))

            return {
                "w1": cut(0, torch.uint8, (e, 2 * i, k // 2)),
                "w2": cut(1, torch.uint8, (e, k, i // 2)),
                "sf1": cut(2, torch.float8_e4m3fn, h.sf1_shape),
                "sf2": cut(3, torch.float8_e4m3fn, h.sf2_shape),
                "alpha1": cut(4, torch.float32, (e,)),
                "alpha2": cut(5, torch.float32, (e,)),
            }

        self._arenas = [
            torch.empty(h.layer_stride_bytes, dtype=torch.uint8,
                        device=self.device)
            for _ in range(n_slots)
        ]
        self._views = [views(a) for a in self._arenas]
        self._slot_layer = [-1] * n_slots
        self._copy_stream = torch.cuda.Stream(device=self.device)
        self._copy_done = [torch.cuda.Event() for _ in range(n_slots)]
        self._consumed = [torch.cuda.Event() for _ in range(n_slots)]
        for ev in self._consumed:
            ev.record()

        # input_global_scale MUST be explicit (flashinfer >= 0.6.18):
        # otherwise w1_alpha (the exact ~1e-5 ModelOpt weight scale carried
        # in the bank's alpha plane) doubles as the FC1 activation-quant
        # scale and zeroes the activations. fc2 input quant is dynamic
        # per-block; uniform 1.0 per expert.
        self._input_gs = torch.ones(e, device=self.device, dtype=torch.float32)
        self._fc2_scale = torch.ones(e, device=self.device,
                                     dtype=torch.float32)
        from flashinfer.fused_moe import B12xMoEWrapper
        self._wrapper = B12xMoEWrapper(
            num_experts=e, top_k=TOP_K, hidden_size=k, intermediate_size=i,
            use_cuda_graph=True, max_num_tokens=self._tile,
            num_local_experts=e, activation="silu",
        )

        self.layers_streamed = 0
        self.bytes_streamed = 0
        logger.info(
            "B12x bank %s: %d experts/layer x %d layers, %d-arena ring "
            "(%.1f MiB each), tile=%d",
            path, e, h.num_layers, n_slots,
            h.layer_stride_bytes / 2**20, self._tile,
        )

    # -- staging --------------------------------------------------------------

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

    def _run_tile(self, v, x, ids, weights) -> torch.Tensor:
        return self._wrapper.run(
            x=x, w1_weight=v["w1"], w1_weight_sf=v["sf1"],
            w1_alpha=v["alpha1"], w2_alpha=v["alpha2"],
            input_global_scale=self._input_gs,
            fc2_input_scale=self._fc2_scale,
            w2_weight=v["w2"], w2_weight_sf=v["sf2"],
            token_selected_experts=ids, token_final_scales=weights,
        )

    def partial(
        self,
        layer_idx: int,
        x: torch.Tensor,
        b70_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Routing-weighted partial for this layer's B70-owned routes.

        Same contract as MarlinPrefillStreamer.partial: ``b70_ids`` are
        compact bank slots with -1 for routes owned elsewhere; -1 routes
        are neutralised by zeroing their router weight and clamping the
        id (b12x applies topk weights internally, so a zero weight
        contributes zero).
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

        m, tile = x.shape[0], self._tile
        if m <= tile:
            y = self._run_tile(v, x.contiguous(), ids, weights).to(x.dtype)
        else:
            y = torch.empty_like(x)
            for lo in range(0, m, tile):
                hi = min(lo + tile, m)
                y[lo:hi] = self._run_tile(
                    v, x[lo:hi].contiguous(), ids[lo:hi], weights[lo:hi]
                ).to(x.dtype)

        self._consumed[slot].record()
        self._slot_layer[slot] = -1
        self.prefetch(layer_idx + 1)

        self.layers_streamed += 1
        self.bytes_streamed += self._hdr.layer_stride_bytes
        return y

    @property
    def stats(self) -> dict[str, object]:
        return {
            "layers_streamed": self.layers_streamed,
            "gib_streamed": self.bytes_streamed / 2**30,
        }
