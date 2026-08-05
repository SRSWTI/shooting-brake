"""Qualified routed-expert container with Phase-5/6a ownership + partitioning."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.logger import init_logger

from .config import require_qualified_config
from .partition import (
    RoutePartition,
    build_device_map,
    partition_routes,
    validate_partition,
)
from .placement import Device, Placement, build_for_qualified
from .provider import ShootingBrakeExpertProviderClient

logger = init_logger(__name__)


def _placement_policy_name() -> str:
    """Placement policy spec from the environment (default: all-CUDA)."""
    return os.environ.get("SHOOTING_BRAKE_PLACEMENT", "all-cuda")

def _build_b70_slot_map(placement: Placement) -> np.ndarray:
    """[num_experts] int32: B70 compact slot for each global expert, -1 for CUDA.

    All B70-capable layers share the same resident set (guaranteed by the
    SplitPolicy / InterleavedPolicy placement families), so layer 0 is
    representative.
    """
    slot_map = np.full(placement.num_experts, -1, dtype=np.int32)
    for expert_id, owner in enumerate(placement.owners[0]):
        if owner.device is Device.B70:
            slot_map[expert_id] = owner.slot
    return slot_map


_b70_provider_singleton: Any = None


def _get_b70_provider(placement: Placement) -> Any:
    """Lazily create and cache the in-process B70 provider singleton."""
    global _b70_provider_singleton
    if _b70_provider_singleton is not None:
        return _b70_provider_singleton

    from .b70_binding import B70ProviderClient

    lib_path = os.environ.get(
        "SHOOTING_BRAKE_B70_LIB", "phase7/libsb_b70_provider.so"
    )
    bank_path = os.environ.get(
        "SHOOTING_BRAKE_B70_BANK", "phase1/expert_bank.bin"
    )
    resident = sorted(
        e for e in range(placement.num_experts)
        if placement.owners[0][e].device is Device.B70
    )
    resident_np = np.array(resident, dtype=np.int32)
    max_batch = int(os.environ.get("SHOOTING_BRAKE_B70_MAX_BATCH", "128"))

    logger.info(
        "Shooting Brake Phase-7: initializing B70 provider "
        "(%d resident experts/layer)...", len(resident)
    )
    provider = B70ProviderClient(lib_path)
    provider.load(bank_path, generation=1, resident_experts=resident_np,
                  max_batch=max_batch)
    _b70_provider_singleton = provider
    logger.info(
        "Shooting Brake Phase-7: B70 provider ready "
        "(resident_per_layer=%d)", provider.resident_per_layer
    )
    return provider


class HybridRoutedExperts(RoutedExperts):
    """Stock routed experts plus compact ownership and route partitioning."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        qualified_model = require_qualified_config(get_current_vllm_config())
        super().__init__(*args, **kwargs)
        self.shooting_brake_qualified_model = qualified_model
        self.shooting_brake_placement: Placement = build_for_qualified(
            qualified_model,
            policy_name=_placement_policy_name(),
        )
        self._device_map_cpu = build_device_map(self.shooting_brake_placement)
        self._b70_slot_map = _build_b70_slot_map(self.shooting_brake_placement)
        # Phase 8a: pinned host buffers for async B70 overlap.
        # Pre-allocated once and reused every step to avoid per-step
        # allocation on the hot path.  Pinned memory enables DMA D2H/H2D.
        self._b70_max_batch = int(
            os.environ.get("SHOOTING_BRAKE_B70_MAX_BATCH", "128")
        )
        self._b70_pinned_hidden: torch.Tensor = torch.empty(
            self._b70_max_batch, 2048, dtype=torch.float16,
            pin_memory=True, device="cpu",
        )
        self._b70_pinned_output: torch.Tensor = torch.empty(
            self._b70_max_batch, 2048, dtype=torch.float32,
            pin_memory=True, device="cpu",
        )
        # Phase 8.5: VRAM surgery state. When enabled, B70-owned expert
        # weights are removed from CUDA VRAM on the first forward call.
        self._vram_surgery_done = False
        self._cuda_remap: torch.Tensor | None = None
        self._layer_idx: int | None = None
        self._device_map_layer: torch.Tensor | None = None
        # Phase-6a partition statistics (per-layer, in-process).
        self.shooting_brake_partition_stats: dict[str, int] = {
            "steps": 0,
            "remote_steps": 0,
            "remote_routes": 0,
            "max_remote_per_step": 0,
        }
        self._shadow_done = False
        # Cache env-var decisions at init time so forward_modular has
        # zero Python branching on the hot path in all-CUDA mode — this
        # makes it a pure pass-through compatible with CUDA graph capture.
        self._hybrid_active = (
            os.environ.get("SHOOTING_BRAKE_HYBRID") == "1"
            and self.shooting_brake_placement.b70_count() > 0
        )
        self._all_cuda_passthrough = not (
            self._hybrid_active
            or os.environ.get("SHOOTING_BRAKE_SHADOW") == "1"
            or os.environ.get("SHOOTING_BRAKE_VRAM_SURGERY") == "1"
        )
        self.shooting_brake_provider = ShootingBrakeExpertProviderClient(
            qualified_model=qualified_model,
            layer_name=self.layer_name,
            placement=self.shooting_brake_placement,
        )
        # Phase-6a diagnostic: verify env vars + code path in EngineCore.
        _init_marker = os.environ.get("SHOOTING_BRAKE_PARTITION_MARKER")
        if _init_marker:
            import json as _json
            with open(_init_marker, "a") as _f:
                _f.write(_json.dumps({
                    "init": True,
                    "layer_name": self.layer_name,
                    "policy": _placement_policy_name(),
                    "b70_count": self.shooting_brake_placement.b70_count(),
                    "is_monolithic": self.quant_method.is_monolithic,
                }) + "\n")

    def _ensure_layer_device_map(
        self, topk_ids: torch.Tensor
    ) -> tuple[int, torch.Tensor]:
        """Lazily resolve this layer's index and cache its device-map row."""
        if self._layer_idx is None:
            from vllm.model_executor.models.utils import extract_layer_index

            self._layer_idx = extract_layer_index(self.layer_name)
        if (
            self._device_map_layer is None
            or self._device_map_layer.device != topk_ids.device
        ):
            self._device_map_layer = self._device_map_cpu[
                self._layer_idx
            ].to(topk_ids.device)
        return self._layer_idx, self._device_map_layer

    def _maybe_perform_vram_surgery(self, layer_idx: int) -> None:
        """Lazily remove B70-owned expert weights from CUDA VRAM.

        Runs once per layer on the first ``forward_modular`` call, after
        vLLM's ``process_weights_after_loading`` has set up all tensors.
        Slices every per-expert weight and scale tensor to contain only
        CUDA-owned experts, then updates the quant config's internal
        references so the kernel's ``@property`` delegates see the smaller
        tensors.

        A global→local remap tensor is built so the CUDA kernel receives
        compact local IDs (B70 routes map to expert 0 with zero weight,
        producing zero contribution).  The real B70 results are added
        separately in the hybrid path.

        Requires ``SHOOTING_BRAKE_VRAM_SURGERY=1``.  Implicitly requires
        ``B70_DEVICE=1`` and ``HYBRID=1`` because B70 experts are removed
        from CUDA and must be computed on the B70 device.
        """
        if self._vram_surgery_done:
            return
        self._vram_surgery_done = True  # set early to prevent re-entry

        if os.environ.get("SHOOTING_BRAKE_VRAM_SURGERY") != "1":
            return
        if os.environ.get("SHOOTING_BRAKE_B70_DEVICE") != "1":
            logger.warning(
                "SHOOTING_BRAKE_VRAM_SURGERY=1 requires "
                "SHOOTING_BRAKE_B70_DEVICE=1; skipping surgery."
            )
            return

        placement = self.shooting_brake_placement

        # Skip layers with no B70 experts (e.g. FP8 layers 32-39).
        b70_count = sum(
            1 for owner in placement.owners[layer_idx]
            if owner.device is Device.B70
        )
        if b70_count == 0:
            return

        # Determine CUDA-owned expert IDs (sorted ascending for contiguous
        # index_select — even for interleaved placement this produces the
        # correct compact-to-global mapping).
        cuda_global_ids = sorted(
            e for e in range(placement.num_experts)
            if placement.owners[layer_idx][e].device is Device.CUDA
        )
        device = self.w13_weight.device
        cuda_idx = torch.tensor(
            cuda_global_ids, device=device, dtype=torch.long,
        )
        num_cuda = len(cuda_global_ids)

        logger.info(
            "Shooting Brake VRAM surgery: layer %d — slicing %d→%d "
            "CUDA experts (freeing %d B70 experts)",
            layer_idx, placement.num_experts, num_cuda, b70_count,
        )

        # --- Slice weight parameters (accessed by forward_modular) ---
        self.w13_weight = torch.nn.Parameter(
            self.w13_weight.data.index_select(0, cuda_idx).clone(),
            requires_grad=False,
        )
        self.w2_weight = torch.nn.Parameter(
            self.w2_weight.data.index_select(0, cuda_idx).clone(),
            requires_grad=False,
        )

        # --- Slice per-block scale parameters ---
        self.w13_weight_scale = torch.nn.Parameter(
            self.w13_weight_scale.data.index_select(0, cuda_idx).clone(),
            requires_grad=False,
        )
        self.w2_weight_scale = torch.nn.Parameter(
            self.w2_weight_scale.data.index_select(0, cuda_idx).clone(),
            requires_grad=False,
        )

        # --- Update quant config internal references ---
        # All scale properties on FusedMoEExpertsModular delegate to
        # quant_config._w1 / _w2 / _a1 / _a2 FusedMoEQuantDesc fields.
        qconfig = self.quant_method.moe_quant_config

        # w1_scale / w2_scale — point to the freshly sliced parameters.
        qconfig._w1.scale = self.w13_weight_scale.data
        qconfig._w2.scale = self.w2_weight_scale.data

        # g1_alphas / g2_alphas (= w13/w2_weight_scale_2 after processing).
        qconfig._w1.alpha_or_gscale = (
            qconfig._w1.alpha_or_gscale.index_select(0, cuda_idx).clone()
        )
        qconfig._w2.alpha_or_gscale = (
            qconfig._w2.alpha_or_gscale.index_select(0, cuda_idx).clone()
        )

        # a1_gscale / a2_gscale (derived from input scales).
        qconfig._a1.alpha_or_gscale = (
            qconfig._a1.alpha_or_gscale.index_select(0, cuda_idx).clone()
        )
        qconfig._a2.alpha_or_gscale = (
            qconfig._a2.alpha_or_gscale.index_select(0, cuda_idx).clone()
        )

        # --- Sync layer attributes to match quant config ---
        self.w13_weight_scale_2 = qconfig._w1.alpha_or_gscale
        self.w2_weight_scale_2 = qconfig._w2.alpha_or_gscale
        self.w13_input_scale = (
            1.0 / qconfig._a1.alpha_or_gscale
        ).clone()
        self.w2_input_scale = (
            1.0 / qconfig._a2.alpha_or_gscale
        ).clone()

        # --- Update metadata ---
        self.local_num_experts = num_cuda

        # --- Update expert object's per-expert attributes ---
        expert_obj = self.quant_method.moe_kernel.fused_experts
        expert_obj.num_experts = num_cuda
        if (
            hasattr(expert_obj, "gemm1_clamp_limit")
            and expert_obj.gemm1_clamp_limit is not None
        ):
            expert_obj.gemm1_clamp_limit = (
                expert_obj.gemm1_clamp_limit[:num_cuda].clone()
            )

        # --- Build global→local remap for forward_modular ---
        # CUDA experts map to their compact local ID (0..num_cuda-1).
        # B70 experts map to 0 (dummy — their routing weight is zeroed,
        # so 0 * E_0(x) = 0 contribution, and the real result comes
        # from the B70 device).
        self._cuda_remap = torch.zeros(
            placement.num_experts, dtype=torch.long, device=device,
        )
        for local_id, global_id in enumerate(cuda_global_ids):
            self._cuda_remap[global_id] = local_id

        # --- Reclaim freed VRAM ---
        torch.cuda.empty_cache()

        # --- Write VRAM marker for integration test ---
        vram_marker = os.environ.get("SHOOTING_BRAKE_VRAM_MARKER")
        if vram_marker:
            vram_now = torch.cuda.memory_allocated() / (1024**3)
            with open(vram_marker, "a") as vf:
                vf.write(f"{layer_idx} {vram_now:.6f}\n")

        logger.info(
            "Shooting Brake VRAM surgery: layer %d done — %d CUDA experts "
            "on device, remap ready (B70→dummy 0)",
            layer_idx, num_cuda,
        )

    def forward_modular(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: Any = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Partition routes via the manifest, validate, then run stock CUDA.

        In all-CUDA mode (no B70 device, no surgery, no shadow), this is
        a pure pass-through to ``super().forward_modular()`` with zero
        Python logic on the hot path — making it compatible with CUDA
        graph capture and replay.
        """
        # Graph-compatible fast path: when no hybrid features are active,
        # skip ALL Python logic and call super() directly. This branch is
        # taken during CUDA graph capture and the super() call's kernel
        # launches are recorded into the graph. During replay, only the
        # recorded kernels execute — this Python line does not re-run.
        if self._all_cuda_passthrough:
            return super().forward_modular(
                x, topk_weights, topk_ids,
                shared_experts, shared_experts_input,
            )
        layer_idx, dml = self._ensure_layer_device_map(topk_ids)
        self._maybe_perform_vram_surgery(layer_idx)
        part = partition_routes(topk_ids, topk_weights, dml, layer_idx)
        validate_partition(
            part, self.shooting_brake_placement.b70_capable_layers
        )
        # After VRAM surgery, CUDA topk_ids must be remapped to compact
        # local IDs.  B70 issue/take still use the original global IDs.
        cuda_topk_ids = (
            self._cuda_remap[topk_ids]
            if self._cuda_remap is not None
            else topk_ids
        )

        stats = self.shooting_brake_partition_stats
        stats["steps"] += 1
        has_remote = part.has_remote()
        if has_remote:
            stats["remote_steps"] += 1
            n_remote = part.num_b70_routes()
            stats["remote_routes"] += n_remote
            stats["max_remote_per_step"] = max(
                stats["max_remote_per_step"], n_remote
            )
            if stats["remote_steps"] == 1:
                logger.info(
                    "Shooting Brake Phase-6a: layer %s first remote step "
                    "(cuda=%d b70=%d routes, M=%d).",
                    layer_idx,
                    part.num_cuda_routes(),
                    n_remote,
                    topk_ids.shape[0],
                )
                # Deterministic cross-process marker for the 6a gate.
                marker = os.environ.get("SHOOTING_BRAKE_PARTITION_MARKER")
                if marker:
                    import json
                    with open(marker, "a") as f:
                        f.write(json.dumps({
                            "layer": layer_idx,
                            "b70_routes": n_remote,
                            "M": topk_ids.shape[0],
                        }) + "\n")
        # Phase 6b: shadow validation of the split-merge math (once).
        # Skip when VRAM surgery is active — B70 experts are no longer in
        # the CUDA weight tensor, so the three-way comparison is invalid.
        if (
            not self._shadow_done
            and has_remote
            and os.environ.get("SHOOTING_BRAKE_SHADOW") == "1"
            and self._cuda_remap is None
        ):
            self._shadow_done = True
            self._shadow_validate(x, topk_weights, topk_ids, part)

        # Phase 6c marker: confirm the hybrid path was exercised.
        if (
            os.environ.get("SHOOTING_BRAKE_HYBRID") == "1"
            and has_remote
            and stats.get("hybrid_steps", 0) == 0
        ):
            stats["hybrid_steps"] = 1
            hmarker = os.environ.get("SHOOTING_BRAKE_HYBRID_MARKER")
            if hmarker:
                import json
                with open(hmarker, "w") as f:
                    json.dump({"layer": layer_idx, "b70_routes": n_remote}, f)

        # Phase 6c/7/8a: hybrid execution. When enabled and remote routes
        # exist, the routed-expert output is split: CUDA computes the
        # CUDA-owned routes (+ shared expert), and the B70-owned routes are
        # computed separately.  Three modes, selected by environment:
        #
        #   B70_DEVICE=1 + ASYNC (default):
        #     Phase 8a — issue B70 BEFORE CUDA forward_modular so the B70
        #     kernel overlaps with CUDA compute.  take() blocks only if the
        #     B70 kernel hasn't finished by the time CUDA is done.
        #
        #   B70_DEVICE=1 + ASYNC=0:
        #     Phase 7 — synchronous reference.  CUDA first, then B70.
        #     Used for debugging / correctness isolation.
        #
        #   B70_DEVICE unset:
        #     Phase 6c — B70 partial computed via CUDA kernel (no device).
        if (
            os.environ.get("SHOOTING_BRAKE_HYBRID") == "1"
            and has_remote
        ):
            cuda_weights = topk_weights * (~part.b70_mask).float()
            b70_device = os.environ.get("SHOOTING_BRAKE_B70_DEVICE") == "1"
            b70_async = os.environ.get(
                "SHOOTING_BRAKE_B70_ASYNC", "1"
            ) != "0"

            if b70_device and b70_async:
                # Phase 8a: async overlap — B70 kernel runs during CUDA
                seq, b70_M = self._b70_issue(
                    x, topk_ids, topk_weights, part, layer_idx,
                )
                y_cuda = super().forward_modular(
                    x, cuda_weights, cuda_topk_ids,
                    shared_experts, shared_experts_input,
                )
                y_b70 = self._b70_take(seq, b70_M, x.device, x.dtype)
            elif b70_device:
                # Phase 7: synchronous B70 (correctness reference)
                y_cuda = super().forward_modular(
                    x, cuda_weights, cuda_topk_ids,
                    shared_experts, shared_experts_input,
                )
                y_b70 = self._b70_partial(
                    x, topk_ids, topk_weights, part, layer_idx,
                )
            else:
                # Phase 6c: CUDA-kernel B70 partial (no device)
                y_cuda = super().forward_modular(
                    x, cuda_weights, cuda_topk_ids,
                    shared_experts, shared_experts_input,
                )
                b70_weights = topk_weights * part.b70_mask.float()
                y_b70 = super().forward_modular(
                    x, b70_weights, cuda_topk_ids,
                )
            return y_cuda + y_b70

        return super().forward_modular(
            x, topk_weights, cuda_topk_ids, shared_experts, shared_experts_input
        )

    def _b70_partial(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        part: RoutePartition,
        layer_idx: int,
    ) -> torch.Tensor:
        """Compute the B70-device partial for B70-owned routes.

        Translates global expert IDs to B70 compact slots, converts the
        activation BF16 → FP16 for the B70 kernel, dispatches via the
        in-process QuixiCore provider, and returns the weighted partial
        as a BF16 CUDA tensor ready for addition.
        """
        provider = _get_b70_provider(self.shooting_brake_placement)

        # Compact ID translation: global expert → B70 slot, CUDA routes → -1
        ids_np = topk_ids.detach().cpu().numpy()
        wts_np = topk_weights.detach().cpu().numpy()
        b70_ids = self._b70_slot_map[ids_np]          # [M, topk]
        b70_weights = wts_np * (b70_ids >= 0).astype(np.float32)

        # Activation: BF16 CUDA → FP16 CPU for the B70 NVFP4 kernel
        hidden_fp16 = x.detach().to(torch.float16).cpu().numpy()

        output_np = provider.dispatch(
            layer=layer_idx,
            hidden_fp16=hidden_fp16,
            ids=b70_ids,
            weights=b70_weights,
        )

        # Result: FP32 CPU → BF16 CUDA
        return torch.from_numpy(output_np).to(x.device).to(x.dtype)

    def _b70_issue(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        part: RoutePartition,
        layer_idx: int,
    ) -> tuple[int, int]:
        """Issue B70 dispatch — kernel starts asynchronously. Returns (seq, M).

        First half of the Phase-8a async overlap:
        1.  D2H the activation (BF16 CUDA → FP16 pinned host, DMA copy).
        2.  D2H routing data and translate global IDs → B70 compact slots.
        3.  Submit to the B70 provider (SYCL queue enqueues kernel, returns).

        After this returns, the caller should do CUDA work (the routed-
        expert forward) and then call :meth:`_b70_take` to collect.
        """
        provider = _get_b70_provider(self.shooting_brake_placement)
        M = x.shape[0]

        # D2H activation: BF16 CUDA → FP16 pinned host buffer.
        # Small tensor (M × 2048 × 2 B = 4 KB–512 KB); the sync is fast.
        x_fp16 = x.detach().to(torch.float16)
        pinned_hidden = self._b70_pinned_hidden[:M]
        pinned_hidden.copy_(x_fp16, non_blocking=True)
        torch.cuda.current_stream().synchronize()

        # D2H routing data (tiny: M × 8 × 4 B).
        ids_np = topk_ids.detach().cpu().numpy()
        wts_np = topk_weights.detach().cpu().numpy()

        # Translate global expert IDs → B70 compact slots, zero CUDA routes.
        b70_ids = self._b70_slot_map[ids_np]
        b70_weights = wts_np * (b70_ids >= 0).astype(np.float32)

        # Submit — SYCL kernel starts on B70, returns immediately.
        seq = provider.issue(
            layer=layer_idx,
            hidden_fp16=pinned_hidden.numpy(),
            ids=b70_ids,
            weights=b70_weights,
        )
        return seq, M

    def _b70_take(
        self,
        sequence: int,
        M: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Collect B70 result. Blocks only if the kernel is still running.

        Second half of the Phase-8a async overlap.  By the time this is
        called (after CUDA forward_modular), the B70 kernel has typically
        already finished — ``sb_b70_take`` returns immediately.

        Writes into the pre-allocated pinned output buffer, then does a
        non-blocking H2D copy (pinned source enables DMA) + dtype cast.
        """
        provider = _get_b70_provider(self.shooting_brake_placement)
        pinned_output = self._b70_pinned_output[:M]
        provider.take(sequence, M, output=pinned_output.numpy())
        # H2D from pinned (DMA) + cast FP32 → model dtype on GPU.
        return pinned_output.to(device, non_blocking=True).to(dtype)

    def _shadow_validate(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        part: RoutePartition,
    ) -> None:
        """Validate Y_cuda_routed + Y_b70_routed ≈ Y_full_routed.

        Computes the routed-expert output (shared expert excluded) three
        ways using the stock CUDA kernel with different weight masks:
          * all routes (original weights)
          * CUDA routes only (B70-route weights zeroed)
          * B70 routes only (CUDA-route weights zeroed)

        Since 0 * E_k(x) == 0 exactly, the only source of difference is
        BF16 summation reordering. The gate confirms the split-merge
        identity that Phase 6c will rely on. (The B70 device kernel's
        own accuracy was independently validated in Phase 3.)
        """
        cuda_w = topk_weights * (~part.b70_mask).float()
        b70_w = topk_weights * part.b70_mask.float()

        with torch.no_grad():
            y_full = super().forward_modular(x, topk_weights, topk_ids)
            y_cuda = super().forward_modular(x, cuda_w, topk_ids)
            y_b70 = super().forward_modular(x, b70_w, topk_ids)

        y_sum = y_cuda + y_b70
        diff = (y_sum - y_full)
        max_abs = float(diff.abs().max())
        mean_abs = float(diff.abs().mean())
        y_norm = float(y_full.float().norm())
        rel_rmse = float(diff.float().norm()) / max(y_norm, 1e-12)
        cos = float(
            torch.dot(
                y_sum.float().flatten(), y_full.float().flatten()
            )
            / (y_sum.float().norm() * y_full.float().norm() + 1e-12)
        )
        has_nan = bool(torch.isnan(y_sum).any() or torch.isnan(y_full).any())

        logger.info(
            "Shooting Brake Phase-6b shadow: max_abs=%.6f mean_abs=%.6f "
            "rel_rmse=%.6f cosine=%.8f nan=%s",
            max_abs, mean_abs, rel_rmse, cos, has_nan,
        )
        marker = os.environ.get("SHOOTING_BRAKE_SHADOW_MARKER")
        if marker:
            import json
            with open(marker, "w") as f:
                json.dump({
                    "max_abs": max_abs,
                    "mean_abs": mean_abs,
                    "rel_rmse": rel_rmse,
                    "cosine": cos,
                    "has_nan": has_nan,
                    "M": topk_ids.shape[0],
                    "b70_routes": part.num_b70_routes(),
                    "gate_pass": (max_abs < 0.1 and cos > 0.999 and not has_nan),
                }, f)
