"""Qualified routed-expert container with Phase-5/6a ownership + partitioning."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.logger import init_logger
try:
    from vllm.compilation.breakable_cudagraph import eager_break_during_capture
except ImportError:
    def eager_break_during_capture(fn):  # type: ignore[misc]
        return fn

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
    """[num_experts] int32: B70 compact slot per global expert, -1 for CUDA.

    Every B70-active layer offloads the same expert ids, so one map
    serves them all. Layer 0 is not necessarily one of them — a subset
    policy may leave it entirely on CUDA — so read from the first layer
    that actually owns B70 experts.
    """
    slot_map = np.full(placement.num_experts, -1, dtype=np.int32)
    active = placement.b70_active_layers()
    if not active:
        return slot_map
    for expert_id, owner in enumerate(placement.owners[active[0]]):
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
    # Every B70-active layer offloads the same expert ids — one resident
    # set covers the whole bank — but layer 0 is not necessarily active
    # (a subset policy may leave it entirely on CUDA), so read the set
    # from the first layer that actually owns B70 experts.
    reference_layer = placement.b70_active_layers()[0]
    resident = sorted(
        e for e in range(placement.num_experts)
        if placement.owners[reference_layer][e].device is Device.B70
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


_SURGERY_HOOK_ATTR = "_shooting_brake_surgery_hook"


def _install_surgery_hook(quant_method: Any) -> None:
    """Run VRAM surgery as soon as weights finish loading.

    Timing is the whole point. vLLM sizes the KV cache from the *peak*
    memory observed during its profiling forward. Freeing the B70-owned
    expert weights inside or after that forward is too late: the peak
    already counted them, so the freed VRAM is left unusable — the
    capacity gain this architecture exists to deliver is silently lost,
    and only shows up as a large "free" figure at the end of the run.

    ``process_weights_after_loading`` is the last hook before profiling,
    and it receives the layer, so one wrapper per quant method serves
    every layer that shares it. The sentinel keeps a shared method from
    being wrapped twice.
    """
    if getattr(quant_method, _SURGERY_HOOK_ATTR, False):
        return
    original = quant_method.process_weights_after_loading

    def process_weights_after_loading(layer: Any) -> None:
        original(layer)
        if isinstance(layer, HybridRoutedExperts):
            layer._maybe_perform_vram_surgery(layer.layer_index)

    quant_method.process_weights_after_loading = process_weights_after_loading
    setattr(quant_method, _SURGERY_HOOK_ATTR, True)


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
        self._passthrough_warned = False
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
        # Cache B70 env decisions (hot-path reads are expensive during graph replay).
        self._b70_device_cached = (
            os.environ.get("SHOOTING_BRAKE_B70_DEVICE") == "1"
        )
        self._hybrid_env = os.environ.get("SHOOTING_BRAKE_HYBRID") == "1"
        self._b70_async_cached = (
            os.environ.get("SHOOTING_BRAKE_B70_ASYNC", "1") != "0"
        )
        # Static output buffer for breakable CUDA graph compatibility.
        # Allocated lazily on first forward (needs hidden_dim from weights).
        self._static_output: torch.Tensor | None = None
        # Tier 3: graph-compatible B70 dispatch via CUDA stream signaling.
        # When enabled, B70 dispatch uses cuStreamWriteValue32/WaitValue32
        # (capture-compatible) instead of Python/ctypes (not capture-compatible).
        self._b70_graph_mode = (
            os.environ.get("SHOOTING_BRAKE_B70_GRAPH") == "1"
            and self._b70_device_cached
            and self._hybrid_env
        )
        self._b70_poller: Any = None
        self._first_forward_done = False
        self._b70_stats = os.environ.get("SHOOTING_BRAKE_B70_STATS") == "1"
        # [b70_routes, total_routes] accumulated on device.  Incrementing
        # a device tensor needs no host sync, so it survives graph
        # capture; read it between steps via collective_rpc.
        self._route_counter: torch.Tensor | None = (
            torch.zeros(2, dtype=torch.int64, device="cuda")
            if self._b70_stats and self._b70_graph_mode
            else None
        )
        if self._b70_graph_mode:
            # CUDA-side slot map (for graph-compatible gather).
            self._b70_slot_map_cuda = torch.tensor(
                self._b70_slot_map, device="cuda", dtype=torch.int32,
            )
            # Pinned buffers for routing data (D2H targets).
            self._pinned_b70_ids = torch.empty(
                self._b70_max_batch, 8, dtype=torch.int32,
                pin_memory=True, device="cpu",
            )
            self._pinned_b70_weights = torch.empty(
                self._b70_max_batch, 8, dtype=torch.float32,
                pin_memory=True, device="cpu",
            )
            # Device-side result buffers (pre-allocated, no torch.empty in forward).
            self._dev_b70_fp32 = torch.empty(
                self._b70_max_batch, 2048, dtype=torch.float32, device="cuda",
            )
            self._dev_b70_bf16 = torch.empty(
                self._b70_max_batch, 2048, dtype=torch.bfloat16, device="cuda",
            )
            # Signal/completion flags (host-mapped, accessible from both sides).
            from .stream_signal import alloc_host_mapped_flag
            self._signal_host, self._signal_dev = alloc_host_mapped_flag(0)
            self._completion_host, self._completion_dev = alloc_host_mapped_flag(0)
            # One shared poller serves every NVFP4 layer (single physical
            # B70).  Registration is deferred to the first forward, where
            # `_ensure_layer_device_map` resolves this layer's real index.
            logger.info(
                "Shooting Brake Tier 3: graph-compatible B70 dispatch "
                "enabled for layer %s", self.layer_name,
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
        _install_surgery_hook(self.quant_method)

    @property
    def layer_index(self) -> int:
        """Absolute layer index, parsed from ``layer_name``."""
        if self._layer_idx is None:
            from vllm.model_executor.models.utils import extract_layer_index

            self._layer_idx = extract_layer_index(self.layer_name)
        return self._layer_idx

    def _ensure_layer_device_map(
        self, topk_ids: torch.Tensor
    ) -> tuple[int, torch.Tensor]:
        """Resolve this layer's index and cache its device-map row."""
        layer_idx = self.layer_index
        if (
            self._device_map_layer is None
            or self._device_map_layer.device != topk_ids.device
        ):
            self._device_map_layer = self._device_map_cpu[layer_idx].to(
                topk_ids.device
            )
        return layer_idx, self._device_map_layer

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

    def _register_b70_poller(self, layer_idx: int) -> None:
        """Bind this layer's flags and pinned buffers to the shared poller.

        A layer that owns no B70 experts becomes pure CUDA passthrough.
        That covers the FP8 layers 32-39, which the bank does not span,
        and every layer a subset placement leaves on CUDA. The gate is
        ownership, not bank capability: dispatching a layer with nothing
        to compute still costs a full round trip per token, which is the
        exact cost a subset placement exists to avoid.

        Route math is unchanged — with no B70-owned experts the partition
        produces no remote routes anyway.
        """
        from .b70_poller import get_b70_poller

        if not self.shooting_brake_placement.is_b70_active(layer_idx):
            self._b70_graph_mode = False
            self._all_cuda_passthrough = True
            logger.info(
                "Shooting Brake Tier 3: layer %d owns no B70 experts — "
                "all-CUDA passthrough", layer_idx,
            )
            return

        poller = get_b70_poller(self.shooting_brake_placement)
        poller.register_layer(
            layer_idx=layer_idx,
            signal_host=self._signal_host,
            completion_host=self._completion_host,
            pinned_hidden=self._b70_pinned_hidden,
            pinned_ids=self._pinned_b70_ids,
            pinned_weights=self._pinned_b70_weights,
            pinned_output=self._b70_pinned_output,
        )
        poller.start()
        self._b70_poller = poller
        logger.info(
            "Shooting Brake Tier 3: layer %d registered with B70 poller",
            layer_idx,
        )

    def _first_forward_setup(self, topk_ids: torch.Tensor) -> None:
        """One-time per-layer setup, run on the very first forward.

        Only the poller binding lives here. It loads the 13.5 GiB expert
        bank and creates the SYCL queue, which must happen neither during
        model construction (it races CUDA context setup and starves
        weight loading of CPU) nor during CUDA graph capture (it would
        race an active stream capture). vLLM's first forward is the eager
        profiling pass, which sits between the two.

        VRAM surgery is deliberately *not* here: it runs earlier, off
        ``process_weights_after_loading``, so the freed memory is already
        reflected in the profiling peak vLLM sizes the KV cache from.
        See :func:`_install_surgery_hook`.
        """
        self._ensure_layer_device_map(topk_ids)
        if self._b70_graph_mode:
            self._register_b70_poller(self.layer_index)
        self._first_forward_done = True

    def forward_modular(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: Any = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dispatch to hybrid or CUDA-only forward.

        In all-CUDA mode (no hybrid features) this is a pure pass-through
        to ``super().forward_modular()`` — CUDA graph compatible.

        Under Tier 3, decode-sized batches take the graph-compatible B70
        path; prefill stays all-CUDA because the pinned staging buffers
        are sized for decode.
        """
        if not self._first_forward_done:
            self._first_forward_setup(topk_ids)
        elif self._b70_poller is not None:
            # A failed dispatch leaves stale data in the output buffer:
            # the poller raises the completion flag regardless, because
            # the CUDA-side wait cannot time out and an unset flag wedges
            # the device.  Replay runs no Python, so surface it here — on
            # the first eager forward after the fault — instead of
            # returning silently wrong routes.
            errors = self._b70_poller.error_count
            if errors:
                raise RuntimeError(
                    f"Shooting Brake Tier 3: {errors} B70 dispatch(es) "
                    "failed; routed-expert output is not trustworthy"
                )

        # Prefill (large M) stays all-CUDA: the B70 staging buffers are
        # sized for decode batches.
        if self._all_cuda_passthrough or x.shape[0] > self._b70_max_batch:
            if self._cuda_remap is not None and not self._passthrough_warned:
                # This layer had its B70-owned experts sliced out of the
                # CUDA weights, so a raw pass-through cannot account for
                # B70 routes. Report it rather than return quietly wrong
                # routed output.
                self._passthrough_warned = True
                logger.warning(
                    "Shooting Brake: layer %d took the all-CUDA path with "
                    "M=%d after VRAM surgery — B70 routes are unaccounted",
                    self.layer_index, x.shape[0],
                )
            return super().forward_modular(
                x, topk_weights, topk_ids,
                shared_experts, shared_experts_input,
            )
        return self._hybrid_forward_modular(
            x, topk_weights, topk_ids,
            shared_experts, shared_experts_input,
        )

    @eager_break_during_capture
    def _hybrid_forward_modular(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: Any = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Hybrid MoE forward.

        Two paths. Under Tier 3 the whole B70 dispatch is CUDA stream
        operations, captured by a normal CUDA graph with no Python in the
        replay. Otherwise the eager path partitions routes, validates the
        split, and dispatches host-side.

        Per-layer setup (index resolution, VRAM surgery, poller binding)
        has already run in ``forward_modular``; it must happen on the
        profiling pass, which never reaches this method.
        """
        layer_idx, dml = self._ensure_layer_device_map(topk_ids)
        if self._b70_graph_mode:
            # Tier 3: pure CUDA stream ops — no partition, validation,
            # stats, or any host-side sync.  Every op below is captured
            # by torch.cuda.graph() without modification.
            b70_ids = self._b70_slot_map_cuda[topk_ids]
            b70_mask = b70_ids >= 0
            # B70-owned routes contribute through the B70 partial, so
            # zero their CUDA weight; CUDA-owned routes are untouched.
            cuda_weights = topk_weights * (~b70_mask).to(topk_weights.dtype)
            cuda_topk_ids = (
                self._cuda_remap[topk_ids]
                if self._cuda_remap is not None
                else topk_ids
            )
            if self._route_counter is not None:
                # Device-side accumulation: no host sync, so this stays
                # inside the captured graph.  Read between steps.
                self._route_counter[0] += b70_mask.sum()
                self._route_counter[1] += b70_mask.numel()
            self._b70_issue_graph(x, b70_ids, topk_weights)
            y_cuda = super().forward_modular(
                x, cuda_weights, cuda_topk_ids,
                shared_experts, shared_experts_input,
            )
            y_b70 = self._b70_take_graph(x.shape[0])
            return y_cuda + y_b70
        part = partition_routes(topk_ids, topk_weights, dml, layer_idx)
        validate_partition(
            part, self.shooting_brake_placement.b70_capable_layers
        )
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
                marker = os.environ.get("SHOOTING_BRAKE_PARTITION_MARKER")
                if marker:
                    import json
                    with open(marker, "a") as f:
                        f.write(json.dumps({
                            "layer": layer_idx,
                            "b70_routes": n_remote,
                            "M": topk_ids.shape[0],
                        }) + "\n")
        # Phase 6b: shadow validation (once, skip during surgery).
        if (
            not self._shadow_done
            and has_remote
            and os.environ.get("SHOOTING_BRAKE_SHADOW") == "1"
            and self._cuda_remap is None
        ):
            self._shadow_done = True
            self._shadow_validate(x, topk_weights, topk_ids, part)

        if (
            self._hybrid_env
            and has_remote
            and stats.get("hybrid_steps", 0) == 0
        ):
            stats["hybrid_steps"] = 1
            hmarker = os.environ.get("SHOOTING_BRAKE_HYBRID_MARKER")
            if hmarker:
                import json
                with open(hmarker, "w") as f:
                    json.dump({"layer": layer_idx, "b70_routes": n_remote}, f)

        # Hybrid execution: split CUDA/B70 routes and combine.
        if self._hybrid_env and has_remote:
            cuda_weights = topk_weights * (~part.b70_mask).float()

            if self._b70_graph_mode:
                # Tier 3: graph-compatible B70 dispatch (pure CUDA stream ops).
                self._b70_issue_graph(x, topk_ids, topk_weights)
                y_cuda = super().forward_modular(
                    x, cuda_weights, cuda_topk_ids,
                    shared_experts, shared_experts_input,
                )
                y_b70 = self._b70_take_graph(x.shape[0])
            elif self._b70_device_cached and self._b70_async_cached:
                # Phase 8a: async overlap — B70 kernel runs during CUDA.
                seq, b70_M = self._b70_issue(
                    x, topk_ids, topk_weights, part, layer_idx,
                )
                y_cuda = super().forward_modular(
                    x, cuda_weights, cuda_topk_ids,
                    shared_experts, shared_experts_input,
                )
                y_b70 = self._b70_take(seq, b70_M, x.device, x.dtype)
            elif self._b70_device_cached:
                # Phase 7: synchronous B70 (correctness reference).
                y_cuda = super().forward_modular(
                    x, cuda_weights, cuda_topk_ids,
                    shared_experts, shared_experts_input,
                )
                y_b70 = self._b70_partial(
                    x, topk_ids, topk_weights, part, layer_idx,
                )
            else:
                # Phase 6c: CUDA-kernel B70 partial (no device).
                y_cuda = super().forward_modular(
                    x, cuda_weights, cuda_topk_ids,
                    shared_experts, shared_experts_input,
                )
                b70_weights = topk_weights * part.b70_mask.float()
                y_b70 = super().forward_modular(
                    x, b70_weights, cuda_topk_ids,
                )
            return self._write_static_output(y_cuda + y_b70)

        return self._write_static_output(super().forward_modular(
            x, topk_weights, cuda_topk_ids, shared_experts, shared_experts_input
        ))

    def _write_static_output(self, result: torch.Tensor) -> torch.Tensor:
        """Copy result into static buffer for stable address across replays.

        Required by ``@eager_break_during_capture``: the output tensor's
        address must not change between capture and replay, or downstream
        graph segments read from a stale address.
        """
        M = result.shape[0]
        if (
            self._static_output is None
            or self._static_output.shape[0] < M
            or self._static_output.dtype != result.dtype
            or self._static_output.device != result.device
        ):
            max_b = max(
                M,
                int(os.environ.get("SHOOTING_BRAKE_B70_MAX_BATCH", "128")),
            )
            self._static_output = torch.empty(
                max_b, result.shape[1],
                dtype=result.dtype, device=result.device,
            )
        self._static_output[:M].copy_(result)
        return self._static_output[:M]

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

    def _b70_issue_graph(
        self,
        x: torch.Tensor,
        b70_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        """Graph-compatible B70 issue — pure CUDA stream operations.

        Replaces the eager-mode ``_b70_issue``, which uses ``.cpu()``,
        numpy, and ctypes and so cannot be captured.  Every operation
        here is a CUDA stream op that ``torch.cuda.graph()`` records
        unmodified:

          1. dtype cast BF16 -> FP16 (CUDA)
          2. D2H copy activation to pinned (cudaMemcpyAsync)
          3. D2H copy routing data to pinned (cudaMemcpyAsync)
          4. cuStreamWriteValue32 signal flag = M (Driver API)

        Args:
            x: [M, hidden] activation on CUDA.
            b70_ids: [M, topk] compact B70 slots, -1 for CUDA-owned
                routes.  Gathered by the caller, which already needs the
                mask to zero CUDA weights.
            topk_weights: [M, topk] routing weights, unmodified.
        """
        from .stream_signal import write_flag
        M = x.shape[0]

        # 1-2. Activation: BF16 -> FP16, D2H to pinned.
        self._b70_pinned_hidden[:M].copy_(
            x.to(torch.float16), non_blocking=True,
        )

        # 3. Routing data.  CUDA-owned routes carry slot -1 and are
        # skipped by the kernel, so their weights are irrelevant; the
        # weights go over as-is.
        self._pinned_b70_ids[:M].copy_(b70_ids, non_blocking=True)
        self._pinned_b70_weights[:M].copy_(topk_weights, non_blocking=True)

        # 4. Signal the poller.  The flag VALUE is M, so the poller
        # dispatches exactly this batch instead of the whole buffer.
        # M is a constant at capture time (one graph per batch size),
        # so it is baked into the replayed write — no extra transfer.
        write_flag(self._signal_dev, M)

    def _b70_take_graph(
        self,
        M: int,
    ) -> torch.Tensor:
        """Graph-compatible B70 take — pure CUDA stream operations.

        Replaces the eager-mode ``_b70_take`` which uses ctypes and
        .to(device).  Every operation here is captured by the graph:

          1. cuStreamWaitValue32 completion flag (Driver API)
          2. H2D copy FP32 result to device (cudaMemcpyAsync)
          3. dtype cast FP32→BF16 (CUDA)
          4. cuStreamWriteValue32 reset completion (Driver API)
        """
        from .stream_signal import wait_flag, write_flag

        # 1. Wait for B70 thread to finish.
        wait_flag(self._completion_dev, 1)

        # 2-3. H2D copy + dtype cast.
        self._dev_b70_fp32[:M].copy_(
            self._b70_pinned_output[:M], non_blocking=True,
        )
        self._dev_b70_bf16[:M].copy_(self._dev_b70_fp32[:M])

        # 4. Reset completion for next replay.
        write_flag(self._completion_dev, 0)

        return self._dev_b70_bf16[:M]

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
