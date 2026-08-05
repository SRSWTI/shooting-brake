"""Qwen-scoped Phase-4 MoE runner that preserves stock CUDA execution."""

from __future__ import annotations

import os
from typing import Any

import torch

from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

from .config import require_qualified_config
from .routed_experts import HybridRoutedExperts

try:
    from vllm.compilation.breakable_cudagraph import eager_break_during_capture
except ImportError:
    def eager_break_during_capture(fn):  # type: ignore[misc]
        return fn


class HybridMoERunner(MoERunner):
    """Run the unchanged vLLM CUDA path while holding the future provider seam.

    When ``VLLM_USE_BREAKABLE_CUDAGRAPH=1`` and hybrid features are active,
    ``_forward_impl`` is decorated with ``@eager_break_during_capture`` so the
    entire MoE forward (shared experts + routed experts + B70 dispatch) runs
    as an eager break between CUDA graph segments.  Attention layers, router,
    norms and embeddings stay in graph segments at full replay speed.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        require_qualified_config(get_current_vllm_config())
        super().__init__(*args, **kwargs)
        if not isinstance(self.routed_experts, HybridRoutedExperts):
            raise RuntimeError("Phase-4 runner requires HybridRoutedExperts")
        self._static_runner_output: torch.Tensor | None = None

    @eager_break_during_capture
    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        self.routed_experts.shooting_brake_provider.begin_all_cuda()
        result = super()._forward_impl(
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids,
        )
        return self._stabilize_output(result)

    def _stabilize_output(
        self, result: Any,
    ) -> Any:
        """Copy result into a static buffer so the output address is stable
        across CUDA graph capture and replay.

        Required by ``@eager_break_during_capture``: a fresh tensor returned
        each call would land at a different address and break downstream graph
        segments that were captured reading from the capture-time address.
        """
        if isinstance(result, torch.Tensor):
            return self._copy_to_static(result)
        if isinstance(result, (tuple, list)):
            return type(result)(
                self._copy_to_static(t) if isinstance(t, torch.Tensor) else t
                for t in result
            )
        return result

    def _copy_to_static(self, tensor: torch.Tensor) -> torch.Tensor:
        M = tensor.shape[0]
        need_realloc = (
            self._static_runner_output is None
            or self._static_runner_output.shape[0] < M
            or self._static_runner_output.dtype != tensor.dtype
            or self._static_runner_output.device != tensor.device
            or self._static_runner_output.ndim != tensor.ndim
            or (
                self._static_runner_output.ndim > 1
                and self._static_runner_output.shape[1] != tensor.shape[1]
            )
        )
        if need_realloc:
            max_b = max(
                M,
                int(os.environ.get("SHOOTING_BRAKE_B70_MAX_BATCH", "128")),
            )
            shape = (max_b,) + tuple(tensor.shape[1:])
            self._static_runner_output = torch.empty(
                shape, dtype=tensor.dtype, device=tensor.device,
            )
        self._static_runner_output[:M].copy_(tensor)
        return self._static_runner_output[:M]
