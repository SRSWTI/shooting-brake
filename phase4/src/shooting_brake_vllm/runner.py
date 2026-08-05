"""Qwen-scoped Phase-4 MoE runner that preserves stock CUDA execution."""

from __future__ import annotations

from typing import Any

import torch

from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

from .config import require_qualified_config
from .routed_experts import HybridRoutedExperts


class HybridMoERunner(MoERunner):
    """Run the unchanged vLLM CUDA path while holding the future provider seam."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        require_qualified_config(get_current_vllm_config())
        super().__init__(*args, **kwargs)
        if not isinstance(self.routed_experts, HybridRoutedExperts):
            raise RuntimeError("Phase-4 runner requires HybridRoutedExperts")

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # This is the only Phase-4 behavioral addition. It does not inspect,
        # reorder, or modify router data or CUDA expert output.
        self.routed_experts.shooting_brake_provider.begin_all_cuda()
        return super()._forward_impl(
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids,
        )
