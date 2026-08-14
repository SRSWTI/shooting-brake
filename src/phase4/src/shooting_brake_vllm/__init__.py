"""Out-of-tree registration for the Shooting Brake Phase-4 adapter."""

from __future__ import annotations

import os

from .config import phase4_enabled

_force_piecewise_fired = False


def force_piecewise_fired() -> bool:
    """Whether this process actually forced the hybrid graph mode."""
    return _force_piecewise_fired



def _patch_force_piecewise() -> None:
    """Force ``CUDAGraphMode.PIECEWISE`` when breakable graphs are active.

    vLLM assigns FULL mode to some batch descriptors (decode-only batches).
    In FULL mode, ``@eager_break_during_capture`` skips the break (by design
    — FULL means "capture as one graph, no breaks").  Our hybrid MoE code
    then runs inside stream capture and fails on ``.any()``, ``.cpu()``, etc.

    Forcing PIECEWISE for ALL descriptors ensures every layer goes through
    the breakable path where ``@eager_break_during_capture`` creates breaks.
    """
    if os.environ.get("VLLM_USE_BREAKABLE_CUDAGRAPH") != "1":
        return
    if os.environ.get("SHOOTING_BRAKE_HYBRID") != "1":
        return

    from vllm.config import CUDAGraphMode
    from vllm.config.vllm import VllmConfig
    from vllm.logger import init_logger

    _logger = init_logger(__name__)
    _orig_post_init = VllmConfig.__post_init__

    def _patched_post_init(self: VllmConfig) -> None:
        global _force_piecewise_fired
        _orig_post_init(self)
        if (
            self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
            and self.model_config is not None
            and not self.model_config.enforce_eager
        ):
            self.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
            _force_piecewise_fired = True
            _logger.info(
                "Shooting Brake: forced CUDAGraphMode.PIECEWISE for "
                "breakable hybrid MoE graph capture."
            )

    VllmConfig.__post_init__ = _patched_post_init


def register() -> None:
    """Register when a model in the split-checkpoint registry is selected."""
    if not phase4_enabled():
        return

    _patch_force_piecewise()

    from vllm.model_executor.custom_op import PluggableLayer, op_registry_oot

    from .routed_experts import (
        HybridRoutedExperts,
        install_preemptive_alloc_hook,
        preemptive_surgery_enabled,
    )
    from .runner import HybridMoERunner

    if preemptive_surgery_enabled():
        # Must land before any layer is constructed: `create_weights` runs
        # inside `RoutedExperts.__init__`, so an instance-level wrapper
        # installed by the adapter would always be one layer too late.
        install_preemptive_alloc_hook()

    registrations = {
        "RoutedExperts": HybridRoutedExperts,
        "MoERunner": HybridMoERunner,
    }
    for name, implementation in registrations.items():
        existing = op_registry_oot.get(name)
        if existing is None:
            PluggableLayer.register_oot(implementation, name=name)
        elif existing is not implementation:
            raise RuntimeError(
                f"cannot install Shooting Brake adapter: {name} is already replaced"
            )


__all__ = ["register"]
