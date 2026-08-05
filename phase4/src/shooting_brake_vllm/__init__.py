"""Out-of-tree registration for the Shooting Brake Phase-4 adapter."""

from __future__ import annotations

from .config import phase4_enabled


def register() -> None:
    """Register only under the explicit frozen all-CUDA selection."""
    if not phase4_enabled():
        return

    from vllm.model_executor.custom_op import PluggableLayer, op_registry_oot

    from .routed_experts import HybridRoutedExperts
    from .runner import HybridMoERunner

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
