"""Phase-4/5 provider boundary: explicitly local-only until Phase 6."""

from __future__ import annotations

from dataclasses import dataclass

from .config import QualifiedModel
from .placement import Placement


@dataclass
class ShootingBrakeExpertProviderClient:
    """Records the local-only contract and rejects premature B70 dispatch."""

    qualified_model: QualifiedModel
    layer_name: str
    placement: Placement
    all_cuda_calls: int = 0

    def begin_all_cuda(self) -> None:
        """Record a routed-expert call whose full contribution stays on CUDA."""
        self.all_cuda_calls += 1

    def issue(self, *args: object, **kwargs: object) -> None:
        """B70 issue/take is deliberately unavailable before Phase 6."""
        del args, kwargs
        raise RuntimeError(
            "B70 route submission is disabled until Phase 6 enables remote routes"
        )

    def take(self, *args: object, **kwargs: object) -> None:
        """A remote partial cannot exist before Phase 6 enables remote routes."""
        del args, kwargs
        raise RuntimeError(
            "B70 result collection is disabled until Phase 6 enables remote routes"
        )
