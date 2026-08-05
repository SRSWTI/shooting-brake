"""Phase-4 admission rules for the all-CUDA Shooting Brake adapter."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass


QUALIFIED_MODEL = "unsloth/Qwen3.6-35B-A3B-NVFP4"
QUALIFIED_ARCHITECTURE = "Qwen3_5MoeForConditionalGeneration"
QUALIFIED_MODEL_TYPE = "qwen3_5_moe_text"
QUALIFIED_HIDDEN_SIZE = 2048
QUALIFIED_LAYERS = 40
QUALIFIED_EXPERTS = 256
QUALIFIED_TOP_K = 8
QUALIFIED_MOE_INTERMEDIATE = 512
# All 40 layers are MoE (256 routed experts each). Layers 0-31 are NVFP4 and
# are extracted into the Phase-1 B70 bank; they are B70-capable. Layers 32-39
# are FP8 and are not in the bank, so their experts are CUDA-forced.
QUALIFIED_BANK_LAYERS = 32
QUALIFIED_BANK_EXPERTS_PER_LAYER = 256
QUALIFIED_FP8_CUDA_ONLY_LAYERS = 8


class QualificationError(RuntimeError):
    """The explicit Phase-4 all-CUDA admission contract was violated."""


@dataclass(frozen=True)
class QualifiedModel:
    model: str
    architecture: str
    hidden_size: int
    num_layers: int
    num_experts: int
    top_k: int
    moe_intermediate_size: int
    bank_layers: int = QUALIFIED_BANK_LAYERS
    bank_experts_per_layer: int = QUALIFIED_BANK_EXPERTS_PER_LAYER

    @property
    def b70_capable_layers(self) -> frozenset[int]:
        """Absolute layer indices whose experts exist in the NVFP4 B70 bank."""
        return frozenset(range(self.bank_layers))


def phase4_enabled() -> bool:
    """Whether this process explicitly selected the Phase-4 local adapter."""
    return (
        os.environ.get("SHOOTING_BRAKE_PHASE4") == "all-cuda"
        and os.environ.get("SHOOTING_BRAKE_MODEL") == QUALIFIED_MODEL
    )


def _architectures(hf_config: object) -> Iterable[str]:
    architectures = getattr(hf_config, "architectures", ())
    return architectures if isinstance(architectures, (list, tuple)) else ()


def require_qualified_config(vllm_config: object) -> QualifiedModel:
    """Fail closed unless this is the frozen eager, single-rank Qwen setup."""
    model_config = getattr(vllm_config, "model_config", None)
    parallel_config = getattr(vllm_config, "parallel_config", None)
    if model_config is None or parallel_config is None:
        raise QualificationError("missing vLLM model or parallel configuration")

    model = getattr(model_config, "model", None)
    if model != QUALIFIED_MODEL:
        raise QualificationError(f"unqualified model: {model!r}")
    # Graph mode is supported: the adapter's forward_modular is a pure
    # pass-through in all-CUDA mode (no Python logic on the hot path).

    hf_config = getattr(model_config, "hf_config", None)
    text_config = getattr(model_config, "hf_text_config", None)
    if QUALIFIED_ARCHITECTURE not in _architectures(hf_config):
        raise QualificationError("unqualified model architecture")
    if getattr(text_config, "model_type", None) != QUALIFIED_MODEL_TYPE:
        raise QualificationError("unqualified Qwen text model type")

    expected = {
        "hidden_size": QUALIFIED_HIDDEN_SIZE,
        "num_hidden_layers": QUALIFIED_LAYERS,
        "num_experts": QUALIFIED_EXPERTS,
        "num_experts_per_tok": QUALIFIED_TOP_K,
        "moe_intermediate_size": QUALIFIED_MOE_INTERMEDIATE,
    }
    mismatches = {
        name: (getattr(text_config, name, None), value)
        for name, value in expected.items()
        if getattr(text_config, name, None) != value
    }
    if mismatches:
        raise QualificationError(f"unqualified Qwen dimensions: {mismatches}")

    if getattr(parallel_config, "tensor_parallel_size", None) != 1:
        raise QualificationError("Phase 4 requires tensor parallel size one")
    if getattr(parallel_config, "pipeline_parallel_size", None) != 1:
        raise QualificationError("Phase 4 requires pipeline parallel size one")
    if getattr(parallel_config, "enable_eplb", False):
        raise QualificationError("Phase 4 does not admit EPLB")

    return QualifiedModel(
        model=model,
        architecture=QUALIFIED_ARCHITECTURE,
        hidden_size=QUALIFIED_HIDDEN_SIZE,
        num_layers=QUALIFIED_LAYERS,
        num_experts=QUALIFIED_EXPERTS,
        top_k=QUALIFIED_TOP_K,
        moe_intermediate_size=QUALIFIED_MOE_INTERMEDIATE,
    )
