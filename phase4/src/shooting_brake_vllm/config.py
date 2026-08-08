"""Phase-4 admission rules for the all-CUDA Shooting Brake adapter."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass


#: Models this adapter admits. Both share an architecture and text model
#: type; what differs is geometry, and that is read from the model's own
#: config rather than restated here. A second hard-coded shape is how the
#: table and the checkpoint drift apart.
QUALIFIED_MODELS = (
    "unsloth/Qwen3.6-35B-A3B-NVFP4",
    "unsloth/Qwen3.5-122B-A10B-NVFP4",
)
#: Default when SHOOTING_BRAKE_MODEL is unset, and the shape the phase
#: tests pin against.
QUALIFIED_MODEL = QUALIFIED_MODELS[0]
QUALIFIED_ARCHITECTURE = "Qwen3_5MoeForConditionalGeneration"
QUALIFIED_MODEL_TYPE = "qwen3_5_moe_text"

#: The 35B's geometry. Retained because the phase tests describe that model
#: specifically; `require_qualified_config` no longer compares against it.
QUALIFIED_HIDDEN_SIZE = 2048
QUALIFIED_LAYERS = 40
QUALIFIED_EXPERTS = 256
QUALIFIED_TOP_K = 8
QUALIFIED_MOE_INTERMEDIATE = 512
# All 40 layers are MoE (256 routed experts each). Layers 0-31 are NVFP4 and
# are extracted into the Phase-1 B70 bank; they are B70-capable. Layers 32-39
# are FP8 and are not in the bank, so their experts are CUDA-forced. The
# 122B splits 47/1 the same way. Which layers a *run* can offload is read
# from the bank header, not from these, so the two cannot disagree.
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
        and os.environ.get("SHOOTING_BRAKE_MODEL") in QUALIFIED_MODELS
    )


def bank_path() -> str:
    """The expert bank this run serves from.

    One resolution point for the whole package. The provider, the
    qualification gate and the host arena must name the same file, or the
    arena is filled from a different model's weights than the B70
    dispatches to. The default is repo-relative rather than cwd-relative,
    so a run started from another directory reads the same bank.
    """
    from pathlib import Path

    default = Path(__file__).resolve().parents[3] / "phase1" / "expert_bank.bin"
    return os.environ.get("SHOOTING_BRAKE_B70_BANK", str(default))


#: `<8sIIIIIQQQQ` — magic, layers, experts/layer, hidden, intermediate, pad,
#: then four record-section sizes. Written by phase1/extract_experts.py.
_BANK_HEADER_FMT = "<8sIIIIIQQQQ"
_BANK_MAGIC = b"SBEXP001"


@dataclass(frozen=True)
class BankHeader:
    """Geometry the expert bank was actually built for."""

    layers: int
    experts_per_layer: int
    hidden_size: int
    moe_intermediate_size: int

    #: A bank is absent, which is the truth for an all-CUDA run.
    @staticmethod
    def absent() -> "BankHeader":
        return BankHeader(0, 0, 0, 0)


def read_bank_header(path: str | None = None) -> BankHeader:
    """Geometry from the expert bank's own header.

    Which layers can be offloaded, and at what shape, are properties of the
    bank that was built — not of constants in this file. Reading them here
    means a bank built for one model cannot be silently paired with
    another. That is not hypothetical: the default path holds whichever
    bank was built last, and a 32-layer 35B bank is numerically compatible
    with a 47-layer model's *bounds* while describing entirely different
    weights.
    """
    import struct
    from pathlib import Path

    if path is None:
        path = bank_path()
    bank = Path(path)
    if not bank.is_file():
        # Absent is the truth for an all-CUDA run. Under an offloading
        # policy it is a trap: `b70_capable_layers` goes empty, every policy
        # CUDA-forces every layer because it only assigns B70/CPU owners
        # `if layer in b70_capable`, and the run completes as all-CUDA while
        # its result JSON still records the hybrid placement it was asked
        # for. A missing or mistyped bank path must not be able to produce a
        # measurement that reads as authoritative.
        if os.environ.get("SHOOTING_BRAKE_HYBRID") == "1":
            raise QualificationError(
                f"expert bank not found at {bank} — an offloading placement "
                "cannot be served without it. Set SHOOTING_BRAKE_B70_BANK to "
                "the bank built for this model."
            )
        return BankHeader.absent()
    with bank.open("rb") as f:
        raw = f.read(struct.calcsize(_BANK_HEADER_FMT))
    magic, layers, experts, hidden, inter, *_rest = struct.unpack(
        _BANK_HEADER_FMT, raw
    )
    if magic != _BANK_MAGIC:
        raise QualificationError(f"{bank} is not a Shooting Brake expert bank")
    return BankHeader(int(layers), int(experts), int(hidden), int(inter))


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
    if model not in QUALIFIED_MODELS:
        raise QualificationError(
            f"unqualified model: {model!r} (admitted: {list(QUALIFIED_MODELS)})"
        )
    # Graph mode is supported: the adapter's forward_modular is a pure
    # pass-through in all-CUDA mode (no Python logic on the hot path).

    hf_config = getattr(model_config, "hf_config", None)
    text_config = getattr(model_config, "hf_text_config", None)
    if QUALIFIED_ARCHITECTURE not in _architectures(hf_config):
        raise QualificationError("unqualified model architecture")
    if getattr(text_config, "model_type", None) != QUALIFIED_MODEL_TYPE:
        raise QualificationError("unqualified Qwen text model type")

    # Geometry comes from the model, not from a table. The admitted set
    # above is the contract; restating each model's dimensions here would
    # only create a second source of truth to drift from the checkpoint.
    # What is still checked is that every dimension exists and is usable.
    geometry = {}
    for name in (
        "hidden_size", "num_hidden_layers", "num_experts",
        "num_experts_per_tok", "moe_intermediate_size",
    ):
        value = getattr(text_config, name, None)
        if not isinstance(value, int) or value < 1:
            raise QualificationError(
                f"model config is missing a usable {name}: {value!r}"
            )
        geometry[name] = value

    # NVFP4 packs two weights per byte and one block scale per 16, so a
    # dimension that is not a multiple of 16 cannot be expressed in the
    # bank's record layout at all.
    for name in ("hidden_size", "moe_intermediate_size"):
        if geometry[name] % 16:
            raise QualificationError(
                f"{name}={geometry[name]} is not a multiple of 16; the NVFP4 "
                "expert-bank layout cannot represent it"
            )

    # The B70 provider keeps these two as compile-time constants: they size
    # buffers and a stack array before the bank is even opened, and both
    # qualified models share them. Geometry is otherwise read from the
    # model, so without this check a third model could pass qualification
    # and then meet a provider built for a different routing width — which
    # would feed the kernel routing slots that were never populated, with
    # no error raised anywhere.
    for name, required in (("num_experts", 256), ("num_experts_per_tok", 8)):
        if geometry[name] != required:
            raise QualificationError(
                f"{name}={geometry[name]} but the B70 provider is compiled "
                f"for {required} (phase1/b70_provider.cpp). Both qualified "
                "models match; a new one needs the provider changed too."
            )

    # Identity, not a bound. A stale 35B bank (32 layers, hidden 2048,
    # intermediate 512) satisfies `32 <= 47` against the 122B and would be
    # served happily, with the B70 returning experts of the wrong shape for
    # every routed token. Comparing the shape the bank was built for
    # against the model's own geometry is what makes that impossible.
    bank = read_bank_header()
    bank_layers = bank.layers
    if bank_layers:
        # Every dimension must match exactly. The layer count is the one
        # exception, and only downward: the FP8 tail is deliberately absent
        # from the bank, so a bank covering fewer layers than the model is
        # normal, while one covering more is a different model's bank.
        wrong: dict[str, tuple[int, int]] = {}
        if bank_layers > geometry["num_hidden_layers"]:
            wrong["layers"] = (bank_layers, geometry["num_hidden_layers"])
        for name, got, want in (
            ("experts_per_layer", bank.experts_per_layer,
             geometry["num_experts"]),
            ("hidden_size", bank.hidden_size, geometry["hidden_size"]),
            ("moe_intermediate_size", bank.moe_intermediate_size,
             geometry["moe_intermediate_size"]),
        ):
            if got != want:
                wrong[name] = (got, want)
        if wrong:
            detail = ", ".join(
                f"{k} bank={g} model={w}" for k, (g, w) in wrong.items()
            )
            raise QualificationError(
                f"expert bank was built for a different model than {model}: "
                f"{detail}. Point SHOOTING_BRAKE_B70_BANK at this model's bank."
            )

    if getattr(parallel_config, "tensor_parallel_size", None) != 1:
        raise QualificationError("Phase 4 requires tensor parallel size one")
    if getattr(parallel_config, "pipeline_parallel_size", None) != 1:
        raise QualificationError("Phase 4 requires pipeline parallel size one")
    if getattr(parallel_config, "enable_eplb", False):
        raise QualificationError("Phase 4 does not admit EPLB")

    return QualifiedModel(
        model=model,
        architecture=QUALIFIED_ARCHITECTURE,
        hidden_size=geometry["hidden_size"],
        num_layers=geometry["num_hidden_layers"],
        num_experts=geometry["num_experts"],
        top_k=geometry["num_experts_per_tok"],
        moe_intermediate_size=geometry["moe_intermediate_size"],
        bank_layers=bank_layers,
        bank_experts_per_layer=geometry["num_experts"],
    )
