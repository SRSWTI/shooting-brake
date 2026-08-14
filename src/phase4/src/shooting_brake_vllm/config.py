"""Phase-4 admission and split-checkpoint model metadata."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass
from .int4_bank_format import (
    MAGIC as INT4_BANK_MAGIC,
    VERSION as INT4_BANK_VERSION,
    read_int4_bank_header,
)

logger = logging.getLogger(__name__)
_text_only_audit_logged = False


QUALIFIED_ARCHITECTURE = "Qwen3_5MoeForConditionalGeneration"
QUALIFIED_MODEL_TYPE = "qwen3_5_moe_text"


@dataclass(frozen=True)
class SupportedModel:
    """Immutable contract for one dense checkpoint and its routed experts."""

    model: str
    routed_experts_model: str
    routed_expert_format: str
    hidden_size: int
    num_layers: int
    num_experts: int
    top_k: int
    moe_intermediate_size: int
    architecture: str = QUALIFIED_ARCHITECTURE
    model_type: str = QUALIFIED_MODEL_TYPE
    default_bank_filename: str = "expert_bank.bin"
    language_model_only: bool = False


_MODEL_SPECS = (
    SupportedModel(
        model="unsloth/Qwen3.6-35B-A3B-NVFP4",
        routed_experts_model="unsloth/Qwen3.6-35B-A3B-NVFP4",
        routed_expert_format="nvfp4",
        hidden_size=2048,
        num_layers=40,
        num_experts=256,
        top_k=8,
        moe_intermediate_size=512,
    ),
    SupportedModel(
        model="unsloth/Qwen3.5-122B-A10B-NVFP4",
        routed_experts_model="unsloth/Qwen3.5-122B-A10B-NVFP4",
        routed_expert_format="nvfp4",
        hidden_size=3072,
        num_layers=48,
        num_experts=256,
        top_k=8,
        moe_intermediate_size=1024,
    ),
    SupportedModel(
        model="srswti/axe-superveloce-88b-nvfp4a16",
        routed_experts_model="srswti/axe-superveloce-88b-int4",
        routed_expert_format="gptq-int4-group128",
        hidden_size=3072,
        num_layers=48,
        num_experts=180,
        top_k=8,
        moe_intermediate_size=1024,
        default_bank_filename="expert_bank_int4.bin",
        language_model_only=True,
    ),
)

SUPPORTED_MODELS = {spec.model: spec for spec in _MODEL_SPECS}
QUALIFIED_MODELS = tuple(SUPPORTED_MODELS)
# Legacy aliases describe the default 35B and remain stable for existing tools.
QUALIFIED_MODEL = QUALIFIED_MODELS[0]
QUALIFIED_HIDDEN_SIZE = 2048
QUALIFIED_LAYERS = 40
QUALIFIED_EXPERTS = 256
QUALIFIED_TOP_K = 8
QUALIFIED_MOE_INTERMEDIATE = 512
QUALIFIED_BANK_LAYERS = 32
QUALIFIED_BANK_EXPERTS_PER_LAYER = 256
QUALIFIED_FP8_CUDA_ONLY_LAYERS = 8


def supported_model(model: str) -> SupportedModel:
    """Return the first-class dense/routed checkpoint contract."""
    try:
        return SUPPORTED_MODELS[model]
    except KeyError as exc:
        raise QualificationError(
            f"unqualified model: {model!r} (admitted: {list(QUALIFIED_MODELS)})"
        ) from exc


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
    bank_source_expert_ids: tuple[int, ...] = ()
    routed_experts_model: str | None = None
    routed_expert_format: str = "nvfp4"

    def __post_init__(self) -> None:
        if not 0 <= self.bank_layers <= self.num_layers:
            raise ValueError(
                f"bank_layers={self.bank_layers} outside 0..{self.num_layers}"
            )
        if not 0 <= self.bank_experts_per_layer <= self.num_experts:
            raise ValueError(
                f"bank_experts_per_layer={self.bank_experts_per_layer} "
                f"outside 0..{self.num_experts}"
            )
        if self.bank_source_expert_ids:
            if len(self.bank_source_expert_ids) != self.bank_experts_per_layer:
                raise ValueError(
                    "bank_source_expert_ids length does not match "
                    f"bank_experts_per_layer={self.bank_experts_per_layer}"
                )
            if tuple(sorted(set(self.bank_source_expert_ids))) != (
                self.bank_source_expert_ids
            ):
                raise ValueError(
                    "bank_source_expert_ids must be unique and increasing"
                )
            if not all(
                0 <= expert < self.num_experts
                for expert in self.bank_source_expert_ids
            ):
                raise ValueError("bank_source_expert_ids contain an invalid ID")

    @property
    def b70_capable_layers(self) -> frozenset[int]:
        """Absolute layers whose routed experts exist in the selected bank."""
        return frozenset(range(self.bank_layers))


def phase4_enabled() -> bool:
    """Whether this process explicitly selected the Phase-4 local adapter."""
    return (
        os.environ.get("SHOOTING_BRAKE_PHASE4") == "all-cuda"
        and os.environ.get("SHOOTING_BRAKE_MODEL") in QUALIFIED_MODELS
    )


def bank_path(model: str | None = None) -> str:
    """Resolve the routed-expert bank independently of dense model weights."""
    from pathlib import Path

    if model is None:
        model = os.environ.get("SHOOTING_BRAKE_MODEL", QUALIFIED_MODEL)
    spec = SUPPORTED_MODELS.get(model)
    filename = (
        spec.default_bank_filename if spec is not None else "expert_bank.bin"
    )
    default = Path(__file__).resolve().parents[3] / "phase1" / filename
    return os.environ.get("SHOOTING_BRAKE_B70_BANK", str(default))


# Legacy NVFP4 header; int4 uses the canonical variable-header module.
_NVFP4_BANK_HEADER_FMT = "<8sIIIIIQQQQ"
_NVFP4_BANK_MAGIC = b"SBEXP001"


@dataclass(frozen=True)
class BankHeader:
    """Format and geometry the routed-expert bank was built for."""

    layers: int
    experts_per_layer: int
    hidden_size: int
    moe_intermediate_size: int
    format: str = "nvfp4"
    version: int = 1
    group_size: int | None = None
    bits: int = 4
    zero_point: int | None = None
    expert_bytes: int | None = None
    source_layers: int | None = None
    source_experts_per_layer: int | None = None
    source_expert_ids: tuple[int, ...] = ()
    data_offset: int | None = None
    resident_set_shared_across_layers: int | None = None

    @property
    def logical_experts_per_layer(self) -> int:
        return self.source_experts_per_layer or self.experts_per_layer

    @staticmethod
    def absent() -> "BankHeader":
        return BankHeader(0, 0, 0, 0, format="absent")


def read_bank_header(path: str | None = None) -> BankHeader:
    """Read either supported bank header without touching accelerator state."""
    import struct
    from pathlib import Path

    if path is None:
        path = bank_path()
    bank = Path(path)
    if not bank.is_file():
        if os.environ.get("SHOOTING_BRAKE_HYBRID") == "1":
            raise QualificationError(
                f"expert bank not found at {bank} — an offloading placement "
                "cannot be served without it. Set SHOOTING_BRAKE_B70_BANK to "
                "the bank built for this model."
            )
        return BankHeader.absent()

    with bank.open("rb") as file:
        magic = file.read(8)
        file.seek(0)
        if magic == _NVFP4_BANK_MAGIC:
            size = struct.calcsize(_NVFP4_BANK_HEADER_FMT)
            raw = file.read(size)
            if len(raw) != size:
                raise QualificationError(f"{bank} has a truncated NVFP4 header")
            _, layers, experts, hidden, inter, *_ = struct.unpack(
                _NVFP4_BANK_HEADER_FMT, raw
            )
            return BankHeader(
                int(layers), int(experts), int(hidden), int(inter),
                format="nvfp4",
            )
        if magic == INT4_BANK_MAGIC:
            try:
                header = read_int4_bank_header(bank)
            except ValueError as exc:
                raise QualificationError(str(exc)) from exc
            return BankHeader(
                header.num_layers,
                header.experts_per_layer,
                header.hidden,
                header.moe_intermediate,
                format="gptq-int4-group128",
                version=INT4_BANK_VERSION,
                group_size=header.group_size,
                bits=header.bits,
                zero_point=header.zero_point,
                expert_bytes=header.expert_stride_bytes,
                source_layers=header.source_num_layers,
                source_experts_per_layer=header.source_experts_per_layer,
                source_expert_ids=header.source_expert_ids,
                data_offset=header.data_offset,
                resident_set_shared_across_layers=(
                    header.resident_set_shared_across_layers
                ),
            )
    raise QualificationError(f"{bank} is not a supported Shooting Brake bank")


def _architectures(hf_config: object) -> Iterable[str]:
    architectures = getattr(hf_config, "architectures", ())
    return architectures if isinstance(architectures, (list, tuple)) else ()


def _summarize_expert_ids(expert_ids: Iterable[int]) -> str:
    ids = tuple(sorted(expert_ids))
    if len(ids) <= 8:
        sample = list(ids)
    else:
        sample = [*ids[:4], "...", *ids[-4:]]
    return f"size={len(ids)}, ids={sample}"


def validate_int4_layer_ownership(
    *,
    layer: int,
    num_experts: int,
    cuda_expert_ids: Iterable[int],
    b70_expert_ids: Iterable[int],
    bank_source_expert_ids: Iterable[int],
) -> None:
    """Prove one layer neither drops nor double-counts an int4 route."""
    cuda_ids = set(cuda_expert_ids)
    b70_ids = set(b70_expert_ids)
    bank_ids = set(bank_source_expert_ids)
    if b70_ids != bank_ids:
        raise QualificationError(
            f"layer {layer} B70 ownership does not match the int4 bank: "
            f"placement {_summarize_expert_ids(b70_ids)}; "
            f"bank {_summarize_expert_ids(bank_ids)}"
        )

    overlap = cuda_ids & bank_ids
    if overlap:
        raise QualificationError(
            f"layer {layer} CUDA/B70 ownership overlaps: "
            f"{_summarize_expert_ids(overlap)}"
        )

    model_ids = set(range(num_experts))
    covered = cuda_ids | bank_ids
    missing = model_ids - covered
    extra = covered - model_ids
    if missing or extra:
        raise QualificationError(
            f"layer {layer} CUDA/B70 ownership has a coverage gap: "
            f"missing {_summarize_expert_ids(missing)}; "
            f"extra {_summarize_expert_ids(extra)}"
        )


def validate_int4_hybrid_contract(
    spec: SupportedModel,
    bank: BankHeader,
    placement: object,
) -> None:
    """Validate the opt-in SBINT401 bank against its exact route owners."""
    if bank.format != "gptq-int4-group128":
        raise QualificationError(
            f"int4 hybrid requires SBINT401, got bank format {bank.format!r}"
        )
    if bank.version != INT4_BANK_VERSION:
        raise QualificationError(
            f"int4 hybrid requires SBINT401 version {INT4_BANK_VERSION}, "
            f"got version {bank.version}"
        )

    expected = {
        "hidden_size": spec.hidden_size,
        "moe_intermediate_size": spec.moe_intermediate_size,
        "group_size": 128,
        "bits": 4,
        "zero_point": 8,
        "source_layers": spec.num_layers,
        "source_experts_per_layer": spec.num_experts,
        "resident_set_shared_across_layers": 1,
    }
    actual = {
        "hidden_size": bank.hidden_size,
        "moe_intermediate_size": bank.moe_intermediate_size,
        "group_size": bank.group_size,
        "bits": bank.bits,
        "zero_point": bank.zero_point,
        "source_layers": bank.source_layers,
        "source_experts_per_layer": bank.source_experts_per_layer,
        "resident_set_shared_across_layers": (
            bank.resident_set_shared_across_layers
        ),
    }
    wrong = {
        name: (actual[name], value)
        for name, value in expected.items()
        if actual[name] != value
    }
    if wrong:
        detail = ", ".join(
            f"{name} bank={got} expected={want}"
            for name, (got, want) in wrong.items()
        )
        raise QualificationError(f"int4 hybrid bank geometry mismatch: {detail}")

    remote_indices = placement.remote_device_indices()
    if remote_indices != (0,):
        raise QualificationError(
            "step-1 int4 hybrid requires exactly B70 device 0; placement "
            f"uses {remote_indices}"
        )
    if placement.cpu_count():
        raise QualificationError(
            "step-1 int4 hybrid does not support CPU-owned experts"
        )

    resident_ids = bank.source_expert_ids
    if not resident_ids:
        raise QualificationError("SBINT401 bank has no source expert IDs")
    for layer in range(spec.num_layers):
        if layer >= bank.layers:
            raise QualificationError(
                f"int4 bank covers {bank.layers} layers, but placement "
                f"offloads layer {layer}"
            )
        b70_ids = tuple(
            expert
            for expert, owner in enumerate(placement.owners[layer])
            if owner.device.value == "b70" and owner.device_index == 0
        )
        cuda_ids = tuple(
            expert
            for expert, owner in enumerate(placement.owners[layer])
            if owner.device.value == "cuda"
        )
        validate_int4_layer_ownership(
            layer=layer,
            num_experts=spec.num_experts,
            cuda_expert_ids=cuda_ids,
            b70_expert_ids=b70_ids,
            bank_source_expert_ids=resident_ids,
        )


def _language_model_only(model_config: object) -> bool:
    """Read vLLM's effective text-only state across supported config shapes."""
    multimodal_config = getattr(model_config, "multimodal_config", None)
    # vLLM 0.27.1 stores language_model_only as an InitVar, so ModelConfig
    # has no post-init attribute to read. For this registry entry the already
    # validated effective architecture is the text-only
    # Qwen3_5MoeForConditionalGeneration class; with no multimodal config
    # there is no vision module path to construct.
    if multimodal_config is None:
        return True
    return getattr(multimodal_config, "language_model_only", False) is True


def require_qualified_config(vllm_config: object) -> QualifiedModel:
    """Fail closed unless this is the frozen eager, single-rank Qwen setup."""
    model_config = getattr(vllm_config, "model_config", None)
    parallel_config = getattr(vllm_config, "parallel_config", None)
    if model_config is None or parallel_config is None:
        raise QualificationError("missing vLLM model or parallel configuration")

    model = getattr(model_config, "model", None)
    if not isinstance(model, str):
        raise QualificationError(f"model name is missing or invalid: {model!r}")
    spec = supported_model(model)
    # Graph mode is supported: the adapter's forward_modular is a pure
    # pass-through in all-CUDA mode (no Python logic on the hot path).

    hf_config = getattr(model_config, "hf_config", None)
    text_config = getattr(model_config, "hf_text_config", None)
    multimodal_config = getattr(model_config, "multimodal_config", None)
    global _text_only_audit_logged
    if not _text_only_audit_logged:
        logger.warning(
            "Shooting Brake effective text-only config: architectures=%r "
            "multimodal_config=%r language_model_only=%r "
            "limit_per_prompt=%r",
            _architectures(hf_config),
            multimodal_config,
            getattr(multimodal_config, "language_model_only", None),
            getattr(multimodal_config, "limit_per_prompt", None),
        )
        _text_only_audit_logged = True
    if spec.architecture not in _architectures(hf_config):
        raise QualificationError("unqualified model architecture")
    if getattr(text_config, "model_type", None) != spec.model_type:
        raise QualificationError("unqualified Qwen text model type")
    if spec.language_model_only and not _language_model_only(model_config):
        raise QualificationError(
            f"{model} must run text-only with language_model_only=True"
        )

    geometry = {}
    expected_geometry = {
        "hidden_size": spec.hidden_size,
        "num_hidden_layers": spec.num_layers,
        "num_experts": spec.num_experts,
        "num_experts_per_tok": spec.top_k,
        "moe_intermediate_size": spec.moe_intermediate_size,
    }
    for name, expected in expected_geometry.items():
        value = getattr(text_config, name, None)
        if value != expected:
            raise QualificationError(
                f"{model} has {name}={value!r}, expected {expected} from the "
                "supported-model registry"
            )
        geometry[name] = value

    # Validate K dimensions against the routed checkpoint's format, not the
    # dense checkpoint's NVFP4 format.
    alignment = 16 if spec.routed_expert_format == "nvfp4" else 128
    for name in ("hidden_size", "moe_intermediate_size"):
        if geometry[name] % alignment:
            raise QualificationError(
                f"{name}={geometry[name]} is not a multiple of {alignment}; "
                f"{spec.routed_expert_format} cannot represent it"
            )


    # Identity, not a bound. A stale 35B bank (32 layers, hidden 2048,
    # intermediate 512) satisfies `32 <= 47` against the 122B and would be
    # served happily, with the B70 returning experts of the wrong shape for
    # every routed token. Comparing the shape the bank was built for
    # against the model's own geometry is what makes that impossible.
    bank = read_bank_header(bank_path(model))
    bank_layers = bank.layers
    if bank_layers:
        if bank.format != spec.routed_expert_format:
            raise QualificationError(
                f"expert bank format {bank.format!r} does not match routed "
                f"checkpoint format {spec.routed_expert_format!r}"
            )
        # Every dimension must match exactly. The layer count is the one
        # exception, and only downward: the FP8 tail is deliberately absent
        # from the bank, so a bank covering fewer layers than the model is
        # normal, while one covering more is a different model's bank.
        wrong: dict[str, tuple[int, int]] = {}
        if bank_layers > geometry["num_hidden_layers"]:
            wrong["layers"] = (bank_layers, geometry["num_hidden_layers"])
        if (
            bank.source_layers is not None
            and bank.source_layers != geometry["num_hidden_layers"]
        ):
            wrong["source_layers"] = (
                bank.source_layers, geometry["num_hidden_layers"]
            )
        for name, got, want in (
            ("experts_per_layer", bank.logical_experts_per_layer,
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

    qualified = QualifiedModel(
        model=model,
        architecture=spec.architecture,
        hidden_size=geometry["hidden_size"],
        num_layers=geometry["num_hidden_layers"],
        num_experts=geometry["num_experts"],
        top_k=geometry["num_experts_per_tok"],
        moe_intermediate_size=geometry["moe_intermediate_size"],
        bank_layers=bank_layers,
        bank_experts_per_layer=(
            bank.experts_per_layer if bank_layers else geometry["num_experts"]
        ),
        bank_source_expert_ids=bank.source_expert_ids,
        routed_experts_model=spec.routed_experts_model,
        routed_expert_format=spec.routed_expert_format,
    )

    if (
        os.environ.get("SHOOTING_BRAKE_HYBRID") == "1"
        and spec.routed_expert_format == "gptq-int4-group128"
    ):
        if os.environ.get("SHOOTING_BRAKE_B70_INT4") != "1":
            raise QualificationError(
                "int4 hybrid requires SHOOTING_BRAKE_B70_INT4=1"
            )
        from .placement import build_placement, policy_from_name

        policy_name = os.environ.get("SHOOTING_BRAKE_PLACEMENT", "all-cuda")
        placement = build_placement(
            policy_from_name(policy_name),
            num_layers=qualified.num_layers,
            num_experts=qualified.num_experts,
            b70_capable=qualified.b70_capable_layers,
        )
        validate_int4_hybrid_contract(spec, bank, placement)

    return qualified
