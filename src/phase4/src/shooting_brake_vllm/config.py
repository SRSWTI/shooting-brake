"""Phase-4 admission and split-checkpoint model metadata."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Sequence
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
    # Absolute transformer layers that own a routed-expert module at all.
    # Empty means "every layer", which is true for the Qwen specs above.
    # Laguna models carry a DENSE MLP at layer 0 -- vLLM constructs no MoE
    # block there, so it has no routed module, no bank row, no poller
    # registration and no route observation. That is a different condition
    # from a routed layer that merely cannot offload (the 99B's FP8 layers
    # 32-39), which is still a real RoutedExperts instance.
    routed_layer_ids: tuple[int, ...] = ()
    # Absolute layers whose experts are present in the bank, in bank-row
    # order. The bank is stored compactly, so row i holds bank_layer_ids[i];
    # for Laguna that is model layer i+1. Empty means the leading prefix
    # range(bank_layers), preserving the Qwen contract exactly.
    bank_layer_ids: tuple[int, ...] = ()


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
    SupportedModel(
        model="srswti/axe-superveloce-99b-nvfp4",
        routed_experts_model="srswti/axe-superveloce-99b-nvfp4",
        routed_expert_format="nvfp4",
        hidden_size=3072,
        num_layers=48,
        num_experts=205,
        top_k=8,
        moe_intermediate_size=1024,
        # The 99B ships as a text-only CausalLM export — no vision wrapper,
        # unlike the 88B's ConditionalGeneration architecture.
        architecture="Qwen3_5MoeForCausalLM",
        default_bank_filename="expert_bank_99b.bin",
        language_model_only=True,
    ),
    SupportedModel(
        # Laguna is a different architecture, not a Qwen variant: 47 sparse
        # MoE layers + 1 dense MLP at layer 0, sliding-window attention
        # instead of GDN, per-head attention gating, and top_k 10.
        # It is NOT `is_hybrid`, so vLLM prefix-caches it -- measured 142x
        # on a repeated 6,023-token prompt where the 99B managed 1.007x.
        model="srswti/axe-superveloce-jota-118b-r20-nvfp4",
        routed_experts_model="srswti/axe-superveloce-jota-118b-r20-nvfp4",
        routed_expert_format="nvfp4",
        hidden_size=3072,
        num_layers=48,
        # 205 experts is deliberate: it matches the 99B exactly, so the
        # fractional placement policy, the resident-set arithmetic and the
        # 76/75 two-card split all carry over. r15 would raise this to 218.
        num_experts=205,
        top_k=10,
        moe_intermediate_size=1024,
        architecture="LagunaForCausalLM",
        model_type="laguna",
        default_bank_filename="expert_bank_jota_118b_r20.bin",
        language_model_only=True,
        routed_layer_ids=tuple(range(1, 48)),
        bank_layer_ids=tuple(range(1, 48)),
    ),
    SupportedModel(
        # Same Laguna architecture as r20, less REAP pruning: 218 experts
        # instead of 205. Every integration mechanism is shared; only the
        # expert count and therefore the bank size (50.65 vs 47.63 GiB) and
        # the remote split (161 = 81/80, vs 151 = 76/75) differ. The
        # fractional placement default resolves to 57 local either way,
        # because round(218 * 0.2634...) == 57.
        model="srswti/axe-superveloce-jota-118b-r15-nvfp4",
        routed_experts_model="srswti/axe-superveloce-jota-118b-r15-nvfp4",
        routed_expert_format="nvfp4",
        hidden_size=3072,
        num_layers=48,
        num_experts=218,
        top_k=10,
        moe_intermediate_size=1024,
        architecture="LagunaForCausalLM",
        model_type="laguna",
        default_bank_filename="expert_bank_jota_118b_r15.bin",
        language_model_only=True,
        routed_layer_ids=tuple(range(1, 48)),
        bank_layer_ids=tuple(range(1, 48)),
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


def _repo_id_from_hub_path(model: str) -> str | None:
    """HF repo id recovered from a hub-cache snapshot path, or ``None``.

    Offline mode (``HF_HUB_OFFLINE=1``) resolves a repo id to its local
    snapshot directory before vLLM ever sees it, so ``model_config.model``
    arrives as ``.../models--org--name/snapshots/<sha>``. The registry is
    keyed by repo id; recover it from the ``models--org--name`` component.
    """
    for part in model.split("/"):
        if part.startswith("models--"):
            pieces = part.split("--")
            if len(pieces) >= 3:
                return f"{pieces[1]}/{'--'.join(pieces[2:])}"
    return None


def supported_model(model: str) -> SupportedModel:
    """Return the first-class dense/routed checkpoint contract."""
    spec = SUPPORTED_MODELS.get(model)
    if spec is None:
        repo_id = _repo_id_from_hub_path(model)
        if repo_id is not None:
            spec = SUPPORTED_MODELS.get(repo_id)
    if spec is None:
        raise QualificationError(
            f"unqualified model: {model!r} (admitted: {list(QUALIFIED_MODELS)})"
        )
    return spec


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
    # Absolute layers owning a routed module, and the absolute layers present
    # in the bank in bank-row order. Empty preserves the legacy contract:
    # every layer routed, bank holds the leading prefix range(bank_layers).
    routed_layer_ids: tuple[int, ...] = ()
    bank_layer_ids: tuple[int, ...] = ()

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
        # Normalise the layer topology before anything reads it. Empty means
        # the legacy Qwen contract, so existing specs are untouched.
        routed = self.routed_layer_ids or tuple(range(self.num_layers))
        banked = self.bank_layer_ids or tuple(range(self.bank_layers))
        for name, ids in (("routed_layer_ids", routed), ("bank_layer_ids", banked)):
            if tuple(sorted(set(ids))) != tuple(ids):
                raise ValueError(f"{name} must be unique and increasing")
            if not all(0 <= layer < self.num_layers for layer in ids):
                raise ValueError(
                    f"{name} contains a layer outside 0..{self.num_layers - 1}"
                )
        if len(banked) != self.bank_layers:
            raise ValueError(
                f"bank_layer_ids has {len(banked)} entries but "
                f"bank_layers={self.bank_layers}"
            )
        orphans = tuple(layer for layer in banked if layer not in frozenset(routed))
        if orphans:
            raise ValueError(f"banked layers {orphans} are not routed layers")
        object.__setattr__(self, "routed_layer_ids", routed)
        object.__setattr__(self, "bank_layer_ids", banked)

    @property
    def b70_capable_layers(self) -> frozenset[int]:
        """Absolute layers whose routed experts exist in the selected bank."""
        return frozenset(self.bank_layer_ids)

    def is_routed_layer(self, layer: int) -> bool:
        """Whether this absolute layer owns a routed-expert module at all.

        False for a dense MLP layer (Laguna's layer 0), which has no
        RoutedExperts instance. This is NOT the same as a routed layer that
        cannot offload -- those are real modules and return True.
        """
        return layer in frozenset(self.routed_layer_ids)

    def bank_row_for_model_layer(self, layer: int) -> int:
        """Translate an absolute model layer to its compact bank row.

        The bank is stored densely, so a model whose first sparse layer is 1
        keeps that layer's experts in row 0. Every crossing into the native
        provider or a compact bank file MUST come through here: passing an
        absolute index straight through reads the neighbouring layer's
        weights and yields plausible wrong output instead of an error.
        """
        try:
            return self.bank_layer_ids.index(layer)
        except ValueError:
            raise KeyError(
                f"layer {layer} has no bank row; bank covers "
                f"{sorted(self.bank_layer_ids)[:1]}..{sorted(self.bank_layer_ids)[-1:]}"
            ) from None

    def model_layer_for_bank_row(self, row: int) -> int:
        """Inverse of `bank_row_for_model_layer`, for traces and diagnostics."""
        if not 0 <= row < len(self.bank_layer_ids):
            raise KeyError(
                f"bank row {row} outside 0..{len(self.bank_layer_ids) - 1}"
            )
        return self.bank_layer_ids[row]


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


def b70_bank_paths(model: str | None = None) -> tuple[str, ...]:
    """Per-card decode banks, position-aligned with the placement's sorted
    remote device indices.

    ``SHOOTING_BRAKE_B70_BANKS`` is a comma-separated path list for
    multi-card configs (position 0 serves the first remote device index).
    Unset falls back to the single legacy bank from :func:`bank_path`, so
    every existing single-card recipe keeps working unchanged.

    Each card needs its own bank holding exactly the expert IDs that card
    owns — the contract check enforces set equality per device, so a
    monolithic bank cannot be shared between two cards.
    """
    raw = os.environ.get("SHOOTING_BRAKE_B70_BANKS")
    if raw is None:
        return (bank_path(model),)
    paths = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not paths:
        raise QualificationError(
            "SHOOTING_BRAKE_B70_BANKS is set but contains no paths"
        )
    return paths


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
    other_remote_expert_ids: Iterable[int] = (),
) -> None:
    """Prove one layer neither drops nor double-counts an int4 route.

    ``b70_expert_ids`` and ``bank_source_expert_ids`` describe ONE remote
    device and its bank. ``other_remote_expert_ids`` are experts owned by
    the *other* remote devices: they count toward layer coverage but must
    stay disjoint from this device's bank — two cards holding the same
    expert would double-count every route to it.
    """
    cuda_ids = set(cuda_expert_ids)
    b70_ids = set(b70_expert_ids)
    bank_ids = set(bank_source_expert_ids)
    other_ids = set(other_remote_expert_ids)
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

    cross_device = bank_ids & other_ids
    if cross_device:
        raise QualificationError(
            f"layer {layer} remote devices overlap — the same expert is "
            f"resident on two cards: {_summarize_expert_ids(cross_device)}"
        )

    model_ids = set(range(num_experts))
    covered = cuda_ids | bank_ids | other_ids
    missing = model_ids - covered
    extra = covered - model_ids
    if missing or extra:
        raise QualificationError(
            f"layer {layer} CUDA/B70 ownership has a coverage gap: "
            f"missing {_summarize_expert_ids(missing)}; "
            f"extra {_summarize_expert_ids(extra)}"
        )


def _validate_int4_bank_header(
    spec: SupportedModel, bank: BankHeader, label: str,
) -> None:
    """One bank's format, version, and geometry against the model spec."""
    if bank.format != "gptq-int4-group128":
        raise QualificationError(
            f"int4 hybrid requires SBINT401, got bank format {bank.format!r} "
            f"for {label}"
        )
    if bank.version != INT4_BANK_VERSION:
        raise QualificationError(
            f"int4 hybrid requires SBINT401 version {INT4_BANK_VERSION}, "
            f"got version {bank.version} for {label}"
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
        raise QualificationError(
            f"int4 hybrid bank geometry mismatch for {label}: {detail}"
        )
    if not bank.source_expert_ids:
        raise QualificationError(
            f"SBINT401 bank for {label} has no source expert IDs"
        )


def validate_int4_hybrid_contract(
    spec: SupportedModel,
    banks: Sequence[BankHeader],
    placement: object,
) -> None:
    """Validate the opt-in SBINT401 banks against their exact route owners.

    ``banks`` is position-aligned with ``placement.remote_device_indices()``
    (the :func:`b70_bank_paths` order). Each card's bank must hold exactly
    the expert IDs that card owns, on every offloaded layer — a subset is
    rejected, and so is any expert resident on two cards.
    """
    remote_indices = placement.remote_device_indices()
    if len(banks) != len(remote_indices):
        raise QualificationError(
            f"placement uses remote devices {remote_indices} but "
            f"{len(banks)} bank(s) were configured; set "
            "SHOOTING_BRAKE_B70_BANKS to one bank per device, in device-"
            "index order"
        )
    if placement.cpu_count():
        raise QualificationError(
            "int4 hybrid does not support CPU-owned experts"
        )
    for index, bank in zip(remote_indices, banks):
        _validate_int4_bank_header(spec, bank, f"B70 device {index}")

    for layer in range(spec.num_layers):
        cuda_ids = tuple(
            expert
            for expert, owner in enumerate(placement.owners[layer])
            if owner.device.value == "cuda"
        )
        b70_ids_by_device = {
            index: tuple(
                expert
                for expert, owner in enumerate(placement.owners[layer])
                if owner.device.value == "b70" and owner.device_index == index
            )
            for index in remote_indices
        }
        for index, bank in zip(remote_indices, banks):
            if layer >= bank.layers:
                raise QualificationError(
                    f"int4 bank for B70 device {index} covers {bank.layers} "
                    f"layers, but placement offloads layer {layer}"
                )
            other_ids = tuple(
                expert
                for other, ids in b70_ids_by_device.items()
                if other != index
                for expert in ids
            )
            validate_int4_layer_ownership(
                layer=layer,
                num_experts=spec.num_experts,
                cuda_expert_ids=cuda_ids,
                b70_expert_ids=b70_ids_by_device[index],
                bank_source_expert_ids=bank.source_expert_ids,
                other_remote_expert_ids=other_ids,
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
        raise QualificationError(
            f"unqualified text model type: expected {spec.model_type!r}, "
            f"got {getattr(text_config, 'model_type', None)!r}"
        )
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
    #
    # Multi-card configs list one decode bank per card in
    # SHOOTING_BRAKE_B70_BANKS; every bank must pass the identity check,
    # and all must agree on layer coverage. The per-card expert-set
    # equality is validated separately by the int4 hybrid contract.
    paths = b70_bank_paths(model)
    banks = tuple(read_bank_header(path) for path in paths)
    for path, bank in zip(paths, banks):
        bank_layers = bank.layers
        if not bank_layers:
            continue
        if bank.format != spec.routed_expert_format:
            raise QualificationError(
                f"expert bank format {bank.format!r} does not match routed "
                f"checkpoint format {spec.routed_expert_format!r} ({path})"
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
                f"{detail} ({path}). Point SHOOTING_BRAKE_B70_BANK / "
                "SHOOTING_BRAKE_B70_BANKS at this model's bank(s)."
            )
    if len({bank.layers for bank in banks}) != 1:
        raise QualificationError(
            "per-card banks disagree on layer coverage: "
            + ", ".join(
                f"{path}={bank.layers}" for path, bank in zip(paths, banks)
            )
        )
    bank_layers = banks[0].layers

    if getattr(parallel_config, "tensor_parallel_size", None) != 1:
        raise QualificationError("Phase 4 requires tensor parallel size one")
    if getattr(parallel_config, "pipeline_parallel_size", None) != 1:
        raise QualificationError("Phase 4 requires pipeline parallel size one")
    if getattr(parallel_config, "enable_eplb", False):
        raise QualificationError("Phase 4 does not admit EPLB")

    # The QualifiedModel's bank view is the UNION across per-card banks:
    # b70_capable_layers and b70_bank_covers() reason about "is this expert
    # reachable off-CUDA at all", which is a union question. Per-card set
    # equality is the hybrid contract's job, not this one's.
    #
    # Two multi-card shapes exist:
    #   * SBINT401: one bank file per card, each a subset with explicit
    #     source IDs — the union is their disjoint sum.
    #   * SBEXP001: ONE monolithic full-coverage bank listed once per card
    #     (per-card subsets are resident lists at provider load, not bank
    #     files) — the union is simply the bank's full expert range.
    if bank_layers and len(banks) > 1:
        with_ids = [bank for bank in banks if bank.source_expert_ids]
        if with_ids and len(with_ids) != len(banks):
            raise QualificationError(
                "multi-card banks mix explicit-source-ID (SBINT401) and "
                "monolithic (SBEXP001) entries; use one shape"
            )
        if with_ids:
            union_ids = tuple(sorted({
                expert for bank in banks
                for expert in bank.source_expert_ids
            }))
            union_count = len(union_ids)
        else:
            counts = {bank.experts_per_layer for bank in banks}
            if len(counts) != 1:
                raise QualificationError(
                    "monolithic multi-card banks disagree on experts per "
                    f"layer: {sorted(counts)}"
                )
            union_ids = ()
            union_count = banks[0].experts_per_layer
    else:
        union_ids = banks[0].source_expert_ids
        union_count = (
            banks[0].experts_per_layer
            if bank_layers else geometry["num_experts"]
        )

    qualified = QualifiedModel(
        model=model,
        architecture=spec.architecture,
        hidden_size=geometry["hidden_size"],
        num_layers=geometry["num_hidden_layers"],
        num_experts=geometry["num_experts"],
        top_k=geometry["num_experts_per_tok"],
        moe_intermediate_size=geometry["moe_intermediate_size"],
        bank_layers=bank_layers,
        bank_experts_per_layer=union_count,
        bank_source_expert_ids=union_ids,
        routed_experts_model=spec.routed_experts_model,
        routed_expert_format=spec.routed_expert_format,
        routed_layer_ids=spec.routed_layer_ids,
        # A zero-layer bank (all-CUDA, nothing offloaded) has no banked
        # layers at all, so the spec's list must not be forwarded -- it
        # would fail the length invariant against bank_layers=0. When a
        # bank IS present its row count must match the spec exactly; a
        # 48-row bank under a 47-sparse-layer model is a build error and
        # is rejected rather than silently shifted by one layer.
        bank_layer_ids=spec.bank_layer_ids if bank_layers else (),
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
        validate_int4_hybrid_contract(spec, banks, placement)

    return qualified
