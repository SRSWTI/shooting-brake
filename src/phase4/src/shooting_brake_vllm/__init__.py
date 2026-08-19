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


def _patch_nested_causallm_naming() -> None:
    """Accept CausalLM exports with ConditionalGeneration tensor naming.

    The 99B checkpoint declares ``Qwen3_5MoeForCausalLM`` but names its
    tensors ``model.language_model.*`` — the nested form the quant pipeline
    inherited from the vision-wrapped export. vLLM's CausalLM classes load
    ``model.*`` and reject the nested prefix outright.

    Compose a prefix rewrite in front of the base loader. For correctly
    named checkpoints the rule matches nothing, so this is a no-op — safe
    to install unconditionally.
    """
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLMBase
    from vllm.model_executor.models.utils import WeightsMapper

    original = Qwen3_5ForCausalLMBase.load_weights
    if getattr(original, "_shooting_brake_nested_naming", False):
        return
    mapper = WeightsMapper(
        orig_to_new_prefix={"model.language_model.": "model."}
    )

    def patched(self, weights):  # type: ignore[no-untyped-def]
        return original(self, mapper.apply(weights))

    patched._shooting_brake_nested_naming = True  # type: ignore[attr-defined]
    Qwen3_5ForCausalLMBase.load_weights = patched


def _patch_nested_quant_ignore() -> None:
    """Normalize the quant config's ignore list for the same nested export.

    The 99B's ``quantization_config.ignore`` names 348 unquantized modules
    with the ``model.language_model.`` prefix. vLLM's CausalLM modules are
    named ``model.*``, so none matched: every module the recipe left in
    BF16 (linear_attn, routers, shared_expert_gate) was constructed
    QUANTIZED and then choked on the checkpoint's plain ``weight`` tensors.

    Rewrite the prefix at config parse. Entries without it — including
    ``re:`` patterns — pass through untouched, so correctly named
    checkpoints are unaffected.
    """
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
        CompressedTensorsConfig,
    )

    original = CompressedTensorsConfig.from_config
    if getattr(original, "_shooting_brake_nested_ignore", False):
        return
    inner = original.__func__

    def patched(cls, config):  # type: ignore[no-untyped-def]
        ignore = config.get("ignore")
        if isinstance(ignore, list):
            config = dict(config)
            config["ignore"] = [
                entry.replace("model.language_model.", "model.", 1)
                if isinstance(entry, str)
                and entry.startswith("model.language_model.")
                else entry
                for entry in ignore
            ]
        return inner(cls, config)

    patched._shooting_brake_nested_ignore = True  # type: ignore[attr-defined]
    CompressedTensorsConfig.from_config = classmethod(patched)


def register() -> None:
    """Register when a model in the split-checkpoint registry is selected."""
    if not phase4_enabled():
        return

    _patch_force_piecewise()
    _patch_nested_causallm_naming()
    _patch_nested_quant_ignore()

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
