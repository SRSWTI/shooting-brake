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


def _patch_cs_doorbell_arming() -> None:
    """Arm the CS-doorbell path only AFTER warmup + CUDA graph capture.

    The baked step contract is all-47-layers-per-step; vLLM's warmup and
    capture run partial dummy sweeps that violate it and wedged the engine
    four times (kill-bench 25). ``Worker.compile_or_warm_up_model`` is the
    single point that runs once, after capture (or after eager warmup when
    enforce_eager), before any real token -- so the fast path can never
    observe a partial step. If this patch never fires, the pollers stay on
    the classic sweep, which is always correct.
    """
    if os.environ.get("SHOOTING_BRAKE_B70_CS_DOORBELL", "0") in ("", "0"):
        return
    if os.environ.get("SHOOTING_BRAKE_HYBRID") != "1":
        return

    from vllm.logger import init_logger
    from vllm.v1.worker.gpu_worker import Worker

    _logger = init_logger(__name__)
    _orig_warm_up = Worker.compile_or_warm_up_model

    def _patched_warm_up(self: Worker):
        # The executor consumes this return value (compilation times);
        # swallowing it breaks engine init, so pass it through verbatim.
        result = _orig_warm_up(self)
        from .b70_poller import arm_all_cs

        arm_all_cs()
        _logger.info(
            "Shooting Brake: CS doorbell armed post-capture "
            "(SHOOTING_BRAKE_B70_CS_DOORBELL=%s)",
            os.environ.get("SHOOTING_BRAKE_B70_CS_DOORBELL"),
        )
        return result

    Worker.compile_or_warm_up_model = _patched_warm_up


def _patch_seam_trace_step() -> None:
    """Wrap ``Worker.execute_model`` with step timing + a profiler trigger.

    Installed when the seam tracer is armed (SB_SEAM_TRACE=1 or
    /tmp/sb_seam_trace.arm) or when /tmp/sb_torch_prof.enable exists at
    boot. Two duties:

    1. Wall-clock step timing feeding the seam tracer.
    2. In-process torch.profiler trigger: touching /tmp/sb_torch_prof.arm
       at runtime profiles the next 25 steps and writes a chrome trace to
       /tmp/sb_torch_prof/decode_trace.json.gz. This exists because the
       API-server process never loads this plugin, so vLLM's
       /start_profile routes are unreachable on this deployment.
    """
    from .seam_trace import seam_tracer as _st_gate

    prof_enabled = os.path.exists("/tmp/sb_torch_prof.enable")
    if _st_gate() is None and not prof_enabled:
        return

    import time

    from vllm.logger import init_logger
    from vllm.v1.worker.gpu_worker import Worker

    _logger = init_logger(__name__)
    _orig_execute = Worker.execute_model
    _state = {"n": 0, "prof": None, "left": 0}
    _ARM = "/tmp/sb_torch_prof.arm"
    _PROF_STEPS = 25

    def _maybe_profiler():
        import torch

        if _state["prof"] is not None:
            _state["left"] -= 1
            if _state["left"] <= 0:
                prof = _state["prof"]
                _state["prof"] = None
                prof.__exit__(None, None, None)
                out = "/tmp/sb_torch_prof/decode_trace.json.gz"
                os.makedirs("/tmp/sb_torch_prof", exist_ok=True)
                prof.export_chrome_trace(out)
                _logger.info("[step-prof] wrote %s", out)
            return
        if prof_enabled and _state["n"] % 32 == 0 and os.path.exists(_ARM):
            os.remove(_ARM)
            prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                with_stack=False,
            )
            prof.__enter__()
            _state["prof"] = prof
            _state["left"] = _PROF_STEPS
            _logger.info("[step-prof] profiling next %d steps", _PROF_STEPS)

    def _timed_execute(self: Worker, scheduler_output, *args, **kwargs):
        from .seam_trace import seam_tracer

        _state["n"] += 1
        _maybe_profiler()
        tracer = seam_tracer()
        if tracer is None:
            return _orig_execute(self, scheduler_output, *args, **kwargs)
        m_hint = getattr(
            scheduler_output, "total_num_scheduled_tokens", 0
        ) or 0
        tracer.step_begin(int(m_hint))
        t0 = time.perf_counter_ns()
        result = _orig_execute(self, scheduler_output, *args, **kwargs)
        tracer.step_end(t0, int(m_hint))
        return result

    Worker.execute_model = _timed_execute


def _apply_file_env_overrides() -> None:
    """Experiment plumbing: /tmp/sb_env_overrides.json -> os.environ.

    The EngineCore child does not reliably inherit ad-hoc env vars (the
    spawn env is rebuilt; observed 2026-08-25). register() runs in every
    vLLM process before config consumption, so a JSON file of overrides
    applied here reaches placement/config/profiler reads deterministically.
    Absent file = no-op; production is untouched.
    """
    path = "/tmp/sb_env_overrides.json"
    if not os.path.exists(path):
        return
    import json

    from vllm.logger import init_logger

    _logger = init_logger(__name__)
    try:
        with open(path) as fh:
            overrides = json.load(fh)
        for key, value in overrides.items():
            os.environ[str(key)] = str(value)
        _logger.info(
            "Shooting Brake: applied %d env override(s) from %s: %s",
            len(overrides), path, ",".join(sorted(overrides)),
        )
    except Exception as exc:  # noqa: BLE001 - experiment plumbing only
        _logger.warning("Shooting Brake: env override load failed: %s", exc)




def _patch_fused_shared() -> None:
    """Fuse Laguna shared-expert gate/up only under the explicit opt-in."""
    if os.environ.get("SHOOTING_BRAKE_FUSED_SHARED", "0") != "1":
        return

    from vllm.logger import init_logger

    from .fused_shared import install

    install()
    init_logger(__name__).info(
        "Shooting Brake: fused shared-expert patch installed"
    )



def _patch_hidden_capture() -> None:
    """Install the opt-in Laguna DFlash feature capture hook."""
    if not os.environ.get("SB_HIDDEN_CAPTURE_DIR", "").strip():
        return

    from .hidden_capture import install

    install()


def register() -> None:
    """Register when a model in the split-checkpoint registry is selected."""
    if not phase4_enabled():
        return

    _apply_file_env_overrides()
    _patch_force_piecewise()
    _patch_nested_causallm_naming()
    _patch_nested_quant_ignore()
    _patch_cs_doorbell_arming()
    _patch_seam_trace_step()
    _patch_fused_shared()
    _patch_hidden_capture()

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
