"""Qualified routed-expert container with Phase-5/6a ownership + partitioning."""

from __future__ import annotations

import importlib
import os
from typing import Any

import numpy as np
import torch
from vllm.config import CUDAGraphMode, get_current_vllm_config
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.logger import init_logger
from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
)
try:
    from vllm.compilation.breakable_cudagraph import eager_break_during_capture
except ImportError:
    def eager_break_during_capture(fn):  # type: ignore[misc]
        return fn

from . import route_stats
from .config import bank_path as _b70_bank_path
from .config import require_qualified_config
from .partition import (
    DispatchBufferGeometry,
    RoutePartition,
    build_cuda_expert_maps,
    build_device_map,
    compact_cuda_routes,
    partition_routes,
    validate_dispatch_buffer_shapes,
    validate_cuda_dummy_slot_placement,
    validate_partition,
)
from .placement import Device, Placement, build_for_qualified
from .provider import ShootingBrakeExpertProviderClient

logger = init_logger(__name__)


def _placement_policy_name() -> str:
    """Placement policy spec from the environment (default: all-CUDA)."""
    return os.environ.get("SHOOTING_BRAKE_PLACEMENT", "all-cuda")


def b70_prefill_stream_enabled() -> bool:
    """Whether prefill computes B70-owned routes on the 5090 from streamed
    weights instead of dispatching them to the B70.

    Off by default. When off, nothing in the streaming path is constructed:
    no host copy of the B70 bank is retained, no arena space is reserved for
    it, and ``_b70_prefill_partial`` keeps its chunked-dispatch behaviour
    exactly.
    """
    return os.environ.get("SHOOTING_BRAKE_B70_PREFILL_STREAM") == "1"


def _validate_dispatch_capture_mode(
    *, graph_mode: bool, stream_capturing: bool,
) -> None:
    """Reject host dispatch before it performs work inside stream capture."""
    if stream_capturing and not graph_mode:
        raise RuntimeError(
            "synchronous B70 dispatch cannot run during CUDA graph capture; "
            "construct the synchronous reference with enforce_eager=True, "
            "or opt into the poller path with SHOOTING_BRAKE_B70_GRAPH=1"
        )


_EXPECTED_ARMS = {
    "doorbell",
    "breakable",
    "full-break-reference",
    "reference",
}


def _validate_expected_arm_configuration(
    *,
    expected_arm: str,
    graph_mode: bool,
    breakable_enabled: bool,
    enforce_eager: bool,
    force_piecewise_fired: bool,
    cudagraph_mode: CUDAGraphMode,
) -> None:
    """Fail before timing when benchmark-arm configuration is ambiguous."""
    if expected_arm not in _EXPECTED_ARMS:
        raise RuntimeError(
            f"unknown SHOOTING_BRAKE_EXPECT_ARM={expected_arm!r}; "
            f"expected one of {sorted(_EXPECTED_ARMS)}"
        )
    expected = {
        "doorbell": (True, False, False, False),
        "breakable": (False, True, False, True),
        "full-break-reference": (True, True, False, True),
        "reference": (False, False, True, False),
    }[expected_arm]
    actual = (
        graph_mode,
        breakable_enabled,
        enforce_eager,
        force_piecewise_fired,
    )
    if actual != expected:
        labels = (
            "graph_mode",
            "breakable_enabled",
            "enforce_eager",
            "force_piecewise_fired",
        )
        expected_detail = ", ".join(
            f"{label}={want!r}"
            for label, want in zip(labels, expected)
        )
        resolved_detail = ", ".join(
            f"{label}={got!r}"
            for label, got in zip(labels, actual)
        )
        raise RuntimeError(
            f"{expected_arm} benchmark arm configuration mismatch: "
            f"expected [{expected_detail}]; resolved [{resolved_detail}]; "
            f"cudagraph_mode={cudagraph_mode.name}"
        )
    if expected_arm == "doorbell":
        if (
            cudagraph_mode is CUDAGraphMode.NONE
            or not cudagraph_mode.has_full_cudagraphs()
        ):
            raise RuntimeError(
                "doorbell benchmark arm requires CUDA graphs with a FULL "
                f"decode mode, got {cudagraph_mode.name}"
            )
    elif expected_arm in {"breakable", "full-break-reference"}:
        if cudagraph_mode is not CUDAGraphMode.PIECEWISE:
            raise RuntimeError(
                f"{expected_arm} benchmark arm requires forced PIECEWISE, "
                f"got {cudagraph_mode.name}"
            )
    elif cudagraph_mode is not CUDAGraphMode.NONE:
        raise RuntimeError(
            "reference benchmark arm requires cudagraph mode NONE, "
            f"got {cudagraph_mode.name}"
        )


def _classify_expected_arm_runtime(
    *,
    expected_arm: str,
    graph_mode: bool,
    runtime_mode: CUDAGraphMode,
    stream_capturing: bool,
) -> str | None:
    """Name and validate the branch exercised by one graph descriptor."""
    if expected_arm != "reference" and runtime_mode is CUDAGraphMode.NONE:
        # Profiling/warmup precedes graph capture and is not a timed graph arm.
        return None
    if expected_arm == "doorbell":
        if runtime_mode not in {
            CUDAGraphMode.FULL,
            CUDAGraphMode.PIECEWISE,
        }:
            raise RuntimeError(
                f"doorbell arm reached unexpected runtime mode "
                f"{runtime_mode.name}"
            )
        if not graph_mode or not stream_capturing:
            raise RuntimeError(
                "doorbell arm did not take doorbell-in-capture branch: "
                f"graph_mode={graph_mode} stream_capturing={stream_capturing}"
            )
        return "doorbell-in-capture"
    if expected_arm == "breakable":
        if (
            runtime_mode is not CUDAGraphMode.PIECEWISE
            or graph_mode
            or stream_capturing
        ):
            raise RuntimeError(
                "breakable arm did not take the eager-break branch: "
                f"runtime_mode={runtime_mode.name} graph_mode={graph_mode} "
                f"stream_capturing={stream_capturing}"
            )
        return "eager-break"
    if expected_arm == "full-break-reference":
        if (
            runtime_mode is not CUDAGraphMode.PIECEWISE
            or not graph_mode
            or stream_capturing
        ):
            raise RuntimeError(
                "full-break reference did not take the doorbell eager gap: "
                f"runtime_mode={runtime_mode.name} graph_mode={graph_mode} "
                f"stream_capturing={stream_capturing}"
            )
        return "doorbell-eager-break"
    if (
        runtime_mode is not CUDAGraphMode.NONE
        or graph_mode
        or stream_capturing
    ):
        raise RuntimeError(
            "reference arm did not take synchronous Python branch: "
            f"runtime_mode={runtime_mode.name} graph_mode={graph_mode} "
            f"stream_capturing={stream_capturing}"
        )
    return "synchronous-python"


def preemptive_surgery_enabled() -> bool:
    """Whether non-CUDA experts are never allocated in VRAM at all.

    Default surgery is post-hoc: vLLM allocates all ``num_experts`` per
    layer, the checkpoint loads into them, and ownership is sliced away
    afterwards. That works only while the *whole* expert bank transiently
    fits in VRAM. It does for the 35B (13.5 GiB) and does not for the 122B
    (59.5 GiB against a 32 GiB card), where the load OOMs long before any
    slice can run -- ``process_weights_after_loading`` does not fire until
    the entire model is on device.

    When enabled, ``create_weights`` allocates only the CUDA-owned experts
    and vLLM's own loader skips the rest, so peak equals steady state. This
    also makes the CUDA expert count a free parameter: post-hoc, it is
    implicitly capped by a transient peak that has nothing to do with the
    configuration being served.
    """
    return os.environ.get("SHOOTING_BRAKE_PREEMPTIVE_SURGERY") == "1"


#: Token count at or above which B70 routes stream instead of dispatching.
#:
#: This counts tokens in the *forward pass*, not tokens per request, and the
#: distinction is the main empirical finding. Sixteen concurrent 256-token
#: prompts batch into one 4096-token forward and stream profitably, while a
#: single 256-token prompt does not -- the same per-request length lands on
#: opposite sides of the crossover depending on load. ``x.shape[0]`` is
#: already the batched count, so the threshold reads the right quantity.
#:
#: Measured, benchmarks/stream_matrix.py, TTFT p50, dispatch / stream:
#:
#:      len  conc      dispatch      stream    ratio
#:      256     1       145.2 ms    319.1 ms    0.45x   dispatch
#:     1024     1       387.1 ms    369.1 ms    1.05x   tie
#:      256    16      1163.7 ms   1015.5 ms    1.15x   stream
#:     1024    16      3645.9 ms   1907.1 ms    1.91x   stream
#:     4096     1      1423.9 ms    706.5 ms    2.02x   stream
#:     4096    16     13127.4 ms   5497.5 ms    2.39x   stream
#:
#: The crossover sits just under 1024 tokens per forward, above the ~311 the
#: cost model predicted from 26.9 us/token/layer against a flat 0.41 GiB
#: transfer. The model overestimated streaming because it assumed every
#: route reaches a distinct expert, so the full bank always moves; at
#: moderate M the touched set is smaller and dispatch stays competitive
#: longer.
#:
#: Default is the measured tie point rather than the analytic crossover.
#: Below it streaming loses badly (0.45x at M=256) while above it the curve
#: is shallow, so the asymmetry favours streaming late.
DEFAULT_B70_STREAM_THRESHOLD = 1024


def b70_stream_threshold() -> int:
    return int(
        os.environ.get(
            "SHOOTING_BRAKE_B70_STREAM_T", str(DEFAULT_B70_STREAM_THRESHOLD)
        )
    )

def _build_b70_slot_map(placement: Placement) -> np.ndarray:
    """Return the checked global→compact map for B70 device zero.

    Step 1 deliberately uses one B70.  Placement already represents multiple
    remote devices, but dispatch does not until the asynchronous multi-device
    step.  Refuse such a placement here instead of collapsing device-local
    slots from two cards into one plausible-looking map.
    """
    indices = placement.remote_device_indices()
    if indices not in ((), (0,)):
        raise RuntimeError(
            "B70 dispatch currently supports exactly device 0; placement "
            f"assigns remote device indices {indices}"
        )

    slot_map = np.full(placement.num_experts, -1, dtype=np.int32)
    active = placement.b70_active_layers()
    if not active:
        return slot_map

    reference_layer = active[0]
    reference_ids = placement.b70_expert_ids(reference_layer)
    for layer in active[1:]:
        ids = placement.b70_expert_ids(layer)
        if ids != reference_ids:
            raise RuntimeError(
                "one shared B70 bank requires the same resident expert IDs "
                f"on every active layer; layer {reference_layer} has "
                f"{reference_ids[:4]}...{reference_ids[-4:]}, layer {layer} "
                f"has {ids[:4]}...{ids[-4:]}"
            )

    for expert_id, owner in enumerate(placement.owners[reference_layer]):
        if owner.device is Device.B70:
            if owner.device_index != 0:
                raise RuntimeError(
                    f"expert {expert_id} targets unsupported B70 device "
                    f"{owner.device_index}"
                )
            slot_map[expert_id] = owner.slot

    compact = slot_map[slot_map >= 0]
    if not np.array_equal(
        np.sort(compact), np.arange(len(compact), dtype=np.int32),
    ):
        raise RuntimeError("B70 compact slots are not dense from zero")
    return slot_map


_b70_provider_singleton: Any = None


def _get_b70_provider(placement: Placement) -> Any:
    """Lazily create and cache the in-process B70 provider singleton."""
    global _b70_provider_singleton
    if _b70_provider_singleton is not None:
        return _b70_provider_singleton

    from .b70_binding import B70ProviderClient

    lib_path = os.environ.get(
        "SHOOTING_BRAKE_B70_LIB", "src/phase7/libsb_b70_provider.so"
    )
    bank_path = _b70_bank_path()
    # Every B70-active layer offloads the same expert ids — one resident
    # set covers the whole bank — but layer 0 is not necessarily active
    # (a subset policy may leave it entirely on CUDA), so read the set
    # from the first layer that actually owns B70 experts.
    reference_layer = placement.b70_active_layers()[0]
    resident = sorted(
        e for e in range(placement.num_experts)
        if placement.owners[reference_layer][e].device is Device.B70
    )
    resident_np = np.array(resident, dtype=np.int32)
    max_batch = int(os.environ.get("SHOOTING_BRAKE_B70_MAX_BATCH", "128"))

    logger.info(
        "Shooting Brake Phase-7: initializing B70 provider "
        "(%d resident experts/layer)...", len(resident)
    )
    provider = B70ProviderClient(lib_path)
    provider.load(bank_path, generation=1, resident_experts=resident_np,
                  max_batch=max_batch)
    _b70_provider_singleton = provider
    logger.info(
        "Shooting Brake Phase-7: B70 provider ready "
        "(resident_per_layer=%d)", provider.resident_per_layer
    )
    return provider


def _build_cpu_id_map(placement: Placement) -> np.ndarray:
    """[num_experts] int32: global expert id if CPU-owned, else -1.

    The CPU arena is keyed by ``(layer, global_expert)`` rather than by a
    compact slot, so unlike the B70 map this is identity-or-sentinel rather
    than a renumbering. Gathering it with ``topk_ids`` yields exactly the id
    vector ``sb_cpu_moe_forward`` expects: CPU routes carry their own id,
    every other route carries -1 and is skipped.

    As with the B70 map, every active layer offloads the same expert ids, so
    one map serves them all; read it from the first layer that owns any.
    """
    id_map = np.full(placement.num_experts, -1, dtype=np.int32)
    active = placement.cpu_active_layers()
    if not active:
        return id_map
    for expert_id, owner in enumerate(placement.owners[active[0]]):
        if owner.device is Device.CPU:
            id_map[expert_id] = expert_id
    return id_map


#: All-out mode logs one warning per process, not one per layer: every layer
#: constructs the same state, and 40 identical lines bury the rest of startup.
_all_out_warned = False

_SURGERY_HOOK_ATTR = "_shooting_brake_surgery_hook"


def _reject_unsupported_preemptive_tiers(placement: Placement) -> None:
    """Require a readable expert bank whenever a host tier is in play.

    ``_load_host_experts`` used to fill the arena by ``index_select``-ing
    the layer's weight tensors with global expert ids, which pre-emptive
    allocation breaks: the offloaded experts are never created in VRAM, so
    the call either indexes out of bounds or quietly addresses a different
    expert. That combination was refused outright.

    It no longer needs to be. ``_load_host_experts_from_bank`` reads those
    experts from the bank, which holds all of them and is built offline
    from the checkpoint — verified byte-for-byte against the VRAM path on
    the 35B at ``allout:16:8:8`` across all 128 host experts.

    What remains is that path's precondition: the bank must exist. Without
    it the tier would load nothing and the routes it owns would contribute
    zero, which is plausible tokens rather than an error.
    """
    offloaded = placement.b70_count() + placement.cpu_count()
    if offloaded and os.environ.get("SHOOTING_BRAKE_HYBRID") != "1":
        raise RuntimeError(
            "SHOOTING_BRAKE_PREEMPTIVE_SURGERY=1 with offloaded experts "
            "requires SHOOTING_BRAKE_HYBRID=1; otherwise compacted routes "
            "would never receive their remote partial"
        )
    if offloaded and os.environ.get("SHOOTING_BRAKE_B70_DEVICE") != "1":
        raise RuntimeError(
            "SHOOTING_BRAKE_PREEMPTIVE_SURGERY=1 with offloaded experts "
            "requires SHOOTING_BRAKE_B70_DEVICE=1"
        )
    if placement.cpu_count() == 0 and not b70_prefill_stream_enabled():
        return
    from .config import read_bank_header

    if read_bank_header().layers == 0:
        raise RuntimeError(
            "SHOOTING_BRAKE_PREEMPTIVE_SURGERY=1 with a host tier needs an "
            f"expert bank: none found at {_b70_bank_path()}. Offloaded "
            "experts are never placed in VRAM under pre-emptive allocation, "
            "so the bank is their only source."
        )


def _preemptive_cuda_ids(layer: Any) -> tuple[int, ...] | None:
    """CUDA-owned global expert ids for ``layer``, or ``None`` to allocate
    the full bank.

    Returns ``None`` for any layer the placement leaves entirely on CUDA, so
    an all-CUDA layer (the FP8 tail) keeps stock behaviour and stock
    indexing rather than a compact map that happens to be the identity.
    """
    from vllm.model_executor.models.utils import extract_layer_index

    placement = layer.shooting_brake_placement
    layer_idx = extract_layer_index(layer.layer_name)
    if placement.layer_b70_count(layer_idx) + placement.layer_cpu_count(
        layer_idx
    ) == 0:
        return None
    return placement.cuda_expert_ids(layer_idx)


_ALLOC_HOOK_ATTR = "_shooting_brake_alloc_hook"
_PREEMPTIVE_METHOD_CANDIDATES = (
    (
        "vllm.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4",
        "CompressedTensorsW4A4Nvfp4MoEMethod",
    ),
    (
        "vllm.model_executor.layers.quantization.modelopt",
        "ModelOptNvFp4FusedMoE",
    ),
)
_preemptive_patched_methods: tuple[str, ...] = ()
_preemptive_alloc_invocations: dict[int, tuple[str, str]] = {}
_preemptive_validation_logged = False


def install_preemptive_alloc_hook() -> None:
    """Make ``create_weights`` allocate only CUDA-owned experts.

    Patched on the quant-method *class*, not an instance: ``create_weights``
    runs inside ``RoutedExperts.__init__`` before the adapter regains
    control, so there is no instance to wrap in time. ``layer.layer_name``
    is already set at that point, which is what makes the placement lookup
    possible this early.

    Both qualified NVFP4 loaders are covered. vLLM 0.26 used the
    compressed-tensors method; vLLM 0.27 routes this checkpoint through
    ``ModelOptNvFp4FusedMoE``. Missing either class silently restores full
    180-expert allocation and OOMs before post-load surgery can run.

    The original method still owns auxiliary tensor semantics. In particular,
    ModelOpt receives the unchanged ``global_num_experts`` keyword and may
    retain global-width activation scales while allocating expert weights at
    the compact CUDA count.
    """
    global _preemptive_patched_methods, _preemptive_validation_logged
    methods: list[type[Any]] = []
    method_names: list[str] = []
    for module_name, class_name in _PREEMPTIVE_METHOD_CANDIDATES:
        try:
            module = importlib.import_module(module_name)
            method = getattr(module, class_name)
        except (ImportError, AttributeError):
            continue
        methods.append(method)
        method_names.append(f"{module_name}.{class_name}")
    if not methods:
        raise RuntimeError(
            "no supported NVFP4 MoE create_weights class exists; candidates="
            f"{_PREEMPTIVE_METHOD_CANDIDATES!r}"
        )
    _preemptive_patched_methods = tuple(method_names)
    _preemptive_alloc_invocations.clear()
    _preemptive_validation_logged = False

    def cuda_ids_for(layer: Any) -> tuple[int, ...] | None:
        if not isinstance(layer, HybridRoutedExperts):
            return None
        return _preemptive_cuda_ids(layer)

    def finish_compact_allocation(
        quant_method: Any,
        layer: Any,
        global_num_experts: int,
        cuda_ids: tuple[int, ...],
    ) -> None:
        _preemptive_alloc_invocations[id(layer)] = (
            layer.layer_name,
            type(quant_method).__name__,
        )
        layer.local_num_experts = len(cuda_ids)
        layer._shooting_brake_cuda_ids = cuda_ids
        expert_map = torch.full(
            (global_num_experts,), -1, dtype=torch.int32
        )
        for local_id, global_id in enumerate(cuda_ids):
            expert_map[global_id] = local_id
        layer.expert_map_manager._expert_map = expert_map
        layer._expert_map = expert_map
        logger.info(
            "Shooting Brake pre-emptive surgery: %s — allocated %d/%d "
            "experts, %d never enter VRAM via %s",
            layer.layer_name,
            len(cuda_ids),
            global_num_experts,
            global_num_experts - len(cuda_ids),
            type(quant_method).__name__,
        )

    def wrap_compressed_tensors(original: Any) -> Any:
        def create_weights(
            self: Any,
            layer: Any,
            num_experts: int,
            hidden_size: int,
            intermediate_size_per_partition: int,
            params_dtype: torch.dtype,
            **extra_weight_attrs: Any,
        ) -> None:
            cuda_ids = cuda_ids_for(layer)
            local_num_experts = (
                num_experts if cuda_ids is None else len(cuda_ids)
            )
            original(
                self,
                layer=layer,
                num_experts=local_num_experts,
                hidden_size=hidden_size,
                intermediate_size_per_partition=(
                    intermediate_size_per_partition
                ),
                params_dtype=params_dtype,
                **extra_weight_attrs,
            )
            if cuda_ids is not None:
                finish_compact_allocation(
                    self, layer, num_experts, cuda_ids
                )

        return create_weights

    def wrap_modelopt(original: Any) -> Any:
        def create_weights(
            self: Any,
            layer: Any,
            num_experts: int,
            hidden_size: int,
            intermediate_size_per_partition: int,
            params_dtype: torch.dtype,
            **extra_weight_attrs: Any,
        ) -> None:
            cuda_ids = cuda_ids_for(layer)
            local_num_experts = (
                num_experts if cuda_ids is None else len(cuda_ids)
            )
            # Keep global_num_experts in extra_weight_attrs unchanged: ModelOpt
            # uses it to preserve 180-wide global input scales.
            original(
                self,
                layer=layer,
                num_experts=local_num_experts,
                hidden_size=hidden_size,
                intermediate_size_per_partition=(
                    intermediate_size_per_partition
                ),
                params_dtype=params_dtype,
                **extra_weight_attrs,
            )
            if cuda_ids is not None:
                finish_compact_allocation(
                    self, layer, num_experts, cuda_ids
                )

        return create_weights

    wrapper_builders = {
        "CompressedTensorsW4A4Nvfp4MoEMethod": wrap_compressed_tensors,
        "ModelOptNvFp4FusedMoE": wrap_modelopt,
    }
    logger.info(
        "Shooting Brake pre-emptive allocation classes present: %s",
        ", ".join(_preemptive_patched_methods),
    )
    for method in methods:
        if getattr(method, _ALLOC_HOOK_ATTR, False):
            continue
        method.create_weights = wrapper_builders[method.__name__](
            method.create_weights
        )
        setattr(method, _ALLOC_HOOK_ATTR, True)


def _validate_preemptive_allocations(layer: Any) -> None:
    """Prove every remotely-owned layer took a compact allocation path."""
    global _preemptive_validation_logged
    if not preemptive_surgery_enabled():
        return
    placement = layer.shooting_brake_placement
    expected_layers = set(placement.b70_active_layers())
    expected_layers.update(placement.cpu_active_layers())
    if not expected_layers:
        return

    from vllm.model_executor.models.utils import extract_layer_index

    actual_layers = {
        extract_layer_index(layer_name)
        for layer_name, _ in _preemptive_alloc_invocations.values()
    }
    if (
        len(_preemptive_alloc_invocations) != len(expected_layers)
        or actual_layers != expected_layers
    ):
        raise RuntimeError(
            "pre-emptive allocation hook invocation mismatch: "
            f"expected_layers={sorted(expected_layers)}, "
            f"actual_layers={sorted(actual_layers)}, "
            f"invocations={len(_preemptive_alloc_invocations)}, "
            f"patched_present={_preemptive_patched_methods!r}, "
            f"candidates={_PREEMPTIVE_METHOD_CANDIDATES!r}"
        )

    layer_idx = extract_layer_index(layer.layer_name)
    cuda_count = len(placement.cuda_expert_ids(layer_idx))
    method_name = type(layer.quant_method).__name__
    weight_names = (
        ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")
        if method_name == "ModelOptNvFp4FusedMoE"
        else (
            "w13_weight_packed",
            "w2_weight_packed",
            "w13_weight_scale",
            "w2_weight_scale",
        )
    )
    for name in weight_names:
        tensor = getattr(layer, name)
        if tensor.shape[0] != cuda_count:
            raise RuntimeError(
                f"{layer.layer_name} {method_name}.{name} has "
                f"{tensor.shape[0]} experts, expected compact CUDA count "
                f"{cuda_count}"
            )

    if (
        method_name == "ModelOptNvFp4FusedMoE"
        and layer.quant_method.use_global_sf
    ):
        for name in ("w13_input_scale", "w2_input_scale"):
            tensor = getattr(layer, name)
            if tensor.shape[0] != placement.num_experts:
                raise RuntimeError(
                    f"{layer.layer_name} {name} has {tensor.shape[0]} "
                    f"experts, expected global scale width "
                    f"{placement.num_experts}"
                )

    if not _preemptive_validation_logged:
        logger.warning(
            "Shooting Brake pre-emptive allocation verified: %d/%d compact "
            "layers via %s",
            len(_preemptive_alloc_invocations),
            len(expected_layers),
            ", ".join(sorted({
                method for _, method in _preemptive_alloc_invocations.values()
            })),
        )
        _preemptive_validation_logged = True


#: Highest CUDA allocation seen while weights were being loaded, in bytes.
#: Sampled during the post-load hook, which is the last moment the load
#: peak is still the process high-water mark: the KV cache is allocated
#: afterwards and is far larger, so `max_memory_allocated` read at the end
#: of a run describes the KV cache and is identical under both surgery
#: strategies. This is the figure that decides whether a model loads.
_load_peak_bytes = 0
_load_audit_logged = False


def _record_load_peak() -> None:
    global _load_peak_bytes
    if torch.cuda.is_available():
        _load_peak_bytes = max(
            _load_peak_bytes, torch.cuda.max_memory_allocated()
        )


def load_peak_gib() -> float:
    """Peak CUDA allocation during weight loading, in GiB."""
    return _load_peak_bytes / 2**30


def _maybe_dump_weight_digest(layer: Any) -> None:
    """Hash the post-surgery expert tensors, when asked.

    A deterministic oracle for "does the compact layout hold the right
    experts". Token-level comparison cannot answer that any more: two runs
    of identical code disagree by ~0.11 nats and share only 4/8 sequences,
    because B70 partials are accumulated asynchronously and the CUDA
    kernel's reductions are not order-stable. Weights, by contrast, are
    fixed once loading ends, so a digest is exact and the two surgery
    strategies must produce identical bytes or one of them is wrong.

    The scales matter as much as the weights. Post-hoc slices tensors that
    stock processing has *already* transformed — `reorder_w1w3_to_w3w1`,
    `swizzle_blockscale` — whereas pre-emptive runs that same processing on
    an already-compact bank. Those steps are only equivalent if they
    commute with slicing along the expert dimension. If they do not, the
    digests differ and one path is feeding the kernel a layout it was never
    validated against, which is worth knowing on its own.

    Off unless ``SHOOTING_BRAKE_WEIGHT_DIGEST`` names an output file.
    """
    path = os.environ.get("SHOOTING_BRAKE_WEIGHT_DIGEST")
    if not path:
        return
    import hashlib
    import json as _json

    def digest(t: Any) -> str | None:
        """Hash raw bytes, never a converted copy.

        The block scales are ``float8_e4m3fn`` after ``swizzle_blockscale``
        and ``.numpy()`` raises ``unsupported ScalarType`` on them, which
        would abort weight loading rather than skip a field. Reinterpreting
        as uint8 also means the digest covers the exact bytes the kernel
        reads, with no dtype conversion in between.
        """
        if t is None:
            return None
        c = t.detach().contiguous().cpu()
        raw = c.view(torch.uint8) if c.element_size() == 1 else (
            c.flatten().view(torch.uint8)
        )
        return hashlib.sha256(raw.numpy().tobytes()).hexdigest()[:16]

    def span(t: Any) -> list[float] | None:
        """Min/max as values. For the activation scalars a magnitude says
        far more than 'the hashes differ' — an 11% spread is the difference
        between reducing over the CUDA-owned experts and over all 256."""
        if t is None:
            return None
        f = t.detach().float()
        return [round(f.min().item(), 10), round(f.max().item(), 10)]

    qconfig = layer.quant_method.moe_quant_config
    record = {
        "layer": layer.layer_index,
        "num_experts": int(layer.w13_weight.shape[0]),
        # Layer attributes.
        "w13_weight": digest(layer.w13_weight),
        "w2_weight": digest(layer.w2_weight),
        "w13_weight_scale": digest(layer.w13_weight_scale),
        "w2_weight_scale": digest(layer.w2_weight_scale),
        # What the kernel is actually handed. The scale properties delegate
        # to the quant config, and post-hoc surgery assigns these
        # explicitly while pre-emptive leaves stock processing's own
        # tensors in place — so they need not equal the layer attributes
        # above, and a digest of the attribute alone could miss a
        # divergence in the tensor that is really read.
        "qc_w1_scale": digest(qconfig._w1.scale),
        "qc_w2_scale": digest(qconfig._w2.scale),
        # Values, not hashes: these are per-expert scalars, small enough to
        # print and far more diagnostic. Kernel-format conversion folds the
        # activation scalar into them, so when they differ the ratio says
        # whether it is exactly that fold or a wrong-expert bug — a hash
        # can only say "different".
        "qc_w1_alpha": span(qconfig._w1.alpha_or_gscale),
        "qc_w2_alpha": span(qconfig._w2.alpha_or_gscale),
        "w13_weight_scale_2": span(
            getattr(layer, "w13_weight_scale_2", None)
        ),
        "w2_weight_scale_2": span(getattr(layer, "w2_weight_scale_2", None)),
        # Full per-expert vector. Min/max cannot separate a fold from a
        # permutation that happens to preserve the extremes; the whole
        # vector can. A fold multiplies every expert by one scalar, so the
        # elementwise ratio between the two strategies is constant. An
        # addressing bug permutes, and the ratios scatter.
        "w2_alpha_vec": (
            None if qconfig._w2.alpha_or_gscale is None else
            [round(v, 12) for v in
             qconfig._w2.alpha_or_gscale.detach().float().flatten().tolist()]
        ),
        "a1_gscale": span(qconfig._a1.alpha_or_gscale),
        "a2_gscale": span(qconfig._a2.alpha_or_gscale),
        "cuda_remap": digest(getattr(layer, "_cuda_remap", None)),
    }
    with open(path, "a") as f:
        f.write(_json.dumps(record) + "\n")


def _install_surgery_hook(quant_method: Any) -> None:
    """Run VRAM surgery as soon as weights finish loading.

    Timing is the whole point. vLLM sizes the KV cache from the *peak*
    memory observed during its profiling forward. Freeing the B70-owned
    expert weights inside or after that forward is too late: the peak
    already counted them, so the freed VRAM is left unusable — the
    capacity gain this architecture exists to deliver is silently lost,
    and only shows up as a large "free" figure at the end of the run.

    ``process_weights_after_loading`` is the last hook before profiling,
    and it receives the layer, so one wrapper per quant method serves
    every layer that shares it. The sentinel keeps a shared method from
    being wrapped twice.
    """
    if getattr(quant_method, _SURGERY_HOOK_ATTR, False):
        return
    original = quant_method.process_weights_after_loading

    def process_weights_after_loading(layer: Any) -> None:
        _record_load_peak()
        global _load_audit_logged
        if torch.cuda.is_available() and not _load_audit_logged:
            logger.warning(
                "Shooting Brake pre-invariant CUDA model allocation: "
                "allocated=%.3f GiB load_peak=%.3f GiB",
                torch.cuda.memory_allocated() / 2**30,
                load_peak_gib(),
            )
            _load_audit_logged = True
        if isinstance(layer, HybridRoutedExperts):
            _validate_preemptive_allocations(layer)
        # The NVFP4 input global scales are deliberately NOT compacted
        # before this call. `prepare_nvfp4_moe_layer_for_fi_or_cutlass`
        # reduces them with a whole-tensor `.max()` and then expands the
        # scalar to `w13.shape[0]`, so their length sets the activation
        # scale's *value*, never its width. Compacting them first would
        # reduce over the CUDA-owned experts alone and hand the kernel a
        # tighter scale than the stock path computes — measured as up to
        # 0.12 nats/token of drift and greedy forks against post-hoc
        # surgery. Leaving them global reproduces stock numerics exactly.
        original(layer)
        if isinstance(layer, HybridRoutedExperts):
            # Order matters: both host tiers source their weights from the
            # same tensors surgery is about to slice away, so they must copy
            # them out first. Reversed, the arena would silently load
            # whatever experts survived the slice — and for the B70 set,
            # surgery removes exactly those, so it would load nothing.
            if layer._cpu_active or b70_prefill_stream_enabled():
                layer._load_host_experts(layer.layer_index)
            layer._maybe_perform_vram_surgery(layer.layer_index)
            _maybe_dump_weight_digest(layer)

    quant_method.process_weights_after_loading = process_weights_after_loading
    setattr(quant_method, _SURGERY_HOOK_ATTR, True)


class HybridRoutedExperts(RoutedExperts):
    """Stock routed experts plus compact ownership and route partitioning."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        qualified_model = require_qualified_config(get_current_vllm_config())
        # Must precede `super().__init__`, which runs `create_weights`: the
        # pre-emptive allocation hook reads ownership off this attribute to
        # decide how many experts to allocate, and would otherwise see a
        # half-built layer. Plain attributes assign fine before
        # `nn.Module.__init__` — only Parameters and Modules may not.
        self.shooting_brake_placement: Placement = build_for_qualified(
            qualified_model,
            policy_name=_placement_policy_name(),
        )
        validate_cuda_dummy_slot_placement(self.shooting_brake_placement)
        if preemptive_surgery_enabled():
            _reject_unsupported_preemptive_tiers(self.shooting_brake_placement)
        super().__init__(*args, **kwargs)
        if self.hidden_size != qualified_model.hidden_size:
            raise RuntimeError(
                f"routed layer hidden size {self.hidden_size} != qualified "
                f"model hidden size {qualified_model.hidden_size}"
            )
        self.shooting_brake_qualified_model = qualified_model
        self._device_map_cpu = build_device_map(self.shooting_brake_placement)
        self._b70_slot_map = _build_b70_slot_map(self.shooting_brake_placement)
        # Phase 8a: pinned host buffers for async B70 overlap.
        # Pre-allocated once and reused every step to avoid per-step
        # allocation on the hot path.  Pinned memory enables DMA D2H/H2D.
        self._b70_max_batch = int(
            os.environ.get("SHOOTING_BRAKE_B70_MAX_BATCH", "128")
        )
        self._dispatch_geometry = DispatchBufferGeometry(
            max_batch=self._b70_max_batch,
            hidden_size=qualified_model.hidden_size,
            top_k=qualified_model.top_k,
        )
        # Both sides use this one model-derived shape. For the 88B step-1
        # model it is [max_batch, 3072]; a stale 2048 would now fail the
        # registration check instead of making every native dispatch fail.
        self._b70_pinned_hidden: torch.Tensor = torch.empty(
            *self._dispatch_geometry.hidden_shape, dtype=torch.float16,
            pin_memory=True, device="cpu",
        )
        self._b70_pinned_output: torch.Tensor = torch.empty(
            *self._dispatch_geometry.hidden_shape, dtype=torch.float32,
            pin_memory=True, device="cpu",
        )
        # Phase 8.5: VRAM surgery state. When enabled, B70-owned expert
        # weights are removed from CUDA VRAM on the first forward call.
        self._vram_surgery_done = False
        self._passthrough_seen: set[int] = set()
        self._cuda_remap: torch.Tensor | None = None
        self._layer_idx: int | None = None
        self._device_map_layer: torch.Tensor | None = None
        # Phase-6a partition statistics (per-layer, in-process).
        self.shooting_brake_partition_stats: dict[str, int] = {
            "steps": 0,
            "remote_steps": 0,
            "remote_routes": 0,
            "max_remote_per_step": 0,
        }
        self._shadow_done = False
        self._aggregation_capture_done = False
        # Cache env-var decisions at init time so forward_modular has
        # zero Python branching on the hot path in all-CUDA mode — this
        # makes it a pure pass-through compatible with CUDA graph capture.
        self._hybrid_active = (
            os.environ.get("SHOOTING_BRAKE_HYBRID") == "1"
            and self.shooting_brake_placement.b70_count() > 0
        )
        self._all_cuda_passthrough = not (
            self._hybrid_active
            or os.environ.get("SHOOTING_BRAKE_SHADOW") == "1"
            or os.environ.get("SHOOTING_BRAKE_VRAM_SURGERY") == "1"
        )
        # Cache B70 env decisions (hot-path reads are expensive during graph replay).
        self._b70_device_cached = (
            os.environ.get("SHOOTING_BRAKE_B70_DEVICE") == "1"
        )
        self._hybrid_env = os.environ.get("SHOOTING_BRAKE_HYBRID") == "1"
        self._b70_async_cached = (
            os.environ.get("SHOOTING_BRAKE_B70_ASYNC", "1") != "0"
        )
        # Static output buffer for breakable CUDA graph compatibility.
        # Allocated lazily on first forward (needs hidden_dim from weights).
        self._static_output: torch.Tensor | None = None
        # Tier 3: graph-compatible B70 dispatch via CUDA stream signaling.
        # When enabled, B70 dispatch uses cuStreamWriteValue32/WaitValue32
        # (capture-compatible) instead of Python/ctypes (not capture-compatible).
        self._b70_graph_mode = (
            os.environ.get("SHOOTING_BRAKE_B70_GRAPH") == "1"
            and self._b70_device_cached
            and self._hybrid_env
        )
        self._expected_arm = os.environ.get("SHOOTING_BRAKE_EXPECT_ARM")
        self._arm_runtime_observations: set[tuple[str, str, bool]] = set()
        self.shooting_brake_arm_config: dict[str, Any] | None = None
        if self._expected_arm is not None:
            from . import force_piecewise_fired

            vllm_config = get_current_vllm_config()
            breakable_enabled = (
                os.environ.get("VLLM_USE_BREAKABLE_CUDAGRAPH") == "1"
            )
            enforce_eager = bool(vllm_config.model_config.enforce_eager)
            fired = force_piecewise_fired()
            cudagraph_mode = vllm_config.compilation_config.cudagraph_mode
            _validate_expected_arm_configuration(
                expected_arm=self._expected_arm,
                graph_mode=self._b70_graph_mode,
                breakable_enabled=breakable_enabled,
                enforce_eager=enforce_eager,
                force_piecewise_fired=fired,
                cudagraph_mode=cudagraph_mode,
            )
            self.shooting_brake_arm_config = {
                "expected_arm": self._expected_arm,
                "graph_env": os.environ.get("SHOOTING_BRAKE_B70_GRAPH"),
                "graph_mode": self._b70_graph_mode,
                "breakable_env": os.environ.get(
                    "VLLM_USE_BREAKABLE_CUDAGRAPH"
                ),
                "breakable_enabled": breakable_enabled,
                "enforce_eager": enforce_eager,
                "force_piecewise_fired": fired,
                "cudagraph_mode": cudagraph_mode.name,
            }
            logger.info_once(
                "Shooting Brake benchmark arm resolved: %s",
                self.shooting_brake_arm_config,
            )
        self._b70_poller: Any = None
        self._first_forward_done = False
        self._b70_stats = os.environ.get("SHOOTING_BRAKE_B70_STATS") == "1"
        # [b70_routes, total_routes, cpu_routes] accumulated on device.
        # Incrementing a device tensor needs no host sync, so it survives
        # graph capture; read it between steps via collective_rpc. The
        # third slot stays zero unless all-out mode is on — allocated
        # unconditionally because `_cpu_active` is not resolved yet here,
        # and 8 bytes is not worth the ordering constraint.
        self._route_counter: torch.Tensor | None = (
            torch.zeros(3, dtype=torch.int64, device="cuda")
            if self._b70_stats and self._b70_graph_mode
            else None
        )
        # Opt-in global route observers. They bind on the first forward,
        # when the device and top-k are known. Each None check below is the
        # entire steady-state cost of its disabled observer.
        self._route_histogram: Any = None
        self._route_trace: Any = None
        # Both synchronous prefill and graph decode gather through this map.
        # It must exist before vLLM's KV-profile dummy forward.
        self._initialize_b70_slot_map_cuda()
        if self._b70_graph_mode:
            # Pinned buffers for routing data (D2H targets).
            self._pinned_b70_ids = torch.empty(
                *self._dispatch_geometry.route_shape, dtype=torch.int32,
                pin_memory=True, device="cpu",
            )
            self._pinned_b70_weights = torch.empty(
                *self._dispatch_geometry.route_shape, dtype=torch.float32,
                pin_memory=True, device="cpu",
            )
            # Device-side result buffers (pre-allocated, no torch.empty in forward).
            self._dev_b70_fp32 = torch.empty(
                self._b70_max_batch, self.hidden_size,
                dtype=torch.float32, device="cuda",
            )
            self._dev_b70_bf16 = torch.empty(
                self._b70_max_batch, self.hidden_size,
                dtype=torch.bfloat16, device="cuda",
            )
            # Signal/completion flags (host-mapped, accessible from both sides).
            from .stream_signal import alloc_host_mapped_flag
            self._signal_host, self._signal_dev = alloc_host_mapped_flag(0)
            self._completion_host, self._completion_dev = alloc_host_mapped_flag(0)
            # One shared poller serves every NVFP4 layer (single physical
            # B70).  Registration is deferred to the first forward, where
            # `_ensure_layer_device_map` resolves this layer's real index.
            logger.info(
                "Shooting Brake Tier 3: graph-compatible B70 dispatch "
                "enabled for layer %s", self.layer_name,
            )
        # All-out mode: the CPU DDR5 cold tier. Structurally a second copy
        # of the Tier 3 machinery above (flags, pinned staging, native
        # poller) pointed at a different compute backend, so both remote
        # tiers dispatch and join identically. Off unless a placement
        # actually assigns CPU experts, which only AllOutPolicy does.
        self._cpu_active = (
            self._b70_graph_mode
            and self.shooting_brake_placement.cpu_count() > 0
        )
        self._cpu_poller: Any = None
        self._cpu_host: Any = None
        self._cpu_intermediate: int | None = None
        # Resolved once: the flag is read at construction like every other
        # adapter switch, so a mid-run environment change cannot desync the
        # arena contents (loaded at weight-load time) from the forward path.
        self._b70_prefill_stream = (
            b70_prefill_stream_enabled() and self._b70_graph_mode
        )
        self._cpu_id_map = _build_cpu_id_map(self.shooting_brake_placement)
        if self._cpu_active:
            hidden = self.hidden_size
            self._cpu_id_map_cuda = torch.tensor(
                self._cpu_id_map, device="cuda", dtype=torch.int32,
            )
            # bf16 staging, unlike the B70's fp16: the CPU kernels consume
            # bf16 directly, so the model's own dtype crosses unconverted.
            self._cpu_pinned_hidden = torch.empty(
                self._b70_max_batch, hidden, dtype=torch.bfloat16,
                pin_memory=True, device="cpu",
            )
            self._cpu_pinned_ids = torch.empty(
                *self._dispatch_geometry.route_shape, dtype=torch.int32,
                pin_memory=True, device="cpu",
            )
            self._cpu_pinned_weights = torch.empty(
                *self._dispatch_geometry.route_shape, dtype=torch.float32,
                pin_memory=True, device="cpu",
            )
            self._cpu_pinned_output = torch.empty(
                self._b70_max_batch, hidden, dtype=torch.float32,
                pin_memory=True, device="cpu",
            )
            self._dev_cpu_fp32 = torch.empty(
                self._b70_max_batch, hidden, dtype=torch.float32,
                device="cuda",
            )
            self._dev_cpu_bf16 = torch.empty(
                self._b70_max_batch, hidden, dtype=torch.bfloat16,
                device="cuda",
            )
            from .stream_signal import alloc_host_mapped_flag
            self._cpu_signal_host, self._cpu_signal_dev = (
                alloc_host_mapped_flag(0)
            )
            self._cpu_completion_host, self._cpu_completion_dev = (
                alloc_host_mapped_flag(0)
            )
            global _all_out_warned
            if not _all_out_warned:
                _all_out_warned = True
                cpu_layers = self.shooting_brake_placement.cpu_active_layers()
                logger.warning(
                    "Shooting Brake ALL-OUT MODE: %d layers will each "
                    "compute %d routed experts on CPU cores. That is ~5x "
                    "the B70's per-expert cost, which is why this is off by "
                    "default; it buys VRAM, not speed.",
                    len(cpu_layers),
                    self.shooting_brake_placement.layer_cpu_count(
                        cpu_layers[0]
                    ),
                )
        self.shooting_brake_provider = ShootingBrakeExpertProviderClient(
            qualified_model=qualified_model,
            layer_name=self.layer_name,
            placement=self.shooting_brake_placement,
        )
        # Phase-6a diagnostic: verify env vars + code path in EngineCore.
        _init_marker = os.environ.get("SHOOTING_BRAKE_PARTITION_MARKER")
        if _init_marker:
            import json as _json
            with open(_init_marker, "a") as _f:
                _f.write(_json.dumps({
                    "init": True,
                    "layer_name": self.layer_name,
                    "policy": _placement_policy_name(),
                    "b70_count": self.shooting_brake_placement.b70_count(),
                    "is_monolithic": self.quant_method.is_monolithic,
                }) + "\n")
        _install_surgery_hook(self.quant_method)

    def _initialize_b70_slot_map_cuda(
        self, device: str | torch.device = "cuda",
    ) -> None:
        """Materialize the global-to-B70 slot map before any forward."""
        self._b70_slot_map_cuda: torch.Tensor | None = None
        if self._hybrid_active:
            self._b70_slot_map_cuda = torch.tensor(
                self._b70_slot_map, device=device, dtype=torch.int32,
            )

    @property
    def layer_index(self) -> int:
        """Absolute layer index, parsed from ``layer_name``."""
        if self._layer_idx is None:
            from vllm.model_executor.models.utils import extract_layer_index

            self._layer_idx = extract_layer_index(self.layer_name)
        return self._layer_idx

    def _ensure_layer_device_map(
        self, topk_ids: torch.Tensor
    ) -> tuple[int, torch.Tensor]:
        """Resolve this layer's index and cache its device-map row."""
        layer_idx = self.layer_index
        if (
            self._device_map_layer is None
            or self._device_map_layer.device != topk_ids.device
        ):
            self._device_map_layer = self._device_map_cpu[layer_idx].to(
                topk_ids.device
            )
        return layer_idx, self._device_map_layer

    def _maybe_perform_vram_surgery(self, layer_idx: int) -> None:
        """Lazily remove B70-owned expert weights from CUDA VRAM.

        Runs once per layer on the first ``forward_modular`` call, after
        vLLM's ``process_weights_after_loading`` has set up all tensors.
        Slices every per-expert weight and scale tensor to contain only
        CUDA-owned experts, then updates the quant config's internal
        references so the kernel's ``@property`` delegates see the smaller
        tensors.

        A global→local remap tensor is built so the CUDA kernel receives
        compact local IDs (B70 routes map to expert 0 with zero weight,
        producing zero contribution).  The real B70 results are added
        separately in the hybrid path.

        Requires ``SHOOTING_BRAKE_VRAM_SURGERY=1``, ``B70_DEVICE=1``, and
        ``HYBRID=1`` because B70 experts are removed from CUDA and must be
        computed on the B70 device. Missing flags reject rather than skip.
        """
        if self._vram_surgery_done:
            return
        self._vram_surgery_done = True  # set early to prevent re-entry

        placement = self.shooting_brake_placement

        # Skip layers with nothing offloaded (e.g. FP8 layers 32-39). Both
        # tiers count: the slice below keeps only CUDA-owned experts, so a
        # layer holding CPU experts and no B70 ones still has VRAM to
        # reclaim, and counting only B70 would leave it behind.
        b70_count = placement.layer_b70_count(layer_idx)
        cpu_count = placement.layer_cpu_count(layer_idx)
        if b70_count + cpu_count == 0:
            return

        # Determine CUDA-owned expert IDs (sorted ascending for contiguous
        # index_select — even for interleaved placement this produces the
        # correct compact-to-global mapping).
        cuda_global_ids = sorted(placement.cuda_expert_ids(layer_idx))
        device = self.w13_weight.device
        num_cuda = len(cuda_global_ids)

        # Deliberately ahead of the env gates below. Once `create_weights`
        # has allocated a compact bank, the finalize step is a consequence
        # of that allocation, not of the surgery flags: it builds
        # `_cuda_remap` and retires the loader's `_expert_map`. Gated, a
        # pre-emptive run without SHOOTING_BRAKE_VRAM_SURGERY would pair
        # compact weights with a null remap and a live competing map, and
        # index an 8-row bank with global ids — wrong tokens, no error.
        if self.w13_weight.shape[0] == num_cuda:
            logger.info(
                "Shooting Brake: layer %d already compact at %d/%d experts "
                "(%d B70 + %d CPU never allocated)",
                layer_idx, num_cuda, placement.num_experts,
                b70_count, cpu_count,
            )
            self._finalize_compact_experts(layer_idx, cuda_global_ids, device)
            return

        if os.environ.get("SHOOTING_BRAKE_VRAM_SURGERY") != "1":
            return
        if os.environ.get("SHOOTING_BRAKE_B70_DEVICE") != "1":
            raise RuntimeError(
                "SHOOTING_BRAKE_VRAM_SURGERY=1 with offloaded experts "
                "requires SHOOTING_BRAKE_B70_DEVICE=1"
            )
        if os.environ.get("SHOOTING_BRAKE_HYBRID") != "1":
            raise RuntimeError(
                "SHOOTING_BRAKE_VRAM_SURGERY=1 with offloaded experts "
                "requires SHOOTING_BRAKE_HYBRID=1"
            )

        cuda_idx = torch.tensor(
            cuda_global_ids, device=device, dtype=torch.long,
        )
        logger.info(
            "Shooting Brake VRAM surgery: layer %d — slicing %d→%d CUDA "
            "experts (freeing %d B70 + %d CPU experts)",
            layer_idx, placement.num_experts, num_cuda, b70_count, cpu_count,
        )

        # --- Slice weight parameters (accessed by forward_modular) ---
        self.w13_weight = torch.nn.Parameter(
            self.w13_weight.data.index_select(0, cuda_idx).clone(),
            requires_grad=False,
        )
        self.w2_weight = torch.nn.Parameter(
            self.w2_weight.data.index_select(0, cuda_idx).clone(),
            requires_grad=False,
        )

        # --- Slice per-block scale parameters ---
        self.w13_weight_scale = torch.nn.Parameter(
            self.w13_weight_scale.data.index_select(0, cuda_idx).clone(),
            requires_grad=False,
        )
        self.w2_weight_scale = torch.nn.Parameter(
            self.w2_weight_scale.data.index_select(0, cuda_idx).clone(),
            requires_grad=False,
        )

        # --- Update quant config internal references ---
        # All scale properties on FusedMoEExpertsModular delegate to
        # quant_config._w1 / _w2 / _a1 / _a2 FusedMoEQuantDesc fields.
        qconfig = self.quant_method.moe_quant_config

        # w1_scale / w2_scale — point to the freshly sliced parameters.
        qconfig._w1.scale = self.w13_weight_scale.data
        qconfig._w2.scale = self.w2_weight_scale.data

        # g1_alphas / g2_alphas (= w13/w2_weight_scale_2 after processing).
        qconfig._w1.alpha_or_gscale = (
            qconfig._w1.alpha_or_gscale.index_select(0, cuda_idx).clone()
        )
        qconfig._w2.alpha_or_gscale = (
            qconfig._w2.alpha_or_gscale.index_select(0, cuda_idx).clone()
        )

        # a1_gscale / a2_gscale (derived from input scales).
        qconfig._a1.alpha_or_gscale = (
            qconfig._a1.alpha_or_gscale.index_select(0, cuda_idx).clone()
        )
        qconfig._a2.alpha_or_gscale = (
            qconfig._a2.alpha_or_gscale.index_select(0, cuda_idx).clone()
        )

        # --- Sync layer attributes to match quant config ---
        self.w13_weight_scale_2 = qconfig._w1.alpha_or_gscale
        self.w2_weight_scale_2 = qconfig._w2.alpha_or_gscale
        self.w13_input_scale = (
            1.0 / qconfig._a1.alpha_or_gscale
        ).clone()
        self.w2_input_scale = (
            1.0 / qconfig._a2.alpha_or_gscale
        ).clone()

        self._finalize_compact_experts(layer_idx, cuda_global_ids, device)

    def _finalize_compact_experts(
        self,
        layer_idx: int,
        cuda_global_ids: list[int],
        device: torch.device,
    ) -> None:
        """Bookkeeping both surgery strategies need once VRAM holds exactly
        the CUDA-owned experts.

        Separate from the slicing because pre-emptive allocation arrives
        here with nothing to slice, and duplicating the remap construction
        for that path is how the two strategies would silently diverge.
        """
        num_cuda = len(cuda_global_ids)

        # --- Retire the loader's expert map ---
        # It exists only so vLLM's weight loader skips non-CUDA experts
        # during load. Leaving it live would hand the kernel a second,
        # competing global→local mapping on top of `_cuda_remap` below —
        # and unlike a crash, a double remap silently computes the wrong
        # experts. Clearing it makes the forward path byte-identical to the
        # post-hoc strategy, which is what lets the two be compared.
        if getattr(self, "_expert_map", None) is not None:
            self._expert_map = None
            self.expert_map_manager._expert_map = None

        # --- Update metadata ---
        self.local_num_experts = num_cuda

        # --- Update expert object's per-expert attributes ---
        expert_obj = self.quant_method.moe_kernel.fused_experts
        expert_obj.num_experts = num_cuda
        if (
            hasattr(expert_obj, "gemm1_clamp_limit")
            and expert_obj.gemm1_clamp_limit is not None
        ):
            expert_obj.gemm1_clamp_limit = (
                expert_obj.gemm1_clamp_limit[:num_cuda].clone()
            )

        # --- Build and verify both directions of the CUDA compaction ---
        global_to_local, local_to_global = build_cuda_expert_maps(
            self.shooting_brake_placement, layer_idx,
        )
        expected_globals = torch.tensor(cuda_global_ids, dtype=torch.long)
        if not torch.equal(local_to_global, expected_globals):
            raise RuntimeError(
                f"layer {layer_idx} CUDA tensor order {cuda_global_ids} does "
                f"not match placement compaction {local_to_global.tolist()}"
            )
        # Keep -1 for non-CUDA globals. `compact_cuda_routes` tests ownership
        # before replacing those sentinels with zero-weight dummy slot zero.
        self._cuda_remap = global_to_local.to(device)

        # --- Reclaim freed VRAM ---
        torch.cuda.empty_cache()

        # --- Write VRAM marker for integration test ---
        vram_marker = os.environ.get("SHOOTING_BRAKE_VRAM_MARKER")
        if vram_marker:
            vram_now = torch.cuda.memory_allocated() / (1024**3)
            with open(vram_marker, "a") as vf:
                vf.write(f"{layer_idx} {vram_now:.6f}\n")

        logger.info(
            "Shooting Brake VRAM surgery: layer %d done — %d CUDA experts "
            "on device, remap ready (B70→dummy 0)",
            layer_idx, num_cuda,
        )

    def _w13_is_reordered(self) -> bool:
        """Whether the fused ``w13`` holds ``[up, gate]`` rather than
        ``[gate, up]``.

        vLLM reorders for the FlashInfer NVFP4 MoE kernels, under exactly the
        condition reproduced here from
        ``prepare_nvfp4_moe_layer_for_fi_or_cutlass``: a gated act-and-mul
        layer on one of the three FlashInfer backends. VLLM_CUTLASS shares
        that code path but is excluded from the reorder, so the backend
        identity matters and "is it FlashInfer-ish" is not a safe proxy.

        Read from vLLM's own enum rather than matched on a backend name, so
        a renamed or newly added backend surfaces as an AttributeError at
        load time instead of a silently swapped activation. Unknown or
        missing backend means no reorder, which matches the default kernel
        layout.
        """
        backend = getattr(self.quant_method, "nvfp4_backend", None)
        if backend is None:
            return False
        from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
            NvFp4MoeBackend,
        )

        reordering = (
            NvFp4MoeBackend.FLASHINFER_CUTLASS,
            NvFp4MoeBackend.FLASHINFER_TRTLLM,
            NvFp4MoeBackend.FLASHINFER_B12X,
        )
        return (
            backend in reordering
            and self.activation.is_gated
            and self.moe_config.is_act_and_mul
        )

    def _verify_bank_plane(
        self, layer_idx: int, expert_id: int,
        gate_q: torch.Tensor, up_q: torch.Tensor, down_q: torch.Tensor,
        gate_sf: torch.Tensor, up_sf: torch.Tensor, down_sf: torch.Tensor,
        gs13: float, gs2: float, a1: float, a2: float,
    ) -> None:
        """Check the bank would produce the same arena as VRAM does.

        Runs under ``SHOOTING_BRAKE_ARENA_VERIFY_BANK=1``, against the
        arrays about to be handed to ``load_expert`` — so equality here
        implies equality in the arena, without needing an accessor to read
        packed memory back out.

        Reference configuration is post-hoc surgery with an ``allout``
        placement: the cold tier's working setup, and the only one where
        the VRAM path is valid at all.
        """
        import numpy as np

        from .expert_bank import ExpertBank, global_scale_divisor

        bank = getattr(self, "_verify_bank", None)
        if bank is None:
            bank = ExpertBank(_b70_bank_path())
            self._verify_bank = bank

        planes = bank.expert(layer_idx, expert_id)
        checks = (
            ("gate_q", gate_q, planes.gate), ("up_q", up_q, planes.up),
            ("down_q", down_q, planes.down),
            ("gate_sf", gate_sf, planes.gate_sf),
            ("up_sf", up_sf, planes.up_sf),
            ("down_sf", down_sf, planes.down_sf),
        )
        for name, from_vram, from_bank in checks:
            got = from_vram.contiguous().cpu().numpy().view(np.uint8)
            want = np.ascontiguousarray(from_bank).view(np.uint8)
            if got.shape != want.shape or not np.array_equal(
                got.reshape(-1), want.reshape(-1)
            ):
                raise RuntimeError(
                    f"arena mismatch at layer={layer_idx} expert={expert_id} "
                    f"plane={name}: VRAM {got.shape} vs bank {want.shape}, "
                    f"{int((got.reshape(-1) != want.reshape(-1)).sum()) if got.shape == want.shape else 'shape'} "
                    "bytes differ"
                )

        # The scalars are floats, so compare as ratios rather than bytes.
        for name, from_vram, raw, act in (
            ("w13", gs13, planes.w13_inv_global, a1),
            ("w2", gs2, planes.w2_inv_global, a2),
        ):
            folded = raw / global_scale_divisor(act)
            if abs(folded - from_vram) > 1e-6 * max(abs(from_vram), 1e-30):
                raise RuntimeError(
                    f"arena scale mismatch at layer={layer_idx} "
                    f"expert={expert_id} plane={name}: VRAM {from_vram:.6e} "
                    f"vs bank {raw:.6e}/{act:.4f}={folded:.6e}"
                )

    def _verify_bank_wiring(
        self, host: Any, layer_idx: int, expert_id: int,
    ) -> None:
        """Confirm the bank path put each plane in the slot it belongs to.

        Reads the arena back and compares every slot against a reference
        dequantized from the layer's own VRAM tensors — the same swizzled
        source ``_verify_cpu_expert`` uses on the VRAM path, so this is a
        genuinely independent oracle rather than the bank checking itself.

        Only meaningful while VRAM still holds every expert, which is why
        the caller gates it on that.
        """
        from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (  # noqa: E501
            dequantize_to_dtype,
            kE2M1ToFloat_handle,
        )

        qconfig = self.quant_method.moe_quant_config
        device = self.w13_weight.device
        kE2M1ToFloat_handle.val = kE2M1ToFloat_handle.val.to(device)
        ref13 = dequantize_to_dtype(
            self.w13_weight.data[expert_id],
            self.w13_weight_scale.data[expert_id],
            qconfig._w1.alpha_or_gscale[expert_id].float(),
            torch.float32,
        ).cpu()
        ref2 = dequantize_to_dtype(
            self.w2_weight.data[expert_id],
            self.w2_weight_scale.data[expert_id],
            qconfig._w2.alpha_or_gscale[expert_id].float(),
            torch.float32,
        ).cpu()
        inter = ref13.shape[0] // 2
        lo, hi = ref13[:inter].contiguous(), ref13[inter:].contiguous()
        # VRAM's halves are [up, gate] on the FlashInfer backends; the bank
        # path wrote [gate, up]. Comparing without this rule would flag a
        # correct load, or bless a swapped one.
        gate, up = (lo, hi) if not self._w13_is_reordered() else (hi, lo)
        self._verify_cpu_expert(
            host, layer_idx, expert_id, gate, up, ref2.contiguous(),
        )

    def _load_host_experts_from_bank(
        self, layer_idx: int, host_ids: tuple[int, ...],
    ) -> None:
        """Fill the arena from the expert bank rather than from VRAM.

        The path pre-emptive allocation requires: VRAM holds only the
        CUDA-owned experts, so the host tiers' weights exist nowhere else.

        Less work than the VRAM path, not more. The bank is written from
        the raw checkpoint, so ``w13`` is already ``[gate, up]`` (no
        ``reorder_w1w3_to_w3w1`` to undo) and the block scales are already
        linear (no ``swizzle_blockscale`` to reverse).

        The one thing the bank cannot supply is the activation global
        scale, which the arena's per-plane scalar has folded in. That
        scalar is per layer and uniform across experts, so it is read from
        the quant config — which is valid even under pre-emptive, where the
        *per-expert* alphas cover only the CUDA set and the host experts
        are absent from it entirely.
        """
        from .cpu_expert_host import PackedPlane, get_cpu_host
        from .expert_bank import ExpertBank, global_scale_divisor

        placement = self.shooting_brake_placement
        bank = getattr(self, "_arena_bank", None)
        if bank is None:
            bank = ExpertBank(_b70_bank_path())
            self._arena_bank = bank
        if bank.hidden != self.hidden_size:
            raise RuntimeError(
                f"expert bank hidden {bank.hidden} != model {self.hidden_size}"
            )

        qconfig = self.quant_method.moe_quant_config
        a1 = float(qconfig._a1.alpha_or_gscale.flatten()[0])
        a2 = float(qconfig._a2.alpha_or_gscale.flatten()[0])

        inter = bank.intermediate
        self._cpu_intermediate = inter
        max_experts = placement.cpu_count() + (
            placement.b70_count() if b70_prefill_stream_enabled() else 0
        )
        host = get_cpu_host(
            num_layers=placement.num_layers,
            num_experts=placement.num_experts,
            hidden=bank.hidden,
            intermediate=inter,
            max_experts=max_experts,
            num_threads=int(os.environ.get("SHOOTING_BRAKE_CPU_THREADS", "0")),
        )
        self._cpu_host = host

        for expert_id in host_ids:
            p = bank.expert(layer_idx, expert_id)
            gs13 = p.w13_inv_global / global_scale_divisor(a1)
            gs2 = p.w2_inv_global / global_scale_divisor(a2)
            host.load_expert(
                layer_idx, expert_id,
                PackedPlane(torch.from_numpy(np.ascontiguousarray(p.gate)),
                            torch.from_numpy(np.ascontiguousarray(p.gate_sf)),
                            gs13),
                PackedPlane(torch.from_numpy(np.ascontiguousarray(p.up)),
                            torch.from_numpy(np.ascontiguousarray(p.up_sf)),
                            gs13),
                PackedPlane(torch.from_numpy(np.ascontiguousarray(p.down)),
                            torch.from_numpy(np.ascontiguousarray(p.down_sf)),
                            gs2),
            )

        if os.environ.get("SHOOTING_BRAKE_CPU_VERIFY") == "1" and host_ids:
            # Checks the argument wiring above — which plane reached which
            # slot, and with which scalar. The reference is built from
            # *VRAM's* swizzled tensors, which is what `dequantize_to_dtype`
            # expects; building it from the bank's linear scales reports
            # rel=5.9e-01 while printing OK, and checks nothing.
            #
            # That reference only exists while VRAM still holds every
            # expert, i.e. under post-hoc surgery, which is precisely what
            # SHOOTING_BRAKE_ARENA_FROM_BANK=1 is for. Under pre-emptive the
            # experts are absent and there is nothing to compare against, so
            # the check declines rather than inventing a reference.
            expert_id = host_ids[0]
            if self.w13_weight.shape[0] != placement.num_experts:
                logger.warning(
                    "Shooting Brake: CPU_VERIFY needs post-hoc surgery to "
                    "build a reference (run with "
                    "SHOOTING_BRAKE_ARENA_FROM_BANK=1 instead); skipping"
                )
            else:
                self._verify_bank_wiring(host, layer_idx, expert_id)
        # What is and is not verified here, precisely.
        #
        # SHOOTING_BRAKE_ARENA_VERIFY_BANK=1 proves the bank *reader* agrees
        # with the VRAM path byte for byte — 128 host experts at
        # allout:16:8:8, six planes and two scalars each, zero mismatches,
        # and the comparison distinguishes neighbouring experts by 522k
        # bytes so passing it is a real result. But that gate runs inside
        # `_load_host_experts`; it never watches *this* function call
        # `load_expert`. The argument wiring below — which plane lands in
        # which slot, and which scalar goes with it — is therefore verified
        # by inspection against the VRAM path's identical call, not by an
        # oracle. A gate/up slot swap here would pass every automated check
        # in the tree, and at 3.3% of routes it would sit under the token
        # harness's 0.11-nat floor.
        #
        # To close that: SHOOTING_BRAKE_ARENA_FROM_BANK=1 forces this path
        # under post-hoc surgery, where VRAM still holds every expert and
        # `_verify_cpu_expert`'s swizzled reference is constructible. Run
        # 35B allout with it and SHOOTING_BRAKE_CPU_VERIFY=1 to check the
        # wiring against a known-good oracle.
        #
        # No CPU_VERIFY block on this path in the meantime: that check
        # builds its reference with `dequantize_to_dtype`, which expects
        # swizzled block scales. The bank's are linear, so it reported
        # rel=5.9e-01 against the VRAM path's 5.65e-07 while still printing
        # OK — worse than no check.

        logger.info(
            "Shooting Brake: layer %d loaded %d host experts from the bank "
            "— arena %.2f / %.2f GiB",
            layer_idx, len(host_ids),
            host.arena_used_bytes / 2**30, host.arena_capacity_bytes / 2**30,
        )

    def _load_host_experts(self, layer_idx: int) -> None:
        """Copy this layer's host-resident experts into the DRAM arena, packed.

        Two tiers draw on the same arena, and which experts land here depends
        on both:

        * **CPU tier** (``allout:`` placement) — experts the CPU poller
          computes at decode. Always loaded when present.
        * **B70 prefill streaming** (``SHOOTING_BRAKE_B70_PREFILL_STREAM=1``)
          — B70-owned experts, loaded *in addition*, because above the
          crossover it is cheaper to move an expert's weights to the 5090
          once per layer than to dispatch tokens to a kernel that re-reads
          them. The B70 keeps its own copy and still serves decode; this is a
          second copy for a different regime, not a migration.

        One arena keyed by ``(layer, expert)`` serves both: the two id sets
        are disjoint by construction, so a single streamer and a single ring
        cover them with no per-tier bookkeeping.

        Reading the B70 set back over PCIe instead of holding it in DRAM was
        rejected on arithmetic: the B70 sits on a PCH Gen3 x4 link, so
        recovering 0.41 GiB per layer would cost ~107 ms against ~8 ms to
        push the same bytes from DRAM over the 5090's direct Gen5 x16.


        Must run before VRAM surgery slices the CUDA weights: surgery keeps
        only CUDA-owned experts, so once it has run the CPU tier's source
        weights are gone. :func:`_install_surgery_hook` orders the two.

        Weights stay in NVFP4 rather than being dequantized. That is a
        capacity decision before it is a speed one: bf16 costs 2 bytes per
        weight against NVFP4's 0.5625, and at 3.56x a 60 GiB expert bank
        becomes 216 GiB and stops fitting in host DRAM at all. It is also
        faster, since this tier's cost model is bytes-read.

        The only transformation applied is to the block scales. The
        checkpoint stores them swizzled for a hardware kernel this tier does
        not use, so they are converted to linear order once, here, rather
        than being un-swizzled on every access.

        Weight layout matches ``nn.Linear``: vLLM fuses gate and up into
        ``w13_weight`` as ``[E, 2*I, H/2]`` with gate first, and keeps down as
        ``w2_weight`` ``[E, H, I/2]`` — exactly the arena's expected shapes,
        so the split is a view, not a transpose.
        """
        from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (  # noqa: E501
            convert_swizzled_to_linear,
        )

        from .cpu_expert_host import PackedPlane, get_cpu_host

        placement = self.shooting_brake_placement
        cpu_ids = placement.cpu_expert_ids(layer_idx)
        stream_b70 = b70_prefill_stream_enabled()
        b70_ids = placement.b70_expert_ids(layer_idx) if stream_b70 else ()
        host_ids = cpu_ids + b70_ids
        if not host_ids:
            return

        # Pre-emptive allocation never creates the offloaded experts in
        # VRAM, so there is nothing here to slice: the bank is the only
        # source that still holds them. It is also the cheaper one — the
        # bank is raw checkpoint order with linear scales, so neither the
        # w13 half-swap nor the un-swizzle below applies. Proven equivalent
        # byte-for-byte against the VRAM path under
        # SHOOTING_BRAKE_ARENA_VERIFY_BANK=1 on the 35B at allout:16:8:8,
        # across all 128 host experts.
        # The override forces this path under post-hoc surgery too, where
        # VRAM still holds every expert — the one configuration in which
        # the bank path's argument wiring can be checked against
        # `_verify_cpu_expert`'s known-good swizzled reference.
        if preemptive_surgery_enabled() or (
            os.environ.get("SHOOTING_BRAKE_ARENA_FROM_BANK") == "1"
        ):
            self._load_host_experts_from_bank(layer_idx, host_ids)
            return

        device = self.w13_weight.device
        idx = torch.tensor(list(host_ids), device=device, dtype=torch.long)
        qconfig = self.quant_method.moe_quant_config

        w13_q = self.w13_weight.data.index_select(0, idx)
        w2_q = self.w2_weight.data.index_select(0, idx)
        w13_sf = self.w13_weight_scale.data.index_select(0, idx)
        w2_sf = self.w2_weight_scale.data.index_select(0, idx)
        w13_gs = qconfig._w1.alpha_or_gscale.index_select(0, idx).float()
        w2_gs = qconfig._w2.alpha_or_gscale.index_select(0, idx).float()

        hidden = self.hidden_size
        two_inter = w13_q.shape[1]
        inter = two_inter // 2

        # Arena is sized for both tiers up front: it reserves address space
        # and commits lazily, but the reservation has to cover the worst
        # case or a later layer's load runs out of room.
        max_experts = placement.cpu_count() + (
            placement.b70_count() if stream_b70 else 0
        )
        host = get_cpu_host(
            num_layers=placement.num_layers,
            num_experts=placement.num_experts,
            hidden=hidden,
            intermediate=inter,
            max_experts=max_experts,
            num_threads=int(os.environ.get("SHOOTING_BRAKE_CPU_THREADS", "0")),
        )
        self._cpu_host = host
        # Kept for the prefill streamer, which needs the plane shape to
        # slice a streamed block back into gate/up/down.
        self._cpu_intermediate = inter

        def linear_sf(sf: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
            """Swizzled -> linear block scales, as raw e4m3 bytes."""
            lin = convert_swizzled_to_linear(
                sf.view(torch.float8_e4m3fn), rows, cols, 16,
            )
            return lin.reshape(rows, cols // 16).view(torch.uint8).cpu()

        w13_q_cpu = w13_q.cpu()
        w2_q_cpu = w2_q.cpu()
        # Which half of the fused w13 is gate depends on the NVFP4 backend.
        # FlashInfer's kernels want [w3, w1], so
        # prepare_nvfp4_moe_layer_for_fi_or_cutlass calls reorder_w1w3_to_w3w1
        # and the first half becomes *up*, not gate
        # (flashinfer_fp4_moe.py:336). Reading it as gate silently computes
        # silu(x @ up) * (x @ gate) -- a different function, no error, and
        # invisible to a self-check that splits its own reference the same
        # way. Measured cost when it reached 94.7% of routes: 0.49
        # nats/token, indistinguishable from dropping them entirely.
        gate_first = not self._w13_is_reordered()
        for slot, expert_id in enumerate(host_ids):
            sf13 = linear_sf(w13_sf[slot], two_inter, hidden)
            sf2 = linear_sf(w2_sf[slot], hidden, inter)
            gs13 = float(w13_gs[slot])
            lo_q, hi_q = w13_q_cpu[slot, :inter], w13_q_cpu[slot, inter:]
            lo_sf, hi_sf = sf13[:inter], sf13[inter:]
            gate_q, up_q = (lo_q, hi_q) if gate_first else (hi_q, lo_q)
            gate_sf, up_sf = (lo_sf, hi_sf) if gate_first else (hi_sf, lo_sf)
            host.load_expert(
                layer_idx, expert_id,
                PackedPlane(gate_q.contiguous(), gate_sf.contiguous(), gs13),
                PackedPlane(up_q.contiguous(), up_sf.contiguous(), gs13),
                PackedPlane(w2_q_cpu[slot].contiguous(),
                            sf2.contiguous(), float(w2_gs[slot])),
            )
            if os.environ.get("SHOOTING_BRAKE_ARENA_VERIFY_BANK") == "1":
                self._verify_bank_plane(
                    layer_idx, expert_id, gate_q, up_q, w2_q_cpu[slot],
                    gate_sf, up_sf, sf2, gs13, float(w2_gs[slot]),
                    float(qconfig._a1.alpha_or_gscale.flatten()[0]),
                    float(qconfig._a2.alpha_or_gscale.flatten()[0]),
                )

        if os.environ.get("SHOOTING_BRAKE_CPU_VERIFY") == "1":
            from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (  # noqa: E501
                dequantize_to_dtype,
                kE2M1ToFloat_handle,
            )

            # break_fp4_bytes indexes a module-level E2M1 table with the
            # weight's nibbles, so the table must be on the weight's device.
            # vLLM's own emulation callers do exactly this; legal here
            # because weight loading is not graph capture.
            kE2M1ToFloat_handle.val = kE2M1ToFloat_handle.val.to(device)
            ref13 = dequantize_to_dtype(
                w13_q[0], w13_sf[0], w13_gs[0], torch.float32,
            ).cpu()
            ref2 = dequantize_to_dtype(
                w2_q[0], w2_sf[0], w2_gs[0], torch.float32,
            ).cpu()
            # Split the reference by the same backend rule the load used.
            # Hardcoding [:inter] as gate made this check agree with a load
            # that had gate and up swapped -- self-consistent, and blind to
            # the one thing it was written to catch.
            ref_lo = ref13[:inter].contiguous()
            ref_hi = ref13[inter:].contiguous()
            ref_gate, ref_up = (
                (ref_lo, ref_hi) if gate_first else (ref_hi, ref_lo)
            )
            self._verify_cpu_expert(
                host, layer_idx, host_ids[0],
                ref_gate, ref_up, ref2.contiguous(),
            )
            del ref13, ref2
        del w13_q, w2_q, w13_q_cpu, w2_q_cpu

        logger.info(
            "Shooting Brake: layer %d loaded %d host experts into DRAM "
            "(%d cpu-tier, %d b70-prefill) — arena %.2f / %.2f GiB",
            layer_idx, len(host_ids), len(cpu_ids), len(b70_ids),
            host.arena_used_bytes / 2**30, host.arena_capacity_bytes / 2**30,
        )

    def _verify_cpu_expert(
        self,
        host: Any,
        layer_idx: int,
        expert_id: int,
        gate: torch.Tensor,
        up: torch.Tensor,
        down: torch.Tensor,
    ) -> None:
        """Check the packed arena reproduces vLLM's own dequantization.

        This is the step in the CPU tier proven by construction rather than
        by measurement, and every way of getting it wrong yields plausible
        numbers instead of an error: nibble order within a byte, the sign
        bit, e4m3's subnormal branch, the gate/up split of the fused
        ``w13``, the arena's row-major expectations, and the direction the
        global scale folds — vLLM's reference quantiser divides by it where
        ``dequantize_to_dtype`` multiplies.

        The reference planes come from ``dequantize_to_dtype`` on the
        *swizzled* checkpoint tensors, i.e. the same values the CUDA kernel
        sees. That makes this strictly stronger than comparing against
        already-dequantized weights: it also proves the swizzled-to-linear
        scale conversion done at load time, which is the one transformation
        this tier applies to the checkpoint.
        """
        gen = torch.Generator().manual_seed(0)
        x = (torch.randn(4, self.hidden_size, generator=gen) * 0.5).bfloat16()

        xf = x.float()
        want = (
            torch.nn.functional.silu(xf @ gate.T) * (xf @ up.T)
        ) @ down.T

        got = host.expert_forward(layer_idx, expert_id, x)
        err = (got - want).abs().max().item()
        scale = want.abs().max().item()
        ok = torch.allclose(got, want, rtol=1e-2, atol=1e-2)
        # WARNING, not INFO: this is an opt-in self-check whose only purpose
        # is to be read, and vLLM's logger config filters INFO from plugin
        # modules, which silently hid the result of the first all-out run.
        logger.warning(
            "Shooting Brake all-out VERIFY: layer %d expert %d — "
            "max|err|=%.3e ref_max=%.3e rel=%.2e %s",
            layer_idx, expert_id, err, scale,
            err / scale if scale > 0 else float("nan"),
            "OK" if ok else "MISMATCH",
        )
        if not ok:
            raise RuntimeError(
                f"CPU arena FFN disagrees with torch reference for layer "
                f"{layer_idx} expert {expert_id}: max|err|={err:.3e} against "
                f"ref max {scale:.3e}. Weight layout or arena copy is wrong."
            )

    def _register_cpu_poller(self, layer_idx: int) -> None:
        """Bind this layer's flags and pinned buffers to the CPU poller.

        Mirrors :meth:`_register_b70_poller`: a layer owning no CPU experts
        registers nothing and dispatches nothing, so concentrating the tier
        into a few layers costs only those layers a round trip.
        """
        from .cpu_expert_host import get_cpu_poller

        if not self.shooting_brake_placement.is_cpu_active(layer_idx):
            self._cpu_active = False
            return
        if self._cpu_host is None:
            self._cpu_active = False
            logger.warning(
                "Shooting Brake all-out: layer %d has CPU experts but no "
                "arena was loaded; falling back to B70+CUDA only", layer_idx,
            )
            return

        poller = get_cpu_poller(self._cpu_host)
        poller.register_layer(
            layer_idx=layer_idx,
            signal_host=self._cpu_signal_host,
            completion_host=self._cpu_completion_host,
            pinned_hidden=self._cpu_pinned_hidden,
            pinned_ids=self._cpu_pinned_ids,
            pinned_weights=self._cpu_pinned_weights,
            pinned_output=self._cpu_pinned_output,
        )
        poller.start()
        self._cpu_poller = poller
        logger.info(
            "Shooting Brake all-out: layer %d registered with CPU poller",
            layer_idx,
        )

    def _register_b70_poller(self, layer_idx: int) -> None:
        """Bind this layer's flags and pinned buffers to the shared poller.

        A layer that owns no B70 experts becomes pure CUDA passthrough.
        That covers the FP8 layers 32-39, which the bank does not span,
        and every layer a subset placement leaves on CUDA. The gate is
        ownership, not bank capability: dispatching a layer with nothing
        to compute still costs a full round trip per token, which is the
        exact cost a subset placement exists to avoid.

        Route math is unchanged — with no B70-owned experts the partition
        produces no remote routes anyway.
        """
        from .b70_poller import get_b70_poller

        if not self.shooting_brake_placement.is_b70_active(layer_idx):
            self._b70_graph_mode = False
            # Passthrough means *nothing* is offloaded, which is not the same
            # as owning no B70 experts: an all-out placement can leave a
            # layer with CPU experts and no B70 ones. Claiming passthrough
            # there would return before the CPU tier ever dispatched and
            # silently drop every cold route, so the placement is asked
            # about both tiers rather than just this one.
            if self.shooting_brake_placement.is_cpu_active(layer_idx):
                logger.info(
                    "Shooting Brake Tier 3: layer %d owns no B70 experts but "
                    "does own CPU experts — CPU tier stays active",
                    layer_idx,
                )
                return
            self._all_cuda_passthrough = True
            logger.info(
                "Shooting Brake Tier 3: layer %d owns no B70 experts — "
                "all-CUDA passthrough", layer_idx,
            )
            return

        validate_dispatch_buffer_shapes(
            self._dispatch_geometry,
            pinned_hidden=self._b70_pinned_hidden,
            pinned_ids=self._pinned_b70_ids,
            pinned_weights=self._pinned_b70_weights,
            pinned_output=self._b70_pinned_output,
        )
        poller = get_b70_poller(self.shooting_brake_placement)
        poller.register_layer(
            layer_idx=layer_idx,
            signal_host=self._signal_host,
            completion_host=self._completion_host,
            pinned_hidden=self._b70_pinned_hidden,
            pinned_ids=self._pinned_b70_ids,
            pinned_weights=self._pinned_b70_weights,
            pinned_output=self._b70_pinned_output,
        )
        poller.start()
        self._b70_poller = poller
        logger.info(
            "Shooting Brake Tier 3: layer %d registered with B70 poller",
            layer_idx,
        )

    def _first_forward_setup(self, topk_ids: torch.Tensor) -> None:
        """One-time per-layer setup, run on the very first forward.

        Only the poller binding lives here. It loads the 13.5 GiB expert
        bank and creates the SYCL queue, which must happen neither during
        model construction (it races CUDA context setup and starves
        weight loading of CPU) nor during CUDA graph capture (it would
        race an active stream capture). vLLM's first forward is the eager
        profiling pass, which sits between the two.

        VRAM surgery and the CPU arena load are deliberately *not* here:
        both run earlier, off ``process_weights_after_loading``, so the
        freed memory is already reflected in the profiling peak vLLM sizes
        the KV cache from. See :func:`_install_surgery_hook`.
        """
        self._ensure_layer_device_map(topk_ids)
        if self._b70_graph_mode:
            self._register_b70_poller(self.layer_index)
        if self._cpu_active:
            self._register_cpu_poller(self.layer_index)
        if route_stats.enabled() or route_stats.trace_enabled():
            # Geometry comes from the resolved model, never the legacy 35B
            # aliases in config.py. This model is 48 layers x 180 experts:
            # sizing at 40 x 256 indexes past the end of the histogram on
            # layers 40-47, and hands the locality analyzer an expert count of
            # 256 against an actual 180 -- which silently corrupts its
            # uniform-routing chance baseline, the one number that makes a
            # measured reuse figure mean anything.
            geometry = self.shooting_brake_qualified_model
            if route_stats.enabled():
                self._route_histogram = route_stats.get_counter(
                    geometry.num_layers, geometry.num_experts, topk_ids.device,
                )
            if route_stats.trace_enabled():
                self._route_trace = route_stats.get_trace(
                    geometry.num_layers, geometry.num_experts, topk_ids,
                )
        self._first_forward_done = True

    @eager_break_during_capture
    def _observe_route_trace(self, topk_ids: torch.Tensor) -> None:
        """Run the D2H trace copy eagerly between breakable graph segments."""
        self._route_trace.observe(self.layer_index, topk_ids)

    def forward_modular(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: Any = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dispatch to hybrid or CUDA-only forward.

        In all-CUDA mode (no hybrid features) this is a pure pass-through
        to ``super().forward_modular()`` — CUDA graph compatible.

        Under Tier 3, decode-sized batches take the graph-compatible B70
        path; prefill stays all-CUDA because the pinned staging buffers
        are sized for decode.
        """
        if not self._first_forward_done:
            self._first_forward_setup(topk_ids)
        elif self._b70_poller is not None:
            # A failed dispatch leaves stale data in the output buffer:
            # the poller raises the completion flag regardless, because
            # the CUDA-side wait cannot time out and an unset flag wedges
            # the device.  Replay runs no Python, so surface it here — on
            # the first eager forward after the fault — instead of
            # returning silently wrong routes.
            errors = self._b70_poller.error_count
            if errors:
                raise RuntimeError(
                    f"Shooting Brake Tier 3: {errors} B70 dispatch(es) "
                    "failed; routed-expert output is not trustworthy"
                )

        # Route histogram (SHOOTING_BRAKE_ROUTE_STATS=1), distinct from
        # _route_counter, which is the 3-element B70/CPU share counter that
        # telemetry reads. This sits above the all-CUDA passthrough
        # deliberately: routing is decided by the router before any dispatch,
        # so it does not depend on placement, and calibrating in all-CUDA
        # mode is both faster and free of B70 round trips. The ids here are
        # global -- the CUDA remap below rewrites them into compacted local
        # indices, which would make the table describe the placement instead
        # of the model.
        if self._route_histogram is not None:
            self._route_histogram.observe(self.layer_index, topk_ids)
        # Token-level locality trace (SHOOTING_BRAKE_ROUTE_TRACE=<path>).
        # When disabled, this single None check is its only hot-path cost.
        # The eager break makes the side effect run on graph replay rather
        # than only once while the graph is captured.
        if self._route_trace is not None:
            self._observe_route_trace(topk_ids)

        # A layer with nothing offloaded has no remote work to do at any
        # batch size.
        if self._all_cuda_passthrough:
            return super().forward_modular(
                x, topk_weights, topk_ids,
                shared_experts, shared_experts_input,
            )

        # Prefill exceeds the decode-sized staging buffers, so it takes its
        # own path (Phase 6) rather than the graph-captured one: it chunks
        # the B70 dispatch to fit those buffers and streams cold expert
        # weights to CUDA instead of computing them on CPU cores.
        #
        # This replaces an earlier branch that simply handed the parent raw
        # global expert ids. That was wrong after VRAM surgery — surgery
        # compacts the CUDA weight tensor to CUDA-owned experts only, so a
        # global id no longer addresses the expert it names — and it
        # silently contributed nothing for offloaded routes.
        if x.shape[0] > self._b70_max_batch:
            return self._prefill_forward_offloaded(
                x, topk_weights, topk_ids,
                shared_experts, shared_experts_input,
            )
        return self._hybrid_forward_modular(
            x, topk_weights, topk_ids,
            shared_experts, shared_experts_input,
        )

    def _record_expected_arm_runtime(self) -> None:
        """Record the branch actually reached for a timed arm descriptor."""
        if self._expected_arm is None:
            return
        runtime_mode = CUDAGraphMode.NONE
        if is_forward_context_available():
            runtime_mode = get_forward_context().cudagraph_runtime_mode
        stream_capturing = torch.cuda.is_current_stream_capturing()
        branch = _classify_expected_arm_runtime(
            expected_arm=self._expected_arm,
            graph_mode=self._b70_graph_mode,
            runtime_mode=runtime_mode,
            stream_capturing=stream_capturing,
        )
        if branch is None:
            return
        observation = (runtime_mode.name, branch, stream_capturing)
        if observation in self._arm_runtime_observations:
            return
        self._arm_runtime_observations.add(observation)
        logger.info(
            "Shooting Brake benchmark arm layer=%s runtime_mode=%s "
            "stream_capturing=%s branch=%s",
            self.layer_name,
            runtime_mode.name,
            stream_capturing,
            branch,
        )

    @eager_break_during_capture
    def _hybrid_forward_modular(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: Any = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Hybrid MoE forward.

        Two paths. Under Tier 3 the whole B70 dispatch is CUDA stream
        operations, captured by a normal CUDA graph with no Python in the
        replay. Otherwise the eager path partitions routes, validates the
        split, and dispatches host-side.

        Per-layer setup (index resolution, VRAM surgery, poller binding)
        has already run in ``forward_modular``; it must happen on the
        profiling pass, which never reaches this method.
        """
        layer_idx, dml = self._ensure_layer_device_map(topk_ids)
        self._record_expected_arm_runtime()
        if not self._b70_graph_mode:
            _validate_dispatch_capture_mode(
                graph_mode=False,
                stream_capturing=torch.cuda.is_current_stream_capturing(),
            )
        if self._b70_graph_mode:
            # Tier 3: pure CUDA stream ops — no partition, validation,
            # stats, or any host-side sync.  Every op below is captured
            # by torch.cuda.graph() without modification.
            if self._b70_slot_map_cuda is None:
                raise RuntimeError(
                    "B70 CUDA slot map was not initialized before graph capture"
                )
            b70_ids = self._b70_slot_map_cuda[topk_ids]
            b70_mask = b70_ids >= 0
            if self._cpu_active:
                cpu_ids = self._cpu_id_map_cuda[topk_ids]
                cpu_mask = cpu_ids >= 0
                # Both offloaded tiers contribute through their own
                # partial, so both must be zeroed out of the CUDA weights.
                # Missing either double-counts those routes.
                offloaded = b70_mask | cpu_mask
            else:
                offloaded = b70_mask
            # The compaction map keeps -1 for every offloaded global id.
            # `compact_cuda_routes` masks those weights before replacing the
            # sentinel with dummy local slot zero.
            if self._cuda_remap is not None:
                cuda_topk_ids, cuda_weights, _ = compact_cuda_routes(
                    topk_ids, topk_weights, self._cuda_remap,
                )
            else:
                cuda_topk_ids = topk_ids
                cuda_weights = (
                    topk_weights * (~offloaded).to(topk_weights.dtype)
                )
            if self._route_counter is not None:
                # Device-side accumulation: no host sync, so this stays
                # inside the captured graph.  Read between steps.
                self._route_counter[0] += b70_mask.sum()
                self._route_counter[1] += b70_mask.numel()
                if self._cpu_active:
                    self._route_counter[2] += cpu_mask.sum()
            # Issue both remote tiers before the CUDA work so both overlap
            # with it. The CPU tier is the slower of the two (~195us vs
            # ~40us), so it must start first to have any chance of hiding
            # under the CUDA expert compute.
            if self._cpu_active:
                self._cpu_issue_graph(x, cpu_ids, topk_weights)
            self._b70_issue_graph(x, b70_ids, topk_weights)
            y_cuda = super().forward_modular(
                x, cuda_weights, cuda_topk_ids,
                shared_experts, shared_experts_input,
            )
            y_b70 = self._b70_take_graph(x.shape[0])
            if self._cpu_active:
                return y_cuda + y_b70 + self._cpu_take_graph(x.shape[0])
            return y_cuda + y_b70
        part = partition_routes(topk_ids, topk_weights, dml, layer_idx)
        if os.environ.get("SHOOTING_BRAKE_VALIDATE_PARTITION") == "1":
            validate_partition(
                part, self.shooting_brake_placement.b70_capable_layers
            )
        if self._cuda_remap is not None:
            cuda_topk_ids, compact_cuda_weights, _ = compact_cuda_routes(
                topk_ids, topk_weights, self._cuda_remap,
            )
        else:
            cuda_topk_ids = topk_ids
            compact_cuda_weights = None

        stats = self.shooting_brake_partition_stats
        stats["steps"] += 1
        marker = os.environ.get("SHOOTING_BRAKE_PARTITION_MARKER")
        shadow_enabled = os.environ.get("SHOOTING_BRAKE_SHADOW") == "1"
        hybrid_marker = os.environ.get("SHOOTING_BRAKE_HYBRID_MARKER")
        diagnostics_enabled = (
            self._b70_stats
            or marker is not None
            or shadow_enabled
            or hybrid_marker is not None
        )
        has_remote = False
        n_remote = 0
        if diagnostics_enabled:
            # These conversions synchronize CUDA. They are diagnostics only,
            # and every enabling switch above is explicitly opt-in.
            has_remote = part.has_remote()
            if has_remote:
                n_remote = part.num_b70_routes()
                stats["remote_steps"] += 1
                stats["remote_routes"] += n_remote
                stats["max_remote_per_step"] = max(
                    stats["max_remote_per_step"], n_remote
                )
                if stats["remote_steps"] == 1:
                    logger.info(
                        "Shooting Brake Phase-6a: layer %s first remote step "
                        "(cuda=%d b70=%d routes, M=%d).",
                        layer_idx,
                        part.num_cuda_routes(),
                        n_remote,
                        topk_ids.shape[0],
                    )
                    if marker:
                        import json
                        with open(marker, "a") as f:
                            f.write(json.dumps({
                                "layer": layer_idx,
                                "b70_routes": n_remote,
                                "M": topk_ids.shape[0],
                            }) + "\n")

        # Phase 6b: expensive value validation is an explicit debug path.
        if (
            not self._shadow_done
            and has_remote
            and shadow_enabled
            and self._cuda_remap is None
        ):
            self._shadow_done = True
            self._shadow_validate(x, topk_weights, topk_ids, part)

        if (
            self._hybrid_env
            and has_remote
            and stats.get("hybrid_steps", 0) == 0
        ):
            stats["hybrid_steps"] = 1
            if hybrid_marker:
                import json
                with open(hybrid_marker, "w") as f:
                    json.dump({"layer": layer_idx, "b70_routes": n_remote}, f)

        # A remotely-owned layer always dispatches its B70 partial. The
        # provider's -1 skip ABI makes an all-CUDA route set a valid zero
        # remote partial, avoiding a tensor-to-host `any()` in normal decode.
        if self._hybrid_env:
            cuda_weights = (
                compact_cuda_weights
                if compact_cuda_weights is not None
                else topk_weights * (~part.b70_mask).to(topk_weights.dtype)
            )

            if self._b70_graph_mode:
                # Tier 3: graph-compatible B70 dispatch (pure CUDA stream ops).
                self._b70_issue_graph(x, topk_ids, topk_weights)
                y_cuda = super().forward_modular(
                    x, cuda_weights, cuda_topk_ids,
                    shared_experts, shared_experts_input,
                )
                y_b70 = self._b70_take_graph(x.shape[0])
            elif self._b70_device_cached and self._b70_async_cached:
                # Phase 8a: async overlap — B70 kernel runs during CUDA.
                seq, b70_M = self._b70_issue(
                    x, topk_ids, topk_weights, part, layer_idx,
                )
                y_cuda = super().forward_modular(
                    x, cuda_weights, cuda_topk_ids,
                    shared_experts, shared_experts_input,
                )
                y_b70 = self._b70_take(seq, b70_M, x.device, x.dtype)
            elif self._b70_device_cached:
                # Phase 7: synchronous B70 (correctness reference).
                y_cuda = super().forward_modular(
                    x, cuda_weights, cuda_topk_ids,
                    shared_experts, shared_experts_input,
                )
                y_b70 = self._b70_partial(
                    x, topk_ids, topk_weights, part, layer_idx,
                )
            else:
                # Phase 6c: CUDA-kernel B70 partial (no device).
                y_cuda = super().forward_modular(
                    x, cuda_weights, cuda_topk_ids,
                    shared_experts, shared_experts_input,
                )
                b70_weights = topk_weights * part.b70_mask.float()
                y_b70 = super().forward_modular(
                    x, b70_weights, cuda_topk_ids,
                )
            self._maybe_capture_aggregation_oracle(
                layer_idx, x, topk_ids, topk_weights, y_cuda, y_b70,
            )
            return self._write_static_output(y_cuda + y_b70)

        return self._write_static_output(super().forward_modular(
            x, topk_weights, cuda_topk_ids, shared_experts, shared_experts_input
        ))

    def _maybe_capture_aggregation_oracle(
        self,
        layer_idx: int,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        normal_cuda: torch.Tensor,
        normal_b70: torch.Tensor,
    ) -> None:
        """Capture one real layer for the opt-in split-aggregation oracle.

        The layer's actual activation, router ids, router weights, and hybrid
        partials are saved.  Three additional routed-only cases reuse the
        real activation and weights with deterministic global ids: all CUDA,
        all B70, and a 4/4 straddle.  They therefore exercise the compact
        NVFP4 tensor and the int4 provider with identical arithmetic inputs.
        The offline CPU half independently dequantizes the canonical int4
        bank and checks the B70 partial plus the cross-format aggregation.
        """
        path = os.environ.get("SHOOTING_BRAKE_AGGREGATION_CAPTURE")
        target_layer = int(
            os.environ.get("SHOOTING_BRAKE_AGGREGATION_LAYER", "0")
        )
        if (
            path is None
            or self._aggregation_capture_done
            or layer_idx != target_layer
        ):
            return
        if (
            self._b70_graph_mode
            or self._b70_async_cached
            or not self._b70_device_cached
            or torch.cuda.is_current_stream_capturing()
        ):
            raise RuntimeError(
                "aggregation capture requires synchronous eager B70 dispatch"
            )
        if self._cuda_remap is None:
            raise RuntimeError(
                "aggregation capture requires the compact CUDA expert map"
            )

        cuda_ids = self.shooting_brake_placement.cuda_expert_ids(layer_idx)
        b70_ids = self.shooting_brake_placement.b70_expert_ids(layer_idx)
        if len(cuda_ids) < self.top_k or len(b70_ids) < self.top_k:
            raise RuntimeError(
                "aggregation capture needs at least top_k experts on each tier"
            )
        case_ids = {
            "all_cuda": cuda_ids[:self.top_k],
            "all_b70": b70_ids[:self.top_k],
            "straddling": (
                cuda_ids[:self.top_k // 2]
                + b70_ids[:self.top_k - self.top_k // 2]
            ),
        }
        payload: dict[str, Any] = {
            "format": "shooting-brake-int4-aggregation-v1",
            "layer": layer_idx,
            "actual_x": x.detach().cpu(),
            "actual_global_ids": topk_ids.detach().cpu(),
            "actual_weights": topk_weights.detach().cpu(),
            "actual_cuda_partial": normal_cuda.detach().cpu(),
            "actual_b70_partial": normal_b70.detach().cpu(),
            "actual_hybrid_output": (normal_cuda + normal_b70).detach().cpu(),
            "cases": {},
        }
        for name, global_ids in case_ids.items():
            ids = torch.tensor(
                global_ids, dtype=topk_ids.dtype, device=topk_ids.device,
            ).expand(topk_ids.shape[0], -1)
            part = partition_routes(
                ids, topk_weights, self._device_map_layer, layer_idx,
            )
            local_ids, cuda_weights, _ = compact_cuda_routes(
                ids, topk_weights, self._cuda_remap,
            )
            with torch.no_grad():
                cuda_partial = super().forward_modular(
                    x, cuda_weights, local_ids,
                )
                b70_partial = self._b70_partial(
                    x, ids, topk_weights, part, layer_idx,
                )
            payload["cases"][name] = {
                "global_ids": ids.detach().cpu(),
                "weights": topk_weights.detach().cpu(),
                "cuda_partial": cuda_partial.detach().cpu(),
                "b70_partial": b70_partial.detach().cpu(),
                "hybrid_routed": (
                    cuda_partial + b70_partial
                ).detach().cpu(),
            }
        torch.save(payload, path)
        self._aggregation_capture_done = True
        logger.warning(
            "Shooting Brake aggregation oracle captured layer %d to %s",
            layer_idx,
            path,
        )


    def _write_static_output(self, result: torch.Tensor) -> torch.Tensor:
        """Copy result into static buffer for stable address across replays.

        Required by ``@eager_break_during_capture``: the output tensor's
        address must not change between capture and replay, or downstream
        graph segments read from a stale address.
        """
        M = result.shape[0]
        if (
            self._static_output is None
            or self._static_output.shape[0] < M
            or self._static_output.dtype != result.dtype
            or self._static_output.device != result.device
        ):
            max_b = max(
                M,
                int(os.environ.get("SHOOTING_BRAKE_B70_MAX_BATCH", "128")),
            )
            self._static_output = torch.empty(
                max_b, result.shape[1],
                dtype=result.dtype, device=result.device,
            )
        self._static_output[:M].copy_(result)
        return self._static_output[:M]

    def _b70_partial(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        part: RoutePartition,
        layer_idx: int,
    ) -> torch.Tensor:
        """Compute the B70-device partial for B70-owned routes.

        Translates global expert IDs to B70 compact slots, converts the
        activation BF16 → FP16 for the B70 kernel, dispatches via the
        in-process QuixiCore provider, and returns the weighted partial
        as a BF16 CUDA tensor ready for addition.
        """
        provider = _get_b70_provider(self.shooting_brake_placement)

        # Compact ID translation: global expert → B70 slot, CUDA routes → -1
        ids_np = topk_ids.detach().cpu().numpy()
        wts_np = topk_weights.detach().cpu().numpy()
        b70_ids = self._b70_slot_map[ids_np]          # [M, topk]
        b70_weights = wts_np * (b70_ids >= 0).astype(np.float32)

        # Activation: BF16 CUDA → FP16 CPU for the B70 NVFP4 kernel
        hidden_fp16 = x.detach().to(torch.float16).cpu().numpy()

        output_np = provider.dispatch(
            layer=layer_idx,
            hidden_fp16=hidden_fp16,
            ids=b70_ids,
            weights=b70_weights,
        )

        # Result: FP32 CPU → BF16 CUDA
        return torch.from_numpy(output_np).to(x.device).to(x.dtype)

    def _b70_issue(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        part: RoutePartition,
        layer_idx: int,
    ) -> tuple[int, int]:
        """Issue B70 dispatch — kernel starts asynchronously. Returns (seq, M).

        First half of the Phase-8a async overlap:
        1.  D2H the activation (BF16 CUDA → FP16 pinned host, DMA copy).
        2.  D2H routing data and translate global IDs → B70 compact slots.
        3.  Submit to the B70 provider (SYCL queue enqueues kernel, returns).

        After this returns, the caller should do CUDA work (the routed-
        expert forward) and then call :meth:`_b70_take` to collect.
        """
        provider = _get_b70_provider(self.shooting_brake_placement)
        M = x.shape[0]

        # D2H activation: BF16 CUDA → FP16 pinned [M, model hidden] buffer.
        # The width is `_dispatch_geometry.hidden_size` (3072 for step 1).
        x_fp16 = x.detach().to(torch.float16)
        pinned_hidden = self._b70_pinned_hidden[:M]
        pinned_hidden.copy_(x_fp16, non_blocking=True)
        torch.cuda.current_stream().synchronize()

        # D2H routing data (tiny: M × model top-k × 4 B).
        ids_np = topk_ids.detach().cpu().numpy()
        wts_np = topk_weights.detach().cpu().numpy()

        # Translate global expert IDs → B70 compact slots, zero CUDA routes.
        b70_ids = self._b70_slot_map[ids_np]
        b70_weights = wts_np * (b70_ids >= 0).astype(np.float32)

        # Submit — SYCL kernel starts on B70, returns immediately.
        seq = provider.issue(
            layer=layer_idx,
            hidden_fp16=pinned_hidden.numpy(),
            ids=b70_ids,
            weights=b70_weights,
        )
        return seq, M

    def _b70_take(
        self,
        sequence: int,
        M: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Collect B70 result. Blocks only if the kernel is still running.

        Second half of the Phase-8a async overlap.  By the time this is
        called (after CUDA forward_modular), the B70 kernel has typically
        already finished — ``sb_b70_take`` returns immediately.

        Writes into the pre-allocated pinned output buffer, then does a
        non-blocking H2D copy (pinned source enables DMA) + dtype cast.
        """
        provider = _get_b70_provider(self.shooting_brake_placement)
        pinned_output = self._b70_pinned_output[:M]
        provider.take(sequence, M, output=pinned_output.numpy())
        # H2D from pinned (DMA) + cast FP32 → model dtype on GPU.
        return pinned_output.to(device, non_blocking=True).to(dtype)

    def _b70_issue_graph(
        self,
        x: torch.Tensor,
        b70_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        """Graph-compatible B70 issue — pure CUDA stream operations.

        Replaces the eager-mode ``_b70_issue``, which uses ``.cpu()``,
        numpy, and ctypes and so cannot be captured.  Every operation
        here is a CUDA stream op that ``torch.cuda.graph()`` records
        unmodified:

          1. dtype cast BF16 -> FP16 (CUDA)
          2. D2H copy activation to pinned (cudaMemcpyAsync)
          3. D2H copy routing data to pinned (cudaMemcpyAsync)
          4. cuStreamWriteValue32 signal flag = M (Driver API)

        Args:
            x: [M, hidden] activation on CUDA.
            b70_ids: [M, topk] compact B70 slots, -1 for CUDA-owned
                routes.  Gathered by the caller, which already needs the
                mask to zero CUDA weights.
            topk_weights: [M, topk] routing weights, unmodified.
        """
        from .stream_signal import write_flag
        M = x.shape[0]

        # 1-2. Activation: BF16 -> FP16, D2H to pinned.
        self._b70_pinned_hidden[:M].copy_(
            x.to(torch.float16), non_blocking=True,
        )

        # 3. Routing data.  CUDA-owned routes carry slot -1 and are
        # skipped by the kernel, so their weights are irrelevant; the
        # weights go over as-is.
        self._pinned_b70_ids[:M].copy_(b70_ids, non_blocking=True)
        self._pinned_b70_weights[:M].copy_(topk_weights, non_blocking=True)

        # 4. Signal the poller.  The flag VALUE is M, so the poller
        # dispatches exactly this batch instead of the whole buffer.
        # M is a constant at capture time (one graph per batch size),
        # so it is baked into the replayed write — no extra transfer.
        write_flag(self._signal_dev, M)

    def _b70_take_graph(
        self,
        M: int,
    ) -> torch.Tensor:
        """Graph-compatible B70 take — pure CUDA stream operations.

        Replaces the eager-mode ``_b70_take`` which uses ctypes and
        .to(device).  Every operation here is captured by the graph:

          1. cuStreamWaitValue32 completion flag (Driver API)
          2. H2D copy FP32 result to device (cudaMemcpyAsync)
          3. dtype cast FP32→BF16 (CUDA)
          4. cuStreamWriteValue32 reset completion (Driver API)
        """
        from .stream_signal import wait_flag, write_flag

        # 1. Wait for B70 thread to finish.
        wait_flag(self._completion_dev, 1)

        # 2-3. H2D copy + dtype cast.
        self._dev_b70_fp32[:M].copy_(
            self._b70_pinned_output[:M], non_blocking=True,
        )
        self._dev_b70_bf16[:M].copy_(self._dev_b70_fp32[:M])

        # 4. Reset completion for next replay.
        write_flag(self._completion_dev, 0)

        return self._dev_b70_bf16[:M]

    def _cpu_issue_graph(
        self,
        x: torch.Tensor,
        cpu_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        """Graph-compatible CPU-tier issue — pure CUDA stream operations.

        Identical in shape to :meth:`_b70_issue_graph`; only the staged
        dtype and the flag differ. The activation crosses as bf16 because
        the CPU kernels consume bf16 natively, so unlike the B70 path there
        is no cast.

        Args:
            x: [M, hidden] activation on CUDA.
            cpu_ids: [M, topk] global expert ids for CPU-owned routes,
                -1 elsewhere. Gathered by the caller, which already needs
                the mask to zero the CUDA weights.
            topk_weights: [M, topk] routing weights, unmodified.
        """
        from .stream_signal import write_flag
        M = x.shape[0]

        self._cpu_pinned_hidden[:M].copy_(x, non_blocking=True)
        self._cpu_pinned_ids[:M].copy_(cpu_ids, non_blocking=True)
        self._cpu_pinned_weights[:M].copy_(topk_weights, non_blocking=True)

        # The flag VALUE is M, so the poller computes exactly this batch
        # rather than the whole staging buffer. M is constant at capture
        # time (one graph per batch size), so it is baked into the replay.
        write_flag(self._cpu_signal_dev, M)

    def _cpu_take_graph(self, M: int) -> torch.Tensor:
        """Graph-compatible CPU-tier take — pure CUDA stream operations."""
        from .stream_signal import wait_flag, write_flag

        wait_flag(self._cpu_completion_dev, 1)
        self._dev_cpu_fp32[:M].copy_(
            self._cpu_pinned_output[:M], non_blocking=True,
        )
        self._dev_cpu_bf16[:M].copy_(self._dev_cpu_fp32[:M])
        write_flag(self._cpu_completion_dev, 0)
        return self._dev_cpu_bf16[:M]

    # -- Phase 6: prefill with offloaded experts ---------------------------

    def _prefill_forward_offloaded(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: Any = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Prefill for a layer whose experts are not all on CUDA.

        Sums the same three partials as the decode path and relies on the
        same identity — every route is owned by exactly one tier, so the
        weighted partials add up to the full routed output. What differs is
        only how each remote partial is obtained, because the decode path's
        staging buffers are sized for decode:

          * CUDA — offloaded routes zeroed, ids remapped through surgery's
            compaction.
          * B70  — dispatched in ``_b70_max_batch``-sized chunks.
          * CPU  — weights streamed to CUDA and computed there, since CPU
            cores turn compute-bound at this M.

        This is not merely faster than the branch it replaced. That branch
        passed raw global ids into the surgically compacted weight tensor
        and contributed nothing for offloaded routes, which cost ~0.49
        nats/token of prompt logprob against an all-CUDA baseline while
        still emitting identical tokens (src/phase7/prefill_probe.py).
        """
        layer_idx, _ = self._ensure_layer_device_map(topk_ids)

        b70_ids = self._b70_slot_map_cuda[topk_ids]
        b70_mask = b70_ids >= 0
        cpu_ids = None
        if self._cpu_active:
            cpu_ids = self._cpu_id_map_cuda[topk_ids]
            offloaded = b70_mask | (cpu_ids >= 0)
        else:
            offloaded = b70_mask

        if self._cuda_remap is not None:
            cuda_topk_ids, cuda_weights, _ = compact_cuda_routes(
                topk_ids, topk_weights, self._cuda_remap,
            )
        else:
            cuda_topk_ids = topk_ids
            cuda_weights = topk_weights * (~offloaded).to(topk_weights.dtype)

        y = super().forward_modular(
            x, cuda_weights, cuda_topk_ids,
            shared_experts, shared_experts_input,
        )
        if cpu_ids is not None:
            y = y + self._cpu_prefill_partial(
                layer_idx, x, cpu_ids, topk_weights
            )
        if self._b70_graph_mode:
            # Two id spaces, and they are not interchangeable. The provider
            # is addressed by compact per-layer slot (``_b70_slot_map``);
            # the DRAM arena is keyed by global expert id, exactly like the
            # cold tier's ``_cpu_id_map``. Streaming reads the arena, so it
            # needs the global form -- handing it slots looks up a different
            # expert and, for a slot below the CUDA-resident count, one that
            # was never loaded at all.
            b70_global_ids = (
                torch.where(b70_mask, topk_ids, -1)
                if self._b70_prefill_stream
                else None
            )
            y = y + self._b70_prefill_partial(
                layer_idx, x, b70_ids, topk_weights, b70_global_ids,
            )
        return y

    def _b70_prefill_partial(
        self,
        layer_idx: int,
        x: torch.Tensor,
        b70_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        b70_global_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """B70 partial for a batch larger than the staging buffers.

        Two strategies, chosen by token count, because they have different
        cost curves against the same work:

        * **Dispatch** (default) sends tokens to the B70 and leaves the
          weights where they are. Cost scales with tokens — ~26.9 us per
          token per layer, measured. Chunked rather than resized: sizing the
          pinned buffers for the largest prefill would cost ~48 MiB of
          pinned host memory and ~48 MiB of VRAM *per layer*, and that VRAM
          comes straight out of the KV cache this tier exists to grow.
          Each chunk's result is copied out before the next dispatch,
          because ``_b70_take_graph`` returns a view the next chunk
          overwrites.
        * **Streaming** (``SHOOTING_BRAKE_B70_PREFILL_STREAM=1``, at or
          above :func:`b70_stream_threshold`) moves the weights to the 5090
          once and computes there. Cost is flat in M — 0.41 GiB per layer
          over the 5090's direct Gen5 x16 link — so it wins once there are
          enough tokens to amortise the transfer.

        Chunking is not what makes dispatch slow here: collapsing 25
        dispatches into 2 moves TTFT by 5.1% (src/phase7/prefill_chunk_bench.py).
        The B70's NVFP4 kernel runs at ~3.8x its own VRAM-bandwidth floor on
        prefill shapes because it does not amortise weight reads across
        tokens — it was written for decode, where an expert sees one or two
        rows. Streaming sidesteps that kernel rather than fixing it.

        Decode never reaches this method at all, so the B70 keeps serving
        the regime it is good at, where dispatch beats streaming ~9x at M=1.
        """
        if (
            self._b70_prefill_stream
            and b70_global_ids is not None
            and x.shape[0] >= b70_stream_threshold()
            and self._cpu_host is not None
            # Choosing which experts to move is data-dependent, so the
            # streamer syncs to read the expert list; a host sync during
            # capture invalidates the graph. Same guard as the cold tier.
            and not torch.cuda.is_current_stream_capturing()
        ):
            from .cpu_stream import get_streamer

            streamer = get_streamer(
                self._cpu_host, self.hidden_size, self._cpu_intermediate,
            )
            # Global ids, not the compact slots the provider takes: the
            # arena is keyed by (layer, global expert).
            return streamer.forward(
                layer_idx, x, b70_global_ids, topk_weights,
            )

        M = x.shape[0]
        cap = self._b70_max_batch
        out = torch.empty(M, self.hidden_size, dtype=x.dtype, device=x.device)
        for start in range(0, M, cap):
            end = min(start + cap, M)
            self._b70_issue_graph(
                x[start:end], b70_ids[start:end], topk_weights[start:end],
            )
            out[start:end] = self._b70_take_graph(end - start)
        return out

    def _cpu_prefill_partial(
        self,
        layer_idx: int,
        x: torch.Tensor,
        cpu_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Cold-tier partial for a prefill-sized batch.

        Above :func:`cpu_stream.stream_threshold` the weights are streamed to
        CUDA and the GEMM runs there; below it they are computed in place on
        CPU cores. The threshold exists because weight traffic per expert is
        fixed while arithmetic grows with the token count, so the CPU crosses
        from bandwidth-bound to compute-bound and then degrades sharply.

        Both sides are implemented so the crossover can be measured rather
        than assumed — see src/phase7/stream_crossover_bench.py.

        Streaming is skipped outright while a CUDA graph is being captured.
        Choosing which experts to move is data-dependent, so the streamer
        syncs to the host to read the expert list, and a host sync during
        capture invalidates the whole graph. vLLM captures at token counts
        driven by ``max_num_batched_tokens``, not just ``max_num_seqs``, so
        a batch large enough to reach this path does get captured on real
        configurations -- it aborted engine startup at max_num_seqs=80. The
        flag-driven path below is pure stream operations and captures
        cleanly, which is exactly why decode uses it.
        """
        from .cpu_stream import get_streamer, stream_threshold

        capturing = torch.cuda.is_current_stream_capturing()
        if x.shape[0] >= stream_threshold() and not capturing:
            streamer = get_streamer(
                self._cpu_host, self.hidden_size, self._cpu_intermediate,
            )
            return streamer.forward(layer_idx, x, cpu_ids, topk_weights)

        M = x.shape[0]
        cap = self._b70_max_batch
        out = torch.empty(M, self.hidden_size, dtype=x.dtype, device=x.device)
        for start in range(0, M, cap):
            end = min(start + cap, M)
            self._cpu_issue_graph(
                x[start:end], cpu_ids[start:end], topk_weights[start:end],
            )
            out[start:end] = self._cpu_take_graph(end - start)
        return out

    def _shadow_validate(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        part: RoutePartition,
    ) -> None:
        """Validate Y_cuda_routed + Y_b70_routed ≈ Y_full_routed.

        Computes the routed-expert output (shared expert excluded) three
        ways using the stock CUDA kernel with different weight masks:
          * all routes (original weights)
          * CUDA routes only (B70-route weights zeroed)
          * B70 routes only (CUDA-route weights zeroed)

        Since 0 * E_k(x) == 0 exactly, the only source of difference is
        BF16 summation reordering. The gate confirms the split-merge
        identity that Phase 6c will rely on. (The B70 device kernel's
        own accuracy was independently validated in Phase 3.)
        """
        cuda_w = topk_weights * (~part.b70_mask).float()
        b70_w = topk_weights * part.b70_mask.float()

        with torch.no_grad():
            y_full = super().forward_modular(x, topk_weights, topk_ids)
            y_cuda = super().forward_modular(x, cuda_w, topk_ids)
            y_b70 = super().forward_modular(x, b70_w, topk_ids)

        y_sum = y_cuda + y_b70
        diff = (y_sum - y_full)
        max_abs = float(diff.abs().max())
        mean_abs = float(diff.abs().mean())
        y_norm = float(y_full.float().norm())
        rel_rmse = float(diff.float().norm()) / max(y_norm, 1e-12)
        cos = float(
            torch.dot(
                y_sum.float().flatten(), y_full.float().flatten()
            )
            / (y_sum.float().norm() * y_full.float().norm() + 1e-12)
        )
        has_nan = bool(torch.isnan(y_sum).any() or torch.isnan(y_full).any())

        logger.info(
            "Shooting Brake Phase-6b shadow: max_abs=%.6f mean_abs=%.6f "
            "rel_rmse=%.6f cosine=%.8f nan=%s",
            max_abs, mean_abs, rel_rmse, cos, has_nan,
        )
        marker = os.environ.get("SHOOTING_BRAKE_SHADOW_MARKER")
        if marker:
            import json
            with open(marker, "w") as f:
                json.dump({
                    "max_abs": max_abs,
                    "mean_abs": mean_abs,
                    "rel_rmse": rel_rmse,
                    "cosine": cos,
                    "has_nan": has_nan,
                    "M": topk_ids.shape[0],
                    "b70_routes": part.num_b70_routes(),
                    "gate_pass": (max_abs < 0.1 and cos > 0.999 and not has_nan),
                }, f)
