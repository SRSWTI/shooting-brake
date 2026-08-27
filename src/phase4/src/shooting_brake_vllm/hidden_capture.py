"""Flag-gated DFlash verifier hidden-state capture for vLLM V1.

The hook is intentionally installed only when ``SB_HIDDEN_CAPTURE_DIR`` is
set.  Capture requests carry their corpus metadata in a ``sbcap`` request id;
ordinary serving requests are never written.  Laguna exposes post-layer
residual states through ``EagleModelMixin``.  vLLM packs each request's query
rows contiguously, so this module splits every prefill chunk using the runner's
persistent input-batch order and immediately copies each selected layer to CPU.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
from pathlib import Path
from typing import Any, Sequence

import torch

CAPTURE_ENV = "SB_HIDDEN_CAPTURE_DIR"
FORMAT = "shooting_brake_dflash_v1"
# poolside/Laguna-S-2.1-DFlash dflash_config.target_layer_ids.  DFlash layer
# ids index HF hidden_states (embedding is entry zero); vLLM's Laguna mixin
# captures post-layer boundaries, hence the +1 boundary tuple below.
TARGET_LAYER_IDS = (1, 10, 19, 29, 38, 47)
VLLM_AUX_BOUNDARIES = tuple(layer + 1 for layer in TARGET_LAYER_IDS)
_REQUEST_RE = re.compile(
    r"(?:^|-)sbcap\.([A-Za-z0-9_-]+)\.(\d+)\.(\d+)(?:-|$)"
)


def _encode_id(corpus_id: str) -> str:
    return base64.urlsafe_b64encode(corpus_id.encode("utf-8")).decode("ascii").rstrip("=")


def _decode_id(encoded: str) -> str:
    padding = "=" * (-len(encoded) % 4)
    return base64.urlsafe_b64decode(encoded + padding).decode("utf-8")


def make_capture_request_id(corpus_id: str, response_start: int, total_tokens: int) -> str:
    """Build the request id understood by the EngineCore-side hook."""
    if response_start < 0 or total_tokens < 1 or response_start > total_tokens:
        raise ValueError("invalid response/token bounds")
    return f"sbcap.{_encode_id(str(corpus_id))}.{response_start}.{total_tokens}"


def parse_capture_request_id(request_id: str) -> tuple[str, int, int] | None:
    """Recover ``(corpus_id, response_start, total_tokens)`` from vLLM ids."""
    match = _REQUEST_RE.search(request_id)
    if match is None:
        return None
    try:
        corpus_id = _decode_id(match.group(1))
        response_start = int(match.group(2))
        total_tokens = int(match.group(3))
    except (ValueError, UnicodeDecodeError):
        return None
    if response_start < 0 or total_tokens < 1 or response_start > total_tokens:
        return None
    return corpus_id, response_start, total_tokens


def capture_filename(corpus_id: str) -> str:
    """Return a readable, collision-resistant filename for a corpus id."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(corpus_id)).strip("._")
    slug = (slug or "sample")[:80]
    digest = hashlib.sha256(str(corpus_id).encode("utf-8")).hexdigest()[:16]
    return f"{slug}--{digest}.pt"


class CaptureAccumulator:
    """Accumulate CPU chunks and atomically persist one DFlash record.

    GPU tensors are accepted only at the ``add_gpu_chunk`` boundary.  Each
    layer slice is synchronously copied to fp16 CPU before a combined chunk is
    constructed, so this object never retains a GPU tensor between calls.
    """

    def __init__(self, output_dir: str | os.PathLike[str]) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._active: dict[str, dict[str, Any]] = {}

    def path_for(self, corpus_id: str) -> Path:
        return self.output_dir / capture_filename(corpus_id)

    def add_cpu_chunk(
        self,
        request_id: str,
        token_ids: torch.Tensor,
        states: torch.Tensor,
    ) -> Path | None:
        """Append CPU ``[T]`` tokens and ``[T, K, H]`` fp16 states."""
        metadata = parse_capture_request_id(request_id)
        if metadata is None:
            return None
        corpus_id, response_start, total_tokens = metadata
        output = self.path_for(corpus_id)
        if output.exists():
            self._active.pop(request_id, None)
            return output
        if token_ids.device.type != "cpu" or states.device.type != "cpu":
            raise ValueError("add_cpu_chunk accepts CPU tensors only")
        if token_ids.ndim != 1 or states.ndim != 3:
            raise ValueError("expected token_ids [T] and states [T, K, H]")
        if token_ids.shape[0] != states.shape[0]:
            raise ValueError("token/state chunk lengths differ")
        if states.shape[1] != len(TARGET_LAYER_IDS):
            raise ValueError(
                f"expected {len(TARGET_LAYER_IDS)} layer slices, got {states.shape[1]}"
            )

        entry = self._active.get(request_id)
        if entry is None:
            entry = {
                "corpus_id": corpus_id,
                "response_start": response_start,
                "total_tokens": total_tokens,
                "tokens": [],
                "states": [],
                "seen": 0,
            }
            self._active[request_id] = entry
        elif (
            entry["corpus_id"] != corpus_id
            or entry["response_start"] != response_start
            or entry["total_tokens"] != total_tokens
        ):
            raise RuntimeError(f"capture metadata changed for {request_id}")

        remaining = total_tokens - int(entry["seen"])
        if remaining <= 0:
            return output if output.exists() else None
        take = min(remaining, int(token_ids.shape[0]))
        if take:
            entry["tokens"].append(token_ids[:take].to(dtype=torch.int32).clone())
            entry["states"].append(
                states[:take].to(dtype=torch.bfloat16).contiguous().clone()
            )
            entry["seen"] += take
        if entry["seen"] < total_tokens:
            return None

        tokens = torch.cat(entry["tokens"], dim=0)
        by_layer = torch.cat(entry["states"], dim=0)
        if tokens.shape[0] != total_tokens or by_layer.shape[0] != total_tokens:
            raise RuntimeError(f"incomplete capture for {corpus_id}")
        loss_mask = torch.arange(total_tokens, dtype=torch.int64) >= response_start
        # SpecForge's DFlash offline reader consumes [T, K*H].  Keep the
        # explicit [T,K,H] view too; torch.save preserves the shared storage.
        flat = by_layer.view(total_tokens, -1)
        record = {
            "format": FORMAT,
            "id": corpus_id,
            "input_ids": tokens,
            "loss_mask": loss_mask,
            "hidden_states": flat,
            "hidden_states_by_layer": by_layer,
            "target_layer_ids": torch.tensor(TARGET_LAYER_IDS, dtype=torch.int64),
            "response_start": response_start,
        }
        temporary = output.with_suffix(output.suffix + ".part")
        try:
            torch.save(record, temporary)
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
            self._active.pop(request_id, None)
        return output

    def add_gpu_chunk(
        self,
        request_id: str,
        token_ids: torch.Tensor,
        layer_states: Sequence[torch.Tensor],
    ) -> Path | None:
        """Copy one packed request slice off GPU, then append it on CPU."""
        if len(layer_states) != len(TARGET_LAYER_IDS):
            raise ValueError(
                f"expected {len(TARGET_LAYER_IDS)} aux states, got {len(layer_states)}"
            )
        cpu_tokens = token_ids.detach().to(device="cpu", dtype=torch.int32).clone()
        # Copy one layer at a time: do not materialize [T,K,H] on the GPU.
        cpu_layers = [
            state.detach().to(device="cpu", dtype=torch.bfloat16).clone()
            for state in layer_states
        ]
        cpu_states = torch.stack(cpu_layers, dim=1).contiguous()
        del cpu_layers
        # Attention-sink positions (BOS, deepest boundary) carry massive
        # activations that saturate to non-finite in reduced precision --
        # observed as exactly one NaN row at position 0/layer 48 (2026-08-25).
        # Those features carry no draft-training signal; zero them rather
        # than poisoning the whole record, and account for it.
        bad = ~torch.isfinite(cpu_states.float())
        if bad.any():
            rows = int(bad.any(dim=(1, 2)).sum())
            cpu_states[bad] = 0
            print(
                f"[sb-hidden-capture] zeroed {rows} non-finite row(s) "
                f"for {request_id}",
                flush=True,
            )
        return self.add_cpu_chunk(request_id, cpu_tokens, cpu_states)


_DEBUG_ONCE = [0]


def _capture_runner_state(runner: Any, accumulator: CaptureAccumulator) -> None:
    state = runner.execute_model_state
    if _DEBUG_ONCE[0] < 3:
        _DEBUG_ONCE[0] += 1
        aux = "unset"
        reqs: list[str] = []
        if state is not None:
            aux = "yes" if state.aux_hidden_states is not None else "None"
            reqs = list(runner.input_batch.req_ids)[:3]
        print(
            f"[sb-hidden-capture] exec#{_DEBUG_ONCE[0]} "
            f"state={'set' if state is not None else 'None'} aux={aux} "
            f"reqs={reqs}",
            flush=True,
        )
    if state is None or state.aux_hidden_states is None:
        return
    scheduler_output = state.scheduler_output
    request_ids = list(runner.input_batch.req_ids)
    offset = 0
    for request_id in request_ids:
        count = int(scheduler_output.num_scheduled_tokens.get(request_id, 0))
        if count <= 0:
            continue
        end = offset + count
        if parse_capture_request_id(request_id) is not None:
            print(
                f"[sb-hidden-capture] chunk req={request_id} rows={count}",
                flush=True,
            )
            token_slice = runner.input_ids.cpu[offset:end]
            layer_slices = [hidden[offset:end] for hidden in state.aux_hidden_states]
            written = accumulator.add_gpu_chunk(request_id, token_slice, layer_slices)
            if written is not None:
                print(
                    f"[sb-hidden-capture] wrote {written} "
                    f"layers={TARGET_LAYER_IDS}",
                    flush=True,
                )
        offset = end


def install(output_dir: str | os.PathLike[str] | None = None) -> None:
    """Patch vLLM's GPUModelRunner for Laguna auxiliary-state capture."""
    configured = str(output_dir or os.environ.get(CAPTURE_ENV, "")).strip()
    if not configured:
        return

    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if getattr(GPUModelRunner.execute_model, "_sb_hidden_capture", False):
        return

    accumulator = CaptureAccumulator(configured)
    original_init = GPUModelRunner.__init__
    original_layers = GPUModelRunner._get_eagle3_aux_layers_from_config
    original_execute = GPUModelRunner.execute_model

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        # Must be true before load_model() configures Laguna and before CUDA
        # graph capture decides whether the model returns auxiliary tensors.
        self.use_aux_hidden_state_outputs = True

    def patched_layers(self: Any) -> tuple[int, ...]:
        configured_layers = original_layers(self)
        if configured_layers and tuple(configured_layers) != VLLM_AUX_BOUNDARIES:
            raise RuntimeError(
                "SB hidden capture requires Laguna DFlash auxiliary boundaries "
                f"{VLLM_AUX_BOUNDARIES}, got {tuple(configured_layers)}"
            )
        return VLLM_AUX_BOUNDARIES

    def patched_execute(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_execute(self, *args, **kwargs)
        _capture_runner_state(self, accumulator)
        return result

    patched_init._sb_hidden_capture = True  # type: ignore[attr-defined]
    patched_layers._sb_hidden_capture = True  # type: ignore[attr-defined]
    patched_execute._sb_hidden_capture = True  # type: ignore[attr-defined]
    GPUModelRunner.__init__ = patched_init
    GPUModelRunner._get_eagle3_aux_layers_from_config = patched_layers
    GPUModelRunner.execute_model = patched_execute
    print(
        f"[sb-hidden-capture] installed dir={accumulator.output_dir} "
        f"target_layers={TARGET_LAYER_IDS} "
        f"vllm_boundaries={VLLM_AUX_BOUNDARIES}",
        flush=True,
    )


__all__ = [
    "CAPTURE_ENV",
    "FORMAT",
    "TARGET_LAYER_IDS",
    "VLLM_AUX_BOUNDARIES",
    "CaptureAccumulator",
    "capture_filename",
    "install",
    "make_capture_request_id",
    "parse_capture_request_id",
]
