"""Run a real-model prefill/decode probe with numerical and provider health checks.

Environment controls keep candidate libraries and model revisions explicit.
Outputs include prompt logprobs, generated tokens, and worker snapshots so a
successful completion cannot hide an inactive or failing remote provider.
This is a correctness probe, not a serving performance benchmark.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

#: Long enough that prefill exceeds the default 128-token dispatch threshold,
#: and ordinary enough that a degraded expert path shows up as visibly broken
#: text rather than a plausible alternate phrasing.
PROMPT = (
    "You are reading a technical description of a computer system. "
    "A mixture-of-experts language model routes each token to a small "
    "number of expert networks chosen from a much larger pool. Most "
    "experts are idle for any given token, so the memory holding them is "
    "mostly cold. A system can exploit that by keeping frequently used "
    "experts in fast memory close to the processor and moving rarely used "
    "experts to slower, cheaper memory further away. The cost of this "
    "trade is paid only when a cold expert is actually selected, and the "
    "benefit is that a much larger model fits in the same budget. "
    "Considering only the information above, explain in one sentence why "
    "moving rarely used experts to slower memory is usually a good trade."
)

MAX_TOKENS = 32


def _install_logit_diagnostics(worker) -> None:
    """Inspect real prefill tensors without changing their values or results."""
    import torch

    model = worker.model_runner.model
    compute_logits = model.compute_logits

    def traced_logits(hidden_states, *args, **kwargs):
        logits = compute_logits(hidden_states, *args, **kwargs)
        if hidden_states.shape[0] > 1:
            tensors = {
                "hidden_states": hidden_states,
                "logits": logits,
                "logprobs": logits.log_softmax(dim=-1, dtype=torch.float32),
            }
            report = {
                name: {
                    "shape": list(tensor.shape),
                    "nonfinite_rows": (~torch.isfinite(tensor).all(dim=-1))
                    .nonzero()
                    .flatten()
                    .tolist(),
                }
                for name, tensor in tensors.items()
            }
            print("PROBE_LOGIT_DIAGNOSTICS " + json.dumps(report), flush=True)
        return logits

    model.compute_logits = traced_logits


def _install_layer_diagnostics(worker) -> None:
    """Stop an eager diagnostic run at the first nonfinite module output."""
    import torch

    def check_output(module, inputs, output, *, name):
        values = output if isinstance(output, (tuple, list)) else (output,)
        for index, value in enumerate(values):
            if not isinstance(value, torch.Tensor) or not value.dtype.is_floating_point:
                continue
            if value.numel() and not torch.isfinite(value.float()).all().item():
                raise RuntimeError(
                    f"PROBE_FIRST_NONFINITE module={name} type={type(module).__name__} "
                    f"output={index} shape={tuple(value.shape)} dtype={value.dtype}"
                )

    from functools import partial

    for name, module in worker.model_runner.model.named_modules():
        module.register_forward_hook(partial(check_output, name=name))


def _enable_aggregation_capture(worker, path: str, layer: int) -> None:
    """Use the existing aggregation oracle on the request, not engine warmup."""
    os.environ["SHOOTING_BRAKE_AGGREGATION_CAPTURE"] = path
    os.environ["SHOOTING_BRAKE_AGGREGATION_LAYER"] = str(layer)


def main() -> int:
    from vllm.plugins import load_general_plugins

    load_general_plugins()
    from vllm import LLM, SamplingParams

    revision = os.environ.get("SB_REVISION")
    engine_options = {
        "model": os.environ["SHOOTING_BRAKE_MODEL"],
        "gpu_memory_utilization": float(os.environ.get("SB_GPU_UTIL", "0.90")),
        "max_model_len": int(os.environ.get("SB_MML", "8192")),
        "max_num_seqs": int(os.environ.get("SB_MAX_SEQS", "64")),
        "max_num_batched_tokens": int(os.environ.get("SB_MNBT", "2048")),
        "enforce_eager": os.environ.get("SB_EAGER") == "1",
        "moe_backend": os.environ.get("SB_MOE_BACKEND", "cutlass"),
        "seed": 0,
    }
    if revision:
        engine_options.update(revision=revision, tokenizer_revision=revision)
    library_path = os.environ.get("SHOOTING_BRAKE_B70_LIB")
    library_sha256 = None
    if library_path:
        with Path(library_path).open("rb") as handle:
            library_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    llm = LLM(**engine_options)
    from shooting_brake_vllm.telemetry import collect_worker_stats

    workers_before = llm.collective_rpc(collect_worker_stats)
    if os.environ.get("SB_LOGIT_DIAGNOSTICS") == "1":
        llm.collective_rpc(_install_logit_diagnostics)
    if os.environ.get("SB_LAYER_DIAGNOSTICS") == "1":
        if not engine_options["enforce_eager"]:
            raise ValueError("layer diagnostics require SB_EAGER=1")
        llm.collective_rpc(_install_layer_diagnostics)
    if capture_path := os.environ.get("SB_AGGREGATION_CAPTURE"):
        llm.collective_rpc(
            _enable_aggregation_capture,
            args=(capture_path, int(os.environ["SB_AGGREGATION_LAYER"])),
        )

    tok = llm.get_tokenizer()
    n_prompt = len(tok.encode(PROMPT))

    # prompt_logprobs is the sensitive measurement here. Token ids only
    # reveal a broken prefill when the damage flips an argmax, and the first
    # generated token of a prompt like this one is predictable enough to
    # survive substantial damage. Prompt logprobs are computed *during*
    # prefill, one per prompt position, so they read that pass directly: a
    # path that drops routed-expert output shows up as a clearly worse
    # cumulative logprob even when every sampled token matches.
    out = llm.generate(
        [PROMPT],
        SamplingParams(
            temperature=0.0,
            max_tokens=int(os.environ.get("SB_MAX_TOKENS", str(MAX_TOKENS))),
            logprobs=5,
            prompt_logprobs=0,
        ),
    )[0]
    gen = out.outputs[0]
    workers_after = llm.collective_rpc(collect_worker_stats)

    # Sum the prompt logprobs, skipping position 0 (no prediction exists).
    plp = [next(iter(d.values())).logprob for d in (out.prompt_logprobs or [])[1:] if d]
    nonfinite_prompt_logprobs = [
        {"position": position, "token_id": token_id, "logprob": str(value.logprob)}
        for position, values in enumerate(out.prompt_logprobs or [])
        if position > 0 and values
        for token_id, value in values.items()
        if not math.isfinite(value.logprob)
    ]
    errors = []
    prompt_finite = bool(plp) and all(math.isfinite(value) for value in plp)
    generated_finite = bool(gen.logprobs) and all(
        values and all(math.isfinite(value.logprob) for value in values.values())
        for values in gen.logprobs
    )
    if not gen.token_ids or not gen.text.strip():
        errors.append("model returned no generated tokens or decoded text")
    if not out.prompt_logprobs:
        errors.append("prompt logprobs were not returned")
    elif len(plp) != len(out.prompt_token_ids) - 1:
        errors.append(
            f"incomplete prompt logprobs: got {len(plp)}, expected {len(out.prompt_token_ids) - 1}"
        )
    if nonfinite_prompt_logprobs:
        errors.append(f"nonfinite prompt logprobs at {len(nonfinite_prompt_logprobs)} positions")
    if not generated_finite:
        errors.append("generated logprobs are missing or nonfinite")
    require_b70 = (
        os.environ.get("SHOOTING_BRAKE_HYBRID") == "1"
        and os.environ.get("SHOOTING_BRAKE_B70_DEVICE") == "1"
    )
    if require_b70:
        selectors = os.environ.get("SHOOTING_BRAKE_B70_SELECTORS", "")
        expected_devices = len([part for part in selectors.split(",") if part.strip()]) or 1
        if not workers_after or len(workers_before) != len(workers_after):
            errors.append("worker telemetry is missing or changed during inference")
        for index, worker in enumerate(workers_after):
            before = workers_before[index] if index < len(workers_before) else {}
            providers = worker.get("synchronous_provider", {})
            if not providers or providers.get("available") is False:
                errors.append(f"worker {index}: no native provider health")
                continue
            if len(providers) != expected_devices:
                errors.append(f"worker {index}: expected {expected_devices} native devices")
            for device, health in providers.items():
                previous = before.get("synchronous_provider", {}).get(device, {})
                if not health.get("available") or health.get("last_error"):
                    errors.append(f"worker {index}, device {device}: unhealthy provider")
                if health.get("dispatches_raw", 0) <= previous.get("dispatches_raw", 0):
                    errors.append(f"worker {index}, device {device}: no inference dispatches")
                if previous and health.get("generation_raw") != previous.get("generation_raw"):
                    errors.append(f"worker {index}, device {device}: provider generation changed")
            if os.environ.get("SHOOTING_BRAKE_B70_GRAPH") == "1":
                poller = worker.get("poller", {})
                devices = poller.get("per_device", {})
                if not poller.get("available") or set(devices) != set(providers):
                    errors.append(f"worker {index}: missing graph poller devices")
                for device, counters in devices.items():
                    previous = before.get("poller", {}).get("per_device", {}).get(device, {})
                    if counters.get("errors", 0):
                        errors.append(f"worker {index}, device {device}: native poller errors")
                    if counters.get("dispatches", 0) <= previous.get("dispatches", 0):
                        errors.append(f"worker {index}, device {device}: graph poller did no work")

    first = gen.logprobs[0] if gen.logprobs else {}
    result = {
        "label": os.environ.get("SB_LABEL", "?"),
        "engine_options": engine_options,
        "native_library": library_path,
        "native_library_sha256": library_sha256,
        "workers_before": workers_before,
        "workers_after": workers_after,
        "passed": not errors,
        "errors": errors,
        "generated_logprobs_finite": generated_finite,
        "placement": os.environ.get("SHOOTING_BRAKE_PLACEMENT", "all-cuda"),
        "max_batch": os.environ.get("SHOOTING_BRAKE_B70_MAX_BATCH", "128"),
        "prompt_tokens": n_prompt,
        "prompt_logprobs_count": len(plp),
        "prompt_logprobs_expected": len(out.prompt_token_ids) - 1,
        "nonfinite_prompt_logprobs": nonfinite_prompt_logprobs,
        "prompt_logprob_sum": sum(plp) if prompt_finite else None,
        "prompt_logprob_mean": sum(plp) / len(plp) if prompt_finite else None,
        "first_token_top5": {
            int(k): round(v.logprob, 5) if math.isfinite(v.logprob) else None
            for k, v in first.items()
        },
        "token_ids": list(gen.token_ids),
        "text": gen.text,
    }

    dest = os.environ.get("SB_OUT")
    if dest:
        with open(dest, "w") as fh:
            json.dump(result, fh, indent=2)

    print("PROBE_RESULT " + json.dumps(result))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
