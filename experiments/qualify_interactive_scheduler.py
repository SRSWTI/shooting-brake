"""Run real mixed-length Jota inference with strict probability/health checks.

Uses the existing provider environment and campaign worker instrumentation.
This is correctness qualification, not a serving performance measurement.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scheduler-cls", default="experiments.interactive_scheduler.InteractiveScheduler"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    from shooting_brake_vllm.telemetry import collect_worker_stats
    from vllm import LLM, SamplingParams

    from src.phase7.prefill_probe import PROMPT

    model = os.environ["SHOOTING_BRAKE_MODEL"]
    revision = os.environ["SB_REVISION"]
    llm = LLM(
        model=model,
        revision=revision,
        tokenizer_revision=revision,
        trust_remote_code=True,
        max_model_len=16384,
        max_num_batched_tokens=2048,
        max_num_seqs=2,
        gpu_memory_utilization=0.85,
        moe_backend="cutlass",
        enable_prefix_caching=False,
        scheduler_cls=args.scheduler_cls,
        worker_extension_cls="experiments.campaign_worker.CampaignWorkerExtension",
        seed=0,
    )
    before = llm.collective_rpc(collect_worker_stats)
    llm.collective_rpc(
        "campaign_start",
        kwargs={
            "label": "qualification",
            "output_root": str(args.output.resolve()),
            "tracing": "0",
        },
    )
    outputs = llm.generate(
        [PROMPT, PROMPT * 80],
        SamplingParams(
            temperature=0, max_tokens=32, ignore_eos=True, logprobs=5, prompt_logprobs=0
        ),
    )
    trace = llm.collective_rpc("campaign_stop")
    after = llm.collective_rpc(collect_worker_stats)
    errors = []
    records = []
    for index, output in enumerate(outputs):
        prompt_values = [
            value.logprob
            for position in (output.prompt_logprobs or [])[1:]
            if position
            for value in position.values()
        ]
        generated = output.outputs[0]
        generated_values = [
            value.logprob for position in (generated.logprobs or []) for value in position.values()
        ]
        if len(prompt_values) != len(output.prompt_token_ids) - 1 or not all(
            math.isfinite(value) for value in prompt_values
        ):
            errors.append(f"request {index}: missing or nonfinite prompt logprobs")
        if (
            len(generated.token_ids) != 32
            or len(generated.logprobs or []) != 32
            or not generated_values
            or not all(math.isfinite(value) for value in generated_values)
        ):
            errors.append(f"request {index}: incomplete generation or invalid generated logprobs")
        records.append(
            {
                "prompt_tokens": len(output.prompt_token_ids),
                "prompt_logprobs_count": len(prompt_values),
                "prompt_logprobs_sum": sum(prompt_values)
                if all(math.isfinite(value) for value in prompt_values)
                else None,
                "generated_tokens": generated.token_ids,
                "text": generated.text,
                "generated_logprobs_finite": all(
                    math.isfinite(value) for value in generated_values
                ),
            }
        )
    if len(outputs) != 2:
        errors.append("expected two completed requests")
    for result in trace:
        if not result["passed_trace_coverage"]:
            errors.extend(result["errors"])
        if set(result["devices"]) != {"0", "1"}:
            errors.append("expected both native devices")
    result = {
        "passed": not errors,
        "errors": errors,
        "scheduler_cls": args.scheduler_cls,
        "interactive_token_cap": os.environ.get("SB_INTERACTIVE_TOKEN_CAP", "512"),
        "model": model,
        "revision": revision,
        "native_library": os.environ["SHOOTING_BRAKE_B70_LIB"],
        "outputs": records,
        "before": before,
        "after": after,
        "trace": trace,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({"passed": not errors, "errors": errors, "outputs": records}, indent=2))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
