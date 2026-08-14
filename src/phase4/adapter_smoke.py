#!/usr/bin/env python3
"""Run one real eager all-CUDA request through the selected Phase-4 adapter."""

from __future__ import annotations

import argparse
import os

from shooting_brake_vllm.config import QUALIFIED_MODEL, phase4_enabled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=QUALIFIED_MODEL)
    parser.add_argument("--prompt", default="State the integer after 41.")
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = parser.parse_args()

    if not phase4_enabled():
        raise RuntimeError(
            "set SHOOTING_BRAKE_PHASE4=all-cuda and "
            f"SHOOTING_BRAKE_MODEL={QUALIFIED_MODEL}"
        )
    if args.model != QUALIFIED_MODEL:
        raise RuntimeError(f"only {QUALIFIED_MODEL} is qualified")

    from vllm import LLM, SamplingParams
    from vllm.model_executor.custom_op import op_registry_oot
    from vllm.plugins import load_general_plugins

    load_general_plugins()
    if op_registry_oot.get("MoERunner").__name__ != "HybridMoERunner":
        raise RuntimeError("HybridMoERunner was not selected")
    if op_registry_oot.get("RoutedExperts").__name__ != "HybridRoutedExperts":
        raise RuntimeError("HybridRoutedExperts was not selected")

    llm = LLM(
        model=args.model,
        enforce_eager=True,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=8192,
    )
    output = llm.generate(
        [args.prompt],
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
        use_tqdm=False,
    )[0].outputs[0]
    if not output.token_ids:
        raise RuntimeError("adapter returned no generated tokens")
    print("Phase-4 all-CUDA adapter smoke PASS")
    print(f"token_ids={output.token_ids}")
    print(f"text={output.text!r}")


if __name__ == "__main__":
    main()
