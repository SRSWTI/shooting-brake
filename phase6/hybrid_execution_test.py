#!/usr/bin/env python3
"""Phase-6c real-hybrid-execution gate.

Runs one real eager request two ways:
  1. ``all-cuda`` placement (baseline, no hybrid)
  2. ``split:128`` placement + ``SHOOTING_BRAKE_HYBRID=1`` (real hybrid:
     B70-route CUDA weights zeroed, B70 partial computed separately and added)

Gate: identical token output (the split-merge is exact to BF16-ULP, proven in
6b, so greedy decode must select the same tokens). A hybrid marker confirms
the split-merge path was actually exercised.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "unsloth/Qwen3.6-35B-A3B-NVFP4"
PROMPT = "State the integer after 41."


def run_case(
    placement: str,
    hybrid: bool,
    output_path: Path,
    hybrid_marker: Path | None = None,
) -> None:
    env = dict(os.environ)
    env.update(
        # oneAPI's setvars.sh drops the venv from PATH, taking `ninja` with
        # it and making vLLM's compile step die with FileNotFoundError.
        PATH=f"{ROOT / '.venv' / 'bin'}:{env.get('PATH', '')}",
        VLLM_PLUGINS="shooting_brake_vllm",
        SHOOTING_BRAKE_PHASE4="all-cuda",
        SHOOTING_BRAKE_MODEL=MODEL,
        SHOOTING_BRAKE_PLACEMENT=placement,
    )
    if hybrid:
        env["SHOOTING_BRAKE_HYBRID"] = "1"
        if hybrid_marker:
            env["SHOOTING_BRAKE_HYBRID_MARKER"] = str(hybrid_marker)
    else:
        env.pop("SHOOTING_BRAKE_HYBRID", None)

    code = f"""
import json
from vllm.plugins import load_general_plugins
load_general_plugins()
from vllm import LLM, SamplingParams
llm = LLM(model={MODEL!r}, enforce_eager=True, tensor_parallel_size=1,
          pipeline_parallel_size=1, gpu_memory_utilization=0.90, max_model_len=8192)
out = llm.generate([{PROMPT!r}], SamplingParams(temperature=0.0, max_tokens=8),
                   use_tqdm=False)[0].outputs[0]
with open({str(output_path)!r}, "w") as f:
    json.dump({{"token_ids": out.token_ids, "text": out.text}}, f)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT, env=env, text=True, check=False, capture_output=True,
    )
    if completed.returncode:
        sys.stderr.write(completed.stderr[-4000:])
        raise RuntimeError(f"run failed (placement={placement} hybrid={hybrid})")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="sb-phase6c-") as tmp:
        base_path = Path(tmp) / "all_cuda.json"
        hybrid_path = Path(tmp) / "hybrid.json"
        hybrid_marker = Path(tmp) / "hybrid_marker.json"

        print("running all-cuda baseline...")
        run_case("all-cuda", hybrid=False, output_path=base_path)
        print("running split:128 + HYBRID=1 (real hybrid execution)...")
        run_case("split:128", hybrid=True, output_path=hybrid_path,
                 hybrid_marker=hybrid_marker)

        base = json.loads(base_path.read_text())
        hybrid = json.loads(hybrid_path.read_text())

        # Confirm the hybrid path was exercised.
        if not hybrid_marker.exists():
            raise RuntimeError(
                "Phase-6c: hybrid marker never written — "
                "the split-merge path was not exercised"
            )
        hmarker = json.loads(hybrid_marker.read_text())
        print(f"hybrid marker: {hmarker}")

        # Token output must match (split-merge is exact to BF16-ULP).
        if base["token_ids"] != hybrid["token_ids"]:
            print(f"  all-cuda: {base['token_ids']}")
            print(f"  hybrid:   {hybrid['token_ids']}")
            print(f"  all-cuda text: {base['text']!r}")
            print(f"  hybrid text:   {hybrid['text']!r}")
            raise RuntimeError("Phase-6c: token mismatch between all-cuda and hybrid")

        print(f"\nPhase-6c real-hybrid-execution gate PASS")
        print(f"  tokens: {base['token_ids']}")
        print(f"  text:   {base['text']!r}")
        print(f"  hybrid exercised: layer {hmarker['layer']}, "
              f"{hmarker['b70_routes']} B70 routes")


if __name__ == "__main__":
    main()
