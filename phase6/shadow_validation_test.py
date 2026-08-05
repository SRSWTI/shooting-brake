#!/usr/bin/env python3
"""Phase-6b shadow-validation gate.

Runs one real eager request through the adapter with ``split:128`` placement
and ``SHOOTING_BRAKE_SHADOW=1``. On the first step that has B70 routes, the
adapter computes the routed-expert output three ways (all routes, CUDA-only,
B70-only) and validates the split-merge identity:

    Y_cuda_routed + Y_b70_routed ≈ Y_full_routed

Gate: max_abs < 0.1, cosine > 0.999, no NaN.  This confirms the mathematical
identity that Phase 6c will rely on (the B70 device kernel itself was already
validated independently in Phase 3).
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


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="sb-phase6b-") as tmp:
        shadow_marker = Path(tmp) / "shadow.json"
        output_path = Path(tmp) / "tokens.json"

        env = dict(os.environ)
        env.update(
            VLLM_PLUGINS="shooting_brake_vllm",
            SHOOTING_BRAKE_PHASE4="all-cuda",
            SHOOTING_BRAKE_MODEL=MODEL,
            SHOOTING_BRAKE_PLACEMENT="split:128",
            SHOOTING_BRAKE_SHADOW="1",
            SHOOTING_BRAKE_SHADOW_MARKER=str(shadow_marker),
        )

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
    json.dump({{"token_ids": out.token_ids}}, f)
"""
        print("running shadow validation (split:128)...")
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT, env=env, text=True, check=False, capture_output=True,
        )
        if completed.returncode:
            sys.stderr.write(completed.stderr[-4000:])
            raise RuntimeError("shadow run failed")

        tokens = json.loads(output_path.read_text())["token_ids"]
        print(f"tokens: {tokens}")

        if not shadow_marker.exists():
            raise RuntimeError(
                "Phase-6b: shadow marker was never written — "
                "no step had B70 routes or the hook didn't fire"
            )
        result = json.loads(shadow_marker.read_text())
        print(f"shadow result: {json.dumps(result, indent=2)}")

        if not result["gate_pass"]:
            raise RuntimeError(
                f"Phase-6b shadow gate FAILED: "
                f"max_abs={result['max_abs']:.6f} cosine={result['cosine']:.8f}"
            )

    print(f"\nPhase-6b shadow-validation gate PASS")
    print(f"  max_abs={result['max_abs']:.6f}  "
          f"mean_abs={result['mean_abs']:.6f}  "
          f"rel_rmse={result['rel_rmse']:.6f}  "
          f"cosine={result['cosine']:.8f}")


if __name__ == "__main__":
    main()
