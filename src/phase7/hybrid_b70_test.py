#!/usr/bin/env python3
"""Phase-7d integration gate: real model + real B70 device.

Runs one real eager request two ways:
  1. ``all-cuda`` placement (baseline, no hybrid)
  2. ``split:128`` + ``HYBRID=1`` + ``B70_DEVICE=1`` (real hybrid: B70-owned
     routes computed on the actual Intel Arc Pro B70 via QuixiCore NVFP4)

Gate: identical or near-identical token output. The B70 NVFP4 kernel differs
from the CUDA FlashInfer-CUTLASS kernel at the quantization level (Phase 3
tolerance: cosine > 0.98), so greedy decode may occasionally diverge — but for
a clear prompt like "State the integer after 41." the expected answer "42"
should be stable.

Requires: oneAPI environment sourced (for SYCL runtime libraries).
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
    b70_device: bool,
    output_path: Path,
    hybrid_marker: Path | None = None,
) -> None:
    env = dict(os.environ)
    env.update(
        VLLM_PLUGINS="shooting_brake_vllm",
        SHOOTING_BRAKE_PHASE4="all-cuda",
        SHOOTING_BRAKE_MODEL=MODEL,
        SHOOTING_BRAKE_PLACEMENT=placement,
    )
    if hybrid:
        env["SHOOTING_BRAKE_HYBRID"] = "1"
        if b70_device:
            env["SHOOTING_BRAKE_B70_DEVICE"] = "1"
        if hybrid_marker:
            env["SHOOTING_BRAKE_HYBRID_MARKER"] = str(hybrid_marker)
    else:
        env.pop("SHOOTING_BRAKE_HYBRID", None)
        env.pop("SHOOTING_BRAKE_B70_DEVICE", None)

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
        sys.stderr.write(completed.stderr[-5000:])
        raise RuntimeError(f"run failed (placement={placement} hybrid={hybrid} b70={b70_device})")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="sb-phase7d-") as tmp:
        base_path = Path(tmp) / "all_cuda.json"
        b70_path = Path(tmp) / "b70_hybrid.json"
        hybrid_marker = Path(tmp) / "hybrid_marker.json"

        print("running all-cuda baseline...")
        run_case("all-cuda", hybrid=False, b70_device=False, output_path=base_path)
        print("running split:128 + HYBRID=1 + B70_DEVICE=1 (real B70)...")
        run_case("split:128", hybrid=True, b70_device=True,
                 output_path=b70_path, hybrid_marker=hybrid_marker)

        base = json.loads(base_path.read_text())
        b70 = json.loads(b70_path.read_text())

        # Confirm the hybrid path was exercised.
        if not hybrid_marker.exists():
            raise RuntimeError("Phase-7d: hybrid marker never written")
        hmarker = json.loads(hybrid_marker.read_text())
        print(f"hybrid marker: {hmarker}")

        print(f"  all-cuda tokens: {base['token_ids']}")
        print(f"  b70-hybrid tokens: {b70['token_ids']}")
        print(f"  all-cuda text: {base['text']!r}")
        print(f"  b70-hybrid text: {b70['text']!r}")

        if base["token_ids"] == b70["token_ids"]:
            print(f"\nPhase-7d real-B70 hybrid gate PASS (exact token parity)")
        else:
            # Near-parity: check semantic match (both say "42" or equivalent)
            base_text = base["text"].strip().lower()
            b70_text = b70["text"].strip().lower()
            if "42" in base_text and "42" in b70_text:
                print(f"\nPhase-7d real-B70 hybrid gate PASS (semantic parity, "
                      f"minor token-level divergence expected from NVFP4 kernel-cross)")
            else:
                raise RuntimeError(
                    f"Phase-7d: significant token divergence.\n"
                    f"  all-cuda: {base['text']!r}\n"
                    f"  b70-hybrid: {b70['text']!r}"
                )

        print(f"  hybrid exercised: layer {hmarker['layer']}, "
              f"{hmarker['b70_routes']} B70 routes")


if __name__ == "__main__":
    main()
