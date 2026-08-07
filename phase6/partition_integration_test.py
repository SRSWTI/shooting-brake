#!/usr/bin/env python3
"""Phase-6a integration gate: runtime partition validated, output unchanged.

Runs one real eager request twice through the Shooting Brake adapter:
  1. ``all-cuda`` placement  (baseline)
  2. ``split:128`` placement (partition computed + validated every step)

Both must produce identical token output, proving Phase 6a partitions routes
at runtime without changing execution. The ``split:128`` run also exercises
the partition invariants (disjoint, covering, B70-only-in-capable) on every
layer/step; if any invariant fails the engine crashes. A marker file written
on the first remote step confirms B70 routes actually appeared.
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
    placement: str, output_path: Path, marker_path: Path | None = None
) -> None:
    env = dict(os.environ)
    # oneAPI's setvars.sh drops the venv from PATH, which strips `ninja` and
    # makes vLLM's compile step die with FileNotFoundError. Re-prepend it so
    # this gate runs the same whether or not the caller sourced setvars.
    env["PATH"] = f"{ROOT / '.venv' / 'bin'}:{env.get('PATH', '')}"
    env["VLLM_PLUGINS"] = "shooting_brake_vllm"
    env["SHOOTING_BRAKE_PHASE4"] = "all-cuda"
    env["SHOOTING_BRAKE_MODEL"] = MODEL
    env["SHOOTING_BRAKE_PLACEMENT"] = placement
    if marker_path:
        env["SHOOTING_BRAKE_PARTITION_MARKER"] = str(marker_path)
        # The partition only runs when something defeats the all-CUDA
        # pass-through that Tier 3 introduced; without this the adapter
        # returns straight from super().forward_modular() and the gate
        # asserts on a code path that never executed. No B70_DEVICE, so the
        # remote partial is computed by the CUDA kernel with masked weights
        # — which is what this gate wants: partition math, not hardware.
        env["SHOOTING_BRAKE_HYBRID"] = "1"

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
        sys.stderr.write(completed.stderr[-3000:])
        raise RuntimeError(f"{placement} run failed")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="sb-phase6a-") as tmp:
        base_path = Path(tmp) / "all_cuda.json"
        split_path = Path(tmp) / "split128.json"
        marker_path = Path(tmp) / "remote_marker.json"

        print("running all-cuda baseline...")
        run_case("all-cuda", base_path)
        print("running split:128 (partition validated every step)...")
        run_case("split:128", split_path, marker_path=marker_path)

        base = json.loads(base_path.read_text())
        split = json.loads(split_path.read_text())

        if base["token_ids"] != split["token_ids"]:
            raise RuntimeError(
                "Phase-6a output mismatch (execution must be unchanged):\n"
                f"  all-cuda:  {base['token_ids']}\n"
                f"  split:128: {split['token_ids']}"
            )
        print(f"token parity PASS: {base['token_ids']}")
        print(f"text: {base['text']!r}")

        # confirm the split run actually saw B70 routes
        if not marker_path.exists():
            raise RuntimeError(
                "Phase-6a: split:128 run never wrote the marker — "
                "forward_modular may not be the active code path"
            )
        markers = [
            json.loads(line) for line in marker_path.read_text().splitlines()
        ]
        remote_markers = [m for m in markers if m.get("b70_routes", 0) > 0]
        if not remote_markers:
            raise RuntimeError(
                f"Phase-6a: no layer saw B70 routes. "
                f"markers: {markers[:3]}..."
            )
        print(f"remote marker confirmed: {remote_markers[0]}")
        print(f"  ({len(remote_markers)} layers with remote routes)")

    print("\nPhase-6a integration gate PASS")


if __name__ == "__main__":
    main()
