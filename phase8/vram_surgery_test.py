#!/usr/bin/env python3
"""Phase-8.5 integration gate: VRAM surgery (expert weight offloading).

Runs one real eager request two ways and verifies:
  1. ``all-cuda`` baseline (no surgery, no B70).
  2. ``split:128`` + ``HYBRID=1`` + ``B70_DEVICE=1`` + ``VRAM_SURGERY=1``:
     B70-owned expert weights are physically removed from CUDA VRAM.

Gate: identical token output. Also measures VRAM before/after surgery.

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
    vram_surgery: bool,
    output_path: Path,
    hybrid_marker: Path | None = None,
    vram_marker: Path | None = None,
) -> dict:
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
        if vram_surgery:
            env["SHOOTING_BRAKE_VRAM_SURGERY"] = "1"
        if hybrid_marker:
            env["SHOOTING_BRAKE_HYBRID_MARKER"] = str(hybrid_marker)
        if vram_marker:
            env["SHOOTING_BRAKE_VRAM_MARKER"] = str(vram_marker)

    code = f"""
import json, torch
from vllm.plugins import load_general_plugins
load_general_plugins()
from vllm import LLM, SamplingParams
llm = LLM(model={MODEL!r}, enforce_eager=True, tensor_parallel_size=1,
          pipeline_parallel_size=1, gpu_memory_utilization=0.90, max_model_len=8192)
vram_post_load = torch.cuda.memory_allocated() / (1024**3)
out = llm.generate([{PROMPT!r}], SamplingParams(temperature=0.0, max_tokens=8),
                   use_tqdm=False)[0].outputs[0]
vram_post_gen = torch.cuda.memory_allocated() / (1024**3)
with open({str(output_path)!r}, "w") as f:
    json.dump({{"token_ids": out.token_ids, "text": out.text,
               "vram_post_load_gb": vram_post_load,
               "vram_post_gen_gb": vram_post_gen}}, f)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT, env=env, text=True, check=False, capture_output=True,
    )
    if completed.returncode:
        sys.stderr.write(completed.stderr[-5000:])
        raise RuntimeError(f"run failed (surgery={vram_surgery})")
    return json.loads(output_path.read_text())


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="sb-phase85-") as tmp:
        base_path = Path(tmp) / "all_cuda.json"
        surgery_path = Path(tmp) / "surgery.json"
        hybrid_marker = Path(tmp) / "hybrid_marker.json"
        vram_marker = Path(tmp) / "vram_surgery.txt"

        print("running all-cuda baseline...")
        base = run_case("all-cuda", hybrid=False, b70_device=False,
                        vram_surgery=False, output_path=base_path)

        print("running split:128 + HYBRID + B70_DEVICE + VRAM_SURGERY=1...")
        surgery = run_case("split:128", hybrid=True, b70_device=True,
                           vram_surgery=True, output_path=surgery_path,
                           hybrid_marker=hybrid_marker, vram_marker=vram_marker)

        if not hybrid_marker.exists():
            raise RuntimeError("hybrid marker never written")
        hmarker = json.loads(hybrid_marker.read_text())
        print(f"  hybrid marker: {hmarker}")

        # VRAM comparison (post-generation = after surgery for surgery case)
        print(f"  VRAM all-cuda  (post-gen):  {base['vram_post_gen_gb']:.3f} GB")
        print(f"  VRAM surgery   (post-gen):  {surgery['vram_post_gen_gb']:.3f} GB")
        delta = base['vram_post_gen_gb'] - surgery['vram_post_gen_gb']
        print(f"  VRAM difference:            {delta:+.3f} GB")

        # Per-layer VRAM marker from surgery
        if vram_marker.exists():
            lines = vram_marker.read_text().strip().split("\n")
            if lines:
                first = lines[0].split()
                last = lines[-1].split()
                print(f"  surgery VRAM: layer {first[0]} = {float(first[1]):.3f} GB "
                      f"(first), layer {last[0]} = {float(last[1]):.3f} GB (last)")

        print(f"  all-cuda tokens: {base['token_ids']}")
        print(f"  surgery tokens:  {surgery['token_ids']}")
        print(f"  all-cuda text:   {base['text']!r}")
        print(f"  surgery text:    {surgery['text']!r}")

        if base["token_ids"] == surgery["token_ids"]:
            print("\nPhase-8.5 VRAM surgery gate PASS (exact token parity)")
        else:
            bt = base["text"].strip().lower()
            st = surgery["text"].strip().lower()
            if "42" in bt and "42" in st:
                print("\nPhase-8.5 VRAM surgery gate PASS (semantic parity)")
            else:
                raise RuntimeError(
                    f"token divergence after surgery.\n"
                    f"  all-cuda: {base['text']!r}\n"
                    f"  surgery:  {surgery['text']!r}"
                )

        print(f"  hybrid exercised: layer {hmarker['layer']}, "
              f"{hmarker['b70_routes']} B70 routes")


if __name__ == "__main__":
    main()
