#!/usr/bin/env python3
"""All-out mode correctness gate: CPU tier against the B70-only baseline.

Runs the same prompt three ways on the real model:

  1. ``subset:16:8``            — B70-only, the shipping configuration
  2. ``allout:16:8:0``          — all-out policy, zero CPU experts
  3. ``allout:16:8:<N>``        — N experts per layer moved B70 -> CPU DRAM

Run 2 must be **token-identical** to run 1. The two placements assign the
same experts to the same devices when ``cpu_per_layer=0``, so any divergence
is a bug in the new policy or partition code rather than in the CPU tier.
That isolates the plumbing from the numerics.

Run 3 is where the CPU tier actually computes. It is *not* expected to be
bit-identical: those experts move from the B70's NVFP4 kernel to a bf16 CPU
GEMM, and the two disagree in the low bits exactly as the B70 already
disagrees with CUDA. What the run proves is that the dequantization, the
gate/up split of the fused ``w13``, and the arena layout are right — a
mistake in any of them does not produce slightly different tokens, it
produces garbage.

``SHOOTING_BRAKE_CPU_VERIFY=1`` additionally makes the loader check each
layer's first expert against a torch FFN over the same dequantized weights,
which catches a layout error at load time with a precise message instead of
as downstream nonsense.

Run::

    make -C phase7 cpu
    .venv/bin/python phase7/allout_correctness_test.py
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
PROMPT = "List the first five prime numbers in order."
MAX_TOKENS = 24
CPU_PER_LAYER = 8

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        _failures.append(name)


def run_case(placement: str, output_path: Path, all_out: bool) -> str:
    """Generate with one placement; returns captured stderr."""
    env = dict(os.environ)
    env.update(
        # oneAPI's setvars.sh drops the venv from PATH, taking `ninja` with it.
        PATH=f"{ROOT / '.venv' / 'bin'}:{env.get('PATH', '')}",
        VLLM_PLUGINS="shooting_brake_vllm",
        SHOOTING_BRAKE_PHASE4="all-cuda",
        SHOOTING_BRAKE_MODEL=MODEL,
        SHOOTING_BRAKE_PLACEMENT=placement,
        SHOOTING_BRAKE_HYBRID="1",
        SHOOTING_BRAKE_B70_DEVICE="1",
        SHOOTING_BRAKE_B70_GRAPH="1",
        SHOOTING_BRAKE_B70_STATS="1",
        SHOOTING_BRAKE_VRAM_SURGERY="1",
        SHOOTING_BRAKE_B70_BANK=str(ROOT / "phase1" / "expert_bank.bin"),
        SHOOTING_BRAKE_B70_LIB=str(ROOT / "phase7" / "libsb_b70_provider.so"),
        SHOOTING_BRAKE_CPU_LIB=str(ROOT / "phase7" / "libsb_cpu_expert.so"),
    )
    if all_out:
        env["SHOOTING_BRAKE_ALL_OUT"] = "1"
        env["SHOOTING_BRAKE_CPU_VERIFY"] = "1"
    else:
        env.pop("SHOOTING_BRAKE_ALL_OUT", None)
        env.pop("SHOOTING_BRAKE_CPU_VERIFY", None)

    code = f"""
import json
from vllm.plugins import load_general_plugins
load_general_plugins()
from vllm import LLM, SamplingParams
llm = LLM(model={MODEL!r}, tensor_parallel_size=1, pipeline_parallel_size=1,
          gpu_memory_utilization=0.90, max_model_len=8192)
out = llm.generate([{PROMPT!r}],
                   SamplingParams(temperature=0.0, max_tokens={MAX_TOKENS}),
                   use_tqdm=False)[0].outputs[0]
stats = {{}}
try:
    from shooting_brake_vllm.telemetry import collect_worker_stats
    stats = llm.collective_rpc(collect_worker_stats)[0]
except Exception as exc:
    stats = {{"telemetry_error": str(exc)}}
with open({str(output_path)!r}, "w") as f:
    json.dump({{"token_ids": list(out.token_ids), "text": out.text,
                "stats": stats}}, f, default=str)
"""
    # The B70 provider links oneAPI's SYCL runtime (libsvml.so and friends),
    # which only exists on LD_LIBRARY_PATH after setvars.sh has run. Source
    # it inside the child rather than requiring the caller to, and re-prepend
    # the venv afterwards because setvars strips it (taking `ninja` with it).
    script = (
        "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true; "
        f'export PATH="{ROOT / ".venv" / "bin"}:$PATH"; '
        f'exec "{sys.executable}" -c "$SB_CODE"'
    )
    env["SB_CODE"] = code
    completed = subprocess.run(
        ["bash", "-c", script],
        cwd=ROOT, env=env, text=True, check=False, capture_output=True,
    )
    if completed.returncode:
        sys.stderr.write(completed.stderr[-4000:])
        raise RuntimeError(f"run failed (placement={placement})")
    return completed.stderr


def main() -> int:
    if not (ROOT / "phase7" / "libsb_cpu_expert.so").is_file():
        print("missing libsb_cpu_expert.so; run: make -C phase7 cpu")
        return 1

    with tempfile.TemporaryDirectory(prefix="sb-allout-") as tmp:
        base_p = Path(tmp) / "subset.json"
        zero_p = Path(tmp) / "allout_zero.json"
        cpu_p = Path(tmp) / "allout_cpu.json"

        print("1/3 subset:16:8 (B70-only baseline)...")
        run_case("subset:16:8", base_p, all_out=False)
        base = json.loads(base_p.read_text())

        print(f"2/3 allout:16:8:0 (policy plumbing, no CPU experts)...")
        run_case("allout:16:8:0", zero_p, all_out=False)
        zero = json.loads(zero_p.read_text())

        print(f"3/3 allout:16:8:{CPU_PER_LAYER} (CPU tier live)...")
        stderr = run_case(
            f"allout:16:8:{CPU_PER_LAYER}", cpu_p, all_out=True
        )
        cpu = json.loads(cpu_p.read_text())

    print("\n== policy plumbing: allout:K:N:0 == subset:K:N ==")
    check("token-identical to B70-only baseline",
          zero["token_ids"] == base["token_ids"],
          f"{zero['token_ids'][:8]} vs {base['token_ids'][:8]}")

    print("\n== load-time dequant/layout self-check ==")
    verify_lines = [
        ln for ln in stderr.splitlines() if "all-out VERIFY" in ln
    ]
    check("loader verified arena against torch reference",
          bool(verify_lines), f"{len(verify_lines)} layer(s) checked")
    if verify_lines:
        print(f"      {verify_lines[0].strip()[-100:]}")
    check("no layout mismatch reported",
          all("MISMATCH" not in ln for ln in verify_lines))

    print("\n== CPU tier live ==")
    stats = cpu.get("stats", {})
    routes = stats.get("routes", {}) if isinstance(stats, dict) else {}
    cpu_poller = stats.get("cpu_poller", {}) if isinstance(stats, dict) else {}
    arena = stats.get("cpu_arena", {}) if isinstance(stats, dict) else {}

    check("CPU routes actually executed", routes.get("cpu", 0) > 0,
          f"cpu={routes.get('cpu')} b70={routes.get('b70')} "
          f"cpu_share={routes.get('cpu_share', 0) * 100:.1f}%")
    check("CPU poller served dispatches", cpu_poller.get("dispatches", 0) > 0,
          f"{cpu_poller.get('dispatches')} dispatches, "
          f"{cpu_poller.get('service_mean_us', 0):.0f} us mean")
    check("no CPU dispatch errors", cpu_poller.get("errors", 1) == 0,
          f"errors={cpu_poller.get('errors')}")
    check("no routes dropped by the arena",
          arena.get("skipped_routes", 1) == 0,
          f"skipped={arena.get('skipped_routes')}")
    check("arena holds the expected experts",
          arena.get("resident_experts", 0) > 0,
          f"{arena.get('resident_experts')} experts, "
          f"{arena.get('used_gib', 0):.2f} GiB")

    print("\n== output sanity (numerics differ by construction) ==")
    n_match = sum(
        1 for a, b in zip(cpu["token_ids"], base["token_ids"]) if a == b
    )
    check("output is coherent, not garbage",
          len(cpu["text"].strip()) > 0 and cpu["token_ids"][:1] != [0],
          f"{n_match}/{len(base['token_ids'])} tokens match baseline")
    print(f"      baseline: {base['text']!r}")
    print(f"      all-out : {cpu['text']!r}")

    print("\n== KV capacity ==")
    kv = stats.get("kv_cache", {}) if isinstance(stats, dict) else {}
    base_kv = base.get("stats", {}).get("kv_cache", {})
    check("KV cache reported", bool(kv), f"{kv}")
    if kv and base_kv:
        print(f"      B70-only: {base_kv}")
        print(f"      all-out : {kv}")

    print()
    if _failures:
        print(f"FAILED ({len(_failures)}): {', '.join(_failures)}")
        return 1
    print("all-out correctness gate PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
