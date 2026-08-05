#!/usr/bin/env python3
"""Compare one real eager stock-vLLM request with the Phase-4 local adapter."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from shooting_brake_vllm.config import QUALIFIED_MODEL

ROOT = Path(__file__).resolve().parents[1]
PROMPT = "State the integer after 41."


def run_case(adapter: bool, output_path: Path) -> None:
    env = dict(os.environ)
    env["VLLM_PLUGINS"] = "shooting_brake_vllm"
    if adapter:
        env["SHOOTING_BRAKE_PHASE4"] = "all-cuda"
        env["SHOOTING_BRAKE_MODEL"] = QUALIFIED_MODEL
    else:
        env.pop("SHOOTING_BRAKE_PHASE4", None)
        env.pop("SHOOTING_BRAKE_MODEL", None)

    code = f'''
import json
from vllm.plugins import load_general_plugins
load_general_plugins()
from vllm.model_executor.custom_op import op_registry_oot
from vllm import LLM, SamplingParams
adapter = {adapter!r}
if adapter:
    assert op_registry_oot.get("MoERunner").__name__ == "HybridMoERunner"
    assert op_registry_oot.get("RoutedExperts").__name__ == "HybridRoutedExperts"
else:
    assert "MoERunner" not in op_registry_oot
    assert "RoutedExperts" not in op_registry_oot
llm = LLM(
    model={QUALIFIED_MODEL!r},
    enforce_eager=True,
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    gpu_memory_utilization=0.90,
    max_model_len=8192,
    enable_return_routed_experts=True,
)
request = llm.generate(
    [{PROMPT!r}], SamplingParams(temperature=0.0, max_tokens=8), use_tqdm=False
)[0]
output = request.outputs[0]
routes = output.routed_experts
if routes is None:
    raise RuntimeError("vLLM did not return the requested router-ID trace")
payload = {{
    "token_ids": output.token_ids,
    "text": output.text,
    "routing_data": routes.tolist(),
}}
with open({str(output_path)!r}, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True)
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"{'adapter' if adapter else 'stock'} run failed")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="shooting-brake-phase4-") as temporary:
        stock_path = Path(temporary) / "stock.json"
        adapter_path = Path(temporary) / "adapter.json"
        run_case(adapter=False, output_path=stock_path)
        run_case(adapter=True, output_path=adapter_path)
        stock = json.loads(stock_path.read_text())
        adapter = json.loads(adapter_path.read_text())

        # The generation output is the Phase-4 gate. At temperature 0 the
        # emitted token IDs (and therefore the text) are deterministic and
        # must match exactly between stock and the all-CUDA adapter.
        #
        # The per-token routed-expert trace is NOT a valid cross-process
        # comparison surface: it is the router's own topk_ids captured
        # verbatim (see vllm routed_experts_capturer.py), and the router
        # gate-matmul -> softmax -> top-8 is bit-nondeterministic across
        # separate process launches (CUDA reduction order varies). At
        # near-tie boundaries this flips which of the 256 experts lands in
        # the 8th slot, so even the per-token expert SET differs. This was
        # confirmed empirically: stock vLLM run twice differs from itself
        # in the trace while producing identical token IDs. The Shooting
        # Brake adapter runs strictly downstream of the router and capture
        # and cannot affect the trace, so the trace is reported only as a
        # soft drift diagnostic, never as a gate.
        if stock["token_ids"] != adapter["token_ids"]:
            raise RuntimeError(
                "all-CUDA adapter token-id mismatch: "
                + json.dumps(
                    {"stock": stock["token_ids"], "adapter": adapter["token_ids"]}
                )
            )
        if stock["text"] != adapter["text"]:
            raise RuntimeError(
                "all-CUDA adapter text mismatch: "
                + json.dumps({"stock": stock["text"], "adapter": adapter["text"]})
            )
        # Soft diagnostic: count token-layers whose expert set differs.
        # Expected to be nonzero purely from router nondeterminism.
        set_drift = 0
        for s_tok, a_tok in zip(
            stock["routing_data"], adapter["routing_data"]
        ):
            for s_layer, a_layer in zip(s_tok, a_tok):
                if set(s_layer) != set(a_layer):
                    set_drift += 1
        total_token_layers = sum(len(t) for t in stock["routing_data"])
        print("Phase-4 eager all-CUDA parity PASS")
        print(f"token_ids={stock['token_ids']}")
        print(f"text={stock['text']!r}")
        print(f"router_trace_tokens={len(stock['routing_data'])}")
        print(
            f"routed-expert set drift {set_drift}/{total_token_layers} "
            "token-layers (expected >0 from router nondeterminism; not a gate)"
        )


if __name__ == "__main__":
    main()
