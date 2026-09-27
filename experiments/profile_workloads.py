"""Run bounded serving workloads with native trace coverage and resource samples.

Run as `python -m experiments.profile_workloads`. The target must be an isolated
loopback vLLM development server with CampaignWorkerExtension installed. Existing
Shooting Brake HTTP, Prometheus, and PCIe samplers are reused.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import aiohttp
import picologging as logging
from transformers import AutoTokenizer

from benchmarks import server_context_probe as pcie_probe
from benchmarks.matrix_runner import _scrape_metrics
from benchmarks.slo_split_matrix import build_prompt, one_request, summarize

logger = logging.getLogger(__name__)


async def rpc(session, target, method, **kwargs):
    async with session.post(
        target + "/collective_rpc",
        json={"method": method, "kwargs": kwargs, "timeout": 120},
        timeout=aiohttp.ClientTimeout(total=130),
    ) as response:
        response.raise_for_status()
        payload = await response.json()
    results = payload["results"]
    if len(results) != 1:
        raise RuntimeError(f"expected one CUDA worker, received {len(results)}")
    return results[0]


async def run(args):
    if urlparse(args.target).hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("development RPC must remain on a loopback endpoint")
    if urlparse(args.target).port == 8017:
        raise ValueError("refusing to profile the production port")
    args.output.mkdir(parents=True, exist_ok=False)
    corpus = args.corpus.read_text()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
        local_files_only=True,
    )
    manifest = {
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "corpus_sha256": hashlib.sha256(corpus.encode()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "client_source_sha256": hashlib.sha256(
            Path(one_request.__code__.co_filename).read_bytes()
        ).hexdigest(),
        "workload_contract": "fixed corpus/nonces across arms; ignore_eos; authoritative continuous usage; unprofiled warmup excluded",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    pcie_probe.PCI_WATCH = {
        "rtx5090": "0000:01:00.0",
        "b70_0_upstream": "0000:13:00.0",
        "b70_1_upstream": "0000:0f:00.0",
        "shared_bridge": "0000:09:00.0",
        "root_port": "0000:00:02.1",
    }
    pcie = pcie_probe.PcieSampler(interval=0.25)
    stop_metrics = threading.Event()
    metrics_thread = threading.Thread(
        target=_scrape_metrics,
        args=(stop_metrics, args.target + "/metrics", args.output / "metrics.jsonl", 0.25),
        daemon=True,
    )
    gpu_log = (args.output / "gpu.csv").open("w")
    gpu_sampler = await asyncio.create_subprocess_exec(
        "nvidia-smi",
        "--query-gpu=timestamp,pci.bus_id,clocks.current.graphics,clocks.current.memory,temperature.gpu,power.draw,utilization.gpu,memory.used",
        "--format=csv",
        "--loop-ms=500",
        stdout=gpu_log,
        stderr=asyncio.subprocess.STDOUT,
    )
    pcie.start()
    metrics_thread.start()
    cases = []
    prompt_cache = {}

    def prompt(context, slot=0, family="cold"):
        key = (context, slot, family)
        if key not in prompt_cache:
            prompt_cache[key] = build_prompt(
                corpus,
                context,
                tokenizer,
                f"profile-{family}-{context}-slot{slot}",
            )
        return prompt_cache[key]

    async def case(session, label, specifications, *, mixed=False, trace=None):
        if trace is None:
            trace = args.tracing
        # Build/tokenize prompts before opening the measurement window.
        request_specs = [
            {
                "prompt": text,
                "max_tokens": output_tokens,
                "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "client_prompt_tokens": len(tokenizer.encode(text, add_special_tokens=False)),
            }
            for text, output_tokens in specifications
        ]
        started = await rpc(
            session,
            args.target,
            "campaign_start",
            label=label,
            output_root=str(args.output),
            tracing="1" if trace else "0",
        )
        results = []
        before = time.monotonic_ns()
        stop = None
        case_error = None
        try:

            async def request(index, first_event=None):
                spec = request_specs[index]
                return await one_request(
                    session,
                    args.target,
                    args.model,
                    spec["prompt"],
                    spec["max_tokens"],
                    args.timeout,
                    request_id=f"{label}-r{index}",
                    first_token_event=first_event,
                    ignore_eos=True,
                )

            if mixed:
                first_token = asyncio.Event()
                leader = asyncio.create_task(request(0, first_token))
                waiter = asyncio.create_task(first_token.wait())
                await asyncio.wait((leader, waiter), return_when=asyncio.FIRST_COMPLETED)
                if not first_token.is_set():
                    waiter.cancel()
                    await asyncio.gather(waiter, return_exceptions=True)
                    results = [await leader]
                    raise RuntimeError("decode leader ended before its first token")
                await waiter
                if leader.done():
                    results = [await leader]
                    raise RuntimeError("decode leader completed before prefill injection")
                results = await asyncio.gather(leader, request(1))
            else:
                results = await asyncio.gather(
                    *(request(index) for index in range(len(request_specs)))
                )
        except Exception as exc:
            logger.exception("Campaign workload failed: %s", label)
            case_error = f"{type(exc).__name__}: {exc}"
        finally:
            request_end = time.monotonic_ns()
            try:
                stop = await rpc(session, args.target, "campaign_stop")
            except Exception as exc:
                logger.exception("Campaign trace finalization failed: %s", label)
                stop = {"passed_trace_coverage": False, "errors": [f"{type(exc).__name__}: {exc}"]}
        wall = (request_end - before) / 1e9
        record = {
            "label": label,
            "tracing": trace,
            "mixed": mixed,
            "start_control": started,
            "stop_control": stop,
            "request_window_start_ns": before,
            "request_window_end_ns": request_end,
            "requests": results,
            "case_error": case_error,
            "inputs": [
                {key: value for key, value in spec.items() if key != "prompt"}
                for spec in request_specs
            ],
            "summary": summarize(results, wall),
        }
        (args.output / f"{label}.client.json").write_text(json.dumps(record, indent=2) + "\n")
        cases.append({"label": label, "summary": record["summary"], "trace_coverage": stop})
        print(label, json.dumps(record["summary"]), flush=True)
        if (
            case_error
            or not stop["passed_trace_coverage"]
            or len(results) != len(request_specs)
            or not all(result["ok"] for result in results)
        ):
            raise RuntimeError(f"{label}: request or trace-coverage failure")
        if any(
            result["tokens"] != spec["max_tokens"] for result, spec in zip(results, request_specs)
        ):
            raise RuntimeError(f"{label}: fixed output length was not satisfied")
        return record

    try:
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as session:
            async with session.get(args.target + "/metrics") as response:
                response.raise_for_status()
                initial_metrics = await response.text()
            (args.output / "initial-metrics.txt").write_text(initial_metrics)
            cache_lines = [
                line
                for line in initial_metrics.splitlines()
                if line.startswith("vllm:cache_config_info{")
            ]
            expected_cache = "True" if args.suite == "cache" else "False"
            if (
                len(cache_lines) != 1
                or f'enable_prefix_caching="{expected_cache}"' not in cache_lines[0]
            ):
                raise RuntimeError("observed prefix-cache policy does not match workload suite")
            await case(session, "warmup", [(prompt(1024, family="warmup"), 32)], trace=False)
            if args.suite in ("core", "all"):
                for context in args.contexts:
                    await case(
                        session, f"cold_{context}_c1", [(prompt(context), args.output_tokens)]
                    )
                for concurrency in (2, 4):
                    await case(
                        session,
                        f"decode_1024_c{concurrency}",
                        [
                            (prompt(1024, index, "concurrent"), args.output_tokens)
                            for index in range(concurrency)
                        ],
                    )
                leader = prompt(1024, family="mixed-leader")
                await case(session, "quiet_leader", [(leader, 256)])
                await case(
                    session,
                    "mixed_1k_decode_32k_prefill",
                    [(leader, 256), (prompt(32768, family="mixed-prefill"), 32)],
                    mixed=True,
                )
            if args.suite in ("capacity", "all"):
                await case(
                    session,
                    "capacity_32k_c4",
                    [(prompt(32768, index, "capacity"), 32) for index in range(4)],
                )
                await case(
                    session,
                    "capacity_96k_c2",
                    [(prompt(98304, index, "capacity"), 16) for index in range(2)],
                )
            if args.suite == "cache":
                for context in (8192, 32768):
                    prefix = prompt(context, family="cache")
                    await case(session, f"cache_prime_{context}", [(prefix, 32)])
                    suffix = "\nExplain the correctness constraints in the preceding material. " * 8
                    await case(
                        session,
                        f"cache_continue_{context}",
                        [(prefix + suffix, args.output_tokens)],
                    )
    finally:
        stop_metrics.set()
        metrics_thread.join(timeout=12)
        pcie_summary = pcie.stop()
        gpu_sampler.terminate()
        await asyncio.wait_for(gpu_sampler.wait(), timeout=10)
        gpu_log.close()
        (args.output / "pcie.json").write_text(
            json.dumps({"summary": pcie_summary, "samples": pcie.samples}, indent=2) + "\n"
        )
        (args.output / "summary.json").write_text(json.dumps(cases, indent=2) + "\n")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="http://127.0.0.1:8018")
    parser.add_argument("--model", default="srswti/axe-superveloce-jota-118b-r15-nvfp4")
    parser.add_argument("--revision", default="357b5c1f87b70cb89f7f66bee6dbbf90eaee279a")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=("core", "capacity", "cache", "all"), default="core")
    parser.add_argument("--contexts", type=int, nargs="+", default=(1024, 8192, 32768, 130048))
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--tracing", action="store_true")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
