#!/usr/bin/env python3
"""Compare bounded Reyu functions with outputs computed by the actual Python source.

This exercises pure source algorithms, not vLLM scheduling integration or GPU
kernels. The scheduler's original function AST is loaded without importing its
vLLM base class; no scheduler implementation or function is mocked/replaced.
"""
from __future__ import annotations

import argparse
import ast
from collections import namedtuple
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time

SPEC = Path(__file__).resolve().parent
REPO = SPEC.parents[1]
sys.dont_write_bytecode = True


def load_module(relative, name):
    path = REPO / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_function_ast(relative, name):
    """Execute the unchanged top-level source function, excluding import side effects."""
    path = REPO / relative
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
    if len(nodes) != 1:
        raise ValueError(f"Expected one source function {name} in {relative}")
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def load_nvfp4_shape():
    """Load BankShape and its original struct constants, without torch/checkpoints."""
    relative = "src/phase1/extract_experts.py"
    path = REPO / relative
    tree = ast.parse(path.read_text(), filename=str(path))
    wanted = {"HEADER_FMT", "HEADER_SIZE", "GSCALE_FMT", "GSCALE_BYTES", "MAGIC"}
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Import) and any(alias.name == "struct" for alias in node.names):
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in wanted for target in node.targets):
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "BankShape":
            nodes.append(node)
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["BankShape"]


def reyu_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "List(" + ", ".join(reyu_value(item) for item in value) + ")"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{key}: {reyu_value(item)}" for key, item in value.items()) + " }"
    raise TypeError(f"Unsupported comparison value {value!r}")


def collect_cases():
    cases = []
    sources = []

    def add(module, function, arguments, expected, source, source_symbol):
        cases.append({"model": module, "function": function, "arguments": arguments,
                      "expected": expected, "source": source, "source_symbol": source_symbol})

    placement_path = "src/phase4/src/shooting_brake_vllm/placement.py"
    placement = load_module(placement_path, "sb_spec_placement_source")
    for remote_n in range(65):
        for weights in ((1, 1), (3, 2), (2, 3), (1, 7), (95, 75)):
            policy = placement.FractionalRemotePolicy((0, 1), 0.0, weights)
            counts = policy._remote_counts(remote_n)
            add("placement", "apportionTwoRemote", [remote_n, *weights],
                {"card0": counts[0], "card1": counts[1]}, placement_path, "FractionalRemotePolicy._remote_counts")
    production_counts = placement.FractionalRemotePolicy((0, 1), 0.0, (95, 75))._remote_counts(170)
    add("placement", "apportionTwoRemote", [170, 95, 75],
        {"card0": production_counts[0], "card1": production_counts[1]}, placement_path, "FractionalRemotePolicy._remote_counts")
    sources.append({"path": placement_path, "load": "normal module import and real FractionalRemotePolicy instance",
                    "domain": "two cards, remote counts 0..64 and production 170; rational/float ordering checked on these inputs only"})

    bank_path = "src/phase1/extract_experts.py"
    BankShape = load_nvfp4_shape()
    for hidden, intermediate in itertools.product((16, 32, 128, 256, 2048), (16, 64, 128, 512)):
        for layers, experts in ((1, 1), (2, 7), (32, 256)):
            shape = BankShape(hidden, intermediate, experts, list(range(layers)), "model")
            add("banks", "nvfp4ExpertBytes", [hidden, intermediate], shape.expert_bytes, bank_path, "BankShape.__init__")
            add("banks", "nvfp4BankBytes", [layers, experts, hidden, intermediate], shape.total_bytes, bank_path, "BankShape.total_bytes")
    sources.append({"path": bank_path, "load": "unchanged BankShape class AST and original struct constants; no torch or checkpoints"})
    int4_path = "src/phase4/src/shooting_brake_vllm/int4_bank_format.py"
    int4 = load_module(int4_path, "sb_spec_int4_source")
    for experts in (1, 2, 8, 85, 170, 256, 991, 992, 993, 1024):
        header = int4.Int4BankHeader(1, 1, experts, 512, 512, 128, 4, 8,
            tuple(index * 4096 for index in range(6)), (4096,) * 6, 24576, experts * 24576, tuple(range(experts)))
        header.validate()
        add("banks", "int4DataOffset", [experts], header.data_offset, int4_path, "Int4BankHeader.data_offset")
    sources.append({"path": int4_path, "load": "normal module import; real validated Int4BankHeader.data_offset"})

    profile_path = "experiments/analyze_campaign_profile.py"
    profile = load_module(profile_path, "sb_spec_profile_source")
    intervals = [(start, end) for start in range(4) for end in range(start, 5)]
    for left, right in itertools.product(intervals, repeat=2):
        add("campaign", "unionLength", [*left, *right], profile.duration([left, right]), profile_path, "duration")
        add("campaign", "overlapLength", [*left, *right], profile.intersection_duration([left], [right]), profile_path, "intersection_duration")
    invalid_rejected = False
    try:
        profile.duration([(2, 1)])
    except ValueError:
        invalid_rejected = True
    if not invalid_rejected:
        raise AssertionError("Actual interval source no longer rejects a negative interval")
    sources.append({"path": profile_path, "load": "normal module import", "negative_interval_rejected": invalid_rejected})

    scheduler_path = "experiments/interactive_scheduler.py"
    budget = load_function_ast(scheduler_path, "interactive_budget")
    RequestCounters = namedtuple("RequestCounters", "num_output_tokens num_computed_tokens num_prompt_tokens")
    counters = list(itertools.product((0, 1), (0, 1, 3), (1, 3)))
    for first, second in itertools.product(counters, repeat=2):
        for baseline, cap in ((8, 2), (2, 8), (4, 4)):
            requests = [RequestCounters(*first), RequestCounters(*second)]
            expected = budget(requests, baseline, cap)
            add("scheduler", "interactiveBudget", [*first, *second, baseline, cap], expected, scheduler_path, "interactive_budget")
    for baseline, cap in ((8, 2), (2, 8)):
        add("scheduler", "interactiveBudget", [0, 0, 0, 0, 0, 0, baseline, cap], budget([], baseline, cap), scheduler_path, "interactive_budget")
    sources.append({"path": scheduler_path, "load": "unchanged pure function AST; input counter records; vLLM integration not exercised"})

    import numpy as np
    locality_path = "benchmarks/route_locality.py"
    locality = load_module(locality_path, "sb_spec_locality_source")
    dtype = np.dtype([("step", np.uint64), ("row", np.uint32), ("experts", np.int32, (1,))])
    for length in range(6):
        for sequence in itertools.product(range(3), repeat=length):
            records = np.array([(index, 0, (expert,)) for index, expert in enumerate(sequence)], dtype=dtype)
            hits, requests = locality._lru_counts(records, (2,))[2]
            if requests != len(sequence):
                raise AssertionError("Actual route count does not match source fixture extent")
            add("observability", "lruCapacityTwoHits", [list(sequence)], hits, locality_path, "_lru_counts")
    for previous in range(5):
        for current in range(previous, previous + 4):
            for same_row in (False, True):
                records = np.array([(previous, 0, (0,)), (current, 0 if same_row else 1, (1,))], dtype=dtype)
                continues = len(locality._consecutive_runs(records)) == 1
                add("observability", "continuesRun", [previous, current, same_row], continues, locality_path, "_consecutive_runs")
    sources.append({"path": locality_path, "load": "normal module import; actual NumPy record arrays",
                    "domain": "all expert sequences of length 0..5 over IDs 0..2, capacity 2; consecutive/gapped/duplicate-step and row-boundary pairs"})

    return cases, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reyu", default=os.environ.get("REYU_BIN", "reyu"))
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--out", type=Path, default=SPEC / "implementation-comparison.json")
    args = parser.parse_args()
    cases, sources = collect_cases()
    imports = sorted({(case["model"], case["function"]) for case in cases})
    lines = ["module implementation_correspondence {"]
    for module, function in imports:
        source = f"../models/{module}"
        lines.append(f"  import {module}.{function} from {json.dumps(source)}")
    batches = [cases[start:start + 128] for start in range(0, len(cases), 128)]
    for index, batch in enumerate(batches):
        lines.append(f"  run batch{index}Test = all {{")
        for case in batch:
            arguments = ", ".join(reyu_value(value) for value in case["arguments"])
            expected = reyu_value(case["expected"])
            lines.append(f"    assert({case['function']}({arguments}) == {expected}),")
        lines.append("  }")
    lines.append("}")
    generated = "\n".join(lines) + "\n"
    prefix = shlex.split(args.reyu)
    if len(prefix) == 1 and prefix[0].endswith(".js"):
        prefix.insert(0, "node")
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=".correspondence-", dir=SPEC) as directory:
        path = Path(directory) / "implementation_correspondence.ryu"
        path.write_text(generated)
        command = prefix + ["test", str(path), "--seed", "42", "--max-samples", "1"]
        try:
            result = subprocess.run(command, cwd=REPO, text=True, capture_output=True, timeout=args.timeout)
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout or b""
            stderr = error.stderr or b""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            result = subprocess.CompletedProcess(command, 124, stdout, stderr + f"\nTimed out after {args.timeout} seconds")
    import re
    passed_batches = {int(index) for index in re.findall(r"ok batch(\d+)Test passed", result.stdout)}
    passed = sum(len(batches[index]) for index in passed_batches)
    for source in sources:
        source["sha256"] = hashlib.sha256((REPO / source["path"]).read_bytes()).hexdigest()
    report = {"schema_version": 1, "scope": "Bounded pure-function differential checks against actual source outputs",
              "limitations": ["Not full implementation equivalence", "No GPU kernels or serving processes exercised", "No scheduler superclass/integration exercised"],
              "source_evidence": sources, "case_count": len(cases), "passing_cases": passed,
              "batch_count": len(batches), "passing_batches": len(passed_batches),
              "cases": cases, "command": command, "returncode": result.returncode,
              "stdout": result.stdout, "stderr": result.stderr,
              "generated_spec_sha256": hashlib.sha256(generated.encode()).hexdigest(),
              "seconds": round(time.monotonic() - start, 4),
              "ok": result.returncode == 0 and passed == len(cases)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{'PASS' if report['ok'] else 'FAIL'}: {passed}/{len(cases)} implementation correspondence cases")
    if not report["ok"]:
        print(result.stdout)
        print(result.stderr)
    print(f"Report: {args.out}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
