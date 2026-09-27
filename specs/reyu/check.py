#!/usr/bin/env python3
"""Check source correspondence and exercise every declared Reyu model.

No production processes, GPU workloads, package installs or existing project test
suites are invoked. Successful simulation is sampled evidence, not a proof.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import time

SPEC = Path(__file__).resolve().parent
REPO = SPEC.parents[1]


def load_catalog():
    inventory = json.loads((SPEC / "inventory.json").read_text())
    maps = [json.loads(path.read_text()) for path in sorted((SPEC / "maps").glob("*.json"))]
    return inventory, maps


def validate_catalog(inventory, maps):
    errors = []
    covered = {}
    models = {}
    for branch in maps:
        identity = branch["id"]
        chapter = SPEC / "chapters" / f"{identity}.md"
        if not chapter.is_file():
            errors.append(f"Missing chapter: {chapter.relative_to(SPEC)}")
        for source in branch["sources"]:
            relative = source["path"]
            if not (REPO / relative).is_file():
                errors.append(f"Source does not exist: {relative}")
            covered.setdefault(relative, set()).add(identity)
            for model in source.get("models", []):
                if not (SPEC / model).is_file():
                    errors.append(f"Missing source-linked model: {model}")
        for model in branch["models"]:
            relative = model["path"]
            if relative in models and models[relative] != model:
                errors.append(f"Conflicting model registration: {relative}")
            models[relative] = model
            if not (SPEC / relative).is_file():
                errors.append(f"Missing model: {relative}")
    for source in inventory["files"]:
        relative = source["path"]
        path = REPO / relative
        if not path.is_file():
            errors.append(f"Inventoried source removed: {relative}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != source["sha256"]:
            errors.append(f"Source drift requires correspondence review: {relative}")
        if source["branch"] not in covered.get(relative, set()):
            errors.append(f"Missing owning-branch correspondence: {relative} -> {source['branch']}")
    helpers = {"models/architecture.ryu"}
    for path in sorted((SPEC / "models").glob("*.ryu")):
        relative = str(path.relative_to(SPEC))
        if relative not in models and relative not in helpers:
            errors.append(f"Unregistered executable model: {relative}")
    documents = [SPEC / "README.md", *sorted((SPEC / "chapters").glob("*.md"))]
    for document in documents:
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", document.read_text()):
            if target.startswith(("http:", "https:", "mailto:", "#")):
                continue
            relative = target.split("#", 1)[0]
            if relative and not (document.parent / relative).exists():
                errors.append(f"Broken navigation link in {document.relative_to(SPEC)}: {target}")
    return sorted(set(errors)), list(models.values())


def command_prefix(value):
    parts = shlex.split(value)
    if not parts:
        raise ValueError("Empty Reyu command")
    if len(parts) == 1 and parts[0].endswith(".js"):
        parts.insert(0, "node")
    if not (Path(parts[0]).is_file() or shutil.which(parts[0])):
        raise ValueError(f"Reyu command not found: {parts[0]}; pass --reyu /path/to/reyu or cli.js")
    return parts


def execute(command, timeout):
    start = time.monotonic()
    try:
        result = subprocess.run(command, cwd=REPO, text=True, capture_output=True, timeout=timeout)
        return {"command": command, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr,
                "seconds": round(time.monotonic() - start, 4)}
    except subprocess.TimeoutExpired as exc:
        def text(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
        return {"command": command, "returncode": 124, "stdout": text(exc.stdout),
                "stderr": text(exc.stderr) + "\nVerification command timed out",
                "seconds": round(time.monotonic() - start, 4)}


def check_model(model, prefix, args):
    source = SPEC / model["path"]
    common = [str(source), "--main", model["main"]]
    results = []
    typecheck = execute(prefix + ["typecheck", str(source)], args.timeout)
    results.append({"stage": "typecheck", **typecheck})
    if typecheck["returncode"]:
        return {"model": model["path"], "ok": False, "checks": results}
    with tempfile.TemporaryDirectory(prefix="shooting-brake-reyu-") as directory:
        target = Path(directory) / "compiled.json"
        compiled = execute(prefix + ["compile"] + common + ["--target", "json", "--out", str(target)], args.timeout)
        if compiled["returncode"] == 0:
            if not target.is_file():
                compiled["returncode"] = 1
                compiled["stderr"] += "\nCompiler did not produce the requested JSON IR"
            else:
                payload = target.read_bytes()
                json.loads(payload)
                compiled["artifact_sha256"] = hashlib.sha256(payload).hexdigest()
                compiled["artifact_bytes"] = len(payload)
                compiled["stdout_omitted_bytes"] = len(compiled["stdout"].encode())
                compiled["stdout"] = ""
                compiled["artifact_retained"] = False
        results.append({"stage": "compile-json", **compiled})
    scenarios = execute(prefix + ["test"] + common + ["--seed", str(args.seed), "--max-samples", "1"], args.timeout)
    passing = re.search(r"(\d+) passing", scenarios["stdout"])
    scenarios["passing_scenarios"] = int(passing.group(1)) if passing else 0
    if scenarios["returncode"] == 0 and not scenarios["passing_scenarios"]:
        scenarios["returncode"] = 1
        scenarios["stderr"] += "\nNo executable scenarios were reported"
    results.append({"stage": "scenarios", **scenarios})
    simulate = prefix + ["run"] + common + ["--invariant", model.get("invariant", "inv"),
        "--max-samples", str(args.samples), "--max-steps", str(args.steps),
        "--seed", str(args.seed), "--verbosity", "1"]
    if model.get("witnesses"):
        simulate += ["--witnesses", *model["witnesses"]]
    results.append({"stage": "sampled-invariants", **execute(simulate, args.timeout)})
    return {"model": model["path"], "ok": all(r["returncode"] == 0 for r in results), "checks": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reyu", default=os.environ.get("REYU_BIN", "reyu"), help="Reyu executable, command, or built cli.js")
    parser.add_argument("--validate-only", action="store_true", help="Only check catalog, files and source hashes")
    parser.add_argument("--model", action="append", help="Only exercise the named module (catalog is still checked)")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--out", type=Path, default=SPEC / "verification.json")
    args = parser.parse_args()
    if min(args.samples, args.steps, args.jobs, args.timeout) < 1:
        parser.error("samples, steps, jobs and timeout must be positive")
    inventory, maps = load_catalog()
    errors, models = validate_catalog(inventory, maps)
    chosen = [m for m in models if not args.model or m["main"] in args.model]
    if args.model:
        unknown = set(args.model) - {m["main"] for m in models}
        errors.extend(f"Unknown model: {name}" for name in sorted(unknown))
    report = {"schema_version": 1, "scope": "source hashes, model compilation, concrete runs and sampled invariants",
              "proof_claim": "None: neither simulation nor source linkage establishes implementation equivalence.",
              "inventory_files": len(inventory["files"]), "branches": len(maps),
              "catalog_errors": errors, "samples_per_model": args.samples,
              "max_steps": args.steps, "seed": args.seed, "models": []}
    if errors:
        for error in errors:
            print(f"CATALOG: {error}")
    if not args.validate_only:
        try:
            prefix = command_prefix(args.reyu)
        except ValueError as error:
            parser.error(str(error))
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for result in pool.map(lambda model: check_model(model, prefix, args), chosen):
                report["models"].append(result)
                print(f"{'PASS' if result['ok'] else 'FAIL'} {result['model']}", flush=True)
                for check in result["checks"]:
                    if check["returncode"]:
                        print(check["stage"], check["stdout"], check["stderr"], flush=True)
    report["ok"] = not errors and all(model["ok"] for model in report["models"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {args.out}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
