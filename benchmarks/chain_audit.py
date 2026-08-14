#!/usr/bin/env python3
"""Audit a Track A matrix output tree for completeness.

A cell is complete iff its run_manifest.json exists and reports
return_code == 0 -- the same condition matrix_runner.py's --skip-existing
uses, so "complete" here means exactly "the runner would skip it".

Exit 0 when every expected cell is complete, 1 otherwise. Prints one row
per missing/failed cell so the log says what is wrong, not just that
something is.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_CONTEXTS = "1024,4096,8192,16384,32768,65536,98304,127000"
DEFAULT_PROFILES = "synchronous,concurrent,sweep"


def slugify_model(name: str) -> str:
    return name.replace("/", "__")


def cell_state(cell_dir: Path) -> tuple[str, str]:
    manifest = cell_dir / "run_manifest.json"
    if not manifest.exists():
        return ("MISSING", "no run_manifest.json")
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return ("CORRUPT", f"unreadable manifest: {exc}")
    rc = data.get("return_code")
    if rc == 0:
        return ("OK", "")
    return ("FAILED", f"return_code={rc!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--model", default="unsloth/Qwen3.6-35B-A3B-NVFP4")
    parser.add_argument("--contexts", default=DEFAULT_CONTEXTS)
    parser.add_argument("--profiles", default=DEFAULT_PROFILES)
    args = parser.parse_args()

    contexts = [c.strip() for c in args.contexts.split(",") if c.strip()]
    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    model_dir = args.output_root / slugify_model(args.model)

    ok, bad = 0, []
    for ctx in contexts:
        for profile in profiles:
            state, why = cell_state(model_dir / f"ctx_{ctx}" / profile)
            if state == "OK":
                ok += 1
            else:
                bad.append((ctx, profile, state, why))

    total = len(contexts) * len(profiles)
    print(f"[audit] root={model_dir}")
    print(f"[audit] complete {ok}/{total}")
    for ctx, profile, state, why in bad:
        print(f"[audit]   {state:8s} ctx_{ctx}/{profile}  {why}")

    if bad:
        print("[audit] VERDICT=ABORT")
        return 1
    print("[audit] VERDICT=PROCEED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
