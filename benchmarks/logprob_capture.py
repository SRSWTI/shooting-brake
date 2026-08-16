#!/usr/bin/env python3
"""Capture or diff prompt logprobs over the OpenAI completions API.

The register-path A/B gate: SHOOTING_BRAKE_BANK_REGISTER=1 changes the H2D
transport, not one byte of weights or one kernel, so prompt logprobs must
match the flag-off arm to float determinism (expected max_abs_delta ~0,
gate at 1e-3 to absorb cross-boot nondeterminism in reduction order).

  capture:  logprob_capture.py capture --target URL --prompt-file F --out A.json
  diff:     logprob_capture.py diff --a A.json --b B.json --out ab.json \
                [--gate 1e-3]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path


def capture(args) -> None:
    prompt = Path(args.prompt_file).read_text()
    if args.max_chars:
        # prompt_logprobs mode only: vLLM's prompt-logprobs path allocates
        # M x vocab fp32 logits (~1.6 GiB at 2K tokens) and OOM-kills the
        # engine above ~1K tokens on a 0.90-utilization server. But <1024
        # tokens never engages the streamer (B70_STREAM_THRESHOLD), so
        # prompt-logprobs CANNOT gate the streamer path over the API --
        # use --gen-tokens for that.
        prompt = prompt[: args.max_chars]
    if args.gen_tokens:
        # Streamer-path gate: full prompt prefills through the streamer,
        # then N greedy tokens with per-step chosen-token logprobs (one
        # vocab row per step -- no big buffer, engine survives).
        body = {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": args.gen_tokens,
            "temperature": 0,
            "logprobs": 0,
        }
    else:
        body = {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
            "prompt_logprobs": 0,
        }
    req = urllib.request.Request(
        args.target.rstrip("/") + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=args.timeout) as r:
        resp = json.load(r)
    choice = resp["choices"][0]
    if args.gen_tokens:
        lp = choice.get("logprobs") or {}
        vals = lp.get("token_logprobs") or []
        vals = [v for v in vals if v is not None]
        text = choice.get("text", "")
    else:
        plp = choice.get("prompt_logprobs")
        if not plp:
            sys.exit("server returned no prompt_logprobs; needs vLLM OpenAI "
                     "completions with prompt_logprobs support")
        # Position 0 has no prediction behind it; entries {token_id: {...}}.
        vals = [next(iter(d.values()))["logprob"] for d in plp[1:] if d]
        text = ""
    if not vals:
        sys.exit("no logprobs in response")
    out = {"n": len(vals), "prompt_file": args.prompt_file,
           "gen_tokens": args.gen_tokens, "text": text, "logprobs": vals}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    print(f"captured {len(vals)} logprobs -> {args.out}")


def diff(args) -> None:
    a = json.loads(Path(args.a).read_text())
    b = json.loads(Path(args.b).read_text())
    if a["n"] != b["n"]:
        sys.exit(f"length mismatch: {a['n']} vs {b['n']} -- different "
                 "tokenization or prompt; A/B invalid")
    deltas = [x - y for x, y in zip(a["logprobs"], b["logprobs"])]
    absd = [abs(d) for d in deltas]
    out = {
        "n": len(deltas),
        "mean_delta": sum(deltas) / len(deltas),
        "mean_abs_delta": sum(absd) / len(absd),
        "max_abs_delta": max(absd),
        "gate": args.gate,
        "pass": max(absd) <= args.gate,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    if not out["pass"]:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--target", default="http://127.0.0.1:8016")
    c.add_argument("--model", default="shooting-brake-88b")
    c.add_argument("--prompt-file", required=True)
    c.add_argument("--max-chars", type=int, default=0)
    c.add_argument("--gen-tokens", type=int, default=0)
    c.add_argument("--timeout", type=float, default=600.0)
    c.add_argument("--out", required=True)
    d = sub.add_parser("diff")
    d.add_argument("--a", required=True)
    d.add_argument("--b", required=True)
    d.add_argument("--gate", type=float, default=1e-3)
    d.add_argument("--out", required=True)
    args = ap.parse_args()
    {"capture": capture, "diff": diff}[args.cmd](args)


if __name__ == "__main__":
    main()
