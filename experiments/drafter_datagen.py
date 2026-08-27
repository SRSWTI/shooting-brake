#!/usr/bin/env python3
"""Drafter Stage-1 data generation: on-policy r15 outputs.

Builds a deterministic pilot prompt set from the box's own serving
distribution (agentic coding over real repo files + doc-grounded tasks
from the TTFT corpus + multi-step instructions), then generates r15
responses through the live server.

Design constraints (docs/drafter-finetune-plan.md, stage 1):
- ON-POLICY: responses come from the pruned r15 target itself.
- RESUMABLE: output is append-only jsonl keyed by prompt id; rerunning
  skips completed ids, so server reboots/bench windows just pause it.
- PAUSABLE: touch /tmp/sb_datagen.pause to make workers idle (checked
  between requests) so ITL/TTFT measurements are never contaminated.
- Low concurrency (default 3) to stay polite to interactive use.

Usage:
  python experiments/drafter_datagen.py \
      [--n 3000] [--concurrency 3] [--max-tokens 700] \
      [--out benchmarks/results/drafter_corpus/pilot.jsonl]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE = "http://127.0.0.1:8017"
PAUSE_FILE = "/tmp/sb_datagen.pause"

CODE_TEMPLATES = [
    "Explain what this code does, then identify its invariants and any "
    "risks:\n\n```\n{snippet}\n```",
    "Review this code for bugs and edge cases. Be specific and cite "
    "lines:\n\n```\n{snippet}\n```",
    "Refactor this code for clarity without changing behavior. Show the "
    "full result and justify each change:\n\n```\n{snippet}\n```",
    "Write thorough unit tests for the following code. Cover edge cases "
    "and failure modes:\n\n```\n{snippet}\n```",
    "Document this code: module docstring, per-function docstrings, and "
    "inline comments where non-obvious:\n\n```\n{snippet}\n```",
    "You are debugging a production incident. This code is the suspect. "
    "List hypotheses ranked by likelihood, and the exact checks to "
    "confirm each:\n\n```\n{snippet}\n```",
]

DOC_TEMPLATES = [
    "Summarize the following text in detail, then list the five most "
    "important claims with your assessment of each:\n\n{snippet}",
    "Answer as an expert: what are the key technical ideas in this text, "
    "and what would you challenge?\n\n{snippet}",
    "Rewrite the following text to be clearer and more precise, keeping "
    "all technical content:\n\n{snippet}",
]

AGENTIC = [
    "Plan the implementation of a rate limiter for a multi-tenant API: "
    "requirements, design options with tradeoffs, chosen design, then "
    "step-by-step implementation plan with tests.",
    "You have a flaky integration test that fails ~5% of runs. Lay out a "
    "systematic debugging campaign: instrumentation, bisection, and the "
    "fix categories you expect.",
    "Design a migration from a monolithic Postgres schema to a sharded "
    "layout with zero downtime. Cover dual-writes, backfill, cutover, "
    "and rollback.",
    "Write a detailed code review checklist for concurrent C++ code, "
    "with a short example of each failure mode.",
    "Given a service with p99 latency regressions after a deploy, walk "
    "through root-cause analysis end to end, naming the exact tools and "
    "queries you would use.",
    "Implement an LRU cache with TTL in Python, then harden it for "
    "thread safety and explain every locking decision.",
    "Explain how CUDA graphs work, when they help, their failure modes, "
    "and write a minimal capture/replay example.",
    "Design a benchmark harness that produces trustworthy latency "
    "numbers: warmup, isolation, statistics, and the mistakes that "
    "invalidate results.",
]


def _code_snippets(rng: random.Random, want: int) -> list[str]:
    files: list[Path] = []
    for pattern in ("src/**/*.py", "src/**/*.cpp", "src/**/*.hpp",
                    "benchmarks/*.py", "experiments/*.py"):
        files.extend(REPO.glob(pattern))
    files = [f for f in files if 2_000 < f.stat().st_size < 200_000]
    rng.shuffle(files)
    out: list[str] = []
    for f in files:
        try:
            text = f.read_text(errors="ignore")
        except OSError:
            continue
        lines = text.splitlines()
        if len(lines) < 40:
            continue
        start = rng.randrange(0, max(1, len(lines) - 120))
        out.append("\n".join(lines[start:start + 120])[:6000])
        if len(out) >= want:
            break
    return out


def build_prompts(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    corpus = Path.home() / "sb_corpus_big.txt"
    doc_text = corpus.read_text(errors="ignore") if corpus.exists() else ""
    n_code = int(n * 0.5)
    n_doc = int(n * 0.3)
    n_agent = n - n_code - n_doc

    prompts: list[dict] = []
    for snip in _code_snippets(rng, n_code):
        prompts.append({"kind": "code",
                        "prompt": rng.choice(CODE_TEMPLATES).format(snippet=snip)})
    for _ in range(n_doc):
        if not doc_text:
            break
        start = rng.randrange(0, max(1, len(doc_text) - 9000))
        snip = doc_text[start:start + rng.randrange(2500, 9000)]
        prompts.append({"kind": "doc",
                        "prompt": rng.choice(DOC_TEMPLATES).format(snippet=snip)})
    while len(prompts) < n:
        base = rng.choice(AGENTIC)
        prompts.append({"kind": "agentic", "prompt": base})
    rng.shuffle(prompts)
    for i, p in enumerate(prompts):
        p["id"] = f"pilot-{seed}-{i:05d}"
    return prompts[:n]


def generate(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out.exists():
        with out.open() as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    prompts = [p for p in build_prompts(args.n, args.seed)
               if p["id"] not in done]
    print(f"[datagen] {len(done)} done, {len(prompts)} remaining", flush=True)

    lock = threading.Lock()
    queue = list(reversed(prompts))
    stats = {"ok": 0, "fail": 0, "tokens": 0, "t0": time.time()}

    def worker() -> None:
        while True:
            with lock:
                if not queue:
                    return
                item = queue.pop()
            while os.path.exists(PAUSE_FILE):
                time.sleep(10)
            for attempt in range(1000):
                # Re-check the pause file HERE: without this, workers stuck
                # in server-down retry loops fire the instant a fresh boot
                # comes up, mid-benchmark (OOM'd a window on 2026-08-25).
                while os.path.exists(PAUSE_FILE):
                    time.sleep(10)
                try:
                    # RAW completions path, not chat: this vLLM's
                    # non-streaming chat API DISCARDS reasoning_content, and
                    # serving runs with thinking ON -- the drafter must learn
                    # to draft those tokens. Render the chat prompt via
                    # /tokenize, generate raw, save the verbatim completion
                    # text (thinking markup included).
                    tok_body = json.dumps({
                        "model": args.model,
                        "messages": [{"role": "user",
                                      "content": item["prompt"]}],
                        "add_generation_prompt": True,
                        "continue_final_message": False,
                    }).encode()
                    tok_req = urllib.request.Request(
                        BASE + "/tokenize", data=tok_body,
                        headers={"Content-Type": "application/json"})
                    prompt_tokens = json.load(
                        urllib.request.urlopen(tok_req, timeout=900))["tokens"]
                    body = json.dumps({
                        "model": args.model,
                        "prompt": prompt_tokens,
                        "add_special_tokens": False,
                        "max_tokens": args.max_tokens,
                        "temperature": args.temperature,
                        "top_p": 0.95,
                    }).encode()
                    req = urllib.request.Request(
                        BASE + "/v1/completions", data=body,
                        headers={"Content-Type": "application/json"})
                    r = json.load(urllib.request.urlopen(req, timeout=900))
                    choice = r["choices"][0]
                    rec = {
                        "id": item["id"], "kind": item["kind"],
                        "prompt": item["prompt"],
                        "raw_completion": choice.get("text") or "",
                        "prompt_token_count": len(prompt_tokens),
                        "finish": choice.get("finish_reason"),
                        "usage": r.get("usage", {}),
                    }
                    with lock:
                        with out.open("a") as fh:
                            fh.write(json.dumps(rec) + "\n")
                        stats["ok"] += 1
                        stats["tokens"] += rec["usage"].get(
                            "completion_tokens", 0)
                        if stats["ok"] % 25 == 0:
                            dt = time.time() - stats["t0"]
                            print(f"[datagen] {stats['ok']} responses, "
                                  f"{stats['tokens']} gen tokens, "
                                  f"{stats['tokens']/max(dt,1):.1f} tok/s "
                                  f"({stats['fail']} fails)", flush=True)
                    break
                except Exception:
                    # Server down (reboot/bench window): back off and retry.
                    time.sleep(min(60, 5 + attempt * 5))
            else:
                with lock:
                    stats["fail"] += 1

    threads = [threading.Thread(target=worker, daemon=True)
               for _ in range(args.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"[datagen] complete: {stats['ok']} ok, {stats['fail']} failed, "
          f"{stats['tokens']} generated tokens -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=15)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--model", default="shooting-brake-jota-r15")
    ap.add_argument("--out",
                    default="benchmarks/results/drafter_corpus/pilot.jsonl")
    sys.exit(generate(ap.parse_args()))
