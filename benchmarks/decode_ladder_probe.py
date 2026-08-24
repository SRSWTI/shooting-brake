#!/usr/bin/env python3
"""Decode ladder: ITL / TPOT / tok/s at several context lengths, 512 tokens out.

Mirrors the RTX PRO 6000 reference shape (512 generated tokens per request,
contexts 1K..64K) so the curves are directly comparable. Prompts are
prefix-disjoint (unique random header + corpus slice), so prefill is cold and
decode starts from a genuine KV of the target depth.

With speculative decoding, tokens arrive in bursts, so chunk-gap medians lie;
the honest headline is TPOT = (wall - ttft) / (tokens - 1), reported alongside
the burst-aware ITL. Acceptance comes from /metrics deltas per request.

Usage:
  .venv/bin/python benchmarks/decode_ladder_probe.py --label ngram \
      --contexts 1024,8192,16384,32768,65536 --out-tokens 512
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import requests


def spec_counters(url: str) -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        for line in requests.get(f"{url}/metrics", timeout=10).text.splitlines():
            if line.startswith("#") or "spec_decode" not in line:
                continue
            name, _, val = line.rpartition(" ")
            key = name.split("{")[0]
            out[key] = out.get(key, 0.0) + float(val)
    except Exception:
        pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8017")
    ap.add_argument("--model", default="shooting-brake-jota-r15")
    ap.add_argument("--corpus", type=Path, default=Path.home() / "sb_corpus_big.txt")
    ap.add_argument("--contexts", default="1024,8192,16384,32768,65536")
    ap.add_argument("--out-tokens", type=int, default=512)
    ap.add_argument("--label", default="run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", type=Path,
                    default=Path("benchmarks/results/b70_gemv_audit/decode_ladder.json"))
    a = ap.parse_args()

    words = a.corpus.read_text(errors="ignore").split()
    rng = random.Random(a.seed)
    rows = []
    for ctx in (int(x) for x in a.contexts.split(",")):
        # ~2.26 tokens/word measured on this corpus (code-heavy); leave room for the header
        n_words = max(8, int(ctx / 2.26) - 16)
        start = rng.randrange(0, max(1, len(words) - n_words - 1))
        header = f"Doc {rng.randrange(1 << 40):#x}. "
        prompt = header + " ".join(words[start:start + n_words])

        c0 = spec_counters(a.url)
        t0 = time.perf_counter()
        r = requests.post(
            f"{a.url}/v1/completions", stream=True, timeout=1800,
            json={"model": a.model, "prompt": prompt,
                  "max_tokens": a.out_tokens, "temperature": 0.0,
                  "stream": True,
                  "stream_options": {"include_usage": True}},
        )
        r.raise_for_status()
        arrivals, ptok, otok = [], None, None
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            payload = line[6:]
            if payload == b"[DONE]":
                break
            d = json.loads(payload)
            if d.get("usage"):
                ptok = d["usage"]["prompt_tokens"]
                otok = d["usage"]["completion_tokens"]
            if d.get("choices") and d["choices"][0].get("text"):
                arrivals.append(time.perf_counter())
        wall = time.perf_counter() - t0
        c1 = spec_counters(a.url)

        if len(arrivals) < 2:
            print(f"ctx {ctx}: too few chunks ({len(arrivals)})")
            continue
        ttft = arrivals[0] - t0
        n = otok or len(arrivals)
        tpot = (arrivals[-1] - arrivals[0]) / max(1, n - 1) * 1e3
        gaps = sorted((b - x) * 1e3 for x, b in zip(arrivals, arrivals[1:]))
        drafted = c1.get("vllm:spec_decode_num_draft_tokens_total", 0) - \
            c0.get("vllm:spec_decode_num_draft_tokens_total", 0)
        accepted = c1.get("vllm:spec_decode_num_accepted_tokens_total", 0) - \
            c0.get("vllm:spec_decode_num_accepted_tokens_total", 0)
        acc = accepted / drafted * 100 if drafted else None
        row = dict(label=a.label, ctx=ctx, prompt_tokens=ptok, out_tokens=n,
                   ttft_s=round(ttft, 3), tpot_ms=round(tpot, 3),
                   chunk_gap_p50_ms=round(gaps[len(gaps) // 2], 3),
                   decode_tok_s=round((n - 1) / (arrivals[-1] - arrivals[0]), 1),
                   wall_s=round(wall, 2),
                   drafted=int(drafted), accepted=int(accepted),
                   acceptance_pct=round(acc, 1) if acc is not None else None)
        rows.append(row)
        print(f"ctx {ctx:>6}: ptok={ptok} ttft={ttft:6.2f}s TPOT={tpot:6.2f}ms "
              f"decode={row['decode_tok_s']:6.1f} tok/s "
              f"accept={row['acceptance_pct']}% ({int(accepted)}/{int(drafted)})")

    a.json_out.parent.mkdir(parents=True, exist_ok=True)
    hist = json.loads(a.json_out.read_text()) if a.json_out.exists() else []
    hist.extend(rows)
    a.json_out.write_text(json.dumps(hist, indent=1))
    print(f"appended {len(rows)} rows -> {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
