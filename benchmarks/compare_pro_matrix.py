#!/usr/bin/env python3
"""One-to-one comparison: run6 F_matrix vs the RTX PRO 6000 bench matrix.

Ours: benchmarks/results/run6_final/matrix/F_matrix/ctx_*/report.json
      (GuideLLM concurrent profile, one benchmark entry per stream rung).
PRO:  ~/srswti/benchmarks-vllm/bench-matrix/superveloce_88b_nvfp4a16_c6/
      srswti__axe-superveloce-88b-nvfp4a16/ctx_*/concurrent/benchmarks.csv
      (mean column; C rows 1..6 in order -- verified column semantics:
      [mean, median, stddev, percentile-list]).

Emits a markdown grid of TTFT mean (s) ours vs PRO with gap ratios, and an
output-tok/s grid. C=10 is ours alone (PRO matrix stops at 6).

Usage:
  .venv/bin/python benchmarks/compare_pro_matrix.py \
      --ours benchmarks/results/run6_final/matrix \
      --out benchmarks/results/run6_final/PRO_COMPARISON.md
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PRO_ROOT = Path(
    "~/srswti/benchmarks-vllm/bench-matrix/superveloce_88b_nvfp4a16_c6/"
    "srswti__axe-superveloce-88b-nvfp4a16"
).expanduser()
CONTEXTS = (1024, 4096, 8192, 16384, 32768, 65536, 98304, 127000)
RATES = (1, 2, 3, 4, 5, 6, 10)


def pro_concurrent_means(ctx: int) -> dict[int, dict[str, float]]:
    """{C: {ttft_s, out_tps}} from the PRO csv's mean columns."""
    fp = PRO_ROOT / f"ctx_{ctx}" / "concurrent" / "benchmarks.csv"
    if not fp.is_file():
        return {}
    rows = list(csv.reader(fp.open()))
    hdr = [f"{a}|{b}" for a, b in zip(rows[0], rows[1])]
    strat_i = hdr.index("Benchmark|Strategy")
    ttft_i = hdr.index("Time to First Token|Successful ms")  # first = mean
    otps_i = hdr.index("Token Throughput|Successful Output Tokens/Sec")
    out, c = {}, 0
    for r in rows[2:]:
        if not any(r) or r[strat_i] != "concurrent":
            continue
        c += 1
        try:
            out[c] = {
                "ttft_s": float(r[ttft_i]) / 1000.0,
                "out_tps": float(r[otps_i]),
            }
        except ValueError:
            continue
    return out


def ours_concurrent_means(matrix_root: Path, ctx: int) -> dict[int, dict[str, float]]:
    """{C: {ttft_s, out_tps, ok}} from our F_matrix guidellm report.json."""
    fp = matrix_root / "F_matrix" / f"ctx_{ctx}" / "report.json"
    if not fp.is_file():
        return {}
    doc = json.loads(fp.read_text())
    out = {}
    for bm in doc.get("benchmarks", []):
        # Verified schema (run5 reports): the stream rung lives at
        # config.strategy.streams (int); metrics.<name>.successful.mean.
        strategy = (bm.get("config") or {}).get("strategy") or {}
        rate = strategy.get("streams") or strategy.get("max_concurrency")
        if rate is None:
            continue
        rate = int(round(float(rate)))
        metrics = bm.get("metrics") or {}

        def stat(name: str) -> float | None:
            m = metrics.get(name) or {}
            s = m.get("successful") or {}
            v = s.get("mean")
            return float(v) if v is not None else None

        ttft_ms = stat("time_to_first_token_ms")
        otps = stat("output_tokens_per_second")
        ok = ((bm.get("request_totals") or {}).get("successful")
              or (bm.get("run_stats") or {}).get("requests_made", {}).get("successful"))
        if ttft_ms is None:
            continue
        out[rate] = {"ttft_s": ttft_ms / 1000.0,
                     "out_tps": otps or 0.0, "ok": ok}
    return out


def fmt_gap(ours: float | None, pro: float | None) -> str:
    if ours is None:
        return "-"
    if pro is None:
        return f"{ours:.2f}s"
    ratio = ours / pro
    mark = " **WIN**" if ratio <= 1.0 else ""
    return f"{ours:.2f}s / {pro:.2f}s ({ratio:.2f}x){mark}"


def report_ttft(fp: Path, want_rate: int = 1) -> float | None:
    """C=1-equivalent TTFT mean (s) from any of our guidellm report.json files.

    Synchronous cells carry no streams field -> rate 1 by definition.
    """
    if not fp.is_file():
        return None
    for bm in json.loads(fp.read_text()).get("benchmarks", []):
        strategy = (bm.get("config") or {}).get("strategy") or {}
        if strategy.get("type_") == "synchronous":
            rate = 1
        else:
            r = strategy.get("streams") or strategy.get("max_concurrency")
            if r is None:
                continue
            rate = int(round(float(r)))
        if rate != want_rate:
            continue
        s = ((bm.get("metrics") or {}).get("time_to_first_token_ms") or {}).get(
            "successful") or {}
        v = s.get("mean")
        return float(v) / 1000.0 if v is not None else None
    return None


# (run_label, ctx) -> report.json path. run4/run5 are frozen artifacts; the
# 130048-vs-127000 rows are compared as "128K-class" (ours is the largest
# prompt fitting the 131,072 window with 512 output + template overhead).
HISTORY_RUNS = ("run4", "run5", "run6")
HISTORY_CTX = (8192, 16384, 32768, 65536, 130048)


def history_path(run: str, ctx: int, run6_root: Path) -> Path | None:
    r4 = Path("benchmarks/results/run4_88b_bank/matrix/B_context")
    r5 = Path("benchmarks/results/run5_88b_register/matrix")
    if run == "run4":
        return r4 / f"ctx_{ctx}" / "report.json"
    if run == "run5":
        if ctx in (8192, 16384, 32768):
            return r5 / "B_context" / f"ctx_{ctx}" / "report.json"
        if ctx == 65536:
            return r5 / "C_longctx" / "ctx_65536_c1-4" / "report.json"
        if ctx == 130048:
            return r5 / "B_context" / "ctx_130048" / "report.json"
        return None
    if run == "run6":
        return run6_root / "F_matrix" / f"ctx_{127000 if ctx == 130048 else ctx}" / "report.json"
    return None


def history_table(run6_root: Path) -> list[str]:
    lines = [
        "",
        "# Our runs, C=1 TTFT mean (s) -- the campaign, context by context",
        "",
        "run4 = pre-repacked bank; run5 = registered page-cache DMA;",
        "run6 = run5 + KV levers + stream-threshold fix + dual model names.",
        "",
        "| ctx | run4 | run5 | run6 | PRO 6000 | run6 vs PRO |",
        "|---|---|---|---|---|---|",
    ]
    for ctx in HISTORY_CTX:
        vals = {}
        for run in HISTORY_RUNS:
            fp = history_path(run, ctx, run6_root)
            vals[run] = report_ttft(fp) if fp else None
        pro = pro_concurrent_means(127000 if ctx == 130048 else ctx).get(1)
        cells = [f"{vals[r]:.2f}" if vals[r] else "-" for r in HISTORY_RUNS]
        pro_s = f"{pro['ttft_s']:.2f}" if pro else "-"
        gap = "-"
        if vals["run6"] and pro:
            ratio = vals["run6"] / pro["ttft_s"]
            gap = f"{ratio:.2f}x" + (" **WIN**" if ratio <= 1.0 else "")
        label = "128K-class" if ctx == 130048 else str(ctx)
        lines.append(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} | {pro_s} | {gap} |")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ours", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    lines = [
        "# run6 vs RTX PRO 6000 -- TTFT mean (ours / PRO, ratio; lower is better)",
        "",
        "Same checkpoint, same GuideLLM harness, concurrent profile means.",
        "PRO source: bench-matrix/superveloce_88b_nvfp4a16_c6. C=10 is ours alone.",
        "",
        "| ctx \\ C | " + " | ".join(str(c) for c in RATES) + " |",
        "|---|" + "---|" * len(RATES),
    ]
    tps_lines = [
        "",
        "# Output tok/s mean (ours / PRO)",
        "",
        "| ctx \\ C | " + " | ".join(str(c) for c in RATES) + " |",
        "|---|" + "---|" * len(RATES),
    ]
    for ctx in CONTEXTS:
        pro = pro_concurrent_means(ctx)
        ours = ours_concurrent_means(args.ours, ctx)
        row, tps_row = [f"| {ctx} "], [f"| {ctx} "]
        for c in RATES:
            o, p = ours.get(c), pro.get(c)
            row.append("| " + fmt_gap(o and o["ttft_s"], p and p["ttft_s"]) + " ")
            if o is None:
                tps_row.append("| - ")
            else:
                ptps = f"{p['out_tps']:.0f}" if p else "-"
                tps_row.append(f"| {o['out_tps']:.0f} / {ptps} ")
        lines.append("".join(row) + "|")
        tps_lines.append("".join(tps_row) + "|")

    hist = history_table(args.ours)
    doc = "\n".join(lines + tps_lines + hist) + "\n"
    args.out.write_text(doc)
    print(doc)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
