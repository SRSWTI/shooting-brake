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
# The PRO matrix has no 128-token grid and no C>6, so the decode and peak rows
# below are ours-only by construction -- compared against their nearest
# context (1024) with the caveat stated inline.


def pro_sweep_peak(ctx: int) -> dict[str, float]:
    """{strategy: out_tps} from the PRO sweep profile (throughput = peak)."""
    fp = PRO_ROOT / f"ctx_{ctx}" / "sweep" / "benchmarks.csv"
    if not fp.is_file():
        return {}
    rows = list(csv.reader(fp.open()))
    hdr = [f"{a}|{b}" for a, b in zip(rows[0], rows[1])]
    s_i = hdr.index("Benchmark|Strategy")
    t_i = hdr.index("Token Throughput|Successful Output Tokens/Sec")
    out = {}
    for r in rows[2:]:
        if not any(r):
            continue
        try:
            out[r[s_i]] = float(r[t_i])
        except (ValueError, IndexError):
            continue
    return out


def our_grid_cells(root: Path, grid: str) -> dict[str, list[dict]]:
    """{cell_name: [{strategy, streams, out_tps, itl_ms, ttft_s}, ...]}."""
    base = root / grid
    if not base.is_dir():
        return {}
    out: dict[str, list[dict]] = {}
    for cell in sorted(base.iterdir()):
        fp = cell / "report.json"
        if not fp.is_file():
            continue
        rows = []
        for bm in json.loads(fp.read_text()).get("benchmarks", []):
            st = (bm.get("config") or {}).get("strategy") or {}
            m = bm.get("metrics") or {}

            def mean(name: str) -> float | None:
                s = (m.get(name) or {}).get("successful") or {}
                v = s.get("mean")
                return float(v) if v is not None else None

            rows.append({
                "strategy": st.get("type_"),
                "streams": st.get("streams") or st.get("max_concurrency"),
                "out_tps": mean("output_tokens_per_second"),
                "itl_ms": mean("inter_token_latency_ms"),
                "ttft_s": (mean("time_to_first_token_ms") or 0) / 1000 or None,
            })
        out[cell.name] = rows
    return out


def decode_and_peak_tables(root: Path) -> list[str]:
    """Decode-shaped grid + peak-throughput rows, both ours-only vs PRO@1K."""
    lines: list[str] = []
    dec = our_grid_cells(root, "A_decode")
    if dec:
        pro1k = pro_concurrent_means(1024)
        lines += [
            "",
            "# Decode-shaped grid (128-token prompt, 512 out) -- ours",
            "",
            "The PRO matrix has no 128-token cell; its nearest is ctx_1024,",
            "shown for the C rungs it covers (1-6). C>6 is ours alone.",
            "",
            "| C | out tok/s ours | out tok/s PRO@1K | ITL ms ours | ITL ms PRO@1K |",
            "|---|---|---|---|---|",
        ]
        for name in sorted(dec):
            for r in dec[name]:
                c = r["streams"]
                if c is None:
                    continue
                p = pro1k.get(int(c), {})
                po = f"{p['out_tps']:.0f}" if p.get("out_tps") else "-"
                pi = f"{p['itl_ms']:.2f}" if p.get("itl_ms") else "-"
                oo = f"{r['out_tps']:.0f}" if r["out_tps"] else "-"
                oi = f"{r['itl_ms']:.2f}" if r["itl_ms"] else "-"
                lines.append(f"| {int(c)} | {oo} | {po} | {oi} | {pi} |")
    sat = our_grid_cells(root, "D_saturation")
    if sat:
        lines += [
            "",
            "# Peak throughput, unbounded offered load",
            "",
            "PRO bracket from their sweep profile: 798 @1K, 613 @4K, 462 @8K",
            "out tok/s. Our saturation cells sit at 128 and 2048 tokens, so the",
            "2048 row interpolates against ~700 on their curve [INFERENCE].",
            "",
            "| our cell | strategy | out tok/s |",
            "|---|---|---|",
        ]
        for name in sorted(sat):
            for r in sat[name]:
                if not r["out_tps"]:
                    continue
                lines.append(
                    f"| {name} | {r['strategy']} | {r['out_tps']:.1f} |")
        for ctx in (1024, 4096, 8192):
            pk = pro_sweep_peak(ctx)
            if pk:
                got = ", ".join(f"{k} {v:.0f}" for k, v in pk.items())
                lines.append(f"| PRO ctx_{ctx} (reference) | sweep | {got} |")
    return lines




def pro_concurrent_means(ctx: int) -> dict[int, dict[str, float]]:
    """{C: {ttft_s, itl_ms, tpot_ms, out_tps}} from the PRO csv's mean columns.

    Column semantics verified: each metric group's FIRST column is the mean
    (the following ones are median, stddev, then a bracketed percentile list).
    """
    fp = PRO_ROOT / f"ctx_{ctx}" / "concurrent" / "benchmarks.csv"
    if not fp.is_file():
        return {}
    rows = list(csv.reader(fp.open()))
    hdr = [f"{a}|{b}" for a, b in zip(rows[0], rows[1])]
    cols = {
        "ttft_s": "Time to First Token|Successful ms",
        "itl_ms": "Inter Token Latency|Successful ms",
        "tpot_ms": "Time per Output Token|Successful ms",
        "out_tps": "Token Throughput|Successful Output Tokens/Sec",
    }
    idx = {k: hdr.index(v) for k, v in cols.items() if v in hdr}
    strat_i = hdr.index("Benchmark|Strategy")
    out: dict[int, dict[str, float]] = {}
    c = 0
    for r in rows[2:]:
        if not any(r) or r[strat_i] != "concurrent":
            continue
        c += 1
        row = {}
        for k, i in idx.items():
            try:
                v = float(r[i])
            except (ValueError, IndexError):
                continue
            row[k] = v / 1000.0 if k == "ttft_s" else v
        if row:
            out[c] = row
    return out


def ours_concurrent_means(matrix_root: Path, ctx: int) -> dict[int, dict[str, float]]:
    """{C: {ttft_s, itl_ms, tpot_ms, out_tps, ok}} from our F_matrix reports."""
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
        ok = ((bm.get("request_totals") or {}).get("successful")
              or (bm.get("run_stats") or {}).get("requests_made", {}).get("successful"))
        if ttft_ms is None:
            continue
        out[rate] = {
            "ttft_s": ttft_ms / 1000.0,
            "itl_ms": stat("inter_token_latency_ms"),
            "tpot_ms": stat("time_per_output_token_ms"),
            "out_tps": stat("output_tokens_per_second") or 0.0,
            "ok": ok,
        }
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


def all_metrics_c1_table(root: Path) -> list[str]:
    """Every shared metric at C=1, ours vs PRO, context by context."""
    lines = [
        "",
        "# Every shared metric at C=1 (ours / PRO)",
        "",
        "| ctx | TTFT s | ITL ms | TPOT ms | out tok/s |",
        "|---|---|---|---|---|",
    ]
    for ctx in CONTEXTS:
        p = pro_concurrent_means(ctx).get(1, {})
        o = ours_concurrent_means(root, ctx).get(1, {})
        if not o and not p:
            continue

        def pair(key: str, fmt: str = "{:.2f}") -> str:
            a, b = o.get(key), p.get(key)
            sa = fmt.format(a) if a else "-"
            sb = fmt.format(b) if b else "-"
            better = ""
            if a and b:
                win = a > b if key == "out_tps" else a < b
                better = " **W**" if win else ""
            return f"{sa} / {sb}{better}"

        lines.append(
            f"| {ctx} | {pair('ttft_s')} | {pair('itl_ms')} | "
            f"{pair('tpot_ms')} | {pair('out_tps', '{:.0f}')} |"
        )
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

    doc = "\n".join(
        lines
        + tps_lines
        + all_metrics_c1_table(args.ours)
        + decode_and_peak_tables(args.ours)
        + history_table(args.ours)
    ) + "\n"
    args.out.write_text(doc)
    print(doc)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
