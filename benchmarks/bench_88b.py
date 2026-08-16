#!/usr/bin/env python3
"""Serving benchmark suite for srswti/axe-superveloce-88b on 5090 + Arc Pro B70.

Everything measurable is measured by GuideLLM, which is the vLLM team's own
harness -- this file only decides *which* cells to run and records provenance.
Nothing here reimplements load generation, prompt synthesis, or statistics.

Why GuideLLM's own mechanisms are used as-is
--------------------------------------------
* **Prompt synthesis**: ``kind=synthetic_text`` prepends a per-sample counter
  before tokenising (``data/deserializers/synthetic.py:433``), so every prompt
  starts with a distinct token. That is what keeps the numbers honest under
  prefix caching, which this vLLM build enables by default
  (``vllm/config/cache.py:93``).
* **Forced output length**: setting ``output_tokens`` makes GuideLLM send
  ``ignore_eos=True`` and ``stop=None`` (``openai/request_handlers.py:509``),
  so every request emits exactly the requested count. Cells are therefore
  comparable regardless of whether the model would have stopped early.
* **Warmup / cooldown**: fields on the profile (``schemas/profiles.py:63``),
  which is why they must be passed inline on ``--profile``. ``matrix_runner.py``
  declared them as top-level config and never emitted them, so its artifacts
  claimed a warmup that never happened. Passed correctly here.
* **Thinking toggle**: ``extras.body.*`` is merged into the JSON request body
  (``openai/request_handlers.py:515``). Thinking on/off is not cosmetic on a
  sparse MoE: different tokens take different routes, which changes how much
  work lands on the B70 and how many PCIe dispatches happen. Both are measured.

Capacity, not throughput
------------------------
This run's server reports 262,144 KV tokens and "maximum concurrency for
131,072 tokens per request: 2.00x" at ``max_num_seqs=64``. (An earlier
``max_num_seqs=8`` launch reported 292,103 tokens / 2.23x; raising the seat count
grew the CUDA graph capture set and spent 0.36 GiB of KV. The figures here are
the ones this matrix actually ran against.)

Capacity is charged in whole 4,176-token attention blocks, not tokens, because
the hybrid allocator pads the attention page to match GDN/Mamba. So 262,144
tokens is 62 seats, and a 640-token request costs a full 4,176. Cells whose
block-rounded ``streams x blocks`` exceeds that are admission-limited: the
server queues rather than running them concurrently. Those cells are marked
``capacity_bound`` so a queueing artifact is never read as a throughput
result.
"""

from __future__ import annotations

import argparse
import json
import math
import zlib
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

MODEL_SERVED = "shooting-brake-88b"
MODEL_TOKENIZER = "srswti/axe-superveloce-88b-nvfp4a16"
TARGET = "http://127.0.0.1:8016"

# Fallbacks measured from a PRIOR 128K server's startup log. Both values move
# with config (max_num_seqs, resident streamer buffers), which bit us once: a
# summary header quoted 262,144/62 seats against a server running 190,990/45.
# Always pass --kv-tokens/--max-num-seqs scraped from THIS run's server log;
# main() overrides these globals before any cell math runs.
KV_TOKENS = 262_144

# The hybrid allocator pads the attention block to match the GDN/Mamba page
# ("attention block size to 4176" in the server log). A sequence therefore
# consumes whole 4,176-token blocks, so a 640-token request costs 4,176 -- not
# 640. Capacity is KV_TOKENS / 4,176 blocks. Every admission check below
# rounds to blocks.
KV_BLOCK_TOKENS = 4_176
MAX_MODEL_LEN = 131_072
MAX_NUM_SEQS = 64
MAX_CONCURRENT_SEATS = KV_TOKENS // KV_BLOCK_TOKENS

# Chat template overhead measured on this model: a 128-token synthetic prompt
# arrives as 140 input tokens. Long-context cells need that margin or the
# request is rejected for exceeding max_model_len.
TEMPLATE_OVERHEAD = 16


@dataclass
class Cell:
    grid: str
    name: str
    profile: str
    prompt_tokens: int
    output_tokens: int = 512
    streams: list[int] | None = None
    sweep_size: int | None = None
    max_requests: int = 20
    max_seconds: int | None = None
    thinking: bool = False
    request_format: str = "/v1/chat/completions"
    notes: str = ""
    rampup: float = 0.0

    @property
    def cell_id(self) -> str:
        return f"{self.grid}/{self.name}"

    def blocks_per_seq(self) -> int:
        """Whole 4,176-token attention blocks one sequence occupies."""
        tokens = self.prompt_tokens + TEMPLATE_OVERHEAD + self.output_tokens
        return math.ceil(tokens / KV_BLOCK_TOKENS)

    def peak_streams(self) -> int:
        if self.streams:
            return max(self.streams)
        if self.profile in ("throughput", "sweep"):
            return MAX_NUM_SEQS
        return 1

    def peak_kv_tokens(self) -> int:
        """Worst-case live KV, block-rounded the way the allocator charges."""
        return self.peak_streams() * self.blocks_per_seq() * KV_BLOCK_TOKENS

    def capacity_bound(self) -> bool:
        return self.peak_kv_tokens() > KV_TOKENS

    def over_context(self) -> bool:
        return (
            self.prompt_tokens + TEMPLATE_OVERHEAD + self.output_tokens
        ) > MAX_MODEL_LEN


def build_cells(output_tokens: int) -> list[Cell]:
    cells: list[Cell] = []

    # -- Grid A: decode scaling. Short prompt so the cell is dominated by
    # decode, one concurrent profile covering every stream count. This is the
    # grid that answers "do we beat the PRO 6000's 135 tok/s at C=1, ~200 at
    # C=2" -- compare aggregate output tok/s, not per-request rate.
    # One cell per concurrency rather than one cell with every stream count:
    # GuideLLM applies max_requests per strategy, so a shared cap of 24 would
    # label a row C=62 while never admitting more than 24 requests. Each rung
    # gets at least two full waves (2 x C) and never fewer than 20 requests, so
    # the achieved concurrency in the report can actually reach the target.
    for streams in (1, 2, 4, 8, 16, 32, MAX_CONCURRENT_SEATS):
        cells.append(Cell(
            grid="A_decode", name=f"c{streams:03d}", profile="concurrent",
            prompt_tokens=128, output_tokens=output_tokens,
            streams=[streams], max_requests=max(20, 2 * streams),
            notes="decode-dominated; aggregate tok/s vs PRO 6000 baseline",
        ))

    # -- Grid B: prefill / TTFT vs context, one stream so TTFT is unqueued.
    # 130,048 is the largest prompt that still leaves room for 512 output plus
    # chat-template overhead inside a 131,072 window.
    for ctx in (128, 512, 2048, 8192, 16384, 32768, 65536, 130048):
        reqs = 20 if ctx <= 16384 else (8 if ctx <= 65536 else 4)
        cells.append(Cell(
            grid="B_context", name=f"ctx_{ctx}", profile="synchronous",
            prompt_tokens=ctx, output_tokens=output_tokens,
            max_requests=reqs,
            notes="TTFT vs context at C=1; cold prefix per request",
        ))

    # -- Grid C: concurrency at long context, where KV capacity binds before
    # compute does. 2.23x at 131,072 means C=2 is the ceiling up there.
    cells.append(Cell(
        grid="C_longctx", name="ctx_65536_c1-4", profile="concurrent",
        prompt_tokens=65536, output_tokens=output_tokens,
        streams=[1, 2, 3, 4], max_requests=8,
        notes=("long-context concurrency. 16 blocks x 4,176 = 66,816 tokens per "
               "seat, so C=3 fits at 200,448 and C=4 needs 267,264 > 262,144: "
               "the C=4 rung is the admission boundary and must be read as "
               "capacity, not throughput"),
    ))
    cells.append(Cell(
        grid="C_longctx", name="ctx_128928_c1-3", profile="concurrent",
        prompt_tokens=128928, output_tokens=output_tokens,
        streams=[1, 2, 3], max_requests=6,
        notes=("31 blocks x 4,176 = 129,456 tokens per seat: C=2 fits at "
               "258,912, C=3 needs 388,368 and is the admission boundary"),
    ))

    # -- Grid D: GuideLLM's own saturation finders. sweep interpolates from
    # synchronous to throughput and reports the knee; throughput just floods.
    cells.append(Cell(
        grid="D_saturation", name="sweep_ctx2048", profile="sweep",
        prompt_tokens=2048, output_tokens=output_tokens,
        sweep_size=6, max_requests=MAX_CONCURRENT_SEATS, max_seconds=300,
        notes="GuideLLM sweep: locates the saturation knee",
    ))
    cells.append(Cell(
        grid="D_saturation", name="throughput_ctx128", profile="throughput",
        prompt_tokens=128, output_tokens=output_tokens,
        max_requests=2 * MAX_CONCURRENT_SEATS, max_seconds=300,
        notes=("unbounded offered load; ceiling is the 62 KV seats rather "
               "than the 64-seat max_num_seqs flag"),
    ))

    # -- Grid E: thinking on vs off, paired. Same prompts, same forced output
    # length, so any delta is routing/template, not sequence length.
    for thinking in (False, True):
        cells.append(Cell(
            grid="E_thinking", name=f"think_{'on' if thinking else 'off'}",
            profile="concurrent", prompt_tokens=128, output_tokens=output_tokens,
            streams=[1, 4], max_requests=12, thinking=thinking,
            notes="paired A/B; MoE routing differs with reasoning tokens",
        ))

    return cells


def cell_seed(cell: Cell, base_seed: int) -> int:
    """Deterministic per-cell seed.

    Every GuideLLM subprocess restarts its synthetic generator from scratch, so
    a single shared seed makes cell N's sample K byte-identical to cell M's
    sample K (same unique counter, same seeded Faker stream). Cells whose
    prompts are >= the 4,176-token attention block would then hand each other
    complete cacheable prefixes, and prefix caching is on by default -- so a
    later cell's TTFT would measure a cache hit, not prefill.

    Paired cells (Grid E thinking on/off) deliberately keep the SAME seed:
    comparing routing cost requires identical prompts.
    """
    if cell.grid == "E_thinking":
        return base_seed
    return base_seed ^ (zlib.crc32(cell.cell_id.encode()) & 0x7FFF_FFFF)


def guidellm_command(cell: Cell, out_dir: Path, seed: int) -> list[str]:
    backend = (
        f"kind=openai_http,target={TARGET},model={MODEL_SERVED},"
        f"request_format={cell.request_format},"
        f"extras.body.chat_template_kwargs.enable_thinking="
        f"{'true' if cell.thinking else 'false'}"
    )
    # warmup/cooldown/rampup_duration are profile fields (schemas/profiles.py:
    # 59-72); passing them anywhere else is the silent no-op matrix_runner.py
    # shipped -- it declared them as top-level config and never emitted them.
    trim = cell.max_requests >= TRANSIENT_PHASE_MIN_REQUESTS
    transients = "warmup=0.1,cooldown=0.1" if trim else "warmup=0,cooldown=0"
    profile = (
        f"kind={cell.profile},{transients},"
        f"rampup_duration={cell.rampup}"
    )
    if cell.profile == "throughput":
        # This guidellm requires profile.throughput.max_concurrency (the
        # suite's one failed cell). Offer 2x the admission cap so the
        # ceiling measured is the server's, not the client's.
        profile += f",max_concurrency={2 * MAX_NUM_SEQS}"

    cmd = [
        sys.executable, "-m", "guidellm", "run",
        "--backend", backend,
        "--tokenizer",
        f"kind=huggingface_auto,model={MODEL_TOKENIZER},"
        f"load_kwargs.trust_remote_code=true",
        "--profile", profile,
        "--data",
        f"kind=synthetic_text,prompt_tokens={cell.prompt_tokens},"
        f"output_tokens={cell.output_tokens}",
        # StaticRandomArgs field is `value`, not `seed` (schemas/random.py:43).
        "--seed", f"kind=static,value={seed}",
    ]
    if cell.streams:
        cmd += ["--override", "profile.streams",
                ",".join(str(s) for s in cell.streams)]
    if cell.sweep_size is not None:
        cmd += ["--override", "profile.sweep_size", str(cell.sweep_size)]

    cmd += ["--constraint", f"kind=max_requests,count={cell.max_requests}"]
    if cell.max_seconds is not None:
        cmd += ["--constraint", f"kind=max_duration,seconds={cell.max_seconds}"]
    cmd += ["--constraint", "kind=max_errors,count=5"]

    for fmt in ("json", "csv", "html"):
        cmd += ["--output", f"kind={fmt},path={out_dir / f'report.{fmt}'}"]
    cmd.append("--disable-console-interactive")
    return cmd


# Percentage transient phases need a sample large enough to survive being
# trimmed. Measured: at max_requests=2 with warmup=cooldown=0.1, GuideLLM
# returned rc=0 and every metric 0.0 -- duration -0.0004 s, because both
# requests fell in transient phases and the measurement window closed before it
# opened. The same cell at N=4 with transients OFF measured cleanly (duration
# 17.97 s, 4/4 successful, 58.6 tok/s), and at N=12 with transients ON also
# measured cleanly (65.1 s, 12/12). So the rule is not a request floor, it is:
# only trim transients when there is enough sample left to be worth trimming.
# Costly long-context cells therefore keep small N and simply include their
# transients, which is stated in the manifest rather than hidden.
TRANSIENT_PHASE_MIN_REQUESTS = 12


def _measurement_is_real(report_path: Path) -> tuple[bool, dict]:
    """Confirm a cell actually measured something.

    GuideLLM exits 0 whether or not any request landed in the measurement
    window, so return code alone cannot distinguish a result from an empty
    window. Read the report and require a positive duration with at least one
    successful request.
    """
    if not report_path.exists():
        return False, {"reason": "report.json missing"}
    try:
        rep = json.loads(report_path.read_text())
    except json.JSONDecodeError as exc:
        return False, {"reason": f"report.json unparseable: {exc}"}
    bms = rep.get("benchmarks") or []
    if not bms:
        return False, {"reason": "no benchmarks in report"}
    checks = []
    for i, bm in enumerate(bms):
        totals = bm.get("metrics", {}).get("request_totals") or {}
        successful = totals.get(
            "successful", len(bm.get("requests", {}).get("successful") or [])
        )
        duration = bm.get("duration")
        ok = successful > 0 and isinstance(duration, (int, float)) and duration > 0
        checks.append({
            "index": i,
            "successful_measured": successful,
            "retained_including_transients": len(
                bm.get("requests", {}).get("successful") or []
            ),
            "errored": totals.get(
                "errored", len(bm.get("requests", {}).get("errored") or [])
            ),
            "duration": duration,
            "warmup_duration": bm.get("warmup_duration"),
            "cooldown_duration": bm.get("cooldown_duration"),
            "measured": ok,
        })
    return all(c["measured"] for c in checks), {"benchmarks": checks}


def run_cell(cell: Cell, root: Path, seed: int, skip_existing: bool) -> dict:
    out_dir = root / cell.grid / cell.name
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "run_manifest.json"

    if skip_existing and manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        # Requires `measured`, not just rc==0: GuideLLM exits 0 on an empty
        # measurement window, so resuming on rc alone would permanently accept
        # a zero-filled cell.
        if prior.get("return_code") == 0 and prior.get("measured") is True:
            print(f"  skip (already measured): {cell.cell_id}")
            return prior

    if cell.over_context():
        manifest = {
            "cell": cell.cell_id, "status": "skipped_over_context",
            "reason": (
                f"prompt {cell.prompt_tokens} + overhead {TEMPLATE_OVERHEAD} + "
                f"output {cell.output_tokens} exceeds max_model_len {MAX_MODEL_LEN}"
            ),
            "return_code": None,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"  SKIP over-context: {cell.cell_id}")
        return manifest

    seed = cell_seed(cell, seed)
    cmd = guidellm_command(cell, out_dir, seed)
    started = time.time()
    print(f"  run {cell.cell_id}  profile={cell.profile} "
          f"prompt={cell.prompt_tokens} out={cell.output_tokens} "
          f"streams={cell.streams or '-'}"
          f"{'  [CAPACITY-BOUND]' if cell.capacity_bound() else ''}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - started

    (out_dir / "stdout.log").write_text(proc.stdout)
    (out_dir / "stderr.log").write_text(proc.stderr)

    manifest = {
        "cell": cell.cell_id,
        "grid": cell.grid,
        "profile": cell.profile,
        "prompt_tokens": cell.prompt_tokens,
        "output_tokens": cell.output_tokens,
        "streams": cell.streams,
        "sweep_size": cell.sweep_size,
        "max_requests": cell.max_requests,
        "max_seconds": cell.max_seconds,
        "thinking": cell.thinking,
        "request_format": cell.request_format,
        "notes": cell.notes,
        # Provenance: the resolved command actually executed, so a reader never
        # has to trust that a declared setting was applied.
        "command": cmd,
        "seed": seed,
        "return_code": proc.returncode,
        "elapsed_seconds": round(elapsed, 2),
        "capacity": {
            "peak_kv_tokens_if_full": cell.peak_kv_tokens(),
            "kv_tokens_available": KV_TOKENS,
            "capacity_bound": cell.capacity_bound(),
            "interpretation": (
                "offered concurrency exceeds KV capacity; treat as admission "
                "capacity, not throughput"
                if cell.capacity_bound() else "fits in KV"
            ),
        },
    }

    # GuideLLM exits 0 even when no request landed in the measurement window,
    # so return code is not evidence that anything was measured. Check the
    # report and record the verdict, otherwise an all-zero cell would sit in
    # the artifacts looking exactly like a result.
    measured, detail = _measurement_is_real(out_dir / "report.json")
    manifest["measured"] = measured
    manifest["measurement_detail"] = detail
    manifest["transients_trimmed"] = (
        cell.max_requests >= TRANSIENT_PHASE_MIN_REQUESTS
    )

    manifest_path.write_text(json.dumps(manifest, indent=2))
    if proc.returncode != 0:
        status = f"FAIL rc={proc.returncode}"
    elif not measured:
        status = "FAIL empty-measurement-window"
    else:
        status = "ok"
    print(f"    {status} in {elapsed:.1f}s")
    return manifest


def _metric(bm: dict, key: str, stat: str = "mean"):
    """Read one statistic for a metric over successful requests.

    Percentiles live in a nested ``percentiles`` dict keyed ``p50``/``p95``/
    ``p99``, not as top-level fields, so asking for ``p95`` at the top level
    silently returns None -- which is how a tail-latency column ends up empty
    while looking merely unpopulated.
    """
    node = bm.get("metrics", {}).get(key, {}).get("successful") or {}
    if stat.startswith("p") and stat not in node:
        return (node.get("percentiles") or {}).get(stat)
    return node.get(stat)


def summarize(root: Path) -> str:
    """Flatten every cell's report.json into one table.

    Each GuideLLM sub-benchmark becomes a row carrying the concurrency
    GuideLLM actually observed. That figure is computed client-side from each
    request's start/end timestamps (``metrics.py:890``), so a *queued* request
    still counts as concurrent -- it is offered concurrency, not evidence that
    vLLM admitted that many sequences. For capacity-bound rows the two diverge,
    which is why those rows are flagged rather than read as throughput.
    """
    rows = []
    for manifest_path in sorted(root.rglob("run_manifest.json")):
        man = json.loads(manifest_path.read_text())
        report = manifest_path.parent / "report.json"
        if not report.exists():
            continue
        rep = json.loads(report.read_text())
        for bm in rep.get("benchmarks", []):
            # `requests.successful` is the retained sample list and INCLUDES
            # warmup/cooldown requests, while every latency/throughput statistic
            # beside it is computed only over the measurement window. Reporting
            # the list length next to those statistics overstates N -- at ctx
            # 8192 the list holds 20 but only 14 requests are measured. The
            # authoritative measured count is `metrics.request_totals`.
            totals = bm.get("metrics", {}).get("request_totals") or {}
            suc = totals.get("successful")
            attempted = len(bm.get("requests", {}).get("successful") or [])
            if suc is None:
                suc = attempted
            rows.append({
                "cell": man.get("cell"),
                "profile": man.get("profile"),
                "prompt_tokens_requested": man.get("prompt_tokens"),
                "thinking": man.get("thinking"),
                "client_concurrency": _metric(bm, "request_concurrency"),
                "successful": suc,
                "attempted_retained": attempted,
                "errored": totals.get(
                    "errored", len(bm.get("requests", {}).get("errored") or [])
                ),
                "duration_s": bm.get("duration"),
                "input_tokens": _metric(bm, "prompt_token_count"),
                "output_tokens": _metric(bm, "output_token_count"),
                "output_tok_per_s": _metric(bm, "output_tokens_per_second"),
                "total_tok_per_s": _metric(bm, "tokens_per_second"),
                "ttft_ms_mean": _metric(bm, "time_to_first_token_ms"),
                "ttft_ms_p95": _metric(bm, "time_to_first_token_ms", "p95"),
                "itl_ms_mean": _metric(bm, "inter_token_latency_ms"),
                "tpot_ms_mean": _metric(bm, "time_per_output_token_ms"),
                "tpot_ms_p95": _metric(bm, "time_per_output_token_ms", "p95"),
                "req_latency_s": _metric(bm, "request_latency"),
                "measured": man.get("measured"),
                # Per-row, not per-cell. A cell offering streams [1,2,3,4] is
                # flagged capacity-bound because of its C=4 rung; copying that
                # flag onto the C=1 row would erase the fit-vs-queue boundary
                # the cell exists to locate. Each row is judged on its own
                # observed concurrency and its own token count.
                "row_capacity_bound": None,
                "cell_peak_capacity_bound": man.get("capacity", {}).get(
                    "capacity_bound"),
            })
            row = rows[-1]
            c_obs = row["client_concurrency"]
            in_tok = row["input_tokens"] or man.get("prompt_tokens") or 0
            out_tok = row["output_tokens"] or 0
            if c_obs is not None and (in_tok or out_tok):
                blocks = math.ceil((in_tok + out_tok) / KV_BLOCK_TOKENS)
                seats = max(1, round(c_obs))
                row["row_kv_tokens"] = seats * blocks * KV_BLOCK_TOKENS
                row["row_capacity_bound"] = row["row_kv_tokens"] > KV_TOKENS

    (root / "summary.json").write_text(json.dumps(rows, indent=2))

    # A partially blank table is worse than no table: it looks authoritative
    # while hiding which numbers are absent. Refuse to publish unless every
    # measured row carries the fields the matrix exists to report.
    required = (
        "client_concurrency", "input_tokens", "output_tokens",
        "output_tok_per_s", "ttft_ms_mean", "ttft_ms_p95",
        "itl_ms_mean", "tpot_ms_mean", "tpot_ms_p95",
    )
    incomplete = [
        {"cell": r["cell"], "missing": [f for f in required if r.get(f) is None]}
        for r in rows
        if r.get("measured") and any(r.get(f) is None for f in required)
    ]
    if incomplete:
        raise RuntimeError(
            "summary is missing required fields on measured rows, refusing to "
            f"publish: {json.dumps(incomplete, indent=2)}"
        )

    def fmt(v, nd=1):
        if v is None:
            return "-"
        return f"{v:.{nd}f}" if isinstance(v, float) else str(v)

    lines = [
        "# 88B serving matrix - measured",
        "",
        f"Model: `{MODEL_SERVED}` ({MODEL_TOKENIZER})",
        f"Server: max_model_len={MAX_MODEL_LEN}, max_num_seqs={MAX_NUM_SEQS}, "
        f"KV={KV_TOKENS} tokens, attention block={KV_BLOCK_TOKENS} tokens "
        f"(=> {MAX_CONCURRENT_SEATS} concurrent seats)",
        f"Harness: GuideLLM, synthetic_text, ignore_eos forced, output={rows[0]['output_tokens'] if rows else '-'} tokens",
        "",
        "| cell | prof | in tok | client C | ok/err | out tok/s | total tok/s | TTFT ms | TTFT p95 | ITL ms | TPOT ms | TPOT p95 | KV fit | measured |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['cell']} | {r['profile']} | {fmt(r['input_tokens'],0)} | "
            f"{fmt(r['client_concurrency'],2)} | {r['successful']}/{r['errored']} | "
            f"**{fmt(r['output_tok_per_s'])}** | {fmt(r['total_tok_per_s'])} | "
            f"{fmt(r['ttft_ms_mean'])} | {fmt(r['ttft_ms_p95'])} | "
            f"{fmt(r['itl_ms_mean'],2)} | {fmt(r['tpot_ms_mean'],2)} | "
            f"{fmt(r['tpot_ms_p95'],2)} | "
            f"{'QUEUE' if r.get('row_capacity_bound') else 'fits'} | "
            f"{'yes' if r['measured'] else 'NO'} |"
        )
    text = "\n".join(lines) + "\n"
    (root / "MEASURED.md").write_text(text)
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path("benchmarks/results/run2_88b_128k/matrix"))
    ap.add_argument("--output-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=2928)
    ap.add_argument("--grids", default="",
                    help="Comma-separated grid names to run (exact), e.g. "
                         "B_context,A_decode; empty = all.")
    ap.add_argument("--cells", default="",
                    help="Comma-separated grid/name cells to run (exact), "
                         "e.g. B_context/ctx_8192; empty = all.")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summarize", action="store_true",
                    help="Rebuild MEASURED.md/summary.json from existing reports.")
    ap.add_argument("--kv-tokens", type=int, default=None,
                    help="GPU KV cache size in tokens from THIS server's log; "
                         "overrides the stale module fallback.")
    ap.add_argument("--max-num-seqs", type=int, default=None,
                    help="max_num_seqs of THIS server; overrides the fallback.")
    args = ap.parse_args()

    global KV_TOKENS, MAX_NUM_SEQS, MAX_CONCURRENT_SEATS
    if args.kv_tokens is not None:
        KV_TOKENS = args.kv_tokens
    if args.max_num_seqs is not None:
        MAX_NUM_SEQS = args.max_num_seqs
    MAX_CONCURRENT_SEATS = KV_TOKENS // KV_BLOCK_TOKENS

    if args.summarize:
        print(summarize(args.root))
        return 0

    cells = build_cells(args.output_tokens)
    if args.grids:
        wanted = {g.strip() for g in args.grids.split(",") if g.strip()}
        cells = [c for c in cells if c.grid in wanted]
    if args.cells:
        wanted_cells = {c.strip() for c in args.cells.split(",") if c.strip()}
        cells = [c for c in cells if f"{c.grid}/{c.name}" in wanted_cells]
        missing = wanted_cells - {f"{c.grid}/{c.name}" for c in cells}
        if missing:
            ap.error(f"unknown cells: {sorted(missing)}")

    args.root.mkdir(parents=True, exist_ok=True)
    print(f"{len(cells)} cells -> {args.root}")

    if args.dry_run:
        for c in cells:
            flag = " [CAPACITY-BOUND]" if c.capacity_bound() else ""
            over = " [OVER-CONTEXT]" if c.over_context() else ""
            print(f"  {c.cell_id:32s} {c.profile:12s} prompt={c.prompt_tokens:6d} "
                  f"out={c.output_tokens} streams={str(c.streams or '-'):12s}"
                  f" peakKV={c.peak_kv_tokens():7d}{flag}{over}")
        return 0

    manifests = []
    for cell in cells:
        manifests.append(run_cell(cell, args.root, args.seed, args.skip_existing))

    summary = args.root / "suite_manifest.json"
    summary.write_text(json.dumps({
        "model_served": MODEL_SERVED,
        "tokenizer": MODEL_TOKENIZER,
        "target": TARGET,
        "server_reported": {
            "kv_tokens": KV_TOKENS,
            "max_model_len": MAX_MODEL_LEN,
            "max_num_seqs": MAX_NUM_SEQS,
        },
        "output_tokens": args.output_tokens,
        "seed": args.seed,
        "cells": manifests,
    }, indent=2))
    print(f"\nsuite manifest -> {summary}")
    failed = [m for m in manifests if m.get("return_code") not in (0, None)]
    # A cell that ran but measured nothing is a failure, not a data point.
    empty = [
        m for m in manifests
        if m.get("return_code") == 0 and m.get("measured") is False
    ]
    if failed:
        print(f"{len(failed)} cell(s) failed: "
              f"{', '.join(m['cell'] for m in failed)}")
    if empty:
        print(f"{len(empty)} cell(s) produced an empty measurement window: "
              f"{', '.join(m['cell'] for m in empty)}")
    return 1 if (failed or empty) else 0


if __name__ == "__main__":
    raise SystemExit(main())
