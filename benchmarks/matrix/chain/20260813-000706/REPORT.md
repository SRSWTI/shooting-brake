# Benchmark chain report — 20260813-000706

Scope: Track A (guidellm SLO matrix, live hybrid server) + Track B (in-process
offload sweep). Nothing was re-run for this report; every figure below is a
field read out of a checked-in artifact, cited by path and key. No artifact was
modified and the GPU power cap was left as found.

Provenance:

| what | evidence |
|---|---|
| Track A wrapper exited, audit clean | `chain/20260813-000706/chain.log:3-6` — `complete 24/24`, `VERDICT=PROCEED` |
| Track A root | `benchmarks/matrix/hybrid_131k_c6/unsloth__Qwen3.6-35B-A3B-NVFP4/` — 8 contexts × 3 profiles = 24 cell dirs |
| Track B started / finished | `chain.log:10-12` — `starting Track B`, `Track B exited rc=0` |
| Track B parameters | `chain.log:11` — `contexts='2048 8192 32768 131072' concurrency='1 4 8 16 32 64' max_model_len=131072 trials=3` |
| Track B artifacts | `benchmarks/results/offload_full/{all-cuda,hybrid-subset-8-8,hybrid-subset-16-8,hybrid-subset-24-64,hybrid-split-128}.json` |
| adapter build | `adapter_sha = fede998f0b0aca46`, identical in all five Track B JSONs (`.adapter_sha`) |

One caveat on Track A: the placement is **not recorded** in any Track A cell
artifact. `matrix_config.json` has no `placement` key, and the guidellm command
files record only the endpoint. `subset:16:8` is the default of the server
launcher (`benchmarks/serve_hybrid.sh:48` — `PLACEMENT="${PLACEMENT:-subset:16:8}"`),
and Track A's concurrency-1 ITL p50 (5.52–5.61 ms at ≤32k, §3) matches Track B's
`subset:16:8` single-stream ITL p50 of 5.308 ms far better than any other arm, so
`subset:16:8` is consistent with the data — but it is `[INFERENCE]`, not a
recorded fact.

---

## 1. Track B — per-placement decode, latency, capacity, B70 routing

All values from `benchmarks/results/offload_full/<file>.json`.

- decode tok/s = `.single_stream.decode_tok_per_s_mean` (3 trials, 400 output
  tokens, `.single_stream.trials` / `.output_tokens`)
- ITL p50/p99 = `.single_stream.itl_p50_ms` / `.itl_p99_ms`
- KV max_tokens = `.workers[0].kv_cache.max_tokens`
- B70 share / dispatch service µs = `.workers_decode_only[0].routes.b70_share`
  and `.workers_decode_only[0].poller.service_mean_us` — the snapshot taken
  **after single-stream decode and before the batched/frontier phases**, which
  is the decode-only figure. The run-total snapshot (`.workers[0]`, taken at the
  very end) is given alongside because that is the one the console summary
  prints, and for service time the two differ by more than 10×.

| placement | file | decode tok/s | % of all-cuda | ITL p50 ms | ITL p99 ms | KV max_tokens | KV × | B70 share (decode) | B70 share (run) | dispatch svc µs (decode) | dispatch svc µs (run) | dispatches (run) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all-cuda | `all-cuda.json` | 252.599 | 100% | 3.958 | 4.206 | 387,760 | 1.00× | — (no `routes` key) | — | — (no `poller` key) | — | — |
| subset:8:8 | `hybrid-subset-8-8.json` | 214.744 | 85.0% | 4.660 | 4.866 | 786,000 | 2.03× | 96.08% | 96.77% | 221.5 | 2426.6 | 5,706,968 / 5,897,664 routes |
| subset:16:8 | `hybrid-subset-16-8.json` | 188.294 | 74.5% | 5.308 | 5.546 | 1,129,744 | 2.91× | 96.35% | 97.10% | 186.1 | 2654.3 | 1,173,408 |
| subset:24:64 | `hybrid-subset-24-64.json` | 171.435 | 67.9% | 5.838 | 6.086 | 1,238,736 | 3.19× | 72.98% | 73.81% | 154.8 | 2191.4 | 1,797,912 |
| split:128 | `hybrid-split-128.json` | 159.920 | 63.3% | 6.254 | 6.469 | 1,146,512 | 2.96× | 50.68% | 50.02% | 128.3 | 1650.6 | 2,363,680 |

Cross-check: the console summary at `track_b.log:1035-1039` prints the same
ordering from `.single_stream.tok_per_s_mean` (250.6 / 213.5 / 187.1 / 170.5 /
159.1) — that column is end-to-end tok/s including TTFT, ~0.5–1.5% below the
`decode_tok_per_s_mean` used above. Both are in the artifacts; they are not in
conflict.

`all-cuda` carries no B70 telemetry at all: `.workers[0]` has no `routes` and no
`poller` key (the only worker keys are `hybrid_layers`, `kv_cache`,
`cuda_memory`, `host_memory`, `cuda_power`). This is why the summary table
prints `svc_us = nan` for that row (`track_b.log:1035`) — absence of the B70
path, not a failed measurement.

Two things worth reading off this table:

1. **Capacity and latency move together and monotonically, except at the top of
   the capacity range.** `subset:24:64` buys the most KV (1,238,736 tok, 3.19×)
   but `split:128` — which frees comparable VRAM (1,146,512 tok, 2.96×) — is
   *slower* (159.9 vs 171.4 tok/s) because it activates 32 layers instead of 24
   and therefore pays more dispatches (2,363,680 vs 1,797,912 run-total). Same
   capacity class, worse latency: concentration wins, consistent with
   `README.md:107-110`.
2. **Dispatch service time is not the per-dispatch cost of a decode token.** The
   decode-only figures (128–222 µs) are within an order of magnitude of the
   ~91 µs floor quoted in `benchmarks/README.md:104` and `:121`. The run-total
   figures (1650–2654 µs) are dominated by the batched and capacity-frontier
   phases, where the single SYCL queue serializes concurrent layers
   (`benchmarks/README.md:118-119`). Quoting the run-total number as a decode
   dispatch cost would overstate it by 10–17×.

---

## 2. Capacity frontier per context length

**The premise "which contexts all-cuda cannot admit at all" does not hold: there
is no context in this sweep that all-cuda fails to admit.** Every placement,
including all-cuda, completed every concurrency wave it attempted at all four
context lengths. What differs across placements is how far up the concurrency
ladder the harness *goes*, and that ceiling is derived analytically from measured
KV capacity before any request is sent.

`max_completed_wave` per `.capacity_frontier[*]`:

| target prompt tokens | all-cuda | subset:8:8 | subset:16:8 | subset:24:64 | split:128 |
|---|---|---|---|---|---|
| 2,048 | 64 | 64 | 64 | 64 | 64 |
| 8,192 | 48 | 64 | 64 | 64 | 64 |
| 32,768 | 12 | 24 | 48 | 48 | 48 |
| 131,072 | 4 | 8 | 12 | 12 | 12 |

Evidence that these are ceilings on the *ladder*, not admission failures:

- The wave list is `[1,4,8,12,16,24,32,48,64]` truncated to
  `fits = kv_max_tokens // (target_prompt_tokens + decode_tokens)` plus exactly
  one wave beyond it (`benchmarks/offload_benchmark.py:534-546`), with
  `decode_tokens = 400` (`offload_benchmark.py:664`, `run_offload_sweep.sh:61`).
- Recomputing that formula from each file's `.workers[0].kv_cache.max_tokens`
  reproduces the attempted ladder exactly in all 20 (placement × context) rows.
  For all-cuda at 131,072: `387760 // 131472 = 2` → attempt `[1, 4]`, which is
  precisely `.capacity_frontier[3].points` (`concurrent_requests` 1 and 4).
- In all 20 rows `max_completed_wave` equals the **last** wave attempted, i.e.
  the loop never hit its `break` on exception or zero-output
  (`offload_benchmark.py:551-558`). Nothing failed anywhere.
- The harness itself documents that this metric cannot express non-admission:
  "`max_completed_wave` reads as completion, not as fit: vLLM preempts rather
  than rejecting, so a wave whose KV footprint exceeds capacity still finishes
  and still reports a wave number" (`offload_benchmark.py:522-525`).

So the honest capacity-frontier result is: **all-cuda admits 131,072-token
prompts, but its 387,760-token KV cache confines it to a 4-deep wave there,
versus 8 for `subset:8:8` and 12 for the three larger-capacity placements — a
3× concurrency ceiling difference at long context, and 4× at 32,768 (12 vs 48).**
That is the capacity win, stated in the units the artifact actually supports.

Where exhaustion shows up instead is throughput and tail latency, exactly as the
docstring predicts. all-cuda at 131,072 (`all-cuda.json .capacity_frontier[3].points`):
concurrency 1 → 0.910 tok/s aggregate with TTFT p50 8,779 ms; concurrency 4 →
18.855 tok/s, TTFT p50 25,954 ms, ITL p99 288.8 ms. At 32,768 the ITL p99 sits at
92–102 ms for every wave ≥4 while ITL p50 stays at 5.4–7.5 ms — the p99/p50
spread is the preemption, not a per-token slowdown.

Also note the frontier and context sweep were driven at a **actual** 100,793
prompt tokens for the 131,072 target (`.capacity_frontier[3].points[*].actual_prompt_tokens`,
identical in all five files). The nominal 131k context was never exercised at
131k in Track B; Track A did reach it (§3, 127,010 mean prompt tokens).

---

## 3. Track A long-context cells (65536 / 98304 / 127000)

Read from `benchmarks/matrix/hybrid_131k_c6/unsloth__Qwen3.6-35B-A3B-NVFP4/ctx_<N>/<profile>/report.json`,
key path `.benchmarks[i].metrics.*.successful` (percentiles under
`.percentiles.p50` / `.p99`) and `.metrics.request_totals`. All cells ran 512
output tokens (`.metrics.output_token_count.successful.mean = 512.0`) under
`max_requests=20`, `max_errors=5`, and `max_duration=720 s`
(`ctx_127000/synchronous/guidellm_command.json`; note this is 720 s, not the
`max_seconds: 180.0` recorded in `matrix_config.json` — the runner scales the
cap with context length, and the short-context cells did run at 180 s).

### 3a. Concurrency-1 latency floor (`synchronous` profile)

| ctx dir | prompt tok (mean) | successful / incomplete | ITL p50 ms | ITL p99 ms | TTFT p50 ms | per-stream decode tok/s (1000/ITL p50) | `output_tokens_per_second` p50 |
|---|---|---|---|---|---|---|---|
| ctx_65536 | 65,545.95 | 20 / 0 | 6.4269 | 6.6817 | 32,263.8 | 155.60 | 0.0620 ⚠ |
| ctx_98304 | 98,313.93 | 14 / 1 | 6.1413 | 6.1491 | 45,622.9 | 162.83 | 0.0439 ⚠ |
| ctx_127000 | 127,009.91 | 11 / 1 | 6.3016 | 6.3079 | 61,090.5 | 158.69 | 0.0328 ⚠ |

### 3b. `concurrent` profile, N = 1…6 (all three cells)

| ctx | N | successful / incomplete | ITL p50 ms | ITL p99 ms | TTFT p50 ms |
|---|---|---|---|---|---|
| 65536 | 1 | 20 / 0 | 5.945 | 6.403 | 29,375 |
| 65536 | 2 | 20 / 0 | 61.722 | 61.820 | 39,396 |
| 65536 | 3 | 20 / 0 | 106.866 | 120.205 | 37,592 |
| 65536 | 4 | 20 / 0 | 163.560 | 177.413 | 37,911 |
| 65536 | 5 | 20 / 0 | 222.826 | 236.765 | 37,932 |
| 65536 | 6 | 20 / 0 | 284.057 | 294.289 | 36,027 |
| 98304 | 1 | 14 / 1 | 6.142 | 6.147 | 45,631 |
| 98304 | 2 | 14 / 2 | 93.857 | 93.901 | 61,481 |
| 98304 | 3 | 14 / 2 | 161.581 | 184.015 | 58,634 |
| 98304 | 4 | 12 / 4 | 254.544 | 273.288 | 57,908 |
| 98304 | 5 | 12 / 4 | 345.246 | 364.204 | 56,925 |
| 98304 | 6 | 11 / 5 | 438.221 | 453.122 | 54,920 |
| 127000 | 1 | 11 / 1 | 6.302 | 6.305 | 61,127 |
| 127000 | 2 | 10 / 2 | 85.740 | 124.234 | 62,625 |
| 127000 | 3 | 9 / 3 | 214.062 | 244.510 | 78,232 |
| 127000 | 4 | 8 / 4 | 337.248 | 363.596 | 76,211 |
| 127000 | 5 | 8 / 4 | 459.652 | 484.735 | 76,390 |
| 127000 | 6 | 6 / 6 | 585.135 | 602.738 | 184,796 |

ITL grows almost exactly linearly in N (65536: 5.9 → 61.7 → 106.9 → 163.6 →
222.8 → 284.1 ms, i.e. ≈ +55 ms per added stream). That is the single-SYCL-queue
serialization described in `benchmarks/README.md:117-119` and `:191-200`,
measured on the live server rather than in-process.

The `sweep` profile in each of these three cells contains three benchmarks —
`synchronous` (N=1), `throughput` (max_concurrency 512), and `constant`
(max_concurrency 512) — per `.benchmarks[i].config.strategy.type_`. The
`throughput` arm is where the tail blows out: ITL p50/p99 = 574.3/633.4 ms at
65536, 629.5/631.3 ms at 98304, 602.4/604.2 ms at 127000, with TTFT p50 up to
322,574 ms (65536 `sweep` benchmark index 1).

### 3c. Does decode throughput hold flat across prompt length? **No.**

First, the claim's actual location: it is in the repo-root `README.md:153-156`,
not in `benchmarks/README.md` (grep for `flat` across `benchmarks/` returns only
`benchmarks/README.md:118` about flatness *in batch size* and
`benchmarks/stream_matrix.py:10` about streaming cost). The root README says:
"Decode throughput is flat across prompt length (`benchmarks/results/smoke2d`:
75/80/76% of baseline at 363/1543/3123 prompt tokens)". That evidence spans
363 → 3,123 prompt tokens — three points inside the first 3k of context.

Extending the same measurement across Track A's full context range (synchronous
profile, `1000 / .metrics.inter_token_latency_ms.successful.percentiles.p50`):

| ctx dir | prompt tok (mean) | ITL p50 ms | per-stream decode tok/s | % of ctx_1024 |
|---|---|---|---|---|
| ctx_1024 | 1,033.95 | 5.5804 | 179.20 | 100.0% |
| ctx_4096 | 4,105.75 | 5.5182 | 181.22 | 101.1% |
| ctx_8192 | 8,202.00 | 5.5177 | 181.24 | 101.1% |
| ctx_16384 | 16,394.00 | 5.6133 | 178.15 | 99.4% |
| ctx_32768 | 32,777.95 | 5.7460 | 174.03 | 97.1% |
| ctx_65536 | 65,545.95 | 6.4269 | 155.60 | 86.8% |
| ctx_98304 | 98,313.93 | 6.1413 | 162.83 | 90.9% |
| ctx_127000 | 127,009.91 | 6.3016 | 158.69 | 88.6% |

Verdict: **the claim holds up to ~32k and breaks beyond it.** Within
1,024 → 8,192 tokens decode is flat to within +1.1%, and the README's own
evidence window (≤3,123 tokens) is inside that band, so the claim is not wrong
about what it measured. But from 32,768 tokens on, per-stream decode loses
9–13%: −2.9% at 32k, then −13.2% / −9.1% / −11.4% at 65,536 / 98,304 / 127,000.
The decay is not monotone (98,304 is faster than 65,536), so the size of the
loss is noisy, but every long-context cell sits 9–13% below the short-context
plateau — outside any plausible reading of "flat".

Track B's own all-cuda arm shows the same shape independently and without the
B70 in the path: `all-cuda.json .context_sweep[*].decode_tok_per_s` = 251.8 /
250.1 / 240.0 / 216.1 tok/s at 1,543 / 6,273 / 25,173 / 100,793 prompt tokens —
a 14.2% loss from short to long. Since all-cuda dispatches nothing to the B70,
this decay **cannot** be attributed to B70 dispatch, and it is also not explained
by the README's mechanism ("B70 dispatch is a fixed per-layer cost that does not
grow with sequence length" — `README.md:154-155`). The mechanism claim is intact;
the flatness conclusion drawn from it over-generalizes, because attention itself
grows with sequence length.

The other four Track B arms cannot be used for this comparison — see finding
(4c).

---

## 4. Errors, anomalies, and their evidence

No hard failure anywhere in the chain. `Track B exited rc=0` (`chain.log:12`);
every Track A `run_manifest.json` carries `"return_code": 0`; every Track A cell
reports `.metrics.request_totals.errored == 0` and
`.scheduler_state.errored_requests == 0` across all 80 benchmark entries in the
24 cells; every hybrid Track B arm reports
`.workers_decode_only[0].poller.errors == 0` and `.workers[0].poller.errors == 0`.
Every Track A `guidellm_stderr.log` is 138 bytes containing only the HF
unauthenticated-requests warning, and every `guidellm_stdout.log` is 0 bytes
(the runs used `--disable-console`).

Four real findings, in descending order of how much they can mislead a reader:

### 4a. The capacity-frontier console output is wrong for every placement — harness printer bug

`track_b.log` reports, five times over (lines 428-432, 587-591, 703-707,
863-867, 1022-1026):

```
capacity frontier (largest concurrent wave that completed):
   2048 tok prompt: none
   8192 tok prompt: none
  32768 tok prompt: none
  131072 tok prompt: none
```

This contradicts the JSON, which has `max_completed_wave` of 64/48/12/4 for
all-cuda and up to 64/64/48/12 for the hybrids (§2). The log is wrong, not the
JSON. Cause: the printer reads `row.get("concurrent_requests", 0)` from the
frontier row (`benchmarks/offload_benchmark.py:735`), but the frontier row's
keys are `target_prompt_tokens`, `max_completed_wave`, and `points`
(`offload_benchmark.py:571-575`) — `concurrent_requests` exists only inside
`points[*]` (`:563`). `n` is therefore always `0`, so the `else` branch prints
`none` unconditionally (`:744-747`). The same bug makes lines 738-742 dead code,
which is why no frontier throughput or ITL is in the log at all. Cosmetic:
`.capacity_frontier` in the JSON is complete and self-consistent, and no
measurement was lost. Anyone who read only the log would conclude nothing was
admitted anywhere.

### 4b. ctx_98304 and ctx_127000 were cut off by the 720 s duration cap

These are the only Track A cells that did not complete their 20 requests. Errors
are zero; the shortfall is truncation and cancellation.

| cell | benchmark | successful | incomplete | cancelled (`.scheduler_state.cancelled_requests`) | duration s |
|---|---|---|---|---|---|
| ctx_98304 / synchronous | N=1 | 14 | 1 | 6 | 720.000 |
| ctx_98304 / concurrent | N=6 | 11 | 5 | 9 | 720.000 |
| ctx_98304 / sweep | throughput | 8 | 12 | 12 | 720.000 |
| ctx_127000 / synchronous | N=1 | 11 | 1 | 6 | 720.000 |
| ctx_127000 / concurrent | N=6 | 6 | 6 | 14 | 720.000 |
| ctx_127000 / sweep | throughput | 6 | 14 | 14 | 720.000 |

All 20 such entries hit `duration = 720.000000` exactly, against
`max_duration=720` in the guidellm command — the wall clock stopped them.
Compare `ctx_65536/synchronous`, which finished 20/20 in 704.019 s: it cleared
the cap with 16 s to spare, so 65,536 is the largest context in this matrix that
completes its full request budget. `ctx_98304/sweep` benchmark 2 (`constant`) is
the extreme case: 8 successful, 0 incomplete, 12 cancelled — the offered rate was
never reached before the cap.

Consequence: the ≥98,304 rows in §3 are 11–14-request samples, not 20. The ITL
percentiles are still meaningful (they are per-token, thousands of samples), but
request-level statistics from those cells are thin and should not be compared
against short-context cells as equals.

### 4c. Track B's `context_sweep` produced 8-token decodes on 12 of 20 (placement × context) points, corrupting the derived tok/s

`.context_sweep[*]` requested 400 decode tokens (`--decode-tokens 400`,
`run_offload_sweep.sh:61`; confirmed by `.single_stream.output_tokens = 400`).
Actual `output_tokens` returned, with the resulting `decode_tok_per_s`:

| placement | 1,543 tok | 6,273 tok | 25,173 tok | 100,793 tok |
|---|---|---|---|---|
| all-cuda | 275 → 251.8 | 265 → 250.1 | 254 → 240.0 | 362 → 216.1 |
| subset:8:8 | 246 → 214.1 | **8** → 229.0 | 262 → 206.1 | **8** → 387.3 |
| subset:16:8 | 278 → 187.4 | **8** → 197.5 | **8** → 210.5 | **8** → 300.1 |
| subset:24:64 | 400 → 170.7 | 264 → 170.0 | **8** → 188.1 | **8** → 260.3 |
| split:128 | **8** → 161.3 | 309 → 158.9 | **8** → 178.5 | **8** → 244.7 |

The value is exactly 8 every time it is short, never some other small number —
so this is a floor being hit, not gradual early termination. The derived rates
from those points are arithmetically meaningless: `subset:8:8` reports **387.3
tok/s at 100,793 prompt tokens**, i.e. 1.8× its own measured single-stream rate
of 214.7 tok/s and 1.53× the all-cuda baseline, which is impossible for an arm
that routes 96% of experts over PCIe. The same artifact is visible in the log
(`track_b.log:586`, `:702`, `:862`, `:1021`) and produces the false impression
that every hybrid placement gets *faster* with longer prompts.

This is why §3c uses Track A ITL and Track B's all-cuda arm — the only
context-sweep rows with credible token counts — and not the hybrid arms.
`.capacity_frontier[*].points[*].total_output_tokens` shows the same floor at
concurrency 1 (8 tokens for a 1-request wave, e.g. `all-cuda.json`
`.capacity_frontier[3].points[0]`), so the frontier's concurrency-1 aggregate
`output_tok_per_s` values (0.910 tok/s at 131,072, etc.) are subject to the same
caveat; the ITL p50/p99 at those points is unaffected.

### 4d. Non-blocking warnings, recorded for completeness

- `agent_report.log` is 0 bytes — the report agent spawned at `chain.log:13` had
  not written anything at the time this report was produced.
- `matrix_config.json` records `"contexts": [127000]` and `"max_seconds": 180.0`,
  neither of which describes the matrix as a whole: eight context dirs exist, and
  the long-context cells ran at 720 s (§3). The file reflects the last resumed
  invocation under `"skip_existing": true`, not the union of the 24 cells. Do not
  cite it as the run configuration.
- `track_b.log:145` / `:482` — `Using default MoE config. Performance might be
  sub-optimal!` (no tuned `E=256,N=512,device_name=NVIDIA_GeForce_RTX_5090,dtype=fp8_w8a8.json`).
  Applies to all five arms equally, so it does not bias the comparison.
- `track_b.log:402-403` / `:561-562` — `Triton kernel JIT compilation during
  inference: _compute_slot_mapping_kernel` and `fused_moe_kernel`, flagged by
  vLLM as a latency spike. Both fire during the first requests of each arm,
  before the 3-trial single-stream measurement, and all five arms show the same
  pair.
- `track_b.log:397` / `:400` — `VLLM_ALLOW_INSECURE_SERIALIZATION=1` (pickle),
  and `:401` a deprecation warning about raw prompts to `InputProcessor`.
  Neither affects the measurements.
- GPU power headroom was never the limit: across the five Track B arms
  `.workers[0].cuda_power` reports 237.4–369.7 W drawn against
  `limit_watts = 575.0` at 41–57 °C (and 224.0–302.8 W at 40–45 °C in the
  decode-only snapshot `.workers_decode_only[0].cuda_power`). The cap was left
  in place and untouched.

---

## Summary

1. Capacity/latency tradeoff is monotone and the concentration effect
   reproduces: `subset:16:8` at 74.5% of all-cuda decode for 2.91× KV;
   `subset:24:64` gives the most KV (3.19×) at 67.9%; `split:128` is dominated —
   3% less KV than `subset:24:64` for 7% less throughput, because 32 active
   layers dispatch more than 24.
2. No context is un-admittable by all-cuda. Its penalty is a 3–4× lower
   concurrency ceiling at long context (4 vs 12 waves at 131k; 12 vs 48 at 32k),
   which is the capacity win restated in the units the artifact supports.
3. "Decode throughput is flat across prompt length" holds only to ~32k. Beyond
   that it drops 9–13% (Track A ITL), and all-cuda drops 14% in Track B — so the
   decay is not a B70 dispatch effect and the root-README generalization from a
   ≤3,123-token sample should be narrowed.
4. Zero errors chain-wide, but three artifacts need caveats before use: the
   frontier console output is uniformly wrong (printer bug reading a key that
   isn't there), the ≥98,304 Track A cells are 11–14-request samples truncated by
   the 720 s cap, and 12 of 20 Track B context-sweep points decoded 8 tokens
   instead of 400 and produced unusable tok/s figures.
