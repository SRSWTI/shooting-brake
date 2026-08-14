# Measured: 88B across RTX 5090 + one Arc Pro B70

The numbers this checkpoint saves. Everything here was measured on
2026-08-14, not projected. Where a figure is derived rather than observed it
says so.

Reproduce with:

```bash
bash benchmarks/serve_88b.sh          # shell 1, ~215 s to ready
.venv/bin/python benchmarks/sb_chat.py # shell 2
```

## Throughput

Concurrency 1. Client-side, from the server's own `usage.completion_tokens`
via `stream_options.include_usage` — never by counting SSE frames, since a
frame is not a token.

| probe | completion tokens | wall | tok/s |
|---|---|---|---|
| short question | 160 | 2.49 s | **64.3** |
| follow-up, multi-turn | 220 | 3.66 s | **60.2** |
| longer generation | 300 | 4.58 s | **65.5** |
| two-turn total | 380 | 6.15 s | 61.8 |

**60–65 tok/s** is the honest range at concurrency 1 with short prompts.

Not recorded for these three probes: they were taken non-streaming, so no
TTFT or ITL was captured. `benchmarks/sb_chat.py` does measure
client-observed TTFT at the first non-empty SSE delta, and reports
server-side per-token latency only when vLLM's
`RequestStateStats.first_token_latency` is present, labelling it
`unavailable` otherwise. Also not measured: any concurrency above 1, and any
context beyond a few hundred prompt tokens.

## Reference point

An RTX PRO 6000 Blackwell, same checkpoint, same vLLM 0.27.1, measures
**135 tok/s** at concurrency 1 and ~200 at 2 concurrent.

On **capacity** it is a like-for-like target: 96 GB against a 5090 plus two
B70s. On **bandwidth** it is not. Theirs is 1,792 GB/s uniform over one
address space; ours is 1,792 + 608 + 608 across three devices, which is
neither additive for a single dependent computation nor uniform, so
aggregate GB/s is not a fair comparison and no claim here rests on it.

We are at roughly 46% of their concurrency-1 figure with **one** B70 active,
before the next planned optimisations. Note this is already an optimised
configuration — CUDA graphs, load-time expert compaction, per-device
quantization and a purpose-written int4 kernel are all in play.

Its own log is worth keeping in mind: `sm120` has no native FP4 compute, so
it runs NVFP4 as weight-only compression through Marlin. That applies
identically to our 5090.

## Configuration

| tier | contents | size |
|---|---|---|
| 5090 (31.8 GiB) | dense NVFP4 + experts `0..53` NVFP4 + KV | 19.34 GiB weights |
| B70 `0000:11:00.0` | experts `54..179` int4, `src/phase1/expert_bank_int4.bin` | 28.47 GiB |
| second B70 | **unused** — needs device selection and a per-card poller | — |
| host DRAM | **no weights**; activations transit a pinned FP32 staging buffer | ~18 KiB/layer |

KV: 393,216 tokens, 12.00x concurrency at `--max-model-len 32768`. The
hybrid allocator forced the attention block to 4,176 tokens so the attention
page is at least the Mamba page, then padded the Mamba page to match — so
per-token KV cost is a function of block sizing and is **not** comparable
across different `--max-model-len` or `--max-num-seqs` settings.

`--max-num-batched-tokens 256` is deliberate: it bounds prefill chunk size
below the provider's `max_batch`, at the cost of more chunks on long
prompts. Decode rate is unaffected; TTFT grows with context.

## Evidence the B70 actually did the work

| check | result |
|---|---|
| compaction at load | **48/48 layers** via `ModelOptNvFp4FusedMoE` |
| CUDA allocation | 19.34 GiB against 48.91 for the full model |
| provider dispatches (health-gated, delta) | **6,144 = 48 layers x 128 tokens** |
| provider errors | **0** |
| M histogram | 6,096 at M=1, 48 at M=17–32 (prefill) — accounting closes exactly |
| plugin route counters | 6,135 steps with remote routes, 37,807 remote routes |
| all-CUDA-route steps | 9 |

The 9 all-CUDA steps are worth a note. Uniform independent routing predicts
`(54/180)^8 x 6144 = 0.40`. Measuring 9 is ~22x chance, which proves the
top-8 assignments are non-uniform and/or correlated with the specific
`0..53` set. It does **not** prove expert-ID locality and does **not** settle
two-card topology; `benchmarks/route_topology.py` plus a real trace does.

## Correctness

Verified:

- bank vs original safetensors: **bit-exact**, including all 24 tensors the
  chain oracle consumes (layer 0, compact ids 0–7 = source 54–61, all three
  projections)
- provider chain, bank → upload → int4 kernel → copyout, against an
  independent CPU dequant: **1.148e-06** (bound 5e-05)
- `-1` skip sentinel: an all-`-1` dispatch returns 3,072 exact zeros
- additive decomposition: `left(k) + right(8-k)` reconstructs a single
  8-route dispatch at **1.506e-07** for every k in 0..8
- resident-map rejection: exact / wrong-length / subset all rejected with
  both sets printed
- NVFP4 `SBEXP001` path unchanged, Phase-1 golden gate passes
- peak RSS loading a 27.41 GiB bank: **+19.5 MiB** over a bare SYCL queue

**Open gate, and it blocks further performance claims:** the live CUDA+B70
combine. Routing-weight scaling is shared — both paths hand the same kernel
compact ids and fp32 router weights, and `launch_down` applies
`router_weight * value` — so that half is covered by the provider oracle.
But acquisition and addition differ between the sync and graph paths,
leaving **freshness, completion-flag ordering, static-buffer selection and
the graph-path addition** unproven.

Provider accuracy plus healthy dispatch counts cannot catch a mis-scaled or
stale remote partial. The historical failure in this exact code path
produced fluent text with identical output tokens and ~0.49 nats/token worse
logprobs.

## Measured per-dispatch cost, for whoever optimises next

Standalone chain test, real geometry, `-1`-padded route counts:

| active routes k | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| device kernel µs | 8.1 | 30.8 | 32.2 | 33.6 | 32.6 | 37.1 | 56.6 | 72.3 | 79.9 |
| wall − kernel µs | 110 | 110 | 109 | 109 | 113 | 108 | 108 | 108 | 108 |

Two things follow. Kernel time is **not linear** in route count — k=4 is
faster than k=1 — so any cost model assuming per-route linearity is wrong.
And non-kernel cost is a **flat ~108 µs floor** independent of the work
done, which is seven queue submissions per dispatch (3 H2D, zero_output,
gate_up, down, 1 D2H).

SYCL graph replay of that same seven-command chain, measured with correctness
verified at 7.63e-08 against eager and 1.13e-06 against the CPU oracle:

| k | eager | graph replay |
|---|---|---|
| 1 | 105.1 µs | **75.3 µs** |
| 4 | 114.1 µs | **83.6 µs** |

That is ~28% off **provider-dispatch latency**, roughly 1.4 ms/token across
48 layers, so single-digit percent end to end. The k=8 pair from that run is
discarded: eager and graph shared one queue and the eager arm was inflated
by capture, giving 365 µs against the 188 µs measured standalone. An
isolated-queue rerun is queued.

## What is not known

No critical-path trace exists. Attempts to decompose the token into "B70
time" and "5090 time" by subtraction are invalid, because the graph path
writes the doorbell, runs the CUDA MoE partial for its 54 experts, and then
waits — those terms overlap. The correct form is:

```
token = sum over layers [ serial CUDA work
                          + max(B70 round trip, CUDA MoE partial)
                          + combine ]
        + lm_head + sampling
```

Getting the terms requires a profile-off timeline with per-phase CUDA
ranges, paired poller-side timestamps for signal-observed / kernel /
completion, and a dummy-provider A/B that returns zeros immediately to
isolate the CUDA half directly. Until that exists, every optimisation
estimate is a bound on a component, not a throughput prediction.
