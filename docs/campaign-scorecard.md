# Shooting Brake — Campaign Scorecard

Every number here is measured on this rig, with the artifact that produced it
named. No projections in the tables; projections are labelled inline and kept
out of the headline rows.

**Rig:** 1x RTX 5090 (32 GB, SM120) + 2x Intel Arc Pro B70 (32 GB each, Xe2/BMG),
one box, 59.4 GB host RAM, both B70s on one shared PCIe Gen4 x4 uplink.
**Model:** `srswti/axe-superveloce-jota-118b-r15-nvfp4` — 118B MoE, 15% REAP-pruned,
NVFP4 (E2M1 weights + E4M3 block scales), 58.2 GiB on disk, 47 MoE layers +
dense layer 0, 218 experts/layer, top_k=10, hidden 3072, hybrid sliding/full
attention.
**Split:** 5090 runs attention/GDN/router + local & shared experts (vLLM in-tree
CUTLASS W4A4); 2x B70 run 170 routed NVFP4 experts (85/card) reached by a
custom doorbell protocol, 47 host-visible sync points per token.

## 1. Prefill — the campaign's main lever (32K band, per token)

| stage | us/token | cumulative | artifact |
|---|---:|---:|---|
| per-route GEMV (start) | 1,705 | 1.00x | `docs/kill-bench-level-up.md` Bench 23-25 |
| grouped NVFP4 GEMM | 526 | **3.24x** | Bench 23 |
| + fp16 result wire | 430 | **3.97x** | Bench 24-25 |
| + prefetch_dist=1 | 404 | **4.22x** | `9b41ba5b` |
| + 2-chunk pipelining (shipped) | **387** | **4.40x** | Phase 4, `0b1cec38` |

Wall clock: a **32K prompt went 55.9 s -> 12.7 s**.

## 2. Cold TTFT — final lever only (paired A/B, 3 repeats/arm, quality gates green)

| ctx | baseline | pipelined | delta |
|---|---:|---:|---:|
| 1K | 0.439 s | 0.445 s | wash |
| 8K | 3.281 s | 3.374 s | wash (below tile crossover) |
| 16K | 6.513 s | **6.122 s** | -0.39 s (1.064x) |
| 32K | 13.449 s | **12.696 s** | -0.75 s (1.059x) |
| 64K | 28.689 s | **27.045 s** | -1.64 s (1.061x) |
| 96K | 45.755 s | **43.652 s** | -2.10 s (1.048x) |
| 127K | 62.432 s | **59.449 s** | -2.98 s (1.050x) |

Artifact: `benchmarks/results/b70_gemv_audit/p4_smoke_{baseline,pipeline2}.json`.
Trace cross-check: per-dispatch service 13.04 -> 12.16 ms (dev0), 15.16 -> 14.06
(dev1), **bytes/dispatch constant at 25.33 MB** — the overlap is physical, not a
measurement artifact.

## 3. Serving SLO grid — 36/36 comparable rows favour the shipped arm, 0 errors

| ctx | C | ITL | aggregate tok/s |
|---|---:|---|---|
| 8K | 1 | 14.35 -> **12.14 ms** | 47.4 -> **55.8** (+18%) |
| 8K | 4 | 37.5 -> **32.5 ms** | 85.0 -> **94.5** (+11%) |
| 4K | 6 | 44.6 -> **43.7 ms** | 116 -> **126** (+9%) |
| 32K | 4 | 100.0 -> **93.9 ms** | 31.1 -> **33.2** (+7%) |
| 64K | 1 | 14.78 -> **14.20 ms** | 13.8 -> **15.1** (+9%) |
| 1K | 4 | 23.4 -> **22.5 ms** | 153 -> **158** (+4%) |

Artifacts: `bench-matrix/jota_r15_c6` (baseline, 24 cells) vs
`bench-matrix/jota_r15_pipeline2` (shipped, 16 cells), 20/20 successful
requests per cell. Production-KV bridge (`bench-matrix/jota_r15_prod_kv`)
confirms these stay quotable at C=1..4 (kill-bench 26).

## 4. Capacity (changed 2026-08-24)

| | before | after |
|---|---:|---:|
| KV pool | 8.29 GiB | **10.69 GiB** (+29%) |
| KV tokens | ~163K | **211,083** |
| max concurrency @131K ctx | 1.25x | **1.61x** |
| context served | 131,072 | 131,072 (the max that fits) |

Prefix caching on, measured **~139x** on repeated prompts — the dominant
real-world effect for agentic loops, where turn N resends turn N-1's context.

## 5. Versus RTX PRO 6000 Blackwell (96 GB, single card, their published run)

| | before | after |
|---|---:|---:|
| prefill gap @8K | ~8.0x behind | ~8.0x (wash at this size) |
| prefill gap @32K | 5.75x behind | **5.43x** |
| prefill gap @127K | 3.30x behind | **3.14x** |
| decode @127K | 1.03x (**parity**) | 1.03x (**parity**) |

Their prefill is attention-bound and superlinear; ours is transport-bound and
flat — so **the gap closes as context grows**. Their decode degrades with KV
depth; ours does not move, because our bottleneck (fixed per-layer dispatch)
does not know the context length. Artifact:
`benchmarks/results/rtx_pro_6000_r15_slo/` (their CSVs + our derived summary).

## 6. What did NOT improve — same standard of evidence

- **Decode: 12.1 -> 12.5 ms, unmoved.** Every shipped lever targeted prefill.
  Decomposition: 3.07 ms B70 service (21.6%) + ~9.9 ms 5090-side inter-dispatch
  gap (69.7%) + ~2 ms API/sampler tail.
- **1K/8K prefill: wash.** Below the tile crossover (chunking drops
  rows-per-expert under the big-tile threshold).
- **More KV bought capacity, not throughput.** At 32K, C>=5 converts queueing
  into batching: per-request ITL +34..60%, aggregate tok/s flat.
- **Killed with numbers:** FP8 wire (11.5x worse error than the bf16 leg for a
  ~4% gain, reverted `819c82e0`); W4A4 4-bit activations (0.82 cosine,
  terminal); B12x kernel (2.6x faster, quality gate failed); Doorbell 2.0
  live-append (80-95 us/bracket vs the 61 us it replaces); baked chains
  (numerically correct at 9.4e-4 but wedges vLLM graph capture, parked);
  ngram spec-decode (8.6 ms short ctx, 24.8 ms at 107K — flag only);
  DFlash drafter (0 of 7,665 tokens accepted — trained on the unpruned target).

## 7. One-line summary

Prefill **4.40x faster**, serving throughput **+3..18%** across the grid,
capacity **+29%**, decode **unmoved**, and the gap to an 8k-USD datacenter card
closed from 3.30x to **3.14x** on prefill while holding **parity on decode** at
127K.
