# Shooting Brake

![Shooting Brake](assets/Indecent-Porsche-911-Shooting-Brake-1.webp)

> A shooting brake was never for everyone - it's the rare machine that refuses to
> sacrifice speed for capacity, built in limited numbers for people who wanted both.

One 118B MoE model — NVFP4, 58.2 GiB of weights, 131,072 context — split across two GPU
vendors inside every forward pass. An **RTX 5090** runs attention, the router, and the
local and shared experts. **Two Intel Arc Pro B70s** hold and compute 170 routed experts,
85 per card, reached over a pinned-host-memory doorbell: 47 host-visible sync points per
token, no shared runtime between the vendors.

```bash
./serve_production.sh          # OpenAI-compatible server on :8017
```

## Against one RTX PRO 6000 (96 GB), same checkpoint, same harness

Single stream, guidellm, 512 output tokens, prefix caching off on both sides.

| context | prefill, us/token | decode ITL, ms |
|---|---|---|
| 8K | 406 vs 50.5 — **8.0x behind** | 11.40 vs 7.25 |
| 32K | 421 vs 71.2 — **5.9x behind** | 11.40 vs 8.03 |
| 64K | 446 vs 95.4 — **4.7x behind** | 11.40 vs 9.05 |
| 127K | 494 vs 145.5 — **3.4x behind** | 11.40 vs 11.10 — **parity, 1.03x** |

The reference's prefill is attention-bound and superlinear — 50.5 to 145.5
us/token from 8K to 127K. Ours is transport-bound and nearly flat — 406 to 494
us/token. So the gap narrows from 8.0x to 3.4x as context grows. Decode has the
same shape: the reference's ITL grows with the KV cache, ours is
context-independent, and they meet at 127K.

All of the above is **cold, with prefix caching off on both sides** — the
reference run used `--no-enable-prefix-caching` so the comparison isolates
compute. With prefix caching on, a repeat prefix here lands at:

| context | cold TTFT | warm TTFT | speedup |
|---|---|---|---|
| 8K | 3.37 s | **0.027 s** | 126x |
| 32K | 14.08 s | **0.104 s** | 135x |
| 64K | 30.04 s | **0.100 s** | 300x |
| 127K | 65.29 s | **0.247 s** | 265x |

Warm TTFT is 27-247 ms across the whole 1K-127K range. On traffic where
prefixes repeat, the prefill gap is not perceptible.

Numbers: [`benchmarks/results/rtx_pro_6000_r15_slo/DERIVED_SUMMARY.json`](benchmarks/results/rtx_pro_6000_r15_slo/DERIVED_SUMMARY.json).
Method, and every killed idea: [`docs/campaign-scorecard.md`](docs/campaign-scorecard.md).
