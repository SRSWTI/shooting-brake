# run4: pre-repacked Marlin bank, production config — measured

Server: `serve_88b_128k.sh` defaults (marlin prefill ON via SBMARL01 bank,
ZE_AFFINITY_MASK=1 Gen4 card, MNBT=8192, MNS=16). KV 190,990 tokens,
attention block 4,176 (=> 45 short seats / ~1.4 seats at 131K). Harness:
GuideLLM, synthetic_text, ignore_eos, 512-token outputs, per-cell seeds.
Full grid: `matrix/MEASURED.md` (21/21 cells, 0 errors).

## Headline vs run2 baseline (B70-dispatch prefill, Gen3 card) and PRO 6000

| metric | run2 | run4 | PRO 6000 | gap now (was) |
|---|---|---|---|---|
| decode C=1 (140 in / 512 out) | 57.5 tok/s | **78.3** | 135 | 1.72x (2.35x) |
| peak aggregate | 173.5 @ C=16 | **235.2 @ C=8** | ~200 @ C=2 | above their C=2 (was below) |
| ITL, C<=4 | 16.5 ms | **12.2-12.9 ms** | 7.34 ms | 1.67x |
| TTFT @ 8K, C=1 | 21.86 s | **2.19 s** | 0.56 s | 3.9x (38.9x) |
| TTFT @ 32K | 84.2 s | **9.85 s** | 2.78 s | 3.5x (30.3x) |
| TTFT @ 130K | ~335 s | **55.1 s** | 39.1 s @ 127K | **1.4x** |

## Findings

1. **Long context is where this topology competes.** 128K prefill within
   1.4x of the PRO 6000 (system throughput 3,774 tok/s during a 130K
   prefill); their ms/token curve climbs past 64K (attention becoming
   visible), ours stays buried under a constant, so the gap narrows with
   context.
2. **Decode ceiling ~225-235 aggregate; knee moved 16 -> 8.** C=8/16/32/62
   and the unbounded-flood cell (222.7) all land at the same plateau: 48
   serial B70 dispatches per token step bound aggregate decode regardless of
   batch rows. Faster per-token decode (Gen4) saturates the same serial
   floor earlier. This is the top optimization target; the decode-step trace
   decides between graph replay / dual-B70 / persistent kernel.
3. **Marlin prefill floor ~1.1 s** (ctx_512 TTFT 1.13 s): the streamer moves
   all 27.4 GiB per prefill regardless of prompt size. Still beats dispatch
   at every measured context >= 512. Pinned/threaded staging (52.8 GiB/s DMA
   ceiling vs ~18.5 pageable) roughly halves it.
4. Thinking on/off identical (81.4 vs 81.1 tok/s C=1) — reasoning tokens do
   not change routing cost measurably. Replicates run2.
5. C_longctx at 128,928 tokens: only C=1 truly runs (1.4 seats); C=2/3 rungs
   are admission queueing (TTFT 100-110 s = queue time), flagged
   capacity-bound with the corrected KV constants.

## Bookkeeping corrected this run

- Summary header/capacity flags previously quoted a stale KV=262,144 / 62
  seats; bench_88b.py now takes --kv-tokens/--max-num-seqs from THIS
  server's log (190,990 / 16).
- `D_saturation/throughput_ctx128` failed once: this guidellm requires
  `profile.throughput.max_concurrency`; the builder now passes 2x the
  admission cap. Cell reran clean (78/0, 222.7 tok/s).
