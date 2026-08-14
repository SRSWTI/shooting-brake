# Long-context prefill: dispatch vs streaming — 2026-08-13

One machine (X870E / 9950X3D / RTX 5090 / Arc Pro B70), one code state, one
script, same day. Both arms completed **10/10 requests with zero truncation**,
so unlike `benchmarks/matrix/hybrid_131k_c6` these cells are directly comparable.

## Configuration

Identical except for one environment variable:

```bash
# arm A - dispatch (current default)
bash benchmarks/serve_hybrid.sh

# arm B - streaming
SHOOTING_BRAKE_B70_PREFILL_STREAM=1 bash benchmarks/serve_hybrid.sh

# both arms, identical sweep
CONTEXTS=65536,98304,127000 PROFILES=synchronous \
MAX_REQUESTS=10 MAX_SECONDS=600 OUTPUT_ROOT=$PWD/bench-matrix/longctx/<arm> \
  bash benchmarks/run_matrix.sh
```

`subset:16:8`, `max_model_len=131072`, 512 output tokens, single stream.
Artifacts: `benchmarks/matrix/longctx/{dispatch,stream}/`.

## Result

| ctx | TTFT dispatch | TTFT stream | speedup | ITL disp | ITL stream | e2e disp | e2e stream | gain |
|---|---|---|---|---|---|---|---|---|
| 65,536 | 30.98 s | 11.21 s | 2.76x | 6.20 ms | 6.14 ms | 16.5 tok/s | 38.8 tok/s | 2.35x |
| 98,304 | 47.98 s | 18.81 s | 2.55x | 6.38 ms | 6.33 ms | 11.0 tok/s | 25.4 tok/s | 2.30x |
| 127,000 | 63.62 s | 26.66 s | 2.39x | 6.49 ms | 6.50 ms | 8.5 tok/s | 18.7 tok/s | 2.21x |

Prefill rate: dispatch 2,116 -> 1,996 tok/s; streaming 5,848 -> 4,764 tok/s.

**Decode is unaffected** (ITL within 1% at every length), which is the intended
behaviour: the flag only changes the prefill path. **KV capacity is unaffected**
(842,038 vs 840,052 tokens) - the streaming mirror lives in host DRAM, not in
5090 VRAM, so it does not eat the capacity the B70 exists to provide.

Cost: ~4-6 GiB of host DRAM for the expert mirror (measured as resident-set
growth; 43 GiB was free).

## Two predictions that failed

Recorded because they were wrong in ways that teach something:

1. **Predicted TTFT ~5-6 s at 127K; measured 26.7 s.** The 6.6 GiB weight
   transfer is not the dominant cost. Streaming prefill runs at 4,764-5,848
   tok/s against all-CUDA's ~25,500, so the streamed-weight *compute* path on
   the 5090 is the remaining bottleneck - not PCIe, and not the B70.
2. **Predicted the advantage would widen with context; it narrows**
   (2.76x -> 2.39x). Streaming prefill decays 5,848 -> 4,764 tok/s across the
   range, which a flat transfer plus linear compute cannot produce. Most likely
   attention's O(N^2) term becoming visible once the dispatch bottleneck is
   removed; all-CUDA prefill decays with length in the same way.

## Also settled: there is no "65K cliff"

The dispatch arm reproduces the old matrix almost exactly (16.5 vs 16.5, 11.0
vs 11.3, 8.5 vs 8.7 tok/s) and shows a **smooth decay with flat ~2,000 tok/s
prefill** - no discontinuity. The apparent cliff in
`benchmarks/matrix/hybrid_131k_c6` sat on a seam: contexts 1K-32K were measured
2026-08-06 on the previous machine *before* commit `1fbecdc0` ("fix silent
prefill route loss"), when B70-owned routes were dropped during prefill and it
therefore ran at all-CUDA speed (20,320 tok/s at 32K) while producing subtly
wrong logprobs. Contexts 65K+ were measured 2026-08-12/13 after the fix.
`--skip-existing` welded the two into one table.

Those 1K-32K cells should be re-run before being quoted.

## Recommendation

Enable `SHOOTING_BRAKE_B70_PREFILL_STREAM=1` for long-context serving: 2.2x
end-to-end at the 131K thesis, decode and KV capacity untouched, no code
change. `benchmarks/serve_hybrid.sh` does not currently set it.

The remaining prefill gap is a 5090-side streamed-weight compute problem, which
is a different target from the B70 kernel work.
