# run6 vs RTX PRO 6000 -- TTFT mean (ours / PRO, ratio; lower is better)

Same checkpoint, same GuideLLM harness, concurrent profile means.
PRO source: bench-matrix/superveloce_88b_nvfp4a16_c6. C=10 is ours alone.

| ctx \ C | 1 | 2 | 3 | 4 | 5 | 6 | 10 |
|---|---|---|---|---|---|---|---|
| 1024 | 0.53s / 0.08s (6.38x) | 1.03s / 0.09s (11.11x) | 1.04s / 0.10s (10.89x) | 1.04s / 0.10s (10.62x) | 1.05s / 0.10s (10.65x) | 1.17s / 0.10s (11.26x) | 1.71s |
| 4096 | 0.54s / 0.27s (2.00x) | 1.07s / 0.29s (3.76x) | 1.29s / 0.29s (4.45x) | 1.58s / 0.32s (4.98x) | 1.94s / 0.29s (6.63x) | 2.52s / 0.30s (8.48x) | 3.46s |
| 8192 | 1.09s / 0.57s (1.91x) | 2.10s / 0.58s (3.60x) | 3.00s / 0.59s (5.05x) | 3.41s / 0.60s (5.68x) | 3.69s / 0.61s (6.10x) | 4.53s / 0.74s (6.09x) | 4.83s |
| 16384 | 2.28s / 1.22s (1.86x) | 4.49s / 2.09s (2.15x) | 5.50s / 1.27s (4.33x) | 6.31s / 1.28s (4.93x) | 6.91s / 1.29s (5.38x) | 6.60s / 1.30s (5.09x) | 8.28s |
| 32768 | 5.15s / 2.85s (1.81x) | 8.63s / 2.91s (2.96x) | 9.17s / 2.94s (3.12x) | 12.42s / 5.51s (2.26x) | 12.44s / 6.05s (2.06x) | 14.30s / 6.03s (2.37x) | 14.20s |
| 65536 | 12.69s / 7.50s (1.69x) | 19.77s / 18.76s (1.05x) | 25.77s / 24.27s (1.06x) | 25.73s / 25.78s (1.00x) **WIN** | 33.70s / 28.44s (1.18x) | 36.39s / 30.61s (1.19x) | 24.67s |
| 98304 | 23.18s / 26.12s (0.89x) **WIN** | 36.01s / 39.70s (0.91x) **WIN** | 50.11s / 43.99s (1.14x) | 55.74s / 47.05s (1.18x) | 57.06s / 52.24s (1.09x) | 91.62s / 54.50s (1.68x) | 52.54s |
| 127000 | 33.99s / 39.06s (0.87x) **WIN** | 67.54s / 58.70s (1.15x) | 87.67s / 66.34s (1.32x) | 87.57s / 71.75s (1.22x) | 94.41s / 81.19s (1.16x) | 94.47s / 97.39s (0.97x) **WIN** | 114.49s |

# Output tok/s mean (ours / PRO)

| ctx \ C | 1 | 2 | 3 | 4 | 5 | 6 | 10 |
|---|---|---|---|---|---|---|---|
| 1024 | 78 / 135 | 107 / 191 | 140 / 236 | 189 / 298 | 147 / 300 | 205 / 339 | 191 / - |
| 4096 | 78 / 128 | 106 / 178 | 113 / 217 | 147 / 271 | 128 / 273 | 159 / 304 | 173 / - |
| 8192 | 73 / 119 | 96 / 161 | 102 / 193 | 131 / 234 | 116 / 238 | 140 / 264 | 139 / - |
| 16384 | 64 / 103 | 79 / 134 | 89 / 156 | 111 / 182 | 99 / 186 | 107 / 198 | 113 / - |
| 32768 | 49 / 78 | 57 / 95 | 62 / 106 | 72 / 120 | 72 / 120 | 65 / 123 | 69 / - |
| 65536 | 32 / 46 | 34 / 34 | 29 / 32 | 29 / 33 | 36 / 33 | 37 / 33 | 37 / - |
| 98304 | 21 / 17 | 21 / 18 | 22 / 19 | 21 / 19 | 22 / 19 | 22 / 19 | 22 / - |
| 127000 | 16 / 12 | 15 / 13 | 15 / 12 | 15 / 12 | 15 / 11 | 15 / 10 | 15 / - |

# Every shared metric at C=1 (ours / PRO)

| ctx | TTFT s | ITL ms | TPOT ms | out tok/s |
|---|---|---|---|---|
| 1024 | 0.53 / 0.08 | 11.88 / 7.28 | 12.90 / 7.43 | 78 / 135 |
| 4096 | 0.54 / 0.27 | 11.92 / 7.30 | 12.96 / 7.82 | 78 / 128 |
| 8192 | 1.09 / 0.57 | 11.95 / 7.34 | 14.05 / 8.44 | 73 / 119 |
| 16384 | 2.28 / 1.22 | 12.06 / 7.46 | 16.48 / 9.83 | 64 / 103 |
| 32768 | 5.15 / 2.85 | 12.20 / 7.60 | 22.24 / 13.15 | 49 / 78 |
| 65536 | 12.69 / 7.50 | 12.37 / 7.85 | 37.13 / 22.49 | 32 / 46 |
| 98304 | 23.18 / 26.12 **W** | 13.49 / 9.26 | 58.73 / 60.25 **W** | 21 / 17 **W** |
| 127000 | 33.99 / 39.06 **W** | 12.86 / 9.52 | 79.22 / 85.79 **W** | 16 / 12 **W** |

# Decode-shaped grid (128-token prompt, 512 out) -- ours

The PRO matrix has no 128-token cell; its nearest is ctx_1024,
shown for the C rungs it covers (1-6). C>6 is ours alone.

| C | out tok/s ours | out tok/s PRO@1K | ITL ms ours | ITL ms PRO@1K |
|---|---|---|---|---|
| 1 | 81 | 135 | 11.88 | 7.28 |
| 2 | 116 | 191 | 15.33 | 9.60 |
| 4 | 189 | 298 | 19.93 | 11.28 |
| 8 | 252 | - | 26.60 | - |
| 16 | 190 | - | 41.69 | - |
| 32 | 241 | - | 46.44 | - |
| 62 | 271 | - | 45.45 | - |

# Peak throughput, unbounded offered load

PRO bracket from their sweep profile: 798 @1K, 613 @4K, 462 @8K
out tok/s. Our saturation cells sit at 128 and 2048 tokens, so the
2048 row interpolates against ~700 on their curve [INFERENCE].

| our cell | strategy | out tok/s |
|---|---|---|
| sweep_ctx2048 | synchronous | 77.6 |
| sweep_ctx2048 | throughput | 195.5 |
| sweep_ctx2048 | constant | 84.0 |
| sweep_ctx2048 | constant | 91.1 |
| sweep_ctx2048 | constant | 98.2 |
| sweep_ctx2048 | constant | 105.3 |
| throughput_ctx128 | throughput | 270.0 |
| PRO ctx_1024 (reference) | sweep | synchronous 135, throughput 798, constant 59 |
| PRO ctx_4096 (reference) | sweep | synchronous 128, throughput 613, constant 59 |
| PRO ctx_8192 (reference) | sweep | synchronous 119, throughput 462, constant 30 |

# Our runs, C=1 TTFT mean (s) -- the campaign, context by context

run4 = pre-repacked bank; run5 = registered page-cache DMA;
run6 = run5 + KV levers + stream-threshold fix + dual model names.

| ctx | run4 | run5 | run6 | PRO 6000 | run6 vs PRO |
|---|---|---|---|---|---|
| 8192 | 2.19 | 1.08 | 1.09 | 0.57 | 1.91x |
| 16384 | 4.54 | - | 2.28 | 1.22 | 1.86x |
| 32768 | 9.85 | 5.14 | 5.15 | 2.85 | 1.81x |
| 65536 | 22.26 | 12.94 | 12.69 | 7.50 | 1.69x |
| 128K-class | 55.14 | 35.44 | 33.99 | 39.06 | 0.87x **WIN** |
