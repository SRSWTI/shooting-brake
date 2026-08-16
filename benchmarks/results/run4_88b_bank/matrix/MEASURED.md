# 88B serving matrix - measured

Model: `shooting-brake-88b` (srswti/axe-superveloce-88b-nvfp4a16)
Server: max_model_len=131072, max_num_seqs=16, KV=190990 tokens, attention block=4176 tokens (=> 45 concurrent seats)
Harness: GuideLLM, synthetic_text, ignore_eos forced, output=512.0 tokens

| cell | prof | in tok | client C | ok/err | out tok/s | total tok/s | TTFT ms | TTFT p95 | ITL ms | TPOT ms | TPOT p95 | KV fit | measured |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_decode/c001 | concurrent | 140 | 1.00 | 14/0 | **78.3** | 99.7 | 313.0 | 324.5 | 12.23 | 12.82 | 13.08 | fits | yes |
| A_decode/c002 | concurrent | 140 | 2.00 | 15/0 | **113.2** | 144.2 | 549.7 | 609.1 | 15.66 | 16.70 | 16.85 | fits | yes |
| A_decode/c004 | concurrent | 140 | 4.00 | 20/0 | **181.3** | 230.9 | 1121.3 | 1217.2 | 20.25 | 22.40 | 22.49 | fits | yes |
| A_decode/c008 | concurrent | 140 | 6.98 | 20/0 | **235.2** | 299.5 | 2070.3 | 2426.2 | 26.81 | 30.80 | 32.95 | fits | yes |
| A_decode/c016 | concurrent | 140 | 14.00 | 31/0 | **229.9** | 292.8 | 9915.1 | 29996.4 | 43.55 | 62.83 | 107.02 | fits | yes |
| A_decode/c032 | concurrent | 140 | 27.41 | 64/0 | **225.0** | 286.5 | 36146.9 | 58424.1 | 45.24 | 115.76 | 160.49 | fits | yes |
| A_decode/c062 | concurrent | 140 | 47.17 | 111/0 | **223.5** | 284.6 | 97090.2 | 139558.8 | 46.07 | 235.61 | 321.24 | QUEUE | yes |
| B_context/ctx_128 | synchronous | 140 | 1.00 | 14/0 | **78.0** | 99.3 | 301.0 | 307.3 | 12.31 | 12.87 | 13.44 | fits | yes |
| B_context/ctx_130048 | synchronous | 130060 | 1.00 | 2/0 | **14.8** | 3774.2 | 55144.8 | 55388.4 | 13.50 | 121.18 | 121.80 | fits | yes |
| B_context/ctx_16384 | synchronous | 16396 | 1.00 | 14/0 | **48.0** | 1584.3 | 4543.8 | 4668.7 | 12.62 | 21.47 | 22.25 | fits | yes |
| B_context/ctx_2048 | synchronous | 2060 | 1.00 | 14/0 | **65.8** | 330.4 | 1618.4 | 1630.5 | 12.29 | 15.43 | 15.66 | fits | yes |
| B_context/ctx_32768 | synchronous | 32780 | 1.00 | 6/0 | **34.7** | 2254.0 | 9851.6 | 10011.6 | 12.87 | 32.08 | 32.55 | fits | yes |
| B_context/ctx_512 | synchronous | 524 | 1.00 | 14/0 | **69.9** | 141.4 | 1131.8 | 1141.8 | 12.28 | 14.47 | 14.78 | fits | yes |
| B_context/ctx_65536 | synchronous | 65548 | 1.00 | 6/0 | **20.3** | 2623.1 | 22264.7 | 22411.4 | 12.91 | 56.37 | 56.71 | fits | yes |
| B_context/ctx_8192 | synchronous | 8204 | 1.00 | 14/0 | **61.5** | 1047.4 | 2193.8 | 2219.6 | 12.30 | 16.56 | 16.68 | fits | yes |
| C_longctx/ctx_128928_c1-3 | concurrent | 128940 | 1.00 | 4/0 | **11.0** | 2792.4 | 52974.4 | 53377.5 | 13.17 | 116.61 | 117.49 | fits | yes |
| C_longctx/ctx_128928_c1-3 | concurrent | 128940 | 2.00 | 6/0 | **10.3** | 2601.7 | 100477.3 | 110887.0 | 12.93 | 209.15 | 229.50 | QUEUE | yes |
| C_longctx/ctx_128928_c1-3 | concurrent | 128940 | 2.20 | 6/0 | **10.3** | 2602.4 | 109980.4 | 167574.5 | 12.89 | 227.67 | 340.23 | QUEUE | yes |
| C_longctx/ctx_65536_c1-4 | concurrent | 65548 | 1.00 | 6/0 | **20.6** | 2658.9 | 21950.3 | 22065.7 | 12.86 | 55.71 | 55.91 | fits | yes |
| C_longctx/ctx_65536_c1-4 | concurrent | 65548 | 2.00 | 7/0 | **21.7** | 2804.1 | 33284.7 | 43512.6 | 36.87 | 101.81 | 114.81 | fits | yes |
| C_longctx/ctx_65536_c1-4 | concurrent | 65548 | 2.47 | 7/0 | **21.7** | 2801.1 | 47027.9 | 78054.8 | 45.24 | 137.00 | 203.29 | fits | yes |
| C_longctx/ctx_65536_c1-4 | concurrent | 65548 | 3.00 | 8/0 | **22.7** | 2923.7 | 51974.4 | 95175.2 | 50.13 | 151.55 | 244.25 | QUEUE | yes |
| D_saturation/sweep_ctx2048 | sweep | 2060 | 1.00 | 33/0 | **67.5** | 338.9 | 1580.3 | 1583.1 | 11.85 | 14.92 | 14.96 | fits | yes |
| D_saturation/sweep_ctx2048 | sweep | 2060 | 18.09 | 62/0 | **191.4** | 961.5 | 72574.7 | 135353.3 | 53.73 | 195.37 | 315.80 | fits | yes |
| D_saturation/sweep_ctx2048 | sweep | 2060 | 1.73 | 41/0 | **83.3** | 418.6 | 1598.3 | 1604.3 | 17.40 | 20.48 | 20.55 | fits | yes |
| D_saturation/sweep_ctx2048 | sweep | 2060 | 2.72 | 49/0 | **97.4** | 489.4 | 1605.6 | 1612.5 | 24.07 | 27.16 | 27.39 | fits | yes |
| D_saturation/sweep_ctx2048 | sweep | 2060 | 3.73 | 58/0 | **113.3** | 569.3 | 1612.0 | 1619.9 | 28.81 | 31.90 | 32.18 | fits | yes |
| D_saturation/sweep_ctx2048 | sweep | 2060 | 5.85 | 60/0 | **128.5** | 645.5 | 1623.0 | 1638.3 | 44.31 | 47.40 | 49.61 | fits | yes |
| D_saturation/throughput_ctx128 | throughput | 140 | 19.81 | 78/0 | **222.7** | 283.6 | 45411.5 | 58451.8 | 46.15 | 134.75 | 161.31 | fits | yes |
| E_thinking/think_off | concurrent | 140 | 1.00 | 8/0 | **81.1** | 103.3 | 293.0 | 295.6 | 11.85 | 12.40 | 12.48 | fits | yes |
| E_thinking/think_off | concurrent | 140 | 4.00 | 12/0 | **186.3** | 237.2 | 1118.8 | 1211.2 | 19.89 | 22.04 | 22.24 | fits | yes |
| E_thinking/think_on | concurrent | 138 | 1.00 | 8/0 | **81.4** | 103.3 | 289.6 | 292.7 | 11.82 | 12.36 | 12.41 | fits | yes |
| E_thinking/think_on | concurrent | 138 | 4.00 | 12/0 | **188.0** | 238.6 | 1103.3 | 1191.0 | 19.72 | 21.83 | 21.85 | fits | yes |
