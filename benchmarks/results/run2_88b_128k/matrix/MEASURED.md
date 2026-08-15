# 88B serving matrix - measured

Model: `shooting-brake-88b` (srswti/axe-superveloce-88b-nvfp4a16)
Server: max_model_len=131072, max_num_seqs=64, KV=262144 tokens, attention block=4176 tokens (=> 62 concurrent seats)
Harness: GuideLLM, synthetic_text, ignore_eos forced, output=512.0 tokens

| cell | prof | in tok | client C | ok/err | out tok/s | total tok/s | TTFT ms | TTFT p95 | ITL ms | TPOT ms | TPOT p95 | KV fit | measured |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_decode/c001 | concurrent | 140 | 1.00 | 14/0 | **57.5** | 73.1 | 505.4 | 525.0 | 16.52 | 17.47 | 17.60 | fits | yes |
| A_decode/c002 | concurrent | 140 | 2.00 | 15/0 | **80.4** | 102.3 | 856.5 | 1002.7 | 21.92 | 23.55 | 23.73 | fits | yes |
| A_decode/c004 | concurrent | 140 | 4.00 | 20/0 | **125.6** | 159.9 | 1704.9 | 1842.6 | 29.11 | 32.39 | 32.69 | fits | yes |
| A_decode/c008 | concurrent | 140 | 7.01 | 20/0 | **157.5** | 200.6 | 3094.1 | 3577.4 | 40.22 | 46.18 | 49.65 | fits | yes |
| A_decode/c016 | concurrent | 140 | 15.02 | 32/0 | **173.5** | 221.0 | 6136.6 | 7084.3 | 69.43 | 81.28 | 86.49 | fits | yes |
| A_decode/c032 | concurrent | 140 | 27.28 | 64/0 | **153.5** | 195.5 | 44328.0 | 66332.3 | 95.43 | 181.82 | 230.05 | fits | yes |
| A_decode/c062 | concurrent | 140 | 48.89 | 109/0 | **129.7** | 165.2 | 135448.3 | 185882.4 | 99.11 | 363.47 | 464.62 | fits | yes |
| B_context/ctx_128 | synchronous | 140 | 1.00 | 14/0 | **57.1** | 72.7 | 474.0 | 533.4 | 16.69 | 17.58 | 19.50 | fits | yes |
| B_context/ctx_130048 | synchronous | 130060 | 1.00 | 2/0 | **2.9** | 743.4 | 338631.0 | 338691.9 | 12.43 | 673.80 | 673.89 | fits | yes |
| B_context/ctx_16384 | synchronous | 16396 | 1.00 | 14/0 | **10.8** | 356.0 | 43761.5 | 52701.6 | 13.14 | 98.59 | 120.04 | fits | yes |
| B_context/ctx_2048 | synchronous | 2060 | 1.00 | 14/0 | **39.3** | 197.2 | 6031.6 | 6568.2 | 14.60 | 26.36 | 29.26 | fits | yes |
| B_context/ctx_32768 | synchronous | 32780 | 1.00 | 6/0 | **6.6** | 432.3 | 84246.3 | 88873.6 | 13.02 | 177.54 | 190.78 | fits | yes |
| B_context/ctx_512 | synchronous | 524 | 1.00 | 14/0 | **52.4** | 106.1 | 1689.3 | 1759.3 | 16.04 | 19.31 | 20.06 | fits | yes |
| B_context/ctx_65536 | synchronous | 65548 | 1.00 | 6/0 | **3.3** | 421.5 | 185038.6 | 212377.8 | 13.85 | 375.23 | 428.91 | fits | yes |
| B_context/ctx_8192 | synchronous | 8204 | 1.00 | 14/0 | **18.8** | 320.5 | 21864.1 | 25555.7 | 13.29 | 55.97 | 66.64 | fits | yes |
| C_longctx/ctx_128928_c1-3 | concurrent | 128940 | 1.00 | 4/0 | **2.0** | 501.8 | 335461.9 | 335716.3 | 12.45 | 667.62 | 668.12 | fits | yes |
| C_longctx/ctx_128928_c1-3 | concurrent | 128940 | 2.00 | 5/0 | **1.9** | 473.8 | 497422.6 | 670009.0 | 354.87 | 1325.70 | 1551.23 | fits | yes |
| C_longctx/ctx_128928_c1-3 | concurrent | 128940 | 2.00 | 5/0 | **1.9** | 473.3 | 498109.8 | 671649.0 | 355.30 | 1327.47 | 1554.97 | fits | yes |
| C_longctx/ctx_65536_c1-4 | concurrent | 65548 | 1.00 | 6/0 | **3.6** | 458.3 | 165620.5 | 165822.2 | 12.03 | 335.48 | 335.91 | fits | yes |
| C_longctx/ctx_65536_c1-4 | concurrent | 65548 | 2.00 | 7/0 | **3.5** | 450.3 | 251911.8 | 331203.6 | 171.51 | 663.19 | 768.25 | fits | yes |
| C_longctx/ctx_65536_c1-4 | concurrent | 65548 | 2.88 | 7/0 | **3.6** | 467.9 | 294506.2 | 496404.1 | 388.98 | 963.42 | 1459.65 | fits | yes |
| C_longctx/ctx_65536_c1-4 | concurrent | 65548 | 3.56 | 7/0 | **3.6** | 468.3 | 400603.6 | 706793.3 | 482.05 | 1263.54 | 1980.73 | QUEUE | yes |
| E_thinking/think_off | concurrent | 140 | 1.00 | 8/0 | **56.8** | 72.3 | 495.8 | 533.0 | 16.81 | 17.75 | 17.95 | fits | yes |
| E_thinking/think_off | concurrent | 140 | 4.00 | 12/0 | **124.9** | 159.1 | 1728.9 | 1886.7 | 29.57 | 32.89 | 32.94 | fits | yes |
| E_thinking/think_on | concurrent | 138 | 1.00 | 8/0 | **56.8** | 72.1 | 476.9 | 507.9 | 16.82 | 17.72 | 17.90 | fits | yes |
| E_thinking/think_on | concurrent | 138 | 4.00 | 12/0 | **125.9** | 159.9 | 1712.4 | 1878.5 | 29.36 | 32.64 | 32.82 | fits | yes |
