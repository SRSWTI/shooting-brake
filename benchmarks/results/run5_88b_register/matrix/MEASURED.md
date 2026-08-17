# 88B serving matrix - measured

Model: `shooting-brake-88b` (srswti/axe-superveloce-88b-nvfp4a16)
Server: max_model_len=131072, max_num_seqs=64, KV=262144 tokens, attention block=4176 tokens (=> 62 concurrent seats)
Harness: GuideLLM, synthetic_text, ignore_eos forced, output=512.0 tokens

| cell | prof | in tok | client C | ok/err | out tok/s | total tok/s | TTFT ms | TTFT p95 | ITL ms | TPOT ms | TPOT p95 | KV fit | measured |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B_context/ctx_130048 | synchronous | 130060 | 1.00 | 2/0 | **21.0** | 5367.1 | 35443.0 | 35450.4 | 12.94 | 82.13 | 82.14 | fits | yes |
| B_context/ctx_32768 | synchronous | 32780 | 1.00 | 6/0 | **48.8** | 3172.2 | 5142.4 | 5148.4 | 12.15 | 22.17 | 22.21 | fits | yes |
| B_context/ctx_65536 | synchronous | 65548 | 1.00 | 6/0 | **30.1** | 3888.4 | 12788.9 | 12888.8 | 12.42 | 37.38 | 37.54 | fits | yes |
| B_context/ctx_8192 | synchronous | 8204 | 1.00 | 14/0 | **72.2** | 1229.9 | 1077.8 | 1082.9 | 11.91 | 13.99 | 14.06 | fits | yes |
| C_longctx/ctx_128928_c1-3 | concurrent | 128940 | 1.00 | 4/0 | **15.5** | 3921.7 | 35190.6 | 35211.0 | 12.95 | 81.65 | 81.79 | fits | yes |
| C_longctx/ctx_128928_c1-3 | concurrent | 128940 | 2.00 | 6/0 | **14.3** | 3609.3 | 69871.8 | 76896.1 | 12.97 | 149.41 | 163.14 | fits | yes |
| C_longctx/ctx_128928_c1-3 | concurrent | 128940 | 2.40 | 5/0 | **14.8** | 3733.3 | 93379.7 | 118477.6 | 12.95 | 195.31 | 244.31 | fits | yes |
| C_longctx/ctx_65536_c1-4 | concurrent | 65548 | 1.00 | 6/0 | **29.8** | 3845.6 | 12938.6 | 13101.4 | 12.52 | 37.77 | 38.04 | fits | yes |
| C_longctx/ctx_65536_c1-4 | concurrent | 65548 | 2.00 | 7/0 | **32.5** | 4197.0 | 19611.0 | 25705.3 | 28.09 | 66.34 | 73.32 | fits | yes |
| C_longctx/ctx_65536_c1-4 | concurrent | 65548 | 2.79 | 7/0 | **32.5** | 4198.3 | 32049.8 | 48937.0 | 35.77 | 98.29 | 132.69 | fits | yes |
| C_longctx/ctx_65536_c1-4 | concurrent | 65548 | 2.46 | 7/0 | **32.5** | 4199.5 | 28502.4 | 48906.1 | 33.18 | 88.79 | 132.39 | fits | yes |
