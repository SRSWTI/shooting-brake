> **FINAL VERDICT (2026-08-25, post-integration): PARKED.** Both backends
> were implemented flag-gated (`SB_GROUPED_BACKEND=onednn|mxfp4`, default
> native) and taken through the gate ladder. oneDNN: bit-exact vs shipped on
> the real bank (identity harness, M=2048 cells bitwise; small-M NaN churn
> proven PRE-EXISTING via pre-edit .so rebuild), armed cleanly in production
> serving, and won the paired 3x3 cold-TTFT ladder at every ctx >= 16K — but
> only by -2.7..-3.0% (32K: 14.07 -> 13.69 s; 127K: 65.31 -> 63.49 s).
> A 2.43x GEMM leg moving e2e by 2.8% means serving prefill is NO LONGER
> GEMM-BOUND (~7-9% share); the binding constraint is transport/dispatch/
> attention. Operator call: not worth a libdnnl dependency in the serving
> process for ~wash — parked behind the flag, default stays native. MXFP4:
> fastest kernel arm but lossy by format (rel 1.0-1.75 vs native on real
> bank; cosine 0.814 adversarial) — W4A4-class, flag only. The next prefill
> lever is the dispatch/wire path, not the GEMM. Artifacts:
> benchmarks/results/backend_bakeoff/{onednn,native}_ttft_ladder.json,
> /tmp/gate_bakeoff captures, src/phase7/xe2_probe/grouped_backend_verify.

> **ADDENDUM — bracket decomposition (same day): prefill is WIRE-BOUND and
> near its architectural floor.** From 23,641 live M=2048 dispatch traces:
> dev1 gates every layer (service p50 15.82 ms vs dev0 13.80, duty 90.6%
> vs 77.1% — the 6.47/3.23 GB/s uplink asymmetry); 74% of 32K TTFT is
> B70-bracket time; only 38% of the oneDNN compute saving surfaced in TTFT
> (the rest hides under the 25.33 MB/dispatch wire). Deeper pipelining
> (PIPELINE=4 + onednn) tested and WORSE (32K: 15.18 vs 13.69 s) — finer
> chunks fragment compute more than they overlap wire. Remaining software
> levers (expert rebalance toward dev0 ~5%, onednn +2.7%, shared-stage cuts
> masked under wire) stack to ~5-8% max. A significant prefill win from
> here requires topology, not code: fix the dev1 x4-lane asymmetry (slot
> swap) or attack the 26% on the 5090 side. Campaign closed at this floor;
> traces and ladders in benchmarks/results/backend_bakeoff/.

# Shooting Brake — Kernel Bake-off Verdict (2026-08-25)

One idle Arc Pro B70 (dev0), vLLM server down, JIT spir64 unless noted.
Geometry held constant: K=3072 hidden, I=1024 intermediate, 85 experts/card,
top_k=10 (~5 routes/card in production), decode M=1-8, prefill fill 30
rows/expert (512-tok chunk) and 120 rows/expert (2048-tok chunk).
Every number below was measured on this rig today; repro commands in the
appendix. L2 = 24 MiB (clinfo), not 18.

## Headline

| # | candidate | matched-fill result | verdict |
|---|---|---|---|
| 0 | memory ceiling | **603.6 GB/s** flat 512 MiB-4 GiB | new roofline constant (not 510/608) |
| 2 | ESIMD decode port (llm-scaler) | 312 GB/s @ M=1 = 51.7% of ceiling | **DEAD** (twice over) |
| 1 | sycl-tla bf16 grouped MoE | 3.58 ms/layer @ 30 rows/exp | **DEAD** (bf16 floor 2.66 ms > our 2.355) |
| 3 | vllm-xpu Xe2 fp16 grouped | 3.15 ms/layer @ 30 rows/exp | dead (same bf16/fp16 floor) |
| 3 | **vllm-xpu Xe2 MXFP4 grouped** | **1.025 ms/layer @ 30 rows/exp (2.30x ours)** | live, runner-up |
| 4 | **oneDNN grouped NVFP4 (f16 x f4_e2m1)** | **0.970 ms/layer @ 30 rows/exp (2.43x ours)** | **WINNER — adoption candidate** |

Shipped baseline: production grouped NVFP4 prefill = 2.355 ms/layer = 20.5
TFLOP/s (`src/phase7/xe2_nvfp4/grouped_moe.hpp`), which back-solves to 48.3
GFLOP/layer = 2,560 routed rows = **30 rows/expert** — that is the matched-fill
column everything above is judged at.

## Test 0 — bandwidth ceiling and the decode lever

`quixicore_xpu_bench --kernel membw`: 603.0 / 603.6 / 603.7 GB/s at 512 / 2048 /
4096 MiB. Flat across working sets: honest streaming ceiling.

Split GEMV (`--kernel nvfp4_moe --approx split`), M=1, K=3072, I=1024, E=85:

| dtype | routes | median | counted weight GB/s | % ceiling |
|---|---|---:|---:|---:|
| f16 (production) | 10 | 93.5 us | 504.4 | 83.6% |
| f16 | 5 (prod/card) | 49.1 us | 480.8 | 79.7% (working set 26.5 MB ~ L2, reuse-suspect) |
| f32 (bench default!) | 10 | 180.9 us | 260.9 | 43% — measurement artifact, not the production path |

The brief's "545 GB/s" baseline is the f16 path; the bench defaults to f32 —
always pass `--dtype f16`. Physics: at rows=10 the distinct working set is 8
experts x 5.31 MB = 42.5 MB (> L2), zero-cache floor 70.4 us vs 93.5 measured =
75% of floor; at production rows=5 we sit ~90% of floor. A perfect decode
kernel recovers <= 5-15 us/layer against a decode budget where B70 service is
3.07 ms of 12.1 ms total. **Prize ceiling ~2-8% end-to-end, realistically ~2%.**

## Test 2 — ESIMD MoE (llm-scaler custom-esimd-kernels): DEAD

Built the wheel for BMG (fixes below), benched
`moe_forward_full_gelu_tanh_routed` (fp16 act x FP8-E4M3 weights — the suite
has no fp16-weight MoE; its dedicated decode kernel is hardcoded H=2816/I=352,
unusable), E=85, K=3072, I=1024, top_k=10, all-distinct routing:

| M | median | distinct-weight GB/s | % ceiling |
|---:|---:|---:|---:|
| 1 | 302.4 us | 312.1 | 51.7% |
| 2 | 518.3 us | 364.2 | 60.3% |
| 4 | 970.7 us | 388.9 | 64.4% |
| 8 | 1903.8 us | 396.6 | 65.7% |

Kill #1: their streaming efficiency (52-66%) is *below* our split GEMV (~84%);
an NVFP4 port at their demonstrated efficiency lands ~150-190 us at M=1 vs our
93.5. Kill #2: Test 0's prize ceiling. Closed.

Build fixes needed (for the record): stale `build/` ninja AOT-targeted mtl-h;
`DNNLROOT` default points at a nonexistent 2025.3; and the device-link step
takes its `-device` list from `torch.xpu.get_arch_list()` (includes XeLPG,
where fp8 DPAS doesn't compile) — set `TORCH_XPU_ARCH_LIST=bmg
OMNI_XPU_DEVICE=bmg DNNLROOT=/opt/intel/oneapi/dnnl/latest`. The wheel's
bundled sycl runtime conflicts with a sourced oneAPI 2026.1 env (instant exit
3) — run it WITHOUT setvars.

## Test 1 — sycl-tla grouped MoE (example 12): the 3.3x gap was a mirage

Reference reproduced as-built on this card: 73.3-83.7 TFLOP/s bf16 at their
default fat fill (avg ~230 rows/expert, 32 groups, M-occupancy 0.62-0.65).
At our fill (85 groups, exact r15 geometry, `num_experts` edited 32->85 +
existing `MOE_ROWS` hook):

| fill | TFLOP/s (GEMM1/GEMM2) | BW util | ms/layer |
|---|---|---|---:|
| 30 rows/exp | 13.1 / 14.2 | 420-471 GB/s | 3.58 |
| 120 rows/exp | 49.9 / 52.2 | 440-516 GB/s | 3.80 |

Two corrections to the campaign narrative:
1. At matched fill **we already beat the Intel reference in TFLOP/s and wall
   time** (20.5 TF/s / 2.355 ms vs 13-14 TF/s / 3.58 ms). The scorecard's
   "20.5 vs 73-84" line compares across fills and should be retired.
2. bf16 weights = 1.60 GB/layer = 2.66 ms floor at 603.6 GB/s. **No bf16/fp16
   kernel can beat our shipped 2.355 ms on this card.** The gap was dtype
   traffic + fill, not scheduler quality.

Borrowable idea (`moe_tile_scheduler.hpp`): persistent CTAs (grid = Xe-core
count) walking a flat linearized tile sequence with per-group tile counts
computed on-device from the row-count array — zero launch quantization,
imbalance is free. But its efficiency is purely M-occupancy of a 256-row tile
(0.117 -> 13.5 TF/s, 0.47 -> 50-57, 0.65 -> 73-84, ~linear); the scheduler
balances, it can't create rows. Superseded by Test 4's result anyway.

## Test 3 — vllm-xpu-kernels Xe2 grouped GEMM

Prebuilt `_xpu_C.abi3.so` (tree build dir) loads under `.venv-xpu` torch
2.13.0+xpu. fp16 W16A16 path at our shapes (E=85):

| fill | TFLOP/s | ms/layer | note |
|---|---|---:|---|
| 30 rows/exp | 15.5 / 14.8 | 3.15 | flat 30->120: pure weight streaming |
| 120 rows/exp | 60.2 / 54.5 | 3.31 | ~500 GB/s weight stream, at the fp16 floor |

Slightly better than sycl-tla (atomic persistent scheduler + small-M policies)
but same dtype-floor prison: dead vs our 2.355.

**MXFP4 path** (E2M1 weights — same element format as NVFP4 — + e8m0/32 scales):

| fill | ms/layer | TFLOP/s |
|---|---:|---|
| 30 rows/exp | **1.025** | 43.6-48.8 |
| 120 rows/exp | 2.99 | 62.1-69.4 (dequant-bound; weights only ~140 GB/s) |

Correctness: their pytest `test_xe_grouped_gemm_mxfp4` 8/8 passed (fp16,
no-bias) on this card, plus a standalone E=85 check vs an fp32 dequant
reference — max rel err 5.4e-4. **2.30x our shipped kernel at matched fill.**
Adoption cost: scale-format delta (e8m0/32 vs our e4m3/16 + per-expert alpha)
means either lossy requantization (risky — B12x precedent) or porting our
scale path into `gemm_xe2.hpp`.

## Test 4 — oneDNN grouped NVFP4 (vendor/oneDNN): WINNER

oneDNN 2026 ships a grouped MoE encoding (`memory::desc::grouped`, variable-M,
device offsets buffer) and the exact NVFP4 recipe: f16/bf16 src x f4_e2m1
weights, f8_e4m3 block-16 weight scales, per-expert global f32 via grouped
`mul:f32:0` post-op. Built `benchdnn` from the clone (SYCL runtime, Release,
DNNL_EXPERIMENTAL); GPU impl selected: `grouped_gemm:micro:m_axis`.

Correctness (benchdnn --mode=C, f16:f4_e2m1:f16, wei:7:f8_e4m3:16x1, E=85):
- uniform 30 rows/expert: PASSED
- production-like spread 15-995 rows (3728 total): PASSED

Perf (benchdnn --mode=P, min over >=2 s per problem):

| fill | GEMM1 (N=2048,K=3072) | GEMM2 (N=3072,K=1024) | ms/layer | vs ours |
|---|---|---|---:|---:|
| 30 rows/exp | 0.614 ms @ 52.3 TF/s | 0.357 ms @ 45.0 TF/s | **0.970** | **2.43x** |
| 15-995 spread (3728 rows) | 0.809 ms @ 58.0 TF/s | 0.511 ms @ 45.9 TF/s | **1.320** | 0.354 us/row — imbalance is free |
| 120 rows/exp | 1.579 ms @ 81.3 TF/s | 1.059 ms @ 60.6 TF/s | **2.638** | — |

81 TFLOP/s at 120-fill with fp4 dequant = Intel-reference-class fill rate at
one quarter the weight traffic. Format-exact: no scale requantization, no
quality risk from the format itself.

Runtime note: benchdnn must run with the freshly built lib pinned
(`LD_LIBRARY_PATH=vendor/oneDNN/build/src`) — setvars puts the stock 2026.0
libdnnl first and the binary dies on a missing symbol.

## Projection (labelled, not a headline row)

GEMM leg 216 us/token at 2.355 ms/layer -> ~89-94 us/token at 0.97-1.03
ms/layer. Prefill 387 -> ~260-265 us/token [INFERENCE — assumes the GEMM-leg
share scales with kernel time; transport share unchanged]. That is the brief's
"half the gap" scenario: 127K prefill gap to the PRO 6000 from 3.14x toward
~2.3x.

## Ranked adoption plan (mapped to the gate ladder)

1. **oneDNN grouped NVFP4** — gate (1) done above. Next: (2) numerics vs our
   dequant oracle at checkpoint scales incl. per-expert alpha post-op and our
   [E, 2I, H] bank layout (wtag repack at load), (3) flag-gated path in the
   phase7 provider (bank bit-identical), (4) identity harness + 120-prompt
   sweep, (5) full SLO revalidation vs banked grids, (6) ledger + scorecard.
   Integration surface: both grouped GEMMs via oneDNN, keep our
   scatter/gating/finalize kernels; gating eltwise can stay ours or move to a
   post-op later. No default flip before (5) is green.
2. **vllm-xpu MXFP4** — runner-up / fallback if oneDNN integration hits a
   wall: within 6% of oneDNN at matched fill, but needs a scale-path port
   (e4m3/16+alpha into `gemm_xe2.hpp`) to avoid lossy requantization.
3. **Scorecard correction** — retire the "20.5 vs 73-84 TFLOP/s" framing;
   matched-fill numbers above replace it.
4. **Closed by measurement:** ESIMD decode port (Test 2, two independent
   kills), any bf16/fp16-weight prefill port (2.66 ms floor), and — per Test 0
   — further decode GEMV work beyond ~2% end-to-end.

## Repro appendix

```sh
# Test 0
src/QuixiCore-XPU/build-sycl/quixicore_xpu_bench --kernel membw --M {512,2048,4096}
src/QuixiCore-XPU/build-sycl/quixicore_xpu_bench --kernel nvfp4_moe --approx split \
  --M 1 --N 85 --rows 10 --K 3072 --dim 1024 --dtype f16   # f16, not the f32 default

# Test 1 (num_experts=85 edit committed in vendor example; MOE_ROWS hook pre-existing)
cd vendor/intel-xpu/sycl-tla && cmake --build build-sycl --target 12_xe20_moe_gemm_cute_interface
MOE_ROWS=30 build-sycl/examples/12_xe20_moe_gemm_cute_interface/12_xe20_moe_gemm_cute_interface \
  --n=1024 --k=3072 --num_layers=1 --verify=0    # exact r15; --n=2048 for their config

# Test 2 (no oneAPI env when running python)
TORCH_XPU_ARCH_LIST=bmg OMNI_XPU_DEVICE=bmg DNNLROOT=/opt/intel/oneapi/dnnl/latest \
  CXX=icpx .venv-xpu/bin/pip wheel vendor/intel-xpu/llm-scaler-latest/sglang/custom-esimd-kernels \
  --wheel-dir /tmp/esimd-wheels --no-deps --no-build-isolation   # oneAPI sourced for build
.venv-xpu/bin/python /tmp/bench_esimd_decode.py

# Test 3 (from vendor/intel-xpu/vllm-xpu/vllm-xpu-kernels/build/lib.linux-x86_64-cpython-312)
~/srswti/shooting-brake/.venv-xpu/bin/python /tmp/bench_vllm_xpu_grouped.py
~/srswti/shooting-brake/.venv-xpu/bin/python -m pytest \
  ../../tests/fused_moe/test_grouped_gemm.py::test_xe_grouped_gemm_mxfp4 \
  -k "float16 and False" -q --import-mode=importlib
~/srswti/shooting-brake/.venv-xpu/bin/python /tmp/verify_mxfp4_e85.py

# Test 4 (oneAPI sourced; pin the built lib)
LD_LIBRARY_PATH=vendor/oneDNN/build/src:$LD_LIBRARY_PATH \
  vendor/oneDNN/build/tests/benchdnn/benchdnn --matmul --engine=gpu --mode=P \
  --max-ms-per-prb=2000 --dt=f16:f4_e2m1:f16 --wtag=abc \
  --grouped=0:85:30+30+...(x85) --attr-scales=wei:7:f8_e4m3:16x1 \
  2550x3072:85x3072x2048     # and 2550x1024:85x1024x3072
```
