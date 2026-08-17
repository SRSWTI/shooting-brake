# 88B on 5090 + Arc Pro B70 — next steps

State as of run6 (`benchmarks/results/run6_final/`).

## SHIPPED: KV recovery — 209,715 → 269,633 tokens (+28.6%), 128K C=2 flips to a WIN

The run6 `--kv-cache-memory=2.9e9` was sized under three OOM scars, not
measurement. Measured on the production boot (registration on):
steady-state free VRAM 1,577 MiB; a C=10 + 8K-prefill burst peaks only
~740 MiB above idle. Handing 800 MiB to KV is safe with margin:

* **New recipe: `--kv-cache-memory=3700000000`** → **269,633 KV tokens
  = 2.06 full seats @131K** (was 1.60).
* Verified under stress: C=10 + 8K prefill peaks at 31,766 MiB
  (841 MiB headroom); ITL regression probe **11.705 ms — unchanged**.
* **128K C=2 spot probe: TTFTs 36.8 s / 70.3 s, mean 53.6 s** — vs run6's
  67.5 s (queue-bound) and the PRO's 58.7 s. The second seat now admits
  immediately instead of queuing; the cell flips from 1.15× loss to
  ~0.91× WIN pending the full GuideLLM matrix rerun (2-request spot
  probe, not the harness cell).
* The "legacy repack buffers" recovery item is STALE and closed: the
  runtime repack died when the pre-repacked SBMARL01 bank shipped; the
  streamer holds only its 2×584.7 MiB ring arenas, both load-bearing.
  The remaining KV item is the 4,176-token attention-block padding
  (hybrid mamba page coupling) — a vLLM-geometry change, parked with the
  SGLang capacity-solve reference as the map.

## PLANNED: 122B bring-up (unsloth/Qwen3.5-122B-A10B-NVFP4, already plugin-qualified)

Config verified on disk: **attention/GDN geometry is field-for-field
identical to the 88B** (heads 32/GQA-2, head_dim 256, full-attn every 4th
of 48 layers, GDN k=128/16-64 heads, conv 4, vocab 248,320) — the whole
plugin transfers; deltas are `num_experts` 180→256 and the quant recipe.
Three unlocks verified in the checkpoint:

1. **Experts are calibrated W4A4** (e2m1 g16 weights + FP4 g16 dynamic-local
   activations with shipped scales; attention FP8 W8A8 dynamic; layer-47
   experts kept FP8). The 2.6× FP4-activation MoE kernels killed in round 1
   were killed for *uncalibrated* forced-W4A4 — this checkpoint is the
   intended regime. Gate with logprob envelope; if it passes, the ≤32K
   prefill compute gap shrinks structurally.
2. **Native MTP head ships: 785 `mtp.*` tensors** (1-layer MoE draft with
   its own 256 experts, ~1.4 GB). vLLM auto-detects
   `qwen3_5_moe → Qwen3_5MoeMTP n_predict=1` — speculation is a config
   flag, not a training project. Rollback mechanism for hybrid GDN is
   in-tree and proposer-agnostic (three-scout recon); runtime gate before
   trust. Spec cap 2 (vllm#34948 crash class above).
3. **First-party GPTQ bank source** (Qwen/Qwen3.5-122B-A10B-GPTQ-Int4):
   bits=4/g128/sym/desc_act=false all verified — direct bank source, zero
   transcode, calibrated. Three-arm bank bake-off queued: GPTQ-native vs
   NVFP4-transcode vs NVFP4-native (SBEXP001).

**Placement (sized):** dual-B70 mandatory and exact — 128/128 experts =
29.9 GB/card int4. All-remote placement frees the 5090: dense+embed ~8 GB
+ MTP experts 1.4 GB → **~17 GB KV ≈ 1.4M tokens ≈ 5.3 seats @262K** —
the capacity asymmetry vs the PRO nearly vanishes on this model (their 96 GB
holds 74 GB of weights). Decode improves: ~8 remote routes split ~4/card
in parallel ≈ 50–58 µs/card leg vs today's 94 µs single-card window.

**The one hard constraint: host RAM.** The streamed prefill bank is
59.8 GB vs 63 GB total host RAM — page-cache residency (the 53.9 GiB/s
design) does not fit beside the system. Options, priced per 48-layer pass:
RAM upgrade to 128 GB → 1.11 s/pass (clean, recommended); NVMe tail
(~45 GB cached + ~15 GB at 7 GB/s) → ~3 s/pass; B70-VRAM pull-back
(~10 GB/s aggregate, consumes the 5090 link) → ~6 s/pass, worst. Without
the upgrade the 122B still wins on decode + capacity + MTP; short-context
prefill degrades.

## SHIPPED: run6 — full PRO-matched smoke matrix; 4 outright wins

Final config: Marlin + `SHOOTING_BRAKE_BANK_REGISTER=1`, explicit
`--kv-cache-memory=2.9e9` (209,715 KV tokens; the utilization knob's
profiler estimate varies ~50 MiB per boot and runtime concurrency transients
OOM'd 3 boots — explicit bytes ended it), `SHOOTING_BRAKE_B70_MAX_BATCH=256`
(the REAL dispatch-vs-stream crossover knob; routed_experts.py:2130 —
ctx_1024 went 2.24 s -> 0.53 s, 4.2x), dual `--served-model-name`. Spot:
8K TTFT 1.035 s, ITL 11.87 ms.

Full grid: `benchmarks/results/run6_final/PRO_COMPARISON.md` (generator:
`benchmarks/compare_pro_matrix.py`) — 8 contexts x C={1..6,10} vs the PRO's
bench-matrix, mean-vs-mean, same harness, same checkpoint. Headlines:

| cell | ours | PRO | verdict |
|---|---|---|---|
| 127K C=1 | **33.99 s** | 39.06 s | **0.87x WIN** |
| 127K C=6 | 94.47 s | 97.39 s | **0.97x WIN** |
| 98K C=1 | 23.18 s | 26.12 s | **0.89x WIN** |
| 98K C=2 | 36.01 s | 39.70 s | **0.91x WIN** |
| 64K C=4 | 25.73 s | 25.78 s | 1.00x tie (C=2/3: 1.05/1.06) |
| 8K..32K C=1 | 1.09..5.15 s | 0.57..2.85 s | 1.81-1.91x behind (compute) |
| out tok/s @ 98K+127K | 15-22 | 10-19 | **ours at every C** |

Campaign C=1 (all runs, same cells): 8K 2.19 -> 1.08 -> 1.09; 32K 9.85 ->
5.14 -> 5.15; 64K 22.26 -> 12.94 -> 12.69; 128K-class 55.14 -> 35.44 ->
**33.99** vs PRO 39.06. Monotone, no regressions anywhere.

Operational facts run6 pinned: the OOM class was concurrency transients
(C=10 needs ~450+ MiB slack the boot profiler never charges); the 29.9 GiB
b12x bank page cache was evicted (kernel lost the bake-off, dead weight);
two boots died to a dead knob (`SHOOTING_BRAKE_B70_STREAM_T` gates the
dormant cpu_stream tier, not the Marlin branch — the comment in
`serve_88b_128k.sh` now names both).

### run6 decode + saturation (`benchmarks/run6_decode.sh`, same config)

The F_matrix grid has no 128-token cell and no throughput profile, so the
decode-shaped and peak rows were measured separately on the byte-identical
config (209,715 KV tokens, `B70_MAX_BATCH=256`). Full tables in
`PRO_COMPARISON.md`; generator `compare_pro_matrix.py`.

**Decode is unchanged from run4 within noise — as expected, we never touched
the decode path this campaign:**

| C | run4 out tok/s | run6 out tok/s | run4 ITL | run6 ITL |
|---|---|---|---|---|
| 1 | 78.3 | 80.7 | 12.23 | 11.88 |
| 4 | 181.3 | 189.1 | 20.25 | 19.93 |
| 8 | 235.2 | 251.7 | 26.81 | 26.60 |
| 16 | 229.9 | **189.7** | 43.55 | 41.69 |
| 32 | 225.0 | 240.5 | 45.24 | 46.44 |
| 62 | 223.5 | **271.1** | 46.07 | 45.45 |

Two rungs moved beyond noise and neither is explained: **C=16 lost 17%**
(throughput down, ITL *better* — fewer requests actually co-resident, not
slower tokens) while **C=62 gained 21%**. KV seats are not the cause (128+512
tokens/seat fits trivially in either config). Logged as unexplained rather
than claimed; the c016 cell ran 26 requests vs run4's 31, so measurement
variance is the leading hypothesis.

**Peak throughput, unbounded offered load:** 222.7 -> **270.0** out tok/s at
128 tokens (+21% vs run4); 191.4 -> 195.5 at 2048 tokens (+2%).

**And the honest loss:** the PRO's sweep peaks at **798** out tok/s @1K, 613
@4K, 462 @8K. Our 2048-token peak of 195.5 against ~700 interpolated on their
curve is **~3.6x behind**; our 270 @128 against their >=798 is **~3x behind**.
Peak throughput at short context stacks many concurrent short prefills —
precisely the regime where we are 1.8-11x behind per-request. This is the
single worst number in the whole comparison and it is structural, not a
tuning miss.

Vendor recon and the resulting decision tiers now live in
`docs/vendor-extraction.md` (12-agent deep scope over the vendored repos,
four live verifications, TAKE/BENCH/PARK/REJECT verdicts).

## MEASURED: B70 int4 GEMV bandwidth audit — decode roadmap ① CLOSED

Three-arm audit of the decode kernel's achieved bandwidth on the idle B70
(same silicon as the serving card; kernel bandwidth is link-independent).
Artifacts: `benchmarks/results/b70_gemv_audit/` (`gemv_bw_audit.json`,
`sgl_w4a16.json`, `vllm_xpu_w4a16.json`, `itl_probe.json`); driver
`benchmarks/b70_sgl_w4a16_bench.py`; probe `benchmarks/b70_itl_probe.py`.

**The discriminating question — "are we at ~55% or ~75% of peak at M=1?" —
had a third answer: 68% sustained, 26–44% cold, and the cold half is DVFS,
not kernel code.**

### Instrument fixes that preceded any trustworthy number

* **Xe2 memory compression inflated the old fixture.** `xpu_bench`'s
  constant-fill int4_moe bank measured up to **111% of the 608 GB/s spec**
  — physically impossible for DRAM reads. Random-filled (incompressible)
  fixture shipped in `perf/harness/xpu_bench.cpp`; every constant-fill
  bandwidth figure from before this fix is inflated ~25% at M≥8.
* **New `membw` mode** (same file): measured device read ceiling
  **599.2 GB/s = 98.6% of spec**, incompressible data, ±0.1% spread. All
  utilization percentages below are against this measured ceiling.
* **Cold-process runs of the same kernel swing 2.6×** (47–124 µs at M=1,
  tight within-run) — GuC SLPC clock state, proven by concurrent
  `act_freq` sampling (cold runs execute at 517–2200 MHz median, never
  reaching 2800). Every microbench on this card MUST pin clocks
  (`min_freq`=`max_freq` in `/sys/class/drm/cardN/device/tile0/gt0/freq0/`,
  pin verified to hold via `act_freq` — PCODE-resolved; `cur_freq` only
  shows GuC's request) or run long sustained windows. The vendored zml
  claim "xe has no clock control" is WRONG for this kernel: `min_freq` /
  `max_freq` exist and the min==max pin holds (pinned cold M=1:
  46.7–47.1 µs dead flat, act_freq 2633–2800).

### The numbers (sustained clocks, incompressible, rotating disjoint route sets)

Geometry E=126 / K=3072 / I=1024 / g128, 4 valid routes per token,
weight bytes = 4,866,048 per route (== bank stride). % of measured
599 GB/s ceiling:

| M | ours `split` | ours `down2` | sgl-kernel fused_experts | vllm-xpu XpuFusedMoe |
|---|---|---|---|---|
| 1 | **47.9 µs / 68%** | 50.4 / 64% | 100.7 / 32% | 101.2 / 32% |
| 2 | 82.5 / 79% | **76.4 / 85%** | 131.9 / 49% | 132.4 / 49% |
| 4 | 154.1 / 84% | 152.6 / 85% | 210.1 / 62% | 210.9 / 62% |
| 8 | 295.9 / 88% | **282.5 / 92%** | 342.0 / 76% | 341.4 / 76% |
| 16 | 580.2 / 90% | **553.5 / 94%** | 604.5 / 86% | 602.2 / 86% |
| 32 | 1162.1 / 89% | **1093.5 / 95%** | 1108.4 / 94% | 1106.2 / 94% |

* **Our kernel wins every decode shape.** At the production M=1 rung the
  vendored candidates are **2.1× slower end-to-end** — their 5-launch
  orchestration (prepare + scatter + GEMM1 + silu_and_mul + GEMM2) is
  exactly wrong for the latency-bound M=1 regime our fused 2-kernel split
  was written for. Both vendored providers measure identical (±0.5%): same
  Xe20 grouped-GEMM family underneath, their own bench header says so.
  Correctness-gated before timing (cosine 0.99998 vs their naive
  reference). **vendor-extraction BENCH FIRST 1c: CLOSED, NEGATIVE for
  decode.** (They close only at M=32 — a prefill shape the B70 no longer
  serves.)
* **M=1 at 68% is bf16-GEMV-class, not the 51–59% int8 trap.** Max kernel
  headroom at M=1 is ~1.35× (48 → ~36 µs/layer), and gate_up (⅔ of bytes,
  1,024 exposed subgroups) is the binding stage — occupancy, not decode
  efficiency. The in-tree fix direction, if ever needed: a gate_up
  analogue of `down_wide`.
* **`down2` is a free provider win at pairs≥8** (85–95% vs 79–89%);
  slightly worse at M=1. Candidate: dispatch `down2` when
  `M × valid_routes ≥ 8`, keep `split` at M=1. One-line switch in
  `b70_provider.cpp` (`int4_moe_split_down_wide_sycl` already exported).
* **The old "44 µs @ k=8" standalone table was cache-hot** (E=8 fixture,
  39 MB working set vs 18.6 MiB L2 + compressible fill). Cold-expert
  reality at k≈5.6 is ~70 µs/layer. Do not price overlap decisions
  against the old table.

### Production A/B: the DVFS pin does NOT move serving ITL

Fresh run6-config boot, warmed (2×8K + PSI<0.1), streaming probe
128-in/256-out C=1:

| arm | ITL | act_freq during decode |
|---|---|---|
| baseline min=400 | **11.89 ms** (== run6's 11.87) | median 2800, idle 0.4% |
| pinned min=2800 | 12.10 ms | median 2800 |
| sporadic (12 s idle), early-20 gaps | 11.08–11.16 ms both arms | — |

Decode self-warms the card: 48 dispatches per ~12 ms step is one doorbell
every 250 µs — the GPU never idles down, and after real idle the request's
own prefill re-ramps it before decode starts. **DVFS is a microbenchmark
hazard, not a serving lever at C=1.** (Untested: whether very low-QPS
multi-second-gap traffic at higher concurrency ever exposes it.)

## SHIPPED: doorbell XPU host registration — ITL 11.92 → 11.71 ms (−1.8%)

Decode lever ② from the audit, landed same day. The doorbell staging
buffers (`routed_experts.py` `_b70_pinned_hidden`/`_pinned_b70_ids`/
`_pinned_b70_weights`/`_b70_pinned_output`) are torch `pin_memory=True` —
registered with the **CUDA** caching host allocator only. Registration is
context-scoped, so from the B70's Level Zero side those ranges were
pageable: staged H2D + synchronous D2H on every doorbell round trip,
96 transfers per decode step.

Fix: `B70Provider::register_host_range` (`b70_provider.cpp`, wraps
`syclex::prepare_for_device_copy`, ranges released at shutdown), called
from `sb_b70_poll_register` for all four buffers per layer. Default ON;
kill switch `SHOOTING_BRAKE_B70_XPU_REGISTER=0`. This is the same move as
run5's `cudaHostRegister` on the Marlin bank, pointed the other way.

Measured ladder, every step gated before the next:

1. **Standalone provenance A/B** (`b70_dispatch_latency environment`, new
   `+reg` arms, clock-pinned): cudaHostAlloc 29.3 → **20.5 µs** per
   dispatch at the production 180 µs duty cycle — registration fully
   recovers sycl-native latency on CUDA-pinned ranges.
2. **Numerics + lifecycle smoke** (`experiments/b70_xpu_register_smoke.py`,
   real bank, real poller, torch-pinned buffers): cross-arm max delta
   2.27e-13 on 1.75e-6-scale outputs = atomic-reorder noise; 0 dispatch
   errors; clean register→release→shutdown.
3. **Production A/B, same binary, flag on/off boots** (4 probe runs per
   arm, `benchmarks/b70_itl_probe.py`, 128-in/256-out C=1):
   OFF 11.895/11.920/11.914/11.934 (mean 11.916) vs
   ON **11.700/11.701/11.706/11.712 (mean 11.705)** — zero overlap
   between arms; flag-off reproduces the pre-change baseline exactly.

**−211 µs/step = −1.8% ITL — the first decode improvement since run4, and
a new best ITL for this config.** In-situ saving is ~2.2 µs/transfer vs
8.8 standalone: the rest is masked by the CUDA-partial overlap, which is
direct evidence for the max() cost model and prices the remaining
transport headroom at roughly another ~2 µs/transfer if the B70 leg were
ever the binding one.

Operational scar from the A/B, recorded: `ZE_AFFINITY_MASK` exported by an
earlier shell experiment leaked into a serve relaunch and silently loaded
the bank onto the **Gen3** B70 (the serve script honors caller values).
Caught by the operator watching nvtop. Relaunches must scrub the env
(`env -u ZE_AFFINITY_MASK`) or use a fresh shell.

## MEASURED: decode overlap trace — the max() question ANSWERED, presumption overturned

The gate that blocked the decode roadmap ("which leg binds is unmeasured")
is closed. Instrument: a native per-dispatch trace ring in the poller
(`b70_capi.cpp` `TraceEntry`/`sb_b70_poll_trace_snapshot`, host
CLOCK_MONOTONIC — verified 3 µs off Python's `monotonic_ns`), dumped by an
opt-in thread (`SHOOTING_BRAKE_B70_TRACE_DUMP=<path>`), merged single-clock
with a same-process torch-profiler capture (dev-mode `/start_profile`,
`--profiler-config.profiler=torch`). 154 decode steps, 7,392 doorbell
windows against 320K CUDA GPU events. Artifact:
`benchmarks/results/b70_gemv_audit/decode_overlap_trace.json`.

**The 12.35 ms step budget (C=1, registration on, profiler adds ~5%):**

| component | ms | share |
|---|---|---|
| B70 windows (signal→completion) | 4.45 | 36% |
| — **exposed: GPU idle inside windows** | **2.99** | **24%** |
| — CUDA-busy under windows | 1.55 | 12% |
| inter-dispatch gaps | 7.82 | 63% |
| — CUDA-busy (attention/GDN/partial/sampler) | 5.90 | 48% |
| — **GPU-idle (host/scheduling)** | **1.92** | **16%** |

**The "B70 leg is hidden under the CUDA partial" presumption was WRONG** —
it came from the cache-hot standalone table (44 µs vs 100 µs). Reality:
94.3 µs windows per dispatch with the GPU idling through 66.7% of them.
Consequences, each now priced in ms:

1. **The B70 leg is live again: −3.0 ms/step reachable** (−24% ITL). In
   order of cheapness: `down2` adaptive dispatch (in-tree), **dual-B70
   route split** (banks built, split math proven; halves the leg →
   ~−1.5 ms), kernel work (occupancy-capped ~1.35×).
2. **Host/scheduling idle is a second pool: −1.9 ms/step** — the
   launch-economics class the B70 cookbook measured (graph replay took a
   comparable model 4×). SYCL-graph replay into `issue()` (built,
   validated, parked at item 5 of the old plan) re-enters the queue.
3. The CUDA-busy floor is 7.45 ms/step — the architecture's decode floor
   without speculation; native MTP (see 122B section) divides it by
   accepted tokens.
4. This also reconciles the registration result: 2.2 of 8.8 µs/transfer
   survived because transfer overhead partially sits in the hidden 33%;
   the trace measures the split directly instead of inferring it.

Instrument caveat: L0 `kernel_ns` read 0 this boot (`B70_PROFILE` did not
reach the provider queue) — window timing is unaffected; chase before the
kernel/transport sub-split is needed.

### Decode front, re-priced by the trace

1. Dual-B70 route split + `down2` — attack the 3.0 ms exposed pool.
2. SYCL-graph/submission economics — attack the 1.9 ms host-idle pool.
3. 150 W cap (④) — free, unaffected.
4. Kernel replacement stays DEAD (audit); 122B native MTP divides the
   7.45 ms CUDA floor.

## SHIPPED: registered page-cache DMA (run5) — PRO 6000 beaten at 130K

`SHOOTING_BRAKE_BANK_REGISTER=1` (`marlin_prefill.py::_open_bank_source`):
the Marlin bank is mmap'd `MAP_PRIVATE|PROT_WRITE` over an `O_RDONLY` fd
(COW never fires — we never write; sidesteps `cudaHostRegisterReadOnly` and
its device-attribute lottery) and `cudaHostRegister`'d in the worker
post-fork. The streamer's existing `non_blocking` ring copy becomes true
copy-engine DMA at the ceiling. Probe:
`benchmarks/results/slab_h2d/hostregister_probe.json`; production results:
`benchmarks/results/run5_88b_register/`.

Measured, GuideLLM synchronous C=1, same harness as run2/run4:

| ctx | run2 | run4 (bank) | run5 (register) | PRO 6000 | gap |
|---|---|---|---|---|---|
| 8,192 | 21.9 s | 2.19 s | **1.08 s** | 0.556 s | 1.94x |
| 32,768 | 84.2 s | 9.85 s | **5.14 s** | 2.71 s | 1.90x |
| 65,536 | 185 s | 22.3 s | **12.79 s** | 7.16 s | 1.79x |
| 130,048 | 339 s | 55.1 s | **35.44 s** | 38.79 s @127K | **0.91x — WIN** |

Decode ITL 11.9-12.9 ms in all runs (doorbell untouched).

Concurrent long-context (register arm, GuideLLM concurrent profile, means;
PRO figures are the mean column of their `ctx_*/concurrent/benchmarks.csv` —
an earlier reading that quoted their fastest percentile as "~40 s at C=6" is
WRONG and withdrawn; their concurrent means rise 39.1 -> 97.4 s over C=1..6):

| ctx / C | ours | PRO | gap |
|---|---|---|---|
| 64K C=1 | 12.9 s | 7.5 s | 1.72x |
| 64K C=2 | 19.6 s | 18.8 s | **1.04x — tie** |
| 64K C~2.5-2.8 | 28.5-32.0 s | 24.3 s (C=3) | 1.2-1.3x |
| 128K C=1 | **35.2 s** | 39.1 s | **0.90x — WIN** |
| 128K C=2 | 69.9 s | 58.7 s | 1.19x |
| 128K C~2.4 (req 3) | 93.4 s | 66.3 s (C=3) | 1.41x |

Amortization confirmed at 64K: our C=1->2 scaling is +52% for 2x work vs
their +151% — shared streaming + co-batched prefill flattens our curve, and
the 1.72x gap collapses to a tie at C=2. The 128K C>=2 losses are
CAPACITY-BOUND (runner-flagged): 31 blocks x 4,176 tokens per 128K seq
seats only TWO sequences in our KV budget, the third queues into TTFT,
while the PRO packs 6 into one 96 GB pool. ITL held 12.95 ms throughout —
this is KV real estate, not a streaming defect. The recovery lever is
already on the books: the ~67K KV tokens held by the streamer's legacy
repack buffers (dead since the pre-repacked bank) plus 4,176-block padding
waste.

Microbench trio (`benchmarks/prefill_floor_bench.py`, artifacts in
`run5_88b_register/floor_*.json`), all kill conditions cleared:

* **register**: full 27.4 GiB bank pinned in 3.7 s (7.4 GiB/s), DMA spot
  checks 53.7-53.9 GiB/s across the whole bank, THP vmstat deltas all zero,
  MemAvailable -27.7 GiB exactly.
* **compute floor** (weights resident, zero streaming): remote-MoE 48-layer
  0.098 s @ M=2048 / 0.339 s @ 8192 / 1.296 s @ 32768 — linear, 7.07
  ms/layer @ 8K.
* **overlap**: whole-bank stream concurrent with 48 layers of M=32K Marlin
  compute degrades 2.4% vs ideal max() — `TTFT = max(stream, compute)` is
  measured fact; copy engine and SMs coexist.

Numerics gate (`run5_88b_register/firsttok_ab.json`): 8 shifted-window
streamer-path prefills, first-token logprob deltas on-vs-off, max |d| 0.211
(bug signature 0.49), mean 0.086 == the cross-boot noise floor measured on
the dispatch path, 7/8 argmax match. Two dead ends recorded so they are not
re-fought: (1) API `prompt_logprobs` CANNOT gate the streamer — >1K-token
prompts OOM-kill the engine (M x vocab fp32 logits vs ~1.5 GiB free) and
<1024-token prompts never engage the streamer (threshold); gate on
generated-token logprobs, and only up to the first greedy divergence —
cross-boot argmax flips on flat distributions are noise, not corruption.
(2) Bit-equality across boots does not exist even flag-off (autotune picks
different reduction orders); the envelope, not zero, is the gate.

Operational notes:
* The pin evicts ~9 GiB of server working set on first prefill (one-time,
  measured 2.4M pages swapped): first requests after boot see tens-of-seconds
  TTFT until swap-in completes. Warm with 2 x 8K requests and wait for
  `/proc/pressure/memory` avg10 < 0.1 before measuring (or serving).
  Follow-up worth doing: eager-init the streamer at boot so the thrash
  window ends before health goes green.
* JIT extension builds invoke bare `ninja`; non-activated shells need
  `PATH="$PWD/.venv/bin:$PATH"` (two boots died to this).
* Cross-vendor portability: the plugin replaces vLLM's generic
  `RoutedExperts`/`MoERunner` pluggable layers; DeepSeek V2/V3.x and
  GLM4-MoE construct experts through the same factory, so the expert plane
  ports; the quant contract (int4 g128 vs FP8-native checkpoints) is the
  only real lift.

**Bottleneck moved.** Stream (0.51 s/pass) now hides under compute at every
context, so stream-once/Tier A (MNBT=32K, recipe in `serve_88b_128k.sh`;
M-tiling shipped in `partial()` via `SHOOTING_BRAKE_PREFILL_TILE`) buys only
~5% and is parked. The remaining 1.7-1.9x at <=64K is pure compute — a
grouped-GEMM contest, and the candidate list is now source-verified instead
of assumed (see "Kernel round 2"). SM100 numbers are never evidence for
SM120: they are separate compile-time gates
(`CUTLASS_ARCH_MMA_SM120A_ENABLED` needs `-arch=sm_120a`, `..._SM120F_ENABLED`
the forward-compatible `f` family) — and the `a`-vs-`f` suffix distinction is
the recurring root cause behind nearly every SM120 kernel bug we found.

**Kernel bake-off, step 1 CLOSED — NEGATIVE. Marlin keeps the crown.**
vLLM ships `FlashInferB12xExperts` (`flashinfer.fused_moe.B12xMoEWrapper`), a
consumer-Blackwell (SM12x) native-FP4 fused MoE excluded from auto-selection
by an upstream SM121 MMA guard, opt-in via `moe_backend="flashinfer_b12x"`.
It is 2.6x faster than Marlin and unusable.

Speed (synthetic weights, our exact shapes E=126/K=3072/N=1024/top-8;
`prefill_floor_bench.py --mode b12x`, artifact
`run5_88b_register/floor_b12x.json`):

| M | marlin ms/layer | b12x W4A4 | b12x W4A16 |
|---|---|---|---|
| 2048 | 2.04 | 0.92 (2.2x) | - |
| 8192 | 7.07 | **2.70 (2.6x)** | **8.16 (slower)** |
| 32768 | 27.0 | 10.42 (2.6x) | - |

Quality killed it (`benchmarks/b12x_bank_poc.py` against the repo's validated
CPU NVFP4 dequantizer, artifact `b12x_poc/poc.json`):

* **W4A4 is terminal.** Driving the kernel with *flashinfer's own* quantizer
  — bypassing our bank entirely — per-layer cosine tops out at **0.82**
  (`b12x_encoding_probe.py`); ours reached 0.19, so the delta between those
  is our encoding gap, but the 0.82 ceiling is theirs and it compounds over
  48 layers. The cause is structural, not a bug: FC1 input quant uses a
  **static** e2m1 grid with no per-block rescue (their own test
  `test_input_global_scale_decouples_weight_alpha` states the contract), and
  the NVFP4-activation literature puts flush-to-zero at 0.05-0.20 with
  `up_proj` the worst offender. Half-order, nibble-order and sf2-layout
  permutations all measured *identical* error (`b12x_halfswap_probe.py`),
  which rules out layout and leaves the arithmetic.
* **W4A16 is slower**, so its better fidelity buys nothing: 8.16 vs Marlin's
  7.07 ms/layer (`b12x_w4a16_probe.py`). No speed, no reason.

Salvage, all real:

* **An upstream vLLM bug, filing-ready.** `FlashInferB12xExperts`' load-time
  bake multiplies ModelOpt's per-expert `weight_scale_2` into the e4m3 block
  scales; for this checkpoint the product falls below e4m3's smallest
  subnormal and **86.5% of scales flush to zero** (`b12x_zero_bisect.py`).
  ModelOpt itself clamps that scale to [2^-9, 448] "to avoid underflow->0",
  so vLLM's b12x bake re-introduces a hazard the source library already
  fixed (cf. ModelOpt PR #1397, the overflow-direction sibling). Our fix —
  bake only the gate/up *ratio* (O(1), e4m3-safe) and carry the true scale as
  per-expert fp32 alpha planes — is implemented in `b12x_bank_format.py`
  (6-plane ABI) + `src/phase1/build_b12x_bank.py`.
* **bank-v2 tooling**: 29.9 GiB NVFP4 bank built and bit-validated from the
  checkpoint's native planes in 6 min, zero requantization. Ports to any
  future FP4 kernel.
* flashinfer went 0.6.16.post3 -> vendored nightly 0.6.18 for
  `input_global_scale` (in 0.6.16 `w1_alpha` doubles as the FC1
  activation-quant scale, making the fix unexpressible), then **rolled back**
  to 0.6.16.post3 once the verdict landed: the serving path never touches
  b12x, so the nightly's sampler/FP8 regression risk was unwarranted.

## Kernel round 2 — source-verified candidates (four-repo recon)

Four vendored clones read at source level (`vendor/b12x`, `vendor/qutlass`,
`vendor/cutlass`, `vendor/cute`). Net: one real candidate, one multi-week
lever, one confirmed wall, one design aid.

**`vendor/b12x` — a sibling lab, and the next kernel candidate.** An
SM120/121 CuTe-DSL inference-kernel lab aimed at our exact silicon (they even
bench on the RTX PRO 6000, our comparison target).

* **`w4a8_nvfp4` uses dynamic per-32-block activation quant** (UE8M0+E4M3,
  amax per block, computed inline — `dynamic.py:3628+`). That is a
  categorically different quality regime from the static-global W4A4 that
  measured 0.82 today; per-block dynamic fp8 activations are near-lossless in
  practice, and our KV cache already runs `fp8_e4m3`. It takes ModelOpt NVFP4
  natively with a documented **direct-multiplier** convention
  (`_impl.py:4953`) — no reciprocal guessing this time. Expected roughly half
  Marlin's latency *if* it passes the gates.
* **Two named streaming hazards — the exact class that burned us today,
  identified before integration:**
  - derived scale grids cached by `data_ptr` (`_impl.py:3448`) -> stale under
    our rotating arenas. Mitigation: pre-materialize the grids into bank-v3
    planes at build time (we own that pipeline now), or price per-layer
    re-prepare.
  - `_W13_NORMALIZED_STORAGES` (`_impl.py:4366`) does a one-time in-place row
    swap and is deliberately never cleared -> silent skip on pointer reuse.
    Mitigation: never use `w13_layout='w31'`.
* **Decode-war gift:** `MoEMicroKernelW4A16SmallMDirect` — M=1-8, native
  ModelOpt layout, `e4m3_k16` scales = our checkpoint verbatim, no repack. A
  direct candidate for the 5090's local-54 decode partial, plus their
  decode-policy sweep tooling as ready-made methodology.
* **Not applicable:** their `comm.pcie` is CUDA-IPC-only (dead on a
  mixed-vendor box); their vLLM plugin is FP6-only and the NVFP4 glue lives
  in the maintainer's private fork — but their docs bless calling
  `torch.ops.b12x.*` directly, which is exactly how our plugin consumes it.
* **Sobering:** zero b12x-vs-Marlin comparisons and zero 5090 numbers exist
  in that repo. We measure everything ourselves; the floor-bench harness is
  already wired (`prefill_floor_bench.py --mode`).

**`vendor/qutlass` (IST-DASLab — the Marlin authors) — the W4A4-quality
answer, priced in weeks.** Fused Hadamard-as-micro-GEMM in the quantize
epilogue with runtime-loaded rotation matrices: the literature-proven fix for
exactly the 0.82 ceiling we measured. But it is dense-only (hard 2-D checks
in `bindings.cpp`; no grouped/MoE path or roadmap), weights must be
requantized from bf16 through their pipeline (our ModelOpt planes cannot feed
it), it pins CUDA 12.8/torch 2.8 against our CUDA 13/torch 2.13, and carries
no in-repo quality numbers (claims live in the papers). Verdict: parked R&D —
the credible path to ~2.7 ms/layer *with* quality.

**`vendor/cutlass` — confirms the wall, and shows the one door.** sm120's
grouped collective hard-asserts F8F6F4 operands
(`sm120_array_mma_builder.inl:83-84`): there is **no int4xbf16 mixed-input
grouped GEMM for our silicon** anywhere in the tree. The mixed-input grouped
kernels live on sm90/sm100 only — including a CuTe-DSL Python grouped
mixed-input example, the cleanest skeleton if we ever hand-build
"Machete-for-sm120" (Machete itself is Hopper-built; its sm120 support in
vLLM is unverified). That fusion does not exist and would have to be written
by hand: the deepest lever, weeks+.

**`vendor/cute` — PyCuTe:** pure-Python layout algebra, no kernels.
Design/prototyping aid only.

### Revised ladder

| when | move | expected |
|---|---|---|
| now | campaign on the proven Marlin+register stack: KV levers -> gates -> smoke matrix (contexts x C=1-6,10) -> PRO report | locked baselines; the 130K C=1 win defended with a full matrix |
| kernel round 2 | bench b12x `w4a8_nvfp4` + native `w4a16` at our shapes. **Gate quality first, speed second** — today's order cost hours. Pre-materialize derived grids into bank v3 if it wins | ~3.5-5 ms/layer at near-W4A16 quality -> 8K TTFT toward ~0.8-0.9 s |
| decode war | b12x small-M-direct for the local partial + their sweep methodology + our trace-gated ladder (one-step trace -> hotness placement -> graph replay) | 1.65x -> ~1.2-1.3x; speculation is the only lever that goes below 1x |
| parked R&D | QuTLASS rotations (W4A4 + quality), hand-built sm120 mixed-input grouped GEMM, FP6-W6A8 (needs a bf16 base) | the 2.6x-with-quality endgame |

**Process change earned today:** bake off *quality before speed*. The b12x
detour measured 2.6x, then spent hours discovering the number was unusable.
A single `b12x_encoding_probe.py`-style run against flashinfer's own
quantizer would have closed it in minutes.

### Round-2 execution plan (deep-scout verified, 4 agents over vendor/b12x)

Corrections to the first-pass recon, all with file:line evidence in the scout
transcripts (`history://W4A8Contract`, `://W4A8Quality`, `://W4A16Decode`,
`://B12xBuildIntegration`):

* **w4a8_nvfp4 is the same persistent-grid kernel as nvfp4**, specialized by a
  const-baked `quant_mode` (`torch.ops.b12x.tp_moe_dynamic_launch`). Weights
  stay ModelOpt-native — packed fp4 + FlashInfer-swizzled e4m3 K/16 scales,
  i.e. **our bank v2 planes verbatim**. The derived grids (UE8M0 K/32 +
  e4m3 K/16 residual, `_derive_w4a8_weight_grids`, `_impl.py:3448`) are pure
  torch (frexp/exp2/clamp), **CPU-callable offline** -> bank-v3
  pre-materialization is confirmed feasible, and the kernel launch takes the
  grids as explicit tensor args, sidestepping the `data_ptr` cache.
* **The w4a16 large-M path is cache-free**: `quant_mode=="w4a16"` dispatch
  bypasses `_get_weight_views`/`_W13_NORMALIZED_STORAGES` entirely
  (`_impl.py:10644-10715`); all its kernel caches are shape-keyed. Arena
  rotation is safe there. But "native ModelOpt layout" is weight-bytes only:
  **scales are always re-permuted** into b12x's own packed-scale order
  (`prepare.py:292-420`) — a one-time prep we would bake as a bank plane.
* **Alpha trap confirmed**: their benchmark loader derives the fused w13
  alpha from gate_proj's `weight_scale_2` only, silently discarding up_proj
  (`benchmark_moe.py:906-909`). Never copy it. For w4a8, alpha reduces to
  pure `weight_scale_2` (MXFP8 activations carry no global scale) — our
  ratio-bake convention maps, pending a numerics check.
* **Operational hazards**: (a) documented segfault class when first-use CuTe
  JIT happens inside a live vLLM engine-core worker (`smoke_bf16_gemv.py`
  docstring) -> eager-compile at boot, then
  `b12x.freeze_kernel_resolution("serving")`; (b) `w13_layout="w13"` is the
  safe order (w31 triggers the never-cleared in-place swap); (c) workspace
  pool is exact-shape-keyed and refuses to grow mid-capture — warm every
  shape before graph capture.
* **Install is free**: pure Python, no build step, `nvidia-cutlass-dsl==4.6.0`
  already in `.venv` byte-for-byte, no flashinfer conflict.
* **Decode (parked until round 2 lands):** small-M-direct is M<=8,
  `expert_map=None`, modelopt-layout-only -> our 54-of-180 needs compact
  local ids; the general path's `use_expert_map` mode (their Kimi-K3
  benchmark analog is exactly our residency shape) needs the repacked
  "packed" layout instead. Mutually exclusive; measure both.

Execution order (quality first — the process rule this campaign earned):

1. **CPU probe — MEASURED (first ever on real scales;
   `benchmarks/results/run6_final/w4a8_residual_probe.json`).** Their
   `nvfp4_mx_residual_quality_report` (`reference.py:1035`) over 48 layers x
   4 remote experts x 3 projections of the real checkpoint: gate/up are
   EXACT everywhere (pair exponent spread 2-3, zero flush); down_proj is
   clean on 45/48 layers; **layers 0, 1, 41 hit the e4m3 residual-underflow
   pathology on down_proj** (flushed 6.2% / 4.3% / 0.4%, pair spreads to
   2^17). Verdict: PASS with per-layer mitigation — the streamer is
   per-layer, so the three affected layers can stay on Marlin (hybrid) if
   w4a8 wins. First-layer scale wildness is the classic pattern; the probe
   turned a vague "quality risk" into three named layers.
2. **Quality gate — ORACLE LEG MEASURED, PASSED**
   (`benchmarks/b12x_w4a8_gate.py --cpu-only`, artifact
   `run6_final/w4a8_gate_cpu.json`). b12x's own w4a8 reference
   (`moe_reference_w4a8_mx`) fed from OUR bank-v2 planes, scored against the
   exact fp32 checkpoint oracle, m=64, real routed inputs:

   | layer | cosine | rel L2 | note |
   |---|---|---|---|
   | 24 (clean) | 0.9985 | 0.055 | |
   | 0 / 1 / 41 (residual-flush) | 0.9984 / 0.9983 / 0.9983 | ~0.058 | flush is output-invisible |

   Where W4A4 measured 0.82, w4a8's dynamic per-block quant lands 0.998+ on
   real weights through our own planes — the per-layer hybrid mitigation for
   0/1/41 is unnecessary. Two integration facts the gate surfaced:
   (a) bank-v2 sf planes need a swizzle adapter (flashinfer emit order ->
   b12x's TRT-LLM arrangement; both directions byte-verified in the gate
   script); (b) our gate-first row stacking is b12x's `w13_layout="w31"` —
   the wrong default silently computes silu(up)*gate and scores 0.87, and
   the kernel's w31 path is the in-place-swap hazard, so **bank v3 emits
   up-first planes**. Remaining leg: kernel-vs-oracle on GPU (their in-repo
   gate: cos > 0.998) — folds into the speed floor run.
2b. **Kernel gate — MEASURED, PASSED** (`w4a8_gate_gpu.json`): kernel vs
   their w4a8 oracle cos 0.9993/0.9992 (their in-repo bar: 0.998); kernel vs
   exact fp32 cos 0.9985/0.9983 — the kernel adds nothing beyond w4a8's own
   quantization. Ran through the production-shaped call: up-first planes
   (bank-v2 gate-first rows pre-swapped; the wrong layout silently computes
   silu(up)*gate and scores 0.87 — hit twice, once per arm).
3. **Speed floor — MEASURED, FAILED** (`prefill_floor_bench.py --mode
   vb12x`, artifacts `run6_final/floor_vb12x_w4a8.json` / `_w4a16.json`;
   real bank planes, layer 24, eager, same loop discipline as the 7.07
   incumbent):

   | kernel | M=8192 ms/layer | quality | verdict |
   |---|---|---|---|
   | **Marlin (incumbent)** | **7.07** | W4A16 ref | **keeps the crown** |
   | vendor-b12x w4a16 | 7.93 (+12%) | Marlin-class | slower |
   | vendor-b12x w4a8_nvfp4 | 8.34 (+18%) | 0.9985 (passed) | slower |
   | flashinfer-b12x W4A4 | 2.70 (-62%) | 0.82 | dead (round 1) |

   w4a8 loses uniformly (2.61 @ 2K vs 2.04; 16.15 @ 16K; 31.62 @ 32K vs
   27.0), GPU-event-verified kernel-bound (8.59 ms events vs 8.34 wall, the
   1.55 ms/call helper overhead is not the story). The scouts' "half
   Marlin's latency" projection came from a lab with zero 5090 numbers —
   now the 5090 numbers exist, and they say otherwise at our geometry.
4. **Verdict: ROUND 2 CLOSED, NEGATIVE for large-M prefill.** Marlin keeps
   the crown a second time. Bank v3 / streamer switch / serve A/B are moot.
   The process rule paid: quality gates ran on CPU during the matrix, the
   speed kill landed in ~30 min of GPU. Durable salvage: (a) w4a8 quality
   0.998+ on real weights is a standing result — any future faster sm_120
   w4a8 kernel inherits a ready gate, byte-verified swizzle adapters, and
   the up-first bank recipe; (b) the small-M-direct decode candidate (M<=8)
   is untouched by this verdict — different regime, still queued for the
   decode war; (c) hotness-ordered placement remains available on its own
   merits (fewer streamed bytes + less remote compute, kernel-agnostic)
   but is gated on an explicit go decision.

## Measured facts

Correctness gate **closed**: 200 alternating replays through the production
graph path, worst error vs independent CPU dequant 2.15e-09, minimum
discrimination ratio 1,152,510x, 0/200 non-exact CUDA+B70 additions. Freshness,
flag ordering, buffer selection and graph-path addition are all verified.

Serving, 128-token prompt / 512 forced output tokens, GuideLLM, `ignore_eos`,
0 errors, every rung KV-fitting:

| requested C | out tok/s | TTFT mean / p95 | ITL ms |
|---|---|---|---|
| 1 | 57.5 | 505 / 525 ms | 16.52 |
| 2 | 80.4 | 857 / 1003 ms | 21.92 |
| 4 | 125.6 | 1705 / 1843 ms | 29.11 |
| 8 | 157.5 | 3094 / 3577 ms | 40.22 |
| 16 | **173.5** | 6137 / 7084 ms | 69.43 |
| 32 | 153.5 | 44.3 s | 95.43 |
| 62 | 129.7 | 135.4 s | 99.11 |

Peak 173.5 tok/s at C=16; C=32 and C=62 are Pareto-dominated (less throughput
*and* worse latency). Scoped to this 128-in/512-out shape.

Server geometry: 262,144 KV tokens at `max_num_seqs=64`; attention block padded
to **4,176 tokens** to match the GDN/Mamba page, so capacity is **62 seats** and
a 640-token request still costs a full block. Raising `max_num_seqs` 8 -> 64 cost
0.36 GiB of KV (292,103 -> 262,144 tokens).

Dispatch cost model (standalone, real geometry, `-1`-padded):

| k | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| kernel µs | 8.1 | 30.8 | 32.2 | 33.6 | **32.6** | 37.1 | 56.6 | 72.3 | **79.9** |
| wall−kernel µs | 110 | 110 | 109 | 109 | 113 | 108 | 108 | 108 | 108 |

Kernel cost is flat k=1..4 then explodes; the ~108 µs floor is constant, so it is
7 queue submissions rather than bandwidth.

Prefill TTFT vs context, C=1 (per-request; whole-cell wall time is the wrong
discriminator because each cell also decodes 512 tokens x N requests, and N
drops from 20 to 8 above 16K):

| ctx | input tok | TTFT mean | p95 | **ms per prompt token** | scaling |
|---|---|---|---|---|---|
| 128 | 140 | 474 ms | 533 ms | 3.385 | — |
| 512 | 524 | 1,689 ms | 1,759 ms | 3.224 | 3.56x for 3.74x tok |
| 2,048 | 2,060 | 6,032 ms | 6,568 ms | 2.928 | 3.57x for 3.93x tok |
| 8,192 | 8,204 | 21,864 ms | 25,556 ms | 2.665 | 3.62x for 3.98x tok |
| 16,384 | 16,396 | 43,762 ms | 52,702 ms | 2.669 | **2.00x for 2.00x tok** |
| 32,768 | 32,780 | 84,246 ms | 88,874 ms | 2.570 | **1.93x for 2.00x tok** |

**Prefill is exactly linear in context over a 234x range.** The TTFT ratio tracks
the token ratio precisely, and at 32K falls slightly below it. So attention
contributes nothing measurable *as a scaling term* through 32K — if it did, the
ratio would exceed the token ratio. Cost is per-token work that is constant in
context.

This rules out attention as a scaling term. It does **not** attribute the cost
within a layer; that still needs a per-layer MoE-vs-rest profile, and linearity
says nothing about attention as a constant.

**Prefill runs at 389 tok/s** against decode's 57.5 tok/s — only 6.8x, for a
3B-active model where prefill has thousands of rows of parallelism and decode has
one. At 32K that is **84 s to first token**; linear extrapolation puts a 128K
prompt near **5.6 minutes**. This is the largest single deficit in the system,
larger than the 2.35x decode gap vs the PRO 6000, and it is what a long-context
user feels first.

## Head-to-head vs a single RTX PRO 6000 Blackwell 96 GB

Same checkpoint (`srswti/axe-superveloce-88b-nvfp4a16`), same GuideLLM harness,
same `output_tokens=512` with `ignore_eos`, `/v1/chat/completions`, synchronous
C=1, 20 requests per cell. Validated per row: both sides report exactly 512
output tokens and matching input token counts. **Measured N** is
`metrics.request_totals.successful`, not the retained sample list: 20 requests
were issued per cell but transient trimming leaves N=14 at 8K and 16K, and N=6 at
32K (SB issued 8 there). The 32K row is therefore directional.

| ctx | PRO TTFT | **SB TTFT** | gap | PRO ITL | **SB ITL** | gap |
|---|---|---|---|---|---|---|
| 8,192 | 563 ms | 21,864 ms | **38.9x** | 7.34 ms | 13.29 ms | **1.81x** |
| 16,384 | 1,206 ms | 43,762 ms | **36.3x** | 7.45 ms | 13.14 ms | **1.76x** |
| 32,768 | 2,783 ms | 84,246 ms | **30.3x** | 7.59 ms | 13.02 ms | **1.72x** |

Effective cold-prefill rate, derived as `input_tokens / TTFT` — **not** a
GuideLLM or server throughput figure, since TTFT also carries HTTP, scheduling
and chat-template cost: PRO **11,778-14,576 tok/s** vs SB **375-389 tok/s**.

**Decode is within 1.8x** while serving 126 of 180 experts per layer across a
Gen4 x4 link (6.24 GB/s) against the PRO's local GDDR7 (~1.8 TB/s). The doorbell
graph path, the int4 kernel at 68.3% of roofline, and the CUDA-partial overlap
are all doing their job.

Note this corrects a figure repeated earlier in the project: the decode gap is
**1.8x, not 2.35x**. The 57.5-vs-135 comparison used per-request output tok/s,
which amortizes TTFT into the number. On pure inter-token latency we are much
closer.

**Prefill is 30-39x behind and it is our problem, not the model's** — the same
checkpoint on the same vLLM on one CUDA card reaches 14,576 tok/s. We are slower
at 32K (84.2 s) than the PRO 6000 is at 150K (51.2 s).

**Leading hypothesis, not a finding.** The `O(M x top_k)` weight re-reading in the
per-route kernel remains the best explanation for the gap, and the independent
~40x estimate landing near the measured 38.9x is suggestive. It is not proof:
that estimate assumed ~100 TFLOP/s achievable (unmeasured), the PRO run's server
launch settings are **not matched** to ours (`max_num_batched_tokens`,
`gpu_memory_utilization`, `max_num_seqs`, attention backend all unknown here), and
no per-layer trace has attributed any fraction of the 30-39x. Two numbers
agreeing can be coincidence. The per-layer prefill profile is still the gate
before committing to a kernel.

**Their curve shows what ours cannot.** PRO ms/token: 0.069 (8K) -> 0.113 (64K)
-> 0.265 (98K) -> 0.342 (150K). The quadratic attention term becomes visible from
~64K. Our curve looks flat only because a 30-39x constant buries it, so the
earlier claim that attention contributes "nothing measurable" was wrong — on a
fast enough device it plainly does.

### Long-context, from the completed cells

| cell | measured N | TTFT | ITL |
|---|---|---|---|
| ctx 65,536 C=1 | 6 | **185 s** | 13.85 ms |
| ctx 128,940 C=1 | 4 | **335 s** | 12.45 ms |
| ctx 65,536 C=3.56 | 7 | 401 s | 482 ms (flagged QUEUE) |

The linear model held: 335 s measured at 128,940 tokens against a 336 s
projection. ITL stays 12-14 ms even at 128K context, so the decode path is
insensitive to context length while prefill is not.

### Priority consequence

* **Decode: 1.8x behind.** Graph replay, persistence, dual-B70 all target this.
  Tens of percent each, diminishing.
* **Prefill: 30-39x behind**, single identified cause, and the fix is a
  known-shape CUDA kernel rather than novel ESIMD work.

Time spent on the 108 µs dispatch floor is time not spent on a 30x gap.

## SHIPPED: Marlin prefill (phase-disaggregated prefill on the 5090)

`src/phase4/src/shooting_brake_vllm/marlin_prefill.py`, opt-in via
`SHOOTING_BRAKE_PREFILL_MARLIN=1`. During prefill, each layer's 126 remote
experts stream from the mmap'd int4 bank (host DRAM, 18.5 GiB/s) to the 5090,
split-repack bit-exactly into persistent Marlin buffers, and run through
vLLM's `fused_marlin_moe` (`uint4b8` == the bank's AutoGPTQ sym/g128/zp8
contract). Decode keeps the B70 doorbell untouched: same int4 weights in both
phases, so the served model is numerically unchanged.

Measured (same GuideLLM harness/seeds as baseline; config: MNBT=8192, MNS=8,
Gen4 card, KV 194,735 tokens -- the streamer's persistent buffers cost ~67K KV
tokens of budget):

| ctx | B70-dispatch baseline | Marlin prefill | vs PRO 6000 |
|---|---|---|---|
| 8,192 | 21.86 s | **3.07 s** (p95 3.14, N=14) | 5.5x (was 38.9x) |
| 32,768 | 84.25 s | **13.10 s** (p95 13.12, N=6) | 4.7x (was 30.3x) |

Decode ITL in the same runs: 11.5-11.7 ms (doorbell path untouched).

Correctness, three levels: (1) tensor vs independent CPU fp32 dequant of real
bank planes, cosine 0.999985 (`benchmarks/results/marlin_poc/poc.json`);
(2) live prompt-logprob A/B marlin-on vs marlin-off, mean delta **+0.0047
nats/token** (mean |d| 0.053, max 0.28; historical-bug signature was 0.49
systematic) -- `benchmarks/results/prefill_profile/marlin_logprob_ab.json`;
(3) greedy A/B: divergent-but-fluent, consistent with kernel numerics.

Integration scars, recorded so they are never re-fought: Marlin silently
returns zeros when scale dtype differs from activation dtype; six boot OOMs
were caused by vLLM's compiled forward tracing the streamer (AOT
functionalization retains copies of every touched tensor) and were fixed by
the custom-op boundary (`direct_register_custom_op`), NOT by trimming buffers;
`@torch.compiler.disable` does not escape vLLM's compile pipeline; the in-situ
VERIFY branch cannot live inside a traced region (control flow + side effects
bake at trace time) -- black-box logprob A/B replaces it.

Next headroom, in order: offline pre-repacked Marlin bank (removes 27 ms/layer
repack -> ~2.1 s @ 8K); threaded pinned staging toward the 52.8 GiB/s DMA
ceiling (~1.4 s @ 8K); recover the 67K KV tokens (repack buffers could alias
the fused kernel's workspace); EPLB-based expert placement (llm-d alpha) to
shrink both phases' remote sets. Cleanup owed: remove the dead VERIFY branch
in `routed_experts.py`, make MNS/KV trade explicit in serve script comments.

## The gate that blocks most of the roadmap

Per-layer cost is `max(B70 round trip, CUDA MoE partial)` — the graph path
deliberately overlaps them. **Which leg binds is unmeasured.** If the CUDA
partial is the longer leg, then graph replay, persistent kernels and dual-B70
all save exactly zero end-to-end.

No per-dispatch microbenchmark may be multiplied by 48 layers and subtracted
from TPOT. That arithmetic produced four withdrawn claims this session
(`173 µs/layer`, `364 µs/layer`, MTP tok/s forecasts, persistence "24%").

## Ordered plan

1. **Prefill measurements before prefill code.**
   (a) **DONE — slab H2D ceiling**, reproducible harness
   `benchmarks/slab_h2d_bench.py`, artifact
   `benchmarks/results/slab_h2d/slab_h2d.json` (raw timings, CUDA-event H2D,
   fault pass, NUMA snapshot). One layer slab (126 experts, 584.7 MiB) from the
   real `expert_bank_int4.bin` at its real offset:

   | phase | GiB/s | 48-layer forward (27.4 GiB) |
   |---|---|---|
   | pageable mmap -> GPU | **18.54** | **1.48 s** |
   | mmap -> pinned stage (1 thread) | 15.71 | — |
   | pinned -> GPU DMA (CUDA events) | **52.78** | 0.52 s upper bound |
   | e2e serialized stage+H2D | 12.30 | 2.23 s |
   | double-buffer bound min(stage,h2d) | 15.71 | 1.74 s |

   Notable: **plain pageable beats both the serialized and the double-buffered
   pinned pipeline**, because CUDA's pageable path already pipelines through its
   own staging buffers. The simplest implementation is currently the fastest;
   reaching toward the 52.8 GiB/s DMA ceiling requires multi-threaded staging,
   ~3x headroom. Cold-disk is a non-issue (bank page-caches; fault pass 9 ms
   cold -> 0.9 ms warm). Conservative cost: 32K prompt at 8K chunks =
   4 x 1.48 s ~= 6 s transfer vs 84 s TTFT today. **Design survives its first
   kill condition.** The H2D-alone figure is an upper bound no pipeline
   reaches, not a pipeline result.
   (b) **DONE — prefill attribution.** Artifact
   `benchmarks/results/prefill_profile/attribution_8k.json`, trace + capture
   alongside. One profiled 8K prefill on the production config: **GPU idle 97.0%
   of the 27.7 s span** (26.9 s waiting on the B70); ALL 5090 work -- Marlin MoE
   0.28 s, attention 0.08 s, GDN 0.07 s, transfers 0.14 s -- totals **0.82 s**.
   ~140 ms per M=2048 B70 dispatch, same order as the DPAS artifact's 58.9 ms on
   the 3x smaller 35B shape. Prediction was >=80%; measured 97%. **Streaming
   design GO.** Cross-check: 5090 compute scaled to all 180 experts ~= 1.5 s vs
   1.48 s measured transfer floor -- balanced max() ~= 1.5 s TTFT @ 8K target.
   Caveat: B70 profiling markers add ~11% wall (21.86 unprofiled -> 24.27 s);
   the shares are the result, not the absolute times.
   (c) **Activation budget**: largest chunk that fits beside weights + KV + two
   slab buffers. Startup log: 2.51 GiB peak activation @ 8,192 tokens, 1.59 GiB
   CUDA graphs — so 8K chunks are the likely operating point, not 32K; a 32K
   single forward does NOT fit the current budget.
2. **Trace one decode step.** CUDA + SYCL timeline showing the B70 round trip and
   the CUDA MoE partial side by side. Resolves the max() question, gating graph
   replay / persistence / dual-B70.
3. **Route trace -> hotness-ordered placement.** `split:54` is an arbitrary
   *index* range, but the router is skewed (9 all-CUDA steps in 6,144 vs 0.40
   predicted under uniformity, 22x chance). Placement should be hotness-ordered.
   Existing machinery (`ExpertGroupPolicy`, `FractionalRemotePolicy`), no new
   kernels. ~5.9 GiB spare 5090 VRAM ~= 25 more experts/layer at 4.87 MB each.
   Changes the k distribution every kernel decision is priced against, so it
   precedes all kernel work.
4. **Verification-mechanism probe (two arms).** `ngram`/`suffix` speculative
   decoding needs no draft weights. Arm A on repetition-heavy prompts forces
   acceptance high to measure whether multi-token verification helps this
   hardware *at all* (upper bound only — a real proposer costs more). Arm B on
   representative prompts measures n-gram's actual value. **A low-acceptance
   Arm B must not be read as evidence against MTP.**
5. **SYCL graph replay into provider `issue()`.** Built and validated: k=1
   105.1 -> 75.3 µs, k=4 114.1 -> 83.6 µs, 7.63e-08 vs eager, 1.13e-06 vs CPU
   oracle. k=8 unverified (eager/graph shared a queue, inflating eager to 365 µs
   vs 188 µs standalone; isolated rerun written, unrun). Gated on item 3.
6. **Persistent B70 kernel — R&D, not a task.** Removing host submissions means
   *fusing* `zero_output`/`gate_up`/`down` into one resident kernel with a
   device-wide barrier between phases, plus a cross-vendor host-mapped doorbell,
   plus watchdog/recovery. Probe first, and only two things: host-mapped atomic
   visibility with forward progress on Xe2, and a working cross-workgroup
   barrier. If either fails the idea dies cheaply.
7. **Second B70 (k<=4 per card).** Banks built (`dev0` 12-95, `dev1` 96-179);
   split math proven at 1.506e-07. Benefit is entirely k=8 -> k=4 (79.9 -> 32.6
   µs), so if item 3 drops typical k to 2-3 the second card buys ~nothing. Must
   follow item 4, and needs the shared `09:00.0` Gen4 x4 uplink measured under
   concurrent load first.

## Prefill: the mechanism

**We run a decode kernel on prefill shapes.** Our B70 MoE kernel does
`O(M x top_k)` weight reads; a CUDA grouped/sorted MoE GEMM does `O(E)`. From
`nvfp4_moe_grouped.md` at M=2048, one layer: `split` (per-route, production)
issued **24.6 GiB** of logical weight reads against `grouped`'s **861 MiB** — 28x.

At M=1 per-route is *optimal*: you read exactly the 8 experts you need. At
M=8192 it re-reads the same expert hundreds of times. This is the same 28x that
separates us from an all-CUDA device, which reads each expert once per chunk.

Note the asymmetry inside our own layer: **the CUDA side already uses the grouped
path** (`ModelOptNvFp4FusedMoE` via Marlin), so the 54 local experts are
efficient; the 126 on the B70 are not. Per-layer cost is
`max(5090 leg, B70 leg)`, which yields a falsifiable prediction: **the B70 leg
dominates prefill and the 5090 idles**. If the per-layer profile shows otherwise,
this whole model is wrong.

Scale of the gap: 3B active params is ~6 GFLOP/token, so 32,768 tokens is ~196
TFLOP. A 5090 at ~100 TFLOP/s achievable would take ~2 s. We measure 84 s —
**~40x off**, which is structural rather than a tuning gap.

### The streaming alternative, with its real costs

Sending prefill to the 5090 instead of the B70 is the obvious response, but three
things make it new code rather than a flag, and an earlier draft of this file got
all three wrong:

1. **The current streamer is NVFP4-only.** `_load_host_experts_from_bank` imports
   the legacy `ExpertBank` (`SBEXP001`) reader; `ExpertStreamer` hardcodes NVFP4
   packing (`_block_bytes = 3*(plane/2 + plane/16)`) and dequant
   (`e2m1 x e4m3 x gscale`, `block_size=16`). Pointed at `SBINT401` it fails at
   load, so `SHOOTING_BRAKE_B70_PREFILL_STREAM=1` is **not** a runnable A/B today.
2. **It does not feed the grouped Marlin path.** It dequantizes each expert and
   calls generic `F.linear`. A **transient-weight grouped int4 CUDA MoE kernel**
   does not exist and would have to be written.
3. **Transfer is per *forward*, not per prompt.** 126 experts x 4.87 MB x 48
   layers = ~29.5 GB per forward, flat in M. With
   `max_num_batched_tokens=8192` a 32K prompt is **four** forwards. So the
   governing formula is
   `bytes_per_forward / rate x ceil(prompt / max_num_batched_tokens)` —
   streaming only wins with **few, large** forwards.

**Measured (was "unverified input" in an earlier draft):** the host->5090 path
was benchmarked directly — see the slab table in the ordered plan and
`benchmarks/results/slab_h2d/slab_h2d.json`. The previously cited ~6.2 GiB/s
does not describe this machine's host->5090 path (pageable measures 18.54
GiB/s). Code inspection had already established the streamer sources from the
host arena (`_host.expert_block` -> CUDA directly); the B70 is not in that path.

Design if it survives measurement: mmap `SBINT401` (29.4 GB, fits page cache in
~47 GiB free RAM), stage a contiguous layer slab into pinned host memory, overlap
layer N+1 H2D with layer N compute, and run a transient-weight grouped int4 CUDA
MoE kernel. Benchmark standalone before any PRO 6000 comparison.

## Prefill: prior art and adaptation cost

**A grouped/DPAS MoE kernel has already been tried here and lost.**
`src/QuixiCore-XPU/perf/results/nvfp4_moe_grouped.md`, M=2048, f16, NVFP4:

| variant | median | µs/token | logical rate |
|---|---|---|---|
| `split` (per-route, production) | 58.9 ms | 28.8 | 437.6 GB/s |
| `grouped` (DPAS) | **91.1 ms** | 44.5 | 9.9 GB/s |

1.55x slower while issuing ~28x **fewer** logical weight reads. Numerically
correct (output L1 94818.6 vs 94819.6). Time is entirely the two GEMMs
(`gate_up` 78.1%, `down` 21.5%); the sort prepass is 0.2%. It is gated off
unconditionally and no provider path selects it.

That artifact also names the exact reasoning error to avoid: *"`split` is
bandwidth-bound at this shape, so cutting traffic is the only lever that can help
— but that argument predicted this kernel would win, and it did not."*

Note the 9.9 GB/s is a **modeled logical** rate (`pairs * expert_bytes`), not a
hardware counter, so it is not evidence of a bug — after removing 28x redundant
reads a compute-bound GEMM naturally shows a lower bytes/time ratio. The honest
description is severe DPAS/loop underutilisation with the cause **unlocalised**;
the artifact records that three ablation attempts were degenerate, so no valid
bound exists on dequant vs loop structure. Localising it needs EU/XMX occupancy
and memory counters.

**What keeps the Intel ESIMD arm worth one smoke** — it differs on all three
axes that could matter:

* format: **NVFP4** there, **int4** here
* geometry: 35B shape (K=2048, I=512, E=256) there; 88B is K=3072, I=1024, E=180
* implementation: a home-grown grouped kernel there vs Intel's 1,040-line ESIMD
  kernel at `vendor/intel-xpu/llm-scaler-latest/{vllm,sglang}/custom-esimd-kernels*/csrc/moe_prefill/moe_prefill_int4.sycl`,
  which uses explicit `sycl::ext::intel::esimd::xmx` intrinsics, a VNNI a_tile,
  and a gather/sort prepass.

Adaptation notes from reading the source (**not** from running it):

* `unshuffle[8] = {0,2,4,6,1,3,5,7}` is scoped to *nibble order within one int32
  row* (`shift = unshuffle[k] * 4`), so verbatim AutoGPTQ packing is
  `unshuffle[k] = k` — a constexpr change if that is the only permutation.
* `KP_PER_GROUP = BS/8` already matches our group_size 128.
* The real work is the **operand layout**, and it is not drop-in. The up kernel
  hardcodes a fused contiguous `[E, K/8, 2I]` base (`eid * K_packed * 2I`) and
  computes `SiLU(gate) * up` inside **one** invocation. SBINT401 is AoS records
  with six separate planes and an `expert_stride`. Two separate calls **cannot**
  reproduce that fused op, because the SwiGLU multiply needs gate and up live in
  registers together. The cheap route is to change the ESIMD signature and
  pointer math to take separate gate/up qweight+scale pointers plus the expert
  stride (and likewise for `down`), keeping the fusion intact, then
  correctness-test against the current SBINT401 kernel. The alternative is a
  repacked bank, which the streaming builder could produce but which doubles
  bank maintenance.

## Cheap hygiene, not strategy

* **bf16 instead of fp32 on D2H.** We ship 12,288 B/row and immediately downcast
  to bf16 on CUDA. At M=1 the whole 18.5 KB round trip is ~4.7 µs on Gen3 x4 —
  noise against a 108 µs floor. Only material at high concurrency, where we
  should not be operating. Real waste, small win.
* ~~Verify which card we are on~~ — **REOPENED and answered: we are on the SLOW
  card.** `SHOOTING_BRAKE_B70_DEVICE=1` is a *boolean enable*
  (`routed_experts.py:417,962`), not a device index; `select_b70()` takes the
  first enumerated L0 GPU = `11:00.0` = **Gen3 x4** (nvtop: the loaded 28.5 GiB
  card runs Gen3@4x, the Gen4 card idles). Every measurement to date -- 60-65
  tok/s, Grid A/B, the 1.72-1.81x decode gap -- was on the slower B70. Fix is
  zero code: `ZE_AFFINITY_MASK=1` hides device 0 so the Gen4 card enumerates
  first. **A/B MEASURED** (`benchmarks/results/prefill_profile/gen4_*.json`,
  ZE_AFFINITY_MASK=1 confirmed to expose only 15:00.0):

  | metric | Gen3 (all prior data) | Gen4 | PRO 6000 | new gap |
  |---|---|---|---|---|
  | decode ITL | 16.52 ms | **11.3 ms** (2 probes: 11.31/11.23) | 7.34 | **1.54x** |
  | decode tok/s C=1 | ~60-65 | **~88** | ~135 | 1.53x |
  | TTFT @ 8K | 21.86 s | **18.36 s** (n=1) | 0.56 s | 32.6x |

  The decode saving is 5.21 ms / ~48 dispatches = **108.5 us per dispatch --
  the size of the entire wall-minus-kernel floor**, falsifying the
  bandwidth-only sizing above (~3 us): the floor is largely link-LATENCY.
  Open question, needs its own measurement: what remains of the 108 us floor on
  the Gen4 link (the standalone dispatch experiments presumably also ran on the
  first-enumerated Gen3 card). Consequences: production must set
  ZE_AFFINITY_MASK=1 (or a BDF knob in select_b70); dual-B70 economics worsen --
  the second card sits on the Gen3 slot and would drag the slow link back in
  for half the routes. An earlier draft of this entry claimed "already optimal"
  from a standalone SYCL index probe -- enumeration context is not transferable
  between processes. Endpoint sysfs
  fields read 2.5 GT/s x1 on both B70s and are idle-parked/misleading; the
  binding link is the narrowest ancestor bridge. `11:00.0` sits under `0a:04.0`
  at 8.0 GT/s x4 (**Gen3 x4**, 2.91 GB/s measured H2D); `15:00.0` sits under
  `0a:08.0` at 16.0 GT/s x4 (**Gen4 x4**, 6.24 GB/s). SYCL enumerates
  `11:00.0` as index 0 and `15:00.0` as index 1. Both share
  `09:00.0` (16 GT/s x4) upstream, which is the contention risk for dual-card.
* **Prefix caching is disabled** on this server (`enable_prefix_caching=False`),
  almost certainly because the hybrid GDN/Mamba cache does not support it. So
  there is no "free warm-prefix win" to claim, and the suite's per-cell seeds are
  defensive rather than load-bearing. Enabling it would be its own correctness
  project, not a configuration change.
* **Admission control.** Capping `max_num_seqs` limits *running* sequences, not
  accepted ones, so excess callers queue and their TTFT moves into the queue.
  Needs gateway-level bounded queue or rejection, and the operating point should
  be chosen from p95 TTFT, not peak throughput.

## Closed / not available

* **MTP head: declared in config, absent from both quantized checkpoints.**
  `text_config.mtp_num_hidden_layers: 1` and vLLM registers `Qwen3_5MoeMTP` /
  `qwen3_next_mtp`, but the safetensors index has **0 MTP tensors** in
  *both* derivatives — nvfp4a16 (0/79,430) and int4 (0/79,116) — across 10
  naming conventions (`mtp`, `draft`, `nextn`, `eh_proj`, `enorm`, `hnorm`,
  `shared_head`, ...), and layer indices are exactly 0-47 with only `model` and
  `lm_head` prefixes. Upstream Qwen3.5 checkpoints reportedly *do* ship
  `mtp.*` keys, so the quantization step dropped them.
  **This is not closed, it is a build:** `vendor/speculators` contains the full
  MTP training stack (`src/speculators/models/mtp/`, `convert/mtp_converter`,
  e2e conversion + finetuning + acceptance tests). Options are to re-extract the
  head from an upstream unquantized checkpoint, or train one. Field flags before
  relying on it: a reported TP=2 `cudaErrorIllegalAddress` crash on hybrid-GDN
  Qwen3.6, and acceptance collapsing 61% -> 0% across a multi-turn tool-calling
  session on Qwen3.5-35B-A3B.
* **Expert prefetch: nothing to prefetch.** All 180 experts per layer are
  resident (54 CUDA + 126 B70); zero weight bytes move at decode time.
* **Route fusion, epoch-tagged CAS `zero_output`**: rejected with documented
  negative results (`perf/results/int4_moe_route_fusion.md`).
* **2:4 sparsity**: zeros are unstructured; would need a fresh prune+requantize.

## Benchmark suite notes

`benchmarks/bench_88b.py` drives GuideLLM only; it decides cells and records
provenance. Traps already hit and guarded:

* GuideLLM exits **0** on an empty measurement window. At `max_requests=2` with
  percentage transients every metric was 0.0 and `duration` was -0.0004 s. Cells
  below 12 requests therefore run with transients off, and `_measurement_is_real`
  fails any cell with no successful requests or non-positive duration.
* Percentiles live under `metrics[k].successful.percentiles.p95`, not at the top
  level — asking for `p95` directly returns None and silently blanks the column.
* Per-cell deterministic seeds (`zlib.crc32`, not `hash()`) so cells >= the
  4,176-token block cannot hand each other cacheable prefixes. Grid E keeps a
  shared seed deliberately, because the thinking A/B needs identical prompts.
* `client_concurrency` is GuideLLM's client-side mean in-flight count, not proof
  the server admitted that many sequences. Capacity is judged per row.
* Still to fix: `ok/err` counts warmup/cooldown requests while the statistics
  beside them use only the measurement window (C=62 shows 124 vs 109 measured).
  Switch to `metrics.request_totals`.
-------------------------------------------------------------------------------------

research you wanted my help in here is what i gathered :

Good news on all three — real primary-source material for each, and the picture on #2/#3 turned out more interesting than expected. Findings below with citations.

## 1. Level Zero / SYCL forward-progress on Xe2

The short version: **there is no published, architecture-specific occupancy/forward-progress guarantee for Xe2** — and that's not a documentation gap, it's the actual design of the spec. Here's the chain of evidence:

- **Baseline (no cooperative launch):** standard SYCL `nd_range` execution gives only *weakly parallel* forward progress. Intel's own SYCL workshop docs are explicit: work-items in a work-group are scheduled concurrently to a single compute unit, but there can be many more work-groups in an ND-range than compute units, and work-items are not guaranteed to make independent progress — interleaving execution with barriers, effectively running sequentially, is a valid execution model the runtime may choose. SYCL 2020 itself doesn't expose a way to request or query anything stronger.

- **The actual mechanism for cross-workgroup sync:** Level Zero's answer to this is the cooperative-kernel path — query the maximum cooperative group count via zeKernelSuggestMaxCooperativeGroupCount, assert your launch is under it, then append via zeCommandListAppendLaunchCooperativeKernel on a queue with the COOPERATIVE_KERNELS group flag set. That max is a **runtime query result**, not a spec constant — it depends on the kernel's register/SLM footprint and the specific device.

- **SYCL layers this via a proposed extension** (`sycl_ext_oneapi_root_group` + `sycl_ext_oneapi_launch_queries`), which is very explicit about what's guaranteed and what isn't: Level Zero and CUDA both support cooperative kernel launches for cross-workgroup sync, but the semantics aren't identical — CUDA gives stronger forward-progress guarantees for CUDA threads, so the extension deliberately avoids exposing "cooperative launch" directly and instead abstracts it behind a root-group query. The intel/llvm implementation notes say plainly: a Level Zero implementation would launch root-sync kernels via zeCommandListAppendLaunchCooperativeKernel and query limits via zeKernelSuggestMaxCooperativeGroupCount — so the SYCL abstraction is just a thin wrapper over exactly the L0 calls above.

- **The universal floor, stated as policy, not per-SKU data:** both the SYCL extension and the Unified Runtime spec independently state the same fallback contract. UR: any device can support cooperative kernels by restricting the maximum number of workgroups to 1; devices with real cross-workgroup sync can specify a larger maximum per kernel. The SYCL doc says the same thing from the other direction: if a device doesn't natively support cooperative launch, an implementation can always fall back to a trivial one — kernels using root-group sync restricted to at most one work-group.

**Implication for your probe:** what you're empirically finding (some N workgroups reliably co-resident) *is* the documented mechanism working as intended — it's just that the mechanism is explicitly a runtime query, not a static number anyone publishes per-architecture. I didn't find an Intel doc enumerating whether Battlemage/Xe2 client parts (vs. PVC datacenter parts) expose `COOPERATIVE_KERNELS` at all — that's worth checking directly via `zeDeviceGetCommandQueueGroupProperties` on your actual device rather than searching further, since I couldn't find a per-SKU support table anywhere in Intel's public docs.

## 2 & 3. vLLM 0.27.1 spec-decode + does Qwen3.5-MoE have an MTP/EAGLE head

**vLLM 0.27.1 itself** is a single-commit patch on 0.27.0: "Support quantized DSpark Markov heads" (#50424) — quantization plumbing for a specific draft-head class, not new MTP wiring (that already existed). More on DSpark below since it's relevant to your question.

**Yes — Qwen3.5-MoE ships a native MTP head, and it's the Qwen3-Next mechanism reused, not a new one.** This is now solidly triangulated:

- Every official Qwen org model card I checked (2B, 4B, 27B, 35B-A3B, and the 3.6 line) gives the identical recommended launch: vllm serve Qwen/Qwen3.5-2B ... --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'.
- vLLM's own config source confirms this isn't a special case — it's one entry in a family: MTP_MODEL_TYPES = ("deepseek_mtp", "mimo_mtp", "glm4_moe_mtp", "ernie_mtp", "qwen3_next_mtp", "longcat_flash_mtp"), each a native per-architecture MTP loader.
- The method name and mechanism predate Qwen3.5 — it was built for Qwen3-Next itself back in October: vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct ... --speculative-config '{"method": "qwen3_next_mtp", "num_speculative_tokens": 2}' — and simply carried forward unmodified, confirming exactly the reuse-across-the-hybrid-family you suspected.
- The weights are physically in the base checkpoint, not a separate download — a community repo splitting them out confirms it directly: the MTP draft head extracted from the official Qwen/Qwen3.5-2B checkpoint, extracted verbatim by reading the two safetensors shards that contain mtp.* keys per model.safetensors.index.json.
- vLLM-Ascend's spec-decode guide groups the whole family together explicitly: for Hybrid Mamba models (e.g., Qwen-Next and Qwen3.5 series), num_speculative_tokens should be equal on P and D nodes — again, same mechanism, same terminology as Qwen3-Next.

**So: no header scan needed** — `--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'` is the answer, and it's auto-resolved (draft weights come from the target checkpoint itself, per the "Automatically resolved for mtp (reuses target model)" note in the vLLM-Ascend docs).

**EAGLE — not native.** vLLM's eagle3 path takes hidden states from three verifier layers and is architecture-general, but it requires its own separately-trained draft weights passed via `"model": "<repo>"` — there's no Alibaba-published EAGLE head for Qwen3.5 that I found. What exists is community-trained (e.g. Red Hat's P-EAGLE speculator for Qwen3-8B), via the vllm-project/speculators training framework which supports Eagle3 for "diverse model architectures" generally, not as a per-model official artifact.

**A third thing you'll run into: DSpark.** This is a newer, distinct speculative-decoding architecture (DeepSeek-originated — a low-rank "Markov head," optionally paired with a DFlash block-diffusion drafter) that vLLM merged into mainline mid-2026 and which 0.27.1's one commit touched. It is **also external/separately-trained**, like EAGLE, not baked into the base checkpoint — you pass `"method":"dspark","model":"<draft-repo>"`. It's positioned as a competitor to both native MTP and EAGLE for these hybrid-GDN bodies, with community-trained drafters already circulating for Qwen3 and Qwen3.6 targets.

Quick comparison of the three paths for a Qwen3.5/Qwen3-Next-class target:

| | Native MTP | EAGLE(3) | DSpark |
|---|---|---|---|
| Weights | In target checkpoint | Separate draft repo, required | Separate draft repo, required |
| vLLM method | `qwen3_next_mtp` | `eagle3` | `dspark` |
| Draft model field | Auto-resolved | Required | Required |
| Publisher for Qwen3.5 | Alibaba (official) | Community only | Community only |

**Two field-reliability flags worth knowing before you commit to `qwen3_next_mtp` in production:** an open bug reports a hard crash at TP=2 (`cudaErrorIllegalAddress` in `gpu_model_runner.py`'s accepted-tokens sync) on hybrid-GDN Qwen3.6, and a separate report shows acceptance rate collapsing from 61% to 0% across a multi-turn tool-calling session on Qwen3.5-35B-A3B with the same config. Worth testing your exact TP/turn pattern before relying on it.

Here's the curated list, plus what's still open.

## Most worth bookmarking

**Topic 1 (Level Zero / SYCL forward progress)**

| Link | Why it's the one to keep |
|---|---|
| [sycl_ext_oneapi_root_group.asciidoc](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/proposed/sycl_ext_oneapi_root_group.asciidoc) | The actual answer to your question — states the L0/CUDA guarantee mismatch explicitly and gives the implementation-notes mapping to `zeKernelSuggestMaxCooperativeGroupCount` / `zeCommandListAppendLaunchCooperativeKernel` |
| [Level Zero Core Programming Guide](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/PROG.html) | Primary spec — "Cooperative Kernels" and "Events" sections are the relevant parts |
| [sycl_ext_oneapi_launch_queries.asciidoc](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/proposed/sycl_ext_oneapi_launch_queries.asciidoc) | The companion query API (`max_num_work_group_sync`) — this is what you'd call instead of hardcoding a probed number |
| [UR EXP-COOPERATIVE-KERNELS](https://oneapi-src.github.io/unified-runtime/core/EXP-COOPERATIVE-KERNELS.html) | Short — has the one-line universal floor ("any device can restrict to 1") |
| [Intel GPU Occupancy Guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/thread-mapping-and-gpu-occupancy.html) | Practical Xe-Core occupancy math, useful for reasoning about what a probed number *should* look like on your hardware |

**Topics 2/3 (vLLM spec-decode / Qwen3.5 MTP)**

| Link | Why it's the one to keep |
|---|---|
| [`vllm.config.speculative` source](https://docs.vllm.ai/en/v0.11.0/api/vllm/config/speculative.html) | `MTP_MODEL_TYPES` and `SpeculativeMethod` — ground truth for what methods exist, better than any blog post |
| [Qwen/Qwen3.5-35B-A3B model card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) | Official, matches your target size class — has the exact serve command |
| [vllm-project/speculators](https://github.com/vllm-project/speculators) | The training framework repo — README changelog is the best single place to track EAGLE/DFlash/DSpark status for Qwen-family targets going forward |
| [vLLM-Ascend spec-decode guide](https://docs.vllm.ai/projects/ascend/en/main/user_guide/feature_guide/speculative_decoding.html) | Best concise reference table for how `num_speculative_tokens` and draft-model resolution differ across methods |
| [PR #50424](https://github.com/vllm-project/vllm/pull/50424) | What 0.27.1 actually changed — worth the two-minute read since it names the exact classes (`Qwen3DSparkModel`, `DSparkMarkovHead`) |
| [Issue #41190](https://github.com/vllm-project/vllm/issues/41190) (TP=2 crash) and [#36872](https://github.com/vllm-project/vllm/issues/36872) (acceptance collapse) | Bookmark these specifically before you deploy `qwen3_next_mtp` — check comment threads for whether they're closed by the time you look |

Skip the individual HF community draft-model cards (Dogacel, satgeze, pablohassan, dbirks, etc.) unless you're actually shopping for a specific drafter — they're useful as existence proofs but they're unverified third-party repos, not something to cite as fact.

## What's genuinely missing

Two real gaps, not just hedging:

1. **Whether Xe2/Battlemage's compute-runtime driver actually reports `COOPERATIVE_KERNELS` support at all.** I searched compute-runtime's own docs and DeepWiki indexing and came up empty — Intel documents *implicit scaling* and USM defaults per-generation but I couldn't find a support matrix for the cooperative-kernel queue flag specifically. This is the one thing search won't resolve for you; it needs `zeDeviceGetCommandQueueGroupProperties` run against your actual device, or a grep through compute-runtime's `ProductHelper`/`GfxCoreHelper` source for where that flag gets set per-platform.

2. **I never re-checked whether issue #41190 (TP=2 crash) is fixed as of 0.27.1** — I found it, but didn't trace it forward through the PR history to confirm current status. Worth a `gh issue view` before you plan around it.

Smaller things I didn't chase but could if useful: EAGLE3 draft-model support specifically for Qwen3.5 (I only confirmed Qwen3-8B via Red Hat's P-EAGLE, not the 3.5 line), and I didn't pull vLLM 0.27.0's own release notes (only 0.27.1's diff) — there could be spec-decode-relevant changes in 0.27.0 that carried forward silently.


Here's the curated list, plus what's still open.

## Most worth bookmarking

**Topic 1 (Level Zero / SYCL forward progress)**

| Link | Why it's the one to keep |
|---|---|
| [sycl_ext_oneapi_root_group.asciidoc](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/proposed/sycl_ext_oneapi_root_group.asciidoc) | The actual answer to your question — states the L0/CUDA guarantee mismatch explicitly and gives the implementation-notes mapping to `zeKernelSuggestMaxCooperativeGroupCount` / `zeCommandListAppendLaunchCooperativeKernel` |
| [Level Zero Core Programming Guide](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/PROG.html) | Primary spec — "Cooperative Kernels" and "Events" sections are the relevant parts |
| [sycl_ext_oneapi_launch_queries.asciidoc](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/proposed/sycl_ext_oneapi_launch_queries.asciidoc) | The companion query API (`max_num_work_group_sync`) — this is what you'd call instead of hardcoding a probed number |
| [UR EXP-COOPERATIVE-KERNELS](https://oneapi-src.github.io/unified-runtime/core/EXP-COOPERATIVE-KERNELS.html) | Short — has the one-line universal floor ("any device can restrict to 1") |
| [Intel GPU Occupancy Guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/thread-mapping-and-gpu-occupancy.html) | Practical Xe-Core occupancy math, useful for reasoning about what a probed number *should* look like on your hardware |

**Topics 2/3 (vLLM spec-decode / Qwen3.5 MTP)**

| Link | Why it's the one to keep |
|---|---|
| [`vllm.config.speculative` source](https://docs.vllm.ai/en/v0.11.0/api/vllm/config/speculative.html) | `MTP_MODEL_TYPES` and `SpeculativeMethod` — ground truth for what methods exist, better than any blog post |
| [Qwen/Qwen3.5-35B-A3B model card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) | Official, matches your target size class — has the exact serve command |
| [vllm-project/speculators](https://github.com/vllm-project/speculators) | The training framework repo — README changelog is the best single place to track EAGLE/DFlash/DSpark status for Qwen-family targets going forward |
| [vLLM-Ascend spec-decode guide](https://docs.vllm.ai/projects/ascend/en/main/user_guide/feature_guide/speculative_decoding.html) | Best concise reference table for how `num_speculative_tokens` and draft-model resolution differ across methods |
| [PR #50424](https://github.com/vllm-project/vllm/pull/50424) | What 0.27.1 actually changed — worth the two-minute read since it names the exact classes (`Qwen3DSparkModel`, `DSparkMarkovHead`) |
| [Issue #41190](https://github.com/vllm-project/vllm/issues/41190) (TP=2 crash) and [#36872](https://github.com/vllm-project/vllm/issues/36872) (acceptance collapse) | Bookmark these specifically before you deploy `qwen3_next_mtp` — check comment threads for whether they're closed by the time you look |

Skip the individual HF community draft-model cards (Dogacel, satgeze, pablohassan, dbirks, etc.) unless you're actually shopping for a specific drafter — they're useful as existence proofs but they're unverified third-party repos, not something to cite as fact.

## What's genuinely missing

Two real gaps, not just hedging:

1. **Whether Xe2/Battlemage's compute-runtime driver actually reports `COOPERATIVE_KERNELS` support at all.** I searched compute-runtime's own docs and DeepWiki indexing and came up empty — Intel documents *implicit scaling* and USM defaults per-generation but I couldn't find a support matrix for the cooperative-kernel queue flag specifically. This is the one thing search won't resolve for you; it needs `zeDeviceGetCommandQueueGroupProperties` run against your actual device, or a grep through compute-runtime's `ProductHelper`/`GfxCoreHelper` source for where that flag gets set per-platform.

2. **I never re-checked whether issue #41190 (TP=2 crash) is fixed as of 0.27.1** — I found it, but didn't trace it forward through the PR history to confirm current status. Worth a `gh issue view` before you plan around it.

Smaller things I didn't chase but could if useful: EAGLE3 draft-model support specifically for Qwen3.5 (I only confirmed Qwen3-8B via Red Hat's P-EAGLE, not the 3.5 line), and I didn't pull vLLM 0.27.0's own release notes (only 0.27.1's diff) — there could be spec-decode-relevant changes in 0.27.0 that carried forward silently.-------------------------------------------- you can see the vendors/specualtors and vendors/spec-forge repio for these roeusrces and understan how it works?