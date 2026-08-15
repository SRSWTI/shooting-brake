# 88B on 5090 + Arc Pro B70 — next steps

State as of the 128K matrix run (`benchmarks/results/run2_88b_128k/`).

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