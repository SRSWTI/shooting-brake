# Kill Bench: Level Up

Successor to `docs/kill-bench.md`, opened 2026-08-22. That file is the ledger of
what was killed and what was measured. This one is narrower and forward-facing:
**the remaining path to making Shooting Brake compute-bound.**

`kill-bench.md` stays authoritative for history. Nothing here repeats a result
that lives there; where a number comes from that file, it is cited by line.

---

## 0. The frame: we are chasing the compute floor

Every prefill optimisation until today made the *kernel* faster. That era is
over, and the reason is arithmetic:

| term | µs/token | share |
|---|---:|---:|
| PCIe transfer | 175 | 43% |
| B70 compute | ~165 | 41% |
| everything else (5090, scheduler, overhead) | ~64 | 16% |
| **total** | **404** | |

Transfer now **exceeds** compute. So the next win is not a faster kernel — it is
*moving fewer bytes*, and then *hiding what is left behind compute*. The target
state is:

```
transfer < compute, fully overlapped  =>  total = compute + fixed overhead
```

At that point prefill costs what the GEMMs cost and nothing more. That is the
"level up": Shooting Brake stops being a transport problem and becomes a compute
problem. Only then is a faster kernel worth writing again.

**Order matters and is not negotiable.** Prefill first, to the compute floor.
Decode after. The two share the doorbell but have completely disjoint
bottlenecks (§4), and mixing them is how measurements get attributed to the
wrong cause.

### Where we start from

```
1,705 us/token   per-route GEMV (2026-08-22 morning)
  526            grouped NVFP4 GEMM                        3.24x
  430            + fp16 result wire                        3.97x
  404            + prefetch_dist 6->1                      4.22x  (warm)
  420            same, first request after a fresh boot    4.06x
```

Live arm: `GROUPED=1 MAX_BATCH=2048 OUT_FP16=1 SB_MNBT=2048 SB_MNS=6
SYCL_UR_USE_LEVEL_ZERO_V2=0`, `max_model_len=163840` (KV concurrency 1.01x --
the ceiling). Quality verified indistinguishable from the pre-grouped baseline
(`kill-bench.md` Bench 26).

---

## 1. Hardware constraints — proven 2026-08-22, do not re-litigate

Every plan below is shaped by these. They were each established by source plus
measurement on this silicon, not inferred.

| constraint | evidence |
|---|---|
| **One compute engine (CCS).** Two of our kernels can NEVER overlap on one B70. | `compute-runtime/shared/source/xe2_hpg_core/bmg/os_agnostic_product_helper_bmg.inl:35` sets `NumberOfCCSEnabled = 1`; `zeDeviceGetCommandQueueGroupProperties` returns `group 0 COMPUTE numQueues=1`. Measured: a 37.2 ms floor under every combination of `USE_COMPUTE_ENGINE=-1`, `USE_IMMEDIATE_COMMANDLISTS=1`, two in-order queues, and one out-of-order queue with event chains. |
| **One copy engine (BCS).** H2D and D2H serialise with each other, but DO overlap compute. | `hw_cmds_base.h:36` `bcsEngineCount = 1u`; `hw_info_bmg.cpp:86` `ftrBcsInfo = 1` (bit 0 only); `group 1 COPY numQueues=1`. |
| **No directional copy split.** | `USE_COPY_ENGINE=lower:upper` is a round-robin range, not H2D:D2H. `BcsSplitSettings` ships `enabled=false` with empty masks (`product_helper.inl:575-578`); the split machinery (`bcs_split.cpp:80-142`) needs multiple real CSRs. |
| **NEO copy/blitter env keys are inert.** | `release_variables_base.inl:7-19` -- release builds read only that file, and it contains no copy, BCS, blitter, or split variable. Every such key tried was a no-op by construction. |
| **Device USM costs host RAM 1:1.** ~24.34 GiB/card, ~48.7 GiB for the pair on a 59.4 GiB box. | Measured by `experiments/b70_mem_topology_probe`, unchanged by every NEO debug key. Kills any host-resident prefill weight bank. |
| **Both B70s share one Gen4 x4 uplink.** 6.44 GB/s aggregate concurrent vs 6.47+3.23 solo. | `experiments/b70_mem_topology_probe`. This is why transfer is the ceiling. |
| **No B70<->B70 P2P, no cross-vendor DMA.** | Recon 2026-08-22; provider also creates one SYCL context per card, so peer sharing is structurally impossible in the current code regardless. |
| **`SYCL_UR_USE_LEVEL_ZERO_V2=0` is mandatory.** | V2 segfaults on a plain USM `memcpy` -- null fn ptr in `ur_command_list_manager::isGraphCaptureActive`, oneAPI 2026.1. See Bench 23. |
| **`malloc_host` is already optimal for the copy path.** | `prepare_for_device_copy` lowers to `urUSMImportExp`, which imports only when the type is `ZE_MEMORY_TYPE_UNKNOWN`; a `malloc_host` pointer reports `HOST`, so it is a no-op. NEO already takes the fast `findAllocationDataForRange` branch (`cmdlist_hw.inl:3316-3338`) rather than the raw-host fallback at `:3340-3354`. |

### One thing that turned out to be already done

The 5090's local-routed + shared-expert work **already overlaps** the B70 window.
`routed_experts.py:2388` branches on `M > _b70_max_batch`; with
`MAX_BATCH=2048` and `MNBT=2048` an M=2048 step never exceeds the cap, so it
takes the overlapping arm at `2508 issue-all -> 2510 CUDA local/shared -> 2514
take`. There is nothing missing at this seam for the 404 µs/token result.

This retro-explains part of a win I mis-attributed: raising `MAX_BATCH`
256 -> 2048 moved 732 -> 526 µs/token, and I credited all of it to tile
occupancy. It was **two** mechanisms -- better rows/expert *and* silently
switching onto the code path where the CUDA MoE work hides inside the B70
window. At the 8K profile, local/shared Marlin MoE was 279 ms against 151 ms for
GDN+attention (`prefill_profile/attribution_8k.json:14-22`), so that hidden
block is large.

Attention/GDN itself can never join the overlap: `qwen3_next.py:514-552` orders
attention -> post-attention norm -> MoE, and the B70 dispatch needs the
resulting hidden state and routing decision. That ceiling is real.

---

## 2. Prefill, ranked — the path to the compute floor

### 2.1 FP8 on the wire — the big lever

Halves **both** transfer legs: 12.6 -> 6.3 MiB per card per layer each way,
~25.1 MB per layer across both directions and both cards. Transfer **7.8 -> 3.9
ms**, which flips the binding term from transfer to compute for the first time.

**Key insight: no GEMM change is required.** The Xe2 mainloop is `A_DTYPE::BITS16`
only and explicitly dequantises quantised A *before* the GEMM
(`fused_moe_interface.py:407-417`). So we only need FP8 **on the wire**, with a
cheap elementwise widen on arrival -- exactly the mirror of the fp16 output
narrowing already shipped. The GEMM keeps consuming BITS16, untouched.

Both halves of the plumbing already exist:

| piece | path |
|---|---|
| 5090 quantise | `vllm._custom_ops.scaled_fp8_quant(x, scale=None, use_per_token_if_dynamic=True)` at `_custom_ops.py:1832-1898` -> `(E4M3FN [M,H], fp32 [M,1])`. Already used by `fused_moe/utils.py:138-162`. |
| convention to match, not invent | `scale = maxabs/448` (min-clamped), `q = clamp(x/scale)`, `x_hat = float(q)*scale` |
| B70 widen | `torch.ops._C_cache_ops.convert_fp8` -- `torch_bindings.cpp:298-301`, kernel `cache.cpp:1399-1442`, wrapper `:1484-1543` |
| XPU quantise (for D2H) | `torch_bindings.cpp:164-172`, one float scale/token at `fp8_quant.cpp:447-454` |
| fused variant, later (§2.3) | `sycl-tla/include/cutlass/fp8_to_fp16.h:42-74`; BMG template at `xe_flash_attn_prefill_mma.hpp:211-266`; working example `sycl-tla/examples/09_bmg_grouped_gemm_f8/` |

**Two landmines, both already identified:**

1. `convert_fp8` takes a single **host scalar** `double scale` -- it cannot
   consume `[M,1]` per-token scales. The fix is ~20 lines: the kernel already
   assigns one work-group per row (`block_idx = item.get_group(0)`), so
   `scale_` becomes `scales[block_idx]`.
2. Pass **`"fp8_e4m3"`, not `"auto"`**. The explicit branch (`:1526-1543`)
   bitcasts the byte; the `kAuto` branch performs a *numeric* uint8 cast. Same
   call, silently wrong answer -- the same class of bug as the layout character
   in Bench 25.

**Quality gate, and it is not optional.** This changes numerics. E4M3 carries 3
mantissa bits against fp16's 10 and bf16's 7.

- Gate the **full combined MoE output and logprobs**, never a remote partial in
  isolation. Two cards each round with their own per-token scale, then their
  partials are summed with the local and shared contributions; cancellation in
  that sum can magnify relative error in small final coordinates even when each
  partial looks clean on its own.
- Input is **not** the safer direction. Input error propagates through both
  GEMMs and the SwiGLU; output is rounded once. If one direction goes first on
  caution grounds, it is the output.
- Per-token E4M3 is the right first choice. Scale cost is ~8 KiB/card against
  6.3 MB saved -- negligible.
- Acceptance is `benchmarks/b70_ttft_smoke.py` plus the 120-prompt argmax sweep
  against the current arm, stratified warm/cold with repeats
  (`kill-bench.md` Bench 26 for the method, and why weaker instruments failed).

### 2.2 Intra-dispatch pipelining

Hides transfer under compute on the single BCS/CCS pair. Full design in
`kill-bench.md` Bench 29; **the terms there are stale** and should be re-derived
before building, because compute has since dropped to ~165 µs/token.

Ceiling is exactly `max(compute, transfer)` per dispatch and not one microsecond
better -- the two chunks' kernels cannot overlap (§1). So:

```
today, serial:              14.8 ms
after FP8, serial:          ~10.9 ms   (transfer 3.9 + compute 7.0)
after FP8 + pipelining:      ~7.0 ms   (transfer hides entirely under compute)
```

**FP8 and pipelining compose, and the order matters.** Pipelining alone, at
today's transfer of 7.8 ms against 7.0 ms compute, lands transfer-bound and
returns ~1.4-1.5x on the leg. After FP8 drops transfer to 3.9 ms it hides
*completely*, and the leg becomes pure compute. Doing pipelining first would
measure worse and teach the wrong lesson.

Safety notes carried forward: chunking **by token** means each token's `top_k`
routes stay inside one chunk, so chunks write disjoint output rows -- no atomics
race on `output` at all. The hazard is confined to the per-chunk scratch
(`hist`, `offs`, `cursor`, `rows`, `slot_row/exp/w`, `g_act`, `g_mid`,
`g_gated`, `g_outr`, `atom`). Use FreeToken's `[nchunks, ...]` **view** idiom
(`offload_cache.py:240`) -- one pool with a leading dimension -- rather than two
pointer sets, because a missed pointer is silent garbage.

Memory is roughly **flat**, not doubled: per-chunk buffers sized at
`M/nchunks` routes leave the total unchanged. Only the ~85-int per-expert arrays
duplicate. And the extra memory is B70-side, so KV and `max_model_len` are
untouched -- unlike every other memory lever tried.

Primary implementation is **one out-of-order queue with explicit event chains**,
not two in-order queues: they tie once compute dominates, and it is one state
machine instead of two. `nchunks` = 2 or 4; 4 hides more transfer but costs ~15%
kernel efficiency as rows/expert falls 120 -> 30. Genuinely ambiguous, so sweep.

### 2.3 FP8 stage B — fuse the widen into the mainloop

Removes stage A's separate pass and its full-size device scratch by expanding
FP8 A in registers. Surfaces identified: `gemm_xe2.hpp:55-57` (the `A_DTYPE`
enum), `:396-397` (A/B load), `:466-467` (A fragment -> 16-bit MMA fragment).
`ptr_A_scale` is already in the grouped-GEMM ABI but discarded at
`grouped_gemm_interface.cpp:35-42`.

Only worth doing after stage A proves the numerics survive. Stage A delivers the
entire byte saving with zero kernel risk; stage B only recovers its overhead.

### 2.4 Verify BCS direct submission is actually on

`hw_info_bmg.cpp:36-40` advertises direct submission for `ENGINE_BCS`;
`os_context_linux.cpp:105-110` requires vm-bind (or the OpenVINO light path) in
addition to product support. Small, but free to check, and it sits directly on
the leg that binds us.

---

## 3. Prefill dead ends — closed, with receipts

Do not spend time here again.

| idea | why it is dead |
|---|---|
| Two kernels overlapping on one B70 | 1 CCS (§1) |
| H2D and D2H on separate engines | 1 BCS, no linked group, split disabled (§1) |
| NEO copy/blitter env keys | inert in release builds (§1) |
| B70<->B70 P2P / dma-buf | nothing useful on this rig; one context per card anyway |
| Persistent host-pointer import / `prepare_for_device_copy` | no-op on `malloc_host` (§1) |
| Host-resident prefill weight bank (Marlin 44.4 / NVFP4 48.4 / W4A8 56.5 GiB) | device USM costs host RAM 1:1, leaving ~2.9 GiB free while serving |
| Per-card row compaction | at top_k=10 over an 85/85 split only **0.0739%** of rows omit either card -- ~9.3 KiB, erased by the row IDs needed to describe it |
| LRU / EPLB hotness placement for bytes | all 170 experts are already resident; hotness is cache machinery, not a byte saver |
| Asymmetric expert split by card speed | the 27.6% Gen3/Gen4 prefill gap was a synthetic-prompt artifact; on natural text routing is already balanced at 1.000x (`kill-bench.md` Bench 17-adjacent) |
| Bigger `MAX_BATCH` past 2048 | 256->512->1024->2048 gave 732->601->572->526; the tile is full and returns are gone |
| Deeper prefetch | monotonically worse: 24 -> 89.7, 12 -> 102.5, 6 -> 114.7, 1 -> 132.7 GB/s |
| The 128x256x16 tile | 1.41x faster standalone, **7% slower in situ** -- the probe ran fp16 output with a 16-bit store atom; production must write fp32 |

---

## 4. Decode — deferred until prefill hits the compute floor

Carried forward from `kill-bench.md`, not started. **Decode is not a B70
problem**, and that single fact should drive everything here.

The trace reconstructs ITL exactly (`kill-bench.md:1167`):

```
48 x 64 us B70 service  +  47 x 211 us 5090 inter-dispatch gap  =  11.4 ms ITL
```

| term | ms | share |
|---|---:|---:|
| B70 dispatch service | 3.07 | 21.6% |
| **5090-side inter-dispatch gap** | **9.92** | **69.7%** |

**An infinitely fast B70 buys 21.6%**, landing at 11.16 ms (`:1177`). So every
B70-side idea -- grouped kernels, faster GEMMs, more local experts -- is capped
at a fifth of decode before it starts. Two independent measurements already
confirm this: grouped buys nothing at M=1 (each expert is read once regardless),
and decode ITL **saturates at L~40** -- 21 more local experts moved nothing
(`:1649`, Bench 20).

The gap is the target. Reference point: the 88B ran **one** lane at a 142 µs gap;
r15 runs **two** at 211 µs (`:1180`), so the second lane costs ~69 µs.

### Ranked, highest expected value first

1. **Immediate command lists on the doorbell submission path.** The Level Zero
   docs are explicit that these are a *latency* feature -- "dedicated to very
   low-latency submission usage models" -- not a throughput one. Tested on
   prefill 2026-08-22 and correctly did nothing (37.27 ms, unchanged). The
   209-212 µs gap **is** a submission-latency problem, which is exactly what
   they are for. Right tool, wrong place, and never tried in the right place.
   `zeCommandListCreateImmediate` with the compute queue group ordinal.

2. **n-gram / suffix speculative decoding.** vLLM 0.27.1 ships `ngram`,
   `ngram_gpu`, and `suffix` -- prompt-lookup methods needing **no draft
   weights at all**, which matters because r15 ships **zero** `mtp`/`eagle`/
   draft tensors. Estimated 1.3-2x on decode from the M-scaling trace (M=1 at
   67.6 µs, M=2 at 105.3, M=7 at 285.2 -- sublinear, so verifying 4 speculated
   tokens costs ~1.5-2x rather than 4x). Composes with prefix caching rather
   than competing: caching exploits repetition in the *input*, speculation in
   the *output*. Costs a flag to test.

3. **`dflash` speculative decoding.** Registered in vLLM 0.27.1 and
   `laguna_dflash.py` ships; the official Poolside recipe uses a separate
   drafter (`poolside/Laguna-S-2.1-DFlash`, `num_speculative_tokens: 15`). The
   drafter is **not downloaded**, and it was trained against `Laguna-S-2.1`,
   not our 15%-REAP-pruned derivative -- so acceptance rate is the open
   question and it needs ~2.5-3 GiB of the remaining VRAM headroom. Potentially
   ~3x, but gated on a download and an acceptance test.

4. **Overlapping the doorbell call with 5090 compute during decode.** Bench 17
   measured that the dispatch is **not** currently hidden inside the gap work;
   if it could be, ITL drops 11.0 -> ~7.9 ms (~22%). Needs profiler correlation
   first to confirm the gap is genuinely available rather than already
   dependent -- it is a sum, not a max, in the current trace.

5. **Speculative B70 dispatch ahead of the true router** (`kill-bench.md:247`).
   Issue the doorbell on a predicted top-k before the router finishes, sliding
   the dispatch earlier. Explicit kill condition already written down:
   **recall < 60%** on our model and it stays parked.

### Decode dead ends

| idea | why |
|---|---|
| FP8 KV cache | measured: costs **10.4% ITL** to buy 2.4% prefill. Wrong trade. |
| More local experts | ITL saturates at L~40 (Bench 20) |
| Grouped kernel at M=1 | each expert is read exactly once already; grouping has nothing to remove |
| Batching for decode throughput | measured worse than single-sequence decode |
| Per-layer route-aware placement | +7.8% on natural text, but the provider ABI requires one resident expert set shared across all layers (`resident_set_shared_across_layers`), so it is not expressible without a bank/ABI change |

---

## 5. Measurement discipline — earned the hard way on 2026-08-22

Eight wrong numbers in one day, every one caught by a control rather than by
intuition. These are the rules that caught them.

1. **Never trust a single sample.** The prefetch win read +5.6% from one pair of
   runs and **+3.3-4.1%** once repeated and stratified. The two arms had
   different noise characteristics (1.0% vs 5.4% spread) which no single pair
   could reveal.
2. **Stratify warm and cold.** The first smoke after a boot is systematically
   slower (420 vs 404 µs/token). This produced a phantom "+12% decode
   regression" that vanished at three samples (12.46 -> 11.40 ms).
3. **Prefix-disjoint prompts, or you are measuring the cache.** GuideLLM reuses
   one synthetic prompt per cell, so `ctx_1024/C=1` read TTFT
   **min 63 / med 89 / max 126 ms** on a prompt whose cold cost is ~430 ms --
   entirely cache hits. Fixed by per-cell seeds (`matrix_runner.py` `_cell_seed`,
   crc32 not `hash()` so a `--skip-existing` resume reproduces prompts).
4. **`--skip-existing` is config-blind.** One `--output-root` per arm, diffed
   afterwards, never merged. Reusing a root after a config change silently
   compares two builds as one surface, which reads exactly like a partial
   regression.
5. **The probe must match production.** The 128x256x16 tile won standalone by
   1.41x and lost 7% in situ because the probe used fp16 output with a 16-bit
   store atom while production must write fp32.
6. **This rig has never been reproducible.** The *pre-existing* GEMV path flips
   argmax on **9% of prompts run twice**, and its 3-pass top-1 logprob spread
   (0.339 nats) is **wider** than the grouped path's (0.277). No prefill change
   here can be validated by exact output comparison; the only valid instrument
   is many independent single-forward-pass prompts with the reference compared
   against **itself** as the ceiling.
7. **`loginctl enable-linger` before any long run.** Both server deaths on
   2026-08-21 were `logind` reaping the session -- graceful SIGTERM, no CUDA
   error, no OOM. `setsid nohup` does not survive logout.
8. **Drain, don't kill.** `pkill` on a benchmark client with a 127K prefill
   in flight took EngineCore down with it (`EngineDeadError`).
9. **Long matrices need tiering.** Device USM's 1:1 host shadow leaves ~5 GiB,
   which an hour of GuideLLM consumes. `benchmarks/matrix_tiered.sh` restarts
   the server per context tier; the reclaim is exact (0.5G -> 55.5G).
10. **State the mechanism, then test it -- and expect to be wrong.** Every
    mechanism story I told on 2026-08-22 was wrong; every measurement was right.
    Deeper prefetch "should" have helped and was monotonically worse.

---

## 6. Instruments

| tool | what it answers |
|---|---|
| `benchmarks/b70_ttft_smoke.py` | cold+warm TTFT 1K-32K, prefix-disjoint, exact token counts, **refuses to print speed for an incorrect server** |
| `benchmarks/b70_matrix_probe.py` | context x concurrency surface with per-card trace attribution on one clock |
| `benchmarks/matrix_runner.py` | GuideLLM serving matrix (per-cell seeds since `f150f17d`) |
| `benchmarks/matrix_tiered.sh` | drives the above one context tier per server lifetime |
| `benchmarks/matrix_progress.sh` | one progress line: cells, health, host mem/swap |
| `src/phase7/xe2_probe/xe2_grouped_probe_nvfp4` | GEMM-only policy/prefetch sweeps, no server |
| `src/phase7/xe2_probe/xe2_nvfp4_verify` | NVFP4 correctness gate (E4M3 decode, nibble order, layout) -- 20 s, **must stay green** |
| `experiments/b70_dual_queue_probe` | whether two queues overlap on one B70, and by how much |
| `experiments/b70_mem_topology_probe` | per-card H2D/D2H, concurrent aggregate, host-RAM cost of device USM |

---

## 7. Next action

**FP8 wire, stage A.** In order:

1. ~20-line SYCL per-token widen, adapted from `cache.cpp:1399-1442`.
2. H2D FP8: `scaled_fp8_quant` on the 5090, carry `(E4M3 bytes, fp32 [M,1])`.
3. D2H FP8: mirror with the XPU quant op.
4. Gate: combined MoE logprobs + 120-prompt argmax sweep vs the current arm,
   stratified warm/cold with repeats.
5. Measure: expect transfer 7.8 -> 3.9 ms and the binding term to flip to
   compute.

No speedup gets quoted until step 4 passes.

---

## 8. Reference baseline: the same model on 1x RTX PRO 6000 (2026-08-23)

`srswti/axe-superveloce-jota-118b-r15-nvfp4`, vLLM 0.27.1, TP=1, one RTX PRO
6000 Blackwell (SM120, 96 GiB), `--no-enable-prefix-caching`, guidellm, 512
output tokens. 134 strategies, **zero errors**. Data and recipes in
`benchmarks/results/rtx_pro_6000_r15_slo/`, derived table in
`DERIVED_SUMMARY.json`.

This is the honest comparator: same weights, same server, caching off on both
sides, so it prices the disaggregation exactly.

| ctx | ref TTFT | ref us/tok | SB us/tok | ratio | ref ITL | SB ITL | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1K | 0.068 s | 63.8 | 489 | 7.7x | 7.02 ms | 11.40 | 1.62x |
| 8K | 0.415 s | 50.5 | 406 | 8.0x | 7.25 ms | 11.40 | 1.57x |
| 16K | 0.947 s | 57.6 | 397 | 6.9x | 7.52 ms | 11.40 | 1.52x |
| 32K | 2.336 s | 71.2 | 421 | 5.9x | 8.03 ms | 11.40 | 1.42x |
| 64K | 6.256 s | 95.4 | 446 | 4.7x | 9.05 ms | 11.40 | 1.26x |
| 96K | 11.911 s | 121.1 | -- | -- | 10.11 ms | 11.40 | 1.13x |
| 127K | 18.929 s | 145.5 | 494 | 3.4x | 11.10 ms | 11.40 | **1.03x** |

SB prefill from the cold512 matrix (per-cell seeds, `f150f17d`); SB ITL from
`b70_itl_probe` C=1 with flatness established 1K-30K by `b70_matrix_probe`.
Beyond 30K SB ITL flatness is `[INFERENCE]` from the mechanism, not measured.

### Two structural findings, and they both favour long context

**1. The prefill gap halves as context grows -- 8.0x at 8K to 3.4x at 127K.**
The reference is attention-bound and superlinear (50.5 -> 145.5 us/token, 2.9x
over the range). Shooting Brake is transport-bound and nearly flat (406 -> 494,
1.2x) because our cost is bytes over a Gen4 x4 uplink, and bytes per token do
not grow with context. Their curve is even U-shaped, with a minimum at 4K
(45.6 us/token) before attention takes over.

**2. Decode reaches parity at 127K: 11.10 vs 11.40 ms, 1.03x.** Their ITL grows
with KV depth (7.02 -> 11.10 ms). Ours does not move at all, because decode is
69.7% the fixed 211 us/layer 5090-side inter-dispatch gap (§4) and only 21.6%
B70 service -- neither term knows the context length. A weakness at 1K becomes a
wash at 127K purely from the shape of the bottleneck.

Concurrency, for completeness: the reference peaks at **C=4** and degrades past
it (1K: 304 tok/s at C=4, 254 at C=5), and at 96K/127K only C=1-2 admit at all.
Peak aggregate output: 304 / 269 / 231 / 170 / 116 / 61 / 40 / 37 tok/s across
1K..127K.

### What this does to the plan

It sharpens the target rather than changing it. Projected FP8 + pipelining puts
the B70 leg near pure compute at ~255-286 us/token end-to-end (§2.2), against
this reference:

| ctx | ref us/tok | SB today | SB projected | projected ratio |
|---|---:|---:|---:|---:|
| 8K | 50.5 | 406 | ~260 | ~5.1x |
| 32K | 71.2 | 421 | ~270 | ~3.8x |
| 127K | 145.5 | 494 | ~310 | **~2.1x** |

So the level-up lands Shooting Brake within roughly **2x of a single RTX PRO
6000 at 127K context, on a consumer 5090 plus two Arc Pro B70s** -- against a
96 GiB datacenter part that holds the entire model in one address space. That is
the number worth chasing, and long context is where this architecture is
structurally strongest. Decode is already at parity there.

---

## 9. FP8 wire: screened, and it is DOMINATED (2026-08-23)

`benchmarks/fp8_wire_numerics.py`, real NVFP4 weights from layer 3, activations
drawn at the amax the checkpoint itself calibrated (`input_global_scale=2064`
-> amax 1.302). Dequant verified before trusting any number: weight std
**0.03024**, plausible for a trained MLP; the script aborts if it is not.

| wire | rel L2 | vs the bf16 leg already in production |
|---|---:|---:|
| bf16 activations (today's H2D, unquestioned) | 2.30e-3 | 1x |
| fp16 partial (today's D2H, gated Bench 26) | 2.07e-4 | 0.09x |
| FP8 partial, per-token | 2.65e-2 | **11.5x** |
| FP8 partial, per-block 16 | 2.23e-2 | 9.7x |
| FP8 both legs | 4.51e-2 | 19.6x |

### Three things this settles

**1. Block granularity does not rescue it.** Per-token 2.65e-2 -> block-16
2.23e-2 is a **16% improvement for 16x the scale bytes** (block 16 over hidden
3072 = 768 B of scales on a 3072 B payload, +25%). I predicted finer blocks
would fix this by analogy with `w4a8_nvfp4`'s near-lossless per-32-block
activation quant. Wrong, and for an identifiable reason: block scaling fixes
*dynamic range*, and this error is *mantissa* -- E4M3 has 3 bits, ~6% per
element, and no amount of rescaling adds bits. Ninth wrong mechanism story,
same pattern.

**2. The independent-per-card-scale worry was unfounded.** Shared scale and
independent scales are **identical** at 2.45e-2, and cancellation amplification
is only **1.22x**. I flagged this as the trap that would make me ship a bug by
gating a partial in isolation. It is not a trap. Risk retired by measurement.

**3. Input is the worse direction, as predicted.** f8in 3.66e-2 with 30% max
relative error, against f8out 2.65e-2 and 5.9%. Input propagates through both
GEMMs and the SwiGLU; output is rounded once.

### Why FP8 is dominated -- the arithmetic that matters

Pipelining's ceiling is `max(transfer, compute)` per dispatch (one CCS, one BCS,
§1). Transfer and compute are now **balanced**: 175 vs ~165 us/token. So:

| config | serial | pipelined `max()+other` | vs today |
|---|---:|---:|---:|
| today | 175 + 165 + 64 = 404 | -- | 1.00x |
| + pipelining only | -- | 175 + 64 = **239** | **1.69x** |
| + FP8 out + pipelining | -- | 165 + 64 = 229 | 1.76x |
| + FP8 both + pipelining | -- | 165 + 64 = 229 | 1.76x |

**FP8 buys 4% on top of pipelining, for 11x the numerical error of the bf16 leg
already in production.** And FP8-both is worth exactly the same as FP8-out,
because both land under the compute floor -- so the riskier input leg buys
literally nothing.

The ordering assumption I wrote in §2.2 was correct *when I wrote it* (transfer
262 vs compute 187) and is now stale: fp16 + prefetch cut transfer to 175 and
compute to 165, and once the two terms are within 6% of each other, `max()`
already captures almost the entire win. **FP8 was only ever valuable as a
prerequisite. Pipelining removed the need for it.**

### Verdict

**Do not build the FP8 wire.** Not rejected on quality -- it may well pass a
120-prompt sweep -- rejected because it is worth 4% and pipelining is worth 69%.
Revisit only if pipelining lands and compute becomes the wall by a wide margin,
at which point cutting bytes below compute is again worth something.

Screen retained as `benchmarks/fp8_wire_numerics.py`; data in
`benchmarks/results/b70_gemv_audit/fp8_wire_numerics.json`.
