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

---

## 10. Pipelining Phase 1: per-chunk plumbing, gated bit-identical (2026-08-23)

`SHOOTING_BRAKE_B70_PIPELINE=<nchunks>` (default 1). Route-scaled scratch pooled
with a leading `[nchunks]`; chunk loop still sequential on the one in-order
queue. Gate: `benchmarks/pipeline_identity_gate.py` -- 15 cells (3 layers x
M 1/7/33/256/2048), old .so twice for the envelope, new .so at flag unset/1/4.

- **M=2048 (production tile): bitwise identical** across builds, flags, runs --
  including nchunks=4 sequential.
- M=256: stable NaN masks identical old-vs-new byte-for-byte; run-to-run
  low-bit atomic drift (~1e-16 rel) pre-exists in the baseline.
- nchunks=4 at M=256 drifts <= 1e-9 abs: per-token contribution ORDER changes
  with chunk composition -- same non-associativity class the baseline already
  exhibits run-to-run. Rows stay disjoint; the sums have the same terms.

**Open finding (pre-existing, NOT this change):** raw `issue`/`take` at M<=33
returns unstable garbage (NaNs, 1e38s, varying run-to-run) in the COMMITTED
build -- A1-vs-A2 shows it with zero new code involved. Grouped cells are clean;
split/fused small-M cells are not. Production is insulated: decode rides the
poller ring, prefill chunks are >=256. Repro: the gate harness against
/tmp/sb_old.so. Investigate before any new consumer uses raw issue/take at
small M.

## 11. Pipelining Phases 2-3: correct under concurrency, NOT yet faster (2026-08-24)

Phase 2 (committed `67b5fdcc`): dedicated in-order copy queue, per-chunk RAW
barriers, one WAR barrier against the previous dispatch, per-chunk D2H behind
chunk markers. Gate at real concurrency: **M=2048 bitwise identical** at
nchunks 2 and 4; M=256 <= 4.4e-16 with equal NaN masks -- the baseline's own
atomic class. A missed hazard would read 1e38, not 1e-16.

Phase 3 standalone (1 card, M=2048, warm p50, issue/take):

| nchunks | p50 ms | vs serial |
|---|---:|---:|
| 1 | 18.27 | 1.00x |
| 2 | 19.42 | **0.94x -- SLOWER** |
| 4 | 23.29 | 0.78x -- much slower |

The prediction (~1.36x) did NOT survive contact. Two suspects before any
conclusion, both testable:
1. **Pageable-host contamination.** The harness feeds plain numpy memory; the
   dual-queue probe that measured 1.26-1.5x overlap used pinned malloc_host,
   and production passes registered ranges. Pageable H2D stages through the
   runtime and can serialize -- the standalone number may be measuring the
   staging path, not the design.
2. **nchunks=4 crosses the tile policy.** 241 -> 60 rows/expert lands in the
   small-tile regime; that cost is real and bounds nchunks at 2 for this M.

Status: flag default-off, nothing shipped regressed, correctness fully gated.
NEXT: re-measure with registered/pinned host buffers, then Phase 4 in-situ
paired A/B (1K-127K, 3 repeats) where the plugin's real buffers apply --
the standalone harness is disqualified as a speed instrument until suspect 1
is resolved. Tenth mechanism lesson: the probe's conditions are part of the
claim.

## 12. Pipelining lands: +5-6% at 16K-127K, mechanism confirmed (2026-08-24)

Registration re-measure (`sb_b70_register_host`, commit `318d146c`): the
Phase-3 regression was entirely the pageable-host staging path. Registered,
standalone nchunks=2 is 17.97 vs 18.18 ms serial.

Phase 4 paired A/B (3 repeats/arm, cold, prefix-disjoint, same session,
MML=163840, quality gate green on all runs):

| ctx | baseline | PIPELINE=2 | speedup | ranges overlap? |
|---|---:|---:|---:|---|
| 1K | 0.439 s | 0.445 s | 0.99x | yes (wash) |
| 8K | 3.281 s | 3.374 s | 0.97x | yes (wash) |
| 16K | 6.513 s | 6.122 s | **1.064x** | no |
| 32K | 13.449 s | 12.696 s | **1.059x** | no |
| 64K | 28.689 s | 27.045 s | **1.061x** | no |
| 96K | 45.755 s | 43.652 s | **1.048x** | no |
| 127K | 62.432 s | 59.449 s | **1.050x** | no |

Aggregate 460 -> 438 us/token (1.047x). 32K production band: 410 -> 387
us/token, **4.40x** vs the 1,705 baseline. 127K: 59.45 s vs the RTX PRO 6000's
18.93 s -> **3.14x** (was 3.30x).

- **Trace cross-check passed:** per-dispatch service at M=2048 fell
  13.04 -> 12.16 ms (d0) and 15.16 -> 14.06 ms (d1) with bytes/dispatch
  constant at 25.33 MB. Service down, bytes flat = the overlap is physical.
  Gen3 hides more absolute time, as its slower transfers predict.
- **1K/8K wash is the tile-policy crossover:** single-dispatch prompts chunk
  to 512-1024 rows -> 60 rows/expert -> small-tile regime eats the overlap.
  A rows-per-expert floor on the pipelined arm would recover 8K; parked.
- **The 1.29x prediction was wrong; the probe's heavy-kernel row (1.10x) was
  right.** The max(transfer, compute) model ignored that the CCS also runs
  the narrow passes and barriers, and that only the copy engine overlaps.
  Eleventh mechanism lesson, same shape as the first ten.
- **Phase 0 answered en passant:** shared-uplink duty during a C=1 ladder is
  ~36% average (2.3 of 6.44 GB/s). Not saturated; FP8 stays retired.
- Decode untouched by construction (M=1 takes the split path, pipelined=false).

Phase 5 running detached: `bench-matrix/jota_r15_pipeline2` via matrix_tiered
(PIPELINE=2 passed through tier reboots, one root per arm; baseline surface is
the banked `jota_r15_c6` grid). Monitor: `benchmarks/matrix_progress.sh`.

## 13. The result wire stays fp16 -- 8-bit and 4-bit both closed (2026-08-24)

Prototyped an 8-bit result wire ([M*H u8][M f32 per-token scales], both D2H
paths, flag-gated). Measured SLOWER than fp16 at M=2048 (Gen4 17.94 -> 20.31,
Gen3 20.56 -> 22.24 ms): the quantize pass costs more than the 1.9 ms of D2H
it saves unless written as a proper parallel reduction, and the projected
ceiling even then is ~1.10x -- against a wire error already screened at 11.5x
the shipped leg. Reverted and removed per the bad-results rule (prototype
preserved in history at `ef94db4a` for the record).

An NVFP4-format wire is closed by the same screen without building it: e2m1
carries 1 mantissa bit against e4m3's 3, and 4-bit activations measured 0.82
cosine (terminal) in the W4A4 qualification. The checkpoint's NVFP4 is a
WEIGHT format; the wire carries activations, a different sensitivity class.

The wire hierarchy, final: fp32 -> fp16 was 10x BETTER than the bf16 leg
(shipped, Bench 26); fp16 -> 8-bit is 11.5x worse for ~5%; 4-bit is terminal.
fp16 is the knee. Transfer work below fp16 belongs to topology (dedicated
PCIe lanes on the product), not precision.

---

## 14. Decode campaign: prepped, queued behind the Phase-5 matrix (2026-08-24)

Target: beat the RTX PRO 6000's decode curve (7.02 ms @1K -> 11.10 @127K; ours
11.40 flat) at most or all contexts, then hold it across concurrency. Every
microsecond off our flat line moves the crossover left: 10.1 ms beats them at
96K+, 9.0 at 64K+, 8.0 at 32K+.

### Recon results (SglangDecodeScout, VllmSpecWiring -- full cites in agents)

- **Spec decode is plumbing-compatible end to end.** The doorbell path has no
  M==1 assumption: staging slices [:M] up to max_batch=128, the poller forwards
  M unchanged, the provider accepts 0<M<=max_batch. Verification rows are
  num_seqs*(1+k): ngram k=4 at MNS=6 -> 30 rows; dflash k=15 -> 96. Both fit.
  GDN and linear-attention backends explicitly size for k. The poolside_v1
  parser is post-generation only. CUDA-graph keys become token-row sizes --
  padding may replay 2/4/8/16-row graphs; our M-scaling is sublinear (measured
  M=2 105 us, M=7 285 us), so verification is cheap.
- **Copy-ready configs** (serve script now takes SB_SPEC):
  - ngram: `{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":2,"prompt_lookup_max":5}`
    (V1 runner only -- V2 rejects ngram; surfaces at boot if it bites)
  - dflash: `{"method":"dflash","model":"poolside/Laguna-S-2.1-DFlash","num_speculative_tokens":15}`
    (installed vLLM ships DFlashLagunaForCausalLM; drafter downloading;
    trained against UNPRUNED Laguna-S-2.1 -- acceptance on r15 is the unknown)
  - suffix: DEAD here -- validator requires arctic-inference, not installed.
- **SGLang recon:** strongest transferable idea is full-forward decode graphs
  (their default); their overlap scheduler is a core-engine-loop change, not
  portable to a plugin; their only no-weight proposer is the same ngram; a
  Laguna-eligible fused sigmoid router exists (minor, µs-scale).

### Session plan (server frees when the matrix banks its last cell)

1. `benchmarks/decode_gap_probe.py` -- decompose the 211 us gap by layer index
   using the architecture itself as the instrument: linear vs full attention
   layers alternate through identical orchestration, so the linear-layer floor
   IS the orchestration cost and the full-linear delta IS attention compute.
2. ngram arm: `SB_SPEC='{"method":"ngram",...}'`, ITL probe + acceptance rate
   + quality gate. Prediction: effective decode beats the PRO 6000 curve at
   every context if acceptance >= ~50% on code/prose.
3. dflash arm: same gate, k=15. Prediction withheld -- acceptance on the
   REAP-pruned target is unmeasurable from the armchair.
4. Concurrency surface: guidellm concurrent profiles vs the PRO 6000 bundle's
   own concurrent results (benchmarks/results/rtx_pro_6000_r15_slo) at matched
   contexts and rates.

Dead, replicated in recon: suffix (missing dep), MTP (no tensors), more local
experts, grouped-at-M=1, FP8 KV, overlap scheduler (core-loop surgery).

## 15. Decode session 1: gap decomposed, spec decode triaged (2026-08-24)

**Gap probe (the decisive number):** decode gap is 146-150 us/layer and
UNIFORM across layer types (full_attention 145.6/150.7 vs sliding 146.4/149.0
us). Attention compute contributes nothing measurable at M=1: the gap is 100%
orchestration (piecewise graph replay + doorbell handshake + glue), ~6.9 ms of
the 12.1 ms ITL. B70 service sum: 3.2-3.3 ms.

**Spec decode ladder (512 out, prefix-disjoint, TPOT; ptok = measured):**

| arm | ~1.4K | ~13K | ~24K | ~37-60K | ~104-107K | verdict |
|---|---:|---:|---:|---:|---:|---|
| baseline PIPELINE=2 | 12.3 | 12.7 | ~12.7 | 13.2-14.2 | ~14 | flat, uniform |
| ngram k=4 (CPU) | **8.59** | 17.2 | **10.1** | 18.5 | 24.8 | wins short, CPU lookup scales with prompt, loses long |
| ngram_gpu k=4 | 21.0 | 14.2 | 11.6 | 20.3-26.5 | 30.6 | worse everywhere -- dead |
| dflash k=15 | 43.8 | 47.2 | 49.7 | 62.0 | 75.5 | **0/7,665 accepted** -- drafter trained on unpruned Laguna-S-2.1, dead on REAP-pruned r15 without a finetuned drafter |

ngram greedy text diverged from the no-spec reference on one prompt -- within
this rig's measured 9% run-to-run argmax flip rate, but any adoption requires
the 120-prompt sweep (Bench 26 instrument). dflash also cost 4.7 GiB of KV
(drafter on the 5090): MML capped at ~62K at util 0.85.

**The lever that survives triage: FULL-graph decode capture** (SGLang's
default: one host submission per token vs our 47 piecewise replays). The gap
probe says ~6.9 ms is attackable; success lands flat ITL at ~5-7 ms --
below the RTX PRO 6000 at EVERY context (their 7.02 @1K -> 11.10 @127K).
ngram k=4 could stack on top for short-context agentic loops, gated on the
sweep. Next session: FULL capture experiment.

Matrix state: paused at 16/24 cells; resume with the same MATRIX_ROOT command.

## 16. FULL-graph decode: tried, did not deliver on vLLM 0.27.1 (2026-08-24)

`--compilation-config {"cudagraph_mode":"FULL_AND_PIECEWISE"}` boots clean
(the plugin's force-PIECEWISE patch requires VLLM_USE_BREAKABLE_CUDAGRAPH=1,
which the r15 script unsets). Measured: ITL 11.89 ms vs 12.12 baseline
(inside boot noise), gap 143 vs 146-150 us -- no material change. The 146 us
orchestration floor is NOT per-segment replay overhead vLLM's FULL mode can
remove on this stack/version; either the hybrid attention backends downgraded
the mode or the gap lives in the scheduler/doorbell glue itself.

Decode standings after session 1: baseline 12.1 ms flat; ngram k=4 8.6 ms at
short ctx (workload flag, quality sweep owed); everything else triaged dead.
Remaining decode levers, in order: (a) decompose the 143 us further with a
torch-profiler capture correlated to the doorbell trace -- scheduler glue vs
replay vs handshake; (b) the 61 us handshake (immediate command lists /
submission path); (c) ngram behind a workload flag after the 120-prompt sweep.

## 17. Decode verdict + production defaults locked (2026-08-24)

**Remaining decode systems levers: parked, with the arithmetic on record.**
Handshake work (47 x 61 us, maybe halvable -> ~1.4 ms) plus norm/router kernel
fusion (~1-1.9 ms) land at best ~9.5 ms ITL after weeks of gated C++/CUDA --
moving the PRO 6000 crossover from 127K to ~64K and never touching their
short-context numbers. 83 vs 100 tok/s is invisible to a single user. Not the
efficient path; revisit only with nothing better to do.

**The decode roadmap is a training artifact:** finetune Laguna-S-2.1-DFlash on
r15's own generations. Serving integration is proven end-to-end (k=15 boots,
verifies, M<=128 doorbell fits); acceptance is the only broken piece (0/7,665
because the drafter never saw the pruned target). At normal DFlash acceptance
that is 4-6 ms effective decode -- past the PRO 6000 at EVERY context.

**ngram k=4 is a WORKLOAD FLAG, never a default.** It WAS tested at long
context: 8.6 ms TPOT at ~1.4K ctx but 17-25 ms at 13K-107K -- the CPU lookup
scales with prompt length and inverts our flat-line advantage exactly where we
are strongest. Flag: SB_SPEC on the serve script. Any adoption for short-ctx
agent loops still owes the 120-prompt argmax sweep.

**Production defaults now equal the verified best arm** (serve script + tier
driver): GROUPED=1, OUT_FP16=1, MAX_BATCH=2048, PIPELINE=2, MNBT=2048, MNS=6.
Everything else in the tree from the decode session is instruments and docs;
the two negative-result prototypes (FP8 wire, FULL-graph) are reverted /
config-only. Session-end state: what ships IS what was measured best.

## 18. Doorbell 2.0 silicon probe: GREEN, 7.5x faster than the host poller (2026-08-24)

`experiments/b70_cs_doorbell_probe.cpp` -- pre-recorded L0 command list:
MI_SEMAPHORE_WAIT on a shared host word, then PIPE_CONTROL host-scope write.
No host thread in the loop, either direction. Round trip (2000 iters):

| ring source | Gen4 p50 | Gen3 p50 | p99 |
|---|---:|---:|---:|
| host store | 4.71 us | 5.61 us | 6.5-7.0 |
| **5090 DMA write (cudaMemset on pinned page)** | **8.12 us** | **9.15 us** | ~85 |

vs the 61 us host-poller handshake: **7.5x**. Projected decode: 47 x ~52 us
saved = ~2.4 ms/token -> ITL 12.1 -> ~9.7 ms, plus two pinned poller cores
freed.

Three driver facts the probe established (each would have cost a debug week):
1. `appendWaitOnMemory` REJECTS external-sysmem-imported pages on this driver
   (every action/scope combo -> INVALID_ARGUMENT) but accepts plain L0 host
   USM and -- decisively -- **CUDA-pinned pages directly** (cudaHostAlloc,
   end-to-end verified with bounded-spin trap detection; 0/2100 timeouts).
2. cudaHostRegister of L0 host-USM pages fails (file-backed VMA) -- the
   composition must be CUDA-allocates, CS-waits.
3. The accepted wait desc is act=EQUAL, waitScope=HOST(4); write scope HOST.

Integration path (next): per-layer pre-recorded WAIT(seq) -> kernels ->
WRITE(done) chains in the provider, DeepEP parity protocol (two buffers,
monotonic counters, release-publish). Kernel handles via SYCL native interop
or the raw-L0 side path. Gate: identity harness, then ITL probe, then smoke.

## 19. Doorbell 2.0 probe #2: the ride-along chain works, 7.9-8.7 us (2026-08-24)

`experiments/b70_sycl_zex_interop_probe.cpp` validates the integration design
that needs NO kernel-handle extraction: zex WAIT/WRITE appended directly on the
provider-shaped in-order SYCL queue (which IS an L0 immediate command list on
this stack), bracketing ordinary SYCL kernel + memcpy submissions.

  ring -> [WAIT -> kernel -> D2H -> WRITE] -> host sees completion
  Gen4: p50 7.86 us (p99 9.99)   Gen3: p50 8.67 us (p99 11.09)   vs 61 us poller

One production-grade hazard caught by the probe, not by users: the D2H memcpy
rides the BCS while zex commands order only within the CCS immediate list --
the completion fired BEFORE the payload landed. Fix: a marker kernel after the
copy joins SYCL's cross-engine event chain on the CCS; WRITE goes after it.
Ordering + payload visibility verified CORRECT on both cards after the fix.

Integration blueprint (next, behind SHOOTING_BRAKE_B70_CS_DOORBELL=1):
per step, layer 0 rides the host poller as today (M becomes known); the poller
then pre-enqueues layers 1..46 as [WAIT(signal_L==seq) -> H2D -> kernels ->
D2H -> marker -> WRITE(completion_L)] on the SYCL queue -- host leaves the
critical path for 46 of 47 layers, no parked-engine hazard across steps
(the WAIT parks only after the previous layer drains; steps are atomic).
Measured pieces: transport 8-9 us (probe 1), full chain 7.9-8.7 us (probe 2).
The ~9.7 ms ITL row stays a PREDICTION until the paired A/B runs.

## 20. Doorbell 2.0 integration: built, BLOCKED on UR-adapter internals (2026-08-24)

`issue_cs_chain` (provider) + CS poller mode (capi) behind
SHOOTING_BRAKE_B70_CS_DOORBELL=1, default OFF. Three integration variants,
three boot-time wedges, ~10 min per observation:

1. Cached immediate-list handle from get_native(queue): wedged at graph-capture
   warmup, pollers parked in urEventWait -- the V2 UR adapter ROTATES lists;
   appends landed on a retired list that never executes.
2. Fetch-fresh handles per append: same wedge -- V2 also splits SYCL work
   across internal compute/copy lists, so "the queue's list" carries no
   ordering contract for foreign appends at all.
3. Provider-OWNED raw in-order immediate list (doorbell + copies + completion)
   stitched to SYCL kernels via native events (make_event in, get_native(tail)
   out): wedged EARLIER, during initial profiling, frames entirely inside
   adapter internals.

**Lore correction, load-bearing:** wedged stacks show
libur_adapter_level_zero_v2.so live in serving despite
SYCL_UR_USE_LEVEL_ZERO_V2=0 -- that knob is INERT on oneAPI 2026.1. Every
number this campaign ever shipped ran on the V2 adapter; Bench 23's "V1
required" note is stale.

**Still proven:** the transport (18/19): 8-9 us CS doorbell round trip vs the
61 us handshake, CUDA-pinned pages accepted, both cards, full chain shape.
The blocker is exclusively raw-L0-append x live-V2-adapter interaction.

**Production safety verified:** flag-off boot with the new .so is green; first
warm ITL 11.65 ms = exact baseline band (later probes 13.8-16.5 = documented
n=2 rig noise; morning baseline spanned 11.7-14.9 on identical config).

**Unblock plan, in order:**
1. UR_ADAPTERS_FORCE_LOAD=libur_adapter_level_zero.so.0 (true V1) for the CS
   arm -- one boot; the design was built for V1's one-stable-list semantics,
   and V1 has never actually run on this rig.
2. Standalone repro harness: load the provider .so, poll_register/poll_start
   with fake CUDA-side signal writes, drive M=1 steps through issue_cs_chain
   -- seconds per iteration, gdb-visible, no engine on top.
3. If V2 contract gap confirmed: CS arm ships on V1, A/B'd against V2-classic
   honestly, or the chain moves to a dedicated CCS-ordinal queue pending
   NEO guidance.

Projection IF unblocked (prediction, not measurement): ITL 12.1 -> 9.7-10.6 ms
(47 x ~52 us handshake removal, range covers chain kernel-start cost), decode
crossover vs the RTX PRO 6000 moves 127K -> ~96K, and two pinned poller cores
free. Fusion (+~1 ms) and the finetuned drafter (4-6 ms effective, wins at
every context) stack on the same chain.

**Unblock path 1 result (2026-08-24): V1-forced boot wedges identically**
(UR_ADAPTERS_FORCE_LOAD=libur_adapter_level_zero.so.0, CS arm on). Same hang
at the first M<=32 activity around graph-memory profiling. Verdict: the
blocker is the chain design x live-engine interaction, NOT V2-vs-V1 list
management -- which also retires the "ship on V1" escape hatch. The
standalone poller harness (path 2) is now the only next vehicle, and the
right one: the wedge point VARIES across variants (before profiling, at
capture-memory), which smells like a race my probes' single-threaded shape
never exposes -- the harness can run the poller thread + fake CUDA writer
concurrently under gdb in seconds per iteration.

## 21. Doorbell 2.0: wedge FOUND AND FIXED via harness; design C too slow (2026-08-24)

`experiments/b70_cs_harness.py` reproduced the four-boot wedge in 20 seconds:
a RACE -- after enqueueing a step's chains the poller kept scanning, and the
classic sweep STOLE-AND-CLEARED a later layer's signal before its hardware
WAIT fired, deadlocking the in-order queue (step 1, layer 14, poller parked in
take()'s wait_and_throw). Fix: cs_step_fence() -- the poller parks on a SYCL
tail barrier until the step drains (it has nothing else to do mid-step).
50 steps clean after the fix. Every vLLM wedge signature is explained by this
race (timing-dependent wedge points included).

**But measured speed says design C loses:** 381 us/layer round trip vs the
classic poller's 118 us. The provider-owned raw list pays FOUR cross-runtime
event seams per layer (raw->ev_in->SYCL barrier; SYCL tail->get_native->
WaitOnEvents->raw), each tens of us on this stack. The 8 us transport is
intact; the stitching eats it.

**Also decisive:** the harness process runs the V1 adapter
(libur_adapter_level_zero.so.0 in the stacks) -- SYCL_UR_USE_LEVEL_ZERO_V2=0
WORKS in a plain process; something in the vLLM environment overrides it to
V2. Design A (ride-along zex on the queue's own immediate list, 7.9 us in
probe #2) is stable under V1's one-list-per-queue semantics.

**Next iteration, exactly:** (1) find what forces V2 under vLLM and pin V1
for serving, (2) re-run design A (ride-along) in the harness under V1 --
prediction: ~120 us/layer classic -> ~75-85 us/layer (handshake replaced,
seams zero because everything rides ONE list), (3) then the vLLM ITL A/B.
Flag remains default-off; classic path untouched and verified.

## 22. Doorbell 2.0 CLOSED: dead by measurement, three designs deep (2026-08-24)

Mode 2 (`issue_cs_ride`, design A ride-along: brackets on the SYCL queue's own
immediate list, zero event seams) built, gated, and measured against mode 1 and
classic in the harness. Correctness clean everywhere. Speed ladder (us/layer,
M=1, 47 layers, burst-verified writer-independent):
  classic 120 | design C 383-418 | design A 436-454.
Bisect attribution (SHOOTING_BRAKE_B70_RIDE_BISECT): satisfied WAIT ~80 us,
each HOST-scope WRITE ~95 us, markers ~15 us each, wait/write SCOPE irrelevant,
copy-engine irrelevant, burst (pre-armed signals) irrelevant. Fully stripped
chain still 252 vs classic 120.

**The mechanism lesson (12th of the campaign): the probe's 8 us bracket was a
pre-recorded 2-command list replayed on an idle engine. Live zex appends
interleaved with real kernels pay ~10x -- per-append submission granularity
against an executing immediate list. Probe conditions are part of the claim.**

Minimum bracket pair ~175 us > the 61 us host handshake it would replace.
Per-layer live-append doorbells are DEAD on this silicon. Only surviving shape:
fully pre-recorded per-layer command lists (DeepEP-style), blocked on kernel
handle extraction from quixicore's SYCL RTC path -- a separate project.
Flag stays default-off; classic poller keeps the crown. Decode stands at
12.1 ms; the next lever in the gate order is w2 fusion.

## 23. Kernel-handle extraction SOLVED: baked chains are buildable (2026-08-24)

`experiments/b70_cs_harness.py`'s verdict said only pre-recorded lists survive.
`experiments/b70_baked_chain_probe.cpp` proves they are buildable, all three
blockers in one run on the B70:
1. **Handles**: named SYCL FUNCTOR kernels -> get_kernel_id -> kernel_bundle
   -> get_native yields real ze_kernel_handle_t. No runtime internals.
2. **Arg ABI**: functor fields ARE the kernel args, in declaration order --
   zeKernelGetProperties reports numKernelArgs == field count (6 and 7, zero
   hidden args), and the raw zeKernelSetArgumentValue launch is **BIT-EXACT**
   against the SYCL launch of the same functor (identity gate, 300 replays).
3. **Speed**: baked WAIT -> gate_up -> w2_epilogue -> D2H -> WRITE replays at
   186 us vs 178 us for live SYCL submit+wait of the same kernels -- bracket
   machinery in a BAKED list costs ~20 us over pure device work, vs 61 us
   host handshake, vs 175 us live-append brackets (kill-bench 22).

The probe's w2 kernel is ALSO the fusion prototype: rowmajor register
accumulation across routes, no atomics, no zero-fill memset, direct fp16
write -- two levers converged into one kernel.

**Projection (labeled)**: real quixicore kernels are ~68 us device work at
M=1, so a baked per-layer chain lands ~88-95 us vs the classic poller's 120
-> ITL 12.1 -> ~10.6-11.0 ms, plus one ExecuteCommandLists per STEP replacing
47 handshakes. Remaining engineering, next session: (a) hoist quixicore's
kernel-name classes out of the anonymous namespace (get_kernel_id needs
visibility) or export a handle-getter from that TU; (b) provider bakes 47
per-layer lists at bank load (M=1 first, per-M variants later); (c) identity
gate + ITL A/B through the harness before any serving flag.

## 24. Baked chains (mode 3): CORRECT end to end; 138 vs classic 119 us/layer (2026-08-24)

Full pipeline landed behind SHOOTING_BRAKE_B70_CS_DOORBELL=3:
- quixicore exports named FUNCTOR kernels (Nvfp4BakedGateUp, Nvfp4BakedW2Out16
  <I,R>) + nvfp4_moe_baked_handles() -- same-TU handle extraction, field-order
  arg ABI (u32 scalars separately probe-proven: b70_u32_abi_probe.cpp).
- provider baked_record_layer(): ONE closed regular command list per layer --
  WAIT(sig==1) -> clear -> barrier -> H2D x3 -> gate_up -> fused w2 epilogue
  (rowmajor, register accumulation, no atomics, no zero-fill, direct fp16)
  -> D2H -> barrier -> WRITE(comp=1). baked_execute_step(): one
  ExecuteCommandLists for all 47 + synchronize (fence semantics).
- **Parity PASS: worst max-rel 9.4e-4 across 47 layers** vs the classic split
  path, real bank, real pinned rings, 30-step replay.

The bisection that got there (kill-bench discipline, seven discriminators):
kernels bit-exact (b70_baked_kernel_ab.cpp stages 1-2), pinned-ring copies
clean (stage 3), u32 ABI clean, ring import irrelevant, zero-hidden/zero-
weights/neg-ids all prove inputs land... and the actual bug was THE GATE:
the harness's torch RNG was unseeded, so cross-process parity compared
different INPUTS, not different modes. Instrument rule, now enforced in the
harness: parity gates must be deterministic before they may accuse anything.

**Speed: 138.1 us/layer (R=2, device-scope clears for layers 1+) vs classic
119.5.** Correct but not yet a win; NOT flag-ready. The ~19 us decomposes to
(a) ~12-command list structural overhead (~20 us measured in the baked probe)
and (b) the token-major epilogue's SLM activation refills (pairs x row_tiles
fills; 7.7 MB/layer at R=2 vs quixicore's 1 MB at R=16-atomic). Next levers,
profiling-led: per-command timestamps on the baked list, epilogue refill
restructure (global-activation kernel fusion into gate_up, or pair-major
tiles with a token-major reduction), fold the ids+weights H2D into one copy,
and re-testing R in {2,4} after the refill fix. Target stays ~90 us -> ITL
12.1 -> ~10.6-11.0 ms; the vLLM SLO A/B runs the day the harness beats 119.

## 25. Baked chains: harness floor found, vLLM capture wedge found, PARKED (2026-08-24)

In-list timestamp profiling (SHOOTING_BRAKE_B70_BAKED_PROFILE=1) settled the
"138 vs 119" question: H2D 5.3 + gate_up 63.5 + w2 56.9 + D2H 2.0 = 127.7 us
-- the kernels ARE the time, and they are bandwidth physics: 10 routes x
5.3 MB = 53 MB/layer at this card's GB/s = ~115 us. The harness classic
baseline (119.5) is the SAME physics + a ~5 us spin: the harness's tight
C-loop writer has NO 61-us-class host handshake to remove, so the doorbell's
prize is invisible there BY CONSTRUCTION. Both arms sit on the bandwidth
floor; baked pays ~18 us of brackets. Instrument lesson #13: know what your
baseline is made of before you race it.

Live vLLM A/B (the only terrain that can show the win):
- Arm A (classic): ITL 12.49 ms, TTFT 0.080 s -- healthy, matches the 12.1-
  12.5 band.
- Arm B (SHOOTING_BRAKE_B70_CS_DOORBELL=3): WEDGED at CUDA graph capture 3/4,
  8+ minutes. Root cause class: baked_execute_step is all-47-or-block --
  vLLM's warmup/capture dummy passes (partial sweeps; recorded-not-executed
  signal writes) violate that contract and park the poller in
  zeCommandQueueSynchronize. The classic sweep tolerates partial steps by
  construction. The harness cannot reproduce this (its writer always
  completes steps).

**PARKED per the agreed stop-loss.** State: parity-correct (9.4e-4), fully
instrumented, capture-unsafe documented, flag default-off, classic path
untouched and re-verified live (arm A). To revive: (1) arm mode 3 only after
capture completes (first-real-step latch), (2) bounded execute with partial-
step recovery, (3) re-run this exact A/B. Expected prize remains the host-hop
share of the 211 us inter-dispatch gap; the drafter finetune (4-6 ms
effective) outranks it and is the next lever.

## 26. Production-KV bridge: banked grid stays quotable (2026-08-24)

The 16-cell pipeline2 grid was banked at KV=8.29 GiB; production now runs
10.69 GiB (+29%). Re-ran ctx=32768 (all three profiles) and 65536/synchronous
under the production config into `bench-matrix/jota_r15_prod_kv` as a bridge
control. 10 comparable cells, zero errors both sides:
- **C=1..4: -1.1% to +1.7% ITL, throughput flat** -> the banked grid remains
  quotable for the single-user / small-team range.
- **C=5,6 at 32K: ITL +33.7% / +59.6%, aggregate tok/s -7.9% / +0.5%.** Not
  noise -- MECHANISM: at 8.29 GiB the scheduler QUEUED those requests; at
  10.69 GiB they fit KV simultaneously and truly batch. Queueing became
  batching. Per-request ITL rises, aggregate throughput does not, because the
  box is already saturated at 32K x C>=5. More KV buys capacity and
  prefix-cache retention, NOT throughput, at contexts this long.
- 65536/synchronous +8.3% ITL on a single stream (one data point, above the
  ~4% boot band; unexplained, flagged not explained).
Remaining 98K/127K concurrency cells cancelled deliberately: at 127K the KV
pool admits 1.61 concurrent, so those cells measure queueing, and the capacity
figure (211,083 tokens, 1.61x) already states that.
