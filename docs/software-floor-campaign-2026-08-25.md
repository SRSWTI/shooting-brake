# Software-Floor Campaign — measured aim points (2026-08-25)

Goal set by the operator: make the software hit the physics floor of the
current hardware so that a PCIe slot upgrade (Gen5 x16 per Arc card) is the
only remaining lever. This doc records the two probe results that aim the
campaign (A1, B1), the recon-established facts, and the ranked plan.
Everything here is measured on this rig today unless marked projection.

## A1 — asymmetric expert placement (95/75): SHIPPED-READY, +2%

`fractional:2:<F>:95,75` placement grammar added (largest-remainder
apportionment, positional over B70 device indices; device 0 = Gen4 card).
Even-split output verified byte-identical to the production default.

3-pass cold-TTFT ladder vs the banked 3-pass native control:

| ctx | native p50 | 95/75 p50 | delta |
|---|---:|---:|---:|
| 1K | 0.480 s | 0.479 s | -0.2% |
| 8K | 3.365 s | 3.339 s | -0.8% |
| 16K | 6.832 s | 6.760 s | -1.1% |
| 32K | 14.080 s | 13.774 s | -2.2% |
| 64K | 30.084 s | 29.289 s | -2.6% |
| 96K | 47.818 s | 46.708 s | -2.3% |
| 127K | 65.286 s | 63.687 s | -2.5% |

Aggregate 481.0 -> 471.2 us/token (**-2.0%**). Decode unchanged (12.11
ms/tok spot check). Below the 5-8% hope because rebalance only shifts
D2H results + compute; H2D activations are duplicated to both cards, so
the shared-uplink term is untouched. Verdict: free, monotone with
context, zero quality risk (same kernels, ownership only) — recommend
default after one identity-gate pass. Artifact:
`benchmarks/results/backend_bakeoff/a1_9575_ttft_ladder.json`.

## B1 — decode decomposition: the "7.9 ms of waiting" was a myth

torch.profiler over 25 live decode steps (in-process trigger; chrome
trace at `/tmp/sb_torch_prof/decode_trace.json.gz`, 11.79 ms/step wall,
matches the banked 12.1):

| component | ms/step | share | detail |
|---|---:|---:|---|
| GPU busy | 8.07 | 68.4% | real compute, not orchestration |
| .. gemm/moe (local+shared experts, router) | 4.89 | | **584 GEMM launches/step** (~6/layer at M=1) |
| .. elementwise glue | 2.27 | | **1,710 launches/step**, ~1.3 us each; `cvt_fp16_to_fp4` 288x/step |
| .. attention (flash splitkv) | ~0.45 | | healthy |
| .. memcpy (pinned H2D/D2H) | 0.63 | | staging |
| GPU idle | 3.72 | 31.6% | Arc waits + launch gaps + sampler/tail |

Also established: decode seams are graph-replayed with **no per-layer
Python** (the seam tracer only caught the eager M=29 prefill step —
`/tmp/sb_seam_trace.npz`; per-layer Arc-wait there: 0.90 ms mean at
M=29). The prior campaign narrative attributed ~7.9 ms to "the 5090
waiting and orchestrating"; the correct split is ~8.1 ms genuine 5090
compute + ~3.7 ms idle.

**Consequence — decode lever ranking changes:**
1. **5090 local-MoE fusion** (new #1 kernel lever): 4.89 ms of M=1 GEMMs
   in 584 launches + 2.27 ms of glue incl. per-layer activation fp4
   requant. FreeToken (vendored) shows inline-dequant GEMV beats W4A16
   tensor cores at M=1 small-intermediate; vLLM-Moet has SM120 kernels.
   Credible aim: 7.2 -> ~3 ms/step. Requires owning the vLLM in-tree
   W4A4 local-expert leg (operator authorized upstream surgery).
2. **Drafter** (unchanged, biggest end-to-end): 12.1 -> 4-6 ms effective.
   `vendor/speculators` is the direct vLLM-V1 path; vendored vLLM already
   has Laguna hidden-state hooks + `DFlashLagunaForCausalLM`. Training
   plan: `docs/drafter-finetune-plan.md`.
3. **Idle attack** (doorbell 4th design + overlap): bounded by 3.72 ms
   idle, of which Arc service ~2.5 ms partially overlaps local compute
   already. Doorbell design 4 (sliding window + sequence-tagged signals +
   replay-only arming; NEO immediate command lists for submission cost)
   is real but now demonstrably third in line.

## Recon verdicts that gate the plan (13 agents, this session)

- CUDA-accepts-Intel-memory: **dead at driver source level** (FOLL_LONGTERM
  pin fails on xe BOs; dma-buf import blocked by libcuda policy). The
  shipped CUDA-owns-pages inversion is the only road.
- Intel host shadow (47.4 GiB DRAM): **not fixable in userspace** — xe
  KMD/TTM territory; `NEO_LOCAL_MEMORY_ALLOCATION_MODE` is Windows-only,
  BMG `isLocalOnlyAllowed=false`. Park.
- Abortable Arc wait: needs a NEO extension patch (conditional BB-start
  machinery is latent); immediate command lists exploitable now.
- `PolicyBigM` aliases `PolicySmallM` (m32_k16) in
  `src/phase7/xe2_nvfp4/grouped_moe.cpp` — unused m64/m128 policies exist.
  Possible free prefill GEMM gain at 120-row fills; bench first.
- Grouped prefill pipeline carries ~580 MiB/layer avoidable intermediate
  traffic (gather 120, SwiGLU round trip 100, g_outr + atomic scatter
  ~360) ~= 0.96 ms/layer fusion prize.
- Chunk pipeline serializes compute on one in-order queue; copy engine
  overlaps but compute chunks never do.
- KV/prefix persistence vehicles: Tutti (GPU-driven NVMe KV store, vLLM
  connector architecture in-tree, CUDA-only = fine for 5090 KV) ahead of
  TensorCast (needs source build for torch 2.13 + low-DRAM config; no
  shipped vLLM KV connector). The 139x reuse lever.
- B70 cookbook: 8K->16K scheduler chunk measured +17.6% prefill / +12%
  decode elsewhere (HOL/VRAM risk) — cheap config arm to trial.

## Ranked plan (prize x cost, all aims measured)

| # | lever | phase | aim | cost |
|---|---|---|---:|---|
| 1 | 95/75 placement default | prefill | -2.0% TTFT (measured) | config, done |
| 2 | Shared-stage fusion + PolicyBigM fix | prefill | ~0.9-1.4 ms/layer of 12.16 | our kernels, days |
| 3 | 5090 local-MoE fusion (launch collapse + M=1 GEMV) | decode | 8.07 -> ~4-5 ms busy | vLLM surgery, days-week |
| 4 | Drafter training (speculators pipeline) | decode | 12.1 -> 4-6 ms effective | training job, start now |
| 5 | Scheduler chunk 16K trial | prefill | up to +17% (cookbook, unverified here) | config, hours |
| 6 | Transfer-window staggering | prefill | part of 34% contention | provider, days |
| 7 | Doorbell design 4 + immediate lists | decode idle | <=2 of 3.72 ms idle | harness first, NEO patch later |
| 8 | KV persistence tier (Tutti connector) | TTFT for returning sessions | 139x on reuse (measured mechanism) | integration, week+ |

Prefill floor with 1+2+6: bracket -> wire time; then Gen5 x16 is the only
remaining prefill lever. Decode floor with 3+4(+7): ~4-6 ms effective —
decode has no hardware lever, software takes it all the way.

Gate discipline unchanged: every lever ships through microbench ->
numerics -> flag -> identity harness -> SLO ladder -> scorecard.

## Execution results (same day, evening session)

Levers 1-4 of the ranked plan were executed. Measured outcomes:

| lever | microbench | in-situ | verdict |
|---|---|---|---|
| reduce scatter (`SB_GROUPED_SCATTER=reduce`) | -4.7%/layer at M=2048; **bit-deterministic across runs** (kills the atomic-scatter NaN-churn class) | TTFT -0.3..-0.6% (wire-bound hides it) | gates PASSED (vs-base 7e-7 reorder; identity envelope cleaner than base). Flag ready; default flip = Ship item |
| BigM d32 (`SB_GROUPED_BIGM=d32`) | -11.5%/layer at M=2048, **bit-exact** vs shipped | same as above | gates PASSED. Combined with reduce: 8.12 -> 6.79 ms/layer (-16.4%) |
| fused shared expert (`SHOOTING_BRAKE_FUSED_SHARED=1`) | 47 GEMM + 47 quant launches removed/token, ARMED on all 47 layers, correct | ITL wash: warm ~11.7 both arms (absorbed by Arc-wait overlap) | NEUTRAL - parked behind flag. Step-2 (shared-as-49th-expert) killed on the same absorption arithmetic |
| drafter stage 1 | on-policy corpus generation running (3000 prompts, `experiments/drafter_datagen.py`, resumable/pausable) | - | stage-2 decision open: rent 96GB (recommended by plan) vs on-rig capture hook |

The wire-bound lesson, third confirmation: Arc-side compute wins do not
move TTFT today; they lower the compute floor so the bracket collapses to
pure wire - which is the campaign's end state (then Gen5 x16 pays fully).

## Incidental findings (same session)

- **Truncated-thinking responses are EMPTY**: vLLM 0.27.1 + poolside_v1
  reasoning parser returns empty content AND reasoning_content when
  generation hits max_tokens inside an unclosed think block. Model healthy
  (raw completions + finish=stop are correct). Pre-existing upstream
  behavior, a real API landmine for short-budget requests. Ticket-worthy.
- **VRAM headroom regression class**: 2 concurrent chunk prefills OOM the
  5090 through `scaled_fp4_experts_quant` (300 MiB transient) when boot
  headroom is ~390 MiB (desktop squatters: gnome-remote-desktop 504 MiB +
  an unkillable CrashReporter 60 MiB). Mitigation in effect: 512 MiB KV
  trim (`SB_KV_BYTES=10936647680`, -9.9K KV tokens) -- needs operator
  ratification or squatter eviction for the banked default to be safe.
- Plugin `init_logger` lines are invisible in serve logs (non-vllm logger
  namespace); use print() for boot-time evidence.
- Bench scripts are not HF-offline; the A1 ladder pulled a NEW model
  snapshot (`eae47502`) into the HF cache mid-session. Serving stayed
  pinned to `357b5c1f`, but pin bench tokenizers offline to keep it so.
- **Shared-stage fusion, remaining two-thirds PARKED with arithmetic**: the
  reduce scatter (shipped above) was the valuable third. SwiGLU-as-epilogue
  needs a dual-accumulator gate/up mainloop redesign (halves live in
  different N-tiles); gather-into-GEMM breaks XE_LOAD_2D block copies.
  Combined remaining prize ~0.4-0.5 ms/layer of traffic -- invisible while
  wire-bound, and the layer already gave up 16.4% this session. Revival
  condition: wire stops binding (Gen5 x16) AND the bracket becomes
  compute-gated at these stages.

## Drafter pipeline status (end of session)

On-rig hidden-state capture is BUILT AND PROVEN end-to-end
(`src/phase4/src/shooting_brake_vllm/hidden_capture.py` +
`experiments/drafter_capture.py`): 2/2 smoke records captured on real
silicon, exact Poolside DFlash 6-slice contract (target layers
1/10/19/29/38/47, verified against the published Laguna-S-2.1-DFlash
config), [T, 6, 3072] bf16 + loss mask per corpus id. The rent-a-GPU
option is now optional, not required. Hard-won capture facts:

- vLLM 0.27.1 non-streaming chat DISCARDS reasoning_content entirely;
  the corpus generator uses RAW /v1/completions (thinking markup verbatim)
  or the drafter would never learn to draft thinking tokens.
- Prefix-cache hits are fatal to capture (cached tokens produce no hidden
  states; observed rows=17 of 49): capture requests carry a unique
  `cache_salt`.
- fp16 storage saturates on Laguna's residual stream -> bf16.
- BOS/attention-sink rows are non-finite at the deepest boundary (model
  reality, not pipeline): zeroed with per-record accounting.
- Aux-hidden arming interacts with the torch.compile cache: capture boots
  need `VLLM_DISABLE_COMPILE_CACHE=1` or a stale artifact returns bare
  tensors and crashes warmup.

Remaining, wall-clock-gated: 3,000-record corpus regenerating overnight
(raw schema, `/tmp/sb_datagen_night.log`), then the full capture run
(capture boot + `drafter_capture.py`, ~hours at prefill speed), then
SpecForge/speculators training on the 5090, then the acceptance gate per
`docs/drafter-finetune-plan.md`.

## Overnight run + host-OOM incident (late session)

22:22: the KERNEL OOM-killer took EngineCore (`global_oom`, host RAM, not
GPU): the 47.4 GiB driver shadow leaves ~12 GiB for everything, and
server + datagen x2 + an agent runtime tipped it (8 GiB swap was already
7.2 used). Collateral: datagen, the overnight orchestrator, and the
TrainPrep agent died with it. Recovery + hardening:
- datagen runs at concurrency 1 overnight (operator-requested lighter
  load; ~56 tok/s, ~3k records by morning).
- `experiments/drafter_overnight.sh` now gates on CORPUS RECORD COUNT
  (>=2900), not on the datagen process existing -- the process-existence
  gate fired early after the OOM and raced a production boot.
- Host-RAM discipline recorded: at most one agent runtime while serving.

Training/acceptance scaffolding audited and finished
(`experiments/drafter_train/`, TrainPrep2): warm-start-mandatory SpecForge
offline DFlash pilot, streaming .pt adapter, poolside 69-key <-> training
81-key QKV mapping validated both directions, distribution-based quality
gate (text equality is unusable on this box), 10/10 CPU contract tests.
One-command: `train.sh` (one 5090, serving stopped), then `acceptance.sh`
(boots with SB_SPEC, gates >=60% acceptance and <=6 ms/tok effective).

## Root cause of the night's instability: /tmp is a 30 GB tmpfs

**`/tmp` is tmpfs on this box -- every byte written there is HOST RAM**, the
one resource already scarce (47.4 GiB driver shadow on 59.4 GiB). Bulk
scratch written to /tmp during smoke testing reached 24 GB and caused, in
order: the 22:22 kernel OOM-kill of EngineCore, a second engine death, and
a SIGKILLed checkpoint save mid-write. Clearing it returned **23 GB of RAM**
(available 31 -> 54 GB). Standing rules now enforced in
`experiments/drafter_overnight.sh`:
- `require_lean_tmpfs()` refuses to start a GPU stage while /tmp holds
  >5 GB.
- All bulk paths are disk-backed: captures `~/sb_hidden_capture`, training
  `experiments/drafter_train/work`, state `~/sb_pipeline_state`.
- Long-lived processes run as systemd USER UNITS (`sb-serve`, `sb-datagen`,
  `sb-pipeline`); linger is on, so a dead terminal session can no longer
  take the stack down (it did, at 23:13).

## Drafter training: proven on GPU, with measured numbers

Seven smoke iterations against the real 5090 turned six latent failures
into fixes. Each was a genuine defect that would have wasted an unattended
night:

| # | failure | fix |
|---|---|---|
| 1 | exclusivity check demanded ZERO compute apps; GNOME residents (~616 MiB) always hold contexts | refuse only on vLLM processes or any >1 GiB consumer (`train.sh`) |
| 2 | warm-start + resume passed together (SpecForge forbids); `=null` overrides are string-typed and never unset | `pilot.yaml` keeps both null; each branch sets exactly one |
| 3 | strategy whitelist rejected `LagunaDFlashDraftModel` | extended the dflash provider's `compatible_architectures`, mirroring dspark's existing multi-arch pattern (vendor edit) |
| 4 | offline `snapshot_download` under a redirected cache_dir left the bare repo id -> "No checkpoint found" | pinned the local `357b5c1f` snapshot path (the one we serve) |
| 5 | capture `input_ids` are int32; DFlash index-put needs int64 | `.long()` at the vendored offline reader (vendor edit) |
| 6 | Adam OOM inside `_multi_tensor_adam` (30.25 GiB used) | `optimizer_cpu_offload: true` -- masters+moments (~13.5 GB) to host RAM, free because serving is stopped during training; plus `expandable_segments` for the measured 2.93 GiB fragmentation |

**Measured on this rig (two independent runs, 40 epochs x 2 records = 20
optimizer steps):**

| metric | value |
|---|---|
| optimizer steps/hour | **737 - 739** |
| optimizer step time | 4.88 s |
| data wait per step | 0.030 s (0.6% -- loader is not the bottleneck) |
| samples/second | 0.82 |
| learning signal | acc 0.44 -> 0.785, loss 1.72 -> 0.44 over 20 steps |
| checkpoint on disk | **15 GB** (bf16 model + fp32 masters + Adam moments) |
| export | 2.23 GB, **69 native tensors**, `DFlashLagunaForCausalLM`, `model_type: laguna`, exact `dflash_config` -- matches the poolside reference contract |

**Projection [INFERENCE, from the measured step rate]:** the disk-bounded
pilot (45 GB budget ~= 590 records) is 590 x 6 epochs / 4 accumulation =
~885 optimizer steps = **~1.2 h of training**, replacing the plan's
unmeasured 6-24 h band. Step time scales with sequence length; the bench
records averaged 1,373 tokens against a corpus average near 1,500-2,500,
so treat 1.2-2.5 h as the honest range.

Disk budget corrected from that measurement: capture 68 -> **45 GB** and
`max_checkpoints` 3 -> **1**, because 3 x 15 GB + 68 GB far exceeds the
82 GB free on the shared volume.

**Vendor edits made (both minimal, commented, upstream-shaped):**
`vendor/SpecForge/specforge/algorithms/dflash/providers.py` (architecture
compatibility set) and
`vendor/SpecForge/specforge/algorithms/common/hidden_states_data.py`
(int64 token ids).

## Overnight pipeline: design corrections found by running it

`experiments/drafter_overnight.sh` reached its final shape only after the
box rejected three earlier versions. Each correction is a general lesson:

1. **Gate on the artifact, not the producer.** v1 waited on
   `pgrep drafter_datagen`; when the generator died, the wait fell through
   instantly and raced a production boot. v2 gates on corpus record count.
2. **Size the wait to the real constraint.** v2 waited for 2,900 records
   while the disk budget admits ~590 captures -- it would have spent the
   entire night generating data this pilot cannot consume. v3 waits for
   900 (capture budget + selector margin) and lets generation continue
   behind it for a future, larger cycle.
3. **Make teardown idempotent.** The EXIT trap unconditionally booted a
   production server; restarting the unit therefore stopped a healthy
   server and raced a second boot onto port 8017. The trap now no-ops when
   `/health` already answers.

Resilience posture for unattended running: `sb-serve`, `sb-datagen`, and
`sb-pipeline` are systemd **user units** (linger on). A closed terminal or
dead session can no longer take the stack down -- the exact failure at
23:13 that cost the corpus two hours.

Measured ETA at hand-off: 4.5 records/min -> 900 records in ~2.0 h, then
capture (~590 records at prefill speed), ~1.2-2.5 h training, then the
acceptance gate. Stage markers land in `~/sb_pipeline_state/`, so a
morning rerun resumes rather than repeats.

## Drafter serving path: PROVEN, with a hard VRAM ceiling found

The riskiest untested interface was export -> vLLM: if the trained
checkpoint would not load, the whole pilot would be wasted at the final
step. Validated on real silicon (1-optimizer-step probe drafter, disk
backed, cleaned up after):

- **The drafter loads and speculates.** `DFlashLagunaForCausalLM` +
  `method=dflash` + `num_speculative_tokens=15` accepted by vLLM 0.27.1
  from our own `export_laguna.py` output. The 69-key native layout is
  loadable, not just contract-correct.
- **The measurement path the gate depends on works**: `/metrics` exposed
  `spec_decode_num_draft_tokens_total` +4,485 over 299 drafts, so
  acceptance is computable per request.
- **Acceptance 0.0% for the probe, which is the CORRECT result**: one
  optimizer step on two records is effectively the stock warm start, and
  the stock drafter's documented acceptance on r15 is 0 of 7,665. The
  instrument agrees with the known baseline.
- **The cost curve, measured**: 15 drafted tokens at 0% acceptance gives
  **43.96 ms/token** against ~11.7 baseline -- speculation with a bad
  drafter is **3.75x slower**. This is exactly why the gate demands >=60%
  acceptance before shipping, and it is now a measured number rather than
  an argument.

**Hard ceiling found (would have failed the 5 AM gate):** the drafter is a
third resident on the 5090. vLLM's own figure: a 131,072-token request
needs **9.64 GiB** of KV; the baseline runs at 10.19 GiB, leaving 0.55 GiB
of slack against a drafter needing ~1.96 GiB of weights plus graphs. Two
boots proved it: at the baseline pool the engine OOMs during init (469 MiB
free), and at KV 7.45 GiB vLLM refuses 131072 outright (estimated max
99,248). The candidate therefore boots at **max_model_len 98304, KV
8.0e9**, and BOTH arms now run the same six rungs (1K..98304) so the
comparison stays apples-to-apples.

Restoring the 127K rung under speculation needs ~2 GiB of 5090 VRAM. The
cheapest source is the split itself: `cuda_fraction` 0.22 -> 0.10 moves
~26 experts to the B70s and frees ~5.7 GB (47 layers x ~4.7 MB/expert),
at the cost of more wire traffic -- a real, gated follow-up with a number
attached, not a guess.

## The acceptance gate itself: validated, and its noise floor measured

The gate was the last stage never exercised. All four of its components are
now proven against REAL data from this box, not fixtures:

- **Ladder probe** (`benchmarks/decode_ladder_probe.py`, never run before):
  emits exactly the schema `gate.py` validates, including
  `acceptance_pct: null` when nothing is drafted. Live row at ctx=1024:
  TPOT 15.32 ms, decode 65.3 tok/s, TTFT 0.60 s.
- **Gate parser**: accepts real non-speculative output, accepts the
  speculative shape (64.66% acceptance), and correctly REJECTS an
  inconsistent row (drafted > 0 with a null acceptance).
- **Verdict paths**: PASS -> rc=0 with all four gates green; acceptance
  failure -> rc=1 flipping both acceptance gates; latency failure -> rc=1
  flipping the TPOT gate. Verified with real quality evidence.
- **Quality sweep**: 120 prompts captured per arm in **43 s**.

**The most useful number of the night -- the gate's self-noise floor.**
Comparing the server against ITSELF (two 120-prompt captures, same config,
nothing changed) gives **99.17% of prompts under 0.10 nats JSD, p95 =
0.0181, worst single prompt = 0.10357 nats**. Two consequences:

1. A hard `max_jsd <= 0.10` gate would false-fail a PERFECT drafter, because
   this box's own nondeterminism already exceeds it on 1 prompt in 120.
   The gate correctly keys on the *percentage* instead.
2. The percentage gate is well calibrated: the 99.17% floor sits **9.17
   points above** the 90% threshold, so real quality regressions have room
   to register without the box's noise triggering a false alarm.

That is the gate's discriminating power, measured rather than assumed --
and it is the direct consequence of the documented temp-0
non-reproducibility on this stack.

## Capture stage: measured at scale, and a shared-pause trap removed

The capture stage had only ever run on 2 records. Measured on 20 real
corpus records against the capture-armed server:

| metric | value |
|---|---|
| capture rate | **54.5 records/min, 1,313 tokens/s** |
| storage | 1,019 MB / 20 records = **~51 MB per record** |
| projected pilot (814 records) | **0.25 h** -- not the hours assumed |
| projected pilot storage | ~41 GB, inside the 45 GB budget |

Refined budget arithmetic from the real corpus (488 records, mean 1,500
tokens, median 974 -- selector inputs verified present on 488/488): the
45 GB budget admits **~814 records**, not the ~590 estimated from the
2-record sample. Training 814 records is 1,221 optimizer steps = **1.66 h**
at the measured 737 steps/hour.

**Trap removed:** `drafter_capture.py` shared datagen's pause file, so
pausing datagen -- which is exactly what one does to give the capture
server the box -- silently paused capture too (0 records in 1,200 s before
the cause was found). Capture now uses its own `/tmp/sb_capture.pause`.
The 20 records captured during the measurement are real pilot progress:
capture is resumable by corpus id, so the pipeline will skip them.

Night schedule with every stage now measured rather than guessed:
corpus to threshold ~1.5 h -> capture 0.25 h -> training 1.66 h ->
acceptance ~0.5 h = **~4 h**, inside the window.
