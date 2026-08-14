# Shooting Brake - session handoff

**Date:** 2026-08-13 · **Repo:** `~/srswti/shooting-brake`

Read this to resume cold. It contains everything measured, everything
concluded, everything retracted, and the architecture direction we landed on.
Numbers are marked **measured** or **[INFERENCE]**. Where I was wrong during the
session, the correction is recorded rather than the original claim quietly
deleted - the retractions are load-bearing.

---

## 0. The one-paragraph version

Shooting Brake runs an MoE model across an RTX 5090 and (now two) Intel Arc Pro
B70s. Today we: measured a 2.2-2.8x long-context win from a config flag nobody
had turned on; built a grouped NVFP4 MoE kernel that turned out 1.55x **slower**
and parked it; discovered that 66% of the per-layer offload cost is neither
compute nor data movement but cross-runtime orchestration; proved the famous
"65K cliff" was a measurement seam and not real; and - after a second B70
arrived - established that the correct multi-device architecture is a **hub**
(5090 at the centre, no device-to-device traffic, no collectives at all), which
sidesteps every failure mode the in-tree Intel-only dual-B70 journal spent weeks
fighting.

---

### Repo layout (reorganised 2026-08-13)

```
src/            our code - phase0..phase10, plus QuixiCore-XPU (own git remote)
vendor/         third-party clones, each keeping its own .git so we can pull
benchmarks/     harness + benchmarks/matrix/ (was bench-matrix/) + results/
experiments/    one-off probes and scratch
tests/          cross-cutting tests
pyproject.toml  single dependency/install entry point
```

Gotchas from the move:
- The vLLM plugin is an **editable install**; its `.pth` in
  `.venv/lib/python3.12/site-packages/__editable__.shooting_brake_vllm-0.1.0.pth`
  had to be repointed from `phase4/src` to `src/phase4/src`. If the plugin
  stops importing after any future move, that file is why.
- `torch` (2.11.0+cu130) and `vllm` (0.26.0) are **not** pinned in
  `pyproject.toml` on purpose - a plain `pip install torch` would replace the
  CUDA-130 build with a generic one. Install those from the right index first,
  then `pip install -e .`.
- Deleted: `src/phase1/expert_bank_122b.bin` (60 GB, zero references anywhere).
  `src/phase1/expert_bank.bin` (14 GB) is **kept** - it is load-bearing for
  serving and every benchmark, regenerable via `src/phase1/extract_experts.py`.
- Removed duplicate clones: `intel-xpu/uccl` (shallow dup of top-level `uccl`)
  and `phase0/guidellm` (dup of top-level `guidellm`).

## 1. Hardware, measured

Gigabyte X870E AORUS MASTER, BIOS F8 (2026-07-16), Ryzen 9 9950X3D,
59.4 GiB DDR5 (~43 GiB free), Ubuntu, **kernel 7.0.0-29-generic**,
**IOMMU ACTIVE** (35 `iommu_groups`, no `iommu=` on `/proc/cmdline`).

```
5090   01:00.0 -> 00:01.1                       Gen5 x16, own CPU lanes
B70-A  15:00.0 -> 0a:08.0   Gen4 x4   (original card)
B70-B  11:00.0 -> 0a:04.0   Gen3 x4   (added 2026-08-13)
                    both -> 09:00.0   Gen4 x4   SHARED uplink to CPU
```

Both B70 branch bridges report `max_link_speed` equal to `current_link_speed`,
so **Gen3 x4 on B70-B is a hard cap, not idle downtraining**. The `2.5 GT/s x1`
readings that appear on the cards themselves are an idle-parked state - direct
measurement contradicts them (see below). ReBAR is enabled with 32 GiB BARs on
both cards.

### Measured PCIe bandwidth (`experiments/b70_multi_topology.cpp`)

| measurement | result |
|---|---|
| B70-A alone, H2D | **6.24 GB/s** (79% of Gen4 x4 theoretical) |
| B70-B alone, H2D | **2.91 GB/s** (74% of Gen3 x4 theoretical) |
| Both concurrently, aggregate | **4.53 GB/s** |
| Sum if independent | 9.15 GB/s |
| **B70 <-> B70 via host DRAM** | **1.89 GB/s** |
| Peer-access capability query | **YES**, both directions |

Two facts drive everything:

1. **The cards are asymmetric** - 2.1x difference in host bandwidth.
2. **Running both at once is *worse* than the fast card alone** (4.53 < 6.24).
   The shared uplink does not merely fail to scale; it degrades.

Adding the second card **doubled VRAM (96 GB total) and doubled local Intel
bandwidth (~608 GB/s each, ~1216 GB/s aggregate) but did not increase
host<->Intel bandwidth at all.**

> Correction made during the session: I had been using **456 GB/s** for B70
> memory bandwidth in all balance math. The correct figure is **608 GB/s**
> (Puget review). Rebalance anything derived from the old number.

---

## 2. The single most important number

```
B70 decode dispatch, per layer, measured 186.1 us
  |- real MoE kernel work ........  52.8 us   (28%)   measured, xpu_bench --M 1
  |- transport + launch ..........  10.5 us   ( 6%)   measured, synthetic harness
  '- orchestration overhead ...... ~123   us   (66%)  NEVER MEASURED DIRECTLY
```

`experiments/b70_dispatch_latency.cpp` measured the SYCL side end to end:

| measurement | result |
|---|---|
| Empty kernel, submit only | 1.16 us |
| Empty kernel, submit + wait | 5.12 us |
| 12 KiB pinned round trip | 16.84 us |
| **Full decode dispatch** (H2D 4 KiB + kernel + D2H 8 KiB + wait) | **10.46 us** |
| Copy sweep | 1 KiB 15.66 us · 64 KiB 47.05 · 1 MiB 375 · 4 MiB 1428 (5.87 GB/s) |

**The SYCL side is fast.** The missing ~123 us is in the
CUDA -> CPU poller -> SYCL -> CPU -> `cuStreamWaitValue32` handoff. That is
software we own (`src/phase1/b70_provider.cpp`, `src/phase7/`) and it has never had a
timestamp put on it.

> **Retraction:** earlier in the session I ranked "persistent kernel + doorbell"
> as the #1 fix, on the assumption that per-dispatch launch overhead dominated.
> The 10.46 us measurement kills that: removing kernel launch entirely saves at
> most ~5 us of 186, about 3%. Demoted.

---

## 3. What we measured today

### 3.1 Long-context prefill: dispatch vs streaming (the one real win)

Both arms 10/10 requests, zero truncation, one machine, one code state, same
script. `subset:16:8`, 512 output tokens, single stream.
Artifacts: `benchmarks/matrix/longctx/`, writeup `benchmarks/matrix/longctx/RESULT.md`.

| ctx | TTFT dispatch | TTFT stream | ITL disp | ITL stream | e2e disp | e2e stream | gain |
|---|---|---|---|---|---|---|---|
| 65,536 | 30.98 s | **11.21 s** | 6.20 ms | 6.14 ms | 16.5 tok/s | **38.8** | 2.35x |
| 98,304 | 47.98 s | **18.81 s** | 6.38 ms | 6.33 ms | 11.0 tok/s | **25.4** | 2.30x |
| 127,000 | 63.62 s | **26.66 s** | 6.49 ms | 6.50 ms | 8.5 tok/s | **18.7** | 2.21x |

Prefill rate: dispatch 2,116 -> 1,996 tok/s; streaming 5,848 -> 4,764 tok/s.
Decode ITL unchanged within 1%. KV capacity unchanged (842,038 vs 840,052
tokens) because the mirror lives in **host DRAM**, not 5090 VRAM. Costs ~4-6 GiB
of system RAM.

```bash
PREFILL_STREAM=1 bash benchmarks/serve_hybrid.sh
```

Now a documented knob in `benchmarks/serve_hybrid.sh`, **default still 0**
because short prompts have not been re-measured with it on.

Two predictions I got wrong, recorded because they teach something:
- I predicted TTFT ~5-6 s at 127K; measured 26.7 s. The 6.6 GiB transfer is not
  the dominant cost - the streamed-weight *compute path* on the 5090 is
  (4,764 tok/s vs ~25,500 all-CUDA).
- I predicted the advantage would widen with context; **it narrows**
  (2.76x -> 2.39x). Likely attention's O(N^2) becoming visible once dispatch
  stops dominating.

### 3.2 There is no "65K cliff"

The dispatch arm decays smoothly with flat ~2,000 tok/s prefill and reproduces
the old matrix (16.5 vs 16.5, 11.0 vs 11.3, 8.5 vs 8.7 tok/s). The apparent wall
in `benchmarks/matrix/hybrid_131k_c6` sat on a **seam**:

| cell | ran | TTFT | implied prefill |
|---|---|---|---|
| `ctx_32768/concurrent` | **Aug 06** | 1,609 ms | **20,362 tok/s** |
| `ctx_65536/concurrent` | **Aug 13** | 29,489 ms | **2,222 tok/s** |

Twice the context, 18x the TTFT - measured a week apart, on different
motherboards, across commit `1fbecdc0` ("fix silent prefill route loss",
Aug 7 18:36).

**The bug:** VRAM surgery deletes B70-owned experts from the 5090's weight
tensor and *compacts* the survivors. The old prefill path passed **original
global expert ids** into that compacted tensor, so B70-owned routes fell out of
the weighted sum. Prefill ran at all-CUDA speed (20,320 tok/s at 32K) while
producing ~0.49 nats/token worse logprobs and **identical output tokens**. Fast
because it was wrong; invisible because nobody checked logprobs; unreported
because it made the benchmark look good.

Arithmetic check: `512 / (1.613 s TTFT + 2.944 s decode) = 112 tok/s`, matching
the 114.6 in the heatmap. Post-fix the same cell should read ~27 tok/s.

**The 1K-32K cells are invalid and must be re-run before being quoted.**

### 3.3 Decode placement trade (`benchmarks/results/offload_full/`)

| placement | decode tok/s | % all-CUDA | KV tokens | KV x | B70 route share |
|---|---|---|---|---|---|
| all-cuda | 252.6 | 100% | 387,760 | 1.00x | - |
| subset:8:8 | 214.7 | 85.0% | 786,000 | 2.03x | 96.1% |
| subset:16:8 | 188.3 | 74.5% | 1,129,744 | 2.91x | 96.4% |
| subset:24:64 | 171.4 | 67.9% | 1,238,736 | 3.19x | 73.0% |
| split:128 | 159.9 | 63.3% | 1,146,512 | 2.96x | 50.7% |

Per-layer exposure: all-CUDA ITL 3.958 ms, `subset:16:8` 5.308 ms, delta
1.350 ms over 16 active layers = **84.4 us exposed per layer** (of 186.1 us
service; ~100 us hides under CUDA expert compute).

**The split is far off throughput-optimal.** Balanced when both finish together:

```
f_5090 = 1792 / (1792 + 608) = 75%      (one B70)
f_5090 = 1792 / (1792 + 1216) = 60%     (two B70s)
```

`subset:16:8` gives the B70 **96.4%**. That is deliberate - VRAM freed scales
with B70 share - but it is exactly why decode sits at 74.5%. **Offload the
minimum that makes the model fit, never the maximum possible.**

### 3.4 Grouped NVFP4 MoE kernel - built, correct, parked

`src/QuixiCore-XPU/kernels/moe/nvfp4_moe/variants/xpu_sycl/nvfp4_moe_grouped.sycl.cpp`

Sorts routes by expert, runs each expert's rows as a DPAS GEMM so a dequantized
weight tile is reused across 32 rows. Cuts logical weight reads ~28x at M=2048.

**Result: 1.55x SLOWER.** 91.1 ms vs `split` 58.9 ms (M=2048, f16, K=2048,
I=512, E=256, top_k 8, spread routing, nonzero weights).

Stage profiling (`nvfp4_moe_grouped_profiled_sycl`, per-event timestamps):

| stage | ms | share |
|---|---|---|
| clears + histogram + scan + join + scatter | 0.15 | **0.2%** |
| `gate_up` GEMM | 71.7 | 78.1% |
| `down` GEMM | 22.0 | 21.5% |
| queue idle | 0.01 | 0.0% |

The sort prepass is free. The GEMMs are the problem, and **why** is still
unknown - three attempts to ablate weight-load+dequant were all degenerate
(compile-time-synthesisable tiles; reductions cancelling to exactly zero; values
collapsing into denormal range). No valid bound exists on the dequant fraction.

Gated off: `nvfp4_moe_grouped_profitable()` returns false unconditionally, and a
smoke test asserts it stays false. Details:
`src/QuixiCore-XPU/perf/results/nvfp4_moe_grouped.md`.

**Two fp16 bugs found and fixed** (would have made the experimental path wrong
if enabled):
1. **Overflow.** ModelOpt's global scale carries a 2^22 fixup, so a fully
   dequantized weight is ~8e4 - past fp16's 65504. Every tile element would have
   been `inf`.
2. **Subnormal collapse.** The decoders' 2^-22 scaling puts the tile in fp16
   subnormals; measured **3% error at block scale 0x3B, 18% at 0x2D**, and
   1.1e-3 end-to-end against a 1.7e-5 bound. **An exponent problem, not a
   mantissa one** - bf16 (8 exponent bits) is bit-identical either way, so an
   fp16-free test would never see it.

Fix: apply `kGlobalScaleFixup` *before* narrowing, so the staged tile holds the
true `e2m1 * e4m3` product (max 2688, needs <=6 significant bits, therefore
**exact** in both fp16 and bf16) and the per-expert global scale multiplies the
fp32 accumulator in the epilogue.

Correctness: matches the in-tree CPU reference at **3.2e-7 (f16)** and
**3.1e-6 (bf16)** - exactly one rounding of the SwiGLU intermediate.

### 3.5 Test and benchmark infrastructure was quietly broken

Both fixed; both affect `fused`/`split` measurements, not just the new kernel.

- **Fixture used power-of-two block scales** (`0x38`=1.0, `0x30`=0.5). Those
  round-trip cleanly through *any* precision bug. Replaced with
  `{0x3B, 0x2D, 0x35, 0x43}` (nonzero mantissas). Verified by reverting the fix:
  f16 fails at 1.11e-3 vs bound 1.66e-5, 67x over.
- **Benchmark memset `expert_ids` to 0** - every route to expert 0, served out
  of L2, and all-zero weights and activations so it timed arithmetic on zeros.
  Now spread routing, nonzero host data, and a fixture check that throws if any
  output is non-finite or all-zero. That check caught two degenerate ablations
  that would otherwise have produced bogus timings.

---

## 4. Model and format facts

**Qwen3.6-35B-A3B** (`src/phase0/capability_manifest.yaml`): hidden 2048, 40 layers
(32 NVFP4 MoE + 8 FP8), 256 experts, top_k 8, `moe_intermediate_size` 512,
shared expert 512. `w13 = [2I, K] = [1024, 2048]`, `w2 = [K, I] = [2048, 512]`.
Expert = 1.69 MiB packed. **Whole bank = 13.5 GiB** (`src/phase1/expert_bank.bin`).

### NVFP4, verified against ModelOpt source

`modelopt/torch/quantization/qtensor/nvfp4_tensor.py:399-405`:

```
weight = e2m1_value * e4m3_block_scale * weights_scaling_factor_2
weights_scaling_factor_2 = amax / (E2M1_MAX * m_fp8) = amax / (6 * 448)
```

**The 2^22 fixup is NOT part of the format.** QuixiCore's decoders reinterpret
bits as fp16 *without rebiasing the exponent* - branchless, but they return
values scaled by 2^-14 (e2m1) and 2^-8 (e4m3). Their product is the true weight
times 2^-22, and `kGlobalScaleFixup = 4194304` undoes exactly that. It belongs
to the decoder, not to NVFP4.

**Format sizes are equal**, so per-device formats cost nothing in capacity:
- NVFP4: 4 bits + E4M3/16 = **4.5 bits/param**
- int4 g32 + fp16 scale: 4 + 16/32 = **4.5 bits/param**

**The B70 has no FP4 tensor path.** Xe2 DPAS eats fp16/bf16/int8/int4, so FP4 is
storage-only there and always pays dequant. **int4 is the B70's native shape.**

---

## 5. Prior art found in-tree (do not re-derive)

### 5.1 `intel-xpu/vllm-xpu/b70_ai_things/` - dual-B70 journal

**Different physical machine** (Threadripper 1950X -> migrated Ubuntu 26.04 /
kernel 7.0.0-22 / **IOMMU OFF**, hostname `b70s4dayz`). Topology numbers are NOT
transferable to the current 9950X3D box. **Purely Intel-only - no NVIDIA card
ever appears in-process.**

It dead-ended on exactly the problem we now face, and the numbers are decisive:

| config | decode tok/s |
|---|---|
| TP=1 (one card) | 7.84 |
| **TP=2 (two cards)** | **4.18 - worse than one card** |
| PP=2 | 6.11 (+46% over TP=2) |
| DP=2, C64 | ~525 aggregate |

Its own MoE analysis (`JOURNAL.md:2477-2495`), per token:

| scheme | collectives/token | cost |
|---|---|---|
| TP=2 | ~80 all-reduces | ~23 ms -> **~25 tok/s ceiling** |
| EP=2 | ~80 all-to-alls | >= TP |
| **PP=2** | **1 handoff** | **~0.3 ms, negligible** |

Other findings:
- `--enable-expert-parallel` on vLLM-XPU exists but is **crippled**
  (llm-scaler bugs #477, #479, #382, #489).
- 35B-A3B int4 = 19.6 GiB **fits one card**, so they chose int4 + 2x
  data-parallel.
- Host-staged allreduce plateaus **1.16-1.22 GB/s**; **P2P allreduce
  9.43-9.77 GB/s (8.4x)** - but P2P in a real vLLM serve throws
  `UR_RESULT_ERROR_DEVICE_LOST` and **wedges the driver**; recovery needs
  `modprobe -r xe; modprobe xe`. Production stayed `CCL_TOPO_P2P_ACCESS=0`.
- `CCL_ENABLE_SYCL_KERNELS=0` is required for init stability (vLLM #41663) but
  **breaks SYCL-graph recording** of collectives; `=1` unlocks capture but
  gives up the stability margin. That dilemma consumed much of the journal.
- Production ultimately settled on **NVFP4 TP=2** for stability after W8A8 TP=2
  crashed twice under long-context load.
- The widely-quoted "~362 tok/s" is a **third-party** figure from GitHub issue
  #41663, not measured there.

### 5.2 Puget Systems, 4x B70 (external)

`intel/llm-scaler-vllm:0.14.0-b8.2.1`, oneCCL, TP=4, FP16:

- **Qwen3.6-35B-A3B MoE: 16.3 tok/s single-user**, 122 at C8.
- Required: `VLLM_WORKER_MULTIPROC_METHOD=spawn` (fork-unsafe SYCL context),
  `CCL_TOPO_P2P_ACCESS=0` (P2P caused PCIe `RxErr` + `Engine reset: bcs`),
  remove `/etc/OpenCL/vendors/intel64.icd`, `NEO_ReadDeviceBinaryBuiltins=0`.
- B70 = **608 GB/s**, 32 GB, $949; 5090 = 1792 GB/s, 32 GB.

**Shooting Brake gets 188.3 tok/s on the same model family with one 5090 and one
B70.** Different quantization (their FP16 ~70 GB vs our NVFP4 ~25 GB) so not
like-for-like on capacity - but as evidence that a fast hub beats pure-Intel
tensor parallelism, it is decisive.

### 5.3 Single-device audit of our own stack

Everything assumes exactly one B70:

- `src/phase1/b70_provider.cpp:150-176` `select_b70()` - enumerates and picks
  whichever B70 comes first. **No parameter, no env var, no BDF selector.**
- `SHOOTING_BRAKE_B70_DEVICE` is a **boolean "enable offload"**, not a device
  index. Setting it to 1 to target card 2 does nothing.
- One `sycl::queue` per provider `Impl`; no pool, no second device slot.
- The C ABI (`sb_b70_*`) carries **no device index** in any signature.
- Python `Device` enum is a flat 3-way (CUDA/B70/CPU) - the remote tier is a
  boolean, not an id.
- Bank residency is one resident set per provider load, so two providers must
  each bind a distinct device **and** a disjoint resident-expert subset.

**This is the single blocking prerequisite for using the second card.**

---

## 6. The architecture: hub, not mesh

### 6.1 The realization

With the 5090 as hub, **no collective is needed at all**:

```
5090 (attn, router, KV) --4 KiB--> B70-A experts --partials--> 5090 sums
                        --4 KiB--> B70-B experts --partials-->
```

Each B70 owns a **disjoint** expert set, computes its own partials, returns
them. The 5090 performs the combine - which it already does. That is
scatter/gather to a hub, not all-reduce.

Consequences, each removing a documented failure mode:

- **B70 <-> B70 traffic is zero.** The worst number on the machine (1.89 GB/s)
  is not on the critical path.
- **No oneCCL**, so none of the journal's 11 failure modes.
- **No P2P**, so no IOMMU-off reboot and no driver wedge.
- **No graph-capture-vs-collective dilemma.**

The journal hit a wall because it had no hub. We have one. **That is the whole
thesis.**

### 6.2 Why every existing technique is the wrong shape

| library / technique | assumes | why it fails here |
|---|---|---|
| NCCL | NVIDIA only, NVLink/PCIe P2P | cannot see the Intel cards |
| oneCCL | Intel only, collective-shaped | right for Intel<->Intel, which we avoid needing |
| UCCL | RDMA/NIC fabric (400G CX-7, RoCE, IBGDA), cross-rack | **there is no network in this problem** |
| TP | homogeneous devices, fast fabric, all-reduce per layer | ~80 all-reduces/token; measured 0.53x |
| EP | all-to-all per layer | >= TP cost; crippled on XPU |
| PD disaggregation | KV transfer between nodes | KV for 65K tokens far too large for 6.2 GB/s |

They all share three assumptions we violate: **same vendor**, **fast symmetric
interconnect**, **collective-shaped communication**. Our constraints invert the
objective:

1. **Heterogeneous** (~3x speed difference, and the two B70s differ 2.1x in link
   bandwidth) -> every split must be **throughput-weighted**, never even.
2. **Slow, contended, asymmetric interconnect with generous local memory** ->
   minimize **round-trip count**, not bytes.
3. **Device- and quant-agnostic** -> each device runs its native-best format and
   a kernel written for it.

**"Minimize round trips, split by throughput ratio, native format per device,
hub topology"** is not a configuration of TP/EP/PD. It is a different objective
function, and it does not appear to be written down anywhere.

### 6.3 The transport to build

Not a collective library. A **hub-and-spoke activation transport**, spec derived
directly from measurement:

| requirement | measured justification |
|---|---|
| Star topology, 5090 at centre, never B70<->B70 | 1.89 GB/s D2D vs 6.24 GB/s to host |
| Throughput-weighted asymmetric split | 6.24 vs 2.91 GB/s -> ~68/32, never 50/50 |
| Stagger transfers, never blast both | concurrent 4.53 < single card 6.24 |
| Latency-first, not bandwidth-first | 12 KiB payloads; the enemy is 123 us |
| Cross-runtime doorbell in pinned host memory | CUDA and Level Zero both poll; no CPU thread in path |
| Topology self-discovery | link gen/width per device, so splits are derived not hardcoded |

Roughly 2,000 lines. The cross-runtime problem is the interesting part: CUDA and
Level Zero cannot address each other's memory, but **both can map host pinned
memory**, so the arena plus doorbell/completion flags in that shared region is
the whole substrate.

---

## 7. Attack vectors, ranked (post-measurement)

| # | attack | targets | effort | status |
|---|---|---|---|---|
| 0 | **Device-selectable provider** | prerequisite for card 2 | days | **not started** |
| 1 | **Instrument the dispatch path** | locate the ~123 us | hours | not started |
| 2 | Sequence-level expert locality | may remove trips entirely | hours | not started |
| 3 | Heterogeneous pipeline parallelism | 48 round trips -> 2 | weeks | not started |
| 4 | Microbatch pipelining | residual exposure | week | not started |
| 5 | Expert prefetch by prediction | the serial dependency | weeks | not started |
| 6 | Multi-queue on B70 | batched throughput | days | not started |
| - | ~~Persistent kernel + doorbell~~ | **demoted** - saves ~5 us of 186 | - | measured out |

**#2 is the cheapest and most informative**: `benchmarks/route_stats.csv` proves
there is no *global* hot expert set (147 of 256 for 80% coverage), but nobody has
checked whether **consecutive tokens in one sequence** reuse experts. If they do,
an LRU expert-weight cache in 5090 VRAM skips round trips entirely. Hours of
work against data already on disk.

**Deliberately not on the list: enabling P2P.** The capability query now reports
YES on this box (kernel 7.0, IOMMU **on**) - which is the exact cell the journal
never tested (it tested 6.18+IOMMU-pt=False and 7.0+IOMMU-off=True). But the
recorded failure mode is a driver wedge needing `modprobe -r xe`, and the hub
design does not need it. **The peer copy was NOT attempted.**

---

## 8. Capacity plan for a bigger model

Whole 35B expert bank is 13.5 GiB against 64 GiB of Intel VRAM. **The Intel side
is mostly empty.**

| tier | ceiling | params |
|---|---|---|
| Experts -> 2x B70 | ~60 GiB | ~107B expert params |
| Non-expert + KV -> 5090 | ~20 GiB weights + ~11 GiB KV | ~20B non-expert |

A **70-120B NVFP4 MoE** fits with room for KV. That regime **has never been
benchmarked** - every measurement in this repo compares against an all-CUDA
baseline that fits on one card, where the B70 can only look like overhead.

**Streaming cost scales with offload**: 6.6 GiB -> 1.06 s per forward today;
~30 GiB -> ~4.8 s. **Host DRAM (59.4 GiB), not VRAM, becomes the binding
constraint.**

Format plan: quantize each format **from the bf16 source** (never NVFP4 ->
int4 - it compounds two lossy steps). 5090 keeps NVFP4 (native Blackwell FP4
tensor cores; unverified whether this vLLM build actually uses them - worth a
15-minute check). B70s get **int4**, because
`llm-scaler/sglang/custom-esimd-kernels/moe_prefill_int4.sycl` already
implements the exact grouped, sorted, DPAS-tiled MoE prefill kernel we need -
for int4, not NVFP4.

---

## 9. Files touched this session

| file | change |
|---|---|
| `README.md` | rewritten as single source of truth; old README preserved below a `# DEPRECATED` marker |
| `HANDOFF.md` | this file |
| `benchmarks/matrix/longctx/RESULT.md` | dispatch vs streaming A/B writeup |
| `src/QuixiCore-XPU/perf/results/nvfp4_moe_grouped.md` | grouped kernel post-mortem |
| `src/QuixiCore-XPU/kernels/moe/nvfp4_moe/variants/xpu_sycl/nvfp4_moe_grouped.sycl.cpp` | new grouped kernel (parked) |
| `src/QuixiCore-XPU/kernels/moe/nvfp4_moe/variants/xpu_sycl/nvfp4_dequant.hpp` | shared decoders + fixup rationale |
| `src/QuixiCore-XPU/include/quixicore/xpu/ops.hpp` | grouped op decl + STATUS note |
| `src/QuixiCore-XPU/src/dispatch/moe.cpp` | grouped dispatch; `profitable()` forced false |
| `src/QuixiCore-XPU/tests/xpu_ops_smoke.cpp` | realistic block scales; multi-block/skew/empty-expert test; safety-gate assertion |
| `src/QuixiCore-XPU/perf/harness/xpu_bench.cpp` | spread routing, nonzero data, fixture check, stage profiling, grouped variant |
| `benchmarks/serve_hybrid.sh` | `PREFILL_STREAM` knob, documented with measured numbers |
| `experiments/b70_dispatch_latency.cpp` | dispatch latency decomposition harness |
| `experiments/b70_multi_topology.cpp` | two-B70 topology/bandwidth/peer probe |
| `intel-xpu/oneCCL`, `intel-xpu/uccl` | cloned for reference |

Verification state: `quixicore_xpu_ops_smoke` **PASS**, 0 failures, with the new
safety-gate assertion active. Server torn down, both GPUs free.

---

## 10. Open questions for the next session

1. Where do the **123 us** go? Nothing else is decidable until this is measured.
2. Do **consecutive tokens in a sequence** reuse experts? (`route_stats.csv`)
3. Why are the grouped kernel's **GEMMs** slow? All three ablations were
   degenerate; needs a runtime device table with a non-power-of-two length.
4. Does the 5090 actually use **native Blackwell FP4 tensor cores** in this
   vLLM build, or silently dequantize to bf16? Decides the format strategy.
5. What does a **70-120B NVFP4 MoE** actually do on this machine?
6. Is the peer copy safe on kernel 7.0 + IOMMU on? (untested cell; only attempt
   deliberately, with the driver-wedge recovery path ready)
