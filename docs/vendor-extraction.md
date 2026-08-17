# Vendor extraction — what we take, what it buys, where it plugs in

Decision document from a 12-agent deep recon over the vendored repos, plus live
verification against our own checkpoint and running server. Written after both
MoE kernel bake-offs closed negative, so the framing is: **the cheap kernel
levers are spent; what is actually left, priced honestly.**

## How to read this

Every claim carries provenance. Three tiers, never blurred:

* **[measured-here]** — we measured it on this box, artifact in `benchmarks/results/`.
* **[measured-elsewhere]** — someone measured it on named hardware. First-class
  evidence *only* if that hardware is sm_120 family (see transfer rule); label
  the card otherwise.
* **[code-verified]** — read in source at file:line; a contract, not a number.
* **[claim]** — README/paper assertion, unreproduced. Treated as a hypothesis.

**The sm_120 transfer rule.** The RTX PRO 6000 Blackwell Workstation is the same
sm_120 family as our RTX 5090 (GB202 die). Tuning, kernel selection, arch guards
and per-device measurements taken on a PRO 6000 **transfer to us**. Numbers from
sm_100 (B200 datacenter) **do not** — different instruction set (tcgen05/TMEM),
different builder specializations. Multi-GPU collective-scaling numbers from
any card do not transfer to our single-CUDA-GPU box.

## Live verifications run during this recon

These four checks changed decisions, so they lead.

1. **The bf16 base checkpoint exists and is genuinely ours.**
   `~/.cache/huggingface/hub/models--0xSero--Qwen3.5-88B` (164 GB) is
   `Qwen3_5MoeForConditionalGeneration`, 48 layers, hidden 3072, 180 experts,
   moe_intermediate 1024, top_k 8 — geometry identical to
   `srswti/axe-superveloce-88b-nvfp4a16`. **[measured-here]**
   This single artifact gates three otherwise-blocked levers (rotation
   requantization, FP6-W6A8, draft-head training). It was nearly deleted during
   the disk crunch. **Protect it.**
2. **Native MTP does not exist for this model.** Zero `mtp.*` keys in the
   quantized checkpoint *and* zero in the bf16 base. **[measured-here]**
   Consequence: MTP speculative decoding is unavailable by construction;
   speculation means training an EAGLE3 head, not setting a flag.
3. **Our server is running the Triton/FLA GDN fallback.** Boot log:
   `Using Triton/FLA GDN prefill kernel (requested=auto, head_k_dim=128)`.
   **[measured-here]** Cause is an arch gate, not a capability gap — see the
   sm_120 gate finding below.
4. **Checkpoint tensor formats confirmed** (cross-checked by a scout against the
   live index): `lm_head` + shared experts are NVFP4 W4A16; `q/k/v/o_proj` and
   `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` are **per-tensor-scalar static
   FP8** (no group size). **[code-verified]** This is what makes the FP8 dense
   GEMM candidate below a layout match rather than a guess.

---

# TAKE NOW — zero or near-zero cost, high confidence

### 1. Cap the decode B70 at ~150 W
MoE decode on this card **self-limits to ~140 W regardless of cap** — it is
bandwidth-bound, not frequency-bound. Measured on the same B70 SKU: 150 W gives
**125.7 tok/s vs 230 W's 115.4 tok/s — i.e. 8.2% *better* at lower power**
(`vendor/intel-arc-pro-b70-inference-cookbook/docs/POWER-SWEET-SPOTS.md`).
**[measured-elsewhere, same card class]**
*Where:* `hwmon power1_cap` on the decode B70, before the server boots.
*Cost:* none, no code. *Caveat:* dense (non-MoE) shapes scale +18-30% with
power, so this is MoE-decode-specific.
Related pitfall: the `xe` driver has no `gt_min/gt_max` clock control and its
PMU frequency readback is unreliable (reports 3400 MHz against a 2400 MHz
request) — never gate a design decision on reported clock.

### 2. b12x's pure-torch NVFP4/MXFP8 **encode** + swizzle helpers
`vendor/b12x/b12x/_lib/intrinsics.py` (~lines 55-320) holds
`quantize_grouped_nvfp4_torch`, `swizzle_block_scale`,
`fp4_quantize_values_torch`, `pack_grouped_fp4_values`,
`quantize_grouped_mxfp8_torch`, `pow2_ceil_ue8m0_torch` — **plain torch,
CPU-callable**, proven byte-exact against their own GPU TMA kernel in their
tests. **[code-verified + their tests]**
*Where:* `src/phase1/build_b12x_bank.py`. We own a validated CPU *dequantizer*;
this fills the **encode** direction, which we currently hand-roll.
*Cost:* surgical extraction only — the same file carries thousands of lines of
`@cute.jit` GPU-only kernels, so **do not import wholesale**.

### 3. Their A/B timing primitive
`vendor/b12x/validation/cutlass_migration/core/exact_cache_abba.py` (~393-1008):
1%-trimmed mean gated on `nvidia-smi` P-state, throttle mask, and clock delta,
with ABBA ordering to cancel drift. **[code-verified]**
*Where:* our bake-off harness (`prefill_floor_bench.py`), which currently uses a
plain median of 5-7 with **no throttle gate** — a real hole, since our 5090 is
the same card that thermally varies.
*Cost:* lift the primitive only. **Reject** the surrounding SHA-pinned
104-position formal-release machinery as wildly disproportionate to our needs.

### 4. PyCuTe as a layout-verification aid
`vendor/CuTe/pycute` is confirmed pure Python layout algebra —
`layout.py`/`algebra.py`/`swizzle.py` plus SVG/LaTeX visualizers, **zero
kernels, zero CUDA**. **[code-verified]**
*Where:* verifying bank plane and swizzle math offline. We spent real time this
session deriving flashinfer↔b12x swizzle adapters empirically by index-tracing;
this is the tool for doing that symbolically next time.
*Cost:* none. Not a runtime dependency.

### 5. Graph-capture discipline + the pitfall catalogue
Portable design rules extracted from b12x's transport layer and the dual-B70
journal: stable-address fixed-capacity buffers, "prepare channels before
capture", stream-affine binding, GC-quarantine of abandoned runtimes.
**[code-verified]** Plus the XPU landmine table in the *Meta-findings* section.
*Where:* our graph-replayed decode path and any future capture work.

### 6. Never sync `vendor/vllm` over the installed vLLM
They are the same 0.27.1 line and line-for-line identical across every
subsystem we touch, with **one exception**: the *installed* tree has
`calculate_kv_scales` wired, including a **hybrid-model auto-disable guard**
(`models/config.py:450-463`, citing vllm#37554: "uninitialized recurrent state
corrupts scales during the calibration pass") that the vendored tree lacks
entirely. **[code-verified]**
*Action:* treat `vendor/vllm` as read-only reference. Syncing its
`cache.py`/`attention.py`/`gpu_model_runner.py` forward would silently drop a
safety guard written for **exactly our GDN/Mamba + fp8-KV architecture**.

---

# BENCH FIRST — real candidates, ordered by expected value

### 1. B70 int4 GEMV bandwidth audit — the highest-value decode diagnostic
A prior dual-B70 campaign measured the **"int8 GEMV trap"**: at M=1, oneDNN/MKL
int8 GEMV reaches only **309-361 GB/s (51-59%)** of the B70's **608 GB/s** peak,
while bf16 GEMV reaches **434-460 GB/s (71-76%)**
(`vendor/intel-xpu/vllm-xpu/b70_ai_things/zml/W8A8_SWEEP_RESULTS.md:12-38`).
**[measured-elsewhere, same card class, different kernel]**
*Why this is first:* our decode deficit is 1.35-1.63× and we have never measured
our own per-route int4 GEMV's achieved bandwidth. If we are also at ~55% of 608
GB/s, the decode gap is a **bandwidth-utilization problem we can fix**, not
silicon. If we are at 75%, the gap is real and we stop looking here.
*Cost:* an afternoon. One kernel, one counter, no integration.
*Related hard warning:* naive weight-only int8 (materialize bf16 weights, then
bf16 GEMM) measured **0.5-1.2×, sometimes slower than bf16** — fused
epilogue-dequant is mandatory, not optional
(`b70_ai_things/zml/ZML_INT8_PERF_HANDOFF.md:88-93`).

### 1b. Register the doorbell buffers with the B70's SYCL context — NEW, cheap
**Found from a fresh `vllm-xpu-kernels` pull (upstream PR #519, commit
`f1c4861`), and it is the missing half of our own biggest win.**

Intel just landed `xpu_host_register` / `xpu_host_unregister`
(`csrc/utils/host_register.cpp`) — explicitly "the SYCL counterpart of
`cudaHostRegister`", wrapping `syclex::prepare_for_device_copy(ptr, n_bytes,
ctx)`. Their own docstring describes our exact situation **[code-verified]**:

> "Host memory that cannot be obtained from the caching host allocator --
> notably a shared mmap region -- is otherwise pageable, which forces staged
> (H2D) or synchronous (D2H) copies. **Registration is scoped to the device's
> context, so each device that transfers to or from the range must register it
> separately.**"

That last clause is the finding. Our doorbell staging buffers
(`routed_experts.py:926,930,1042,1046,1105`) are allocated `pin_memory=True`
— which pins them into the **CUDA** caching host allocator. The B70's SYCL
context knows nothing about that registration, so from the Arc's side those
buffers are **pageable**, forcing staged H2D and *synchronous* D2H on **every
doorbell round trip, every layer, every decode step**. Verified three ways
**[measured-here]**:

* `prepare_for_device_copy` / `release_from_device_copy` are **absent from our
  compiled `src/phase7/libsb_b70_provider.so`** (checked with `strings`) — we
  have never registered anything on the XPU side.
* the provider takes raw host pointers (`const sycl::half* hidden`,
  `b70_capi.cpp:44,270,352`) and copies from them.
* the primitive **is available in our installed oneAPI 2026.1**
  (`sycl/usm.hpp:341`) — no toolchain upgrade required.

*Where:* one registration call per buffer at provider init, for each B70
context that touches it. Our own CUDA-side precedent is exactly this move
(pageable 18.5 -> registered 53.9 GiB/s).
*Buys (unmeasured, order-of-magnitude):* the win is removing **fixed
per-transfer overhead**, not bytes — at M=1 the payload is only 6 KiB, so
bandwidth is irrelevant and the cost is the staging bounce plus a D2H
serialization point. At 48 layers x 2 transfers per decode step, a 5 us
per-transfer overhead is ~480 us (**~4% of our 11.88 ms ITL**); 20 us would be
~1.9 ms (**~16%**). Decode is our worst metric (1.35-1.63x behind), and the
doorbell wait itself is already hidden (B70 44 us vs CUDA MoE 100 us), so any
saving here is not masked by overlap.
*Cost:* a few lines in the provider + a rebuild. Pairs with item 1: that audit
measures the *kernel's* achieved bandwidth, this one measures the *transport's*
overhead — two afternoon experiments on the same metric, from opposite ends.
*Secondary:* the same primitive would let the B70s DMA the 27.4 GiB expert bank
straight from mmap'd page cache at boot instead of a staged copy — a boot-time
win (ours is ~130 s), not a serving one.

**The other two commits in that pull are off our path:** `#527` rewrites
`csrc/moe/grouped_topk.cpp` (718 lines) but our router runs topk on the **5090**
and dispatches only expert IDs to the B70; `#526` adds an XPU paged-decode
config tuple, and our attention runs on the 5090 via FlashInfer.

### 2. GDN prefill: route to flashinfer's sm_120 kernel via our plugin
**This is a bug-class hit we found live.** vLLM's selector
(`qwen_gdn_linear_attn.py:116-133`) sets `supports_flashinfer` only for
`is_device_capability(90)` (Hopper) or `is_device_capability_family(100)`
(sm_10x). Our sm_120 satisfies **every other condition** — `head_k_dim == 128` ✓,
CUDA runtime ≥ 13 ✓ — but fails the family check, so we silently fall back to
Triton/FLA. **[code-verified + measured-here in our boot log]**
Meanwhile flashinfer ships `delta_rule_dsl/delta_rule_sm120.py`
(`chunk_gated_delta_rule_sm120`) — a GDN kernel named for **our exact card** —
present in our **installed** 0.6.16.post3. **[code-verified]**
*Where:* `ChunkGatedDeltaRule` is a `PluggableLayer`, the same out-of-tree
mechanism our plugin already uses to replace `RoutedExperts`/`MoERunner`. So this
needs **no vLLM patch** — an OOT override behind our own env flag.
*Buys:* bounded by GDN's share of prefill busy time — `gdn_attention 0.070 s` of
0.82 s GPU-busy at 8K (`attribution_8k.json`), so **≤7% of an 8K TTFT**.
Honest: modest for prefill. Decode-side GDN selection is a **separate** path we
have not yet located — worth finding, since decode is the open front.
*Gate:* firsttok/logprob envelope — this changes a numerics path.

### 3. `gemm.tensor_fp8_linear` for our attention/linear-attn projections
`vendor/b12x/b12x/gemm/tensor_fp8_linear/_kernel.py` — a static per-tensor E4M3
dense GEMM for sm_120 whose `pack_weight(weight_fp8, input_scale*weight_scale)`
contract is a **verbatim match** for our `q/k/v/o_proj` and `linear_attn.*`
tensors (verified above). **[code-verified]**
Their postmortem (`docs/sm120_dense_fp8_deepgemm_port.md`) records a tile-selection
fix — (128,128)→(64,128) M-independent default — taking it from **1.44× slower to
1.6× faster than FlashInfer CUTLASS, measured on an RTX PRO 6000**.
**[measured-elsewhere, same sm_120 family → transfers]**
*Where:* the `unattributed_gemm 0.149 s` line in our own 8K attribution — the
second-largest busy category after MoE, and the one we have never attacked.
*Cost:* real integration (a non-MoE dense path in our plugin), and their doc also
narrates a weight-requantization bug (dropped power-of-two re-quant step) worth
reading as QA before we repeat it.

### 4. Predictive expert placement — colibri's PILOT prefetcher
A fully measured algorithm for exactly our unspent "hotness-ordered placement"
lever: **71.6% one-layer-ahead routing recall on a real 256-expert MoE**, via
EMA-smoothed router logits, a two-step correction, and an LFRU eviction guard.
**[measured-elsewhere, different model]**
*Where:* the B70 doorbell prefetch path and/or the streamer's expert ordering.
*Buys:* fewer streamed bytes and less remote compute — compounds with **any**
kernel, which is precisely why it survived both negative bake-offs.

### 5. Adaptive local/remote split — exllamav3's `MoeCpuHost`
`moe_cpu_host.py` implements a **bandwidth-probed break-even threshold** that
adaptively decides which experts stream over a PCIe link versus compute locally
— architecturally near-identical to our split, except ours is a **static 54/126**
chosen once. **[code-verified]**
*Where:* tuning the split across our **two asymmetric links** (Gen4 x4 and
Gen3 x4) instead of assuming symmetry.

### 6. MoE-decode sweep methodology + the NVFP4-native micro kernel
`vendor/b12x/scripts/sweep_moe_decode_max_active_clusters.py` is the closest
in-repo precedent for a **MoE-decode-specific** tuning sweep: CUDA-graph replay,
`torch.cuda.Event` timing, `_mean_ci` = mean ± z·SEM (not trimmed).
**[code-verified]** Clone the methodology for our own decode knobs.
Correction to an earlier assumption: `MoEMicroKernelW4A16SmallMDirect` is
**W4A16-only**, while our local 54 experts run NVFP4 — the quant-mode-correct
analog is `MoEMicroKernelBackend` (`moe/_shared/kernels/micro.py:382`).
*Hazard:* unlike the w4a16 branch, the NVFP4 branch **does** enter the
`data_ptr`-keyed `_WEIGHT_CACHE`/`_W13_NORMALIZED_STORAGES` caches
(`_impl.py:4508-4655`, `:4366`) — a live staleness hazard under our rotating
arenas. Pre-derive and pass explicitly, or don't take it.

### 7. b12x's PCIe **DMA discipline** — the transferable half of `comm.pcie`
The collectives are dead for us (CUDA-IPC, needs >=2 CUDA GPUs), but the DMA
layer underneath is a separate concern and it is the good part.

* **CE beats SM-copy by 1.65x on PCIe.** Their own header: "NCCL's SM-copy
  transport sustains ~34 GB/s on this fabric while CE peer copies run at
  ~56 GB/s" (`pcie_dma.py:1-9`). **[measured-elsewhere]** This retro-validates
  our registered path (18.5 pageable -> 53.9 GiB/s registered = the CE path)
  and, forward-looking, it is a **guard**: if a future transfer silently lands
  on an SM-driven copy kernel we lose ~40% with no error. The 9 us submit-wall
  we measured is the signature to assert on.
* **Separate CE stream + flag stream; device-resident *monotonic* flags;
  `FLAG_STRIDE = 128`** (`pcie_dma.py:37,184-205`). **[code-verified]** The
  monotonic never-reset counter is precisely what lets their graphs "replay
  without host patching" — the property our decode-graph replay needs, and the
  same class as the XPU NEO-overflow landmine (un-reclaimed events). If we
  build the persistent doorbell, this is the proven shape: sync traffic on its
  own stream so it never queues behind a 584 MiB copy, 128 B stride to avoid
  false sharing between slots.
* **`recommend_prefetch_depth`** (`overlap_probe.py:410-426`) makes prefetch
  depth a *measured policy with a kill condition* — enable depth-1 only if some
  context benefits AND none regresses past a threshold. **[code-verified]**
  We hardcode depth-1 (2 arenas) because it fit VRAM, and never re-checked
  after register-DMA cut the transfer from 1.48 to 0.51 s/pass. Depth-0 at some
  contexts would free 584 MiB — material at 1.60 KV seats.
* **Correctness-gate-before-timing** and median-of-slowest, plus "measure real
  traffic on the deployed topology instead of inferring from PCIe link labels".
  Our floor bench times first and gates separately; one harness that refuses to
  report a number for a wrong kernel is strictly better.

### 7b. The wire codec — DEMOTED, keep for the record
`_dma_kernels.py:1468-1502` (`DmaKernels._quant`/`_dequant_store`) are
standalone elementwise kernels needing only `elems % 128 == 0` (true for
hidden=3072), giving 48.4% wire reduction. **[code-verified + claim]**
**But it buys us essentially nothing and the earlier ranking was wrong.**
Arithmetic: our decode payload at M=1 is 3072 x 2 B = 6 KiB; over the Gen3 x4
link (~3.9 GB/s) that is ~1.5 us, so halving it saves 0.77 us against an
11.88 ms ITL — **0.006%**. At M=16 it saves ~12 us, ~0.1%. The codec only pays
on prefill-sized payloads, and prefill does not use the doorbell (it uses
Marlin streaming). Revisit only if a future design pushes large activations
across that link.

### 8. KV capacity — SGLang's mamba-capacity pool solve
Our 1.60 seats at 131K is a hard wall versus the PRO's ~6. SGLang carries an
explicit capacity solve for hybrid mamba/attention pools worth reading against
our `mamba_block_size`-coupled 4176-token attention page. **[code-verified]**
Also from the same corpus: **retokenized-ITL** measurement, which is the
spec-decode-fair way to report latency (relevant when/if we get a draft head).

### 9. Config-level sm_120 knobs from the community wiki
`vendor/rtx6kpro/optimization/nvfp4-quantization.md`:
`VLLM_NVFP4_GEMM_BACKEND=cutlass` selector, the `sm120f` family-conditional PTX
requirement for NVFP4 conversion, a hybrid "NVFP4 + BF16 shared-expert + layer-0"
VRAM/quality technique, and a flag that **FP8 KV cache is broken on SM120 for
GLM-5** (we run fp8 KV — worth a numerics spot-check).
`hardware/gpu-configs.md` has a **GB202 power sweep: 500 W ≈ 600 W parity; 300 W
MaxQ costs ~4% single-user, ~30% at 64-concurrent** — our exact die.
**[measured-elsewhere, same die]**

---

# PARK — real, but expensive or gated

### QuTLASS rotations — unblocked by the bf16 base, still weeks
**Definitive answer on the cheap version: NO.** Activation-only online rotation
is impossible. The same Hadamard must rotate **both** operands (RxRᵀ=I holds in
one basis only), proven by their own tests quantizing the weight through the
identical `fusedQuantizeNv(b, h, global_scale)` call with the same `h`
(`tests/nvfp4_test.py:216-224`, `benchmarks/bench_nvfp4_sm120.py:44-51`).
**[code-verified]**
So rotation requires an **offline rotated + requantized NVFP4 checkpoint** —
which our bf16 base now makes *possible* (verification #1) rather than blocked.
Remaining cost: their `bindings.cpp` is hard-gated to 2-D dense
(`A.dim()==2 && B.dim()==2`), no grouped/MoE path exists, so **we would write the
grouped-GEMM host wrapper ourselves**; and `third_party/cutlass` is an empty
submodule, making the build network-dependent. Build itself is plausible
(torch ≥ 2.11 admits our 2.13; `sm_120a` gencode and a `TARGET_CUDA_ARCH=120`
branch both exist and are exercised). All in-repo evidence is
kernel-vs-reference **correctness**; the Qwen3-8B/14B prefill-speedup plots are
MXFP4-vs-BF16 on **dense** models via an un-vendored HF harness. **[claim]**
*Verdict:* the only credible route to W4A4-class speed **with** quality. Weeks.
Revisit when the short-context gap is the last thing standing.

### Hand-built sm_120 grouped int4×bf16 GEMM — gate on a roofline first
**sm_120 tensor cores are F8F6F4-only.** `cute/arch/mma_sm120.hpp` never admits
`int4b_t`, `uint4b_t` or `bfloat16_t` as an A/B operand, and this is
static-asserted in **both** the dense (`sm120_mma_builder.inl:76-79`) and
**grouped** (`sm120_array_mma_builder.inl:78-79`) builders. The nearest CuTe-DSL
grouped mixed-input example is SM100/tcgen05/TMEM-only. **[code-verified]**
Reusable pieces if we ever do it: the grouped tile scheduler resolves to the
**arch-generic** `PersistentTileSchedulerSm90Group` and is proven compiling on
sm_120 today (`examples/87_blackwell_geforce_gemm_blockwise/87c`); a portable
bf16 atom exists at raw `cute::arch` level (`SM80_16x8x16_F32BF16BF16F32_TN`)
with **no** CollectiveBuilder wiring; the int4-unpack LOP3+PRMT math is
arch-generic.
*Effort:* ~5-8 weeks. **And the decisive point: it would target `mma.sync` — the
same instruction class Marlin already uses.** So it buys better scheduling, not
new silicon capability.
*Gate:* **do not attempt unless a roofline analysis first shows Marlin is >20%
off the sm_120 bf16 tensor-core ceiling at our shape.** That analysis is cheap,
analytical, and we have not done it. It is now the honest next step on
short-context prefill.

### Speculative decoding — a training project
`vendor/speculators` (not `SpecForge`, which targets SGLang) is the correct
vLLM-native vehicle: it already registers `qwen3_5_moe_text` model classes for
both EAGLE3 hidden-state capture and MTP stitching
(`base_components.py:86-96`, `mtp/model_definitions.py:57-70,191-199`) and ships
a documented `--speculative-config` entry point. **[code-verified]**
Two hard constraints, both verified:
* **MTP is out** — no `mtp.*` keys in our checkpoint *or* base (verification #2).
* **The zero-training n-gram/prompt-lookup probe is out, and it is worse than
  unavailable — it is dangerous.** vllm#39273 is an open, still-unfixed
  silent-corruption bug on **exactly** `qwen3_5_text` hybrid GDN+full-attention,
  root-caused to missing SSM-state rollback on token rejection in
  `v1/worker/mamba_utils.py`; last confirmed broken 2026-07-20 with "draft
  acceptance looked healthy… while silently corrupting output." **[claim, but a
  named upstream issue on our exact model family — treat as blocking.]**
  This kills the "cheap risk-free probe" I previously proposed.
*So:* speculation means training an EAGLE3 head from the bf16 base. Nearest
ready-made configs target Qwen3.5-**A3B** (35B), not our 88B. Encouraging
same-family evidence: MTP on Qwen3.5-397B-A17B NVFP4 measured **89.2%
acceptance, MTP=2 sweet spot, +51-72% throughput** (with MTP>3 crashing)
**[measured-elsewhere, same family, larger model]** — so the ceiling is real if
we pay the training cost.

### Persistent B70 decode kernel
Template exists: `PersistentTileSchedulerMoE`
(`vendor/intel-xpu/.../grouped_gemm/collective/gemm/moe_tile_scheduler.hpp:44-300`)
— CUTLASS-pattern persistent CTA, fixed grid, ragged per-expert M, swizzled
raster. **[code-verified]** Also `dnnl_matmul_w4a16_int4`
(`csrc/xpu/onednn/int4_gemm_w4a16.h:14-156`) is a complete oneDNN fused
weight-decompress int4 recipe with group-wise scales and symmetric/asymmetric
zero points.
*Why parked:* the dual-B70 team independently put "megakernel / persistent decode
kernel" **last** on their own roadmap, after cheaper GEMV and graph levers
(`20260703_faster_dd_plan.md:96-141`) — a hardware-matched signal to do items 1
and 2 first. Also gated by the SLM-in-graph landmine below.

### Second B70 / cross-card work
B70↔B70 P2P **is** available but conditionally: kernel ≥ 7.0 **and IOMMU
disabled** (`docs/P2P_GPU.md:417-420`) — our box currently runs with AMD-Vi
active across 35 groups **[measured-here, earlier this campaign]**, so P2P is
likely *off* for us today. Bandwidth is asymmetric: **push ≈11.08 GB/s vs pull
≈3.24 GB/s** (3.4× penalty — always push, never pull), and **`ATOMICS=N`**, so
no device-side cross-card barrier is possible; ordering must be host-driven or
IPC-event-based. **[measured-elsewhere, same card class]**

### Intel Graphics Compiler
Park. It is already running invisibly under every SYCL/L0 call. Its only value
is ISA-level shader dumps to confirm the compiler emitted the DPAS/GEMV path we
expect — reachable via `IGC_ShaderDumpEnable=1` on the stock driver, without the
multi-hour LLVM-scale build.

---

# REJECT — dead ends, recorded so we stop looking

| Target | Why it's dead for us |
|---|---|
| **TileRT** | Closed binaries (`libtilert_dsv32.so` not even in-tree), Dockerfile hard-pins `CUDAARCHS=100` (sm_100 B200), torch 2.11, DeepSeek-256-expert 128×128-block FP8 format. Zero kernel source. **[code-verified]** |
| **AdaptiveCpp** | A redundant *second* SYCL toolchain. Every real B70 kernel found in this recon builds with oneAPI `icpx`, which our `.venv-xpu` already matches. Pure duplication risk. |
| **b12x `comm.pcie` collectives** | Entirely CUDA-IPC + NCCL; every runtime gates on `device_count() >= 2` CUDA GPUs. Structurally dead on 1×CUDA + 2×non-CUDA. (The *codec* inside it is liftable — see BENCH FIRST #7.) |
| **b12x MLA / sparse-MLA / NSA-indexer / mHC attention** | We are GDN-hybrid, not MLA; mHC is hardcoded to DeepSeek hidden sizes 4096/7168 vs our 3072. |
| **b12x `trellis_linear`, `wo_projection`** | Format mismatch — block-scaled MXFP8 vs our per-tensor-scalar FP8 tensors. (`wo_projection`'s §8 requantization postmortem is still worth reading as QA.) |
| **uccl** | CUDA/RDMA-only. |
| **SGLang distributed EP** (DeepEP/Mooncake/Mori/Nixl, two-batch overlap) | Hard-requires NVLink/IB fabric. We are PCIe-only. Also: `vendor/sglang` and `vendor/sglang-upstream` are **byte-identical** — no fork delta to mine. |
| **SGLang MoE kernels as a Marlin challenger** | Their sm_120 MoE story is the *same* Marlin/FlashInfer-CUTLASS families we already benched to a negative verdict. |
| **flashinfer 0.6.18 upgrade** | Already tried and rolled back this campaign: its `input_global_scale` fix does not touch W4A4 activation quant, so it does not reopen the cosine-0.82 verdict, and it carries sampler/FP8 regression risk. Its `_moe_dynamic/` dispatch path is new but MoE-negative. |
| **PRO 6000 collective evidence JSONs** | Every real number is TP8/DCP4 8-GPU *collective scaling*; only name/PCI/UUID are per-device. Does **not** transfer. Honest correction to my earlier hope. |
| **`vendor/rtx6kpro` as a source of our competitor's run flags** | The 88B comparison directory is empty in this checkout and sibling results are unfetched LFS pointers. Useless for *that*. **But see the corrected hardware verdict below — the first pass wrongly dismissed the whole repo by conflating provenance ("not our model, not our team") with applicability.** |

### Corrected: `vendor/rtx6kpro/hardware/**` is largely inapplicable but not empty

~90% is multi-GPU NVIDIA tensor-parallel topology — c-payne/Broadcom switch
fabrics, dual-CPU root complexes, xGMI, NCCL ring tuning, GPU-to-GPU P2P
bandwidth, DCP, 8-16 GPU rigs. We have **one** CUDA GPU and one root complex,
so none of it applies, including the otherwise-excellent
`hardware/collapse-report.md` (its trigger needs a PCIe switch dispatching
posted writes to GPUs behind >=2 root complexes; note in passing that **reads
are unaffected because non-posted completions carry flow control** — our H2D
copy-engine path is on the safe side of that distinction).

Four items **do** apply, all checked against this box:

1. **BAR1 / Resizable BAR.** Their named footgun: some BIOSes default BAR1 to
   256 MB and "cripple" transfer performance (`hardware/pcie-bandwidth.md`).
   **Checked: ours reports Total 32768 MiB = full VRAM.** Correctly configured,
   no action — but now verified instead of assumed. **[measured-here]**
2. **`pcie_aspm=off pcie_port_pm=off`.** Without `pcie_port_pm=off`, GPU dynamic
   power management can suspend the root port during a Gen1<->Gen5 link retrain,
   producing "Surprise Link Down" (`aer_uncor_status: 0x00000020`) and **system
   lockups**. **Checked: we set no PCIe kernel params (kernel defaults), and AER
   counters on `0000:01:00.0` are 0** — so this is *latent hardening*, not a live
   bug. Cheap insurance for a box that DMAs 27.4 GiB across that link every
   forward pass. **[measured-here]**
3. **GB202 power sweep** (`hardware/gpu-configs.md`): 500 W ~= 600 W parity;
   300 W MaxQ costs ~4% single-user and ~30% at 64-concurrent. Same die as our
   5090. **[measured-elsewhere, same die]**
4. **IOMMU mode is a genuine tension.** Their multi-GPU NVFP4 recipes want
   `iommu=pt`; the dual-B70 journal needs IOMMU **disabled** for B70<->B70 P2P.
   We run defaults with AMD-Vi active and are already at the DMA ceiling, so
   there is nothing to gain today — but a future dual-B70 P2P design makes this
   a boot-parameter decision with a real tradeoff.

**One reframing datapoint worth more than the four:** the wiki notes 4x 5090s
lose to a single PRO 6000 for TP because the PRO has ~1.5 TB/s on-die bandwidth
versus ~50 GB/s of PCIe between 5090s. The corollary for a *single*-card
comparison is the important part: **our 5090 and the PRO 6000 are the same
GB202-class GDDR7 memory system — per-card bandwidth is essentially equal.**
Their advantage over us is **capacity (96 vs 32 GB) and SM count**, not memory
speed. So we are not fighting a faster machine; we are fighting one that never
has to move weights. That is exactly why our long-context wins are real and our
short-context losses are structural.

### External research triaged: MMA, arXiv 2512.16056 (multipath host-GPU copies)

**Mechanism: REJECT.** MMA relays a host->GPU copy through *peer CUDA GPUs*
(peer reads host DRAM over its own PCIe link, forwards to the target over
NVLink), reaching 245 GB/s vs a 53 GB/s single-link baseline on 8x H20. Three
independent blockers for us: (a) we have exactly **one** CUDA GPU — the B70s
cannot be CUDA relays; (b) **no NVLink** on consumer Blackwell, and the NVLink
hop is what makes the relay pay; (c) even hypothetically, a B70 relay would
read host DRAM over Gen4 x4 (~7 GB/s) and then forward with no cross-vendor
P2P, i.e. B70->host->5090, **consuming the 5090's own Gen5 link anyway** plus
extra DRAM traffic — strictly negative. Their own scope table measures **0.94x
(a small loss) at TP=8** where no spare peer exists; that is our permanent
regime.

**Four residues worth keeping:**

1. **Third-party confirmation that we are at the PCIe ceiling.** Their Table 1:
   Gen5 x16 = 64 GB/s theoretical, **52-60 GB/s typical measured**; their native
   baseline is 53 GB/s. Ours is 53.9 GiB/s = **57.9 GB/s = 90% of theoretical**,
   top of their measured band. **[measured-here vs measured-elsewhere]**
   Consequence: no headroom remains on the link. Further prefill-transfer gains
   must come from moving **fewer bytes**, never from moving them faster. This
   retires the "is there PCIe headroom?" question for good.
2. **Our doorbell sync is already better than theirs — do not "improve" it.**
   Their §3.3 enumerates and rejects `cudaDeviceSynchronize`,
   `cudaLaunchHostFunc` (stream->CPU only), and CPU polling, then builds a
   **spin kernel** polling a `cudaHostAllocMapped` flag with `__ldcg` +
   `__nanosleep(100)`: one resident thread block, 1-2 us, requires the CUDA
   context stay scheduled. We use `cuStreamWriteValue32_v2` /
   `cuStreamWaitValue32_v2` (`stream_signal.py`) — **hardware stream memory
   operations: zero SM footprint, no kernel, natively graph-capturable, no
   tuning surface, and already fully hidden** (B70 44 us vs CUDA MoE 100 us).
   The paper never mentions stream memory ops. If we ever need the *stream->CPU*
   direction (we do not — the B70 thread polls a host flag), `cudaLaunchHostFunc`
   is the documented primitive.
3. **Outstanding-queue depth 2 is optimal** (depth 1 leaves idle gaps between
   transfers, >2 coarsens scheduling) — independent confirmation of our 2-arena
   double buffer, which we chose because it fit VRAM. Pairs with b12x's
   `recommend_prefetch_depth`. **Do not** import their chunk-size numbers
   (2.81 MB H2D): those size multipath load-balancing granularity, not
   single-path efficiency. On one link our single 584.7 MiB contiguous copy per
   layer is the right shape.
4. **GPUDirect Storage would be a ~7x downgrade for us.** Their §7 notes GDS
   targets NVMe->GPU and "NVMe throughput (~7 GB/s per drive) is an order of
   magnitude below DRAM bandwidth". Our bank is deliberately page-cache resident
   at 53.9 GiB/s. This also explains the pin-eviction thrash: losing page cache
   drops us onto exactly that NVMe path. **Never propose GDS for the bank.**

**The one applicable pointer, and why it stays parked.** Their related-work
contrast with **ServerlessLLM** names the technique that does fit our box:
*partitioned parallel load* over independent links (each device pulls its own
shard through its own PCIe link) rather than relay. Our three links are
independent and both B70 links are **idle during prefill**. Priced: aggregate
weight-read bandwidth 57.9 -> ~68 GB/s (**+18%**). But stream is already fully
hidden under compute at >=8K, so +18% on transfer buys **~0** there; and at
short context, where stream *is* exposed, the B70's prefill compute is exactly
what we abandoned (its per-route kernel left the 5090 97% idle). So this remains
the parked "5-10% B70 expert cut" — now with arithmetic instead of intuition.

---

# Meta-findings — the patterns worth more than any single asset

### 1. sm_120 falls through arch gates — but the gates are mostly protecting us

**The audit that corrects an earlier overclaim.** I first recorded "sm_120 falls
through arch gates in four independent places" and called it a cheap-to-fix
class. Then I audited every backend selector our running server actually logs
(`run6_final/server_decode.log`) against what is installed. The gates are real,
but in three of four cases **the kernel behind the gate quantizes activations to
FP4 — the W4A4 regime we measured at cosine 0.82 per layer and killed in round
1.** Fixing those gates would buy speed we cannot use.

| instance | gate real? | would fixing it help? |
|---|---|---|
| GDN prefill (`family(100)`, `qwen_gdn_linear_attn.py:116-133`) | yes | **YES — linear attention, not a quantized GEMM; numerics preserved** |
| `FlashInferB12xExperts` MoE (upstream SM121 guard) | yes | no — W4A4, measured 0.82 |
| NVFP4 dense `FlashInferCuteDsl` (`family(100)`) | yes | no — and its sibling `FlashInferCutlass` *does* admit sm_120 via a `>=` check, but `input_quant_key() -> kNvfp4Dynamic` = W4A4 |
| CUTLASS grouped/dense F8F6F4-only asserts | not a gate — an ISA fact | n/a |

**What the server actually selects, and why each is right:**

| selector | chosen | verdict |
|---|---|---|
| NVFP4 dense GEMM | `MarlinNvFp4LinearKernel` | correct — docstring is "weight-only GEMM (W4A16)"; of the six NVFP4 linear kernels **only `flashinfer.py` overrides `input_quant_key`**, so every faster option is W4A4 |
| ModelOpt FP8 linear | `FlashInferFP8ScaledMMLinearKernel` | correct for our per-tensor-scalar FP8 `q/k/v/o_proj` + `linear_attn.*`. The vLLM #47749 "silent Marlin fallback" signature does **not** apply: our checkpoint is genuinely mixed, so two kernels for two formats is right |
| Attention | `FLASHINFER`, `decode_backend=flashinfer-native`, `arch=sm120`, autotune cache under **`120f`** | correct, and already on the family-forward arch suffix |
| MoE | `MARLIN` of 8 candidates | litigated across two bake-offs |
| top-k/top-p sampler | logs "Using FlashInfer" | **phantom — never invoked.** `forward_cuda` opens `if (k is None and p is None) or generators: return self.forward_native(...)`; our harness sends no top-k/top-p, so we always take the native path. The log line announces *availability*, not use — do not chase FlashInfer issue #3389 on our numbers |
| GDN prefill | **`Triton/FLA` fallback** | ⚠️ the only genuine gap |

**Action, revised:** stop treating arch gates as a general opportunity class.
The specific live item is GDN prefill; everything else our server selects is
either correct or correct-for-quality-reasons we measured ourselves.

### 1b. GDN **decode** has no backend selector at all — a different problem
**[code-verified]** vLLM's GDN decode path calls vendored FLA Triton kernels
directly (`fused_sigmoid_gating_delta_rule_update`,
`fused_recurrent_gated_delta_rule_packed_decode`, imported from
`vllm.third_party.flash_linear_attention.ops`). There is **no selector to be
gated out of** — unlike prefill, no integration exists.

Meanwhile our **installed** flashinfer 0.6.16.post3 ships `flashinfer/gdn_kernels/`
with `gdn_decode_bf16_state`, `_bf16_wy_output_only`, `_mtp`, `_nontranspose`,
`_pretranspose`, plus `blackwell/` and `delta_rule_dsl/`. Its documented contract
lines up with what vLLM already passes:

* `gated_delta_rule()` — **T=1 single-token decode**, i.e. exactly our step shape.
* State **pool mode** `[pool_size, HV, V, K]` indexed by `initial_state_indices`
  — the same shape as vLLM's `ssm_state_indices` / `non_spec_state_indices_tensor`.
* **Split-pool writes** (`output_state_indices != initial_state_indices`) — matches
  vLLM's separate read/write index tensors.
* `gated_delta_rule_mtp()` for T>=1 — already the right shape for spec-decode
  verify, if a draft head ever lands.
* An ILP=4 higher-occupancy variant exists.

The one flashinfer linear-attention backend that *does* have a proper selector
(`mamba/ops/ssu_dispatch.py` -> `flashinfer.mamba.selective_state_update`) is the
**Mamba-2 SSM** path, not GDN; our boot log never selects it.

*Cost:* this is an integration we would write ourselves against a documented API
through our existing `PluggableLayer` hook — days, not hours, and not a research
project since the contract shapes match. Unverified: whether these kernels accept
`head_k_dim=128` and our exact state dtype/layout.
*Why it matters:* decode is our worst metric (1.35-1.63x), and unlike the prefill
gap this one is not reachable by flipping a gate.

### 2. Small-M kernels leave 30-50% of memory bandwidth on the floor
On the B70: int8 GEMV at M=1 reaches 51-59% of 608 GB/s where bf16 reaches
71-76%. Native `s4×s4` DPAS **exists and is bit-exact** on this card, yet a
naive tiled mainloop caps at **~64 TOPS versus int8's 367 TOPS**.
**[measured-elsewhere, same card class]**
The lesson generalizes past Intel: *having the instruction is not having the
performance*. It is also why our own decode number may be a utilization problem
rather than a hardware verdict — and why BENCH FIRST #1 is ranked first.

### 3. On the B70, decode is dominated by launch overhead, not compute
A comparable model measured **~950 kernel launches per token: 32.2 ms of CPU
enqueue against 9.45 ms of actual GPU work**, and graph capture took it from
21.8 → 93.0 tok/s (**4.0-4.3×**)
(`cookbook/docs/nemotron35-30a3/NEMOTRON-B70.md:23-31`).
**[measured-elsewhere, same card class, different model]**
This quantifies why the doorbell/persistent-kernel direction is aimed at a real
problem class — and warns that our own gains there may come from *submission*
economics rather than kernel math.

### 4. XPU graph replay works — with three landmines that will bite by construction
All **[measured-elsewhere on this card class, live crashes]**:
* **NEO command-list overflow.** `XPUGraphImpl::replay` submits via
  `submit_with_event` and **never syncs**; a tight per-step replay loop with no
  host sync overflows `linear_stream.h:84` after ~1-2k tokens (many-op graphs) to
  ~96k tokens. A doorbell firing every decode step is exactly this shape — we
  must sync every N replays or confirm our path uses a reclaiming primitive.
* **SLM kills capture.** Any kernel using `work_group_scratch_memory` cannot be
  captured (`sycl_ext_oneapi_work_group_scratch_memory … not yet available with
  the SYCL Graph extension`), gated to oneAPI 2026.0. If our int4 GEMV uses SLM
  for its reduction, it cannot enter a captured graph at all.
* **One graph cannot span two devices** — `command_graph::begin_recording`
  rejects a second device. Cross-B70 work means two graphs plus an external
  L0-IPC event or SYCL `external_semaphore`.
Plus a correctness trap: a persistent-scheduler work counter must be
`at::zeros`, not `at::empty` — SYCL does not guarantee group 0 runs first, and a
dirty leftover becomes the starting tile index, *especially* under graph replay.

### 5. sm_120's gaps are ecosystem, not silicon
No tcgen05/TMEM/WGMMA/FA3/DeepGEMM/FlashMLA-sparse support paths
(`vendor/rtx6kpro/inference-engines/flashinfer.md`,
`hardware/sm120-vs-sm100-architecture.md`). The wiki's framing — a
*software-ecosystem* gap for local inference, not a hardware deficiency — matches
everything we measured independently: every datacenter-tuned kernel path skips
our card, and the fallbacks are what we actually run.

---

# What this changes about the roadmap

**Prefill, short context (1.8-1.9× behind).** The next step is **not** a kernel
port. It is a **roofline analysis of Marlin against the sm_120 bf16 tensor-core
ceiling** — cheap, analytical, and it decides whether 5-8 weeks of hand-built
grouped GEMM could ever pay, given that such a kernel targets the same
`mma.sync` class Marlin already uses. Alongside it, two *unattacked* non-MoE
targets now have named candidates: `tensor_fp8_linear` for the 0.149 s
`unattributed_gemm` and the flashinfer sm_120 GDN kernel for the 0.070 s
`gdn_attention`.

**Decode (1.35-1.63× behind, the open front).** Sequence is now evidence-led:
(1) audit our B70 int4 GEMV's achieved bandwidth against 608 GB/s — if we are at
~55%, that is the gap; (2) the 150 W power cap, free; (3) graph/submission
economics with the three landmines designed around from the start;
(4) predictive placement (PILOT) and adaptive split (`MoeCpuHost`);
(5) persistent kernel last, as the hardware-matched team independently
concluded. Speculation remains the only sub-1× lever and is now correctly priced
as **draft-head training from the bf16 base**, with n-gram ruled out as actively
unsafe on our model family.

**Protected assets.** The 164 GB bf16 base is load-bearing for three levers.
Do not delete it.
