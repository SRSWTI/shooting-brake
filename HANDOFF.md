# Shooting Brake — handoff

Written 2026-08-18. This is the single document to read before continuing. It
covers what the system is, what every load-bearing file does, exactly where the
88B stands (with numbers re-measured on the current build today), what the
vendor trees give us and at what price, and what has to be true before the 122B
boots.

Provenance discipline is inherited from `docs/vendor-extraction.md` and used
throughout:

* **[measured-here]** — measured on this box; artifact under `benchmarks/results/`.
* **[measured-elsewhere]** — measured by someone else on named hardware.
* **[code-verified]** — read in source at `file:line`. A contract, not a number.
* **[claim]** — README/paper assertion, unreproduced. A hypothesis.
* **[INFERENCE]** — our arithmetic on top of measured inputs.

**The sm_120 transfer rule.** The RTX PRO 6000 Blackwell Workstation is the same
sm_120 family as our RTX 5090 (GB202 die). Kernel selection, arch guards and
per-device measurements from a PRO 6000 **transfer to us**. Numbers from sm_100
(B200) **do not** — different instruction set (tcgen05/TMEM), different builder
specializations. Multi-GPU collective-scaling numbers from any card never
transfer to a single-CUDA-GPU box.

---

# 1. What Shooting Brake is

**The thesis:** serve a large MoE model at long context on one 32 GB consumer
GPU by keeping the *dense* work local and pushing the *routed experts* onto
cheap non-NVIDIA accelerators over PCIe — and beat a single 96 GB RTX PRO 6000
on the cells where capacity, not compute, decides.

That works because of an asymmetry the vendor recon nailed precisely: our 5090
and the PRO 6000 are the **same GB202-class GDDR7 memory system — per-card
bandwidth is essentially equal**. Their advantage is capacity (96 vs 32 GB) and
SM count, not memory speed. *We are not fighting a faster machine; we are
fighting one that never has to move weights.* That single sentence explains both
our long-context wins and our short-context losses.

## The architecture: hub and spoke, never tensor parallelism

```
RTX 5090  0000:01:00.0  Gen5 x16   attention, GDN, router, KV, local experts
    |
    +-- Arc Pro B70  0000:15:00.0  Gen4 x4   remote experts   (6.24 GB/s H2D)
    |
    `-- Arc Pro B70  0000:11:00.0  Gen3 x4   remote experts   (2.91 GB/s H2D)
             both B70s share upstream bridge 09:00.0, Gen4 x4
```

No B70↔B70 traffic. No oneCCL, no all-reduce, no cross-vendor P2P. Each remote
card computes an independent routed-expert partial; the 5090 sums them with its
own local partial. This is why the vendor dual-B70 journal's TP=2 wedge
catalogue (GP-faults, BCS engine resets, `zeMemOpenIpcHandle` failures) is
**structurally inapplicable** to us — it needs cross-card edges we don't have.

## Two phases, two completely different data paths

This is the single most important thing to understand about the codebase.

**Decode — the "doorbell".** Weights are already resident on the B70. Per layer,
the 5090 writes a host-mapped signal flag with `cuStreamWriteValue32_v2`, a
native poller thread on the host sees it, dispatches the SYCL MoE kernel on the
B70, and writes a completion flag the 5090 waits on with
`cuStreamWaitValue32_v2`. Zero SM footprint, no kernel, natively CUDA-graph
capturable. 48 dispatches per decode step. Only expert IDs and a 6 KiB
activation cross the wire — bandwidth is irrelevant here, **latency is
everything**.

We independently confirmed this design beats the published alternative: arXiv
2512.16056 §3.3 enumerates and rejects `cudaDeviceSynchronize`,
`cudaLaunchHostFunc` and CPU polling, then builds a spin kernel polling a
`cudaHostAllocMapped` flag with `__ldcg` + `__nanosleep(100)` — one resident
thread block, 1–2 µs, requires the CUDA context stay scheduled. The paper never
mentions stream memory ops. **Do not "improve" our doorbell toward theirs.**

**Prefill — Marlin streaming.** The doorbell is catastrophically wrong for large
M: our per-route kernel does `O(M × top_k)` weight reads where a grouped GEMM
does `O(E)`. At M=2048 that measured **24.6 GiB of logical weight reads vs
861 MiB — 28×**. So prefill abandons the B70 entirely: the remote experts stream
from an offline pre-repacked Marlin bank in host page cache to the 5090 over
Gen5 x16 and run through vLLM's `fused_marlin_moe`. Same int4 weights in both
phases, so the served model is numerically unchanged.

The crossover is `SHOOTING_BRAKE_B70_MAX_BATCH` (256 today): forwards with
M ≤ that take the doorbell, larger take the streamer.

**`TTFT = max(stream, compute)` is measured fact, not a model** — whole-bank
streaming concurrent with 48 layers of M=32K Marlin compute degrades only 2.4%
vs the ideal max(). The copy engine and the SMs genuinely coexist.

---

# 2. The machine

| component | detail |
|---|---|
| CPU | AMD Ryzen 9 9950X3D, 16 core |
| RAM | 59 GiB usable (63 GB nominal), swap 7 GiB — **the binding constraint for the 122B** |
| GPU 0 | RTX 5090, GB202, sm_120, 32,607 MiB |
| GPU 1/2 | 2× Intel Arc Pro B70 (Battlemage G31), **32,656 MiB each** [measured-here, `xpu-smi`] |
| NVMe | Samsung 990 PRO 2 TB, Gen4 x4, ~92% full, 141 GB free |
| stack | vLLM 0.27.1, torch 2.13.0+cu130, CUDA 13.0, oneAPI 2026.1, flashinfer 0.6.16.post3 |

Measured link facts [measured-here]:

* 5090 H2D registered page-cache DMA **53.9 GiB/s = 57.9 GB/s = 90% of Gen5 x16
  theoretical**. Third-party Table 1 puts typical measured Gen5 x16 at
  52–60 GB/s, so **we are at the top of the band and no headroom remains.**
  Further prefill-transfer gains must come from moving *fewer bytes*, never
  faster ones. This retires the "is there PCIe headroom?" question permanently.
* Pageable mmap→GPU 18.54 GiB/s. Registration is worth 2.9×.
* B70 device read ceiling **599.2 GB/s = 98.6% of the 608 spec**, incompressible.
* NVMe O_DIRECT 6.05 GB/s single stream, **6.28 GB/s with 2 threads** reading
  into a `cudaHostRegister`ed region; 4+ readers *regress*.
* BAR1 reports full 32,768 MiB — correctly configured, verified not assumed.
* AER counters on `0000:01:00.0` are 0. We set no PCIe kernel params.
  `pcie_aspm=off pcie_port_pm=off` is latent hardening we have not applied.

**Two instrument corrections that gate every future B70 number:**

1. **Xe2 memory compression inflates constant-fill fixtures** — the old harness
   measured up to **111% of spec**, physically impossible. Random-fill only.
   Every constant-fill bandwidth figure from before this fix is inflated ~25% at
   M≥8.
2. **DVFS swings cold runs 2.6×** (47–124 µs at M=1). Microbenches MUST pin
   `min_freq == max_freq` in `/sys/class/drm/cardN/device/tile0/gt0/freq0/` or
   run long sustained windows. The pin *holds* — verify via `act_freq` (the
   PCODE-resolved value), never `cur_freq` (GuC's request, hides SLPC
   overrides). The vendored zml claim "xe has no clock control" is **wrong** on
   this kernel. Use `scripts/b70_tune.sh`.

   But: a production A/B showed DVFS **does not move serving ITL** (11.89 ms
   baseline vs 12.10 pinned). Decode self-warms the card — 48 dispatches per
   ~12 ms step is one doorbell every 250 µs, so the GPU never idles down.
   **DVFS is a microbenchmark hazard, not a serving lever.**

---

# 3. Repo map — what each file actually does

## `src/phase4/src/shooting_brake_vllm/` — the vLLM out-of-tree plugin

21 modules. vLLM's `RoutedExperts` and `MoERunner` are `PluggableLayer`s; we
replace them out-of-tree. No vLLM patch anywhere.

| file | job |
|---|---|
| `__init__.py` | OOT `register()`; gated on `SHOOTING_BRAKE_PHASE4=all-cuda` + a qualified `SHOOTING_BRAKE_MODEL`. Monkeypatches `VllmConfig.__post_init__` to force `CUDAGraphMode.PIECEWISE` under `VLLM_USE_BREAKABLE_CUDAGRAPH=1`. |
| `routed_experts.py` | **3,019 lines, the core.** `HybridRoutedExperts`: builds the placement, allocates pinned staging + host-mapped flags per layer, owns the decode dispatch. `_hybrid_forward_modular` (2160-2374) has a CUDA-graph branch (2198-2227, pure stream ops, zero Python branching) and four eager sub-paths. `_b70_issue_graph`/`_b70_take_graph` (2611-2685) are the graph-safe signal/wait pair. `forward_modular:2128` is the prefill/decode crossover. |
| `placement.py` | 994 lines. Device-agnostic ownership manifest. `DeviceTarget` already carries a B70 `index`, and `FractionalRemotePolicy` (444-489) **already balances across N remote devices**. Grammar parsed in `policy_from_name` (892-961): `all-cuda`, `split:<N>`, `interleaved:<N>`, `subset:<K>:<N>`, `allout:<K>:<N>:<C>`, `fractional:<N>:<F>`. |
| `config.py` | Model admission. `_MODEL_SPECS` (42-71) holds 3 entries: 35B NVFP4, 122B NVFP4, 88B GPTQ-int4. `require_qualified_config` (462-577) checks architecture/geometry against the spec, then validates the bank header against the model. TP/PP must be 1, EPLB off. |
| `partition.py` | Per-step route classification. `validate_cuda_dummy_slot_placement` (92-110) enforces the invariant that a layer with remote experts must keep ≥1 real CUDA expert — the fused CUDA partial masks remote routes through a real local dummy slot. `build_device_map` (189-202) collapses all B70 indices to code 1. |
| `b70_binding.py` | ctypes client over `libsb_b70_provider.so`. |
| `b70_poller.py` | Python side of the native polling thread; `register_layer` binds flags + the four staging buffers. |
| `stream_signal.py` | `cudaHostAlloc` + `cudaHostGetDevicePointer` flags; `cuStreamWriteValue32_v2`/`cuStreamWaitValue32_v2`. Device-agnostic. |
| `marlin_prefill.py` | The prefill streamer. Opens the SBMARL01 bank, `n_slots` device arenas, `partial()` does M-tiling. Registered as a custom op via `direct_register_custom_op`. |
| `b12x_prefill.py` | Bank-v2 (SBB12X01, native FP4) alternative to Marlin. Not on the serving path. |
| `bank_source.py` | 78 lines. Pageable `np.memmap` or `cudaHostRegister`'d mmap. **Hard-fails on a failed register — never silently falls back.** |
| `int4_bank_format.py` / `marlin_bank_format.py` / `b12x_bank_format.py` / `expert_bank.py` | The four on-disk bank ABIs and their readers. |
| `cpu_expert_host.py` / `cpu_stream.py` | Third residency tier (host DRAM, CPU compute). Dormant. Structurally the closest existing analogue to "add a second remote backend" — its lifecycle pattern is the template for card 2. |
| `route_stats.py` / `telemetry.py` | Observability. Layer-keyed, not device-keyed. |
| `runner.py` | `HybridMoERunner`, thin wrapper preserving stock CUDA `MoERunner`. |
| `provider.py` | Near-dead Phase-4 stub. Real dispatch bypasses it. |

## `src/phase1/` — the C++ provider and the bank builders

| file | job |
|---|---|
| `b70_provider.cpp` (1,508 lines) | The real provider. `select_b70()` (233-315) resolves a device by decimal index **or lowercased PCI BDF** — BDF selection already works here. `load()` mmaps and parses the bank (magic-dispatched SBEXP001 vs SBINT401), allocates USM, uploads weights. `issue()`/`take()` are the dispatch pair. `register_host_range` imports CUDA-pinned staging into the SYCL context. |
| `b70_provider.hpp` | `ProviderConfig` (has `device_selector`), `Capability` (has `device_pci_bdf`, `device_index`, `source_expert_ids`, `max_batch_remote`, `supported_hidden_sizes`). |
| `extract_experts_int4.py` | AutoGPTQ int4 → SBINT401. **Model-agnostic** — `discover_shape` reads config.json and accepts auto_round or plain `quant_method=gptq && !desc_act`. Also carries `dequantize_nvfp4`/`cross_validate_nvfp4` (814-975) which already decode compressed-tensors correctly. |
| `extract_experts.py` | A **second, generic** NVFP4/compressed-tensors builder writing SBEXP001. Implements `weight_packed`/`weight_scale`/`weight_global_scale` with the divisor convention. Never run against the 122B. |
| `build_marlin_bank.py` | SBINT401 → SBMARL01 offline repack via `torch.ops._C.gptq_marlin_repack` + `marlin_moe_permute_scales`. Bit-exact self-validating. **Inherits `source_expert_ids` from its input**, so it is fully expert-subset aware. |
| `build_b12x_bank.py` | NVFP4 → SBB12X01. **Hardcoded to 88B geometry** (`EXPERT_ID_BASE=54`, `REMOTE_EXPERTS=126`). |

## `src/phase7/` — the C ABI

`b70_capi.{h,cpp}` is a thin `extern "C"` shim compiled into the same
translation-unit set as `b70_provider.cpp` (no separate .so). Exports 53
symbols. It also hosts `B70Poller` — the native background thread — and the
**trace ring**: a lock-free single-producer circular log of 65,536 × 40-byte
`TraceEntry` records (`t0_ns`, `t1_ns`, `kernel_ns`, `total_ns`, `layer`, `M`),
snapshot via `sb_b70_poll_trace_snapshot`, dumped by an opt-in thread under
`SHOOTING_BRAKE_B70_TRACE_DUMP=<path>`. That instrument produced the decode
overlap trace in §5 and cannot perturb what it measures.

`Makefile`: `icpx -fsycl` compiles `b70_capi.cpp` + `../phase1/b70_provider.cpp`
into `libsb_b70_provider.so`, linked against `-lquixicore_xpu
-lquixicore_xpu_ops`.

## `src/QuixiCore-XPU/` — the SYCL kernel library

CMake, `icpx -fsycl`, targets `bmg`. Produces static libs in `build-sycl/`.
Kernel entry points in `include/quixicore/xpu/ops.hpp`:

* `int4_moe_split` — **the production decode kernel.** Fused 2-kernel split
  (gate_up, then down).
* `int4_moe_split_down_wide_sycl` (`kernels/moe/int4_moe/int4_moe_kernel.hpp:52`)
  — the `down2` variant. Built, exported, smoke-tested. **Not called from
  `issue()`.** See §6.
* `nvfp4_moe_split` / `nvfp4_moe_fused` — NVFP4 equivalents.
* `nvfp4_moe_grouped` — **EXPERIMENTAL and unconditionally unprofitable.** 91.1
  vs 58.9 ms at M=2048 while issuing ~28× *fewer* logical reads. Gated off; no
  provider path selects it. Its artifact records the reasoning error to avoid:
  *"`split` is bandwidth-bound at this shape, so cutting traffic is the only
  lever that can help — but that argument predicted this kernel would win, and
  it did not."*

`perf/harness/xpu_bench.cpp` is the microbench: `--kernel membw|int4_moe`,
`--M/--N/--K/--dim/--iters/--warmup`, `--approx down2`.

## The gates (all outside `tests/`)

`tests/` holds only two CPU-only pytest files (route topology/locality
simulators). The real gates live in `src/phase4-8/`:

| gate | what it proves | needs |
|---|---|---|
| `src/phase4/graph_aggregation_oracle.py` | **THE gate.** Drives the production `_b70_issue_graph`/`_b70_take_graph` inside a real CUDA graph, alternating A/B fixtures, checking the raw FP32 remote result against an independent CPU int4 oracle and asserting the CUDA+B70 combine is bit-exact **every replay**. Catches flag-ordering staleness races that 3 replays cannot. | real B70 + bank + built .so |
| `src/phase4/int4_hybrid_enablement_test.py` | placement/bank contract, 9 tests | bank on disk |
| `src/phase4/placement_unit_test.py` | 30 tests, placement math | CPU only |
| `src/phase6/*_test.py`, `src/phase7/*_test.py`, `src/phase8/async_overlap_test.py` | all-cuda vs hybrid token-identical output | real hardware |
| `experiments/b70_122b_pilot_smoke.py` | 122B-geometry SBINT401 decode vs fp32 CPU oracle on real B70, gate `< 1e-3` | pilot bank + B70 |

## Benchmarks

* `benchmarks/serve_88b_128k.sh` — the production launch recipe (§4).
* `benchmarks/bench_88b.py` — GuideLLM driver. Six grids: `A_decode` (128-token
  prompt × C rungs), `B_context` (TTFT sweep), `C_longctx`, `D_saturation`,
  `E_thinking`, **`F_matrix`** (the PRO-matched 8 contexts × C={1..6,10}).
  Guards learned the hard way: GuideLLM exits 0 on an empty measurement window;
  percentiles live under `metrics[k].successful.percentiles.p95`; per-cell
  deterministic `zlib.crc32` seeds so cells can't hand each other cacheable
  prefixes.
* `benchmarks/compare_pro_matrix.py` — reads
  `~/srswti/benchmarks-vllm/bench-matrix/superveloce_88b_nvfp4a16_c6/…/ctx_*/{concurrent,sweep}/benchmarks.csv`.
  Each metric group's **first** column is the mean.
* `benchmarks/b70_itl_probe.py` — streaming ITL/TTFT probe with concurrent B70
  `act_freq` sampling. The A/B instrument.
* `benchmarks/prefill_floor_bench.py` — kernel floor bench, `--mode
  marlin|b12x|vb12x`.
* `experiments/b70_dispatch_latency.cpp` — dispatch decomposition,
  `environment` mode gives per-provenance latency at a configurable idle gap.
* `experiments/b70_multi_topology.cpp` — the two-B70 topology probe.

---

# 4. The 88B: what we serve and how it compares

## The model

`srswti/axe-superveloce-88b-nvfp4a16`. 48 layers, hidden 3072, **180 experts**,
top_k 8, moe_intermediate 1024, 32 attention heads GQA-2, head_dim 256, hybrid
36 GDN + 12 full-attention (every 4th), vocab 248,320. Dense/attention groups are
per-tensor-scalar static FP8; `lm_head` and shared experts are NVFP4 W4A16.

## Production recipe — `benchmarks/serve_88b_128k.sh`

```
SHOOTING_BRAKE_PLACEMENT=split:54     54 CUDA-local experts/layer, 126 → B70
SHOOTING_BRAKE_HYBRID=1
SHOOTING_BRAKE_B70_INT4=1             SBINT401 bank
SHOOTING_BRAKE_B70_GRAPH=1            Tier-3 CUDA-graph doorbell
SHOOTING_BRAKE_PREEMPTIVE_SURGERY=1   } delete B70-owned expert weights
SHOOTING_BRAKE_VRAM_SURGERY=1         } from 5090 VRAM at load
SHOOTING_BRAKE_PREFILL_MARLIN=1       stream int4 experts to the 5090
SHOOTING_BRAKE_B70_PREFILL_STREAM=0
SHOOTING_BRAKE_BANK_REGISTER=1        cudaHostRegister the Marlin bank mmap
SHOOTING_BRAKE_B70_MAX_BATCH=256      the real dispatch/stream crossover knob
ZE_AFFINITY_MASK=1                    the Gen4 B70 (0000:15:00.0)
--max-model-len 131072 --max-num-batched-tokens 8192 --max-num-seqs 16
--kv-cache-memory=3700000000          269,633 KV tokens = 2.06 seats @131K
--language-model-only --reasoning-parser qwen3   port 8016
```

Operational scars, each of which cost a boot:

* `ZE_AFFINITY_MASK` leaking from a prior shell silently loads the bank onto the
  **Gen3** card (−31% ITL). Relaunch with `env -u ZE_AFFINITY_MASK`.
* JIT extension builds invoke bare `ninja`; non-activated shells need
  `PATH="$PWD/.venv/bin:$PATH"`.
* `SHOOTING_BRAKE_B70_STREAM_T` is a **dead knob** on this config — it gates the
  dormant cpu_stream tier, not the Marlin branch. Two boots died to it.
* The bank pin evicts ~9 GiB of server working set on first prefill (2.4M pages
  swapped). Warm with 2 large requests and wait for `/proc/pressure/memory`
  `avg10 < 0.1` before measuring.
* The KV utilization knob's profiler estimate varies ~50 MiB per boot and
  concurrency transients OOM'd 3 boots. **Explicit bytes ended it.**

## Where we beat the PRO 6000

Same checkpoint, same GuideLLM harness, mean-vs-mean, `run6_final/PRO_COMPARISON.md`:

| cell | ours | PRO | verdict |
|---|---|---|---|
| 127K C=1 | **33.99 s** | 39.06 s | **0.87× WIN** |
| 127K C=6 | 94.47 s | 97.39 s | **0.97× WIN** |
| 98K C=1 | 23.18 s | 26.12 s | **0.89× WIN** |
| 98K C=2 | 36.01 s | 39.70 s | **0.91× WIN** |
| 64K C=4 | 25.73 s | 25.78 s | 1.00× tie |
| 8K–32K C=1 | 1.09–5.15 s | 0.57–2.85 s | 1.81–1.91× behind |
| out tok/s @ 98K+127K | 15–22 | 10–19 | **ours at every C** |

Campaign progression at C=1, monotone with no regressions anywhere: 8K 2.19 →
1.08 → 1.09; 32K 9.85 → 5.14 → 5.15; 64K 22.26 → 12.94 → 12.69; 128K-class
55.14 → 35.44 → **33.99**.

Amortization is the mechanism: at 64K our C=1→2 scaling is **+52% for 2× work
vs their +151%** — shared streaming plus co-batched prefill flattens our curve,
and the 1.72× gap collapses to a tie at C=2.

## Where we lose, honestly

**Short-context prefill, 1.8–1.9× behind.** Pure compute. A grouped-GEMM
contest we have now lost twice (see §7).

**Peak throughput at short context — the worst number in the comparison.** The
PRO's sweep peaks at **798 out tok/s @1K**, 613 @4K, 462 @8K. Our 2048-token
peak is 195.5 against ~700 interpolated on their curve — **~3.6× behind**; our
270 @128 against their ≥798 is **~3×**. Peak throughput at short context stacks
many concurrent short prefills, precisely the regime where we are 1.8–11×
behind per-request. **This is structural, not a tuning miss.**

## Verified on the current build, 2026-08-18 [measured-here]

Everything below was re-run today rather than inherited.

| gate | recorded | measured now |
|---|---|---|
| graph oracle, 200 replays, worst abs err | 2.15e-09 | **2.153683453798294e-09** |
| … min discrimination ratio | 1,152,510× | **1,152,510.4** |
| … non-exact CUDA+B70 adds | 0/200 | **0/200** (both flag arms) |
| B70 read ceiling (`membw`) | 599.2 GB/s | **599.635 GB/s** |
| int4 MoE M=1 sustained | 47.9 µs / 68% | **48.57 µs / 65.9%** |
| int4 MoE M=8 | 295.9 µs / 88% | **298.7 µs / 85.7%** |
| KV tokens @ 3.7e9 | 269,633 | **269,633** (2.06 seats) |
| prefill, 18,610 tokens C=1 | interpolates 16K 2.28 s / 32K 5.15 s | **3.00–3.24 s** |
| pytest / placement / int4-hybrid | — | 6/6, 30/30, 9/9 |

**Doorbell host registration, production A/B.** The four doorbell staging
buffers are torch `pin_memory=True` — registered with the **CUDA** caching host
allocator only. Registration is context-scoped, so from the B70's Level Zero
side those ranges were pageable, forcing a staged H2D and a *synchronous* D2H on
every round trip, 96 transfers per decode step.
`B70Provider::register_host_range` wraps `syclex::prepare_for_device_copy`
against the provider's own context; ranges release at shutdown. Default ON, kill
switch `SHOOTING_BRAKE_B70_XPU_REGISTER=0`.

Same binary, flag-toggled boots, 4 probes/arm, 128-in/256-out C=1, warm + PSI
settled (`benchmarks/results/xpu_register_ab/AB_SUMMARY.json`):

| arm | runs (ms) | mean | spread |
|---|---|---|---|
| OFF | 12.180 / 12.133 / 12.094 / 11.931 | 12.084 | 0.249 |
| ON | 11.924 / 11.919 / 11.899 / 11.911 | **11.913** | **0.025** |

**−1.42%**, zero overlap between arms. Registration also **collapses run-to-run
spread by 10×**, which matters more than the mean for a benchmark harness. Boot
logs zero "registration unavailable" warnings — all 48 layers × 4 buffers land.
The in-situ saving is smaller than the standalone because ~75% of the
per-transfer overhead sits under the CUDA-partial overlap — direct evidence for
the max() cost model.

**A VRAM note that will bite the 122B.** Both arms sit ~0.17 ms above the
2026-08-17 absolutes, and a boot at `--kv-cache-memory=3.7e9` OOM'd on an 8K
prefill **by 2 MiB**. Cause: `gnome-remote-desktop` now holds 504 MiB of 5090
VRAM that run6 did not have, consuming the 841 MiB headroom the KV-recovery
sizing depended on. The 3.7e9 recipe is not wrong; it has **no margin against a
changing desktop**. Size 122B KV from a live boot's reported free memory.

---

# 5. Decode anatomy — the one measurement everything is priced against

For months the roadmap rested on an unproven assumption: that the B70 round trip
hides under the 5090's local CUDA partial, because per-layer cost is
`max(B70 leg, CUDA partial)`. If the CUDA partial were the longer leg, then
graph replay, persistent kernels and dual-B70 all save **exactly zero**.

The trace ring settled it. 154 decode steps, 7,392 doorbell windows merged
single-clock with a same-process torch-profiler capture against 320K CUDA GPU
events (`b70_gemv_audit/decode_overlap_trace.json`):

| component | ms | share |
|---|---|---|
| step total (C=1, profiler adds ~5%) | 12.35 | 100% |
| B70 windows (signal→completion) | 4.45 | 36% |
| — **exposed: 5090 idle inside windows** | **2.99** | **24%** |
| — CUDA-busy under windows | 1.55 | 12% |
| inter-dispatch gaps | 7.82 | 63% |
| — CUDA-busy (attn/GDN/partial/sampler) | 5.90 | 48% |
| — **GPU-idle (host/scheduling)** | **1.92** | **16%** |

Per dispatch: **94.3 µs window, 62.9 µs exposed.** 48 dispatches per step.

**The presumption was wrong.** It came from a cache-hot standalone table (44 µs
vs 100 µs) built on an E=8 fixture with a 39 MB working set against 18.6 MiB of
L2 plus compressible fill. Cold-expert reality at k≈5.6 is ~70 µs/layer. **Do
not price overlap decisions against the old table.**

Consequences, each now in milliseconds:

1. **The B70 leg is live: −3.0 ms/step reachable (−24% ITL).**
2. **Host/scheduling idle is a second pool: −1.9 ms/step.**
3. **The CUDA-busy floor is 7.45 ms/step** — the architecture's decode floor
   without speculation. Only speculation divides it.

Instrument caveat: L0 `kernel_ns` read 0 that boot (`B70_PROFILE` did not reach
the provider queue). Window timing is unaffected; chase before a
kernel/transport sub-split is needed.

**The dispatch cost model** (standalone, real geometry, `-1`-padded):

| k | 0 | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| kernel µs | 8.1 | 30.8 | 32.2 | 32.6 | 56.6 | 79.9 |
| wall − kernel µs | 110 | 110 | 109 | 113 | 108 | 108 |

Kernel cost is flat k=1..4 then explodes; **the ~108 µs floor is constant, so it
is 7 queue submissions, not bandwidth.** That number is the target of §6's
second lever.

---

# 6. The two unshipped decode levers

## `down2` — measured live today, a concurrency lever only

`int4_moe_split_down_wide_sycl` widens the `down` stage's workgroup shape. It is
compiled, exported and smoke-tested; `grep down_wide src/phase1/b70_provider.cpp`
returns **nothing** — `issue()` has never called it.

Measured today, 6,000 iters, sustained clocks, production geometry
(E=126, K=3072, I=1024, top_k 8) [measured-here]:

| M | split | down2 | delta |
|---:|---:|---:|---|
| 1 | 48.5 µs / 66.0% | 50.8 / 63.1% | **down2 loses 4.7%** |
| 2 | 83.6 / 76.6% | 77.6 / 82.5% | down2 **+7.2%** |
| 4 | 155.3 / 82.4% | 153.2 / 83.6% | +1.4% |
| 8 | 307.3 / 83.3% | 287.9 / 89.0% | **+6.3%** |

**M is concurrency at decode** — one token per sequence per step. So at C=1, the
headline ITL, `down2` is *worse*. It pays from C≥2 only.

Priced at C=8: 19.4 µs × 48 dispatches = 0.93 ms/step gross; at the C=1 exposure
ratio (66.7%) that is ~0.62 ms against the 26.60 ms C=8 ITL ≈ **−2.3%**
[INFERENCE — the exposed fraction at C=8 is unmeasured; the trace was C=1].

Implementation: a predicate on `M × valid_routes ≥ 8` in `issue()`, dispatching
`int4_moe_split_down_wide_sycl` instead of `int4_moe_split`. Twenty minutes,
near-zero risk — it is a variant selection, both are correctness-gated.

**For the 122B this does not improve.** All-remote 256 experts means ~8
routes/token split ~4 per card, so each card sees the M=1-equivalent 4-pair
shape at C=1 — where `down2` still loses. Same story, still a C≥2 lever.

## SYCL-graph replay — capture works, but the lever is ~1%, not ~8%

`grep command_graph src/phase1/b70_provider.cpp` returns nothing: `issue()` is
eager. The graph work lives only in `src/QuixiCore-XPU/tests/xpu_graph_smoke.cpp`
and `src/runtime/graph.cpp`.

The mechanism is the 108 µs flat floor above: 7 queue submissions per dispatch
collapse to one replay. An earlier standalone run recorded **k=1 105.1 → 75.3 µs
(−28%)** and that number drove the roadmap. **It does not reproduce.**

**Re-measured 2026-08-18, `./b70_dispatch_latency int4-graph`, three runs**
[measured-here]. Eager seven-command chain (A) vs one command-graph replay (B),
same shape both arms, p50 µs:

| k | eager | graph | delta |
|---:|---|---|---|
| 1 | 50.76 / 50.16 / 50.53 | 46.46 / **54.71** / **54.53** | noise — graph *lost* 2 of 3 |
| 4 | 58.98 / 58.57 / 59.07 | 56.17 / 56.09 / 56.31 | −2.8 µs (−4.7%) |
| 8 | 106.98 / 106.54 / 106.74 | 102.64 / 102.58 / 102.63 | −4.2 µs (−3.9%) |

The old −28% came from the contaminated run `docs/next-steps-88b.md` already
flags — eager and graph sharing a queue, inflating the eager baseline. The real
delta is a consistent but small ~4 µs at k=4–8, and **nothing at k=1**.

Re-priced at the production shape (~5.6 of 8 routes remote at C=1, so k≈4–8):
4 µs × 48 dispatches = 0.19 ms/step gross, × 66.7% exposed ≈ **−1.0% ITL**
[INFERENCE]. Not −7.7%.

**And it is not free.** The same run surfaces a cost nobody had recorded:

```
finalize_us = 250,000        M_contract: captured_M=1, dynamic_M=no
required_decode_graph_buckets = 32
reason = captured_copy_sizes_and_kernel_ranges
projected_32_bucket_finalize_ms = 8221
```

Batch size is **baked into the captured copy sizes and kernel ranges**, so this
is not capture-once — it is capture 32 times at ~250 ms each. **~8 seconds of
boot for ~1% ITL.**

Correctness is fine (`eager_graph_max_relative=7.63e-08`,
`graph_cpu_max_relative=1.13e-06` against a 5e-05 bound), so the option stays
open. It is simply no longer the top lever, and the decode ranking in §6 below
reflects that.

Feasible only because the doorbell staging buffers are **static addresses** —
route *contents* change every step, addresses do not. Risk class is
staleness/freshness, exactly what the 200-replay oracle exists to catch.

**Three XPU graph landmines to design around from the start**
[measured-elsewhere, same card class, live crashes]:

1. **NEO command-list overflow.** `XPUGraphImpl::replay` submits via
   `submit_with_event` and never syncs; a tight per-step replay loop overflows
   `linear_stream.h:84` after ~1–2k tokens (many-op graphs) to ~96k tokens. A
   doorbell firing every decode step is *exactly* this shape — sync every N
   replays or confirm a reclaiming primitive.
2. **SLM kills capture — CLOSED, does not apply to us** [measured-here].
   The landmine is specific to the `work_group_scratch_memory` *extension*,
   which is "not yet available with the SYCL Graph extension". Our kernel uses
   classic SYCL 2020 `sycl::local_accessor`
   (`int4_moe.sycl.cpp:60,176,277` — a `2 * kGateReductionSubgroups * kSG`
   float scratch for the subgroup reduction), which captures fine:
   `int4-graph` reports `graph_supported=1` and replays the real seven-command
   chain correctly. Note `tests/xpu_graph_smoke.cpp` proves nothing here — it
   only captures `ops::silu`, which uses no local memory.
3. **One graph cannot span two devices.** `command_graph::begin_recording`
   rejects a second device. Dual-B70 means two graphs plus an external L0-IPC
   event or SYCL `external_semaphore`.

Plus a correctness trap: a persistent-scheduler work counter must be `at::zeros`,
not `at::empty` — SYCL does not guarantee group 0 runs first, and a dirty
leftover becomes the starting tile index, especially under replay.

**For the 122B, graph replay is the *dual-card* question, not an ITL lever.**
Two providers each pay the 7-submission floor, and host-side submission cost is
serial on the calling thread even though the cards run concurrently — so the
~4 µs saving may double while the concurrent-dispatch serialization is the real
unknown. Landmine 3 applies directly: one graph cannot span two devices, so this
is two graphs plus an external L0-IPC event or SYCL `external_semaphore`.

## Decode levers, ranked after the 2026-08-18 re-measurement

| lever | expected | cost | status |
|---|---|---|---|
| **dual-B70 route split** | halves the remote leg → ~−1.5 ms ≈ **−12%** | on the 122B path anyway | banks built, split math proven |
| **kernel occupancy** — gate_up exposes only 1,024 subgroups, ⅔ of the bytes | ~1.35× cap → ~0.6 ms ≈ **−5%** | real kernel work; a gate_up analogue of `down_wide` | not started |
| **`down2`** at C≥2 | ~**−2.3%** at C=8, **zero at C=1** | ~20 min, one predicate | kernel built + exported |
| **150 W power cap** | 8.2% throughput, same card class | free, no code | not applied |
| **SYCL graph replay** | ~**−1.0%** | 8 s boot + 32-bucket cache | capture verified working |
| speculation (122B native MTP) | divides the 7.45 ms CUDA floor | Phase F gates | 122B only |

Graph replay was ranked first before today. It is now second-to-last. The
re-measurement cost ten minutes and saved scheduling a mispriced lever — which
is the campaign's own process rule (*bake off quality before speed*, and
re-measure any number you did not take yourself) applied to our own roadmap.

## Also free, also unapplied

**150 W power cap on the decode B70.** MoE decode self-limits to ~140 W
regardless of cap — bandwidth-bound, not frequency-bound. Measured on the same
B70 SKU: **150 W gives 125.7 tok/s vs 230 W's 115.4 — 8.2% *better* at lower
power** [measured-elsewhere, same card class]. `hwmon power1_cap` before boot.
Caveat: dense (non-MoE) shapes scale +18–30% with power, so this is
MoE-decode-specific.

---

# 7. Kernel bake-offs — both closed negative, recorded so they are not re-fought

**Process rule this campaign earned: bake off *quality before speed*.** The b12x
detour measured 2.6× and then spent hours discovering the number was unusable. A
single encoding probe against the vendor's own quantizer would have closed it in
minutes.

## Round 1 — `FlashInferB12xExperts` (W4A4): 2.6× faster, unusable

| M | Marlin ms/layer | b12x W4A4 | b12x W4A16 |
|---|---|---|---|
| 2048 | 2.04 | 0.92 (2.2×) | — |
| 8192 | 7.07 | **2.70 (2.6×)** | 8.16 (slower) |
| 32768 | 27.0 | 10.42 (2.6×) | — |

Quality killed it. Driving the kernel with **flashinfer's own quantizer** —
bypassing our bank entirely — per-layer cosine tops out at **0.82**, and that
ceiling compounds over 48 layers. The cause is structural, not a bug: FC1 input
quant uses a **static** e2m1 grid with no per-block rescue (their own test
`test_input_global_scale_decouples_weight_alpha` states the contract).
Half-order, nibble-order and sf2-layout permutations all measured *identical*
error, which rules out layout and leaves the arithmetic.

**Salvage, all real:**

* **A filing-ready upstream vLLM bug.** `FlashInferB12xExperts`' load-time bake
  multiplies ModelOpt's per-expert `weight_scale_2` into the e4m3 block scales;
  for this checkpoint the product falls below e4m3's smallest subnormal and
  **86.5% of scales flush to zero**. ModelOpt itself clamps that scale to
  [2^-9, 448] "to avoid underflow→0", so vLLM re-introduces a hazard the source
  library already fixed. Our fix — bake only the gate/up *ratio* (O(1),
  e4m3-safe) and carry the true scale as per-expert fp32 alpha planes — is
  implemented in `b12x_bank_format.py` + `src/phase1/build_b12x_bank.py`.
* **bank-v2 tooling**: a 29.9 GiB NVFP4 bank built and bit-validated from the
  checkpoint's native planes in 6 minutes, zero requantization. Ports to any
  future FP4 kernel.

## Round 2 — `vendor/b12x` w4a8_nvfp4: quality passed, speed failed

w4a8 uses **dynamic per-32-block** activation quant (UE8M0+E4M3, amax per block,
computed inline) — a categorically different regime from static-global W4A4.

Quality gate **PASSED** on real weights through our own planes: cosine
**0.9985/0.9983/0.9984** across layers 24/0/1/41 vs the exact fp32 checkpoint
oracle (W4A4 measured 0.82). Kernel-vs-oracle on GPU: 0.9993/0.9992 against
their in-repo bar of 0.998 — **the kernel adds nothing beyond w4a8's own
quantization**.

Speed gate **FAILED**:

| kernel | M=8192 ms/layer | quality | verdict |
|---|---|---|---|
| **Marlin (incumbent)** | **7.07** | W4A16 ref | **keeps the crown** |
| vendor-b12x w4a16 | 7.93 (+12%) | Marlin-class | slower |
| vendor-b12x w4a8_nvfp4 | 8.34 (+18%) | 0.9985 (passed) | slower |
| flashinfer-b12x W4A4 | 2.70 (−62%) | 0.82 | dead (round 1) |

GPU-event-verified kernel-bound. The scouts' "half Marlin's latency" projection
came from a lab with **zero 5090 numbers**; now the 5090 numbers exist.

**Durable salvage:** (a) w4a8 quality 0.998+ on real weights is a standing
result — any future faster sm_120 w4a8 kernel inherits a ready gate, byte-verified
swizzle adapters, and the up-first bank recipe; (b) the small-M-direct decode
candidate (M≤8) is untouched by this verdict — different regime.

**Two integration facts worth their own line.** Bank-v2 sf planes need a swizzle
adapter (flashinfer emit order → b12x's TRT-LLM arrangement; both directions
byte-verified). And our gate-first row stacking is b12x's `w13_layout="w31"` —
the wrong default **silently computes `silu(up)*gate` and scores 0.87**. We hit
that twice, once per arm. Bank v3 emits up-first planes. Never use `w31`: its
`_W13_NORMALIZED_STORAGES` does a one-time in-place row swap that is
deliberately never cleared, so it silently skips on pointer reuse.

## The CUTLASS wall

**sm_120 tensor cores are F8F6F4-only.** `cute/arch/mma_sm120.hpp` never admits
`int4b_t`, `uint4b_t` or `bfloat16_t` as an A/B operand, static-asserted in
**both** the dense (`sm120_mma_builder.inl:76-79`) and **grouped**
(`sm120_array_mma_builder.inl:78-79`) builders. There is **no int4×bf16
mixed-input grouped GEMM for our silicon** anywhere in the tree; the mixed-input
grouped kernels live on sm90/sm100 only.

Reusable pieces if we ever hand-build one: the grouped tile scheduler resolves to
the **arch-generic** `PersistentTileSchedulerSm90Group` and is proven compiling
on sm_120 today; a portable bf16 atom exists at raw `cute::arch` level
(`SM80_16x8x16_F32BF16BF16F32_TN`) with no CollectiveBuilder wiring; the
int4-unpack LOP3+PRMT math is arch-generic.

Effort ~5–8 weeks. **And the decisive point: it would target `mma.sync` — the
same instruction class Marlin already uses.** So it buys better scheduling, not
new silicon capability.

**Gate: do not attempt unless a roofline analysis first shows Marlin is >20% off
the sm_120 bf16 tensor-core ceiling at our shape.** That analysis is cheap,
analytical, and has not been done. **It is the honest next step on short-context
prefill.**

---

# 8. Vendor trees — what to take, what to bench, what to leave

## TAKE NOW — zero or near-zero cost

1. **150 W B70 decode cap** — see §6.
2. **b12x's pure-torch NVFP4/MXFP8 *encode* + swizzle helpers.**
   `vendor/b12x/b12x/_lib/intrinsics.py:55-320` —
   `quantize_grouped_nvfp4_torch`, `swizzle_block_scale`,
   `fp4_quantize_values_torch`, `pack_grouped_fp4_values`,
   `quantize_grouped_mxfp8_torch`, `pow2_ceil_ue8m0_torch`. Plain torch,
   CPU-callable, byte-exact against their own GPU TMA kernel in their tests. We
   own a validated CPU *dequantizer*; this fills the **encode** direction we
   currently hand-roll. **Surgical extraction only** — the same file carries
   thousands of lines of `@cute.jit` GPU-only kernels.
3. **Their A/B timing primitive.**
   `vendor/b12x/validation/cutlass_migration/core/exact_cache_abba.py:393-1008`
   — 1%-trimmed mean gated on `nvidia-smi` P-state, throttle mask and clock
   delta, with ABBA ordering to cancel drift. Our `prefill_floor_bench.py` uses
   a plain median of 5–7 with **no throttle gate**, a real hole. Lift the
   primitive; **reject** the surrounding SHA-pinned 104-position formal-release
   machinery.
4. **PyCuTe as a layout-verification aid.** `vendor/CuTe/pycute` is pure Python
   layout algebra — `layout.py`/`algebra.py`/`swizzle.py` plus SVG/LaTeX
   visualizers, zero kernels. We derived the flashinfer↔b12x swizzle adapters
   empirically by index-tracing; this is the tool for doing it symbolically.
5. **Graph-capture discipline + the pitfall catalogue** — stable-address
   fixed-capacity buffers, "prepare channels before capture", stream-affine
   binding, GC-quarantine of abandoned runtimes.
6. **Never sync `vendor/vllm` over the installed vLLM.** Same 0.27.1 line,
   line-for-line identical across every subsystem we touch, with **one
   exception**: the *installed* tree has `calculate_kv_scales` wired including a
   **hybrid-model auto-disable guard** (`models/config.py:450-463`, citing
   vllm#37554: "uninitialized recurrent state corrupts scales during the
   calibration pass") that the vendored tree lacks. Syncing forward would
   silently drop a safety guard written for **exactly our GDN/Mamba + fp8-KV
   architecture.** Treat `vendor/vllm` as read-only reference.

## BENCH FIRST — live candidates, ordered

**1. GDN prefill: route to flashinfer's sm_120 kernel — a live bug-class hit.**
vLLM's selector (`qwen_gdn_linear_attn.py:116-133`) sets `supports_flashinfer`
only for `is_device_capability(90)` (Hopper) or
`is_device_capability_family(100)` (sm_10x). Our sm_120 satisfies **every other
condition** — `head_k_dim == 128` ✓, CUDA ≥ 13 ✓ — but fails the family check,
so we silently run the Triton/FLA fallback (confirmed in our boot log).
Meanwhile flashinfer ships `delta_rule_dsl/delta_rule_sm120.py`
(`chunk_gated_delta_rule_sm120`) — a GDN kernel **named for our exact card** —
in our *installed* 0.6.16.post3.

`ChunkGatedDeltaRule` is a `PluggableLayer`, the same OOT mechanism our plugin
already uses, so this needs **no vLLM patch**. Buys ≤7% of an 8K TTFT
(`gdn_attention` is 0.070 s of 0.82 s GPU-busy at 8K). Modest, but it is a
genuine gap and the only arch gate worth fixing (see the audit below). Gate on a
firsttok/logprob envelope — it changes a numerics path.

**2. `gemm.tensor_fp8_linear` for our attention/linear-attn projections.**
`vendor/b12x/b12x/gemm/tensor_fp8_linear/_kernel.py` — a static per-tensor E4M3
dense GEMM for sm_120 whose `pack_weight(weight_fp8, input_scale*weight_scale)`
contract is a **verbatim match** for our `q/k/v/o_proj` and `linear_attn.*`
tensors. Their postmortem (`docs/sm120_dense_fp8_deepgemm_port.md`) records a
tile-selection fix — (128,128)→(64,128) M-independent default — taking it from
**1.44× slower to 1.6× faster than FlashInfer CUTLASS, measured on an RTX PRO
6000** [measured-elsewhere, same sm_120 family → **transfers**].

Target: the `unattributed_gemm 0.149 s` line in our own 8K attribution — the
second-largest busy category after MoE and **the one we have never attacked**.
Their doc also narrates a weight-requantization bug (a dropped power-of-two
re-quant step) worth reading as QA before we repeat it.

**3. Predictive expert placement — colibri's PILOT prefetcher.** A fully
measured algorithm for our unspent hotness-ordered-placement lever: **71.6%
one-layer-ahead routing recall on a real 256-expert MoE**, via EMA-smoothed
router logits, a two-step correction, and an LFRU eviction guard
[measured-elsewhere, different model]. Buys fewer streamed bytes and less remote
compute — **compounds with any kernel**, which is exactly why it survived both
negative bake-offs. Our router is measurably skewed already: 9 all-CUDA steps in
6,144 against 0.40 predicted under uniformity, a 22× chance.

**4. Adaptive local/remote split — exllamav3's `MoeCpuHost`.**
`moe_cpu_host.py` implements a **bandwidth-probed break-even threshold**
deciding which experts stream versus compute locally — architecturally
near-identical to our split, except ours is a **static 54/126 chosen once**.
Directly relevant to tuning across our **two asymmetric links** rather than
assuming symmetry.

**5. MoE-decode sweep methodology.**
`vendor/b12x/scripts/sweep_moe_decode_max_active_clusters.py` — CUDA-graph
replay, `torch.cuda.Event` timing, `_mean_ci` = mean ± z·SEM. Clone the
methodology for our decode knobs. Correction to an earlier assumption:
`MoEMicroKernelW4A16SmallMDirect` is **W4A16-only** while our local 54 experts
run NVFP4; the quant-mode-correct analog is `MoEMicroKernelBackend`
(`moe/_shared/kernels/micro.py:382`). **Hazard:** unlike the w4a16 branch, the
NVFP4 branch **does** enter the `data_ptr`-keyed
`_WEIGHT_CACHE`/`_W13_NORMALIZED_STORAGES` caches — a live staleness hazard
under our rotating arenas. Pre-derive and pass explicitly, or don't take it.

**6. b12x's PCIe DMA discipline — the transferable half of `comm.pcie`.** The
collectives are dead for us (CUDA-IPC, needs ≥2 CUDA GPUs), but the DMA layer
underneath is the good part:

* **CE beats SM-copy by 1.65× on PCIe** — their header: "NCCL's SM-copy
  transport sustains ~34 GB/s on this fabric while CE peer copies run at
  ~56 GB/s". This retro-validates our registered path (18.5 → 53.9 GiB/s = the
  CE path) and forward-looking it is a **guard**: if a future transfer silently
  lands on an SM-driven copy kernel we lose ~40% with **no error**. The 9 µs
  submit-wall we measured is the signature to assert on.
* **Separate CE stream + flag stream; device-resident *monotonic* flags;
  `FLAG_STRIDE = 128`.** The monotonic never-reset counter is precisely what
  lets their graphs "replay without host patching" — the property our decode
  graph replay needs, and the same class as the NEO-overflow landmine. If we
  build the persistent doorbell, this is the proven shape.
* **`recommend_prefetch_depth`** makes prefetch depth *a measured policy with a
  kill condition*. We hardcode depth-1 (2 arenas) because it fit VRAM and never
  re-checked after register-DMA cut transfer from 1.48 to 0.51 s/pass. Depth-0
  at some contexts would free 584 MiB — material at 1.60 KV seats.
* **Correctness-gate-before-timing.** Our floor bench times first and gates
  separately; a harness that refuses to report a number for a wrong kernel is
  strictly better.

**7. KV capacity — SGLang's mamba-capacity pool solve.** Worth reading against
our `mamba_block_size`-coupled **4,176-token attention page**, which is the
remaining KV waste item: a 640-token request still costs a full block. Same
corpus carries **retokenized-ITL** measurement, the spec-decode-fair way to
report latency.

**8. Config-level sm_120 knobs.** `vendor/rtx6kpro/optimization/nvfp4-quantization.md`:
the `VLLM_NVFP4_GEMM_BACKEND=cutlass` selector, the `sm120f` family-conditional
PTX requirement for NVFP4 conversion, a hybrid "NVFP4 + BF16 shared-expert +
layer-0" VRAM/quality technique, and a flag that **FP8 KV cache is broken on
SM120 for GLM-5** — we run fp8 KV, worth a numerics spot-check.
`hardware/gpu-configs.md`: **GB202 power sweep, 500 W ≈ 600 W parity; 300 W MaxQ
costs ~4% single-user, ~30% at 64-concurrent** — our exact die.

## Closed negative, do not re-open

* `sgl-kernel-xpu`'s W4A16 grouped MoE GEMM (`moe_grouped_mm_nt_xe20_w4a16`).
  Built for `bmg`, correctness-gated at cosine 0.99998, benched at our exact
  geometry. **Loses at every decode shape: 2.1× slower at M=1 (100.7 vs
  47.9 µs), 49% vs 79% of ceiling at M=2.** Closes only at M=32, a prefill shape
  the B70 no longer serves. Their 5-launch orchestration (prepare + scatter +
  GEMM1 + silu_and_mul + GEMM2) is exactly wrong for latency-bound M=1. vLLM's
  `XpuFusedMoe` int4 measures identical (±0.5%) — same Xe20 grouped-GEMM family
  underneath. Both remain installed in `.venv-xpu` for future shapes. **Salvage
  was `down2`** (§6).
* Naive weight-only int8 (materialize bf16 weights, then bf16 GEMM) measured
  **0.5–1.2×, sometimes slower than bf16** — fused epilogue-dequant is
  mandatory, not optional.

## PARK — real, expensive or gated

**QuTLASS rotations (IST-DASLab, the Marlin authors) — the W4A4-quality answer,
priced in weeks.** Fused Hadamard-as-micro-GEMM in the quantize epilogue with
runtime-loaded rotation matrices: the literature-proven fix for exactly the 0.82
ceiling. **Definitive answer on the cheap version: NO.** Activation-only online
rotation is impossible — the same Hadamard must rotate **both** operands
(RxRᵀ=I holds in one basis only), proven by their own tests quantizing the
weight through the identical `fusedQuantizeNv(b, h, global_scale)` call with the
same `h`. So rotation requires an **offline rotated + requantized NVFP4
checkpoint**.

Remaining cost: `bindings.cpp` is hard-gated to 2-D dense (`A.dim()==2 &&
B.dim()==2`), no grouped/MoE path exists, so we would write the grouped-GEMM
host wrapper ourselves; `third_party/cutlass` is an empty submodule making the
build network-dependent. Build itself is plausible (torch ≥ 2.11 admits our
2.13; `sm_120a` gencode and a `TARGET_CUDA_ARCH=120` branch both exist). All
in-repo evidence is kernel-vs-reference **correctness**; the speedup plots are
MXFP4-vs-BF16 on **dense** models via an un-vendored harness [claim]. **The only
credible route to W4A4-class speed with quality. Revisit when the short-context
gap is the last thing standing.**

**Persistent B70 decode kernel.** Template exists —
`PersistentTileSchedulerMoE` (CUTLASS-pattern persistent CTA, fixed grid, ragged
per-expert M, swizzled raster), plus `dnnl_matmul_w4a16_int4`, a complete oneDNN
fused weight-decompress int4 recipe. Parked because the dual-B70 team
independently put "megakernel / persistent decode kernel" **last** on their own
roadmap after cheaper GEMV and graph levers — a hardware-matched signal. Also
gated by the SLM-in-graph landmine.

**Intel Graphics Compiler.** Already running invisibly under every SYCL/L0 call.
Its only value is ISA-level shader dumps to confirm the compiler emitted the
DPAS/GEMV path we expect — `IGC_ShaderDumpEnable=1` on the stock driver, no
multi-hour LLVM build.

## REJECT — recorded so we stop looking

| target | why dead |
|---|---|
| **TileRT** | Closed binaries, Dockerfile hard-pins `CUDAARCHS=100` (B200), torch 2.11, DeepSeek 128×128-block FP8. Zero kernel source. |
| **AdaptiveCpp** | A redundant second SYCL toolchain. Every real B70 kernel builds with `icpx`. Pure duplication risk. |
| **b12x `comm.pcie` collectives** | CUDA-IPC + NCCL; gates on `device_count() >= 2` CUDA GPUs. |
| **b12x MLA / sparse-MLA / NSA-indexer / mHC** | We are GDN-hybrid, not MLA; mHC hardcodes DeepSeek 4096/7168 vs our 3072. |
| **b12x `trellis_linear`, `wo_projection`** | Block-scaled MXFP8 vs our per-tensor-scalar FP8. |
| **uccl** | CUDA/RDMA only. |
| **SGLang distributed EP** | Hard-requires NVLink/IB. Also `vendor/sglang` and `vendor/sglang-upstream` are **byte-identical** — no fork delta to mine. |
| **SGLang MoE kernels as a Marlin challenger** | Same Marlin/FlashInfer-CUTLASS families already benched negative. |
| **flashinfer 0.6.18 upgrade** | Tried and rolled back: its `input_global_scale` fix does not touch W4A4 activation quant, so it does not reopen the 0.82 verdict, and it carries sampler/FP8 regression risk. |
| **PRO 6000 collective evidence JSONs** | Every real number is TP8/DCP4 8-GPU collective scaling. |
| **GPUDirect Storage for the bank** | Targets NVMe→GPU at ~7 GB/s; our bank is deliberately page-cache resident at 53.9 GiB/s. **~7× downgrade. Never propose it.** This also explains pin-eviction thrash: losing page cache drops us onto exactly that path. |
| **MMA multipath host-GPU copies (arXiv 2512.16056)** | Relays through *peer CUDA GPUs* over NVLink. We have one CUDA GPU, no NVLink, and a B70 relay would read host DRAM over Gen4 x4 then forward B70→host→5090, consuming the 5090's own link anyway. Their own table measures **0.94× at TP=8** where no spare peer exists — our permanent regime. |
| **Wire codec (`_dma_kernels.py` quant/dequant)** | 48.4% wire reduction, but our M=1 decode payload is 6 KiB ≈ 1.5 µs over Gen3 x4, so halving saves 0.77 µs against 11.9 ms ITL = **0.006%**. Prefill doesn't use the doorbell. |
| **ServerlessLLM-style partitioned parallel load** | Priced: aggregate weight-read 57.9 → ~68 GB/s (+18%). But stream is already fully hidden under compute at ≥8K so it buys ~0 there, and at short context the B70's prefill compute is exactly what we abandoned. |

## The arch-gate audit — the correction that matters most

An earlier note recorded "sm_120 falls through arch gates in four independent
places" and called it a cheap-to-fix class. **That was an overclaim.** Auditing
every backend selector our running server actually logs: the gates are real, but
in three of four cases **the kernel behind the gate quantizes activations to FP4
— the W4A4 regime we measured at cosine 0.82 and killed.** Fixing them would buy
speed we cannot use.

| instance | gate real? | would fixing help? |
|---|---|---|
| GDN prefill (`family(100)`) | yes | **YES — linear attention, not a quantized GEMM; numerics preserved** |
| `FlashInferB12xExperts` MoE (SM121 guard) | yes | no — W4A4, measured 0.82 |
| NVFP4 dense `FlashInferCuteDsl` (`family(100)`) | yes | no — and its sibling `FlashInferCutlass` *does* admit sm_120, but `input_quant_key() -> kNvfp4Dynamic` = W4A4 |
| CUTLASS F8F6F4-only asserts | not a gate — an ISA fact | n/a |

**What the server selects, and why each is right:**

| selector | chosen | verdict |
|---|---|---|
| NVFP4 dense GEMM | `MarlinNvFp4LinearKernel` | correct — of the six NVFP4 linear kernels **only `flashinfer.py` overrides `input_quant_key`**, so every faster option is W4A4 |
| ModelOpt FP8 linear | `FlashInferFP8ScaledMMLinearKernel` | correct for per-tensor-scalar FP8. vLLM #47749's "silent Marlin fallback" does **not** apply — our checkpoint is genuinely mixed |
| Attention | `FLASHINFER`, `decode_backend=flashinfer-native`, `arch=sm120`, autotune cache under **`120f`** | correct, already on the family-forward suffix |
| MoE | `MARLIN` of 8 candidates | litigated across two bake-offs |
| top-k/top-p sampler | logs "Using FlashInfer" | **phantom — never invoked.** `forward_cuda` opens `if (k is None and p is None) or generators: return self.forward_native(...)` and our harness sends no top-k/top-p. The log announces *availability*. **Do not chase FlashInfer #3389 on our numbers.** |
| GDN prefill | **Triton/FLA fallback** | ⚠️ the only genuine gap |

**GDN *decode* has no backend selector at all — a different problem.** vLLM's GDN
decode calls vendored FLA Triton kernels directly
(`fused_sigmoid_gating_delta_rule_update`,
`fused_recurrent_gated_delta_rule_packed_decode`). There is **no selector to be
gated out of** — no integration exists. Meanwhile our *installed* flashinfer
0.6.16.post3 ships `flashinfer/gdn_kernels/` with `gdn_decode_bf16_state`,
`_bf16_wy_output_only`, `_mtp`, `_nontranspose`, `_pretranspose`, plus
`blackwell/` and `delta_rule_dsl/`, and its documented contract lines up with
what vLLM already passes:

* `gated_delta_rule()` — **T=1 single-token decode**, exactly our step shape.
* State **pool mode** `[pool_size, HV, V, K]` indexed by
  `initial_state_indices` — the same shape as vLLM's `ssm_state_indices`.
* **Split-pool writes** (`output_state_indices != initial_state_indices`) —
  matches vLLM's separate read/write index tensors.
* `gated_delta_rule_mtp()` for T≥1 — **already the right shape for spec-decode
  verify**, which matters directly for the 122B.
* An ILP=4 higher-occupancy variant exists.

Days, not hours, through our existing `PluggableLayer` hook, and not a research
project since the contract shapes match. Unverified: whether these kernels accept
`head_k_dim=128` and our exact state dtype/layout. **Decode is our worst metric
and unlike the prefill gap this one is not reachable by flipping a gate.**

## Two more meta-findings worth keeping

**Small-M kernels leave 30–50% of bandwidth on the floor.** On the B70, int8
GEMV at M=1 reaches 51–59% of 608 GB/s where bf16 reaches 71–76%. Native `s4×s4`
DPAS exists and is bit-exact on this card, yet a naive tiled mainloop caps at
**~64 TOPS versus int8's 367 TOPS**. *Having the instruction is not having the
performance.*

**On the B70, decode is dominated by launch overhead, not compute.** A comparable
model measured **~950 kernel launches per token: 32.2 ms of CPU enqueue against
9.45 ms of actual GPU work**, and graph capture took it from 21.8 → 93.0 tok/s
(**4.0–4.3×**) [measured-elsewhere, same card class, different model]. This is
why the doorbell/graph direction aims at a real problem class — and warns that
our gains there may come from *submission* economics rather than kernel math.

**sm_120's gaps are ecosystem, not silicon.** No tcgen05/TMEM/WGMMA/FA3/
DeepGEMM/FlashMLA-sparse paths. Every datacenter-tuned kernel skips our card and
the fallbacks are what we run.

---

# 9. The 122B — prerequisites before anything boots

Target: `unsloth/Qwen3.5-122B-A10B-NVFP4` on the 5090 with **both** B70s.

## Why the plugin transfers

Attention/GDN geometry is **field-for-field identical to the 88B**: 48 layers,
hidden 3072, 32 heads GQA-2, head_dim 256, full-attention every 4th, GDN key dim
128 / 16 key heads / 64 value heads, conv 4, moe_intermediate 1024, vocab
248,320, top_k 8, max_position 262,144. The only deltas are **`num_experts`
180 → 256** and the quantization recipe. Attention, KV and Marlin-prefill
machinery all carry over unchanged.

## Checkpoint facts, measured from the safetensors headers [measured-here]

| bucket | GB |
|---|---:|
| routed experts, 48 layers | 66.29 |
| GDN | 3.20 |
| embed | 1.53 |
| attn | 0.94 |
| visual (skipped under `--language-model-only`) | 0.90 |
| lm_head | 0.76 |
| shared expert | 0.26 |
| **dense, text-only, no MTP** | **6.77** |
| MTP head (separate file) | 5.05 |

Quantization is heterogeneous by design, verified against `config.json`:

* `group_1` — W4A4: E2M1 group-16 weights, FP4 group-16 dynamic-local
  activations, **checkpoint-provided per-expert `input_global_scale`**. Covers
  routed experts in layers **0–46** plus shared experts.
* `group_0` — FP8 W8A8 channel/dynamic: `self_attn.{q,k,v,o}_proj`,
  `linear_attn.{in_proj_qkv,in_proj_z,out_proj}`, `lm_head`, **and layer 47's
  routed experts and shared expert**.
* `kv_cache_scheme`: FP8, static per-tensor, symmetric.
* `ignore` carries 241 non-visual entries plus **`re:^mtp.*`**.

**This is why the round-1 W4A4 failure does not reject this checkpoint.** That
test forced an *uncalibrated* activation grid; the 122B ships calibrated input
scales. This is the intended regime, and vLLM's CUTLASS NVFP4 MoE consumes
`a1_gscale` today on the already-selected backend — **no flashinfer upgrade is
needed for the standard path.** Only MARLIN forces weight-only; every other
backend threads calibrated scales. (The B12x wrapper stays API-blocked in
0.6.16.post3 — `input_gs=w1_alpha` hardcoded — but the low-level kernels accept
per-expert `input_global_scale`, so a private-API bypass exists without a
version bump. Justify it with a roofline first.)

## Native MTP: it exists here, unlike the 88B

**This is the biggest single difference from the 88B campaign.** The 88B has
**zero** `mtp.*` keys in both the quantized derivatives *and* the bf16 base — the
quantization step dropped them, so speculation there means training an EAGLE3
head. The 122B ships one.

`model_mtp.safetensors`: **785 tensors, all BF16**, 5.05 GB (768 expert tensors
= 4.83 GB; 17 dense = 0.22 GB). Listed in `model.safetensors.index.json`, so
vLLM loads it with the model. A real one-layer MoE draft block with its own 256
experts, its own `self_attn`, and its own `shared_expert`.

Three corrections to earlier planning:

1. **It is 5.05 GB, not ~1.4 GB.** The KV budget loses 3.6 GB relative to plan.
2. **No side-loading and no null-quant-config gate is needed.** `re:^mtp.*` is
   in the compressed-tensors ignore list and `config.json` sets
   `unsloth_fixed_mtp: true`, so the head stays BF16 by construction. This
   matters because Intel's same-family field data is emphatic: MoE MTP gains
   **collapse with a quantized draft head (+3%) versus BF16-preserved (+36%;
   dense got +79%)**.
3. **The method name is `"mtp"`.** On vLLM 0.27.1, `qwen3_next_mtp` and
   `qwen3_5_mtp` are deprecated aliases remapped to `"mtp"`
   (`config/speculative.py:697-701`); `qwen3_5_moe` resolves to `Qwen3_5MoeMTP`
   (`:516-525`). Earlier research recommending `qwen3_next_mtp` is stale for
   this install.

```
--speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

`n_spec > 1` emits a warning about re-invoking one layer (`:907-916`), not a
hard cap. **Cap at 2 anyway**: MTP>3 is a live upstream crash class (vllm#34948,
same GDN-rejection family) and 2 is the measured sweet spot.

**Rollback is in-tree and proposer-agnostic** [code-verified]: this 0.27.1 tree
carries a split-pool rollback for **both** GDN recurrent and causal-conv state —
`_update_states_after_model_execute` (`gpu_model_runner.py:1568-1623`, gated on
`is_hybrid`, not proposer type) computes `num_accepted_tokens` generically, and
the GDN mixer threads `spec_state_indices_tensor` + `num_accepted_tokens` into
`fused_sigmoid_gating_delta_rule_update` (`qwen_gdn_linear_attn.py:1264-1273,
1370-1393`) so a rejected token's state slot is never selected. **Runtime gate
still required before trust** — code-verified ≠ runtime-verified.

**Acceptance-rate monitoring is a mandatory serving gate, not telemetry.** One
wiring bug in the Intel field notes silently collapsed acceptance ~90% → ~0 with
no model change. A collapse toward zero must **fail** the gate, not be reported
as a merely slow speculative run.

**Verify-shape economics on our silicon:** verify turns M=1 into M=spec+1 on the
B70 leg — 47.9 → ~120 µs at M=3 for mean acceptance ~2.7 tokens/step. Strongly
net-positive **iff the B70 leg stays hidden at M=3** — and the trace says the
leg is 66.7% *exposed* at M=1, so this needs its own measurement.

## Hardware plan

```
RTX 5090   dense 6.77 GB + CUDA expert 255 + KV        (+ 5.05 GB if MTP on)
B70 Gen4   0000:15:00.0   experts 0..127     bank dev0        27.84 GiB
B70 Gen3   0000:11:00.0   experts 128..254   bank dev1_owned  27.63 GiB
```

`fractional:2:0.00390625` verified against `FractionalRemotePolicy` → exactly
that split. Placement grammar and policy already exist; only the per-device
runtime plumbing is missing.

**Why 255 remote / 1 local and not 256/0:** the fused CUDA partial masks remote
routes through a real local dummy slot, so
`validate_cuda_dummy_slot_placement` (`partition.py:92-110`) **rejects zero CUDA
experts**. True all-remote needs a separate no-CUDA fast path — price it as
implementation work, do not assume it. One expert across 48 layers costs
~0.26 GB.

**Explicit PCI BDFs are mandatory.** Level Zero enumerates the *Gen3* card as
index 0; that mistake cost 40% for weeks. `select_b70()` already resolves BDF
strings — the gap is only that the C ABI never populates
`ProviderConfig::device_selector`. **Never use `ZE_AFFINITY_MASK` to select two
cards inside one process.**

**Shared-uplink facts:** both B70s hang off `09:00.0` Gen4 x4. Solo 6.24 GB/s
(Gen4) and 2.91 (Gen3); concurrent bulk traffic **degrades to 4.53 GB/s
aggregate**. Doorbell payloads are ~1.7 MB/step total so bandwidth is
irrelevant — but **concurrent two-card dispatch *latency* is the one unmeasured
number in the whole topology**, and it gates the decode projection.

**The shared trunk holds.** With the 88B server resident: 5090 registered H2D
57.7 → 54.8 GB/s (−5%), 2-thread NVMe O_DIRECT 6.39 → 6.22 (−2.5%), B70
doorbell median 87 → 89 µs (+2%), aggregate DRAM-side 61.5 GB/s. Contention on
that trunk is retired as a primary hybrid-mode risk. (The 87 µs tight-loop
doorbell corroborates the 94.3 µs production trace window.)

## Capacity — the honest projection

| | PRO 6000 96 GB | us |
|---|---|---|
| weights, text-only | ~73 GB | 6.77 GB dense + 0.26 local expert |
| + MTP BF16 | ~78 GB | +5.05 GB |
| KV left | ~15 GB (~10 with MTP) | **~19 GB (~13.5 with MTP)** |
| KV tokens @ 13,722 B/tok | ~1.1M | **~1.4M / ~1.0M** |

[INFERENCE from measured inputs.] **Capacity parity, maybe a slight edge — not
the asymmetry that won the 88B's ≥98K cells.** Plan accordingly: the 122B
comparison will be decided by prefill and decode, not by seats.

## RAM — the one hard constraint

The monolithic remote prefill bank is **55.7 GiB**. The registered all-hot path
pins its mmap with `cudaHostRegister`, so those pages are unevictable. On a
59 GiB host that cannot coexist with ~3.5 GB of server RSS beyond the bank,
~9 GB of system working set, and transient headroom.

Prefill touches the **whole** remote bank every pass. Decode hotness ordering
changes where experts live but **cannot reduce prefill bytes** at ≥8K: top-8
routing across thousands of tokens reaches the full expert set. So:

$$T_{pass} \ge \max\!\left(\frac{B_{bank}}{BW_{H2D}}, \frac{B_{cold}}{BW_{NVMe}}\right)$$

| configuration | cold/pass | pass floor | 8K / 32K / 128K transfer floors |
|---|---:|---:|---|
| 35 GB pinned (safe today) | 24.8 GB | 3.95 s | 3.9 / 15.8 / 63.2 s |
| 40 GB pinned (tight) | 19.8 GB | 3.15 s | 3.2 / 12.6 / 50.4 s |
| 44 GB pinned (OOM territory) | 15.8 GB | 2.52 s | 2.5 / 10.1 / 40.3 s |
| **128 GB RAM, all hot** | 0 | **1.03 s** | **1.0 / 4.1 / 16.5 s** |

Transfer floors only; compute is additive.

**Hybrid is VIABLE but 3–4× slower per pass.** Recommendation: **2×64 GB
DDR5-6000. Avoid four-DIMM 96 GB — the AM5 down-clock attacks exactly the host
bandwidth the streamer is buying.**

The strategic read [INFERENCE]: hybrid RAM makes the ≤32K column materially
worse (8K TTFT ~3.5–4 s, i.e. 4–5× behind, versus the 88B's 1.9×), but the
long-context cells may survive — at 128K the ~50 s hybrid transfer floor is
comparable to what a PRO 122B will spend on attention anyway. **The RAM upgrade
buys the short-context column, which is where we already lose.**

## Assets on disk

| artifact | contents | status |
|---|---|---|
| `expert_bank_int4_122b_dev0.bin` | 48 layers, ids 0..127, 27.844 GiB | header verified, size exact |
| `expert_bank_int4_122b_dev1_owned.bin` | 48 layers, ids 128..254, 27.626 GiB | **the one matching 255/1** |
| `expert_bank_int4_122b_dev1.bin` | ids 128..255, 27.844 GiB | **stale** — predates the local-expert decision; 27.8 GiB reclaimable |
| `expert_bank_int4_122b.bin.marlin` | SBMARL01, 256 experts, 55.688 GiB | superset of the 255 remote; harmless (255 maps to −1, never indexed) but wastes 0.22 GiB |
| `experiments/b70_122b_pilot_smoke.py` | 2-layer real-weight B70-vs-CPU oracle | committed |

Gates already passed on the GPTQ extraction: shard bytes → bank planes bit-exact;
bank → fresh shard repack bit-exact; two-layer pilot on B70 silicon vs fp32 CPU
oracle at **rel 4.8e-07, cosine ≈ 1.0**; GPTQ-int4 vs NVFP4 cross-format on
sampled real tensors cosine 0.986–0.995, rel L2 0.10–0.17. That last one
validates the **scale convention**, not model-level quality.

GPTQ contract verified: `quant_method=gptq`, 4-bit symmetric, group 128,
`desc_act=false`, trivial `g_idx`, GPTQModel-v2 `qzeros=0x88888888` (stores the
effective zero point directly). The extractor also supports AutoGPTQ-v1
`0x77777777`. **Confusing the two is an off-by-one zero-point error.** The GPTQ
checkpoint's `dynamic` block excludes attn, shared_expert, mtp and visual — so
it quantizes **only** the routed experts, all 48 layers uniformly.

**The reciprocal trap.** In compressed-tensors NVFP4, stored block scales combine
with `weight_global_scale` using the library's **divisor** convention. ModelOpt's
`weight_scale_2` is the **opposite**. Reusing one decoder for the other corrupts
every expert by a squared global-scale factor.

## Why GPTQ-int4 first, and why NVFP4-native is not obviously better

The bake-off favored NVFP4 at the decode shape (8 pairs: 67.9 µs / 92.8% vs
82.5 / 78.8% = **+18%**; tie at 16; +5% at 32) with zero transcode error. **But
that measured speed only, and two residency facts outrank it:**

1. NVFP4 costs **5,308,416 B/expert/layer** (packed e2m1 + e4m3 g16 scales)
   against int4's 4,866,048. At 128 experts × 48 layers that is **30.4 GiB on a
   32,656 MiB card** versus int4's 27.84 GiB. ~1.5 GiB of headroom before
   staging.
2. **NVFP4-native cannot cover layer 47** — those routed experts are FP8 W8A8,
   not W4A4. GPTQ covers all 48 uniformly.

So GPTQ-int4 is the first-boot bank on merit, not just because it is built. The
mixed source (5090 loads NVFP4, B70s load GPTQ-int4 experts) is an explicit
compromise that **must pass a logprob envelope** — never describe it as lossless,
and never silently relabel GPTQ as NVFP4.

**Never mix an NVFP4 prefill bank with a GPTQ-int4 decode bank inside one
request.** Prefill and decode must execute the same remote expert weights.

If NVFP4-native is later pursued: `src/phase1/extract_experts.py` already
implements the `weight_packed`/`weight_scale`/`weight_global_scale` divisor
decode generically and its docstring already anticipates 122B sizing. It is the
starting point, not `build_b12x_bank.py` (hardcoded 88B geometry, wrong
ModelOpt multiplier convention).

## Code blockers — what must change

| blocker | evidence | required change |
|---|---|---|
| No per-device C ABI load | `b70_capi.h` exposes only `sb_b70_load`; `sb_b70_load` never sets `config.device_selector` | add `sb_b70_load_on_device(..., const char* selector)`; **additive**, keeps `sb_b70_load` source- and ABI-compatible. Zero changes needed in `b70_provider.cpp` — `select_b70()` already handles BDFs |
| One provider | `routed_experts.py:321` `_b70_provider_singleton` | cache one provider per placement B70 index, each with its own bank path and resident IDs |
| One poller | `b70_poller.py:312` `_poller_singleton`, keyed on nothing | key by device index; one provider + SYCL queue + poller per card |
| One route map, one signal pair | `_b70_slot_map`, `_signal_dev`, `_completion_dev` singular | build maps from each bank's `source_expert_ids`; allocate flags/staging/outputs per card. `stream_signal.alloc_host_mapped_flag` is already device-agnostic — just call it twice more |
| **Prefill needs its own slot map** | the 88B gets away with one because its Marlin and doorbell banks share `source_expert_ids` (54..179); the 122B's do not (Marlin 0..255 vs dev0 0..127 / dev1 128..254) | a **third**, prefill-specific map. *Not in earlier planning.* |
| One-card qualification | `validate_int4_hybrid_contract` rejects any remote index tuple other than `(0,)` | validate disjoint per-device ownership and complete union coverage |
| Model/bank format mismatch | `config.py:53-61` declares the 122B routed source as `nvfp4`; first-boot banks are SBINT401 GPTQ | make the mixed source explicit in model metadata or add explicit routed-bank source selection. **Never weaken the bank-format check globally.** |
| Mixed layer-47 allocation | the preemptive allocation hook covers `CompressedTensorsW4A4Nvfp4MoEMethod` and `ModelOptNvFp4FusedMoE`, not the FP8 MoE method layer 47 uses | add gated compact FP8 allocation, or deliberately keep layer-47 ownership compatible with the unmodified loader. Verify load counts and peak VRAM — do not let this surface as an opaque load-time OOM |
| MTP interception hazard | the class-wide routed-expert surgery is keyed on class name with **no prefix filter**, and the MTP block builds its MoE through the same `RoutedExperts` layer | add an explicit `mtp.` exemption **before** loading the draft block, or an unguarded boot strips the draft head's experts |
| `g_*` geometry globals | `b70_provider.cpp:44-53`, file-scope, written by `adopt_int4_bank_geometry` (91-109) | both 122B banks adopt **identical** values (48/3072/1024/256, rest zeroed), so sequential loads are safe. Only a *concurrent* second `load()` races. Correct fix is moving the ten fields into `Impl`; it does not gate first boot |
| Two-context host registration unproven | `prepare_for_device_copy` is context-scoped | register shared staging with both contexts, or allocate per-device staging. Prove it in the standalone smoke |
| `experiments/b70_dual_card_smoke.py` | does not exist | write it |

## Ordered plan

**Phase A — establish the comparison.** Run the stock RTX PRO 6000 with the same
checkpoint and harness. Record *before* tuning our side: weight memory after
load, explicit KV allocation and reported KV-token count, seats at 131K/262K,
max context that boots, C=1 TTFT/ITL at 128-token/8K/32K, whether 262K fits, and
greedy output + per-token logprobs for fixed parity prompts. Preserve raw config,
vLLM version, command line, result JSON.

No checked vendor result provides a relevant single-PRO-6000 122B baseline;
published recipes use two 96 GB cards. **A one-card capacity failure is a
comparison result, not a benchmark failure to hide.**

**Phase B — make the dual-B70 contract explicit.** The ABI selector, provider/
poller/map/flag/staging per card, both signals issued before either wait, sum
both remote partials then add the CUDA partial. Keep every per-device change
opt-in behind the 122B placement: a one-card 88B placement must still
instantiate exactly one provider and one poller.

**Phase C — standalone two-card gate.** Before vLLM loads any 122B weights,
prove: both banks load into the *selected physical* cards; source-ID maps are
disjoint and union to 0..254; wrong-card and duplicate-selector configs fail
closed; both dispatches in flight before either wait; each card's partial
matches an independent CPU dequant oracle; their sum matches the full remote
oracle; sentinel routes for the other card contribute nothing; completion
reset/order correct; static output addresses graph-replay safe; **200 alternating
replays**; shared host staging accessible from both contexts; and **concurrent
two-card dispatch latency median/p95/p99 recorded**. This is the first point at
which dual-card runtime can be called working.

**Phase D — first vLLM boot.** Text-only, MTP off, 255/1, explicit BDFs,
`SHOOTING_BRAKE_BANK_REGISTER=0` (pageable — no RAM dependency), conservative
explicit `--kv-cache-memory` **after** observing the dense footprint, no
GuideLLM. Mandatory evidence: every expected tensor loads exactly once with
layer 47 accounted for; both B70s report the intended device identity and
resident set; peak 5090 and per-B70 memory recorded; **first forward contains no
NaN or Inf**; fixed greedy prompts coherent; per-token logprobs inside a
predeclared envelope against the PRO run; route counters show both B70s and CUDA
expert 255 participating without drops or double counting.

**Named crash class to gate:** a sibling SGLang compressed-tensors loader left
45/60 hybrid-GDN attention layers **uninitialized → 100% NaN** on the 397B
sibling (untested on vLLM). That is not proof vLLM fails, but it makes the
load-count audit and first-forward finite-value probe mandatory. Useful
397B-proven flag: `VLLM_NVFP4_GEMM_BACKEND=cutlass`; driver 590.48+/CUDA 13 ✓.

**Phase E — smoke performance, then matrix.** 128-token ITL probe recording both
B70 clocks/power/utilization/route counts and concurrent leg latency; C=1 TTFT
at 8K and 32K; KV tokens and seats; one 262K feasibility request if capacity
permits; **only then** the same GuideLLM cells and seeds as the PRO baseline.
Report hybrid results **as such** — never compare them to an all-hot estimate
without labeling the memory configuration.

**Phase F — native MTP.** `mtp.` surgery exemption first. Start at
`num_speculative_tokens=1` as a rollback/correctness gate, then 2 for
measurement. Gate on: greedy and logprob equivalence with MTP off; accepted-token
counts and acceptance rate; rejection rollback across **both** GDN and
full-attention layers; no persistent-state corruption after a rejection;
**retokenized ITL**, not just aggregate throughput; output quality under
low-acceptance prompts.

**Phase G — multipliers, only after ordinary serving and MTP are correct.** RAM
upgrade / full pin; native NVFP4 bank (with the two residency caveats above);
calibrated W4A4 prefill via the installed CUTLASS path; B12x private launch only
after an exact-shape roofline; hybrid stream-once/Tier-A with large chunks.
Routing-selective prefill fetch is **not** a valid optimization — the full bank
is touched.

## Hard prohibitions

* Do not call the 122B production-ready before a real two-card smoke and vLLM
  boot pass.
* Do not use `ZE_AFFINITY_MASK` to select two cards in one serving process.
* Do not use enumeration indices in production config; use PCI BDFs.
* Do not introduce oneCCL, B70↔B70 collectives, or peer-memory traffic.
* Do not assume zero CUDA experts works.
* Do not mix an NVFP4 prefill bank with a GPTQ-int4 decode bank in one request.
* Do not claim GPTQ remote experts are the native NVFP4 checkpoint.
* Do not enable MTP before excluding `mtp.` from routed-expert surgery.
* Do not use a quantized MTP head because the main experts are quantized.
* Do not pin the full 55.7 GiB bank on the current 59 GiB host.
* Do not use four or more NVMe readers — throughput regresses past two.
* Do not treat transfer floors, KV estimates, or dual-card latency projections as
  measured serving results.
* Do not start the benchmark matrix before finite-output and logprob gates pass.

---

# 10. Open questions

1. **Where is the PRO 6000 122B baseline running, and how do results reach this
   box?** `~/srswti/benchmarks-vllm/bench-matrix/` has no 122B directory and
   neither ssh host in `~/.ssh/config` accepts connections. Phase A is blocked
   on this. Also worth knowing early: does 122B even *boot* on one 96 GB card
   (73 GB weights text-only, 78 with MTP)?
2. **RAM: order 2×64 GB now, or bring up on hybrid first?** Correctness does not
   care; the headline matrix does.
3. **~~Does the int4 GEMV kernel use SLM?~~ ANSWERED 2026-08-18 — no landmine,
   but graph replay is a ~1% lever, not ~8%.** Capture works
   (`local_accessor`, not the blocked extension; `graph_supported=1`, replay
   numerically clean). The old −28% figure does not reproduce: the real delta
   is ~4 µs at k=4–8 and *nothing* at k=1, and finalize costs ~8 s of boot
   across 32 M-buckets. Re-ranked to second-to-last in §6. **The live question
   this replaces: schedule the dual-B70 route split and the gate_up occupancy
   work instead — they are now the top two decode levers.**
4. **`down2` — land it now?** Twenty minutes, +6–7% on the B70 leg at C≥2, zero
   at C=1. Best folded into whatever change next touches `issue()`.
5. **Marlin roofline against the sm_120 bf16 tensor-core ceiling.** Cheap,
   analytical, undone, and it is the gate on whether short-context prefill has
   any path left at all.
6. **Reclaim `expert_bank_int4_122b_dev1.bin`?** 27.8 GiB of stale artifact
   against 141 GB free, and NVFP4 bank builds would want ~60 GB.

## Protected assets

`~/.cache/huggingface/hub/models--0xSero--Qwen3.5-88B` (164 GB) is the genuine
bf16 base for the 88B — `Qwen3_5MoeForConditionalGeneration`, 48 layers, hidden
3072, 180 experts, geometry identical to the served checkpoint. It gates three
otherwise-blocked levers: rotation requantization, FP6-W6A8, and EAGLE3
draft-head training (`speculators`' `load_verifier_weights()` is
quantization-naive and our served `lm_head` is NVFP4, so verifier weights must
come from the bf16 base). **It was nearly deleted during a disk crunch. Protect
it.**
