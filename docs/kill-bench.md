# Kill Bench

The standing microbenchmark ledger. Every idea that claims it can move an SLO
metric (ITL, TTFT, throughput) earns an entry here BEFORE any production code
is written for it.

## Rules

1. **Every entry states, before running:** the hypothesis with a number
   attached, the exact microbenchmark that tests it, and the kill condition.
   No kill condition = not an entry.
2. **Verdicts are one of:** `WORKED` (number confirmed, graduates to
   production work), `KILLED` (kill condition hit — record why, and what it
   cost to find out), `PARTIAL` (something survived, scope narrowed), or
   `PENDING` / `RUNNING`.
3. **Killed entries are never deleted.** The negative result is the product;
   it is what stops us re-fighting the same idea in three months. (Precedent:
   two MoE bake-offs, the wire codec, MMA multipath — all killed with numbers,
   all still saving us time.)
4. **Learnings append.** If a bench taught us something orthogonal to its
   verdict (an instrument fix, a driver quirk, a new floor), it goes under
   `Learned:` in the entry even if the entry itself died.
5. **Numbers carry provenance.** `[estimate]` until an artifact exists on
   disk, then `[measured-here]` with the artifact path. An entry cannot
   graduate on an `[estimate]`.
6. **Small is fine.** A 3 µs/dispatch saving is a real entry if the
   measurement is real. What is banned is unmeasured hand-waving, in either
   direction — optimism or dismissal.
7. Context for all decode entries: the measured step budget
   (`benchmarks/results/b70_gemv_audit/decode_overlap_trace.json`, C=1,
   11.9–12.35 ms/step): **pool A** = 2.99 ms exposed B70 windows, **pool B**
   = 1.92 ms host/scheduling GPU-idle, **pool C** = 7.45 ms CUDA-busy floor.
   A+B cleared ≈ PRO 6000 ITL parity without speculation.

## Ledger

| # | name | attacks | status |
|---|---|---|---|
| 1 | Decode-step timeline census | pool B (attribution) | PENDING |
| 2 | FULL vs PIECEWISE decode graph A/B | pool B | PENDING |
| 3 | vLLM async/overlapped scheduling A/B | pool B | PENDING |
| 4 | Xe2 device-side host-flag spin probe | pool A+B (doorbell endgame) | **PARTIAL** — latency 2.6–3.1 µs one-way (passes ~7×); sustained host-write visibility dies after ~0.5 ms (kernel still executing) |
| 5 | Packed single-H2D dispatch record | pool A (transport) | **KILLED** — 0.98 µs/dispatch measured vs 2 µs kill line; marginal enqueue is 0.431 µs, not the 2.4 µs assumed |
| 6 | Next-layer route predictability (offline) | pool A (speculative dispatch) | PENDING |
| 7 | FrequencyPolicy hit-rate curves (offline) | placement, cold bytes | **PARTIAL** — skew 1.79×/1.44× clears the 1.3× kill line, but on FOREIGN data (40×256); recompute on a real 88B capture |
| 8 | **RBAR cross-vendor direct P2P** | pool A (return-path hop) | 8b/8c dma-buf KILLED; **8g repriced to ~2% by Bench 4 — HOLD** |
| 9 | 8K TTFT re-attribution | prefill (the missing ~0.3–0.4 s) | PENDING |
| 10 | Marlin roofline vs sm_120 bf16 ceiling | prefill compute verdict | PENDING |
| 11 | 1G-hugepage cudaHostRegister (610 fork) | 122B bank pin time + page tables | PARKED (post-RAM-upgrade, needs re-baseline) |
| 12 | 5090 local-expert budget sweep | pool A (kernel term only) | PENDING |
| 13 | **Poller wakeup path (35 µs/dispatch)** | pool A (the 61 µs fixed wall) | **KILLED** — the 34 µs was a cross-clock artifact; dispatch is ~81% weight-bandwidth-bound, overhead is 15.4 µs and accounted for |
| 14 | **B70 GEMV bandwidth efficiency** | pool A (the term that actually dominates) | **PENDING** — M=1 is latency-bound at 406 GB/s while the same kernel hits 510 at M≥2; closing that is **−0.66 ms/token, −5.7% ITL** |
| 15 | **Dual-B70 concurrent doorbell gate** | pool A (the dual-card split's two named risks) | **WORKED, both formats** — oracle ≤1.9e-6 int4 / ≤7.5e-7 nvfp4, isolation exact, 200 replays clean; dual = **1.23× max-solo**, not 2×; nvfp4 doorbell ~17% faster than int4; 99B monolithic bank needs no per-split rebuilds |
| 16 | **sm_120 W4A4 MoE backend qualification (99B first boot)** | 5090-side CUDA partial | **WORKED via VLLM_CUTLASS** — FlashInfer trtllm-gen tactics hang (25 min @ 100% GPU, measured) or fault (misaligned address) on consumer Blackwell; vLLM's in-tree CUTLASS experts produced the **first correct 99B dual-B70 tokens** |

---

## Bench 1 — decode-step timeline census

- **Hypothesis:** the 1.92 ms/step of GPU-idle between dispatches [measured-here,
  decode_overlap_trace.json] decomposes into nameable causes: vLLM scheduler,
  launch overhead, breakable-graph segment glue, sampler.
- **Method:** existing instruments — `SHOOTING_BRAKE_B70_TRACE_DUMP` ring +
  same-process torch-profiler capture, merged on CLOCK_MONOTONIC. Classify
  every GPU-idle gap ≥ 20 µs over ≥100 decode steps.
- **Kill condition:** none — this is attribution, it cannot fail, it can only
  reprice benches 2–5.
- **Verdict:** —

## Bench 2 — FULL vs PIECEWISE decode graph A/B

- **Hypothesis:** forced PIECEWISE (`_patch_force_piecewise`) leaves per-layer
  glue on replay; Tier 3 dispatch is pure stream ops and was designed to be
  FULL-graph-capturable. If glue is pool B, FULL capture recovers a large
  fraction of 1.92 ms. [estimate]
- **Method:** two boots, same config, decode ITL probe
  (`benchmarks/b70_itl_probe.py`, 4 runs/arm) — one with breakable/PIECEWISE
  as today, one FULL-graph decode.
- **Kill condition:** ITL delta < 0.3 ms, or FULL capture faults on the
  doorbell path (then the GDN-corruption class vllm#51008 must be re-checked
  before any retry).
- **Verdict:** —

## Bench 3 — vLLM async/overlapped scheduling A/B

- **Hypothesis:** scheduler prepares step N+1 while N executes (SGLang-style
  zero-overhead scheduling); some of pool B is scheduler latency, worth
  0.3–1 ms. [estimate]
- **Method:** flag A/B on the production config, 4 ITL probes per arm.
- **Kill condition:** ITL delta < 0.2 ms or instability with the hybrid
  GDN/Mamba path.
- **Verdict:** —

## Bench 4 — Xe2 device-side host-flag spin probe

- **Hypothesis:** a resident B70 kernel spinning on a USM-host word observes a
  host write in < 20 µs with forward progress intact — the prerequisite for
  removing the host poller from the doorbell loop (NIC-interrupt model).
- **Method:** standalone SYCL probe: host writes flag, device spin-kernel
  timestamps visibility; run alongside a busy queue to test starvation;
  check `zeDeviceGetCommandQueueGroupProperties` for cooperative-kernel
  support while there.
- **Kill condition:** visibility > 20 µs, watchdog/starvation trouble, or no
  forward-progress guarantee at the occupancy we need.
- **Verdict: PARTIAL — 2026-08-18. Latency passes by ~7×; sustained
  visibility fails. State it precisely: the kernel keeps RUNNING (it retired
  100,000,001 spin iterations while frozen) — what dies is its VIEW of host
  memory. Not a scheduling/residency failure.**
  Probe: `experiments/b70_hostflag_spin_probe.cpp`.
  - **Recon, all green:** Arc Pro B70, 256 EUs, `usm_host_allocations` yes,
    atomic **system scope** yes, and compute queue group 0 advertises
    **COOPERATIVE_KERNELS** [measured-here].
  - **Visibility PASSES.** Host↔device ping-pong on USM-host flags, single
    host clock (no cross-clock correlation needed): **p50 RTT 5.28–6.24 µs,
    p99 ≤ 6.5 µs, min 4.93 µs** over 200–2000 timed round trips ⇒ **~2.6–3.1
    µs one-way** [measured-here]. The kill line was 20 µs one-way; we sit ~7×
    inside it. The doorbell-notify latency itself is simply not a problem.
  - **Forward progress FAILS — this is the blocker.** The resident kernel's
    view of the host flag freezes **permanently** after only ~0.35–0.7 ms of
    ping-pong (freeze observed at seq 2, 15, 66, 66, 130, 193, 252, 325, 386
    across nine runs). After freezing it polls **100,000,001 consecutive
    times observing no change at all** — not even a host-written ABORT
    sentinel — then exits on its own guard [measured-here, via a
    device-reported exit-reason word].
  - **Ruled out, each by direct experiment; every one still froze:**
    1. cacheable `atomic_ref::load(acquire, system)` — froze at seq 2.
    2. `fetch_add(0)` — reads coherently, but it is a read-modify-WRITE whose
       write-back races the host's next store and silently eats it. A real
       bug, fixed, not the bug.
    3. `compare_exchange_strong` — no write on mismatch, strict single-writer.
    4. uncached loads (`annotated_ptr` +
       `read_hint<cache_control<uncached, L1|L2|L3>>`). Demonstrably changed
       codegen (p50 6.24 → 5.31 µs) and still froze.
    5. cache-line isolation, 256 B stride per flag. False sharing was a real
       bug too — also not the bug.
    6. host-side `clflush` + `sfence` on every store (No-Snoop hypothesis).
    7. single-context residency — responder as the ONLY context on the
       engine, to test GuC timeslicing.
  - **Engine params, for the record:** ccs `job_timeout_ms=5000`,
    `preempt_timeout_us=640000`, `timeslice_duration_us=1000`
    [measured-here]. **None of the three matches the ~0.5 ms freeze window**,
    which is why the obvious scheduling explanations do not hold.
  - **Cost to find out:** one afternoon and one recoverable `ccs` engine
    reset — self-inflicted, a harness timeout tore the USM mapping out from
    under a still-resident kernel. Driver recovered; GPU healthy after.
  - **Consequence, the load-bearing bit:** the persistent-kernel doorbell is
    NOT available on this stack today. Therefore **Bench 8's BAR work is
    currently priced as the ~2% transport lever, NOT the ~25% pool-A lever.**
    Do not spend a kernel module on it on the strength of the pool-A story.
    The dual-B70 split (~1.5 ms, ~12%, no tail risk) outranks it right now.
  - **Follow-ups that could flip this** (all cheap, none attempted yet):
    1. dispatch the responder as a genuine **cooperative kernel** — the queue
       group advertises support and a plain `single_task` may simply carry no
       forward-progress guarantee.
    2. put the request flag in **B70 VRAM** and have the host write it through
       the BAR, so the device polls LOCAL memory. This inverts the direction
       and is exactly the shape Bench 8's push arm needs anyway — one probe
       would serve both benches.
    3. `ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED` / write-combined on the flag
       allocation via raw L0, rather than a load-side hint.
    4. raise `timeslice_duration_us` (root, sysfs-tunable) and re-run.
  - **Learned (external research, 2026-08-19) — reprices the follow-ups:**
    1. **No forward-progress guarantee exists for spinning compute work, in
       ANY mainstream GPU API.** Khronos tracks this as an open gap: given a
       2-workgroup dispatch where one spins waiting on a store from the
       other, no spec text stops the scheduler sleeping the writer forever
       [measured-elsewhere, Khronos tracking issue]. Even CUDA's purpose-built
       cooperative launch does not fully close it unless you control every
       other kernel in the process. **Important scoping note: this is NOT
       what bit us.** Our kernel demonstrably kept executing — it retired
       100,000,001 spin iterations while frozen. We lost memory VISIBILITY,
       not residency. File this as the reason never to design a
       cross-workgroup spin barrier, not as the explanation for Bench 4.
    2. **Cooperative-kernel launch: genuinely unconfirmed.** No spec text or
       compute-runtime source ties `ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_
       COOPERATIVE_KERNELS` to any scheduling or coherence guarantee — the
       flag may only unlock the entry point. Absence of evidence is NOT a no
       here; it has to be dispatched once and measured. Still follow-up #1.
    3. **`BIAS_UNCACHED` is a DEVICE-side cache-hierarchy bias, not a host
       PAT/WC control** — the name misleads by analogy to CPU memory types.
       It is passed on `ze_device_mem_alloc_desc_t` AND
       `ze_host_mem_alloc_desc_t` together for one shared allocation, biasing
       each side independently [measured-elsewhere, Intel test suite usage].
       **This makes follow-up #3 a genuinely different lever from what we
       already tried** — we used a LOAD-side hint (`read_hint<cache_control<
       uncached, L1|L2|L3>>`), never an ALLOCATION-side bias. Untested.
       Caveat carried: TornadoVM measured CACHED up to **4× faster** than
       UNCACHED, framing UNCACHED as intended for stream-once buffers
       [measured-elsewhere]. A doorbell flag re-read in a tight poll is the
       inverse of that assumption, so probe BOTH biases — do not assume
       uncached is correct just because it is fresher.
    4. **Run future probes with the watchdog disarmed up front.** i915 needed
       `enable_hangcheck=0` plus per-engine `preempt_timeout_ms=0` for
       intentionally long-running compute; xe restructured that surface.
       **We already located the xe-native paths** [measured-here]:
       `/sys/class/drm/card2/device/tile0/gt0/engines/ccs/{job_timeout_ms,
       preempt_timeout_us,timeslice_duration_us}` = 5000 / 640000 / 1000.
       Disarm before probing rather than discovering mid-run that a stock
       timeout silently reset the context.
    5. Real xe hang signatures do occur on Battlemage-class silicon ("TLB
       invalidation fence timeout", "Force wake domain failed to ack wake
       -ETIMEDOUT") [measured-elsewhere, Jan 2026 reports, iGPU not discrete].
       Architecturally plausible for us, NOT confirmed as our failure mode.

## Bench 5 — packed single-H2D dispatch record

- **Hypothesis:** fusing the doorbell's three H2D copies (hidden/ids/weights)
  into one contiguous staged record saves 3–6 µs/dispatch ≈ 0.15–0.3 ms/step,
  and shrinks what a future SYCL graph must capture. [estimate]
- **Method:** new arm in `experiments/b70_dispatch_latency.cpp`, clock-pinned,
  at the production 180 µs duty cycle.
- **Kill condition:** < 2 µs/dispatch saving.
- **Verdict: KILLED 2026-08-19.** Probe:
  `experiments/b5_fused_h2d_probe.cpp` (own file rather than an arm in
  `b70_dispatch_latency.cpp` — that harness never touches the provider).
  Run on **device index 1 (Gen4, production card)** at the true M=1 decode
  geometry read out of `b70_provider.cpp:1232-1237` — hidden 6144 B, ids
  32 B, weights 32 B, out 12288 B — 5000 iters, interleaved arms
  [measured-here]:

  | arm | submit mean | full dispatch mean |
  |---|---:|---:|
  | A. 3 separate H2D enqueues (today) | 5.583 µs | 12.916 µs |
  | B. 1 fused H2D enqueue | 4.721 µs | 11.936 µs |

  - **Saving: 0.980 µs/dispatch end-to-end = 0.047 ms/step = 0.38% of a
    12.35 ms step.** Kill condition was < 2 µs/dispatch. **HIT.**
  - **Why the estimate was 5× too high, and the lesson:** Bench 13 step 3
    measured a 3-op submission at 7.29 µs and I divided by three to get
    "2.4 µs per enqueue". Wrong. Most of that is FIXED per-submission-batch
    overhead; the **marginal** cost of one extra enqueue is **0.431 µs**
    [measured-here]. Never derive a marginal cost by dividing an average —
    measure the delta. This mistake briefly had Bench 5 "graduating" on a
    number that was never measured.
  - Two of the three copies are **32 bytes**. Bytes were never the cost, and
    now we know the enqueue is not either. The invasive part of this change
    (a contiguous staging record spanning the Python pinned buffers and the
    C++ device allocation) buys 0.38% and is not worth it.
  - Learned: the same 0.431 µs/enqueue figure prices ANY future "fuse the
    submissions" idea on this stack. Do not re-fight it without a new
    mechanism.

## Bench 6 — next-layer route predictability (offline, zero GPU)

- **Hypothesis:** layer N's routes predict layer N+1's well enough
  (PILOT measured 71.6% recall on a 256-expert MoE [measured-elsewhere]) to
  issue the B70 dispatch speculatively ahead of the true router, sliding the
  window off the critical path; mispredicts pay a small correction dispatch
  (~33 µs at k≈2).
- **Method:** pure offline analysis of an existing
  `SHOOTING_BRAKE_ROUTE_TRACE` capture: P(routes@N+1 | routes@N) per layer,
  plus EMA variants.
- **Kill condition:** recall < 60% on our model — then speculation stays
  MTP/EAGLE-only.
- **Verdict:** —

## Bench 7 — FrequencyPolicy hit-rate curves (offline, zero GPU)

- **Hypothesis:** hotness-ordered placement cuts cold-tier/remote bytes
  2.2–2.6× at zero VRAM cost (simulated on 122B [estimate]; histogram infra
  exists, nothing reads it).
- **Method:** compute achievable hit-rate curves from existing
  `benchmarks/route_stats.csv` production traces before writing any policy.
- **Kill condition:** measured skew on production traffic < 1.3× over
  index-ordered placement.
- **Verdict: PARTIAL — 2026-08-19. Method proven and the kill condition is
  cleared, but on FOREIGN data. Numbers do not transfer; recompute.**
  - Analysis is on `benchmarks/results/route_stats.csv`
    (1.31M tokens, 420M routes, 40 layers × 256 experts, top_k 8).
    **That geometry is neither the 88B (48/180) nor the 122B (48/256)** — see
    the resolved geometry note in Bench 12. Treat every number below as a
    demonstration that the ANALYSIS works, not as our models' hit rates.
  - Skew, hotness-ordered vs index-ordered local tier [measured, foreign]:

    | local tier | hot-ordered | index-ordered | skew |
    |---|---:|---:|---:|
    | 25% of experts | 47.4% of routes | 26.4% | **1.79×** |
    | 50% of experts | 74.1% | 51.5% | **1.44×** |
    | 75% of experts | 91.0% | 75.6% | 1.21× |

    Kill line was < 1.3×. Cleared at the tier sizes that matter (25–50%);
    only the 75% tier falls below, and that tier size implies almost no
    remote traffic anyway.
  - **The cold tail is not dead** — a load-bearing negative for REAP:
    **zero** experts had zero activations, **zero** were below 1% of
    uniform, and only 2.2% were below 10% of uniform. This expert pool is
    fully utilised; pruning removes rarely-used weight, not dead weight.
  - Learned: REAP and FrequencyPolicy are the same measurement with two
    different actions (delete the tail vs demote it), **but they must rank
    on different axes** — REAP on saliency `mean(g_j·‖f_j‖₂)`, placement on
    route frequency. An expert can be high-frequency/low-saliency (REAP
    deletes what placement wants pinned) or low-frequency/high-saliency
    (REAP keeps what placement wants cold). One calibration pass, two
    rankings.
  - **To graduate:** re-run on a real 88B capture
    (`SHOOTING_BRAKE_ROUTE_STATS=1`, 48 layers × 180 experts).

## Bench 8 — RBAR cross-vendor direct write (the peripheral moonshot)

- **Hypothesis:** the B70 can DMA its doorbell result (payload + completion
  flag) DIRECTLY into 5090 VRAM — no host-DRAM bounce, one hop instead of
  two on the return path. Saves 5–15 µs/dispatch ≈ 0.25–0.7 ms/step
  [estimate]. Bonus if it works: `cuStreamWaitValue32` parks on a VRAM flag
  the B70 writes, collapsing completion-detection latency too.
- **Facts already banked:** 5090 BAR1 = 32,768 MiB = full VRAM (ReBAR on,
  [measured-here], vendor recon); BAR1 phys `0x1800000000–0x1fffffffff`,
  B70 BAR2 phys `0x2800000000–0x2fffffffff`, both 32 GiB [measured-here].
- **CORRECTED 2026-08-18 — two banked "facts" were false, both load-bearing:**
  - ~~"one root complex, no PCIe switch"~~ **FALSE** [measured-here,
    `lspci -tv`]. The 5090 is CPU-attached (root port `00:01.1`), but both
    B70s sit behind a *three-level* chipset switch chain
    (`00:02.1 → 03:00.0 → 09:00.0 → 0f:00.0` / `13:00.0`) on the AM5
    800-series chipset. Cross-vendor P2P in EITHER direction must climb that
    chain, cross the shared Gen4 x4 chipset uplink, reach the root complex,
    and be routed back down to `00:01.1`. So: switch-ordering and
    posted-write-collapse concerns are IN scope after all; the uplink is
    shared with USB/SATA/WiFi/2.5GbE; and **whether an AMD client root
    complex routes RC-crossing peer traffic at all is now the single biggest
    unknown in this bench** — it gates BOTH directions, not just one.
  - ~~"B70↔B70 P2P constraints (IOMMU off)"~~ **FALSE.** AMD-Vi is live
    (`ivhd0`); all three GPUs are in `DMA-FQ` *translating* domains
    [measured-here]. Consequence: no raw-physical-address peer DMA is
    expressible at all, which kills any 8f-style "scan the BAR, DMA to the
    offset" arm outright. It does NOT require `iommu=pt`:
    `dma_map_resource` and `nvidia_p2p_dma_map_pages` both build proper IOVA
    mappings under a translating IOMMU by design — no boot-param change, no
    isolation downgrade, and faults stay contained instead of becoming
    stray writes.
- **Method — a ladder, cheapest kill first:**
  - **8a. Prereq recon:** open vs proprietary NVIDIA kernel module
    (dma-buf export needs open modules), kernel lockdown state, IOMMU state,
    L0 external-memory import caps on the B70
    (`zeDeviceGetExternalMemoryProperties`).
  - **8b. Export:** `cuMemGetHandleForAddressRange(…, DMA_BUF_FD)` on a 5090
    allocation → dma-buf fd. Kill point: `CUDA_ERROR_NOT_SUPPORTED`.
  - **8c. Import:** Level Zero `zeMemAllocDevice` with
    `ze_external_memory_import_fd_t` (DMA_BUF) on the B70 context. Kill
    point: import rejected, or importer maps through host bounce (detect via
    bandwidth signature).
  - **8d. Correctness:** B70 CE (`zeCommandListAppendMemoryCopy`, copy-engine
    ordinal) writes a pattern into the imported pointer; CUDA reads back and
    verifies bytes.
  - **8e. Numbers:** B70→5090 write latency at 4 B / 6 KiB / 64 KiB and
    bandwidth at 16 MiB; reverse-read penalty; then the production-shaped
    probe: `cuStreamWaitValue32` parked on a VRAM flag written by the B70,
    end-to-end vs today's host-DRAM doorbell (20.5 µs/dispatch transport,
    [measured-here]).
  - **Fallback arm (8f), only if dma-buf dies:** mmap
    `/sys/bus/pci/devices/0000:01:00.0/resource1`, READ-ONLY pattern-scan to
    locate a magic-filled CUDA buffer's BAR offset, then write only inside
    that located region. Unsupported, fragile across driver reallocations —
    a probe, never a production path.
- **Kill condition:** export or import unsupported on this driver stack; or
  measured end-to-end saving < 3 µs/dispatch; or any driver instability
  (Xid, GPU reset, `xe` wedge — one incident kills it, this box serves).
- **Why nobody ships this:** cross-vendor P2P has no vendor owner. dma-buf is
  the kernel's neutral interconnect — exactly the class of systems trick this
  project exists to exploit.
- **Verdict (arms 8b/8c — the dma-buf export direction): KILLED 2026-08-18.**
  - Probe: `experiments/rbar_dmabuf_probe.cpp`. Recon (8a) was all-green:
    NVIDIA Open Kernel Module 595.84, lockdown `[none]`,
    `CONFIG_DMA_SHARED_BUFFER=y`, `CONFIG_PCI_P2PDMA=y`, B70 BARs readable.
  - Kill: `CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 0` AND the advisory-check
    bypass confirmed it — `cuMemGetHandleForAddressRange(DMA_BUF_FD)` returns
    `CUDA_ERROR_NOT_SUPPORTED (801)` [measured-here]. Driver-policy refusal.
  - **Version-bump ruled out at source:** the RM gate
    (`subdevice_ctrl_gpu_kernel.c`, `INDEX_DMABUF_CAPABILITY`:
    `osDmabufIsSupported() && !APM && !PPC64LE`, where
    `osDmabufIsSupported` = compile-time `CONFIG_DMA_SHARED_BUFFER`) is
    **byte-identical** across `595.71.05-p2p` and `610.57.04-p2p` in
    `vendor/open-gpu-kernel-modules`. Installing the 610 fork would NOT open
    this door. The visible gate should evaluate YES on this kernel, yet the
    driver returns 0/801 — the real gate lives in unpublished code
    (595.84-specific or GeForce SKU policy in RM/libcuda). Unfalsifiable
    from source; dead either way.
  - Learned:
    1. `vendor/open-gpu-kernel-modules` (aikitoria fork, HEAD 610.57.04-p2p)
       proves the underlying MECHANISM on our exact silicon class: forced
       BAR1 P2P between consumer Blackwell cards under `iommu=pt` measures
       **0.36–0.45 µs GPU→GPU write latency vs 14.3 µs without P2P, 43→56
       GB/s uni** (their README, 5090s + PRO 6000 [measured-elsewhere]).
       BAR1-as-DMA-target through a root complex works; only NVIDIA's
       export-side policy blocks OUR pairing in this direction.
    2. The fork's NVIDIA↔NVIDIA P2P patch is dead weight on a 1-CUDA-GPU
       box; do NOT install it for this bench (same conclusion banked so we
       never revisit).
    3. The fork carries an experimental **5000× cudaHostRegister for
       1G-hugepage-backed memory + smaller device page tables** — does not
       engage on our file-backed page-cache banks, but is a named candidate
       for the 122B's 55.7 GiB bank after the RAM upgrade → logged as
       Bench 11.
    4. **The vendor already ships the hard half of the push direction.**
       `nvidia_p2p_get_pages`, `nvidia_p2p_get_pages_persistent`,
       `nvidia_p2p_dma_map_pages` and their unmap/put pairs are
       `NV_EXPORT_SYMBOL` in `nvidia/nv-p2p.c` AND live in the running
       595.84 (`__ksymtab_nvidia_p2p_get_pages [nvidia]` in
       `/proc/kallsyms` [measured-here]). Critically
       `nvidia_p2p_dma_map_pages(struct pci_dev *peer, …)` takes an
       ARBITRARY third-party PCI device, sets `peer_dma_dev.dev = &peer->dev`
       and returns IOVAs valid for that peer — i.e. it will IOMMU-map 5090
       BAR1 pages for the B70's `pci_dev`. This is the GPUDirect RDMA path
       built for NICs (`nvidia-peermem`), unblocked, supported, and
       orthogonal to the dma-buf export gate that killed 8b. It also pins
       the pages and supplies a `free_callback` invalidation contract,
       which removes the relocation hazard the BAR-scan arms had.
    5. **`xe`'s dma-buf importer is peer2peer-capable by construction:**
       `xe_dma_buf_attach_ops = { .allow_peer2peer = true,
       .invalidate_mappings = xe_dma_buf_move_notify }`, consumed by
       `xe_gem_prime_import` via `dma_buf_dynamic_attach`
       [measured-elsewhere, mainline `drm/xe/xe_dma_buf.c`]. So a custom
       exporter handing xe a struct-page-less P2P sg_table is architecturally
       accepted. Observable signature if peer2peer is NOT honoured: xe
       silently migrates the BO to `XE_PL_TT` (host RAM) — detectable by
       placement and by bandwidth, so it can never fool us into calling a
       host bounce a direct path.
    6. **Third independent reason never to install the fork:** the forced-PCIe
       mode it enables trips `if (mem_info->force_pcie) return -ENOTSUPP;`
       at the top of `nvidia_p2p_dma_map_pages` — installing it would
       *disable* the exact API learning 4 depends on.
  - **Arm 8g (LIVE, replaces the export direction — no driver change):**
    flip the direction. Both B70s expose FULL 32 GB VRAM at BAR2
    (`resource2`/`resource2_wc` = 34,359,738,368 B [measured-here]).
    `cuMemHostRegister(…, CU_MEMHOSTREGISTER_IOMEMORY)` is stock, documented
    CUDA API for registering third-party PCIe BARs; the 5090's copy engine
    then PULLS the doorbell result straight from B70 VRAM over its own Gen5
    link — one hop, no host DRAM, no patched driver. Ladder: (1) L0 fills a
    B70 device buffer with an offset-coded magic pattern; (2) root mmaps
    resource2, strided 64 B probe-scan locates the buffer's BAR offset
    (reads only — safe); (3) `cuMemHostRegister(IOMEMORY)` on that window;
    (4) `cuMemcpyAsync` BAR→5090 VRAM, verify bytes; (5) latency 12 KiB +
    bandwidth 32 MiB vs the two-hop path. Kill: register rejects the PFNMAP
    VMA, CE faults on the window, or saving < 3 µs/dispatch. Caveat named
    up front: xe/TTM may relocate BOs; a production design would need a
    pinned/contig allocation contract — probe first, contract later.
- **Verdict (overall Bench 8):** RUNNING — arm 8g.

## Bench 9 — 8K TTFT re-attribution

- **Hypothesis:** measured floors (stream 0.51 s/pass, remote-MoE compute
  0.339 s @ 8K [measured-here]) predict a max() TTFT near ~0.7 s; we measure
  1.08 s. ~0.3–0.4 s is unattributed — chunk boundaries, cold arenas,
  admission, HTTP, or something we have not named.
- **Method:** one profiled 8K prefill on the current run6+registration
  config, same merge technique as the decode trace; bucket the wall time
  until every 100 ms is named.
- **Kill condition:** none — attribution. But if the gap is real overhead,
  it reprices every prefill lever above the kernel level.
- **Verdict:** —

## Bench 10 — Marlin roofline vs sm_120 bf16 tensor ceiling

- **Hypothesis:** `fused_marlin_moe` at our exact shape (E=126, K=3072,
  I=1024, g128) sits within 20% of the sm_120 bf16 tensor-core ceiling — in
  which case short-context prefill is CLOSED for the 88B and all prefill hope
  rides on the 122B's calibrated W4A4.
- **Method:** analytical roofline + one GEMM microbench at the exact shape
  (extend `prefill_floor_bench.py`); compare achieved FLOP/s and bytes/s
  against measured ceilings.
- **Kill condition (of the kernel-hunt, not the bench):** Marlin ≥ 80% of
  ceiling → stop looking for grouped-GEMM replacements, permanently.
- **Verdict:** —

## Bench 12 — 5090 local-expert budget sweep

- **Context that sizes this correctly** (`window_decomposition.json`
  [measured-here]): the doorbell window is **61 µs FIXED (22 device-queue +
  39 host) + ~10.1 µs per expert**, flat across k. Production k ≈ 5.6.
  Moving an expert to the 5090 therefore shaves the KERNEL term only — the
  dispatch still fires and still costs 61 µs, because a layer avoids a
  dispatch only if ALL 8 routes are local (P ≈ 0.43⁸ ≈ 0.1%, negligible).
- **Hypothesis:** pinning the hottest N expert indices to spare 5090 VRAM
  cuts k, and therefore kernel time, at the cost of KV capacity. Sizing from
  measured data: one expert index (48 layers × 4.64 MiB) = **0.218 GiB** of
  5090 VRAM; hotness-ordered placement gives **22% of experts ⇒ 43% of
  routes** [measured-here, `route_stats.csv`, 1.31M tokens / 420M routes].
  At 10 GB local: k 5.6 → 3.2, kernel −24.3 µs/dispatch, × 66.7% exposed ×
  48 dispatches = −0.78 ms, minus ~0.31 ms added to pool C on the 5090
  ⇒ **net ≈ 0.47 ms/step ≈ 3.8%** [estimate].
- **Method:** boot the 88B at several `subset:K:C` local-expert counts,
  4 ITL probes per arm, against the current production placement as control.
  Report ITL vs KV GB surrendered, and confirm the k→kernel slope in situ.
- **Kill condition:** < 0.3 ms/step at 10 GB of local experts, OR the KV
  reduction costs more capacity than the box needs (single-tenant: capacity
  is nearly free, so this arm should usually favour latency).
- **GEOMETRY CONTRADICTION — RESOLVED 2026-08-19 [measured-here]:**
  Authoritative source is each checkpoint's `config.json`, not any doc:

  | model | layers | routed experts | top_k | hidden | moe_int |
  |---|---:|---:|---:|---:|---:|
  | `srswti/axe-superveloce-88b-nvfp4a16` | 48 | **180** | 8 | 3072 | 1024 |
  | `0xSero/Qwen3.5-99B` (REAP) | 48 | **205** | 8 | 3072 | 1024 |
  | `AxionML/Qwen3.5-122B-A10B-NVFP4` | 48 | **256** | 8 | 3072 | 1024 |

  1. `docs/122b.md` was correct (88B = 48 layers / 180 experts) — that file
     is now superseded by `docs/superveloce-99b-dual-b70.md`.
     `benchmarks/README.md`'s `offloaded = 256−C` is **stale** — it predates
     the 88B and should not be used for sizing.
  2. **`benchmarks/results/route_stats.csv` is a FOREIGN capture**: its
     header says `n_layer=40 n_expert=256`, which matches NEITHER the 88B
     (48/180) nor the 122B (48/256). Any hit-rate curve derived from it —
     including the "22% of experts ⇒ 43% of routes" figure quoted above —
     is **NOT VALID for our models** and must be recomputed from a real
     88B capture. The measured 1.79× / 1.44× skew in Bench 7 carries the
     same caveat: the methodology is proven, the numbers are borrowed.
  3. **Dispatches per step = 48, not 16.** `subset:16:8` is a Track-B sweep
     config (`LayerSubsetPolicy(active_layers=16, cuda_per_layer=8)`), NOT
     the run6 production trace. Arithmetic settles it: `b70_window_sum
     4.45 ms ÷ 94.3 µs = 47.2`, and `4.45 ms / 48 = 92.7 µs/dispatch`,
     consistent with the 94.3 µs median window. run6 ran all 48 layers
     B70-active.
- **Still blocked on:** a real 88B route histogram (180 experts, 48 layers)
  before the hit-rate curve can size the VRAM budget. Capture with
  `SHOOTING_BRAKE_ROUTE_STATS=1`. Should also run AFTER Bench 13, since the
  poller fix moves the baseline this would be measured against.
- **Note:** Bench 12 and the dual-B70 split are **substitutes, not
  complements** — both shrink the kernel term, neither touches the 61 µs
  fixed cost. Doing both perfectly still leaves 61 µs × 48 = **2.93 ms/step**,
  which is essentially all of pool A. That wall is Bench 13.
- **Verdict:** —

## Bench 13 — the poller wakeup path (the 61 µs fixed wall)

- **The number** (`window_decomposition.json` [measured-here]): graph replay
  accounts for only **4 µs of the 39 µs host leg**. The remaining
  **~35 µs/dispatch = 1.7 ms/step = 14% of ITL** is poller wakeup + `take()`.
  The source note calls it *"the largest unattributed decode item; the poller
  already spins, so 39 µs is anomalous and unexplained."*
- **Why this is now the top of the board:** Bench 4 measured the hardware
  floor for exactly this class of flag handoff at **~2.6 µs one-way / 5.3 µs
  round trip, p99 ≤ 6.5 µs** [measured-here,
  `experiments/b70_hostflag_spin_probe.cpp`]. Production spends **35 µs**
  doing what the silicon does in ~2.6 µs — a **13× gap**. Unlike Bench 8 this
  needs no BAR trick, no kernel module, and no new hardware path.
- **Hypothesis:** most of the 35 µs is software — wakeup/scheduling latency,
  a lock, a syscall, an allocation, or a queue hop on the completion path —
  not a hardware floor. Recovering even half is ~0.85 ms/step (~7% ITL),
  more than Bench 12 (3.8%), dual-B70 (5.9%) and gate_up (4.3%) individually,
  and unlike those three it attacks the fixed term that caps all of them.
- **Method:** profile the completion path specifically — the poll loop,
  `take()`, and everything between the B70's completion write and the CUDA
  stream resuming. Perf/ftrace on the poller thread; count syscalls, futex
  waits, and allocations per dispatch. Compare against the 5.3 µs
  ping-pong floor from Bench 4 as the target.
- **Kill condition:** the 35 µs turns out to be irreducible hardware
  completion latency (would contradict Bench 4's measurement, so this
  outcome is itself a strong finding), or < 5 µs/dispatch recoverable.
- **Prerequisite it also unblocks:** dual-B70 needs **one poller thread per
  card** — two cards on one calling thread serialize the 39 µs host leg to
  ~78 µs and the win is zero [measured-here]. Fixing the poller path and
  making it per-card are the same piece of work.
- **Verdict: STEP 1 DONE — 2026-08-19. The mechanism is NOT the cost.
  ~34 of the 39 µs is software.** Probe:
  `experiments/b13_doorbell_flag_probe.cpp` (5090 only — no model, no bank,
  no B70; reproduces the exact production handoff:
  `cuStreamWriteValue32(signal)` → host spin → host writes completion →
  `cuStreamWaitValue32` releases, inside a captured+replayed CUDA graph).
  - Measured over 2000 timed replays [measured-here]:

    | leg | p50 | p99 |
    |---|---:|---:|
    | A: graph launch → host observes signal | **3.58 µs** | 4.07 |
    | B: host writes completion → waitValue releases | **1.19 µs** | 1.47 |
    | **A+B full round trip** | **4.79 µs** | 5.38 |

  - **`cuStreamWaitValue32` is ELIMINATED as the suspect.** It was the prime
    hypothesis going in; it costs **1.19 µs**. The GPU front-end is not
    parked slowly, and the completion path is not the problem.
  - **The whole hardware handoff costs 4.79 µs. Production spends ~39 µs.**
    So ≈34 µs/dispatch = **1.63 ms/step ≈ 13% of ITL is software overhead**,
    not a hardware floor. The Bench 13 hypothesis is CONFIRMED.
  - Remaining suspects, in order: (1) the poller's **linear sweep over all
    48 registered layers** (`for (const PollLayer& entry : snapshot)` in
    `b70_capi.cpp`) — each iteration reads a host-mapped flag written by the
    GPU, so each is a coherency miss, and the sweep restarts from layer 0
    after every dispatch; (2) host-side work inside `issue()`/`take()` that
    is not counted as device time; (3) CPU contention between the poller
    thread and the engine thread.
  - **Next step:** instrument the sweep directly — count flags read per
    dispatch and time one full sweep — before optimising anything.
- **B70 Gen3/Gen4 identification [measured-here, 2026-08-19]**, since the
  docs never recorded which BDF is which:

  | L0 index | PCI | H2D | link |
  |---|---|---:|---|
  | 0 | `0000:11:00.0` | 3.23 GB/s | **Gen3 x4** |
  | 1 | `0000:15:00.0` | **6.46 GB/s** | **Gen4 x4** |

  Production is already correct: `serve_88b_128k.sh` sets
  `ZE_AFFINITY_MASK=1` / `SHOOTING_BRAKE_B70_DEVICE=1` → the Gen4 card.
  `sysfs max_link_speed` is useless here — it reports the ASPM-downtrained
  2.5 GT/s x1 for both; measure with `experiments/b70_pcie_bw` instead.
  **CAVEAT this raises:** `window_decomposition.json` records its instrument
  as "Gen3 B70". Its **61 µs fixed / ~10.1 µs-per-expert split may therefore
  be from the SLOW card** and should be re-measured on index 1 before being
  trusted for production sizing. Bench 4's floor was likewise re-measured:
  **4.72 µs RTT / ~2.36 µs one-way on Gen4** (was 5.28 / 2.64 on Gen3).

### Bench 13 — step 2 & 3 results (2026-08-19)

- **Step 2: the 48-flag sweep is INNOCENT.** Probe
  `experiments/b13_sweep_probe.cpp` replicates the poller's steady-state loop
  (48 separately-allocated host-mapped flags, linear sweep, GPU writes one).
  A full sweep costs **0.11 µs** — 2.3 ns/flag [measured-here]. Even the
  restart-at-layer-0 waste averages 0.05 µs/dispatch. **300× too small** to
  explain the gap. Do not optimise the sweep.
- **Step 3: submission is the cost, and it is PER-OPERATION.** Probe
  `experiments/b13_wait_probe.cpp`, run on **device index 1 (Gen4, the
  production card)** at the 12 KiB doorbell shape, 3000 iterations
  [measured-here]:

  | leg | mean | p50 | p90 | p99 |
  |---|---:|---:|---:|---:|
  | blocking `ev.wait()` | **9.46** | 8.35 | 17.87 | 19.58 |
  | spin on SYCL event | **6.37** | 6.33 | 6.76 | 7.23 |
  | spin on `zeEventQueryStatus` | 6.47 | 6.32 | 6.80 | 7.36 |
  | **SUBMIT (enqueue 3 ops)** | **7.29** | 7.27 | 7.36 | 7.97 |
  | submit + spin TOTAL | 13.66 | 13.60 | 14.06 | 14.70 |

- **Two fixes, both measured, both low-risk:**
  1. **Fuse the 3 H2D copies into one** — enqueue costs **~2.4 µs per op**
     (7.29 µs / 3). Fusing 3 → 1 saves ~4.8 µs/dispatch = **0.23 ms/step
     ≈ 1.9% ITL**. *This independently VALIDATES Bench 5*, whose estimate was
     3–6 µs/dispatch — now [measured-here], not [estimate]. Bench 5 should
     graduate.
  2. **Replace `impl_->copy_out->wait_and_throw()` (b70_provider.cpp:1396)
     with a spin.** Saves **3.09 µs/dispatch = 0.15 ms/step ≈ 1.2% ITL**.
     The blocking wait is bimodal — p50 8.35 µs but p90 17.87 µs — because
     Intel's runtime spins briefly then sleeps and waits for an interrupt.
     The poller is a dedicated thread that already spins; it must never
     sleep. Zero downside.
  - Together: **~7.9 µs/dispatch = 0.38 ms/step ≈ 3.1% ITL**, no bank
    rebuild, no placement change, no new hardware path.
- **Methodological finding, applies to every earlier number:** the blocking
  wait's **mean is 58% above its median** (10.26 vs 6.51 in the first run).
  `window_decomposition.json` states "trace-ring medians used throughout".
  Step time sums 48 dispatches, so the **mean** is the correct statistic and
  medians **understate** the real cost. Any per-dispatch median in this
  ledger should be re-read as a lower bound.
- **Still unexplained: ~19 µs of the 34.** Identified so far: ~12 µs
  submission (5 ops × 2.4 µs) + ~3 µs blocking-wait penalty ≈ 15 µs. The
  remainder is NOT reproducible in a standalone probe, which points at
  something the probe cannot see: **CPU contention between the poller thread
  and the vLLM engine thread**, or real work inside `issue()` (route
  remapping, buffer prep) that the synthetic probe omits. Next step must be
  in-situ — perf on the poller thread during a live decode — not another
  standalone probe.

### Bench 13 — FIX LANDED: spin-then-block in take() (2026-08-19)

- **Change:** `src/phase1/b70_provider.cpp`, in `B70Provider::take()`. The
  single `impl_->copy_out->wait_and_throw()` is now preceded by a **bounded
  spin** on `sycl::info::event::command_execution_status`
  (200,000 iterations, ~2 orders of magnitude over the measured p99), then
  falls through to `wait_and_throw()` unchanged.
- **Why it is safe:**
  - `take()` runs on the dedicated native poller thread, which already spins
    in its outer loop (`b70_capi.cpp`). It has no business sleeping.
  - The spin is **bounded**, so a wedged device cannot burn a core forever —
    it falls through to the blocking wait exactly as before.
  - `wait_and_throw()` is still called unconditionally, so asynchronous
    exception surfacing is byte-for-byte unchanged. After a successful spin
    it observes an already-complete event and returns immediately.
  - No API, ABI, or behavioural change visible to any caller.
- **Measured basis** [`experiments/b13_wait_probe.cpp`, Gen4 B70, production
  12 KiB shape, 3000 iters, measured-here]:

  | wait strategy | mean | p50 | p90 | p99 |
  |---|---:|---:|---:|---:|
  | blocking `wait_and_throw()` | **9.46 µs** | 8.35 | 17.87 | 19.58 |
  | spin on SYCL event status | **6.37 µs** | 6.33 | 6.76 | 7.23 |
  | spin on `zeEventQueryStatus` | 6.47 µs | 6.32 | 6.80 | 7.36 |

  **3.09 µs/dispatch → 0.15 ms/step ≈ 1.2% ITL** [estimate for end-to-end;
  the per-dispatch delta is measured, the step-level translation is not yet].
  Note the blocking path is **bimodal** — p50 8.35 vs p90 17.87 — which is
  the interrupt wakeup showing up. Native L0 polling is not better than the
  SYCL query, so there is no reason to drop to L0 here.
- **Build:** `make -C src/phase7` rebuilds `libsb_b70_provider.so` clean with
  the change (icpx needs `/opt/intel/oneapi/2026.1/bin` on PATH; the harness
  shell cannot `source setvars.sh`).
- **NOT YET VERIFIED END-TO-END.** The per-dispatch number is measured; the
  ITL translation is not. `experiments/b70_dispatch_latency.cpp` does not
  link the provider, and the provider's own test
  (`b70_provider_test.cpp`) is a standalone SYCL bank test that never calls
  `issue()`/`take()`. Closing this needs a server boot on the rebuilt `.so`
  and 4 ITL probes against the run6 control (`split:54`, all 48 layers,
  Gen4 B70, `--kv-cache-memory=2900000000`). Until that runs, the 1.2%
  stays `[estimate]` and MUST NOT be quoted as achieved.

### Bench 13 — END-TO-END VERDICT ON THE SPIN FIX: **KILLED** (2026-08-19)

- **It does nothing.** Same build, same boot config, runtime toggle so both
  arms share one binary (`SHOOTING_BRAKE_B70_SPIN_WAIT`), 88B / `split:54` /
  all 48 layers / Gen4 B70 / `--kv-cache-memory=2900000000` /
  `BANK_REGISTER=1`, warmed past the pin-eviction window, PSI settled,
  clocks sustained (act_freq median 2800), 4 ITL runs per arm
  [measured-here, `benchmarks/results/b70_gemv_audit/itl_probe.json`]:

  | arm | ITL mean (excl. first) | TTFT mean | per-run ITL |
  |---|---:|---:|---|
  | `spin OFF` (control) | **11.5098 ms** | 0.2915 s | 11.48 / 11.53 / 11.50 / 11.51 |
  | `spin ON` (fix) | **11.5130 ms** | 0.2900 s | 11.45 / 11.57 / 11.45 / 11.52 |

  **Delta +0.003 ms against a ±0.06 ms run-to-run spread. Noise.**
  Predicted was −0.15 ms (−1.2%). Not observed.
- **Validity checked before concluding:** `SHOOTING_BRAKE_B70_SPIN_WAIT=1`
  confirmed present in the live worker's `/proc/<pid>/environ`, and
  `SHOOTING_BRAKE_B70_LIB` pointed at the rebuilt `.so` (built 00:27, booted
  00:33). Both arms genuinely ran different code.
- **Why the probe over-predicted — the transferable lesson.**
  `b13_wait_probe` submitted work and waited on it IMMEDIATELY, which is the
  timing pattern most likely to land in the runtime's sleep path. Production
  interleaves `issue()`, trace/counter bookkeeping, then `take()`, and the
  blocking wait never pays the wakeup it paid in the probe. **A synthetic
  probe's timing pattern is part of what it measures.** The 3.09 µs was real
  for the probe and irrelevant to production.
- **Disposition:** code kept, flag defaults OFF (kill-bench rule 3 — the
  negative result is the product, and this stops the idea returning). Set
  `SHOOTING_BRAKE_B70_SPIN_WAIT=1` to re-enable and re-measure.
- **Also banked: a clean, unprofiled 88B baseline.** ITL **11.51 ms**,
  TTFT **0.2915 s** at `split:54`. The 12.35 ms/step in
  `decode_overlap_trace.json` carries torch-profiler overhead; **11.51 ms is
  the honest decode number** and every future % should be quoted against it,
  not against 12.35.
- **What survives:** the ~34 µs/dispatch of software overhead is still real
  and still unexplained — Bench 13 step 1 measured the mechanism at 4.79 µs
  against production's ~39 µs. Two candidate fixes are now dead (fused H2D,
  spin wait), which means the cost is in neither submission count nor wait
  strategy. Remaining hypothesis, and it is now the ONLY one standing: **CPU
  contention between the poller thread and the vLLM engine thread.** That
  cannot be probed synthetically — it needs `perf` on the live poller during
  decode. Next step.

### Bench 13 — FINAL VERDICT: **KILLED** 2026-08-19. The 34 µs does not exist.

- **The claim was an artifact.** `host_remainder = wall − device_total`
  subtracts a copy-engine GPU timestamp from a host `CLOCK_MONOTONIC` span —
  two unrelated clocks. `b70_capi.cpp:129-131` warns about exactly this
  ("must not be used for decomposition without first establishing a common
  timebase") and the `total_mean_us` docstring repeats it. We built a whole
  bench on that subtraction anyway.
- **What the dispatch actually is**, measured on ONE clock in production
  (`SHOOTING_BRAKE_B70_TRACE_DUMP`, 49,344 M=1 dispatches, no profiling
  perturbation) [measured-here]:

  | term | value |
  |---|---:|
  | weights read per M=1 dispatch (k=5.6 remote × 4.866 MB) | **27.25 MB** |
  | achieved B70 read bandwidth at `split_M1`, sustained clocks | 406.08 GB/s |
  | ⇒ kernel time | **67.10 µs** |
  | measured service, p50, host clock | **82.54 µs** |
  | ⇒ overhead | **15.44 µs** |
  | independently measured floor (`b5_fused_h2d_probe`) | 12.92 µs |
  | **unexplained** | **2.52 µs** |

  **The B70 decode dispatch is ~81% memory-bandwidth-bound on expert
  weights.** The overhead is ~15 µs and it is already accounted for.
- **The model predicts, it was not fitted.** Using the single overhead
  constant from M=1, the implied distinct-expert count at other batch sizes
  falls out correctly, with route overlap rising monotonically exactly as
  real MoE routing does:

  | M | measured | implied distinct experts | of routes | overlap |
  |---|---:|---:|---:|---:|
  | 2 | 122.02 µs | 8.9 | 11.2 | 21% |
  | 4 | 210.81 µs | 16.3 | 22.4 | 27% |
  | 8 | 347.56 µs | 27.7 | 44.8 | 38% |

- **Every candidate cause was eliminated by direct measurement before the
  premise itself fell:** `cuStreamWaitValue32` 1.19 µs; 48-flag sweep
  0.11 µs; marginal enqueue 0.431 µs; blocking-vs-spin 0.00 ms end-to-end;
  CPU contention 0.005% run-queue wait on the poller thread (tid pegged at
  100.2%, 3.5 ms rq-wait over 67 s of CPU, 1,811 slices — it owns a core).
  Five clean negatives, then the sixth finding was that the target was a
  measurement error.
- **Rule to carry forward:** never subtract a device profiling timestamp
  from a host clock. Fit against a physical model (bytes ÷ bandwidth)
  instead — it both explains the number and predicts the neighbours.
- **A correct baseline was banked on the way through:** 88B at `split:54`,
  Gen4 B70, unprofiled — **ITL 11.51 ms, TTFT 0.2915 s**. Step reconstructs
  as 48 × (90.4 µs service + 151.8 µs gap) = 11.63 ms vs 11.51 measured.
  The 12.35 ms in `decode_overlap_trace.json` carries profiler overhead;
  quote 11.51 ms.

## Bench 14 — B70 GEMV bandwidth efficiency (the real pool-A lever)

- **Hypothesis: the M=1 decode kernel is LATENCY-bound, not bandwidth-bound,
  and the fix is to make M=1 behave like M=2.** The tell is that achieved
  bandwidth climbs with batch size and then plateaus
  [measured-elsewhere-in-repo, `gemv_bw_audit.json` 2026-08-17, sustained
  clocks, incompressible hash-filled bank, 31 rotating disjoint route sets
  to defeat L2 — this data PREDATES the 2026-08-19 session and nothing we
  changed affects it]:

  | variant | M=1 | M=2 | M=4 |
  |---|---:|---:|---:|
  | `split` | **406.1** | 471.6 | 505.2 |
  | `split_down2` | 386.3 | **509.6** | 510.2 |

  Rising-then-flat is the classic signature of too few memory requests in
  flight — at M=1 there is one row across ~5.6 experts, not enough
  concurrency to keep the memory system saturated.
- **CORRECTION to the first draft of this entry:** it compared 406 against
  `membw.read_gbps` = 599.26 GB/s and claimed 9% ITL. **That is the wrong
  reference.** 599 GB/s is a pure streaming read of 1 GiB with no dequant
  and no math; this kernel cannot reach it. The kernel's own realistic
  ceiling is the **~510 GB/s it already achieves at M≥2** (85% of pure
  read — respectable for int4 dequant + MAC).

  | target | kernel | per token | ITL |
  |---|---:|---:|---:|
  | today, 406 GB/s | 67.10 µs | — | — |
  | **its own plateau, 510 GB/s** | **53.4 µs** | **−0.66 ms** | **−5.7%** |
  | pure-read 599 GB/s | 45.5 µs | −1.04 ms | −9.0% *(not reachable)* |

  **Realistic target: −5.7% ITL.** Quote that, not 9%.
- **Why this is credible where Bench 13 was not:** it attacks a term
  *measured to dominate* (67.1 of 82.5 µs per dispatch), against a target
  the same kernel already demonstrates at a different batch size — not a
  residual from a cross-clock subtraction, and not an extrapolation.
- **Method:** profile `int4_moe_split` at the exact production shape
  (E=126, K=3072, I=1024, g128, top_k 8, M=1) for occupancy, memory
  coalescing and whether the 32% shortfall is latency-bound (needs more
  concurrent expert reads) or access-pattern-bound. `down2_M1` already
  measures 386 GB/s vs `split_M1` 406 GB/s, so the two variants bracket the
  problem. Related and already logged: the `gate_up_occupancy` note in
  `window_decomposition.json` estimates 1.35× on the kernel ≈ −4.3% ITL.
- **Kill condition:** the shortfall is inherent to the access pattern (e.g.
  scattered per-expert reads cannot coalesce further) and < 5% of the gap is
  recoverable.
- **Verdict:** —

## Bench 15 — dual-B70 concurrent doorbell gate (standalone, no vLLM)

- **Hypothesis:** two per-card providers/pollers in one process can serve
  the production doorbell shape (both signals before either wait) with
  per-card partials matching the CPU int4 oracle, no cross-card leak, and
  a dual-graph replay near max(solo) rather than sum(solo). The handoff
  named concurrent two-card doorbell latency and two-card graph replay as
  the two unmeasured risks of the dual split.
- **Method:** `experiments/b70_dual_card_smoke.py` — 122B dev0/dev1 split
  banks (the gate is model-agnostic plumbing; same 3072/1024/48L geometry
  as the 88B/99B), dev0 bank → `0000:15:00.0` (Gen4), dev1 bank →
  `0000:11:00.0` (Gen3), pollers pinned to cores 5/6, M=1 top-8 fixtures
  straddling the cards 4+4 and 5+3, plus one-card-only sentinel fixtures
  in each direction with NaN-poisoned outputs. One CUDA graph rings both
  doorbells, then waits both completions.
- **Kill condition:** any partial > 5e-5 peak-relative vs the CPU oracle;
  any nonzero output on an unrouted card; poller errors; or dual replay
  ≥ 1.8× max(solo) (would mean the doorbells serialize and the split's
  win is fiction).
- **Verdict: WORKED — 2026-08-19** [measured-here,
  `benchmarks/results/b70_gemv_audit/dual_card_smoke.json`]:
  - Correctness: peak-rel ≤ **1.9e-6** on every per-card partial and every
    sum, across A→B→A and both sentinels. 25× inside the kill line.
  - Isolation: unrouted card's output **exactly zero**, both directions,
    through NaN-poisoned buffers — the -1 skip ABI holds across cards.
  - Stress: 200 alternating replays, **0 poller errors**, worst rel
    1.85e-6. No flag-ordering race at this depth.
  - **Concurrent latency, the first measurement of the open number:**
    solo Gen4 73.7 µs, solo Gen3 72.8 µs, dual **90.3 µs = 1.23×
    max-solo** (0.62× sum-solo). Two doorbells in flight cost +16.6 µs
    over one, not +73 µs. The shared Gen4 x4 chipset uplink does not
    serialize doorbell-scale traffic.
  - The +23% concurrency penalty is REAL and unattributed. Candidates, in
    order: all six D2H staging copies share one CUDA stream (production
    has the same shape, so this is representative, not an artifact);
    uplink arbitration on the ~18 KiB payloads. Attributing it is a
    follow-up only if the first vLLM dual-card ITL lands short of the
    corrected prediction.
  - **Hazards logged, not blockers:**
    1. One transient `std::bad_alloc` on the SECOND provider load
       (~28 GB device USM); identical retry succeeded, dev1-alone always
       succeeds. Suspected host commit/page-cache pressure right after
       the first 28 GB stream — worth a bounded retry in bring-up
       tooling. NOT reproduced since.
    2. Stress-phase per-poller service means are asymmetric (dev0 ~2.4×
       dev1: 140.6 vs 85.7 µs int4, 147.8 vs 58.4 µs nvfp4) in a way
       route counts do not explain, REPRODUCIBLE across formats, and the
       dev0 mean exceeds the whole dual graph window — so it is a
       measurement-window artifact or a wait-path effect, not dispatch
       cost. The sleep-path hypothesis was tested and is DEAD:
       `SHOOTING_BRAKE_B70_SPIN_WAIT=1` changed nothing (dual 78.6 vs
       79.0 µs, asymmetry intact) — the killed Bench 13 spin fix stays
       killed even in the dual arrival pattern. Unattributed; per-card
       telemetry now exists precisely to watch this in situ. It does not
       price the decode window.
  - **NVFP4 leg — the 99B bank, same gate, 2026-08-19** [measured-here,
    `dual_card_smoke_nvfp4.json`]. One monolithic SBEXP001 bank
    (`expert_bank_99b.bin`, 48.65 GiB, byte-exact vs checkpoint via
    `src/phase1/validate_99b_bank.py` — 16/16 sampled records EXACT),
    resident split 103/102, after generalizing the provider's NVFP4
    geometry (256-expert hardcode → adopted from header; upload strides
    likewise):
    - Correctness ≤ **7.5e-7** vs vLLM's own `dequantize_to_dtype` oracle
      (global scale applied explicitly as the bank's stored multiplier —
      the reciprocal trap cannot silently flip). Isolation exact, 200
      replays clean.
    - **NVFP4 beats int4 end-to-end through the doorbell:** solo 61.4 /
      63.5 µs vs int4's 73.7 / 72.8 µs at the identical 4-route fixture —
      ~17%, consistent with the bake-off's +18% kernel-level figure.
    - Dual 79.0 µs = **1.246× max-solo** — the ~+23% concurrency penalty
      is format-independent, pointing at the shared stream/uplink, not
      the kernel.
    - Resident-subset loads off one mmap: 22–28 s/card. **No per-split
      bank rebuilds exist in the NVFP4 world** — the half-day-per-arm
      bank tax was an SBINT401-ism. Bench 12's sweep just got cheap.
  - **What this does NOT establish:** normal full-vLLM graph capture,
    graph-mode KV capacity, or any serving ITL claim. The production 99B
    placement is ~3.98 routes/card at M=1, not the old 2.8 estimate.

## Bench 16 — sm_120 W4A4 MoE backend qualification (the 99B first boot)

- **Not a planned bench** — a bring-up record with measured negatives worth
  never re-fighting. Goal was the first 99B dual-B70 boot gate: greedy
  tokens, finite logprobs, doorbells on both cards, eager mode.
- **Checkpoint export inconsistencies found and patched in the plugin**
  (report upstream; the export pipeline should be fixed at the source):
  1. `srswti/axe-superveloce-99b-nvfp4` declares `Qwen3_5MoeForCausalLM`
     but names tensors `model.language_model.*` (the vision-wrapped
     nesting). Patch: prefix rewrite composed into the CausalLM loader
     (`_patch_nested_causallm_naming`) — a no-op for correct checkpoints.
  2. Its `quantization_config.ignore` carries the same nested prefix on
     348/349 entries, so every BF16 module (GDN, routers,
     shared_expert_gate) was constructed QUANTIZED and rejected the plain
     `weight` tensors. Patch: prefix normalization at
     `CompressedTensorsConfig.from_config` (`_patch_nested_quant_ignore`).
  3. The checkpoint ships **zero `mtp.*` tensors** despite
     `mtp_num_hidden_layers: 1` — MTP for the 99B requires a side-loaded
     head or stays off.
- **The sm_120 kernel findings, each [measured-here] on 2026-08-19:**
  1. **FlashInfer `FLASHINFER_CUTLASS` MoE autotuning wedges this exact
     configuration:** the first `trtllm::fused_moe::gemm1` tuner profile ran
     **25+ minutes at 100% GPU utilization without completing**
     (py-spy: engine parked in `stream.synchronize()` inside
     `autotuner.choose_one`; log frozen at 0/21 profiles). The operator name
     does **not** prove an sm_100-only binary: FlashInfer 0.6.16.post3 has a
     distinct `gen_cutlass_fused_moe_sm120_module`. The failure is an
     observed tactic/autotuner bug on this shape, not a justified
     architecture attribution.
  2. **`VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS="trtllm::fused_moe::gemm1,
     trtllm::fused_moe::gemm2"` bypasses the sweep** — full autotune then
     completes in **19 s** (84 configs, cached on disk). But that backend's
     untuned heuristic tactic faults with `CUDA misaligned address` ~40
     layers into the profile pass. `FLASHINFER_CUTLASS` is therefore OUT for
     this model/shape on vLLM 0.27.1 + FlashInfer 0.6.16.post3.
  3. This does **not** mean every FlashInfer NVFP4 backend is broken.
     `FLASHINFER_CUTEDSL` is source-gated to SM100/SM103 and is not a 5090
     candidate. `FLASHINFER_B12X` is a separate SM120/SM121 CuTe-DSL kernel
     with CUDA-graph workspaces; it is installed but unqualified against the
     plugin's one-expert compact allocation and must be an isolated later A/B.
  4. **`moe_backend="cutlass"` (vLLM in-tree `CutlassExpertsFp4`) WORKS**
     and remains the serving baseline (`benchmarks/serve_99b_dual.sh`).
  5. **Our latent dtype bug, exposed by the stricter consumer:**
     `compact_cuda_routes` widened router ids to int64 via its `long`
     remap; Marlin (88B) and FlashInfer tolerated it, vLLM CUTLASS's
     stable-ABI wrapper type-checks `Int` and threw. Fixed at the source —
     the remap now returns ids in the router's own dtype.
- **First boot PASSED (eager, dual-B70, 2026-08-19):**
  - Greedy: "The integer after 41 is" → " 42.\n\nTo find the next
    integer after 41, simply"; "Water is composed of hydrogen and" →
    " oxygen in a fixed mass ratio of 1:8…". Logprobs finite.
  - Doorbell trace: the surviving file contains **3,599 entries from one
    poller** (M=256 prefill chunks, M=7 profile traffic, M=2 decode across all
    48 layers). Both pollers wrote the same path and atomically replaced it,
    so this artifact is not a merged two-card dispatch count.
  - Surgery: 48/48 layers compacted (10.4 GiB CUDA weights). Eager-mode KV
    sized **14.04 GiB = 341,723 tokens** at gpu_util 0.85 with **0 GiB
    CUDA-graph memory**; graph-mode capacity remains unmeasured.
  - Instrument note: FlashInfer's cold autotune + the trtllm hang is what
    the earlier "40-minute boot" was doing. Autotune results now cached;
    subsequent boots skip it.
- **NOT yet established:** graph-mode boot, 99B ITL/TTFT/throughput,
  graph-mode KV capacity, per-device production traces, or long-context
  stability. Next steps are in `docs/superveloce-99b-dual-b70.md`.

## Bench 17 — 99B graph-mode characterisation campaign (2026-08-19)

- **Not a single bench** — the measurement pass taken *before* choosing a
  prefill lever, because two of the three levers we had ranked turned out to
  be artifacts. Instruments: `benchmarks/b70_prefill_cost.py`,
  `benchmarks/b70_matrix_probe.py`, `experiments/b70_mem_topology_probe.cpp`.
  Artifacts: `benchmarks/results/b70_gemv_audit/99b_matrix_even.json`,
  `99b_matrix_swap.json`, `99b_matrix_corpus.json`, `99b_prefill_cost.json`,
  `99b_prefill_ladder.json`.
- Arm for every number below: `benchmarks/serve_99b_dual.sh` as written
  (graph `doorbell` arm, `--moe-backend cutlass`, MML 32768, MNBT 2048,
  MNS 4, `MAX_BATCH=256`, placement `fractional:2:1/205`), unless stated.

### 1. Prefill is linear and B70-bound

| prompt tokens | TTFT | µs/token |
|---|---|---|
| 1,027 | 1.884 s | 1,835 |
| 4,081 | 7.478 s | 1,833 |
| 8,173 | 14.992 s | 1,834 |
| 16,414 | 30.315 s | 1,847 |
| 30,020 | 55.885 s | 1,862 |
| 31,290 | 57.3 s | 1,831 |

Linear across a 60x range: **no attention superlinearity up to 31K**. B70
service is **86-92%** of TTFT. Chunked-dispatch service fits
`service(M) = a + b*M` to within **0.03%** at M=256, with a = 43-121 µs and
b = 27.1-34.8 µs/token.

### 2. `B70_MAX_BATCH` is dead — measured, not argued

Fixed cost is **1.34%** of service at M=256, so the whole chunk-count term is
worth at most that. Modelled 8K B70 service: 13.874 s at C=256 vs 13.712 s at
C=2048 = **-1.17%**. Retracted as a lever.

### 3. Decode is flat in context, and the gap is the story

ITL at C=1 across 1K -> 30K context: 14.30 / 14.55 / 13.97 / 14.00 / 14.06 /
14.20 / 14.39 ms. **KV size costs decode nothing.** Clean 4-run probe:
**14.23 ms** mean (88B production: 11.51 ms). Of that, the 48-layer B70 sweep
is **12.86-12.99 ms** and the inter-dispatch gap is **207-213 µs**; at M=1
the two cards are symmetric (**63.8 vs 64.1 µs**).

### 4. Concurrency saturates almost immediately

| ctx | C=1 | C=2 | C=4 |
|---|---|---|---|
| 1K TTFT / ITL / agg tok/s | 1.88 s / 14.55 ms / 21.8 | 3.54 s / 17.32 ms / 26.5 | 7.39 s / 22.08 ms / 28.6 |
| 8K TTFT / ITL / agg tok/s | 14.99 s / 14.06 ms / 4.03 | 26.25 s / 17.66 ms / 4.05 | 41.02 s / 22.84 ms / 4.14 |

At 8K aggregate throughput is **flat** (4.03 -> 4.14): prefill owns the box.

### 5. Transport, measured per card (`b70_mem_topology_probe`)

| card | H2D | D2H | clock under load | power |
|---|---|---|---|---|
| `0000:15:00.0` | **6.470 GB/s** | 6.583 | 2,633 MHz | 218.6 W |
| `0000:11:00.0` | **3.229 GB/s** | 3.291 | 2,550 MHz | 229.0 W |

Ratio **2.004x** — Gen4 x4 vs Gen3 x4. Concurrent both-card H2D aggregates
**6.440 GB/s**, below the 9.70 sum: the shared uplink contends. sysfs is
useless for this — both cards report `2.5 GT/s x1` at idle *and* under
sustained load, and `max_link_speed` also reads Gen1 x1.

### 6. Card vs expert-range: a 2x2 that changed the plan

Swapping `SHOOTING_BRAKE_B70_SELECTORS` moves an expert range to the other
card. Per-token cost at M=256:

| | on Gen4 | on Gen3 | card penalty |
|---|---|---|---|
| experts [1..102] | 27.15 µs | 29.54 µs | 1.088x |
| experts [103..204] | 33.03 µs | 35.34 µs | 1.070x |
| **range penalty** | **1.216x** | **1.196x** | |

PCIe generation costs **~8%** (transport 6.9% + clock 3.2%). The ~20% range
effect looked like the bigger lever. It is not real.

### 7. **Synthetic prompts route anti-correlated to real text**

`SHOOTING_BRAKE_ROUTE_TRACE` (eager only — see §9), 10,139 tokens of this
repo's docs vs 11,023 tokens of random tokens, 3.9M routes each:

| | dev0 [1..102] | dev1 [103..204] | high/low |
|---|---|---|---|
| natural prose | **51.87%** | 48.13% | 0.928 |
| synthetic random | 44.73% | **55.27%** | 1.236 |

Per-expert correlation between the two workloads: **-0.21**. All 205 experts
are used in both; natural max/mean is only **1.63x**, top-10 share 7.2%.

Confirmed end-to-end in graph mode at matched token counts:

| | dev0 µs/tok | dev1 µs/tok | imbalance | µs/token TTFT |
|---|---|---|---|---|
| natural prose | 31.95 | 31.95 | **1.000x** | 1,787 |
| synthetic random | 27.3 | 35.3 | 1.29x | 1,847 |

**On real text the even 102/102 split is already balanced to 1.000x.** The
gating card costs 9.2% less on prose (predicted 8.8%); end-to-end TTFT is
3.3% better. Modelled alternatives on prose: best single global split shift
**+0.03%**; a split tuned on synthetic prompts applied to prose is
**-6.43%**, i.e. actively harmful. Per-layer perfect balance would be
**+7.80%**, but the provider takes one resident expert list for *all* layers
(`resident_set_shared_across_layers`), so per-layer assignment is not
expressible without a provider/ABI change.

**Kill condition met for the asymmetric-split lever.** It was an artifact of
benchmarking with random tokens. `b70_matrix_probe.py --corpus` now exists so
no per-card claim is made on synthetic prompts again.

### 8. Device residency costs host RAM 1:1 — the structural wall

`b70_mem_topology_probe --alloc-gib 24`, server down:

| | card `11:00.0` | card `15:00.0` |
|---|---|---|
| +24 GiB device USM | **-23.65 GiB host** | **-24.09 GiB host** |
| after free | +0.43 GiB | -0.02 GiB |

Unchanged by `NEOReadDebugKeys=1` with `UseKmdMigration=0`,
`EnableDeviceUsmAllocationPool=0`, `ForceLocalMemoryAccessMode=0`,
`EnableRecoverablePageFaults=0`, `EnableBOChunking=0`. BAR2 is already
**32 GiB** (ReBAR on), so it is not a small-BAR workaround.

Boot instrumentation matches: MemAvailable goes 50.22 -> 29.38 -> 3.78 GiB in
two provider-load steps, **47.44 GiB total**, against 2 x 24.34 GiB of
resident experts. Process RSS is 1.2 GiB and page cache 1.3 GiB, so it is
neither.

**Consequence:** a host-resident prefill bank (Marlin 44.4 GiB, NVFP4 v2
48.4 GiB, W4A8 v3 56.5 GiB at the 99B's 204x48 geometry) cannot coexist with
48.7 GiB of residency shadow on a 59.44 GiB box. The 5090-side streaming
prefill category is **closed here** until B70 residency shrinks. It is also
why the box swaps: a fresh engine had 1.07 GiB swapped within minutes, and a
4.5 h-old server measured 3.8% slower (ITL 14.79 vs 14.23 ms).

### 9. Two instrument facts worth not rediscovering

- **`SHOOTING_BRAKE_ROUTE_TRACE` is not CUDA-graph-safe.** It stages
  `topk_ids` D2H inside the routed-expert forward; with it set, capture dies
  with `cudaErrorStreamCaptureUnsupported` before the engine serves. Route
  traces require `--enforce-eager` and no `EXPECT_ARM`. `serve_99b_dual.sh`
  now unsets it defensively — a stray export from a calibration run killed a
  boot exactly this way.
- The native trace clock is `CLOCK_MONOTONIC` and agrees with Python's
  `time.monotonic_ns()`, so cells can be attributed to dispatch windows
  exactly rather than by counting.

### 10. Prefix caching is off, and cannot be turned on

`enable_prefix_caching=False` in the engine config, because the model is
hybrid: **36 `linear_attention` + 12 `full_attention` layers** (full attention
every 4th). Recurrent state is not prefix-cacheable, so vLLM disables the
feature wholesale. Measured: the identical 5,524-token prompt three times
costs 9.788 / 9.729 / 9.723 s (**0.994x, 0.993x**), a superset of a cached
half costs 0.999x of cold, and the server reports `Prefix cache hit rate:
0.0%` throughout.

**Every request pays full prefill, permanently.** At ~1,790 µs/token an 8K
system prompt costs ~14.5 s on *every* call. This raises the value of prefill
work and removes the usual "amortise it with caching" escape.
