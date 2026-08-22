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
    `src/phase1/validate_expert_bank.py`, then named
    `validate_99b_bank.py` — 16/16 sampled records EXACT),
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

### 11. Long context to 110K (MML 131072, prose+code corpus)

| prompt tokens | TTFT | µs/token | ITL |
|---|---|---|---|
| 24,614 | 43.8 s | 1,780 | 14.69 ms |
| 33,931 | 60.7 s | 1,788 | 14.79 ms |
| 67,738 | 123.4 s | 1,822 | 15.45 ms |
| **110,372** | **206.0 s** | 1,866 | **16.13 ms** |

KV pool 14.1 GiB, **max concurrency 4.44x for 131,072-token requests**
(~582K KV tokens). Prefill µs/token rises only **4.8%** over a 4.5x token
range, so the quadratic attention term is ~5.9% of prefill even at 110K —
**MoE dispatch dominates at every context length we can serve.** Decode is
*not* flat here: ITL climbs 14.69 -> 16.13 ms (**+9.8%**), the first
context-dependent decode cost measured on this rig.

Long-context concurrency (prose, out=64): C=1 0.72 -> C=2 1.45 -> C=4 2.14
aggregate tok/s. Unlike the 8K surface (flat 4.03 -> 4.14), concurrency does
buy throughput once prefill chunks from several requests pack the 2048-token
MNBT budget.

A 110K request costs **206 s of TTFT with no prefix cache to amortise it**
(§10). That is the production headline, not decode.

### 12. Decode is 5090-bound, and the 88B regression is the gap

The trace reconstructs ITL exactly: `48 x 64 µs service + 47 x 211 µs gap =
12.99 ms`, matching the measured 48-layer sweep p50 of 12.99 ms, with
1.24 ms outside the sweep (embed, lm_head, sample, scheduler).

| component | ms | share of 14.23 ms ITL |
|---|---|---|
| B70 dispatch service | 3.07 | **21.6%** |
| 5090-side inter-dispatch gap | 9.92 | **69.7%** |
| non-sweep remainder | 1.24 | 8.7% |

**An infinitely fast B70 buys 21.6%**, landing at 11.16 ms — i.e. roughly the
88B's 11.51 ms. Chasing the doorbell is chasing a fifth of decode.

The 88B runs **one** lane at a 142 µs gap; the 99B runs **two** at 211 µs.
That 69 µs/layer excess is 3.24 ms over 47 layers, which is **119% of the
observed 2.72 ms ITL regression** — the entire decode gap between the models
is accounted for by inter-dispatch time, and the 99B is marginally faster
than the 88B everywhere else. `[INFERENCE]` that the excess is the second
lane's issue/wait cost; isolating it needs a one-card 99B, which does not fit
(204 experts x 48 layers = 48.4 GiB > 31.85 GiB VRAM).

### 13. What this campaign changed

- **Dead:** `B70_MAX_BATCH` (-1.17%), static asymmetric split (+0.03% on
  prose, -6.43% if tuned on synthetic), 5090-side streaming prefill of any
  kernel (host-RAM wall, §8).
- **Alive, sized:** per-layer route-aware placement (+7.80%, needs a provider
  ABI change), the 5090 inter-dispatch gap (69.7% of decode), and shrinking
  B70 residency (frees host RAM 1:1, the only thing that reopens §8).
- **Reframed:** prefill is the product problem (206 s at 110K, no caching),
  decode is within 24% of the 88B and structurally capped at 11.16 ms by the
  5090 leg.

## Bench 18 — what actually gates 99B prefill (2026-08-19, later same day)

Bench 17 said "prefill is the product problem". This is the search for the
lever, and it ends somewhere unexpected: **placement, not kernels**.

### 1. `--moe-backend` A/B — the local MoE kernel is ~1.6% of ITL

Identical prompts (1,456 tok via `--corpus`), identical cells, `SB_GPU_UTIL`
lowered to 0.83 to keep >=1.5 GiB spare on the 5090:

| backend | gate | ITL p50 | TTFT | KV | verdict |
|---|---|---|---|---|---|
| `cutlass` (VLLM_CUTLASS W4A4) | pass, identical text | **13.956 ms** | 2.732 s | 14.27 GiB | keep |
| `emulation` (Triton -> BF16) | pass, identical text | 14.185 ms (+1.6%) | 2.708 s | 13.72 GiB | reference |
| `marlin` (W4A16 at-load convert) | pass, identical text | 15.439 ms (+10.6%) | 2.698 s | 13.57 GiB | reject |

All three produced byte-identical greedy text and finite logprobs. **B70
service was identical across arms** (6,224 / 6,184 µs at M=256) -- the control
that proves the A/B isolated the 5090 side.

The decisive datum is `emulation`: a deliberately slow dequant-to-BF16
reference costs only **1.6%**. So the local-expert kernel is not where decode
time goes, and there is nothing to win by swapping it. Marlin's +10.6% is a
`MarlinExperts`-at-batch-1 pathology, not kernel quality.

`triton` is **not a valid NvFP4 MoE backend** -- `map_nvfp4_backend` accepts
only cutlass / flashinfer_{trtllm,cutlass,cutedsl,b12x} / marlin / humming /
emulation. `humming` was inconclusive (135 s boot budget was too short, not a
failure); `flashinfer_trtllm` remains unqualified and is a known GPU-wedge
risk (Bench 16 §9).

### 2. Grouped B70 kernel, re-run at the **99B** geometry

The 2026-08-13 verdict was measured at the 35B shape. Re-run at ours
(N=102 resident, K=3072, dim=1024, top-8), `quixicore_xpu_bench`:

| variant | M | µs/token | weight read | achieved GB/s |
|---|---|---|---|---|
| split (per-route GEMV) | 256 | 66.3 | 9,216 MiB | **569.7** |
| split | 2048 | 67.7 | 73,728 MiB | **557.3** |
| grouped | 256 | 166.2 | 459 MiB | **11.3** |
| grouped | 2048 | 113.9 | 2,394 MiB | **10.8** |
| grouped | 8192 | 110.4 | 9,387 MiB | **10.9** |

Cross-check: split at M=256 is 66.3 µs/token for 8 routes on one card, i.e.
~33.1 µs/token/layer at production's ~4 routes/card -- against 27.1-34.8
measured in vLLM. The bench is trustworthy.

**Split is at hardware limits** (557-570 GB/s logical, at/above the card's
~510 GB/s via L2 reuse). Grouped reads **30.8x fewer bytes** and still loses
1.68-2.5x because it runs at **~2% of bandwidth**. Stage timings localise it
exactly (M=2048): `gate_up` 177.8 ms + `down` 55.0 ms = 99.9% of 233 ms;
histogram 0.109, scan 0.020, scatter 0.108 ms are free. The routing
infrastructure is fine; the grouped GEMMs are broken. A grouped kernel at
even 100 GB/s would be 5.4x faster than split -- real headroom, but it is a
new Xe2 grouped GEMM, i.e. weeks.

Split is also flat in M (66.3 -> 67.7), independently re-confirming why
`B70_MAX_BATCH` measured -1.17%.

### 3. The residency shadow is **committed**, not accounted

Holding 24 GiB on *both* cards at once (the real serving shape):

```
  0000:11:00.0 holding 24.0 GiB -> MemAvailable 30.36 GiB
  0000:15:00.0 holding 24.0 GiB -> MemAvailable  6.21 GiB
  total device held 48.0 GiB, host consumed 48.17 GiB
  touched+read 8.0 GiB in 29.34 s (0.27 GiB/s)   <- swap thrash
  MemAvailable now 0.32 GiB
```

1:1 to within 0.4%, and the follow-on host allocation ran at **0.27 GiB/s**,
~25x below DRAM speed. The shadow is hard. (First attempt at this probe
reported 1.5e9 GiB/s -- `-O2` had deleted the touch loop. Write-then-read
into a `volatile` sink.)

### 4. Bank source bandwidths, measured

| source | rate | one 44.4 GiB pass |
|---|---|---|
| pinned host DRAM (88B's measured path) | 53.9 GiB/s | **0.82 s** |
| pageable page cache | 18.5 GiB/s | 2.4 s |
| **NVMe O_DIRECT** | **5.6 GB/s** | **8.5 s** |
| B70 VRAM D2H (shared Gen4 x4 uplink) | 6.44 GB/s | 7.5 s |

Disk has **84 GB free (96% full)**. zstd -1 on a real 2 GiB slice of the bank
compresses to only **80.2%** (an earlier 55% figure was a sparser region and
is withdrawn), and `zstd -d` runs at **3.33 GB/s single-frame** -- `-T0` did
not parallelise, so an NVMe-compressed bank needs per-layer frames and 2-3
decoders to stay ahead of the 5.6 GB/s read. `nproc` is 8.

### 5. Where the 88B's prefill advantage actually comes from

Not the kernel. The **layout**:

| | 5090 | B70 #1 | B70 #2 |
|---|---|---|---|
| 88B | ~22.4 GiB (dense + **54 local experts**) + 2.9 GiB KV | 27.4 GiB (126 experts) | **unused** |
| 99B as shipped | 10.34 GiB (dense + **1 local expert**) + 14.27 GiB KV | 24.34 GiB (102) | 24.34 GiB (102) |

Three consequences, all measured elsewhere in this ledger: 30% of the 88B's
routes never crossed PCIe; its shadow was 27.4 GiB not 48.7, which is the
**only** reason its bank could be DRAM-pinned at 53.9 GiB/s; and one lane
gave it a 142 µs inter-dispatch gap against our 208 µs.

The 99B as shipped sits in the worst corner of that space. `fractional:2:F`
already supports any local count (`cuda_n = round(num_experts *
cuda_fraction)`, CUDA takes the high ids), so this is a **flag**, not a code
change. `SHOOTING_BRAKE_PLACEMENT` and `SB_GPU_UTIL` are now overridable in
`serve_99b_dual.sh` for exactly this sweep.

### 6. The frontier, and why it has a floor

Per-unit, all measured: expert on CUDA **0.2373 GiB**, expert in a Marlin
bank **0.2175 GiB**, dense **10.10 GiB**, KV **24.7 KiB/token**, B70
**201 µs per route per token**. Empirically `used ~= U * 31.84 + 1.5` GiB, so
`U=0.90` still leaves ~1.7 GiB spare -- **0.83 was leaving 2.2 GiB unused**.

| local L | KV | KV tokens | shadow | DRAM free | 8K TTFT (with DRAM partial bank) |
|---|---|---|---|---|---|
| 1 (shipped) | 14.3 GiB | 582K | 48.4 | ~2.8 | 15.5 s |
| 30 | 8.1 | 330K | 41.5 | ~12 | ~7.7 s |
| 54 (88B-like) | 2.45 | 99K | 35.8 | ~17.6 | **~4.5 s** |
| 62 | 0.55 | 22K | 33.9 | ~19.5 | **~3.4 s** |

`[ESTIMATE]` -- bank not built; Marlin per-layer scaled from the 88B floor
bench at our shape.

The floor exists because **a banked remote expert costs DRAM twice**: 0.2373
shadow (decode needs it B70-resident) + 0.2175 bank (prefill) = 0.455 GiB,
capping banked experts at ~117 of a 53.4 GiB budget -- while CUDA cannot
absorb the other 88. Hence ~3.4 s is the wall on this box, i.e. rough 88B
parity, not a win.

Ruled out on the way: KV in B70 VRAM (790 MiB/token at 32K over 6.44 GB/s =
**123 ms/token**, a 9x decode regression); host KV offload (~32 ms/token and
no free DRAM anyway); bank in B70 spare VRAM (4.4 s/pass *and* it adds 28 GiB
of fresh shadow); single-B70 99B (needs L=75 => 27.9 GiB CUDA weights =>
negative KV).

### 7. The one upgrade that breaks the tradeoff

**+64 GiB host RAM.** At 123 GiB total, shadow 48.7 + full 44.4 GiB bank +
engine fits, the bank pins at 53.9 GiB/s, and 8K lands at **~1.4 s with KV
left at 530K tokens** -- 2.2x better than the 88B *without* trading context.
Every software path measured here is strictly worse per unit of effort.

### 8. Placement measured, and the cost model that falls out

`SHOOTING_BRAKE_PLACEMENT=fractional:2:0.2634` (54 local experts),
`SB_GPU_UTIL=0.90`, prose corpus, identical prompts:

| cell | L=1 | L=54 | delta |
|---|---|---|---|
| 8,501 tok TTFT | 15.476 s | **12.21 s** | **-21.1%** |
| B70 service @ M=256 | 6,224 µs | 4,906 µs | -21.2% |
| 17,372 tok TTFT | 1,847 µs/tok | 1,425 µs/tok | -22.9% |
| 1,456 tok TTFT | 2.732 s | 2.332 s | -14.6% |
| ITL | 13.956 ms | 13.657 ms | -2.1% |

Boot facts: CUDA weights **23.056 GiB** (predicted 22.91), **48/48 compact
layers verified** -- preemptive surgery handles 54 local experts, previously
only ever exercised at 1. KV **3.69 GiB** at **4.00x concurrency for 32K**.
B70 residency 36.1 GiB (18,362 + 18,605 MiB). Host MemAvailable **2.78 ->
15.29 GiB**: the 1:1 shadow model confirmed in situ, +12.5 GiB freed against
-12.6 GiB of residency.

Two points fit with no slack:

```
TTFT(8.5K) = 2.91 s + 0.0616 s x (experts resident on B70)
   L=1  : 2.91 + 204(0.0616) = 15.48   measured 15.476
   L=54 : 2.91 + 151(0.0616) = 12.21   measured 12.21
```

**Each expert moved off the B70 is worth 61.6 ms at 8.5K.** We moved 26% of
them and got 21%; nothing was lost, there was simply no more to take. Halving
TTFT by placement alone needs L~153 = 36 GiB of expert weights on a 32 GiB
card. Placement is a spent lever past ~L=61.

### 9. Three-way VRAM competition (measured the hard way)

`SHOOTING_BRAKE_B70_MAX_BATCH=2048` at L=54 **fails to boot**:

```
ValueError: 0.91 GiB KV cache is needed, but only 0.74 GiB available
            estimated maximum model length is 25152
```

So `MAX_BATCH=2048` costs **~2.95 GiB of 5090 VRAM**, and local experts, KV,
and dispatch staging all draw on the same pool. The engine refused rather
than thrashing, which is the correct failure.

This matters for the streamer, because the bank streams **once per vLLM
forward pass**, not once per prompt:

```
TTFT = 2.91 + max( 0.0616 x N_b70 , ceil(T/MNBT) x bank_GiB / 53.9 )
```

| config | passes @8.5K | B70 term | 5090 term | TTFT |
|---|---|---|---|---|
| L=54, no bank (measured) | - | 9.30 s | - | **12.21 s** |
| L=54, 64-expert bank, MNBT 2048 | 5 | 5.36 s | 1.29 s | ~8.3 s |
| L=54, full bank (+16 GiB RAM), MNBT 2048 | 5 | 0 | 3.05 s | ~6.0 s |
| L=54, full bank, MNBT 8192 | 2 | 0 | 1.22 s | ~4.1 s |
| L=54, full bank, MNBT 32768 | 1 | 0 | 0.61 s | **~3.5 s** |

That is why the 88B ran `MNBT=32768`: one step, one bank pass. Parity needs
all three of L, KV and MNBT balanced -- not any one maximised.

### 10. Why the 88B fit and the 99B does not

| | remote | format | cards | shadow | bank | total | vs 59.44 |
|---|---|---|---|---|---|---|---|
| 88B | 126 | int4 | **1** | 27.4 GiB | 27.4 GiB | **54.8** | fits |
| 99B @L=54 | 151 | NVFP4 | 2 | 36.1 GiB | 33.0 GiB | **69.1** | **short 9.7** |

Device VRAM was never the constraint: 96.3 GiB total, the checkpoint uses
59.2 GiB across three cards, and **27.6 GiB of B70 VRAM sits idle**. It is
unusable for a bank because allocating B70 VRAM costs host DRAM 1:1 -- free
VRAM you have to pay DRAM for -- and readback is capped at 6.44 GB/s anyway.

Three compounding differences, none software-fixable: more remote experts
(151 vs 126), NVFP4 is 9% fatter than int4, and **every remote expert is
charged twice** (0.2373 shadow so decode can reach it + 0.2175 bank so
prefill can stream it = 0.455 GiB), capping banked experts at ~117 of a
53 GiB budget while the 99B has 151 to place.

**+16 GiB of host RAM closes a 12.1 GiB gap** and is worth ~3.5x on 8K TTFT.
No placement, kernel, format or compression lever measured today comes close
to that per unit of effort.

### 11. Bank format decision: NVFP4, not Marlin

Marlin is 8% smaller per expert (0.2175 vs 0.2373 GiB) which matters while
DRAM is short -- but it needs NVFP4 -> int4 requantisation, which is new code
that changes the served weights and demands a full numerical gate.

An **NVFP4** bank streamed into `CutlassExpertsFp4` -- the backend already
serving the local experts and already qualified (Bench 18 §1) -- uses the
*same bytes* from the existing `expert_bank_99b.bin` and the *same kernel*,
so the streamed partial should be bit-identical to the resident one. That
reduces the gate to a smoke test instead of a requantisation study. Start
there; revisit Marlin only if the 8% is what stands between us and fitting.

### 12. Third point validates the model; fp8 KV rejected

| config | predicted | measured |
|---|---|---|
| L=1 | 15.48 s | 15.476 s |
| L=54 | 12.21 s | 12.21 s |
| **L=61 + fp8 KV** | **11.78 s** | **11.924 s** |

`TTFT(8.5K) = 2.91 + 0.0616 x N_b70` holds to 1.2% across a 3x range of
remote counts. The coefficient is real.

**fp8 KV works on this hybrid model** -- `--kv-cache-dtype fp8` is accepted
and the attention block size moves `2096 -> 4176 tokens`, so the mamba
page-equality constraint does *not* block it (that was the open question).
But it is a bad trade:

| | L=54, bf16 KV | L=61, fp8 KV |
|---|---|---|
| TTFT 8.5K | 12.21 s | 11.92 s (-2.4%) |
| **ITL** | **13.657 ms** | 15.075 ms (**+10.4%**) |
| KV | 3.69 GiB, 4.00x @32K | 1.35 GiB, 2.55x @32K |
| 5090 spare | 2.5 GiB | 1.38 GiB (under the 1.5 floor) |

fp8 forces dequant in the attention path
(`decode_backend=flashinfer-native, kv_cache_dtype=float8_e4m3fn`), spending
10.4% of decode to buy 2.4% of prefill. Rejected. Note also the trade steepens
near the wall: L=54->61 added 1.65 GiB of weights but cost 2.34 GiB of KV, so
**L~57 is the practical maximum** at a 1.5 GiB spare-VRAM floor.

**Adopted as the serve default** (`fractional:2:0.2634`, `SB_GPU_UTIL=0.90`):
strictly better than the shipped 1/205 on both prefill and decode, keeps 131K
KV tokens at 4.00x concurrency for 32K. The old long-context profile is one
env var away.

### 13. Pinned H2D to the 5090, measured here

| transfer | 64 MiB | 256 MiB | 1 GiB |
|---|---|---|---|
| pinned | 47.21 | 56.78 | **57.27 GB/s** |
| pageable | 25.71 | 20.50 | 20.93 GB/s |

Better than the 88B's inherited 53.9 GiB/s. Bank pass cost: **0.28 s** for a
64-expert NVFP4 bank (15.2 GiB), **0.67 s** for all 151 (35.8 GiB). The 5090
leg has slack; the B70 term is what dominates.

### 14. Why the existing streamer cannot be used as-is

`SHOOTING_BRAKE_B70_PREFILL_STREAM=1` is already implemented for NVFP4 and
would read this model's `SBEXP001` bank directly -- but
`_load_host_experts_from_bank` sizes the host arena for **every** B70-owned
expert:

```
(205-L) x 0.2373  <=  15.29 + (L-54) x 0.2373 - 2   =>   L >= 102
```

and L=102 needs 34.3 GiB of CUDA weights on a 31.84 GiB card. So the
all-or-nothing streamer can never fit, and a **partial** bank is mandatory.

The change is contained, because `_prefill_forward_offloaded` already sums
three tiers and already passes both id spaces (compact provider slots +
global arena ids):

```
stream_ids   = _stream_id_map_cuda[topk_ids]        # arena index, else -1
dispatch_ids = where(stream_ids >= 0, -1, b70_ids)  # mask streamed out
partial      = stream_partial(stream_ids) + dispatch_partial(dispatch_ids)
```

plus a subset argument to the arena loader. Expected: ~8.3 s at 8.5K with a
64-expert bank, versus 12.21 s now.

## Bench 19 — Laguna r20 first serve on dual B70 (2026-08-20)

**Not a planned bench** — the bring-up record for
`srswti/axe-superveloce-jota-118b-r20-nvfp4` (`LagunaForCausalLM`), with
measured negatives and one estimate of mine that was wrong by 18 points.
Recipe: `benchmarks/serve_jota_r20_dual.sh`. Plan:
`docs/superveloce-jota-r20.md`.

### 1. Prefix caching — the reason this model exists

Identical 6,031-token prose prompt, three runs, then a **prefix-disjoint
control**. Wall clock is the weak evidence; the hit counters are the strong
evidence.

| run | wall | prompt_tok | `hits/queries` |
|---|---:|---:|---|
| 0 (cold) | 9.695 s | 6,031 | 0 / 6,046 |
| 1 | **0.071 s** | 6,031 | 6,016 / 12,077 |
| 2 | **0.068 s** | 6,031 | 12,032 / 18,108 |
| control (disjoint, 6,635 tok) | **10.611 s** | 6,635 | **0 new hits** |

- **142.7× cold→warm.** The repeat hit **6,016 of 6,031 blocks (99.75%)**.
- The control returned to 10.6 s with zero new hits, so the speedup is
  caching and not a warmed engine, a clock ramp, or a scheduler artifact.
- The 99B measures **1.007×** on this identical test (9.788 / 9.729 /
  9.723 s) and always will: `Qwen3_5MoeForCausalLM` is `is_hybrid=True` with
  `supports_mamba_prefix_caching=False`, so vLLM disables prefix caching for
  the whole model. `LagunaForCausalLM` is not `IsHybrid`.

### 2. Decode — faster than the 99B, and my prediction was wrong

| metric | 99B (Bench 18) | r20 | delta |
|---|---:|---:|---:|
| ITL, C=1, 4 runs excl-first | 13.960 ms | **11.956 ms** | **−14.4%** |
| KV @ L=54 | 3.69 GiB | **9.98 GiB** | **+170%** |
| CUDA graph pool | 0 GiB (eager) | 0.41 GiB | — |

ITL is a clean comparison: prefix caching touches prefill only. Both B70
clocks held 2800 MHz median with `idle_fraction` 0.02.

**I predicted 14.6 ms (+4.4%). Measured 11.96 ms (−14.4%) — wrong by 18
points.** The model was `46 gaps x 208 us + 47 dispatches x 80 us`, scaling
service by top_k 10/8 while holding the inter-layer gap constant. The gap is
~76% of ITL and it clearly shrank: r20 replaces 36 GDN/Mamba layers with
sliding-window-512 attention and carries 3.18 GiB of dense weights against
the 99B's 10.24. **Lesson: I scaled the term I understood and froze the term
that dominates.**

r20's ITL lands **within 3.9% of the 88B's 11.51 ms**, the target this whole
campaign has been chasing.

### 3. The coordinate contract, proven on silicon

The load-bearing risk was a compact 47-row bank read as a leading prefix,
serving the neighbouring layer's experts — plausible tokens, finite
logprobs, and a file that is exactly the right length either way.

| trace | entries | model layers | bank rows | `layer == row+1` |
|---|---:|---|---|---|
| device0 | 2,820 | 1..47 (n=47) | 0..46 | **True** |
| device1 | 2,820 | 1..47 (n=47) | 0..46 | **True** |

All 5,640 entries. On the 99B the same assertion must be *identity* and is;
on r20 it must be *offset by one* and is. Model layer 0 never appears — it is
the dense MLP. `M` values `[1, 2, 4, 8, 256]`, so chunked prefill dispatches
through the doorbell too.

Non-empty traces are the only possible evidence for one specific failure: if
vLLM had selected a monolithic expert class it would call
`forward_monolithic`, our code would never run, and **no guard of ours could
fire**. Token correctness would not have revealed it.

### 4. Measured, not estimated

| quantity | value |
|---|---|
| cold prefill | **1,607 µs/token** (6,031 tok in 9.695 s) vs the 99B's 1,436 |
| warm prefill | ~0 — 99.75% block hit rate |
| bank | 47 rows x 205 experts, 51,146,665,300 B, byte-exact vs checkpoint |
| resident split | 76 / 75, disjoint, 151 remote + 54 CUDA |
| top_k | 10, via `sb_b70_load_v2` (v1 frozen at 8) |

Cold prefill is ~12% *worse* per token than the 99B — consistent with +22.4%
routes (470 vs 384) partly offset by one fewer MoE layer. r20 pays slightly
more once, then nothing.

### 5. What this does NOT establish

- **Output quality.** r20 is 20% REAP-pruned from 118B with a 512-token
  window. Two greedy prompts and finite logprobs are not a quality
  measurement. `" oxygen.\n\nAuthor:\n\nHi, I'm trying to understand..."` is
  coherent but drifts. **This gates shipping regardless of latency.**
- Throughput/concurrency, long context, TTFT at a known length on a cold
  cache, sustained stability, per-card attribution under load.
- The 0.0624 s probe TTFT is **cache-assisted** (`excl_first` on a repeated
  shape) and must not be quoted as a prefill number.

### 6. Instrument notes

- KV numbers require a **settle** step: a boot 2 s after `pkill -9` profiled
  a partially-released pool and reported 2.43 GiB where a settled boot
  reported 3.7. Poll `nvidia-smi` to baseline before any boot that produces
  a number.
- A poll loop matching the bare word `Traceback` reported a boot failure on a
  benign `deep_gemm` optional-import warning. Match
  `Engine core initialization failed`.
- `benchmarks/b70_itl_probe.py` defaults to `/v1/chat/completions`; handing
  it `/v1/completions` returns 400.

## Bench 20 — Laguna r15 local-expert sweep (2026-08-20)

**Ordered as an optimisation; delivered a negative result and a trade.**
The question was whether `L=57` — inherited arithmetic, the 99B's measured
optimum of 54 rescaled onto 218 experts — is right for Laguna, whose memory
economics differ completely (dense 3.18 vs 10.24 GiB, KV 1.70x per token).
I predicted ~20% on the 99B's L=1->54 precedent. **That was wrong.**

`benchmarks/laguna_l_sweep.py`, five boots, util 0.85, MML 131072:

| L | remote | KV GiB | conc @131K | prefill us/tok | ITL ms | card MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 198 | 16.66 | 2.51 | 1940.4 | 11.596 | 28923 |
| 40 | 178 | 11.89 | 1.79 | 1759.4 | 10.868 | 28747 |
| 48 | 170 | 10.10 | 1.52 | 1667.2 | 10.530 | 28798 |
| 57 | 161 | 7.97 | 1.20 | 1636.6 | 10.850 | 28864 |
| 61 | 157 | 6.96 | 1.05 | 1576.8 | 10.571 | 28796 |

1. **No free speed. L=57 was already near the prefill optimum.** The
   inherited default was well-chosen. Prefill is monotone in L and still
   improving at the ceiling.
2. **Decode ITL saturates at L~40.** From 40 to 61 — 21 more local experts —
   ITL reads 10.87/10.53/10.85/10.57, non-monotone, flat inside ~1.5%
   noise. Experts past 40 buy decode *nothing*. This is the finding worth
   keeping: the decode gap is 5090-side per-layer work (Bench 17's 208 us),
   not expert locality, so buying locality cannot move it.
3. **A cheap trade exists.** L=48 over L=57: ~2% cold prefill for +2.13 GiB
   KV and +27% concurrency. Adopted as the r15 default, because the 139x
   prefix cache (Bench 19) erases cold prefill on repeat traffic while
   concurrency is paid on every request.
4. **Total card usage is L-invariant** — 28.75-28.92 GiB across all five.
   vLLM sizes KV to fill whatever weights leave, so L never threatens the
   ~29 GiB desktop ceiling. Tuning L is free with respect to VRAM.
5. **Hard ceiling L=61.** L=65 leaves 6.0 GiB KV; one 131072-token request
   needs 6.64. Boot refuses, correctly.

**Measurement caveat, mine.** Defeating the prefix cache required a unique
prompt slice per point, so slice *content* varies across L — and content
shifts per-token cost several percent through routing balance (Bench 17
measured 9.2% gate-cost difference prose vs synthetic). Same-slice pairs are
the only clean ones: L=20->48 is **-14.1%**, L=40->61 is **-10.4%**. The
trend is real; individual 2-4% steps are not. A cleaner design would fix the
slice and flush the cache between points.

**Not established:** r20 was NOT swept. It has 205 experts and its own KV
cost; its `fractional:2:0.2634` (54 local) is still inherited arithmetic and
should not be assumed optimal from this. Nothing here measures quality.

## Bench 21 — r15 serving surface, and the chunk-stall tail (2026-08-20)

Ten cells, context x concurrency x 128 output, real prose corpus, per-card
trace attribution on one clock. `benchmarks/results/.../r15_L48_matrix.json`.
**Zero failed requests across the matrix, including 131072 context.**

| ctx | C | ptok | TTFT p50 | us/tok | ITL p50 | ITL p99 | tok/s/req | agg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1 | 928 | 1.564 | 1685 | 10.95 | 11.40 | 91.3 | 43.3 |
| 1024 | 2 | 1061 | 3.364 | 3172 | 13.25 | 13.72 | 75.5 | 49.1 |
| 1024 | 4 | 1149 | 7.443 | 6481 | 18.39 | 30.58 | 54.4 | 51.2 |
| 8192 | 1 | 7122 | 11.803 | 1657 | 11.12 | 11.52 | 90.0 | 9.7 |
| 8192 | 2 | 8770 | 24.601 | 2805 | 13.76 | 388.24 | 72.7 | 8.2 |
| 8192 | 4 | 13604 | 45.827 | 3369 | 20.03 | 3458.55 | 49.9 | 5.5 |
| 32768 | 1 | 19907 | 33.441 | 1680 | 11.56 | 24.77 | 86.5 | 3.6 |
| 32768 | 2 | 21529 | 56.906 | 2643 | 14.60 | 3460.77 | 68.5 | 3.4 |
| 32768 | 4 | 37159 | 154.589 | 4160 | 22.80 | 3549.29 | 43.9 | 2.0 |
| 131072 | 1 | 125756 | 220.589 | 1754 | 14.86 | 26.36 | 67.3 | 0.6 |

1. **Prefill is linear over a 135x token range.** 1685/1657/1680/1754 us/tok
   at C=1 from 928 to 125,756 tokens — 4% spread. The linear model holds at
   L=48 exactly as it did at L=57.
2. **Concurrency does not buy throughput past ~1K context; it costs it.**
   Aggregate goes 9.7 -> 8.2 -> 5.5 at 8K and 3.6 -> 3.4 -> 2.0 at 32K.
   Prefill is serialised and dominates, so adding streams adds queueing.
   Only at 1K does concurrency help at all (43 -> 51 tok/s).
3. **131072 context serves, and costs 3.7 minutes.** 220.6 s TTFT for
   125,756 tokens. That is the headline SLO problem, and prefix caching is
   the only thing that touches it.

### The chunk stall — diagnosed, then fixed

The p99 column is not noise, it is **arithmetic**: a decoding request waits
one entire prefill chunk behind another request's prefill.

```
MNBT x per-token prefill rate = predicted stall
2048 x 1670 us = 3.42 s   measured p99: 3.44 / 3.46 / 3.49 / 3.55 s
 512 x 1670 us = 0.86 s   measured p99: 0.872 / 0.895 s
 256 x 1670 us = 0.43 s   measured p99: 0.437 / 0.449 s
```

Three predictions, three matches inside 5%. A/B on **matched prompt slices**
with MNBT the only variable (an unmatched first attempt compared 21,528
against 46,957 tokens for the same `ctx=32768` target and had to be thrown
away — see the instrument note below):

| cell | ITL p99 | ITL p50 | TTFT p50 | agg tok/s |
|---|---:|---:|---:|---:|
| 8192x1 | -1.2% | -0.8% | +0.1% | 0.0% |
| 8192x4 | **-74.7%** | -0.1% | **-12.8%** | -0.3% |
| 32768x2 | **-74.3%** | -0.2% | +0.2% | -1.3% |

**512 costs nothing measurable** and hands back ~0.33 GiB of KV (smaller
activation buffers): 9.97 -> 10.30 GiB. TTFT at 8192x4 *improves* 12.8% from
fairer scheduling.

**256 does not continue the trend — it inverts.** p99 halves again, but a
47K prompt becomes 184 chunks, the scheduler interleaves most decode tokens
into the prefill window, and ITL p50 at 32768x2 goes **16.1 -> 204.9 ms, a
12.7x regression**. Small chunks do not remove the stall; they spread it
across every token. Median dies to save the tail.

**Adopted: MNBT=512** on r15 (measured) and r20 ([INFERENCE] — same
arithmetic, prefill rates within 4%, and shipping a known 3.4 s tail would
be worse than shipping the inference).

### Instrument defect found

`b70_matrix_probe.py` context targeting is **loose**: it slices words, and
tokens-per-word varies with corpus content, so `ctx=32768` delivered 21,528
tokens in one run and 46,957 in another. This also explains the 2-4% wobble
in Bench 20's prefill column. Any A/B across separate invocations must
assert on `actual.prompt_tokens_median`, not trust `target`. A proper fix
binary-searches word count against the real tokenizer.

**Not established:** quality. Sustained multi-hour stability. GuideLLM as a
formal acceptance gate. r20's own surface.

## Bench 22 — r15 cold-vs-warm surface (2026-08-21, IN FLIGHT)

**Status: 31 of 48 cells.** Resumable — rerun the identical command and the
instrument skips completed cells:

```
SB_MNS=6 benchmarks/serve_jota_r15_dual.sh          # server, MNS 6 for C<=6
HF_HUB_OFFLINE=1 .venv/bin/python benchmarks/r15_cold_warm_matrix.py \
  --model shooting-brake-jota-r15 \
  --tokenizer-dir srswti/axe-superveloce-jota-118b-r15-nvfp4 \
  --corpus ~/sb_corpus_big.txt \
  --contexts 1024,4096,8192,16384,32768,61440,92160,126976 \
  --concurrency 1,2,3,4,5,6 --output-tokens 512 \
  --json-out benchmarks/results/b70_gemv_audit/r15_cold_warm_matrix.json
```

Done: 1024/4096/8192/16384/32768 complete at C=1..6, plus 61440 C=1.
Remaining: 61440 C=2..6, 92160 C=1..6, 126976 C=1..6 (~2 h, all prefill).

### What it already establishes

1. **Bench 21's concurrency finding was an artifact of cold prompts.** I
   reported "concurrency costs aggregate throughput." Warm, it does not:
   1K aggregate goes 89.2 -> 145.6 -> 159.4 -> **211.1** tok/s at C=1..4, a
   2.4x scale-up. Cold it barely moves (69.5 -> 125.2). What Bench 21
   measured was prefill serialisation, not a concurrency ceiling. Production
   runs warm.
2. **ITL is identical cold vs warm at every concurrency** (10.87/10.87,
   13.20/13.21, 18.36/18.38, ...). Caching touches prefill only, exactly as
   expected, and it doubles as a determinism check on the instrument.
3. **The ITL tiers are CUDA-graph batch buckets, not a smooth curve.**
   10.87 / 13.20 / **18.36, 18.41** / **27.67, 27.92** ms for C=1..6. C=3
   costs what C=4 costs; C=5 costs what C=6 costs. Captured sizes are
   {1,2,4,8} with padding, so **always fill to the bucket boundary** -- C=3
   wastes a slot and C=5 wastes two. C=7/8 should also read ~27.9 ms, making
   `max_num_seqs=8` free relative to 6 per token. Unconfirmed.
4. **The prefix cache does not degrade, it falls off a cliff.** Warm hit rate
   at ctx=16384: C=4 -> 0.9994, C=5 -> **0.200**, C=6 -> **0.000**, and at
   C=6 warm TTFT equals cold to 1 ms (99.2017 vs 99.2026 s). Hits at C=5 were
   exactly 16,384 of 81,916 -- one prompt's worth survived, four were evicted.
   **Zero preemptions**, so it is not scheduler thrash. 98,304 tokens is only
   44% of the ~221K pool, so raw capacity does not explain it either.
   **Mechanism unknown** and worth chasing: `gpu_cache_usage_perc` came back
   -1 because the gauge was renamed in 0.27, which would help diagnose.
   Practical rule until then: bank on caching for ONE shared prefix of up to
   ~65K tokens, not for several large distinct ones.

### Instrument

`benchmarks/r15_cold_warm_matrix.py` (`ca77b678`). Token targeting lands
within 3 tokens of any target up to 126,976 (drift <=0.2% across all 31
cells); cold passes read 0.000 hit rate, warm 0.99+. Two of my own bugs were
found and fixed en route: a guess-and-correct prompt sizer that oscillated
(6330 tokens for a 4096 target, because each correction sampled a different-
density corpus region), and quoting warm-TTFT *ratios* at the measurement
floor where 0.017 s vs 0.070 s swings the ratio 4x for no physical reason.
Warm TTFT is reported in absolute ms.

## Group-32 / NVFP4 grouped kernel — plan corrected before any build

Recorded here because it **invalidates what I proposed**: converting the bank
to group-32 was never sufficient.

- The vendored Xe2 grouped kernel implements **MXFP4, not NVFP4**. `B_DTYPE`
  has no NVFP4 entry and the scale decode is `bits << 23`, i.e. E8M0
  exponent-only at group 32. Ours is E4M3 at group 16 plus an FP32 global
  scale, so even at group 32 the scale *format* is wrong.
- Upstream `sycl-tla` (95 PRs newer than the pinned `.deps` snapshot) has real
  Xe block-scaled grouped GEMM (`xe_array_mma_blockscaled_mxfp.hpp`,
  `xe_tile_scheduler_group.hpp`, `examples/51_xe35_block_scaled_grouped_gemm`,
  arch tag `IntelXe`) -- but for `mx_float4_t` only. Every nvfp4 path is
  `sm100`/`sm103`, i.e. NVIDIA. **Nobody upstream has an Xe NVFP4 grouped
  GEMM.**
- `nv_float4_t` already exists (`float_subbyte.h:506`) with
  `ScaleFactorType = float_ue4m3_t`, and the Xe collective gates scale loads
  through a `ScaleCopyTraits` specialization. So the NVFP4 route is a bounded
  traits specialization at SF vector 16 (plus two scale groups per 32-wide
  K-tile, since every policy has `tile_k=32` and scales load only when
  `k_tile*tile_k % group_size == 0` -- group 16 would silently apply the first
  group's scale to all 32 elements). That keeps the 54 GiB bank **bit-exact**:
  no requantisation, no quality gate, no rebuild. Strictly better than MXFP4
  conversion, which would stack E8M0's power-of-two-only scales on top of the
  16->32 merge.

**Next action, gate pre-committed.** `src/phase7/xe2_probe/` builds and is
unrun (deliberately: it saturates B70 bandwidth and would corrupt Bench 22).
It measures the unmodified kernel on synthetic MXFP4 at our real geometry
(E=85, K=3072, N=1024, ~30 rows/expert). **Best >=200 GB/s justifies the
`ScaleCopyTraits` port; below that the plan dies.** Reference points: the
split path we ship achieves 437.6 GB/s and is already saturated; our own
grouped attempt managed 9.9 GB/s and lost 1.55x.

Note the tension: MNBT=512 (Bench 21) gives only ~30 rows/expert, and small M
is where grouped GEMMs lose. MNBT would want re-tuning after the kernel lands.

## Bench 23 — Xe2 grouped MoE GEMM at r15 geometry: the gate PASSES (2026-08-21)

`src/phase7/xe2_probe/`, speed-only, unmodified vendored kernel, synthetic
MXFP4, r15's real shape: E=85 resident experts, K=3072, N=1024, M=2560
routes (512 tok x top10 / 2 cards) = **30.1 rows/expert**.

| shape | policy | ms | GB/s |
|---|---|---:|---:|
| gate/up N=1024 K=3072 | `w4a16_policy_m_32` (32x64x32) | 0.369 | **441.7** |
| gate/up | `w4a16_policy` (128x256x32) | 0.789 | 206.6 |
| gate/up | `w4a16_policy_m_16` | 0.483 | 337.6 |
| gate/up | `w4a16_policy_m_8` | 0.776 | 210.1 |
| down N=3072 K=1024 | `w4a16_policy_m_32` | 0.344 | **474.4** |

Stable across runs: 442.0 / 442.7 / 441.7 GB/s. Pre-committed gate was
>=200 GB/s. **Passed by 2.2x.**

**The headline.** Intel's kernel reaches the same bandwidth as our already
saturated split path (437.6 GB/s) while moving **30.1x fewer bytes** --
127.5 MiB per grouped pass against 3,840 MiB. That is the whole prize, and it
is now measured on this silicon at our shape rather than inferred from a
synthetic bake-off at someone else's.

Real per-token cost, summing the three actual projections:
`(0.369 + 0.369 + 0.344) ms x 47 layers / 512 tok` = **99.3 us/token**
against **1,705 us/token** measured in production = **17.2x on the B70 GEMM
leg**. With B70 at 86-92% of TTFT that is ~6.6x end-to-end: 8K cold TTFT
13.8 s -> ~2.1 s, 124K 212 s -> ~32 s.

### Two things the M sweep settled

1. **Policy choice is worth 2.1x and is M-dependent.** At 30 rows/expert the
   small-M policy wins 441.7 vs 206.6 GB/s for the big one. Production must
   use the M-dependent selector; a fixed policy throws away half the win.
2. **MNBT=512 costs ~2x of the grouped kernel's efficiency.** At M=10240
   (MNBT 2048, 120.5 rows/expert) the BIG policy wins and per-token drops to
   **52.5 us/token** -- a 33x leg, ~7.9x end-to-end. So the Bench 21 decision
   to drop MNBT to 512 for the ITL p99 fix directly fights this. Both are
   real; the tradeoff needs measuring once the kernel is in, not guessing now.

### What this does NOT establish

- **Correctness.** Speed only, by design. Values were random bits.
- **NVFP4.** This ran MXFP4 (E8M0/group-32), which the kernel already
  supports. Our bank is E4M3/group-16. The `ScaleCopyTraits` specialization
  for `float_ue4m3_t` is still required and its cost is unmeasured -- though
  the E4M3 decode is a scalar op at scale-group load, not inside the DPAS
  loop, so it should be near-free. `[INFERENCE]`
- **Uneven routing.** `rows_per_expert` was even, the optimistic case for the
  work queue. Natural text measured max/mean 1.63 (Bench 17), which will cost
  something.

### Environment bug found: Level-Zero V2 adapter segfaults

oneAPI 2026.1 defaults to the Level-Zero **V2** adapter, and it jumps to a
null pointer on a plain USM `memcpy`:

```
#0  0x0000000000000000 in ?? ()
#1  ur_command_list_manager::isGraphCaptureActive()  <- libur_adapter_level_zero_v2.so
#2  v2::ur_queue_immediate_out_of_order_t::enqueueUSMMemcpy(...)
```

Workaround: **`SYCL_UR_USE_LEVEL_ZERO_V2=0`** (or `UR_LOADER_USE_LEVEL_ZERO_V2=0`).
`UR_L0_USE_IMMEDIATE_COMMANDLISTS=0` does NOT fix it. OpenCL works but is ~5%
slower (420.1 vs 441.7 GB/s), so it is not the answer.

**Open risk worth checking:** whether `libsb_b70_provider.so` can reach that
same path in production. It has not misbehaved, so it either avoids it or uses
a different queue mode -- but that is an absence of symptoms, not a proof.

### Instrument note

The probe first died as a bare `EXIT=139` with **no output at all**, because
`printf` to a pipe is block-buffered and the buffer dies with the process --
indistinguishable from crashing before the first statement. `setvbuf(...,
_IONBF, ...)` made it diagnosable in one run. Third silent-success trap of the
day; the lesson that stuck is to check for the artifact, never the exit code.

## Bench 24 — NVFP4 grouped path built and running (2026-08-21)

`src/phase7/xe2_nvfp4/` — our fork of the vendored kernel (provenance in
`.provenance`). Upstream is MXFP4 only; this adds NVFP4.

### Three blockers, three resolutions

1. **Scale layout — NOT a blocker, checked first and free.** `extract_experts.py`
   writes compressed-tensors scales verbatim: `[N, K/16]` uint8, row-major.
   The kernel indexes `Scales[n * group_num + group_idx]` with
   `group_num = K/group_size`. **Identical.** Record arithmetic confirms it
   byte-exact: 3,145,728 + 393,216 + 1,572,864 + 196,608 + 8 = 5,308,424,
   which is the real record size. No rearrangement anywhere, and the 54 GiB
   bank is untouched.
2. **group-16 vs `tile_k=32` — solved with a tile policy, not mainloop work.**
   The scale reload is gated on `k_tile * tile_k % group_size == 0`, so a
   32-wide tile spanning two 16-groups loads only the first scale and applies
   it to all 32 values -- silently wrong. Rather than teach the mainloop to
   carry two scales, added `tile_k=16` policies so `tile_k == group_size` and
   the existing gate is already correct. Measured cost: `m_32_k16` reads
   419.5 GB/s against `m_32`'s 439.9, i.e. **4.6% for correctness**. This was
   the item ranked "medium risk, real work"; it became a 20-line policy class.
3. **Per-expert FP32 global scale — removed from the kernel entirely.**
   `sum_k A_k (e2m1 * s_block * alpha) = alpha * sum_k A_k e2m1 * s_block`.
   Alpha is constant per expert, so it belongs on that expert's OUTPUT rows,
   which the provider already touches during the un-sort. One multiply per
   output element instead of one per weight, and zero kernel surface.

### Measured, r15 geometry (E=85, K=3072, N=1024, 30.1 rows/expert)

| path | tile_k | ms | GB/s | correct at gs=16 | our format |
|---|---:|---:|---:|---|---|
| MXFP4 `m_32` | 32 | 0.390 | 439.9 | NO | no |
| MXFP4 `m_32_k16` | 16 | 0.409 | 419.5 | yes | no |
| **NVFP4 `m_32_k16`** | 16 | 0.503 | **340.4** | yes | **yes** |
| NVFP4 `m_16_k16` | 16 | 0.803 | 213.5 | yes | yes |
| NVFP4 `k16` (128x256x16) | 16 | 1.435 | 119.4 | yes | yes |

**The E4M3 decode costs 23%** (0.503 vs 0.409 ms), not the "near-free" I
predicted. I argued the decode sits at a group boundary rather than the inner
loop -- true at `tile_k=32`, but at `tile_k=16` the reload fires every tile, so
it is effectively per-tile, and E4M3 -> float is real work next to a shift.

Revised projection: 138.6 us/token against **1,705 measured in production** =
**12.3x on the B70 GEMM leg**, ~**5.8x end-to-end** at 90% B70 share. Prefill
1,705 -> ~294 us/token; 8K cold TTFT 13.8 s -> ~2.4 s; 124K 212 s -> ~37 s.
Down from the 6.6x projected in Bench 23 purely because of the decode cost.

### NOT established -- this is the next gate, and it is the important one

**Correctness is unverified.** These numbers prove the path compiles, runs,
addresses the right bytes and moves them at 340 GB/s. They prove **nothing**
about the arithmetic: inputs were random bits with no reference comparison.
Specifically unproven: that `reinterpret_cast<float_e4m3_t const&>(byte)` is
the right bit interpretation for compressed-tensors scale bytes, and that the
stored trailer is a multiplier rather than a divisor (`extract_experts.py:269`
says `param = 1/global`, so alpha multiplies -- but that has not been
exercised end to end). A fast kernel computing the wrong product is worse than
no kernel, so nothing ships until argmax and prompt-logprob equivalence pass
against the current path.

Also still open: route sorting in the provider (histogram/prefix/scatter, plus
the alpha application on un-sort), M-dependent policy selection in production
(worth 2.1x and shape-dependent), and the MNBT tension from Bench 23.

## Bench 25 — NVFP4 correctness gate PASSED, and it corrected the speed (2026-08-21)

`src/phase7/xe2_probe/xe2_nvfp4_verify.cpp`. Controlled inputs rather than a
fit: E2M1 encodes {0,.5,1,1.5,2,3,4,6}, so a weight byte of `0x21` holds one
nibble worth 0.5 and one worth 1.0. With one-hot activations and every block
scale set to E4M3 1.0 (`0x38`), the output reads the convention off directly.

### The gate found a real bug on its first run

Every k returned 0.5 -- always the low nibble, never alternating. Cause:

```cpp
constexpr char actual_layout_of_B = LayoutKindB ^ ('R' ^ 'C');   // it INVERTS
```

Passing `'R'` makes B column-major internally, so a packed byte holds two
adjacent **N** values at the same k, not two K values. Our bank is `[N, K/2]`
with K contiguous, so **the layout char must be `'C'`**. With that fixed:

```
k=0 -> 0.5   k=1 -> 1.0   k=2 -> 0.5   k=3 -> 1.0   ...  (8/8 alternating)
```

**All four format questions are now settled on evidence:**

| question | answer | how |
|---|---|---|
| scale dtype | E4M3 | checkpoint says `torch.float8_e4m3fn` |
| scale decode | correct | `0x38` -> exactly 1.0 in the kernel |
| alpha | multiplier, `1/global` | bank trailer == `1/gscale`, byte-exact |
| scale layout | `[N, K/16]` row-major | bank bytes == checkpoint bytes |
| nibble order | low nibble = even k | 8/8 alternating with layout `'C'` |

### Which invalidated my own speed numbers

Benches 23 and 24 both ran `layoutB='R'` -- the transposed access pattern.
Same bytes moved, different access order, materially different bandwidth.
Re-measured with `'C'`, r15 geometry, group 16:

| path | ms | GB/s | us/token |
|---|---:|---:|---:|
| MXFP4 `m_32` (tile_k 32, incorrect at gs=16) | 0.399 | 429.2 | 110.0 |
| MXFP4 `m_32_k16` | 0.495 | 346.2 | 136.3 |
| **NVFP4 `m_32_k16`** | 0.623 | **275.0** | **171.6** |
| NVFP4 `m_16_k16` | 0.904 | 189.7 | 248.9 |

**Corrected projection: 171.6 us/token vs 1,705 in production = 9.9x on the
B70 GEMM leg, ~5.2x end-to-end.** Prefill 1,705 -> ~327 us/token; 8K cold TTFT
13.8 s -> ~2.6 s; 124K 212 s -> ~41 s.

Revision history of this one number, all downward, each time because the
correctness work exposed a flaw in my own measurement: **6.6x (Bench 23) ->
5.8x (Bench 24) -> 5.2x (here)**. The first two were measured with a
transposed weight layout that happened to run faster. Speed measured before
correctness is worth nothing, and this is the third time today that a setup
that ran cleanly was measuring the wrong thing.

### Where the 275 vs 429 GB/s gap goes

Two costs, both now separated: `tile_k` 32 -> 16 for group-16 correctness is
429.2 -> 346.2 (**19%**), and the E4M3 decode on top is 346.2 -> 275.0
(**21%**). The decode is dearer than the "near-free" I predicted because at
`tile_k=16` the scale reload fires every tile, making it per-tile rather than
per-group work. Both are the price of correctness on our real format; neither
is optional.

### Still to do

Route sorting in the provider (histogram/prefix/scatter, gather in, scatter
out, alpha on the way back), provider integration behind a flag, `.so` rebuild,
then a real before/after boot. Nothing is wired into the serving path yet, so
the server today behaves exactly as it did this morning.

## Bench 26 - r15 grouped + fp16 wire: full acceptance gate (2026-08-22)

Run before committing a long benchmark matrix to this config. Every row is
measured on the live server, not projected.

| check | result | reference | verdict |
|---|---|---|---|
| correctness gate | correct greedy tokens, finite logprobs | - | PASS |
| output quality, 120-prompt argmax | 108/120 and 106/120 vs baseline | control (baseline vs itself) 109/120 | INDISTINGUISHABLE |
| prefill, aggregate 1K-32K | 430 us/token | 1,705 this morning | 3.97x |
| decode ITL p50, C=1 | 11.40 ms (3 runs: 11.52/11.38/11.40) | 11.1-11.2 ms | +2.2%, inside noise |
| ITL p99, C=4 @ 8K | 852 ms | 872 ms at the shipped MNBT=512 | EQUAL |
| prefix cache | 5-178x warm speedup | - | works |
| KV pool | 8.38 GiB | 10.51 GiB at MNBT=512 | -20%, the MNBT trade |

**The MNBT=512 -> 2048 trade came out free.** Bench 21 shipped 512 because 2048
gave a 3,440 ms p99 ITL stall. That stall is `MNBT x prefill us/token`, and
prefill is now 4x cheaper, so the 4x bigger chunk costs the same tail it used
to: 852 vs 872 ms. Predicted 881, measured 852 -- 3%.

**Decode is structurally untouchable by this work** and the measurement agrees:
`use_split = M <= 32` keeps M=1 on the old per-route GEMV. First sample read
12.46 ms and would have been reported as a +12% regression; it was a cold first
run (ttft 226 ms vs 29 ms warm). Three samples killed it. Fourth time today a
single measurement lied.

### The finding that outlives the speedup

**This rig has never been reproducible, and nobody had measured that.** The
PRE-EXISTING per-route GEMV path flips argmax on **9% of prompts run twice**
(109/120 agreement with itself), and its 3-pass top-1 logprob spread (0.339
nats) is WIDER than the grouped path's (0.277).

Consequences, which are permanent:
- No prefill change here can be validated by exact output comparison. The only
  valid instrument is many independent single-forward-pass prompts with the
  reference compared against ITSELF as the ceiling.
- Two quality instruments were built and thrown away first: greedy agreement
  over 24 tokens (invalid -- one flip changes every later token; read 57% and
  looked catastrophic while `count` was 24/24 at delta 0.006), and a 6-prompt
  top-1 delta (underpowered -- the baseline's own spread was the same size as
  the effect).
- Benchmark matrices on this box will not produce bit-identical reruns. Treat
  per-cell deltas under ~10% argmax / ~0.13 nats as noise.

### Operational note that cost two servers today

Both server deaths today were `logind` reaping user processes on session
logout, NOT crashes: graceful `Process manager: send sigterm to process
EngineCore`, 1h46m after the last request, no CUDA/SYCL error, no OOM.
`setsid nohup` does not survive it. Any multi-hour run needs
`loginctl enable-linger` first, or it dies at logout with partial results.

## Bench 27 - r15 concurrent technical batch + the reasoning-channel trap (2026-08-22)

Six systems-engineering questions (dma-buf/BAR1, CUDA graph capture, fp16
overflow, MoE read amplification, ASPM link reporting, pinned memory), streamed
at C=6 on grouped + fp16 wire. Questions chosen because today's work
established ground truth for each, so answers can be GRADED not eyeballed.

### Serving metrics, C=6 concurrent, streamed

| metric | value |
|---|---|
| TTFT | 0.28-0.31 s (mean 0.30) |
| ITL p50 | 27.25 ms |
| ITL p99 | 161 ms |
| per-stream decode | 29-30 tok/s |
| aggregate throughput | 138.9 tok/s |
| wall | 17.0 s for 2,363 output tokens |

Against C=1 (ITL 11.40 ms, ~88 tok/s single stream): C=6 buys **1.58x
aggregate** at **2.4x per-stream ITL**. Concurrency scales, sub-linearly.

### Answer quality, graded against measurements taken today

- **fp16 overflow: correct, and it reproduced our actual bug from first
  principles.** `3072 x 1e6 = 3.072e9`, `x 8.7e-05 = 267,264 > 65,504` ->
  Inf, then `Inf - Inf = NaN` in the SwiGLU. That is exactly Bench 25's fault.
  Its proposed fix (clamp before store) is inferior to ours (widen the store):
  clamping preserves the overflow as a saturated value, silently wrong. It does
  flag that the alternative changes numerics.
- **MoE read amplification: exact.** Derived `2560/85 = 30.1x`. Our own probe
  printed `amplification: 30.1x`.
- **CUDA graph capture: weak.** Hedged ("may not be fully deterministic") and
  padded its remedy list. The real constraint is that capture forbids
  operations requiring host synchronisation; `cudaMemcpyAsync` from PINNED
  memory is capturable.

### The trap: thinking cannot be turned off here

`chat_template_kwargs` accepts neither `{"thinking": false}` nor
`{"enable_thinking": false}` -- reasoning is emitted regardless:

| request | finish | reasoning | content |
|---|---|---|---|
| no kwarg, 500 tok | length | 1,682 | **0** |
| `thinking=False`, 500 | length | 1,657 | **0** |
| `enable_thinking=False`, 500 | length | 1,756 | **0** |
| `enable_thinking=False`, 1200 | **stop** | 2,154 | **0** |
| `thinking=False`, 1600 | length | 5,244 | **0** |

For some prompts the model reasons and then **ends its turn without answering**
(`finish=stop`, content empty). Deterministic, 4/4 on the ASPM prompt.

Consequences for benchmarking:
- Timing metrics stay VALID. Reasoning tokens are output tokens, so TTFT, ITL
  and throughput measure real work either way.
- "Did it answer" is NOT valid on `/v1/chat/completions` at small output caps.
- `/v1/completions` bypasses the reasoning channel entirely and is the right
  endpoint for any measurement that inspects text. Bench 26's 120-prompt
  quality sweep used it, so that result is unaffected.
- A bare `Q:/A:` completion prompt makes this checkpoint hallucinate
  multiple-choice scaffolding (`B: ... C: ... Answer: A`, `\boxed{A}`) -- an
  artefact of MCQ tuning, not a regression. Use the chat template for prose.

### Bench 27b - the fix for the reasoning channel (2026-08-22)

`chat_template_kwargs` cannot disable thinking on this deployment (Bench 27).
What works: render the chat template yourself and hand the model an **already
closed** thinking block. `poolside_v1` keys on `</think>`, so a pre-closed pair
leaves no reasoning channel to spend the budget in.

```python
s = tok.apply_chat_template([{"role": "user", "content": q}],
                            tokenize=False, add_generation_prompt=True)
prompt = s + "<think>\n\n</think>\n\n"      # then POST /v1/completions
```

4/4 answered, including the ASPM prompt that was deterministically empty 4/4
before. It is also strictly faster, because no tokens are spent reasoning:

| | TTFT | ITL p50 | ITL p99 | tok/s/stream | aggregate |
|---|---|---|---|---|---|
| C=4, forced non-thinking | 0.29-0.32 s | **18.55 ms** | 118.9 ms | **44.1** | **170.5 tok/s** |
| C=6, reasoning on | 0.30 s | 27.25 ms | 161 ms | 29.9 | 138.9 tok/s |
| C=1 | 0.03 s | 11.40 ms | 13.2 ms | ~88 | - |

ITL scales 11.40 -> 18.55 -> 27.25 ms across C=1/4/6: sub-linear, so
concurrency buys aggregate throughput.

### Answer quality, graded (not an A/B -- see caveat)

Correct: fp16 Inf-minus-Inf NaN derived from first principles with our exact
arithmetic; MoE read amplification `2560/85 = 30.1x` matching our probe; the
tile-scheduler question (dynamic atomic dequeue of tiles, not whole GEMMs).

Wrong: **ASPM.** The model accepts sysfs's 2.5 GT/s as real and invents
compression/caching to explain 6.4 GB/s. The real answer -- measured here today
-- is that ASPM downtrains the link at idle so sysfs reports the IDLE state,
and it upshifts under load. Weak: CUDA graph capture (hedged, padded remedies).

Pattern: strong where clean arithmetic decides it, weak where the answer
requires distrusting a reported number.

**Caveat, stated plainly:** these six prompts were NOT run against the
pre-grouped baseline, so this is the checkpoint's capability ceiling (15%
REAP-pruned), not evidence about our changes either way. The evidence about our
changes is Bench 26's 120-prompt sweep, which found no measurable difference.

## Bench 28 - full matrix on grouped+fp16, and why it needs tiering (2026-08-22)

Config under test: `GROUPED=1 MAX_BATCH=2048 OUT_FP16=1 SB_MNBT=2048 SB_MNS=6`,
the 430 us/token arm that cleared Bench 26's quality gate.

**First attempt died at 8/24 cells.** One long `matrix_runner` invocation cannot
finish here. Device USM on the B70s costs host RAM 1:1 (~48.7 GiB for the pair)
against 59.4 GiB total, leaving ~5 GiB, and an hour of GuideLLM consumed it
monotonically: 2.9G/1.2G swap at 02:54 -> 0.5G/0.0G at 03:34. At 03:45 server
and runner were both gone with **no error in either log** -- the server's last
line is a successful 200 OK. `loginctl enable-linger` was already on, so this
was memory pressure, not logind.

**Fix: `benchmarks/matrix_tiered.sh`** -- one context tier per server lifetime.
Proven to work; the reclaim is exact:

```
07:05:28  tier 1024 done, mem 8.9G
07:05:32  tier 4096: server stopped -> mem 55.5G   <- accumulation zeroed
```

Within a tier memory still bleeds (9.0 -> 5.3G over ~35 min at 32K); each tier
boundary returns it. `--skip-existing` makes a death cost one tier, not the run.

Observed pace, 3 profiles per context (`concurrent` sweeps C=1..6 internally):
~17 min/cell at 16K, ~80 min/cell at 65K. High-context cells dominate.

Note the inversion: memory is *steadier* at high context (7.3-7.8G at 65K vs
5.3G at 32K). Fewer concurrent requests fit in the 8.38 GiB KV pool, so there is
less GuideLLM churn -- the cells that stress KV hardest stress host RAM least.

**Resume with the identical command; completed cells are skipped:**
```bash
loginctl enable-linger "$USER"       # required, not optional
./benchmarks/matrix_tiered.sh        # detach with setsid nohup for long runs
./benchmarks/matrix_progress.sh "$PWD/bench-matrix/jota_r15_c6"
```

## Bench 29 - intra-dispatch pipelining: the plan, not yet built (2026-08-22)

**Not a measurement.** The design and its falsifiable prediction, written down
before building so the number can be checked against it rather than rationalised
after.

### Why it is the only lever left

At 430 us/token the budget decomposes (all three terms measured, not modelled):

| term | us/token | how it was measured |
|---|---:|---|
| PCIe transfer | 175 | 1.128 MB/token over both cards' shared 6.44 GB/s uplink |
| B70 compute | 187 | standalone full layer, 8.15 ms at M=2048, x47 / 2048 |
| everything else | 68 | residual: 430 - 175 - 187 |

They run **strictly in series** inside one dispatch: copy in, compute, copy out.
Two expensive resources, each idle while the other works.

```
today       [copy in][    compute    ][copy out]
pipelined   [copy A ][ compute A ][ compute B ]
                     [  copy B  ][ copy out A ][copy out B]
```

Prediction: `max(175, 187 x 1.03) + 68 = 261 us/token = 6.5x`. The 1.03 is the
grouped kernel's penalty for halving rows/expert (120 -> 60), interpolated from
the standalone 512/2048 points (2.361 / 8.154 ms per layer).

Note the ordering dependency: this only pays **because Bench 26's fp16 wire
already pushed transfer (175) below compute (187)**. At the old fp32 wire (262)
overlapping would merely have exposed the wire as the new ceiling, capping at
4.8x. The two changes are not independent and fp16 had to come first.

### Why cross-layer overlap is NOT available

Layer N+1's activations are layer N's output. Strictly sequential -- there is no
earlier work to hide the transfer behind. This is the difference from every
published MoE streaming design (see the FreeToken audit below): those stream
**weights**, which are input-independent and therefore overlappable across
layers. We stream **activations**, which are not. Hence *intra*-dispatch.

### Scope and the one real hazard

Provider-only: `src/phase1/b70_provider.cpp`. No ABI, no plugin, no bank, no
checkpoint. Flag-gated `SHOOTING_BRAKE_B70_PIPELINE=1`, default off, A/B against
430 us/token. Extra memory lands on the B70s (~550 MB -> ~1.1 GB of 7.5 GB
free), NOT the 5090, so KV and max_model_len are untouched -- unlike every other
memory lever tried today.

**Hazard:** both halves currently share one set of scratch buffers -- `hist`,
`offs`, `cursor`, `slot_row/exp/w`, `g_act/g_mid/g_gated`. Two chunks in flight
race on device-scope atomics into them. Each needs a per-chunk copy. Miss one
and it is silent garbage: the same failure class as the `slot_row` out-of-bounds
read and the fp16 `Inf - Inf` NaN, both of which reached a booted server today
and were caught only by the correctness gate. Build it alone, gate it, and do
not report a number that has not cleared `benchmarks/b70_ttft_smoke.py` plus the
120-prompt sweep.

## Bench 29b - vendor/FreeToken audit: what transfers and what does not

`vendor/FreeToken` advertises "full-layer double-buffered prefill streaming" and
a "bandwidth-adaptive CPU-GPU co-execution (q*) policy", so it was read before
building Bench 29.

**Not reusable.** `python/freetoken/moe/fused_nvfp4.py` is Triton on CUDA
(`_e2m1_lut`, `e4m3_kernel_view`, dequant inside the K-loop -- the same NVFP4
format and the same in-kernel dequant strategy as ours) but there is no
SYCL/Level-Zero path; the xpu/sycl grep hits are arch-detection strings only.
Our bottleneck is the Intel B70 leg. Their kernels cannot run on it.

**Their overlap solves a different leg.** `layers/moe.py:380` --
*"stream whole layers -- double-buffered behind the previous layer's GEMMs when
prefill_overlap is on"* -- moves **expert weights** host->GPU because the
experts do not fit in VRAM. Ours are already B70-resident, so that transfer does
not exist for us. Their pattern is cross-layer and legal precisely because
weights are input-independent; see Bench 29 for why ours cannot be.

**What does transfer, three things:**
1. **The design shape is confirmed by an independent implementation.** A
   `prefill_overlap` boolean flag, and `[2, num_experts, ...]` double-buffer
   *views* over one pre-allocated pool (`moe/offload_cache.py:240`) rather than
   separately allocated buffers. That view idiom is a cleaner answer to Bench
   29's per-chunk duplication hazard than allocating a second set: index a
   leading dim of 2 instead of maintaining two pointer sets.
2. **Elastic VRAM/KV rebalance without restart** (`engine/engine.py`,
   `scheduler/scheduler.py`). Directly relevant operationally: today every
   MAX_BATCH / MNBT / L step cost a ~100 s reboot, and Bench 20's L-sweep cost
   five. Worth reading before the next placement sweep.
3. **The `q*` policy is the same problem shape as our local/remote expert
   split** -- which Bench 26 measured as a weak lever (2.5-3%), so this is
   confirmation to keep NOT spending time there, not an invitation to.

### Bench 28b - `--skip-existing` is config-blind (2026-08-22)

The tiered matrix resumed over 8 cells banked by the first, dead attempt. Those
cells were verified to be the same config two ways, because the obvious way does
not work:

- `serve_jota_r15_dual.sh` **exports** its knobs rather than echoing them, so
  the server log cannot tell you what config produced a result. Proving it
  needed a live `/proc/<pid>/environ` snapshot.
- Independent confirmation from the data itself: TTFT is config-diagnostic. The
  earliest cell (ctx 1024, C=1, written 02:35) reads **0.469 s**, against
  measured references of 1.766 s pre-grouped, 0.630 s grouped-fp32-wire, and
  0.506 s grouped-fp16-wire. Unambiguous.

**The trap, for the next run:** `--skip-existing` keys on the output path only.
It has no idea which build or which flags produced a cell. Reuse the same
`--output-root` after a config change -- e.g. benchmarking Bench 29's pipelining
against this surface -- and it will skip all 24 cells and compare nothing, or
half-fill a root and present two configs as one result set. That reads as a
partial regression and is indistinguishable from a real one.

Rule: **one `--output-root` per config.** Name it after the arm
(`jota_r15_c6` = grouped + fp16 wire + MAX_BATCH 2048 + MNBT 512->2048). A new
arm gets a new root, and the two are diffed afterwards rather than merged.
