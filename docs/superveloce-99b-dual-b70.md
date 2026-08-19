# Superveloce 99B on Dual B70

**Supersedes `docs/122b.md`** (deleted 2026-08-19). The 99B is the 122B with
~20% of experts REAP-pruned, so the model contract, quantization layout, MTP
facts and hard prohibitions carried forward here verbatim — only the expert
count changed. Everything else in this document is new, measured on
2026-08-19, and several entries **correct** numbers the 122B document and the
kill-bench were quoting.

Read this top to bottom before writing any dual-B70 code. The two sections
that will save you the most time are **"The decode cost model"** (it changes
how you size everything) and **"Killed — do not retry"**.

---

## 1. Where we are

| | |
|---|---|
| Serving today | 88B, one B70, `split:54` |
| Target | 99B (`srswti/axe-superveloce-99b-nvfp4`), two B70s |
| 99B status | **FIRST BOOT PASSED (2026-08-19, eager, dual-B70)**: correct greedy tokens, finite logprobs, 3,599 doorbell dispatches across both cards, KV 14.04 GiB = 341,723 tokens. Bank: `expert_bank_99b.bin` (monolithic SBEXP001, byte-exact, Bench 15/16). Recipe: `benchmarks/serve_99b_dual.sh` — requires `--moe-backend cutlass` + FlashInfer skip-ops (Bench 16, sm_120 findings). Not yet: graph mode, ITL/TTFT numbers, per-card GB/s. |
| Dual-B70 status | **code complete and hardware-gated**: per-lane dispatch/provider/poller landed, Bench 15 PASSED both formats, Bench 16 first serve passed. |

### Measured production baseline — quote these, not older numbers

88B, `split:54`, all 48 layers B70-active, Gen4 B70, `--kv-cache-memory=2.9e9`
(209,715 KV tokens), `BANK_REGISTER=1`, unprofiled, warmed past the
pin-eviction window, clocks sustained at 2800 MHz, 4 runs:

| metric | value |
|---|---:|
| **ITL** | **11.51 ms** |
| **TTFT** | **0.2915 s** |
| service per dispatch (host clock, p50) | 82.54 µs |
| gap between dispatches (p50) | 142.10 µs |
| dispatches per token | 48 |

Step reconstructs as `48 × (90.4 µs service + 151.8 µs gap) = 11.63 ms`
against 11.51 ms measured — under 1% error, so the model below is sound.

> **The 12.35 ms/step figure in `decode_overlap_trace.json` carries
> torch-profiler overhead. Do not quote it.** Every percentage in this
> document is against **11.51 ms**.

---

## 2. The decode cost model — read this before sizing anything

This is the single most useful result from 2026-08-19. One formula explains
the whole B70 leg:

```
dispatch_time  =  15.4 µs overhead  +  (distinct experts × 4.866 MB) / achieved_bandwidth
```

At M=1 decode on the 88B:

| term | value |
|---|---:|
| remote routes per layer (k) | 5.6 (54 of 180 experts are local on the 5090) |
| weight bytes read | **27.25 MB** |
| achieved bandwidth at M=1 | 406.08 GB/s |
| ⇒ kernel | **67.10 µs** |
| overhead | **15.44 µs** |
| **total** | **82.54 µs** ✓ matches measurement |

**The dispatch is ~81% memory-bandwidth-bound on expert weights.** Overhead is
15.4 µs and fully accounted for (an independent probe measured the floor at
12.92 µs).

The model *predicts* rather than fits — using only the M=1 overhead constant it
recovers correct distinct-expert counts at other batch sizes, with route
overlap rising monotonically exactly as real MoE routing does:

| M | measured | implied distinct experts | of routes | overlap |
|---|---:|---:|---:|---:|
| 2 | 122.02 µs | 8.9 | 11.2 | 21% |
| 4 | 210.81 µs | 16.3 | 22.4 | 27% |
| 8 | 347.56 µs | 27.7 | 44.8 | 38% |

**Consequence: to speed up decode you must either move fewer weight bytes, or
move them faster. Nothing else in the dispatch is big enough to matter.**
Dual-B70 is the first lever; it halves the bytes per card.

---

## 3. Hardware facts (verified 2026-08-19)

### Which B70 is which — this was never written down before

| L0 index | PCI BDF | measured H2D | link | production uses |
|---|---|---:|---|---|
| 0 | `0000:11:00.0` | 3.23 GB/s | **Gen3 x4** | no |
| 1 | `0000:15:00.0` | **6.46 GB/s** | **Gen4 x4** | **yes** |

`serve_88b_128k.sh` sets `ZE_AFFINITY_MASK=1` / `SHOOTING_BRAKE_B70_DEVICE=1`,
so today's serving is already on the fast card. **Verify with
`experiments/b70_pcie_bw`, never with sysfs** — `max_link_speed` reports the
ASPM-downtrained `2.5 GT/s x1` for both cards and is useless here.

> **Caveat this raises:** `window_decomposition.json` records its instrument as
> "Gen3 B70". Its 61 µs-fixed / 10.1 µs-per-expert split may be from the slow
> card and should be re-measured on index 1 before being trusted.

### Topology — deeper than previously documented

```
RTX 5090   0000:01:00.0   Gen5 x16, own CPU root port (00:01.1)
B70 Gen4   0000:15:00.0   \  both behind a THREE-level chipset chain:
B70 Gen3   0000:11:00.0   /  00:02.1 → 03:00.0 → 09:00.0 → 0f:00.0 / 13:00.0
                             sharing one Gen4 x4 chipset uplink
```

The old "one root complex, no PCIe switch" claim was **false**. Measured
aggregate under concurrent bulk traffic: **4.53 GB/s** shared between both
cards.

**Does the shared uplink bind for dual-B70 decode? No.** Per token, two cards
move roughly `48 × 2 × (6 KiB in + 12 KiB out) ≈ 1.7 MB`, i.e. ~150 MB/s at
11.5 ms/token — two orders of magnitude under 4.53 GB/s. **It will bind on
bank streaming**, so keep experts resident in VRAM.

### Other verified facts

- **IOMMU is ON** (AMD-Vi `ivhd0`, all GPUs in `DMA-FQ` translating domains).
  The old "IOMMU off" note was false. No raw-physical-address peer DMA is
  expressible; every peer path needs a real IOVA mapping.
- **8 logical CPUs only** (`/proc/cpuinfo`), despite the 9950X3D nameplate.
  Matters because each B70 needs its own spinning poller thread.
- **The poller is not CPU-starved**: measured 3.5 ms run-queue wait across 67 s
  of CPU, 1,811 slices, pegged at 100.2%. It owns a core.

---

## 4. Dual-B70: what to expect

Each card owns half the remote experts, so each reads half the bytes, in
parallel:

| | today (1 card) | dual-B70 |
|---|---:|---:|
| remote experts touched per dispatch | 5.6 | 2.8 per card |
| weight bytes per card | 27.25 MB | **13.6 MB** |
| kernel | 67.1 µs | **~33.5 µs** |
| overhead | 15.4 µs | 15.4 µs |
| **dispatch** | **82.5 µs** | **~49 µs** |

**Expected: ~−1.6 ms/token, roughly −11% to −16% ITL** depending on how much of
the window overlaps CUDA work. `[estimate]` — derived from the measured cost
model, not yet observed.

### Why this is bigger than the old estimate

`window_decomposition.json` projected only **−5.9%**, because it assumed
"fixed 61 µs stays". **That 61 µs was inflated by a cross-clock measurement
error** (see §6). The real fixed cost is 15.4 µs, so far more of the dispatch
is halvable than that projection allowed.

### Two risks that would eat the win

1. **Serialized pollers.** If both cards share one poller thread they take
   turns: `2 × (15.4 + 33.5) ≈ 98 µs`, **worse than today's 82.5 µs.**
   **One poller thread per card is a correctness requirement, not a tuning
   choice.**
2. **Less work per card is more latency-bound.** At 2.8 experts instead of 5.6
   there is less memory parallelism, so achieved bandwidth may fall below
   406 GB/s and the kernel will not halve cleanly. **This is the main reason
   the estimate is a range.** Measure achieved GB/s per card in the first
   dual-card run.

---

## 5. What to build

### Already done — do not rewrite

`placement.py` fully supports multiple remote devices:
- `FractionalRemotePolicy(remote_device_indices=(0,1), cuda_fraction=…)`
  already splits experts across devices via
  `divmod(remote_n, len(remote_device_indices))`
- `Placement` carries `MULTI_DEVICE_SCHEMA = "shooting-brake.placement.v2"`,
  per-device `device_capacities`, and `remote_device_indices()`
- `policy_from_name` already parses `fractional-remote:devices=…`

### The five blockers — ALL FIXED 2026-08-19 (plus a sixth the list missed)

| # | was | fix that landed |
|---|---|---|
| 1 | `_build_b70_slot_map` raised if remote indices `!= (0,)` | per-device maps (`_build_b70_slot_map(placement, device_index)`), plus a separate **union map** (`_build_b70_union_slot_map`) for the monolithic prefill artifacts |
| 2 | `validate_int4_hybrid_contract` demanded exactly device 0 | per-device set equality against `SHOOTING_BRAKE_B70_BANKS`, cross-card overlap rejected |
| 3 | `_b70_provider_singleton` was one global | `_b70_providers` dict keyed by device index; selectors from `SHOOTING_BRAKE_B70_SELECTORS` (BDFs, mandatory for multi-card) |
| 4 | `_poller_singleton` was one global | `_pollers` dict; one native thread per card, optional pinning via `SHOOTING_BRAKE_B70_POLL_CPUS` |
| 5 | one poller thread would sweep all layers of all cards | one `B70Poller` instance per card's provider; `sb_b70_poll_start(poller, pin_cpu)` pins in the thread |
| 6 | (unlisted) the C ABI had no device selector | `sb_b70_load(..., const char* device_selector)`; empty keeps legacy first-device |

Per-layer state now lives on `_B70Lane` (routed_experts.py): slot map,
pinned staging, device result buffers, flags, poller handle — one lane per
card, sharing nothing mutable. Flag allocations are padded to 256 B at the
source (`stream_signal.alloc_host_mapped_flag`), so rule 3 below is now
structural rather than a convention.

### Bank contract — the trap that will bite first

`config.py:294` enforces **exact set equality** between placement and bank:

```cpp
if (b70_ids != bank_ids) -> QualificationError
```

A subset is rejected. **Each card needs its own bank file built for exactly the
expert IDs that card owns.** Changing the split means rebuilding banks
(~29 GB each for the 88B). Budget for this; it is half a day per arm and it is
why the Bench 12 local-expert sweep is still unrun.

### The clean way to structure it

**Do not add a `device_index` parameter to the existing singletons.** That is
the tempting shortcut and it produces a codebase where every call site has to
remember which card it is talking about. Build a **per-device object graph**
instead — one complete stack per card, sharing nothing mutable:

```
Placement  (already multi-device — do not rewrite)
    │
    ├─ B70Device[0]  ─ own SYCL context + in-order queue
    │                ─ own bank mmap (experts this card owns)
    │                ─ own provider          (libsb_b70_provider)
    │                ─ own poller THREAD     (pinned to its own core)
    │                └─ own (signal, completion) flag pair PER LAYER
    │
    └─ B70Device[1]  ─ ... identical, independent

5090 = the only hub. No B70↔B70 edge. No oneCCL. No peer memory.
```

Five rules that keep it clean:

1. **One object per card, not one object with a switch.** Replace
   `_b70_provider_singleton` and `_poller_singleton` with dicts keyed by
   device index. Construction order and teardown become obvious, and a
   single-card config is just a dict of length one — the existing behaviour
   falls out as a special case rather than a branch.
2. **Nothing mutable is shared between cards.** Each has its own queue, bank,
   flags and thread. The only shared object is the immutable `Placement`.
3. **Flags are per layer AND per device**, each on its own cache line. Bench 4
   learned this the hard way: two flags sharing a line let one side's write
   clobber the other's, and it presents as a random hang after O(100)
   round trips, not as an obvious bug.
4. **The CUDA graph waits on both completions.** `_b70_take_graph` currently
   parks on one `cuStreamWaitValue32`. With two cards it needs two, in
   sequence — the cards still execute in parallel, the stream just resumes
   after the slower one. Cost of the second wait is ~1.19 µs [measured].
5. **Split routes once, at partition time.** `partition.py` already separates
   CUDA / B70 / CPU routes. Extend it to bucket B70 routes by owning device
   so neither poller ever sees the other card's work.

### Keep one CUDA expert — this is not optional

`validate_cuda_dummy_slot_placement()` **rejects zero CUDA experts**: the
fused CUDA partial always executes and masks remote routes through a real
local dummy slot. So the lowest-risk first boot is *not* an even all-remote
split. Use **N−1 experts split across the two B70s and one expert local on
the 5090** (the 122B equivalent was `fractional:2:0.00390625`). It costs about
one expert across 48 layers and keeps the proven fused-CUDA path intact.
A true all-remote path needs a separate no-CUDA fast path — price it as
implementation work, do not assume it.

### What is now established — and what still is not

**Established** [`experiments/b70_multi_topology.cpp`]: one process can own
**two independent per-device SYCL contexts and queues**; the correct topology
is a 5090 hub with no B70↔B70 edge and no oneCCL; both cards share the
`09:00.0` Gen4 x4 uplink and degrade to 4.53 GB/s aggregate under concurrent
bulk traffic, but doorbell payloads are far too small for that to price
decode.

**Established 2026-08-19** [`experiments/b70_dual_card_smoke.py`, Bench 15,
`benchmarks/results/b70_gemv_audit/dual_card_smoke.json`]: two of the three
open risks are retired —

- **Concurrent two-card doorbell latency:** solo 73.7 µs (Gen4) / 72.8 µs
  (Gen3); both in flight **90.3 µs = 1.23× max-solo**, not 2×. The cards
  genuinely overlap; the +23% concurrency penalty is real, unattributed
  (single-stream staging copies are the lead suspect), and small enough to
  leave alone unless the first ITL lands short.
- **CUDA graph replay with two cards:** capture clean, 200 alternating
  replays, zero poller errors, partials ≤ 1.9e-6 vs the CPU oracle, and
  cross-card isolation exact in both directions.

**Still NOT established:** full-vLLM stability with two lanes, and per-card
achieved GB/s at the production per-card route count (~2.8 for the 99B) —
the Bench 14 latency-bound risk. Both wait on the first boot. One bring-up
hazard on record: a transient `std::bad_alloc` on the second ~28 GB
provider load (retry succeeded; suspected host commit pressure right after
the first card's stream).

### Suggested order — updated

1. ~~Per-device provider + poller~~ **done** (blockers 3, 4, 5, 6).
2. ~~Per-device slot map and validation~~ **done** (blockers 1, 2).
3. ~~Standalone two-card gate~~ **done — Bench 15 WORKED.**
4. Build two banks for the chosen 99B split (205 experts: N−1 remote
   split across the cards + 1 CUDA expert, per the rule above). Decide the
   source first: fresh extraction from the nvfp4 checkpoint, or slicing
   the existing 122B int4 banks if the REAP kept-expert mapping is
   recoverable (check `recipe.yaml` in the checkpoint; would save the
   half-day-per-arm build AND change the bank-format decision).
5. Boot, confirm correctness, then **measure achieved GB/s per card before
   celebrating any ITL number** — if it falls well below 406 GB/s the
   kernel has gone latency-bound and the win will not be what §4 projects.

---

## 6. Kill-bench: what WORKED, and where the failures live

`docs/kill-bench.md` is the full ledger — every hypothesis, method, kill
condition and verdict, including all the negative results with their numbers.
**Do not re-fight anything in it.** Summarised here is only what survived,
plus the instrument lessons that transfer.

### What survived and is worth having

| bench | verdict | what you get |
|---|---|---|
| **7 — FrequencyPolicy hit-rate curves** | **PARTIAL, method proven** | Hotness-ordered placement beats index-ordered by **1.79× at a 25% local tier, 1.44× at 50%** — clears the 1.3× kill line. ~40 lines of code, zero VRAM cost. **But measured on a foreign 40×256 capture; recompute on the 99B before shipping.** |
| **4 — device-side host-flag spin** | **PARTIAL** | Notify latency is a non-problem: **2.36 µs one-way, p99 ≤ 6.5 µs on the Gen4 card** — 7× inside the 20 µs kill line. Also confirmed the B70 advertises `COOPERATIVE_KERNELS`. Residency is the blocker, not latency. |
| **13 — poller path** | KILLED, but produced the model | The whole reason we now have the §2 cost model and a correct 11.51 ms baseline. The bench died; its instrumentation is the most valuable thing in the ledger. |
| **14 — GEMV bandwidth** | PENDING, sized | The one live decode lever besides dual-B70: −5.7% ITL. |

### Instrument lessons that will save you a day each

1. **Never subtract a device profiling timestamp from a host clock.** This
   manufactured an entire fictional 34 µs bench. Fit against a physical model
   (bytes ÷ bandwidth) instead — it explains the number *and* predicts its
   neighbours, which is how you know it is real.
2. **A synthetic probe's timing pattern is part of what it measures.** The
   spin-wait fix showed 3.09 µs in a probe and exactly 0.00 ms in production,
   because the probe submitted-then-waited immediately and production does not.
3. **Use the mean, not the median, for anything summed per step.** A step adds
   48 dispatches. On the blocking wait the mean ran 58% above the median
   because of a sleep tail; medians hid it completely.
4. **Never derive a marginal cost by dividing an average.** "7.29 µs for 3
   enqueues" is not "2.4 µs per enqueue" — the true marginal cost was
   **0.431 µs**, and the difference briefly made a dead bench look alive.
5. **Flags on separate cache lines**, always. See §5 rule 3.
6. **Check provenance before trusting a capture.** `route_stats.csv` looked
   authoritative and belongs to a model we do not run.

---

## 7. Killed — do not retry

Full detail and methods in `docs/kill-bench.md`. Each of these cost real time
to disprove; the negative result is the product.

| idea | why it died | measured |
|---|---|---|
| **The "34 µs of wasted software"** | **It never existed.** It came from `wall − device_total`, subtracting a copy-engine GPU timestamp from a host `CLOCK_MONOTONIC` span. `b70_capi.cpp:129-131` warns about exactly this. | overhead is 15.4 µs, fully accounted |
| Fused single H2D (old Bench 5) | marginal enqueue cost is tiny; two of the three copies are 32 bytes | 0.98 µs/dispatch vs a 2 µs kill line |
| Spin instead of blocking wait | works in a probe, does nothing in production | 11.5098 vs 11.5130 ms — noise |
| `cuStreamWaitValue32` too slow | it isn't | 1.19 µs |
| Poller's 48-flag sweep | trivial | 0.11 µs |
| CPU contention on the poller | it owns a core | 0.005% run-queue wait |
| Cross-vendor BAR P2P (Bench 8) | dma-buf export refused by driver policy; repriced from 25% to ~2% | `CUDA_ERROR_NOT_SUPPORTED` |
| Persistent-kernel doorbell (Bench 4) | latency is fine (2.36 µs one-way) but a resident kernel's view of host memory freezes after ~0.5 ms | not viable on this stack today |

**Rule learned the hard way: never subtract a device profiling timestamp from
a host clock. Fit against a physical model (bytes ÷ bandwidth) instead — it
both explains the number and predicts its neighbours.**

---

## 8. The 99B specifically

### Geometry (from `config.json`, authoritative)

| | 88B | **99B** | 122B |
|---|---:|---:|---:|
| layers | 48 | **48** | 48 |
| routed experts | 180 | **205** | 256 |
| experts/token | 8 | **8** | 8 |
| hidden | 3072 | **3072** | 3072 |
| MoE intermediate | 1024 | **1024** | 1024 |

Attention/GDN geometry is field-for-field identical across all three, so the
plugin's attention, KV and Marlin-prefill machinery transfers unchanged.

**Per-token work is identical across all three models** — 8 experts × 48
layers of the same shape. The 99B is not slower than the 88B; it is more
*storage* for the same per-token bandwidth.

### Why 99B is the right size — the systems argument

Expert bytes (NVFP4 ≈ 0.5625 B/param, 9,437,184 params/expert ≈ 5.31 MB):

| model | experts | total expert bytes | per B70 if split 2 ways |
|---|---:|---:|---:|
| 88B | 180 | 45.9 GB | 23.0 GB |
| **99B** | **205** | **52.3 GB** | **26.1 GB** ✅ |
| 122B | 256 | 65.3 GB | 32.7 GB ❌ over 32 GB |

**At 99B the experts fit in two B70s with ~6 GB headroom each. At 122B they do
not.** That is the strongest argument for the 99B and it is a systems argument,
not a quality one — it may remove the host-RAM warehouse tier and its cold-tier
streaming entirely. `[INFERENCE — confirm against the placement policy]`

### REAP notes

REAP's actual criterion (read from `vendor/llm-compressor`, not assumed):

```
S_j = mean( g_j × ‖f_j‖₂ )      # gate weight × expert output norm
```

It is a **mean**, so **frequency divides out**. REAP does *not* prune rarely
used experts — it prunes experts with the smallest average contribution
magnitude.

- A `count == 0` expert scores exactly 0 (`count.clamp(min=1.0)`) and is always
  pruned.
- An expert seen **once** has a single-sample saliency. With a narrow
  calibration set, decisions about the cold tail are made from 1–5 samples of
  the cold tail. Reference recipe is **512 diverse chat samples**
  (`ultrachat_200k`); `benchmarks/route_capture_prompts.jsonl` currently holds
  **10**, all technical.
- **Placement must rank on a different axis than REAP** — REAP on saliency,
  placement on route frequency. One calibration pass, two rankings.

### Route data warning

`benchmarks/results/route_stats.csv` is **40 layers × 256 experts** — matches
neither the 88B (48×180) nor the 99B (48×205). It is a **foreign capture**.
Any hit-rate curve derived from it does not transfer. Re-capture with
`SHOOTING_BRAKE_ROUTE_STATS=1` on the target model.

---

## 9. Hard prohibitions (carried forward, still binding)

- Do not call the 99B production-ready before a real two-card smoke and vLLM
  boot pass.
- **Do not use `ZE_AFFINITY_MASK` to select two cards inside one serving
  process.** Use explicit per-device contexts.
- Do not use enumeration indices in production configuration; use PCI BDFs.
- Do not introduce oneCCL, B70↔B70 collectives, or peer-memory traffic.
- Do not assume zero CUDA experts works; the fused CUDA partial needs a real
  dummy slot.
- Do not mix an NVFP4 prefill bank with an int4 decode bank in one request.
- Do not enable MTP before excluding the `mtp.` block from routed-expert
  surgery, and keep the BF16 MTP head.
- Do not pin a bank larger than host RAM can hold unevictably (63 GB box).
- Do not use four or more NVMe readers; throughput regresses beyond two.
- Do not treat transfer floors, KV estimates, or dual-card latency projections
  as measured serving results.
- **Do not install the aikitoria `open-gpu-kernel-modules` fork.** Three
  independent reasons: its NV↔NV P2P patch is dead weight on a 1-CUDA-GPU box;
  the dma-buf gate is byte-identical across versions so it opens nothing; and
  its forced-PCIe mode trips `if (mem_info->force_pcie) return -ENOTSUPP;`,
  disabling the `nvidia_p2p_*` API that any future peer work would need.

---

## 10. What you do NOT need to know

Skip these; they are closed or irrelevant to dual-B70:

- **Cross-vendor BAR P2P / dma-buf / kernel modules** — closed, ~2% at best.
- **The poller's internals** — measured clean from five directions.
- **Fusing H2D copies** — killed, 0.98 µs.
- **Blocking vs spinning waits** — killed, zero effect. Code is behind
  `SHOOTING_BRAKE_B70_SPIN_WAIT`, defaulting off.
- **`window_decomposition.json`'s 61 µs / 10.1 µs split** — built on the
  cross-clock error and possibly measured on the Gen3 card. Superseded by §2.
- **`route_stats.csv`** — foreign model.
- **The 122B** — superseded by the 99B unless RAM is upgraded.

---

## 11. Open, sized, not started

| item | size | notes |
|---|---:|---|
| **Dual-B70 split** | **−11 to −16% ITL** | this document |
| **Pool B** (host/scheduler GPU-idle) | **1.92 ms ≈ 16%** | **completely untouched**; kill-bench 1, 2, 3 are cheap flag flips |
| B70 GEMV bandwidth at M=1 (kill-bench 14) | −5.7% | kernel hits 510 GB/s at M≥2 but only 406 at M=1 — latency-bound, not bandwidth-bound |
| 5090 local-expert budget (kill-bench 12) | ~−3.8% | blocked on a real route capture and a bank rebuild |
| MTP speculative decode | large, unquantified | 785 `mtp.*` tensors ship in the checkpoint |

**Pool B is the cheapest unclaimed money after dual-B70** — 1.92 ms per step
where the GPU does nothing at all, and nobody has looked at it yet.
