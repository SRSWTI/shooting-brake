# Hardware and Topology Contract

## Purpose and status

This document defines the hardware assumptions, topology audit, transport baseline, and admission gates for the Shooting Brake design described in [`architecture.md`](architecture.md). It is a design and measurement contract, not evidence that the target topology or cross-vendor transport has already been validated.

The terms below are deliberate:

- **Observed report** means a value recorded by the current repository notes. It still requires reproduction and interpretation.
- **Upstream claim** means a vendor specification or guidance, not a measurement on this workstation.
- **Design target** means the topology or behavior the runtime is intended to support.
- **Unverified assumption** means the design may use the assumption for planning, but must not present it as a hardware fact until the audit below supplies evidence.

## Target topology

The design target is one local PCIe host containing:

- one NVIDIA RTX 5090 as the state-owning GPU;
- four Intel B70 cards as separately addressed resident-expert workers;
- the host CPU and DDR5 as the exact CPU-fallback tier for weights that fit its configured memory budget;
- NVMe as the full model repository and as a startup, recovery, and background-staging tier.

The RTX 5090 remains authoritative for the sequential residual stream, attention and KV state, router and canonical top-k selection, dense layers, shared experts, local experts, sampling, and request state. Each B70 owns only its resident expert arena, command queues, and request execution. See [`memory.md`](memory.md) for the corresponding memory planes and ownership rules.

These are separate memory and bandwidth domains. Four B70s and one RTX 5090 are **not** 160 GB of unified VRAM, and the B70s are **not** one bandwidth-coherent 2.4 TB/s device. Capacity or vendor peak-bandwidth sums must not be used as if every byte were locally accessible by every processor.

The repository describes the current desktop host as a Core Ultra system without the server-class AMX/AVX-512 environment assumed by some CPU-MoE results. Its immediate CPU baseline is therefore the existing AVX2/AVX-VNNI path; a future Xeon/AMX host would be a different measured configuration, not evidence for this one.

## Required physical-topology audit

Before cross-device performance conclusions, capture the following for the RTX 5090 and for **each** B70, identified by stable PCI address:

| Evidence | Required result |
|---|---|
| Link | Negotiated PCIe generation and width at idle and under transfer load |
| Path | Root port, any intervening switch, and the relationship between all five GPUs and NVMe |
| Locality | NUMA node for the device and for its pinned host rings |
| Addressing | ReBAR state and observed IOMMU/ACS behavior |
| Capacity | Actual allocatable VRAM, not nominal board capacity |
| Transfers | Sustained GPU-to-pinned-host and pinned-host-to-GPU bandwidth |
| Contention | Transfer results alone, with concurrent RTX 5090 copies, with concurrent NVMe activity, and under the intended combined load |
| Concurrency | Independent DMA and expert compute while all five GPUs are active |
| Stability | Errors, retries, queue stalls, and completion correctness during sustained load |
| Environment | Exact GPU driver versions and the runtime/API versions used for the measurement |
| Operating envelope | Temperatures, clocks, power state, throttling indicators, and whether results are steady-state |

A topology diagram without these per-slot results is descriptive only. A device enumeration is not proof of negotiated width, NUMA locality, concurrent DMA stability, or usable transport latency.

### Suspicious B70 link report

The repository records this **observed report**:

```text
B70 sysfs: 2.5 GT/s ×1 current and maximum
```

This line must not be silently treated either as the verified topology or as a harmless idle-link power state. Reproduce it per B70, identify the PCI address and slot, and measure the negotiated link under load. If the card remains at 2.5 GT/s ×1 under load, treat it as genuinely Gen1 ×1 for the gate below. If it retrains, retain both the idle and loaded evidence and explain the transition.

## Cross-vendor transport baseline

Direct NVIDIA-to-Intel peer memory access is **not established**. CUDA P2P is not a general CUDA-to-Level-Zero interoperability guarantee; permissive PCIe topology or IOMMU settings alone do not prove driver support, shareable memory handles, synchronization semantics, or correct completion ordering.

The required baseline is host-pinned staging:

- allocate one preallocated inbound ring and one preallocated outbound ring per B70;
- place those rings on the CPU/NUMA node closest to that B70, subject to audit rather than assumption;
- use bounded, double- or triple-buffered queues with no per-token allocation and no global mutex on the token path;
- use monotonic sequence information, explicit deadlines, and completion-safe slot reuse;
- keep stable activation/result buffers, pre-created events and kernels, and pre-bound weight arenas;
- use immediate command submission where the measured Intel runtime supports it; do not rebuild pipelines or descriptors per sparse layer.

Intel guidance favoring immediate command lists is an **upstream claim**. Low submission overhead on these B70s remains a measurement requirement. The repository's approximately 0.8 ms synchronous-submit precedent is a warning about the class of overhead to eliminate, not a measured result for the proposed B70 worker.

The first performance proof should use the Colibri C runtime, CUDA backend, and an Intel SYCL/DPC++ or Level Zero module behind a C ABI in one process, so IPC is not confounded with the primitive. Worker-process isolation is a later operational choice and may be adopted only after the empty-ring round trip has been measured. Even then, transport remains shared pinned rings—not JSON, protobuf, gRPC, Python, or ordinary pipes.

### Publication and reuse rules

The later revised protocol is authoritative. A slot follows:

```text
FREE
  -> CUDA_WRITING
  -> REQUEST_READY
  -> XPU_RUNNING
  -> RESPONSE_READY
  -> CUDA_READING
  -> FREE
```

Publication uses release/acquire ordering. The request cannot become `REQUEST_READY` before CUDA device-to-host completion, and the response cannot become `RESPONSE_READY` before the B70 output transfer completes. A completion is rejected unless request ID, generation, placement epoch, layer, token count, and buffer version all match. These checks prevent a late B70 result from entering a later token or placement epoch.

Pinned staging is a baseline, not proof that it meets the token-path budget. NUMA locality, page locking, DMA overlap, and stable concurrency are **unverified assumptions** until the matrix below has been run on every slot.

## Bandwidth and latency evidence matrix

Measure every B70 independently and all B70s concurrently. Preserve raw samples or traces sufficient to reproduce p50, p95, and p99; an average alone is insufficient.

| Path or primitive | Payloads | Required concurrency variants | Required evidence |
|---|---:|---|---|
| Pinned host → B70 | bandwidth sweep plus 12, 24, 48, 96 KiB latency points | isolated; concurrent RTX 5090 copy; concurrent NVMe; intended combined load | negotiated link under load, throughput, p50/p95/p99, NUMA placement, temperature/clocks |
| B70 → pinned host | bandwidth sweep plus 12, 24, 48, 96 KiB latency points | isolated; concurrent RTX 5090 copy; concurrent NVMe; intended combined load | negotiated link under load, throughput, p50/p95/p99, NUMA placement, temperature/clocks |
| RTX 5090 → pinned ring → B70 → pinned ring → RTX 5090 | empty-ring round trip and representative activation/result sizes | isolated; all four workers active; NVMe background activity | end-to-end p50/p95/p99, queue wait, direction-specific copies, publication/completion timing |
| B70 DMA plus expert compute | representative routed-expert shapes | one worker; all four B70s plus RTX 5090 active | overlap, service-time distribution, errors/stalls, sustained clocks and throttling |
| NVMe background staging | representative background reads | alone; concurrent GPU copies and compute | GPU-transfer and tail-latency interference; confirmation that foreground tokens do not issue storage reads |

The measurement record must identify the card and slot, root path, NUMA placement, ReBAR/IOMMU/ACS state, driver/runtime versions, power and thermal state, test duration, payload shape, queue depth, and concurrency mode. Results from one B70 or an idle system do not establish four-card behavior.

No transport claim may be made from topology inspection alone. In particular, NVIDIA–Intel P2P may be described as supported only after direct evidence establishes all of the following on this exact software and hardware configuration:

1. mutually usable allocation or memory-handle semantics;
2. correct synchronization and publication ordering in both directions;
3. stale-completion and slot-reuse correctness under concurrency;
4. per-size p50/p95/p99 latency and sustained bandwidth;
5. stable simultaneous operation with all five GPUs and NVMe contention;
6. driver/runtime versions and thermal/throttling evidence for the run;
7. end-to-end output agreement against the host-pinned baseline.

Absent that evidence, host-pinned staging remains the production baseline.

## Performance bounds are not measurements

An ideal W4 expert is estimated to read about 18 MiB. Dividing by the B70's **upstream-claimed** 608 GB/s bandwidth gives an approximately 31 µs perfect weight-read lower bound; two balanced experts give approximately 62 µs. These figures exclude group scales, dequantization, XMX utilization losses, intermediates, elementwise work, launch cost, and routing imbalance.

The resulting roughly 100–160 µs sparse-layer design budget is plausible only with balanced work, persistent or immediate submission, overlapped transport, and no per-layer host synchronization. It remains unproven until a standalone expert benchmark and the transport matrix exist. Marketing TOPS or peak bandwidth must not be substituted for batch-one MoE latency evidence.

## Hard gates and fallback decisions

The following gates are normative:

1. **Topology gate:** if a B70 is genuinely Gen1 ×1 under load, stop architecture benchmarking and fix topology, firmware, or slot configuration first. Device-local kernel speed cannot compensate for a broken host link.
2. **Five-GPU stability gate:** every B70 must sustain stable independent DMA and compute under simultaneous five-GPU load before the topology is admitted for serving experiments.
3. **Foreground transport gate:** if empty cross-vendor round-trip p99 is too high for the foreground latency path, batch more aggressively, restrict B70s to prefill/background work, and avoid mandatory B70 participation in foreground batch-one decode. Do not hide the result behind device-local benchmarks.
4. **Storage-path gate:** ordinary foreground decode must never synchronously read an expert from NVMe. NVMe is limited to startup, recovery, frozen formats, and optional background preload. A configuration that requires foreground storage reads is not robust interactive serving.

Large batch-one decode gains, low CUDA-to-Level-Zero p99, and stable four-B70 submission on this motherboard are plausible but unproven. The audit and matrix above are the evidence required to change that status.
