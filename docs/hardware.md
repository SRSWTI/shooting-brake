# Hardware and Topology Contract

## Purpose and status

This document defines the production hardware boundary, topology checks, transport baseline, and safety gates for the Shooting Brake design in [`architecture.md`](architecture.md). It follows the active plan in [`../plan.md`](../plan.md). It is a design and qualification contract, not a claim that the upstream-vLLM plus QuixiCore-XPU production path is implemented or qualified.

Evidence terms are strict:

- **Production target** is behavior required of the upstream vLLM 0.26+ integration.
- **Colibri reference evidence** is an observation from the existing native CUDA+B70 implementation. It proves a narrower transport, correctness, placement, or failure property; it is not a production-vLLM result.
- **Upstream claim** is a vendor or upstream-project property, not a measurement on this host.
- **Derived bound** is arithmetic from claims or observations, not achievable-performance evidence.

## Production topology and ownership

The supported production topology is one local PCIe host containing:

- one NVIDIA RTX 5090, owned exclusively by the upstream vLLM CUDA worker;
- one Intel B70, owned exclusively by one isolated persistent QuixiCore-XPU provider process;
- host CPU and memory for orchestration, the fixed pinned-memory request ring, telemetry, and exact emergency recovery only; and
- NVMe for model artifacts, startup, provider restart, and recovery, never ordinary foreground expert execution.

The CUDA worker owns serving, scheduling, continuous batching, attention, KV and recurrent state, router logits, canonical top-k, dense and shared paths, hot routed experts, residual state, sampling, and request state. The B70 provider owns only its cold/overflow routed-expert weights, stable XPU buffers, XPU streams, qualified kernels, and request execution. It never owns router/top-k, shared experts, attention, KV, sampling, or serving.

These are separate memory and bandwidth domains. The RTX 5090 and B70 are not unified VRAM, and their peak bandwidths must not be added as if every byte were locally accessible. The CPU is not a normal expert tier and performs no normal-path matrix multiplication. See [`memory.md`](memory.md) and [`placement.md`](placement.md).

CUDA and Intel dependencies remain process-isolated. The versioned pinned-memory protocol is the compatibility boundary between the upstream-vLLM CUDA state owner and the separately pinned QuixiCore-XPU B70 provider environment. A qualified secondary llm-scaler INT4 provider environment remains independently pinned behind the same boundary.

## Required physical-topology audit

Before production performance claims, record the following for the RTX 5090 and the single B70, identified by stable PCI address:

| Evidence | Required result |
|---|---|
| Link | Negotiated PCIe generation and width at idle and under transfer load |
| Path | Root port, intervening switch, and relationship between the RTX 5090, B70, host memory, and NVMe |
| Locality | NUMA node for each device and the pinned ring |
| Addressing | ReBAR state and observed IOMMU/ACS behavior |
| Capacity | Actual allocatable VRAM after runtime reservations, not nominal board capacity |
| Transfers | Sustained CUDA↔pinned-host and pinned-host↔B70 bandwidth plus p50/p95/p99 latency |
| Concurrency | Independent DMA and expert compute while the CUDA state path and B70 provider are active |
| Stability | Errors, retries, queue stalls, stale completions, and completion correctness during a soak |
| Environment | Exact CUDA, NVIDIA driver, QuixiCore-XPU, oneAPI, Level Zero, Intel driver, and selected binding/runtime versions; when the secondary INT4 path is exercised, also llm-scaler and `vllm-xpu-kernels` versions |
| Operating envelope | Temperatures, clocks, power state, throttling indicators, and steady-state duration |

Enumeration or a topology diagram is descriptive only. It does not establish negotiated width, NUMA locality, DMA overlap, tail latency, or stable simultaneous execution.

### Suspicious B70 link reference

Repository notes contain this historical observation:

```text
B70 sysfs: 2.5 GT/s ×1 current and maximum
```

Treat it as **Colibri/reference evidence requiring reproduction and interpretation**, not as the qualified production link. Record the B70 PCI address and slot and measure the link under load. If it remains at 2.5 GT/s ×1, treat the path as Gen1 ×1 and stop production integration benchmarking until topology, firmware, or slot configuration is corrected. If it retrains, retain both idle and loaded evidence.

## Production transport

Direct NVIDIA-to-Intel peer memory is not assumed. CUDA P2P does not establish CUDA-to-Level-Zero interoperability, handle sharing, synchronization, or completion ordering. The production baseline is a fixed, bounded pinned-host shared-memory ring:

```text
upstream vLLM CUDA worker
    ↕ fixed versioned pinned-memory request/completion ring
isolated persistent QuixiCore-XPU B70 provider
    └── qualified QuixiCore-XPU tiny/batched/prefill NVFP4 MoE kernels
```

The ring must:

- be allocated and capacity-negotiated at startup;
- contain multiple reusable slots with stable addresses;
- hold activation rows, canonical expert IDs and routing weights, route masks/maps, and one output row per transported token;
- use monotonically increasing request sequence, provider/weight generation, layer, placement generation, token count, route count, and buffer version;
- publish with explicit release/acquire ordering;
- apply bounded queues, backpressure, deadlines, and completion-safe slot reuse;
- use pre-created events, CUDA copy streams, XPU streams, and stable provider tensors;
- perform no hot-path allocation, `.item()`, device-wide synchronization, weight transfer, or quantization; and
- send no request for a layer/step with zero B70-owned routes.

A representative slot lifecycle is:

```text
FREE
  -> CUDA_WRITING
  -> REQUEST_READY
  -> XPU_RUNNING
  -> RESPONSE_READY
  -> CUDA_READING
  -> FREE
```

`REQUEST_READY` cannot publish before CUDA D2H completion. `RESPONSE_READY` cannot publish before the provider output is visible to the CUDA-side process. A completion is accepted only when sequence, slot, layer, placement generation, weight/provider generation, token count, and buffer version match. A timeout or invalid completion identifies exact failed token-route entries for exact CUDA/CPU recovery or explicit request failure.

Pinned staging is the required baseline, not proof of adequate performance. Ordinary pipes, JSON, protobuf, and gRPC are not token-path transports. Direct peer transport may replace pinned staging only after equivalent ordering, lifecycle, failure, output-agreement, and tail-latency evidence on this exact stack.

## Colibri reference evidence

The native worker in `colibri-variants/colibri-qwen36/c/b70_moe_sycl.cpp` has already demonstrated, for the proven Colibri GS64 path:

- persistent B70 expert weights and compact `(layer, expert) -> slot` ownership;
- Colibri-only signed-S4 GS64 conversion, FP16 activation staging, and ESIMD gate/up/SiLU/down execution;
- canonical selected IDs and routing weights supplied to B70;
- one routing-weighted hidden-size partial returned to the CUDA side;
- asynchronous issue/take separation;
- exact failed-route recomputation rather than silent contribution loss;
- numerical agreement with the CPU reference;
- end-to-end CUDA+B70 generation with zero normal-path CPU expert fallback; and
- controlled hybrid throughput close to the corresponding all-CUDA Colibri expert configuration.

Observed Colibri one-token B70 issue/take was roughly 56–100 µs per active MoE layer. This remains transport/reference evidence only. Phase 1 separately qualified the full-bank QuixiCore-XPU provider core across `M=1`, decode batches, and `M=128` prefill with fixed buffers. Phase 2 then measured the real process ring: warmed publication-to-observation p50 was 245.640, 292.517, 581.815, and 1,896.458 µs for sequential one-route `M=1`, `M=8`, `M=32`, and `M=128` requests, with unfiltered long-tail p99 reported in [`progress.md`](progress.md#phase-2-process-ring-evidence). Neither evidence proves upstream-vLLM integration, CUDA/B70 overlap, graph behavior, production throughput, or production tail latency.

The B70's 608 GB/s specification is an upstream claim. For an ideal 18 MiB W4 expert it implies a roughly 31 µs weight-read lower bound, but that omits scales, dequantization, launch, intermediates, imbalance, transport, and join. It must not be reported as measured latency.

## Qualification matrix

Preserve raw samples sufficient for p50/p95/p99 and record topology, runtime versions, payload, queue depth, concurrency, clocks, power, and thermal state.

| Path or primitive | Required cases | Required evidence |
|---|---|---|
| CUDA → pinned host | compact activation rows `[M_remote, hidden]` drawn from full scheduler batches `M=1`, `2..32`, larger decode, and prefill | bandwidth, p50/p95/p99, copy-stream overlap, no device-wide synchronization |
| Pinned host → B70 and reverse | the same active `[M_remote, hidden]` rows plus route metadata and one `[M_remote, hidden]` wire partial | negotiated link under load, NUMA placement, p50/p95/p99, provider copy time |
| Empty fixed-ring round trip | queue depths and slots through wraparound/backpressure | publication/completion time, stale-reply rejection, bounded CPU consumption |
| Ring plus QuixiCore-XPU provider | tiny, small/grouped decode, and grouped prefill NVFP4 MoE shapes | queue, copies, kernel, total remote path, exposed CUDA wait, errors/stalls |
| Concurrent CUDA+B70 layer | local CUDA routed/shared work overlapping the one B70 operation | branch overlap, critical path, join time, output agreement |
| Recovery and restart | timeout, invalid generation, provider loss/restart, cancellation, work in flight | no stale acceptance or silent contribution loss; exact recovery or explicit failure |
| NVMe artifact activity | startup/restart and separately induced background reads | foreground interference and confirmation of zero ordinary decode reads |

## Safety and admission gates

1. **Topology:** a B70 that remains Gen1 ×1 under load blocks cross-device production benchmarking.
2. **Runtime isolation:** CUDA and XPU ownership must remain in separate pinned environments/processes; startup fails closed on device, protocol, manifest, capability, shape, dtype, layout, group-size, or generation mismatch.
3. **Ring safety:** wraparound, backpressure, cancellation, simultaneous completion, stale replies, provider restart, and shutdown with work in flight must not reuse a slot early or accept an invalid partial.
4. **Steady-state allocation:** no weights or scratch tensors are allocated, uploaded, converted, or migrated during dispatch.
5. **Failure semantics:** B70 loss must never silently omit a selected expert. Exact failed routes are recomputed on an available CUDA/CPU correctness path or the request fails explicitly.
6. **Storage path:** ordinary warmed decode performs no synchronous NVMe expert read.
7. **Thermal and power stability:** performance evidence is inadmissible if it hides throttling, unstable clocks, device errors, or an unreported power limit.
8. **Production evidence:** acceptance is based on the controlled comparison with stock all-CUDA upstream vLLM in [`benchmarking.md`](benchmarking.md), not on Colibri timing or theoretical bandwidth.
