# Shooting Brake Benchmarking Contract

## Purpose and status

This document defines evidence, benchmark configurations, metrics, and production acceptance for the one-RTX-5090 plus one-isolated-B70 design in [`../plan.md`](../plan.md). Correctness requirements are detailed in [`correctness.md`](correctness.md); topology and scheduling contracts are in [`hardware.md`](hardware.md) and [`scheduling.md`](scheduling.md).

This is a measurement contract, not a claim that the upstream-vLLM 0.26+ `HybridMoERunner`/`HybridRoutedExperts` adapter or batched PyTorch-XPU/llm-scaler provider has been implemented or qualified.

## Evidence classes

- **Production measurement:** reproduced with the pinned upstream-vLLM CUDA state owner, isolated llm-scaler provider, versioned ring, qualified model/provider manifest, and controlled workload in this contract.
- **Colibri reference evidence:** observed with the proven native GS64 Colibri path. It can establish a narrower transport, correctness, placement, or failure baseline but cannot be presented as production-vLLM performance.
- **Upstream claim:** a vendor/project specification or code property, not local performance evidence.
- **Derived bound:** arithmetic based on a claim or observation, not measured achievable latency.
- **Design target:** an experiment or gate that remains unproven until its production evidence is recorded.

## Normative measurement rules

1. The primary performance baseline is stock all-CUDA upstream vLLM at the exact pinned commit and supported eager/graph mode, not older Colibri timing.
2. Compare the identical source checkpoint, prompt matrix, scheduler settings, context lengths, output counts, request-order rotations, and compatible quantization semantics.
3. Report median and range across at least three interleaved runs. Preserve raw per-request/per-layer records and sample counts; report p50/p95/p99 where applicable.
4. Separate prefill from decode, `M=1` from `M=2..32` and larger continuous batches, cold/first-call from warmed steady state, and device-local from transport, complete remote path, and end-to-end service.
5. Record exact hardware topology, negotiated links under load, clocks, power, thermals, drivers, runtime commits, provider bundle, protocol, artifacts, placement, and generations.
6. Include failures, fallbacks, cancellations, stale/discarded completions, and divergent outputs. Never remove them silently from timing summaries.
7. Correctness is a separate linked artifact. A checksum in a timing record correlates evidence but does not itself prove correctness.
8. Report capacity gained together with throughput, TTFT, ITL, and prefill cost. Do not sum nominal device capacities or call them unified VRAM.
9. CPU expert matrix compute and foreground NVMe reads must be zero on the healthy hybrid normal path. Any recovery or storage activity is separately counted and explained.
10. Isolated bandwidth, TFLOPS, GEMM latency, or Colibri issue/take timing cannot establish production benefit.

## Frozen run manifest

Every comparable result records:

- run ID, timestamp, repetition/rotation position, warmup, sample count, and exclusions;
- upstream vLLM commit and eager/PIECEWISE/full-graph mode as applicable;
- PyTorch XPU, llm-scaler, `vllm-xpu-kernels`, oneAPI, Level Zero, CUDA, and driver versions;
- model architecture, source checkpoint identity, and CUDA/B70 artifact fingerprints;
- provider protocol and kernel-bundle versions;
- dimensions, top-k, activation/output dtype, quantization/group-size/layout for both device artifacts;
- model/provider capability manifest, weight generation, placement generation/fingerprint, and ring capacities;
- RTX 5090 and B70 identity, PCI addresses/root path, NUMA placement, negotiated link under load, ReBAR/IOMMU/ACS state;
- clocks, power settings/measurements, temperatures, throttling, and soak duration;
- prompt/input set, request order, seed/teacher-forced tokens, context lengths, output count, and route-trace identity;
- upstream scheduler settings, concurrency, batching class, exact `M`, and prefill/decode mix;
- CUDA/B70 expert ownership, resident packed bytes, and all runtime/safety reservations;
- ring slot count, queue depth, completion policy, and fixed token/route capacities; and
- correctness-artifact identity.

A material manifest difference makes results non-comparable unless the difference is the explicit independent variable.

## Historical and reference evidence

The proven Colibri GS64 native path has demonstrated persistent compact B70 ownership, FP16 activation staging, canonical selected IDs/weights, ESIMD expert execution, one routing-weighted hidden-size partial, asynchronous issue/take, exact failed-route recovery, CPU-reference numerical agreement, end-to-end CUDA+B70 generation with zero normal-path CPU expert fallback, and controlled hybrid throughput close to its corresponding all-CUDA Colibri expert configuration.

Historical observations include:

- roughly 56–100 µs one-token B70 issue/take per active MoE layer;
- an illustrative 56.8 µs × 40 layers = 2.27 ms/token serialized contribution; and
- roughly 3.46 ms inter-token latency for a recorded fast all-5090 vLLM workload.

These values establish neither expected hybrid-vLLM latency nor an acceptance target. Batching, process-ring overhead, llm-scaler dispatch, local/remote overlap, piecewise graph boundaries, and synchronization determine production performance.

The B70 608 GB/s figure is an upstream claim. An ideal 18 MiB W4 read gives a derived lower bound near 31 µs, excluding scales, dequantization, intermediates, elementwise work, launch, imbalance, transport, and join. Do not report it as measured kernel or layer time.

A repository sysfs report of `2.5 GT/s ×1` current and maximum is historical reference evidence requiring reproduction under load. If the production B70 remains Gen1 ×1, the topology gate in [`hardware.md`](hardware.md) blocks cross-device production benchmarking.

## Controlled production configurations

Interleave at least these configurations using the frozen manifest and workload:

| Configuration | Purpose |
|---|---|
| Stock all-CUDA upstream vLLM | Throughput, TTFT, ITL, prefill, correctness, and latency ceiling |
| CUDA hot experts + isolated B70 llm-scaler provider | Shooting Brake production result |
| CUDA hot experts + CPU cold experts | Explicit offload baseline only; not the production normal path |
| Reduced CUDA budget without B70 | Capacity/control baseline |
| Native Colibri B70 worker, where model shape and artifact are compatible | Provider/transport overhead comparator, clearly labeled reference evidence |

The all-CUDA configuration must first fit the comparison model. If the capacity target itself does not fit entirely on the 5090, use a smaller shape-compatible model for the direct performance ceiling and separately report the larger model's measured capacity envelope. Never imply an impossible all-CUDA run occurred.

All-CUDA mode through the out-of-tree adapter must match stock vLLM output and performance within measurement noise before B70 routes are enabled. This isolates adapter overhead from provider benefit.

## Qualification sequence and matrices

### 1. Provider capability and direct XPU matrix

Exercise the isolated persistent provider with preselected IDs/weights and the exact qualified model shapes:

| Workload | Required cases | Expected family |
|---|---|---|
| Decode | `M=1` | tiny INT4 preselected-route |
| Continuous decode | every representative `M=2..32` | tiny or small-batch INT4, selected by measured threshold |
| Larger decode | representative negotiated `M>32` | grouped-route path |
| Prefill | representative short/medium/large negotiated `M` | gather-v2, grouped up/down, activation, weighted accumulation |

For every direct-provider class cover all staged routes remote, mixed local/remote semantic subsets, duplicate/non-sorted IDs, multiple tokens selecting one expert, unequal and near-zero weights, boundary IDs, and compact-slot remapping. Measure provider dispatch, H2D/D2H, remap, up, activation, down, weighted accumulation, complete device-local partial, first-call compilation, allocation count, memory growth, power, clocks, and p50/p95/p99. Exercise zero-remote layers only at the adapter/ring/end-to-end gate, where they must produce no provider dispatch and a zero CUDA remote lane.

The direct provider gate requires correct compact `[M_remote, hidden]` weighted wire partials plus exact row maps, explicit rejection of unsupported shapes, stable multiple-shape reuse, and no steady-state weight upload or tensor allocation. The CUDA-side gate separately verifies deterministic scatter into a zero-initialized `[M, hidden]` batch buffer. The provider must never run router/top-k or shared-expert work.

### 2. Fixed pinned-ring transport matrix

Measure the production multi-slot process ring, not an ad hoc IPC path:

```text
CUDA D2H -> pinned publication -> provider H2D
-> no-op/tiny kernel -> provider D2H -> pinned publication
-> CUDA H2D -> completion
```

Exercise:

- `M=1`, representative `M=2..32`, larger decode, and prefill activation/result sizes;
- only remote-bearing rows, including zero-row/no-submission behavior;
- queue depths and ring slots through wraparound and bounded backpressure;
- idle and concurrent RTX 5090 local-MoE/shared/attention-like work;
- provider restart, stale sequence/generation, timeout, cancellation, simultaneous completion, buffer reuse, and shutdown with work in flight; and
- startup/recovery NVMe activity separately from the warmed path.

Report each copy direction, publication/queue time, completion time, total round trip, pinned NUMA placement, CPU signaling/polling cost, allocations, stale/discarded completions, and exposed CUDA wait. The gate requires bounded p99, no early reuse, no global synchronization, no hot-path allocation, correct release/acquire ordering, and no stale output acceptance.

### 3. Batched provider mathematics

Compare independently:

```text
Y_reference = weighted sum of exactly the B70-owned routes
Y_provider_wire  = returned weighted [M_remote, hidden] partial before CUDA scatter
```

Validate CUDA and B70 artifacts independently against the same higher-precision source checkpoint before validating their joined output. Cover `M=1`, `M=2..32`, representative prefill `M`, all/mixed/no remote routes, repeated experts, unequal weights, invalid placement/weight generation, and injected provider error. Unsupported shape/layout/group-size combinations fail explicitly.

### 4. Hybrid layer matrix

For each batch class, decompose:

$$
T_{\mathrm{MoE}} \approx
\max\left(
T_{\mathrm{CUDA\ local}},
T_{\mathrm{CUDA\to\ host\to\ B70}} + T_{\mathrm{B70}} + T_{\mathrm{B70\to\ host\to\ CUDA}}
\right) + T_{\mathrm{join}}.
$$

For hidden size 2048 and FP16, one transported activation row or result row is 4096 bytes; report actual bytes and dtype rather than assuming this shape for every model. Record CUDA router/partition, D2H, ring queue, provider H2D, remap/kernel/accumulation, provider D2H, CUDA H2D, local CUDA routed/shared work, overlap, join/add, and exposed wait.

Acceptance requires one or zero B70 operations per active layer and scheduler step, no per-request submissions, correct zero-remote handling, exact output agreement within the qualified artifact tolerance, and no device-wide synchronization or per-forward allocation.

### 5. Scheduler and end-to-end matrix

Exercise:

| Scenario | Required evidence |
|---|---|
| `M=1` decode | request/output throughput, TTFT, ITL p50/p95/p99, per-layer branch/remote/join timing |
| Changing continuous batch `M=2..32` | aggregate and per-request throughput/ITL, one B70 submission per layer/step, queue/backpressure, cancellation |
| Larger decode batch | grouped-kernel threshold, upstream admission/chunking at capacity boundaries with no intra-layer-step split or reassembly, per-request fairness and tails |
| Short and long prefill | prompt throughput, TTFT, gather/up/down/accumulate timing, transfers, local/remote overlap |
| Mixed prefill/decode | upstream scheduler behavior, decode tails, row mapping, provider kernel-family partitions |
| Zero-remote layers | no ring slot, request, copy, provider work, or join wait |
| Long context/KV pressure | 5090 KV bytes, admitted concurrency/context, weight capacity gained, TTFT/ITL |
| B70 timeout/loss/restart | exact failed routes, recovery/failure latency, stale-result rejection, generation bump |
| Cancellation and ring pressure | bounded queue, slot lifecycle, no use-after-cancel or early reuse |
| Long soak | memory stability, clocks, power, thermals, errors, stale completions, output checks |

Begin hybrid qualification in eager mode. Measure PIECEWISE CUDA graphs only after eager correctness, with the external provider operation as a graph break. Full CUDA graph results are not comparable unless the external dependency is represented safely and explicitly.

### 6. Capacity measurement

For each controlled configuration, record actual allocated and reserved bytes:

- RTX 5090 state-owner/dense/shared weights, routed experts, KV/recurrent state, scratch/join, graph/runtime, and headroom;
- B70 cold/overflow expert weights, stable activation/route/output/scratch tensors, provider/runtime, and headroom;
- pinned ring bytes and slot/token/route capacities;
- other host memory for control, telemetry, and exact emergency recovery; and
- NVMe bytes/read reason.

Report capacity gain as the additional qualified model-weight residency and/or 5090 KV/context/concurrency budget enabled by B70 after all reservations. Pair it with throughput, TTFT, ITL, prefill, power, and failure behavior relative to stock all-CUDA vLLM. Nominal combined VRAM is not a measured gain.

## Required production metrics

Across at least three interleaved runs, report median and range plus appropriate percentiles for:

- request throughput and output-token throughput;
- per-request TTFT and ITL;
- prefill throughput;
- CUDA/B70 route shares and remote-bearing token rows;
- B70 submissions per active layer/step, required to be zero or one;
- provider queue, copies, kernel family/time, and exposed wait;
- CUDA local routed and shared-expert time, overlap, join/add time, and graph mode;
- CPU recovery count/time and cause;
- cancellation, timeout, restart, backpressure, and stale/discarded completions;
- RTX 5090, B70, pinned-host, other host, and NVMe memory/activity;
- capacity/context/concurrency gained;
- power and thermal behavior; and
- per-layer routed output, final-logit, and generated-token agreement.

## Structured record contract

Each request/layer/step record includes at least:

```text
run and correctness-artifact IDs
request/token/layer and prefill/decode class
scheduled M, remote rows, and remote routes
canonical selected IDs/weights and CUDA/B70 owners
placement, weight/provider, request sequence, and ring slot
provider kernel family and capability version
CUDA partition/D2H, ring queue, B70 H2D/kernel/D2H, CUDA H2D/join
CUDA local routed/shared time, branch overlap, and exposed wait
B70 submission count for the layer/step
TTFT, ITL, request/output throughput association
bytes and reservations by memory domain
cancellation/failure/recovery/stale-discard reason
CPU recovery and NVMe foreground activity
output token/logit checksum
```

Timing units and clock sources are declared. Non-applicable fields remain explicit. Raw records remain machine-readable and sufficient to recompute aggregates.

## Correctness and failure evidence

Performance is accepted only when the linked correctness artifact demonstrates canonical router-ID/weight agreement, exact local/remote partitioning, one weighted B70 partial per token, exactly-once join, bounded numerical drift for the qualified CUDA/B70 artifacts, coherent final logits/tokens, and exact failure behavior.

Failed, recovered, timed-out, canceled, stale, or divergent samples remain visible. Injected B70 loss must either recompute the exact failed token-route entries on an admitted CUDA/CPU path or fail explicitly. Healthy normal-path runs have zero CPU expert matrix work and zero foreground NVMe expert reads.

## Publication gates

A production claim is publishable only when:

1. the manifest, raw records, and linked correctness evidence are available;
2. topology under load, versions, clocks, power, thermals, placement, and actual memory reservations are explicit;
3. cold/warm, prefill/decode, and all batching classes are separate;
4. p50/p95/p99, sample counts, median/range across interleaved runs, exclusions, and failures are countable;
5. provider direct, fixed-ring, mathematical, layer, scheduler, failure, and capacity gates for the claimed scope pass;
6. all-CUDA adapter mode matches stock upstream vLLM before hybrid results are compared;
7. the hybrid result is compared with the identical stock all-CUDA upstream-vLLM workload;
8. capacity gained is paired with throughput/latency/power cost;
9. normal-path CPU expert matrix work and foreground NVMe expert reads are zero; and
10. every Colibri value is labeled reference/baseline evidence rather than production behavior.
