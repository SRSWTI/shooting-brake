# Shooting Brake Benchmarking Contract

## Purpose and status

This document defines the benchmark and measurement contract for Shooting Brake. It elaborates the system invariants and profiling boundaries in [`architecture.md`](architecture.md); it does not replace that overview.

This is a **design and acceptance contract**, not a claim that the B70 transport, kernels, model integration, or performance targets have been validated. No numerical result is established here unless it is explicitly identified as a reported observation. Correctness qualification is governed separately by [`correctness.md`](correctness.md); scheduling decisions that consume tail-latency measurements are defined in [`scheduling.md`](scheduling.md).

The evidence categories used here are:

- **Reported observation:** a value recorded by the source design that still requires reproduction on the benchmark host.
- **Upstream claim:** a vendor or upstream-project property, not a local measurement.
- **Derived bound:** arithmetic based on an upstream claim; it is not achievable-performance evidence.
- **Design target:** a required experiment, metric, or gate.
- **Unverified assumption:** a premise that must not be used as a result.

## Normative measurement rules

1. Report the hardware topology, negotiated links under load, software revisions, kernel/provider revisions, quantization format, packing, clocks, power state, and workload identity with every result set.
2. Preserve raw per-request or per-token records. Aggregates alone are insufficient.
3. Report p50, p95, and p99. A mean or throughput number must not replace tail latency.
4. Separate first invocation, compilation/prepack, and other cold costs from warmed steady state. The first call must be measured and reported; it may be excluded only from the explicitly labeled steady-state aggregate.
5. Separate prefill from decode, batch one from continuous batching, and device-local work from transport and end-to-end generation.
6. Exercise idle and contended conditions explicitly. A result from an idle machine must not stand in for NVMe, CPU-expert, RTX 5090-compute, or concurrent-copy contention.
7. Exercise every relevant PCIe slot and root-port path. Results from one favorable topology must not be generalized to another.
8. Use fixed, identified inputs and route distributions. Synthetic distributions and historical GLM route counts must be labeled and reported separately.
9. Keep exact correctness evidence separate from performance evidence. An output token or logit checksum in a timing record correlates the two artifacts; it does not by itself prove correctness.
10. Preserve failures, fallbacks, timeouts, stale completions, discarded completions, and output differences in the records. Failed or divergent samples must not be silently removed from latency statistics.
11. Report the sample count, warm-up policy, repetition policy, aggregation method, and any exclusions. Each exclusion requires a reason and must remain countable from the raw records.
12. Do not infer end-to-end benefit from isolated TFLOPS, GEMM latency, or device bandwidth.

## Provenance and performance floor

The B70's official 608 GB/s bandwidth is an **upstream claim**. For an ideal approximately 18 MiB W4 expert, it gives the following **derived weight-read lower bounds**:

```text
18 MiB / 608 GB/s ≈ 31 µs
36 MiB / 608 GB/s ≈ 62 µs  # two selected experts
```

These figures omit group scales, dequantization, XMX tile utilization losses, intermediate traffic, elementwise work, command launch, and route imbalance. The source design considers a roughly 100–160 µs sparse-layer budget plausible only with balanced work, persistent or immediate command submission, overlapped transport, and no per-layer host synchronization. That is an **unverified design target**, not benchmark evidence.

A source sysfs report showed `2.5 GT/s ×1` as both current and maximum for a B70. This is a **reported observation requiring confirmation under load**. Device-local kernel measurements cannot be used to bypass the hardware-link gate below.

## Reproducible run manifest

Each result set must have an immutable run manifest. At minimum it records:

- run ID and timestamp;
- model and exact model artifact/manifest identity;
- runtime commit, kernel/provider version, driver versions, and operating-system/kernel version;
- quantization format, group/scale convention, packing, and return mode;
- prompt/input identity, random seed where applicable, and teacher-forced token identity where applicable;
- route source and distribution: synthetic pattern, fixed trace identity, or historical GLM trace identity;
- device identities, PCIe slots, root ports, NUMA placement, negotiated link width and speed under load;
- active B70 count, queue depth, synchronization mode, pinned/pageable status, ring-slot configuration, and runtime/provider;
- clocks, power settings, measured power, and thermal state;
- contention class and the exact concurrent workload;
- cold/warm phase, warm-up count, measured sample count, repetition count, and exclusion policy;
- batching class and scheduler configuration, including batching-window/deadline settings;
- resident placement and placement epoch;
- output/correctness artifact identity.

A result is not comparable when a material manifest field differs unless the comparison calls out that difference.

## Unified benchmark matrix

The four matrices below are cumulative. Device-local measurements qualify the operator, transport measurements qualify the path and lifecycle, transport-plus-compute measures the actual remote critical path, and end-to-end measurements decide whether the system improves serving behavior.

### 1. Device-local B70 matrix

Use the exact GLM expert shapes:

```text
gate/up: 6144 -> 4096
SwiGLU:  4096 -> 2048
down:    2048 -> 6144
```

The semantic full partial is remap, GEMM1, activation, GEMM2, and weighted gather. It returns one combined weighted partial per original token for the B70-owned route subset.

| Variable | Required values |
|---|---|
| Token rows | `1, 2, 4, 8, 16, 32, 64`; retain `128` as the extended standalone-kernel row from the earlier matrix |
| Active-expert distribution | one active expert; two evenly active experts; one hot plus several one-row experts; many zero-row experts; one expert receiving all rows; historical GLM route counts when available |
| Selected experts per card | `1, 2, 4, 8` |
| Quantization | W4 baseline; W3 only as an explicitly labeled experiment |
| Return mode | per-expert and locally reduced |
| Runtime/provider | SYCL, Triton XPU, and direct Level Zero where applicable; report unsupported combinations rather than omitting them |
| Packing | model-native and prepacked XMX |
| Process sequence | one fixed shape and multiple shapes in one process |
| Run phase | first call/compilation, prepack, and warmed steady state separately |

Measure separately:

- remap;
- GEMM1;
- activation;
- GEMM2;
- gather;
- complete device-local expert partial;
- allocation and synchronization overhead;
- effective weight bandwidth;
- XMX utilization;
- prepack cost;
- command-launch cost;
- power and clocks;
- p50/p95/p99 for each timed component and the complete partial.

The multiple-shape, single-process sequence is mandatory because it detects stale persistent-buffer reuse previously found in an upstream INT4 path.

Correctness qualification for this matrix is a separate artifact. It covers exact output shape, reference tolerance, stale-buffer reuse, NaN/Inf, zero-row experts, scale overread, and the defined behavior for empty, invalid, duplicate, masked, padded, and zero-weight route subsets. Performance samples must carry the corresponding output checksum and qualification-artifact identity.

### 2. Transport matrix

#### Hardware-link truth

Measure pinned host-to-B70 and B70-to-pinned-host independently at `12, 24, 48, 96 KiB`, including negotiated link speed and width under load. Report p50/p95/p99 under:

- idle conditions;
- concurrent RTX 5090 copy activity;
- concurrent NVMe activity.

#### Host-staged round trip

Measure the complete lifecycle:

```text
RTX 5090 D2H
-> pinned host publication/ring
-> B70 H2D
-> vector, tiny, or no-op kernel
-> B70 D2H
-> pinned host publication/ring
-> RTX 5090 H2D
-> completion
```

| Variable | Required values |
|---|---|
| Payload | `12, 24, 48, 96, 192 KiB`; `12/24/48/96 KiB` satisfy the revised hardware-truth set, while `192 KiB` preserves the larger required transport case |
| Active B70 cards | `1, 2, 4` |
| Queue depth | `1, 2, 4, 8` |
| Synchronization | poll, event, immediate-list event |
| Ring state | one slot; multiple slots; backpressure; wraparound |
| Contention | idle; NVMe traffic; CPU expert/cold compute; RTX 5090 compute or attention-like load; concurrent RTX 5090 copy activity |
| Topology | every populated PCIe slot/root-port path, with NUMA placement reported |
| Failure/lifecycle | stale generation; timeout; producer crash; B70 restart; simultaneous completion; buffer reuse; shutdown with work in flight |
| Run phase | first use and warmed steady state separately |

Measure each direction, host publication/queue time, kernel/no-op time, completion time, and complete round trip at p50/p95/p99. Record allocations, CPU polling consumption, stale/discarded completions, and whether a slot was blocked by backpressure.

### 3. Transport-plus-compute matrix

Replace the no-op with the qualified B70 partial and run the device-local token rows and route distributions through the real host-staged path. For every token batch, report the components of:

```text
T_remote =
    T_cuda_d2h
  + T_queue
  + T_b70_h2d
  + T_remap
  + T_gemm1
  + T_activation
  + T_gemm2
  + T_gather
  + T_b70_d2h
  + T_cuda_h2d
  + T_join
```

Compare the p50/p95/p99 of the complete remote path against both:

```text
T_cpu_fallback
T_5090_local
```

Use the same input rows, expert IDs, route weights, quantization, output semantics, cold/warm classification, topology, and contention class for each comparison. Report device-local partial time alongside transport-plus-partial time so transport and queueing cannot be hidden by the kernel result.

The scheduler eligibility result is computed separately for each measured batch class:

```text
B70 eligible(batch class) iff
    T_remote,p99(batch class) < T_cpu_fallback,p99(batch class)
```

A B70 that loses at batch one but wins at batch 4 or greater may be used for batched decode, prefill, concurrent sessions, and background requests in the winning classes. It must not be forced into latency-critical batch-one decode because its device-local GEMM is fast. See [`scheduling.md`](scheduling.md).

### 4. End-to-end matrix

Run end-to-end scenarios only after the relevant correctness, device-local, transport, and transport-plus-compute gates pass.

| Scenario | Required metrics and evidence |
|---|---|
| Batch-one decode | TTFT, tokens/s, inter-token latency, and per-layer p50/p95/p99 |
| Continuous batching / four concurrent chats | aggregate tokens/s, per-request p50/p95/p99, B70 token-batch sizes, queue depth, and deadline behavior |
| Long prefill | prompt tokens/s, B70 balance, transfer sizes, and p50/p95/p99 |
| Multi-turn | prefix/KV hit rate, recomputed tokens, output agreement, and state coherence evidence |
| Real route distributions | experts contacted per token, owners by tier/device, remote calls per layer, and behavior for hot/skewed/zero-row routes |
| Route shift | adaptation time, migration bytes, placement epoch, and output agreement |
| Static placement | hit share per tier, placement memory, CPU fallbacks, total decode rate, p95/p99, and comparisons with all required baselines |
| B70 failure | failure/timeout point, fallback latency, stale/discarded completion count, and separate output-correctness evidence |
| Cold miss | CPU latency, disk-read count, NVMe bytes/read latency, and fallback reason |
| Background load | foreground p50/p95/p99 impact and dispatch/preemption-boundary behavior |
| Long-context KV pressure | memory by tier, placement pressure, TTFT, inter-token latency, and storage-tier use |
| Long soak | memory growth, stale completions, thermal throttling, clocks, power, and output checks |

The semantic integration target and the GLM architecture target are distinct:

- The Qwen 35B development target exercises real routing and multi-token generation against an existing state-owner reference.
- The deterministic Colibri GLM fixture and exact standalone GLM shapes validate the future GLM contract without claiming that the full 372 GB model is memory-resident.

If a full-GLM experiment still uses storage for experts not resident across the RTX 5090, B70, and RAM tiers, report it as such. It must not be labeled a no-NVMe result.

## Structured record contract

Every later benchmark emits one structured record per request or token with at least these source-defined fields:

```text
model
commit/kernel version
quant format
prompt/input identity
token index
layer
selected expert IDs
expert owners
placement epoch
bytes transferred
queue time
CUDA D2H
B70 H2D
remap
GEMM1
activation
GEMM2
gather
B70 D2H
CUDA H2D
join
CPU fallback time
NVMe bytes/read latency
TTFT
inter-token latency
output token/logit checksum
memory by tier
failure/fallback reason
```

The record also carries the run ID needed to join it to the immutable manifest. When a field does not apply, encode an explicit not-applicable value rather than dropping the field. Timing units and clock source must be declared. Records must remain available in a machine-readable form suitable for replay and aggregation.

Telemetry aggregates additionally include:

- route count by tier/device;
- selected experts per device per layer;
- launch and queue delay;
- B70 and RTX 5090 local compute time;
- CPU fallback time and join wait;
- cache/promotion hit rate;
- prediction precision and recall, when prediction is enabled;
- placement migrations;
- late and discarded completions;
- NVMe reads during active decode;
- watts per generated token.

Aggregate every applicable metric as p50/p95/p99, with explicit separate groups for prefill/decode, batch one/continuous batching, first-run/warm steady state, device-local/transport/transport-plus-compute/end-to-end, route distribution, topology, and contention class.

## Correctness evidence versus performance evidence

Correctness is a prerequisite and a parallel artifact, not a latency percentile:

- The correctness artifact identifies the oracle, inputs, route IDs/weights, tolerances or exact checks, output shape, token/logit agreement, failure injection, and pass/fail result.
- The performance artifact contains all samples, including samples associated with a mismatch, timeout, fallback, stale result, or discarded completion.
- Each performance record carries an output checksum and correctness-artifact identity for correlation.
- Performance summaries must not average numerical error into timing, remove divergent outputs, or call a faster result acceptable when its output differs.
- Fixed prompts and teacher-forced tokens require canonical router-ID agreement, exactly-once route accounting, bounded and characterized numerical drift, coherent multi-turn state, CPU fallback on B70 timeout, ignored stale output, and zero NVMe expert reads during warmed generation before their performance result is accepted.

## Gates and acceptance rules

### Hardware topology gate

If the B70 is genuinely operating at PCIe Gen1 ×1 under load, stop cross-device architecture benchmarking and fix topology, firmware, or slot configuration. Device-local speed cannot compensate for the broken link.

### Device-local operator gate

Before integration, the qualified operator must show:

- one and two resident experts beat the current CPU cold path at the relevant measured batch sizes;
- exact output shape and satisfied reference tolerance;
- no stale-buffer reuse, NaN/Inf, memory growth, or scale overread;
- safe zero-row handling;
- bounded p99;
- first-call compilation reported separately and excluded only from the labeled steady-state aggregate.

No bulk W4 conversion proceeds until the one-expert conversion separately passes all-zero, small, saturating-quantized, random, repeated-identical, changing-token-row, changing-expert-ID, and zero-row cases.

### Transport gate

The host-staged ring must have:

- bounded p99;
- no global synchronization;
- no per-request or hot-loop allocation;
- no buffer reuse before both runtimes complete with it;
- no full-core polling unless measurements justify it;
- correct behavior for backpressure, wraparound, stale generation, timeout, crashes/restarts, simultaneous completion, buffer reuse, and shutdown with work in flight.

### B70 batch-class gate

B70 is eligible for a batch class only when its measured complete remote p99 is lower than CPU-fallback p99 for that class. Batch-one eligibility is decided by the batch-one comparison; a win at batch 4 or greater does not imply a batch-one win.

### Static-placement gate

Static B70 placement must beat all three baselines:

1. CPU-only cold execution;
2. frequency-only RTX 5090 placement without B70;
3. random/uniform B70 placement.

If it does not, do not proceed to a more complex placement algorithm.

### Optimization gate

Use trace-grounded recovered time—not isolated TFLOPS—to rank optimization candidates. A faster isolated GEMM is accepted only if:

- the device-local full partial is faster;
- transport-plus-partial is faster;
- end-to-end generation improves;
- output remains correct;
- p99 does not regress;
- memory does not grow;
- the improvement survives real route distributions.

### Result-publication gate

A result may be presented as comparable evidence only when:

- its manifest and raw structured records are available;
- first-call and warmed results are both visible and separately labeled;
- p50/p95/p99 and sample counts are present;
- topology and contention are explicit;
- prefill/decode and batch-one/continuous-batch results are separate;
- failures, fallbacks, exclusions, and output differences are countable;
- exact correctness evidence is linked but reported separately;
- the claimed scope matches the exercised layer: device-local, transport, transport-plus-compute, or end-to-end.
