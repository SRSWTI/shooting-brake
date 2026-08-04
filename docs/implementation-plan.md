# Shooting Brake Implementation Plan Companion

## Authority and purpose

[`../plan.md`](../plan.md) is the sole authoritative active plan. This document is a subordinate execution and evidence checklist for the same **Phase 0 through Phase 10** sequence. It does not authorize a second sequence, rename phases, add prerequisites, or override a gate in `plan.md`. If wording differs, `plan.md` controls.

**Status:** Phase 0, Phase 1, and Phase 2 are complete against the pinned production configuration; Phase 3 is next. Existing Colibri CUDA+B70 work remains reference evidence only. Completion evidence for the production-oriented QuixiCore-XPU provider and process ring is recorded in [`progress.md`](progress.md#phase-1-persistent-provider-evidence) and [`progress.md`](progress.md#phase-2-process-ring-evidence); later phases remain unpassed until their own evidence is recorded.

The active production direction is:

```text
upstream vLLM 0.26+ on one RTX 5090
    -> CUDA scheduler, state, router, canonical top-k, local/shared experts
    -> Qwen-scoped out-of-tree HybridMoERunner / HybridRoutedExperts
    -> versioned pinned-memory request ring
    -> isolated persistent QuixiCore-XPU provider on one B70
    -> qualified QuixiCore-XPU tiny / batched / prefill NVFP4 MoE kernels
    -> weighted [M_remote, hidden] wire partial plus token_row_map
    -> asynchronous CUDA copy and addition
```

The CPU performs orchestration and exact emergency recovery only. Normal-path CPU matrix compute is prohibited.

## Execution rules

1. Work advances only through Phase 0–10 in [`../plan.md`](../plan.md).
2. A later phase may prepare fixtures, but its results cannot waive an earlier phase gate.
3. A gate passes only with reproducible evidence from the exact pinned model, provider, protocol, hardware, and workload scope.
4. The native Colibri GS64 path is a comparator and correctness oracle, not proof of production batching, upstream-vLLM integration, or QuixiCore-XPU-provider overhead.
5. Stock upstream vLLM is the CUDA reference. All-CUDA behavior through the adapter must match stock behavior before B70 routes are enabled.
6. vLLM remains the canonical router/top-k authority. The B70 consumes selected IDs and weights and never recomputes routing.
7. Every selected route contributes exactly once, is recovered exactly, or causes explicit failure.
8. The B70 returns one routing-weighted compact `[M_remote, hidden]` wire partial; the CUDA owner scatters it into the full `[M, hidden]` batch and joins it before final tensor/expert-parallel reduction.
9. CPU matrix compute is allowed only for exact emergency recovery, never as an ordinary performance tier.
10. Unsupported protocol versions, shapes, layouts, quantization, top-k, models, provider generations, placement generations, or weight generations fail explicitly.
11. Correctness and performance evidence remain separate. First invocation and warmed steady state, eager and graph mode, prefill and decode, and batch-one and continuous batching must not be conflated.
12. Kernel or transport microbenchmarks do not establish end-to-end benefit.
13. Failures, cancellations, timeouts, restarts, rejected work, recoveries, and stale completions are retained as evidence.

See [`architecture.md`](architecture.md), [`expert-fabric.md`](expert-fabric.md), and [`correctness.md`](correctness.md) for the detailed architecture, protocol, and numerical contracts.

## Evidence record

Use this record for each phase gate. Blank fields mean “not established.”

```text
Phase: <0..10> — <name>
Status: <not-started | in-progress | blocked | passed | failed | reconsideration-required>
Scope: <model, adapter, provider, protocol, shape, placement, workload>
Inputs and provenance: <commits, versions, manifests, artifact hashes>
Environment: <RTX 5090, B70, links, drivers, runtimes, clocks, memory>
Oracle and tolerances: <stock vLLM, source checkpoint, Colibri/reference path>
Cases executed: <including edge, failure, cancellation, restart, and omitted cases>
Observed correctness evidence: <artifact identifiers and results>
Observed performance evidence: <raw records and p50/p95/p99 where applicable>
Known exceptions: <none or explicit unresolved deviations>
Decision: <pass | fail | block | reconsider, applying plan.md's gate>
Reviewed by/date: <identity and timestamp>
```

A record links raw or structured artifacts rather than copying only a favorable aggregate. See [`benchmarking.md`](benchmarking.md) for the measurement contract.

## Phase dependency chain

```text
Phase 0  freeze baselines and compatibility contracts
   -> Phase 1  isolated persistent QuixiCore-XPU B70 provider
   -> Phase 2  batched versioned pinned-memory protocol
   -> Phase 3  independent provider mathematics
   -> Phase 4  Qwen-scoped upstream-vLLM adapter
   -> Phase 5  compact immutable expert ownership
   -> Phase 6  eager hybrid execution
   -> Phase 7  continuous-batch decode and grouped prefill
   -> Phase 8  piecewise CUDA graphs and exposed-wait removal
   -> Phase 9  failure, restart, recovery, and operations
   -> Phase 10 controlled production benchmark
```

## Phase 0 — Freeze baselines and compatibility contracts

**Status: COMPLETE.** Evidence: [`../phase0/SUMMARY.md`](../phase0/SUMMARY.md), [`../phase0/freeze.yaml`](../phase0/freeze.yaml), and [`../phase0/capability_manifest.yaml`](../phase0/capability_manifest.yaml).

### Objective

Make both independently evolving runtimes and every cross-runtime artifact explicit before coupling them.

### Required work

Record and pin:

- exact upstream vLLM 0.26+ commit and CUDA dependencies;
- QuixiCore-XPU provider release and selected NVFP4 MoE operator bundle, plus llm-scaler and `vllm-xpu-kernels` revisions for the secondary INT4 alternative;
- PyTorch-XPU, `vllm-xpu-kernels`, oneAPI, Level Zero, firmware, and driver versions;
- RTX 5090 and B70 identity, topology, memory, and negotiated links;
- source checkpoint, tokenizer, model configuration, and artifact hashes;
- CUDA and B70 weight artifacts, quantization, group size, packing, layout, scale dtype, and conversion generation;
- one fixed correctness prompt/teacher-forced set;
- one fixed performance workload and scheduler configuration;
- stock all-CUDA vLLM eager and supported graph-mode baselines.

Define before runtime integration:

- the versioned pinned-memory provider protocol;
- the provider capability schema;
- the model and weight-artifact schema;
- the placement/ownership schema;
- request, completion, provider, placement, and weight generations;
- maximum negotiated `M`, hidden size, top-k, dtypes, capacities, and kernel families;
- exact unsupported-shape and recovery behavior.

The compatibility table must treat upstream vLLM, the primary QuixiCore-XPU provider, and the secondary llm-scaler INT4 alternative as separate upgrade surfaces. llm-scaler's complete vLLM 0.21 patch remains reference evidence and is not applied to vLLM 0.26+. Any API or binding adaptation remains inside the isolated provider or narrow out-of-tree adapter.

### Gate

Startup validation rejects incompatible protocol versions, models, shapes, dtypes, layouts, group sizes, top-k, weight generations, placement generations, and kernel capabilities with an actionable error. The fixed stock-vLLM correctness and performance baselines are reproducible and all required manifests are fingerprinted.

### Evidence checklist

- immutable revision and environment inventory;
- signed or hashed model/provider/capability manifests;
- successful compatibility negotiation for the supported configuration;
- negative negotiation cases for every declared incompatibility class;
- stock all-CUDA vLLM eager baseline;
- stock all-CUDA supported graph-mode baseline;
- fixed correctness and performance workload identities;
- explicit record that existing Colibri results are reference inputs, not this phase's production pass.

### Decision rule

No provider or adapter coupling begins while any compatibility field is implicit or any baseline/artifact identity is unstable.

## Phase 1 — Build the isolated QuixiCore-XPU B70 provider

**Status: COMPLETE.** The direct native provider-core gate passed on the physical B70; see [`progress.md`](progress.md#phase-1-persistent-provider-evidence).

### Objective

Create one persistent B70 provider, using QuixiCore-XPU's framework-neutral C++ ABI or PyTorch binding, that exposes only the qualified routed-expert operations required by Shooting Brake.

### Required work

The provider must:

- select the B70 explicitly and fail if the admitted device is absent or ambiguous;
- import only the Phase-0-pinned QuixiCore-XPU operator bundle;
- expose capability, load, issue, take, health, generation, and shutdown on the provider core; expose startup load plus capability/health/shutdown on the isolated process control shell; use the completed Phase-2 pinned ring for process-facing activation/route payload transport;
- load each compact B70 expert artifact once; validate exact bank size/header/layout at startup; record and qualify its SHA-256 externally in the frozen manifest; require the completed Phase-2 ring's per-request weight-generation match;
- maintain `(layer, global expert) -> compact slot` mapping;
- preallocate grow-only or fixed XPU activation, route, scratch, and output tensors;
- accept only canonical preselected IDs, routing weights, masks, and row maps;
- choose tiny, small/batched, and grouped/prefill kernels from measured and qualified shape thresholds;
- return one weighted `[M_remote, hidden]` wire partial, `token_row_map`, and per-token/per-route status;
- report unsupported shapes and kernel failures explicitly;
- perform no router, top-k, shared-expert, attention, KV/GDN, residual, LM-head, or sampling work.

Retain `colibri-variants/colibri-qwen36/c/b70_moe_sycl.cpp` as the native GS64 comparator. Integrate the pinned QuixiCore-XPU bundle through its framework-neutral C++ ABI or PyTorch binding; do not fork or extract its kernels into a separate code lineage. Keep llm-scaler through `vllm-xpu-kernels` as a secondary INT4 alternative if NVFP4 quality is insufficient.

### Gate

Direct provider cases pass for `M=1`, decode-relevant `M=2..32`, and representative prefill `M`. Steady-state dispatch performs no weight upload, tensor allocation, or artifact conversion. Repeated mixed shapes do not reuse stale buffers or grow memory.

Recorded evidence: the exact 14,495,580,220-byte full bank loaded after startup size/header/layout validation, and its externally computed SHA-256 is frozen in the manifest; `M=1`, `M=2..32`, duplicate valid top-8, and fused `M=128` passed against the saved reference; invalid layer/IDs and stale generation/sequence failed closed; allocations remained at the post-load baseline across nine successful dispatches; shutdown was idempotent; and the official compressed-tensors representation oracle passed. The cold first dispatch and warmed `M=1` dispatch are reported separately in [`progress.md`](progress.md#phase-1-persistent-provider-evidence).

### Evidence checklist

- provider capability response tied to pinned revisions;
- compact expert load and fingerprint verification;
- first-load versus warmed dispatch separation;
- supported and unsupported shape behavior;
- zero contribution for an already-staged row whose valid remote subset is empty;
- sequential mixed-shape and repeated-dispatch stability;
- allocation and memory-growth observations;
- kernel-selection trace for tiny, batched, and prefill cases;
- provider failure and shutdown behavior.

### Decision rule

A provider that silently falls back, reallocates in steady state, uploads weights per request, or performs model-owner work does not pass.

## Phase 2 — Implement the batched versioned pinned-memory protocol

### Objective

Extend the proven Colibri issue/take lifecycle into a runtime-neutral multi-slot protocol suitable for vLLM scheduler-step batches.

### Required work

Implement a fixed-capacity pinned-memory ring with:

- multiple reusable slots;
- explicit slot ownership and state transitions;
- protocol version and negotiated maximum capacities;
- request and completion sequence numbers;
- provider, placement, and weight generations;
- layer identifier;
- batched activation rows, route IDs, routing weights, masks, and original token-row map;
- per-token/per-route status and failed-route bits;
- explicit acquire/release publication ordering;
- timeout, cancellation, backpressure, shutdown, and provider-restart handling.

For `[M, hidden]`, stage only token rows with at least one B70-owned route. Preserve the original row mapping so CUDA scatter and reduction remain deterministic. The hot path must use fixed buffers and avoid `.item()`, pageable staging, whole-device synchronization, and unbounded CPU polling.

No model execution is needed for the first ring lifecycle qualification; a deterministic vector/no-op operation may establish the transport state machine before the provider partial is attached.

### Gate

The ring survives wraparound, concurrent slots, full-ring backpressure, stale replies, invalid generations, cancellation, shutdown with work in flight, provider restart, and injected kernel failure without accepting stale output, reusing a live buffer, or losing an expert contribution.

### Evidence checklist

- slot-state traces for normal and failure lifecycles;
- wraparound beyond sequence-space boundaries relevant to the implementation;
- concurrent producer/provider/consumer activity;
- negative version and capacity negotiation;
- stale request/completion and generation rejection;
- provider restart with generation bump;
- cancellation before and after provider issue;
- buffer-reuse and memory-ordering evidence;
- allocation, synchronization, CPU-consumption, and p50/p95/p99 transport records.

### Decision rule

Any ambiguous ownership, premature reuse, stale acceptance, lost completion, unbounded queue, or global device synchronization blocks integration.

**Completion evidence — PASS on 2026-08-04.** `phase2/ring_protocol.hpp` and `phase2/shared_ring.{hpp,cpp}` implement the protocol-v2 fixed-layout eight-slot ABI and lifecycle. `phase2/shared_ring_tests.cpp` passed malformed payload/bounds checks, all-or-nothing failure, cancellation/deadline quarantine, stale completion, delayed writer, full-ring backpressure/reuse, provider death/generation replacement, cross-process generation retirement, and 2,000,000-request wraparound. `phase2/memfd_transport_{host.cu,provider.cpp}` passed independently mapped CUDA/Level-Zero byte transport. `phase2/b70_ring_{provider.cpp,integration_test.cpp}` passed real-B70 numerical execution, all eight queued slots, wrapped reuse, zero-remote no-submission, stale identity rejection, exact dispatch/allocation accounting, clean shutdown, and warmed p50/p95/p99 stage measurement for `M=1/8/32/128`. Full values and limitations are recorded in [`progress.md`](progress.md#phase-2-process-ring-evidence).

## Phase 3 — Validate provider mathematics independently

### Objective

Prove that the provider's returned partial contains exactly the B70-owned routed contributions before combining it with vLLM.

### Required identity

For each token row:

$$
Y_{\text{provider}}[t] \approx
\sum_{e \in R_{\text{B70}}(t)}
 w_{t,e}\,\operatorname{Expert}_e(x_t)
=Y_{\text{reference}}[t].
$$

Validate CUDA and B70 artifacts independently against the same higher-precision source checkpoint before validating their sum. Record the accepted quantization and accumulation tolerance rather than borrowing one from an unrelated model or format.

### Required cases

- `M=1`, every qualified decode range through `M=32`, and representative prefill sizes;
- all staged routes remote and mixed local/remote semantic subsets;
- duplicate and non-sorted selected IDs as allowed by canonical inputs;
- multiple tokens selecting the same expert;
- unequal, zero, and near-zero routing weights;
- boundary global IDs and compact-slot remapping;
- padded/invalid route positions;
- invalid provider, placement, and weight generations;
- unsupported shapes and capacities;
- injected device or kernel failure.

For a full batch with no B70-owned routes, the adapter must issue no provider request and must preserve a zero CUDA remote lane. Within an already-staged row, an empty valid remote subset must produce zero provider contribution. The provider must never include local CUDA contributions.

### Gate

Every supported shape and route case satisfies the recorded numerical tolerance and route identity. Unsupported inputs fail explicitly. Error metadata identifies exactly which routes require recovery without accepting a partial contribution as complete.

### Evidence checklist

- source checkpoint and both artifact fingerprints;
- reference implementation and tolerance rationale;
- per-case maximum/mean error and NaN/Inf checks;
- route/slot mapping and weighted-sum trace;
- duplicate, invalid, boundary, and zero-route behavior;
- failure metadata and exact recovery input reconstruction;
- sequential mixed-shape stability.

### Decision rule

Any unexplained route loss, double count, slot confusion, generation confusion, numerical divergence, or silent generic-kernel fallback blocks the adapter phase.

## Phase 4 — Add the upstream-vLLM out-of-tree adapter

### Objective

Install the narrow post-top-k seam without replacing vLLM's scheduler, model state, router, shared expert, or serving stack.

### Required work

Implement Qwen-scoped:

```text
HybridMoERunner
HybridRoutedExperts
ShootingBrakeExpertProviderClient
```

Inject the implementation explicitly through vLLM's modular MoE factory/pluggable-layer mechanism for only the qualified Qwen architecture and configuration. Preserve stock vLLM behavior for:

- scheduler and continuous batching;
- attention, KV cache, and DeltaNet/GDN state;
- CUDA router and canonical top-k;
- shared expert and residual path;
- LM head, sampling, and serving APIs;
- unsupported models and configurations.

Begin in eager mode. First route every expert locally through the adapter. Do not enable B70 ownership until the all-CUDA adapter path matches stock vLLM.

### Gate

All-CUDA execution through the adapter matches stock upstream vLLM output and performance within the predeclared measurement noise. Router IDs/weights, routed output, shared-expert behavior, final logits, generated tokens, request completion, and cancellation remain equivalent.

### Evidence checklist

- explicit adapter selection and stock-path non-selection cases;
- canonical router/top-k equality;
- per-layer routed-output agreement;
- final-logit and generated-token agreement;
- eager scheduler, completion, and cancellation behavior;
- performance distributions against stock vLLM on the same workload;
- unsupported architecture/configuration staying on stock vLLM.

### Decision rule

Do not attribute an adapter regression to the B70 and do not enable remote routes until the local-only adapter path is equivalent.

## Phase 5 — Load compact immutable expert ownership

### Objective

Load one explicit per-model placement generation with disjoint CUDA and B70 expert ownership.

### Required work

Construct and validate:

```text
(layer, global expert) -> CUDA local slot
(layer, global expert) -> B70 compact slot
```

Load only hot CUDA-owned experts into the CUDA local bank and only cold B70-owned experts into the provider bank. Reuse vLLM `ExpertMapManager` concepts where compatible, but do not represent the B70 as a fake expert-parallel rank.

The ownership manifest records model, weight artifact, compact slot, provider, and placement generations. Placement changes happen outside in-flight request generations. No expert weight moves during normal inference.

### Gate

Every routed expert in the qualified placement has exactly one normal-path owner. CUDA and B70 ownership are disjoint and cover the admitted model placement. Every loaded slot fingerprint matches the manifest, and steady-state inference performs no expert upload, conversion, promotion, or migration.

### Evidence checklist

- full ownership-map validation;
- duplicate, missing, out-of-range, and wrong-generation rejection;
- CUDA and B70 artifact fingerprints by slot;
- memory admission and reserved-capacity record;
- no foreground weight-transfer observation;
- placement rollback before service if provider load fails.

### Decision rule

Ambiguous ownership, incomplete coverage, artifact mismatch, or foreground weight movement blocks hybrid execution.

## Phase 6 — Integrate eager hybrid execution

### Objective

Execute the first correct end-to-end upstream-vLLM hybrid path in eager mode.

### Required per-layer flow

```text
CUDA router and canonical top-k
 -> partition selected routes by placement generation
 -> publish B70 request
 -> execute local CUDA routed experts and shared expert
 -> validate and collect B70 weighted partial
 -> asynchronous host-to-CUDA copy
 -> CUDA addition before final reduction
 -> residual continuation
```

Start with tensor parallelism/expert parallelism equal to one. Qualify other parallel modes separately after the single-rank boundary is correct. If the selected CUDA backend cannot safely skip remote route IDs, compact local routes and unpermute/reduce them rather than passing unsupported sentinels.

Implement an exact failure path: recover precisely the failed B70 route bits on CUDA or CPU when viable, or fail the request explicitly. CPU recovery is exceptional and visible, not a normal tier.

### Gate

For the fixed correctness set, deterministic generation, final logits, per-layer routed output, selected routes, weights, and ownership agree with the all-CUDA reference within the selected artifact tolerance. Every selected route contributes exactly once. No normal successful request performs CPU matrix compute.

### Evidence checklist

- no-remote, all-remote, and mixed local/remote layers;
- per-layer local, remote, and summed output comparison;
- final logits and generated-token comparison;
- zero-route and boundary-route behavior;
- issue-before-local-work overlap trace;
- provider timeout/failure with exact failed-route recovery;
- stale completion rejection;
- CPU recovery counts separated from orchestration;
- no steady-state allocation or weight transfer, plus a trace of synchronization and exposed-wait behavior for the eager correctness baseline.

### Decision rule

A lost/doubled route, accepted stale partial, unexplained output drift, silent fallback, or normal-path CPU expert computation fails the phase.

## Phase 7 — Add continuous-batch decode and grouped prefill

### Objective

Move from correct eager single-step integration to production scheduler-step aggregation.

### Required work

- constrain upstream scheduler-step admission to the provider's negotiated token/route capacity, then aggregate every admitted step's remote token rows into exactly one B70 layer request;
- preserve token/route maps across mixed request lifecycles;
- select tiny versus batched decode kernels from qualified thresholds;
- add QuixiCore-XPU's grouped prefill NVFP4 MoE path;
- retain router, top-k, shared expert, attention, and state work on CUDA;
- skip provider submission for zero-remote-route layers;
- support changing batch sizes, mixed prefill/decode, completion, cancellation, and backpressure;
- handle oversized prefill by upstream scheduler chunking before layer execution; never split one layer step across multiple ring transactions.

### Gate

Correctness holds across changing batch sizes, mixed prefill/decode scheduling, concurrent requests, request completion, cancellation, zero-remote-route layers, capacity boundaries, and provider backpressure. Every admitted scheduler step fits one negotiated ring slot; the provider receives its batched canonical remote routes and returns one correct `[M_remote, hidden]` wire partial, which CUDA scatters into one `[M, hidden]` batch buffer.

### Evidence checklist

- decode `M=1` and changing `M=2..32`;
- representative grouped prefill sizes;
- mixed prefill/decode scheduler steps;
- concurrent request row-map isolation;
- completion and cancellation at each ring lifecycle boundary;
- zero-remote fast path;
- upstream admission/chunking behavior at capacity boundaries, with every admitted layer step fitting one ring slot and no intra-layer-step split or reassembly;
- kernel-selection thresholds and queue distributions;
- per-token routed-output, logits, and generated-token agreement.

### Decision rule

Batching cannot weaken route identity, request isolation, cancellation, or completion generations. Any cross-request row confusion or partial acceptance fails the phase.

## Phase 8 — Restore piecewise CUDA graphs and remove exposed waits

### Objective

Recover vLLM CUDA efficiency around the unavoidable external-provider boundary without weakening the process-ring contract.

### Required work

- keep the graph break localized around the external provider operation;
- use dedicated CUDA copy streams, events, preallocated staging buffers, and nonblocking completion;
- remove `.item()`, device-wide synchronizations, exposed Python polling, and avoidable serialization from the steady-state path;
- preserve zero-remote graph paths;
- measure overlap among remote transfer/provider execution, local routed experts, and shared-expert work;
- consider ExLlama-style stream-memory operations only after the correct process-ring implementation is the measured baseline.

### Gate

Supported graph mode preserves output agreement and improves or maintains eager throughput on the same workload. No device-wide synchronization or hot-path allocation appears in steady state, and optimization does not bypass protocol versioning, ownership, or recovery metadata.

### Evidence checklist

- eager versus graph correctness and performance;
- graph captures and breaks localized to the intended seam;
- CUDA stream/event timeline;
- local/remote/shared overlap and exposed-wait measurements;
- zero-remote and remote-active steps;
- allocation and synchronization traces;
- cancellation, timeout, and recovery under graph mode.

### Decision rule

Retain eager mode for an unsupported or regressing configuration. Graph enablement never waives correctness or lifecycle gates.

## Phase 9 — Failure, restart, and operational qualification

### Objective

Make B70 loss and provider lifecycle events explicit, bounded, and semantically exact.

### Required work

Implement and qualify:

- heartbeat and bounded request timeout;
- provider process restart with generation bump;
- rejection of completions from old provider generations;
- exact failed-route CUDA or CPU recovery when viable;
- explicit request failure when exact recovery is not viable;
- placement/load rollback and provider quarantine;
- bounded queues and backpressure;
- cancellation and shutdown with work in flight;
- structured telemetry without activation or output payload logging;
- startup health checks and actionable admission errors.

### Gate

Injected B70 loss, provider crash, kernel failure, malformed completion, timeout, cancellation, and restart never produce silently incomplete output. Each affected request either recomputes exactly or fails explicitly. No stale pre-restart result is accepted.

### Evidence checklist

- failure injection at every slot lifecycle state;
- provider kill/restart and generation trace;
- in-flight request resolution during restart;
- exact failed-route masks and recovery output agreement;
- unrecoverable request error behavior;
- queue bounds and backpressure under sustained load;
- quarantine and placement rollback;
- recovery frequency, latency, and CPU matrix counts;
- telemetry redaction and lifecycle completeness.

### Decision rule

Any silent route loss, ambiguous recovery, stale acceptance, unbounded queue, or hidden retry loop fails operational qualification.

## Phase 10 — Controlled production benchmark

### Objective

Measure capacity, correctness, latency, and throughput against the same upstream-vLLM workload after Phases 0–9 pass.

### Required configurations

| Configuration | Purpose |
|---|---|
| Stock all-CUDA upstream vLLM | Correctness, throughput, and latency ceiling |
| CUDA hot experts + isolated B70 QuixiCore-XPU NVFP4 provider | Shooting Brake production result |
| CUDA hot experts + exact CPU expert execution | Benchmark-only offload/recovery baseline; never the production normal path |
| Reduced CUDA expert budget without B70 | Capacity/control baseline |
| Native Colibri B70 worker where shape-compatible | Provider-overhead reference only |

Use identical checkpoint source, prompt matrix, scheduler settings, context lengths, output count, placement profile, and request-order rotations. Interleave at least three runs per accepted comparison and record median plus range. Colibri timing is never substituted for the stock-vLLM baseline.

### Required measurements

- request and output throughput;
- TTFT and inter-token latency percentiles;
- grouped prefill throughput;
- batch-size and concurrency distributions;
- CUDA, B70, and recovery route shares;
- remote rows and provider submissions per layer/step;
- queue, D2H/H2D copy, provider, kernel, and exposed-wait time;
- CUDA local routed and shared-expert time;
- recovery count and reason;
- RTX 5090, B70, pinned-host, and total host memory;
- provider cancellation, timeout, restart, and backpressure behavior;
- per-layer routed-output, final-logit, and generated-token agreement;
- capacity gained alongside throughput and latency cost.

### Gate

The result demonstrates production-grade continuous-batch decode and grouped prefill with exact failure semantics, no normal-path CPU matrix compute, measurable capacity gain, and honestly reported throughput/latency against the same stock all-CUDA upstream-vLLM workload.

The gate does not require hybrid throughput to exceed an all-CUDA model that already fits entirely on the RTX 5090. It requires the claimed capacity/service benefit to be measured together with its performance cost and correctness evidence.

### Evidence checklist

- all prerequisite phase records linked;
- identical workload and artifact identities across configurations;
- first-run and warmed results separated;
- eager and supported graph results separated;
- prefill, decode, batch-one, and continuous batching separated;
- p50/p95/p99 and raw distributions;
- output agreement and all failure/recovery events;
- memory and capacity accounting;
- Colibri measurements labeled only as comparator/reference evidence;
- explicit limitations and unsupported configurations.

### Decision rule

Accept only claims directly supported by the controlled evidence. A microbenchmark, aggregate device-memory total, nominal bandwidth, isolated kernel speed, or older Colibri throughput number is not a production result.

## Measurement contract for all phases

Every applicable structured record identifies:

```text
model and checkpoint fingerprint
upstream vLLM commit and adapter version
provider, kernel, PyTorch-XPU, oneAPI, Level Zero, and driver versions
protocol, provider, placement, and weight generations
quantization, group size, layout, and artifact fingerprint
prompt/input and scheduler-step identity
token rows, layer, selected expert IDs, and canonical routing weights
expert owners and compact slots
bytes transferred and pinned slot
queue, copy, provider, kernel, CUDA-local, join, and exposed-wait times
recovery routes, CPU matrix time, failure/cancellation reason
TTFT, inter-token latency, and throughput
output token/logit or per-layer checksum
RTX 5090, B70, pinned-host, and total host memory
```

Aggregate p50/p95/p99 rather than only averages. Report correctness separately from performance and identify omitted cases explicitly.

## Architecture reconsideration conditions

Stop advancement and reconsider the relevant design if evidence shows:

1. the pinned-memory remote p99 cannot provide a useful capacity/service advantage for any admitted decode or prefill class;
2. provider mathematics cannot reproduce the qualified source artifacts within the declared tolerance;
3. cross-vendor drift changes selected output behavior materially;
4. provider queueing serializes the CUDA critical path without useful capacity benefit;
5. steady-state execution requires expert weight movement;
6. normal successful requests require CPU matrix compute;
7. failure/restart handling cannot guarantee exactly-once contribution or explicit failure;
8. the isolated PyTorch-XPU wrapper dominates useful provider work.

Condition 8 permits profiling a native Torch-free provider around the already-qualified kernels. It does not permit changing the protocol or semantics without repeating their applicable gates.

## Immediate next step

Execute Phase 3. Compare the process-ring weighted partial with an independent oracle across mixed local/remote subsets, duplicate/non-sorted and boundary expert IDs, repeated experts across tokens, unequal and near-zero weights, explicit token-row remapping, already-staged rows whose valid remote subset becomes empty, invalid generations, and provider failures. Verify deterministic CUDA scatter into the full `[M, hidden]` buffer before beginning Phase 4 all-CUDA upstream-vLLM adapter parity.

## Superseded sequencing

Any prior Stage 0–11, five-GPU, DDR5/NVMe-tier, GLM-first, multi-B70, or Colibri-as-production-host sequence is superseded by [`../plan.md`](../plan.md). Useful measurements from that work remain historical reference evidence in [`progress.md`](progress.md) and the relevant contract documents, but they do not define the active order or production topology.
