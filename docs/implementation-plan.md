# Shooting Brake Implementation Plan

## Purpose and status

This document is the active delivery sequence for Shooting Brake. It elaborates the system boundaries and invariants in [`architecture.md`](architecture.md); it does not replace that overview.

**Document status:** approved design plan, not an implementation-status report. Nothing in this document says that a stage, benchmark, hardware configuration, conversion, or integration is complete. Every stage remains unpassed until its gate has a filled evidence record and an explicit decision.

The terms below distinguish the kinds of statements used in this plan:

- **Normative rule / design target:** behavior that the implementation and its evidence must satisfy.
- **Reported observation:** a result recorded by the source plan that still requires reproduction on the target system.
- **Upstream claim:** capability attributed to an imported project; it is not accepted as Shooting Brake evidence until locally verified.
- **Unverified assumption:** a planning premise that must be tested before it can support a gate decision.

In particular, the reported `2.5 GT/s ×1` B70 sysfs value is a **reported observation requiring confirmation**, not established link capability. Likewise, upstream support for INT4 grouped GEMMs, remapping, activation, and weighted gather is an **upstream claim** until the provenance, format, regression, and correctness gates below pass.

## Normative execution rules

1. The only active order is **Stage 0 through Stage 11, in sequence**. A later stage may prepare notes or fixtures, but its implementation and performance claims may not be used to bypass an earlier gate.
2. A stage passes only from observed, reproducible evidence. Repository documentation, an upstream benchmark, matching tensor dimensions, a successful isolated launch, or an average latency is not sufficient by itself.
3. Full-model integration is prohibited until the provenance/kernel choice, hardware link, one-expert format conversion, device-local kernel, weighted-partial semantics, host-ring transport, transport-plus-compute crossover, end-to-end correctness, placement, and optimization gates have passed. Concretely, **Stage 11 may not start before Gates G0–G10 pass**.
4. Colibri remains the semantic oracle. Placement may change performance but must not change selected routes, route weights, expert math, or the deterministic ownership/join contract described in [`architecture.md`](architecture.md).
5. Exact correctness evidence is recorded separately from performance evidence. First-run and warmed steady state, device-local and end-to-end, prefill and decode, and batch-one and continuous-batch results must not be conflated.
6. Kernel tuning follows semantic and end-to-end integration. An isolated GEMM win cannot authorize a kernel change.
7. NVMe is preload/recovery/background storage where possible, not an ordinary foreground tier. If a run needs foreground storage reads, record them; never relabel that run as no-NVMe.
8. Failures, timeouts, stale completions, fallbacks, worker restarts, and rejected configurations are evidence, not results to omit.
9. See [`correctness.md`](correctness.md) for numerical and route invariants and [`risk-register.md`](risk-register.md) for stop conditions and unresolved risks.

## Gate dependency chain

```text
G0 provenance and provider compatibility
 -> G1 physical link and DMA truth
 -> G2 one-expert W4 conversion
 -> G3 exact-shape device-local matrix
 -> G4 weighted B70 partial semantics
 -> G5 versioned host-ring safety
 -> G6 transport-plus-compute eligibility
 -> G7 small-model and GLM-fixture correctness
 -> G8 static placement benefit
 -> G9 optimized placement benefit and stability
 -> G10 end-to-end kernel-tuning benefit
 -> G11 full GLM integration evidence
```

A failure follows the decision rule in the stage that owns the gate. It does not silently convert into a waiver.

## Gate status and evidence record

Use one copy of this compact record for **every** gate. Blank fields mean “not established,” never “passed.” The template intentionally contains no pre-filled completion state.

```text
Gate: G<stage> — <name>
Status: <not-started | in-progress | blocked | passed | failed | reconsideration-required>
Scope: <hardware/model/shape/batch/provider/commit/configuration>
Inputs and provenance: <revisions, patches, manifests, artifact hashes>
Environment: <devices, links, drivers, firmware, clocks/power, memory, topology>
Oracle and tolerances: <reference implementation, exact checks, numerical limits>
Test matrix: <cases executed; omitted cases and reasons>
Observed evidence: <artifact/report identifiers; p50/p95/p99; correctness/failure data>
Known exceptions: <none, or explicit unresolved deviations>
Decision: <pass/fail/block/reconsider, with the stage decision rule applied>
Reviewed by/date: <identity and timestamp>
```

A gate record must link raw or structured artifacts rather than copy only a favorable aggregate. Measurement fields are governed by [`benchmarking.md`](benchmarking.md).

## Stage 0 — Freeze provenance and compatibility

**Ownership:** kernel/provider provenance and format boundary; see [`model-format.md`](model-format.md), [`correctness.md`](correctness.md), and [`risk-register.md`](risk-register.md).

### Inputs

- The five newly incorporated repository checkouts and their exact revisions.
- The `llm-scaler` pinned XPU-kernel revision and its local patch.
- One canonical expert input/output artifact.
- Exact Colibri quantization metadata, including packing and scale conventions.
- Candidate providers: patched `llm-scaler` INT4, standalone vLLM-XPU INT4, Triton MXFP4, and Xe-Fuse BF16.

### Work

1. Record immutable revisions for all five new repositories.
2. Preserve the pinned `llm-scaler` kernel revision and patch as the known comparison baseline.
3. Define and hash one canonical expert input/output artifact.
4. Capture the exact Colibri quantization metadata associated with that artifact.
5. Build a compatibility matrix covering all four candidate providers without treating unlike formats as drop-in equivalents.
6. Reconcile the patched and standalone vLLM-XPU API and activation differences, and identify whether the Xe2 scale-prefetch guard is present.

### Tests and evidence

- Identical-input A/B between patched `llm-scaler` and standalone vLLM-XPU.
- Packed-weight compatibility checks where representations are intended to match.
- Explicit conversion paths where representations differ, particularly Triton MXFP4 and Xe-Fuse BF16.
- Sequential execution of different shapes in one process.
- Scale-prefetch boundary cases, including small or insufficiently aligned scale surfaces.
- API, activation, numerical-result, latency, and tail-behavior comparisons.

### Gate G0

Do not choose the upstream kernel until all of the following are established:

- scale-prefetch safety is present;
- API differences are reconciled;
- identical packed weights produce compatible results;
- sequential shapes are safe.

### Decision rule

Choose a provider only when its pinned revision and required safety behavior are reproducible under the canonical artifact. If any condition is missing, keep the provider unselected and Stage 1 blocked; “newer upstream” is not a substitute for evidence.

**Gate record:** complete the standard template as **G0**, including the chosen provider/revision or an explicit blocked decision.

## Stage 1 — Hardware transport truth

**Ownership:** topology, link negotiation, and DMA; see [`hardware.md`](hardware.md), [`benchmarking.md`](benchmarking.md), and [`risk-register.md`](risk-register.md).

### Inputs

- The reported B70 sysfs value `2.5 GT/s ×1 current and maximum`, treated as unconfirmed.
- Physical slot/root-port/switch, PCIe generation and width, NUMA, ReBAR, IOMMU/ACS, firmware, driver, and available-VRAM data for every card.
- Pinned host buffers and independent CUDA/XPU copy paths.
- Concurrent 5090 and NVMe load generators representative of later operation.

### Work

1. Resolve the suspicious link report before cross-device performance work.
2. Capture negotiated link state at idle and under load.
3. Verify every B70’s independent DMA and compute behavior under simultaneous five-GPU load.
4. Associate measurements with exact topology, driver/firmware, clocks, power, thermals, and throttling state.

### Tests and measurements

Measure:

- negotiated link under load;
- pinned host-to-B70 bandwidth;
- B70-to-pinned-host bandwidth;
- latency for 12, 24, 48, and 96 KiB;
- p50, p95, and p99;
- behavior with concurrent 5090 copy activity;
- behavior with concurrent NVMe activity.

Compatible physical-audit detail from the superseded plan remains required evidence: root port/switch, NUMA node, ReBAR, IOMMU/ACS, actual VRAM, simultaneous GPU/NVMe contention, thermals/throttling, and driver versions.

### Gate G1

The physical link and DMA path must be characterized under load. If the B70 is genuinely Gen1 ×1, stop architecture benchmarking and fix topology, firmware, or slot configuration first. Device-local kernel speed cannot compensate for a broken host link.

### Decision rule

- Confirmed Gen1 ×1: mark **reconsideration required** and do not enter Stage 2.
- A healthy, stable negotiated link with reproducible DMA/tail evidence: pass G1.
- Ambiguous sysfs or load behavior: keep G1 blocked and investigate; do not infer health from idle state.

**Gate record:** complete the standard template as **G1**, with raw link-state captures and contention distributions.

## Stage 2 — One-expert Colibri-to-XPU W4 conversion

**Ownership:** source-format-to-provider-prepack bridge; see [`model-format.md`](model-format.md) and [`correctness.md`](correctness.md).

### Inputs

- G0-pinned provider, revision, safety patch, and canonical artifact.
- Exact Colibri group-128 W4 packing and quantization metadata.
- Exactly one GLM gate/up expert and one matching down expert.
- Colibri CPU dequant/reference, 5090 execution, patched `llm-scaler` B70, and standalone vLLM-XPU B70 comparison paths.

### Work

Convert exactly one gate/up and one down expert. Validate the bridge progressively rather than comparing only its final tensor:

1. nibble values;
2. scales;
3. dequantized rows;
4. gate/up ordering;
5. first GEMM output;
6. SwiGLU output;
7. down output;
8. complete expert output.

### Tests and evidence

Run every comparison path against:

- all-zero input;
- small values;
- saturating quantized values;
- random input;
- repeated identical input;
- changing token rows;
- changing expert IDs;
- zero-row experts.

Record exact packing/metadata mismatches separately from expected numerical quantization drift.

### Gate G2

No bulk conversion until the single expert passes all listed inputs and intermediate checkpoints against the stated references.

### Decision rule

Any unexplained nibble, scale, row-order, intermediate, or final-output mismatch fails G2. Fix the format bridge at its first divergent checkpoint; do not compensate downstream or bulk-convert around it.

**Gate record:** complete the standard template as **G2**, including source and converted artifact hashes and per-checkpoint comparisons.

## Stage 3 — Device-local B70 grouped-GEMM matrix

**Ownership:** exact-shape kernel qualification; see [`model-format.md`](model-format.md), [`benchmarking.md`](benchmarking.md), [`correctness.md`](correctness.md), and [`risk-register.md`](risk-register.md).

### Inputs

- G2-validated expert representation.
- Exact GLM expert shapes:

  ```text
  gate/up: 6144 -> 4096
  SwiGLU:  4096 -> 2048
  down:    2048 -> 6144
  ```

- Token-row counts `1, 2, 4, 8, 16, 32, 64`.
- Synthetic distributions plus historical GLM route counts when available.
- Pinned reference results and steady-state compilation policy.

### Work

Exercise the selected grouped-GEMM path over:

- one active expert;
- two evenly active experts;
- one hot expert plus several one-row experts;
- many zero-row experts;
- one expert receiving all rows;
- historical GLM route distributions when available.

Instrument remap, GEMM1, activation, GEMM2, gather, the complete device-local partial, and allocation/synchronization overhead separately.

### Tests and measurements

- Run all listed row counts and distributions.
- Run multiple shapes sequentially in one process to catch the stale persistent-buffer defect previously observed in the patched INT4 lineage.
- Check output dimensions, reference agreement, NaN/Inf, memory growth, zero-row safety, scale bounds/overread, and steady-state p99.
- Exclude first-call compilation from steady-state numbers while reporting it separately.
- Where captured, include effective weight bandwidth, XMX utilization, prepack cost, command-launch cost, numerical error, power, and clocks; these compatible details from the old kernel stage do not supersede the revised shapes or row matrix.

### Gate G3

All of the following are required:

- exact output shape;
- no stale-buffer reuse;
- no NaN/Inf;
- no memory growth;
- reference tolerance satisfied;
- bounded p99;
- first-call compilation excluded from steady state;
- zero-row experts safe;
- no scale overread.

### Decision rule

A provider/shape is qualified only when every condition holds across the matrix. A fast median cannot waive a correctness, safety, memory, or p99 failure. Do not proceed to the semantic partial on an unqualified kernel.

**Gate record:** complete the standard template as **G3**, attaching per-stage timings and the sequential-shape regression evidence.

## Stage 4 — B70 weighted-partial operator

**Ownership:** provider-neutral expert semantics; see [`expert-fabric.md`](expert-fabric.md) and [`correctness.md`](correctness.md).

### Inputs

- G3-qualified remap, grouped-GEMM, activation, and gather operations.
- Device-local expert-ID mapping for one B70-resident bank.
- Hidden states, B70-local route subsets, route weights, and valid masks from canonical fixtures.
- Placement-epoch metadata.

### Work

Implement the semantic operation:

```text
b70_expert_partial(
    hidden[T,H],
    local_ids[T,Kb],
    weights[T,Kb],
    valid_mask[T,Kb]
) -> partial[T,H]
```

Its initial internal composition is:

```text
remap -> GEMM1 -> SwiGLU -> GEMM2 -> weighted gather
```

Composition from current vLLM-XPU operations is acceptable initially. Fuse or replace orchestration only later, after semantics stabilize and profiling establishes a need.

### Tests and evidence

For each route subset, verify:

$$
\mathrm{B70Partial} \approx \sum_{e\in\text{B70 routes}} w_e\,\mathrm{Expert}_e(x).
$$

Cover:

- no B70 routes;
- one B70 route;
- multiple B70 routes;
- duplicate-ID rejection or explicitly defined handling;
- invalid local ID;
- masked or padded route;
- expert-parallel mapping;
- zero route weight;
- placement-epoch change.

### Gate G4

Every accepted route subset must satisfy the weighted-partial identity within the recorded reference tolerance, with defined behavior for every listed edge/error case.

### Decision rule

Undefined duplicate/invalid-ID behavior, route loss/double counting, epoch confusion, or unexplained numerical divergence fails G4. No cross-vendor model execution may begin until the operator contract is stable.

**Gate record:** complete the standard template as **G4**, recording route fixtures, ID maps, epoch, oracle, and tolerance.

## Stage 5 — Host-staged ring without model execution

**Ownership:** versioned cross-vendor transport and lifecycle; see [`expert-fabric.md`](expert-fabric.md), [`hardware.md`](hardware.md), [`correctness.md`](correctness.md), and [`risk-register.md`](risk-register.md).

### Inputs

- G1-characterized CUDA-to-host and host-to-XPU paths.
- Preallocated pinned host storage.
- Explicit producer/consumer ownership, completion, generation, timeout, and shutdown state.
- A vector or no-op B70 kernel; **no model execution**.

### Work

Implement only:

```text
5090 D2H
 -> pinned ring
 -> B70 H2D
 -> vector/no-op kernel
 -> B70 D2H
 -> pinned ring
 -> 5090 H2D
```

The ring must be versioned so a late completion cannot be mistaken for current work. Resource ownership must survive backpressure, errors, restart, wraparound, and shutdown.

### Tests and measurements

- one slot;
- multiple slots;
- backpressure;
- wraparound;
- stale generation;
- timeout;
- producer crash;
- B70 worker restart;
- simultaneous request completion;
- buffer reuse;
- shutdown with work in flight.

Measure p50/p95/p99 and CPU consumption for one B70 and four B70s concurrently under an idle system, NVMe traffic, CPU cold compute, and 5090 attention-like/copy load. These are compatible stress dimensions retained from the superseded empty-round-trip stage.

### Gate G5

- No allocation in the hot loop.
- No polling that consumes an entire CPU core without evidence that it is necessary.
- No buffer is reused before both runtimes have completed with it.
- The listed lifecycle and failure cases complete without stale data, ownership violation, or global synchronization.

### Decision rule

Any premature reuse, ambiguous completion, stale-generation acceptance, unbounded p99, global synchronization, or unrecovered lifecycle failure blocks G5. Optimize waiting only from measured evidence; do not trade correctness for a low synthetic median.

**Gate record:** complete the standard template as **G5**, with slot-state traces, allocation evidence, CPU utilization, and latency distributions.

## Stage 6 — Transport plus the actual B70 partial

**Ownership:** remote-path crossover and batch eligibility; see [`expert-fabric.md`](expert-fabric.md), [`hardware.md`](hardware.md), and [`benchmarking.md`](benchmarking.md).

### Inputs

- G4-qualified `b70_expert_partial`.
- G5-qualified host ring.
- Identical route/input fixtures for remote B70, CPU fallback, and 5090-local paths.
- Token-batch classes from batch one through batch 64.

### Work

Replace the ring’s vector/no-op kernel with `b70_expert_partial`. Measure the complete remote critical path:

$$
\begin{aligned}
T_{remote}={}&T_{cuda\_d2h}+T_{queue}+T_{b70\_h2d}+T_{remap}\\
&+T_{gemm1}+T_{activation}+T_{gemm2}+T_{gather}\\
&+T_{b70\_d2h}+T_{cuda\_h2d}+T_{join}.
\end{aligned}
$$

Compare it with $T_{CPU\ fallback}$ and $T_{5090\ local}$ for every token-batch class. Preserve both component and end-to-end distributions.

### Tests and measurements

- Batch one through batch 64.
- p50/p95/p99 for every component and full path.
- Correctness checks on the returned weighted partial.
- Contention representative of 5090 owner work, NVMe activity, concurrent sessions, and background requests.
- Allocation, synchronization, queue-depth, timeout, and fallback observations.

### Decision gate G6

For each batch class, B70 is eligible only if:

$$
T_{remote,p99}<T_{CPU\ fallback,p99}.
$$

If B70 loses at batch one but wins at batch four or larger, it may be used for batched decode, prefill, concurrent sessions, and background requests. It must not be forced into latency-critical batch-one decode merely because device-local GEMM is fast.

### Decision rule

Record eligibility separately by batch/workload class. If host-staged p99 cannot beat CPU fallback for **any useful batch class**, trigger architecture reconsideration rather than continuing to Stage 7. A win only against the 5090-local path is not the stated eligibility test.

**Gate record:** complete the standard template as **G6**, with an explicit eligible/ineligible table by batch and workload class.

## Stage 7 — Small end-to-end model

**Ownership:** first real semantic integration and failure behavior; see [`architecture.md`](architecture.md), [`expert-fabric.md`](expert-fabric.md), [`model-format.md`](model-format.md), [`correctness.md`](correctness.md), and [`risk-register.md`](risk-register.md).

### Inputs

- Qualified G0–G6 provider, conversion, device-local partial, and transport path.
- Existing Qwen 35B model and its all-5090 or current Lucebox reference.
- Deterministic Colibri GLM fixture.
- Exact standalone GLM expert-shape benchmarks.
- Fixed prompts, teacher-forced tokens, canonical router outputs, and route ownership fixtures.

### Work

Use two distinct development targets:

1. **Semantic integration target — Qwen 35B.** Use it because the current vLLM-XPU kernels directly support Qwen-style MoE, it fits the available hierarchy more realistically than full GLM, a reference path exists, and it exercises actual routing and multi-token generation.
2. **GLM architecture target — deterministic Colibri GLM fixture plus exact standalone GLM shapes.** This validates the future GLM contract without claiming that the full 372 GB model is memory-resident.

Execute all sparse layers for the semantic target through the fabric. Exercise multi-token generation, multi-turn state, concurrent requests, and deliberate B70 failure. Use the GLM fixture to validate the owner/router/provider/join boundary before full-model residency work.

### Tests and evidence

For fixed prompts and teacher-forced tokens:

- router IDs match the canonical owner;
- B70 receives only its assigned route subset;
- 5090 and B70 partials join exactly once;
- no route is lost or counted twice;
- top-1 logits/tokens remain consistent;
- numerical drift is bounded and characterized;
- multi-turn state remains coherent;
- B70 timeout falls back to CPU;
- stale output is ignored;
- warmed generation performs zero NVMe expert reads.

Also test concurrent requests and an intentional B70 failure, preserving the compatible end-to-end coverage from the superseded small-model stage.

### Gate G7

All listed route, numerical, token, state, failure, stale-output, and warmed-storage conditions must be supported by end-to-end evidence on their applicable target. The deterministic GLM fixture is evidence for the GLM contract, not a claim that full GLM is resident or integrated.

### Decision rule

Any material route/token change, lost/doubled route, incoherent state, incorrect fallback, accepted stale result, or unexplained drift fails G7. Regular warmed expert reads from NVMe trigger architecture reconsideration. Do not proceed on Qwen throughput alone.

**Gate record:** complete the standard template as **G7**, separating Qwen evidence from GLM-fixture evidence and correctness from performance.

## Stage 8 — Static expert placement

**Ownership:** first placement policy and baseline comparison; see [`placement.md`](placement.md), [`benchmarking.md`](benchmarking.md), and [`architecture.md`](architecture.md).

### Inputs

- G7-correct model path.
- Learned per-layer routing frequencies from Lucebox/Colibri-style traces.
- Measured per-tier memory capacities and G6 batch eligibility.
- Three baselines: CPU-only cold execution, frequency-only 5090 placement without B70, and random/uniform B70 placement.

### Work

Start with static placement only:

```text
5090: hottest experts
B70:  next warm expert bank
CPU:  remaining exact experts that fit
NVMe: recovery/preload
```

There is no swapping during a token, prediction, or live migration in this stage. The B70 serves a resident expert bank.

### Tests and measurements

Measure:

- hit share per tier;
- B70 token batch sizes;
- experts contacted per token;
- CPU fallbacks;
- remote calls per layer;
- B70 queue depth;
- placement memory;
- total decode rate;
- p95/p99 latency;
- output agreement and fallback/failure counts.

Run the same requests/traces and measurement contract for the proposed placement and all three baselines.

### Gate G8

Static B70 placement must beat:

1. CPU-only cold execution;
2. frequency-only 5090 placement without B70;
3. random/uniform B70 placement.

### Decision rule

If static placement does not beat all three baselines under comparable correctness and workload conditions, do not proceed to a more complex placement algorithm. Diagnose transport, queueing, fan-out, residency, or workload fit instead.

**Gate record:** complete the standard template as **G8**, with baseline-identical trace/input IDs and per-tier metrics.

## Stage 9 — Placement optimization

**Ownership:** traffic- and cost-aware placement; see [`placement.md`](placement.md), [`benchmarking.md`](benchmarking.md), and [`risk-register.md`](risk-register.md).

### Inputs

- G8-winning static placement as the fixed baseline.
- The same frozen route-trace replay used for baseline comparisons.
- Session and recent-window frequencies, expert coactivation, layer criticality, per-expert compute time, fan-out, queue delay, transport cost, and capacity data.
- Separate prefill and decode observations.

### Work

Only after static placement works, add and evaluate:

- session-frequency weighting;
- recent-window weighting;
- expert coactivation;
- layer criticality;
- per-expert measured compute time;
- expected device fan-out;
- queue delay;
- transport cost;
- expert replication;
- prefill/decode-specific placement;
- promotion and demotion.

The target score is:

$$
\operatorname{benefit}(e,d)=f_e\left(T_{fallback,e}-T_{device,e,d}-T_{transport,d}-T_{queue,d}\right)-\lambda_{fanout}\Delta F,
$$

subject to B70 and 5090 memory capacity. Frequency is the baseline, not the final objective.

### Tests and measurements

- Introduce changes incrementally and replay the same trace against the static baseline.
- Measure p50/p95/p99 layer and end-to-end time, decode rate, queue delay, fan-out, tier hit share, migrations, fallbacks, memory, and output agreement.
- Separate prefill/decode and session/global behavior.
- Stress capacity boundaries, promotion/demotion churn, replication, placement-epoch changes, and workload shifts.

### Gate G9

The revised plan does not supply a new numeric exit threshold for this stage. The compatible hard gate retained from the superseded placement stage is therefore controlling: optimization must lower p95/p99 layer time on the same route-trace replay **without correctness regression or migration instability**.

### Decision rule

Accept only changes with reproducible critical-path benefit and stable residency under capacity constraints. Reject added complexity that merely reduces misses while increasing queueing, fan-out, tail latency, migration churn, or semantic error. Do not tune kernels to hide a placement-policy loss.

**Gate record:** complete the standard template as **G9**, identifying each policy increment and its delta from static placement.

## Stage 10 — Kernel tuning

**Ownership:** evidence-driven provider optimization; see [`benchmarking.md`](benchmarking.md), [`model-format.md`](model-format.md), [`correctness.md`](correctness.md), and [`risk-register.md`](risk-register.md).

### Inputs

- Stable, correct G4 partial semantics and G5 transport.
- G7 end-to-end model fixtures and G9 placement/workload distributions.
- Structured per-component profiles from `vllm-xpu-breakdown`.
- Baseline pinned kernel/provider, exact converted weights, route traces, and memory measurements.

### Work

Only now consider:

- Xe-Forge tile search;
- Xe-Fuse fused-epilogue ideas;
- Triton MXFP4 comparison;
- persistent scratch and atomic buffers;
- dedicated small-$M$ policies;
- route-aware batching;
- remap/gather fusion.

Use `vllm-xpu-breakdown` to rank actual recovered time, not isolated TFLOPS. Any alternative format remains an explicit conversion-and-validation path rather than a presumed compatible replacement.

### Tests and measurements

For every candidate, compare with the pinned baseline on identical artifacts and real route distributions:

- isolated GEMM and full device-local partial;
- transport-plus-partial;
- end-to-end generation;
- exact/within-tolerance output evidence;
- p50/p95/p99;
- memory use and persistent-buffer lifecycle;
- first-run and warmed behavior;
- all regression shapes, zero-row cases, and sequential shapes.

### Gate G10

A faster isolated GEMM is accepted only if:

- the device-local full partial is faster;
- transport-plus-partial is faster;
- end-to-end generation improves;
- output remains correct;
- p99 does not regress;
- memory does not grow;
- the improvement survives real route distributions.

### Decision rule

All conditions are conjunctive. If any condition fails, retain the prior provider/kernel path. No tuning result may weaken the format, scale-overread, stale-buffer, route, or numerical gates.

**Gate record:** complete the standard template as **G10**, with baseline/candidate revisions and deltas at all three performance scopes.

## Stage 11 — Full GLM integration

**Ownership:** final model-owner/provider integration; see [`architecture.md`](architecture.md), [`expert-fabric.md`](expert-fabric.md), [`hardware.md`](hardware.md), [`model-format.md`](model-format.md), [`placement.md`](placement.md), [`correctness.md`](correctness.md), [`benchmarking.md`](benchmarking.md), and [`risk-register.md`](risk-register.md).

### Inputs

- Explicit passed records for G0 through G10.
- The vetted expert-provider ABI, GLM W4 conversion, B70 weighted partial, host ring, eligible batch classes, placement profile, and tuned-or-retained kernel.
- State-owner 5090 path, shared experts, exact CPU fallback, model/placement manifests, and storage plan.
- Long-context and incremental multi-B70 test plans.

### Work

Only after all preceding gates:

1. add an expert-provider boundary to Colibri;
2. preserve the 5090 model-owner path;
3. load selected GLM experts into B70-resident W4 storage;
4. keep shared experts on the 5090;
5. execute routed B70 subsets;
6. combine deterministic partials;
7. retain CPU fallback;
8. treat NVMe as preload/recovery where possible;
9. test long-context KV pressure;
10. add more B70s incrementally.

Build and verify model, packed-bank, and placement manifests; boot-time health checks; profile persistence; per-tier capacity reservations; and worker quarantine/restart behavior where these are needed by the full configuration.

### Tests and evidence

- Re-run the G7 semantic suite with full GLM, including fixed prompts, teacher-forced tokens, route ownership, deterministic joins, top-1 consistency, bounded drift, multi-turn state, timeout fallback, and stale-output rejection.
- Exercise long-context KV pressure and memory admission.
- Add B70s one at a time; repeat correctness, queue, transport, failure, and p99 measurements at each count.
- Verify resident-bank behavior and report every foreground NVMe expert read.
- Compare prefill/decode, batch-one/continuous batching, first-run/warmed, and device-local/end-to-end behavior separately.
- Exercise concurrent requests, worker quarantine/restart, and graceful degradation without allowing the B70 queue to serialize the 5090 critical path.

### Gate G11

G11 is the integration acceptance gate, not permission to start integration. Entry requires G0–G10 to be passed. Exit requires the full-configuration semantic, failure, storage, memory-pressure, incremental-device, and end-to-end evidence above, with no hidden exception to the architecture invariants.

The plan records a 64 GB DDR5 configuration in which a fully RAM-resident 372 GB GLM model is impossible. Treat that capacity as a configuration fact to re-verify in the gate environment. The first full-GLM run may still require a storage tier for experts not resident across the 5090, B70, and RAM; such a run must be reported honestly and must not be labeled no-NVMe.

### Decision rule

- Missing prior gate: Stage 11 remains blocked.
- Semantic/failure invariant violation, material route/token drift, critical-path serialization, or regular warmed NVMe expert reads: fail or require architecture reconsideration according to the hard-stop rules below.
- Storage-backed first run: characterize it honestly; it is not evidence of a no-NVMe configuration.
- Additional B70s: accept incrementally only when each added configuration preserves correctness and improves or maintains the applicable end-to-end acceptance metrics without a p99 or memory regression.

**Gate record:** complete the standard template as **G11**, listing all prerequisite gate records and separating first full run from warmed steady state.

## Measurement contract for all stages

Every applicable benchmark emits one structured record per request or token with:

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

Aggregate and report:

- p50/p95/p99, not only averages;
- prefill and decode separately;
- batch one and continuous batching separately;
- first-run and warmed steady state separately;
- device-local and end-to-end separately;
- exact correctness evidence separately from performance.

See [`benchmarking.md`](benchmarking.md) for benchmark methodology and reporting ownership.

## Architecture reconsideration conditions

Stop advancement and reconsider the architecture if any of the following is observed:

1. B70 PCIe remains Gen1 ×1.
2. Host-staged p99 cannot beat CPU fallback for any useful batch class.
3. W4 format conversion cannot reproduce the canonical expert result.
4. Cross-vendor numerical drift changes route/token behavior materially.
5. B70 queueing serializes the 5090 critical path.
6. Normal warmed decode regularly reads experts from NVMe.
7. B70 requires frequent weight migration instead of serving a resident bank.
8. More than a negligible number of requests require timeout recovery.
9. PyTorch/XPU worker overhead dominates the kernel.

Only condition 9 has a prescribed implementation-only response: preserve the architecture but replace the worker implementation with a native SYCL/C++ service around the vetted grouped-GEMM kernel. The plan does not define “negligible” numerically; the workload-specific threshold must be stated before measuring rather than selected after seeing results.

## Final revised priority

The operational priority is ordered and must remain consistent with Stages 0–11:

1. Pin and A/B current `vllm-xpu-kernels` against the `llm-scaler` patched version.
2. Carry forward the missing Xe2 scale-prefetch safety guard.
3. Resolve the B70 PCIe/DMA report.
4. Convert and validate one Colibri group-128 W4 expert.
5. Benchmark exact GLM gate/up and down shapes on B70.
6. Build the B70 weighted-partial operation from remap plus two grouped GEMMs plus gather.
7. Build the versioned pinned-host ring.
8. Measure batch-one through batch-64 transport-plus-compute crossover.
9. Integrate Qwen 35B as the first real end-to-end heterogeneous model.
10. Integrate the deterministic Colibri GLM fixture.
11. Add static expert placement.
12. Add profiling markers and headless breakdown reports.
13. Only then use Xe-Forge, Xe-Fuse, or Triton to optimize kernels.
14. Finally integrate full GLM and additional B70s.

The upstream XPU kernel repository means the B70 expert math need not be designed from scratch **if its local gates pass**. The remaining new engineering is the weight-format bridge, partial-expert adapter, resident-weight service, cross-vendor host ring, failure semantics, and placement scheduler. This is a design conclusion, not a completion claim.

## Superseded implementation sequence

The earlier `readme.md` **Stage 0–8** sequence is superseded by the revised Stage 0–11 sequence in this document. It is not a second active plan, and its stage numbers or gates must not be used to authorize work out of order.

Compatible detail from that older sequence has been incorporated only where the revision left it useful and non-conflicting: the physical topology inventory in Stage 1, kernel measurement fields in Stage 3, transport contention cases in Stage 5, end-to-end failure/concurrency coverage in Stage 7, and the same-trace placement stability gate in Stage 9. Conflicting or displaced requirements are controlled by the revision. In particular:

- the revised exact GLM shapes and row matrix replace the older abbreviated one-expert benchmark definition;
- the revised host-ring and crossover stages replace the older empty-round-trip gate;
- Qwen 35B plus the deterministic Colibri GLM fixture replace an unspecified “small supported MoE model” as the active development targets;
- the revised full-GLM capacity statement forbids claiming a no-NVMe run when the configured resident tiers cannot hold the model;
- the old service-scheduling and optional SGLang stages are not gates in this active sequence and cannot be used to bypass any revised gate.
