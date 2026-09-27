# Shooting Brake: executable behavioral specification

This is an understanding-first description of the checked-out implementation, organized breadth first: read this system overview, then the subsystem chapters, then the actions and executable scenarios. Source code is authoritative. A model is an explicit abstraction, not a claim of line-for-line translation or a proof about GPU drivers.

## 1. High-level design

Shooting Brake serves a mixture-of-experts language model through a vLLM plugin. CUDA owns attention, routing, shared experts and the selected local routed experts. Remote routed experts can live on one or more Intel B70 devices; an optional all-out configuration adds a CPU/DRAM tier. Per-layer partial results meet on CUDA before model execution continues. Offline programs prepare expert banks and reference data; a separate collection of campaigns, probes and research programs measures or changes proposed implementations.

```text
                             client / benchmark / campaign
                                          |
                             vLLM API, scheduler, model
                                          |
                           Shooting Brake plugin and admission
                                          |
                         immutable ownership and compact slots
                                          |
                         one routed-expert layer invocation
                          /               |                \
                CUDA local experts    B70 lane(s)       optional CPU tier
                          |          pinned staging          |
                          |          issue / take            |
                          |          SYCL / native           |
                          \_______________|__________________/
                                          |
                           combine routed partial results
                                          |
                         vLLM continues / publishes tokens
```

The scheduler, attention implementation, HTTP stack and CUDA expert kernels largely belong to vLLM and other dependencies. The specification explains their boundary with Shooting Brake; it does not pretend their internals were reimplemented in Reyu.

## 2. Breadth-first subsystem map

| Subsystem | Responsibility | Source anchor | Detailed branch |
|---|---|---|---|
| System lifecycle | Startup outcome, routing, concurrent partial work, join, output validity | `serve_production.sh`; `routed_experts.py` | [System](chapters/system.md) |
| Configuration and admission | Model qualification, environment flags, supported bank/model shapes, incompatible mode rejection | `config.py`; plugin `__init__.py`; serving recipes | [Admission and launch](chapters/admission.md) |
| Placement and partition | One normal-path owner per expert; compact local slots; split routed work without omission or duplication | `placement.py`, `partition.py`; phase5/6 gates | [Placement](chapters/placement.md) |
| Banks and reference data | Extract, encode, validate and load NVFP4/int4/Marlin/B12x representations; numerical reference provenance | phase0/1/3; plugin bank modules | [Banks and validation](chapters/banks.md) |
| vLLM integration | Register replacements, qualify construction, runner seam, shared experts, hidden-state capture | plugin registration, runner, fusion and capture | [Integration](chapters/integration.md) |
| Hybrid execution | Local/remote masks, synchronous/asynchronous/graph execution and joining partials | `routed_experts.py`, `provider.py`, `expert_bank.py` | [Hybrid execution](models/hybrid.ryu) |
| Weight residency and prefill | Preemptive/post-load surgery, mirrors, chunking, Marlin/B12x alternatives and CPU streaming | plugin prefill and routed-expert modules; phase8 | [Prefill and residency](chapters/prefill.md) |
| Native B70 provider | Bank lifetime, shape/identity checks, single-flight issue/take, output format and queue completion | `src/phase1/b70_provider.{hpp,cpp}` | [Provider](chapters/provider.md) |
| Graph-compatible signaling | Per-card/per-layer staging, signal value as batch size, poller ownership, completion and replay reset | plugin poller/signals; phase7 C ABI; phase9 helpers | [Doorbell](chapters/doorbell.md) |
| Command-streamer alternatives | Experimental hardware waits, chains and captured/baked dispatch alternatives | phase1 provider; phase7 `b70_capi.cpp`; CS experiments | [Command streamer](models/command_streamer.ryu) |
| Shared-ring transport | Slot ownership, sequence/generation identity, cancellation, deadlines and retirement | `src/phase2/` | [Ring transport](chapters/ring.md) |
| CPU expert tier | Packed expert storage, routing, host compute, asynchronous stream/poller completion | plugin CPU modules; phase7 CPU C ABI | [CPU tier](chapters/cpu.md) |
| Native compute | NVFP4/int4 decode and grouped prefill; packing, gather/scatter, accumulation and wire narrowing | phase1; `src/phase7/xe2_nvfp4/` | [Kernel contracts](chapters/compute.md) |
| Embedded QuixiCore-XPU | Native operation families, runtime/device selection, public dispatch and command-graph lifecycle | `src/QuixiCore-XPU/` | [Runtime](chapters/quixi_runtime.md), [operations](chapters/quixi_ops.md) |
| Benchmark clients and matrices | Identity checks, cold/warm workloads, SLO matrices, comparisons and chat/tool clients | `benchmarks/` | [Benchmark workflows](chapters/benchmarks.md) |
| Profiling and campaigns | Workload execution, measurements, manifests, mixed traffic and scheduler qualification | campaign/profile/interactive scheduler programs | [Campaign workflows](chapters/campaign.md) |
| Route/trace analysis | Binary route traces, accounting, locality, topology and measurement interpretation | route analysis, telemetry and trace modules | [Observability](chapters/observability.md) |
| Drafter research | Data generation, target hidden capture, warmstart, training, export and acceptance | `experiments/drafter_*` | [Drafter](chapters/drafter.md) |
| Sparse-memory research | Independent learned-memory experiments and training/evaluation protocols | `experiments/sparse-memory/` | [Sparse memory](chapters/sparse_memory.md) |
| Hardware probes | Transport, host registration, DMA, device memory and synchronization experiments | native experiment sources; pinned-staging probes | [Hardware experiments](chapters/hardware_probes.md) |
| Numerical and performance probes | Quality gates, shape/arm identity, sample aggregation and interpretation | benchmark probe scripts | [Performance experiments](chapters/performance_probes.md) |
| Qualification and build evidence | Build linkage, parity/path checks and historical comparison outputs | phase7 harnesses; phase10/results | [Qualification](chapters/qualification.md) |

A phase directory is a development-history grouping, not an architectural component. The chapters preserve the relationship without making the reader follow historical phase numbers.

## 3. Three distinctions that apply everywhere

### Behavior versus configuration

The README describes an 85/85 remote expert split; the executable r15 recipe defaults to 95/75. Recipe comments discuss 512 batched tokens while the executable default is 2048. Environment overrides can select still other configurations. The models parameterize meaningful choices and identify the recipe being discussed rather than silently treating explanatory text as live configuration.

### Completion versus success

The classic native poller in `src/phase7/b70_capi.cpp` publishes completion even after provider failure to release CUDA's untimed wait. A completion flag therefore does **not** certify a valid output. Detailed models distinguish release, result validity and error observation. Any stronger safety property must be checked rather than assumed.

### CUDA graph signaling versus hardware command-streamer execution

`SHOOTING_BRAKE_B70_GRAPH` selects a CUDA-capture-compatible handoff. It is not the same switch as `SHOOTING_BRAKE_B70_CS_DOORBELL`. The r15 recipe sets graph mode but does not itself set the CS switch; inherited overrides and native selection rules must be considered separately. Experimental command-streamer branches are not silently substituted for the classic host polling path.

## 4. How to read deeper

At each branch, read in this order:

1. Responsibility, actors, inputs/outputs and source anchors.
2. State and atomicity boundary: what can interleave and what is collapsed.
3. Named actions: guards, changes, preservation, failures and mode branches.
4. A concrete `run ...Test` scenario.
5. Invariants, reachable witnesses and temporal assumptions.
6. Abstractions and implementation-comparison evidence.

The models use small finite identities to explore ordering and ownership. They do not substitute these small values for production resource sizing. Tensor payloads are abstract unless a functional/numerical contract is explicitly modeled. Floating-point error, PCIe timing, cache coherence and kernel performance require implementation evidence, not just a successful model run.

## 5. Coverage and evidence meaning

The [source inventory](inventory.json) and per-branch `maps/` distinguish behavioral models, functional contracts, harnesses, embedded/external dependency boundaries, and generated evidence. Source hashes record which implementation was inspected. A file being listed is **not** proof that every statement in it was formally verified. Each detailed chapter states its granularity and remaining assumptions.

The verification report records actual commands and outcomes separately for parsing/type/effect checking, compilation to Reyu JSON IR, executable scenarios, sampled invariant exploration and implementation comparisons. No successful simulation is described as exhaustive verification; no compiled model is described as automatic implementation equivalence.

Original implementation files are not changed by this specification work. Large banks, virtual environments, caches, profiler captures and benchmark results are inputs or evidence, not new state-machine implementations. Separate checkouts under `vendor/`, `experiments/misc/` and QuixiCore's `.reference/` are documented at dependency boundaries rather than falsely claimed as first-party code.

## 6. Run the specification

From the Shooting Brake repository root, with the local Reyu checkout built:

```sh
python3 specs/reyu/check.py --reyu "$HOME/reyu/bin/reyu"
.venv/bin/python specs/reyu/compare.py --reyu "$HOME/reyu/bin/reyu"
```

`REYU_BIN` can supply another executable or a built `cli.js` path. `check.py` checks source hashes and navigation, typechecks and compiles every registered model to JSON IR, executes named scenarios, and explores each invariant with reproducible samples. `--model system` narrows execution while retaining the whole-catalog check. `--validate-only` checks correspondence metadata without executing models. Compilation artifacts are temporary; their hashes and sizes are retained in the report.

`compare.py` derives expected values from the actual implementation and checks them against model functions. It does not duplicate the expected algorithms in a second Python implementation. Where a pure function/class is loaded from its original AST to avoid importing vLLM or model weights, the report says so explicitly. This checks those source algorithms, not their framework integration.

The comparison runner uses the project's NumPy installation for real route-record arrays; another Python with NumPy installed is also suitable. It does not load model weights.

The [ring chapter](chapters/ring.md) also describes the added native, CPU-only correspondence harness against the real shared-ring implementation. None of these commands launches the production server or a GPU benchmark.

After implementation changes, review the affected chapter/model and update its source correspondence deliberately. Do not simply refresh a hash to silence source drift: the hash indicates that the explanation needs review, not that the new source is wrong.

## 7. Recorded verification

The completed verification run covers **21 specification branches**, **29 executable models** plus their shared architecture observations, and **427 inventoried source/build/evidence files**. These counts describe the declared inventory and abstraction boundaries, not statement-by-statement formal verification of the repository.

| Check | Observed result | Evidence |
| --- | --- | --- |
| Source identity, owning-branch correspondence, model registration and navigation | No catalog errors; all 427 recorded source hashes match | [Verification report](verification.json) |
| Reyu type/effect checking and JSON IR compilation | All 29 executable models passed | [Verification report](verification.json) |
| Named executable scenarios | All 335 passed with seed 42 | [Verification report](verification.json) |
| Sampled invariants | All 29 model runs passed, configured for 100 samples and at most 80 steps each | [Verification report](verification.json) |
| Actual Python implementation correspondence | All 1,686 bounded input/output cases passed across nine model functions | [Comparison report](implementation-comparison.json) |
| Real native shared-ring API correspondence | Five sequential CPU scenarios matched | [Native report](native-ring-comparison.json) |

The Python comparisons execute actual source for weighted placement apportionment, NVFP4 expert/bank sizing, validated int4 header offsets, interval union/intersection, interactive scheduling budgets, consecutive route runs, and capacity-two LRU accounting. Assertions are grouped into batches to avoid repeated evaluator setup; no input cases are omitted. The report retains every input and source-derived expected result.

The native harness exercises success/reuse, pre-claim cancellation, post-claim cancellation/quarantine, stale completion, and generation retirement. Its post-claim test uses cancellation, **not deadline expiry**. It is not a concurrency stress test.

These runs did **not** launch production serving or GPU workloads, prove kernel numerical parity or performance, exhaustively model-check temporal properties, or establish a refinement proof between all implementation code and the models. Each chapter states its atomicity, finite-domain, external-dependency and numerical abstractions. Source-only comparison opportunities listed in chapters are not additional executed checks; the reports above identify what was actually exercised.
