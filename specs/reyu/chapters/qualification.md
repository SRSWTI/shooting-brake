# Qualification, oracle, and build-evidence workflows

[The executable qualification model](../models/qualification.ryu) describes this branch as an assertion workflow around the serving system, not as another serving backend. Source files are authoritative; test names, comments, saved measurements, and a passing assertion are evidence about a particular configuration, not production behavior or a universal guarantee.

## 1. High-level responsibility and boundaries

This branch answers five questions breadth first:

1. **Can the native pieces be composed?** The Phase-7 Makefile links the Phase-1 B70 provider, Phase-7 grouped kernels, QuixiCore-XPU, Level Zero, and oneDNN into one B70 library, while deliberately building the CPU expert library with stock `g++` and pthreads.
2. **Was the intended path actually exercised?** Real-model harnesses demand a hybrid marker, provider/poller counter movement, CPU-route counts, or other positive evidence rather than accepting plausible text from an accidentally all-CUDA run.
3. **Does a candidate agree with an appropriate oracle?** Exact token equality is used where two configurations should be semantically identical; semantic text, prompt logprobs, numerical arrays, or source-level ownership assertions are used where kernels legitimately differ in low bits.
4. **Do risky state and representation boundaries hold?** Tests cover preemptive compact expert maps, missing-bank refusal, result-wire dtype, grouped SwiGLU overflow scaling, scratch reuse, and invalid-route zeroing.
5. **What was measured?** Chunk and crossover programs measure hypotheses and deployment thresholds. Phase-10 JSON files preserve one historical all-CUDA/hybrid comparison. Neither class of evidence changes runtime policy by itself.

The actors are the harness process, fresh child vLLM engines, the Shooting-Brake plugin, the actual provider libraries, and numerical/reference implementations. Provider lifecycle belongs to the provider branch; placement ownership belongs to placement; grouped math belongs to compute; graph/poller lifecycle belongs to runtime/doorbell; CPU execution belongs to the CPU tier. This chapter models only how these are selected, observed, compared, and accepted or rejected.

## 2. Source inventory and interface table

| Source | Role in this branch | Principal interface or assertion |
|---|---|---|
| [`src/phase7/Makefile`](../../../src/phase7/Makefile) | Build linkage | `all`, `b70`, and `cpu` targets; B70 composition and deliberately independent CPU build. |
| [`src/phase7/hybrid_b70_test.py`](../../../src/phase7/hybrid_b70_test.py) | Real-model integration harness | `run_case`, `main`: all-CUDA versus `split:128` hybrid, marker required, exact-token or narrowly semantic `42` acceptance. |
| [`src/phase7/allout_correctness_test.py`](../../../src/phase7/allout_correctness_test.py) | Real-model all-out assertion harness | `run_case`, `main`, `check`: policy-equivalence run, live CPU run, layout verification, telemetry health and coherent-output assertions. |
| [`src/phase7/preemptive_surgery_test.py`](../../../src/phase7/preemptive_surgery_test.py) | CPU-executable source/contract assertions | Exact/disjoint ownership, sorted compact CUDA ids, `-1` masks, stock all-CUDA tail, loader-name tripwire, initialization ordering, accessor consistency, missing-bank refusal. |
| [`src/phase7/nvfp4_testutil.py`](../../../src/phase7/nvfp4_testutil.py) | Shared numerical fixture | `make_plane`, `dequant`, `make_expert`, `ffn`: packed NVFP4 bytes and a vLLM dequantization/FP32 SwiGLU oracle for CPU-tier gates. |
| [`src/phase7/prefill_probe.py`](../../../src/phase7/prefill_probe.py) | Real-model numerical/provider-health probe | `main` plus diagnostic installers: prompt-logprob completeness and finiteness, generated-logprob finiteness, provider identity/health/generation/dispatch, graph-poller activity. |
| [`src/phase7/prefill_probe.sh`](../../../src/phase7/prefill_probe.sh) | Multi-configuration launcher/comparison | `run_case`: all-CUDA, Phase-6 offload, forced streaming, and all-out recipes; cached outputs and a descriptive prompt-logprob table. |
| [`src/phase7/prefill_chunk_bench.py`](../../../src/phase7/prefill_chunk_bench.py) | Mechanism benchmark | `measure`, `run`, `main`: warm the real long prompt, collect TTFT and derived prefill throughput/dispatch count, retain worker telemetry. |
| [`src/phase7/prefill_chunk_bench.sh`](../../../src/phase7/prefill_chunk_bench.sh) | Fresh-process sweep launcher | One process per `SHOOTING_BRAKE_B70_MAX_BATCH` because pinned buffers are construction-sized; sweeps chunks, contexts, and trials. |
| [`src/phase7/stream_crossover_bench.py`](../../../src/phase7/stream_crossover_bench.py) | CPU-versus-streaming deployment calibration | `build_routes`, `timed`, `main`: sample realistic cold routes, compare CPU arena execution against CUDA streaming, report first measured crossover. |
| [`tests/test_b70_output_wire.py`](../../../tests/test_b70_output_wire.py) | Actual-provider result-wire and grouped-kernel assertions | `_write_unit_bank`, fixtures, scalar wire test, grouped range/scratch/empty-route test across both pipeline values and FP16/FP32 wire modes. |
| [`src/phase10/results/all-cuda.json`](../../../src/phase10/results/all-cuda.json) | Generated historical artifact | Captured all-CUDA correctness, single-stream, batch, prefill and worker measurements. No executable symbol or runtime effect. |
| [`src/phase10/results/hybrid.json`](../../../src/phase10/results/hybrid.json) | Generated historical artifact | Captured `subset:16:8` hybrid measurements for the same report shape. No executable symbol or runtime effect. |
| [`src/phase10/results/comparison.json`](../../../src/phase10/results/comparison.json) | Generated comparison artifact | Derived token agreement, KV capacity, throughput and inter-token-latency comparison. No executable symbol or runtime effect. |

## 3. Build linkage: evidence, not execution semantics

The [Makefile](../../../src/phase7/Makefile) has two intentionally different products.

* `libsb_b70_provider.so` is built with `icpx -fsycl`. Its link graph includes `b70_capi.cpp`, the Phase-1 `b70_provider.cpp`, the locally modified grouped-MoE object, the separate oneDNN grouped object, QuixiCore metadata and operations libraries, Level Zero, and oneDNN. The Xe2 kernel headers are explicit prerequisites because those headers contain kernel code; otherwise a header edit can leave a stale object and produce a false backend-disagreement result.
* `libsb_cpu_expert.so` is built from `cpu_expert_capi.cpp` with stock `g++`, pthreads, and `-march=native`, with no SYCL objects or link flags. This is a buildability and dependency-isolation contract, not evidence that CPU results match another backend.
* The grouped oneDNN translation unit is kept separate so oneDNN headers do not pollute the CUTLASS-SYCL compile and SPIR-V extension flags do not reach that object.

The model action `inspectBuildLink` therefore records two source-inspection facts—B70 composition and CPU independence. It does not claim that invoking these targets succeeded in this authoring session, and it does not duplicate provider or kernel state machines.

## 4. Real-model parity and path-exercise constraints

### 4.1 Hybrid B70 parity

[`hybrid_b70_test.py`](../../../src/phase7/hybrid_b70_test.py) runs two fresh eager engines over the prompt “State the integer after 41.” The baseline forces `all-cuda`. The candidate uses `split:128`, hybrid mode, and B70 device selection. Each child writes token ids and text to a temporary JSON file.

Acceptance is ordered:

1. child execution must return successfully;
2. the candidate must write `SHOOTING_BRAKE_HYBRID_MARKER`, proving the intended hybrid path was reached;
3. exact token equality passes; otherwise both normalized texts must contain `42`.

The semantic fallback is deliberately narrow and recognizes expected NVFP4 cross-kernel numerical drift. It is not general semantic equivalence. A matching answer without the marker fails because it could be an all-CUDA false positive.

### 4.2 All-out plumbing and live CPU work

[`allout_correctness_test.py`](../../../src/phase7/allout_correctness_test.py) separates policy plumbing from CPU numerics with three child engines:

* `subset:16:8` is the B70-only baseline;
* `allout:16:8:0` should assign the same devices and therefore must be token-identical to the baseline;
* `allout:16:8:8` makes the CPU tier live and enables loader verification.

The final run is accepted only when verification lines exist and contain no mismatch, CPU routes and poller dispatches are positive, poller errors and arena skipped routes are zero, resident experts exist, and generated output is nonempty/nonzero. The harness reports KV-cache snapshots but does not make a KV improvement an assertion. Nor does it demand token equality for CPU execution: coherent output is a weak end-to-end sanity check layered on top of direct layout and telemetry checks.

A source-description mismatch is worth preserving: the harness prose calls the CPU path a “bf16 CPU GEMM,” while the shared fixture and CPU API branch describe packed NVFP4 expert storage with dequantized reference weights and floating-point accumulation. The safe interpretation is that packed NVFP4 is the stored representation and the CPU computation expands/accumulates it; the comment must not be read as a bf16 weight-bank format guarantee.

## 5. Preemptive allocation assertions

[`preemptive_surgery_test.py`](../../../src/phase7/preemptive_surgery_test.py) is CPU-executable and intentionally targets mistakes that could silently compute the wrong expert:

* Across several `cuda_per_layer` values, CUDA, B70 and CPU owner sets are pairwise disjoint and cover all 256 experts exactly once.
* CUDA global ids are ascending, and the local compact id is the position in that list. This makes preemptive allocation agree with post-hoc sorted `index_select` layout.
* B70/CPU-owned experts map to `-1`; precisely the configured CUDA count remains.
* Layers outside the active offload range keep all experts and therefore remain on the stock path.
* Equal CUDA counts in subset and all-out policies yield the same CUDA set.
* The vLLM source still gates a special global-id scale-loader branch on the literal substring `input_scale`, while the qualified checkpoint method still creates `w13_input_global_scale`. Since the latter does not contain the former contiguously, compact indexing is currently safe. This is a version-sensitive source tripwire, not a permanent vLLM guarantee.
* A plain placement attribute may be assigned before `nn.Module.__init__`; an actual `Parameter` may not.
* The public CUDA-id accessor matches a direct owner-table scan.
* Subset placement is admitted without a host bank, all-out is admitted with one, and all-out is refused when its configured bank path is missing. Missing-bank refusal prevents plausible output with silently zero host contributions.

These assertions exercise placement and admission contracts without reimplementing either subsystem in this model.

## 6. Prefill numerical oracle and diagnostic branches

[`prefill_probe.py`](../../../src/phase7/prefill_probe.py) runs a long ordinary prompt and requests prompt plus generated logprobs. Prompt logprobs are the sensitive prefill observation: unchanged sampled tokens can hide a large degradation if the argmax does not flip.

The probe records engine options, optional model revision, native-library path and SHA-256, telemetry before/after inference, placement, effective max batch, prompt-logprob count/sum/mean, nonfinite positions, top-five first-token logprobs, generated tokens, and text. Its hard errors include empty output, missing/incomplete/nonfinite prompt logprobs, and missing/nonfinite generated logprobs. For hybrid+B70 runs it additionally requires:

* stable worker count;
* the expected number of available native devices;
* no provider last error;
* increased raw dispatch counters;
* unchanged provider generation across the request;
* in graph mode, matching available poller devices, zero poller errors, and increased poller dispatches.

Optional diagnostics monkey-patch model methods only for the probe: logit diagnostics report shape/nonfinite rows without changing values; layer diagnostics require eager mode and stop at the first nonfinite module output; aggregation capture activates an existing oracle on the actual request rather than engine warmup.

[`prefill_probe.sh`](../../../src/phase7/prefill_probe.sh) fixes the common hybrid settings, then launches all-CUDA, Phase-6 B70, forced B70-weight streaming, and all-out configurations. Forced streaming sets threshold 1 because this prompt is below the production threshold; otherwise that case would silently measure dispatch again. The final Python snippet prints mean-logprob deltas and token identity. Despite shell comments saying Phase-6 and all-out “must” land at the all-CUDA logprob, the comparison code imposes no numeric tolerance and exits successfully after printing. The executable model makes `oracleClose` an explicit external assessment rather than falsely attributing an enforced threshold to the script.

Cached probe JSON is reused by default. Consequently, provenance fields—configuration, revision, library digest, and timestamps outside this script if supplied—matter when comparing cases; file presence alone is not fresh evidence.

## 7. Performance diagnostics and CPU implementation comparisons

### 7.1 Prefill chunk sweep

[`prefill_chunk_bench.py`](../../../src/phase7/prefill_chunk_bench.py) tests whether repeated whole-working-set reads per B70 chunk explain slow prefill. For each context it warms on the same long prompt, then records mean/min TTFT and derived prompt tokens per second. It also derives dispatches per layer as ceiling(prompt tokens / chunk size) and retains worker telemetry. Configuration helpers set all hybrid controls together, avoiding the earlier false measurement where a hybrid label still ran all-CUDA.

[`prefill_chunk_bench.sh`](../../../src/phase7/prefill_chunk_bench.sh) launches a fresh process per chunk size because pinned staging capacity is fixed when a layer is constructed. The default sweep is chunks 128/256/512/1024/2048, contexts 1536/4096, three trials, placement `subset:16:8`. A falling curve supports chunk/dispatch-overhead hypotheses; a flat curve supports a kernel-throughput hypothesis. None is a correctness invariant.

### 7.2 CPU compute versus CUDA streaming

[`stream_crossover_bench.py`](../../../src/phase7/stream_crossover_bench.py) is the direct CPU implementation comparison. It constructs eight packed cold experts matching `allout:16:8:8`, attempts to pin the arena, and compares `CpuExpertHost.moe_forward` against `ExpertStreamer.forward` over batch sizes 1 through 2048. Routes are sampled over all 256 experts and filtered to cold ids; if a small batch selects none, one cold route is forced so the measurement represents the decode case of interest. Timings synchronize CUDA around each sample and use the median after warmup.

The first measured batch where streaming wins is printed as the suggested `SHOOTING_BRAKE_CPU_STREAM_T`. If none wins, the program recommends leaving streaming off or placing the threshold above the range. This result is deployment-specific: topology, pinning success, compiler/CPU, routing distribution, model shape, and GPU all move the crossover. CUDA absence is an explicit unavailable outcome, not evidence that CPU wins.

Additional concrete CPU comparisons available to the parent are:

* use `nvfp4_testutil.make_expert` and `ffn` to compare native CPU results with vLLM dequantization over identical packed bytes;
* compare subset and `allout:...:0` token ids to isolate policy from CPU arithmetic;
* require positive CPU route/dispatch counts, zero errors/skips, and loader verification before interpreting coherent text;
* compare streamed and host execution on identical packed arena blocks, routes, and weights, separating numerical parity from latency crossover.

## 8. Result-wire and grouped-provider assertions

[`tests/test_b70_output_wire.py`](../../../tests/test_b70_output_wire.py) is conditional on `SB_TEST_B70_LIBRARY`; absence skips the module, but an explicitly selected missing/broken library or provider load fails. Its fixture loads the actual binding under a test-local package name, writes a deterministic one-layer bank, disables grouped execution for the scalar case, and restores every environment variable after shutdown.

`test_b70_output_wire_matches_actual_result` parameterizes the wire flag as unset, `0`, `1`, and `01`. Only exact `1` requests FP16; all other values require FP32. It issues/takes one route, asserts `(1, 256)` shape, exact dtype, finite values, and closeness to the analytically expected `sigmoid(1)` result. The model retains this exact-string policy as the abstract predicate `flagIsOne`.

`test_grouped_swiglu_preserves_range_and_reused_scratch` runs both pipeline settings and FP32/FP16 wire modes with native grouped atomic scatter. Inputs bracket the FP16 intermediate-overflow boundary (`255`, `256`, and `±512`) while down-scaling final results into range. It asserts finite output and closeness to a float64 oracle, then dispatches again after replacing hidden values to show per-route `slot_w` scales do not leak through reused scratch. Finally, all expert ids become `-1` and output must be exactly zero. These are assertions about the provider/compute contract; the grouped algorithm itself is specified in the compute branch.

## 9. Phase-10 saved evidence and contradiction handling

Phase 10 contains saved JSON only—no launcher, runtime component, or policy selector. The captured [`comparison.json`](../../../src/phase10/results/comparison.json) reports:

* exact token agreement in 3 of 8 prompts, with mean common-prefix length 18.25;
* 211,696 reported KV-cache tokens for all-CUDA versus 888,704 for hybrid;
* single-stream throughput about 247.91 versus 186.27 token/s and median ITL about 3.99 versus 5.34 ms;
* at recorded concurrency 1/4/16/32/64, hybrid throughput is lower and hybrid median ITL higher.

Those are facts about the captured [`all-cuda.json`](../../../src/phase10/results/all-cuda.json) and [`hybrid.json`](../../../src/phase10/results/hybrid.json), not guarantees. In particular, 3/8 exact agreement contradicts any blanket claim that hybrid generation is token-identical. It does not contradict the focused hybrid gate, whose documented contract allows a narrow semantic fallback on one stable prompt. The artifacts also cannot establish causality, statistical confidence beyond their recorded trial counts, current-code performance, or hardware portability.

## 10. Executable scenarios and properties

The Reyu model makes the assertion boundary visible:

| Scenario | Reachable observation |
|---|---|
| `buildCompositionTest` | Both build-link facts are present; this says nothing about runtime results. |
| `hybridExactParityTest` | Baseline exists, marker/path is exercised, tokens agree. |
| `hybridSemanticParityTest` | Marker exists and the narrow semantic oracle passes despite token divergence. |
| `hybridMarkerRequiredTest` | Matching output without path evidence fails. |
| `allOutCpuHealthFailureTest` | Correct plumbing and coherent text still fail when CPU poller health fails. |
| `preemptiveSuccessTest` | Ownership, ordering, mask, loader, and bank checks all pass. |
| `missingBankRefusalTest` | The host-tier candidate is refused before serving. |
| `prefillOracleTest` | Complete finite logprobs, healthy stable provider, advancing dispatch, and external baseline comparison all hold. |
| `chunkFreshProcessEdgeTest` | Reusing construction-sized buffers across chunk settings invalidates the sweep. |
| `unavailableCrossoverTest` | Missing CUDA records unavailability, not a threshold. |
| `wireFp16Test` / `wireNonOneMeansFp32Test` | Exact `1` selects FP16; every abstract non-`1` value selects FP32. |
| `groupedScratchReuseFailureTest` | Numerical range alone cannot hide cross-dispatch scratch contamination. |

The invariant says a passed assessment must be complete and assertion-satisfying; failed assessment must expose an error; correct refusal is intentionally distinct from pass and failure. Baseline-dependent checks cannot pass without a captured baseline. Witness values expose exact/semantic hybrid acceptance, missing-bank refusal, grouped scratch failure, and unavailable crossover.

## 11. Abstraction limits and external dependencies

A child-process launch, model generation, provider dispatch, telemetry snapshot, and array comparison are atomic observations in the Reyu model. Tensor values are reduced to finite/close/zero predicates; timing distributions are reduced to point counts and measurement validity. The model does not prove floating-point tolerances, hardware memory ordering, vLLM plugin loading, subprocess isolation, or performance.

The workflows depend on vLLM, PyTorch, NumPy, oneAPI/SYCL, CUDA, the qualified model/checkpoint, expert banks, Phase-1 and Phase-7 native libraries, QuixiCore-XPU, oneDNN, and particular hardware. Version-sensitive source inspection in the preemptive test and environment-inherited launcher state are explicit assumptions. Tests are assertions over those dependencies. They neither become serving code nor establish behavior for untested shapes, models, devices, variants, or revisions.
