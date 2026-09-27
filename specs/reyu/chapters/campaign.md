# Profiling campaigns and interactive scheduling

This branch is an experimental measurement and qualification layer around the serving system. It does not replace the production request path. Read it breadth first: the campaign client defines workloads and owns client-side evidence; a worker extension opens bounded native/profile windows and persists worker evidence; an offline analyzer correlates those windows with Nsight activity; a separate opt-in scheduler experiment changes one per-step budget and is qualified by a real-model harness. The executable abstractions are [campaign.ryu](../models/campaign.ryu) and [scheduler.ryu](../models/scheduler.ryu).

## 1. Responsibility, actors, and interfaces

```text
fixed corpus + suite configuration
              |
              v
 profile_workloads client -------- Prometheus / PCIe / nvidia-smi samplers
       |      |       |
       |      |       +-- client request records and summary
       |      +---------- continuous-usage SSE request measurement
       +----------------- loopback collective_rpc control
                              |
                              v
                    CampaignWorkerExtension
                    | snapshot counters
                    | optional vLLM profiler
                    | monotonic/NVTX clock samples
                    | bounded native trace extraction
                    +-- exclusive worker JSON
                              |
                  Nsight SQLite + worker JSON
                              |
                              v
                 analyze_campaign_profile
                 union/overlap/uncovered attribution

 InteractiveScheduler (explicit scheduler_cls only)
       |
       +-- temporary decode-sensitive token cap
       +-- inherited vLLM async scheduler and allocator
       +-- finally restores configured baseline
       |
 qualify_interactive_scheduler: two mixed-length real requests,
 finite probability data, trace coverage, and two-device presence
```

The campaign controller requires one CUDA worker, a loopback development endpoint, a port other than production port 8017, a fresh output directory, and a prefix-cache setting that agrees with the selected suite. The worker extension requires live B70 pollers and exposes control over `/collective_rpc`; its module documentation therefore requires a loopback-only development server. The profiler is optional, while the campaign control window and native trace accounting are required. This separation is explicit in `campaign.ryu`: `controlActive` does not imply `profilerActive`.

The scheduler is also separate from the profiler and campaign controller. It is not registered in production. It subclasses vLLM's `AsyncScheduler`, retaining vLLM allocation and ordering, and temporarily modifies only `max_num_scheduled_tokens` for a scheduling call.

## 2. Source-backed state and operations

| Model state / operation | Source state or event | Observable contract |
|---|---|---|
| `configure` / `rejectConfiguration` | [`profile_workloads.run`](../../../experiments/profile_workloads.py) | Only loopback, never port 8017; output directory must not exist; corpus and source identities enter the manifest; observed prefix-cache policy must match `cache` versus non-cache suite. |
| `startWindow` | `profile_workloads.case` calling [`CampaignWorkerExtension.campaign_start`](../../../experiments/campaign_worker.py) | Prompt construction/tokenization occurs before the window. Worker synchronizes CUDA, snapshots telemetry, resets peak memory, optionally starts the profiler, takes five clock samples, then records the monotonic start. An already active window, unsafe label, invalid tracing value, existing destination, missing poller, or profiler start failure rejects start. |
| `observeLeaderToken`, `injectMixedPrefill` | `profile_workloads.case(..., mixed=True)` | The long prefill is injected only after the leader produces its first token and only if the leader remains live. Leader completion before either condition is an error. |
| `collectRequests` | `one_request` dependency and `case` | Concrete request count, delivered-token count, and largest continuous-usage delta derive result-count validity, fixed-length validity, and bundled-ITL missingness. A source request is successful only after HTTP success, terminal `DONE`, authoritative continuous usage, and at least one output-token delivery. Token totals come from usage deltas, not SSE frames or text pieces. |
| `collectFailedRequest` | `one_request` exception result; [`test_campaign_logging.py`](../../../tests/test_campaign_logging.py) | Failure remains `ok=False` with an error and traceback log. It is not inserted as a successful latency observation. |
| `stopWindow` | [`CampaignWorkerExtension.campaign_stop`](../../../experiments/campaign_worker.py) | Called from the client's `finally`; synchronizes CUDA, takes stop clock samples, stops and clears the optional profiler, snapshots counters, filters bounded trace entries, checks dispatch/row/native-error/clock coverage, exclusively writes worker JSON, then clears active state. |
| `acceptEvidence`, `rejectEvidence` | `profile_workloads.case` postconditions | Any case error, trace-coverage failure, result-count mismatch, unsuccessful request, or short output fails closed. Client JSON is still written before these checks. |
| interval functions | [`merge_intervals`, `duration`, `intersection_duration`](../../../experiments/analyze_campaign_profile.py) | Half-open activity intervals are unioned before duration and intersection; overlapping engines are not summed. Negative intervals are rejected. |
| `admit`, rejection actions | [`InteractiveScheduler.__init__`](../../../experiments/interactive_scheduler.py) | Cap must accommodate every running request; speculative decoding is excluded. Default cap is environment `SB_INTERACTIVE_TOKEN_CAP` or 512. |
| `beginSchedule` | `interactive_budget`, `InteractiveScheduler.schedule` | Cap applies iff some running request has emitted output and computed its whole prompt. Empty, cold prefill, prompt-complete-without-output, and preempted prompt recomputation retain baseline. `min(baseline, cap)` never raises a smaller baseline. |
| `finishSchedule`, `failSchedule` | `schedule` `finally` block | Baseline is restored after inherited scheduling on both normal and exceptional exits. |
| `qualify` | [`qualify_interactive_scheduler.main`](../../../experiments/qualify_interactive_scheduler.py) | Qualification requires two completed outputs, exactly 32 generated tokens per output, complete finite prompt/generated logprobs, passing trace coverage, and exactly native devices `0` and `1`. |

## 3. Campaign workload lifecycle

### 3.1 Provenance and background collectors

[`experiments/profile_workloads.py`](../../../experiments/profile_workloads.py) hashes the corpus, its own source, and the source file containing `one_request`; it records all arguments and a workload contract in `manifest.json`. It fixes PCIe watch addresses, starts a 250 ms PCIe sampler, a 250 ms Prometheus scraper thread, and a 500 ms `nvidia-smi` subprocess. These collectors span all cases and are distinct from each worker's bounded control window. Final cleanup stops the scraper and PCIe sampler, terminates `nvidia-smi`, closes its log, and writes PCIe and case summaries even if a workload raises.

Prompts are deterministic per `(context, slot, family)` through a nonce supplied to `build_prompt`, and the cache avoids rebuilding a prompt within the run. Prompt text is excluded from client JSON, while prompt hash and client-tokenizer count remain as identity evidence. Manifest/source hashes establish provenance; they do not prove server binary identity.

### 3.2 Suites and traffic shape

After validating `/metrics`, every run performs an unprofiled 1024-token-context warmup with 32 output tokens. The executable suite branches are:

- **Core:** cold single-request cases for each configured context; concurrent decode at concurrency 2 and 4 with distinct prompt families; a quiet 256-token-output leader; then a mixed case where the same leader begins decoding before a 32K-context prefill is injected.
- **Capacity:** four concurrent 32K-context requests producing 32 tokens and two concurrent 96K-context requests producing 16 tokens.
- **Cache:** for 8K and 32K contexts, prime a prefix and then continue it with a fixed suffix. The observed metrics label must say prefix caching is enabled.
- **All:** core plus capacity. It is not cache mode; its observed prefix-cache label must be false.

Tracing defaults from the command-line flag but may be disabled per case; warmup always disables it. Control and native evidence are still collected when tracing is disabled.

### 3.3 Measurement semantics: missing is not zero

The request dependency `benchmarks.slo_split_matrix.one_request` initializes token count to zero while a request is pending or failed. That storage default is not a valid zero-latency sample. A request becomes successful only after continuous usage has produced authoritative positive delivery evidence and the stream ends with `DONE`. `ttft_s`, end-to-end time, and per-output-token time are then computed. If any delivery advances usage by more than one token, individual-token ITL percentiles are `None`: bundled delivery makes ITL **unavailable**, not zero. Decode-only TPOT is likewise absent for a one-token output.

`campaign.ryu` therefore uses `Missing`, `ObservedZero`, and `ObservedPositive` instead of encoding all three as integer zero. A failed HTTP request has missing client latency. In contrast, a valid bounded window may observe zero native-service duration; that is `ObservedZero`. Summaries include only `ok` request records in latency aggregates and report a no-success shape containing counts and errors if none succeeded.

The source tests make the intended distinction concrete. [`test_campaign_measurements.py`](../../../tests/test_campaign_measurements.py) checks usage totals, event deltas, first-delivery bounds, TTFT, TPOT, bundled-ITL absence, HTTP-404 exclusion, per-layer native row totals, device errors, entry/dispatch equality, and clock bounds. [`test_campaign_logging.py`](../../../tests/test_campaign_logging.py) checks that a connection failure stays an error record and that picologging records traceback context. These are harness contracts; this chapter does not claim they were run.

### 3.4 Worker evidence and failure behavior

[`experiments/campaign_worker.py`](../../../experiments/campaign_worker.py) installs no new hot-path hook. `campaign_snapshot` delegates to plugin telemetry. `_campaign_clock_samples` uses the first live poller's native library and takes five paired samples inside named NVTX ranges. For each sample, native monotonic time must lie between host calls. The worker start path is exclusive: it refuses a second active window and refuses to overwrite a prior worker JSON. If clock sampling fails after profiler start, start explicitly stops the profiler and clears its object before rethrowing.

Stop filters each poller's trace snapshot to entries wholly bounded by the campaign's monotonic window. Per device it compares trace entry count to dispatch delta, sum of trace `M` to row delta, and native error delta to zero. Trace capacity is reported as $2^{16}$; the code does not treat that declaration as evidence that no trace entry was overwritten. Clock-bound failure also fails coverage. Worker JSON is written with exclusive-create mode before `_campaign_active` is cleared. If exclusive writing fails, active state remains set; that source behavior is preserved rather than idealized as successful cleanup.

The timing contract is deliberately narrow: `t0/t1` are host service bounds after signal observation; kernel/total are raw device spans only with provider profiling; cross-copy-engine total spans require independent clock validation. Completion and coverage therefore do not prove a particular kernel is on the critical path.

### 3.5 Offline attribution

[`experiments/analyze_campaign_profile.py`](../../../experiments/analyze_campaign_profile.py) opens Nsight SQLite read-only. Matching NVTX clock markers form offset intervals; all intervals must intersect. Their common midpoint is the selected monotonic-to-Nsight offset, and the intersection width remains uncertainty. `execute_*_context_*_generation_*` annotations are parsed into prefill, decode, or mixed phases; unrecognized annotations fail instead of being silently discarded.

CUDA kernels and copies are attached to an execute step by their runtime API start. Activity outside a CPU execute range is reported as unattributed. Kernel names are classified into grouped expert GEMM, dense GEMM, attention, activation quantization, router top-k, GEMV, elementwise, normalization, or other. Native service intervals are shifted by the correlated offset and attached only when bounded by a GPU step plus clock uncertainty; otherwise they increment `unmatched_native_entries`.

For every step, CUDA and native durations are interval unions. `native_service_without_cuda_activity` subtracts the CUDA/native intersection; `uncovered_span` subtracts the union from the GPU span and clamps at zero. The analyzer does not invent a cause for uncovered time, and native-without-CUDA overlap does not establish critical-path causality. The pure functions in `campaign.ryu` expose the two-interval form for direct CPU differential checks. [`test_campaign_profile_analysis.py`](../../../tests/test_campaign_profile_analysis.py) supplies overlapping, touching, empty, symmetric-intersection, and negative-span examples.

## 4. Interactive token-budget policy

[`experiments/interactive_scheduler.py`](../../../experiments/interactive_scheduler.py) is explicitly inactive unless selected as `scheduler_cls`. `interactive_budget` scans current running requests. The exact decode predicate is:

```text
num_output_tokens > 0
and num_computed_tokens >= num_prompt_tokens
```

If any request satisfies it, the step budget is `min(baseline, cap)`; otherwise it is `baseline`. This includes mixed traffic: one genuinely decoding request caps the whole scheduling step even while another request is prefilling. A request whose prompt is complete but has not emitted output is not decoding. A preempted decode request with output history but an incompletely recomputed prompt is also not considered decoding until recomputation reaches the prompt length.

`InteractiveScheduler.schedule` stores the baseline, computes the temporary budget, delegates to `AsyncScheduler.schedule(throttle_prefills=...)`, and restores the baseline in `finally`. The policy does not define which request consumes the capped tokens, change allocation, reserve a per-request token, or add fairness. Those are inherited vLLM behavior and outside this model.

The constructor also rejects speculative decoding and a cap smaller than `max_num_running_reqs`. The cap is parsed with `int`, so a malformed environment value fails construction through Python conversion. [`tests/test_interactive_scheduler.py`](../../../tests/test_interactive_scheduler.py) covers empty/cold/prompt-complete states, true decode, a baseline already below cap, and preempted recomputation.

## 5. Qualification versus performance

[`experiments/qualify_interactive_scheduler.py`](../../../experiments/qualify_interactive_scheduler.py) creates a real in-process LLM with max model length 16384, batched-token baseline 2048, two sequences, GPU utilization 0.85, CUTLASS MoE, prefix caching disabled, chosen scheduler class, campaign worker extension, and seed zero. It runs one normal prompt and one prompt repeated 80 times, both greedy for exactly 32 tokens with EOS ignored and prompt/generated logprobs requested.

The harness records telemetry before and after, surrounds generation with an unprofiled campaign window, and writes `result.json` only into a newly created output directory. Its verdict checks output count and length, logprob completeness/finiteness, worker trace coverage, and exact two-device presence. It records scheduler class, cap string, model, revision, native library, outputs, telemetry, and trace. Its own module documentation says this is correctness qualification, **not serving performance measurement**. Successful qualification cannot establish latency improvement, fairness, or production suitability.

## 6. Executable scenarios and properties

`campaign.ryu` names reachable witnesses for a fully accepted campaign, missing failed-request latency, valid zero native service, an unprofiled but active control window, and bundled-delivery ITL missingness. Scenarios cover cold traced success, ordered mixed injection, bundled mixed delivery, unprofiled zero activity, failed HTTP, incomplete trace rejection, invalid configuration, and interval accounting.

`scheduler.ryu` names witnesses for capped decode, uncapped cold prefill, a baseline smaller than cap, restoration after an inherited scheduling failure, and successful qualification. Scenarios cover empty, cold, prompt-complete-without-output, mixed decode, recomputation, exception restoration, cap/speculation rejection, and qualification pass/failure. `inv` asserts budget restoration outside scheduling and that a temporary cap never raises the baseline.

These finite models collapse request payloads, tensor values, device execution, filesystem calls, clocks, SQLite queries, and inherited vLLM scheduling into source-labeled atomic observations. They are behavioral explanations, not proofs of the Python implementation or hardware.

## 7. Contradictions, limits, and dependencies

No source contradiction was found between the policy implementation and its focused tests. Two boundaries are easy to misstate:

1. A tracing-disabled window is still a valid campaign control/native-trace window; profiler state is not campaign state.
2. The interactive scheduler is an opt-in experiment and the qualifier is a correctness gate. Neither is a production default or a performance claim.

The campaign depends on vLLM collective RPC/profiler/NVTX annotations, Shooting Brake telemetry and B70 pollers, continuous-usage SSE behavior from the benchmark client, aiohttp, tokenizer/model files, Prometheus, PCIe sysfs sampling, `nvidia-smi`, Nsight SQLite schema, and physical provider traces. The scheduler depends on the external vLLM `AsyncScheduler` and request counters. Changes in those external contracts can invalidate the abstraction even if the Reyu scenarios remain reachable.

## 8. CPU implementation comparisons

The following comparisons require no accelerator workload and are suitable for Main's consolidated verification:

- Differentially enumerate nonnegative integer intervals and compare `merge_intervals`, `duration`, and `intersection_duration` against an independent point-set or sweep-line oracle. Include overlap, containment, adjacency, empty input, and negative-span rejection.
- Enumerate request-state tuples `(num_output_tokens, num_computed_tokens, num_prompt_tokens)` and baselines/caps, compare `interactive_budget` with the rule “cap iff output $>0$ and computed $\ge$ prompt,” and assert the result never exceeds baseline.
- Source-inspect `InteractiveScheduler.schedule` to confirm restoration remains in `finally` around the sole inherited `schedule` call.
- Feed synthetic successful, bundled, failed, and no-success request dictionaries into `summarize` and confirm missing ITL values are excluded while zero/failure is not relabeled as observed latency.
- Use a temporary SQLite database and worker JSON with synthetic clock markers/activity to compare analyzer attribution, unmatched-native accounting, and exclusive output refusal without invoking a server or GPU.
