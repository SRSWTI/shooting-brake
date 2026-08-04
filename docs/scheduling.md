# Shooting Brake Scheduling Policy

## Purpose and status

This document defines decode, continuous-batch, prefill, provider, and recovery scheduling for the one-RTX-5090 plus one-B70 production design. It follows [`../plan.md`](../plan.md), uses ownership from [`placement.md`](placement.md), and supplies the measurement classes in [`benchmarking.md`](benchmarking.md).

This is a normative target, not a claim that the Qwen-scoped `HybridMoERunner`/`HybridRoutedExperts` integration or batched llm-scaler provider is complete. The proven Colibri worker is a single-token reference, not the production scheduler.

## Core invariants

1. Upstream vLLM retains serving, request admission, continuous batching, attention/KV/recurrent scheduling, and canonical CUDA router/top-k.
2. The immutable placement map partitions selected routes into CUDA-owned and B70-owned sets without changing IDs, routing weights, or reduction semantics.
3. There is at most one aggregated B70 operation per active MoE layer and upstream-vLLM scheduler step, never one submission per request.
4. Only rows with at least one B70-owned route are staged. A zero-remote-route layer performs no B70 submission.
5. Publish the B70 request as soon as routing and asynchronous D2H staging permit; run local CUDA routed experts and the CUDA shared expert concurrently.
6. The B70 returns one already-weighted, route-reduced `[M_remote, hidden]` wire partial for the staged token rows. CUDA scatters it into the full `[M, hidden]` batch and performs the join before final tensor/expert-parallel reduction.
7. CPU matrix work is absent from the normal schedule. CPU or available CUDA executes only exact failed-route recovery.
8. The hot path uses fixed pinned-ring and grow-only provider buffers. It performs no per-forward allocation, weight movement, quantization, `.item()`, Python polling loop on the CUDA worker, or device-wide synchronization.
9. Request cancellation, ring backpressure, provider timeout/restart, and stale generations preserve exactly-once route accounting.
10. Kernel selection is a provider capability/policy decision qualified by shape measurements; model code does not encode undocumented model-name thresholds.

## Scheduler-step transaction

For `M` token rows scheduled at an active routed-MoE layer:

```text
upstream vLLM CUDA router/top-k
    -> HybridRoutedExperts partitions canonical routes
    -> stage only remote-bearing rows in a fixed ring slot
    -> publish one B70 layer request
       || run stock-compatible local CUDA routed experts
       || run the CUDA shared expert where its dependency permits
    -> provider executes all B70-owned token/route pairs
    -> provider returns one weighted [M_remote, hidden] wire partial plus token_row_map
    -> asynchronous H2D to a preallocated CUDA buffer
    -> CUDA local + B70 remote routed addition
    -> stock shared/residual continuation
```

The request retains original token/request mapping, layer, sequence, placement generation, weight/provider generation, canonical expert IDs and weights, route mask, and cancellation state. Aggregation changes only execution shape; it never merges request lifetimes or router semantics.

The local branch represents B70-owned route positions as invalid/skipped routes when the selected CUDA backend supports that contract. Otherwise `HybridRoutedExperts` compacts local routes and unpermutes/reduces the result. Either implementation is qualified against the same all-CUDA output.

## Workload and kernel classes

The initial logical provider policy is:

| Scheduled work | Aggregation | Initial B70 kernel family |
|---|---|---|
| `M=1` decode | dispatch the scheduler step without an added provider microbatch wait | llm-scaler tiny INT4 preselected-route path |
| `1<M<=32` continuous-batch decode | combine all remote-bearing rows from the scheduler step | qualified tiny or small-batch INT4 path |
| larger decode batch | one layer/step operation | grouped-route path |
| prefill | group the scheduler step's remote token-route pairs while preserving token mapping | `gather-v2` → grouped `up-v2` → activation → grouped `down-v2` → weighted accumulation |
| zero B70-owned rows | no ring publication | no provider kernel |

Thresholds are benchmark-selected for each qualified shape and provider bundle. The provider must accept precomputed IDs/weights and must not call a fused entry point that recomputes router/top-k or shared-expert work.

### No cross-step batching queue

Upstream vLLM already forms the continuous batch. Shooting Brake must not hold a ready scheduler step to accumulate unrelated future requests. One step's rows are the aggregation boundary; ring backpressure may bound admission but does not authorize a hidden microbatch delay. This keeps per-request deadlines, cancellation, and vLLM fairness under the upstream scheduler.

For 32 concurrent one-token requests, the required shape is one `M=32` B70 layer operation, not 32 `M=1` submissions.

## Decode

Decode has the tightest sequential dependency. For each active MoE layer:

1. CUDA computes canonical IDs and weights.
2. The adapter resolves the immutable placement generation and builds local and remote masks.
3. If remote rows exist, it reserves one bounded ring slot, begins nonblocking D2H staging, and publishes after copy completion.
4. CUDA local routed work and the shared expert proceed without waiting for B70 where dependencies allow.
5. The provider copies into preallocated XPU tensors, remaps only B70-owned routes, and selects the qualified tiny/small/grouped decode kernel.
6. It writes one weighted partial row per transported token and publishes completion after the XPU-to-host result is visible.
7. CUDA rejects stale or invalid completion metadata, asynchronously copies the partial, scatters rows if needed, and adds it exactly once.
8. On exact route failure, recovery replaces only those failed contributions or fails the request explicitly.

The current Colibri observation of roughly 56–100 µs one-token issue/take per active MoE layer is useful for transport and latency decomposition. It does not determine production decode eligibility or performance: upstream-vLLM scheduling, process-ring overhead, llm-scaler dispatch, local overlap, and graph boundaries are different.

## Continuous batching

The upstream scheduler supplies one decode row per ready sequence. At every active layer, Shooting Brake aggregates all rows with B70-owned routes into the single request for that step. It preserves:

- original request and token row;
- canonical route order, IDs, and weights;
- local/remote route mask;
- placement and provider generations; and
- per-token cancellation/recovery status.

One provider job may contain many experts and requests, but every request becomes join-ready only when its required local and remote contributions, or exact replacements, are complete. Cancellation after publication marks the affected rows and prevents a stale result from entering a reused request; it does not allow early slot reuse while either runtime still accesses the buffers.

Batch-class telemetry is grouped as `M=1`, `2..32`, and larger decode, with exact `M`, active remote rows, remote routes, and selected kernel reported. Aggregate throughput never substitutes for per-request TTFT/ITL percentiles.

## Prefill and mixed steps

Prefill retains attention, KV/recurrent state, router/top-k, shared expert, and residual work on CUDA. For the routed branch, preserve the full prompt batch already selected by vLLM:

1. collect remote-bearing rows and their canonical route pairs;
2. gather/group by the provider's compact B70 expert slots;
3. execute llm-scaler's prefill gather, grouped up, activation, grouped down, and weighted-accumulation phases in stable buffers;
4. return one weighted partial per original token row; and
5. scatter/add on CUDA before the residual continuation.

Prefill is not implemented as repeated `M=1` decode calls. The provider's negotiated token/route capacities constrain upstream scheduler admission, so every admitted layer step fits one ring slot and produces at most one request/completion transaction. A larger prompt is chunked into multiple upstream scheduler steps before layer execution; the adapter never splits one layer step across multiple ring transactions. A mixed prefill/decode step retains row/request mapping and deadlines. If the provider internally partitions one admitted request among kernel families, that remains one operation and one completion and is reported as such.

Decode latency takes priority through upstream vLLM scheduling and bounded provider/ring capacity. Background profiling, artifact preload, provider warmup, and duplicate correctness sampling are never participants in a foreground join.

## Ring capacity, backpressure, and fairness

Ring slot and token/route capacities are negotiated at startup. When no slot is free:

- the CUDA adapter applies bounded backpressure at the provider boundary;
- it does not allocate an overflow buffer or submit through an unversioned side channel;
- cancellation and request deadlines remain visible to upstream vLLM; and
- timeout leads to exact recovery or explicit request failure, never dropped routes.

Queue time, slot occupancy, remote rows, backpressure duration, and exposed join wait are mandatory telemetry. A full-core busy poll is not assumed; completion signaling/polling policy must have measured CPU cost.

## CUDA graph policy

Begin hybrid integration in eager mode. A host-mediated provider operation cannot live inside a full CUDA graph replay. The optimization sequence is:

1. eager correctness;
2. upstream vLLM PIECEWISE CUDA graphs with a break around the provider operation; and
3. dedicated copy streams, events, stable buffers, and segmented capture that removes exposed waits without hiding the external dependency.

Graph optimization must preserve output agreement and exactly-once recovery. The steady-state path must contain no device-wide synchronization. ExLlama-style stream-memory mechanics are an optimization reference only after the fixed process ring is correct.

## Failure, cancellation, and restart

A provider timeout, device loss, kernel error, invalid generation, or malformed completion produces per-token/per-route failure status. The adapter then recomputes the exact failed routes on an available CUDA/CPU correctness path or fails explicitly. It never joins both recovered and late remote contributions.

Provider restart:

1. stops new publication;
2. invalidates the provider generation and all outstanding completions;
3. drains/cancels bounded in-flight slots without early reuse;
4. reloads and verifies the immutable B70 expert bank;
5. renegotiates capabilities, buffers, and generations; and
6. resumes only after ownership is complete.

CPU recovery is failure handling, not a scheduled concurrent branch. Healthy production traces have zero CPU expert matrix operations.

## Required scheduler telemetry

Record per layer/step and correlate to the run manifest:

- upstream service class, request IDs, token rows, prefill/decode kind, and exact `M`;
- canonical expert IDs/weights, owners, remote-bearing rows, and route counts;
- placement, weight/provider, request, and ring-slot generations;
- ring queue/occupancy/backpressure, copies, provider kernel family, and completion status;
- exactly zero or one B70 request per active layer/step;
- CUDA local routed/shared times, B70 queue/copy/kernel time, branch overlap, exposed join wait, and CUDA add time;
- cancellation, timeout, recovery, provider restart, stale/discarded completion, and explicit failure;
- CPU recovery count/time and NVMe reads, both zero on the healthy warmed path;
- TTFT, ITL, request/output throughput, and prefill throughput; and
- per-layer/final-logit/generated-token correctness artifact identity.

## Scheduling acceptance

The policy is qualified only when:

- all scheduler-step remote rows are aggregated into one B70 layer request;
- changing continuous-batch sizes, mixed prefill/decode, completion, cancellation, and zero-remote-route layers preserve output agreement;
- the dispatch table uses measured provider shape thresholds and no full-fused B70 router/shared path;
- local CUDA and B70 branches overlap where dependencies permit;
- fixed buffers show no hot-path allocation, weight movement, or device-wide synchronization;
- bounded backpressure and restart never accept stale output or lose/double a route;
- injected failure recomputes exact routes or fails explicitly;
- normal-path CPU expert matrix work and warmed foreground NVMe reads are zero; and
- throughput, TTFT, ITL, and capacity are reported against the identical stock all-CUDA upstream-vLLM workload, with Colibri results labeled reference evidence only.
