# Shooting Brake Scheduling Policy

## Purpose and status

This document defines the scheduling policy for decode, continuous batching, prefill, and background work in Shooting Brake. It elaborates the ownership and per-layer execution model in [`architecture.md`](architecture.md). Transport timing and B70 eligibility are measured under [`benchmarking.md`](benchmarking.md); expert ownership is supplied by [`placement.md`](placement.md).

This is a **normative design policy**, not a claim that the scheduler or its performance has been implemented or validated. The policy preserves the architecture's correctness rule: the state owner computes canonical routes, required local/remote/CPU partials execute concurrently where possible, and the state owner performs the required ordered join.

## Core invariants

1. The RTX 5090 remains the state owner. It owns attention and KV state, router execution, dense/shared paths, local experts, ordered partial reduction, and sampling.
2. Scheduling changes where and when an already selected route executes; it does not change canonical expert IDs, route weights, precision, or reduction semantics.
3. Publish eligible B70 work immediately after routing. Do not serialize remote dispatch behind independent local work.
4. Execute RTX 5090-local experts, B70-resident experts, and required CPU cold work concurrently.
5. Join only the required participants for the current sparse layer, and join each required partial exactly once in the defined order.
6. Prediction and promotion are optional. Foreground execution never waits for predicted routes, background prepack, NVMe staging, or expert promotion.
7. No ordinary foreground token waits for an NVMe expert read. Missing or unusable remote work follows the exact CPU fallback policy.
8. Batching is bounded by both a maximum batching window and request deadlines. Throughput aggregation must not create an unbounded wait.
9. Foreground requests nearing deadline dispatch without additional microbatch delay.
10. Foreground requests preempt background queues at dispatch boundaries. Work already inside a non-preemptible device stage is not assumed to be interruptible.
11. B70 assignment is determined by measured complete remote-path tail latency for the relevant batch class, not by device-local GEMM speed.

## Work classes

| Class | Latency policy | Aggregation policy | B70 policy |
|---|---|---|---|
| Latency-critical batch-one decode | Tightest deadline; publish required work immediately; do not delay solely to form a microbatch | No optional batching delay when the foreground request is ready or nearing deadline | Eligible only when measured batch-one remote p99 is lower than measured batch-one CPU-fallback p99 |
| Foreground interactive continuous decode | Strict per-request deadline | Small, deadline-bounded aggregation; dispatch early as the oldest request nears its deadline | Use only for measured winning token-batch classes |
| Normal service continuous decode | Service deadline | Small aggregation window, still bounded by the oldest request and maximum window | Use only for measured winning token-batch classes |
| Prefill | Prompt-processing deadline; less latency-sensitive than decode but still bounded | Aggressively group prompt token-route pairs and use larger transfers/windows | Eligible in measured winning prefill/batch classes; may use multiple B70s concurrently |
| Background agent or analysis | Lower priority than foreground | Larger bounded window | Eligible in measured winning batch classes |
| Maintenance/background calibration | Lowest foreground priority; runs only without delaying required foreground dispatch | Batch route calibration, prepack validation, or duplicate correctness sampling | Eligible in measured winning batch classes; preempted by foreground at dispatch boundaries |

The table does not establish one fixed deadline value. Each request carries its service-class deadline, and the configured maximum batching window is part of the benchmark manifest. A class name never authorizes waiting past either bound.

## Route aggregation key

When multiple sequences or prompt rows are available, aggregate token-route pairs by:

```text
(layer, device, expert, quantization format)
```

This grouping converts repeated GEMV-like work into grouped GEMM while preserving the original token identity, route weight, placement epoch, and deterministic inverse-gather mapping. Rows from different layers, owners, expert identities, or quantization formats are not combined into one expert job.

## Bounded dispatch rule

For each ready device/expert queue, dispatch when any one of these conditions becomes true:

```text
enough expert rows accumulated
OR oldest request nears its deadline
OR maximum batching window expires
```

The conditions are disjunctive. The scheduler must not wait for all three.

- **Enough rows:** the configured row threshold for the measured batch class has been reached.
- **Nears deadline:** waiting for more rows would consume the remaining dispatch budget of the oldest request. This condition overrides throughput aggregation.
- **Window expires:** time since the oldest queued eligible row reached the queue has reached the class's maximum batching window.

An optional prediction, promotion, prepack, or preload event is never a fourth required condition. Its absence cannot extend the window or block dispatch.

## Decode

Decode produces one new token per active sequence and has the tightest critical path.

For every sparse layer:

1. The state owner computes router logits and canonical top-k routes.
2. The scheduler resolves the current placement epoch and partitions the selected routes into RTX 5090-local, eligible B70-resident, and required CPU work.
3. It immediately publishes B70 work that is eligible for the current measured batch class.
4. It launches RTX 5090-local expert work concurrently.
5. It launches required CPU cold work concurrently rather than waiting for the local or B70 path to finish.
6. Each destination computes one weighted partial per original token for its assigned route subset.
7. The state owner waits at the explicit join for the slowest **required** participant, rejects stale-epoch or otherwise invalid completions, and combines valid partials exactly once in the defined order.
8. On a B70 timeout or failure, the exact CPU fallback supplies the required computation under the failure policy; stale late output is ignored.
9. The state owner continues the sequential model path only after the required join is complete.

A B70 may be memory-bandwidth limited at batch one. That is acceptable only if the measured complete resident remote path still beats CPU cold execution at batch-one p99. The scheduler does not infer eligibility from bandwidth arithmetic or isolated kernel measurements.

### Batch-one rule

For latency-critical batch-one decode:

```text
eligible(B70, batch-one) iff
    T_remote,p99(batch-one) < T_cpu_fallback,p99(batch-one)
```

`T_remote` includes CUDA D2H, queueing, B70 H2D, remap, both GEMMs, activation, gather, B70 D2H, CUDA H2D, and join. A device-local B70 win without a complete-path p99 win is insufficient.

If the inequality is not satisfied, selected experts use the qualifying local or CPU path for that class. The scheduler must not hold the request in hope that more tokens, a promotion, or a prediction will make the B70 path attractive.

## Continuous batching

With multiple active sequences, the scheduler collects the one decode row from each ready sequence and aggregates token-route pairs by the route aggregation key. It retains per-request deadlines while building device jobs.

Dispatch follows the bounded rule:

- foreground interactive queues use a strict deadline and the smallest allowed waiting budget;
- normal service queues use a small aggregation window;
- background agent or analysis queues may use a larger window;
- any queue dispatches immediately when its oldest request nears deadline, even below the preferred row threshold.

The scheduler may form independent jobs for different B70s and run them concurrently with RTX 5090-local and required CPU work. It does not wait for a slow or empty destination queue to fill before dispatching a ready destination whose own condition has fired.

Per-request joins remain explicit. Aggregating rows into one device job does not merge request lifetimes: each request becomes join-ready only when all of its required partials are complete or its defined exact fallback has completed.

## Prefill

Prefill uses all batch structure already present in the prompt:

- group prompt tokens by routed expert using the route aggregation key;
- dispatch large grouped jobs to eligible B70s;
- permit concurrent work across multiple B70 cards;
- retain attention and sequential state ownership on the RTX 5090;
- use larger staging transfers;
- allow a longer, still bounded, B70 batching window;
- allow background preload of experts predicted from earlier prompt chunks.

Prediction-based preload is an optimization only. If predicted weights or packed experts are not ready when required prompt work reaches its dispatch condition, the scheduler proceeds through the currently eligible resident/local/CPU paths. Prefill never waits beyond its deadline or maximum batching window for a prediction or promotion.

A larger B70 gain in prefill or high-concurrency decode than in batch-one decode is consistent with the intended XMX execution shape. It does not confer batch-one eligibility.

## Background work

The B70 pool may absorb the following only when doing so does not delay required foreground dispatch:

- speculative or background branches;
- route-profile calibration;
- batch agent sessions;
- low-priority prompt ingestion;
- placement-prepack validation;
- duplicate computations used for correctness sampling.

Background queues use larger bounded windows and lower dispatch priority. Foreground requests preempt them at dispatch boundaries. The scheduler does not assume it can interrupt an already submitted kernel or transfer; therefore it must bound background submissions so a non-preemptible stage cannot monopolize the next foreground dispatch opportunity.

Background results used for speculation, calibration, prediction, or correctness sampling are not required participants in a foreground layer join. Foreground never waits for them.

## Priorities, deadlines, and fairness

Priority controls dispatch order among queues that are ready at the same boundary; deadlines cap aggregation independently of priority.

1. A foreground queue whose oldest request nears deadline dispatches before optional/background work.
2. Other foreground interactive and normal-service queues dispatch according to their service deadlines and bounded windows.
3. Prefill uses aggressive aggregation but remains subject to its request deadline and must not block deadline-critical decode dispatch.
4. Background agent/analysis and maintenance work use spare dispatch opportunities and are preempted by foreground at dispatch boundaries.

Within a batch, retain the original request/token mapping so one hot expert, a skewed route distribution, or a large background batch cannot erase an older foreground request's deadline. Queue depth, queue time, batching-window expiration, and deadline-triggered dispatches are benchmark telemetry; see [`benchmarking.md`](benchmarking.md).

This policy does not authorize changing canonical routing, dropping required rows, or substituting approximate computation to meet a deadline. If a required B70 result is late or unavailable, use the defined exact fallback path.

## Concurrent execution and join semantics

The scheduler constructs one layer execution graph with independent branches:

```text
canonical routes
  ├─ RTX 5090 local expert subsets ─┐
  ├─ eligible B70 expert subsets ──┼─ ordered required join -> next model stage
  └─ required CPU expert subsets ──┘
```

The branches launch as soon as their inputs and reusable transport resources are ready. Independent branches do not wait for one another before launch.

A join waits for exactly the participants required by the canonical route partition after applying the failure policy:

- local, remote, and CPU partials assigned to the current token/layer/placement epoch;
- an exact CPU fallback that replaces a failed or timed-out required remote partial;
- no prediction, promotion, preload, speculative branch, calibration job, or duplicate correctness sample.

A remote completion that is stale, late after fallback, from the wrong placement epoch, or otherwise invalid is ignored and recorded; it is not joined a second time. Queueing multiple requests or experts never changes exactly-once route accounting.

## B70 eligibility by batch class

Eligibility is an empirical scheduling input derived from the transport-plus-compute matrix in [`benchmarking.md`](benchmarking.md). For every operational class, compare like-for-like p99 values:

```text
B70 eligible(class) iff
    T_remote,p99(class) < T_cpu_fallback,p99(class)
```

The comparison uses the same token rows, route distribution, quantization, output semantics, topology, contention, and cold/warm classification. `T_remote` is the complete path, not only B70 compute.

| Measured outcome | Scheduling consequence |
|---|---|
| B70 wins at batch one | B70 may serve latency-critical batch-one decode, subject to current health, residency, deadline, and placement |
| B70 loses at batch one | Do not use it for latency-critical batch-one decode |
| B70 wins only at batch `>= 4` | Use it only in winning classes such as batched decode, prefill, concurrent sessions, or background requests |
| B70 loses for a particular larger batch/topology/contention class | Do not generalize a win from another class; use the qualifying local/CPU path for the losing class |
| Measurement is absent, stale, or not comparable | Treat B70 as ineligible for that class until the required complete-path measurement exists |

The eligibility table is not a placement decision by itself. The selected expert must also be resident on a healthy B70 in the current placement epoch, and dispatch must still meet the request's bounded conditions. Conversely, optional promotion or prediction never blocks a ready qualifying path.

## Required scheduler telemetry

Each request/token record must make scheduling behavior auditable:

- service and batch class;
- request deadline and maximum batching window;
- token index, layer, selected expert IDs, route weights, and owners;
- placement epoch and B70 eligibility class used;
- row threshold, rows accumulated, oldest-row age, and which dispatch condition fired;
- queue depth and queue time per destination;
- local, remote, and CPU launch/completion times;
- required join participants and join wait;
- fallback, timeout, stale/discarded completion, and failure reason;
- whether prediction, promotion, or preload was available, used, late, or ignored;
- output token/logit checksum.

Report batch-one and continuous batching separately, prefill and decode separately, first-run and warmed operation separately, and p50/p95/p99 rather than only averages. Scheduling acceptance depends on end-to-end evidence and output agreement, not queue throughput alone.

## Scheduling acceptance rules

The scheduling policy is acceptable only when measurement demonstrates all of the following for the exercised classes:

- no foreground request waits for optional prediction, promotion, prepack, preload, or background work;
- dispatch occurs when any bounded condition fires, and never waits beyond the applicable maximum window or deadline budget;
- foreground work preempts background queues at dispatch boundaries;
- RTX 5090-local, eligible B70, and required CPU branches overlap where dependencies allow;
- each required route is accounted for exactly once at the ordered join;
- timeout/failure invokes exact fallback and ignores stale late output;
- batch-one B70 use is backed by a measured complete remote-path p99 win over CPU fallback for batch one;
- wins at batch 4 or greater are confined to the measured winning batched-decode, prefill, concurrent-session, or background classes;
- p99, output agreement, topology, contention, and cold/warm behavior remain visible in the evidence.
