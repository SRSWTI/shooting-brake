# Shooting Brake Correctness Contract

## Purpose and status

This document defines the correctness contract for the heterogeneous expert fabric described in [`architecture.md`](architecture.md). It is a **normative design target and validation plan**, not evidence that an implementation, hardware configuration, kernel, or benchmark has passed. The conditions that can stop or narrow the design are tracked in [`risk-register.md`](risk-register.md).

The terms used below distinguish four kinds of statement:

- **Invariant** — behavior every conforming execution must preserve.
- **Acceptance gate** — evidence required before a path may be treated as correct.
- **Design target** — intended behavior that still requires measurement or validation.
- **Upstream assessment** — a feasibility claim inherited from the design source, not an observed result in this repository.

## Exactly-once route invariant

Placement must be invisible to model semantics. For every sparse layer, every route selected by the model's current router must be accounted for **exactly once**:

1. the selected expert identity, route weight, token mapping, and canonical route order are preserved;
2. the route is assigned to one active execution path: a resident owner or an exact request-scoped CPU fallback;
3. its weighted expert result is committed to the layer join once; and
4. no selected route is lost, substituted, zero-filled, rerouted to a different logical expert, or committed twice.

A physical owner may change across placement epochs. The logical expert computation may not. A placement change therefore affects where work runs, not which work the router selected or how many times it contributes.

A zero route weight does not waive accounting: the selected route must still receive defined handling and must not be mistaken for missing work. Masked or padded entries are handled according to the canonical router semantics and must not be promoted into selected routes. An empty B70 subset is valid; it contributes the additive identity and does not excuse missing routes assigned elsewhere.

### Forbidden approximations

The default configuration must not use any of the following:

- expert dropping;
- approximate top-k;
- placement-biased routing;
- route caching that skips the current router;
- lossy fallback;
- treating a missing expert as zero;
- forced capacity overflow;
- silent precision reduction; or
- stale placement IDs.

Prediction, caching, placement, batching, and transport optimizations are permitted only when they leave the exactly-once route invariant unchanged. Faster execution without a correctness oracle is not a valid result.

## The only permitted outcomes

For each selected expert, exactly one of these outcomes is permitted:

1. its resident owner succeeds and its result is committed once;
2. an exact, request-scoped CPU fallback executes and its result is committed once; or
3. the request fails explicitly.

There is no fourth outcome. In particular, timeout, worker reset, invalid metadata, or transport failure must not turn into a zero, a lossy substitute, an unreported route drop, or an indefinite wait.

If a B70 worker misses its deadline, the coordinator must stop waiting indefinitely. It may run the exact CPU fallback if that remains viable; otherwise it must fail the request explicitly. Any later B70 completion is stale and must be rejected. Repeated worker failures are recorded and cause worker quarantine.

## Deterministic dispatch and joining

The state owner remains the source of the real router output. Dispatch partitions that output by execution owner without changing route identity or order. For the subset assigned to a B70, the returned partial must represent

$$
\mathrm{B70Partial} \approx \sum_{e\in\text{B70 routes}} w_e\,\mathrm{Expert}_e(x),
$$

where $\approx$ acknowledges only characterized floating-point effects. It does not permit missing, substituted, or duplicated routes.

The join must satisfy all of the following:

- the 5090, B70, and CPU subsets together account for all selected routes;
- the B70 receives only the subset assigned to it;
- owner-local and remote partials join once;
- canonical route order is retained;
- duplicate work cannot be added a second time;
- a fixed configuration produces deterministic output; and
- no unexplained route divergence is accepted.

A duplicate completion for an already completed logical route is rejected, not accumulated. An undefined duplicate expert ID in an input route subset is rejected before it can alter the join; any explicitly valid repeated expert selection remains represented as distinct canonical route positions and each such selected position is accounted for once.

## Publication and stale-completion protection

Request and response publication use release/acquire ordering:

- a request is not published until the CUDA device-to-host transfer has completed; and
- a response is not published until the B70 output transfer has completed.

Before a completion can affect the join, it must match the active request on every one of these fields:

- request ID;
- generation;
- placement epoch;
- layer;
- token count; and
- buffer version.

A mismatch in any field rejects the completion. A placement-epoch change invalidates work from the prior placement even if the expert ID is otherwise valid. Late, duplicate, and mismatched work therefore fails closed: it cannot be added to a later token, another layer, a new buffer generation, or a new expert placement. The active request must then complete through another permitted exact outcome or fail explicitly.

## Validation hierarchy

Correctness is established in four levels. Passing a lower level does not imply that a higher level passes.

### Level 1 — Tensor comparison

Compare individual expert outputs with the canonical CPU oracle, including the gate/up/down computation, SiLU multiply, W4 scales, and floating-point accumulation. Record numerical error rather than relying on kernel throughput alone.

### Level 2 — Layer replay

Replay a heterogeneous MoE layer with the same router output. Require the same selected expert IDs, exact route accounting and order, and bounded layer-output error. Exercise routes on B70, 5090, and CPU in the same layer, including deadline fallback, stale completion, and worker reset.

### Level 3 — Teacher-forcing replay

Across long sequences with fixed prompts and teacher-forced tokens, compare logits and route IDs with the canonical owner. Require canonical route IDs, only the assigned subset at each B70, exactly-once joining, bounded and characterized numerical drift, consistent top-1 logits/tokens, and coherent multi-turn state.

### Level 4 — Generation oracle

Run token-exact greedy replay over an agreed corpus where practical and run the required quality benchmarks. Generated tokens are compared end to end, including multi-turn context and intentional B70 failure. Mixed quantization or reduced-return modes require explicit quality approval before acceptance.

## Numerical acceptance

Cross-vendor kernels may change floating-point accumulation order. Numerical equality is therefore evaluated with these requirements:

- expert tensor and layer-output error is bounded and characterized;
- logit error is bounded and characterized;
- route divergence must be explained; unexplained divergence fails the gate;
- top-1 logits and generated tokens remain consistent in fixed-prompt teacher-forcing checks;
- greedy generation is token-exact over an agreed corpus where practical; and
- quality approval is explicit before mixed quantization or reduced-return output is enabled.

No numeric tolerance is specified by the source design. A test plan must agree and record tolerances before evaluation; this document does not invent them. A path does not pass merely because its output appears close, and a performance improvement does not override route or token behavior that changes materially.

## Multi-turn and KV-state equivalence

Multi-turn state and prefix/KV persistence must remain observationally equivalent to the canonical execution across turns:

- every sparse layer executes through the fabric during the supported-model gate;
- multi-turn context remains coherent;
- prefix and KV persistence do not reuse expert results in place of running the current router;
- concurrent requests cannot exchange route, generation, epoch, layer, token-count, or buffer state;
- teacher-forced route IDs and top-1 logits/tokens remain consistent; and
- greedy generated tokens are compared with the oracle, including after an intentional B70 failure.

A unit tensor comparison cannot establish multi-turn or KV equivalence. That claim requires the end-to-end replay and generation gates above.

## Explicit failure matrix

| Condition | Required handling | Forbidden effect | Evidence required |
|---|---|---|---|
| Resident owner completes with matching identity and current epoch | Accept once and mark its logical route complete | A second commit of the same route | Layer replay covering one and multiple routes |
| No routes are assigned to a B70 | Return/consume the defined empty partial and continue accounting elsewhere | Treating the empty subset as proof that all selected routes completed | Empty-subset layer replay |
| One or multiple valid routes are assigned to a B70 | Compute only that subset and join each selected route once | Lost, extra, or owner-mismatched routes | Subset tests for one and multiple routes |
| B70 deadline expires and exact CPU fallback is viable | Execute the request-scoped CPU fallback; reject any late B70 completion | Indefinite wait or later double addition | Deadline test with CPU-oracle comparison |
| B70 deadline expires and exact fallback is not viable | Fail the request explicitly | Zero fill, lossy result, silent drop, or indefinite wait | Explicit-failure test |
| Completion arrives after fallback, cancellation, or request advancement | Reject it as stale | Mutation of the current or later request | Late-completion replay |
| Completion duplicates already committed work | Reject the duplicate | Double-counted expert contribution | Duplicate-completion replay |
| Request ID, generation, placement epoch, layer, token count, or buffer version mismatches | Reject before the join; use another permitted exact outcome or fail explicitly | Cross-request, cross-token, cross-layer, cross-buffer, or cross-placement contamination | One negative case for each identity field |
| Placement epoch changes while work is outstanding | Invalidate the prior-epoch completion; recompute exactly if viable or fail explicitly | Addition of a result from the old owner/placement | Placement-epoch-change replay |
| Worker resets or becomes unhealthy | Use exact CPU fallback if viable, otherwise fail explicitly; quarantine on repeated failures | Partial success represented as a complete route set | Worker-reset and intentional-failure replay |
| Local expert ID or expert-parallel mapping is invalid | Reject the work before execution or join | Substitution with a different resident expert | Invalid-ID and mapping tests |
| Input contains an undefined duplicate route ID | Reject or apply the canonical explicitly defined handling before dispatch; never double count | Ambiguous or repeated accumulation | Duplicate-ID test documenting the canonical behavior |
| Route is masked/padded | Preserve the canonical mask/padding semantics | Executing padding as a selected expert or dropping a real selected route | Masked/padded route test |
| Selected route has zero weight | Account for it under canonical route semantics | Treating it as evidence that some other selected route may be omitted | Zero-weight route test |
| Cross-vendor result exceeds the agreed numerical gate or materially changes route/token behavior | Reject that execution path as validated; investigate against the oracle | Accepting speed as a substitute for correctness | All four validation levels, as applicable |

## End-to-end acceptance

A heterogeneous path is correctness-eligible only after it demonstrates, against Colibri's all-CPU or CUDA oracle as applicable:

- all selected experts accounted for exactly once in a real heterogeneous layer;
- route identity and canonical order preserved;
- deterministic output for a fixed configuration;
- bounded and characterized tensor/logit drift under agreed tolerances;
- coherent teacher-forced, multi-turn, KV/prefix, and greedy-generation behavior;
- exact CPU degradation or explicit failure under deadline and worker faults;
- stale and duplicate work rejected; and
- no correctness regression while placement or scheduling optimizations are introduced.
