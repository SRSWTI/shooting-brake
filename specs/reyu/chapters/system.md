# System overview: one hybrid routed-expert invocation

Read the [repository HLD](../README.md) first, then [system.ryu](../models/system.ryu). This branch is the small whole-system observation model beneath the breadth-first architecture map. It does not replace the detail models.

## Boundary and actors

vLLM drives the plugin. The local CUDA stream and two selected remote B70 lanes produce one layer's partial results. CUDA joins them. Startup succeeds or rejects configuration; a later invocation can reuse the execution path. Attention, HTTP delivery, token sampling and vLLM's scheduler are external at this level. `Joined` means the routed-expert result was assembled, **not** that the HTTP client received a valid token.

Source anchors:

- [production supervisor](../../../serve_production.sh)
- [plugin registration](../../../src/phase4/src/shooting_brake_vllm/__init__.py), `register`
- [hybrid layer implementation](../../../src/phase4/src/shooting_brake_vllm/routed_experts.py), `HybridRoutedExperts._hybrid_forward_modular`, `_b70_issue_graph`, `_b70_take_graph`
- [native poller](../../../src/phase7/b70_capi.cpp), `B70Poller::loop`

## HLD actions and detail boundaries

| HLD observation/action | Meaning | Lower-level expansion |
|---|---|---|
| `admit` / `reject` | Construction can proceed or admission fails | Configuration/model/bank checks, registration, placement and provider startup |
| `route` | The layer has an ownership-classified routing decision | Per-layer expert owners, compact IDs, offloaded route masking |
| `launch` | The invocation starts local/remote work | Staging copies, signals, eager issue or graph replay |
| `finishLocal` | Local CUDA partial is available | Local expert compute; no floating-point payload modeled here |
| `releaseLane` | One remote lane releases its waiting consumer | Poller/provider issue/take and completion publication; success is separate |
| `join` | All required partial paths have reached the join | Per-lane completion waits, copies, casts, partial summation |
| `observeFailure` | Host-side code notices unsuccessful remote execution | Error counters and eager-side checks outside the captured CUDA graph |
| `nextInvocation` | A new invocation starts with fresh observations | Detailed reuse/reset rules; this HLD action itself is model bookkeeping |

The pure functions in [architecture.ryu](../models/architecture.ryu) provide shared observations for detail models. `allPartsReleased` is a join-readiness predicate. `observeResult` explicitly separates waiting, successful result, and release without a valid result. These functions do not add runtime flags, locks or guarantees.

## State, atomicity and invariants

`s.phase` records the invocation stage. `released` and `valid` record different facts about each lane. `localDone`, `joined` and `errorObserved` describe observations. All are model state, not a proposed memory layout.

At this level each startup outcome and lane computation is atomic. The detailed branches split operations wherever intermediate state is observable. In particular the HLD cannot establish memory visibility, flag reset correctness or safe buffer reuse by itself.

`inv` asserts:

- only selected lanes appear in the release/valid sets;
- a valid lane result has also released its waiter;
- a joined invocation has completed its local work and all remote releases;
- an observed remote failure belongs to a joined invocation with at least one invalid partial.

It intentionally does **not** assert `joined implies valid`: the native classic poller releases completion even after a provider failure. Whether and when such a failure becomes observable to the caller is part of the detailed failure path, not an invariant assumed away here.

## Executable explanations

- `concurrentJoinTest`: the second lane finishes before local work and the first lane, yet the final join waits for all three.
- `failedCompletionIsNotSuccessTest`: one lane releases after failure, join becomes possible, and the invalid-result observation remains distinguishable from success.
- `cannotJoinEarlyTest`: local completion plus only one remote release cannot join.
- `nextInvocationClearsObservationsTest`: model bookkeeping from a completed invocation is not reused as a fresh completion.
- `rejectedAdmissionTest`: a rejected startup cannot route work.

`successfulJoin` and `releasedFailure` are witnesses, not universal properties. `joinProgress` is a temporal statement conditional on weak fairness for `join`; the normal suite typechecks it but does not model-check liveness.

## Limits and source comparison

Two lanes are a concrete bounded instance of the production topology, not a restriction on the underlying placement API. Local CUDA computation is treated as eventually able to finish only in selected scenarios; the simulator may also remain in a state because no scheduler fairness is imposed. The HLD makes no timing, numerical accuracy, GPU-memory-coherence, crash recovery or API-level delivery guarantee.

The source map records the relevant implementation files. Passing these scenarios establishes the behavior of this abstraction. Separate detail models and implementation comparisons are needed to support the correspondence to source; no refinement proof is claimed.
