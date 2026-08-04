# Memory Planes and Ownership

## Purpose and status

This document defines the memory planes, owners, persistence boundaries, invalidation rules, and budget constraints for the Shooting Brake design in [`architecture.md`](architecture.md). It is a design contract; it does not claim that the complete model is resident, that device memories form a unified pool, or that the target capacities have been validated. Hardware and transfer assumptions are governed by [`hardware.md`](hardware.md).

“Long-term memory” is not one cache. Model weights, inference context, placement learning, and application semantics have different authorities and lifecycles and must remain separate.

## Global rules

1. Every allocation belongs to exactly one plane and one current memory domain. Capacity on the RTX 5090, each B70, host DDR5, and NVMe must be accounted independently; their nominal capacities must not be summed as unified VRAM.
2. The RTX 5090 is the state owner. It remains authoritative for active context, attention/KV state, router decisions, dense and shared layers, sampling, and the sequential residual stream.
3. B70s are resident expert workers, not context owners. Initial serving does not distribute KV, prefix, sequence, or sampler state to them.
4. Device placement changes performance, not the requested model computation. A missing or unhealthy resident copy must resolve to an exact available execution path, not an approximation silently substituted for it.
5. NVMe is outside the normal warmed token path. Ordinary foreground decode must not synchronously read expert weights from storage.
6. Each plane has an explicit budget and admission policy. Allocation failure must not be hidden by borrowing unbounded space from another plane or by introducing a foreground NVMe dependency.
7. Persistence does not imply foreground ownership. Host or NVMe copies may be recovery or inactive-state backing while the active authoritative state remains elsewhere.

## Plane 1: model-weight memory

### Tier ownership

| Tier | Intended contents | Owner and lifetime | Foreground rule |
|---|---|---|---|
| RTX 5090 hot tier | Dense core, shared experts, and hottest routed experts | State-owner runtime; resident for the loaded model/placement epoch | Local execution; preserve capacity reserved for context and runtime scratch |
| B70 warm tiers | Separately resident routed-expert banks | Each B70 worker owns its own arena and queues; residency is permanent or placement-epoch-stable | Execute only experts present in that worker's admitted bank; no token-path weight migration |
| DDR5 cold tier | Complete exact cold routed bank **only for models that fit the configured host-memory budget** | CPU fallback provider | Exact CPU execution and timeout recovery without changing requested expert IDs, weights, or precision |
| NVMe frozen tier | Full model repository, packed source or frozen alternate formats, cold-start material, and checkpoints | Storage/recovery layer | Startup, checkpoint recovery, and optional background preload only; never an ordinary per-token expert tier |

“Hot,” “warm,” “cold,” and “frozen” describe placement and access policy, not a coherent address space. A weight resident on one B70 is not local to another B70 or to the RTX 5090. The transport path moves activations and weighted partials, not expert weights, during ordinary decode.

### Persistence and invalidation

The NVMe repository is the durable source for model startup and recovery. RTX 5090 and B70 resident arenas are derived runtime state and can be reconstructed from validated packed artifacts. DDR5 fallback contents remain admissible only while they exactly match the active model, expert precision/format, and placement metadata.

A model hash or model-version change invalidates resident-device banks, DDR5 fallback banks, packed-artifact associations, and any placement profile tied to the previous model. A placement-epoch change invalidates in-flight routing to the old placement but does not by itself permit a stale weight or response to be reused. Worker completions must match request ID, generation, placement epoch, layer, token count, and buffer version before reduction.

### Budget and admission

Budget the RTX 5090 separately for model-owner tensors, local routed experts, active KV/prefix state, pinned-transport participation, scratch, and safety headroom. Budget each B70 for its own packed expert bank, activation/result buffers, command resources, and safety headroom. Budget host DDR5 for the CPU fallback bank, pinned rings, inactive context backing, runtime memory, and operating-system headroom.

Admission must use actual allocatable capacity rather than nominal board or DIMM capacity. If the exact DDR5 fallback bank does not fit after host reservations, the runtime must not describe it as complete.

### The 64 GiB / GLM limitation

With 64 GiB of DDR5, a fully RAM-resident 372 GB GLM model is impossible. Even after placing some experts across the RTX 5090 and B70s, the first full-GLM run still requires a storage tier for experts not resident on those GPUs or in RAM. That run must be reported as storage-backed; it is not a “no-NVMe” result.

For this configuration, GLM-5.2 is an experimental, locality-dependent workload. A smaller model should be used for production unless the exact fallback set fits the admitted DDR5 budget. The NVMe-backed full-GLM path must not be described as robust interactive serving, because ordinary foreground NVMe expert reads are prohibited. A locality miss that can be served only by a synchronous storage read is an admission/configuration failure for the warmed interactive path, not a new execution tier.

## Plane 2: context memory

### Authority and contents

The RTX 5090 state-owner runtime owns active:

- KV cache;
- prefix cache;
- sequence and request state;
- sampler state;
- any future multimodal or tool state that participates in model execution.

This authority is not delegated to B70 workers. B70 requests carry the activations and local route metadata needed for routed-expert computation; they do not confer ownership of attention or conversation state.

### Initial non-distribution of KV

KV participates in every attention layer. Distributing it to the B70s would add a mandatory remote dependency to a path that is otherwise wholly owned by the RTX 5090. The initial architecture therefore **must not distribute KV to B70s**. No aggregate device-memory figure may be used to imply that B70 VRAM expands the 5090 KV pool.

The desired context behavior is exact multi-turn KV reuse, prefix deduplication across requests, and bounded active GPU allocation. Existing persistent KV/prefix facilities are the intended basis; repeatedly recomputing the full conversation is not the target behavior.

### Persistence and invalidation

Active context stays with the 5090 authority. Host or NVMe may persist **inactive** contexts; such persistence is not permission to put a storage access on every attention layer or token. Restore must re-establish state-owner authority before the context becomes active.

Context entries are invalidated when their content identity or model version no longer matches. Sequence lifecycle and sampler state must remain consistent with the restored KV/prefix state; a partially matching context must not be reused as if complete.

### Budget

Active GPU KV and prefix allocation is bounded. Its configured reservation must be made alongside model weights, local experts, scratch, pinned transport, and safety capacity before serving. Admission or eviction handles exhaustion; allocation must not spill active KV to B70 memory under the initial design. Host/NVMe persistence is limited to inactive contexts and has its own retention and capacity policy.

## Plane 3: placement-learning memory

### Ownership and contents

The placement scheduler owns the learned execution profile. Persist at least:

- global per-layer expert frequency;
- session frequency;
- recent-window frequency;
- expert coactivation counts;
- per-device service-time distributions;
- transport p50, p95, and p99;
- queue delay;
- migration outcomes;
- prediction precision and recall;
- failure rates;
- profile version and model hash.

This plane learns where an exact expert computation should execute; it does not change canonical router results, selected experts, route weights, expert precision, or reduction semantics.

### Persistence, invalidation, and budget

Persist the profile across runs so a restart need not relearn every placement from zero. The model hash and profile version are part of its identity. A model-hash mismatch invalidates model-specific statistics for placement decisions. Incompatible profile versions must be rejected or explicitly rebuilt, never interpreted under a different schema. Session and recent-window data expire according to their defined windows; a placement-epoch transition prevents stale in-flight completions from being accepted.

The scheduler must bound session histories, recent windows, coactivation data, and retained performance distributions. Placement telemetry is control-plane data: its storage and processing budget must not consume unbounded token-path memory or turn profile persistence into a synchronous foreground dependency.

## Plane 4: application semantic memory

### Ownership and boundary

Agent memory, embeddings, retrieval indexes, user facts, and vector search remain outside the model execution fabric. An application or retrieval service owns them. They affect request construction by supplying retrieved context through the API; they do not own model weights, KV, canonical routing, expert placement, or worker state.

### Persistence, invalidation, and budget

The external retrieval store owns durability, retention, tenancy, deletion, content refresh, embedding/index versioning, and its capacity budget. Its invalidation policy must not be conflated with model-weight or KV invalidation. Once retrieved material is included in a request, the resulting active model context is governed by the context plane and the 5090 state owner.

The execution runtime must not reserve GPU expert or KV capacity for an implicit application-memory database. Conversely, application retrieval must not reach into placement-learning records or device-resident expert arenas as a semantic store.

## Cross-plane lifecycle

### Startup

1. Audit actual device, host-memory, pinned-memory, scratch, safety, and storage capacity as required by [`hardware.md`](hardware.md).
2. Validate the model manifest and packed artifacts against the active model identity.
3. Admit 5090 owner tensors and local experts within the 5090 budget.
4. Admit a separate resident bank on each B70 within that card's budget.
5. Establish an exact DDR5 fallback bank only if it fits after all host reservations.
6. Restore only placement profiles whose model hash and profile version match.
7. Reserve bounded active context capacity on the 5090.
8. Keep NVMe activity in startup, recovery, or background staging.

### Placement change

A new placement epoch may replace resident expert assignments only through controlled background loading and admission. Requests retain their request-scoped placement epoch. A completion from an earlier epoch is rejected rather than reduced into a newer token. Context ownership does not move when expert placement changes.

### Recovery and eviction

- Evict or replace derived resident weight copies without changing the durable model identity.
- Evict active context only through the context lifecycle; persist only inactive contexts to host/NVMe.
- Age or rebuild placement-learning records under their bounded retention rules.
- Leave application-memory retention to the external store.
- Never convert an expert-residency miss into an ordinary foreground NVMe read.

## Required reporting

Capacity and performance reports must state, separately:

- actual allocated and reserved bytes on the RTX 5090;
- actual allocated and reserved bytes on each B70;
- DDR5 used for exact fallback, pinned transport, inactive context, and runtime/headroom;
- which portion of the exact cold bank is present in DDR5;
- NVMe bytes and activity attributable to startup, recovery, background preload, or any experimental foreground miss;
- whether KV remained wholly under 5090 authority;
- active model hash, placement epoch, and placement-profile version.

A report is misleading if it labels aggregate GPU capacity “unified VRAM,” calls a partially resident 372 GB GLM deployment “RAM-resident,” claims NVIDIA–Intel P2P without the evidence required by [`hardware.md`](hardware.md), or hides foreground storage reads inside an alleged warmed decode result.
