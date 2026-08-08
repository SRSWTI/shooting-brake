# Expert Fabric

## Purpose, authority, and status

This document specifies the versioned pinned-memory protocol between the Qwen-scoped `HybridMoERunner`/`HybridRoutedExperts` adapter in upstream vLLM 0.26+ and one isolated, persistent QuixiCore-XPU B70 routed-expert provider. [`../plan.md`](../plan.md) is authoritative.

This is the normative production transport design. Phase 1 completed and directly qualified the native provider core and isolated control process; Phase 2 completed the protocol-v2 multi-slot process ring, physical CUDA/Level-Zero mapping probe, isolated B70 server, lifecycle stress, and direct numerical/latency gate; and Phase 3 completed the independent provider-mathematics matrix over that process ring on the physical B70. The existing Colibri native worker remains the transport, placement, and failure-semantics comparator, not the production endpoint. Phase 4 upstream-vLLM adapter integration and CUDA scatter/join remain open.

The terms **MUST**, **MUST NOT**, **SHOULD**, and **MAY** are normative.

Related contracts:

- [`correctness.md`](correctness.md) defines route, join, numerical, and recovery correctness.
- [`model-format.md`](model-format.md) defines the common source and provider-specific artifacts.
- [`placement.md`](placement.md) defines placement inputs where consistent with the active plan.
- [`scheduling.md`](scheduling.md) defines admission and deadline policy where consistent with the active plan.

## Runtime boundary

```text
upstream vLLM CUDA worker on RTX 5090
  scheduler, state, attention, KV/GDN, canonical router/top-k
  CUDA-owned routed experts, shared expert, residual, sampling
        |
        | fixed versioned pinned-memory request ring
        v
isolated persistent B70 provider process
  qualified QuixiCore-XPU NVFP4 MoE operators
  compact B70 expert bank, fixed XPU buffers, streams, kernel policy
        |
        `-> one weighted [M_remote, hidden] partial + route status
```

CUDA tensors do not become XPU tensors directly. The adapter asynchronously stages selected activation rows and route metadata through pinned host memory. The provider copies those buffers into preallocated XPU tensors, runs preselected-route kernels, and copies its reduced partial back through pinned memory. NCCL/XCCL collectives, direct CUDA-to-XPU calls, ordinary Python object exchange, JSON, protobuf, gRPC, and per-request pipes are not this token-path transport.

There is one B70 provider for the production target. It owns only:

- B70-resident routed-expert weights in compact slots;
- fixed pinned/XPU input, output, status, and scratch storage;
- XPU streams and provider-private events;
- selection among qualified tiny, small/grouped, and prefill kernels; and
- request execution, health, and telemetry.

The provider MUST NOT own or recompute attention, KV/recurrent state, router logits, softmax, top-k, route normalization, shared experts, residuals, LM head, sampling, serving, or the final routed-result join.

CPU owns orchestration, queue management, telemetry, and exact emergency recovery. It MUST NOT perform normal-path expert matrix compute.

## Canonical routing and compact ownership

The CUDA state owner supplies canonical:

```text
hidden             [M, hidden]  FP16 or BF16
topk_ids           [M, topk]    int32
topk_weights       [M, topk]    FP32 or FP16
```

For a captured placement generation:

```text
(layer, global expert) -> CUDA local slot
(layer, global expert) -> B70 compact slot
```

These mappings partition selected routes without changing their weights. Every selected route has exactly one normal-path owner. B70 compact slots are dense provider-local indices and MUST NOT be exposed as global model identities, fake expert-parallel ranks, or a reason to reorder canonical route positions.

The adapter stages only token rows with at least one B70-owned route. It MUST retain the original token-row map and a stable identity for every staged canonical route position. If no row has a remote route, it MUST skip provider submission and use the additive-identity remote partial.

## Batched transaction contract

One transaction represents the aggregate remote work for one active MoE layer and one vLLM scheduler step. The adapter MUST aggregate across active requests; it MUST NOT submit one Level Zero/XPU operation per request when those rows belong to the same layer step.

Logical request payload:

```text
activation          [M_remote, hidden]  FP16 or BF16
expert_ids          [M_remote, topk]    int32 canonical global IDs
routing_weights     [M_remote, topk]    FP32 or FP16
remote_route_mask   [M_remote, topk]
token_row_map       [M_remote]          original row in CUDA batch
route_position_map  [num_remote_routes] canonical token/route identity
```

Logical response payload:

```text
remote_partial      [M_remote, hidden]  FP16 or FP32
route_status        [num_remote_routes] terminal completion/recovery status
token_status        [M_remote]          optional summary, never a substitute for route status
```

`M_remote <= M`. Fixed capacities negotiated at startup bound `M_remote`, `hidden`, `topk`, routes, ring slots, and in-flight operations. Dimensions and byte extents MUST be validated with overflow-safe arithmetic before publication. A request beyond a negotiated bound fails before any payload is consumed; the ring and XPU tensors MUST NOT resize on the token path.

Padded positions MUST be masked. The provider MUST reject invalid global IDs, compact slots, route maps, shapes, dtypes, or unsupported canonical duplicate behavior. It MUST NOT speculatively read weights or scales for invalid/masked routes.

## Request and completion descriptors

Concrete structs MAY add size fields and reserved words, but the protocol MUST carry at least:

```text
protocol_version
descriptor_size
request_seq
ring_slot
provider_generation
placement_generation
weight_generation
layer
num_batch_tokens           # full scheduler-step M
num_staged_tokens          # compact M_remote
num_routes
hidden_size
topk
activation_dtype
weight_dtype
output_dtype
request_buffer_version
output_buffer_version
placement_fingerprint
route_subset_fingerprint
deadline
status / error
```

The completion echoes every identity and shape field needed to bind it to the immutable request. Meanings are distinct:

- `request_seq` is the monotonic publication identity and protects ring wraparound;
- `ring_slot` identifies the fixed storage lane;
- `provider_generation` changes after provider restart or destructive reset;
- `placement_generation` and its fingerprint bind global-to-compact ownership;
- `weight_generation` binds the resident expert bank;
- buffer versions distinguish successive contents in stable allocations; and
- the route-subset fingerprint binds the reduced partial to the exact canonical remote routes.

No field substitutes for another. Counters MUST be wide enough that a live stale reference cannot alias current work. If safe wraparound cannot be proved, admission stops and a fresh provider generation/ring is established.

Before CUDA copies or joins a response, the adapter MUST match protocol, sequence, slot, provider/placement/weight generations, layer, dimensions, dtypes, buffer versions, provider identity, placement fingerprint, route-subset fingerprint, and terminal status. A mismatch makes the payload uncommittable.

## B70 partial semantics

For staged row `r`:

$$
\mathrm{remote\_partial}[r,:]
=
\sum_{j:\mathrm{remote\_route\_mask}[r,j]}
w_{r,j}\,\mathrm{Expert}_{e_{r,j}}(X[r,:]).
$$

The provider MUST:

- use the canonical IDs and weights supplied by CUDA;
- remap global IDs only through the current compact ownership table;
- run only B70-owned routed experts;
- preserve each original weight without subset renormalization;
- return exactly one already-weighted, already-summed row per staged token;
- emit zero for a staged row whose valid remote subset is empty;
- support multiple tokens selecting the same expert and multiple experts selecting one activation without duplicate activation transfer;
- exclude failed routes from the returned partial and identify them exactly in `route_status`; and
- allocate no weights, payloads, or scratch tensors during steady-state dispatch.

If execution cannot prove which routes contributed after an error, the provider MUST mark the entire transaction's remote subset uncommittable. It MUST NOT return an ambiguous combined partial.

The initial provider selects qualified QuixiCore-XPU NVFP4 MoE families:

| Batch class | Provider-local policy |
|---|---|
| `M=1` decode | Qualified NVFP4 MoE fused/split path |
| `1<M<=32` decode/continuous batch | Qualified NVFP4 MoE batched path |
| Larger decode batch | Qualified NVFP4 MoE grouped path |
| Prefill | Qualified NVFP4 MoE grouped path |

The selected primary is QuixiCore-XPU NVFP4. llm-scaler INT4 through `vllm-xpu-kernels` remains a qualified secondary alternative only if NVFP4 quality is insufficient and must pass the same capability, correctness, and performance gates before use.

Thresholds are capability/benchmark facts owned by the provider, not model-name branches in vLLM. Full-fused functions that recompute router/top-k or shared-expert work are prohibited.

## Concurrent CUDA execution and join

After publishing the remote request, CUDA runs its local routed experts concurrently using the stock-compatible vLLM backend. Remote route positions are invalid/skipped for that backend; if the selected CUDA kernel cannot skip them, `HybridRoutedExperts` MUST compact local routes and unpermute/reduce their results.

The provider response is scattered by `token_row_map` into a preallocated CUDA `[M, hidden]` buffer. The state owner then computes:

```text
routed = CUDA_local + B70_remote + exact_recovery
```

in a documented stable accumulation order. The B70 partial MUST join **before any final tensor-parallel or expert-parallel reduction**. Shared-expert and residual behavior remain canonical upstream vLLM CUDA behavior. Completion order MUST NOT choose numerical accumulation order.

Initial qualification is TP=1. TP/EP configurations are separate capabilities and MUST fail closed until their join/reduction position is validated.

## Pinned-memory ring and ownership

The ring has fixed capacity, stable addresses, multiple request slots, and separate preallocated request/response payload regions. A logical slot follows:

```text
FREE
  -> CUDA_WRITING
  -> REQUEST_READY
  -> XPU_RUNNING
  -> RESPONSE_READY
  -> CUDA_READING
  -> FREE
```

These are cross-process ownership states; CUDA events, XPU/SYCL events, and DMA operations are provider-private sub-events.

| Transition | Required proof |
|---|---|
| `CUDA_WRITING -> REQUEST_READY` | CUDA D2H and all request descriptor writes are complete |
| `REQUEST_READY -> XPU_RUNNING` | Provider acquired and validated the complete descriptor |
| `XPU_RUNNING -> RESPONSE_READY` | XPU kernels, XPU D2H, route status, and completion descriptor writes are complete |
| `RESPONSE_READY -> CUDA_READING` | CUDA adapter acquired and validated the completion |
| `CUDA_READING -> FREE` | CUDA H2D/join reads and all provider references are complete |

Cross-owner publication and reuse MUST use atomic release/acquire semantics. Volatile access, elapsed time, an event without host publication, or a device-wide synchronization is not a substitute.

`FREE` is a proof obligation. Timeout, cancellation, fallback, shutdown, provider death, and generation change do not permit reuse while either runtime might still touch a slot. If completion cannot be proved after failure, storage is quarantined until generation teardown makes reuse safe.

The decode path MUST NOT allocate, call `.item()`, synchronize the entire CUDA device, grow a queue, or hold a global mutex while awaiting provider work.

## Lifecycle

The provider exposes versioned operations equivalent to:

```text
capability
load
issue
take
health
shutdown
```

- **Capability** returns device identity/memory, protocol and kernel-bundle versions, dependency fingerprint, supported model dimensions, dtypes, top-k, weight formats, group sizes, scale formats, layouts, kernel families, maximum capacities, stable-address/graph guarantees, provider generation, weight generation, and placement fingerprint.
- **Load** validates the model/provider artifact and installs each B70-owned expert once in a compact resident bank. Upload, prepack, quantization, and allocation are control-plane work.
- **Issue** validates and nonblockingly claims one bounded ring slot. Full capacity returns immediate backpressure.
- **Take** nonblockingly reports a terminal response after output visibility; bounded event/futex notification is preferred over indefinite CPU polling.
- **Health** reports readiness, generation, placement/weight generation, queue saturation, and terminal fault state without exposing XPU objects.
- **Shutdown** stops admission and performs a bounded drain; otherwise it invalidates the generation and quarantines unresolved resources.

The production provider is a separate process to isolate CUDA from PyTorch-XPU/oneAPI/Level Zero runtime and dependency state. It is not a second vLLM server.

## Capability negotiation and fail-closed admission

At startup, the CUDA adapter validates the model manifest, placement, CUDA artifact, B70 artifact, and provider capability response as one compatibility set. Unknown, absent, or mismatched protocol, architecture, dimension, top-k, dtype, quantization, group size, scale format, layout, kernel, capacity, generation, or fingerprint is a hard incompatibility.

Unsupported models remain on stock upstream vLLM CUDA. They MUST NOT enter a generic B70 kernel. A qualified hybrid placement MUST NOT start if either normal-path owner or its exact artifact is missing. Kernel policy is selected only from the negotiated provider capability table.

## Deadline, recovery, cancellation, and restart

Every issue has a bounded deadline and every queue has fixed capacity. Backpressure, an expired deadline, an unhealthy provider, invalid generation, device loss, kernel error, or cancellation closes the remote join lane.

For a provably partial completion, `route_status` partitions successful and failed canonical remote route positions. CUDA may consume the weighted partial for exactly the successful subset and recompute exactly the failed routes. If membership is ambiguous, it discards the whole partial and recovers the whole remote subset.

Exact recovery preserves original token row, global expert ID, route position, activation, and weight. It runs on an available CUDA correctness path or, as an emergency only, the exact CPU path. If neither is viable, the request fails explicitly. A late response after recovery is drained only for resource safety and MUST NOT reopen the lane.

On provider restart:

1. stop admission to the old generation;
2. close unresolved lanes for exact recovery or explicit failure;
3. create a new provider generation;
4. reconstruct and validate rings, fixed XPU buffers, streams, resident banks, weight generation, and placement;
5. admit new work only after capability and health validation; and
6. reject all old-generation publications and completions.

## Proven reference and production gap

The Colibri native worker has proven:

- persistent B70 weights and compact `(layer, expert) -> slot` ownership;
- exact signed-S4 GS64 conversion into its native K-major/marlin-derived layout with FP16 scales;
- FP16 activation/weight staging;
- routing-weighted accumulation into one hidden-size partial;
- asynchronous issue/take;
- exact failed-route recovery; and
- numerical agreement with its CPU reference.

Separately, qualified QuixiCore-XPU kernels have proven NVFP4 fused gate/up/SiLU/down execution on B70.

That Colibri worker remains a correctness/latency comparator. Its own logical transaction is single-token, one-pending-operation, fixed-scratch execution. Separately, Phase 1 qualified QuixiCore-XPU split/fused execution, batched shapes through `M=128`, and fixed provider buffers; Phase 2 qualified the multi-slot process ring, per-route/per-token status, and isolated real-B70 data plane; and Phase 3 qualified the primary NVFP4 weighted wire partial before CUDA scatter. Phase 3 used compact resident order `255,0,7,63,127,191,254,1` and proved canonical-global-to-local remapping, `M=1..128`, all-remote and mixed sparse/interleaved ownership, duplicate/non-sorted/boundary IDs, a $2^{-12}$ route weight, local-route invariance, zero-remote no-publication, exact identity/status/allocation accounting, and sequence-bound split/fused failure publication. It does not prove scheduler-step aggregation inside upstream vLLM, the CUDA scatter/join, adapter parity, or end-to-end behavior.

### Completed Phase-3 wire gate

`phase3/generate_reference.py` authenticates the 14,495,580,220-byte packed bank by SHA-256, authenticates the frozen NVFP4 shard manifest, byte-compares sampled expert records with the NVFP4 artifact, independently computes float64 BF16-source and decoded-NVFP4 expert outputs, and freezes and validates `phase3/reference_fixture.bin`. `phase3/provider_math_test` then passed against the physical B70 through the standalone process and protocol-v2 ring, including trusted placement/weight bootstrap negatives, unsupported `M=0/129`, and failure completions that expose no partial payload.

This is evidence for the compact `[M_remote, hidden]` wire partial and its process semantics only. CUDA has not yet scattered those rows into `[M, hidden]`, joined them with local CUDA experts, or proved the upstream-vLLM layer seam.

## Hard gates

The fabric is eligible only after:

1. capability mismatches fail before execution;
2. ring stress covers concurrent slots, wraparound, stale replies, invalid generations, provider restart, cancellation, timeout, and injected kernel failure;
3. no hot-path allocation, weight upload, global device synchronization, unbounded queue, or unsafe buffer reuse occurs;
4. the completed Phase-3 provider gate remains reproducible for the weighted sum of exactly its B70-owned routes across `M=1..128`, all-remote and mixed ownership, compact remapping, duplicates, non-sorted and boundary IDs, and near-zero weights;
5. the completed Phase-3 ring cases continue to preserve zero-remote no-publication, local-route invariance, exact status/identity/allocation accounting, and explicit split/fused failure publication;
6. the CUDA+B70 partial joins before final TP/EP reduction and matches the all-CUDA layer oracle within the declared artifact/precision tolerance; and
7. every failure produces exact route recovery or explicit request failure, with no normal-path CPU matrix work outside the declared all-out opt-in (see [`architecture.md`](architecture.md)).
