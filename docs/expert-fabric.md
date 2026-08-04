# Expert Fabric

## Purpose and status

This document specifies the provider-neutral expert execution and cross-vendor transport boundary for Shooting Brake. It elaborates the per-layer execution flow in [architecture.md](architecture.md#per-layer-execution); it does not replace the system architecture.

**Status:** approved design specification. The interfaces, state machine, and gates below are requirements and design targets, not claims that the code, hardware path, process isolation, or latency targets have been implemented or measured.

The words **MUST**, **MUST NOT**, **SHOULD**, and **MAY** are normative.

Related design documents:

- [placement.md](placement.md) defines which provider owns each expert at a placement epoch.
- [scheduling.md](scheduling.md) defines admission, deadlines, and fallback policy above this ABI.
- [correctness.md](correctness.md) defines route and numerical correctness gates.
- [model-format.md](model-format.md) defines source weights and provider-private prepacking.
- [benchmarking.md](benchmarking.md) defines the measurements that decide whether the remote path is eligible.
- [hardware.md](hardware.md) and [memory.md](memory.md) define topology and buffer placement.
- [risk-register.md](risk-register.md) tracks transport, runtime, and recovery risks.

## Evidence and claim boundaries

The source design establishes the following architecture contract:

- The state owner computes canonical top-k routes.
- The scheduler sends each destination one activation per original token, plus that destination's local expert IDs and canonical routing weights.
- Each destination returns one combined weighted partial per original token.
- The state owner joins provider partials in a deterministic order.
- For hidden size 6144 with BF16 transport, one activation or result vector is 12 KiB.

These are design facts about the approved architecture, not observations of a running implementation.

The source also records upstream or repository claims that are not independently established here:

- Intel Level Zero guidance recommends immediate command lists for low-latency work, and newer Arc B-series adapters standardize that path.
- An existing Vulkan precedent is described as paying approximately 0.8 ms of synchronous submission overhead per sparse layer.

The following remain assumptions until the prescribed measurements pass:

- Direct NVIDIA-to-Intel peer memory access is not a dependable production baseline. Physical PCIe reachability alone does not establish compatible driver memory handles, synchronization, or correctness.
- Host-pinned staging is therefore the baseline, but it is not assumed to beat exact CPU fallback for every batch class.
- Same-process execution is the first latency proof. Separate worker processes are a later operational choice, not proof of transport performance.

No direct interop path, B70 speedup, bounded tail latency, crash isolation, or full-model integration is claimed complete by this document.

## Boundary and module ownership

The expert fabric consists of a scheduler-facing coordinator, provider-neutral provider instances, and provider-private implementations:

```text
state owner / scheduler
        |
        | ExpertWork, ExpertCompletion, lifecycle operations
        v
expert_fabric
  |-- provider_cuda       existing CUDA-resident expert adapter
  |-- provider_xpu        SYCL / Level Zero B70 implementation
  |     `-- xpu_transport host-pinned rings and completion protocol
  `-- provider_cpu        exact DDR5 fallback adapter
```

The intended source boundaries are:

| Module | Responsibility | MUST NOT own |
|---|---|---|
| `expert_fabric` | Request planning, placement-epoch capture, fan-out, route accounting, deterministic join, and fallback decisions | CUDA, Level Zero, Vulkan, SYCL, or framework objects |
| `expert_provider` | Provider-neutral data types and lifecycle operations | Backend-specific handles or allocation policy |
| `provider_cuda` | Adapter around the existing CUDA-resident expert path | Cross-provider scheduling policy |
| `provider_xpu` | B70 remap, expert kernels, weighted gather, and provider-private command objects | Canonical routing or join policy |
| `xpu_transport` | Pinned rings, sequence and generation checks, release/acquire publication, deadlines, and completion-safe reuse | Model routing policy |
| `provider_cpu` | Exact DDR5 fallback for assigned routes | Approximate recovery unless a separate model mode explicitly permits it |
| `expert_placement` | Placement epochs, static ownership, live counters, coactivation, and replication | In-flight buffer ownership |
| `expert_manifest` | Model identity, ownership, packed offsets, and checksums | Runtime scheduling |
| `expert_telemetry` | Per-provider timing, queue, transfer, kernel, failure, and fallback observations | Admission or correctness decisions |

Provider-private code MAY use CUDA streams and events, Level Zero command lists and events, SYCL queues, Vulkan objects, or framework tensors internally. None of those objects may cross the `expert_provider` boundary. The scheduler ABI MUST contain only fixed-width scalars, provider-neutral status values, and views of preallocated storage.

Provider-specific weight packing is likewise private to the provider. `load_bank` consumes model-format metadata and prepares a provider bank; it does not expose packed backend descriptors to the scheduler.

## Provider-neutral data contract

The following is a semantic ABI. Concrete C declarations may add explicit ABI-size fields and reserved fields, but MUST preserve these meanings and MUST remain backend-neutral.

### Buffer views

```c
typedef struct {
    void    *data;          /* preallocated storage */
    uint64_t bytes;         /* accessible byte extent */
    uint32_t element_type;  /* provider-neutral enum */
    uint32_t memory_class;  /* host, host-pinned, or scheduler device view */
    uint64_t buffer_id;     /* stable identity of the allocation */
    uint64_t buffer_version;/* reuse/version identity for these contents */
} ExpertBufferView;
```

A buffer view is not an ownership transfer and is not a backend object. The provider MUST NOT retain or access a view outside the accepted work item's lifetime. A provider that needs backend-visible storage MUST bind or register it during initialization, or copy through its preallocated private arena; it MUST NOT allocate on the token path.

### `ExpertWork`

```c
typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;

    uint64_t request_id;
    uint64_t generation;
    uint64_t placement_epoch;
    uint64_t session_id;
    uint64_t deadline_ns;

    uint32_t layer_id;
    uint32_t token_count;                /* T */
    uint32_t hidden_size;                /* H */
    uint32_t max_local_routes_per_token; /* Kb */

    uint64_t sequence;

    ExpertBufferView hidden_bf16;       /* [T,H] */
    ExpertBufferView local_expert_ids;  /* [T,Kb], int32 */
    ExpertBufferView route_weights;     /* [T,Kb] */
    ExpertBufferView valid_route_mask;  /* [T,Kb] */
    ExpertBufferView partial_bf16;      /* caller-provided [T,H] output */
} ExpertWork;
```

Normative rules:

1. `request_id` identifies the logical layer request. It MUST NOT be silently reused within the identity domain in which a late completion could still arrive.
2. `generation` identifies the live provider/transport incarnation. Restart or destructive reset MUST create a new generation before new work is accepted.
3. `placement_epoch` is captured when the request plan is formed. The provider MUST execute only the expert mapping for that epoch.
4. `session_id`, `layer_id`, `token_count`, and `hidden_size` bind work to its original model position and tensor shape.
5. `sequence` is the monotonic publication identity used by the bounded ring. It prevents wraparound from making an old publication appear current.
6. Every buffer's `buffer_id` and `buffer_version` bind the request to a particular allocation incarnation and contents. Reuse MUST advance the version before a new publication.
7. `hidden_bf16` contains exactly one activation row for every original token, even when the provider owns multiple selected experts for that token.
8. `local_expert_ids`, `route_weights`, and `valid_route_mask` describe only the route subset assigned to this provider, while retaining original-token row identity.
9. Padded positions MUST be masked. Invalid local IDs and unsupported duplicate-ID cases MUST be rejected, not read speculatively.
10. `partial_bf16` is supplied before submission and has capacity for exactly `[T,H]`. Submission MUST NOT allocate an output buffer.
11. `deadline_ns` is an absolute scheduler deadline in the agreed monotonic-clock domain. Absence of a useful deadline is not permission to wait without bound.

The implementation MUST validate all dimensions and byte extents before publication. Integer overflow while deriving any extent is a hard request error.

### `ExpertCompletion`

```c
typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;

    uint64_t request_id;
    uint64_t generation;
    uint64_t placement_epoch;
    uint64_t session_id;
    uint64_t sequence;

    uint32_t layer_id;
    uint32_t token_count;
    uint32_t hidden_size;
    uint32_t status;

    uint64_t output_buffer_id;
    uint64_t output_buffer_version;
    ExpertBufferView partial_bf16; /* [T,H], same original-token order */

    uint64_t queue_ns;
    uint64_t input_copy_ns;
    uint64_t kernel_ns;
    uint64_t output_copy_ns;
    uint32_t error_code;
    uint32_t reserved;
} ExpertCompletion;
```

A successful completion MUST describe the caller-provided output buffer and MUST return exactly `token_count` rows in original-token order. Timing fields are observations when supported; missing timing capability MUST be represented explicitly rather than fabricated.

Before a completion can enter the join, the fabric MUST match all of:

- ABI version;
- request ID;
- generation;
- placement epoch;
- session ID;
- layer ID;
- token count and hidden size;
- sequence;
- output buffer ID and buffer version;
- expected provider and expected route subset;
- successful terminal status.

The revised request contract explicitly requires rejection on request ID, generation, placement epoch, layer, token count, and buffer version mismatch. The additional identity and shape checks above close the same stale-work class at the provider-neutral boundary. A mismatch is a stale or corrupt completion; its payload MUST NOT be joined.

## Lifecycle operations

The provider interface exposes these operations and no backend objects:

```text
init
load_bank
publish_placement
submit
poll_completion
cancel_if_not_started
health
telemetry
shutdown
```

### `init`

`init` probes and binds one provider instance to its configured device and NUMA locality. It MUST create bounded queues, register or allocate all hot-path buffers, create command objects and events, and establish a fresh generation before the provider becomes ready. Initialization failure leaves the provider ineligible for scheduling.

### `load_bank`

`load_bank` validates model identity, manifest checksums, expert ownership, offsets, and supported formats, then constructs provider-private weight arenas. Loading and prepacking are control-plane operations, never token-path work. A bank is not executable until its placement publication succeeds.

### `publish_placement`

`publish_placement` atomically makes a fully prepared mapping available for a new `placement_epoch`. It MUST NOT mutate the expert mapping observed by already accepted work. The provider MUST retain any old bank state needed by in-flight work until those references reach a terminal completion or are safely invalidated by a generation reset.

### `submit`

`submit` is nonblocking with respect to queue capacity and device completion. It MUST validate the work contract and return one of a bounded set of outcomes such as accepted, backpressure, deadline already expired, unhealthy, stale epoch, or invalid request.

An accepted result means the provider owns the work until a terminal completion, a confirmed not-started cancellation, or generation invalidation. It does not mean the kernels completed. A full queue MUST return backpressure immediately; it MUST NOT grow and MUST NOT wait without a bound.

### `poll_completion`

`poll_completion` returns up to caller-supplied bounded capacity and MUST support a nonblocking form. Polling MAY use a short, configured foreground busy-poll interval only when measurements justify it. Otherwise it SHOULD sleep using event/futex-style notification. It MUST NOT consume an entire CPU core indefinitely and MUST NOT perform a global device synchronization.

### `cancel_if_not_started`

Cancellation is an ownership operation, not a promise to preempt a running accelerator kernel:

- If the work has not been claimed, `cancel_if_not_started` MAY remove it and MUST publish a terminal cancelled completion.
- If the work has been claimed or started, the operation returns `too_late`; the provider may finish it, but the scheduler may already have closed that join lane and used fallback.
- A late success after deadline or fallback MUST fail join-lane identity/state validation and MUST never be added.
- Cancellation MUST NOT permit either side to reuse storage while a runtime can still access it.

### `health` and `telemetry`

`health` reports generation, readiness, queue saturation, placement epoch, and terminal fault state using provider-neutral values. `telemetry` reports bounded counters and timing observations. Neither call exposes backend events or allows telemetry collection to block the token path.

### `shutdown`

`shutdown` stops admission first, then performs a bounded drain or invalidates the generation and cancels outstanding work according to policy. It MUST release or quarantine every slot only after backend access is complete. A hung worker cannot make shutdown wait forever; the state owner must retain exact fallback and restart paths.

## B70 partial operator

The B70 provider implements this semantic operation:

```text
b70_expert_partial(
    hidden[T,H],
    local_ids[T,Kb],
    weights[T,Kb],
    valid_mask[T,Kb]
) -> partial[T,H]
```

For each original token `t`:

```text
partial[t, :] = sum over j where valid_mask[t,j]
                route_weights[t,j]
                * Expert(local_expert_ids[t,j], hidden[t,:])
```

The provider-internal flow is:

```text
remap -> gate/up GEMM -> SwiGLU -> down GEMM -> weighted inverse gather
```

The semantic requirements are more important than the initial kernel composition:

- The B70 returns **one and only one weighted partial row per original token**, not one result per local expert.
- A token with no valid B70 route returns the additive-identity row for the B70 subset.
- Multiple local experts reuse the token activation; they do not cause duplicate host transfers of that activation.
- Weights are the canonical router weights supplied by the state owner. The provider MUST NOT renormalize its local subset.
- Masked/padded routes contribute nothing.
- Invalid local IDs are errors. Duplicate IDs require an explicitly tested model rule; otherwise they are rejected.
- Expert-parallel local-ID remapping MUST preserve the manifest and placement-epoch mapping.
- A zero route weight contributes exactly the additive identity under the defined arithmetic.

The correctness gate compares the returned partial with the weighted sum of the same B70 route subset. Required cases include no B70 routes, one route, multiple routes, duplicates, invalid IDs, masked routes, expert-parallel mapping, zero weights, and placement-epoch changes.

## Fan-out, route ownership, and deterministic join

For every sparse layer, `expert_fabric` constructs an immutable request plan from the canonical routes and captured placement epoch.

The plan MUST contain, for every selected route:

- original token row;
- canonical route position or another stable route identity;
- global expert identity;
- canonical route weight;
- exactly one current execution owner;
- the provider-local expert identity for that epoch;
- a join lane and terminal state.

Fan-out groups routes by destination provider. Each destination receives all `T` original-token activations once, even if only some token rows have valid routes there. The local route mask selects the provider's work. A provider returns a single `[T,H]` partial.

The join MUST:

1. accept at most one terminal contribution for each planned provider route subset;
2. reject identities, epochs, generations, versions, shapes, providers, or route subsets that do not match the immutable plan;
3. wait only until the request's bounded deadline or an earlier terminal policy decision;
4. close a lane before scheduling its exact fallback, so a late remote result cannot double-count;
5. add provider partials in one documented stable order, independent of completion order;
6. continue only after every selected route is accounted for by either its planned provider or exact fallback.

A suitable stable order is the immutable order assigned in the request plan, such as provider class and stable provider ID, followed by the separately defined local/CPU lane. The particular order is an implementation choice; changing it is a numerical behavior change and must not depend on race timing.

Deterministic join means identical accepted partials are accumulated in the same order. It does not by itself claim bitwise identity between different device kernels. Numerical tolerances and oracle comparisons are defined in [correctness.md](correctness.md).

### Exact CPU fallback

DDR5 fallback remains the exact recovery path for models whose exact fallback weights fit. On remote rejection, backpressure, missed deadline, cancellation, unhealthy state, or worker restart, the fabric MUST execute the missing canonical routes with their original expert IDs and original route weights. It MUST NOT substitute zeros, renormalize the surviving routes, approximate a missing expert, or accept both fallback and a late remote partial.

Fallback is route-accounting exact: every canonical selected route contributes once. Numerical comparison across different kernels follows the declared precision policy; the scheduler may not weaken route identity to hide a provider failure.

## Host-staged ring

### Topology

The baseline allocates inbound and outbound pinned storage per B70 on the CPU/NUMA node closest to that card. The ABI describes one logical transaction slot. An implementation MAY store request and response payloads in separate inbound and outbound rings, but the pair MUST share one immutable transaction identity and one ownership protocol.

Rings are fixed-capacity and preallocated. Double or triple buffering is permitted, but queue depth is bounded. There is no per-token allocation and no global mutex on the token path.

### Logical slot schema

Each logical slot contains cache-line-separated publication metadata and preallocated payload regions:

```text
publication header (cache-line aligned)
  abi version / header size
  atomic logical state
  sequence
  generation
  request ID
  placement epoch
  session ID
  layer ID
  token count / hidden size / Kb
  input buffer ID + version
  output buffer ID + version
  status / error code
  deadline
  timestamps

request payload
  hidden_bf16       [T,H]
  local_expert_ids  [T,Kb] int32
  route_weights     [T,Kb]
  valid_route_mask  [T,Kb]

response payload
  partial_bf16      [T,H]
```

Capacities are fixed at initialization. A request exceeding a slot's declared maxima is rejected before publication; the ring MUST NOT resize in response.

### Canonical ownership states

The canonical cross-runtime publication state machine is:

```text
FREE
  -> CUDA_WRITING
  -> REQUEST_READY
  -> XPU_RUNNING
  -> RESPONSE_READY
  -> CUDA_READING
  -> FREE
```

These are **logical ownership/publication states**, not a list of every DMA or kernel event:

| Logical state | Owner | Permitted transfer sub-events | Publication rule |
|---|---|---|---|
| `FREE` | Ring allocator | None | Payload is reusable only after all prior backend use is complete |
| `CUDA_WRITING` | NVIDIA/state-owner producer | NVIDIA D2H starts and completes; request metadata and payload are filled | Producer publishes `REQUEST_READY` only after D2H completion |
| `REQUEST_READY` | Published to B70 worker | Worker validates identity and claims the slot | Worker observes payload only after acquire |
| `XPU_RUNNING` | B70 worker | B70 H2D, remap/kernels, B70 D2H | Worker remains owner through output-transfer completion |
| `RESPONSE_READY` | Published to NVIDIA/state-owner consumer | Consumer validates completion and claims response | Worker publishes only after B70 D2H completion |
| `CUDA_READING` | NVIDIA/state-owner consumer | NVIDIA H2D, deterministic join read, terminal accounting | Consumer releases `FREE` only after its copy/read is complete and B70 has no references |

The older detailed transfer names are therefore sub-events of the canonical states:

```text
NVIDIA_D2H       is inside CUDA_WRITING
READY_FOR_B70    is REQUEST_READY
B70_H2D          is inside XPU_RUNNING
B70_RUNNING      is inside XPU_RUNNING
B70_D2H          is inside XPU_RUNNING
RESULT_READY     is RESPONSE_READY
NVIDIA_H2D       is inside CUDA_READING
CONSUMED         is the completed-consumer event before FREE
```

There are not two competing state machines. The canonical states define cross-thread/process ownership; backend events prove when the transfer sub-events permit the next publication.

### Release/acquire publication

Every transition that publishes ownership to another runtime or permits reuse MUST use atomic release/acquire semantics:

1. The current owner writes payload and immutable identity fields.
2. It waits on or queries the relevant backend completion event without a global synchronization.
3. It performs a release store or successful release compare-exchange of the next publication state.
4. The next owner observes that state with an acquire load or acquire compare-exchange before reading metadata or payload.

Specifically:

- `CUDA_WRITING -> REQUEST_READY` occurs only after CUDA D2H has completed.
- `REQUEST_READY -> XPU_RUNNING` is the worker's claim after validation.
- `XPU_RUNNING -> RESPONSE_READY` occurs only after the B70 output transfer to host has completed.
- `RESPONSE_READY -> CUDA_READING` is the state owner's claim after validation.
- `CUDA_READING -> FREE` occurs only after CUDA H2D and every consumer read are complete.

A volatile field, ordinary store, backend event without host publication, or timing delay is not a substitute for release/acquire ordering.

### Identity, wraparound, and reuse validation

A slot claim and a returned completion MUST validate the tuple:

```text
(abi_version,
 request_id,
 generation,
 placement_epoch,
 session_id,
 layer_id,
 token_count,
 hidden_size,
 sequence,
 input_buffer_id/version,
 output_buffer_id/version)
```

The response additionally matches the expected provider, route-subset identity, and status.

The roles of the counters are distinct:

- **Sequence** distinguishes successive publications as a bounded ring wraps.
- **Generation** distinguishes provider/worker incarnations across restart or destructive reset.
- **Placement epoch** distinguishes expert ownership and local-ID mappings.
- **Buffer version** distinguishes successive contents of the same preallocated allocation.
- **Request ID** distinguishes logical layer work.

No one counter substitutes for another. Counters MUST be wide enough and managed so that a live stale reference can never alias a current identity. If safe wrap cannot be proved, the provider must stop admission and create a fresh generation before reuse.

A mismatch never causes a best-effort join. The slot or completion is quarantined or discarded, the lane is resolved through exact fallback, and telemetry records the reason.

### Completion-safe reuse

`FREE` is a proof obligation, not merely an available bit. A slot may return to `FREE` only when:

- CUDA has finished any D2H that sourced the request;
- B70 has finished any H2D reads of the request;
- B70 kernels no longer reference request or response storage;
- B70 D2H has finished writing the response;
- CUDA has finished H2D reads of the response;
- the deterministic join has consumed the copied result or otherwise ended its reference;
- terminal accounting has made late completion ineligible for the request.

Timeout, cancellation, fallback, shutdown, or worker death does not waive these conditions. If completion cannot be proved after a runtime fault, the affected storage is quarantined until generation teardown safely destroys or unregisters it; it is not recycled into a later token.

## Deadlines, backpressure, cancellation, and restart

### Bounded waiting

Every work item has an explicit deadline, and every queue has fixed capacity. The scheduler and provider MUST have bounded waits for slot acquisition, completion polling, drain, and shutdown. No operation may turn a missed deadline into indefinite waiting.

A deadline is checked at least before admission, before starting remote work when practical, and at join closure. Starting work that cannot plausibly complete by the deadline SHOULD be avoided based on measured latency, but correctness never relies on prediction.

### Backpressure

When no slot or provider queue entry is available, `submit` returns backpressure. The scheduler may use another exact owner or CPU fallback according to policy. It MUST NOT allocate an overflow node, grow a queue, block until an arbitrary future completion, or hold a global mutex while waiting.

Foreground polling may be bounded and measured. Background work should sleep. Busy polling that consumes a whole CPU core requires evidence and an explicit configured bound.

### Cancellation and late completion

Once a join lane is closed for fallback, it never reopens for the old remote work. If not-started cancellation succeeds, the slot follows a terminal cancellation/release path. If cancellation is too late, the provider completes or is generation-invalidated, and its eventual completion is drained only for resource safety. It cannot contribute to the request.

### Worker restart

On B70 worker crash, hang, or restart:

1. stop admission to the old generation;
2. mark all its unresolved join lanes unavailable and schedule exact fallback within the request policy;
3. create a new generation for the replacement worker;
4. rebuild and validate provider-private command objects, events, weight banks, and placement publication;
5. accept new work only after health and epoch readiness pass;
6. reject every completion or ring publication carrying the old generation.

A worker restart does not imply that old shared buffers are safe to reuse. The state owner must either prove backend access ended or quarantine and reconstruct the affected ring.

## Same-process proof and later isolation

### First proof: same process

The first microbenchmark and integration use one process containing:

- the existing C runtime and CUDA backend;
- a SYCL/DPC++ or Level Zero Intel module behind a C ABI;
- the same provider-neutral `ExpertWork`, `ExpertCompletion`, and ring semantics specified here.

This deliberately removes IPC from the first latency proof. It does **not** isolate the state owner from an Intel runtime crash. The proof must first establish the empty host-staged round trip and actual B70 partial without per-request allocation, global synchronization, premature buffer reuse, or unbounded tail behavior.

### Optional production isolation

After the primitive passes, separate B70 worker processes may be selected based on measurement and operational needs. Benefits may include runtime-failure containment, explicit device and NUMA affinity, isolated weight arenas and command queues, and simpler restart handling.

Process isolation MUST preserve the same logical ring, identity validation, deadlines, and provider semantics. The transport remains shared pinned rings; JSON, protobuf, gRPC, Python object exchange, and ordinary pipes are not token-path transports. Process separation must not be baked into the first proof or used to obscure transport latency.

## Hard gates

The expert fabric is not eligible for model integration or scheduling merely because device-local GEMM is fast.

### Empty transport gate

Exercise one and multiple slots, backpressure, ring wraparound, stale generations, timeout, producer crash, B70 restart, simultaneous completions, buffer reuse, and shutdown with work in flight.

The gate is hard:

- no allocation in the hot loop;
- no global synchronization;
- no unbounded queue or wait;
- bounded p99 under the prescribed idle and contention cases;
- no polling that consumes an entire CPU core without evidence;
- no buffer reuse before both runtimes have completed with it;
- stale work never enters a later token.

### Semantic gate

A heterogeneous layer must account for every selected expert in canonical route identity and order across B70, CUDA, and CPU destinations. It must cover deadline fallback, stale completion, and worker reset, and agree numerically with the exact CPU or CUDA oracle under the declared tolerance.

For every B70 route subset:

```text
B70Partial ≈ sum over B70 routes e of w_e * Expert_e(x)
```

The response shape remains `[T,H]`: one weighted partial per original token.

### Eligibility gate

Measure the complete remote critical path:

```text
CUDA D2H
+ queue
+ B70 H2D
+ remap
+ gate/up GEMM
+ activation
+ down GEMM
+ weighted gather
+ B70 D2H
+ CUDA H2D
+ deterministic join
```

A B70 is eligible for a batch class only when its measured p99 remote time is less than exact CPU fallback p99 for that class. If it loses for batch-one decode but wins for larger batches, it may serve only the winning classes such as batched decode, prefill, concurrent sessions, or background requests. The scheduler MUST NOT force B70 into latency-critical batch-one decode on the strength of kernel-only measurements.
