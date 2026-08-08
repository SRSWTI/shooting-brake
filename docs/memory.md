# Memory Planes and Ownership

## Purpose and status

This document defines memory ownership, persistence, fixed-buffer boundaries, invalidation, and capacity accounting for the production design in [`architecture.md`](architecture.md) and [`../plan.md`](../plan.md). It is a design contract, not a claim that the upstream-vLLM plus B70 provider path has been implemented or that a target model has been admitted.

## Global rules

1. Every allocation has one owner and one address domain. RTX 5090 VRAM, B70 VRAM, pinned host memory, pageable host memory, and NVMe are never described as unified VRAM.
2. The upstream vLLM CUDA worker is the state owner. Active attention, KV/recurrent state, router/top-k, dense/shared paths, residual stream, local experts, sampling, and serving state stay on the RTX 5090.
3. The isolated B70 provider owns only B70-resident routed-expert weights, stable XPU buffers, streams, and qualified provider-kernel execution: primary QuixiCore-XPU NVFP4, or the secondary llm-scaler INT4 fallback.
4. Hot routed experts have immutable CUDA ownership; cold/overflow routed experts have immutable B70 ownership for the loaded placement generation. There is no foreground weight migration.
5. CPU memory and compute are orchestration and exact emergency recovery only. CPU matrix compute is absent from the normal path — with one declared exception, the opt-in host-DRAM cold expert tier ("all-out mode"), which requires `SHOOTING_BRAKE_ALL_OUT=1` together with an `allout:` placement and is unreachable from any default placement. It exists so a routed-expert bank larger than combined 5090+B70 VRAM can load at all. See the amendment in [`architecture.md`](architecture.md).
6. NVMe supplies validated artifacts at startup or recovery. Ordinary warmed inference never synchronously loads an expert from storage.
7. The normal path transports activation rows, canonical route metadata, and one weighted compact `[M_remote, hidden]` B70 wire partial plus its row map; CUDA scatters it into the full `[M, hidden]` batch. It never transports expert weights.
8. Capacity claims use actual allocated and reserved bytes, and capacity gain is reported together with throughput and latency cost against stock all-CUDA upstream vLLM.

## Plane 1: immutable model weights

| Domain | Resident contents | Owner and lifetime | Normal-path rule |
|---|---|---|---|
| RTX 5090 VRAM | embeddings, attention, KV/recurrent machinery, dense layers, router, shared expert, LM head, and hot routed experts | upstream vLLM CUDA worker; immutable for the loaded model/placement generation | stock-compatible CUDA execution; reserve space for KV, runtime scratch, copy/join buffers, and safety headroom |
| B70 VRAM | compact cold/overflow routed-expert bank in the provider-qualified NVFP4 layout | isolated B70 provider; loaded once and immutable for the weight/placement generation | execute only B70-owned selected routes; no router, shared expert, attention, or sampling |
| Host memory | manifest/configuration, control state, pinned ring, telemetry, and any explicitly admitted exact recovery representation | CPU orchestration/recovery owner | no normal-path matrix multiplication and no normal expert tier |
| NVMe | source checkpoint, CUDA artifact, B70 artifact, provider bundle, and recovery material | durable artifact store | startup/provider restart/recovery only |

CUDA and B70 physical quantization may differ, but both artifacts must be derived from the identical higher-precision source checkpoint. The model/provider manifest records artifact fingerprints, tensor dimensions, top-k, dtype, group size, layout, kernel capability, protocol version, placement fingerprint, and weight generation. Startup fails closed on disagreement.

A model or weight-generation change invalidates both device banks, placement maps, recovery representations, and in-flight work. A placement-generation change invalidates routing to the old compact slots. No late completion may cross either boundary.

### Capacity admission

Budget actual allocatable memory separately:

```text
RTX 5090:
  state-owner tensors
+ CUDA-owned hot experts
+ KV/prefix/recurrent state reservation
+ local MoE/shared-expert scratch
+ preallocated remote-partial and join buffers
+ graph/runtime/safety headroom

B70:
  compact cold/overflow expert bank
+ grow-only activation/route/output tensors
+ kernel scratch and streams
+ provider/runtime/safety headroom

Host:
  fixed pinned request/completion ring
+ exact emergency-recovery resources, if configured
+ provider/CUDA control state and telemetry
+ operating-system headroom
```

An admitted placement satisfies measured packed-byte budgets in both device domains. Nominal board capacity or parameter arithmetic is insufficient. If every selected routed expert does not have exactly one CUDA or B70 normal-path owner, the placement is incomplete and serving must fail closed rather than introduce a CPU or NVMe normal tier.

Measured capacity gain is the additional model-weight and/or 5090 KV budget made usable by moving qualified cold/overflow expert bytes to B70, after all B70 provider buffers and safety reservations. Report the actual resident bytes in each domain and the corresponding context/concurrency envelope. Do not claim the sum of board capacities as a gain.

## Plane 2: fixed transport and provider buffers

The CUDA adapter and B70 provider communicate through a versioned fixed pinned-memory ring with multiple bounded slots. Associated buffers are allocated and negotiated at startup:

```text
activation       [capacity_tokens, hidden]
expert_ids       [capacity_tokens, topk]
routing_weights  [capacity_tokens, topk]
route_mask       [capacity_tokens]
token/route map  [capacity_routes]
output           [capacity_tokens, hidden]
status           [capacity_routes]
```

Each request carries at least request sequence, ring slot, layer, full scheduler-row count `M`, staged-row count `M_remote`, route count, placement generation, weight/provider generation, activation/output dtype, and status. Only token rows containing at least one B70-owned route are copied; `token_row_map[M_remote]` preserves each original row. The provider returns one already-weighted, B70-route-reduced `[M_remote, hidden]` wire partial. CUDA scatters it into a preallocated zero-initialized `[M, hidden]` buffer, so unstaged tokens have zero B70 contribution.

The provider mirrors these capacities with grow-only, stable-address XPU tensors for activation, routes, maps, scratch, and output. No per-forward tensor, scratch, event, descriptor, or weight allocation is allowed. A layer/step with zero remote routes consumes no B70 slot and performs no B70 submission.

A slot is reusable only after both runtimes have completed their access. Sequence, layer, slot, placement generation, weight/provider generation, token count, and buffer version must match before CUDA accepts the output. Publication uses release/acquire ordering; completion publishes only after the B70 output copy is visible to the CUDA process.

## Plane 3: active inference state

The RTX 5090 exclusively owns active:

- KV and prefix cache;
- DeltaNet/GDN or other recurrent state;
- sequence, request, cancellation, and scheduler state;
- residual activations and CUDA graph state; and
- sampler state.

B70 requests contain only the activation rows and route metadata required for a single active MoE layer. They confer no sequence, context, or serving ownership. B70 VRAM does not expand the CUDA KV pool. Active KV exhaustion is handled by vLLM admission/eviction policy, not by spilling active state to B70.

Host or NVMe persistence for inactive application context is outside the per-layer expert path. Restoring inactive state must re-establish CUDA authority before the request becomes active.

## Plane 4: placement, capability, and lifecycle state

The control plane persists:

- immutable `(layer, global expert) -> CUDA local slot | B70 compact slot` ownership;
- placement and weight generations plus artifact fingerprints;
- provider protocol and primary/secondary kernel-bundle versions;
- supported dimensions, dtypes, top-k values, group sizes, layouts, maximum tokens/routes, slots, and streams;
- route frequency and measured service/queue distributions used to choose the hot CUDA set;
- ring capacity and address-stability guarantees; and
- provider health, failure, restart, and recovery counters.

This metadata can choose an owner before serving starts; it cannot alter canonical top-k or silently select a different expert. The initial production placement is immutable for the loaded run. A replacement map is admitted only after all new weights are loaded and validated, then published as a new generation with no in-flight request referring to mutated slots.

## Plane 5: exact emergency recovery

CPU and available CUDA capacity form a correctness path, not a throughput tier. On B70 timeout, device loss, invalid generation, or kernel failure, the system identifies the exact failed token-route entries and either:

- recomputes only those routes using an admitted exact CUDA/CPU representation; or
- fails the request explicitly.

Recovery cannot change selected expert IDs, routing weights, source model, or reduction semantics. A late B70 completion after recovery is discarded. Normal operation reports zero CPU expert matrix work; any nonzero recovery count is explicit failure telemetry and is not included as ordinary hybrid service.

NVMe may restore a provider after failure, but a foreground request must not wait on a synchronous expert load disguised as recovery. If exact immediate recovery resources are unavailable, fail explicitly.

## Colibri reference scope

The proven Colibri GS64 native path demonstrates persistent B70 weights, compact expert slots, FP16 staging, a routing-weighted hidden-size partial, asynchronous issue/take, exact failure recomputation, and zero normal-path CPU expert fallback. Those facts justify the ownership and lifecycle shape above.

They do not prove the production batched buffers, fixed multi-slot process ring, QuixiCore-XPU tiny/grouped/prefill NVFP4 MoE kernels, the secondary llm-scaler INT4 fallback, upstream-vLLM memory behavior, production graph stability, or capacity gain. Those are measured under [`benchmarking.md`](benchmarking.md).

## Cross-plane lifecycle

### Startup

1. Record exact upstream-vLLM and provider dependency versions.
2. Validate the model, CUDA artifact, B70 artifact, placement map, and provider capability manifest against the same source checkpoint.
3. Reserve the RTX 5090 state, KV, scratch, join, graph, and safety budgets.
4. Load only CUDA-owned hot experts on the 5090.
5. Start the isolated provider, select the B70 explicitly, reserve stable XPU buffers, and load only B70-owned cold/overflow experts.
6. Allocate and initialize the fixed versioned pinned ring.
7. Negotiate capacity, generations, dimensions, dtypes, top-k, group size, layout, slots, and streams; fail closed on mismatch.
8. Admit exact emergency recovery only if its source and resource limits are explicit.
9. Enter serving with NVMe outside the warmed foreground path.

### Shutdown or provider restart

Stop publication, bound or cancel in-flight work, invalidate the provider generation, and reject every completion from the old generation. Reconstruct device weights only from validated artifacts, renegotiate capabilities and ring generations, and resume only after the immutable ownership map is complete.

## Required reporting

Every capacity/performance result separately reports:

- actual allocated and reserved RTX 5090 bytes by state-owner, hot-expert, KV/recurrent, scratch/join, graph/runtime, and headroom categories;
- actual allocated and reserved B70 bytes by cold/overflow weights, stable tensors/scratch, runtime, and headroom;
- pinned-ring bytes, slot count, negotiated token/route capacities, and host NUMA placement;
- other host bytes used for control, telemetry, and exact emergency recovery;
- NVMe reads and bytes by startup, restart, or recovery, with zero ordinary warmed-decode reads;
- CUDA/B70 ownership counts and placement/weight/provider generations;
- CPU recovery count and bytes/computation, expected to be zero on the normal path;
- 5090 KV/context/concurrency budget; and
- measured capacity gain plus throughput/TTFT/ITL cost relative to the same stock all-CUDA upstream-vLLM workload.

A report is misleading if it calls aggregate device memory unified VRAM, treats B70 memory as CUDA KV capacity, treats CPU/DDR5 or NVMe as a normal expert tier, hides provider buffers/headroom, or reports capacity without the paired service cost.
