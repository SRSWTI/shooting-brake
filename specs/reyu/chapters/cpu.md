# CPU cold tier

The CPU branch is the optional third routed-expert tier below CUDA and B70. Read it breadth first: it owns a packed host arena and one shared worker pool, can return a routing-weighted CPU partial through a native poller, and can lend the same packed expert bytes to a CUDA prefill streamer. It does **not** own model admission, expert placement, CUDA/B70 aggregation, or the vLLM scheduler. Those callers must route only CPU-owned experts here and must add the returned partial to the other tiers.

The executable abstraction is [`cpu_tier.ryu`](../models/cpu_tier.ryu). It first exposes the whole branch—residency, host versus streamed execution, completion, and destruction—and then splits the host algorithm into bucket/gather/FFN/scatter stages. Tensor arithmetic is summarized as result validity and an exact-zero observation; the source harnesses remain the numerical comparison surface.

## 1. Responsibility, actors, and interfaces

```text
placement / routed layer
        |
        | CPU global IDs, bf16 activations, fp32 route weights
        v
CpuExpertHost Python binding -------------------------------+
        |                                                    |
        | ctypes                                             | borrowed packed block
        v                                                    v
native packed arena + persistent workers             ExpertStreamer on CUDA
        |                                                    |
        | direct call or CpuPoller signal                    | copy-event ring
        v                                                    v
fp32 routing-weighted host partial                    CUDA dtype partial
        \___________________________  _______________________/
                                    \/
                        caller adds tier partials
```

Actors and ownership are deliberately asymmetric:

* [`cpu_expert_capi.cpp`](../../../src/phase7/cpu_expert_capi.cpp) owns the virtual arena, resident-slot table, packed decode, reusable scratch, route buckets, persistent worker threads, and the native polling thread.
* [`cpu_expert_capi.h`](../../../src/phase7/cpu_expert_capi.h) freezes the C ABI and states the pointer and destruction contracts.
* [`cpu_expert_host.py`](../../../src/phase4/src/shooting_brake_vllm/cpu_expert_host.py) validates tensors before crossing ctypes, owns one native host handle, wraps borrowed arena memory as a tensor, optionally CUDA-registers the committed prefix, and wraps the poller handle. Process-wide helpers choose one host and one poller because duplicates would duplicate weights or compete for DRAM.
* [`cpu_stream.py`](../../../src/phase4/src/shooting_brake_vllm/cpu_stream.py) is an alternative execution location for large CPU-owned batches: weights stay logically owned by the CPU tier but packed bytes move through a small CUDA ring and the 5090 performs dequantization and FFN compute.
* The layer module outside this ownership set owns mapped signal/completion flags and all pinned activation, ID, weight, and output tensors registered with the poller. Raw native pointers remain valid only while those objects live.
* The active graph transport is the CUDA Driver stream-memory API in `stream_signal.py`. [`sb_stream.cu`](../../../src/phase4/src/shooting_brake_vllm/sb_stream.cu) is an obsolete standalone kernel-spinning helper with no phase4 caller or build rule; it is not silently substituted for the active path.

### Public contracts

| Interface | Inputs and outputs | Source-backed contract |
|---|---|---|
| `sb_cpu_create` / `CpuExpertHost.__init__` | model bounds, geometry, maximum resident count, worker count | Reserves a 2 MiB-rounded anonymous mapping, advises transparent huge pages, builds a dense `(layer, expert) -> slot` table initialized nonresident, and creates the persistent pool. A zero native thread count selects roughly one quarter of hardware concurrency, or four when unknown. |
| `load_expert` | three packed NVFP4 planes and three global scales | Python requires CPU `uint8` nibble and linear scale arrays with exact shapes; native code copies `gate_q|gate_sf|up_q|up_sf|down_q|down_sf` into one 64-byte-aligned extent. Reload overwrites the slot without growing the arena. |
| `expert_forward` | one resident identity and bf16 rows | Overwrites an fp32 output with one packed SwiGLU FFN. A nonresident identity errors. |
| `moe_forward` | bf16 `[M,H]`, int32 `[M,K]` IDs, fp32 `[M,K]` weights | Zeroes fp32 output, ignores negative/out-of-range IDs, counts in-range nonresident IDs, buckets resident routes, and scatter-adds each weighted contribution. `M==0` or `topk==0` succeeds with zeros natively. |
| `expert_block` / `expert_gscales` | resident identity | Returns a borrowed flat alias to the six packed byte sections plus the three scales out of band. The alias is read-only by convention and lasts only until host destruction. |
| `pin_arena` | committed arena prefix | Registers page-rounded `[base, used)` after loading. Failure logs and preserves a staged pageable-copy fallback; CUDA error 712 is treated as already registered. |
| `CpuExpertPoller` | layer, raw flags, pinned arrays | Appends layer registrations, snapshots them safely while running, interprets signal value as `M`, clears it before work, executes `moe_forward`, and always publishes completion. |
| `ExpertStreamer.forward` | CUDA activation, CPU route IDs, route weights | Sorts active routes by expert, streams each needed packed block once, dequantizes on CUDA, computes SwiGLU, and `index_add_` scatter-accumulates fp32 before casting to the activation dtype. An empty route set returns exact zeros without a transfer. |

## 2. State and operation map

The Reyu state record groups the arena, current routed operation, poller, borrowed block, ring slot, flags, and lifetime fault. Its small expert domain is an identity abstraction, not a production capacity.

| Model state/action | Implementation semantics | Atomicity choice |
|---|---|---|
| `createHost`, `loadExpert`, `reloadExpert`, `finishLoading` | `HugeArena`, `CpuExpertHost::load_expert`, Python shape guards | Each public load call is atomic. Its six `memcpy`s are not modeled as externally observable because the source provides no concurrent load/compute contract. |
| `routeBatch(routes, residentRoutes)` → `bucketRoutes` | native histogram/residency scan and zero-output initialization | Parameters are explicitly bounded by the finite model; scenario fixtures choose empty, resident-only, and mixed values. `bucketRoutes` derives skipped count and either releases exact zeros or proceeds to compute. |
| `gatherExpertRows` → `computeHostFfn` → `scatterWeightedRows` | `CpuExpertHost::moe_forward` lines 553–627 | Split to expose the actual module algorithm and the ordering that avoids rereading each expert per token. |
| `borrowPackedBlock(expert)`, `pinArena` | `sb_cpu_expert_block`, `expert_gscales`, `CpuExpertHost.pin_arena` | Borrow validity depends on the guarded resident identity; registration success/failure remains separate and a pin failure does not invalidate the block. |
| `issuePackedCopy(routes)` → `computeStreamedExpert` → `scatterStreamedRows` | `ExpertStreamer.forward` ring protocol | Bounded route count is supplied by scenarios. Copy issue requires the previous compute event; compute waits for copy; scatter records the event that permits reuse. |
| `createPoller`, `registerLayer(layer)`, `startPoller`, `signalBatch(M)` | poller handle and graph-compatible issue side | Registration derives validity from bounded layer identity, captures borrowed pointers, and signal value is a guarded batch size. Active signaling is Driver stream-memory operations, not the obsolete CUDA helper. |
| `pollDispatch` | `CpuPoller::loop` | Native `moe_forward` is atomic at this transport layer. Success/error derives from registered layer validity; both clear signal and publish completion, while error count distinguishes failure. |
| `stopPoller`, `destroyPoller`, `releaseBorrow`, `closeHostSafely` | caller-required shutdown order | Safe ownership is explicit rather than inferred from destructors. |
| `closeHostUnsafely` | ABI-permitted misuse | The implementation does not enforce host-outlives-poller or host-outlives-borrow. The model makes the resulting invalidation reachable as a lifetime violation. |

## 3. Packed arena and numerical mechanics

### Reservation and residency

`HugeArena` rounds the reservation to 2 MiB and uses `MAP_PRIVATE|MAP_ANONYMOUS`; `MADV_HUGEPAGE` is advisory. Physical pages are committed by first touch. Its bump allocator aligns extents to 64 bytes and never frees individual experts. One expert consumes

$$3\left(\frac{HI}{2}+\frac{HI}{16}\right)$$

bytes for three weight planes, where $H$ is hidden width and $I$ intermediate width. The slot vector is dense over model identity but stores `-1` for nonresident pairs. `experts_` stores pointers into the arena plus gate/up/down global scales. Reload finds the existing slot, replaces bytes and scales in place, and leaves `arena_used` unchanged.

Weights retain checkpoint-native NVFP4 representation. Each byte contains low-nibble-first E2M1 values; every 16 weights share a raw float8-E4M3FN scale; the reconstructed value is `e2m1 * e4m3 * global_scale`. Linear scale order is required because the CPU and streamed dequantizers do not consume the checkpoint kernel's swizzled form. E4M3FN NaN remains NaN rather than being clamped.

### Persistent worker synchronization

`ThreadPool` counts its caller as worker 0 and starts only `n-1` threads. A `parallel_for` publishes the function, total and a new generation; workers spin up to 4096 pause hints before sleeping on the condition variable. The caller executes chunk 0. Spawned workers decrement `pending_`, and the caller spins until it reaches zero before clearing the borrowed function pointer. Destruction flips `running_`, notifies, and joins every spawned worker. This is a reusable synchronous barrier, not a work-stealing scheduler.

Each projection partitions output rows. A row is decoded once into thread-local fp32 scratch, then reused across all token rows; that ordering avoids multiplying NVFP4 decode work by `M`. Scratch up to 4096 weights lives on the stack, with a vector fallback for larger widths. Gate and up consume bf16 activations and accumulate fp32; gate storage is replaced in place by `SiLU(gate) * up`; down consumes fp32 and produces fp32. AVX2 uses a 16-entry signed magnitude shuffle and preserves low/high nibble ordering; non-AVX2 uses scalar decoding.

### Routed gather/compute/scatter

`moe_forward` first zeroes all `M*hidden` output elements. Negative IDs are deliberate absent routes. Out-of-range positive IDs are also ignored. An in-range but nonresident ID increments the monotonic skipped counter and is excluded; this is a recorded correctness fault, not a returned error.

For resident routes it:

1. histograms expert IDs and computes prefix offsets;
2. computes the largest bucket, including repeated appearances of the same expert within one token's top-k;
3. fills expert-major token-row and weight arrays;
4. gathers source rows into dense bf16 scratch;
5. calls the packed FFN once per active expert, streaming that expert's bytes once for all its rows; and
6. scatter-adds `weight * expert_output` to the original token row.

There is no internal concurrent-call protection around shared scratch or bucket vectors. The live design serializes use through one shared poller/arena; callers must not infer a general reentrant host API.

## 4. CUDA streaming path and borrowed lifetime

The streaming policy threshold defaults to 128 tokens through `SHOOTING_BRAKE_CPU_STREAM_T`. This is a policy boundary, not an isolated crossover claim: comments explain that CPU decode can overlap CUDA work while streaming occupies the CUDA critical path. One process-wide streamer is shared across sequential model layers.

`ExpertStreamer` allocates a ring of packed `uint8` device slots and a dedicated copy stream. Each slot has `copy_done` and `compute_done`; initial compute events remove the first-use special case. `_group_routes` flattens IDs, drops negatives, sorts by expert on device, and synchronizes once to materialize the unique expert list and bucket boundaries. For each expert:

1. the copy stream waits for that slot's previous compute event;
2. one borrowed contiguous arena block is copied nonblocking;
3. the current compute stream waits for `copy_done`;
4. the six packed sections are sliced, and device-cached global-scale tensors feed vLLM's fused `dequantize_to_dtype(..., block_size=16, swizzle=False)`;
5. selected activation rows execute gate/up SiLU and down matmuls;
6. fp32 weighted rows are accumulated with `index_add_`; and
7. `compute_done` makes the slot reusable before a later ring tenant arrives.

`get_streamer` pins only after all loading is expected complete, then caches borrowed blocks and global-scale tensors by `(layer, expert)`. The block tensor aliases mmap storage. Host destruction invalidates those cached tensors immediately; neither Python nor C++ has a reference count that delays `munmap`.

Pinning is fail-soft, but its teardown is incomplete in the checked-in binding: `CpuExpertHost.close` calls `sb_cpu_destroy` without a matching `cudaHostUnregister`, and `_pinned_bytes` is only recorded. The source therefore does not itself establish a safe pinned-arena unregistration sequence. This is a source-visible lifecycle gap, not repaired in the model.

## 5. Poller, signaling, and shutdown

The native poller may accept appended layer registrations while running. It publishes a new count atomically and rebuilds a local snapshot under the registration mutex only when the count changes. For each entry, zero means idle and a positive signal value is exactly the active row count `M`. The poller clears signal before dispatch so a later replay may signal again, performs a sequentially consistent fence, invokes the shared host's `moe_forward`, records service time and error count, fences again, and writes completion `1`.

Completion is a release event, **not** a success certificate. `cuStreamWaitValue32` has no timeout, so even an out-of-range layer error must release CUDA. Consumers must inspect `error_count`; the completion flag alone can expose an untrustworthy output. When idle, the poller spins for 511 iterations and then sleeps 20 microseconds so it does not permanently take a core away from the FFN workers.

The required lifetime order is:

1. stop issuing stream writes and make the current operation quiescent;
2. stop and destroy the poller while its host and registered flags/buffers still exist;
3. stop using cached `expert_block` aliases and the streamer ring;
4. unregister pinned arena memory where the consumer provides that missing operation; and
5. destroy the host, which joins the worker pool and unmaps the arena.

The C ABI documents host-outlives-poller and buffer-outlives-poller but cannot enforce them. `CpuExpertHost.close` likewise does not find or stop the singleton poller. The model's unsafe-close witness intentionally demonstrates that raw pointers can outlive their backing allocation if caller ownership is wrong.

### Obsolete CUDA helper distinction

`sb_stream.cu` launches a one-thread kernel on the current/default launch context to write plus `__threadfence_system`, or to spin on a volatile flag until exact equality. Its wrappers only print launch errors and return no status; it takes no explicit stream argument and its wait has no timeout. Repository source inspection found no phase4 caller/build rule. The active plugin path uses CUDA Driver `cuStreamWriteValue32_v2` and `cuStreamWaitValue32_v2` from `stream_signal.py`, including graph capture. The Reyu signal mechanism is fixed to that active path, and attempting to select `ObsoleteKernelHelper` is an intentionally disabled action.

## 6. Executable scenarios and properties

The model provides reachable observations rather than temporal proof:

| Scenario | Observable contract |
|---|---|
| `packedArenaReloadTest` | Both finite experts are resident; reloading one does not grow committed arena usage. |
| `routedGatherComputeScatterTest` | A resident batch traverses route bucketing, gather, packed FFN and weighted scatter and releases a valid nonzero partial. |
| `emptyRoutesReturnZeroTest` | No selected routes returns an exact valid zero partial. |
| `nonresidentRouteIsCountedTest` | Resident work remains valid while the omitted nonresident route increments the skipped counter. |
| `cudaStreamingRingLifetimeTest` | A borrowed block can use the fail-soft unpinned path; copy/compute event ordering restores slot reuse and releases a valid result. |
| `pollerCompletionSuccessTest` | Signal-as-batch is cleared and valid output releases the CUDA waiter. |
| `pollerFailureStillReleasesTest` | Dispatch error increments telemetry but still sets completion; released output is not valid. |
| `safeShutdownOwnershipTest` | Completion is consumed, poller stops and is destroyed, then the idle host closes. |
| `unsafeBorrowShutdownWitnessTest` | Destroying a host with a borrowed arena alias invalidates it and records a lifetime violation. |
| `obsoleteCudaHelperInactiveTest` | The unintegrated kernel helper cannot replace active Driver stream-memory operations. |

`inv` checks bounded resident identity, arena accounting, flag range, result/release consistency, borrow validity, running-poller buffer ownership, and the active signaling mechanism. It does not claim memory-order, numerical tolerance, DMA bandwidth, CUDA event correctness, or race freedom of the implementation.

## 7. Owned harness and benchmark inventory

These files are explanation and comparison surfaces, not production modules, and no harness was run during authoring:

* [`cpu_expert_test.py`](../../../src/phase7/cpu_expert_test.py) compares single and routed FFNs with a PyTorch reference; exercises shared-expert bucket indexing, exact empty zeros, counted nonresidency, Python validation, and in-place reload.
* [`cpu_packed_test.py`](../../../src/phase7/cpu_packed_test.py) compares random packed bytes against vLLM `dequantize_to_dtype`, checks packed density, global-scale order, routed accumulation, and rejection of unpacked planes.
* [`cpu_poller_test.py`](../../../src/phase7/cpu_poller_test.py) owns host-side stand-in flags and long-lived staging buffers; compares signal-driven results with direct calls, verifies `M` bounds touched rows, and verifies error completion.
* [`cpu_stream_test.py`](../../../src/phase7/cpu_stream_test.py) requires CUDA; checks contiguous section order, arena pinning, streamed-versus-host partials across ring reuse, negative-route skipping, exact empty zero, per-layer identity, and stream counts.
* [`cpu_expert_bench.py`](../../../src/phase7/cpu_expert_bench.py) is intended to report median decode latency and implied bandwidth across thread counts. It has drifted from the packed binding: `bench` constructs bf16 tensors and passes them directly to `CpuExpertHost.load_expert`, whose live signature requires `PackedPlane` objects, and its byte calculation is bf16 rather than packed bytes. Its header/cost statements therefore describe benchmark intent, but the checked-in call path is not a runnable packed-tier measurement without correction.

A second source-visible reporting defect is `ExpertStreamer.stats["slot_bytes"]`: it reports `3 * hidden * intermediate` bytes for a `uint8` ring slot, while allocation uses `3 * (plane/2 + plane/16)`. `gib_streamed` uses the correct `_block_bytes`. Consumers comparing telemetry should use the latter allocation formula.

## 8. Abstraction limits and implementation comparisons

The model intentionally omits tensor contents, exact packed decode, floating-point non-associativity, NUMA placement, transparent-huge-page realization, actual PCIe overlap, and Python/vLLM scheduling. It treats a native direct FFN as a staged logical computation and poller `moe_forward` as one atomic service call. It also models one registered layer and a two-expert universe; production counts are configuration data, not different state transitions.

Concrete CPU-executable comparisons available to the parent verification owner are:

1. packed arena FFN versus vLLM dequantization of identical bytes, including nibble order and global-scale direction;
2. routed host output versus an explicit per-route weighted PyTorch sum, including repeated experts and exact all-negative zero;
3. sparse residency output plus monotonic skipped-route count;
4. direct `moe_forward` versus native poller output and failure-release telemetry;
5. source inspection of reload arena usage, committed-prefix pin range, missing unregister, and host/poller ownership;
6. on a CUDA-capable environment, streamed versus host partials across more experts than ring slots and exact empty-route zero; and
7. source inspection of the obsolete helper's absence from phase4 call/build paths.

The CPU tests use fp32 host accumulation while CUDA streaming uses bf16 GEMMs with fp32 accumulation and another reduction order, so their intended property is bounded numerical agreement, not bit identity. Performance comments and thresholds are measured-policy context, not invariants, and the drifted benchmark cannot currently establish them.
