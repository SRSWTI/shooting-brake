# Native B70 provider

[The executable provider model](../models/provider.ryu) is the first detailed branch below the system-level “B70 lane” box. Read it breadth first: this chapter starts with the provider boundary and its lifecycle, then narrows into load, classic dispatch, native command queues, failure and teardown mechanics. Kernel arithmetic is intentionally linked to the compute and embedded-runtime models rather than repeated here.

## 1. Responsibility, actors and interfaces

The provider owns one selected Intel B70 device context, a read-only mapped expert bank, persistent device/host staging allocations, an in-order SYCL queue, and at most one classic `issue`/`take` dispatch. Its callers own request arrays and final output storage. The provider copies request data into its allocations and publishes final bytes into caller storage only after an exact-identity `take` succeeds. The API and observable records are declared by [`B70Provider`, `ProviderConfig`, `Capability`, `Health`, and `DispatchResult`](../../../src/phase1/b70_provider.hpp).

At this boundary:

- **Inputs:** bank path; maximum batch, top-k, generation, compact resident-expert ordering and optional device selector; then `(generation, sequence, layer, hidden, compact IDs, weights, M)` per classic dispatch.
- **Outputs:** immutable capability after load, mutable health, a result wire selected as fp32 or fp16, exact completed identity and optional timing, and status codes `ok`, `busy`, `not_loaded`, `invalid_argument`, `generation_mismatch`, `sequence_mismatch`, `device_error`, or `shutdown`.
- **External actors:** Linux file/mmap facilities, SYCL/Level Zero, embedded QuixiCore-XPU operations, the optional grouped-prefill implementation, and the caller/poller that sequences calls and owns buffers.
- **Concurrency boundary:** public state transitions are serialized by `Impl::mutex`; asynchronous SYCL failures are accumulated behind `async_mutex`. `issue` enqueues a whole in-order chain and returns before completion; `take` waits its last copy event and is the classic publication boundary.

The provider model abstracts tensor contents and device commands but retains generation/sequence identity, single flight, shape rejection, kernel-family choice, poison, output ownership, recoverable native-queue busy, terminal native disable, telemetry and teardown. See the detailed compute and embedded-runtime chapters/models for NVFP4/int4 arithmetic and the command-streamer chapter/model for exact hardware wait/write ordering.

## 2. Lifecycle and state at a glance

| State or operation | Source-backed meaning | Model action/observation |
|---|---|---|
| Unloaded | No accepted bank or queue; failed load releases partial mappings and allocations | `init`, `rejectLoad` |
| Loaded/ready | Bank geometry and placement accepted, device selected, persistent buffers uploaded, capability published | `load` |
| Pending | One classic request identity and final copy event exist; another `issue` is `busy` | `issue`, `witnessBusyStillPending` |
| Take rejection | Wrong generation/sequence or short/null output leaves the pending dispatch intact and caller output untouched | `rejectTakeGeneration`, `rejectTakeSequence`, `rejectTakeBuffer` |
| Take success | Waits the tail, reports identity/kernel/timing, retires pending state, then copies provider staging into caller storage | `take`, `witnessSplitPublished` |
| Poisoned take | A sequence-bound injected or asynchronous device error retires the exact pending dispatch without modifying caller output/result sentinels | `poisonPending`, `takePoison`, `witnessPoisonPreservesCaller` |
| Direct CS dispatch | Modes 1/2 write the ring output and completion directly and deliberately do not use classic pending bookkeeping | `nativeDispatch` |
| Baked flight | Pre-recorded lists may return `busy` while still executing; progress can continue and repeated waits may drain | `executeBaked`, `observeBakedProgress`, `bakedWaitBusy`, `bakedDrain` |
| Native poison | CS setup/append failure latches `cs_failed`; baked record/submit/wait failure or `baked_disable` separately latches `baked_failed`. Either native path can fall back to classic issue/take. | `nativeLatchFailure`, `bakedQueueFailure`, `disableBaked` |
| Stopped | First shutdown drains best effort, frees provider resources, releases imported host ranges, unmaps/closes bank, and marks stopped; later shutdowns are no-ops | `shutdown`, `witnessShutdownReleased` |

`Health::allocations` is a cumulative allocation count, not current live allocation count. It increases only during load/lazy native setup in the implementation tests’ covered classic path and does not decrement at shutdown. The model therefore separates `allocations` from `resourcesOwned`.

## 3. Loading and immutable capability

[`B70Provider::load`](../../../src/phase1/b70_provider.cpp) checks stopped/already-loaded state and basic config first. It opens a regular file, maps it read-only, recognizes `SBEXP001` or `SBINT401`, derives geometry, validates exact byte extents and selects a B70. A missing file/runtime/device is reported as `device_error`; malformed contract input is generally `invalid_argument`. Every rejected load calls `release_resources_locked(false)`, so partial ownership does not escape.

### NVFP4 (`SBEXP001`)

The reader derives plane sizes from bank-declared layers, experts, hidden and intermediate sizes. Hidden/intermediate dimensions must be multiples of 16; source experts are bounded by 256 but are not assumed to equal 256 (the source explicitly notes a 205-expert model). An empty resident list expands to every source expert. A nonempty list is ordered compact-slot-to-canonical-ID mapping and must contain unique in-range IDs. Loading gathers each selected canonical expert into six structure-of-arrays device planes: packed gate/up weights and scales, down weights and scales, and two global dequant multipliers.

### Int4 (`SBINT401` v2)

Int4 is not automatic: `SHOOTING_BRAKE_B70_INT4=1` is required. The reader validates version 2, 128-byte fixed prefix, shared resident set, group size 128, 4 bits, effective zero point 8, divisibility, six contiguous 4096-byte-aligned planes, expert/layer strides, exact file size, zero header padding and strictly increasing source expert IDs. A configured resident list, if supplied, must exactly equal the bank’s embedded source-ID map. The entire packed AoS arena is uploaded in 32 MiB mmap slices directly to one device allocation; completed slices are `MADV_DONTNEED`ed to bound file-backed RSS without an anonymous bounce buffer.

### Allocation and feature flags

Persistent request, route, scratch, output and host copyout buffers are sized from `max_batch`, top-k and bank geometry. `SHOOTING_BRAKE_B70_OUT_FP16=1` additionally allocates device/host fp16 output staging and is published as `Capability::output_fp16`; upstream accumulators remain fp32. `SHOOTING_BRAKE_B70_GROUPED=1` allocates NVFP4-only grouped buffers, with `SHOOTING_BRAKE_B70_PIPELINE` clamped to 1–8 and a second in-order copy queue only when chunk count exceeds one. Profiling and spin wait are off unless their exact `=1` variables are present. The provider rejects a placement whose computed persistent byte requirement exceeds device global memory.

Device selection prefers Level Zero Intel GPUs whose name contains `B70`. An empty selector chooses the first; a decimal selector chooses the enumerated index; otherwise a case-insensitive PCI BDF must match. Capability records the chosen index/BDF, total/estimated available bytes, backend, dynamic geometry, source-ID map for int4, wire width and kernel families.

## 4. Classic `issue`/`take`

[`B70Provider::issue`](../../../src/phase1/b70_provider.cpp) evaluates guards in this observable order: stopped, loaded queue, pending/busy, generation, pointers/batch/layer, then every compact ID and routing weight. IDs may be `-1` (skip) or `[0,resident_count)`; less than `-1`, out-of-range IDs and non-finite weights are rejected before pending state is claimed.

Accepted inputs are copied to persistent device buffers. Kernel selection is:

1. **Int4:** always `int4_moe_split`; int4 does not use grouped or native CS paths.
2. **NVFP4, $M\le32$:** `nvfp4_moe_split`.
3. **NVFP4, $M>32$, grouped ready:** try grouped execution per token chunk. A refusal falls back to fused execution; partially written output is fully overwritten.
4. **NVFP4, $M>32$, grouped unavailable/refused:** `nvfp4_moe_fused`.

With grouped pipelining, explicit event barriers carry copy-queue-to-compute RAW dependencies and previous-compute-to-next-copy WAR dependencies. Token chunks own disjoint output rows. Without that branch, all input copies, compute, optional fp16 narrowing and output staging copy are placed on the main in-order queue. `issue` records generation, sequence, $M$, reported split/fused family, increments dispatch count and marks pending only after the chain has been constructed.

[`B70Provider::take`](../../../src/phase1/b70_provider.cpp) requires a pending exact generation/sequence. A stored dispatch error is handled before output-pointer validation: it sets `last_error`, retires pending work and returns `device_error` without touching caller output or `DispatchResult`. Otherwise the output and result pointers/capacity must cover `M * hidden`; an invalid buffer returns `invalid_argument` without retirement. The provider optionally performs a bounded spin, then always `wait_and_throw`s the final copy event so asynchronous errors surface. Only after a successful wait does it build the result, retire pending state and `memcpy` fp32 or fp16 staging into the caller’s wire buffer.

One nuance is explicit: `DispatchResult::kernel` stores only `split` versus `fused`; int4 and grouped success are not independently named there. Int4 tests expect `split`; grouped success leaves `pending_split == false` and is reported as `fused`. The model preserves the actual selected abstract kernel for understanding, rather than pretending the current result string distinguishes it.

## 5. Native command-queue alternatives

These branches are alternatives to classic host issue/take and are not selected merely by CUDA graph mode.

- **Owned chain (`issue_cs_chain`, mode 1):** accepts NVFP4 split shapes only ($1\le M\le32$). A provider-owned raw immediate list waits for `signal == M`, clears it, copies three inputs and signals a Level Zero event; SYCL kernels wait on that event; the raw list waits for the SYCL tail, copies output and writes completion. Lazy initialization failure sets `cs_failed`, making the failure sticky. `cs_step_fence` must drain the SYCL tail before a classic poller resumes scanning signals.
- **Ride-along (`issue_cs_ride`, mode 2):** appends wait/clear and completion brackets to the SYCL queue’s current immediate list. Marker kernels bridge raw-list ordering to copies that may use a different engine. `SHOOTING_BRAKE_B70_RIDE_BISECT` can deliberately weaken/skip ordering pieces for diagnosis and is not a production default.
- **Baked decode (`baked_record_layer`, mode 3):** requires NVFP4 and fp16 output. It records one closed raw command list per layer for $M=1$: wait/clear, input copies, gate/up, fused W2 epilogue directly to fp16, output copy, completion and a final host-only progress write. Re-registering a layer is idempotent. `baked_execute_step` submits every recorded layer list. A deadline returns `busy` without setting failure because the lists remain live on hardware waits. `baked_wait` permits bounded re-wait. The caller uses monotone progress count to distinguish continuing work from a no-progress partial step; `baked_disable` is the explicit terminal process-lifetime latch after abandonment.

A `busy` baked result is therefore not analogous to classic single-flight `busy`: classic `busy` rejects a second request; baked `busy` says an already-submitted native step remains in flight. The model gives both the same public status but distinct `Pending` and `BakedFlight` states.

## 6. Ownership, telemetry, failure and shutdown

Provider-owned memory includes mapped bank metadata, queue/context, uploaded weights, input/route/scratch/output buffers, host output staging, optional grouped/pipeline state, and lazily created command-streamer resources. Request arrays and the final destination remain caller-owned. `register_host_range` imports an externally pinned caller range into the provider’s SYCL context; it never takes ownership of the memory, and the caller must keep the range alive. Registration is fail-open and all successful registrations are released before the queue/context during shutdown.

`device_memory` queries the already-selected device and returns `false` without changing either caller output parameter when the runtime lacks the aspect or throws. It is telemetry, never a dispatch failure. `health()` overlays any captured asynchronous message on its returned copy without consuming the error.

Shutdown best-effort drains/destroys baked and raw lists, drains the copy queue before freeing buffers it may read, releases imported host ranges while context is alive, frees all provider allocations, clears asynchronous and pending state, unmaps/closes the bank, then sets `loaded=false`, `pending=false`, `stopped=true`. Calling shutdown again is harmless. Calls after shutdown return `shutdown` where the API explicitly checks stopped.

## 7. Control process and complete owned-file inventory

[`b70_provider_main.cpp`](../../../src/phase1/b70_provider_main.cpp) is a narrow stdin control process, not the serving dispatch bridge. It strictly parses one `--bank`, `--max-batch` and `--generation`, installs SIGINT/SIGTERM handlers, loads with top-k 8, emits initial capability JSON, then accepts newline-delimited `capability`, `health`, or `shutdown`. EOF and signals also shut down; unknown commands emit JSON errors. It never exposes `issue` or `take`.

Important implementation mismatch: the C++ `Capability` now contains `device_index`, `device_pci_bdf`, `output_fp16`, and `source_expert_ids`, but this control executable’s `write_capability` does not serialize those fields. In particular, consumers cannot safely discover the fp16 result wire through this executable’s JSON even though the API comment says the choice must be published. Source behavior is documented, not repaired here.

[`b70_provider_tests.cpp`](../../../src/phase1/b70_provider_tests.cpp) is the public NVFP4 API harness. It checks capability/health, wrong generation/layer/compact IDs, single flight, stale takes, $M\in\{1,2,4,8,16,32,128\}$, duplicate top-8 routes, compact resident gathering, stable allocation count, idempotent shutdown and optional sequence-bound poison preserving caller sentinels. It compares every output to a fixed golden with `1e-6 + 1e-2 * abs(expected)` tolerance.

[`b70_provider_int4_chain_test.cpp`](../../../src/phase1/b70_provider_int4_chain_test.cpp) validates exact int4 resident maps and compares an $M=1$ top-8 provider result against a bounded-memory CPU dequant/SwiGLU/down-projection oracle. It reads one expert record at a time, reports a worst-case CPU-oracle resident footprint of 4,901,432 bytes, requires peak-relative error at most `5e-5`, checks partition recomposition at `5e-6`, exact zero for all skipped routes, timing sweeps and sampled RSS. It is present source but is **not** a target in the Phase-1 Makefile.

[`b70_provider_test.cpp`](../../../src/phase1/b70_provider_test.cpp) bypasses the public provider API. It directly parses the legacy NVFP4 bank, uploads one layer, invokes QuixiCore split/fused operations, compares to the legacy golden, and reports timing for repeated-row batches. Its device finder accepts a name containing `B70` **or `Arc`**, while the actual provider requires Intel vendor plus `B70`; this harness is broader and must not be treated as the provider’s selection contract.

[`Makefile`](../../../src/phase1/Makefile) builds that low-level harness, the public NVFP4 API harness and stdin control executable with `icpx -fsycl`, linking embedded QuixiCore-XPU libraries. It does not build the int4 chain harness and does not encode the optional test-fault macro.

## 8. Executable scenarios and properties

The model includes reachable witnesses and named runs for:

- classic split success and publication;
- busy issue plus stale generation/sequence takes that preserve pending work;
- too-small output followed by a valid take;
- post-kernel poison retiring without caller mutation;
- grouped-enabled and grouped-disabled large-batch modes (the implementation also falls back to fused if the grouped dependency refuses a call);
- int4 split selection;
- shape rejection before buffer claim;
- direct owned-chain publication without classic pending state;
- baked timeout with continuing progress and later drain;
- terminal baked disable with caller output preserved;
- host-range release at shutdown; and
- fail-closed rejected load.

The invariant ties pending identity to pending phase, bounds publication count by accepted dispatch count, separates cumulative allocation telemetry from live ownership, prevents int4/grouped conflation, and requires stopped state to own no provider resources.

## 9. Abstraction limits and implementation comparisons

The Reyu model does not calculate packed weights, floating-point reductions, device capacity, PCIe timing, event-engine coherence or SYCL exceptions. It collapses successful bank authentication/upload to `load`, accepted command construction to `issue`, and device completion/publication to `take` or native completion. “Poison” denotes a result that must not be published; it does not claim the provider object is permanently unusable after a classic dispatch failure. Native `cs_failed`/baked disable are modeled as sticky only for native modes, matching fallback intent.

Concrete comparison paths already in owned source are:

1. the int4 harness’s streaming CPU oracle versus provider output, including exact skipped-route zero and partition additivity;
2. the public NVFP4 harness versus a frozen golden across split/fused batch boundaries and duplicate routes; and
3. the low-level QuixiCore harness versus the same legacy golden independent of public `B70Provider` state.

Those are potential implementation checks, not results claimed by this chapter. They require their real bank/reference files and, for provider output, B70/SYCL hardware. No test, build, formatter, lint or hardware workload was run while authoring this branch.
