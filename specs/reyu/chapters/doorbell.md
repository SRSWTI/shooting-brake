# Doorbells and command-streamer alternatives

This branch refines the high-level “B70 lane(s) / pinned staging / issue-take” box into two executable state machines: [the current-stream doorbell protocol](../models/doorbell.ryu) and [the optional command-streamer modes](../models/command_streamer.ryu). Read them in that order. The first explains the transport used by the production r15 recipe; the second explains native alternatives that are selected by a separate environment variable and are dormant until explicitly armed.

## 1. Responsibility, actors, and interfaces

For each `(model layer, physical B70)` edge, CUDA and one native poller exchange data through long-lived pinned buffers and two distinct host-mapped words:

1. The **current CUDA stream** casts/stages hidden rows, compact route IDs, and route weights into that lane's pinned input buffers.
2. The stream writes the **batch size `M`** into the lane's signal word. `0` means idle. This word is not a Boolean request and has no sequence field.
3. Exactly one **native poller per physical B70** observes the signal and drives that card's provider. One poller serves all registered layers on that card; two cards use two independent native threads and queues.
4. The poller writes `1` to the lane's completion word after successful output publication **or after failure**. Completion is release, not validity.
5. The CUDA stream waits for exact equality with `1`, copies/casts the pinned result, and writes `0` to completion only after the copy. The lane can then be reused.
6. A later **eager Python forward boundary** inspects the poller's cumulative error counter. Captured graph replay performs no Python check, so stale output may already have been copied before the failure becomes an exception.

The process-wide Python ownership is in [`b70_poller.py`](../../../src/phase4/src/shooting_brake_vllm/b70_poller.py): `get_b70_poller` keys instances by remote device index, `register_layer` gives native code raw addresses whose allocations must outlive the poller, and `arm_all_cs` is the explicit post-capture latch for optional command-streamer modes. The ctypes provider and ABI contract live in [`b70_binding.py`](../../../src/phase4/src/shooting_brake_vllm/b70_binding.py) and [`b70_capi.h`](../../../src/phase7/b70_capi.h); the hot native state machine is `B70Poller::loop` in [`b70_capi.cpp`](../../../src/phase7/b70_capi.cpp).

The CUDA-graph flag (`SHOOTING_BRAKE_B70_GRAPH`) and native command-streamer selector (`SHOOTING_BRAKE_B70_CS_DOORBELL`) are independent. The r15 production recipe enables graph-compatible signaling but does not set the CS selector, so the native side runs the classic host sweep. Graph mode does not imply chain, ride, or baked mode.

## 2. HLD state and operation map

| HLD operation | Owner | Observable state/change | Source anchor |
|---|---|---|---|
| Allocate a flag | Python/CUDA runtime | One 256-byte portable mapped allocation; host and device pointers name the same storage | `stream_signal.py::alloc_host_mapped_flag` |
| Stage one lane | Current CUDA stream | BF16→FP16, async D2H hidden/IDs/weights complete before the subsequent stream write | `routed_experts.py::_b70_issue_graph` |
| Ring | Current CUDA stream | `signal := M`, where capture fixes `M` for that graph | `stream_signal.py::write_flag`; `_b70_issue_graph` |
| Claim classic work | Native poller thread | Read `M`; set signal to `0`; sequentially consistent host fence | `B70Poller::loop` classic sweep |
| Issue/take | Native poller/provider | A poller-local monotonically increasing provider sequence identifies the C++ calls; sequence is not stored in either flag | `B70Poller::loop`; `sb_b70_issue`; `sb_b70_take` |
| Release | Native poller | Fence, then `completion := 1` on success or failure; dispatch/error counters advance independently | `B70Poller::loop` |
| Join and reuse | Current CUDA stream | Exact-equality wait for `1`; H2D and cast; then `completion := 0` on that stream | `stream_signal.py::wait_flag`; `routed_experts.py::_b70_take_graph` |
| Observe failure | Eager Python control | On the next eligible eager forward, any nonzero lane poller error count raises and labels results untrustworthy | `routed_experts.py::forward_modular` |
| Register while running | Python + native poller | Append under mutex; hot loop refreshes an immutable snapshot only when atomic layer count changes | `B70Poller::register_layer`; `B70Poller::add/loop` |
| Reset metrics | Caller/native poller | Counters are atomically zeroed; one in-flight dispatch may straddle the boundary | `B70Poller::reset`; `sb_b70_poll_reset` |

The [`doorbell` model](../models/doorbell.ryu) makes each lane independent, allows the two cards to interleave, and makes stream ordering explicit without modeling tensor contents. Its `providerSequence` is the poller's call identity from the classic implementation. The modeled signal remains exactly `0 | M`, and completion remains exactly `0 | 1`; there is deliberately no invented completion sequence.

## 3. Classic polling mechanics

### 3.1 Registration and lifetime

`B70Poller.register_layer` passes a compact **bank row**, not necessarily an absolute model layer. It retains no native reference to the tensors; the Python lane must keep every flag and pinned buffer alive until native poller destruction. Native registration checks non-null pointers and nonzero `topk`, compares `topk` with the loaded provider capability, and appends a `PollLayer` snapshot entry. It attempts to import the four staging ranges into the B70 SYCL context unless `SHOOTING_BRAKE_B70_XPU_REGISTER=0`. Import failure is fail-open: correctness remains on the pageable/staged path, but latency can regress.

Start is idempotent. CPU affinity is best-effort: a pin failure logs and continues unpinned. Correct ownership is nevertheless one thread/queue/provider per card; sharing the same host core between two pollers is documented as destroying intended parallel service, not changing the flag protocol.

### 3.2 Signal, provider call, and release

The classic loop refreshes its local vector only when `layer_count_` changes. It scans every registered layer, skips `signal==0`, clears a nonzero signal before provider work, increments one local provider sequence, and calls `issue`. It calls `take` only if issue succeeded. Service time, rows, shape bucket, trace ring, and optional provider timings are recorded around this pair.

The critical failure behavior is unconditional release. Whether issue fails, take fails, or both succeed, the poller executes a sequentially consistent fence and stores `completion=1`. CUDA's stream wait has no timeout; withholding completion would permanently park the device and can make the process unkillable. Consequently:

- `completion==1` means the waiter may proceed;
- it does **not** mean the output buffer was freshly written;
- the error counter, not completion, carries the validity alarm;
- a failed replay can copy stale contents before Python observes the counter.

`forward_modular` checks all lane pollers only on a subsequent eager invocation after first-forward setup. It sums the counts and raises. Replay itself has no Python execution. The model's `waitAndCopy`, `reachEagerBoundary`, and `observeErrors` actions retain this timing instead of pretending failure is synchronous with the CUDA wait.

### 3.3 Current-stream reset and exact equality

`stream_signal.wait_flag` passes `_WAIT_VALUE_EQ`, not greater-or-equal. The real CUDA test initializes the word to `2`, verifies following work remains parked, changes it to `1`, and repeats the same check under graph replay. `_b70_take_graph` enqueues wait, H2D result copy, FP32→BF16 copy, and completion reset on the same stream. Thus a lane is not reusable merely because native code published completion; reuse follows the consumer's copy and reset.

`stream_signal.py` calls Driver/Runtime APIs through ctypes but does not check their return values. The model abstracts successful allocation and enqueuing; CUDA API refusal is an external failure boundary, not a modeled provider error.

## 4. Optional command-streamer modes

`B70Poller::loop` reads `SHOOTING_BRAKE_B70_CS_DOORBELL` once into a process-local static selector: a single character `1`, `2`, or `3` chooses chain, ride, or baked; unset, malformed, and other values select classic. Every optional mode additionally requires `cs_armed_`, a nonempty complete registered layer set, and no previously latched failure. Python calls `arm_all_cs` only from the post-capture hook. Never arming leaves classic polling permanently active and safe.

### 4.1 Mode 1: provider-owned chain

For `M<=32` and NVFP4, `issue_cs_chain` builds a provider-owned in-order raw Level-Zero path:

`WAIT(signal==M) → WRITE(signal=0) → three H2D copies → input event → SYCL barrier/kernels/narrow → SYCL tail event → D2H → WRITE(completion=1)`.

Input and tail events are explicit raw/SYCL seams. The completion write follows output transfer. `M>32`, zero, invalid layer/pointers, int4 banks, generation mismatch, stopped/not-loaded state, extension/list/pool/event failures, and append failures are refusals. Initialization failures latch provider CS failure.

### 4.2 Mode 2: ride-along

`issue_cs_ride` fetches the SYCL queue's current immediate command list on every call and appends the hardware wait/write brackets to that same list. Marker kernels bridge the raw bracket and SYCL event chain across possible copy/compute engines. It has the same `M<=32`, NVFP4, identity, and pointer checks as mode 1, plus the assumption that the queue has compatible one-list-per-queue semantics. Diagnostic `SHOOTING_BRAKE_B70_RIDE_BISECT` bits can deliberately weaken waits, clears, or markers; they are diagnostic alternatives, not production guarantees.

### 4.3 Mandatory full-step fence and fallback

After appending every layer in a mode-1/2 step, the poller must call `cs_step_fence` and wait for the queue tail. Resuming the classic scan earlier can clear a later layer's signal before its hardware `WAIT` consumes it, deadlocking the in-order queue. Any append or fence failure increments the poller error count, permanently latches `cs_disabled`, and falls through to classic scanning for remaining work. Larger/prefill-shaped `M>32` signals also stay on the classic path without disabling CS.

The command-streamer model exposes `FenceRequired` as a real state: it cannot start another fast step until `fenceFastStep`. It also has separate chain and ride scenarios even though both share the poller's eligibility and fallback policy.

### 4.4 Mode 3: baked lists

Baked mode records exactly one regular raw command list per registered layer, after capture has finished and all layers are known. It requires NVFP4 and provider `out_fp16`; it hard-codes `M=1`. Each list waits for `signal==1`, clears it, barriers before staging, executes baked gate/up and weighted-down kernels, copies FP16 output, barriers, publishes completion, and finally writes a host-visible per-layer progress word. Layer zero clears its signal at host scope because the host uses it to trigger the whole step; later clear writes can stay device-scoped.

The poller submits all recorded lists as one step. A bounded wait returning busy is not itself failure: the first eager decode can spend seconds in Python between layer doorbells. The poller samples the provider's monotone progress words. If progress advances, it waits another slice. A full slice with no progress identifies a dead partial step. Recovery then:

1. writes `1` into all signals to satisfy stuck hardware waits;
2. fences and attempts a bounded poison drain;
3. asks the provider to disable baked mode permanently;
4. preserves completions so any CUDA waiter can escape, even though its result may be garbage;
5. clears a signal only when that layer's completion is already `1`;
6. preserves a signal with completion `0`, because it may be a real doorbell that raced recovery and must be served by the classic sweep;
7. increments the error counter and latches classic fallback.

This is not CUDA graph capture/replay. CUDA graph mode produces the stream-side writes/waits; baked mode changes how the B70 command streamer consumes those words. The two controls and state machines remain separate in the model.

## 5. Named executable scenarios and properties

### `doorbell.ryu`

- `twoLaneInterleavingTest`: two independent card lanes stage and ring before either joins; provider and completion order differ; both reset safely.
- `issueFailureReleasesTest`: issue failure still releases CUDA, CUDA copies stale data, and the error becomes observable only at an eager boundary.
- `takeFailureReleasesTest`: take failure has the same release-without-validity contract.
- `noPrematureReuseTest`: a signaled lane cannot be staged or reset as though empty.
- `exactCompletionValueTest`: the consumer joins on `1`, then resets to `0` for reuse.
- Witnesses expose concurrent lanes, a released failure, an unobserved error window, and fully reusable lanes.

`inv` keeps flag domains, validity/release ordering, stale-copy meaning, and monotone error observation consistent. It is a safety predicate over the abstraction, not proof of CUDA cache coherence or device progress.

### `command_streamer.ryu`

- `productionClassicGraphTest`: demonstrates the production-recipe combination “CUDA graph enabled, native mode classic.”
- `chainFenceTest`: reaches the mandatory fence and rejects a rescan/start before draining it.
- `rideFailureFallbackTest`: a ride append failure permanently latches classic fallback.
- `bakedHealthyStepTest`: all registered layers make monotone progress and drain normally.
- `bakedPoisonFallbackTest`: a partial baked step is poison-drained and disabled.
- `bakedRacePreservationTest`: recovery retains an unconsumed racing signal for classic service.
- `unarmedModeStaysClassicTest`: configured-but-unarmed command streaming cannot start, while the same signal is served by the classic sweep.
- `largeBatchBypassesFastPathTest`: an armed chain mode leaves `M=33` to classic polling without disabling the optional mode.
- `bakedWireMismatchUsesClassicTest`: an FP32 result wire makes baked recording ineligible, while the same `M=1` request remains serviceable by the classic sweep.

The finite two-layer domain represents completeness and ordering, not the production layer count. Tensor math, raw event handles, queue clocks, and time durations are abstracted to success/failure/progress transitions.

## 6. Assigned source inventory and correspondence

| Assigned file | Role in this branch | What is and is not modeled |
|---|---|---|
| [`b70_binding.py`](../../../src/phase4/src/shooting_brake_vllm/b70_binding.py) | ctypes ABI owner and synchronous provider lifecycle | Provider handle/load/issue/take identity, output-width query, poller symbols, health and teardown are interface context. Numerical provider work belongs to the provider model. |
| [`b70_poller.py`](../../../src/phase4/src/shooting_brake_vllm/b70_poller.py) | Python lifetime, registration, registry, counters, traces, arming | Per-card ownership, raw-buffer lifetime, delayed error observation contract, start/reset/arm are modeled. JSON trace dumping and timing arithmetic are documented but not state-machine behavior here. |
| [`stream_signal.py`](../../../src/phase4/src/shooting_brake_vllm/stream_signal.py) | Production CUDA Driver flag operations | Mapped flag identity, exact-equality wait, current-stream write/wait/reset are modeled. Driver implementation and cache coherence are external. |
| [`b70_capi.cpp`](../../../src/phase7/b70_capi.cpp) | Native poll loop and C ABI implementation | Classic claim/issue/take/release, all CS branches, counters, registration fail-open, fallback, recovery, and lifecycle are modeled or explained. Trace timestamps are observational. |
| [`b70_capi.h`](../../../src/phase7/b70_capi.h) | Published C ABI and ownership/failure prose | Return/status, pointer lifetime, signal/completion contract, registration/start/reset/counters/teardown are interface evidence. |
| [`b70_stream_test.py`](../../../src/phase7/b70_stream_test.py) | CPU-only placement/prefill-stream harness | It checks B70/CPU/CUDA ownership partition, arena sizing, capable layers, threshold and compact-slot/global-ID separation. It does **not** exercise doorbells or `B70Poller`; it is inventoried rather than behaviorally attributed to this model. |
| [`sb_stream.cu`](../../../src/phase9/sb_stream.cu) | Historical/experimental CUDA-kernel flag helper | Its one-thread write+system-fence and spin-wait illustrate the primitive problem, but production uses Driver stream memory operations. No explicit stream is accepted, and launch failures are printed rather than returned. |
| [`test_stream_signal.py`](../../../tests/test_stream_signal.py) | Real CUDA eager/graph-replay contract test | Exact equality, current-stream blocking, replay reuse, and cleanup are evidence for the modeled join/reset semantics. It requires CUDA and is not a CPU gate. |

Relevant dependencies outside the ownership list are [`routed_experts.py`](../../../src/phase4/src/shooting_brake_vllm/routed_experts.py), which orders staging/ring/wait/copy/reset and observes poller errors, and [`b70_provider.hpp`](../../../src/phase1/b70_provider.hpp) plus [`b70_provider.cpp`](../../../src/phase1/b70_provider.cpp), which implement the provider issue/take and command-streamer chains.

## 7. Source contradictions and limits

Two checked-in ABI inconsistencies must remain visible:

- `b70_binding.py` configures and invokes `sb_b70_poll_arm_cs`, and `b70_capi.cpp` exports it, but `b70_capi.h` does not declare it. Python's dynamic symbol lookup makes the live path possible; the header is incomplete for a conventional C/C++ caller.
- `b70_capi.h` describes the registered poller output as pinned FP32, while the implementation can write FP16 when `sb_b70_out_fp16` reports it. `b70_binding.py::out_dtype` correctly asks the loaded provider instead of trusting environment or prose. The C pointer type does not protect against a width mismatch.

The CUDA Runtime/Driver calls in `stream_signal.py` do not check return status. `sb_stream.cu` only prints launch failures and uses the current/default launch context because it accepts no stream handle. Neither file supplies timeout semantics. These observations are not repaired in the models.

The model collapses CUDA copies, SYCL work, and system fences to atomic transitions at their externally relevant order boundaries. It does not claim fairness for a spinning host thread, hardware WAIT progress, numerical correctness, latency, visibility on an unsupported platform, or equivalence of CUDA and Level-Zero memory models.

## 8. Implementation comparisons, including CPU-only options

A faithful end-to-end doorbell comparison is not CPU-only: `stream_signal.py` loads CUDA libraries and the production poller calls a loaded SYCL/Level-Zero B70 provider. The existing `test_stream_signal.py` is the narrow real-hardware comparison for eager and graph replay, while command-streamer behavior requires B70 hardware and its provider.

Useful CPU-executable checks are therefore limited and should not be misreported as transport validation:

- parse `b70_capi.h`, `b70_capi.cpp`, and `b70_binding.py` to compare exported/bound symbols and expose the missing `sb_b70_poll_arm_cs` declaration;
- inspect `B70Poller::loop` ordering to assert the failure path reaches the fence and `completion[0]=1` after issue/take status handling;
- inspect `_b70_issue_graph`/`_b70_take_graph` ordering to assert write-after-staging and reset-after-copy;
- run `b70_stream_test.py` only as its stated CPU placement/prefill-stream harness, never as doorbell evidence.

No assigned CPU implementation reproduces CUDA exact-equality stream waits or Level-Zero command-streamer execution. A source correspondence check can catch drift in the abstraction; only hardware-backed scenarios can compare the synchronization implementation itself.
