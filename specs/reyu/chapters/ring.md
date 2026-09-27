# Shared ring, cancellation, and transport qualification

[The executable model](../models/ring.ryu) follows the source in breadth-first order: the service boundary, the slot state machine, identity and payload validation, closure and recovery, and finally the independent copy-only transport probe. It is an abstraction of the implementation, not a replacement for the ABI structs, Linux atomics, device runtime, or numerical provider tests.

## 1. Responsibility, actors, and boundaries

The Phase-2 service moves one compacted remote-expert request at a time from a single host producer to a single B70-provider consumer while allowing eight published requests to occupy deterministic slots. The host owns reservation, request bytes, publication, cancellation/deadline closure, consumption, and reclamation. The provider owns in-order claim, completion bytes, and completion publication. Shared atomic state is authoritative; two nonblocking eventfds only hint that work or a completion may exist.

The fixed [`SBRING02` wire ABI](../../../src/phase2/ring_protocol.hpp) and [`SharedRing`](../../../src/phase2/shared_ring.hpp) API are the service contract. [`b70_ring_provider.cpp`](../../../src/phase2/b70_ring_provider.cpp) authenticates a bootstrap over Unix `SOCK_SEQPACKET`, maps canonical experts to provider-local slots, and bridges validated claims into the Phase-1 `B70Provider`. The control ABI in [`b70_ring_control.hpp`](../../../src/phase2/b70_ring_control.hpp) is separate from the hot ring and carries bootstrap, startup, test-fault, and shutdown packets.

There are two important non-equivalences:

* The historical [`provider_protocol.yaml`](../../../src/phase0/provider_protocol.yaml) is a logical protocol-v1 design. It describes head/tail-style request and completion rings, compact IDs supplied by the host, `EMPTY/SUBMITTED/IN_PROGRESS/COMPLETED/FAILED`, a per-route failed mask, and timeout recovery intent. The executable Phase-2 cutover is ABI major 2 with fixed layouts, canonical IDs plus a remote mask and canonical route position, per-slot request/output generations, route-subset authentication, all-or-nothing terminal validation, cancellation flags, and destructive generation retirement. Protocol v1 is not a serialization description of `SBRING02`.
* [`memfd_transport_protocol.hpp`](../../../src/phase2/memfd_transport_protocol.hpp), [`memfd_transport_host.cu`](../../../src/phase2/memfd_transport_host.cu), and [`memfd_transport_provider.cpp`](../../../src/phase2/memfd_transport_provider.cpp) implement the independent `SBP2MEMF` v1 copy probe. That probe has one request arena and one response arena, not eight SBRING02 slots; it measures and byte-checks CUDA→shared→B70→shared→CUDA transfer. It has no placement/weight generations, route descriptors, cancellation quarantine, or expert computation.

## 2. Interfaces and fixed data contract

[`ring_protocol.hpp`](../../../src/phase2/ring_protocol.hpp) requires Linux x86-64 little-endian and always-lock-free 32- and 64-bit atomics. Its fixed service geometry is eight slots, at most 128 batch tokens, at most 128 staged tokens, hidden size 2048, top-k 8, and at most 1024 route positions. The header is 4096 bytes. Each request and completion descriptor is 320 bytes, each publication line is 64 bytes, each payload lane is 1,589,248 bytes, and the exact mapping is 12,726,272 bytes.

Each slot has disjoint regions for fp16 activations, canonical int32 expert IDs, fp32 routing weights, a byte remote mask, uint32 full-batch row mapping, uint16 canonical route positions, fp32 output, route status, and token status. A request descriptor carries ABI/layout identity; request, ring, provider, placement, and weight generations; scheduler step and deadline; shape and dtype; exact offsets/extents; placement, route-subset, and weight SHA-256 values; and the output-buffer version. A completion must echo the immutable request identity and additionally carries provider nonce, completion/error codes, output/status extents, and timings.

The public [`SharedRing`](../../../src/phase2/shared_ring.hpp) is move-only and owns duplicated/mapped descriptors. `create` makes and seals a memfd against growth, shrinkage, and resealing, maps it shared, creates two nonblocking close-on-exec eventfds, initializes the exact header, and marks all slots free. `attach` checks exact file size and seals before mapping, then validates every header layout and identity field. Typed payload spans are returned only after descriptor validation. `visit_mapping` exists only as a startup/shutdown hook for physical-memory registration; it is explicitly not a token-path API.

The control plane's fixed packets authenticate packet magic/version/size/reserved bytes. Bootstrap transfers exactly the mapping, request eventfd, and completion eventfd through one SCM_RIGHTS record. Startup returns provider load state and allocation baseline. The optional sequence-bound test fault is either armed, unsupported in a normal build, invalid, or rejected by the provider. Shutdown returns dispatch count and initial/final allocation counts.

## 3. Source-backed state and operation table

| Operation | Required state/identity | State change and observation | Source |
|---|---|---|---|
| `create` / `attach` | Four generations are nonzero; exact sealed mapping and expected identity on attach | Opens one mapping with every slot `FREE`, local host/provider sequence counters at 1 | `SharedRing::create`, `SharedRing::attach`, `validate_header` |
| `host_begin` | Live generation, exact next host sequence, valid shape/deadline, no other `HOST_WRITING`, deterministic slot free | CAS `FREE→HOST_WRITING`; increment that slot's request generation and output-buffer version; initialize exact descriptor and zero/default payload; advance host sequence | `SharedRing::host_begin` |
| `host_publish` | Ticket matches publication; lane `HOST_WRITING`; descriptor and canonical payload valid | Compute `SBRROUTE2` SHA-256; release-store `REQUEST_READY`; write request hint | `SharedRing::host_publish`, `route_hash` |
| cancel/timeout before publish | Matching `HOST_WRITING` ticket | Return slot to `FREE` and roll back the sole unpublished host sequence reservation | `host_cancel`, `host_timeout` |
| cancel/deadline after publish | `REQUEST_READY` | OR closure flag, retain tombstone in sequence, hint provider | same |
| cancel/deadline after claim | `PROVIDER_RUNNING` or `RESPONSE_READY` | OR closure flag; return `quarantined`; never expose a late output | same |
| `provider_claim` | Live generation, provider not busy, exact `next_provider_seq` slot is `REQUEST_READY` | CAS to `PROVIDER_RUNNING`, set local single-flight busy, validate request; return success, cancellation/deadline tombstone, or a valid claim that must be closed with an error | `SharedRing::provider_claim` |
| `provider_complete` | Valid matching claim, provider busy, lane `PROVIDER_RUNNING`; no `OK_PARTIAL` | Write completion identity/status; release-store `RESPONSE_READY`; clear busy; increment provider sequence; hint host | `SharedRing::provider_complete` |
| `host_consume` | Matching ticket and `RESPONSE_READY` | CAS to `HOST_READING`; exact completion/terminal validation; expose output only for unclosed `OK_ALL`; otherwise stale, quarantined, cancelled, deadline, or device failure | `SharedRing::host_consume` |
| `host_reclaim` | Matching `HOST_READING` ticket | Release-store `FREE`; only this permits deterministic wrapped-slot reuse | `SharedRing::host_reclaim` |
| `teardown_generation` | Exact ring/provider generation | Atomically OR global dead flag and hint both peers; all later operations observe `generation_dead` | `SharedRing::teardown_generation` |
| replacement | Old mapping is retired | Caller creates a different mapping with fresh ring/provider identity; claimed old lanes are never recycled | `test_child_kill_replacement` |

The resulting legal ownership path is:

```text
FREE --host--> HOST_WRITING --publish--> REQUEST_READY
  ^                                      |
  |                                      | provider, in contiguous sequence
  |                                      v
  +--reclaim-- HOST_READING <-- RESPONSE_READY <-- PROVIDER_RUNNING
```

A full ring applies backpressure. Even after `host_consume`, `HOST_READING` blocks reuse until `host_reclaim`. Slot choice is `(request_seq - 1) % 8`; individual slot request generation and output-buffer version increase on each reuse. Host and provider sequences are separately contiguous, and a single provider claim may be running. Exhaustion of request sequence or a generation/version increment retires safe reuse rather than wrapping an identity silently.

## 4. Validation, authentication, and provider bridge

### Startup authentication

`b70_ring_provider` strictly parses one socket path, optional bank and resident-expert list, and required nonzero placement/weight generations and 32-byte SHA-256 values. Resident canonical IDs must be unique and in `[0,255]`. It accepts one seqpacket peer, then requires an exact bootstrap packet, exactly three distinct descriptors, provider PID equal to its own PID, trusted placement/weight identity, an exact sealed regular mapping, two nonblocking close-on-exec eventfds, and proof that the eventfds are distinct.

The provider computes the placement fingerprint as SHA-256 over `SBPLAC01`, resident count, and ordered canonical IDs encoded little-endian. An empty explicit list means all 256 canonical experts. It opens the bank once, hashes the entire file, compares the trusted weight SHA-256, and passes `/proc/self/fd/<verified-fd>` to Phase 1 so the loaded pathname refers to the authenticated open object. Only after `SharedRing::attach`, `B70Provider::load`, and health/allocation checks does it send a successful startup reply.

### Request and payload authentication

`validate_request_descriptor` rejects ABI/version, flags, padding, generation, fingerprint, shape, dtype, slot, offset, extent, or capacity disagreement. The payload validator requires strictly increasing staged-to-full token rows, rows inside the full batch, masks in `{0,1}`, canonical IDs `[0,255]` and finite weights for remote entries, the canonical route position equal to its top-k position, sentinel `0xffff` for nonremote positions, at least one remote route per staged row, exact remote count, and an exact route-subset hash.

The route hash is not a hash of every activation byte. It binds a prefix, layer, full/staged counts, top-k, route count, token row map, every mask and route position, and canonical ID plus exact fp32 weight bits for remote positions. Placement and weight fingerprints separately bind ownership and bank identity.

### Service dispatch

The bridge rechecks claim shapes and canonical routing even after `SharedRing` validation. It rejects nonfinite fp16 activations, missing resident experts, malformed rows/routes, and nonfinite enabled weights. It copies activations into stable host storage, translates enabled canonical IDs to compact provider slots, zeros all disabled local IDs/weights, then calls Phase 1 `issue` and `take` with the ring provider generation and request sequence. Provider statuses map to protocol completion/error pairs: busy and identity/argument mismatches become rejected; shutdown becomes provider-draining; unloaded/device errors become execution-failed. A successful take still requires every output element to be finite.

Normal success marks each remote route and each staged token contributed, keeps nonremote routes `not_remote`, publishes `OK_ALL/NONE`, and includes kernel/total timing. Failure marks remote routes and tokens not contributed and publishes no output extent. At the ring layer `OK_PARTIAL` is explicitly rejected: terminal validation is all-or-nothing. `AMBIGUOUS` is representable in the ABI and requires unknown route/token statuses, but its output remains wholly uncommittable.

The service poll loop treats eventfds only as hints and retries the authoritative claim. It also watches control socket failure. Shutdown is accepted only as an exact empty control command; success requires the provider loaded, nonpending, nonstopped, and at its startup allocation count. A test-fault control is sequence-bound and compiled out unless `SHOOTING_BRAKE_ENABLE_TEST_FAULTS` is defined.

## 5. Deadlines, cancellation, quarantine, and recovery

Cancellation and timeout have deliberately different return labels but identical ownership tiers:

1. In `HOST_WRITING`, the unpublished reservation can be abandoned safely. The selected lane returns to free and the host sequence is rolled back if it still points just beyond this request.
2. In `REQUEST_READY`, the host sets a cancellation/deadline flag. The request remains in the provider's contiguous sequence so no invisible hole is created. Claim returns a valid tombstone that the provider must complete as cancelled/deadline-exceeded.
3. In `PROVIDER_RUNNING` or already `RESPONSE_READY`, the flag quarantines the lane. The provider is allowed to finish so its compute lifetime and slot ownership close normally, but `host_consume` checks the flag after validating the completion and exposes no output.

A malformed or stale completion also exposes no output. Host consume first acquires `HOST_READING`; the host can still reclaim the lane after observing the error. Provider death is different: a claimed lane can no longer be closed safely, so the old shared generation is marked dead. All attached peers acquire that flag, all token operations fail `generation_dead`, and even old-slot reclamation is forbidden. Recovery is construction of a fresh mapping with fresh generations and identity, not mutation or reuse of the dead one.

One implementation nuance is preserved: when `provider_claim` has successfully changed the state and established a valid claim, descriptor, payload, unknown cancellation-bit, cancellation, or deadline errors still leave provider single-flight ownership active. The service loop therefore calls `provider_complete` to close those claims explicitly. Returning the error alone would strand the provider sequence.

## 6. The independent memfd transport probe

`SBP2MEMF` has a 4096-byte header followed by disjoint 1 MiB request and response arenas. Cache-line-separated request and completion publications contain sequence/state plus extents and timing. The only required benchmark sizes are 4 KiB, 8 KiB, 512 KiB, and 1 MiB.

The CUDA host creates and seals the memfd, initializes an exact idle layout, creates eventfds and a seqpacket listener, optionally forks the provider, transfers all three fds by SCM_RIGHTS, receives an acknowledgement, selects exactly an NVIDIA GeForce RTX 5090, CUDA-registers the mapping, and allocates fixed source, destination, and verification buffers. For each iteration it generates deterministic bytes on CUDA, copies device→shared request, release-publishes the request, waits for completion, validates exact metadata, clears the completion, copies shared response→CUDA, and verifies every byte. It reports cold fresh-runtime and warm percentile measurements and uses an explicit sequence-bearing shutdown handshake. Destruction attempts a safe idle shutdown, tears down CUDA resources, and terminates a child if necessary.

The provider validates exact bootstrap metadata and one SCM_RIGHTS record, exact regular read/write memfd extent and grow/shrink seals, layout, initially idle publications, and a unique Intel B70 Level-Zero device. It allocates one fixed B70 buffer. Every request must be `ready` or `shutdown`, use the expected sequence, have zero flags, and fit both disjoint arenas. Data goes shared→B70→shared through dependent profiled SYCL copies. It returns explicit `bad_layout`, `bad_state`, `bad_sequence`, `bad_extent`, or `device_failure` and terminates the session on such errors. It exits successfully only after a valid zero-byte shutdown. Unlike SBRING02, its eventfd reader requires the counter to be exactly one rather than treating coalescing as an ignorable hint.

## 7. Executable scenarios and properties

The model names reachable witnesses for a published request, committed output, quarantine, retirement, and a completed probe round trip. Its scenarios cover:

* authenticated success, output visibility, reclamation, and generation/version advance on reuse;
* malformed host payload rejection before publication and repair while still host-owned;
* pre-claim cancellation tombstone completion;
* post-claim deadline quarantine despite a late provider success;
* stale completion identity and malformed published descriptor failures with no output exposure;
* bootstrap authentication failure blocking service traffic;
* destructive retirement followed only by a fresh generation;
* independent probe success/clean shutdown and invalid-extent failure.

The invariant states the modeled safety boundary: provider busy exactly matches provider ownership; output is visible only while the host owns a validated, unclosed successful completion; retirement hides output; provider sequence never passes host sequence; and free lanes are never provider-owned. The one-slot model intentionally abstracts the eight-way index arithmetic while retaining deterministic ownership, backpressure, reuse identity, and SPSC order.

## 8. Harnesses, build surface, and evidence limits

[`shared_ring_tests.cpp`](../../../src/phase2/shared_ring_tests.cpp) is a CPU/fork protocol harness. It covers exact/disjoint layout, release/acquire visibility, notification hints, malformed bounds/routes, full-ring backpressure, cancellation and timeout quarantine, ambiguous all-or-nothing failure, stale completion rejection, fork visibility, killed-provider replacement, cross-process retirement, and 50,000 iterations by default (optionally two million). Its own banner explicitly says it is protocol-only, not CUDA/B70 acceptance.

[`b70_ring_integration_test.cpp`](../../../src/phase2/b70_ring_integration_test.cpp) is the real provider/ring harness. It checks startup hash negatives, attach-generation/PID negatives, zero-remote bypass, M=1 and duplicate-top8 numerical results, all eight queued slots, live-wrap backpressure, per-slot generation/version advancement, timing consistency, and clean shutdown dispatch/allocation accounting. It depends on the Phase-1 bank, golden fixture, provider implementation, SYCL, and B70 hardware.

[`Makefile`](../../../src/phase2/Makefile) keeps three build groups distinct: CUDA+SYCL copy-probe binaries, host-only shared-ring tests, and the real B70 provider/integration pair. The B70 service compiles the Phase-1 provider source directly and links embedded QuixiCore-XPU libraries. The specification does not infer production deployment merely from a build target.

No command, test, build, formatter, linter, CUDA workload, or B70 workload was run while authoring this chapter. The model collapses byte arrays, SHA-256 calculation, syscalls, release/acquire memory ordering, device copies, kernel mathematics, and wall-clock time into explicit validated booleans or atomic actions. Those abstractions preserve observable ownership, ordering, identity, terminal status, and output exposure but cannot establish C++/ABI equivalence or hardware memory coherence.
The specification also supplies [`ring_correspondence.cpp`](../ring_correspondence.cpp), a focused CPU-only differential harness over the actual `SharedRing` implementation. It does not replace or mock `SharedRing` and does not instantiate the B70 compute provider: it drives both real host-side and provider-side ring API ownership endpoints, supplies a finite successful payload only after a real claim, and asserts source `RingStatus`, slot state, quarantine, stale-identity non-exposure, retirement, and fresh-generation behavior. Its post-claim cancellation case has the same `cancel_flags` quarantine mechanics as the model's `postclaimDeadlineQuarantineTest`; the closure reason differs, so its JSON labels that correspondence as a cancellation variant rather than claiming exact scenario identity.

From repository root, the exact parent-owned CPU compile/link command is:

```sh
g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -pthread -Isrc/phase2 specs/reyu/ring_correspondence.cpp src/phase2/shared_ring.cpp -o /tmp/ring_correspondence
```

Running `/tmp/ring_correspondence` exercises these model correspondences:

| C++ observation | Matching Reyu run |
|---|---|
| publish → claim → successful complete → consume → reclaim | `successAndSafeReuseTest` (through first reclaim) |
| published cancellation tombstone, provider closes it, host sees quarantine | `preclaimCancellationTombstoneTest` |
| post-claim cancellation, late success remains private | `postclaimDeadlineQuarantineTest` (same quarantine tier; cancellation variant) |
| raw completion provider-generation echo is stale and exposes no output | `staleCompletionNeverExposesOutputTest` |
| dead old generation rejects consume/reclaim; fresh generation accepts sequence 1 | `retirementRequiresFreshGenerationTest` |

The harness emits one small JSON object only after all assertions pass. It was authored but not compiled or run here.


## 9. Potential CPU implementation comparisons

The following bounded comparisons can be run without a GPU by the parent verification workflow:

1. Compile and run `specs/reyu/ring_correspondence.cpp` with the exact command above to compare source transitions and output exposure directly with the named Reyu runs. This needs no GPU and uses the real `SharedRing` memfd/eventfd implementation.
2. Build and run only `shared_ring_tests` to extend that comparison to exact layout, malformed publication, full-ring backpressure, fork visibility, and cross-process retirement. This is the broadest CPU behavior comparison in the owned implementation sources.
3. Use a source-level layout inspector against `ring_protocol.hpp` to print `sizeof`, `alignof`, offsets, `kPayloadStride`, and `kMappingBytes`, then compare those values with the chapter and static assertions. This checks ABI constants, not state-machine behavior.
4. Use the existing fork retirement cases to compare shared dead-flag visibility across independent mappings. A unit-of-one in-process check would not establish that cross-process property.
5. Source-inspect `provider_protocol.yaml` beside `ring_protocol.hpp` and `memfd_transport_protocol.hpp` to ensure documentation or inventories do not conflate protocol-v1 logical fields, SBRING02 ABI-v2 fields, and SBP2MEMF probe fields.

The B70 bridge's canonical-to-compact mapping, bank authentication, numerical output, and allocation stability cannot be fully compared by the host-only harness; those remain real integration/hardware checks in `b70_ring_integration_test.cpp` and later Phase-3 coverage.
