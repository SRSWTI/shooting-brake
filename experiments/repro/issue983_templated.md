## Pre-submission Checklist

- [x] I have searched for similar feature requests and found none
- [x] This feature request includes a clear description of the proposed functionality

## Feature Title

Timeout or abort-token variant of `zexCommandListAppendWaitOnMemory`

## Feature Category

API enhancement — Level Zero (`zex` experimental extension)

## GPU Hardware

Intel Arc Pro B70 (Battlemage G31), 2x installed.

## Operating System

Linux — Ubuntu 26.04 LTS, kernel `7.0.0-30-generic`, `xe` KMD.

## API

Level Zero (specifically the `zex` driver-experimental extension implemented in
this repository at
`level_zero/api/driver_experimental/public/zex_cmdlist.cpp:16-30`, exposed via
`level_zero/core/source/driver/extension_function_address.cpp:82-83`).

## Problem Statement

`zexCommandListAppendWaitOnMemory` emits an Xe2 `MI_SEMAPHORE_WAIT` in polling
mode with no timeout and no abort path, and `zeCommandQueueDestroy` neither waits
for nor cancels an outstanding wait. If the value being waited on never arrives,
the only recovery is a device reset, or writing the semaphore value from the host
yourself — which corrupts whatever protocol the word belonged to.

That is fine when the producer is another engine on the same device. It is not
fine when the producer is outside the driver's control.

**Evidence in this repository, at `32a2e4c2b0`:**

*The wait is polling-mode with nothing encoded to bound it.*
`level_zero/core/source/cmdlist/cmdlist_hw.inl:4921` emits
`addMiSemaphoreWaitCommand(...)`; the generic wrapper hardwires wait mode at
`shared/source/command_container/command_encoder.inl:923`; and the Xe2 programmer
maps it to polling at
`shared/source/command_container/command_encoder_from_xe_hpg_core_to_xe3_core.inl:46-47`
(`setWaitMode(... WAIT_MODE_POLLING_MODE ...)`, `setRegisterPollMode(... MEMORY_POLL)`).

*The Xe2 command has no field that could express a timeout.* The generated
structure at
`shared/source/generated/xe2_hpg_core/hw_cmds_generated_xe2_hpg_core.inl:6461-6556`
has exactly these non-reserved fields: `DwordLength`, `CompareOperation`,
`WaitMode`, `RegisterPollMode`, `IndirectSemaphoreDataDword`,
`WorkloadPartitionIdOffsetEnable`, `MemoryType`, `MiCommandOpcode`, `CommandType`,
`SemaphoreDataDword`, `SemaphoreAddress`, `WaitTokenNumber`. No timeout, no abort,
no switch-on-unsuccessful.

Related detail: the generic encoder does take a `switchOnUnsuccessful` argument,
and the Xe2 implementation accepts but does not use it. If that is a deliberate
gap rather than an oversight, knowing why would be useful.

*Destroy does not cancel.*
`level_zero/core/source/cmdqueue/cmdqueue.cpp:69-88` — `CommandQueue::destroy()`
unregisters the CSR client, destroys its buffers, notifies the debugger, and
`delete this`. There is no synchronization, no pending-wait cancellation, and no
rewrite of an already-submitted wait command.

## Proposed Solution

In order of preference:

1. **A timeout on the wait** — e.g. a `zexCommandListAppendWaitOnMemory` variant
   taking a deadline, implemented as a conditional-batch-buffer-start loop over a
   timestamp, with a defined completion status when it expires.
2. **An abort-token variant** — a second address that, when set, makes the wait
   fall through. That is enough to make the wait cancellable from the host
   without corrupting the protocol word.
3. **Failing either, documented guidance** on safe recovery from a parked
   `MI_SEMAPHORE_WAIT`: specifically whether there is any supported way to
   abandon a submitted command list without a device reset, and what the intended
   lifetime contract is when a queue is destroyed with a wait outstanding.

**The machinery for (1) or (2) appears to already exist.** Conditional
batch-buffer-start is already emitted for relaxed in-order dependencies, in two
places: `level_zero/core/source/cmdlist/cmdlist_hw.inl:3712-3714`
(`programConditionalDataMemBatchBufferStart(... NEO::CompareOperation::less, true, ...)`)
and `:5251-5254` (the event variant). So a poll-and-branch on a deadline or an
abort token looks expressible with primitives the driver already emits, rather
than needing new hardware capability.

## Expected Benefits

- A stalled or dead external producer becomes a recoverable error instead of a
  device reset.
- Cross-vendor and cross-process producer/consumer designs become safe to ship;
  today the failure mode is unbounded.
- Even option (3) — documentation only — would help: the current behaviour of
  `zeCommandQueueDestroy` with a wait outstanding is surprising to discover from
  a hang.

## Use Case Scenarios

I use this extension to build a cross-vendor doorbell: an NVIDIA GPU (or the
host) writes a word in pinned memory, and the Intel card's command streamer waits
on that word from a pre-recorded command list, so no CPU thread sits in the
per-layer critical path. It replaces a host-poller handshake and it works well.

Measured on 2x Arc Pro B70, pre-recorded WAIT -> WRITE list, 2000 iterations per
configuration, no failures:

| producer | device | p50 | min | p90 | p99 |
|---|---|---|---|---|---|
| host store | `0000:15:00.0` | 4.70 us | 3.54 us | 5.87 us | 7.68 us |
| host store | `0000:11:00.0` | 5.54 us | 4.09 us | 6.00 us | 7.16 us |
| foreign GPU (RTX 5090) | `0000:11:00.0` | 9.38 us | 6.73 us | 11.32 us | 74.96 us |
| foreign GPU (RTX 5090) | `0000:15:00.0` | 12.32 us | 5.93 us | **2095 us** | **2234 us** |

The mechanism is a large win — the host-poller handshake it replaces costs
~61 us. But the last row is the point of this request. When the producing GPU was
under unrelated load, p90/p99 of the round trip went to ~2.1–2.2 ms, three orders
of magnitude past p50. Nothing is wrong on the Intel side there; the foreign
producer simply got scheduled late.

That is the shape of the problem: once the producer is not something Level Zero
schedules, "late" has no bound, and "never" is a real outcome (process killed,
foreign driver reset, a bug in my own producer). Today each of those turns a
parked command list into a device reset.

## Code Examples / API Design

Sketch of (2), the abort-token form, which seems the smaller change:

```c
// Existing:
ze_result_t zexCommandListAppendWaitOnMemory(
    ze_command_list_handle_t hCmdList,
    zex_wait_on_mem_desc_t  *desc,
    void                    *ptr,
    uint32_t                 data,
    ze_event_handle_t        hSignalEvent);

// Proposed variant: fall through if *pAbort becomes non-zero.
ze_result_t zexCommandListAppendWaitOnMemoryAbortable(
    ze_command_list_handle_t hCmdList,
    zex_wait_on_mem_desc_t  *desc,
    void                    *ptr,
    uint32_t                 data,
    void                    *pAbort,        // driver-known allocation
    uint32_t                 abortData,
    ze_event_handle_t        hSignalEvent);
```

Host-side recovery then becomes a store to `pAbort` rather than a device reset,
and the completion status distinguishes "satisfied" from "aborted".

A timeout form would suit callers who do not want to manage a second allocation:

```c
ze_result_t zexCommandListAppendWaitOnMemoryWithTimeout(
    ze_command_list_handle_t hCmdList,
    zex_wait_on_mem_desc_t  *desc,
    void                    *ptr,
    uint32_t                 data,
    uint64_t                 timeoutNs,
    ze_event_handle_t        hSignalEvent);
```

## Alternatives & Workarounds

- **Host-poller handshake** (what this replaces): a pinned host thread spins on
  the word and then submits. Recoverable, but costs ~61 us versus ~4.7 us, and
  puts a CPU thread in the per-layer critical path.
- **Write the semaphore value from the host to unstick it.** Works mechanically,
  but the word is protocol state shared with the foreign producer, so forging it
  corrupts the protocol — the consumer proceeds as though data arrived when it
  did not.
- **Device reset.** Current only clean recovery. Too coarse for a serving
  process.
- **Bound the producer instead.** Not possible when the producer is another
  vendor's driver under independent scheduling, which is exactly the case here.

## Priority / Importance

Medium - Nice to have enhancement

I want to be straight about this rather than inflate it: the mechanism works today
and is a large improvement over the alternative, so this is not blocking me. It is
a robustness gap that makes the design hard to ship to anyone else, and option (3)
(documentation) would already reduce the surprise substantially.

## Estimated Impact

Affects any user driving `zexCommandListAppendWaitOnMemory` from a producer the
Level Zero driver does not schedule — cross-vendor, cross-process, or
host-thread producers that can stall. Narrow audience today, but it is the
enabling primitive for heterogeneous multi-vendor pipelines, and the current
failure mode (unrecoverable without a device reset) is the main thing standing
between "works on my rig" and "safe to ship".

## References

- `level_zero/api/driver_experimental/public/zex_cmdlist.cpp:16-30` — extension implementation
- `level_zero/core/source/driver/extension_function_address.cpp:82-83` — exposure
- `level_zero/core/source/cmdlist/cmdlist_hw.inl:4921` — wait emission
- `shared/source/command_container/command_encoder.inl:923` — wait-mode hardwired
- `shared/source/command_container/command_encoder_from_xe_hpg_core_to_xe3_core.inl:46-47` — Xe2 polling mode
- `shared/source/generated/xe2_hpg_core/hw_cmds_generated_xe2_hpg_core.inl:6461-6556` — Xe2 `MI_SEMAPHORE_WAIT` fields (no timeout/abort)
- `level_zero/core/source/cmdqueue/cmdqueue.cpp:69-88` — destroy does not cancel
- `level_zero/core/source/cmdlist/cmdlist_hw.inl:3712-3714`, `:5251-5254` — conditional BB-start already emitted

## Additional Notes

Filed here rather than against `oneapi-src/level-zero` because
`zexCommandListAppendWaitOnMemory` is an Intel driver experimental extension
implemented in this repository and does not exist in the loader/spec repository —
I checked before filing. If you would rather see this as a vendor-neutral API it
presumably belongs in the specification process instead; happy to take it there.

Related: [#982](https://github.com/intel/compute-runtime/issues/982) is a
separate rough edge on the same entry point (an unregistered host pointer is
silently waited on via an internal copy rather than rejected). Independent of
this request, but the two were found together.
