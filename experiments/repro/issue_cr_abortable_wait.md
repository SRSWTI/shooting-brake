# Summary

`zexCommandListAppendWaitOnMemory` emits an Xe2 `MI_SEMAPHORE_WAIT` in polling
mode with no timeout and no abort path, and `zeCommandQueueDestroy` neither waits
for nor cancels an outstanding wait. If the value being waited on never arrives,
the only recovery is a device reset, or writing the semaphore value from the host
yourself — which corrupts whatever protocol the word belonged to.

That is fine when the producer is another engine on the same device. It is not
fine when the producer is outside the driver's control. I'd like to request a
timeout parameter or abort-token variant, or documented guidance for safe
recovery.

Filing here rather than against `oneapi-src/level-zero` because
`zexCommandListAppendWaitOnMemory` is an Intel driver experimental extension
implemented in this repo
(`level_zero/api/driver_experimental/public/zex_cmdlist.cpp:16-30`, exposed via
`level_zero/core/source/driver/extension_function_address.cpp:82-83`) and does
not exist in the loader/spec repository. If you'd rather see this as a
vendor-neutral API it presumably belongs in the specification process instead —
happy to take it there.

# Use case, and why the stall is not hypothetical

I use this extension to build a cross-vendor doorbell: an NVIDIA GPU (or the
host) writes a word in pinned memory, and the Intel card's command streamer
waits on that word from a pre-recorded command list, so no CPU thread sits in the
per-layer critical path. It replaces a host-poller handshake and it works well.

Measured on 2x Arc Pro B70 (Battlemage G31), pre-recorded WAIT -> WRITE list,
2000 iterations per configuration, no failures:

| producer | device | p50 | min | p90 | p99 |
|---|---|---|---|---|---|
| host store | `0000:15:00.0` | 4.70 us | 3.54 us | 5.87 us | 7.68 us |
| host store | `0000:11:00.0` | 5.54 us | 4.09 us | 6.00 us | 7.16 us |
| foreign GPU (RTX 5090) | `0000:11:00.0` | 9.38 us | 6.73 us | 11.32 us | 74.96 us |
| foreign GPU (RTX 5090) | `0000:15:00.0` | 12.32 us | 5.93 us | **2095 us** | **2234 us** |

The mechanism is a large win — the host-poller handshake it replaces costs
~61 us. But look at the last row. When the producing GPU was under unrelated
load, the p90/p99 of the round trip went to ~2.1-2.2 ms, three orders of
magnitude past p50. Nothing is wrong on the Intel side there; the foreign
producer simply got scheduled late.

That is the shape of the problem: once the producer is not something the Level
Zero driver schedules, "late" has no bound, and "never" is a real outcome
(process killed, foreign driver reset, a bug in my own producer). Today each of
those turns a parked command list into a device reset.

# Evidence in this repo (at `32a2e4c2b0`)

**The wait is polling-mode with nothing encoded to bound it.**
`level_zero/core/source/cmdlist/cmdlist_hw.inl:4921` emits
`addMiSemaphoreWaitCommand(...)`; the generic wrapper hardwires wait mode at
`shared/source/command_container/command_encoder.inl:923`; and the Xe2
programmer maps it to polling at
`shared/source/command_container/command_encoder_from_xe_hpg_core_to_xe3_core.inl:46-47`
(`setWaitMode(... WAIT_MODE_POLLING_MODE ...)`, `setRegisterPollMode(...
MEMORY_POLL)`).

**The Xe2 command has no field that could express a timeout.** The generated
structure at
`shared/source/generated/xe2_hpg_core/hw_cmds_generated_xe2_hpg_core.inl:6461-6556`
has exactly these non-reserved fields: `DwordLength`, `CompareOperation`,
`WaitMode`, `RegisterPollMode`, `IndirectSemaphoreDataDword`,
`WorkloadPartitionIdOffsetEnable`, `MemoryType`, `MiCommandOpcode`,
`CommandType`, `SemaphoreDataDword`, `SemaphoreAddress`, `WaitTokenNumber`. No
timeout, no abort, no switch-on-unsuccessful.

Related: the generic encoder does take a `switchOnUnsuccessful` argument, and the
Xe2 implementation accepts but does not use it. If that is a deliberate gap
rather than an oversight, knowing why would be useful.

**Destroy does not cancel.**
`level_zero/core/source/cmdqueue/cmdqueue.cpp:69-88` — `CommandQueue::destroy()`
unregisters the CSR client, destroys its buffers, notifies the debugger, and
`delete this`. There is no synchronization, no pending-wait cancellation, and no
rewrite of an already-submitted wait command.

**The machinery for a bounded loop already exists.** Conditional
batch-buffer-start is already emitted for relaxed in-order dependencies, in two
places: `level_zero/core/source/cmdlist/cmdlist_hw.inl:3712-3714`
(`programConditionalDataMemBatchBufferStart(... NEO::CompareOperation::less,
true, ...)`) and `:5251-5254` (the event variant). So a poll-and-branch on an
abort token looks like it would be expressible with primitives the driver
already emits, rather than needing new hardware capability.

# Ask, in preference order

1. A timeout on the wait — e.g. a `zexCommandListAppendWaitOnMemory` variant
   taking a deadline, implemented as a conditional-BB-start loop over a
   timestamp, with a defined completion status when it expires.
2. An abort-token variant: a second address that, when set, makes the wait fall
   through. That is enough to make the wait cancellable from the host without
   corrupting the protocol word, and it looks close to what conditional BB start
   already does.
3. Failing either, documented guidance on safe recovery from a parked
   `MI_SEMAPHORE_WAIT` — specifically whether there is any supported way to
   abandon a submitted command list without a device reset, and what the
   intended lifetime contract is when a queue is destroyed with a wait
   outstanding.

Even (3) alone would help. Right now the documented behaviour of destroy does
not say that a submitted wait outlives the queue, and that is a surprising thing
to discover from a hang.

# Environment

- 2x Arc Pro B70 (`Intel Corporation Battlemage G31 [8086:e223]`), driver `1.15.39122+12`
- NEO / compute-runtime `26.27.39122.12`, Level Zero loader `libze1 1.28.6`
- oneAPI 2026.1, kernel `7.0.0-30-generic`, Ubuntu 26.04 LTS, `xe` KMD
- `SYCL_UR_USE_LEVEL_ZERO_V2=0`
