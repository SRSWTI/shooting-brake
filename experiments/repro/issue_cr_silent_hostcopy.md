# Summary

`zexCommandListAppendWaitOnMemory` called with a host pointer the driver does not
already know about does not fail. It silently allocates an internal buffer,
`memcpy_s`es the current contents of your pointer into it, and waits on **the
copy**. Subsequent host or peer writes to the original pointer are never
observed, so the wait never completes.

The failure mode is a hang with a success return code and no diagnostic. It cost
me a while to find, and the fix could be as small as a doc sentence.

# Version

- NEO / compute-runtime `26.27.39122.12` (`libze-intel-gpu1 26.27.39122.12-1~26.04~ppa1`)
- Level Zero loader `libze1 1.28.6-1~26.04~ppa1`
- Source references below are against this repo at `32a2e4c2b0`.

# Environment

- 2x Arc Pro B70 (`Intel Corporation Battlemage G31 [8086:e223]`), driver `1.15.39122+12`
- oneAPI 2026.1, kernel `7.0.0-30-generic`, Ubuntu 26.04 LTS, `xe` KMD

# Steps to reproduce

1. `malloc()` a page (or take any pointer not obtained from
   `zeMemAllocHost`/`zeMemAllocShared` and not imported).
2. `zexCommandListAppendWaitOnMemory(cmdlist, desc, ptr, expectedValue, ...)` on
   that pointer, close and submit the list.
3. From the host — or from another device writing that same page — store
   `expectedValue` to `ptr`.
4. The command list never completes. The append in step 2 returned
   `ZE_RESULT_SUCCESS`.

# Observed behavior

The wait is satisfied against an internal copy rather than the caller's memory.
The path is:

- `level_zero/core/source/cmdlist/cmdlist_hw.inl:4884` — the wait resolves its
  address with host copy explicitly permitted:
  `resolveAlignedAllocation(..., {.hostCopyAllowed = true});`
- `level_zero/core/source/cmdlist/cmdlist_hw.inl:3349-3352` — an unregistered
  pointer falls through to `getHostPtrAlloc(... flags.hostCopyAllowed ...)`.
- `level_zero/core/source/device/device.cpp:1816-1820` — if external-host
  allocation fails, it falls back:
  `if (allocation == nullptr && hostCopyAllowed) { allocation = ...allocateInternalGraphicsMemoryWithHostCopy(...) }`
- `shared/source/memory_manager/memory_manager.cpp:880-882` — the copy itself:
  `memcpy_s(allocation->getUnderlyingBuffer(), allocation->getUnderlyingBufferSize(), ptr, size);`

The fallback returns a valid allocation, so the caller sees success. The
semaphore address programmed into `MI_SEMAPHORE_WAIT` is the internal buffer's,
and it is a snapshot taken at append time.

# Expected behavior

For a *wait*, a snapshot copy can never be correct — the entire point of the
operation is to observe a future write. Any of these would be an improvement,
roughly in increasing order of effort:

1. **Document it.** State in the `zex` wait-on-memory documentation that the
   watched pointer must be a driver-known allocation
   (`zeMemAllocHost`/`zeMemAllocShared`, or imported), and that passing anything
   else yields a wait on a private copy that will not observe external writes.
2. **Warn.** Emit a verbose-log warning when the host-copy fallback is taken on
   a wait, so `ZE_DEBUG`/verbose users see it.
3. **Reject.** Do not pass `hostCopyAllowed = true` from the wait path at
   `cmdlist_hw.inl:4884`, and return an error for an unregistered pointer.
   A host copy is a sensible fallback for a *transfer* source, but for a wait
   the semantics are always wrong.

I'd be happy with (1). (3) seems most correct to me, but I appreciate it is a
behaviour change and you may have callers relying on the current acceptance.

# Note

Found while building a cross-vendor doorbell where the Intel command streamer
waits on a word written by another vendor's GPU. Once the page is allocated with
`zeMemAllocHost` the mechanism works well — 2000 iterations, no failures, ~4.7 us
p50 host round trip. The only rough edge was this one, and it presented as an
unexplained hang rather than an error.

Related feature request about bounding/cancelling these waits: see my other
issue on a timeout or abort token for the same entry point.
