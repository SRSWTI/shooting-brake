## Pre-submission Checklist

- [x] I am using the latest GPU driver version available for this distribution (`26.27.39122.12-1~26.04~ppa1`, intel-graphics PPA)
- [x] I have searched for similar issues and found none

## GPU Hardware

Intel Arc Pro B70 (Battlemage G31).

## DRI Devices Information

```
$ ls -l /dev/dri/by-path/ | grep -E '11:00|15:00'
lrwxrwxrwx+1 root root  8 Aug 27 09:51 pci-0000:11:00.0-card -> ../card2
lrwxrwxrwx+1 root root 13 Aug 27 09:51 pci-0000:11:00.0-render -> ../renderD129
lrwxrwxrwx+1 root root  8 Aug 27 09:51 pci-0000:15:00.0-card -> ../card3
lrwxrwxrwx+1 root root 13 Aug 27 09:51 pci-0000:15:00.0-render -> ../renderD130
```

## GPU Detailed Information (lspci output)

```
$ lspci -nn | grep Battlemage
11:00.0 VGA compatible controller [0300]: Intel Corporation Battlemage G31 [Intel Graphics] [8086:e223]
15:00.0 VGA compatible controller [0300]: Intel Corporation Battlemage G31 [Intel Graphics] [8086:e223]
```

## Driver Version

`26.27.39122.12-1~26.04~ppa1` — SYCL device driver version string reports `1.15.39122+12`.

## Installed GPU Driver Packages

```
intel-ocloc                          26.27.39122.12-1~26.04~ppa1
intel-opencl-icd                     26.27.39122.12-1~26.04~ppa1
libigc2                              2.38.3-3~26.04
libigdgmm12:amd64                    22.10.0-1~26.04~ppa1
libze-dev:amd64                      1.28.6-1~26.04~ppa1
libze-intel-gpu-raytracing           1.2.4-1~26.04~ppa3
libze-intel-gpu1                     26.27.39122.12-1~26.04~ppa1
libze1:amd64                         1.28.6-1~26.04~ppa1
```

## Driver Installation Details

Installed from the Intel graphics PPA for Ubuntu 26.04 via `apt`. Kernel-side is
the in-tree `xe` KMD shipped with kernel `7.0.0-30-generic`; no out-of-tree DKMS
module.

## Linux Distribution

Ubuntu 26.04 LTS

## Kernel Version & Boot Parameters

```
$ uname -a
Linux 7.0.0-30-generic #30-Ubuntu SMP PREEMPT_DYNAMIC Fri Jul 31 18:22:54 UTC 2026 x86_64 GNU/Linux

$ cat /proc/cmdline
BOOT_IMAGE=/boot/vmlinuz-7.0.0-30-generic root=UUID=... ro quiet splash \
  crashkernel=2G-4G:320M,4G-32G:512M,32G-64G:1024M,64G-128G:2048M,128G-:4096M
```

KMD in use: `xe`.

## Actual Behavior

`zexCommandListAppendWaitOnMemory` called with a host pointer the driver does not
already know about does not fail. It allocates an internal buffer, `memcpy_s`es
the current contents of the caller's pointer into it, and waits on **the copy**.

Because the wait is satisfied against a snapshot, subsequent host or peer writes
to the original pointer are never observed and the wait never completes. The
append call returns `ZE_RESULT_SUCCESS`, so the only symptom is a hang with no
diagnostic.

The path, at `32a2e4c2b0`:

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
and its contents are a snapshot taken at append time.

## Expected Behavior

For a *wait*, a snapshot copy can never be correct — the entire purpose of the
operation is to observe a future write. Any of the following would be an
improvement, in increasing order of effort:

1. **Document it.** State in the `zex` wait-on-memory documentation that the
   watched pointer must be a driver-known allocation
   (`zeMemAllocHost`/`zeMemAllocShared`, or imported), and that anything else
   yields a wait on a private copy that will not observe external writes.
2. **Warn.** Emit a verbose-log warning when the host-copy fallback is taken on
   a wait, so `ZE_DEBUG`/verbose users can see it.
3. **Reject.** Do not pass `hostCopyAllowed = true` from the wait path at
   `cmdlist_hw.inl:4884`, and return an error for an unregistered pointer. A
   host copy is a reasonable fallback for a *transfer* source, but for a wait
   the semantics are always wrong.

(1) alone would have saved me the debugging time. (3) seems most correct, but I
appreciate it is a behaviour change and there may be callers relying on the
current acceptance.

## Reproduction Rate

Always reproduces - 100%

## Steps to Reproduce

1. `malloc()` a page — or take any pointer not obtained from
   `zeMemAllocHost`/`zeMemAllocShared` and not imported.
2. Call `zexCommandListAppendWaitOnMemory(cmdlist, desc, ptr, expectedValue, ...)`
   on that pointer. Note it returns `ZE_RESULT_SUCCESS`.
3. Close and submit the command list.
4. From the host — or from another device writing that same page — store
   `expectedValue` to `ptr`.
5. The command list never completes.

For contrast, repeating the same sequence with a page from `zeMemAllocHost`
completes normally.

## Is this a regression?

Not known. This is the only driver version I have tested.

## First Known Failing Driver Version

Unknown — first and only version tested: `26.27.39122.12`.

## API Call Logs

The relevant observation is that no error is produced: the append returns
`ZE_RESULT_SUCCESS` and no verbose message is emitted on the host-copy fallback.
That absence is the substance of the report. Happy to capture `ZE_DEBUG` output
if a specific level would show the fallback being taken.

## strace Logs

Not captured — the failure is internal to the driver's allocation resolution,
not a syscall-level error. Can provide on request.

## System Logs / dmesg Output

No `xe` / DRM messages are emitted; the GPU is healthy and simply waiting on a
word that never changes.

## Backtrace (if crash or hang occurred)

The hang is a submitted `MI_SEMAPHORE_WAIT` that never satisfies, so there is no
host-side backtrace to capture — the host thread has already returned from the
append and the submit. The call chain that produces it is the four file:line
references under **Actual Behavior**, which I traced by reading the source
rather than from a live backtrace. I want to be explicit that this is a
source-derived diagnosis plus an observed hang, not a captured stack.

## Source Code / Reproducer

I do not have a minimal standalone reproducer to attach, and I would rather say
so than hand over something I have not run. Demonstrating the hang means
deliberately parking a semaphore wait on the device, which on this setup needs a
device reset to clear, so I stopped short of scripting it.

What I do have is the positive control: the same mechanism works correctly when
the watched page comes from `zeMemAllocHost`, measured over 2000 iterations with
no failures and a ~4.7 us median host round trip. The defect was found when that
page was *not* driver-allocated, and the four-step sequence under **Steps to
Reproduce** is what triggered it.

If a runnable reproducer is required for triage, I am happy to write one — please
confirm that is wanted, given it intentionally wedges a command list.

## Command Line / Application Details

Reproduced inside a cross-vendor producer/consumer program where the Intel
command streamer waits on a doorbell word in pinned host memory written by
another vendor's GPU. Built with `icpx -fsycl -std=c++20`, linked against
`-lze_loader`, run with `SYCL_UR_USE_LEVEL_ZERO_V2=0`.

## oneAPI Version (if applicable)

oneAPI 2026.1 (`/opt/intel/oneapi/compiler/2026.1`).

## Screenshots / Video

Not applicable.

## Additional Notes

Found while building a cross-vendor doorbell: the Intel card's command streamer
waits on a word written by an NVIDIA GPU, replacing a CPU-poller handshake.
Once the page is allocated with `zeMemAllocHost` the mechanism works well
(2000 iterations, no failures, ~4.7 us p50 host round trip). This was the only
rough edge, and it presented as an unexplained hang rather than an error, which
is why I think even the documentation-only fix has real value.

Related: [#983](https://github.com/intel/compute-runtime/issues/983) is a
feature request on the same entry point (a timeout or abort token so a parked
wait is recoverable). Independent of this report, but the two were found
together.
