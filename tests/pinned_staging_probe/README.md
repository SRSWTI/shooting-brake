# Pinned-Staging Probe (scratch test)

Minimal, standalone illustration of the cross-vendor host-staged transport
primitive chain:

```
memfd_create -> mmap(MAP_SHARED) -> cudaHostRegister(...,cudaHostRegisterMapped)
                                  -> Level Zero import of the same fd
```

**This is not the qualified transport.** The production-qualified,
hardware-validated version of this exact chain is
`src/phase2/memfd_transport_host.cu` + `src/phase2/memfd_transport_provider.cpp`,
which already passed the Phase-2 gate on the physical RTX 5090 + Arc Pro B70
(see `docs/progress.md`, "Phase 2 process-ring evidence") with a real eight-slot
ring, stale-completion rejection, wraparound stress, and warmed p50/p95/p99
latencies. Use that for anything that needs to be cited as evidence.

This folder exists only to let you compile and eyeball the two-hop
(GPU -> host DRAM -> GPU) data path in isolation, with inline profiling, before
touching the real ring protocol. It intentionally has no retry, backpressure,
or failure handling — do not lift it into `src/phase4/` as-is.

## Files

- `host_side.cu` — allocates a `memfd`, `mmap`s it `MAP_SHARED`, registers it
  with CUDA via `cudaHostRegister(..., cudaHostRegisterMapped)`, writes a
  known pattern from a device kernel via `cudaMemcpyAsync`, and passes the fd
  to the provider process over a `SCM_RIGHTS` unix-socket message (same
  handoff mechanism as `src/phase2/memfd_transport_host.cu`, simplified).
- `provider_side.cpp` — receives the fd, `mmap`s the same pages directly (no
  Level Zero import step — see "Verified result" below), copies them to a
  real B70 device allocation and back with `zeCommandListAppendMemoryCopy`,
  and reports timing.

## Build

Requires `nvcc` (CUDA toolkit) for `host_side.cu` and Level Zero
(`ze_api.h`, link `-lze_loader`) for `provider_side.cpp`. Neither toolchain
is assumed present in every environment this repo is checked out in — build
failures here mean "toolchain absent," not "chain is wrong."

`host_side.cu` must be built for the actual GPU's SASS architecture, not
the compiler's default virtual target — on a driver/toolkit combination
where they diverge, `fill_pattern<<<>>>` launches silently no-op (caught by
the `cudaGetLastError()` check after the launch), and the rest of the
pipeline "succeeds" while faithfully copying zeros end to end. Find the
right value with `nvidia-smi --query-gpu=compute_cap --format=csv`
(`12.0` -> `sm_120`, `9.0` -> `sm_90`, etc.):

```sh
nvcc -O2 -std=c++17 -arch=sm_120 host_side.cu -o host_side   # match your GPU
g++  -O2 -std=c++17 provider_side.cpp -o provider_side -lze_loader
```

## Run

```sh
./provider_side --socket /tmp/pinned-staging-probe.sock &
./host_side --socket /tmp/pinned-staging-probe.sock
```

Prints per-hop timing: `cudaHostRegister`, GPU->pinned-host write, socket
handoff, plain mmap of the shared fd, host(mmap)->B70 copy, B70->host
copy, and an end-to-end pattern verification.

## Verified result (real RTX 5090 + real Arc Pro B70, this repo's dev box)

Two real bugs were found and fixed getting this to actually pass, in order:

1. **Level Zero import is a dead end for an arbitrary `memfd`.**
   `zeMemAllocHost` + `ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_FD` (and
   `_DMA_BUF`) reject a plain anonymous `memfd_create` fd with
   `ZE_RESULT_ERROR_INVALID_ARGUMENT` — that import path expects a handle a
   Level-Zero/dma-buf-aware exporter produced, not an arbitrary shmem fd.
   Fix: skip import entirely. Plain `mmap(MAP_SHARED)` the fd in the
   provider process and hand that pointer straight to
   `zeCommandListAppendMemoryCopy` as the copy source — Level Zero walks
   any resident host virtual address itself, exactly like
   `src/phase2/memfd_transport_provider.cpp:368-407` already does via SYCL
   `queue.memcpy()`.
2. **`cudaHostRegisterMapped` + `cudaHostGetDevicePointer` breaks
   cross-process page identity.** Requesting a separate device-mapped
   pointer diverges from the `mmap`'d memfd's actual page-cache pages once
   a second process maps the same fd fresh. Fix: match
   `src/phase2/memfd_transport_host.cu:640-641` exactly —
   `cudaHostRegisterPortable`, and pass the plain host pointer (`shared`,
   not a separate device pointer) straight into `cudaMemcpyAsync`; UVA
   makes that valid directly.

Neither of those was actually the transport's fault — the real first
failure was (3) a silent CUDA kernel-launch no-op from an `nvcc`
virtual-arch/driver mismatch, caught only by adding `cudaGetLastError()`
after the launch and fixed by `-arch=sm_120`. After all three fixes, one
run:

```
[host] cudaHostRegister: 139.7 ms   (one-time pin cost, 4 MiB)
[host] GPU->pinned-host write: 4.3 ms for 4 MiB (cold, unwarmed)
[host] fd handoff + provider ack: 19.1 ms
[provider] plain mmap of shared fd: 0.02 ms
[provider] host(mmap)->B70 copy: 4 MiB in a few ms (first-touch, not steady state)
[provider] B70->host copy: 4 MiB, 3.5 GB/s
[provider] end-to-end pattern verify: PASS
```

Single cold run, 4 MiB, no warmup loop — not a benchmark. For real
percentiles under repetition, backpressure, and failure injection, use the
qualified `phase2` ring (`docs/progress.md`, "Phase 2 process-ring
evidence").
