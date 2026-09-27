# Hardware probes and transport experiments

The [executable model](../models/hardware_probes.ryu) describes the repository's qualification, kill-bench, and reproduction programs. These programs are **not** one alternate serving subsystem. They ask narrower questions at boundaries that production depends on: can a host page be observed coherently, can a foreign device allocation be imported, do two queues overlap, does a fixed command chain preserve order, does a real provider result match an oracle, and does a runtime configuration reproduce a failure?

## 1. Branch-level design

```text
fixture / bank / captured input
            |
      runtime + device admission
            |
  allocate and establish ownership
     /          |             \
host page    device USM    exported device allocation
register/map  queue/list    fd or dma-buf import
     \          |             /
        publish / submit / ring
                  |
      wait, event, flag, sequence, or queue sync
                  |
      observe bytes, parity, health, or timing
                  |
          classify and release
```

The actors are a controller process, CUDA, Level Zero/SYCL or PyTorch XPU, sometimes a native provider/poller, and occasionally a second process. Ownership remains explicit:

* CUDA-pinned and memfd workflows keep page lifetime at the host/controller while the provider obtains a second mapping or fd reference.
* SYCL/Level Zero workflows tie USM and native objects to a device context or queue.
* dma-buf workflows export a device allocation and import the fd into the other vendor runtime; process exit, rather than disciplined teardown, is the lifetime boundary in both cross-vendor programs.
* provider workflows own a bank and native provider until shutdown; dual-card workflows own one disjoint lane per B70.
* worker prototypes leave shared-memory creation/unlinking to the controller and use sequence words as their only handoff protocol.

Completion has different meanings. A queue wait proves submitted work retired. A completion word proves the consumer reached its publication step. Neither proves numerical validity unless bytes, finiteness, health, or an oracle are checked separately. The model therefore keeps `payloadObserved`, `oracleCompared`, `numericalOk`, and `orderingOk` distinct.

## 2. Executable state and operations

| Model operation | Concrete responsibility | Source families |
|---|---|---|
| `choose`, `admitRuntime`, `refuseRuntime` | Select a mechanism and represent device/runtime discovery independently from experiment success | every program; notably no-device and wrong-backend branches |
| `registerMappedHost`, `publishHostFlag`, `acknowledgeHostFlag` | Pin/map cache-line-separated words, publish a sequence, and observe a coherent acknowledgement | B13 flag/sweep, B70 resident-hostflag |
| `registerPinnedMapping`, `producePinnedPayload`, `consumePinnedPayload` | Create/pin a shared host mapping, complete CUDA production, transfer the fd, copy through Level Zero, optionally validate | pinned-staging pair and early transport programs |
| `exportDeviceFd`, `importDeviceFd`, `copyExternalPattern` | Export device memory, import it into the foreign runtime, and require full-pattern provenance | RTX-to-B70 and B70-to-RTX dma-buf probes |
| `submitQueueLanes`, `completeQueueLanes` | Distinguish submitted lanes, completed lanes, endpoint integrity and measured overlap | PCIe, topology, dual-queue, fused-H2D |
| `appendCommandChain`, `ringCommand`, `commandCompletes` | Establish a hardware wait/write chain, ring only after submission, then observe completion after kernels/copies | zex doorbell, SYCL immediate interop, baked chains |
| `loadProvider`, `issueProvider`, `takeProvider`, `compareOracle` | Admit bank/geometry, dispatch bounded rows, retrieve finite output, and compare the same fixture | pilot, dual-card, INT4 and graph/provider probes |
| `publishWorkerSlot`, `workerCompletes` | Publish a shared-memory sequence and wait for the worker's matching completion | early Python worker prototypes |
| `captureAttention` | Check before graph capture, replay a captured backend, time it, and check again | isolated decode-attention probe |
| `retainTouchedAllocations` | Retain device allocations and force backing before observing runtime or Linux memory effects | Level Zero v2 crash and host-shadow reproducers |
| `runFormatKernel` | Separate timed format execution from output/oracle validation | NVFP4/MXFP4/FP16 comparison and focused INT4 path |
| `assess`, `recordDiagnostic`, `release` | Classify strict evidence versus diagnostic-only output and release modeled ownership | common experiment boundary |

A `Passed` verdict is intentionally unavailable to timing-only branches unless the requested observation exists. This preserves source contradictions: several executables print `PASSED`, `done`, or return zero without enforcing their prose claim.

## 3. Host registration, pinned staging, and early bounce transport

[The B13 flag probe](../../../experiments/b13_doorbell_flag_probe.cpp) captures CUDA `WRITE(signal) -> WAIT(completion)` over one mapped pinned allocation. The CPU spins for the signal, writes completion, and synchronizes the stream. Its five-second signal timeout breaks measurement but returns zero after cleanup; `CAN_MAP_HOST_MEMORY` is printed rather than enforced. [The sweep probe](../../../experiments/b13_sweep_probe.cpp) instead synchronizes the GPU write before scanning independent pinned pages, so it measures host cache/TLB/linear-scan cost, not concurrent flag-detection latency. [The wait probe](../../../experiments/b13_wait_probe.cpp) compares blocking `event::wait`, SYCL-status polling, and native `zeEventQueryStatus` polling around identical H2D-kernel-D2H work; native extraction failure only omits that arm.

[The resident host-flag probe](../../../experiments/b70_hostflag_spin_probe.cpp) reverses ownership: a B70 kernel persistently polls host USM through uncached loads and acknowledges with system-scope atomics. The x86 publisher uses release stores, `clflush`, `sfence`, and `pause`. Idle and saturating-kernel arms distinguish coherence from forward progress under contention. A null busy scratch silently weakens the contended arm.

The cold scratch pair is [host_side.cu](../../../tests/pinned_staging_probe/host_side.cu) and [provider_side.cpp](../../../tests/pinned_staging_probe/provider_side.cpp). The host creates a memfd, maps it, uses `cudaHostRegisterPortable` on the plain pointer, fills device memory, completes D2H, and passes the fd by `SCM_RIGHTS`. The provider uses ordinary `mmap`, not Level Zero external-memory import, performs H2D and D2H, sparsely checks page-start bytes, and acknowledges. The host ignores ack receive status/value, so provider failure can still look successful locally.

The warmed pair is [host_bench.cu](../../../tests/pinned_staging_probe/host_bench.cu) and [provider_bench.cpp](../../../tests/pinned_staging_probe/provider_bench.cpp). One memfd plus go/done eventfds is transferred once and reused under a fixed compile-time schedule. The full-route interval includes CUDA fill/D2H, process scheduling, Level Zero H2D/D2H, and both eventfd handshakes. Neither side validates payload bytes. The provider's supposed private H2D/D2H reference actually submits H2D in both named event paths. These files are scratch illustrations: they have no retry, backpressure, negotiated schedule, timeout, failure injection, or production ring semantics.

The older bounce implementations make the same architectural boundary explicit. [transport_test.c](../../../experiments/transport_test.c) resets/closes/executes a regular list for each CUDA-D2H -> pinned host -> B70-H2D/D2H -> CUDA-H2D round trip. [transport_test_v2.c](../../../experiments/transport_test_v2.c) uses an immediate list and prefers a copy queue. [transport_debug.c](../../../experiments/transport_debug.c) reduces the question to one 4-KiB CUDA-pinned-to-Level-Zero copy. [transport_debug2.c](../../../experiments/transport_debug2.c) restores the full immediate-list bounce and per-leg timing. None validates round-trip content; some sources are uninitialized, and fixed-size enumeration arrays can overflow when runtimes return more entries than expected.

## 4. fd/dma-buf and direct cross-vendor paths

[rbar_dmabuf_probe.cpp](../../../experiments/rbar_dmabuf_probe.cpp) allocates 64 MiB on a device named 5090, exports it with `cuMemGetHandleForAddressRange`, requires a B70 that advertises dma-buf import, imports the fd with `zeMemAllocDevice`, then makes B70 write into imported RTX memory. CUDA reads the original allocation and every word must match before timing begins. It compares copy- and compute-engine BAR writes against pinned-host writes and reports stage-labelled `KILLED` or `SURVIVED` JSON. Page misalignment has no VMM fallback; only the fd is explicitly closed on success.

[xvendor_p2p_probe.cpp](../../../experiments/xvendor_p2p_probe.cpp) tests the opposite direction. Level Zero exports a 32-MiB B70 allocation, CUDA imports the foreign fd as opaque external memory and pulls it. Every word must retain the B70 pattern. A rejected CUDA import is a specific killed outcome, not evidence that the reverse direction also fails. The pinned-host H2D arm is the CPU-memory second-hop comparator. Resources otherwise rely on process exit.

[The path named b70_b70_dmabuf_control.cpp](../../../experiments/b70_b70_dmabuf_control.cpp) contradicts its suffix and apparent purpose: the tracked object is entirely NUL bytes and contains no C++ symbols or executable behavior. It is inventoried as an artifact, not modeled as a dma-buf control program.

## 5. PCIe, queue, memory, and topology qualification

[b70_pcie_bw.cpp](../../../experiments/b70_pcie_bw.cpp) measures synchronous pinned-host H2D/D2H at one size or a size sweep and classifies the best direction into link bands. Device selection is generic, and zero/negative iterations can invalidate statistics. [b5_fused_h2d_probe.cpp](../../../experiments/b5_fused_h2d_probe.cpp) compares three production-shaped small H2D submissions with one fused record, then forces a kernel/D2H tail. It does not validate copied bytes and declares the idea killed when mean savings are below two microseconds.

[b70_dual_queue_probe.cpp](../../../experiments/b70_dual_queue_probe.cpp) compares a serial in-order queue, two independent in-order queues, and an out-of-order queue with explicit dependencies. It checks first/last payload values for lane crossing, then calls overlap confirmed only above a 1.15 speed ratio. Work-buffer contents are intentionally timing load and are not validated.

[b70_multi_topology.cpp](../../../experiments/b70_multi_topology.cpp) asks three separate questions: solo per-card H2D asymmetry, two-thread shared-uplink contention, and peer accessibility. Direct peer copy is opt-in with argument `p2p`; the safe default only queries capability and measures a host-mediated `device A -> host -> device B` baseline. Fewer than two devices exits zero without a test. [b70_mem_topology_probe.cpp](../../../experiments/b70_mem_topology_probe.cpp) adds per-card/concurrent bandwidth, retained and touched device allocations, `/proc/meminfo`, optional Intel free-memory reporting, and a pageable CPU allocate-touch-read baseline. Its combined all-card cleanup can pair a block with the wrong queue after an earlier card allocation failure.

## 6. Command-streamer and ABI experiments

[b70_cs_doorbell_probe.cpp](../../../experiments/b70_cs_doorbell_probe.cpp) builds a closed regular Level Zero list `WAIT(A==1) -> WRITE(B=1)`. The host arm submits before a release-store ring. If the expected zex action/scope is rejected, it sweeps alternative encodings and plain host USM. The optional CUDA arm actually allocates a separate CUDA-pinned page and rings it by DMA; a 200-ms timeout rescues the list with a host store but still permits final exit zero.

[b70_sycl_zex_interop_probe.cpp](../../../experiments/b70_sycl_zex_interop_probe.cpp) appends zex waits/writes directly to a SYCL-owned immediate list around a kernel and D2H. A command-streamer marker kernel after the possibly copy-engine D2H is load-bearing: it joins the cross-engine dependency before host-visible completion. A native command-queue handle instead of an immediate-list handle is refused.

[b70_baked_chain_probe.cpp](../../../experiments/b70_baked_chain_probe.cpp) obtains native handles for synthetic gate/up and W2 kernels, binds pointer-only arguments, and records a fixed `WAIT -> reset signal -> kernels -> D2H -> completion` list. The list is replayed without pointer/scalar patching and checked bit-exactly against the SYCL reference once. [b70_u32_abi_probe.cpp](../../../experiments/b70_u32_abi_probe.cpp) isolates mixed pointer/u32 field indexing and compares 256 exact outputs; diagnostic native API errors only affect exit if output differs. [b70_baked_kernel_ab.cpp](../../../experiments/b70_baked_kernel_ab.cpp) validates shipped 11/12-argument handles against the split SYCL path, then optionally adds CUDA-pinned foreign-host copies. Missing CUDA runtime skips only that third stage.

[b70_cs_harness.py](../../../experiments/b70_cs_harness.py) drives the actual Phase-7 provider/poller ABI with 47 pinned lane dictionaries. Serial and burst modes ring CPU NumPy views of CUDA-pinned words. Wedge replay intentionally rings one layer, waits for provider recovery/error accounting, requires signal scrubbing, clears stale completions, then continues. Its 600-second diagnostic sleeps and cleanup-skipping error exits are reproducer behavior, not service policy.

## 7. Native provider, compute, and numerical probes

[b70_122b_pilot_smoke.py](../../../experiments/b70_122b_pilot_smoke.py) reads exact SBINT401 planes for two pilot layers, dequantizes the same bytes on CPU in float64, issues deterministic three-row/six-route provider calls, and requires worst peak-relative error below $10^{-3}$. Exceptional paths are not protected by `finally`. [test_b70_int4_path.py](../../../experiments/test_b70_int4_path.py) is the strongest focused CPU differential: it first requires exact converter layout for quantized values and FP16 scales, then compares an eight-route B70 result with a CPU oracle that models FP16 boundaries, cosine above 0.999, and mean absolute error below 0.01.

[b70_captured_replay.py](../../../experiments/b70_captured_replay.py) extracts one NVFP4 bank row into a temporary one-layer bank and compacts captured global routes into its resident interval. It records digests, routes, nonfinite counts and provider health. Passing means finite output and no native health error; there is deliberately no numerical oracle. [b70_xpu_register_smoke.py](../../../experiments/b70_xpu_register_smoke.py) compares inherited registered/unregistered staging arms using replay stability, poller counters and trace-ring service timing. FP32 atomic accumulation permits a scaled epsilon rather than bit identity.

[b70_dual_card_smoke.py](../../../experiments/b70_dual_card_smoke.py) owns two disjoint BDF-selected provider/poller lanes. CUDA graph capture issues both before waiting on either, copies both partials back to CUDA, and sums them. Same-bank CPU references, per-card fixtures, NaN-poisoned foreign-card isolation, alternating replay stress and native error counters distinguish numerical error, stale output and cross-card leakage.

[b70_dispatch_latency.cpp](../../../experiments/b70_dispatch_latency.cpp) is a multi-mode laboratory: launch/copy/dispatch, host-memory provenance, real NVFP4 kernels, INT4 graph capture, persistent polling and queue waves. Default `all` omits environment, real-kernel and INT4-graph modes. An unknown mode still prints `done`. The INT4 graph branch is materially stronger than the timing branches because it checks replay mutation and a host CPU reference.

The remaining synthetic probes are narrower. [b70_compute_sanity.py](../../../experiments/b70_compute_sanity.py) enumerates XPU and measures GEMMs; its claimed reference remains on XPU and no threshold controls `PASSED`. [b70_expert_sycl.cpp](../../../experiments/b70_expert_sycl.cpp) times synthetic oneMKL MoE work, but its CPU comparison reuses expert-0 weights for routes 0/1/2 and only prints `PASS`/`CHECK`. [b70_profile.cpp](../../../experiments/b70_profile.cpp) profiles synthetic kernels and serialized transfer dispatch without checking output; its fused expression uses the same value for both nominal gate and up operands. [b70_onednn_nvfp4_bench.sh](../../../experiments/b70_onednn_nvfp4_bench.sh) runs benchdnn correctness/performance modes, but the prose `ref` disqualifier is not enforced and lack of `set -e` lets an earlier failed pipeline be masked by the last one. [b70_expert_sycl.cpp](../../../experiments/b70_expert_sycl.cpp) and these programs are diagnostic baselines, not real-bank provider qualification.

[attention_decode_probe.py](../../../experiments/attention_decode_probe.py) compares vLLM FA2 and FlashInfer decode implementations against a float64 GQA reference on CUDA tensors before and after graph capture, requiring closeness and relative RMSE at most 0.005. It excludes projections, routing, transport, scheduling, HTTP and service latency.

## 8. Worker prototypes and reproductions

[b70_worker.py](../../../experiments/b70_worker.py) reads expert IDs/weights and activation from controller-owned shared memory, computes random-weight Torch XPU experts, writes FP32 output, then publishes the matching completion sequence. [b70_esimd_worker.py](../../../experiments/b70_esimd_worker.py) contradicts its name: it imports an ESIMD module but never calls it, ignores routed expert IDs, and performs positional random FP16 Torch matmuls. [dual_gpu_test.py](../../../experiments/dual_gpu_test.py) overlaps the worker with local CUDA computation and compares latency to a CPU cold-expert arm, but the CPU, CUDA and worker weights are unrelated and the B70 output is not combined or validated. These are control/latency prototypes only.

[bench_nvfp4_vs_mxfp4.py](../../../experiments/repro/bench_nvfp4_vs_mxfp4.py) drives one Xe2 grouped-GEMM entry point with NVFP4, MXFP4 and FP16 storage. Fresh unseeded operands and no output inspection make it a throughput comparison, and per-case exceptions are printed while `main` still returns zero. [build_device_usm_host_shadow.sh](../../../experiments/repro/build_device_usm_host_shadow.sh) is only the oneAPI build harness for [device_usm_host_shadow.cpp](../../../experiments/repro/device_usm_host_shadow.cpp). That reproducer retains and optionally touches device-USM chunks while comparing system `MemAvailable`/`MemFree` with process RSS; its final sample occurs after freeing allocations but before queue/context teardown, despite the comment. [l0v2_memset_crash.cpp](../../../experiments/repro/l0v2_memset_crash.cpp) isolates allocate-plus-memset under Level Zero adapter v1/v2 and in-order/out-of-order queues. A null allocation breaks the loop but still prints `PASS`, so completed allocations must be compared with requested allocations externally.

## 9. Named model scenarios and properties

* `mappedHostRoundTripTest` reaches a registered CUDA-mapped host-word round trip.
* `pinnedScratchDiagnosticTest` reaches completed pinned staging while preserving the absence of payload validation.
* `dmaBufPatternTest` requires fd export/import and a matching full-pattern observation.
* `nativeQueueOverlapTest` separates endpoint correctness from observed two-lane overlap.
* `commandOrderingFailureTest` shows that a published completion with broken payload order is failure.
* `providerCpuOracleTest` requires valid bank/geometry, issue/take, finite output and oracle agreement.
* `allocationEarlyNullDiagnosticTest` exposes a source-level false-pass risk where fewer allocations complete than were requested.
* `missingRuntimeTest` reaches explicit unavailability.
* `attentionReplayFailureTest` distinguishes initial/captured numerical failure.
* `workerCompletionIsNotParityTest` preserves the fact that matching a shared-memory sequence is not numerical parity.

The invariant bounds fixture counts, forbids more completions than submissions, requires export before import, and prevents `numericalOk` without both payload observation and oracle comparison. Released state has no modeled host/device/fd ownership.

## 10. External conditions, limits, and CPU comparisons

These programs depend on particular combinations of an RTX 5090/CUDA driver and runtime, Intel B70/oneAPI SYCL/Level Zero, experimental zex entry points, CUDA graph stream-memory operations, PyTorch CUDA/XPU, vLLM/FlashInfer, QuixiCore kernels, real expert banks, Linux memfd/eventfd/SCM_RIGHTS, `/proc`, dma-buf-capable kernel drivers, compatible IOMMU/PCIe topology, stable clocks, and idle cards. Several sources select a generic or first device while labelling it 5090/B70; labels do not qualify hardware identity.

The Reyu model does not claim driver cache coherence, latency, bandwidth, floating-point parity, or exhaustive resource cleanup. Submission, copying, arithmetic, and timing are abstract observations. Spins that are unbounded in source stay an external liveness condition rather than being silently turned into guaranteed progress.

Useful CPU-executable comparisons are limited and concrete:

1. decode identical INT4/NVFP4 bytes and evaluate the same routed SwiGLU fixture on CPU, as the pilot and focused INT4 probes already do;
2. checksum/full-compare the shared pinned mapping before and after provider copies and add same-buffer `memcpy` floors to localize corruption or page/cache overhead;
3. allocate/touch anonymous CPU memory in the same chunks as the host-shadow reproducer to separate ordinary Linux accounting from device-driver backing;
4. validate grouped format kernels at small shapes against decoded CPU GEMM using identical seeded operands;
5. inspect request versus completed allocation counts for Level Zero reproducers.

A CPU timing number cannot replace a PCIe, dma-buf, command-streamer, adapter-v1/v2, or GPU-concurrency observation. Likewise, a GPU-resident float64 reference is independent arithmetic but not a CPU implementation comparison.
