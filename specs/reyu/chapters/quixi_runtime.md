# Embedded QuixiCore-XPU runtime

[Executable model: `quixi_runtime.ryu`](../models/quixi_runtime.ryu)

## 1. Responsibility and whole-branch design

`src/QuixiCore-XPU` is a first-party, standalone Intel-XPU backend embedded in this checkout. It is not the Shooting Brake B70 provider, not the Phase-7 imported Xe2 grouped-kernel subtree, and not a vendor checkout. Its runtime branch has five layers:

1. **Portable identity layer.** A C++20 library publishes backend metadata and contract-family names without requiring oneAPI.
2. **SYCL runtime layer.** When enabled, it discovers Intel GPUs, creates one in-order queue, and owns current-queue-only command graphs.
3. **Native typed operation layer.** PyTorch and C++ callers pass a live `sycl::queue`, raw USM pointers, shapes, storage dtypes, variants, and blocking policy into native operations. The operation families and arithmetic are explained by the adjacent Quixi operations branch; this chapter retains only queue, lifetime, capture, and binding effects.
4. **Canonical generated adapter layer.** A generic `KernelCall` ABI and generated dispatch table exist, but every explicit generated XPU adapter in the assigned source is a stub returning `not_implemented`. The working typed native calls do not become wired merely because a similarly named canonical operation exists.
5. **Qualification/workflow layer.** CMake, presets, examples, smoke programs, the PyTorch parity program, and the performance orchestrator produce different kinds of evidence. A successful portable build, a hardware-skipped C++ smoke, PyTorch-XPU parity, and a timed SYCL benchmark are deliberately not treated as equivalent.

```text
portable metadata library (always)
              |
      SYCL capability gate
              |
  Intel device selection -> in-order queue ----------------------+
              |                                                   |
       native typed ops <---- current torch.xpu stream ---- PyTorch binding
              |                                                   |
      oneAPI command graph                              at::xpu::XPUGraph
              |
       device benchmark

canonical KernelCall -> generated OperationId switch -> generated failure stub
                         (no edge to native typed ops in assigned source)
```

The runnable model first exposes this branch shape, then deepens device selection, queue ownership, graph transitions, binding validation, canonical-stub results, and evidence workflows. Numerical kernels are abstracted as one enqueue so that runtime ordering is explicit without duplicating the operational-family model.

## 2. Actors, interfaces, and ownership

| Actor | Interface | Ownership and observable contract |
|---|---|---|
| Runtime selector | `gpu_devices`, `make_gpu_queue`, `describe_device` | Selects Intel GPUs from one SYCL platform; returned queue is by value, in order, optionally profiling-enabled. |
| Native graph | `CommandGraph` | Move-only owner of a queue, a modifiable graph, and an optional executable graph. Replay enqueues and returns an event; only synchronization/reset waits. |
| PyTorch graph | binding `XpuGraph` | Wraps `at::xpu::XPUGraph`, remembers the current XPU capture stream, validates an optional pool pair, and forgets the stream on reset. It is a different graph implementation from native `CommandGraph`. |
| PyTorch op wrappers | functions exported by `PYBIND11_MODULE` | Validate selected tensor properties, allocate outputs, obtain the current stream queue, and submit native typed operations with `blocking=false`. Input/output tensors and persistent captured scratch remain framework/caller-owned. |
| Canonical adapter | `KernelCall`, `contract_api::dispatch` | Non-owning pointer/count records and opaque context/stream/workspace. The assigned dispatcher executes no native kernel: explicit cases return `not_implemented`; its default returns `adapter_not_wired`. |
| Build system | CMake targets and presets | Always builds metadata; builds runtime/dispatch/kernels only behind SYCL compiler acceptance. oneDNN is optional. Installed export contains the base XPU target, not XPUOps. |
| Evidence programs | smoke/parity/benchmark programs | Exercise host metadata, host numerical oracles, PyTorch-XPU eager references, or timed device events. Skip behavior and missing metrics constrain what a successful exit means. |

The native C++ op boundary is defined in the Quixi operations branch. The seam is queue semantics: most binding calls use the tensor's current XPU stream queue, pass raw `data_ptr()` storage directly, select a native variant, and request nonblocking submission. Therefore tensor storage and captured addresses must remain valid until ordered work completes. The binding registers no automatic autograd formulas; `gelu_backward` is an explicit operation rather than an autograd integration claim ([README](../../../src/QuixiCore-XPU/README.md#L49-L53)).

## 3. Source-backed state and operation table

| Model action/state | Source operation | Guard and effect |
|---|---|---|
| `discoverDevices` | [`gpu_devices`](../../../src/QuixiCore-XPU/src/runtime/runtime.cpp#L34-L52) | Per platform, retain only Intel-vendor GPUs. Replace the current winner for a larger count, or for an equal-count Level Zero platform when the current winner is not Level Zero. |
| `createQueue` | [`make_gpu_queue`](../../../src/QuixiCore-XPU/src/runtime/runtime.cpp#L54-L70) | Reject no-device selection; clamp oversized index to the last selected device; always use `in_order`, optionally add profiling. |
| `createNativeGraph` | [`CommandGraph::CommandGraph`](../../../src/QuixiCore-XPU/src/runtime/graph.cpp#L14-L32) | Own queue and fresh modifiable graph; support is not rejected until capture begins. |
| `nativeCaptureBegin` | [`capture_begin`](../../../src/QuixiCore-XPU/src/runtime/graph.cpp#L34-L47) | Require graph aspect, inactive recording, and no existing executable; wait queue, then start current-queue recording. |
| `nativeCaptureEnd` | [`capture_end`](../../../src/QuixiCore-XPU/src/runtime/graph.cpp#L49-L56) | Require recording; end and finalize into an executable graph. |
| `replayNative` | [`replay`](../../../src/QuixiCore-XPU/src/runtime/graph.cpp#L58-L63) | Require executable; enqueue graph and return an event without waiting. |
| `synchronize` / `resetNative` | [`synchronize`, `reset`](../../../src/QuixiCore-XPU/src/runtime/graph.cpp#L65-L76) | Synchronize waits. Reset rejects active capture, waits, drops executable, and creates a fresh modifiable graph on the same queue. |
| `destroyGraph` | defaulted destructor | RAII releases owned graph/PImpl state; source performs no explicit wait, end-recording, or reset. The model therefore does not manufacture implicit completion. |
| `torchCaptureBegin` | binding `XpuGraph::capture_begin` | Obtain current XPU stream, require a two-element nonnegative pool when supplied, begin PyTorch capture, retain the stream. |
| `resetTorch` | binding `XpuGraph::reset` | Reset PyTorch graph and clear retained stream. Subsequent binding synchronize rejects “no capture stream.” |
| `bindingInvoke` | `check`, `check_same_device`, wrappers | XPU and contiguous are universal binding checks; selected multi-tensor wrappers also check same device, dtype, dimensions, and shapes before current-stream enqueue. |
| `captureSplitMoe` | binding `nvfp4_moe` split branch | During capture, caller-owned FP32 `[M*top_k,2I]` scratch is mandatory; eager execution may allocate it. |
| `dispatchCanonical` | `contract_api::dispatch` and generated stubs | Explicit generated case ignores `KernelCall` and returns `not_implemented`; default reports `adapter_not_wired`. |
| `configureBuild` / `buildLibraries` | CMake capability branches | Base target is portable. Enabling SYCL with a compiler that rejects `-fsycl` is fatal. oneDNN absence is nonfatal. |
| `buildExtension` | `build.sh` and `setup.py` | Requires shared `icpx -fsycl` XPUOps, `.sycl` extension compilation/device link, PyTorch SyclExtension, and Python environment. |
| `recordCppChecks` | native smoke executables | On-device checks use host formulas where applicable, but several return success without an eligible GPU; “checked” need not mean hardware exercised. |
| `recordTorchParity` | `test_parity.py` | Requires `torch.xpu`; compares native extension outputs to eager PyTorch-XPU references and exercises binding graph/state edges. |
| `runBenchmark` | `bench_kernels.py` | Timed kernel rows only run for `sycl`; dev kernels-only can produce zero rows and success. SYCL missing binary yields a failing `missing` row. |

## 4. Runtime selection and queue mechanics

### 4.1 One-platform selection

[`gpu_devices`](../../../src/QuixiCore-XPU/src/runtime/runtime.cpp#L34-L52) does not merge devices across platforms. It filters every platform to devices satisfying both `is_gpu()` and vendor ID `0x8086`, then retains one platform vector. A platform with more qualifying devices beats a Level Zero platform with fewer devices. Level Zero is only the equal-count tie breaker used to avoid selecting duplicate OpenCL aliases. Equal alternatives that do not improve count or backend preference preserve the first winner. The model's two bounded platform arguments expose the count and tie behavior; they are an exploration bound, not a platform-count claim.

[`make_gpu_queue`](../../../src/QuixiCore-XPU/src/runtime/runtime.cpp#L54-L70) re-enumerates. No selected device throws a `runtime_error`. An oversized unsigned index is clamped, not rejected. Both queue forms are in order; profiling adds `enable_profiling`. The layer installs no explicit async handler or context. [`describe_device`](../../../src/QuixiCore-XPU/src/runtime/runtime.cpp#L72-L86) emits name, vendor, driver, compute units, work-group limit, and subgroup sizes. Contrary to the header comment's “name + backend,” it does not include a SYCL backend identifier.

### 4.2 Native graph lifecycle and errors

The native graph's state machine is:

```text
construct -> modifiable --capture_begin--> recording --capture_end--> finalized
                ^                                                |       |
                +---------------------- reset(wait) --------------+       +-- replay (enqueue, remains finalized)

begin unsupported       -> runtime_error, state unchanged
begin while recording   -> logic_error, state unchanged
begin while finalized   -> logic_error requiring reset
end while not recording -> logic_error
replay before finalized -> logic_error
reset while recording   -> logic_error
scope destruction       -> RAII release; no explicit wait/reset in source
```

Capture begin waits the owned queue before recording, while replay is asynchronous. Reset waits before discarding executable and rebuilding modifiable state. `queue()` returns a mutable reference to the owned queue. Copy is deleted and move is defaulted; methods have no moved-from guard before dereferencing the PImpl. Graph support is precisely the device's `ext_oneapi_graph` aspect.

The [`xpu_graph_smoke`](../../../src/QuixiCore-XPU/tests/xpu_graph_smoke.cpp) fixture captures a native SiLU call, replays against one input, changes the same stable allocation, replays again, and resets before freeing USM. It returns success when no selected GPU or graph aspect exists, so a zero process status alone does not prove replay happened.

### 4.3 PyTorch graph is a separate lifecycle

The binding's [`XpuGraph`](../../../src/QuixiCore-XPU/bindings/pytorch/tk_xpu_ext.sycl#L61-L108) delegates capture/replay to `at::xpu::XPUGraph`, not native `CommandGraph`. Capture reads `getCurrentXPUStream`, validates an optional two-element nonnegative pool, begins capture, then retains that stream. `synchronize` waits only that retained capture stream and rejects after reset because reset clears it. `instantiate`, pool query, debug mode, and dump are delegated. The model preserves the observed pool/reset/current-stream edges without equating PyTorch's internal graph states with the oneAPI graph implementation.

## 5. Binding contracts and lifetime limits

[`dtype_of`](../../../src/QuixiCore-XPU/bindings/pytorch/tk_xpu_ext.sycl#L26-L34) accepts Float32, Float16, and BFloat16 only. [`check`](../../../src/QuixiCore-XPU/bindings/pytorch/tk_xpu_ext.sycl#L40-L43) requires XPU-resident contiguous tensors. Multi-input wrappers selectively add same-device, dtype, rank, shape, and quant-layout checks. They allocate outputs through PyTorch and submit native operations to [`queue_of`](../../../src/QuixiCore-XPU/bindings/pytorch/tk_xpu_ext.sycl#L36-L38), the tensor device's current XPU stream queue. There are no data copies at the binding/native seam.

The exported surface includes activation/norm/matmul/attention/argmax, selected NVFP4/FP8/MoE/GDN operations, explicit stream synchronization, `XPUGraph`, and device count ([module registration](../../../src/QuixiCore-XPU/bindings/pytorch/tk_xpu_ext.sycl#L504-L550)). This is a deliberately smaller, hand-wired surface than the canonical operation registry. The wrappers call `qx::ops` directly; they do not call `contract_api::dispatch`.

A capture-specific lifetime rule appears in split NVFP4 MoE: without supplied scratch, eager execution allocates a temporary tensor, but active capture rejects that path. Captured split MoE requires caller-owned FP32 scratch with exact `[M*top_k,2I]` shape that remains alive and at a stable address for graph lifetime ([binding](../../../src/QuixiCore-XPU/bindings/pytorch/tk_xpu_ext.sycl#L370-L401)). This chapter models the lifetime obligation, not MoE arithmetic.

The build boundary is also operational. [`build.sh`](../../../src/QuixiCore-XPU/bindings/pytorch/build.sh) records that a static archive did not register SYCL device images and led to submission failure. It therefore creates `build-sycl-shared` with `icpx`, `-fsycl`, and `BUILD_SHARED_LIBS=ON`, then builds the `.sycl` extension. [`setup.py`](../../../src/QuixiCore-XPU/bindings/pytorch/setup.py) links `quixicore_xpu_ops`, `quixicore_xpu`, and oneDNN and embeds library rpaths. This is a binding build contract, not proof any runtime path was executed.

## 6. Canonical ABI: contract shape versus wired behavior

[`kernel_abi.hpp`](../../../src/QuixiCore-XPU/include/quixicore/contract/kernel_abi.hpp) defines a C++ adapter vocabulary: status code and borrowed operation/detail strings; dtype and memory-space tags; non-owning `TensorView`; typed attributes; and `KernelCall` arrays plus opaque backend context, stream, and workspace. It is not demonstrated C linkage: there is no `extern "C"`, exported C symbol, explicit enum numbering, ABI version field, or pointer lifetime/alignment convention in this header.

[`contract_stubs.hpp`](../../../src/QuixiCore-XPU/include/quixicore/xpu/contract_stubs.hpp) explicitly calls itself generated scaffolding. Its 273 inline functions ignore `KernelCall` and return `StatusCode::not_implemented` with a generated reason. The descriptor array makes those failures discoverable. [`contract.hpp`](../../../src/QuixiCore-XPU/include/quixicore/xpu/contract.hpp) routes every explicit switch arm to one of those functions. The default returns `adapter_not_wired`, using `unknown_operation` if no operation name is available. Thus:

- generated canonical name present does **not** mean native implementation;
- a native typed operation present does **not** mean canonical adapter wired;
- stub reason categories are metadata, but their runtime result is the same `not_implemented` code;
- the binding's successful native path bypasses the generated adapter entirely.

[`kernel-stubs.yaml`](../../../src/QuixiCore-XPU/.quixicore/kernel-stubs.yaml) reports 303 canonical operations and 273 generated stubs at its recorded revision/date. That negative inventory is useful traceability, not executable coverage. The operations/families themselves are detailed by the adjacent Quixi operations branch.

## 7. Build, capability, test, and performance workflow

### 7.1 Target graph and package limitation

[`CMakeLists.txt`](../../../src/QuixiCore-XPU/CMakeLists.txt) always creates `quixicore_xpu` from metadata source with C++20/PIC. `QUIXICORE_XPU_ENABLE_SYCL` defaults off. When enabled, the compiler must accept `-fsycl`; runtime, dispatch, `*.sycl.cpp`, and optionally `*.onednn.cpp` form `quixicore_xpu_ops`. oneDNN absence is nonfatal. AoT targets are an optional startup/performance lever and empty/JIT by default.

The SYCL branch also creates a device probe, broad ops smoke, int4 MoE smoke, graph smoke, and non-CTest benchmark executable. Installation exports only the base `quixicore_xpu` target and headers. Although XPUOps has an export name, it is omitted from `install(TARGETS ...)`, so the installed package does not deliver `QuixiCore::XPUOps`. Binding construction instead consumes a build-tree shared library.

[`CMakePresets.json`](../../../src/QuixiCore-XPU/CMakePresets.json) defines portable `dev` and inherited `sycl` presets. The shell wrappers configure/build/test with explicit first argument, then `QUIXICORE_PRESET`, then `dev`; bench uses the environment/default and asks the Python orchestrator for `--phase all`; clean removes selected build/cache paths. `coverage-report` prints manifest sections—it is not code-coverage measurement.

### 7.2 What the checks compare

- [`backend_smoke.cpp`](../../../src/QuixiCore-XPU/tests/backend_smoke.cpp) checks static identity/family lookup and intentionally freezes the stale `planned` value.
- [`xpu_ops_smoke.cpp`](../../../src/QuixiCore-XPU/tests/xpu_ops_smoke.cpp) contains broad device checks against host-side formulas and structural oracles. It returns success without a GPU; selected unsupported FP8 exceptions are also reported as unsupported while preserving success.
- [`int4_moe_smoke.cpp`](../../../src/QuixiCore-XPU/tests/int4_moe_smoke.cpp) compares device int4 routed-MoE results to independent host dequantization/FFN accumulation for f16 and bf16 fixtures, including invalid routes and an empty expert. It also returns success without a GPU.
- [`xpu_graph_smoke.cpp`](../../../src/QuixiCore-XPU/tests/xpu_graph_smoke.cpp) uses a scalar host SiLU oracle but hardware-skip success semantics.
- [`test_parity.py`](../../../src/QuixiCore-XPU/bindings/pytorch/test_parity.py) requires XPU availability and compares against PyTorch XPU eager operations. It is not a CPU comparator. It also checks split/fused MoE agreement, captured scratch lifetime, replay over changed input, reset/pool errors, and GDN state-index edges.

These sources define potential implementation comparisons that Main can choose to execute. No such command was run while authoring this chapter.

### 7.3 Performance evidence and CPU comparison seam

[`bench_kernels.py`](../../../src/QuixiCore-XPU/perf/bench_kernels.py) creates an exclusive dated run directory, records environment/compiler/git metadata, runs ordered build-health phases with fail-fast behavior, and then runs the device benchmark matrix only for the `sycl` preset. YAML configurations replace default rows for covered kernel names; unreadable YAML and missing PyYAML silently retain defaults. A missing benchmark executable creates a failing `missing` row. Metrics are accepted from JSON-looking stdout; malformed or absent metrics do not independently fail a zero-return process. A dev kernels-only request produces no rows and succeeds because no SYCL kernel phase is entered.

The harness has no timed CPU runner. `variant=vendor` is still a device implementation selection, not a CPU baseline. The credible CPU comparisons currently available are **correctness** comparisons inside C++ smoke programs. A future performance comparison could feed the same kernel/dtype/shape records to a CPU runner, but CPU wall time and SYCL event time would remain distinct measurement domains. No CPU/XPU speed claim follows from the checked-in workflow.

## 8. Metadata contradictions and source priority

Three status surfaces disagree:

- [`.quixicore/backend.yaml`](../../../src/QuixiCore-XPU/.quixicore/backend.yaml) says the backend is `active` and defers operation maturity to the kernel manifest.
- [`README.md`](../../../src/QuixiCore-XPU/README.md) describes an active native backend with implemented and experimental operations.
- [`backend.hpp`](../../../src/QuixiCore-XPU/include/quixicore/xpu/backend.hpp) hard-codes `status="planned"`, and [`status_for_kernel_family`](../../../src/QuixiCore-XPU/src/backend.cpp#L18-L23) reports every known family as `planned`; its smoke test requires those values.

The executable metadata API is therefore stale relative to implemented native sources. This specification preserves the contradiction rather than normalizing it. For operational availability, native sources and the operation manifest take precedence; for what `runtime_backend_info` actually returns, `planned` remains exact.

## 9. Executable scenarios and properties

The model provides reachable witnesses and named runs for:

- Level Zero equal-count tie breaking and oversized queue-index clamping;
- a larger non-Level-Zero platform beating a smaller Level Zero platform;
- no-device queue failure;
- successful native capture, two replays, explicit completion, and reset;
- unsupported capture, replay-before-finalize, reset-during-capture, and recapture-before-reset errors;
- destruction with queued work still abstractly incomplete, reflecting absence of an explicit destructor wait;
- PyTorch pool validation and post-reset missing-stream synchronization error;
- captured split MoE rejection without persistent scratch and enqueue with scratch;
- direct native binding enqueue versus generated `not_implemented`/`adapter_not_wired` results;
- tensor validation rejection without enqueue;
- portable C++ smoke success without hardware evidence;
- SYCL compiler refusal, shared extension build, and PyTorch parity evidence;
- empty successful dev kernel phase, successful device rows, and missing benchmark failure.

`inv` checks selected-index bounds, queue/graph ownership, retained-stream ownership, enqueue/completion monotonicity, build dependency ordering, and the requirement that benchmark evidence has positive device rows. These are properties of the abstraction, not a proof about oneAPI, PyTorch, or hardware.

## 10. Exact assigned-file inventory

| Assigned source | Role in this branch |
|---|---|
| [`.clang-format`](../../../src/QuixiCore-XPU/.clang-format) | Mechanical C++ formatting policy; no capability evidence. |
| [`.editorconfig`](../../../src/QuixiCore-XPU/.editorconfig) | Editor newline/indent/whitespace policy; no runtime behavior. |
| [`.gitignore`](../../../src/QuixiCore-XPU/.gitignore) | Excludes build, binding, cache, local mirror, profiler, and raw-result artifacts. |
| [`.quixicore/backend.yaml`](../../../src/QuixiCore-XPU/.quixicore/backend.yaml) | Declarative active-backend identity; contradicts executable `planned` metadata. |
| [`.quixicore/kernel-stubs.yaml`](../../../src/QuixiCore-XPU/.quixicore/kernel-stubs.yaml) | Generated negative inventory/provenance for canonical stubs. |
| [`AGENTS.md`](../../../src/QuixiCore-XPU/AGENTS.md) | Correctness-first, measurement and provenance policy; forbids unsupported performance claims. |
| [`CHANGELOG.md`](../../../src/QuixiCore-XPU/CHANGELOG.md) | Unreleased scaffolding/history record, not runtime evidence. |
| [`CLAUDE.md`](../../../src/QuixiCore-XPU/CLAUDE.md) | Tool-specific measurement doctrine and historical assumptions, not capability proof. |
| [`CMakeLists.txt`](../../../src/QuixiCore-XPU/CMakeLists.txt) | Authoritative target/capability/test/install graph. |
| [`CMakePresets.json`](../../../src/QuixiCore-XPU/CMakePresets.json) | Portable dev and SYCL preset definitions. |
| [`CONTRIBUTING.md`](../../../src/QuixiCore-XPU/CONTRIBUTING.md) | Ownership, test/benchmark/manifest, and PR evidence policy. |
| [`LICENSE`](../../../src/QuixiCore-XPU/LICENSE) | MIT license and warranty/liability disclaimer. |
| [`README.md`](../../../src/QuixiCore-XPU/README.md) | User-facing identity, active/experimental status, scope, and workflow claims. |
| [`SECURITY.md`](../../../src/QuixiCore-XPU/SECURITY.md) | Disclosure and shared-contract versus backend issue-routing policy. |
| [`bindings/pytorch/setup.py`](../../../src/QuixiCore-XPU/bindings/pytorch/setup.py) | SyclExtension source/link/rpath declaration. |
| [`bindings/pytorch/test_parity.py`](../../../src/QuixiCore-XPU/bindings/pytorch/test_parity.py) | PyTorch-XPU oracle and graph/state harness. |
| [`bindings/pytorch/build.sh`](../../../src/QuixiCore-XPU/bindings/pytorch/build.sh) | Shared ops and `.sycl` device-link build workflow. |
| [`bindings/pytorch/tk_xpu_ext.sycl`](../../../src/QuixiCore-XPU/bindings/pytorch/tk_xpu_ext.sycl) | Actual zero-copy current-stream native binding and PyTorch graph wrapper. |
| [`cmake/QuixiCoreXPUConfig.cmake.in`](../../../src/QuixiCore-XPU/cmake/QuixiCoreXPUConfig.cmake.in) | Installed base-target package loader. |
| [`examples/backend_info.cpp`](../../../src/QuixiCore-XPU/examples/backend_info.cpp) | Prints static metadata; no dynamic capability probe. |
| [`examples/sycl_device_probe.cpp`](../../../src/QuixiCore-XPU/examples/sycl_device_probe.cpp) | Enumerates all SYCL platforms/devices; broader than runtime Intel-GPU selection and success on none. |
| [`include/quixicore/contract/kernel_abi.hpp`](../../../src/QuixiCore-XPU/include/quixicore/contract/kernel_abi.hpp) | Generic C++ adapter records/statuses and failure constructors. |
| [`include/quixicore/xpu/backend.hpp`](../../../src/QuixiCore-XPU/include/quixicore/xpu/backend.hpp) | Static, stale backend/family metadata declaration. |
| [`include/quixicore/xpu/contract.hpp`](../../../src/QuixiCore-XPU/include/quixicore/xpu/contract.hpp) | Generated operation switch to stubs/default failure. |
| [`include/quixicore/xpu/contract_stubs.hpp`](../../../src/QuixiCore-XPU/include/quixicore/xpu/contract_stubs.hpp) | Generated `not_implemented` functions and descriptors. |
| [`include/quixicore/xpu/graph.hpp`](../../../src/QuixiCore-XPU/include/quixicore/xpu/graph.hpp) | Move-only current-queue native graph API. |
| [`include/quixicore/xpu/runtime.hpp`](../../../src/QuixiCore-XPU/include/quixicore/xpu/runtime.hpp) | Storage dtype, Intel device, queue, and description API. |
| [`perf/bench_kernels.py`](../../../src/QuixiCore-XPU/perf/bench_kernels.py) | Build-health and device-only benchmark orchestrator. |
| [`scripts/bench`](../../../src/QuixiCore-XPU/scripts/bench) | Strict wrapper selecting environment/default preset for all-phase harness. |
| [`scripts/build`](../../../src/QuixiCore-XPU/scripts/build) | Strict preset build wrapper. |
| [`scripts/clean`](../../../src/QuixiCore-XPU/scripts/clean) | Removes selected standard build/cache paths, not every ignored variant. |
| [`scripts/configure`](../../../src/QuixiCore-XPU/scripts/configure) | Strict preset configure wrapper. |
| [`scripts/coverage-report`](../../../src/QuixiCore-XPU/scripts/coverage-report) | Manifest display utility, not execution/code coverage. |
| [`scripts/test`](../../../src/QuixiCore-XPU/scripts/test) | Strict preset CTest wrapper. |
| [`src/backend.cpp`](../../../src/QuixiCore-XPU/src/backend.cpp) | Exact metadata lookup and all-known-families-`planned` behavior. |
| [`src/runtime/graph.cpp`](../../../src/QuixiCore-XPU/src/runtime/graph.cpp) | Native oneAPI command-graph implementation and errors. |
| [`src/runtime/runtime.cpp`](../../../src/QuixiCore-XPU/src/runtime/runtime.cpp) | Device selection, queue construction, dtype helpers, and description implementation. |
| [`tests/backend_smoke.cpp`](../../../src/QuixiCore-XPU/tests/backend_smoke.cpp) | Portable static metadata assertion harness. |
| [`tests/int4_moe_smoke.cpp`](../../../src/QuixiCore-XPU/tests/int4_moe_smoke.cpp) | Host-oracle routed-int4 device harness with skip-on-no-GPU behavior. |
| [`tests/xpu_graph_smoke.cpp`](../../../src/QuixiCore-XPU/tests/xpu_graph_smoke.cpp) | Native graph replay/changed-input/USM-lifetime harness with capability skips. |
| [`tests/xpu_ops_smoke.cpp`](../../../src/QuixiCore-XPU/tests/xpu_ops_smoke.cpp) | Broad native-op host-oracle/structural harness with selective capability skips. |

## 11. Abstraction limits and external dependencies

The model does not simulate SYCL scheduling, oneAPI graph internals, USM coherence, PyTorch allocator pools, tensor arithmetic, oneDNN, Level Zero, compiler/device-image registration, or wall-clock performance. It records their observable ownership, capability, ordering, and error boundaries. Queue work is an integer count; `synchronize`/reset makes all abstract enqueues complete. Graph destruction deliberately does not invent synchronization absent from source.

External dependencies are oneAPI DPC++/SYCL, optional oneDNN, Level Zero, PyTorch XPU and its `XPUGraph`, CMake/CTest, and the host operating/toolchain environment. Native operation family dispatch and numeric behavior belong to the Quixi operations model. Host formula checks and PyTorch eager parity are potential implementation comparisons; benchmark rows are device evidence only. No test, build, lint, formatter, benchmark, or hardware workload was run for this authored branch.
