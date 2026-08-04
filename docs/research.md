# Shooting Brake Prior-Art and Provenance Ledger

## Purpose and status

This document records how existing repositories, articles, and papers inform Shooting Brake. It elaborates the repository roles and ownership boundaries in [architecture.md](architecture.md); it is not an implementation report, benchmark report, or assertion that any proposed integration works on the target machine.

The source material for this ledger is the repository role matrix and repository details in `architecture.md`, plus the prior-art and revised repository-inspection notes in `readme.md`. The later revised conclusions in `readme.md` take precedence over earlier proposals. No new web research, build, test, model run, or benchmark was performed for this ledger.

## Evidence classes and normative rules

Every statement below falls into one of four classes:

- **Recorded source observation** — a property that the existing documents say was found by inspecting a repository, interface, patch, or checkout. It is not necessarily revalidated against the current working tree.
- **Upstream claim** — a result reported by an article or paper. It is not a Shooting Brake result and must never be presented as local evidence.
- **Design decision or target** — intended Shooting Brake behavior or an approved build-versus-borrow boundary. It remains unimplemented unless another document explicitly establishes implementation.
- **Unverified assumption** — a proposition that requires local measurement or correctness evidence before it can be relied upon.

The following rules are normative:

1. A cited paper or article result **MUST** be labeled as an upstream claim and **MUST NOT** be copied into [benchmarking.md](benchmarking.md) as a local result.
2. A repository revision records what was inspected; it **MUST NOT** be read as a claim that the current checkout is still at that revision.
3. Source compatibility **MUST** be established with controlled tests. Similar tensor dimensions, supported dtypes, or matching device names are not compatibility evidence.
4. Colibri remains the model owner and correctness oracle. Borrowed kernels, schedulers, profilers, or generated code do not replace that authority.
5. Shooting Brake **MUST** borrow the narrowest useful component. It must not embed an entire serving framework or runtime merely to obtain a kernel, lifecycle pattern, or scheduling idea.
6. Prediction and prefetch **MUST NOT** be required for correctness. A failed, late, or canceled prediction must preserve exact execution through the authoritative owner or fallback.
7. Foreground decode **MUST NOT** depend on ordinary flash/NVMe expert reads. Flash-backed work informs storage and recovery design only until local evidence establishes a safe, explicitly gated role.
8. Generated or autotuned code is a candidate, never ground truth. It requires deterministic oracle comparison, representative shapes, sequential mixed-shape coverage, build/hardware provenance, and tail-regression checks.

## Build-versus-borrow summary

| Source | Approved boundary |
|---|---|
| Colibri | Reuse as the core runtime and oracle; generalize only heterogeneous-worker boundaries. |
| Luce Spark / Lucebox | Borrow placement algorithms and invariants; build the cross-vendor execution fabric. |
| vllm-xpu-kernels | Borrow the narrow grouped-MoE compute path behind a stable worker ABI; do not fork all of vLLM. |
| llm-scaler | Preserve as a pinned, patched comparison oracle until safety and API differences are reconciled. |
| Intel XPU Triton | Evaluate later as an explicitly converted MXFP4 provider; never treat it as integer-W4-compatible by inspection. |
| Xe-Fuse | Use later as a BF16 fused-compute reference and possible backend; do not use as the initial W4 runtime. |
| Xe-Forge | Use only after correctness as a candidate generator/autotuner; Shooting Brake owns verification and acceptance. |
| vllm-xpu-breakdown | Reuse headless profiling, replay, and reporting; build Shooting Brake semantic trace spans. |
| LLM.xpu | Borrow lifecycle and shared-host-buffer invariants; do not adopt it as the runtime. |
| ATSInfer | Borrow measured-cost placement principles; keep remote experts whole. |
| ProMoE, Fate, and Pre-gated MoE | Consider predictive hints later; build exact fallback and admission safeguards locally. |
| PowerInfer and DejaVu | Borrow locality and heterogeneous sparse-execution concepts only. |
| Flash-storage inference | Treat as storage prior art, not permission to put NVMe on the normal token path. |
| SGLang | Optionally integrate at the request boundary later; do not make it the first heterogeneous runtime. |
| KTransformers | Borrow CPU batching/fallback concepts where the host supports them; retain Colibri as the present CPU baseline. |
| Mooncake | Defer distributed transfer/storage abstractions until the design crosses machines; build local PCIe rings first. |

## Repository and runtime ledger

### Luce Spark / Lucebox

- **Provenance:** `lucebox/`; origin recorded as `Luce-Org/lucebox-hub`. No inspected revision is recorded in the scoped documents.
- **Recorded source observation:** The documents attribute traffic-derived per-layer expert frequency, persistent placement profiles, bounded cache capacity, hot/cold separation, asynchronous promotion, self-tuning serving, routing statistics, and learned residency decisions to Luce Spark behavior in Lucebox.
- **Shooting Brake may reuse:** The policy concepts and invariants: measured hotness, bounded capacity, durable profiles, asynchronous promotion, and traffic-calibrated placement epochs. These feed the placement work described in [placement.md](placement.md).
- **Shooting Brake must build:** An $N$-device, cross-vendor execution fabric whose secondary devices compute with resident weights and whose placement objective includes distributed critical-path cost, transfer, queues, and fan-out—not only cache miss rate.
- **Does not prove:** That a CUDA/CPU cache architecture generalizes to NVIDIA-to-Intel execution; that learned placement improves this topology; that promotion is cheap enough; or that any latency/throughput target is achieved locally.

### Colibri

- **Provenance:** `colibri-variants/`; origin recorded as `JustVugg/colibri`. The source list in `README.md` names <https://github.com/JustVugg/colibri>. No inspected revision is recorded in the scoped documents.
- **Recorded source observation:** The existing documents identify Colibri as providing GLM-5.2 loading and execution, packed quantized formats, quantized CPU fallback, CUDA state-owner execution, routing and route telemetry, tier management, persistent KV/context, resource planning, OpenAI-compatible serving, backend/tier correctness tests, and CUDA/Vulkan multi-device precedents.
- **Shooting Brake may reuse:** Colibri directly as model owner, request-serving substrate, canonical router, state-owner path, fallback path, route-trace source, and correctness oracle.
- **Shooting Brake must build or generalize:** Fixed CUDA/Vulkan assumptions, at-most-one-secondary-device logic, synchronous secondary submission, frequency-only placement, and CPU-side per-expert accumulation where one device partial suffices. The worker boundary must leave attention, KV state, router logits, global top-k, shared experts, dense layers, sampling, and request state under the Colibri-based owner.
- **Does not prove:** A native Intel XPU expert-worker path, CUDA-to-Level-Zero transport behavior, B70 numerical equivalence, multi-B70 stability, or Shooting Brake performance.

### vllm-xpu-kernels

- **Provenance:** `intel-xpu/vllm-xpu/vllm-xpu-kernels/`; origin `vllm-project/vllm-xpu-kernels`; inspected revision `dd3bc2127cff`. `README.md` also names <https://github.com/vllm-project/vllm-xpu-kernels> and an Intel Arc Pro B-series vLLM optimization article at <https://vllm.ai/blog/2025-11-11-intel-arc-pro-b>.
- **Recorded source observation:** The inspected interfaces expose Xe2/Xe3 grouped GEMM and a fused-MoE sequence covering row remap, gate/up grouped GEMM, activation, down grouped GEMM, and weighted inverse gather. Recorded formats include BF16/FP16, FP8, INT4, and MXFP4 with group sizes 32, 64, 128, and 256, including zero-row experts and externally prepared contiguous expert rows.
- **Shooting Brake may reuse:** The narrow remap/grouped-GEMM/activation/gather operations as the primary B70 W4/INT4 compute candidate. Adapt them to accept only routes owned by one B70, translate global to local expert IDs, and return one weighted partial per original token.
- **Shooting Brake must build:** A stable provider-neutral B70 worker ABI, resident expert service, CUDA-owner transport, route-subset packaging, failure handling, and local correctness/measurement harnesses. The runtime must not embed or fork all of vLLM.
- **Does not prove:** Colibri W4 compatibility. Matching dimensions and group size do not establish nibble convention, zero point, gate/up ordering, packing, scale representation, padding, alignment, or Colibri-specific correction behavior. It also does not prove cross-vendor transport, residency service behavior, or end-to-end performance.
- **Required gate:** Convert exactly one expert first and compare numerically with Colibri CPU and state-owner-GPU references before bulk conversion; also reconcile the `llm-scaler` safety patch and API differences.

### llm-scaler

- **Provenance:** `intel-xpu/llm-scaler/`; described as an Intel llm-scaler checkout. No llm-scaler repository revision is recorded. Its Docker configuration records a vllm-xpu-kernels pin at `3cab97a` and applies `intel-xpu/llm-scaler/vllm/patches/vllm_xpu_kernels.patch`.
- **Recorded source observation:** The documents retain this checkout as a known-working patched B70 INT4 baseline. Its patch guards Xe2 block-2D scale prefetch when a scale surface is too small or insufficiently aligned; `K=1408`, `group_num=11` is the recorded failure-class example. The patch also differs from current upstream in grouped-GEMM arguments and `gelu_tanh` behavior.
- **Shooting Brake may reuse:** The pin, patch, inputs, and behavior as an identical-input A/B comparison oracle and a source of the required prefetch safety condition.
- **Shooting Brake must build:** A reconciled current provider that preserves or proves unnecessary the safety guard for every supported shape and explicitly resolves the old INT4/MXFP4 format flags, current simplified dispatch, activation behavior, mixed-shape execution, numerical behavior, and tails.
- **Does not prove:** That the older API is the long-term source of truth, that current upstream is safe merely because it is newer, or that the patched baseline integrates with Colibri or the proposed worker fabric.

### Intel XPU Triton backend

- **Provenance:** `intel-xpu/intel-xpu-backend-for-triton/`; source named as <https://github.com/intel/intel-xpu-backend-for-triton>. No inspected revision is recorded.
- **Recorded source observation:** The existing documents identify an alternative ragged-expert implementation using MXFP4 E2M1 with block size 32. They also record that the referenced benchmark path is CUDA-centric.
- **Shooting Brake may reuse:** Evaluate it later as an alternative MXFP4 kernel provider after explicit conversion and oracle validation.
- **Shooting Brake must build:** Format conversion, a provider adapter, B70 resident-worker integration, and local correctness and performance evidence.
- **Does not prove:** Compatibility with Colibri integer W4, a complete B70 worker runtime, or performance on the Shooting Brake path. It must never be treated as a format-compatible drop-in.

### Xe-Fuse

- **Provenance:** `intel-xpu/vllm-xpu/Xe-Fuse/`; origin `IntelLabs/Xe-Fuse`; inspected revision `3e6f0425ecb8`.
- **Recorded source observation:** The inspected BMG-G31/Xe20 SYCL/CUTLASS paths provide GEMM epilogue fusion, fused SiLU/SwiGLU/GeGLU, RMSNorm/residual operations, batched expert slices, compile-time tile selection, BF16 execution, and W8A8 INT8 dequantization.
- **Shooting Brake may reuse:** A later native B70 fused-FFN benchmark, a BF16 compute upper-bound reference, epilogue/activation-fusion techniques, and possibly a worker backend after compatible quantized support exists.
- **Shooting Brake must build:** W4 support or conversion, ragged router/remap/gather, residency service, CUDA-to-XPU transport, and verified end-to-end MoE tests.
- **Does not prove:** Immediate W4 viability, real GLM route handling, resident-worker behavior, cross-vendor integration, or any local performance upper bound until run under controlled local conditions.

### Xe-Forge

- **Provenance:** `intel-xpu/vllm-xpu/Xe-Forge/`; origin `IntelLabs/Xe-Forge`; inspected revision `ea0d20ab7fed`.
- **Recorded source observation:** The repository was recorded as proposing tiles, validating DPAS constraints, generating SYCL/CUTLASS source, compiling, benchmarking, and feeding measurements into subsequent search. Its current MoE template uses synthetic expert counts and random tensors, BF16/F16 rather than real INT4 GLM format, no router/top-k, and `verify=0` by default; the template can report success without output comparison.
- **Shooting Brake may reuse:** Tile search and generated-kernel optimization only after the baseline W4 worker is correct and measured.
- **Shooting Brake must build:** The acceptance harness: deterministic inputs, canonical Colibri output, representative route/row distributions, sequential mixed-shape runs, exact build/hardware provenance, enabled comparison, and p95/p99 regression rejection.
- **Does not prove:** Numerical correctness, real-route coverage, INT4 compatibility, production stability, or performance of accepted Shooting Brake kernels. Generated code is never a correctness authority.

### vllm-xpu-breakdown

- **Provenance:** `intel-xpu/vllm-xpu-breakdown/`; origin `yangulei/vllm-xpu-breakdown`; inspected revision `dc477078bc0d`.
- **Recorded source observation:** Its headless components record dispatched XPU operations, shapes/dtypes, CPU and device time, kernel/memcpy/memset events, correlation-derived module hierarchy, prefill/decode views, trace/report formats, replay fidelity, history, and roofline-guided rankings.
- **Shooting Brake may reuse:** Headless profiling, shape reconstruction, replay, report generation, regression history, and kernel correlation. The Flask UI remains optional and outside the runtime contract.
- **Shooting Brake must build:** Semantic spans for route decision, ownership, queueing, each CUDA/XPU transfer, remap, GEMMs, activation, gather, join, and fallback, with expert/residency/placement/PCIe/epoch/fallback metadata. See [benchmarking.md](benchmarking.md) and [expert-fabric.md](expert-fabric.md).
- **Does not prove:** Expert identity, residency, scheduling rationale, pinned versus pageable transfer, direction, PCIe path, queue wait, overlap, placement epoch, fallback cause, or NVMe recovery without the added spans. Profiling infrastructure alone is not performance evidence.

### LLM.xpu

- **Provenance:** `intel-xpu/LLM.xpu/`; origin `xinming-wei/LLM.xpu`; inspected revision `689be270aa29`.
- **Recorded source observation:** The useful recorded pattern allocates reusable shared host activation buffers, divides leading-dimension rows by device ownership, points asynchronous device requests at those slices, joins explicitly, consumes the logical output in place, and permits preemption only at stage boundaries.
- **Shooting Brake may reuse:** Lifecycle invariants for a pinned-host ring, row-sliced ownership, asynchronous submission, explicit joins, and bounded preemption.
- **Shooting Brake must build:** The MoE-specific CUDA-owner/B70 protocol, discrete-device residency, cross-runtime synchronization, expert placement, and exact fallback.
- **Does not prove:** A base runtime for Shooting Brake. Its recorded maintained path is Llama 2/3 through OpenVINO on NPU+iGPU, with no validated MoE, discrete B70 fabric, CUDA execution/interop, expert placement, B70 resident weights, or NVMe expert tier. Borrow the lifecycle, not the runtime.

## Paper and concept ledger

The papers below are design inputs only. Except for the explicitly quoted ATSInfer number, the scoped documents do not record paper performance figures; none are reproduced here. Absence of a recorded figure is intentional and must not be filled from memory or uncited web material.

### ATSInfer

- **Provenance:** *ATSInfer: Automated Tensor Scheduling for Hybrid CPU-GPU LLM Inference*, <https://arxiv.org/abs/2607.10183>; described in `readme.md` as July 2026 work.
- **Contribution recorded in existing documents:** Tensor-level CPU/GPU placement using measured performance density and switching costs, periodic recalibration, and static-plus-dynamic scheduling.
- **Upstream claim, not a local result:** The paper is recorded as reporting up to **1.94× prefill** and **3.29× decode** improvement on its tested consumer CPU/GPU systems. Shooting Brake has not reproduced those numbers, and they are not evidence for a five-GPU cross-vendor machine.
- **Shooting Brake may reuse:** Performance-per-byte ranking, transfer/switching penalties, avoidance of fragmented alternation, hardware-dependent recalibration, and offline initialization of non-expert state-owner placement.
- **Shooting Brake must build differently:** Treat gate, up, and down as one resident remote-expert scheduling unit; do not fragment one routed expert matrix-by-matrix across NVIDIA and Intel devices.
- **Does not prove:** Benefits on the target topology, the weighted-partial protocol, B70 kernel efficiency, cross-vendor tails, or any local placement policy.

### ProMoE

- **Provenance:** *ProMoE*, <https://arxiv.org/abs/2410.22134>.
- **Contribution recorded in existing documents:** Stronger prediction ideas for proactive expert prefetch, including adjacent-layer signals, shallow-aware cache admission, and cancelable predictions. The documents discuss ProMoE together with Fate rather than assigning each mechanism individually.
- **Shooting Brake may reuse:** Prediction ideas later, but only when route-trace replay establishes recall, copies are cancelable, bandwidth is spare, false positives cannot evict required foreground experts, and exact fallback remains available.
- **Shooting Brake must build:** Trace-grounded validation, cancellation, cache admission, bandwidth isolation, and an exact non-predictive path.
- **Does not prove:** Prediction accuracy for GLM-5.2 routes, useful prefetch lead time, safe cache behavior, or latency gains on this hardware. Prediction can never be necessary for correctness.

### Fate

- **Provenance:** *Fate*, <https://arxiv.org/abs/2502.12224>.
- **Contribution recorded in existing documents:** The same combined ProMoE/Fate category: proactive expert prefetch, adjacent-layer signals, shallow-aware admission, and cancelable prediction.
- **Shooting Brake may reuse:** Predictive placement/prefetch ideas under the same replay, cancellation, spare-bandwidth, non-eviction, and exact-fallback gates.
- **Shooting Brake must build:** Local route-trace evaluation and safe prediction lifecycle integrated with versioned placement epochs.
- **Does not prove:** Local recall, transfer overlap, cache efficiency, correctness, or performance. The existing documents do not attribute a distinct Fate result that can safely be claimed separately.

### Pre-gated MoE

- **Provenance:** *Pre-gated MoE*, <https://arxiv.org/abs/2308.12066>.
- **Contribution recorded in existing documents:** It is named in the prior-work source list, but the scoped documents do not record a distinct mechanism, reusable implementation, revision, or performance result for it.
- **Shooting Brake may reuse:** Nothing is approved for direct reuse from the present provenance record. It may remain a conceptual reference for future route-availability or pre-gating investigation only after its precise claims are documented from an approved source pass.
- **Shooting Brake must build:** The canonical router and exact route semantics remain in Colibri; any future hint path must be optional and independently validated.
- **Does not prove:** Route predictability, correctness of speculative routing, expert residency benefit, or any Shooting Brake performance. Its citation alone is not evidence.

### PowerInfer

- **Provenance:** *PowerInfer*, <https://arxiv.org/abs/2312.12456>.
- **Contribution recorded in existing documents:** Contextual activation locality, hot/cold execution, workload-dependent placement, and sparse compute across heterogeneous devices.
- **Shooting Brake may reuse:** These concepts as motivation for measured hot/cold placement and heterogeneous sparse execution.
- **Shooting Brake must build:** The model-specific route telemetry, whole-expert ownership, cross-vendor activation/partial transport, exact fallback, and measured placement objective.
- **Does not prove:** That its locality assumptions hold for GLM-5.2, that its CPU/GPU split maps to a 5090 plus B70s, or that any upstream performance transfers to Shooting Brake.

### DejaVu

- **Provenance:** *Deja Vu*, <https://arxiv.org/abs/2310.17157>.
- **Contribution recorded in existing documents:** It is grouped with PowerInfer as conceptual prior art for contextual activation locality, hot/cold execution, workload-dependent placement, and heterogeneous sparse compute.
- **Shooting Brake may reuse:** The conceptual lesson that activation sparsity and locality are workload-dependent and therefore require measurement rather than static marketing assumptions.
- **Shooting Brake must build:** Canonical MoE routing, resident-expert execution, measured placement epochs, and exact fallback for this model and topology.
- **Does not prove:** GLM expert locality, route-prediction safety, B70 suitability, or local latency/throughput. The existing documents do not assign a distinct DejaVu implementation component for direct reuse.

### Flash-storage / flash-backed LLM inference

- **Provenance:** *Flash-storage LLM inference*, <https://arxiv.org/abs/2312.11514>. The existing source list supplies no repository revision or locally inspected implementation.
- **Contribution recorded in existing documents:** The paper is cited as storage prior art; the approved architecture separately treats NVMe as packed-model source, recovery, background preload, and inactive-context persistence rather than an ordinary per-token expert tier.
- **Shooting Brake may reuse:** Storage-layout, recovery, or background-preload ideas only after they are tied to explicit source findings and local measurements.
- **Shooting Brake must build:** A frozen packed-bank manifest, background loading, DDR5/VRAM residency policy, and a foreground path that does not ordinarily read experts from NVMe. See [memory.md](memory.md).
- **Does not prove:** That a faster SSD improves decode, that flash latency is safe for interactive expert misses, that 64 GiB DDR5 makes the full cold model robust, or that an upstream flash-backed result transfers to this workload.

## Framework and systems ledger

### SGLang

- **Provenance:** Named in the `readme.md` framework-strategy discussion; no repository origin, inspected revision, article, or paper URL is recorded in the scoped provenance.
- **Contribution recorded in existing documents:** Mature continuous batching, request prioritization, agent workloads, prefix caching, structured generation, and production serving.
- **Shooting Brake may reuse:** Initially, SGLang may call or front Colibri at the request boundary. Only after the B70 primitive and placement scheduler are stable may a custom heterogeneous MoE layer invoke the same C ABI fabric.
- **Shooting Brake must build:** The cross-vendor primitive, placement scheduler, transport, and correctness path independently of SGLang.
- **Does not prove:** B70 expert execution, transport correctness, model parity, or performance. Integrating it early would enlarge the debugging surface; it must not reimplement the transport.

### KTransformers

- **Provenance:** Named in the `readme.md` framework-strategy discussion; no repository origin, inspected revision, article, or paper URL is recorded in the scoped provenance.
- **Contribution recorded in existing documents:** CPU expert batching, host-side fallback organization, AMX-oriented deployments, and hybrid scheduling concepts.
- **Shooting Brake may reuse:** CPU batching and fallback organization where they fit Colibri and the actual host ISA. KTransformers-style execution may become more relevant on a future Xeon host with AMX and substantially more RAM.
- **Shooting Brake must build or retain:** Colibri AVX2/AVX-VNNI as the immediate workstation baseline and the exact CPU fallback contract used by the heterogeneous runtime.
- **Does not prove:** Performance on the current Core Ultra desktop CPU, which the existing documents say lacks the server-class AMX/AVX-512 environment assumed by some high-end CPU MoE results; nor does it prove the cross-vendor fabric.

### Mooncake

- **Provenance:** Named in the `readme.md` framework-strategy discussion as a Mooncake-like transfer/storage system; no origin, inspected revision, article, or paper URL is recorded in the scoped provenance.
- **Contribution recorded in existing documents:** A possible transfer/storage-disaggregation model when weights or contexts move across machines.
- **Shooting Brake may reuse:** Nothing on the initial local token path. Reconsider the abstraction only if a future approved design crosses hosts.
- **Shooting Brake must build:** Lower-level local pinned-host rings and explicit CUDA/Level Zero command lifecycles for the five local PCIe cards.
- **Does not prove:** That distributed object transfer meets local per-layer latency, that it is suitable for same-host PCIe devices, or that storage disaggregation belongs in the initial architecture.

## Interpretation and update policy

This ledger establishes provenance and responsibility, not adoption success. When a source is re-inspected or a local experiment is completed:

1. Record the exact source origin, revision, patch state, model/format, hardware, driver/runtime, inputs, and command in the appropriate implementation or benchmarking document.
2. Keep upstream claims separate from locally observed measurements.
3. Promote a candidate from “may reuse” only after the correctness gates in [correctness.md](correctness.md) and the measurement rules in [benchmarking.md](benchmarking.md) are satisfied.
4. If new evidence conflicts with this ledger, update the build-versus-borrow boundary explicitly rather than silently turning a reference project into a runtime dependency.
