# Shooting Brake Prior-Art and Provenance Ledger

## Purpose and authority

This ledger records which upstream projects supply production components, which supply mechanics or ideas only, and which local results are proven reference evidence. [`../plan.md`](../plan.md) is the sole authoritative active architecture and implementation plan. This document does not claim that the planned upstream-vLLM-plus-B70 production integration is implemented.

The source inspection and B70 kernel evidence summarized here reflect the workspace recorded on 2026-08-04. They are qualification evidence for named components, not a claim that the planned production integration has been built, tested, or benchmarked end to end.

## Evidence classes

- **Inspected source fact** — behavior or an interface observed in a named checkout. It is not a local production result.
- **Upstream claim** — a result reported by an external project, article, or paper. It is never Shooting Brake benchmark evidence.
- **Proven Colibri reference evidence** — behavior measured in `colibri-variants/colibri-qwen36/`. It establishes feasibility for that implementation only.
- **QuixiCore-XPU B70 qualification evidence** — behavior measured for the named kernel checkout on the actual B70. It qualifies that kernel experiment only, not the production provider process or vLLM integration.
- **Production decision** — the required destination architecture from [`../plan.md`](../plan.md). It remains planned until the corresponding Phase 0–10 gate is completed.
- **Unverified production assumption** — a proposition requiring evidence against upstream vLLM and the isolated B70 provider.

Repository revisions identify what was inspected; they do not assert that a checkout has remained unchanged. Compatibility must be established through the model/provider manifest and correctness gates, never inferred from similar dimensions, dtypes, device names, or model-family names.

## Authoritative source boundary

| Source | Production role | Explicit non-role |
|---|---|---|
| Upstream vLLM 0.26+ | RTX 5090 CUDA state owner and serving host: scheduler, continuous batching, attention, KV/recurrent state, canonical router/top-k, local routed experts, shared expert, residual, LM head, sampling, and APIs. | It does not directly dispatch CUDA tensors to XPU operators and is not replaced by an old patched vLLM fork. |
| `QuixiCore-XPU/` | Selected primary B70 provider kernel source: native SYCL NVFP4 MoE kernels with a framework-neutral C++ ABI and optional PyTorch binding. | It executes only preselected routed experts; it is not the CUDA model host, router/top-k authority, shared expert, transport, or serving runtime. |
| `intel-xpu/llm-scaler/` plus `vllm-xpu-kernels` | Secondary INT4 W4A16 kernel alternative if NVFP4 quality is insufficient. | Its vLLM 0.21.0 patch is not the CUDA host and must not be applied unchanged to upstream vLLM 0.26+; it is not the selected primary provider source. |
| `sonar/` | Protocol-design reference for an XPU-aware vLLM/Aphrodite fork and modular MoE seam. | AGPL-3.0; not adopted as the production host, and its XPU kernels remain external through the same `vllm-xpu-kernels` wheel. |
| Shooting Brake | Qwen-scoped out-of-tree `HybridMoERunner` / `HybridRoutedExperts`, provider client, versioned pinned-memory request ring, capability/model/provider manifests, placement ownership, join, telemetry, and exact failed-route recovery. | It does not fork the full CUDA host or silently qualify arbitrary models and kernel fallbacks. |
| ExLlamaV3 | Connector-mechanics reference for a pinned shared-memory ring, sequence protocol, and CUDA stream-ordered flag publication/wait. | It is not the model host, MoE runtime, weight format, placement policy, or serving stack. |
| Colibri Qwen3.6 variant | Proven transport, correctness, placement, issue/take, and failure-semantics reference; its native B70 worker remains a correctness/latency comparator. | It is not the production CUDA model host and its single-token timings do not predict production vLLM performance. |

The intended production transaction is:

```text
upstream vLLM 0.26+ on RTX 5090
    -> canonical CUDA router and top-k
    -> Qwen-scoped HybridMoERunner / HybridRoutedExperts
       |- local routes -> qualified stock-compatible CUDA MoE backend
       `- remote routes -> versioned pinned-memory request ring
                           -> isolated persistent QuixiCore-XPU B70 provider
                           -> qualified QuixiCore-XPU NVFP4 MoE kernels
                           -> weighted [M_remote, hidden] wire partial plus token_row_map
    -> asynchronous CUDA copy and addition
```

CPU work is limited to orchestration, placement, queue management, telemetry, and exact emergency recovery. No source is approved to introduce normal-path CPU matrix multiplication.

## Primary source ledger

### Upstream vLLM

- **Provenance:** local checkout `vllm/`, inspected as `v0.26.1rc0-285-g1c0d20791`.
- **Inspected source facts:** the Qwen3.5/Qwen3-next path provides CUDA-owned attention and recurrent state, router and top-k, shared and routed experts, and modular MoE construction. `FusedMoEFactory` accepts injected runner and routed-expert classes, and out-of-tree layer/custom-op registration exists.
- **Production decision:** retain upstream vLLM as the sole CUDA state owner and serving runtime. Inject a Qwen-scoped post-top-k adapter through `HybridMoERunner` and `HybridRoutedExperts`; preserve the canonical vLLM router/top-k and stock non-routed model path.
- **Required local work:** qualify the exact adapter seam, CUDA local-route compaction/skip semantics, eager execution, piecewise graph boundary, all-CUDA parity, failure recovery, and upgrade compatibility.
- **Does not prove:** that remote routes work, that a selected CUDA MoE backend accepts `-1` expert mappings, that full CUDA graphs can contain external XPU work, or that any B70 performance claim transfers to production.

### QuixiCore-XPU NVFP4 MoE

- **Provenance:** local `QuixiCore-XPU/` checkout, inspected from source and built on the Intel Arc Pro B70 on 2026-08-04. It is an MIT-licensed native SYCL kernel library with a framework-neutral raw-pointer C++ ABI and an optional PyTorch binding.
- **Inspected source facts:** its purpose-built `nvfp4_moe` fused and split operations accept preselected `int32` expert IDs and `float32` routing weights. The NVFP4 contract uses packed E2M1 weights, E4M3 block scales over blocks of 16, and per-expert FP32 global scales. Nonblocking dispatch returns a SYCL event for asynchronous chaining; this API fact does not claim that the production provider process is implemented.
- **B70 qualification evidence:** all operations passed the correctness smoke gate on the B70, including `nvfp4_moe` fused and split, with maximum absolute error approximately `1e-9`. Reported split-kernel results were a `46.5 µs` median at `M=1` (`270 GB/s` weight bandwidth), a `61.5 µs` median at `M=2` (`409 GB/s`), approximately `60–215 µs` at `M=4` (`233 GB/s`), a `173.8 µs` median at `M=8` (`579 GB/s`), and a `343.3 µs` median at `M=16` (`586 GB/s`).
- **Production decision:** use only qualified preselected-route QuixiCore-XPU NVFP4 MoE operations in the separate persistent B70 provider. The provider accepts canonical IDs and routing weights, owns persistent compact B70 weights and preallocated buffers, and returns one weighted `[M_remote, hidden]` wire partial plus the row map used by CUDA to construct the full `[M, hidden]` contribution.
- **Format rule:** the production B70 artifact is NVFP4 and derives from the same higher-precision source checkpoint as the separately converted CUDA NVFP4 or FP8 artifact. Conversion occurs offline or at initialization, never on the token path.
- **Does not prove:** production process isolation, ring correctness, batched provider behavior, vLLM integration, cross-runtime failure recovery, end-to-end production latency, or production throughput.

### llm-scaler and vLLM XPU kernels

- **Provenance:** `intel-xpu/llm-scaler/`; latest inspected image/release `intel/llm-scaler-vllm:0.21.0-b1`. Its environment pins a `vllm-xpu-kernels` revision and carries an older safety patch. The separate checkout `intel-xpu/vllm-xpu/vllm-xpu-kernels/` was inspected at `dd3bc2127cff`.
- **Inspected source facts:** the stack contains B70/Xe2 ESIMD INT4 MoE families for tiny decode, small/batched decode, grouped routes, and prefill gather/up/down/weighted accumulation. Its operators consume XPU tensors and obtain an XPU/SYCL stream; they are not a CUDA-to-XPU transport mechanism. Some full-fused entry points own routing or shared-expert work and therefore cross the approved boundary.
- **Production role:** this is the qualified secondary INT4 W4A16 alternative if primary QuixiCore-XPU NVFP4 quality is insufficient. Only preselected-route compute operations may enter the isolated provider; llm-scaler remains a kernel-design reference, not the selected host or primary provider source.
- **Version rule:** the vLLM 0.21.0 patch is historical B70-kernel provenance, not a patch set for the upstream 0.26+ CUDA host. Kernel bindings, layouts, shapes, safety guards, activation behavior, and numerical tolerance require qualification for any pinned secondary provider release.
- **Format rule:** the secondary signed-S4 GS128 contract is distinct from primary NVFP4 and from Colibri's reference GS64 contract. Any secondary artifact derives independently from the identical higher-precision source model; conversion or requantization occurs offline or at initialization, never on the token path.
- **Does not prove:** equivalence to the primary NVFP4 path, production process isolation, ring correctness, vLLM integration, failure recovery, or production performance.

### Sonar modular MoE seam

- **Provenance:** local `sonar/` checkout; AGPL-3.0 vLLM/Aphrodite fork with XPU platform support.
- **Inspected source facts:** `modular_kernel.py` exposes a modular MoE seam useful for reasoning about provider protocol boundaries. Sonar's XPU kernels are external through the same `vllm-xpu-kernels` wheel rather than an in-tree native kernel source.
- **Approved reuse:** protocol-design concepts only.
- **Explicit non-adoption:** Sonar is not the production host or provider. Its AGPL license, full-runtime fork, and external-kernel dependency keep it outside the selected upstream-vLLM-plus-QuixiCore architecture.

### ExLlamaV3 connector mechanics

- **Provenance:** `exllamav3-quant-inference/exllamav3/`, especially `exllamav3/model/moe_cpu_host.py` and `exllamav3_ext/cpu/moe_handoff.{h,cu}`.
- **Inspected source facts:** the connector uses a pinned ring carrying FP16 activations, selected expert IDs, routing weights, and FP32 output, with sequence-based coordination and `cuStreamWriteValue32` / `cuStreamWaitValue32`.
- **Approved reuse:** port the ring lifecycle, sequence discipline, and CUDA stream-memory mechanics where they fit the Shooting Brake versioned process protocol.
- **Required changes:** replace the CPU expert worker with the isolated B70 provider; add batch/token-route metadata, placement and weight generations, stale-completion rejection, provider restart semantics, per-route status, and negotiated capacities.
- **Explicit non-adoption:** do not port ExLlamaV3's `BlockSparseMLP`, all-or-nothing CPU-offload branch, contiguous expert ownership, EXL3 weight assumptions, or serving runtime.
- **Does not prove:** that Level Zero completion participates in a CUDA graph, that the connector is safe unchanged across processes, or that ExLlamaV3 is an appropriate production host.

### Colibri Qwen3.6 reference

- **Provenance:** `colibri-variants/colibri-qwen36/`; origin recorded as `JustVugg/colibri`. No inspected upstream revision is recorded here.
- **Proven reference evidence:** the local variant demonstrates persistent B70 expert weights, compact `(layer, expert) -> slot` ownership, exact signed-S4 GS64 conversion, asynchronous issue/take, canonical selected-route execution, routing-weighted hidden-size partials, numerical agreement with its CPU reference, correct end-to-end CUDA+B70 generation, zero normal-path CPU expert fallback in the recorded suite, and exact failed-route recomputation.
- **Scope of proof:** the current transaction is single-token and the native worker has one in-order queue, one pending operation, and fixed scratch. The controlled measurements and traces in [progress.md](progress.md) are Colibri reference results.
- **Production reuse:** preserve the lifecycle invariants, placement semantics, exact route ownership, recovery semantics, and native worker as a comparator.
- **Explicit non-adoption:** Colibri is not the production state owner. Its CUDA model path, serving stack, single-token protocol, and timings are not substitutes for upstream vLLM Phase 0–10 acceptance.
- **Does not prove:** a batched QuixiCore-XPU B70 provider, continuous-batch decode, grouped prefill, the versioned process ring, an out-of-tree vLLM adapter, piecewise CUDA graphs, provider restart, or production performance.

## Secondary source ledger

These sources may inform a narrow subsystem. None changes the primary production decision.

| Source | Reusable input | Boundary and missing evidence |
|---|---|---|
| Luce Spark / Lucebox | Traffic-derived hotness, bounded residency, durable placement profiles, and asynchronous promotion concepts. | Shooting Brake must build cost-aware CUDA/B70 placement from measured routes; Lucebox does not prove this topology or provider. |
| vllm-xpu-breakdown | Headless XPU profiling, shape reconstruction, replay, kernel correlation, and reports. | Shooting Brake must add semantic route, ownership, queue, copy, join, generation, and recovery spans. Profiling tooling is not performance evidence. |
| LLM.xpu | Reusable host-buffer lifecycle, row slicing, asynchronous submission, explicit join, and bounded stage preemption. | Borrow lifecycle invariants only; it is not the runtime or a proven discrete-B70 MoE fabric. |
| Intel XPU Triton backend | Later MXFP4 provider candidate after explicit conversion. | MXFP4 E2M1 is not Colibri integer W4 by inspection and is not the initial production provider. |
| Xe-Fuse | BF16 fused-compute reference and possible future provider techniques. | It lacks the qualified W4 routed-provider path and does not establish production B70 performance. |
| Xe-Forge | Candidate generation and autotuning after correctness. | Synthetic shapes and disabled comparison are not acceptance; generated kernels require deterministic oracle validation and tail checks. |
| ATSInfer | Measured-cost placement, switching penalties, and recalibration concepts. | Keep each routed expert whole. Its reported gains are upstream claims and do not transfer to this system. |
| ProMoE, Fate, and Pre-gated MoE | Optional future prediction/prefetch ideas. | Prediction is never required for correctness and must not evict foreground residency or bypass exact recovery. |
| PowerInfer and DejaVu | Locality and heterogeneous sparse-execution concepts. | Route locality is workload-dependent and must be measured for the qualified model. |
| Flash-storage inference | Startup, recovery, and background-staging ideas. | It is not permission to place ordinary foreground decode expert reads on NVMe. |
| KTransformers | CPU fallback organization concepts. | CPU is exact emergency recovery only, not a normal routed-expert tier or production matrix path. |
| SGLang and Mooncake | Possible future request-boundary or cross-host concepts. | Neither belongs in the initial local provider path or replaces upstream vLLM and the pinned ring. |

The ATSInfer figures previously recorded—up to 1.94× prefill and 3.29× decode on its tested systems—remain upstream claims only. They are not Shooting Brake results and are not evidence for the RTX 5090/B70 path.

## Provenance and promotion rules

1. Production claims must cite the exact upstream vLLM commit, provider image and dependencies, model source, CUDA and B70 artifact fingerprints, protocol version, capability manifest, placement generation, hardware, and workload.
2. Colibri measurements must be labeled **Colibri reference evidence** and compared only with like-for-like Colibri configurations.
3. Production performance is accepted only by Phase 10 against the same upstream all-CUDA vLLM workload.
4. A new model remains on stock upstream vLLM CUDA until its architecture, routing semantics, dimensions, weight artifacts, kernel family, and mixed-result numerics pass an explicit capability qualification.
5. Generated, converted, or autotuned kernels are candidates. Canonical route semantics and a common higher-precision model source remain the authority.
6. Prediction, storage, and fallback mechanisms may improve availability or recovery but must not alter exactly-once selected-route execution.
7. If implementation evidence changes a source boundary, update [`../plan.md`](../plan.md) first; this ledger follows that decision rather than creating a second architecture.
