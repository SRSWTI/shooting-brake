# Shooting Brake Progress

Last updated: 2026-08-04

## Status at a glance

The project has **proven a Colibri-based CUDA+B70 reference path**. It has **not implemented or qualified the production upstream-vLLM integration described in [`../plan.md`](../plan.md)**.

| Area | Status | Meaning |
|---|---|---|
| Colibri CUDA+B70 reference | Proven for the recorded single-token Qwen3.6 GS64 path | Establishes transport, residency, selected-route compute, placement, exact recovery, and end-to-end feasibility. |
| Native B70 comparator | Proven for the recorded shapes and tests | Provides a Torch-free correctness/latency baseline derived from QuixiCore-XPU NVFP4 MoE kernels. |
| Upstream vLLM 0.26+ CUDA state owner | Planned, not integrated | The inspected checkout and injection seams are known; no production `HybridMoERunner` path is complete. |
| Isolated persistent QuixiCore-XPU B70 provider | Planned, not implemented | The intended primary NVFP4 batched decode/prefill provider does not yet exist; llm-scaler remains a secondary INT4 fallback. |
| Versioned batched pinned-memory request ring | Planned, not implemented | The current Colibri transaction is single-token and not the production protocol. |
| Qwen-scoped out-of-tree adapter | Planned, not implemented | Canonical vLLM routing, local/remote partition, weighted-partial join, and eager parity remain to be built. |
| Phase 0–10 production qualification | Not complete | No phase gate should be inferred from the Colibri evidence. |

## Production target

```text
upstream vLLM 0.26+ on RTX 5090
    |- scheduler, continuous batching, serving
    |- attention, KV/recurrent state
    |- canonical router/top-k
    |- CUDA local routed experts and shared expert
    `- Qwen-scoped HybridMoERunner / HybridRoutedExperts
          -> versioned pinned-memory request ring
          -> isolated persistent QuixiCore-XPU B70 provider
          -> qualified QuixiCore-XPU tiny/batched/prefill NVFP4 MoE kernels
          -> secondary llm-scaler INT4 / vllm-xpu-kernels fallback
          -> weighted [M_remote, hidden] wire partial plus token_row_map
          -> asynchronous CUDA join
```

The CPU performs orchestration, placement, queue management, telemetry, and exact emergency recovery only. It does not perform normal-path matrix multiplication.

The inspected production baselines are:

- upstream CUDA host checkout: `vllm/` at `v0.26.1rc0-285-g1c0d20791`;
- primary B70 kernel/provider source: `QuixiCore-XPU/`, native MIT-licensed SYCL NVFP4 MoE library with a framework-neutral C++ ABI and PyTorch binding;
- secondary B70 kernel alternative: `intel-xpu/llm-scaler/`, ESIMD INT4 kernel-design reference and fallback if NVFP4 quality is insufficient; latest inspected vLLM release `intel/llm-scaler-vllm:0.21.0-b1`;
- protocol-design reference: `sonar/`, an AGPL-3.0 vLLM/Aphrodite XPU fork with a modular MoE seam whose XPU kernels remain external; it is not adopted as the production host;
- mechanics reference: ExLlamaV3 pinned ring and CUDA stream-memory coordination only;
- proven behavioral reference: `colibri-variants/colibri-qwen36/`.

The llm-scaler vLLM 0.21.0 patch remains a reference-only secondary INT4 alternative; it is not the production CUDA host and must not be applied unchanged to vLLM 0.26+. Selecting that patch instead of primary QuixiCore-XPU would reintroduce the same host-version divergence.

## Completed Colibri+B70 reference evidence

Except for the explicitly labeled QuixiCore-XPU benchmark evidence, this section records **Colibri reference evidence**, not proof of the planned production runtime. The QuixiCore-XPU results prove only the measured provider kernels, not the production integration.

### Reference model and path

The completed work used the Qwen3.6-35B-A3B Colibri INT4 GS64 checkpoint:

```text
/home/shooting-brake007/.cache/huggingface/hub/models--Kreuzzelg--qwen36-35b-a3b-colibri-i4-gs64/snapshots/c619aa594ad1e70af82168fb6b4878427896e21c
```

Recorded model facts:

- hidden size 2,048;
- 40 layers;
- 256 routed experts per layer;
- top-k 8;
- 10,240 layer-expert weight sets.

In the reference path, the RTX 5090 owns embeddings, dense projections, DeltaNet state, attention KV cache, router and canonical top-k, shared and hot routed experts, final norm, LM head, and greedy selection. One Intel B70 owns a compact persistent overflow-expert bank and returns one weighted partial per token through pinned/USM host buffers.

### Transport and device bring-up

- PyTorch XPU, oneAPI compiler, and Level Zero access were validated against an Intel Arc Pro B70 reporting approximately 34.2 GB.
- A CUDA plus Level Zero pinned-host transport benchmark selected the discrete B70 and completed the activation/partial lifecycle.

| Colibri/reference transport measurement | Result |
|---|---:|
| Full 4 KB activation + 8 KB partial round trip | 43.5 µs |
| Pinned host to B70, 4 KB | 9.2 µs |
| B70 to pinned host, 8 KB | 8.4 µs |
| B70 host-to-device, 1 MiB | 353.9 µs / 2,963 MB/s |

These values support host-staged feasibility for the measured path. They are not a production vLLM latency result.

### Native B70 kernel and worker

- Inspected and built QuixiCore-XPU on the actual B70, then passed its correctness smoke gate for all operations, including `nvfp4_moe` fused and split execution.
- Established that Colibri signed-S4 GS64 weights cannot be reinterpreted as GS128 without changing scale semantics.
- Preserved signed S4 values and group size 64; converted scales to FP16; converted gate/up and down projections once during expert upload.
- Implemented a Torch-free SYCL/ESIMD shared library with persistent expert storage, compact `(layer, expert) -> slot` mapping, pinned/USM buffers, separate issue/take entry points, asynchronous exception propagation, and exact failed-route masks.
- Fixed an early cross-layer slot aliasing bug.

#### QuixiCore-XPU NVFP4 MoE benchmark evidence

QuixiCore-XPU `nvfp4_moe` was benchmarked on B70 on 2026-08-04 after the correctness gate passed. All operations passed, including fused and split `nvfp4_moe`; maximum absolute error was approximately $10^{-9}$.

| Split `nvfp4_moe` workload | Median latency | Effective weight bandwidth |
|---|---:|---:|
| `M=1` | 46.5 µs | 270 GB/s |
| `M=2` | 61.5 µs | 409 GB/s |
| `M=4` | 60–215 µs | 233 GB/s |
| `M=8` | 173.8 µs | 579 GB/s |
| `M=16` | 343.3 µs | 586 GB/s |

This is direct provider-kernel evidence, not an implemented or qualified upstream-vLLM production path.

| Colibri native GS64 reference measurement | Result |
|---|---:|
| Deterministic one-route worker test | 39.0 µs |
| Deterministic eight-route worker test | 46.0 µs total |
| Eight-route cosine versus Colibri CPU reference | 1.00000012 |
| Eight-route maximum absolute error | 0.000002 |

Early PyTorch/oneMKL prototypes also established feasibility but were rejected as the current comparator because repeated launches dominated:

| Historical prototype measurement | Result |
|---|---:|
| B70 BF16 4096×8192×4096 GEMM | 1.91 ms / 144.2 TFLOPS |
| PyTorch XPU model-shape expert pipeline | 45.3 µs/expert |
| oneMKL FP32 heterogeneous end-to-end | approximately 2.73 tok/s |
| oneMKL BF16 heterogeneous end-to-end | approximately 2.56 tok/s |

### Colibri CUDA state owner, ownership, and recovery

The reference implementation added:

- persistent CUDA activation, attention KV, DeltaNet recurrent/convolution, dense, router, expert, and output state;
- canonical CUDA top-k, route weighting, shared expert, routed expert, LM-head, and greedy-selection operations;
- disjoint CUDA/B70 ownership and asynchronous issue before collection;
- exclusion of B70-owned experts from the CUDA LFRU tier;
- exact direct/recovery route masks and transactional whole-step rollback;
- separate CUDA, B70, direct-CPU, and recovery-CPU counters;
- failure injection proving rollback to the Colibri CPU oracle.

Recorded correctness observations:

- a deterministic 16-token Colibri CPU-versus-CUDA trace selected the same token sequence;
- CUDA-versus-CPU vocabulary-logit cosine was approximately 0.997340;
- maximum absolute vocabulary-logit difference was approximately 0.158876;
- greedy selected token IDs agreed;
- the B70 eight-route comparison achieved the values above;
- forced CUDA whole-step failure restored state and continued through the CPU oracle.

These are reference observations with the recorded quantization and runtime. Production vLLM tolerances and recovery behavior remain unqualified.

### Colibri placement and benchmark evidence

A controlled placement resident across the two GPUs assigned:

- 5,640 layer-experts to CUDA (55.1%, 11.59 GB);
- 4,600 layer-experts to B70 (44.9%, 7.16 GiB);
- zero CPU-owned experts in the normal path.

All 10,240 routed-expert weight sets were resident. The hotter CUDA bank received more traffic by design.

Controlled 256-token reference comparison:

| Colibri configuration | Step latency | Throughput | CPU expert fallback | Peak host RSS |
|---|---:|---:|---:|---:|
| All routed experts on RTX 5090 | 37.6 ms/token | 25.57 tok/s | 0 | 22.51 GB |
| RTX 5090 12 GB + B70 ESIMD | 38.2 ms/token | 25.13 tok/s | 0 | 22.47 GB |
| RTX 5090 12 GB + CPU overflow | 109.0 ms/token | 8.65 tok/s | nonzero | 31.06 GB |

Within this controlled Colibri comparison, the B70 configuration retained approximately 98.3% of the all-5090 throughput and was approximately 2.9× faster than CPU overflow. This must not be restated as vLLM production performance.

The four-workload 45% residency suite forced 512 output tokens per run with `IGNORE_EOS=1`:

| Workload | E2E tok/s | Step ms | Steady tok/s | TTFT | CUDA routes | B70 routes | B70 issue-to-take |
|---|---:|---:|---:|---:|---:|---:|---:|
| STEM | 17.66 | 53.3 | 18.76 | 1.74 s | 61.0% | 39.0% | 60.6 µs |
| Science | 17.00 | 54.7 | 18.28 | 2.16 s | 84.0% | 16.0% | 53.1 µs |
| General knowledge | 17.69 | 53.3 | 18.76 | 1.72 s | 76.2% | 23.8% | 53.5 µs |
| Creative writing | 17.99 | 52.7 | 18.98 | 1.54 s | 61.0% | 39.0% | 59.0 µs |

Aggregate recorded reference facts:

- 2,048 output tokens;
- mean/median E2E throughput 17.585/17.675 tok/s, range 17.00–17.99;
- mean steady decode 18.695 tok/s;
- mean decode step 53.5 ms/token;
- mean TTFT 1.79 s;
- weighted B70 route share 29.31%, range 16.0–39.0%;
- weighted B70 issue-to-take 56.81 µs;
- zero normal and zero recovery CPU expert routes;
- zero LFRU swaps;
- peak host RSS 22.34–22.35 GB;
- approximately 7.58 GB less RTX expert storage than the earlier 19.17 GB full-RTX expert allocation.

The 44.9% B70 residency share did not imply equal route traffic: measured placement deliberately retained the hottest experts on CUDA. `IGNORE_EOS=1` makes these equal-length performance traces, not a quality evaluation.

Durable trace artifacts:

| Artifact | SHA-256 |
|---|---|
| `experiments/traces/qwen_45pct_stem.trace.log` | `93e98feca7f5b405a2f5095aadf7882b289d4707e78ffca52b69ce015b876c75` |
| `experiments/traces/qwen_45pct_science.trace.log` | `792f8d22b4a3c1c6e363a4fa369016872f2608e9b60ae9c93c32259b6408b825` |
| `experiments/traces/qwen_45pct_general.trace.log` | `d5265f2394f3a18c7a81f95fbe493600ab32133fffd4e096901fa676eaa74f4e` |
| `experiments/traces/qwen_45pct_creative.trace.log` | `a58e4448bac1bc9a46eccf974ff779e50c2cf64964468b3b6ded755dee856def` |

The corresponding prompt files remain beside those traces.

### Reference implementation artifacts

The principal completed reference code is:

- `colibri-variants/colibri-qwen36/c/b70_moe_sycl.cpp` — native persistent GS64 B70 comparator;
- `b70_tier.{c,h}` — loading, compact ownership, issue/take, readiness, and failure propagation;
- `qwen36_cuda.{c,h}` and `backend_cuda.{cu,h}` — persistent Colibri CUDA state-owner implementation;
- `qwen36_tier.{c,h}` — disjoint ownership, dispatch, recovery, and telemetry;
- `experiments/test_b70_int4_path.py` — deterministic conversion and route-output validation;
- `experiments/transport_test.c` and related diagnostics — host-staged transport evidence.

Historical oneMKL/Python workers remain investigative artifacts and are not the planned provider.

## Phase 0–10 production status

| Phase | Status | Work still required |
|---|---|---|
| 0 — Freeze baselines and compatibility contracts | **Not complete** | Freeze exact host/provider/dependency versions, source checkpoint and both artifact fingerprints, fixed correctness/performance workloads, all-CUDA vLLM eager/graph baselines, versioned provider protocol, and model/provider capability schemas. |
| 1 — Isolated QuixiCore-XPU B70 provider | **Not implemented** | Build the persistent isolated provider around qualified QuixiCore-XPU NVFP4 MoE operators, explicit device selection, grow-only tensors, load/issue/take/health API, kernel policy, and direct `M=1`, `M=2..32`, and prefill qualification; retain llm-scaler INT4 only as a separately qualified secondary fallback. |
| 2 — Batched pinned-memory protocol | **Not implemented** | Build multiple versioned ring slots, batch descriptors, token/route map, sequence and generation checks, acquire/release publication, per-route status, timeout, restart, and stale-reply rejection. |
| 3 — Independent provider mathematics | **Not complete** | Validate weighted `[M_remote, hidden]` wire results and row maps over the full route/shape/failure matrix, verify CUDA scatter into `[M, hidden]`, and validate CUDA/B70 artifacts against one higher-precision source model. |
| 4 — Upstream-vLLM adapter | **Not implemented** | Add Qwen-scoped `HybridMoERunner`, `HybridRoutedExperts`, and provider client; prove all-CUDA adapter parity before remote routes. |
| 5 — Compact expert ownership | **Not implemented in vLLM** | Load immutable CUDA/B70 compact banks from a validated placement manifest; prove exactly one normal owner and no inference-time weight movement. |
| 6 — Eager hybrid execution | **Not implemented** | Partition canonical vLLM routes, overlap local CUDA and remote B70 execution, copy/add the remote partial before final reduction, and validate layer/logit/generation agreement. |
| 7 — Continuous-batch decode and grouped prefill | **Not implemented** | Aggregate scheduler-step rows into one layer request and qualify tiny/grouped/prefill QuixiCore-XPU NVFP4 MoE paths across mixed scheduling and cancellation, with llm-scaler INT4 / vllm-xpu-kernels retained only as the secondary fallback. |
| 8 — Piecewise CUDA graphs | **Not implemented** | Add an explicit break around the external provider and remove exposed waits/device-wide synchronization. ExLlama stream-memory mechanics are only an optimization reference. |
| 9 — Failure/restart operations | **Not implemented** | Add heartbeat, bounded timeouts/queues, provider restart and generation bump, exact batched failed-route recovery, rollback, and structured telemetry. |
| 10 — Controlled production benchmark | **Not run** | Compare stock all-CUDA vLLM, CUDA+B70, CUDA+CPU cold experts, reduced-CUDA control, and native comparator with identical source/workload settings and full correctness/capacity/tail accounting. |

No row is marked complete merely because an analogous behavior works in Colibri.

## Next concrete deliverables

Work proceeds in the Phase 0–10 order from [`../plan.md`](../plan.md). The immediate deliverables are:

1. **Phase 0 frozen contract package:** exact upstream vLLM and provider dependency fingerprints; source checkpoint plus CUDA/B70 artifact fingerprints; fixed correctness prompts and production workload; stock all-CUDA vLLM eager and supported graph baselines; versioned request/completion schema; model, weight, placement, and provider capability manifests with fail-closed startup validation.
2. **Phase 1 isolated provider:** a persistent isolated B70 provider using only qualified QuixiCore-XPU NVFP4 MoE operators for the primary path, with explicit B70 selection, resident compact weights, grow-only buffers, capability/load/issue/take/health/shutdown operations, and measured kernel selection for `M=1`, batched decode, and prefill; llm-scaler INT4 / vllm-xpu-kernels remains a separately qualified secondary fallback if NVFP4 quality is insufficient.
3. **Phase 2 versioned batched ring:** multiple pinned slots, token/route maps, request and provider generations, stale-completion rejection, per-route failure status, bounded capacities, and stress coverage for wraparound, concurrency, restart, and injected failure.
4. **Phase 3 provider oracle matrix:** batched weighted-partial comparisons for mixed ownership, duplicate/non-sorted experts, repeated experts across tokens, boundary IDs, near-zero weights, already-staged rows whose valid remote subset becomes empty, invalid generations, and failures.
5. **Phase 4 all-CUDA adapter parity:** Qwen-scoped out-of-tree `HybridMoERunner` / `HybridRoutedExperts` installed in upstream vLLM, with stock-equivalent all-CUDA behavior before any B70 route is enabled.

Only after these deliverables pass their gates should eager hybrid execution, continuous batching/prefill, piecewise graphs, operational recovery, and the Phase 10 production benchmark be described as active or complete.
