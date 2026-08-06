# Shooting Brake Progress

Last updated: 2026-08-04

## Status at a glance

The project has **proven a Colibri-based CUDA+B70 reference path** and completed the production-oriented Phase 0 through Phase 3 gates. It has **not implemented or qualified the upstream-vLLM adapter, CUDA scatter/join, or end-to-end production integration described in [`../plan.md`](../plan.md)**.

| Area | Status | Meaning |
|---|---|---|
| Colibri CUDA+B70 reference | Proven for the recorded single-token Qwen3.6 GS64 path | Establishes transport, residency, selected-route compute, placement, exact recovery, and end-to-end feasibility. |
| Native B70 comparator | Proven for the recorded shapes and tests | Provides a Torch-free correctness/latency baseline derived from QuixiCore-XPU NVFP4 MoE kernels. |
| Upstream vLLM 0.26+ CUDA state owner | Planned, not integrated | The inspected checkout and injection seams are known; no production `HybridMoERunner` path is complete. |
| Isolated persistent QuixiCore-XPU B70 provider | **Phase 1 implemented and direct-provider gate passed** | Explicit B70 selection, full 8,192-expert NVFP4 bank load, fixed USM buffers, protocol-v1 capability, issue/take, health/shutdown, split/fused policy, generation/sequence rejection, and direct correctness/allocation-stability tests pass. |
| Versioned batched pinned-memory request ring | **Phase 2 implemented and direct ring gate passed** | The protocol-v2 eight-slot shared ring, process-isolated B70 service, canonical route payloads, lifecycle/failure semantics, two-million-request wraparound stress, real-B70 numerical validation, and warmed latency percentiles pass. This is a direct provider/ring gate, not upstream-vLLM integration. |
| Independent provider-wire mathematics | **Phase 3 complete; physical-B70 gate passed** | Frozen BF16-source/NVFP4 fixture provenance, byte-audited bank records, compact canonical-ID remapping, weighted `[M_remote, hidden]` results, route/status identity, zero-publication, failure, and boundary cases passed before CUDA scatter. |
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

#### Phase-1 persistent provider evidence

The production-oriented Phase-1 provider now loads the full 32-layer, 8,192-expert NVFP4 bank into one explicitly selected `Intel(R) Arc(TM) Pro B70 Graphics`. The bank is 14,495,580,220 bytes including its 60-byte header; layers `32..39` of the mixed-precision checkpoint are FP8 and remain CUDA-local unless separately converted and qualified.

The independent representation oracle uses the installed `compressed_tensors.NVFP4PackedCompressor.decompress`. It verifies checkpoint-to-bank packed weights and raw E4M3 scale bytes exactly, verifies both FP32 QuixiCore multipliers, and agrees with the saved expert output at maximum absolute error \(1.070\times10^{-6}\) and RMSE \(3.051\times10^{-7}\).

One full-bank provider run reported:

| Provider case | Kernel | Kernel time | Issue-to-take total | Max abs error | Bad / nonfinite |
|---|---|---:|---:|---:|---:|
| first cold `M=1`, one valid route | split | 473,482.7 µs | 516,785.3 µs | \(2.183\times10^{-10}\) | 0 / 0 |
| warmed `M=1`, one valid route | split | 35.3 µs | 77.6 µs | \(2.183\times10^{-10}\) | 0 / 0 |
| `M=2` | split | 492.5 µs | 529.2 µs | \(2.183\times10^{-10}\) | 0 / 0 |
| `M=4` | split | 56.9 µs | 114.1 µs | \(2.183\times10^{-10}\) | 0 / 0 |
| `M=8` | split | 74.4 µs | 129.3 µs | \(2.183\times10^{-10}\) | 0 / 0 |
| `M=16` | split | 111.7 µs | 201.4 µs | \(2.183\times10^{-10}\) | 0 / 0 |
| `M=32` | split | 210.4 µs | 352.6 µs | \(2.183\times10^{-10}\) | 0 / 0 |
| representative prefill `M=128` | fused | 1,245.1 µs | 1,716.0 µs | \(2.183\times10^{-10}\) | 0 / 0 |
| `M=8`, all top-8 slots valid duplicate expert with unequal weights summing to one | split | 361.9 µs | 431.1 µs | \(2.328\times10^{-10}\) | 0 / 0 |

The first row includes one-time SYCL/kernel warmup and is intentionally separated from warmed dispatch. These are single observed provider-test timings, not production latency distributions. The test also rejects invalid layer/IDs, stale generation/sequence, and concurrent issue; keeps the provider allocation count unchanged across nine successful mixed-shape dispatches; and verifies idempotent shutdown plus post-shutdown rejection.

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
- `phase1/b70_provider.{hpp,cpp}` — full-bank QuixiCore-XPU provider core with fixed buffers and issue/take lifecycle;
- `phase1/b70_provider_main.cpp` — isolated persistent capability/health/shutdown control process;
- `phase1/b70_provider_tests.cpp` — direct shape, route, lifecycle, and allocation-stability gate;
- `phase1/validate_reference.py` — official compressed-tensors representation oracle;
- `phase1/extract_experts.py` — byte-preserving NVFP4 bank extractor with per-record and final-size validation.
- `phase2/ring_protocol.hpp` and `phase2/shared_ring.{hpp,cpp}` — fixed-layout protocol-v2 wire ABI and eight-slot process-shared ring;
- `phase2/shared_ring_tests.cpp` — deterministic fork/process-death/stale-generation/wraparound protocol suite;
- `phase2/memfd_transport_{host.cu,provider.cpp}` — independently mapped CUDA/Level-Zero pinned-memory transport probe;
- `phase2/b70_ring_{provider.cpp,integration_test.cpp}` — isolated real-B70 server and numerical/lifecycle/latency gate.

### Phase 2 process-ring evidence

The production-oriented process boundary passed on the physical RTX 5090 and Arc Pro B70 on 2026-08-04:

- a sealed `memfd` mapping registered by CUDA and independently mapped by the Level Zero provider preserved every tested byte across CUDA D2H, process notification, B70 H2D/D2H, and CUDA H2D;
- the eight-slot protocol-v2 ring passed all-or-nothing device failure, stale-completion rejection, delayed cross-process writes, provider death/generation replacement, full-ring backpressure, safe reclamation, and deterministic wraparound;
- the extended protocol-only wraparound run completed 2,000,000 requests;
- the isolated B70 server loaded the Phase-1 bank once, accepted canonical global IDs plus explicit remote masks, suppressed zero-remote submission, and preserved the Phase-1 allocation count through all successful requests;
- real B70 results matched the saved FP32 partial at `atol=1e-6`, `rtol=1e-2` for `M=1`, duplicate top-8 `M=8`, all eight live slots, wrapped reuse, and the benchmark shapes below;
- stale provider generation, request sequence, and provider PID identities failed closed; shutdown was clean.

The warmed benchmark used eight warmups and 100 measured sequential requests per shape. Each row had one remote route. `publication_to_observation` spans host release-publication through successful host consumption; `provider_total` is the Phase-1 issue/take interval; `ring_process_overhead` is their per-sample difference and includes wakeup, validation/compaction, server scheduling, and completion observation.

| `M` | Publication to observation p50/p95/p99 | Provider total p50/p95/p99 | Kernel p50/p95/p99 | Ring/process overhead p50/p95/p99 |
|---:|---:|---:|---:|---:|
| 1 | 245.640 / 375.199 / 397.498 µs | 102.917 / 124.167 / 125.781 µs | 44.062 / 53.229 / 53.438 µs | 143.946 / 288.481 / 309.321 µs |
| 8 | 292.517 / 686.828 / 2,795.787 µs | 86.145 / 172.604 / 2,245.209 µs | 38.750 / 79.271 / 99.480 µs | 195.519 / 451.286 / 524.429 µs |
| 32 | 581.815 / 713.304 / 1,968.519 µs | 234.791 / 237.604 / 1,470.209 µs | 103.542 / 105.938 / 110.416 µs | 347.634 / 475.752 / 498.647 µs |
| 128 | 1,896.458 / 2,929.813 / 4,290.717 µs | 1,065.209 / 2,050.104 / 3,390.990 µs | 601.979 / 603.333 / 1,518.854 µs | 831.926 / 986.683 / 1,078.048 µs |

The long-tail outliers are retained rather than filtered. This benchmark closes the Phase-2 direct protocol gate; it does **not** establish the 200 tok/s target, CUDA/B70 overlap, production placement, continuous batching, or upstream-vLLM performance.

Historical oneMKL/Python workers remain investigative artifacts and are not the planned provider.

## Phase 3 independent provider-wire mathematics evidence

Phase 3 closes the independent B70 process-wire mathematics gate. It proves that the returned compact `[M_remote, hidden]` payload is the routing-weighted sum of exactly the staged B70-owned routes **before CUDA scatter**.

### Implementation and frozen artifacts

- `phase3/generate_reference.py` — independent little-endian fixture generator and validator. It authenticates the entire 14,495,580,220-byte expert bank at SHA-256 `0ce6377ba3c9848da42b6063574ea884052d2e0f5e605d86d1684a1e5826e8db`, authenticates the canonical NVFP4 shard manifest at SHA-256 `320fad67387d36509947a691fa269d5a55dfb08f0cd7da6434868a6861bff2fa`, and validates the frozen source/NVFP4 snapshot contracts.
- `phase3/reference_fixture.bin` plus `reference_fixture.{hpp,cpp}` — frozen schema/identity/finite/uniqueness-validated fixture at SHA-256 `3ebac16d0f09907cee4718ac1054d21939e420eabaf76ebe79c75fa5d0132606`.
- `phase3/math_cases.{hpp,cpp}` — canonical ownership materialization, independent NVFP4 weighted-partial oracle, per-element accumulation budget, local-route invariance check, and source/NVFP4 artifact metrics.
- `phase3/provider_math_test.cpp` and `phase3/Makefile` — authoritative real process-ring harness, compact provider launch, trusted-bootstrap negatives, split/fused fault injection, and direct unsupported-shape checks.

The generator byte-audits the sampled packed E2M1 weights, raw E4M3FN block-scale bytes, and FP32 global-scale fields in the expert bank against the frozen NVFP4 artifact. It then computes both the BF16-source expert and independently decoded NVFP4 expert in float64 for layers `0` and `31`, canonical experts `0,1,7,63,127,191,254,255`, and eight distinct deterministic FP16 inputs. The source tensor index records 71,903,645,408 bytes; the NVFP4 tensor index records 26,473,821,704 bytes.

### Source-to-NVFP4 quality evidence

The frozen matrix covers 16 layer/expert pairs and all eight inputs. Every metric must be finite with nonzero source and NVFP4 norms.

| Metric | Observed | Frozen boundary |
|---|---:|---:|
| Worst per-expert relative RMSE | `0.1683012879458` | `<= 0.18` |
| Minimum per-expert cosine | `0.985919468279` | `>= 0.98` |
| Aggregate relative RMSE | `0.1579548618065` | `<= 0.18` |
| Aggregate cosine | `0.987528585785` | `>= 0.98` |

Provider output is compared elementwise with the independent NVFP4 weighted oracle using

$$
10^{-6}\sum |w| + \left(10^{-2}+\gamma_{2r}\right)
  \sum |w_e\,\operatorname{Expert}_e(x)|,
$$

where \(r\) is the number of remote routes in the staged row and \(\gamma_{2r}\) is the standard float32 accumulation factor. Non-finite provider output fails immediately.

### Physical-B70 provider-wire matrix

The recorded physical-B70 execution covered:

- zero-remote materialization with no ring-slot publication and zero provider dispatch;
- every `M=1..128` with one remote route per staged row;
- all-remote `M=4` with duplicate and non-sorted IDs, repeated expert `7`, boundary experts `0/255`, and an exactly \(2^{-12}\) routing weight whose contribution is sensitivity-checked;
- mixed `M=10` ownership with four sparse staged rows (`0,2,5,9`), interleaved local/remote routes, and byte-identical remote materialization plus identical provider results when only local routes change;
- canonical-to-compact remapping through resident order `255,0,7,63,127,191,254,1`;
- exact request/completion generation, sequence, nonce, placement/weight SHA-256, row/route extents, route status, token status, dispatch count, and allocation-baseline accounting;
- sequence-bound `after_kernel_before_copyout` failures for both split and fused paths, with execution-failed/core-device status, no exposed payload, no contributed statuses, and unchanged poison output;
- direct rejection of unsupported `M=0` and `M=129` without changing dispatch, allocation, or pending state;
- full expert-bank SHA-256 binding plus trusted placement-generation and weight-generation bootstrap negatives.

`phase3/provider_math_test` passed on the physical Intel Arc Pro B70. Its exact terminal success record was:

```text
Phase-3 provider mathematics PASS
```

### Boundary and non-claims

This evidence proves the isolated QuixiCore-XPU NVFP4 provider's process-ring partial and artifact/source agreement. It does **not** prove:

- upstream-vLLM integration or the Qwen-scoped adapter;
- CUDA scatter into `[M, hidden]`, the CUDA add/join, or summed CUDA+B70 execution;
- per-layer, final-logit, or generated-token parity;
- concurrent request behavior, continuous batching, grouped prefill, throughput, latency targets, or CUDA/B70 overlap;
- Phase 4 or any later gate, end-to-end readiness, or production acceptance.

QuixiCore-XPU remains the selected primary B70 provider source, llm-scaler remains the qualified secondary INT4 fallback, and Colibri remains reference-only evidence.

## Phase 0–10 production status

| Phase | Status | Work still required |
|---|---|---|
| 0 — Freeze baselines and compatibility contracts | **Complete** | Exact runtime/hardware/checkpoint identities, fixed correctness and workload inputs, CUDA/B70 artifact SHA-256 fingerprints, the accepted all-CUDA vLLM graph-mode baseline, provider protocol, and capability manifest are recorded. Eager equivalence is intentionally deferred to the Phase-4 adapter-parity gate. |
| 1 — Isolated QuixiCore-XPU B70 provider | **Complete; direct gate passed** | Full-bank load, explicit B70 selection, fixed buffers, protocol-v1 capability, provider-core load/issue/take/health/shutdown, split/fused policy, `M=1`, `M=2..32`, duplicate top-8, `M=128` prefill, fail-closed validation, and allocation stability passed on the actual B70. The isolated executable supplies startup load and control; the Phase-2 ring supplies process-facing route payload transport. llm-scaler remains a separately qualified secondary fallback. |
| 2 — Batched pinned-memory protocol | **Complete; direct ring gate passed** | Protocol-v2 fixed-layout ABI, eight disjoint slots, canonical route/row payloads, release/acquire state transitions, bounded admission, cancellation/deadline quarantine, stale generation/PID rejection, process restart semantics, protocol stress, real-B70 numerical execution, allocation stability, and p50/p95/p99 stage measurements passed. |
| 3 — Independent provider mathematics | **Complete; physical-B70 provider-wire gate passed** | The authenticated BF16-source/NVFP4 fixture, byte-audited artifact records, weighted compact partial, `M=1..128` and semantic route matrix, compact remapping, identity/status/allocation accounting, failure poisoning, bootstrap negatives, and unsupported boundaries passed before CUDA scatter. This does not complete the CUDA join or upstream-vLLM adapter. |
| 4 — Upstream-vLLM adapter | **Complete; all-CUDA gate passed (2026-08-05)** | Qwen-scoped out-of-tree `HybridMoERunner`, `HybridRoutedExperts`, and `ShootingBrakeExpertProviderClient` installed via `phase4/`. `adapter_parity.py` confirms identical token output and text vs stock at temperature 0; the routed-expert trace is excluded from the gate (router nondeterminism: stock differs from itself across processes). `adapter_smoke.py` passes. The `issue()`/`take()` provider boundary raises until Phase 6, so all routes stay CUDA-local. Next: load compact expert ownership (Phase 5). |
| 5 — Compact expert ownership | **Complete; placement gate passed (2026-08-05)** | Versioned manifest (`shooting_brake_vllm.placement`) maps every `(layer, expert)` to exactly one owner for all 10,240 experts. Layers 0–31 (NVFP4) are B70-capable; layers 32–39 (FP8) are CUDA-forced. Slots validated dense/gap-free; B70-owned experts cross-checked against the real bank header (32×256). Manifest carries a `generation` id and round-trips to JSON (swappable for a future predictive/speculative offloader). `phase5/placement_test.py` passes all invariants + negatives. Adapter holds the manifest; execution stays all-CUDA until Phase 6. B70 is not a fake EP rank. Next: eager hybrid execution (Phase 6). |
| 6 — Eager hybrid execution | **Complete; hybrid gate passed (2026-08-05)** | 6a: runtime route partition + invariants validated every step. 6b: shadow split-merge `Y_cuda + Y_b70 ≈ Y_full` confirmed (max_abs=0.0015, cosine=0.99999). 6c: real hybrid under `SHOOTING_BRAKE_HYBRID=1` — B70-route CUDA weights zeroed, B70 partial computed separately and added; token output identical to all-CUDA (`[271,760,7308,1238,220,19,16,369]`), hybrid path exercised (layer 31, 27 B70 routes). B70 partial currently via CUDA kernel (correctness-first); Phase 7+ uses actual B70 device. |
| 7 — Continuous-batch decode and grouped prefill | **Complete; real-B70 hybrid gate passed (2026-08-05)** | In-process ctypes binding (`phase7/b70_capi.cpp` + `b70_binding.py`) wires the Intel Arc Pro B70 into the live vLLM forward pass. SYCL+CUDA coexist in one process. Under `B70_DEVICE=1`, B70-owned routes compute on the real B70 via QuixiCore NVFP4 (BF16→FP16 activation, global→compact slot translation, issue/take, FP32→BF16 result). `phase7/hybrid_b70_test.py` passes with exact token parity. M ≤ 128 per dispatch. Remaining production hardening: CUDA/B70 overlap (Phase 8), failure/restart (Phase 9). |
| 8 — Piecewise CUDA graphs and overlap | **8a + 8.5 complete; 8b complete with one open risk** | **8a (async overlap):** issue/take split, B70 kernel overlaps CUDA compute. Three-way parity verified. **8.5 (VRAM surgery):** B70-owned expert weights removed from CUDA VRAM. Surgery now runs off `process_weights_after_loading`, not the first forward — vLLM sizes the KV cache from the *peak* memory of its profiling pass, so freeing later leaves the capacity unusable (KV cache 1.89 GiB before the fix, 8.64 GiB after). **8b (Tier 3, graph-compatible B70 dispatch):** the entire B70 dispatch is now native CUDA stream operations captured by a normal CUDA graph — pinned D2H copies, `cuStreamWriteValue32` to signal, `cuStreamWaitValue32` to join, H2D result copy. No Python, ctypes, `.item()`, or device sync on the decode path. The host watcher is a native thread in `libsb_b70_provider.so` (`sb_b70_poll_*`): a Python poll loop costs ~55 µs per wakeup on this host, more than the B70 kernel, and a spinning one starves the engine thread of the GIL. Breakable CUDA graphs (the Phase-8 plan's step 2) were implemented and abandoned — they disable torch.compile by design (vLLM PR #50750, RFC #42770) and produce garbage under Qwen3.6's GDN attention in **stock vLLM with no adapter**, so the incompatibility is upstream. **Prefill pass-through, resolved:** batches above `SHOOTING_BRAKE_B70_MAX_BATCH` bypass Tier 3 into an all-CUDA pass-through (observed at M=245/260/564/1104/1487/2048), which by inspection should drop every B70 route. An A/B on an identical 245-token prompt under `subset:16:8` — where 97.3% of routes are B70-owned — sending the same prefill through Tier 3 instead produced byte-identical output, so the two paths agree and this is not a correctness defect. The mechanism is still unexplained. |
| 9 — Failure/restart operations | **Partial** | Tier 3 fault containment only: the CUDA-side `cuStreamWaitValue32` cannot time out, so a dead poller wedges the device and the process stops responding to SIGKILL. Failed dispatches therefore still raise the completion flag, are counted, and raise on the next eager forward. Still required: heartbeat, bounded timeouts/queues, provider restart and generation bump, exact batched failed-route recovery, rollback, structured telemetry. |
| 10 — Controlled production benchmark | **First controlled run recorded (2026-08-05)** | `phase10/benchmark.py` + `compare.py`: single-stream and batched sweeps, per-token ITL/TTFT streamed through `AsyncLLM`, prefill throughput, device-side route shares, poller counters, KV capacity, and a token-agreement matrix. Results below. Not yet run: CPU-cold-expert offload baseline, reduced-CUDA control, native comparator, cancellation/restart behaviour. |

No row is marked complete merely because an analogous behavior works in Colibri.

### Phase 10 first controlled result (2026-08-05)

`unsloth/Qwen3.6-35B-A3B-NVFP4`, RTX 5090 + Arc Pro B70, `gpu_memory_utilization=0.90`,
`max_model_len=8192`, `max_num_seqs=64`, temperature 0. Baseline is the same adapter
with an all-CUDA placement, so the comparison isolates the hybrid path rather than the
presence of the plugin.

| | all-CUDA | `split:128` | `subset:16:8` |
|---|---|---|---|
| single-stream | 247.9 tok/s | 157.5 (63%) | **186.3 (75%)** |
| ITL p50 | 3.99 ms | 6.30 | 5.34 |
| ITL p99 | 4.82 ms | — | 6.12 |
| prefill (1487 tok) | 23,295 tok/s | 24,029 | 15,298 |
| KV cache | 211,696 tok | 905,472 | 888,704 |
| B70 route share | 0% | 50.9% | 97.3% |
| poller service mean | — | 91.1 µs | 296.0 µs |
| dispatch errors | — | 0 | 0 |

Capacity is reported as KV cache tokens, not free VRAM: vLLM allocates up to
`gpu_memory_utilization` either way, so free VRAM is near-identical between
configurations and hides the result entirely. **KV capacity is 4.2x.**

Getting there took three fixes, each worth more than any estimate:

1. **Dispatch was submission-bound, not compute-bound.** The QuixiCore kernel measures
   46.5 µs at M=1, but a dispatch cost 243.5 µs: two empty `single_task` kernels
   bracketed the real one purely to produce `kernel_us`, the queue had profiling
   enabled, and `take()` waited on the kernel before submitting the D2H copy instead of
   pipelining it. The queue is in-order, so `issue()` now enqueues everything and
   `take()` waits once — 243.5 µs -> 91.1 µs, 133 -> 158 tok/s.
2. **A B70-active layer costs a fixed dispatch per token.** `split:128` spreads 4096
   offloaded experts over all 32 capable layers and pays that cost 32 times;
   `subset:16:8` puts 3968 experts — the same capacity — in 16 layers. 158 -> 186 tok/s
   for 1.8% less KV cache.
3. **Surgery timing.** See the Phase 8.5 row.

Token agreement is not exact and should not be expected to be: the B70's NVFP4 kernel
differs from CUDA's in the last bits, and greedy decoding turns any near-tie into a
permanent fork. Under `split:128`, 6/8 prompts matched exactly and the two forks
diverged at tokens 25 and 30; under `subset:16:8`, where 97.3% of routes are on B70,
3/8 matched with mean agreement of 18.2 tokens. Every hybrid continuation was coherent
and correct — the sharpest example is `aujourd'hui` vs `aujourd’hui`, a one-token
apostrophe difference. `compare.py` reports the divergence index alongside exact-match
count, because exact match is the wrong bar for a different kernel.

Batched throughput degrades with concurrency (70% of baseline at 4, 44% at 16), because
a single physical B70 with one SYCL queue serializes every layer's dispatch while the
5090 scales. Single-stream decode and prefill are the regimes where the hybrid holds up.

## Next concrete deliverables

Work proceeds in the Phase 0–10 order from [`../plan.md`](../plan.md). Phases 0
through 8 are complete; Phase 10 has a first controlled result. In priority order:

1. **Close the remaining latency gap.** At `subset:16:8`, ITL is 5.34 ms against
   3.99 ms all-CUDA: 1.35 ms over 16 active layers, ~84 µs exposed per layer
   against 296 µs of poller service. The kernel is 46.5 µs, so most of the
   service time is still dispatch overhead — three H2D copies per dispatch could
   be one if the staging buffers were contiguous.
2. **Broaden the workload matrix.** Measured so far: single-stream, a concurrency
   sweep at one prompt length, one 1487-token prefill, and an 8-prompt greedy
   agreement matrix. Not measured: SLO-style service metrics (throughput at a
   fixed TTFT/ITL target), context-length sweeps, multi-turn conversations with
   prefix reuse, and tool-calling / structured-output workloads. The hybrid's
   capacity win is a KV-cache win, so long-context and many-turn workloads are
   exactly where it should look best and are currently untested.
3. **Sweep the placement curve.** `subset:<K>:<N>` trades capacity against
   active-layer count; only `split:128` and `subset:16:8` have been measured.
4. **Explain the prefill pass-through.** Not a defect — an A/B shows the
   pass-through and Tier 3 produce byte-identical output — but the code path
   reads as though it should drop every B70 route, and that gap between
   inspection and measurement is itself a risk.
5. **Phase 9 proper:** heartbeat, bounded timeouts, provider restart and
   generation bump, exact batched failed-route recovery, rollback, telemetry.
6. **Phase 10 remaining arms:** CPU-cold-expert offload baseline, reduced-CUDA
   control, native comparator, cancellation/restart behaviour.

Only after Phase 4 and the subsequent ownership/hybrid gates pass may CUDA scatter/join, layer/logit/generation parity, continuous batching/prefill, piecewise graphs, operational recovery, or the Phase 10 production benchmark be described as active or complete.
