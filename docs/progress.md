# Shooting Brake Progress

Last updated: 2026-08-04

## Project goal

Build and validate heterogeneous Qwen3.6-35B-A3B MoE inference using:

- NVIDIA GeForce RTX 5090 as the state owner and hot-expert GPU.
- Intel Arc Pro B70 as a resident overflow-expert worker.
- CUDA for the RTX 5090 path.
- Native Level Zero/SYCL ESIMD INT4 kernels for the B70 path.
- Pinned host memory only for the small activation, route metadata, and weighted-partial transfers between vendors.
- CPU execution only as an exact emergency fallback, not as part of the normal decode path.

Current model:

- Qwen3.6-35B-A3B Colibri INT4 GS64 checkpoint.
- Hidden size 2,048.
- 40 layers.
- 256 routed experts per layer.
- Top-k 8.
- 10,240 total layer-expert weight sets.

Checkpoint:

```text
/home/shooting-brake007/.cache/huggingface/hub/models--Kreuzzelg--qwen36-35b-a3b-colibri-i4-gs64/snapshots/c619aa594ad1e70af82168fb6b4878427896e21c
```

## Current working architecture

The current Colibri runtime now has a complete heterogeneous state-owner path:

```text
CPU orchestration
    |
    v
RTX 5090 CUDA state owner
    |- embedding
    |- dense projections
    |- DeltaNet recurrent/convolution state
    |- full-attention KV cache
    |- router and canonical top-k
    |- shared expert
    |- hot routed experts
    |- final norm, LM head, and greedy selection
    |
    +---- async route request ----> Intel B70 Level Zero/SYCL ESIMD
                                  |- resident overflow experts
                                  `- one weighted partial per token
    |
    <---- pinned weighted partial -+
    |
    `- residual update and next layer
```

Normal heterogeneous decode no longer runs attention, DeltaNet, router, shared expert, LM head, or dense/state operations through the old CPU implementation. The CPU path remains available as a whole-step correctness oracle and emergency fallback.

Routed expert ownership is disjoint:

- A routed expert is owned by CUDA, B70, direct CPU fallback, or recovery CPU fallback for a given request.
- CUDA and B70 work are issued before collection so their expert computation can overlap.
- Failed CUDA or B70 routes are returned as bitmasks and recomputed exactly on CPU.
- No routed expert is silently dropped.
- B70-owned experts are excluded from the CUDA LFRU tier.

## Work completed

### 1. Runtime and repository evaluation

- Evaluated ExLlamaV3, Colibri, vLLM, vLLM XPU kernels, llm-scaler, Lucebox, LLM.xpu, and related Intel kernel repositories.
- Selected Colibri as the initial integration runtime because it already had the exact Qwen3.6 model structure, routing telemetry, and a compact C execution path.
- Used llm-scaler’s existing Qwen-style BMG ESIMD kernels rather than inventing a new INT4 kernel format.
- Used Lucebox and vLLM as architecture and future optimization references, especially for persistent state ownership, fused Qwen DeltaNet/attention paths, compact expert placement, and production scheduling.

### 2. Intel B70 environment and transport validation

- Created and validated `.venv-xpu` with PyTorch XPU support.
- Detected `Intel(R) Arc(TM) Pro B70 Graphics`, approximately 34.2 GB.
- Verified oneAPI compiler and Level Zero access.
- Built a CUDA plus Level Zero transport benchmark using pinned host memory.
- Confirmed that cross-vendor transport is not the primary bottleneck.

Measured transport:

| Transfer | Result |
|---|---:|
| Full 4 KB activation + 8 KB partial round trip | 43.5 us |
| Pinned host to B70, 4 KB | 9.2 us |
| B70 to pinned host, 8 KB | 8.4 us |
| B70 host-to-device, 1 MiB | 353.9 us / 2,963 MB/s |

### 3. Early B70 compute prototypes

- Verified B70 BF16 GEMM operation and model-like expert shapes through PyTorch XPU.
- Implemented initial FP32 and BF16 oneMKL/SYCL expert workers.
- Integrated the early worker into Colibri and proved correct end-to-end heterogeneous generation.
- Rejected the oneMKL path as the final backend because repeated GEMM/kernel launches dominated latency.

Early compute measurements:

| Measurement | Result |
|---|---:|
| B70 BF16 4096x8192x4096 GEMM | 1.91 ms / 144.2 TFLOPS |
| PyTorch XPU model-shape expert pipeline | 45.3 us/expert |
| PyTorch estimate, eight experts per layer | 362.7 us/layer |
| PyTorch compute-only estimate, 40 layers | 14.51 ms/token |
| oneMKL FP32 heterogeneous end-to-end | approximately 2.73 tok/s |
| oneMKL BF16 heterogeneous end-to-end | approximately 2.56 tok/s |

### 4. llm-scaler ESIMD kernel bring-up

- Built `intel-xpu/llm-scaler/vllm/custom-esimd-kernels-vllm` with the Intel oneAPI compiler.
- Verified the generated `moe_int4_ops` extension and its Qwen INT4 MoE symbols.
- Ran the repository’s router and small full-MoE tests on the B70.
- Determined the exact weight layout used by the optimized decode kernels.
- Confirmed that Colibri GS64 weights could not simply be passed to an unmodified GS128 kernel without changing quantization semantics.
- Kept the original integer values and FP16-converted scales by adapting the native worker to GS64 rather than requantizing the model to a different group size.

Important format details:

- Colibri input: signed S4, group size 64, FP32 scales.
- B70 resident representation: offset-binary INT4, IPEX K-major/marlin nibble shuffle, FP16 scales, group size preserved at 64.
- Gate and up projections are concatenated into the worker’s expected `2 * intermediate` layout.
- Down projection is converted separately.
- Conversion is performed once during expert upload; no weight conversion or transfer occurs during steady-state decode.

Standalone ESIMD measurements during development:

| Measurement | Result |
|---|---:|
| Initial llm-scaler full MoE, eight experts plus shared path | approximately 75.1 us |
| Initial per-expert equivalent | approximately 9.4 us |
| Native Colibri converter/worker deterministic one-route test | 39.0 us |
| Native Colibri converter/worker deterministic eight-route test | 46.0 us total |
| Eight-route output cosine versus CPU reference | 1.00000012 |
| Eight-route maximum absolute error | 0.000002 |

### 5. Native B70 worker integration

- Replaced the temporary FP32/oneMKL worker with a torch-free native SYCL/ESIMD shared library.
- Added exact Colibri signed-S4 to worker K-major conversion.
- Added persistent B70 expert storage.
- Added compact `[layer, expert] -> B70 slot` ownership mapping.
- Fixed an early cross-layer corruption bug where the same expert ID in different layers reused one storage slot.
- Added separate issue and take entry points so host-to-B70 transfer and expert execution start in `qt_issue`, while `qt_take` waits for and accumulates the returned partial.
- Used pinned/USM host buffers for asynchronous transfer.
- Added SYCL asynchronous exception handling with `wait_and_throw()`.
- Added failure propagation so a failed B70 request returns route bits for CPU recomputation.
- Added B70 route, capacity, claimed-slot, and issue-to-take telemetry.

The first correct synchronous ESIMD end-to-end run reached approximately 13.13 tok/s. Splitting issue and take raised that run to approximately 14.46 tok/s, with roughly 0.40 ms/token spent waiting in `qt_take`.

### 6. Heat-aware disjoint expert placement

- Replaced the original fixed per-layer prefix ownership with a compact, global, heat-ordered placement plan.
- CUDA reserves the hottest experts first.
- B70 claims the next experts from the same deterministic order.
- B70 capacity is now independent from the CUDA memory-budget safety ceiling.
- Prevented B70-owned experts from entering the CUDA LFRU cache.
- Added separate successful CUDA, successful B70, direct CPU, and recovery CPU route counters.
- Added explicit B70 capacity and actual claimed-slot reporting.

An earlier controlled placement used:

- 7,264 CUDA-owned experts.
- 2,976 claimed B70 experts out of a 3,040-slot bank.
- Zero normal or recovery CPU expert routes.

The latest 45% residency experiment uses:

- 5,640 CUDA-owned experts, 55.1%.
- 4,600 B70-owned experts, 44.9%.
- Zero CPU-owned experts in the normal path.
- All 10,240 routed-expert weight sets resident across the two GPUs.

### 7. Persistent RTX 5090 CUDA state-owner path

- Added a dedicated Qwen3.6 CUDA state object.
- Moved model activation and recurrent state ownership to the RTX 5090.
- Added persistent device-side dense tensor representations and scratch allocations.
- Added CUDA embedding and zero/copy primitives.
- Added Qwen `(1 + weight)` RMSNorm.
- Added gated grouped-query attention, q/k normalization, partial RoPE, causal attention, persistent KV cache updates, and output projection.
- Added DeltaNet causal convolution and recurrent update paths.
- Added router logits, canonical top-k selection, route normalization, shared expert, CUDA resident experts, final normalization, LM head, and device-side greedy selection.
- Added peer/device expert issue and take operations used by the tier dispatcher.
- Added device-state snapshot, rollback, request disablement, state export/import, reset, and KV growth support.
- Added transactional fallback: if the CUDA state-owner step fails, the request restores the prior state and completes through the CPU oracle.
- Added route ownership masks so CUDA, B70, direct CPU, and recovery CPU contributions are accumulated exactly once.
- Added a CUDA failure-injection test path.

Correctness observations:

- A deterministic 16-token CPU-versus-CUDA trace produced the same selected token sequence.
- CUDA versus CPU vocabulary-logit cosine was approximately 0.997340.
- Maximum absolute vocabulary-logit difference was approximately 0.158876.
- Greedy selected token IDs agreed.
- Eight-route B70 versus CPU cosine was 1.00000012 with effectively zero mean absolute error.
- Forced CUDA whole-step failure successfully restored state and continued through the CPU oracle.

### 8. Benchmarking and telemetry improvements

- Added `IGNORE_EOS=1` fixed-length benchmark mode.
- Added B70 ownership, capacity, route-share, and dispatch statistics.
- Added explicit direct-CPU and recovery-CPU counters.
- Corrected CUDA hit counters so only successfully issued CUDA routes are counted.
- Suppressed Vulkan capability probing when CUDA is explicitly enabled.
- Preserved Vulkan only as an unselected non-CUDA fallback backend.
- Rebuilt both the CLI and server in CUDA and CPU configurations.
- Ran CUDA backend regression tests.
- Added build-configuration stamping so switching between `CUDA=0` and `CUDA=1` forces the correct relink instead of leaving a stale binary.

## Files changed

### Core Colibri runtime

#### `colibri-variants/colibri-qwen36/c/backend_cuda.cu`

- Extended the CUDA backend beyond the original resident-expert helpers.
- Added persistent Qwen device-state support and device memory primitives.
- Added embedding, RMSNorm, attention, DeltaNet, routing, shared/routed expert, LM-head, greedy-selection, copy, synchronization, and rollback-related operations.
- Added failure-injection handling used to verify whole-step transactional fallback.
- Removed an unsafe/no-op scale refresh path for tensors that do not carry scale storage.

#### `colibri-variants/colibri-qwen36/c/backend_cuda.h`

- Added the public CUDA state-owner and Qwen kernel API used by `qwen36_cuda.c` and the tier dispatcher.
- Added expert-group issue/take and peer-transfer declarations.

#### `colibri-variants/colibri-qwen36/c/qwen36_cuda.c`

- New persistent Qwen3.6 CUDA state-owner implementation.
- Owns dense tensors, activation buffers, DeltaNet recurrent/convolution state, full-attention KV cache, routing buffers, shared-expert buffers, logits, and greedy output.
- Implements state snapshot/rollback, reset, state export/import, and request lifecycle management.
- Coordinates CUDA-local and B70 overflow experts in one layer transaction.

#### `colibri-variants/colibri-qwen36/c/qwen36_cuda.h`

- New API between the model loop and the persistent CUDA state owner.
- Declares initialization, availability, step, state-management, request, and lifecycle operations.

#### `colibri-variants/colibri-qwen36/c/qwen36_tier.c`

- Reworked routed-expert planning and accounting.
- Added compact heat-ordered CUDA/B70 ownership.
- Added B70 claim calls after CUDA placement.
- Prevented B70 experts from being queued into CUDA LFRU.
- Added asynchronous B70 issue/take integration.
- Added exact failure masks and direct/recovery CPU recomputation.
- Added separate CUDA, B70, direct CPU, recovery CPU, and skip counters.
- Added ownership and route-share reporting.
- Removed the former implicit behavior where CUDA could consume every expert before B70 ownership was assigned.

#### `colibri-variants/colibri-qwen36/c/qwen36_tier.h`

- Updated routed-expert issue API to accept route weights needed by the asynchronous B70 worker.
- Documents route-bit ownership and recomputation behavior.

#### `colibri-variants/colibri-qwen36/c/b70_tier.c`

- Added native B70 shared-library loading and symbol resolution.
- Added compact slot ownership and `[layer, expert]` mapping.
- Added exact signed-S4/GS64 upload conversion.
- Added separate B70 issue/take calls.
- Added ready-state tracking so ownership is not considered usable before weight upload succeeds.
- Added initialization/upload/dispatch failure handling.
- Added capacity, claimed-slot, and dispatch statistics.

#### `colibri-variants/colibri-qwen36/c/b70_tier.h`

- Added the converter interface.
- Added compact ownership claim and capacity APIs.
- Added asynchronous record/collect contract and failure-mask documentation.

#### `colibri-variants/colibri-qwen36/c/b70_moe_sycl.cpp`

- Replaced the old oneMKL/FP32 worker with native BMG-targeted SYCL/ESIMD INT4 kernels derived from the existing llm-scaler implementation.
- Preserves GS64 scales.
- Stores converted gate/up/down expert weights persistently on B70.
- Uses pinned/USM host buffers and separate issue/take entry points.
- Fuses routed gate/up, SiLU, down projection, route weighting, and accumulation.
- Adds asynchronous exception propagation and profiling.

#### `colibri-variants/colibri-qwen36/c/qwen36.c`

- Integrated the persistent CUDA state-owner decode path while retaining the CPU step as oracle/fallback.
- Added route partitioning and CUDA/B70 issue/take lifecycle.
- Added request reset and state rollback behavior.
- Added `IGNORE_EOS` fixed-token trace mode.
- Skips Vulkan probing when `COLI_CUDA=1`.
- Added or updated state-owner diagnostics and timing output.

#### `colibri-variants/colibri-qwen36/c/qwen36_serve.c`

- Updated server initialization ordering so the persistent CUDA state owner is initialized correctly.
- Keeps the server build aligned with the CLI CUDA/CPU build configuration.

#### `colibri-variants/colibri-qwen36/c/Makefile`

- Added `qwen36_cuda.c` to CUDA-enabled Qwen targets.
- Linked the B70/CUDA state-owner sources into CLI and server targets.
- Added build-configuration stamping so CUDA/CPU mode changes trigger relinking.
- Preserved separate CUDA and CPU builds.

### llm-scaler package adjustment

#### `intel-xpu/llm-scaler/vllm/custom-esimd-kernels-vllm/python/custom_esimd_kernels_vllm/__init__.py`

- Simplified package initialization sufficiently to load the locally built `moe_int4_ops` extension without the unrelated missing general extension causing a circular import failure.

### Experiments and validation tools

#### `experiments/transport_test.c`

- Initial CUDA plus Level Zero pinned-memory transport benchmark.

#### `experiments/transport_debug.c`
#### `experiments/transport_debug2.c`

- Device-selection and discrete-B70 transport diagnostics used to correct an early run that selected the wrong Intel device.

#### `experiments/b70_compute_sanity.py`

- PyTorch XPU/BF16 compute and model-shape sanity benchmark.

#### `experiments/b70_expert_sycl.cpp`

- Early standalone oneMKL/SYCL expert prototype.

#### `experiments/b70_profile.cpp`

- Low-level SYCL launch, transfer, and temporary fused-kernel profiler.

#### `experiments/b70_worker.py`
#### `experiments/b70_esimd_worker.py`

- Early process-worker and ESIMD extension experiments. These are investigative artifacts, not the current production path.

#### `experiments/dual_gpu_test.py`

- Early dual-GPU orchestration experiment.

#### `experiments/test_b70_int4_path.py`

- Deterministic converter and native B70 worker validation.
- Checks exact converted values/layout, one-route output, non-sorted eight-route IDs, unequal route weights, all three expert projections, and final weighted accumulation.

#### `experiments/traces/qwen_45pct_*.trace.log`

- Four complete fixed-length heterogeneous benchmark traces.

#### `experiments/traces/qwen_45pct_*.prompt.txt`

- Exact STEM, science, general-knowledge, and creative-writing prompts used by the 45% residency suite.

## Benchmark history

These measurements come from different implementation stages and are not all direct apples-to-apples comparisons. In particular, earlier Colibri runs kept most non-expert work on CPU, later state-owner runs moved the full decode path to CUDA, and prompt/output lengths differ where noted.

### Original Colibri CUDA expert-tier baselines

Before the persistent CUDA state-owner port:

| Configuration | Measured throughput | Notes |
|---|---:|---|
| RTX 5090, full expert VRAM | 18.35 tok/s | 100% expert-tier VRAM hit |
| RTX 5090, 12 GB expert budget | 15.39 tok/s | approximately 71% expert hit |
| RTX 5090, 8 GB expert budget | 13.73 tok/s | approximately 47.8% expert hit |

These runs still spent substantial time in the CPU implementation for DeltaNet, attention/KV, router/shared expert, and LM head.

### First working heterogeneous Colibri runs

| Configuration | Measured throughput | Interpretation |
|---|---:|---|
| Temporary B70 FP32/oneMKL path | approximately 2.73 tok/s | Correct output, excessive launch overhead |
| Temporary B70 BF16/oneMKL path | approximately 2.56 tok/s | Dtype change did not solve launch overhead |
| Native ESIMD, synchronous collect | approximately 13.13 tok/s | Correct native worker path |
| Native ESIMD, asynchronous issue/take | approximately 14.46 tok/s | B70 work overlaps CUDA/host work |

### Controlled 256-token persistent-state-owner comparison

A later state-owner comparison used the same model generation setup across the three configurations:

| Configuration | Step latency | Throughput | CPU expert fallback | Peak host RSS |
|---|---:|---:|---:|---:|
| All routed experts on RTX 5090 | 37.6 ms/token | 25.57 tok/s | 0 | 22.51 GB |
| RTX 5090 12 GB + B70 ESIMD | 38.2 ms/token | 25.13 tok/s | 0 | 22.47 GB |
| RTX 5090 12 GB + CPU overflow | 109.0 ms/token | 8.65 tok/s | nonzero | 31.06 GB |

The B70 configuration retained approximately 98.3% of the all-5090 routed-expert throughput and was approximately 2.9 times faster than the CPU-overflow configuration in this controlled run.

### Qwen science three-way benchmark

This later science workload allowed normal EOS, so the configurations generated approximately 398-406 tokens rather than a forced 512:

| Configuration | Generated | E2E throughput | Step latency | Steady decode | TTFT | Route share CUDA/B70/CPU |
|---|---:|---:|---:|---:|---:|---:|
| Full RTX 5090 | 406 | 21.50 tok/s | 44.9 ms | 22.27 tok/s | 0.71 s | 100/0/0% |
| RTX 5090 + B70 ESIMD | 400 | 21.35 tok/s | 45.1 ms | 22.17 tok/s | 0.72 s | 93.9/6.1/0% |
| RTX 5090 + CPU overflow | 398 | 13.80 tok/s | 70.7 ms | 14.14 tok/s | 0.75 s | 94.2/0/5.8% |

That test proved low overhead at a 6.1% B70 route share, but it did not exercise the desired 40-50% B70 resident bank.

### Latest 45% B70 residency suite

Final ownership:

| Device | Resident experts | Share | Expert memory |
|---|---:|---:|---:|
| RTX 5090 CUDA | 5,640 | 55.1% | 11.59 GB |
| Intel B70 ESIMD | 4,600 | 44.9% | 7.16 GiB |
| CPU normal path | 0 | 0% | fallback only |

All four measured runs forced exactly 512 generated tokens with `IGNORE_EOS=1`.

| Workload | Prompt tokens | Generated | E2E tok/s | Step ms | Steady tok/s | TTFT | CUDA routes | B70 routes | B70 issue-to-take |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| STEM | 78 | 512 | 17.66 | 53.3 | 18.76 | 1.74 s | 61.0% | 39.0% | 60.6 us |
| Science | 95 | 512 | 17.00 | 54.7 | 18.28 | 2.16 s | 84.0% | 16.0% | 53.1 us |
| General knowledge | 77 | 512 | 17.69 | 53.3 | 18.76 | 1.72 s | 76.2% | 23.8% | 53.5 us |
| Creative writing | 70 | 512 | 17.99 | 52.7 | 18.98 | 1.54 s | 61.0% | 39.0% | 59.0 us |

Aggregate 45% suite:

| Metric | Result |
|---|---:|
| Output tokens | 2,048 |
| Mean E2E throughput | 17.585 tok/s |
| Median E2E throughput | 17.675 tok/s |
| E2E range | 17.00-17.99 tok/s |
| Mean steady decode | 18.695 tok/s |
| Mean decode step | 53.5 ms/token |
| Mean TTFT | 1.79 s |
| Weighted B70 route share | 29.31% |
| B70 route-share range | 16.0-39.0% |
| Total CUDA routed selections | 534,749 |
| Total B70 routed selections | 221,731 |
| Total B70 issue/take calls | 77,100 |
| Weighted B70 issue-to-take | 56.81 us |
| Normal CPU expert routes | 0 |
| Recovery CPU expert routes | 0 |
| LFRU swaps | 0 |
| Peak host RSS | 22.34-22.35 GB |

The requested 44.9% resident ownership was met exactly. Route traffic was lower than resident share because the hottest 55.1% of expert slots were deliberately assigned to CUDA. The CUDA bank received 70.69% of observed routes; the colder B70 bank received 29.31%.

Compared with the earlier full-5090 expert allocation of 19.17 GB, the latest layout uses 11.59 GB for RTX expert storage, freeing approximately 7.58 GB of RTX expert VRAM.

## Durable benchmark artifacts

Frozen heat seed used for the latest suite:

```text
/tmp/shooting_brake_heat_aggregate_20260803.bin
SHA-256: 41072f1f0b9875dfb61880cad680f630c6d08cd1a0a0f2abd977e0f7c63b2283
```

Captured traces:

| Artifact | SHA-256 |
|---|---|
| `experiments/traces/qwen_45pct_stem.trace.log` | `93e98feca7f5b405a2f5095aadf7882b289d4707e78ffca52b69ce015b876c75` |
| `experiments/traces/qwen_45pct_science.trace.log` | `792f8d22b4a3c1c6e363a4fa369016872f2608e9b60ae9c93c32259b6408b825` |
| `experiments/traces/qwen_45pct_general.trace.log` | `d5265f2394f3a18c7a81f95fbe493600ab32133fffd4e096901fa676eaa74f4e` |
| `experiments/traces/qwen_45pct_creative.trace.log` | `a58e4448bac1bc9a46eccf974ff779e50c2cf64964468b3b6ded755dee856def` |

Exact prompts:

- `experiments/traces/qwen_45pct_stem.prompt.txt`
- `experiments/traces/qwen_45pct_science.prompt.txt`
- `experiments/traces/qwen_45pct_general.prompt.txt`
- `experiments/traces/qwen_45pct_creative.prompt.txt`

## Build and verification already completed

Successful builds:

```bash
cd colibri-variants/colibri-qwen36/c
make qwen36 CUDA=1 CUDA_ARCH=sm_120
make qwen36_serve CUDA=1 CUDA_ARCH=sm_120
make qwen36 CUDA=0
make qwen36_serve CUDA=0
make cuda-test CUDA_ARCH=sm_120
```

Observed verification:

- CUDA CLI build succeeded.
- CUDA server build succeeded.
- CPU CLI build succeeded.
- CPU server build succeeded.
- CUDA backend regression test succeeded on RTX 5090.
- Native B70 converter and one/eight-route numerical validation succeeded.
- Full heterogeneous generation completed with coherent output.
- All 10,240 routed-expert weight sets were resident across CUDA and B70 in the latest run.
- Zero normal CPU expert fallback in the latest heterogeneous suite.
- Zero recovery CPU expert fallback in the latest heterogeneous suite.
- Forced CUDA failure rolled back device state and successfully continued through the CPU oracle.
- Vulkan capability probing did not appear in any of the four latest CUDA+B70 traces.

## Known limitations and next work

### 1. Performance remains below the optimized vLLM/NVFP4 reference

The current persistent Colibri state-owner path reaches approximately 25 tok/s in the controlled short state-owner benchmark and 17-19 tok/s in the longer forced-512-token suite. A separate optimized vLLM NVFP4 benchmark on the RTX 5090 reported approximately 268 tok/s at short context.

These are different runtime and quantization paths. The remaining gap is not primarily the B70 expert worker. The current Colibri CUDA path still uses less optimized dense/state kernels, synchronization, and launch structure than vLLM/CUTLASS NVFP4.

Likely high-impact remaining work:

1. Use the vLLM/CUTLASS NVFP4 model path as the state-owner runtime.
2. Keep the validated B70 ESIMD worker as an overflow expert provider.
3. Fuse and optimize DeltaNet projection, convolution, recurrence, normalization, and output work.
4. Use stable-shape CUDA graphs after state and metadata paths are fully device-resident.
5. Replace remaining scalar or batch-one dequantizing GEMM paths with tensor-core kernels.
6. Keep routing, placement, submission, and B70 join off the per-token CPU critical path.
7. Replay identical selected expert IDs and token sequences when comparing full-5090, B70-overflow, and CPU-overflow configurations.

### 2. Latest 45% test is resident-share, not uniform traffic-share

The latest experiment places 44.9% of expert weight sets on the B70, but only 29.31% of observed routes went to it. The hottest experts remain on CUDA by design.

If a future experiment requires 40-50% of route traffic on B70, placement must partition by cumulative heat rather than by expert count. That change may reduce throughput because it deliberately moves hotter experts away from the state owner.

### 3. Fixed-length trace quality caveat

`IGNORE_EOS=1` is appropriate for equal-length performance traces but intentionally ignores natural stopping. The raw benchmark prompts also caused the model to expose planning text instead of consistently producing polished final answers. Performance traces should not be treated as a formal quality evaluation. A future quality comparison should use the checkpoint’s exact chat template and normal EOS handling.

### 4. Historical experimental files are not all production code

The following are investigative and should not be treated as the current backend:

- `experiments/b70_worker.py`
- `experiments/b70_esimd_worker.py`
- `experiments/b70_expert_sycl.cpp`
- early oneMKL worker variants

The current B70 path is the native shared library in `colibri-variants/colibri-qwen36/c/b70_moe_sycl.cpp`, loaded through `b70_tier.c`.

## Handoff summary

The routed-expert fabric is working end to end:

- RTX 5090 owns the complete Qwen state and main CUDA decode path.
- Intel B70 owns and executes a persistent compact bank of overflow routed experts.
- B70 uses native Level Zero/SYCL ESIMD INT4 kernels with the model’s original GS64 quantization semantics.
- CUDA and B70 expert work overlap through issue/take APIs.
- CPU expert fallback is absent during successful heterogeneous runs.
- Device failures return exact recomputation masks rather than losing expert contributions.
- The 45% B70 residency target has been demonstrated across four 512-token workloads.

The main next milestone is no longer proving cross-vendor MoE execution. It is integrating the proven B70 expert-provider boundary with a substantially faster vLLM/CUTLASS/NVFP4 state-owner path, then measuring the same fixed workload under all-5090, B70-overflow, and CPU-overflow configurations.
