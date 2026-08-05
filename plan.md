# Shooting Brake: Upstream vLLM CUDA + Intel B70 Hybrid MoE Plan

**Verdict: proceed. Keep upstream vLLM as the RTX 5090 state-owner runtime, run QuixiCore-XPU NVFP4 MoE kernels inside a separate persistent B70 XPU provider as the primary compute path, with llm-scaler INT4 as a qualified secondary fallback, and maintain only a narrow Shooting Brake MoE adapter and versioned transport contract.**

This plan reflects the inspected workspace as of 2026-08-04:

- The local upstream vLLM checkout in `vllm/` is `v0.26.1rc0-285-g1c0d20791`.
- The latest inspected `intel-xpu/llm-scaler/` vLLM release is `intel/llm-scaler-vllm:0.21.0-b1`; it remains a qualified secondary INT4 fallback.
- The llm-scaler vLLM patch is tied to vLLM 0.21.0 and must not be applied unchanged to vLLM 0.26+.
- `QuixiCore-XPU/` is the selected primary B70 provider kernel source: a MIT-licensed native SYCL library with a framework-neutral raw-pointer C++ ABI, purpose-built NVFP4 routed-MoE kernels that accept preselected routes, and measured 46.5 µs at M=1 decode on the B70 (correctness gate passed). llm-scaler's ESIMD MoE kernels remain a qualified secondary INT4 fallback and kernel-design reference. Both can run behind a stable Shooting Brake provider API using oneAPI and SYCL, with optional PyTorch XPU and `intel-xpu/vllm-xpu/vllm-xpu-kernels/` bindings.
- The already-proven implementation in `colibri-variants/colibri-qwen36/` remains the transport, correctness, placement, and failure-semantics reference—not the production model host.

**Current milestone status:** Phase 0 through Phase 7 are complete. Phase 0 evidence is recorded in [`phase0/SUMMARY.md`](phase0/SUMMARY.md), [`phase0/freeze.yaml`](phase0/freeze.yaml), and [`phase0/capability_manifest.yaml`](phase0/capability_manifest.yaml). Phase 1 evidence is recorded in [`docs/progress.md`](docs/progress.md#phase-1-persistent-provider-evidence) and implemented by `phase1/b70_provider.{hpp,cpp}`, `phase1/b70_provider_main.cpp`, `phase1/b70_provider_tests.cpp`, and `phase1/validate_reference.py`. Phase 2 evidence is recorded in [`docs/progress.md`](docs/progress.md#phase-2-process-ring-evidence) and implemented by the protocol/ring, transport probe, isolated B70 ring provider, and integration tests under `phase2/`. Phase 3 evidence is generated and authenticated by `phase3/generate_reference.py`, frozen in `phase3/reference_fixture.bin`, and exercised on the physical B70 by `phase3/provider_math_test`. Phase 4 is implemented by the Qwen-scoped out-of-tree adapter under `phase4/` (`HybridMoERunner`, `HybridRoutedExperts`, `ShootingBrakeExpertProviderClient`) and qualified by `phase4/adapter_parity.py` and `phase4/adapter_smoke.py`. Phase 5—the versioned compact-expert ownership manifest—is implemented by `shooting_brake_vllm.placement` (in the `phase4/` plugin) and qualified by `phase5/placement_test.py`.

```text
upstream vLLM 0.26+ CUDA model runner on RTX 5090
    ├── scheduler, continuous batching, serving API
    ├── attention, KV cache, DeltaNet/GDN recurrent state
    ├── router and canonical top-k
    ├── hot routed experts and shared expert
    ├── residual, LM head, sampling
    │
    └── Shooting Brake HybridMoERunner / HybridRoutedExperts
            ├── local routes -> stock compatible vLLM CUDA MoE backend
            └── remote routes -> versioned pinned-memory request ring
                                  -> persistent native C++ B70 provider
                                  -> qualified QuixiCore-XPU tiny/batched/prefill NVFP4 MoE kernels (primary); llm-scaler INT4 fallback (secondary)
                                  -> one weighted [M_remote, hidden] wire partial
                                  -> asynchronous CUDA copy and addition
```

The CPU performs orchestration, placement, queue management, telemetry, and exact emergency recovery only. It does not execute normal-path matrix multiplications.

---

## What is already proven

The current provider in `colibri-variants/colibri-qwen36/c/b70_moe_sycl.cpp` proves the essential cross-vendor lifecycle:

```c
b70_moe_init(...)
b70_moe_upload(...)
b70_moe_issue(...)
b70_moe_take(...)
b70_moe_shutdown()
```

It already demonstrates:

- Persistent B70 expert weights and compact `(layer, expert) -> slot` ownership.
- Exact Colibri signed-S4 GS64 conversion into the current B70 kernel layout.
- FP16 activation staging, selected expert IDs, and routing weights.
- NVFP4 fused gate/up/SiLU/down execution (primary); ESIMD INT4 (Colibri reference / llm-scaler secondary).
- Routing-weighted accumulation into one hidden-size partial.
- Asynchronous issue/take separation.
- Exact failed-route recomputation rather than silent contribution loss.
- Numerical agreement with the CPU reference.
- Correct end-to-end CUDA+B70 text generation with zero normal-path CPU expert fallback.
- Controlled heterogeneous throughput close to the corresponding all-CUDA Colibri expert configuration.

The current logical transaction is still single-token:

```text
activation [2048] FP32 host input
expert IDs [routes] int32
routing weights [routes] FP32
    ↓
B70 converts activation/weights to FP16
executes selected resident experts
    ↓
weighted routed partial [2048] FP32 host output
```

The production, model-runtime-neutral transaction is batched:

```text
activation       [M_remote, hidden] FP16/BF16
expert IDs       [M_remote, topk]   int32
routing weights  [M_remote, topk]   FP32 or FP16
remote route mask and token_row_map [M_remote] -> original rows [M]
layer, request, sequence, placement-generation metadata
    ↓
remote partial   [M_remote, hidden] FP16 or FP32
per-route and staged-token completion and recovery metadata
```

The current native worker is a valid correctness and latency baseline, but it is not the production endpoint: it has one in-order queue, one pending operation, one fixed scratch set, no continuous-batch aggregation, and no large-prefill grouped path.

The production-first decision is therefore:

1. Preserve the proven transport, residency, placement, issue/take, and failure contracts.
2. Run an isolated persistent B70 provider process, using QuixiCore-XPU's framework-neutral C++ ABI as the primary compute path (no PyTorch required for steady-state dispatch).
3. Reuse QuixiCore-XPU's NVFP4 MoE fused/split decode and grouped prefill kernels as the primary B70 compute; keep llm-scaler's ESIMD INT4 kernels as a qualified secondary fallback for shapes or quality regimes where NVFP4 is insufficient.
4. Extract or fork kernels only if profiling proves material wrapper overhead that cannot be addressed within the existing ABIs.

We do not need to rediscover cross-vendor communication, independently re-derive the provider wire mathematics, or rewrite the Intel kernels. The batched provider boundary and its B70 wire partial are now implemented and directly qualified; the remaining engineering starts by inserting that boundary into vLLM's modular routed-expert lifecycle in Phase 4. QuixiCore-XPU was built, correctness-validated, and benchmarked on the B70 in this workspace (2026-08-04): its `nvfp4_moe` split kernel measured 46.5 µs at M=1 and passed all direct kernel correctness gates.

---

## Why vLLM is the preferred CUDA runtime

The current vLLM checkout already supports the exact Qwen-family execution structure:

- `Qwen3_5DecoderLayer`
- `Qwen3NextSparseMoeBlock`
- `QwenGatedDeltaNetAttention`
- full attention and linear-attention layer types
- CUDA-owned KV/recurrent state
- replicated router
- top-8 expert selection
- shared expert
- modular routed-expert execution

The actual Qwen MoE call flow is:

```text
CUDA hidden states [M, 2048]
    -> Qwen3NextSparseMoeBlock.forward
    -> MoERunner
    -> CUDA router gate
    -> FusedTopKRouter.select_experts
    -> topk_ids [M, 8]
    -> topk_weights [M, 8]
    -> RoutedExperts.forward_modular
    -> CUDA expert implementation
    -> weighted/reduced [M, 2048]
    -> shared expert and residual continuation
```

Relevant source seams:

- `vllm/vllm/model_executor/models/qwen3_5.py`
- `vllm/vllm/model_executor/models/qwen3_next.py`
- `vllm/vllm/model_executor/layers/fused_moe/layer.py`
- `vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py`
- `vllm/vllm/model_executor/layers/fused_moe/routed_experts.py`
- `vllm/vllm/model_executor/layers/fused_moe/modular_kernel.py`
- `vllm/vllm/model_executor/layers/fused_moe/expert_map_manager.py`

Most importantly, `FusedMoEFactory` already accepts injected:

```python
runner_cls
runner_args
routed_experts_cls
routed_experts_args
```

and vLLM supports out-of-tree pluggable layers through:

```python
PluggableLayer.register_oot(...)
CustomOp.register_oot(...)
```

This means we can install a custom Qwen-only hybrid routed-expert runner without replacing vLLM's scheduler, attention, GDN state, KV cache, LM head, or serving stack.

### Recommended vLLM seam

Use an out-of-tree replacement for:

```text
MoERunner / RoutedExperts
```

The custom runner should preserve vLLM's existing router and shared-expert behavior. It changes only the post-top-k routed-expert execution.

The first implementation should be injected explicitly into a Qwen sparse-block variant through:

```python
FusedMoEFactory(
    runner_cls=HybridMoERunner,
    routed_experts_cls=HybridRoutedExperts,
    ...
)
```

This is safer than globally replacing every vLLM MoE layer. Only the tested Qwen3.5/3.6 configuration should select it.


### Upstream compatibility and ownership boundary

Shooting Brake must track two independent compatibility surfaces:

| Surface | Current inspected baseline | Upgrade policy |
|---|---|---|
| CUDA state owner | upstream vLLM `v0.26.1rc0-285-g1c0d20791` | Rebase regularly; keep modifications out-of-tree and Qwen-scoped |
| B70 kernel provider (primary) | `QuixiCore-XPU/` native SYCL NVFP4 MoE kernel library (MIT) | Pin a validated build; framework-neutral C++ ABI, no PyTorch required for steady-state dispatch |
| B70 kernel provider (secondary) | `intel-xpu/llm-scaler/` vLLM `0.21.0-b1` INT4 ESIMD kernel stack | Qualified secondary INT4 fallback; pin a validated worker image/environment; port bindings independently |
| Cross-vendor protocol | Shooting Brake-owned | Versioned and stable across both runtime upgrades |
| Model/weight contract | Shooting Brake manifest | Explicitly records architecture, tensor shapes, quantization, ownership, and kernel capability |

The `0.21.0 -> 0.26+` difference does not require running an old vLLM CUDA server. It only means:

- llm-scaler's complete `vllm_for_multi_arc.patch` cannot be copied from `intel-xpu/llm-scaler/` into the newer checkout;
- B70 Python wrappers may need adaptation when PyTorch XPU or `intel-xpu/vllm-xpu/vllm-xpu-kernels/` APIs change;
- the native QuixiCore-XPU NVFP4 MoE and llm-scaler ESIMD INT4 operator signatures, supported shapes, weight layout, and numerical behavior must each be validated per provider release;
- changes in vLLM's `MoERunner`, modular-kernel, router, graph, or expert-map APIs are absorbed by the narrow Shooting Brake CUDA adapter.

New upstream model support is inherited only when the model exposes a compatible routed-MoE seam. Every model still needs an explicit Shooting Brake capability manifest and correctness qualification; unsupported models remain on stock vLLM CUDA rather than entering an unvalidated B70 path.

---

## Exact per-layer execution

For a batch of M scheduled tokens:

### 1. CUDA routing

The RTX 5090 computes:

```text
x                 [M, 2048]
router_logits     [M, 256]
topk_ids          [M, 8]
topk_weights      [M, 8]
```

The inspected vLLM CUDA router normally produces:

- `topk_ids`: signed int32
- `topk_weights`: float32
- both on the CUDA device

The B70 must consume these canonical selected IDs and weights. It must not recompute router logits, softmax, or top-k.

### 2. Partition routes

Use one persistent ownership table per layer:

```text
global expert ID -> CUDA local slot
global expert ID -> B70 compact slot
```

For every selected route:

```text
CUDA-resident expert -> local CUDA mask
B70-resident expert  -> remote B70 mask
neither available    -> recovery mask/error
```

The two masks must partition the original selected routes without changing their original routing weights.

### 3. Start B70 asynchronously

Stage only rows containing remote routes:

```text
activation rows
remote expert IDs
remote routing weights
token/route mapping
layer ID
placement-generation ID
request sequence
```

Use a fixed pinned-memory ring. Do not allocate, call `.item()`, or synchronize the entire CUDA device on the decode path.

### 4. Run local CUDA experts concurrently

While B70 is working, use vLLM's normal CUDA expert backend for local routes.

Remote route positions must be represented as invalid/skipped routes. vLLM's expert-map convention uses:

```text
global_id -> local_id
-1 for non-local
```

That is a useful ownership model, but it is not guaranteed that every NVFP4/FlashInfer/TRT-LLM MoE kernel accepts -1 route IDs. The selected production CUDA backend must be tested explicitly.

If its existing kernel cannot skip arbitrary remote routes, the hybrid runner must compact local routes before calling it and unpermute/reduce afterward.

### 5. Return one B70 partial per staged token row

The B70 returns the compact wire response:

```text
remote_partial[M_remote, 2048]
```

already:

- multiplied by the original routing weights;
- summed over all B70 routes belonging to each staged token row;
- paired with the original `token_row_map` and per-route status.

After completion validation, CUDA scatters this response into a preallocated, zero-initialized `[M, 2048]` buffer. Unstaged tokens therefore have zero B70 contribution. That full-batch CUDA buffer matches vLLM's routed-result shape and keeps wire transport independent of both the full scheduler batch and the number of remote experts.

### 6. Join on CUDA

Validate and copy `remote_partial`, scatter it through `token_row_map` into the preallocated full-batch CUDA buffer, and compute:

$$
Y_{\text{routed}} = Y_{\text{CUDA local}} + Y_{\text{B70 remote}}
$$

Then allow the stock MoERunner logic to combine the CUDA shared expert and continue into the residual path.

The B70 partial must be added before any final tensor/expert-parallel reduction.

---

## Reuse ExLlamaV3's connector mechanics, not its MoE runtime

ExLlamaV3 contains a valuable CUDA-side transport design:

- `exllamav3-quant-inference/exllamav3/exllamav3/model/moe_cpu_host.py`
- `exllamav3-quant-inference/exllamav3/exllamav3_ext/cpu/moe_handoff.h`
- `exllamav3-quant-inference/exllamav3/exllamav3_ext/cpu/moe_handoff.cu`

Its existing ring uses:

```text
[x fp16]
[selected expert IDs int32]
[routing weights fp16]
[output fp32]
```

and performs stream-ordered coordination using:

```text
cuStreamWriteValue32
cuStreamWaitValue32
```

That is almost exactly the transport contract needed.

The strongest reuse opportunity is:

1. Port the pinned shared-memory ring and sequence protocol.
2. Port the CUDA stream flag publication/wait mechanics.
3. Replace the CPU expert worker with the B70 Level Zero/SYCL worker.
4. Keep vLLM as the model runtime.

This is better than porting ExLlama's whole `BlockSparseMLP`.

### Why ExLlamaV3 should not be the primary host

ExLlamaV3's current CPU MoE offload is all-or-nothing per MoE layer:

```text
cpu_offload=False -> CUDA experts
cpu_offload=True  -> CPU offload branch only
```

It does not currently run arbitrary CUDA-resident and remote-resident experts from the same layer and then combine them.

Other observed limitations:

- No general ExpertProvider interface.
- CUDA expert ownership is based on contiguous ranges.
- The current B70 placement uses arbitrary compact hot/cold mappings.
- No explicit Qwen3.6 architecture registration in this checkout.
- EXL3 weights are not compatible with B70 GS64 signed-S4.
- Its fast CUDA MoE graph assumes CUDA-addressable EXL3 pointer tables.
- External B70 completion cannot simply be inserted inside that fused CUDA graph.
- Production serving and continuous batching are weaker than vLLM's scheduler path.

ExLlama remains viable for a focused prototype because its transport ring is excellent. It is not the lowest-risk production host.

---

## B70 compute: QuixiCore-XPU NVFP4 primary, llm-scaler INT4 secondary

QuixiCore-XPU is the selected primary B70 kernel source: MIT-licensed, framework-neutral raw-pointer C++ ABI, purpose-built NVFP4 routed-MoE kernels that accept preselected expert IDs and routing weights. Its `nvfp4_moe_fused` and `nvfp4_moe_split` functions were built and benchmarked on the B70 in this workspace (2026-08-04): M=1 split measured 46.5 µs, all ops passed the correctness gate. The llm-scaler ESIMD INT4 stack remains a qualified secondary fallback for shapes or quality regimes where NVFP4 is insufficient. Neither is a CUDA-to-XPU transport plugin.

Observed constraints:

- Operations are registered under PyTorch's XPU dispatch key.
- Kernels acquire the current XPU/SYCL stream from XPU tensors.
- `XpuFusedMoe` assumes its input, routing tensors, expert weights, and outputs are XPU tensors.
- Existing expert maps express XPU-local ownership; they do not transport non-local CUDA routes.
- XCCL and NCCL remain separate homogeneous collective backends.
- No mixed NCCL/XCCL all-to-all path provides the required request transaction.
- Available helpers cover CPU↔XPU or XPU↔XPU, not direct CUDA↔XPU tensor dispatch.
- Several full-fused entry points own router/top-k/shared-expert work, which conflicts with the RTX 5090 state-owner contract.

Therefore this does not work:

```text
CUDA vLLM tensor -> directly call XpuFusedMoe
```

The correct reuse is:

```text
upstream CUDA vLLM
    -> Shooting Brake post-top-k adapter
    -> pinned-host request ring
    -> persistent B70 process with preallocated device tensors
    -> QuixiCore-XPU NVFP4 MoE preselected-route kernels (primary); llm-scaler ESIMD INT4 (secondary fallback)
    -> pinned-host weighted partial
    -> CUDA join
```

The XPU tensor requirement is local to the B70 process. CUDA tensors never become XPU tensors directly: the adapter copies only selected activation rows and route metadata through pinned memory, and the provider copies those buffers into already-allocated XPU tensors.

---

## Required provider architecture

### Recommended process model

Use one persistent B70 provider process per B70 device:

```text
upstream vLLM CUDA worker process
    ↕ fixed pinned shared-memory ring
Shooting Brake B70 provider process
    └── QuixiCore-XPU NVFP4 MoE operators (primary, C++/SYCL ABI); llm-scaler INT4 fallback (secondary, PyTorch XPU optional)
```

This is safer than loading Torch CUDA, PyTorch XPU, oneAPI, Level Zero, and both dispatchers into the same vLLM worker. It isolates:

- CUDA runtime and CUDA graph state.
- oneAPI and Level Zero runtime state.
- PyTorch CUDA and XPU dispatchers.
- SYCL queue ownership.
- provider crashes, hangs, and device loss.
- independent vLLM and B70-kernel dependency versions.

A complete second XPU vLLM instance is unnecessary. The provider owns only expert weights, fixed buffers, XPU streams, kernel selection, and request execution; it does not own attention, KV/recurrent state, routing, shared experts, sampling, or serving.

Start with QuixiCore-XPU's C++/SYCL ABI as the primary provider interface because it requires no PyTorch in the steady-state dispatch path and has been benchmarked on the B70. The llm-scaler PyTorch/XPU bindings remain available as a secondary fallback for INT4 W4A16 if NVFP4 quality is insufficient. Preserve the Colibri native worker as a correctness and latency comparator.

Do not extract or fork kernels preemptively. Both QuixiCore-XPU and llm-scaler kernels are upstream reusable; the provider selects between them by qualified capability.

### Ring descriptor

A practical request descriptor needs at least the same normative fields as the fabric contract:

```c
struct B70Request {
    uint32_t protocol_version;
    uint32_t descriptor_size;

    uint64_t request_seq;
    uint64_t provider_generation;
    uint64_t placement_generation;
    uint64_t weight_generation;
    uint64_t placement_fingerprint;
    uint64_t route_subset_fingerprint;
    uint64_t deadline_ns;

    uint32_t ring_slot;
    uint32_t layer;
    uint32_t num_batch_tokens;   /* full scheduler-step M */
    uint32_t num_staged_tokens;  /* compact M_remote */
    uint32_t num_routes;
    uint32_t hidden_size;
    uint32_t topk;

    uint32_t activation_dtype;
    uint32_t weight_dtype;
    uint32_t output_dtype;
    uint32_t request_buffer_version;
    uint32_t output_buffer_version;
    uint32_t status;
    uint32_t error;
    uint32_t reserved;
};
```

Associated fixed buffers:

```text
activation          [capacity_staged_tokens, hidden]
expert_ids          [capacity_staged_tokens, topk]
routing_weights     [capacity_staged_tokens, topk]
remote_route_mask   [capacity_staged_tokens, topk]
token_row_map       [capacity_staged_tokens] -> [num_batch_tokens]
route_position_map  [capacity_routes]
output              [capacity_staged_tokens, hidden]
route_status        [capacity_routes]
```

Completion must publish only after the SYCL output copy is visible to the CUDA-side process. It echoes all identity, generation, shape, dtype, fingerprint, buffer-version, and status fields needed to bind the result to the immutable request.

The sequence, slot, provider/placement/weight generations, fingerprints, and buffer versions jointly prevent stale replies, ownership, routes, or expert slots from being paired with a newer model request.

### Failure semantics

Never silently omit a remote expert.

On B70 timeout, device loss, invalid generation, or kernel error:

- mark the exact failed token-route entries;
- either recompute those routes on an available CUDA/CPU correctness path;
- or fail the request explicitly.

The existing Colibri recovery-mask behavior should be retained conceptually, but vLLM will need a batched route-status representation rather than one 32-bit single-token mask.

---

## Batching and kernel reuse are mandatory

The current native B70 worker is a one-token, one-pending-operation design:

```cpp
S.activation       = [hidden]
S.expert_ids       = [topk]
S.routing_weights  = [topk]
S.output           = [hidden]
bool S.pending
```

That proves decode correctness but cannot support vLLM continuous batching. The provider must allocate grow-only, stable-address buffers:

```text
activation       [max_tokens, hidden]
expert_ids       [max_tokens, topk]
routing_weights  [max_tokens, topk]
route/token map  [max_routes]
output           [max_tokens, hidden]
pending slots    [ring_capacity]
```

There must be one aggregated B70 dispatch per active MoE layer and scheduler step, not one submission per request. For 32 concurrent single-token requests:

```text
required: one layer operation with M=32
avoid:    32 independent M=1 Level Zero submissions
```

### Decode and prefill dispatch policy

The primary QuixiCore-XPU implementation provides fused and split NVFP4 MoE kernels that accept preselected expert IDs and routing weights. The secondary llm-scaler implementation contains equivalent INT4 families:

- QuixiCore-XPU `nvfp4_moe_fused`: one work-group per (token, expert) pair, fused w13 → SwiGLU → w2 → atomic weighted sum; preferred for high-occupancy shapes.
- QuixiCore-XPU `nvfp4_moe_split`: two-kernel higher-occupancy form for low-M decode (M=1–8); measured 46.5 µs at M=1 on B70.
- llm-scaler tiny/small-batch N-major INT4 kernels for decode-sized `M` (secondary fallback).
- llm-scaler prefill gather/grouped-up/grouped-down/weighted-accumulation phases (secondary fallback).
- Grow-only reusable buffers intended to keep addresses stable in both paths.

Use the following logical dispatch table, qualified by benchmark rather than hard-coded permanently:

| Workload | Initial kernel policy |
|---|---|
| `M=1` decode | QuixiCore-XPU `nvfp4_moe_split` NVFP4 preselected-route path (primary); llm-scaler tiny INT4 (secondary) |
| `1<M<=32` decode/continuous batch | QuixiCore-XPU `nvfp4_moe_fused` or `nvfp4_moe_split` NVFP4 by shape threshold (primary); llm-scaler tiny/small-batch INT4 (secondary) |
| larger decode batch | grouped NVFP4 MoE path (primary); grouped INT4 route path (secondary) |
| prefill | QuixiCore-XPU NVFP4 MoE grouped path (primary); llm-scaler `moe_prefill_*_v2` gather/up/down/accumulate (secondary) |

Do not call a full-fused entry point that recomputes router logits/top-k or executes the shared expert. The B70 path must:

- accept canonical preselected IDs and routing weights from CUDA;
- remap global IDs through the provider's compact B70 ownership table;
- process only B70-owned token-route pairs;
- return one already-weighted and reduced partial per original token;
- preserve the selected NVFP4 (primary) or GS64/GS128 INT4 (secondary) weight contract;
- allocate no per-forward weight or scratch tensors;
- support zero-remote-route batches without submitting B70 work.

Kernel policy belongs in the provider capability table, not in vLLM model code.

---

## Weight-format decision

The B70 and RTX 5090 do not need the same physical quantization format, but both copies must represent the same source model.

The clean deployment artifact is:

```text
Original BF16/FP16 model
    ├── CUDA hot-expert representation
    │      e.g. NVFP4/FP8/Marlin supported by vLLM
    └── B70 cold-expert representation
           NVFP4 (E2M1 + E4M3 block scale, block 16) — QuixiCore-XPU primary;
           signed-S4 GS64 + FP16 scales — Colibri reference / llm-scaler secondary
```

At runtime:

- RTX 5090 loads only CUDA-owned experts.
- B70 loads only B70-owned experts.
- Dense, router, shared-expert, attention, GDN, and LM-head weights load on CUDA.
- No expert weight is transferred during decode.

### Current format caveat

The production B70 provider format options are:

```text
Primary (QuixiCore-XPU):
  NVFP4: E2M1 weights + E4M3 block scales (block 16) + per-expert FP32 global scale
  w13: [E, 2I, K/2] uint8, w13_scales: [E, 2I, K/16] uint8
  w2:  [E, K, I/2] uint8, w2_scales:  [E, K, I/16] uint8

Secondary (llm-scaler / Colibri):
  signed-S4, group size 64 or 128, FP16 scales, K-major/N-major layout
```

A GS64 tensor cannot be reinterpreted as NVFP4 or GS128 because they have different scale semantics, block sizes, and code mappings. Three valid choices:

1. Quantize B70-owned experts to NVFP4 and use QuixiCore-XPU's production fused/split kernels (primary).
2. Preserve the proven GS64 Colibri worker and generalize its batched kernels.
3. Quantize B70-owned experts to GS128 and use the llm-scaler production N-major batch path (secondary).

Do not convert GS64 to GS128 on the token path. Any conversion/requantization is initialization-time or offline work.

If CUDA uses an NVFP4 or FP8 checkpoint while B70 uses a different format, validate that both came from the identical unquantized model. Otherwise the joined partials could represent subtly different base weights.

---

## Shooting Brake configuration and capability negotiation

Every deployable model/provider combination needs a generated or validated manifest. Configuration must describe facts, not select kernels by undocumented model-name conditionals.

Minimum model manifest:

```yaml
model_architecture: Qwen3_5MoeForCausalLM
hidden_size: 2048
num_layers: 40
num_experts: 256
top_k: 8
routed_intermediate_size: 512
shared_intermediate_size: 512
router_normalization: canonical-vllm
```

Minimum B70 artifact/provider manifest:

```yaml
provider_protocol: 1
provider_backend_primary: quixicore-xpu-sycl
provider_backend_secondary: torch-xpu-llm-scaler
kernel_bundle_primary: quixicore-xpu-nvfp4-moe-qualified
kernel_bundle_secondary: llm-scaler-0.21.0-b1-qualified
activation_dtypes: [fp16, bf16]
output_dtypes: [fp32, fp16]
weight_format_primary: nvfp4
weight_format_secondary: signed-s4
nvfp4_block_size: 16
nvfp4_scale_dtype: e4m3
nvfp4_global_scale_dtype: fp32
group_size_secondary: 64
scale_dtype_secondary: fp16
layout_primary: quixicore-native
layout_secondary: k-major-marlin
max_tokens: <qualified value>
max_routes_per_token: 8
decode_kernels_primary: [nvfp4-moe-fused, nvfp4-moe-split]
decode_kernels_secondary: [tiny-int4, grouped-int4]
prefill_kernels_primary: [nvfp4-moe-grouped]
prefill_kernels_secondary: [gather-v2, up-v2, down-v2, accumulate-v2]
```

The provider handshake must return:

- protocol version;
- B70 device identity and usable memory;
- kernel-bundle version and dependency fingerprint;
- supported dtypes, dimensions, top-k values, group sizes, and layouts;
- maximum tokens, routes, in-flight slots, and streams;
- graph/address-stability guarantees;
- loaded weight generation and placement fingerprint.

Startup must fail closed if the CUDA adapter, model manifest, placement file, weight artifact, or provider capabilities disagree. A model unsupported by B70 continues on stock upstream vLLM CUDA; it is never silently forced through a generic kernel.

This boundary allows vLLM and the B70 kernel environment to advance independently:

```text
vLLM upgrade
    -> adapt/test only HybridMoERunner and model injection

QuixiCore-XPU / llm-scaler / PyTorch-XPU upgrade
    -> adapt/test only provider wrappers, layouts, and kernel capabilities for the selected primary or secondary backend

protocol change
    -> explicit version bump with compatibility handling
```

---

## CUDA graphs

The current host-mediated B70 issue/take cannot execute inside vLLM's FULL CUDA graph mode.

A full graph replay contains CUDA operations only. It cannot directly encode:

- process-ring publication;
- Level Zero submission;
- SYCL completion;
- external host-state change;
- dynamic error/recovery information.

Safe rollout:

1. **Correctness:** eager MoE execution.
2. **Optimization:** vLLM PIECEWISE CUDA graphs with a break around the B70 provider operation.
3. **Advanced:** fixed buffers, stream memory operations, and segmented graph capture around the external dependency.

vLLM's `breakable_cudagraph.py` explicitly supports piecewise graph boundaries, not full-graph external execution.

The ExLlama `cuStreamWriteValue32`/`cuStreamWaitValue32` mechanism is valuable for eliminating Python synchronization, but it does not automatically make Level Zero work part of a CUDA graph.

---

## Performance model and implications

The B70 path is fast enough to be useful, but production vLLM has a much tighter latency window than the CPU-heavy Colibri reference.

Measured B70 one-token issue/take has been roughly:

```text
approximately 56–100 µs per active MoE layer
```

For decode, expert GEMV is primarily weight-bandwidth-bound. The device owning a selected expert supplies that expert's weight bandwidth:

- CUDA-owned routes depend on RTX 5090 VRAM bandwidth and the selected CUDA MoE kernel.
- B70-owned routes depend on B70 VRAM bandwidth and the ESIMD kernel.
- The RTX 5090 does not read B70-resident expert weights.
- The 5090 stages the activation, receives one partial, and performs the final CUDA addition.

For one layer:

$$
T_{\mathrm{MoE}} \approx
\max\left(
T_{\mathrm{CUDA\ local}},
T_{\mathrm{CUDA\to host\to B70}} + T_{\mathrm{B70}} +
T_{\mathrm{B70\to host\to CUDA}}
\right)
+ T_{\mathrm{join}}
$$

For hidden size 2048 and FP16 transport, one activation or returned partial is 4096 bytes per transported token. Expert weights are never moved on the token path.

Prefill uses larger grouped matrix operations. Its result depends on CUDA attention/GDN, local CUDA grouped MoE, remote B70 grouped MoE, route distribution, and activation transport. Per layer:

$$
T_{\mathrm{prefill}} \approx
T_{\mathrm{router,CUDA}} +
\max\left(
T_{\mathrm{local\ grouped\ MoE}},
T_{\mathrm{remote\ transfer}} + T_{\mathrm{remote\ grouped\ MoE}}
\right)
+ T_{\mathrm{join}}
$$

Thus neither “5090 bandwidth” nor “B70 bandwidth” alone determines total throughput. The sequential CUDA state path plus the slower concurrent expert branch determines the critical path.

If all 40 layers exposed 56.8 µs of serialized B70 latency:

$$
40 \times 56.8\ \mu\text{s} = 2.27\text{ ms/token}
$$

The measured fast all-5090 vLLM workload was approximately 3.46 ms inter-token latency. Therefore the Colibri result cannot predict vLLM hybrid speed: batching, overlap, graph boundaries, and synchronization are decisive.

Production throughput depends on:

1. Aggregating scheduled tokens into one B70 operation per active layer.
2. Overlapping B70 execution with local CUDA routed/shared-expert work.
3. Avoiding `.item()`, Python polling, and device-wide synchronization.
4. Reusing fixed pinned and XPU buffers.
5. Keeping the highest-traffic experts on CUDA.
6. Skipping B70 submission when a layer has no remote routes.
7. Preserving piecewise CUDA graphs around the provider boundary.
8. Measuring exposed wait rather than quoting isolated kernel time.

The current placement evidence is directionally correct: B70 should contribute substantial weight capacity while receiving a smaller share of latency-critical routes. The objective is not equal route distribution. It is:

> Move enough cold expert capacity to B70 to satisfy model and KV-cache capacity while retaining the hottest experts on the RTX 5090.

For a model that fits entirely on the 5090, all-CUDA remains the expected throughput ceiling. Shooting Brake's value is enabling larger models, larger KV caches, longer contexts, greater concurrency, and materially better service than CPU expert offload while staying close to the CUDA ceiling.

---

## Implementation order and acceptance gates

### Phase 0 — Freeze baselines and compatibility contracts

**Status: COMPLETE — gate passed on 2026-08-04.** The frozen environment, source/artifact identities and SHA-256 fingerprints, workloads, protocol/capability schemas, and all-CUDA vLLM graph baseline are recorded under [`phase0/`](phase0/); see [`phase0/SUMMARY.md`](phase0/SUMMARY.md).

Record:

- exact upstream vLLM commit;
- llm-scaler, PyTorch XPU, `vllm-xpu-kernels`, oneAPI, Level Zero, and driver versions;
- source checkpoint identity;
- CUDA and B70 weight-artifact fingerprints;
- one fixed correctness prompt set and one fixed performance workload;
- the production-relevant all-CUDA vLLM supported graph-mode baseline. Eager equivalence is intentionally deferred to the Phase-4 all-CUDA adapter-parity gate.

Deliver the versioned provider protocol and model/provider capability schemas before coupling either runtime to the other.

**Gate:** incompatible protocol, shape, dtype, layout, group size, top-k, or weight generation fails at startup with an actionable error.

### Phase 1 — Build the isolated B70 provider

The completed Phase-1 provider implementation:

- selects the B70 explicitly;
- imports the qualified QuixiCore-XPU NVFP4 MoE operator bundle as primary, with llm-scaler INT4 as secondary fallback;
- preallocates grow-only device activation, route, scratch, and output tensors;
- loads each compact B70 expert once;
- accepts preselected IDs and routing weights;
- exposes capability, load, issue, take, health, and shutdown on the native provider core; the isolated executable performs startup load and exposes capability/health/shutdown control, while Phase 2 supplies the process-facing route payload transport;
- selects tiny/small/grouped/prefill kernels from measured shape thresholds (QuixiCore-XPU NVFP4 primary; llm-scaler INT4 secondary);
- never runs router, shared-expert, attention, or sampling work.

Retain `colibri-variants/colibri-qwen36/c/b70_moe_sycl.cpp` as a native comparator. Do not extract or fork QuixiCore-XPU or llm-scaler kernels unless profiling identifies a concrete wrapper overhead.


**Gate:** direct XPU tests pass for `M=1`, `M=2..32`, and representative prefill `M`; no weight upload or tensor allocation occurs during steady-state dispatch.

**Status: COMPLETE — gate passed on the physical B70 on 2026-08-04.** The 14,495,580,220-byte bank (SHA-256 `0ce6377ba3c9848da42b6063574ea884052d2e0f5e605d86d1684a1e5826e8db`) loaded as 8,192 resident NVFP4 experts. Direct tests passed `M=1`, `M=2..32`, duplicate valid top-8 routes, and fused `M=128`; rejected invalid layer/IDs and stale generation/sequence; preserved the allocation count across nine successful dispatches; and passed idempotent shutdown. The official compressed-tensors representation oracle passed with maximum absolute difference \(1.070\times10^{-6}\). See [`docs/progress.md`](docs/progress.md#phase-1-persistent-provider-evidence).

The Phase-1 executable is intentionally a control shell around the tested provider core. It does not serialize activation or route payloads over stdin. Cross-process `issue`/`take` data-plane reachability is the Phase-2 pinned-ring deliverable, not an unimplemented Phase-1 JSON transport.

### Phase 2 — Implement the batched pinned-memory protocol

Extend the proven issue/take transaction with:

- multiple ring slots;
- request and completion sequence numbers;
- layer and placement generation;
- batched token/route descriptors;
- per-token/per-route status;
- timeout and provider-generation handling;
- explicit acquire/release publication ordering;
- fixed maximum capacities negotiated at startup.

For `[M, hidden]`, transfer only rows with at least one B70-owned route. Preserve the original token-row map for deterministic scatter and reduction.

**Gate:** stress the ring through wraparound, concurrent slots, stale replies, invalid generations, provider restart, and injected kernel failure without accepting stale output or losing an expert contribution.

**Status: COMPLETE — direct process-ring gate passed on the physical RTX 5090 and B70 on 2026-08-04.** The clean protocol-v2 ABI uses eight fixed disjoint slots, canonical global expert IDs plus explicit remote masks and route positions, token-row maps, independent provider/placement/weight/ring generations, request/completion sequences, release/acquire state publication, bounded admission, cancellation/deadline quarantine, and provider-death generation retirement. Deterministic protocol tests passed all-or-nothing failure, stale completion, delayed cross-process publication, full-ring backpressure, safe reuse, process death/replacement, and 2,000,000-request wraparound stress. A separately mapped CUDA/Level-Zero `memfd` probe passed byte correctness. The isolated B70 server loaded the Phase-1 bank once and passed zero-remote suppression, `M=1`, duplicate-top-8 `M=8`, eight queued slots, wrapped reuse, stale identity rejection, clean shutdown, exact dispatch/allocation accounting, and warmed `M=1/8/32/128` numerical/latency runs. The warmed publication-to-observation p50 was 245.640, 292.517, 581.815, and 1,896.458 µs respectively; long-tail p99 was retained. See [`docs/progress.md`](docs/progress.md#phase-2-process-ring-evidence). This closes only the direct ring boundary; it does not claim upstream-vLLM integration, continuous batching, CUDA/B70 overlap, or production throughput.

### Phase 3 — Validate provider mathematics independently

Compare:

```text
Y_reference_remote = weighted sum of exactly the B70-owned staged routes
Y_provider_wire    = returned [M_remote, hidden] partial before CUDA scatter
```

Required cases:

- `M=1`, `M=2..32`, and representative prefill `M`;
- all staged routes remote and mixed local/remote semantic subsets;
- duplicate and non-sorted expert IDs;
- multiple tokens selecting the same expert;
- unequal and near-zero routing weights;
- boundary expert IDs and compact-slot remapping;
- invalid placement/weight generation;
- injected device/kernel failure.

A layer with no B70-owned route is not a provider-mathematics case: adapter/ring and layer-replay gates must prove zero provider submission and an additive-identity CUDA remote lane.

Validate both CUDA and B70 expert artifacts against the same higher-precision source checkpoint before validating their summed result.

**Gate:** the frozen BF16-source-to-NVFP4 artifact-quality thresholds and the fixed provider-wire numerical budget pass for every supported shape; unsupported shapes fail explicitly.

**Status: COMPLETE — independent provider-mathematics gate passed on the physical B70 on 2026-08-04.** `phase3/generate_reference.py` authenticates the full expert-bank SHA-256 and the frozen NVFP4 shard manifest, byte-audits sampled bank records against the NVFP4 artifact, computes independent float64 BF16-source and NVFP4 expert outputs for layers 0 and 31 and experts 0, 1, 7, 63, 127, 191, 254, and 255 across eight deterministic FP16 inputs, and freezes and validates `phase3/reference_fixture.bin`. `phase3/provider_math_test` passed zero-remote no-publication; the `M=1..128` one-remote-route sweep; all-remote duplicate, non-sorted, boundary, and \(2^{-12}\)-weight `M=4`; mixed sparse and interleaved ownership with local-route invariance; process-ring identity, status, and allocation accounting; split- and fused-path sequence-bound injected failures; unsupported `M=0` and `M=129`; trusted bank, placement, and weight-bootstrap negatives; and compact resident list `255,0,7,63,127,191,254,1` with canonical-to-local remapping. This proves the B70 wire partial before CUDA scatter and artifact/source agreement. It does not claim upstream-vLLM integration, CUDA scatter/join, layer/logit/generation parity, concurrency or throughput, or production acceptance.

### Phase 4 — Add the upstream-vLLM out-of-tree adapter

**Status: COMPLETE — gate passed on 2026-08-05.** The adapter (`HybridMoERunner`, `HybridRoutedExperts`, `ShootingBrakeExpertProviderClient`) is installed out-of-tree only for the explicitly qualified eager Qwen3.6 config via `phase4/src/shooting_brake_vllm/`. The all-CUDA gate is met: `phase4/adapter_parity.py` confirms identical token output (`[271, 760, 7308, 1238, 220, 19, 16, 369]`) and text between stock vLLM and the adapter at temperature 0. The per-token routed-expert trace is intentionally excluded from the gate because it is the router's own nondeterministic `topk_ids` (confirmed empirically: stock vLLM differs from itself across processes in the trace while producing identical tokens); the Shooting Brake adapter runs strictly downstream of the router and cannot affect it. The adapter's `issue()`/`take()` provider boundary raises until Phase 6, so all routes stay on CUDA.

Implement Qwen-scoped:

```text
HybridMoERunner
HybridRoutedExperts
ShootingBrakeExpertProviderClient
```

Keep stock vLLM:

- scheduler and continuous batching;
- CUDA attention, KV cache, and DeltaNet/GDN state;
- CUDA router and canonical top-k;
- shared expert and residual;
- LM head, sampling, and serving APIs.

Begin in eager mode. Inject the implementation only for an explicitly qualified architecture/configuration. Do not globally replace every vLLM MoE layer.

**Gate:** all-CUDA mode through the adapter matches stock vLLM output and performance within measurement noise before enabling B70 routes.

### Phase 5 — Load compact expert ownership
**Status: COMPLETE — gate passed on 2026-08-05.** The versioned placement manifest (`shooting_brake_vllm.placement`) maps every `(layer, expert)` to exactly one owner (CUDA or B70) for all 40 MoE layers × 256 experts (10,240 total). Layers 0–31 (NVFP4) are B70-capable; layers 32–39 (FP8) are CUDA-forced and rejected if assigned to B70. Device-local slots are validated dense and gap-free; every B70-owned expert is cross-checked against the real Phase-1 bank header (32×256 = 8,192). The manifest carries a `generation` id and round-trips to JSON, so a future predictive/speculative offloader can swap it at coarse boundaries without touching the per-step contract. `phase5/placement_test.py` validates all invariants, the bank cross-reference, and negatives. The adapter (`HybridRoutedExperts`) builds and holds the manifest at init; execution stays all-CUDA (Phase 4 parity preserved, smoke test passes with `split:128`) until Phase 6 partitions routes. The B70 is **not** represented as a fake vLLM EP rank.

Create one immutable per-model placement map:

```text
(layer, global expert) -> CUDA local slot
(layer, global expert) -> B70 compact slot
```

Load only hot CUDA-owned experts and only cold B70-owned experts. Reuse vLLM `ExpertMapManager` concepts where compatible, but do not represent B70 as a fake expert-parallel rank.

**Gate:** every routed expert has exactly one normal-path owner, CUDA+B70 ownership covers the qualified placement, and no expert weights move during inference.

### Phase 6 — Integrate eager hybrid execution
**Status: COMPLETE — gate passed on 2026-08-05 (6a+6b+6c).** Stage 6a (runtime route partition + invariants, execution unchanged) and 6b (shadow split-merge validation: `Y_cuda + Y_b70 ≈ Y_full`, max_abs=0.0015, cosine=0.99999) are complete. Stage 6c (real hybrid execution) is implemented in `HybridRoutedExperts.forward_modular` under `SHOOTING_BRAKE_HYBRID=1`: B70-route CUDA weights are zeroed, the B70 partial is computed separately and added to the CUDA result. `phase6/hybrid_execution_test.py` passes — token output is identical between `all-cuda` and `split:128`+hybrid (`[271, 760, 7308, 1238, 220, 19, 16, 369]`), and the hybrid path was exercised (layer 31, 27 B70 routes). The B70 partial currently uses the CUDA kernel (correctness-first); Phase 7+ replaces it with the actual B70 device via the QuixiCore provider.

Per layer:

```text
CUDA router/top-k
-> partition routes
-> publish B70 request
-> run local CUDA routed experts and shared expert
-> collect B70 weighted partial
-> asynchronous H2D copy
-> CUDA addition
-> residual continuation
```

The B70 partial must join before final tensor/expert-parallel reduction. Start at TP=1; qualify TP/EP separately after the single-rank contract is correct.

**Gate:** deterministic generation, final logits, per-layer routed output, and route ownership match the all-CUDA reference within the selected quantization tolerance.

### Phase 7 — Add continuous-batch decode and grouped prefill
**Status: COMPLETE — real-B70 hybrid gate passed on 2026-08-05.** The B70 NVFP4 MoE provider is wired into the live vLLM forward pass via an in-process ctypes binding (`phase7/b70_capi.cpp` + `shooting_brake_vllm.b70_binding.py`). SYCL and CUDA coexist in the same EngineCore process (validated: separate driver stacks, no conflict). Under `SHOOTING_BRAKE_B70_DEVICE=1`, B70-owned routes are computed on the actual Intel Arc Pro B70 via QuixiCore NVFP4: activation is cast BF16→FP16, global expert IDs are translated to B70 compact slots, dispatched via issue/take, and the FP32 result is added to the CUDA output. `phase7/hybrid_b70_test.py` passes with **exact token parity** (`[271, 760, 7308, 1238, 220, 19, 16, 369]`) — the B70's NVFP4 kernel output is close enough to CUDA's FlashInfer-CUTLASS that greedy decode is identical. Continuous-batch aggregation (variable M, one dispatch per layer) works for the qualified decode/prefill shapes (M ≤ 128).

Constrain each upstream scheduler step to the provider's negotiated token/route capacity, then aggregate all of that step's remote token rows into exactly one B70 layer request. Oversized prefill is chunked by upstream scheduling before layer execution, never split into multiple ring transactions inside one layer step. Tune decode thresholds between tiny and grouped kernels. Add QuixiCore-XPU's NVFP4 MoE grouped prefill path (primary) or llm-scaler's INT4 prefill gather/up/down/accumulate path (secondary) without moving router or shared-expert work to B70.

**Gate:** correctness holds across changing batch sizes, mixed prefill/decode scheduling, request completion, cancellation, and zero-remote-route layers.

### Phase 8 — Restore piecewise CUDA graphs and remove exposed waits

Introduce a break only around the external provider operation. Use dedicated CUDA copy streams, events, pinned buffers, and nonblocking provider completion. Consider ExLlama-style stream memory operations only after the process-ring implementation is correct.

**Gate:** graph mode preserves output agreement and improves or maintains eager throughput; no device-wide synchronization appears on the steady-state path.

### Phase 9 — Failure, restart, and operational qualification

Implement:

- heartbeat and bounded request timeout;
- provider process restart and generation bump;
- exact failed-route CUDA or CPU correctness recovery;
- load/placement rollback;
- structured telemetry without payload logging;
- bounded queues and backpressure.

**Gate:** injected B70 loss never produces silently incomplete output. The request either recomputes exactly or fails explicitly.

### Phase 10 — Controlled production benchmark

Use the identical checkpoint source, prompt matrix, scheduler settings, context lengths, output count, placement profile, and request-order rotations:

| Configuration | Purpose |
|---|---|
| Stock all-CUDA vLLM | Throughput and latency ceiling |
| CUDA hot experts + B70 QuixiCore-XPU provider | Shooting Brake result (primary NVFP4; secondary llm-scaler INT4) |
| CUDA hot experts + CPU cold experts | Offload baseline |
| Reduced CUDA budget without B70 | Capacity/control baseline |
| Native B70 worker, where shape-compatible | Provider-overhead comparator |

Record median and range across at least three interleaved runs:

- request and output throughput;
- per-request ITL and TTFT percentiles;
- prefill throughput;
- CUDA/B70/CPU route shares;
- remote tokens and B70 submissions per layer/step;
- provider queue, copy, kernel, and exposed-wait times;
- CUDA local MoE and shared-expert times;
- CPU recovery count;
- RTX 5090, B70, pinned-host, and total host memory;
- cancellation/restart behavior;
- per-layer, final-logit, and generated-token agreement.

Performance is accepted only against the upstream vLLM baseline, not the older Colibri timing. Report capacity gained together with throughput and latency cost.

---

## Final decision

Use:

```text
upstream vLLM 0.26+ for the RTX 5090 CUDA state owner and serving stack
Shooting Brake out-of-tree HybridMoERunner/HybridRoutedExperts adapter
Shooting Brake versioned pinned-memory provider protocol and placement manifest
an isolated persistent B70 provider process using QuixiCore-XPU's C++/SYCL ABI as primary compute
qualified QuixiCore-XPU NVFP4 MoE fused/split/grouped kernels inside that process (primary)
qualified `intel-xpu/llm-scaler/` tiny/batched/prefill ESIMD INT4 kernels as secondary fallback
the existing native `colibri-variants/colibri-qwen36/` B70 worker as a correctness/performance comparator
`exllamav3-quant-inference/exllamav3/` pinned-ring and CUDA stream-memory design as an optimization reference
```

Do not use:

```text
the complete llm-scaler vLLM 0.21.0 patch as the CUDA host for vLLM 0.26+
B70 as a fake NCCL/XCCL expert-parallel rank
direct CUDA-tensor calls into XPU-dispatch operators
full B70 fused functions that recompute router/top-k/shared-expert work
vLLM parameter offloading as selected-expert execution
ExLlamaV3's all-or-nothing CPU-offload branch unchanged
decode-time expert weight transfer or quantization
silent generic-kernel fallback for an unqualified model
```

**Exact plan:** retain upstream vLLM's rapidly evolving CUDA model support, isolate all Intel dependencies in a version-pinned B70 provider, and keep the coupling surface to a post-top-k routed-expert adapter plus a versioned protocol. Port vLLM adapter APIs and B70 kernel bindings independently as their upstream projects evolve.

This architecture lets Shooting Brake support future upstream models without pretending every new model is automatically B70-compatible. A new model uses the B70 only after its dimensions, routing semantics, weight artifact, supported kernel family, and mixed-result numerics are qualified in the capability manifest. Otherwise it runs unchanged on stock vLLM CUDA.

The success criterion is not merely “both GPUs were used.” It is production-grade continuous-batch decode and prefill with exact failure semantics, no normal-path CPU matrix compute, measurable capacity gain, and throughput/latency reported against the same upstream all-CUDA vLLM workload.
