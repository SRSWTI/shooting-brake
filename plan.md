# Shooting Brake: Upstream vLLM CUDA + Intel B70 Hybrid MoE Plan

**Verdict: proceed. Keep upstream vLLM as the RTX 5090 state-owner runtime, run `intel-xpu/llm-scaler` ESIMD kernels inside a separate persistent B70 XPU provider, and maintain only a narrow Shooting Brake MoE adapter and versioned transport contract.**

This plan reflects the inspected workspace as of 2026-08-04:

- The local upstream vLLM checkout in `vllm/` is `v0.26.1rc0-285-g1c0d20791`.
- The latest inspected `intel-xpu/llm-scaler/` vLLM release is `intel/llm-scaler-vllm:0.21.0-b1`.
- The llm-scaler vLLM patch is tied to vLLM 0.21.0 and must not be applied unchanged to vLLM 0.26+.
- llm-scaler's ESIMD MoE kernels are substantially less coupled to vLLM. They can run behind a stable Shooting Brake provider API using PyTorch XPU, `intel-xpu/vllm-xpu/vllm-xpu-kernels/`, oneAPI, and SYCL.
- The already-proven implementation in `colibri-variants/colibri-qwen36/` remains the transport, correctness, placement, and failure-semantics reference—not the production model host.

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
                                  -> persistent PyTorch-XPU B70 provider
                                  -> intel-xpu/llm-scaler tiny/batched/prefill ESIMD kernels
                                  -> one weighted [M, hidden] partial
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
- ESIMD fused gate/up/SiLU/down execution.
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
activation       [M, hidden] FP16/BF16
expert IDs       [M, topk]   int32
routing weights  [M, topk]   FP32 or FP16
remote route mask and token/route map
layer, request, sequence, placement-generation metadata
    ↓
remote partial   [M, hidden] FP16 or FP32
per-token/per-route completion and recovery metadata
```

The current native worker is a valid correctness and latency baseline, but it is not the production endpoint: it has one in-order queue, one pending operation, one fixed scratch set, no continuous-batch aggregation, and no large-prefill grouped path.

The production-first decision is therefore:

1. Preserve the proven transport, residency, placement, issue/take, and failure contracts.
2. Run an isolated PyTorch-XPU B70 provider initially.
3. Reuse `intel-xpu/llm-scaler/`'s existing decode and prefill kernels inside that provider.
4. Extract those kernels into a Torch-free C ABI only if profiling proves Python/PyTorch-XPU dispatch overhead is material.

We do not need to rediscover cross-vendor communication or rewrite the Intel kernels. The remaining engineering is the batched provider boundary and its insertion into vLLM's modular routed-expert lifecycle.

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
| B70 kernel provider | `intel-xpu/llm-scaler/` vLLM `0.21.0-b1` kernel stack | Pin a validated worker image/environment; port bindings independently |
| Cross-vendor protocol | Shooting Brake-owned | Versioned and stable across both runtime upgrades |
| Model/weight contract | Shooting Brake manifest | Explicitly records architecture, tensor shapes, quantization, ownership, and kernel capability |

The `0.21.0 -> 0.26+` difference does not require running an old vLLM CUDA server. It only means:

- llm-scaler's complete `vllm_for_multi_arc.patch` cannot be copied from `intel-xpu/llm-scaler/` into the newer checkout;
- B70 Python wrappers may need adaptation when PyTorch XPU or `intel-xpu/vllm-xpu/vllm-xpu-kernels/` APIs change;
- the native ESIMD operator signatures, supported shapes, weight layout, and numerical behavior must be validated per provider release;
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

### 5. Return one B70 partial per token

The B70 should return:

```text
remote_partial[M, 2048]
```

already:

- multiplied by the original routing weights;
- summed over all B70 routes belonging to each token;
- zero for tokens without a B70 route.

This matches vLLM's routed-result shape exactly and keeps transport independent of the number of remote experts.

### 6. Join on CUDA

Copy `remote_partial` back to a preallocated CUDA buffer and compute:

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

## Use llm-scaler for B70 compute, not as the mixed-device host runtime

The llm-scaler stack is the preferred B70 kernel source and initial provider implementation. Its patched vLLM runtime is not itself a CUDA-to-XPU transport plugin.

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
    -> persistent B70 process with preallocated XPU tensors
    -> intel-xpu/llm-scaler preselected-route ESIMD kernels
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
    └── PyTorch XPU + intel-xpu/llm-scaler ESIMD operators
```

This is safer than loading Torch CUDA, PyTorch XPU, oneAPI, Level Zero, and both dispatchers into the same vLLM worker. It isolates:

- CUDA runtime and CUDA graph state.
- oneAPI and Level Zero runtime state.
- PyTorch CUDA and XPU dispatchers.
- SYCL queue ownership.
- provider crashes, hangs, and device loss.
- independent vLLM and B70-kernel dependency versions.

A complete second XPU vLLM instance is unnecessary. The provider owns only expert weights, fixed buffers, XPU streams, kernel selection, and request execution; it does not own attention, KV/recurrent state, routing, shared experts, sampling, or serving.

Start with the existing llm-scaler Python/XPU bindings because they already cover the required batching regimes. Preserve the native Torch-free worker as:

- a correctness and latency comparator;
- an emergency narrow-shape provider;
- a possible later deployment target if profiling proves dispatcher overhead significant.

Do not extract or fork kernels preemptively. Upstream kernel reuse is cheaper to maintain than a second private implementation.

### Ring descriptor

A practical request descriptor needs at least:

```c
struct B70Request {
    uint64_t request_seq;
    uint64_t weight_generation;

    uint32_t ring_slot;
    uint32_t layer;
    uint32_t num_tokens;
    uint32_t num_routes;

    uint32_t activation_dtype;
    uint32_t output_dtype;
    uint32_t status;
    uint32_t reserved;
};
```

Associated fixed buffers:

```text
activation       [capacity_tokens, 2048]
expert_ids       [capacity_tokens, 8]
routing_weights  [capacity_tokens, 8]
route_mask       [capacity_tokens]
output           [capacity_tokens, 2048]
```

Completion must publish only after the SYCL output copy is visible to the CUDA-side process.

`request_seq`, `ring_slot`, `layer`, and `weight_generation` prevent stale replies or stale expert slots from being paired with a newer model request.

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

The inspected llm-scaler implementation already contains the required families:

- Tiny/small-batch N-major INT4 kernels for decode-sized `M`; the Python path selects a tiny kernel for `hidden_states.shape[0] <= 32`.
- Lower-level up/down/finalize entry points that accept precomputed route IDs and weights.
- Prefill gather/grouped-up/grouped-down/weighted-accumulation phases.
- Grow-only reusable buffers intended to keep addresses stable.

Use the following logical dispatch table, qualified by benchmark rather than hard-coded permanently:

| Workload | Initial kernel policy |
|---|---|
| `M=1` decode | llm-scaler tiny INT4 preselected-route path |
| `1<M<=32` decode/continuous batch | llm-scaler tiny or small-batch INT4 path, whichever wins per shape |
| larger decode batch | grouped route path |
| prefill | `moe_prefill_gather_forward_v2` → `moe_prefill_up_forward_v2` → activation → `moe_prefill_down_forward_v2` → weighted accumulation |

Do not call a full-fused entry point that recomputes router logits/top-k or executes the shared expert. The B70 path must:

- accept canonical preselected IDs and routing weights from CUDA;
- remap global IDs through the provider's compact B70 ownership table;
- process only B70-owned token-route pairs;
- return one already-weighted and reduced partial per original token;
- preserve the selected GS64 or GS128 weight contract;
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
           signed-S4 + FP16 scales + ESIMD layout
```

At runtime:

- RTX 5090 loads only CUDA-owned experts.
- B70 loads only B70-owned experts.
- Dense, router, shared-expert, attention, GDN, and LM-head weights load on CUDA.
- No expert weight is transferred during decode.

### Current format caveat

The current B70 provider is:

```text
signed-S4
group size 64
FP16 scale storage
K-major/marlin-derived worker layout
```

Some reusable llm-scaler N-major kernels are documented around group size 128. A GS64 tensor cannot be reinterpreted as GS128 because it has a different number of quantization scales.

Two valid choices:

1. Preserve the proven GS64 worker and generalize its batched kernels.
2. Quantize B70-owned experts once to GS128 and use the llm-scaler production N-major batch path.

Do not convert GS64 to GS128 on the token path. Any conversion/requantization is initialization-time or offline work.

If CUDA uses an NVFP4 checkpoint while B70 uses current Colibri INT4 weights, validate that both came from the identical unquantized model. Otherwise the joined partials could represent subtly different base weights.

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
provider_backend: torch-xpu-llm-scaler
kernel_bundle: llm-scaler-0.21.0-b1-qualified
activation_dtypes: [fp16]
output_dtypes: [fp16, fp32]
weight_format: signed-s4
group_size: 64
scale_dtype: fp16
layout: k-major-marlin
max_tokens: <qualified value>
max_routes_per_token: 8
decode_kernels: [tiny-int4, grouped-int4]
prefill_kernels: [gather-v2, up-v2, down-v2, accumulate-v2]
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

llm-scaler/PyTorch-XPU upgrade
    -> adapt/test only provider wrappers, layouts, and kernel capabilities

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

Record:

- exact upstream vLLM commit;
- llm-scaler, PyTorch XPU, `vllm-xpu-kernels`, oneAPI, Level Zero, and driver versions;
- source checkpoint identity;
- CUDA and B70 weight-artifact fingerprints;
- one fixed correctness prompt set and one fixed performance workload;
- all-CUDA vLLM eager and supported graph-mode baselines.

Deliver the versioned provider protocol and model/provider capability schemas before coupling either runtime to the other.

**Gate:** incompatible protocol, shape, dtype, layout, group size, top-k, or weight generation fails at startup with an actionable error.

### Phase 1 — Build the isolated llm-scaler B70 provider

Create one persistent PyTorch-XPU process that:

- selects the B70 explicitly;
- imports only the qualified operator bundle from `intel-xpu/llm-scaler/`;
- preallocates grow-only XPU activation, route, scratch, and output tensors;
- loads each compact B70 expert once;
- accepts preselected IDs and routing weights;
- exposes capability, load, issue, take, health, and shutdown operations;
- selects tiny/small/grouped/prefill kernels from measured shape thresholds;
- never runs router, shared-expert, attention, or sampling work.

Retain `colibri-variants/colibri-qwen36/c/b70_moe_sycl.cpp` as a native comparator. Do not extract llm-scaler kernels unless profiling identifies a concrete wrapper overhead.

**Gate:** direct XPU tests pass for `M=1`, `M=2..32`, and representative prefill `M`; no weight upload or tensor allocation occurs during steady-state dispatch.

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

### Phase 3 — Validate provider mathematics independently

Compare:

```text
Y_reference = weighted sum of exactly the B70-owned routes
Y_provider  = returned [M, hidden] partial
```

Required cases:

- `M=1`, `M=2..32`, and representative prefill `M`;
- all routes remote, mixed local/remote, and no remote routes;
- duplicate and non-sorted expert IDs;
- multiple tokens selecting the same expert;
- unequal and near-zero routing weights;
- boundary expert IDs and compact-slot remapping;
- invalid placement/weight generation;
- injected device/kernel failure.

Validate both CUDA and B70 expert artifacts against the same higher-precision source checkpoint before validating their summed result.

**Gate:** agreed FP16/INT4 tolerance passes for every supported shape; unsupported shapes fail explicitly.

### Phase 4 — Add the upstream-vLLM out-of-tree adapter

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

Create one immutable per-model placement map:

```text
(layer, global expert) -> CUDA local slot
(layer, global expert) -> B70 compact slot
```

Load only hot CUDA-owned experts and only cold B70-owned experts. Reuse vLLM `ExpertMapManager` concepts where compatible, but do not represent B70 as a fake expert-parallel rank.

**Gate:** every routed expert has exactly one normal-path owner, CUDA+B70 ownership covers the qualified placement, and no expert weights move during inference.

### Phase 6 — Integrate eager hybrid execution

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

Aggregate all scheduler-step remote token rows into one B70 layer request. Tune decode thresholds between tiny and grouped kernels. Add llm-scaler's prefill gather/up/down/accumulate path without moving router or shared-expert work to B70.

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
| CUDA hot experts + B70 llm-scaler provider | Shooting Brake result |
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
an isolated persistent PyTorch-XPU B70 process
qualified `intel-xpu/llm-scaler/` tiny/batched/prefill ESIMD kernels inside that process
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
