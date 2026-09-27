# vLLM integration and hybrid execution

This branch explains how Shooting Brake enters vLLM, replaces the routed-expert container and runner, chooses an execution path, and joins partial outputs. Read it breadth first: the registration and request-level picture comes before module mechanics. Placement, doorbell transport, and prefill/residency each have independent chapters; this chapter names their contracts without absorbing their implementations.

Executable abstractions: [integration.ryu](../models/integration.ryu) and [hybrid.ryu](../models/hybrid.ryu).

## 1. Responsibility, actors, and boundary

The plugin is discovered through vLLM's general-plugin interface. Its registration hook is gated before it changes process-global vLLM classes. Once enabled, it installs `HybridRoutedExperts` and `HybridMoERunner` as an inseparable pair. vLLM remains responsible for API serving, scheduling, attention, routing logits/top-k selection, and the stock modular expert kernel. Shooting Brake owns qualification at the seam, expert ownership/remapping, remote dispatch, and addition of routed partials.

At one sparse layer, the logical result is:

```text
router global IDs and weights
        |
        +--> mask/remap CUDA-owned routes --> stock modular CUDA partial
        +--> per-card compact B70 IDs ------> one partial per B70
        +--> CPU-owned global IDs ----------> optional CPU partial
                                              |
                   CUDA partial + every returned remote partial
                                              |
                                      routed layer result
```

Ownership is exclusive. A remote route is zero-weighted out of the CUDA partial before its remote contribution is added. Slot `0` may be substituted for an offloaded ID only after its weight is zero; it is a fused-kernel-valid dummy address, not ownership. Placement determines ownership; the integration path consumes it.

The historical `ShootingBrakeExpertProviderClient` is not the live B70 transport. `HybridMoERunner._forward_impl` calls only `begin_all_cuda()` on it as bookkeeping, then delegates. Its `issue()` and `take()` deliberately raise. Live B70 calls are made directly by `HybridRoutedExperts` through the native binding and poller modules owned by the provider and doorbell branches.

## 2. HLD state and operations

| State/operation | Source-backed meaning | Modeled action |
|---|---|---|
| Plugin disabled | `register()` returns without registry or monkey-patch changes when `phase4_enabled()` is false | `skipDisabled` |
| Pair registered | Both `RoutedExperts` and `MoERunner` names are installed; a conflicting foreign replacement is rejected | `register`, `rejectRegistryConflict` |
| Runner constructed | Configuration is requalified and the contained routed layer must be `HybridRoutedExperts` | `constructQualified`, `rejectWrongContainer` |
| Legacy seam observed | Each delegated runner forward increments `all_cuda_calls`; no live remote data crosses this object | `delegateRunner` |
| Shared fusion | Compatible inference-only Laguna gate/up projections are merged and cached; unsafe modules permanently fall back | `armFusion`, `disarmFusion` |
| Hidden capture | Tagged requests accumulate selected layer boundaries on CPU, truncate at requested token count, then atomically publish a record | `captureOneToken`, `persistCapture` |
| All-CUDA | A layer with no active offload is a stock `forward_modular` pass-through | `routeAllCuda`, `finishAllCuda` |
| Eager reference | Remote ownership is masked from the local call and recomputed with the CUDA kernel when no B70 device is configured | `computeReference` |
| Eager synchronous | Local CUDA executes, then blocking provider dispatch returns B70 output | `computeLocalSync`, `dispatchSync` |
| Eager asynchronous | Every B70 issue precedes local CUDA; every take follows it | `issueAsync`, `computeLocalAsync`, `takeAsync` |
| Graph hybrid | CPU and all B70 doorbells are issued before local CUDA; captured waits/copies happen afterward; the partials are added on CUDA | `issueGraphRemotes`, `computeLocalGraph`, `releaseGraph`, `join` |
| Offloaded prefill | Masked local work is joined with B70 and optional CPU contributions produced by prefill engines | `computePrefillLocal`, `finishPrefillRemotes`, `join` |
| Released failure | A poller may release CUDA after native failure to avoid an untimed GPU wait; validity is observed later in Python | `observeReleasedFailure` |

`integration.ryu` treats a registration or monkey-patch call as atomic. `hybrid.ryu` preserves issue/local/take ordering but makes each numerical kernel atomic. Its route IDs `0/1/2` are finite representatives of CUDA/B70/CPU ownership, not production expert counts.

## 3. Registration and runner/container mechanics

[`shooting_brake_vllm/__init__.py`](../../../src/phase4/src/shooting_brake_vllm/__init__.py) is a referenced dependency because it owns `register()`. The gate is evaluated **before** `/tmp/sb_env_overrides.json` is applied, so that file cannot turn an initially disabled plugin on. Once enabled, registration applies experiment overrides; optionally forces `CUDAGraphMode.PIECEWISE` for breakable hybrid graphs; normalizes nested Qwen checkpoint names and quantization ignore paths; installs post-warmup command-streamer arming, seam/profiler hooks, Laguna shared fusion, and hidden capture; then registers the two OOT implementations. Preemptive allocation is patched before any routed layer is constructed because vLLM calls `create_weights` inside its base constructor.

The registry accepts an absent entry or the same implementation. A different existing implementation raises rather than composing two replacements. This preserves the runner/container pairing.

[`runner.py`](../../../src/phase4/src/shooting_brake_vllm/runner.py) re-runs model qualification in `HybridMoERunner.__init__`, delegates base construction, and requires the resulting routed container to be `HybridRoutedExperts`. `_forward_impl` is an eager-break boundary for breakable graphs. It optionally diagnoses non-finite inputs, records the legacy all-CUDA seam call, then calls the stock runner, which still owns shared/routed composition. Static-output helpers exist for stable addresses across graph segments, although the current `_forward_impl` directly returns the base result. NaN diagnostics are opt-in and synchronize tensors; they are diagnostic, not normal-path validation.

[`provider.py`](../../../src/phase4/src/shooting_brake_vllm/provider.py) records `begin_all_cuda` calls. The class name can suggest a live provider, but source behavior contradicts that interpretation: `issue` and `take` always raise “disabled until Phase 6.” Current remote dispatch is in `routed_experts.py` and native binding/poller modules.

## 4. Container setup and common routing rules

[`routed_experts.py`](../../../src/phase4/src/shooting_brake_vllm/routed_experts.py) is the execution center.

Construction qualifies the model, builds and validates immutable placement before base construction, derives device maps, constructs one `_B70Lane` per physical card, allocates pinned staging at model-derived hidden/top-k geometry, caches environment decisions, and creates graph flags/device buffers only when graph mode is truly active. Graph mode requires all three of `SHOOTING_BRAKE_B70_GRAPH=1`, a configured B70 device, and hybrid enabled. The CUDA-graph switch is distinct from the command-streamer doorbell switch.

The first eager profiling forward is the setup point for the resolved modular backend, absolute layer/device map, provider/poller registration, and optional route observers. Loading a provider during construction could race CUDA setup and starve weight loading; doing it during capture would violate capture. Weight surgery and host-arena population happen earlier through the post-load hook because vLLM must see freed VRAM when sizing the KV cache.

A monolithic MoE backend is rejected because Shooting Brake overrides only `forward_modular`. Construction checks it and first forward checks again after backend resolution. Source documents a residual limitation: if vLLM bypasses this method entirely and calls inherited `forward_monolithic`, neither guard runs; dispatch-count evidence is therefore necessary at qualification time.

`forward_modular` observes global IDs before remapping, passes all-CUDA layers directly to vLLM, sends batches larger than `SHOOTING_BRAKE_B70_MAX_BATCH` to the separate prefill composition, and otherwise enters hybrid decode. If a graph poller previously incremented an error counter, the next Python-visible forward raises because the released buffer may be stale.

## 5. Decode path selection and joining

### All-CUDA

`_all_cuda_passthrough` is true when hybrid, shadow, and surgery features are absent. The input IDs and weights go unchanged to the stock modular method. This is the baseline exercised by the adapter smoke and parity harnesses.

### Eager reference without a B70

The eager path computes `partition_routes`, optionally validates it, and compacts CUDA routes when surgery supplied a remap. If hybrid is enabled but no B70 device is selected, a second stock CUDA call computes only the B70 mask and is added to the local partial. This is a semantic integration reference, not native heterogeneous execution.

### Eager native synchronous and asynchronous

With a real device and `SHOOTING_BRAKE_B70_ASYNC=0`, local CUDA is computed first and `_b70_partial` synchronously copies tensors to CPU/NumPy, dispatches each provider, sums card outputs, and copies the FP32 result back to model dtype on CUDA.

Asynchronous eager is the default when a device is present. `_b70_issue` stages BF16 activations as FP16, synchronizes the current CUDA stream after D2H staging, translates global IDs into each lane's private compact slots, zeros foreign-route weights, and issues all providers. CUDA then computes the masked local partial. `_b70_take` collects into preallocated pinned outputs, performs nonblocking H2D copies/casts, and sums disjoint card partials. This overlaps B70 kernels with CUDA but still contains host synchronization and Python/ctypes calls, so it cannot run inside stream capture.

### CUDA-graph hybrid

Graph mode performs no Python partition or host synchronization during replay. Each lane gathers through its own global-to-slot CUDA map. The union mask plus optional CPU mask zeros all offloaded weights from CUDA. CPU is issued first because it is normally slower, then every B70 doorbell is rung before local CUDA. `_b70_take_graph` equality-waits for completion, copies the lane result back, casts to BF16, and resets completion. Multiple B70 providers execute in parallel even though CUDA waits and adds in a deterministic sequence.

Completion is not output validity. Native classic pollers release the completion flag after dispatch failure to prevent a permanent CUDA wait. The replay may therefore consume stale output. The plugin detects the poller error counter only when Python regains control and rejects the next forward as untrustworthy; there is no per-route recovery.

### Returned partial joining and address stability

Eager hybrid returns `_write_static_output(y_cuda + y_b70)` so breakable graph segments retain stable output addresses. Graph hybrid directly returns `y_cuda + summed B70 + optional CPU`. Both rely on exclusive ownership: no route can be counted twice, and every required route must appear in one partial. The offline oracle checks the captured sum exactly, separate from numerical agreement of the native partial.

## 6. Offloaded prefill

Large token-row batches take `_prefill_forward_offloaded`. It computes a CUDA union mask, adds an optional CPU mask, compacts/remaps CUDA IDs, and obtains the local partial first. CPU prefill either streams weights to CUDA above its threshold outside active capture or chunks through its graph-compatible poller. B70 prefill selects, in order, the opt-in opaque Marlin/B12x custom-op path outside capture; the DRAM arena streamer when enabled, above threshold, available, and not capturing; otherwise chunked B70 doorbells. Each chunk rings every card before waiting, and its result is copied before reusable lane buffers are overwritten.

Provider slots and host-arena identities are deliberately different. Per-card providers receive compact slots. Streaming receives global IDs because the shared host arena is keyed by `(layer, global expert)`. Mixing them can silently read a different expert.

If a layer owns B70 experts but large-batch offloaded prefill runs without graph/Tier-3 mode, the implementation raises. It does not fall back because the needed flag/buffer machinery was not allocated, and proceeding would drop masked B70 contributions while still producing plausible finite tokens. Detailed streamer, residency, and doorbell mechanics belong to their independent chapters.

## 7. Bank reader and numerical oracle

[`expert_bank.py`](../../../src/phase4/src/shooting_brake_vllm/expert_bank.py) is the CPU-only reader used by host loading and oracles. `ExpertBank` validates the legacy `SBEXP001` header and exact file length, memory maps it read-only, and slices packed NVFP4 gate/up/down planes plus scales. `Int4ExpertBank` validates `SBINT401` dimensions and plane sizes, maps explicit source expert IDs to compact residents with binary search, and exposes GPTQ int4/group128 planes. `open_expert_bank` dispatches by magic. `global_scale_divisor` requires a positive activation scale and documents why host planes must share CUDA's activation fold for partials to be addable.

[`int4_aggregation_oracle.py`](../../../src/phase4/int4_aggregation_oracle.py) is the principal CPU implementation comparison. It independently decodes K-major GPTQ nibbles as `(nibble - 8) * scale`, computes gate/up/SwiGLU/down per routed expert, applies route weights, and compares this CPU partial with captured native B70 output. It separately requires each stored hybrid output to equal its stored CUDA+B70 partial sum exactly; requires inactive B70 output to be zero for all-CUDA and inactive CUDA output to be zero for all-B70; and applies relative-L2/cosine thresholds only to provider-versus-CPU numerics.

[`int4_aggregation_oracle_test.py`](../../../src/phase4/int4_aggregation_oracle_test.py) is a focused CPU harness for nibble order, zero point, negative scales, and zero error metrics. It is inventory evidence, not a claim that the oracle was run for this documentation.

[`graph_aggregation_oracle.py`](../../../src/phase4/graph_aggregation_oracle.py) intends to compare A→B→A captured graph replays with the CPU oracle, distinguish each result from the previous fixture, stress completion/reset ordering, and check exact CUDA+B70 addition. There is a source contradiction: it calls `get_b70_poller(placement)` and `register_layer(layer_idx=...)`, while the live poller API described by current source requires the newer `top_k` and compact `bank_row` interface. The harness is therefore API-drifted and should not be represented as currently runnable without repair.

## 8. Shared-expert fusion

[`fused_shared.py`](../../../src/phase4/src/shooting_brake_vllm/fused_shared.py) is installed only by an explicit Laguna flag. Importing it alone changes nothing. It patches `LagunaMLP.forward` idempotently. Training or enabled autograd always delegates to stock behavior.

For inference, `_build_merged_nvfp4` requires compressed-tensors W4A4 NVFP4, a CUTLASS-compatible swizzled layout, no bias, exact dtypes and shapes, equal activation-global scale, equal logical output width/padding, and no unsafe output-row padding. Gate and up packed rows/scales are concatenated. A single GEMM uses gate alpha; when up alpha differs, `_make_output_rescale` multiplies only the up half by `up_alpha / gate_alpha`. One input quantization, one merged FP4 GEMM, raw SiLU/multiply, and stock down projection replace the two projections. Any incompatibility or runtime exception permanently disarms that module and immediately returns the original result.

[`tests/test_fused_shared.py`](../../../src/phase4/tests/test_fused_shared.py) is a CPU algebra harness proving concatenated codes/block scales and the per-half alpha correction match two separate matmuls. It does not exercise the CUDA custom operations.

## 9. Hidden-state capture

[`hidden_capture.py`](../../../src/phase4/src/shooting_brake_vllm/hidden_capture.py) is another explicit Laguna opt-in. `install` patches `GPUModelRunner` initialization before model loading/capture, requires auxiliary boundaries `(2, 11, 20, 30, 39, 48)`, and observes execution only after the original `execute_model` returns. Request IDs carry base64 corpus identity, response start, and total token count; malformed or ordinary IDs are ignored.

`CaptureAccumulator` copies one layer at a time to CPU, stacks `[T,K,H]`, replaces non-finite features with zero, truncates chunks at the declared total, stores int32 token IDs and BF16 features, constructs the response loss mask, and maintains flattened and by-layer views over shared storage. It writes a `.part` file and uses `os.replace` for atomic publication, clears active state in `finally`, and treats an existing destination as an idempotent completion.

[`tests/test_hidden_capture.py`](../../../src/phase4/tests/test_hidden_capture.py) covers tagged-ID parsing, multi-chunk truncation, dtypes/shapes/loss mask, shared flattened storage, atomic-part cleanup, duplicate idempotence, and ignoring ordinary requests.

## 10. Harness inventory and observable contracts

| Assigned file | Role and contract |
|---|---|
| [`adapter_smoke.py`](../../../src/phase4/adapter_smoke.py) | Live eager all-CUDA harness: requires the enable gate, exact qualified model, both OOT registrations, and at least one generated token. |
| [`adapter_parity.py`](../../../src/phase4/adapter_parity.py) | Cross-process stock-versus-adapter harness: exact greedy token IDs/text are gates. Router set drift is explicitly diagnostic because top-k ties can vary across process launches. |
| [`int4_hybrid_enablement_test.py`](../../../src/phase4/int4_hybrid_enablement_test.py) | CPU admission/placement harness for the 88B int4 path: 54 CUDA/126 B70 per layer, compaction/dummy-zero behavior, 3072-wide staging, language-model-only admission, and malformed ownership/header rejection. It uses a stand-in vLLM configuration and does not prove live vLLM APIs. |
| [`hybrid_execution_test.py`](../../../src/phase6/hybrid_execution_test.py) | Live eager comparison of all-CUDA with `split:128` hybrid using CUDA recomputation for the remote mask when no physical B70 is configured. Requires a hybrid marker and identical greedy tokens. |
| [`int4_aggregation_oracle.py`](../../../src/phase4/int4_aggregation_oracle.py) | Offline CPU functional comparison and exact captured-partial sum check. |
| [`int4_aggregation_oracle_test.py`](../../../src/phase4/int4_aggregation_oracle_test.py) | CPU unit harness for dequantization and metric identity. |
| [`graph_aggregation_oracle.py`](../../../src/phase4/graph_aggregation_oracle.py) | Live graph freshness/order oracle intent; currently contradicted by source-visible poller API drift. |

No harness, test, build, formatter, linter, GPU workload, or server workload was run while authoring this chapter.

## 11. Executable scenarios and properties

`integration.ryu` provides reachable witnesses for disabled registration, paired registration, delegated runner calls, armed and disarmed shared fusion, atomic hidden capture, foreign-registry rejection, and wrong-container rejection.

`hybrid.ryu` provides named scenarios for all-CUDA passthrough, CUDA recomputation reference, synchronous eager, asynchronous issue/local/take ordering, successful graph joining, completion-with-invalid-result followed by error observation, prefill joining, and rejection of an unsafe non-graph prefill configuration.

Its invariant checks route-domain bounds, pairwise ownership disjointness, issued/released/valid nesting, local completion before joining, and the remote-masking obligation. These are model properties under the declared abstraction, not proofs of CUDA, driver, native provider, or floating-point behavior.

## 12. Limits and external dependencies

vLLM's plugin loader, registry, configuration lifecycle, scheduler, graph compiler, stock modular MoE implementation, and Laguna/GPU runner classes are external. PyTorch tensor semantics, CUDA graph capture, CUDA memory ordering, native B70/CPU execution, filesystem atomic-replace behavior, and hardware timing are also external. Numerical tensors are reduced to contribution identity in the Reyu model; the CPU oracles are the appropriate concrete differential checks.

Independent detail branches own: exact placement and bank identity; preemptive/post-load residency and prefill streamers; graph flags, poller lifecycle, and command-streamer variants; and CPU-tier kernel/storage details. Integration assumes their advertised ownership, release, and partial-output contracts and makes their cross-branch ordering explicit.
