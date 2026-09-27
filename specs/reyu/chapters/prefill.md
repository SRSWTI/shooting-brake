# Weight residency and offloaded prefill

This branch explains how a routed-expert layer changes weight residency and how a batch larger than decode staging capacity is completed. Read it breadth first: first the ownership boundary, then the state transitions, then the exact engine branches and source-visible edge cases. The executable abstractions are [`residency.ryu`](../models/residency.ryu) and [`prefill.ryu`](../models/prefill.ryu).

## 1. Responsibility, actors, and interfaces

The subsystem has two coupled responsibilities:

1. **Residency preparation.** Keep CUDA-owned experts addressable by compact local IDs, preserve any weights needed by CPU or DRAM-streaming paths, and ensure offloaded routes still receive a remote partial. Post-load surgery briefly holds the complete expert bank in VRAM and slices it later. Preemptive surgery allocates only CUDA owners and obtains host-tier weights from an offline bank.
2. **Large-batch composition.** Compute the CUDA partial with offloaded route weights masked to zero, compute CPU and B70 partials by the applicable engine, then add the disjoint partials. Large batches do not fit the decode-sized staging buffers, so native dispatch is capacity-chunked. Alternative engines move one layer's weights to CUDA and use grouped kernels.

The main actors are vLLM's routed-expert object and fused CUDA kernel; Shooting Brake's immutable placement/remap, host arena and B70 lanes; the native B70 and CPU pollers; a host-backed Marlin or B12x bank; CUDA copy/compute streams and events; and the external checkpoint/bank builders. Attention, request scheduling, numerical kernel internals, CUDA event implementation, and bank construction are dependency boundaries.

The live branch selector is in [`routed_experts.py`](../../../src/phase4/src/shooting_brake_vllm/routed_experts.py), which is shared with the integration specification. `forward_modular` sends `M > SHOOTING_BRAKE_B70_MAX_BATCH` (default 128) to `_prefill_forward_offloaded`. That threshold is the forward's aggregate token-row count, not request length. The assigned streamer modules implement two alternative prefill engines; the mirror module is not imported by the live routed-expert path.

## 2. Source-backed state and operations

| State or operation | Source authority | Behavioral meaning in the models |
|---|---|---|
| Full versus compact CUDA allocation | `routed_experts.py::preemptive_surgery_enabled`, `_preemptive_cuda_ids` | `FullThenSlice` has a transient all-expert VRAM footprint; `CompactFromStart` never allocates offloaded rows. |
| Host materialization before slicing | `HybridRoutedExperts._load_host_experts`, `_load_host_experts_from_bank` | Post-load host copies precede slicing. Under preemptive allocation the offline bank is the only source for absent experts. |
| Compact indexing handoff | `HybridRoutedExperts._maybe_perform_vram_surgery`, `_finalize_compact_experts` | Global-to-local CUDA remap must exist and the loader's temporary expert map must be retired before serving. |
| Large-M dispatch | `forward_modular`, `_prefill_forward_offloaded` | CUDA masks offloaded routes, then CPU and B70 partials are added. A B70-active layer without graph/Tier-3 mode is rejected instead of silently dropping its partial. |
| Per-card chunk issue/join/copy | `_b70_prefill_partial_dispatch` | For every capacity chunk, all card doorbells are issued before any card is taken; card results are summed; the sum is copied into durable output before reusable lane views are overwritten. |
| CPU large-M alternative | `_cpu_prefill_partial` | Above the CPU stream threshold and outside capture, stream weights to CUDA; otherwise native CPU issue/take is capacity-chunked. |
| B70 NVFP4 DRAM streaming | `_b70_prefill_partial` | At or above `SHOOTING_BRAKE_B70_STREAM_T` (default 1024), with a populated arena and outside capture, compute B70-owned routes on CUDA from global expert IDs. |
| Marlin selection | [`marlin_prefill.py::MarlinPrefillStreamer`](../../../src/phase4/src/shooting_brake_vllm/marlin_prefill.py) | The first prefill alternative when `SHOOTING_BRAKE_PREFILL_MARLIN=1`, graph mode is active, and the current stream is not capturing. |
| B12x selection | [`marlin_prefill.py::_get_streamer`](../../../src/phase4/src/shooting_brake_vllm/marlin_prefill.py), [`b12x_prefill.py::B12xPrefillStreamer`](../../../src/phase4/src/shooting_brake_vllm/b12x_prefill.py) | `SHOOTING_BRAKE_PREFILL_B12X=1` changes which singleton serves the Marlin custom-op boundary; the Marlin flag is still required to reach it. |
| Proposed DRAM mirror | [`prefill_mirror.py`](../../../src/phase4/src/shooting_brake_vllm/prefill_mirror.py) | Selects and sizes a balanced duplicate subset, but remains standalone helper state with no serving transition. |

`residency.ryu` abstracts packed tensors to resident expert counts but retains the unsafe ordering and indexing conditions. `prefill.ryu` abstracts numerical values to result validity while retaining capacity, per-card issue/take order, copy-before-reuse, engine precedence, and rejection.

## 3. Residency algorithms

### 3.1 Post-load surgery

`HybridRoutedExperts._maybe_perform_vram_surgery` runs lazily on the first `forward_modular` after vLLM has processed loaded weights. For a layer with B70 or CPU owners, it determines sorted CUDA global IDs and, when the tensor is still full-size, requires `SHOOTING_BRAKE_VRAM_SURGERY=1`, `SHOOTING_BRAKE_B70_DEVICE=1`, and `SHOOTING_BRAKE_HYBRID=1`. Missing execution flags are errors, not a request to retain unsafe compact weights.

Before this slice, `_load_host_experts` copies any CPU-owned experts and, when legacy B70 prefill streaming is enabled, B70-owned experts into a shared packed NVFP4 host arena. The sets are disjoint by placement. It preserves packed weights, converts block scales from the checkpoint's swizzled order to the arena's linear order once, and handles the backend-dependent fused gate/up order. Only after host materialization may surgery `index_select(...).clone()` weights and scales down to CUDA owners, redirect quant-config descriptors, and create the global-to-local remap. Offloaded routes later map to a valid compact slot with zero weight; their real partial is added separately.

This mode reduces steady-state VRAM but cannot reduce the transient loading peak. The source explicitly identifies that limitation as acceptable for the 35B bank and insufficient for a 122B full-bank load.

### 3.2 Preemptive surgery

The preemptive loader hooks allocate only CUDA-owned rows. The checkpoint loader temporarily uses an expert map, after which `_finalize_compact_experts` installs the plugin's authoritative remap and retires the loader map. Finalization is required even if the tensor already has the desired compact length; skipping it can index a compact bank with global IDs and produce wrong tokens without an indexing exception.

`_reject_unsupported_preemptive_tiers` requires hybrid and B70-device execution whenever anything is offloaded. A readable bank is additionally required when CPU ownership or B70 DRAM prefill streaming needs host copies. `_load_host_experts_from_bank` reads raw checkpoint-order planes with already-linear scales and folds the layer activation global scale. It deliberately distinguishes compact bank rows from absolute model layer/expert coordinates in the CPU arena.

### 3.3 Mirror helper: useful design, disconnected implementation

[`prefill_mirror.py::select_mirror_ids`](../../../src/phase4/src/shooting_brake_vllm/prefill_mirror.py) round-robins across sorted B70 device lists, sorts the chosen global IDs, and clamps naturally when all cards are exhausted. `build_slot_map` maps selected globals densely and uses `-1` elsewhere. `arena_gib` accounts for three NVFP4 planes at one packed nibble plus one E4M3 block scale per 16 weights. `validate_mirror_budget` compares requested bytes to `/proc/meminfo` `MemAvailable` minus configurable headroom before allocation.

The module docstring describes a runtime prefill mirror, but repository source has no runtime import from `HybridRoutedExperts`; the only assigned caller is [`prefill_mirror_unit_test.py`](../../../src/phase6/prefill_mirror_unit_test.py). Therefore it is a tested experimental helper, not current serving behavior. `residency.ryu` makes that contradiction explicit by preserving `mirrorIntegrated == false` in every reachable state.

## 4. Prefill engine branches

### 4.1 Common composition and failure boundary

`_prefill_forward_offloaded` first gathers B70 compact slots and optional CPU global IDs. It computes the CUDA partial using the compact remap when surgery exists, otherwise by zeroing weights for all offloaded routes. CPU partials are added next. If the layer owns B70 experts but graph/Tier-3 mode is inactive, it raises before returning because no enabled branch would restore the masked B70 routes. This guards against plausible but incorrect output.

The provider's compact slot space and the DRAM arena's global expert-ID space are deliberately different. Native B70 dispatch gathers each physical card's slot map. DRAM streaming receives global IDs masked to `-1` for other owners. Interchanging the two can select a different expert or an arena entry never loaded.

### 4.2 Chunked multi-card dispatch

`_b70_prefill_partial_dispatch` allocates the final `[M, hidden]` output and precomputes one route-ID tensor per card. For every `[start:end]` of at most `max_batch` rows it:

1. calls `_b70_issue_graph` for **every** lane;
2. takes the first lane and adds every other lane's partial;
3. assigns the joined chunk to `out[start:end]` before beginning the next chunk.

The copy is load-bearing because `_b70_take_graph` returns a view backed by a reusable per-lane device buffer. Ringing all cards before waiting allows the cards to work concurrently. `prefill.ryu` has separate `issued`, `taken`, `cursor`, and `copiedRows` state so both ordering requirements can fail independently.

The CPU-native fallback similarly copies each taken chunk into durable output, but has one native poller rather than a per-card sum.

### 4.3 DRAM/NVFP4 streaming

The legacy stream branch is enabled only when the option was resolved at construction together with graph mode, the batch reaches the forward-row threshold, a host arena exists, global IDs were prepared, and CUDA stream capture is inactive. Its streamer groups routes by expert, transfers packed NVFP4 blocks over a copy stream, dequantizes and computes on CUDA, and accumulates weighted rows. During capture it falls back to the pure-stream-operation native chunk path because selecting touched experts requires a host synchronization.

The CPU tier makes the analogous choice at its lower threshold (default 128 in `cpu_stream.py`): CUDA streaming outside capture, capacity-chunked CPU poller execution otherwise. This is a useful implementation comparison because the same ownership/masking contract is exercised by two compute locations. A CPU-executable differential can compare route selection, dense/sentinel ID maps, and chunk assembly; packed numerical equivalence requires the existing native reader/oracle or GPU kernels and is outside these abstract models.

### 4.4 Marlin ring

[`MarlinPrefillStreamer.__init__`](../../../src/phase4/src/shooting_brake_vllm/marlin_prefill.py) opens a pre-repacked `SBMARL01` sidecar and rejects a missing file or activation/scale dtype mismatch; the latter is fail-closed because the kernel can otherwise return zeros silently. It creates one or more device arenas, typed views for `m13`, `m2`, `s13`, and `s2`, a copy stream, and per-slot `copy_done` and `consumed` events.

`prefetch(layer)` uses `layer % ring_size`, waits until the compute stream has consumed that slot, copies the contiguous layer arena, records copy completion, and marks the slot's layer. `partial` returns zeros when no B70 route is active; otherwise it waits for copy completion, masks sentinel routes by zero weight and clamps their ID, and runs `fused_marlin_moe`. Token rows are tiled to cap temporary scratch while reusing the one streamed arena. It then records consumption, invalidates the slot label, and prefetches the next layer so copy can overlap current/remaining compute.

A registered custom op keeps mmap access, cross-stream copies and event side effects opaque to vLLM compilation. `_get_streamer` is a process singleton: its first construction fixes the selected engine and activation dtype for that worker.

### 4.5 B12x ring

[`B12xPrefillStreamer`](../../../src/phase4/src/shooting_brake_vllm/b12x_prefill.py) follows the same ring/event/tiling contract but reads `SBB12X01` native-FP4 planes: `w1`, `w2`, pre-swizzled block scales, and per-expert fp32 alphas. It requires bf16 activations and constructs a `flashinfer.fused_moe.B12xMoEWrapper`. Explicit all-one input and FC2 scales prevent the weight alpha from being misused as the FC1 activation-quantization scale.

B12x applies route weights inside the wrapper. Sentinel IDs are neutralized by zeroing their weights and clamping IDs to zero. Its W4A4 activation quantization is numerically distinct from Marlin W4A16; selection is an opt-in alternative, not a source-backed claim of bitwise parity.

## 5. Assigned harnesses and source-visible limitations

| Assigned file | Exact role | What it establishes or does not establish |
|---|---|---|
| [`prefill_mirror_unit_test.py`](../../../src/phase6/prefill_mirror_unit_test.py) | CPU-only functional harness | Checks balanced selection, degenerate/all-CUDA cases, dense slot maps, sizing, host-headroom refusal, and layer stability. It does not connect the helper to serving. |
| [`async_overlap_test.py`](../../../src/phase8/async_overlap_test.py) | Hardware integration harness | Launches all-CUDA, asynchronous B70, and synchronous B70 eager runs; requires async and sync exact tokens and accepts baseline divergence only when all texts contain `42`. This is decode overlap evidence, not a prefill-streamer test. |
| [`vram_surgery_test.py`](../../../src/phase8/vram_surgery_test.py) | Hardware integration harness | Exercises post-load surgery, checks path markers and output parity, and prints VRAM measurements. It does **not** assert that measured VRAM decreased. Its `vram_post_load` sample precedes lazy first-forward surgery; the post-generation sample is the relevant compact state. |
| [`smoke_benchmark.py`](../../../src/phase8/smoke_benchmark.py) | Manual benchmark client | Warms up, times one 512-token generation and eight 256-token generations, then prints placement/surgery. Although its module description mentions TTFT, it does not isolate TTFT; the reported `Mean ITL` divides whole-request wall time by output tokens. It has no parity assertion. |

No gate or benchmark above was run while authoring this chapter.

## 6. Executable scenarios and properties

`prefill.ryu` contains:

- `multiCardChunkJoinTest`: two chunks, both cards issued before takes, joined result copied before reuse;
- `cannotReuseBeforeCopyTest` and `cannotTakeBeforeAllCardsIssuedTest`: edge-ordering failures;
- `dramStreamingSuccessTest`, `marlinSuccessTest`, and `b12xAlternativeTest`: reachable alternative-engine witnesses;
- `cpuNativeChunkFallbackTest`: capture/under-threshold CPU-native shape;
- `missingGraphRejectedTest`: the masked-B70-without-restoration error.

Its invariant keeps durable progress equal to copied progress, taken lanes a subset of issued lanes, remote completion tied to all rows, and join tied to valid local and remote work.

`residency.ryu` contains post-load success and missing-flag rejection; inability to slice before host copies; preemptive bank-backed success and missing-bank rejection; inability to serve while the loader map remains live; and accepted/refused but disconnected mirror plans. Its invariant preserves CUDA/offload coverage, compact-map handoff, and non-integration of the experimental helper.

These runs are bounded witnesses, not numerical proofs and not claims of exhaustive exploration.

## 7. Abstraction limits and implementation comparisons

The models intentionally omit tensor payloads, floating-point accumulation order, DMA/cache-coherence details, CUDA-event memory semantics, allocator fragmentation, timing, and subprocess/vLLM behavior. A model `remoteValid` bit means that the selected engine supplied its abstract partial; it does not imply Marlin/B12x numerical equivalence.

Potential parent-run CPU comparisons are:

1. execute the existing mirror helper harness to compare selection, slot maps, sizing and budget errors against `residency.ryu` scenarios;
2. source-inspect or add a pure small-array differential for chunk partitioning and copy-before-reuse, using independent immutable per-chunk arrays rather than a native poller mock;
3. compare post-load and bank-reader packed planes with the existing real expert-bank reader on a tiny bank, including gate/up ordering and linearized scales;
4. inspect the phase8 harness result predicates separately from their prose, especially the lack of a required VRAM reduction and the smoke benchmark's non-TTFT timer.

Marlin/B12x ring execution, true asynchronous copies, vLLM custom-op compilation, native CPU/B70 execution and token parity require their real dependencies and hardware; a CPU mock would not validate those mechanisms and is not proposed.
