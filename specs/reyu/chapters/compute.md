# Grouped quantized compute

This branch explains the B70-oriented grouped NVFP4 prefill path breadth first: why it exists, its observable interface and state, then each routing and arithmetic stage, backend alternatives, and finally the diagnostic programs that established individual claims. The executable abstraction is [`compute.ryu`](../models/compute.ryu). It is not a model of vLLM, the provider lifecycle, SYCL scheduling, or Xe instruction timing.

## 1. Responsibility, actors, and boundary

The entry point [`grouped_moe_nvfp4`](../../../src/phase7/xe2_nvfp4/grouped_moe.hpp) consumes one chunk of token activations and already-compacted route IDs. It groups the routes by resident expert, reads each resident expert's packed weights once through grouped GEMMs, computes SwiGLU and the down projection, and accumulates weighted expert results back into token order. It targets prefill: the interface comments explicitly say decode does not have the repeated-weight-read amplification this path removes.

The actors are:

- **Caller/provider:** owns an in-order SYCL queue, packed bank planes, activation/output storage, every scratch pointer, tensor geometry, route compaction, and fallback after a `false` return.
- **Grouped wrapper:** owns route histogram/scan/scatter/gather, backend selection, overflow scaling, down projection dispatch, and output accumulation. It catches every exception and returns `false`; exceptions do not cross the provider boundary.
- **Native Xe2 engine:** runs copied/modified CUTE/CUTLASS-SYCL grouped-GEMM machinery. [`.provenance`](../../../src/phase7/xe2_nvfp4/.provenance) fixes its derivation at vLLM-XPU kernels tree `a07c4844`: upstream supplies MXFP4 E8M0/group-32 machinery, while this tree adds NVFP4 E4M3/group-16 behavior.
- **oneDNN alternative:** binds the same caller queue through SYCL interop, keeps process-lifetime primitive/queue contexts and canonical scale copies, and returns control to native on refusal or failure.
- **MXFP4 alternative:** reuses the E2M1 payload but derives approximate E8M0/group-32 scales from two adjacent E4M3/group-16 scales. It is explicitly quality-gated, not numerically equivalent NVFP4.

The public shape contract is:

| Input/output | Shape and meaning |
|---|---|
| `act_src` | FP16 `[M, hidden]` source-token activations |
| `ids`, `route_w` | `[M, top_k]` compact expert IDs and FP32 route weights; IDs outside `[0,E)` are unowned and skipped |
| `w13`, `s13`, `alpha13` | packed E2M1 `[E,2I,H/2]`, E4M3 `[E,2I,H/16]`, FP32 `[E]` |
| `w2`, `s2`, `alpha2` | packed E2M1 `[E,H,I/2]`, E4M3 `[E,H,I/16]`, FP32 `[E]` |
| gathered scratch | `g_act [rows_cap,H]`, `g_mid [rows_cap,2I]` FP32, `g_gated [rows_cap,I]` FP16, `g_outr [rows_cap,H]` FP32 |
| routing scratch | histogram/offset/cursor/rows by expert; slot-to-row/expert/weight; inverse `slot_of [M*top_k]`; one-int work queue `atom` |
| `out` | FP32 `[M,hidden]` partial output |

The function refuses `M<=0`, `experts<=0`, `top_k<=0`, or any `group_size` other than 16. `rows_cap` is allocation capacity, while the device-resident `offs[E]` is actual resident-route count. The oneDNN arm additionally requires `rows_cap >= M*top_k`. The wrapper does not visibly validate all pointer capacities or integer multiplication overflow; those remain caller obligations.

## 2. State and operation map

The model uses one finite fixture: `M=2`, `top_k=2`, `E=2`, `H=2`, `I=2`, `rows_cap=4`, with IDs `[0,1,-1,0]`. This makes each transformation inspectable without pretending production dimensions are small.

| Model action | Source operation | Observable effect / atomicity abstraction |
|---|---|---|
| `configure` / `rejectUnsupportedShape` | `grouped_moe_nvfp4` entry guard and process-static environment parsing | Guarded parameters select native, oneDNN, or MXFP4; unsupported geometry refuses without a launch |
| `histogram` | stages 1, lines 275–286 | Atomically counts only resident IDs; fixture counts are `(2,1)` |
| `scan` | stage 2, lines 288–301 | One device task emits exclusive offsets `[0,2,3]`, initializes cursor and rows |
| `scatterRoutes` | stage 3, lines 303–324 | Relaxed atomic cursor assigns expert-major slots and writes inverse route map; order within an expert is unspecified |
| `gatherActivations` | stage 4, lines 326–342 | Copies only slots below `offs[E]`; this bound prevents stale scratch from indexing arbitrary rows |
| backend-specific gate/up action | stage 5, lines 344–391 | Produces FP32 `[resident routes,2I]`; resets/reuses `atom` for native/MXFP4 |
| `swiglu` | stage 6, lines 393–438 | Computes a bounded row power-of-two shift from the gate/up scalar, applies both `alpha13` factors abstractly, writes the narrowed value, and folds inverse scaling into route weight |
| `downGemm` | stage 7, lines 440–465 | Resets/reuses `atom`, maps `[rows,I] × [E,H,I]` to FP32 `[rows,H]` |
| `atomicOutput` | stage 8 default, lines 502–518 | Zeroes output and performs relaxed weighted atomic additions |
| `deterministicOutput` | stage 8 `reduce`, lines 477–500 | Traverses each token's top-k positions in fixed order and skips negative inverse entries |

The queue is in order, so the real implementation composes asynchronous submissions without host synchronization between these stages. The model makes each stage one atomic action: it captures ordering, guards, state reuse, and results, but not individual work-item races or device timing.

## 3. Route grouping and scratch ownership

Histogramming ignores both `-1` compaction markers and any ID `>=E`. The exclusive scan deliberately uses one thread because the deployed expert count is small and its measured cost was negligible. Scatter uses a relaxed per-expert cursor; it records `slot_row`, `slot_exp`, `slot_w`, and `slot_of`. `slot_of` begins as all `-1`, preserving the distinction between an absent route and a route with zero weight.

The gather bound `s < offs[E]` is a correctness condition, not only an optimization. The source records a prior failure in which gathering all capacity slots consumed uninitialized `slot_row`, caused out-of-bounds access, and eventually produced NaN log probabilities. The inverse-map output mode similarly tests `slot_of >= 0`; multiplying an absent route by zero is unsafe because IEEE `0 * NaN` is NaN.

Scratch is caller-owned and intentionally reused. `atom` is cleared before each native or MXFP4 GEMM leg. The model's `scratchEpoch` advances from gate/up to down only after the reset boundary. `rows_cap` is stable allocation geometry; actual work remains device-side in `rows` or inclusive `offs+1`, avoiding a host read of `offs[E]`.

## 4. Quantized grouped GEMMs

[`grouped_gemm_xe2.hpp`](../../../src/phase7/xe2_nvfp4/xe_2/grouped_gemm_xe2.hpp) maps each expert's row count into grouped tiles, computes packed-plane offsets, and drives a persistent atomic work queue. Four-bit weights use `E*N*K/2` bytes and scales at `E*N*(K/group_size)`. NVFP4 and MXFP4 share packed E2M1 payloads, but not scales.

[`gemm_xe2.hpp`](../../../src/phase7/xe2_nvfp4/xe_2/gemm_xe2.hpp) adds a distinct `NVFP4` dtype because E4M3 scale bytes cannot be decoded with MXFP4's exponent-only `bits << 23`. Native NVFP4 reads bank-native `[N,K/16]` scale rows, converts E4M3 bits through CUTLASS, multiplies decoded E2M1 fragments, accumulates in FP32, and writes FP32. The per-expert global alpha is deliberately applied outside the GEMM: it factors out of a linear dot product and avoids one multiply per weight.

`kLayoutB='C'` appears counterintuitive because the imported kernel inverts the layout selector; this value makes the bank's K-contiguous `[N,K/2]` bytes decode adjacent K nibbles correctly. Group size and K-tile are coupled. A K-tile of 32 would load only the first scale for two 16-element NVFP4 groups, silently applying a wrong scale. [`gemm_xe2_policy.hpp`](../../../src/phase7/xe2_nvfp4/xe_2/gemm_xe2_policy.hpp) therefore adds K=16 policies and FP32-compatible stores.

Native selection estimates rows per expert on the host as `(M*top_k)/(2*E)`: the divisor 2 is explicitly tied to the shipped two-card split. Below 64 it uses the small 32-row policy. At or above 64, `SB_GROUPED_BIGM` chooses process-statically among `m32` (default), `m64`, `m64n128`, `m128`, and `d32`. This estimate is not topology-general and the source says it must be re-derived for another card count.

## 5. Overflow-safe SwiGLU and weighted output

Gate/up results remain FP32 while the kernel finds each route row's largest absolute gated value. For finite maxima above FP16's `65504`, it chooses the smallest power-of-two right shift that brings the row into range, stores the shifted values in FP16 `g_gated`, and multiplies the route weight by the inverse power of two. Since down projection is linear, that factor can be restored during weighted scatter.

The executable finite witness uses raw gated value `70000`: shift 1 stores `35000`, changes route weight from 1 to 2, down projection by 3 yields `105000`, and three owned routes contribute `3 × 105000 × 2 = 630000`. Algebraically this equals `3 × 70000 × 3`. This witnesses transformation preservation over exact integers. It **does not prove numerical bit equivalence**: FP16 narrowing, nonlinear SiLU evaluation, DPAS accumulation order, infinities/NaNs, and float atomic order remain outside the model. Non-finite maxima do not enter the source's finite-overflow shift branch.

`alpha2` is also constant per expert and is multiplied during output accumulation. Default atomic scatter is membership-correct but not bit-stable: relaxed additions may occur in a different order and the source reports first-bit run-to-run churn. `SB_GROUPED_SCATTER=reduce` is parsed once, then assigns one work-item per token/channel and accumulates top-k positions in fixed order. It avoids the output memset and atomic read-modify-write traffic and is intended to be bit-stable across runs for the same backend/input.

The model therefore treats the two modes as producing the same exact representative scalar while marking only reduce mode `bitStable`. That is an explicit abstraction, not a claim that native atomic and deterministic GPU results are bit-identical.

## 6. Backend selection and failure modes

### Native

Native is the default for an unset or unrecognized `SB_GROUPED_BACKEND`. It operates directly on the bank-native weight and scale planes. A launch exception is caught by the outer function, which returns `false` for caller-owned GEMV fallback; this model stops at refusal/error visibility rather than inventing provider behavior.

### oneDNN

`SB_GROUPED_BACKEND=onednn` initializes the atomic `g_armed` latch. [`onednn_grouped_gemm`](../../../src/phase7/xe2_nvfp4/grouped_moe_onednn.cpp) stores one context per provider queue, one primitive pair per `(K,N)`, and process-lifetime canonical scale copies keyed by bank-plane pointer. The primitive accepts `rows_cap` as its descriptor's row upper bound while device offsets select actual rows. Packed weights alias zero-copy. Scales do not: first touch transposes `[E,N,K/16]` to canonical `[E,K/16,N]`.

A changed live `rows_cap` or expert count throws; allocation failure, oneDNN error, standard exception, or unknown exception all call `disarm`. `g_armed.exchange(false)` makes the transition permanent and limits logging to the first failure. The grouped wrapper sees `false` and runs native in the same GEMM leg. There is intentionally no re-arm path.

There is a source-level wording conflict worth preserving. `grouped_moe.cpp` calls oneDNN "bit-exact vs native," while [`grouped_backend_verify.cpp`](../../../src/phase7/xe2_probe/grouped_backend_verify.cpp) says DPAS accumulation order differs and accepts `max_rel < 5e-2` plus cosine `>0.9999`, not exact bytes. The specification uses the weaker, executable comparison contract; no bit-equivalence claim is made.

### approximate MXFP4

`SB_GROUPED_BACKEND=mxfp4` derives one E8M0/group-32 scale from each pair of E4M3/group-16 scales. It decodes their magnitudes, takes the mean in log2 space, rounds to a power of two, and caches the allocated plane by original bank pointer for process lifetime. A sign bit is treated as checkpoint corruption and the magnitude is clamped. Allocation failure returns `nullptr` and the current leg runs native; unlike oneDNN, no permanent disarm latch is set. This backend changes quantization and is approximate by construction.

Backend selection is process-static. Runtime environment changes after first parsing do not change it. The current default remains native; neither oneDNN nor MXFP4 is silently promoted.

## 7. Executable scenarios and properties

[`compute.ryu`](../models/compute.ryu) provides `init`, nondeterministic `step`, and `inv`, plus reachable witnesses:

- `nativeAtomicTest`: all eight logical stages, unowned-route filtering, native GEMMs, atomic output.
- `deterministicReduceTest`: same pipeline with fixed-order inverse-map reduction.
- `overflowScalingTest`: exact finite power-of-two scaling witness and final weighted value `630000`.
- `oneDnnSuccessTest`: armed oneDNN serves the gate/up leg.
- `oneDnnPermanentDisarmTest`: cached `rows_cap=3` conflicts with the invocation's `rows_cap=4`; `QueueCtx::bound` failure clears the latch and native completes the dispatch.
- `mxFp4ApproximateTest`: derived/cached scales are visibly approximate.
- `mxFp4AllocationFallbackTest`: scale allocation refusal selects native for the invocation.
- `unsupportedGroupTest`: unsupported entry geometry remains refused.
- `cannotGatherBeforeScatterTest`: stage ordering is enforced.

The invariant bounds owned/mapped/gathered routes, constrains reused scratch generations, prevents approximate labeling on a non-MXFP4 result, ties bit stability to deterministic completion, and makes a cleared oneDNN latch imply observed fallback.

## 8. Xe2 probes are harnesses, not serving paths

The five C++ programs under [`xe2_probe`](../../../src/phase7/xe2_probe/) are diagnostic harnesses:

- [`xe2_grouped_probe.cpp`](../../../src/phase7/xe2_probe/xe2_grouped_probe.cpp) is a speed-only policy sweep. Its original vendored MXFP4 data is explicitly not NVFP4 correctness evidence; the fork build adds group-16/NVFP4 policy measurements and uses a bandwidth threshold to decide whether porting is worthwhile.
- [`xe2_nvfp4_verify.cpp`](../../../src/phase7/xe2_probe/xe2_nvfp4_verify.cpp) uses one-hot activations, packed byte `0x21`, and E4M3 scale byte `0x38` to distinguish low-nibble-even (`0.5,1.0,…`) from swapped packing. It checks format arithmetic, not a full MoE layer.
- [`xe2_grouped_moe.cpp`](../../../src/phase7/xe2_probe/xe2_grouped_moe.cpp) maps a real bank and times histogram through atomic output. It predates the current wrapper: it stores intermediate/output legs as FP16 and says SwiGLU/scatter were timed but not reference-verified, whereas current `grouped_moe.cpp` uses FP32 `g_mid`/`g_outr`, overflow scaling, inverse mapping, and selectable deterministic output. Its timing must not be promoted into current correctness evidence.
- [`onednn_grouped_probe.cpp`](../../../src/phase7/xe2_probe/onednn_grouped_probe.cpp) has a CPU scalar oracle for E2M1/E4M3 arithmetic, demonstrates canonical-scale repacking and offsets-as-upper-bound behavior, includes an empty expert group, and separately measures full geometry.
- [`grouped_backend_verify.cpp`](../../../src/phase7/xe2_probe/grouped_backend_verify.cpp) invokes the current full entry on a deterministic production-shaped fixture, rejects empty/nonfinite output, dumps native/oneDNN results, and performs tolerance/cosine comparison. It is an A/B gate, not formal equivalence.

The four build scripts are reproducibility assets, not runtime components. [`build.sh`](../../../src/phase7/xe2_probe/build.sh) targets the vendored MXFP4 speed probe and records required SPIR-V extensions. [`build_nvfp4.sh`](../../../src/phase7/xe2_probe/build_nvfp4.sh) places the local fork first on include paths. [`build_onednn_probe.sh`](../../../src/phase7/xe2_probe/build_onednn_probe.sh) links the vendor oneDNN tree. [`build_backend_verify.sh`](../../../src/phase7/xe2_probe/build_backend_verify.sh) links already-built Phase-7 grouped objects and oneDNN. None of these scripts is invoked by serving.

## 9. Potential CPU implementation comparisons

No CPU expert implementation is owned by this specification, so the following are comparison opportunities rather than claims that were run:

1. Reuse the scalar E2M1/E4M3 oracle pattern from `onednn_grouped_probe.cpp` to compute both gate/up and down for a small fixture, then compare native, oneDNN, and the exact NVFP4 branch before route accumulation.
2. Bucket the same route IDs on CPU, retain top-k order, and compare deterministic scatter exactly where operations have identical rounding; compare atomic output by tolerance because addition order is intentionally unconstrained.
3. Exercise overflow rows around `65504`, independently calculate the selected power-of-two shift, and compare the product after down projection and restored route weight. Include non-finite rows to confirm that the source only rescales finite maxima.
4. Compare MXFP4 against the CPU NVFP4 oracle as a quality metric, never as an exactness check, because its paired log-space scale conversion is lossy by design.

These checks are listed in the correspondence map for the parent to execute where appropriate. They require actual source/device availability and are not established by the Reyu state machine.

## 10. Limits and dependencies

CUTE/CUTLASS-SYCL, oneAPI SYCL, Xe2 DPAS/block-I/O extensions, oneDNN, and the provider's bank/storage lifetime are external boundaries. The copied `xe_2` headers are imported kernel machinery with a local NVFP4 delta; they are not a general Shooting-Brake matrix library. Environment parsing is process-static. Performance figures and dated silicon comments are measurements, not invariants. The model does not represent exact E4M3/E2M1 bits, FP16 rounding, cache lifetime, memory allocation, work-group scheduling, device exceptions, or performance. Those are retained as source-linked assumptions and implementation checks rather than idealized away.
