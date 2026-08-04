# Shooting Brake Correctness Contract

## Purpose, authority, and status

This document defines the normative correctness contract for the architecture in [`../plan.md`](../plan.md). It is a design and qualification target, not a claim that the upstream-vLLM plus B70 production path has been implemented or passed.

The production direction is one upstream vLLM 0.26+ CUDA state owner on the RTX 5090 and one isolated, persistent QuixiCore-XPU provider on the B70. Phase 1 qualified the native provider core directly for batched NVFP4 expert execution, Phase 2 made that issue/take path process-facing through the protocol-v2 eight-slot ring, and Phase 3 completed the independent provider-mathematics gate on the physical B70. The existing Colibri implementation remains separate reference evidence for transport, placement, failure handling, and the signed-S4 GS64 native worker. Phase 4 upstream-vLLM adapter integration, CUDA scatter/join, and later layer/logit/generation parity remain open.

The terms **MUST**, **MUST NOT**, **SHOULD**, and **MAY** are normative.

## Canonical computation and ownership

For each qualified sparse layer and scheduler step, upstream vLLM on CUDA owns:

- hidden state, attention, KV cache, and recurrent state;
- router logits and the canonical top-k selection;
- canonical global expert IDs and unmodified routing weights;
- CUDA-resident routed experts and the shared expert;
- the routed-result join, any final tensor/expert-parallel reduction, residual continuation, LM head, and sampling.

`HybridMoERunner`/`HybridRoutedExperts` may partition only the already-selected routed-expert work. The B70 MUST NOT recompute router logits, softmax, top-k, route normalization, or shared-expert work.

For `M` scheduled token rows:

```text
hidden             [M, hidden]  FP16 or BF16
topk_ids           [M, topk]    int32, canonical CUDA result
topk_weights       [M, topk]    FP32 or FP16, canonical CUDA result
remote route mask and token/route map
```

Only rows with at least one B70-owned route need cross the pinned-memory boundary. Their original token-row and canonical route-position identities MUST be retained so that the returned rows can be scattered deterministically into the full batch.

One immutable placement generation maps each routed expert to exactly one normal-path owner:

```text
(layer, global expert) -> CUDA local slot
(layer, global expert) -> B70 compact slot
```

The B70 is not an expert-parallel rank. A B70 compact slot is provider-private and MUST be resolved only through the validated placement and weight generation. CPU is orchestration and exact emergency recovery only; it MUST NOT execute normal-path expert matrix work.

## Exactly-once route invariant

For every canonical selected token-route position, exactly one weighted expert contribution MUST enter the routed result:

1. preserve original token row, route position, global expert ID, and routing weight;
2. assign the route to its sole current CUDA or B70 owner;
3. commit that owner's result once, or close that lane and recompute the same route on an exact CUDA/CPU recovery path;
4. reject every later or duplicate completion for the closed lane; and
5. fail the request explicitly if exact recovery is unavailable.

Placement MUST be invisible to logical model semantics. No selected route may be dropped, substituted, zero-filled, renormalized, reassigned to a different logical expert, or counted twice.

Masked/padded entries remain governed by canonical vLLM router semantics and MUST NOT be promoted to selected routes. A selected route with zero weight still has a defined route identity and completion state. A batch or token with no remote route contributes the additive identity on the remote lane; it does not waive accounting for CUDA-owned routes.

The following are forbidden:

- approximate or placement-biased top-k;
- rerouting around an unavailable owner;
- expert dropping or capacity overflow that changes canonical selections;
- treating a failed remote route as zero;
- renormalizing surviving routes;
- accepting stale placement, weight, provider, sequence, or buffer generations;
- silently changing activation, weight, scale, accumulation, or output precision; and
- normal-path CPU matrix compute.

## B70 transaction result

The B70 provider accepts only its compacted subset of canonical routes. For original token row `m`, it computes:

$$
Y_{\mathrm{B70}}[m,:]
=
\sum_{j\in R_{\mathrm{B70}}(m)}
w_{m,j}\,\mathrm{Expert}_{e_{m,j}}(X[m,:]).
$$

The response is one already-weighted, already-summed partial:

```text
remote_partial      [M_remote, hidden]  FP16 or FP32
token-row map       [M_remote]
route completion    per original token/route position
```

`M_remote` is the number of staged rows and may be zero. After deterministic scatter, unstaged rows are zero. The provider MUST NOT return one host-visible tensor per expert or require repeated activation transfer for multiple remote routes belonging to one token.

Per-route completion metadata MUST state exactly which remote routes are represented in the partial. Failed routes MUST be excluded from that partial and marked for exact recovery. If the provider cannot prove the successful subset after an error, the entire remote subset is uncommittable and MUST be recomputed exactly or the request MUST fail. A combined partial with ambiguous membership MUST NOT enter the join.

## CUDA join and reduction ordering

After identity and completion validation, the state owner copies the remote partial into a preallocated CUDA buffer and forms:

$$
Y_{\mathrm{routed}}
=
Y_{\mathrm{CUDA\ local}}
+
Y_{\mathrm{B70\ remote}}
+
Y_{\mathrm{exact\ recovery}}.
$$

The terms are accumulated in a documented stable order, not completion-race order. The remote partial MUST join on CUDA **before any final tensor-parallel or expert-parallel reduction**. The shared expert remains CUDA-owned and is combined according to canonical upstream vLLM semantics; the B70 result MUST NOT bypass or duplicate that logic.

At initial qualification TP=1. Any later TP/EP mode requires its own proof that the hybrid routed partial joins at the same semantic point as stock vLLM and participates in the final reduction exactly once.

## Publication, identity, and stale-completion protection

The pinned-memory ring uses release/acquire publication:

- CUDA publishes a request only after its device-to-host copies and descriptor writes are complete;
- the B70 provider claims a request only after acquire validation;
- the provider publishes a response only after the XPU-to-host output and status writes are complete;
- CUDA claims a response only after acquire validation; and
- a slot is reusable only after every CUDA and XPU reference has ended.

Before any response can affect the join, it MUST match the immutable request plan on all applicable identity fields:

- provider protocol and descriptor version;
- request sequence and ring slot;
- provider generation;
- placement and weight generation/fingerprint;
- layer;
- staged token count, route count, hidden size, and top-k;
- token-row and route-subset identity;
- activation/output dtype;
- request and output buffer identity/version; and
- successful or explicitly partial terminal status.

Sequence distinguishes ring publications and wraparound. Provider generation distinguishes worker incarnations. Placement/weight generation distinguishes compact ownership and resident weights. Buffer version distinguishes reused storage. None substitutes for another.

A mismatch, late reply, duplicate reply, provider restart, or completion after recovery closes the remote lane without consuming its payload. The missing routes then follow an exact recovery outcome or the request fails explicitly. Storage whose backend use cannot be proven complete MUST be quarantined rather than reused.

## Exact failure semantics

The only permitted outcomes for each selected route are:

1. its current normal-path owner succeeds and the route commits once;
2. that lane is closed and the same logical route is recomputed exactly on an available CUDA or CPU correctness path, then commits once; or
3. the request fails explicitly.

Timeout, device loss, kernel error, invalid capability, invalid compact slot, generation mismatch, backpressure, cancellation, and restart MUST NOT become silent partial success or indefinite waiting. Recovery uses the original activation, global expert ID, routing weight, and canonical route position. CPU recovery is request-scoped emergency work, never a steady-state placement tier.

The existing Colibri recovery-mask behavior is proven reference evidence for exact failed-route recovery. Production requires a batched per-token/per-route status representation rather than Colibri's single-token mask.

## Capability and model fail-closed behavior

The CUDA adapter MUST compare the model manifest, placement, CUDA artifact, B70 artifact, and provider handshake before publishing hybrid work. Protocol version, model identity, architecture, dimensions, top-k, activation/output dtypes, quantization format, group size, scale dtype, layout, kernel family, capacities, provider generation, weight generation, and placement fingerprint MUST agree.

An absent, unknown, or mismatched field is incompatibility. Unsupported shapes, dtypes, models, route counts, and kernel families MUST fail before execution. A model that is not qualified for B70 remains on stock upstream vLLM CUDA; it MUST NOT be silently sent through a generic B70 kernel. A hybrid placement whose required owner/artifact is unavailable MUST fail startup or use an explicitly validated all-CUDA placement.

## Numerical contract

Both provider artifacts MUST derive independently from the same BF16/FP16 higher-precision source checkpoint and bind to its exact identity. CUDA and B70 expert outputs are validated separately against that source before their sum is validated.

Allowed numerical differences are only those covered by a declared precision contract and pre-agreed tolerance for:

- source-to-provider quantization;
- activation conversion;
- gate/up, activation, and down-projection arithmetic;
- routing-weight multiplication;
- provider accumulation and returned-partial dtype;
- CUDA scatter/add order; and
- any later qualified TP/EP reduction.

Tolerances MUST be fixed before observing results and MUST specify absolute/relative error, zero-reference handling, and NaN/Inf policy. Speed does not waive an unexplained mismatch. Route identity, ownership, generation, shape, and completion checks are exact even when tensor comparison uses a tolerance.

The primary QuixiCore-XPU NVFP4 path has now passed two distinct numerical gates. Phase 1 qualified direct provider execution; Phase 3 independently authenticated the full packed bank and frozen NVFP4 shard manifest, byte-audited sampled bank records against the NVFP4 artifact, and computed float64 BF16-source and NVFP4 expert outputs for layers 0 and 31, experts `0,1,7,63,127,191,254,255`, and eight deterministic FP16 inputs. The frozen Phase-3 quality gate passed with worst sampled-expert relative RMSE `0.1683012879458`, minimum cosine `0.985919468279`, aggregate relative RMSE `0.1579548618065`, and aggregate cosine `0.987528585785`. These facts qualify the sampled source-to-NVFP4 artifact agreement and the tested B70 wire partial, not the CUDA artifact, heterogeneous CUDA join, or end-to-end model. The proven native Colibri signed-S4 GS64 path remains reference-only; llm-scaler INT4 remains a separately qualified secondary alternative requiring its own artifact and numerical gate.

## Qualification hierarchy

Passing a lower level does not establish a higher one.

1. **Artifact/source comparison:** validate each CUDA or B70 artifact independently against the common higher-precision source at dequantized rows, gate/up output, activation, down output, and complete expert output. Phase 3 closes this level only for the sampled primary B70 NVFP4 records and outputs described below; the CUDA artifact and secondary INT4 artifact remain separate gates.
2. **Batched provider-wire mathematics:** compare the returned weighted `[M_remote, hidden]` wire partial with exactly the B70-owned staged routes for supported full scheduler batches. Phase 3 closes this level for the frozen primary NVFP4 fixture and tested `M=1..128` process-ring cases.
3. **CUDA scatter and layer replay:** scatter the compact rows deterministically into the full `[M, hidden]` CUDA buffer, preserve canonical CUDA top-k, and compare the post-join routed result with all-CUDA execution, including mixed ownership, no remote routes, compact remapping, and join-before-reduction placement.
4. **Teacher-forced replay:** compare routes, per-layer results, logits, recurrent/KV behavior, and top-1 tokens across long and multi-turn sequences.
5. **Generation:** compare deterministic greedy generation and required quality workloads, including intentional B70 failure and restart.

Required staged-provider cases include all staged routes remote and mixed local/remote semantic subsets; duplicate and non-sorted IDs under canonical semantics; multiple tokens choosing one expert; unequal, zero, and near-zero weights; boundary compact slots; changing decode/prefill batch sizes; invalid capability and generations; timeout; kernel failure; and stale completion. Adapter/ring cases separately include a zero-remote batch, which must issue no provider request and must preserve an additive-identity CUDA remote lane, plus ring wraparound and cancellation.

**Phase-3 observed gate.** `phase3/provider_math_test` passed on the physical B70. It compared the process-ring wire partial against the independent NVFP4 oracle for a one-remote-route sweep at every `M=1..128`; an all-remote `M=4` case with duplicate, non-sorted, boundary IDs and an observable $2^{-12}$ weight; and sparse/interleaved mixed ownership with local-only route perturbations proving remote-lane invariance. The provider ran with compact resident order `255,0,7,63,127,191,254,1`, exercising canonical-global-to-provider-local remapping. The same gate proved zero-remote no-publication, exact raw wire identity/status, unchanged allocation accounting, sequence-bound split- and fused-kernel failures with no exposed payload, explicit rejection of `M=0` and `M=129`, and trusted bank/placement/weight bootstrap rejection.

This closes the independent B70 provider-mathematics and process-ring failure matrix for the tested primary NVFP4 bundle. It does **not** prove the CUDA artifact, CUDA scatter/add, upstream-vLLM adapter behavior, join-before-reduction placement, layer/logit/generation parity, concurrency, throughput, recovery execution, or production acceptance; those remain Phase 4 and later gates.

No document currently records the upstream-vLLM plus B70 production path as having passed qualification levels 3–5. Phase 3 is complete only at the independent artifact/source and pre-CUDA-scatter provider-wire boundary described above.

## Failure matrix

| Condition | Required outcome | Forbidden outcome |
|---|---|---|
| Matching current B70 completion | Accept its proved route subset once | Duplicate or race-ordered addition |
| No remote route | Skip B70 submission and use a zero remote lane | Treat all selected routes as complete |
| Partial route failure with exact status | Join only proved successes; recover exactly the marked failures | Add failed/ambiguous routes or recover successes twice |
| Ambiguous B70 failure | Discard the whole remote partial; recover the remote subset exactly or fail | Guess which routes completed |
| Deadline/backpressure/device loss/kernel error | Close remote lane; exact CUDA/CPU recovery or explicit failure | Indefinite wait, zero fill, or lossy substitute |
| Late or duplicate completion | Reject and drain only for resource safety | Reopen a closed lane |
| Sequence, provider, placement, weight, or buffer generation mismatch | Reject before H2D/join; recover or fail | Best-effort acceptance |
| Invalid global ID or compact slot | Reject before execution | Substitute another resident expert |
| Provider capability mismatch | Fail startup/request as appropriate; keep unsupported model on stock CUDA | Generic-kernel fallback |
| Numerical gate failure | Mark that provider/artifact combination unqualified | Widen tolerance after the result |

## End-to-end acceptance

The hybrid path is correctness-eligible only when all selected routes are accounted for exactly once; both artifacts trace to one higher-precision source; batched B70 partials pass their oracle; the CUDA join occurs before final TP/EP reduction; stale and ambiguous work cannot enter a request; exact failure recovery or explicit failure is demonstrated; no normal-path CPU matrix work occurs; and teacher-forced, recurrent/KV, logit, and generated-token behavior pass their declared gates.
