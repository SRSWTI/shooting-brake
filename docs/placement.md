# Expert Placement

## Purpose and status

This document specifies the production routed-expert ownership policy for one RTX 5090 CUDA state owner and one isolated B70 provider. It follows [`../plan.md`](../plan.md) and the boundaries in [architecture.md](architecture.md), [memory.md](memory.md), [scheduling.md](scheduling.md), and [benchmarking.md](benchmarking.md).

This is a design and qualification contract. The current Colibri implementation is reference evidence for compact ownership, correctness, and failure semantics; it is not evidence that the upstream-vLLM plus batched llm-scaler production placement is implemented or qualified.

## Normative ownership invariants

- Upstream vLLM on the RTX 5090 computes canonical router logits, top-k IDs, and routing weights. Placement never recomputes, suppresses, redirects, or invents a route.
- The RTX 5090 owns all sequential state, attention, KV/recurrent state, dense and shared paths, embeddings, residuals, sampling, and serving state.
- Each routed expert has exactly one immutable normal-path owner for the loaded placement generation: a CUDA local slot or a B70 compact slot.
- The hottest routed experts remain on CUDA. Qualified cold/overflow experts reside on B70 to increase usable model/KV capacity while sending a smaller share of latency-critical routes remotely.
- Gate, up, activation contract, and down weights for a routed expert remain co-located with that complete expert. No expert is split across CUDA and B70.
- Foreground inference moves activation rows, canonical IDs/weights, and one weighted compact `[M_remote, hidden]` B70 wire partial plus its row map; CUDA scatters it into the full `[M, hidden]` batch. It never moves expert weights.
- CPU is exact emergency recovery only. It is not a normal owner, cold tier, placement target, or source of normal-path matrix work.
- NVMe is artifact/startup/restart storage only, not an ownership or foreground execution tier.
- A selected route without a live, generation-matching CUDA or B70 owner produces an exact recovery mask/error. It is never silently omitted.

The B70 is a provider, not a fake expert-parallel rank. The B70 partial joins on CUDA before any final tensor/expert-parallel reduction.

## Immutable placement map

Every qualified model/provider combination has one explicit per-model map:

```text
(layer, global expert) -> CUDA local slot
(layer, global expert) -> B70 compact slot
```

The two mappings partition the routed experts admitted for normal service. The manifest records:

- source checkpoint and CUDA/B70 artifact fingerprints;
- architecture, layer count, hidden/intermediate dimensions, expert count, and top-k;
- CUDA and B70 physical quantization/layout contracts;
- provider protocol, dependency/kernel-bundle versions, and weight generation;
- placement generation and ownership fingerprint;
- actual packed bytes and slot per expert;
- supported decode/prefill kernel families and shape capacities; and
- exact recovery availability.

Startup fails closed if the CUDA adapter, model manifest, placement file, artifacts, or provider handshake disagree. An unsupported model runs unchanged on stock all-CUDA upstream vLLM; it is not forced through an unvalidated B70 path.

A request captures the active placement and weight/provider generations. The map and compact slots do not mutate while that request is in flight. A new map may publish only after every new owner is loaded, validated, and capacity-admitted. Late results from an old generation are rejected.

## Initial placement policy

The initial production policy is deliberately static:

```text
RTX 5090: state-owner tensors + shared expert + hottest routed experts
B70:     qualified cold/overflow routed experts
CPU:     exact emergency recovery only
NVMe:    startup and recovery artifacts only
```

Choose the CUDA hot set by measured workload frequency and expected critical-path savings, subject to preserving required RTX 5090 capacity for KV/recurrent state, local/shared scratch, remote join buffers, graph/runtime state, and safety headroom. Assign the remaining qualified routed experts to the B70 compact bank, subject to measured packed capacity and provider-buffer/headroom reservations.

The objective is not equal expert count or equal route share. It is:

> Move enough cold/overflow expert capacity to B70 to admit the target model and desired 5090 KV/concurrency budget while retaining the highest-traffic, most latency-sensitive experts on CUDA.

For a model that fits completely on the 5090, stock all-CUDA upstream vLLM is the expected throughput ceiling. Hybrid placement earns acceptance through measurable capacity gain at an explicit throughput/latency cost, not by claiming that both devices were utilized.

### Capacity constraints

For owner $d \in \{\mathrm{CUDA}, \mathrm{B70}\}$:

$$
\sum_{e:p(e)=d} \operatorname{packed\_bytes}(e) + R_d \le C_d,
$$

where $C_d$ is measured allocatable memory and $R_d$ includes all non-expert runtime, buffer, graph, KV/scratch, and safety reservations in that domain. Use the actual qualified physical representation, not nominal parameters or vendor capacity.

The placement is admissible only if:

1. every routed expert has exactly one normal-path owner;
2. every CUDA and B70 slot contains the artifact named by the manifest;
3. the B70 provider supports the model dimensions, top-k, group size, layout, dtypes, and required batch capacities;
4. no normal token depends on CPU or NVMe expert execution; and
5. the claimed capacity gain is measured after all reservations.

## Hot-set selection evidence

Use fixed, identified route traces from the production workload. Persist per-layer expert frequency and, where useful, coactivation and measured service distributions. A simple first score is expected CUDA critical-path time saved per packed byte:

$$
\operatorname{score}(l,e)=
\frac{f_{l,e}\left(T_{\mathrm{B70},l,e}^{\mathrm{exposed}}-T_{\mathrm{CUDA},l,e}\right)}
{\operatorname{packed\_bytes}(l,e)}.
$$

This is a ranking aid, not a correctness authority. Terms come from like-for-like measurements for the relevant decode/continuous-batch/prefill classes. Frequency alone is a valid baseline; isolated TFLOPS, advertised bandwidth, or equal striping are not sufficient evidence.

Because production has one B70, the previous multi-worker fan-out and balancing objective does not apply. Placement still considers:

- per-layer frequency and route skew;
- CUDA/B70 packed capacity and safety reservations;
- measured local and complete remote-path service time by batching class;
- B70 queue and exposed join wait;
- 5090 KV/context/concurrency value recovered by offloading an expert;
- provider shape/layout capability; and
- failure/recovery cost.

Prefill and decode may produce different scoring evidence, but the initial loaded map remains one immutable ownership map unless a separately qualified configuration explicitly provides multiple complete artifacts/maps. The scheduler may choose kernel families by shape; it may not change ownership per token.

## Per-layer route partition

For scheduled token rows `M`, `HybridMoERunner`/`HybridRoutedExperts` receives CUDA canonical top-k IDs and weights, then applies the placement map:

```text
CUDA-owned selected route -> local CUDA mask/compaction
B70-owned selected route  -> remote route mask and token/route map
missing/unhealthy owner    -> exact recovery mask/error
```

The masks partition the original routes without changing IDs or weights. Rows with no B70 route do not enter the remote request. B70 remaps global IDs to compact slots, evaluates only its owned route pairs, applies the original route weights, reduces by staged token row, and returns one `[M_remote, hidden]` wire partial with `token_row_map`. CUDA scatters it into a zero-initialized `[M, hidden]` buffer and adds that full-batch buffer to the local routed partial.

One active layer and scheduler step produces at most one aggregated B70 operation, regardless of the number of requests or B70-owned experts. Placement must not induce per-request B70 submissions.

## Failure and recovery ownership

On timeout, provider loss, invalid generation, missing compact slot, or kernel error:

1. mark the exact failed token-route entries;
2. invalidate or reject the B70 completion;
3. recompute those exact routes using an admitted CUDA/CPU correctness path, or fail the request explicitly; and
4. ignore any late B70 output after recovery.

Recovery does not make CPU a normal placement tier. Recovery count must be zero in a healthy normal-path benchmark and reported whenever nonzero. Provider restart increments its generation and reloads the complete immutable B70 bank before new hybrid requests are admitted.

## Colibri reference scope

The proven Colibri GS64 native path demonstrates persistent compact `(layer, expert) -> slot` B70 ownership, canonical route metadata, weighted-partial return, exact failed-route recomputation, and end-to-end CUDA+B70 generation without normal-path CPU expert fallback. Its placement evidence supports keeping high-traffic experts on CUDA while using B70 for substantial resident capacity.

Those results are reference/baseline evidence only. They do not validate the production Qwen-scoped out-of-tree adapter, llm-scaler batched provider, continuous batching, grouped prefill, immutable production manifest, actual capacity gain, or comparison with stock all-CUDA upstream vLLM.

## Placement qualification

A placement result reports, for the identical workload and checkpoint source:

- CUDA and B70 expert counts and actual packed resident bytes by layer;
- RTX 5090 state, KV/recurrent, scratch/join, graph/runtime, and headroom bytes;
- B70 stable tensor/scratch/runtime/headroom bytes;
- route share and remote-token share by layer and batching class;
- zero-remote-route layers and exactly one or zero B70 submissions per layer/step;
- local CUDA time, provider queue/copy/kernel time, exposed join wait, TTFT, ITL, and throughput;
- CPU exact-recovery count and cause;
- per-layer routed-output, final-logit, and generated-token agreement; and
- capacity gained and context/concurrency enabled relative to the same stock all-CUDA upstream-vLLM configuration.

Qualification compares at minimum the controlled configurations in [benchmarking.md](benchmarking.md): stock all-CUDA upstream vLLM, CUDA hot+B70 provider, CUDA hot+CPU cold offload baseline, reduced CUDA budget without B70, and the native B70 comparator where shapes match. Colibri timing is never the production acceptance baseline.
