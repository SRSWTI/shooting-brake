# Expert Placement

## Purpose and status

This document specifies how Shooting Brake assigns routed experts to the state-owner GPU, B70 workers, and exact fallback tiers. It elaborates the placement boundary in [architecture.md](architecture.md); it does not replace the system architecture.

This is an approved design and measurement plan, not evidence that adaptive placement, promotion, replication, prediction, or any particular capacity split has been implemented or validated on the target machine. Luce Spark and Colibri provide upstream policy ideas—traffic-derived frequency, persistent profiles, bounded capacity, hot/cold separation, asynchronous promotion, and routing telemetry—but their results are not measurements of this five-GPU fabric.

Related contracts are described in [hardware.md](hardware.md), [memory.md](memory.md), [correctness.md](correctness.md), [benchmarking.md](benchmarking.md), and [scheduling.md](scheduling.md).

## Normative invariants

Placement MUST change performance only; it MUST NOT change model semantics. In particular:

- The router MUST make the same decisions independent of placement. Selected expert IDs, route weights, expert precision, and faithful reduction behavior remain unchanged.
- Placement and prediction MUST NOT manufacture, suppress, or redirect a route to suit residency.
- The primary 5090 remains the state owner for the sequential residual stream, attention and KV state, router execution, dense layers, shared experts, embeddings, and sampling.
- Secondary GPUs are compute workers with resident routed-expert weights. Ordinary decode moves activations and route metadata to those weights and returns weighted partials; it MUST NOT move expert weights on the foreground token path.
- Exact DDR5 execution remains the fallback for an expert that is absent, unhealthy, or late. NVMe is for preload, recovery, and background movement, not an ordinary foreground decode tier.
- A remote routed expert is the indivisible placement and scheduling unit. Its gate, up, and down matrices MUST remain co-located under one whole-expert owner or replica. Splitting those matrices would add serial backend transitions, break the one-activation/one-result contract, and prevent a fused resident expert kernel.
- Promotions, replication, route prediction, and prefetch are optimizations only. None may be correctness-critical.

For dense and other state-owner tensors, tensor-level placement may be evaluated separately. This document requires whole-expert placement for remote routed experts.

## Stage 1: prove static placement

The first usable policy MUST be static for the lifetime of the measured run:

```text
5090: hottest routed experts within a bounded routed-expert budget
B70:  next warm resident expert bank
CPU:  remaining exact experts that fit in DDR5
NVMe: recovery and preload only
```

Shared experts and the state-owner work listed above remain on the 5090. The initial assignment SHOULD:

1. rank the bounded 5090 routed-expert set using learned per-layer frequency and expected critical-path savings;
2. assign the remaining warm experts to B70s, respecting each device's usable capacity;
3. balance expected selected-expert bytes or measured work across workers within each layer;
4. penalize layouts that touch four cards for a lightly loaded layer;
5. leave unassigned experts available through exact CPU execution; and
6. avoid replication until measured saved latency justifies its capacity cost.

During this stage there is no swapping during a token, prediction, promotion, demotion, or live migration. Learned Lucebox/Colibri-style route frequencies may initialize the layout, but frequency is a static baseline rather than the final objective. A naive round-robin expert hash is acceptable only as a transport or random/uniform comparison, never as the final policy.

### Capacity and ownership constraints

For an initial single-owner assignment $p(e)$, every device $d$ MUST satisfy

$$
\sum_{e:p(e)=d} \operatorname{bytes}(e) \le C_d,
$$

where $\operatorname{bytes}(e)$ is the measured packed size of the complete expert and $C_d$ is measured usable expert capacity after the device's other required allocations. Capacity planning MUST use the actual quantized/prepacked representation, not nominal parameter size.

The placement table MUST identify the whole-expert execution owner for each placement epoch. A remote call sends one activation plus route metadata to that owner and receives one weighted expert partial. Replication may later provide more than one resident execution copy, but dispatch still selects a complete copy; it never splits one expert's matrices across devices or changes which expert the router requested.

### Static measurements and hard gate

Measure at least:

- hit share for the 5090, B70, CPU, and recovery/preload tiers as applicable;
- B70 token batch sizes;
- experts and devices contacted per token and remote calls per layer;
- CPU fallback count or share;
- B70 queue depth;
- placement memory by device;
- total decode rate; and
- p95 and p99 latency.

Static B70 placement MUST beat all three baselines before any adaptive policy proceeds:

1. CPU-only cold-expert execution;
2. frequency-only 5090 placement with no B70 workers; and
3. random or uniform B70 placement.

The comparison MUST preserve correctness and use the same workload/route evidence as the candidate. If static B70 placement does not prove value against these baselines, work stops at diagnosis; session adaptation, coactivation optimization, replication, promotion, and prediction MUST NOT be introduced to conceal the failure.

## Weighted hypergraph objective

At layer $l$, the router selects a set $S_l$ of experts (eight in the source design). Co-selected experts form a weighted hyperedge: their assignment jointly determines device fan-out, per-device work, transport, queueing, and the layer's distributed critical path. The goal is therefore not simply to maximize cache hit rate.

A first critical-path objective is

$$
\min_p \sum_l \mathbb{E}_{S_l}\!\left[
\max_d\!\left(
\sum_{e\in S_l,\,p(e)=d} T_{d,e}(B)
+ \mathbf{1}_{d\ne 5090}T_{\mathrm{transport},d}(B)
\right)
+ \alpha\left|\{d:p(e)=d,\ e\in S_l\}\right|
\right],
$$

subject to the capacity constraint above. The maximum term represents the expected layer critical path and penalizes worker imbalance. The device-count term penalizes fixed command/transport overhead and needless fan-out. The relative weight $\alpha$ MUST come from measured hardware behavior rather than assumption.

The complete optimization evidence SHOULD account for:

- global, session, and recent route frequency;
- expert coactivation;
- B70 and 5090 capacity and complete packed expert size;
- measured per-device, per-expert kernel time at relevant batch sizes;
- number of devices touched and per-device work imbalance;
- transport time, queue depth, and queue delay;
- migration cost and observed migration outcome;
- replication benefit and capacity cost;
- CPU fallback cost;
- layer criticality;
- deadline class; and
- prefill/decode differences where measurements justify separate layouts.

A useful per-expert/device ranking term is

$$
\operatorname{benefit}(e,d)=
f_e\left(
T_{\mathrm{fallback},e}
-T_{\mathrm{device},e,d}
-T_{\mathrm{transport},d}
-T_{\mathrm{queue},d}
\right)
-\lambda_{\mathrm{fanout}}\Delta F,
$$

again subject to measured B70 and 5090 capacity. This expresses expected critical-path savings rather than popularity alone: a frequent expert is not valuable on a device if transport, queueing, imbalance, or added fan-out consumes the saved fallback time. Conversely, measured compute savings, coactivation, or layer criticality may make a less frequent expert more valuable. Frequency remains an input and baseline, never the final objective.

## Frequency at three timescales

For layer $l$ and expert $e$, the policy may combine:

$$
F_{l,e}=\alpha F^{\mathrm{global}}_{l,e}
+\beta F^{\mathrm{session}}_{l,e}
+\gamma F^{\mathrm{recent}}_{l,e}.
$$

- **Global frequency** is the stable workload profile persisted across restarts.
- **Session frequency** represents the current conversation, agent, or tenant.
- **Recent frequency** is a short adaptation window for workload shifts.

These components MUST remain observable separately even if the policy uses a combined score. Coefficients and window lengths are measurement-dependent policy parameters, not established constants. Static placement begins with the global/per-layer evidence; session and recent weighting are introduced only after the static hard gate passes.

## Immutable placement epochs

Placement changes publish as versioned, immutable epochs:

1. a request captures one placement epoch at admission;
2. all tokens and routed expert calls for that request use that epoch until completion;
3. background work may prepare a later epoch without mutating the captured map; and
4. the later map becomes selectable only after its experts are ready and the epoch is published.

This rule prevents an in-flight request from referring to an expert whose owner moved. During asynchronous promotion, the CPU or prior owner remains authoritative until publication completes. A failed, canceled, or incomplete promotion leaves exact execution available and MUST NOT affect router output.

## Adaptive placement after the static gate

Only after static placement proves value may optimization be introduced, in evidence-producing increments:

1. per-layer frequency;
2. measured device cost;
3. balanced layer striping;
4. coactivation-aware assignment;
5. selective replication;
6. session and recent-window adaptation; and
7. optional prediction plus background prefetch.

Each increment MUST run against the same route-trace replay used for its comparison. Its gate is lower p95/p99 layer time with no correctness regression or migration instability. Adding policy complexity without an attributable improvement fails the gate.

### Session rescue capacity

A small session-rescue region on each B70 may be evaluated for:

- session-dominant experts;
- predicted next-layer experts;
- duplicated stragglers; and
- temporary hot-set adaptation.

The source design suggests **potentially** 1–3 GiB per B70 only after measurement. That range is an unverified design hypothesis, not reserved capacity or evidence that the target hardware has room for it. The region's existence and size depend on measured packed expert size, foreground allocations, hit benefit, queue behavior, migration cost, and the capacity displaced from the stable bank. If those measurements do not justify it, the region is zero.

### Asynchronous promotion and demotion

Promotion and demotion MUST occur outside the foreground token dependency chain. Promotion prepares a complete expert copy in spare capacity, validates readiness, and exposes it only through a later placement epoch. Until that publication, exact CPU execution or the prior resident owner remains available. Foreground execution MUST NOT wait synchronously for NVMe or for an unfinished promotion.

Promotion is never correctness-critical: cancellation, copy failure, stale demand, or insufficient capacity merely prevents the candidate epoch from becoming active. Migration cost and outcome MUST be recorded so the policy can reject churn that costs more than it saves.

### Selective replication

Replication is permitted only when measured saved critical-path or straggler latency exceeds:

- the replica's complete packed capacity cost;
- any promotion/migration cost;
- added maintenance and placement constraints; and
- the opportunity cost of the displaced expert bank.

Candidate uses include measured coactivation/fan-out reduction and duplicated stragglers. Replicas MUST preserve whole-expert co-location and exact routed computation. Replication MUST NOT change router selection or route weights, and loss of a replica MUST fall back to another exact owner or CPU rather than alter the route.

### Optional prediction

Cross-layer route prediction and prefetch are optional and MUST NOT be foundational. They may be enabled only when route-trace replay demonstrates useful recall and all of the following hold:

- speculative copies are cancelable;
- prediction uses spare bandwidth;
- false positives cannot evict experts required by foreground work;
- prediction failure falls back to exact execution; and
- no router decision is changed or manufactured to match predicted residency.

Prediction comes after static placement and the preceding measured optimization steps. It cannot repair a static layout that failed its baselines.

## Persistent placement profile

The placement-learning store SHOULD persist, with explicit profile version and model hash:

- global per-layer expert frequency;
- session frequency, subject to the serving system's session-lifetime policy;
- recent-window frequency;
- expert coactivation counts;
- device service-time distributions;
- transport p50, p95, and p99;
- queue delay;
- migration outcomes;
- prediction precision and recall; and
- failure rates.

This extends upstream “which experts are hot?” telemetry into “where should each complete expert execute?” A stored profile is evidence for initialization, not an authority over current measurements. Model-hash or profile-version mismatch MUST prevent incompatible placement data from being applied, and recalibration is required when measured hardware behavior changes.

## Evidence classification

To avoid overstating readiness:

- **Normative design rules** are the invariants, static gate, capacity constraints, epoch behavior, and optimization gates in this document.
- **Upstream claims and reusable concepts** come from Luce Spark, Colibri, and related prior work; they motivate the policy but do not validate this topology.
- **Design targets** include critical-path minimization, lower p95/p99 layer time, selective replication, session adaptation, and prediction.
- **Unverified, measurement-dependent assumptions** include usable per-device expert capacity, the proposed 1–3 GiB rescue range, objective weights, timescale coefficients, migration thresholds, replication value, predictor quality, and any performance benefit.

No adaptive policy is accepted until the static placement measurements establish that B70 residency itself beats the required baselines without changing correctness.