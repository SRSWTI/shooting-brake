# Superveloce Jota r20 on Dual B70

**Living plan.** Update the progress log at the bottom as items land. Opened
2026-08-20.

Target: `srswti/axe-superveloce-jota-118b-r20-nvfp4` served through the
Shooting Brake plugin with the dual-B70 split, to the same standard the 99B
reached — correct greedy tokens, finite logprobs, both cards dispatching,
measured prefill and decode — so that the **only** remaining work is the
GuideLLM acceptance matrix.

Companion docs: `docs/superveloce-99b-dual-b70.md` (the 99B's serving state,
still the production model), `docs/kill-bench.md` (the microbenchmark
ledger — every claim that can move an SLO gets an entry there, not here).

---

## 1. Why r20

Laguna is a **different architecture**, not a Qwen variant. The decision rests
on one property no amount of RAM, kernel work or placement tuning can buy on
the 99B.

```
Qwen3_5MoeForCausalLM: is_hybrid=True,  supports_mamba_prefix_caching=False
LagunaForCausalLM:     is_hybrid=False
```

The 99B has 36 GDN/linear-attention layers carrying a **recurrent state**, so
vLLM disables prefix caching for the whole model. Measured: `0.0%` hit rate,
and the same 5,524-token prompt three times at 9.788 / 9.729 / 9.723 s. Zero
amortisation, permanently.

Laguna is 36 sliding-window + 12 full attention — ordinary KV, nothing
recurrent. **Measured 2026-08-20** via `benchmarks/laguna_gate.py` (stock
vLLM, CPU offload, no plugin involvement):

```
prefix_caching=True   is_hybrid=False   window=512
run 0: 37.7252s  ptok=6023      <- cold
run 2:  0.2653s  ptok=6023      <- warm
speedup = 142.198x
```

Correct tokens, finite logprobs, PASS. The absolute latencies are meaningless
(weights stream from host every forward); the **ratio** is valid because both
arms traverse the identical path, and 142× is inflated relative to the real
path because the cold arm is ~4× slower than it would be on the B70s. What is
**not** performance-dependent is that the prefill is genuinely *skipped*.

Prefill is **86–92% of our TTFT**. A 10-turn agentic session with an 8K
system prompt costs 102,500 tokens of cumulative prefill on the 99B (147 s
today, 43.5 s with the 96 GB RAM upgrade) versus 12,500 tokens (~7 s) on r20.

### r20, not r15

Both are Laguna and both need identical integration work. r20 keeps **205
experts, byte-identical to the 99B's width**, so `fractional:2:0.2634`, the
54-CUDA/151-remote policy, the resident-set arithmetic and the 76/75 two-card
split all carry over untouched. r15 raises it to 218: new placement maths,
new resident-set artifacts, banks 1.063× larger. r15 is strictly more
expensive for the same benefit — add it later as another spec row.

`r10` is unusable: shard 2 never downloaded (67.1%, 0 bytes in flight).

---

## 2. Geometry, measured from the checkpoints

| | 99B (production) | jota-r20 | jota-r15 |
|---|---:|---:|---:|
| architecture | `Qwen3_5MoeForCausalLM` | `LagunaForCausalLM` | `LagunaForCausalLM` |
| `model_type` | `qwen3_5_moe_text` | `laguna` | `laguna` |
| checkpoint | 58.21 GiB | **50.60 GiB** | 53.62 GiB |
| layers | 48, all MoE | 48: **47 sparse + 1 dense (layer 0)** | same |
| experts / top_k | 205 / 8 | **205 / 10** | 218 / 10 |
| routes/token | 384 | **470 (+22.4%)** | 470 |
| bank (47×205×5.0625 MiB) | 48.65 GiB | **47.63 GiB** | 50.65 GiB |
| per-expert, all layers | 0.2373 GiB | **0.2324 GiB** | 0.2324 GiB |
| dense on 5090 (loaded) | 10.24 GiB | **3.18 GiB** | 3.18 GiB |
| attention | 12 full + 36 GDN | 12 full + **36 SWA(512)** | same |
| kv heads × head_dim | 2 × 256 | **8 × 128** | 8 × 128 |
| KV per token | 29.5 KiB | **50.2 KiB (1.70×)** | 50.2 KiB |
| vocab | 248,320 | **100,352** | 100,352 |
| max position | 262,144 | **1,048,576** | 1,048,576 |
| tensor naming | needs de-nesting patch | **clean** | clean |
| `gating` | `attn_output_gate=True` | **`per-head`**, `g_proj [72,1536]` | same |
| router | softmax | **sigmoid**, `norm_topk_prob`, bias, scale 2.5 | same |
| prefix caching | **structurally off** | **on** | on |

Per-expert bytes are unchanged (`3 × 3072 × 1024 × 0.5625 B = 5.0625 MiB`), so
the SBEXP001 per-expert record layout is identical — only the **row count**
(48→47) and **top_k** (8→10) move.

### Projected configuration outcomes

`[ESTIMATE]` — extrapolated from the measured 99B anchor (L=54 → 23.056 GiB
CUDA weights, 3.69 GiB KV, 131,072 tokens, 4.00× @32K) on the measured
0.2373-GiB-per-expert slope. Not measured for r20.

| @96 GB RAM, full mirror | L | mirror | CUDA | KV | tokens | C@32K |
|---|---:|---:|---:|---:|---:|---:|
| 99B | 34 | 171 | 18.31 G | 8.44 G | 299,657 | 9.1× |
| **r20** | **31** | **174** | **10.38 G** | **16.37 G** | **341,498** | **10.4×** |

At today's 64 GB, matching current KV: r20 reaches **73% mirror coverage** vs
the 99B's 35% (the 7.06 GiB dense saving buys local experts, which shrink the
host shadow), but **79,667 tokens vs 131,072** because L rises to 85.

**Decode does not improve.** `[ESTIMATE]` 46 gaps × 208 µs + 47 dispatches ×
80 µs (service scaled by 10/8 routes) + 1.24 ms non-sweep ≈ **14.6 ms** vs the
99B's measured 13.96 ms. The 208 µs inter-layer gap is 5090-side and neither
model nor RAM touches it.

---

## 3. What vLLM gives us free — verified, not assumed

| claim | evidence |
|---|---|
| Our OOT hook lands | `LagunaMoE` passes no `runner_cls`/`routed_experts_cls` (`laguna.py:209-226`) → factory defaults to `RoutedExperts`/`MoERunner` (`layer.py:369,403`) → `PluggableLayer.__new__` swaps in our hybrids (`custom_op.py:47-66`) |
| Per-head gating correct | `g_out = total_num_heads` = 72 (`laguna.py:334`), matches checkpoint `g_proj [72,1536]`; softplus in fp32, broadcast over `head_dim` (`laguna.py:443-452`) |
| Dense layer 0 handled | Both configs carry `mlp_only_layers`; vLLM builds MoE on exactly layers 1–47 (`laguna.py:514-538`), matching the checkpoint |
| Shared expert stays separate | `n_shared_experts` NOT passed; only the module is (`laguna.py:210`) — not fused into the routed grouped GEMM |
| Router scale is the runner's job | `apply_routed_scale_to_output=True` (`laguna.py:225`); combine order is scale-then-add (`moe_runner.py:740-775`) |
| Long context unaffected by the 512 window | `max_model_len` caps to the window only when `disable_sliding_window` (`config/model.py:2246-2269`, default false) |
| `DFlashLaguna` will never be selected | it validates *uniform* layer types (`laguna_dflash.py:45-60`) |
| Native `LagunaConfig` exists | `vllm/transformers_utils/configs/laguna.py` — remote code not required |
| SGLang offers nothing we need | first-class Laguna at `vendor/sglang/.../models/laguna.py:644`, but vLLM 0.27.1 already has the hybrid KV coordinator, `SlidingWindowSpec` window-bounded admission, SWA prefix-hit lookup and out-of-window reclamation |

**Zero vLLM-side changes required.** We stay on vLLM.

---

## 4. The central design: one coordinate contract

Everything else keys off this, so it landed first.

A compact 47-row bank stores **model layer 1 at row 0**. The native provider
indexes rows directly — `b70_provider.cpp:597-615` computes
`(layer * source_experts_per_layer + expert) * expert_bytes` and
`:1255-1258` accepts only `layer < g_layers`. Passing an absolute model index
straight through therefore reads the **neighbouring layer's experts**.

That failure is silent. No crash, no NaNs, and the bank's size check cannot
see it because the file is exactly the right length either way
(`b70_provider.cpp:224-227` checks header + `layers × experts × expert_bytes`).
It serves plausible, wrong tokens.

**Contract:**

- Model-facing state — placement, partition maps, telemetry, route
  statistics, CPU arena keys — stays in **absolute** transformer coordinates
  `0..47`, with layer 0 absent from routed execution.
- The bank is stored **compactly**, rows `0..46` = model layers `1..47`.
- `QualifiedModel.bank_layer_ids` names the absolute layers in row order;
  `bank_row_for_model_layer()` is the **single** checked translation.
- Every crossing into the native provider or a compact bank file goes through
  it. Nothing else subtracts.
- `routed_layer_ids` says which layers own a routed module **at all**. This is
  distinct from a routed layer that merely cannot offload: the 99B's FP8
  layers 32–39 are real `RoutedExperts` instances and return
  `is_routed_layer() == True`; Laguna's dense layer 0 has no module and
  returns False.

Empty tuples mean the legacy contract, so every Qwen path is unchanged —
verified identical for all four legacy shapes across every bank size 0..N.

---

## 5. Landed

| commit | contents | gate |
|---|---|---|
| `7ca6358e` | per-device trace dumps; all-lane dispatch error reporting | — |
| `78eb60bc` | r20 spec row; `routed_layer_ids`/`bank_layer_ids`; `bank_row_for_model_layer()`; `is_routed_layer()` | 37 checks |
| `6955b12f` | `b70_bank_covers` reads explicit ids, not a prefix | 41 checks |

`src/phase6/layer_topology_unit_test.py` — 41 GPU-free checks including both
boundaries (layer 1 → row 0, layer 47 → row 46), rejection of dense layer 0,
rejection of a 48-row bank against a 47-sparse-layer model, and the decisive
one: coverage **refuses** a valid 1..47 placement when evaluated under the old
prefix rule.

Regressions clean: `partition_unit_test` PASS, `prefill_mirror_unit_test`
PASS. `src/phase5/placement_test.py` fails on a missing
`src/phase1/expert_bank.bin` (the 35B artifact) — **pre-existing**, `HEAD`
references it unconditionally and only the 99B/int4 banks exist on this box.
It needs parameterising (Phase H).

---

## 6. Plan

Ordered by dependency. Each item: `path:line` → change → why → risk → kind.
**Acceptance** is what must be observably true before the phase is done.

### Phase A — Placement expresses "no routed module"

Blocks a boot today: a fractional policy on r20 assigns B70 owners to layer 0,
`b70_bank_covers` sees a B70 owner outside `{1..47}` and raises. That is
correct fail-closed behaviour, but it means r20 cannot place.

| item | change | risk | kind |
|---|---|---|---|
| `placement.py:123-167,257-307,327-356,438-440,525-540,618-638,688-713,748-837` | carry routed-layer presence in `Placement` and its manifest; keep the full 48-row absolute domain with layer 0 an explicitly documented unused sentinel; counts, capacity accounting, validation and serialisation must ignore it | medium | design |
| `partition.py:94-112,115-151,188-202,241-264` | retain a full `[48,205]` model-indexed device map, row 0 sentinel; validate `partition_routes` is invoked only for `routed_layer_ids` | medium | mechanical |

Current policies fabricate 205 CUDA "experts" for the dense layer. That does
not dispatch, but it makes `count`, `cuda_count`, manifests and
`DeviceCapacity` accounting **semantically false**, and a future loop would
attempt expert work for a layer that has none.

**Acceptance:** `fractional:2:0.2634` on an r20 `QualifiedModel` yields zero
B70/CPU owners in layer 0 and exactly 151 remote owners in each of layers
1–47; `validate_placement` passes; the manifest round-trips through
`from_dict`; `b70_bank_covers` returns True; new checks in
`layer_topology_unit_test.py`.

### Phase B — Build the r20 bank

The 99B bank **cannot** be reused: different weights, and 48 rows against 47
sparse layers.

| item | change | risk | kind |
|---|---|---|---|
| `extract_experts.py:149-181` | accept an ordered source-layer list `[1..47]`; the guard at `:172-177` rejecting a run not starting at 0 is **load-bearing** — relax it only together with the `bank_layer_ids` contract | high | design |
| `extract_experts.py:184-206,209-258` | discover and preserve the checkpoint's expert-key root; today `model.language_model.layers...` is hardcoded, Laguna is `model.layers.<L>.mlp.experts.<E>...` | high | mechanical |
| `extract_experts.py:59-75,95-108,294-305` | keep config-derived hidden/intermediate/expert geometry; write compact rows in declared source order | medium | mechanical |

SBEXP001 encodes **no top_k** (`b70_provider.cpp:167-179`), so no format bump
and no version change. Output: `src/phase1/expert_bank_jota_118b_r20.bin`,
47 rows × 205 experts, ~47.63 GiB.

**Acceptance:** file size exactly `header + 47 × 205 × expert_bytes`; Phase C
validator passes byte-exact on source layers 1, a middle layer, and 47.

### Phase C — Byte-exact bank validator through the map

| item | change | risk | kind |
|---|---|---|---|
| `validate_99b_bank.py:34-36,52-94,105-157` | generalise to an SBEXP001 validator (or add a Laguna profile); resolve clean vs nested key roots; translate every sampled row through `[1..47]`; explicitly cover source layers **1 and 47** | high | mechanical |

The existing size check only proves header self-consistency. A bank shifted by
one layer **passes it** and serves plausible garbage. This validator is the
only thing that can catch that, so it must exist before the first boot, not
after.

**Acceptance:** passes on the Phase B artifact; **fails** on a deliberately
shifted copy (build one, run it, keep the evidence).

### Phase D — Provider runtime top-k, fail-closed ABI

`b70_provider.cpp:42` has `constexpr kTopK = 8` and `:664` rejects anything
else. All 13 production uses are sizing / validation / capability / runtime
propagation; **no algorithmic assumption exists**, and the Quixi NVFP4 and
int4 kernels already take a runtime `std::size_t top_k` with no template
parameter or compile-time loop bound. Mechanical, but the ABI is not.

| item | change | risk | kind |
|---|---|---|---|
| `b70_capi.h:68-71`, `b70_capi.cpp:319-339` | **versioned** load symbol (`sb_b70_load_v2`) or mandatory exported `sb_b70_abi_version()` checked before any load, carrying `top_k` | **high** | design |
| `b70_provider.cpp:42,662-668` | delete `kTopK`; validate nonzero, `uint32_t`-representable, `<= source_experts_per_layer` **after** geometry adoption (`:720-850`) | low | mechanical |
| `b70_provider.cpp:234-253,1076-1078,1186-1188` | `persistent_device_bytes` takes `top_k`; thread it into the three per-row terms | medium | mechanical |
| `b70_provider.cpp:994-1003,1118-1123` | checked-multiply overflow guard; size `ids`/`weights`/`scratch` from `config.top_k` | medium | mechanical |
| `b70_provider.cpp:1043-1047` | report `{config.top_k}` — keep it a **singleton**, allocations are fixed per loaded instance | low | mechanical |
| `b70_provider.cpp:1261-1281,1295-1340` | `route_elements = M * config.top_k`; pass width to int4 split, NVFP4 split, NVFP4 fused | medium | mechanical |
| `b70_capi.cpp:61-70,418-467` | validate poll-registration `topk` equals the loaded width; `PollLayer::topk` is otherwise dead and sizes only host registration | medium | mechanical |
| `b70_binding.py:68-78,163-170,192-196` | ctypes to the versioned entry; `top_k` on `B70ProviderClient.load` | high | mechanical |
| `routed_experts.py:406-455,2713-2724,2766-2781,2802-2805`, `b70_poller.py:352-370` | plumb `qualified_model.top_k` into `_get_b70_provider`/`get_b70_poller`; validate cached providers agree | high | mechanical |
| `b12x_prefill.py:38` | remove `TOP_K=8` — the only executable top-8 constant in the plugin outside legacy specs | low | mechanical |

**Why versioned, not appended:** on the SysV x86-64 ABI a stale `.so` silently
ignores an extra argument, keeps width 8 from `b70_capi.cpp:326-330`, and
processes 8 of 10 routes against `[M,10]` buffers. Silent wrong output.
**Do not** repurpose `generation` for this — that is placement identity,
compared per issue at `b70_provider.cpp:1252-1253`.

Buffer cost at `top_k=10, max_batch=256, hidden=3072, intermediate=1024`:
`ids` +2,048 B, `weights` +2,048 B, `scratch` 16→20 MiB — **+4.004 MiB/card**
against ~7.5 GiB free. Per-row `bytes_per_batch_row` 84,032 → 100,432 B.

Build: `source /opt/intel/oneapi/setvars.sh --force && make -C src/phase7 b70`
(target at `Makefile:37-49`; no Quixi rebuild needed).

**Acceptance:** the **99B still boots and serves correctly** through the new
ABI at `top_k=8` (this is the regression that matters); a deliberately stale
`.so` is rejected loudly rather than loading; `sb_b70_load_v2` reports
`supported_topk == {10}` for r20.

### Phase E — Route every crossing through the map

| item | change | risk | kind |
|---|---|---|---|
| `routed_experts.py:2143-2151` | graph registration → bank row | high | mechanical |
| `routed_experts.py:2719-2724` | synchronous eager dispatch → bank row | high | mechanical |
| `routed_experts.py:2775-2780` | asynchronous eager issue → bank row | high | mechanical |
| `routed_experts.py:1318-1339,1686-1753,3063-3067` | `ExpertBank.expert(row, expert)`; keep the CPU arena **absolute** | high | mechanical |
| `b70_poller.py:96-143,177-180` | the integer passed to native is a **bank row**; keep both coordinates in registration metadata; translate native trace rows back to model layers before writing diagnostics | medium | mechanical |
| `b70_binding.py:260-313` | rename/document `layer` as bank row; bounds-check against loaded bank layers before the C call | medium | mechanical |
| `marlin_prefill.py:128-158,162-225`, `b12x_prefill.py:132-147,161-206` | receive bank row; `prefetch(row+1)` then stays valid; make out-of-range prefetch a loud error, not a silent return | medium | mechanical |

Add sparse-layer membership checking immediately after `extract_layer_index`
rather than parsing architecture-specific prefixes.

**Acceptance:** 99B unchanged (identity map, verified by the existing gates);
an r20 trace shows layer 1 dispatching to row 0 and layer 47 to row 46;
per-device trace files report **model** layer ids, not rows.

### Phase F — Fail-closed guards

Cheap, and each one prevents a silent-wrong-output mode.

| item | guard | why |
|---|---|---|
| `routed_experts.py:985-1002,1284-1296` | require `quant_method.is_monolithic is False` | we override `forward_modular`; vLLM sends monolithic methods to inherited `forward_monolithic`, **bypassing offload entirely**. FlashInfer TRTLLM can prefer a monolithic class (`oracle/nvfp4.py:67-80,525-575`) |
| `routed_experts.py:989-1026` | assert global experts == 205, `top_k` == 10, routed intermediate == 1024 at TP=1; validate bank intermediate as well as hidden; validate incoming `topk_ids/topk_weights` width | placement maps use the qualified count; today only hidden size is compared |
| `routed_experts.py:2207-2214`, `runner.py:43-60` | assert the runner is modular and `apply_router_weight_on_input == False`; add **no** router maths | `norm_topk_prob` and scale 2.5 are already in vLLM's weights/runner output scaling. Re-normalising per tier is wrong; input-side weighting is not equivalent to provider output weighting through SwiGLU |
| `routed_experts.py:1071-1079,2095-2130,2266-2272` | resolve `_all_cuda_passthrough` from ownership once absolute `layer_index` is known, independent of `_b70_graph_mode`; assert the layer is routed | passthrough is fully established only during graph-mode poller registration; an all-CUDA sparse layer in eager mode can miss it |
| `routed_experts.py:2941-3013` | **fail** if offload+prefill runs without graph/Tier-3 | pre-existing and affects the 99B: prefill always masks B70 routes out of the CUDA partial but only adds the B70 partial when `_b70_graph_mode`. Do **not** merely relax `:2996` — the graph issue/take helpers depend on buffers allocated under graph mode |
| `config.py` | reject `moe_apply_router_weight_on_input=true` | both candidate configs set it false; provider kernels consume output weights |

### Phase G — Real-silicon qualification

In order. Each step's output is recorded in `docs/kill-bench.md`, not here.

1. **First boot**, `--moe-backend cutlass`, `VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS`
   as in `serve_99b_dual.sh` (FlashInfer's trtllm-gen fused-moe tuner wedges
   this GPU: 25 min at 100% with no completion, or a `CUDA misaligned
   address` fault — Bench 16).
2. **Correctness gate**: greedy tokens + finite logprobs on two prompts,
   both cards dispatching, per-device doorbell traces present and
   independent.
3. **Dual split**: confirm 151 remote experts land 76/75 with disjoint
   per-card resident lists on the monolithic bank.
4. **Prefix cache on the real path** — the number the whole decision rests
   on. Repeated 8K prefix, cold vs warm, with `enable_prefix_caching` and hit
   rate read from the server's own metrics.
5. **Prefill ladder** (`benchmarks/b70_prefill_cost.py`) and **decode ITL**
   (`benchmarks/b70_itl_probe.py`, now parameterised for URL/model/dual
   clocks). Compare against the 99B's measured 15.48 s → 12.21 s 8.5K TTFT
   and 13.96 ms ITL.
6. **Quality.** The open risk. r20 is 20% REAP-pruned from 118B with a
   512-token window; the gate only proved it says "42". Needs a real eval
   before anything ships.

### Phase H — Tests

| item | change |
|---|---|
| `src/phase6/partition_integration_test.py:24-49,72-111` | add r20 as a first-class real-model profile alongside Qwen; assert 47 `HybridRoutedExperts` at absolute layers 1–47, **no** module or hook at layer 0, width-10 routing; exercise a real sparse all-CUDA passthrough layer |
| `src/phase6/shadow_validation_test.py:25-61`, `hybrid_execution_test.py:23-58` | same profile treatment; derive the split from the profile rather than fixed `split:128` |
| `src/phase5/placement_test.py:4-14,45,86-198` | parameterise Qwen and r20; model bank capability as explicit absolute ids; drop the 40×256 / capable-0-31 / FP8-32-39 assertions from the Laguna case; **fix the unconditional `expert_bank.bin` dependency that fails today** |
| `src/phase6/partition_unit_test.py:35-36,54-166` | add 48×205, routed layers 1–47, width-10 cases; keep existing top-8 synthetic cases as regressions |
| `src/phase6/prefill_mirror_unit_test.py:47-217` | 47 routed/bank layers for Laguna arena budgeting; keep model layer ids separate from compact row count |
| `tests/test_route_locality.py`, `test_route_topology.py` | add top-k-10 cases for 205; keep top-8 tests |

### Phase I — Deferred, explicitly

Not blockers. Listed so they are not mistaken for oversights.

- **B12x prefill** (`b12x_prefill.py:44-206`, `build_b12x_bank.py:39-212`,
  `b12x_bank_format.py:34-149`): needs top-k 10, clean keys, compact rows,
  and a **format-aware** scale conversion — Laguna stores `weight_packed` /
  `weight_scale` / `weight_global_scale` while the builder expects ModelOpt's
  `weight` / `weight_scale` / `weight_scale_2`. Not a suffix rename. Also
  `expert_id_base` + count cannot express a non-contiguous resident set.
  Keep `SHOOTING_BRAKE_PREFILL_B12X=0`.
- **int4 / Marlin tooling for Laguna** (`extract_experts_int4.py`,
  `build_marlin_bank.py`): only if an int4 Laguna artifact is ever wanted.
  Requantisation carries real numerical risk and NVFP4 needs none.
- **`prefill_mirror.py:79-131,135-179`**: arena APIs multiply by model depth
  48; Laguna needs 47. Helper-only today (no production call site), so it is
  a false-rejection risk when wired in, not a boot blocker.
- **Telemetry/diagnostic wording** (`telemetry.py:332-342` and the Qwen-era
  comments listed in the audit): cosmetic, do last.
- **Scope the nested-name monkeypatches** (`__init__.py:59-124`): verified
  **no-ops** for Laguna (they rewrite only `model.language_model.` prefixes,
  and Laguna's ignore list starts `model.layers.1.mlp.gate`), but they are
  unnecessary process-global side effects and should be gated per spec.
- **r15**: another spec row plus a 218-expert bank and its placement
  artifacts, after r20 works.
- **GuideLLM**: the acceptance gate, on the winner, once. Not before.

---

## 7. Hard prohibitions

Each of these has a specific failure mode behind it.

1. **Do not reuse `expert_bank_99b.bin`.** Different weights, 48 rows.
2. **Do not build a 48-row bank with a dummy layer 0.** Costs ~517 MiB/card
   of never-read resident weights and ~1.01 GiB on disk, and entrenches the
   prefix assumption we just removed.
3. **Do not delete `extract_experts.py:172-177`** without landing the
   `bank_layer_ids` contract first. Alone, it silently shifts every layer.
4. **Do not append `top_k` to the existing `sb_b70_load` symbol.** A stale
   `.so` ignores it and processes 8 of 10 routes.
5. **Do not bump `generation` for ABI compatibility.** It is placement
   identity.
6. **Do not add sigmoid / top-k / renormalisation / scaling code to the
   plugin.** vLLM selects experts before `forward_modular`; we receive final
   `topk_ids`/`topk_weights`, mask disjoint owners **without** renormalising,
   send the same weights remotely, and sum partials.
7. **Do not relax the graph-mode condition at `routed_experts.py:2996`
   alone.** The graph issue/take helpers depend on graph-mode buffers.
8. **Do not enable B12x or `SHOOTING_BRAKE_PREFILL_MARLIN` for Laguna** until
   Phase I lands. The NVFP4 bank is not a Marlin int4 bank.
9. **Do not quote a Laguna performance number** until Phase G measures it on
   the B70s. Everything in §2's projection table is `[ESTIMATE]`.
10. **Do not switch the MoE backend.** `--moe-backend cutlass` /
    `VLLM_CUTLASS` is the only qualified NVFP4 path on sm_120; it selected
    correctly for r20 in the Phase-0 gate.

---

## 8. Open questions

| question | how it gets answered | blocks |
|---|---|---|
| Does r20's output **quality** hold up at 20% REAP + SWA(512)? | real eval after Phase G boot | shipping |
| Real-path prefix-cache speedup (142× is on a 4×-slow arm) | Phase G step 4 | the whole decision's magnitude, not its sign |
| Does the 512 window need vLLM's inclusive/exclusive convention checked? SGLang converts to `sliding_window - 1` (`laguna.py:424-427`); vLLM passes 512 directly (`laguna.py:298`) | compare logprobs against HF reference on a >512-token prompt | correctness, low probability |
| Does `CompressedTensorsW4A4Nvfp4MoEMethod` get selected for Laguna (expected) or something else? | log the class at Phase G boot | Phase F guard shape |
| Decode ITL: does 47 layers × +22.4% routes land at ~14.6 ms? | Phase G step 5 | nothing; it is a number, not a gate |

---

## 9. Progress log

| date | item | state |
|---|---|---|
| 2026-08-20 | Phase 0: architecture gate via CPU offload — loads, correct tokens, `prefix_caching=True`, **142.198×** cold/warm | **done**, `benchmarks/laguna_gate.py`, `/tmp/laguna_gate.json` |
| 2026-08-20 | vLLM + SGLang source audit (4 parallel agents) | **done**, §3 |
| 2026-08-20 | Coordinate contract + r20 spec row (`78eb60bc`) | **done**, 37 checks |
| 2026-08-20 | Bank coverage reads explicit ids (`6955b12f`) | **done**, 41 checks |
| 2026-08-20 | Phase A — Placement expresses "no routed module" | in progress |
