# Sparse-memory research

[Executable model](../models/sparse_memory.ryu) · [Traceability map](../maps/sparse_memory.json)

## Responsibility and branch map

This directory is a standalone learned-memory research branch, not part of the current production serving path. It asks, breadth first, whether a frozen Qwen decoder can use an attached product-key memory, where a large value table could live, what its retrieval costs, whether learned rows store controlled knowledge without damaging retained capability, and whether a small model's bank can absorb a larger same-family teacher's behavior.

The actors are:

1. **Frozen decoder.** Supplies hidden states and logits. Its parameters remain frozen in E5–E7.
2. **Product-key selector.** Projects and normalizes a hidden state, scores two sub-key codebooks, and returns `top_k` Cartesian row indices and weights.
3. **Value-table owner.** Holds the large row table on the fast CUDA device, a peer XPU, or host DRAM. It gathers and reduces locally so the boundary carries indices/weights outward and one reduced vector back.
4. **Additive attachment.** Runs beside a whole decoder block and adds a learned residual delta through a forward hook; it never replaces the block FFN.
5. **Experiment drivers.** E0–E2 establish staleness and placement/kernel feasibility, E4 measures a no-memory SLO baseline, E5 tests retrofit learning and row utilization, E6 tests controlled knowledge storage and retention, and E7 tests teacher distillation. There is no assigned E3 implementation.
6. **Evaluation-only dependencies.** Hugging Face models/datasets, PyTorch, Triton, and external benchmark datasets are dependencies. Their implementations are not first-party behavior modeled here.

The [model](../models/sparse_memory.ryu) intentionally traverses this branch in that order. Its bounded state computes gate outcomes from guarded counts and error/CE/KL units; fixture values establish reachability only and are not reported as observations or prior result summaries.

## Interfaces and state

| Interface/state | Owner | Implemented behavior |
|---|---|---|
| `PKMConfig` | [`pkm.py`](../../../experiments/sparse-memory/pkm.py) | Requires `n_rows` to be a perfect square and `d_query` to be even; derives `c = sqrt(n_rows)` and half-query width. |
| `select(query, subkeys, cfg)` | `pkm.py` | Scores each query half against its codebook, keeps `k` candidates per half, forms `k²` pair sums, keeps the best `k`, maps pairs to `row1*c + row2`, then softmaxes selected scores in fp32. It never materializes all `c²` scores. |
| `gather(indices, weights, values)` | `pkm.py` | Fetches selected rows and performs the weighted reduction. Heads remain separate in this reference helper. |
| `ProductKeyMemory` | [`memory_layer.py`](../../../experiments/sparse-memory/memory_layer.py) | Owns query projection/normalization, trainable subkeys, shared values, readout, alpha, optional gate, routing census, optional addressing loss, optional selection bias, and checkpoint-recompute hygiene. |
| `attach_memory` | `memory_layer.py` | Registers hooks at chosen decoder layers. Each hook reads layer input and adds the memory delta to layer output, preserving tuple tails. |
| `Attachment.remove` | `memory_layer.py` | Removes hook handles and clears ownership. |
| experiment JSON/checkpoint output | E0–E7 drivers | Records configurations and observations; E6/E7 also save memory module state. These artifacts are evidence products, not proof of generality. |

The main retrieval formula implemented by `ProductKeyMemory.forward` is

\[
\Delta h = \alpha\,g(h)\,W_{up}\left(\sum_{head}\sum_{j=1}^{k} w_{head,j}V[r_{head,j}]\right),
\qquad h_{out}=h_{block}+\Delta h.
\]

When the gate is disabled, the multiplicative gate is absent (equivalently one). The table is shared across heads. `embedding_bag` performs a weighted sum per token/head and the implementation sums heads before `W_up`.

## Retrieval, update, and placement mechanics

### Address selection

The projection has shape `d_model -> n_heads*d_query`. The default `query_norm="batch"` applies `BatchNorm1d(n_heads*d_query)` **before** splitting heads. Layer normalization and no normalization remain explicit ablations. Optional key normalization divides each sub-key by a clamped norm.

For dot-product mode, each half score is

\[
s_{h,a,i}=q_{h,a}\cdot k_{h,a,i},\quad a\in\{0,1\}.
\]

For inverse-distance mode, the score is `-log(1e-3 + ||q-k||²)`. The code then selects top-k subkeys for each half, computes all `k²` sums, selects the best k pairs, constructs Cartesian row IDs, and softmaxes their **unbiased** selected scores. If `bias_gamma != 0`, selection uses `score + sel_bias`, but returned weights are gathered from original scores. `balance_step` compares per-subkey use with `1/c` and shifts each bias by a fixed signed gamma outside the optimizer.

The alternative addressing objective recomputes scores from a detached query so its gradient reaches keys, not the query projection or batch normalization. Per half, it masks to each token's top-k, averages softmax probabilities across tokens/heads, and minimizes negative marginal entropy. This encourages uniform marginal sub-key use without forcing an individual query to be uniform.

### Value updates and initialization

Values receive sparse gradients only for touched rows. Dense parameters use AdamW in the experiment drivers; values use SparseAdam by default, with SGD exposed in E5. The source explicitly says neither reproduces Meta's optimizer exactly.

`value_init="random"` is the default and `alpha` starts small. `value_init="zeros"` makes the branch exactly zero initially while still allowing value gradient through nonzero `W_up`; zeroing `W_up` instead would starve values and routing. This distinction is encoded in source comments and diagnostics, but the formal model abstracts optimizer arithmetic.

Gradient-checkpoint recomputation is distinguished by grad-enabled state plus a per-step tag. The census counts the original forward once. During recomputation, BatchNorm still uses batch statistics but its running-stat update and batch counter are restored. Evaluation helpers explicitly set memory modules to evaluation mode because the attachment's `ModuleList` is not a child of the wrapped decoder.

### Placement and quantized gather

The intended boundary is asymmetric: the selector emits k row IDs and k weights; the table owner reduces k rows; only one value-width vector returns.

* [`e1_xpu_gather.py`](../../../experiments/sparse-memory/e1_xpu_gather.py) prices `index_select` plus `bmm` on XPU/CUDA and reports local fetch bytes versus raw-row and reduced-vector PCIe bytes.
* [`e1b_fused_gather.py`](../../../experiments/sparse-memory/e1b_fused_gather.py) compares the materializing reference, chunked fallback, and floating-point `embedding_bag`. Int8 omits `embedding_bag` and uses the materializing/chunked paths.
* [`e1c_triton_gather.py`](../../../experiments/sparse-memory/e1c_triton_gather.py) implements int8 gather, optional per-row scale, fp32 accumulation/output, and rejects a candidate if relative error is at least `1e-3`.
* [`e1d_triton_tune.py`](../../../experiments/sparse-memory/e1d_triton_tune.py) sweeps block width, fp32 versus bf16 output, and warp count; unsupported and numerically wrong candidates are skipped. It compares rows/second, not merely bytes/second, with an E1b constant.
* [`e1e_triton_int4.py`](../../../experiments/sparse-memory/e1e_triton_int4.py) packs signed values offset by eight into low/high nibbles, reconstructs both planes, applies per-row scale and weight, accumulates fp32, and emits bf16. Candidates above `5e-2` relative error are discarded.
* [`e2_host_tier.py`](../../../experiments/sparse-memory/e2_host_tier.py) compares bf16 host-table alternatives. “Reduce at DRAM” uses CPU `embedding_bag` then transfers `M*d` values. “Reduce at GPU” materializes and transfers `M*k*d` values. The reported second total still includes the same CPU gather timing, because the source prices gather cost in both branches.

A potential CPU implementation should therefore be compared against `F.embedding_bag` with identical bf16 rows, indices, and weights. Compare output error to an explicit weighted sum; compare pinned and pageable transfer separately; and report row fetch bandwidth plus end-to-end gather-and-copy latency. A CPU kernel that transfers raw rows is not equivalent to the reduce-at-DRAM contract.

## Experiment progression and gates

### E0 — query staleness proxy

[`e0_query_staleness.py`](../../../experiments/sparse-memory/e0_query_staleness.py) collects hidden states from fixed prompts. At selected depths it compares exact row selections with (a) the previous token's hidden state and (b) the same token at roughly half the depth. It runs random subkeys and Lloyd-fitted centroids. The source explicitly calls this a proxy because the codebooks are not trained memory keys. It also checks `gather` against an explicit weighted sum. An overlap does not establish quality or a hidden PCIe round trip by itself.

### E1 — table-device gather

E1 first prices the simple materializing implementation. E1b exposes fusion and dtype branches. E1c introduces a fused int8 Triton kernel with a strict reference gate. E1d treats rows/second as the selection criterion while sweeping launch configuration. E1e tests packed int4 and reports both read-only and read-plus-write rates against source constants. These are experimental kernels, not integration into the production serving runtime.

### E2 — host tier

E2 tests row-granularity host placement, pinned by default. The key mode branch is where reduction occurs. Host reduction cuts transfer elements by k, but CPU gather work remains and must be measured. No source here implements overlap with a live decoder.

### E4 — baseline SLO

[`e4_baseline_slo.py`](../../../experiments/sparse-memory/e4_baseline_slo.py) sends streaming completion requests to a fixed local endpoint/model. Unique randomized prefixes defeat prefix caching. It records TTFT, median/p99 inter-token gaps, TPOT, per-stream throughput, aggregate throughput, and errors; cells over a prompt-token budget are skipped. Its `project` function applies hard-coded memory-cost constants to concurrency-one observations. The projection is arithmetic printed after measurement, not a memory-enabled served run.

### E5 — frozen-backbone retrofit

[`e5_retrofit_ladder.py`](../../../experiments/sparse-memory/e5_retrofit_ladder.py) freezes each backbone, attaches memory at block-end layers in the 45–80% depth band, and trains only memory parameters on a fixed Pile-derived stream. The evaluation split is carved before training from a fixed corpus and hashed; insufficient train batches assert rather than silently shortening the run. The 9B rung can enable gradient checkpointing; CUDA OOM is recorded and skipped.

The driver measures backbone-only CE by zeroing alpha, at-init CE with the branch enabled, then trained CE. The gate is not loss alone: it reports top-k/top-1 coverage, entropy-equivalent effective rows, maximum row share, gradients, routing probes, and equal-sized census windows. Clean evaluation routing is measured separately. `query_norm`, key normalization, initialization, scoring mode, balancing mode, value optimizer, warmup, and site count remain experimental branches.

A source-level historical contradiction is preserved: E5's module docstring says fixed `d_value=512` keeps memory parameters identical across rungs, while the later `memory_layer.py` findings and E6 scaled harness say this under-dimensions larger backbones and recommend `d_value=d_model`/`d_query=d_model/2`. E5 CLI defaults are still fixed at 512. The code, not the older motivation, determines a run.

### E6 — controlled knowledge injection

[`e6_knowledge_injection.py`](../../../experiments/sparse-memory/e6_knowledge_injection.py) generates invented name/attribute facts from closed vocabularies and splits them into injected and matched controls. Only injected facts are trained, through five paraphrases. Candidate answers are scored by mean continuation log-probability. The driver records:

* backbone-only, untrained-memory, and trained-memory conditions;
* acquisition = trained injected accuracy minus at-init injected accuracy;
* control drift = trained control accuracy minus at-init control accuracy;
* storage effect = acquisition minus control drift;
* held-out real-text CE retention deltas;
* a held-out surface template;
* a fact-in-prompt ceiling measured with the branch disabled; and
* fraction of achievable ceiling recovered.

The optional per-token gate is `sigmoid(mean(Linear(LayerNorm(h))))`. The source records that fact-only training led it to saturate open; the proposed mixed CE/KL objective is not implemented in E6. Thus “gate enabled” is a mode, not evidence of retention repair.

The shell harnesses are research recipes, not result truth:

* [`run_e6_ladder.sh`](../../../experiments/sparse-memory/run_e6_ladder.sh) runs the model ladder sequentially and waits for another tagged run to release the GPU.
* [`run_e6_scaled.sh`](../../../experiments/sparse-memory/run_e6_scaled.sh) keeps 65,536 rows but scales row and query widths with the backbone.
* [`run_e6_gate.sh`](../../../experiments/sparse-memory/run_e6_gate.sh) enables the gate for scaled 9B/4B/0.8B configurations.
* [`run_e6_seeds.sh`](../../../experiments/sparse-memory/run_e6_seeds.sh) fixes fact data and varies 9B memory initialization/batch-order seed.

Their comments contain prior observations and hypotheses. This chapter does not promote them into newly verified numerics.

### E7 — teacher distillation

[`e7_teacher_distill.py`](../../../experiments/sparse-memory/e7_teacher_distill.py) loads a frozen student and a frozen same-tokenizer teacher. Local cached Codex, UltraChat, or Pile parquet is packed into fixed token blocks. The only trained objects are attached memory modules. The objective is token-mean forward KL,

\[
\frac1N\sum_t \sum_v p_T(v\mid x_{\le t})\left(\log p_T(v\mid x_{\le t})-\log p_S(v\mid x_{\le t})\right),
\]

computed in token chunks to bound fp32 logit memory, plus the optional key-only addressing loss. Evaluation toggles alpha to compare the identical student alone and with bank, then separately evaluates teacher CE. It reports KL reduction, top-1 agreement gain, and

\[
\text{gap\_closed}=\frac{CE_{student}-CE_{student+bank}}{CE_{student}-CE_{teacher}},
\]

unless the denominator is effectively zero. Missing cached parquet and insufficient train batches are hard failures. E7 saves module state/config/sites and removes hooks afterward. This is a scaled-down research test, not evidence for the stated 27B/99B target until that configuration is actually run.

## Capability retention

[`capability_eval.py`](../../../experiments/sparse-memory/capability_eval.py) is evaluation-only. It adapts MMLU, ARC, HellaSwag, WinoGrande, and PIQA to continuation scoring, and optionally evaluates GSM8K by greedy generation and final-number extraction. `paired_retention` disables row census and toggles alpha to isolate memory-off/on behavior in the same model. It reports per-suite deltas, mean delta, and worst delta.

Tier-3 agentic coding benchmarks are explicitly **not measured** by this module. External benchmark checkouts and dataset implementations are dependencies, not first-party subsystem sources.

## Executable scenarios and properties

The model contains reachable witnesses for:

* `peerInt4RetrievalTest`: peer-resident packed retrieval, locally accepted reference error, and additive injection;
* `hostReduceBeforeTransferTest`: host DRAM reduction before transfer;
* `invalidShapeStopsSelectionTest`: a non-square row count enters an observed failure and selection is disabled;
* `hostCannotUseDeviceGatherTest`: placement preconditions reject the device-gather action for host ownership;
* `kernelMismatchIsRejectedTest`: an error above tolerance rejects the E1 candidate;
* `storageRetentionRemainSeparateTest`: the E6 formula computes positive storage effect from injected acquisition minus matched-control drift while a positive held-out CE delta fails retention;
* `distillationProgressionTest`: the entire E0→E1→E2→E4→E5→E6→E7 branch computes a successful outcome from lower bank CE and lower teacher KL;
* `noDistillationImprovementTest`: worse bank CE/KL completes measurement without claiming gap reduction; and
* `inverseDistanceAblationTest`: the non-default score mode is reachable without presenting it as preferred.

The invariant keeps reduction after selection, injection after reduction, shape failure terminal/observed, knowledge tied to the computed storage formula, retention tied to held-out CE delta, utilization tied to bounded row counts, and gap reduction tied to lower bank CE and KL. The half-coverage utilization predicate is an explicit formal abstraction rather than a source acceptance threshold. Atomic tensor and training operations preserve ordering and externally visible mode/error distinctions while deliberately omitting real-valued tensor state.

## Limits and status

* No assigned source installs this memory in the production plugin or proves production SLOs.
* E0 uses proxy codebooks; E1/E2 are synthetic gather benchmarks; E4 is a baseline plus projection; E5–E7 are experimental training/evaluation drivers.
* Numerical comments and hard-coded comparison constants are provenance inside the source, not rerun evidence here.
* GPU/XPU timing, training, model downloads, datasets, served endpoints, and checkpoint results are outside the formal model.
* The model treats successful numerical-reference checks qualitatively. It does not claim floating-point equivalence across devices or quantizations.
* Dataset/model licenses, dataset correctness, PyTorch/Triton kernels, and external benchmark harnesses remain dependency boundaries.
