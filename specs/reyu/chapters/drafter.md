# Drafter research lifecycle

The [executable drafter model](../models/drafter.ryu) describes a separate research workflow around the serving system. It does **not** put speculative decoding into the ordinary production recipe, model the internals of SpecForge, or claim that a passing offline checkpoint has been deployed. The checked-in workflow collects on-policy target outputs, replays the exact prompt-plus-completion sequence through a hidden-state capture hook, validates and stages those features, converts a Poolside Laguna warm start, invokes external training, reconstructs the native checkpoint layout, and evaluates a baseline/candidate pair. The overnight supervisor has a production fallback, but its health-only early return can preserve a still-running candidate instead of restoring the ordinary recipe.

## 1. High-level branch

```text
healthy r15 target server
        |
        v
prompt synthesis --raw /v1/completions--> append-only on-policy JSONL
        |
        v
capture-only server + unique request salt
        |
        +--> exact rendered prompt tokens + retokenized raw completion
        +--> target hidden hook: bf16 [T, 6, 3072]
        v
adapter validation and no-copy SpecForge feature links
        |                         Poolside Laguna native checkpoint
        |                                      |
        |                           schema/key/shape validation
        |                           fused QKV -> q/k/v warm start
        |                                      |
        +---------------------> one-GPU offline DFlash training
                                               |
                                   q/k/v -> fused native QKV
                                               |
                             native Laguna checkpoint export
                                               |
                  no-spec baseline <-------> DFlash candidate
                                               |
                  quality + acceptance + effective-decode gates
                                               |
                     research result; production restore attempt
```

The actors and boundaries are:

* **Target r15 server:** external OpenAI-compatible `/tokenize`, `/v1/completions`, `/v1/models`, and health endpoints. Data generation deliberately uses raw completions because this server's non-streaming chat response discards reasoning content.
* **Hidden-capture plugin hook:** an implementation dependency outside this ownership set. It converts the request identity into an atomic `.pt` feature record. The driver waits for the final filename; it does not write tensor payloads itself.
* **Adapter and converters:** CPU-executable first-party validation/conversion code. They own exact feature and checkpoint schemas, not training mathematics.
* **SpecForge and Transformers/PyTorch:** external framework contracts. SpecForge reads the linked features, provides the DFlash objective and CLI, and returns `training_state.pt`; PyTorch/Transformers implement tensor operators, attention backends, optimizers and checkpoint loading.
* **Single RTX 5090 training job:** an experimental, exclusive hardware workload. The pilot is offline, one node, one process, and `NO_SHARD`; this is not serving.
* **Acceptance services:** the same bounded-context baseline and candidate are booted serially. The gate consumes HTTP evidence plus the external decode-ladder output.
* **Overnight supervisor:** owns idempotent stage markers, service stop/start, disk/tmpfs guards, and a health-based fallback policy. A stage marker means the script completed that stage, not mathematical proof of correctness; a healthy endpoint is not configuration identity.

## 2. State and operation map

| Stage | Inputs and identity | Observable success | Failure or edge branch | Model action |
|---|---|---|---|---|
| Corpus | Seeded prompt id `pilot-<seed>-<index>`, kind, prompt, raw completion, token counts | Append-only JSONL reaches at least 500 records for the overnight pilot | Existing ids resume; malformed old lines are ignored by datagen; an individual request retries up to 1000 times; datagen can finish with failed items, while the supervisor independently refuses fewer than 500 records | `corpusReady`, `rejectTinyCorpus` |
| Capture | First occurrence of each corpus id; rendered prompt tokens plus completion tokens; salted request id carries corpus id, response start and length | One final feature filename exists for every deduplicated record | Duplicate ids are accounted and skipped; empty completions, bad HTTP responses, write timeout, or exhausted retries leave a missing record and a nonzero capture result | `captureComplete`, `rejectMissingCapture` |
| Adapter | `shooting_brake_dflash_v1` mapping | Valid records are linked as `.ckpt`; manifest is atomically replaced | Invalid records fail closed. Records whose first 3072 tokens contain fewer than two supervised tokens are valid but skipped. The adapter itself can emit a zero-record manifest; training is where an empty usable set becomes unusable | `stageValidCaptures`, `rejectCaptureContract` |
| Warm start | Poolside native config plus exact 69-key native tensor set | Training config and exact 81-key q/k/v tensor set are atomically emitted in shards | Missing/duplicate/unexpected keys, wrong config, incomplete shards, or wrong fused-QKV shape raise an error | `warmstartConverted`, `rejectWarmstart` |
| Training | Linked features, converted warm start or one complete latest checkpoint, pinned pilot config | External SpecForge returns a DFlash `training_state.pt` | Missing environment, live vLLM, multiple visible GPUs, a competing >1 GiB CUDA consumer, empty usable data, OOM, or framework failure prevents a trained state | `trainFresh`, `trainResume`, training rejection actions |
| Export | `strategy == "dflash"`, exact 81-key draft state, validated reference config and 69 native keys | q/k/v are shape-checked and fused; output key set equals reference; native config is replaced atomically | Multiple checkpoint lineages, no state, schema mismatch, missing/extra tensor, or projection shape mismatch rejects export | `exportNativeCheckpoint`, `rejectExport` |
| Acceptance | 120 aligned top-5 distributions and six ordered ladder rows | All four quality/acceptance/latency gates pass | Bad schemas or counters reject evidence; a well-formed but below-threshold result rejects the candidate | `acceptEvidence`, `rejectBadEvidence`, `rejectThresholds` |
| Restore/fallback | `/health` and systemd user units | Any healthy endpoint is preserved; only an absent endpoint triggers an ordinary production boot | A healthy candidate suppresses ordinary restoration; a failed production boot is logged for operator action | `preserveHealthyCandidate`, `restoreProduction`, `rejectRestore`; separate candidate/production health observations |

The model makes each row atomic so that breadth-first lifecycle navigation remains readable. Real datagen/capture workers interleave under locks; HTTP calls, feature-file writes, shard replacement, checkpoint writes, service boots and GPU work are not single transactions with one another.

## 3. Data generation and target capture

[`experiments/drafter_datagen.py`](../../../experiments/drafter_datagen.py) builds a deterministic prompt list from three sources. Half is intended to be snippets from repository Python/C++/header files between 2 KB and 200 KB, read as up to 120 lines and clipped to 6000 characters. Thirty percent is sampled from the optional local `~/sb_corpus_big.txt`. The remainder uses fixed agentic prompts; if prose is absent, agentic prompts fill the shortfall. The seeded list is shuffled before stable ids are assigned.

`generate()` scans an existing output and skips ids it can decode, making the file append-only and resumable. Each worker observes `/tmp/sb_datagen.pause` both before entering retry and again while retrying. It asks `/tokenize` to render a one-turn user chat, then submits those exact ids to raw `/v1/completions` with `add_special_tokens=false`, temperature 0.8 and top-p 0.95 by default. The saved `raw_completion` therefore includes reasoning markup that the chat API would omit. Writes and counters share a lock. Server outages back off from 5 to 60 seconds, but exceptions are otherwise intentionally broad. After 1000 failed attempts the item increments `fail`; the program reports failures but does not itself return a failing status. The overnight file-count gate is the downstream protection.

[`experiments/drafter_capture.py`](../../../experiments/drafter_capture.py) validates corpus JSON structure, treats the first duplicate id as authoritative, and prevents duplicate training weight. `_tokenize_record()` repeats the chat rendering and separately tokenizes the raw completion without special tokens. This text round trip can differ from originally sampled token ids at merge boundaries; the source explicitly accepts that difference. The returned sequence is `[prompt tokens | completion tokens]`, and `response_start` is the prompt length.

For each pending id, `capture()` derives the plugin request id, submits the entire fixed sequence as prompt plus a single generated token, and supplies the request id as `cache_salt`. The salt is a correctness requirement: a prefix-cache hit would skip target execution and therefore omit hidden rows. The generated token is not part of the feature capture. Capture uses its own `/tmp/sb_capture.pause`, intentionally distinct from datagen pause. The hook's atomic final filename is the completion signal. Missing files after all retries determine exit status.

## 4. Feature validation and masks

[`experiments/drafter_train/adapter.py`](../../../experiments/drafter_train/adapter.py) memory-maps one capture at a time and releases the record before reading the next. `validate_record()` requires:

* all eight keys: `id`, `format`, `input_ids`, `loss_mask`, `response_start`, `hidden_states`, `hidden_states_by_layer`, and `target_layer_ids`;
* nonempty string identity and format `shooting_brake_dflash_v1`;
* `input_ids`: `int32 [T]`;
* `loss_mask`: `bool [T]`, false before `response_start` and true for the complete suffix;
* exact target layer ids `(1, 10, 19, 29, 38, 47)` as `int64 [6]`;
* layer form `bfloat16 [T, 6, 3072]` and flat form `bfloat16 [T, 18432]`;
* identical data pointer and storage offset plus contiguity for both hidden views, so flattening introduced no tensor copy;
* at least two consecutive supervised tokens and finite hidden values, checked in 256-row chunks.

The training window is 3072 tokens. SpecForge truncates before computing loss, so `stage_capture()` skips, rather than rejects, a structurally valid record if fewer than two supervised tokens survive that prefix. It hard-links each usable `.pt` record to `.ckpt`, falling back to a symlink, and refuses an existing destination that points elsewhere. The manifest records total and supervised tokens, byte count, skipped identities, layer ids, hidden size, and paths; a temporary file is renamed into place. The adapter does not copy the dataset tensors and does not reject an all-skipped result by itself.

[`experiments/drafter_train/tests/test_adapter.py`](../../../experiments/drafter_train/tests/test_adapter.py) is the focused contract harness. Its cases specify sorted linked records, SpecForge `OfflineManifestReader` compatibility, normalized batch shapes `[1,T]` and `[1,T,18432]`, shared storage, release of each memory-mapped record, rejection of copied-flat/mask/layer-id corruption, and equality between adapter and pilot maximum lengths. These are tests describing expected behavior; this chapter does not claim they were run in this authoring pass.

## 5. Warm start and model semantics

[`experiments/drafter_train/prepare_warmstart.py`](../../../experiments/drafter_train/prepare_warmstart.py) resolves a local directory or downloads only checkpoint/config assets. The native configuration is accepted only for the six-layer Laguna DFlash geometry: hidden 3072, intermediate 12288, head dimension 128, 72 query heads, 8 KV heads, vocabulary 100352, sliding window 512, per-head gating, no attention bias, and target layers `[1,10,19,29,38,47]`. Native capture-point ids must be target ids plus one: `[2,11,20,30,39,48]`.

The converter changes the architecture registration to `LagunaDFlashDraftModel` and the model type to `qwen3`, adds the explicit DFlash/training fields, and removes the vLLM capture-point metadata. Exactly 69 native keys are expected. Each six-layer fused `qkv_proj` must have shape

$$
(72\times128 + 2\times8\times128,\ 3072) = (11264,3072).
$$

It is split along dimension zero into query `(9216,3072)`, key `(1024,3072)`, and value `(1024,3072)`. Each slice is cloned because safetensors rejects shared allocations and to release the fused source before a shard flush. The training schema has 81 keys. Output is limited to 512 MiB shards; temporary shards and the optional index are renamed only after serialization.

[`experiments/drafter_train/laguna_dflash_model.py`](../../../experiments/drafter_train/laguna_dflash_model.py) is the first-party adaptation registered into SpecForge. It validates the exact geometry above before allocating the model. `LagunaDFlashAttention` projects the draft query from draft hidden state while concatenating target-context and draft keys/values. It supports the external flex-attention path, eager mask preparation, or another Transformers attention implementation. The output is multiplied by `softplus(g_proj(hidden))`; for this configuration, one scalar gate is broadcast across each 128-wide query head. The layer normalizes target context with the same input norm used for draft hidden state, applies attention with a residual, then post-attention norm, dense MLP and a second residual.

`LagunaDFlashDraftModel` directly constructs six Laguna layers rather than constructing and replacing Qwen layers, avoiding doubled peak construction memory. Six independently normalized captured target slices are concatenated, projected from width 18432 to 3072, and passed into SpecForge's inherited DFlash forward. Missing `target_hidden` or a width other than 18432 raises immediately. Attention kernels, rotary embedding implementation, DFlash loss/head behavior, cache behavior, optimizer and numerical accuracy remain framework contracts; the Reyu state machine does not reimplement them.

[`experiments/drafter_train/prepare_warmstart.py`](../../../experiments/drafter_train/prepare_warmstart.py) and [`experiments/drafter_train/export_laguna.py`](../../../experiments/drafter_train/export_laguna.py) are inverse at the QKV schema boundary, not claimed floating-point round-trip identities. Export locates a direct state, exactly one `*-latest` lineage, or the lexically latest step state. It requires strategy `dflash`, exact training keys, the same validated native reference schema, and projection shapes before concatenation. Any trained tensor left after consuming the native schema is an error. It writes the validated original Poolside config, not an improvised mutation of the Qwen training config.

## 6. Training configuration and orchestration

[`experiments/drafter_train/pilot.yaml`](../../../experiments/drafter_train/pilot.yaml) pins a local target snapshot and offline feature/cache paths. It selects DFlash, bfloat16, flex attention, a 3072-token window, batch size 1, four-step accumulation, six epochs, cosine schedule at learning rate `6e-4`, 4% warmup, gradient norm 1, 512 anchors, 64 objective chunk blocks, loss decay 7, seed 42, and newest-only checkpoint retention. Optimizer masters/moments are CPU-offloaded. Deployment is local colocated, one node and one process, with no FSDP sharding. Both warm-start and resume fields are null in YAML because SpecForge's string-typed overrides cannot reliably unset a baked field.

[`experiments/drafter_train/train.sh`](../../../experiments/drafter_train/train.sh) checks for an executable Python, a named warm source, exactly one CUDA device string, no vLLM process, and no CUDA compute process above 1 GiB. It sets expandable CUDA allocator segments and forces Hugging Face, Transformers and datasets offline. It validates/stages captures, creates or reuses the converted warm start, then selects **exactly one** of two external SpecForge inputs: `training.resume_from` when the latest complete training state exists, otherwise `model.draft_checkpoint_path`. It invokes export only after the CLI returns successfully.

[`experiments/drafter_train/train_runner.py`](../../../experiments/drafter_train/train_runner.py) exists to import and register the Laguna model before handing process control to `specforge.cli.main`. It contributes no independent optimizer or loop semantics.

[`experiments/drafter_train/tests/test_scaffold.py`](../../../experiments/drafter_train/tests/test_scaffold.py) codifies the native-to-training config transformation, 69-to-81 key transformation, single-GPU offline pilot settings, exclusive warm-start/resume branches, distribution-based quality comparison, inclusive gate thresholds, and shell syntax. It does not qualify GPU training or live serving by itself.

## 7. Export and acceptance

[`experiments/drafter_train/acceptance.sh`](../../../experiments/drafter_train/acceptance.sh) first requires exported weights/config and a gate corpus. It deletes stale evidence, disables hidden capture, and runs both arms at `max_model_len=98304`. The baseline gets 10,936,647,680 KV-cache bytes; the candidate gets 8,000,000,000 by default because the drafter is a third RTX 5090 resident. It captures 120 deterministic quality prompts without speculation, boots native DFlash with 15 speculative tokens, captures the same candidate distributions, runs six context rungs `(1024,8192,16384,32768,65536,98304)` with 512 output tokens, then summarizes.

[`experiments/drafter_train/gate.py`](../../../experiments/drafter_train/gate.py) samples 120 fixed 230-word prompts using seed `20260826`. Each arm must return one finite, nonempty top-5 log-probability mapping per prompt. The quality metric is Jensen-Shannon divergence in nats across the union of listed tokens plus one aggregate unlisted-tail bucket. Baseline and candidate schemas, top-logprob depth, counts and indices must align. At least 108 of 120 prompts must have JSD at most 0.10.

The ladder must contain exactly six rows with the expected label and ordered contexts. Counts must be nonnegative, accepted cannot exceed drafted, prompts must be nonempty, output must expose at least two tokens, timings must be finite and positive, and reported acceptance percentage must agree with counters to within 0.05 percentage point. Passing additionally requires aggregate acceptance at least 60%, every rung with a positive drafted count and at least 60% acceptance, and every rung with effective decode time at most 6.0 ms/token. Threshold equality passes.

This quality gate compares next-token distributions, not generated-string equality. The latency/acceptance counters come from `benchmarks/decode_ladder_probe.py`, an external dependency to this ownership set. A gate result therefore qualifies only the measured recipe and evidence; it is not a general proof of model quality, speed, or production safety.

## 8. Overnight lifecycle and source contradictions

[`experiments/drafter_overnight.sh`](../../../experiments/drafter_overnight.sh) is a resumable experimental supervisor. It keeps large captures outside `/tmp`, refuses GPU stages when tmpfs exceeds 5 GB, sizes a capture subset using `6 * 3072 * 2` hidden bytes per token, checks disk headroom, arms capture through `/tmp/sb_env_overrides.json`, stops serving for conversion/training, and records `.done` markers. Its `EXIT` trap leaves an already healthy service alone or boots the ordinary r15 recipe.

Important source-truth qualifications:

1. The header says corpus → capture → train → acceptance → production restore and states that stage failure restores production. The implementation checks only whether *some* server answers `/health`. If the speculative candidate remains healthy after acceptance, `restore_production()` returns without checking its configuration, so the candidate can remain live even though no source promotes it into the ordinary recipe. If no endpoint is healthy, the production boot can still log `PRODUCTION RESTORE FAILED -- operator needed`. The model separates `candidateHealthy`, `productionHealthy`, `preserveHealthyCandidate`, and `RestoreFailed`.
2. Capture and training failures exit nonzero. By contrast, a nonzero acceptance result is logged as `acceptance FAILED -- drafter stays unshipped`; the script does not mark acceptance, then reaches `pipeline complete` and `exit 0`. Thus the supervisor's final process status is not a reliable acceptance signal. The absence of the acceptance marker and the gate summary are authoritative.
3. Passing acceptance does not copy the candidate into the normal recipe or mutate production defaults. It can nevertheless leave the candidate service answering because the fallback identifies health, not recipe identity. `Accepted` therefore implies `candidatePromoted == false`, `candidateHealthy == true`, and `productionHealthy == false` in the model.
4. The adapter validates all records and can skip every record due to the 3072-token truncation rule while still returning success with a zero-record manifest. Later training/framework behavior, not the adapter, rejects the unusable dataset.
5. Datagen's generated completion ids are not stored. Capture retokenizes completion text and explicitly permits rare boundary differences. This is on-policy text replay, not exact sampled-token provenance.

## 9. Executable scenarios and properties

The Reyu model provides named paths rather than claiming implementation proof:

* `acceptedBoundaryTest`: 500-record minimum, four valid-but-truncated records, fresh warm start, and inclusive quality/acceptance/6.0 ms boundaries.
* `resumeTrainingTest`: the mutually exclusive resume branch and a stronger passing measurement.
* `malformedCaptureRejectedTest`: tensor/mask/identity validation failure.
* `missingCaptureRejectedTest`: capture hook never publishes every requested feature file.
* `invalidWarmstartRejectedTest`: native configuration/key/shape admission failure.
* `allTruncatedRejectedAtTrainingTest`: the adapter truthfully emits a zero-record manifest when every supervised suffix is truncated, after which training is unavailable.
* `qualityBelowBoundaryRejectedTest`: well-formed evidence with 107/120 quality prompts passing.
* `trainingRequiresExclusivityTest`: fresh training is disabled without the GPU exclusivity precondition.
* `acceptanceRequiresAllGatesTest`: 107/120 cannot enter `Accepted` even if aggregate token acceptance is 60%.
* `healthyCandidateSuppressesRestoreTest`: a passing, healthy candidate satisfies the supervisor's health check and prevents an ordinary production boot without becoming a promoted recipe.
* `restoreFailureVisibleTest`: after an earlier training failure leaves no healthy candidate or production endpoint, the ordinary fallback boot can also fail visibly.

`inv` preserves count bounds, schema order (`exportNative` implies `trained`), exclusive nonempty training mode for trained states, explicit rejection reasons, and the central research/production separation. `witnessAccepted`, `witnessTruncationEdge`, `witnessResearchOnly`, and `witnessFailure` make success, usable truncation, non-promotion, and error states directly reachable.

## 10. Abstraction limits and CPU comparisons

The model represents threshold arithmetic, state progression, identity counts, exact shape flags, suffix-mask validity, QKV schema direction, training-mode exclusivity, and restore outcome. It abstracts tensor contents, random sampling values, lock schedules, filesystem crash consistency, HTTP transport, systemd, CUDA allocation, attention numerics, optimizer state, speculative decoding internals, and wall-clock performance. Flex attention, eager attention, DFlash objective/head behavior and the decode-ladder producer are external contracts and require their own evidence.

Useful CPU-only implementation comparisons include constructing valid/corrupted small capture tensors and calling `validate_record`; staging a record set and reading it through SpecForge's offline reader; checking all-truncated staging produces an accounted zero-record manifest; building the Poolside config and comparing the exact 69/81 key sets; splitting and rejoining a synthetic correctly shaped fused QKV tensor; exercising `resolve_training_state` with ambiguous/no/latest lineages; comparing identical and perturbed top-logprob payloads; and feeding valid/bad ladder JSON to `_ladder_rows` and `summarize`. These checks can establish serialization, validation and threshold behavior without claiming GPU training, server integration or latency qualification.
