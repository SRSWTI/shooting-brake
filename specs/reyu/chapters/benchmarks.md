# Benchmark identity, comparison, and client workflows

This branch deepens the high-level “serving recipes and benchmark clients” box into two executable abstractions: [`benchmarks.ryu`](../models/benchmarks.ryu) covers provenance, run/cell lifecycle, cold/warm evidence, offload arms, and comparison; [`clients.ryu`](../models/clients.ryu) covers SLO-measurement streams and interactive chat/tool loops. These models describe checked-in control flow. They do not certify performance, numerical equality, cache state, or a particular server deployment.

## 1. Responsibility, actors, and interfaces

The benchmark surface has five actors:

1. **Operator and shell orchestration** select an endpoint, model, contexts, profiles, placements, and process boundaries.
2. **Matrix runner** binds an immutable workload/client/runner/server-manifest identity to a model output directory, constructs GuideLLM commands, records per-cell manifests, and resumes only compatible evidence.
3. **Live and in-process clients** obtain SSE token delivery, usage, telemetry, and worker observations. “Cold” in the r15 matrix means prefix-cache miss; it is unrelated to the optional CPU/DRAM cold-expert tier.
4. **Reducers and comparison tools** derive tables, ratios, prefix agreement, byte equality, or digest classifications from stored artifacts. Most do not establish that compared artifacts have the same full configuration.
5. **Human-facing clients** retain conversation state. The chat client never replays reasoning. The experimental agent preserves raw assistant XML, executes local tools, and returns ordinary tool messages to the model.

The principal boundaries are OpenAI-compatible `/v1/completions` or `/v1/chat/completions`, `/health`, `/metrics`, GuideLLM subprocesses, vLLM’s in-process engine API, plugin worker RPC telemetry, native trace files, and JSON/JSONL/NPZ evidence. Server internals, tokenizer behavior, driver timing, and tool side effects are external to the Reyu transition boundary.

## 2. Breadth-first source inventory

Every file assigned to this branch appears below. Links are to source; “modeled” identifies the behavioral contract represented in the two Reyu modules.

| Source | Role and exact anchors | Modeled contract |
|---|---|---|
| [`benchmarks/matrix_runner.py`](../../../benchmarks/matrix_runner.py) | `RunConfig`, `_identity_json`, `_client_identity`, `_run_identity`, `_resume_cells`, `_prepare_run`, `_cell_seed`, `_guidellm_command`, `_run_profile`, `main` | Canonical whole-run identity; prepare-before-network order; fail-closed reuse; cell manifest lifecycle; rc=0 completion; nonfatal metric scrape records. |
| [`tests/test_matrix_runner_identity.py`](../../../tests/test_matrix_runner_identity.py) | `_config`, `_evidence`, `_cell`, `_assert_rejected_unchanged`, `test_fresh_identity_and_matching_resume_preserve_evidence`, changed-workload/server/source/client tests, malformed/legacy/orphan tests, duration and seed-policy tests | Source-level executable contract showing which identity fields matter, rejection without overwrite, and that minimal rc=0 manifests need no report. It is inventory evidence, not a test run performed for this chapter. |
| [`benchmarks/run_matrix.sh`](../../../benchmarks/run_matrix.sh) | top-level strict shell entrypoint | Preflights health/metrics, then invokes one runner process per `(context,profile)` with a shared output root and cooling delays. |
| [`benchmarks/matrix_tiered.sh`](../../../benchmarks/matrix_tiered.sh) | `say`, `mem_gib`, `stop_server`, `start_server`, `cells`, top-level context loop | Restarts the server once per context tier, records runner rc without gating later tiers, and treats the memory floor as advisory. |
| [`benchmarks/matrix_progress.sh`](../../../benchmarks/matrix_progress.sh) | top-level observer | Counts `report.json`, probes fixed port 8017, process state, and memory. This is a weaker advisory completion predicate than runner manifests. |
| [`benchmarks/watch_and_chain.sh`](../../../benchmarks/watch_and_chain.sh) | `log`, `spawn_agent`, `b70_mib`, top-level watcher/audit/teardown/Track-B chain | Gates on external `chain_audit.py`, not watched-process status; teardown and B70-memory checks can warn and proceed; Track-B rc is final. |
| [`benchmarks/offload_benchmark.py`](../../../benchmarks/offload_benchmark.py) | `StreamTiming`, `apply_config_env`, `build_engine`, `stream_one`, `stream_many`, `adapter_source_sha`, workload functions, `run_capacity_frontier`, `worker_stats`, `run`, `main` | Fresh construction-time all-CUDA/hybrid/all-out modes; measured arrivals/telemetry; derived rates; recorded but not automatically compared correctness; capacity termination. |
| [`benchmarks/run_offload_sweep.sh`](../../../benchmarks/run_offload_sweep.sh) | `run_one` and top-level mode/placement loops | Fresh process per arm, default all-CUDA/hybrid sweep, opt-in CPU all-out and prefill-stream arms, pathname-only reuse. |
| [`benchmarks/offload_summarize.py`](../../../benchmarks/offload_summarize.py) | `_load`, `main` | Informational sorting and baseline/KV ratios; baseline inferred from zero B70 share; no identity or SLO gate and no CPU-service attribution. |
| [`benchmarks/compare.py`](../../../benchmarks/compare.py) | `CONFIGS`, `run_config`, `_common_prefix`, `compare_correctness`, `_delta`, `compare_throughput`, `compare_capacity`, `main` | Positional artifact comparison and reduced `comparison.json`; no full provenance check or failure on output divergence. |
| [`benchmarks/compare_pro_matrix.py`](../../../benchmarks/compare_pro_matrix.py) | `pro_sweep_peak`, `our_grid_cells`, `decode_and_peak_tables`, `pro_concurrent_means`, `ours_concurrent_means`, `fmt_gap`, report/history table functions, `main` | Cross-machine markdown reports from stored measurements, positional/nominal identity assumptions, explicit and implicit inferences. |
| [`benchmarks/r15_cold_warm_matrix.py`](../../../benchmarks/r15_cold_warm_matrix.py) | `Metrics`, `ExactPromptMint`, `Res`, `one_request`, `analyse_window`, `run_pass`, `main` | Exact-or-just-below prompt minting, repeated cold/warm prompt sets, metrics/trace windows, warnings retained without rejection, weak resume key. |
| [`benchmarks/pipeline_identity_gate.py`](../../../benchmarks/pipeline_identity_gate.py) | `capture`, `compare`, `main` | Deterministic input capture and pairwise byte equality. The documented multi-run nondeterminism envelope is not implemented. |
| [`benchmarks/weight_digest.py`](../../../benchmarks/weight_digest.py) | `_load_only`, `_read`, `compare`, `main` | Fresh-process post-hoc/preemptive capture; selected exact field comparisons and narrow accepted activation/fold differences; incomplete provenance. |
| [`benchmarks/run6_decode.sh`](../../../benchmarks/run6_decode.sh) | top-level historical orchestration | Boots/waits/warms an 88B server, observes PSI, then runs decode/saturation grids with `--skip-existing`; no embedded config digest. |
| [`benchmarks/run6_finish.sh`](../../../benchmarks/run6_finish.sh) | top-level historical orchestration | Deletes selected F cells, boots/warm-checks, runs F only, and prints presence-based completion; several checks are intentionally nonfatal. |
| [`benchmarks/slo_split_matrix.py`](../../../benchmarks/slo_split_matrix.py) | `build_prompt`, `one_request`, `run_cell`, `summarize`, `main_async`, `main` | Continuous-usage validation, request validity, bundled-delivery ITL invalidation, and partial-failure aggregation. Despite its name, it defines no numeric SLO classifier. |
| [`benchmarks/stream_matrix.py`](../../../benchmarks/stream_matrix.py) | `MODES`, `percentile`, `cell`, `run_mode`, `run`, `main` | Serial per-mode fresh engines, warmup/reset/measure/snapshot/shutdown, dispatch/stream/all-out branches, and per-mode output files. |
| [`benchmarks/stream_matrix.sh`](../../../benchmarks/stream_matrix.sh) | top-level launcher | oneAPI/native environment and CLI defaults. It starts one Python process that constructs several engines, despite its “one process per mode” comment. |
| [`benchmarks/sb_chat.py`](../../../benchmarks/sb_chat.py) | `Turn`, `wait_for_server`, `stream_turn`, `main` | Health gate, multi-turn command loop, permissive SSE EOF, usage accounting, pending-user rollback, and reasoning-excluded history. |
| [`benchmarks/sb_agent.py`](../../../benchmarks/sb_agent.py) | `_read`, `_write`, `_edit`, `_glob`, `_grep`, `_bash`, `TOOLS`, `make_tool_schema`, `render_system_prompt`, `run_tool`, `_coerce`, `parse_tool_calls`, `TagFilter`, `ChatRequestError`, `Turn`, `stream_turn`, `trim_oversized_history`, `main` | Custom XML tool protocol, bounded local tool surface, sequential tool results, context trimming, error-as-result, raw XML history, and 20-generation bound. |

## 3. Matrix identity, manifest, and cell mechanics

`matrix_runner.py` canonicalizes JSON with sorted keys, compact separators, and rejected nonfinite values. Its schema-v1 identity includes the benchmark configuration, exact client Python and installed-distribution inventory, SHA-256 of the raw runner source, and either canonical operator-supplied server manifest content or `not_provided`. `output_root`, `skip_existing`, and the manifest path are operational and excluded. Manifest formatting and relocation therefore do not change identity; server changes are invisible when no manifest is supplied. The model uses a positive integer as the finite stand-in for the canonical JSON hash.

Preparation chooses `<output_root>/<model.replace('/', '__')>`. That slug is not injective: `a/b` and `a__b` collide. A populated legacy directory without identity, malformed/different identity, malformed or mismatched cell manifest, or orphan evidence rejects the entire resume before network preflight and without overwriting existing evidence. A fresh run writes identity/config before metrics preflight, so an unreachable server can leave an offline-prepared directory.

A cell is keyed by context and profile. `_run_profile` first writes a manifest with null finish/return code, records the GuideLLM command, redirects subprocess output, and appends metric scrapes. It then records finish time and rc. A scrape error is an error record, not a cell failure. Nonzero rc is persisted and raised; null/interrupted and nonzero attempts are retryable. Exact integer zero is complete, even when no report exists. Conversely, `matrix_progress.sh` calls any `report.json` complete; that predicate can disagree with resume.

The shell composition contradicts the immutable identity boundary. Both `run_matrix.sh` and `matrix_tiered.sh` repeatedly invoke the runner with only one cell/context while `contexts` and `profiles` belong to the full stored identity. After the first invocation, a different cell/context at the same model directory has a different identity and is rejected before network work. The strict wrapper stops; the tiered wrapper logs runner failures and can still print “done.” This is an implementation contradiction, not a proposed fix. Cooling sleeps do not establish a cold cache, and only the tiered server restart establishes process-level coldness before the first profile of each context.

## 4. Cold/warm and offload workflows

`r15_cold_warm_matrix.py` mints deterministic exact-or-just-below token prompts, runs the same prompt set cold then warm, records API usage/timings, Prometheus deltas, native trace windows, memory, and B70 frequency, and persists after each cell. Prefix hits during “cold,” token drift outside tolerance, and request errors are recorded or warned; they do not reject the cell. Resume uses only `(context_tokens, concurrency)`, not model, corpus, seed, output length, server config, device count, or prior success. A totally failed cold pass can persist and then crash while formatting `None` drift. The model therefore separates “recorded” from “qualified.”

`offload_benchmark.py` constructs one of three engine modes:

- `all-cuda`: plugin present but B70/surgery/Tier 3 disabled;
- `hybrid`: placement, B70, surgery, and graph mode enabled;
- `all-out`: hybrid plus CUDA+B70+CPU/DRAM Tier 3.

It measures arrival times, token IDs/logprobs, routes, KV/memory, and worker service telemetry, then derives rates and percentiles. Multi-token chunks divide one arrival gap evenly. Context prompt sizes are approximate. Capacity exceptions and zero-output stop the frontier without a distinct failure artifact. The serialized identity omits many effective workload and environment values; `adapter_source_sha` scans `phase4/src` rather than `src/phase4/src`, so it does not cover the actual adapter tree. `ADAPTER_VARS` also omits the preemptive-surgery variable, allowing inherited contamination. These omissions are modeled as comparison provenance gaps, not silently repaired.

The most useful CPU implementation comparison already expressible by source is an opt-in three-arm `all-cuda` versus `hybrid` versus `all-out:<layers>:<cuda>:<cpu>` run under identical prompts, admission limits, effective environment, checkpoint, and source/hardware identity, optionally paired with `ALLOUT_STREAM_PLACEMENTS`. Existing summary output has end-to-end rates, KV, B70 share, and B70 service but no explicit CPU route/service column; a CPU attribution claim therefore requires inspecting richer worker evidence or extending a future harness, not inferring it from the placement label.

## 5. Result comparison and evidence classes

The chapter uses three evidence classes:

- **Measured:** client arrival timestamps and usage; provider arrays; worker/native telemetry; captured digest fields; server reports.
- **Source-derived:** percentiles, rates, common-prefix lengths, byte equality, digest equality, ratios, and table selection applied to measured inputs.
- **Inferred/manual:** same-checkpoint claims without a digest, 128K-class equivalence between unequal contexts, saturation interpolation, qualitative output correctness, baseline identity inferred from zero B70 share, or accepted scale differences.

`compare.py` zips lists positionally after length checks; it does not verify per-row prompt/context/concurrency identity or top-level model/placement/source identity. Its `run_config` invokes absent `benchmarks/benchmark.py`, while `offload_benchmark.py` says it is the intended driver. It reports divergence rather than failing. Its capacity consumer expects flat rows while the producer emits `{max_completed_wave, points}`.

`compare_pro_matrix.py` derives PRO concurrency from row order, tolerates missing data, ignores its parsed successful-count field, and emits “same checkpoint/harness” without checking either. It explicitly labels one 2048-token saturation statement as inference, but other nominal equivalences remain unverified.

`pipeline_identity_gate.py` gives a strong pairwise byte comparison for deterministic captures. It does not implement its header’s A1/A2 nondeterminism envelope or store full provenance in NPZ. `weight_digest.py` compares selected addressing-sensitive fields, allows specific activation/fold field differences, and can skip missing/mismatched alpha vectors; “WEIGHTS IDENTICAL” is narrower than literal all-field identity.

## 6. SLO measurement semantics

`slo_split_matrix.py` builds a leading-nonce prompt, issues deterministic streaming completions, and trusts cumulative `usage.completion_tokens`, never SSE frame count. Counts must be integers and nondecreasing. A request is valid only after `[DONE]`, a positive token delivery, and final usage. HTTP/chunk/protocol/timeout/JSON failures become failed records. TTFT starts at first positive token delivery; bundled deltas invalidate p50/p99 ITL instead of pretending a frame is one token.

A cell may retain metrics when only some requests succeed. It prints `FAILED` only when none succeed and returns success even when all cells have no successful requests. There are no numeric TTFT/ITL/throughput thresholds. The model’s `MeasurementsAvailable`, `PartialMeasurements`, and `NoSuccessfulRequests` classify evidence availability only and must not be read as SLO compliance.

`stream_matrix.py` is an in-process offload crossover sweep, not the same protocol. Each mode gets a fresh engine, warmups, stats reset, timed cells, snapshots, shutdown, and a sibling output file. Its prompt variants append suffixes to a common warmed base; exact prompt equality is prevented, but prefix-cache invalidation is not established. Empty `MODES` is described as “none” by the shell comment but actually leaves argparse’s required list without a value.

## 7. Chat and tool-client loops

`sb_chat.py` waits for health, then accepts session commands for reset, system prompt, reasoning generation/display, stats, and exit. A normal turn appends the user, streams deltas, and removes that pending user on handled connection failure. Reasoning may be shown but only final content enters assistant history. Completion volume comes from server usage. Malformed SSE JSON is ignored; transport EOF is accepted without requiring `[DONE]` or examining `finish_reason`, so an empty or incomplete stream can still become a turn.

`sb_agent.py` embeds six local tools in ordinary system text because the server-side OpenAI `tools` path was observed unsuitable. Its XML parser accepts multiple calls, coerces known scalar types, overwrites repeated parameters, and leaves unknowns as strings. Tool failures are returned as normal `error: ...` tool content. Each successful generation appends raw assistant XML; parsed tools execute sequentially; bare `role=tool` results are appended; then generation repeats for at most 20 iterations. Reasoning remains excluded from history. `TagFilter` hides tool XML only from display and can leave a short pending display suffix at EOF. A recognized context error can replace large interior messages and retry; other HTTP/connection errors pop only the current tail, which may be a tool result rather than the entire user turn. The tool surface can read, overwrite, exact-edit, glob, grep, and execute shell commands within its selected working directory; it is an experimental local mutation surface, not production serving logic.

## 8. Executable scenarios and properties

`benchmarks.ryu` includes reachable witnesses for:

- rc=0 matrix completion without any result report;
- nonfatal metrics scrape failure;
- failed/null retry and identity mismatch rejection before execution;
- recorded cold-prefix hits/token drift that do not reject;
- comparison of unmatched cold/warm or offload identities;
- measured CPU evidence only on an all-out arm;
- pairwise pipeline byte success and non-bitwise failure without an invented envelope.

`clients.ryu` includes:

- valid and bundled cumulative-usage streams;
- missing-`[DONE]` measurement failure and partial-cell evidence;
- chat success on EOF with reasoning excluded from history;
- pending-user rollback on chat failure;
- agent tool errors as ordinary results, context trimming/retry, and the 20-round cap;
- server-readiness timeout.

The invariants keep counters nonnegative, CPU evidence tied to all-out mode, matrix completion tied to rc, warm after cold, reasoning absent from replay history, and summary states consistent with successful/failed request counts. They are finite behavioral checks, not claims about real latency, numerical tolerance, cache coherence, or exhaustive implementation equivalence.

## 9. Limits and external dependencies

GuideLLM, vLLM, model/tokenizer checkpoints, the plugin, CUDA/XPU/oneAPI runtimes, provider libraries, Prometheus, native traces, local hardware controls, and external PRO result trees remain dependencies. Fixed finite identity integers abstract canonical JSON and SHA-256 without modeling collision resistance. Timing magnitudes, percentiles, tensors, paths, file bytes, shell process matching, and destructive local tool effects are abstracted at their observable decision points. Historical run6 comments are prior evidence and sometimes disagree on the KV-token interpretation of the same byte allocation; neither value is promoted to a model invariant.
