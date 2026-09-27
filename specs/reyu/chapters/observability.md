# Route observability, locality, and topology

This branch explains how Shooting Brake observes router output, persists it, reduces it into frequency/locality/topology summaries, and scopes runtime telemetry. Read it breadth first: responsibility and data flow first, then the state table, then the exact algorithms and executable examples in [the Reyu model](../models/observability.ryu).

## 1. Responsibility and boundary

The subsystem has five related but distinct jobs:

1. **Observe global routes before placement changes their identity.** [`RouteCounter.observe`](../../../src/phase4/src/shooting_brake_vllm/route_stats.py) accumulates a device histogram, while `RouteTrace.observe` stages token rows into the versioned `SBRTv1` stream. `HybridRoutedExperts.forward_modular` invokes both above the all-CUDA pass-through and before compact CUDA remapping, so the observations describe model-selected global expert IDs rather than local slots.
2. **Validate and analyze traces offline.** [`benchmarks/route_locality.py`](../../../benchmarks/route_locality.py) owns the binary reader, strict record validation, consecutive-run/Jaccard analysis, sliding working sets, LRU accounting, and uniform nulls.
3. **Evaluate candidate two-card layouts on the same selected records.** [`benchmarks/route_topology.py`](../../../benchmarks/route_topology.py) filters a concurrency-1 proxy, optionally retains request-tail runs, counts routes per card, evaluates the integer kernel-cost knots, and sweeps every legal contiguous boundary.
4. **Expose scoped worker observations.** [`telemetry.py`](../../../src/phase4/src/shooting_brake_vllm/telemetry.py) is a `collective_rpc` surface for route shares, eager liveness, per-device pollers, health baselines, histogram liveness, memory, and optional power. [`seam_trace.py`](../../../src/phase4/src/shooting_brake_vllm/seam_trace.py) is a separate sampled, opt-in decode seam timeline.
5. **Drive one frequency calibration.** [`calibrate_routes.py`](../../../src/phase7/calibrate_routes.py) requests all-CUDA execution, warms up, resets counters, runs a corpus, asks the worker to dump CSV, then analyzes that CSV. [`calibrate_routes.sh`](../../../src/phase7/calibrate_routes.sh) is only its environment launcher.

The router, vLLM worker scheduling, PyTorch/CUDA event semantics, NumPy, the dataset loader, and the model itself are dependencies. This specification does not treat their internals as Shooting Brake state. It also makes no new latency or locality claim: the analyzers compute observations from a supplied trace; they do not establish that a particular production workload has a particular distribution.

## 2. Actors and interfaces

```text
HybridRoutedExperts.forward_modular
     | global topk_ids [rows, top_k]
     +--> RouteCounter.observe ---- device [layer,expert] counts
     |                                  |
     |                                  +-- collective_rpc --> CSV/skew analysis
     |
     +--> eager RouteTrace.observe -- 8 staging slots -- SBRTv1 file
                                                       |
                                     +-----------------+----------------+
                                     |                                  |
                            route_locality.py                   route_topology.py
                          runs/windows/LRU/nulls          selection/card counts/sweep

Worker.execute_model wrapper ---- step boundaries ---- SeamTracer
Hybrid graph branch ------------ four seam marks -----/

collective_rpc ---------------- collect/reset telemetry inside EngineCore worker
```

The histogram and detailed trace are independent opt-ins. `SHOOTING_BRAKE_ROUTE_STATS=1` lazily creates the histogram. A non-empty `SHOOTING_BRAKE_ROUTE_TRACE` path lazily creates the binary writer. `SB_SEAM_TRACE=1` or `/tmp/sb_seam_trace.arm` enables the seam singleton. Missing keys in `collect_worker_stats` usually mean “subsystem inactive”; they are intentionally not synthesized as zero.

## 3. Source-backed state and operation table

| State or operation | Source owner | Implemented behavior | Model abstraction |
|---|---|---|---|
| Histogram geometry | `route_stats.py::RouteCounter.__init__` | Allocates normal, device-resident `int64` counts `[num_layers,num_experts]` and token counts `[num_layers]`; construction temporarily disables inference mode so later RPC reset may mutate the tensors. | `histogramEnabled`, `histogramRoutes` |
| Histogram observation | `RouteCounter.observe` | Flattens global IDs, `scatter_add_`s one for every selected expert, and adds row count to the layer token count. Out-of-range layer indices return without an update. | Each top-2 example record contributes exactly two routes. |
| Trace identity | `RouteTrace.__init__` | Writes an `SBRTv1\0\0`, version-1 header containing header size, record size, top-k, layer count, and expert count. | `traceEnabled`, `recordCount` |
| Per-layer step | `RouteTrace.observe` | Uses an independent monotonically increasing step counter for every layer. Rows within one call receive the same step and row indices `0..M-1`. | The concrete example uses one layer/row and steps `0,1,3,4`; the gap represents omitted multi-row work. |
| Staging | `RouteTrace._allocate_staging`, `_finish_slot`, `_append_slot` | Eight CPU slots, pinned for CUDA input, are reused in a ring. CUDA events protect asynchronous D2H copies. A new row high-water drains all slots and reallocates. Output blocks contain 65,536 records. | `pendingRecords` commits atomically to `durableRecords`; payload copies and blocks are abstract. |
| Flush/close | `RouteTrace.flush`, `close` | Flush waits all pending slots and flushes the file but leaves the writer open. Close is idempotent, calls flush, then closes even if flushing raises. | `flushTrace` preserves `Recording`; `closeTrace` moves to `Closed`. |
| Trace read validation | `route_locality.py::read_trace` | Rejects a truncated/bad header, wrong version/sizes, invalid geometry, truncated final record, out-of-range layer/expert, duplicate expert within top-k, and duplicate `(layer,step,row)`. | `rejectMalformedTrace` is the observable fail-closed result. |
| Consecutive runs | `_consecutive_runs` | Sorts one layer by row then step; a run continues only when the row is unchanged and step is exactly previous+1. | Pure `continuesRun`; gaps suppress Jaccard/window comparison. |
| Consecutive Jaccard | `_layer_metrics` | For adjacent records inside each run, computes `|previous ∩ current| / |previous ∪ current|`; aggregates by comparison count, not by record count. | Top-2 Jaccard is scaled by six in `topTwoJaccardSixths`. |
| Sliding working set | `_working_set_sums` | Per run and window width, maintains expert occurrence counts while sliding and adds the number of distinct experts. No window crosses a row or step gap. | The fixed width-2 example stores exact distinct sums and window count. |
| LRU | `_lru_counts` | Sorts the entire layer by step then row, processes expert IDs in stored order, moves a hit to MRU, inserts a miss, and evicts the LRU above capacity. Unlike Jaccard/windows, LRU does **not** reset on a step gap. | Capacity-two `lruCapacityTwo*` pure helpers and exact hits/requests. |
| Uniform null | `uniform_jaccard`, `uniform_working_set`, `_uniform_lru_rates` | Jaccard and working-set nulls are analytic. LRU uses a deterministic 100,000-token, fixed-seed, without-replacement top-k simulation because `capacity/E` is not exact for an ordered top-k set. | Documented boundary; the finite model does not reproduce the Monte Carlo loop. |
| Single-row filter | `route_topology.py::select_single_row_steps` | Groups by `(layer,step)` and retains only groups containing exactly one row. A one-row prefill is indistinguishable. | Selection prerequisite, not a claim that retained rows are necessarily decode. |
| Tail-run selection | `select_last_step_runs` | Splits unique global steps at gaps and keeps the last requested number of consecutive runs; rejects non-positive, empty, or insufficient selection. | `recordRoute` applies `continuesRun` to every bounded input; the example produces runs `0:1` and `3:4`. |
| Candidate counts | `simulate_topologies` | Contiguous: count IDs in A and B ranges. Replicated occurrence: split each record’s total remote occurrences as `ceil(remote/2)` and `floor(remote/2)`. Single-card: count IDs above the configured current-CUDA prefix. | `recordRoute` functionally accumulates all three candidates from its expert-ID parameters; analysis publishes the selected accumulator. |
| Critical cost | `topology_summary` | The critical route count is `max(A,B)` for two cards and A for one card. Kernel cost indexes `KERNEL_US` by integer per-card route count and takes the slower card’s knot; there is no interpolation. | `kernelCostTenths` preserves the first three knots as exact integers. It is accounting, not a performance prediction. |
| Boundary sweep | `sweep_boundaries` | Holds the CUDA prefix fixed, enumerates every boundary from `cuda_experts+1` through `num_experts-1`, and reports observed mean-max, exact uniform mean-max, knot cost, and zero fractions. `format_sweep` separately marks the configured boundary and reports the minimum observed row. | With four experts the model functionally maintains both legal cuts, then applies the source’s first-minimum tie behavior. |
| Telemetry reset | `telemetry.py::reset_worker_stats` | Zeros route-share tensors, eager counters, every B70 poller once, and route histogram tensors. Native provider health is monotonic, so reset snapshots `(generation,dispatches)` instead of mutating it. | `resetMeasurement` installs a baseline and clears warmup histogram traffic. |
| Provider health scope | `_provider_health_stats` | Delta is available only when generation is unchanged and raw dispatches have not moved backward. Otherwise the raw values remain auditable but the scoped result becomes unavailable with a reason. | `collectProviderHealth` separates availability from delta. |
| Telemetry snapshot | `collect_worker_stats` | Traverses actual `HybridRoutedExperts`; aggregates route shares, eager expectations, benchmark-arm observations, per-card pollers, CPU tier, host arena/streamer, histogram liveness, cache config, CUDA/B70/host memory, provider health, and optional sampled power. Reads occur between steps and may synchronize device counters. | Only lifecycle/availability and provider-baseline behavior are modeled; inventory fields remain documented observations. |
| Seam sampling | `SeamTracer` | Worker step wrapper samples every Nth decode-shaped step (`M <= 32`). A fallback detects non-increasing layer index if the wrapper never engaged. Each recorded seam owns four host timestamps and four CUDA events. Capturing streams and large M are skipped. | Independent `sampleDecodeSeam`, capture skip, and shape skip actions. |
| Seam dump | `SeamTracer.dump` | Once only and only with records: CUDA synchronize, compute three intra-seam GPU intervals and inter-layer gap, use NaN for partial event pairs, write one NPZ. Dump occurs at threshold or `atexit`. | `seamDumped` requires at least one record. |
| Calibration | `calibrate_routes.py::run` | Forces all-CUDA configuration and route statistics, loads qualifying corpus rows, warms up, calls the shared reset RPC, executes batches, dumps from the worker through RPC, shuts down, then reads/analyzes CSV in the driver. | `Warmup -> Reset -> Measuring -> Dumped`; warmup routes are observably excluded. |

## 4. Actual module mechanics

### 4.1 Histogram versus token trace

The histogram is capture-safe: both updates remain device-side and shape-static. It loses time, row, and top-k co-occurrence identity but is sufficient for per-layer frequency coverage (`n50`, `n80`, `n90`, top-N shares). `analyze` sorts each live layer descending, uses ceiling percentage thresholds, skips inactive layers, then averages only live layers.

The detailed trace retains each `(step, layer, row, experts[top_k])`. Its D2H side effect is deliberately wrapped by `eager_break_during_capture`, so replay produces records rather than recording only the capture pass. The shared writer assumes layers call it in corresponding forward order; per-layer counters align calls without a separate begin-step synchronization. Version 1 has no request ID and no prefill/decode flag.

That missing identity is a hard interpretation limit. Multi-row `(layer,step)` groups are certainly not concurrency-1 decode and may be filtered. A retained one-row group could still be a one-row prefill. Under continuous batching a reused row slot cannot be assigned to a request. `route_topology.py` therefore states a capture procedure—strictly sequential requests, multi-token prompts, and optional tail-run selection—rather than inferring identity the format does not carry.

### 4.2 Small CPU comparison sequence

`recordRoute(step,row,e0,e1)` is parameterized: for every accepted bounded record it computes run continuity, set overlap/union, two ordered LRU accesses, configured contiguous counts, replicated-occurrence counts, single-card counts, and both legal boundary candidates. The scenario below is intentionally small enough to compare by hand or against the Python analyzers:

| Record | Step | Layer,row | Ordered experts | Consecutive with prior? |
|---:|---:|---|---|---|
| 0 | 0 | 0,0 | `[0,1]` | no prior |
| 1 | 1 | 0,0 | `[0,2]` | yes |
| 2 | 3 | 0,0 | `[2,3]` | no: step 2 is absent |
| 3 | 4 | 0,0 | `[2,3]` | yes |

Expected values derived from the source formulas (the parent verification pass performs the implementation comparison):

- Consecutive runs are `0:1` and `3:4`.
- Jaccard values are `1/3` and `1`; the model stores `2 + 6 = 8` sixths across two comparisons.
- Width-2 distinct working sets are `3` and `2`; sum `5` across two windows.
- Capacity-2 LRU starts empty and processes IDs in record order. Cache states after records are `[1,0]`, `[2,0]`, `[3,2]`, `[3,2]` in MRU/LRU notation. Hits per record are `0,1,1,2`, hence 4 hits from 8 requests. The step gap does not clear LRU state, matching `_lru_counts`.

For topology, hold CUDA at expert `{0}` and start with contiguous A=`{1}`, B=`{2,3}`. Per-record `(A,B,max)` is `(1,0,1)`, `(0,1,1)`, `(0,2,2)`, `(0,2,2)`. The expected critical-route total is 6. Applying the checked-in knots gives `30.8 + 30.8 + 32.2 + 32.2 = 126.0 us`; the model stores 1260 tenths solely to avoid floating-point abstraction.

Replicated-occurrence accounting splits remote counts `1,1,2,2` into ceiling/floor card counts, so the expected critical count is 1 for each record: total 4 and knot total 123.2 us. Sweeping the only other legal small boundary, boundary 3 gives A=`{1,2}`, B=`{3}` and critical counts `1,1,1,1`; the source formula therefore selects it for this fixture. These are functional expected values awaiting the parent’s differential check, not observed device results or a claim about a production route distribution.

### 4.3 Locality aggregation details

Metrics are first computed per active layer. Aggregate Jaccard weights every valid consecutive comparison equally. Aggregate working-set means weight every complete window equally. Aggregate LRU sums hits and requests before dividing. Optional `expert_bytes` multiplies misses and hits into fetched/saved bytes; it must be positive. “Round trips saved” and “expert payloads saved” are exactly hit counts in this hypothetical one-fetch-per-expert cache accounting, not evidence that a serving implementation currently installs such a cache.

The topology analyzer can consume exactly the same post-selection records for locality using `--with-locality`; it does not reparse a broader trace. Its zero-card fractions identify rows where an all-sentinel host dispatch might be skippable, but the displayed device-kernel accounting does not add a host round-trip saving.

### 4.4 Telemetry lifecycle and interpretation

`collect_worker_stats` runs inside the EngineCore worker because the driver cannot directly reach these counters. It intentionally distinguishes inactive from idle:

- route-share output is absent until total routes are nonzero;
- poller output always has an `available` flag and reason when absent;
- CPU poller, arena, and streamer are independent because B70 prefill streaming may use host arena/streamer without a CPU poller;
- route histogram snapshot is a compact liveness summary, while full counts leave through `dump_route_histogram`;
- provider health exposes raw generation/dispatch values and scoped baselines, refusing a delta across generation change or counter regression;
- CUDA power is a fallible point sample, not integrated energy, and failure is swallowed without dropping the rest of the snapshot;
- hybrid KV cache reports raw scheduler values and explicitly refuses to invent a max-token value.

Reset ordering matters. Calibration warms the engine first, then `reset_worker_stats` removes profiling/capture/correctness contamination from mutable counters and takes a provider baseline. The route histogram reset directly zeros the normal tensors whose allocation deliberately escaped inference-tensor restrictions.

### 4.5 Seam trace lifecycle

The worker wrapper calls `step_begin`, times the original `Worker.execute_model`, and calls `step_end`. Sampling cadence counts eligible step boundaries; steps above `_MAX_SEAM_M=32` disarm sampling for that step. In the graph-hybrid routed layer, `begin` records entry, then marks occur after local CUDA enqueue, after B70 takes, and after final adds. Current-stream capture suppresses records because recording those events would bake profiler events into the graph.

Dump is terminal for that tracer. It synchronizes CUDA, writes layer/M/host arrays, three event-pair elapsed times, and inter-layer gaps only when layer index increases. A `RuntimeError` from an incomplete event pair becomes NaN for that record instead of aborting the dump. If no record was ever accepted, `dump` is a no-op and `_dumped` remains false.

## 5. Modes, failures, and contradictions

### Disabled and mode branches

- With neither histogram nor trace configured, singleton getters return `None`; no observability tensor or writer is allocated.
- Histogram can run in all-CUDA mode because its observer is above placement dispatch. This is why calibration does not require B70 round trips.
- Trace enablement has more hot-path impact: it creates an eager graph break and D2H staging.
- Topology defaults describe a candidate 12-CUDA, boundary-96 split and a separate current-single-card CUDA prefix 54 for a 180-expert trace. They are analyzer defaults, not the production r15 recipe’s placement.
- Seam tracing is independently armed by an environment flag or a file marker, because the source documents EngineCore environment curation.

### Failures preserved in the model/documentation

- Trace writer rejects post-close observation, layer/shape/device drift, and model geometry drift after singleton creation.
- Reader rejects malformed identity, dimensions, records, and duplicate keys rather than repairing them.
- Topology rejects top-k beyond the measured knot table, invalid range geometry, empty selection, and invalid tail-run requests.
- Telemetry marks scoped provider health unavailable when generation changes or dispatch count regresses. This does not erase the raw health observation.
- Seam capture/large-shape skips produce no seam record; a partial CUDA event pair produces NaN at dump.

### Checked-in calibration launcher contradiction

The calibration algorithm is source-readable, but the checked-in launcher/import paths do not line up with this repository layout:

- `src/phase7/calibrate_routes.sh` changes directory from `src/phase7` to `src`, then invokes `python src/phase7/calibrate_routes.py`, which resolves to `src/src/phase7/calibrate_routes.py` in this checkout.
- `src/phase7/calibrate_routes.py` prepends `Path(__file__).parents[1] / "benchmarks"`, which resolves to `src/benchmarks`, while `offload_benchmark.py` lives in the repository-root `benchmarks/` directory.
- Its usage comment names `./phase7/calibrate_routes.sh`, consistent with a historical invocation from `src`, not the repository-root path shown by the present tree.

No spec file corrects or idealizes those paths. Unless an unmodeled external layout/module installation supplies the missing locations, the checked-in shell-to-Python calibration workflow is not directly runnable as written. The internal warmup/reset/collect/dump/analyze semantics above describe `run` itself.

## 6. Executable scenarios and properties

[`observability.ryu`](../models/observability.ryu) contains `init`, nondeterministic `step`, and `inv`, plus these named scenarios:

- `disabledObservabilityTest` observes the no-allocation state; `disabledRejectsRecordTest` separately asserts that it cannot accept a trace record.
- `malformedTraceRejectedTest` observes the explicit rejected state; `rejectedTraceCannotCloseTest` separately asserts that it cannot close a writer.
- `exactRouteSequenceTest`: the four rows reach exact consecutive/window/LRU totals and contiguous topology totals.
- `flushThenResumeTest`: flush makes current records durable without closing, so another observation remains enabled.
- `replicatedTopologyTest` and `boundarySweepTest`: exercise alternative card accounting on the same rows.
- `emptyTraceStateTest` observes a closed zero-record trace; `emptyTraceCannotAnalyzeTest` separately asserts that topology analysis is disabled with no selected record.
- `seamSamplingAndSkipTest`: separates accepted decode sample, stream-capture skip, large-M skip, and terminal dump.
- `calibrationExcludesWarmupTest`: warmup contributes two routes, reset clears them, and the final dump contains only eight measured routes.
- `providerGenerationChangeTest`: a provider generation change invalidates the scoped health delta.

Reachable witnesses expose disabled, exact-locality, contiguous, replicated, swept-boundary, warmup-excluded, generation-rejected, and seam-lifecycle states. `inv` checks bounded trace accounting, durability, LRU request conservation, analysis completeness, seam dump preconditions, provider baseline requirements, and rejection/error consistency.

Pure functions intended for direct CPU differential checks are:

- `continuesRun(previousStep,currentStep,sameRow)` ↔ `_consecutive_runs` continuation predicate;
- `lruCapacityTwoHit`, `lruCapacityTwoNextRecent`, `lruCapacityTwoNextOlder`, and `lruCapacityTwoHits(experts)` ↔ capacity-two `OrderedDict` behavior and full-sequence hit totals in `_lru_counts`;
- `topTwoJaccardSixths(overlap)` ↔ top-2 Jaccard cases in `_layer_metrics`;
- `kernelCostTenths(routes)` ↔ `KERNEL_US[0:3]`.

A successful Reyu scenario establishes behavior of this explicit finite abstraction only. It does not prove the binary writer, CUDA event timing, NumPy reductions, or vLLM integration equivalent.

## 7. Assigned source inventory

| Assigned file | Role in this branch |
|---|---|
| [`benchmarks/route_locality.py`](../../../benchmarks/route_locality.py) | Functional offline trace format, validation, nulls, locality algorithms, report. |
| [`benchmarks/route_topology.py`](../../../benchmarks/route_topology.py) | Functional selection, topology counting, measured-knot accounting, boundary sweep, synthetic trace generator, CLI. |
| [`src/phase4/src/shooting_brake_vllm/route_stats.py`](../../../src/phase4/src/shooting_brake_vllm/route_stats.py) | Live global-route histogram and binary trace producer/lifecycle. |
| [`src/phase4/src/shooting_brake_vllm/seam_trace.py`](../../../src/phase4/src/shooting_brake_vllm/seam_trace.py) | Sampled eager seam timeline and NPZ lifecycle. |
| [`src/phase4/src/shooting_brake_vllm/telemetry.py`](../../../src/phase4/src/shooting_brake_vllm/telemetry.py) | Worker RPC collection/reset/dump surface and scoped provider health. |
| [`src/phase7/calibrate_routes.py`](../../../src/phase7/calibrate_routes.py) | Calibration workflow logic; depends on benchmark engine helpers and dataset. |
| [`src/phase7/calibrate_routes.sh`](../../../src/phase7/calibrate_routes.sh) | Environment/venv launcher; contains the path contradiction above. |
| [`tests/test_route_locality.py`](../../../tests/test_route_locality.py) | CPU harness comparing repeated routes with deterministic uniform traces across Jaccard, working sets, LRU, and report sections. |
| [`tests/test_route_topology.py`](../../../tests/test_route_topology.py) | CPU harness for analytic uniform max, clustered topology separation, row filtering, tail runs, and boundary sweep. |

## 8. Limits and external dependencies

The model bounds records to one layer, top-2, four experts, capacity two, and at most four already-reader-sorted `(step,row)` coordinates. Within that domain, `recordRoute` accepts varying coordinates and expert IDs and computes locality/LRU/topology state from those inputs; the fixed sequence appears only in run scenarios and reachable witnesses. Production geometry remains in trace headers and runtime model qualification. The model does not allocate tensors, serialize bytes, simulate eight asynchronous slots, reproduce the 100,000-token LRU null, execute the full combinatorial uniform-topology expectation, or model floating-point percentile behavior. It preserves the relevant ordering, identity, reset, selection, accounting, availability, and error observations.

Potential CPU implementation comparisons are trace write/read round-trip, malformed-file rejection, run splitting, parameterized locality totals, capacity-two LRU state transitions, topology totals, boundary-sweep minimum, and uniform analytic expectation. The two assigned test files already encode larger deterministic comparisons. They are harness contracts, not claims that this authoring task ran them.
