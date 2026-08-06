# Shooting Brake — benchmarks

Two tracks. They answer different questions and must not be confused.

**Track A** holds the *architecture* fixed and sweeps *load* — the
serving-grade SLO picture against a live server.
**Track B** holds the *load* fixed and sweeps the *architecture* — the
offload-amount tradeoff curve, in-process and fast.

Both reuse the existing Phase 0 all-CUDA baseline; neither re-runs it.

---

## Quick start (commands, in order)

Power-cap first (holds a fixed ceiling so the two tracks are comparable;
one sudo prompt), then the server in its own shell, then the matrix.

```bash
cd ~/srswti/shooting-brake

# 0. cap the 5090 at 575 W (sudo, one password prompt)
bash benchmarks/gpu_power.sh cap 575

# 1. Shell 1 — launch the hybrid server (blocks; ~3-5 min to ready)
bash benchmarks/serve_hybrid.sh
#    If 131k refuses to start, retry lower:
#     MAX_MODEL_LEN=65536 bash benchmarks/serve_hybrid.sh

# 2. Shell 2 — smoke pass first (~15 min, 3 contexts)
CONTEXTS=1024,4096,8192 OUTPUT_ROOT=$PWD/bench-matrix/smoke \
  bash benchmarks/run_matrix.sh

# 3. Shell 2 — full SLO matrix (hours, resumable)
bash benchmarks/run_matrix.sh

# 4. teardown
pkill -INT -f 'vllm serve'
bash benchmarks/gpu_power.sh reset
```

Track B runs **after** the server is down (both need the GPU exclusively):

```bash
bash benchmarks/run_offload_sweep.sh      # all-cuda + 4 placements, in-process
```

---

## Track A — guidellm SLO matrix (live server)

`serve_hybrid.sh` + `run_matrix.sh` + `matrix_runner.py`

guidellm drives the server's OpenAI endpoint at controlled offered load
while `matrix_runner.py` scrapes vLLM's Prometheus `/metrics` in parallel.
Per cell it emits json/csv/html.

### guidellm profiles (`PROFILES=` env, default `synchronous,concurrent,sweep`)

guidellm supports eight: `synchronous, concurrent, throughput, constant,
poisson, sweep, async, replay`. The default triangle covers latency floor
+ concurrency tail + the SLO knee. Add `constant,poisson,throughput` for
sustained-load and realistic-burst coverage:

```bash
PROFILES=synchronous,concurrent,sweep,constant,poisson,throughput \
  bash benchmarks/run_matrix.sh
```

| profile | load model | isolates |
|---|---|---|
| synchronous | one at a time | pure latency floor (TTFT, ITL) |
| concurrent | keep N in flight (closed loop) | throughput + ITL tail at fixed N |
| throughput | fire as fast as possible | peak aggregate tok/s |
| constant | fixed req/s sustained | behavior under steady load |
| poisson | Poisson arrivals | realistic bursty traffic |
| sweep | interpolate rate | the throughput/latency curve + knee |

### Thermal pacing

The wrapper runs one (context, profile) cell at a time so cooldowns land
where they should: `COOLDOWN_CTX=120` s between context lengths,
`COOLDOWN_PROFILE=15` s between profiles (both env-configurable). Pair
with `gpu_power.sh cap 575`.

### Knobs

`MAX_MODEL_LEN` (131072) — load-bearing capacity knob. Lower for a strict
8k head-to-head; the model supports 262144 natively. `PLACEMENT`
(subset:16:8). `CONTEXTS`, `RATES`, `OUTPUT_TOKENS`, `MAX_SECONDS`.

---

## Track B — offload sweep (in-process, no server)

`run_offload_sweep.sh` + `offload_benchmark.py` + `offload_summarize.py`

### The offload design space (what Track B maps)

Two knobs, everything reduces to them:

| knob | trades | lever |
|---|---|---|
| **K** — B70-active layers | one dispatch/token per active layer (~91 µs fixed) | `subset:K:..` |
| **offload/layer** | ~1.6 MiB VRAM freed per expert → KV cache | `subset:..:C`, offloaded = 256−C |

Product K × offload/layer = total freed VRAM. Ratio = the
latency/capacity tradeoff. Capacity-matched, **concentrate wins**: same
≈6.7 GiB freed, `split:128` (32 active) → 158 tok/s vs `subset:16:8`
(16 active) → 186 tok/s.

### Architecturally fixed (not knobs — walls)

1. FP8 layers 32–39 cannot offload (bank spans only the 32 NVFP4 layers).
2. Shared expert stays on CUDA.
3. One resident set for the whole bank → every active layer offloads the
   same expert IDs.
4. One physical B70, one SYCL queue → dispatches serialize (this is what
   caps batched throughput).
5. No P2P/RDMA/NCCL across vendors — 5090 and B70 meet only in pinned
   host DRAM over PCIe (~91 µs dispatch floor).
6. CUDA graph capture forbids host syncs / data-dependent branching →
   every active layer always dispatches, even on all-CUDA tokens.
7. GDN attention state caps `max_num_seqs` (~83 all-CUDA).
8. Breakable CUDA graphs are dead for this model (GDN garbage under
   breakable replay, even stock vLLM) — Tier 3 is the only graph path.

---

## Metric glossary

Every guidellm metric is a full distribution (mean/median/std_dev/min/max/
**percentiles**/pdf), broken out by successful/incomplete/errored/total.

| metric | meaning |
|---|---|
| `time_to_first_token_ms` | TTFT |
| `time_to_first_output_token_ms` | first *non-reasoning* token (matters for `<think>`) |
| `inter_token_latency_ms` | ITL — decode smoothness |
| `output_tokens_per_second` | decode throughput |
| `prompt_tokens_per_second` | prefill throughput |
| `request_latency` | end-to-end per request |
| `request_concurrency`, `requests_per_second` | offered vs achieved load |
| `request_totals` | successful/incomplete/errored — when a cell starts refusing |
| scheduler `*_delay_avg` | harness overhead, separable from server cost |
| server `/metrics` scrape | KV utilization, queue depth, prefill vs decode time |

Track B additionally reports (from Shooting Brake telemetry via
`collective_rpc`): **KV tokens** (the capacity win), **B70 route share**,
**dispatch service µs**, **dispatch errors**.

---

## How the two GPUs talk

No NCCL, no oneCCL, no RDMA. The 5090 and B70 rendezvous in pinned host
DRAM: CUDA writes activation/weights with `cudaMemcpyAsync`, signals with
`cuStreamWriteValue32` on a host-mapped flag (graph-captureable), a native
poller thread reads the flag and submits the SYCL `queue->memcpy` H2D to
the B70 over PCIe, the QuixiCore NVFP4 kernel runs, the result copies D2H
to the same pinned pages, the poller raises completion, the CUDA stream
(parked on `cuStreamWaitValue32`) resumes. Physical path: 5090 VRAM →
host DRAM → B70 VRAM → host DRAM → 5090 VRAM, all over PCIe.

---

## Optimization roadmap (after the baseline is down)

The baseline above is the **control**. Everything below is judged against
it. Ordered by value, not by number.

### #1 — Frequency-aware placement  *(new capability; highest-value)*

**What:** route experts to B70 by *measured activation hotness* instead
of by ID range. Hot experts stay on CUDA; only the cold tail offloads.
**Why it matters:** current `split`/`subset` policies assume uniform
hotness — real traffic concentrates. Same freed VRAM, much smaller B70
route share → B70 kernel stops dominating → latency drops. Could dominate
every static policy.
**How:**
1. Run a profiling pass with `SHOOTING_BRAKE_B70_STATS=1` to collect
   per-expert route frequencies (the device-side counters already exist).
2. Build a frequency-ordered manifest: top-N hottest experts → CUDA, the
   cold tail → B70, respecting the one-resident-set constraint.
3. Implement a `FrequencyPolicy(PlacementPolicy)` and register it in
   `policy_from_name`. The manifest is already versioned + swappable for
   this.
4. Re-run Track B → A/B against `subset:16:8`.
**Category:** thing to check — opens a new region of the design space.

### #5 — Multi-queue / multi-provider B70  *(architectural; targets batched)*

**What:** break the single-SYCL-queue serialization so multiple layers'
dispatches run concurrently on the B70.
**Why it matters:** today hybrid is 44% of baseline at concurrency 16
*entirely* because one queue serializes every layer while the 5090 scales.
This is the only lever on the batched regime.
**How:** needs QuixiCore multi-queue (or N provider instances), careful
per-layer ordering, more B70 device memory. Run Track A's
concurrent/sweep profiles before and after.
**Category:** thing to check — changes the scaling characteristic. Do
this only if the baseline shows batched throughput matters for the
workload, and after #1 (which may move enough routes off B70 that the
serialization hurts less).

### #3 — Merge the three H2D copies  *(pure optimization)*

**What:** the dispatch issues three separate `queue->memcpy` (hidden,
ids, weights). Make the pinned staging contiguous and issue one copy.
**Why it matters:** shaves Level Zero submission overhead off the 91 µs
dispatch floor.
**How:** allocate one pinned buffer `[hidden | ids | weights]`, view each
region, pass offsets. Recompile the provider.
**Category:** optimization — same config, faster. Cheap; interleave anytime.

### #2 — Shared pinned buffers across layers  *(optimization + correctness)*

**What:** the per-layer pinned buffers are sized to
`SHOOTING_BRAKE_B70_MAX_BATCH`; only one layer is ever in flight (the CUDA
stream serializes issue/take), so share one buffer set across layers and
raise the cap so prefill goes through the verified Tier 3 path at full
batch.
**Why it matters:** closes the open prefill-passthrough question
(byte-identical output today, but the code path reads like it should drop
B70 routes) and enables correct long-batch prefill.
**How:** hoist the pinned tensors to module-level singletons, size from
`max_num_batched_tokens`, drop the `x.shape[0] > max_batch` bypass.
**Category:** optimization + removes a known correctness risk.

### #4 — llm-scaler INT4 secondary  *(deferred)*

Qualified fallback kernel behind the provider API; swap in if NVFP4
underperforms for some shape. Not pursued now.

---

## Next steps (priority order)

1. **Run the baseline** — Track A SLO matrix + Track B offload sweep on
   `subset:16:8`. This is the control and the design-space map.
2. **Frequency-aware placement (#1)** — collect route frequencies, build
   the manifest, A/B in Track B. The real research result.
3. **Cheap wins #3 + #2** — merge copies; share buffers. Re-measure moved
   cells.
4. **Multi-queue (#5)** if batched throughput matters — re-run Track A
   concurrent/sweep before and after.

The baseline both gives you the numbers to quote *and* justifies which of
#1/#5 to pursue: it already shows batched is where hybrid loses, which is
the evidence #5 is worth the lift.
