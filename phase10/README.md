# Shooting Brake — Phase 10 benchmark artifacts

Two tracks. They answer different questions and must not be confused.

**Track A** holds the *architecture* fixed and sweeps *load* — the
serving-grade SLO picture against a live server.
**Track B** holds the *load* fixed and sweeps the *architecture* — the
offload-amount tradeoff curve, in-process and fast.

Both reuse the existing Phase 0 all-CUDA baseline; neither re-runs it.

---

## Track A — guidellm SLO matrix (live server)

`track_a_serve_hybrid.sh` + `track_a_guidellm_matrix.sh`

### What it measures

guidellm drives the server's OpenAI endpoint at controlled offered load
and reports production metrics:

| profile | what it tells you |
|---|---|
| **synchronous** | one request at a time — the pure latency floor (TTFT, inter-token latency, decode tok/s). No queuing. |
| **concurrent** | fixed concurrency (rates 1,2,3,4,5,6) — throughput and ITL *tail* under sustained load. This is where headroom runs out. |
| **sweep** | interpolates rate from baseline to peak — traces the throughput/latency curve and locates the knee where the SLO breaks. |

The matrix sweeps **prompt length (context) × concurrency**. At short
contexts both all-CUDA and hybrid serve. The interesting cells are the
long ones: above ~32k, all-CUDA cannot admit the request at all, and the
hybrid's 4.2× KV cache is the only reason it runs. Compare contexts ≤8192
against the Phase 0 baseline; above that, the absence of a baseline *is*
the capacity result.

### How to run

```bash
# 1. start the hybrid server (131k context, subset:16:8) in its own shell
bash phase10/track_a_serve_hybrid.sh

# 2. wait until ready, then run the matrix
bash phase10/track_a_guidellm_matrix.sh
```

Outputs land in `bench-matrix/hybrid_131k_c6/`, one subdirectory per
context length, each with synchronous/concurrent/sweep results plus a
Prometheus scrape of `vllm:` metrics sampled every 5 s.

### Knobs (env)

`MAX_MODEL_LEN` (131072) — the load-bearing one. Lower for a strict
head-to-head with the 8k baseline; raise toward 262144 (native ceiling)
to push the capacity story harder.

`PLACEMENT` (subset:16:8) — the offload policy; see Track B.

---

## Track B — offload sweep (in-process, no server)

`track_b_offload_sweep.sh` + `track_b_summarize.py`

### What it measures

Each B70-active layer costs **one dispatch per token** — a fixed
overhead that does not shrink with how many routes that layer sends. The
total number of offloaded experts decides how much CUDA VRAM is freed for
KV cache. So two knobs:

- **active layers** — how many of the 32 NVFP4 layers own B70 experts.
  Fewer active layers → fewer dispatches → lower decode latency, for the
  same freed VRAM.
- **experts per active layer** — how many each active layer keeps on
  CUDA. Fewer → more freed VRAM → more KV, but a larger route share hits
  B70.

`subset:<active>:<cuda_per_layer>` concentrates the offload; `split:N`
spreads it across all 32 layers. The sweep runs each policy as a fresh
process and prints the curve: tok/s, ITL, KV capacity, B70 route share,
dispatch service time.

### How to run

```bash
bash phase10/track_b_offload_sweep.sh
```

No server needed — uses the in-process harness, so the GPU is held only
for the duration of each placement run. Results in
`phase10/results/offload/`; `track_b_summarize.py` prints the table and
flags the placement that frees the most VRAM at the least latency cost.

### Knobs

`PLACEMENTS` — space-separated policies. Default curve:
`subset:8:8 subset:16:8 subset:24:64 split:128` (active-layer count
rising 8→16→24→32).

---

## Reading the numbers

| metric | meaning | where it comes from |
|---|---|---|
| **TTFT** | time to first token — prefill cost, felt by the user as responsiveness | guidellm / per-token stream |
| **ITL p50/p99** | inter-token latency — decode smoothness; p99 is the stutter users notice | guidellm / per-token stream |
| **tok/s (single)** | decode throughput, one request — the latency floor inverted | both tracks |
| **tok/s (concurrent)** | aggregate throughput under load — what the box actually delivers | guidellm concurrent |
| **KV tokens** | how much context/concurrency fits — the hybrid's reason to exist | telemetry (`collective_rpc`) |
| **B70 route share** | fraction of routed experts on the B70 — how much work left the 5090 | device-side counters, graph-safe |
| **service µs** | poller's issue+take time per dispatch — lower bound on per-layer cost | native poller counters |

### Why the two GPU vendors can talk at all

No NCCL, no oneCCL, no RDMA — those do not cross vendor boundaries. The
5090 and the B70 rendezvous in **pinned host DRAM**: CUDA writes
activation/weights there with `cudaMemcpyAsync`, signals with
`cuStreamWriteValue32` on a host-mapped flag (graph-captureable), a
native poller thread reads the flag and submits the SYCL `queue->memcpy`
H2D to the B70 over PCIe, the QuixiCore NVFP4 kernel runs, the result
copies D2H back to the same pinned pages, the poller raises the
completion flag, and the CUDA stream (parked on
`cuStreamWaitValue32`) resumes and copies H2D. Physical path: 5090 VRAM
→ host DRAM → B70 VRAM → host DRAM → 5090 VRAM, all over PCIe.

---

## Status (2026-08-05)

Built and syntax-checked. Not yet run — deferred until the 5090 is
clock-locked at a fixed power target so the two tracks are comparable
run-to-run.
