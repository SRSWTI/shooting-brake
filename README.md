# Shooting Brake

![Shooting Brake](/Users/rohit/Downloads/shooting.webp)

A shooting brake was never for everyone — it's the rare machine that refuses to sacrifice speed for capacity, built in limited numbers for people who wanted both. Shooting Brake makes the same bet on silicon, and does it first: the industry's first heterogeneous NVIDIA-Intel inference build. An RTX 5090 leads (32GB GDDR7, ~1.7TB/s bandwidth, full PCIe5.0 x16), handling what NVIDIA does best — fast, compute-bound prefill. Behind it, an Intel Arc Pro B70 brings 32GB of VRAM at roughly a quarter the cost per card, expanding the usable memory pool rather than chasing raw speed.
---

## The problem

Modern MoE models are enormous. A 35-billion-parameter MoE like Qwen3.6-35B-A3B needs ~23 GB just for expert weights — and that's the *small* one. The models people actually want to serve (300B+ parameters) need 150+ GB. An RTX 5090 has 32 GB. You'd need five or six of them at $4,000 each to hold one model.

But here's the thing: in a MoE model, a **small set of "hot" experts handles most tokens**, and the rest sit idle most of the time. You're paying $4,000 per 32 GB of NVIDIA VRAM to store experts that rarely fire. That's like buying a fleet of sports cars to use as storage units.

**Shooting Brake asks: what if the hot experts live on the fast NVIDIA GPU, and the cold experts live on cheap Intel VRAM?**

The Intel Arc Pro B70 gives you 32 GB for $900 — **$28 per GB versus $125 per GB for a 5090.** Same capacity class, one-quarter the price. The trade-off: it's a different vendor, different driver stack, and there's no direct peer-to-peer link between NVIDIA and Intel GPUs. Shooting Brake is the software fabric that bridges that gap.

---

## How it works

The architecture has a clear division of labor. The NVIDIA GPU owns everything latency-critical; the Intel GPU is a specialized accelerator for one job — computing expert feed-forward networks.

```
  ┌─────────────────────────────────────────────────────────┐
  │                    RTX 5090 (32 GB)                      │
  │                                                         │
  │   vLLM scheduler ──► attention ──► router/top-k ──┐     │
  │         │                          │               │     │
  │    KV cache                     hot experts      route   │
  │    (842K tokens)               (CUDA, fast)     decision │
  │                                                         │
  └─────────────────────────────────────┬───────────────────┘
                                        │  cold expert routes
                                        │  (hidden states + indices)
                                        ▼
                              ┌──────────────────┐
                              │  Pinned host DRAM │
                              │  (shared memory)  │
                              └────────┬─────────┘
                                       │  flag signal
                                       ▼
  ┌────────────────────────────────────┴────────────────────┐
  │                  Intel Arc Pro B70 (32 GB)                │
  │                                                          │
  │   Native C++ poller ──► NVFP4 expert kernels            │
  │   (spins on flag)      (QuixiCore-XPU, custom)              │
  │                         │                                │
  │                    weighted partial                      │
  └────────────────────────┬─────────────────────────────────┘
                           │  result via host DRAM
                           ▼
  ┌──────────────────────────────────────────────────────────┐
  │  RTX 5090: join partials ──► residual ──► next layer     │
  └──────────────────────────────────────────────────────────┘
```

**Per decode step, for each layer that has cold experts on the B70:**

1. The 5090 computes the router logits and selects the top-8 experts.
2. Routes destined for hot experts execute locally on the 5090 (fast path).
3. Routes destined for cold experts get written to a pinned host-memory buffer.
4. A `cuStreamWriteValue32` flag signals the B70's native poller thread.
5. The B70 reads the inputs, runs its NVFP4 expert kernels, writes back weighted partials.
6. A completion flag signals the 5090, which reads the partials and joins them into the residual stream.

No NCCL. No oneCCL. No RDMA. No cross-vendor P2P (it doesn't exist). Just pinned host DRAM and CUDA stream-value flags — the lowest-common-denominator communication that works across any vendor boundary.

### Why a native C++ poller?

A Python poll loop costs ~55 µs per wakeup under the GIL and starves the engine thread. Shooting Brake moved the host-side watcher into a **native C++ thread** (`phase7/b70_capi.cpp`) that spins on `_mm_pause()` and signals the SYCL queue with sub-microsecond latency. This is what makes the dispatch path compatible with CUDA graph capture — zero Python on the decode path.

### Why mixed precision?

The model runs in two precision regimes, aligned to hardware boundaries:

- **Layers 0–31: NVFP4** (4-bit float) — these experts live on the B70, which has qualified QuixiCore-XPU NVFP4 kernels.
- **Layers 32–39: FP8** (8-bit float) — these stay on the 5090, forced to CUDA.

This isn't a workaround — it's a design choice. NVFP4 halves the weight footprint, letting the B70 hold twice as many experts per gigabyte. The precision split is configured in the placement manifest and is swappable.

---

## What's been built

### Complete and measured

| milestone | what it does |
|---|---|
| **B70 NVFP4 provider** (`phase1/`, `phase7/`) | Persistent C++ process that loads 8,192 NVFP4 experts into B70 VRAM and executes them via QuixiCore-XPU kernels. Pipelined SYCL dispatch: issue() enqueues all work, take() waits once. |
| **vLLM adapter** (`phase4/`) | Out-of-tree plugin (`shooting_brake_vllm`) that hooks into vLLM's MoE layer. Routes experts to CUDA or B70 based on a versioned placement manifest. Transparent to vLLM — the scheduler doesn't know experts left the GPU. |
| **Tier 3 graph-compatible dispatch** | The entire B70 round-trip is native CUDA stream operations, captured by `torch.cuda.graph()`. The 5090's decode graph includes D2H copies, flag signals, and H2D result joins — no graph breaks. |
| **VRAM surgery** | After weight loading, Shooting Brake frees the 5090 VRAM occupied by offloaded expert weights. This runs as a `process_weights_after_loading` hook, before vLLM sizes the KV cache. Result: **6.4 GB freed → KV cache 4× larger** (211K → 842K tokens). |
| **Layer-subset placement** | Instead of spreading B70 ownership across all 32 capable layers (more dispatches), the `subset:K:C` policy concentrates it into the last K capable layers. Same capacity, fewer dispatches per token. |
| **Benchmark harness** (`benchmarks/`) | Two tracks: guidellm SLO matrix (live server, sweeps context × load) and in-process offload sweep (sweeps architecture). Plus GPU power management for repeatable measurement. |

### Benchmark results

Measured on Qwen3.6-35B-A3B-NVFP4, single-stream decode, 512 output tokens:

| metric | all-CUDA (baseline) | hybrid (subset:16:8) |
|---|---|---|
| decode throughput | 248 tok/s | **170–186 tok/s** |
| ITL p50 | 4.0 ms | 5.3–5.9 ms |
| KV cache capacity | 211,696 tokens | **842,038 tokens (4×)** |
| B70 route share | 0% | 97.3% |
| dispatch errors | — | **0** |

At **65,536-token context** (where things get interesting):

| metric | value |
|---|---|
| TTFT (prefill) | 2.9 s |
| prefill throughput | 22,800 tok/s |
| decode throughput | 170 tok/s |
| ITL p50 | 5.8 ms |

Decode speed stays flat across context lengths — the B70 dispatch overhead is a fixed per-layer cost that doesn't grow with sequence length. That's the key property: **once the prompt is prefilled, decode is decode, whether the prompt was 1K or 65K tokens.**

The 32% single-stream speed gap versus all-CUDA is the expected cost of offloading. The 4× KV capacity gain is the payoff. At long context and high concurrency — where all-CUDA runs out of KV blocks and starts refusing requests — the hybrid keeps serving.

### What was hard (and how we solved it)

These are the engineering decisions that made the system work. Details are in the linked docs; the headlines:

- **Cross-vendor communication has no P2P.** NVIDIA and Intel GPUs share no address space. The only path is through host DRAM. We built a pinned-memory ring with `cuStreamWriteValue32`/`WaitValue32` flags as the signal mechanism — the lowest-friction cross-vendor handshake available. ([`docs/expert-fabric.md`](docs/expert-fabric.md))

- **CUDA graphs can't cross vendor boundaries.** We made the B70 dispatch entirely native CUDA stream ops (D2H copies, flag writes) that get captured inside the 5090's own graph. The host watcher is a native C++ thread, so no Python runs during graph replay. ([`docs/architecture.md`](docs/architecture.md))

- **vLLM sizes KV cache from profiling peak.** If VRAM surgery ran during profiling, the freed weights distorted the KV budget. We moved surgery to the `process_weights_after_loading` hook — before profiling, so vLLM sees the real peak and allocates correctly. This alone gave 4.6× more KV cache.

- **Breakable CUDA graphs produce garbage with GDN attention.** Qwen3.6 uses Gated DeltaNet (GDN) hybrid attention (30 GDN + 10 full-attention layers). vLLM's breakable graph mode corrupts GDN state on replay — a known upstream bug (vLLM #51008). We route around this: Tier 3 dispatch works with normal (full) CUDA graphs, which are compatible with GDN.

---

## The hardware reality

### The PCIe bottleneck we found

The current dev machine (Intel Core Ultra 9 285K, Z890 motherboard) has a critical topology limitation:

```
  5090  ──► CPU direct PCIe 5.0 x16  =  64 GB/s    ✓ full speed
  B70   ──► PCH (chipset) PCIe 3.0 x4 =  3.9 GB/s  ✗ 16× narrower
```

The consumer CPU has only 20 PCIe 5.0 lanes. The 5090 takes 16, the NVMe takes 4, and the B70 gets forced onto the chipset's PCIe 3.0 x4 port — a 3.9 GB/s straw. This is the dominant cost in dispatch overhead at batched sizes.

At single-stream (M=1), payloads are tiny (~32 KB/layer), so PCIe **latency** matters more than bandwidth — the gap is modest. At concurrency (M=16+), payloads grow and the 3.9 GB/s ceiling becomes a hard wall.

### What would help

| change | impact | why |
|---|---|---|
| **HEDT platform** (Threadripper Pro / Xeon W, 128+ PCIe 5.0 lanes) | **the single biggest lever** | Moves the B70 from PCH (3.9 GB/s) to direct CPU PCIe 5.0 x16 (64 GB/s) — 16× more bandwidth. Also enables multiple B70s, each on direct lanes. |
| Multi-queue B70 dispatch (software) | **high** | Breaks the single-SYCL-queue serialization that limits batched throughput. Biggest software win. |
| Frequency-aware placement (software) | **high** | Routes experts to B70 by measured hotness — hot experts stay on the 5090, cold ones go to B70. Fewer dispatches for the same capacity. |
| Merged H2D copies (software) | **medium** | Contiguous pinned staging buffer, one memcpy per dispatch. Lowers the 91 µs dispatch floor. |
| NUMA pinning | **N/A on current machine** | Single socket, single NUMA node — no cross-socket penalty. Would matter on dual-socket EPYC. |
| Faster DRAM | **negligible** | DRAM (~80 GB/s) is 20× overprovisioned vs the B70's PCIe path (3.9 GB/s). Never the bottleneck. |

---

## The economics

For expert weight storage — the bulk of a large MoE — the metric that matters is **$ per GB of VRAM**:

```
  B70 (32 GB)     $900     →  $28 / GB
  5080 (16 GB)    $1,250   →  $78 / GB
  5090 (32 GB)    $4,000   →  $125 / GB
```

The B70 is **4.5× cheaper per GB** than the 5090. For cold expert storage (experts that rarely fire), this is the right hardware — you're paying for capacity, not speed.

### Why you still need NVIDIA

The NVIDIA GPU owns the parts the B70 **cannot do**:

- **vLLM itself** — the scheduler, continuous batching, paged attention, CUDA graph capture. There is no vLLM for Intel GPU.
- **Attention** — FlashAttention and the GDN/flash-linear-attention kernels are CUDA-only. The B70 has no attention implementation.
- **KV cache** — random-access memory pattern needs the 5090's 1,792 GB/s bandwidth.
- **Prefill** — compute-bound; tensor cores dominate.
- **Hot experts** — the small set of experts that fire most often should stay on the fastest available silicon.

The B70 does exactly one thing: **dense FFN/expert computation via NVFP4 kernels.** That happens to be where most of a MoE model's *parameters* live.

---

## The vision: 300B models on a budget

The architecture extends naturally to larger models. A 300B-parameter MoE (13B active) in NVFP4 is ~150 GB of expert weights. The configurations:

| config | NVIDIA side | B70 side | total VRAM | cost |
|---|---|---|---|---|
| **1× 5090 + 4× B70** | 32 GB (hot experts + attention + KV) | 128 GB (cold bank) | 160 GB | **$7,600** |
| 1× 5090 + 5× B70 | 32 GB | 160 GB | 192 GB | $8,500 |
| 5× 5090 (homogeneous) | 160 GB | — | 160 GB | **$20,000** |

One 5090 plus four B70s delivers the same 160 GB at **2.6× lower cost** than five 5090s. The 5090 owns the hot path (attention, hot experts, KV cache); the B70s hold the cold expert bank. Shooting Brake's placement manifest routes tokens accordingly.

This requires a HEDT platform (Threadripper Pro / Xeon W) to give each B70 a direct PCIe 5.0 connection — the consumer platform can't feed even one B70 at full speed today.

---

## Roadmap

### Software optimizations (current hardware)

These squeeze maximum value from the existing 3.9 GB/s B70 path:

1. **Frequency-aware placement** — collect route frequencies during serving, build a hot/cold manifest, keep hot experts on the 5090. Highest-value optimization.
2. **Multi-queue B70 dispatch** — break single-SYCL-queue serialization. Biggest throughput lift at concurrency.
3. **Merged H2D copies** — one contiguous pinned staging buffer, one memcpy. Lowers dispatch floor.
4. **Shared pinned buffers** — module-level singletons, raise max batch for prefill through Tier 3.

### Hardware upgrades

5. **HEDT platform** — Threadripper Pro or Xeon W with 128+ PCIe 5.0 lanes. Moves B70 from 3.9 GB/s to 64 GB/s. The single highest-impact change.
6. **Multiple B70s** — scale the cold expert bank. Each B70 adds 32 GB at $900. Requires the HEDT platform for direct lanes.

### Multi-GPU NVIDIA side

7. **Tensor-parallel support** — the architecture's principle is agnostic to the CUDA GPU configuration (1× 5090 or 3× 5080), but the current implementation is single-CUDA-device. Supporting TP=2/3 across multiple NVIDIA cards needs:
   - The adapter to work with vLLM's tensor-parallel execution model
   - Coordinated B70 access from multiple CUDA workers
   - Surgery/placement logic that accounts for distributed weights
   
   This is real engineering (estimated 2–3 weeks), not a config flip. The benefit: **3× RTX 5080 (48 GB, $3,750)** could replace one 5090 (32 GB, $4,000) for less money and more VRAM — but consumer cards lack NVLink, so all-reduce runs over PCIe, adding per-layer overhead. The trade works best when VRAM per NVIDIA card is the binding constraint.

### Production hardening

8. **Phase 9** — heartbeat monitoring, bounded timeouts, provider restart, generation bumping, batched failed-route recovery.
9. **Full SLO benchmark** — complete the guidellm matrix across all context lengths (1K–127K) and load profiles (synchronous, concurrent, sweep).

---

## Quick start

### Prerequisites

```bash
# One-time: build the B70 provider (needs oneAPI + icpx)
source /opt/intel/oneapi/setvars.sh --force
cd phase7 && make
cd ../phase4 && uv pip install --python .venv/bin/python --no-deps -e .
```

### Serve the hybrid model

```bash
# Shell 1: launch the hybrid server (5090 + B70, subset:16:8 placement)
bash benchmarks/serve_hybrid.sh

# Verify it's up
curl -s http://127.0.0.1:8000/health
```

### Run benchmarks

```bash
# Track A: SLO matrix against the live server (context × load profiles)
bash benchmarks/run_matrix.sh

# Track B: in-process offload sweep (no server needed, sweeps architecture)
bash benchmarks/run_offload_sweep.sh

# Compare all-CUDA vs hybrid
python benchmarks/compare.py
```

Benchmark details, metrics, and the offload design space are in [`benchmarks/README.md`](benchmarks/README.md).

---

## Key environment variables

| variable | default | purpose |
|---|---|---|
| `VLLM_PLUGINS` | `shooting_brake_vllm` | Load the Shooting Brake adapter |
| `SHOOTING_BRAKE_HYBRID` | `0` | Enable B70 expert offload |
| `SHOOTING_BRAKE_PLACEMENT` | `split:128` | Expert placement policy (`split:N`, `subset:K:C`, `all-cuda`) |
| `SHOOTING_BRAKE_B70_DEVICE` | `1` | Intel GPU device index |
| `SHOOTING_BRAKE_VRAM_SURGERY` | `0` | Free 5090 VRAM occupied by offloaded weights |
| `SHOOTING_BRAKE_B70_GRAPH` | `0` | Enable Tier 3 graph-compatible dispatch |
| `SHOOTING_BRAKE_MODEL` | — | HuggingFace model ID |

---

## Repository structure

| directory | what's inside |
|---|---|
| `phase1/` | Native B70 NVFP4 provider core — expert bank, SYCL kernels, control process |
| `phase4/` | vLLM out-of-tree adapter — routing, placement, hybrid forward, VRAM surgery |
| `phase7/` | Native C++ poller and C ABI — `B70Poller` class, `sb_b70_*` functions |
| `benchmarks/` | SLO matrix, offload sweep, comparison, GPU power management |
| `QuixiCore-XPU/` | Primary MIT-licensed B70 NVFP4 MoE kernel source |
| `colibri-variants/` | Proven CUDA+B70 reference (oracle/comparator) |
| `docs/` | Detailed architecture, contracts, evidence, and progress docs |

---

## Documentation

For deep dives into specific areas:

| document | scope |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Production boundaries, data flow, ownership model |
| [`docs/progress.md`](docs/progress.md) | Phase-by-phase completion evidence and current status |
| [`docs/correctness.md`](docs/correctness.md) | Route semantics, numerical agreement, failure recovery |
| [`docs/hardware.md`](docs/hardware.md) | GPU topology, host-staged transport, hardware qualification |
| [`docs/expert-fabric.md`](docs/expert-fabric.md) | Provider API, pinned ring protocol, weighted-partial contract |
| [`docs/placement.md`](docs/placement.md) | Static ownership, layer-subset policy, future frequency-aware routing |
| [`docs/benchmarking.md`](docs/benchmarking.md) | Measurement rules and methodology |
| [`docs/memory.md`](docs/memory.md) | CUDA, B70, pinned-host, and KV memory budgets |
| [`benchmarks/README.md`](benchmarks/README.md) | Benchmark tracks, metrics, offload design space, optimization roadmap |
| [`docs/research.md`](docs/research.md) | Source provenance, prior art, build-vs-borrow decisions |

---

## Model

**Qwen3.6-35B-A3B-NVFP4** — 40 layers, 256 experts per layer (top-8 routing), hidden size 2048. 30 GDN (Gated DeltaNet) + 10 full-attention layers. Native context: 262,144 tokens.

Expert bank: `phase1/expert_bank.bin` — 13.5 GB, 8,192 NVFP4 experts.

---

## One-line definition

> Shooting Brake keeps scheduling, attention, and KV cache in upstream vLLM on one RTX 5090, while Intel Arc Pro B70 GPUs execute offloaded routed experts through pinned-memory dispatch — serving models that wouldn't fit on the NVIDIA card alone, at one-quarter the cost per gigabyte.
