# Multi-socket CPU-MoE optimizations for llama.cpp (branch `multisocket-cpu-moe`)

DeepSeek-V4-Flash (284B MoE, MXFP4 experts) decode on a 4-socket Cascade Lake box +
one RTX 3090: **3.26 -> 25.07 tok/s single-stream** (7.7x), batch-4 serving aggregate
**2.11 -> 42.84 tok/s** (20x), big-prompt prefill **137 -> 284 tok/s**, model load
**150 s -> ~30-60 s**. Six surgical commits, each measured same-session A/B against the
previous one, each byte-identical on a 48-token greedy probe.

## 1. Hardware

| Component | Detail |
|---|---|
| CPU | 4x Intel Xeon Gold 6252N (Cascade Lake) @ 2.30 GHz base / 3.60 GHz max |
| Cores | 24 cores / 48 threads per socket = 96 cores / 192 threads |
| NUMA | 4 nodes, ~189 GiB each (754 GiB total); node distance 10/21 |
| Memory | DDR4-2933, 6 channels per socket; measured interleaved read ~127 GB/s (96t), ~114 GB/s node-local |
| GPU | NVIDIA GeForce RTX 3090, 24576 MiB, PCIe Gen3 x16 (verify under load: idles at Gen1) |
| Notes | No AMX, no AVX512-BF16 (Cascade Lake); CUDA host compiler must be g++-14 |

Model: `DeepSeek-V4-Flash-0731-MXFP4.gguf` (ggml-org; 144.34 GiB, Q8_0 dense on GPU,
MXFP4 experts on CPU via `--n-cpu-moe 43`).

Build:
```
cmake -B build-publish -DGGML_CUDA=ON -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14 \
      -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_OPENMP=OFF -DCMAKE_BUILD_TYPE=Release
```

Bench command pattern (all numbers below):
```
numactl --interleave=all ./build-publish/bin/llama-bench -m <model> -ngl 99 -ncmoe 43 \
  --numa distribute -t 96 -b 4096 -ub 4096 -p <P> -n 256 -r 2 -mmp 0 -o md  [+ envs]
```

## 2. The journey — commits in order

| # | Commit | What | Primary metric (same-session A/B) |
|---|---|---|---|
| 0 | 221f0f635 (master) | baseline | tg256: **3.26** @96t (6.91 @24t — stock *loses* speed above ~24 threads); pp2048 178 |
| 1 | 1b03ae94e | fused MoE FFN kernel (multi-token, pair-indexed) | tg256: 3.26 -> **18.73** @48t (7.17 @96t) |
| 2 | feaddb9c9 | expert-home NUMA sharding (`GGML_EXPERT_SHARD=1`) | tg256@96t: 7.17 -> **19.85**; scaling now ascends to 96t |
| 3 | 42d804074 | parallel fastload (`GGML_FASTLOAD_THREADS`) | spawn->healthy: 119 s -> **17 s**; tg256 19.85 -> 22.14 (THP bonus) |
| 4 | b345e3f7a | expert replication (`GGML_EXPERT_REPLICAS=1`) | tg256: 21.21 -> **23.47** (+10.7%, paired r3) |
| 5 | 15ee10149 | pin-once cores (`GGML_PIN_CORE`, default on) + server persistent threadpools | tg256: 22.93 -> **25.91** (+13.0%, paired r3) |
| 6 | 44919290f | cudaHostRegister expert pinning (`GGML_EXPERT_PIN=1`) | pp4096 (offload): 137.16 -> **283.25** (+106%) |

### Why each one works

1. **Fused MoE kernel.** Stock `mul_mat_id` re-reads a shared quantized activation
   vector through M-state cache lines and one globally-contended chunk counter —
   `perf c2c` shows cross-core HITM storms; that is the whole 96-thread collapse.
   The fused op computes gate/up/swiglu(+clamps)(+down projection) as one kernel,
   gives every (token, expert-slot) pair its own chunk counter, quantizes
   activations per-thread, and publishes shared scratch with `clwb` so readers hit
   shared-state lines. Generalized to 8-token ubatches (pair = token x slot), which
   is what makes concurrent serving and speculative verify batches viable.
   Kill-switch: `GGML_CPU_DISABLE_FUSION=1`.
2. **Expert sharding.** Interleaved expert weights make 75% of reads remote. Expert
   `e` is homed to node `e*n_nodes/n_as` (VMA mbind at load, first-touch fills land
   home), only that node's threads compute it, and a work-conserving steal pass keeps
   early-finishing nodes busy. Also forces plain CPU buffers: `cudaMallocHost` pins
   pages *before* any NUMA policy can apply, silently defeating placement (mbind
   relabels but cannot move GUP-pinned pages) — this was the single nastiest bug of
   the project. UPI traffic dropped 2.9 -> ~0.5 GB/token and thread scaling turned
   positive through 96 threads.
3. **Parallel fastload.** Stock no-mmap load writes 145+ GB single-threaded through
   ~36M page faults. Fastload populates destination pages N-wide with
   `MADV_HUGEPAGE` + `MADV_POPULATE_WRITE` (THP cuts fault count 512x), then fills
   with N-wide `pread`. Default 16 threads is the measured optimum — 64 threads
   evict a cold page cache faster than they fill it (19.5 -> 3.7 GB/s). The THP
   pages also bought ~2 tok/s of decode (TLB).
4. **Replication (k=1).** The group-limited router co-selects correlated experts:
   E[max experts on one node] = 2.8/6 per op no matter how you statically place
   them. One extra copy on the antipodal node + per-op least-loaded assignment
   drops E[max] to 2.2. Costs +147 GB RAM. k=3 (full replication) reaches the 2.0
   floor but ties k=1 — below E[max]~2.2, arrival skew owns the op tail, so k=1 is
   the recommended setting.
5. **Pin-once.** Stock re-applies a node-wide affinity mask by syscall on every
   graph (~86 syscalls/thread/token). Pin each worker to exactly one hwthread once,
   never touch affinity again. The win is the deleted syscalls, not migration
   thrash. In llama-server this *requires* persistent threadpools (included in the
   commit): freshly spawned pool workers inherit the creator's pinned 1-core mask,
   and the stock server rebuilds a disposable pool ~44x/token — measured 85 of 96
   workers piled on cpu0, 0.55 tok/s.
6. **Expert pinning.** `mmap + mbind + populate`, *then* `cudaHostRegister` composes
   NUMA residency with DMA pinning (the reverse order is bug #2 above). Batch
   prefill streams 135 GB of experts over PCIe per big chunk; pinned DMA runs at
   ~11.5 GB/s wire rate vs ~5.5 GB/s single-core pageable staging. Includes a
   retry-without-`cudaHostRegisterReadOnly` fallback (rejected by some
   kernel/driver combos, and the ggml wrapper silently lost the registration).

### Final verified numbers (branch tip, build 1721, fresh runs)

| Metric | Stock | This branch |
|---|---|---|
| tg256 @96t (bench) | 3.26 | **25.07 ± 0.51** |
| pp4096 offload (bench) | ~178 (pinned-host stock) | **283.71 ± 0.65** |
| batch-4 server aggregate (4x128 tok) | 2.11 | **42.84** |
| spawn -> /health OK (no-mmap) | 150 s | **59 s** (shard+replicas+pin; 17 s without replicas) |
| single-stream API decode | ~3 | **23.8-24.4** |

## 3. Config & runtime guide (zero-code wins included)

Serving envs (what `run-optimized.sh` sets):

| Env | Value | Effect |
|---|---|---|
| `GGML_EXPERT_SHARD` | 1 | expert-home sharding + plain CPU buffers + fastload path |
| `GGML_EXPERT_REPLICAS` | 1 | two-home replication (+147 GB RAM, +10-14% decode) |
| `GGML_PIN_CORE` | (default on) | pin-once worker affinity |
| `GGML_EXPERT_PIN` | 1 | cudaHostRegister primaries after placement |
| `GGML_CUDA_REGISTER_HOST` | 1 | required for the register wrapper to act |
| `GGML_OP_OFFLOAD_MIN_BATCH` | 2048 | **zero-code win, upstream env**: sub-2048-token ubatches stay on CPU. Default (32) ships all 135 GB of experts over PCIe for ANY prefill >= 32 tokens: measured TTFT on a ~300-token prompt 29 s -> 7.7 s (pp291 9.9 -> 38.0 tok/s). Crossover with pinned DMA at x16 is ~1.5k tokens. |
| `GGML_FASTLOAD_THREADS` | (default 16) | load fill parallelism; do not raise on cold page cache |
| `GGML_EXPERT_STEAL` | (default on) | work-conserving steal; =0 for A/B |
| `GGML_CPU_DISABLE_FUSION` | unset | =1 reverts to stock mul_mat_id (kill-switch) |

Slot configuration (measured matrix in PARALLEL-MATRIX.md): **config D,
`--parallel 4 --kv-unified -c 131072`** — single-stream tg ~19-20 API (ties
`--parallel 1` within noise), tg@4 ~38-43 aggregate, and unified KV keeps prefix-cache
hits across slot reassignment. Without the multi-token fused kernel this config
collapsed to 3.3 tok/s aggregate; do not use `--parallel >1` with fusion disabled.

Threads: `-t 96 -tb 96` with `--numa distribute` + `numactl --interleave=all`.
`--no-mmap` is required for NUMA placement (mmap pages are file-backed, not
first-touch placeable). `kernel.numa_balancing=0` is recommended (measured harmless
during steady decode, but it fights explicit placement during load/migration).

### MTP speculative decoding (option, not in the default runner)

The DSpark MTP sidecar (`-md DeepseekV4-Flash-20260731-DSpark.gguf -ngld 99`)
measured 23.1 -> 25.3 tok/s warm on the kitchen-sink branch with the multi-token
fused kernel doing verify batches. **It does not fit VRAM at the recommended
serving config** (measured on this branch, b/ub 4096, `--parallel 4`):

- `-c 131072`: draft weight alloc (10386 MiB) OOMs at load.
- `-c 65536` / `-c 32768`: weights fit, but the draft's pp compute buffer
  (2454 MB, driven by `-ub 4096`, context-independent) still OOMs.

To use it: quantize the sidecar to Q8 (~5.5 GB), and/or reduce the draft ubatch /
main context until the 10.4 GB weights + ~2.4 GB compute + KV fit beside the ~10 GB
dense slab in 24 GB. Untuned levers: `--draft-max/--draft-min`, acceptance rate
never profiled.

## 4. What didn't work (measured nulls — do not re-attempt without new evidence)

| Idea | Result | Mechanism |
|---|---|---|
| Two-level NUMA barrier | tied | barrier time is ~90% inherent arrival skew of ragged expert pipelines, not rendezvous-line cost |
| Tournament (log2 pairwise) barrier | tied (22.84 vs 22.93) | same — the 64 us/crossing figure was skew, misattributed |
| Barrier dissolution (per-expert countdown + ready flags) | tied (23.94 vs 24.11) | every saved sync ms reappeared as ready-flag spin in compute buckets |
| k=3 (full) replication | 18.88 -> 18.94 | node balance stops binding below E[max]~2.2; also closes exhaustive minimax assignment (perfect balance worth ~0) |
| Round-robin expert homes | == block homes | router picks correlated experts; E[max]=2.8 either way; expert frequency is flat |
| SMT (192t) | negative | fill-buffer/MLP bound; second hwthread adds nothing |
| Software prefetch in vec_dot | -8.7% @96t (+17.5% @64t) | conditional optimization, reverted |
| Q3_K experts (17% smaller) | 7% slower | K-quant dequant ALU exceeds the stall shadows; MXFP4 is QAT-native AND fastest |
| Persistent threadpool alone | +2.4% | included in this branch only as the pin-once prerequisite, not on its own merit |
| GPU/CPU overlap restructure | not built (~+1 tok/s) | of 16.3 ms GPU/token, ~13 ms is serial-to-unblock-CPU (MLA o-proj, DSA indexer, router) |

Dev tooling used to find all of the above (rdtsc bucket profiler `GGML_CPU_PROF`,
scheduler ledger `GGML_SCHED_PROF`, per-node GPU attribution `GGML_CUDA_PROF`,
routing stats `GGML_EXPERT_STATS`) lives on the `ncmoe-fusion-port` branch, tag
`kitchen-sink-v1` — deliberately not ported here.

## 5. Measurement discipline

Same-session A/B only. This box drifts 1-2 tok/s day to day on identical configs
(measured 24.11 one session, ~22.9 the next); r2-r3 repetitions read ±0.5 tok/s.
Every number above was taken as a paired comparison in one session, clean box before
and after (no llama processes, VRAM < 1 GiB). tg256 reads ~3 tok/s higher than tg64 —
never compare across test lengths. Greedy-output chain identity (48-token probe,
temp 0, fixed seed) was verified byte-for-byte across every commit; the fused kernel
differs from *stock* only in T=1 silu rounding (scalar vs SIMD lane, low bit), which
is why the chain is anchored at commit 1 == the canonical fused reference.
