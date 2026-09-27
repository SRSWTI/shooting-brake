# Shooting Brake: profiling-first optimization contract

Approved 2026-09-09. This document defines the measurements and decisions for the next campaign; it is not a report of completed profiling or promised speedups. Experiment outcomes, failed attempts, and promotion decisions remain in [the campaign changelog](new/CHANGELOG.md).

## Required sequence

1. Freeze the current qualified baseline.
2. Profile the distinct serving workloads and reconstruct their critical paths.
3. Find the largest exposed costs, including memory and capacity constraints.
4. Test one targeted change against captured real shapes and inputs.
5. Run actual-model correctness checks.
6. Repeat unprofiled serving comparisons and update the RTX PRO 6000 gap.

Main writes all code and runs experiments. Read-only scouts may locate existing interfaces. Production serving is not replaced by an experiment. Driver, firmware, physical topology, destructive storage changes, or other disruptive changes require discussion first. Speculative decoding and replacing vLLM with the separate CUDA engine remain outside this campaign.

## Baseline and provenance

Starting candidate: `src/phase7/libsb_b70_provider.rangesafe.so`, SHA-256 `e2b22d9d64b3e51bf3c658fefecd76a133069ad22068c99503e6a8994fdafed3`. This is the numerical-overflow repair, not a speed improvement. Use native grouped NVFP4, FP16 result wire, two provider chunks, CUTLASS local MoE, classic native doorbells, and the existing 95/75 B70 expert placement.

Pin model and tokenizer `srswti/axe-superveloce-jota-118b-r15-nvfp4` at revision `357b5c1f87b70cb89f7f66bee6dbbf90eaee279a`. Preserve the launch command, effective environment, serving source and binary digests, runtime versions, GPU identities, topology, cache policy, actual KV allocation, and workload/token identity. Use isolated endpoints and distinct output directories. Never reuse incompatible evidence.

The source checkpoint has 48 layers: 12 full-attention layers with 48 query heads, 36 sliding-attention layers with 72 query heads and window 512, eight KV heads, head dimension 128, 47 routed MoE layers, and one dense MLP layer. Measure both attention geometries. NVFP4 weight storage does not imply NVFP4 KV.

## Workloads to measure

| Path | Required observation |
|---|---|
| Cold prefill | New prompt, prefix caching disabled, actual input length, first-token latency |
| Cached-prefix continuation | Known repeated prefix plus new suffix, cache hit/reused-token evidence, suffix processing cost |
| Single-request decode | Context-dependent token latency, per-step execution, full versus sliding attention |
| Concurrent decode | Completed throughput, individual latency distribution, actual scheduled batch size |
| Mixed prefill/decode | Decode stalls while another prompt is admitted and processed |
| Capacity pressure | Admission, KV occupancy, preemption/recomputation, transient peaks, failures |
| Startup/restoration | Weight loading, initialization, compilation, graph capture; persistent restore only if a real implementation is selected |

Start from 1K, 8K, 32K, and 127K cold-context targets, with output lengths and actual token counts recorded. Extend concurrency only within observed memory/admission limits. Use an explicit stopping/draining policy for bounded load runs. An incomplete request is retained in evidence, not silently treated as a successful latency sample. Do not deliberately drive the workstation into an unbounded OOM/retry loop.

## Per-step accounting

| Stage | Measurements |
|---|---|
| Request processing | Arrival, tokenization, queue delay, admission, active/waiting requests, scheduled tokens |
| Attention preparation | Normalization, projections, positional operations, KV writes |
| Attention | Full/sliding type, actual query and KV lengths, kernel time |
| Routing/local experts | Router/top-k, ownership mapping, quantization, local/shared expert kernels |
| Outbound transfer | Dtype conversion, staging, copied bytes, copy-engine duration, contention |
| Each B70 | Signal detection, issue/take host work, input DMA, grouped/split expert computation, output DMA |
| Return/aggregation | Host copy, completion publication, CUDA wait, return DMA, cast and sum |
| Output | Final normalization, logits projection, sampling, detokenization, client streaming |

Attach step/layer/device/chunk identity and actual row counts wherever observable. Keep the two B70s separate. Report missing visibility explicitly. Allocation and residency snapshots are not peak-memory measurements.

### Timing rules

- Separate host submission time, device execution time, queueing, and exposed dependency waits.
- Do not add overlapping device durations to infer token latency. Reconstruct the dependency chain and identify which branch finishes last.
- Correlate host/CUDA/Intel clock domains before merging timestamps. Preserve uncertainty and unexplained remainder.
- The poller service timer starts after signal observation; it is not the complete cross-vendor handoff. Completion-publication time is also outside that timer.
- The existing `Worker.execute_model` host wrapper is not automatically a GPU completion timer.
- Intel kernel/total timing fields are unavailable when provider profiling is disabled, even though their current representation is zero. Report them as unmeasured, not zero-duration work.
- Provider profiling adds event timestamping and marker submissions. Measure instrumented versus uninstrumented overhead, and use unprofiled runs for final serving claims.
- The existing eager seam tracer skips graph capture and must not be treated as complete replay-path coverage. Verify actual event/dispatch coverage.
- Use short broad timelines first, detailed kernel/transfer studies separately. Do not turn on every expensive profiler at once and call the resulting latency production performance.

## Candidate optimizations

These are hypotheses to select from after the critical-path measurements, not committed gains.

| Candidate | Evidence needed | Main constraint |
|---|---|---|
| Remove redundant casts, quantization, or temporaries | Repeated operations occupy exposed NVIDIA time | Numerical semantics and buffer lifetime |
| Improve local/shared CUDA expert execution | CUDA expert branch finishes last | Kernel speedup may remain hidden by remote work |
| Improve full/sliding attention | Attention contributes material exposed latency | Exact geometry, precision, KV layout, graph support |
| Eliminate extra host result staging | Host copy is measurable on the return critical path | Registration, coherence, ownership, completion ordering |
| Improve transfer scheduling | Copies serialize or contend | Extra buffering and memory pressure |
| Change expert placement | Device imbalance or local weights constrain useful KV | Moving the bottleneck between compute, transfers, and capacity |
| Tune chunking and admission | Prefill stalls decode or wastes resources | Throughput versus interactive latency |
| Reduce signal-detection/submission delay | Handoff delay is materially exposed | Previous command-streamer integration failures |
| Improve PCIe topology | Concurrent transfer measurements establish the bottleneck | Physical changes and interference with other devices |

OneDNN already exists and was retested: isolated grouped-pipeline improvements did not establish a meaningful serving win. Do not count it as a newly implemented optimization or promote it from microbenchmark results alone.

## Architectural options for later selection

### Persistent prefixes and tiered KV

Compare lookup + read + restore time with recomputing the same prefix. Verify exact prefix matching, checkpoint/KV-format identity, byte-correct restoration, and readiness ordering. Measure effects on other requests, host RAM, transfer bandwidth, and eviction. Retained KV, active attention KV, and the model's supported context length are different quantities.

The inspected Tutti checkout contains relevant storage/index/transfer components but an incomplete real vLLM gather/scatter path and unresolved persistent identity/restore integration. TensorCast is a storage substrate, not a qualified in-tree vLLM adapter. Neither is a current speedup.

### GPUDirect Storage

GDS can avoid CPU bounce buffering on a supported storage-to-GPU path; it also supports compatibility paths that stage through host memory. Validate the actual selected path, device/filesystem support, restoration cost, and contention with the B70 PCIe traffic. It is not extra VRAM and does not itself make SSD-resident active KV efficient. Reference: [NVIDIA GDS overview](https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html).

### Other workstation resources

Consider independent inference requests, embeddings/reranking, cache retention, separate prefill/decode, or additional expert execution. Each requires measured topology, transfer cost, runtime support, and memory availability. Distinguish single-request acceleration from aggregate throughput. Do not introduce another per-token dependency solely because a device has spare memory.

## Correctness and promotion

Keep original checkpoint bytes, routing semantics, and required outputs. Validate captured real inputs, large intermediate ranges, empty/unowned routes, relevant prefill/decode shapes, repeated graph replay, scratch reuse, and both devices. Do not exclude a failing token or relax a gate solely to claim success. A baseline failure in a new reference test requires investigation of the reference/precision contract, not an immediate claim that the backend is broken.

For a selected optimization: isolated correctness -> isolated measurement -> actual-model numerical/provider checks -> repeated unprofiled serving comparisons. Keep failures and sample counts. Promote only after evidence supports the intended workload benefit without unacceptable correctness, memory, or tail-latency regressions.

## RTX PRO 6000 comparisons

Use `benchmarks/results/rtx_pro_6000_r15_slo` as the recorded target. Preserve its missing checkpoint-hash and differing GuideLLM-version caveats. Do not label a historical comparison a fresh matched hardware run.

Maintain two tracks when hardware access permits:

1. Controlled comparison: same checkpoint/tokenizer, prompt/output lengths, cache semantics, output requirements, client behavior, and workload arrival pattern.
2. Hardware-tuned comparison: each machine's qualified best settings under the same latency/error requirements.

Report TTFT, ITL, completed throughput, successful/incomplete/errored counts, actual concurrency, and memory/admission capacity. Keep cold, cached, and mixed-load results separate. Maximum configured context/concurrency is not demonstrated sustained capacity. Evaluate throughput subject to latency/error limits, not unconstrained peak tokens per second.
