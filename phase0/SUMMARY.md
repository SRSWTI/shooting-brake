# Shooting Brake Phase 0 — Freeze Contract Package Summary

**Date:** 2026-08-04
**Status:** ✅ COMPLETE
**Gate:** All Phase-0 items frozen. Phases 0, 1, and 2 are complete; Phase 3 is next.

## Deliverables

| Item | File | Status |
|---|---|---|
| Version fingerprints | `freeze.yaml` | ✅ Frozen |
| Source checkpoint identity | `freeze.yaml` | ✅ Frozen |
| CUDA/B70 artifact SHA-256 fingerprints | `freeze.yaml`, `capability_manifest.yaml` | Frozen |
| Correctness prompt set (12) | `correctness_prompts.jsonl` | ✅ Frozen |
| Performance workload | `workload.yaml` | ✅ Frozen |
| Provider protocol v1 | `provider_protocol.yaml` | ✅ Delivered |
| Capability manifest v1 | `capability_manifest.yaml` | ✅ Delivered |
| CUDA baseline (graph mode) | `baseline_cuda_graph.json` | ✅ Recorded |
| Benchmark script | `benchmark_cuda.py` | ✅ Working |
| Env config | `env.sh` | ✅ Saved |

## Frozen Versions

- **vLLM:** 0.26.0 (PyPI)
- **PyTorch CUDA:** 2.11.0+cu130
- **PyTorch XPU:** 2.13.0+xpu
- **CUDA:** 13.0 (nvcc 13.3)
- **NVIDIA Driver:** 595.84
- **QuixiCore-XPU:** commit `53a9f87`
- **oneAPI:** DPC++/C++ 2026.1.1
- **Intel Runtime:** libze-intel-gpu 26.22.38646.6
- **FlashInfer:** 0.6.14 (CUTLASS NVFP4 MoE backend)
- **Source checkpoint:** `Qwen/Qwen3.6-35B-A3B` snapshot `995ad96e`
- **CUDA artifact:** `unsloth/Qwen3.6-35B-A3B-NVFP4` (24.67 GiB)

## CUDA Baseline Results (RTX 5090, graph mode)

| Workload | Metric | Value |
|---|---|---:|
| Decode (batch=1, 512 tok) | Throughput | **242.63 tok/s** |
| Decode (batch=1, 512 tok) | Mean ITL | **4.1 ms** |
| Batched (8 req, 256 tok) | Aggregate throughput | **1095.77 tok/s** |
| Correctness (12 prompts) | Pass rate | **11/12** |

The 1 correctness failure (`factual_002` Garcia Marquez) is a truncation artifact — the model's `<think>` reasoning block exceeds `max_tokens=32` before reaching the answer.

## Model Memory Layout (32 GiB VRAM)

| Component | Size |
|---|---:|
| Model weights (NVFP4+FP8 mixed) | 23.27 GiB |
| KV cache (FP8) | 1.91 GiB |
| CUDA graph pool | 0.92 GiB |
| Peak activation | 1.89 GiB |
| Non-torch overhead | 0.23 GiB |
| **Total** | **28.22 GiB** (of 28.22 GiB budget at 0.90 util) |

## vLLM Kernel Selection

- **NVFP4 MoE experts** (layers 0-31): `FLASHINFER_CUTLASS` backend
- **FP8 MoE experts** (layers 32-39): `TRITON` backend
- **Attention**: `FLASHINFER` backend
- **KV cache**: FP8 per-tensor
- **CUDA graphs**: PIECEWISE (19 graphs) + FULL decode (11 graphs)

## Comparison: Colibri Reference vs vLLM Production

| Path | Decode tok/s | Notes |
|---|---:|---|
| Colibri CUDA full VRAM (INT4 GS64) | 18.35 | Reference evidence |
| Colibri CUDA 12GB budget (INT4 GS64) | 15.39 | Reference evidence |
| **vLLM CUDA NVFP4 (this baseline)** | **242.63** | Production baseline |

The 13× improvement reflects vLLM's optimized serving stack, CUTLASS NVFP4 kernels, CUDA graphs, and FlashInfer attention — not a model quality difference.

## Known Limitations

1. **TTFT not captured** — vLLM 0.26 sync `generate()` doesn't populate `RequestOutput.metrics` timing fields. Will be fixed with server-based benchmarking in a later phase.
2. **Eager baseline scope decision** — Phase 0 accepted the production-relevant graph-mode baseline. Eager equivalence is intentionally part of the Phase-4 all-CUDA adapter-parity gate rather than an open Phase-0 item.
3. **MoE config warning** — vLLM uses default fused MoE config for `E=256,N=512` on RTX 5090. A tuned config could improve batched throughput further.

## Gate Check

- [x] Fail-closed protocol/model/provider compatibility fields and actions frozen in the Phase-0 schemas → **PASS**. Phase 1 additionally enforces explicit B70 identity, exact bank size/header/layout, API generation/sequence checks, and invalid layer/ID rejection. Phase 2 makes process-ring generation, request/completion sequencing, canonical route payloads, stale identity rejection, and bounded slot ownership executable in the protocol-v2 wire ABI.
- [x] All version fingerprints recorded → **PASS**
- [x] Source checkpoint identity frozen → **PASS**
- [x] CUDA baseline reproducible → **PASS** (consistent across 3 runs)
- [x] Provider protocol delivered → **PASS** (frozen Phase-0 logical schema v1; the executable Phase-2 wire ABI is the clean protocol-v2 cutover required by exact fixed-layout semantics)

**Phase 0 remains complete. Phases 1 and 2 are also complete; proceed to Phase 3 (the independent provider mathematics matrix).**
