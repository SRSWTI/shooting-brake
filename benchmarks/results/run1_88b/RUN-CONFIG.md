# Run 1 — 88B hybrid, single B70

Environment and configuration for the first end-to-end inference run of
`srswti/axe-superveloce-88b` with routed experts computed on an Intel Arc Pro B70
in int4. Recorded before the run so the result is reproducible rather than
merely observed. Full package set in `env-freeze.txt` (311 packages).

## Environment changes made to unblock the launch

Both were version mismatches inside the venv, upstream of any Shooting Brake
code. Both are recorded because they change what the run executes.

| change | from | to | why |
|---|---|---|---|
| `openai` | 2.6.1 | **3.0.0** | vLLM 0.27.1 imports `NamespaceTool` from `openai.types.responses` when constructing `LLM()`; 2.6.1 does not have it. Failed before model load. |
| `flashinfer-cubin` | 0.6.3 | **removed** | Mismatched against `flashinfer-python` 0.6.16.post3, and no cubin release matches that version. Removing it eliminates the mismatch; FlashInfer JITs for sm120 instead of loading prebuilt cubins. |

Rejected alternatives, deliberately:
- Monkeypatching `vllm.entrypoints.llm.log_non_default_args` to a no-op. Works
  for offline `LLM()` but hides the incompatibility until `vllm serve`, which is
  where the concurrency and long-context numbers come from.
- `FLASHINFER_DISABLE_VERSION_CHECK=1`. Would load stale 0.6.3 binaries against
  a 0.6.16 runtime — wrong numbers instead of an error.

**Consequence to carry forward:** removing `flashinfer-cubin` moves CUDA kernel
delivery from prebuilt cubins to JIT. Expect a slower first token (the reference
RTX PRO 6000 run spent ~37 s autotuning and ~12 s capturing CUDA graphs) and
treat startup latency as not comparable to a cubin-installed environment.

## Pre-existing environment conflict, not introduced here

`uv pip check` reports eight conflicts, **all of them `sglang` against
everything else**: it pins `openai==2.6.1`, `torch==2.9.1`,
`transformers==4.57.1`, `xgrammar==0.1.27`, `llguidance<0.8.0`,
`quack-kernels==0.2.4`, `torchaudio==2.9.1`, `torchcodec==0.8.0`.

This venv is a mixed sglang + vLLM environment and the two are mutually
incompatible. `sglang` is not used by any Shooting Brake path. The `openai` pin
above is why 2.6.1 was installed. None of these conflicts affect vLLM.

If sglang is ever needed, it belongs in a separate venv.

## Run-critical versions

```
vllm==0.27.1                 # same as the RTX PRO 6000 reference run
torch==2.13.0                # +cu130
transformers==5.15.0
flashinfer-python==0.6.16.post3
flashinfer-cubin             # absent, by decision above
openai==3.0.0
safetensors==0.8.0
triton==3.7.1
numpy==2.2.6
```

Not pinned in `pyproject.toml` on purpose: `torch` and `vllm`, because a plain
`pip install torch` replaces the CUDA-130 build with a generic one. Install
those from the correct index first, then `pip install -e .`.

## Placement under test

| tier | contents | size |
|---|---|---|
| CUDA (5090, 31.8 GiB) | dense weights + routed experts `0..53` in NVFP4 | 5.97 + 12.81 = 18.78 GiB |
| B70-A | routed experts `54..179` in int4, from `src/phase1/expert_bank_int4.bin` | 27.41 GiB of 31.9 |
| B70-B | unused in this run | — |
| host DRAM | nothing in the datapath | — |

KV headroom ~6.5 GiB at a measured 12.93 KiB/token, so roughly 543K tokens.

Bank: 29,429,862,400 bytes, `SBINT401` v2, resident IDs exactly `54..179`,
`data_offset` 4096, `expert_stride_bytes` 4,866,048. Validated bit-exact against
the original safetensors shards on 27 samples (layers 0/23/47 x experts
54/116/179 x gate/up/down).

## Expected result, stated before measuring

**80–95 tok/s single-stream.**

Derived from measurement, not hope. `ProviderInt4Finish` swept active route
counts 0..8 on real hardware and found a flat non-kernel dispatch cost:

| k routes | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| kernel µs | 8.1 | 30.8 | 32.2 | 33.6 | 32.6 | 37.1 | 56.6 | 72.3 | 79.9 |
| wall − kernel µs | 110 | 110 | 109 | 109 | 113 | 108 | 108 | 108 | 108 |

One card takes all 8 routes, so ~188 µs per dispatch x 48 layers ≈ 9.0 ms/token
of dispatch alone. The ~108 µs fixed floor contributes ~5.2 ms of that.

**A result far above 100 tok/s should be treated as evidence of a bug, not
success.** The historical failure in this code path passed global expert IDs into
a surgically compacted tensor, silently dropping every remote route: it ran at
full all-CUDA speed, emitted identical output tokens, and produced ~0.49
nats/token worse logprobs. Fast because it was wrong.

## Correctness gate

Sync (`SHOOTING_BRAKE_B70_ASYNC=0`, graph unset) versus graph/poller candidate,
same weights, same formats, same placement — only the dispatch mechanism
differs. Tolerance must be justified rather than bitwise: `routed_experts.py`
documents that identical paths can differ by ~0.11 nats because B70 and CUDA
partials plus atomic reductions are not order-stable.

Not valid as exactness oracles, and deliberately not used as such:
- Comparing placements that move an expert across the CUDA/B70 boundary. CUDA
  holds NVFP4 and the B70 holds GPTQ int4 — two independent quantizations of the
  same bf16 source, measured 14.55% relative L2 apart, cosine 0.9895. Outputs
  *must* differ.
- An all-CUDA reference. 49.53 GiB does not fit on a 31.8 GiB card; that is the
  reason this project exists.
- The RTX PRO 6000 run. Quality baseline, not bitwise ground truth, same
  quantization argument.

The exact aggregation oracle — one captured layer's real `x`, route IDs and
weights, expected output rebuilt from the compact NVFP4 CUDA partial plus a CPU
int4-bank dequant partial for remote routes — is defined but deferred until
after this run.

## Already verified before this run

| what | result |
|---|---|
| provider chain, bank → upload → int4 kernel → copyout vs CPU oracle | max peak-relative **1.148e-06**, bound 5e-05 |
| `-1` skip sentinel | all-`-1` dispatch returns 3,072 exact zeros |
| additive route decomposition | left(k) + right(8−k) == one 8-route dispatch, worst **1.506e-07** |
| exact / wrong-length / subset resident-map rejection | all PASS, both sets printed in the message |
| NVFP4 `SBEXP001` golden gate | PASS, incl. empty `source_expert_ids` |
| peak RSS above bare SYCL baseline | **+19.5 MiB** — the 27.41 GiB mmap does not accumulate |

Note the scope of the additive-decomposition result: it validates the kernel and
provider boundary on **one** card. Device selection, two provider lifecycles,
concurrent issue/take, separate staging, CUDA-side reduction and shared-uplink
behaviour are all still unexercised.
